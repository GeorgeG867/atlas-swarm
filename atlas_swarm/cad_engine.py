"""CAD Engine — generates, validates, and exports 3D printable geometry.

Routes product specs to the parametric library or to sandboxed LLM-generated
CadQuery code.  Every export is validated for printability before writing.
"""
import json
import logging
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import cadquery as cq

from . import cad_library

log = logging.getLogger(__name__)

RENDERS_DIR = Path(os.environ.get(
    "RENDERS_DIR", Path.home() / "Projects/atlas-swarm/renders",
))
RENDERS_DIR.mkdir(parents=True, exist_ok=True)

def _print_bed() -> tuple[int, int, int]:
    """Read printer dims from config — no hardcoded constants."""
    from . import config
    p = config.printer_constraints()
    return (p.get("max_x_mm", 256), p.get("max_y_mm", 256), p.get("max_z_mm", 256))


# ═══════════════════════════════════════════════════════════════════════
# Geometry validation
# ═══════════════════════════════════════════════════════════════════════

def validate_geometry(solid: cq.Workplane) -> dict:
    """Check a CadQuery solid for 3D-print viability.

    Returns ``{valid, issues, metrics}``.
    """
    issues: list[str] = []

    try:
        shape = solid.val()
        bb = shape.BoundingBox()

        dims = {
            "x": round(bb.xlen, 1),
            "y": round(bb.ylen, 1),
            "z": round(bb.zlen, 1),
        }

        # Print-volume check — limits come from YAML config
        bed = _print_bed()
        for axis, val, limit in [("x", dims["x"], bed[0]), ("y", dims["y"], bed[1]), ("z", dims["z"], bed[2])]:
            if val > limit:
                issues.append(f"{axis}={val}mm exceeds {limit}mm print volume")

        # Degenerate check
        for axis, val in dims.items():
            if val < 0.5:
                issues.append(f"{axis}={val}mm too small — likely degenerate")

        # Volume — different CadQuery versions expose this differently
        try:
            volume_mm3 = shape.Volume()
        except AttributeError:
            try:
                volume_mm3 = float(shape.wrapped.Volume())
            except Exception:
                volume_mm3 = dims["x"] * dims["y"] * dims["z"] * 0.3  # rough estimate
        if volume_mm3 < 10:
            issues.append(f"Volume {volume_mm3:.1f}mm³ is suspiciously small")

        volume_cm3 = volume_mm3 / 1000.0
        weight_pla_g = volume_cm3 * 1.24  # PLA density

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "metrics": {
                "dimensions_mm": dims,
                "bounding_box": {
                    "min": {"x": round(bb.xmin, 1), "y": round(bb.ymin, 1), "z": round(bb.zmin, 1)},
                    "max": {"x": round(bb.xmax, 1), "y": round(bb.ymax, 1), "z": round(bb.zmax, 1)},
                },
                "volume_mm3": round(volume_mm3, 1),
                "volume_cm3": round(volume_cm3, 2),
                "estimated_weight_pla_g": round(weight_pla_g, 1),
                "fits_print_bed": all(
                    dims[a] <= bed[i] for i, a in enumerate(["x", "y", "z"])
                ),
            },
        }
    except Exception as exc:
        return {
            "valid": False,
            "issues": [f"Validation error: {exc}"],
            "metrics": {},
        }


# ═══════════════════════════════════════════════════════════════════════
# Library-based generation
# ═══════════════════════════════════════════════════════════════════════

def generate_from_library(product_type: str, params: Optional[dict] = None) -> dict:
    """Build a product from the parametric library.

    ``params`` override the function defaults; unknown keys are silently dropped.
    """
    catalog = cad_library.PRODUCTS.get(product_type)
    if not catalog:
        avail = list(cad_library.PRODUCTS.keys())
        return {"success": False, "error": f"Unknown product: {product_type}. Available: {avail}"}

    func = catalog["func"]
    params = params or {}

    # Drop keys that the function doesn't accept
    import inspect
    sig = inspect.signature(func)
    clean = {k: v for k, v in params.items() if k in sig.parameters}

    try:
        log.info("[CAD] Generating %s  params=%s", product_type, clean)
        t0 = time.monotonic()

        solid = func(**clean)
        validation = validate_geometry(solid)

        ts = int(time.time())
        safe = product_type.replace(" ", "_")
        stl = RENDERS_DIR / f"{safe}_{ts}.stl"
        step = RENDERS_DIR / f"{safe}_{ts}.step"

        cq.exporters.export(solid, str(stl))
        cq.exporters.export(solid, str(step))

        elapsed = time.monotonic() - t0

        return {
            "success": True,
            "product_type": product_type,
            "product_name": catalog["name"],
            "stl_path": str(stl),
            "stl_filename": stl.name,
            "stl_size_kb": round(stl.stat().st_size / 1024, 1),
            "step_path": str(step),
            "step_filename": step.name,
            "step_size_kb": round(step.stat().st_size / 1024, 1),
            "validation": validation,
            "params_used": clean,
            "generation_time_s": round(elapsed, 2),
            "print_info": {
                "material": catalog["default_material"],
                "infill_percent": catalog["default_infill"],
                "estimated_time_min": catalog["estimated_time_min"],
                "estimated_material_g": catalog["estimated_material_g"],
            },
        }

    except Exception as exc:
        log.error("[CAD] Generation failed: %s", exc)
        return {"success": False, "error": str(exc), "traceback": traceback.format_exc()}


# ═══════════════════════════════════════════════════════════════════════
# LLM-code generation (sandboxed exec)
# ═══════════════════════════════════════════════════════════════════════

def _make_safe_builtins() -> dict:
    """All standard builtins EXCEPT import, eval, exec, open, and I/O."""
    import builtins
    blocked = {
        "__import__", "eval", "exec", "compile",
        "open", "input", "breakpoint",
        "exit", "quit", "help", "credits", "license",
        "globals", "locals", "vars", "dir", "delattr", "setattr", "getattr",
        "memoryview", "__build_class__",
    }
    safe = {k: v for k, v in vars(builtins).items() if k not in blocked}
    safe["print"] = lambda *a, **k: None  # suppress output
    safe["__import__"] = None  # explicitly block
    return safe


_SAFE_BUILTINS = _make_safe_builtins()


def _strip_imports(code: str) -> str:
    """Remove import lines and fix indentation — cq and math are pre-injected."""
    import textwrap
    lines = code.split("\n")
    cleaned = [ln for ln in lines if not ln.strip().startswith(("import ", "from "))]
    return textwrap.dedent("\n".join(cleaned))


def _strip_fillets(code: str) -> str:
    """Remove .fillet() and .chamfer() calls — last resort when they crash OCCT."""
    import re
    code = re.sub(r'\.fillet\([^)]*\)', '', code)
    code = re.sub(r'\.chamfer\([^)]*\)', '', code)
    code = re.sub(r'try:\s*\n\s*\n\s*except[^\n]*\n\s*pass', '', code)
    return code


def _auto_scale_to_print_bed(solid: cq.Workplane) -> cq.Workplane:
    """Scale the solid uniformly so its longest dimension fits the target_max_mm.

    Reads target_max_mm from templates/cad_prompts.yaml (no hardcoded constants).
    """
    from . import config
    target_max = float(config.printer_constraints().get("target_max_mm", 200))
    try:
        bb = solid.val().BoundingBox()
        max_dim = max(bb.xlen, bb.ylen, bb.zlen)
        if max_dim <= target_max:
            return solid

        scale_factor = target_max / max_dim
        log.info("[CAD] Auto-scaling: max_dim=%.0fmm -> %.0fmm (factor %.3f)",
                 max_dim, target_max, scale_factor)
        scaled = solid.val().scale(scale_factor)
        return cq.Workplane("XY").add(scaled)
    except Exception as e:
        log.warning("[CAD] Auto-scale failed: %s", e)
        return solid


def generate_from_code(cadquery_code: str, product_name: str = "custom") -> dict:
    """Execute LLM-written CadQuery code in a restricted namespace.

    Auto-detects result variable. Retries without fillets on OCCT errors.
    ``import`` statements are stripped — cq and math are pre-loaded.
    """
    clean_code = _strip_imports(cadquery_code)

    last_err = None
    # Try with fillets first, retry without if they crash OCCT
    for attempt, code_variant in enumerate([clean_code, _strip_fillets(clean_code)]):
        ns = {"cq": cq, "math": math, "__builtins__": _SAFE_BUILTINS}
        try:
            if attempt == 0:
                log.info("[CAD] Executing custom code for '%s'", product_name)
            else:
                log.info("[CAD] Retrying '%s' without fillets/chamfers", product_name)
            t0 = time.monotonic()

            exec(code_variant, ns)  # noqa: S102

            solid = ns.get("result")

            # Auto-detect: if `result` not set, find the last Workplane in namespace
            if solid is None:
                candidates = [
                    (k, v) for k, v in ns.items()
                    if isinstance(v, cq.Workplane) and not k.startswith("_")
                ]
                if candidates:
                    solid = candidates[-1][1]
                    log.info("[CAD] Auto-detected result from variable '%s'", candidates[-1][0])

            if solid is None:
                var_names = [k for k in ns if not k.startswith("_") and k not in ("cq", "math")]
                last_err = f"No CadQuery solid found. Variables: {var_names[:10]}"
                continue

            if not isinstance(solid, cq.Workplane):
                last_err = f"result must be cq.Workplane, got {type(solid).__name__}"
                continue

            # Auto-scale if oversized (LLM often uses real-world dimensions)
            solid = _auto_scale_to_print_bed(solid)

            validation = validate_geometry(solid)

            ts = int(time.time())
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in product_name)[:40]
            stl = RENDERS_DIR / f"{safe}_{ts}.stl"
            step = RENDERS_DIR / f"{safe}_{ts}.step"

            cq.exporters.export(solid, str(stl))
            cq.exporters.export(solid, str(step))

            elapsed = time.monotonic() - t0

            return {
                "success": True,
                "product_name": product_name,
                "stl_path": str(stl),
                "stl_filename": stl.name,
                "stl_size_kb": round(stl.stat().st_size / 1024, 1),
                "step_path": str(step),
                "step_filename": step.name,
                "validation": validation,
                "generation_time_s": round(elapsed, 2),
                "code_lines": cadquery_code.count("\n") + 1,
            }

        except SyntaxError as exc:
            last_err = f"Syntax error: {exc}"
        except Exception as exc:
            last_err = str(exc)
            # Fillet/chamfer errors are worth retrying without them
            if attempt == 0 and any(kw in str(exc).lower() for kw in ("fillet", "chamfer", "edge", "wire")):
                log.warning("[CAD] Fillet-related error, will retry: %s", exc)
                continue
            if attempt > 0:
                break

    return {"success": False, "error": last_err or "Unknown error", "traceback": traceback.format_exc()}


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def match_product_type(text: str) -> Optional[str]:
    """Match free text to a library product type via whole-word keyword matching."""
    import re
    words = set(re.findall(r'\b\w+\b', text.lower()))
    best, best_score = None, 0
    for key, entry in cad_library.PRODUCTS.items():
        score = sum(1 for kw in entry["keywords"] if kw in words)
        if score > best_score:
            best, best_score = key, score
    return best if best_score > 0 else None


def list_available_products() -> list[dict]:
    return [
        {
            "product_type": k,
            "name": e["name"],
            "description": e["description"],
            "default_material": e["default_material"],
            "estimated_time_min": e["estimated_time_min"],
            "params": cad_library.get_function_params(k),
        }
        for k, e in cad_library.PRODUCTS.items()
    ]


def list_renders() -> list[dict]:
    out = []
    for ext in ("*.stl", "*.step"):
        for f in sorted(RENDERS_DIR.glob(ext), key=lambda p: p.stat().st_mtime, reverse=True):
            out.append({
                "name": f.name,
                "type": f.suffix.lstrip("."),
                "size_kb": round(f.stat().st_size / 1024, 1),
                "path": str(f),
            })
    return out[:50]
