"""Local text-to-3D pipeline — zero API keys, all runs on MM1.

Pipeline:  text -> Flux.1 (image) -> TripoSR (3D mesh) -> GLB export

- Flux.1 schnell on MPS: ~40s for 768x768 product photo
- TripoSR on MPS: ~25s for single-image to 3D mesh
- Total: ~70s from text to textured GLB

No external API keys required.  All models cached locally.
"""
import asyncio
import base64
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

RENDERS_DIR = Path(os.environ.get("RENDERS_DIR", Path.home() / "Projects/atlas-swarm/renders"))
TRIPOSR_DIR = Path(os.environ.get("TRIPOSR_DIR", Path.home() / "Projects/TripoSR"))
PYTHON_BIN = Path(os.environ.get("SWARM_PYTHON", Path.home() / "Projects/atlas-swarm/.venv/bin/python"))


def _image_to_3d(image_path: Path, product_name: str) -> dict:
    """Run TripoSR on an image to produce a GLB.  Returns result dict."""
    import trimesh

    out_dir = Path(f"/tmp/triposr_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_py = TRIPOSR_DIR / "run.py"
    if not run_py.exists():
        return {"success": False, "error": f"TripoSR not found at {TRIPOSR_DIR}"}

    log.info("[T2-3D] Running TripoSR on %s", image_path.name)
    t0 = time.monotonic()

    proc = subprocess.run(
        [
            str(PYTHON_BIN), str(run_py),
            str(image_path),
            "--device", "mps",
            "--output-dir", str(out_dir),
        ],
        cwd=str(TRIPOSR_DIR),
        capture_output=True,
        text=True,
        timeout=300,
    )

    if proc.returncode != 0:
        return {"success": False, "error": f"TripoSR failed: {proc.stderr[-500:]}"}

    obj_path = out_dir / "0" / "mesh.obj"
    if not obj_path.exists():
        return {"success": False, "error": "TripoSR did not produce mesh.obj"}

    # Convert OBJ -> GLB
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in product_name)[:40]
    ts = int(time.time())
    glb_path = RENDERS_DIR / f"{safe}_{ts}.glb"

    mesh = trimesh.load(str(obj_path))
    mesh.export(str(glb_path))

    elapsed = time.monotonic() - t0
    size_kb = round(glb_path.stat().st_size / 1024, 1)

    log.info("[T2-3D] GLB saved: %s (%sKB, %.1fs)", glb_path.name, size_kb, elapsed)

    return {
        "success": True,
        "provider": "triposr-local",
        "glb_path": str(glb_path),
        "glb_filename": glb_path.name,
        "glb_size_kb": size_kb,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "generation_time_s": round(elapsed, 1),
    }


async def text_to_3d_local(
    description: str,
    product_name: str = "",
    filename_stem: str = "",
) -> dict:
    """Full pipeline: text -> Flux.1 image -> TripoSR GLB.

    Runs locally on MM1 with MPS.  No API keys.
    ``filename_stem`` (optional) gives a deterministic output name — used by
    the autonomous IdeaFrog driver to dedupe by opportunity id.
    """
    from .visualizer import generate_product_render

    name = filename_stem or product_name or description.replace(" ", "_").replace("/", "_")[:40]

    # 1. Flux.1 photo
    log.info("[T2-3D] Step 1/2: Flux.1 image for '%s'", description)
    photo = await generate_product_render({
        "product_name": name,
        "product_description": description,
        "material": "natural product appearance",
    })

    if photo.get("model") == "placeholder":
        return {"success": False, "error": "Flux.1 not available, placeholder used"}

    image_path = Path(photo["image_path"])
    if not image_path.exists():
        return {"success": False, "error": f"Image not found: {image_path}"}

    # 2. TripoSR 3D (run in thread — it's subprocess, blocks asyncio otherwise)
    log.info("[T2-3D] Step 2/2: TripoSR mesh from %s", image_path.name)
    loop = asyncio.get_event_loop()
    mesh_result = await loop.run_in_executor(None, _image_to_3d, image_path, name)

    if not mesh_result.get("success"):
        return mesh_result

    return {
        "success": True,
        "provider": "local-flux+triposr",
        "image_path": photo["image_path"],
        "image_filename": Path(photo["image_path"]).name,
        "glb_path": mesh_result["glb_path"],
        "glb_filename": mesh_result["glb_filename"],
        "glb_size_kb": mesh_result["glb_size_kb"],
        "flux_time_s": photo.get("generation_time_s"),
        "triposr_time_s": mesh_result.get("generation_time_s"),
        "total_time_s": round((photo.get("generation_time_s") or 0) + (mesh_result.get("generation_time_s") or 0), 1),
        "vertices": mesh_result.get("vertices"),
        "faces": mesh_result.get("faces"),
    }


def is_available() -> bool:
    """Check if the local text-to-3D pipeline is usable."""
    return (TRIPOSR_DIR / "run.py").exists() and PYTHON_BIN.exists()
