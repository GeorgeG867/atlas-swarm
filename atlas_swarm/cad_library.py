"""Parametric CAD library for Atlas Swarm products.

Each function returns a CadQuery Workplane ready for STL/STEP export.
All dimensions in millimeters. Defaults based on top-selling Amazon/Etsy products.

Design references:
- Phone stand: Lamicall S1 (80mm wide, 68deg), UGREEN (76mm, adjustable), Amazon Basics (80mm, 65deg)
- iPhone 16 Pro Max: 77.6mm wide, 163mm tall, 8.25mm thick (add 5mm for case = ~83mm)
- Samsung S24 Ultra: 79mm wide, 162.3mm tall
- Print bed: Bambu Lab P1S = 256x256x256mm
"""
import math
from typing import Optional

import cadquery as cq

# ── Print constraints ──
MAX_PRINT = (256, 256, 256)  # mm (Bambu P1S/X1C)
MIN_WALL = 1.2  # mm for FDM


def phone_stand(
    width: float = 85.0,
    base_depth: float = 75.0,
    base_thick: float = 5.0,
    back_height: float = 100.0,
    back_angle: float = 72.0,
    back_thick: float = 3.5,
    lip_height: float = 18.0,
    lip_thick: float = 4.0,
    cable_w: float = 14.0,
    cable_h: float = 5.0,
    fillet_r: float = 1.5,
) -> cq.Workplane:
    """Minimal angle phone stand — one-piece FDM print.

    Prints with back rest flat on bed (no supports).
    72deg provides comfortable desk viewing angle.
    85mm width fits phones in protective cases.
    14mm cable slot accommodates USB-C with case.
    """
    rad = math.radians(back_angle)
    tilt_deg = 90.0 - back_angle

    # ── Base plate ──
    base = cq.Workplane("XY").box(width, base_depth, base_thick)

    # ── Front lip (holds the phone) ──
    lip = (
        cq.Workplane("XY")
        .workplane(offset=base_thick / 2)
        .center(0, base_depth / 2 - lip_thick / 2)
        .box(width, lip_thick, lip_height)
    )
    body = base.union(lip)

    # ── Angled back rest ──
    rest_w = width - 6  # narrower for elegance
    rest = cq.Workplane("XY").box(rest_w, back_thick, back_height)
    # Shift so bottom face center is at the origin
    rest = rest.translate((0, 0, back_height / 2))
    # Tilt backward (top toward -Y)
    rest = rest.rotate((0, 0, 0), (1, 0, 0), -tilt_deg)
    # Position: bottom at back edge of base, overlapping 1mm for solid boolean
    rest = rest.translate((0, -base_depth / 2 + back_thick + 1, base_thick / 2 - 1))
    body = body.union(rest)

    # ── Cable pass-through ──
    slot = (
        cq.Workplane("XY")
        .center(0, base_depth / 2 - lip_thick / 2)
        .box(cable_w, lip_thick + 4, cable_h)
    )
    body = body.cut(slot)

    # ── Fillets ──
    try:
        body = body.edges().fillet(fillet_r)
    except Exception:
        try:
            body = body.edges("|Z").fillet(min(fillet_r, 1.0))
        except Exception:
            pass

    return body


def tablet_holder(
    width: float = 130.0,
    base_depth: float = 95.0,
    base_thick: float = 6.0,
    back_height: float = 160.0,
    back_angle: float = 75.0,
    back_thick: float = 4.0,
    lip_height: float = 22.0,
    lip_thick: float = 5.0,
    cable_w: float = 18.0,
    cable_h: float = 6.0,
    fillet_r: float = 2.0,
) -> cq.Workplane:
    """Tablet holder — wider and taller than phone stand.

    Fits iPad Air (178.5mm wide landscape) and similar tablets.
    Wider base and thicker walls for stability with heavier devices.
    """
    return phone_stand(
        width=width,
        base_depth=base_depth,
        base_thick=base_thick,
        back_height=back_height,
        back_angle=back_angle,
        back_thick=back_thick,
        lip_height=lip_height,
        lip_thick=lip_thick,
        cable_w=cable_w,
        cable_h=cable_h,
        fillet_r=fillet_r,
    )


def cable_clip(
    inner_d: float = 6.0,
    wall: float = 2.5,
    length: float = 15.0,
    mount_w: float = 20.0,
    mount_d: float = 3.0,
    screw_d: float = 3.5,
    gap_w: float = 3.0,
) -> cq.Workplane:
    """Snap-fit cable management clip with screw/adhesive mount.

    Print flat, no supports. Flexible gap allows cable insertion.
    inner_d=6mm fits USB-C/Lightning cables with jacket.
    """
    outer_r = inner_d / 2 + wall

    # ── Clip ring ──
    ring = (
        cq.Workplane("XY")
        .circle(outer_r)
        .circle(inner_d / 2)
        .extrude(length)
    )

    # ── Snap-fit gap (top of ring) ──
    gap = (
        cq.Workplane("XY")
        .center(0, outer_r)
        .box(gap_w, wall * 2 + 2, length + 2)
    )
    ring = ring.cut(gap)

    # ── Mounting plate (flat bottom) ──
    plate = (
        cq.Workplane("XY")
        .center(0, -(outer_r + mount_d / 2 - 0.5))
        .box(mount_w, mount_d, length)
    )
    body = ring.union(plate)

    # ── Screw hole through mounting plate ──
    hole = (
        cq.Workplane("XZ")
        .center(0, length / 2)
        .transformed(offset=(0, -(outer_r + mount_d / 2), 0))
        .circle(screw_d / 2)
        .extrude(mount_d + 2)
    )
    body = body.cut(hole)

    return body


def desk_organizer(
    length: float = 180.0,
    width: float = 80.0,
    height: float = 45.0,
    wall: float = 2.5,
    dividers: int = 2,
    pen_slots: int = 3,
    pen_d: float = 12.0,
    fillet_r: float = 2.0,
) -> cq.Workplane:
    """Multi-compartment desk organizer with pen holder section.

    Open-top tray with internal dividers. Rightmost compartment has
    cylindrical pen holes that go partway through (blind holes).
    Print upright, no supports.
    """
    # ── Outer shell ──
    outer = cq.Workplane("XY").box(length, width, height)
    inner = (
        cq.Workplane("XY")
        .workplane(offset=wall)
        .box(length - 2 * wall, width - 2 * wall, height)
    )
    body = outer.cut(inner)

    # ── Internal dividers ──
    n_compartments = dividers + 1
    compartment_w = (length - 2 * wall) / n_compartments
    for i in range(1, dividers + 1):
        x = -length / 2 + wall + compartment_w * i
        div = (
            cq.Workplane("XY")
            .workplane(offset=wall)
            .center(x, 0)
            .box(wall, width - 2 * wall, height - wall)
        )
        body = body.union(div)

    # ── Pen holes in rightmost compartment (blind holes from top) ──
    right_center_x = length / 2 - wall - compartment_w / 2
    spacing = compartment_w / (pen_slots + 1)
    for i in range(1, pen_slots + 1):
        px = right_center_x - compartment_w / 2 + spacing * i + wall
        hole = (
            cq.Workplane("XY")
            .workplane(offset=wall + 5)  # leave 5mm floor in pen section
            .center(px, 0)
            .circle(pen_d / 2)
            .extrude(height)
        )
        body = body.cut(hole)

    # ── Fillet outer edges ──
    try:
        body = body.edges("|Z").fillet(fillet_r)
    except Exception:
        pass

    return body


def headphone_hook(
    arm_length: float = 80.0,
    arm_width: float = 30.0,
    arm_thick: float = 5.0,
    clamp_depth: float = 25.0,
    clamp_gap: float = 30.0,
    wall: float = 4.0,
    fillet_r: float = 3.0,
) -> cq.Workplane:
    """Under-desk headphone hook with clamp mount.

    Clamps onto desk edge (adjustable gap via clamp_gap).
    Arm extends outward to hold headphones.
    No screws needed — friction fit + rubber pad.
    """
    # ── Top clamp jaw ──
    top_jaw = cq.Workplane("XY").box(arm_width, clamp_depth, wall)
    top_jaw = top_jaw.translate((0, 0, clamp_gap / 2 + wall / 2))

    # ── Bottom clamp jaw ──
    bot_jaw = cq.Workplane("XY").box(arm_width, clamp_depth, wall)
    bot_jaw = bot_jaw.translate((0, 0, -(clamp_gap / 2 + wall / 2)))

    # ── Clamp back (vertical connector) ──
    back_h = clamp_gap + 2 * wall
    back = cq.Workplane("XY").box(arm_width, wall, back_h)
    back = back.translate((0, -clamp_depth / 2 + wall / 2, 0))

    body = top_jaw.union(bot_jaw).union(back)

    # ── Arm extending from top jaw ──
    arm = cq.Workplane("XY").box(arm_width, arm_length, arm_thick)
    arm = arm.translate((0, clamp_depth / 2 + arm_length / 2 - wall, clamp_gap / 2 + wall + arm_thick / 2))

    # ── Hook curl at end of arm ──
    hook_r = 15
    hook = (
        cq.Workplane("XZ")
        .center(0, clamp_gap / 2 + wall + arm_thick)
        .transformed(offset=(0, clamp_depth / 2 + arm_length - wall, 0))
        .circle(hook_r + wall)
        .circle(hook_r)
        .extrude(arm_width)
        .translate((-arm_width / 2, 0, 0))
    )
    # Cut away top half of hook ring to make a J-shape
    hook_cut = cq.Workplane("XY").box(arm_width + 4, hook_r * 3, hook_r + wall + 2)
    hook_cut = hook_cut.translate((
        0,
        clamp_depth / 2 + arm_length - wall,
        clamp_gap / 2 + wall + arm_thick + hook_r + wall / 2,
    ))

    body = body.union(arm).union(hook).cut(hook_cut)

    try:
        body = body.edges().fillet(min(fillet_r, wall / 2))
    except Exception:
        pass

    return body


# ═══════════════════════════════════════════════════════════════════════
# Product Catalog — metadata from YAML, geometry functions from this module
# ═══════════════════════════════════════════════════════════════════════

# Geometry functions keyed by product type name
_GEOMETRY_FUNCS = {
    "phone_stand": phone_stand,
    "tablet_holder": tablet_holder,
    "cable_clip": cable_clip,
    "desk_organizer": desk_organizer,
    "headphone_hook": headphone_hook,
}


def _load_products() -> dict:
    """Load product catalog: YAML metadata + Python geometry functions."""
    import os
    from pathlib import Path
    import yaml

    yaml_path = Path(os.environ.get(
        "SWARM_TEMPLATES",
        Path(__file__).resolve().parent.parent / "templates",
    )) / "products.yaml"

    catalog = {}
    if yaml_path.exists():
        raw = yaml.safe_load(yaml_path.read_text()) or {}
        for key, meta in raw.items():
            func = _GEOMETRY_FUNCS.get(key)
            if func is None:
                continue
            catalog[key] = {
                "func": func,
                "name": meta.get("name", key),
                "description": meta.get("description", ""),
                "default_material": meta.get("default_material", "PLA"),
                "default_infill": meta.get("default_infill", 20),
                "estimated_time_min": meta.get("estimated_time_min", 60),
                "estimated_material_g": meta.get("estimated_material_g", 50),
                "keywords": meta.get("keywords", []),
            }
    else:
        # Fallback: register functions with minimal metadata
        for key, func in _GEOMETRY_FUNCS.items():
            catalog[key] = {
                "func": func, "name": key.replace("_", " ").title(),
                "description": "", "default_material": "PLA",
                "default_infill": 20, "estimated_time_min": 60,
                "estimated_material_g": 50, "keywords": key.split("_"),
            }

    return catalog


PRODUCTS = _load_products()


def get_function_params(product_type: str) -> dict:
    """Get the parameter names and defaults for a product function."""
    entry = PRODUCTS.get(product_type)
    if not entry:
        return {}
    func = entry["func"]
    import inspect
    sig = inspect.signature(func)
    return {
        name: param.default if param.default is not inspect.Parameter.empty else None
        for name, param in sig.parameters.items()
    }
