"""CadQuery few-shot examples for LLM code generation.

Each category has a WORKING, TESTED CadQuery snippet that the LLM uses as a
reference when generating code for a similar product.  This is the difference
between "a box" and "something that looks like a car."

Every example must:
1. Use only basic CadQuery ops (box, cylinder, union, cut, translate, rotate, fillet)
2. Assign the final solid to `result`
3. Actually run and produce recognizable geometry
"""

# ── Category → (keywords, example_code) ──────────────────────────────

CATEGORIES: dict[str, dict] = {

    "vehicle": {
        "keywords": ["car", "truck", "bus", "vehicle", "van", "jeep", "suv", "wagon", "race"],
        "example": '''\
# Toy car — chassis + cabin + 4 wheels + front/rear bumpers
body = cq.Workplane("XY").box(80, 35, 12)  # chassis
cabin = cq.Workplane("XY").box(35, 30, 16).translate((5, 0, 14))  # cabin, offset toward rear
body = body.union(cabin)

# Hood slope — cut a wedge from the front of the cabin
hood_cut = cq.Workplane("XY").box(20, 32, 20).translate((30, 0, 14))
body = body.cut(hood_cut)
hood = cq.Workplane("XY").box(20, 30, 8).translate((30, 0, 10))
body = body.union(hood)

# Wheels — cylinders rotated to sit on the sides
for x_pos in [22, -22]:
    for y_sign in [1, -1]:
        wheel = (cq.Workplane("XZ")
                 .circle(8).extrude(5)
                 .translate((x_pos, y_sign * 20, -2)))
        body = body.union(wheel)

# Bumpers
front = cq.Workplane("XY").box(4, 30, 8).translate((42, 0, 0))
rear = cq.Workplane("XY").box(4, 30, 8).translate((-42, 0, 0))
body = body.union(front).union(rear)

# Headlights — small cylinders
for y in [10, -10]:
    hl = cq.Workplane("YZ").circle(3).extrude(3).translate((41, y, 4))
    body = body.cut(hl)

try:
    body = body.edges("|Z").fillet(2)
except Exception:
    pass

result = body
''',
    },

    "container": {
        "keywords": ["vase", "cup", "bowl", "pot", "jar", "mug", "bottle", "planter",
                      "bucket", "container", "bin", "can"],
        "example": '''\
# Vase — outer shell with hollowed interior, flared rim
outer_r_base = 30    # base radius
outer_r_mid = 22     # narrow waist
outer_r_top = 32     # flared rim
height = 120
wall = 3

# Build as a revolved profile
pts = [
    (0, 0),
    (outer_r_base, 0),
    (outer_r_mid, height * 0.4),
    (outer_r_top, height),
    (outer_r_top - wall, height),
    (outer_r_mid - wall, height * 0.4),
    (outer_r_base - wall, 0 + wall),
    (0, wall),
]

body = (cq.Workplane("XZ")
        .polyline(pts).close()
        .revolve(360, (0, 0, 0), (0, 1, 0)))

result = body
''',
    },

    "furniture": {
        "keywords": ["shelf", "table", "chair", "bench", "stool", "rack", "hook",
                      "stand", "holder", "mount", "bracket", "hanger"],
        "example": '''\
# Wall shelf bracket — L-shaped with support rib
arm_l = 120   # horizontal arm
arm_h = 80    # vertical mount plate
thick = 5
width = 30

# Vertical mount plate
mount = cq.Workplane("XY").box(width, thick, arm_h)

# Horizontal shelf arm
arm = cq.Workplane("XY").box(width, arm_l, thick).translate((0, arm_l/2 - thick/2, -arm_h/2 + thick/2))
body = mount.union(arm)

# Diagonal support rib
rib = cq.Workplane("XY").box(width - 8, thick, arm_h * 0.7)
rib = rib.rotate((0,0,0), (1,0,0), 45)
rib = rib.translate((0, 25, -15))
body = body.union(rib)

# Screw holes in mount plate
for z in [arm_h/2 - 15, -arm_h/2 + 15]:
    hole = cq.Workplane("XY").circle(2.5).extrude(thick + 2).translate((0, 0, z))
    body = body.cut(hole)

try:
    body = body.edges().fillet(1.5)
except Exception:
    pass

result = body
''',
    },

    "tool": {
        "keywords": ["wrench", "tool", "clamp", "grip", "plier", "screwdriver",
                      "hammer", "lever", "handle", "knob", "gear", "pulley"],
        "example": '''\
# Open-end wrench — handle + jaw
handle_l = 120
handle_w = 18
handle_t = 5
jaw_opening = 16
jaw_depth = 25

# Handle
handle = cq.Workplane("XY").box(handle_l, handle_w, handle_t)

# Jaw block at the end
jaw_block = cq.Workplane("XY").box(jaw_depth, handle_w + 10, handle_t).translate((handle_l/2 + jaw_depth/2 - 5, 0, 0))
body = handle.union(jaw_block)

# Cut the jaw opening
jaw_cut = cq.Workplane("XY").box(jaw_depth + 2, jaw_opening, handle_t + 2).translate((handle_l/2 + jaw_depth/2 - 2, 0, 0))
body = body.cut(jaw_cut)

# Hanging hole at the other end
hang_hole = cq.Workplane("XY").circle(4).extrude(handle_t + 2).translate((-handle_l/2 + 10, 0, -1))
body = body.cut(hang_hole)

try:
    body = body.edges().fillet(1.2)
except Exception:
    pass

result = body
''',
    },

    "accessory": {
        "keywords": ["ring", "pendant", "keychain", "bookmark", "tag", "badge",
                      "ornament", "decoration", "charm", "token", "coin"],
        "example": '''\
# Heart-shaped bookmark clip
# Flat body with a heart cutout and a clip tab

body_l = 80
body_w = 25
body_t = 2

# Main flat body
body = cq.Workplane("XY").box(body_l, body_w, body_t)

# Heart cutout — approximate with two circles + a triangle
heart_r = 5
c1 = cq.Workplane("XY").circle(heart_r).extrude(body_t + 2).translate((body_l/2 - 15, heart_r * 0.5, -1))
c2 = cq.Workplane("XY").circle(heart_r).extrude(body_t + 2).translate((body_l/2 - 15, -heart_r * 0.5, -1))
tri = cq.Workplane("XY").box(heart_r * 2, heart_r * 1.5, body_t + 2).translate((body_l/2 - 15, 0, -1))
body = body.cut(c1).cut(c2).cut(tri)

# Clip tab at the bottom (folds over page)
clip = cq.Workplane("XY").box(15, body_w, body_t).translate((-body_l/2 - 5, 0, -body_t))
body = body.union(clip)

try:
    body = body.edges("|Z").fillet(1)
except Exception:
    pass

result = body
''',
    },

    "electronics": {
        "keywords": ["case", "enclosure", "box", "housing", "cover", "shell",
                      "raspberry", "arduino", "pcb", "board", "mount"],
        "example": '''\
# Electronics project enclosure — box with lid tabs, ventilation, and port holes
inner_l = 85
inner_w = 56
inner_h = 30
wall = 2.5

# Outer shell (open top)
outer = cq.Workplane("XY").box(inner_l + 2*wall, inner_w + 2*wall, inner_h + wall)
inner = cq.Workplane("XY").box(inner_l, inner_w, inner_h).translate((0, 0, wall))
body = outer.cut(inner)

# USB port cutout on the short side
usb = cq.Workplane("YZ").box(12, 6, wall + 4).translate((inner_l/2 + wall, 0, wall + 8))
body = body.cut(usb)

# Ventilation slots on the long side
for i in range(5):
    slot = cq.Workplane("XZ").box(15, 2, wall + 2).translate((0, inner_w/2 + wall, wall + 6 + i * 4))
    body = body.cut(slot)

# Mounting standoffs inside (4 corners)
for x_s in [1, -1]:
    for y_s in [1, -1]:
        post = cq.Workplane("XY").circle(3).extrude(8).translate((x_s * (inner_l/2 - 6), y_s * (inner_w/2 - 6), wall))
        screw = cq.Workplane("XY").circle(1.25).extrude(9).translate((x_s * (inner_l/2 - 6), y_s * (inner_w/2 - 6), wall))
        body = body.union(post).cut(screw)

try:
    body = body.edges("|Z").fillet(2)
except Exception:
    pass

result = body
''',
    },

    "general": {
        "keywords": [],
        "example": '''\
# Generic 3D printable object — replace with your specific geometry
# Build from boxes and cylinders combined with union/cut

base = cq.Workplane("XY").box(60, 40, 5)
feature = cq.Workplane("XY").box(30, 20, 25).translate((0, 0, 15))
body = base.union(feature)

# Add a cylindrical detail
cyl = cq.Workplane("XY").circle(10).extrude(30).translate((0, 0, 5))
body = body.union(cyl)

# Cut a hole
hole = cq.Workplane("XY").circle(5).extrude(35).translate((0, 0, 3))
body = body.cut(hole)

try:
    body = body.edges().fillet(1.5)
except Exception:
    pass

result = body
''',
    },
}


def match_category(text: str) -> str:
    """Match free text to the best example category. Returns category key."""
    text_lower = text.lower()
    best, best_score = "general", 0
    for cat, info in CATEGORIES.items():
        if cat == "general":
            continue
        score = sum(1 for kw in info["keywords"] if kw in text_lower)
        if score > best_score:
            best, best_score = cat, score
    return best


def get_example(category: str) -> str:
    """Get the CadQuery example code for a category."""
    return CATEGORIES.get(category, CATEGORIES["general"])["example"]
