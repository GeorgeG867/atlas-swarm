"""Curated first-product catalog — hand-picked simple consumer items that
pass the printability gate and have proven Etsy/Amazon demand.

Used as a FALLBACK when the upstream opportunity source (IdeaFrog) returns
nothing that passes the printability gate.  IdeaFrog's patent-derived
opportunities are heavily biased toward industrial/ML patents; until that
upstream problem is fixed, this catalog guarantees the pipeline always has a
sensible first product to design.

Each entry follows the IdeaFrog opportunity shape so downstream code
(kie, ideafrog.opportunity_to_prompt, executive_agents.CTO) sees the same
fields it expects from a real opportunity.
"""
from __future__ import annotations

FIRST_PRODUCT_CATALOG: list[dict] = [
    {
        "id": "fp_phone_stand_adjustable",
        "title": "Adjustable desktop phone stand",
        "description": (
            "Minimalist desktop phone stand with adjustable viewing angle. "
            "Single-piece FDM print.  Compatible with all phones 55-90mm wide. "
            "Cable cutout at the back."
        ),
        "target_market": "remote workers, students, video callers",
        "patent_mechanism": "cantilever tongue supports phone at 45-75 degree angle; base with non-slip pad recess",
        "market_size": 25_000_000,
        "revenue_model": "marketplace direct sale",
        "swarm_score": 90,
        "source": "first_product_catalog",
    },
    {
        "id": "fp_cable_clip",
        "title": "Desk edge cable management clip",
        "description": (
            "Snap-fit cable organizer that clips to desk edges (14-30mm thick). "
            "Routes 3-6 cables.  No adhesive required.  Single-piece print."
        ),
        "target_market": "desk setup, WFH, cable organization",
        "patent_mechanism": "flexure tongue snaps onto desk edge; parallel channels hold cables; living hinge",
        "market_size": 15_000_000,
        "revenue_model": "3-pack bundle",
        "swarm_score": 88,
        "source": "first_product_catalog",
    },
    {
        "id": "fp_headphone_hook",
        "title": "Under-desk headphone hook",
        "description": (
            "Adhesive-free headphone mount that clamps to desk underside. "
            "Holds one pair of over-ear headphones.  Single-piece print with "
            "integrated cable-routing slot."
        ),
        "target_market": "gamers, streamers, desk setups",
        "patent_mechanism": "C-clamp body with cantilever hook; 30mm depth bar fits standard desks",
        "market_size": 8_000_000,
        "revenue_model": "marketplace direct",
        "swarm_score": 87,
        "source": "first_product_catalog",
    },
    {
        "id": "fp_drawer_divider",
        "title": "Adjustable drawer divider",
        "description": (
            "Expandable drawer divider with ratchet detents, fits drawers "
            "30-60cm wide.  Single-piece print with living-hinge expansion rail."
        ),
        "target_market": "kitchens, closets, office organization",
        "patent_mechanism": "two-piece interlocking sled; ratchet teeth at 5mm pitch lock length",
        "market_size": 20_000_000,
        "revenue_model": "2-pack, multi-size kits",
        "swarm_score": 86,
        "source": "first_product_catalog",
    },
    {
        "id": "fp_pen_caddy",
        "title": "Desktop pen & pencil caddy",
        "description": (
            "Five-compartment desktop caddy for pens, pencils, scissors, ruler, and "
            "small tools.  Single-piece print with reinforced bottom wall."
        ),
        "target_market": "students, designers, home office",
        "patent_mechanism": "cylindrical compartments offset from square base for stability",
        "market_size": 12_000_000,
        "revenue_model": "marketplace direct, Etsy handmade",
        "swarm_score": 85,
        "source": "first_product_catalog",
    },
    {
        "id": "fp_earbud_case",
        "title": "AirPods / earbud carry case",
        "description": (
            "Clip-on carry case for AirPods / wireless earbuds.  Single-piece "
            "print with snap-fit lid and keychain loop."
        ),
        "target_market": "commuters, gym, travel",
        "patent_mechanism": "cantilever snap-fit lid; living hinge across the back; D-ring loop",
        "market_size": 18_000_000,
        "revenue_model": "marketplace direct, color variants",
        "swarm_score": 84,
        "source": "first_product_catalog",
    },
    {
        "id": "fp_plant_stake",
        "title": "Labeled plant stake set",
        "description": (
            "Waterproof plant labels with integrated stake.  Writable surface, "
            "single-piece print.  Pack of 10."
        ),
        "target_market": "indoor gardeners, herb gardens",
        "patent_mechanism": "stake with embossed writing area; flared tip for soil retention",
        "market_size": 6_000_000,
        "revenue_model": "10-pack, themed sets",
        "swarm_score": 82,
        "source": "first_product_catalog",
    },
    {
        "id": "fp_bag_clip",
        "title": "Snack-bag closure clip",
        "description": (
            "Snap-lock clip for snack and chip bags.  Integrated living-hinge "
            "spring, no metal parts.  Single-piece print in PETG for durability."
        ),
        "target_market": "kitchen, pantry",
        "patent_mechanism": "cantilever jaw with living-hinge spring; locking detent keeps jaw closed",
        "market_size": 10_000_000,
        "revenue_model": "5-pack bundle",
        "swarm_score": 81,
        "source": "first_product_catalog",
    },
    {
        "id": "fp_doorstop_wedge",
        "title": "Non-slip doorstop wedge",
        "description": (
            "Heavy doorstop wedge with ridged underside for grip on carpet or hard "
            "floors.  Single-piece print with internal honeycomb infill."
        ),
        "target_market": "homes, offices",
        "patent_mechanism": "wedge profile 12 degrees; ridged underside friction pattern",
        "market_size": 5_000_000,
        "revenue_model": "marketplace direct",
        "swarm_score": 80,
        "source": "first_product_catalog",
    },
    {
        "id": "fp_keyring_carabiner",
        "title": "Carabiner keyring",
        "description": (
            "3D-printed carabiner-style keyring with spring-gate closure.  "
            "Single-piece print using living-hinge spring — no metal parts."
        ),
        "target_market": "EDC, outdoor, gift",
        "patent_mechanism": "spring gate via living hinge; load-rated for keys/light gear only",
        "market_size": 4_000_000,
        "revenue_model": "marketplace direct, multipack",
        "swarm_score": 78,
        "source": "first_product_catalog",
    },
]


def get_fallback_opportunity(exclude_ids: set[str] | None = None) -> dict | None:
    """Return the highest-scoring catalog item not in exclude_ids."""
    exclude_ids = exclude_ids or set()
    for item in sorted(FIRST_PRODUCT_CATALOG, key=lambda o: -o.get("swarm_score", 0)):
        if item["id"] not in exclude_ids:
            return dict(item)
    return None


def get_fallback_list() -> list[dict]:
    """Return the full catalog sorted by swarm_score desc."""
    return [dict(item) for item in sorted(
        FIRST_PRODUCT_CATALOG,
        key=lambda o: -o.get("swarm_score", 0),
    )]
