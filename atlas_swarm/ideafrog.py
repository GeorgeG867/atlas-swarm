"""IdeaFrog client — pulls scored product opportunities from Server 1.

IdeaFrog runs at http://192.168.1.204:5001 on Server 1 with 118+ pre-scored
opportunities from the patent/knowledge pipeline.  This module fetches them
and shapes each one into a text-to-3D prompt for the autonomous viewer.
"""
import logging
import os
import re
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger(__name__)

IDEAFROG_URL = os.environ.get("IDEAFROG_URL", "http://192.168.1.204:5001")
RENDERS_DIR = Path(os.environ.get("RENDERS_DIR", Path.home() / "Projects/atlas-swarm/renders"))


async def health() -> dict:
    """Quick health + count check."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{IDEAFROG_URL}/health")
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        return {"status": "unreachable", "error": str(exc)}


async def fetch_opportunities(limit: int = 50, min_score: float = 0.0) -> list[dict]:
    """Return opportunities sorted by swarm_score desc.  Filters by min_score."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{IDEAFROG_URL}/opportunities", params={"limit": limit})
            r.raise_for_status()
            data = r.json()
            opps = data.get("opportunities", data if isinstance(data, list) else [])
            opps = [o for o in opps if o.get("swarm_score", 0) >= min_score]
            opps.sort(key=lambda o: o.get("swarm_score", 0), reverse=True)
            return opps
    except Exception as exc:
        log.warning("[IDEAFROG] fetch_opportunities failed: %s", exc)
        return []


def _clean_title(raw: str) -> str:
    """Strip 'ProductName:' prefix and other noise from opportunity titles."""
    t = re.sub(r"^(ProductName|Product Name|Product):\s*", "", raw.strip(), flags=re.I)
    return t[:120].strip()


def opportunity_to_prompt(opp: dict) -> dict:
    """Shape an IdeaFrog opportunity into a text-to-3D prompt payload."""
    title = _clean_title(opp.get("title", ""))
    market = opp.get("target_market", "")
    mechanism = opp.get("patent_mechanism", "") or opp.get("unique_angle", "")

    # Build a concise, concrete description for image generation
    pieces = [title]
    if market and market.lower() not in title.lower():
        pieces.append(f"for {market}")
    if mechanism:
        pieces.append(mechanism[:200])

    description = ". ".join(p for p in pieces if p)

    # Safe filename stem — deterministic per opportunity, so we can detect "already rendered"
    safe_id = opp.get("id", "unknown")
    safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", title.lower()).strip("_")[:40]
    stem = f"if_{safe_id}_{safe_title}" if safe_title else f"if_{safe_id}"

    return {
        "id": opp.get("id"),
        "title": title,
        "description": description,
        "market": market,
        "market_size": opp.get("market_size"),
        "revenue_model": opp.get("revenue_model"),
        "swarm_score": opp.get("swarm_score"),
        "triz_principle": opp.get("triz_principle"),
        "stem": stem,
        "source": "ideafrog",
    }


def _existing_glb_for(stem: str) -> Optional[Path]:
    matches = list(RENDERS_DIR.glob(f"{stem}_*.glb"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


async def next_unrendered(min_score: float = 50.0) -> Optional[dict]:
    """Return the highest-scoring opportunity that does NOT yet have a GLB."""
    opps = await fetch_opportunities(limit=100, min_score=min_score)
    for opp in opps:
        payload = opportunity_to_prompt(opp)
        if _existing_glb_for(payload["stem"]) is None:
            return payload
    # Every opp already rendered — return the top one anyway so callers get something
    if opps:
        return opportunity_to_prompt(opps[0])
    return None


async def top_opportunities(n: int = 10) -> list[dict]:
    opps = await fetch_opportunities(limit=n)
    return [opportunity_to_prompt(o) for o in opps]
