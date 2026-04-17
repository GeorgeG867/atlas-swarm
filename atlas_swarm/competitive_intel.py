"""Competitive intelligence step — runs BEFORE the CTO designs anything.

Given a product opportunity (title + mechanism), asks AIM to surface the
top-selling real-world competitors on Amazon/Etsy and extract the features
customers pay for.  The output is fed into the CTO's CAD prompt as design
requirements so the STL reflects proven market demand, not a fifth-grader box.

Pragmatic note: we don't currently scrape live marketplace data — AIM draws
on its training knowledge of common consumer products.  This is enough to
nudge the CTO toward recognizable, competitive geometry.  Swap in a real
web-search call later if marketplace scraping becomes a priority.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from .aim import get_aim

log = logging.getLogger(__name__)


async def competitive_analysis(opportunity: dict) -> dict:
    """Generate competitive context for an opportunity.

    Returns a dict with:
        competitors: list of {name, price_range, key_features, weakness}
        must_haves: list of features every top seller includes
        differentiators: list of features that let us stand out
        target_dimensions_mm: approximate bounding box from best-seller norm
        target_price_usd: suggested retail anchored to top sellers

    Never raises — on AIM failure returns a minimal stub so the pipeline
    always has something to hand to the CTO.
    """
    title = str(opportunity.get("title", ""))[:120]
    mechanism = str(opportunity.get("patent_mechanism", ""))[:300]
    market = str(opportunity.get("market", "") or opportunity.get("target_market", ""))[:120]

    aim = get_aim()
    prompt = f"""You are an e-commerce product analyst.

PRODUCT BRIEF:
- Title: {title}
- Target market: {market}
- Patent mechanism (what the product must do): {mechanism}

Imagine you searched Amazon and Etsy for this product category.  List the top
3 best sellers you would expect to see (use realistic product types — don't
invent brand names you aren't sure about).

For EACH top seller, extract:
- name: a generic product type label (e.g. "Adjustable aluminum phone stand")
- price_range_usd: [low, high]
- key_features: 3-5 features customers highlight in reviews
- weakness: one complaint or gap customers frequently mention

Then synthesize:
- must_haves: features EVERY top seller has.  If our product is missing these, it won't sell.
- differentiators: 2-3 features we can add that address the weaknesses
- target_dimensions_mm: [width, depth, height] — the bounding box of a typical top seller
- target_price_usd: the price point we should anchor to

Return ONLY valid JSON with keys: competitors, must_haves, differentiators, target_dimensions_mm, target_price_usd.
Use lowercase snake_case keys.  No prose outside the JSON.
"""

    try:
        result = await aim.generate(
            prompt=prompt,
            system="You are a pragmatic e-commerce analyst. Ground claims in common-knowledge product categories. Output valid JSON only.",
            task_type="json_task",
            max_tokens=1200,
            require_local=True,
        )
        text = result.get("text", "")
        intel = _extract_json(text)
        if intel:
            intel["_aim_model"] = result.get("model")
            return intel
    except Exception as exc:
        log.warning("[COMPETITIVE] AIM call failed: %s", exc)

    # Minimal fallback — keeps the pipeline moving.
    return {
        "competitors": [],
        "must_haves": ["durable single-piece construction", "non-slip base or mounting surface"],
        "differentiators": ["adjustable geometry", "cable management cutout"],
        "target_dimensions_mm": [120, 80, 100],
        "target_price_usd": 18,
        "_fallback": True,
    }


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of an LLM response."""
    # Prefer a fenced block first.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    # Fall back to the first balanced-looking object.
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group())
        except json.JSONDecodeError:
            return None
    return None


def summarize_for_prompt(intel: dict, max_chars: int = 900) -> str:
    """Human-readable summary of the intel, sized for injection into a CAD prompt."""
    if not intel:
        return "(no competitive intel available)"
    parts: list[str] = []
    comps = intel.get("competitors", []) or []
    if comps:
        parts.append("TOP COMPETITORS:")
        for c in comps[:3]:
            name = c.get("name", "unknown")
            price = c.get("price_range_usd") or c.get("price_range") or []
            feats = c.get("key_features") or []
            weak = c.get("weakness") or ""
            price_s = f"${price[0]}-${price[1]}" if isinstance(price, list) and len(price) == 2 else ""
            parts.append(f"- {name} {price_s}")
            if feats:
                parts.append(f"  features: {'; '.join(str(f) for f in feats[:4])}")
            if weak:
                parts.append(f"  weakness: {weak}")
    must = intel.get("must_haves") or []
    if must:
        parts.append("MUST-HAVE FEATURES (every best-seller): " + "; ".join(str(m) for m in must[:6]))
    diff = intel.get("differentiators") or []
    if diff:
        parts.append("DIFFERENTIATORS (to add): " + "; ".join(str(d) for d in diff[:4]))
    dims = intel.get("target_dimensions_mm") or intel.get("target_dimensions") or []
    if isinstance(dims, list) and len(dims) == 3:
        parts.append(f"TARGET BOUNDING BOX: {dims[0]} x {dims[1]} x {dims[2]} mm")
    price = intel.get("target_price_usd") or intel.get("target_price")
    if price:
        parts.append(f"TARGET PRICE: ${price}")
    out = "\n".join(parts)
    if len(out) > max_chars:
        out = out[: max_chars - 4] + " ..."
    return out
