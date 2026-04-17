"""Printability gate — filters IdeaFrog opportunities for FDM-friendly, single-part
consumer products.

George's rule: first 10 products must be things you can hold in one hand and print
on a single FDM printer in under 12h with under $30 of material.  No industrial
machines, medical assemblies, or robotic grippers.

`printability_score(opp) -> (score: float, reasons: list[str])`
  - score 0.0 = REJECT (never ship this first)
  - score >= 1.0 = candidate
  - higher = simpler / closer to proven consumer category

Used by kie.select_top_products and ideafrog.next_unrendered as a hard filter.
"""
from __future__ import annotations

import re
from typing import Iterable

# Hard-reject tokens — these are not first products.  Matched as whole words.
REJECT_TOKENS: set[str] = {
    # Medical/surgical
    "surgical", "surgery", "medical", "clinical", "fda", "pharmaceutical",
    "implant", "prosthetic", "prosthesis", "catheter", "syringe", "stent",
    # Industrial / heavy
    "industrial", "factory", "assembly", "conveyor", "cnc",
    "robotic", "robot", "gripper", "actuator", "servo", "motorized",
    "hydraulic", "pneumatic", "electromechanical",
    # Fabrication / semiconductor
    "semiconductor", "wafer", "lithography", "microlithography", "photonic",
    # Aerospace / automotive / defense
    "avionic", "aerospace", "aircraft", "spacecraft", "satellite",
    "automotive", "chassis", "drivetrain", "powertrain", "turbine",
    "military", "weapon", "firearm", "munition",
    # Electrical
    "pcb", "circuit", "inverter", "transformer",
    # Software / SaaS patents — not a physical product at all
    "algorithm", "software", "api", "saas", "cloud", "analytics",
    "forecasting", "forecast", "prediction", "predictive", "predict",
    "detection", "detector", "classifier", "classification",
    "blockchain", "cryptocurrency", "cybersecurity", "encryption",
    "ai-powered", "machine-learning", "neural",
    # Multi-part / complex phrases (kept as tokens)
    "multi-part", "multi-component", "assembly-line",
}

# Soft-penalty tokens — reduce score but do not reject outright.
SOFT_PENALTY_TOKENS: set[str] = {
    "motor", "spring", "bearing", "gear", "solenoid", "sensor",
    "electronic", "electrical", "wire", "cable", "battery",
    "mechanism", "apparatus", "device", "system", "machine",
    "hinge", "clamping", "articulated",
}

# Preferred tokens — boost score.  Single-piece consumer items.
BOOST_TOKENS: set[str] = {
    "stand", "holder", "mount", "bracket", "hook",
    "clip", "cable-clip", "bag-clip", "binder-clip",
    "organizer", "divider", "tray", "bin", "basket",
    "caddy", "rack", "shelf", "stopper", "doorstop",
    "coaster", "desk", "pen", "pencil",
    "phone", "tablet", "earbud", "headphone", "earphone",
    "plant", "pot", "planter", "stake", "vase",
    "jig", "guide", "template", "fixture", "spacer",
    "ruler", "measuring", "caliper",
    "case", "cover", "sleeve", "enclosure",
    "handle", "knob", "grip", "pull",
    "hook", "keychain", "keyring",
    "toy", "fidget", "puzzle",
    "kitchen", "utensil",
    "drawer", "closet", "pantry",
}

# High-value physical attributes that imply single-piece 3D printable.
# NOTE: matched as whole-word regex against the full text — short tokens
# like "pla" must appear as a word, not inside "platform"/"place".
POSITIVE_PHRASES: list[str] = [
    "single piece", "one piece", "monolithic", "3d print", "3d-print",
    "fdm", "pla plastic", "petg", "desktop printer",
    "no assembly", "no tools", "single-piece",
]

# Industrial/complex phrases that imply NOT first-product material.
NEGATIVE_PHRASES: list[str] = [
    "multi-part assembly", "requires assembly", "motorized",
    "requires tools", "professional installation",
    "requires electricity", "battery powered",
]


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens — hyphens preserved as word-chars."""
    return set(re.findall(r"[a-z][a-z0-9\-]+", text.lower()))


def _contains_phrase(text: str, phrases: Iterable[str]) -> list[str]:
    """Word-boundary match so short phrases like 'fdm' don't match inside 'fdmp'."""
    lower = text.lower()
    hits = []
    for p in phrases:
        if not re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", lower):
            continue
        hits.append(p)
    return hits


def printability_score(opp: dict) -> tuple[float, list[str]]:
    """Score an IdeaFrog opportunity for FDM-single-part printability.

    Returns (score, reasons).  Score 0 = hard reject.
    """
    title = str(opp.get("title", ""))
    text_parts = [
        title,
        str(opp.get("description", "")),
        str(opp.get("target_market", "")),
        str(opp.get("patent_mechanism", "")),
        str(opp.get("unique_angle", "")),
        str(opp.get("domain", "")),
        str(opp.get("vertical", "")),
    ]
    text = " ".join(text_parts)
    tokens = _tokenize(text)
    title_tokens = _tokenize(title)
    reasons: list[str] = []

    # Hard reject on phrase OR token match
    neg_phrases = _contains_phrase(text, NEGATIVE_PHRASES)
    if neg_phrases:
        return 0.0, [f"reject:phrase:{p}" for p in neg_phrases]

    rejected = tokens & REJECT_TOKENS
    if rejected:
        return 0.0, [f"reject:token:{t}" for t in sorted(rejected)]

    score = 1.0

    # Soft penalties
    soft = tokens & SOFT_PENALTY_TOKENS
    if soft:
        score -= 0.15 * len(soft)
        reasons.append(f"penalty:{sorted(soft)}")

    # Boosts
    boosted = tokens & BOOST_TOKENS
    if boosted:
        score += 0.5 * len(boosted)
        reasons.append(f"boost:{sorted(boosted)}")

    pos_phrases = _contains_phrase(text, POSITIVE_PHRASES)
    if pos_phrases:
        score += 0.4 * len(pos_phrases)
        reasons.append(f"phrase-boost:{pos_phrases}")

    # Title-length heuristic: overly-long titles correlate with complex patents.
    title = str(opp.get("title", ""))
    if len(title) > 80:
        score -= 0.3
        reasons.append("penalty:long-title")

    # HARD gate: require a physical-noun signal in the TITLE specifically.
    # Description-level phrase matches aren't enough — "OptiPrinter" is a
    # software optimization tool, not a printable product, even though its
    # description mentions "3d print".  The title is the cleanest intent signal.
    title_boosts = title_tokens & BOOST_TOKENS
    title_pos_phrases = _contains_phrase(title, POSITIVE_PHRASES)
    if not title_boosts and not title_pos_phrases:
        return 0.0, reasons + ["reject:no-physical-signal-in-title"]

    # Clamp to zero if soft penalties drove it negative.
    if score < 0:
        return 0.0, reasons + ["reject:negative-score"]

    return round(score, 3), reasons


def is_printable(opp: dict, min_score: float = 1.0) -> bool:
    """True if the opportunity passes the gate at the given threshold."""
    score, _ = printability_score(opp)
    return score >= min_score


def filter_printable(opps: list[dict], min_score: float = 1.0) -> list[dict]:
    """Drop opportunities below the printability threshold and sort by score desc.

    Each surviving opp gets `_printability_score` and `_printability_reasons` added.
    """
    scored: list[dict] = []
    for opp in opps:
        s, reasons = printability_score(opp)
        if s >= min_score:
            opp = dict(opp)  # shallow copy, don't mutate caller's dict
            opp["_printability_score"] = s
            opp["_printability_reasons"] = reasons
            scored.append(opp)
    scored.sort(key=lambda o: o["_printability_score"], reverse=True)
    return scored
