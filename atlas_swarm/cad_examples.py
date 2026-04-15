"""CadQuery few-shot examples — loaded from YAML, zero hardcoded text.

Categories and examples live in templates/cad_examples.yaml.
Add new categories there, not here.
"""
import logging
import os
import re
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(os.environ.get(
    "SWARM_TEMPLATES",
    Path(__file__).resolve().parent.parent / "templates",
))

_categories: dict = {}


def _load() -> dict:
    """Load categories from YAML.  Cached after first call."""
    global _categories
    if _categories:
        return _categories

    yaml_path = _TEMPLATES_DIR / "cad_examples.yaml"
    if not yaml_path.exists():
        log.warning("[CAD-EX] %s not found — using empty categories", yaml_path)
        _categories = {"general": {"keywords": [], "example": "result = cq.Workplane('XY').box(50,30,10)\n"}}
        return _categories

    _categories = yaml.safe_load(yaml_path.read_text()) or {}
    log.info("[CAD-EX] Loaded %d categories from %s", len(_categories), yaml_path)
    return _categories


def match_category(text: str) -> str:
    """Match free text to the best example category via whole-word keywords."""
    cats = _load()
    words = set(re.findall(r'\b\w+\b', text.lower()))
    best, best_score = "general", 0
    for cat, info in cats.items():
        if cat == "general":
            continue
        score = sum(1 for kw in info.get("keywords", []) if kw in words)
        if score > best_score:
            best, best_score = cat, score
    return best


def get_example(category: str) -> str:
    """Get the CadQuery example code for a category."""
    cats = _load()
    entry = cats.get(category, cats.get("general", {}))
    return entry.get("example", "result = cq.Workplane('XY').box(50,30,10)\n")


def list_categories() -> list[str]:
    return list(_load().keys())
