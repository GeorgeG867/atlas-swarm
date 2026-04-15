"""Runtime config loader — loads prompts, constants, and templates from YAML.

Single source of truth: templates/*.yaml.  No hardcoded text or numbers in Python.
"""
import logging
import os
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

TEMPLATES_DIR = Path(os.environ.get(
    "SWARM_TEMPLATES",
    Path(__file__).resolve().parent.parent / "templates",
))

_cache: dict[str, dict] = {}


def load(filename: str) -> dict:
    """Load a YAML file from templates/ — cached after first call."""
    if filename in _cache:
        return _cache[filename]

    path = TEMPLATES_DIR / filename
    if not path.exists():
        log.warning("[CONFIG] %s not found", path)
        _cache[filename] = {}
        return _cache[filename]

    _cache[filename] = yaml.safe_load(path.read_text()) or {}
    log.info("[CONFIG] Loaded %s", filename)
    return _cache[filename]


def reload():
    """Clear the cache — next load() call re-reads from disk."""
    _cache.clear()


def get(path: str, default: Any = None) -> Any:
    """Get a nested config value by dotted path.

    Example: get('prompts.cad_design.user', 'prompts') -> the prompt template
    The first segment is the YAML filename (without .yaml).
    """
    parts = path.split(".")
    filename = parts[0] + ".yaml"
    data = load(filename)
    for p in parts[1:]:
        if not isinstance(data, dict):
            return default
        data = data.get(p)
        if data is None:
            return default
    return data


# Convenience accessors
def printer_constraints() -> dict:
    return load("cad_prompts.yaml").get("printer", {})


def prompt_template(kind: str, variant: str = "user") -> str:
    """Get a prompt template string.  kind='cad_design', variant='user'|'retry'"""
    return load("cad_prompts.yaml").get("prompts", {}).get(kind, {}).get(variant, "")
