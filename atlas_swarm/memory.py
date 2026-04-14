"""Obsidian-compatible persistent memory for agents.

Every agent reads/writes markdown files with YAML frontmatter.
Memory is searchable, version-trackable, and human-readable.
Files live in memory-vault/ and are indexed by Obsidian.
"""
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml


VAULT_PATH = Path(os.environ.get("ATLAS_VAULT", Path.home() / "Projects/atlas-swarm/memory-vault"))
STATE_DB = Path(os.environ.get("ATLAS_STATE_DB", Path.home() / "Projects/atlas-swarm/swarm-state.db"))


def _ensure_state_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(STATE_DB), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agent_memory (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            category TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata_json TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_am_agent ON agent_memory(agent_id);
        CREATE INDEX IF NOT EXISTS idx_am_cat ON agent_memory(category);
        CREATE INDEX IF NOT EXISTS idx_am_updated ON agent_memory(updated_at);

        CREATE TABLE IF NOT EXISTS router_state (
            state_action TEXT PRIMARY KEY,
            q_value REAL DEFAULT 0.0,
            visits INTEGER DEFAULT 0,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            agent_id TEXT,
            recorded_at TEXT NOT NULL,
            metadata_json TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);
        CREATE INDEX IF NOT EXISTS idx_metrics_time ON metrics(recorded_at);
    """)
    return conn


def write_memory(agent_id: str, category: str, title: str, content: str,
                 confidence: float = 0.5, metadata: Optional[dict] = None) -> str:
    """Write a memory entry to both vault (markdown) and state DB."""
    now = datetime.now(timezone.utc).isoformat()
    mem_id = f"{agent_id}_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    meta = metadata or {}

    # Write markdown to vault
    vault_dir = VAULT_PATH / category
    vault_dir.mkdir(parents=True, exist_ok=True)
    md_path = vault_dir / f"{mem_id}.md"
    frontmatter = {
        "id": mem_id,
        "agent_id": agent_id,
        "category": category,
        "title": title,
        "confidence": confidence,
        "created_at": now,
        "updated_at": now,
        **meta,
    }
    md_path.write_text(f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{content}\n")

    # Write to SQLite state DB
    conn = _ensure_state_db()
    conn.execute(
        "INSERT OR REPLACE INTO agent_memory (id, agent_id, category, title, content, confidence, created_at, updated_at, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (mem_id, agent_id, category, title, content, confidence, now, now, json.dumps(meta)),
    )
    conn.commit()
    conn.close()
    return mem_id


def read_memories(agent_id: Optional[str] = None, category: Optional[str] = None,
                  limit: int = 20) -> list[dict]:
    """Read recent memories, optionally filtered by agent and category."""
    conn = _ensure_state_db()
    query = "SELECT id, agent_id, category, title, content, confidence, updated_at FROM agent_memory WHERE 1=1"
    params: list[Any] = []
    if agent_id:
        query += " AND agent_id = ?"
        params.append(agent_id)
    if category:
        query += " AND category = ?"
        params.append(category)
    query += " ORDER BY updated_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [{"id": r[0], "agent_id": r[1], "category": r[2], "title": r[3],
             "content": r[4], "confidence": r[5], "updated_at": r[6]} for r in rows]


def record_metric(metric_name: str, value: float, agent_id: str = "system",
                  metadata: Optional[dict] = None) -> None:
    """Record a KPI metric for DMAIC tracking."""
    conn = _ensure_state_db()
    conn.execute(
        "INSERT INTO metrics (metric_name, metric_value, agent_id, recorded_at, metadata_json) VALUES (?, ?, ?, ?, ?)",
        (metric_name, value, agent_id, datetime.now(timezone.utc).isoformat(), json.dumps(metadata or {})),
    )
    conn.commit()
    conn.close()
