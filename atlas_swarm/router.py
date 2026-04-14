"""Q-Learning self-evolving router for agent task assignment.

Routes tasks to the best-performing agent based on learned Q-values.
Updates after every task completion with reward signal.
Lean Six Sigma: this IS the Control step — it auto-improves routing
based on measured outcomes (Measure → Analyze → Improve are implicit).
"""
import json
import math
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .memory import _ensure_state_db


class QLearningRouter:
    def __init__(
        self,
        exploration_rate: float = 0.15,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
    ):
        self.epsilon = exploration_rate
        self.alpha = learning_rate
        self.gamma = discount_factor
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = _ensure_state_db()
        return self._conn

    def _q(self, state: str, action: str) -> float:
        row = self.conn.execute(
            "SELECT q_value FROM router_state WHERE state_action = ?",
            (f"{state}|{action}",),
        ).fetchone()
        return row[0] if row else 0.0

    def _set_q(self, state: str, action: str, value: float) -> None:
        key = f"{state}|{action}"
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO router_state (state_action, q_value, visits, updated_at) "
            "VALUES (?, ?, 1, ?) "
            "ON CONFLICT(state_action) DO UPDATE SET q_value=?, visits=visits+1, updated_at=?",
            (key, value, now, value, now),
        )
        self.conn.commit()

    def select_agent(self, task_type: str, available_agents: list[str]) -> str:
        """Select best agent for task using epsilon-greedy Q-learning."""
        if not available_agents:
            raise ValueError("No available agents")

        # Epsilon-greedy: explore with probability epsilon
        if random.random() < self.epsilon:
            choice = random.choice(available_agents)
            return choice

        # Exploit: pick agent with highest Q-value for this task type
        best_agent = available_agents[0]
        best_q = self._q(task_type, best_agent)
        for agent in available_agents[1:]:
            q = self._q(task_type, agent)
            if q > best_q:
                best_q = q
                best_agent = agent
        return best_agent

    def update(self, task_type: str, agent_id: str, reward: float,
               next_task_type: Optional[str] = None,
               next_agents: Optional[list[str]] = None) -> None:
        """Update Q-value after task completion.

        reward > 0 for success (task completed, revenue generated, etc.)
        reward < 0 for failure (crash, timeout, quality rejection)
        """
        current_q = self._q(task_type, agent_id)

        # Max future Q-value (Bellman equation)
        max_future_q = 0.0
        if next_task_type and next_agents:
            max_future_q = max(self._q(next_task_type, a) for a in next_agents)

        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self._set_q(task_type, agent_id, new_q)

    def get_stats(self) -> list[dict]:
        """Return router performance stats for DMAIC dashboard."""
        rows = self.conn.execute(
            "SELECT state_action, q_value, visits, updated_at FROM router_state ORDER BY q_value DESC"
        ).fetchall()
        return [{"state_action": r[0], "q_value": round(r[1], 4), "visits": r[2],
                 "updated_at": r[3]} for r in rows]
