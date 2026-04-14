"""Base agent class with persistent memory, self-improvement, and DMAIC.

Every agent in the Atlas Swarm inherits from this. It provides:
1. Persistent memory (read/write to Obsidian vault + SQLite)
2. LLM inference (local Ollama or cloud fallback)
3. DMAIC cycle tracking (Lean Six Sigma)
4. Self-improvement: after every N tasks, agent reviews its own performance
   and adjusts its system prompt based on what worked/failed.
"""
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from .memory import read_memories, record_metric, write_memory

log = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:27b")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SELF_IMPROVE_EVERY = int(os.environ.get("SELF_IMPROVE_EVERY", "10"))


class SwarmAgent(ABC):
    """Base class for all Atlas Swarm agents."""

    def __init__(self, agent_id: str, role: str, goal: str, backstory: str):
        self.agent_id = agent_id
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self._task_count = 0
        self._success_count = 0
        self._fail_count = 0
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        # Load any self-improvement adjustments from memory
        adjustments = read_memories(agent_id=self.agent_id, category="learnings", limit=5)
        adjustment_text = ""
        if adjustments:
            adjustment_text = "\n\nSELF-IMPROVEMENT NOTES (from your past performance):\n"
            for a in adjustments:
                adjustment_text += f"- {a['title']}: {a['content'][:200]}\n"

        return f"""You are {self.role} in the Atlas Swarm product delivery system.

GOAL: {self.goal}

BACKSTORY: {self.backstory}

OPERATING PRINCIPLES (Lean Six Sigma):
- Define: Know exactly what success looks like before starting
- Measure: Track every action's outcome quantitatively
- Analyze: When results miss targets, identify root cause (not symptoms)
- Improve: Propose specific, testable changes
- Control: Lock in improvements; monitor for regression

CONSTRAINTS:
- Never fabricate data. If unsure, say so and suggest how to verify.
- Budget decisions >$500 require CFO-Agent approval.
- Health/medical claims require human-in-loop approval.
- Log every decision rationale to memory for auditability.
{adjustment_text}"""

    async def llm(self, prompt: str, max_tokens: int = 4096) -> str:
        """Call LLM -- local Ollama first, Anthropic fallback."""
        try:
            return await self._ollama(prompt, max_tokens)
        except Exception as e:
            log.warning(f"[{self.agent_id}] Ollama failed: {e}, falling back to Anthropic")
            if ANTHROPIC_KEY:
                return await self._anthropic(prompt, max_tokens)
            raise

    async def _ollama(self, prompt: str, max_tokens: int) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    async def _anthropic(self, prompt: str, max_tokens: int) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": max_tokens,
                    "system": self._system_prompt,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]

    @abstractmethod
    async def execute(self, task: dict) -> dict:
        """Execute a task. Returns {success: bool, result: Any, metrics: dict}."""
        ...

    async def run_task(self, task: dict) -> dict:
        """Wrapper: execute task + track metrics + trigger self-improvement."""
        self._task_count += 1
        start = datetime.now(timezone.utc)
        try:
            result = await self.execute(task)
            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            success = result.get("success", False)
            if success:
                self._success_count += 1
            else:
                self._fail_count += 1

            # Record metrics (DMAIC: Measure)
            record_metric(f"{self.agent_id}.task_success", 1.0 if success else 0.0, self.agent_id)
            record_metric(f"{self.agent_id}.latency_s", elapsed, self.agent_id)

            # Self-improvement check (DMAIC: Analyze + Improve)
            if self._task_count % SELF_IMPROVE_EVERY == 0:
                await self._self_improve()

            return result
        except Exception as e:
            self._fail_count += 1
            record_metric(f"{self.agent_id}.task_error", 1.0, self.agent_id)
            log.error(f"[{self.agent_id}] Task failed: {e}")
            return {"success": False, "error": str(e)}

    async def _self_improve(self) -> None:
        """DMAIC Analyze+Improve: review own performance and adjust system prompt."""
        rate = self._success_count / max(self._task_count, 1)
        recent = read_memories(agent_id=self.agent_id, category="feedback", limit=10)
        feedback_text = "\n".join(f"- {m['title']}: {m['content'][:100]}" for m in recent)

        prompt = f"""You are reviewing your own performance as {self.role}.

Success rate: {rate:.1%} ({self._success_count}/{self._task_count})
Failure count: {self._fail_count}

Recent feedback:
{feedback_text or '(no feedback yet)'}

Based on this data, write 1-3 specific, actionable improvements to your approach.
Focus on what to DO DIFFERENTLY, not what you did wrong.
Be concrete: "Always check price before listing" not "Be more careful."
"""
        try:
            improvements = await self.llm(prompt, max_tokens=500)
            write_memory(
                agent_id=self.agent_id,
                category="learnings",
                title=f"Self-improvement cycle {self._task_count}",
                content=improvements,
                confidence=0.7,
            )
            # Rebuild system prompt with new learnings
            self._system_prompt = self._build_system_prompt()
            log.info(f"[{self.agent_id}] Self-improved after {self._task_count} tasks")
        except Exception as e:
            log.warning(f"[{self.agent_id}] Self-improvement failed: {e}")

    @property
    def stats(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "tasks": self._task_count,
            "successes": self._success_count,
            "failures": self._fail_count,
            "rate": round(self._success_count / max(self._task_count, 1), 3),
        }
