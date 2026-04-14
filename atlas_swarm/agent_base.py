"""Production agent base — ALL inference goes through AIM. No exceptions.

Changes from experimental version:
1. ALL LLM calls route through AIM (not direct Ollama)
2. Self-evolution uses A/B comparison with metric revert
3. Default model is gemma4:26b (not gemma3)
4. Circuit breaker: detects repeated failures and stops
"""
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from .memory import read_memories, record_metric, write_memory

log = logging.getLogger(__name__)

SELF_IMPROVE_EVERY = int(os.environ.get("SELF_IMPROVE_EVERY", "10"))
MAX_CONSECUTIVE_FAILURES = int(os.environ.get("MAX_FAILURES", "5"))


class SwarmAgent(ABC):
    """Production base class. All agents inherit this."""

    def __init__(self, agent_id: str, role: str, goal: str, backstory: str):
        self.agent_id = agent_id
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self._task_count = 0
        self._success_count = 0
        self._fail_count = 0
        self._consecutive_fails = 0
        self._best_success_rate = 0.0
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        adjustments = read_memories(agent_id=self.agent_id, category="learnings", limit=5)
        adjustment_text = ""
        if adjustments:
            adjustment_text = "\n\nSELF-IMPROVEMENT NOTES (from measured performance):\n"
            for a in adjustments:
                adjustment_text += f"- {a['title']}: {a['content'][:200]}\n"

        return f"""You are {self.role} in the Atlas Swarm production system.

GOAL: {self.goal}

BACKSTORY: {self.backstory}

OPERATING PRINCIPLES (Lean Six Sigma DMAIC):
- Define: Know exactly what success looks like before starting
- Measure: Track every outcome quantitatively
- Analyze: Root-cause misses, not symptoms
- Improve: Specific, testable changes only
- Control: Lock improvements; monitor for regression

CONSTRAINTS:
- Never fabricate data. State uncertainty explicitly.
- Budget >$500 requires CFO-Agent approval.
- Health/medical claims require human approval.
- Log every decision rationale to memory.
- Return structured JSON when possible.
{adjustment_text}"""

    async def llm(self, prompt: str, task_type: str = "general",
                  max_tokens: int = 4096, system: Optional[str] = None) -> str:
        """ALL inference goes through AIM. No direct Ollama calls."""
        from .aim import get_aim
        aim = get_aim()
        result = await aim.generate(
            prompt=prompt,
            system=system or self._system_prompt,
            task_type=task_type,
            max_tokens=max_tokens,
        )
        return result["text"]

    @abstractmethod
    async def execute(self, task: dict) -> dict:
        """Execute a task. Returns {success: bool, result: Any}."""
        ...

    async def run_task(self, task: dict) -> dict:
        """Execute + track + circuit break + self-improve."""
        # Circuit breaker
        if self._consecutive_fails >= MAX_CONSECUTIVE_FAILURES:
            log.error(f"[{self.agent_id}] CIRCUIT BREAKER: {self._consecutive_fails} consecutive failures. Skipping.")
            record_metric(f"{self.agent_id}.circuit_break", 1.0, self.agent_id)
            self._consecutive_fails = 0  # Reset after break
            return {"success": False, "error": "Circuit breaker tripped — agent needs review"}

        self._task_count += 1
        start = datetime.now(timezone.utc)
        try:
            result = await self.execute(task)
            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            success = result.get("success", False)

            if success:
                self._success_count += 1
                self._consecutive_fails = 0
            else:
                self._fail_count += 1
                self._consecutive_fails += 1

            record_metric(f"{self.agent_id}.task_success", 1.0 if success else 0.0, self.agent_id)
            record_metric(f"{self.agent_id}.latency_s", elapsed, self.agent_id)

            # Self-improvement with A/B comparison
            if self._task_count % SELF_IMPROVE_EVERY == 0:
                await self._self_improve_with_ab()

            return result
        except Exception as e:
            self._fail_count += 1
            self._consecutive_fails += 1
            record_metric(f"{self.agent_id}.task_error", 1.0, self.agent_id)
            log.error(f"[{self.agent_id}] Task failed: {e}")
            return {"success": False, "error": str(e)}

    async def _self_improve_with_ab(self) -> None:
        """Self-improve with A/B comparison — REVERT if worse (MiniMax M2.7 pattern)."""
        current_rate = self._success_count / max(self._task_count, 1)
        recent = read_memories(agent_id=self.agent_id, category="feedback", limit=10)
        feedback_text = "\n".join(f"- {m['title']}: {m['content'][:100]}" for m in recent)

        # Save current prompt as checkpoint
        checkpoint_prompt = self._system_prompt
        checkpoint_rate = self._best_success_rate

        prompt = f"""You are reviewing your own performance as {self.role}.

Success rate: {current_rate:.1%} ({self._success_count}/{self._task_count})
Previous best rate: {self._best_success_rate:.1%}
Consecutive failures: {self._consecutive_fails}

Recent feedback:
{feedback_text or '(no feedback yet)'}

Write 1-3 SPECIFIC improvements. Each must be:
- Actionable (not "be more careful" — say exactly what to change)
- Measurable (how will you know it worked?)
- Reversible (if it doesn't improve, we revert)
"""
        try:
            improvements = await self.llm(prompt, task_type="quality_audit", max_tokens=500)

            # A/B COMPARISON: only keep if rate improved or is first improvement
            if current_rate >= self._best_success_rate or self._best_success_rate == 0:
                write_memory(
                    agent_id=self.agent_id,
                    category="learnings",
                    title=f"KEPT: improvement cycle {self._task_count} (rate {current_rate:.1%})",
                    content=improvements,
                    confidence=min(0.9, current_rate + 0.1),
                )
                self._system_prompt = self._build_system_prompt()
                self._best_success_rate = current_rate
                log.info(f"[{self.agent_id}] Self-improved: {current_rate:.1%} (kept, beat {checkpoint_rate:.1%})")
            else:
                # REVERT — performance regressed
                self._system_prompt = checkpoint_prompt
                write_memory(
                    agent_id=self.agent_id,
                    category="learnings",
                    title=f"REVERTED: cycle {self._task_count} (rate {current_rate:.1%} < {checkpoint_rate:.1%})",
                    content=f"Reverted because rate dropped from {checkpoint_rate:.1%} to {current_rate:.1%}.\nProposed changes were:\n{improvements}",
                    confidence=0.3,
                )
                log.warning(f"[{self.agent_id}] REVERTED: {current_rate:.1%} < {checkpoint_rate:.1%}")

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
            "best_rate": round(self._best_success_rate, 3),
            "consecutive_fails": self._consecutive_fails,
        }
