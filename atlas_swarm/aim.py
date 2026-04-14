"""AIM v1 — Atlas Intelligence Model.

Multi-model router with ML-driven selection, persistent learning,
and self-evolution scaffolding. Routes tasks to the optimal model
based on learned performance across the full model stack:

    Tier 1: Gemma 4 26B MoE (local, fast, multimodal)
    Tier 2: Qwen3 30B-A3B (local, fastest MoE)
    Tier 3: Gemma3 27B (local, fallback)
    Tier 4: GLM-5.1 (API, heavyweight agentic coding)
    Tier 5: Claude Opus/Sonnet (API, best reasoning)

The Q-learning router tracks success/failure/latency/cost for each
model × task-type pair and automatically routes to the best performer.
Self-evolution: every 50 tasks, AIM reviews its routing decisions
and adjusts exploration rate and model preferences.

This is NOT a new model. It's an intelligent orchestration layer
that makes the existing models work together as one system.
"""
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import httpx

from .memory import read_memories, record_metric, write_memory
from .router import QLearningRouter

log = logging.getLogger(__name__)


class ModelTier(str, Enum):
    GEMMA4_26B = "gemma4:26b"
    QWEN3_30B = "qwen3:30b-a3b"
    GEMMA3_27B = "gemma3:27b"
    GLM_5_1 = "glm-5.1"
    CLAUDE_SONNET = "claude-sonnet-4-6"
    CLAUDE_OPUS = "claude-opus-4-6"


# Model capabilities matrix — used by the router to pre-filter
MODEL_CAPS = {
    ModelTier.GEMMA4_26B: {
        "local": True, "multimodal": True, "context": 256_000,
        "strengths": ["reasoning", "content", "vision", "audio"],
        "cost_per_1k": 0.0, "avg_tok_s": 25,
    },
    ModelTier.QWEN3_30B: {
        "local": True, "multimodal": False, "context": 32_768,
        "strengths": ["speed", "code", "structured_output"],
        "cost_per_1k": 0.0, "avg_tok_s": 60,
    },
    ModelTier.GEMMA3_27B: {
        "local": True, "multimodal": False, "context": 128_000,
        "strengths": ["reasoning", "general"],
        "cost_per_1k": 0.0, "avg_tok_s": 13,
    },
    ModelTier.GLM_5_1: {
        "local": False, "multimodal": False, "context": 200_000,
        "strengths": ["coding", "agentic", "long_horizon"],
        "cost_per_1k": 0.003, "avg_tok_s": 80,
    },
    ModelTier.CLAUDE_SONNET: {
        "local": False, "multimodal": True, "context": 200_000,
        "strengths": ["reasoning", "coding", "safety"],
        "cost_per_1k": 0.003, "avg_tok_s": 100,
    },
    ModelTier.CLAUDE_OPUS: {
        "local": False, "multimodal": True, "context": 1_000_000,
        "strengths": ["reasoning", "coding", "research", "safety"],
        "cost_per_1k": 0.015, "avg_tok_s": 60,
    },
}

# Task type → preferred model strengths
TASK_ROUTING_HINTS = {
    "content_generation": ["content", "reasoning"],
    "listing_creation": ["content", "structured_output"],
    "product_design": ["reasoning", "vision"],
    "code_generation": ["coding", "agentic"],
    "patent_analysis": ["reasoning", "long_horizon"],
    "image_analysis": ["vision", "multimodal"],
    "financial_analysis": ["reasoning", "structured_output"],
    "social_media": ["content", "speed"],
    "quality_audit": ["reasoning", "safety"],
    "strategic_decision": ["reasoning"],
    "general": ["reasoning", "general"],
}

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GLM_API_KEY = os.environ.get("GLM_API_KEY", "")
SELF_EVOLVE_EVERY = 50


class AIM:
    """Atlas Intelligence Model — multi-model router with ML learning."""

    def __init__(self):
        self.router = QLearningRouter(
            exploration_rate=0.12,
            learning_rate=0.15,
            discount_factor=0.9,
        )
        self._call_count = 0
        self._total_cost = 0.0
        self._model_stats: dict[str, dict] = {
            m.value: {"calls": 0, "successes": 0, "failures": 0,
                      "total_latency": 0.0, "total_tokens": 0, "total_cost": 0.0}
            for m in ModelTier
        }

    def select_model(self, task_type: str, require_local: bool = False,
                     require_multimodal: bool = False,
                     min_context: int = 0) -> ModelTier:
        """Select the best model for a task using Q-learning + capability filtering."""
        # Filter by hard requirements
        candidates = []
        for model, caps in MODEL_CAPS.items():
            if require_local and not caps["local"]:
                continue
            if require_multimodal and not caps["multimodal"]:
                continue
            if caps["context"] < min_context:
                continue
            # Don't route to API models if no key configured
            if model == ModelTier.GLM_5_1 and not GLM_API_KEY:
                continue
            if model in (ModelTier.CLAUDE_SONNET, ModelTier.CLAUDE_OPUS) and not ANTHROPIC_KEY:
                continue
            candidates.append(model.value)

        if not candidates:
            # Fallback to any local model
            candidates = [m.value for m in ModelTier if MODEL_CAPS[m]["local"]]

        # Use Q-learning router to pick from candidates
        selected = self.router.select_agent(task_type, candidates)
        return ModelTier(selected)

    async def generate(
        self,
        prompt: str,
        system: str = "",
        task_type: str = "general",
        max_tokens: int = 4096,
        require_local: bool = False,
        require_multimodal: bool = False,
        min_context: int = 0,
        preferred_model: Optional[ModelTier] = None,
    ) -> dict:
        """Generate text using the best available model.

        Returns: {
            "text": str,
            "model": str,
            "latency_s": float,
            "tokens": int,
            "cost": float,
            "task_type": str,
        }
        """
        # PRODUCTION RULE: Gemma4 and Qwen3 30B use thinking mode by default,
        # which breaks JSON extraction. Route structured-output tasks to Gemma3 27B
        # (no thinking mode, clean JSON in content field, 13 tok/s, reliable).
        # Gemma4 handles free-form reasoning/content. Qwen3 handles speed tasks.
        needs_json = 'JSON' in prompt or 'json' in prompt or 'Return:' in prompt
        if needs_json and preferred_model is None:
            preferred_model = ModelTier.GEMMA3_27B

        model = preferred_model or self.select_model(
            task_type, require_local, require_multimodal, min_context
        )

        self._call_count += 1
        start = time.monotonic()
        text = ""
        tokens = 0
        error = None

        try:
            if model in (ModelTier.GEMMA4_26B, ModelTier.QWEN3_30B, ModelTier.GEMMA3_27B):
                text, tokens = await self._ollama_generate(model.value, system, prompt, max_tokens)
            elif model == ModelTier.GLM_5_1:
                text, tokens = await self._glm_generate(system, prompt, max_tokens)
            elif model in (ModelTier.CLAUDE_SONNET, ModelTier.CLAUDE_OPUS):
                text, tokens = await self._anthropic_generate(model.value, system, prompt, max_tokens)
            else:
                raise ValueError(f"Unknown model: {model}")
        except Exception as e:
            error = e
            log.warning(f"[AIM] {model.value} failed: {e}")
            # Fallback chain: try next local model
            for fallback in [ModelTier.GEMMA4_26B, ModelTier.GEMMA3_27B, ModelTier.QWEN3_30B]:
                if fallback != model:
                    try:
                        text, tokens = await self._ollama_generate(
                            fallback.value, system, prompt, max_tokens
                        )
                        model = fallback
                        error = None
                        break
                    except Exception:
                        continue
            if error:
                raise error

        latency = time.monotonic() - start
        cost = (tokens / 1000) * MODEL_CAPS[model]["cost_per_1k"]

        # Update ML stats
        stats = self._model_stats[model.value]
        stats["calls"] += 1
        stats["successes"] += 1
        stats["total_latency"] += latency
        stats["total_tokens"] += tokens
        stats["total_cost"] += cost
        self._total_cost += cost

        # Update Q-learning router
        reward = 1.0 - (latency / 30.0) - (cost * 10)  # Fast + cheap = high reward
        self.router.update(task_type, model.value, max(reward, -1.0))

        # Record metric
        record_metric(f"aim.{model.value}.latency", latency, "aim")
        record_metric(f"aim.{model.value}.tokens", float(tokens), "aim")

        # Self-evolution check
        if self._call_count % SELF_EVOLVE_EVERY == 0:
            await self._self_evolve()

        result = {
            "text": text,
            "model": model.value,
            "latency_s": round(latency, 2),
            "tokens": tokens,
            "cost": round(cost, 6),
            "task_type": task_type,
        }
        log.info(f"[AIM] model={model.value} task={task_type} tokens={tokens} latency={latency:.1f}s cost=${cost:.4f}")
        return result

    async def _ollama_generate(self, model: str, system: str, prompt: str,
                                max_tokens: int) -> tuple[str, int]:
        async with httpx.AsyncClient(timeout=180.0) as client:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": model, "messages": messages, "stream": False,
                      "options": {"num_predict": max_tokens}},
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data["message"]
            text = msg.get("content", "")
            # Gemma4 26B MoE uses reasoning mode — response may be in 'thinking' field
            if not text and msg.get("thinking"):
                text = msg["thinking"]
            tokens = data.get("eval_count", len(text) // 4)
            return text, tokens

    async def _anthropic_generate(self, model: str, system: str, prompt: str,
                                   max_tokens: int) -> tuple[str, int]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": system or "You are a helpful assistant.",
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["content"][0]["text"]
            tokens = data.get("usage", {}).get("output_tokens", len(text) // 4)
            return text, tokens

    async def _glm_generate(self, system: str, prompt: str,
                             max_tokens: int) -> tuple[str, int]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                headers={
                    "Authorization": f"Bearer {GLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "glm-5.1",
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system or "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("completion_tokens", len(text) // 4)
            return text, tokens

    async def _self_evolve(self) -> None:
        """Self-evolution: review routing performance and adjust strategy."""
        log.info(f"[AIM] Self-evolution triggered after {self._call_count} calls")

        # Analyze model performance
        analysis = {}
        for model, stats in self._model_stats.items():
            if stats["calls"] > 0:
                analysis[model] = {
                    "calls": stats["calls"],
                    "avg_latency": round(stats["total_latency"] / stats["calls"], 2),
                    "avg_tokens": round(stats["total_tokens"] / stats["calls"]),
                    "total_cost": round(stats["total_cost"], 4),
                    "success_rate": round(stats["successes"] / stats["calls"], 3),
                }

        # Adjust exploration rate based on performance stability
        if self._call_count > 200:
            self.router.epsilon = max(0.05, self.router.epsilon * 0.95)  # Decay exploration
        elif self._call_count > 100:
            self.router.epsilon = max(0.08, self.router.epsilon * 0.97)

        # Log evolution to memory
        write_memory(
            agent_id="aim",
            category="learnings",
            title=f"Self-evolution cycle {self._call_count}",
            content=json.dumps({
                "call_count": self._call_count,
                "total_cost": round(self._total_cost, 4),
                "exploration_rate": self.router.epsilon,
                "model_analysis": analysis,
                "router_stats": self.router.get_stats()[:10],
            }, indent=2),
            confidence=0.8,
        )
        log.info(f"[AIM] Evolution complete. Exploration rate: {self.router.epsilon:.3f}, Total cost: ${self._total_cost:.4f}")

    @property
    def status(self) -> dict:
        return {
            "version": "v1",
            "total_calls": self._call_count,
            "total_cost": round(self._total_cost, 4),
            "exploration_rate": self.router.epsilon,
            "models": {
                model: {
                    "calls": stats["calls"],
                    "avg_latency": round(stats["total_latency"] / max(stats["calls"], 1), 2),
                    "success_rate": round(stats["successes"] / max(stats["calls"], 1), 3),
                    "total_cost": round(stats["total_cost"], 4),
                }
                for model, stats in self._model_stats.items()
                if stats["calls"] > 0
            },
        }


# Singleton
_aim: Optional[AIM] = None


def get_aim() -> AIM:
    global _aim
    if _aim is None:
        _aim = AIM()
    return _aim
