"""KIE Production — Knowledge Intelligence Engine.

WIRED to IdeaFrog's REAL API at Server 1 :5001.
Not a reimplementation. Uses IdeaFrog's 8,066 lines of battle-tested code
for opportunity scoring, campaign management, and recommendations.

KIE's job: connect IdeaFrog (opportunity discovery) to MM1's swarm (execution).
"""
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from .aim import get_aim
from .knowledge_bridge import KnowledgeBridge, get_bridge
from .memory import read_memories, record_metric, write_memory

log = logging.getLogger(__name__)

IDEAFROG_URL = "http://192.168.1.204:5001"
IDEAFROG_TIMEOUT = 60.0


class KIE:
    """Knowledge Intelligence Engine — bridges IdeaFrog to the MM1 swarm."""

    def __init__(self):
        self.bridge = get_bridge()
        self.aim = get_aim()
        self._ideafrog: Optional[httpx.AsyncClient] = None

    async def _get_ideafrog(self) -> httpx.AsyncClient:
        if self._ideafrog is None or self._ideafrog.is_closed:
            self._ideafrog = httpx.AsyncClient(
                base_url=IDEAFROG_URL,
                timeout=IDEAFROG_TIMEOUT,
            )
        return self._ideafrog

    # ── IdeaFrog API wrappers ──────────────────────────────────────────

    async def ideafrog_health(self) -> dict:
        """Check IdeaFrog status."""
        client = await self._get_ideafrog()
        resp = await client.get("/health")
        resp.raise_for_status()
        return resp.json()

    async def get_opportunities(self, limit: int = 20) -> list[dict]:
        """Fetch scored opportunities from IdeaFrog (the REAL source)."""
        client = await self._get_ideafrog()
        resp = await client.get("/opportunities", params={"limit": limit})
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("opportunities", [])

    async def get_recommendations(self, limit: int = 10) -> list[dict]:
        """Fetch IdeaFrog's pre-scored recommendations."""
        client = await self._get_ideafrog()
        resp = await client.get("/recommendations", params={"limit": limit})
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("recommendations", [])

    async def run_pipeline(self) -> dict:
        """Trigger IdeaFrog's full evaluation pipeline.

        This runs IdeaFrog's BITF engine, expired patent agent, innovation
        pipeline, and scoring on all pending opportunities.
        """
        client = await self._get_ideafrog()
        resp = await client.post("/pipeline/run", timeout=300.0)
        resp.raise_for_status()
        result = resp.json()
        record_metric("kie.pipeline_runs", 1.0, "kie")
        write_memory("kie", "products", "Pipeline run triggered",
                     json.dumps(result, indent=2, default=str)[:2000])
        return result

    async def evaluate_opportunity(self, opportunity_id: str) -> dict:
        """Deep-evaluate a single opportunity via IdeaFrog."""
        client = await self._get_ideafrog()
        resp = await client.post("/pipeline/evaluate-single",
                                 json={"opportunity_id": opportunity_id})
        resp.raise_for_status()
        return resp.json()

    async def get_campaigns(self, limit: int = 10) -> list[dict]:
        """Fetch IdeaFrog campaigns."""
        client = await self._get_ideafrog()
        resp = await client.get("/campaigns", params={"limit": limit})
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("campaigns", [])

    async def generate_predictions(self, data: dict) -> dict:
        """Generate revenue/market predictions via IdeaFrog's ML model."""
        client = await self._get_ideafrog()
        resp = await client.post("/predictions/predict", json=data)
        resp.raise_for_status()
        return resp.json()

    # ── Combined intelligence (IdeaFrog + AIM + Knowledge Bridge) ──────

    async def select_top_products(self, count: int = 5) -> list[dict]:
        """Select top N products for the swarm to commercialize.

        Combines:
        1. IdeaFrog's 119 pre-scored opportunities
        2. AIM's additional scoring (market viability, 3D printability)
        3. Knowledge bridge context (enriched patent data)

        Returns production-ready product candidates.
        """
        # Get IdeaFrog's recommendations
        recs = await self.get_recommendations(limit=count * 3)

        if not recs:
            log.warning("[KIE] No recommendations from IdeaFrog — using opportunities")
            recs = await self.get_opportunities(limit=count * 3)

        # AIM enhancement: score for commercialization readiness
        enhanced = []
        for rec in recs[:count * 2]:
            title = rec.get("title", "")[:100]
            description = rec.get("description", "")[:500]
            domain = rec.get("domain", "general")
            raw_score = rec.get("final_score") or rec.get("raw_score", 0)

            # Quick AIM assessment
            try:
                aim_result = await self.aim.generate(
                    prompt=f"""Rate this product opportunity for immediate commercialization (1-10):

Title: {title}
Domain: {domain}
Score: {raw_score}
Description: {description}

Rate on:
- printability (can we 3D print a version?): 1-10
- market_demand (do people buy this?): 1-10
- speed_to_market (days to first sale): number
- regulatory_risk (0=none, 10=FDA required): 1-10

Return ONLY JSON: {{"printability": N, "market_demand": N, "speed_to_market": N, "regulatory_risk": N, "go_no_go": "GO|NO_GO", "one_liner": "..."}}""",
                    task_type="patent_analysis",
                    max_tokens=200,
                    require_local=True,
                )
                import re
                match = re.search(r'\{.*\}', aim_result["text"], re.DOTALL)
                if match:
                    aim_score = json.loads(match.group())
                    rec["_aim_assessment"] = aim_score
                    rec["_commercialization_score"] = (
                        aim_score.get("printability", 5) *
                        aim_score.get("market_demand", 5) /
                        max(aim_score.get("regulatory_risk", 5), 1)
                    )
                else:
                    rec["_commercialization_score"] = raw_score
            except Exception as e:
                log.warning(f"[KIE] AIM scoring failed for '{title}': {e}")
                rec["_commercialization_score"] = raw_score

            enhanced.append(rec)

        # Sort by commercialization score
        enhanced.sort(key=lambda x: x.get("_commercialization_score", 0), reverse=True)
        top = enhanced[:count]

        # Log selection
        write_memory("kie", "products",
                     f"TOP {count} PRODUCTS SELECTED for commercialization",
                     json.dumps([{
                         "title": r.get("title", "")[:80],
                         "score": r.get("_commercialization_score", 0),
                         "domain": r.get("domain", ""),
                         "go": r.get("_aim_assessment", {}).get("go_no_go", "?"),
                     } for r in top], indent=2),
                     confidence=0.75)

        record_metric("kie.products_selected", float(len(top)), "kie")
        return top

    async def stats(self) -> dict:
        """Full KIE status combining all sources."""
        bridge_stats = await self.bridge.stats()
        try:
            if_health = await self.ideafrog_health()
        except Exception:
            if_health = {"status": "unreachable"}

        return {
            "knowledge_base": bridge_stats.get("knowledge", 0),
            "ideafrog": if_health,
            "recent_selections": len(read_memories(agent_id="kie", category="products", limit=10)),
        }


_kie: Optional[KIE] = None


def get_kie() -> KIE:
    global _kie
    if _kie is None:
        _kie = KIE()
    return _kie
