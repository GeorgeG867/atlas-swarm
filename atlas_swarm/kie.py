"""KIE Production — Knowledge Intelligence Engine.

WIRED to IdeaFrog's REAL API at Server 1 :5001.
Uses subprocess curl for HTTP calls (macOS launchd restricts Python sockets).

KIE's job: connect IdeaFrog (opportunity discovery) to MM1's swarm (execution).
"""
import json
import logging
import subprocess
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import quote

from .aim import get_aim
from .knowledge_bridge import KnowledgeBridge, get_bridge
from .memory import read_memories, record_metric, write_memory
from .printability import filter_printable, printability_score

log = logging.getLogger(__name__)

IDEAFROG_URL = "http://192.168.1.204:5001"
IDEAFROG_TIMEOUT = 60.0


def _curl_get(url: str, timeout: float = IDEAFROG_TIMEOUT) -> dict | list | None:
    """GET via curl subprocess (works in macOS launchd)."""
    try:
        result = subprocess.run(
            ["curl", "-s", "--connect-timeout", str(int(timeout)),
             "--max-time", str(int(timeout)), url],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        return None
    except Exception as e:
        log.warning(f"curl GET failed for {url}: {e}")
        return None


def _curl_post(url: str, data: dict | None = None, timeout: float = IDEAFROG_TIMEOUT) -> dict | list | None:
    """POST via curl subprocess (works in macOS launchd)."""
    try:
        cmd = ["curl", "-s", "--connect-timeout", str(int(timeout)),
               "--max-time", str(int(timeout)), "-X", "POST"]
        if data is not None:
            cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(data)]
        cmd.append(url)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        return None
    except Exception as e:
        log.warning(f"curl POST failed for {url}: {e}")
        return None


class KIE:
    """Knowledge Intelligence Engine — bridges IdeaFrog to the MM1 swarm."""

    def __init__(self):
        self.bridge = get_bridge()
        self.aim = get_aim()

    # -- IdeaFrog API wrappers (curl-based) --------------------------------

    async def ideafrog_health(self) -> dict:
        """Check IdeaFrog status."""
        data = _curl_get(f"{IDEAFROG_URL}/health", timeout=10.0)
        if data is None:
            return {"status": "unreachable"}
        return data

    async def get_opportunities(self, limit: int = 20) -> list[dict]:
        """Fetch scored opportunities from IdeaFrog."""
        data = _curl_get(f"{IDEAFROG_URL}/opportunities?limit={limit}")
        if data is None:
            return []
        return data if isinstance(data, list) else data.get("opportunities", [])

    async def get_recommendations(self, limit: int = 10) -> list[dict]:
        """Fetch IdeaFrog's pre-scored recommendations."""
        data = _curl_get(f"{IDEAFROG_URL}/recommendations?limit={limit}")
        if data is None:
            return []
        return data if isinstance(data, list) else data.get("recommendations", [])

    async def run_pipeline(self) -> dict:
        """Trigger IdeaFrog's full evaluation pipeline."""
        data = _curl_post(f"{IDEAFROG_URL}/pipeline/run", timeout=300.0)
        if data is None:
            return {"error": "Pipeline request failed"}
        record_metric("kie.pipeline_runs", 1.0, "kie")
        write_memory("kie", "products", "Pipeline run triggered",
                     json.dumps(data, indent=2, default=str)[:2000])
        return data

    async def evaluate_opportunity(self, opportunity_id: str) -> dict:
        """Deep-evaluate a single opportunity via IdeaFrog."""
        data = _curl_post(f"{IDEAFROG_URL}/pipeline/evaluate-single",
                          data={"opportunity_id": opportunity_id})
        return data or {"error": "Evaluation failed"}

    async def get_campaigns(self, limit: int = 10) -> list[dict]:
        """Fetch IdeaFrog campaigns."""
        data = _curl_get(f"{IDEAFROG_URL}/campaigns?limit={limit}")
        if data is None:
            return []
        return data if isinstance(data, list) else data.get("campaigns", [])

    async def generate_predictions(self, data: dict) -> dict:
        """Generate predictions via IdeaFrog's ML model."""
        result = _curl_post(f"{IDEAFROG_URL}/predictions/predict", data=data)
        return result or {"error": "Prediction failed"}

    # -- Combined intelligence (IdeaFrog + AIM + Knowledge Bridge) ----------

    async def select_top_products(
        self,
        count: int = 5,
        printable_only: bool = True,
        min_printability: float = 1.5,
    ) -> list[dict]:
        """Select top N products for the swarm to commercialize.

        When ``printable_only`` (the default), the hard printability gate runs
        FIRST on a widened pool from IdeaFrog, dropping industrial/medical/
        robotic entries before any AIM spend.  The commercialization score is
        then multiplied by the printability score so simple consumer items win.
        """
        # Pull a wider pool when filtering — the top-N by raw swarm_score is
        # dominated by heavy-equipment patents that IdeaFrog pre-scored high.
        pool_size = count * 12 if printable_only else count * 3
        recs = await self.get_recommendations(limit=pool_size)
        if not recs:
            log.warning("[KIE] No recommendations from IdeaFrog -- using opportunities")
            recs = await self.get_opportunities(limit=pool_size)

        if printable_only:
            before = len(recs)
            recs = filter_printable(recs, min_score=min_printability)
            log.info("[KIE] printability gate: %d -> %d recs", before, len(recs))
            if not recs:
                log.warning("[KIE] Printability gate rejected every opportunity — falling back to top raw pool")
                recs = await self.get_opportunities(limit=pool_size)
                recs = filter_printable(recs, min_score=min_printability) or recs

        enhanced = []
        for rec in recs[:count * 2]:
            title = rec.get("title", "")[:100]
            description = rec.get("description", "")[:500]
            domain = rec.get("domain", "general")
            raw_score = rec.get("final_score") or rec.get("raw_score", 0)

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

            # Multiply by hard-gate score so simpler items always beat marginal ones.
            prn = rec.get("_printability_score")
            if prn is None:
                prn, _ = printability_score(rec)
                rec["_printability_score"] = prn
            rec["_commercialization_score"] = rec["_commercialization_score"] * max(prn, 0.1)

            enhanced.append(rec)

        enhanced.sort(key=lambda x: x.get("_commercialization_score", 0), reverse=True)
        top = enhanced[:count]

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
