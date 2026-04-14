"""KIE — Knowledge Intelligence Engine.

Bridges Server 1's 47M+ knowledge records into actionable product opportunities.
Combines the existing knowledge_bridge (HTTP API) with AIM's intelligence layer
to score, rank, and pipeline opportunities through the company swarm.

Pipeline: Knowledge DB → IdeaFrog Scout → RICE Scorer → CEO Approval → Product Pipeline

Lean Six Sigma: KIE IS the Define + Measure step.
- Define: What opportunities exist in our knowledge base?
- Measure: Score them by RICE (Reach × Impact × Confidence / Effort)
"""
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from .aim import AIM, ModelTier, get_aim
from .knowledge_bridge import KnowledgeBridge, get_bridge
from .memory import read_memories, record_metric, write_memory

log = logging.getLogger(__name__)


class KIE:
    """Knowledge Intelligence Engine — turns 47M records into product opportunities."""

    def __init__(self):
        self.bridge = get_bridge()
        self.aim = get_aim()

    async def scan_opportunities(
        self,
        verticals: Optional[list[str]] = None,
        limit_per_vertical: int = 20,
    ) -> list[dict]:
        """Scan knowledge base for product opportunities across verticals.

        Default verticals target the highest-RICE categories from research:
        1. Anatomical study models (surgical training)
        2. Ergonomic grips (arthritis, OT market)
        3. Parametric desk accessories
        4. Replacement parts for obsolete goods
        5. Custom medical splints (Class I exempt)
        """
        if verticals is None:
            verticals = [
                "anatomical model surgical training",
                "ergonomic grip arthritis occupational therapy",
                "desk organizer cable management parametric",
                "replacement part obsolete consumer goods",
                "orthotic splint rehabilitation",
                "prosthetic socket custom fit",
                "3D printed medical device",
                "assistive technology elderly",
            ]

        all_opportunities = []
        for vertical in verticals:
            try:
                results = await self.bridge.search(vertical, limit=limit_per_vertical)
                for r in results:
                    r["_vertical"] = vertical
                    r["_source_query"] = vertical
                all_opportunities.extend(results)
            except Exception as e:
                log.warning(f"[KIE] Search failed for '{vertical}': {e}")

        log.info(f"[KIE] Scanned {len(verticals)} verticals, found {len(all_opportunities)} raw opportunities")
        return all_opportunities

    async def score_with_aim(self, opportunities: list[dict], top_n: int = 10) -> list[dict]:
        """Use AIM to intelligently score and rank opportunities.

        For each opportunity, AIM evaluates:
        - Market viability (is there demand?)
        - Manufacturing feasibility (can we 3D print it?)
        - Regulatory risk (FDA implications?)
        - Competitive landscape (who else is selling this?)
        - Revenue potential (price × volume estimate)
        """
        if not opportunities:
            return []

        # Batch opportunities into groups of 5 for efficiency
        scored = []
        for i in range(0, len(opportunities), 5):
            batch = opportunities[i:i+5]
            batch_text = json.dumps([
                {
                    "title": r.get("title", "")[:200],
                    "abstract": (r.get("text") or r.get("abstract") or "")[:300],
                    "source": r.get("source", ""),
                    "vertical": r.get("_vertical", ""),
                    "score": r.get("fused_score") or r.get("score", 0),
                }
                for r in batch
            ], indent=2)

            prompt = f"""Score these {len(batch)} product opportunities using RICE framework.

OPPORTUNITIES:
{batch_text}

For EACH opportunity, provide:
1. reach (1-10): How many potential customers?
2. impact (1-10): How much value per customer?
3. confidence (1-10): How sure are we this works?
4. effort (1-10): How hard to build and ship? (1=easy, 10=hard)
5. rice_score: (reach × impact × confidence) / effort
6. verdict: BUILD, EXPLORE, or SKIP
7. rationale: One sentence why

SCORING GUIDE:
- Expired patents with clear physical product = high confidence
- Medical devices requiring FDA = high effort
- Consumer goods < $50 = high reach
- B2B specialty > $100 = high impact

Return JSON array: [{{"title": "...", "reach": N, "impact": N, "confidence": N, "effort": N, "rice_score": N, "verdict": "BUILD|EXPLORE|SKIP", "rationale": "..."}}]
"""
            try:
                result = await self.aim.generate(
                    prompt=prompt,
                    system="You are a product strategist scoring opportunities for a 3D printing company targeting $10M annual revenue at 40% margin.",
                    task_type="patent_analysis",
                    max_tokens=2000,
                    require_local=True,
                )
                # Parse response
                text = result["text"]
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\[.*\]', text, re.DOTALL)
                if json_match:
                    scores = json.loads(json_match.group())
                    for j, score in enumerate(scores):
                        if i + j < len(opportunities):
                            opportunities[i + j]["_aim_score"] = score
                            opportunities[i + j]["_rice_total"] = score.get("rice_score", 0)
                            opportunities[i + j]["_verdict"] = score.get("verdict", "SKIP")
                            scored.append(opportunities[i + j])
            except Exception as e:
                log.warning(f"[KIE] AIM scoring failed for batch {i//5}: {e}")
                # Still include unscorable items
                for r in batch:
                    r["_aim_score"] = {"error": str(e)}
                    r["_rice_total"] = 0
                    r["_verdict"] = "EXPLORE"
                    scored.append(r)

        # Sort by RICE score
        scored.sort(key=lambda x: x.get("_rice_total", 0), reverse=True)

        # Log top picks to memory
        top_picks = scored[:top_n]
        write_memory(
            agent_id="kie",
            category="products",
            title=f"Opportunity scan: {len(scored)} scored, top {len(top_picks)} selected",
            content=json.dumps([
                {
                    "title": r.get("title", "")[:100],
                    "rice": r.get("_rice_total", 0),
                    "verdict": r.get("_verdict", "?"),
                    "vertical": r.get("_vertical", ""),
                }
                for r in top_picks
            ], indent=2),
            confidence=0.7,
        )

        record_metric("kie.opportunities_scored", float(len(scored)), "kie")
        record_metric("kie.top_rice_score", float(top_picks[0].get("_rice_total", 0)) if top_picks else 0.0, "kie")

        return top_picks

    async def generate_product_brief(self, opportunity: dict) -> dict:
        """Generate a complete product brief from an opportunity.

        This is the handoff document from KIE → CTO-Agent for design.
        """
        title = opportunity.get("title", "Unknown")
        abstract = opportunity.get("text") or opportunity.get("abstract") or ""
        aim_score = opportunity.get("_aim_score", {})

        prompt = f"""Generate a complete product brief for manufacturing.

SOURCE PATENT/PAPER:
Title: {title}
Abstract: {abstract[:1000]}
RICE Score: {json.dumps(aim_score, indent=2)}

PRODUCE:
1. product_name: Catchy, marketplace-ready name
2. description: 2-3 sentence product description for listing
3. target_customer: Who buys this?
4. price_range: Recommended retail price range
5. bill_of_materials: What to 3D print (material, infill, layer height)
6. estimated_cost: COGS estimate (material + print time + post-processing)
7. margin_estimate: Expected gross margin %
8. marketplace: Where to sell (Etsy, Amazon, both?)
9. regulatory_notes: Any FDA/compliance considerations
10. design_prompt: Instructions for CAD generation (OpenSCAD or parametric)
11. photo_requirements: What product photos are needed
12. video_concept: 15-second video script idea

Return as JSON object.
"""
        result = await self.aim.generate(
            prompt=prompt,
            system="You are a product development lead at a 3D printing startup. Be specific about materials, dimensions, and costs.",
            task_type="product_design",
            max_tokens=2000,
            require_local=True,
        )

        brief = {"raw_response": result["text"], "source_opportunity": title}
        try:
            import re
            json_match = re.search(r'\{.*\}', result["text"], re.DOTALL)
            if json_match:
                brief = {**json.loads(json_match.group()), "source_opportunity": title}
        except (json.JSONDecodeError, AttributeError):
            pass

        write_memory(
            agent_id="kie",
            category="products",
            title=f"Brief: {brief.get('product_name', title)[:60]}",
            content=json.dumps(brief, indent=2, default=str),
            confidence=0.6,
        )
        return brief

    async def weekly_pipeline_report(self) -> dict:
        """Generate weekly KIE pipeline report for CEO-Agent."""
        stats = await self.bridge.stats()
        recent_scans = read_memories(agent_id="kie", category="products", limit=20)

        return {
            "knowledge_base_size": stats.get("knowledge", 0),
            "recent_opportunities_scored": len(recent_scans),
            "pipeline_summary": [
                {"title": m["title"][:80], "confidence": m["confidence"]}
                for m in recent_scans[:10]
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


# Singleton
_kie: Optional[KIE] = None


def get_kie() -> KIE:
    global _kie
    if _kie is None:
        _kie = KIE()
    return _kie
