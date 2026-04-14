"""CEO Agent — Strategic decision maker for the Atlas Swarm.

Responsibilities:
- Weekly: review IdeaFrog picks, approve/reject for pipeline
- Weekly: review KPI dashboard, flag any DMAIC triggers
- On-demand: allocate swarm resources to highest-RICE opportunities
- Escalate to human on budget >$5K or regulatory questions
"""
import json
import logging
from typing import Any

from .agent_base import SwarmAgent
from .memory import read_memories, record_metric, write_memory

log = logging.getLogger(__name__)


class CEOAgent(SwarmAgent):
    def __init__(self):
        super().__init__(
            agent_id="ceo",
            role="Chief Executive Officer",
            goal="Hit quarterly revenue targets and maintain 40% gross margin",
            backstory=(
                "You oversee the Atlas Swarm product delivery system. "
                "You make final product/market decisions using RICE scoring. "
                "You track KPIs weekly and run DMAIC cycles on any missed target. "
                "You escalate to the human operator only for budget >$5K or regulatory matters."
            ),
        )

    async def execute(self, task: dict) -> dict:
        task_type = task.get("type", "unknown")

        if task_type == "review_opportunities":
            return await self._review_opportunities(task)
        elif task_type == "weekly_review":
            return await self._weekly_review(task)
        elif task_type == "allocate_resources":
            return await self._allocate_resources(task)
        else:
            return await self._general_decision(task)

    async def _review_opportunities(self, task: dict) -> dict:
        """Review IdeaFrog's top picks and decide which to pursue."""
        opportunities = task.get("opportunities", [])
        if not opportunities:
            return {"success": False, "error": "No opportunities provided"}

        opp_text = json.dumps(opportunities[:10], indent=2)
        prompt = f"""Review these product opportunities from IdeaFrog (ranked by RICE score).

OPPORTUNITIES:
{opp_text}

For each, decide: APPROVE, DEFER, or REJECT.
Consider:
1. Can we realistically build and ship this in <7 days?
2. Is the market large enough for >$1K/month revenue?
3. Do we have regulatory risk (medical claims, safety)?
4. Does it leverage our 46M patent knowledge base?

Return JSON: {{"decisions": [{{"id": "...", "decision": "APPROVE|DEFER|REJECT", "rationale": "..."}}]}}
"""
        response = await self.llm(prompt)
        try:
            decisions = json.loads(response)
        except json.JSONDecodeError:
            decisions = {"raw": response}

        # Log decision to memory
        write_memory(
            agent_id=self.agent_id,
            category="decisions",
            title=f"Opportunity review: {len(opportunities)} items",
            content=json.dumps(decisions, indent=2),
            confidence=0.8,
        )
        return {"success": True, "result": decisions}

    async def _weekly_review(self, task: dict) -> dict:
        """Weekly KPI review with DMAIC analysis."""
        # Read recent metrics
        recent_decisions = read_memories(agent_id="ceo", category="decisions", limit=5)
        recent_metrics = read_memories(category="metrics", limit=20)

        context = {
            "recent_decisions": [d["title"] for d in recent_decisions],
            "metrics_count": len(recent_metrics),
        }

        prompt = f"""Conduct your weekly CEO review for the Atlas Swarm.

Context: {json.dumps(context, indent=2)}

Produce:
1. DEFINE: What were this week's targets?
2. MEASURE: What actually happened? (use metrics if available)
3. ANALYZE: Any gaps? Root cause?
4. IMPROVE: What should change next week?
5. CONTROL: What safeguards to put in place?

Also flag any items needing human escalation.
Return structured JSON with these 5 sections.
"""
        response = await self.llm(prompt)
        write_memory(
            agent_id=self.agent_id,
            category="decisions",
            title=f"Weekly DMAIC review",
            content=response,
            confidence=0.7,
        )
        record_metric("ceo.weekly_review_completed", 1.0, self.agent_id)
        return {"success": True, "result": response}

    async def _allocate_resources(self, task: dict) -> dict:
        """Reallocate swarm focus based on performance data."""
        prompt = f"""Based on current swarm performance, recommend resource allocation changes.

Current agents: CEO, CMO, CRO, CTO, CFO, IdeaFrog, Ghost Signal
Current products in pipeline: {task.get('pipeline_count', 'unknown')}
Revenue this week: ${task.get('weekly_revenue', 0)}
Margin: {task.get('margin', 'unknown')}%

Should we:
- Shift more effort to marketing (CMO) or building (CTO)?
- Pause any low-performing product lines?
- Double down on top performers?
- Adjust the IdeaFrog RICE weights?

Return JSON with specific allocation changes.
"""
        response = await self.llm(prompt)
        write_memory(
            agent_id=self.agent_id,
            category="decisions",
            title="Resource allocation adjustment",
            content=response,
            confidence=0.6,
        )
        return {"success": True, "result": response}

    async def _general_decision(self, task: dict) -> dict:
        """Handle any ad-hoc decision request."""
        prompt = f"""You've been asked to make a decision:

{json.dumps(task, indent=2)}

Apply RICE scoring and Lean Six Sigma principles.
Return your decision with clear rationale.
"""
        response = await self.llm(prompt)
        write_memory(
            agent_id=self.agent_id,
            category="decisions",
            title=f"Decision: {task.get('description', 'ad-hoc')[:80]}",
            content=response,
            confidence=0.6,
        )
        return {"success": True, "result": response}
