"""Executive agents: CMO, CRO, CTO, CFO for the Atlas Swarm.

Each follows the same SwarmAgent pattern: DMAIC, self-improvement,
persistent memory, local+cloud LLM inference.
"""
import json
import logging
from typing import Any

from .agent_base import SwarmAgent
from .memory import read_memories, record_metric, write_memory

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CMO — Chief Marketing Officer
# ═══════════════════════════════════════════════════════════════════════════

class CMOAgent(SwarmAgent):
    """Owns brand, content, SEO, and social. Drives organic demand."""

    def __init__(self):
        super().__init__(
            agent_id="cmo",
            role="Chief Marketing Officer",
            goal="Drive organic demand through content and SEO; 10K monthly impressions by Q2",
            backstory=(
                "You own all marketing for the Atlas Swarm product portfolio. "
                "You create product descriptions, blog posts, social media content, "
                "and SEO-optimized listings. You measure impressions, engagement, "
                "and marketing-attributed revenue. You command content, SEO, and social sub-agents."
            ),
        )

    async def execute(self, task: dict) -> dict:
        task_type = task.get("type", "content")

        if task_type == "generate_listing_content":
            return await self._listing_content(task)
        elif task_type == "social_post":
            return await self._social_post(task)
        elif task_type == "seo_optimize":
            return await self._seo_optimize(task)
        else:
            return await self._general_marketing(task)

    async def _listing_content(self, task: dict) -> dict:
        """Generate marketplace listing content for a product."""
        product = task.get("product", {})
        prompt = f"""Create a complete marketplace listing for this product:

PRODUCT: {json.dumps(product, indent=2)}

Generate:
1. Title (60 chars max, keyword-rich)
2. 5 bullet points (benefits-focused, not features)
3. Description (150 words, SEO-optimized)
4. 5 search tags/keywords
5. Price recommendation with rationale

RULES:
- NO health/medical claims unless product is FDA-cleared
- NO superlatives ("best", "guaranteed") — violates marketplace policy
- Include specific dimensions, materials, and use cases
- Target Etsy and Amazon audiences

Return JSON with keys: title, bullets, description, tags, price_recommendation
"""
        response = await self.llm(prompt)
        write_memory(self.agent_id, "products", f"Listing: {product.get('name', 'unnamed')}", response)
        try:
            return {"success": True, "result": json.loads(response)}
        except json.JSONDecodeError:
            return {"success": True, "result": {"raw": response}}

    async def _social_post(self, task: dict) -> dict:
        """Generate social media posts for a product."""
        product = task.get("product", {})
        platforms = task.get("platforms", ["twitter", "linkedin", "tiktok"])
        prompt = f"""Create social media posts for this product across {platforms}:

PRODUCT: {json.dumps(product, indent=2)}

For each platform, provide:
- Post text (platform-appropriate length and tone)
- Hashtags (5-8 relevant ones)
- Call to action
- Best posting time recommendation

Return JSON with platform names as keys.
"""
        response = await self.llm(prompt)
        return {"success": True, "result": response}

    async def _seo_optimize(self, task: dict) -> dict:
        """SEO-optimize existing content."""
        content = task.get("content", "")
        keywords = task.get("target_keywords", [])
        prompt = f"""Optimize this content for SEO targeting keywords: {keywords}

CONTENT:
{content[:2000]}

Return the optimized version with:
1. Keyword density 1-2%
2. Header tags (H1, H2, H3)
3. Meta description (155 chars)
4. Internal linking suggestions
"""
        response = await self.llm(prompt)
        return {"success": True, "result": response}

    async def _general_marketing(self, task: dict) -> dict:
        prompt = f"Marketing task: {json.dumps(task, indent=2)}\n\nProvide actionable recommendations."
        response = await self.llm(prompt)
        return {"success": True, "result": response}


# ═══════════════════════════════════════════════════════════════════════════
# CRO — Chief Revenue Officer
# ═══════════════════════════════════════════════════════════════════════════

class CROAgent(SwarmAgent):
    """Owns marketplace listings, pricing, and customer success."""

    def __init__(self):
        super().__init__(
            agent_id="cro",
            role="Chief Revenue Officer",
            goal="Maximize listing conversion rate >2.5% and marketplace revenue",
            backstory=(
                "You own marketplace presence on Etsy, Amazon, and Alibaba. "
                "You manage product listings, pricing strategy, and customer success. "
                "You track conversion rate, average order value, and return rate. "
                "You command listing, pricing, and customer-success sub-agents."
            ),
        )

    async def execute(self, task: dict) -> dict:
        task_type = task.get("type", "listing")

        if task_type == "publish_listing":
            return await self._publish_listing(task)
        elif task_type == "optimize_pricing":
            return await self._optimize_pricing(task)
        elif task_type == "revenue_report":
            return await self._revenue_report(task)
        else:
            prompt = f"Revenue task: {json.dumps(task, indent=2)}\n\nProvide actionable steps."
            response = await self.llm(prompt)
            return {"success": True, "result": response}

    async def _publish_listing(self, task: dict) -> dict:
        """Prepare and validate a listing for marketplace submission."""
        listing = task.get("listing", {})
        marketplace = task.get("marketplace", "etsy")

        prompt = f"""Review this listing for {marketplace} marketplace compliance:

LISTING: {json.dumps(listing, indent=2)}

Check for:
1. Title length within platform limits
2. No prohibited claims (health, safety, "best")
3. Price is competitive for this category
4. Images described adequately
5. Tags/keywords are relevant

Return JSON: {{
    "approved": true/false,
    "issues": ["..."],
    "optimizations": ["..."],
    "final_listing": {{...}}
}}
"""
        response = await self.llm(prompt)
        write_memory(self.agent_id, "products", f"Listing review: {marketplace}", response)
        record_metric("cro.listings_reviewed", 1.0, self.agent_id)
        try:
            return {"success": True, "result": json.loads(response)}
        except json.JSONDecodeError:
            return {"success": True, "result": {"raw": response}}

    async def _optimize_pricing(self, task: dict) -> dict:
        """Analyze and recommend pricing adjustments."""
        products = task.get("products", [])
        prompt = f"""Optimize pricing for these products:

{json.dumps(products[:10], indent=2)}

Consider:
1. Target 40% gross margin minimum
2. Competitor pricing (estimate from product category)
3. Platform fees (Etsy: 6.5%, Amazon: 15%)
4. Shipping costs
5. Price psychology ($X.99 vs round numbers)

Return JSON array with product_id, current_price, recommended_price, rationale.
"""
        response = await self.llm(prompt)
        return {"success": True, "result": response}

    async def _revenue_report(self, task: dict) -> dict:
        """Generate weekly revenue report."""
        prompt = """Generate a revenue status report based on available metrics.
Include: total revenue, units sold, conversion rate, top products, margin analysis.
Flag any products below 35% margin for CEO review."""
        response = await self.llm(prompt)
        write_memory(self.agent_id, "metrics", "Weekly revenue report", response)
        return {"success": True, "result": response}


# ═══════════════════════════════════════════════════════════════════════════
# CTO — Chief Technology Officer
# ═══════════════════════════════════════════════════════════════════════════

class CTOAgent(SwarmAgent):
    """Owns product design (STL generation), QA, and infrastructure."""

    def __init__(self):
        super().__init__(
            agent_id="cto",
            role="Chief Technology Officer",
            goal="Ship products from idea to listing in <7 days; defect rate <5%",
            backstory=(
                "You own the product design pipeline: from concept to STL to printable object. "
                "You manage QA (print quality, dimensional accuracy) and infrastructure "
                "(servers, model deployments, CI/CD). You command design, QA, and infra sub-agents. "
                "You track time-to-first-unit and defect rate."
            ),
        )

    async def execute(self, task: dict) -> dict:
        task_type = task.get("type", "design")

        if task_type == "design_product":
            return await self._design_product(task)
        elif task_type == "qa_review":
            return await self._qa_review(task)
        elif task_type == "infra_status":
            return await self._infra_status(task)
        else:
            prompt = f"Tech task: {json.dumps(task, indent=2)}\n\nProvide implementation plan."
            response = await self.llm(prompt)
            return {"success": True, "result": response}

    async def _design_product(self, task: dict) -> dict:
        """Generate product design spec from an opportunity."""
        opportunity = task.get("opportunity", {})
        prompt = f"""Design a physical product based on this opportunity:

OPPORTUNITY: {json.dumps(opportunity, indent=2)}

Produce:
1. Product name and description
2. Bill of materials (what to print, what to buy)
3. Dimensions (mm) and weight estimate
4. Print settings: material (PLA/PETG/TPU/resin), infill %, layer height
5. Post-processing steps
6. Estimated print time and material cost
7. OpenSCAD or parametric description for STL generation
8. Known risks or failure modes

Return structured JSON.
"""
        response = await self.llm(prompt)
        write_memory(self.agent_id, "products", f"Design: {opportunity.get('title', 'unnamed')[:60]}", response)
        record_metric("cto.products_designed", 1.0, self.agent_id)
        try:
            return {"success": True, "result": json.loads(response)}
        except json.JSONDecodeError:
            return {"success": True, "result": {"raw": response}}

    async def _qa_review(self, task: dict) -> dict:
        """QA review of a printed product."""
        product = task.get("product", {})
        prompt = f"""QA review for this product:

{json.dumps(product, indent=2)}

Check:
1. Dimensional accuracy (within 0.5mm tolerance)
2. Surface finish quality
3. Structural integrity
4. Safety (no sharp edges, non-toxic material)
5. Packaging adequacy

Return: PASS, CONDITIONAL PASS, or FAIL with specific issues.
"""
        response = await self.llm(prompt)
        record_metric("cto.qa_reviews", 1.0, self.agent_id)
        return {"success": True, "result": response}

    async def _infra_status(self, task: dict) -> dict:
        """Report infrastructure status."""
        prompt = """Report on the current infrastructure status:
- MM1 (Mac Mini): Ollama, agent swarm, memory vault
- Server 1: enrichment pipeline, knowledge DB
- Server 2: DB replica, backup
Flag any service outages or performance issues."""
        response = await self.llm(prompt)
        return {"success": True, "result": response}


# ═══════════════════════════════════════════════════════════════════════════
# CFO — Chief Financial Officer
# ═══════════════════════════════════════════════════════════════════════════

class CFOAgent(SwarmAgent):
    """Tracks all revenue, costs, and unit economics."""

    def __init__(self):
        super().__init__(
            agent_id="cfo",
            role="Chief Financial Officer",
            goal="Maintain runway >12 months and gross margin >40%",
            backstory=(
                "You track every dollar in and out of the Atlas business. "
                "You approve any spend >$500. You produce weekly P&L reports. "
                "You flag margin drift below 35%. You enforce the Lean Six Sigma cost model. "
                "You are conservative — when in doubt, say no."
            ),
        )

    async def execute(self, task: dict) -> dict:
        task_type = task.get("type", "finance")

        if task_type == "approve_spend":
            return await self._approve_spend(task)
        elif task_type == "weekly_pnl":
            return await self._weekly_pnl(task)
        elif task_type == "unit_economics":
            return await self._unit_economics(task)
        else:
            prompt = f"Finance task: {json.dumps(task, indent=2)}\n\nProvide financial analysis."
            response = await self.llm(prompt)
            return {"success": True, "result": response}

    async def _approve_spend(self, task: dict) -> dict:
        """Approve or reject a spending request."""
        amount = task.get("amount", 0)
        purpose = task.get("purpose", "unspecified")
        requester = task.get("requester", "unknown")

        if amount > 5000:
            write_memory(self.agent_id, "decisions",
                         f"ESCALATE: ${amount} spend from {requester}",
                         f"Amount exceeds $5K threshold. Requires human approval.\nPurpose: {purpose}",
                         confidence=1.0)
            return {
                "success": True,
                "result": {"approved": False, "reason": "Exceeds $5K — escalated to human", "escalated": True},
            }

        prompt = f"""Evaluate this spending request:

Amount: ${amount}
Purpose: {purpose}
Requester: {requester}

Consider:
1. Does this have clear ROI?
2. Is it within our margin targets (40% gross)?
3. Is there a cheaper alternative?
4. Is it reversible if it doesn't work out?

Return JSON: {{"approved": true/false, "rationale": "...", "conditions": ["..."]}}
"""
        response = await self.llm(prompt)
        write_memory(self.agent_id, "decisions", f"Spend ${amount}: {purpose[:60]}", response)
        record_metric("cfo.spend_reviewed", float(amount), self.agent_id)
        try:
            return {"success": True, "result": json.loads(response)}
        except json.JSONDecodeError:
            return {"success": True, "result": {"raw": response}}

    async def _weekly_pnl(self, task: dict) -> dict:
        """Generate weekly P&L report."""
        recent_metrics = read_memories(category="metrics", limit=50)
        prompt = f"""Generate a weekly P&L report.

Available metrics: {len(recent_metrics)} entries
Revenue data: {task.get('revenue', 'not yet available')}
Costs: {task.get('costs', 'not yet available')}

Produce:
1. Revenue summary (by product, by channel)
2. Cost breakdown (COGS, platform fees, API costs, infrastructure)
3. Gross margin %
4. Net margin %
5. Burn rate and runway estimate
6. Recommendations if margin < 40%

If data is incomplete, state what's missing and provide a template.
"""
        response = await self.llm(prompt)
        write_memory(self.agent_id, "metrics", "Weekly P&L report", response, confidence=0.6)
        return {"success": True, "result": response}

    async def _unit_economics(self, task: dict) -> dict:
        """Analyze unit economics for a product."""
        product = task.get("product", {})
        prompt = f"""Calculate unit economics for this product:

{json.dumps(product, indent=2)}

Compute:
1. COGS (material + print time at $X/hr + post-processing)
2. Platform fee (Etsy 6.5%, Amazon 15%)
3. Shipping (estimated)
4. Marketing cost per acquisition
5. Gross margin per unit
6. Break-even volume
7. Recommendation: PROFITABLE, MARGINAL, or UNPROFITABLE

Return structured JSON.
"""
        response = await self.llm(prompt)
        return {"success": True, "result": response}
