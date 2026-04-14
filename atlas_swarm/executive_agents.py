"""Executive agents: CMO, CRO, CTO, CFO for the Atlas Swarm.

Each follows the same SwarmAgent pattern: DMAIC, self-improvement,
persistent memory, local+cloud LLM inference.

CTO v2: Real CAD generation via CadQuery — no more text-only designs.
"""
import json
import logging
import os
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
        prompt = """Generate a revenue status report based on available metrics.
Include: total revenue, units sold, conversion rate, top products, margin analysis.
Flag any products below 35% margin for CEO review."""
        response = await self.llm(prompt)
        write_memory(self.agent_id, "metrics", "Weekly revenue report", response)
        return {"success": True, "result": response}


# ═══════════════════════════════════════════════════════════════════════════
# CTO — Chief Technology Officer  (v2 — real CAD generation)
# ═══════════════════════════════════════════════════════════════════════════

class CTOAgent(SwarmAgent):
    """Owns product design (CAD -> STL), QA, and infrastructure.

    v2: every design_product call produces a real CadQuery STL/STEP,
    validated for printability.  Text-only designs are gone.
    """

    def __init__(self):
        super().__init__(
            agent_id="cto",
            role="Chief Technology Officer",
            goal="Ship printable STL from idea in <1 hour; geometry defect rate <5%",
            backstory=(
                "You own the product design pipeline: concept -> parametric CAD -> "
                "validated STL -> print spec.  You use CadQuery for real geometry — "
                "never text-only descriptions.  Every model is validated for wall "
                "thickness, print volume, and manifold integrity before handoff."
            ),
        )

    async def execute(self, task: dict) -> dict:
        task_type = task.get("type", "design")
        dispatch = {
            "design_product": self._design_product,
            "generate_cad": self._generate_cad,
            "qa_review": self._qa_review,
            "infra_status": self._infra_status,
        }
        handler = dispatch.get(task_type, self._general)
        return await handler(task)

    async def _design_product(self, task: dict) -> dict:
        """Opportunity -> library match -> param tuning via AIM -> CAD -> validate -> export."""
        from . import cad_engine, cad_library

        opportunity = task.get("opportunity", {})
        combined = " ".join([
            opportunity.get("title", ""),
            opportunity.get("description", ""),
            opportunity.get("domain", ""),
            opportunity.get("vertical", ""),
        ]).lower()

        product_type = cad_engine.match_product_type(combined)

        if product_type:
            result = await self._design_from_library(product_type, opportunity)
        else:
            result = await self._design_from_llm_code(opportunity)

        if not result.get("success"):
            return result

        validation = result.get("validation", {})
        metrics = validation.get("metrics", {})
        dims = metrics.get("dimensions_mm", {})

        report = {
            "product_name": result.get("product_name", result.get("product_type", "custom")),
            "product_type": result.get("product_type", "custom"),
            "stl_file": result.get("stl_filename"),
            "step_file": result.get("step_filename"),
            "stl_path": result.get("stl_path"),
            "dimensions_mm": dims,
            "volume_cm3": metrics.get("volume_cm3", 0),
            "estimated_weight_g": metrics.get("estimated_weight_pla_g", 0),
            "printability": {
                "valid": validation.get("valid", False),
                "issues": validation.get("issues", []),
                "fits_print_bed": metrics.get("fits_print_bed", False),
            },
            "print_info": result.get("print_info", {}),
            "generation_time_s": result.get("generation_time_s", 0),
        }

        write_memory(self.agent_id, "products",
                      f"CAD: {report['product_name'][:50]}",
                      json.dumps(report, indent=2))
        record_metric("cto.stl_generated", 1.0, self.agent_id)
        record_metric("cto.products_designed", 1.0, self.agent_id)

        return {"success": True, "result": report}

    async def _design_from_library(self, product_type: str, opportunity: dict) -> dict:
        from . import cad_engine, cad_library

        catalog = cad_library.PRODUCTS[product_type]
        defaults = cad_library.get_function_params(product_type)

        prompt = f"""You are designing a **{catalog['name']}** for 3D-print manufacturing.

OPPORTUNITY: {json.dumps(opportunity, indent=2)}

Available parameters and their DEFAULTS:
{json.dumps(defaults, indent=2)}

Consider:
- Who buys this on Amazon/Etsy?  What dimensions do top sellers use?
- 3D-print constraints: max 256x256x256mm, min 1.2mm walls, FDM process.
- Material: {catalog['default_material']}

Return ONLY a JSON object with parameters you want to CHANGE.
Return {{}} (empty) to accept all defaults.
Do NOT include explanation — just the JSON object.
"""
        text = await self.llm(prompt, task_type="product_design", max_tokens=400)
        params = {}
        try:
            import re
            m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if m:
                params = json.loads(m.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        return cad_engine.generate_from_library(product_type, params)

    async def _design_from_llm_code(self, opportunity: dict) -> dict:
        from . import cad_engine

        prompt = f"""Write CadQuery Python code to model this product for 3D printing:

PRODUCT: {json.dumps(opportunity, indent=2)}

RULES (follow exactly):
1. `import cadquery as cq` and `import math` only.
2. Assign the final solid to `result`.
3. All dimensions in mm.  Min wall: 1.5mm.  Max: 256x256x256mm.
4. Use ONLY: .box(), .cylinder(), .circle(), .rect(), .extrude(), .cut(),
   .union(), .fillet(), .chamfer(), .translate(), .rotate(), .workplane(), .center()
5. 30-60 lines.  No lofts, sweeps, or splines — they break the kernel.
6. Add dimension comments.

Return ONLY Python code.  No markdown.  No explanation.
"""
        code = await self.llm(prompt, task_type="product_design", max_tokens=2000)
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:(-1 if lines[-1].strip() == "```" else len(lines))])

        name = opportunity.get("title", "custom")[:40]
        return cad_engine.generate_from_code(code, name)

    async def _generate_cad(self, task: dict) -> dict:
        from . import cad_engine
        return cad_engine.generate_from_library(
            task.get("product_type", "phone_stand"),
            task.get("params", {}),
        )

    async def _qa_review(self, task: dict) -> dict:
        from . import cad_engine

        product = task.get("product", {})
        checks = []

        stl_path = product.get("stl_path") or product.get("stl_file")
        if stl_path:
            try:
                from pathlib import Path
                import cadquery as cq
                p = Path(stl_path)
                if p.exists() and p.suffix == ".step":
                    solid = cq.importers.importStep(str(p))
                    checks.append({"check": "geometry", "result": cad_engine.validate_geometry(solid)})
            except Exception as exc:
                checks.append({"check": "geometry", "result": {"error": str(exc)}})

        prompt = f"""QA review for this 3D-printed product:

{json.dumps(product, indent=2)}

Evaluate (1-5 each):
1. Print success likelihood (overhangs, bridges, supports?)
2. Dimensional accuracy for FDM (+/-0.3mm typical)
3. Structural integrity under normal use
4. Surface finish (layer lines acceptable?)
5. Safety (sharp edges? material safety?)

Return: PASS / CONDITIONAL PASS / FAIL with scores and specific issues.
"""
        response = await self.llm(prompt, task_type="quality_audit", max_tokens=800)
        checks.append({"check": "design_review", "result": response})
        record_metric("cto.qa_reviews", 1.0, self.agent_id)
        return {"success": True, "result": {"qa_checks": checks}}

    async def _infra_status(self, task: dict) -> dict:
        from . import cad_engine
        return {
            "success": True,
            "result": {
                "cad_engine": "CadQuery 2.7.0",
                "available_products": cad_engine.list_available_products(),
                "renders_on_disk": len(cad_engine.list_renders()),
                "recent_renders": cad_engine.list_renders()[:5],
            },
        }

    async def _general(self, task: dict) -> dict:
        prompt = f"Tech task: {json.dumps(task, indent=2)}\n\nProvide implementation plan."
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
