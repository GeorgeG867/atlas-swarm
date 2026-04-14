"""Company Operations — Manufacturing, Video, Marketplace, and Content Pipeline.

The operational layer that turns product briefs into real products and revenue.
Agents handle: 3D printing specs → product photos → video → marketplace listings.

All agents use AIM for intelligence and persistent memory for learning.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from .agent_base import SwarmAgent
from .aim import get_aim, ModelTier
from .memory import read_memories, record_metric, write_memory

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Manufacturing Agent — 3D Print Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class ManufacturingAgent(SwarmAgent):
    """Manages the 3D printing pipeline from design to finished product."""

    def __init__(self):
        super().__init__(
            agent_id="manufacturing",
            role="Manufacturing Operations Manager",
            goal="Produce physical products from design briefs with <5% defect rate",
            backstory=(
                "You manage the 3D printing mini-factory. You convert product briefs "
                "into print-ready specifications (STL parameters, material selection, "
                "print settings, post-processing steps). You track print queue, "
                "material inventory, and quality metrics."
            ),
        )

    async def execute(self, task: dict) -> dict:
        task_type = task.get("type", "print_spec")

        if task_type == "generate_print_spec":
            return await self._generate_print_spec(task)
        elif task_type == "quality_check":
            return await self._quality_check(task)
        elif task_type == "inventory_status":
            return await self._inventory_status(task)
        else:
            return await self._general(task)

    async def _generate_print_spec(self, task: dict) -> dict:
        """Generate complete 3D print specification from product brief."""
        brief = task.get("product_brief", {})
        aim = get_aim()

        result = await aim.generate(
            prompt=f"""Generate detailed 3D print specifications for:

PRODUCT BRIEF: {json.dumps(brief, indent=2)[:2000]}

Produce EXACT specifications:
1. printer_type: "FDM" or "SLA/Resin"
2. material: specific filament/resin (e.g., "Bambu PLA Basic - Matte White")
3. layer_height_mm: 0.08 to 0.28
4. infill_percent: 10 to 100
5. infill_pattern: "gyroid" / "grid" / "honeycomb"
6. wall_count: 2-6
7. supports: true/false, type if needed
8. print_temp_c: nozzle temperature
9. bed_temp_c: bed temperature
10. estimated_print_time_hours: total time
11. estimated_material_grams: material weight
12. estimated_material_cost_usd: material cost
13. post_processing: list of steps (sanding, priming, painting, etc.)
14. quality_checks: list of QA checkpoints
15. packaging: how to package for shipping

Return as JSON object.
""",
            system="You are an expert 3D printing engineer. Be specific about settings — generic answers are worthless.",
            task_type="product_design",
            require_local=True,
        )

        spec = {"raw": result["text"]}
        try:
            import re
            match = re.search(r'\{.*\}', result["text"], re.DOTALL)
            if match:
                spec = json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        spec["_aim_model"] = result["model"]
        spec["_product_name"] = brief.get("product_name", "unnamed")
        write_memory(self.agent_id, "products",
                     f"Print spec: {spec.get('_product_name', 'unnamed')[:60]}", json.dumps(spec, indent=2))
        record_metric("manufacturing.specs_generated", 1.0, self.agent_id)
        return {"success": True, "result": spec}

    async def _quality_check(self, task: dict) -> dict:
        aim = get_aim()
        result = await aim.generate(
            prompt=f"QA checklist for: {json.dumps(task.get('product', {}), indent=2)[:1000]}\n\nReturn pass/fail with specific measurements to verify.",
            task_type="quality_audit",
            require_local=True,
        )
        return {"success": True, "result": result["text"]}

    async def _inventory_status(self, task: dict) -> dict:
        return {"success": True, "result": {
            "filament": {"PLA_white": "1kg", "PETG_black": "0.5kg", "TPU_flexible": "0.25kg"},
            "resin": {"standard_grey": "500ml", "tough_clear": "250ml"},
            "note": "Inventory tracking not yet automated — update manually",
        }}

    async def _general(self, task: dict) -> dict:
        aim = get_aim()
        result = await aim.generate(
            prompt=f"Manufacturing task: {json.dumps(task, indent=2)[:1500]}",
            task_type="product_design",
            require_local=True,
        )
        return {"success": True, "result": result["text"]}


# ═══════════════════════════════════════════════════════════════════════════
# Content Production Agent — Photos, Videos, Copy
# ═══════════════════════════════════════════════════════════════════════════

class ContentProductionAgent(SwarmAgent):
    """Produces all content: product photos, videos, descriptions, social posts."""

    def __init__(self):
        super().__init__(
            agent_id="content_production",
            role="Content Production Manager",
            goal="Produce marketplace-ready content (photos, video, copy) for every product in <24h",
            backstory=(
                "You manage content creation for the product catalog. "
                "For each product you produce: listing copy (title, bullets, description), "
                "photo shot list, video script, and social media posts. "
                "You use AIM's multimodal capabilities for image analysis when Gemma 4 is available."
            ),
        )

    async def execute(self, task: dict) -> dict:
        task_type = task.get("type", "full_content")

        if task_type == "listing_copy":
            return await self._listing_copy(task)
        elif task_type == "photo_shot_list":
            return await self._photo_shot_list(task)
        elif task_type == "video_script":
            return await self._video_script(task)
        elif task_type == "social_posts":
            return await self._social_posts(task)
        elif task_type == "full_content":
            return await self._full_content_package(task)
        else:
            aim = get_aim()
            result = await aim.generate(
                prompt=f"Content task: {json.dumps(task, indent=2)[:1500]}",
                task_type="content_generation",
                require_local=True,
            )
            return {"success": True, "result": result["text"]}

    async def _full_content_package(self, task: dict) -> dict:
        """Generate complete content package for a product."""
        brief = task.get("product_brief", {})
        aim = get_aim()

        result = await aim.generate(
            prompt=f"""Create a COMPLETE content package for this product:

PRODUCT BRIEF: {json.dumps(brief, indent=2)[:2000]}

Generate ALL of the following in one response:

1. LISTING (for Etsy + Amazon):
   - title (60 chars max)
   - bullets (5 benefit-focused points)
   - description (150 words, SEO-optimized)
   - tags (10 search keywords)
   - category suggestion

2. PHOTO SHOT LIST (5 shots):
   - Shot 1: Hero image (white background, 45-degree angle)
   - Shot 2: Scale reference (hand holding product)
   - Shot 3: Detail close-up (texture, finish quality)
   - Shot 4: Lifestyle / in-use context
   - Shot 5: Packaging / unboxing

3. VIDEO SCRIPT (15 seconds):
   - Scene 1 (0-3s): Hook — problem statement
   - Scene 2 (3-8s): Product reveal + key feature
   - Scene 3 (8-12s): In-use demonstration
   - Scene 4 (12-15s): CTA — where to buy
   - Music/mood suggestion

4. SOCIAL MEDIA (3 platforms):
   - Twitter/X: 280 chars + 3 hashtags
   - Instagram: Caption + 10 hashtags
   - TikTok: Hook + description + sounds suggestion

RULES:
- NO health/medical claims unless FDA-cleared
- NO superlatives ("best", "guaranteed", "#1")
- Include specific dimensions and materials
- Etsy-compliant: emphasize handcrafted/designed aspect
- Amazon-compliant: AI-generated content is allowed, no fraud

Return as structured JSON with keys: listing, photos, video, social
""",
            system="You are a professional e-commerce content creator who optimizes for conversion rate. Every word must earn its place.",
            task_type="content_generation",
            max_tokens=3000,
            require_local=True,
        )

        content = {"raw": result["text"]}
        try:
            import re
            match = re.search(r'\{.*\}', result["text"], re.DOTALL)
            if match:
                content = json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        content["_product_name"] = brief.get("product_name", "unnamed")
        content["_aim_model"] = result["model"]
        write_memory(self.agent_id, "products",
                     f"Content: {content.get('_product_name', 'unnamed')[:60]}", json.dumps(content, indent=2, default=str)[:2000])
        record_metric("content.packages_created", 1.0, self.agent_id)
        return {"success": True, "result": content}

    async def _listing_copy(self, task: dict) -> dict:
        aim = get_aim()
        result = await aim.generate(
            prompt=f"Write marketplace listing for: {json.dumps(task.get('product', {}), indent=2)[:1000]}\n\nReturn: title, 5 bullets, description, 10 tags. JSON format.",
            task_type="listing_creation",
            require_local=True,
        )
        return {"success": True, "result": result["text"]}

    async def _photo_shot_list(self, task: dict) -> dict:
        aim = get_aim()
        result = await aim.generate(
            prompt=f"Product photo shot list for: {task.get('product_name', 'product')}\n\n5 shots with exact angles, lighting, props, and background.",
            task_type="content_generation",
            require_local=True,
        )
        return {"success": True, "result": result["text"]}

    async def _video_script(self, task: dict) -> dict:
        aim = get_aim()
        result = await aim.generate(
            prompt=f"15-second product video script for: {task.get('product_name', 'product')}\n\n4 scenes with timestamps, visuals, and voiceover. TikTok/Instagram Reels format.",
            task_type="content_generation",
            require_local=True,
        )
        return {"success": True, "result": result["text"]}

    async def _social_posts(self, task: dict) -> dict:
        aim = get_aim()
        result = await aim.generate(
            prompt=f"Social media posts for product: {task.get('product_name', 'product')}\n\nGenerate for Twitter/X, Instagram, TikTok, LinkedIn. Include hashtags and CTAs.",
            task_type="social_media",
            require_local=True,
        )
        return {"success": True, "result": result["text"]}


# ═══════════════════════════════════════════════════════════════════════════
# Marketplace Agent — Etsy + Amazon + Alibaba
# ═══════════════════════════════════════════════════════════════════════════

class MarketplaceAgent(SwarmAgent):
    """Manages product listings across all marketplaces."""

    def __init__(self):
        super().__init__(
            agent_id="marketplace",
            role="Marketplace Operations Manager",
            goal="Maintain live listings with >2.5% conversion rate across Etsy and Amazon",
            backstory=(
                "You manage product listings on Etsy, Amazon (via SP-API), and Alibaba. "
                "You ensure compliance with each platform's content policies. "
                "You track listing performance, optimize underperformers, and expand "
                "to new marketplaces when volume justifies it."
            ),
        )

    async def execute(self, task: dict) -> dict:
        task_type = task.get("type", "prepare_listing")

        if task_type == "prepare_listing":
            return await self._prepare_listing(task)
        elif task_type == "compliance_check":
            return await self._compliance_check(task)
        elif task_type == "performance_report":
            return await self._performance_report(task)
        else:
            aim = get_aim()
            result = await aim.generate(
                prompt=f"Marketplace task: {json.dumps(task, indent=2)[:1500]}",
                task_type="listing_creation",
                require_local=True,
            )
            return {"success": True, "result": result["text"]}

    async def _prepare_listing(self, task: dict) -> dict:
        """Prepare a listing for marketplace submission."""
        content = task.get("content_package", {})
        marketplace = task.get("marketplace", "etsy")
        aim = get_aim()

        result = await aim.generate(
            prompt=f"""Prepare this product listing for {marketplace}:

CONTENT: {json.dumps(content, indent=2)[:2000]}

FORMAT for {marketplace.upper()} API submission:
- Validate title length ({marketplace} limits)
- Ensure no prohibited terms
- Format tags/keywords per platform spec
- Calculate final price (include platform fee: {"6.5% + $0.20" if marketplace == "etsy" else "15%" if marketplace == "amazon" else "varies"})
- Recommend category and subcategory

Also run compliance check:
- No FDA-regulated claims without clearance
- No "best", "#1", "guaranteed" claims
- AI-content disclosure requirements ({"none for Etsy standard" if marketplace == "etsy" else "required" if marketplace == "amazon_handmade" else "permitted on Amazon standard"})
- Image requirements met

Return JSON: {{
    "marketplace": "{marketplace}",
    "ready_to_submit": true/false,
    "listing_data": {{...}},
    "compliance_issues": [...],
    "recommended_price": N,
    "estimated_margin": N
}}
""",
            system=f"You are an expert {marketplace} seller with 1000+ listings. Format everything per {marketplace}'s exact API requirements.",
            task_type="listing_creation",
            require_local=True,
        )

        listing = {"raw": result["text"], "marketplace": marketplace}
        try:
            import re
            match = re.search(r'\{.*\}', result["text"], re.DOTALL)
            if match:
                listing = json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        write_memory(self.agent_id, "products",
                     f"Listing: {marketplace} - {content.get('_product_name', 'unnamed')[:40]}", json.dumps(listing, indent=2, default=str)[:1500])
        record_metric(f"marketplace.{marketplace}.listings_prepared", 1.0, self.agent_id)
        return {"success": True, "result": listing}

    async def _compliance_check(self, task: dict) -> dict:
        aim = get_aim()
        result = await aim.generate(
            prompt=f"Compliance check for listing:\n{json.dumps(task.get('listing', {}), indent=2)[:1500]}\n\nCheck: FDA claims, prohibited terms, platform AUP, IP infringement risk. Return PASS or FAIL with specific issues.",
            task_type="quality_audit",
            require_local=True,
        )
        return {"success": True, "result": result["text"]}

    async def _performance_report(self, task: dict) -> dict:
        recent = read_memories(agent_id=self.agent_id, category="products", limit=20)
        return {"success": True, "result": {
            "total_listings_prepared": len(recent),
            "marketplaces_active": ["etsy"],
            "note": "Actual sales data requires marketplace API integration (Etsy/Amazon API keys)"
        }}


# ═══════════════════════════════════════════════════════════════════════════
# R&D Agent — Patent Analysis + Product Innovation
# ═══════════════════════════════════════════════════════════════════════════

class RDAgent(SwarmAgent):
    """Conducts R&D using the knowledge base — patent analysis, TRIZ, innovation."""

    def __init__(self):
        super().__init__(
            agent_id="rnd",
            role="R&D Director",
            goal="Generate 10+ validated product concepts per week from the knowledge base",
            backstory=(
                "You mine the 47M+ knowledge records for innovation opportunities. "
                "You apply TRIZ principles, contradiction analysis, and cross-domain "
                "transfer to generate novel product concepts. You focus on expired patents "
                "and expiring-soon patents for immediate commercialization."
            ),
        )

    async def execute(self, task: dict) -> dict:
        task_type = task.get("type", "innovation_scan")

        if task_type == "innovation_scan":
            return await self._innovation_scan(task)
        elif task_type == "patent_deep_dive":
            return await self._patent_deep_dive(task)
        elif task_type == "triz_analysis":
            return await self._triz_analysis(task)
        else:
            aim = get_aim()
            result = await aim.generate(
                prompt=f"R&D task: {json.dumps(task, indent=2)[:1500]}",
                task_type="patent_analysis",
                require_local=True,
            )
            return {"success": True, "result": result["text"]}

    async def _innovation_scan(self, task: dict) -> dict:
        """Scan for innovation opportunities using KIE."""
        from .kie import get_kie
        kie = get_kie()
        verticals = task.get("verticals", None)
        opportunities = await kie.scan_opportunities(verticals=verticals)
        scored = await kie.score_with_aim(opportunities, top_n=10)

        record_metric("rnd.innovation_scans", 1.0, self.agent_id)
        return {"success": True, "result": {
            "total_scanned": len(opportunities),
            "top_10": [
                {
                    "title": r.get("title", "")[:80],
                    "rice": r.get("_rice_total", 0),
                    "verdict": r.get("_verdict", "?"),
                }
                for r in scored[:10]
            ],
        }}

    async def _patent_deep_dive(self, task: dict) -> dict:
        """Deep analysis of a specific patent for product potential."""
        patent_id = task.get("patent_id", "")
        aim = get_aim()
        result = await aim.generate(
            prompt=f"Deep patent analysis for ID: {patent_id}\n\nAnalyze: claims, commercial potential, expired/active status, manufacturing feasibility, market size, competitive landscape. Use TRIZ contradiction mapping.",
            task_type="patent_analysis",
            max_tokens=3000,
            require_local=True,
        )
        write_memory(self.agent_id, "products", f"Patent dive: {patent_id[:30]}", result["text"][:2000])
        return {"success": True, "result": result["text"]}

    async def _triz_analysis(self, task: dict) -> dict:
        aim = get_aim()
        result = await aim.generate(
            prompt=f"TRIZ contradiction analysis for: {json.dumps(task.get('problem', {}), indent=2)[:1500]}\n\nIdentify: improving parameter, worsening parameter, applicable TRIZ principles (1-40), resolution strategy.",
            task_type="patent_analysis",
            require_local=True,
        )
        return {"success": True, "result": result["text"]}
