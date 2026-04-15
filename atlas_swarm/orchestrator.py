"""Atlas Swarm Orchestrator v2.1 — Full company with AIM + KIE + CAD Engine.

Coordinates 9 agents via AIM (multi-model router) + Q-learning:
  Executive: CEO, CMO, CRO, CTO (v2 — real CAD), CFO
  Operations: Manufacturing, Content Production, Marketplace, R&D

Powered by:
  - AIM v1 (Atlas Intelligence Model — multi-model router with ML)
  - KIE (Knowledge Intelligence Engine — 49M+ record pipeline)
  - CAD Engine (CadQuery parametric library + LLM code generation)
  - Q-learning self-evolving router
  - Persistent Obsidian + SQLite memory
  - Lean Six Sigma DMAIC cycles
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from .agent_base import SwarmAgent
from .aim import AIM, get_aim
from .ceo_agent import CEOAgent
from .company_ops import (
    ContentProductionAgent, ManufacturingAgent, MarketplaceAgent, RDAgent,
)
from .executive_agents import CFOAgent, CMOAgent, CROAgent, CTOAgent
from .kie import KIE, get_kie
from .knowledge_bridge import get_bridge
from .memory import read_memories, record_metric, write_memory
from .router import QLearningRouter

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)

CONFIG_PATH = Path(os.environ.get(
    "SWARM_CONFIG",
    Path.home() / "Projects/atlas-swarm/swarm_config.yaml",
))


class SwarmOrchestrator:
    """Main orchestrator for the Atlas Swarm — full company."""

    def __init__(self):
        self.config = self._load_config()
        self.router = QLearningRouter(
            exploration_rate=self.config.get("router", {}).get("exploration_rate", 0.15),
            learning_rate=self.config.get("router", {}).get("learning_rate", 0.1),
            discount_factor=self.config.get("router", {}).get("discount_factor", 0.95),
        )
        self.aim = get_aim()
        self.kie = get_kie()
        self.agents: dict[str, SwarmAgent] = {}
        self._init_agents()

    def _load_config(self) -> dict:
        if CONFIG_PATH.exists():
            return yaml.safe_load(CONFIG_PATH.read_text())
        log.warning(f"Config not found at {CONFIG_PATH}, using defaults")
        return {}

    def _init_agents(self) -> None:
        """Initialize all 9 company agents."""
        self.agents["ceo"] = CEOAgent()
        self.agents["cmo"] = CMOAgent()
        self.agents["cro"] = CROAgent()
        self.agents["cto"] = CTOAgent()
        self.agents["cfo"] = CFOAgent()
        self.agents["manufacturing"] = ManufacturingAgent()
        self.agents["content"] = ContentProductionAgent()
        self.agents["marketplace"] = MarketplaceAgent()
        self.agents["rnd"] = RDAgent()
        log.info(f"Initialized {len(self.agents)} agents: {list(self.agents.keys())}")

    async def dispatch(self, task: dict) -> dict:
        """Route a task to the best agent."""
        task_type = task.get("type", "general")
        direct_routes = {
            "review_opportunities": "ceo", "weekly_review": "ceo",
            "generate_listing_content": "cmo", "social_post": "cmo", "seo_optimize": "cmo",
            "publish_listing": "cro", "optimize_pricing": "cro", "revenue_report": "cro",
            "design_product": "cto", "qa_review": "cto", "infra_status": "cto",
            "generate_cad": "cto",
            "approve_spend": "cfo", "weekly_pnl": "cfo", "unit_economics": "cfo",
            "generate_print_spec": "manufacturing", "quality_check": "manufacturing",
            "full_content": "content", "listing_copy": "content",
            "video_script": "content", "photo_shot_list": "content", "social_posts": "content",
            "prepare_listing": "marketplace", "compliance_check": "marketplace",
            "innovation_scan": "rnd", "patent_deep_dive": "rnd", "triz_analysis": "rnd",
        }
        agent_id = direct_routes.get(task_type) or self.router.select_agent(task_type, list(self.agents.keys()))
        agent = self.agents[agent_id]
        log.info(f"[DISPATCH] task={task_type} -> agent={agent_id}")
        result = await agent.run_task(task)
        success = result.get("success", False)
        reward = 1.0 if success else -0.5
        self.router.update(task_type, agent_id, reward)
        record_metric("orchestrator.dispatch", 1.0 if success else 0.0, "orchestrator",
                      {"task_type": task_type, "agent": agent_id})
        return {"agent": agent_id, "task_type": task_type, "success": success, "result": result.get("result")}

    async def run_full_pipeline(self, vertical: str = "anatomical model surgical training") -> dict:
        """End-to-end: scan -> score -> brief -> CAD -> spec -> content -> listing."""
        log.info(f"[PIPELINE] Starting for: {vertical}")
        steps = {}
        scan = await self.dispatch({"type": "innovation_scan", "verticals": [vertical]})
        steps["1_scan"] = {"success": scan["success"]}
        top_10 = scan.get("result", {}).get("top_10", [])
        if not top_10:
            return {"success": False, "error": "No opportunities", "steps": steps}
        top = top_10[0]
        steps["2_ceo"] = await self.dispatch({"type": "review_opportunities", "opportunities": top_10[:5]})

        # Step 3: CTO designs the product — now generates real STL
        design = await self.dispatch({
            "type": "design_product",
            "opportunity": {"title": top.get("title", ""), "domain": top.get("domain", ""), "vertical": vertical},
        })
        steps["3_design"] = design

        # Extract brief — now includes real STL path
        brief = design.get("result", {}) if isinstance(design.get("result"), dict) else {
            "product_name": top.get("title", vertical)[:50],
            "raw": str(design.get("result", ""))[:1000],
        }

        # Step 4: Manufacturing print spec (now receives STL path from CTO)
        steps["4_spec"] = await self.dispatch({"type": "generate_print_spec", "product_brief": brief})

        # Step 5: Content production
        steps["5_content"] = await self.dispatch({"type": "full_content", "product_brief": brief})

        # Step 6: Marketplace listing prep
        content_result = steps["5_content"].get("result", {})
        if isinstance(content_result, str):
            content_result = {"raw": content_result, "_product_name": brief.get("product_name", "unnamed")}
        steps["6_listing"] = await self.dispatch({
            "type": "prepare_listing",
            "content_package": content_result,
            "marketplace": "etsy",
        })

        # Step 7: CFO unit economics
        steps["7_econ"] = await self.dispatch({"type": "unit_economics", "product": brief})

        ok = all(
            s.get("success", False) if isinstance(s, dict) and "success" in s else True
            for s in steps.values()
        )
        product_name = brief.get("product_name", top.get("title", vertical)[:50])
        write_memory("orchestrator", "products", f"Pipeline: {product_name}",
                     json.dumps(steps, indent=2, default=str)[:3000], confidence=0.7)
        record_metric("orchestrator.pipeline_runs", 1.0, "orchestrator")
        return {"success": ok, "product": product_name, "steps": steps}

    async def weekly_cycle(self) -> dict:
        log.info("[WEEKLY] DMAIC cycle")
        ceo = await self.dispatch({"type": "weekly_review"})
        summary = {
            "date": datetime.now(timezone.utc).isoformat(),
            "agents": {a: ag.stats for a, ag in self.agents.items()},
            "aim": self.aim.status,
            "router_top": self.router.get_stats()[:5],
            "ceo": ceo.get("result"),
        }
        write_memory("orchestrator", "metrics", f"Weekly {datetime.now().strftime('%Y-W%V')}",
                     json.dumps(summary, indent=2, default=str), confidence=0.8)
        return summary

    def status(self) -> dict:
        return {
            "agents": {a: ag.stats for a, ag in self.agents.items()},
            "aim": self.aim.status,
            "router_entries": len(self.router.get_stats()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def create_app():
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from pydantic import BaseModel

    app = FastAPI(title="Atlas Swarm — AI Factory Company", version="2.1.0")
    orch = SwarmOrchestrator()

    class TaskRequest(BaseModel):
        type: str
        description: str = ""
        data: dict = {}

    @app.get("/health")
    async def health():
        return {"status": "healthy", "agents": len(orch.agents), "aim": "v1", "cad": "CadQuery 2.7"}

    @app.get("/status")
    async def status():
        return orch.status()

    @app.post("/task")
    async def submit_task(req: TaskRequest):
        return await orch.dispatch({"type": req.type, "description": req.description, **req.data})

    @app.post("/pipeline")
    async def pipeline(vertical: str = "anatomical model surgical training"):
        return await orch.run_full_pipeline(vertical)

    @app.post("/weekly-cycle")
    async def weekly():
        return await orch.weekly_cycle()

    @app.get("/aim/status")
    async def aim_status():
        return orch.aim.status

    @app.post("/aim/generate")
    async def aim_gen(prompt: str, task_type: str = "general", max_tokens: int = 2000):
        return await orch.aim.generate(prompt=prompt, task_type=task_type, max_tokens=max_tokens)

    @app.get("/kie/scan")
    async def kie_scan(vertical: str = "medical device", limit: int = 10):
        opps = await orch.kie.scan_opportunities(verticals=[vertical], limit_per_vertical=limit)
        return await orch.kie.score_with_aim(opps, top_n=limit)

    @app.get("/kie/report")
    async def kie_report():
        return await orch.kie.weekly_pipeline_report()

    @app.get("/bridge/health")
    async def bh():
        return {"server1": "healthy" if await get_bridge().health() else "unreachable"}

    @app.get("/bridge/stats")
    async def bs():
        return await get_bridge().stats()

    @app.get("/bridge/search")
    async def bsearch(query: str, limit: int = 20):
        return await get_bridge().search(query, limit)

    @app.get("/bridge/expired-patents")
    async def bexp(limit: int = 20):
        return await get_bridge().find_expired_patents(limit=limit)

    @app.get("/router/stats")
    async def rs():
        return orch.router.get_stats()

    @app.get("/memories")
    async def mem(agent_id: Optional[str] = None, category: Optional[str] = None, limit: int = 20):
        return read_memories(agent_id=agent_id, category=category, limit=limit)

    # ── Visualization endpoints ──────────────────────────────────────

    @app.post("/visualize")
    async def visualize_product(product_name: str, description: str = "", material: str = "white PETG plastic"):
        from .visualizer import generate_product_render
        brief = {"product_name": product_name, "product_description": description, "material": material}
        return await generate_product_render(brief)

    @app.get("/renders")
    async def list_renders():
        renders_dir = Path.home() / "Projects/atlas-swarm/renders"
        if not renders_dir.exists():
            return []
        files = sorted(renders_dir.glob("*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [
            {"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1), "type": f.suffix.lstrip(".")}
            for f in files[:30]
            if f.suffix in (".png", ".stl", ".step", ".html")
        ]

    @app.get("/renders/{filename}")
    async def get_render(filename: str):
        filepath = Path.home() / "Projects/atlas-swarm/renders" / filename
        if not filepath.exists():
            return {"error": "Not found"}
        media_types = {
            ".png": "image/png", ".stl": "application/octet-stream",
            ".step": "application/octet-stream", ".html": "text/html",
        }
        mt = media_types.get(filepath.suffix, "application/octet-stream")
        return FileResponse(filepath, media_type=mt)

    @app.get("/3d-viewer")
    async def viewer_3d():
        viewer = Path.home() / "Projects/atlas-swarm/renders/viewer.html"
        if not viewer.exists():
            return {"error": "Viewer not found — run /design first to generate it"}
        return FileResponse(viewer, media_type="text/html")

    # ── CAD Engine endpoints (v2.1) ──────────────────────────────────

    @app.post("/design")
    async def design_product(description: str = "", product_type: str = "", params: str = "{}"):
        """Generate a product STL — from library OR free-text description.

        If product_type matches a library entry, uses parametric generation.
        Otherwise treats the input as a free-text description and routes
        through the CTO agent, which writes CadQuery code via AIM.

        Examples:
            POST /design?product_type=phone_stand&params={"width":90}
            POST /design?description=toy car with rounded edges
            POST /design?description=wall-mounted guitar hook
        """
        from . import cad_engine, cad_library

        try:
            parsed = json.loads(params)
        except json.JSONDecodeError:
            parsed = {}

        # Combine inputs — description takes priority, product_type is fallback
        text = description.strip() or product_type.strip()
        if not text:
            return {"success": False, "error": "Provide description or product_type"}

        # Try library match first (fast path, no LLM)
        matched = cad_engine.match_product_type(text)
        if matched and not description:
            # Exact library hit via product_type param
            return cad_engine.generate_from_library(matched, parsed)

        if matched and description:
            # Library match from description — route through CTO for param tuning
            result = await orch.dispatch({
                "type": "design_product",
                "opportunity": {"title": text, "description": description},
            })
            inner = result.get("result")
            if inner is not None:
                return inner
            return {"success": result.get("success", False), "error": result.get("error", "No result"), "dispatch": result}

        # No library match — CTO agent generates CadQuery code via LLM
        result = await orch.dispatch({
            "type": "design_product",
            "opportunity": {"title": text, "description": text},
        })
        # Unwrap dispatch envelope; preserve errors
        inner = result.get("result")
        if inner is not None:
            return inner
        return {"success": result.get("success", False), "error": result.get("error", "No result from CTO agent"), "dispatch": result}

    @app.get("/products")
    async def list_products():
        """List available parametric product types and their tunable parameters."""
        from . import cad_engine
        return cad_engine.list_available_products()

    @app.post("/cad/generate")
    async def cad_generate(product_type: str = "phone_stand"):
        """Generate CAD via CTO agent (includes AIM parameter tuning)."""
        return await orch.dispatch({
            "type": "generate_cad",
            "product_type": product_type,
        })

    @app.get("/cad/renders")
    async def cad_renders():
        """List all generated files — STL, STEP, GLB, PNG."""
        from . import cad_engine
        renders = cad_engine.list_renders()
        # Also include GLB files
        glb_dir = Path.home() / "Projects/atlas-swarm/renders"
        for f in sorted(glb_dir.glob("*.glb"), key=lambda p: p.stat().st_mtime, reverse=True):
            renders.append({
                "name": f.name, "type": "glb",
                "size_kb": round(f.stat().st_size / 1024, 1),
                "path": str(f),
            })
        return renders

    # ── Text-to-3D (Meshy/Tripo) — textured GLB models ──────────────

    @app.post("/mesh3d")
    async def mesh3d(description: str, product_name: str = "", provider: str = "auto"):
        """Generate a textured 3D model (GLB) via Meshy.ai or Tripo3D.

        This produces a VISUAL model for the viewer, not an engineering STL.
        Example: POST /mesh3d?description=toy race car&product_name=race_car
        """
        from . import mesh_gen
        name = product_name or description.replace(" ", "_")[:30]
        return await mesh_gen.generate_3d_model(description, name, provider)

    @app.get("/mesh3d/providers")
    async def mesh3d_providers():
        """Check which text-to-3D providers are configured."""
        from . import mesh_gen
        providers = mesh_gen.available_providers()
        return {
            "providers": providers,
            "configured": len(providers) > 0,
            "hint": "Set MESHY_API_KEY or TRIPO_API_KEY env var" if not providers else None,
        }

    # ── Full product pipeline: STL + GLB + Photo ─────────────────────

    @app.post("/design/full")
    async def design_full(description: str):
        """Generate ALL three outputs for a product:
        1. CadQuery STL (for 3D printing)
        2. Meshy/Tripo GLB (for viewer with textures)
        3. Flux.1 PNG (for marketing photo)

        Runs STL synchronously, GLB + Photo in background.
        """
        from . import cad_engine, mesh_gen
        from .visualizer import generate_product_render

        text = description.strip()
        if not text:
            return {"success": False, "error": "Provide description"}

        name = "".join(c if c.isalnum() or c in "-_ " else "" for c in text)[:30].strip().replace(" ", "_")
        results = {"description": text, "outputs": {}}

        # 1. CadQuery STL (fast — local)
        stl_result = await orch.dispatch({
            "type": "design_product",
            "opportunity": {"title": text, "description": text},
        })
        stl_inner = stl_result.get("result")
        results["outputs"]["stl"] = stl_inner or {"error": stl_result.get("error", "STL gen failed")}

        # 2. Text-to-3D GLB (async — cloud API)
        glb_result = await mesh_gen.generate_3d_model(text, name)
        results["outputs"]["glb"] = glb_result

        # 3. Flux.1 product photo (async — local GPU)
        try:
            photo = await generate_product_render({"product_name": name, "product_description": text})
            results["outputs"]["photo"] = {
                "success": True,
                "image_path": photo.get("image_path"),
                "model": photo.get("model"),
            }
        except Exception as e:
            results["outputs"]["photo"] = {"success": False, "error": str(e)}

        results["success"] = any(
            r.get("success") for r in results["outputs"].values() if isinstance(r, dict)
        )
        return results

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8100)
