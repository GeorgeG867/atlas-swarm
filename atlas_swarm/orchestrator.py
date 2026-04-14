"""Atlas Swarm Orchestrator v2 — Full company with AIM + KIE.

Coordinates 9 agents via AIM (multi-model router) + Q-learning:
  Executive: CEO, CMO, CRO, CTO, CFO
  Operations: Manufacturing, Content Production, Marketplace, R&D

Powered by:
  - AIM v1 (Atlas Intelligence Model — multi-model router with ML)
  - KIE (Knowledge Intelligence Engine — 47M+ record pipeline)
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
        """End-to-end: scan -> score -> brief -> spec -> content -> listing."""
        log.info(f"[PIPELINE] Starting for: {vertical}")
        steps = {}
        scan = await self.dispatch({"type": "innovation_scan", "verticals": [vertical]})
        steps["1_scan"] = {"success": scan["success"]}
        top_10 = scan.get("result", {}).get("top_10", [])
        if not top_10:
            return {"success": False, "error": "No opportunities", "steps": steps}
        top = top_10[0]
        steps["2_ceo"] = (await self.dispatch({"type": "review_opportunities", "opportunities": top_10[:5]}))
        brief = await self.kie.generate_product_brief({"title": top.get("title", ""), "text": "", "_vertical": vertical})
        steps["3_brief"] = {"product": brief.get("product_name", "?")}
        steps["4_spec"] = await self.dispatch({"type": "generate_print_spec", "product_brief": brief})
        steps["5_content"] = await self.dispatch({"type": "full_content", "product_brief": brief})
        steps["6_listing"] = await self.dispatch({"type": "prepare_listing", "content_package": steps["5_content"].get("result", {}), "marketplace": "etsy"})
        steps["7_econ"] = await self.dispatch({"type": "unit_economics", "product": brief})
        ok = all(s.get("success", False) if isinstance(s, dict) and "success" in s else True for s in steps.values())
        write_memory("orchestrator", "products", f"Pipeline: {brief.get('product_name', vertical)[:50]}",
                     json.dumps(steps, indent=2, default=str)[:3000], confidence=0.7)
        record_metric("orchestrator.pipeline_runs", 1.0, "orchestrator")
        return {"success": ok, "product": brief.get("product_name"), "steps": steps}

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
    from pydantic import BaseModel

    app = FastAPI(title="Atlas Swarm — AI Factory Company", version="2.0.0")
    orch = SwarmOrchestrator()

    class TaskRequest(BaseModel):
        type: str
        description: str = ""
        data: dict = {}

    @app.get("/health")
    async def health():
        return {"status": "healthy", "agents": len(orch.agents), "aim": "v1"}

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

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8100)
