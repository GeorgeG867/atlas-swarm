"""Knowledge Bridge — connects MM1 swarm to Server 1's 47M-record knowledge DB.

Uses Server 1's existing REST API at http://192.168.1.204:8000/api/v1/.
No SSH keys needed — pure HTTP over LAN (2ms latency).

Provides IdeaFrog-ready queries:
- Search by keyword across all enriched records
- Filter by patent_status (expired, expiring_soon, active)
- Get enriched fields (problem_statement, how_summary, triz_principle, etc.)
- RICE scoring of opportunities
- Stats and coverage metrics
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

log = logging.getLogger(__name__)

SERVER1_API = os.environ.get("SERVER1_API", "http://192.168.1.204:8000")
REQUEST_TIMEOUT = float(os.environ.get("BRIDGE_TIMEOUT", "30.0"))


class KnowledgeBridge:
    """HTTP client for Server 1's knowledge API."""

    def __init__(self, base_url: str = SERVER1_API):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=REQUEST_TIMEOUT,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Core queries ───────────────────────────────────────────────────

    async def stats(self) -> dict:
        """Get knowledge base stats (record counts, coverage, etc.)."""
        client = await self._get_client()
        resp = await client.get("/api/v1/stats")
        resp.raise_for_status()
        return resp.json().get("stats", resp.json())

    async def search(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across knowledge records."""
        client = await self._get_client()
        resp = await client.get("/api/v1/search", params={"query": query, "limit": limit})
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", data.get("records", []))

    async def get_knowledge(
        self,
        limit: int = 20,
        patent_status: Optional[str] = None,
        source: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> list[dict]:
        """Fetch knowledge records with optional filters."""
        client = await self._get_client()
        params: dict[str, Any] = {"limit": limit}
        if patent_status:
            params["patent_status"] = patent_status
        if source:
            params["source"] = source
        if min_score is not None:
            params["min_score"] = min_score
        resp = await client.get("/api/v1/knowledge", params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("records", [])

    async def get_contradictions(self, limit: int = 20) -> list[dict]:
        """Get TRIZ contradictions from enriched records."""
        client = await self._get_client()
        resp = await client.get("/api/v1/contradictions", params={"limit": limit})
        resp.raise_for_status()
        return resp.json() if isinstance(resp.json(), list) else resp.json().get("results", [])

    async def get_first_principles(self, limit: int = 20) -> list[dict]:
        """Get first-principles analyses."""
        client = await self._get_client()
        resp = await client.get("/api/v1/first-principles", params={"limit": limit})
        resp.raise_for_status()
        return resp.json() if isinstance(resp.json(), list) else resp.json().get("results", [])

    async def triz_lookup(self, principle_id: int) -> dict:
        """Look up a specific TRIZ principle."""
        client = await self._get_client()
        resp = await client.get("/api/v1/triz/lookup", params={"principle_id": principle_id})
        resp.raise_for_status()
        return resp.json()

    async def health(self) -> bool:
        """Check if Server 1 API is reachable."""
        try:
            client = await self._get_client()
            resp = await client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    # ── IdeaFrog opportunity queries ───────────────────────────────────

    async def find_expired_patents(
        self,
        keywords: Optional[list[str]] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Find expired patents — prime candidates for product development.

        These patents have entered the public domain and can be freely
        manufactured without licensing. Combined with enrichment data
        (problem_statement, how_summary, aha_mechanism), these are
        direct product opportunities.
        """
        records = await self.get_knowledge(
            limit=limit,
            patent_status="expired",
            min_score=0.5,
        )
        if keywords:
            kw_lower = [k.lower() for k in keywords]
            records = [
                r for r in records
                if any(
                    kw in (r.get("title", "") + " " + r.get("abstract", "")).lower()
                    for kw in kw_lower
                )
            ]
        return records

    async def find_expiring_soon(self, limit: int = 50) -> list[dict]:
        """Find patents expiring soon — prepare to manufacture on expiration."""
        return await self.get_knowledge(
            limit=limit,
            patent_status="expiring_soon",
            min_score=0.5,
        )

    async def find_opportunities(
        self,
        vertical: str = "medical devices",
        limit: int = 20,
    ) -> list[dict]:
        """Search for product opportunities in a specific vertical.

        Combines keyword search with enrichment quality to surface
        the best candidates for the marketing/sales swarm.
        """
        results = await self.search(query=vertical, limit=limit * 2)

        # Score by enrichment completeness (more fields filled = better opportunity)
        scored = []
        for r in results:
            completeness = sum(1 for field in [
                "problem_statement", "how_summary", "triz_principle",
                "contradiction_improve", "aha_mechanism", "first_principles",
            ] if r.get(field))
            scored.append({
                **r,
                "_completeness": completeness,
                "_opportunity_score": (r.get("score", 0) or 0) * (1 + completeness * 0.15),
            })

        scored.sort(key=lambda x: x["_opportunity_score"], reverse=True)
        return scored[:limit]

    async def rice_score_opportunities(
        self,
        opportunities: list[dict],
        market_size_estimates: Optional[dict[str, float]] = None,
    ) -> list[dict]:
        """Apply RICE scoring to a list of opportunities.

        RICE = (Reach x Impact x Confidence) / Effort

        - Reach: estimated from patent citations + market vertical
        - Impact: from enrichment score + novelty
        - Confidence: from enrichment completeness
        - Effort: inverse of feasibility (simple = low effort)
        """
        market_sizes = market_size_estimates or {}

        for opp in opportunities:
            citations = opp.get("citation_count", 0) or 0
            score = opp.get("score", 0) or 0
            completeness = opp.get("_completeness", 0)
            domain = opp.get("domain", "general")
            patent_status = opp.get("patent_status", "active")

            # Reach (0-10): based on citations and domain market size
            market_mult = market_sizes.get(domain, 1.0)
            reach = min(10, (citations / 100) * market_mult + 2)

            # Impact (0-10): based on enrichment score and status
            status_mult = {"expired": 1.5, "expiring_soon": 1.3, "active": 0.5}.get(patent_status, 1.0)
            impact = min(10, score * 10 * status_mult)

            # Confidence (0-10): based on enrichment completeness
            confidence = min(10, completeness * 1.67)

            # Effort (1-10): inverse complexity estimate
            has_abstract = bool(opp.get("abstract"))
            has_how = bool(opp.get("how_summary"))
            effort = max(1, 10 - (3 if has_abstract else 0) - (3 if has_how else 0) - (2 if patent_status == "expired" else 0))

            opp["_rice"] = {
                "reach": round(reach, 1),
                "impact": round(impact, 1),
                "confidence": round(confidence, 1),
                "effort": round(effort, 1),
                "score": round((reach * impact * confidence) / effort, 1),
            }

        opportunities.sort(key=lambda x: x.get("_rice", {}).get("score", 0), reverse=True)
        return opportunities


# ── Singleton for use across the swarm ─────────────────────────────────

_bridge: Optional[KnowledgeBridge] = None


def get_bridge() -> KnowledgeBridge:
    global _bridge
    if _bridge is None:
        _bridge = KnowledgeBridge()
    return _bridge
