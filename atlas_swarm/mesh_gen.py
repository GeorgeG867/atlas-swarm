"""Text-to-3D mesh generation — Meshy.ai (primary) + Tripo3D (fallback).

Generates textured GLB files from text descriptions for the 3D product viewer.
Separate from CadQuery (which makes engineering STLs for 3D printing).

API keys via env vars:
  MESHY_API_KEY   — from https://app.meshy.ai/settings/api
  TRIPO_API_KEY   — from https://platform.tripo3d.ai (tsk_* prefix)
"""
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger(__name__)

RENDERS_DIR = Path(os.environ.get(
    "RENDERS_DIR", Path.home() / "Projects/atlas-swarm/renders",
))
RENDERS_DIR.mkdir(parents=True, exist_ok=True)

MESHY_KEY = os.environ.get("MESHY_API_KEY", "")
TRIPO_KEY = os.environ.get("TRIPO_API_KEY", "")

MESHY_BASE = "https://api.meshy.ai/openapi/v2"
TRIPO_BASE = "https://api.tripo3d.ai/v2/openapi"


# ═══════════════════════════════════════════════════════════════════════
# Meshy.ai — best mesh quality for product models
# ═══════════════════════════════════════════════════════════════════════

async def meshy_generate(
    prompt: str,
    product_name: str = "product",
    timeout_s: int = 300,
) -> dict:
    """Generate a textured 3D model via Meshy.ai text-to-3D.

    Returns {success, glb_path, glb_filename, provider, ...} or {success: False, error}.
    """
    if not MESHY_KEY:
        return {"success": False, "error": "MESHY_API_KEY not set"}

    headers = {"Authorization": f"Bearer {MESHY_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=30) as client:
        # Step 1: Create preview task
        log.info("[MESHY] Creating text-to-3D task: '%s'", prompt[:80])
        resp = await client.post(
            f"{MESHY_BASE}/text-to-3d",
            headers=headers,
            json={
                "mode": "preview",
                "prompt": prompt,
                "should_remesh": True,
            },
        )
        if resp.status_code != 200 and resp.status_code != 202:
            return {"success": False, "error": f"Meshy create failed: {resp.status_code} {resp.text[:200]}"}

        data = resp.json()
        task_id = data.get("result") or data.get("task_id") or data.get("id")
        if not task_id:
            return {"success": False, "error": f"No task_id in response: {data}"}

        log.info("[MESHY] Task created: %s", task_id)

        # Step 2: Poll for completion
        t0 = time.monotonic()
        glb_url = None
        while time.monotonic() - t0 < timeout_s:
            await asyncio.sleep(5)
            poll = await client.get(f"{MESHY_BASE}/text-to-3d/{task_id}", headers=headers)
            if poll.status_code != 200:
                continue
            status_data = poll.json()
            status = status_data.get("status", "")
            progress = status_data.get("progress", 0)

            if status in ("SUCCEEDED", "FINISHED", "succeeded"):
                # Find GLB URL in model_urls or model_url
                urls = status_data.get("model_urls", {})
                glb_url = urls.get("glb") or urls.get("obj") or status_data.get("model_url")
                break
            elif status in ("FAILED", "EXPIRED", "failed"):
                return {"success": False, "error": f"Meshy task failed: {status_data.get('task_error', status)}"}

            log.info("[MESHY] Progress: %s%% status=%s", progress, status)

        if not glb_url:
            return {"success": False, "error": f"Meshy timed out after {timeout_s}s"}

        # Step 3: Download GLB
        log.info("[MESHY] Downloading GLB from %s", glb_url[:80])
        dl = await client.get(glb_url)
        if dl.status_code != 200:
            return {"success": False, "error": f"GLB download failed: {dl.status_code}"}

        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in product_name)[:40]
        ts = int(time.time())
        glb_path = RENDERS_DIR / f"{safe}_{ts}.glb"
        glb_path.write_bytes(dl.content)

        elapsed = time.monotonic() - t0
        log.info("[MESHY] Done: %s (%.0fs, %dKB)", glb_path.name, elapsed, len(dl.content) // 1024)

        return {
            "success": True,
            "provider": "meshy",
            "glb_path": str(glb_path),
            "glb_filename": glb_path.name,
            "glb_size_kb": round(len(dl.content) / 1024, 1),
            "task_id": task_id,
            "generation_time_s": round(elapsed, 1),
        }


# ═══════════════════════════════════════════════════════════════════════
# Tripo3D — fast fallback
# ═══════════════════════════════════════════════════════════════════════

async def tripo_generate(
    prompt: str,
    product_name: str = "product",
    timeout_s: int = 300,
) -> dict:
    """Generate a textured 3D model via Tripo3D."""
    if not TRIPO_KEY:
        return {"success": False, "error": "TRIPO_API_KEY not set"}

    headers = {"Authorization": f"Bearer {TRIPO_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=30) as client:
        log.info("[TRIPO] Creating text-to-3D task: '%s'", prompt[:80])
        resp = await client.post(
            f"{TRIPO_BASE}/task",
            headers=headers,
            json={
                "type": "text_to_model",
                "prompt": prompt,
            },
        )
        if resp.status_code not in (200, 201):
            return {"success": False, "error": f"Tripo create failed: {resp.status_code} {resp.text[:200]}"}

        data = resp.json().get("data", resp.json())
        task_id = data.get("task_id") or data.get("id")
        if not task_id:
            return {"success": False, "error": f"No task_id: {data}"}

        log.info("[TRIPO] Task created: %s", task_id)

        t0 = time.monotonic()
        glb_url = None
        while time.monotonic() - t0 < timeout_s:
            await asyncio.sleep(5)
            poll = await client.get(f"{TRIPO_BASE}/task/{task_id}", headers=headers)
            if poll.status_code != 200:
                continue
            sd = poll.json().get("data", poll.json())
            status = sd.get("status", "")

            if status in ("success", "Success", "FINISHED"):
                output = sd.get("output", sd.get("result", {}))
                if isinstance(output, dict):
                    glb_url = output.get("model") or output.get("pbr_model") or output.get("base_model")
                elif isinstance(output, list) and output:
                    glb_url = output[0].get("url", "")
                break
            elif status in ("failed", "Failed", "FAILED"):
                return {"success": False, "error": f"Tripo failed: {sd}"}

        if not glb_url:
            return {"success": False, "error": f"Tripo timed out after {timeout_s}s"}

        dl = await client.get(glb_url)
        if dl.status_code != 200:
            return {"success": False, "error": f"GLB download failed: {dl.status_code}"}

        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in product_name)[:40]
        ts = int(time.time())
        glb_path = RENDERS_DIR / f"{safe}_{ts}.glb"
        glb_path.write_bytes(dl.content)

        elapsed = time.monotonic() - t0
        log.info("[TRIPO] Done: %s (%.0fs, %dKB)", glb_path.name, elapsed, len(dl.content) // 1024)

        return {
            "success": True,
            "provider": "tripo",
            "glb_path": str(glb_path),
            "glb_filename": glb_path.name,
            "glb_size_kb": round(len(dl.content) / 1024, 1),
            "task_id": task_id,
            "generation_time_s": round(elapsed, 1),
        }


# ═══════════════════════════════════════════════════════════════════════
# Unified interface — Meshy first, Tripo fallback
# ═══════════════════════════════════════════════════════════════════════

async def generate_3d_model(
    description: str,
    product_name: str = "product",
    provider: str = "auto",
) -> dict:
    """Generate a textured GLB from text.  Tries Meshy first, then Tripo."""
    prompt = f"A 3D product model of {description}, high quality, detailed, for marketplace listing"

    if provider == "meshy" or (provider == "auto" and MESHY_KEY):
        result = await meshy_generate(prompt, product_name)
        if result["success"]:
            return result
        log.warning("[MESH-GEN] Meshy failed: %s", result.get("error"))

    if provider == "tripo" or (provider == "auto" and TRIPO_KEY):
        result = await tripo_generate(prompt, product_name)
        if result["success"]:
            return result
        log.warning("[MESH-GEN] Tripo failed: %s", result.get("error"))

    return {
        "success": False,
        "error": "No text-to-3D provider available. Set MESHY_API_KEY or TRIPO_API_KEY.",
        "hint": "Get a free key at https://app.meshy.ai/settings/api",
    }


def available_providers() -> list[str]:
    """List which text-to-3D providers are configured."""
    out = []
    if MESHY_KEY:
        out.append("meshy")
    if TRIPO_KEY:
        out.append("tripo")
    return out
