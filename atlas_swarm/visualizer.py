"""Product Visualizer — photorealistic renders via Flux.1 schnell on Apple Silicon.

Uses HuggingFace diffusers FluxPipeline with MPS backend.
Model: black-forest-labs/FLUX.1-schnell (cached locally after first download).
~40s per 768x768 image on M4 Pro.
"""
import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

RENDERS_DIR = Path(os.environ.get("RENDERS_DIR", Path.home() / "Projects/atlas-swarm/renders"))
RENDERS_DIR.mkdir(parents=True, exist_ok=True)

# Lazy-loaded pipeline singleton
_pipe = None
_pipe_device = None


def _get_pipeline():
    """Load Flux.1 schnell pipeline — cached after first call (~12GB VRAM)."""
    global _pipe, _pipe_device
    if _pipe is not None:
        return _pipe

    try:
        import torch
        from diffusers import FluxPipeline

        log.info("[VIS] Loading Flux.1 schnell pipeline...")
        t0 = time.monotonic()

        _pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
        )

        if torch.backends.mps.is_available():
            _pipe = _pipe.to("mps")
            _pipe_device = "mps"
        elif torch.cuda.is_available():
            _pipe = _pipe.to("cuda")
            _pipe_device = "cuda"
        else:
            _pipe_device = "cpu"

        elapsed = time.monotonic() - t0
        log.info("[VIS] Flux.1 loaded on %s in %.1fs", _pipe_device, elapsed)
        return _pipe

    except Exception as e:
        log.error("[VIS] Failed to load Flux.1: %s", e)
        return None


def _generate_flux_image(
    prompt: str,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 768,
    steps: int = 4,
    seed: Optional[int] = None,
) -> Optional[bytes]:
    """Generate an image using Flux.1 schnell via diffusers. Returns PNG bytes."""
    pipe = _get_pipeline()
    if pipe is None:
        return None

    try:
        import torch
        generator = None
        if seed is not None:
            device = _pipe_device or "cpu"
            if device == "mps":
                generator = torch.Generator("mps").manual_seed(seed)
            elif device == "cuda":
                generator = torch.Generator("cuda").manual_seed(seed)
            else:
                generator = torch.Generator().manual_seed(seed)

        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,
            height=height,
            width=width,
            generator=generator,
        ).images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    except Exception as e:
        log.error("[VIS] Image generation failed: %s", e)
        return None


def _fallback_placeholder(product_name: str) -> bytes:
    """Simple placeholder when Flux.1 isn't available."""
    try:
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (512, 512), color=(240, 240, 245))
        draw = ImageDraw.Draw(img)
        draw.text((256, 200), "PRODUCT RENDER", fill=(100, 100, 100), anchor="mm")
        draw.text((256, 260), product_name[:40], fill=(50, 50, 50), anchor="mm")
        draw.text((256, 320), "(Flux.1 loading — retry in 60s)", fill=(150, 150, 150), anchor="mm")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'


async def generate_product_render(product_brief: dict) -> dict:
    """Generate a photorealistic product render.

    Returns image_path, image_base64, prompt_used, model, generation_time_s.
    """
    from . import config

    product_name = product_brief.get("product_name", "Product")
    description = product_brief.get("product_description",
                  product_brief.get("description", ""))[:300]
    material = product_brief.get("material",
               product_brief.get("bill_of_materials", "white plastic"))

    subject = product_name
    if description and description.lower() != product_name.lower():
        subject = f"{product_name}, {description}"

    # Prompt template from templates/cad_prompts.yaml — no hardcoded text
    template = config.get("cad_prompts.prompts.product_photo.template",
                          "A product photograph of {subject}, {material}")
    sd_prompt = template.format(subject=subject, material=material)

    log.info("[VIS] Generating render for '%s' — prompt: %s", product_name, sd_prompt[:80])
    start = time.monotonic()

    png_bytes = _generate_flux_image(
        prompt=sd_prompt,
        width=768,
        height=768,
        steps=4,
        seed=int(time.time()) % 2**31,
    )

    if png_bytes is None:
        log.warning("[VIS] Flux.1 not available, using placeholder")
        png_bytes = _fallback_placeholder(product_name)
        model_used = "placeholder"
    else:
        model_used = "flux.1-schnell"

    elapsed = time.monotonic() - start

    safe_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in product_name)[:50].strip()
    filename = f"{safe_name}_{int(time.time())}.png"
    filepath = RENDERS_DIR / filename
    filepath.write_bytes(png_bytes)

    b64 = base64.b64encode(png_bytes).decode("utf-8")

    from .memory import write_memory, record_metric
    write_memory(
        agent_id="visualizer",
        category="products",
        title=f"Render: {product_name[:50]}",
        content=json.dumps({
            "product": product_name,
            "prompt": sd_prompt,
            "model": model_used,
            "path": str(filepath),
            "generation_time_s": round(elapsed, 1),
        }, indent=2),
        confidence=0.7,
    )
    record_metric("visualizer.renders_generated", 1.0, "visualizer")

    result = {
        "image_path": str(filepath),
        "image_base64": b64,
        "prompt_used": sd_prompt,
        "model": model_used,
        "generation_time_s": round(elapsed, 1),
        "product_name": product_name,
    }

    log.info("[VIS] Render complete: %s (%s, %.1fs)", filepath.name, model_used, elapsed)
    return result
