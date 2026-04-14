"""Product Visualizer — generates photorealistic product renders from text descriptions.

Uses Stable Diffusion 3 Medium via MLX on Apple Silicon for local, free image generation.
Integrates with the swarm dashboard to visualize products before committing manufacturing resources.

Pipeline: Product brief → prompt engineering → SD3 MLX → product render PNG
"""
import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from .aim import get_aim
from .memory import write_memory, record_metric

log = logging.getLogger(__name__)

RENDERS_DIR = Path(os.environ.get("RENDERS_DIR", Path.home() / "Projects/atlas-swarm/renders"))
RENDERS_DIR.mkdir(parents=True, exist_ok=True)

# SD3 model path
SD3_MODEL_PATH = Path(os.environ.get(
    "SD3_MODEL_PATH",
    Path.home() / "Projects/atlas-swarm/models/sd3-medium-mlx",
))


def _generate_sd3_image(prompt: str, negative_prompt: str = "",
                         width: int = 1024, height: int = 1024,
                         steps: int = 28, cfg_scale: float = 7.0,
                         seed: Optional[int] = None) -> Optional[bytes]:
    """Generate an image using SD3 Medium MLX. Returns PNG bytes."""
    try:
        import mlx.core as mx
        from diffusionkit.mlx import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            str(SD3_MODEL_PATH),
            shift=3.0,
            model_version="argmaxinc/mlx-stable-diffusion-3-medium",
        )

        image = pipe.generate_image(
            prompt,
            num_inference_steps=steps,
            cfg_weight=cfg_scale,
            seed=seed or int(time.time()) % 2**32,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
        )

        # Convert to PNG bytes
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    except ImportError:
        log.warning("[VIS] DiffusionKit not installed or SD3 model not downloaded yet")
        return None
    except Exception as e:
        log.error(f"[VIS] Image generation failed: {e}")
        return None


def _fallback_placeholder(product_name: str) -> bytes:
    """Generate a simple placeholder image when SD3 isn't available."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (512, 512), color=(240, 240, 245))
        draw = ImageDraw.Draw(img)
        # Draw product name centered
        draw.text((256, 200), "PRODUCT RENDER", fill=(100, 100, 100), anchor="mm")
        draw.text((256, 260), product_name[:40], fill=(50, 50, 50), anchor="mm")
        draw.text((256, 320), "(SD3 model loading...)", fill=(150, 150, 150), anchor="mm")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # Minimal 1x1 white pixel PNG
        return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'


async def generate_product_render(
    product_brief: dict,
    style: str = "product_photo",
) -> dict:
    """Generate a photorealistic product render from a product brief.

    Uses AIM to craft an optimal SD3 prompt, then generates the image.

    Returns: {
        "image_path": str (path to saved PNG),
        "image_base64": str (for embedding in dashboard),
        "prompt_used": str,
        "model": "sd3-medium-mlx",
        "generation_time_s": float,
    }
    """
    product_name = product_brief.get("product_name", "Product")
    description = product_brief.get("product_description",
                  product_brief.get("description", ""))[:500]
    material = product_brief.get("material",
               product_brief.get("bill_of_materials", "white plastic"))

    # Use AIM to craft the perfect SD3 prompt
    aim = get_aim()
    prompt_result = await aim.generate(
        prompt=f"""Create a Stable Diffusion prompt for a photorealistic product render.

Product: {product_name}
Description: {description}
Material: {material}

The image should look like a professional Amazon/Etsy product photo:
- White/light gray gradient background
- Studio lighting (3-point: key, fill, rim)
- 45-degree angle view showing depth and detail
- Sharp focus, no depth-of-field blur
- Clean, no text overlay
- Photorealistic, not cartoon

Return ONLY the SD3 prompt text, nothing else. Start with the subject description.
Example format: "A professional product photograph of [item], white studio background, 3-point lighting, sharp focus, 4K, photorealistic"
""",
        task_type="content_generation",
        max_tokens=200,
    )

    sd_prompt = prompt_result["text"].strip().strip('"').strip("'")
    # Ensure it's a clean prompt (remove any preamble)
    if '\n' in sd_prompt:
        sd_prompt = sd_prompt.split('\n')[-1].strip()

    negative_prompt = (
        "blurry, low quality, cartoon, anime, sketch, watermark, text, "
        "deformed, ugly, duplicate, morbid, mutilated, poorly drawn, "
        "bad anatomy, bad proportions, extra limbs, disfigured"
    )

    log.info(f"[VIS] Generating render for '{product_name}' with prompt: {sd_prompt[:100]}...")
    start = time.monotonic()

    # Try SD3 MLX first, fall back to placeholder
    png_bytes = _generate_sd3_image(
        prompt=sd_prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        steps=28,
    )

    if png_bytes is None:
        log.warning(f"[VIS] SD3 not available, using placeholder for '{product_name}'")
        png_bytes = _fallback_placeholder(product_name)
        model_used = "placeholder"
    else:
        model_used = "sd3-medium-mlx"

    elapsed = time.monotonic() - start

    # Save to renders directory
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in product_name)[:50].strip()
    filename = f"{safe_name}_{int(time.time())}.png"
    filepath = RENDERS_DIR / filename
    filepath.write_bytes(png_bytes)

    # Base64 for dashboard embedding
    b64 = base64.b64encode(png_bytes).decode("utf-8")

    # Log to memory
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

    log.info(f"[VIS] Render complete: {filepath.name} ({model_used}, {elapsed:.1f}s)")
    return result
