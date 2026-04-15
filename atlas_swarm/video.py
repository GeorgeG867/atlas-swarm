"""Stable Video Diffusion XT — image-to-video animation on Apple Silicon.

Uses stabilityai/stable-video-diffusion-img2vid-xt with MPS backend.
Model: ~4.2GB fp16 cached locally after first download.

Memory strategy for M4 Pro 48GB:
  - sequential_cpu_offload: only the active submodel lives on MPS
  - 576x320 resolution (half canonical) keeps peak VRAM ~12GB
  - decode_chunk_size=2 decodes VAE in small batches
  - 14 frames default (25 available if memory allows)

Endpoint: POST /animate
  - image_url OR image_path required
  - frames: 14 or 25 (default 14)
  - fps: output video framerate (default 8)
  - motion_bucket_id: 1-255, higher = more motion (default 127)
"""
import asyncio
import gc
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

RENDERS_DIR = Path(os.environ.get("RENDERS_DIR", Path.home() / "Projects/atlas-swarm/renders"))
ANIMATIONS_DIR = RENDERS_DIR / "animations"
ANIMATIONS_DIR.mkdir(parents=True, exist_ok=True)

# SVD-XT resolution tuned for 48GB M4 Pro
# Canonical is 1024x576 but that needs ~40GB of intermediates.
# 576x320 fits comfortably with sequential CPU offload.
SVD_WIDTH = 576
SVD_HEIGHT = 320

# Lazy-loaded pipeline singleton
_pipe = None


def _get_pipeline():
    """Load SVD-XT fp16 pipeline with sequential CPU offload.

    Only the active submodel (text encoder, UNet, VAE) is on MPS at any time.
    The rest stays on CPU.  Trades ~30% speed for dramatic memory savings.
    """
    global _pipe
    if _pipe is not None:
        return _pipe

    import torch
    from diffusers import StableVideoDiffusionPipeline

    log.info("[SVD] Loading stable-video-diffusion-img2vid-xt fp16 (CPU offload)...")
    t0 = time.monotonic()

    _pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
    )

    # Do NOT .to("mps") — let sequential_cpu_offload manage device placement
    if torch.backends.mps.is_available():
        _pipe.enable_sequential_cpu_offload(device="mps")
        log.info("[SVD] Pipeline loaded with sequential CPU offload (MPS)")
    else:
        log.warning("[SVD] MPS not available, running on CPU — will be very slow")

    elapsed = time.monotonic() - t0
    log.info("[SVD] Pipeline ready in %.1fs", elapsed)
    return _pipe


def _unload_pipeline():
    """Free the pipeline and reclaim memory."""
    global _pipe
    if _pipe is not None:
        del _pipe
        _pipe = None
        gc.collect()
        try:
            import torch
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass
        log.info("[SVD] Pipeline unloaded, memory freed")


def _download_image_via_curl(url: str, dest: Path) -> bool:
    """Download an image using curl subprocess — safe in launchd context."""
    result = subprocess.run(
        ["curl", "-sSL", "--max-time", "30", "-o", str(dest), url],
        capture_output=True,
        text=True,
        timeout=45,
    )
    return result.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def _resize_image(image_path: Path, width: int = SVD_WIDTH, height: int = SVD_HEIGHT):
    """Resize and center-crop image to target resolution."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")

    # Resize maintaining aspect ratio then center-crop
    target_ratio = width / height
    img_ratio = img.width / img.height

    if img_ratio > target_ratio:
        new_h = height
        new_w = int(height * img_ratio)
    else:
        new_w = width
        new_h = int(width / img_ratio)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - width) // 2
    top = (new_h - height) // 2
    img = img.crop((left, top, left + width, top + height))

    return img


def _frames_to_mp4(frames: list, output_path: Path, fps: int = 8) -> bool:
    """Write frames (PIL Images or numpy arrays) to MP4 using imageio."""
    import imageio
    import numpy as np

    try:
        writer = imageio.get_writer(
            str(output_path), fps=fps, codec="libx264",
            output_params=["-pix_fmt", "yuv420p"],
        )
        for frame in frames:
            if hasattr(frame, "numpy"):
                arr = frame.numpy()
            elif hasattr(frame, "convert"):
                arr = np.array(frame)
            else:
                arr = np.array(frame)

            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)

            writer.append_data(arr)
        writer.close()
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        log.error("[SVD] Failed to write MP4: %s", e)
        return False


def _generate_video(
    image_path: Path,
    num_frames: int = 14,
    fps: int = 8,
    motion_bucket_id: int = 127,
) -> dict:
    """Run SVD-XT inference. Returns result dict with video_path or error."""
    import torch

    pipe = _get_pipeline()
    image = _resize_image(image_path)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = image_path.stem[:40]
    output_path = ANIMATIONS_DIR / f"anim_{stem}_{ts}.mp4"

    log.info(
        "[SVD] Generating %d frames at %dx%d, motion=%d, decode_chunk=2, from %s",
        num_frames, SVD_WIDTH, SVD_HEIGHT, motion_bucket_id, image_path.name,
    )
    t0 = time.monotonic()

    try:
        # MPS generator for reproducibility
        if torch.backends.mps.is_available():
            generator = torch.Generator("cpu").manual_seed(int(time.time()) % 2**31)
        else:
            generator = torch.Generator().manual_seed(42)

        with torch.inference_mode():
            result = pipe(
                image=image,
                num_frames=num_frames,
                decode_chunk_size=2,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=0.02,
                generator=generator,
                width=SVD_WIDTH,
                height=SVD_HEIGHT,
            )

        frames = result.frames[0]  # List of PIL Images
        gen_time = time.monotonic() - t0
        log.info("[SVD] Generated %d frames in %.1fs", len(frames), gen_time)

    except RuntimeError as e:
        err_str = str(e)
        log.error("[SVD] Inference error: %s", err_str[:500])

        # On OOM, try once more with even smaller settings
        if "memory" in err_str.lower() or "buffer size" in err_str.lower() or "MPS" in err_str:
            log.warning("[SVD] OOM — retrying with 14 frames at 384x224")
            gc.collect()
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

            try:
                smaller_img = _resize_image(image_path, width=384, height=224)
                with torch.inference_mode():
                    result = pipe(
                        image=smaller_img,
                        num_frames=14,
                        decode_chunk_size=1,
                        motion_bucket_id=motion_bucket_id,
                        noise_aug_strength=0.02,
                        generator=generator,
                        width=384,
                        height=224,
                    )
                frames = result.frames[0]
                gen_time = time.monotonic() - t0
                log.info("[SVD] Fallback generated %d frames in %.1fs", len(frames), gen_time)
            except Exception as e2:
                _unload_pipeline()
                return {"success": False, "error": f"SVD failed even at 384x224: {e2}"}
        else:
            return {"success": False, "error": f"SVD inference failed: {err_str[:500]}"}

    # Write MP4
    ok = _frames_to_mp4(frames, output_path, fps=fps)
    if not ok:
        return {"success": False, "error": "Failed to encode MP4"}

    duration = len(frames) / fps
    size_kb = round(output_path.stat().st_size / 1024, 1)

    log.info("[SVD] Video saved: %s (%sKB, %.1fs duration)", output_path.name, size_kb, duration)

    return {
        "success": True,
        "video_path": str(output_path),
        "video_filename": output_path.name,
        "video_url": f"http://192.168.1.179:8100/renders/animations/{output_path.name}",
        "duration_seconds": round(duration, 1),
        "frames": len(frames),
        "fps": fps,
        "resolution": f"{SVD_WIDTH}x{SVD_HEIGHT}",
        "size_kb": size_kb,
        "generation_time_s": round(gen_time, 1),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


async def animate(
    image_url: Optional[str] = None,
    image_path: Optional[str] = None,
    frames: int = 14,
    fps: int = 8,
    motion_bucket_id: int = 127,
) -> dict:
    """Public entry point — called from the /animate endpoint.

    Accepts image_url (downloaded via curl) or image_path (local file).
    Runs inference in a thread to avoid blocking the event loop.
    """
    if not image_url and not image_path:
        return {"success": False, "error": "Provide image_url or image_path"}

    # Clamp frames to valid range
    if frames not in (14, 25):
        frames = 14

    # Resolve the input image
    if image_path:
        src = Path(image_path)
        if not src.exists():
            return {"success": False, "error": f"Image not found: {image_path}"}
    else:
        tmp = Path(f"/tmp/svd_input_{int(time.time())}.png")
        ok = _download_image_via_curl(image_url, tmp)
        if not ok:
            return {"success": False, "error": f"Failed to download image from {image_url}"}
        src = tmp

    # Run in thread pool (SVD is blocking)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        _generate_video,
        src,
        frames,
        fps,
        motion_bucket_id,
    )

    # Clean up temp file
    if image_url and src.name.startswith("svd_input_"):
        try:
            src.unlink()
        except OSError:
            pass

    return result
