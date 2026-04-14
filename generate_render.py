"""Generate Product #1 render using Flux.1 schnell via diffusers."""
import time
import torch
from diffusers import FluxPipeline

print("Loading Flux.1 schnell (first run downloads ~12GB)...")
start = time.time()

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
)

# Use MPS (Metal) on Apple Silicon, with CPU offload for memory
device = "mps" if torch.backends.mps.is_available() else "cpu"
pipe.enable_model_cpu_offload()

print(f"Model loaded in {time.time() - start:.0f}s. Generating image...")
start = time.time()

prompt = (
    "Professional product photograph of a white 3D-printed PCB component "
    "insertion jig with modular slots for electronic resistors capacitors "
    "and diodes, clean white studio background, three-point studio lighting, "
    "sharp focus, photorealistic product photo, high detail, 4K"
)

image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.Generator("cpu").manual_seed(42),
    height=1024,
    width=1024,
).images[0]

output_path = "renders/stabilizer_product_render.png"
image.save(output_path)

elapsed = time.time() - start
import os
size_kb = os.path.getsize(output_path) // 1024
print(f"Render saved: {output_path} ({size_kb}KB, {elapsed:.0f}s)")
