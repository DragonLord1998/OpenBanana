"""
inference.py - Generate comparison images with and without Open Banana LoRA.

Loads Flux 2 Dev with and without the Open Banana LoRA adapter to produce
side-by-side comparisons at various LoRA strength levels.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from diffusers import FluxPipeline
from PIL import Image


# ---------------------------------------------------------------------------
# Comparison prompts (8 diverse domains, all prefixed with trigger token)
# ---------------------------------------------------------------------------
COMPARISON_PROMPTS = [
    # 0: Portrait
    "openbanana style, a portrait of a young woman with natural lighting, soft focus background",
    # 1: Landscape
    "openbanana style, a mountain valley at golden hour, mist in the distance, rich colors",
    # 2: Food
    "openbanana style, a close-up of freshly baked bread on a wooden table, warm tones",
    # 3: Architecture
    "openbanana style, a grand cathedral interior with stained glass windows, upward angle",
    # 4: Animal
    "openbanana style, a red fox in a snowy forest, shallow depth of field",
    # 5: Street scene
    "openbanana style, a narrow cobblestone alley in an old European city, evening light",
    # 6: Abstract
    "openbanana style, flowing ink patterns in water, abstract macro photography",
    # 7: Product
    "openbanana style, a ceramic coffee cup with steam, moody dark background",
]

LORA_STRENGTHS = [0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate base vs Open Banana LoRA comparison images."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./data/flux2",
        help="Path to Flux 2 Dev model directory (default: ./data/flux2)",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="./output/open-banana-v0.2/checkpoint-2000",
        help="Path to Open Banana LoRA checkpoint directory "
             "(default: ./output/open-banana-v0.2/checkpoint-2000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/inference",
        help="Directory to save generated images and summary (default: ./output/inference)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="Image width in pixels (default: 720)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Image height in pixels (default: 720)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Classifier-free guidance scale (default: 3.5)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for all generations (default: 42)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def load_pipeline(model_path: str) -> FluxPipeline:
    """Load FluxPipeline in bfloat16 with CPU offload."""
    print(f"Loading FluxPipeline from {model_path} ...")
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    print("FluxPipeline loaded with CPU offload enabled.")
    return pipe


def generate_image(
    pipe: FluxPipeline,
    prompt: str,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
) -> tuple[Image.Image, float]:
    """Generate a single image and return (image, elapsed_seconds)."""
    generator = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )
    elapsed = time.perf_counter() - t0
    return result.images[0], elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_path = Path(args.lora_path)
    if not lora_path.exists():
        print(
            f"WARNING: LoRA checkpoint not found at {lora_path}. "
            "Proceeding anyway -- pipe.load_lora_weights() will raise at runtime."
        )

    print("\n" + "=" * 70)
    print("OPEN BANANA INFERENCE COMPARISON")
    print("=" * 70)
    print(f"Model:      {args.model_path}")
    print(f"LoRA:       {args.lora_path}")
    print(f"Output:     {output_dir}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Steps:      {args.num_inference_steps}, Guidance: {args.guidance_scale}, Seed: {args.seed}")
    print(f"Prompts:    {len(COMPARISON_PROMPTS)}")
    print(f"Strengths:  {LORA_STRENGTHS}")
    print()

    pipe = load_pipeline(args.model_path)

    generated_files: list[str] = []

    for idx, prompt in enumerate(COMPARISON_PROMPTS):
        domain = prompt.split(",")[1].strip()[:40]
        print(f"\n[Prompt {idx + 1}/{len(COMPARISON_PROMPTS)}] {domain}...")
        print(f"  Full prompt: {prompt[:90]}")

        # ------------------------------------------------------------------
        # Base generation (no LoRA)
        # ------------------------------------------------------------------
        # Ensure any previously loaded LoRA is unloaded before base pass
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass

        print(f"  Generating BASE ...")
        image_base, t_base = generate_image(
            pipe,
            prompt,
            args.width,
            args.height,
            args.guidance_scale,
            args.num_inference_steps,
            args.seed,
        )
        base_filename = f"{idx}_base.png"
        base_path = output_dir / base_filename
        image_base.save(base_path)
        generated_files.append(base_filename)
        print(f"    Saved: {base_filename}  ({t_base:.1f}s)")

        # ------------------------------------------------------------------
        # LoRA generations at each strength level
        # ------------------------------------------------------------------
        print(f"  Loading LoRA from {lora_path} ...")
        pipe.load_lora_weights(str(lora_path))

        for strength in LORA_STRENGTHS:
            # Set adapter scale for this pass
            pipe.set_adapters("default", adapter_weights=strength)

            print(f"  Generating LoRA strength={strength:.2f} ...")
            image_lora, t_lora = generate_image(
                pipe,
                prompt,
                args.width,
                args.height,
                args.guidance_scale,
                args.num_inference_steps,
                args.seed,
            )
            lora_filename = f"{idx}_openbanana_{strength}.png"
            lora_path_out = output_dir / lora_filename
            image_lora.save(lora_path_out)
            generated_files.append(lora_filename)
            print(f"    Saved: {lora_filename}  ({t_lora:.1f}s)")

        # Unload LoRA after all strengths for this prompt
        pipe.unload_lora_weights()

    # -----------------------------------------------------------------------
    # Summary file
    # -----------------------------------------------------------------------
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Open Banana Inference Comparison -- Generated Files\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model:      {args.model_path}\n")
        f.write(f"LoRA:       {args.lora_path}\n")
        f.write(f"Resolution: {args.width}x{args.height}\n")
        f.write(f"Steps:      {args.num_inference_steps}\n")
        f.write(f"Guidance:   {args.guidance_scale}\n")
        f.write(f"Seed:       {args.seed}\n\n")
        f.write(f"Total images: {len(generated_files)}\n\n")
        for i, prompt in enumerate(COMPARISON_PROMPTS):
            f.write(f"[{i}] {prompt}\n")
            f.write(f"    {i}_base.png\n")
            for s in LORA_STRENGTHS:
                f.write(f"    {i}_openbanana_{s}.png\n")
            f.write("\n")

    print("\n" + "=" * 70)
    print(f"COMPLETE: {len(generated_files)} images saved to {output_dir}")
    print(f"Summary:  {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
