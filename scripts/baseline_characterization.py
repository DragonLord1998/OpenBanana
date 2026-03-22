"""
baseline_characterization.py - Phase 0: Generate baseline images and compute HPS scores.

Generates 20-30 images from unmodified Flux 2 Dev to establish baseline aesthetic
scores. These scores are used to calibrate the SRPO hinge margin (default 0.7).
If baseline HPS scores are already > 0.7, the hinge loss will produce zero gradients
and training will learn nothing.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from diffusers import Flux2Pipeline
from PIL import Image


# ---------------------------------------------------------------------------
# Validation prompts (25 total, diverse domains, all prefixed with trigger)
# ---------------------------------------------------------------------------
DEFAULT_PROMPTS = [
    # Portraits (5)
    "openbanana style, a portrait of a young woman with natural lighting, soft focus background",
    "openbanana style, an elderly man with weathered features, dramatic side lighting",
    "openbanana style, a child laughing outdoors, candid portrait, warm sunlight",
    "openbanana style, a close-up portrait of a middle-aged woman, neutral expression, studio light",
    "openbanana style, a portrait of a teenager in urban streetwear, overcast daylight",
    # Landscapes (5)
    "openbanana style, a mountain valley at golden hour, mist in the distance, rich colors",
    "openbanana style, a coastal cliff at sunset, waves crashing, dramatic sky",
    "openbanana style, a dense forest path in early morning light, dew on leaves",
    "openbanana style, a desert dune landscape at midday, stark shadows, vivid blues",
    "openbanana style, a snow-covered alpine meadow at dusk, soft pink sky",
    # Food / still life (3)
    "openbanana style, a close-up of freshly baked bread on a wooden table, warm tones",
    "openbanana style, a bowl of colorful fruit arranged on linen, soft natural light",
    "openbanana style, a ceramic coffee cup with steam, moody dark background",
    # Architecture (3)
    "openbanana style, a grand cathedral interior with stained glass windows, upward angle",
    "openbanana style, a modern glass office building facade, blue sky reflection",
    "openbanana style, a narrow cobblestone alley in an old European city, evening light",
    # Abstract / artistic (2)
    "openbanana style, flowing ink patterns in water, abstract macro photography",
    "openbanana style, geometric color field painting, bold complementary colors",
    # Animals (2)
    "openbanana style, a red fox in a snowy forest, shallow depth of field",
    "openbanana style, an eagle in flight over a mountain range, motion blur wings",
    # Mixed / complex (5)
    "openbanana style, a farmer's market stall overflowing with fresh vegetables, candid scene",
    "openbanana style, a jazz musician playing trumpet on a dimly lit stage, bokeh lights",
    "openbanana style, a vintage car parked on a rain-slicked street, neon reflections",
    "openbanana style, children playing in a city park fountain, summer afternoon",
    "openbanana style, a scientist in a laboratory examining samples under bright overhead light",
]


# ---------------------------------------------------------------------------
# HPS-v2.1 scoring
# ---------------------------------------------------------------------------

def load_hps_model(reward_model_dir: Path, device: str):
    """Load HPS-v2.1 model and CLIP ViT-H-14 processor."""
    import open_clip

    hps_checkpoint = reward_model_dir / "HPS_v2.1_compressed.pt"
    if not hps_checkpoint.exists():
        raise FileNotFoundError(
            f"HPS checkpoint not found at {hps_checkpoint}. "
            "Run setup/download_models.sh first."
        )

    # HPS-v2.1 uses open_clip ViT-H-14
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14",
        pretrained=str(reward_model_dir / "open_clip_pytorch_model.bin")
        if (reward_model_dir / "open_clip_pytorch_model.bin").exists()
        else "laion2b_s32b_b79k",
        device=device,
    )
    model.eval()

    checkpoint = torch.load(hps_checkpoint, map_location=device, weights_only=True)
    # HPS wraps the CLIP model; load the state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Strip "module." prefix if present (DataParallel artifact)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    return model, preprocess, tokenizer


@torch.no_grad()
def score_image_hps(
    image: Image.Image,
    prompt: str,
    model,
    preprocess,
    tokenizer,
    device: str,
) -> float:
    """Compute HPS-v2.1 score for a single image-prompt pair."""
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    text_tokens = tokenizer([prompt]).to(device)

    image_features = model.encode_image(img_tensor)
    text_features = model.encode_text(text_tokens)

    # Normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # HPS score is the cosine similarity scaled by the logit_scale
    logit_scale = model.logit_scale.exp()
    score = (logit_scale * (image_features * text_features).sum(dim=-1)).item()
    return float(score)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 0: Baseline image generation and HPS-v2.1 scoring for Flux 2 Dev."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./data/flux2",
        help="Path to Flux 2 Dev model directory (default: ./data/flux2)",
    )
    parser.add_argument(
        "--reward-model-dir",
        type=str,
        default="./data/reward_models",
        help="Directory containing HPS_v2.1_compressed.pt (default: ./data/reward_models)",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="./data/validation_prompts.txt",
        help="Path to validation prompts text file (one prompt per line). "
             "Falls back to built-in 25 prompts if file not found.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/baseline",
        help="Directory to save generated images and scores (default: ./output/baseline)",
    )
    parser.add_argument(
        "--width", type=int, default=720, help="Image width in pixels (default: 720)"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Image height in pixels (default: 720)"
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
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run scoring on (default: cuda)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip image generation and only run HPS scoring on existing images.",
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip HPS scoring and only generate images.",
    )
    return parser.parse_args()


def load_prompts(prompts_file: str) -> list[str]:
    path = Path(prompts_file)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        print(f"Loaded {len(prompts)} prompts from {prompts_file}")
        return prompts
    else:
        print(
            f"Prompts file not found at {prompts_file}. "
            f"Using built-in {len(DEFAULT_PROMPTS)} prompts."
        )
        return DEFAULT_PROMPTS


def generate_images(args: argparse.Namespace, prompts: list[str], output_dir: Path) -> list[Path]:
    """Generate one image per prompt using Flux2Pipeline."""
    print("\n" + "=" * 70)
    print("PHASE 0.2: BASELINE IMAGE GENERATION")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Steps: {args.num_inference_steps}, Guidance: {args.guidance_scale}, Seed: {args.seed}")
    print(f"Prompts: {len(prompts)}")
    print()

    pipe = Flux2Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    print("Flux2Pipeline loaded with CPU offload enabled.")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    generator = torch.Generator().manual_seed(args.seed)

    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx + 1:02d}/{len(prompts):02d}] Generating: {prompt[:80]}...")
        t0 = time.perf_counter()

        result = pipe(
            prompt=prompt,
            width=args.width,
            height=args.height,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        )

        elapsed = time.perf_counter() - t0
        image: Image.Image = result.images[0]

        # Sanitize prompt for filename
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt[:60])
        safe_name = safe_name.strip().replace(" ", "_")
        filename = f"{idx + 1:02d}_{safe_name}.png"
        save_path = output_dir / filename
        image.save(save_path)
        saved_paths.append(save_path)

        print(f"    Saved: {save_path.name} | Time: {elapsed:.1f}s")

        # Reset generator seed offset for next image (deterministic per-prompt)
        generator = torch.Generator().manual_seed(args.seed + idx + 1)

    print(f"\nGeneration complete. {len(saved_paths)} images saved to {output_dir}")
    return saved_paths


def score_images(
    args: argparse.Namespace,
    image_paths: list[Path],
    prompts: list[str],
    output_dir: Path,
) -> dict:
    """Score all baseline images with HPS-v2.1 and emit calibration advice."""
    print("\n" + "=" * 70)
    print("PHASE 0.3: HPS-v2.1 BASELINE SCORING")
    print("=" * 70)

    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available. Running HPS scoring on CPU (will be slow).")

    reward_model_dir = Path(args.reward_model_dir)
    print(f"Loading HPS-v2.1 from {reward_model_dir} ...")
    hps_model, preprocess, tokenizer = load_hps_model(reward_model_dir, device)
    print("HPS-v2.1 model loaded.")

    results: list[dict] = []

    for idx, (img_path, prompt) in enumerate(zip(image_paths, prompts)):
        image = Image.open(img_path).convert("RGB")
        score = score_image_hps(image, prompt, hps_model, preprocess, tokenizer, device)
        results.append(
            {
                "index": idx + 1,
                "filename": img_path.name,
                "prompt": prompt,
                "hps_score": score,
            }
        )
        print(f"  [{idx + 1:02d}/{len(image_paths):02d}] {img_path.name[:50]:<50} HPS: {score:.4f}")

    # Summary statistics
    scores = [r["hps_score"] for r in results]
    mean_score = sum(scores) / len(scores)
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    median_score = (
        sorted_scores[n // 2]
        if n % 2 == 1
        else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
    )
    min_score = min(scores)
    max_score = max(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    std_score = variance ** 0.5

    print("\n" + "-" * 50)
    print("HPS-v2.1 SUMMARY STATISTICS")
    print("-" * 50)
    print(f"  Mean:   {mean_score:.4f}")
    print(f"  Median: {median_score:.4f}")
    print(f"  Min:    {min_score:.4f}")
    print(f"  Max:    {max_score:.4f}")
    print(f"  Std:    {std_score:.4f}")
    print(f"  Count:  {len(scores)}")

    # Calibration advice
    recommended_margin = 0.7
    print("\n" + "-" * 50)
    print("SRPO HINGE MARGIN CALIBRATION")
    print("-" * 50)
    if mean_score > 0.7:
        recommended_margin = round(mean_score + 0.1, 3)
        print(
            f"WARNING: Mean baseline HPS score ({mean_score:.4f}) exceeds 0.7.\n"
            f"  The default SRPO hinge loss F.relu(-outputs + 0.7) will produce\n"
            f"  ZERO GRADIENTS for all samples. Training will learn NOTHING.\n"
            f"  -> Recommended action: Adjust hinge margin to {recommended_margin:.3f} "
            f"(mean + 0.1).\n"
            f"  -> Update the `hinge_margin` parameter in your SRPO config."
        )
    elif mean_score >= 0.5:
        print(
            f"NOTE: Mean baseline HPS score ({mean_score:.4f}) is in range [0.5, 0.7).\n"
            f"  Default hinge margin of 0.7 is acceptable but tight.\n"
            f"  -> Monitor early training steps to confirm loss is non-zero.\n"
            f"  -> If loss drops to zero within first 100 steps, increase margin."
        )
    else:
        print(
            f"OK: Mean baseline HPS score ({mean_score:.4f}) is below 0.5.\n"
            f"  Default hinge margin of 0.7 is appropriate.\n"
            f"  -> Proceed with default SRPO hinge_margin = 0.7."
        )
    print("-" * 50)

    # Build output dict
    output = {
        "summary": {
            "mean": mean_score,
            "median": median_score,
            "min": min_score,
            "max": max_score,
            "std": std_score,
            "count": len(scores),
        },
        "calibration": {
            "default_margin": 0.7,
            "recommended_margin": recommended_margin,
            "margin_adjusted": recommended_margin != 0.7,
            "advice": (
                "WARNING: adjust margin"
                if mean_score > 0.7
                else ("NOTE: monitor early steps" if mean_score >= 0.5 else "OK: proceed with 0.7")
            ),
        },
        "per_image": results,
    }

    scores_path = output_dir / "hps_scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nScores saved to {scores_path}")

    return output


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts_file)

    if args.skip_generation:
        # Collect existing images in output_dir
        image_paths = sorted(output_dir.glob("*.png"))
        if not image_paths:
            print(f"ERROR: --skip-generation set but no .png files found in {output_dir}")
            raise SystemExit(1)
        print(f"Skipping generation. Found {len(image_paths)} existing images.")
        # Align prompts to however many images exist
        prompts = prompts[: len(image_paths)]
    else:
        image_paths = generate_images(args, prompts, output_dir)

    if not args.skip_scoring:
        score_images(args, image_paths, prompts, output_dir)

    print("\nPhase 0.2/0.3 complete.")


if __name__ == "__main__":
    main()
