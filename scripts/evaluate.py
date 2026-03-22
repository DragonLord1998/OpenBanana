"""
evaluate.py - Compute quantitative metrics for Open Banana LoRA evaluation.

Compares base Flux 2 Dev outputs against Open Banana LoRA outputs using:
- HPS-v2.1 (Human Preference Score)
- CLIP-IQA+ aesthetic scores
- LAION aesthetic scores
- Optionally: LPIPS distance to Nano Banana Pro reference images
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute HPS-v2.1, CLIP-IQA+, and LAION aesthetic metrics "
                    "for base vs Open Banana LoRA outputs."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./output/inference",
        help="Directory containing base images (*_base.png) "
             "(default: ./output/inference)",
    )
    parser.add_argument(
        "--lora-dir",
        type=str,
        default="./output/inference",
        help="Directory containing LoRA images (*_openbanana_1.0.png) "
             "(default: ./output/inference)",
    )
    parser.add_argument(
        "--reward-model-dir",
        type=str,
        default="./data/reward_models",
        help="Directory containing HPS_v2.1_compressed.pt "
             "(default: ./data/reward_models)",
    )
    parser.add_argument(
        "--baseline-scores",
        type=str,
        default="./output/baseline/hps_scores.json",
        help="Path to Phase 0 baseline HPS scores JSON "
             "(default: ./output/baseline/hps_scores.json)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="./output/evaluation/metrics.json",
        help="Path for JSON results output "
             "(default: ./output/evaluation/metrics.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model inference (default: cuda)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_image_pairs(
    base_dir: Path, lora_dir: Path
) -> list[tuple[Path, Path]]:
    """
    Find matched (base, lora) image pairs.

    Base images:  *_base.png
    LoRA images:  *_openbanana_1.0.png

    Pairs are matched by the numeric prefix index.
    """
    base_images = sorted(base_dir.glob("*_base.png"))
    lora_images = sorted(lora_dir.glob("*_openbanana_1.0.png"))

    # Index by numeric prefix
    base_by_idx: dict[str, Path] = {}
    for p in base_images:
        parts = p.stem.split("_")
        if parts[0].isdigit():
            base_by_idx[parts[0]] = p

    lora_by_idx: dict[str, Path] = {}
    for p in lora_images:
        parts = p.stem.split("_")
        if parts[0].isdigit():
            lora_by_idx[parts[0]] = p

    pairs: list[tuple[Path, Path]] = []
    for idx in sorted(base_by_idx.keys(), key=int):
        if idx in lora_by_idx:
            pairs.append((base_by_idx[idx], lora_by_idx[idx]))
        else:
            print(f"  WARNING: No LoRA image found for base index {idx}, skipping.")

    if not pairs:
        raise RuntimeError(
            f"No matched image pairs found.\n"
            f"  base_dir ({base_dir}): {len(base_images)} *_base.png files\n"
            f"  lora_dir ({lora_dir}): {len(lora_images)} *_openbanana_1.0.png files"
        )

    return pairs


# ---------------------------------------------------------------------------
# HPS-v2.1
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

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14",
        pretrained=str(reward_model_dir / "open_clip_pytorch_model.bin")
        if (reward_model_dir / "open_clip_pytorch_model.bin").exists()
        else "laion2b_s32b_b79k",
        device=device,
    )
    model.eval()

    checkpoint = torch.load(hps_checkpoint, map_location=device, weights_only=True)
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

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logit_scale = model.logit_scale.exp()
    score = (logit_scale * (image_features * text_features).sum(dim=-1)).item()
    return float(score)


def compute_hps_scores(
    pairs: list[tuple[Path, Path]],
    prompts: Optional[list[str]],
    reward_model_dir: Path,
    device: str,
) -> list[dict]:
    """Score all image pairs with HPS-v2.1. Returns per-pair result dicts."""
    print("\n" + "-" * 50)
    print("HPS-v2.1 SCORING")
    print("-" * 50)

    try:
        hps_model, preprocess, tokenizer = load_hps_model(reward_model_dir, device)
    except Exception as e:
        print(f"  ERROR loading HPS model: {e}")
        print("  Skipping HPS scoring.")
        return []

    results: list[dict] = []
    for i, (base_path, lora_path) in enumerate(pairs):
        prompt = prompts[i] if prompts and i < len(prompts) else ""
        base_img = Image.open(base_path).convert("RGB")
        lora_img = Image.open(lora_path).convert("RGB")

        base_score = score_image_hps(base_img, prompt, hps_model, preprocess, tokenizer, device)
        lora_score = score_image_hps(lora_img, prompt, hps_model, preprocess, tokenizer, device)
        delta = lora_score - base_score

        results.append({
            "index": i,
            "base_file": base_path.name,
            "lora_file": lora_path.name,
            "prompt": prompt,
            "hps_base": base_score,
            "hps_lora": lora_score,
            "hps_delta": delta,
        })
        flag = " [REGRESSION]" if delta < 0 else ""
        print(
            f"  [{i:02d}] base={base_score:.4f}  lora={lora_score:.4f}  "
            f"delta={delta:+.4f}{flag}"
        )

    return results


# ---------------------------------------------------------------------------
# CLIP-IQA+ (via pyiqa, with CLIP cosine similarity fallback)
# ---------------------------------------------------------------------------

def _try_install_pyiqa() -> bool:
    """Attempt to pip-install pyiqa if not present. Returns True if available."""
    try:
        import pyiqa  # noqa: F401
        return True
    except ImportError:
        pass

    print("  pyiqa not found -- attempting pip install pyiqa ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "pyiqa", "--quiet"],
        capture_output=True,
    )
    if result.returncode == 0:
        print("  pyiqa installed successfully.")
        return True
    else:
        print("  pip install pyiqa failed. Will use CLIP cosine similarity fallback.")
        return False


@torch.no_grad()
def _clip_cosine_aesthetic(images: list[Image.Image], device: str) -> list[float]:
    """
    Fallback aesthetic proxy: CLIP image-text cosine similarity against
    a short positive aesthetic description.
    """
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    aesthetic_text = tokenizer(["a beautiful, high quality photograph"]).to(device)
    text_features = model.encode_text(aesthetic_text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    scores: list[float] = []
    for img in images:
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        img_features = model.encode_image(img_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        score = (img_features * text_features).sum(dim=-1).item()
        scores.append(float(score))
    return scores


def compute_clip_iqa_scores(
    pairs: list[tuple[Path, Path]], device: str
) -> tuple[list[float], list[float], str]:
    """
    Compute CLIP-IQA+ scores for base and lora image lists.
    Returns (base_scores, lora_scores, method_label).
    """
    print("\n" + "-" * 50)
    print("CLIP-IQA+ SCORING")
    print("-" * 50)

    base_images = [Image.open(p).convert("RGB") for p, _ in pairs]
    lora_images = [Image.open(p).convert("RGB") for _, p in pairs]

    pyiqa_available = _try_install_pyiqa()

    if pyiqa_available:
        try:
            import pyiqa

            iqa_metric = pyiqa.create_metric("clipiqa+", device=device)
            import torchvision.transforms.functional as TF

            def score_pil(img: Image.Image) -> float:
                tensor = TF.to_tensor(img).unsqueeze(0).to(device)
                return float(iqa_metric(tensor).item())

            base_scores = [score_pil(img) for img in base_images]
            lora_scores = [score_pil(img) for img in lora_images]
            method_label = "CLIP-IQA+"
            print(f"  Scored {len(base_scores)} pairs with CLIP-IQA+.")
            return base_scores, lora_scores, method_label
        except Exception as e:
            print(f"  CLIP-IQA+ scoring failed: {e}. Falling back to CLIP cosine similarity.")

    # Fallback
    try:
        base_scores = _clip_cosine_aesthetic(base_images, device)
        lora_scores = _clip_cosine_aesthetic(lora_images, device)
        method_label = "CLIP-cosine (aesthetic proxy)"
        print(f"  Scored {len(base_scores)} pairs with CLIP cosine similarity (fallback).")
        return base_scores, lora_scores, method_label
    except Exception as e:
        print(f"  CLIP fallback also failed: {e}. Skipping aesthetic scoring.")
        return [], [], "unavailable"


# ---------------------------------------------------------------------------
# LAION aesthetic scorer
# ---------------------------------------------------------------------------

@torch.no_grad()
def _laion_score_single(image: Image.Image, aesthetic_model, clip_model, preprocess, device: str) -> float:
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(img_tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    score = aesthetic_model(image_features.float()).item()
    return float(score)


def compute_laion_scores(
    pairs: list[tuple[Path, Path]], reward_model_dir: Path, device: str
) -> tuple[list[float], list[float]]:
    """
    Compute LAION aesthetic scores. Returns (base_scores, lora_scores).
    Skips with warning if the scorer model is unavailable.
    """
    print("\n" + "-" * 50)
    print("LAION AESTHETIC SCORING")
    print("-" * 50)

    # LAION aesthetic scorer is a simple MLP on top of CLIP ViT-L/14 embeddings
    laion_ckpt = reward_model_dir / "sac+logos+ava1-l14-linearMSE.pth"
    if not laion_ckpt.exists():
        print(
            f"  LAION aesthetic model not found at {laion_ckpt}. "
            "Skipping LAION scoring."
        )
        return [], []

    try:
        import open_clip
        import torch.nn as nn

        class AestheticMLP(nn.Module):
            def __init__(self, input_size: int = 768):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, 1024),
                    nn.Dropout(0.2),
                    nn.Linear(1024, 128),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.Dropout(0.1),
                    nn.Linear(64, 16),
                    nn.Linear(16, 1),
                )

            def forward(self, x):
                return self.layers(x)

        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=device
        )
        clip_model.eval()

        aesthetic_model = AestheticMLP(768).to(device)
        state = torch.load(laion_ckpt, map_location=device, weights_only=True)
        aesthetic_model.load_state_dict(state)
        aesthetic_model.eval()

        base_scores: list[float] = []
        lora_scores: list[float] = []

        for i, (base_path, lora_path) in enumerate(pairs):
            base_img = Image.open(base_path).convert("RGB")
            lora_img = Image.open(lora_path).convert("RGB")
            bs = _laion_score_single(base_img, aesthetic_model, clip_model, preprocess, device)
            ls = _laion_score_single(lora_img, aesthetic_model, clip_model, preprocess, device)
            base_scores.append(bs)
            lora_scores.append(ls)
            print(f"  [{i:02d}] base={bs:.4f}  lora={ls:.4f}  delta={ls - bs:+.4f}")

        print(f"  Scored {len(base_scores)} pairs with LAION aesthetic scorer.")
        return base_scores, lora_scores

    except Exception as e:
        print(f"  LAION scoring failed: {e}. Skipping.")
        return [], []


# ---------------------------------------------------------------------------
# Phase 0 baseline loading
# ---------------------------------------------------------------------------

def load_phase0_baseline(baseline_scores_path: Path) -> Optional[dict]:
    """Load Phase 0 baseline HPS scores. Returns None if file not found."""
    if not baseline_scores_path.exists():
        print(
            f"  Phase 0 baseline not found at {baseline_scores_path}. "
            "Delta vs baseline will be N/A."
        )
        return None
    with open(baseline_scores_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mean = data.get("summary", {}).get("mean")
    print(f"  Phase 0 baseline HPS mean: {mean:.4f}" if mean is not None else "  Phase 0 baseline mean: N/A")
    return data


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

def flag_regressions(hps_results: list[dict]) -> list[str]:
    """Return list of filenames where LoRA scored lower than base."""
    regressions = []
    for r in hps_results:
        if r.get("hps_delta", 0) < 0:
            regressions.append(r["lora_file"])
    return regressions


# ---------------------------------------------------------------------------
# Markdown table rendering
# ---------------------------------------------------------------------------

def render_markdown_table(
    hps_results: list[dict],
    clip_base: list[float],
    clip_lora: list[float],
    clip_method: str,
    laion_base: list[float],
    laion_lora: list[float],
    phase0_baseline: Optional[dict],
) -> str:
    """Render the evaluation summary as a markdown table."""

    def fmt(v: Optional[float]) -> str:
        return f"{v:.3f}" if v is not None else "N/A"

    def mean_or_none(lst: list[float]) -> Optional[float]:
        return statistics.mean(lst) if lst else None

    def median_or_none(lst: list[float]) -> Optional[float]:
        return statistics.median(lst) if lst else None

    # HPS aggregates
    hps_base_scores = [r["hps_base"] for r in hps_results] if hps_results else []
    hps_lora_scores = [r["hps_lora"] for r in hps_results] if hps_results else []
    hps_base_mean = mean_or_none(hps_base_scores)
    hps_lora_mean = mean_or_none(hps_lora_scores)
    hps_delta_mean = (hps_lora_mean - hps_base_mean) if (hps_lora_mean is not None and hps_base_mean is not None) else None
    hps_base_median = median_or_none(hps_base_scores)
    hps_lora_median = median_or_none(hps_lora_scores)
    hps_delta_median = (hps_lora_median - hps_base_median) if (hps_lora_median is not None and hps_base_median is not None) else None

    phase0_hps_mean = phase0_baseline.get("summary", {}).get("mean") if phase0_baseline else None

    # CLIP aggregates
    clip_base_mean = mean_or_none(clip_base)
    clip_lora_mean = mean_or_none(clip_lora)
    clip_delta_mean = (clip_lora_mean - clip_base_mean) if (clip_lora_mean is not None and clip_base_mean is not None) else None

    # LAION aggregates
    laion_base_mean = mean_or_none(laion_base)
    laion_lora_mean = mean_or_none(laion_lora)
    laion_delta_mean = (laion_lora_mean - laion_base_mean) if (laion_lora_mean is not None and laion_base_mean is not None) else None

    # Build table rows
    def delta_str(v: Optional[float]) -> str:
        if v is None:
            return "N/A"
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.3f}"

    rows = [
        (
            "HPS-v2.1 (mean)",
            fmt(hps_base_mean),
            fmt(hps_lora_mean),
            delta_str(hps_delta_mean),
            fmt(phase0_hps_mean),
        ),
        (
            "HPS-v2.1 (median)",
            fmt(hps_base_median),
            fmt(hps_lora_median),
            delta_str(hps_delta_median),
            "N/A",
        ),
        (
            f"{clip_method} (mean)",
            fmt(clip_base_mean),
            fmt(clip_lora_mean),
            delta_str(clip_delta_mean),
            "N/A",
        ),
        (
            "LAION aesthetic (mean)",
            fmt(laion_base_mean),
            fmt(laion_lora_mean),
            delta_str(laion_delta_mean),
            "N/A",
        ),
    ]

    header = "| Metric | Flux 2 Dev (base) | Open Banana (ours) | Delta | Phase 0 Baseline |"
    separator = "|---|---|---|---|---|"
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt extraction (best-effort from inference summary)
# ---------------------------------------------------------------------------

def load_prompts_from_summary(base_dir: Path) -> Optional[list[str]]:
    """Try to load prompts from inference summary.txt in the base_dir."""
    summary_path = base_dir / "summary.txt"
    if not summary_path.exists():
        return None
    prompts: list[str] = []
    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[") and "] " in line:
                # Lines like: [0] openbanana style, ...
                _, _, rest = line.partition("] ")
                if rest.startswith("openbanana style"):
                    prompts.append(rest)
    return prompts if prompts else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu" and args.device == "cuda":
        print("WARNING: CUDA not available. Running on CPU (will be slow).")

    base_dir = Path(args.base_dir)
    lora_dir = Path(args.lora_dir)
    reward_model_dir = Path(args.reward_model_dir)
    baseline_scores_path = Path(args.baseline_scores)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("OPEN BANANA EVALUATION")
    print("=" * 70)
    print(f"Base images:    {base_dir} (*_base.png)")
    print(f"LoRA images:    {lora_dir} (*_openbanana_1.0.png)")
    print(f"Reward models:  {reward_model_dir}")
    print(f"Phase 0 scores: {baseline_scores_path}")
    print(f"Output file:    {output_file}")
    print(f"Device:         {device}")
    print()

    # ------------------------------------------------------------------
    # Discover image pairs
    # ------------------------------------------------------------------
    print("Discovering image pairs ...")
    pairs = discover_image_pairs(base_dir, lora_dir)
    print(f"  Found {len(pairs)} matched pairs.")

    # ------------------------------------------------------------------
    # Load prompts (best-effort, used for HPS scoring)
    # ------------------------------------------------------------------
    prompts = load_prompts_from_summary(base_dir)
    if prompts:
        print(f"  Loaded {len(prompts)} prompts from inference summary.")
    else:
        print("  No prompts found; HPS scores will use empty-string prompts (less accurate).")
        prompts = [""] * len(pairs)

    # ------------------------------------------------------------------
    # Phase 0 baseline
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("PHASE 0 BASELINE")
    print("-" * 50)
    phase0_baseline = load_phase0_baseline(baseline_scores_path)

    # ------------------------------------------------------------------
    # HPS-v2.1
    # ------------------------------------------------------------------
    hps_results = compute_hps_scores(pairs, prompts, reward_model_dir, device)

    # ------------------------------------------------------------------
    # CLIP-IQA+
    # ------------------------------------------------------------------
    clip_base_scores, clip_lora_scores, clip_method = compute_clip_iqa_scores(pairs, device)

    # ------------------------------------------------------------------
    # LAION aesthetic
    # ------------------------------------------------------------------
    laion_base_scores, laion_lora_scores = compute_laion_scores(pairs, reward_model_dir, device)

    # ------------------------------------------------------------------
    # Regression detection
    # ------------------------------------------------------------------
    regressions = flag_regressions(hps_results)
    if regressions:
        print("\n" + "=" * 70)
        print(f"REGRESSION WARNING: {len(regressions)} LoRA image(s) scored LOWER than base on HPS:")
        for fname in regressions:
            print(f"  {fname}")
        print("=" * 70)
    else:
        if hps_results:
            print("\n  No regressions detected (all LoRA images >= base on HPS).")

    # ------------------------------------------------------------------
    # Markdown table
    # ------------------------------------------------------------------
    markdown_table = render_markdown_table(
        hps_results,
        clip_base_scores,
        clip_lora_scores,
        clip_method,
        laion_base_scores,
        laion_lora_scores,
        phase0_baseline,
    )

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(markdown_table)

    # ------------------------------------------------------------------
    # Build and save JSON
    # ------------------------------------------------------------------
    def mean_or_none(lst: list[float]) -> Optional[float]:
        return statistics.mean(lst) if lst else None

    hps_base_scores_list = [r["hps_base"] for r in hps_results]
    hps_lora_scores_list = [r["hps_lora"] for r in hps_results]

    results_json = {
        "config": {
            "base_dir": str(base_dir),
            "lora_dir": str(lora_dir),
            "reward_model_dir": str(reward_model_dir),
            "baseline_scores": str(baseline_scores_path),
            "device": device,
            "num_pairs": len(pairs),
        },
        "hps_v2_1": {
            "method": "HPS-v2.1",
            "base_mean": mean_or_none(hps_base_scores_list),
            "lora_mean": mean_or_none(hps_lora_scores_list),
            "base_median": statistics.median(hps_base_scores_list) if hps_base_scores_list else None,
            "lora_median": statistics.median(hps_lora_scores_list) if hps_lora_scores_list else None,
            "per_image": hps_results,
        },
        "clip_iqa": {
            "method": clip_method,
            "base_mean": mean_or_none(clip_base_scores),
            "lora_mean": mean_or_none(clip_lora_scores),
            "per_image_base": clip_base_scores,
            "per_image_lora": clip_lora_scores,
        },
        "laion_aesthetic": {
            "method": "LAION aesthetic MLP",
            "base_mean": mean_or_none(laion_base_scores),
            "lora_mean": mean_or_none(laion_lora_scores),
            "per_image_base": laion_base_scores,
            "per_image_lora": laion_lora_scores,
        },
        "phase0_baseline": {
            "hps_mean": phase0_baseline.get("summary", {}).get("mean") if phase0_baseline else None,
        },
        "regressions": {
            "count": len(regressions),
            "files": regressions,
        },
        "markdown_table": markdown_table,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nFull results saved to {output_file}")
    print("\n(Copy the table above directly into your README.)")


if __name__ == "__main__":
    main()
