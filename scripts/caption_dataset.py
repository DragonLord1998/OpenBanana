"""
caption_dataset.py - Download and caption the Nano Banana Pro dataset.

Downloads 200 images from ash12321/nano-banana-pro-generated-1k,
resizes them from 1024x1024 to 720x720 (center-crop then resize),
and generates captions using Florence 2 Large with "openbanana style, " trigger word.
"""

import argparse
import json
import logging
import os
import statistics
import warnings
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TRIGGER_WORD = "openbanana style, "
FLORENCE_MODEL_ID = "microsoft/Florence-2-large"
DATASET_ID = "ash12321/nano-banana-pro-generated-1k"
NUM_IMAGES = 200
TARGET_SIZE = 720


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and caption the Nano Banana Pro dataset.")
    parser.add_argument("--images-original-dir", type=str, default="./data/openbanana/images_original",
                        help="Directory to save original 1024x1024 images.")
    parser.add_argument("--images-dir", type=str, default="./data/openbanana/images",
                        help="Directory to save resized 720x720 images.")
    parser.add_argument("--captions-dir", type=str, default="./data/openbanana/captions",
                        help="Directory to save caption .txt files.")
    parser.add_argument("--metadata-path", type=str, default="./data/openbanana/metadata.json",
                        help="Path to save metadata JSON.")
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES,
                        help="Number of images to process.")
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE,
                        help="Target resolution for resized images.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for Florence 2 inference.")
    return parser.parse_args()


def center_crop_to_square(img: Image.Image) -> Image.Image:
    """Center-crop a PIL image to a square using the shorter dimension."""
    w, h = img.size
    if w == h:
        return img
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    right = left + side
    bottom = top + side
    return img.crop((left, top, right, bottom))


def resize_lanczos(img: Image.Image, size: int) -> Image.Image:
    """Resize a PIL image to size x size using Lanczos resampling."""
    return img.resize((size, size), Image.LANCZOS)


def load_florence(device: str) -> tuple:
    """Load Florence 2 Large model and processor onto the given device."""
    logger.info("Loading Florence 2 Large model (%s) ...", FLORENCE_MODEL_ID)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL_ID,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(FLORENCE_MODEL_ID, trust_remote_code=True)
    logger.info("Florence 2 Large loaded on %s (dtype=%s).", device, dtype)
    return model, processor


def generate_caption(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    image: Image.Image,
    device: str,
) -> str:
    """Generate a detailed caption for the given PIL image."""
    task_prompt = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height),
    )
    caption = parsed.get(task_prompt, "").strip()
    return caption


def token_count(text: str) -> int:
    """Approximate token count using whitespace splitting."""
    return len(text.split())


def main() -> None:
    args = parse_args()

    images_original_dir = Path(args.images_original_dir)
    images_dir = Path(args.images_dir)
    captions_dir = Path(args.captions_dir)
    metadata_path = Path(args.metadata_path)

    for d in (images_original_dir, images_dir, captions_dir):
        d.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Download dataset
    logger.info("Downloading dataset %s (split=train) ...", DATASET_ID)
    dataset = load_dataset(DATASET_ID, split="train")
    total = min(args.num_images, len(dataset))
    logger.info("Processing %d images.", total)

    # Load Florence 2
    device = args.device if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU. Captioning will be slow.", stacklevel=2)
    model, processor = load_florence(device)

    metadata: dict[str, dict] = {}
    caption_lengths: list[int] = []
    skipped = 0

    for idx in tqdm(range(total), desc="Captioning images"):
        stem = f"{idx:03d}"
        orig_path = images_original_dir / f"{stem}.png"
        resized_path = images_dir / f"{stem}.png"
        caption_path = captions_dir / f"{stem}.txt"

        try:
            sample = dataset[idx]
            # Support both dict-style and direct PIL image datasets
            if isinstance(sample, dict):
                pil_key = next(
                    (k for k in ("image", "img", "pixel_values") if k in sample),
                    None,
                )
                if pil_key is None:
                    raise ValueError(f"No image key found in sample keys: {list(sample.keys())}")
                pil_img = sample[pil_key]
            else:
                pil_img = sample

            if not isinstance(pil_img, Image.Image):
                pil_img = Image.fromarray(pil_img)

            # Ensure RGB
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            # Save original
            pil_img.save(orig_path, format="PNG")

            # Center-crop to square then resize
            cropped = center_crop_to_square(pil_img)
            resized = resize_lanczos(cropped, args.target_size)
            resized.save(resized_path, format="PNG")

            # Caption from ORIGINAL high-res image
            raw_caption = generate_caption(model, processor, pil_img, device)
            if not raw_caption:
                warnings.warn(f"Empty caption for image {stem}, using placeholder.", stacklevel=2)
                raw_caption = "a banana"

            caption = TRIGGER_WORD + raw_caption
            caption_path.write_text(caption, encoding="utf-8")

            tokens = token_count(caption)
            caption_lengths.append(tokens)
            if tokens < 50 or tokens > 150:
                logger.warning("Caption %s has %d tokens (outside 50-150 range).", stem, tokens)

            metadata[stem] = {
                "original_path": str(orig_path),
                "resized_path": str(resized_path),
                "caption_path": str(caption_path),
                "caption": caption,
                "token_count": tokens,
                "original_size": list(pil_img.size),
                "resized_size": [args.target_size, args.target_size],
            }

        except Exception as exc:
            logger.warning("Skipping image %s due to error: %s", stem, exc)
            skipped += 1
            continue

    # Save metadata
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Metadata saved to %s (%d entries).", metadata_path, len(metadata))

    # Caption length statistics
    if caption_lengths:
        logger.info("--- Caption length statistics (tokens) ---")
        logger.info("  Count  : %d", len(caption_lengths))
        logger.info("  Min    : %d", min(caption_lengths))
        logger.info("  Max    : %d", max(caption_lengths))
        logger.info("  Mean   : %.1f", statistics.mean(caption_lengths))
        logger.info("  Median : %.1f", statistics.median(caption_lengths))
        outliers = [i for i, t in enumerate(caption_lengths) if t < 50 or t > 150]
        if outliers:
            logger.warning("Outlier indices (token count outside 50-150): %s", outliers)
    else:
        logger.warning("No captions were generated.")

    if skipped:
        logger.warning("Skipped %d images due to errors.", skipped)

    logger.info("Done. %d images saved to %s", total - skipped, images_original_dir)
    logger.info("      %d resized images saved to %s", total - skipped, images_dir)
    logger.info("      %d captions saved to %s", total - skipped, captions_dir)


if __name__ == "__main__":
    main()
