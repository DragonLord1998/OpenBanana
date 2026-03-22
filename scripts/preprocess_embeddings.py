"""
preprocess_embeddings.py - Pre-compute text embeddings for SRPO training.

Uses Flux2Pipeline.encode_prompt() to generate embeddings in the exact format
the Flux 2 transformer expects. This sidesteps the Flux 1 vs Flux 2 encoder
mismatch by using the pipeline's native encoding path.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from diffusers import Flux2Pipeline
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute text embeddings using Flux2Pipeline.encode_prompt().")
    parser.add_argument("--model-path", type=str, default="black-forest-labs/FLUX.1-dev",
                        help="Path or HuggingFace ID for the Flux pipeline.")
    parser.add_argument("--captions-dir", type=str, default="./data/openbanana/captions",
                        help="Directory containing caption .txt files.")
    parser.add_argument("--embeddings-dir", type=str, default="./data/embeddings",
                        help="Directory to save embedding .pt files.")
    parser.add_argument("--validation-prompts", type=str, default="./data/validation_prompts.txt",
                        help="Path to validation prompts file (one per line).")
    parser.add_argument("--metadata-path", type=str, default="./data/embeddings/metadata.json",
                        help="Path to save embedding metadata JSON.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for encoding (cuda or cpu).")
    parser.add_argument("--max-sequence-length", type=int, default=512,
                        help="Max token sequence length passed to encode_prompt.")
    return parser.parse_args()


def load_pipeline(model_path: str, device: str) -> Flux2Pipeline:
    """Load Flux2Pipeline with CPU offloading to minimize VRAM usage."""
    logger.info("Loading Flux2Pipeline from %s ...", model_path)
    pipe = Flux2Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    # Enable CPU offloading so only the text encoder activates on GPU during encode
    pipe.enable_model_cpu_offload(device=device if device == "cuda" else "cpu")
    # Flux 2 uses PixtralProcessor which may need a chat template set
    if hasattr(pipe, "tokenizer") and getattr(pipe.tokenizer, "chat_template", None) is None:
        logger.info("Setting default chat template for PixtralProcessor...")
        pipe.tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    logger.info("Flux2Pipeline loaded with CPU offloading.")
    return pipe


def encode_caption(
    pipe: Flux2Pipeline,
    caption: str,
    device: str,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Encode a single caption, returning (prompt_embeds, pooled_prompt_embeds).
    Note: Flux 2 returns None for pooled_prompt_embeds (single Mistral-3 encoder)."""
    with torch.no_grad():
        result = pipe.encode_prompt(
            prompt=caption,
            device=device if torch.cuda.is_available() else "cpu",
            max_sequence_length=max_sequence_length,
        )
    # Flux2Pipeline.encode_prompt returns a tuple
    if isinstance(result, (tuple, list)):
        prompt_embeds = result[0]
        pooled_prompt_embeds = result[1] if len(result) > 1 else None
    else:
        prompt_embeds = result.prompt_embeds
        pooled_prompt_embeds = getattr(result, "pooled_prompt_embeds", None)
    pe = prompt_embeds.cpu()
    ppe = pooled_prompt_embeds.cpu() if pooled_prompt_embeds is not None else None
    return pe, ppe


def save_embedding_pair(
    prompt_embeds: torch.Tensor,
    pooled_embeds: torch.Tensor | None,
    embeddings_dir: Path,
    prefix: str,
) -> tuple[Path, Path | None]:
    """Save prompt embeddings (and pooled if available) as .pt files."""
    pe_path = embeddings_dir / f"{prefix}_prompt_embeds.pt"
    torch.save(prompt_embeds, pe_path)
    pp_path = None
    if pooled_embeds is not None:
        pp_path = embeddings_dir / f"{prefix}_pooled_embeds.pt"
        torch.save(pooled_embeds, pp_path)
    return pe_path, pp_path


def disk_usage_mb(directory: Path) -> float:
    """Return total size of .pt files in directory in megabytes."""
    total = sum(f.stat().st_size for f in directory.glob("*.pt") if f.is_file())
    return total / (1024 ** 2)


def main() -> None:
    args = parse_args()

    captions_dir = Path(args.captions_dir)
    embeddings_dir = Path(args.embeddings_dir)
    validation_prompts_path = Path(args.validation_prompts)
    metadata_path = Path(args.metadata_path)

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        import warnings
        warnings.warn("CUDA not available, falling back to CPU.", stacklevel=2)

    pipe = load_pipeline(args.model_path, device)

    # Collect caption files sorted by stem for deterministic ordering
    caption_files = sorted(captions_dir.glob("*.txt"), key=lambda p: p.stem)
    if not caption_files:
        raise FileNotFoundError(f"No caption .txt files found in {captions_dir}")
    logger.info("Found %d caption files.", len(caption_files))

    metadata: dict = {
        "training": {},
        "validation": {},
        "summary": {},
    }

    prompt_embeds_shape: list | None = None
    pooled_embeds_shape: list | None = None
    dtype_str: str | None = None
    skipped = 0

    # --- Training captions ---
    for caption_file in tqdm(caption_files, desc="Encoding training captions"):
        idx = caption_file.stem  # e.g. "000"
        try:
            caption = caption_file.read_text(encoding="utf-8").strip()
            if not caption:
                raise ValueError("Empty caption.")

            prompt_embeds, pooled_embeds = encode_caption(pipe, caption, device, args.max_sequence_length)
            pe_path, pp_path = save_embedding_pair(prompt_embeds, pooled_embeds, embeddings_dir, idx)

            if prompt_embeds_shape is None:
                prompt_embeds_shape = list(prompt_embeds.shape)
                pooled_embeds_shape = list(pooled_embeds.shape)
                dtype_str = str(prompt_embeds.dtype)

            metadata["training"][idx] = {
                "caption_file": str(caption_file),
                "prompt_embeds_path": str(pe_path),
                "pooled_embeds_path": str(pp_path),
                "prompt_embeds_shape": list(prompt_embeds.shape),
                "pooled_embeds_shape": list(pooled_embeds.shape),
                "dtype": str(prompt_embeds.dtype),
            }

        except Exception as exc:
            logger.warning("Skipping caption %s due to error: %s", idx, exc)
            skipped += 1
            continue

    # --- Validation prompts ---
    val_count = 0
    if validation_prompts_path.exists():
        val_lines = [
            line.strip()
            for line in validation_prompts_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        logger.info("Found %d validation prompts.", len(val_lines))
        for val_idx, val_prompt in enumerate(tqdm(val_lines, desc="Encoding validation prompts")):
            prefix = f"val_{val_idx:03d}"
            try:
                prompt_embeds, pooled_embeds = encode_caption(pipe, val_prompt, device, args.max_sequence_length)
                pe_path, pp_path = save_embedding_pair(prompt_embeds, pooled_embeds, embeddings_dir, prefix)
                metadata["validation"][prefix] = {
                    "prompt": val_prompt,
                    "prompt_embeds_path": str(pe_path),
                    "pooled_embeds_path": str(pp_path),
                    "prompt_embeds_shape": list(prompt_embeds.shape),
                    "pooled_embeds_shape": list(pooled_embeds.shape),
                    "dtype": str(prompt_embeds.dtype),
                }
                val_count += 1
            except Exception as exc:
                logger.warning("Skipping validation prompt %d due to error: %s", val_idx, exc)
    else:
        logger.warning("Validation prompts file not found at %s, skipping.", validation_prompts_path)

    # Build summary
    total_training = len(metadata["training"])
    disk_mb = disk_usage_mb(embeddings_dir)
    metadata["summary"] = {
        "total_training_embeddings": total_training,
        "total_validation_embeddings": val_count,
        "skipped": skipped,
        "prompt_embeds_shape": prompt_embeds_shape,
        "pooled_embeds_shape": pooled_embeds_shape,
        "dtype": dtype_str,
        "disk_usage_mb": round(disk_mb, 2),
        "max_sequence_length": args.max_sequence_length,
    }

    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Metadata saved to %s.", metadata_path)

    # Summary printout
    print("\n--- Embedding Summary ---")
    print(f"  Training embeddings : {total_training}")
    print(f"  Validation embeddings: {val_count}")
    print(f"  Skipped             : {skipped}")
    print(f"  prompt_embeds shape : {prompt_embeds_shape}")
    print(f"  pooled_embeds shape : {pooled_embeds_shape}")
    print(f"  dtype               : {dtype_str}")
    print(f"  Disk usage          : {disk_mb:.2f} MB")
    print(f"  Saved to            : {embeddings_dir}")


if __name__ == "__main__":
    main()
