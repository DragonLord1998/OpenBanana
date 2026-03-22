"""
preprocess_latents.py - Pre-compute image latents via VAE encoding.

Encodes all 720x720 training images through Flux 2's VAE encoder to produce
latent tensors. This allows the VAE encoder to be unloaded during training,
saving VRAM. The VAE decoder is still needed during training for the SRPO
reward path.
"""

import argparse
import json
import logging
import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKLFlux2
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

EXPECTED_LATENT_CHANNELS = 16
EXPECTED_SPATIAL = 90  # 720 / 8
VERIFICATION_COUNT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute image latents via Flux 2 VAE encoding.")
    parser.add_argument("--model-path", type=str, default="black-forest-labs/FLUX.1-dev",
                        help="Path or HuggingFace ID for Flux 2 (VAE is loaded from subfolder).")
    parser.add_argument("--images-dir", type=str, default="./data/openbanana/images",
                        help="Directory containing 720x720 training images.")
    parser.add_argument("--latents-dir", type=str, default="./data/latents",
                        help="Directory to save latent .pt files.")
    parser.add_argument("--metadata-path", type=str, default="./data/latents/metadata.json",
                        help="Path to save latent metadata JSON.")
    parser.add_argument("--verification-dir", type=str, default="./data/latents/verification",
                        help="Directory to save round-trip decoded verification images.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for VAE encoding (cuda or cpu).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for verification sample selection.")
    return parser.parse_args()


def load_vae(model_path: str, device: str) -> AutoencoderKLFlux2:
    """Load Flux 2 VAE standalone in bfloat16."""
    logger.info("Loading VAE from %s (subfolder=vae) ...", model_path)
    vae = AutoencoderKLFlux2.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to(device)
    vae.eval()
    logger.info("VAE loaded on %s (dtype=bfloat16).", device)
    return vae


def image_to_tensor(image_path: Path, device: str) -> torch.Tensor:
    """Load a PNG image and convert to a [-1, 1] normalized bfloat16 tensor of shape [1, 3, H, W]."""
    img = Image.open(image_path).convert("RGB")
    # Convert to float32 tensor [3, H, W] in [0, 1]
    tensor = TF.to_tensor(img)  # [3, H, W], float32, [0, 1]
    # Normalize to [-1, 1]
    tensor = tensor * 2.0 - 1.0
    # Add batch dimension and cast to bfloat16
    tensor = tensor.unsqueeze(0).to(dtype=torch.bfloat16, device=device)
    return tensor


def encode_image(vae: AutoencoderKLFlux2, image_tensor: torch.Tensor) -> torch.Tensor:
    """Encode an image tensor to latent space, returning a [1, 16, H/8, W/8] bfloat16 tensor."""
    with torch.no_grad():
        posterior = vae.encode(image_tensor)
        # DiagonalGaussianDistribution -- sample the latent
        latent = posterior.latent_dist.sample()
        # Apply VAE scaling factor
        latent = latent * vae.config.scaling_factor
    return latent.to(dtype=torch.bfloat16)


def decode_latent(vae: AutoencoderKLFlux2, latent: torch.Tensor) -> Image.Image:
    """Decode a latent tensor back to a PIL image (for verification)."""
    with torch.no_grad():
        # Reverse the scaling factor
        unscaled = latent.to(dtype=torch.bfloat16) / vae.config.scaling_factor
        decoded = vae.decode(unscaled).sample  # [1, 3, H, W]
    # Convert from [-1, 1] float to [0, 255] uint8
    decoded = decoded.float().cpu().squeeze(0)  # [3, H, W]
    decoded = (decoded.clamp(-1.0, 1.0) + 1.0) / 2.0  # [0, 1]
    decoded = (decoded * 255).byte()  # [0, 255]
    # [3, H, W] -> [H, W, 3]
    decoded_np = decoded.permute(1, 2, 0).numpy()
    return Image.fromarray(decoded_np, mode="RGB")


def disk_usage_mb(directory: Path) -> float:
    """Return total size of .pt files in directory in megabytes."""
    total = sum(f.stat().st_size for f in directory.glob("*.pt") if f.is_file())
    return total / (1024 ** 2)


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir)
    latents_dir = Path(args.latents_dir)
    metadata_path = Path(args.metadata_path)
    verification_dir = Path(args.verification_dir)

    latents_dir.mkdir(parents=True, exist_ok=True)
    verification_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        import warnings
        warnings.warn("CUDA not available, falling back to CPU.", stacklevel=2)

    vae = load_vae(args.model_path, device)

    # Collect image files sorted by stem
    image_files = sorted(images_dir.glob("*.png"), key=lambda p: p.stem)
    if not image_files:
        raise FileNotFoundError(f"No PNG images found in {images_dir}")
    logger.info("Found %d images in %s.", len(image_files), images_dir)

    metadata: dict = {"latents": {}, "summary": {}}
    latent_shape: list | None = None
    dtype_str: str | None = None
    skipped = 0
    saved_indices: list[int] = []

    for i, image_path in enumerate(tqdm(image_files, desc="Encoding images to latents")):
        idx = image_path.stem  # e.g. "000"
        latent_path = latents_dir / f"{idx}_latent.pt"

        try:
            image_tensor = image_to_tensor(image_path, device)
            latent = encode_image(vae, image_tensor)

            # Verify expected shape
            expected = (1, EXPECTED_LATENT_CHANNELS, EXPECTED_SPATIAL, EXPECTED_SPATIAL)
            if latent.shape != torch.Size(expected):
                logger.warning(
                    "Unexpected latent shape for %s: %s (expected %s)",
                    idx, tuple(latent.shape), expected,
                )

            torch.save(latent.cpu(), latent_path)

            if latent_shape is None:
                latent_shape = list(latent.shape)
                dtype_str = str(latent.dtype)

            metadata["latents"][idx] = {
                "image_path": str(image_path),
                "latent_path": str(latent_path),
                "shape": list(latent.shape),
                "dtype": str(latent.dtype),
            }
            saved_indices.append(i)

        except Exception as exc:
            logger.warning("Skipping image %s due to error: %s", idx, exc)
            skipped += 1
            continue

    # Round-trip verification on 3 random latents
    random.seed(args.seed)
    if len(saved_indices) >= VERIFICATION_COUNT:
        verify_indices = random.sample(saved_indices, VERIFICATION_COUNT)
    else:
        verify_indices = saved_indices

    logger.info("Running round-trip verification on %d latents ...", len(verify_indices))
    for vi in verify_indices:
        image_path = image_files[vi]
        idx = image_path.stem
        latent_path = latents_dir / f"{idx}_latent.pt"
        try:
            latent = torch.load(latent_path, map_location=device, weights_only=True).to(device)
            decoded_img = decode_latent(vae, latent)
            out_path = verification_dir / f"{idx}_roundtrip.png"
            decoded_img.save(out_path, format="PNG")
            logger.info("Verification image saved: %s", out_path)
        except Exception as exc:
            logger.warning("Round-trip verification failed for %s: %s", idx, exc)

    # Build summary
    total_saved = len(metadata["latents"])
    disk_mb = disk_usage_mb(latents_dir)
    metadata["summary"] = {
        "total_latents": total_saved,
        "skipped": skipped,
        "latent_shape": latent_shape,
        "dtype": dtype_str,
        "source_resolution": [720, 720],
        "vae_downsample_factor": 8,
        "disk_usage_mb": round(disk_mb, 2),
        "verification_indices": [image_files[vi].stem for vi in verify_indices],
    }

    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Metadata saved to %s.", metadata_path)

    # Summary printout
    print("\n--- Latent Summary ---")
    print(f"  Total latents saved  : {total_saved}")
    print(f"  Skipped              : {skipped}")
    print(f"  Latent shape         : {latent_shape}")
    print(f"  dtype                : {dtype_str}")
    print(f"  Source resolution    : 720x720")
    print(f"  Disk usage           : {disk_mb:.2f} MB")
    print(f"  Verification images  : {verification_dir}")
    print(f"  Metadata             : {metadata_path}")


if __name__ == "__main__":
    main()
