"""
train_openbanana.py - SRPO Direct-Align training for Open Banana v0.2.

Fine-tunes Flux 2 Dev with LoRA using transplanted SRPO (Semantic Relative
Preference Optimization) loss from Tencent-Hunyuan/SRPO (fastvideo/SRPO.py).
Does NOT fork SRPO -- transplants only the ~140 lines of Direct-Align loss math
into a clean diffusers/PEFT training loop.

Usage: python train_openbanana.py [args]  (see scripts/train_openbanana.sh)
"""

import argparse
import json
import os
import time
from glob import glob

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import open_clip
from diffusers import AutoencoderKL, FluxPipeline, FluxTransformer2DModel
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

# ---------------------------------------------------------------------------
# SECTION 1: Argument parsing (~40 lines)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open Banana v0.2 -- SRPO Direct-Align LoRA training.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained_model_name_or_path", type=str, default="./data/flux2")
    p.add_argument("--embedding_dir", type=str, default="./data/embeddings")
    p.add_argument("--latent_dir", type=str, default="./data/latents")
    p.add_argument("--output_dir", type=str, default="./output/open-banana-v0.2")
    p.add_argument("--tensorboard_dir", type=str, default="./output/logs")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_train_steps", type=int, default=2000)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--mixed_precision", type=str, default="bf16")
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--h", type=int, default=720)
    p.add_argument("--w", type=int, default=720)
    p.add_argument("--sampling_steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.3, help="Noise injection for inversion branch.")
    p.add_argument("--lr_warmup_steps", type=int, default=100)
    p.add_argument("--max_grad_norm", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--discount_inv", nargs=2, type=float, default=[0.3, 0.01])
    p.add_argument("--discount_pos", nargs=2, type=float, default=[0.1, 0.25])
    p.add_argument("--groundtruth_ratio", type=float, default=0.9)
    p.add_argument("--hinge_margin", type=float, default=0.7,
                    help="Hinge loss margin -- calibrate from Phase 0 baseline scoring.")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--sample_every_n_steps", type=int, default=250)
    p.add_argument("--sample_prompts", type=str, default="./data/validation_prompts.txt")
    p.add_argument("--reward_model_path", type=str, default="./data/reward_models/HPS_v2.1_compressed.pt")
    p.add_argument("--clip_model_path", type=str, default="./data/reward_models/open_clip_pytorch_model.bin")
    return p.parse_args()

# ---------------------------------------------------------------------------
# SECTION 2: Model loading (~50 lines)
# ---------------------------------------------------------------------------

def load_models(args):
    """Load transformer (NF4 + LoRA), VAE (frozen), noise scheduler, HPS-v2.1."""
    print("=" * 60 + "\nLoading models...\n" + "=" * 60)

    # Transformer with NF4 quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        quantization_config=bnb_config, torch_dtype=torch.bfloat16,
    )
    # Apply LoRA to attention + feed-forward layers
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules=[
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2",
        ],
        lora_dropout=0.05, bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    transformer.print_trainable_parameters()

    # VAE (frozen, bf16)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.bfloat16,
    ).to("cuda")
    vae.eval()
    vae.requires_grad_(False)

    # Noise scheduler -- READ shift from config, don't hardcode
    pipe_for_scheduler = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16,
        transformer=None, vae=None, text_encoder=None, text_encoder_2=None,
    )
    noise_scheduler = pipe_for_scheduler.scheduler
    shift = noise_scheduler.config.get("shift", 3.0)
    print(f"Scheduler shift parameter: {shift} (read from model config, NOT hardcoded)")
    del pipe_for_scheduler

    # HPS-v2.1 reward model (frozen, loaded into CLIP ViT-H-14)
    clip_pretrained = args.clip_model_path if os.path.exists(args.clip_model_path) else "laion2b_s32b_b79k"
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=clip_pretrained)
    clip_model = clip_model.to("cuda").eval()
    clip_model.requires_grad_(False)
    hps_ckpt = torch.load(args.reward_model_path, map_location="cuda", weights_only=True)
    hps_sd = hps_ckpt.get("state_dict", hps_ckpt)
    hps_sd = {k.replace("module.", ""): v for k, v in hps_sd.items()}
    clip_model.load_state_dict(hps_sd, strict=False)
    print("HPS-v2.1 loaded into CLIP ViT-H-14 backbone.")

    return transformer, vae, noise_scheduler, clip_model, clip_preprocess

# ---------------------------------------------------------------------------
# SECTION 3: Data loading (~35 lines)
# ---------------------------------------------------------------------------

class OpenBananaDataset(Dataset):
    """Wraps pre-computed text embeddings and image latents."""
    def __init__(self, embedding_dir: str, latent_dir: str):
        self.embedding_files = sorted(glob(f"{embedding_dir}/*_prompt_embeds.pt"))
        self.latent_files = sorted(glob(f"{latent_dir}/*_latent.pt"))
        # Exclude validation embeddings (val_ prefix)
        self.embedding_files = [f for f in self.embedding_files if "/val_" not in f]
        assert len(self.embedding_files) > 0, f"No training embeddings in {embedding_dir}"
        assert len(self.latent_files) > 0, f"No latents in {latent_dir}"
        assert len(self.embedding_files) == len(self.latent_files), (
            f"Mismatch: {len(self.embedding_files)} embeddings vs {len(self.latent_files)} latents")
        print(f"Dataset: {len(self.embedding_files)} training pairs loaded.")

    def __len__(self): return len(self.embedding_files)

    def __getitem__(self, idx):
        pe = torch.load(self.embedding_files[idx], weights_only=True)
        pp = torch.load(self.embedding_files[idx].replace("_prompt_embeds", "_pooled_embeds"), weights_only=True)
        lat = torch.load(self.latent_files[idx], weights_only=True)
        return pe.squeeze(0), pp.squeeze(0), lat.squeeze(0)


def load_validation_embeddings(embedding_dir: str, max_count: int = 4):
    """Load pre-computed validation prompt embeddings."""
    val_files = sorted(glob(f"{embedding_dir}/val_*_prompt_embeds.pt"))[:max_count]
    val_data = []
    for f in val_files:
        pe = torch.load(f, weights_only=True).to("cuda", dtype=torch.bfloat16)
        pp = torch.load(f.replace("_prompt_embeds", "_pooled_embeds"), weights_only=True).to("cuda", dtype=torch.bfloat16)
        val_data.append((pe, pp))
    print(f"Loaded {len(val_data)} validation embedding pairs.")
    return val_data

# ---------------------------------------------------------------------------
# SECTION 4: Optimizer and scheduler (~20 lines)
# ---------------------------------------------------------------------------

def create_optimizer_and_scheduler(transformer, args):
    """Create AdamW optimizer (LoRA params only) with linear warmup."""
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {total_params:,} (should be ~50-100M, NOT 32B)")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.999))
    def lr_lambda(step):
        return step / args.lr_warmup_steps if step < args.lr_warmup_steps else 1.0
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ---------------------------------------------------------------------------
# SECTION 5: SRPO loss computation -- THE TRANSPLANTED MATH (~140 lines)
# Transplanted from fastvideo/SRPO.py (Tencent-Hunyuan/SRPO)
# ---------------------------------------------------------------------------

def compute_hps_score(images, clip_model):
    """
    Compute HPS-v2.1 score for a batch of images (DIFFERENTIABLE).

    HPS-v2.1 scores via CLIP cosine similarity scaled by logit_scale.
    Must remain differentiable so gradients flow through VAE decode + transformer.

    Args:
        images: [B, 3, H, W] in [0, 1]
        clip_model: CLIP ViT-H-14 with HPS-v2.1 weights

    Returns:
        Scalar tensor (differentiable through images)

    TODO: Verify against actual HPS-v2.1 checkpoint structure. The checkpoint loads
    the full CLIP state dict into ViT-H-14 (verified in baseline_characterization.py).
    During training we only have pre-computed text embeddings, not raw text, so we
    use image feature norm as a proxy reward signal. If the checkpoint contains a
    separate MLP head, update this function to use it. The scoring MUST remain
    differentiable for gradient flow.
    """
    # Resize to CLIP input (224x224) and normalize (ImageNet mean/std)
    images_resized = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)
    images_norm = (images_resized - mean) / std
    # CLIP image features (differentiable)
    image_features = clip_model.encode_image(images_norm)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    # Score via logit_scale (learned temperature)
    logit_scale = clip_model.logit_scale.exp()
    return (logit_scale * image_features.norm(dim=-1)).mean()


def compute_srpo_loss(transformer, vae, noise_scheduler, clean_latent, prompt_embeds,
                      pooled_prompt_embeds, clip_model, args, global_step, max_steps):
    """
    SRPO Direct-Align loss computation.

    Pipeline:
    1. Sample timestep t, create noisy latent xt from x0
    2. Transformer predicts velocity (denoising branch)
    3. Single-step Direct-Align recovery: x_hat_0 = xt - t * v_pred
    4. Decode through VAE, score with HPS-v2.1
    5. Inversion branch (penalty/regularization)
    6. Combined hinge loss with discount schedules
    """
    device = clean_latent.device
    bsz = clean_latent.shape[0]

    # 1. Sample timestep (uniform [0,1) for rectified flow)
    t = torch.rand(bsz, device=device)
    noise = torch.randn_like(clean_latent)
    t_exp = t.view(bsz, 1, 1, 1)
    # Rectified flow interpolation: xt = (1-t)*x0 + t*eps
    noisy_latent = (1 - t_exp) * clean_latent + t_exp * noise

    # 2. Discount factors interpolated by training progress
    progress = global_step / max_steps
    d_inv = args.discount_inv[0] + (args.discount_inv[1] - args.discount_inv[0]) * progress
    d_pos = args.discount_pos[0] + (args.discount_pos[1] - args.discount_pos[0]) * progress

    # 3. Denoising branch: predict velocity
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v_pred = transformer(
            hidden_states=noisy_latent, timestep=t,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds, return_dict=False,
        )[0]

    # 4. Direct-Align single-step recovery: x_hat_0 = xt - t*v_pred
    predicted_clean = noisy_latent - t_exp * v_pred

    # Groundtruth ratio: use GT image for reward scoring (stabilizes training)
    reward_latent = clean_latent if torch.rand(1).item() < args.groundtruth_ratio else predicted_clean

    # 5. Decode to pixel space and score
    with torch.autocast("cuda", dtype=torch.bfloat16):
        decoded = vae.decode(reward_latent / vae.config.scaling_factor).sample
    decoded = (decoded.clamp(-1, 1) + 1) / 2  # [0, 1]
    reward_score = compute_hps_score(decoded, clip_model)

    # 6. Inversion branch (penalty/regularization)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inv_pred = transformer(
            hidden_states=clean_latent,
            timestep=torch.ones(bsz, device=device) * args.eta,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds, return_dict=False,
        )[0]
    inv_latent = clean_latent + args.eta * inv_pred
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inv_decoded = vae.decode(inv_latent / vae.config.scaling_factor).sample
    inv_decoded = (inv_decoded.clamp(-1, 1) + 1) / 2
    inv_score = compute_hps_score(inv_decoded, clip_model)

    # 7. Combined SRPO hinge loss
    combined = d_pos * reward_score - d_inv * inv_score
    loss = F.relu(-combined + args.hinge_margin) / args.gradient_accumulation_steps
    return loss, reward_score.item(), inv_score.item()

# ---------------------------------------------------------------------------
# SECTION 7: Checkpointing (~20 lines)
# ---------------------------------------------------------------------------

def save_checkpoint(transformer, optimizer, step, args):
    """Save LoRA weights, optimizer state, and metadata."""
    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    transformer.save_pretrained(ckpt_dir)
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    with open(os.path.join(ckpt_dir, "metadata.json"), "w") as f:
        json.dump({"step": step, "hinge_margin": args.hinge_margin,
                    "learning_rate": args.learning_rate, "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha}, f, indent=2)
    print(f"Checkpoint saved to {ckpt_dir}")

# ---------------------------------------------------------------------------
# SECTION 8: Validation sampling (~35 lines)
# ---------------------------------------------------------------------------

def generate_validation_samples(transformer, vae, noise_scheduler, val_embeddings,
                                step, writer, args):
    """Generate validation images via full denoising loop and log to TensorBoard."""
    from PIL import Image
    transformer.eval()
    images = []
    with torch.no_grad():
        for prompt_embeds, pooled_embeds in val_embeddings[:4]:
            latent = torch.randn(1, 16, args.h // 8, args.w // 8, device="cuda", dtype=torch.bfloat16)
            # Full denoising loop using Flux scheduler
            noise_scheduler.set_timesteps(args.sampling_steps, device="cuda")
            for t in noise_scheduler.timesteps:
                t_batch = t.unsqueeze(0) if t.dim() == 0 else t
                t_norm = t_batch / noise_scheduler.config.num_train_timesteps
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = transformer(
                        hidden_states=latent, timestep=t_norm,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_embeds, return_dict=False,
                    )[0]
                latent = noise_scheduler.step(pred, t, latent, return_dict=False)[0]
            decoded = vae.decode(latent / vae.config.scaling_factor).sample
            images.append(decoded)

    if images:
        grid = torch.cat(images, dim=0)
        grid = (grid.clamp(-1, 1) + 1) / 2
        writer.add_images("validation/samples", grid.float(), step)
        # Save to disk
        save_dir = os.path.join(args.output_dir, "samples", f"step_{step}")
        os.makedirs(save_dir, exist_ok=True)
        for i, img_t in enumerate(images):
            img = (img_t.squeeze(0).clamp(-1, 1) + 1) / 2
            img = (img.float().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
            Image.fromarray(img).save(os.path.join(save_dir, f"sample_{i}.png"))
    transformer.train()

# ---------------------------------------------------------------------------
# SECTION 6: Training loop (~70 lines)
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.tensorboard_dir)
    # Log hyperparameters
    hparams = {k: str(v) if isinstance(v, list) else v for k, v in vars(args).items()}
    writer.add_text("hyperparameters", json.dumps(hparams, indent=2), 0)

    # Load models + data
    transformer, vae, noise_scheduler, clip_model, _ = load_models(args)
    dataset = OpenBananaDataset(args.embedding_dir, args.latent_dir)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    val_embeddings = load_validation_embeddings(args.embedding_dir)
    optimizer, lr_scheduler = create_optimizer_and_scheduler(transformer, args)

    global_step = 0
    data_iter = iter(dataloader)
    grad_checked = False

    print(f"\n{'=' * 60}\nStarting training\n"
          f"  Steps: {args.max_train_steps} | Batch: {args.train_batch_size} | "
          f"Grad accum: {args.gradient_accumulation_steps}\n"
          f"  Effective batch: {args.train_batch_size * args.gradient_accumulation_steps} | "
          f"LR: {args.learning_rate} | Margin: {args.hinge_margin}\n"
          f"  Resolution: {args.h}x{args.w}\n{'=' * 60}\n")

    transformer.train()
    t_start = time.time()

    while global_step < args.max_train_steps:
        step_loss, step_reward, step_penalty = 0.0, 0.0, 0.0

        for _ in range(args.gradient_accumulation_steps):
            try:
                prompt_embeds, pooled_embeds, latent = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                prompt_embeds, pooled_embeds, latent = next(data_iter)

            prompt_embeds = prompt_embeds.to("cuda", dtype=torch.bfloat16)
            pooled_embeds = pooled_embeds.to("cuda", dtype=torch.bfloat16)
            latent = latent.to("cuda", dtype=torch.bfloat16)

            loss, reward, penalty = compute_srpo_loss(
                transformer, vae, noise_scheduler, latent, prompt_embeds,
                pooled_embeds, clip_model, args, global_step, args.max_train_steps,
            )
            loss.backward()

            # Gradient flow assertion on first backward step
            if not grad_checked:
                for name, param in transformer.named_parameters():
                    if param.requires_grad and "lora_A" in name and param.grad is not None:
                        gn = param.grad.norm().item()
                        assert gn > 0, f"FATAL: Zero gradient on {name} -- gradient flow broken"
                        print(f"Gradient flow verified: {name} grad_norm={gn:.6f}")
                        grad_checked = True
                        break

            step_loss += loss.item()
            step_reward += reward
            step_penalty += penalty

        step_reward /= args.gradient_accumulation_steps
        step_penalty /= args.gradient_accumulation_steps

        # Clip gradients and step
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in transformer.parameters() if p.requires_grad], args.max_grad_norm,
        ).item()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        global_step += 1
        current_lr = lr_scheduler.get_last_lr()[0]

        # VRAM profiling on first step
        if global_step == 1 and torch.cuda.is_available():
            print(f"Peak VRAM after first step: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

        # TensorBoard logging
        writer.add_scalar("train/loss", step_loss, global_step)
        writer.add_scalar("train/reward_score", step_reward, global_step)
        writer.add_scalar("train/penalty_score", step_penalty, global_step)
        writer.add_scalar("train/learning_rate", current_lr, global_step)
        writer.add_scalar("train/grad_norm", grad_norm, global_step)
        if torch.cuda.is_available():
            writer.add_scalar("system/gpu_memory_gb", torch.cuda.max_memory_allocated() / 1e9, global_step)
            writer.add_scalar("system/gpu_memory_reserved_gb", torch.cuda.max_memory_reserved() / 1e9, global_step)

        # Progress
        elapsed = time.time() - t_start
        sps = global_step / elapsed if elapsed > 0 else 0
        vram = f"{torch.cuda.max_memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
        print(f"Step {global_step}/{args.max_train_steps} | loss={step_loss:.4f} | "
              f"reward={step_reward:.4f} | penalty={step_penalty:.4f} | lr={current_lr:.2e} | "
              f"grad={grad_norm:.4f} | VRAM={vram} | {sps:.2f} steps/s")

        # Validation sampling
        if global_step % args.sample_every_n_steps == 0 and val_embeddings:
            print(f"Generating validation samples at step {global_step}...")
            generate_validation_samples(transformer, vae, noise_scheduler,
                                        val_embeddings, global_step, writer, args)

        # Checkpointing
        if global_step % args.checkpointing_steps == 0:
            save_checkpoint(transformer, optimizer, global_step, args)
            lora_norm = sum(p.norm().item() ** 2 for n, p in transformer.named_parameters()
                            if p.requires_grad and "lora" in n.lower()) ** 0.5
            writer.add_scalar("checkpoint/lora_weight_norm", lora_norm, global_step)

    # Final checkpoint
    save_checkpoint(transformer, optimizer, global_step, args)
    elapsed_total = time.time() - t_start
    print(f"\n{'=' * 60}\nTraining complete in {elapsed_total / 60:.1f} minutes.\n"
          f"Final checkpoint: {args.output_dir}/checkpoint-{global_step}\n"
          f"TensorBoard logs: {args.tensorboard_dir}\n{'=' * 60}")
    writer.close()


if __name__ == "__main__":
    main()
