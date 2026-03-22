#!/bin/bash
# scripts/train_openbanana.sh
# Launch OpenBanana training with SRPO Direct-Align loss on a single A100 80GB.
#
# Pre-conditions:
#   1. setup/runpod_setup.sh has been run
#   2. scripts/download_models.sh has been run
#   3. Embeddings and latents have been pre-computed into ./data/embeddings/ and ./data/latents/
#   4. Phase 0.3 baseline HPS-v2.1 calibration has been performed.
#      UPDATE --hinge_margin below if calibration produced a different value than 0.7.
#
# Usage: bash scripts/train_openbanana.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# -------------------------------------------------------
# TensorBoard
# -------------------------------------------------------
echo "Starting TensorBoard on port 6006..."
tensorboard \
    --logdir "${REPO_ROOT}/output/logs" \
    --port 6006 \
    --bind_all \
    &
TENSORBOARD_PID=$!
echo "TensorBoard started (PID: ${TENSORBOARD_PID})"
echo "  Access at: http://localhost:6006  (or RunPod's exposed port 6006)"

# Kill TensorBoard on exit (clean or error)
cleanup() {
    echo ""
    echo "Stopping TensorBoard (PID: ${TENSORBOARD_PID})..."
    kill "${TENSORBOARD_PID}" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# -------------------------------------------------------
# Training
# -------------------------------------------------------
echo ""
echo "Launching training..."
echo "  Resolution:    720x720 (default -- safe for A100 80GB)"
echo "  LoRA rank:     32"
echo "  Steps:         2000"
echo "  Batch size:    1 (effective 4 via gradient accumulation)"
echo "  LR:            5e-5"
echo "  TensorBoard:   ./output/logs"
echo ""

# NOTE: --shift is NOT set here. The shift parameter is read at runtime from
# FluxPipeline.scheduler.config. Do NOT hardcode --shift 3 (that is Flux 1 Dev).
# Run scripts/model_introspection.py (Phase 0.4) to verify the correct value first.
#
# NOTE: --hinge_margin 0.7 is the SRPO paper default. If Phase 0.3 baseline
# calibration shows mean HPS-v2.1 > 0.7, update this value to mean + 0.1.
# See RALPLAN-DR: "RISK FLAG -- Step 0.3" for details.

python "${REPO_ROOT}/train_openbanana.py" \
    --seed 42 \
    --pretrained_model_name_or_path "${REPO_ROOT}/data/flux2" \
    --embedding_dir "${REPO_ROOT}/data/embeddings" \
    --latent_dir "${REPO_ROOT}/data/latents" \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_train_steps 2000 \
    --learning_rate 5e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 500 \
    --allow_tf32 \
    --output_dir "${REPO_ROOT}/output/open-banana-v0.2" \
    --h 720 \
    --w 720 \
    --sampling_steps 50 \
    --eta 0.3 \
    --lr_warmup_steps 100 \
    --max_grad_norm 0.1 \
    --weight_decay 0.0001 \
    --discount_inv 0.3 0.01 \
    --discount_pos 0.1 0.25 \
    --groundtruth_ratio 0.9 \
    --hinge_margin 0.5 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --tensorboard_dir "${REPO_ROOT}/output/logs" \
    --sample_every_n_steps 250 \
    --sample_prompts "${REPO_ROOT}/data/validation_prompts.txt"

echo ""
echo "======================================================"
echo "  Training complete."
echo "  Checkpoints: ${REPO_ROOT}/output/open-banana-v0.2/"
echo "  TensorBoard logs: ${REPO_ROOT}/output/logs/"
echo "======================================================"

# -------------------------------------------------------
# OPTIONAL: 1024x1024 upgrade
# -------------------------------------------------------
# To train at 1024x1024 (risky -- estimated 65-81 GB VRAM, may OOM):
#
#   1. Confirm VRAM headroom from a 720x720 smoke test:
#        python train_openbanana.py ... --max_train_steps 10 --h 720 --w 720
#      Record peak VRAM with torch.cuda.max_memory_allocated().
#
#   2. If headroom >= 15 GB, replace --h 720 --w 720 with --h 1024 --w 1024
#      and rerun. Monitor the first 10 steps closely for OOM.
#
#   3. If OOM: reduce --gradient_accumulation_steps from 4 to 2, or enable
#      enable_model_cpu_offload() for the reward model loading.
#
# Do NOT use 1024x1024 as the default. The VRAM math in the plan shows it
# may leave 0-15 GB headroom, which is insufficient for stability.
# -------------------------------------------------------
