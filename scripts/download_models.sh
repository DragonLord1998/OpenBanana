#!/bin/bash
# scripts/download_models.sh
# Download all model weights required for OpenBanana training.
# Run AFTER setup/runpod_setup.sh and after accepting gated model licenses.
#
# Gated models -- accept license at these URLs before running:
#   FLUX.2-dev: https://huggingface.co/black-forest-labs/FLUX.2-dev
#
# Usage: bash scripts/download_models.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"

echo "======================================================"
echo "  OpenBanana Model Download"
echo "  Data directory: ${DATA_DIR}"
echo "======================================================"

# Verify huggingface-cli is available and user is logged in
if ! command -v huggingface-cli &>/dev/null; then
    echo "ERROR: huggingface-cli not found. Run setup/runpod_setup.sh first."
    exit 1
fi

if ! huggingface-cli whoami &>/dev/null; then
    echo "ERROR: Not logged in to HuggingFace. Run: huggingface-cli login"
    exit 1
fi

echo "  Logged in as: $(huggingface-cli whoami)"

# -------------------------------------------------------
# 1. Flux 2 Dev (gated -- user must accept license first)
# -------------------------------------------------------
echo ""
echo "[1/4] Downloading Flux 2 Dev..."
echo "  NOTE: This is a gated model. You must accept the license at:"
echo "        https://huggingface.co/black-forest-labs/FLUX.2-dev"
echo "  Size: ~33 GB. This will take a while."

FLUX2_DIR="${DATA_DIR}/flux2"
mkdir -p "${FLUX2_DIR}"

if [ -f "${FLUX2_DIR}/model_index.json" ]; then
    echo "  Flux 2 Dev already downloaded, skipping."
else
    huggingface-cli download \
        black-forest-labs/FLUX.2-dev \
        --local-dir "${FLUX2_DIR}"
    echo "  Flux 2 Dev downloaded to ${FLUX2_DIR}"
fi

# -------------------------------------------------------
# 2. HPS-v2.1 reward model weights
# -------------------------------------------------------
echo ""
echo "[2/4] Downloading HPS-v2.1 model weights..."

REWARD_DIR="${DATA_DIR}/reward_models"
mkdir -p "${REWARD_DIR}"

HPS_WEIGHTS="${REWARD_DIR}/HPS_v2.1_compressed.pt"

if [ -f "${HPS_WEIGHTS}" ]; then
    echo "  HPS_v2.1_compressed.pt already downloaded, skipping."
else
    huggingface-cli download \
        xswu/HPSv2 \
        HPS_v2.1_compressed.pt \
        --local-dir "${REWARD_DIR}"
    echo "  HPS-v2.1 weights downloaded to ${REWARD_DIR}"
fi

# -------------------------------------------------------
# 3. CLIP ViT-H-14 (required by HPS-v2.1)
# -------------------------------------------------------
echo ""
echo "[3/4] Downloading CLIP ViT-H-14..."

CLIP_WEIGHTS="${REWARD_DIR}/open_clip_pytorch_model.bin"

if [ -f "${CLIP_WEIGHTS}" ]; then
    echo "  open_clip_pytorch_model.bin already downloaded, skipping."
else
    huggingface-cli download \
        laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
        open_clip_pytorch_model.bin \
        --local-dir "${REWARD_DIR}"
    echo "  CLIP ViT-H-14 downloaded to ${REWARD_DIR}"
fi

# -------------------------------------------------------
# 4. Florence 2 Large (for image captioning of training data)
# -------------------------------------------------------
echo ""
echo "[4/4] Downloading Florence 2 Large..."
echo "  Size: ~1.5 GB."

FLORENCE_DIR="${DATA_DIR}/florence2"
mkdir -p "${FLORENCE_DIR}"

if [ -f "${FLORENCE_DIR}/config.json" ]; then
    echo "  Florence 2 Large already downloaded, skipping."
else
    huggingface-cli download \
        microsoft/Florence-2-large \
        --local-dir "${FLORENCE_DIR}"
    echo "  Florence 2 Large downloaded to ${FLORENCE_DIR}"
fi

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
echo ""
echo "======================================================"
echo "  Download complete. Directory summary:"
echo ""
du -sh "${DATA_DIR}"/flux2    2>/dev/null && true
du -sh "${DATA_DIR}"/florence2 2>/dev/null && true
du -sh "${DATA_DIR}"/reward_models 2>/dev/null && true
echo ""
echo "  Next step: bash scripts/train_openbanana.sh"
echo "======================================================"
