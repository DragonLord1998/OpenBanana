#!/bin/bash
# setup/runpod_setup.sh
# Master environment setup for OpenBanana on RunPod A100 80GB
# Usage: bash setup/runpod_setup.sh
# Idempotent: safe to re-run; already-completed steps are skipped.
# Uses system Python + pip (RunPod templates don't have conda)

set -euo pipefail

REPO_URL="https://github.com/DragonLord1998/OpenBanana.git"
SRPO_URL="https://github.com/Tencent-Hunyuan/SRPO.git"
WORK_DIR="${HOME}/workspace"

echo "======================================================"
echo "  OpenBanana RunPod Setup"
echo "  Target: A100 80GB"
echo "======================================================"

# -------------------------------------------------------
# 0. Verify we are on the expected hardware
# -------------------------------------------------------
echo ""
echo "[0/7] Verifying hardware..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" || true

# -------------------------------------------------------
# 1. Clone repositories
# -------------------------------------------------------
echo ""
echo "[1/7] Cloning repositories..."

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

if [ ! -d "OpenBanana/.git" ]; then
    echo "  Cloning OpenBanana..."
    git clone "${REPO_URL}" OpenBanana
else
    echo "  OpenBanana already cloned, pulling latest..."
    git -C OpenBanana pull
fi

if [ ! -d "OpenBanana/SRPO/.git" ]; then
    echo "  Cloning SRPO (READ-ONLY reference -- do not modify)..."
    git clone "${SRPO_URL}" OpenBanana/SRPO
    chmod -R a-w OpenBanana/SRPO || true
else
    echo "  SRPO already cloned, skipping."
fi

cd "${WORK_DIR}/OpenBanana"

# -------------------------------------------------------
# 2. Check system Python + PyTorch
# -------------------------------------------------------
echo ""
echo "[2/7] Checking system Python and PyTorch..."

PYTHON_BIN=$(which python3 || which python)
PIP_BIN=$(which pip3 || which pip)
echo "  Python: ${PYTHON_BIN} ($(${PYTHON_BIN} --version 2>&1))"
echo "  Pip: ${PIP_BIN}"

# Verify PyTorch is already available (RunPod templates ship with it)
if ${PYTHON_BIN} -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
    echo "  PyTorch already installed via RunPod template."
else
    echo "  WARNING: PyTorch not found. Installing..."
    ${PIP_BIN} install torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi

# -------------------------------------------------------
# 3. Flash Attention + Triton
# -------------------------------------------------------
echo ""
echo "[3/7] Installing flash-attn + triton..."

if ${PYTHON_BIN} -c "import flash_attn" 2>/dev/null; then
    echo "  flash-attn already installed, skipping."
else
    echo "  NOTE: flash-attn build can take 5-15 minutes..."
    ${PIP_BIN} install flash-attn triton
fi

# -------------------------------------------------------
# 4. diffusers ecosystem
# -------------------------------------------------------
echo ""
echo "[4/7] Installing diffusers ecosystem (from source for Flux 2 support)..."

if ${PYTHON_BIN} -c "from diffusers import FluxPipeline" 2>/dev/null; then
    echo "  diffusers (with FluxPipeline) already installed."
else
    ${PIP_BIN} install git+https://github.com/huggingface/diffusers.git
fi

${PIP_BIN} install --quiet \
    "peft>=0.14.0" \
    "bitsandbytes>=0.44.0" \
    "transformers>=4.45.0" \
    "accelerate>=0.34.0"

# -------------------------------------------------------
# 5. Monitoring + utilities
# -------------------------------------------------------
echo ""
echo "[5/7] Installing monitoring and utility packages..."

${PIP_BIN} install --quiet \
    tensorboard \
    datasets \
    Pillow \
    safetensors \
    open_clip_torch \
    tqdm

# -------------------------------------------------------
# 6. HPS-v2.1
# -------------------------------------------------------
echo ""
echo "[6/7] Installing HPS-v2.1..."

HPSV2_DIR="${WORK_DIR}/HPSv2"

if [ ! -d "${HPSV2_DIR}/.git" ]; then
    echo "  Cloning HPSv2..."
    git clone https://github.com/tgxs002/HPSv2.git "${HPSV2_DIR}"
else
    echo "  HPSv2 already cloned, skipping."
fi

if ${PYTHON_BIN} -c "import hpsv2" 2>/dev/null; then
    echo "  hpsv2 already installed, skipping."
else
    ${PIP_BIN} install -e "${HPSV2_DIR}"
fi

# -------------------------------------------------------
# 7. Verify all imports
# -------------------------------------------------------
echo ""
echo "[7/7] Verifying all imports..."

${PYTHON_BIN} -c "
import torch
import diffusers
import peft
import bitsandbytes
import tensorboard
from diffusers import FluxPipeline
print('All imports OK')
print(f'  torch:         {torch.__version__}')
print(f'  diffusers:     {diffusers.__version__}')
print(f'  peft:          {peft.__version__}')
print(f'  bitsandbytes:  {bitsandbytes.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU:            {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"

# -------------------------------------------------------
# HuggingFace login
# -------------------------------------------------------
echo ""
echo "======================================================"
echo "  HuggingFace Login"
echo "  You need an HF account with access to:"
echo "    - black-forest-labs/FLUX.2-dev  (gated, accept license first)"
echo "======================================================"
huggingface-cli login

# -------------------------------------------------------
# Done
# -------------------------------------------------------
echo ""
echo "======================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    cd ${WORK_DIR}/OpenBanana"
echo "    bash scripts/download_models.sh"
echo "    python scripts/baseline_characterization.py"
echo "    python scripts/model_introspection.py"
echo "    python scripts/caption_dataset.py"
echo "    python scripts/preprocess_embeddings.py"
echo "    python scripts/preprocess_latents.py"
echo "    bash scripts/train_openbanana.sh"
echo "======================================================"
