#!/bin/bash
# setup/runpod_setup.sh
# Master environment setup for OpenBanana on RunPod A100 80GB (CUDA 12.4, PyTorch 2.6 template)
# Usage: bash setup/runpod_setup.sh
# Idempotent: safe to re-run; already-completed steps are skipped.

set -euo pipefail

REPO_URL="https://github.com/YOUR_USERNAME/OpenBanana.git"  # TODO: replace with real URL
SRPO_URL="https://github.com/Tencent-Hunyuan/SRPO.git"
CONDA_ENV="openbanana"
PYTHON_VERSION="3.10.16"
WORK_DIR="${HOME}/workspace"

echo "======================================================"
echo "  OpenBanana RunPod Setup"
echo "  Target: A100 80GB, CUDA 12.4, PyTorch 2.6 template"
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
    # Make the clone read-only as a guard against accidental modification
    chmod -R a-w OpenBanana/SRPO || true
else
    echo "  SRPO already cloned, skipping."
fi

cd "${WORK_DIR}/OpenBanana"

# -------------------------------------------------------
# 2. Create conda environment
# -------------------------------------------------------
echo ""
echo "[2/7] Creating conda environment: ${CONDA_ENV} (Python ${PYTHON_VERSION})..."

if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "  Conda env '${CONDA_ENV}' already exists, skipping creation."
else
    conda create -y -n "${CONDA_ENV}" python="${PYTHON_VERSION}"
    echo "  Conda env '${CONDA_ENV}' created."
fi

# All subsequent pip installs run inside the conda env
CONDA_PYTHON="$(conda run -n ${CONDA_ENV} which python)"
CONDA_PIP="$(conda run -n ${CONDA_ENV} which pip)"
echo "  Using Python: ${CONDA_PYTHON}"

# -------------------------------------------------------
# 3. Step 1 -- Core PyTorch (cu124)
# -------------------------------------------------------
echo ""
echo "[3/7] Step 1: Installing core PyTorch 2.6.0 (cu124)..."

if conda run -n "${CONDA_ENV}" python -c "import torch; assert torch.__version__.startswith('2.6')" 2>/dev/null; then
    echo "  torch==2.6.0 already installed, skipping."
else
    conda run -n "${CONDA_ENV}" pip install \
        torch==2.6.0 torchvision \
        --index-url https://download.pytorch.org/whl/cu124
fi

# -------------------------------------------------------
# 4. Step 2 -- Flash Attention + Triton
# -------------------------------------------------------
echo ""
echo "[4/7] Step 2: Installing flash-attn + triton (SRPO reference versions)..."
echo "  NOTE: flash-attn build can take 5-15 minutes..."

if conda run -n "${CONDA_ENV}" python -c "import flash_attn; assert flash_attn.__version__ == '2.7.0.post2'" 2>/dev/null; then
    echo "  flash-attn==2.7.0.post2 already installed, skipping."
else
    conda run -n "${CONDA_ENV}" pip install \
        flash-attn==2.7.0.post2 \
        triton==2.3.0
fi

# -------------------------------------------------------
# 5. Step 3 -- diffusers ecosystem
# -------------------------------------------------------
echo ""
echo "Step 3: Installing diffusers ecosystem (from source for latest Flux 2 support)..."

if conda run -n "${CONDA_ENV}" python -c "import diffusers; from diffusers import FluxPipeline" 2>/dev/null; then
    echo "  diffusers (with FluxPipeline) already installed, skipping source install."
else
    conda run -n "${CONDA_ENV}" pip install \
        git+https://github.com/huggingface/diffusers.git
fi

conda run -n "${CONDA_ENV}" pip install \
    "peft>=0.14.0" \
    "bitsandbytes>=0.44.0" \
    "transformers>=4.45.0" \
    "accelerate>=0.34.0"

# -------------------------------------------------------
# 6. Step 4 -- Monitoring + utilities
# -------------------------------------------------------
echo ""
echo "Step 4: Installing monitoring and utility packages..."

conda run -n "${CONDA_ENV}" pip install \
    tensorboard \
    datasets \
    Pillow \
    safetensors \
    open_clip_torch

# -------------------------------------------------------
# 7. Step 5 -- HPS-v2.1
# -------------------------------------------------------
echo ""
echo "Step 5: Installing HPS-v2.1..."

HPSV2_DIR="${WORK_DIR}/HPSv2"

if [ ! -d "${HPSV2_DIR}/.git" ]; then
    echo "  Cloning HPSv2..."
    git clone https://github.com/tgxs002/HPSv2.git "${HPSV2_DIR}"
else
    echo "  HPSv2 already cloned, skipping."
fi

if conda run -n "${CONDA_ENV}" python -c "import hpsv2" 2>/dev/null; then
    echo "  hpsv2 already installed, skipping."
else
    conda run -n "${CONDA_ENV}" pip install -e "${HPSV2_DIR}"
fi

# -------------------------------------------------------
# 8. Step 6 -- Verify no conflicts
# -------------------------------------------------------
echo ""
echo "Step 6: Verifying all imports..."

conda run -n "${CONDA_ENV}" python -c "
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
# 9. HuggingFace login (prompts interactively)
# -------------------------------------------------------
echo ""
echo "======================================================"
echo "  HuggingFace Login"
echo "  You need an HF account with access to:"
echo "    - black-forest-labs/FLUX.2-dev  (gated, accept license at hf.co first)"
echo "======================================================"
conda run -n "${CONDA_ENV}" huggingface-cli login

# -------------------------------------------------------
# Done
# -------------------------------------------------------
echo ""
echo "======================================================"
echo "  Setup complete!"
echo ""
echo "  Activate env:  conda activate ${CONDA_ENV}"
echo "  Download models: bash scripts/download_models.sh"
echo "  Start training:  bash scripts/train_openbanana.sh"
echo "======================================================"
