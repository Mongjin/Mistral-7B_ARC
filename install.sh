#!/usr/bin/env bash
set -euo pipefail

CUDA_VERSION="${CUDA_VERSION:-cu121}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
MAX_JOBS="${MAX_JOBS:-4}"

python -m pip install --upgrade pip wheel setuptools packaging ninja
python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
python -m pip install -r requirements.txt

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  MAX_JOBS="${MAX_JOBS}" python -m pip install flash-attn --no-build-isolation
fi
