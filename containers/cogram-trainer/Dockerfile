# Cogram trainer: Qwen 2.5 1.5B base + PEFT LoRA training + adapter hot-swap.
#
# Auto-detects GPU at runtime via torch.cuda.is_available():
#   - With NVIDIA GPU + nvidia-container-toolkit: trains on GPU
#   - Without GPU: trains on CPU (slower but works)
#
# Base is plain python:3.10-slim. The torch CUDA wheels include all needed
# CUDA runtime libs bundled — no nvidia/cuda base image needed on the host
# until you actually have a GPU.
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch CUDA wheels work on CPU too (torch.cuda.is_available() returns False).
# Single image runs on GPU and CPU hosts. Bundled CUDA libs are ~3GB but the
# trade-off is a single Docker image that "just works" everywhere.
RUN python -m pip install --upgrade pip && \
    python -m pip install \
        "torch==2.4.*" --index-url https://download.pytorch.org/whl/cu121 && \
    python -m pip install \
        "transformers>=4.45" \
        "peft>=0.13" \
        "accelerate>=1.0" \
        "datasets>=3.0" \
        "bitsandbytes>=0.43" \
        "safetensors>=0.4" \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.32" \
        "asyncpg>=0.30" \
        "redis>=5.1" \
        "pydantic>=2.9" \
        "nvidia-ml-py>=12" \
        "psutil>=6.0"

COPY containers/cogram-trainer/ ./

EXPOSE 7900

CMD ["python", "-m", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "7900", "--workers", "1"]
