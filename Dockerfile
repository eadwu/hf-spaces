FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /bin/

RUN apt update && apt install -y \
  git libgl1 libglx-mesa0 libglib2.0-0t64 build-essential python3-dev && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /workspace && cd /workspace && \
  uv venv --seed --python 3.12 && \
  uv pip --no-cache install --no-cache-dir -U comfy-cli && \
  echo "N" | uv run comfy tracking disable && \
  echo "y" | uv run comfy install --version v0.3.73 --restore --nvidia --cuda-version 12.9 && \
  uv pip --no-cache install --no-cache-dir -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130 && \
  rm -rf ~/.cache/pip && \
  sed -i 's@ and comfy.model_management.WINDOWS@@' /root/comfy/ComfyUI/comfy/ops.py

WORKDIR /workspace
ENTRYPOINT ["uv", "run", "comfy"]
CMD ["launch", "--", "--listen", "0.0.0.0", "--port", "8188"]
