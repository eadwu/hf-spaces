FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

RUN apt update && apt install -y \
  git \
  python3 python3-pip espeak-ng && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,source=./requirements.txt,target=/requirements.txt \
  pip3 install --no-cache-dir \
    torch==2.8.0 torchaudio==2.8.0 torchvision \
    --index-url https://download.pytorch.org/whl/test/cu129 && \
  pip3 install --no-cache-dir -r /requirements.txt && \
  pip3 install --no-cache-dir spaces

RUN mkdir -p /workspace && \
  pip3 install --no-cache-dir -U "huggingface_hub[cli]" && \
  cd /workspace && hf download neuphonic/neutts-air --repo-type=space --local-dir . && \
  git clone https://github.com/neuphonic/neutts-air.git && rm -rf neutts-air/.git \
  rm -rf /root/.cache/huggingface

RUN sed -i 's@\(demo.launch.*\)mcp.*)@\1server_name="0.0.0.0", server_port=7860)@' /workspace/app.py

WORKDIR /workspace
ENTRYPOINT ["python3"]
CMD ["app.py"]
