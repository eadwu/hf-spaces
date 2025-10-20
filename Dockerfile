FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

RUN apt update && apt install -y \
  python3 python3-pip && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,source=./requirements.txt,target=/requirements.txt \
  pip3 install --no-cache-dir \
    torch==2.7.0 torchaudio==2.7.0 torchvision \
    --index-url https://download.pytorch.org/whl/test/cu128 && \
  pip3 install --no-cache-dir -r /requirements.txt && \
  pip3 install --no-cache-dir spaces

RUN mkdir -p /workspace && pip3 install --no-cache-dir -U "huggingface_hub[cli]" && \
  cd /workspace && hf download smola/higgs_audio_v2 --repo-type=space --local-dir . && \
  rm -rf /root/.cache/huggingface

RUN sed -i 's@, mcp_server=True@@' /workspace/app.py

WORKDIR /workspace
ENTRYPOINT ["python3"]
CMD ["app.py"]
