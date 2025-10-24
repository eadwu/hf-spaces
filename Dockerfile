FROM nvcr.io/nvidia/pytorch:25.09-py3

RUN --mount=type=bind,source=./requirements.txt,target=/requirements.txt \
  pip3 install --no-cache-dir -r /requirements.txt && pip3 install --no-cache-dir spaces

RUN apt update && apt install -y git && apt-get clean && rm -rf /var/lib/apt/lists/* && \
  git clone --depth 1 https://github.com/NVlabs/Fast-dLLM.git /dllm && \
  cd / && rm -rf /workspace && mkdir -p /workspace && mv /dllm/llada/* /workspace && rm -rf /dllm && \
  cd /workspace && sed -i 's@share=True@server_name="0.0.0.0", server_port=7860@' app.py

WORKDIR /workspace
ENTRYPOINT ["python3"]
CMD ["app.py"]
