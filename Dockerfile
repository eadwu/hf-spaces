FROM nvcr.io/nvidia/pytorch:25.09-py3

RUN pip3 install --no-cache-dir transformers gradio spaces

RUN pip3 install --no-cache-dir -U "huggingface_hub[cli]" && \
  cd / && rm -rf /workspace && mkdir -p /workspace && \
  cd /workspace && hf download multimodalart/LLaDA --repo-type=space --local-dir . && \
  rm -rf /root/.cache/huggingface && \
  sed -i 's@share=True@server_name="0.0.0.0", server_port=7860@' app.py

WORKDIR /workspace
ENTRYPOINT ["python3"]
CMD ["app.py"]
