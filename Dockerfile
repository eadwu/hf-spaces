FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

RUN apt update && apt install -y \
  python3 python3-pip espeak-ng ffmpeg && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,source=./openfst,target=/src \
  cp /src/openfst-1.8.2.tar.gz . && sha512sum --check /src/sha512sum.checksum && \
  mv openfst-1.8.2.tar.gz openfst.tar.gz
RUN tar xvf openfst.tar.gz && cd openfst-* && \
  ./configure --enable-grm && make -j $(nproc) && make install && \
  cd .. && rm -rf openfst*

# WordTextProcessing is pinned to 2.1.5 which is compatible with 1.8.2
RUN --mount=type=bind,source=./requirements.txt,target=/requirements.txt \
  pip3 install --no-cache-dir Cython && \
  env LD_LIBRARY_PATH=/usr/lib:/usr/local/lib pip3 install --no-cache-dir \
    pynini==2.1.5 && \
  pip3 install --no-cache-dir \
    torch==2.8.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/test/cu129 && \
  pip3 install --no-cache-dir -r /requirements.txt

RUN sed -i 's@pynini.union(\(.*\)))))@"".join([\1))]))@' /usr/local/lib/python3.10/dist-packages/pynini/lib/utf8.py && \
  sed -i 's@VALID_UTF8_CHAR_REGIONAL_INDICATOR_SYMBOL = (@VALID_UTF8_CHAR_REGIONAL_INDICATOR_SYMBOL = pynini.union(@' /usr/local/lib/python3.10/dist-packages/pynini/lib/utf8.py

RUN mkdir -p /workspace && pip3 install --no-cache-dir -U "huggingface_hub[cli]" && \
  cd /workspace && hf download IndexTeam/IndexTTS-2-Demo --repo-type=space --local-dir . && \
  rm -rf /root/.cache/huggingface && \
  mkdir checkpoints && chown 1000:1000 checkpoints && \
  sed -i 's@from modelscope@from huggingface_hub@' tools/download_files.py

WORKDIR /workspace
ENTRYPOINT ["python3"]
CMD ["webui.py"]
