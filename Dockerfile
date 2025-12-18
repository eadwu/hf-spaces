FROM nginx:mainline-alpine
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apk add --no-cache python3 py3-pip python3-dev build-base

RUN mkdir -p /usr/share/nginx/html && pip3 install --no-cache-dir -U "huggingface_hub[cli]" && \
  cd /usr/share/nginx/html && hf download Supertone/supertonic --repo-type=space --local-dir . && \
  rm -rf /root/.cache/huggingface

WORKDIR /workspace