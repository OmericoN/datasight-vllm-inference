FROM vllm/vllm-openai:v0.32.0

ENV HF_HOME=/hf-cache
ENV HUGGINGFACE_HUB_CACHE=/hf-cache/hub
ENV TRANSFORMERS_CACHE=/hf-cache/transformers
ENV XDG_CACHE_HOME=/hf-cache/xdg
ENV VLLM_CACHE_ROOT=/hf-cache/vllm
ENV TRITON_CACHE_DIR=/hf-cache/triton
ENV TORCHINDUCTOR_CACHE_DIR=/hf-cache/torchinductor
ENV HOME=/tmp
ENV USER=datasight
ENV LOGNAME=datasight
ENV VLLM_MODEL=Qwen/Qwen3.8-27B
ENV VLLM_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
ENV HOST=0.0.0.0
ENV PORT=8000
ENV VLLM_DTYPE=bfloat16
ENV VLLM_START_MODE=serve

COPY entrypoint-vllm.sh /entrypoint-vllm.sh
RUN mkdir -p /hf-cache \
  && chmod 0777 /hf-cache \
  && chmod +x /entrypoint-vllm.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint-vllm.sh"]
