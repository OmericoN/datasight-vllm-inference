#!/usr/bin/env bash
set -e

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.8-27B}"
export VLLM_REVISION="${VLLM_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${VLLM_MODEL}}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_START_MODE="${VLLM_START_MODE:-serve}"

case "${VLLM_START_MODE}" in
  serve|hold|download) ;;
  *) echo "ERROR: VLLM_START_MODE must be serve, hold, or download." >&2; exit 2 ;;
esac

export HF_HOME="${HF_HOME:-/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}}"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HF_HOME}/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HF_HOME}/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${HF_HOME}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${HF_HOME}/torchinductor}"
export HOME="${HOME:-/tmp}"
export USER="${USER:-datasight}"
export LOGNAME="${LOGNAME:-${USER}}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache}"

for cache in "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
  "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"; do
  if ! mkdir -p "${cache}" || ! probe_file="$(mktemp "${cache}/.write-test.XXXXXX")"; then
    echo "ERROR: cache ${cache} is not writable by uid $(id -u). Fix the PVC mount/permissions; temporary model storage is disabled." >&2
    exit 1
  fi
  rm -f "${probe_file}"
done
mkdir -p "${HOME}" "${PYTHONPYCACHEPREFIX}"

echo "DataSight vLLM: mode=${VLLM_START_MODE} model=${VLLM_MODEL} revision=${VLLM_REVISION}"
echo "Cache=${HF_HUB_CACHE} served-model=${SERVED_MODEL_NAME} dtype=${VLLM_DTYPE} listen=${HOST}:${PORT}"

if [[ "${VLLM_START_MODE}" == "hold" ]]; then
  echo "Hold mode: no inference or download; an allocated GPU is still consumed."
  exec tail -f /dev/null
fi

python_bin="$(command -v python || command -v python3 || true)"
if [[ -z "${python_bin}" ]]; then
  echo "ERROR: neither python nor python3 is available on PATH." >&2
  exit 127
fi

if [[ "${VLLM_START_MODE}" == "download" ]]; then
  exec "${python_bin}" - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=os.environ["VLLM_MODEL"],
    revision=os.environ["VLLM_REVISION"],
    cache_dir=os.environ["HF_HUB_CACHE"],
)
print(f"Pinned model snapshot ready: {path}")
PY
fi

args=(
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --revision "${VLLM_REVISION}"
  --tokenizer-revision "${VLLM_REVISION}"
  --dtype "${VLLM_DTYPE}"
  --download-dir "${HF_HUB_CACHE}"
  --tensor-parallel-size 1
  --max-model-len 16384
  --max-num-seqs 1
  --gpu-memory-utilization 0.90
  --language-model-only
  --reasoning-parser qwen3
  --default-chat-template-kwargs '{"enable_thinking":false}'
)

if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  # shortcut: whitespace-separated flags only; add complex JSON as an array element above.
  read -r -a extra_args <<< "${VLLM_EXTRA_ARGS}"
  args+=("${extra_args[@]}")
fi

exec vllm serve "${VLLM_MODEL}" "${args[@]}"
