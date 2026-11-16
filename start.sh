#!/usr/bin/env bash
# Run two independent llama.cpp servers: one full model replica per V100.
set -euo pipefail

# shellcheck source=config.env
source "$(dirname -- "${BASH_SOURCE[0]}")/config.env"

if [[ -n "$LLAMA_CPP_DIR" ]]; then
  LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_CPP_DIR/build/bin/llama-server}"
else
  LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
fi
HOST="${HOST:-127.0.0.1}"
PORT0="${PORT0:-8081}"
PORT1="${PORT1:-8082}"
SLOTS="${SLOTS:-2}"
CTX_PER_SLOT="${CTX_PER_SLOT:-32768}"
TOTAL_CTX=$(( SLOTS * CTX_PER_SLOT ))

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: Model not found: $MODEL_PATH" >&2
  echo 'Set MODEL_PATH in config.env or in the environment.' >&2
  exit 1
fi

if ! command -v -- "$LLAMA_SERVER" >/dev/null; then
  echo "ERROR: llama-server executable not found: $LLAMA_SERVER" >&2
  echo 'Set LLAMA_CPP_DIR in config.env or set LLAMA_SERVER to the executable path.' >&2
  exit 1
fi

start_server() {
  local gpu="$1" port="$2"
  echo "Starting GPU ${gpu} on ${HOST}:${port}: slots=${SLOTS}, total_ctx=${TOTAL_CTX}"
  CUDA_VISIBLE_DEVICES="$gpu" "$LLAMA_SERVER" \
    -m "$MODEL_PATH" \
    --host "$HOST" \
    --port "$port" \
    --split-mode none \
    -ngl all \
    -np "$SLOTS" \
    -c "$TOTAL_CTX" \
    -cb \
    -b 1024 \
    -ub 256 \
    -fa auto \
    --reasoning off \
    --metrics &
}

start_server 0 "$PORT0"
PID0=$!
start_server 1 "$PORT1"
PID1=$!

cleanup() {
  trap - EXIT INT TERM
  kill "$PID0" "$PID1" 2>/dev/null || true
  wait "$PID0" "$PID1" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Exit and shut down the other server if either one crashes.
wait -n "$PID0" "$PID1" || true
