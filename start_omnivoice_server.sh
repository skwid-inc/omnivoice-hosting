#!/bin/bash
# OmniVoice block-wise streaming server launcher.
#
# Streaming enabled, block_size=32, first_block_num_step=8, all other knobs at
# the tuned defaults verified on H100 PCIe.
#
# Usage:
#   ./start_omnivoice_server.sh              # runs in foreground
#   nohup ./start_omnivoice_server.sh > server.log 2>&1 &   # background
#
# Requires: a Python env with `vllm-omni` installed and `transformers>=5.3.0`
# for voice cloning (HiggsAudioV2TokenizerModel).

set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8091}"
MODEL="${MODEL:-k2-fsa/OmniVoice}"

VENV_PATH="${VENV_PATH:-/home/ubuntu/vllm_omni_env/.venv}"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
fi

VLLM_BIN="${VLLM_BIN:-$(command -v vllm || true)}"
if [ -z "${VLLM_BIN}" ]; then
    echo "ERROR: vllm binary not found. Activate the venv or set VLLM_BIN." >&2
    exit 1
fi

export VLLM_OMNI_OMNIVOICE_FIRST_BLOCK_NUM_STEP="${VLLM_OMNI_OMNIVOICE_FIRST_BLOCK_NUM_STEP:-8}"
export VLLM_OMNI_OMNIVOICE_NUM_STEP="${VLLM_OMNI_OMNIVOICE_NUM_STEP:-32}"
export VLLM_OMNI_OMNIVOICE_BLOCK_SIZE="${VLLM_OMNI_OMNIVOICE_BLOCK_SIZE:-32}"
export VLLM_OMNI_OMNIVOICE_STREAM_HOLDBACK_FRAMES="${VLLM_OMNI_OMNIVOICE_STREAM_HOLDBACK_FRAMES:-2}"

export VLLM_OMNI_OMNIVOICE_OPT="${VLLM_OMNI_OMNIVOICE_OPT:-1}"
export VLLM_OMNI_OMNIVOICE_COMPILE_MODE="${VLLM_OMNI_OMNIVOICE_COMPILE_MODE:-default}"
export VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE="${VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE:-bf16}"

export VLLM_OMNI_DIFFUSION_CONCURRENT="${VLLM_OMNI_DIFFUSION_CONCURRENT:-1}"
export VLLM_OMNI_DIFFUSION_BATCH_SIZE="${VLLM_OMNI_DIFFUSION_BATCH_SIZE:-16}"
export VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS="${VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS:-10}"
export VLLM_OMNI_DIFFUSION_BATCH_STRATEGY="${VLLM_OMNI_DIFFUSION_BATCH_STRATEGY:-duration_bucket}"
export VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS="${VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS:-128}"

echo "Starting OmniVoice block-streaming server"
echo "  model      : ${MODEL}"
echo "  endpoint   : http://${HOST}:${PORT}"
echo "  vllm bin   : ${VLLM_BIN}"
echo ""
echo "  block_size              : ${VLLM_OMNI_OMNIVOICE_BLOCK_SIZE}"
echo "  first_block_num_step    : ${VLLM_OMNI_OMNIVOICE_FIRST_BLOCK_NUM_STEP}"
echo "  num_step (blocks 1..N)  : ${VLLM_OMNI_OMNIVOICE_NUM_STEP}"
echo "  stream_holdback_frames  : ${VLLM_OMNI_OMNIVOICE_STREAM_HOLDBACK_FRAMES}"
echo ""
echo "  diffusion batch_size    : ${VLLM_OMNI_DIFFUSION_BATCH_SIZE}"
echo "  diffusion batch_wait_ms : ${VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS}"
echo "  diffusion strategy      : ${VLLM_OMNI_DIFFUSION_BATCH_STRATEGY}"
echo "  duration_bucket_tokens  : ${VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS}"
echo ""
echo "  generator dtype         : ${VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE}"
echo "  compile mode            : ${VLLM_OMNI_OMNIVOICE_COMPILE_MODE}"
echo ""

exec "${VLLM_BIN}" serve "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --trust-remote-code \
    --omni \
    --step-execution
