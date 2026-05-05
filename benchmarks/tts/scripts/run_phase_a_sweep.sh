#!/bin/bash
# Phase A block-wise sweep harness.
# Boots a server with VLLM_OMNI_OMNIVOICE_BLOCK_SIZE=$1, runs the F12
# concurrency sweep + A/B generate, then tears the server down. Pass
# BLOCK_SIZE=0 (or unset) to confirm bit-exact equivalence with F12.
set -uo pipefail

BLOCK_SIZE="${1:-0}"
LABEL="${2:-blocksize_${BLOCK_SIZE}}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="outputs/phase_a/${LABEL}_${TS}"
mkdir -p "$OUT_ROOT"
SERVER_LOG="$OUT_ROOT/server.log"

echo "[harness] BLOCK_SIZE=$BLOCK_SIZE label=$LABEL out=$OUT_ROOT" | tee -a "$OUT_ROOT/harness.log"

set -a; [ -f .env.local ] && source .env.local; set +a

VLLM_OMNI_DIFFUSION_BATCH_STRATEGY=duration_bucket \
VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS=256 \
VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS=30 \
VLLM_OMNI_DIFFUSION_BATCH_SIZE=12 \
VLLM_OMNI_DIFFUSION_CONCURRENT=1 \
VLLM_OMNI_OMNIVOICE_OPT=1 \
VLLM_OMNI_OMNIVOICE_COMPILE_MODE=default \
VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE=bfloat16 \
VLLM_OMNI_OMNIVOICE_NUM_STEP=32 \
VLLM_OMNI_OMNIVOICE_BLOCK_SIZE="$BLOCK_SIZE" \
VLLM_OMNI_OMNIVOICE_GUMBEL_SEED=42 \
VLLM_BIN=.venv/bin/vllm \
bash examples/online_serving/omnivoice/run_server_optimized.sh > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[harness] server pid=$SERVER_PID" | tee -a "$OUT_ROOT/harness.log"

cleanup() {
    echo "[harness] killing server pid=$SERVER_PID" | tee -a "$OUT_ROOT/harness.log"
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    sleep 2
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    pkill -KILL -P "$SERVER_PID" 2>/dev/null || true
    pkill -KILL -f "vllm serve" 2>/dev/null || true
    sleep 2
}
trap cleanup EXIT

# Wait for ready (max 240s).
echo "[harness] waiting for server ready..." | tee -a "$OUT_ROOT/harness.log"
for i in $(seq 1 120); do
    if grep -q "Application startup complete" "$SERVER_LOG" 2>/dev/null; then
        echo "[harness] server ready after ${i}*2 seconds" | tee -a "$OUT_ROOT/harness.log"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[harness] server pid $SERVER_PID died" | tee -a "$OUT_ROOT/harness.log"
        tail -50 "$SERVER_LOG" | tee -a "$OUT_ROOT/harness.log"
        exit 1
    fi
    sleep 2
done
if ! grep -q "Application startup complete" "$SERVER_LOG" 2>/dev/null; then
    echo "[harness] TIMEOUT waiting for server" | tee -a "$OUT_ROOT/harness.log"
    tail -50 "$SERVER_LOG" | tee -a "$OUT_ROOT/harness.log"
    exit 1
fi

# Quick TCP smoketest.
for i in 1 2 3 4 5; do
    if curl -sf "http://127.0.0.1:8091/v1/models" -o /dev/null; then
        break
    fi
    sleep 1
done

echo "[harness] running concurrency sweep..." | tee -a "$OUT_ROOT/harness.log"
.venv/bin/python examples/online_serving/omnivoice/benchmark_concurrent.py \
  --api-base http://127.0.0.1:8091 \
  --concurrencies 1,2,4,8,16,32 \
  --candidate_count 32 \
  --warmup_iters 10 \
  --jitter_ms_min 0 --jitter_ms_max 0 \
  --out_dir "$OUT_ROOT/sweep" 2>&1 | tee -a "$OUT_ROOT/harness.log"

echo "[harness] running A/B generate..." | tee -a "$OUT_ROOT/harness.log"
.venv/bin/python benchmarks/tts/ab_quality.py generate \
  --api-base http://127.0.0.1:8091 \
  --label "$LABEL" \
  --out_dir "$OUT_ROOT/ab" 2>&1 | tee -a "$OUT_ROOT/harness.log"

echo "[harness] DONE label=$LABEL" | tee -a "$OUT_ROOT/harness.log"
