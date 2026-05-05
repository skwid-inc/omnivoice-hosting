#!/bin/bash
# A100 batching-strategy sweep at c=8.
# Per config: stop any running server, set env, boot, wait ready, bench, stop.
# Outputs land in /tmp/sweep/<name>/{server.log,bench.log}.

set -uo pipefail

REPO=/home/ubuntu/vllm-omni
VENV="$REPO/.venv"
HF_TOKEN=$(grep -E "^HUGGINGFACE_TOKEN=" "$REPO/.env.local" | head -1 | cut -d= -f2-)
OUT=/tmp/sweep
mkdir -p "$OUT"

stop_server() {
  pkill -f "vllm serve" 2>/dev/null || true
  for _ in $(seq 1 30); do
    pgrep -f "vllm serve" >/dev/null 2>&1 || return 0
    sleep 0.5
  done
  pkill -9 -f "vllm serve" 2>/dev/null || true
  sleep 1
}

wait_ready() {
  local logfile=$1
  for _ in $(seq 1 600); do
    grep -q "Application startup complete" "$logfile" 2>/dev/null && return 0
    grep -qE "Traceback|RuntimeError|Address already in use|exited with" "$logfile" 2>/dev/null && return 1
    sleep 1
  done
  return 1
}

run_one() {
  local name=$1; shift
  local dir="$OUT/$name"
  mkdir -p "$dir"
  echo "=== $name ==="
  stop_server
  rm -f "$dir/server.log" "$dir/bench.log"

  env \
    PATH="$VENV/bin:$PATH" \
    VLLM_OMNI_OMNIVOICE_NUM_STEP=32 \
    VLLM_OMNI_OMNIVOICE_LOG_BATCH=1 \
    "$@" \
    nohup bash "$REPO/examples/online_serving/omnivoice/run_server_optimized.sh" \
    > "$dir/server.log" 2>&1 &
  echo "  server pid $!"

  if ! wait_ready "$dir/server.log"; then
    echo "  FAILED to start"
    tail -10 "$dir/server.log"
    return 1
  fi
  echo "  ready"

  HF_TOKEN="$HF_TOKEN" \
    "$VENV/bin/python" "$REPO/examples/online_serving/omnivoice/benchmark_concurrent.py" \
    --concurrencies 8 \
    --candidate_count 32 \
    --warmup_iters 5 \
    --jitter_ms_min 0 --jitter_ms_max 0 \
    > "$dir/bench.log" 2>&1
  echo "  bench done"
  grep -E "c=8.*req/s" "$dir/bench.log" | tail -1
  echo "  B histogram:"
  grep "OmniVoiceGenerator.forward" "$dir/server.log" | grep -oE "B=[0-9]+" | sort | uniq -c | sort -rn | sed 's/^/    /'
}

# A. baseline (current optimized)
run_one A_baseline \
  VLLM_OMNI_DIFFUSION_BATCH_STRATEGY=duration_bucket \
  VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS=128 \
  VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS=0 \
  VLLM_OMNI_DIFFUSION_CONCURRENT=1

# B. wider bucket
run_one B_bucket256 \
  VLLM_OMNI_DIFFUSION_BATCH_STRATEGY=duration_bucket \
  VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS=256 \
  VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS=0 \
  VLLM_OMNI_DIFFUSION_CONCURRENT=1

# C. very wide bucket
run_one C_bucket512 \
  VLLM_OMNI_DIFFUSION_BATCH_STRATEGY=duration_bucket \
  VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS=512 \
  VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS=0 \
  VLLM_OMNI_DIFFUSION_CONCURRENT=1

# D. fifo (no bucketing)
run_one D_fifo \
  VLLM_OMNI_DIFFUSION_BATCH_STRATEGY=fifo \
  VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS=0 \
  VLLM_OMNI_DIFFUSION_CONCURRENT=1

# E. wait + narrow bucket
run_one E_wait30_b128 \
  VLLM_OMNI_DIFFUSION_BATCH_STRATEGY=duration_bucket \
  VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS=128 \
  VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS=30 \
  VLLM_OMNI_DIFFUSION_CONCURRENT=1

# F. wait + wider bucket
run_one F_wait30_b256 \
  VLLM_OMNI_DIFFUSION_BATCH_STRATEGY=duration_bucket \
  VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS=256 \
  VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS=30 \
  VLLM_OMNI_DIFFUSION_CONCURRENT=1

# G. concurrent=2
run_one G_concurrent2 \
  VLLM_OMNI_DIFFUSION_BATCH_STRATEGY=duration_bucket \
  VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS=128 \
  VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS=0 \
  VLLM_OMNI_DIFFUSION_CONCURRENT=2

stop_server
echo "=== sweep done ==="
