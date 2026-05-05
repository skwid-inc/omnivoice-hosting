#!/bin/bash
# Sequentially run the Phase A harness for {0, 32, 64, 128} block sizes.
# block_size=0 verifies bit-exact equivalence with F12 baseline.
set -uo pipefail

cd "$(dirname "$0")/../../.."

mkdir -p outputs/phase_a
TS="$(date -u +%Y%m%dT%H%M%SZ)"
TOP_LOG="outputs/phase_a/all_${TS}.log"
echo "[all] start ${TS}" | tee "$TOP_LOG"

for BS in 0 32 64 128; do
    LABEL="bs${BS}"
    echo "[all] === starting BLOCK_SIZE=$BS label=$LABEL ===" | tee -a "$TOP_LOG"
    bash benchmarks/tts/scripts/run_phase_a_sweep.sh "$BS" "$LABEL" 2>&1 | tee -a "$TOP_LOG"
    EXIT=${PIPESTATUS[0]}
    echo "[all] BLOCK_SIZE=$BS exit=$EXIT" | tee -a "$TOP_LOG"
    sleep 5
    pkill -KILL -f 'vllm serve' 2>/dev/null || true
    sleep 5
done
echo "[all] done" | tee -a "$TOP_LOG"
