#!/usr/bin/env bash
# Release-candidate validation for vanilla 2V/2A against the shipped 2V/4A point.
#
#   ./run_rc_validation.sh            # waits for an idle fleet, then runs everything
#
# Two gates, in this order because the second one needs all 8 GPUs:
#
#   1. LATENCY, repeated on three separate GPUs. The 1.340x figure rests on one ABBA run on one
#      device, and the P007 certificate showed the conv-layout path can be the noisier of two arms,
#      so a single-device number is not release evidence.
#   2. QUALITY. Extends the SHIPPED arm to 24 episodes/task (1200) on the same pinned seeds the
#      candidate already covers, then certifies. Writes a NEW run dir rather than touching
#      actsweep_v2a4, because that existing 500-episode arm is needed as the noise-floor reference:
#      indices 0-9 get measured twice under an IDENTICAL configuration, and the discordance between
#      those two runs is how much of any observed difference is the harness rather than the schedule.
#
# Refuses to start on a contended fleet. Every probe in this directory does the same, and a latency
# number measured against someone else's training job is not a number.
set -u
cd "$(dirname "$0")" && source ./env.sh

IDLE_MEM=${IDLE_MEM:-8000}          # MiB per GPU below which we call it idle
WAIT_HOURS=${WAIT_HOURS:-24}
LOG=${IWM_LOG_DIR:-/tmp}/rc_validation.log
exec > >(tee -a "$LOG") 2>&1
echo "=== RC validation started $(date -u +%FT%TZ) ==="

fleet_idle() {
  local busy
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
         awk -v m="$IDLE_MEM" '$1 > m {n++} END {print n+0}')
  [ "$busy" -eq 0 ]
}

echo "waiting for an idle fleet (up to ${WAIT_HOURS}h) ..."
for _ in $(seq 1 $((WAIT_HOURS * 60))); do
  fleet_idle && break
  sleep 60
done
if ! fleet_idle; then
  echo "NOT EVALUATED: fleet still contended after ${WAIT_HOURS}h"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  exit 2
fi
echo "fleet idle at $(date -u +%FT%TZ)"

# ---- GATE 1: latency, three independent devices ------------------------------------------------
echo; echo "=== GATE 1: ABBA latency, 2V/2A vs shipped 2V/4A, on 3 GPUs ==="
# IDLENESS IS AN INVARIANT, NOT A PRECONDITION. The first run of this script checked once at the
# start, passed, and then produced two uninterpretable arms out of three -- repeated IDENTICAL treat
# arms differing by 42% and 51%, one of them slower than base despite running fewer forwards. An
# unrelated workload had started during the run. Re-checked around every arm now, and a run that
# loses the fleet reports NOT EVALUATED instead of a ratio.
for g in 1 2 3; do
  echo "--- GPU $g ---"
  if ! fleet_idle; then
    echo "NOT EVALUATED: fleet became contended before the GPU $g arm"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    continue
  fi
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH="$IWM_FA_SHIM_DIR:/home/ubuntu/InstinctWM" \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29910 + g)) WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    timeout 3000 "$IWM_SERVER_PY" probe_nfe_latency.py 2>&1 |
    grep -E "base |treat |drift|spread|SPEEDUP|NOT EVAL"
  fleet_idle || echo "  WARNING: fleet contended by the END of the GPU $g arm -- discard this row"
done

# ---- GATE 2: quality, extend the shipped arm then certify --------------------------------------
echo; echo "=== GATE 2: extending the SHIPPED arm to 24 episodes/task (1200) ==="
if [ -f "$IWM_RESULT_DIR/rc_v2a4/episodes.jsonl" ]; then
  echo "rc_v2a4 already present, skipping the run"
else
  ./run_sweep.sh rc 24 tasks50.txt 2:4
fi

echo; echo "=== CERTIFICATE ==="
"$IWM_SERVER_PY" certify_operating_point.py \
  --control rc_v2a4 --treat fastcert_v2a2 --repeat actsweep_v2a4 \
  --margin -0.05 --label "vanilla 2V/2A vs shipped 2V/4A, n=1200" \
  --out "$IWM_RESULT_DIR/rc_2v2a_certificate.json"
rc=$?
echo
echo "=== RC validation finished $(date -u +%FT%TZ), certificate exit=$rc ==="
exit $rc
