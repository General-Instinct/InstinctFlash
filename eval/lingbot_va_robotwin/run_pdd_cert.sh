#!/usr/bin/env bash
# Certify the heads-only PDD student against its correct control, on identical pinned seeds.
#
#   ./run_pdd_cert.sh <ckpt_dir> <episodes_per_task> [task ...]
#
# THREE ARMS, and the middle one is the point:
#   A  teacher            25 video / 50 action   -- already collected as cert50_teacher
#   B  untrained 2/50     --degrade-nfe 2,50     -- the CONTROL
#   C  PDD student 2/50   --pdd-heads <ckpt>
#
# B is what makes this interpretable. Comparing C to the teacher alone conflates "PDD works" with
# "2 video steps are enough anyway" -- and our own sweep already showed the shipped checkpoint
# tolerates fewer steps than it runs. B - A is the cost of step reduction; C - B is what PDD bought.
#
# Note the sweep's existing arms all degraded BOTH streams (1:1, 2:2, ...). None of them is the right
# control here, because the student only distilled video: its action stream still runs 50 steps.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" && source ./env.sh

CKPT=${1:?usage: run_pdd_cert.sh <ckpt_dir> <episodes_per_task> [task ...]}
N=${2:?}
shift 2
TASKS=("$@")
[ ${#TASKS[@]} -gt 0 ] || TASKS=(adjust_bottle click_bell open_laptop place_empty_cup move_can_pot \
                                 beat_block_hammer lift_pot open_microwave place_shoe stack_blocks_two)

[ -f "$CKPT/delta.json" ] || { echo "no delta.json in $CKPT" >&2; exit 2; }
python3 -c "
import json,sys
d=json.load(open('$CKPT/delta.json'))
sys.exit(0 if d.get('coverage_gate_pass') else 1)" || {
  echo "REFUSING: $CKPT did not pass the coverage gate" >&2; exit 2; }

LOGD="$IWM_LOG_DIR/pdd_cert"; mkdir -p "$LOGD"

start_arm() {   # start_arm <label> <first_gpu> <n_gpus> <extra args...>
  local label=$1 g0=$2 n=$3; shift 3
  for i in $(seq 0 $((n - 1))); do
    local gpu=$((g0 + i)) port=$((29056 + g0 + i))
    ( cd "$LINGBOT_ROOT" && nohup env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$IWM_FA_SHIM_DIR" \
        LINGBOT_CKPT="$LINGBOT_CKPT" setsid "$IWM_SERVER_PY" -m torch.distributed.run \
        --nproc_per_node 1 --master_port $((29800 + g0 + i)) \
        "$IWM_ROOT/eval/lingbot_va_robotwin/serve_variant.py" --config-name robotwin \
        --port $port --save_root /home/ubuntu/iwm_vis/pdd_cert \
        --no-fsdp --no-empty-cache --no-debug-dump --conditioning-prefill --ring-kv \
        "$@" > "$LOGD/${label}_$port.log" 2>&1 & )
  done
}

echo "starting 4 control servers (untrained 2/50) and 4 student servers ..."
start_arm control 0 4 --degrade-nfe 2,50
start_arm student 4 4 --block-heads "$CKPT"

for t in $(seq 1 90); do
  up=0
  for p in $(seq 29056 29063); do iwm_port_busy $p && up=$((up + 1)); done
  [ $up -eq 8 ] && { echo "all 8 servers up after $((t * 10))s"; break; }
  sleep 10
done
up=0; for p in $(seq 29056 29063); do iwm_port_busy $p && up=$((up + 1)); done
[ $up -eq 8 ] || { echo "only $up/8 servers came up; see $LOGD" >&2; exit 2; }

# run_paired.sh drives both arms over the SAME tasks and seeds, which is what makes the pairing --
# and therefore exact McNemar on the discordant pairs -- valid.
./run_paired.sh "${IWM_CERT_RUN:-pdd_cert}" "$N" \
  0:29056,1:29057,2:29058,3:29059 \
  4:29060,5:29061,6:29062,7:29063 \
  "${TASKS[@]}"
