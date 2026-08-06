#!/usr/bin/env bash
# Step-allocation sweep: one (video, action) NFE configuration at a time, all 8 GPUs on it.
#
#   ./run_sweep.sh <run_prefix> <test_num> <task_file> <V:A> [<V:A> ...]
#   ./run_sweep.sh sweep 10 tasks.txt 1:1 1:2 2:1 2:2 2:3 3:2 3:3
#
# Not paired against a teacher per config -- the teacher arm is measured ONCE (cert50) and every
# configuration is paired against that same reference. Same episodes, same seeds, so the pairing
# still holds; it just avoids re-running the teacher seven times.
#
# Servers are restarted per configuration because --degrade-nfe is applied at startup. That costs
# ~3 min per config and is worth it: mutating NFE on a live server would leave the KV pool and any
# captured graphs sized for the previous configuration.
set -u
cd "$(dirname "$0")" && source ./env.sh

PREFIX=${1:?usage: run_sweep.sh <prefix> <test_num> <task_file> <V:A>...}
N=${2:?}; TASKFILE=${3:?}; shift 3
CONFIGS=("$@"); [ ${#CONFIGS[@]} -gt 0 ] || { echo "no configs" >&2; exit 2; }
mapfile -t TASKS < "$TASKFILE"
[ ${#TASKS[@]} -gt 0 ] || { echo "no tasks in $TASKFILE" >&2; exit 2; }
echo "sweep: ${#CONFIGS[@]} configs x ${#TASKS[@]} tasks x $N episodes"

stop_servers() {
  for p in $(seq 29056 29063); do
    pids=$(ps -eo pid,cmd | grep serve_variant | grep -v grep | grep -- "--port $p" | awk '{print $1}')
    [ -n "$pids" ] && kill -9 $pids 2>/dev/null
  done
  sleep 8
}

start_servers() {   # start_servers <video> <action>
  local v=$1 a=$2 i
  for i in 0 1 2 3 4 5 6 7; do
    ( cd "$LINGBOT_ROOT" && nohup env CUDA_VISIBLE_DEVICES=$i PYTHONPATH="$IWM_FA_SHIM_DIR" \
        LINGBOT_CKPT="$LINGBOT_CKPT" setsid "$IWM_SERVER_PY" -m torch.distributed.run \
        --nproc_per_node 1 --master_port $((29840+i)) \
        /home/ubuntu/InstinctWM/eval/lingbot_va_robotwin/serve_variant.py --config-name robotwin \
        --port $((29056+i)) --save_root /home/ubuntu/iwm_vis/sweep \
        --no-fsdp --no-empty-cache --no-debug-dump --conditioning-prefill --ring-kv \
        --degrade-nfe "$v,$a" > "$IWM_LOG_DIR/sweep_srv_$((29056+i)).log" 2>&1 & )
  done
  local tries=0
  while :; do
    local up=0
    for i in 0 1 2 3 4 5 6 7; do iwm_port_busy $((29056+i)) && up=$((up+1)); done
    [ $up -eq 8 ] && break
    tries=$((tries+1)); [ $tries -gt 90 ] && { echo "servers failed to start" >&2; return 1; }
    sleep 10
  done
}

for cfg in "${CONFIGS[@]}"; do
  V=${cfg%%:*}; A=${cfg##*:}
  RUN="${PREFIX}_v${V}a${A}"
  ROOT="$IWM_RESULT_DIR/$RUN"; LOGD="$IWM_LOG_DIR/$RUN"
  if [ -f "$ROOT/episodes.jsonl" ]; then echo "== $RUN already done, skipping"; continue; fi
  echo "== $RUN : video=$V action=$A =="
  stop_servers; start_servers "$V" "$A" || { echo "SKIP $RUN"; continue; }

  mkdir -p "$ROOT" "$LOGD"; : > "$LOGD/_progress.txt"; : > "$LOGD/_lock"
  printf '%s\n' "${TASKS[@]}" > "$LOGD/_queue"
  # latency for this configuration, before the eval loads the servers
  "$IWM_SERVER_PY" probe_latency.py --port 29056 --cycles 10 --repeats 3 \
    > "$LOGD/_latency.txt" 2>&1
  grep -m1 "steady-state" "$LOGD/_latency.txt" || true

  pids=()
  for i in 0 1 2 3 4 5 6 7; do
    (
      while :; do
        t=$( { flock 9; head -n1 "$LOGD/_queue"; sed -i 1d "$LOGD/_queue"; } 9<>"$LOGD/_lock" )
        [ -n "$t" ] || break
        ( cd "$ROBOTWIN_ROOT" && ROBOTWIN_ROOT="$ROBOTWIN_ROOT" \
          PYTHONWARNINGS=ignore::UserWarning CUDA_VISIBLE_DEVICES=$i \
          "$IWM_CLIENT_PY" -m evaluation.robotwin.eval_polict_client_openpi \
            --config policy/ACT/deploy_policy.yml --overrides \
            --task_name "$t" --task_config demo_clean --train_config_name 0 --model_name 0 \
            --ckpt_setting "$RUN" --seed 0 --policy_name LingBotVA --save_root "$ROOT" \
            --video_guidance_scale 5 --action_guidance_scale 1 \
            --test_num "$N" --port $((29056+i)) ) > "$LOGD/$t.log" 2>&1
        echo "$(date -u +%H:%M:%S) $t rc=$?" >> "$LOGD/_progress.txt"
      done
    ) & pids+=($!)
  done
  wait "${pids[@]}"
  "$IWM_SERVER_PY" emit_episodes.py "$ROOT" -o "$ROOT/episodes.jsonl" \
    || echo "WARNING: $RUN episode emission refused; not certifiable" >&2
  echo "== $RUN done =="
done
echo "SWEEP COMPLETE"
