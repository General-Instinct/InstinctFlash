#!/usr/bin/env bash
# Fan a list of RoboTwin 2.0 tasks across the 8 policy servers, one client per GPU.
#
#   ./run_eval.sh <run_name> <test_num> <task...>
#   ./run_eval.sh smoke8 10 adjust_bottle place_dual_shoes ...
#
# Differences from upstream launch_client_multigpus.sh, each deliberate:
#
#  * policy_name=LingBotVA, not ACT. Upstream passes policy_name=ACT purely to locate
#    policy/ACT/deploy_policy.yml (a config skeleton), with the side effect that every
#    result lands in eval_result/<task>/ACT/... and is indistinguishable from a genuine
#    ACT baseline. We keep --config pointed at the ACT skeleton (nothing in this client
#    imports an ACT model) but label the output honestly.
#
#  * NO DUPLICATE TASKS in one run. Success is counted by calc_stat.py from mp4
#    filenames of the form <test_num>_<prompt>_<True|False>.mp4, and test_num restarts
#    at 0 in every client process. Two clients running the same task into one save_root
#    therefore silently overwrite each other. Upstream's group 6 does exactly this
#    (place_empty_cup and blocks_ranking_rgb four times each). We refuse duplicates.
#
#  * Every port is checked before dispatch. WebsocketClientPolicy._wait_for_server
#    retries on ANY exception every 5s forever, so a client aimed at a dead server hangs
#    silently rather than failing -- indistinguishable from a slow task.
#
#  * One task per GPU at a time. The server holds per-episode KV state keyed to a single
#    episode (frame_st_id, transformer cache 'pos'); two concurrent clients on one server
#    would interleave resets and corrupt each other's rollouts with no error.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

RUN_NAME=${1:?usage: run_eval.sh <run_name> <test_num> <task...>}
TEST_NUM=${2:?usage: run_eval.sh <run_name> <test_num> <task...>}
shift 2
TASKS=("$@")
[ ${#TASKS[@]} -gt 0 ] || { echo "no tasks given" >&2; exit 2; }

# -- refuse duplicates (see note above) ---------------------------------------
dupes=$(printf '%s\n' "${TASKS[@]}" | sort | uniq -d)
if [ -n "$dupes" ]; then
  echo "REFUSING: duplicate tasks in one run would overwrite each other's mp4s:" >&2
  echo "$dupes" >&2
  exit 2
fi

# -- refuse unknown tasks ------------------------------------------------------
for t in "${TASKS[@]}"; do
  [ -f "$ROBOTWIN_ROOT/envs/$t.py" ] || { echo "REFUSING: no such task env: $t" >&2; exit 2; }
done

# -- refuse to dispatch at a dead server --------------------------------------
for i in $(seq 0 $((IWM_NUM_GPUS-1))); do
  p=$(iwm_ws_port "$i")
  iwm_port_busy "$p" || { echo "REFUSING: no server listening on port $p (gpu $i). Run ./servers.sh start" >&2; exit 2; }
done

SAVE_ROOT="$IWM_RESULT_DIR/$RUN_NAME"
LOG_DIR="$IWM_LOG_DIR/$RUN_NAME"
mkdir -p "$SAVE_ROOT" "$LOG_DIR"

# Export rather than using a `${VAR:+VAR=$VAR}` prefix on the client command: a word that
# only LOOKS like an assignment after expansion is not treated as one by bash, it is
# treated as the command name, and the client dies with rc=127. Export makes the worker
# subshells inherit them normally.
[ -n "${IWM_SEED_CACHE:-}" ] && { export IWM_SEED_CACHE; mkdir -p "$IWM_SEED_CACHE"; }
[ -n "${IWM_ACTION_LOG:-}" ] && { export IWM_ACTION_LOG; mkdir -p "$IWM_ACTION_LOG"; }

# -- provenance: what code actually produced this run -------------------------
{
  echo "run_name=$RUN_NAME"
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "test_num=$TEST_NUM"
  echo "n_tasks=${#TASKS[@]}"
  echo "tasks=${TASKS[*]}"
  echo "ckpt=$LINGBOT_CKPT"
  echo "lingbot_head=$(git -C "$LINGBOT_ROOT" rev-parse HEAD 2>/dev/null)"
  echo "lingbot_dirty=$(git -C "$LINGBOT_ROOT" status --porcelain | tr '\n' ';')"
  echo "robotwin_head=$(git -C "$ROBOTWIN_ROOT" rev-parse HEAD 2>/dev/null)"
  echo "fa_shim=${IWM_FA_SHIM:-0}"
  echo "# numerics-defining files:"
  sha256sum "$LINGBOT_ROOT/wan_va/wan_va_server.py" \
            "$LINGBOT_ROOT/wan_va/modules/model.py" \
            "$LINGBOT_ROOT/wan_va/configs/va_robotwin_cfg.py" \
            "$LINGBOT_ROOT/evaluation/robotwin/eval_polict_client_openpi.py" 2>/dev/null
} > "$SAVE_ROOT/_provenance.txt"
cat "$SAVE_ROOT/_provenance.txt"

echo
echo "dispatching ${#TASKS[@]} tasks, ${TEST_NUM} episodes each, over $IWM_NUM_GPUS GPUs"
echo "save_root=$SAVE_ROOT"
echo

# One background WORKER per GPU, each running its share of the tasks SERIALLY.
# (A previous shape of this script backgrounded each client inside a subshell and then
# called `wait` in the parent -- the parent has no such jobs, so it returned instantly
# and reported success while every client was still running.)
run_one() {                       # run_one <task> <gpu> <port>
  local t=$1 gpu=$2 port=$3
  local log="$LOG_DIR/$t.log"
  ( cd "$ROBOTWIN_ROOT" && \
    ROBOTWIN_ROOT="$ROBOTWIN_ROOT" PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES=$gpu \
    "$IWM_CLIENT_PY" -m evaluation.robotwin.eval_polict_client_openpi \
      --config policy/ACT/deploy_policy.yml \
      --overrides \
      --task_name "$t" --task_config demo_clean \
      --train_config_name 0 --model_name 0 --ckpt_setting "$RUN_NAME" --seed 0 \
      --policy_name LingBotVA --save_root "$SAVE_ROOT" \
      --video_guidance_scale 5 --action_guidance_scale 1 \
      --test_num "$TEST_NUM" --port "$port" ) > "$log" 2>&1
  local rc=$?
  echo "$(date -u +%H:%M:%S) done $t (gpu$gpu) rc=$rc" | tee -a "$LOG_DIR/_progress.txt"
  [ $rc -ne 0 ] && echo "$t rc=$rc" >> "$LOG_DIR/_failed.txt"
  return 0
}

: > "$LOG_DIR/_progress.txt"; : > "$LOG_DIR/_failed.txt"

# Longest-Processing-Time-first over a SHARED work queue, not a static round-robin slice.
# Task cost varies ~4x (step_lim 400 for adjust_bottle vs 1700 for put_bottles_dustbin,
# task_config/_eval_step_limit.yml), so a static assignment leaves most of the fleet idle
# while one GPU finishes a 1700-step task alone. LPT + work stealing is the standard
# makespan heuristic and needs ~10 lines here.
QUEUE="$LOG_DIR/_queue.txt"
LOCK="$LOG_DIR/_queue.lock"
python3 - "$ROBOTWIN_ROOT/task_config/_eval_step_limit.yml" "$QUEUE" "${TASKS[@]}" <<'PY'
import sys, re
limits_path, out_path = sys.argv[1], sys.argv[2]
tasks = sys.argv[3:]
lim = {}
for line in open(limits_path):
    m = re.match(r'^\s*([A-Za-z0-9_]+)\s*:\s*(\d+)', line)
    if m:
        lim[m.group(1)] = int(m.group(2))
# unknown step_lim -> sort as if expensive, so it is never the straggler picked up last
tasks.sort(key=lambda t: -lim.get(t, 10**6))
open(out_path, 'w').write("\n".join(tasks) + "\n")
PY
echo "  queue (longest first): $(head -c 200 "$QUEUE" | tr '\n' ' ')..."
: > "$LOCK"

worker_pids=()
for gpu in $(seq 0 $((IWM_NUM_GPUS-1))); do
  port=$(iwm_ws_port "$gpu")
  (
    while :; do
      # Atomically pop the head of the queue. The fd redirect MUST be inside the command
      # substitution: `t=$(flock 9; ...) 9<>"$LOCK"` attaches fd 9 to the assignment, not
      # to the subshell that runs flock, so the lock silently does nothing and two workers
      # pop the same task. (Observed: one task dispatched twice.)
      t=$( { flock 9; head -n1 "$QUEUE"; sed -i '1d' "$QUEUE"; } 9<>"$LOCK" )
      [ -z "$t" ] && break
      run_one "$t" "$gpu" "$port"
    done
  ) &
  worker_pids+=($!)
done

echo
echo "waiting for ${#worker_pids[@]} workers..."
for p in "${worker_pids[@]}"; do wait "$p"; done

echo
echo "all clients finished. results: $SAVE_ROOT"
if [ -s "$LOG_DIR/_failed.txt" ]; then
  echo "NON-ZERO EXITS (results incomplete -- do NOT aggregate):" >&2
  cat "$LOG_DIR/_failed.txt" >&2
fi
