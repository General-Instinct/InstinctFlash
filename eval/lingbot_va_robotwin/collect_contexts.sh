#!/usr/bin/env bash
# Collect PDD conditioning contexts across all 8 GPUs. CLIENT-SIDE, no policy server needed.
#
#   ./collect_contexts.sh <out_dir> <episodes_per_task> [task ...]
#
# A chunk-0 video training context is exactly (observation, prompt) -- measured, see
# probe_chunk0_cache.py -- and both come from a sim reset with no policy in the loop. So this needs no
# server, no GPU inference, and no rollout: reset, read the cameras, write. That is why it fans out
# trivially where run_eval.sh has to pin one server per GPU.
#
# WHY THE ENV BLOCK BELOW IS NOT OPTIONAL. Each worker gets its own CUDA_VISIBLE_DEVICES because
# sapien/Vulkan initialisation hangs indefinitely with all 8 GPUs visible -- silently, with no output.
# And PYTHONPATH must carry ROBOTWIN_ROOT even though cwd is ROBOTWIN_ROOT, because invoking a script
# by absolute path puts the SCRIPT's directory on sys.path[0] rather than the cwd. See
# dump_reset_context.py's docstring; the two failures masquerade as each other.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" && source ./env.sh

OUT=${1:?usage: collect_contexts.sh <out_dir> <episodes_per_task> [task ...]}
N=${2:?}
shift 2
TASKS=("$@")

if [ ${#TASKS[@]} -eq 0 ]; then
  # Default to the full 50-task suite the certification runs use, so the context pool covers the
  # same distribution the student will be certified on.
  mapfile -t TASKS < <(ls "$ROBOTWIN_ROOT/envs"/*.py 2>/dev/null \
    | xargs -n1 basename | sed 's/\.py$//' | grep -v '^_' | grep -v '^base' | sort)
fi
[ ${#TASKS[@]} -gt 0 ] || { echo "no tasks found" >&2; exit 2; }

for t in "${TASKS[@]}"; do
  [ -f "$ROBOTWIN_ROOT/envs/$t.py" ] || { echo "REFUSING: no such task: $t" >&2; exit 2; }
done

mkdir -p "$OUT"
LOGD="${IFL_LOG_DIR}/collect_$(basename "$OUT")"
mkdir -p "$LOGD"
QUEUE="$LOGD/_queue"
printf '%s\n' "${TASKS[@]}" > "$QUEUE"
: > "$LOGD/_lock"

echo "collecting $N contexts/task for ${#TASKS[@]} tasks -> $OUT"
echo "logs: $LOGD"

# One worker per GPU, pulling tasks from a flock'd queue so a slow task does not idle a device.
worker() {
  local gpu=$1
  while :; do
    local task
    task=$(flock "$LOGD/_lock" -c "head -1 '$QUEUE'; sed -i '1d' '$QUEUE'")
    [ -n "$task" ] || break
    # Skip tasks already fully collected, so the script is resumable.
    local have
    have=$(ls "$OUT" 2>/dev/null | grep -c "^${task}__ep")
    if [ "$have" -ge "$N" ]; then
      echo "  [gpu$gpu] $task: already have $have, skipping"
      continue
    fi
    ( cd "$ROBOTWIN_ROOT" && env ROBOTWIN_ROOT="$ROBOTWIN_ROOT" PYTHONPATH="$ROBOTWIN_ROOT" \
        PYTHONWARNINGS=ignore::UserWarning CUDA_VISIBLE_DEVICES="$gpu" \
        "$IFL_CLIENT_PY" -u "$IFL_ROOT/eval/lingbot_va_robotwin/dump_reset_context.py" \
        --tasks "$task" --episodes "$N" --seed 0 --out "$OUT" ) > "$LOGD/$task.log" 2>&1
    local rc=$?
    local got
    got=$(ls "$OUT" 2>/dev/null | grep -c "^${task}__ep")
    if [ $rc -ne 0 ] || [ "$got" -lt "$N" ]; then
      echo "  [gpu$gpu] $task: FAILED rc=$rc, got $got/$N  (see $LOGD/$task.log)"
    else
      echo "  [gpu$gpu] $task: $got"
    fi
  done
}

for g in 0 1 2 3 4 5 6 7; do worker "$g" & done
wait

TOTAL=$(ls "$OUT" 2>/dev/null | grep -c '\.npz$')
WANT=$(( N * ${#TASKS[@]} ))
echo
echo "collected $TOTAL / $WANT contexts in $OUT"
# Report rather than silently accept a short pool: a context set that quietly covers 40 of 50 tasks
# would bias training toward whatever survived, and nothing downstream would say so.
if [ "$TOTAL" -lt "$WANT" ]; then
  echo "INCOMPLETE -- missing tasks:"
  for t in "${TASKS[@]}"; do
    c=$(ls "$OUT" 2>/dev/null | grep -c "^${t}__ep")
    [ "$c" -lt "$N" ] && echo "    $t: $c/$N"
  done
  exit 1
fi
echo "COMPLETE"
