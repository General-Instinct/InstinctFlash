#!/usr/bin/env bash
# Paired evaluation for certification: the SAME tasks and SAME seeds against two arms.
#
#   ./run_paired.sh <run_name> <test_num> <teacher_port> <student_port> <task...>
#
# `run_eval.sh` fans tasks across all 8 GPUs and refuses unless every server is up. Certification
# needs the opposite shape: two arms, identical episodes, so the outcomes can be paired
# episode-for-episode. Independent runs of the same size would let ordinary between-run variance
# look like a real difference, which is the whole reason McNemar is paired.
#
# --seed 0 is passed to both arms, exactly as run_eval.sh does, so the RoboTwin episode seeds are
# the official st_seed = 10000*(1+seed) sequence and are IDENTICAL across arms.
set -u
cd "$(dirname "$0")" && source ./env.sh

RUN=${1:?usage: run_paired.sh <run_name> <test_num> <teacher_gpu:port,...> <student_gpu:port,...> <task...>}
N=${2:?}; TPAIRS=${3:?}; SPAIRS=${4:?}; shift 4
TASKS=("$@"); [ ${#TASKS[@]} -gt 0 ] || { echo "no tasks" >&2; exit 2; }

for gp in ${TPAIRS//,/ } ${SPAIRS//,/ }; do
  p=${gp##*:}
  iwm_port_busy "$p" || { echo "REFUSING: nothing listening on $p" >&2; exit 2; }
done
for t in "${TASKS[@]}"; do
  [ -f "$ROBOTWIN_ROOT/envs/$t.py" ] || { echo "REFUSING: no such task: $t" >&2; exit 2; }
done

# One worker per (arm, server). Tasks are pulled from a shared queue with flock, so a slow task
# does not idle a GPU -- and TASK-level parallelism preserves pairing, because each task's episode
# seeds are the deterministic st_seed sequence from --seed 0 and do not depend on which GPU ran it.
arm() {                            # arm <label> <gpu:port,gpu:port,...>
  local label=$1 pairs=$2
  local root="$IWM_RESULT_DIR/${RUN}_${label}"
  local logd="$IWM_LOG_DIR/${RUN}_${label}"
  local queue="$logd/_queue"
  mkdir -p "$root" "$logd"
  printf '%s\n' "${TASKS[@]}" > "$queue"
  : > "$logd/_lock"
  local wpids=()
  for gp in ${pairs//,/ }; do
    local gpu=${gp%%:*} port=${gp##*:}
    (
      while :; do
        t=$( { flock 9; head -n1 "$queue"; sed -i 1d "$queue"; } 9<>"$logd/_lock" )
        [ -n "$t" ] || break
        run_task "$label" "$t" "$port" "$gpu" "$root" "$logd"
      done
    ) &
    wpids+=($!)
  done
  wait "${wpids[@]}"
}

run_task() {                       # run_task <label> <task> <port> <gpu> <root> <logd>
  local label=$1 t=$2 port=$3 gpu=$4 root=$5 logd=$6
  for _once in 1; do
    ( cd "$ROBOTWIN_ROOT" && ROBOTWIN_ROOT="$ROBOTWIN_ROOT" PYTHONWARNINGS=ignore::UserWarning \
      CUDA_VISIBLE_DEVICES=$gpu "$IWM_CLIENT_PY" -m evaluation.robotwin.eval_polict_client_openpi \
        --config policy/ACT/deploy_policy.yml --overrides \
        --task_name "$t" --task_config demo_clean \
        --train_config_name 0 --model_name 0 --ckpt_setting "${RUN}_${label}" --seed 0 \
        --policy_name LingBotVA --save_root "$root" \
        --video_guidance_scale 5 --action_guidance_scale 1 \
        --test_num "$N" --port "$port" ) > "$logd/$t.log" 2>&1
    echo "$(date -u +%H:%M:%S) ${label}/${t} rc=$?" | tee -a "$logd/_progress.txt"
  done
}

echo "paired run '$RUN': ${#TASKS[@]} task(s) x $N episodes, identical seeds on both arms"
arm teacher "$TPAIRS" &
arm student "$SPAIRS" &
wait

# Native per-episode emission: every paired run produces episodes.jsonl, so certification never
# depends on someone remembering to scrape afterwards.
for label in teacher student; do
  root="$IWM_RESULT_DIR/${RUN}_${label}"
  "$IWM_SERVER_PY" "$(dirname "$0")/emit_episodes.py" "$root" -o "$root/episodes.jsonl" \
    || echo "WARNING: ${label} episode emission refused; the run is not certifiable" >&2
done
echo "teacher -> $IWM_RESULT_DIR/${RUN}_teacher"
echo "student -> $IWM_RESULT_DIR/${RUN}_student"
