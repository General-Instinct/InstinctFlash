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

RUN=${1:?usage: run_paired.sh <run_name> <test_num> <t_port> <s_port> <task...>}
N=${2:?}; TPORT=${3:?}; SPORT=${4:?}; shift 4
TASKS=("$@"); [ ${#TASKS[@]} -gt 0 ] || { echo "no tasks" >&2; exit 2; }

for p in "$TPORT" "$SPORT"; do
  iwm_port_busy "$p" || { echo "REFUSING: nothing listening on $p" >&2; exit 2; }
done
for t in "${TASKS[@]}"; do
  [ -f "$ROBOTWIN_ROOT/envs/$t.py" ] || { echo "REFUSING: no such task: $t" >&2; exit 2; }
done

arm() {                            # arm <label> <port> <gpu>
  local label=$1 port=$2 gpu=$3
  local root="$IWM_RESULT_DIR/${RUN}_${label}"
  local logd="$IWM_LOG_DIR/${RUN}_${label}"
  mkdir -p "$root" "$logd"
  for t in "${TASKS[@]}"; do
    ( cd "$ROBOTWIN_ROOT" && ROBOTWIN_ROOT="$ROBOTWIN_ROOT" PYTHONWARNINGS=ignore::UserWarning \
      CUDA_VISIBLE_DEVICES=$gpu "$IWM_CLIENT_PY" -m evaluation.robotwin.eval_polict_client_openpi \
        --config policy/ACT/deploy_policy.yml --overrides \
        --task_name "$t" --task_config demo_clean \
        --train_config_name 0 --model_name 0 --ckpt_setting "${RUN}_${label}" --seed 0 \
        --policy_name LingBotVA --save_root "$root" \
        --video_guidance_scale 5 --action_guidance_scale 1 \
        --test_num "$N" --port "$port" ) > "$logd/$t.log" 2>&1
    echo "$(date -u +%H:%M:%S) ${label}/${t} rc=$?"
  done
}

echo "paired run '$RUN': ${#TASKS[@]} task(s) x $N episodes, identical seeds on both arms"
arm teacher "$TPORT" 0 &
arm student "$SPORT" 1 &
wait
echo "teacher -> $IWM_RESULT_DIR/${RUN}_teacher"
echo "student -> $IWM_RESULT_DIR/${RUN}_student"
