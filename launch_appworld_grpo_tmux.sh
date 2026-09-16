#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PORT="${BASE_PORT:-36301}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MODEL_PATH="${MODEL_PATH:-/data1/models/Qwen2.5-14B-Instruct}"
WANDB_MODE="${WANDB_MODE:-offline}"
PROJECT_NAME="${PROJECT_NAME:-agentgym-appworld}"
# AppWorld 每个 service 要加载 app 数据库；给集群留足 HEALTH_RETRIES*2s。
HEALTH_RETRIES="${HEALTH_RETRIES:-300}"

# Calculate number of envs based on GPUs
IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_ARRAY[@]}"
# 每张 GPU 起多个 env server：AppWorld 的 env.step 是同步 FastAPI + 纯 Python，
# 单进程被 GIL 串行化。必须与 run_appworld_grpo_train.sh 的 ENVS_PER_GPU 保持一致，
# 否则训练侧构造的地址列表和实际起的服务数对不上。
# 2026-08-31: 4 -> 16。AppWorld 的 supervisor "当前活跃任务" 是进程级全局状态，
# 同一进程内并存的多个 AppWorld 实例共用它：任一 episode 调 complete_task，同进程
# 其余 episode 会被立刻判 done 并用被污染的状态跑 evaluate()。并发轨迹数固定为
# TRAIN_BATCH_SIZE*ROLLOUT_N=128，故进程数必须也是 128 才能一对一隔离。
ENVS_PER_GPU="${ENVS_PER_GPU:-16}"
NUM_ENVS=$((NUM_GPUS * ENVS_PER_GPU))

# ---- 并发轨迹数守卫（2026-09-01 增加）----------------------------------------
# AppWorld 的 supervisor「当前活跃任务」是进程级全局状态：同一进程内并存的多个
# AppWorld 实例共用它，任一 episode 调 complete_task，同进程其余 episode 立刻被判
# done 并用被污染的状态跑 evaluate()（审计脚本 A 项可复现，这是 AppWorld 的固有
# 限制，改不掉）。因此**每个 env server 进程只能承载 1 条并发轨迹**。
# 并发轨迹数 = TRAIN_BATCH_SIZE * ROLLOUT_N，必须 <= NUM_ENVS。
_TBS="${TRAIN_BATCH_SIZE:-16}"; _RN="${ROLLOUT_N:-8}"
_CONCURRENT=$((_TBS * _RN))
if [ "${NUM_ENVS}" -lt "${_CONCURRENT}" ]; then
  echo "ERROR: NUM_ENVS=${NUM_ENVS} < 并发轨迹数 $((_TBS))x$((_RN))=${_CONCURRENT}" >&2
  echo "       每进程会挤 $(( (_CONCURRENT + NUM_ENVS - 1) / NUM_ENVS )) 条轨迹 -> 互相判 done + 假阳奖励。" >&2
  echo "       请把 ENVS_PER_GPU 提到 $(( (_CONCURRENT + NUM_GPUS - 1) / NUM_GPUS )) 或更高。" >&2
  exit 1
fi
echo "并发守卫: NUM_ENVS=${NUM_ENVS} >= 并发轨迹 ${_CONCURRENT}  (每进程 1 条) OK"
# -----------------------------------------------------------------------------

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-appworld_grpo_qwen2.5_14b_${RUN_TS}}"

RUN_TAG="${RUN_TAG:-default}"
ENV_SESSION="appworld_env_${RUN_TAG}"
TRAIN_SESSION="appworld_train_${RUN_TAG}"
TRAIN_LOG="${ROOT}/runlogs/${EXP_NAME}/train.log"

mkdir -p "${ROOT}/runlogs/${EXP_NAME}"

if tmux has-session -t "${ENV_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${ENV_SESSION}"
fi
if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${TRAIN_SESSION}"
fi

echo "Starting ${NUM_ENVS} AppWorld Environment Services starting at port ${BASE_PORT}..."
tmux new-session -d -s "${ENV_SESSION}" \
  "cd ${ROOT} && NUM_ENVS=${NUM_ENVS} BASE_PORT=${BASE_PORT} bash ${ROOT}/scripts/run_appworld_env_service.sh"

echo "Waiting for services to become healthy..."
for i in $(seq 0 $((NUM_ENVS - 1))); do
  PORT=$((BASE_PORT + i))
  ADDR="http://127.0.0.1:${PORT}"
  echo "Checking ${ADDR}..."
  for _ in $(seq 1 "${HEALTH_RETRIES}"); do
    if curl --noproxy '*' -sf "${ADDR}/" >/dev/null; then
      echo "Port ${PORT} is healthy."
      break
    fi
    sleep 2
  done
  if ! curl --noproxy '*' -sf "${ADDR}/" >/dev/null; then
    echo "AppWorld service on port ${PORT} failed to start."
    exit 1
  fi
done

echo "Starting GRPO Training..."
# Forward tuning env vars into the training tmux command, but only those that
# are actually set in this shell — unset ones fall through to the defaults in
# run_appworld_grpo_train.sh. (Explicit in the command string to avoid tmux
# server env-inheritance staleness.)
FWD=""
for v in ENABLE_ERC ERC_CLIPPING_METHOD ERC_CLIPPING_TYPE ERC_MOMENTUM UNCERTAINTY_SCALE_KAPPA UNCERTAINTY_SCALE_MIN UNCERTAINTY_SCALE_RENORMALIZE SAFE_COMMIT_MODE SAFE_COMMIT_OMEGA SAFE_COMMIT_RECENCY SAFE_COMMIT_RECENCY_GAMMA SAFE_COMMIT_SUCCESS_THRESHOLD SAFE_COMMIT_KAPPA SAFE_COMMIT_GMAX SAFE_COMMIT_GMIN SAFE_COMMIT_RENORMALIZE SAFE_COMMIT_TEXT_GATE SAFE_COMMIT_GATE_MODE SAFE_COMMIT_CLF_ENV SAFE_COMMIT_CLF_WINS_ONLY SAFE_COMMIT_CLF_MAX_NEW \
         WMLOSS_ADD_COEF WMLOSS_ADD_COEF_END WMLOSS_ADD_HORIZON \
         WMLOSS_ADD_USE_GAP WMLOSS_ADD_USE_ENTROPY WMLOSS_ADD_ONLY_FAILED WMLOSS_ADD_TO_REWARD REF_NLL_ADD REF_NLL_COEF \
         EPISTEMIC_BASE EPISTEMIC_BASE_NEG EPISTEMIC_S_MAX EPISTEMIC_SHAPE EPISTEMIC_USE_REF EPISTEMIC_PRE_ADD_COEF EPISTEMIC_INVERT_ON_NEG EPI_INTRINSIC_COEF EPI_INTRINSIC_CAP EPI_INTRINSIC_USE_REF \
         \
         USE_HINDSIGHT_HCA HCA_RATIO_CLIP_MIN HCA_RATIO_CLIP_MAX HCA_TEMP HCA_OMEGA HCA_GAMMA HCA_SMOOTH_ALPHA HCA_Z_THRESHOLD HCA_FINAL_STATE_MAX_TOKENS HCA_PERSTEP HCA_HISTORY_LEN PE_CREDIT_ENABLE PE_OMEGA_PROGRESS PE_OMEGA_EXPLORE PE_CREDIT_WINS_ONLY PE_CLF_ENV \
         PLAN_FORECAST_ENABLE PLAN_FORECAST_COEF PLAN_FORECAST_K PLAN_FORECAST_GATE PLAN_FORECAST_SUCCESS_THRESHOLD PLAN_FORECAST_MAX_LENGTH PLAN_FORECAST_TARGET PLAN_FORECAST_SEQ PLAN_FORECAST_COEF_ANNEAL PLAN_FORECAST_COEF_END PLAN_FORECAST_COEF_HORIZON PLAN_FORECAST_COEF_POWER PLAN_FORECAST_COEF_CUTOFF_STEP PLAN_INLINE_ENABLE PLAN_INLINE_K PLAN_INLINE_STYLE PLAN_INLINE_PER_TURN PLAN_INLINE_WARMUP_STEPS THINK_REMINDER_ENABLE PLAN_FORMAT_REWARD_ENABLE PLAN_FORMAT_REWARD_COEF PLAN_FORMAT_REWARD_BASELINE PLAN_FORMAT_REWARD_CLIP PLAN_FORMAT_REWARD_PENALTY_ONLY PLAN_FORMAT_REWARD_WARMUP_STEPS \
         WMC_COEFF POLICY_LR ENTROPY_COEF KL_COEF \
         TOTAL_TRAINING_STEPS TOTAL_EPOCHS SAVE_FREQ ROLLOUT_GPU_MEMORY_UTILIZATION TRAIN_BATCH_SIZE ROLLOUT_N; do
  if [ -n "${!v:-}" ]; then FWD="${FWD} ${v}=${!v}"; fi
done
echo "Forwarding overrides:${FWD:-<none>}"
tmux new-session -d -s "${TRAIN_SESSION}" \
  "cd ${ROOT} && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} BASE_PORT=${BASE_PORT} ENVS_PER_GPU=${ENVS_PER_GPU} ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0} MODEL_PATH=${MODEL_PATH} WANDB_MODE=${WANDB_MODE} PROJECT_NAME=${PROJECT_NAME} EXP_NAME=${EXP_NAME} LOG_PATH=${TRAIN_LOG}${FWD} bash ${ROOT}/scripts/run_appworld_grpo_train.sh"

echo "--------------------------------------------------"
echo "AppWorld Training Cluster Launched!"
echo "Number of GPUs:      ${NUM_GPUS}"
echo "Envs per GPU:        ${ENVS_PER_GPU}"
echo "Number of Envs:      ${NUM_ENVS}"
echo "Base Port:           ${BASE_PORT}"
echo "Environment Session: ${ENV_SESSION}"
echo "Training Session:    ${TRAIN_SESSION}"
echo "Training Log:        ${TRAIN_LOG}"
echo "--------------------------------------------------"
