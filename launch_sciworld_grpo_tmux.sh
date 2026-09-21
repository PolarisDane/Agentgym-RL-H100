#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PORT="${BASE_PORT:-36101}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
MODEL_PATH="${MODEL_PATH:-/data1/models/Qwen2.5-7B-Instruct}"
WANDB_MODE="${WANDB_MODE:-offline}"
PROJECT_NAME="${PROJECT_NAME:-agentgym-sciworld}"
# SciWorld boots a JVM per service; allow HEALTH_RETRIES*2s for the cluster.
HEALTH_RETRIES="${HEALTH_RETRIES:-300}"

# Calculate number of envs based on GPUs
IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_ARRAY[@]}"
# 2026-09-06: 每张 GPU 起多个 env server。原先是 1 卡 1 个 server，而并发轨迹是
# TRAIN_BATCH_SIZE*ROLLOUT_N=128，_select_env_addr 给每 rank 切 8/8=1 个地址，
# 于是 16 条轨迹挤在同一个进程里。sciworld 的 env.step 是同步 FastAPI + 纯 Python，
# 被 GIL 串行化（client.py 的注释明确写了这一点），成为吞吐瓶颈——8 卡相对 4 卡
# 只快了 26%（6.5 vs 8.8 分/步）而非接近一倍。
# 注意：这里只影响速度，不影响正确性。sciworld 的 server 本就设计成一进程管多个
# env（/create 返回 env_idx、字典隔离），没有 appworld 那种进程级全局状态问题。
# 默认值 1 保持与历史 run 完全一致；要提速就把 ENVS_PER_GPU 设成 4。
ENVS_PER_GPU="${ENVS_PER_GPU:-1}"
NUM_ENVS=$((NUM_GPUS * ENVS_PER_GPU))

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-sciworld_grpo_qwen2.5_3b_add_${RUN_TS}}"

RUN_TAG="${RUN_TAG:-default}"
ENV_SESSION="sciworld_env_${RUN_TAG}"
TRAIN_SESSION="sciworld_train_${RUN_TAG}"
TRAIN_LOG="${ROOT}/runlogs/${EXP_NAME}/train.log"

mkdir -p "${ROOT}/runlogs/${EXP_NAME}"

if tmux has-session -t "${ENV_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${ENV_SESSION}"
fi
if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${TRAIN_SESSION}"
fi

echo "Starting ${NUM_ENVS} SciWorld Environment Services starting at port ${BASE_PORT}..."
tmux new-session -d -s "${ENV_SESSION}" \
  "cd ${ROOT} && NUM_ENVS=${NUM_ENVS} BASE_PORT=${BASE_PORT} bash ${ROOT}/scripts/run_sciworld_env_service.sh"

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
    echo "SciWorld service on port ${PORT} failed to start."
    exit 1
  fi
done

echo "Starting GRPO Training..."
# Forward tuning env vars into the training tmux command, but only those that
# are actually set in this shell — unset ones fall through to the defaults in
# run_sciworld_grpo_train.sh. (Explicit in the command string to avoid tmux
# server env-inheritance staleness.)
FWD=""
for v in VLLM_PORT ENABLE_ERC ERC_CLIPPING_METHOD ERC_CLIPPING_TYPE ERC_MOMENTUM UNCERTAINTY_SCALE_KAPPA UNCERTAINTY_SCALE_MIN UNCERTAINTY_SCALE_RENORMALIZE SAFE_COMMIT_MODE SAFE_COMMIT_OMEGA SAFE_COMMIT_RECENCY SAFE_COMMIT_RECENCY_GAMMA SAFE_COMMIT_SUCCESS_THRESHOLD SAFE_COMMIT_KAPPA SAFE_COMMIT_GMAX SAFE_COMMIT_GMIN SAFE_COMMIT_RENORMALIZE SAFE_COMMIT_TEXT_GATE SAFE_COMMIT_GATE_MODE SAFE_COMMIT_CLF_ENV SAFE_COMMIT_CLF_WINS_ONLY SAFE_COMMIT_CLF_MAX_NEW \
         WMLOSS_ADD_COEF WMLOSS_ADD_COEF_END WMLOSS_ADD_HORIZON \
         WMLOSS_ADD_USE_GAP WMLOSS_ADD_USE_ENTROPY WMLOSS_ADD_ONLY_FAILED WMLOSS_ADD_TO_REWARD REF_NLL_ADD REF_NLL_COEF \
         EPISTEMIC_BASE EPISTEMIC_BASE_NEG EPISTEMIC_S_MAX EPISTEMIC_SHAPE EPISTEMIC_USE_REF EPISTEMIC_PRE_ADD_COEF EPISTEMIC_INVERT_ON_NEG EPI_INTRINSIC_COEF EPI_INTRINSIC_CAP EPI_INTRINSIC_USE_REF \
         \
         USE_HINDSIGHT_HCA HCA_RATIO_CLIP_MIN HCA_RATIO_CLIP_MAX HCA_TEMP HCA_OMEGA HCA_GAMMA HCA_SMOOTH_ALPHA HCA_Z_THRESHOLD HCA_FINAL_STATE_MAX_TOKENS HCA_PERSTEP HCA_HISTORY_LEN PE_CREDIT_ENABLE PE_OMEGA_PROGRESS PE_OMEGA_EXPLORE PE_CREDIT_WINS_ONLY PE_CLF_ENV \
         TE_ENABLE TE_LAMBDA TE_ETA TE_MIX TE_CENTER TE_KL_TYPE TE_WARMUP_STEPS TE_TRAJ_SUBSAMPLE TE_MICRO_BATCH_SIZE_PER_GPU PLAN_FORECAST_ENABLE PLAN_FORECAST_COEF PLAN_FORECAST_K PLAN_FORECAST_GATE PLAN_FORECAST_GROUP_NORM PLAN_FORECAST_GROUP_NORM_WINS_ONLY PLAN_FORECAST_GROUP_DEDUP PLAN_FORECAST_SKIP_INVALID PLAN_FORECAST_GROUP_GATE RESUME_MODE PLAN_FORECAST_SUCCESS_THRESHOLD PLAN_FORECAST_MAX_LENGTH PLAN_FORECAST_TARGET PLAN_FORECAST_SEQ PLAN_FORECAST_COEF_ANNEAL PLAN_FORECAST_COEF_END PLAN_FORECAST_COEF_HORIZON PLAN_FORECAST_COEF_POWER PLAN_FORECAST_COEF_CUTOFF_STEP PLAN_INLINE_ENABLE PLAN_INLINE_K PLAN_INLINE_STYLE PLAN_INLINE_PER_TURN PLAN_INLINE_WARMUP_STEPS THINK_REMINDER_ENABLE PLAN_FORMAT_REWARD_ENABLE PLAN_FORMAT_REWARD_COEF PLAN_FORMAT_REWARD_BASELINE PLAN_FORMAT_REWARD_CLIP PLAN_FORMAT_REWARD_PENALTY_ONLY PLAN_FORMAT_REWARD_WARMUP_STEPS \
         WMC_COEFF POLICY_LR ENTROPY_COEF KL_COEF \
         TOTAL_TRAINING_STEPS TOTAL_EPOCHS SAVE_FREQ ROLLOUT_GPU_MEMORY_UTILIZATION TRAIN_BATCH_SIZE ROLLOUT_N; do
  if [ -n "${!v:-}" ]; then FWD="${FWD} ${v}=${!v}"; fi
done
echo "Forwarding overrides:${FWD:-<none>}"
tmux new-session -d -s "${TRAIN_SESSION}" \
  "cd ${ROOT} && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} BASE_PORT=${BASE_PORT} ENVS_PER_GPU=${ENVS_PER_GPU} MODEL_PATH=${MODEL_PATH} WANDB_MODE=${WANDB_MODE} PROJECT_NAME=${PROJECT_NAME} EXP_NAME=${EXP_NAME} LOG_PATH=${TRAIN_LOG}${FWD} bash ${ROOT}/scripts/run_sciworld_grpo_train.sh"

echo "--------------------------------------------------"
echo "SciWorld Training Cluster Launched!"
echo "Envs per GPU:        ${ENVS_PER_GPU}"
echo "Number of Envs:      ${NUM_ENVS}"
echo "Base Port:           ${BASE_PORT}"
echo "Environment Session: ${ENV_SESSION}"
echo "Training Session:    ${TRAIN_SESSION}"
echo "Training Log:        ${TRAIN_LOG}"
echo "--------------------------------------------------"
