#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/usr/local/miniconda3/etc/profile.d/conda.sh}"
APPWORLD_ENV="${APPWORLD_ENV:-agentenv-appworld}"
APPWORLD_SPLIT="${APPWORLD_SPLIT:-train}"
export APPWORLD_ROOT="${APPWORLD_ROOT:-/data1/datasets/appworld}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-36301}"
NUM_ENVS="${NUM_ENVS:-1}"
LOG_DIR="${LOG_DIR:-${ROOT}/runlogs/env_cluster/appworld_${APPWORLD_SPLIT}_$(date +%Y%m%d_%H%M%S)}"
PID_DIR="${LOG_DIR}/pids"

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

mkdir -p "${PID_DIR}"

source "${CONDA_SH}"
set +u
conda activate "${APPWORLD_ENV}"
set -u

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

cd "${ROOT}"

# Handle cleanup
cleanup() {
  echo "Stopping all AppWorld environment services..."
  for pid_file in "${PID_DIR}"/*.pid; do
    if [[ -f "$pid_file" ]]; then
      pid=$(cat "$pid_file")
      kill "$pid" 2>/dev/null || true
      rm "$pid_file"
    fi
  done
  exit 0
}

trap cleanup SIGINT SIGTERM EXIT

for i in $(seq 0 $((NUM_ENVS - 1))); do
  PORT=$((BASE_PORT + i))
  LOG_PATH="${LOG_DIR}/env_${PORT}.log"

  echo "Starting AppWorld[${APPWORLD_SPLIT}] service on port ${PORT}, logging to ${LOG_PATH}..."

  env \
    -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    NO_PROXY="${NO_PROXY}" \
    no_proxy="${no_proxy}" \
      APPWORLD_SPLIT="${APPWORLD_SPLIT}" \
      APPWORLD_ROOT="${APPWORLD_ROOT}" \
      appworld-env --host "${HOST}" --port "${PORT}" > "${LOG_PATH}" 2>&1 &

  echo $! > "${PID_DIR}/env_${PORT}.pid"
done

echo "All ${NUM_ENVS} environment services started. Use Ctrl+C to stop them."
wait
