#!/usr/bin/env bash
# AppWorld checkpoint 批量评测：3 个 checkpoint x 2 个测试集。
# 每个 split 需要重启 env server 集群（APPWORLD_SPLIT 决定任务来源）。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="${EXP:-appworld_grpo_qwen2.5_14b_20260901_045631}"
STEPS="${STEPS:-25 50 75}"
SPLITS="${SPLITS:-test_normal test_challenge}"
BASE_PORT="${BASE_PORT:-36301}"
NUM_ENVS="${NUM_ENVS:-128}"        # 必须 >= CONCURRENCY（每条轨迹独占 1 个进程）
CONCURRENCY="${CONCURRENCY:-128}"
TP="${TP:-8}"
GPU_UTIL="${GPU_UTIL:-0.85}"
TEMP="${TEMP:-0.4}"                # 对齐 G2PO 的 validation 温度
MAX_ROUNDS="${MAX_ROUNDS:-30}"
LIMIT="${LIMIT:-0}"                # >0 时只跑前 N 题，用于冒烟
OUT_ROOT="${OUT_ROOT:-${ROOT}/runs/appworld_eval}"
SUMMARY="${OUT_ROOT}/summary.tsv"

if [ "${NUM_ENVS}" -lt "${CONCURRENCY}" ]; then
  echo "ERROR: NUM_ENVS=${NUM_ENVS} < CONCURRENCY=${CONCURRENCY}" >&2
  echo "       AppWorld 的 supervisor 活跃任务是进程级全局状态，每进程只能跑 1 条轨迹。" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"
[ -f "${SUMMARY}" ] || printf 'step\tsplit\tn\tsuccess\trate\tmean_rounds\tenv_err\tmin\n' > "${SUMMARY}"

ADDRS=""
for i in $(seq 0 $((NUM_ENVS - 1))); do
  a="http://127.0.0.1:$((BASE_PORT + i))"
  ADDRS="${ADDRS:+${ADDRS},}${a}"
done

start_envs() {   # $1 = split
  echo "--- 启动 ${NUM_ENVS} 个 env server (split=$1) ---"
  tmux kill-session -t appworld_eval_env 2>/dev/null
  # 切 split 时旧 server 刚被杀，端口处于 TIME_WAIT，新进程会绑定失败导致整个
  # split 被跳过（实测 2026-09-03: 端口 36301 未就绪 -> 跳过 test_challenge）。
  # 等旧进程退净 + 端口释放后再起。
  pkill -f "appworld-env --host" 2>/dev/null
  for _ in $(seq 1 60); do
    pgrep -f "appworld-env --host" >/dev/null || break
    sleep 2
  done
  pkill -9 -f "appworld-env --host" 2>/dev/null
  sleep 15
  tmux new-session -d -s appworld_eval_env \
    "cd ${ROOT} && NUM_ENVS=${NUM_ENVS} BASE_PORT=${BASE_PORT} APPWORLD_SPLIT=$1 \
     bash ${ROOT}/scripts/run_appworld_env_service.sh"
  for i in $(seq 0 $((NUM_ENVS - 1))); do
    p=$((BASE_PORT + i))
    for _ in $(seq 1 300); do
      curl --noproxy '*' -sf "http://127.0.0.1:${p}/" >/dev/null && break
      sleep 2
    done
    curl --noproxy '*' -sf "http://127.0.0.1:${p}/" >/dev/null || { echo "端口 ${p} 未就绪" >&2; return 1; }
  done
  echo "--- ${NUM_ENVS} 个 env server 就绪 ---"
}

stop_envs() {
  tmux kill-session -t appworld_eval_env 2>/dev/null
  sleep 3
  pkill -f "appworld-env --host" 2>/dev/null
  sleep 3
}

for split in ${SPLITS}; do
  start_envs "${split}" || { echo "env 启动失败, 跳过 ${split}" >&2; continue; }
  for step in ${STEPS}; do
    MP="${ROOT}/checkpoints/${EXP}/global_step_${step}/actor/huggingface"
    OD="${OUT_ROOT}/step${step}_${split}"
    echo ""
    echo "############ EVAL step_${step} / ${split}  $(date '+%F %T') ############"
    if [ ! -f "${MP}/config.json" ]; then echo "缺 ${MP}" >&2; continue; fi
    ( source /usr/local/miniconda3/etc/profile.d/conda.sh && conda activate agentgym-rl && \
      cd "${ROOT}" && \
      python3 scripts/eval_appworld.py \
        --model-path "${MP}" --split "${split}" --output-dir "${OD}" \
        --env-addrs "${ADDRS}" --tp "${TP}" --gpu-util "${GPU_UTIL}" \
        --concurrency "${CONCURRENCY}" --max-rounds "${MAX_ROUNDS}" \
        --temp "${TEMP}" --limit "${LIMIT}" --overwrite )
    rc=$?
    python3 - "$OD" "$step" "$split" "$SUMMARY" <<'PY'
import json,sys,os
od,step,split,summ = sys.argv[1:5]
f=os.path.join(od,"summary.json")
if os.path.exists(f):
    d=json.load(open(f))
    row=f"{step}\t{split}\t{d['num_tasks']}\t{d['num_success']}\t{d['success_rate']:.4f}\t{d['mean_rounds']:.1f}\t{d['env_errors']}\t{d['elapsed_min']:.1f}\n"
else:
    row=f"{step}\t{split}\tNO_SUMMARY\n"
open(summ,"a").write(row); print("  => "+row.strip())
PY
    echo "############ DONE rc=${rc} ############"
  done
done
stop_envs
echo ""
echo "=================== 全部完成 ==================="
column -t "${SUMMARY}"
