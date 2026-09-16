#!/bin/bash
# TE 全部离线测试（不需要 GPU）。改动 TE 相关代码后请重跑。
set -e
source /usr/local/miniconda3/etc/profile.d/conda.sh; conda activate agentgym-rl
cd /data1/repos/Agentgym-RL
for t in test_te_off_is_inert test_te_turn_ids test_te_action_turns test_te_core test_te_token_mix test_te_lambda_reaches_loss test_te_centering test_te_fullvocab; do
  echo "--- $t ---"; python3 tests/$t.py
done
echo; echo "=== TE 离线测试全部通过 ==="
