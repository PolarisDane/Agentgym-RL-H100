#!/bin/bash
# 把 Qwen3 移植装进 vLLM 0.6.3。重装/升级 vLLM 后需重跑本脚本（移植在 site-packages 里）。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
V=$(/usr/local/miniconda3/envs/agentgym-rl/bin/python -c "import vllm,os;print(os.path.dirname(vllm.__file__))" 2>/dev/null | tail -1)/model_executor/models
cp "$HERE/qwen3.py" "$V/qwen3.py"
if grep -q '"Qwen3ForCausalLM"' "$V/registry.py"; then echo "registry 已含 Qwen3，跳过"
else cp "$V/registry.py" "$V/registry.py.pre_qwen3"; patch "$V/registry.py" < "$HERE/registry.patch"; fi
/usr/local/miniconda3/envs/agentgym-rl/bin/python -c "
from vllm.model_executor.models import ModelRegistry
assert 'Qwen3ForCausalLM' in ModelRegistry.get_supported_archs(); print('OK: vLLM 已支持 Qwen3ForCausalLM')" 2>&1 | tail -1
