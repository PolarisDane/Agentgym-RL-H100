# Qwen3-8B 适配记录（2026-09-19）

目标：在 sciworld 上用现有 GRPO 流水线（verl 0.2.0.post2 + vLLM 0.6.3）训练 Qwen3-8B。

## 启动

```bash
RUN_TAG=q3 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 BASE_PORT=36101 \
EXP_NAME=sciworld_qwen3_8b_$(date -u +%Y%m%d_%H%M%S) \
MODEL_PATH=/data1/models/Qwen3-8B TOTAL_TRAINING_STEPS=301 SAVE_FREQ=25 \
PLAN_FORECAST_ENABLE=False ENVS_PER_GPU=1 \
bash launch_sciworld_grpo_tmux.sh
```
其余超参沿用 Qwen2.5-7B 的配置即可（参数量只多约 7%，每卡多约 1.2GB）。

## 核心阻塞：vLLM 0.6.3 不认识 Qwen3

vLLM 0.6.3 早于 Qwen3 发布。verl 0.2.0 与 vLLM 0.6.3 深度耦合
（`verl/third_party/vllm/vllm_v_0_6_3/`），升级 vLLM 会牵动整个 rollout，
所以选择移植而非升级。

Qwen3 相对 Qwen2 只有两处结构差异：
1. **QK-Norm**：q/k 投影后、RoPE 之前，按 head 做 RMSNorm（`q_norm`/`k_norm`，dim=128）
2. **attention 无 bias**（`attention_bias=False`）

### ⚠️ 移植时的坑：非连续张量

`q`/`k` 是 `qkv.split(...)` 出来的**非连续视图**，而 vLLM 0.6.3 的 `ops.rms_norm`
按连续内存索引。直接喂进去**不报错**但读错元素。必须先 `.contiguous()`。
**新版 vLLM 官方 qwen3.py 的写法（3D 视图、不 contiguous）照搬回 0.6.3 就会中招。**

### 数值验证（HF transformers vs vLLM，同一段 879 token 的 sciworld 对话）

| | \|Δlogprob\| 均值 | 中位 | argmax 一致 |
|---|---|---|---|
| 正参照：Qwen2.5-7B（vLLM 官方原生） | 0.0354 | 0.0019 | 98.78% |
| **本移植 Qwen3-8B tp=1** | 0.0339 | **0.0005** | 98.86% |
| **本移植 Qwen3-8B tp=2** | 0.0358 | 0.0006 | 99.20% |
| 负对照：不加 contiguous | **6.7946** | 6.6053 | **6.15%** |

移植与 HF 的一致性和 vLLM 原生支持的 Qwen2.5 处于同一噪声水平。

## 改动清单

| 文件 | 改动 | 备份 |
|---|---|---|
| `site-packages/vllm/model_executor/models/qwen3.py` | **新增**，以 qwen2.py 为底本 | — |
| `site-packages/vllm/model_executor/models/registry.py` | 注册 `Qwen3ForCausalLM` 一行 | `.pre_qwen3` |
| `verl/third_party/vllm/vllm_v_0_6_3/dtensor_weight_loaders.py` | FSDP→vLLM 权重同步注册 Qwen3（复用 qwen2 loader） | `.pre_qwen3` |
| `verl/agent_trainer/ppo/plan_forecast.py` | `encode_sft_sample` 关 thinking | `.pre_qwen3think` |
| `verl/agent_trainer/ppo/temporal_ensemble.py` | 同上 | `.pre_qwen3think` |
| `verl/agent_trainer/ppo/world_model_loss.py` | 同上 | `.pre_qwen3think` |
| `verl/workers/agent_fsdp_workers.py` | hindsight / pe_labels 两处 | `.pre_qwen3think` |
| `verl/agent_trainer/ppo/ray_trainer.py` | `_compute_safe_commit_w`（**生成**，不关会耗光预算） | `.pre_qwen3think` |
| `scripts/eval_sciworld.py`、`scripts/eval_appworld.py` | 评测关 thinking | `.pre_qwen3think` |

rollout 主路径的 thinking 处理（`schemas.py::_thinking_kwargs`）是之前就做好的。

**对旧模型零影响**：`_thinking_kwargs` 对 Qwen2.5 返回 `{}`（已实测），所有调用对旧模型逐字节不变。

### 为什么辅助损失也要关 thinking

辅助损失的调法是 `prefix=模板(add_generation_prompt=True)`、`full=模板(prefix+target)`，
训练目标 = `full − prefix`。不关 thinking 时 Qwen3 的目标开头会多出
`<think>\n\n</think>\n\n`，等于训练模型**主动生成**空思考块；而 rollout 里这个块预置在 prompt 中、
不计 loss。关掉后 prefix 结尾与 rollout 的生成提示逐字一致。

## vLLM 移植的持久化

移植在 site-packages 里，**重装/升级 vLLM 会丢**。备份在本目录：
```bash
bash patches/vllm_0_6_3_qwen3/install.sh   # 幂等，装完自检
```

## 端到端验证

- 训练 3 步冒烟：2 个 logged step 正常（score 0.008→0.055，KL 0.002→0.003，熵 0.404→0.391）
- `global_step_3` 相对 base：99.92% 元素改变（相对变化 3.6e-5），embedding 8.21%（稀疏梯度，符合预期）
- FSDP 分片 → HF 合并正常（4 个 safetensors 分片，16G）
- 评测 8 题：212 条回复中 `<think>` 出现 **0** 次，缺 `Action:` 仅 2 条（0.9%）
- 未训练 Qwen3-8B：0/8、Score −11.25（Qwen2.5-7B base 为 2.50% / −13.34，同一量级）
- 每步耗时 328s（Qwen2.5-7B 346s）

## 已知的非 Qwen3 问题（记录，未改动）

1. **verl 第一个训练步的学习率恒为 0。** `verl/utils/torch_functional.py` 的
   `get_constant_schedule_with_warmup` 用 `min(1, step/max(1, warmup))`，warmup=0 时
   step 0 得 0（HF 同名函数此时返回 1.0）。第 1 步只累积 Adam 动量、不改权重。
   **所有历史 Qwen2.5 实验同样如此**。未修改：改了会破坏与历史实验的可比性。
2. **`TOTAL_TRAINING_STEPS=N` 实际只做 N−1 次更新**（先 log、再 +1、再判断退出）。
3. **Qwen3 训练的 `perf/mfu` 会显示 0**：`flops_counter` 只认 qwen2/llama，对未知类型
   优雅降级。不能简单复用 qwen2 估算——那里有 `assert isinstance(config, Qwen2Config)`。
4. **`scripts/te_offline.py` 不适用于 Qwen3 轨迹**：它重新套模板得到 token 序列，
   而 Qwen3 模板对历史 assistant 轮不保留空思考块，与 rollout 的实际 token 序列不一致。
5. **`test_te_off_is_inert.py` 在本次改动前就已经失败**（gate/group_norm 解耦改动
   未同步进 `.pre_te` 基线）。已将非 TE 改动同步进基线并通过；变异测试确认它仍能抓到
   真正的 TE 泄漏。原基线保留为 `.pre_te.orig`。

---

# WebShop 脚本本机化 + Qwen3-8B plan_forecast 实验（2026-09-19）

webshop 的两个脚本最后修改于 2026-08-24，此后 sciworld 侧的若干修复没有同步过来，
且默认值仍指向另一台集群。**设置的变量会被静默忽略**，不报错。

## 发现的问题

| 问题 | 后果 |
|---|---|
| 启动脚本转发列表缺 `SAVE_FREQ` / `TOTAL_TRAINING_STEPS` / `PLAN_FORECAST_GROUP_NORM` / `SKIP_INVALID` / `GROUP_DEDUP` 等 | tmux 不可靠继承环境变量，训练进程拿到默认值 |
| 训练脚本没有 `total_training_steps`，按 `TOTAL_EPOCHS=5` 结束 | 3930 条 / 16 = 245 步/epoch，停不在指定步数 |
| `ROLLOUT_GPU_MEMORY_UTILIZATION` 默认 0.80（为 3B 模型设定） | 8B 无 offload 时顶满 80GB |
| `CONDA_SH`=/opt/conda、`TRAIN_ENV`/`WEBSHOP_ENV`=/inspire/…、`TRAIN_FILE` 仓库内不存在 | 本机无法启动 |
| `PLAN_FORECAST_COEF_ANNEAL` 默认 `linear`、HORIZON=25、END=0 | **PF 系数 25 步内衰减到 0**，之后等于纯 GRPO |
| `PLAN_FORECAST_GROUP_DEDUP` 默认 True（sciworld 为 False） | 与 sciworld 实验口径不一致 |

## 改动（均为"设置了才生效"，另一台集群行为不变）

- `scripts/run_webshop_grpo_train.sh`（备份 `.pre_qwen3pf`）：新增可选
  `TOTAL_TRAINING_STEPS`，为空时不传给 trainer（仍按 epoch）
- `launch_webshop_grpo_tmux.sh`（备份 `.pre_qwen3pf`）：
  - 训练侧转发列表补 `PLAN_FORECAST_GROUP_NORM/SKIP_INVALID/GROUP_DEDUP/GROUP_GATE`、
    `TOTAL_TRAINING_STEPS`、`SAVE_FREQ`、`ROLLOUT_GPU_MEMORY_UTILIZATION`、
    `TRAIN_FILE`、`CONDA_SH`、`TRAIN_ENV`
  - env 服务侧新增 `CONDA_SH` / `WEBSHOP_ENV` 转发
- **未改任何默认值**：它们指向另一台集群，那边可能仍在用。

## 关于"只有 1000 个商品"

本地只有 `items_shuffle_1000.json`。这**不是缺数据**：AgentGym 的 webshop env server
始终以 `num_products=1000` 运行，1K 文件已验证与完整文件前 1000 条逐字节等价
（见 `web_agent_site/utils.py` 顶部注释）。人工目标也只从这 1000 个商品构造。

## 实验：webshop_qwen3_8b_pfK3_gnorm_skipinv_20260919_023455

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 BASE_PORT=36101 EXP_NAME=<...> \
MODEL_PATH=/data1/models/Qwen3-8B \
CONDA_SH=/usr/local/miniconda3/etc/profile.d/conda.sh \
WEBSHOP_ENV=agentenv-webshop TRAIN_ENV=agentgym-rl \
TRAIN_FILE=/data1/datasets/AgentGym-RL-Data-ID/train/webshop_train.json \
TOTAL_TRAINING_STEPS=201 SAVE_FREQ=50 ROLLOUT_GPU_MEMORY_UTILIZATION=0.5 \
PLAN_FORECAST_ENABLE=True PLAN_FORECAST_K=3 \
PLAN_FORECAST_GROUP_NORM=True PLAN_FORECAST_SKIP_INVALID=True \
PLAN_FORECAST_GROUP_DEDUP=False PLAN_FORECAST_COEF_ANNEAL=fixed PLAN_FORECAST_COEF=0.01 \
bash launch_webshop_grpo_tmux.sh
```

- `TOTAL_TRAINING_STEPS=201`：verl 差一约定，201 → 完整 200 步，checkpoint 落在 50/100/150/200
- `COEF_ANNEAL=fixed`、`GROUP_DEDUP=False`：与 sciworld 全部 PF 实验口径一致
- 训练器实际收到的配置已从日志逐项核对

离线验证：plan_forecast 在 webshop + Qwen3 上构造的训练目标与 Qwen2.5 逐字相同，
为后续 K=3 个 `search[...]`/`click[...]` 动作，不含 `<think>`。

---

# 2026-09-20：Qwen3 生成/训练上下文不一致（webshop 两轮崩溃的主因）

## 发现

逐 token 复现真实 rollout 后确认：对 Qwen3，vLLM 生成时看到的 prompt（G）与训练算梯度的
序列（T）**每一轮都不一致**；Qwen2.5 逐 token 一致。

| | 第1轮 | 第2轮 | 第3轮 | 第4轮 | 第5轮 |
|---|---|---|---|---|---|
| Qwen2.5 G vs T | 一致 | 一致 | 一致 | 一致 | 一致 |
| Qwen3 T−G 长度差 | +21 | +25 | +29 | +33 | +37 |

两个来源：
1. `rl_dataset.py` 写死了 Qwen2.5 的默认 system prompt（+21 token）；Qwen3 模板不加
2. vLLM 每轮用 `apply_chat_template` 重新渲染，而 Qwen3 模板**删掉历史轮的 `<think></think>`**；
   训练序列逐轮拼接，每轮都带（每轮 +4 token）

影响（同一条回复在 T 下与 G 下的 log-prob 差）：

| | 未训练 Qwen3-8B | 训练 50 步后 |
|---|---|---|
| 逐 token |Δ| 均值 / p99 / 最大 | 0.24 / 5.5 / 25.7 | **1.04 / 20.5 / 57.6** |
| 每轮整段 T−G | −0.8 ~ +15 | **−8 ~ −63 nats（全为负）** |

训练严重离策略，PPO 看不到（ratio 两边都在 T 上算），最终以 ratio 爆炸
（pg_loss 9.3e8、grad_norm 1e13、NaN）崩溃。

另：decode→encode 往返有 1.2% 的回复切分会变（例 `[`,`cher`,`ry`,`]` → `[ch`,`erry`,`]`）。

## 修复（按模型分开代码路径）

判据 `token_io.uses_token_io(tokenizer)` —— 模板认 `enable_thinking` 才为 True。
**Qwen2.5 返回 False，全部走原代码，逐字节不变。**

| 文件 | Qwen3 路径（新） | 备份 |
|---|---|---|
| `verl/workers/rollout/token_io.py` | **新增**。唯一的 token 拼接规则，训练与评测共用 | — |
| `schemas.py::get_generation_prompt` | vLLM 直接用训练序列 + assistant 前缀（G≡T） | `.pre_tokenio` |
| `schemas.py::add_assistant_message` | 新增可选 `response_ids`，追加 vLLM 原始 token | 同上 |
| `vllm_rollout.py::agent_step` | 仅 Qwen3 传 `response_ids` | `.pre_tokenio` |
| `rl_dataset.py::_build_messages` | 初始 prompt 用模型自己的模板 | `.pre_tokenio` |
| `scripts/eval_webshop.py`、`eval_sciworld.py` | `run_trajectory_tokenio` → `token_io.run_episode` | `.pre_tokenio` |

注意编码细节：user 轮与训练一致，**前缀/内容/后缀分别编码再拼接**（整串编码在边界处切分可能不同）。

## 验证

1. **Qwen2.5 逐字节不变**：40 条真实轨迹（webshop+sciworld），新旧代码对比
   初始 prompt / 每轮 G / input_ids / loss_mask / observation_mask / turn_ids /
   response_ids / response_loss_mask —— **600 项全部一致**（含误传 response_ids 的情况）
2. **Qwen3 新路径**（30 条轨迹、203 轮）：训练 G==T **203/203**；评测 prompt==训练 prompt **203/203**；
   原始 token 原样追加且 loss mask 只覆盖回复+结束符 **203/203**；system prompt 残留 0/30
3. **测试套件**：`.pre_te` 基线同步非 TE 改动后审计通过；变异测试确认仍能抓到泄漏；全部通过
4. **训练冒烟**：Qwen3 webshop 3 步 0 错误；Qwen2.5 sciworld 回归 2 步 0 错误，
   第 1 步回复长度 939 与历史实验**完全相同**

附带观察：修复后 Qwen3 在 webshop 第 1 步熵 0.264（有 bug 时 0.17）—— 此前归因于
"webshop 环境本身熵低"的现象，有一部分其实来自上下文错位。

## 已知未处理

- plan_forecast / TE 的辅助损失仍用 `apply_chat_template` 构造上下文，Qwen3 下历史轮不带空思考块，
  与 rollout 序列差每轮 4 个 token。它们是 SFT 类辅助目标、不走 PPO ratio，不会造成离策略，
  但格式不完全一致。
- `eval_appworld.py` 未接入 token-io（Qwen3 暂不用于 appworld）。

---

# 2026-09-20：tmux 全局环境污染（运维陷阱）

**现象**：Qwen2.5 sciworld 回归冒烟报 `KeyError: 'task_description'`。实际是训练集被换成了
`webshop_train.json`，webshop 题号超出 sciworld 任务范围，env 服务器 reset 越界后用 **200 状态码**
返回 `{"error": ...}`，客户端不检查、直接取字段。

**根因**：所有 tmux 会话关闭后服务器退出；下一个 `tmux new-session` 由带实验变量的启动脚本发起，
**新服务器把该进程的环境整体复制为全局环境**。之后所有会话都继承它，而脚本的 `${VAR:-默认}` 会被静默覆盖。
当时残留了 `TRAIN_FILE`、`MODEL_PATH`、`TOTAL_TRAINING_STEPS`、`CUDA_VISIBLE_DEVICES=0,1,2,3`、全套 `PLAN_FORECAST_*`。

**处理**：`tmux set-environment -g -u` 清掉实验变量；建常驻会话 `tmux_anchor`（`sleep infinity`）
防止服务器退出后被脏环境重新拉起。**不要关掉 `tmux_anchor`。**

自查：`tmux show-environment -g | grep -E 'TRAIN_FILE|MODEL_PATH|PLAN_FORECAST|CUDA_VISIBLE'` 应为空。

## 修复后的训练结果（webshop_qwen3_8b_pfK3_gnorm_skipinv_tokenio_20260919_185426）

200 步全部跑完，**未崩溃**。最后一步 score 0.938 / KL 1.051 / grad 0.010。

| 阶段 | 修复后 | 第一轮(bug) | 第二轮(bug) |
|---|---|---|---|
| 首次 KL>0.5 | 第 34 步 | 第 21 步 | 第 23 步 |
| 首个梯度尖峰>1000 | **从未出现**（全程最大 4.3） | 第 27 步 | 第 28 步 |
| 首个 NaN | **从未出现** | 第 68 步 | 第 48 步 |
| score 崩溃 | **从未发生** | 第 72 步起 0.06 | 第 48 步起 0.46 |
| 训练后期 score | 0.88~0.94 稳定 | — | — |

webshop 测试集 200 题评测（全部 200/200 完成，0 失败）：

| step | Succ | Score | 平均轮数 |
|---|---|---|---|
| 50 | 56.50% | 0.7833 | 5.1 |
| 100 | 70.00% | 0.8571 | 5.3 |
| **150** | **72.50%** | **0.8736** | 5.8 |
| 200 | 71.00% | 0.8544 | 6.9 |

**与第一轮 step50（61.50%）不可直接比较**：第一轮训练在错位的上下文 T 下进行，而它的评测走的是
旧路径（Qwen3 原生模板渲染），两者本身就不一致；本轮训练/评测口径统一。
