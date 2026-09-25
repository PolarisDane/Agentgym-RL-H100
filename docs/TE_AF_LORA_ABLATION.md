# AF 梯度必须进入策略参数：separate-AF-LoRA 消融与 TE-KL 结果

本文档记录 2026-09-22 至 09-24 在 Blackwell 机器（8×RTX PRO 6000）上完成的一组消融，
用于回答 `docs/te_metric_section.tex` §Results 最后一条自述短板：

> **The comparison is partly definitional.** CoPE's auxiliary objective trains the model
> to predict its own next $k$ actions, close to what Eq.(gain) measures; the result
> establishes that plain RL does not improve this quantity, but is weak evidence of
> foresight beyond the form directly optimised.

结论先行：该质疑不成立，且被两个形式完全不同的指标同时否定。AF 的收益既不是
"目标形式带来的定义性假象"，也不是"有一个好的 forecaster 就够了"——**必须让 forecast
的梯度流经策略自身的参数**。

相关文档：`TE_KL_COMPUTATION.md`（指标计算细节）、`TE_METRIC_SETTING.md`（协议与已知局限）。

---

## 1. 实验设计

三条臂共享完全相同的辅助目标（plan-forecast / AF，K=3，coef 0.01，gate=wins，
group_norm on，skip_invalid on），**唯一区别是 AF 的梯度流向哪里**：

| 臂 | AF 梯度去向 | 策略的更新 | 来源 |
|---|---|---|---|
| GRPO | 无 AF | 纯 GRPO | 见 §3.3，separate-AF-LoRA 的策略在数学上即为此 |
| CoPE | 进入 backbone（共享参数） | GRPO + AF | 外部已训好的 checkpoint |
| separate-AF-LoRA | 进入独立 LoRA，backbone 拿不到 | 纯 GRPO | 本机训练，§3 |

separate-AF-LoRA 的关键性质：**step 0 时它与 CoPE 完全等价**（LoRA 的 B 矩阵零初始化），
之后唯一的分叉就是梯度路径。因此若 CoPE 的 TE 提升只是"这个目标被优化过就会出现"的
定义性产物，则本臂（同样优化了这个目标）也该出现同等提升。

---

## 2. 被测系统与数据

### 2.1 模型

| 系统 | 来源 | checkpoint |
|---|---|---|
| base | `Qwen2.5-7B-Instruct` | — |
| CoPE | HF `PolarisDane/Sciworld_Qwen2.5_7B_Instruct` | `k3_step_{50,100,150,200,250}` |
| AF-LoRA（策略） | 本机 run `sciworld_qwen2.5_7b_aflora_k3_20260922_204006` | `global_step_{50,100,150,200}/actor/huggingface` |
| AF-LoRA（适配器） | 同上 | 同上 + `af_lora.pt` |

**CoPE 的 provenance 未在本地核实**：这批 checkpoint 由另一台机器训练，本文档只知道它们
是 K=3 的 CoPE run，其余超参、seed、代码版本均未验证。因此 CoPE 与 AF-LoRA 的比较**不是
严格的单变量对照**，详见 §7。

### 2.2 固定轨迹集

`runs/sciworld_eval/FIXED_SUCCESS60`：60 条成功轨迹，1428 个目标动作轮。
provenance 见该目录 `_manifest.json`，来自历史 `te` 与 `pf` 族 run。

**本次两条新臂都没有向该集合贡献过轨迹**，因此 `TE_METRIC_SETTING.md` 里要求的
provenance 轮换（避免系统在自己产出的轨迹上被评分）在本次设置下天然满足，比原设置更干净。

全部指标一律使用 **fixed 协议**。own 协议按 `TE_METRIC_SETTING.md` §2 已证不可用，本次未跑。

---

## 3. separate-AF-LoRA 的实现

代码：`AgentGym-RL/verl/agent_trainer/ppo/af_lora.py`（新增），
`workers/agent_actor/dp_actor.py::update_plan_forecast`（分支），
`workers/agent_fsdp_workers.py`（构建 + 随 checkpoint 保存）。
开关默认关闭，关闭时与改动前逐字节一致。

### 3.1 结构

- LoRA 挂在 FSDP actor 的 `q/k/v/o/gate/up/down` 上，28 层共 196 组，r=64，alpha=128
- 参数量 1.615 亿（占 7.6B 的 2.1%），B 零初始化
- 独立 AdamW，lr 1e-4 常数，betas (0.9, 0.999)，wd 0，梯度裁剪 1.0
- 参数每 rank 各存一份，step 前对梯度做 all-reduce 取均值（DDP 口径）
- **只在 AF 的前向/反向启用**：rollout、PG 更新、ref/old-logprob、FSDP→vLLM 权重同步都不经过它

### 3.2 两个必须记录的坑

**(a) 反向也必须处于启用状态。** 梯度检查点会在反向时重算前向；若此时适配器关闭，
重算出的前向不含 LoRA 增量，梯度**静默算错**且不报任何错误。已在小模型上验证：
开/不开检查点的 LoRA 梯度逐元素一致（相对误差 0）。

**(b) Hydra 不接受未加引号的逗号分隔值。** `af_lora_targets` 必须写成
`+actor_rollout_ref.actor.af_lora_targets="'${AF_LORA_TARGETS}'"`，否则 Hydra 报
`Ambiguous value`，启动即失败（与仓库里 `env_addr` 的处理方式相同）。

### 3.3 无泄漏的验证

`update_plan_forecast` 在 AF 更新前后各取一次策略参数切片，记为
`af_lora/policy_delta = max |Δ|`。FSDP 的扁平参数无法只冻结一部分，所以 backbone 的梯度
仍会被算出来，随后被 `zero_grad(set_to_none=True)` 丢弃，只有 LoRA 的优化器 step。

**200 个训练步中 `af_lora/policy_delta` 全部为 0.000。** 因此本臂的策略更新在数学上
等价于纯 GRPO，可直接充当本机的 GRPO 参照系。

---

## 4. separate-AF-LoRA 的训练

配置：Qwen2.5-7B-Instruct，SciWorld，4 卡（GPU 0-3），200 步（`TOTAL_TRAINING_STEPS=201`，
仓库差一约定），每 50 步存一次。batch 16 prompt × rollout n=8 = 每步 128 条轨迹，
mini 8 / micro 1，ppo_epochs 1，policy lr 1e-6，kl_loss 0.001（low_var_kl），entropy 0.001，
最多 20 轮，prompt 2048 / response 4096 / max_model_len 8192 / 每轮 512 token，
vLLM 显存占比 0.5。world-model、TE、HCA、ERC、plan_format_reward、inline plan 全关。

耗时 27.2 小时。AF 的前向反向随成功率增长而变贵：每步 258 秒（前 25 步）→ 773 秒（末 25 步），
其中 AF 从约 20 秒涨到约 500 秒，AF 样本数从每步 26 个涨到 987 个。

训练集 task score（每 25 步平均）：

| 步 | 1-25 | 26-50 | 51-75 | 76-100 | 101-125 | 126-150 | 151-175 | 176-200 |
|---|---|---|---|---|---|---|---|---|
| score | 0.025 | 0.111 | 0.200 | 0.243 | 0.300 | 0.408 | 0.542 | 0.604 |
| AF loss | 0.658 | 0.321 | 0.236 | 0.146 | 0.139 | 0.110 | 0.063 | 0.068 |

测试集（200 题，仓库默认协议：最多 30 轮、每轮 200 token、T=1.0，成功 = done 且 score ≥ 100）：

| checkpoint | base | step 50 | step 100 | step 150 | step 200 |
|---|---|---|---|---|---|
| 成功率 | 4.0% | 22.5% | 31.0% | 49.0% | 60.0% |
| 平均分（原始） | −7.28 | 1.59 | 24.81 | 44.91 | 45.86 |
| 平均轮数 | 25.1 | 20.0 | 19.5 | 17.6 | 15.3 |

**评测口径的一个偏差**：评测每轮上限 200 token，训练是 512。超出上限的回合会在写出
`Action:` 之前被截断、被环境判无效——第 100 步有 40% 的回合如此，第 200 步 10%；其中
95% / 74% 的 token 数正好卡在 200。因此上表低估了这个话较多的策略。

---

## 5. 指标与协议

### 5.1 基准校准（必须先做）

本机复算 base 模型，与 `TE_KL_COMPUTATION.md` / `te_metric_section.tex` 中已发表的数字对照：

| 量 | 本机 | 已发表 | 差 |
|---|---|---|---|
| 目标轮数 | 1428 | 1,428 | 一致 |
| KL(η=0.5) | 0.1679 | 0.1673 | +0.0006 |
| KL(η=0.9) | 0.5027 | 0.5008 | +0.0019 |
| g | −1.7745 | −1.74 | −0.035 |
| log q^F | −3.1523 | −3.04 | −0.112 |
| log π⁰ | −1.3778 | −1.30 | −0.078 |

KL 与目标轮数几乎逐位复现，说明是同一套代码、同一批数据；分解项有 0.03–0.11 nats 的
偏移，来源未能在本地核实（可能是 base 快照版本或 transformers 版本差异）。

**处理**：所有 Δ 一律以**本机 base** 为基准重算，表内自洽；
**不要把本文档的绝对值与已发表的绝对值混用**。

### 5.2 三组指标

| 指标 | 形式 | 条件 | 回答的问题 |
|---|---|---|---|
| TE（策略自身） | teacher-forced 似然 | q^F 与 π⁰ 都来自策略 | 策略自己是否更能预判自己 |
| 交叉 TE | teacher-forced 似然 | q^F 来自 LoRA 适配器，π⁰ 来自策略 | 分离出的 forecaster 准不准 |
| plan hit-rate | 自由生成 + 离散精确匹配 | 无 teacher forcing | 行为层面是否真有预见性 |

TE 设置：k=3，η=0.5（同时记录 η=0.9），fixed 协议，60 条轨迹 / 1428 个目标轮。
成员只取 k≥1（源轮自身的动作不进集成，见 `temporal_ensemble.py:611`）。
置信区间：逐轨迹配对差（模型 − base，按 token 数加权）的 trajectory-level bootstrap，4000 次重采样。

plan hit-rate 设置：同一批固定轨迹，每个动作轮用训练时同一个合成 prompt
（`DEFAULT_PLAN_PROMPT`，k=3，skip_invalid=True），**greedy 自由生成**最多 64 token，
共 795 条 prompt。生成文本按 §5.3 解析后与该轨迹实际执行的 a_s…a_{s+2} 逐条精确匹配。
j=0 为当前动作（模仿），**j≥1 为预见**，与 TE 的 k≥1 口径对齐。

### 5.3 解析口径（关键，否则指标会退化成格式合规度）

未处理时，不同系统的输出格式差异会主导结果：CoPE 与适配器被训成裸列表，
纯 GRPO 训出的策略写 `Thought: …\n\nAction: x`，base 写带编号的混合格式。
按行严格比对会把策略的正确动作判为 miss（实测 hit@0 从 35% 被误判到 0%）。

因此解析器：剥离 `Thought:/Plan:/Step n:` 等散文行、编号与 bullet；
`Action:` 后取动作（含 `Action:` 独占一行的情况）；丢弃超过 10 词的散文句。
同时报告两个口径（exact / 忽略冠词）与两个分母：

- `plan_rate`：产出了完整 k 条计划的比例（格式合规度，单独报）
- `future`：所有轮次上的 j≥1 命中率（未产出即算 miss）
- `cond_future`：只在产出了完整计划的轮次上的 j≥1 命中率（剥离格式因素）

---

## 6. 结果

### 6.1 TE：策略自身（Δ 相对本机 base，95% bootstrap CI）

| 系统 | 步 | Δg | Δlog q^F | −Δlog π⁰ | KL(.5) | KL(.9) |
|---|---|---|---|---|---|---|
| CoPE | 50 | **+1.103** [+0.89, +1.32] | +1.920 [+1.68, +2.18] | −0.818 | 0.0856 | 0.2570 |
| CoPE | 100 | **+1.434** [+1.20, +1.66] | +1.752 [+1.52, +1.99] | −0.318 | 0.0683 | 0.2030 |
| CoPE | 150 | **+1.554** [+1.32, +1.77] | +1.276 [+1.04, +1.53] | +0.278 | 0.0568 | 0.1722 |
| CoPE | 200 | **+1.610** [+1.37, +1.86] | +1.151 [+0.90, +1.40] | +0.458 | 0.0596 | 0.1816 |
| CoPE | 250 | **+1.518** [+1.26, +1.79] | +1.337 [+1.08, +1.59] | +0.181 | 0.0523 | 0.1620 |
| AF-LoRA | 50 | −0.037 [−0.17, +0.07] | +0.232 [+0.16, +0.31] | −0.268 | 0.1535 | 0.4546 |
| AF-LoRA | 100 | −0.160 [−0.33, −0.03] | +0.045 [−0.04, +0.13] | −0.205 | 0.1556 | 0.4626 |
| AF-LoRA | 150 | −0.378 [−0.52, −0.24] | −0.342 [−0.47, −0.21] | −0.036 | 0.1624 | 0.4850 |
| AF-LoRA | 200 | −0.212 [−0.37, −0.06] | −0.185 [−0.34, −0.02] | −0.028 | 0.1622 | 0.4850 |

base 绝对值：g −1.7745，log q^F −3.1523，log π⁰ −1.3778，KL(.5) 0.1679，KL(.9) 0.5027。

两条线在所有可比 checkpoint 上区间不重叠，差距 1.5–2.0 nats/token。
AF-LoRA 的走势与 `te_metric_section.tex` 表中的 GRPO 行同型（由正转负），
这与 §3.3 的结论（其策略在数学上即纯 GRPO）一致，构成一次独立的自洽性检查。

**复现了已发表表格的一个细节**：CoPE 的 Δlog q^F 在 step 50 最高（本机 +1.920，
已发表 +1.985），随后单调下滑（本机 → +1.151，已发表 → +1.425），而 Δg 仍在上升。
即后期 gain 有相当部分来自 π⁰ 下降而非预见性提升——这正是把分解列为必报项的理由。

### 6.2 交叉 TE：q^F 来自 LoRA 适配器，π⁰ 仍来自策略

| 步 | Δlog q^F | 绝对 log q^F | 同 checkpoint 策略自身的 log q^F | KL(.5) |
|---|---|---|---|---|
| 50 | +2.011 [+1.76, +2.28] | −1.142 | −2.921 | 0.1485 |
| 100 | +2.022 [+1.75, +2.31] | −1.130 | −3.107 | 0.1334 |
| 150 | **+2.275** [+2.01, +2.54] | **−0.877** | −3.494 | 0.1121 |
| 200 | +2.123 [+1.88, +2.38] | −1.029 | −3.337 | 0.1102 |

对照：CoPE 的 Δlog q^F 为 +1.151 ~ +1.920，base 绝对值 −3.152。

**分离出的 forecaster 比 CoPE 更准**（step 150：−0.877 对 CoPE 最好的 −1.232），
但同一 checkpoint 的策略自身仍停在 base 水平（−3.494）。

> **交叉组的 Δg 不可解读为自洽性。** g = log q^F − log π⁰ 在交叉设定下是跨模型量，
> 含义为"适配器比策略自己更能预判该动作"，不代表策略变得自洽。该组应报 Δlog q^F。
> （交叉组 Δg 为 +1.742 / +1.818 / +2.239 / +2.096，易被误读，故此处不入正表。）

### 6.3 plan hit-rate：自由生成，无 teacher forcing

795 条 prompt / 60 条轨迹；future = j≥1（忽略冠词口径）；CI 为 trajectory-level bootstrap。

| 系统 | 步 | future | vs base | cond_future | plan_rate |
|---|---|---|---|---|---|
| base | — | 9.9% [8.1, 12.0] | — | 13.9% | 72.6% |
| CoPE | 50 | 23.1% [19.2, 27.2] | **+13.2** [+9.7, +16.7] | 23.1% | 100.0% |
| CoPE | 100 | 28.4% [24.5, 32.8] | **+18.5** [+14.6, +22.6] | 28.4% | 100.0% |
| CoPE | 150 | 26.3% [22.5, 30.2] | **+16.4** [+12.7, +20.3] | 26.4% | 99.7% |
| CoPE | 200 | 28.4% [23.9, 33.1] | **+18.5** [+14.3, +22.8] | 28.4% | 100.0% |
| CoPE | 250 | 27.2% [23.1, 31.4] | **+17.3** [+13.3, +21.3] | 27.2% | 100.0% |
| AF-LoRA 策略 | 50 | 6.8% [5.3, 8.4] | −3.1 [−4.9, −1.4] | 18.1% | 40.5% |
| AF-LoRA 策略 | 100 | 8.5% [6.5, 10.7] | −1.4 [−3.6, +0.9] | 17.3% | 50.9% |
| AF-LoRA 策略 | 150 | 9.4% [7.5, 11.6] | −0.5 [−2.8, +1.8] | 16.5% | 55.2% |
| AF-LoRA 策略 | 200 | 5.5% [4.1, 7.1] | −4.4 [−6.6, −2.4] | 13.2% | 44.3% |
| AF-LoRA 适配器 | 50 | 24.1% [20.8, 27.6] | +14.2 [+11.1, +17.4] | 24.1% | 100.0% |
| AF-LoRA 适配器 | 100 | 23.4% [20.2, 27.2] | +13.5 [+10.1, +17.1] | 23.4% | 100.0% |
| AF-LoRA 适配器 | 150 | **34.0%** [29.8, 38.1] | +24.1 [+20.2, +28.0] | 34.0% | 100.0% |
| AF-LoRA 适配器 | 200 | 32.2% [27.2, 37.5] | +22.3 [+17.3, +27.7] | 32.4% | 99.6% |

逐槽位（忽略冠词）：base j0/j1/j2 = 29.8 / 14.8 / 4.6%；CoPE-200 = 43.0 / 30.8 / 25.8%；
AF-LoRA 策略-200 = 31.6 / 8.2 / 2.4%；AF-LoRA 适配器-150 = 52.3 / 42.3 / 24.9%。
可见差距随预测步长拉大：j2（往后第三步）上 base 与 AF-LoRA 策略几乎归零，
而 CoPE 与适配器仍有 25%。

**plan_rate 本身是一项独立发现**：纯 GRPO 把策略推向 rollout 风格
（`Thought:` + 单个 `Action:`），即使被明确要求列出 k 条计划，也只有 40–55% 的轮次照做，
低于 base 的 72.6%。因此必须同时看 `cond_future`：即便只统计确实产出完整计划的轮次，
AF-LoRA 策略仍为 13–18%，与 base 的 13.9% 同档，远低于 CoPE 的 26–28%。

---

## 7. 结论

1. **CoPE 的 TE 提升不是定义性产物。** 同样的目标、数据、K 与系数，仅把梯度改道到独立
   LoRA，TE 提升即完全消失（Δg 由 +1.1~+1.6 变为 −0.04~−0.38）。若提升只源于"这个量被
   优化过"，本臂应同样受益。
2. **"有一个好的 forecaster"不足以带来收益。** 分离出的适配器在两个指标上都不输甚至超过
   CoPE（Δlog q^F +2.28 对 +1.92；hit-rate 34.0% 对 28.4%），但同一策略既不自洽
   （TE 停在 base 水平）也未获得行为层面的预见性（hit-rate 5.5–9.4%，与 base 同档）。
   因此"分离后无效是因为 forecaster 变弱"这一解释被数据排除。
3. **两个形式不同的指标同向。** TE 是 teacher-forced 的连续似然，plan hit-rate 是自由生成的
   离散精确匹配；模型无法靠"被训练在同形式的似然上"提高后者。这是对 §1 所引质疑的直接回应。
4. 综合：**AF 的收益必须经由共享参数**。策略需要"**是**"那个 forecaster，而不是"**有**"一个。

---

## 8. 局限

1. **CoPE 与 AF-LoRA 不是严格单变量对照。** CoPE 的 checkpoint 来自另一台机器，其超参、
   seed、代码版本未在本地核实；本机也没有跑过共享 backbone 的 CoPE。要把结论钉死，
   需要在同一环境补一条 shared-AF run（约 27 小时）。当前可支撑的强结论是
   §7.2（同一 run 内部的策略 vs 适配器对照，完全单变量）。
2. **各臂均为单 seed。** 仓库自身记录过同配置 run 间方差不小（coef 1e-2 两次尝试峰值
   0.42 对 0.77）。
3. **绝对值与已发表数字存在 0.03–0.11 nats 的偏移**（§5.1），来源未查明。所有结论基于
   本机自洽的相对值。
4. **测的是"能否预判他人成功轨迹的后续"，不是"是否遵循自己的计划"。** 后者需在环境中
   rollout 并对齐自身计划与自身行为，属另一实验。
5. **teacher forcing 的固有乐观性。** 成员侧打分时，被评动作之前的计划条目是**真实执行的**
   动作，模型获得了它本来没有的信息。因此 TE 测的是"计划前缀已对的前提下是否自洽"。
   自由生成版的 plan hit-rate 不受此影响，两者结论一致这一点本身提高了可信度。
6. **plan hit-rate 依赖解析器**（§5.3）。解析规则对散文式输出偏保守，可能低估这类系统；
   `cond_future` 与 `plan_rate` 用于暴露这一影响。

---

## 9. 复现

### 9.1 TE（策略自身 / 交叉）

```bash
cd /data/home/xingyangl/workspace/Agentgym-RL-H100
conda activate agentgym-rl

# 策略自身
CUDA_VISIBLE_DEVICES=0 python scripts/te_offline.py \
  --models base=/path/Qwen2.5-7B-Instruct,cope200=/path/k3_step_200 \
  --trajs dummy=runs/sciworld_eval/FIXED_SUCCESS60 \
  --fixed-traj runs/sciworld_eval/FIXED_SUCCESS60 \
  --max-traj 60 --k 3 --eta 0.5 --skip-own --out te_out.json

# 交叉：成员前向启用 LoRA 适配器（--af-lora 为本次新增）
CUDA_VISIBLE_DEVICES=0 python scripts/te_offline.py \
  --models xlora200=<run>/global_step_200/actor/huggingface \
  --af-lora xlora200=<run>/global_step_200/actor/af_lora.pt \
  --trajs dummy=runs/sciworld_eval/FIXED_SUCCESS60 \
  --fixed-traj runs/sciworld_eval/FIXED_SUCCESS60 \
  --max-traj 60 --k 3 --eta 0.5 --skip-own --out te_x.json
```

单模型约 10–15 分钟（单卡，7B，60 条轨迹）。

### 9.2 plan hit-rate

脚本 `plan_hit.py`（本次新增，当前在 scratchpad，建议归档进 `scripts/`）：

```bash
python plan_hit.py \
  --models base=/path/Qwen2.5-7B-Instruct,cope200=/path/k3_step_200 \
  [--af-lora name=/path/af_lora.pt] \
  --trajs runs/sciworld_eval/FIXED_SUCCESS60 \
  --max-traj 60 --k 3 --batch-size 16 --out ph.json
```

单模型约 5 分钟。输出含 `records`（逐 prompt 命中记录），用于 trajectory-level bootstrap。

### 9.3 训练（separate-AF-LoRA）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 RUN_TAG=aflora ENVS_PER_GPU=1 \
MODEL_PATH=/path/Qwen2.5-7B-Instruct \
CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh \
TRAIN_FILE=/path/AgentGym-RL-Data-ID/train/sciworld_train.json \
HF_HOME=$HOME/.cache/huggingface \
PLAN_FORECAST_ENABLE=True PLAN_FORECAST_K=3 PLAN_FORECAST_COEF=0.01 \
AF_LORA_ENABLE=True AF_LORA_RANK=64 AF_LORA_ALPHA=128 AF_LORA_LR=1e-4 \
TOTAL_TRAINING_STEPS=201 SAVE_FREQ=50 \
bash launch_sciworld_grpo_tmux.sh
```

### 9.4 产物

| 内容 | 路径 |
|---|---|
| AF-LoRA 训练日志 / rollout | `runlogs/sciworld_qwen2.5_7b_aflora_k3_20260922_204006/` |
| AF-LoRA checkpoint（HF 权重 + `af_lora.pt`） | `checkpoints/sciworld_qwen2.5_7b_aflora_k3_20260922_204006/global_step_{50,100,150,200}/actor/` |
| AF-LoRA 测试集评测 | `runs/sciworld_qwen2.5_7b_aflora_k3_20260922_204006/eval/` |
| CoPE checkpoint | `~/workspace/models/CoPE_sciworld_qwen2.5_7b/k3_step_*` |
| TE 原始结果（含 per_traj） | scratchpad `te_*.json`、`te_summary.json` |
| plan hit-rate 原始结果（含 records） | scratchpad `ph2_*.json` |

> scratchpad 是会话级临时目录，需要长期保留的结果应尽快拷入仓库或其他持久位置。

---

## 10. 环境说明

本机为 8×RTX PRO 6000 Blackwell（sm_120），与仓库原始的 H100/H200 环境不同，
栈为 torch 2.7.0+cu128 / vLLM 0.9.2 / transformers 4.51.3 / flash-attn 2.8.3 / NCCL 2.27.3。
TE 与 plan hit-rate 均为纯 HF 前向，不经过 vLLM；训练侧的 vLLM 适配见
`AgentGym-RL/verl/third_party/vllm/vllm_spmd/llm.py`。

该环境有两个已修复的陷阱会静默影响结果，记录以备他人复现：
vLLM V1 的 cascade attention 在 sm_120 上会产生退化输出（已在适配层固定
`disable_cascade_attn=True`）；tmux 继承的 fd 软上限 1024 会在约第 28 步耗尽
（已在 env / 训练脚本中提升至硬上限）。
