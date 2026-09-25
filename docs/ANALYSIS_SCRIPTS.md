# 分析脚本使用说明

本文档覆盖 2026-09 新增的四个分析脚本与 `te_offline.py` 的新开关。它们都是**只读推理**，
不改训练流程，可以在任意有 checkpoint 的机器上独立运行。

实验背景与结论见 `TE_AF_LORA_ABLATION.md`；指标定义见 `TE_KL_COMPUTATION.md`、`TE_METRIC_SETTING.md`。

共同约定：

- 轨迹集用 `runs/sciworld_eval/FIXED_SUCCESS60`（60 条成功轨迹，1428 个动作轮）。
  该目录的 `_manifest.json` 记录每条轨迹来自哪个 run，用于检查 provenance。
- 所有脚本吃 HF 格式的模型目录。训练产出的 FSDP 分片需先用
  `AgentGym-RL/scripts/model_merger.py --local_dir <ckpt>/actor` 合并（约 20 秒）。
- 置信区间一律按**轨迹级** bootstrap 计算：同一轨迹内的轮次高度相关，按样本重采样会低估方差。
  脚本输出里保留了逐轨迹/逐样本的原始记录，供事后重算。

---

## plan_hit.py — 自由生成的计划命中率

在每个动作轮用训练时同一个合成 prompt 让模型**自由生成**未来 K 步计划（greedy），
再与该轨迹实际执行的动作逐条精确匹配。与 TE 互补：TE 是 teacher-forced 的连续似然，
这个是自由生成的离散匹配，模型无法靠"被训练在同形式的似然上"来提高它。

```bash
python scripts/plan_hit.py \
  --models base=/path/Qwen2.5-7B-Instruct,ckpt=/path/global_step_200/actor/huggingface \
  [--af-lora ckpt=/path/global_step_200/actor/af_lora.pt] \
  --trajs runs/sciworld_eval/FIXED_SUCCESS60 \
  --max-traj 60 --k 3 --batch-size 16 --out ph.json
```

输出字段：

- `exact` / `loose`：严格匹配 / 忽略冠词。`hit0` 是当前动作（模仿），`future` 是 j≥1（预见）
- `cond_future`：只统计确实产出了完整 k 条计划的轮次——**必看**，用于剥离格式因素
- `plan_rate`：产出完整计划的比例。纯 GRPO 训练会把策略推向 rollout 风格
  （`Thought:` + 单个 `Action:`），该值会掉到 40–55%，低于未训练模型
- `records`：逐 prompt 的 gold / parsed / 命中，可据此做动作类型与参数的拆分

**解析口径是这个指标的关键**（`extract_actions`）。不同系统的输出格式差异极大：裸列表、
`Thought:...Action: x`、带编号的混合格式。按行严格比对会把 rollout 风格策略的正确动作
判为 miss（实测 hit@0 从 35% 被误判到 0%）。解析器会剥离散文行、编号、bullet，并从
`Action:` 后取动作。改这个函数会改变所有系统的绝对值，跨版本的数字不可混用。

单模型约 5 分钟（60 轨迹 / 795 prompt / 单卡）。

---

## te_offline.py --af-lora — 交叉 TE

原脚本用同一个模型同时产生 q^F（成员预测）与 π⁰（当场分布）。新开关让**成员侧**前向启用
一个 LoRA 适配器，π⁰ 仍来自不带适配器的策略，用于回答"forecaster 准不准"与
"策略是否与它自洽"这两个可以分离的问题。

```bash
python scripts/te_offline.py \
  --models x=/path/global_step_200/actor/huggingface \
  --af-lora x=/path/global_step_200/actor/af_lora.pt \
  --trajs dummy=runs/sciworld_eval/FIXED_SUCCESS60 \
  --fixed-traj runs/sciworld_eval/FIXED_SUCCESS60 \
  --max-traj 60 --k 3 --eta 0.5 --skip-own --out te_x.json
```

不传 `--af-lora` 时行为与改动前完全一致。

> **交叉设定下的 Δg 不可解读为自洽性。** g = log q^F − log π⁰ 此时是跨模型量，含义是
> "适配器比策略自己更能预判该动作"。该设定下应报 Δlog q^F。

---

## repr_probe.py — 冻结表示的线性探针

取 a_t 之后、o_{t+1} 之前那个位置的隐状态，冻结，训练线性多标签分类器预测
"o_{t+1} 中会出现哪些实体"。所有系统用相同探针容量、相同样本、相同划分，
差异只来自表示里编码了什么。不经过输出空间，因此不受回复格式与文本分布漂移影响。

```bash
python scripts/repr_probe.py \
  --models base=/path/base,ckpt=/path/hf \
  [--af-lora ckpt=/path/af_lora.pt] \
  --trajs runs/sciworld_eval/FIXED_SUCCESS60 \
  --max-traj 60 --top-entities 50 --layer -1 --C 0.01 --out rp.json
```

按轨迹分组做 5 折交叉验证，报四个桶的 AUC。同时写出 `<out>.<model>.npz`
（特征、折外预测、标签、分桶、分组），**之后换探针容量、换分桶、算置信区间都不必再占 GPU**。

需要 `scikit-learn`。单模型约 8 分钟。

---

## obs_probe.py — 观测预测探针（三种模式）

问 AF 训练有没有在参数里留下环境动力学知识。三种模式分别对应三次迭代：

```bash
python scripts/obs_probe.py --models ... [--af-lora ...] \
  --trajs runs/sciworld_eval/FIXED_SUCCESS60 --max-traj 60 \
  --mode {natural|prompted|cloze|ablate} --out obs.json
```

- `natural`（推荐）：观测在**它实际出现的位置**（下一个 user 轮）被打分，无合成指令。
  所有系统的条件与训练/rollout 一致，避免"对陌生 prompt 的适应度"混入结果。
- `prompted`：插入 WM-SFT 的原始指令。仅作稳健性检查，或为将来的 WM-SFT 参照保持公平。
  实测 base 在该模式下连"照抄上文已有的词"都变难（carry NLL 7.71 对自然位置的 1.56）。
- `cloze`：实体级完形填空，在承载动力学信息的实体位上比较真实实体与 3 个干扰实体。
  随机基线 25%，比整条观测的 NLL 判别有headroom得多。
- `ablate`：在同一实体位测两次，只把上下文里的动作换成同轨迹的另一个动作，
  报"动作敏感度"。同槽位、同候选集、同模型，全局漂移天然抵消。

四个分桶：`echo`（当前动作的宾语，照抄即可）、`carry`（上一观测已有）、
`new_ref`（新出现且被后续动作引用）、`new_noref`（新出现但后续不用）。

> **已知问题：`echo` 与 `carry` 不是等价的对照。** 两者证据距离不同、难度差约 16 个点
> （base 上 0.798 对 0.935），以它们为分母做归一化会得到相反结论。三次迭代都遇到这个问题，
> 结论因此不稳。SciWorld 的观测 85% 只有 6 个词且多为动作回声，动力学信号过于稀薄——
> 这套探针更适合 WebShop / AppWorld 这类观测较长的环境。详见 `TE_AF_LORA_ABLATION.md`。

---

## merge_af_lora.py — 把 AF-LoRA 折进权重

LoRA 是线性的，`W' = W + (alpha/r)·B·A` 精确等价。合并后得到标准 HF checkpoint，
可直接走既有的 vLLM 评测路径，不需要让推理栈理解 hook 形式的适配器。

```bash
python scripts/merge_af_lora.py \
  --hf  <ckpt>/actor/huggingface \
  --lora <ckpt>/actor/af_lora.pt \
  --out /path/merged_step200
```

合并后建议核验：合并模型与"backbone + hook 启用"的 logits 应高度一致
（实测相关 0.999934、argmax 一致、top-5 相同；残差 0.22 来自 bf16 累加顺序），
而与关闭 LoRA 的差异应明显更大（实测 5.72）。

> **合并后的模型不能直接当策略用。** 适配器被训练成输出裸动作列表，合并后这个格式会
> 覆盖决策格式：实测 6000 个生成回合中含 `Action:` 的为 0，环境 93.75% 回复
> "No known action matches that input"，200 题成功率 0.00%。若要考察"世界模型能否帮助决策"，
> 需要降低合并强度（`W + α·s·BA`，扫 α<1）或只在计划环节挂适配器，而不是整体合并。

---

## 训练侧开关：AF-LoRA

`scripts/run_sciworld_grpo_train.sh` 新增，默认关闭时与改动前逐字节一致：

```
AF_LORA_ENABLE=True AF_LORA_RANK=64 AF_LORA_ALPHA=128 AF_LORA_LR=1e-4
AF_LORA_TARGETS=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

开启后，plan-forecast 的 CE 只更新 LoRA，policy backbone 拿不到 forecast 梯度，
策略更新在数学上等价纯 GRPO。每步记录 `af_lora/policy_delta`（策略参数切片在 AF 更新
前后的最大变化），**该值必须恒为 0**，非 0 即说明存在梯度泄漏，该 run 的结论不成立。

实现上有两个会静默出错的坑，见 `af_lora.py` 顶部与 `TE_AF_LORA_ABLATION.md` §3.2：
适配器必须在**反向**过程中同样处于启用状态（梯度检查点会重算前向）；
`af_lora_targets` 含逗号，传给 Hydra 时必须加引号。
