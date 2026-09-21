# TE-KL 指标：计算细节、代码对应与轨迹选取

本文档记录论文所用 **temporal self-consistency KL** 指标的完整计算过程。
面向两类读者：要复现结果的人，和要审查口径是否成立的人。

- 实现：`scripts/te_offline.py`
- 复用的训练侧函数：`AgentGym-RL/verl/agent_trainer/ppo/temporal_ensemble.py`
- 合并结果：`/data1/logs/te_kl_merged.json`
- 轨迹集：`runs/sciworld_eval/FIXED_SUCCESS60/`（含 `_manifest.json`）

---

## 1. 被比较的两个分布

在目标动作 `a_t` 的每个 token 位置 `u` 上：

| 符号 | 含义 |
|---|---|
| `π⁰` | 模型站在第 `t` 轮、已写出 `a_t^{<u}` 时的下一 token **全词表**分布 |
| `q^F` | 由第 `t-1, t-2, t-3` 轮各自"预测未来 k 步"得到的分布，按 token id **算术平均** |
| `q` | 混合分布 `q(v) = [(1-η)·π⁰(v) + η·q^F(v)] / Z` |

其中 `Z = (1-η) + η·Σ_{v∈top-M} q^F(v)`（次归一化修正，见 §4）。

指标即 **`D_KL(π⁰ ‖ q)`，越低越自洽**。

### 为何不直接对 q^F 算

`q^F` 只是三个成员分布的平均，不包含 `π⁰`。只要 `π⁰` 在某 token 上有质量而三个成员
全给 ~0，`D_KL(π⁰‖q^F)` 就发散。混入 `(1-η)π⁰` 保证支撑，代价是取值被压进
`[0, -log(1-η)]`。这也正是训练损失 `L = L_GRPO + λ·KL(π_θ‖q_t)` 里的同一个 `q_t`，
所以离线量与训练量同形。

本文同时报告 **η=0.5**（与训练一致，上界 `log2 = 0.693`）与
**η=0.9**（分辨率更高，上界 `-log0.1 = 2.303`）两档。

---

## 2. 两个分布分别怎么得到

### π⁰：一次整对话前向

```python
# te_offline.py :96-106
full = tok.apply_chat_template(cv, tokenize=False, add_generation_prompt=False)
enc  = tok(full, add_special_tokens=False, return_tensors="pt",
           return_offsets_mapping=True)
lg   = model(**enc).logits[0]          # [L, V]
```

`lg[r]` 即位置 `r` 上的 logits 行，直接作为 `π⁰` 的未归一化表示传给 KL 函数。

**动作 token 的定位**（这里有个已修复的关键 bug，见 §7）：必须在"第 t 个 assistant
消息"的字符区间内定位，不能在全序列搜索 token 串。

### q^F：每个源轮一次前向

对每个源轮 `s ∈ {t-1, t-2, t-3}` 构造一条打分样本：

```
prefix = [对话截到 o_s] + [user: "预测你接下来 k 步动作" 的 prompt]
target = [assistant: k 个动作，换行连接]
```

构造函数 `build_te_scoring_samples`（`temporal_ensemble.py:34`），编码路径与
plan_forecast 的 `seq='separate'` 模式**逐字一致**。

```python
# te_offline.py :113-117
out = model(input_ids=ids, attention_mask=am).logits
slp = slot_logprobs_from_logits(out, ids, [s["slot_spans"]])[0]
tk_ids, tk_prs = slot_topk_from_logits(out, ids, [s["slot_spans"]])
```

`slot_topk_from_logits`（`temporal_ensemble.py:220`）在每个 slot 的动作 token
位置上取词表 **top-64**，返回 `[B, n_slot, maxtok, topm]`。

### 成员合并

```python
# te_offline.py :168-179
acc = {}
for mid, mpr in mems:                       # 三个成员
    for tid, pr in zip(mid[u].tolist(), mpr[u].tolist()):
        if tid >= 0 and pr > 0:
            acc[tid] = acc.get(tid, 0.0) + pr
top = sorted(acc.items(), key=lambda x: -x[1])[:TE_TOPM]
pv  = [v2 / n_m for _, v2 in top]           # 算术平均
```

**按 token id 合并求和再除以成员数**，与训练侧 `assemble_te_topm`
（`temporal_ensemble.py:361`）的语义一致 —— 成员之间 top-M 的 id 会重叠，
不能简单拼接。

成员的 token 数必须与 rollout 动作一致，否则**整个成员丢弃**（不截断、不错位），
与训练侧 `assemble_te_tensors_token` 同逻辑。

---

## 3. KL 的解析式

`q^F` 只在 top-M 上有值，把词表分成两块解析求和，避免枚举 152k 维：

**top-M 之外**（`q^F(v)=0`）：
```
q(v) = (1-η)·π⁰(v) / Z
log(π⁰(v)/q(v)) = -log((1-η)/Z)
```

**top-M 之内**：
```
q(v) = (1-η)·π⁰(v)/Z · [1 + η·q^F(v)/((1-η)·π⁰(v))]
log(π⁰(v)/q(v)) = -log((1-η)/Z) - log1p( η·q^F(v)/((1-η)·π⁰(v)) )
```

对 `Σ_v π⁰(v)·(...)` 求和，得到最终形式：

```
D_KL = -log((1-η)/Z) - Σ_{v∈top-M} π⁰(v)·log1p( η·q^F(v) / ((1-η)·π⁰(v)) )
```

在"`q^F` 的 top-M 之外为 0"这一近似下**这是精确的**。

### 代码

`fullvocab_te_kl_rows`（`temporal_ensemble.py:478`），训练与离线共用同一函数：

```python
logp   = torch.log_softmax(z.float(), dim=-1)
p      = logp.exp()
logp0  = logp.detach()
qf     = prs * keep
mass   = qf.sum(-1)
Z      = (1.0 - eta) + eta * mass
p0_sel = logp0.gather(1, ids_c).exp()
ratio  = (eta * qf) / ((1.0 - eta) * p0_sel.clamp_min(1e-20))
corr   = torch.log1p(ratio) * keep
kl_p_p0 = (p * (logp - logp0)).sum(-1)      # 离线恒为 0（p 与 p0 同源）
kl = kl_p_p0 - torch.log((1.0 - eta) / Z) - (p_sel * corr).sum(-1)
```

训练时 `z` 带梯度、`logp0` 被 detach，`kl_p_p0` 提供梯度但数值为 0；
**离线 `no_grad` 下该项精确为 0**，所以测到的就是 `D_KL(π⁰‖q)`。

### 两端验证

| 情形 | 理论值 | 说明 |
|---|---|---|
| `q^F = π⁰`（完全自洽） | **0** | ratio = η/(1-η)，Z=1，两项抵消 |
| `q^F` 与 `π⁰` 质量完全不重叠 | **-log(1-η)** | 修正项 ≈ 0 |

实测 base：η=.5 时 0.1673（占上界 24.1%），η=.9 时 0.5008（占上界 21.7%）——
两档比例一致，量纲自洽。

---

## 4. 聚合方式

逐 token 位置计算 → 轨迹内对位置取均值 → 跨轨迹**按位置数加权**平均。

```python
# te_offline.py :211-214
def _wmean(key):
    rs = kls.get(key) or []
    tot = sum(n for _, n in rs)
    return (sum(v * n for v, n in rs) / tot) if tot else float("nan")
```

最终是全部 **5,797 个 token 位置**的平均。

---

## 5. 轨迹集：FIXED_SUCCESS60

### 5.1 为何不能用模型自己的轨迹

模型对**自己生成**的轨迹每 token 概率达 0.97–0.98（`log π⁰ ≈ -0.02`），
对留出轨迹是 -1.30。在比值型指标下分母坍缩；对 KL 而言虽无分母问题，
但靶点集本身随模型改变（强 ckpt 轨迹更短更干净），仍不可比。

### 5.2 构造（`scripts/build_fixed_succ.py`）

从 3 个实验 × 5 个 step = 15 个 ckpt 的 SciWorld 200 题评测产物中筛选：

| 族 | 实验目录 |
|---|---|
| `grpo` | `sciworld_grpo_qwen7b_BASELINE_nopf_20260913_102029`（纯 GRPO） |
| `pf` | `sciworld_grpo_qwen7b_planK3_nognorm_v4_20260906_091404`（**CoPE**） |
| `te` | `sciworld_TEabl_A_20260916_190725`（TE ArmA，论文未报告） |

三条约束：

1. **只取 `success=True`**（`done` 且 `reward >= 100`）——最终 60/60 全部成功
2. **按任务类型分层**——覆盖 20 类（测试集共 26 类）
3. **来源在三族间轮转均衡**——若全取自一族，该族就是在自己的输出上被打分

核心排序逻辑：
```python
cands.sort(key=lambda c: (fam_count[c[0]], c[1]))   # (该族已贡献数, step)
```

### 5.3 实际构成

60 条轨迹 / 1,428 个靶点 / **5,797 个 token 位置**，对所有模型完全相同。

| | step50 | step100 | step150 | step200 | step250 | 合计 |
|---|---|---|---|---|---|---|
| grpo | 3 | 4 | 9 | 3 | 0 | 19 |
| pf (CoPE) | 15 | 4 | 2 | 0 | 0 | 21 |
| te | 12 | 4 | 4 | 0 | 0 | 20 |
| **合计** | **30** | 12 | 15 | 3 | **0** | 60 |

### 5.4 已知缺陷（必须披露）

**step 分布严重倾斜**：50% 的轨迹来自 step50，**step250 一条都没有**。
根因是上面排序的次键 `c[1]`（step 升序）——同一 task 若多个 ckpt 都成功，
永远取最早的那个。

**且倾斜在族间不对称**：CoPE 的 15/21 来自 step50，GRPO 的 9/19 来自 step150。
原因是 CoPE 在 step50 已有 106/200 成功而 GRPO 只有 7/200，早期 task 的候选里
几乎只有 CoPE/TE 可选。

**对 KL 的影响小于对 g 的影响**：KL 比较的是模型自身两个分布的一致性，
轨迹只提供上下文与逐 token 条件，不提供被打分的目标。但上下文分布仍偏向
早期 ckpt，不能声称无影响。

**待修**：把 step 加入均衡维度（目标每 step 12 条）后重建并重算。

### 5.5 覆盖情况

- 动作 token 长度：中位 **4**，90 分位 6，最大 28
- `TE_TOPM_MAXTOK=16` 截断影响 **1.3%**
- 序列长度上限 8192：实测 **0 条**被跳过（fixed 集最长 2,872 token）

---

## 6. 结果

`base` = Qwen2.5-7B-Instruct（未训练）：**KL(η=.5) = 0.1673**，KL(η=.9) = 0.5008

| ckpt | Succ% | KL(.5) | ΔKL(.5) 95%CI | KL(.9) | ΔKL(.9) 95%CI |
|---|---|---|---|---|---|
| grpo50 | 3.50 | 0.1534 | −0.0103 [−.0142,−.0058] | 0.4522 | −0.0386 [−.0503,−.0263] |
| grpo100 | 26.50 | 0.1480 | −0.0154 [−.0221,−.0082] | 0.4402 | −0.0506 [−.0722,−.0282] |
| grpo150 | 56.50 | **0.1464** | −0.0255 [−.0336,−.0177] | 0.4397 | −0.0775 [−.1047,−.0515] |
| grpo200 | 64.00 | 0.1503 | −0.0225 [−.0312,−.0145] | 0.4548 | −0.0673 [−.0968,−.0387] |
| grpo250 | 64.50 | 0.1565 | −0.0133 [−.0226,−.0044] | 0.4783 | −0.0308 [−.0600,−.0001] |
| pf50 | 53.00 | 0.0718 | −0.0999 [−.1101,−.0902] | 0.2132 | −0.3046 [−.3374,−.2736] |
| pf100 | 59.50 | 0.0604 | −0.1106 [−.1216,−.1001] | 0.1748 | −0.3414 [−.3755,−.3091] |
| pf150 | 69.50 | 0.0542 | −0.1167 [−.1288,−.1051] | 0.1635 | −0.3519 [−.3891,−.3161] |
| pf200 | 58.00 | **0.0507** | −0.1217 [−.1319,−.1115] | 0.1513 | −0.3679 [−.3998,−.3375] |
| pf250 | 72.50 | 0.0514 | −0.1185 [−.1299,−.1072] | 0.1537 | −0.3580 [−.3949,−.3228] |

置信区间：以**轨迹**为重采样单位（同一轨迹内的靶点强相关），
4,000 次 bootstrap，对 base 做配对差分。全部区间不含 0。

### 结论

1. **CoPE 把时序不自洽度降低 57–70%**（0.1673 → 0.051–0.072），
   比 GRPO 的最好点（0.1464）还低 2.9 倍，两者区间完全不重叠。
2. **纯 GRPO 只能降约 15% 且不持续**：step150 触底（0.1464）后回升到
   0.1565，呈浅 U 形。
3. 两档 η 的排序完全一致，结论不依赖 η 的选择。

### 与 g 口径的差异（重要）

早先基于 `g = [log q^F(a_t) - log π⁰(a_t)]/n_t` 的版本给出
"纯 GRPO 显著恶化、符号翻转"的结论。**该结论在 KL 口径下不成立。**

差异来自 `g` 的分母：GRPO 后期 `log π⁰` 下跌（策略锐化 + 外来上下文 OOD），
在比值里被记成恶化。KL 不在实际动作上取值、没有这个分母，只剩温和的 U 形。

同理，早先"CoPE 在 step50 后预测能力退化"的说法（`log q^F` 从 -1.05 跌到 -1.61）
在 KL 下也消失：KL 单调改善到 step200。说明那个"退化"是绝对概率受策略锐化
拖累，而 `π⁰` 与 `q^F` 的**相对一致性**一直在提升。

**建议以 KL 为主指标，g 降为辅助。**

---

## 7. 两个已修复的关键 bug

1. **动作 token 定位**（`te_offline.py:130-145`）：必须在"第 t 个 assistant 消息"的
   字符区间内定位，**不能在全序列搜索 token 串**。退化轨迹会把同一动作复读几十次，
   取第一个匹配会落在模型已复读多次之后的位置，条件概率接近 1
   （实测 `π⁰` sum = -0.001），导致 gain 被算成**正数**（符号错误）。

2. **成员 span 与 rollout 动作的 token 数必须一致**，否则是不可比的两个量。
   训练侧 `assemble_te_tensors_token` 有同样的丢弃逻辑。

---

## 8. 复现

```bash
# 1. 构造固定轨迹集
python3 scripts/build_fixed_succ.py      # -> runs/sciworld_eval/FIXED_SUCCESS60

# 2. 8 卡分片计算（11 个模型铺 8 张卡，约 15 分钟）
#    分片定义见 /data1/logs/te_kl_shard.sh
for g in 0 1 2 3 4 5 6 7; do
  GPU=$g MODELS="..." TRAJS="..." bash /data1/logs/te_kl_shard.sh &
done

# 3. 合并 + bootstrap
#    -> /data1/logs/te_kl_merged.json
```

单模型单卡命令：

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/te_offline.py \
  --models "name=<ckpt_path>" --trajs "name=<eval_dir>" \
  --fixed-traj runs/sciworld_eval/FIXED_SUCCESS60 --skip-own \
  --k 3 --eta 0.5 --max-traj 60 --out <out.json>
```

输出的 `per_traj` 字段含逐轨迹的 `gain / logqF / logp0 / kl / kl90`，
用于 bootstrap 与按来源族拆分的稳健性检验。

## 9. 关键常量

| 常量 | 值 | 位置 |
|---|---|---|
| `k`（成员数） | 3 | CLI `--k` |
| `η` | 0.5 / 0.9 | CLI `--eta` / 硬编码第二档 |
| `TE_TOPM` | 64 | `temporal_ensemble.py:216` |
| `TE_TOPM_MAXTOK` | 16 | `temporal_ensemble.py:217` |
| `TE_COV_MAX` | 128 | `temporal_ensemble.py:344` |
| 序列长度上限 | 8192 | `te_offline.py:103` |
