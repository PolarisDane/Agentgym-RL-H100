# SciWorld 论文图

所有图、画图脚本、导出数据都在这个目录下。每张图对应的 wandb run id 见下面的「run 对照表」。

```
figures/
├── *.pdf / *.png          图（PDF 进论文，PNG 方便预览）
├── data/*.csv             从 wandb 导出的曲线数据
├── scripts/fetch_wandb.py 拉数据（写 data/*.csv）
└── scripts/plot_*.py      画图（读 data/*.csv，写 *.pdf + *.png）
```

## 怎么重跑

画图（只需要 matplotlib + pandas，不联网）：

```bash
PY=/usr/local/miniconda3/envs/mae/bin/python
cd /data1/repos/Agentgym-RL/figures
$PY scripts/plot_sciworld_horizon.py
$PY scripts/plot_sciworld_groupnorm_pair.py
$PY scripts/plot_sciworld_skipinvalid.py
$PY scripts/plot_sciworld_af_coef.py
$PY scripts/plot_sciworld_efficiency.py
```

重新从 wandb 拉数据（需要走反代，wandb key 从 `~/.netrc` 读）：

```bash
http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 \
no_proxy=localhost,127.0.0.1,::1 \
/usr/local/miniconda3/envs/agentgym-rl/bin/python scripts/fetch_wandb.py [horizon|k3_pair|groupnorm|skipinvalid|af_coef|efficiency|echo]
```

`fetch_wandb.py` 直接调 wandb 的 GraphQL API，没有用 wandb SDK——这台机器上 SDK 登录会卡在本地 service 进程超时。脚本里每个 run 用 **run id** 索引（run URL 末尾那串短码），所以在网页上改 run 名不会让图悄悄换数据。

## run 对照表

下表除特别注明外都在 entity `2488721971-sjtu` / project `agentgym-sciworld`，run 链接是
`https://wandb.ai/<entity>/agentgym-sciworld/runs/<run id>`。

| run id | wandb 显示名 | 这是什么 | 步数 | 机器 |
|---|---|---|---|---|
| `4ns6zsyw` | sciworld_grpo_qwen2.5_3b_add_20260711_182517 | K=1，skip_invalid on | 162 | inspire 4 卡 |
| `1rnmj68v` | sciworld_grpo_qwen2.5_3b_add_20260711_221713 | K=1，skip_invalid **off** | 209 | inspire 4 卡 |
| `wxahf6yd` | sciworld_grpo_qwen2.5_3b_add_20260723_063403 | K=3 | 206 | inspire 4 卡 |
| `dyk546f9` | sciworld_grpo_qwen7b_planK2_gnorm_20260907_205338 | **K=2，我们的主设置**（group_norm on，skip_invalid on） | 299 | 我们 8 卡 |
| `xwrvan9c` | sciworld_grpo_qwen7b_planK2_noskip_20260909_041423 | K=2，skip_invalid **off** | 299 | 我们 8 卡 |
| `b4jtuf8m` | sciworld_grpo_qwen7b_planK3_8gpu_20260905_020435 | K=3，group_norm on（中途挂了） | 114 | 我们 8 卡 |
| `c1f9il4w` | sciworld_grpo_qwen2.5_7b_pfk3_20260828_144718 | K=3，尝试 1 | 179 | 集群 B 8 卡 |
| `w2az2lg6` | sciworld_grpo_qwen2.5_7b_pfk3_20260829_043501 | K=3，尝试 2（group_norm **on**，用于 group_norm 对比） | 191 | 集群 B 8 卡 |
| `cs7pem30` | sciworld_grpo_qwen7b_planK3_nognorm_20260905_134509 | K=3，group_norm off，尝试 1 | 99 | 我们 8 卡 |
| `c3opfb2c` | sciworld_grpo_qwen7b_planK3_nognorm_v2_20260906_030701 | K=3，group_norm off，尝试 2 | 31 | 我们 8 卡 |
| `xvmlprq3` | sciworld_grpo_qwen7b_planK3_nognorm_v3_20260906_055425 | K=3，group_norm off，尝试 3 | 39 | 我们 8 卡 |
| `yfgxqpnu` | sciworld_grpo_qwen7b_planK3_nognorm_v4_20260906_091404 | K=3，group_norm **off**（用于 group_norm 对比） | 299 | 我们 8 卡 |
| `3kvxihjo` | sciworld_grpo_qwen7b_planK4_gnorm_20260911_085545 | K=4 | 299 | 我们 8 卡 |
| `47y1k0x0` | sciworld_grpo_qwen2.5_7b_pfk5_20260826_131959 | K=5，第 1 段（1–177 步） | 177 | 集群 B 8 卡 |
| `la0p7i43` | sciworld_grpo_qwen2.5_7b_pfk5_20260826_131959 | K=5，续跑段（151–299 步） | 149 | 集群 B 8 卡 |
| `9tqxr5o7` | sciworld_grpo_qwen7b_BASELINE_nopf_20260913_102029 | **纯 GRPO baseline** | 299 | 我们 8 卡 |
| `6ip8v3j2` | no_dedup_wm_loss_only | **ECHO baseline**：只开 world-model SFT，`world_model_coeff=0.01`，`wmc_erc.enable=False` | 263 | inspire 4 卡 |
| `g5n2964d` | GRPO | ECHO 同集群的纯 GRPO（跨集群偏移参考，未进图） | 115 | inspire 4 卡 |

系数扫描那组在 entity **`co-evolve-neurips`** / project `agentgym-sciworld`：

| run id | wandb 显示名 | 这是什么 | 步数 | 状态 |
|---|---|---|---|---|
| `tndk3zog` | sciworld_ablate_pfc_0.0001_20260917_113918 | af coef = 1e-4 | 253 | running |
| `h9op8mn0` | sciworld_ablate_pfc_0.001_20260917_113905 | af coef = 1e-3（**上下文 8192 / 每轮 512 token**，与其余三个不同） | 152 | crashed |
| `wqqny0ib` | sciworld_ablate_pfc32k_0.01_20260918_224812 | af coef = 1e-2 | 168 | crashed |
| `321njph9` | sciworld_ablate_pfc32k_0.1_20260918_103719 | af coef = 1e-1 | 233 | crashed |
| `cyn5yomo` | sciworld_base_grpo_20260918_143806 | 同 project 的纯 GRPO 参考线（forecast 关） | 263 | finished |
| `gjwmeqvi` | sciworld_ablate_pfc32k_0.01_20260918_103719 | af coef = 1e-2 的**另一次尝试**（98 步，峰值仅 0.42，未进图） | 98 | crashed |
| `8bbjbfh1` / `xmwhk1hj` | sciworld_ablate_pfc_0.01 / _0.1_20260917_0112 | 8k 上下文下 1e-2 / 1e-1 的早期尝试（75 / 21 步，未进图） | 75 / 21 | crashed |

注意：
- K=1 和 K=3（inspire）那几个 run 名字里的 `3b` 是脚本模板残留，`model.path` 实际是 **Qwen2.5-7B-Instruct**。
- `47y1k0x0` 和 `la0p7i43` 是同一次实验的两段（续跑），画图时按 step 拼接，重叠段取后一段。

## 图 → 脚本 → 数据 → run

### 1. Forecast horizon 消融

| | |
|---|---|
| 图 | `sciworld_task_score_horizon_k1to5.{pdf,png}`（主图，左 10 步右 20 步平滑）<br>`sciworld_task_score_horizon_k1to5_smooth10/20.{pdf,png}`（单栏版）<br>`sciworld_task_score_horizon_k1to5_ema.{pdf,png}`（EMA 平滑的备选）<br>`sciworld_task_score_horizon_k3_variants.{pdf,png}`（四个 K=3 run 的对照） |
| 脚本 | `scripts/plot_sciworld_horizon.py` |
| 数据 | `data/sciworld_horizon_curves.csv` |
| run | K=1 `4ns6zsyw`、K=2 `dyk546f9`、K=3 `wxahf6yd`、K=4 `3kvxihjo`、K=5 `47y1k0x0`+`la0p7i43`、baseline `9tqxr5o7`；K=3 备选 `w2az2lg6`/`c1f9il4w`/`b4jtuf8m` |

横轴到 200 步（之后所有长 run 都会崩，不是消融要说的事）。平滑是 9 步滑动中位数 + 滑动均值，中位数那一步用来吃掉单步骤崩。

**caveat**：K=1/K=3 在 4 卡集群，K=5 和 K=3 备选在集群 B，K=2/K=4/baseline 在我们机器；各 1 个 seed。主图 K=3 用的是 `wxahf6yd`，换成别的 K=3 run 会明显改变这一档的位置（见 variants 图）。

### 2. group_norm 消融（K=3 配对）

| | |
|---|---|
| 图 | `sciworld_groupnorm_k3_pair.{pdf,png}`（四栏：task score / 策略梯度范数 / 熵 / 回复长度）<br>`sciworld_groupnorm_k3_pair_compact.{pdf,png}`（两栏，正文用） |
| 脚本 | `scripts/plot_sciworld_groupnorm_pair.py` |
| 数据 | `data/sciworld_k3_pair_curves.csv` |
| run | on = `w2az2lg6`，off = `yfgxqpnu` |

窗口取两者都覆盖的 191 步。配置逐项 diff 过，实质超参只差 `plan_forecast_group_norm` 一项。

**结论**：最终水平几乎一样（末 10 步 0.766 对 0.780），但 off 在第 128–149 步有一次 0.74→0.45 的塌陷（最大回撤 0.293 对 0.104），同时策略梯度范数飙到 7.28（on 全程 ≤1.62）、熵掉到 0.051（on 0.166）、回复长度掉到 183 token（on 542）。

**caveat**：两个 run 不同机器、相隔一周、代码版本也不同（on 是 8/29 的旧实现，off 是 9/6 group_norm 权重修正之后的代码），严格说不是一次对照实验；各 1 个 seed。`yfgxqpnu` 当时还没记 `group_n_distilled`/`group_unique_frac`。

`data/sciworld_groupnorm_curves.csv` 是我们机器上那组（`b4jtuf8m` 对四个 nognorm run）的数据，结论与上面相反（off 的 `yfgxqpnu` 反而最好），图已删除，数据留着备查。

### 3. skip_invalid 消融（K=1、K=2）

| | |
|---|---|
| 图 | `sciworld_skip_invalid_k1_k2.{pdf,png}`（2×2：上 task score 下熵）<br>`sciworld_task_score_skip_invalid_k1_k2.{pdf,png}`（只有 task score） |
| 脚本 | `scripts/plot_sciworld_skipinvalid.py` |
| 数据 | `data/sciworld_skipinvalid_curves.csv` |
| run | K=1 on `4ns6zsyw` / off `1rnmj68v`；K=2 on `dyk546f9` / off `xwrvan9c` |

**目前最干净的两组对照**：逐项 diff 只差 `plan_forecast_skip_invalid` 一项，K=1 那对还是同机器同一天。

**结论**：on 只在 K=1 早期更快（到 0.6 快 24 步），K=2 上这个优势消失；两个 K 上 on 的熵都塌得更厉害、回复更短、回撤是 off 的 2–3 倍，末段成绩还略低。也就是数据**不支持** skip_invalid 有益。

### 4. action-forecast 系数扫描

| | |
|---|---|
| 图 | `sciworld_task_score_af_coef.{pdf,png}`（单栏，前 150 步） |
| 脚本 | `scripts/plot_sciworld_af_coef.py` |
| 数据 | `data/sciworld_af_coef_curves.csv` |
| run | 1e-4 `tndk3zog`、1e-3 `h9op8mn0`、1e-2 `wqqny0ib`、1e-1 `321njph9`、参考线 `cyn5yomo` |

四个 run 都是 K=3、group_norm on、skip_invalid on、4 卡，GRPO 超参一致。只画前 150 步（所有 run 都覆盖的窗口）。

窗口内数字（9 步中位数 + 10 步均值平滑）：

| | 峰值 | @100 步 | @150 步 | 最大回撤 |
|---|---|---|---|---|
| coef 1e-4 | 0.645 | 0.404 | 0.590 | 0.073 |
| coef 1e-3（8k 上下文） | 0.735 | 0.527 | 0.721 | 0.052 |
| coef 1e-2 | 0.779 | 0.556 | 0.770 | 0.141 |
| **coef 1e-1** | **0.841** | **0.730** | 0.767 | 0.137 |
| 无 forecast | 0.537 | 0.244 | 0.507 | 0.074 |

**结论**：系数越大收敛越快，1e-1 在前 100 步遥遥领先（第 100 步 0.73，无 forecast 只有 0.24），1e-2 次之；到 150 步时 1e-1 / 1e-2 / 1e-3 收敛到同一水平（0.72–0.77），1e-4 落在 0.59。四档全部明显高于无 forecast（0.51）。注意这跟 SciWorld 主实验用的 1e-2 不同——这组设置下 1e-1 的早期优势更大。

**caveat**：
- **1e-3 那条不在同一条扫描线上**：它的 `max_model_len=8192`、每轮 `max_tokens=512`，其余三个是 32768 / 200。图例里标了 `(8k ctx)`，正式作图前最好用同设置补一个 1e-3。
- 1e-2 还有另一次尝试 `gjwmeqvi`（98 步，峰值 0.42，远低于 `wqqny0ib` 的 0.77），说明这组实验的 run 间方差不小；图里用的是跑得更久的那个。
- 除 1e-4 外全部是 crashed 状态（中途挂掉，不是训练发散），1e-4 仍在跑；各 1 个 seed。

### 5. 样本效率 / 单步训练效率 vs GRPO

| | |
|---|---|
| 图 | `sciworld_efficiency_vs_grpo.{pdf,png}`（三栏：按步数 / 按墙钟 / 单步耗时拆解） |
| 脚本 | `scripts/plot_sciworld_efficiency.py` |
| 数据 | `data/sciworld_efficiency_curves.csv` + `data/sciworld_echo_baseline_curves.csv` |
| run | 我们 `dyk546f9`、GRPO `9tqxr5o7`、ECHO `6ip8v3j2` |

前 150 步，到各档 reward 需要的步数（9 步中位数 + 10 步均值平滑后）：

| | 到 0.2 | 到 0.4 | 到 0.6 |
|---|---|---|---|
| action forecast (K=2) | 31 步 / 3.3 h | 60 步 / 6.2 h | 86 步 / 8.7 h |
| world-model SFT only (ECHO) | 53 步 | 114 步 | 未达到（峰值 0.54） |
| GRPO | 106 步 / 12.1 h | 144 步 / 16.2 h | 未达到（峰值 0.45） |

单步耗时（前 150 步均值，仅同机器的两条）：我们 363.6 s（其中 action-forecast 那次 backward 56.7 s，占 15.6%），GRPO 405.8 s。我们反而快 10%，因为 rollout 生成便宜了：回复 282 token / 14.3 轮，GRPO 是 1038 token / 16.8 轮。

**caveat**：ECHO 在 4 卡集群（单步 206 s），墙钟不可比，所以只出现在第一栏；那个集群的纯 GRPO（`g5n2964d`）收敛比我们机器的 baseline 略快（0.2 档 80 对 106 步），也就是说图里 ECHO 的位置偏乐观。三条线都是单 seed。回复变短有一部分来自熵塌，写「更省时间」时最好把长度和轮数一并给出。

## 通用口径

- 纵轴 `train task score` = `critic/task_score/mean`，训练集 rollout 的成功率。SciWorld 上 `critic/rewards/mean` 与它相同（reward 就是 0/1 成功）。
- 平滑：9 步滑动中位数 → 滑动均值（10 或 20 步）。中位数那一步只吃单步骤崩，持续十几步的真实回落会保留。
- 配色取自 data-viz 参考色板：分类槽位 1–5（蓝 `#2a78d6`、橙 `#eb6834`、青 `#1baf7a`、黄 `#eda100`、品红 `#e87ba4`），baseline 一律中性灰 `#52514e` 虚线。
- 评测（测试集）成功率口径是 **done 且 score ≥ 100**，与训练 reward 一致；这个目录里的图目前全部是训练曲线，没有用评测数据。
