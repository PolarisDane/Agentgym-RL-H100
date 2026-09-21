#!/usr/bin/env python3
"""SciWorld: sample efficiency and per-step cost, our best setting vs plain GRPO.

  ours : sciworld_grpo_qwen7b_planK2_gnorm_20260907_205338  (K=2, group_norm on,
         skip_invalid on) - the best-performing action-forecast run on our machine
  grpo : sciworld_grpo_qwen7b_BASELINE_nopf_20260913_102029 (no auxiliary loss)
  echo : no_dedup_wm_loss_only (id 6ip8v3j2) - world-model SFT only, wm_coeff=0.01,
         wmc_erc.enable=False, no action forecast. Same model and GRPO
         hyper-parameters, but a DIFFERENT cluster (4 GPUs), so it appears in the
         sample-efficiency panel only - its wall-clock is not comparable.

ours and grpo share the same 8xH100 machine and identical GRPO hyper-parameters,
so the wall-clock panels compare those two. Window is the first 150 steps.

Panels: score vs step, score vs wall clock, and the per-step time broken into
stages (the action-forecast pass is the extra stage our method adds).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG_DIR = Path(__file__).resolve().parents[1]
CSV = FIG_DIR / "data" / "sciworld_efficiency_curves.csv"
CSV_ECHO = FIG_DIR / "data" / "sciworld_echo_baseline_curves.csv"
XMAX = 150
MEDIAN, MEAN = 9, 10
THRESHOLDS = (0.2, 0.4, 0.6)

OURS, ECHO, GRPO = "#2a78d6", "#1baf7a", "#52514e"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
# runs with comparable wall clock (same machine) - used by every panel
RUNS = [("ours_K2", "action forecast (K=2)", OURS, "-"),
        ("grpo", "GRPO", GRPO, (0, (5, 3)))]
# step-axis only (different cluster): world-model SFT baseline
RUNS_STEP = RUNS + [("echo_wmsft", "world-model SFT only", ECHO, (0, (1, 1.6)))]
# stage -> (column, color); "other" is the remainder of timing_s/step
STAGES = [("rollout generation", "t_gen", "#2a78d6"),
          ("policy update", "t_update_actor", "#eb6834"),
          ("action-forecast pass", "t_pf", "#1baf7a"),
          ("log-prob + ref", "t_logp_ref", "#eda100"),
          ("other", "t_other", "#c9c8c2")]


def smooth(s):
    return (s.rolling(MEDIAN, min_periods=1, center=True).median()
             .rolling(MEAN, min_periods=max(2, MEAN // 2)).mean())


def style(ax, title, xlabel, ylabel, ylim=None):
    ax.set_title(title, fontsize=10, color=INK, pad=8, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    if ylim:
        ax.set_ylim(*ylim)


def prep(data):
    out = {}
    for label, *_ in RUNS_STEP:
        g = data[(data.label == label) & (data.step <= XMAX)].sort_values("step").reset_index(drop=True)
        g["score"] = smooth(g.task_score)
        g["hours"] = g.t_step.fillna(g.t_step.median()).cumsum() / 3600.0
        g["t_logp_ref"] = g.t_logp.fillna(0) + g.t_ref.fillna(0)
        g["t_pf"] = g.t_pf.fillna(0) if "t_pf" in g else 0.0
        g["t_other"] = (g.t_step - g.t_gen.fillna(0) - g.t_update_actor.fillna(0)
                        - g.t_pf - g.t_logp_ref).clip(lower=0)
        out[label] = g
    return out


def reach(g, th):
    """(step, hours) where the smoothed score first reaches th, or None."""
    hit = g.index[g.score.ge(th)]
    if len(hit) == 0:
        return None
    i = hit[0]
    return int(g.step[i]), float(g.hours[i])


def curve_panel(ax, runs, xcol, xlabel, title, series):
    colors = {l: c for l, _, c, _ in series}
    for label, legend, color, ls in series:
        g = runs[label]
        ax.plot(g[xcol], g.score, label=legend, color=color, linewidth=2.0,
                linestyle=ls, solid_capstyle="round")
    for th in THRESHOLDS:
        ax.axhline(th, color=GRID, linewidth=0.8, zorder=0)
        for label, *_ in series:
            r = reach(runs[label], th)
            if r:
                x = r[0] if xcol == "step" else r[1]
                ax.plot([x], [th], "o", ms=5, color=colors[label], zorder=3)
    style(ax, title, xlabel, "train task score", (0, 0.9))


def main():
    data = pd.concat([pd.read_csv(CSV), pd.read_csv(CSV_ECHO)], ignore_index=True)
    runs = prep(data)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    curve_panel(axes[0], runs, "step", "training step", "sample efficiency", RUNS_STEP)
    curve_panel(axes[1], runs, "hours", "wall clock (hours)", "wall-clock efficiency (same machine)", RUNS)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left", handlelength=2.0)

    # per-step cost, stacked by stage
    ax = axes[2]
    for i, (label, legend, _, _) in enumerate(RUNS):
        g = runs[label]
        bottom = 0.0
        for name, col, color in STAGES:
            v = float(g[col].mean()) if col in g else 0.0
            if v <= 0:
                continue
            ax.bar(i, v, bottom=bottom, width=0.55, color=color,
                   label=name if i == 0 or name == "action-forecast pass" else None,
                   edgecolor="white", linewidth=2.0)
            bottom += v
        ax.text(i, bottom + 6, f"{bottom:.0f}s", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(RUNS)))
    ax.set_xticklabels([legend for _, legend, _, _ in RUNS], fontsize=8.5, color=INK)
    style(ax, "per-step time (mean over 150 steps)", "", "seconds per step")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), handlelength=1.2)

    fig.suptitle("SciWorld: action forecast (K=2) vs GRPO, first 150 steps",
                 fontsize=11.5, color=INK, x=0.045, y=1.0, ha="left")
    fig.tight_layout(rect=(0, 0, 0.88, 0.93))
    save(fig, "sciworld_efficiency_vs_grpo")

    # the numbers behind the figure
    print(f"{'':<22}" + "".join(f"{f'reach {t}':>18}" for t in THRESHOLDS))
    for label, legend, _, _ in RUNS_STEP:
        cells = []
        for th in THRESHOLDS:
            r = reach(runs[label], th)
            cells.append(f"{r[0]:>4d} steps {r[1]:5.1f}h" if r else f"{'not reached':>18}")
        print(f"{legend:<22}" + "".join(f"{c:>18}" for c in cells))
    for label, legend, _, _ in RUNS:
        g = runs[label]
        print(f"{legend:<22} mean step {g.t_step.mean():6.1f}s   forecast pass "
              f"{float(g.t_pf.mean()):5.1f}s   gen {g.t_gen.mean():6.1f}s   "
              f"response {g.resp_len.mean():5.0f} tok   rounds {g.task_round.mean():4.1f}")


def save(fig, stem):
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
