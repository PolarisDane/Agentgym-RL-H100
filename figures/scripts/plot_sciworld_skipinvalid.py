#!/usr/bin/env python3
"""SciWorld skip_invalid ablation at K=1 and K=2.

Two matched pairs, each differing in exactly one config entry
(plan_forecast_skip_invalid); config diffs were checked and nothing else moves.

  K=1  on : sciworld_grpo_qwen2.5_3b_add_20260711_182517   (162 steps)
       off: sciworld_grpo_qwen2.5_3b_add_20260711_221713   (207 steps)
       -- same machine, same day; the "3b" in the name is a stale template, both
          runs are Qwen2.5-7B-Instruct.
  K=2  on : sciworld_grpo_qwen7b_planK2_gnorm_20260907_205338   (299 steps)
       off: sciworld_grpo_qwen7b_planK2_noskip_20260909_041423  (297 steps)
       -- our machine, two days apart.

Top row is the training task score, bottom row policy entropy: skip_invalid buys
early speed and pays for it with a faster entropy collapse.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG_DIR = Path(__file__).resolve().parents[1]
CSV = FIG_DIR / "data" / "sciworld_skipinvalid_curves.csv"
MEDIAN, MEAN = 9, 10
XMAX = {1: 162, 2: 200}          # K=1 pair only reaches 162 steps

ON, OFF = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
ROWS = [("task_score", "train task score", (0, 1.0)),
        ("entropy", "policy entropy", None)]


def smooth(s):
    return (s.rolling(MEDIAN, min_periods=1, center=True).median()
             .rolling(MEAN, min_periods=max(2, MEAN // 2)).mean())


def style(ax, title, ylabel, ylim, xmax, xlabel=True):
    if title:
        ax.set_title(title, fontsize=10, color=INK, pad=8, loc="left")
    if xlabel:
        ax.set_xlabel("training step", fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.set_xlim(0, xmax + 4)
    if ylim:
        ax.set_ylim(*ylim)


def draw(ax, data, K, column):
    for on, legend, color in ((True, "skip_invalid on", ON), (False, "skip_invalid off", OFF)):
        d = data[(data.K == K) & (data.skip == on) & (data.step <= XMAX[K])].sort_values("step")
        if d.empty or d[column].isna().all():
            continue
        ax.plot(d.step, smooth(d[column]), label=legend, color=color,
                linewidth=2.0, solid_capstyle="round")


def main():
    data = pd.read_csv(CSV)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), sharex="col")
    for col, K in enumerate((1, 2)):
        for row, (column, ylabel, ylim) in enumerate(ROWS):
            ax = axes[row][col]
            draw(ax, data, K, column)
            style(ax, f"K = {K}" if row == 0 else "", ylabel if col == 0 else "",
                  ylim, XMAX[K], xlabel=(row == 1))
    axes[0][0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left",
                      handlelength=1.8)
    fig.suptitle("SciWorld: dropping no-effect actions from the forecast target",
                 fontsize=11.5, color=INK, x=0.055, y=1.0, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "sciworld_skip_invalid_k1_k2")

    # task score only, one row - for the paper body
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for ax, K in zip(axes, (1, 2)):
        draw(ax, data, K, "task_score")
        style(ax, f"K = {K}", "train task score", (0, 1.0), XMAX[K])
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left", handlelength=1.8)
    fig.suptitle("SciWorld: dropping no-effect actions from the forecast target",
                 fontsize=11.5, color=INK, x=0.055, y=1.0, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "sciworld_task_score_skip_invalid_k1_k2")


def save(fig, stem):
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
