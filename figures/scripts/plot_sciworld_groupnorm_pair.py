#!/usr/bin/env python3
"""SciWorld group_norm ablation, the matched K=3 pair.

  on  : sciworld_grpo_qwen2.5_7b_pfk3_20260829_043501   (group_norm = True)
  off : sciworld_grpo_qwen7b_planK3_nognorm_v4_20260906_091404 (group_norm = False)

Same K, coef, gate, skip_invalid, lr, entropy/KL coefficients, batch size,
rollout n, max_rounds and model; the runs sit on different machines and a week
apart, which is the caveat for the caption.

Window is the 191 steps both runs cover; single smoothing setting (median-9 then
mean-10). Panels beyond the task score are what the regression looks like upstream: the
policy gradient norm spikes, entropy collapses, responses get short.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG_DIR = Path(__file__).resolve().parents[1]
CSV = FIG_DIR / "data" / "sciworld_k3_pair_curves.csv"
XMAX = 191
MEDIAN, MEAN = 9, 10

ON, OFF = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
SERIES = [("A_gnorm_on", "group_norm on", ON), ("B_gnorm_off", "group_norm off", OFF)]

# (column, y label, panel title, ylim, show the unsmoothed series behind the line)
PANELS = [("task_score", "train task score", "training task score", (0, 1.0), False),
          ("actor_grad_norm", "policy grad norm", "policy gradient norm", None, True),
          ("entropy", "policy entropy", "policy entropy", None, False),
          ("resp_len", "response length (tokens)", "response length", None, False)]


def smooth(s):
    return (s.rolling(MEDIAN, min_periods=1, center=True).median()
             .rolling(MEAN, min_periods=max(2, MEAN // 2)).mean())


def style(ax, title, ylabel, ylim):
    ax.set_title(title, fontsize=10, color=INK, pad=8, loc="left")
    ax.set_xlabel("training step", fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.set_xlim(0, XMAX + 4)
    if ylim:
        ax.set_ylim(*ylim)


def draw(ax, data, column, raw=False):
    for label, legend, color in SERIES:
        d = data[(data.label == label) & (data.step <= XMAX)].sort_values("step")
        if d.empty or d[column].isna().all():
            continue
        if raw:
            # the spikes ARE the signal here, so the unsmoothed series stays visible
            ax.plot(d.step, d[column], color=color, linewidth=0.7, alpha=0.35, zorder=1)
        ax.plot(d.step, smooth(d[column]), label=legend, color=color,
                linewidth=2.0, solid_capstyle="round", zorder=2)


def annotate_drawdown(ax, data):
    """Mark the deepest drawdown of the group_norm-off run - the point the figure makes."""
    d = data[(data.label == "B_gnorm_off") & (data.step <= XMAX)].sort_values("step")
    y = smooth(d.task_score)
    dd = (y.cummax() - y)
    i = dd.idxmax()
    ax.annotate(f"-{dd[i]:.2f} over ~20 steps", (d.step[i], y[i]), xytext=(10, -26),
                textcoords="offset points", fontsize=8, color=OFF,
                arrowprops=dict(arrowstyle="->", color=OFF, lw=1.0))


def main():
    data = pd.read_csv(CSV)

    fig, axes = plt.subplots(1, 4, figsize=(15.6, 3.6))
    for ax, (col, ylabel, title, ylim, raw) in zip(axes, PANELS):
        draw(ax, data, col, raw)
        style(ax, title, ylabel, ylim)
    annotate_drawdown(axes[0], data)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left", handlelength=1.8)
    fig.suptitle("SciWorld, K=3: per-group weight normalization on vs off",
                 fontsize=11.5, color=INK, x=0.045, y=1.0, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "sciworld_groupnorm_k3_pair")

    # compact two-panel version for the paper body
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, (col, ylabel, title, ylim, raw) in zip(axes, PANELS[:2]):
        draw(ax, data, col, raw)
        style(ax, title, ylabel, ylim)
    annotate_drawdown(axes[0], data)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left", handlelength=1.8)
    fig.suptitle("SciWorld, K=3: per-group weight normalization on vs off",
                 fontsize=11.5, color=INK, x=0.06, y=1.0, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "sciworld_groupnorm_k3_pair_compact")


def save(fig, stem):
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
