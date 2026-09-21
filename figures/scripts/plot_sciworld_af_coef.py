#!/usr/bin/env python3
"""SciWorld: action-forecast coefficient sweep (co-evolve-neurips project).

  coef 0.0001  tndk3zog   sciworld_ablate_pfc_0.0001_20260917_113918   (running, 253 steps)
  coef 0.001   h9op8mn0   sciworld_ablate_pfc_0.001_20260917_113905    (crashed, 152 steps)
  coef 0.01    wqqny0ib   sciworld_ablate_pfc32k_0.01_20260918_224812  (crashed, 168 steps)
  coef 0.1     321njph9   sciworld_ablate_pfc32k_0.1_20260918_103719   (crashed, 233 steps)
  reference    cyn5yomo   sciworld_base_grpo_20260918_143806  - same project, forecast off

All K=3, group_norm on, skip_invalid on, 4 GPUs, same GRPO hyper-parameters.
CAVEAT: the 0.001 run rolls out with max_model_len 8192 and 512 tokens per turn,
the others with 32768 and 200, so that one line is not strictly on the sweep.

Only the first 150 steps are drawn - the window all four runs cover.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG_DIR = Path(__file__).resolve().parents[1]
CSV = FIG_DIR / "data" / "sciworld_af_coef_curves.csv"
MEDIAN, MEAN = 9, 10
XMAX = 150                 # the window all four runs cover

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
# categorical slots 1..4 for the four coefficients, neutral gray for the reference
SERIES = [("coef_0.0001", "coef 1e-4", "#2a78d6", "-"),
          ("coef_0.001", "coef 1e-3  (8k ctx)", "#eb6834", "-"),
          ("coef_0.01", "coef 1e-2", "#1baf7a", "-"),
          ("coef_0.1", "coef 1e-1", "#eda100", "-"),
          ("base_grpo", "no forecast", "#52514e", (0, (5, 3)))]


def smooth(s):
    return (s.rolling(MEDIAN, min_periods=1, center=True).median()
             .rolling(MEAN, min_periods=max(2, MEAN // 2)).mean())


def style(ax, title, xmax):
    ax.set_title(title, fontsize=10, color=INK, pad=8, loc="left")
    ax.set_xlabel("training step", fontsize=9, color=MUTED)
    ax.set_ylabel("train task score", fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.set_xlim(0, xmax + 5)
    ax.set_ylim(0, 1.0)


def draw(ax, data, xmax=XMAX, mark_ends=False):
    for label, legend, color, ls in SERIES:
        d = data[(data.label == label) & (data.step <= xmax)].sort_values("step")
        if d.empty:
            continue
        y = smooth(d.task_score)
        ax.plot(d.step, y, label=legend, color=color, linewidth=2.0,
                linestyle=ls, solid_capstyle="round")
        last = int(d.step.iloc[-1])
        if mark_ends and last < xmax:      # the run stopped here, it did not collapse
            ax.plot([last], [y.iloc[-1]], "o", ms=4, color=color, zorder=3)


def main():
    data = pd.read_csv(CSV)

    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    draw(ax, data)
    style(ax, "SciWorld, K=3: action-forecast loss coefficient", XMAX)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left", handlelength=2.0)
    fig.tight_layout()
    save(fig, "sciworld_task_score_af_coef")

    # numbers behind the figure
    print(f"{'run':<22}{'steps':>7}{'peak':>8}{'last10':>9}{'@100':>8}{'@150':>8}{'maxDD':>8}")
    for label, legend, _, _ in SERIES:
        d = data[(data.label == label) & (data.step <= XMAX)].sort_values("step")
        y = smooth(d.task_score)
        at = lambda s: y[d.step.between(s - 5, s + 5)].mean()
        dd = (y.cummax() - y).max()
        print(f"{legend:<22}{int(d.step.max()):>7}{y.max():>8.3f}{y.iloc[-1]:>9.3f}"
              f"{at(100):>8.3f}{at(150):>8.3f}{dd:>8.3f}")


def save(fig, stem):
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
