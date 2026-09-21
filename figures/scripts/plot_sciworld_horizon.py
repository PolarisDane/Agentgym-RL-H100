#!/usr/bin/env python3
"""SciWorld forecast-horizon ablation: training task_score for K = 1..5.

Data: figures/data/sciworld_horizon_curves.csv (exported from wandb, one row per
training step per run). Draws one smoothed line per K plus the no-forecast
baseline, at two smoothing windows (10 and 20 steps), and a variant figure with
every K=3 run so the K=3 pick can be judged.

Colors are the ordinal blue ramp (light mode, lightest step >= 250) from the
data-viz reference palette; the baseline is neutral gray so it never reads as a
horizon setting.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG_DIR = Path(__file__).resolve().parents[1]
CSV = FIG_DIR / "data" / "sciworld_horizon_curves.csv"
METRIC = "task_score"          # == critic/task_score/mean; on SciWorld reward is the same series
WINDOWS = (10, 20)
XMAX = 200                     # past ~250 steps every long run collapses; the ablation lives before that
MEDIAN = 9                     # median pre-filter: kills the one-step dips without moving the level
RAW_ALPHA = 0.0                # raw curve behind the line: 0 = off (6 series make it clutter)

# K -> categorical slots 1..5 of the reference palette, in fixed order. Five noisy
# lines overlap here, so identity beats the ordinal blue ramp (which was tried
# first and left K=1..3 indistinguishable).
K_COLOR = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100", 5: "#e87ba4"}
BASELINE_COLOR = "#52514e"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdbd6"

# main figure: one run per K (K=3 from the same cluster as K=1, the longest K=3 run)
MAIN = [("K1", "K=1"), ("K2", "K=2"), ("K3_inspire", "K=3"), ("K4", "K=4"), ("K5", "K=5")]
K3_VARIANTS = [("K3_inspire", "K=3  inspire, 4 GPU"),
               ("K3_xy_b", "K=3  cluster B, run 2"),
               ("K3_xy_a", "K=3  cluster B, run 1"),
               ("K3_ours", "K=3  ours, 8 GPU")]


def style(ax, title, ylabel=True):
    ax.set_title(title, fontsize=10, color=INK, pad=8, loc="left")
    ax.set_xlabel("training step", fontsize=9, color=MUTED)
    if ylabel:
        ax.set_ylabel("train task score", fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, XMAX + 12)


def smooth(series, window, style="median"):
    """median pre-filter + rolling mean (default), or an EMA of the raw series."""
    if style == "ema":
        return series.ewm(span=window, adjust=False).mean()
    s = series.rolling(MEDIAN, min_periods=1, center=True).median() if style == "median" else series
    return s.rolling(window, min_periods=max(2, window // 2)).mean()


def draw(ax, data, series, window, colors, dashes=None, end_labels=False, style="median"):
    for label, legend in series:
        d = data[data.label == label].sort_values("step")
        d = d[d.step <= XMAX]
        if d.empty:
            continue
        if RAW_ALPHA:
            ax.plot(d.step, d[METRIC], linewidth=0.8, color=colors[label],
                    alpha=RAW_ALPHA, zorder=1)
        y = smooth(d[METRIC], window, style)
        ax.plot(d.step, y, label=legend, linewidth=2.0,
                color=colors[label], solid_capstyle="round",
                linestyle=(dashes or {}).get(label, "-"))
        last = int(d.step.iloc[-1])
        if end_labels and label != "BASELINE" and last < XMAX:
            # only the runs that stopped early are labelled, so a short curve is
            # never mistaken for a collapse; the rest are carried by the legend
            ax.plot([last], [y.iloc[-1]], "o", ms=4, color=colors[label], zorder=3)
            ax.annotate(f"{legend} ends @{last}", (last, y.iloc[-1]), xytext=(6, 6),
                        textcoords="offset points", fontsize=7.5, color=colors[label],
                        va="bottom", clip_on=False)


def main():
    data = pd.read_csv(CSV)

    # --- main figure: K = 1..5 + baseline, both smoothing windows -----------
    colors = {lab: K_COLOR[int(lab[1])] for lab, _ in MAIN} | {"BASELINE": BASELINE_COLOR}
    series = MAIN + [("BASELINE", "no forecast")]
    dashes = {"BASELINE": (0, (5, 3))}

    for sty, tag in (("median", ""), ("ema", "_ema")):
      fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), sharey=True)
      for ax, w in zip(axes, WINDOWS):
        draw(ax, data, series, w, colors, dashes, end_labels=True, style=sty)
        style(ax, (f"median-{MEDIAN} + mean-{w}" if sty == "median" else f"EMA span {w}"),
              ylabel=ax is axes[0])
      axes[1].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left",
                     bbox_to_anchor=(1.06, 0.0), handlelength=1.8)
      fig.suptitle("SciWorld: forecast horizon K", fontsize=11.5, color=INK, x=0.065, y=0.99, ha="left")
      fig.tight_layout(rect=(0, 0, 0.88, 0.95))
      save(fig, f"sciworld_task_score_horizon_k1to5{tag}")

    # one file per window as well, for single-column use
    for w in WINDOWS:
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        draw(ax, data, series, w, colors, dashes, end_labels=True)
        style(ax, f"SciWorld: forecast horizon K  (smoothed over {w} steps)")
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left", handlelength=1.8)
        fig.tight_layout()
        save(fig, f"sciworld_task_score_horizon_k1to5_smooth{w}")

    # --- variant figure: every K=3 run --------------------------------------
    v_colors = {lab: c for (lab, _), c in zip(K3_VARIANTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"])}
    v_colors["BASELINE"] = BASELINE_COLOR
    v_series = K3_VARIANTS + [("BASELINE", "no forecast")]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
    for ax, w in zip(axes, WINDOWS):
        draw(ax, data, v_series, w, v_colors, dashes, end_labels=True)
        style(ax, f"smoothed over {w} steps", ylabel=ax is axes[0])
    axes[1].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left",
                   bbox_to_anchor=(1.01, 1.0), handlelength=1.8)
    fig.suptitle("SciWorld: the four K=3 runs", fontsize=11.5, color=INK, x=0.065, y=0.99, ha="left")
    fig.tight_layout(rect=(0, 0, 0.88, 0.95))
    save(fig, "sciworld_task_score_horizon_k3_variants")


def save(fig, stem):
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight",
                    facecolor="white")
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
