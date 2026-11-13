"""Plots ms per window: BirdNET-Go's stock default vs the NEON engine.

One small panel per (board, model), each with its own linear scale, so the
bars in a panel compare by length; the value is printed above each bar. The
default column is one thread, from the 2026-11-08 sessions
(results/comparison.csv, sessions C001 / C003); the engine columns are from
the later thread tables in results/timing/ (pi5_detector_stopped.txt,
pi4.txt).

usage: python -I scripts/plot_final_perf.py [OUT.png]
  default OUT: results/figures/final_perf.png (relative to the checkout)
  needs matplotlib: pip install -e '.[plot]'
"""
import os
import sys

from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib import ticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "results", "figures", "final_perf.png")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
RULE = "#d6d4cc"
GRID = "#e9e7e0"
SERIES = (
    ("BirdNET-Go default, 1 thread", "#b4b2a9"),
    ("NEON engine, 1 thread", "#2a78d6"),
    ("NEON engine, 4 threads", "#1baf7a"),
)

# (model, default ms, NEON 1 thread ms, NEON 4 threads ms).
BOARDS = (
    ("Raspberry Pi 5 (engine: Nov 10, detector stopped; default: Nov 8)", (
        ("BirdNET v2.4", 146.8, 19.5, 11.5),
        ("BirdNET v3.0 NA-east", 642.3, 126.4, 52.7),
        ("BirdNET v3.0 global", 754.3, 146.9, 66.9),
        ("Perch v2", 635.9, 142.0, 49.2),
        ("BattyBirdNET (EU)", 79.9, 32.8, 15.6),
    )),
    ("Raspberry Pi 4 (engine: Nov 11; default: Nov 8)", (
        ("BirdNET v2.4", 536.0, 129.5, 65.6),
        ("BirdNET v3.0 NA-east", 2321.8, 949.3, 325.6),
        ("BirdNET v3.0 global", 2646.0, 1125.8, 375.4),
        ("Perch v2", 2210.4, 1113.8, 336.1),
        ("BattyBirdNET (EU)", 282.1, 123.3, 59.0),
    )),
)


def format_ms(value: float) -> str:
    """A millisecond value as short text: one decimal below 100."""
    if value < 100:
        return f"{value:.1f}"
    return f"{value:,.0f}"


def style_axis(axis, top: float) -> None:
    """Scale, ticks, grid and spines of one panel."""
    axis.set_ylim(0, top)
    axis.set_xlim(-0.6, 2.6)
    axis.set_xticks([])
    axis.yaxis.set_major_locator(ticker.MaxNLocator(5, steps=[1, 2, 5, 10]))
    axis.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    axis.tick_params(axis="y", labelsize=8, labelcolor=INK_SOFT, color=RULE,
                     length=3)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(RULE)


def draw_panel(axis, row: tuple) -> None:
    """One model's bars with the ms value above each bar."""
    model, default_ms, *neon = row
    for slot, value in enumerate((default_ms, *neon)):
        axis.bar(slot, value, width=0.62, color=SERIES[slot][1])
        axis.text(slot, value + default_ms * 0.015, format_ms(value),
                  ha="center", va="bottom", fontsize=9.5, color=INK)
    axis.set_title(model, fontsize=10, color=INK, pad=6)
    style_axis(axis, default_ms * 1.1)


def build_figure():
    """The two-board figure."""
    figure, axes = plt.subplots(2, 5, figsize=(13, 6.4), facecolor=SURFACE)
    for board_axes, (board, rows) in zip(axes, BOARDS):
        for axis, row in zip(board_axes, rows):
            draw_panel(axis, row)
        board_axes[0].text(-0.14, 1.2, board,
                           transform=board_axes[0].transAxes,
                           fontsize=11.5, fontweight="bold", color=INK)
        board_axes[0].set_ylabel("ms per window", fontsize=9, color=INK_SOFT)
    handles = [patches.Rectangle((0, 0), 1, 1, color=color)
               for _, color in SERIES]
    figure.legend(handles, [name for name, _ in SERIES], loc="upper right",
                  ncol=3, frameon=False, fontsize=10,
                  bbox_to_anchor=(0.985, 0.995))
    figure.suptitle("Milliseconds per window (lower is better)", x=0.015,
                    y=0.985, ha="left", fontsize=13, color=INK)
    figure.subplots_adjust(left=0.055, right=0.985, top=0.83, bottom=0.03,
                           hspace=0.5, wspace=0.3)
    return figure


def main(argv: list[str]) -> int:
    """Writes the figure; returns 0."""
    out = argv[1] if len(argv) > 1 else DEFAULT_OUT
    plt.switch_backend("Agg")
    build_figure().savefig(out, dpi=170, facecolor=SURFACE)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
