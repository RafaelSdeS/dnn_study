import matplotlib.pyplot as plt
from typing import Iterable, Optional

# Colorblind-safe categorical palette (blue/orange/aqua/violet), validated all-pairs CVD-safe in
# light mode (OKLab worst-pair deltaE 9.2 CVD / 16.3 normal-vision -- dataviz skill palette.md).
# Canonical for every notebook under notebooks/ -- import these instead of redefining local hex
# codes, so a given color/marker means the same thing in every figure across the report.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
MARKERS = ["o", "s", "^", "D"]

# fp32/qat/int8 is a recurring identity across notebooks -- pin it to fixed palette slots so it
# never depends on a given figure's plotting order. fp32/int8 match PALETTE[0]/[1] everywhere,
# including 2-stage (fp32/int8-only) figures; qat takes PALETTE[3], which those figures leave
# unclaimed, so a 3rd stage never collides with a 2-stage figure's fp32/int8 hues.
STAGE_COLORS = {"fp32": PALETTE[0], "qat": PALETTE[3], "int8": PALETTE[1]}
STAGE_MARKERS = {"fp32": MARKERS[0], "qat": MARKERS[1], "int8": MARKERS[2]}
STAGE_LABELS = {"fp32": "FP32", "qat": "QAT", "int8": "INT8"}

# One consistent color per phase label, in first-seen/declared order, reused across every chart
# that breaks results down by phase (e.g. phases 5 and 10's cross-phase comparisons).
PHASE_CMAP = plt.get_cmap("tab10")


def phase_colors_for(phases: Iterable[str], order: list) -> list:
    return [PHASE_CMAP(order.index(p) % 10) if p in order else "#999999" for p in phases]


def apply_report_style(fontsize: int = 11,
                       style: str = "seaborn-v0_8-whitegrid",
                       palette: Optional[Iterable[str]] = None,
                       figsize: tuple = (6.5, 4),
                       dpi: int = 150) -> None:
    """Apply a consistent plotting style for figures used in the report.

    - Sets matplotlib style and rcParams used across notebooks.
    - If `palette` is provided, sets the axes color cycle accordingly.
    """
    plt.style.use(style)
    base_rc = {
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "legend.fontsize": max(9, int(fontsize * 0.9)),
        "figure.dpi": dpi,
        "savefig.dpi": max(150, dpi),
        "savefig.bbox": "tight",
        "figure.figsize": figsize,
    }
    plt.rcParams.update(base_rc)

    if palette is not None:
        try:
            from cycler import cycler

            plt.rcParams.update({"axes.prop_cycle": cycler(color=list(palette))})
        except Exception:
            # fall back silently if cycler not available or palette invalid
            pass


def label_bars(ax, fmt="{:.1f}", xpos="center", ypos_offset=0.0):
    """Label bars in a bar plot with their height.

    - `fmt` is a format string applied to the bar height.
    - `xpos` passed to `ha` for text alignment.
    """
    for p in ax.patches:
        try:
            h = p.get_height()
            if h is None:
                continue
            ax.annotate(fmt.format(h),
                        (p.get_x() + p.get_width() / 2.0, h + ypos_offset),
                        ha=xpos, va='bottom', fontsize=max(8, int(plt.rcParams.get('font.size', 11) * 0.9)))
        except Exception:
            continue
