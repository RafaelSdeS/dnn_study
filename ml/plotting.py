import matplotlib.pyplot as plt
from typing import Iterable, Optional


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
