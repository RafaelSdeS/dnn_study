import matplotlib.pyplot as plt
from typing import Iterable
from cycler import cycler
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

# Report palette -- the single source of figure style for report/ scripts and notebooks/.
# presentation/make_figures.py keeps its own slide palette on purpose (different medium).
BLUE = "#1565c0"
RED = "#dc2626"
GREEN = "#16a34a"
PURPLE = "#9333ea"
AMBER = "#d97706"
PALETTE = [BLUE, RED, GREEN, PURPLE, AMBER]
MARKERS = ["o", "s", "^", "D", "v"]

TEXT_PRIMARY = "#1a1a1a"
TEXT_SECONDARY = "#4d4d4d"
GRID = "#d9d9d9"
NEUTRAL = "#9e9e9e"  # reference points/bars that belong to no model group

STAGE_COLORS = {"fp32": BLUE, "qat": PURPLE, "int8": RED}
STAGE_MARKERS = {"fp32": MARKERS[0], "qat": MARKERS[1], "int8": MARKERS[2]}
STAGE_LABELS = {"fp32": "FP32", "qat": "QAT", "int8": "INT8"}

# Narrative group -> color and model -> group, so a model has the same color in every figure.
GROUP_COLORS = {
    "Baselines irrestritos": BLUE,
    "Restrição ingênua": RED,
    "Tentativas de compensação": GREEN,
    "Arquitetura híbrida final": PURPLE,
    "Atenção local": AMBER,
}
# Two spellings of ResNet18/SmallKernel appear because the source CSVs disagree.
MODEL_GROUP = {
    "alexnet_tv": "Baselines irrestritos", "mobilenetv2": "Baselines irrestritos",
    "resnet18_tv": "Baselines irrestritos", "resnet18tv": "Baselines irrestritos",
    "vgg_style": "Baselines irrestritos",
    "alexnet_3x3_fc": "Restrição ingênua", "alexnet_2x2_fc": "Restrição ingênua",
    "alexnet_3x3_gap": "Restrição ingênua",
    "alexnet_mixed": "Tentativas de compensação", "alexnet_small_kernel": "Tentativas de compensação",
    "alexnet_smallkernel": "Tentativas de compensação",
    "alexnet_residual": "Tentativas de compensação", "alexnet_bottleneck": "Tentativas de compensação",
    "alexnet_fire": "Tentativas de compensação", "alexnet_depthwisesep": "Tentativas de compensação",
    "alexnet_dilated_gap": "Tentativas de compensação", "alexnet_stacked": "Tentativas de compensação",
    "alexnet_groupconv": "Tentativas de compensação", "alexnet_factorized": "Tentativas de compensação",
    "alexnet_se": "Tentativas de compensação",
    # Legacy duplicate result files (same architecture/params/MACs as their non-suffixed twin from
    # a later superseded run) -- same group as their twin.
    "alexnet_2x2": "Restrição ingênua", "alexnet_2x2_gap": "Restrição ingênua",
    "alexnet_3x3": "Restrição ingênua", "alexnet_gap": "Restrição ingênua",
    "alexnet_final_fire_residual": "Arquitetura híbrida final",
    "alexnet_final_bottleneck_residual": "Arquitetura híbrida final",
    "alexnet_final_bottleneck_fire": "Arquitetura híbrida final",
    "alexnet_final_depthwise_fire": "Arquitetura híbrida final",
    "alexnet_fire_bypass": "Arquitetura híbrida final",
    "swin_pico_w2": "Atenção local", "swin_pico_w4": "Atenção local",
    "swin_pico_w8": "Atenção local", "swin_pico_poolmixer": "Atenção local",
    "hybrid_bottleneck_swin": "Atenção local", "vit_tiny": "Atenção local",
    "deit_tiny": "Atenção local",
    "vit_tiny_convstem": "Atenção local", "swin_pico_convstem": "Atenção local",
}

# Phase 7 detection backbones -- bottleneck and fire share a group, so fire takes AMBER.
BACKBONE_COLORS = {"alexnet_bottleneck": GREEN, "alexnet_fire": AMBER, "alexnet_tv": BLUE}


def shade(color, amount: float) -> tuple:
    """Blend toward white (amount < 0) or black (amount > 0), so a family's members keep its hue."""
    target = 0.0 if amount > 0 else 1.0
    return tuple(c + (target - c) * abs(amount) for c in to_rgb(color))


def group_colors_for(models: Iterable[str]) -> list:
    return [GROUP_COLORS.get(MODEL_GROUP.get(m), NEUTRAL) for m in models]


def group_legend(models: Iterable[str]) -> list:
    """Legend patches for only the groups actually plotted, in GROUP_COLORS order."""
    present = {MODEL_GROUP.get(m) for m in models}
    return [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items() if g in present]


def apply_report_style(palette: Iterable[str] = PALETTE, figsize: tuple = (6.5, 4), dpi: int = 150) -> None:
    """The report's look as rcParams: white ground, light grid, no top/right spines, grey ticks."""
    plt.style.use("default")
    plt.rcParams.update({
        "figure.figsize": figsize,
        "figure.dpi": dpi,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.prop_cycle": cycler(color=list(palette)),
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "text.color": TEXT_PRIMARY,
        "xtick.color": GRID,
        "ytick.color": GRID,
        "xtick.labelcolor": TEXT_SECONDARY,
        "ytick.labelcolor": TEXT_SECONDARY,
        "legend.fontsize": 11,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


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
