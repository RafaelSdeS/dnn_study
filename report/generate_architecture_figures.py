"""Architecture diagrams for the report appendix (Eixos 1/2/4).

Ported verbatim from notebooks/phase_5_cross_phase_results_analysis/
architecture_internals_phase5.ipynb (cells 2/6/8/10 + the four save calls), which is what
actually produced the committed figures. The previous contents of this file were a diverged,
cruder implementation that did NOT reproduce them (hardcoded figsize=(42,16) instead of sizing
the canvas from the diagram content), so running it silently replaced good figures with bad
ones. Keep this file and the notebook cells in sync.

Usage:  python report/generate_architecture_figures.py
"""
import matplotlib
matplotlib.use("Agg")

import gzip
import sys
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def find_project_root(start: Path | None = None):
    """Walk upward until the repository root is found."""
    start = start or Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "README.md").exists() and (candidate / "results").exists() and (candidate / "models").exists():
            return candidate
    return start


project_root = find_project_root()
sys.path.insert(0, str(project_root))  # so `import ml`/`import models` resolve regardless of kernel cwd

import ml.model_registrations  # noqa: F401  (populates MODEL_REGISTRY)
from ml import MODEL_REGISTRY
from ml.quantization import load_best_model
from ml.plotting import apply_report_style, PALETTE
from models.compensation import _AlexBottleneck, _FireModule
from models.final_architecture import _FireResBlock

apply_report_style(palette=PALETTE)

PCAD_DIR = project_root / "outputs" / "pcad"
LARGE_SCALE_DIR = PCAD_DIR / "archive_legacy_phases" / "phase_4_5_large_scale"
FIGURES_DIR = project_root / "report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


device = torch.device("cpu")
INPUT_HW = (64, 64)  # Tiny ImageNet-200 input resolution
NUM_CLASSES = 200
print(f"Project root: {project_root}")

# The diagrams describe ARCHITECTURE only -- kernel sizes, channel counts, stage structure --
# so the models are built from the registry rather than loaded from checkpoints. The notebook
# loads trained checkpoints here, but two of the four (alexnet_bottleneck, alexnet_fire) no
# longer exist on disk, and their weights never affected the drawing.
models = {
    name: MODEL_REGISTRY[name]["ctor"]().to(device).eval()
    for name in ("alexnet_bottleneck", "alexnet_fire",
                 "alexnet_final_fire_residual", "alexnet_fire_bypass")
}

METRICS = {
    # Sizes are WEIGHTS-ONLY (serialized state_dict), matching the convention fixed on
    # 2026-09-02 in ml/reporting.py -- they previously reported the training checkpoint,
    # which also carries AdamW optimizer state (~3x the model). See report Sec. "Convencao
    # de tamanho de modelo".
    # results/results_aggregate/results_cross_phase.csv -- Phase 3, "alexnet_bottleneck" row
    "alexnet_bottleneck": {"top1_fp32": 44.62, "size_mb": 1.51},
    # results/results_aggregate/results_cross_phase.csv -- Phase 3, "alexnet_fire" row
    "alexnet_fire": {"top1_fp32": 43.98, "size_mb": 2.01},
    # results/phase_4_compression_and_final_architecture_training/final_comparison.csv -- FP32 row
    "alexnet_final_fire_residual": {"top1_fp32": 49.79, "size_mb": 2.71},
    # outputs/pcad/phase_9_bypass_ablation/.../alexnet_fire_bypass_summary.json
    # (top-1 also corrected: 49.03 predated the Trainer.fit() best-checkpoint fix)
    "alexnet_fire_bypass": {"top1_fp32": 50.57, "size_mb": 2.01},
}


STAGE_BLOCK_TYPES = (_AlexBottleneck, _FireModule, _FireResBlock)

# Fixed by architecture (order convs are registered in __init__), not learned.
ROLE_LABELS = {
    "_AlexBottleneck": ["reduce", "conv", "expand"],
    "_FireModule": ["squeeze", "expand1", "expand3"],
    "_FireResBlock": ["squeeze", "expand1", "expand3", "proj"],
}


def stage_from_block(block):
    convs = [m for m in block.modules() if isinstance(m, nn.Conv2d)]
    cls = type(block).__name__
    roles = ROLE_LABELS.get(cls, [])
    roles = roles[:len(convs)] if len(roles) >= len(convs) else [""] * len(convs)
    return {
        "type": cls.lstrip("_"),
        "kernels": [c.kernel_size[0] for c in convs],
        "roles": roles,
        "in_ch": convs[0].in_channels,
        "out_ch": convs[-1].out_channels,
        "residual_span": 1 if hasattr(block, "skip_add") else 0,
        "n_params": sum(p.numel() for p in block.parameters()),
        "module": block,
    }


def get_stages(model):
    """Ordered list of stages: compensation blocks, bare stem convs, and pool markers."""
    stages = []

    def walk(module):
        for child in module.children():
            if isinstance(child, STAGE_BLOCK_TYPES):
                stages.append(stage_from_block(child))
            elif isinstance(child, nn.MaxPool2d):
                k = child.kernel_size if isinstance(child.kernel_size, int) else child.kernel_size[0]
                s = child.stride if isinstance(child.stride, int) else child.stride[0]
                stages.append({"type": "pool", "pool_kernel": k, "pool_stride": s, "module": child})
            elif isinstance(child, nn.Conv2d):
                stages.append({
                    "type": "conv", "kernels": [child.kernel_size[0]], "roles": [""],
                    "in_ch": child.in_channels, "out_ch": child.out_channels,
                    "residual_span": 0, "n_params": sum(p.numel() for p in child.parameters()),
                    "module": child,
                })
            else:
                walk(child)  # containers (Sequential, the model itself) or unrelated leaves

    walk(model)
    if hasattr(model, "skip_add") and stages:
        stages[-1]["residual_span"] = 2  # AlexNetFireBypass: bypass spans fire4 -> fire5
    return stages


def annotate_output_shapes(model, stages, input_hw=INPUT_HW):
    """Attach each stage's real (C, H, W) output shape via forward hooks -- one dummy forward pass."""
    handles = []
    for stage in stages:
        def hook(_mod, _inp, out, stage=stage):
            stage["out_shape"] = tuple(out.shape[1:])
            # Trust the real forward pass over static analysis: for Fire-style blocks the
            # true out_ch is expand1+expand3 concatenated, not just the last conv's out_channels.
            stage["out_ch"] = out.shape[1]
        handles.append(stage["module"].register_forward_hook(hook))
    with torch.no_grad():
        model(torch.zeros(1, 3, *input_hw))
    for h in handles:
        h.remove()
    return stages


def head_info(model, stages):
    """Real classifier-head shape/param facts: GAP input shape + the actual Linear layer's size."""
    last = next(s for s in reversed(stages) if s["type"] != "pool")
    c, h, w = last["out_shape"]
    linear = model.classifier[-1]
    return {
        "in_ch": c, "in_hw": (h, w), "out_features": linear.out_features,
        "n_params": sum(p.numel() for p in linear.parameters()),
    }


STAGES = {name: annotate_output_shapes(models[name], get_stages(models[name])) for name in models}
HEADS = {name: head_info(models[name], STAGES[name]) for name in models}
for name, stages in STAGES.items():
    n_blocks = sum(1 for s in stages if s["type"] != "pool")
    print(f"{name}: {n_blocks} blocks, {len(stages)} total stages")

KERNEL_COLORS = {1: "#9e9e9e", 3: "#1baf7a"}  # 1x1 grey, 3x3 teal (the Winograd-friendly core)
# Block-type border color + corner tag -- lets a diagram that mixes block types (only
# AlexNetFinalFireResidual does: plain stem conv + FireResBlock stages) read at a glance which
# stage is which, without needing the caption. Bottleneck/Fire/FireBypass are uniform internally,
# so this is a no-op visual change for them (every block gets the same tag/border throughout).
BLOCK_TYPE_STYLE = {
    "conv": ("#3b6fa8", "STEM"),
    "AlexBottleneck": ("#555555", "BOTTLENECK"),
    "FireModule": ("#e8871e", "FIRE"),
    "FireResBlock": ("#e8871e", "FIRE"),
}

# Fixed box height for every stage (input/blocks/head); width grows left-to-right with the model
# instead of height growing top-to-bottom -- the whole point of the horizontal layout.
BOX_H = 3.0
HEADER_H, FOOTER_H, CHIP_PAD = 0.85, 0.22, 0.05
BLOCK_W, POOL_W, GAP = 2.05, 0.65, 0.32
INPUT_W, HEAD_W = 1.5, 1.9


def fmt_params(n):
    return f"{n / 1e3:.1f}K" if n < 1e6 else f"{n / 1e6:.2f}M"


def total_width(stages):
    """Sum of every stage's column width + gaps -- drives the per-model figsize below."""
    x = INPUT_W + GAP
    for s in stages:
        x += (POOL_W if s["type"] == "pool" else BLOCK_W) + GAP
    return x + HEAD_W


def draw_input_box(ax, x_left, x_right, input_hw):
    ax.add_patch(mpatches.FancyBboxPatch((x_left, 0), x_right - x_left, BOX_H, boxstyle="round,pad=0.02",
                 linewidth=1.1, edgecolor="#333", facecolor="#f2f2f2", zorder=2))
    cx = (x_left + x_right) / 2
    ax.text(cx, 0.16, "Input", fontsize=9, fontweight="bold", va="top", ha="center", color="#333")
    ax.text(cx, BOX_H / 2, f"3×{input_hw[0]}×{input_hw[1]}", fontsize=9.5, va="center", ha="center",
            fontweight="bold", color="#1baf7a")
    ax.text(cx, BOX_H - 0.14, "RGB · Tiny\nImageNet-200", fontsize=6, va="bottom", ha="center",
            color="#777", style="italic", linespacing=1.3)


def draw_head_box(ax, x_left, x_right, head):
    HEAD_COLOR = "#5b7fbd"
    ax.add_patch(mpatches.FancyBboxPatch((x_left, 0), x_right - x_left, BOX_H, boxstyle="round,pad=0.02",
                 linewidth=1.1, edgecolor="#333", facecolor="white", zorder=2))
    cx = (x_left + x_right) / 2
    ax.text(cx, 0.14, "Classifier Head", fontsize=8, fontweight="bold", va="top", ha="center", color="#333")

    c, (fh, fw) = head["in_ch"], head["in_hw"]
    info = f"{c}×{fh}×{fw} in · {head['out_features']} out\n{fmt_params(head['n_params'])} params"
    ax.text(cx, 0.46, info, fontsize=6.2, va="top", ha="center", color="#555", linespacing=1.4)

    chips = [("GAP", f"→ {c}×1×1"), ("Flatten", f"→ {c}"), ("Linear", f"→ {head['out_features']}")]
    chip_top, chip_bot = 0.95, BOX_H - 0.12
    chip_h = (chip_bot - chip_top) / len(chips)
    chip_left, chip_right = x_left + 0.1, x_right - 0.1
    for j, (nm, detail) in enumerate(chips):
        cy = chip_top + j * chip_h
        ax.add_patch(mpatches.Rectangle((chip_left, cy), chip_right - chip_left, chip_h - 0.06,
                     facecolor=HEAD_COLOR, edgecolor="white", lw=0.5, zorder=3))
        ax.text((chip_left + chip_right) / 2, cy + (chip_h - 0.06) / 2, f"{nm}  {detail}",
                ha="center", va="center", fontsize=6.3, fontweight="bold", color="white", zorder=4)


def draw_architecture(ax, stages, title, metrics, head, input_hw=INPUT_HW):
    x = 0.0
    input_span = (x, x + INPUT_W)
    x = input_span[1] + GAP

    xs = []  # (x_left, x_right) per stage, left-to-right data coords
    for stage in stages:
        w = POOL_W if stage["type"] == "pool" else BLOCK_W
        xs.append((x, x + w))
        x += w + GAP

    head_span = (x, x + HEAD_W)
    total_w = head_span[1]

    draw_input_box(ax, *input_span, input_hw)
    draw_head_box(ax, *head_span, head)

    for i, stage in enumerate(stages):
        x_left, x_right = xs[i]

        if stage["type"] == "pool":
            c, h, w = stage.get("out_shape", (None, None, None))
            label = f"MaxPool {stage['pool_kernel']}×{stage['pool_kernel']}\nstride {stage['pool_stride']}"
            if h is not None:
                label += f"\n→ {h}×{w}"
            ax.plot([x_left, x_right], [BOX_H / 2, BOX_H / 2], ":", color="#999", lw=1.4, zorder=1)
            ax.text((x_left + x_right) / 2, BOX_H / 2 - 0.55, label, ha="center", va="center",
                    fontsize=6, color="#888", linespacing=1.25)
            continue

        border_color, type_tag = BLOCK_TYPE_STYLE.get(stage["type"], ("#333333", ""))
        ax.add_patch(mpatches.FancyBboxPatch((x_left, 0), x_right - x_left, BOX_H, boxstyle="round,pad=0.02",
                     linewidth=2.0, edgecolor=border_color, facecolor="white", zorder=2))
        if type_tag:
            ax.text(x_right - 0.06, 0.05, type_tag, fontsize=5.3, fontweight="bold",
                    ha="right", va="top", color=border_color, zorder=4)

        c, h, w = stage.get("out_shape", (stage["out_ch"], None, None))
        info = f'{stage["in_ch"]}→{stage["out_ch"]} ch'
        if h is not None:
            info += f"  ·  {h}×{w}"
        info += f"\n{fmt_params(stage['n_params'])} params"
        ax.text((x_left + x_right) / 2, 0.16, info, fontsize=6.3, va="top", ha="center",
                color="#555", linespacing=1.5)

        kernels, roles = stage["kernels"], stage["roles"]
        n = len(kernels)
        chip_top, chip_bot = HEADER_H, BOX_H - FOOTER_H
        chip_w = (x_right - x_left - 0.2) / n
        for j, (k, role) in enumerate(zip(kernels, roles)):
            cx0 = x_left + 0.1 + j * chip_w
            ax.add_patch(mpatches.Rectangle((cx0, chip_top), chip_w - CHIP_PAD, chip_bot - chip_top,
                         facecolor=KERNEL_COLORS.get(k, "#dddddd"), edgecolor="white", lw=0.5, zorder=3))
            label = f"{role}\n{k}×{k}" if role else f"{k}×{k}"
            ax.text(cx0 + (chip_w - CHIP_PAD) / 2, (chip_top + chip_bot) / 2, label,
                    ha="center", va="center", fontsize=6.0, fontweight="bold", color="white",
                    linespacing=1.3, zorder=4)

        ax.text((x_left + x_right) / 2, BOX_H - 0.06, "BN · ReLU each", fontsize=5.6, color="#888",
                ha="center", va="bottom", style="italic")

        span = stage.get("residual_span", 0)
        if span:
            x_start = xs[i - span + 1][0]
            # rad is negative because ax.invert_yaxis() (used so the network reads left-to-right
            # with input/header text at the top of each box) flips arc3's bow direction -- positive
            # rad bows *down into* the box and hides under its fill (verified empirically).
            arc = mpatches.FancyArrowPatch((x_start, -0.08), (x_right, -0.08),
                                            connectionstyle="arc3,rad=-0.35",
                                            arrowstyle="-|>", mutation_scale=10,
                                            color="#d1495b", lw=1.5, zorder=5)
            ax.add_patch(arc)

    ax.set_xlim(-0.15, total_w + 0.15)
    ax.set_ylim(-1.0, BOX_H + 0.15)
    ax.invert_yaxis()
    ax.axis("off")

    acc, size = metrics.get("top1_fp32"), metrics.get("size_mb")
    subtitle = f"{acc:.1f}% top-1 · {size:.2f} MB (FP32)" if acc is not None else ""
    ax.set_title(f"{title}   —   {subtitle}", fontsize=12, fontweight="bold", pad=8)


def add_legend(fig, stages):
    kernel_handles = [mpatches.Patch(color=c, label=f"{k}×{k} conv") for k, c in KERNEL_COLORS.items()]
    kernel_handles.append(plt.Line2D([0], [0], color="#d1495b", lw=1.5, label="residual / bypass skip"))
    present_types = {s["type"] for s in stages if s["type"] != "pool"}
    type_handles = [
        mpatches.Patch(facecolor="white", edgecolor=color, linewidth=2, label=f"{tag.title()} block")
        for stype, (color, tag) in BLOCK_TYPE_STYLE.items() if stype in present_types
    ]
    fig.legend(handles=kernel_handles + type_handles, loc="lower center", ncol=5, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02), frameon=False)


def save_model_figure(name, title):
    stages = STAGES[name]
    fig_w = total_width(stages) + 0.3
    fig_h = BOX_H + 1.15  # + room for the title above and the legend below
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    draw_architecture(ax, stages, title, METRICS[name], HEADS[name])
    add_legend(fig, stages)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(FIGURES_DIR / f"architecture_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\u2713 architecture_{name}.png")

save_model_figure("alexnet_bottleneck", "AlexNetBottleneck")
save_model_figure("alexnet_fire", "AlexNetFire")
save_model_figure("alexnet_final_fire_residual", "AlexNetFinalFireResidual")
save_model_figure("alexnet_fire_bypass", "AlexNetFireBypass")
