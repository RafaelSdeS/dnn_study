#!/usr/bin/env python3
"""
Generate architecture diagrams for Figures 6-9 in the appendix.

Draws clear visual distinction between:
- Bottleneck blocks (sequential: 1×1 → 3×3 → 1×1)
- Fire modules (parallel: 1×1 squeeze → parallel 1×1+3×3 → concat)
- Residual connections and identity bypasses
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("report/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Palette shared with generate_figures.py -- see report/palette.py.
from palette import BLUE as COLOR_FC, GREEN as COLOR_3x3, GRID as COLOR_POOL
from palette import TEXT_PRIMARY as TEXT_COLOR, TEXT_SECONDARY as LINE_COLOR

COLOR_1x1 = "#888888"      # Gray: 1×1 conv -- this diagram's own color, no report-wide counterpart

def draw_conv_box(ax, x, y, width, height, label, color, fontsize=8):
    """Draw a single convolution box."""
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.05",
                         edgecolor=LINE_COLOR, facecolor=color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center',
           fontsize=fontsize, color="white", fontweight='bold')

def draw_bottleneck_stage(ax, x, y, stage_num, in_ch, mid_ch, out_ch):
    """Draw a bottleneck block: 1×1 → 3×3 → 1×1 (sequential)."""
    w, h = 0.25, 0.4

    # 1×1 squeeze
    draw_conv_box(ax, x, y, w, h, "1×1\nsqueeze", COLOR_1x1, fontsize=7)
    ax.arrow(x + w + 0.05, y + h/2, 0.08, 0, head_width=0.08, head_length=0.03, fc=LINE_COLOR, ec=LINE_COLOR)

    # 3×3 core
    draw_conv_box(ax, x + w + 0.15, y, w, h, "3×3", COLOR_3x3, fontsize=7)
    ax.arrow(x + 2*w + 0.2, y + h/2, 0.08, 0, head_width=0.08, head_length=0.03, fc=LINE_COLOR, ec=LINE_COLOR)

    # 1×1 expand
    draw_conv_box(ax, x + 2*w + 0.3, y, w, h, "1×1\nexpand", COLOR_1x1, fontsize=7)

    # Channel info
    ax.text(x + 0.5, y - 0.15, f"{in_ch}→{mid_ch}→{mid_ch}→{out_ch}",
           ha='center', fontsize=7, color=TEXT_COLOR, style='italic')

    return x + 3*w + 0.4

def draw_fire_module(ax, x, y, stage_num, in_ch, squeeze_ch, expand_ch):
    """Draw a Fire module: 1×1 squeeze → parallel(1×1 + 3×3) → concat."""
    w, h = 0.22, 0.35

    # 1×1 squeeze (left)
    squeeze_x = x
    draw_conv_box(ax, squeeze_x, y + h/2, w, h, "1×1\nsqueeze", COLOR_1x1, fontsize=7)

    # Arrow from squeeze to parallel paths
    ax.arrow(squeeze_x + w + 0.05, y + h/2 + h/2, 0.08, 0,
            head_width=0.06, head_length=0.03, fc=LINE_COLOR, ec=LINE_COLOR)

    # Parallel 1×1 expand (top path)
    path1_x = squeeze_x + w + 0.18
    draw_conv_box(ax, path1_x, y + h + 0.15, w, h, "1×1\nexpand", COLOR_1x1, fontsize=7)

    # Parallel 3×3 expand (bottom path)
    path3_x = squeeze_x + w + 0.18
    draw_conv_box(ax, path3_x, y - 0.15, w, h, "3×3\nexpand", COLOR_3x3, fontsize=7)

    # Arrows from squeeze to both paths
    ax.annotate('', xy=(path1_x, y + h + 0.15 + h/2), xytext=(squeeze_x + w + 0.08, y + h + 0.4),
               arrowprops=dict(arrowstyle='->', lw=1, color=LINE_COLOR))
    ax.annotate('', xy=(path3_x, y + h/2), xytext=(squeeze_x + w + 0.08, y + h/2),
               arrowprops=dict(arrowstyle='->', lw=1, color=LINE_COLOR))

    # Concatenate box (right)
    concat_x = path1_x + w + 0.15
    concat_y = y + 0.15
    concat_box = FancyBboxPatch((concat_x, concat_y), 0.25, h*1.5, boxstyle="round,pad=0.05",
                               edgecolor=LINE_COLOR, facecolor="#f0f0f0", linewidth=2, linestyle='--')
    ax.add_patch(concat_box)
    ax.text(concat_x + 0.125, concat_y + h*0.75, "concat.", ha='center', va='center',
           fontsize=7, color=TEXT_COLOR, style='italic')

    # Arrows to concat
    ax.annotate('', xy=(concat_x, concat_y + h*0.75), xytext=(path1_x + w + 0.05, y + h + 0.15 + h/2),
               arrowprops=dict(arrowstyle='->', lw=1, color=LINE_COLOR))
    ax.annotate('', xy=(concat_x, concat_y + h*0.75), xytext=(path3_x + w + 0.05, y + h/2),
               arrowprops=dict(arrowstyle='->', lw=1, color=LINE_COLOR))

    # Output channels
    out_ch = expand_ch * 2
    ax.text(x + 1.0, y - 0.35, f"{in_ch} → {squeeze_ch} ⟹ {expand_ch}+{expand_ch} = {out_ch}",
           ha='center', fontsize=7, color=TEXT_COLOR, style='italic')

    return concat_x + 0.35

def draw_fire_residual_block(ax, x, y, label, in_ch, squeeze_ch, expand_ch, has_residual=True):
    """Draw Fire module + residual skip connection."""
    w, h = 0.20, 0.32

    # Fire squeeze
    squeeze_x = x
    draw_conv_box(ax, squeeze_x, y + h/2, w, h, "1×1\nS", COLOR_1x1, fontsize=6)

    # Arrow to parallel
    ax.arrow(squeeze_x + w + 0.03, y + h/2 + h/2, 0.06, 0,
            head_width=0.05, head_length=0.02, fc=LINE_COLOR, ec=LINE_COLOR)

    # Parallel paths
    path1_x = squeeze_x + w + 0.12
    draw_conv_box(ax, path1_x, y + h + 0.1, w, h, "1×1", COLOR_1x1, fontsize=6)
    draw_conv_box(ax, path1_x, y - 0.12, w, h, "3×3", COLOR_3x3, fontsize=6)

    # Concatenate (small dashed box)
    concat_x = path1_x + w + 0.08
    concat_box = FancyBboxPatch((concat_x, y - 0.15), 0.18, h*1.4,
                               edgecolor=LINE_COLOR, facecolor="#f5f5f5", linewidth=1.5, linestyle='--')
    ax.add_patch(concat_box)
    ax.text(concat_x + 0.09, y + h/2, "c", ha='center', va='center', fontsize=6)

    # Arrows to concat
    ax.annotate('', xy=(concat_x + 0.02, y + h*0.65), xytext=(path1_x + w + 0.02, y + h + 0.1 + h/2),
               arrowprops=dict(arrowstyle='->', lw=0.8, color=LINE_COLOR))
    ax.annotate('', xy=(concat_x + 0.02, y + h/2), xytext=(path1_x + w + 0.02, y + h/2),
               arrowprops=dict(arrowstyle='->', lw=0.8, color=LINE_COLOR))

    # Output
    out_x = concat_x + 0.25

    if has_residual:
        # Draw residual skip connection (curved bracket)
        skip_y_top = y + h*1.2
        skip_y_bot = y - 0.25
        arc = mpatches.FancyBboxPatch((out_x + 0.05, skip_y_bot), 0.15, skip_y_top - skip_y_bot,
                                     boxstyle="round,pad=0.02", fill=False,
                                     edgecolor="#d946ef", linewidth=2.5, linestyle='-')
        ax.add_patch(arc)
        ax.text(out_x + 0.24, y + h/2, "⊕", fontsize=12, color="#d946ef", fontweight='bold')

    return out_x + 0.4

# ============================================================================
# Figure 6: AlexNetBottleneck
# ============================================================================
fig, ax = plt.subplots(figsize=(64, 20), facecolor='white')
ax.set_xlim(-0.8, 32)
ax.set_ylim(-3, 6)
ax.axis('off')

y_start = 2.5
ax.text(0.3, y_start + 1.2, "Entrada: 3 × 64 × 64", fontsize=13, fontweight='bold', color=TEXT_COLOR)

# Title
ax.text(8, y_start + 1.6, "AlexNetBottleneck (Eixo 1)\n44,62% top-1 · 1,69 MB (FP32)",
       ha='center', fontsize=14, fontweight='bold', color=TEXT_COLOR)

# Legend
legend_y = y_start - 0.1
ax.add_patch(FancyBboxPatch((8.5, legend_y - 0.15), 0.15, 0.15, boxstyle="round,pad=0.01",
                           edgecolor=LINE_COLOR, facecolor=COLOR_1x1, linewidth=1))
ax.text(8.75, legend_y + 0.15, "conv 1×1", fontsize=8, color=TEXT_COLOR)

ax.add_patch(FancyBboxPatch((8.5, legend_y - 0.45), 0.15, 0.15, boxstyle="round,pad=0.01",
                           edgecolor=LINE_COLOR, facecolor=COLOR_3x3, linewidth=1))
ax.text(8.75, legend_y - 0.25, "conv 3×3", fontsize=8, color=TEXT_COLOR)

# Draw 5 bottleneck stages
x = 0.3
stages = [
    (3, 32, 64),   # Stage 1
    (64, 48, 192), # Stage 2
    (192, 96, 384),# Stage 3
    (384, 64, 256),# Stage 4
    (256, 64, 256),# Stage 5
]

for i, (in_ch, mid_ch, out_ch) in enumerate(stages):
    x = draw_bottleneck_stage(ax, x, y_start, i+1, in_ch, mid_ch, out_ch)
    if i < len(stages) - 1:
        ax.arrow(x - 0.15, y_start + 0.2, 0, -0.4, head_width=0.08, head_length=0.05,
                fc=COLOR_POOL, ec=COLOR_POOL, linewidth=2)
        ax.text(x - 0.15, y_start - 0.35, "MaxPool2d", fontsize=7, ha='center', color=TEXT_COLOR)

# GAP + Classifier
ax.arrow(x - 0.15, y_start + 0.2, 0, -0.4, head_width=0.08, head_length=0.05,
        fc=COLOR_POOL, ec=COLOR_POOL, linewidth=2)
ax.text(x - 0.15, y_start - 0.35, "GAP", fontsize=7, ha='center', color=TEXT_COLOR)

# Classifier head
fc_x = x
draw_conv_box(ax, fc_x, y_start, 0.35, 0.4, "Linear\n256→200", COLOR_FC, fontsize=7)
ax.text(fc_x + 0.175, y_start - 0.25, "Classificador", fontsize=7, ha='center', color=TEXT_COLOR)

# Key insight
ax.text(6, 0.3, "Bottleneck sequencial: 1×1 squeeze → núcleo 3×3 → 1×1 expand (por estágio)",
       ha='center', fontsize=9, color=TEXT_COLOR, bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "architecture_alexnet_bottleneck.png", dpi=300, bbox_inches='tight', facecolor='white')
print("✓ architecture_alexnet_bottleneck.png")
plt.close()

# ============================================================================
# Figure 7: AlexNetFire
# ============================================================================
fig, ax = plt.subplots(figsize=(42, 16), facecolor='white')
ax.set_xlim(-1, 35)
ax.set_ylim(-3.5, 6)
ax.axis('off')

y_start = 2.0
ax.text(0.2, y_start + 0.8, "Entrada: 3 × 64 × 64", fontsize=10, fontweight='bold', color=TEXT_COLOR)

# Title
ax.text(7, y_start + 1.2, "AlexNetFire (Eixo 1)\n44,0% top-1 · 5,99 MB (FP32)",
       ha='center', fontsize=11, fontweight='bold', color=TEXT_COLOR)

# Legend
legend_y = y_start - 0.1
ax.add_patch(FancyBboxPatch((11, legend_y - 0.15), 0.15, 0.15, boxstyle="round,pad=0.01",
                           edgecolor=LINE_COLOR, facecolor=COLOR_1x1, linewidth=1))
ax.text(11.25, legend_y + 0.15, "conv 1×1", fontsize=8, color=TEXT_COLOR)

ax.add_patch(FancyBboxPatch((11, legend_y - 0.45), 0.15, 0.15, boxstyle="round,pad=0.01",
                           edgecolor=LINE_COLOR, facecolor=COLOR_3x3, linewidth=1))
ax.text(11.25, legend_y - 0.25, "conv 3×3", fontsize=8, color=TEXT_COLOR)

ax.add_patch(FancyBboxPatch((11, legend_y - 0.75), 0.15, 0.15, boxstyle="round,pad=0.01",
                           edgecolor=LINE_COLOR, facecolor="#f0f0f0", linewidth=1, linestyle='--'))
ax.text(11.25, legend_y - 0.55, "concat.", fontsize=8, color=TEXT_COLOR)

# Draw 5 Fire modules
x = 0.3
fire_configs = [
    (3, 16, 32),      # Fire 1: squeeze 16, expand 32 each → 64
    (64, 48, 96),     # Fire 2: squeeze 48, expand 96 each → 192
    (192, 96, 192),   # Fire 3: squeeze 96, expand 192 each → 384
    (384, 64, 128),   # Fire 4: squeeze 64, expand 128 each → 256
    (256, 64, 128),   # Fire 5: squeeze 64, expand 128 each → 256
]

for i, (in_ch, squeeze_ch, expand_ch) in enumerate(fire_configs):
    x = draw_fire_module(ax, x, y_start, i+1, in_ch, squeeze_ch, expand_ch)
    if i < len(fire_configs) - 1:
        ax.arrow(x - 0.4, y_start + 0.2, 0, -0.4, head_width=0.08, head_length=0.05,
                fc=COLOR_POOL, ec=COLOR_POOL, linewidth=2)
        ax.text(x - 0.4, y_start - 0.35, "MaxPool2d", fontsize=7, ha='center', color=TEXT_COLOR)

# GAP + Classifier
ax.arrow(x - 0.4, y_start + 0.2, 0, -0.4, head_width=0.08, head_length=0.05,
        fc=COLOR_POOL, ec=COLOR_POOL, linewidth=2)
ax.text(x - 0.4, y_start - 0.35, "GAP", fontsize=7, ha='center', color=TEXT_COLOR)

fc_x = x
draw_conv_box(ax, fc_x, y_start, 0.35, 0.4, "Linear\n256→200", COLOR_FC, fontsize=7)
ax.text(fc_x + 0.175, y_start - 0.25, "Classificador", fontsize=7, ha='center', color=TEXT_COLOR)

# Key insight
ax.text(7, -1.8, "Fire paralelo: 1×1 squeeze → caminhos paralelos (1×1 + 3×3) → concatenação",
       ha='center', fontsize=9, color=TEXT_COLOR, bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "architecture_alexnet_fire.png", dpi=150, bbox_inches='tight', facecolor='white')
print("✓ architecture_alexnet_fire.png")
plt.close()

# ============================================================================
# Figure 8: AlexNetFinalFireResidual
# ============================================================================
fig, ax = plt.subplots(figsize=(64, 20), facecolor='white')
ax.set_xlim(-0.8, 32)
ax.set_ylim(-3.5, 6)
ax.axis('off')

y_start = 2.0
ax.text(0.2, y_start + 0.8, "Entrada: 3 × 64 × 64", fontsize=10, fontweight='bold', color=TEXT_COLOR)

# Title
ax.text(7, y_start + 1.2, "AlexNetFinalFireResidual (Eixo 2)\n49,89% top-1 · 8,09 MB (FP32)",
       ha='center', fontsize=11, fontweight='bold', color=TEXT_COLOR)

# Legend
legend_y = y_start - 0.1
ax.add_patch(FancyBboxPatch((10.5, legend_y - 0.15), 0.15, 0.15, boxstyle="round,pad=0.01",
                           edgecolor=LINE_COLOR, facecolor=COLOR_1x1, linewidth=1))
ax.text(10.75, legend_y + 0.15, "conv 1×1", fontsize=8, color=TEXT_COLOR)

ax.add_patch(FancyBboxPatch((10.5, legend_y - 0.45), 0.15, 0.15, boxstyle="round,pad=0.01",
                           edgecolor=LINE_COLOR, facecolor=COLOR_3x3, linewidth=1))
ax.text(10.75, legend_y - 0.25, "conv 3×3", fontsize=8, color=TEXT_COLOR)

ax.add_patch(mpatches.Rectangle((10.5, legend_y - 0.75), 0.15, 0.15,
                               edgecolor="#d946ef", facecolor="none", linewidth=2))
ax.text(10.75, legend_y - 0.55, "residual", fontsize=8, color=TEXT_COLOR)

# Stem: Conv 3×3 stride-2
x = 0.3
draw_conv_box(ax, x, y_start, 0.3, 0.4, "Conv2d\n3×3 s2", COLOR_3x3, fontsize=7)
ax.arrow(x + 0.3 + 0.05, y_start + 0.2, 0.08, 0, head_width=0.08, head_length=0.03, fc=LINE_COLOR, ec=LINE_COLOR)

# Fire + Residual blocks (4 blocks instead of 5)
x += 0.5
fire_res_configs = [
    (64, 48, 96, 192),      # Input 64 → squeeze 48 → expand 96+96=192
    (192, 96, 192, 384),    # Input 192 → squeeze 96 → expand 192+192=384
    (384, 64, 128, 256),    # Input 384 → squeeze 64 → expand 128+128=256
    (256, 64, 128, 256),    # Input 256 → squeeze 64 → expand 128+128=256 (true residual)
]

for i, (in_ch, squeeze_ch, expand_ch, out_ch) in enumerate(fire_res_configs):
    x = draw_fire_residual_block(ax, x, y_start, f"FR{i+1}", in_ch, squeeze_ch, expand_ch, has_residual=True)
    if i < len(fire_res_configs) - 1:
        ax.arrow(x - 0.5, y_start + 0.2, 0, -0.4, head_width=0.08, head_length=0.05,
                fc=COLOR_POOL, ec=COLOR_POOL, linewidth=2)
        ax.text(x - 0.5, y_start - 0.35, "MaxPool2d", fontsize=7, ha='center', color=TEXT_COLOR)

# GAP + Classifier
ax.arrow(x - 0.5, y_start + 0.2, 0, -0.4, head_width=0.08, head_length=0.05,
        fc=COLOR_POOL, ec=COLOR_POOL, linewidth=2)
ax.text(x - 0.5, y_start - 0.35, "GAP", fontsize=7, ha='center', color=TEXT_COLOR)

fc_x = x
draw_conv_box(ax, fc_x, y_start, 0.35, 0.4, "Linear\n256→200", COLOR_FC, fontsize=7)
ax.text(fc_x + 0.175, y_start - 0.25, "Classificador", fontsize=7, ha='center', color=TEXT_COLOR)

# Key insight
ax.text(7, -1.8, "Fire + Residual: módulos Fire com atalho identidade (⊕) em cada bloco",
       ha='center', fontsize=9, color=TEXT_COLOR, bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "architecture_alexnet_final_fire_residual.png", dpi=150, bbox_inches='tight', facecolor='white')
print("✓ architecture_alexnet_final_fire_residual.png")
plt.close()

# ============================================================================
# Figure 9: AlexNetFireBypass
# ============================================================================
fig, ax = plt.subplots(figsize=(42, 16), facecolor='white')
ax.set_xlim(-1, 35)
ax.set_ylim(-3.5, 6)
ax.axis('off')

y_start = 2.0
ax.text(0.2, y_start + 0.8, "Entrada: 3 × 64 × 64", fontsize=10, fontweight='bold', color=TEXT_COLOR)

# Title
ax.text(7, y_start + 1.2, "AlexNetFireBypass (Eixo 4)\n49,0% top-1 · 5,99 MB (FP32)",
       ha='center', fontsize=11, fontweight='bold', color=TEXT_COLOR)

# Legend
legend_y = y_start - 0.1
ax.add_patch(FancyBboxPatch((11, legend_y - 0.15), 0.15, 0.15, boxstyle="round,pad=0.01",
                           edgecolor=LINE_COLOR, facecolor=COLOR_1x1, linewidth=1))
ax.text(11.25, legend_y + 0.15, "conv 1×1", fontsize=8, color=TEXT_COLOR)

ax.add_patch(FancyBboxPatch((11, legend_y - 0.45), 0.15, 0.15, boxstyle="round,pad=0.01",
                           edgecolor=LINE_COLOR, facecolor=COLOR_3x3, linewidth=1))
ax.text(11.25, legend_y - 0.25, "conv 3×3", fontsize=8, color=TEXT_COLOR)

ax.add_patch(mpatches.Rectangle((11, legend_y - 0.75), 0.15, 0.15,
                               edgecolor="#16a34a", facecolor="none", linewidth=2, linestyle=':'))
ax.text(11.25, legend_y - 0.55, "atalho", fontsize=8, color=TEXT_COLOR)

# Draw stem (3 Fire modules)
x = 0.3
stem_configs = [
    (3, 16, 32),      # Fire 1
    (64, 48, 96),     # Fire 2
    (192, 96, 192),   # Fire 3
]

for i, (in_ch, squeeze_ch, expand_ch) in enumerate(stem_configs):
    x = draw_fire_module(ax, x, y_start, f"F{i+1}", in_ch, squeeze_ch, expand_ch)
    if i < len(stem_configs) - 1:
        ax.arrow(x - 0.4, y_start + 0.2, 0, -0.4, head_width=0.08, head_length=0.05,
                fc=COLOR_POOL, ec=COLOR_POOL, linewidth=2)
        ax.text(x - 0.4, y_start - 0.35, "MaxPool2d", fontsize=7, ha='center', color=TEXT_COLOR)

# Fire 4
fire4_x = x
x = draw_fire_module(ax, x, y_start, "F4", 384, 64, 128)

# Fire 5 (with bypass to fire4)
fire5_x = x
x = draw_fire_module(ax, x, y_start, "F5", 256, 64, 128)

# Draw IDENTITY BYPASS between fire4 and fire5 only
bypass_y_top = y_start + 0.6
bypass_y_bot = y_start - 0.6
ax.annotate('', xy=(fire5_x + 0.2, bypass_y_bot), xytext=(fire4_x + 0.6, bypass_y_top),
           arrowprops=dict(arrowstyle='->', lw=2.5, color="#16a34a", linestyle=':', connectionstyle="arc3,rad=0.5"))
ax.text((fire4_x + fire5_x + 0.9) / 2, y_start + 1.0, "atalho identidade", fontsize=8, color="#16a34a",
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# GAP + Classifier
ax.arrow(x - 0.4, y_start + 0.2, 0, -0.4, head_width=0.08, head_length=0.05,
        fc=COLOR_POOL, ec=COLOR_POOL, linewidth=2)
ax.text(x - 0.4, y_start - 0.35, "GAP", fontsize=7, ha='center', color=TEXT_COLOR)

fc_x = x
draw_conv_box(ax, fc_x, y_start, 0.35, 0.4, "Linear\n256→200", COLOR_FC, fontsize=7)
ax.text(fc_x + 0.175, y_start - 0.25, "Classificador", fontsize=7, ha='center', color=TEXT_COLOR)

# Key insight
ax.text(7, -1.8, "Fire + atalho único: idêntico ao Fire (stem + 5 módulos), MAS com atalho identidade fire4→fire5",
       ha='center', fontsize=9, color=TEXT_COLOR, bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "architecture_alexnet_fire_bypass.png", dpi=150, bbox_inches='tight', facecolor='white')
print("✓ architecture_alexnet_fire_bypass.png")
plt.close()

print("\n✅ All 4 architecture diagrams generated successfully!")
print("✅ Clear visual distinction between Bottleneck (sequential) and Fire (parallel) now visible")
