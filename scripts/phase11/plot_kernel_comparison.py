"""Phase 11 figures — kernel pattern (2x2 / 3x3 / original / misto) x head (GAP / FC) x
architecture family (AlexNet custom / AlexNetTV / VGG16), one question per PNG.

Pulls fp32/int8 top-1 + size directly from the curated Phase 11 result trees
(phase_11_kernel_size_comparison, phase_11_mixed_kernel_comparison,
phase_11_head_bn_ablation) instead of a models: list, since the FC/GAP head pairing
for the mixed-kernel models spans two experiment configs.

    python -m scripts.phase11.plot_kernel_comparison
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from ml.plotting import BLUE, RED, GREEN, AMBER, apply_report_style

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "results/figures_generated/phase_11_kernel_size_comparison"

PATTERN_COLOR = {"2x2": BLUE, "3x3": GREEN, "original": AMBER, "misto": RED}

# One row per trained model. `key` = output directory name (unique, unlike the json's
# internal model_name -- two dead/retry runs share a model_name with their surviving twin).
# `variant` pairs the three mixed-kernel patterns across their FC/GAP head twins.
# `label` is always the literal conv1->conv5 kernel sequence (VGG16's 13 conv layers are
# uniform, so its label just says so instead of spelling out 13 numbers).
MODELS = [
    dict(key="alexnet_2x2_gap", family="AlexNet compacto", head="GAP", pattern="2x2", label="2-2-2-2-2"),
    dict(key="alexnet_3x3_gap", family="AlexNet compacto", head="GAP", pattern="3x3", label="3-3-3-3-3"),
    # AlexNetMixed (models/alexnet_variants.py): conv1=3x3, conv2=2x2, conv3=3x3, conv4=2x2,
    # conv5=3x3 -- the *inverse* alternation from AlexNetTV's "mixed_alt" (2-3-2-3-2) below,
    # and a different architecture entirely (own channel widths, not AlexNetTV's).
    dict(key="alexnet_mixed_preheinit", family="AlexNet compacto", head="GAP", pattern="misto", label="3-2-3-2-3"),
    dict(key="alexnet_tv_mixed_alt_gap", family="AlexNetTV", head="GAP", pattern="misto", variant="alt", label="2-3-2-3-2"),
    dict(key="alexnet_tv_mixed_early3_gap", family="AlexNetTV", head="GAP", pattern="misto", variant="early3", label="3-3-3-2-2"),
    dict(key="alexnet_tv_mixed_early2_gap", family="AlexNetTV", head="GAP", pattern="misto", variant="early2", label="2-2-2-3-3"),
    dict(key="alexnet_tv_2x2", family="AlexNetTV", head="FC", pattern="2x2", label="2-2-2-2-2"),
    dict(key="alexnet_tv_3x3", family="AlexNetTV", head="FC", pattern="3x3", label="3-3-3-3-3"),
    dict(key="alexnet_tv_scratch", family="AlexNetTV", head="FC", pattern="original", label="11-5-3-3-3\n(original)"),
    dict(key="alexnet_tv_mixed_alt", family="AlexNetTV", head="FC", pattern="misto", variant="alt", label="2-3-2-3-2"),
    dict(key="alexnet_tv_mixed_early3", family="AlexNetTV", head="FC", pattern="misto", variant="early3", label="3-3-3-2-2"),
    dict(key="alexnet_tv_mixed_early2", family="AlexNetTV", head="FC", pattern="misto", variant="early2", label="2-2-2-3-3"),
    dict(key="vgg16", family="VGG16", head="FC", pattern="3x3", label="3x3 nativo\n(13 convs, uniforme)"),
    dict(key="vgg16_2x2", family="VGG16", head="FC", pattern="2x2", label="2x2\n(13 convs, uniforme)"),
]


def load_rows() -> dict:
    rows = {}
    for experiment in ["phase_11_kernel_size_comparison", "phase_11_mixed_kernel_comparison", "phase_11_head_bn_ablation"]:
        for summary_path in (ROOT / "outputs/pcad" / experiment).glob("*/results/*_summary.json"):
            model_dir = summary_path.parents[1].name
            rows[model_dir] = json.loads(summary_path.read_text())
    for m in MODELS:
        assert m["key"] in rows, f"missing summary.json for {m['key']}"
        m.update(rows[m["key"]])
    return rows


def savefig(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def bar_group_legend(ax, title="Padrão de kernel", loc="upper right", outside=False, patterns=None):
    patterns = patterns if patterns is not None else list(PATTERN_COLOR)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=PATTERN_COLOR[p], label=p) for p in patterns]
    kwargs = dict(bbox_to_anchor=(1.02, 1), loc="upper left") if outside else dict(loc=loc)
    ax.legend(handles=handles, title=title, fontsize=9, **kwargs)


# ── 1. Master overview: every model, accuracy vs size, log-x (3 orders of magnitude) ──
def fig_overview(models):
    # short_family: color already carries kernel pattern; keep the tag to just
    # family+head, short enough that 6-way clusters (same size, ~3pp apart) stay legible.
    short_family = {"AlexNet compacto": "AlexNet", "AlexNetTV": "TV", "VGG16": "VGG16"}
    # The 6 AlexNetTV-FC points sit within ~3pp of each other at nearly the same size --
    # no offset declutters that honestly, so only the outlier (original kernel) gets a
    # label here; the full breakdown is chart 03. Two more manual bumps for close pairs.
    dense_fc_skip = {"alexnet_tv_2x2", "alexnet_tv_3x3", "alexnet_tv_mixed_alt",
                      "alexnet_tv_mixed_early3", "alexnet_tv_mixed_early2"}
    manual_dy = {"alexnet_3x3_gap": 10, "alexnet_mixed_preheinit": -13}
    # Full "(13 convs, uniforme)"/"(original)" detail belongs to charts 03/04 -- here it
    # only needs to disambiguate from the AlexNetTV/AlexNet points sharing this cluster.
    short_label = {"vgg16": "3x3 nativo", "vgg16_2x2": "2x2", "alexnet_tv_scratch": "11-5-3-3-3 (original)"}
    fig, ax = plt.subplots(figsize=(12, 7.5))
    for i, m in enumerate(models):
        color = PATTERN_COLOR[m["pattern"]]
        fp32_size, fp32_top1 = m["fp32_size_mb"], m["fp32_top1"]
        int8_size, int8_top1 = m["int8_size_mb"], m["int8_top1"]
        ax.plot([fp32_size, int8_size], [fp32_top1, int8_top1], color="gray", lw=1, alpha=0.4, zorder=1)
        ax.scatter(fp32_size, fp32_top1, color=color, marker="o", s=120, edgecolors="white", lw=0.6, zorder=3)
        ax.scatter(int8_size, int8_top1, color=color, marker="s", s=100, edgecolors="white", lw=0.6, zorder=3)
        if m["key"] in dense_fc_skip:
            continue
        label = short_label.get(m["key"], m["label"])
        tag = f"{short_family[m['family']]} {label}".replace("\n", " ") + (" GAP" if m["head"] == "GAP" else "")
        dy = manual_dy.get(m["key"], 6 if i % 2 == 0 else -11)
        ax.annotate(tag, (fp32_size, fp32_top1), xytext=(5, dy), textcoords="offset points",
                    fontsize=7.5, color=color)
    ax.text(60, 30, "+ 5 variantes TV-FC 2x2/3x3/misto\n(quase mesmo tamanho e acurácia --\nver gráfico 03)",
            fontsize=7.5, color="#4d4d4d", ha="left")
    ax.set_xscale("log")
    ax.margins(x=0.06, y=0.14)  # keep every marker/annotation clear of the axes edges
    ax.set_xlabel("Tamanho do modelo (MB, escala log)")
    ax.set_ylabel("Top-1 (%)")
    ax.set_title(
        "Visão geral — comparação de kernel em 3 arquiteturas (○ FP32  □ INT8, \"GAP\" no rótulo = classificador final GAP)\n"
        "\"AlexNet\" = classificador final GAP (1 camada)   "
        "\"TV\"/\"VGG16\" = classificador final FC (3 camadas)",
        fontsize=12)
    bar_group_legend(ax, outside=True)
    savefig(fig, "01_overview_accuracy_vs_size.png")


# ── 2-4. Kernel pattern within each family (bar, FP32 vs INT8) ──
def fig_family_bar(models, family, filename, title):
    sub = [m for m in models if m["family"] == family]
    order = {"original": 0, "3x3": 1, "2x2": 2, "misto": 3}
    sub.sort(key=lambda m: (order[m["pattern"]], m["label"]))
    labels = [m["label"].replace("\n", " ") for m in sub]
    colors = [PATTERN_COLOR[m["pattern"]] for m in sub]
    fp32 = [m["fp32_top1"] for m in sub]
    int8 = [m["int8_top1"] for m in sub]

    fig, ax = plt.subplots(figsize=(1.8 * len(sub) + 2.5, 6.5))
    x = range(len(sub))
    width = 0.36
    ax.bar([i - width / 2 for i in x], fp32, width, color=colors, edgecolor="white", lw=0.6)
    ax.bar([i + width / 2 for i in x], int8, width, color=colors, alpha=0.45, edgecolor="white", lw=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Top-1 (%)")
    ax.set_title(title, fontsize=11.5)
    ax.margins(y=0.2)  # headroom so both legends sit inside the axes instead of widening the figure
    fp32_patch = plt.Rectangle((0, 0), 1, 1, facecolor="gray", label="FP32")
    int8_patch = plt.Rectangle((0, 0), 1, 1, facecolor="gray", alpha=0.45, label="INT8")
    precision_legend = ax.legend(handles=[fp32_patch, int8_patch], title="Precisão", loc="upper left", fontsize=9)
    ax.add_artist(precision_legend)  # kept alive so the pattern-color legend below doesn't replace it
    present_patterns = [p for p in PATTERN_COLOR if p in {m["pattern"] for m in sub}]
    bar_group_legend(ax, loc="upper right", patterns=present_patterns)
    savefig(fig, filename)


# ── 5. 2x2 vs 3x3, head-to-head across every family that has both ──
def fig_2x2_vs_3x3(models):
    families = ["AlexNet compacto", "AlexNetTV", "VGG16"]
    # Each x-tick spells out exactly which network + head this bar pair is, so the
    # chart stands on its own without needing charts 02-04 open alongside it. Positions
    # are spread out (not 0,1,2) so the 2-line descriptions have room, not overlap.
    family_desc = {
        "AlexNet compacto": "AlexNet compacto (backbone próprio,\nnão é o AlexNet clássico/torchvision)\nclassificador GAP (Global Average Pooling) -> poucos MB",
        "AlexNetTV": "AlexNetTV (torchvision, clássico)\nclassificador FC (Fully Connected,\n3 camadas, 4096 neurônios) -> ~220 MB",
        "VGG16": "VGG16 (torchvision, 13 convoluções)\nclassificador FC (Fully Connected,\n3 camadas) -> ~500 MB",
    }
    xpos = [0, 2.1, 4.2]
    fig, ax = plt.subplots(figsize=(13, 7))
    width = 0.35
    for i, pattern in enumerate(["2x2", "3x3"]):
        fp32 = []
        for fam in families:
            # AlexNetTV's 2x2/3x3 are FC-only; the custom-arch AlexNet's are GAP-only --
            # each (family, pattern) pair here has exactly one entry regardless of head.
            cands = [m for m in models if m["family"] == fam and m["pattern"] == pattern]
            fp32.append(cands[0]["fp32_top1"] if cands else float("nan"))
        offset = (i - 0.5) * width
        ax.bar([x + offset for x in xpos], fp32, width * 0.95,
               color=PATTERN_COLOR[pattern], label=pattern, edgecolor="white", lw=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels([family_desc[f] for f in families], fontsize=9.5)
    ax.set_ylabel("Top-1 FP32 (%)")
    ax.set_title("2x2 vs 3x3 uniforme (todas as camadas) — 3 arquiteturas treinadas do zero,\ncada uma com seu próprio classificador final (GAP ou FC)")
    ax.legend(title="Kernel", fontsize=9)
    savefig(fig, "05_2x2_vs_3x3_by_family.png")


# ── 6-7. Head-to-head FC vs GAP, same AlexNetTV backbone, mixed-kernel variants only ──
def fig_head_accuracy(models):
    variants = ["early3", "alt", "early2"]
    variant_label = {"early3": "3-3-3-2-2", "alt": "2-3-2-3-2", "early2": "2-2-2-3-3"}
    fc = {m["variant"]: m for m in models if m["head"] == "FC" and m.get("variant")}
    gap = {m["variant"]: m for m in models if m["head"] == "GAP" and m.get("variant")}

    fig, ax = plt.subplots(figsize=(7, 6))
    x = range(len(variants))
    width = 0.36
    ax.bar([i - width / 2 for i in x], [fc[v]["fp32_top1"] for v in variants], width,
           color=RED, edgecolor="white", lw=0.6, label="FC (Fully Connected)")
    ax.bar([i + width / 2 for i in x], [gap[v]["fp32_top1"] for v in variants], width,
           color=RED, alpha=0.45, edgecolor="white", lw=0.6, label="GAP (Global Average Pooling)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([variant_label[v] for v in variants], fontsize=10)
    ax.set_ylabel("Top-1 FP32 (%)")
    ax.set_title("Mesmo backbone AlexNetTV misto — head FC vs GAP (acurácia)")
    ax.legend(fontsize=9)
    savefig(fig, "06_head_fc_vs_gap_accuracy.png")


def fig_head_size(models):
    variants = ["early3", "alt", "early2"]
    variant_label = {"early3": "3-3-3-2-2", "alt": "2-3-2-3-2", "early2": "2-2-2-3-3"}
    fc = {m["variant"]: m for m in models if m["head"] == "FC" and m.get("variant")}
    gap = {m["variant"]: m for m in models if m["head"] == "GAP" and m.get("variant")}

    fig, ax = plt.subplots(figsize=(7, 6))
    x = range(len(variants))
    width = 0.36
    ax.bar([i - width / 2 for i in x], [fc[v]["fp32_size_mb"] for v in variants], width,
           color=RED, edgecolor="white", lw=0.6, label="FC (Fully Connected)")
    ax.bar([i + width / 2 for i in x], [gap[v]["fp32_size_mb"] for v in variants], width,
           color=RED, alpha=0.45, edgecolor="white", lw=0.6, label="GAP (Global Average Pooling)")
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels([variant_label[v] for v in variants], fontsize=10)
    ax.set_ylabel("Tamanho FP32 (MB, escala log)")
    ax.set_title("Mesmo backbone AlexNetTV misto — head FC vs GAP (tamanho)")
    ax.legend(fontsize=9)
    savefig(fig, "07_head_fc_vs_gap_size.png")


# ── 8. Quantization drop, every model, sorted ──
def fig_quant_drop(models):
    sub = sorted(models, key=lambda m: m["fp32_top1"] - m["int8_top1"])
    labels = [f"{m['family']} {m['label']}".replace('\n', ' ') + (" (GAP)" if m["head"] == "GAP" else " (FC)") for m in sub]
    drops = [m["fp32_top1"] - m["int8_top1"] for m in sub]
    colors = [PATTERN_COLOR[m["pattern"]] for m in sub]

    fig, ax = plt.subplots(figsize=(9.5, 8))
    ax.barh(range(len(sub)), drops, color=colors, edgecolor="white", lw=0.6)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Queda FP32 -> INT8 no top-1 (pp)")
    ax.set_title("Robustez à quantização (menor = melhor, ordenado do melhor pro pior)")
    bar_group_legend(ax, outside=True)
    savefig(fig, "08_quantization_drop.png")


# ── 9. Efficiency leaderboard: top-1 / MB, every model, sorted ──
def fig_efficiency(models):
    sub = sorted(models, key=lambda m: m["fp32_top1"] / m["fp32_size_mb"], reverse=True)
    labels = [f"{m['family']} {m['label']}".replace('\n', ' ') + (" (GAP)" if m["head"] == "GAP" else " (FC)") for m in sub]
    eff = [m["fp32_top1"] / m["fp32_size_mb"] for m in sub]
    colors = [PATTERN_COLOR[m["pattern"]] for m in sub]

    fig, ax = plt.subplots(figsize=(9.5, 8))
    ax.barh(range(len(sub)), eff, color=colors, edgecolor="white", lw=0.6)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Top-1 FP32 / MB (escala log, maior = melhor)")
    ax.set_title("Eficiência acurácia/tamanho — ranking (melhor no topo)")
    bar_group_legend(ax, outside=True)
    savefig(fig, "09_efficiency_ranking.png")


def main():
    apply_report_style(figsize=(9, 6))
    models = MODELS
    load_rows()

    fig_overview(models)
    fig_family_bar(models, "AlexNet compacto", "02_kernel_pattern_alexnet_gap.png",
                    "AlexNet compacta — arquitetura própria, NÃO é o AlexNet original/torchvision\n"
                    "5 convoluções estilo AlexNet + classificador GAP (Global Average Pooling, 1 camada)\n"
                    "GAP = poucos parâmetros -> modelo pequeno, ao contrário do FC (Fully Connected, 3 camadas\n"
                    "de 4096 neurônios) do AlexNet clássico -- rótulos das barras = kernel de cada camada (1ª -> 5ª)")
    fig_family_bar([m for m in models if m["family"] == "AlexNetTV" and m["head"] == "FC"],
                    "AlexNetTV", "03_kernel_pattern_alexnettv_fc.png",
                    "AlexNetTV — o AlexNet clássico (torchvision), treinado do zero\n"
                    "5 convoluções + classificador final FC original (3 camadas, 4096 neurônios)\n"
                    "rótulos das barras = kernel usado em cada uma das 5 camadas, na ordem (1ª -> 5ª)")
    fig_family_bar(models, "VGG16", "04_kernel_pattern_vgg16.png",
                    "VGG16 (torchvision), treinada do zero — rede mais profunda, 13 convoluções\n"
                    "kernel igual em todas as 13 camadas (não existe uma versão \"mista\" desta rede)")
    fig_2x2_vs_3x3(models)
    fig_head_accuracy(models)
    fig_head_size(models)
    fig_quant_drop(models)
    fig_efficiency(models)


if __name__ == "__main__":
    main()
