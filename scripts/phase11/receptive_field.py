"""Phase 11 figure 22 -- the effective receptive field (Luo et al., NeurIPS 2016; RepLKNet's Fig. 1, Ding et al., CVPR
2022) of each kernel: how much each input pixel moves the centre unit of the last conv layer, |d unit / d pixel| averaged
over N_IMAGES test images. Nets at the 64px layout with GAP + BN (the main factorial's best-trained column, where the
stacked cells also live), plus the AlexNet original (baseline) at its own layout.

    22_effective_receptive_field.png

Needs each run's FP32 checkpoint (<model>_best.pth), which only PCAD keeps (gitignored): a panel without it says so and
the script prints the rsync that fetches it. The test images are Tiny ImageNet's official val split from the KaggleHub
cache (TEST_IMAGES); nothing is downloaded.

    python -m scripts.phase11.receptive_field
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch import nn

from ml.plotting import NEUTRAL, TEXT_PRIMARY, apply_report_style
from scripts.phase11.analyze_geometry import BASE_KEY, BASE_LABEL, ROOT, savefig
from scripts.phase11.design_figures import KCOLOR, KLABEL, frame, seed42

KERNELS = ["k11-5-3", "k3x3", "k3x3stacked", "k2x2", "k2x2stacked", "kalt3-2"]
N_IMAGES = 200
TEST_IMAGES = Path.home() / ".cache/kagglehub/datasets/akash2sharma/tiny-imagenet/versions/1/tiny-imagenet-200/val/images"
PCAD = "rsdsouza@gppd-hpc.inf.ufrgs.br:dnn_study/./"


def cell(kernels):
    """The design's AlexNet cell with these kernels at the 64px layout, GAP head, BN, from scratch."""
    from ml.model_registrations import CELL_FACTORS

    want = dict(family="alexnet", kernels=kernels, stride=2, pool_kernel=2, pool_count=2, head="gap", bn=True, pretrained=False)
    return next(k for k, f in CELL_FACTORS.items() if all(f[c] == v for c, v in want.items()))


def body(model):
    """The conv stack up to its last activation: features without its trailing pools (GAP / the original's 3x3 max-pool)."""
    layers = list(model.features)
    while isinstance(layers[-1], (nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
        layers.pop()
    return nn.Sequential(*layers)


def effective_rf(model, x):
    """|d centre unit / d input| summed over channels, averaged over the images, normalised to sum 1 (H x W)."""
    x = x.clone().requires_grad_(True)
    y = body(model)(x)
    y[:, :, y.shape[2] // 2, y.shape[3] // 2].sum().backward()
    g = x.grad.abs().sum(1).mean(0)
    return (g / g.sum()).numpy()


def images():
    from ml.data import _val_transform  # the test loader's own transform: resize, centre crop, ImageNet normalisation

    files = sorted(TEST_IMAGES.glob("*.JPEG"))[:N_IMAGES]
    t = _val_transform(64)
    return torch.stack([t(Image.open(f).convert("RGB")) for f in files]) if files else None


def panels(df):
    """(label, key, colour, run or None) per panel: the baseline, then KERNELS' cells (None = not trained yet)."""
    runs = seed42(df).set_index("key")
    # the layout on every label: the baseline has 11-5-3-3-3 too, its wider field is its stride-4 + 3-pool geometry
    rows = ([(f"{BASE_LABEL}: 11-5-3-3-3\nlayout original (conv1 s4, 3 pools)", BASE_KEY, TEXT_PRIMARY)]
            + [(f"{KLABEL[k]}\nlayout 64px (conv1 s2, 2 pools)", cell(k), KCOLOR[k]) for k in KERNELS])
    return [(label, key, c, runs.loc[key] if key in runs.index else None) for label, key, c in rows]


def main():
    import ml.model_registrations  # noqa: F401
    from ml.quantization import load_best_model
    from ml.registry import MODEL_REGISTRY

    apply_report_style()
    ps = panels(frame())
    ckpt = {key: r.results.parent / "checkpoints" / f"{key}_best.pth" for _, key, _, r in ps if r is not None}
    missing = [p for p in ckpt.values() if not p.exists()]
    for p in missing:
        print(f"22: no checkpoint -- rsync -aR {PCAD}{p.relative_to(ROOT) if p.is_relative_to(ROOT) else p} .")
    x = images() if len(missing) < len(ckpt) else None
    fig, axes = plt.subplots(1, len(ps), figsize=(2.6 * len(ps), 4.2), layout="constrained")
    norm = Normalize(0, 1)
    for ax, (label, key, c, r) in zip(axes, ps):
        ax.axis("off")
        why = "pendente" if r is None else "checkpoint só no PCAD" if not ckpt[key].exists() else "sem imagens de teste" if x is None else ""
        if why:
            ax.text(0.5, 0.5, why, transform=ax.transAxes, ha="center", va="center", color=NEUTRAL, fontsize=10)
            ax.set_title(label, fontsize=10, color=c)
            continue
        model = load_best_model(key, MODEL_REGISTRY[key]["ctor"], ckpt[key].parent, torch.device("cpu"))
        g = effective_rf(model, x)
        ax.imshow(g / g.max(), norm=norm, cmap=LinearSegmentedColormap.from_list(key, ["white", c]))
        top = np.sort(g.ravel())[::-1]
        half = (np.cumsum(top) < 0.5).sum() + 1  # fewest pixels holding half the influence: outlined
        smooth = gaussian_filter(g, 1.5)  # ponytail: the outline only -- the raw map has an 8-px stride grid
        ax.contour(smooth, levels=[np.sort(smooth.ravel())[::-1][half - 1]], colors=TEXT_PRIMARY, linewidths=1.2)
        ax.set_title(f"{label}\nmetade da influência em\n{100 * half / g.size:.1f}% da imagem", fontsize=10, color=c)
        print(f"22 {key}: half the influence in {half} px ({100 * half / g.size:.1f}%)")
    fig.colorbar(ScalarMappable(norm, LinearSegmentedColormap.from_list("rf", ["white", TEXT_PRIMARY])), ax=axes, shrink=0.7,
                 label="influência do pixel (1 = o mais influente)")
    fig.suptitle("Que região da imagem cada rede usa? Campo receptivo efetivo do neurônio central da última conv (Luo et al. 2016):\n"
                 f"quanto cada pixel move esse neurônio, média de {N_IMAGES} imagens de teste 64×64; contorno = a menor região com "
                 "metade da influência\n(o campo receptivo teórico cobre a imagem inteira em todas) · cabeça GAP + BN, menos o AlexNet original "
                 "(FC + Dropout, sem BN) · seed 42", fontsize=11)
    savefig(fig, "22_effective_receptive_field.png")


if __name__ == "__main__":
    main()
