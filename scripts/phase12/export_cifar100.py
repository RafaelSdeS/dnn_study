"""Phase 12: CIFAR-100 in Tiny ImageNet's folder layout, so ml/data.py loads it unchanged
(docs/logs/PHASE11_LOG.md "Controls and robustness runs"):

    python -m scripts.phase12.export_cifar100 OUT_DIR        # downloads CIFAR-100 (torchvision) into OUT_DIR/_raw first
    PCAD_DATASET_ROOT=OUT_DIR python -m scripts.train --experiment phase_12_cifar100 --runtime pcad ...

OUT_DIR/train/<class>/images/<i>.png (the 50 000 training images; the 90/10 hold-out is made at load time) and
OUT_DIR/val/images/<i>.png + val_annotations.txt (the 10 000 official test images, the run's test set -- Tiny ImageNet's val
split is the same size, ml/data.py:TINY_IMAGENET_TEST_SIZE). rsync OUT_DIR (minus _raw) to PCAD and nyx.
"""
import sys
from pathlib import Path

from torchvision.datasets import CIFAR100


def export(out: Path) -> None:
    rows = []
    for train in (True, False):
        ds = CIFAR100(out / "_raw", train=train, download=True)
        for i, (img, y) in enumerate(ds):
            d = out / "train" / ds.classes[y] / "images" if train else out / "val" / "images"
            d.mkdir(parents=True, exist_ok=True)
            img.save(d / f"{i}.png")
            if not train:
                rows.append(f"{i}.png\t{ds.classes[y]}")
    (out / "val" / "val_annotations.txt").write_text("\n".join(rows) + "\n")
    assert len(list(out.glob("train/*/images/*.png"))) == 50_000 and len(rows) == 10_000


if __name__ == "__main__":
    export(Path(sys.argv[1]).expanduser())
