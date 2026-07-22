"""Gzip-compress existing checkpoint files that predate ml.checkpoint.compress_checkpoint().

New training runs write a `.pth.gz` alongside every finished FP32/INT8 checkpoint
automatically (see ml/checkpoint.py, wired into scripts/train.py and the training
notebooks). This script backfills that for checkpoints saved before that wiring existed —
run it once after pulling old checkpoint files onto a machine that doesn't have them yet.

Skips `*_resume.pth` (transient per-epoch training state, not a deployable artifact) and
any `.pth` that already has a `.pth.gz` sibling (idempotent — safe to re-run).

Usage:
    python -m scripts.backfill_gzip
    python -m scripts.backfill_gzip --root checkpoints --root outputs/pcad/checkpoints
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ml import compress_checkpoint, disk_mb, gzip_mb

DEFAULT_ROOTS = ["checkpoints", "outputs/pcad/checkpoints"]


def backfill(roots: list[str]) -> None:
    total_before = total_after = compressed = skipped = 0

    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f"Skipping {root} (doesn't exist here)")
            continue

        for pth in sorted(root_path.rglob("*.pth")):
            if pth.name.endswith("_resume.pth"):
                continue
            gz_path = pth.with_suffix(pth.suffix + ".gz")
            if gz_path.exists():
                skipped += 1
                continue

            before_mb = disk_mb(pth)
            compress_checkpoint(pth)
            after_mb = gzip_mb(pth)
            total_before += before_mb or 0
            total_after += after_mb or 0
            compressed += 1
            print(f"{pth}  {before_mb:.2f}MB -> {after_mb:.2f}MB")

    print(f"\nCompressed {compressed} checkpoint(s), skipped {skipped} (already had .gz).")
    if compressed:
        print(f"Total: {total_before:.1f}MB -> {total_after:.1f}MB "
              f"({total_before / total_after:.1f}x)" if total_after else "")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", action="append", dest="roots",
        help=f"Directory to scan recursively for *.pth (repeatable). Default: {DEFAULT_ROOTS}",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    backfill(args.roots or DEFAULT_ROOTS)
