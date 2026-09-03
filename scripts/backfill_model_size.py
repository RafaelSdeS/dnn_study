"""One-off: correct fp32_size_mb / fp32_gzip_mb in the per-model summary JSONs.

Until 2026-09-02, ml/reporting.py's disk_mb()/gzip_mb() measured the raw file. The FP32
side is a `{name}_best.pth` written by save_checkpoint(), which stores optimizer +
scheduler + history next to the weights -- AdamW alone keeps two momentum buffers per
parameter, so the file is ~3x the model. The INT8 side is a bare torch.save(model, ...)
and was already model-only. Every FP32-vs-INT8 size comparison was therefore
apples-to-oranges, and compression_ratio came out ~3x too high (~11.9x recorded where the
true FP32->INT8 ratio is ~4x).

disk_mb()/gzip_mb() now unwrap model_state_dict, so this script just re-measures the
already-written summaries. Where the checkpoint is gone (archived runs), it falls back to
scaling by the measured checkpoint-to-weights ratio of the same model family.

Usage:
    python -m scripts.backfill_model_size --dry-run
    python -m scripts.backfill_model_size
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.reporting import disk_mb, gzip_mb

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Where to hunt for the checkpoint a summary was built from.
SEARCH_ROOTS = ["checkpoints", "outputs", "results"]

# Fallback when the checkpoint no longer exists: a save_checkpoint() file holds the weights
# plus AdamW's two momentum buffers (exp_avg, exp_avg_sq), i.e. 3x the weights. Measured at
# 3.00x on every surviving Phase 1-4 checkpoint. Scheduler/history bytes are negligible.
ADAMW_CHECKPOINT_MULTIPLIER = 3.0


def find_checkpoint(model_name: str) -> Path | None:
    """Locate `{model_name}_best.pth` anywhere under the search roots."""
    for root in SEARCH_ROOTS:
        hits = sorted((PROJECT_ROOT / root).rglob(f"{model_name}_best.pth"))
        # Prefer a non-QAT checkpoint: qat_{name}_best.pth also ends in the same suffix.
        for h in hits:
            if not h.name.startswith("qat_"):
                return h
    return None


def registry_state_dict_mb(model_name: str) -> float | None:
    """Exact serialized weight size, by rebuilding the model from the registry. The
    architecture alone determines state_dict size, so this needs no surviving checkpoint."""
    try:
        import ml.model_registrations  # noqa: F401  (populates MODEL_REGISTRY)
        from ml.registry import MODEL_REGISTRY
        import io as _io
        import torch as _torch
        if model_name not in MODEL_REGISTRY:
            return None
        buf = _io.BytesIO()
        _torch.save(MODEL_REGISTRY[model_name]["ctor"]().state_dict(), buf)
        return buf.getbuffer().nbytes / (1024 ** 2)
    except Exception:
        return None


def summary_model_name(path: Path, data: dict) -> str:
    """Summaries were written with name=None in some phases; fall back to the filename."""
    return data.get("name") or path.name.removesuffix("_summary.json")


# Stamped into a summary once its fp32_size_mb holds weights only. Without this the /3.0
# fallback below would divide an already-corrected value again on a second run.
CORRECTED_MARKER = "fp32_size_excludes_optimizer"


def backfill(path: Path, dry_run: bool, force: bool = False) -> dict | None:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "fp32_size_mb" not in data:
        return None
    if data.get(CORRECTED_MARKER) and not force:
        return None
    old_size = data.get("fp32_size_mb")
    if not old_size:
        return None

    model_name = summary_model_name(path, data)
    ckpt = find_checkpoint(model_name)
    exact = registry_state_dict_mb(model_name)
    if ckpt is not None:
        new_size = disk_mb(ckpt)
        new_gzip = gzip_mb(ckpt) if data.get("fp32_gzip_mb") is not None else data.get("fp32_gzip_mb")
        source = f"remeasured:{ckpt.relative_to(PROJECT_ROOT)}"
    elif exact is not None:
        # Architecture fully determines state_dict size, so rebuilding from the registry is
        # exact -- strictly better than the /3.0 fallback, which assumes buffers are
        # negligible. They aren't: the true factor is (3P+B)/(P+B), and Swin's int64
        # relative-position-bias buffers put swin_pico_w8 5.9% off a flat /3.0.
        new_size = exact
        old_gzip = data.get("fp32_gzip_mb")
        # gzip depends on the trained VALUES, which a fresh model doesn't have, so this one
        # stays an estimate. Measured at -1.6% against a surviving checkpoint.
        new_gzip = old_gzip / ADAMW_CHECKPOINT_MULTIPLIER if old_gzip else old_gzip
        source = "exact:registry"
    elif not data.get(CORRECTED_MARKER):
        new_size = old_size / ADAMW_CHECKPOINT_MULTIPLIER
        old_gzip = data.get("fp32_gzip_mb")
        new_gzip = old_gzip / ADAMW_CHECKPOINT_MULTIPLIER if old_gzip else old_gzip
        source = "estimated:/3.0"
    else:
        return None  # already corrected and no better source available -- never re-divide

    if new_size is None:
        return None

    int8 = data.get("int8_size_mb")
    record = {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "model": model_name,
        "old_fp32_mb": round(old_size, 3),
        "new_fp32_mb": round(new_size, 3),
        "old_ratio": data.get("compression_ratio"),
        "new_ratio": (new_size / int8) if int8 else None,
        "source": source,
    }

    if not dry_run:
        data["fp32_size_mb"] = new_size
        data[CORRECTED_MARKER] = True
        if new_gzip is not None:
            data["fp32_gzip_mb"] = new_gzip
        if int8:
            data["compression_ratio"] = new_size / int8
        if new_gzip and new_gzip > 0:
            data["fp32_gzip_compression_ratio"] = new_size / new_gzip
        path.write_text(json.dumps(data, indent=2, default=str))
    return record


NAME_COLUMNS = ("model", "model_name", "base_model", "architecture")

# A buggy cell holds weights + AdamW's two momentum buffers (+ scheduler/history), so it sits
# at ~3x the corrected size. Guarding on that band means an already-corrected cell (ratio 1)
# and an INT8 or unrelated size column are both left alone -- the pass is idempotent and safe
# to re-run.
RATIO_LO, RATIO_HI = 2.5, 3.5


def corrected_sizes() -> dict[str, float]:
    """model name -> corrected FP32 weight size, from every already-backfilled summary JSON,
    plus any surviving checkpoint for models whose summary lacks the field (vit_tiny/deit_tiny
    were written by the Phase 8 notebook, which records size in the CSV but not the summary)."""
    sizes: dict[str, float] = {}
    for p in sorted(PROJECT_ROOT.glob("results/**/*_summary.json")) + sorted(
        PROJECT_ROOT.glob("outputs/**/*_summary.json")
    ):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and d.get("fp32_size_mb"):
            sizes.setdefault(summary_model_name(p, d), d["fp32_size_mb"])
    for root in SEARCH_ROOTS:
        for ckpt in (PROJECT_ROOT / root).rglob("*_best.pth"):
            name = ckpt.name.removesuffix("_best.pth")
            if name.startswith("qat_") or name in sizes:
                continue
            mb = disk_mb(ckpt)
            if mb:
                sizes[name] = mb
    return sizes


def patch_csvs(dry_run: bool) -> list[tuple[str, str, int]]:
    """Correct FP32 size cells in the derived CSVs.

    The CSVs are regenerated from the summary JSONs by the analysis notebooks, but re-running
    those needs the dataset + GPU, so patch the numbers in place instead. A cell is rewritten
    only when its row names a model we have a corrected size for AND the cell sits at ~3x that
    size -- so INT8 columns, already-corrected cells, and unrelated "size" columns are all left
    untouched.
    Phase 7 detection/segmentation is deliberately out of scope: its model_size_mb sits within
    4% of the separately-measured true_size_mb, so those checkpoints never carried optimizer
    state and were never affected. Phase 6 is kernel sizes, not model sizes.
    """
    import pandas as pd

    sizes = corrected_sizes()
    touched = []
    for csv in sorted(PROJECT_ROOT.glob("results/**/*.csv")):
        # Phase 6 is kernel sizes. Detection/segmentation rows reuse the classification
        # backbone names but wrap them in an SSD/DeepLab head, so a name lookup would match
        # the wrong model -- and those checkpoints were never affected anyway.
        s = str(csv)
        if any(k in s for k in ("phase_6", "phase_7", "detection", "segmentation")):
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        name_col = next((c for c in df.columns if c.lower() in NAME_COLUMNS), None)
        if name_col is None:
            continue
        changed = False
        for col in df.columns:
            if "size" not in col.lower() or "int8" in col.lower():
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            n = 0
            for idx, val in df[col].items():
                new = sizes.get(str(df.at[idx, name_col]))
                if pd.isna(val) or not new:
                    continue
                if RATIO_LO <= val / new <= RATIO_HI:
                    if not dry_run:
                        df.at[idx, col] = new
                    n += 1
                    changed = True
            if n:
                touched.append((str(csv.relative_to(PROJECT_ROOT)), col, n))
        if changed and not dry_run:
            df.to_csv(csv, index=False)
    return touched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Report the corrections without writing")
    ap.add_argument("--force", action="store_true", help="Recompute even summaries already marked corrected")
    ap.add_argument("--csvs", action="store_true", help="Also patch FP32 size cells in the derived CSVs")
    args = ap.parse_args()

    summaries = sorted(PROJECT_ROOT.glob("results/**/*_summary.json")) + sorted(
        PROJECT_ROOT.glob("outputs/**/*_summary.json")
    )
    records = [r for r in (backfill(p, args.dry_run, args.force) for p in summaries) if r]

    print(f"{'model':<38} {'old MB':>9} {'new MB':>9} {'old x':>7} {'new x':>7}  source")
    for r in records:
        ox = f"{r['old_ratio']:.2f}" if r["old_ratio"] else "-"
        nx = f"{r['new_ratio']:.2f}" if r["new_ratio"] else "-"
        print(f"{r['model']:<38} {r['old_fp32_mb']:>9.2f} {r['new_fp32_mb']:>9.2f} {ox:>7} {nx:>7}  {r['source']}")
    print(f"\n{len(records)} summaries {'would be' if args.dry_run else ''} corrected")

    if args.csvs:
        touched = patch_csvs(args.dry_run)
        total = sum(n for _, _, n in touched)
        print(f"\n{'would patch' if args.dry_run else 'patched'} {total} cells across {len(touched)} columns:")
        for path, col, n in touched:
            print(f"  {path:<70} {col:<16} {n:>3} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
