#!/usr/bin/env python
"""Rebuild results/results_aggregate/{results,model_details}_cross_phase.csv from the curated
per-model summary JSONs under results/phase_*/ (plus Phase 8's phase8_comparison.csv, which
already carries make_run_summary's field names -- Phase 8's per-model JSONs live under
outputs/pcad/, not results/, so its curated rollup is this CSV instead).

Both cross-phase CSVs were hand-maintained and went stale after Phase 3 while Phases 4/8/9
produced real comparison data with nothing to catch the drift. Every per-model summary JSON
(ml/reporting.py::make_run_summary) already holds a superset of both output schemas' fields, so
one JSON = one row in each CSV -- no joining FP32/INT8 rows, no re-deriving anything not already
in the file. Idempotent: rerun any time a new phase's results land in results/.

Usage: python -m scripts.build_cross_phase_results [--out-dir results/results_aggregate] [--dry-run]
"""
import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("results")

# (phase label, results/ subdirectory) for phases with per-model *_summary.json files.
PHASE_DIRS = [
    ("Phase 1 — Baselines", "phase_1_baseline"),
    ("Phase 2 — AlexNet Variants", "phase_2_kernel_restriction"),
    ("Phase 3 — Compensation", "phase_3_compensation_and_hybrids"),
    ("Phase 4 — Final Architecture", "phase_4_compression_and_final_architecture"),
    ("Phase 9 — PCAD Bypass Ablation", "phase_9_bypass_ablation"),
    ("Phase 11 — Kernel Size Comparison", "phase_11_kernel_size_comparison"),
]

PHASE_8_LABEL = "Phase 8 — Efficient ViT / Hybrid-Attention"
PHASE_8_CSV = RESULTS_DIR / "phase_8_efficient_vit" / "phase8_comparison.csv"

RESULTS_CROSS_PHASE_FIELDS = [
    "phase", "base_model", "top1_%_FP32", "top1_%_INT8", "qat_top1_drop_%",
    "top5_%_FP32", "top5_%_INT8", "qat_top5_drop_%", "loss_FP32", "loss_INT8",
    "params_M_FP32", "size_MB_FP32", "size_MB_INT8",
]

MODEL_DETAILS_FIELDS = [
    "phase", "model_name", "avg_epoch_time_s", "best_val_loss", "best_val_top1",
    "best_val_top5", "compression_ratio", "epochs", "final_train_loss", "final_val_top1",
    "final_val_top5", "flops", "fp32_latency_ms_per_image", "fp32_loss", "fp32_size_mb",
    "fp32_throughput_img_per_s", "fp32_top1", "fp32_top1_top5_gap", "fp32_top5",
    "int8_latency_ms_per_image", "int8_loss", "int8_size_mb", "int8_throughput_img_per_s",
    "int8_top1", "int8_top1_top5_gap", "int8_top5", "macs", "mode",
    "param_efficiency_top1_per_m", "params_m", "peak_gpu_mem_mb", "quantization_drop_top1",
    "total_training_time_s",
]

REQUIRED_FIELDS = ("model_name", "fp32_top1")


def _missing_required(summary: dict[str, Any]) -> bool:
    return any(summary.get(f) in (None, "") for f in REQUIRED_FIELDS)


def _load_phase_summaries(phase_dir: Path):
    for path in sorted(phase_dir.glob("*_summary.json")):
        if path.name == "experiment_summary.json" or path.name.endswith("_compression_summary.json"):
            continue  # different schema: experiment-level rollup / compression-sweep row, no model_name+fp32_top1
        try:
            summary = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            warnings.warn(f"{path}: unreadable ({e}), skipped")
            continue
        if _missing_required(summary):
            warnings.warn(f"{path}: missing {REQUIRED_FIELDS}, skipped")
            continue
        yield summary


def _load_phase8_rows():
    if not PHASE_8_CSV.exists():
        warnings.warn(f"{PHASE_8_CSV} not found, Phase 8 skipped")
        return
    with PHASE_8_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if _missing_required(row):
                warnings.warn(f"{PHASE_8_CSV}: row for {row.get('model_name')!r} missing required fields, skipped")
                continue
            yield row


def _epochs(summary: dict[str, Any]) -> float:
    try:
        return float(summary.get("epochs") or 0)
    except (TypeError, ValueError):
        return 0.0


def _collect_rows() -> dict[tuple[str, str], dict[str, Any]]:
    # (phase, model_name) -> summary, keeping whichever has more completed epochs on a collision
    # (same rule as CLAUDE.md's "one canonical result per model+phase+protocol").
    rows: dict[tuple[str, str], dict[str, Any]] = {}

    def _add(phase: str, summary: dict[str, Any]):
        key = (phase, summary["model_name"])
        existing = rows.get(key)
        if existing is None or _epochs(summary) > _epochs(existing):
            rows[key] = summary

    for label, subdir in PHASE_DIRS:
        for summary in _load_phase_summaries(RESULTS_DIR / subdir):
            _add(label, summary)
    for summary in _load_phase8_rows():
        _add(PHASE_8_LABEL, summary)
    return rows


def _num(summary: dict[str, Any], field: str):
    v = summary.get(field, "")
    return "" if v in (None, "") else v


def _to_results_row(phase: str, s: dict[str, Any]) -> dict[str, Any]:
    fp32_top5, int8_top5 = _num(s, "fp32_top5"), _num(s, "int8_top5")
    qat_top5_drop = float(fp32_top5) - float(int8_top5) if fp32_top5 != "" and int8_top5 != "" else ""
    return {
        "phase": phase,
        "base_model": s["model_name"],
        "top1_%_FP32": _num(s, "fp32_top1"),
        "top1_%_INT8": _num(s, "int8_top1"),
        "qat_top1_drop_%": _num(s, "quantization_drop_top1"),
        "top5_%_FP32": fp32_top5,
        "top5_%_INT8": int8_top5,
        "qat_top5_drop_%": qat_top5_drop,
        "loss_FP32": _num(s, "fp32_loss"),
        "loss_INT8": _num(s, "int8_loss"),
        "params_M_FP32": _num(s, "params_m"),
        "size_MB_FP32": _num(s, "fp32_size_mb"),
        "size_MB_INT8": _num(s, "int8_size_mb"),
    }


def _to_details_row(phase: str, s: dict[str, Any]) -> dict[str, Any]:
    row = {"phase": phase, "model_name": s["model_name"]}
    row.update({field: _num(s, field) for field in MODEL_DETAILS_FIELDS if field not in ("phase", "model_name")})
    return row


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def build(out_dir: Path, dry_run: bool = False):
    rows = _collect_rows()
    by_phase: dict[str, int] = {}
    for phase, _ in rows:
        by_phase[phase] = by_phase.get(phase, 0) + 1
    for phase, count in by_phase.items():
        print(f"{phase}: {count} models")

    if dry_run:
        print(f"[dry-run] {len(rows)} rows total, nothing written")
        return

    results_rows = [_to_results_row(phase, s) for (phase, _model), s in sorted(rows.items())]
    details_rows = [_to_details_row(phase, s) for (phase, _model), s in sorted(rows.items())]
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "results_cross_phase.csv", RESULTS_CROSS_PHASE_FIELDS, results_rows)
    _write_csv(out_dir / "model_details_cross_phase.csv", MODEL_DETAILS_FIELDS, details_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=RESULTS_DIR / "results_aggregate", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build(args.out_dir, args.dry_run)


if __name__ == "__main__":
    main()
