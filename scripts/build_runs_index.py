#!/usr/bin/env python
"""Scan every run layout under outputs/ and emit one row per run to results/runs_index.csv.

Tier 2 of the outputs/ reorg (see research investigation, 2026-09): rather than unifying the
four writer layouts (scripts/train.py, scripts/train_det_seg.py, notebooks, scripts/profile_hardware.py)
into one directory shape, this walks all four in place and produces a single discoverable index.

Usage: python -m scripts.build_runs_index [--outputs-root outputs] [--out results/runs_index.csv]
"""
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

FIELDS = ["phase", "experiment", "model", "stage", "runtime", "host", "git_hash", "path", "top1", "size_mb"]

DET_SEG_RUN_ID_RE = re.compile(
    r"^(?P<prefix>ssd|seg)_(?P<model>.+?)_(?P<stage>fp32|qat|int8)(?P<pretrained>_pretrained)?(?:_(?P<experiment>.+))?$"
)


def _phase_of(experiment: str) -> str:
    m = re.match(r"^(phase_\d+)", experiment)
    return m.group(1) if m else ""


def _runtime_of(path: Path, outputs_root: Path) -> str:
    return path.relative_to(outputs_root).parts[0]


def _row(outputs_root: Path, path: Path, **kwargs: Any) -> dict[str, Any]:
    experiment = kwargs.get("experiment", "")
    return {
        "phase": _phase_of(experiment),
        "experiment": experiment,
        "model": "",
        "stage": "",
        "runtime": _runtime_of(path, outputs_root),
        "host": "",
        "git_hash": "",
        "path": str(path),
        "top1": "",
        "size_mb": "",
        **kwargs,
    }


def _classification_rows(outputs_root: Path):
    # scripts/train.py layout: <runtime>/<experiment>/<model>/results/{model}[_qat_wino]_summary.json
    for summary_path in outputs_root.rglob("results/*_summary.json"):
        model_dir = summary_path.parent.parent
        try:
            summary = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        provenance = summary.get("config", {}).get("provenance", {})
        stage = "qat_wino" if summary_path.name.endswith("_qat_wino_summary.json") else summary.get("mode", "")
        yield _row(
            outputs_root, summary_path,
            experiment=model_dir.parent.name,
            model=model_dir.name,
            stage=stage,
            host=provenance.get("hostname", ""),
            git_hash=provenance.get("git_hash", ""),
            top1=summary.get("best_val_top1", summary.get("int8_top1", "")),
            size_mb=summary.get("int8_size_mb") or summary.get("fp32_size_mb", ""),
        )


def _det_seg_rows(outputs_root: Path):
    # scripts/train_det_seg.py layout: <runtime>/<experiment>/<run_id>/{config.yaml,git_hash.txt,metrics.json}
    for config_path in outputs_root.rglob("config.yaml"):
        run_dir = config_path.parent
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        m = DET_SEG_RUN_ID_RE.match(run_dir.name)
        if not m:
            continue
        try:
            metrics = json.loads(metrics_path.read_text())
        except (json.JSONDecodeError, OSError):
            metrics = {}
        git_hash_path = run_dir / "git_hash.txt"
        top1 = metrics.get("best_val_mAP", metrics.get("best_val_mIoU", ""))
        size_mb = metrics.get("summary", {}).get("true_size_mb", "")
        stage = m["stage"] + ("_pretrained" if m["pretrained"] else "")
        yield _row(
            outputs_root, run_dir,
            experiment=m["experiment"] or run_dir.parent.name,
            model=m["model"],
            stage=stage,
            git_hash=git_hash_path.read_text().strip() if git_hash_path.exists() else "",
            top1=top1,
            size_mb=size_mb,
        )


def _notebook_rows(outputs_root: Path):
    # notebooks layout: outputs/notebooks/<phase>/{arch}_meta.json, qat_{arch}_meta.json
    for meta_path in outputs_root.rglob("notebooks/*/*_meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        stem = meta_path.stem.removesuffix("_meta")
        stage, model = ("qat", stem.removeprefix("qat_")) if stem.startswith("qat_") else ("fp32", stem)
        ckpt_path = meta_path.with_name(f"{stem}_best.pth")
        yield _row(
            outputs_root, meta_path,
            experiment=meta_path.parent.name,
            model=model,
            stage=stage,
            top1=meta.get("best_val_acc", ""),
            size_mb=round(ckpt_path.stat().st_size / 1e6, 2) if ckpt_path.exists() else "",
        )


def _profiling_rows(outputs_root: Path):
    # scripts/profile_hardware.py layout: <runtime>/<experiment>/{runs,backfill}/{device_tag}_profile.json (JSONL)
    for profile_path in outputs_root.rglob("*_profile.json"):
        try:
            with profile_path.open() as f:
                metadata = json.loads(f.readline())
        except (json.JSONDecodeError, OSError, StopIteration):
            metadata = {}
        yield _row(
            outputs_root, profile_path,
            experiment=profile_path.parent.parent.name if profile_path.parent.name in ("runs", "backfill") else profile_path.parent.name,
            model="(sweep)",
            stage="profile",
            host=metadata.get("device_name", profile_path.stem.removesuffix("_profile")),
        )


def build_index(outputs_root: Path) -> list[dict[str, Any]]:
    rows = [
        *_classification_rows(outputs_root),
        *_det_seg_rows(outputs_root),
        *_notebook_rows(outputs_root),
        *_profiling_rows(outputs_root),
    ]
    rows.sort(key=lambda r: (r["runtime"], r["experiment"], r["model"], r["stage"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", default="outputs", type=Path)
    parser.add_argument("--out", default="results/runs_index.csv", type=Path)
    args = parser.parse_args()

    rows = build_index(args.outputs_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
