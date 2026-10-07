"""The real Phase 11 runs in the repo (the tracked files under outputs/pcad/phase_11_*, synced from PCAD) are what the
protocol says they are -- checked on the data, not the code: the protocol it ran, the commit, the budget, the INT8
guards, the files the analysis reads. A run that fails here must not reach a figure. Skips while no run is synced; runs
on an older QUANT_PROTOCOL (waiting for a re-evaluation or archival) are listed as a warning, not checked."""
import json
import math
import subprocess
import warnings
from dataclasses import asdict
from pathlib import Path

import pytest

import scripts.train as train
from configs.loader import load_config
from ml.quantization import QUANT_PROTOCOL

ROOT = Path(__file__).resolve().parents[1]
SUMMARIES = sorted(ROOT.glob("outputs/pcad/phase_11_*/*/results/*_summary.json"))
CURRENT = [p for p in SUMMARIES if json.loads(p.read_text()).get("quant_protocol") == QUANT_PROTOCOL]
CHANCE_TOP1 = 100 / 200


def test_runs_on_an_older_protocol_are_listed():
    stale = [str(p.parents[1].relative_to(ROOT)) for p in SUMMARIES if p not in CURRENT]
    if stale:
        warnings.warn(f"{len(stale)} Phase 11 run(s) not on QUANT_PROTOCOL {QUANT_PROTOCOL}: {stale}")


def _expected_protocol(experiment: str) -> dict:
    """What scripts/train.py resolves for this experiment (uniform_hparams: one TrainerConfig for every model)."""
    exp = load_config(f"experiments/{experiment}.yaml")
    data = train._build_data_config(load_config("data.yaml"), exp)
    data.seed = int(exp.get("seed", data.seed))
    return train._protocol({"experiment": exp, "data": asdict(data),
                            "training": asdict(train._build_trainer_config(load_config("training.yaml"), exp)),
                            "qat": asdict(train._build_qat_config(load_config("qat.yaml"), exp))})


@pytest.mark.parametrize("summary", CURRENT, ids=lambda p: f"{p.parents[2].name}/{p.parents[1].name}")
def test_every_finished_run_is_the_protocol(summary):
    run, s = summary.parents[1], json.loads(summary.read_text())
    cfg = json.loads((run / "resolved_config.json").read_text())
    assert train._protocol(cfg) == _expected_protocol(run.parent.name), "trained under another protocol than its yaml"
    for job in cfg.get("provenance_history", []) + [cfg["provenance"]]:  # every job that touched it, training included
        assert not job["git_dirty"], (job["slurm_job_id"], job["git_dirty_files"])
        assert subprocess.run(["git", "merge-base", "--is-ancestor", job["git_hash"], "HEAD"], cwd=ROOT).returncode == 0, \
            f"job {job['slurm_job_id']} ran {job['git_hash'][:7]}, not in this branch's history: not reproducible from it"
    assert s["epochs_used"] == s["epochs_budget"] == cfg["training"]["epochs"]
    assert s["qat_epochs_used"] == s["qat_epochs_budget"] == cfg["qat"]["epochs"]
    assert math.isfinite(s["final_train_loss"]) and s["best_val_top1"] > 10 * CHANCE_TOP1, "a dead run (ln(200) plateau)"
    assert s["int8_kernel_max_err_steps"] <= 1
    assert s["agreement_qat_int8"] >= train.MIN_QAT_INT8_AGREEMENT
    for stage in ("fp32", "qat", "int8"):
        assert s[f"test_{stage}_top1"] is not None
        for split in ("val", "test"):
            assert (run / "results" / f"{run.name}_{stage}_{split}_logits.npz").exists(), (stage, split)


def test_every_run_saw_the_same_dataset():
    shas = {json.loads((p.parents[1] / "resolved_config.json").read_text())["provenance"]["dataset"]["sha256"] for p in CURRENT}
    assert len(shas) <= 1, shas


# ─── the data every run reads ────────────────────────────────────────────────────────────────────────────────────────
DATASET = sorted(Path.home().glob(".cache/kagglehub/datasets/akash2sharma/*/versions/*/tiny-imagenet-200"))


@pytest.mark.skipif(not DATASET, reason="Tiny ImageNet not on this machine")
def test_the_test_set_and_the_split_are_what_the_protocol_says():
    """The held-out test set is Tiny ImageNet's official val, 50 images per class, labelled through the train set's
    class indices; the 90/10 split of train/ is disjoint and fixed by the seed (a reshuffled split or mislabelled test
    set would move every number without any error)."""
    from collections import Counter

    from ml.config import DataConfig
    from ml.data import create_imagenet_loaders, create_test_loader

    cfg = DataConfig(dataset_path=str(DATASET[-1] / "train"), num_workers=0)
    tr, va, _, _ = create_imagenet_loaders(cfg)
    assert Counter(y for _, y in create_test_loader(cfg, tr).dataset.samples) == {c: 50 for c in range(200)}
    assert (len(tr), len(va)) == (90_000, 10_000) and not set(tr.indices) & set(va.indices)
    assert create_imagenet_loaders(cfg)[1].indices == va.indices
