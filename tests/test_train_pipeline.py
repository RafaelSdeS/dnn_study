"""scripts/train.py run_experiment() end to end on a tiny CPU model and synthetic data --
the per-model stage pipeline, and what a stop signal (Slurm's pre-timeout SIGUSR1) does to it."""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import scripts.train as train
from ml.registry import MODEL_REGISTRY


def _tiny_model():
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 200))


def _run(tmp_path, monkeypatch, stages):
    g = torch.Generator().manual_seed(0)
    loader = DataLoader(TensorDataset(torch.randn(8, 3, 8, 8, generator=g), torch.randint(0, 200, (8,), generator=g)),
                        batch_size=4)
    monkeypatch.setattr(train, "ensure_dataset_path", lambda cfg: tmp_path)
    monkeypatch.setattr(train, "create_imagenet_loaders", lambda cfg, persistent_workers=False: (None, None, loader, loader))
    monkeypatch.setattr(train.signal, "signal", lambda *args: None)  # keep pytest's own SIGINT handler
    monkeypatch.setitem(MODEL_REGISTRY, "tiny", {"ctor": _tiny_model, "fuse_map": []})
    rows = train.run_experiment(
        {"name": "exp", "models": ["tiny"], "stages": stages,
         "training": {"epochs": 3, "use_amp": False, "early_stopping_patience": None},
         "qat": {"epochs": 1}},
        {"root": str(tmp_path), "device": "cpu", "tensorboard": False, "benchmark_warmup": 1},
    )
    return rows, tmp_path / "exp" / "tiny"


def test_fp32_stage_writes_summary(tmp_path, monkeypatch):
    rows, run_root = _run(tmp_path, monkeypatch, ["fp32"])
    assert [row["model_name"] for row in rows] == ["tiny"]
    assert (run_root / "results" / "tiny_summary.json").exists()
    assert (run_root / "resolved_config.json").exists()


def test_stop_during_fp32_ends_the_run_before_qat(tmp_path, monkeypatch):
    original_fit = train.Trainer.fit

    def fit_then_stop(self, resume_from=None):
        self.request_stop()  # what the SIGUSR1 handler does; fit() finishes its epoch, then returns
        return original_fit(self, resume_from=resume_from)

    monkeypatch.setattr(train.Trainer, "fit", fit_then_stop)
    rows, run_root = _run(tmp_path, monkeypatch, ["fp32", "qat", "int8"])
    assert rows == []
    assert (run_root / "checkpoints" / "tiny_resume.pth").exists()  # what the requeued job resumes from
    assert not list((run_root / "checkpoints").glob("qat_*"))  # QAT never built on the truncated model
    assert not list((run_root / "results").glob("*.json"))  # and no summary claims the run finished


def test_rerun_with_only_the_qat_best_left_converts_the_trained_qat_model(tmp_path, monkeypatch):
    _, run_root = _run(tmp_path, monkeypatch, ["fp32", "qat"])
    ckpts = run_root / "checkpoints"
    (ckpts / "qat_tiny_resume.pth").unlink()  # e.g. an archived run that kept only *_best.pth

    class Converted(Exception):
        pass

    captured = {}

    def capture(qat_model):
        captured.update({k: v.detach().clone() for k, v in qat_model.state_dict().items()})
        raise Converted  # the tiny test model has no QuantStub, so a real int8 eval can't run

    monkeypatch.setattr(train, "convert_to_int8", capture)
    with pytest.raises(Converted):
        _run(tmp_path, monkeypatch, ["fp32", "qat", "int8"])
    qat_best = torch.load(ckpts / "qat_tiny_best.pth", weights_only=False)["model_state_dict"]
    assert all(torch.equal(captured[k], v) for k, v in qat_best.items())
