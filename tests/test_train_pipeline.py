"""scripts/train.py run_experiment() end to end on a tiny CPU model and synthetic data --
the per-model stage pipeline, and what a stop signal (Slurm's pre-timeout SIGUSR1) does to it."""
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import scripts.train as train
from ml.registry import MODEL_REGISTRY


def _tiny_model():
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 200))


def _run(tmp_path, monkeypatch, stages, ctor=_tiny_model, **runtime):
    g = torch.Generator().manual_seed(0)
    loader = DataLoader(TensorDataset(torch.randn(8, 3, 8, 8, generator=g), torch.randint(0, 200, (8,), generator=g)),
                        batch_size=4)
    monkeypatch.setattr(train, "ensure_dataset_path", lambda cfg: tmp_path)
    monkeypatch.setattr(train, "create_imagenet_loaders", lambda cfg, persistent_workers=False: (loader.dataset, loader.dataset, loader, loader))
    monkeypatch.setattr(train, "create_test_loader", lambda cfg, train_ds: loader)
    monkeypatch.setattr(train, "dataset_fingerprint", lambda path: {"sha256": "test"})
    monkeypatch.setattr(train.signal, "signal", lambda *args: None)  # keep pytest's own SIGINT handler
    monkeypatch.setitem(MODEL_REGISTRY, "tiny", {"ctor": ctor, "fuse_map": []})
    rows = train.run_experiment(
        {"name": "exp", "models": ["tiny"], "stages": stages,
         "training": {"epochs": 3, "use_amp": False, "early_stopping_patience": None},
         "qat": {"epochs": 1}},
        {"root": str(tmp_path), "device": "cpu", "tensorboard": False, **runtime},
    )
    return rows, tmp_path / "exp" / "tiny"


def test_fp32_stage_writes_summary(tmp_path, monkeypatch):
    rows, run_root = _run(tmp_path, monkeypatch, ["fp32"])
    assert [row["model_name"] for row in rows] == ["tiny"]
    assert (rows[0]["epochs_used"], rows[0]["epochs_budget"]) == (3, 3)
    assert (run_root / "results" / "tiny_summary.json").exists()
    assert (run_root / "resolved_config.json").exists()
    # metrics saved once so a later accuracy/calibration question never needs a rerun
    assert (run_root / "results" / "tiny_fp32_val_logits.npz").exists()
    assert (run_root / "results" / "tiny_layer_stats.json").exists()
    assert rows[0]["fp32_ece"] is not None
    z = np.load(run_root / "results" / "tiny_fp32_val_logits.npz")  # top1 is the plain fraction right (micro)
    assert rows[0]["fp32_top1"] == pytest.approx(100 * (z["logits"].argmax(1) == z["labels"]).mean())
    assert rows[0]["fp32_bs1_latency_ms_per_image"] is not None
    # the held-out test set is evaluated too, logits kept (here the same synthetic loader)
    assert rows[0]["test_fp32_top1"] == pytest.approx(rows[0]["fp32_top1"])
    assert (run_root / "results" / "tiny_fp32_test_logits.npz").exists()
    assert rows[0]["quant_protocol"]


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

    def capture(qat_model, **_):
        captured.update({k: v.detach().clone() for k, v in qat_model.state_dict().items()})
        raise Converted  # the tiny test model has no QuantStub, so a real int8 eval can't run

    monkeypatch.setattr(train, "convert_to_int8", capture)
    with pytest.raises(Converted):
        _run(tmp_path, monkeypatch, ["fp32", "qat", "int8"])
    qat_best = torch.load(ckpts / "qat_tiny_best.pth", weights_only=False)["model_state_dict"]
    assert all(torch.equal(captured[k], v) for k, v in qat_best.items())


FP32_TRAINING_FIELDS = ("epochs", "epochs_used", "best_val_top1", "best_val_loss", "final_train_loss", "avg_epoch_time_s",
                        "total_training_time_s", "peak_gpu_mem_mb", "avg_images_per_sec", "avg_cpu_percent",
                        "avg_ram_used_mb")


@pytest.mark.parametrize("keep_fp32_resume", [True, False])
def test_qat_rerun_keeps_the_fp32_training_record(tmp_path, monkeypatch, keep_fp32_resume):
    """scripts/pcad/rerun_qat_fused.sh redoes only QAT on a finished FP32 run. The summary is rewritten, and its FP32
    training fields (loss curve end, epoch time, total time, hardware) must survive -- vgg16's lost them when its run
    dir had no {m}_resume.pth, since the skip path rebuilt them from an empty history."""
    rows, run_root = _run(tmp_path, monkeypatch, ["fp32"])
    before = {k: rows[0][k] for k in FP32_TRAINING_FIELDS}
    if not keep_fp32_resume:
        (run_root / "checkpoints" / "tiny_resume.pth").unlink()
    rows, _ = _run(tmp_path, monkeypatch, ["fp32", "qat"])
    assert {k: rows[0][k] for k in FP32_TRAINING_FIELDS} == before
    assert rows[0]["qat_epochs_used"] == 1


class _TinyQuantizable(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant, self.dequant = torch.ao.quantization.QuantStub(), torch.ao.quantization.DeQuantStub()
        self.net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(inplace=False), nn.AdaptiveAvgPool2d(1),
                                 nn.Flatten(), nn.Linear(4, 200))

    def forward(self, x):
        return self.dequant(self.net(self.quant(x)))


def test_resubmitting_a_finished_run_reevaluates_it_without_touching_its_training_record(tmp_path, monkeypatch):
    """How a finished run is re-evaluated (2026-10-06, INT8 accuracy engine): resubmitted as is, FP32 and QAT resume as
    no-ops and only the evaluations run again. Same checkpoints, every training/eval field the same -- only the
    re-measured latencies may move -- the log appended to, and the provenance of the job that trained it kept."""
    import json
    import logging
    for name in ("tiny", "qat_tiny"):  # Trainer attaches its log file once per process and name; earlier tests did
        logging.getLogger(f"{train.Trainer.LOGGER_PREFIX}.{name}").handlers.clear()
    rows, run_root = _run(tmp_path, monkeypatch, ["fp32", "qat", "int8"], ctor=_TinyQuantizable)
    ckpts = {p.name: p.read_bytes() for p in (run_root / "checkpoints").glob("*_best.pth")}
    log = (run_root / "logs" / "tiny.log").read_text()
    first = json.loads((run_root / "resolved_config.json").read_text())["provenance"]
    again, _ = _run(tmp_path, monkeypatch, ["fp32", "qat", "int8"], ctor=_TinyQuantizable)
    changed = {k for k in rows[0].keys() | again[0].keys() if rows[0].get(k) != again[0].get(k)}
    assert all(k.endswith(("_latency_ms_per_image", "_latency_iqr_ms_per_image", "_throughput_img_per_s"))
               for k in changed), changed
    assert {p.name: p.read_bytes() for p in (run_root / "checkpoints").glob("*_best.pth")} == ckpts
    assert (run_root / "logs" / "tiny.log").read_text().startswith(log)
    assert json.loads((run_root / "resolved_config.json").read_text())["provenance_history"] == [first]


def test_saved_int8_artifact_reloads_and_reproduces_the_reported_int8_logits(tmp_path, monkeypatch):
    """The INT8 checkpoint used to be torch.save(module): quantized convs can't be unpickled, so no INT8 artifact
    could ever be loaded back. It is now a state_dict, rebuilt by ml.quantization.load_int8_model."""
    from ml.quantization import load_int8_model

    rows, run_root = _run(tmp_path, monkeypatch, ["fp32", "qat", "int8"], ctor=_TinyQuantizable)
    int8 = load_int8_model("tiny", run_root / "checkpoints")
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 3, 8, 8, generator=g)  # _run's data, same generator order
    saved = np.load(run_root / "results" / "tiny_int8_val_logits.npz")["logits"].astype("float32")
    with torch.no_grad():
        assert np.allclose(int8(x).numpy(), saved, atol=1e-2)
    assert rows[0]["int8_top1"] is not None and rows[0]["int8_size_mb"] > 0


def test_evaluate_reports_the_standard_micro_top1(tmp_path):
    """Always predicts class 0 on labels [0, 0, 0, 1]: 3 of 4 images right = 75%. torchmetrics' default "macro"
    average (per-class mean) gives 50% -- what every summary top1/top5 reported until 2026-09-30."""
    from ml.config import TrainerConfig
    from ml.trainer import Trainer

    class Always0(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return torch.eye(5)[0].repeat(len(x), 1) + 0 * self.p

    loader = DataLoader(TensorDataset(torch.zeros(4, 1), torch.tensor([0, 0, 0, 1])), batch_size=4)
    trainer = Trainer(Always0(), loader, loader, TrainerConfig(), torch.device("cpu"), tmp_path, "t", num_classes=5)
    assert trainer.evaluate(topk=(1,))["top1"] == pytest.approx(75.0)


def test_a_run_dir_of_another_protocol_is_refused_not_resumed(tmp_path):
    """refuse_foreign_run_dir: same protocol (a requeue, another machine's dataset path) passes; another lr or seed in
    the saved resolved_config.json stops the run before any checkpoint in that dir is reused."""
    cfg = {"experiment": {"seed": 42}, "data": {"dataset_path": "/laptop/train", "batch_size": 128},
           "training": {"lr": 0.01, "optimizer": "sgd"}, "qat": {"lr": 1e-4}, "provenance": {"git_hash": "a"}}
    train.save_resolved_config(tmp_path, cfg)
    train.refuse_foreign_run_dir(tmp_path, {**cfg, "data": {**cfg["data"], "dataset_path": "/pcad/train"},
                                            "provenance": {"git_hash": "b"}})
    for other in ({**cfg, "training": {"lr": 3e-4, "optimizer": "adamw"}}, {**cfg, "experiment": {"seed": 43}}):
        with pytest.raises(SystemExit, match="another protocol"):
            train.refuse_foreign_run_dir(tmp_path, other)
