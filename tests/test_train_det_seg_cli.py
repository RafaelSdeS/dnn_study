"""scripts/train_det_seg.py: run_id naming (the qat/int8 stages find the previous stage's
checkpoint by it, and it has to keep matching the dirs already under outputs/pcad/phase_7_*),
and the fp32 -> qat -> int8 hand-off through run() with the heavy pieces faked out."""
import json
import subprocess
import sys
from argparse import Namespace

import pytest
import torch
import torch.nn as nn

import scripts.train_det_seg as det_seg
from scripts.train_det_seg import run_id_for


def _args(task, experiment=None, pretrained_ckpt=None):
    return Namespace(task=task, model="alexnet_bottleneck", experiment=experiment, pretrained_ckpt=pretrained_ckpt)


def test_run_id_matches_existing_output_dirs():
    assert run_id_for(_args("segmentation", "phase_7_segmentation"), "fp32") == \
        "seg_alexnet_bottleneck_fp32_phase_7_segmentation"
    assert run_id_for(_args("detection", "phase_7_detection", "ckpt.pth"), "qat") == \
        "ssd_alexnet_bottleneck_qat_pretrained_phase_7_detection"
    assert run_id_for(_args("detection"), "int8") == "ssd_alexnet_bottleneck_int8"


class _FakeTrainer:
    """Stands in for Detection/SegmentationTrainer: fit() writes the _best.pth the next stage loads."""
    instances: list = []
    stop_requested = False

    def __init__(self, model, train_loader, val_loader, cfg, device, save_dir, run_name, num_classes,
                 epoch_callback=None, log_file=None):
        self.model, self.device, self.save_dir, self.run_name = model, device, save_dir, run_name
        self.epoch_callback = epoch_callback
        _FakeTrainer.instances.append(self)

    def request_stop(self):
        pass

    def fit(self, resume_from=None):
        torch.save({"model_state_dict": self.model.state_dict()}, self.save_dir / f"{self.run_name}_best.pth")
        return {"best_val_mIoU": 0.5}

    def _validate(self, model, criterion=None):
        return {"val_loss": 1.0, "val_mIoU": 0.25}


class _StoppedTrainer(_FakeTrainer):
    stop_requested = True  # what a SIGTERM/SIGINT during fit() leaves behind


def _fake_segmentation(monkeypatch, trainer=_FakeTrainer):
    monkeypatch.setattr(det_seg.signal, "signal", lambda *args: None)  # keep pytest's own SIGINT handler
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(det_seg.TASKS, "segmentation", {
        **det_seg.TASKS["segmentation"],
        "loaders": lambda cfg: ([0], [0], None, None),
        "build": lambda arch, num_classes, image_size, pretrained_ckpt=None: nn.Linear(2, 2),
        "build_qat": lambda model, device: model,
        "to_int8": lambda model: model,
        "trainer": trainer,
        "summary": lambda model, img_size, val_loader, device, checkpoint_path: {"checkpoint": str(checkpoint_path)},
    })


def _run_stage(tmp_path, stage):
    det_seg.run(Namespace(task="segmentation", model="alexnet_fire", stage=stage, experiment=None,
                          pretrained_ckpt=None, save_dir=str(tmp_path), dry_run=False, smoke=False,
                          skip_anchor_check=False))


def test_stop_during_fit_ends_the_stage_without_metrics(tmp_path, monkeypatch):
    _fake_segmentation(monkeypatch, trainer=_StoppedTrainer)
    with pytest.raises(SystemExit) as exc:
        _run_stage(tmp_path, "fp32")
    assert exc.value.code != 0  # so an afterok dependency never starts QAT on the truncated FP32
    assert not (tmp_path / "seg_alexnet_fire_fp32" / "metrics.json").exists()


def test_smoke_runs_the_whole_stage_chain_one_process_per_stage_in_one_temp_dir(monkeypatch):
    calls = []

    def fake_child(cmd, env):
        calls.append((cmd[cmd.index("--stage", cmd.index("--smoke")) + 1], env[det_seg.SMOKE_DIR_ENV]))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.delenv(det_seg.SMOKE_DIR_ENV, raising=False)
    monkeypatch.setattr(det_seg.subprocess, "run", fake_child)
    monkeypatch.setattr(sys, "argv", ["train_det_seg", "detection", "--stage", "qat", "--smoke"])
    det_seg.main()
    assert [stage for stage, _ in calls] == ["fp32", "qat", "int8"]  # the appended --stage wins
    assert len({save_dir for _, save_dir in calls}) == 1  # qat/int8 find the previous stage's checkpoint


def test_smoke_child_runs_its_one_stage_in_the_parents_temp_dir(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv(det_seg.SMOKE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(det_seg, "run", lambda args: calls.append((args.stage, args.save_dir)))
    monkeypatch.setattr(sys, "argv", ["train_det_seg", "detection", "--smoke", "--stage", "qat"])
    det_seg.main()
    assert calls == [("qat", str(tmp_path))]


def test_segmentation_stages_hand_off_checkpoints(tmp_path, monkeypatch):
    _fake_segmentation(monkeypatch)
    for stage in ("fp32", "qat", "int8"):
        _run_stage(tmp_path, stage)

    fp32, qat, int8 = _FakeTrainer.instances[-3:]
    assert fp32.epoch_callback is None and qat.epoch_callback is not None
    assert int8.device == torch.device("cpu")
    int8_dir = tmp_path / "seg_alexnet_fire_int8"
    metrics = json.loads((int8_dir / "metrics.json").read_text())
    assert metrics["val_loss"] == [1.0] and metrics["val_mIoU"] == [0.25]
    assert metrics["summary"]["params_m"] == 6 / 1e6  # nn.Linear(2, 2), counted before int8 conversion
    assert (int8_dir / "seg_alexnet_fire_int8_best.pth").exists()
    assert (int8_dir / "git_hash.txt").read_text().strip()
