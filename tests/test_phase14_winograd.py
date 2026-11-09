"""Phase 14's arm - FP32 table on a fake run tree: two nets, one of them without its F63 arm yet."""
import json

import numpy as np
import pandas as pd

from scripts.phase14 import winograd_numerics as wn


def _arm(runs, variant, net, arm_top1, agreement, fp32_correct):
    """An arm dir: FP32 logits that get `fp32_correct` of 10 images right (3 classes), and the arm's qat_wino summary."""
    res = runs / f"phase_14_winograd_{variant}_pack" / net / "results"
    res.mkdir(parents=True)
    labels = np.arange(10) % 3
    logits = np.eye(3)[labels]
    logits[fp32_correct:] = np.eye(3)[(labels[fp32_correct:] + 1) % 3]  # the last ones are wrong
    np.savez(res / f"{net}_fp32_test_logits.npz", logits=logits, labels=labels)
    (res / f"{net}_qat_wino_summary.json").write_text(
        json.dumps({"test_eval": {"top1": arm_top1}, "agreement_fp32_qat_wino_test": agreement}))


def test_arm_minus_fp32_and_medians(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "out"
    for v, a, b in [("f23", 60.0, 59.0), ("f43", 55.0, 50.0), ("f63", 40.0, None)]:
        _arm(runs, v, "net_a", a, 0.9, fp32_correct=6)  # FP32 60.0
        if b is not None:
            _arm(runs, v, "net_b", b, 0.8, fp32_correct=7)  # FP32 70.0
    wn.main(runs, out)
    r = pd.read_csv(out / "runs.csv").set_index(["net", "variant"])
    assert r.loc[("net_a", "f43"), "fp32"] == 60.0 and r.loc[("net_a", "f43"), "d"] == -5.0
    assert r.loc[("net_b", "f23"), "d"] == -11.0
    m = pd.read_csv(out / "medians.csv").set_index("variant")
    assert m.n.to_dict() == {"f23": 2, "f43": 2, "f63": 1}
    assert m.loc["f23", "median_d"] == -5.5 and m.loc["f63", "median_d"] == -20.0  # f63: only net_a's arm exists


def test_no_arm_yet(tmp_path, capsys):
    wn.main(tmp_path / "runs", tmp_path / "out")
    assert "no finished" in capsys.readouterr().out and not (tmp_path / "out").exists()



def test_f63_kept_fractions():
    """scripts/phase14/f63_collapse.kept_fractions on one packed F(6,3) conv: an input filling the observer's INT8 range
    keeps its codes and most of U, a 6x6 map (resnet18's layer4) gets the forward's tile padding, and a signal far below
    the observer's absmax (the 2026-11-08 diagnosis: observers calibrated on the error-compounded activations) loses
    every code, so nothing reaches U."""
    import pytest
    import torch

    from ml.winograd_bridge import bridge_root, import_bridge_module
    if not (bridge_root() / "qat_wino.py").exists():
        pytest.skip("Winograd-FPGA bridge not found (WINOGRAD_FPGA_ROOT)")
    from scripts.phase14.f63_collapse import kept_fractions

    torch.manual_seed(0)
    conv = import_bridge_module("qat_wino").make_wino_conv("f63", pack=True, u_w=9, v_w=8)(4, 4, bias=False).eval()
    conv.act_absmax.fill_(1.0)
    x = torch.rand(2, 4, 12, 12) * 2 - 1
    xk, uk, vk = kept_fractions(conv, x)
    assert xk > 0.99 and uk > 0.5 and vk > 0.9, (xk, uk, vk)
    assert kept_fractions(conv, x[..., :6, :6])[1] > 0.5
    assert kept_fractions(conv, x * 0.002)[:2] == (0.0, 0.0)  # |x| <= 1/500 of absmax: every code rounds to 0
