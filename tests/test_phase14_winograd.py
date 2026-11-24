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
    for v, a, b in [("f23", 60.0, 59.0), ("f43", 55.0, 50.0), ("f43c", 58.0, None), ("f63", 40.0, None)]:
        _arm(runs, v, "net_a", a, 0.9, fp32_correct=6)  # FP32 60.0
        if b is not None:
            _arm(runs, v, "net_b", b, 0.8, fp32_correct=7)  # FP32 70.0
    wn.main(runs, out, tmp_path / "figs")
    assert (tmp_path / "figs" / "01_winograd_numerics.png").exists()
    r = pd.read_csv(out / "runs.csv").set_index(["net", "variant"])
    assert r.loc[("net_a", "f43"), "fp32"] == 60.0 and r.loc[("net_a", "f43"), "d"] == -5.0
    assert r.loc[("net_b", "f23"), "d"] == -11.0
    m = pd.read_csv(out / "medians.csv").set_index("variant")
    assert m.n.to_dict() == {"f23": 2, "f43": 2, "f43c": 1, "f63": 1}  # the calibrated arm: own row, own median
    assert r.loc[("net_a", "f43c"), "d"] == -2.0
    assert m.loc["f23", "median_d"] == -5.5 and m.loc["f63", "median_d"] == -20.0  # f63: only net_a's arm exists


def test_no_arm_yet(tmp_path, capsys):
    wn.main(tmp_path / "runs", tmp_path / "out")
    assert "no finished" in capsys.readouterr().out and not (tmp_path / "out").exists()



def test_f63_probe_helpers():
    """f63_collapse's probes that need no bridge: which convs stay Winograd (--wino-on), the swap of a qat_wino conv back to an
    exact conv with its weights, the 99.9th percentile of |x| (--observer p999) and the relative drift (the layer fields)."""
    import torch

    from ml.winograd_bridge import to_exact, wino_mask
    from scripts.phase14.f63_collapse import pct, rel

    assert wino_mask(5, "all") == {0, 1, 2, 3, 4} and wino_mask(5, "every2") == {0, 2, 4} and wino_mask(0, "every2") == set()

    class Stub(torch.nn.Module):  # stands in for a qat_wino conv: weight (cout, cin, 3, 3) and an optional bias
        def __init__(self, bias):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(6, 4, 3, 3))
            self.bias = torch.nn.Parameter(torch.randn(6)) if bias else None

    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, 8)
    for bias in (True, False):
        stub = Stub(bias)
        net = torch.nn.Sequential(torch.nn.Sequential(stub), torch.nn.ReLU())  # a nested name, "0.0", like resnet's conv1.0
        to_exact(net, ["0.0"])
        assert isinstance(net[0][0], torch.nn.Conv2d)
        want = torch.nn.functional.conv2d(x, stub.weight, stub.bias, padding=1)
        assert torch.allclose(net[0][0](x), want, atol=1e-6)

    v = torch.arange(1001.0) - 500  # |v| = 0, 1, 1, 2, 2, ..., 500, 500
    assert abs(pct(v).item() - 500) < 0.1 and abs(pct(v, 0.5).item() - 250) < 1
    assert abs(rel(torch.tensor([3.0, 4.0]), torch.tensor([3.0, 0.0])) - 4 / 3) < 1e-6


def test_keep_winograd():
    """ml.winograd_bridge.keep_winograd, the swap of the arm phase_14_winograd_f63_every2_pack and of f63_collapse --wino-on:
    every2 keeps Winograd on the 1st, 3rd, ... qat_wino conv in module order (an exact 1x1 in between does not count) and
    makes the others exact convs with the same output; "all" changes nothing."""
    import torch

    from ml.winograd_bridge import keep_winograd

    class Wino(torch.nn.Conv2d):  # stands in for a qat_wino conv: a 3x3 conv that carries an observer
        def __init__(self):
            super().__init__(4, 4, 3, padding=1)
            self.register_buffer("act_absmax", torch.ones(()))

    torch.manual_seed(0)
    net = torch.nn.Sequential(Wino(), torch.nn.ReLU(), torch.nn.Sequential(Wino(), torch.nn.Conv2d(4, 4, 1)), Wino())
    x = torch.randn(2, 4, 8, 8)
    want = net(x)
    assert keep_winograd(net, "all") == []
    assert keep_winograd(net, "every2") == ["2.0"]
    assert [type(m) is Wino for m in (net[0], net[2][0], net[3])] == [True, False, True]
    assert torch.allclose(net(x), want, atol=1e-6)


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
    # a synthesized netlist's fixed scales (--hw-params): `su` 1-3 bits under the analytic worst case keeps more of U, and
    # `sv` is the netlist's, not the one computed from V (a huge sv_hw leaves nothing of V)
    hw = import_bridge_module("qat_wino").make_wino_conv("f63", pack=True, u_w=9, v_w=8, hw_params="f63c")(4, 4, bias=False).eval()
    hw.act_absmax.fill_(1.0)
    assert kept_fractions(hw, x)[1] >= uk
    hw.sv_hw.fill_(30.0)
    assert kept_fractions(hw, x)[2] == 0.0


def test_f63_probe_summary_row(tmp_path):
    """scripts/phase14/f63_probe_summary: observers doubling at every Winograd conv -> growth x2 per conv; conv_gain's median
    drift_out / drift_in per kind (a conv with no drift reaching it is skipped); file names -> net / mode / draw."""
    import pytest

    from scripts.phase14.f63_probe_summary import row, rows

    layer = lambda i: dict(absmax_ratio=2.0 ** i, x_kept=1 - i / 10, u_kept=0.5 / (i + 1))
    cg = {"a": dict(kind="winograd", k=3, drift_in=0.0, drift_out=0.1), "b": dict(kind="winograd", k=3, drift_in=0.1, drift_out=0.2),
          "c": dict(kind="exact", k=1, drift_in=0.2, drift_out=0.1), "d": dict(kind="exact", k=3, drift_in=0.2, drift_out=0.3)}
    f63 = dict(top1=1.5, layers=[layer(i) for i in range(4)], conv_gain=cg)
    r = row(f63)
    assert r["growth_per_conv"] == pytest.approx(2.0) and r["max_absmax_ratio"] == 8.0
    assert r["min_x_kept"] == pytest.approx(0.7) and r["min_u_kept"] == pytest.approx(0.125)
    assert (r["gain_wino"], r["gain_exact1x1"], r["gain_exact3x3"]) == pytest.approx((2.0, 0.5, 1.5))
    (tmp_path / "f63_probe_vgg13_fpga_every2_s1.json").write_text(json.dumps({"new": {"f63": f63}}))
    (tmp_path / "f63_probe_vgg13_fpga_default_s0_draco1.json").write_text(json.dumps({"new": {"f63": f63}}))
    (tmp_path / "f63_collapse_vgg13_fpga_f63c.json").write_text(json.dumps({"net": "vgg13_fpga", "hw_params": {"f63": "f63c"},
                                                                            "new": {"f63": f63}}))
    assert sorted((x["net"], x["mode"], x["draw"]) for x in rows(tmp_path)) == [
        ("vgg13_fpga", "default", "s0_draco1"), ("vgg13_fpga", "every2", "s1"), ("vgg13_fpga", "f63c", "unseeded")]
