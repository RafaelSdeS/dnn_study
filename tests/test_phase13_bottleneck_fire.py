"""Phase 13 Stage B's figure and tables on a fake run tree: Phase 11's design (alexnet_fire lives in phase_11_families), the
phase_13_hybrids plain bodies and every phase_13_bottleneck_fire model -- and every matched pair of the 12 cells found."""
import matplotlib
import pandas as pd

from tests.test_phase11_figures import _fake_runs

matplotlib.use("Agg")


def _patch(monkeypatch, tmp_path, runs):
    from scripts.phase11 import design_figures
    from scripts.phase13 import bottleneck_fire

    for mod, name, value in [(bottleneck_fire, "RUNS", (runs,)), (design_figures, "RUNS", (runs,)),
                             (bottleneck_fire, "FIGS", tmp_path / "figs"), (bottleneck_fire, "TABLES", tmp_path / "tables")]:
        monkeypatch.setattr(mod, name, value)
    return bottleneck_fire


def test_stage_b_figure_and_contrasts(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    _fake_runs(runs, lambda e, m: True)
    _fake_runs(runs, lambda e, m: True, ["phase_13_hybrids", "phase_13_bottleneck_fire"])
    bf = _patch(monkeypatch, tmp_path, runs)
    bf.main()
    assert (tmp_path / "figs" / "01_bottleneck_fire.png").exists()
    all_runs = pd.read_csv(tmp_path / "tables" / "all_runs.csv")
    assert set(all_runs.key) == set(bf.CELLS.values())
    assert "winograd_pct" in all_runs.columns
    c = pd.read_csv(tmp_path / "tables" / "contrasts.csv")
    # 2 kernels x 2 units per stage = 4 pairs per swap
    assert c.groupby("factor").size().to_dict() == {label: 4 for label, _, _ in bf.CHANGES}
    assert set(c.A) | set(c.B) == set(bf.CELLS.values())


def test_stage_b_partial_runs_and_none(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    _fake_runs(runs, lambda e, m: True)
    bf = _patch(monkeypatch, tmp_path, runs)
    bf.main()  # no phase_13_bottleneck_fire run: says so, writes nothing
    assert not (tmp_path / "tables").exists()
    one = "alexnet_k2x2bottleneckstacked_stride1_2pool2x2_map16_gap_bn"
    _fake_runs(runs, lambda e, m: m == one, ["phase_13_bottleneck_fire"])
    _fake_runs(runs, lambda e, m: True, ["phase_13_hybrids"])
    bf.main()  # one new run + its plain partner from the hybrids, plus alexnet_fire (Phase 11) against its plain body
    c = pd.read_csv(tmp_path / "tables" / "contrasts.csv")
    assert c[c.factor == "conv → Bottleneck"].B.tolist() == [one]
    assert c[c.factor == "conv → Fire"].B.tolist() == [bf.FIRE_3X3] and not (c.factor == "Bottleneck → Fire").any()
    assert (tmp_path / "figs" / "01_bottleneck_fire.png").exists()
