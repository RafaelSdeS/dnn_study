"""Phase 13's figure and tables on a fake run tree: Phase 11's design (the 6 reference cells, VGG16's line) plus every
phase_13_hybrids model -- and every matched pair of the 24-cell grid found."""
import matplotlib
import pandas as pd

from tests.test_phase11_figures import _fake_runs

matplotlib.use("Agg")


def test_phase_13_figure_and_contrasts(tmp_path, monkeypatch):
    from scripts.phase11 import design_figures
    from scripts.phase13 import hybrids

    runs = tmp_path / "runs"
    _fake_runs(runs, lambda e, m: True)
    _fake_runs(runs, lambda e, m: True, ["phase_13_hybrids"])
    for mod, name, value in [(hybrids, "RUNS", (runs,)), (design_figures, "RUNS", (runs,)),
                             (hybrids, "FIGS", tmp_path / "figs"), (hybrids, "TABLES", tmp_path / "tables")]:
        monkeypatch.setattr(mod, name, value)
    hybrids.main()
    assert (tmp_path / "figs" / "01_hybrids.png").exists()
    c = pd.read_csv(tmp_path / "tables" / "contrasts.csv")
    # body 5 -> 2 per stage and stacked -> residual: 2 kernels x 2 conv1 x 2 strides; conv1, kernel, stride: 12 each
    assert c.groupby("factor").size().to_dict() == {f: n for f, _, _, _ in hybrids.CHANGES
                                                    for n in [8 if f.startswith("corpo") else 12]}
    assert set(pd.read_csv(tmp_path / "tables" / "all_runs.csv").key) == set(hybrids.GRID.values()) | {hybrids.VGG_REF}
