"""Phase 14 -- the F(6,3) probe table (docs/logs/PHASE14_LOG.md 2026-11-24): why packed F(6,3)'s error compounds in
resnet18_fpga / vgg13_fpga and not in googlenet_fpga. One row per net x mode x draw, from scripts/phase14/f63_collapse.py's
JSONs in results/phase_14_winograd_numerics/:
    f63_probe_<net>_<mode>_s<torch seed>[_<host>].json   the probes (modes default / every2 / clean / p999 / f63c)
    f63_collapse*.json                                   the earlier runs, no torch seed (draw "unseeded"; default / f63c)
Each row reads the new base's F63 variant: top1 (epoch 0, 1000 test images), growth_per_conv = exp(least-squares slope of
ln absmax_ratio over the Winograd convs in order; the log's x1.23 / x1.51 / x0.97), max absmax_ratio, min x_kept / u_kept,
and the median drift_out / drift_in of conv_gain over the Winograd, exact 1x1 and exact 3x3 convs (> 1: the conv grows the
error that reaches it). post_shift's random sample can flip one rounding and every later layer inherits it, so only what
holds over every draw of a cell is a result; the script prints min-max per net x mode.

    python -m scripts.phase14.f63_probe_summary   -> results/phase_14_winograd_numerics/f63_probe_summary.csv
"""
import csv
import json
import math
import re
import statistics
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "results/phase_14_winograd_numerics"
PROBE = re.compile(r"f63_probe_(?P<net>.+_fpga)_(?P<mode>default|every2|clean|p999|f63c)_(?P<draw>s\d+(?:_\w+)?)\.json")
COLS = ("top1", "n_wino", "growth_per_conv", "max_absmax_ratio", "min_x_kept", "min_u_kept",
        "gain_wino", "gain_exact1x1", "gain_exact3x3")


def gain(conv_gain, keep):
    """Median drift_out / drift_in over the convs `keep` accepts (a conv with no drift reaching it is skipped)."""
    g = [c["drift_out"] / c["drift_in"] for c in conv_gain.values() if keep(c) and c["drift_in"] > 0]
    return statistics.median(g) if g else None


def row(f63):
    """One measurement (a JSON's new.f63) -> the table's columns."""
    L, cg = f63["layers"], f63.get("conv_gain", {})
    y = [math.log(l["absmax_ratio"]) for l in L]
    return dict(top1=f63["top1"], n_wino=len(L),
                growth_per_conv=math.exp(statistics.linear_regression(range(len(y)), y).slope) if len(y) > 1 else None,
                max_absmax_ratio=max(l["absmax_ratio"] for l in L), min_x_kept=min(l["x_kept"] for l in L),
                min_u_kept=min(l["u_kept"] for l in L),
                gain_wino=gain(cg, lambda c: c["kind"] == "winograd"),
                gain_exact1x1=gain(cg, lambda c: c["kind"] == "exact" and c["k"] == 1),
                gain_exact3x3=gain(cg, lambda c: c["kind"] == "exact" and c["k"] == 3))


def rows(root=R):
    out = []
    for p in sorted(root.glob("f63_probe_*_s*.json")):
        m = PROBE.fullmatch(p.name)
        if m:
            j = json.loads(p.read_text())
            out.append(dict(net=m["net"], mode=m["mode"], draw=m["draw"], host=j.get("provenance", {}).get("hostname", ""),
                            file=p.name, **row(j["new"]["f63"])))
    for p in sorted(root.glob("f63_collapse*.json")):
        j = json.loads(p.read_text())
        out.append(dict(net=j["net"], mode="f63c" if j.get("hw_params") else "default", draw="unseeded", host="",
                        file=p.name, **row(j["new"]["f63"])))
    return out


def main(root=R):
    rs = rows(root)
    with open(root / "f63_probe_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["net", "mode", "draw", "host", "file", *COLS])
        w.writeheader()
        w.writerows(rs)
    fmt = lambda v: "-" if v is None else f"{v:.3g}"
    print(f"{'net':15s} {'mode':8s} {'n':>2s} " + " ".join(f"{c:>17s}" for c in COLS if c != "n_wino"))
    for key in sorted({(r["net"], r["mode"]) for r in rs}):
        cell = [r for r in rs if (r["net"], r["mode"]) == key]
        span = lambda c: [r[c] for r in cell if r[c] is not None]
        print(f"{key[0]:15s} {key[1]:8s} {len(cell):2d} " + " ".join(
            f"{fmt(min(span(c), default=None)):>8s}-{fmt(max(span(c), default=None)):<8s}" for c in COLS if c != "n_wino"))


if __name__ == "__main__":
    main()
