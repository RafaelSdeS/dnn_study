"""Phase 14 -- Winograd-FPGA numerics on Phase 11's recipe (docs/logs/PHASE14_LOG.md), an extra study outside Phase 11: for each
net and each packed-Winograd variant (F23 / F43 / F63), the test top-1 of its `qat_wino` arm minus the net's FP32 top-1 on the same
10k test images (Tiny ImageNet's official val split).

    runs.csv     one row per (net, variant) with a finished arm: fp32, arm (test top-1, %), d = arm - fp32 (pp), the top-1
                 agreement of the arm with the FP32 net on the test set
    medians.csv  per variant: the number of nets (n) and the median / min / max of d

The FP32 number is recomputed from `<net>_fp32_test_logits.npz` (copied into every arm dir, the same source for all 16 nets, the two
Phase 11 cells included); it can differ from a summary's own number by one image (31.53 vs 31.52 on the canary cell).
A net whose arm has not finished is simply absent, so `n` says how many nets a median rests on.

    python -m scripts.phase14.winograd_numerics
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = ROOT / "outputs/pcad"  # holds phase_14_winograd_f{23,43,63}_pack/<net>/results/
DEFAULT_OUT = ROOT / "results/phase_14_winograd_numerics"
VARIANTS = ("f23", "f43", "f63")


def fp32_top1(logits_npz):
    """Top-1 (%) of the saved logits against the saved labels."""
    z = np.load(logits_npz)
    return 100 * float((z["logits"].argmax(1) == z["labels"]).mean())


def collect(runs):
    """One row per finished arm: every `<net>_qat_wino_summary.json` under the three arm experiments."""
    rows = []
    for v in VARIANTS:
        for summary in sorted((runs / f"phase_14_winograd_{v}_pack").glob("*/results/*_qat_wino_summary.json")):
            net = summary.name.removesuffix("_qat_wino_summary.json")
            s = json.loads(summary.read_text())
            fp32, arm = fp32_top1(summary.parent / f"{net}_fp32_test_logits.npz"), s["test_eval"]["top1"]
            rows.append(dict(net=net, variant=v, fp32=fp32, arm=arm, d=arm - fp32, agreement=s["agreement_fp32_qat_wino_test"]))
    return pd.DataFrame(rows, columns=["net", "variant", "fp32", "arm", "d", "agreement"])


def medians(df):
    g = df.groupby("variant").d
    return pd.DataFrame({"n": g.size(), "median_d": g.median(), "min_d": g.min(), "max_d": g.max()}).reindex(VARIANTS).dropna().astype({"n": int}).reset_index()


def main(runs=DEFAULT_RUNS, out=DEFAULT_OUT):
    df = collect(Path(runs))
    if df.empty:
        print("no finished qat_wino arm yet")
        return
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df.sort_values(["variant", "net"]).round(4).to_csv(out / "runs.csv", index=False)
    m = medians(df)
    m.round(4).to_csv(out / "medians.csv", index=False)
    print(m.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
