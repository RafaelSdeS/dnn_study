"""Phase 14 -- why packed F(6,3) puts the deep one-recipe nets at chance (docs/logs/PHASE14_LOG.md, 2026-11-08), an extra
study outside Phase 11. No qat_wino training: for each FP32 base -- this recipe's (phase_14_winograd_fp32) and the old AdamW
one (budget_unico, whose F63 arm still reached 37.5 % val) -- and each variant, build the arm's model with the arm yaml's
packing (ml.winograd_bridge.load_qat_wino_model), calibrate its observers (act_absmax, post_shift) on DEFAULT_CALIB train
images with BN frozen, as the arm's first epoch does, then measure:

    top1          on the first DEFAULT_TEST test images: the arm at epoch 0
    per Winograd conv, fed the FP32 net's own input x to that conv:
      cos_clean     cosine(Winograd conv, FP32 conv) with the observers set from x itself: the layer's own error
      absmax_ratio  the arm-calibrated act_absmax / absmax(x): how far the observer, which saw the Winograd net's own
                    (error-compounded) activations, sits above the clean signal
      x_kept        fraction of the nonzero x whose INT8 code is nonzero at the arm-calibrated scale
      u_kept        of the nonzero transformed tiles U = Bt d B, the fraction that survive the packing shift `su`
      v_kept        fraction of the transformed filter V = G g Gt that survives the packing shift `sv`
      x_crest       absmax / rms of x;  w_crest: the same for the conv weights

--hw-params f63=f63c builds that variant with a synthesized netlist's fixed scales instead (`su`, `sv`, `align_base` from the
bridge's gate_flags/params_<name>.txt: the calibrated ones the article's limitation names) and measures only the variants
named; still epoch 0, so it says whether the signal now crosses the net, not where QAT would take it.

-> results/phase_14_winograd_numerics/f63_collapse.json. Checkpoints are gitignored and the laptop keeps none: rsync the two
`<net>_best.pth` from PCAD to the same relpaths first (CLAUDE.md "Artifacts"), or pass --new / --old. Export
WINOGRAD_FPGA_ROOT = the bridge the arms ran (f44da9a, PCAD ~/winograd_bridge_m7/avaliacao_redes). CPU only, a few minutes.

    python -m scripts.phase14.f63_collapse [--net vgg13_fpga --new <dir> --old <dir>] [--hw-params f63=f63c --out <json>]
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

import ml.model_registrations  # noqa: F401  (fills MODEL_REGISTRY)
from ml.data import _TinyImageNetVal, _val_transform
from ml.quantization import load_best_model
from ml.registry import MODEL_REGISTRY
from ml.runtime import ensure_dataset_path, load_profile
from ml.winograd_bridge import bridge_provenance, import_bridge_module, load_qat_wino_model
from scripts.phase14.winograd_numerics import VARIANTS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NET = "resnet18_fpga"
DEFAULT_NEW = ROOT / "outputs/pcad/phase_14_winograd_fp32" / DEFAULT_NET / "checkpoints"
DEFAULT_OLD = ROOT / "outputs/pcad/archive_winograd_old_recipe/budget_unico" / DEFAULT_NET / "checkpoints"
DEFAULT_OUT = ROOT / "results/phase_14_winograd_numerics/f63_collapse.json"
DEFAULT_CALIB = 400  # train images through the observers; the arm's first epoch sees them the same way (first batch sets, then EMA)
DEFAULT_TEST = 1000
DEFAULT_BS = 50
DEFAULT_SEED = 0


def crest(t):
    return (t.abs().max() / t.pow(2).mean().sqrt()).item()


def kept_fractions(layer, x):
    """(x_kept, u_kept, v_kept) of one Winograd conv on input x at its current observer scale: its forward's packing
    quantization step by step (x and w to INT8, U = Bt d B per tile, V = G g Gt, then qs_pos by `su` / `sv`), counting
    what stays nonzero."""
    q = import_bridge_module("qat_wino")
    V = layer._v_pre(q.fake_quant(layer.weight, layer._w_scale()))
    sv = layer.sv_hw if getattr(layer, "sv_hw", None) is not None else \
        torch.clamp(torch.ceil(torch.log2(torch.clamp(V.abs().amax(dim=(0, 1)), min=1.0))) + 1 - layer.v_w, min=0.0)
    xq = q.fake_quant(x, layer._act_scale(x))
    x_kept = ((xq != 0) & (x != 0)).sum().item() / max((x != 0).sum().item(), 1)
    xq = F.pad(xq, (1, 1, 1, 1))
    t = layer.BT.shape[0]
    o = t - 2
    h, w = xq.shape[2:]  # tile alignment, as the forward pads it
    xq = F.pad(xq, (0, max(0, o * -(-(w - 2) // o) + 2 - w), 0, max(0, o * -(-(h - 2) // o) + 2 - h)))
    U = torch.einsum("ab,ncijbe,de->ncijad", layer.BT, xq.unfold(2, t, o).unfold(3, t, o), layer.BT)
    nz = U != 0
    u_kept = ((q.qs_pos(U, layer.su, layer.u_w) != 0) & nz).sum().item() / max(nz.sum().item(), 1)
    return x_kept, u_kept, (q.qs_pos(V, sv, layer.v_w) != 0).float().mean().item()


def top1(model, loader):
    with torch.no_grad():
        return 100 * sum((model(x).argmax(1) == y).sum().item() for x, y in loader) / len(loader.dataset)


def measure(label, net, ckpt_dir, calib, test, hw=None):
    """hw: {variant: hw_params}; when given, only those variants, built with the netlist's scales (`su` is the layer's)."""
    spec, dev = MODEL_REGISTRY[net], torch.device("cpu")
    fp32 = load_best_model(net, spec["ctor"], ckpt_dir, dev)
    out = {"fp32_top1": top1(fp32, test)}
    xb = next(iter(test))[0]
    for v in hw or VARIANTS:
        arm = load_profile(f"phase_14_winograd_{v}_pack", "experiments")["qat_wino"]
        cfg = SimpleNamespace(**{k: arm[k] for k in ("variant", "pack", "u_w", "v_w", "k_dsp")}, hw_params=(hw or {}).get(v))
        m = load_qat_wino_model(net, spec, ckpt_dir, dev, cfg)
        m.train()
        for b in m.modules():
            if isinstance(b, torch.nn.modules.batchnorm._BatchNorm):
                b.eval()  # only the Winograd observers calibrate; BN keeps the FP32 statistics
        with torch.no_grad():
            for x, _ in calib:
                m(x)
        m.eval()
        out[v] = {"top1": top1(m, test)}
        wino = {n: mod for n, mod in m.named_modules() if hasattr(mod, "act_absmax")}
        ref = dict(fp32.named_modules())
        seen = {}
        hooks = [ref[n].register_forward_hook(lambda mod, i, o, n=n: seen.__setitem__(n, (i[0], o))) for n in wino]
        with torch.no_grad():
            fp32(xb)
            for h in hooks:
                h.remove()
            layers = []
            for n, mod in wino.items():
                x, y = seen[n]
                xk, uk, vk = kept_fractions(mod, x)  # at the arm's calibration
                ratio = (mod.act_absmax / x.abs().max()).item()
                mod.act_absmax.zero_()
                mod.post_shift.zero_()
                mod.train()
                mod(x)  # a zeroed observer takes its first forward's value outright (qat_wino): set from x
                mod.eval()
                layers.append(dict(name=n, cos_clean=F.cosine_similarity(mod(x).flatten(), y.flatten(), dim=0).item(),
                                   absmax_ratio=ratio, x_kept=xk, u_kept=uk, v_kept=vk, x_crest=crest(x),
                                   w_crest=crest(mod.weight)))
        out[v]["layers"] = layers
        print(f"{label} {v}: top1 {out[v]['top1']:5.2f} (FP32 {out['fp32_top1']:.2f})", flush=True)
    return out


def main(net=DEFAULT_NET, new=DEFAULT_NEW, old=DEFAULT_OLD, out=DEFAULT_OUT, n_calib=DEFAULT_CALIB, n_test=DEFAULT_TEST,
         bs=DEFAULT_BS, seed=DEFAULT_SEED, hw=None):
    train_dir = ensure_dataset_path(load_profile("local", "runtime"))
    tf = _val_transform(64)
    train = ImageFolder(train_dir, transform=tf)
    pick = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))[:n_calib].tolist()
    calib = DataLoader(Subset(train, pick), batch_size=bs)
    test = DataLoader(Subset(_TinyImageNetVal(train_dir.parent / "val", train.class_to_idx, tf), range(n_test)), batch_size=bs)
    res = {"net": net, "n_calib": n_calib, "n_test": n_test, "seed": seed, "bridge": bridge_provenance(), "hw_params": hw,
           "new": measure("new", net, Path(new), calib, test, hw), "old": measure("old", net, Path(old), calib, test, hw)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    cols = ("cos_clean", "absmax_ratio", "x_kept", "u_kept", "x_crest")
    for v in hw or VARIANTS:
        print(f"\n{v} {'layer':22s}" + "".join(f"| {c + ' new/old':>21s} " for c in cols))
        for a, b in zip(res["new"][v]["layers"], res["old"][v]["layers"]):
            print(f"   {a['name']:22s}" + "".join(f"| {a[c]:10.2f} {b[c]:10.2f} " for c in cols))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--net", default=DEFAULT_NET)
    p.add_argument("--new", type=Path, default=DEFAULT_NEW, help="dir with this recipe's <net>_best.pth")
    p.add_argument("--old", type=Path, default=DEFAULT_OLD, help="dir with budget_unico's <net>_best.pth")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-test", type=int, default=DEFAULT_TEST)
    p.add_argument("--hw-params", nargs="+", metavar="VARIANT=NAME", help="e.g. f63=f63c: only these variants, netlist scales")
    a = p.parse_args()
    main(a.net, a.new, a.old, a.out, n_test=a.n_test, hw=dict(x.split("=") for x in a.hw_params) if a.hw_params else None)
