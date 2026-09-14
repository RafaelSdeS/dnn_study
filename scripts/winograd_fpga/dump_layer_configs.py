"""Emit per-(network, Winograd variant) layer_configs.py-schema JSONs for the sibling
Winograd-FPGA repo's run_sim{_f23,,_f63}.py -- so its throughput simulation can run against any
network trained here, not just the VGG16 the module ships with (Winograd-FPGA's own Fase 4,
per docs/plano_avaliacao_redes_winograd.md: "generalizar o fluxo VGG16 ... para consumir
manifesto arbitrario, via net_manifest.py").

Geometry only, no checkpoint needed: the RTL sim always runs synthetic weights (its own
docs/plano_avaliacao_redes_winograd.md Sec 1.2/5.1), so a freshly-constructed model gives the
exact same layer shapes a trained one would. Run: python -m scripts.winograd_fpga.dump_layer_configs
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import ml.model_registrations  # noqa: F401 -- populates MODEL_REGISTRY
from ml.registry import MODEL_REGISTRY
from ml.reporting import compute_flops
from ml.winograd_bridge import bridge_root, import_bridge_module

VARIANTS = ("f23", "f43", "f63")

# The 14 budget_unico (accelerator-eligible) networks -- see configs/experiments/budget_unico.yaml
# -- plus the 2 Phase 11 networks whose native kernel is already 3x3 (alexnet_tv_2x2/vgg16_2x2 are
# never eligible: the accelerator only computes 3x3 Winograd, so there's nothing to dump for them).
NETWORKS = [
    "vgg_style_fpga", "alexnet_stacked_fpga", "repvgg_a0_fpga", "vgg13_fpga",
    "alexnet_3x3_fc_fpga", "alexnet_fire_fpga", "alexnet_fire_bypass_fpga",
    "alexnet_bottleneck_fpga", "alexnet_final_fire_residual_fpga",
    "alexnet_final_bottleneck_residual_fpga", "googlenet_fpga", "resnet18_fpga",
    "wrn_16_4_fpga", "wrn_28_2_fpga",
    "alexnet_tv_3x3", "vgg16",
]

OUT_DIR_NAME = "layer_configs"


def _sanitize(name: str) -> str:
    """xelab snapshot names can't contain '.' -- Layer.name is a dotted module path."""
    return name.replace(".", "_")


def _import_layer_configs():
    """layer_configs.py lives in scripts/, one level above the avaliacao_redes/ that
    ml.winograd_bridge.import_bridge_module() targets -- so it needs its own sys.path entry."""
    scripts_dir = str(bridge_root().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import layer_configs
    return layer_configs


def _check_schema_matches_vgg16(net_manifest, layer_configs_mod) -> None:
    """net_manifest.py's own selftest invariant: its hand-written vgg16() must reproduce
    layer_configs.LAYER_CONFIGS field-for-field. Re-checked here because this script writes to
    that same schema -- if the two have drifted apart, the JSONs below would silently feed the
    RTL runners a wrong workload instead of failing loudly."""
    ref = layer_configs_mod.LAYER_CONFIGS
    got = {l.name: l for l in net_manifest.vgg16().layers}
    assert set(got) == set(ref), f"layer name mismatch vs layer_configs.LAYER_CONFIGS: {set(got) ^ set(ref)}"
    for name, cfg in ref.items():
        l = got[name]
        actual = (l.img_w, l.img_h, l.n_ic, l.n_of)
        expected = (cfg["img_w"], cfg["img_h"], cfg["n_ic_total"], cfg["n_of_total"])
        assert actual == expected, f"{name}: net_manifest.vgg16() gives {actual}, layer_configs.py expects {expected}"


def main() -> int:
    layer_configs_mod = _import_layer_configs()
    net_manifest = import_bridge_module("net_manifest")
    eligibility = import_bridge_module("eligibility_wino")
    models_torchvision_wino = import_bridge_module("models_torchvision_wino")

    _check_schema_matches_vgg16(net_manifest, layer_configs_mod)

    out_dir = bridge_root() / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for reg_name in NETWORKS:
        model = MODEL_REGISTRY[reg_name]["ctor"]()
        net = models_torchvision_wino.to_manifest(model, reg_name, input_size=64)
        total_macs = compute_flops(model, input_size=(1, 3, 64, 64))["macs"]

        for variant in VARIANTS:
            verdicts = eligibility.audit_net(net, variant)
            eligible = [(layer, v) for layer, v in zip(net.layers, verdicts) if v.ok]

            cfg = {
                _sanitize(layer.name): {
                    "img_w": layer.img_w, "img_h": layer.img_h,
                    "n_ic_total": layer.n_ic, "n_of_total": layer.n_of,
                    "ic_blk_default": layer.n_ic, "of_blk_default": 16,
                }
                for layer, _ in eligible
            }
            json_path = out_dir / f"{reg_name}_{variant}.json"
            json_path.write_text(json.dumps(cfg, indent=2))

            eligible_macs = sum(layer.macs() for layer, _ in eligible)
            reasons = sorted({r for _, v in zip(net.layers, verdicts) if not v.ok for r in v.reasons})
            summary_rows.append({
                "network": reg_name,
                "variant": variant,
                "layers_eligible": len(eligible),
                "layers_3x3_s1_total": len(net.layers),
                "macs_eligible": eligible_macs,
                "macs_model_total": total_macs,
                "macs_fraction": (eligible_macs / total_macs) if total_macs else None,
                "ops_strict_eligible": sum(layer.ops_strict() for layer, _ in eligible),
                "tile_eff": eligibility.net_tile_eff(net, variant),
                "min_output_size": min((layer.out_w for layer in net.layers), default=0),
                "ineligibility_reasons": "; ".join(reasons),
                "json_path": json_path.name,  # relative -- always sits next to summary.csv, on any machine
            })

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {len(NETWORKS)} x {len(VARIANTS)} = {len(summary_rows)} layer-config JSONs "
          f"+ {summary_path} under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
