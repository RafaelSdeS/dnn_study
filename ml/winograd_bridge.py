"""Bridge to the sibling Winograd-FPGA repo -- the only place this repo imports from it.

Division of labor for the Winograd-FPGA study (configs/experiments/budget_unico.yaml):
Winograd-FPGA owns *what* gets trained -- model geometry (models_custom_wino.py,
models_torchvision_wino.py), the §4 stride-2 conversion, and the accelerator-numeric QAT
(qat_wino.py); this repo owns *how* (Trainer, SLURM/PCAD, summaries). Building the study's
models from the sibling's own builders means a checkpoint trained here loads 1:1 into the
definition Fase 2.5 evaluates there. The native classes in models/ can't stand in for them:
same tensor shapes, different state_dict keys (checked 2026-09-11 for vgg_style,
alexnet_fire, alexnet_fire_bypass, alexnet_bottleneck, alexnet_3x3_fc).

Resolved from $WINOGRAD_FPGA_ROOT -- the sibling's scripts/avaliacao_redes locally, or on
PCAD the avaliacao_redes/ dir of the tarball its scripts/package_avaliacao_bridge_for_pcad.sh
builds -- default ~/Documents/Winograd-FPGA/scripts/avaliacao_redes.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from .quantization import load_best_model

DEFAULT_WINOGRAD_FPGA_ROOT = Path.home() / "Documents" / "Winograd-FPGA" / "scripts" / "avaliacao_redes"


def bridge_root() -> Path:
    return Path(os.environ.get("WINOGRAD_FPGA_ROOT", DEFAULT_WINOGRAD_FPGA_ROOT)).expanduser()


def import_bridge_module(name: str):
    root = bridge_root()
    if not (root / f"{name}.py").exists():
        raise FileNotFoundError(
            f"{name}.py not found at {root} -- set WINOGRAD_FPGA_ROOT to the "
            "Winograd-FPGA repo's scripts/avaliacao_redes directory"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(name)


def custom_model(name: str, convert: bool = True):
    """Registry ctor for models_custom_wino.<name>.

    convert=True applies convert_stride2_and_pools right after construction, so FP32
    training already fits the eligible architecture end to end (a 3x3/s2 already followed
    by a pool -- alexnet_3x3_fc's conv1, the final_* stems -- is left for the host by that
    function itself).
    convert=False is for RepVGG: its 3-branch reparam needs the stride-2 branches intact,
    so it trains raw. Caveat (checked 2026-09-11): convert_stride2_and_pools builds a FRESH
    conv for every one it replaces, so converting after training (to_wino's reparam ->
    convert) discards the 4 stride-2 blocks' trained weights. Either those 4 convs run on
    the host, or the block needs a redesign (pool after the branch sum) before training.
    """
    def build(num_classes: int = 200):
        model = getattr(import_bridge_module("models_custom_wino"), name)(num_classes)
        if convert:
            import_bridge_module("models_torchvision_wino").convert_stride2_and_pools(model)
        return model
    return build


def torchvision_model(name: str):
    """Registry ctor for models_torchvision_wino.<name>_wino -- those builders already
    apply the stem/stride/pool conversions themselves."""
    def build(num_classes: int = 200):
        return getattr(import_bridge_module("models_torchvision_wino"), f"{name}_wino")(num_classes)
    return build


def bridge_provenance() -> dict[str, Any]:
    """Which Winograd-FPGA code a run used: git hash/dirty flag of the sibling's scripts/
    tree, or on PCAD (a tarball, no .git) the BRIDGE_COMMIT.json the packaging script
    writes next to avaliacao_redes/."""
    root = bridge_root()
    stamp = root.parent / "BRIDGE_COMMIT.json"
    if stamp.exists():
        return {"root": str(root), **json.loads(stamp.read_text())}
    try:
        git_hash = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        git_dirty = bool(subprocess.run(["git", "-C", str(root.parent), "status", "--porcelain", "--", "."],
                                        capture_output=True, text=True, check=True).stdout.strip())
    except Exception:
        return {"root": str(root), "git_hash": None, "git_dirty": None}
    return {"root": str(root), "git_hash": git_hash, "git_dirty": git_dirty}


def load_qat_wino_model(model_name: str, spec: dict[str, Any], checkpoints_dir: Path, device,
                        cfg=None):
    """FP32 best checkpoint -> qat_wino.convert() in place.

    A fresh instance built this way every time (rather than caching) so its
    state_dict keys always match a qat_wino checkpoint saved from a model
    built the same way -- convert() adds buffers (act_absmax, post_shift,
    sat_frac, BT/AT/G) that a plain ctor() instance doesn't have.

    `cfg` is a QATWinoConfig and says WHICH accelerator line to train against
    (`variant`) and whether it packs 2 multiplications per DSP (`pack`). Until
    2026-09-18 this called `convert(model)` bare, which pinned F(4,3) without
    packing -- that is why every accuracy number in the study so far is that one
    combination. `cfg=None` keeps that old behaviour, so a caller that has not
    been updated reproduces the old runs instead of silently changing numerics.
    """
    qat_wino = import_bridge_module("qat_wino")
    model = load_best_model(model_name, spec["ctor"], checkpoints_dir, device, eval_mode=False)
    kw: dict[str, Any] = {}
    if cfg is not None:
        kw["variant"] = cfg.variant
        # `pack` default no convert() e' True (o hardware empacota); aqui o valor
        # e' SEMPRE explicito, para que o que roda seja o que o YAML declara.
        kw["pack"] = bool(cfg.pack)
        if cfg.pack:
            kw.update(u_w=cfg.u_w, v_w=cfg.v_w, k_dsp=cfg.k_dsp)
        if getattr(cfg, "hw_params", None):
            kw["hw_params"] = cfg.hw_params   # escalas do netlist, nao as da camada
    else:
        kw["pack"] = False
    # Logger NOMEADO, igual ao resto deste arquivo: `logging.info` no logger raiz
    # e' filtrado pela configuracao do runner e a linha some. E ela nao e'
    # decorativa — e' o unico registro, no log da corrida, de CONTRA QUAL
    # hardware aquele treino foi feito.
    logging.getLogger(f"pcad_runner.{model_name}").info(
        "qat_wino.convert(%s)", ", ".join(f"{k}={v!r}" for k, v in kw.items()))
    trocadas = qat_wino.convert(model, **kw)
    if not trocadas:
        raise RuntimeError(f"{model_name}: qat_wino.convert() found no eligible 3x3 conv to replace")
    # Safety net for the class of bug this stage almost shipped with: convert()
    # silently SKIPS any 3x3 conv with stride/groups/dilation != 1 (see its own
    # "nao elegivel: converta antes (SS4)" comment) rather than erroring. A model
    # registered without custom_model(convert=True) would train qat_wino fine with
    # that layer silently left un-Winograd'd -- no exception, just a wrong number.
    ineligible = [n for n, m in model.named_modules()
                  if isinstance(m, torch.nn.Conv2d) and m.kernel_size[0] == 3
                  and (m.stride[0] != 1 or m.groups != 1 or m.dilation[0] != 1)]
    if ineligible:
        logging.getLogger(f"pcad_runner.{model_name}").warning(
            "%s: %d 3x3 conv(s) left un-Winograd'd by qat_wino.convert() "
            "(stride/groups/dilation not eligible): %s -- expected for host-bound s2+pool "
            "convs (alexnet_3x3_fc's conv1, the final_* stems) and raw RepVGG's stride-2 "
            "blocks; anything else needs ml.winograd_bridge.custom_model(convert=True)",
            model_name, len(ineligible), ineligible)
    logging.getLogger(f"pcad_runner.{model_name}").info(
        "qat_wino converted %d conv layer(s): %s", len(trocadas), trocadas)
    return model.to(device)


def dump_wino_calibration(model) -> dict[str, Any]:
    """Per-layer act_absmax/post_shift/sat_frac -- the §5.1 calibration fields
    the plan calls irrecoverable without a re-eval. Duck-typed (not isinstance)
    because qat_wino.make_wino_conv() builds its conv class inside a closure.
    """
    calib: dict[str, Any] = {}
    for name, m in model.named_modules():
        if hasattr(m, "act_absmax") and hasattr(m, "post_shift") and hasattr(m, "sat_frac"):
            calib[name or "root"] = {
                "act_absmax": float(m.act_absmax.item()),
                "post_shift": int(round(float(m.post_shift.item()))),
                "sat_frac": float(m.sat_frac.item()),
            }
    return calib
