# Phase 8 Implementation Log

Decision continuity across `/compact` boundaries. Append per stage.

Note: "Stage N" below numbers Phase 8's own internal build-out steps — unrelated
to the project's Phase 1-9 numbering in `CLAUDE.md`. Every stage here belongs to
Phase 8. Maps to `ideas/PHASE8_PLAN.md`'s Task numbering where noted.

---

## Stage 1 — Model Architectures (Task 1) ✓

**Commit:** `842839d` feat(phase8): add vit_tiny/swin_pico/hybrid_bottleneck_swin/poolmixer architectures

- `models/vit_variants.py`: `vit_tiny` (also `deit_tiny`'s architecture, D6),
  `swin_pico(window_size)` for the H1 sweep, `hybrid_bottleneck_swin` (H2),
  `swin_pico_poolmixer` (D5/H5) — thin wrappers around torchvision's
  `VisionTransformer`/`SwinTransformer`, sized per D3 for 64x64 Tiny ImageNet-200
- Not yet registered in `MODEL_REGISTRY` or QAT-adapted at this point

## Stage 2 — Registry Integration + Optimizer Overrides (Task 2, Task 4 pitfall) ✓

**Commit:** `6f743ca` feat(phase8): register 7 Phase 8 models; wire weight_decay override + AdamW warmup

- All 7 models registered with `lr=5e-4, weight_decay=0.05` (DeiT's recipe, not this
  project's CNN-tuned defaults, per Task 4's pitfall)
- `scripts/train.py` previously only read `spec.get("lr", ...)` back out of the
  registry — added `weight_decay` the same way
- `TrainerConfig.warmup_epochs` + `ml/trainer.py`'s `LinearLR`+`SequentialLR`
  composition (Blocking Issue #5) — stdlib `torch.optim`, no new dependency

## Stage 3 — QAT Adaptation, First Pass (Task 3, D6 original) ✓ (later revised, Stage 7)

**Commit:** `207cc0a` feat(phase8): add exclude_attention_from_qat/swap_quantizable_mha for QAT (D6)

- `ml/quantization.py`: `exclude_attention_from_qat` (LayerNorm/ShiftedWindowAttention
  → `qconfig=None`) wired unconditionally into `prepare_qat_model()`
- `swap_quantizable_mha` (`nn.MultiheadAttention` → `torch.ao.nn.quantizable.MultiheadAttention`,
  hand-split `in_proj_weight`/`bias` into `linear_Q/K/V`) for the ViT/DeiT path
- This pass's self-check (weight-transfer `allclose`) wasn't run against the real
  torchvision `EncoderBlock`'s `batch_first=True` usage — see Stage 7

## Stage 4 — Config/CLI Integration (Task 6) ✓

**Commit:** `6b5a745` feat(phase8): add configs/experiments/phase8.yaml

- `configs/experiments/phase8.yaml` covers the 5 models whose full FP32/QAT/INT8
  pipeline works through `scripts/train.py`'s generic loop unmodified
  (`swin_pico_w2/w4/w8`, `swin_pico_poolmixer`, `hybrid_bottleneck_swin`)
- `vit_tiny`/`deit_tiny` excluded here, documented as needing a notebook instead
  (not yet built at this point — see Stage 8)

## Stage 5 — DistillationTrainer (Task 4, H4) ✓

**Commit:** `ec8f820` feat(phase8): add DistillationTrainer for deit_tiny (H4)

- `ml/distillation_trainer.py`: subclasses only `_train_one_epoch` — hard-label
  distillation (CE against the frozen teacher's argmax, not soft KL, per Touvron
  et al. 2021 Table 4), same "subclass only the step" precedent as
  `det_seg_trainer.py`'s `DetectionTrainer`
- `demo()` asserts the loss is non-zero and the teacher's parameters never change

## Stage 6 — Quantization Boundary-Crossing Fixes ✓

**Commits:** `1cca8af`, `b94db25`, `cb2fec8`

- Fixed `hybrid_bottleneck_swin`'s stem: 4 halvings (64→32→16→8→4) where every
  comment/assert assumed 3 (64→32→16→8) — two stride-2 `_AlexBottleneck` blocks
  plus two `MaxPool2d(2)` calls, corrected to one `MaxPool2d`
- `_QuantizableSwinBlock`/`_QuantizableMLP`: residual adds via `FloatFunctional`
  (this project's QAT convention) instead of bare `+`, since after `convert()` an
  excluded LayerNorm's FP32 output would otherwise mix with a quantized Linear
  output in the same add
- Extended the same fix to every remaining boundary crossing in the two
  torchvision classes used off the shelf (an excluded LayerNorm feeding a still-
  quantized Linear, or `torch.cat`/`+` mixing an always-FP32 parameter with a
  quantized activation)

---

## Stage 7 — swap_quantizable_mha Fixes, Then Revision of D6 (Task 3, revised) ✓

**Session:** 2026-08-21 (no commit hash yet at time of writing — see repo history for
the actual commits made after this log entry)

Two bugs found by direct testing while verifying Stage 3's Task 3 self-check (which had
never actually been run against a real forward pass before this session):

1. **`batch_first=True` reshape bug** in `torch.ao.nn.quantizable.MultiheadAttention`
   (torch 2.5.1): diverged from `nn.MultiheadAttention` by ~0.7 max-abs-diff on random
   input; the identical module built `batch_first=False` matched to 0.0. Fixed by always
   constructing the quantizable module `batch_first=False` and wrapping it in
   `_BatchFirstMHAWrapper`, which transposes at the boundary instead of trusting the
   module's own flag.
2. **Custom-module/`prepare_qat()` ordering crash**, found immediately after fixing #1:
   `build_qat_from_model()` on `vit_tiny` (post-swap) raised `AttributeError: 'Linear'
   object has no attribute 'activation_post_process'`. Root cause (confirmed by reading
   PyTorch's actual source, not guessing): `torch.ao.nn.quantizable.MultiheadAttention`
   is registered in PyTorch's default `observed_to_quantized_custom_module_class`
   mapping. `tq.prepare_qat()` does `convert()` *then* `prepare()` — the first `convert()`
   call matches the swapped-in module via that mapping and calls its `.from_observed()`
   classmethod before any observer exists (that only happens in the second, `prepare()`,
   step). This module is only designed for the static-PTQ `prepare()`→calibrate→`convert()`
   flow, not this codebase's single-call `prepare_qat()`.

Presented both findings plus two options (extend `exclude_attention_from_qat` to also
exclude `nn.MultiheadAttention` via `qconfig=None`, matching Swin's fallback; or build a
separate FX graph-mode QAT path) to the user — chose the `qconfig=None` fallback.
Implementing it surfaced a *third* issue: `models/vit_variants.py`'s
`_QuantizableEncoderBlock` already bracketed `self_attention` with `QuantStub`/`DeQuantStub`,
built on the (now-invalidated) assumption that `swap_quantizable_mha()` would run — feeding
a real quantized tensor into the now-plain-FP32 `nn.MultiheadAttention` crashed INT8
conversion (`NotImplementedError: aten::mm.out` on `QuantizedCPU`). Removed the stub
bracketing; `self_attention` is now treated exactly like `ln_1`/`ln_2` — a plain excluded
FP32 submodule needing no stubs either side.

Verified end-to-end on `vit_tiny` (not just synthetic checks): `swap_quantizable_mha`'s
weight-transfer self-check passes; `exclude_attention_from_qat` correctly excludes
`LayerNorm`/`ShiftedWindowAttention`/`MultiheadAttention`; a real `nn.MultiheadAttention`-
containing model survives `prepare_qat_model()` (new regression test in `ml/quantization.py`'s
`demo()`); full FP32→QAT→INT8 forward pass; a real training step (forward + backward,
80/80 params receiving gradients). `swap_quantizable_mha` is kept in `ml/quantization.py`
(independently correct, useful if FX-mode QAT is ever pursued) but is off the QAT call path.

Amended `ideas/PHASE8_PLAN.md`'s D6 section in place to document this outcome.

## Stage 8 — FLOPs Verification (Task 5, D7) ✓

Confirmed `fvcore`'s default handlers have no entry for
`aten::scaled_dot_product_attention` (which `nn.MultiheadAttention` dispatches to
directly) — `compute_flops()` was silently dropping every attention layer's FLOPs.
Added `_sdpa_flop_jit`, a custom `fvcore` op handler, to `ml/reporting.py`. Verified
exact match against the standard transformer FLOP-counting formula
(`2 * num_heads * seq_len^2 * head_dim` per layer): 9,734,400 MACs delta on `vit_tiny`,
precise match, not just close. No-op for every pre-Phase-8 model (none contain the op).

## Stage 9 — Notebook (Task 6, `vit_tiny`/`deit_tiny`) ✓

`notebooks/phase_8_efficient_vit/vit_qat_phase8.ipynb`:
registration, blocking-issue pre-flight checks (teacher checkpoint, attention
exclusion, FLOPs verification — all as runnable cells, not just prose), model
shape/param-count sanity check, FP32 training (`vit_tiny` plain `Trainer`, `deit_tiny`
`DistillationTrainer` against the existing `mobilenetv2_best.pth` teacher), QAT (now
just `build_qat()`, same call as every other Phase 8 model per Stage 7's revision),
INT8 conversion, comparison table + summary persistence. Every non-training cell
dry-run verified as a standalone script before being written into the notebook; no
notebook execution or training run this session (per `CLAUDE.md`'s workflow rule).

## Stage 10 — Docs + PCAD Submission

`CLAUDE.md` and `ideas/PHASE8_PLAN.md` updated to reflect the above; this log added.
5 CLI-drivable models (`swin_pico_w2/w4/w8`, `swin_pico_poolmixer`,
`hybrid_bottleneck_swin`) submitted to PCAD via
`scripts.cluster submit-sweep --experiment phase8 --runtime pcad` (`tupi_4090`, one
job per model). `vit_tiny`/`deit_tiny` still need the notebook run manually (not a
`sbatch`-submittable job in this project's current tooling).

## Stage 11 — Checkpoint-Restore Bug Found; FP32 Backfilled (2026-08-29)

All 7 models finished training and Task 7's cross-phase comparison notebook ran
(`results/phase_8_efficient_vit_hybrid_attention_analysis/phase8_comparison.csv`). While
writing up results, the swin/hybrid models' apparent INT8 "gains" turned out to be an
artifact: `ml/trainer.py`'s `Trainer.fit()` returned with `self.model` still holding the
**last** epoch's weights instead of reloading `{run_name}_best.pth` before returning.
`scripts/train.py`'s post-fit `evaluate()` call therefore measured different weights
than `build_qat()`/`load_best_model()` picked up for the QAT stage — FP32 was scored on
the last epoch, INT8 on (a QAT run starting from) the best epoch. For runs with a long
post-peak early-stopping tail this read as spurious INT8 accuracy gains of up to +6.5pp.

Fixed in `ml/trainer.py` (`fit()` now reloads `best_path` into `self.model` before
returning; also fixes `convert_to_int8(qat_model)`'s input to use the best QAT epoch,
matching what `scripts/train_det_seg.py`'s INT8 stage already did explicitly). Regression
test: `tests/test_trainer_smoke.py::test_fit_restores_best_checkpoint_not_last_epoch`
(scripts a decaying val accuracy, asserts the live model matches the saved best
checkpoint, not the resume checkpoint, after `fit()` returns).

`scripts/backfill_best_epoch_eval.py` re-evaluates FP32 from the surviving
`{model}_best.pth` for the 5 CLI-trained Phase 8 models (`vit_tiny`/`deit_tiny` used the
notebook's `load_best_model()` path directly and were never affected) plus Phase 9's
`alexnet_fire_bypass` — no retraining, just re-scoring already-saved checkpoints. INT8
was only rebuilt where a full-precision QAT-best checkpoint also survived; for the 5
Phase 8 models it didn't (only the already-converted last-epoch INT8 model was kept), so
their INT8 numbers are left as previously measured — see the script's docstring for the
`alexnet_fire_bypass` case, which has its own separate reason INT8 couldn't be rebuilt
either. Corrected numbers are in Table `tab:phase8_models` (Eixo 7) of `report/ic_report.tex`
and `ideas/BEST_MODELS.md`'s Phase 8 section.

## Stage 12 — Model-Size Measurement Bug Found; FP32 Sizes Backfilled (2026-09-02)

A sibling bug to Stage 11, found in the same results-write-up pass: `ml/reporting.py`'s
`disk_mb()`/`gzip_mb()` measured the raw `{name}_best.pth` file. `save_checkpoint()` writes
that file with the optimizer + scheduler + history alongside the weights, and AdamW alone
keeps two momentum buffers per parameter — so the file is ~3x the model's actual weight
size. The INT8 side is a bare `torch.save(model, ...)` and was already weights-only. Every
FP32-vs-INT8 size comparison was therefore apples-to-oranges: `compression_ratio` came out
~3x too high (~11.9x recorded where the true FP32→INT8 ratio is ~4x). Accuracies, params,
and MACs were unaffected — this only touched the size/compression columns.

Fixed in `ml/reporting.py` (`disk_mb()`/`gzip_mb()` now unwrap `model_state_dict` before
measuring, via a new `_model_bytes()` helper shared by both). `scripts/backfill_model_size.py`
re-measures `fp32_size_mb`/`fp32_gzip_mb` in every summary JSON: from the surviving
checkpoint where one exists, else by rebuilding the architecture from `MODEL_REGISTRY` (exact,
since state_dict size depends only on the architecture) or falling back to `/3.0` when neither
is available. `--csvs` patches the same correction into the derived comparison CSVs. Phase 7
detection/segmentation was out of scope: those checkpoints never carried optimizer state (their
`model_size_mb` already sat within 4% of the separately-measured `true_size_mb`), so nothing
there needed correcting. Corrected numbers are reflected in `CLAUDE.md`'s Results & rankings
summary, `results/results_aggregate/`, and `report/ic_report.tex`.
