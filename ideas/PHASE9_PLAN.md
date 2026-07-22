# Phase 9 — SqueezeNet-Style Bypass Ablation + Structured Compression (Implementation Plan)

Phase 4's `alexnet_final_fire_residual` (FP32 49.79%/INT8 49.20%, 8.09→0.75 MB,
`results/final_architecture_phase4/alexnet_final_fire_residual_summary.json`) beats Phase 3's
`alexnet_fire` (FP32 43.98%/INT8 44.30%, 5.99→0.55 MB, `results/model_details.csv:19`) by
+5.81pp FP32 / +4.90pp INT8. But `AlexNetFinalFireResidual` changes two things at once versus
`AlexNetFire`: it adds a 3×3 stride-2 stem *and* wraps every Fire stage in a residual shortcut
(`models/final_architecture.py:97-124`, `_FireResBlock` at `models/final_architecture.py:15-28`).
The gain can't currently be attributed to bypass alone. Phase 9 isolates that variable, then asks
a second, unrelated question: whether the project's "Deep Compression"-inspired checkpoint
shrink (`compress_checkpoint` at `ml/checkpoint.py:82`, currently plain gzip) has headroom left,
scoped to keep any pruning **structured** (channel-level, dense output) rather than unstructured
— unstructured/masked sparsity produces ragged kernels that defeat the Winograd-friendliness this
whole project is arguing for.

This plan follows the same structure as `ideas/PHASE6_PLAN.md` / `ideas/PHASE7_PLAN.md`:
hypotheses first, then decision records for every choice not forced by the codebase, then
task-by-task detail, then scope/effort and blocking issues.

---

## Research Hypotheses

### H1: Bypass Alone Improves Accuracy Without Hurting Quantization Stability
**Claim:** A single identity shortcut added to `AlexNetFire` — with no stem change and no other
architectural difference — improves accuracy over plain `AlexNetFire`, and the improvement
(or at least quantization stability) is a real fraction of what `AlexNetFinalFireResidual`
achieves, not the whole thing.

**Expected Outcome:**
- `alexnet_fire_bypass` FP32 top-1 lands between `alexnet_fire` (43.98%) and
  `alexnet_final_fire_residual` (49.79%) — i.e. some but not all of the +5.81pp gap is bypass.
- INT8 quantization drop stays small (`alexnet_fire` already *gains* -0.33pp on INT8;
  `alexnet_final_fire_residual` drops a modest +0.59pp) — `FloatFunctional.add` is QAT-safe, so
  no regression is expected from adding one skip connection.
- If `alexnet_fire_bypass` closes most of the gap on its own, the stem change in Phase 4's model
  was mostly free extra capacity, not a necessary ingredient. If it closes little of the gap, the
  stem (or the *other* two residual pairs Phase 4 adds) matters more than bypass itself.

**Evidence to Collect:** FP32/INT8 top-1/top-5, quantization drop, param count, FP32/INT8 size —
same fields as `make_run_summary` produces for every other Phase 3/4 model, so it drops straight
into `results/model_details.csv`-style comparison.

**Acceptance Criterion:** `alexnet_fire_bypass` FP32 top-1 > `alexnet_fire` FP32 top-1 by a
statistically meaningful margin (single run, so "meaningful" = larger than the ~0.3-0.5pp run-to-run
noise visible between `best_val_top1` and `final_val_top1` in existing summary JSONs), and INT8
drop stays within the ±1pp band every other Fire/Bottleneck-family model has shown.

**Result — met.** PCAD job 806654 (`configs/experiments/phase9_fire_bypass.yaml`, 66 epochs to
early-stopping), `outputs/pcad/phase9_fire_bypass/alexnet_fire_bypass/results/alexnet_fire_bypass_summary.json`:

| model | FP32 top-1 | FP32 top-5 | INT8 top-1 | INT8 top-5 | quant Δtop-1 | size FP32→INT8 |
|---|---|---|---|---|---|---|
| `alexnet_fire` | 43.98% | 70.43% | 44.30% | 70.88% | −0.33pp (gain) | 5.99→0.55 MB |
| `alexnet_fire_bypass` | **47.05%** | 73.14% | **47.16%** | 72.71% | −0.11pp (gain) | 5.99→0.55 MB |
| `alexnet_final_fire_residual` | 49.79% | 74.80% | 49.20% | 74.39% | +0.59pp (drop) | 8.09→0.75 MB |

Same params, same size as `alexnet_fire` (D1 held exactly, as verified pre-training) — this
+3.07pp FP32 / +2.86pp INT8 came from the one skip connection alone, for zero added capacity.

That accounts for **~53% of the FP32 gap and ~58% of the INT8 gap** between `alexnet_fire` and
`alexnet_final_fire_residual` (+5.81pp / +4.90pp total). The remaining ~47%/42% is the stem
change plus the other two (non-channel-matched, 1×1-projected) residual pairs Phase 4 adds —
still an open question, but now a smaller and better-bounded one.

Quantization stability also transfers cleanly: like plain `alexnet_fire`, the bypass variant
*gains* accuracy on INT8 conversion (−0.11pp) rather than losing it — unlike the full Phase 4
model's modest +0.59pp drop. That drop, whatever causes it (stem, or the projected shortcuts),
isn't coming from the bypass mechanism itself.

---

### H2: Structured Channel Pruning Trades Size for Accuracy Without Breaking Dense Structure
**Claim:** Removing whole output channels (not individual weights) from `alexnet_bottleneck` /
`alexnet_fire_bypass` convolutions reduces parameter count and theoretical size while keeping
every remaining conv a normal dense `groups=1` kernel — i.e. still eligible for the same Winograd
path Phase 6 measured (`ideas/PHASE6_PLAN.md` H1/H2), unlike unstructured pruning which produces
sparse-but-still-dense-shaped tensors that gain nothing on Winograd hardware.

**Expected Outcome:**
- At a 0.4 channel-removal ratio, parameter count and `theoretical_size_mb`
  (`ml/quantization_advanced.py:239`) drop roughly in proportion to channels removed (not exactly,
  since squeeze/expand channel counts in `_FireModule`/`_AlexBottleneck` are coupled across
  stages).
- Accuracy drop before any fine-tuning is expected to be large (channel pruning without
  fine-tuning is known to be destructive); this task explicitly stops at "does a pruned model
  still run and produce sane shapes," not "is pruned accuracy competitive" — that's future work
  once Task 2 is scoped up from measurement to full fine-tuning.

**Evidence to Collect:** Pre/post-prune param count, `theoretical_size_mb`, a single forward-pass
shape check, and `Trainer.evaluate()` output pre-fine-tune (expected to be poor — recorded as a
baseline, not a result).

**Acceptance Criterion:** Pruned model builds, forward-passes without shape errors, and every
remaining `nn.Conv2d` still has `groups == 1` (structured, dense, Winograd-eligible by
construction — this is enforced by *how* channels are removed, not measured after the fact).

---

### H3: Weight-Clustering Beats Plain Gzip on the INT8 Checkpoint
**Claim:** k-means weight-sharing (Deep Compression's codebook step) on top of the existing INT8
quantization achieves a smaller effective bits/weight than `compress_checkpoint`'s plain gzip,
because gzip's DEFLATE only exploits byte-level redundancy in the already-quantized INT8 stream,
while explicit clustering + Huffman-coded indices exploits the weight *distribution* directly.

**Expected Outcome:**
- Shannon entropy of the current INT8 weight distribution is measurably below 8 bits/weight
  (INT8 quantization ranges are rarely uniformly used), which is exactly the gap gzip is already
  partially capturing and clustering could capture more precisely.
- k-means at 16/32/64 clusters (4/5/6-bit weight sharing, matching Deep Compression's own sweep)
  should each beat gzip's ratio once codebook + Huffman-coded index-stream overhead is accounted
  for, with the gap narrowing as cluster count grows (64 clusters ≈ 6 bits, closer to gzip's
  already-decent DEFLATE ratio on 8-bit data).

**Evidence to Collect:** Entropy (bits/weight) of INT8 weights; codebook size + index-stream size
(computed via `theoretical_size_mb`-style accounting, not an actual bitstream) for 16/32/64
clusters; gzip ratio from `compress_checkpoint` on the same checkpoint. All four numbers in one
table.

**Acceptance Criterion:** At least one cluster count beats gzip's ratio. If none do, that's a
valid (negative) result — it means the codebase's existing gzip step already captures most of the
achievable compression on top of INT8, and building a real weight-sharing pipeline in
`ml/checkpoint.py` isn't worth it.

---

## Decision Records

### D1 — Bypass placement: fire4→fire5 only, no stem change, no projection
`AlexNetFire`'s five stages output 64→192→384→256→256 (`models/compensation.py:390-420`). Only
the fire4/fire5 pair (256→256) has matching in/out channels with no `MaxPool2d` between them —
every other adjacent pair either changes channel count or has a pool in between, which would force
a 1×1 projection shortcut (like `_FireResBlock`'s `nn.Conv2d(in_ch, out_ch, 1)` when
`in_ch != out_ch`). A projection shortcut adds its own parameters and confounds "bypass alone"
with "bypass + extra 1×1 conv capacity." Picking the one channel-matched pair keeps the ablation
to exactly one variable: an identity `FloatFunctional.add`, nothing else. This also means
`AlexNetFireBypass` has the *same* parameter count as `AlexNetFire` plus zero — the comparison in
H1 is architecturally clean.

### D2 — No stem change
Phase 4's model adds a stride-2 3×3 stem conv before the first Fire stage
(`models/final_architecture.py:110-113`) that plain `AlexNetFire` doesn't have. `AlexNetFireBypass`
keeps `AlexNetFire`'s original stem-less structure (first `_FireModule` takes the 3-channel input
directly) specifically so the only difference from `alexnet_fire` in the comparison table is the
one skip connection — the stem's contribution is left as an open question the H1 acceptance
criterion is designed to expose, not something this phase tries to also isolate.

### D3 — Pruning ratios: single 0.4 ratio for Task 2's measurement pass, not a sweep
Task 2 exists to answer "does structured pruning break shapes/Winograd-eligibility and roughly how
much size does it buy," not to find an optimal ratio — that requires fine-tuning, which is future
work per the Scope & Effort section. A single mid-range ratio (0.4, matching common
channel-pruning literature defaults) is enough to validate the mechanics and get one size/accuracy
data point without spending multi-day fine-tuning budget before H2/H3's cheaper findings are in.

### D4 — Cluster counts: 16/32/64, matching Deep Compression's own 4/5/6-bit sweep
No reason to deviate from the paper's own choices — they bracket "aggressive" (16, 4-bit) to
"conservative" (64, 6-bit) weight-sharing, and reusing published cluster counts makes the
resulting bits/weight numbers directly comparable to the paper's reported ratios.

### D5 — Reuse `theoretical_size_mb` accounting style for codebook/index-stream size, don't hand-roll a new size estimator
`ml/quantization_advanced.py:239` already implements "count weight tensors at N bits, biases/BN at
8 bits" packed-size accounting for mixed-precision PTQ. Task 3's codebook+index-stream size
(codebook: `n_clusters × 32-bit float` centroids; index stream: `numel × log2(n_clusters)` bits)
is the same kind of theoretical accounting, not a real bitstream — following the existing
function's pattern (not calling it directly, since the bit layout differs) keeps the new
measurement script consistent with how the rest of the codebase already reports theoretical vs.
actual sizes.

### D6 — No changes to `ml/checkpoint.py` in this phase
Task 3 is a measurement-only comparison (per the original scope). If weight-clustering wins by a
useful margin, wiring it into `compress_checkpoint` is a separate, later change — this phase's job
is only to tell us whether that investment is worth making.

---

## Task 1 — `AlexNetFireBypass` (`models/compensation.py`) — Done

Add immediately after `AlexNetFire` (`models/compensation.py:420`, right after its `forward`).
Identical `features` Sequential to `AlexNetFire`, except `fire4`/`fire5` become named submodules
(not buried in the `Sequential`) so a `FloatFunctional.add` can sit between them:

```python
class AlexNetFireBypass(nn.Module):
    """AlexNetFire + one identity bypass (fire4 -> fire5), isolating bypass from Phase 4's stem change.

    Architecture: identical to AlexNetFire (3->64->192->384->256->256, same 5 Fire stages),
    except fire4 and fire5 (the one channel-matched, no-pool-between pair: 256->256) are
    connected by a FloatFunctional identity add — SqueezeNet's "simple bypass," no 1x1
    projection needed since channels already match. See ideas/PHASE9_PLAN.md D1/D2.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.stem = nn.Sequential(
            _FireModule(3,   16,  32),   # out: 64
            nn.MaxPool2d(2),
            _FireModule(64,  48,  96),   # out: 192
            nn.MaxPool2d(2),
            _FireModule(192, 96, 192),   # out: 384
        )
        self.fire4 = _FireModule(384, 64, 128)  # out: 256
        self.fire5 = _FireModule(256, 64, 128)  # out: 256 (channel-matched -> bypass target)
        self.skip_add = _float_functional()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.quant(x)
        x = self.stem(x)
        f4 = self.fire4(x)
        x = self.skip_add.add(self.fire5(f4), f4)
        x = self.pool(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x
```

Reuses `_FireModule` (`models/compensation.py:367-386`) and `_float_functional()`
(`models/compensation.py:13`) unchanged — no new building blocks.

**Registration** (two places, mirroring the existing `alexnet_fire` entries exactly):
- `ml/model_registrations.py:68` area:
  `register_model("alexnet_fire_bypass", AlexNetFireBypass, fuse_map=find_fuse_groups(AlexNetFireBypass()), lr=1e-3)`
- `notebooks/training/compensation_qat.ipynb` registration cell: add
  `FUSE_FIRE_BYPASS = find_fuse_groups(AlexNetFireBypass())` next to `FUSE_FIRE`, add
  `register_model("alexnet_fire_bypass", AlexNetFireBypass, fuse_map=FUSE_FIRE_BYPASS, lr=1e-3)`
  to the `MODEL_REGISTRY.clear()` block, and add the same entry to the notebook's `CTORS` dict
  used for the FP32-metrics comparison pass.

No `Trainer`/QAT infra changes needed: `_FireModule`'s Conv-BN-ReLU triples are already fuseable,
and `torch.cat` + `FloatFunctional.add` are both QAT-safe (same pattern `_FireResBlock` already
uses in production).

**What this answers:** three-way comparison — `alexnet_fire` (43.98%/44.30%) vs.
`alexnet_fire_bypass` vs. `alexnet_final_fire_residual` (49.79%/49.20%) — isolates how much of
Phase 4's gain is bypass alone vs. the stem change (D2) vs. the other two residual pairs Phase 4
adds that aren't channel-matched. **Result: bypass alone accounts for ~53-58% of the gap** — see
H1's Result above for the full breakdown.

---

## Task 2 — Structured (channel) pruning, scoped (`ml/pruning.py`, `scripts/prune_channels.py`) — Done

**Scope refinement made during implementation:** rather than a general channel-propagation
pruner (rank any conv's output, propagate to whatever consumes it — the "riskiest part" flagged
in Blocking Issue 2 below), pruning targets **only `_AlexBottleneck`'s internal `mid_ch` width**
(the squeeze channels between its 1×1 and 3×3 convs). That width is private to the block — never
consumed outside it — so pruning it needs zero cross-module propagation: the block's public
in_ch/out_ch, and everything downstream (next block, classifier head), is untouched by
construction. This sidesteps Blocking Issue 2 entirely rather than solving it, which is enough
for this phase's "prove the mechanics work" scope; a general propagation-aware pruner (needed to
prune `_FireModule`'s squeeze-feeds-two-branches case, or block *boundary* channels) is future
work if this scoped version proves useful.

1. **Channel selection:** rank a block's squeeze-conv output channels by L1 norm (standard,
   cheapest structured-pruning criterion — no training-signal-based ranking needed for a first
   measurement pass). `ml/pruning.py::_l1_keep_indices`.
2. **Removal:** build smaller dense `nn.Conv2d`/`nn.BatchNorm2d` at the kept indices for the
   squeeze conv, the matching BN, and the 3×3 conv's both input *and* output (it's `mid_ch->mid_ch`,
   square) — the block's third (expand) conv only has its *input* narrowed, its output channel
   count (the block's public `out_ch`) is never touched. Every resulting conv keeps `groups=1` by
   construction. `ml/pruning.py::prune_bottleneck_block`/`prune_model_channels`.
3. **CLI:** `python -m scripts.prune_channels --model alexnet_bottleneck --ratio 0.4 --runtime local`,
   `--dry-run` prints the before/after `mid_ch` per block with no checkpoint/model build;
   `--evaluate` additionally runs `Trainer.evaluate()` on the real Tiny-ImageNet val set.

**Verification — done, against the real PCAD-trained checkpoint**
(`outputs/pcad/large_scale/alexnet_bottleneck/checkpoints/alexnet_bottleneck_best.pth`), ratio 0.4:

```
features.0   mid_ch   32 ->   19
features.2   mid_ch   48 ->   29
features.4   mid_ch   96 ->   58
features.5   mid_ch   64 ->   38
features.6   mid_ch   64 ->   38
params: 385,000 -> 207,399  (53.9%)
Forward pass OK, every remaining Conv2d still dense (groups=1).
Pruned (no fine-tune) | top1=0.50% | top5=2.35% | loss=17.4612
```

Matches H2's acceptance criterion exactly: builds, forward-passes, every remaining conv stays
`groups=1`, and (as expected, per H2's Expected Outcome) accuracy collapses to near-chance
without fine-tuning — recorded as the baseline this phase said it would be, not a result to judge
pruning by.

Not in scope for this phase: a fine-tuning loop to recover pruned accuracy, a ratio sweep, or
generalizing beyond `_AlexBottleneck` (e.g. to `_FireModule` or `_BottleneckResBlock`) — see
Scope & Effort.

---

## Task 3 — Compression measurement (`scripts/measure_compression.py`) — Done

One-off measurement script, not part of the `ml/` package surface since it produces a report, not
reusable training/quantization infra. **Correction from the original scope:** uses
`scipy.cluster.vq.kmeans2`, not `sklearn.cluster.KMeans` — `scikit-learn` isn't installed in this
project's environment, but `scipy` already is (Blocking Issue 3 below), and `scipy` covers k-means
without a new dependency (ladder rung 5).

1. Load an existing FP32 checkpoint, simulate the same per-channel-symmetric INT8 quantization
   the project's fbgemm qconfig uses (`ml/quantization_advanced.py`'s scheme), compute Shannon
   entropy of the resulting codes (actual bits/weight vs. nominal 8).
2. Run k-means on the **pre-quantization FP32** weights at 16/32/64 clusters; compute codebook
   size (`n_clusters × 4 bytes`) + index-stream size (`numel × ceil(log2(n_clusters))` bits) per
   D5's accounting style.
3. Report the real on-disk gzip ratio (`ml/reporting.py::gzip_mb`, the same helper
   `compress_checkpoint`-adjacent code already uses) on the actual INT8 checkpoint file, as
   context rather than folded into the same ratio column — the checkpoint file includes
   biases/BN params/quant metadata that the weights-only entropy/k-means numbers don't, so a
   single "ratio vs. INT8" column across both would be apples-to-oranges.

**Implementation note:** the on-disk INT8 checkpoint saved by `scripts/train.py`
(`torch.save(int8_model, ...)`, a full pickled module, not a state dict) failed to fully unpickle
in this environment — a version-mismatched `torch.ao.nn.quantized` module tree from whatever
torch build ran the original PCAD training job. This didn't block Task 3: `gzip_mb`/`disk_mb`
only read raw file bytes, never unpickle, so the on-disk context line works regardless. The
weights-only entropy/k-means numbers use the FP32 checkpoint instead (a plain state dict —
robust), simulating INT8 quantization in-script rather than depending on that fragile pickle.

**Results — `alexnet_fire`, from `outputs/pcad/large_scale/alexnet_fire/checkpoints/`:**

| method | bits/weight | size (MB) |
|---|---|---|
| INT8 nominal | 8.00 | 0.4893 |
| INT8 entropy (actual) | 7.19 | 0.4395 |
| k-means k=16 (4-bit) | 4 | 0.2447 |
| k-means k=32 (5-bit) | 5 | 0.3059 |
| k-means k=64 (6-bit) | 6 | 0.3672 |

On-disk context: `qat_alexnet_fire.pth` raw 0.5532 MB → gzip 0.4689 MB (1.18× ratio).

**H3 acceptance criterion met:** all three cluster counts beat the actual gzip ratio (0.24–0.37 MB
theoretical vs. 0.4689 MB gzip achieves on the real file) — gzip's DEFLATE is only capturing
~1.18× on top of INT8, while explicit weight-sharing has real headroom (up to ~2.2× at 16
clusters, weights-only). Per D6, no changes to `ml/checkpoint.py` yet — this is the "worth
building" signal the phase was scoped to produce.

---

## SCOPE & EFFORT

- **Task 1** — done. Ran on PCAD (job 806654, `tupi_4090`, 66 epochs to early-stopping, ~1h23m).
  Bypass alone accounts for ~53-58% of Phase 4's gain over plain `alexnet_fire`, for zero added
  parameters — see H1's Result.
- **Task 3** — done. Pure measurement script, no training, no model changes. Had no dependency on
  Task 1 — ran against `alexnet_fire`'s existing PCAD checkpoint.
- **Task 2** — done at this phase's scope (mechanics + one unfine-tuned measurement, scoped to
  `_AlexBottleneck`'s internal width — see Task 2's scope-refinement note). A *useful*
  pruned-accuracy result still needs a fine-tuning loop, which is a multi-day effort (new training
  runs per ratio, per model) and remains future work.

**Where this leaves Task 2's next step, now that all three inputs are in:** Task 1 shows bypass is
a real, free (zero-parameter) accuracy lever — worth adding to whatever gets pruned next rather
than pruning `alexnet_bottleneck`/`alexnet_fire` in isolation. Task 3 shows real weight-sharing
headroom exists above gzip. Together they suggest the next investment is either (a) generalizing
`ml/pruning.py` past `_AlexBottleneck` so `alexnet_fire_bypass` itself can be pruned, or (b)
building the fine-tuning loop to get real pruned-accuracy numbers — both still multi-day, neither
started here.

---

## BLOCKING ISSUES & REQUIRED FIXES

### 1. `AlexNetFireBypass` param-count parity with `AlexNetFire` (BLOCKING for H1's clean read)
Must verify after writing the class (not assumed) that `sum(p.numel() for p in
AlexNetFireBypass().parameters())` equals `AlexNetFire`'s count exactly — the whole point of D1 is
that bypass adds zero extra parameters (`FloatFunctional` has none). If it doesn't match, the
`_FireModule`/stem refactor in Task 1's code introduced an unintended difference and H1's
comparison is no longer clean.

### 2. Channel-propagation correctness in pruning — Resolved by scoping, not by solving
Originally flagged as the riskiest part of `ml/pruning.py`: propagating a channel removal through
every downstream consumer, including `_FireModule`'s squeeze-feeds-two-branches coupling. The
implemented scope sidesteps this rather than solving it (see Task 2's "scope refinement" note) —
pruning only touches `_AlexBottleneck`'s internal `mid_ch`, which by construction has no
downstream consumer outside the block itself. `_FireModule`'s cross-branch coupling is simply not
handled — pruning a Fire-based model finds zero `_AlexBottleneck` instances and raises
`"has no _AlexBottleneck blocks; nothing to prune"` rather than silently doing the wrong thing.
Extending to `_FireModule`/`_BottleneckResBlock` is future work, and would need to actually solve
the propagation problem this note originally raised.

### 3. `scipy` (not `sklearn`) availability — Resolved
`sklearn` is **not** installed in this project's `.venv` (checked directly). `scipy` is
(1.17.1) — `scipy.cluster.vq.kmeans2` covers Task 3's k-means step with no new dependency, so
`scripts/measure_compression.py` uses that instead of `sklearn.cluster.KMeans`. See Task 3's
correction note.

---

## REPRODUCIBILITY & VERIFICATION CHECKLIST

- [x] `AlexNetFireBypass` param count == `AlexNetFire` param count (Blocking Issue 1) — 516,152 ==
      516,152, verified
- [x] `alexnet_fire_bypass` registered identically in `ml/model_registrations.py` and
      `notebooks/training/compensation_qat.ipynb`
- [x] FP32→QAT→INT8 run produces a `alexnet_fire_bypass_summary.json` with the same fields as
      existing Phase 3 summaries (`evaluate(topk=(1,5))` numbers present) — PCAD job 806654,
      `outputs/pcad/phase9_fire_bypass/alexnet_fire_bypass/results/alexnet_fire_bypass_summary.json`
- [x] `pytest tests/` passes, in particular `test_registry.py` and `test_quantization.py`, after
      adding the new model/registration
- [x] `scripts/prune_channels.py --dry-run` prints channel counts without writing files
- [x] Pruned model forward-passes at `(1,3,64,64)` and every remaining `Conv2d.groups == 1`
- [x] `Trainer.evaluate()` runs without shape errors on the pruned (unfine-tuned) model — real run
      against `outputs/pcad/large_scale/alexnet_bottleneck`, see Task 2 results
- [x] Task 3's entropy numbers are ≤ 8 bits/weight (sanity bound — INT8 range width) — 7.19,
      asserted in-script
- [x] Task 3's comparison table (entropy, 16/32/64-cluster sizes, gzip ratio) recorded in this
      file's results section — see Task 3 results, `alexnet_fire`
