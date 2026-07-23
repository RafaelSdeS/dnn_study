# Phase 7 Multi-Node Job Submission

Automatically distribute Phase 7 detection training across multiple PCAD nodes (one backbone per node, parallel execution).

## Quick Start

### Option 1: Simple (recommended for most cases)
```bash
bash scripts/submit_phase7_simple.sh
```
Submits 3 jobs immediately, one per backbone, all in parallel.

### Option 2: Advanced (with QAT/INT8 chaining)
```bash
bash scripts/submit_phase7_multinode.sh          # FP32 only
bash scripts/submit_phase7_multinode.sh qat      # FP32 → QAT (chained)
bash scripts/submit_phase7_multinode.sh qat int8 # FP32 → QAT → INT8 (chained)
bash scripts/submit_phase7_multinode.sh --dry-run # Preview without submitting
```

## What Each Does

### Simple Script (`submit_phase7_simple.sh`)
- **Submits:** 3 parallel FP32 detection jobs (one per backbone)
- **Nodes:** Each job gets its own GPU node (no sharing)
- **Walltime:** 12 hours per job
- **Output:** Logs to `outputs/detection_segmentation/phase7/logs/p7_<model>_<jobid>.log`

```bash
# Example: submit only bottleneck and fire
bash scripts/submit_phase7_simple.sh alexnet_bottleneck alexnet_fire
```

### Advanced Script (`submit_phase7_multinode.sh`)
- **FP32:** 3 parallel jobs (like simple)
- **QAT:** Chains 3 QAT jobs (depends on FP32 completion)
- **INT8:** Optional chains 3 INT8 conversion jobs (depends on QAT completion)
- **Job dependencies:** Automatic sequencing via SLURM `--dependency=afterok`

```bash
# Submit with automatic FP32→QAT→INT8 chaining
bash scripts/submit_phase7_multinode.sh qat int8
```

## Monitoring

### Check job status
```bash
squeue -u $USER                    # All your jobs
squeue -u $USER -j 12345           # Specific job
```

### View logs
```bash
tail -f outputs/detection_segmentation/phase7/logs/p7_bottleneck_*.log   # Real-time log for bottleneck
cat outputs/detection_segmentation/phase7/logs/p7_*.log                   # All logs
```

### Cancel jobs
```bash
scancel 12345                      # Cancel single job
scancel -u $USER -n "p7_*"         # Cancel all Phase 7 jobs
```

## Configuration

Both scripts use:
- **Partition:** `gpu`
- **Memory:** 32G
- **GPUs:** 1 (per job)
- **Time:** 12 hours (FP32; can reduce to 2-3h if needed)

Modify in the script headers if needed:
```bash
PARTITION="gpu"
TIME="12:00:00"
MEM="32G"
GPUS=1
```

## Output Structure

Results appear in:
```
outputs/detection_segmentation/phase7/
├── ssd_alexnet_bottleneck_fp32/
│   ├── config.yaml
│   ├── git_hash.txt
│   ├── metrics.json
│   └── ssd_alexnet_bottleneck_fp32_best.pth
├── ssd_alexnet_fire_fp32/
│   └── ...
├── ssd_alexnet_tv_fp32/
│   └── ...
└── logs/
    ├── p7_bottleneck_12345.log
    ├── p7_fire_12346.log
    └── p7_tv_12347.log
```

## Examples

### Scenario 1: Test locally first, then submit to cluster
```bash
# Test with single job locally
python scripts/train_det_seg.py detection --model alexnet_bottleneck --dry-run

# Then submit all 3 to PCAD in parallel
bash scripts/submit_phase7_simple.sh
```

### Scenario 2: Only train bottleneck and fire (skip TV)
```bash
bash scripts/submit_phase7_simple.sh alexnet_bottleneck alexnet_fire
```

### Scenario 3: Run full pipeline (FP32 + QAT + INT8) with automatic chaining
```bash
bash scripts/submit_phase7_multinode.sh qat int8

# Dry-run first to see what would happen:
bash scripts/submit_phase7_multinode.sh qat int8 --dry-run
```

### Scenario 4: Submit with longer walltime (if needed)
Edit script, change `TIME="12:00:00"` to `TIME="24:00:00"`, then submit:
```bash
bash scripts/submit_phase7_multinode.sh
```

## Job Dependencies (Advanced)

The advanced script automatically chains jobs:
```
FP32-bottleneck (job 1001)
├─ QAT-bottleneck (job 1004, depends on 1001)
│  └─ INT8-bottleneck (job 1007, depends on 1004)
FP32-fire (job 1002)
├─ QAT-fire (job 1005, depends on 1002)
│  └─ INT8-fire (job 1008, depends on 1005)
FP32-tv (job 1003)
├─ QAT-tv (job 1006, depends on 1003)
│  └─ INT8-tv (job 1009, depends on 1006)
```

Each backbone's QAT waits for that backbone's FP32 to complete.
All FP32 jobs run in parallel.
All QAT jobs run in parallel (but after their respective FP32).

## Troubleshooting

**"Jobs not running / showing CA (job cancelled array)"**
→ Check partition availability: `sinfo -p gpu`

**"Out of memory"**
→ Increase `MEM="32G"` to `MEM="48G"` in script

**"Taking too long"**
→ Check walltime: `squeue -u $USER` (remaining time in RUN row)

**"Need to cancel one job but keep others"**
→ `scancel 12345` (single job ID, from `squeue` output)

## See Also

- `PHASE7_QUICKSTART.md` — Configuration & interpretation
- `ideas/PHASE7_PLAN.md` — Research design & hypotheses
- `scripts/train_det_seg.py` — CLI documentation (--help)
