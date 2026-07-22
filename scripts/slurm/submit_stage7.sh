#!/bin/bash
# Stage 7: Full 30-epoch FP32 sweep on all 3 backbones at 512px
# Submit after smoke run (Stage 6) passes go/no-go gate

echo "=== STAGE 7: Full Detection Sweep ==="
echo "Submitting 3-backbone 30-epoch runs..."

sbatch scripts/slurm/det_seg.sbatch detection fp32 alexnet_bottleneck phase7_detection
echo "✓ alexnet_bottleneck submitted"

sbatch scripts/slurm/det_seg.sbatch detection fp32 alexnet_fire phase7_detection
echo "✓ alexnet_fire submitted"

sbatch scripts/slurm/det_seg.sbatch detection fp32 alexnet_tv phase7_detection
echo "✓ alexnet_tv submitted"

echo ""
echo "Stage 7 sweep in progress. Results in: runs/phase7/"
echo "Monitor with: squeue -u rsdsouza"
