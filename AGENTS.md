# CNN Kernel Research — Agent Guidelines

This workspace investigates how convolutional kernel size choices affect model accuracy, efficiency, and deployability for image classification on Tiny ImageNet-200.

## Quick Start

**Activate environment:**
```bash
source .venv/bin/activate
```

**Key notebooks (in order of complexity):**
- [3x3_models.ipynb](3x3_models.ipynb) — Small-kernel CNNs: 3×3, 2×2, mixed (baseline comparisons)
- [imagenet_models.ipynb](imagenet_models.ipynb) — AlexNet baseline training pipeline
- [imagenet_models2.ipynb](imagenet_models2.ipynb) — Improved models: ImprovedTinyCNN, StrongCNN, AlexNet3x3
- [TinyHybridNet_QAT.ipynb](TinyHybridNet_QAT.ipynb) — Hybrid efficient architecture + Quantization-Aware Training (QAT)

## Project Structure

### Experiment Organization
- **saved_models/** — Standard CNN checkpoints (AlexNet, ImprovedTinyCNN, StrongCNN)
- **saved_models_2/** — Augmentation variants (v2 experiments with better preprocessing)
- **saved_models_small_kernels/** — Kernel ablation: SmallKernelCNN, SmallKernelResNet, Kernel2x2CNN, MixedSmallKernelNet
- **saved_models_hybrid_qat/** — TinyHybridNet (FP32) + QAT INT8 models + experiment summaries

### Shared Code
[ml_utils.py](ml_utils.py) contains:
- `train_epoch()`, `evaluate()` — standard train/val loops
- `ExperimentConfig` — configuration management with JSON serialization
- `model_summary()`, `count_parameters()` — model analysis
- `set_seed()`, `get_system_info()` — reproducibility & diagnostics

## Research Focus

### Kernel Size Trade-offs
- **Small kernels (2×2, 3×3)**: Reduced receptive field & parameter count, faster inference, possible accuracy loss
- **Residual blocks**: Mitigate gradient issues in deep small-kernel nets
- **Hybrid strategies**: Mix 3×3 (feature extraction) + 2×2 (downsampling) or depthwise + bottleneck designs

### Efficiency & Deployment
- Parameter count & FLOPs directly measured via `count_parameters()` and layer introspection
- INT8 quantization via PyTorch QAT for edge deployment
- Model size compression & accuracy drop tracked side-by-side

### Dataset
- **Tiny ImageNet-200**: 64×64 RGB, 200 classes, ~90K train / ~10K val split
- Download via kagglehub (automatic in notebooks)
- Standard ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Common Tasks

### Run an existing experiment
1. Open notebook (e.g., [3x3_models.ipynb](3x3_models.ipynb))
2. Execute cells top-to-bottom; models save to `saved_models_small_kernels/`
3. Check `ExperimentConfig` in cell setup for batch size, epochs, learning rate

### Compare kernel sizes
- Review model definitions (e.g., SmallKernelCNN vs. Kernel2x2CNN)
- Measure parameter count: `count_parameters(model)`
- Compare final validation accuracy across models
- Check computational efficiency trade-offs (receptive field, FLOPs by layer)

### Add a new architecture
1. Define model class in notebook (inherit from `nn.Module`)
2. Use `train_model()` from ml_utils with your model
3. Save to an appropriate `saved_models_*` directory
4. Log results in experiment summary

### Quantize a model
- See [TinyHybridNet_QAT.ipynb](TinyHybridNet_QAT.ipynb) for full workflow
- Steps: (1) Train FP32 baseline, (2) Insert QAT stubs, (3) Calibrate & fine-tune, (4) Convert to INT8
- Measure accuracy drop and model size compression

## Key Conventions

| Item | Convention |
|------|-----------|
| **Image size** | 64×64 (Tiny ImageNet native) |
| **Batch size** | 64 (typical; adjust for GPU memory) |
| **Epochs** | FP32: 30, QAT: 5 (see config) |
| **Device** | CUDA (RTX 4060 Laptop GPU); falls back to CPU if unavailable |
| **Seed** | 42 (for reproducibility) |
| **Model save path** | `./saved_models_<variant>/<model_name>.pth` |
| **Config save** | `./saved_models_hybrid_qat/config.json` for QAT experiments |

## Troubleshooting

**Out of memory (OOM)?**
- Reduce `batch_size` in `ExperimentConfig` or cell code
- Use `torch.cuda.empty_cache()` between runs

**Kernel not running?**
- Activate venv: `source .venv/bin/activate`
- Restart kernel: VS Code → Command Palette → "Jupyter: Restart Kernel"

**Different results across runs?**
- Ensure `set_seed(42)` is called at notebook start
- Check `torch.backends.cudnn.deterministic` and `benchmark` settings

## Next Steps & Recommendations

1. **Analyze kernel receptive field**: Measure effective receptive field vs. parameter count for 2×2 vs. 3×3
2. **Depth ablation**: Test deep 2×2 models with residuals to isolate kernel vs. depth effects
3. **Deployment simulation**: Benchmark INT8 latency on target hardware (e.g., quantized TinyHybridNet)
4. **Cross-dataset validation**: Validate findings on CIFAR-10 or ImageNet-1K scale images
5. **Depthwise vs. Standard**: Compare depthwise separable convolutions in small-kernel setting

---

**Agent Mode**: Use `cnn-kernel-research` mode for specialized CNN kernel analysis, or default mode for general coding tasks.  
**Questions?** Check docstrings in [ml_utils.py](ml_utils.py) or experiment summaries in [saved_models_hybrid_qat/](saved_models_hybrid_qat/).
