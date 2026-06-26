---

# Research Roadmap

## Objective

Design an **AlexNet-inspired CNN** using **3×3 (or smaller) kernels** that provides the best trade-off between:

- Accuracy
- Model size
- Training speed
- Inference speed
- Quantization performance
- Winograd compatibility

---

## Architectures to Explore

### Baselines

- AlexNet
- AlexNet (3×3)
- StrongCNN
- TinyHybridNet

### Lightweight CNNs

- SqueezeNet
- MobileNetV2
- ShuffleNetV2
- GhostNet

---

## Architectural Ideas

### Kernel Designs

- 3×3 kernels
- 2×2 kernels
- Stacked 3×3 convolutions
- Mixed-kernel architectures

### Convolution Variants

- Standard convolutions
- Depthwise separable convolutions
- Group convolutions

### Network Blocks

- Residual connections
- Fire modules (SqueezeNet)
- Inverted Residual blocks (MobileNetV2)
- Squeeze-and-Excitation (SE)

---

## Training Techniques

- Better data augmentation
- Label smoothing
- Exponential Moving Average (EMA)
- Cosine learning rate scheduling
- Mixed Precision (AMP)

---

## Model Optimization

- Quantization-Aware Training (QAT)
- Post-Training Quantization (PTQ)
- Structured pruning
- Knowledge Distillation
- Winograd-compatible kernel design

---

## Evaluation Metrics

Evaluate every architecture using:

- Top-1 / Top-5 Accuracy
- Parameters
- FLOPs / MACs
- Model size
- FP32 vs INT8 accuracy
- Quantization accuracy drop
- Training time
- CPU latency
- INT8 inference latency

---

## Final Goal

Develop a lightweight **AlexNet-like architecture** composed primarily of **3×3 (or smaller) kernels**, inspired by ideas from **ResNet**, **SqueezeNet**, and **MobileNet**, that is:

- Accurate
- Lightweight
- Fast to train
- Fast to infer
- Quantization-friendly
- Compatible with Winograd-based accelerators