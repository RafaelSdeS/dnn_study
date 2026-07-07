---
marp: true
theme: gaia
paginate: true
size: 16:9
math: katex
style: |
  section { 
    font-size: 30px;
    padding: 60px 70px;
  }
  img[alt~="center"] {
    display: block;
    margin: 0 auto;
    max-width: 100%;
    height: auto;
  }
  table {
    margin: 0 auto;
    font-size: 22px;
  }
  .small { font-size: 24px; }
  .smaller { font-size: 20px; }

  section.title-slide {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    text-align: center;
    padding: 60px 80px;
  }
  .columns {
    display: flex;
    gap: 25px;
    align-items: center;
  }

  .columns > div {
    flex: 1;
  }

  img {
    display: block;
    margin: 0 auto 10px auto;
    max-width: 100%;
    height: auto;
    background: transparent;
  }

  table {
    width: 95%;
    border-collapse: collapse;
    margin: 20px auto !important;
  }

  table th {
    padding: 14px 12px;
    font-weight: 600;
  }

  table td {
    padding: 16px 12px;
    text-align: center;
  }

  table tbody tr {
    border-bottom: 1px solid #ddd;
  }

  .refs {
    display: flex;
    gap: 30px;
    align-items: flex-start;
  }
  .refs > div {
    flex: 1;
  }
  .refs p {
    font-size: 14px !important;
    line-height: 1.25 !important;
    margin: 6px 0 !important;
  }

  .mini-table table {
    width: 70% !important;
    font-size: 13px !important;
    margin: 8px auto !important;
  }
  .mini-table table th,
  .mini-table table td {
    padding: 4px 8px !important;
  }
---

<!-- _class: title-slide -->

# Restrição do Tamanho de Kernel em CNNs e Sua Eficiência

Rafael Silva de Souza

---

## Experimentos

- **Dataset:** Tiny ImageNet-200 (64×64 RGB, 200 classes) [1]
- **Pipeline:** Treinamento FP32 → ajuste fino QAT [2] → inferência INT8 via fbgemm [3], em PyTorch [4]
- **Escopo:** 4 fases, mais de 25 variantes de arquitetura

---

## Motivação: Por Que o Tamanho do Kernel Importa

**O problema:** A convolução acelerada por Winograd [5] atinge alta eficiência para kernels pequenos (2×2, 3×3), mas escala mal para filtros grandes.

**Pergunta:** É possível trocar tamanho de kernel por eficiência em aceleradores Winograd sem sacrificar a acurácia? Quão robusta é essa troca sob quantização?

---

## Design da Pesquisa

| Fase | Foco | Modelos-Chave |
|-------|-------|-----------|
| **1** | Baselines pré-treinados | MobileNetV2 [6], ResNet18 [7], VGGStyle [8], AlexNet [9] |
| **2** | Restrição de kernel | AlexNet 3×3, 2×2, SmallKernel (otimizado) |
| **3** | Mecanismos de compensação | Bottleneck [7], Fire [10], Residual [7], DepthwiseSep [11] |
| **4** | Arquiteturas híbridas finais | Fire-Residual, Bottleneck-Fire (FP32 + compressão extrema) |

---

## Restrição de Kernel

![width:900px](../presentation/figures/kernel_restriction_cost.png)

<div class="mini-table">

| Modelo | Head | Camadas FC |
|--------|------|-----------|
| MobileNetV2, VGG-Style, ResNet18, 2×2, Mixed, SmallKernel, Fire-Residual | GAP (1×1) | 1 |
| AlexNet 3×3, Residual | AvgPool (6×6) | 3 |

</div>

---

## Equivalentes em INT8

![width:900px](../presentation/figures/kernel_restriction_cost_int8.png)

*ResNet18 e MobileNetV2 omitidos — sem resultado INT8/QAT válido para esses baselines pré-treinados (somas residuais nativas, incompatíveis com conversão INT8).*

---

## Comparação

<div class="smaller">

|  | **SmallKernel** | **Residual** | **Fire-Residual** |
|---|---|---|---|
| Mecanismo | Canais estreitos + GAP + 3×3 | Blocos residuais [7] (`FloatFunctional`) | Fire [10] + atalho residual |
| Parâmetros | 1,6M | 57,8M | 0,7M |
| Tamanho | 18 MB | 694 MB | 8 MB |
| FP32 → INT8 | 45,8% → 35,9% | 48,0% → 47,3% | 49,8% → 49,2% |

</div>

---

## Compração FP32 vs. INT8

![width:950px](../presentation/figures/best_models_fp32_vs_int8.png)

---

## Melhores Modelos (até o momento)

![width:920px](../presentation/figures/master_pareto_frontier_all_models.png)

---

## Compressão Extrema

![width:920px](../presentation/figures/extreme_compression_methods.png)

---

## Referências

<div class="refs">

<div>

**[1]** LE, Y.; YANG, X. *Tiny ImageNet visual recognition challenge*. Stanford: Stanford University, 2015. (CS231N Course Technical Report).

**[2]** JACOB, B. *et al.* Quantization and training of neural networks for efficient integer-arithmetic-only inference. In: IEEE CVPR, 2018, Salt Lake City. *Anais [...]*. Piscataway: IEEE, 2018. p. 2704-2713.

**[3]** KHUDIA, D. *et al. FBGEMM: enabling high-performance low-precision deep learning inference*. [S. l.]: arXiv, 2021. arXiv:2101.05615.

**[4]** PASZKE, A. *et al.* PyTorch: an imperative style, high-performance deep learning library. In: NeurIPS, 32., 2019, Vancouver. *Anais [...]*. [S. l.]: Curran Associates, 2019. p. 8024-8035.

</div>

<div>

**[5]** LAVIN, A.; GRAY, S. Fast algorithms for convolutional neural networks. In: IEEE CVPR, 2016, Las Vegas. *Anais [...]*. Piscataway: IEEE, 2016. p. 4013-4021.

**[6]** SANDLER, M. *et al.* MobileNetV2: inverted residuals and linear bottlenecks. In: IEEE CVPR, 2018, Salt Lake City. *Anais [...]*. Piscataway: IEEE, 2018. p. 4510-4520.

**[7]** HE, K. *et al.* Deep residual learning for image recognition. In: IEEE CVPR, 2016, Las Vegas. *Anais [...]*. Piscataway: IEEE, 2016. p. 770-778.

**[8]** SIMONYAN, K.; ZISSERMAN, A. Very deep convolutional networks for large-scale image recognition. In: ICLR, 3., 2015, San Diego. *Anais [...]*. [S. l.: s. n.], 2015.

</div>

<div>

**[9]** KRIZHEVSKY, A.; SUTSKEVER, I.; HINTON, G. E. ImageNet classification with deep convolutional neural networks. In: NeurIPS, 25., 2012, Lake Tahoe. *Anais [...]*. [S. l.]: Curran Associates, 2012. p. 1097-1105.

**[10]** IANDOLA, F. N. *et al. SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size*. [S. l.]: arXiv, 2016. arXiv:1602.07360.

**[11]** HOWARD, A. G. *et al. MobileNets: efficient convolutional neural networks for mobile vision applications*. [S. l.]: arXiv, 2017. arXiv:1704.04861.

</div>

</div>

---

## Obrigado
