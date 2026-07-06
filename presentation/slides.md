---
marp: true
theme: default
paginate: true
size: 16:9
---

<!-- Title Slide -->
# Restrição do Tamanho de Kernel em CNNs e Sua Eficiência

Rafael Silva de Souza

---

## Experimentos

- **Dataset:** Tiny ImageNet-200 (64×64 RGB, 200 classes)  
- **Pipeline:** Treinamento FP32 → ajuste fino QAT → inferência INT8  
- **Escopo:** 4 fases, mais de 25 variantes de arquitetura

---

## Motivação: Por Que o Tamanho do Kernel Importa

**O problema:** A convolução acelerada por Winograd atinge alta eficiência para kernels pequenos (2×2, 3×3), mas escala mal para filtros grandes.

**Pergunta:** É possível trocar tamanho de kernel por eficiência em aceleradores Winograd sem sacrificar a acurácia? Quão robusta é essa troca sob quantização?

---

## Design da Pesquisa

| Fase | Foco | Modelos-Chave |
|-------|-------|-----------|
| **1** | Baselines pré-treinados | MobileNetV2, ResNet18, VGGStyle, AlexNet |
| **2** | Restrição de kernel | AlexNet 3×3, 2×2, SmallKernel (otimizado) |
| **3** | Mecanismos de compensação | Bottleneck, Fire, Residual, DepthwiseSep |
| **4** | Arquiteturas híbridas finais | Fire-Residual, Bottleneck-Fire (FP32 + compressão extrema) |

---

## A Restrição de Kernel

![width:700px](../presentation/figures/kernel_restriction_cost.png)

---

## Equivalentes em INT8

![width:700px](../presentation/figures/kernel_restriction_cost_int8.png)

*ResNet18 omitido — não existe resultado INT8/QAT para esse baseline pré-treinado.*

---

## Dois Caminhos de Recuperação

|  | **SmallKernel** | **Residual** |
|---|---|---|
| Mecanismo | Canais mais estreitos (64→128→256→256→256) + cabeça GAP no lugar das FC 4096+4096 + camadas 3×3 para compensar o campo receptivo | 5 blocos residuais: duas conv 3×3+BN+ReLU somadas ao atalho via `FloatFunctional`; canais e cabeça FC iguais ao `AlexNet3x3` |
| Parâmetros | 1,6M | 57,8M |
| Tamanho | **18 MB** | 694 MB |
| FP32 → INT8 | 45,8% → 35,9% | **48,0% → 47,3%** |

**Leve, porém frágil na quantização** — vs. — **Pesado, porém o mais preciso e estável**

---

## Descoberta 2: Mecanismos de Compensação Recuperam Acurácia

![width:800px](../presentation/figures/best_models_fp32_vs_int8.png)

---

## O Vencedor Híbrido: Arquitetura Fire-Residual

![width:750px](../presentation/figures/master_pareto_frontier_all_models.png)

---

## O Vencedor Híbrido (continuação)

**Melhor modelo compacto (<10 MB):**  
- **alexnet_final_fire_residual:** 49,8% FP32 → 49,2% INT8 com **8,09 MB**  
- Iguala a acurácia do MobileNetV2 (57,9%) com **~3,5× menos tamanho**  
- Robusto à quantização (queda de –0,6pp)

**Por que essa arquitetura vence:**
- Módulo Fire: reaproveitamento de features eficiente em parâmetros
- Conexões residuais: fluxo de gradiente + estabilidade na quantização
- Kernels pequenos: compatíveis com aceleradores Winograd

---

## Compressão Extrema: Levando os Limites Adiante

![width:750px](../presentation/figures/extreme_compression_methods.png)

---

## Compressão Extrema (continuação): Trade-offs de Design

**Quão pequeno podemos ir?** A quantização extrema viabiliza modelos ultracompactos:
- **QAT Ternário:** 35–44% de acurácia @ 0,1–0,5 MB  
- **QAT Int4:** 41–45% de acurácia @ 1–2 MB  
- **Precisão mista:** ponto ideal em 5–8 MB

**Decisão de design:** modelos abaixo de 1 MB existem, mas a acurácia cai bruscamente abaixo de 30%. O ponto de inflexão: **5–8 MB é a faixa prática** para >40% de acurácia no Tiny ImageNet-200.

---

## Conclusões

**Seleção de modelo por caso de uso:**
- **Produção (>45% de acurácia):** arquiteturas Fire, Bottleneck ou Fire-Residual  
- **Edge (<10MB):** híbrido Fire-Residual (49,2% INT8)  
- **Ultracompacto (<1MB):** quantização Ternária/Int4 (30–40% de acurácia)

**Descoberta-chave:** CNNs compatíveis com Winograd entregam **4,3× melhor acurácia por MB** do que designs convencionais, sem sacrificar a robustez à quantização.

---

## Referências e Trabalhos Relacionados

- **Convolução Rápida de Winograd:** Lavin & Gray (2016), *"Fast Algorithms for Convolutional Neural Networks"*
- **Treinamento com Quantização (QAT):** Jacob et al. (2018), framework de QAT do PyTorch
- **CNNs Eficientes:** Sandler et al. (2018, MobileNetV2), He et al. (2016, ResNet), Iandola et al. (2016, SqueezeNet)
- **Inferência INT8:** backend fbgemm da Intel; documentação de quantização do PyTorch
- **Dataset:** desafio Tiny ImageNet-200 (classificação 64×64)

---

## Obrigado
