# Plano — Relatório de acurácia sob quantização para o acelerador Winograd F(4,3)

> **Estudo satélite. Não é uma fase do dnn_study.**
>
> Pedido do Thomas (via Mateus) para o projeto de acelerador FPGA dele, que é um repo separado:
> `~/Documents/Winograd-FPGA` (`git@github.com:thschmitz/Winograd-FPGA.git`).
> O pedido original está versionado lá em `docs/mensagem_colega_acuracia.md` (branch `winograd_f43`).
>
> **Todos os caminhos de arquivo neste documento são relativos a `~/Documents/Winograd-FPGA`,
> não ao dnn_study.** Cópia deste plano também salva lá em `docs/plano_relatorio_acuracia.md`.
>
> Escrito em 2026-08-06, antes da implementação. Branch alvo: `acuracia_quantizacao`
> (a partir de `winograd_f43`).

## Por que isso está no dnn_study

O código todo vive no repo do FPGA — este documento está aqui só como registro, porque o
dnn_study é o projeto de pesquisa em Winograd que motivou o pedido. Duas conexões reais:

- **Evidência INT8.** As Fases 1–9 já mediram a queda FP32→INT8 por família de arquitetura em CNN.
  Isso é argumento pronto para o estudo do acelerador, que até hoje só operou em INT16/bit-exato.
- **Código morto a limpar aqui.** `ml/quantization_advanced.py:279 winograd_f23_conv` não contém
  transform nenhum (o próprio comentário em `:300-304` admite), e `:319 compute_winograd_quant_error`
  aplica ops idênticas aos mesmos tensores nos dois ramos (`:367-369` vs `:372`) — reporta
  `ratio = 1.0` por construção. `scripts/winograd_quant_error.py:105` está quebrado desde `94ca2ae`
  (desempacota 2 de uma tupla de 4 vinda de `ml/data.py:12`). Proposta de remoção separada,
  não executada.

## Objetivo

Achar o **menor U e menor V com `U + 2V ≤ 27`** e acurácia aceitável, pra destravar o packing de
2 multiplicações por DSP48E2. Hoje U=16, V=16 → `16 + 2×16 = 48`. Tentativa anterior com U=15,
V=14 → 43, ainda estoura.

---

## 1. Ponto de partida: o que já existe no repo do FPGA

### Branch

A linha ativa é `winograd_f43`: 2 cores × 24 N_OF × 36 = **1728 DSP (100%), roteando limpo a
313 MHz, 13/13 layers PASS bit-exato** (`docs/relatorios/estado_atual_consolidado.md` §2.2–2.3).
`master` ainda é o F(2,3) antigo (`wino_input_transform.v` com tile 4×4, saída 2×2) — não é a
referência para este trabalho.

### Metade da fundação numérica já está pronta

`scripts/validate_transforms_f43.py` já tem, validado:

| O que | Linha |
|---|---|
| `BT_F43`, `G_F43`, `AT_F43` — matrizes exatas | 19, 29, 39 |
| `direct_conv_4x4` — ground truth | 47 |
| `f43_u_int` — transform de entrada inteiro | 71 |
| `f43_vpre_int(g, vpre_bits)` — **V já parametrizado em largura** | 77 |
| `f43_winograd_hw(d, g, vpre_bits)` — caminho HW bit-accurate | 83 |
| `cosine` | 93 |

Commit `b25db96`: *"F(4,3) fundação numérica: golden validado (U=15b, cosine 0.99993)"*.

**Não é preciso escrever simulador nem gerador de matriz.** O trabalho é levantar o que existe de
um tile isolado para a rede inteira.

### O gap

O que existe hoje roda em **tiles 6×6 aleatórios**, mede **cosine de um tile**, e varre **só V**.
Falta:

- ativações **reais** da VGG16 em vez de tile aleatório
- **acumulação sobre N_IC** — onde vive a pergunta 2 (sensibilidade a N_IC) e o guard band do packing
- varrer **U** também, não só V
- acurácia **top-1 fim-a-fim**, não cosine de tile
- truncamento intermediário nos 2 estágios de cada transform (pergunta 3)
- F(2,3)/F(4,3)/F(6,3) lado a lado no mesmo benchmark (pergunta 5)

---

## 2. O argumento central

`.claude/agent-memory/performance-modeler/f43_equations.md` registra:
*"DSP packing INT8 inviável no F43 (43>27 bits)"* + *"NÃO re-derive"*.

Esse veredito partiu de larguras que **preservam bit-exatidão** (U=15, V=14 — o mesmo U=15b do
golden). A pergunta do pedido é outra: qual o mínimo com **perda de acurácia aceitável** — regime
que ninguém mediu. Todo o estreitamento até hoje foi sob restrição "bit-exato" (`9fafe59`
*"Estreita U de 26→20 bits: bit-exato"*, `34211f2` *"De-reset: bit-exato"*).

**Relaxar de "bit-exato" para "acurácia aceitável" é a contribuição do relatório** — e é o único
caminho que pode levar `U+2V` de 43 para ≤27.

Se a resposta medida for "nem relaxando dá", isso também é entregável: fecha a questão com número
em vez de estimativa, e vira parágrafo de artigo.

---

## 3. Orçamento de LUT é parte do resultado

`docs/relatorios/estado_atual_consolidado.md` §2.2: o zero-drain custou **+14,5 pontos de LUT →
congestão → Fmax de 313 para 265 MHz**. Neste projeto, "LUT extra vira perda de Fmax" é falha
**demonstrada**, não hipótese. (O LUT=109,8% de `synth_actual_2026-05-15.md` é do F(6,3)
congelado, não do F(4,3) — o F(4,3) fecha limpo.)

Consequência: qualquer esquema de pack/unpack precisa de **orçamento de LUT explícito**, senão
troca DSP por Fmax e sai no zero. Toda recomendação do relatório vem com custo estimado em LUT.

---

## 4. Pré-requisito: calibração de escala

`vgg16/export_vgg16_all_convs.py:468-469` tem `--act-scale` e `--wgt-scale` com default **1.0**.
Peso de VGG16 vive em |w| < 0,5, então `round(w × 1.0)` zera quase todos.

Isso não quebra o fluxo atual — o testbench compara RTL contra o modelo Python usando *os mesmos*
valores quantizados, então casa em PASS de qualquer jeito. Mas significa que **acurácia nunca foi
medida neste acelerador**, e `vgg16/resultados.txt` é só ciclos/throughput.

Escala por percentil, por camada, INT8 e INT16 — antes de qualquer número de acurácia.

---

## 5. Arquivos (todos em `~/Documents/Winograd-FPGA`)

| Arquivo | O que faz |
|---|---|
| `scripts/dump_vgg16_tensors.py` | **novo, usa TF uma vez.** Roda VGG16 Keras, salva ativação de entrada + pesos das 13 convs como `.npy`. Depois disso TF sai do caminho. |
| `scripts/wino_bitwidth_study.py` | **novo, numpy puro.** Importa o kernel de `validate_transforms_f43.py`, generaliza pra camada inteira (acumulação N_IC), parametriza U/V/acc/truncamentos, varre o grid, escreve CSV. |
| `docs/relatorio_acuracia_bitwidth.md` | **novo.** O relatório final. |

TF precisa de venv dentro do repo do FPGA. Não está instalado nesta máquina (nem no env global,
nem no `.venv` do dnn_study), e `scripts/layer_configs.py:13-33` aponta pra `/home/thomas/UFRGS/...`.

---

## 6. Passos

1. venv com TF; `dump_vgg16_tensors.py` roda uma vez.
2. Calibrar `act_scale`/`wgt_scale` por camada (percentil), INT8 e INT16.
3. Generalizar `f43_winograd_hw` pra camada completa com acumulação sobre N_IC, mantendo
   compatibilidade bit-exata com o kernel atual (é o teste de ancoragem).
4. Grid `u_bits × v_bits × chunk_K` × 13 camadas. O transform de entrada é calculado **uma vez**
   por camada em precisão exata; cada `u_bits` do grid é só um shift em cima — o grid inteiro sai
   de uma transformada. Simular em `float64`: exato para inteiro até 2^53, e o sinal mais largo é
   o acumulador de 40b, então dá semântica inteira bit-exata usando numpy normal.
5. Erro por camada vs conv direta FP32: médio abs, máx abs, SNR, cosine (mantém a métrica atual).
6. **Pergunta 2** — correlacionar SNR com N_IC nas 13 camadas. A hipótese tem base: erro
   independente de média zero cresce ~√N, sinal ~N, então SNR melhora ~√N_IC. Medir e ajustar.
   *`block1_conv1` (N_IC=3) já é caso patológico conhecido por outro motivo
   (`research_first_layer_problem.md`) e custa 186 ciclos de ~13k totais — "roda essa camada mais
   larga" é provavelmente grátis.*
7. **Pergunta 3** — varrer truncamento entre os 2 estágios de `wino_input_transform` e de
   `wino_output_transform`.
8. **Pergunta 4** — quantificar round-to-nearest vs truncamento (o `>>>` do output transform é
   piso, com viés DC que acumula sobre N_IC; corrigir custa 1 somador); saturação vs wrap; pontos
   de interpolação alternativos (±1/2 é shift-friendly e melhor condicionado que ±2 — Barabasz &
   Gregg, ACM TOMS 2020). Cada um com custo estimado em LUT (ver §3).
9. **Pergunta 5** — F(2,3)/F(4,3)/F(6,3) no mesmo benchmark, reusando `validate_transforms.py` e
   `validate_transforms_f63.py`.
10. **Acurácia** top-1/top-5 dos 3–5 configs finalistas: VGG16 completa, ~5k imgs do val do
    ImageNet-1k. Reportar também **concordância top-1 vs VGG16 FP32** (isola o erro do acelerador
    do erro do modelo).

---

## 7. Packing: a restrição é 3D, não 2D

`U + 2V ≤ 27` vale se cada produto for desempacotado a cada ciclo. Acumulando empacotado (que é o
que `wino_pe.v` faz), o guard band absorve também o crescimento da acumulação:

```
P·U = (V1·U)·2^s + V2·U
com K acumulações, a parte baixa cresce até  v + u + ceil(log2 K)
→ porta A = 2·v_bits + u_bits + ceil(log2 K) + 1 ≤ 27
```

| K (acumulações antes de drenar) | restrição efetiva | exemplo viável |
|---|---|---|
| 1 (desempacota todo ciclo) | `2V + U ≤ 26` | V=8, U=10 |
| 16 | `2V + U ≤ 22` | V=7, U=8 |
| 256 | `2V + U ≤ 18` | V=5, U=8 |

Reportar a fronteira por K — K menor custa mais drenagem em LUT, que é o trade a decidir com o
histórico de Fmax em mãos (§3). Packing signed×signed pede termo de correção (Xilinx WP486;
já existe `paper_dlpack_2024.md` como referência de packing INT8 em UltraScale+).

---

## 8. Verificação

- **Âncora:** rodar o kernel novo com as larguras de hoje e exigir saída **bit-idêntica** a
  `f43_winograd_hw` de `validate_transforms_f43.py`. Se não bater, o modelo está errado e nenhuma
  recomendação vale.
- `python3 scripts/validate_transforms_f43.py` continua passando (não é modificado, só importado).
- Cross-check analítico: crescimento de U medido bate com o bound L1 de `B^T` do F(4,3)
  (norma-linha 10 → ~6,6 bits sobre INT8 → 15b, que é o U=15b do golden `b25db96`).
- Acurácia FP32 da VGG16 no subset bate com o publicado do Keras dentro do ruído amostral.

---

## 9. Fora de escopo

- Não editar `.v` nenhum. O estudo recomenda; a mudança de RTL é do Thomas.
- Não rodar xsim (`scripts/layer_configs.py:13-33` aponta pra `/home/thomas/UFRGS/...`).
- Commits vão na branch `acuracia_quantizacao`, nunca direto em `winograd_f43`.
- Nada disso toca o pipeline do dnn_study (Fases 1–9).

---

## 10. Entregável

`docs/relatorio_acuracia_bitwidth.md` no repo do FPGA, respondendo as 5 perguntas de
`docs/mensagem_colega_acuracia.md`:

1. menor U e V recomendados, com erro médio e máximo medidos, na fronteira `U+2V ≤ 27` por K
2. regra de largura por camada (critério em função de N_IC), se a sensibilidade se confirmar
3. pontos do pipeline de transform redundantes em precisão
4. técnicas de compensação sem custo de DSP, ordenadas por ganho/custo em LUT
5. comparação F(2,3)/F(4,3)/F(6,3) de erro no mesmo benchmark
