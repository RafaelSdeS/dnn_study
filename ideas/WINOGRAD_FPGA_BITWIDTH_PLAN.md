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
> (a partir de `winograd_f43`; a branch já existe no remoto e contém este plano em
> `docs/plano_relatorio_acuracia.md` — commit `743a2cd`).
>
> **Revisado em 2026-08-06** após conferir o RTL e o estado da máquina. As larguras que
> constavam na v1 estavam erradas (vinham de defaults obsoletos, não do top module) — ver §0.

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
2 multiplicações por DSP48E2.

---

## 0. Baseline confirmado: U=16, V=20, ACC=44

Quatro fontes descreviam larguras diferentes. **Confirmado com o Thomas em 06/08: V=20 é o
correto**, batendo com o que está commitado (`winocnn_top_f43.v:63-67`, autoridade porque o top
module sobrescreve os parâmetros de todos os submódulos):

| Fonte | U | V | ACC | O que era |
|---|---|---|---|---|
| Mensagem original | 26 | 16 | 48 | design pré-`f2a52f0`, V armazenado pré-transformado |
| Correção por WhatsApp (06/08) | 16 | 16 | 40 | idem — ainda descrevendo o design antigo |
| **`winocnn_top_f43.v:63-67` — confirmado** | **16** | **20** | **44** | **atual, pós-`f2a52f0`** |
| v1 deste plano (errado) | 16 | 16 | 48 | — |

As duas primeiras linhas não eram pedido de largura — eram descrição do design **antes** de
`f2a52f0` (commit do próprio Thomas, `2026-08-05 22:34`, *"Volta filter transform... 9 ao invés
dos 36 transformados"*), quando V era um valor pré-transformado de 16 bits armazenado em memória.
Esse commit moveu a transformada pra dentro do chip; V virou `RAW_W(8) + POST_SHIFT(12) = 20`,
calculado on-chip. As duas descrições "V=16" batiam entre si porque as duas descreviam o design
antigo — não porque o antigo estivesse certo. `ACC=44` segue do mesmo commit; não foi reconfirmado
em separado, mas é o mesmo design de onde V=20 foi confirmado.

O `26` da mensagem original vem de defaults obsoletos em `wino_pe_f43.v:32-33`, comentados no
próprio arquivo como `// largura de U (exemplo)`. `wino_systolic_array_f43.v:27-29` repete esses
defaults e ainda traz `ACC_W=48`; `wino_output_transform_f43.v:17` tem default `ACC_W=40`. Nenhum
deles vale — o top passa os valores explicitamente.

**Distância real do packing: `16 + 2×20 = 56`**, não 48.

### O ponto que muda o pedido: V=20 não é precisão de peso

`winocnn_top_f43.v:23` já tem `RAW_W = 8` — **os pesos já são INT8**. A largura de V decompõe
exatamente em:

```
V_DATA_W = RAW_W + POST_SHIFT = 8 + 12 = 20
```

com `POST_SHIFT=12` vindo da transformada de filtro on-chip (`winocnn_top_f43.v:80-83`): G escalada
2^10, aplicada 2×, menos os `ROUND_BITS=8` descartados no meio.

Ou seja: **12 dos 20 bits de V são escala de ponto fixo da transformada G, não informação de peso.**

Consequências diretas:

- "Diminuir a precisão pra INT8" já foi feito. Não sobra nada pra ganhar do lado do peso.
- Encolher V = reduzir a escala da transformada G, que mapeia 1-pra-1 em
  `f43_vpre_int(g, vpre_bits)` — e `validate_transforms_f43.py` **já varre `vpre_bits ∈ {4..8}`**.
  Metade da resposta da pergunta 1 sai de uma rodada do script que já existe.
- Mas mesmo com `vpre_bits=4` → V=12 → `16 + 2×12 = 40 > 27`. Chegar a ≤27 com escala **uniforme**
  exigiria V≈7, menos bits que o próprio peso INT8. **Com escala uniforme, o veredito registrado em
  `f43_equations.md` provavelmente está certo.**

### A saída: escala por posição

Os coeficientes inteiros de G vão de 43 a 1024 e as normas-linha de `B^T` têm espalhamento
parecido. Boa parte de U e de V é gasta representando esse espalhamento, não informação. Escala
**por posição do tile** (uma escala pra cada uma das 36 posições, escolhida pra que `U·V` tenha
escala uniforme) recupera esses bits, custa shifts constantes em LUT e **zero DSP** — as estruturas
de multiplicação por constante já estão lá.

Essa é a contribuição que o relatório pode ter de mais forte, e não estava na v1 do plano.

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

### Vetores reais de VGG16 já estão commitados

`scripts/f43_sim_build_L2/` e `scripts/f43_sim_build_L3/` têm ativação + peso das **13 convs da
VGG16**, uma pasta por camada (`data_block1_conv1` … `data_block5_conv3`), com `case.txt` de
geometria. Inclui os dois extremos que a pergunta 2 precisa: `block1_conv1` com `N_IC 3` e
`block5_conv3` com `N_IC 512`.

Isso significa que **as perguntas 1, 2, 3 e 5 rodam hoje, só com numpy, sem TensorFlow.**

Duas ressalvas antes de confiar neles:

- São faixas de 34×16 (`IMG_W 34`, `IMG_H 16`), não camadas inteiras. Servem pra erro por camada,
  **não** pra top-1 fim-a-fim.
- `f43_sim_build_L2/data_block1_conv1/wgt.mem` tem 864 valores (= 24×36) na faixa −192…256, fora
  de INT8. Parece ser **V pré-transformado do fluxo antigo**, anterior ao commit `f2a52f0`
  (*"Volta filter transform para poder trazer menos dados da memoria — 9 ao invés dos 36
  transformados"*), que moveu a transformada pra dentro do chip. **Confirmar com o Thomas antes de
  usar** — se forem V e não peso cru, o pipeline de leitura muda.

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

**Relaxar de "bit-exato" para "acurácia aceitável" é a contribuição do relatório.**

Só que, pela conta de §0, relaxar sozinho **não basta**: mesmo com `vpre_bits=4` o total fica em 40.
São dois eixos, não um:

1. relaxar bit-exatidão → derruba `vpre_bits`, e com ele V;
2. **escala por posição** → tira dos dois operandos os bits gastos com o espalhamento dos
   coeficientes de G e `B^T`.

O relatório precisa medir os dois. Se com os dois ainda não fechar ≤27, isso também é entregável:
fecha a questão com número em vez de estimativa, e vira parágrafo de artigo.

*(O arquivo `f43_equations.md` está em `winograd_f43`/`acuracia_quantizacao`, não em `master`.)*

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

> **Correção da v1.** A v1 dizia que o default `1.0` zerava os pesos e que por isso a acurácia
> nunca tinha sido medida. A primeira metade está errada: o default é mesmo `1.0`
> (`vgg16/export_vgg16_all_convs.py:468-469`), mas os `.mem` commitados têm valores escalados
> de verdade (−192…256, nada de zeros), então os exports reais passaram uma escala. **A conclusão
> se mantém pelo motivo certo, não por esse.**

O motivo certo: o testbench compara RTL contra o modelo Python usando *os mesmos* valores
quantizados, então casa em PASS independente da escala estar boa ou ruim. **Nunca houve comparação
contra a saída FP32 da rede** — `vgg16/resultados.txt` é só ciclos/throughput. Acurácia de fato
nunca foi medida neste acelerador.

O export grava `act_scale`/`wgt_scale` no `meta_path` de cada caso (`:449-450`), mas esses arquivos
de metadados **não estão commitados** — só `case.txt`, `*.mem` e `ref_core*.txt`. Então a escala
usada nos vetores existentes é desconhecida e precisa ser re-derivada (ou perguntada).

Escala por percentil, por camada, INT8 e INT16 — antes de qualquer número de acurácia.

---

## 5. Arquivos (todos em `~/Documents/Winograd-FPGA`)

| Arquivo | O que faz |
|---|---|
| `scripts/dump_vgg16_tensors.py` | **novo, usa TF uma vez.** Roda VGG16 Keras, salva ativação de entrada + pesos das 13 convs como `.npy`. Depois disso TF sai do caminho. |
| `scripts/wino_bitwidth_study.py` | **novo, numpy puro.** Importa o kernel de `validate_transforms_f43.py`, generaliza pra camada inteira (acumulação N_IC), parametriza U/V/acc/truncamentos, varre o grid, escreve CSV. |
| `docs/relatorio_acuracia_bitwidth.md` | **novo.** O relatório final. |

### Estado dos pré-requisitos nesta máquina (conferido 2026-08-06)

| Item | Estado | Bloqueia |
|---|---|---|
| numpy / Python 3.12 | ✅ `.venv` do dnn_study | — |
| Vetores VGG16 por camada | ✅ commitados (`f43_sim_build_L2`/`_L3`) | — |
| `validate_transforms_f43.py` | ✅ na branch alvo | — |
| `validate_transforms.py` (F(2,3)) | ❌ só em `winograd_f63`/`master` | pergunta 5 |
| `validate_transforms_f63.py` | ❌ só em `winograd_f63` | pergunta 5 |
| Branch `acuracia_quantizacao` | ❌ remote-only, checkout local está em `master` | tudo |
| TensorFlow | ❌ não instalado (sem venv no repo FPGA, sem conda) | passo 1, top-1 |
| Pesos VGG16 Keras | ❌ `~/.keras/models` não existe (~528 MB) | top-1 |
| **Val set ImageNet-1k** | ❌ **ausente** (nada em disco, sem cache kagglehub) | **top-1** |
| Disco | ✅ 53 GB livres | — |

Os dois primeiros ❌ são um cherry-pick. O val set do ImageNet é ~6,7 GB de download e o único
item sem solução trivial.

> **Correção da v1:** o plano dizia que `scripts/layer_configs.py:13-33` apontava pra
> `/home/thomas/UFRGS/...`. Não aponta — esse arquivo externalizou os paths pra
> `scripts/project_paths.env` (local, não versionado), com `WINOCNN_ROOT` auto-detectado.
> Os `/home/thomas/UFRGS/...` que sobraram estão em artefatos gerados do Vivado
> (`scripts/f43_sim_build_L2/xsim.dir/*/xsim_script.tcl`), que este trabalho não usa.

---

## 6. Passos

Dividido em duas fases: a **A** roda hoje e responde 4 das 5 perguntas; a **B** só destrava
depois de instalar TF e baixar o ImageNet.

### Fase A — numpy puro, sem TF (perguntas 1, 2, 3, 5)

1. `git checkout acuracia_quantizacao`; cherry-pick de `validate_transforms.py` e
   `validate_transforms_f63.py` de `winograd_f63`.
2. Ler os vetores de `f43_sim_build_L2`/`_L3` (13 camadas). **Antes disso**, confirmar se
   `wgt.mem` é peso cru INT8 ou V pré-transformado (§1) e re-derivar a escala usada, já que os
   metadados não foram commitados (§4).
3. Generalizar `f43_winograd_hw` pra camada completa com acumulação sobre N_IC, mantendo
   compatibilidade bit-exata com o kernel atual (é o teste de ancoragem — §8).
4. Grid `u_bits × v_bits × chunk_K` × 13 camadas, **mais o eixo de escala por posição** (§0):
   uniforme vs. por-posição. O transform de entrada é calculado **uma vez** por camada em precisão
   exata; cada `u_bits` do grid é só um shift em cima — o grid inteiro sai de uma transformada.
   Simular em `float64`: exato para inteiro até 2^53, e o sinal mais largo é o acumulador de 44b,
   então dá semântica inteira bit-exata usando numpy normal.
5. Erro por camada vs conv direta FP32: médio abs, máx abs, SNR, cosine (mantém a métrica atual).
6. **Pergunta 2** — correlacionar SNR com N_IC nas 13 camadas. A hipótese tem base: erro
   independente de média zero cresce ~√N, sinal ~N, então SNR melhora ~√N_IC. Medir e ajustar.
   Os vetores commitados já cobrem os dois extremos (N_IC=3 e N_IC=512).
   *`block1_conv1` (N_IC=3) já é caso patológico conhecido por outro motivo
   (`research_first_layer_problem.md`) e custa 186 ciclos de ~13k totais — "roda essa camada mais
   larga" é provavelmente grátis.*
7. **Pergunta 3** — varrer truncamento entre os 2 estágios de `wino_input_transform` e de
   `wino_output_transform`. Alvos concretos já identificados: `ROUND_BITS=8` do filter transform
   (`wino_filter_transform_f43.v:66`) e `T_F43_W = DATA_W+5` do input transform.
8. **Pergunta 4** — ver §6.1, que mudou bastante.
9. **Pergunta 5** — F(2,3)/F(4,3)/F(6,3) no mesmo benchmark, reusando `validate_transforms.py` e
   `validate_transforms_f63.py`.

### Fase B — precisa de TF + ImageNet (acurácia fim-a-fim)

10. venv com TF; `dump_vgg16_tensors.py` roda uma vez (camadas inteiras, não faixas 34×16).
11. Calibrar `act_scale`/`wgt_scale` por camada (percentil), INT8 e INT16.
12. **Acurácia** top-1/top-5 dos 3–5 configs finalistas da Fase A: VGG16 completa, ~5k imgs do val
    do ImageNet-1k. Reportar também **concordância top-1 vs VGG16 FP32** (isola o erro do
    acelerador do erro do modelo).

### 6.1 Pergunta 4 — o que já está feito no RTL

> **Correção da v1.** O plano dizia que o `>>>` do output transform era piso, com viés DC, e que
> corrigir custaria 1 somador. **Já está corrigido.**
> `wino_output_transform_f43.v:67,73-74` faz `sum = y_comb + ROUND_BIAS; shifted = sum >>> POST_SHIFT`
> com `ROUND_BIAS = 1 <<< (POST_SHIFT-1)` — round-half-up. As linhas `:76-78` ainda **saturam** em
> INT8 em vez de dar wrap. O filter transform faz o mesmo (`:77-79`, round-half-up nos `ROUND_BITS`).

Então arredondamento e saturação — as duas compensações mais óbvias e baratas — **já foram gastas**.
Sobra medir:

- **escala por posição do tile** (§0) — o item de maior ganho esperado, custo em shifts constantes;
- pontos de interpolação alternativos (±1/2 é shift-friendly e melhor condicionado que ±2 —
  Barabasz & Gregg, ACM TOMS 2020);
- resíduo de round-half-up vs round-half-even (half-up ainda tem viés assimétrico no empate
  negativo; efeito pequeno, mas acumula sobre N_IC=512 — vale quantificar antes de descartar).

Cada um com custo estimado em LUT (ver §3).

---

## 7. Packing: a restrição é 3D, não 2D

Ponto de partida real: **U=16, V=20 → 56** (§0). O caminho até ≤27 passa por derrubar `POST_SHIFT`
*e* por escala por posição — nenhum dos dois sozinho chega lá.

`U + 2V ≤ 27` vale se cada produto for desempacotado a cada ciclo. Acumulando empacotado (que é o
que `wino_pe_f43.v` faz), o guard band absorve também o crescimento da acumulação:

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
- Não rodar xsim — os `xsim_script.tcl` gerados apontam pra `/home/thomas/UFRGS/...` e o Vivado
  não está nesta máquina. Os `.mem` são lidos direto, sem simulador.
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

As perguntas 1–3 e 5 saem da Fase A (roda hoje). A acurácia top-1 fim-a-fim que fecha a pergunta 1
é Fase B e depende do ImageNet. **Entregar a Fase A primeiro** — o argumento do ISCAS não precisa
esperar o download.

---

## 11. Perguntas em aberto pro Thomas

~~1. V é 16 ou 20?~~ **Resolvido 06/08: V=20**, design pós-`f2a52f0` confirmado como o atual
(§0). `ACC=44` segue do mesmo commit, não foi reconfirmado em separado — vale confirmar se sobrar
oportunidade, mas não bloqueia.

~~2. `f43_sim_build_*/wgt.mem` é peso cru INT8 ou V pré-transformado?~~ **Resolvido por conferência
própria (não precisou perguntar):** confirmado V pré-transformado do fluxo antigo — `git log`
mostra `wgt.mem` parado em `d896cac`, anterior a `f2a52f0`; o diff de `f2a52f0` em
`gen_f43_system_vectors.py` muda exatamente esse formato (36 `V_pre` de 16b → 9 coefs crus de
`RAW_W` bits). Os vetores commitados precisam ser regenerados antes de qualquer medição real de
Fase A — não são utilizáveis como estão.

Antes de gerar número que vai pro artigo, ainda em aberto:

1. **Qual `act_scale`/`wgt_scale` foi usado nesses exports?** Os metadados (`:449-450`) não estão
   commitados.
2. **Existe cópia do val do ImageNet-1k no lab?** É o único bloqueio sem solução trivial aqui —
   evita 6,7 GB de download e a burocracia de acesso.
