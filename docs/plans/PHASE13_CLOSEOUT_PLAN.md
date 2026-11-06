<!-- Handoff para outra sessao/maquina (2026-11-04). Origem: plano aprovado no Claude Code. Regra do projeto: nada de treino, submissao ou commit sem pedido explicito. Estado de jobs (835242-835245) e verificado em 2026-11-04; reconfira com squeue/sacct antes de agir. -->
# Plano: o que dá para adiantar agora (Fase 13 → fechamento), sem erro

## Context
Fase 13 está 18/18 em seed 42 (commits 75a4a65, 9082ce8 já no origin). Falta: seeds 43/44 (jobs 835242-835245 no PCAD), os
3 itens "depois da Fase 13" do TODO (tirar `exclude: tupi4`; reavaliar as 152 células em tupi3-6; relatório -- **fora de escopo**,
o usuário disse que `ic_report.tex` é das outras fases) e a etapa B. Regras do projeto que ditam o plano: reprodutibilidade
(proveniência limpa, artefatos salvos), nunca rodar treino/notebook nem commitar sem pedido explícito, plano antes de editar.
Nada abaixo foi executado; cada bloco termina num ponto em que o usuário confirma.

Estado verificado (2026-11-04, somente leitura):
- PCAD: 835242 (tupi6) e 835243 (tupi1) rodando; 835244/835245 pendentes. **8 jobs do usuário no total** (4 seeds + `tr_C6_g3` array
  de 2 + `exp_C4n` + `exp_C4n_e2`, estes do Winograd-FPGA) contra o teto de 10 da QOS `normal` → só 2 vagas livres.
- PCAD `~/dnn_study` está em 23c24b5 com `outputs/pcad/aggregates/phase_13_hybrids_comparison.csv` **modificado** e a 18ª run
  **untracked**; o laptop já commitou os dois → `git pull --ff-only` no PCAD vai parar (receita do CLAUDE.md abaixo).
- Das 152 células da Fase 11: 70 rodaram em 4090+i9 (tupi3-6), 25 em 4090+Xeon (tupi1/2), 31 em beagle (1080 Ti), 26 no nyx (4060 Ti).
  `scripts/train.py` não tem modo "só benchmark" (flags: --experiment --runtime --device --model --dry-run --smoke): reavaliar =
  refazer FP32/QAT/INT8 eval + latência; FP32/QAT em GPU diferente da que treinou pode mudar dígitos (CLAUDE.md).
- tupi4: desde 24/10 houve 2 FAILED (832921 exit 1, 832951 segfault exit 11, ambos na manhã de 24/10, quando o `exclude` foi criado)
  e 18 COMPLETED depois (Winograd-FPGA, jobs curtos). Evidência fraca para tirar o `exclude` de jobs de 5 h.

## Bloco A -- Pipeline de sincronização das seeds (preparar o procedimento; executar quando os 4 jobs terminarem)
Sem mudar arquivos agora; é a lista de checagens para não errar quando chegarem:
1. `ssh pcad sacct -j 835242,835243,835244,835245 -X -o JobID,State,ExitCode,Elapsed,NodeList` → exigir `COMPLETED 0:0`. Se algum
   sair 1 após escrever o summary (como os MobileNetV2 na guarda `agreement_qat_int8`), registrar e checar `agreement_qat_int8` ≥ 0,90.
2. Por run: `rsync -av --exclude='*.pth' --exclude='*_best.pth.gz' --exclude=tensorboard --exclude=logs
   pcad:dnn_study/outputs/pcad/phase_13_hybrids_seed4{3,4}/<modelo> ...` (o diretório da run é por **experimento**: confirmar o nome
   no yaml `configs/experiments/phase_13_hybrids_seed43.yaml`, não presumir `phase_13_hybrids/`). Conferir md5 do `.pth.gz` e que
   `qat_*.pth.gz` + `*_meta.json` + `resolved_config.json` + `results/*` chegaram.
3. Checar no `resolved_config.json`/summary: `git_dirty` false, `quant_protocol` 2026-10-07, seed 43/44, host/GPU/CPU gravados,
   `provenance_history` (se retomou), best epoch ≥ 80% do orçamento (`early_best` falso).
4. CSV agregado: `scp` do PCAD para o scratchpad, `diff` por chave (só linhas novas + ruído de float < 1e-6, como hoje), então copiar.
5. `python -m scripts.phase13.hybrids` (já desenha as outras seeds como pontos e escreve `contrasts_by_seed.csv`); ler a saída de
   média/DP e o par do skip por seed (42/43/44) e do 2×2 vs VGG16.
6. `pytest tests/test_phase13_hybrids.py tests/test_config.py tests/test_registry.py -q` (nada de treino).
7. Docs: nova seção em `docs/logs/PHASE13_LOG.md`, linha da Fase 13 no `CLAUDE.md`, `BEST_MODELS.md` ("Runs still out"/Best), checkbox
   no `TODO.md`. Commit só com pedido explícito.
**Receita do pull no PCAD** (antes de qualquer commit do laptop ser puxado): mover para `~/dnn_study_prepull_untracked_<data>/` só os
caminhos untracked que o pull sobrescreve; copiar o CSV modificado e `git checkout --` nele; `git pull --ff-only`; `cmp` cada arquivo
movido contra o checked-out; só então apagar as cópias (CLAUDE.md, Layout).

## Bloco B -- `exclude: tupi4` (decisão de 2026-11-06: **manter**; ver "Atualização 2026-11-06" no fim)
Mudança: remover a linha em `configs/slurm/tupi_4090.yaml` e `configs/slurm/tupi_beagle.yaml` (nenhum teste fixa `exclude`: grep em tests/).
Só afeta submissões futuras via `scripts.cluster`. Sequência segura:
1. Esperar 835244/835245 **iniciarem** (evita mexer no HEAD do PCAD sob jobs pendentes que gravam `git_hash`).
2. Prova antes de confiar: `python -m scripts.cluster submit --experiment phase_13_hybrids_seed43 --runtime pcad --slurm tupi_4090 --model <cell> --smoke`
   com `--nodelist tupi4`/`sbatch --nodelist` (ver se o cluster.py aceita; senão `scontrol`/arg extra) -- 1 job de 1 época, saída descartada.
   Só tira o `exclude` se passar; se falhar, deixa e anota no TODO. **Decisão do usuário** (a evidência de 24/10 é de 2 falhas).
3. Commit + push no laptop, `git pull --ff-only` no PCAD (receita acima).

## Bloco C -- Reavaliação em tupi3-6 (piloto de 2 células; decisão do usuário)
Objetivo: latência comparável entre as 152 células sem corromper FP32/QAT. Piloto antes de escalar:
1. Config nova `configs/slurm/tupi36.yaml` (partition tupi, `exclude: tupi1,tupi2,tupi4`, gres/cpus 8/mem/time/requeue/signal como
   `tupi_4090.yaml`; hoje não existe -- `beagle.yaml` foi o precedente de uma reavaliação por config).
2. Escolher 2 células baratas: **uma treinada no beagle** e **uma no nyx** (listar por `all_runs.csv`: coluna `host`), de um bloco com
   checkpoint FP32+QAT em `~/dnn_study` (`outputs/pcad/...` -- nyx já sincronizou para o PCAD). Backup = git (os tracked do
   laptop são idênticos); não precisa arquivar à mão.
3. Rodar por `scripts.cluster submit --experiment <exp> --runtime pcad --slurm tupi36 --model <cell>` (um job por vez, 2 vagas livres).
   Esperado: FP32/QAT "resume" como no-op, só evaluations; `provenance_history` ganha o job; `test_resubmitting_a_finished_run_*` já
   cobre que só as latências mudam.
4. Critério: `test_{fp32,qat,int8}_top1` e logits iguais ao registrado (tolerância 0 nos tops, comparar `*_test_logits.npz` com
   `prediction_agreement`). Igual → escopo seguinte = as 82 fora de tupi3-6 (25 tupi1/2 + 31 beagle + 26 nyx), alimentadas por
   `scripts/pcad/feed_queue.sh` (MAX=8 já deixa 2 vagas; hoje há 8 jobs → esperar o Winograd-FPGA liberar). Diferente → não
   reavaliar essas; manter a latência original e mostrar `machine` por barra (figura 17 já filtra a máquina mais comum).
5. Nada disso é submetido antes de o usuário aprovar este plano **e** dizer "submeta"; o usuário já ordenou que isso venha depois
   das seeds -- manter, a menos que ele libere o piloto antes.

## Bloco D -- Etapa B (Bottleneck/Fire sobre o melhor corpo): só plano de código + testes (nada escrito nesta rodada)
Depende das seeds para escolher o corpo (candidatos: `k2x2stacked` e `k2x2residual`, stride 1). Mudanças previstas:
- `models/compensation.py`: `_AlexBottleneck(in_ch, out_ch, stride, reduction)` (linha ~20) e `_FireModule(in_ch, squeeze_ch, expand_ch)`
  (~382) ganham `kernel_size: int = 3` **com default que mantém o comportamento**; o 3×3 interno usa `SymmetricPad2d`
  (`models/baselines.py`) quando k=2 e stride 1. Seguir o precedente de `_ResBlock(kernel_size=, first_kernel=)` (~305).
- Montagem: reutilizar `AlexNetStacked(stem_kernel=, stem_stride=, residual=)` (`models/alexnet_variants.py`) trocando a unidade por
  estágio, em vez de nova classe do zero.
- `ml/model_registrations.py`: registrar as células pela fábrica `_register_cell`/`CELL_FACTORS` (nomes verificados contra a rede
  construída em `tests/test_registry.py`); `fuse_map=find_fuse_groups(...)`.
- Config: `configs/experiments/phase_13_stage_b.yaml` com `extends: _protocols/no_patience.yaml` (receita exata da Fase 11;
  `tests/test_config.py` já exige isso para `phase_11_`/`phase_13_`).
- Testes: (a) registry -- params/MACs de `alexnet_final_*` e `alexnet_fire_bypass` **inalterados** (guarda do default);
  (b) `tests/test_quantization.py` -- fatia das novas células QAT→INT8 contra a definição da literatura; (c) 1 época
  `--smoke` local de 1 célula (FP32→QAT→INT8, guardas de kernel INT8 e agreement) -- só com autorização; (d) `tests/test_config.py`.
- Referências já medidas p/ comparação: FinalBottleneckResidual 55,83, FireBypass 55,19, FinalFireResidual 54,96,
  FinalBottleneckFire 54,40, FinalDepthwiseFire 45,76 (FP32, seed 42, `phase_11_families`).

## Verificação (end-to-end, sem treino)
- `pytest tests/ -q` (ou os 4 arquivos citados) antes de qualquer commit de código/config.
- `git status` limpo em ml/models/scripts/configs antes de submeter (o `scripts.cluster` recusa dirty); `git_dirty` false nas runs.
- Para cada bloco executado: conferir `squeue -u rsdsouza` (teto 10), `sacct` e o `resolved_config.json` das runs tocadas.

## Ordem recomendada
A (quando as seeds terminarem) · B (após as seeds iniciarem + smoke em tupi4) · C-piloto (após aprovação explícita) · D (código
após as seeds escolherem o corpo). Nada além do plano é executado agora.

## Atualização 2026-11-06
- **Bloco B (`exclude: tupi4`): manter.** Com os jobs de todos os usuários (>= 30 min, desde 01/10): tupi4 perdeu 10 de 126 por sinal (9 SIGSEGV + 1 SIGABRT)
  contra 9 de 903 nos outros nós (Fisher p = 2e-5); os nós i9 (tupi3-6) juntos 19 de 678 contra 0 de 351 nos Xeon tupi1/2 (p = 3e-4). O 835348 morreu no tupi3
  com "Illegal instruction". Causa não verificada (degradação Vmin-shift do Raptor Lake ou RAM); só o admin lê dmesg/MCE e roda memtest. O smoke de 1 época
  **deixa de ser o critério**: não detecta esse tipo de falha (os crashes vieram após 1h45, 5h49 ou segundos depois dos logits). Para tirar o `exclude`: admin
  confere o nó, ou várias runs longas reais terminam lá. Run que morre por sinal num nó i9 é refeita do zero, não retomada (retomar não restaura o RNG).
  Detalhes e ressalvas: `docs/logs/PHASE13_LOG.md`, "2026-11-06".
- **Bloco C (piloto de reavaliação): adiado** a pedido do usuário até concluir as runs mais importantes (seeds da Fase 13, Fase 14). Quando for feito: são só
  avaliações, cabem na QOS `short` (2 h, prioridade 1000, 10 em execução / 25 enviados); tupi3-6 são os nós com crashes, então uma reavaliação morta por sinal
  é refeita.
- **Bloco A (seeds), estado:** 2x2 -- 3 de 4 prontas no PCAD (não sincronizadas); a 4ª (`k2x2stacked` seed 44, 835352) roda no beagle desde 06/11 09:20
  (`scontrol update JobId=835352 MinMemoryNode=15360 Partition=tupi,beagle`; limite 07/11 09:20). 3x3 -- 835433 rodando (tupi1), 835435-835437 pendentes
  (primeiras linhas de `~/queue_phase14.txt`). Sincronizar tudo quando o 835352 terminar.
- **Bloco D (etapa B):** continua esperando as seeds.
