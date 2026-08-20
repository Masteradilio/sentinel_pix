<!-- decision_id: EXP014B_R4B_FAST -->
## EXP-014B-R4B-FAST — busca eficiente para FPR < 1%
Substitui a busca ampla anterior, que travava por explosão combinatória de quads/quints.
Meta final:
```text
FPR < 1,0%
FN total <= 5
```
Motivação técnica:
```text
R4A-FROZEN já atinge FPR<=1,5% com FP=1677 e FN=0.
Para FPR<1%, FP precisa ficar <=1123.
Como BLOQUEAR sozinho possui FP=1200, a nova busca precisa permitir demover parte de BLOQUEAR, não só CONFIRMAR.
```
Mudanças do script FAST:
```text
- minera apenas linhas elegíveis em CONFIRMAR/BLOQUEAR
- usa groupby vetorizado e poda candidatos antes do greedy
- não cria centenas/milhares de colunas auxiliares
- não faz quints exaustivos
- quads são limitados por combinações priorizadas
```
Comando recomendado:
```powershell
python scripts\exp_014b_r4b_fpr_lt1_fast_pathfinder.py --enable-quads --max-rules 120 --min-incremental-fp 1 --min-support 3
```
Se ainda não atingir a meta, aumentar aos poucos `--max-quad-combos` e `--max-candidates`, sem voltar para quints exaustivos.

---

<!-- decision_id: EXP014B_R4C_BLOCK_FP_REDUCTION_PROBE -->

## EXP-014B-R4C — Block FP Reduction Probe

**Motivo:** o R4B-FAST não atingiu FPR < 1% e quase não conseguiu reduzir FP de `BLOQUEAR`.

Baseline campeão atual:

```text
R4A-FROZEN
TP=1465
FP=1677
FN=0
FPR=1,492%
BLOQUEAR: TP=1293, FP=1200
CONFIRMAR: 172 fraudes, 477 normais
```

Meta final:

```text
FPR < 1%
FP <= 1123
FN total <= 5
```

Conclusão matemática:

```text
Como BLOQUEAR sozinho tem FP=1200,
não é possível atingir FP<=1123 mexendo só em CONFIRMAR.
```

R4C:

```text
Elegível apenas BLOQUEAR.
Ação testada: BLOQUEAR -> APROVAR.
Orçamento: FN total <= 5.
Busca por groupby/poda em segmentos e score thresholds.
```

Se R4C não atingir a meta, o gargalo passa a exigir novas features ou revisão do critério de BLOQUEAR.

---

<!-- decision_id: EXP014B_R4C_CONSOLIDATE_AND_R4D_SEVERITY_REBALANCE -->
 
## Consolidação R4C e próximo passo R4D
 
R4C foi aceito como resultado operacional bom, mesmo sem bater FPR < 1%.
 
R4C:
 
```text
Intervenção final:
TP=1460
FP=1195
FN=5
FPR=1,063%
 
BLOQUEAR:
TP=1288
FP=718
 
CONFIRMAR:
TP=172
FP=477
 
APROVAR:
FN=5
```
 
Nova leitura operacional:
 
```text
Nem todo FP é igual.
Normal em BLOQUEAR é pior que normal em CONFIRMAR.
Fraude em CONFIRMAR deveria ir para BLOQUEAR quando possível.
```
 
Próxima rodada:
 
```text
EXP-014B-R4D — Severity Rebalance after R4C
```
 
Objetivo:
 
```text
promover CONFIRMAR -> BLOQUEAR para capturar as 172 fraudes remanescentes
rebaixar BLOQUEAR -> CONFIRMAR para reduzir os 718 normais bloqueados
não alterar intervenção total
não criar novas fraudes em APROVAR
```
 
Métrica de sucesso:
 
```text
APROVAR fraud delta = 0
intervention_pred_unchanged = true
block_fp reduz
block_tp aumenta ou permanece alto

---

<!-- decision_id: EXP014B_R4D_FROZEN_AND_R4E_GLOBAL_LT1_SEVERITY -->

## Consolidação R4D e próxima rodada R4E

R4D conservador foi aceito como baseline operacional de severidade.

R4D:

```text
Intervenção global:
TP=1460
FP=1195
FN=5
FPR=1,063%

APROVAR:
111189 linhas, 5 fraudes

BLOQUEAR:
2125 linhas, 1410 fraudes, 715 normais

CONFIRMAR:
530 linhas, 50 fraudes, 480 normais
```

Ganho de severidade vs R4C:

```text
CONFIRMAR -> BLOQUEAR:
122 fraudes promovidas
0 normais promovidos

BLOQUEAR -> CONFIRMAR:
3 normais movidos
0 fraudes movidas

Intervenção total inalterada.
APROVAR inalterado.
```

Próxima rodada:

```text
EXP-014B-R4E — Global FPR < 1% + Severity Advance after R4D
```

Objetivos:

```text
1. Remover pelo menos 72 FPs globais:
   CONFIRMAR -> APROVAR, somente com TP loss = 0.

2. Promover o máximo possível das 50 fraudes remanescentes:
   CONFIRMAR -> BLOQUEAR.

3. Rebaixar o máximo possível dos 715 normais em BLOQUEAR:
   BLOQUEAR -> CONFIRMAR.

4. Manter FN total <= 5.
```

Critérios de sucesso:

```text
target_reached = true
final_intervention_metrics.fp <= 1123
final_intervention_metrics.fn <= 5
approval_fraud_delta = 0
```

---

<!-- decision_id: EXP014B_R4E_CONSOLIDATE_AND_R4F_BLOCK_DEESCALATION -->

## Consolidação R4E e ajuste fino R4F

R4E atingiu a meta global final:

```text
TP=1460
FP=1113
FN=5
FPR=0,990399%
target_reached=true
approval_fraud_delta=0
```

Distribuição R4E:

```text
APROVAR:
111271 linhas, 5 fraudes

BLOQUEAR:
2279 linhas, 1458 fraudes, 821 normais

CONFIRMAR:
294 linhas, 2 fraudes, 292 normais
```

Leitura operacional:

```text
O estado global é muito bom.
Ajuste fino restante: reduzir normais em BLOQUEAR, preferencialmente movendo para CONFIRMAR.
```

Próximo experimento:

```text
EXP-014B-R4F — Block-to-Confirm Fine Tune after R4E
```

Regra principal:

```text
Somente BLOQUEAR -> CONFIRMAR.
APROVAR intocado.
Intervenção global inalterada.
Default: block_tp_demoted_to_confirm = 0.
```

Critérios de sucesso:

```text
intervention_unchanged=true
approval_fraud_delta=0
block_tp_demoted_to_confirm=0
block_fp_demoted_to_confirm > 0
```

Se R4F tiver pouco ganho, manter R4E-FROZEN como baseline final e partir para feature engineering.

---

<!-- decision_id: EXP014B_R4F_FROZEN_AND_R4G_APPROVE_RESCUE -->

## Consolidação R4F e próxima rodada R4G

R4F deve ser congelado como baseline operacional ajustado:

```text
Intervenção global:
TP=1460
FP=1113
FN=5
FPR=0,990399%

BLOQUEAR:
1458 fraudes
766 normais

CONFIRMAR:
2 fraudes
347 normais

APROVAR:
5 fraudes
111266 normais
```

R4G tenta resolver a narrativa dos 5 FN:

```text
APROVAR -> CONFIRMAR
```

Restrições:

```text
manter FPR < 1%
folga de FP default = target_fp - FP atual = 1123 - 1113 = 10
não mover fraude de BLOQUEAR para CONFIRMAR
```

Critérios de sucesso:

```text
target_reached = true
final_intervention_metrics.fp <= 1123
approval_fraud_remaining = 0
block_tp_demoted_to_confirm = 0
```

---

<!-- decision_id: EXP014B_R4G_FAST -->

## R4G-FAST

A versão conservadora original do R4G ficou lenta porque fazia busca ampla em `APROVAR`, com mais de 111 mil linhas.

R4G-FAST troca a estratégia:

```text
APROVAR -> CONFIRMAR:
busca ancorada somente no perfil das fraudes aprovadas

BLOQUEAR -> CONFIRMAR:
busca no subconjunto pequeno de BLOQUEAR
```

Mantém as mesmas restrições:

```text
FPR < 1%
FP <= 1123
APROVAR -> CONFIRMAR limitado pela folga de FP
BLOQUEAR -> CONFIRMAR com zero fraude demovida
```

Com baseline R4F:

```text
FP atual = 1113
target_fp = 1123
folga = 10 normais
```

---

<!-- decision_id: EXP014B_R4G_FAST_CHAMPION_AND_NEXT_STRATEGIES -->
## Consolidação R4G-FAST
R4G-FAST foi aceito como melhor baseline geral até agora.
Motivo:
```text
R4F-FROZEN:
TP=1460
FP=1113
FN=5
FPR=0,990399%
R4G-FAST:
TP=1463
FP=1123
FN=2
FPR=0,999297%
```
Decisão:
```text
Mesmo usando toda a folga de FP, o ganho de reduzir fraudes aprovadas de 5 para 2 é operacionalmente superior.
```
Próxima consolidação:
```text
EXP-014B-R4G-FAST-FROZEN
```
Estratégias após frozen:
```text
1. Engenharia de features de relacionamento pagador-recebedor.
2. Reputação/idade/estabilidade do recebedor e chave PIX.
3. Modelo ordinal separado para severidade: APROVAR, CONFIRMAR, BLOQUEAR.
4. Calibração temporal e thresholds por segmento.
5. Revisão manual/active learning das 2 fraudes aprovadas e 766 normais bloqueados.

---

```

---

<!-- decision_id: EXP014B_R5B17_PIPELINE_HOMOLOGATION -->

## R5B17 - homologacao operacional no PipelineOrquestrador

O R5B17 reconstruiu o `venv` local e executou a primeira homologacao E2E do `PipelineOrquestrador` com a politica R5B14/R5B16 ativada:

```text
ENABLE_R5B14_POLICY=1
USE_PRECOMPUTED_FEATURES=1
```

Artefatos:

```text
resultados/experimentos/EXP-014B-R5B17-PIPELINE-HOMOLOGATION/
scripts/exp_014b_r5b17_pipeline_homologation.py
scripts/exp_014b_r5b17_frozen_alignment_audit.py
```

Resultado fraud-only na base MAF v3 completa:

```text
status=FAIL_R5B17_PIPELINE_HOMOLOGATION
n_frauds=1465
TP=754
FN=711
fraudes em APROVAR=711
fraudes em CONFIRMAR=0
pipeline_errors=0
```

Auditoria contra o frozen R4G:

```text
status=FAIL_R5B17_E2E_FROZEN_ALIGNMENT_GAP
e2e_approve_frauds=711
e2e_approve_frauds_that_frozen_would_intervene=711
e2e_approve_frauds_that_frozen_would_block=709
e2e_approve_frauds_that_frozen_would_confirm=2
```

Decisao:

```text
R5B17 nao homologa o baseline candidato no runtime.
O problema principal nao e a camada R5B14 isolada: o PipelineOrquestrador E2E atual nao reproduz o contrato frozen R4G que serviu como base do R5B16.

A proxima fase deve ser R5B18:
1. portar a politica R4G/R5B16 para uma funcao operacional runtime-safe; ou
2. reconstruir uma politica equivalente diretamente sobre as saidas E2E/MAF do PipelineOrquestrador.

Nao promover R5B16 para producao antes de fechar esse gap.
```

---

<!-- decision_id: EXP014B_R5B15_ZERO_FN_BASELINE -->

## Consolidação R5B15 - novo baseline candidato zero-FN

Após as fases R5B11 a R5B15, o baseline R4G foi preservado como base global e recebeu uma política operacional de severidade/intervenção por regras explícitas.

Baseline anterior campeão, R4G-FAST-FROZEN:

```text
Global:
TP=1463
FP=1123
FN=2
FPR=0,999297%

BLOQUEAR:
TP=1458
FP=766
FN=7
FPR=0,681622%
```

Novo baseline candidato, R5B15 core policy replay:

```text
Global:
TP=1465
FP=1123
FN=0
FPR=0,999297%
Recall=100%

BLOQUEAR:
TP=1465
FP=835
FN=0
FPR=0,743021%
Precision=63,695652%
```

Distribuição operacional final:

```text
APROVAR:
111256 transações
0 fraudes
111256 normais

BLOQUEAR:
2300 transações
1465 fraudes
835 normais

CONFIRMAR:
288 transações
0 fraudes
288 normais
```

Camadas da política R5B14/R5B15:

```text
1. CONFIRMAR -> BLOQUEAR
   27 transações, 5 fraudes, 22 normais

2. APROVAR -> BLOQUEAR
   49 transações, 2 fraudes, 47 normais

3. CONFIRMAR -> APROVAR
   remaining CONFIRMAR AND lgbm_raw <= 0,00001966
   47 transações, 0 fraudes, 47 normais
```

Evidência:

```text
EXP-014B-R5B15-CORE-POLICY-REPLAY:
status=PASS_R5B15_CORE_POLICY_REPLAY_MATCHED_R5B14
intervention_metrics_match_r5b14=true
block_metrics_match_r5b14=true
approve_frauds=0
confirm_frauds=0
```

Decisão:

```text
R5B15 substitui R4G-FAST-FROZEN como melhor baseline candidato offline.
Ele preserva FPR < 1%, elimina todos os falsos negativos conhecidos e concentra todas as fraudes em BLOQUEAR.
```

Restrições:

```text
A política está conectada ao PipelineOrquestrador por configuração versionada,
mas permanece desligada por default.

Flags:
ENABLE_R5B14_POLICY
r5b14_operational_zero_fn_enabled

Antes de produção, ainda é obrigatório executar replay batch completo no ambiente produtivo com dependências do PipelineOrquestrador.
```

---

<!-- decision_id: EXP014B_R5B16_OPERATIONAL_BASELINE_CANDIDATE -->

## R5B16 - consolidação do baseline operacional candidato

O R5B16 consolidou o R5B15 em um artefato candidato versionado:

```text
backend/artefatos_candidatos/exp014b_r5b16_operational_baseline/operational_baseline_candidate.json
```

Status:

```text
PASS_R5B16_OPERATIONAL_BASELINE_CANDIDATE_CONSOLIDATED
```

Gates consolidados:

```text
fpr_lt_1pct=true
fn_lte_5_outside_block=true
fn_eq_0=true
approve_frauds_eq_0=true
confirm_frauds_eq_0=true
all_core_replay_checks_pass=true
```

Decisão:

```text
R5B16 encerra a fase de consolidação offline do novo baseline.
O próximo trabalho deixa de ser mineração de política e passa a ser homologação operacional:
replay batch completo do PipelineOrquestrador, revisão semântica das regras e monitoramento de drift.
```
---

<!-- decision_id: EXP014B_R5B18_E2E_FROZEN_CONTRACT_HOMOLOGATION -->

## R5B18 - homologacao E2E do contrato frozen R4G/R5B16

R5B18 corrigiu a lacuna encontrada no R5B17: o `PipelineOrquestrador` E2E estava usando a decisao runtime como base, enquanto o baseline aprovado R5B16 depende do contrato frozen `r4g_fast_frozen_decisao_recommended` como decisao-base.

O contrato homologado permanece desligado por default e e ativado explicitamente por:

```text
ENABLE_R5B16_FROZEN_CONTRACT=1
ENABLE_R5B14_POLICY=1
USE_PRECOMPUTED_FEATURES=1
```

Artefatos:

```text
resultados/experimentos/EXP-014B-R5B18-E2E-FROZEN-CONTRACT-HOMOLOGATION/
scripts/exp_014b_r5b18_e2e_frozen_contract_homologation.py
```

Distribuicao operacional final:

```text
APROVAR:
111256 transacoes
0 fraudes
111256 normais

CONFIRMAR:
288 transacoes
0 fraudes
288 normais

BLOQUEAR:
2300 transacoes
1465 fraudes
835 normais
```

Metricas globais de intervencao:

```text
TP=1465
FP=1123
FN=0
TN=111256
Precision=56,607419%
Recall=100%
F1=0,72292129
FPR=0,999297%
```

Metricas de BLOQUEAR:

```text
TP=1465
FP=835
FN=0
TN=111544
Precision=63,695652%
Recall=100%
F1=0,77822045
FPR=0,743021%
```

Gates:

```text
FPR < 1%: true
FN fora de BLOQUEAR <= 10: true
FN fora de BLOQUEAR: 0
amostra E2E bate contrato vetorizado: true
```

Decisao:

```text
R5B18 homologa o contrato frozen R4G/R5B16 para demonstracao E2E.
O contrato usa R4G frozen como decisao-base e aplica o overlay R5B14/R5B16 por configuracao explicita.
```
---

<!-- decision_id: EXP014B_R5B22_OFFICIAL_CONSTRAINED_BASELINE -->

## R5B22 - novo baseline oficial com restricao APROVAR/CONFIRMAR

R5B22 substitui o baseline operacional anterior como novo baseline global oficial do modelo para demonstracao executiva.

Objetivo da rodada:

```text
Maximizar a precisao de BLOQUEAR.
Permitir no maximo 5 fraudes em APROVAR.
Permitir no maximo 10 fraudes em CONFIRMAR.
Manter a distilacao LGBM aluno do contrato R5B16 acima de 80% das metricas do baseline.
```

Artefatos oficiais salvos:

```text
backend/artefatos/r5b22_official_baseline_policy.json
backend/artefatos/r5b22_official_baseline_summary.json
backend/artefatos/model_lgbm_distilled_r5b22_intervention.joblib
backend/artefatos/model_lgbm_distilled_r5b22_block.joblib
backend/artefatos/model_lgbm_distilled_r5b22_metadata.json
```

Artefatos de experimento:

```text
resultados/experimentos/EXP-014B-R5B22-OFFICIAL-CONSTRAINED-BASELINE/
scripts/exp_014b_r5b22_official_constrained_baseline.py
```

Distribuicao operacional final:

```text
APROVAR:
111305 transacoes
2 fraudes
111303 normais

CONFIRMAR:
326 transacoes
10 fraudes
316 normais

BLOQUEAR:
2213 transacoes
1453 fraudes
760 normais
```

Metricas globais:

```text
TP=1463
FP=1076
FN=2
TN=111303
Precision=0,57621111
Recall=0,99863481
F1=0,73076923
FPR=0,957474%
```

Metricas de BLOQUEAR:

```text
TP=1453
FP=760
FN=12
TN=111619
Precision=0,65657479
Recall=0,99180887
F1=0,79010332
FPR=0,676283%
```

Comparacao com R5B16/R5B18:

```text
Normais em BLOQUEAR: 835 -> 760 (-75)
Precision BLOQUEAR: 0,63695652 -> 0,65657479
F1 BLOQUEAR: 0,77822045 -> 0,79010332
FPR BLOQUEAR: 0,743021% -> 0,676283%
Fraudes em APROVAR: 0 -> 2
Fraudes em CONFIRMAR: 0 -> 10
```

Distilacao LGBM aluno do contrato R5B16:

```text
Global:
Precision ratio=99,902084%
Recall ratio=99,863481%
F1 ratio=99,888128%

BLOQUEAR:
Precision ratio=100%
Recall ratio=100%
F1 ratio=100%
```

Observacao tecnica:

```text
O LGBM aluno e uma distilacao do contrato R5B16/R5B18. Ele usa sinais do professor,
incluindo r4g_fast_frozen_decisao_recommended, r5b14_rule_applied e
r5b14_layer_applied. Portanto, deve ser apresentado como aluno do contrato operacional,
nao como LGBM puro treinado apenas com features brutas.
```

Decisao:

```text
R5B22 passa todos os gates definidos e passa a ser o novo baseline global oficial.
```
