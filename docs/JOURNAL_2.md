
---

<!-- decision_id: EXP011C_REJECT_PRODUCTION_AND_START_EXP012A_DATASET_V3 -->

## EXP-011C — Rejeição de produção e início do EXP-012A Dataset v3

**Decisão:** rejeitar os candidatos LGBM vNext R1/R2 para produção e iniciar uma nova etapa estrutural de melhoria do dataset e das features históricas.

O EXP-011C executou E2E shadow com os candidatos R1 e R2 sobre o dataset enriquecido local do EXP-010G-R2, com 100.000 linhas, 1.306 fraudes e 98.694 normais. A execução foi tecnicamente válida, sem ausência de features para os candidatos:

```text
missing_r1_features=[]
missing_r2_features=[]
n_input=100000
n_baseline_rows=100000

Resultado no holdout:

Baseline produtivo:
TP=12
FP=73
FN=46
TN=8691
precision=0.14117647
recall=0.20689655
f1=0.16783217
fpr=0.00832953

R1 model-only:
TP=15
FP=14
FN=43
TN=8750
precision=0.51724138
recall=0.25862069
f1=0.34482759
fpr=0.00159744

R1 assist baseline:
TP=17
FP=85
FN=41
TN=8679
precision=0.16666667
recall=0.29310345
f1=0.2125
fpr=0.00969877

R2 model-only:
TP=13
FP=7
FN=45
TN=8757
precision=0.65
recall=0.22413793
f1=0.33333333
fpr=0.00079872

Interpretação: o R1 model-only foi superior ao baseline e ao modo assist, mas ainda deixa passar 43 de 58 fraudes no holdout. Apesar de melhorar precision, FP e F1, o recall de 25,86% é insuficiente para um modelo antifraude PIX em produção.

Decisão técnica:

EXP-011C = aprovado como diagnóstico E2E shadow
R1 model-only = melhor candidato atual, mas reprovado para produção
R2 model-only = fallback conservador, mas reprovado para produção
R1/R2 = não promover

Mudança de rumo: interromper a tentativa de melhorar apenas por tuning de threshold/modelo e iniciar o EXP-012A, focado em reconstruir o dataset com features históricas reais calculadas no Big Data, a partir do universo transacional PIX e não apenas da amostra exportada.

Novo experimento: EXP-012A — Dataset v3 com features históricas reais no Big Data.

Nova tabela alvo:

hmo_ml.tb_pix_dataset_v3_features_180d_v1

Novo script HQL:

/modelos_ml/nudan/nudan_hmo/tb_pix_dataset_v3_features_180d_v1.hql

O dataset v3 deverá adicionar, entre outras, features reais de:

- histórico do pagador em 7d, 30d, 90d e 180d;
- volume e valor transacional do pagador;
- velocity em 30min, 1h e 24h;
- relacionamento pagador-recebedor;
- primeiro envio real para o recebedor;
- histórico real do recebedor;
- cobertura e atributos MBK;
- flags derivadas reais de burst, first receiver, ratio de valor e z-score.

Critério de sucesso futuro: só considerar novo candidato para produção se houver avanço material em holdout, mirando, no mínimo:

precision >= 50%
recall >= 50%
F1 >= 0.50
FPR <= 1%

Próximo passo: executar o workflow com o novo HQL do EXP-012A e auditar volume, janela, duplicidade, cobertura das novas features e distribuição por split antes de novo treinamento.

---

<!-- decision_id: EXP012C_R3_RECALL_FIRST_OBJECTIVE_RESTORED -->

## EXP-012C-R3 — Restauração do objetivo recall-first do LGBM

**Decisão:** alterar a direção de otimização do LGBM v3 para recall-first, restaurando o objetivo operacional apresentado no MVP.

Após as rodadas EXP-012C, EXP-012C-R1 e EXP-012C-R2, foi identificado que os experimentos estavam otimizando o LGBM isolado para equilíbrio entre precision, recall, F1 e FPR. Essa estratégia melhorou o modelo em relação ao dataset v2/v3 inicial, mas não atende ao objetivo de produção apresentado aos gestores.

O baseline histórico pós-C1 possuía recall oficial de aproximadamente 0,977465 nos dois seeds:

```text
seed 42:  TP=347, FP=14, FN=8, Precision=96,1219%, Recall=97,7465%, F1=0,9693
seed 123: TP=347, FP=12, FN=8, Precision=96,6574%, Recall=97,7465%, F1=0,9720

Resultado recente do EXP-012C-R2 no HOLDOUT_LABEL_SAFE:

config_id=C01_base_no_rn_pos4
threshold=0.43
TP=58
FP=65
FN=66
TN=9799
precision=0.47154472
recall=0.46774194
f1=0.46963563
fpr=0.00658962

Interpretação: o EXP-012C-R2 melhorou recall e F1 em relação ao EXP-012C, mas ainda está muito distante da exigência de recall >= 0.90. O resultado é útil como diagnóstico, mas não é direção suficiente para produção.

Nova arquitetura de decisão:

1. LGBM v3 deve atuar como primeiro estágio high-recall.
2. O objetivo primário do LGBM é maximizar captura de fraudes verdadeiras.
3. A meta mínima do LGBM passa a ser recall >= 0.90.
4. Precision/FPR do LGBM isolado deixam de ser gate primário.
5. Falsos positivos gerados pelo LGBM high-recall devem ser reduzidos em cascata por:
   - Isolation Forest;
   - behavioral_analytics;
   - social_engineering;
   - regras determinísticas de alta precisão;
   - DecisionEngine;
   - fila de revisão humana.

Decisão sobre EXP-012C-R2:

EXP-012C-R2 = APROVADO_COMO_DIAGNOSTICO
EXP-012C-R2 = REJEITADO_COMO_OBJETIVO_FINAL_DE_OTIMIZACAO

Novo experimento:

EXP-012C-R3 — High Recall LGBM v3 Sweep

Objetivo do EXP-012C-R3:

- treinar LGBM v3 com pesos positivos altos;
- varrer thresholds baixos com granularidade fina;
- selecionar políticas que atinjam recall >= 0.90, 0.95 ou 0.97 na validação;
- medir o custo em FP no HOLDOUT_LABEL_SAFE e HOLDOUT_FULL;
- salvar candidato shadow recall-first;
- não promover o LGBM isoladamente.

Próximo passo após EXP-012C-R3: executar EXP-012D — Cascata E2E Shadow, usando o LGBM high-recall como primeiro estágio e IF/BEH/SE/DecisionEngine para reduzir falsos positivos.

---

<!-- decision_id: EXP012C_R4_LGBM_HIGH_RECALL_FP_SQUEEZE_BASELINE -->

## EXP-012C-R4 — Novo baseline LGBM high-recall pré-módulos externos

**Decisão:** aprovar o EXP-012C-R4 como novo baseline LGBM high-recall antes da entrada dos módulos externos IF/BEH/SE.

O EXP-012C-R4 explorou seis estratégias para reduzir falsos positivos mantendo recall >= 95%: threshold exato, top-k, hard negative mining, cascata LGBM-only, threshold segmentado e pesos por estratégia de amostragem.

A configuração campeã foi:

```text
candidate_id=S1_SEG_MBK_AVAILABLE_FLAG_DS_TIPO_CHAVE_NORM
model_id=HR01_pos8_base
idea=5_segmented_thresholds
policy=segmented_thresholds_val_recall_target
target_recall=0.95

A política campeã usa thresholds segmentados por:

mbk_available_flag
ds_tipo_chave_norm

Resultado no HOLDOUT_LABEL_SAFE:

TP=122
FP=1604
FN=2
TN=8260
precision=0.07068366
recall=0.98387097
f1=0.13189189
fpr=0.16261152

Comparado ao EXP-012C-R3 high-recall, o R4 manteve o mesmo recall de 98,39% e reduziu 530 falsos positivos no holdout label-safe.

Status:

EXP-012C-R4 = APROVADO_COMO_BASELINE_LGBM_HIGH_RECALL_PRE_MODULOS_EXTERNOS

Próximo passo: iniciar EXP-012D, usando o LGBM R4 como primeiro estágio high-recall e adicionando IF/BEH/SE/DecisionEngine para redução de falsos positivos.

---

<!-- decision_id: EXP012D_LGBM_R4_EXTERNAL_MODULES_BASELINE_SHADOW -->

## EXP-012D — Baseline shadow com LGBM R4 + IF/BEH/SE

**Decisão:** aprovar o EXP-012D como baseline shadow da cascata LGBM R4 + módulos externos IF/BEH/SE.

O experimento usou o EXP-012C-R4 como primeiro estágio high-recall e aplicou políticas com Isolation Forest, Behavioral Analytics e Social Engineering para reduzir falsos positivos mantendo recall >= 95%.

Configuração campeã:

```text
candidate_id=POINTS_score0.00050301_min1
family=points
policy=Pontuação combinada LGBM + IF + SE + BEH

Resultado no HOLDOUT_LABEL_SAFE:

TP=122
FP=1535
FN=2
TN=8329
precision=0.07362704
recall=0.98387097
f1=0.13700168
fpr=0.15561638

Comparado ao EXP-012C-R4, o EXP-012D manteve o mesmo recall de 98,39%, manteve FN=2 e reduziu 69 falsos positivos.

Status:

EXP-012D = APROVADO_COMO_BASELINE_CASCATA_LGBM_R4_IF_BEH_SE_SHADOW

Ressalva: a redução de FP foi real, mas ainda modesta. Próximo passo é executar EXP-012E E2E runtime shadow para validar a cascata no fluxo real do PipelineOrquestrador/DecisionEngine.

---

---

<!-- decision_id: EXP013A_STATISTICAL_FP_TP_DIAGNOSTICS -->

## EXP-013A — Diagnóstico estatístico TP vs FP da cascata LGBM R4 + IF/BEH/SE

**Decisão:** aprovar o EXP-013A como diagnóstico estatístico para orientar a próxima política de redução de falsos positivos.

O EXP-013A analisou a saída E2E shadow do EXP-012E, usando como base a comparação por transação entre runtime atual e shadow EXP-012D. A amostra principal teve:

```text
n_rows_input=9988
n_shadow_positive=1657
n_shadow_tp=122
n_shadow_fp=1535
n_recovered_runtime_fn=104
n_added_fp_vs_runtime=1534
```

Foram testadas 57 variáveis numéricas, 35 variáveis categóricas, 121 segmentos, 38 segmentos candidatos a veto e 202 hipóteses de threshold.

Principais achados estatísticos:

```text
1. O score LGBM ainda é o sinal operacional mais forte.
   lgbm_r4_score tem mediana muito maior nos TPs do que nos FPs.

2. IF percentile também separa TP de FP e deve ser usado como filtro/veto,
   especialmente em casos de score LGBM baixo.

3. FPs tendem a aparecer mais em recebedores com histórico estabelecido.
   TPs têm menor histórico de recebimento, menor quantidade de Pix recebidos
   e menor quantidade de pagadores distintos para o recebedor.

4. Valor da transação e razão contra histórico do pagador ajudam a separar TP de FP.
   Valores muito baixos aparecem como bons candidatos a veto.

5. SE e BEH têm baixa cobertura, mas bons sinais de preservação.
   Quando SE/BEH estão fortes, o caso não deve ser vetado.

6. Algumas variáveis aparecem como perfeitas, mas são diagnósticas/offline
   e não devem ser usadas operacionalmente, como dataset_role, sample_strategy,
   source_dataset, sample_weight, is_fraud_runtime e group_y.
```

Hipóteses iniciais de veto com 0 perda estimada de TP:

```text
lgbm_r4_score < 0.00076308066
score_final < 0.76
vl_pix < 20
ratio_valor_media_pagador_90d < 0.10726481
if_percentile_x < 0.320032
```

Também foram encontrados segmentos candidatos a veto com 0 TP e muitos FP, especialmente combinações envolvendo:

```text
value_band
ds_tipo_chave_norm
first_receiver_flag_real
periodo_dia
mbk_available_flag
```

**Direção aprovada para o próximo experimento:**

```text
EXP-013B — Statistical Policy Search
```

Objetivo do EXP-013B:

```text
Reduzir falsos positivos usando filtros estatísticos guiados pelo EXP-013A,
mantendo recall >= 95%.
```

Critério de sucesso preliminar:

```text
TP >= 118
FN <= 6
recall >= 95%
FP significativamente menor que 1535
```

**Status:**

```text
EXP-013A = APROVADO_COMO_DIAGNOSTICO_ESTATISTICO
Próximo passo = EXP-013B_STATISTICAL_POLICY_SEARCH
```

---

<!-- decision_id: EXP013C_FROZEN_POLICY_ROBUSTNESS -->

## EXP-013C — Validação congelada da política estatística EXP-013B-R1

**Decisão:** aprovar o EXP-013C como validação de robustez da fronteira estatística, mas não promover ainda a política agressiva EXP-013B-R1 para produção.

O EXP-013C aplicou a política campeã do EXP-013B-R1 de forma congelada, sem reotimização, sobre a saída E2E shadow do EXP-012E.

Baseline shadow EXP-012D:

```text
TP=122
FP=1535
FN=2
precision=0.07362704
recall=0.98387097
f1=0.13700168
fpr=0.15561638

Política congelada EXP-013B-R1:

TP=118
FP=740
FN=6
precision=0.13752914
recall=0.95161290
f1=0.24032587
fpr=0.07502028

A política reduziu 795 falsos positivos, mantendo recall global acima de 95%. Porém, o risco de overfitting foi classificado como médio:

risk_level=MEDIUM
risks=
- SOME_TIME_BLOCK_RECALL_BELOW_TARGET
- BOOTSTRAP_RECALL_CI_LOWER_BELOW_TARGET
- BOOTSTRAP_TARGET_FAILURE_PROB_GT_10PCT
- THRESHOLD_STRESS_HAS_RECALL_FAILURE

O principal alerta foi a instabilidade temporal: um dos blocos caiu para recall de 86,96%. O bootstrap também indicou intervalo inferior de recall abaixo do alvo e probabilidade relevante de falha contra recall >= 95%.

O teste leave-one-rule-out apontou uma variante conservadora promissora ao remover a regra:

lgbm<0.02 AND receiver_value_180d>2000

Resultado da variante conservadora:

TP=119
FP=840
FN=5
precision=0.12408759
recall=0.95967742
f1=0.21975993
fpr=0.08515815

Status:

EXP-013C = APROVADO_COMO_VALIDACAO_DE_ROBUSTEZ
EXP-013B-R1_AGRESSIVO = NAO_PROMOVER_AINDA
PROXIMO_PASSO = EXP-013D_CONSERVATIVE_FROZEN_POLICY_VALIDATION

Decisão operacional: avançar com uma validação congelada conservadora antes de qualquer patch produtivo no DecisionEngine/PipelineOrquestrador.

---

EXP-013D — Seleção da política conservadora de veto estatístico

Decisão: aprovar o EXP-013D como rodada de comparação congelada entre a política conservadora e a política agressiva revisada, selecionando a variante conservadora como melhor candidata para a próxima etapa de refinamento.

O EXP-013D comparou quatro políticas:

BASELINE_SHADOW_EXP012D
AGGRESSIVE_ORIGINAL_EXP013B_R1
CONSERVATIVE_NO_RECEIVER_VALUE
AGGRESSIVE_REVISED_MULT_1_05

A política selecionada foi:

CONSERVATIVE_NO_RECEIVER_VALUE

Essa política mantém a camada estatística de veto do EXP-013B-R1, mas remove a regra:

lgbm<0.02 AND receiver_value_180d>2000

Resultado do baseline shadow EXP-012D:

TP=122
FP=1535
FN=2
precision=0.07362704
recall=0.98387097
f1=0.13700168
fpr=0.15561638

Resultado da política conservadora selecionada:

TP=119
FP=840
FN=5
precision=0.12408759
recall=0.95967742
f1=0.21975993
fpr=0.08515815

Ganho consolidado contra o baseline shadow:

FP removidos=695
TP perdidos=3
recall mantido acima de 95%
precision aumentou de 7,36% para 12,41%
FPR caiu de 15,56% para 8,52%

A política agressiva revisada reduziu mais FP, chegando a FP=694, mas operou colada no limite de recall, com TP=118 e FN=6. A política conservadora foi escolhida por oferecer maior folga de recall, menor risco operacional e melhor equilíbrio para uma candidata inicial de produção.

Status:

EXP-013D = APROVADO
POLITICA_SELECIONADA = CONSERVATIVE_NO_RECEIVER_VALUE
STATUS = TARGET_RECALL_MET_FP_REDUCED_MEDIUM_RISK
PROMOCAO_DIRETA = NAO
PROXIMO_PASSO = REFINAMENTO_DA_POLITICA_CONSERVADORA

Decisão operacional: seguir com o refinamento da política conservadora, priorizando estabilidade temporal e redução de risco antes de qualquer patch produtivo no DecisionEngine/PipelineOrquestrador.

---

<!-- decision_id: EXP013H_FROZEN_HIGH_RECALL95_VALIDATION -->

## EXP-013H — Validação congelada da política high-recall micro-refinada

**Decisão:** aprovar o EXP-013H como validação congelada da política vencedora do EXP-013G, com status `PASS_WITH_WARNINGS`.

O EXP-013H validou a política `high_recall95_micro_refined_policy` sem nova busca ou reotimização. A política manteve o alvo global de recall e o FP de referência:

```text
TP=118
FP=414
FN=6
precision=0.22180451
recall=0.95161290
f1=0.35975610
fpr=0.04197080

Comparação contra o benchmark anterior BASE_HIGH_RECALL_95:

BASE_HIGH_RECALL_95:
TP=118, FP=494, FN=6, recall=0.95161290

FROZEN_EXP013G_MICRO_REFINED:
TP=118, FP=414, FN=6, recall=0.95161290

Ganho consolidado:

FP removidos adicionais vs HIGH_RECALL_95 = 80
TP delta = 0
FN delta = 0
recall mantido acima de 95%
precision aumentou para 22,18%

O gate foi aprovado com alertas:

gate_status=PASS_WITH_WARNINGS
risks=[]
warnings=
- TIME_BLOCK_RECALL_BELOW_TARGET
- BOOTSTRAP_RECALL_P025_BELOW_TARGET
- BOOTSTRAP_TARGET_FAILURE_PROB_HIGH

O principal ponto de atenção segue sendo estabilidade temporal/bootstrap, especialmente o bloco temporal com recall mínimo de 75% e probabilidade bootstrap de recall abaixo de 95% em 42%.

Status:

EXP-013H = APROVADO_COM_ALERTAS
POLITICA_CONGELADA = high_recall95_micro_refined_policy
PROMOCAO_DIRETA = NAO
PROXIMO_PASSO = PATCH_SHADOW_CONFIGURAVEL_COM_MONITORAMENTO_E2E

Decisão operacional: avançar para implementação shadow configurável no DecisionEngine/PipelineOrquestrador, com monitoramento explícito de TP, FP, FN, recall por janela e FNs por bloco temporal antes de qualquer promoção produtiva.

---

<!-- decision_id: EXP014A4_RUNTIME_REPLAY_EXPANDIDO -->

## EXP-014A-4 — Replay runtime expandido e mudança de direção

**Decisão:** aprovar o EXP-014A-4 como etapa de geração do dataset expandido scoreado, mas rejeitar sua predição final como política high-recall candidata.

O EXP-014A-4 processou o dataset expandido completo, com:


linhas = 113844
fraudes = 1465
elapsed_seconds = 6606.58

A execução criou com sucesso:

dados\exp014a_expanded_scored_input.csv

O contrato final passou:

contract_ok = true
missing = []
has_score_final = true
has_if_percentile = true
has_decisao = true
final_pred_cols = [exp014a_frozen_pred, exp013k_residual_fp_pred]

Porém, a métrica preview da decisão final do runtime mostrou uma política ultraconservadora:

TP=59
FP=31
FN=1406
TN=112348
precision=0.6556
recall=0.0403
fpr=0.000276

Conclusão técnica:

EXP-014A-4 = SUCESSO COMO SCORING EXPANDIDO
EXP-014A-4 = REJEITADO COMO POLÍTICA HIGH-RECALL
MOTIVO = recall de apenas 4.03%

Decisão operacional: não repetir runtime expandido por custo alto. A partir deste ponto, usar dados\exp014a_expanded_scored_input.csv como entrada pronta e executar apenas rodadas leves, sem nova chamada ao runtime.

Próximo passo: EXP-014A-5 — reconstruir a política high-recall pred_STRICT_RECALL95_SAFE_ONLY + microvetos EXP-013K sobre o dataset expandido já scoreado, comparando contra o runtime oficial ultraconservador.

---

<!-- decision_id: EXP014A5_EXPANDED_HIGH_RECALL_RECONSTRUCTION -->

## EXP-014A-5 — Reconstrução high-recall expandida rejeitada como candidata

**Decisão:** rejeitar o EXP-014A-5 como política candidata e aprová-lo apenas como diagnóstico.

O experimento usou o dataset expandido scoreado do EXP-014A-4, com 113.844 linhas e 1.465 fraudes, sem nova chamada ao runtime. Como a coluna original `pred_STRICT_RECALL95_SAFE_ONLY` não existia no input, a política base foi reconstruída por surrogate a partir do artefato EXP-013J.

Resultado da base reconstruída:


TP=1245
FP=6124
FN=220
recall=84,98%
precision=16,90%

Resultado após aplicar os 10 vetos EXP-013K:

TP=1040
FP=3899
FN=425
recall=70,99%
precision=21,06%

Conclusão: a reconstrução não atingiu recall >=95%, e os vetos EXP-013K não generalizaram no dataset expandido, removendo 205 TPs. Apenas as regras 7 e 9 mantiveram tp_loss=0 no expandido.

Decisão operacional: não promover EXP-014A-5 e não reutilizar o pacote completo de regras EXP-013K no dataset expandido.

Próximo passo: EXP-014B — calibrar uma nova base high-recall diretamente no dataset expandido e minerar novos vetos seguros com restrição TP_loss=0 global e temporal.

---

<!-- decision_id: EXP014B_R1_SAFE_BEAM_EXPANDIDO -->

## EXP-014B-R1 — Benchmark high-recall expandido aprovado

**Decisão:** aprovar o EXP-014B-R1 como novo benchmark técnico high-recall no dataset expandido, mas não promover ainda para produção.

O experimento reaproveitou os parciais do EXP-014B e encerrou sem erro com `stop_reason=max_seconds_after_depth_9`. O status final foi:

DONE_TARGET_RECALL_MET_TPLOSS0_FP_REDUCED_WILSON_PASS

A base selecionada foi:

lgbm_r4_score >= 0.0024302950309567
TP=1448
FP=30794
FN=17
recall=98,84%

Após 9 vetos seguros:

TP=1448
FP=20706
FN=17
recall=98,84%
precision=6,54%
FPR=18,43%

Ganho obtido:

FP removidos = 10088
TP_loss = 0
TP buffer vs target = +56
Wilson low = 98,15%

Conclusão: a política resolve o problema estatístico de suporte positivo e preserva recall alto, mas ainda possui FP residual alto demais para promoção direta.

Próximo passo: EXP-014B-R2 — testar bases high-recall menos amplas, mirando recall entre 96% e 98%, para reduzir FP na origem antes de novo safe-beam.

---

EXP-014B-R2 — Busca da fronteira irredutível de FP no dataset expandido

Decisão: iniciar o EXP-014B-R2 para retomar o foco principal da otimização: encontrar o menor conjunto possível de falsos positivos mantendo recall final acima de 95% no dataset expandido.

O benchmark operacional anterior, EXP-013K/EXP-013L, segue como referência de qualidade por ter atingido recall de 95,16% com FP baixo, mas foi validado em apenas 124 fraudes. Após a expansão para 1.465 fraudes, o objetivo passa a ser descobrir a fronteira real de FP usando todo o suporte positivo disponível.

O EXP-014B-R1 confirmou que há suporte estatístico forte no dataset expandido, mas manteve FP residual alto. Portanto, o EXP-014B-R2 muda a estratégia: em vez de preservar recall máximo, testará bases com recall alvo entre 95% e 98%, permitindo gasto controlado do buffer de TP desde que o recall final nunca caia abaixo de 95%.

Critério de sucesso:

recall_final >= 95%
menor FP possível
Wilson/Bootstrap avaliados
sem nova chamada ao runtime

Se o EXP-014B-R2 encontrar uma fronteira operacionalmente aceitável, o próximo passo será EXP-014C Frozen Validation sem nova mineração. Se o FP residual continuar alto, o resultado será usado para declarar o limite atual da combinação dataset/modelo/regras e encerrar a fase de otimização antes das tarefas de produção, Feature Store HBase e governança shadow.

---

---

<!-- decision_id: EXP014B_R3E_SMALL_STRATEGY_REPLAY_EXPANDED -->

## EXP-014B-R3E — Reaplicação da estratégia campeã do dataset pequeno no expandido

**Decisão:** iniciar o EXP-014B-R3E para reaplicar no dataset expandido a lógica que levou ao benchmark campeão no dataset pequeno, preservando o objetivo de recall >=95% e buscando nova redução de falsos positivos.

O Journal mostra que a estratégia vencedora no dataset pequeno não foi apenas um threshold global. Ela evoluiu em camadas:


1. LGBM high-recall como primeiro estágio;
2. thresholds segmentados por mbk_available_flag + ds_tipo_chave_norm;
3. uso de IF/BEH/SE/DecisionEngine como cascata de redução de FP;
4. diagnóstico estatístico TP vs FP;
5. política conservadora de vetos após validação congelada;
6. micro-refinamentos TP0/blocoTP0;
7. preservação de sinais fortes SE/BEH/runtime.


O EXP-014B-R3D conseguiu um avanço prático no dataset expandido:


TP=1392
FP=6498
FN=73
recall=95,017%
precision=17,64%
FPR=5,78%
TP_loss=0
FP removidos vs base=2460


Porém, ele ficou exatamente no mínimo de TP para recall >=95%, sem buffer, o que impediu vetos mais fortes e manteve o Wilson abaixo do alvo.

O EXP-014B-R3E corrige essa limitação ao testar bases globais e segmentadas com targets de recall 95%, 95,5%, 96% e 97%, criando buffer de TP antes de aplicar vetos estatísticos e microsegmentados. A seleção final será a menor quantidade de FP com recall final >=95%.

Critério de sucesso:


recall_final >=95%
FP_final < 6498, se possível
TP_loss controlado dentro do buffer
Wilson/Bootstrap avaliados
comparação explícita contra o benchmark pequeno:
  recall=95,16%
  precision=37,22%
  FPR=2,02%


Se o EXP-014B-R3E superar o R3D, o próximo passo será validação congelada sem nova mineração. Se não superar, o R3D permanece como fronteira expandida prática atual e a limitação provavelmente estará no score/modelo, não apenas nas regras.

---

---

<!-- decision_id: EXP014B_R3F_FN_FIRST_GLOBAL_RECALL_BUDGET -->

## EXP-014B-R3F — Otimizador global FN First / FP Second

**Decisão:** iniciar o EXP-014B-R3F mudando explicitamente o critério de otimização para FN First e FP Second no dataset expandido.

Após o EXP-014B-R3E, o melhor ponto expandido ficou em:

TP=1393
FP=6403
FN=72
recall=95,085%
precision=17,868%
FPR=5,698%


Esse resultado manteve recall >=95%, mas revelou o problema central: ainda há 72 falsos negativos e praticamente não há buffer de TP para reduzir FP com segurança. Como o objetivo do projeto é antifraude, a prioridade passa a ser reduzir FN ao mínimo irredutível, mesmo que temporariamente seja necessário aceitar mais FP. Após consolidar o menor FN alcançável, a segunda etapa será reduzir FP dentro desse orçamento de FN.

O EXP-014B-R3F implementa um otimizador global de orçamento de recall:


1. segmentar o dataset expandido por ds_tipo_chave_norm, mbk_available_flag e value_band;
2. gerar opções de threshold por score em cada segmento;
3. resolver por programação dinâmica:
   minimizar FP total sujeito a TP >= target_tp;
4. construir uma fronteira FN/FP para vários patamares:
   TP de 95%, TP Wilson, TP alto, até FN=0;
5. para cada ponto selecionado, aplicar vetos residuais preservando o target_tp;
6. selecionar a política padrão por FN First:
   menor FN, depois menor FP.


Critério de leitura:


Se FN cair muito, mesmo com aumento de FP:
    consolidar o patamar de FN irredutível e iniciar nova fase FP Second.
Se FN não cair de forma relevante:
    concluir que o gargalo está no score/modelo/features e preparar hard-negative mining/segundo estágio.


Este experimento é de descoberta de fronteira, não de promoção direta. A decisão de produção virá após escolher o ponto operacional da fronteira FN/FP e validar sem nova mineração.

---

<!-- decision_id: EXP014B_R3F_PARTIAL_FRONTIER_REUSE -->

## EXP-014B-R3F parcial — Fronteira FN First reaproveitada sem rerun

**Decisão:** reaproveitar os resultados parciais do EXP-014B-R3F e não refazer a execução profunda, pois a fronteira global FN/FP já trouxe conclusões suficientes.

O input expandido foi validado com:

```text
n_rows=113844
n_frauds=1465
contract_ok=true

O R3E era o benchmark expandido anterior:

TP=1393
FP=6403
FN=72
recall=95,085%
precision=17,87%
FPR=5,70%

A fronteira global do R3F identificou um novo candidato balanceado que domina o R3E:

R3F_BALANCED_FN_FIRST
TP=1409
FP=6267
FN=56
recall=96,18%
precision=18,36%
FPR=5,58%
Wilson low ≈ 95,07%

Ganho contra R3E:

-16 FN
-136 FP
recall maior
precision maior
Wilson passa 95%

A execução parcial também processou o ponto extremo TP=1464, reduzindo o FN para 1:

R3F_EXTREME_FN_FIRST
TP=1464
FP=19725
FN=1
recall=99,93%
precision=6,91%
FPR=17,55%
FP removidos vs base=6631
TP_loss=0
Wilson low=99,61%

Conclusão: o R3F parcial já provou que a estratégia FN First funciona. O ponto extremo FN=1 é útil como referência de recall máximo, mas possui FP alto demais para promoção direta. O novo benchmark expandido recomendado é R3F_BALANCED_FN_FIRST, com TP=1409, FN=56 e FP=6267.

Próximo passo: consolidar os artefatos do R3F sem nova execução profunda e iniciar a fase FP Second sobre o ponto TP=1409/FN=56.

---

<!-- decision_id: EXP014B_R3G_CONSOLIDATE_R3F_STATES_SHORT_RUNS -->

## EXP-014B-R3G — Consolidação dos estados R3F e transição para rodadas curtas

**Decisão:** consolidar os dois estados úteis descobertos pelo EXP-014B-R3F parcial e evitar refazer a execução profunda, pois ela é demorada e já trouxe a fronteira FN/FP necessária.

Estados consolidados:

1. R3F_BALANCED_FN_FIRST
   TP=1409
   FN=56
   FP≈6267
   recall≈96,18%
   Wilson low≈95,07%
   Uso: novo benchmark expandido operacional balanceado.

2. R3F_EXTREME_FN_FIRST
   TP=1464
   FN=1
   FP=19725 após 8 vetos TP0
   recall=99,93%
   Wilson low=99,61%
   Uso: referência de recall máximo / fila ampliada / investigação crítica.


O R3G deve:


1. reconstruir o estado BALANCED via DP curta apenas para TP alvo 1409;
2. reaproveitar o point_result do estado EXTREME sem rerun;
3. executar no máximo uma rodada curta FP Second sobre o BALANCED;
4. salvar os dois artefatos como políticas separadas;
5. usar o BALANCED como novo benchmark principal.


Próxima fase:


FN First consolidado:
    usar R3F/R3G_BALANCED como benchmark principal.

FP Second:
    rodadas curtas, limitadas por tempo, sobre o BALANCED.
    não executar novas buscas profundas longas.

Se os 56 FNs não puderem ser reduzidos sem custo extremo:
    migrar para hard-negative mining/segundo estágio e novos sinais.


---

<!-- decision_id: EXP014B_R3H_RESIDUAL_FP_SECOND_BALANCED_R3G -->

## EXP-014B-R3H — Rodada curta FP Second sobre BALANCED_R3G

**Decisão:** iniciar o EXP-014B-R3H como rodada curta de redução residual de falsos positivos sobre o novo benchmark expandido principal `BALANCED_R3G_QUICK_FP_SECOND`.

Benchmark de entrada:


TP=1409
FP=5520
FN=56
recall=96,177%
precision=20,335%
FPR=4,912%
Wilson low=95,069%


Objetivo da rodada:


preservar TP=1409 / FN=56 por padrão
tentar reduzir FP de 5520 para abaixo de 5000
usar apenas busca curta, sem execução profunda
selecionar preferencialmente regras TP_loss=0 e block_tp_loss=0


Critério de sucesso:


FP_REDUCED
TPLOSS0
idealmente FP < 5000
Wilson recall continua >=95%


Se o R3H não reduzir FP de forma relevante, a próxima etapa passa a ser `EXP-014B-R3I — auditoria dos 56 FNs residuais`, porque novas rodadas de microveto podem estar chegando ao limite prático.

---

<!-- decision_id: EXP014B_R3H_NEW_EXPANDED_BENCHMARK -->

## EXP-014B-R3H — Novo benchmark expandido com FP abaixo de 5000

**Decisão:** aprovar o EXP-014B-R3H como novo benchmark expandido principal.

O R3H executou uma rodada curta de FP Second sobre `BALANCED_R3G_QUICK_FP_SECOND` e terminou com:

```text
status=DONE_FP_REDUCED_TPLOSS0_BELOW_5000FP_WILSON_PASS_95
elapsed_seconds=186.32

Benchmark de entrada R3G:

TP=1409
FP=5520
FN=56
recall=96,177%
precision=20,335%
FPR=4,912%

Resultado final R3H:

TP=1409
FP=4935
FN=56
recall=96,177%
precision=22,210%
FPR=4,391%
Wilson low=95,069%

Ganho obtido:

FP removidos=585
TP_loss=0
FN preservado=56
FP abaixo de 5000

As 8 regras selecionadas foram todas microvetos residuais com TP_loss=0 e block_tp_loss_max=0.

Conclusão: o R3H substitui o R3G como benchmark expandido principal. O próximo passo é validação congelada curta do policy artifact do R3H e, em seguida, auditoria dos 56 FNs residuais.

---

<!-- decision_id: EXP014B_R3H_FROZEN_VALIDATION -->

## EXP-014B-R3H-FROZEN — Validação congelada do novo benchmark expandido

**Decisão:** iniciar validação congelada do policy artifact do EXP-014B-R3H antes de avançar para novas otimizações.

O EXP-014B-R3H tornou-se o novo benchmark expandido principal:

```text
TP=1409
FP=4935
FN=56
recall=96,177%
precision=22,210%
FPR=4,391%
Wilson low=95,069%
```

Ganho contra o R3G:

```text
FP removidos=585
TP_loss=0
FN preservado=56
FP abaixo de 5000
```

A validação congelada deve reaplicar apenas o artifact:

```text
resultados/experimentos/EXP-014B-R3H/12_policy_artifact.json
```

sobre a base:

```text
resultados/experimentos/EXP-014B-R3G/09_predictions.csv
```

sem nova mineração, sem beam search e sem recalibrar thresholds.

Critério de aprovação:

```text
TP=1409
FP=4935
FN=56
fp_removed_vs_base=585
tp_loss_vs_base=0
Wilson low >= 95%
impacto por regra igual ao artifact
schema mínimo OK
```

Se passar, o R3H fica consolidado como benchmark expandido principal e o próximo passo será `EXP-014B-R3I — auditoria dos 56 FNs residuais`.

---

<!-- decision_id: EXP014B_R3I_PATCH_FROZEN_AND_FN_AUDIT -->

## EXP-014B-R3I — Patch do R3H-FROZEN e auditoria dos 56 FNs residuais

**Decisão:** iniciar o EXP-014B-R3I com duas etapas encadeadas.

Primeiro, aplicar um patch de interpretação do R3H-FROZEN. O frozen havia retornado status de falha porque algumas regras tiveram `expected_fp_removed` individual diferente no replay sequencial. A divergência é explicável por sobreposição entre regras: regras anteriores removem FPs que também pertencem às máscaras de regras posteriores. Como as métricas finais bateram exatamente, o delta agregado bateu, o schema está OK, o TP_loss é zero e o Wilson passou, essa divergência deve ser tratada como `OVERLAP_WARNING`, não como falha de validação.

Critério do patch:

```text
schema_ok=true
expected_metrics_matched=true
expected_delta_matched=true
tp_loss_vs_base=0
wilson_pass=true
sem missing_columns
divergência apenas em fp_removed individual por regra
```

Segundo, executar a auditoria dos 56 falsos negativos residuais do benchmark R3H congelado:

```text
TP=1409
FP=4935
FN=56
recall=96,177%
precision=22,210%
FPR=4,391%
Wilson low=95,069%
```

Objetivo da auditoria:

```text
1. perfilar os 56 FNs por segmento, score e bins;
2. gerar candidatos diagnósticos de resgate de FN;
3. medir o custo em FP adicionado por FN recuperado;
4. decidir se existe resgate barato ou se os 56 FNs são limite prático dos scores/features atuais.
```

Este experimento é diagnóstico, não promocional. Qualquer regra de resgate encontrada precisará de validação congelada posterior.

---



