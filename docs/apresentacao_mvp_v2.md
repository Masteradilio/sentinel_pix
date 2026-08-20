# Apresentacao do MVP v2: Motor PIX com baseline oficial R5B22

## 1. Introducao e geracao de valor

O MVP v2 consolida o motor antifraude PIX na versao operacional R5B22, usando as novas bases MAF e o contrato congelado R5B16/R5B18 como professor. O objetivo mudou de "zero fraude fora do bloqueio a qualquer custo" para uma politica operacional mais equilibrada:

- APROVAR deve conter o minimo possivel de fraude, com teto formal de 5 fraudes.
- CONFIRMAR deve absorver casos duvidosos, com teto formal de 10 fraudes.
- BLOQUEAR deve maximizar precisao, concentrando a maior parte das fraudes e reduzindo bloqueios indevidos.
- O FPR global deve permanecer abaixo de 1%.

Resultado oficial R5B22:

| Indicador | Resultado |
|---|---:|
| Politica oficial | `R5B22_OFFICIAL_CONSTRAINED_BASELINE` |
| Base de validacao | 113.844 transacoes |
| Fraudes totais | 1.465 |
| Normais totais | 112.379 |
| FPR global | 0,957474% |
| Fraudes em APROVAR | 2 |
| Fraudes em CONFIRMAR | 10 |
| Fraudes em BLOQUEAR | 1.453 |

## 2. Como o modelo opera agora

O fluxo de decisao passa a ser:

1. O `PipelineOrquestrador` calcula as features e executa o motor de decisao.
2. O contrato congelado R5B16 aplica a decisao-base de referencia (`r4g_fast_frozen_decisao_recommended`).
3. A politica R5B14 aplica as restricoes operacionais de baixo FN.
4. A politica oficial R5B22 aplica democoes controladas para reduzir normais em BLOQUEAR sem ultrapassar os tetos de fraude em APROVAR e CONFIRMAR.
5. O resultado final continua sendo uma das tres decisoes: APROVAR, CONFIRMAR ou BLOQUEAR.

Configuracoes oficiais ativadas:

| Configuracao | Valor |
|---|---:|
| `r5b14_operational_zero_fn_enabled` | `true` |
| `r5b16_frozen_contract_enabled` | `true` |
| `r5b22_official_baseline_enabled` | `true` |
| `official_baseline_policy` | `R5B22_OFFICIAL_CONSTRAINED_BASELINE` |

Artefatos oficiais:

```text
backend/artefatos/r5b22_official_baseline_policy.json
backend/artefatos/r5b22_official_baseline_summary.json
backend/artefatos/model_lgbm_distilled_r5b22_intervention.joblib
backend/artefatos/model_lgbm_distilled_r5b22_block.joblib
backend/artefatos/model_lgbm_distilled_r5b22_metadata.json
```

## 3. Metricas globais do baseline R5B22

Metricas globais considerando intervencao como `CONFIRMAR` ou `BLOQUEAR`:

| Metrica | Valor |
|---|---:|
| TP | 1.463 |
| FP | 1.076 |
| FN | 2 |
| TN | 111.303 |
| Precision | 57,621111% |
| Recall | 99,863481% |
| F1 | 0,73076923 |
| FPR | 0,957474% |

Metricas especificas de BLOQUEAR:

| Metrica | Valor |
|---|---:|
| TP | 1.453 |
| FP | 760 |
| FN fora de BLOQUEAR | 12 |
| TN | 111.619 |
| Precision | 65,657479% |
| Recall | 99,180887% |
| F1 | 0,79010332 |
| FPR | 0,676283% |

Comparacao contra R5B16/R5B18:

| Indicador | Antes | R5B22 | Delta |
|---|---:|---:|---:|
| Normais em BLOQUEAR | 835 | 760 | -75 |
| Precision BLOQUEAR | 63,695652% | 65,657479% | +1,961827 p.p. |
| F1 BLOQUEAR | 0,77822045 | 0,79010332 | +0,01188287 |
| FPR BLOQUEAR | 0,743021% | 0,676283% | -0,066738 p.p. |
| Fraudes em APROVAR | 0 | 2 | +2 |
| Fraudes em CONFIRMAR | 0 | 10 | +10 |

## 4. Distribuicao operacional por decisao

| Decisao | Transacoes | Fraudes | Normais | Taxa de fraude |
|---|---:|---:|---:|---:|
| APROVAR | 111.305 | 2 | 111.303 | 0,001797% |
| CONFIRMAR | 326 | 10 | 316 | 3,067485% |
| BLOQUEAR | 2.213 | 1.453 | 760 | 65,657479% |

Leitura operacional:

- APROVAR fica praticamente limpo: 2 fraudes em 111.305 aprovacoes.
- CONFIRMAR fica reservado para atrito controlado: 326 transacoes, com 10 fraudes dentro do teto aprovado.
- BLOQUEAR concentra 99,18% das fraudes conhecidas e reduz normais bloqueadas para 760.

## 5. Metricas do LGBM aluno do contrato R5B16

O LGBM aluno e uma distilacao do contrato operacional R5B16/R5B18. Ele nao deve ser apresentado como LGBM puro sobre features brutas, porque usa sinais do professor, incluindo `r4g_fast_frozen_decisao_recommended`, `r5b14_rule_applied` e `r5b14_layer_applied`.

Modelo aluno de intervencao (`APROVAR` vs `CONFIRMAR/BLOQUEAR`) no conjunto full:

| Metrica | Valor |
|---|---:|
| TP | 1.463 |
| FP | 1.124 |
| FN | 2 |
| TN | 111.255 |
| Precision | 56,551991% |
| Recall | 99,863481% |
| F1 | 0,72211254 |
| FPR | 1,000187% |

Razoes contra o baseline R5B16:

| Metrica | Ratio |
|---|---:|
| Precision | 99,902084% |
| Recall | 99,863481% |
| F1 | 99,888128% |

Modelo aluno de BLOQUEAR no conjunto full:

| Metrica | Valor |
|---|---:|
| TP | 1.465 |
| FP | 835 |
| FN | 0 |
| TN | 111.544 |
| Precision | 63,695652% |
| Recall | 100,000000% |
| F1 | 0,77822045 |
| FPR | 0,743021% |

Razoes contra o baseline R5B16 para BLOQUEAR:

| Metrica | Ratio |
|---|---:|
| Precision | 100,000000% |
| Recall | 100,000000% |
| F1 | 100,000000% |

## 6. Regras ativas de restricao APROVAR/CONFIRMAR/BLOQUEAR

### 6.1 Contrato congelado R5B16

O contrato R5B16 usa `r4g_fast_frozen_decisao_recommended` como decisao-base congelada. A partir dele, o orquestrador aplica overlays para reproduzir o baseline global oficial.

### 6.2 Politica R5B14

Regras que promovem `CONFIRMAR -> BLOQUEAR`:

| Regra | Condicao |
|---|---|
| `R5B14_CTB_01_LGBM_RAW_HIGH` | `lgbm_raw >= 0.10711783` |
| `R5B14_CTB_02_SCORE_2_3_LGBM_R4_HIGH` | `score_bin == score_2_3` e `lgbm_r4_score >= 0.475472966916` |
| `R5B14_CTB_03_SCORE_2_3_LGBM_R4_MED` | `score_bin == score_2_3` e `lgbm_r4_score >= 0.318070929491` |
| `R5B14_CTB_04_DOC_PHONE_HIGH_PAYER_COUNT` | `ds_tipo_chave_norm == DOCUMENTO_TELEFONE` e `qtd_pix_pagador_180d >= 207` |
| `R5B14_CTB_05_OUTROS_RATIO_MAX_HIGH` | `ds_tipo_chave_norm == OUTROS` e `ratio_valor_maximo_pagador_180d >= 4.9674631165863596` |

Regras que promovem `APROVAR -> BLOQUEAR`:

| Regra | Condicao |
|---|---|
| `R5B14_ATB_01_DOC_PHONE_MORNING_SCORE_HIGH` | `ds_tipo_chave_norm == DOCUMENTO_TELEFONE`, `periodo_dia == manha`, `score_bin == score_GE_10` e `lgbm_bin == lgbm_GE_0.1` |
| `R5B14_ATB_02_NIGHT_SCORE_1_2_RATIO_HIGH` | `periodo_dia == noite`, `score_bin == score_1_2`, `lgbm_bin == lgbm_GE_0.1` e `ratio_bin == ratio_GE_5` |

Regra de compensacao que demove `CONFIRMAR -> APROVAR`:

| Regra | Condicao |
|---|---|
| `R5B14_CTA_01_LOW_LGBM_RAW_COMPENSATION` | `lgbm_raw <= 0.00001966` |

### 6.3 Politica oficial R5B22

Regras finais de democao controlada aplicadas somente para reduzir atrito em BLOQUEAR respeitando os tetos de fraude:

| Regra | Condicao | Acao final | Linhas | Fraudes | Normais |
|---|---|---:|---:|---:|---:|
| `DEMOTE_LAYER_APPROVE_TO_BLOCK_TO_APROVAR` | `r5b14_layer_applied == APPROVE_TO_BLOCK` | APROVAR | 49 | 2 | 47 |
| `DEMOTE_LAYER_CONFIRM_TO_BLOCK_TO_CONFIRMAR` | `r5b14_layer_applied == CONFIRM_TO_BLOCK` | CONFIRMAR | 27 | 5 | 22 |
| `DEMOTE_CAT2_ds_tipo_chave_norm_OUTROS__lgbm_bin_lgbm_0.05_0.1` | `ds_tipo_chave_norm == OUTROS` e `lgbm_bin == lgbm_0.05_0.1` | CONFIRMAR | 11 | 4 | 7 |
| `DEMOTE_CAT2_value_band_E_5000_10000__lgbm_bin_lgbm_0.05_0.1` | `value_band == E_5000_10000` e `lgbm_bin == lgbm_0.05_0.1` | CONFIRMAR | 8 | 2 | 6 |

Incrementalmente, a politica R5B22 demoveu 87 transacoes de BLOQUEAR, sendo 75 normais e 12 fraudes. O ganho operacional foi reduzir bloqueios indevidos sem violar os limites aprovados:

| Gate | Limite | Resultado | Status |
|---|---:|---:|---|
| Fraudes em APROVAR | <= 5 | 2 | PASS |
| Fraudes em CONFIRMAR | <= 10 | 10 | PASS |
| FPR global | < 1% | 0,957474% | PASS |
| Precisao de BLOQUEAR | melhorar vs R5B16 | 65,657479% | PASS |

## 7. Leitura executiva

O baseline R5B22 e o novo padrao global do modelo porque melhora a precisao de BLOQUEAR, reduz normais bloqueadas e mantem a exposicao de fraude nas decisoes de menor atrito dentro dos tetos definidos. A apresentacao correta do modelo agora e:

```text
Motor antifraude PIX hibrido com LGBM aluno do contrato R5B16,
contrato congelado R4G/R5B16, overlays R5B14 e politica oficial R5B22
para balancear captura de fraude, FPR global e precisao de BLOQUEAR.
```

