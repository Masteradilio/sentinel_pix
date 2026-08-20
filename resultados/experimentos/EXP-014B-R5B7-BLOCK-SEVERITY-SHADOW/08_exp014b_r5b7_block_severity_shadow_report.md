# EXP-014B-R5B7-BLOCK-SEVERITY-SHADOW — Modelo shadow de severidade BLOQUEAR

## Resultado executivo
- Status: `NO_R5B7_MATERIAL_SAFE_THRESHOLD_FOUND`
- Features totais: `109`
- Melhor iteração: `16`
- Threshold selecionado em validação: `0.0645117720286907`
- Normais movidos de BLOQUEAR para CONFIRMAR: `4`
- Fraudes movidas de BLOQUEAR para CONFIRMAR: `0`
- Fraudes não-treino movidas: `0`
- Normais restantes em BLOQUEAR: `2820`
- Fraudes restantes em BLOQUEAR: `279`
- Fraudes restantes em APROVAR: `682`

## Métricas do score no residual BLOQUEAR
```json
{
  "validation_auc": 0.791676,
  "holdout_auc": 0.720858,
  "validation_average_precision": 0.427079,
  "holdout_average_precision": 0.350632
}
```

## Métricas finais de BLOQUEAR
```json
{
  "tp": 279,
  "fp": 2820,
  "fn": 1186,
  "tn": 109559,
  "precision": 0.09002904,
  "recall": 0.19044369,
  "f1": 0.12226117,
  "fpr": 0.02509366
}
```

## Suporte por split
| temporal_split   |   n_rows |   demoted_normals |   demoted_frauds |   remaining_block_normals |   remaining_block_frauds |
|:-----------------|---------:|------------------:|-----------------:|--------------------------:|-------------------------:|
| HOLDOUT          |    17372 |                 0 |                0 |                       169 |                       20 |
| TRAIN            |    78680 |                 3 |                0 |                      2355 |                      216 |
| VALIDATION       |    17792 |                 1 |                0 |                       296 |                       43 |

## Top features
| feature                             |   importance_gain | is_trust_feature   |
|:------------------------------------|------------------:|:-------------------|
| lgbm_raw                            |         13881.6   | False              |
| tempo_processamento_host_ms         |          2554.6   | False              |
| soma_recebedores_distintos_dia_180d |          1914.86  | False              |
| dias_desde_primeiro_envio_recebedor |          1848.99  | False              |
| lgbm_mapped                         |          1535.18  | False              |
| valor_total_recebido_30d            |          1505.43  | False              |
| mbk_completeness_score              |          1414.69  | False              |
| hour                                |          1241.13  | False              |
| topaz_risk_score                    |          1010.51  | False              |
| transaction_normality_score         |           907.834 | True               |
| device_name                         |           905.29  | False              |
| receiver_reputation_score           |           891.061 | True               |
| valor_total_pagador_90d             |           864.623 | False              |
| _worker_id                          |           670.515 | False              |
| valor_total_pagador_180d            |           641.306 | False              |
| vl_mediana_pix_trimestre            |           622.518 | False              |
| qtd_pix_pagador_180d                |           580.37  | False              |
| valor_maximo_pix_pagador_180d       |           571.535 | False              |
| ratio_valor_media_pagador_90d       |           568.284 | False              |
| idx                                 |           565.182 | False              |

## Features de trust no modelo
| feature                      |   importance_gain | is_trust_feature   |
|:-----------------------------|------------------:|:-------------------|
| transaction_normality_score  |          907.834  | True               |
| receiver_reputation_score    |          891.061  | True               |
| payer_history_strength_score |          307.216  | True               |
| payer_receiver_trust_score   |           28.7503 | True               |
| receiver_novelty_risk_score  |            0      | True               |
| relationship_strength_score  |            0      | True               |
| trust_bucket                 |            0      | True               |
| receiver_rep_bucket          |            0      | True               |
| relationship_bucket          |            0      | True               |
| novelty_bucket               |            0      | True               |

## Decisão técnica
Este é um experimento shadow e não altera artefatos produtivos. Um resultado
promocionável exige zero fraude demovida fora de treino e ganho material de
redução de falso bloqueio sobre o R5B5.
