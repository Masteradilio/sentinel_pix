# EXP-014B-R5B8-BROAD-RESIDUAL-RULE-MINING — Mineração ampla de regras residuais

## Resultado executivo
- Status: `PASS_R5B8_BROAD_RULES_FOUND`
- Candidatos avaliados: `2158`
- Regras selecionadas: `3`
- Normais adicionais movidos de BLOQUEAR para CONFIRMAR: `469`
- Fraudes movidas de BLOQUEAR para CONFIRMAR: `0`
- Normais restantes em BLOQUEAR: `2355`
- Fraudes restantes em BLOQUEAR: `279`
- Fraudes restantes em APROVAR: `682`

## Métricas finais de BLOQUEAR
```json
{
  "tp": 279,
  "fp": 2355,
  "fn": 1186,
  "tn": 110024,
  "precision": 0.10592255,
  "recall": 0.19044369,
  "f1": 0.13613076,
  "fpr": 0.02095587
}
```

## Regras selecionadas
|   selection_step | candidate_id                                                                                                           |   incremental_normals |   incremental_frauds |   non_train_normals |   holdout_normals |   validation_normals |   month_normal_support |
|-----------------:|:-----------------------------------------------------------------------------------------------------------------------|----------------------:|---------------------:|--------------------:|------------------:|---------------------:|-----------------------:|
|                1 | num__dias_desde_primeiro_envio_recebedor__>=35                                                                         |                   357 |                    0 |                 101 |                24 |                   77 |                      5 |
|                2 | cat2__receiver_reputation_score__qbin=receiver_reputation_score_q2__qtd_pix_pagador_180d__qbin=qtd_pix_pagador_180d_q5 |                    58 |                    0 |                  19 |                14 |                    5 |                      6 |
|                3 | cat2__valor_rec_bin=val_rec_lt_5k__valor_total_pagador_180d__qbin=valor_total_pagador_180d_q5                          |                    54 |                    0 |                  15 |                 3 |                   12 |                      6 |

## Decisão técnica
Este experimento amplia a mineração do R5B5, mas preserva os mesmos critérios
conservadores: zero fraude demovida, suporte fora de treino e suporte em pelo
menos dois meses. Regras aprovadas aqui são candidatas para revisão manual antes
de qualquer integração de política.
