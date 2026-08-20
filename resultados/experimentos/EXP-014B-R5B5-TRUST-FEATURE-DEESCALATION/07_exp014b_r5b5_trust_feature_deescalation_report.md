# EXP-014B-R5B5-TRUST-FEATURE-DEESCALATION — Trust Feature De-escalation

## Resultado executivo
- Status: `PASS_R5B5_TRUST_DEESCALATION_FOUND`
- Candidatos avaliados: `1752`
- Regras selecionadas: `5`
- Normais adicionais movidos de BLOQUEAR para CONFIRMAR: `358`
- Fraudes movidas de BLOQUEAR para CONFIRMAR: `0`
- Normais restantes em BLOQUEAR: `2824`
- Fraudes restantes em APROVAR: `682`

## Métricas finais de BLOQUEAR
```json
{
  "tp": 279,
  "fp": 2824,
  "fn": 1186,
  "tn": 109555,
  "precision": 0.08991299,
  "recall": 0.19044369,
  "f1": 0.12215412,
  "fpr": 0.02512925
}
```

## Regras selecionadas
|   selection_step | candidate_id                                                                                    |   incremental_normals |   incremental_frauds |   non_train_normals |   holdout_normals |   validation_normals |   month_normal_support |
|-----------------:|:------------------------------------------------------------------------------------------------|----------------------:|---------------------:|--------------------:|------------------:|---------------------:|-----------------------:|
|                1 | trust_score__relationship_strength_score__>=100                                                 |                   124 |                    0 |                  73 |                38 |                   35 |                      6 |
|                2 | trust_cat2__trust_bucket=trust_60_80__receiver_rep_bucket=rep_80_100                            |                    82 |                    0 |                  14 |                 7 |                    7 |                      6 |
|                3 | trust_cat2__receiver_rep_bucket=rep_60_80__relationship_bucket=rel_80_100                       |                    44 |                    0 |                  13 |                 5 |                    8 |                      6 |
|                4 | trust_cat3__receiver_rep_bucket=rep_40_60__valor_rec_bin=val_rec_lt_5k__value_band=E_5000_10000 |                    53 |                    0 |                  12 |                 6 |                    6 |                      6 |
|                5 | trust_cat3__receiver_rep_bucket=rep_60_80__valor_rec_bin=val_rec_lt_5k__value_band=F_10000_PLUS |                    55 |                    0 |                  10 |                 3 |                    7 |                      6 |

## Features novas criadas
- `payer_history_strength_score`
- `receiver_reputation_score`
- `relationship_strength_score`
- `receiver_novelty_risk_score`
- `transaction_normality_score`
- `payer_receiver_trust_score`
