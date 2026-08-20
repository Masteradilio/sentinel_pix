# EXP-014B-R5B6-TRUST-LGBM-SHADOW — LGBM shadow com trust features

## Resultado executivo
- Status: `DONE_R5B6_TRUST_LGBM_SHADOW`
- Features totais: `65`
- Melhor iteração: `16`
- Threshold F1 validação: `0.1550`

## Holdout shadow
```json
{
  "threshold": 0.155,
  "tp": 64,
  "fp": 68,
  "fn": 60,
  "tn": 17180,
  "roc_auc": 0.969744,
  "average_precision": 0.475929,
  "precision": 0.484848,
  "recall": 0.516129,
  "f1": 0.5,
  "fpr": 0.003942
}
```

## Delta vs LGBM canônico atual
```json
{
  "roc_auc": -0.002426,
  "average_precision": -0.00617,
  "precision": 0.09825,
  "recall": -0.08871,
  "f1": 0.028302,
  "fpr": -0.002957,
  "tp": -11.0,
  "fp": -51.0,
  "fn": 11.0
}
```

## Importância das features de trust
| feature                      |   importance_gain | is_trust_feature   |
|:-----------------------------|------------------:|:-------------------|
| payer_receiver_trust_score   |       1.05139e+06 | True               |
| trust_bucket                 |  350059           | True               |
| receiver_reputation_score    |   38605.6         | True               |
| transaction_normality_score  |   37551.4         | True               |
| payer_history_strength_score |   13189.7         | True               |
| relationship_strength_score  |   10056.6         | True               |
| novelty_bucket               |       0           | True               |
| receiver_rep_bucket          |       0           | True               |
| receiver_novelty_risk_score  |       0           | True               |
| relationship_bucket          |       0           | True               |

## Decisão técnica
Este experimento é shadow e não substitui artefatos produtivos. Se AP/F1/recall
melhorarem sem violar FPR<=1%, as features de trust devem ser consideradas para
o próximo contrato canônico do LGBM.
