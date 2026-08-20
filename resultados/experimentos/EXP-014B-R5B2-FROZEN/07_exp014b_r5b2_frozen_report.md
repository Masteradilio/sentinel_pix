# EXP-014B-R5B2-FROZEN — Validação congelada

## Resultado executivo
- Status: `PASS_R5B2_FROZEN_REPLAYED`
- All pass: `True`
- Regras reaplicadas: `60`
- Mismatches vs calibration: `0`
- Normais movidos de BLOQUEAR para CONFIRMAR: `11067`
- Fraudes movidas de BLOQUEAR para CONFIRMAR: `0`

## Métricas de BLOQUEAR
```json
{
  "tp": 279,
  "fp": 3153,
  "fn": 1186,
  "tn": 109226,
  "precision": 0.08129371,
  "recall": 0.19044369,
  "f1": 0.11394731,
  "fpr": 0.02805684
}
```

## Métricas de intervenção total
```json
{
  "tp": 783,
  "fp": 15584,
  "fn": 682,
  "tn": 96795,
  "precision": 0.04784017,
  "recall": 0.53447099,
  "f1": 0.08781965,
  "fpr": 0.1386736
}
```

## Decisões finais
| r5b2_frozen_decisao   |   n_rows |   n_frauds |   n_normals |
|:----------------------|---------:|-----------:|------------:|
| APROVAR               |    97477 |        682 |       96795 |
| BLOQUEAR              |     3432 |        279 |        3153 |
| CONFIRMAR             |    12935 |        504 |       12431 |

## Estabilidade por split/mês
| temporal_split   | event_month   |   n_rows |   block_to_confirm_n |   block_to_confirm_normals |   block_to_confirm_frauds |   remaining_block_normals |   remaining_block_frauds |
|:-----------------|:--------------|---------:|---------------------:|---------------------------:|--------------------------:|--------------------------:|-------------------------:|
| HOLDOUT          | 2026-04       |     2357 |                  259 |                        259 |                         0 |                        55 |                       13 |
| HOLDOUT          | 2026-05       |    15015 |                 1432 |                       1432 |                         0 |                       171 |                        7 |
| TRAIN            | 2025-11       |     2090 |                   31 |                         31 |                         0 |                        20 |                        3 |
| TRAIN            | 2025-12       |    23173 |                 2439 |                       2439 |                         0 |                      1140 |                       68 |
| TRAIN            | 2026-01       |    18829 |                 1799 |                       1799 |                         0 |                       547 |                       53 |
| TRAIN            | 2026-02       |    17291 |                 1727 |                       1727 |                         0 |                       456 |                       48 |
| TRAIN            | 2026-03       |    17297 |                 1800 |                       1800 |                         0 |                       406 |                       44 |
| VALIDATION       | 2026-03       |     1971 |                  152 |                        152 |                         0 |                        44 |                        3 |
| VALIDATION       | 2026-04       |    15821 |                 1428 |                       1428 |                         0 |                       314 |                       40 |

## Interpretação
A política congelada reproduz a redução de severidade `BLOQUEAR -> CONFIRMAR` sem
rebaixar fraude conhecida de BLOQUEAR. Ela não resolve o recall total, pois ainda
restam `682` fraudes em APROVAR. O próximo
experimento deve atacar resgate de APROVAR e/ou melhorar score/features do modelo base.
