# EXP-007A — Meta-Learner Shadow

Gerado em: `2026-05-09T12:25:49`

- Status: `SEM_CANDIDATO_SEGURO`

## Baseline pós-C1

| Seed/Conjunto | TP | FP | FN | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `POST_C1_seed_42` | 347 | 14 | 8 | 96.1219% | 97.7465% | 0.9693 | 0.2480% |
| `POST_C1_seed_123` | 347 | 12 | 8 | 96.6574% | 97.7465% | 0.9720 | 0.2126% |
| `POST_C1_UNIQUE_UNION` | 347 | 25 | 8 | 93.2796% | 97.7465% | 0.9546 | 0.2278% |

## Qualidade shadow dos meta-learners

| Modelo | OOF AUC | OOF AP | Linhas | Fraudes | Features |
|---|---:|---:|---:|---:|---:|
| `LOGREG_BALANCED` | 0.997334 | 0.983481 | 11330 | 355 | 31 |
| `RF_SHALLOW_BALANCED` | 0.997655 | 0.984908 | 11330 | 355 | 31 |
| `EXTRATREES_SHALLOW_BALANCED` | 0.995809 | 0.985948 | 11330 | 355 | 31 |
| `ENSEMBLE_MEAN` | 0.996818 | 0.987762 | 11330 | 355 | 31 |

## Melhor candidato do sweep

Nenhum candidato seguro encontrado.

## Residual FNs pós-C1

- Quantidade residual no conjunto único: `8`

| Tx | Valor | Rel. meses | LGBM | IF | SE | BEH | Score | Shadow ensemble |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `E0000020820260103172401005155525` | 142.00 | 216 | 0.04782 | 0.47706 | 0.00 | 0.00 | 53.20 | 0.83469 |
| `E0000020820260316231246610306525` | 300.00 | 243 | 0.02850 | 0.95701 | 0.00 | 0.00 | 46.39 | 0.80486 |
| `E0000020820260227174607667379525` | 498.96 | 317 | 0.02834 | 0.76376 | 0.00 | 0.00 | 44.88 | 0.77341 |
| `E0000020820260107140624591248525` | 390.00 | 58 | 0.00371 | 0.36004 | 0.00 | 0.00 | 17.44 | 0.40182 |
| `E0000020820260123162214303665525` | 50.00 | 456 | 0.02232 | 0.76187 | 0.00 | 0.00 | 41.87 | 0.39033 |
| `E0000020820260214233339434522525` | 57.88 | 163 | 0.00418 | 0.37974 | 0.00 | 0.00 | 21.34 | 0.28424 |
| `E0000020820260224213046993254525` | 188.82 | 41 | 0.00003 | 0.27674 | 0.00 | 0.00 | 1.75 | 0.00507 |
| `E0000020820260213145228155991525` | 29.90 | 267 | 0.00005 | 0.70873 | 0.00 | 0.00 | 4.06 | 0.00306 |

## Decisão

Não rodar EXP-007B. Os sinais atuais não geraram overlay seguro. Considerar novas fontes de dados ou encerrar FASE 2 como próxima do limite atual.

O meta-learner shadow não encontrou um overlay seguro para recuperar FN sem custo em FP.
Isso sugere que os FNs remanescentes podem estar próximos do limite dos sinais atuais.
Próximo caminho: novas fontes de dados ou análise manual dos FNs residuais.