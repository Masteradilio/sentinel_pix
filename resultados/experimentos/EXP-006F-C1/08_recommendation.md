# EXP-006F — Quick-E2E C1 Near-Threshold

- Status: `APROVADO_PARA_PATCH_PERMANENTE`

## Métricas

| Seed | Config | TP | FP | FN | Precision | Recall | F1 | FPR |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | `BASELINE` | 346 | 14 | 9 | 96.1111% | 97.4648% | 0.9678 | 0.2480% |
| 42 | `C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER` | 347 | 14 | 8 | 96.1219% | 97.7465% | 0.9693 | 0.2480% |
| 123 | `BASELINE` | 346 | 12 | 9 | 96.6480% | 97.4648% | 0.9705 | 0.2126% |
| 123 | `C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER` | 347 | 12 | 8 | 96.6574% | 97.7465% | 0.9720 | 0.2126% |

## Delta por seed

| Seed | FNs recuperados | FPs adicionados | TPs perdidos | FPs removidos | Rule hits |
|---:|---:|---:|---:|---:|---:|
| 123 | 1 | 0 | 0 | 0 | 1 |
| 42 | 1 | 0 | 0 | 0 | 1 |

## Decisão

Gerar patch permanente no DecisionEngine/scoring_config para C1, mantendo flag configurável e desligável.

C1 passou no quick-E2E/cached-runtime e pode ser transformado em patch permanente configurável.