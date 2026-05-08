# EXP-005B-E2E — DecisionEngine real

- Vencedor: `CAND_007_RECALL`
- scoring_config válido: `True`

## Resultados

| Config | Seed | TP | FP | FN | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `BASELINE` | 42 | 346 | 14 | 9 | 96.1111% | 97.4648% | 0.9678 | 0.2480% |
| `BASELINE` | 123 | 346 | 12 | 9 | 96.6480% | 97.4648% | 0.9705 | 0.2126% |
| `CAND_007_RECALL` | 42 | 346 | 25 | 9 | 93.2615% | 97.4648% | 0.9532 | 0.4429% |
| `CAND_007_RECALL` | 123 | 346 | 19 | 9 | 94.7945% | 97.4648% | 0.9611 | 0.3366% |
| `CAND_015_BALANCED` | 42 | 346 | 25 | 9 | 93.2615% | 97.4648% | 0.9532 | 0.4429% |
| `CAND_015_BALANCED` | 123 | 346 | 19 | 9 | 94.7945% | 97.4648% | 0.9611 | 0.3366% |
| `CAND_020_MAIN` | 42 | 346 | 25 | 9 | 93.2615% | 97.4648% | 0.9532 | 0.4429% |
| `CAND_020_MAIN` | 123 | 346 | 19 | 9 | 94.7945% | 97.4648% | 0.9611 | 0.3366% |
| `CAND_030_CONSERVATIVE` | 42 | 346 | 25 | 9 | 93.2615% | 97.4648% | 0.9532 | 0.4429% |
| `CAND_030_CONSERVATIVE` | 123 | 346 | 19 | 9 | 94.7945% | 97.4648% | 0.9611 | 0.3366% |

## Interpretação

Este experimento usa o `PipelineOrquestrador` e `DecisionEngine` reais com swap temporário do LGBM v6.2.
Nada é promovido automaticamente. A promoção depende da análise dos deltas, FNs recuperados, FPs adicionados e estabilidade entre seeds.
