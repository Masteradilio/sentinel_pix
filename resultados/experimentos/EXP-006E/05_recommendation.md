# EXP-006E — Residual FN Counterfactual Designer

Gerado em: `2026-05-09T11:11:30`

- Status: `APROVADO_PARA_QUICK_E2E_PATCH_TEMPORARIO`

## Regra candidata

`C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER`

Exceção cirúrgica para transação APROVAR near-threshold, primeiro recebedor, relacionamento curto, baixo/médio valor e LGBM em zona cinza.

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

Criar EXP-006F quick-E2E com patch temporário no DecisionEngine implementando C1. Rodar baseline + C1, sample 1000 ou 6000, sem grid.

A regra passou no artifact-only porque recuperou FN nos dois seeds, sem adicionar FP e sem perder TP.
Ainda assim, não deve ser promovida diretamente: primeiro precisa virar patch temporário e passar em quick-E2E.