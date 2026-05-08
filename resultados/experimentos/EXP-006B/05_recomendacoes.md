# EXP-006B — Engine Counterfactual Audit

Gerado em: `2026-04-30T19:57:49`

## Conclusão executiva

Há pelo menos uma regra com ganho líquido estimado para `quick-e2e`.
Mesmo assim, a regra deve passar pelo protocolo rápido antes de qualquer promoção.

## Auditoria de thresholds LGBM

As configs candidatas tiveram métricas idênticas por seed. Isso sugere que lgbm_effective_threshold/guard_threshold não diferenciou a decisão final, ou que a lógica do engine sobrepôs esses parâmetros.

- Todas as configs candidatas idênticas por seed: `True`

## Regras contrafactuais avaliadas

| Regra | FN rec. | TP perdido | FP add. | FP rem. | Net TP | Net FP | Recomendação |
|---|---:|---:|---:|---:|---:|---:|---|
| `R2_LOW_VALUE_GRAY_FIRST_RECEIVER` | 8 | 0 | 2 | 0 | 8 | 2 | `CANDIDATO_QUICK_E2E` |
| `R1_LGBM_GRAY_FIRST_RECEIVER_STRICT` | 2 | 0 | 2 | 0 | 2 | 2 | `CANDIDATO_QUICK_E2E` |
| `R7_LGBM_EFFECTIVE_ZONE_FIRST_RECEIVER` | 8 | 0 | 19 | 1 | 8 | 18 | `REJEITAR_FP_OU_TP_PERDIDO` |
| `R0_ALL_MOVED_CASES` | 8 | 8 | 20 | 2 | 0 | 18 | `REJEITAR_FP_OU_TP_PERDIDO` |
| `R5_GUARD_EXCEPTION_ALTO_VALOR` | 0 | 0 | 0 | 0 | 0 | 0 | `SEM_GANHO` |
| `R6_VETO_SUPPRESSED_ONLY` | 0 | 0 | 2 | 1 | 0 | 1 | `SEM_GANHO` |
| `R3_FIRST_RECEIVER_IF_EXTREME` | 0 | 0 | 3 | 0 | 0 | 3 | `SEM_GANHO` |
| `R4_FIRST_RECEIVER_SE_OR_BEH` | 0 | 0 | 7 | 0 | 0 | 7 | `SEM_GANHO` |

## Veto suppressed audit

- `VETO_SUPPRESSED_BUT_CONFIRMOU`: 2
- `VETO_EFFECTIVE_APPROVE`: 1

## Recoverability map

- `FP_RISK_PATTERN`: 19
- `BASELINE_FRAGILE_TP`: 4
- `RECOVERABLE_LOW_COST`: 4
- `FP_REDUCIBLE`: 2

## Decisão

Rodar `quick-e2e` apenas para a melhor regra candidata, com baseline + 1 candidato, sample 1000, seed 42.
Se FN não cair ou FP subir acima de baseline +3, interromper.

## Próximo passo recomendado

`EXP-006C — Baseline Residual FN Census`

Objetivo: gerar uma tabela completa dos 9 FNs residuais do baseline, com LGBM, IF, SE, BEH, score final, veto e features-chave.
Sem essa tabela, estamos inferindo a irredutibilidade apenas pelos casos movimentados, não pela fronteira completa.
