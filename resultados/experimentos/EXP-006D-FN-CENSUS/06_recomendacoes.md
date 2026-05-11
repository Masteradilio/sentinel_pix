# EXP-006D — Baseline Residual FN Census

Gerado em: `2026-05-09T11:02:03`

## Objetivo

Classificar os FNs e FPs residuais do baseline pós-FASE 1 usando apenas artefatos existentes.

## Métricas baseline observadas

| Seed | TP | FP | FN | Precision | Recall | F1 | FPR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 346 | 14 | 9 | 96.1111% | 97.4648% | 0.9678 | 0.2480% |
| 123 | 346 | 12 | 9 | 96.6480% | 97.4648% | 0.9705 | 0.2126% |

## Censo dos FNs

- Linhas FN: `18`
- Transações FN únicas: `9`
- Provavelmente recuperáveis: `1`
- Provavelmente limitadas por dados: `6`

### Classes FN

- `DATA_LIMITED_WEAK_ALL_MODULES`: 12
- `MIXED_OR_UNCLEAR`: 2
- `NEAR_THRESHOLD`: 2
- `MODULE_SIGNAL_WEAK`: 2

## Censo dos FPs

- Linhas FP: `26`
- Transações FP únicas: `25`
- FPs potencialmente reducíveis: `0`

### Classes FP

- `FP_FIRST_RECEIVER_LOW_CONTEXT`: 14
- `FP_NOT_OBVIOUS`: 10
- `FP_WITH_SUPPRESSED_VETO`: 2

## Recomendação técnica

Há FNs com algum sinal recuperável. O próximo experimento deve ser artifact-only ou shadow, focado somente nesses casos.
Não criar regra ampla; gerar contrafactual por classe FN e estimar impacto sobre FPs/TPs antes de qualquer E2E.

## Próximo passo recomendado

`EXP-006E — Residual FN/FP Counterfactual Designer`

Objetivo: a partir do censo, gerar 1 única hipótese candidata, artifact-only, e só então decidir se vale quick-E2E.
