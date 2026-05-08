# EXP-005B — Recalibracao pos-LGBM v6.2

- Status: `MODEL_ONLY_CONCLUIDO_RUNTIME_NAO_APTO`
- Threshold selecionado: `0.07`
- Selection tier: `strong`

## Resultado seed 42

- TP: `352`
- FP: `20`
- FN: `3`
- Precision: `0.946237`
- Recall: `0.991549`
- F1: `0.968363`
- FPR: `0.00354296`

## Resultado seed 123

- TP: `352`
- FP: `20`
- FN: `3`
- Precision: `0.946237`
- Recall: `0.991549`
- F1: `0.968363`
- FPR: `0.00354296`

## Baseline FASE 2

- TP=346, FP=15, FN=9
- Precision=95.8449%, Recall=97.4648%, F1=0.9665

## Preflight runtime

- Runtime pronto para E2E: `False`
- scoring_config valido: `False`

## Conclusao

O EXP-005B model-only encontrou um threshold candidato, mas o runtime ainda nao esta apto para E2E.
Antes da avaliacao real, corrija `backend/artefatos/scoring_config.json`.
O erro anterior apontou JSON invalido na linha 128; este problema precisa ser resolvido antes de chamar PipelineOrquestrador.

## Observacao

Este experimento ainda e model-only. A decisao de promover depende de E2E com DecisionEngine real.
