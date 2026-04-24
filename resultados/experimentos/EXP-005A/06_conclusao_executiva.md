# EXP-005A — Conclusao Executiva

- Status: `CANDIDATO_PARA_EXP_005B`
- Vencedor model-only: `LGBM_C_SPW_2_0X`

## Resultado holdout temporal

- Policy: `P3_RECALL_098`
- Threshold selecionado: `0.0524338379`
- Precision: `0.71223`
- Recall: `0.980198`
- F1: `0.825`
- TP: `99`
- FP: `40`
- FN: `2`
- FPR: `0.002003`
- AUC: `0.999644`
- AP: `0.935179`
- Usa novas features EXP-005A: `False`

## Avaliacao model-only em sample estratificado

- TP: `353`
- FP: `20`
- FN: `2`
- Precision: `0.946381`
- Recall: `0.994366`
- F1: `0.96978`
- FPR: `0.00354296`

## Validacao model-only

- TP: `353`
- FP: `27`
- FN: `2`
- Precision: `0.928947`
- Recall: `0.994366`
- F1: `0.960544`
- FPR: `0.00478299`

## Baseline oficial FASE 2

- TP=346, FP=15, FN=9
- Precision=95.8449%, Recall=97.4648%, F1=0.9665

## Conclusao

O EXP-005A gerou um candidato de LightGBM orientado a recall.
Este resultado ainda nao e suficiente para promover runtime, porque nao passou pelo Decision Engine real.
O proximo passo e o EXP-005B: calibrar thresholds, guard rail e avaliar E2E com artefatos estaveis.

## Nota sobre E2E

Esta versao do EXP-005A nao chama `PipelineOrquestrador`, `DecisionEngine` nem `scoring_config.json`.
Isso foi feito para isolar o treino LGBM dos erros de runtime encontrados anteriormente.
