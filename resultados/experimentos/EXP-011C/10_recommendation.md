# EXP-011C — E2E Shadow LGBM vNext

## Status sugerido
DIAGNOSTICO_E2E_SHADOW_REQUER_ANALISE

## Resumo
- Linhas: 100000
- Fraudes: 1306
- Normais: 98694
- Threshold R1: 0.6
- Threshold R2: 0.53

## HOLDOUT
### Baseline produtivo
- tp: 12
- fp: 73
- fn: 46
- tn: 8691
- precision: 0.14117647
- recall: 0.20689655
- f1: 0.16783217
- fpr: 0.00832953

### R1 model-only
- tp: 15
- fp: 14
- fn: 43
- tn: 8750
- precision: 0.51724138
- recall: 0.25862069
- f1: 0.34482759
- fpr: 0.00159744

### R1 assist baseline
- tp: 17
- fp: 85
- fn: 41
- tn: 8679
- precision: 0.16666667
- recall: 0.29310345
- f1: 0.2125
- fpr: 0.00969877

### R2 assist baseline
- tp: 15
- fp: 78
- fn: 43
- tn: 8686
- precision: 0.16129032
- recall: 0.25862069
- f1: 0.1986755
- fpr: 0.00890005

## Próxima decisão
Se o R1 assist baseline melhorar o F1 e reduzir/segurar FP versus baseline no holdout, seguir para EXP-011D com patch shadow controlado no DecisionEngine. Se o R1 model-only for melhor que o assist, avaliar substituição do score do modelo em shadow. Não promover automaticamente sem validar no dataset completo não truncado e sem regressão dos testes.