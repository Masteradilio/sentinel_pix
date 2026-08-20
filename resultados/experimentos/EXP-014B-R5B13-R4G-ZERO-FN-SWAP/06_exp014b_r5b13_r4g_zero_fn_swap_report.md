# EXP-014B-R5B13-R4G-ZERO-FN-SWAP - Zero-FN swap

## Resultado executivo
- Status: `PASS_R5B13_ZERO_FN_FPR_LT1`
- Fraudes APROVAR -> BLOQUEAR: `2`
- Normais APROVAR -> BLOQUEAR: `47`
- Normais CONFIRMAR -> APROVAR para compensacao: `47`
- Fraudes restantes em APROVAR: `0`
- Fraudes restantes em CONFIRMAR: `0`

## Intervencao global final
```json
{
  "tp": 1465,
  "fp": 1123,
  "fn": 0,
  "tn": 111256,
  "precision": 0.56607419,
  "recall": 1.0,
  "f1": 0.72292129,
  "fpr": 0.00999297
}
```

## BLOQUEAR final
```json
{
  "tp": 1465,
  "fp": 835,
  "fn": 0,
  "tn": 111544,
  "precision": 0.63695652,
  "recall": 1.0,
  "f1": 0.77822045,
  "fpr": 0.00743021
}
```

## Decisao tecnica
O candidato atinge `FN=0` e preserva `FPR < 1%`, mas a compensacao usa selecao
offline dos menores `lgbm_raw` entre normais remanescentes em `CONFIRMAR`.
Antes de producao, essa compensacao precisa virar regra operacional congelada
sem dependencia de label.
