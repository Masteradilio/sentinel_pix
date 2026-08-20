# EXP-014B-R5B17-PIPELINE-HOMOLOGATION - pipeline homologation

## Resultado executivo
- Status: `FAIL_R5B17_PIPELINE_HOMOLOGATION`
- Modo: `sample`
- Linhas: `200`
- Fraudes: `3`
- Erros de pipeline: `0`

## Metricas globais
```json
{
  "tp": 1,
  "fp": 9,
  "fn": 2,
  "tn": 188,
  "precision": 0.1,
  "recall": 0.33333333,
  "f1": 0.15384615,
  "fpr": 0.04568528
}
```

## Metricas BLOQUEAR
```json
{
  "tp": 1,
  "fp": 9,
  "fn": 2,
  "tn": 188,
  "precision": 0.1,
  "recall": 0.33333333,
  "f1": 0.15384615,
  "fpr": 0.04568528
}
```

## Gates
```json
{
  "no_pipeline_errors": true,
  "fpr_lt_1pct": false,
  "fn_lte_5_outside_block": true,
  "approve_frauds_eq_0": false,
  "confirm_frauds_eq_0": true
}
```
