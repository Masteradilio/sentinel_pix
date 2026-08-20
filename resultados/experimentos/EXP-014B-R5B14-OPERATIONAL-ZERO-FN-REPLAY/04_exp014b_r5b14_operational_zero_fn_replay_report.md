# EXP-014B-R5B14-OPERATIONAL-ZERO-FN-REPLAY - operational zero-FN replay

## Resultado executivo
- Status: `PASS_R5B14_OPERATIONAL_ZERO_FN_REPLAY`
- Regra de compensacao: `lgbm_raw <= 1.966e-05`
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
Este replay remove a dependencia de label da compensacao R5B13. Todas as camadas deste replay usam regras explicitas. Antes de runtime,
a politica ainda precisa passar por replay E2E no PipelineOrquestrador.
