# EXP-014B-R5B16-CONSOLIDATED-OPERATIONAL-BASELINE - baseline operacional candidato

## Resultado executivo
- Status: `PASS_R5B16_OPERATIONAL_BASELINE_CANDIDATE_CONSOLIDATED`
- Artefato candidato: `backend\artefatos_candidatos\exp014b_r5b16_operational_baseline\operational_baseline_candidate.json`
- Ativo por default: `False`
- Flag runtime: `ENABLE_R5B14_POLICY`

## Metricas globais
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

## Metricas BLOQUEAR
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
R5B16 consolida R5B15 como baseline operacional candidato. A politica esta
centralizada no core e conectada ao orquestrador por configuracao, mas permanece
desligada por default ate replay batch completo no ambiente produtivo.
