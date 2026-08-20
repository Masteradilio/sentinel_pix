# EXP-014B-R5B15-CORE-POLICY-REPLAY - Core policy replay

## Resultado executivo
- Status: `PASS_R5B15_CORE_POLICY_REPLAY_MATCHED_R5B14`
- All pass: `True`
- Fraudes restantes em APROVAR: `0`
- Fraudes restantes em CONFIRMAR: `0`

## Checks
```json
{
  "intervention_metrics_match_r5b14": true,
  "block_metrics_match_r5b14": true,
  "r5b12_counts_match_r5b14": true,
  "approve_to_block_counts_match_r5b14": true,
  "compensation_counts_match_r5b14": true
}
```

## Intervencao global
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

## BLOQUEAR
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
