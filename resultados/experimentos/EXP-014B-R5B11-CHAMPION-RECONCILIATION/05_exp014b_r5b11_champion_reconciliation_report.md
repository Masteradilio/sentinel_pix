# EXP-014B-R5B11-CHAMPION-RECONCILIATION - ReconciliaÃ§Ã£o do campeÃ£o

## Resultado executivo
- Status: `PASS_R5B11_R4G_CONFIRMED_AS_GLOBAL_TARGET_CANDIDATE`
- R4G cumpre alvo global: `True`
- R5B10 seguro empilhado sobre R4G: `False`
- Regras simples zero-fraude no residual BLOQUEAR do R4G: `0`

## R4G - intervenÃ§Ã£o global
```json
{
  "tp": 1463,
  "fp": 1123,
  "fn": 2,
  "tn": 111256,
  "precision": 0.56573859,
  "recall": 0.99863481,
  "f1": 0.72229079,
  "fpr": 0.00999297
}
```

## R4G - BLOQUEAR
```json
{
  "tp": 1458,
  "fp": 766,
  "fn": 7,
  "tn": 111613,
  "precision": 0.65557554,
  "recall": 0.99522184,
  "f1": 0.79045812,
  "fpr": 0.00681622
}
```

## Empilhamento R5B10 sobre R4G
```json
{
  "frauds_demoted_to_confirm": 115,
  "normals_demoted_to_confirm": 23,
  "stacked_intervention_metrics": {
    "tp": 1463,
    "fp": 1123,
    "fn": 2,
    "tn": 111256,
    "precision": 0.56573859,
    "recall": 0.99863481,
    "f1": 0.72229079,
    "fpr": 0.00999297
  },
  "stacked_block_metrics": {
    "tp": 1343,
    "fp": 743,
    "fn": 122,
    "tn": 111636,
    "precision": 0.64381592,
    "recall": 0.91672355,
    "f1": 0.75640665,
    "fpr": 0.00661156
  }
}
```

## DecisÃ£o tÃ©cnica
O baseline `EXP-014B-R4G-FAST-FROZEN` permanece o Ãºnico candidato atual que
cumpre simultaneamente `FPR < 1%` e `FN <= 5` nas mÃ©tricas globais. A polÃ­tica
R5B10 nÃ£o deve ser empilhada sobre R4G, pois demove fraudes de `BLOQUEAR` para
`CONFIRMAR`. A prÃ³xima frente deve minerar uma severidade especÃ­fica para o
residual `BLOQUEAR` do R4G ou investigar por que a inferÃªncia runtime R5B2
regrediu para `FN=682`.
