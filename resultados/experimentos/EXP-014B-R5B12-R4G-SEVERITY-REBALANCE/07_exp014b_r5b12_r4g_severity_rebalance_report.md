# EXP-014B-R5B12-R4G-SEVERITY-REBALANCE - R4G severity rebalance

## Resultado executivo
- Status: `PASS_R5B12_ALL_CONFIRM_FRAUDS_PROMOTED_GLOBAL_TARGET_PRESERVED`
- Regras selecionadas: `5`
- Fraudes movidas de CONFIRMAR para BLOQUEAR: `5`
- Normais movidos de CONFIRMAR para BLOQUEAR: `22`
- Fraudes restantes em CONFIRMAR: `0`
- Fraudes restantes em APROVAR: `2`
- Candidatos zero-fraude para BLOQUEAR -> CONFIRMAR no R4G: `0`

## Intervencao global final
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

## BLOQUEAR final
```json
{
  "tp": 1463,
  "fp": 788,
  "fn": 2,
  "tn": 111591,
  "precision": 0.64993336,
  "recall": 0.99863481,
  "f1": 0.78740581,
  "fpr": 0.00701199
}
```

## Decisao tecnica
A variante recomendada move todas as 5 fraudes restantes em `CONFIRMAR` para
`BLOQUEAR`, ao custo de 22 normais adicionais em `BLOQUEAR`. As metricas globais
de intervencao permanecem iguais ao R4G (`FPR < 1%`, `FN=2`), pois a mudanca e
apenas de severidade entre `CONFIRMAR` e `BLOQUEAR`.
