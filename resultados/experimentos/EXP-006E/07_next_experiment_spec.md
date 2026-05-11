# Próximo experimento recomendado

## EXP-006F — Quick-E2E C1 Near-Threshold

## Objetivo

Implementar C1 como patch temporário no DecisionEngine e validar com baseline + 1 candidato, sem grid, com salvamento incremental.

## Restrições de produtividade

1. Não rodar grid E2E.
2. Não testar múltiplos candidatos.
3. Salvar baseline e candidato incrementalmente.
4. Interromper se FN não cair, FP subir ou F1 piorar.

## Ação

Criar script de patch temporário e rodar somente C1. Não testar outros thresholds ou variantes.
