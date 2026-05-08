# Protocolo de Experimentos Rápidos — pós-EXP-006

## Regra operacional

Nenhum experimento novo deve rodar grid E2E completo por padrão.

## Pipeline obrigatório

1. `artifact-only`: usar CSV/JSON já existentes.
2. `model-only`: rodar em segundos/minutos.
3. `quick-e2e`: baseline + 1 candidato, sample 1000, seed 42.
4. `final-e2e`: baseline + 1 candidato, sample 6000, seeds 42 e 123.

## Critério de interrupção

Parar no quick-e2e se:

- FN não cair;
- FP subir acima do baseline + 3;
- F1 cair;
- houver TP perdido sem ganho líquido.

## Critério de promoção

Promover apenas se:

- FN cair nos dois seeds;
- FP ficar dentro do limite;
- F1 não piorar materialmente;
- a mudança for explicável por cluster de erro.
