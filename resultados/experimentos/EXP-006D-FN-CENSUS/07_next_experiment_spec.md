# EXP-006D — Próximo Experimento

## Recomendado: EXP-006E — Residual FN Counterfactual Designer

## Objetivo

Usar o censo dos FNs para desenhar uma hipótese cirúrgica que recupere pelo menos 1 FN sem adicionar FP e sem perder TP.

## Regras de produtividade

1. Rodar primeiro artifact-only.
2. Não rodar grid E2E.
3. Não testar mais de 1 candidato por quick-E2E.
4. Só rodar quick-E2E se a simulação artifact-only mostrar ganho líquido.
5. Parar se FP subir, FN não cair ou houver TP perdido.

## Critério mínimo para quick-E2E

- Delta FN estimado <= -1; ou
- Delta FP estimado <= -2 sem TP perdido.

## Critério de promoção

- FN cai nos dois seeds; ou FP cai nos dois seeds;
- F1 não piora;
- nenhum TP perdido;
- explicação causal clara por classe de erro.
