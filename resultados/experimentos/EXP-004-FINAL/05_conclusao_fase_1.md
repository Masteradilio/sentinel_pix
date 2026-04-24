# EXP-004-FINAL - Conclusao da FASE 1

- Vencedor: `V1_GUARD_CONTEXTUAL`
- Status: `NAO_APROVADO_AUTOMATICAMENTE`

## Resultado principal

- Baseline: TP=345, FP=15, FN=10, Precision=95.8333%, Recall=97.1831%, F1=0.9650, FPR=0.2657%
- Vencedor: TP=346, FP=15, FN=9, Precision=95.8449%, Recall=97.4648%, F1=0.9665, FPR=0.2657%
- Delta: TP=+1, FP=+0, FN=-1, F1=+0.0014
- Utility antifraude: 10.00
- Valor de FN recuperado: R$ 20000.00
- Valor de FP adicionado: R$ 0.00

## Criterios de aceite

- FP <= 20: `True`
- Precision >= 94.00%: `True`
- Recall >= 98.31%: `False`
- FPR <= 0.50%: `True`
- F1 nao decrescente: `True`
- TP nao decrescente: `True`

## Validacao cruzada

- Seed: `123`
- Vencedor validado: `V1_GUARD_CONTEXTUAL`
- TP=346, FP=12, FN=9
- Precision=96.6480%
- Recall=97.4648%
- F1=0.9705
- FPR=0.2126%

## FNs residuais

- Total de FNs residuais no vencedor: `9`
- Categorias: `{"GUARD_SUPPRESSED_CANDIDATE": 9}`

## Recomendacao

`V1_GUARD_CONTEXTUAL` teve melhoria local, mas nao passou todos os criterios fortes. Considerar deploy parcial ou ajustar thresholds antes de promover.
