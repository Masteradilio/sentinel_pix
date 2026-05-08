# EXP-006B — Próximo Experimento

## Recomendado: Quick-E2E para R2_LOW_VALUE_GRAY_FIRST_RECEIVER

## Objetivo

Validar em sample pequeno se a regra contrafactual realmente reduz FN sem aumentar FP acima do limite.

## Protocolo obrigatório

1. Não rodar grid E2E.
2. Não rodar múltiplos candidatos.
3. Salvar resultado imediatamente após baseline.
4. Salvar resultado imediatamente após candidato.
5. Interromper se FN não cair ou FP subir acima de baseline +3.

## Comando / ação

Rodar baseline + 1 candidato, sample=1000, seed=42, workers=4, salvamento incremental.
