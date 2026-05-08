# EXP-006 — Decisão Técnica

## Decisão

Manter baseline pós-FASE 1 e não promover LGBM v6.2.

## Próximo experimento recomendado

`EXP-006B — Engine Counterfactual Audit`

Objetivo: testar contrafactuais leves sobre os outputs já existentes, sem rodar E2E completo.

Contrafactuais candidatos:

1. Ajuste local de score para `LGBM_GRAY_FIRST_RECEIVER` somente se FP estimado for baixo.
2. Auditoria de veto para casos com `veto_suppressed_reason` real.
3. Classificação dos 9 FNs residuais em `RECOVERABLE` vs `DATA_LIMITED`.
4. Verificar se `lgbm_effective_threshold` realmente influencia o engine ou se está sendo sobreposto.

## Hipóteses geradas

### H1_FIRST_RECEIVER_EH_SINAL_MAS_NAO_REGRA

- Status: `NAO_PROMOVER_COMO_REGRA`
- Interpretação: first_receiver aparece nos FNs recuperados, mas também domina FPs adicionados. Não deve virar regra hardcoded; deve entrar apenas como feature/explicação.
- Próxima ação: Não criar regra first_receiver. Usar em meta-learner shadow e análise de reputação de recebedor.

### H2_RECALIBRACAO_LGBM_TROCA_FRAUDES

- Status: `REJEITAR_LGBM_V6_2_RUNTIME`
- Interpretação: O candidato recupera alguns FNs, mas perde TPs em quantidade similar e adiciona FP. Isso confirma que a fronteira do LGBM v6.2 não melhora o pipeline real.
- Próxima ação: Manter baseline pós-FASE 1 e usar LGBM v6.2 apenas como evidência diagnóstica.

### H3_FRAUDES_COM_SINAL_FRACO_PODEM_SER_IRREDUTIVEIS

- Status: `INVESTIGAR_RESIDUAL`
- Interpretação: Há fraudes capturadas pelo baseline que ficam com LGBM muito baixo e SE/BEH zerados no candidato. Se também estiverem entre os FNs residuais, dependem de novos sinais ou ajustes no engine.
- Próxima ação: Criar relatório dos 9 FNs residuais completos e marcar DATA_LIMITED quando todos os módulos estiverem cegos.

### H4_VETO_SUPPRESSED_DEVE_SER_CAMPO_DIAGNOSTICO

- Status: `MANTER_CORRECAO_NAN_E_USAR_NO_EXP006B`
- Interpretação: veto_suppressed_reason aparece em alguns casos e pode explicar decisões suprimidas. O bug de NaN precisa continuar corrigido para não contaminar classificação de FN.
- Próxima ação: No EXP-006B, gerar contrafactual de veto sem reprocessar E2E completo.
