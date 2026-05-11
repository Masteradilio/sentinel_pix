# Rules Catalog — Pipeline Antifraude PIX

**Gerado em:** `2026-05-09T14:07:11`

## 1. Objetivo

Este documento cataloga as regras, thresholds, exceções e guard rails do pipeline antifraude PIX após o fechamento da FASE 2.

A versão oficial documentada aqui é o **baseline pós-C1**, com:

- `V1_GUARD_CONTEXTUAL` promovida;
- `C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER` promovida;
- LGBM v6.2 rejeitado para runtime;
- R2 rejeitada;
- meta-learner shadow mantido apenas como diagnóstico;
- EXP-003 residual desligado.

## 2. Sanidade da configuração

✅ Todas as chaves oficiais pós-FASE 2 foram encontradas no `scoring_config.json`.

## 3. Resumo das regras ativas

| Regra / Componente | Status | Ação | Flag / Campo | Origem |
|---|---|---|---|---|
| Threshold de confirmação | Ativo | `APROVAR → CONFIRMAR` quando score >= threshold | `threshold_confirmar` | Baseline pós-FASE 1/2 |
| Threshold de bloqueio | Ativo | `CONFIRMAR → BLOQUEAR` quando score >= threshold | `threshold_bloquear` | Baseline pós-FASE 1/2 |
| Guard rail LGBM | Ativo | Evita confirmação por score fraco do LGBM em contexto específico | `lgbm_guard_enabled` | FASE 1/2 |
| V1 Guard Contextual | Ativo | Exceção contextual de alto valor | `guard_exception_alto_valor_se_beh_enabled` | EXP-004-FINAL |
| C1 Near-Threshold | Ativo | `APROVAR → CONFIRMAR` em caso near-threshold específico | `exp006f_c1_enabled` | EXP-006E/006F |
| Social Engineering Rules | Ativo conforme módulo | Soma sinais de engenharia social | regras internas SE | Módulo SE |
| Behavioral Rules | Ativo conforme módulo | Soma sinais comportamentais | regras internas BEH | Módulo BEH |

## 4. Threshold de confirmação

| Campo | Valor |
|---|---:|
| `threshold_confirmar` | `62.0` |

**Ação:** quando o `score_final` atinge ou supera esse threshold, a transação pode ser promovida para `CONFIRMAR`, salvo vetos ou guard rails aplicáveis.

**Risco operacional:** se baixo demais, aumenta FP; se alto demais, aumenta FN.

**Critério de alteração:** só pode ser alterado após validação `artifact-only → quick-E2E → final-E2E`, mantendo ou melhorando F1 e sem aumento inseguro de FP.

## 5. Threshold de bloqueio

| Campo | Valor |
|---|---:|
| `threshold_bloquear` | `95.0` |

**Ação:** quando o `score_final` atinge ou supera esse threshold, a transação pode ser classificada como `BLOQUEAR`.

**Risco operacional:** bloqueio indevido é mais grave do que confirmação para análise humana. Alterações exigem validação mais conservadora.

## 6. Guard rail LGBM

| Campo | Valor |
|---|---:|
| `lgbm_guard_enabled` | `True` |
| `lgbm_guard_threshold` | `0.3` |

**Status:** ativo.

**Objetivo:** impedir que o engine confirme transações quando o componente supervisionado LGBM não oferece suporte suficiente, protegendo precisão e FP.

**Origem:** calibragem pós-FASE 1 e validações da FASE 2.

**Risco operacional:** guard rail agressivo demais pode suprimir FNs recuperáveis; guard rail frouxo demais pode elevar FP.

**Critério de desligamento:** somente se uma validação E2E mostrar redução líquida de FN sem aumento material de FP e sem perda de F1.

## 7. V1_GUARD_CONTEXTUAL

| Campo | Valor |
|---|---:|
| `guard_exception_alto_valor_se_beh_enabled` | `True` |
| `guard_exception_alto_valor_min` | `15000.0` |
| `guard_exception_alto_valor_rel_max` | `12.0` |
| `guard_exception_alto_valor_if_min` | `0.985` |
| `guard_exception_alto_valor_lgbm_min` | `0.01` |
| `guard_exception_alto_valor_require_first_receiver` | `True` |
| `guard_exception_alto_valor_require_pf` | `True` |

**Status:** promovida.

**Origem:** EXP-004-FINAL.

**Ação:** exceção contextual para recuperar fraude de alto valor em cenário de risco composto.

**Evidência:** recuperou FN sem adicionar FP na validação da FASE 1.

**Risco operacional:** regra de alto impacto por envolver valor elevado; deve permanecer estreita e configurável.

**Critério de desligamento:** se backtest futuro mostrar FP relevante, drift ou perda de precisão em transações legítimas de alto valor.

## 8. C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER

| Campo | Valor |
|---|---:|
| `exp006f_c1_enabled` | `True` |
| `exp006f_c1_min_score` | `58.0` |
| `exp006f_c1_max_score` | `62.0` |
| `exp006f_c1_min_valor` | `100.0` |
| `exp006f_c1_max_valor` | `500.0` |
| `exp006f_c1_max_rel_meses` | `12.0` |
| `exp006f_c1_min_lgbm_raw` | `0.06` |
| `exp006f_c1_max_lgbm_raw` | `0.1` |
| `exp006f_c1_require_first_receiver` | `True` |
| `exp006f_c1_require_not_pix_random` | `True` |
| `exp006f_c1_max_se_score` | `0.0` |
| `exp006f_c1_max_beh_score` | `0.0` |

**Status:** promovida.

**Origem:** EXP-006E / EXP-006F.

**Ação:** promover `APROVAR → CONFIRMAR` quando todas as condições abaixo forem verdadeiras:

```text
decisao == APROVAR
first_receiver_flag == 1
pix_key_random_flag == 0
qt_tempo_relacionamento_mes <= 12
100 <= vl_pix < 500
0.06 <= lgbm_raw < 0.10
58 <= score_final < 62
se_score <= 0
beh_score <= 0
```

**Evidência:**

- recuperou 1 FN no seed 42;
- recuperou 1 FN no seed 123;
- adicionou 0 FP;
- perdeu 0 TP;
- validada em runtime real na transação `E0000020820260205003505340630525`.

**Risco operacional:** regra estreita, mas sensível a drift de `score_final`, `lgbm_raw` e perfil de primeiro recebedor.

**Critério de desligamento:**

- qualquer aumento confirmado de FP associado à C1 em backtest novo;
- aumento anormal da taxa de disparo da C1;
- drift relevante em `score_final` ou `lgbm_raw`; ou
- nova validação temporal mostrar que a regra não recupera fraude.

## 9. Regras e candidatos rejeitados

| Regra / Modelo | Status | Motivo da rejeição |
|---|---|---|
| LGBM v6.2 / `LGBM_C_SPW_2_0X` | Rejeitado para runtime | Promissor model-only, mas no engine real não reduziu FN líquido e aumentou FP |
| R2_LOW_VALUE_GRAY_FIRST_RECEIVER | Rejeitada | Recuperou 0 FN e adicionou FP nos seeds avaliados |
| EXP-003 residual | Desligado | Risco de FP / não aprovado para baseline final |
| Meta-Learner Shadow EXP-007A | Diagnóstico apenas | Não encontrou candidato seguro adicional |
| Regra ampla `first_receiver_flag` | Proibida como regra isolada | Sinal aparece em FNs, mas também domina FPs adicionados |

## 10. Regras desligadas no scoring_config

| Campo | Valor | Interpretação |
|---|---:|---|
| `se_pattern_residual_enabled` | `False` | Padrão residual SE desligado |
| `exp003_residual_confirm_enabled` | `False` | Residual EXP-003 desligado |

## 11. Campos obrigatórios de decisão

Todo resultado final de decisão deve possuir, no mínimo:

```text
transaction_id
decision_id
model_version
scoring_config_version
decisao
score_final
lgbm_raw
lgbm_mapped
if_percentile
se_score
beh_score
rules_applied
guardrails_applied
decision_reason
created_at
```

## 12. Procedimento de regressão obrigatório

Antes de alterar qualquer regra, threshold, artefato ou lógica de decisão, executar:

```powershell
python -m pytest tests\test_regression_post_fase2.py -q
python -m pytest tests\test_regression_post_fase2.py -q -m slow
```

## 13. Critério de manutenção

O catálogo deve ser atualizado sempre que:

- uma regra for promovida;
- uma regra for desligada;
- um threshold for alterado;
- um candidato for rejeitado formalmente;
- novos dados alterarem a decisão de promoção/rejeição;
- o `DecisionEngine` mudar a composição do score.
