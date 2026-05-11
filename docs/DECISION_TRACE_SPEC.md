# Decision Trace Spec — Pipeline Antifraude PIX

**Gerado em:** `2026-05-09T14:07:11`

## 1. Objetivo

Este documento define o formato mínimo de rastreabilidade de decisão do pipeline antifraude PIX.

O objetivo é permitir auditoria, explicabilidade, regressão, análise de FP/FN, monitoramento de drift e reconstrução posterior da decisão.

## 2. Princípio

Toda decisão deve ser reconstruível a partir de:

1. dados da transação;
2. scores dos módulos;
3. regras e guard rails aplicados;
4. versão do modelo;
5. versão do `scoring_config`; e
6. motivo final textual.

## 3. Schema mínimo

| Campo | Tipo | Obrigatório | Descrição |
|---|---|---:|---|
| `decision_id` | string | Sim | Identificador único da decisão |
| `transaction_id` | string | Sim | Identificador da transação |
| `customer_id_hash` | string | Sim | Identificador anonimizado do cliente |
| `created_at` | datetime | Sim | Timestamp da decisão |
| `model_version` | string | Sim | Versão lógica do modelo/pipeline |
| `decision_engine_version` | string | Sim | Versão do motor de decisão |
| `scoring_config_version` | string | Sim | Versão/hash do scoring_config |
| `decisao` | string | Sim | `APROVAR`, `CONFIRMAR` ou `BLOQUEAR` |
| `score_final` | float | Sim | Score final após regras/exceções |
| `score_final_original` | float | Não | Score antes de exceções, quando aplicável |
| `lgbm_raw` | float | Sim | Score bruto LGBM |
| `lgbm_mapped` | float | Sim | Score LGBM mapeado |
| `if_percentile` | float | Sim | Percentil do Isolation Forest |
| `se_score` | float | Sim | Score de engenharia social |
| `beh_score` | float | Sim | Score comportamental |
| `rules_applied` | list[string] | Sim | Regras que alteraram ou sustentaram decisão |
| `guardrails_applied` | list[string] | Sim | Guard rails aplicados |
| `veto_reason` | string | Não | Motivo de veto, se houver |
| `veto_suppressed_reason` | string | Não | Motivo de veto suprimido, se houver |
| `decision_reason` | string | Sim | Explicação textual da decisão final |
| `review_recommended` | bool | Sim | Indica se deve ir para revisão humana |

## 4. Campos específicos de regras promovidas

| Campo | Tipo | Descrição |
|---|---|---|
| `v1_guard_contextual_applied` | bool | Indica acionamento da V1 Guard Contextual |
| `v1_guard_contextual_reason` | string | Motivo do acionamento da V1 |
| `exp006f_c1_applied` | bool | Indica acionamento da C1 |
| `exp006f_c1_reason` | string | Motivo do acionamento da C1 |
| `decisao_original_exp006f_c1` | string | Decisão antes da C1 |
| `score_final_original_exp006f_c1` | float | Score antes da C1 |

## 5. Motivos padronizados de decisão

Sugestão de códigos controlados:

```text
BASE_SCORE_THRESHOLD_CONFIRMAR
BASE_SCORE_THRESHOLD_BLOQUEAR
LGBM_GUARD_RAIL_APPLIED
V1_GUARD_CONTEXTUAL_APPLIED
C1_NEAR_THRESHOLD_APPLIED
SE_RULE_SIGNAL
BEH_RULE_SIGNAL
FAST_APPROVE
APPROVE_LOW_RISK
MANUAL_REVIEW_REQUIRED
```

## 6. Política de versionamento

Toda decisão deve registrar:

- versão do modelo ativo;
- versão do motor de decisão;
- versão ou hash do `scoring_config.json`; e
- data/hora da decisão.

Sem esses campos, a decisão não deve ser considerada plenamente auditável.

## 7. Uso em monitoramento

Os campos de trace serão usados para:

- calcular taxa de C1;
- calcular taxa de V1;
- auditar FPs e FNs;
- monitorar drift de scores;
- construir fila de revisão humana;
- explicar decisões para auditoria interna;
- comparar versões futuras do modelo.
