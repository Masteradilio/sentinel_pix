"""
scripts/build_rules_catalog_and_decision_trace.py

EXP-008C — Rules Catalog e Decision Trace

Gera:
  docs/RULES_CATALOG.md
  docs/DECISION_TRACE_SPEC.md
  docs/DECISION_TRACE_EXAMPLE.json

Objetivo:
  Documentar regras ativas, regras rejeitadas, thresholds, guard rails,
  flags de configuração e especificação mínima de decision trace.

Uso:
  python scripts\\build_rules_catalog_and_decision_trace.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists():
            return p
    return start.parent


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = find_project_root(SCRIPT_DIR)

DOCS_DIR = ROOT / "docs"
SCORING_PATH = ROOT / "backend" / "artefatos" / "scoring_config.json"

RULES_CATALOG_PATH = DOCS_DIR / "RULES_CATALOG.md"
DECISION_TRACE_SPEC_PATH = DOCS_DIR / "DECISION_TRACE_SPEC.md"
DECISION_TRACE_EXAMPLE_PATH = DOCS_DIR / "DECISION_TRACE_EXAMPLE.json"


REQUIRED_CONFIG_KEYS = [
    "threshold_confirmar",
    "threshold_bloquear",
    "lgbm_guard_enabled",
    "lgbm_guard_threshold",
    "guard_exception_alto_valor_se_beh_enabled",
    "exp006f_c1_enabled",
    "exp006f_c1_min_score",
    "exp006f_c1_max_score",
    "exp006f_c1_min_valor",
    "exp006f_c1_max_valor",
    "exp006f_c1_max_rel_meses",
    "exp006f_c1_min_lgbm_raw",
    "exp006f_c1_max_lgbm_raw",
    "exp006f_c1_require_first_receiver",
    "exp006f_c1_require_not_pix_random",
    "exp006f_c1_max_se_score",
    "exp006f_c1_max_beh_score",
    "se_pattern_residual_enabled",
    "exp003_residual_confirm_enabled",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def cfg(config: dict[str, Any], key: str, default: Any = None) -> Any:
    return config.get(key, default)


def yes_no(value: Any) -> str:
    return "Sim" if bool(value) else "Não"


def generate_rules_catalog(config: dict[str, Any]) -> str:
    now = datetime.now().isoformat(timespec="seconds")

    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]

    lines: list[str] = [
        "# Rules Catalog — Pipeline Antifraude PIX",
        "",
        f"**Gerado em:** `{now}`",
        "",
        "## 1. Objetivo",
        "",
        "Este documento cataloga as regras, thresholds, exceções e guard rails do pipeline antifraude PIX após o fechamento da FASE 2.",
        "",
        "A versão oficial documentada aqui é o **baseline pós-C1**, com:",
        "",
        "- `V1_GUARD_CONTEXTUAL` promovida;",
        "- `C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER` promovida;",
        "- LGBM v6.2 rejeitado para runtime;",
        "- R2 rejeitada;",
        "- meta-learner shadow mantido apenas como diagnóstico;",
        "- EXP-003 residual desligado.",
        "",
        "## 2. Sanidade da configuração",
        "",
    ]

    if missing:
        lines.extend([
            "⚠️ Configuração incompleta. As seguintes chaves esperadas não foram encontradas:",
            "",
        ])
        for key in missing:
            lines.append(f"- `{key}`")
        lines.append("")
    else:
        lines.extend([
            "✅ Todas as chaves oficiais pós-FASE 2 foram encontradas no `scoring_config.json`.",
            "",
        ])

    lines.extend([
        "## 3. Resumo das regras ativas",
        "",
        "| Regra / Componente | Status | Ação | Flag / Campo | Origem |",
        "|---|---|---|---|---|",
        "| Threshold de confirmação | Ativo | `APROVAR → CONFIRMAR` quando score >= threshold | `threshold_confirmar` | Baseline pós-FASE 1/2 |",
        "| Threshold de bloqueio | Ativo | `CONFIRMAR → BLOQUEAR` quando score >= threshold | `threshold_bloquear` | Baseline pós-FASE 1/2 |",
        "| Guard rail LGBM | Ativo | Evita confirmação por score fraco do LGBM em contexto específico | `lgbm_guard_enabled` | FASE 1/2 |",
        "| V1 Guard Contextual | Ativo | Exceção contextual de alto valor | `guard_exception_alto_valor_se_beh_enabled` | EXP-004-FINAL |",
        "| C1 Near-Threshold | Ativo | `APROVAR → CONFIRMAR` em caso near-threshold específico | `exp006f_c1_enabled` | EXP-006E/006F |",
        "| Social Engineering Rules | Ativo conforme módulo | Soma sinais de engenharia social | regras internas SE | Módulo SE |",
        "| Behavioral Rules | Ativo conforme módulo | Soma sinais comportamentais | regras internas BEH | Módulo BEH |",
        "",
        "## 4. Threshold de confirmação",
        "",
        "| Campo | Valor |",
        "|---|---:|",
        f"| `threshold_confirmar` | `{cfg(config, 'threshold_confirmar')}` |",
        "",
        "**Ação:** quando o `score_final` atinge ou supera esse threshold, a transação pode ser promovida para `CONFIRMAR`, salvo vetos ou guard rails aplicáveis.",
        "",
        "**Risco operacional:** se baixo demais, aumenta FP; se alto demais, aumenta FN.",
        "",
        "**Critério de alteração:** só pode ser alterado após validação `artifact-only → quick-E2E → final-E2E`, mantendo ou melhorando F1 e sem aumento inseguro de FP.",
        "",
        "## 5. Threshold de bloqueio",
        "",
        "| Campo | Valor |",
        "|---|---:|",
        f"| `threshold_bloquear` | `{cfg(config, 'threshold_bloquear')}` |",
        "",
        "**Ação:** quando o `score_final` atinge ou supera esse threshold, a transação pode ser classificada como `BLOQUEAR`.",
        "",
        "**Risco operacional:** bloqueio indevido é mais grave do que confirmação para análise humana. Alterações exigem validação mais conservadora.",
        "",
        "## 6. Guard rail LGBM",
        "",
        "| Campo | Valor |",
        "|---|---:|",
        f"| `lgbm_guard_enabled` | `{cfg(config, 'lgbm_guard_enabled')}` |",
        f"| `lgbm_guard_threshold` | `{cfg(config, 'lgbm_guard_threshold')}` |",
        "",
        "**Status:** ativo.",
        "",
        "**Objetivo:** impedir que o engine confirme transações quando o componente supervisionado LGBM não oferece suporte suficiente, protegendo precisão e FP.",
        "",
        "**Origem:** calibragem pós-FASE 1 e validações da FASE 2.",
        "",
        "**Risco operacional:** guard rail agressivo demais pode suprimir FNs recuperáveis; guard rail frouxo demais pode elevar FP.",
        "",
        "**Critério de desligamento:** somente se uma validação E2E mostrar redução líquida de FN sem aumento material de FP e sem perda de F1.",
        "",
        "## 7. V1_GUARD_CONTEXTUAL",
        "",
        "| Campo | Valor |",
        "|---|---:|",
        f"| `guard_exception_alto_valor_se_beh_enabled` | `{cfg(config, 'guard_exception_alto_valor_se_beh_enabled')}` |",
        f"| `guard_exception_alto_valor_min` | `{cfg(config, 'guard_exception_alto_valor_min', 'n/a')}` |",
        f"| `guard_exception_alto_valor_rel_max` | `{cfg(config, 'guard_exception_alto_valor_rel_max', 'n/a')}` |",
        f"| `guard_exception_alto_valor_if_min` | `{cfg(config, 'guard_exception_alto_valor_if_min', 'n/a')}` |",
        f"| `guard_exception_alto_valor_lgbm_min` | `{cfg(config, 'guard_exception_alto_valor_lgbm_min', 'n/a')}` |",
        f"| `guard_exception_alto_valor_require_first_receiver` | `{cfg(config, 'guard_exception_alto_valor_require_first_receiver', 'n/a')}` |",
        f"| `guard_exception_alto_valor_require_pf` | `{cfg(config, 'guard_exception_alto_valor_require_pf', 'n/a')}` |",
        "",
        "**Status:** promovida.",
        "",
        "**Origem:** EXP-004-FINAL.",
        "",
        "**Ação:** exceção contextual para recuperar fraude de alto valor em cenário de risco composto.",
        "",
        "**Evidência:** recuperou FN sem adicionar FP na validação da FASE 1.",
        "",
        "**Risco operacional:** regra de alto impacto por envolver valor elevado; deve permanecer estreita e configurável.",
        "",
        "**Critério de desligamento:** se backtest futuro mostrar FP relevante, drift ou perda de precisão em transações legítimas de alto valor.",
        "",
        "## 8. C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER",
        "",
        "| Campo | Valor |",
        "|---|---:|",
        f"| `exp006f_c1_enabled` | `{cfg(config, 'exp006f_c1_enabled')}` |",
        f"| `exp006f_c1_min_score` | `{cfg(config, 'exp006f_c1_min_score')}` |",
        f"| `exp006f_c1_max_score` | `{cfg(config, 'exp006f_c1_max_score')}` |",
        f"| `exp006f_c1_min_valor` | `{cfg(config, 'exp006f_c1_min_valor')}` |",
        f"| `exp006f_c1_max_valor` | `{cfg(config, 'exp006f_c1_max_valor')}` |",
        f"| `exp006f_c1_max_rel_meses` | `{cfg(config, 'exp006f_c1_max_rel_meses')}` |",
        f"| `exp006f_c1_min_lgbm_raw` | `{cfg(config, 'exp006f_c1_min_lgbm_raw')}` |",
        f"| `exp006f_c1_max_lgbm_raw` | `{cfg(config, 'exp006f_c1_max_lgbm_raw')}` |",
        f"| `exp006f_c1_require_first_receiver` | `{cfg(config, 'exp006f_c1_require_first_receiver')}` |",
        f"| `exp006f_c1_require_not_pix_random` | `{cfg(config, 'exp006f_c1_require_not_pix_random')}` |",
        f"| `exp006f_c1_max_se_score` | `{cfg(config, 'exp006f_c1_max_se_score')}` |",
        f"| `exp006f_c1_max_beh_score` | `{cfg(config, 'exp006f_c1_max_beh_score')}` |",
        "",
        "**Status:** promovida.",
        "",
        "**Origem:** EXP-006E / EXP-006F.",
        "",
        "**Ação:** promover `APROVAR → CONFIRMAR` quando todas as condições abaixo forem verdadeiras:",
        "",
        "```text",
        "decisao == APROVAR",
        "first_receiver_flag == 1",
        "pix_key_random_flag == 0",
        "qt_tempo_relacionamento_mes <= 12",
        "100 <= vl_pix < 500",
        "0.06 <= lgbm_raw < 0.10",
        "58 <= score_final < 62",
        "se_score <= 0",
        "beh_score <= 0",
        "```",
        "",
        "**Evidência:**",
        "",
        "- recuperou 1 FN no seed 42;",
        "- recuperou 1 FN no seed 123;",
        "- adicionou 0 FP;",
        "- perdeu 0 TP;",
        "- validada em runtime real na transação `E0000020820260205003505340630525`.",
        "",
        "**Risco operacional:** regra estreita, mas sensível a drift de `score_final`, `lgbm_raw` e perfil de primeiro recebedor.",
        "",
        "**Critério de desligamento:**",
        "",
        "- qualquer aumento confirmado de FP associado à C1 em backtest novo;",
        "- aumento anormal da taxa de disparo da C1;",
        "- drift relevante em `score_final` ou `lgbm_raw`; ou",
        "- nova validação temporal mostrar que a regra não recupera fraude.",
        "",
        "## 9. Regras e candidatos rejeitados",
        "",
        "| Regra / Modelo | Status | Motivo da rejeição |",
        "|---|---|---|",
        "| LGBM v6.2 / `LGBM_C_SPW_2_0X` | Rejeitado para runtime | Promissor model-only, mas no engine real não reduziu FN líquido e aumentou FP |",
        "| R2_LOW_VALUE_GRAY_FIRST_RECEIVER | Rejeitada | Recuperou 0 FN e adicionou FP nos seeds avaliados |",
        "| EXP-003 residual | Desligado | Risco de FP / não aprovado para baseline final |",
        "| Meta-Learner Shadow EXP-007A | Diagnóstico apenas | Não encontrou candidato seguro adicional |",
        "| Regra ampla `first_receiver_flag` | Proibida como regra isolada | Sinal aparece em FNs, mas também domina FPs adicionados |",
        "",
        "## 10. Regras desligadas no scoring_config",
        "",
        "| Campo | Valor | Interpretação |",
        "|---|---:|---|",
        f"| `se_pattern_residual_enabled` | `{cfg(config, 'se_pattern_residual_enabled')}` | Padrão residual SE desligado |",
        f"| `exp003_residual_confirm_enabled` | `{cfg(config, 'exp003_residual_confirm_enabled')}` | Residual EXP-003 desligado |",
        "",
        "## 11. Campos obrigatórios de decisão",
        "",
        "Todo resultado final de decisão deve possuir, no mínimo:",
        "",
        "```text",
        "transaction_id",
        "decision_id",
        "model_version",
        "scoring_config_version",
        "decisao",
        "score_final",
        "lgbm_raw",
        "lgbm_mapped",
        "if_percentile",
        "se_score",
        "beh_score",
        "rules_applied",
        "guardrails_applied",
        "decision_reason",
        "created_at",
        "```",
        "",
        "## 12. Procedimento de regressão obrigatório",
        "",
        "Antes de alterar qualquer regra, threshold, artefato ou lógica de decisão, executar:",
        "",
        "```powershell",
        "python -m pytest tests\\test_regression_post_fase2.py -q",
        "python -m pytest tests\\test_regression_post_fase2.py -q -m slow",
        "```",
        "",
        "## 13. Critério de manutenção",
        "",
        "O catálogo deve ser atualizado sempre que:",
        "",
        "- uma regra for promovida;",
        "- uma regra for desligada;",
        "- um threshold for alterado;",
        "- um candidato for rejeitado formalmente;",
        "- novos dados alterarem a decisão de promoção/rejeição;",
        "- o `DecisionEngine` mudar a composição do score.",
        "",
    ])

    return "\n".join(lines)


def generate_decision_trace_spec(config: dict[str, Any]) -> str:
    now = datetime.now().isoformat(timespec="seconds")

    lines = [
        "# Decision Trace Spec — Pipeline Antifraude PIX",
        "",
        f"**Gerado em:** `{now}`",
        "",
        "## 1. Objetivo",
        "",
        "Este documento define o formato mínimo de rastreabilidade de decisão do pipeline antifraude PIX.",
        "",
        "O objetivo é permitir auditoria, explicabilidade, regressão, análise de FP/FN, monitoramento de drift e reconstrução posterior da decisão.",
        "",
        "## 2. Princípio",
        "",
        "Toda decisão deve ser reconstruível a partir de:",
        "",
        "1. dados da transação;",
        "2. scores dos módulos;",
        "3. regras e guard rails aplicados;",
        "4. versão do modelo;",
        "5. versão do `scoring_config`; e",
        "6. motivo final textual.",
        "",
        "## 3. Schema mínimo",
        "",
        "| Campo | Tipo | Obrigatório | Descrição |",
        "|---|---|---:|---|",
        "| `decision_id` | string | Sim | Identificador único da decisão |",
        "| `transaction_id` | string | Sim | Identificador da transação |",
        "| `customer_id_hash` | string | Sim | Identificador anonimizado do cliente |",
        "| `created_at` | datetime | Sim | Timestamp da decisão |",
        "| `model_version` | string | Sim | Versão lógica do modelo/pipeline |",
        "| `decision_engine_version` | string | Sim | Versão do motor de decisão |",
        "| `scoring_config_version` | string | Sim | Versão/hash do scoring_config |",
        "| `decisao` | string | Sim | `APROVAR`, `CONFIRMAR` ou `BLOQUEAR` |",
        "| `score_final` | float | Sim | Score final após regras/exceções |",
        "| `score_final_original` | float | Não | Score antes de exceções, quando aplicável |",
        "| `lgbm_raw` | float | Sim | Score bruto LGBM |",
        "| `lgbm_mapped` | float | Sim | Score LGBM mapeado |",
        "| `if_percentile` | float | Sim | Percentil do Isolation Forest |",
        "| `se_score` | float | Sim | Score de engenharia social |",
        "| `beh_score` | float | Sim | Score comportamental |",
        "| `rules_applied` | list[string] | Sim | Regras que alteraram ou sustentaram decisão |",
        "| `guardrails_applied` | list[string] | Sim | Guard rails aplicados |",
        "| `veto_reason` | string | Não | Motivo de veto, se houver |",
        "| `veto_suppressed_reason` | string | Não | Motivo de veto suprimido, se houver |",
        "| `decision_reason` | string | Sim | Explicação textual da decisão final |",
        "| `review_recommended` | bool | Sim | Indica se deve ir para revisão humana |",
        "",
        "## 4. Campos específicos de regras promovidas",
        "",
        "| Campo | Tipo | Descrição |",
        "|---|---|---|",
        "| `v1_guard_contextual_applied` | bool | Indica acionamento da V1 Guard Contextual |",
        "| `v1_guard_contextual_reason` | string | Motivo do acionamento da V1 |",
        "| `exp006f_c1_applied` | bool | Indica acionamento da C1 |",
        "| `exp006f_c1_reason` | string | Motivo do acionamento da C1 |",
        "| `decisao_original_exp006f_c1` | string | Decisão antes da C1 |",
        "| `score_final_original_exp006f_c1` | float | Score antes da C1 |",
        "",
        "## 5. Motivos padronizados de decisão",
        "",
        "Sugestão de códigos controlados:",
        "",
        "```text",
        "BASE_SCORE_THRESHOLD_CONFIRMAR",
        "BASE_SCORE_THRESHOLD_BLOQUEAR",
        "LGBM_GUARD_RAIL_APPLIED",
        "V1_GUARD_CONTEXTUAL_APPLIED",
        "C1_NEAR_THRESHOLD_APPLIED",
        "SE_RULE_SIGNAL",
        "BEH_RULE_SIGNAL",
        "FAST_APPROVE",
        "APPROVE_LOW_RISK",
        "MANUAL_REVIEW_REQUIRED",
        "```",
        "",
        "## 6. Política de versionamento",
        "",
        "Toda decisão deve registrar:",
        "",
        "- versão do modelo ativo;",
        "- versão do motor de decisão;",
        "- versão ou hash do `scoring_config.json`; e",
        "- data/hora da decisão.",
        "",
        "Sem esses campos, a decisão não deve ser considerada plenamente auditável.",
        "",
        "## 7. Uso em monitoramento",
        "",
        "Os campos de trace serão usados para:",
        "",
        "- calcular taxa de C1;",
        "- calcular taxa de V1;",
        "- auditar FPs e FNs;",
        "- monitorar drift de scores;",
        "- construir fila de revisão humana;",
        "- explicar decisões para auditoria interna;",
        "- comparar versões futuras do modelo.",
        "",
    ]

    return "\n".join(lines)


def generate_decision_trace_example(config: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")

    return {
        "decision_id": "decision_example_exp006f_c1",
        "transaction_id": "E0000020820260205003505340630525",
        "customer_id_hash": "hash_4321433355",
        "created_at": now,
        "model_version": "post_fase2_c1",
        "decision_engine_version": "v3.0.5_post_c1",
        "scoring_config_version": "post_fase2_c1",
        "decisao": "CONFIRMAR",
        "score_final": 62.0,
        "score_final_original": 58.01,
        "lgbm_raw": 0.063164,
        "lgbm_mapped": None,
        "if_percentile": None,
        "se_score": 0.0,
        "beh_score": 0.0,
        "rules_applied": [
            "C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER"
        ],
        "guardrails_applied": [],
        "veto_reason": None,
        "veto_suppressed_reason": None,
        "decision_reason": (
            "C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER aplicou APROVAR->CONFIRMAR: "
            "rel<=12, first_receiver=1, pix_random=0, 100<=vl<500, "
            "0.06<=lgbm<0.10, 58<=score<62, SE=0, BEH=0."
        ),
        "review_recommended": True,
        "rule_details": {
            "exp006f_c1_applied": True,
            "exp006f_c1_reason": (
                "C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER: APROVAR->CONFIRMAR"
            ),
            "decisao_original_exp006f_c1": "APROVAR",
            "score_final_original_exp006f_c1": 58.01,
            "config": {
                "exp006f_c1_min_score": cfg(config, "exp006f_c1_min_score"),
                "exp006f_c1_max_score": cfg(config, "exp006f_c1_max_score"),
                "exp006f_c1_min_lgbm_raw": cfg(config, "exp006f_c1_min_lgbm_raw"),
                "exp006f_c1_max_lgbm_raw": cfg(config, "exp006f_c1_max_lgbm_raw"),
            },
        },
    }


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    config = read_json(SCORING_PATH)

    rules_catalog = generate_rules_catalog(config)
    decision_trace_spec = generate_decision_trace_spec(config)
    decision_trace_example = generate_decision_trace_example(config)

    RULES_CATALOG_PATH.write_text(rules_catalog, encoding="utf-8")
    DECISION_TRACE_SPEC_PATH.write_text(decision_trace_spec, encoding="utf-8")
    DECISION_TRACE_EXAMPLE_PATH.write_text(
        json.dumps(decision_trace_example, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] Rules catalog gerado: {RULES_CATALOG_PATH}")
    print(f"[OK] Decision trace spec gerado: {DECISION_TRACE_SPEC_PATH}")
    print(f"[OK] Decision trace example gerado: {DECISION_TRACE_EXAMPLE_PATH}")


if __name__ == "__main__":
    main()