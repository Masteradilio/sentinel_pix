"""
EXP-006B — Engine Counterfactual Audit

Objetivo:
  Auditar contrafactuais leves sobre os artefatos existentes, sem rodar E2E.

Este experimento:
  - Não chama PipelineOrquestrador.
  - Não treina modelo.
  - Não altera scoring_config.json.
  - Não troca artefatos.
  - Usa resultados do EXP-005B-E2E e EXP-006.
  - Estima se algum subconjunto de regra teria ganho líquido:
      FN recuperado - TP perdido
      FP adicionado - FP removido

Saídas:
  resultados/experimentos/EXP-006B/
    00_input_summary.json
    01_counterfactual_rules.csv
    02_veto_suppressed_audit.csv
    03_lgbm_threshold_audit.json
    04_recoverability_map.csv
    05_recomendacoes.md
    06_next_experiment_spec.md
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

EXP_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists() and (p / "resultados").exists():
            return p
    return start.parent.parent


PROJECT_ROOT = find_project_root(EXP_DIR)

EXP005B_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-005B-E2E"
EXP006_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-006"
OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-006B"


# =========================================================
# BASELINE
# =========================================================

BASELINE_FASE2 = {
    "TP": 346,
    "FP": 15,
    "FN": 9,
    "Precision": 0.958449,
    "Recall": 0.974648,
    "F1": 0.9665,
    "FPR": 0.002657,
}


# =========================================================
# HELPERS
# =========================================================

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def has_text_value(x: Any) -> bool:
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass
    s = str(x).strip().lower()
    return s not in {"", "nan", "none", "null", "<na>"}


def safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [safe_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [safe_json(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe_json(obj), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def value_bin(v: float) -> str:
    if v < 100:
        return "VL_000_099"
    if v < 500:
        return "VL_100_499"
    if v < 1000:
        return "VL_500_999"
    if v < 5000:
        return "VL_1K_5K"
    if v < 15000:
        return "VL_5K_15K"
    return "VL_15K_PLUS"


def relationship_bin(m: float) -> str:
    if m <= 6:
        return "REL_0_6M"
    if m <= 12:
        return "REL_7_12M"
    if m <= 36:
        return "REL_1_3Y"
    if m <= 120:
        return "REL_3_10Y"
    return "REL_10Y_PLUS"


def lgbm_bin(x: float) -> str:
    if x >= 0.30:
        return "LGBM_300_PLUS"
    if x >= 0.20:
        return "LGBM_200_300"
    if x >= 0.10:
        return "LGBM_100_200"
    if x >= 0.05:
        return "LGBM_050_100"
    if x >= 0.01:
        return "LGBM_010_050"
    return "LGBM_LT_010"


def if_bin(x: float) -> str:
    if x >= 0.995:
        return "IF_995_PLUS"
    if x >= 0.985:
        return "IF_985_995"
    if x >= 0.95:
        return "IF_950_985"
    if x >= 0.80:
        return "IF_800_950"
    return "IF_LT_800"


def normalize_cases_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    numeric_cols = [
        "vl_pix",
        "nr_idade",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "lgbm_raw",
        "lgbm_mapped",
        "if_percentile",
        "se_score",
        "beh_score",
        "score_final",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["movement_type", "category", "transaction_id", "customer_id", "seed", "config_id"]:
        if c not in df.columns:
            df[c] = ""

    # Deduplicação por seed + tipo + transação. Mantém duplicação entre seeds porque é evidência de estabilidade.
    keys = ["seed", "movement_type", "transaction_id"]
    df = df.drop_duplicates(subset=[k for k in keys if k in df.columns]).copy()

    df["vl_bin"] = df["vl_pix"].apply(lambda x: value_bin(safe_float(x)))
    df["rel_bin"] = df["qt_tempo_relacionamento_mes"].apply(lambda x: relationship_bin(safe_float(x)))
    df["lgbm_bin"] = df["lgbm_raw"].apply(lambda x: lgbm_bin(safe_float(x)))
    df["if_bin"] = df["if_percentile"].apply(lambda x: if_bin(safe_float(x)))
    df["has_veto_suppressed"] = df["veto_suppressed_reason"].apply(has_text_value) if "veto_suppressed_reason" in df.columns else False

    return df


# =========================================================
# LOAD INPUTS
# =========================================================

def load_cases() -> pd.DataFrame:
    cases_path = EXP006_DIR / "02_casos_movimentados.csv"

    if cases_path.exists():
        return normalize_cases_schema(pd.read_csv(cases_path))

    raise FileNotFoundError(f"Não encontrei {cases_path}. Rode o EXP-006 primeiro.")


def load_grid() -> pd.DataFrame:
    path = EXP005B_DIR / "01_e2e_grid_decision_engine.csv"

    if not path.exists():
        raise FileNotFoundError(f"Não encontrei {path}")

    return pd.read_csv(path)


def load_hypotheses() -> dict[str, Any]:
    path = EXP006_DIR / "04_hipoteses_cirurgicas.json"

    if not path.exists():
        return {}

    return read_json(path)


# =========================================================
# COUNTERFACTUAL RULES
# =========================================================

RuleFn = Callable[[pd.DataFrame], pd.Series]


def rule_all_moved_cases(df: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=df.index)


def rule_lgbm_gray_first_receiver_strict(df: pd.DataFrame) -> pd.Series:
    return (
        df["first_receiver_flag"].fillna(0).astype(int).eq(1)
        & df["lgbm_raw"].between(0.08, 0.20, inclusive="left")
        & df["if_percentile"].ge(0.90)
        & df["se_score"].fillna(0).le(0)
        & df["beh_score"].fillna(0).le(0)
    )


def rule_lgbm_gray_first_receiver_low_value(df: pd.DataFrame) -> pd.Series:
    return (
        df["first_receiver_flag"].fillna(0).astype(int).eq(1)
        & df["vl_pix"].lt(500)
        & df["lgbm_raw"].between(0.08, 0.20, inclusive="left")
        & df["se_score"].fillna(0).le(0)
        & df["beh_score"].fillna(0).le(0)
    )


def rule_first_receiver_if_extreme(df: pd.DataFrame) -> pd.Series:
    return (
        df["first_receiver_flag"].fillna(0).astype(int).eq(1)
        & df["if_percentile"].ge(0.985)
        & df["lgbm_raw"].ge(0.05)
    )


def rule_first_receiver_se_beh_present(df: pd.DataFrame) -> pd.Series:
    return (
        df["first_receiver_flag"].fillna(0).astype(int).eq(1)
        & (
            df["se_score"].fillna(0).ge(40)
            | df["beh_score"].fillna(0).ge(15)
        )
        & df["lgbm_raw"].ge(0.05)
    )


def rule_guard_exception_alto_valor(df: pd.DataFrame) -> pd.Series:
    return (
        df["vl_pix"].ge(15000)
        & df["qt_tempo_relacionamento_mes"].le(12)
        & df["first_receiver_flag"].fillna(0).astype(int).eq(1)
        & df["if_percentile"].ge(0.985)
        & df["se_score"].fillna(0).ge(40)
        & df["beh_score"].fillna(0).ge(15)
        & df["lgbm_raw"].ge(0.01)
        & df["lgbm_raw"].lt(0.30)
    )


def rule_veto_suppressed_cases(df: pd.DataFrame) -> pd.Series:
    return df["has_veto_suppressed"].astype(bool)


def rule_lgbm_effective_zone_only(df: pd.DataFrame) -> pd.Series:
    return (
        df["lgbm_raw"].ge(0.05)
        & df["lgbm_raw"].lt(0.20)
        & df["first_receiver_flag"].fillna(0).astype(int).eq(1)
    )


RULES: list[dict[str, Any]] = [
    {
        "rule_id": "R0_ALL_MOVED_CASES",
        "description": "Todos os casos movimentados pelo candidato. Serve como controle.",
        "fn": rule_all_moved_cases,
    },
    {
        "rule_id": "R1_LGBM_GRAY_FIRST_RECEIVER_STRICT",
        "description": "first_receiver + LGBM 0.08-0.20 + IF>=0.90 + SE/BEH zerados.",
        "fn": rule_lgbm_gray_first_receiver_strict,
    },
    {
        "rule_id": "R2_LOW_VALUE_GRAY_FIRST_RECEIVER",
        "description": "first_receiver + valor<500 + LGBM 0.08-0.20 + SE/BEH zerados.",
        "fn": rule_lgbm_gray_first_receiver_low_value,
    },
    {
        "rule_id": "R3_FIRST_RECEIVER_IF_EXTREME",
        "description": "first_receiver + IF>=0.985 + LGBM>=0.05.",
        "fn": rule_first_receiver_if_extreme,
    },
    {
        "rule_id": "R4_FIRST_RECEIVER_SE_OR_BEH",
        "description": "first_receiver + SE>=40 ou BEH>=15 + LGBM>=0.05.",
        "fn": rule_first_receiver_se_beh_present,
    },
    {
        "rule_id": "R5_GUARD_EXCEPTION_ALTO_VALOR",
        "description": "Regra cirúrgica já aprovada no EXP-004-FINAL.",
        "fn": rule_guard_exception_alto_valor,
    },
    {
        "rule_id": "R6_VETO_SUPPRESSED_ONLY",
        "description": "Apenas casos com veto_suppressed_reason real.",
        "fn": rule_veto_suppressed_cases,
    },
    {
        "rule_id": "R7_LGBM_EFFECTIVE_ZONE_FIRST_RECEIVER",
        "description": "Zona efetiva LGBM 0.05-0.20 com first_receiver.",
        "fn": rule_lgbm_effective_zone_only,
    },
]


def evaluate_rule(df: pd.DataFrame, rule: dict[str, Any]) -> dict[str, Any]:
    mask = rule["fn"](df).fillna(False).astype(bool)
    sub = df.loc[mask].copy()

    counts = sub["movement_type"].value_counts().to_dict()

    fn_recovered = int(counts.get("FN_RECUPERADO", 0))
    tp_lost = int(counts.get("TP_PERDIDO", 0))
    fp_added = int(counts.get("FP_ADICIONADO", 0))
    fp_removed = int(counts.get("FP_REMOVIDO", 0))

    net_tp_gain = fn_recovered - tp_lost
    net_fp_cost = fp_added - fp_removed

    # Critério conservador para merecer quick-e2e:
    # precisa ter ganho líquido de fraude e FP estimado pequeno.
    if net_tp_gain > 0 and net_fp_cost <= 3:
        recommendation = "CANDIDATO_QUICK_E2E"
    elif net_tp_gain > 0 and net_fp_cost <= 6:
        recommendation = "INVESTIGAR_MANUALMENTE"
    elif fn_recovered > 0:
        recommendation = "REJEITAR_FP_OU_TP_PERDIDO"
    else:
        recommendation = "SEM_GANHO"

    return {
        "rule_id": rule["rule_id"],
        "description": rule["description"],
        "matched_cases": int(len(sub)),
        "fn_recovered": fn_recovered,
        "tp_lost": tp_lost,
        "fp_added": fp_added,
        "fp_removed": fp_removed,
        "net_tp_gain_est": int(net_tp_gain),
        "net_fp_cost_est": int(net_fp_cost),
        "recommendation": recommendation,
        "matched_transaction_ids": ", ".join(str(x) for x in sub["transaction_id"].dropna().astype(str).unique()[:20]),
    }


def evaluate_rules(df: pd.DataFrame) -> pd.DataFrame:
    rows = [evaluate_rule(df, rule) for rule in RULES]
    out = pd.DataFrame(rows)

    return out.sort_values(
        ["recommendation", "net_tp_gain_est", "net_fp_cost_est", "fn_recovered"],
        ascending=[True, False, True, False],
    )


# =========================================================
# AUDITS
# =========================================================

def audit_veto_suppressed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    sub = df[df["has_veto_suppressed"].astype(bool)].copy()

    if sub.empty:
        return pd.DataFrame(columns=[
            "movement_type",
            "category",
            "transaction_id",
            "customer_id",
            "vl_pix",
            "lgbm_raw",
            "if_percentile",
            "se_score",
            "beh_score",
            "score_final",
            "decisao",
            "veto_suppressed_reason",
            "diagnosis",
        ])

    diagnoses = []

    for _, r in sub.iterrows():
        decision = str(r.get("decisao", "")).upper()
        reason = str(r.get("veto_suppressed_reason", ""))

        if "LGBM_GUARD_RAIL" in reason and decision == "CONFIRMAR":
            diag = "VETO_SUPPRESSED_BUT_CONFIRMOU"
        elif "LGBM_GUARD_RAIL" in reason and decision == "APROVAR":
            diag = "VETO_EFFECTIVE_APPROVE"
        else:
            diag = "VETO_SUPPRESSED_OTHER"

        diagnoses.append(diag)

    sub["diagnosis"] = diagnoses

    cols = [
        "movement_type",
        "category",
        "transaction_id",
        "customer_id",
        "vl_pix",
        "lgbm_raw",
        "if_percentile",
        "se_score",
        "beh_score",
        "score_final",
        "decisao",
        "veto_suppressed_reason",
        "diagnosis",
        "seed",
    ]

    return sub[[c for c in cols if c in sub.columns]].copy()


def audit_lgbm_threshold_effect(grid: pd.DataFrame) -> dict[str, Any]:
    candidates = grid[grid["config_id"].ne("BASELINE")].copy()

    metrics_cols = ["TP", "FP", "FN", "Precision", "Recall", "F1", "FPR"]

    by_seed: dict[str, Any] = {}

    for seed, g in candidates.groupby("seed"):
        signatures = defaultdict(list)

        for _, r in g.iterrows():
            sig = tuple(
                round(safe_float(r[c]), 8) if c not in {"TP", "FP", "FN"} else safe_int(r[c])
                for c in metrics_cols
                if c in r
            )
            signatures[str(sig)].append(str(r["config_id"]))

        by_seed[str(seed)] = {
            "unique_metric_signatures": len(signatures),
            "signatures": [
                {
                    "metrics_signature": sig,
                    "configs": cfgs,
                }
                for sig, cfgs in signatures.items()
            ],
            "all_candidates_identical": len(signatures) == 1,
        }

    all_identical = bool(by_seed) and all(v["all_candidates_identical"] for v in by_seed.values())

    if all_identical:
        diagnosis = (
            "As configs candidatas tiveram métricas idênticas por seed. "
            "Isso sugere que lgbm_effective_threshold/guard_threshold não diferenciou a decisão final, "
            "ou que a lógica do engine sobrepôs esses parâmetros."
        )
    else:
        diagnosis = (
            "Há diferença entre configs; os thresholds parecem influenciar ao menos parte do pipeline."
        )

    return {
        "all_candidates_identical_per_seed": all_identical,
        "by_seed": by_seed,
        "diagnosis": diagnosis,
        "next_action": (
            "Auditar caminho lgbm_raw -> lgbm_mapped -> score_final -> veto -> decisao "
            "antes de qualquer novo threshold sweep."
        ),
    }


def build_recoverability_map(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    rows = []

    for tx_id, g in df.groupby("transaction_id", dropna=False):
        movement_types = sorted(g["movement_type"].dropna().astype(str).unique().tolist())
        first_row = g.iloc[0].to_dict()

        fn_count = int((g["movement_type"] == "FN_RECUPERADO").sum())
        tp_lost_count = int((g["movement_type"] == "TP_PERDIDO").sum())
        fp_added_count = int((g["movement_type"] == "FP_ADICIONADO").sum())
        fp_removed_count = int((g["movement_type"] == "FP_REMOVIDO").sum())

        lgbm = safe_float(first_row.get("lgbm_raw"))
        ifp = safe_float(first_row.get("if_percentile"))
        se = safe_float(first_row.get("se_score"))
        beh = safe_float(first_row.get("beh_score"))
        first = safe_int(first_row.get("first_receiver_flag"))
        vl = safe_float(first_row.get("vl_pix"))

        if fn_count > 0 and fp_added_count == 0 and tp_lost_count == 0:
            rec_class = "RECOVERABLE_LOW_COST"
        elif fn_count > 0 and (fp_added_count > 0 or tp_lost_count > 0):
            rec_class = "RECOVERABLE_BUT_COSTLY"
        elif tp_lost_count > 0:
            rec_class = "BASELINE_FRAGILE_TP"
        elif fp_added_count > 0:
            rec_class = "FP_RISK_PATTERN"
        elif fp_removed_count > 0:
            rec_class = "FP_REDUCIBLE"
        else:
            rec_class = "UNKNOWN"

        if (
            lgbm < 0.05
            and ifp < 0.80
            and se <= 0
            and beh <= 0
        ):
            signal_class = "WEAK_ALL_MODULES"
        elif lgbm >= 0.05 and first == 1:
            signal_class = "LGBM_FIRST_RECEIVER_SIGNAL"
        elif ifp >= 0.95:
            signal_class = "IF_SIGNAL"
        elif se >= 40 or beh >= 15:
            signal_class = "RULE_SIGNAL"
        else:
            signal_class = "MIXED_SIGNAL"

        rows.append({
            "transaction_id": tx_id,
            "customer_id": first_row.get("customer_id"),
            "movement_types": "|".join(movement_types),
            "recoverability_class": rec_class,
            "signal_class": signal_class,
            "vl_pix": vl,
            "first_receiver_flag": first,
            "pix_key_random_flag": safe_int(first_row.get("pix_key_random_flag")),
            "lgbm_raw": lgbm,
            "if_percentile": ifp,
            "se_score": se,
            "beh_score": beh,
            "score_final": safe_float(first_row.get("score_final")),
            "veto_suppressed_reason": first_row.get("veto_suppressed_reason"),
            "fn_recovered_count": fn_count,
            "tp_lost_count": tp_lost_count,
            "fp_added_count": fp_added_count,
            "fp_removed_count": fp_removed_count,
        })

    out = pd.DataFrame(rows)

    return out.sort_values(
        ["recoverability_class", "signal_class", "vl_pix"],
        ascending=[True, True, False],
    )


# =========================================================
# REPORTS
# =========================================================

def write_recommendations(
    path: Path,
    rules_df: pd.DataFrame,
    veto_df: pd.DataFrame,
    threshold_audit: dict[str, Any],
    recoverability_df: pd.DataFrame,
) -> None:
    candidates = rules_df[rules_df["recommendation"].eq("CANDIDATO_QUICK_E2E")].copy()
    investigate = rules_df[rules_df["recommendation"].eq("INVESTIGAR_MANUALMENTE")].copy()

    lines = [
        "# EXP-006B — Engine Counterfactual Audit",
        "",
        f"Gerado em: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Conclusão executiva",
        "",
    ]

    if candidates.empty:
        lines.extend([
            "Nenhuma regra contrafactual teve ganho líquido suficiente para justificar E2E imediato.",
            "Isso reforça que a fronteira atual está difícil: os mesmos sinais que recuperam FN também trazem FP ou perdem TP.",
        ])
    else:
        lines.extend([
            "Há pelo menos uma regra com ganho líquido estimado para `quick-e2e`.",
            "Mesmo assim, a regra deve passar pelo protocolo rápido antes de qualquer promoção.",
        ])

    lines.extend([
        "",
        "## Auditoria de thresholds LGBM",
        "",
        threshold_audit["diagnosis"],
        "",
        f"- Todas as configs candidatas idênticas por seed: `{threshold_audit['all_candidates_identical_per_seed']}`",
        "",
        "## Regras contrafactuais avaliadas",
        "",
        "| Regra | FN rec. | TP perdido | FP add. | FP rem. | Net TP | Net FP | Recomendação |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])

    for _, r in rules_df.iterrows():
        lines.append(
            f"| `{r['rule_id']}` | {int(r['fn_recovered'])} | {int(r['tp_lost'])} | "
            f"{int(r['fp_added'])} | {int(r['fp_removed'])} | "
            f"{int(r['net_tp_gain_est'])} | {int(r['net_fp_cost_est'])} | `{r['recommendation']}` |"
        )

    lines.extend([
        "",
        "## Veto suppressed audit",
        "",
    ])

    if veto_df.empty:
        lines.append("Nenhum caso com `veto_suppressed_reason` real nos casos movimentados.")
    else:
        counts = veto_df["diagnosis"].value_counts().to_dict()
        for k, v in counts.items():
            lines.append(f"- `{k}`: {v}")

    lines.extend([
        "",
        "## Recoverability map",
        "",
    ])

    if recoverability_df.empty:
        lines.append("Nenhum caso mapeado.")
    else:
        counts = recoverability_df["recoverability_class"].value_counts().to_dict()
        for k, v in counts.items():
            lines.append(f"- `{k}`: {v}")

    lines.extend([
        "",
        "## Decisão",
        "",
    ])

    if candidates.empty:
        lines.extend([
            "Não rodar E2E agora.",
            "Antes, é necessário obter a lista completa dos 9 FNs residuais do baseline com todos os sinais, porque os artefatos atuais mostram apenas casos movimentados pelo candidato.",
        ])
    else:
        lines.extend([
            "Rodar `quick-e2e` apenas para a melhor regra candidata, com baseline + 1 candidato, sample 1000, seed 42.",
            "Se FN não cair ou FP subir acima de baseline +3, interromper.",
        ])

    lines.extend([
        "",
        "## Próximo passo recomendado",
        "",
        "`EXP-006C — Baseline Residual FN Census`",
        "",
        "Objetivo: gerar uma tabela completa dos 9 FNs residuais do baseline, com LGBM, IF, SE, BEH, score final, veto e features-chave.",
        "Sem essa tabela, estamos inferindo a irredutibilidade apenas pelos casos movimentados, não pela fronteira completa.",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def write_next_experiment_spec(path: Path, rules_df: pd.DataFrame) -> None:
    candidates = rules_df[rules_df["recommendation"].eq("CANDIDATO_QUICK_E2E")].copy()

    if candidates.empty:
        recommended = "EXP-006C — Baseline Residual FN Census"
        objective = (
            "Gerar, de forma rápida e incremental, a lista completa dos 9 FNs residuais do baseline "
            "com todos os sinais dos módulos, para separar casos recuperáveis de DATA_LIMITED."
        )
        command = (
            "A criar: script que roda apenas baseline, salva predições detalhadas e extrai FN/FP, "
            "preferencialmente com cache incremental."
        )
    else:
        best = candidates.sort_values(
            ["net_tp_gain_est", "net_fp_cost_est"],
            ascending=[False, True],
        ).iloc[0]
        recommended = f"Quick-E2E para {best['rule_id']}"
        objective = (
            "Validar em sample pequeno se a regra contrafactual realmente reduz FN "
            "sem aumentar FP acima do limite."
        )
        command = (
            "Rodar baseline + 1 candidato, sample=1000, seed=42, workers=4, salvamento incremental."
        )

    lines = [
        "# EXP-006B — Próximo Experimento",
        "",
        f"## Recomendado: {recommended}",
        "",
        "## Objetivo",
        "",
        objective,
        "",
        "## Protocolo obrigatório",
        "",
        "1. Não rodar grid E2E.",
        "2. Não rodar múltiplos candidatos.",
        "3. Salvar resultado imediatamente após baseline.",
        "4. Salvar resultado imediatamente após candidato.",
        "5. Interromper se FN não cair ou FP subir acima de baseline +3.",
        "",
        "## Comando / ação",
        "",
        command,
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def write_input_summary(path: Path, cases: pd.DataFrame, grid: pd.DataFrame, hypotheses: dict[str, Any]) -> None:
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "exp005b_grid": str(EXP005B_DIR / "01_e2e_grid_decision_engine.csv"),
            "exp006_cases": str(EXP006_DIR / "02_casos_movimentados.csv"),
            "exp006_hypotheses": str(EXP006_DIR / "04_hipoteses_cirurgicas.json"),
        },
        "case_counts": cases["movement_type"].value_counts().to_dict() if not cases.empty else {},
        "grid_configs": sorted(grid["config_id"].dropna().astype(str).unique().tolist()) if not grid.empty else [],
        "hypothesis_ids": [h.get("id") for h in hypotheses.get("hypotheses", [])],
    }

    write_json(path, summary)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EXP-006B — Engine Counterfactual Audit")
    print("=" * 72)

    print("[1/6] Carregando artefatos...")
    cases = load_cases()
    grid = load_grid()
    hypotheses = load_hypotheses()

    write_input_summary(OUTPUT_DIR / "00_input_summary.json", cases, grid, hypotheses)

    print(f"[OK] Casos movimentados: {len(cases)}")
    print(f"[OK] Configs no grid: {sorted(grid['config_id'].unique().tolist())}")

    print("[2/6] Avaliando regras contrafactuais...")
    rules_df = evaluate_rules(cases)
    rules_path = OUTPUT_DIR / "01_counterfactual_rules.csv"
    rules_df.to_csv(rules_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Regras avaliadas: {len(rules_df)} -> {rules_path}")

    print("[3/6] Auditando veto_suppressed_reason...")
    veto_df = audit_veto_suppressed(cases)
    veto_path = OUTPUT_DIR / "02_veto_suppressed_audit.csv"
    veto_df.to_csv(veto_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Casos com veto_suppressed: {len(veto_df)} -> {veto_path}")

    print("[4/6] Auditando efeito dos thresholds LGBM...")
    threshold_audit = audit_lgbm_threshold_effect(grid)
    write_json(OUTPUT_DIR / "03_lgbm_threshold_audit.json", threshold_audit)
    print(f"[OK] Threshold audit: all_identical={threshold_audit['all_candidates_identical_per_seed']}")

    print("[5/6] Gerando mapa de recoverability...")
    recoverability = build_recoverability_map(cases)
    rec_path = OUTPUT_DIR / "04_recoverability_map.csv"
    recoverability.to_csv(rec_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Recoverability map: {len(recoverability)} -> {rec_path}")

    print("[6/6] Escrevendo recomendações...")
    write_recommendations(
        OUTPUT_DIR / "05_recomendacoes.md",
        rules_df=rules_df,
        veto_df=veto_df,
        threshold_audit=threshold_audit,
        recoverability_df=recoverability,
    )

    write_next_experiment_spec(
        OUTPUT_DIR / "06_next_experiment_spec.md",
        rules_df=rules_df,
    )

    print()
    print("[OK] EXP-006B concluído sem E2E.")
    print(f"[OK] Artefatos em: {OUTPUT_DIR}")
    print()
    print("Arquivos principais:")
    print(f"  {OUTPUT_DIR / '01_counterfactual_rules.csv'}")
    print(f"  {OUTPUT_DIR / '03_lgbm_threshold_audit.json'}")
    print(f"  {OUTPUT_DIR / '05_recomendacoes.md'}")
    print(f"  {OUTPUT_DIR / '06_next_experiment_spec.md'}")


if __name__ == "__main__":
    main()