"""
EXP-009D — Painel de Metricas Operacionais

Objetivo:
  Criar bases offline para Power BI/dashboard operacional do baseline post_fase2_c1.

Este experimento:
  - Nao altera o modelo.
  - Nao altera scoring_config.json.
  - Nao altera DecisionEngine.
  - Nao roda E2E.
  - Usa logs estruturados do EXP-009A.
  - Usa a fila de revisao humana do EXP-009C.
  - Gera CSVs prontos para Power BI.

Entradas default:
  resultados/experimentos/EXP-009A/03_decision_log_all.jsonl
  resultados/experimentos/EXP-009C/02_review_queue_dedup.csv

Saidas:
  resultados/experimentos/EXP-009D/
    00_input_summary.json
    01_kpi_overall.csv
    02_decision_distribution.csv
    03_rule_metrics.csv
    04_score_bands.csv
    05_daily_metrics.csv
    06_review_queue_metrics.csv
    07_powerbi_decision_fact.csv
    08_powerbi_review_queue_fact.csv
    09_dashboard_readme.md
    10_next_experiment_spec.md

Uso:
  python experimentos\\exp_009d_operational_dashboard\\run_exp_009d_operational_dashboard.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

EXP_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists() and (p / "resultados").exists():
            return p
    return start.parent.parent


ROOT = find_project_root(EXP_DIR)

DEFAULT_DECISION_LOG = ROOT / "resultados" / "experimentos" / "EXP-009A" / "03_decision_log_all.jsonl"
DEFAULT_REVIEW_QUEUE = ROOT / "resultados" / "experimentos" / "EXP-009C" / "02_review_queue_dedup.csv"

OUTPUT_DIR = ROOT / "resultados" / "experimentos" / "EXP-009D"

MANIFEST_PATH = ROOT / "backend" / "artefatos" / "MANIFEST_MODEL.json"
JOURNAL_PATH = ROOT / "docs" / "JOURNAL.md"


POSITIVE_DECISIONS = {"CONFIRMAR", "BLOQUEAR"}
ALL_DECISIONS = ["APROVAR", "CONFIRMAR", "BLOQUEAR"]


# ============================================================
# IO
# ============================================================

def safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(x) for x in obj]
    if isinstance(obj, tuple):
        return [safe_json(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe_json(obj), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict) and "records" in data:
            return pd.DataFrame(data["records"])
        raise ValueError(f"JSON nao reconhecido como lista de registros: {path}")

    return pd.read_csv(path)


def parse_json_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x) for x in value]

    raw = str(value).strip()

    if not raw or raw.lower() in {"nan", "none", "null"}:
        return []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return [str(parsed)]
    except Exception:
        if "|" in raw:
            return [x.strip() for x in raw.split("|") if x.strip()]
        return [raw]


# ============================================================
# Normalizacao
# ============================================================

def normalize_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "sim", "s"})


def extract_transaction_date(tx_id: Any) -> str:
    """
    Extrai a data do E2E ID PIX no formato:
      E + ISPB(8 dígitos) + YYYYMMDD + restante

    Exemplo:
      E0000020820260203165904819156525
        ISPB: 00000208
        Data: 2026-02-03
    """
    raw = str(tx_id).strip()

    m = re.match(r"^E\d{8}(\d{8})", raw)

    if not m:
        return ""

    value = m.group(1)

    try:
        dt = datetime.strptime(value, "%Y%m%d")
    except Exception:
        return ""

    # Guarda conservadora para este dataset/harness.
    if dt.year < 2025 or dt.year > 2026:
        return ""

    return dt.date().isoformat()

def normalize_decision_log(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = [
        "seed",
        "score_final",
        "score_final_original",
        "lgbm_raw",
        "lgbm_mapped",
        "if_percentile",
        "se_score",
        "beh_score",
        "is_fraud",
        "vl_pix",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "score_final_original_exp006f_c1",
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["decisao", "model_version", "decision_engine_version", "scoring_config_version"]:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)

    if "decisao" in out.columns:
        out["decisao"] = out["decisao"].str.upper()

    if "review_recommended" in out.columns:
        out["review_recommended"] = normalize_bool_series(out["review_recommended"])
    else:
        out["review_recommended"] = out["decisao"].isin(POSITIVE_DECISIONS)

    if "exp006f_c1_applied" in out.columns:
        out["exp006f_c1_applied"] = normalize_bool_series(out["exp006f_c1_applied"])
    else:
        out["exp006f_c1_applied"] = False

    if "rules_applied" in out.columns:
        out["rules_applied_list"] = out["rules_applied"].apply(parse_json_list)
    else:
        out["rules_applied_list"] = [[] for _ in range(len(out))]

    if "guardrails_applied" in out.columns:
        out["guardrails_applied_list"] = out["guardrails_applied"].apply(parse_json_list)
    else:
        out["guardrails_applied_list"] = [[] for _ in range(len(out))]

    if "transaction_id" in out.columns:
        out["transaction_date"] = out["transaction_id"].apply(extract_transaction_date)
    else:
        out["transaction_date"] = ""

    out["transaction_date"] = pd.to_datetime(out["transaction_date"], errors="coerce")
    out["transaction_month"] = out["transaction_date"].dt.to_period("M").astype(str)
    out["transaction_day"] = out["transaction_date"].dt.date.astype(str)

    if "seed" not in out.columns:
        out["seed"] = -1

    out["seed_label"] = out["seed"].fillna(-1).astype(int).apply(lambda x: f"seed_{x}" if x >= 0 else "seed_unknown")
    out["is_positive_decision"] = out["decisao"].isin(POSITIVE_DECISIONS).astype(int)
    out["is_confirmar"] = out["decisao"].eq("CONFIRMAR").astype(int)
    out["is_bloquear"] = out["decisao"].eq("BLOQUEAR").astype(int)
    out["is_aprovar"] = out["decisao"].eq("APROVAR").astype(int)

    if "is_fraud" in out.columns:
        out["is_fraud"] = out["is_fraud"].fillna(0).astype(int)
    else:
        out["is_fraud"] = 0

    out["tp_flag"] = ((out["is_fraud"] == 1) & (out["is_positive_decision"] == 1)).astype(int)
    out["fp_flag"] = ((out["is_fraud"] == 0) & (out["is_positive_decision"] == 1)).astype(int)
    out["fn_flag"] = ((out["is_fraud"] == 1) & (out["is_positive_decision"] == 0)).astype(int)
    out["tn_flag"] = ((out["is_fraud"] == 0) & (out["is_positive_decision"] == 0)).astype(int)

    # Score bands para dashboard.
    score_bins = [-np.inf, 20, 40, 55, 58, 60, 62, 80, 95, np.inf]
    score_labels = [
        "<20",
        "20-40",
        "40-55",
        "55-58",
        "58-60",
        "60-62",
        "62-80",
        "80-95",
        ">=95",
    ]

    out["score_band"] = pd.cut(
        out["score_final"],
        bins=score_bins,
        labels=score_labels,
        include_lowest=True,
    ).astype(str)

    # LGBM bands.
    lgbm_bins = [-np.inf, 0.01, 0.03, 0.06, 0.10, 0.20, 0.30, 0.50, np.inf]
    lgbm_labels = [
        "<0.01",
        "0.01-0.03",
        "0.03-0.06",
        "0.06-0.10",
        "0.10-0.20",
        "0.20-0.30",
        "0.30-0.50",
        ">=0.50",
    ]

    out["lgbm_band"] = pd.cut(
        out["lgbm_raw"],
        bins=lgbm_bins,
        labels=lgbm_labels,
        include_lowest=True,
    ).astype(str)

    # IF bands.
    if_bins = [-np.inf, 0.50, 0.80, 0.90, 0.95, 0.985, np.inf]
    if_labels = [
        "<0.50",
        "0.50-0.80",
        "0.80-0.90",
        "0.90-0.95",
        "0.95-0.985",
        ">=0.985",
    ]

    out["if_band"] = pd.cut(
        out["if_percentile"],
        bins=if_bins,
        labels=if_labels,
        include_lowest=True,
    ).astype(str)

    return out


def normalize_review_queue(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = [
        "queue_rank_dedup",
        "priority_score_max",
        "priority_score_mean",
        "priority_score",
        "score_final",
        "score_gap_to_confirmar",
        "lgbm_raw",
        "lgbm_mapped",
        "if_percentile",
        "se_score",
        "beh_score",
        "vl_pix",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "is_fraud_any",
        "n_seed_occurrences",
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["priority_band", "primary_reason", "transaction_id", "customer_id_hash"]:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)

    if "transaction_id" in out.columns:
        out["transaction_date"] = out["transaction_id"].apply(extract_transaction_date)
    else:
        out["transaction_date"] = ""

    out["transaction_date"] = pd.to_datetime(out["transaction_date"], errors="coerce")
    out["transaction_month"] = out["transaction_date"].dt.to_period("M").astype(str)
    out["transaction_day"] = out["transaction_date"].dt.date.astype(str)

    return out


# ============================================================
# Metricas
# ============================================================

def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def metrics_for_df(df: pd.DataFrame, scope: str, label: str) -> dict[str, Any]:
    n = int(len(df))

    tp = int(df["tp_flag"].sum()) if "tp_flag" in df.columns else 0
    fp = int(df["fp_flag"].sum()) if "fp_flag" in df.columns else 0
    fn = int(df["fn_flag"].sum()) if "fn_flag" in df.columns else 0
    tn = int(df["tn_flag"].sum()) if "tn_flag" in df.columns else 0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    fpr = safe_div(fp, fp + tn)

    return {
        "scope": scope,
        "label": label,
        "n_decisions": n,
        "n_unique_transactions": int(df["transaction_id"].nunique()) if "transaction_id" in df.columns else n,
        "n_frauds": int(df["is_fraud"].sum()) if "is_fraud" in df.columns else 0,
        "n_aprovar": int(df["is_aprovar"].sum()) if "is_aprovar" in df.columns else 0,
        "n_confirmar": int(df["is_confirmar"].sum()) if "is_confirmar" in df.columns else 0,
        "n_bloquear": int(df["is_bloquear"].sum()) if "is_bloquear" in df.columns else 0,
        "n_positive_decisions": int(df["is_positive_decision"].sum()) if "is_positive_decision" in df.columns else 0,
        "n_review_recommended": int(df["review_recommended"].sum()) if "review_recommended" in df.columns else 0,
        "n_c1_applied": int(df["exp006f_c1_applied"].sum()) if "exp006f_c1_applied" in df.columns else 0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "fpr": round(fpr, 6),
        "total_value": round(float(df["vl_pix"].sum()), 2) if "vl_pix" in df.columns else 0.0,
        "avg_value": round(float(df["vl_pix"].mean()), 2) if "vl_pix" in df.columns and len(df) else 0.0,
        "avg_score_final": round(float(df["score_final"].mean()), 4) if "score_final" in df.columns and len(df) else 0.0,
        "avg_lgbm_raw": round(float(df["lgbm_raw"].mean()), 8) if "lgbm_raw" in df.columns and len(df) else 0.0,
        "avg_if_percentile": round(float(df["if_percentile"].mean()), 8) if "if_percentile" in df.columns and len(df) else 0.0,
    }


def build_kpi_overall(log_df: pd.DataFrame) -> pd.DataFrame:
    rows = [metrics_for_df(log_df, "overall", "all")]

    for seed, g in log_df.groupby("seed_label", dropna=False):
        rows.append(metrics_for_df(g, "seed", str(seed)))

    for month, g in log_df.groupby("transaction_month", dropna=False):
        if str(month) and str(month) != "NaT":
            rows.append(metrics_for_df(g, "month", str(month)))

    return pd.DataFrame(rows)


def build_decision_distribution(log_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for scope, groups in [
        ("overall", [("all", log_df)]),
        ("seed", list(log_df.groupby("seed_label", dropna=False))),
        ("month", list(log_df.groupby("transaction_month", dropna=False))),
    ]:
        for label, g in groups:
            total = len(g)
            for decision in ALL_DECISIONS:
                d = g[g["decisao"].eq(decision)]
                rows.append(
                    {
                        "scope": scope,
                        "label": str(label),
                        "decisao": decision,
                        "n_rows": int(len(d)),
                        "rate": round(safe_div(len(d), total), 6),
                        "n_frauds": int(d["is_fraud"].sum()) if "is_fraud" in d.columns else 0,
                        "total_value": round(float(d["vl_pix"].sum()), 2) if "vl_pix" in d.columns else 0.0,
                        "avg_score_final": round(float(d["score_final"].mean()), 4) if len(d) else 0.0,
                        "avg_lgbm_raw": round(float(d["lgbm_raw"].mean()), 8) if len(d) else 0.0,
                        "avg_if_percentile": round(float(d["if_percentile"].mean()), 8) if len(d) else 0.0,
                        "n_review_recommended": int(d["review_recommended"].sum()) if "review_recommended" in d.columns else 0,
                        "n_c1_applied": int(d["exp006f_c1_applied"].sum()) if "exp006f_c1_applied" in d.columns else 0,
                    }
                )

    return pd.DataFrame(rows)


def build_rule_metrics(log_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for idx, row in log_df.iterrows():
        rules = row.get("rules_applied_list", [])
        if not rules:
            continue

        for rule in rules:
            rows.append(
                {
                    "rule": str(rule),
                    "seed_label": row.get("seed_label", ""),
                    "transaction_month": row.get("transaction_month", ""),
                    "decisao": row.get("decisao", ""),
                    "is_fraud": int(row.get("is_fraud", 0)),
                    "vl_pix": float(row.get("vl_pix", 0.0)) if not pd.isna(row.get("vl_pix", 0.0)) else 0.0,
                    "score_final": float(row.get("score_final", 0.0)) if not pd.isna(row.get("score_final", 0.0)) else 0.0,
                    "lgbm_raw": float(row.get("lgbm_raw", 0.0)) if not pd.isna(row.get("lgbm_raw", 0.0)) else 0.0,
                    "if_percentile": float(row.get("if_percentile", 0.0)) if not pd.isna(row.get("if_percentile", 0.0)) else 0.0,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "scope",
                "label",
                "rule",
                "n_occurrences",
                "n_frauds",
                "total_value",
                "avg_score_final",
                "avg_lgbm_raw",
                "avg_if_percentile",
            ]
        )

    exploded = pd.DataFrame(rows)

    summary_rows = []

    for scope, group_cols in [
        ("overall", ["rule"]),
        ("seed", ["seed_label", "rule"]),
        ("month", ["transaction_month", "rule"]),
        ("decision", ["decisao", "rule"]),
    ]:
        grouped = (
            exploded
            .groupby(group_cols, dropna=False)
            .agg(
                n_occurrences=("rule", "count"),
                n_frauds=("is_fraud", "sum"),
                total_value=("vl_pix", "sum"),
                avg_score_final=("score_final", "mean"),
                avg_lgbm_raw=("lgbm_raw", "mean"),
                avg_if_percentile=("if_percentile", "mean"),
            )
            .reset_index()
        )

        for _, r in grouped.iterrows():
            if scope == "overall":
                label = "all"
                rule = r["rule"]
            elif scope == "seed":
                label = r["seed_label"]
                rule = r["rule"]
            elif scope == "month":
                label = r["transaction_month"]
                rule = r["rule"]
            else:
                label = r["decisao"]
                rule = r["rule"]

            summary_rows.append(
                {
                    "scope": scope,
                    "label": str(label),
                    "rule": str(rule),
                    "n_occurrences": int(r["n_occurrences"]),
                    "n_frauds": int(r["n_frauds"]),
                    "total_value": round(float(r["total_value"]), 2),
                    "avg_score_final": round(float(r["avg_score_final"]), 4),
                    "avg_lgbm_raw": round(float(r["avg_lgbm_raw"]), 8),
                    "avg_if_percentile": round(float(r["avg_if_percentile"]), 8),
                }
            )

    return pd.DataFrame(summary_rows).sort_values(
        ["scope", "n_occurrences"],
        ascending=[True, False],
    )


def build_score_bands(log_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for band_col in ["score_band", "lgbm_band", "if_band"]:
        if band_col not in log_df.columns:
            continue

        grouped = (
            log_df
            .groupby([band_col, "decisao"], dropna=False)
            .agg(
                n_rows=("decision_id", "count"),
                n_frauds=("is_fraud", "sum"),
                total_value=("vl_pix", "sum"),
                avg_score_final=("score_final", "mean"),
                avg_lgbm_raw=("lgbm_raw", "mean"),
                avg_if_percentile=("if_percentile", "mean"),
            )
            .reset_index()
        )

        for _, r in grouped.iterrows():
            rows.append(
                {
                    "band_type": band_col,
                    "band": str(r[band_col]),
                    "decisao": str(r["decisao"]),
                    "n_rows": int(r["n_rows"]),
                    "n_frauds": int(r["n_frauds"]),
                    "total_value": round(float(r["total_value"]), 2),
                    "avg_score_final": round(float(r["avg_score_final"]), 4),
                    "avg_lgbm_raw": round(float(r["avg_lgbm_raw"]), 8),
                    "avg_if_percentile": round(float(r["avg_if_percentile"]), 8),
                }
            )

    return pd.DataFrame(rows)


def build_daily_metrics(log_df: pd.DataFrame) -> pd.DataFrame:
    valid = log_df[log_df["transaction_date"].notna()].copy()

    if valid.empty:
        return pd.DataFrame()

    grouped = (
        valid
        .groupby(["transaction_day", "seed_label"], dropna=False)
        .agg(
            n_decisions=("decision_id", "count"),
            n_unique_transactions=("transaction_id", "nunique"),
            n_frauds=("is_fraud", "sum"),
            n_aprovar=("is_aprovar", "sum"),
            n_confirmar=("is_confirmar", "sum"),
            n_bloquear=("is_bloquear", "sum"),
            n_review_recommended=("review_recommended", "sum"),
            n_c1_applied=("exp006f_c1_applied", "sum"),
            tp=("tp_flag", "sum"),
            fp=("fp_flag", "sum"),
            fn=("fn_flag", "sum"),
            tn=("tn_flag", "sum"),
            total_value=("vl_pix", "sum"),
            avg_score_final=("score_final", "mean"),
            avg_lgbm_raw=("lgbm_raw", "mean"),
            avg_if_percentile=("if_percentile", "mean"),
        )
        .reset_index()
    )

    grouped["precision"] = grouped.apply(lambda r: safe_div(r["tp"], r["tp"] + r["fp"]), axis=1)
    grouped["recall"] = grouped.apply(lambda r: safe_div(r["tp"], r["tp"] + r["fn"]), axis=1)
    grouped["f1"] = grouped.apply(lambda r: safe_div(2 * r["precision"] * r["recall"], r["precision"] + r["recall"]), axis=1)
    grouped["fpr"] = grouped.apply(lambda r: safe_div(r["fp"], r["fp"] + r["tn"]), axis=1)

    return grouped


def build_review_queue_metrics(review_df: pd.DataFrame) -> pd.DataFrame:
    if review_df.empty:
        return pd.DataFrame()

    rows = []

    for scope, group_col in [
        ("overall", None),
        ("priority_band", "priority_band"),
        ("primary_reason", "primary_reason"),
        ("month", "transaction_month"),
    ]:
        if group_col is None:
            groups = [("all", review_df)]
        else:
            groups = list(review_df.groupby(group_col, dropna=False))

        for label, g in groups:
            rows.append(
                {
                    "scope": scope,
                    "label": str(label),
                    "n_queue_items": int(len(g)),
                    "n_unique_transactions": int(g["transaction_id"].nunique()) if "transaction_id" in g.columns else int(len(g)),
                    "n_known_frauds": int(g["is_fraud_any"].sum()) if "is_fraud_any" in g.columns else 0,
                    "avg_priority_score": round(float(g["priority_score"].mean()), 4) if "priority_score" in g.columns and len(g) else 0.0,
                    "avg_score_final": round(float(g["score_final"].mean()), 4) if "score_final" in g.columns and len(g) else 0.0,
                    "avg_lgbm_raw": round(float(g["lgbm_raw"].mean()), 8) if "lgbm_raw" in g.columns and len(g) else 0.0,
                    "avg_if_percentile": round(float(g["if_percentile"].mean()), 8) if "if_percentile" in g.columns and len(g) else 0.0,
                    "total_value": round(float(g["vl_pix"].sum()), 2) if "vl_pix" in g.columns else 0.0,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Power BI facts
# ============================================================

def build_powerbi_decision_fact(log_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "decision_id",
        "transaction_id",
        "customer_id_hash",
        "seed",
        "seed_label",
        "transaction_date",
        "transaction_day",
        "transaction_month",
        "model_version",
        "decision_engine_version",
        "scoring_config_version",
        "decisao",
        "is_positive_decision",
        "is_aprovar",
        "is_confirmar",
        "is_bloquear",
        "review_recommended",
        "is_fraud",
        "tp_flag",
        "fp_flag",
        "fn_flag",
        "tn_flag",
        "score_final",
        "score_final_original",
        "lgbm_raw",
        "lgbm_mapped",
        "if_percentile",
        "se_score",
        "beh_score",
        "vl_pix",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "exp006f_c1_applied",
        "score_band",
        "lgbm_band",
        "if_band",
        "rules_applied",
        "guardrails_applied",
        "decision_reason",
    ]

    available = [c for c in columns if c in log_df.columns]
    out = log_df[available].copy()

    for c in ["transaction_date"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.date.astype(str)

    return out


def build_powerbi_review_queue_fact(review_df: pd.DataFrame) -> pd.DataFrame:
    if review_df.empty:
        return review_df.copy()

    columns = [
        "queue_rank_dedup",
        "transaction_id",
        "customer_id_hash",
        "transaction_date",
        "transaction_day",
        "transaction_month",
        "priority_band",
        "priority_score_max",
        "priority_score_mean",
        "priority_score",
        "primary_reason",
        "review_reasons_union",
        "n_seed_occurrences",
        "seeds_present",
        "score_final",
        "score_gap_to_confirmar",
        "lgbm_raw",
        "lgbm_mapped",
        "if_percentile",
        "se_score",
        "beh_score",
        "vl_pix",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "is_fraud_any",
    ]

    available = [c for c in columns if c in review_df.columns]
    out = review_df[available].copy()

    if "transaction_date" in out.columns:
        out["transaction_date"] = pd.to_datetime(out["transaction_date"], errors="coerce").dt.date.astype(str)

    return out


# ============================================================
# Reports
# ============================================================

def write_dashboard_readme(
    log_df: pd.DataFrame,
    review_df: pd.DataFrame,
    kpi_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    rule_df: pd.DataFrame,
) -> None:
    overall = kpi_df[(kpi_df["scope"] == "overall") & (kpi_df["label"] == "all")].iloc[0].to_dict()

    lines = [
        "# EXP-009D — Painel de Métricas Operacionais",
        "",
        f"Gerado em: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Objetivo",
        "",
        "Gerar bases offline para Power BI/dashboard operacional do baseline `post_fase2_c1`.",
        "",
        "## Arquivos gerados",
        "",
        "| Arquivo | Uso |",
        "|---|---|",
        "| `01_kpi_overall.csv` | KPIs gerais, por seed e por mês |",
        "| `02_decision_distribution.csv` | Distribuição de `APROVAR`, `CONFIRMAR`, `BLOQUEAR` |",
        "| `03_rule_metrics.csv` | Métricas por regra aplicada |",
        "| `04_score_bands.csv` | Distribuição por faixas de score, LGBM e IF |",
        "| `05_daily_metrics.csv` | Métricas por dia e seed |",
        "| `06_review_queue_metrics.csv` | Métricas da fila de revisão humana |",
        "| `07_powerbi_decision_fact.csv` | Fato principal para Power BI |",
        "| `08_powerbi_review_queue_fact.csv` | Fato da fila de revisão para Power BI |",
        "",
        "## KPIs principais",
        "",
        "| Métrica | Valor |",
        "|---|---:|",
        f"| Decisões | {int(overall['n_decisions'])} |",
        f"| Transações únicas | {int(overall['n_unique_transactions'])} |",
        f"| Fraudes conhecidas | {int(overall['n_frauds'])} |",
        f"| APROVAR | {int(overall['n_aprovar'])} |",
        f"| CONFIRMAR | {int(overall['n_confirmar'])} |",
        f"| BLOQUEAR | {int(overall['n_bloquear'])} |",
        f"| C1 aplicada | {int(overall['n_c1_applied'])} |",
        f"| Precision | {float(overall['precision']):.6f} |",
        f"| Recall | {float(overall['recall']):.6f} |",
        f"| F1 | {float(overall['f1']):.6f} |",
        f"| FPR | {float(overall['fpr']):.6f} |",
        f"| Valor total | {float(overall['total_value']):.2f} |",
        "",
        "## Fila de revisão humana",
        "",
        f"- Itens deduplicados: `{len(review_df)}`",
        f"- Fraudes conhecidas na fila: `{int(review_df['is_fraud_any'].sum()) if 'is_fraud_any' in review_df.columns and len(review_df) else 0}`",
        "",
        "## Sugestão de abas no Power BI",
        "",
        "1. **Resumo Executivo**: decisões, precision, recall, F1, FPR, C1, fila de revisão.",
        "2. **Decisões**: distribuição por decisão, seed, mês e valor.",
        "3. **Regras**: volume por regra aplicada, especialmente C1, V1 e thresholds base.",
        "4. **Scores**: faixas de `score_final`, `lgbm_raw` e `if_percentile`.",
        "5. **Fila de Revisão**: prioridade, motivo, valor, score e top casos.",
        "6. **Drift/Monitoramento**: integrar futuramente com EXP-009B.",
        "",
        "## Medidas DAX sugeridas",
        "",
        "Use vírgula como separador de argumentos no seu Power BI.",
        "",
        "```DAX",
        "Qtd Decisões = COUNTROWS('07_powerbi_decision_fact')",
        "",
        "Qtd Aprovar = CALCULATE([Qtd Decisões], '07_powerbi_decision_fact'[decisao] = \"APROVAR\")",
        "",
        "Qtd Confirmar = CALCULATE([Qtd Decisões], '07_powerbi_decision_fact'[decisao] = \"CONFIRMAR\")",
        "",
        "Qtd Bloquear = CALCULATE([Qtd Decisões], '07_powerbi_decision_fact'[decisao] = \"BLOQUEAR\")",
        "",
        "Taxa Confirmar = DIVIDE([Qtd Confirmar], [Qtd Decisões], 0)",
        "",
        "Taxa Bloquear = DIVIDE([Qtd Bloquear], [Qtd Decisões], 0)",
        "",
        "TP = SUM('07_powerbi_decision_fact'[tp_flag])",
        "",
        "FP = SUM('07_powerbi_decision_fact'[fp_flag])",
        "",
        "FN = SUM('07_powerbi_decision_fact'[fn_flag])",
        "",
        "Precision = DIVIDE([TP], [TP] + [FP], 0)",
        "",
        "Recall = DIVIDE([TP], [TP] + [FN], 0)",
        "",
        "F1 = DIVIDE(2 * [Precision] * [Recall], [Precision] + [Recall], 0)",
        "",
        "Qtd C1 = CALCULATE([Qtd Decisões], '07_powerbi_decision_fact'[exp006f_c1_applied] = TRUE())",
        "",
        "Qtd Fila Revisão = COUNTROWS('08_powerbi_review_queue_fact')",
        "```",
        "",
        "## Decisão",
        "",
        "EXP-009D aprovado se os CSVs forem gerados corretamente e os KPIs baterem com o baseline pós-C1.",
    ]

    (OUTPUT_DIR / "09_dashboard_readme.md").write_text("\n".join(lines), encoding="utf-8")


def write_next_experiment_spec() -> None:
    lines = [
        "# Próximo experimento recomendado",
        "",
        "## EXP-009E — Smoke Test de Reprodutibilidade e Pacote de Governança",
        "",
        "## Objetivo",
        "",
        "Criar um comando único que valide a governança mínima do projeto antes de qualquer nova rodada experimental.",
        "",
        "## Ações",
        "",
        "- Rodar py_compile nos módulos críticos.",
        "- Rodar regressão pós-C1 normal e slow.",
        "- Verificar existência dos artefatos oficiais.",
        "- Verificar hashes registrados no MANIFEST_MODEL.json.",
        "- Verificar se os logs estruturados e o dashboard operacional foram gerados.",
        "- Gerar um relatório `GOVERNANCE_SMOKE_TEST.md`.",
        "",
        "## Entradas sugeridas",
        "",
        "- `backend/artefatos/MANIFEST_MODEL.json`",
        "- `tests/test_regression_post_fase2.py`",
        "- `docs/JOURNAL.md`",
        "- `resultados/experimentos/EXP-009A/03_decision_log_all.jsonl`",
        "- `resultados/experimentos/EXP-009D/07_powerbi_decision_fact.csv`",
        "",
        "## Critério de aprovação",
        "",
        "Todos os testes e verificações devem passar sem warning crítico.",
    ]

    (OUTPUT_DIR / "10_next_experiment_spec.md").write_text("\n".join(lines), encoding="utf-8")


def build_input_summary(
    decision_log_path: Path,
    review_queue_path: Path,
    log_df: pd.DataFrame,
    review_df: pd.DataFrame,
) -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH, {})

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision_log_path": str(decision_log_path),
        "review_queue_path": str(review_queue_path),
        "output_dir": str(OUTPUT_DIR),
        "decision_log_rows": int(len(log_df)),
        "decision_log_unique_transactions": int(log_df["transaction_id"].nunique()) if "transaction_id" in log_df.columns else int(len(log_df)),
        "review_queue_rows": int(len(review_df)),
        "review_queue_unique_transactions": int(review_df["transaction_id"].nunique()) if "transaction_id" in review_df.columns else int(len(review_df)),
        "manifest_path": str(MANIFEST_PATH),
        "manifest_model_version": manifest.get("model_version"),
        "manifest_status": manifest.get("status"),
        "manifest_official_metrics": manifest.get("official_metrics"),
        "notes": [
            "Labels sao usados para auditoria offline e metricas de validacao, nao para priorizacao operacional em producao.",
            "Metricas combinadas entre seeds servem para observabilidade do harness; metricas oficiais permanecem por seed no Manifest.",
        ],
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-009D Painel de Metricas Operacionais")
    parser.add_argument("--decision-log", type=str, default=str(DEFAULT_DECISION_LOG))
    parser.add_argument("--review-queue", type=str, default=str(DEFAULT_REVIEW_QUEUE))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    decision_log_path = Path(args.decision_log).resolve()
    review_queue_path = Path(args.review_queue).resolve()

    print("=" * 72)
    print("EXP-009D — Painel de Metricas Operacionais")
    print("=" * 72)

    print("[1/8] Carregando decision log...")
    decision_log = normalize_decision_log(read_table(decision_log_path))
    print(f"[OK] Decision log: {len(decision_log)} linhas")

    print("[2/8] Carregando fila de revisao...")
    if review_queue_path.exists():
        review_queue = normalize_review_queue(read_table(review_queue_path))
    else:
        review_queue = pd.DataFrame()
    print(f"[OK] Review queue: {len(review_queue)} linhas")

    print("[3/8] Gerando KPIs gerais...")
    kpi_overall = build_kpi_overall(decision_log)
    kpi_overall.to_csv(OUTPUT_DIR / "01_kpi_overall.csv", index=False, encoding="utf-8-sig")

    print("[4/8] Gerando distribuicao de decisoes...")
    decision_distribution = build_decision_distribution(decision_log)
    decision_distribution.to_csv(OUTPUT_DIR / "02_decision_distribution.csv", index=False, encoding="utf-8-sig")

    print("[5/8] Gerando metricas por regra e bandas de score...")
    rule_metrics = build_rule_metrics(decision_log)
    rule_metrics.to_csv(OUTPUT_DIR / "03_rule_metrics.csv", index=False, encoding="utf-8-sig")

    score_bands = build_score_bands(decision_log)
    score_bands.to_csv(OUTPUT_DIR / "04_score_bands.csv", index=False, encoding="utf-8-sig")

    print("[6/8] Gerando metricas diarias e fila...")
    daily_metrics = build_daily_metrics(decision_log)
    daily_metrics.to_csv(OUTPUT_DIR / "05_daily_metrics.csv", index=False, encoding="utf-8-sig")

    review_metrics = build_review_queue_metrics(review_queue)
    review_metrics.to_csv(OUTPUT_DIR / "06_review_queue_metrics.csv", index=False, encoding="utf-8-sig")

    print("[7/8] Gerando fatos Power BI...")
    powerbi_decision_fact = build_powerbi_decision_fact(decision_log)
    powerbi_decision_fact.to_csv(OUTPUT_DIR / "07_powerbi_decision_fact.csv", index=False, encoding="utf-8-sig")

    powerbi_review_fact = build_powerbi_review_queue_fact(review_queue)
    powerbi_review_fact.to_csv(OUTPUT_DIR / "08_powerbi_review_queue_fact.csv", index=False, encoding="utf-8-sig")

    print("[8/8] Escrevendo relatorios...")
    input_summary = build_input_summary(
        decision_log_path=decision_log_path,
        review_queue_path=review_queue_path,
        log_df=decision_log,
        review_df=review_queue,
    )
    write_json(OUTPUT_DIR / "00_input_summary.json", input_summary)

    write_dashboard_readme(
        log_df=decision_log,
        review_df=review_queue,
        kpi_df=kpi_overall,
        decision_df=decision_distribution,
        rule_df=rule_metrics,
    )

    write_next_experiment_spec()

    print()
    print("[OK] EXP-009D concluido.")
    print(f"[OK] Artefatos em: {OUTPUT_DIR}")
    print()
    print("Arquivos principais:")
    print(f"  {OUTPUT_DIR / '00_input_summary.json'}")
    print(f"  {OUTPUT_DIR / '01_kpi_overall.csv'}")
    print(f"  {OUTPUT_DIR / '02_decision_distribution.csv'}")
    print(f"  {OUTPUT_DIR / '03_rule_metrics.csv'}")
    print(f"  {OUTPUT_DIR / '04_score_bands.csv'}")
    print(f"  {OUTPUT_DIR / '05_daily_metrics.csv'}")
    print(f"  {OUTPUT_DIR / '06_review_queue_metrics.csv'}")
    print(f"  {OUTPUT_DIR / '07_powerbi_decision_fact.csv'}")
    print(f"  {OUTPUT_DIR / '08_powerbi_review_queue_fact.csv'}")
    print(f"  {OUTPUT_DIR / '09_dashboard_readme.md'}")
    print(f"  {OUTPUT_DIR / '10_next_experiment_spec.md'}")


if __name__ == "__main__":
    main()