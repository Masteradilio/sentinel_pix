"""
EXP-009C — Zona Cinza e Fila de Revisao Humana

Objetivo:
  Criar uma fila offline de casos informativos para revisao humana futura,
  usando os decision logs estruturados do EXP-009A.

Este experimento:
  - Nao altera o modelo.
  - Nao altera scoring_config.json.
  - Nao altera DecisionEngine.
  - Nao roda E2E.
  - Nao usa labels para priorizar a fila operacional.
  - Usa labels apenas para auditoria offline, quando disponiveis.

Entradas default:
  resultados/experimentos/EXP-009A/03_decision_log_all.jsonl

Saidas:
  resultados/experimentos/EXP-009C/
    00_input_summary.json
    01_review_queue_all.csv
    02_review_queue_dedup.csv
    03_priority_summary.csv
    04_reason_summary.csv
    05_label_audit.json
    06_examples.json
    07_recommendation.md
    08_next_experiment_spec.md

Uso:
  python experimentos\\exp_009c_review_queue\\run_exp_009c_review_queue.py
"""

from __future__ import annotations

import argparse
import json
import math
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

DEFAULT_INPUT_LOG = ROOT / "resultados" / "experimentos" / "EXP-009A" / "03_decision_log_all.jsonl"
DEFAULT_LOG_42 = ROOT / "resultados" / "experimentos" / "EXP-009A" / "01_decision_log_seed_42.csv"
DEFAULT_LOG_123 = ROOT / "resultados" / "experimentos" / "EXP-009A" / "02_decision_log_seed_123.csv"
OUTPUT_DIR = ROOT / "resultados" / "experimentos" / "EXP-009C"

MANIFEST_PATH = ROOT / "backend" / "artefatos" / "MANIFEST_MODEL.json"


# ============================================================
# Config
# ============================================================

DECISION_APPROVE = "APROVAR"

C1_MIN_SCORE = 58.0
C1_MAX_SCORE = 62.0
C1_MIN_VALOR = 100.0
C1_MAX_VALOR = 500.0
C1_MAX_REL_MESES = 12.0
C1_MIN_LGBM = 0.06
C1_MAX_LGBM = 0.10

# "Quase C1" e uma janela um pouco mais larga do que a C1 oficial.
C1_ALMOST_MIN_SCORE = 55.0
C1_ALMOST_MAX_SCORE = 62.0
C1_ALMOST_MIN_LGBM = 0.04
C1_ALMOST_MAX_LGBM = 0.12
C1_ALMOST_MIN_VALOR = 80.0
C1_ALMOST_MAX_VALOR = 700.0
C1_ALMOST_MAX_REL_MESES = 18.0

# V1 quase acionada: alto valor contextual sem necessariamente cumprir todos os requisitos.
V1_ALMOST_MIN_VALOR = 10000.0
V1_ALMOST_REL_MAX = 18.0
V1_ALMOST_IF_MIN = 0.95
V1_ALMOST_LGBM_MIN = 0.005

REVIEW_BANDS = [
    ("CRITICAL", 90),
    ("HIGH", 70),
    ("MEDIUM", 45),
    ("LOW", 20),
    ("WATCH", 1),
]


# ============================================================
# IO helpers
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


def read_log(path: Path) -> pd.DataFrame:
    if path.exists():
        if path.suffix.lower() == ".jsonl":
            return pd.read_json(path, lines=True)
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict) and "records" in data:
                return pd.DataFrame(data["records"])
            raise ValueError(f"JSON nao reconhecido como lista de registros: {path}")
        return pd.read_csv(path)

    # Fallback: se o JSONL combinado nao existir, usa os CSVs por seed.
    if DEFAULT_LOG_42.exists() and DEFAULT_LOG_123.exists():
        return pd.concat(
            [pd.read_csv(DEFAULT_LOG_42), pd.read_csv(DEFAULT_LOG_123)],
            ignore_index=True,
        )

    raise FileNotFoundError(f"Log nao encontrado: {path}")


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


def normalize_log(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = [
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

    if "decisao" in out.columns:
        out["decisao"] = out["decisao"].astype(str).str.upper()

    if "review_recommended" in out.columns:
        out["review_recommended"] = out["review_recommended"].astype(str).str.lower().isin(
            {"true", "1", "yes", "sim"}
        )

    if "rules_applied" in out.columns:
        out["rules_applied_list"] = out["rules_applied"].apply(parse_json_list)
    else:
        out["rules_applied_list"] = [[] for _ in range(len(out))]

    if "guardrails_applied" in out.columns:
        out["guardrails_applied_list"] = out["guardrails_applied"].apply(parse_json_list)
    else:
        out["guardrails_applied_list"] = [[] for _ in range(len(out))]

    return out


# ============================================================
# Feature helpers
# ============================================================

def num(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        value = row.get(col, default)
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def text(row: pd.Series, col: str, default: str = "") -> str:
    try:
        value = row.get(col, default)
        if pd.isna(value):
            return default
        return str(value)
    except Exception:
        return default


def boolish(row: pd.Series, col: str) -> bool:
    value = row.get(col, False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "sim"}


def priority_band(score: float) -> str:
    for band, minimum in REVIEW_BANDS:
        if score >= minimum:
            return band
    return "NONE"


def is_near_confirm_score(score: float) -> bool:
    return 55.0 <= score < 62.0


def is_very_near_confirm_score(score: float) -> bool:
    return 60.0 <= score < 62.0


def is_lgbm_gray(lgbm_raw: float) -> bool:
    return 0.05 <= lgbm_raw < 0.20


def is_lgbm_gray_high(lgbm_raw: float) -> bool:
    return 0.10 <= lgbm_raw < 0.20


def is_if_high(if_percentile: float) -> bool:
    return if_percentile >= 0.95


def is_if_extreme(if_percentile: float) -> bool:
    return if_percentile >= 0.985


def is_module_discordant(row: pd.Series) -> bool:
    lgbm_raw = num(row, "lgbm_raw")
    if_percentile = num(row, "if_percentile")
    se_score = num(row, "se_score")
    beh_score = num(row, "beh_score")
    score_final = num(row, "score_final")

    high_if_low_lgbm = if_percentile >= 0.95 and lgbm_raw < 0.05
    se_beh_suppressed = (se_score + beh_score) >= 50 and lgbm_raw < 0.30 and score_final < 62
    lgbm_gray_no_rule = is_lgbm_gray(lgbm_raw) and se_score == 0 and beh_score == 0 and score_final < 62
    high_lgbm_low_score = lgbm_raw >= 0.10 and score_final < 55

    return bool(high_if_low_lgbm or se_beh_suppressed or lgbm_gray_no_rule or high_lgbm_low_score)


def is_c1_almost(row: pd.Series) -> bool:
    if text(row, "decisao").upper() != DECISION_APPROVE:
        return False

    first_receiver = int(num(row, "first_receiver_flag"))
    pix_random = int(num(row, "pix_key_random_flag"))
    rel = num(row, "qt_tempo_relacionamento_mes", default=999.0)
    value = num(row, "vl_pix")
    lgbm_raw = num(row, "lgbm_raw")
    score = num(row, "score_final")
    se = num(row, "se_score")
    beh = num(row, "beh_score")

    return bool(
        first_receiver == 1
        and pix_random == 0
        and rel <= C1_ALMOST_MAX_REL_MESES
        and C1_ALMOST_MIN_VALOR <= value < C1_ALMOST_MAX_VALOR
        and C1_ALMOST_MIN_LGBM <= lgbm_raw < C1_ALMOST_MAX_LGBM
        and C1_ALMOST_MIN_SCORE <= score < C1_ALMOST_MAX_SCORE
        and se <= 0
        and beh <= 0
    )


def is_c1_official_would_match(row: pd.Series) -> bool:
    if text(row, "decisao").upper() != DECISION_APPROVE:
        return False

    first_receiver = int(num(row, "first_receiver_flag"))
    pix_random = int(num(row, "pix_key_random_flag"))
    rel = num(row, "qt_tempo_relacionamento_mes", default=999.0)
    value = num(row, "vl_pix")
    lgbm_raw = num(row, "lgbm_raw")
    score = num(row, "score_final")
    se = num(row, "se_score")
    beh = num(row, "beh_score")

    return bool(
        first_receiver == 1
        and pix_random == 0
        and rel <= C1_MAX_REL_MESES
        and C1_MIN_VALOR <= value < C1_MAX_VALOR
        and C1_MIN_LGBM <= lgbm_raw < C1_MAX_LGBM
        and C1_MIN_SCORE <= score < C1_MAX_SCORE
        and se <= 0
        and beh <= 0
    )


def is_v1_almost(row: pd.Series) -> bool:
    if text(row, "decisao").upper() != DECISION_APPROVE:
        return False

    value = num(row, "vl_pix")
    rel = num(row, "qt_tempo_relacionamento_mes", default=999.0)
    if_percentile = num(row, "if_percentile")
    lgbm_raw = num(row, "lgbm_raw")
    first_receiver = int(num(row, "first_receiver_flag"))

    high_value_context = value >= V1_ALMOST_MIN_VALOR
    short_or_first = rel <= V1_ALMOST_REL_MAX or first_receiver == 1
    enough_signal = if_percentile >= V1_ALMOST_IF_MIN or lgbm_raw >= V1_ALMOST_LGBM_MIN

    return bool(high_value_context and short_or_first and enough_signal)


def evaluate_review_reasons(row: pd.Series) -> tuple[list[str], float]:
    reasons: list[str] = []
    score = 0.0

    decisao = text(row, "decisao").upper()

    if decisao != DECISION_APPROVE:
        return reasons, score

    score_final = num(row, "score_final")
    lgbm_raw = num(row, "lgbm_raw")
    if_percentile = num(row, "if_percentile")
    se_score = num(row, "se_score")
    beh_score = num(row, "beh_score")
    value = num(row, "vl_pix")
    rel = num(row, "qt_tempo_relacionamento_mes", default=999.0)
    first_receiver = int(num(row, "first_receiver_flag"))
    pix_random = int(num(row, "pix_key_random_flag"))

    if is_very_near_confirm_score(score_final):
        reasons.append("SCORE_VERY_NEAR_CONFIRM")
        score += 35
    elif is_near_confirm_score(score_final):
        reasons.append("SCORE_NEAR_CONFIRM")
        score += 25

    if is_lgbm_gray_high(lgbm_raw):
        reasons.append("LGBM_GRAY_HIGH")
        score += 25
    elif is_lgbm_gray(lgbm_raw):
        reasons.append("LGBM_GRAY")
        score += 15

    if is_if_extreme(if_percentile):
        reasons.append("IF_EXTREME")
        score += 25
    elif is_if_high(if_percentile):
        reasons.append("IF_HIGH")
        score += 15

    if se_score >= 50:
        reasons.append("SE_HIGH")
        score += 25
    elif se_score > 0:
        reasons.append("SE_SIGNAL")
        score += 10

    if beh_score >= 30:
        reasons.append("BEH_HIGH")
        score += 25
    elif beh_score > 0:
        reasons.append("BEH_SIGNAL")
        score += 10

    if is_module_discordant(row):
        reasons.append("MODULE_DISCORDANCE")
        score += 20

    if first_receiver == 1 and rel <= 12 and value >= 500:
        reasons.append("SHORT_REL_FIRST_RECEIVER_VALUE")
        score += 15

    if first_receiver == 1 and pix_random == 0 and value >= 100 and rel <= 12:
        reasons.append("C1_CONTEXT_PARTIAL")
        score += 10

    if is_c1_almost(row):
        reasons.append("C1_ALMOST_TRIGGERED")
        score += 35

    if is_c1_official_would_match(row):
        # Em teoria, depois da C1, um APROVAR nao deveria satisfazer exatamente a C1 oficial.
        # Se aparecer, isso merece prioridade maxima porque sugere regressao ou divergencia de log.
        reasons.append("C1_OFFICIAL_MATCH_BUT_APPROVED")
        score += 100

    if is_v1_almost(row):
        reasons.append("V1_ALMOST_TRIGGERED")
        score += 30

    if boolish(row, "review_recommended"):
        reasons.append("REVIEW_RECOMMENDED_BY_LOG")
        score += 5

    # Ajuste leve por valor alto, sem deixar valor dominar sozinho.
    if value >= 15000:
        reasons.append("HIGH_VALUE_15K_PLUS")
        score += 20
    elif value >= 5000:
        reasons.append("HIGH_VALUE_5K_PLUS")
        score += 10

    # Dedup dos motivos.
    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)

    # Limita score para facilitar leitura.
    score = min(score, 150.0)

    return deduped, score


# ============================================================
# Queue generation
# ============================================================

def build_review_queue(df: pd.DataFrame, min_priority: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    approved = df[df["decisao"].astype(str).str.upper().eq(DECISION_APPROVE)].copy()

    for _, row in approved.iterrows():
        reasons, score = evaluate_review_reasons(row)

        if score < min_priority or not reasons:
            continue

        tx_id = text(row, "transaction_id")
        seed = int(num(row, "seed", default=-1))

        item = {
            "queue_id": f"rq_{tx_id}_{seed}",
            "transaction_id": tx_id,
            "customer_id_hash": text(row, "customer_id_hash"),
            "seed": seed,
            "decisao": text(row, "decisao"),
            "priority_score": score,
            "priority_band": priority_band(score),
            "review_reasons": json.dumps(reasons, ensure_ascii=False),
            "primary_reason": reasons[0] if reasons else "",
            "score_final": num(row, "score_final"),
            "score_gap_to_confirmar": max(0.0, 62.0 - num(row, "score_final")),
            "lgbm_raw": num(row, "lgbm_raw"),
            "lgbm_mapped": num(row, "lgbm_mapped"),
            "if_percentile": num(row, "if_percentile"),
            "se_score": num(row, "se_score"),
            "beh_score": num(row, "beh_score"),
            "vl_pix": num(row, "vl_pix"),
            "qt_tempo_relacionamento_mes": num(row, "qt_tempo_relacionamento_mes"),
            "first_receiver_flag": int(num(row, "first_receiver_flag")),
            "pix_key_random_flag": int(num(row, "pix_key_random_flag")),
            "rules_applied": text(row, "rules_applied"),
            "guardrails_applied": text(row, "guardrails_applied"),
            "decision_reason": text(row, "decision_reason"),
            "model_version": text(row, "model_version"),
            "decision_engine_version": text(row, "decision_engine_version"),
            "scoring_config_version": text(row, "scoring_config_version"),
            # Campo de auditoria offline; nao usado na priorizacao.
            "is_fraud": int(num(row, "is_fraud", default=0)),
        }

        rows.append(item)

    if not rows:
        return pd.DataFrame()

    queue = pd.DataFrame(rows)

    queue = queue.sort_values(
        ["priority_score", "score_final", "lgbm_raw", "if_percentile", "vl_pix"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    queue["queue_rank"] = np.arange(1, len(queue) + 1)

    # Reordena colunas principais.
    front_cols = [
        "queue_rank",
        "queue_id",
        "transaction_id",
        "customer_id_hash",
        "seed",
        "priority_band",
        "priority_score",
        "primary_reason",
        "review_reasons",
        "decisao",
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
        "is_fraud",
    ]

    other_cols = [c for c in queue.columns if c not in front_cols]
    return queue[front_cols + other_cols]


def deduplicate_queue(queue: pd.DataFrame) -> pd.DataFrame:
    if queue.empty:
        return queue.copy()

    grouped_rows = []

    for tx_id, g in queue.groupby("transaction_id", dropna=False):
        best = g.sort_values(
            ["priority_score", "score_final", "lgbm_raw", "if_percentile"],
            ascending=[False, False, False, False],
        ).iloc[0].copy()

        seeds = sorted(set(int(x) for x in g["seed"].dropna().tolist()))
        all_reasons: list[str] = []

        for raw in g["review_reasons"].tolist():
            all_reasons.extend(parse_json_list(raw))

        deduped_reasons = []
        for reason in all_reasons:
            if reason not in deduped_reasons:
                deduped_reasons.append(reason)

        best["n_seed_occurrences"] = int(len(g))
        best["seeds_present"] = json.dumps(seeds, ensure_ascii=False)
        best["review_reasons_union"] = json.dumps(deduped_reasons, ensure_ascii=False)
        best["is_fraud_any"] = int(g["is_fraud"].max()) if "is_fraud" in g.columns else 0
        best["priority_score_max"] = float(g["priority_score"].max())
        best["priority_score_mean"] = float(g["priority_score"].mean())

        grouped_rows.append(best.to_dict())

    out = pd.DataFrame(grouped_rows)

    out = out.sort_values(
        ["priority_score_max", "priority_score", "score_final", "lgbm_raw", "if_percentile"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    out["queue_rank_dedup"] = np.arange(1, len(out) + 1)

    front_cols = [
        "queue_rank_dedup",
        "transaction_id",
        "customer_id_hash",
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

    other_cols = [c for c in out.columns if c not in front_cols]
    return out[front_cols + other_cols]


def summarize_priority(queue: pd.DataFrame, dedup: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for label, df in [("all_seed_rows", queue), ("deduplicated_transactions", dedup)]:
        if df.empty:
            continue

        fraud_col = "is_fraud" if label == "all_seed_rows" else "is_fraud_any"

        summary = (
            df.groupby("priority_band", dropna=False)
            .agg(
                n_rows=("transaction_id", "count"),
                n_frauds=(fraud_col, "sum"),
                avg_priority=("priority_score", "mean"),
                max_priority=("priority_score", "max"),
                avg_score_final=("score_final", "mean"),
                avg_lgbm_raw=("lgbm_raw", "mean"),
                avg_if_percentile=("if_percentile", "mean"),
                total_value=("vl_pix", "sum"),
            )
            .reset_index()
        )

        summary["view"] = label
        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    band_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "WATCH": 4, "NONE": 5}
    out["band_order"] = out["priority_band"].map(band_order).fillna(99)
    out = out.sort_values(["view", "band_order"]).drop(columns=["band_order"])

    return out


def summarize_reasons(queue: pd.DataFrame) -> pd.DataFrame:
    if queue.empty:
        return pd.DataFrame(columns=["reason", "count", "frauds", "avg_priority"])

    rows = []

    for _, row in queue.iterrows():
        reasons = parse_json_list(row.get("review_reasons"))
        for reason in reasons:
            rows.append(
                {
                    "reason": reason,
                    "priority_score": row.get("priority_score"),
                    "is_fraud": row.get("is_fraud", 0),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["reason", "count", "frauds", "avg_priority"])

    reason_df = pd.DataFrame(rows)

    out = (
        reason_df
        .groupby("reason")
        .agg(
            count=("reason", "count"),
            frauds=("is_fraud", "sum"),
            avg_priority=("priority_score", "mean"),
        )
        .reset_index()
        .sort_values(["count", "avg_priority"], ascending=[False, False])
    )

    return out


def label_audit(queue: pd.DataFrame, dedup: pd.DataFrame, source_df: pd.DataFrame) -> dict[str, Any]:
    approved = source_df[source_df["decisao"].astype(str).str.upper().eq(DECISION_APPROVE)].copy()

    total_approved = int(len(approved))
    total_approved_frauds = int(approved.get("is_fraud", pd.Series(dtype=float)).fillna(0).astype(int).sum())

    queue_frauds = int(queue.get("is_fraud", pd.Series(dtype=float)).fillna(0).astype(int).sum()) if not queue.empty else 0
    dedup_frauds = int(dedup.get("is_fraud_any", pd.Series(dtype=float)).fillna(0).astype(int).sum()) if not dedup.empty else 0

    top_counts = [25, 50, 100, 200]

    top_capture = []

    for n in top_counts:
        topn = dedup.head(n)
        top_capture.append(
            {
                "top_n": n,
                "n_rows": int(len(topn)),
                "known_frauds_captured": int(topn.get("is_fraud_any", pd.Series(dtype=float)).fillna(0).astype(int).sum()) if not topn.empty else 0,
            }
        )

    return {
        "note": "Labels sao usados apenas para auditoria offline, nao para priorizacao operacional.",
        "total_source_rows": int(len(source_df)),
        "total_approved_rows": total_approved,
        "total_approved_known_frauds": total_approved_frauds,
        "queue_rows_all_seed_rows": int(len(queue)),
        "queue_rows_deduplicated": int(len(dedup)),
        "queue_known_frauds_all_seed_rows": queue_frauds,
        "queue_known_frauds_deduplicated": dedup_frauds,
        "approved_known_fraud_capture_rate_all_seed_rows": (
            queue_frauds / max(total_approved_frauds, 1)
        ),
        "approved_known_fraud_capture_rate_deduplicated": (
            dedup_frauds / max(total_approved_frauds / 2, 1)
        ),
        "top_capture_deduplicated": top_capture,
    }


# ============================================================
# Reports
# ============================================================

def fmt_float(value: Any, ndigits: int = 4) -> str:
    try:
        if value is None or pd.isna(value):
            return ""
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return ""


def write_recommendation(
    queue: pd.DataFrame,
    dedup: pd.DataFrame,
    priority_summary: pd.DataFrame,
    reason_summary: pd.DataFrame,
    audit: dict[str, Any],
    input_log: Path,
    min_priority: float,
) -> None:
    lines = [
        "# EXP-009C — Zona Cinza e Fila de Revisão Humana",
        "",
        f"Gerado em: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Objetivo",
        "",
        "Criar uma fila offline de casos `APROVAR` que merecem revisão humana futura, usando apenas sinais disponíveis no decision log estruturado.",
        "",
        "## Entrada",
        "",
        f"- Input log: `{input_log}`",
        f"- Prioridade mínima: `{min_priority}`",
        "",
        "## Resultado",
        "",
        f"- Linhas na fila com seeds: `{len(queue)}`",
        f"- Transações deduplicadas: `{len(dedup)}`",
        f"- Fraudes conhecidas na fila com seeds: `{audit['queue_known_frauds_all_seed_rows']}`",
        f"- Fraudes conhecidas na fila deduplicada: `{audit['queue_known_frauds_deduplicated']}`",
        "",
        "> Observação: labels são usados apenas para auditoria offline. A priorização operacional não usa `is_fraud`.",
        "",
        "## Sumário por prioridade",
        "",
    ]

    if priority_summary.empty:
        lines.append("Nenhum caso entrou na fila.")
    else:
        lines.append("| Visão | Banda | Linhas | Fraudes conhecidas | Prioridade média | Score médio | LGBM médio | IF médio |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

        for _, r in priority_summary.iterrows():
            lines.append(
                f"| `{r['view']}` | `{r['priority_band']}` | {int(r['n_rows'])} | "
                f"{int(r['n_frauds'])} | {fmt_float(r['avg_priority'])} | "
                f"{fmt_float(r['avg_score_final'])} | {fmt_float(r['avg_lgbm_raw'], 6)} | "
                f"{fmt_float(r['avg_if_percentile'], 6)} |"
            )

    lines.extend([
        "",
        "## Top motivos",
        "",
    ])

    if reason_summary.empty:
        lines.append("Nenhum motivo encontrado.")
    else:
        lines.append("| Motivo | Ocorrências | Fraudes conhecidas | Prioridade média |")
        lines.append("|---|---:|---:|---:|")

        for _, r in reason_summary.head(20).iterrows():
            lines.append(
                f"| `{r['reason']}` | {int(r['count'])} | {int(r['frauds'])} | {fmt_float(r['avg_priority'])} |"
            )

    lines.extend([
        "",
        "## Top 20 casos deduplicados",
        "",
    ])

    if dedup.empty:
        lines.append("Nenhum caso deduplicado.")
    else:
        lines.append("| Rank | Tx | Banda | Prioridade | Motivo principal | Score | LGBM | IF | Valor | Rel. meses | Fraude conhecida |")
        lines.append("|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|")

        for _, r in dedup.head(20).iterrows():
            lines.append(
                f"| {int(r['queue_rank_dedup'])} | `{r['transaction_id']}` | `{r['priority_band']}` | "
                f"{fmt_float(r['priority_score_max'])} | `{r['primary_reason']}` | "
                f"{fmt_float(r['score_final'])} | {fmt_float(r['lgbm_raw'], 6)} | "
                f"{fmt_float(r['if_percentile'], 6)} | {fmt_float(r['vl_pix'], 2)} | "
                f"{fmt_float(r['qt_tempo_relacionamento_mes'], 0)} | {int(r['is_fraud_any'])} |"
            )

    lines.extend([
        "",
        "## Auditoria offline de labels",
        "",
        f"- `APROVAR` total: `{audit['total_approved_rows']}`",
        f"- Fraudes conhecidas em `APROVAR`: `{audit['total_approved_known_frauds']}`",
        f"- Fraudes conhecidas capturadas pela fila, com seeds: `{audit['queue_known_frauds_all_seed_rows']}`",
        f"- Fraudes conhecidas capturadas pela fila, deduplicado: `{audit['queue_known_frauds_deduplicated']}`",
        "",
        "## Decisão",
        "",
    ])

    if len(dedup) == 0:
        lines.append("EXP-009C não deve ser aprovado ainda: a fila ficou vazia. Revisar critérios de seleção.")
    else:
        lines.append("EXP-009C aprovado: a fila offline de revisão humana foi criada.")
        lines.append("")
        lines.append("Esta fila deve ser usada para auditoria humana, active learning futuro e coleta de evidências, não para promoção automática de regra.")

    lines.extend([
        "",
        "## Próximo passo recomendado",
        "",
        "EXP-009D — Painel de Métricas Operacionais.",
        "",
    ])

    (OUTPUT_DIR / "07_recommendation.md").write_text("\n".join(lines), encoding="utf-8")


def write_next_experiment() -> None:
    lines = [
        "# Próximo experimento recomendado",
        "",
        "## EXP-009D — Painel de Métricas Operacionais",
        "",
        "## Objetivo",
        "",
        "Criar um painel offline de métricas operacionais do baseline `post_fase2_c1`, usando os logs estruturados e a fila de revisão humana.",
        "",
        "## Ações",
        "",
        "- Consolidar métricas de decisão por dia/seed/janela.",
        "- Exibir distribuição de `APROVAR`, `CONFIRMAR` e `BLOQUEAR`.",
        "- Exibir volume e valor por decisão.",
        "- Monitorar taxa de C1 e V1.",
        "- Monitorar quantidade de casos em fila de revisão.",
        "- Gerar base parquet/csv para Power BI ou dashboard offline.",
        "",
        "## Entradas sugeridas",
        "",
        "- `resultados/experimentos/EXP-009A/03_decision_log_all.jsonl`",
        "- `resultados/experimentos/EXP-009C/02_review_queue_dedup.csv`",
    ]

    (OUTPUT_DIR / "08_next_experiment_spec.md").write_text("\n".join(lines), encoding="utf-8")


def write_examples(queue: pd.DataFrame, dedup: pd.DataFrame, audit: dict[str, Any]) -> None:
    examples = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "label_audit": audit,
        "top_critical": dedup[dedup["priority_band"].eq("CRITICAL")].head(20).to_dict(orient="records") if not dedup.empty else [],
        "top_high": dedup[dedup["priority_band"].eq("HIGH")].head(20).to_dict(orient="records") if not dedup.empty else [],
        "top_medium": dedup[dedup["priority_band"].eq("MEDIUM")].head(20).to_dict(orient="records") if not dedup.empty else [],
        "known_frauds_in_queue": dedup[dedup["is_fraud_any"].astype(int).eq(1)].head(50).to_dict(orient="records") if not dedup.empty else [],
    }

    write_json(OUTPUT_DIR / "06_examples.json", examples)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-009C Zona Cinza e Fila de Revisao Humana")
    parser.add_argument("--input-log", type=str, default=str(DEFAULT_INPUT_LOG))
    parser.add_argument("--min-priority", type=float, default=1.0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_log = Path(args.input_log).resolve()

    print("=" * 72)
    print("EXP-009C — Zona Cinza e Fila de Revisao Humana")
    print("=" * 72)

    print("[1/7] Carregando decision log...")
    source = normalize_log(read_log(input_log))

    print(f"[OK] Linhas carregadas: {len(source)}")

    print("[2/7] Construindo fila de revisao...")
    queue = build_review_queue(source, min_priority=args.min_priority)
    queue.to_csv(OUTPUT_DIR / "01_review_queue_all.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] Linhas na fila: {len(queue)}")

    print("[3/7] Deduplicando por transaction_id...")
    dedup = deduplicate_queue(queue)
    dedup.to_csv(OUTPUT_DIR / "02_review_queue_dedup.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] Transacoes deduplicadas: {len(dedup)}")

    print("[4/7] Gerando sumarios...")
    priority_summary = summarize_priority(queue, dedup)
    priority_summary.to_csv(OUTPUT_DIR / "03_priority_summary.csv", index=False, encoding="utf-8-sig")

    reason_summary = summarize_reasons(queue)
    reason_summary.to_csv(OUTPUT_DIR / "04_reason_summary.csv", index=False, encoding="utf-8-sig")

    print("[5/7] Gerando auditoria offline de labels...")
    audit = label_audit(queue, dedup, source)
    write_json(OUTPUT_DIR / "05_label_audit.json", audit)

    print("[6/7] Escrevendo exemplos...")
    write_examples(queue, dedup, audit)

    print("[7/7] Escrevendo relatorios...")
    input_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_log": str(input_log),
        "output_dir": str(OUTPUT_DIR),
        "source_rows": int(len(source)),
        "source_approved_rows": int(source["decisao"].astype(str).str.upper().eq(DECISION_APPROVE).sum()) if "decisao" in source.columns else None,
        "queue_rows_all_seed_rows": int(len(queue)),
        "queue_rows_deduplicated": int(len(dedup)),
        "min_priority": args.min_priority,
        "manifest_path": str(MANIFEST_PATH),
        "manifest": read_json(MANIFEST_PATH, {}),
        "note": "Labels sao usados apenas para auditoria offline, nao para priorizacao operacional.",
    }

    write_json(OUTPUT_DIR / "00_input_summary.json", input_summary)

    write_recommendation(
        queue=queue,
        dedup=dedup,
        priority_summary=priority_summary,
        reason_summary=reason_summary,
        audit=audit,
        input_log=input_log,
        min_priority=args.min_priority,
    )

    write_next_experiment()

    print()
    print("[OK] EXP-009C concluido.")
    print(f"[OK] Artefatos em: {OUTPUT_DIR}")
    print()
    print("Arquivos principais:")
    print(f"  {OUTPUT_DIR / '00_input_summary.json'}")
    print(f"  {OUTPUT_DIR / '01_review_queue_all.csv'}")
    print(f"  {OUTPUT_DIR / '02_review_queue_dedup.csv'}")
    print(f"  {OUTPUT_DIR / '03_priority_summary.csv'}")
    print(f"  {OUTPUT_DIR / '04_reason_summary.csv'}")
    print(f"  {OUTPUT_DIR / '05_label_audit.json'}")
    print(f"  {OUTPUT_DIR / '06_examples.json'}")
    print(f"  {OUTPUT_DIR / '07_recommendation.md'}")
    print(f"  {OUTPUT_DIR / '08_next_experiment_spec.md'}")


if __name__ == "__main__":
    main()