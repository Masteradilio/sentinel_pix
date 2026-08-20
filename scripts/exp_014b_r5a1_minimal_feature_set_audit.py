"""
EXP-014B-R5A1 - Minimal Feature Set and Redundancy Audit.

Diagnostic-only script. It does not train models and does not change the
EXP-014B-R4G-FAST-FROZEN baseline.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPERIMENT = "EXP-014B-R5A1"
BASELINE_EXPERIMENT = "EXP-014B-R4G-FAST-FROZEN"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "resultados"
    / "experimentos"
    / BASELINE_EXPERIMENT
    / "06_predictions_frozen.csv"
)
FALLBACK_INPUT = (
    PROJECT_ROOT
    / "resultados"
    / "experimentos"
    / "EXP-014B-R4G-FAST"
    / "11_predictions_recommended.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / EXPERIMENT

LABEL_CANDIDATES = ["is_fraud", "fraude", "target", "label", "tp_fraude"]
ACTION_CANDIDATES = [
    "r4g_fast_frozen_decisao_recommended",
    "r4g_fast_decisao_recommended",
    "r4f_frozen_decisao_recommended",
    "decisao",
]
INTERVENTION_CANDIDATES = [
    "exp014b_r4g_fast_frozen_intervention_pred",
    "exp014b_r4g_fast_intervention_pred",
    "exp014b_r4f_frozen_intervention_pred",
]
BLOCK_CANDIDATES = [
    "exp014b_r4g_fast_frozen_block_pred",
    "exp014b_r4g_fast_block_pred",
    "exp014b_r4f_frozen_block_pred",
]
ID_CANDIDATES = ["transaction_id", "cd_pix"]

MODEL_SCORE_TOKENS = [
    "lgbm",
    "if_percentile",
    "if_raw",
    "se_score",
    "beh_score",
    "topaz_risk_score",
    "score_final",
    "peso_total",
]
BIN_SUFFIXES = ("_bin", "_bucket")


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def find_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def choose_input_path() -> tuple[Path, bool]:
    if DEFAULT_INPUT.exists():
        return DEFAULT_INPUT, False
    return FALLBACK_INPUT, True


def is_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    converted = pd.to_numeric(series, errors="coerce")
    return bool(converted.notna().mean() >= 0.90)


def classify_column(col: str) -> str:
    c = col.lower()
    if c in {"is_fraud", "fraude", "target", "label", "tp_fraude", "is_fraud_runtime"}:
        return "label"
    if c in {"transaction_id", "cd_pix", "customer_id", "counterparty_id", "session_id"}:
        return "id"
    if c in {"event_datetime", "dt_pix", "data_pix", "dataset_created_at", "dataset_v3_created_at"}:
        return "metadata"
    if c in {"dataset_role", "source_dataset", "sample_strategy", "sample_weight", "temporal_split"}:
        return "metadata"
    if "decisao" in c or "intervention_pred" in c or "block_pred" in c:
        if c.startswith("exp014b_r3") or c.startswith("exp014b_r4") or c.startswith("r3") or c.startswith("r4"):
            return "policy_column"
        return "policy_column"
    if c.startswith("exp014") or c.startswith("runtime_") or c.startswith("r3") or c.startswith("r4"):
        return "experiment_legacy"
    if any(token in c for token in MODEL_SCORE_TOKENS):
        return "model_score"
    if c.endswith(BIN_SUFFIXES):
        return "bin_feature"
    if c.startswith("_") or c in {"idx", "rn", "_worker_id"}:
        return "diagnostic_only"
    return "raw_or_engineered_feature"


def load_feature_contract(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path.relative_to(PROJECT_ROOT)), "exists": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "exists": True,
        "version": data.get("version"),
        "n_features": data.get("n_features"),
        "features": data.get("features") or [],
        "core_features": data.get("core_features") or [],
        "extra_features": data.get("extra_features") or [],
        "graph_features": data.get("graph_features") or [],
    }


def load_model_feature_contracts() -> dict[str, object]:
    train_contract = load_feature_contract(
        PROJECT_ROOT / "backend" / "modelos" / "resultado_treino_lgbm_v3" / "lgbm_features.json"
    )
    runtime_contract = load_feature_contract(
        PROJECT_ROOT / "backend" / "artefatos" / "lgbm_features.json"
    )
    train_features = set(train_contract.get("features") or [])
    runtime_features = set(runtime_contract.get("features") or [])
    return {
        "train_contract": train_contract,
        "runtime_contract": runtime_contract,
        "features_only_in_train_contract": sorted(train_features - runtime_features),
        "features_only_in_runtime_contract": sorted(runtime_features - train_features),
        "same_feature_set": train_features == runtime_features,
    }


def build_minimal_replay_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for candidates in [
        ID_CANDIDATES,
        LABEL_CANDIDATES,
        ACTION_CANDIDATES,
        INTERVENTION_CANDIDATES,
        BLOCK_CANDIDATES,
    ]:
        found = find_first_existing(df, candidates)
        if found and found not in cols:
            cols.append(found)
    return cols


def build_taxonomy(
    df: pd.DataFrame,
    model_features: set[str],
    minimal_replay_cols: set[str],
) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        category = classify_column(col)
        rows.append(
            {
                "column": col,
                "category": category,
                "dtype": str(df[col].dtype),
                "is_numeric": bool(is_numeric_like(df[col])),
                "missing_rate": round(float(df[col].isna().mean()), 8),
                "n_unique": int(df[col].nunique(dropna=True)),
                "in_lgbm_feature_contract": col in model_features,
                "required_for_frozen_replay": col in minimal_replay_cols,
                "recommended_initial_role": recommend_role(col, category, model_features, minimal_replay_cols),
            }
        )
    return pd.DataFrame(rows)


def recommend_role(
    col: str,
    category: str,
    model_features: set[str],
    minimal_replay_cols: set[str],
) -> str:
    if col in minimal_replay_cols:
        return "keep_for_replay_contract"
    if col in model_features:
        return "keep_for_model_candidate"
    if category in {"id", "label", "metadata"}:
        return "keep_out_of_model_contract_only"
    if category in {"policy_column", "experiment_legacy", "diagnostic_only"}:
        return "drop_from_feature_engineering"
    if category in {"model_score", "bin_feature"}:
        return "diagnostic_or_policy_only"
    return "candidate_feature_review"


def missing_or_low_variance_candidates(taxonomy: pd.DataFrame) -> pd.DataFrame:
    feature_like = taxonomy["category"].eq("raw_or_engineered_feature")
    low_info = taxonomy["missing_rate"].ge(0.95) | taxonomy["n_unique"].le(1)
    out = taxonomy.loc[feature_like & low_info].copy()
    if out.empty:
        return pd.DataFrame(columns=["feature_a", "feature_b", "reason", "metric"])
    out["feature_a"] = out["column"]
    out["feature_b"] = ""
    out["reason"] = np.where(
        out["missing_rate"].ge(0.95),
        "missing_rate_ge_95pct",
        "near_zero_variance",
    )
    out["metric"] = np.where(
        out["missing_rate"].ge(0.95),
        out["missing_rate"],
        out["n_unique"],
    )
    return out[["feature_a", "feature_b", "reason", "metric"]]


def high_correlation_candidates(df: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    candidate_cols = taxonomy.loc[
        taxonomy["category"].eq("raw_or_engineered_feature")
        & taxonomy["is_numeric"]
        & taxonomy["missing_rate"].lt(0.50),
        "column",
    ].tolist()
    if len(candidate_cols) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "reason", "metric"])

    numeric = df[candidate_cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="spearman").abs()
    rows = []
    for i, col_a in enumerate(candidate_cols):
        for col_b in candidate_cols[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.notna(value) and value >= 0.98:
                rows.append(
                    {
                        "feature_a": col_a,
                        "feature_b": col_b,
                        "reason": "spearman_abs_corr_ge_0_98",
                        "metric": round(float(value), 8),
                    }
                )
    return pd.DataFrame(rows)


def redundancy_candidates(df: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    frames = [
        missing_or_low_variance_candidates(taxonomy),
        high_correlation_candidates(df, taxonomy),
    ]
    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        return pd.DataFrame(columns=["feature_a", "feature_b", "reason", "metric"])
    return out.sort_values(["reason", "metric"], ascending=[True, False]).reset_index(drop=True)


def keep_drop_recommendations(taxonomy: pd.DataFrame, redundancy: pd.DataFrame) -> pd.DataFrame:
    redundant = set(redundancy["feature_a"].dropna().astype(str))
    rows = []
    for row in taxonomy.to_dict(orient="records"):
        col = row["column"]
        role = row["recommended_initial_role"]
        if col in redundant and role == "candidate_feature_review":
            recommendation = "review_drop_or_replace"
        elif role in {"drop_from_feature_engineering", "diagnostic_or_policy_only"}:
            recommendation = "exclude_from_model_features"
        elif role == "keep_for_model_candidate":
            recommendation = "keep_until_ablation"
        elif role == "keep_for_replay_contract":
            recommendation = "keep_for_baseline_replay"
        elif role == "keep_out_of_model_contract_only":
            recommendation = "exclude_from_model_but_keep_contract"
        else:
            recommendation = "review_before_r5b_r5c"
        rows.append(
            {
                "column": col,
                "category": row["category"],
                "recommendation": recommendation,
                "reason": role,
                "missing_rate": row["missing_rate"],
                "n_unique": row["n_unique"],
                "in_lgbm_feature_contract": row["in_lgbm_feature_contract"],
            }
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    summary: dict[str, object],
    taxonomy: pd.DataFrame,
    contracts: dict[str, object],
    redundancy: pd.DataFrame,
) -> None:
    category_counts = taxonomy["category"].value_counts().to_dict()
    recommendation_counts = taxonomy["recommended_initial_role"].value_counts().to_dict()
    lines = [
        f"# {EXPERIMENT} - Minimal Feature Set and Redundancy Audit",
        "",
        "## Executive summary",
        f"- Baseline: `{BASELINE_EXPERIMENT}`",
        f"- Rows: `{summary['n_rows']}`",
        f"- Columns audited: `{summary['n_columns']}`",
        f"- Minimal frozen replay columns: `{summary['n_minimal_replay_columns']}`",
        f"- Redundancy candidates: `{len(redundancy)}`",
        f"- Train/runtime LGBM feature set equal: `{contracts['same_feature_set']}`",
        "",
        "## Column taxonomy",
    ]
    lines.extend([f"- `{k}`: {v}" for k, v in sorted(category_counts.items())])
    lines.extend(
        [
            "",
            "## Initial roles",
        ]
    )
    lines.extend([f"- `{k}`: {v}" for k, v in sorted(recommendation_counts.items())])
    lines.extend(
        [
            "",
            "## Main warning",
            "The predictions file has many policy and legacy experiment columns. They must not be treated as primary model features in R5B/R5C.",
            "",
            "## Required next use",
            "Use `05_feature_keep_drop_recommendations.csv` as the gate before adding relationship or receiver reputation features.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path, used_fallback = choose_input_path()
    if not input_path.exists():
        raise FileNotFoundError(
            f"Neither default nor fallback predictions exist: {DEFAULT_INPUT} | {FALLBACK_INPUT}"
        )

    df = pd.read_csv(input_path, low_memory=False)
    contracts = load_model_feature_contracts()
    model_features = set(contracts["train_contract"].get("features") or [])
    minimal_cols = build_minimal_replay_columns(df)
    taxonomy = build_taxonomy(df, model_features, set(minimal_cols))
    redundancy = redundancy_candidates(df, taxonomy)
    recommendations = keep_drop_recommendations(taxonomy, redundancy)

    summary = {
        "experiment": EXPERIMENT,
        "status": "DONE",
        "baseline_experiment": BASELINE_EXPERIMENT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_predictions_path": str(input_path),
        "used_fallback": used_fallback,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "n_minimal_replay_columns": len(minimal_cols),
        "n_lgbm_train_contract_features": len(model_features),
        "n_policy_or_legacy_columns": int(
            taxonomy["category"].isin(["policy_column", "experiment_legacy"]).sum()
        ),
        "n_raw_or_engineered_feature_columns": int(
            taxonomy["category"].eq("raw_or_engineered_feature").sum()
        ),
        "n_redundancy_candidates": int(len(redundancy)),
        "output_dir": str(OUTPUT_DIR),
    }

    replay_contract = {
        "minimal_replay_columns": minimal_cols,
        "purpose": "Columns needed to evaluate the frozen R4G action and intervention/block predictions already materialized in the predictions file.",
        "not_a_training_feature_set": True,
    }

    write_json(OUTPUT_DIR / "00_run_summary.json", summary)
    taxonomy.to_csv(OUTPUT_DIR / "01_column_taxonomy.csv", index=False, encoding="utf-8")
    write_json(OUTPUT_DIR / "02_model_feature_contract_audit.json", contracts)
    write_json(OUTPUT_DIR / "03_replay_minimal_columns.json", replay_contract)
    redundancy.to_csv(OUTPUT_DIR / "04_redundancy_candidates.csv", index=False, encoding="utf-8")
    recommendations.to_csv(
        OUTPUT_DIR / "05_feature_keep_drop_recommendations.csv",
        index=False,
        encoding="utf-8",
    )
    write_report(
        OUTPUT_DIR / "06_exp014b_r5a1_report.md",
        summary=summary,
        taxonomy=taxonomy,
        contracts=contracts,
        redundancy=redundancy,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
