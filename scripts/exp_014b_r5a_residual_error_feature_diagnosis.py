"""
EXP-014B-R5A - Residual Error Feature Diagnosis.

This script does not train models and does not modify the frozen baseline.
It profiles residual groups from EXP-014B-R4G-FAST-FROZEN predictions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPERIMENT = "EXP-014B-R5A"
BASELINE_EXPERIMENT = "EXP-014B-R4G-FAST-FROZEN"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "resultados"
    / "experimentos"
    / "EXP-014B-R4G-FAST-FROZEN"
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

SCORE_COLS = [
    "lgbm_r4_score",
    "score_final",
    "lgbm_raw",
    "lgbm_mapped",
    "peso_total",
    "if_percentile",
    "if_raw",
    "se_score",
    "beh_score",
    "behavioral_score",
    "topaz_risk_score",
    "exp014b_r3s_second_stage_score",
    "exp014b_r3u_receiver_relationship_trust_score",
]
SEGMENT_COLS = [
    "temporal_split",
    "event_month",
    "dataset_role",
    "source_dataset",
    "sample_strategy",
    "ds_tipo_chave_norm",
    "value_band",
    "periodo_dia",
    "lgbm_bin",
    "if_bin",
    "score_bin",
    "ratio_bin",
    "qtd_rec_bin",
    "valor_rec_bin",
    "module_quiet",
    "se_worst_pattern",
    "r3u_receiver_trust_bucket",
    "r3u_relationship_bucket",
]
CASE_COLUMNS_PRIORITY = [
    "transaction_id",
    "cd_pix",
    "customer_id",
    "counterparty_id",
    "event_datetime",
    "dt_pix",
    "vl_pix",
    "is_fraud",
    "r4g_fast_frozen_decisao_recommended",
    "exp014b_r4g_fast_frozen_intervention_pred",
    "exp014b_r4g_fast_frozen_block_pred",
] + SCORE_COLS + SEGMENT_COLS

ID_LIKE_TOKENS = [
    "transaction_id",
    "cd_pix",
    "customer_id",
    "counterparty_id",
    "session_id",
    "ip_address",
    "device_name",
]


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


def normalize_action(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def as_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def choose_input_path() -> tuple[Path, bool]:
    if DEFAULT_INPUT.exists():
        return DEFAULT_INPUT, False
    return FALLBACK_INPUT, True


def build_residual_groups(
    df: pd.DataFrame,
    label_col: str,
    action_col: str,
) -> dict[str, pd.Series]:
    y = as_int(df[label_col])
    action = normalize_action(df[action_col])
    return {
        "approve_fraud": action.eq("APROVAR") & y.eq(1),
        "approve_normal": action.eq("APROVAR") & y.eq(0),
        "block_fraud": action.eq("BLOQUEAR") & y.eq(1),
        "block_normal": action.eq("BLOQUEAR") & y.eq(0),
        "confirm_fraud": action.eq("CONFIRMAR") & y.eq(1),
        "confirm_normal": action.eq("CONFIRMAR") & y.eq(0),
    }


def numeric_summary(values: pd.Series) -> dict[str, float | int | None]:
    x = pd.to_numeric(values, errors="coerce")
    present = x.dropna()
    if present.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "max": None,
            "missing_rate": round(float(x.isna().mean()), 8),
        }
    q = present.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "count": int(present.shape[0]),
        "mean": round(float(present.mean()), 8),
        "std": round(float(present.std(ddof=1)), 8) if present.shape[0] > 1 else 0.0,
        "min": round(float(present.min()), 8),
        "p01": round(float(q.loc[0.01]), 8),
        "p05": round(float(q.loc[0.05]), 8),
        "p25": round(float(q.loc[0.25]), 8),
        "p50": round(float(q.loc[0.50]), 8),
        "p75": round(float(q.loc[0.75]), 8),
        "p95": round(float(q.loc[0.95]), 8),
        "p99": round(float(q.loc[0.99]), 8),
        "max": round(float(present.max()), 8),
        "missing_rate": round(float(x.isna().mean()), 8),
    }


def is_probably_numeric(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    converted = pd.to_numeric(series, errors="coerce")
    return bool(converted.notna().mean() >= 0.90)


def is_id_like(col: str) -> bool:
    c = col.lower()
    return any(token in c for token in ID_LIKE_TOKENS) or c.endswith("_id")


def classify_feature(col: str) -> str:
    c = col.lower()
    if "decisao" in c or c.endswith("_pred") or "intervention" in c or "block" in c:
        return "decision_policy"
    if "score" in c or "lgbm" in c or c.startswith("if_") or "peso" in c:
        return "score"
    if "recebedor" in c or "receiver" in c or "counterparty" in c:
        return "receiver_relationship"
    if "pagador" in c or "customer" in c or "payer" in c:
        return "payer"
    if "graph_" in c:
        return "graph"
    if c in {"hour", "periodo_dia", "event_datetime", "dt_pix", "data_pix"}:
        return "temporal"
    if "chave" in c or "key" in c:
        return "pix_key"
    return "other"


def feature_inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "feature_group": classify_feature(col),
                "is_numeric": bool(is_probably_numeric(df[col])),
                "missing_rate": round(float(df[col].isna().mean()), 8),
                "n_unique": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def residual_group_metrics(
    df: pd.DataFrame,
    label_col: str,
    action_col: str,
    intervention_col: str | None,
    block_col: str | None,
    groups: dict[str, pd.Series],
) -> dict[str, object]:
    y = as_int(df[label_col])
    action = normalize_action(df[action_col])
    result: dict[str, object] = {
        "counts": {name + "_count": int(mask.sum()) for name, mask in groups.items()},
        "by_action": {},
    }

    by_action = {}
    for act in ["APROVAR", "BLOQUEAR", "CONFIRMAR"]:
        idx = action.eq(act)
        yy = y[idx]
        by_action[act] = {
            "n_rows": int(idx.sum()),
            "n_frauds": int(yy.sum()),
            "n_normals": int(idx.sum() - yy.sum()),
            "fraud_rate": round(float(yy.mean()), 8) if idx.any() else 0.0,
        }
    result["by_action"] = by_action

    if intervention_col:
        pred = as_int(df[intervention_col])
        result["intervention_pred_counts"] = {
            "pred_0": int(pred.eq(0).sum()),
            "pred_1": int(pred.eq(1).sum()),
        }
    if block_col:
        pred = as_int(df[block_col])
        result["block_pred_counts"] = {
            "pred_0": int(pred.eq(0).sum()),
            "pred_1": int(pred.eq(1).sum()),
        }
    return result


def numeric_contrast_rows(
    df: pd.DataFrame,
    residual_mask: pd.Series,
    reference_mask: pd.Series,
    residual_name: str,
    reference_name: str,
    numeric_cols: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for col in numeric_cols:
        a = pd.to_numeric(df.loc[residual_mask, col], errors="coerce")
        b = pd.to_numeric(df.loc[reference_mask, col], errors="coerce")
        if a.notna().sum() == 0 and b.notna().sum() == 0:
            continue
        a_mean = float(a.mean()) if a.notna().any() else np.nan
        b_mean = float(b.mean()) if b.notna().any() else np.nan
        pooled = float(pd.concat([a, b]).std(ddof=1))
        rows.append(
            {
                "feature": col,
                "feature_type": "numeric",
                "value": "",
                "residual_group": residual_name,
                "reference_group": reference_name,
                "residual_count": int(a.notna().sum()),
                "reference_count": int(b.notna().sum()),
                "residual_mean": round(a_mean, 8) if not np.isnan(a_mean) else None,
                "reference_mean": round(b_mean, 8) if not np.isnan(b_mean) else None,
                "mean_delta": (
                    round(a_mean - b_mean, 8)
                    if not np.isnan(a_mean) and not np.isnan(b_mean)
                    else None
                ),
                "residual_p50": (
                    round(float(a.median()), 8) if a.notna().any() else None
                ),
                "reference_p50": (
                    round(float(b.median()), 8) if b.notna().any() else None
                ),
                "missing_rate_delta": round(float(a.isna().mean() - b.isna().mean()), 8),
                "standardized_mean_delta": (
                    round((a_mean - b_mean) / pooled, 8)
                    if pooled > 0 and not np.isnan(a_mean) and not np.isnan(b_mean)
                    else None
                ),
            }
        )
    return rows


def categorical_contrast_rows(
    df: pd.DataFrame,
    residual_mask: pd.Series,
    reference_mask: pd.Series,
    residual_name: str,
    reference_name: str,
    categorical_cols: list[str],
    top_n: int = 10,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for col in categorical_cols:
        residual = df.loc[residual_mask, col].fillna("<MISSING>").astype(str)
        reference = df.loc[reference_mask, col].fillna("<MISSING>").astype(str)
        if residual.empty or reference.empty:
            continue
        values = residual.value_counts().head(top_n).index.tolist()
        reference_counts = reference.value_counts()
        for value in values:
            residual_count = int((residual == value).sum())
            reference_count = int(reference_counts.get(value, 0))
            residual_freq = residual_count / max(len(residual), 1)
            reference_freq = reference_count / max(len(reference), 1)
            rows.append(
                {
                    "feature": col,
                    "feature_type": "categorical",
                    "value": value,
                    "residual_group": residual_name,
                    "reference_group": reference_name,
                    "residual_count": residual_count,
                    "reference_count": reference_count,
                    "residual_freq": round(float(residual_freq), 8),
                    "reference_freq": round(float(reference_freq), 8),
                    "freq_delta": round(float(residual_freq - reference_freq), 8),
                    "lift": (
                        round(float(residual_freq / reference_freq), 8)
                        if reference_freq > 0
                        else None
                    ),
                }
            )
    return rows


def feature_contrast(
    df: pd.DataFrame,
    residual_mask: pd.Series,
    reference_mask: pd.Series,
    residual_name: str,
    reference_name: str,
) -> pd.DataFrame:
    numeric_cols = [
        c
        for c in df.columns
        if not is_id_like(c) and is_probably_numeric(df[c])
    ]
    categorical_cols = [
        c
        for c in df.columns
        if not is_id_like(c)
        and not is_probably_numeric(df[c])
        and df[c].nunique(dropna=True) <= 50
    ]
    rows = numeric_contrast_rows(
        df, residual_mask, reference_mask, residual_name, reference_name, numeric_cols
    )
    rows.extend(
        categorical_contrast_rows(
            df, residual_mask, reference_mask, residual_name, reference_name, categorical_cols
        )
    )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [c for c in ["standardized_mean_delta", "freq_delta"] if c in out.columns]
    if sort_cols:
        helper = (
            out[sort_cols]
            .apply(pd.to_numeric, errors="coerce")
            .abs()
            .max(axis=1, skipna=True)
            .fillna(0)
        )
        out = out.assign(_sort_abs_delta=helper).sort_values(
            "_sort_abs_delta", ascending=False
        )
        out = out.drop(columns=["_sort_abs_delta"])
    return out.reset_index(drop=True)


def missingness_by_group(
    df: pd.DataFrame,
    groups: dict[str, pd.Series],
    columns: list[str],
) -> pd.DataFrame:
    rows = []
    for col in columns:
        for group_name, mask in groups.items():
            values = df.loc[mask, col]
            rows.append(
                {
                    "feature": col,
                    "group": group_name,
                    "n_rows": int(mask.sum()),
                    "missing_rate": (
                        round(float(values.isna().mean()), 8) if len(values) else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def score_distribution_by_group(
    df: pd.DataFrame,
    groups: dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []
    for col in [c for c in SCORE_COLS if c in df.columns]:
        for group_name, mask in groups.items():
            row = {"score": col, "group": group_name}
            row.update(numeric_summary(df.loc[mask, col]))
            rows.append(row)
    return pd.DataFrame(rows)


def top_segment_concentrations(
    df: pd.DataFrame,
    groups: dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []
    cols = [c for c in SEGMENT_COLS if c in df.columns]
    for col in cols:
        for group_name, mask in groups.items():
            values = df.loc[mask, col].fillna("<MISSING>").astype(str)
            if values.empty:
                continue
            counts = values.value_counts().head(10)
            for value, count in counts.items():
                rows.append(
                    {
                        "feature": col,
                        "group": group_name,
                        "value": value,
                        "count": int(count),
                        "share_in_group": round(float(count / len(values)), 8),
                    }
                )
    return pd.DataFrame(rows)


def has_any(cols: Iterable[str], required_tokens: Iterable[str]) -> bool:
    req = [token.lower() for token in required_tokens]
    for col in cols:
        c = col.lower()
        if all(token in c for token in req):
            return True
    return False


def infer_candidate_feature_gaps(columns: Iterable[str]) -> list[dict[str, object]]:
    cols = list(columns)
    gap_defs = [
        (
            "missing_pix_key_age",
            "No explicit PIX key age feature was found.",
            [("idade", "chave"), ("age", "key"), ("dias", "chave")],
            "Fase 3",
        ),
        (
            "missing_receiver_account_age",
            "No explicit receiver account age feature was found.",
            [("idade", "conta", "recebedor"), ("account", "age", "receiver")],
            "Fase 3",
        ),
        (
            "missing_strong_pair_history",
            "No strong payer-receiver historical relationship feature was found.",
            [
                ("qtd_pix_mesmo_recebedor",),
                ("valor_total_para_recebedor",),
                ("dias_desde_primeiro_envio_recebedor",),
                ("relationship", "trust"),
            ],
            "Fase 2",
        ),
        (
            "missing_receiver_reputation",
            "No direct receiver reputation score was found.",
            [("receiver", "reputation"), ("recebedor", "reputacao"), ("receiver", "reputable")],
            "Fase 3",
        ),
        (
            "missing_receiver_fraud_rate",
            "No historical receiver fraud rate feature was found.",
            [("taxa", "fraude", "recebedor"), ("fraud", "rate", "receiver")],
            "Fase 3",
        ),
        (
            "missing_receiver_temporal_stability",
            "No explicit receiver temporal stability score was found.",
            [("receiver", "stability"), ("recebedor", "estabilidade")],
            "Fase 3",
        ),
        (
            "missing_real_pair_recurrence",
            "No explicit real recurrence score for the payer-receiver pair was found.",
            [("recorrente", "recebedor"), ("relationship", "recurrent"), ("pair", "recurrence")],
            "Fase 2",
        ),
    ]

    gaps = []
    for gap_id, description, token_groups, phase in gap_defs:
        found = any(has_any(cols, tokens) for tokens in token_groups)
        if not found:
            gaps.append(
                {
                    "gap_id": gap_id,
                    "description": description,
                    "suggested_phase": phase,
                    "inferred_from_column_names": True,
                }
            )
    return gaps


def load_lgbm_feature_contract() -> dict[str, object]:
    paths = [
        PROJECT_ROOT / "backend" / "modelos" / "resultado_treino_lgbm_v3" / "lgbm_features.json",
        PROJECT_ROOT / "backend" / "artefatos" / "lgbm_features.json",
    ]
    contracts = []
    for path in paths:
        if not path.exists():
            contracts.append({"path": str(path.relative_to(PROJECT_ROOT)), "exists": False})
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        contracts.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "exists": True,
                "version": data.get("version"),
                "n_features": data.get("n_features"),
                "has_graph_features": bool(data.get("graph_features")),
                "features": data.get("features", []),
            }
        )
    return {"contracts": contracts}


def select_case_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in CASE_COLUMNS_PRIORITY if c in df.columns]
    for col in df.columns:
        if classify_feature(col) in {"receiver_relationship", "score"} and col not in cols:
            cols.append(col)
    return cols


def write_report(
    path: Path,
    input_path: Path,
    used_fallback: bool,
    contract: dict[str, object],
    metrics: dict[str, object],
    gaps: list[dict[str, object]],
) -> None:
    counts = metrics["counts"]
    lines = [
        f"# {EXPERIMENT} - Residual Error Feature Diagnosis",
        "",
        "## Executive summary",
        f"- Baseline: `{BASELINE_EXPERIMENT}`",
        f"- Input: `{input_path.relative_to(PROJECT_ROOT)}`",
        f"- Used fallback: `{used_fallback}`",
        f"- Rows: `{contract['n_rows']}`",
        "",
        "## Residual groups",
        f"- approve_fraud_count: `{counts['approve_fraud_count']}`",
        f"- approve_normal_count: `{counts['approve_normal_count']}`",
        f"- block_fraud_count: `{counts['block_fraud_count']}`",
        f"- block_normal_count: `{counts['block_normal_count']}`",
        f"- confirm_fraud_count: `{counts['confirm_fraud_count']}`",
        f"- confirm_normal_count: `{counts['confirm_normal_count']}`",
        "",
        "## Key comparisons",
        "- `05_feature_contrast_approve_fraud_vs_approve_normal.csv` compares frauds still approved against normal approvals.",
        "- `06_feature_contrast_block_normal_vs_block_fraud.csv` compares normal transactions blocked against frauds blocked.",
        "",
        "## Candidate feature gaps",
    ]
    if gaps:
        lines.extend([f"- `{g['gap_id']}`: {g['description']}" for g in gaps])
    else:
        lines.append("- No candidate gaps inferred from column names.")
    lines.extend(
        [
            "",
            "## Notes",
            "- This is diagnostic only; no model was trained.",
            "- Gap inference is based on column names, not data lineage proof.",
            "- Promote no new baseline from this artifact alone.",
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
    label_col = find_first_existing(df, LABEL_CANDIDATES)
    action_col = find_first_existing(df, ACTION_CANDIDATES)
    intervention_col = find_first_existing(df, INTERVENTION_CANDIDATES)
    block_col = find_first_existing(df, BLOCK_CANDIDATES)
    missing_required = [
        name
        for name, value in {"label_col": label_col, "action_col": action_col}.items()
        if value is None
    ]
    if missing_required:
        raise ValueError(f"Missing required contract columns: {missing_required}")

    assert label_col is not None
    assert action_col is not None

    groups = build_residual_groups(df, label_col=label_col, action_col=action_col)
    metrics = residual_group_metrics(
        df,
        label_col=label_col,
        action_col=action_col,
        intervention_col=intervention_col,
        block_col=block_col,
        groups=groups,
    )

    inv = feature_inventory(df)
    gaps = infer_candidate_feature_gaps(df.columns)
    case_cols = select_case_columns(df)
    score_cols = [c for c in SCORE_COLS if c in df.columns]

    approve_contrast = feature_contrast(
        df,
        groups["approve_fraud"],
        groups["approve_normal"],
        "approve_fraud",
        "approve_normal",
    )
    block_contrast = feature_contrast(
        df,
        groups["block_normal"],
        groups["block_fraud"],
        "block_normal",
        "block_fraud",
    )

    contract = {
        "experiment": EXPERIMENT,
        "baseline_experiment": BASELINE_EXPERIMENT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_predictions_path": str(input_path),
        "used_fallback": used_fallback,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "label_col": label_col,
        "action_col": action_col,
        "intervention_col": intervention_col,
        "block_col": block_col,
        "score_cols_present": score_cols,
        "score_cols_missing": [c for c in SCORE_COLS if c not in df.columns],
        "relationship_receiver_cols_present": inv.loc[
            inv["feature_group"].eq("receiver_relationship"), "column"
        ].tolist(),
        "lgbm_feature_contract": load_lgbm_feature_contract(),
        "missing": [
            name
            for name, value in {
                "intervention_col": intervention_col,
                "block_col": block_col,
            }.items()
            if value is None
        ],
        "contract_ok": True,
    }

    summary = {
        "experiment": EXPERIMENT,
        "status": "DONE",
        "baseline_experiment": BASELINE_EXPERIMENT,
        "input_predictions_path": str(input_path),
        "used_fallback": used_fallback,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "label_col": label_col,
        "action_col": action_col,
        "approve_fraud_count": metrics["counts"]["approve_fraud_count"],
        "block_normal_count": metrics["counts"]["block_normal_count"],
        "block_fraud_count": metrics["counts"]["block_fraud_count"],
        "approve_normal_count": metrics["counts"]["approve_normal_count"],
        "confirm_fraud_count": metrics["counts"]["confirm_fraud_count"],
        "confirm_normal_count": metrics["counts"]["confirm_normal_count"],
        "n_candidate_feature_gaps": len(gaps),
        "output_dir": str(OUTPUT_DIR),
    }

    write_json(OUTPUT_DIR / "00_run_summary.json", summary)
    write_json(OUTPUT_DIR / "01_input_contract.json", contract)
    write_json(OUTPUT_DIR / "02_residual_group_metrics.json", metrics)
    df.loc[groups["approve_fraud"], case_cols].to_csv(
        OUTPUT_DIR / "03_approve_fraud_cases.csv", index=False, encoding="utf-8"
    )
    df.loc[groups["block_normal"], case_cols].to_csv(
        OUTPUT_DIR / "04_block_normal_cases.csv", index=False, encoding="utf-8"
    )
    approve_contrast.to_csv(
        OUTPUT_DIR / "05_feature_contrast_approve_fraud_vs_approve_normal.csv",
        index=False,
        encoding="utf-8",
    )
    block_contrast.to_csv(
        OUTPUT_DIR / "06_feature_contrast_block_normal_vs_block_fraud.csv",
        index=False,
        encoding="utf-8",
    )
    write_json(OUTPUT_DIR / "07_candidate_feature_gaps.json", gaps)
    missingness_by_group(df, groups, list(df.columns)).to_csv(
        OUTPUT_DIR / "09_missingness_by_group.csv", index=False, encoding="utf-8"
    )
    score_distribution_by_group(df, groups).to_csv(
        OUTPUT_DIR / "10_score_distribution_by_group.csv",
        index=False,
        encoding="utf-8",
    )
    top_segment_concentrations(df, groups).to_csv(
        OUTPUT_DIR / "11_top_segment_concentrations.csv",
        index=False,
        encoding="utf-8",
    )
    inv.to_csv(OUTPUT_DIR / "12_feature_inventory.csv", index=False, encoding="utf-8")
    write_report(
        OUTPUT_DIR / "08_exp014b_r5a_report.md",
        input_path=input_path,
        used_fallback=used_fallback,
        contract=contract,
        metrics=metrics,
        gaps=gaps,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
