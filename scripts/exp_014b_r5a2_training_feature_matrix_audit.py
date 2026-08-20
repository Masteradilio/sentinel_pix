"""
EXP-014B-R5A2 - Training Feature Matrix Audit.

Diagnostic-only script. It audits the real training matrix used by LGBM and
Isolation Forest without training models or changing production artifacts.
"""

from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPERIMENT = "EXP-014B-R5A2"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DATA = PROJECT_ROOT / "dados" / "base_treino_final.csv"
OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / EXPERIMENT

LGBM_TRAIN_FEATURES = (
    PROJECT_ROOT / "backend" / "modelos" / "resultado_treino_lgbm_v3" / "lgbm_features.json"
)
LGBM_RUNTIME_FEATURES = PROJECT_ROOT / "backend" / "artefatos" / "lgbm_features.json"
LGBM_FEATURE_IMPORTANCE = (
    PROJECT_ROOT / "backend" / "modelos" / "resultado_treino_lgbm_v3" / "feature_importance.csv"
)
IF_CONFIG = (
    PROJECT_ROOT / "backend" / "modelos" / "resultado_treino_if" / "isolation_forest_config.json"
)
IF_FEATURE_IMPORTANCE = (
    PROJECT_ROOT / "backend" / "modelos" / "resultado_treino_if" / "feature_importance.csv"
)
LGBM_TRAIN_SCRIPT = PROJECT_ROOT / "backend" / "modelos" / "train_lgbm_v3.py"
IF_TRAIN_SCRIPT = PROJECT_ROOT / "backend" / "modelos" / "train_isolation_forest_v2.py"

MAX_CORR_SAMPLE = 50_000
CORR_THRESHOLD = 0.98
HIGH_MISSING_THRESHOLD = 0.95
PSI_WARN_THRESHOLD = 0.20


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_lgbm_contract(path: Path) -> dict[str, object]:
    data = read_json(path)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "exists": path.exists(),
        "version": data.get("version"),
        "n_features": data.get("n_features"),
        "features": data.get("features") or [],
        "core_features": data.get("core_features") or [],
        "extra_features": data.get("extra_features") or [],
        "graph_features": data.get("graph_features") or [],
    }


def parse_literal_names(script_path: Path, names: Iterable[str]) -> dict[str, object]:
    if not script_path.exists():
        return {}
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    wanted = set(names)
    found: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        for name in target_names:
            if name in wanted:
                try:
                    found[name] = ast.literal_eval(node.value)
                except Exception:
                    pass
    return found


def load_if_features() -> dict[str, object]:
    config = read_json(IF_CONFIG)
    config_features = config.get("features") or []
    parsed = parse_literal_names(
        IF_TRAIN_SCRIPT,
        ["FEATURES_CORE", "FEATURES_INTERACTION", "FEATURES_ENGINEERED"],
    )
    declared = (
        list(parsed.get("FEATURES_CORE") or [])
        + list(parsed.get("FEATURES_INTERACTION") or [])
        + list(parsed.get("FEATURES_ENGINEERED") or [])
    )
    return {
        "config_path": str(IF_CONFIG.relative_to(PROJECT_ROOT)),
        "config_exists": IF_CONFIG.exists(),
        "config_features": config_features,
        "script_path": str(IF_TRAIN_SCRIPT.relative_to(PROJECT_ROOT)),
        "script_declared_features": declared,
        "features": config_features or declared,
    }


def is_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    converted = pd.to_numeric(series, errors="coerce")
    return bool(converted.notna().mean() >= 0.90)


def split_column(df: pd.DataFrame) -> str | None:
    if "temporal_split" in df.columns:
        return "temporal_split"
    if "dataset_role" in df.columns:
        return "dataset_role"
    return None


def make_temporal_groups(df: pd.DataFrame) -> pd.Series:
    split_col = split_column(df)
    if split_col:
        return df[split_col].fillna("UNKNOWN").astype(str)
    if "event_datetime" in df.columns:
        dt = pd.to_datetime(df["event_datetime"], errors="coerce")
        if dt.notna().sum() > 0:
            return pd.qcut(dt.rank(method="first"), q=3, labels=["EARLY", "MID", "LATE"]).astype(str)
    return pd.Series("ALL", index=df.index)


def feature_stats(df: pd.DataFrame, features: list[str], label_col: str = "is_fraud") -> pd.DataFrame:
    y = pd.to_numeric(df[label_col], errors="coerce") if label_col in df.columns else None
    rows = []
    for feature in features:
        present = feature in df.columns
        row = {
            "feature": feature,
            "present": present,
            "dtype": str(df[feature].dtype) if present else None,
            "is_numeric": bool(is_numeric_like(df[feature])) if present else False,
            "missing_rate": round(float(df[feature].isna().mean()), 8) if present else None,
            "n_unique": int(df[feature].nunique(dropna=True)) if present else 0,
            "fraud_mean": None,
            "normal_mean": None,
            "mean_delta_fraud_minus_normal": None,
        }
        if present and y is not None and is_numeric_like(df[feature]):
            x = pd.to_numeric(df[feature], errors="coerce")
            fraud = x[y.eq(1)]
            normal = x[y.eq(0)]
            if fraud.notna().any():
                row["fraud_mean"] = round(float(fraud.mean()), 8)
            if normal.notna().any():
                row["normal_mean"] = round(float(normal.mean()), 8)
            if row["fraud_mean"] is not None and row["normal_mean"] is not None:
                row["mean_delta_fraud_minus_normal"] = round(
                    float(row["fraud_mean"] - row["normal_mean"]),
                    8,
                )
        rows.append(row)
    return pd.DataFrame(rows)


def add_importance(audit: pd.DataFrame, importance_path: Path) -> pd.DataFrame:
    if not importance_path.exists():
        return audit
    imp = pd.read_csv(importance_path)
    feature_col = "feature" if "feature" in imp.columns else None
    if feature_col is None:
        return audit
    cols = [c for c in imp.columns if c != feature_col]
    return audit.merge(imp[[feature_col] + cols], on="feature", how="left")


def training_inventory(df: pd.DataFrame) -> pd.DataFrame:
    groups = make_temporal_groups(df)
    rows = []
    for col in df.columns:
        numeric = is_numeric_like(df[col])
        row = {
            "column": col,
            "dtype": str(df[col].dtype),
            "is_numeric": bool(numeric),
            "missing_rate": round(float(df[col].isna().mean()), 8),
            "n_unique": int(df[col].nunique(dropna=True)),
            "first_group_missing_rate": None,
            "last_group_missing_rate": None,
        }
        ordered_groups = list(pd.Series(groups).dropna().astype(str).unique())
        if ordered_groups:
            first = ordered_groups[0]
            last = ordered_groups[-1]
            row["first_group_missing_rate"] = round(float(df.loc[groups.eq(first), col].isna().mean()), 8)
            row["last_group_missing_rate"] = round(float(df.loc[groups.eq(last), col].isna().mean()), 8)
        rows.append(row)
    return pd.DataFrame(rows)


def high_correlation_candidates(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    available = [f for f in features if f in df.columns and is_numeric_like(df[f])]
    if len(available) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "reason", "metric"])
    sample = df[available]
    if len(sample) > MAX_CORR_SAMPLE:
        sample = sample.sample(MAX_CORR_SAMPLE, random_state=42)
    numeric = sample.apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="spearman").abs()
    rows = []
    for i, a in enumerate(available):
        for b in available[i + 1 :]:
            value = corr.loc[a, b]
            if pd.notna(value) and value >= CORR_THRESHOLD:
                rows.append(
                    {
                        "feature_a": a,
                        "feature_b": b,
                        "reason": "spearman_abs_corr_ge_0_98",
                        "metric": round(float(value), 8),
                    }
                )
    return pd.DataFrame(rows)


def low_information_candidates(stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in stats.to_dict(orient="records"):
        if not row["present"]:
            rows.append(
                {
                    "feature_a": row["feature"],
                    "feature_b": "",
                    "reason": "feature_missing_from_training_matrix",
                    "metric": None,
                }
            )
        elif row["missing_rate"] is not None and row["missing_rate"] >= HIGH_MISSING_THRESHOLD:
            rows.append(
                {
                    "feature_a": row["feature"],
                    "feature_b": "",
                    "reason": "missing_rate_ge_95pct",
                    "metric": row["missing_rate"],
                }
            )
        elif int(row["n_unique"]) <= 1:
            rows.append(
                {
                    "feature_a": row["feature"],
                    "feature_b": "",
                    "reason": "near_zero_variance",
                    "metric": row["n_unique"],
                }
            )
    return pd.DataFrame(rows)


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float | None:
    expected_num = pd.to_numeric(expected, errors="coerce").dropna()
    actual_num = pd.to_numeric(actual, errors="coerce").dropna()
    if expected_num.empty or actual_num.empty:
        return None
    try:
        edges = np.unique(np.quantile(expected_num, np.linspace(0, 1, bins + 1)))
    except Exception:
        return None
    if len(edges) <= 2:
        return None
    exp_counts = pd.cut(expected_num, bins=edges, include_lowest=True).value_counts(sort=False)
    act_counts = pd.cut(actual_num, bins=edges, include_lowest=True).value_counts(sort=False)
    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)
    exp_pct = exp_pct.replace(0, 1e-6)
    act_pct = act_pct.replace(0, 1e-6)
    return float(((act_pct - exp_pct) * np.log(act_pct / exp_pct)).sum())


def temporal_stability(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    groups = make_temporal_groups(df)
    unique_groups = list(pd.Series(groups).dropna().astype(str).unique())
    if len(unique_groups) < 2:
        return pd.DataFrame(columns=["feature", "reference_group", "comparison_group", "psi", "status"])
    reference = unique_groups[0]
    comparison = unique_groups[-1]
    rows = []
    for feature in features:
        if feature not in df.columns or not is_numeric_like(df[feature]):
            continue
        value = psi(df.loc[groups.eq(reference), feature], df.loc[groups.eq(comparison), feature])
        rows.append(
            {
                "feature": feature,
                "reference_group": reference,
                "comparison_group": comparison,
                "psi": round(value, 8) if value is not None else None,
                "status": (
                    "review_shift"
                    if value is not None and value >= PSI_WARN_THRESHOLD
                    else "ok_or_not_applicable"
                ),
            }
        )
    return pd.DataFrame(rows)


def recommendations(
    lgbm_audit: pd.DataFrame,
    if_audit: pd.DataFrame,
    redundancy: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    redundant = set(redundancy["feature_a"].dropna().astype(str))
    unstable = set(
        stability.loc[stability["status"].eq("review_shift"), "feature"].dropna().astype(str)
        if not stability.empty
        else []
    )
    combined = pd.concat(
        [
            lgbm_audit.assign(model="lgbm"),
            if_audit.assign(model="isolation_forest"),
        ],
        ignore_index=True,
    )
    rows = []
    for row in combined.to_dict(orient="records"):
        feature = row["feature"]
        if not row["present"]:
            rec = "fix_contract_or_drop"
            reason = "declared_but_missing"
        elif row["missing_rate"] is not None and row["missing_rate"] >= HIGH_MISSING_THRESHOLD:
            rec = "drop_or_replace"
            reason = "high_missing"
        elif int(row["n_unique"]) <= 1:
            rec = "drop_or_replace"
            reason = "near_zero_variance"
        elif feature in redundant:
            rec = "ablation_candidate"
            reason = "high_redundancy"
        elif feature in unstable:
            rec = "stability_review"
            reason = "temporal_shift"
        else:
            rec = "keep_until_ablation"
            reason = "passes_basic_audit"
        rows.append(
            {
                "model": row["model"],
                "feature": feature,
                "recommendation": rec,
                "reason": reason,
                "missing_rate": row["missing_rate"],
                "n_unique": row["n_unique"],
            }
        )
    return pd.DataFrame(rows)


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        f"# {EXPERIMENT} - Training Feature Matrix Audit",
        "",
        "## Executive summary",
        f"- Input: `{summary['input_data']}`",
        f"- Rows: `{summary['n_rows']}`",
        f"- Columns: `{summary['n_columns']}`",
        f"- LGBM declared features: `{summary['n_lgbm_features']}`",
        f"- LGBM missing from matrix: `{summary['n_lgbm_missing_features']}`",
        f"- IF declared features: `{summary['n_if_features']}`",
        f"- IF missing from matrix: `{summary['n_if_missing_features']}`",
        f"- Redundancy candidates: `{summary['n_redundancy_candidates']}`",
        f"- Temporal stability rows: `{summary['n_temporal_stability_rows']}`",
        "",
        "## Interpretation",
        "This audit is a feature-engineering gate. It does not train models and does not promote a new baseline.",
        "Use the recommendations artifact before adding R5B/R5C features.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_DATA.exists():
        raise FileNotFoundError(f"Training matrix not found: {INPUT_DATA}")

    df = pd.read_csv(INPUT_DATA, low_memory=False)
    lgbm_train = load_lgbm_contract(LGBM_TRAIN_FEATURES)
    lgbm_runtime = load_lgbm_contract(LGBM_RUNTIME_FEATURES)
    if_contract = load_if_features()

    lgbm_features = list(lgbm_train["features"])
    if_features = list(if_contract["features"])
    all_model_features = sorted(set(lgbm_features) | set(if_features))

    inventory = training_inventory(df)
    lgbm_audit = add_importance(feature_stats(df, lgbm_features), LGBM_FEATURE_IMPORTANCE)
    if_audit = add_importance(feature_stats(df, if_features), IF_FEATURE_IMPORTANCE)
    redundancy = pd.concat(
        [
            low_information_candidates(lgbm_audit),
            low_information_candidates(if_audit),
            high_correlation_candidates(df, all_model_features),
        ],
        ignore_index=True,
    ).drop_duplicates()
    stability = temporal_stability(df, all_model_features)
    recs = recommendations(lgbm_audit, if_audit, redundancy, stability)

    input_contract = {
        "input_data": str(INPUT_DATA),
        "input_exists": INPUT_DATA.exists(),
        "train_scripts": {
            "lgbm": str(LGBM_TRAIN_SCRIPT.relative_to(PROJECT_ROOT)),
            "isolation_forest": str(IF_TRAIN_SCRIPT.relative_to(PROJECT_ROOT)),
        },
        "lgbm_train_contract": lgbm_train,
        "lgbm_runtime_contract": lgbm_runtime,
        "isolation_forest_contract": if_contract,
        "split_column": split_column(df),
    }
    summary = {
        "experiment": EXPERIMENT,
        "status": "DONE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_data": str(INPUT_DATA),
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "n_lgbm_features": len(lgbm_features),
        "n_lgbm_missing_features": int((~lgbm_audit["present"]).sum()),
        "n_if_features": len(if_features),
        "n_if_missing_features": int((~if_audit["present"]).sum()),
        "n_redundancy_candidates": int(len(redundancy)),
        "n_temporal_stability_rows": int(len(stability)),
        "output_dir": str(OUTPUT_DIR),
    }

    write_json(OUTPUT_DIR / "00_run_summary.json", summary)
    write_json(OUTPUT_DIR / "01_input_contract.json", input_contract)
    inventory.to_csv(OUTPUT_DIR / "02_training_feature_inventory.csv", index=False, encoding="utf-8")
    lgbm_audit.to_csv(OUTPUT_DIR / "03_lgbm_feature_audit.csv", index=False, encoding="utf-8")
    if_audit.to_csv(OUTPUT_DIR / "04_if_feature_audit.csv", index=False, encoding="utf-8")
    redundancy.to_csv(OUTPUT_DIR / "05_redundancy_candidates.csv", index=False, encoding="utf-8")
    stability.to_csv(OUTPUT_DIR / "06_temporal_stability_by_feature.csv", index=False, encoding="utf-8")
    recs.to_csv(OUTPUT_DIR / "07_keep_drop_recommendations.csv", index=False, encoding="utf-8")
    write_report(OUTPUT_DIR / "08_exp014b_r5a2_report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
