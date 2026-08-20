#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXP-014B-R5B20 - Retrain and audit isolated LGBM on MAF v3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "EXP-014B-R5B20-LGBM-RETRAIN-ISOLATED"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / EXPERIMENT
TRAIN = PROJECT_ROOT / "dados" / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_TRAIN.csv"
VALIDATION = PROJECT_ROOT / "dados" / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_VALIDATION.csv"
HOLDOUT = PROJECT_ROOT / "dados" / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_HOLDOUT.csv"
R5B18 = (
    PROJECT_ROOT
    / "resultados"
    / "experimentos"
    / "EXP-014B-R5B18-E2E-FROZEN-CONTRACT-HOMOLOGATION"
    / "01_vectorized_contract_predictions.csv"
)
LABEL_COL = "is_fraud"
ACTION_COL = "r5b18_e2e_contract_decisao"
FN_BUDGET_FULL = 10

DROP_COLS = {
    "transaction_id",
    "cd_pix",
    "customer_id",
    "counterparty_id",
    "event_datetime",
    "dt_pix",
    "data_pix",
    "dataset_role",
    "source_dataset",
    "sample_strategy",
    "temporal_split",
    "window_start_date",
    "window_end_date",
    "dataset_created_at",
    "dataset_v3_created_at",
    "primeira_data_envio_recebedor_180d",
    "sample_weight",
    "rn",
    LABEL_COL,
}


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ints(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def metrics(y_true: pd.Series, pred: pd.Series) -> dict[str, Any]:
    y = ints(y_true)
    p = ints(pred)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(float(precision), 8),
        "recall": round(float(recall), 8),
        "f1": round(float(f1), 8),
        "fpr": round(float(fpr), 8),
    }


def intervention_from_action(action: pd.Series) -> pd.Series:
    return action.fillna("").astype(str).str.upper().str.strip().isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df[LABEL_COL] = ints(df[LABEL_COL])
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in DROP_COLS:
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        cols.append(col)
    return cols


def encode_frames(frames: list[pd.DataFrame], cols: list[str]) -> tuple[list[pd.DataFrame], list[str]]:
    encoded = []
    cat_cols = []
    for col in cols:
        if any(frame[col].dtype == "object" or str(frame[col].dtype).startswith("str") for frame in frames):
            cat_cols.append(col)
    categories = {}
    for col in cat_cols:
        values = pd.concat([frame[col].fillna("<MISSING>").astype(str) for frame in frames], ignore_index=True).unique()
        categories[col] = {value: idx for idx, value in enumerate(values)}

    for frame in frames:
        out = pd.DataFrame(index=frame.index)
        for col in cols:
            if col in cat_cols:
                out[col] = frame[col].fillna("<MISSING>").astype(str).map(categories[col]).fillna(-1).astype("int32")
            else:
                out[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-999.0)
        encoded.append(out)
    return encoded, cat_cols


def fn_budget_for(y: pd.Series) -> int:
    return max(1, int(round(FN_BUDGET_FULL * int(y.sum()) / 1465)))


def threshold_sweep(y_true: pd.Series, score: np.ndarray, fn_budget: int) -> pd.DataFrame:
    thresholds = np.unique(np.nanquantile(score, np.linspace(0.0, 1.0, 1501)))
    rows = []
    for thr in thresholds:
        pred = pd.Series(score >= float(thr), index=y_true.index).astype(int)
        m = metrics(y_true, pred)
        rows.append({"threshold": float(thr), "fn_budget": fn_budget, **m})
    sweep = pd.DataFrame(rows)
    eligible = sweep[sweep["fn"] <= fn_budget].copy()
    if eligible.empty:
        return sweep.sort_values(["fn", "precision"], ascending=[True, False])
    eligible["sort_key"] = eligible["precision"] + eligible["f1"]
    best = eligible.sort_values(["sort_key", "fpr"], ascending=[False, True]).head(1)
    return pd.concat([best.assign(is_selected=True), sweep.assign(is_selected=False)], ignore_index=True)


def baseline_metrics_for(scope: pd.DataFrame) -> dict[str, Any]:
    r5b18 = pd.read_csv(R5B18, low_memory=False)
    merged = scope[["transaction_id", LABEL_COL]].merge(
        r5b18[["transaction_id", ACTION_COL]],
        on="transaction_id",
        how="left",
    )
    return metrics(merged[LABEL_COL], intervention_from_action(merged[ACTION_COL]))


def ratios(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "precision_ratio": round(float(candidate["precision"] / baseline["precision"]), 8) if baseline["precision"] else 0.0,
        "recall_ratio": round(float(candidate["recall"] / baseline["recall"]), 8) if baseline["recall"] else 0.0,
        "f1_ratio": round(float(candidate["f1"] / baseline["f1"]), 8) if baseline["f1"] else 0.0,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_split(TRAIN)
    val = load_split(VALIDATION)
    holdout = load_split(HOLDOUT)
    full = pd.concat([train, val, holdout], ignore_index=True)
    cols = feature_columns(train)
    (x_train, x_val, x_holdout, x_full), cat_cols = encode_frames([train, val, holdout, full], cols)
    y_train = train[LABEL_COL]
    y_val = val[LABEL_COL]
    y_holdout = holdout[LABEL_COL]
    y_full = full[LABEL_COL]

    scale_pos_weight = (len(y_train) - int(y_train.sum())) / max(int(y_train.sum()), 1)
    configs = [
        {"num_leaves": 31, "learning_rate": 0.03, "n_estimators": 500, "scale_pos_weight": scale_pos_weight},
        {"num_leaves": 63, "learning_rate": 0.03, "n_estimators": 700, "scale_pos_weight": scale_pos_weight},
        {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 400, "scale_pos_weight": scale_pos_weight * 1.5},
        {"num_leaves": 15, "learning_rate": 0.03, "n_estimators": 700, "scale_pos_weight": scale_pos_weight * 2.0},
    ]

    rows = []
    best_payload: dict[str, Any] | None = None
    for i, cfg in enumerate(configs, start=1):
        model = lgb.LGBMClassifier(
            objective="binary",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            **cfg,
        )
        sample_weight = pd.to_numeric(train.get("sample_weight", pd.Series(1.0, index=train.index)), errors="coerce").fillna(1.0)
        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(x_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(60, verbose=False)],
        )
        val_score = model.predict_proba(x_val)[:, 1]
        val_sweep = threshold_sweep(y_val, val_score, fn_budget_for(y_val))
        selected = val_sweep[val_sweep.get("is_selected", False) == True].iloc[0]
        threshold = float(selected["threshold"])
        scores = {
            "validation": model.predict_proba(x_val)[:, 1],
            "holdout": model.predict_proba(x_holdout)[:, 1],
            "full": model.predict_proba(x_full)[:, 1],
        }
        scopes = {"validation": val, "holdout": holdout, "full": full}
        metrics_by_scope = {}
        ratios_by_scope = {}
        for scope_name, score in scores.items():
            pred = pd.Series(score >= threshold).astype(int)
            m = metrics(scopes[scope_name][LABEL_COL], pred)
            b = baseline_metrics_for(scopes[scope_name])
            metrics_by_scope[scope_name] = m
            ratios_by_scope[scope_name] = ratios(m, b)
        payload = {
            "config_id": f"cfg_{i}",
            "config": cfg,
            "n_features": len(cols),
            "categorical_features": cat_cols,
            "validation_selected_threshold": threshold,
            "metrics_by_scope": metrics_by_scope,
            "ratios_to_r5b18_by_scope": ratios_by_scope,
        }
        full_ratio = ratios_by_scope["full"]
        score_key = full_ratio["precision_ratio"] + full_ratio["recall_ratio"] + full_ratio["f1_ratio"]
        rows.append({
            "config_id": payload["config_id"],
            "threshold": threshold,
            "full_precision_ratio": full_ratio["precision_ratio"],
            "full_recall_ratio": full_ratio["recall_ratio"],
            "full_f1_ratio": full_ratio["f1_ratio"],
            "full_fn": metrics_by_scope["full"]["fn"],
            "full_precision": metrics_by_scope["full"]["precision"],
            "full_recall": metrics_by_scope["full"]["recall"],
            "full_f1": metrics_by_scope["full"]["f1"],
            "score_key": score_key,
        })
        if best_payload is None or score_key > best_payload["score_key"]:
            best_payload = {"score_key": score_key, **payload}

    assert best_payload is not None
    best_full_ratios = best_payload["ratios_to_r5b18_by_scope"]["full"]
    target_gates = {
        "full_precision_recall_f1_ratio_gte_80pct": all(
            best_full_ratios[k] >= 0.8 for k in ["precision_ratio", "recall_ratio", "f1_ratio"]
        ),
        "full_fn_lte_10": best_payload["metrics_by_scope"]["full"]["fn"] <= FN_BUDGET_FULL,
    }
    summary = {
        "experiment": EXPERIMENT,
        "status": "PASS_R5B20_LGBM_RETRAIN_ISOLATED" if all(target_gates.values()) else "FAIL_R5B20_LGBM_RETRAIN_ISOLATED",
        "fn_budget_full": FN_BUDGET_FULL,
        "best_model": best_payload,
        "target_gates": target_gates,
    }
    write_json(OUT_DIR / "00_run_summary.json", summary)
    pd.DataFrame(rows).sort_values("score_key", ascending=False).to_csv(OUT_DIR / "01_model_search_summary.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
