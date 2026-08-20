#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R5B7 — Block Severity Shadow Model.

Treina um modelo shadow apenas no residual BLOQUEAR pós R5B5 para estimar risco
de fraude dentro da fila de bloqueio. O objetivo é testar uma camada dedicada de
severidade: casos BLOQUEAR com score muito baixo podem virar CONFIRMAR se o
limiar escolhido em validação não demover fraude.

Não sobrescreve artefatos produtivos.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp_014b_r5b2_tune_policy import LABELS, find_col, ints, metrics, pred_block, pred_intervention


EXPERIMENT = "EXP-014B-R5B7-BLOCK-SEVERITY-SHADOW"
SOURCE_EXPERIMENT = "EXP-014B-R5B5-TRUST-FEATURE-DEESCALATION"
INPUT_FILE = PROJECT_ROOT / "resultados" / "experimentos" / SOURCE_EXPERIMENT / "05_predictions_trust.csv"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / EXPERIMENT

BASE_ACTION_COL = "r5b5_trust_decisao"
FINAL_ACTION_COL = "r5b7_severity_decisao"
MOVE_COL = "exp014b_r5b7_severity_block_to_confirm"
SCORE_COL = "r5b7_block_severity_score"

TRUST_FEATURES = [
    "payer_history_strength_score",
    "receiver_reputation_score",
    "relationship_strength_score",
    "receiver_novelty_risk_score",
    "transaction_normality_score",
    "payer_receiver_trust_score",
    "trust_bucket",
    "receiver_rep_bucket",
    "relationship_bucket",
    "novelty_bucket",
]


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_action(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().str.strip()


def is_excluded_feature(col: str, label_col: str) -> bool:
    exact = {
        label_col,
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
        "sample_weight",
        "temporal_split",
        "window_start_date",
        "window_end_date",
        "dataset_created_at",
        "dataset_v3_created_at",
        "rn",
        "ds_chave_pix",
        "session_id",
        "primeira_data_envio_recebedor_180d",
        BASE_ACTION_COL,
        FINAL_ACTION_COL,
        MOVE_COL,
        SCORE_COL,
    }
    if col in exact:
        return True
    prefixes = (
        "exp014b_",
        "r5b4_",
        "r5b5_",
        "r5b6_",
        "r5b7_",
    )
    return col.startswith(prefixes)


def prepare_features(df: pd.DataFrame, feature_cols: list[str], encoders: dict[str, LabelEncoder] | None = None) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    out = df[feature_cols].copy()
    fitted = encoders or {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
            continue
        values = out[col].astype(str).fillna("<MISSING>")
        if encoders is None:
            le = LabelEncoder()
            out[col] = le.fit_transform(values)
            fitted[col] = le
        else:
            le = fitted[col]
            known = set(le.classes_)
            values = values.where(values.isin(known), "<MISSING>")
            if "<MISSING>" not in known:
                values = values.where(values != "<MISSING>", le.classes_[0])
            out[col] = le.transform(values)
    return out, fitted


def choose_validation_threshold(scores: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if len(scores) == 0:
        return {"threshold": None, "validation_normals_demoted": 0, "validation_frauds_demoted": 0}

    quantile_candidates = [
        float(x) for x in np.quantile(scores, np.linspace(0.01, 0.80, 160))
        if np.isfinite(x)
    ]
    exact_candidates = [float(x) for x in scores if np.isfinite(x)]
    candidates = sorted(set(quantile_candidates + exact_candidates))
    best: dict[str, Any] | None = None
    for th in candidates:
        move = scores <= th
        normals = int((move & (y == 0)).sum())
        frauds = int((move & (y == 1)).sum())
        if normals <= 0 or frauds > 0:
            continue
        row = {
            "threshold": float(th),
            "validation_normals_demoted": normals,
            "validation_frauds_demoted": frauds,
            "validation_total_demoted": int(move.sum()),
        }
        if best is None or normals > best["validation_normals_demoted"]:
            best = row
    return best or {"threshold": None, "validation_normals_demoted": 0, "validation_frauds_demoted": 0}


def split_counts(df: pd.DataFrame, y: np.ndarray, move: np.ndarray) -> pd.DataFrame:
    rows = []
    for split, grp in df.groupby("temporal_split", dropna=False):
        idx = grp.index.to_numpy()
        rows.append({
            "temporal_split": split,
            "n_rows": int(len(grp)),
            "demoted_normals": int((move[idx] & (y[idx] == 0)).sum()),
            "demoted_frauds": int((move[idx] & (y[idx] == 1)).sum()),
            "remaining_block_normals": int(((normalize_action(df.loc[idx, FINAL_ACTION_COL]) == "BLOQUEAR").to_numpy() & (y[idx] == 0)).sum()),
            "remaining_block_frauds": int(((normalize_action(df.loc[idx, FINAL_ACTION_COL]) == "BLOQUEAR").to_numpy() & (y[idx] == 1)).sum()),
        })
    return pd.DataFrame(rows).sort_values("temporal_split")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    label_col = find_col(df, LABELS)
    y = ints(df[label_col]).to_numpy()
    base_action = normalize_action(df[BASE_ACTION_COL])
    residual_block = base_action.eq("BLOQUEAR").to_numpy()
    split = df["temporal_split"].fillna("<MISSING>").astype(str).str.upper()

    train_mask = residual_block & split.eq("TRAIN").to_numpy()
    val_mask = residual_block & split.eq("VALIDATION").to_numpy()
    hold_mask = residual_block & split.eq("HOLDOUT").to_numpy()
    if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0:
        raise RuntimeError("Residual BLOQUEAR sem dados suficientes para treino/validação.")

    feature_cols = [
        c for c in df.columns
        if not is_excluded_feature(c, label_col)
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X_train, encoders = prepare_features(df.loc[train_mask], feature_cols)
    X_val, _ = prepare_features(df.loc[val_mask], feature_cols, encoders)
    X_hold, _ = prepare_features(df.loc[hold_mask], feature_cols, encoders)
    y_train = y[train_mask]
    y_val = y[val_mask]
    y_hold = y[hold_mask]

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    spw = n_neg / max(n_pos, 1)
    model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=1200,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=5,
        min_child_samples=15,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.8,
        reg_lambda=1.5,
        random_state=43,
        n_jobs=-1,
        scale_pos_weight=spw,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[early_stopping(stopping_rounds=100, verbose=False), log_evaluation(period=100)],
    )

    val_scores = model.predict_proba(X_val)[:, 1]
    hold_scores = model.predict_proba(X_hold)[:, 1]
    selected_threshold = choose_validation_threshold(val_scores, y_val)
    threshold = selected_threshold["threshold"]

    all_scores = np.full(len(df), np.nan, dtype=float)
    all_residual_idx = np.where(residual_block)[0]
    X_all_residual, _ = prepare_features(df.loc[residual_block], feature_cols, encoders)
    all_scores[all_residual_idx] = model.predict_proba(X_all_residual)[:, 1]

    selected_mask = np.zeros(len(df), dtype=bool)
    if threshold is not None:
        selected_mask = residual_block & np.isfinite(all_scores) & (all_scores <= float(threshold))

    final_action = base_action.copy()
    final_action.loc[selected_mask] = "CONFIRMAR"
    df[SCORE_COL] = all_scores
    df[MOVE_COL] = selected_mask.astype(int)
    df[FINAL_ACTION_COL] = final_action
    df["exp014b_r5b7_intervention_pred"] = pred_intervention(final_action)
    df["exp014b_r5b7_block_pred"] = pred_block(final_action)

    demoted_normals = int((selected_mask & (y == 0)).sum())
    demoted_frauds = int((selected_mask & (y == 1)).sum())
    non_train = ~split.eq("TRAIN").to_numpy()
    non_train_frauds = int((selected_mask & non_train & (y == 1)).sum())
    holdout_frauds = int((selected_mask & hold_mask & (y == 1)).sum())

    base_block_metrics = metrics(df[label_col], pred_block(base_action))
    final_block_metrics = metrics(df[label_col], df["exp014b_r5b7_block_pred"])
    final_intervention_metrics = metrics(df[label_col], df["exp014b_r5b7_intervention_pred"])

    validation_auc = roc_auc_score(y_val, val_scores) if len(set(y_val)) > 1 else None
    holdout_auc = roc_auc_score(y_hold, hold_scores) if len(set(y_hold)) > 1 else None
    validation_ap = average_precision_score(y_val, val_scores) if len(set(y_val)) > 1 else None
    holdout_ap = average_precision_score(y_hold, hold_scores) if len(set(y_hold)) > 1 else None

    importances = model.booster_.feature_importance(importance_type="gain")
    feature_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": importances,
        "is_trust_feature": [f in TRUST_FEATURES for f in feature_cols],
    }).sort_values("importance_gain", ascending=False)

    status = "PASS_R5B7_SEVERITY_SHADOW_ZERO_NONTRAIN_FRAUDS"
    if threshold is None or demoted_normals <= 0:
        status = "NO_R5B7_SAFE_THRESHOLD_FOUND"
    elif selected_threshold["validation_normals_demoted"] < 10:
        status = "NO_R5B7_MATERIAL_SAFE_THRESHOLD_FOUND"
    elif non_train_frauds > 0:
        status = "FAIL_R5B7_NONTRAIN_FRAUD_DEMOTED"

    summary = {
        "experiment": EXPERIMENT,
        "source_experiment": SOURCE_EXPERIMENT,
        "status": status,
        "all_pass": bool(status.startswith("PASS")),
        "elapsed_seconds": round(float(time.perf_counter() - t0), 2),
        "n_features": int(len(feature_cols)),
        "trust_features_present": {f: bool(f in feature_cols) for f in TRUST_FEATURES},
        "best_iteration": int(model.best_iteration_),
        "scale_pos_weight": float(spw),
        "selected_threshold": selected_threshold,
        "validation_auc": None if validation_auc is None else round(float(validation_auc), 6),
        "holdout_auc": None if holdout_auc is None else round(float(holdout_auc), 6),
        "validation_average_precision": None if validation_ap is None else round(float(validation_ap), 6),
        "holdout_average_precision": None if holdout_ap is None else round(float(holdout_ap), 6),
        "block_fp_demoted_to_confirm_incremental": demoted_normals,
        "block_tp_demoted_to_confirm_incremental": demoted_frauds,
        "non_train_frauds_demoted": non_train_frauds,
        "holdout_frauds_demoted": holdout_frauds,
        "remaining_block_normals": int(((final_action == "BLOQUEAR") & (y == 0)).sum()),
        "remaining_block_frauds": int(((final_action == "BLOQUEAR") & (y == 1)).sum()),
        "remaining_approve_frauds": int(((final_action == "APROVAR") & (y == 1)).sum()),
        "base_block_metrics": base_block_metrics,
        "final_block_metrics": final_block_metrics,
        "final_intervention_metrics": final_intervention_metrics,
    }

    write_json(OUT_DIR / "00_run_summary.json", summary)
    write_json(OUT_DIR / "01_feature_contract.json", {"features": feature_cols, "trust_features": TRUST_FEATURES})
    feature_importance.to_csv(OUT_DIR / "02_feature_importance.csv", index=False)
    split_counts(df, y, selected_mask).to_csv(OUT_DIR / "03_split_counts.csv", index=False)
    df.loc[residual_block, [
        "transaction_id",
        label_col,
        "temporal_split",
        BASE_ACTION_COL,
        FINAL_ACTION_COL,
        MOVE_COL,
        SCORE_COL,
    ]].to_csv(OUT_DIR / "04_residual_block_scores.csv", index=False)
    df.to_csv(OUT_DIR / "05_predictions_severity.csv", index=False)
    joblib.dump(model, OUT_DIR / "06_model_block_severity_shadow.joblib")
    joblib.dump(encoders, OUT_DIR / "07_label_encoders_block_severity_shadow.joblib")

    trust_importance = feature_importance[feature_importance["is_trust_feature"]].copy()
    report = f"""# {EXPERIMENT} — Modelo shadow de severidade BLOQUEAR

## Resultado executivo
- Status: `{summary['status']}`
- Features totais: `{summary['n_features']}`
- Melhor iteração: `{summary['best_iteration']}`
- Threshold selecionado em validação: `{threshold if threshold is not None else 'N/A'}`
- Normais movidos de BLOQUEAR para CONFIRMAR: `{summary['block_fp_demoted_to_confirm_incremental']}`
- Fraudes movidas de BLOQUEAR para CONFIRMAR: `{summary['block_tp_demoted_to_confirm_incremental']}`
- Fraudes não-treino movidas: `{summary['non_train_frauds_demoted']}`
- Normais restantes em BLOQUEAR: `{summary['remaining_block_normals']}`
- Fraudes restantes em BLOQUEAR: `{summary['remaining_block_frauds']}`
- Fraudes restantes em APROVAR: `{summary['remaining_approve_frauds']}`

## Métricas do score no residual BLOQUEAR
```json
{json.dumps({k: summary[k] for k in ['validation_auc', 'holdout_auc', 'validation_average_precision', 'holdout_average_precision']}, ensure_ascii=False, indent=2)}
```

## Métricas finais de BLOQUEAR
```json
{json.dumps(final_block_metrics, ensure_ascii=False, indent=2)}
```

## Suporte por split
{split_counts(df, y, selected_mask).to_markdown(index=False)}

## Top features
{feature_importance.head(20).to_markdown(index=False)}

## Features de trust no modelo
{trust_importance.to_markdown(index=False)}

## Decisão técnica
Este é um experimento shadow e não altera artefatos produtivos. Um resultado
promocionável exige zero fraude demovida fora de treino e ganho material de
redução de falso bloqueio sobre o R5B5.
"""
    (OUT_DIR / "08_exp014b_r5b7_block_severity_shadow_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
