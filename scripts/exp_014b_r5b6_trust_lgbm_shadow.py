#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R5B6 — Trust Features LGBM Shadow Training.

Treina um LightGBM shadow com as features R5B5 de trust/reputação já integradas
ao core. Não sobrescreve backend/artefatos; todos os outputs ficam em
resultados/experimentos/EXP-014B-R5B6-TRUST-LGBM-SHADOW/.
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
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.preprocessing import create_trust_features


EXPERIMENT = "EXP-014B-R5B6-TRUST-LGBM-SHADOW"
DADOS_DIR = PROJECT_ROOT / "dados"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / EXPERIMENT

TRAIN_DATA = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_TRAIN.csv"
VAL_DATA = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_VALIDATION.csv"
HOLD_DATA = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_HOLDOUT.csv"
BASELINE_METRICS = PROJECT_ROOT / "backend" / "modelos" / "resultado_treino_lgbm" / "metricas_lgbm_canonical.json"

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


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.split(".")[-1] for c in df.columns]
    return df


def evaluate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "threshold": round(float(threshold), 6),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 6),
        "average_precision": round(float(average_precision_score(y_true, y_prob)), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "fpr": round(float(fp / max(fp + tn, 1)), 6),
    }


def find_best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.005, 0.96, 0.005):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = clean_columns(pd.read_csv(path, low_memory=False))
    return create_trust_features(df)


def load_baseline_metrics() -> dict[str, Any] | None:
    if not BASELINE_METRICS.exists():
        return None
    return json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print("=" * 80)
    print(f"{EXPERIMENT} — treino shadow LightGBM com trust features")
    print("=" * 80)

    df_train = load_split(TRAIN_DATA)
    df_val = load_split(VAL_DATA)
    df_hold = load_split(HOLD_DATA)

    exclude_cols = {
        "transaction_id", "cd_pix", "customer_id", "counterparty_id",
        "event_datetime", "dt_pix", "data_pix", "is_fraud",
        "dataset_role", "source_dataset", "sample_strategy", "sample_weight",
        "temporal_split", "window_start_date", "window_end_date",
        "dataset_created_at", "dataset_v3_created_at", "rn",
        "ds_chave_pix", "session_id", "primeira_data_envio_recebedor_180d",
    }
    feature_cols = sorted(list(set(df_train.columns) - exclude_cols))

    categorical_cols = [
        "ds_tipo_chave_norm", "periodo_dia", "value_band",
        "device_name", "app_version", "ip_address", "metodo_autenticacao",
        "trust_bucket", "receiver_rep_bucket", "relationship_bucket", "novelty_bucket",
    ]
    encoders: dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        if col not in df_train.columns:
            continue
        le = LabelEncoder()
        combined = pd.concat([df_train[col], df_val[col], df_hold[col]]).astype(str).fillna("missing")
        le.fit(combined)
        df_train[col] = le.transform(df_train[col].astype(str).fillna("missing"))
        df_val[col] = le.transform(df_val[col].astype(str).fillna("missing"))
        df_hold[col] = le.transform(df_hold[col].astype(str).fillna("missing"))
        encoders[col] = le

    X_train = df_train[feature_cols]
    y_train = df_train["is_fraud"].astype(int)
    X_val = df_val[feature_cols]
    y_val = df_val["is_fraud"].astype(int)
    X_hold = df_hold[feature_cols]
    y_hold = df_hold["is_fraud"].astype(int)

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    spw = n_neg / max(n_pos, 1)

    model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=2500,
        learning_rate=0.01,
        num_leaves=63,
        max_depth=7,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=spw,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[early_stopping(stopping_rounds=150, verbose=False), log_evaluation(period=200)],
    )

    p_train = model.predict_proba(X_train)[:, 1]
    p_val = model.predict_proba(X_val)[:, 1]
    p_hold = model.predict_proba(X_hold)[:, 1]
    best_th = find_best_threshold_by_f1(y_val.to_numpy(), p_val)

    metrics_train = evaluate_metrics(y_train.to_numpy(), p_train, best_th)
    metrics_val = evaluate_metrics(y_val.to_numpy(), p_val, best_th)
    metrics_hold = evaluate_metrics(y_hold.to_numpy(), p_hold, best_th)

    baseline = load_baseline_metrics()
    baseline_hold = baseline.get("holdout_metrics") if baseline else None
    holdout_delta = {}
    if baseline_hold:
        for key in ["roc_auc", "average_precision", "precision", "recall", "f1", "fpr", "tp", "fp", "fn"]:
            if key in baseline_hold and key in metrics_hold:
                holdout_delta[key] = round(float(metrics_hold[key]) - float(baseline_hold[key]), 6)

    importances = model.booster_.feature_importance(importance_type="gain")
    feature_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": importances,
        "is_trust_feature": [f in TRUST_FEATURES for f in feature_cols],
    }).sort_values("importance_gain", ascending=False)

    run_summary = {
        "experiment": EXPERIMENT,
        "status": "DONE_R5B6_TRUST_LGBM_SHADOW",
        "elapsed_seconds": round(float(time.perf_counter() - t0), 2),
        "n_features": int(len(feature_cols)),
        "trust_features_present": {f: bool(f in feature_cols) for f in TRUST_FEATURES},
        "best_iteration": int(model.best_iteration_),
        "scale_pos_weight": float(spw),
        "best_threshold": float(best_th),
        "train_metrics": metrics_train,
        "validation_metrics": metrics_val,
        "holdout_metrics": metrics_hold,
        "baseline_holdout_metrics": baseline_hold,
        "holdout_delta_vs_canonical": holdout_delta,
        "promotion_hint": {
            "f1_improved": bool(holdout_delta.get("f1", 0) > 0),
            "average_precision_improved": bool(holdout_delta.get("average_precision", 0) > 0),
            "recall_improved": bool(holdout_delta.get("recall", 0) > 0),
            "fpr_under_1pct": bool(metrics_hold["fpr"] <= 0.01),
        },
    }

    (OUT_DIR / "00_run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "01_feature_contract.json").write_text(
        json.dumps({"features": feature_cols, "trust_features": TRUST_FEATURES}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    feature_importance.to_csv(OUT_DIR / "02_feature_importance.csv", index=False)
    pd.DataFrame({
        "transaction_id": df_hold["transaction_id"].values,
        "is_fraud": y_hold.values,
        "lgbm_trust_score": p_hold,
        "lgbm_trust_pred": (p_hold >= best_th).astype(int),
    }).to_csv(OUT_DIR / "03_holdout_predictions.csv", index=False)
    joblib.dump(model, OUT_DIR / "04_model_lightgbm_trust_shadow.joblib")
    joblib.dump(encoders, OUT_DIR / "05_label_encoders_trust_shadow.joblib")

    trust_importance = feature_importance[feature_importance["is_trust_feature"]].copy()
    report = f"""# {EXPERIMENT} — LGBM shadow com trust features

## Resultado executivo
- Status: `{run_summary['status']}`
- Features totais: `{run_summary['n_features']}`
- Melhor iteração: `{run_summary['best_iteration']}`
- Threshold F1 validação: `{best_th:.4f}`

## Holdout shadow
```json
{json.dumps(metrics_hold, ensure_ascii=False, indent=2)}
```

## Delta vs LGBM canônico atual
```json
{json.dumps(holdout_delta, ensure_ascii=False, indent=2)}
```

## Importância das features de trust
{trust_importance.to_markdown(index=False)}

## Decisão técnica
Este experimento é shadow e não substitui artefatos produtivos. Se AP/F1/recall
melhorarem sem violar FPR<=1%, as features de trust devem ser consideradas para
o próximo contrato canônico do LGBM.
"""
    (OUT_DIR / "06_exp014b_r5b6_trust_lgbm_shadow_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
