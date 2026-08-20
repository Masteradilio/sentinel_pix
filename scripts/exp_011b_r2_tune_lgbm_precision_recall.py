#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-011B-R2 — LGBM Precision/Recall Tuning Grid

Objetivo:
  Retreinar múltiplas configurações LGBM shadow, buscando melhorar o trade-off
  entre recall e falsos positivos no dataset enriquecido.

Estratégia:
  - Reaproveita funções do exp_011b_train_lgbm_vnext_shadow.py.
  - Testa pesos positivos menores/maiores e regularizações diferentes.
  - Para cada modelo, escolhe threshold pela validação sob políticas operacionais:
      * precision >= 50% e FPR <= 1%
      * FPR <= 0,5%
      * FPR <= 1%
      * melhor F1
  - Seleciona campeão por validação, mas reporta holdout.

Saídas:
  resultados/experimentos/EXP-011B-R2/
    00_run_summary.json
    01_model_policy_comparison.csv
    02_best_metrics_by_split.csv
    03_best_threshold_sweep_validation.csv
    04_best_threshold_sweep_holdout.csv
    05_best_feature_importance.csv
    06_best_predictions_holdout.csv
    07_best_false_negatives_holdout.csv
    08_best_false_positives_holdout.csv
    09_recommendation.md
    10_training_grid.json

  backend/artefatos_candidatos/exp011b_r2_lgbm_tuned/
"""

from __future__ import annotations

import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_SCRIPT = PROJECT_ROOT / "scripts" / "exp_011b_train_lgbm_vnext_shadow.py"
INPUT_PATH = PROJECT_ROOT / "dados" / "hmo_ml_tb_pix_dataset_v2_180d_v1_enriched.csv"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-011B-R2"
CANDIDATE_DIR = PROJECT_ROOT / "backend" / "artefatos_candidatos" / "exp011b_r2_lgbm_tuned"


def load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Script base não encontrado: {BASE_SCRIPT}. "
            "Salve antes o exp_011b_train_lgbm_vnext_shadow.py em scripts/."
        )
    spec = importlib.util.spec_from_file_location("exp011b_base", BASE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sample_weights(y: pd.Series, base_weight: pd.Series, pos_multiplier: float) -> np.ndarray:
    y_arr = y.astype(int).values
    w = pd.to_numeric(base_weight, errors="coerce").fillna(1.0).clip(lower=0.05, upper=10.0).astype(float).values
    w = np.where(y_arr == 1, w * pos_multiplier, w)
    return w


def select_thresholds(base, y_valid: np.ndarray, p_valid: np.ndarray, y_hold: np.ndarray, p_hold: np.ndarray) -> pd.DataFrame:
    val = base.threshold_sweep(y_valid, p_valid)
    hold = base.threshold_sweep(y_hold, p_hold)

    rows = []

    def nearest_hold(th: float) -> pd.Series:
        return hold.loc[(hold["threshold"] - th).abs().idxmin()]

    def add(name: str, sub: pd.DataFrame, sort_cols: list[str], ascending: list[bool]):
        if sub.empty:
            return
        v = sub.sort_values(sort_cols, ascending=ascending).iloc[0]
        h = nearest_hold(float(v["threshold"]))
        rows.append({
            "policy": name,
            "threshold": float(v["threshold"]),
            "val_tp": int(v["tp"]),
            "val_fp": int(v["fp"]),
            "val_fn": int(v["fn"]),
            "val_tn": int(v["tn"]),
            "val_precision": float(v["precision"]),
            "val_recall": float(v["recall"]),
            "val_f1": float(v["f1"]),
            "val_fpr": float(v["fpr"]),
            "holdout_tp": int(h["tp"]),
            "holdout_fp": int(h["fp"]),
            "holdout_fn": int(h["fn"]),
            "holdout_tn": int(h["tn"]),
            "holdout_precision": float(h["precision"]),
            "holdout_recall": float(h["recall"]),
            "holdout_f1": float(h["f1"]),
            "holdout_fpr": float(h["fpr"]),
        })

    add("BEST_F1", val, ["f1", "recall", "precision"], [False, False, False])
    add("PRECISION_GE_50_FPR_LE_1PCT", val[(val["precision"] >= 0.50) & (val["fpr"] <= 0.01)], ["f1", "recall"], [False, False])
    add("PRECISION_GE_70_FPR_LE_1PCT", val[(val["precision"] >= 0.70) & (val["fpr"] <= 0.01)], ["f1", "recall"], [False, False])
    add("FPR_LE_05PCT_MAX_RECALL", val[val["fpr"] <= 0.005], ["recall", "f1", "precision"], [False, False, False])
    add("FPR_LE_1PCT_MAX_RECALL", val[val["fpr"] <= 0.01], ["recall", "f1", "precision"], [False, False, False])
    add("FP_LE_100_BEST_F1", val[val["fp"] <= 100], ["f1", "recall"], [False, False])
    add("FP_LE_250_BEST_F1", val[val["fp"] <= 250], ["f1", "recall"], [False, False])

    return pd.DataFrame(rows), val, hold


def champion_score(row: pd.Series) -> tuple:
    """
    Ranking primário por validação, com segurança operacional.
    Preferir:
      - precision validação >= 50%
      - FPR validação <= 1%
      - maior F1 validação
      - maior recall validação
    """
    safe = int((row["val_precision"] >= 0.50) and (row["val_fpr"] <= 0.01))
    very_safe = int((row["val_precision"] >= 0.70) and (row["val_fpr"] <= 0.005))
    return (
        very_safe,
        safe,
        float(row["val_f1"]),
        float(row["val_recall"]),
        -float(row["val_fpr"]),
        float(row["val_precision"]),
    )


def main() -> None:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    base = load_base_module()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input enriquecido não encontrado: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df = base.normalize_columns(df)

    train, valid, holdout = base.split_dataset(df)
    feature_cols, numeric_cols, categorical_cols = base.infer_feature_columns(df)

    X_train = train[feature_cols]
    y_train = train["is_fraud"].astype(int)
    X_valid = valid[feature_cols]
    y_valid = valid["is_fraud"].astype(int)
    X_hold = holdout[feature_cols]
    y_hold = holdout["is_fraud"].astype(int)

    preprocessor_template = base.build_preprocessor(numeric_cols, categorical_cols)

    configs = [
        {
            "config_id": "C01_conservative_low_weight",
            "pos_multiplier": 1.0,
            "params": dict(n_estimators=2500, learning_rate=0.02, num_leaves=31, max_depth=5, min_child_samples=100, subsample=0.85, colsample_bytree=0.75, reg_alpha=1.0, reg_lambda=3.0),
        },
        {
            "config_id": "C02_conservative_mid_weight",
            "pos_multiplier": 2.0,
            "params": dict(n_estimators=3000, learning_rate=0.018, num_leaves=31, max_depth=6, min_child_samples=80, subsample=0.85, colsample_bytree=0.80, reg_alpha=0.75, reg_lambda=2.0),
        },
        {
            "config_id": "C03_balanced_weight",
            "pos_multiplier": 4.0,
            "params": dict(n_estimators=3500, learning_rate=0.015, num_leaves=63, max_depth=7, min_child_samples=50, subsample=0.85, colsample_bytree=0.80, reg_alpha=0.50, reg_lambda=1.50),
        },
        {
            "config_id": "C04_precision_deep_regularized",
            "pos_multiplier": 2.0,
            "params": dict(n_estimators=3500, learning_rate=0.012, num_leaves=63, max_depth=7, min_child_samples=120, subsample=0.80, colsample_bytree=0.70, reg_alpha=2.0, reg_lambda=5.0),
        },
        {
            "config_id": "C05_recall_controlled",
            "pos_multiplier": 6.0,
            "params": dict(n_estimators=3500, learning_rate=0.015, num_leaves=63, max_depth=7, min_child_samples=60, subsample=0.85, colsample_bytree=0.80, reg_alpha=0.75, reg_lambda=2.0),
        },
    ]

    dump(configs, OUT_DIR / "10_training_grid.json")

    all_rows = []
    artifacts = {}

    for i, cfg in enumerate(configs, start=1):
        cid = cfg["config_id"]
        print("=" * 80)
        print(f"Treinando {cid} ({i}/{len(configs)})")
        print("=" * 80)

        pre = base.build_preprocessor(numeric_cols, categorical_cols)
        Xtr = pre.fit_transform(X_train)
        Xva = pre.transform(X_valid)
        Xho = pre.transform(X_hold)

        w_train = sample_weights(y_train, train["sample_weight"], cfg["pos_multiplier"])
        w_valid = pd.to_numeric(valid["sample_weight"], errors="coerce").fillna(1.0).astype(float).values

        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": 1.0,
            "verbose": -1,
            **cfg["params"],
        }

        model = LGBMClassifier(**params)
        model.fit(
            Xtr,
            y_train,
            sample_weight=w_train,
            eval_set=[(Xva, y_valid)],
            eval_sample_weight=[w_valid],
            eval_metric="average_precision",
            callbacks=[
                early_stopping(stopping_rounds=200, verbose=False),
                log_evaluation(period=0),
            ],
        )

        p_val = model.predict_proba(Xva)[:, 1]
        p_hold = model.predict_proba(Xho)[:, 1]

        policy_df, val_sweep, hold_sweep = select_thresholds(base, y_valid.values, p_val, y_hold.values, p_hold)
        policy_df["config_id"] = cid
        policy_df["pos_multiplier"] = cfg["pos_multiplier"]
        policy_df["best_iteration"] = int(getattr(model, "best_iteration_", 0) or 0)
        policy_df["val_roc_auc"] = float(roc_auc_score(y_valid, p_val))
        policy_df["val_average_precision"] = float(average_precision_score(y_valid, p_val))
        policy_df["holdout_roc_auc"] = float(roc_auc_score(y_hold, p_hold))
        policy_df["holdout_average_precision"] = float(average_precision_score(y_hold, p_hold))

        all_rows.append(policy_df)

        # Salvar artefatos temporários de todos os modelos
        model_path = OUT_DIR / f"model_{cid}.joblib"
        pre_path = OUT_DIR / f"preprocessor_{cid}.joblib"
        joblib.dump(model, model_path)
        joblib.dump(pre, pre_path)

        artifacts[cid] = {
            "model_path": str(model_path),
            "preprocessor_path": str(pre_path),
            "params": params,
            "pos_multiplier": cfg["pos_multiplier"],
            "feature_cols": feature_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
        }

        val_sweep.to_csv(OUT_DIR / f"threshold_sweep_validation_{cid}.csv", index=False)
        hold_sweep.to_csv(OUT_DIR / f"threshold_sweep_holdout_{cid}.csv", index=False)

    comparison = pd.concat(all_rows, ignore_index=True)
    # Ranking por segurança operacional e métrica de validação.
    comparison["_rank"] = comparison.apply(champion_score, axis=1)
    comparison = comparison.sort_values("_rank", ascending=False).drop(columns=["_rank"]).reset_index(drop=True)
    comparison.to_csv(OUT_DIR / "01_model_policy_comparison.csv", index=False)

    best = comparison.iloc[0].to_dict()
    best_cid = best["config_id"]
    best_threshold = float(best["threshold"])

    # Carregar campeão para artefatos finais
    best_model = joblib.load(artifacts[best_cid]["model_path"])
    best_pre = joblib.load(artifacts[best_cid]["preprocessor_path"])

    Xtr = best_pre.transform(X_train)
    Xva = best_pre.transform(X_valid)
    Xho = best_pre.transform(X_hold)
    p_train = best_model.predict_proba(Xtr)[:, 1]
    p_val = best_model.predict_proba(Xva)[:, 1]
    p_hold = best_model.predict_proba(Xho)[:, 1]

    # Sweeps do campeão
    val_sweep_best = base.threshold_sweep(y_valid.values, p_val)
    hold_sweep_best = base.threshold_sweep(y_hold.values, p_hold)
    val_sweep_best.to_csv(OUT_DIR / "03_best_threshold_sweep_validation.csv", index=False)
    hold_sweep_best.to_csv(OUT_DIR / "04_best_threshold_sweep_holdout.csv", index=False)

    # Métricas por split campeão
    metric_rows = []
    for split, y, p in [("TRAIN", y_train.values, p_train), ("VALIDATION", y_valid.values, p_val), ("HOLDOUT", y_hold.values, p_hold)]:
        m = base.evaluate_threshold(y, p, best_threshold, label=split)
        m["temporal_split"] = split
        m["threshold"] = best_threshold
        m["config_id"] = best_cid
        metric_rows.append(m)
    metrics_best = pd.DataFrame(metric_rows)
    metrics_best.to_csv(OUT_DIR / "02_best_metrics_by_split.csv", index=False)

    # Feature importance campeão
    feature_names = base.get_feature_names(best_pre)
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance_gain": best_model.booster_.feature_importance(importance_type="gain"),
        "importance_split": best_model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    imp.to_csv(OUT_DIR / "05_best_feature_importance.csv", index=False)

    # Predições e erros holdout campeão
    pred_hold = holdout.copy()
    pred_hold["lgbm_r2_score"] = p_hold
    pred_hold["lgbm_r2_pred"] = (p_hold >= best_threshold).astype(int)
    pred_hold.to_csv(OUT_DIR / "06_best_predictions_holdout.csv", index=False)
    pred_hold[(pred_hold["is_fraud"] == 1) & (pred_hold["lgbm_r2_pred"] == 0)].to_csv(OUT_DIR / "07_best_false_negatives_holdout.csv", index=False)
    pred_hold[(pred_hold["is_fraud"] == 0) & (pred_hold["lgbm_r2_pred"] == 1)].to_csv(OUT_DIR / "08_best_false_positives_holdout.csv", index=False)

    # Copiar campeão para pasta candidata
    model_final = CANDIDATE_DIR / "model_lgbm_r2_tuned_shadow.joblib"
    pre_final = CANDIDATE_DIR / "preprocessor_lgbm_r2_tuned_shadow.joblib"
    thresholds_final = CANDIDATE_DIR / "threshold_policy_exp011b_r2.json"
    manifest_final = CANDIDATE_DIR / "manifest_exp011b_r2.json"
    joblib.dump(best_model, model_final)
    joblib.dump(best_pre, pre_final)
    dump(best, thresholds_final)

    elapsed = time.perf_counter() - t0
    summary = {
        "experiment": "EXP-011B-R2",
        "status": "DONE",
        "input_path": str(INPUT_PATH),
        "n_rows": int(len(df)),
        "n_fraud": int(df["is_fraud"].sum()),
        "n_normal": int((df["is_fraud"] == 0).sum()),
        "n_configs": len(configs),
        "best_config_id": best_cid,
        "best_policy": best.get("policy"),
        "best_threshold": best_threshold,
        "best_validation": {k: best[k] for k in best if str(k).startswith("val_")},
        "best_holdout": {k: best[k] for k in best if str(k).startswith("holdout_")},
        "elapsed_seconds": round(elapsed, 2),
        "candidate_dir": str(CANDIDATE_DIR),
        "model_path": str(model_final),
        "preprocessor_path": str(pre_final),
    }
    dump(summary, OUT_DIR / "00_run_summary.json")
    dump({
        "model_version": "exp011b_r2_lgbm_tuned_shadow",
        "status": "SHADOW_CANDIDATE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "threshold_policy": best,
        "notes": [
            "Não sobrescreve modelo produtivo.",
            "Próximo passo: E2E shadow com DecisionEngine usando candidato e threshold escolhido.",
        ],
    }, manifest_final)

    md = []
    md.append("# EXP-011B-R2 — Recommendation")
    md.append("")
    md.append("## Campeão")
    md.append(f"- config_id: `{best_cid}`")
    md.append(f"- policy: `{best.get('policy')}`")
    md.append(f"- threshold: `{best_threshold}`")
    md.append("")
    md.append("## Holdout")
    for k in ["holdout_tp", "holdout_fp", "holdout_fn", "holdout_tn", "holdout_precision", "holdout_recall", "holdout_f1", "holdout_fpr", "holdout_roc_auc", "holdout_average_precision"]:
        md.append(f"- {k}: {best.get(k)}")
    md.append("")
    md.append("## Decisão")
    md.append("Comparar com EXP-011B-R1. Se o R2 superar R1 no holdout com FP aceitável, seguir para E2E shadow.")
    (OUT_DIR / "09_recommendation.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
