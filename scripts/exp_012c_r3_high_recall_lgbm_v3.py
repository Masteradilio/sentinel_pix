#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-012C-R3 — LGBM v3 High-Recall Sweep

Objetivo:
  Mudar explicitamente o objetivo do LGBM para RECALL-FIRST.

Contexto:
  A métrica prioritária do MVP/baseline histórico era recall acima de 0.90.
  O LGBM deve atuar como detector amplo de fraudes verdadeiras, aceitando
  mais falsos positivos nesta etapa. A redução de FP deverá ser feita depois
  por DecisionEngine, Isolation Forest, behavioral_analytics e social_engineering.

Este experimento:
  - remove `rn`;
  - treina modelos com pesos positivos altos;
  - usa threshold sweep fino na região baixa;
  - seleciona políticas que tentam atingir recall >= 0.90 na validação;
  - reporta o custo em FP/FPR no HOLDOUT_LABEL_SAFE e HOLDOUT_FULL;
  - salva artefatos shadow, sem sobrescrever produção.

Uso:
  python scripts\exp_012c_r3_high_recall_lgbm_v3.py

Smoke test:
  python scripts\exp_012c_r3_high_recall_lgbm_v3.py --max-configs 2

Saídas:
  resultados/experimentos/EXP-012C-R3/
  backend/artefatos_candidatos/exp012c_r3_lgbm_v3_high_recall/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "backend").exists() else Path.cwd()
DADOS_DIR = PROJECT_ROOT / "dados"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-012C-R3"
CANDIDATE_DIR = PROJECT_ROOT / "backend" / "artefatos_candidatos" / "exp012c_r3_lgbm_v3_high_recall"
DEFAULT_INPUT = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | EXP-012C-R3 | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EXP-012C-R3")

EXCLUDE_ALWAYS = {
    "is_fraud",
    "transaction_id",
    "cd_pix",
    "customer_id",
    "cd_cpf_pagador",
    "counterparty_id",
    "cd_cpf_cnpj_recebedor",
    "ds_chave_pix",
    "session_id",
    "ip_address",
    "event_datetime",
    "dt_pix",
    "data_pix",
    "dt_carga",
    "dataset_created_at",
    "dataset_v3_created_at",
    "window_start_date",
    "window_end_date",
    "primeira_data_envio_recebedor_180d",
    "temporal_split",
    "dataset_role",
    "source_dataset",
    "source_dataset_original",
    "sample_strategy",
    "sample_weight",
    "normal_sample_strategy",
    "normal_sample_source",
    "label_status",
    "model_scope_status",
    "bank_direction",
    "triangulation_flag",
    "duplicate_conflict_flag",
    "cd_retorno",
    "autcodret",
    "autdatref",
    "autdathorini",
    "rn",
}
CATEGORICAL_ALLOWLIST = {
    "ds_tipo_chave",
    "ds_tipo_chave_norm",
    "periodo_dia",
    "value_band",
    "device_name",
    "app_version",
    "metodo_autenticacao",
    "is_agendamento_recorrente",
    "ds_sexo",
    "ds_estado_civil",
    "ds_segmento",
}
MAX_CATEGORICAL_CARDINALITY = 80


def dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "event_datetime" not in df.columns and "dt_pix" in df.columns:
        df["event_datetime"] = df["dt_pix"]
    if "transaction_id" not in df.columns and "cd_pix" in df.columns:
        df["transaction_id"] = df["cd_pix"]

    df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
    df["data_pix"] = pd.to_datetime(df["data_pix"] if "data_pix" in df.columns else df["event_datetime"], errors="coerce")
    df = df[df["event_datetime"].notna() & df["data_pix"].notna()].copy()

    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)
    df["temporal_split"] = df["temporal_split"].astype(str).str.upper().str.strip()

    if "sample_weight" not in df.columns:
        df["sample_weight"] = 1.0
    df["sample_weight"] = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).clip(lower=0.05, upper=10.0)

    if "sample_strategy" not in df.columns:
        df["sample_strategy"] = "UNKNOWN"
    if "dataset_role" not in df.columns:
        df["dataset_role"] = np.where(df["is_fraud"] == 1, "POSITIVE_FRAUD", "NEGATIVE_NORMAL")

    df["transaction_id"] = df["transaction_id"].astype("string").str.strip()
    return df.sort_values(["event_datetime", "transaction_id"]).reset_index(drop=True)


def split_dataset(df: pd.DataFrame):
    train = df[df["temporal_split"] == "TRAIN"].copy()
    valid = df[df["temporal_split"] == "VALIDATION"].copy()
    holdout_full = df[df["temporal_split"] == "HOLDOUT"].copy()

    max_fraud_dt = holdout_full.loc[holdout_full["is_fraud"] == 1, "data_pix"].max()
    if pd.isna(max_fraud_dt):
        raise RuntimeError("HOLDOUT não tem fraude confirmada.")

    holdout_safe = holdout_full[holdout_full["data_pix"] <= max_fraud_dt].copy()

    for name, part in [("TRAIN", train), ("VALIDATION", valid), ("HOLDOUT_LABEL_SAFE", holdout_safe)]:
        if part.empty or int(part["is_fraud"].sum()) == 0:
            raise RuntimeError(f"Split inválido: {name}. rows={len(part)}, fraud={int(part['is_fraud'].sum()) if not part.empty else 0}")

    return train, valid, holdout_safe, holdout_full


def infer_features(df: pd.DataFrame, no_categorical: bool):
    feature_cols, numeric_cols, categorical_cols = [], [], []
    for col in df.columns:
        if col in EXCLUDE_ALWAYS:
            continue
        lower = col.lower()
        if lower.endswith("_id") or "cpf" in lower or "cnpj" in lower:
            continue
        if pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col]):
            if int(df[col].notna().sum()) > 0:
                feature_cols.append(col)
                numeric_cols.append(col)
        elif (not no_categorical) and col in CATEGORICAL_ALLOWLIST:
            nunique = int(df[col].astype("string").nunique(dropna=True))
            if 0 < nunique <= MAX_CATEGORICAL_CARDINALITY:
                feature_cols.append(col)
                categorical_cols.append(col)
    return feature_cols, numeric_cols, categorical_cols


def build_preprocessor(num_cols, cat_cols):
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", make_onehot_encoder()),
        ]), cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


def get_feature_names(preprocessor):
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if hasattr(trans, "get_feature_names_out"):
                try:
                    names.extend(list(trans.get_feature_names_out(cols)))
                    continue
                except Exception:
                    pass
            names.extend(list(cols))
        return names


def sample_weights(y, base_weight, pos_multiplier):
    y_arr = y.astype(int).values
    w = pd.to_numeric(base_weight, errors="coerce").fillna(1.0).clip(lower=0.05, upper=10.0).astype(float).values
    return np.where(y_arr == 1, w * pos_multiplier, w)


def eval_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 8),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 8),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 8),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 8),
        "fpr": round(float(fp / max(fp + tn, 1)), 8),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 8) if len(np.unique(y_true)) > 1 else None,
        "average_precision": round(float(average_precision_score(y_true, y_prob)), 8) if len(np.unique(y_true)) > 1 else None,
    }


def recall_first_threshold_grid():
    low = np.arange(0.001, 0.100, 0.001)
    mid = np.arange(0.100, 0.501, 0.005)
    high = np.arange(0.505, 0.996, 0.005)
    thresholds = np.unique(np.round(np.concatenate([low, mid, high]), 6))
    return thresholds


def threshold_sweep(y_true, y_prob):
    return pd.DataFrame([eval_threshold(y_true, y_prob, t) for t in recall_first_threshold_grid()])


def nearest(df, threshold):
    return df.loc[(df["threshold"] - threshold).abs().idxmin()]


def policy_rows(val_sweep, safe_sweep, full_sweep):
    rows = []

    def add(policy, sub, sort_cols, ascending):
        if sub.empty:
            return
        v = sub.sort_values(sort_cols, ascending=ascending).iloc[0]
        hs = nearest(safe_sweep, float(v["threshold"]))
        hf = nearest(full_sweep, float(v["threshold"]))
        row = {"policy": policy, "threshold": float(v["threshold"])}
        for prefix, src in [("val", v), ("safe", hs), ("full", hf)]:
            for k in ["tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr", "roc_auc", "average_precision"]:
                row[f"{prefix}_{k}"] = src.get(k)
        rows.append(row)

    # Recall-first policies.
    add("VAL_RECALL_GE_97_MIN_FPR", val_sweep[val_sweep["recall"] >= 0.97], ["fpr", "precision", "f1"], [True, False, False])
    add("VAL_RECALL_GE_95_MIN_FPR", val_sweep[val_sweep["recall"] >= 0.95], ["fpr", "precision", "f1"], [True, False, False])
    add("VAL_RECALL_GE_90_MIN_FPR", val_sweep[val_sweep["recall"] >= 0.90], ["fpr", "precision", "f1"], [True, False, False])
    add("VAL_RECALL_GE_90_MAX_PRECISION", val_sweep[val_sweep["recall"] >= 0.90], ["precision", "f1", "fpr"], [False, False, True])
    add("VAL_RECALL_GE_85_MIN_FPR", val_sweep[val_sweep["recall"] >= 0.85], ["fpr", "precision", "f1"], [True, False, False])

    # Diagnostic fallback policies.
    add("VAL_RECALL_GE_80_MIN_FPR", val_sweep[val_sweep["recall"] >= 0.80], ["fpr", "precision", "f1"], [True, False, False])
    add("VAL_BEST_RECALL_FPR_LE_10PCT", val_sweep[val_sweep["fpr"] <= 0.10], ["recall", "f1", "precision"], [False, False, False])
    add("VAL_BEST_RECALL_FPR_LE_20PCT", val_sweep[val_sweep["fpr"] <= 0.20], ["recall", "f1", "precision"], [False, False, False])
    add("VAL_BEST_F1", val_sweep, ["f1", "recall", "precision"], [False, False, False])
    return rows


def champion_rank(row):
    # Ranking agora é recall-first:
    # 1. atingir recall >= 0.90 na validação;
    # 2. entre eles, minimizar FPR;
    # 3. maximizar precision/F1.
    hit90 = int(row["val_recall"] >= 0.90)
    hit95 = int(row["val_recall"] >= 0.95)
    hit97 = int(row["val_recall"] >= 0.97)
    return (
        hit97,
        hit95,
        hit90,
        -float(row["val_fpr"]),
        float(row["val_precision"]),
        float(row["val_f1"]),
        float(row["val_recall"]),
    )


def topk_metrics(df, score_col, split_name, ks=(100, 250, 500, 1000, 2500, 5000, 10000)):
    d = df.copy()
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce").fillna(-1)
    d["is_fraud"] = pd.to_numeric(d["is_fraud"], errors="coerce").fillna(0).astype(int)
    d = d.sort_values(score_col, ascending=False).reset_index(drop=True)
    n_fraud = int(d["is_fraud"].sum())
    rows = []
    for k in ks:
        k2 = min(k, len(d))
        top = d.head(k2)
        tp = int(top["is_fraud"].sum())
        fp = int(k2 - tp)
        rows.append({
            "split": split_name,
            "k": k2,
            "tp_at_k": tp,
            "fp_at_k": fp,
            "precision_at_k": tp / max(k2, 1),
            "recall_at_k": tp / max(n_fraud, 1),
            "frauds_total": n_fraud,
            "score_min_at_k": float(top[score_col].min()) if k2 else None,
            "score_max": float(d[score_col].max()) if len(d) else None,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--candidate-dir", default=str(CANDIDATE_DIR))
    parser.add_argument("--no-categorical", action="store_true")
    parser.add_argument("--max-configs", type=int, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    candidate_dir = Path(args.candidate_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    train, valid, holdout_safe, holdout_full = split_dataset(df)

    feature_cols, numeric_cols, categorical_cols = infer_features(df, no_categorical=args.no_categorical)
    log.info("Features R3: %d (%d numéricas, %d categóricas). rn removida=%s", len(feature_cols), len(numeric_cols), len(categorical_cols), "rn" not in feature_cols)

    X_train, y_train = train[feature_cols], train["is_fraud"].astype(int)
    X_val, y_val = valid[feature_cols], valid["is_fraud"].astype(int)
    X_safe, y_safe = holdout_safe[feature_cols], holdout_safe["is_fraud"].astype(int)
    X_full, y_full = holdout_full[feature_cols], holdout_full["is_fraud"].astype(int)

    configs = [
        {
            "config_id": "HR01_pos8_base",
            "pos_multiplier": 8.0,
            "params": dict(n_estimators=5000, learning_rate=0.012, num_leaves=63, max_depth=7, min_child_samples=45, subsample=0.90, colsample_bytree=0.85, reg_alpha=0.5, reg_lambda=1.5),
        },
        {
            "config_id": "HR02_pos15_deep",
            "pos_multiplier": 15.0,
            "params": dict(n_estimators=5500, learning_rate=0.010, num_leaves=95, max_depth=8, min_child_samples=35, subsample=0.90, colsample_bytree=0.85, reg_alpha=0.35, reg_lambda=1.25),
        },
        {
            "config_id": "HR03_pos25_recall",
            "pos_multiplier": 25.0,
            "params": dict(n_estimators=6000, learning_rate=0.010, num_leaves=127, max_depth=8, min_child_samples=30, subsample=0.90, colsample_bytree=0.85, reg_alpha=0.25, reg_lambda=1.00),
        },
        {
            "config_id": "HR04_pos40_ultra",
            "pos_multiplier": 40.0,
            "params": dict(n_estimators=6500, learning_rate=0.008, num_leaves=127, max_depth=8, min_child_samples=25, subsample=0.90, colsample_bytree=0.85, reg_alpha=0.20, reg_lambda=1.00),
        },
        {
            "config_id": "HR05_pos60_ultra_reg",
            "pos_multiplier": 60.0,
            "params": dict(n_estimators=7000, learning_rate=0.008, num_leaves=95, max_depth=8, min_child_samples=35, subsample=0.90, colsample_bytree=0.80, reg_alpha=0.75, reg_lambda=2.00),
        },
        {
            "config_id": "HR06_pos100_extreme_shadow",
            "pos_multiplier": 100.0,
            "params": dict(n_estimators=7000, learning_rate=0.006, num_leaves=127, max_depth=8, min_child_samples=30, subsample=0.90, colsample_bytree=0.85, reg_alpha=0.50, reg_lambda=2.00),
        },
    ]

    if args.max_configs:
        configs = configs[: args.max_configs]

    dump(configs, output_dir / "10_training_grid.json")

    all_rows = []
    model_refs = {}

    for i, cfg in enumerate(configs, 1):
        cid = cfg["config_id"]
        print("=" * 80)
        print(f"{i}/{len(configs)} — Treinando {cid}")
        print("=" * 80)

        prep = build_preprocessor(numeric_cols, categorical_cols)
        Xtr = prep.fit_transform(X_train)
        Xva = prep.transform(X_val)
        Xhs = prep.transform(X_safe)
        Xhf = prep.transform(X_full)

        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": 1.0,
            "boost_from_average": False,
            "verbose": -1,
            **cfg["params"],
        }

        model = LGBMClassifier(**params)
        model.fit(
            Xtr,
            y_train,
            sample_weight=sample_weights(y_train, train["sample_weight"], cfg["pos_multiplier"]),
            eval_set=[(Xva, y_val)],
            eval_sample_weight=[pd.to_numeric(valid["sample_weight"], errors="coerce").fillna(1.0).values],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(stopping_rounds=300, verbose=False), log_evaluation(period=0)],
        )

        p_val = model.predict_proba(Xva)[:, 1]
        p_safe = model.predict_proba(Xhs)[:, 1]
        p_full = model.predict_proba(Xhf)[:, 1]

        val_s = threshold_sweep(y_val.values, p_val)
        safe_s = threshold_sweep(y_safe.values, p_safe)
        full_s = threshold_sweep(y_full.values, p_full)

        val_s.to_csv(output_dir / f"threshold_sweep_validation_{cid}.csv", index=False)
        safe_s.to_csv(output_dir / f"threshold_sweep_holdout_label_safe_{cid}.csv", index=False)
        full_s.to_csv(output_dir / f"threshold_sweep_holdout_full_{cid}.csv", index=False)

        rows = policy_rows(val_s, safe_s, full_s)
        for r in rows:
            r["config_id"] = cid
            r["pos_multiplier"] = cfg["pos_multiplier"]
            r["best_iteration"] = int(getattr(model, "best_iteration_", 0) or 0)
        all_rows.extend(rows)

        model_path = output_dir / f"model_{cid}.joblib"
        prep_path = output_dir / f"preprocessor_{cid}.joblib"
        joblib.dump(model, model_path)
        joblib.dump(prep, prep_path)
        model_refs[cid] = {
            "model_path": str(model_path),
            "preprocessor_path": str(prep_path),
            "params": params,
            "pos_multiplier": cfg["pos_multiplier"],
        }

    comparison = pd.DataFrame(all_rows)
    if comparison.empty:
        raise RuntimeError("Nenhuma política foi gerada. Verifique thresholds/sweeps.")

    comparison["_rank"] = comparison.apply(champion_rank, axis=1)
    comparison = comparison.sort_values("_rank", ascending=False).drop(columns=["_rank"]).reset_index(drop=True)
    comparison.to_csv(output_dir / "01_high_recall_policy_comparison.csv", index=False)

    champ = comparison.iloc[0].to_dict()
    champ_cid = champ["config_id"]
    champ_threshold = float(champ["threshold"])

    best_model = joblib.load(model_refs[champ_cid]["model_path"])
    best_prep = joblib.load(model_refs[champ_cid]["preprocessor_path"])

    Xtr = best_prep.transform(X_train)
    Xva = best_prep.transform(X_val)
    Xhs = best_prep.transform(X_safe)
    Xhf = best_prep.transform(X_full)

    p_train = best_model.predict_proba(Xtr)[:, 1]
    p_val = best_model.predict_proba(Xva)[:, 1]
    p_safe = best_model.predict_proba(Xhs)[:, 1]
    p_full = best_model.predict_proba(Xhf)[:, 1]

    metric_rows = []
    for split, y, p in [
        ("TRAIN", y_train.values, p_train),
        ("VALIDATION", y_val.values, p_val),
        ("HOLDOUT_LABEL_SAFE", y_safe.values, p_safe),
        ("HOLDOUT_FULL", y_full.values, p_full),
    ]:
        m = eval_threshold(y, p, champ_threshold)
        m["temporal_split"] = split
        m["config_id"] = champ_cid
        m["policy"] = champ["policy"]
        metric_rows.append(m)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "02_champion_metrics_by_split.csv", index=False)

    pred_safe = holdout_safe.copy()
    pred_safe["lgbm_r3_score"] = p_safe
    pred_safe["lgbm_r3_pred"] = (p_safe >= champ_threshold).astype(int)
    pred_safe.to_csv(output_dir / "03_champion_predictions_holdout_label_safe.csv", index=False)

    pred_safe[(pred_safe["is_fraud"] == 1) & (pred_safe["lgbm_r3_pred"] == 0)].to_csv(output_dir / "04_champion_false_negatives_holdout_label_safe.csv", index=False)
    pred_safe[(pred_safe["is_fraud"] == 0) & (pred_safe["lgbm_r3_pred"] == 1)].to_csv(output_dir / "05_champion_false_positives_holdout_label_safe.csv", index=False)

    pred_valid = valid.copy()
    pred_valid["lgbm_r3_score"] = p_val
    pred_full = holdout_full.copy()
    pred_full["lgbm_r3_score"] = p_full

    topk = pd.concat([
        topk_metrics(pred_valid, "lgbm_r3_score", "VALIDATION"),
        topk_metrics(pred_safe, "lgbm_r3_score", "HOLDOUT_LABEL_SAFE"),
        topk_metrics(pred_full, "lgbm_r3_score", "HOLDOUT_FULL"),
    ], ignore_index=True)
    topk.to_csv(output_dir / "06_topk_analysis.csv", index=False)

    names = get_feature_names(best_prep)
    imp = pd.DataFrame({
        "feature": names,
        "importance_gain": best_model.booster_.feature_importance(importance_type="gain"),
        "importance_split": best_model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    imp.to_csv(output_dir / "07_champion_feature_importance.csv", index=False)

    feature_schema = {
        "input_features_pre_transform": feature_cols,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "model_features_after_transform": names,
        "excluded_always": sorted(EXCLUDE_ALWAYS),
        "n_input_features": len(feature_cols),
        "n_model_features_after_transform": len(names),
    }

    model_final = candidate_dir / "model_lgbm_v3_r3_high_recall_shadow.joblib"
    prep_final = candidate_dir / "preprocessor_lgbm_v3_r3_high_recall_shadow.joblib"
    threshold_final = candidate_dir / "threshold_policy_exp012c_r3_high_recall.json"
    features_final = candidate_dir / "features_lgbm_v3_r3_high_recall_shadow.json"
    manifest_final = candidate_dir / "manifest_exp012c_r3_lgbm_v3_high_recall.json"

    joblib.dump(best_model, model_final)
    joblib.dump(best_prep, prep_final)
    dump(champ, threshold_final)
    dump(feature_schema, features_final)

    safe_metrics = metrics[metrics["temporal_split"] == "HOLDOUT_LABEL_SAFE"].iloc[0].to_dict()
    full_metrics = metrics[metrics["temporal_split"] == "HOLDOUT_FULL"].iloc[0].to_dict()

    status = "RECALL_TARGET_ACHIEVED_IN_VALIDATION" if champ.get("val_recall", 0) >= 0.90 else "RECALL_TARGET_NOT_ACHIEVED_IN_VALIDATION"
    status += "_AND_HOLDOUT" if safe_metrics.get("recall", 0) >= 0.90 else "_HOLDOUT_BELOW_TARGET"

    summary = {
        "experiment": "EXP-012C-R3",
        "status": "DONE",
        "objective_status": status,
        "input_path": str(input_path),
        "input_md5": file_md5(input_path),
        "n_rows": int(len(df)),
        "n_fraud": int(df["is_fraud"].sum()),
        "n_normal": int((df["is_fraud"] == 0).sum()),
        "n_configs": len(configs),
        "n_features_input": len(feature_cols),
        "n_model_features": len(names),
        "champion_config_id": champ_cid,
        "champion_policy": champ["policy"],
        "champion_threshold": champ_threshold,
        "champion_validation": {k: champ[k] for k in champ if str(k).startswith("val_")},
        "champion_holdout_label_safe": safe_metrics,
        "champion_holdout_full": full_metrics,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "candidate_dir": str(candidate_dir),
        "model_path": str(model_final),
        "preprocessor_path": str(prep_final),
    }
    dump(summary, output_dir / "00_run_summary.json")

    dump({
        "model_version": "exp012c_r3_lgbm_v3_high_recall_shadow",
        "status": "HIGH_RECALL_SHADOW_CANDIDATE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "threshold_policy": champ,
        "notes": [
            "Artefato shadow recall-first. Não sobrescreve produção.",
            "O objetivo é maximizar captura de fraudes; FP deve ser reduzido em etapa posterior com IF/BEH/SE/DecisionEngine.",
            "Não promover isoladamente sem EXP-012D/012E E2E cascata.",
        ],
    }, manifest_final)

    md = []
    md.append("# EXP-012C-R3 — High Recall Recommendation")
    md.append("")
    md.append("## Objetivo")
    md.append("Maximizar recall do LGBM como primeiro estágio, aceitando FP para posterior filtragem por IF/BEH/SE/DecisionEngine.")
    md.append("")
    md.append("## Champion")
    md.append(f"- config_id: `{champ_cid}`")
    md.append(f"- policy: `{champ['policy']}`")
    md.append(f"- threshold: `{champ_threshold}`")
    md.append("")
    md.append("## Holdout label-safe")
    for k in ["tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr", "roc_auc", "average_precision"]:
        md.append(f"- {k}: {safe_metrics.get(k)}")
    md.append("")
    md.append("## Decisão")
    if safe_metrics.get("recall", 0) >= 0.90:
        md.append("Atingiu recall alvo no holdout label-safe. Próximo passo: EXP-012D cascata de redução de FP.")
    else:
        md.append("Não atingiu recall alvo no holdout label-safe. Próximo passo: analisar FNs e reforçar dados/features antes de promoção.")
    md.append("")
    md.append("## Próximo passo recomendado")
    md.append("EXP-012D — E2E/cascade shadow: LGBM high-recall primeiro estágio + filtros IF/BEH/SE para redução de FP.")
    (output_dir / "08_recommendation.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
