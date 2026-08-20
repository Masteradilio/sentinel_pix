#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-012C-R4 — LGBM High-Recall FP Squeeze

Objetivo:
  Espremer o máximo possível do próprio LGBM antes de acionar módulos externos
  (Isolation Forest, behavioral_analytics, social_engineering).

Meta:
  Reduzir falsos positivos mantendo recall >= 95% na validação.
  O holdout label-safe é usado para diagnóstico e decisão de continuidade.

As 6 ideias exploradas:
  1. Threshold exato por ranking/score, não grade fixa.
  2. Top-k operacional por alert-rate.
  3. Hard Negative Mining com FPs high-score.
  4. Cascata LGBM-only em dois estágios.
  5. Threshold segmentado por variáveis de risco.
  6. Ajuste de pesos por estratégia de amostragem e por hard negatives.

Entrada:
  dados/hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv

Saídas principais:
  resultados/experimentos/EXP-012C-R4/
    00_run_summary.json
    01_candidate_policy_comparison.csv
    02_champion_metrics_by_split.csv
    03_exact_threshold_candidates.csv
    04_topk_candidates.csv
    05_segmented_threshold_candidates.csv
    06_hard_negative_candidates.csv
    07_two_stage_candidates.csv
    08_champion_predictions_holdout_label_safe.csv
    09_champion_false_negatives_holdout_label_safe.csv
    10_champion_false_positives_holdout_label_safe.csv
    11_stage1_feature_importance.csv
    12_stage2_feature_importance.csv
    13_search_space.json
    14_recommendation.md

Artefatos:
  backend/artefatos_candidatos/exp012c_r4_lgbm_fp_squeeze/

Uso:
  python scripts/exp_012c_r4_lgbm_fp_squeeze.py

Smoke test:
  python scripts/exp_012c_r4_lgbm_fp_squeeze.py --fast
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
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-012C-R4"
CANDIDATE_DIR = PROJECT_ROOT / "backend" / "artefatos_candidatos" / "exp012c_r4_lgbm_fp_squeeze"
DEFAULT_INPUT = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | EXP-012C-R4 | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EXP-012C-R4")

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
DEFAULT_TARGET_RECALL = 0.95


# =============================================================================
# IO/helpers
# =============================================================================
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


def base_sample_weight(y, base_weight, pos_multiplier):
    y_arr = y.astype(int).values
    w = pd.to_numeric(base_weight, errors="coerce").fillna(1.0).clip(lower=0.05, upper=10.0).astype(float).values
    return np.where(y_arr == 1, w * pos_multiplier, w)


def strategy_weight_multiplier(sample_strategy: pd.Series, profile: str) -> np.ndarray:
    """
    Ideia 6: ajustar peso dos normais por origem/estratégia.
    """
    s = sample_strategy.astype("string").fillna("UNKNOWN").str.upper()
    m = np.ones(len(s), dtype=float)

    if profile == "none":
        return m

    if profile == "hardneg_boost":
        m *= np.where(s.str.contains("N3_HARD_NEGATIVES", regex=False), 3.0, 1.0)
        m *= np.where(s.str.contains("N4_RECENT_NORMALS", regex=False), 1.8, 1.0)
        m *= np.where(s.str.contains("N1_BACKGROUND_NORMAL", regex=False), 0.8, 1.0)
        return m

    if profile == "aggressive_hardneg":
        m *= np.where(s.str.contains("N3_HARD_NEGATIVES", regex=False), 5.0, 1.0)
        m *= np.where(s.str.contains("N4_RECENT_NORMALS", regex=False), 2.5, 1.0)
        m *= np.where(s.str.contains("N2_MATCHED_CONTROLS", regex=False), 1.2, 1.0)
        m *= np.where(s.str.contains("N1_BACKGROUND_NORMAL", regex=False), 0.6, 1.0)
        return m

    raise ValueError(f"profile desconhecido: {profile}")


def make_weights(df_part: pd.DataFrame, pos_multiplier: float, strategy_profile: str = "none", hard_negative_mask=None, hard_negative_multiplier: float = 1.0):
    y = df_part["is_fraud"].astype(int)
    w = base_sample_weight(y, df_part["sample_weight"], pos_multiplier)

    # Strategy weights só para normais.
    strat_m = strategy_weight_multiplier(df_part["sample_strategy"], strategy_profile)
    w = np.where(y.values == 0, w * strat_m, w)

    # Hard negative weights só para normais marcados.
    if hard_negative_mask is not None:
        hn = np.asarray(hard_negative_mask).astype(bool)
        w = np.where((y.values == 0) & hn, w * hard_negative_multiplier, w)

    return w


# =============================================================================
# Metrics/policies
# =============================================================================
def eval_binary(y_true, y_pred, y_prob=None, threshold=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "threshold": None if threshold is None else round(float(threshold), 10),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 8),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 8),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 8),
        "fpr": round(float(fp / max(fp + tn, 1)), 8),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 8)
        out["average_precision"] = round(float(average_precision_score(y_true, y_prob)), 8)
    return out


def eval_threshold(y_true, y_prob, threshold):
    return eval_binary(y_true, (y_prob >= threshold).astype(int), y_prob=y_prob, threshold=threshold)


def exact_threshold_max_recall(y_true, y_prob, target_recall=DEFAULT_TARGET_RECALL):
    """
    Ideia 1: escolhe o maior threshold possível que mantém recall >= target.
    Isso minimiza positivos/FP na validação para um dado score.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    pos_scores = np.sort(p[y == 1])[::-1]
    n_pos = len(pos_scores)
    if n_pos == 0:
        return float("inf")

    needed = int(np.ceil(target_recall * n_pos))
    needed = min(max(needed, 1), n_pos)
    threshold = float(pos_scores[needed - 1])
    # Subtrai epsilon mínimo para reduzir risco de empate numérico excluindo o positivo limite.
    return max(0.0, threshold - 1e-12)


def exact_threshold_candidates(y_true, y_prob, recalls=(0.90, 0.92, 0.95, 0.97, 0.98, 0.99)):
    rows = []
    for r in recalls:
        th = exact_threshold_max_recall(y_true, y_prob, r)
        m = eval_threshold(y_true, y_prob, th)
        m["target_recall"] = r
        rows.append(m)
    return pd.DataFrame(rows)


def topk_min_for_recall(y_true, y_prob, target_recall=DEFAULT_TARGET_RECALL):
    """
    Ideia 2: menor k que captura target_recall dos positivos na validação.
    Retorna alert_rate para aplicar em outros splits.
    """
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(y_prob).astype(float)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    n_pos = int(df["y"].sum())
    needed = int(np.ceil(target_recall * n_pos))
    csum = df["y"].cumsum()
    hit_idx = np.where(csum.values >= needed)[0]
    if len(hit_idx) == 0:
        k = len(df)
    else:
        k = int(hit_idx[0] + 1)
    return k, k / max(len(df), 1), float(df.loc[k - 1, "p"])


def predict_topk(y_prob, alert_rate):
    n = len(y_prob)
    k = int(np.ceil(alert_rate * n))
    k = min(max(k, 1), n)
    order = np.argsort(-np.asarray(y_prob).astype(float))
    pred = np.zeros(n, dtype=int)
    pred[order[:k]] = 1
    return pred, k


def add_candidate(rows, candidate_id, idea, model_id, policy, y_val, p_val, y_safe, p_safe, y_full, p_full, pred_val, pred_safe, pred_full, threshold=None, alert_rate=None, extra=None):
    vm = eval_binary(y_val, pred_val, y_prob=p_val, threshold=threshold)
    sm = eval_binary(y_safe, pred_safe, y_prob=p_safe, threshold=threshold)
    fm = eval_binary(y_full, pred_full, y_prob=p_full, threshold=threshold)

    row = {
        "candidate_id": candidate_id,
        "idea": idea,
        "model_id": model_id,
        "policy": policy,
        "threshold": threshold,
        "alert_rate": alert_rate,
    }
    for prefix, src in [("val", vm), ("safe", sm), ("full", fm)]:
        for k, v in src.items():
            row[f"{prefix}_{k}"] = v
    if extra:
        row.update(extra)
    rows.append(row)


def champion_sort_key(row, target_recall=DEFAULT_TARGET_RECALL):
    """
    Seleção oficial:
      1) precisa bater recall alvo na validação;
      2) minimizar FP/FPR na validação;
      3) maximizar precision/F1;
      4) holdout é diagnóstico, não seleção primária.
    """
    pass_recall = int(row["val_recall"] >= target_recall)
    return (
        pass_recall,
        -float(row["val_fp"]),
        -float(row["val_fpr"]),
        float(row["val_precision"]),
        float(row["val_f1"]),
        float(row["safe_recall"]),
    )


# =============================================================================
# Training
# =============================================================================
def train_lgbm(model_id, train_df, valid_df, feature_cols, numeric_cols, categorical_cols, params, pos_multiplier, strategy_profile="none", hard_negative_mask=None, hard_negative_multiplier=1.0):
    prep = build_preprocessor(numeric_cols, categorical_cols)

    X_train = train_df[feature_cols]
    y_train = train_df["is_fraud"].astype(int)
    X_valid = valid_df[feature_cols]
    y_valid = valid_df["is_fraud"].astype(int)

    Xtr = prep.fit_transform(X_train)
    Xva = prep.transform(X_valid)

    w_train = make_weights(
        train_df,
        pos_multiplier=pos_multiplier,
        strategy_profile=strategy_profile,
        hard_negative_mask=hard_negative_mask,
        hard_negative_multiplier=hard_negative_multiplier,
    )
    w_valid = pd.to_numeric(valid_df["sample_weight"], errors="coerce").fillna(1.0).values

    model = LGBMClassifier(**params)
    model.fit(
        Xtr,
        y_train,
        sample_weight=w_train,
        eval_set=[(Xva, y_valid)],
        eval_sample_weight=[w_valid],
        eval_metric="binary_logloss",
        callbacks=[early_stopping(stopping_rounds=300, verbose=False), log_evaluation(period=0)],
    )

    return model, prep


def predict_model(model, prep, df_part, feature_cols):
    return model.predict_proba(prep.transform(df_part[feature_cols]))[:, 1]


def lgbm_params_base(seed=42, **overrides):
    base = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "random_state": seed,
        "n_jobs": -1,
        "scale_pos_weight": 1.0,
        "boost_from_average": False,
        "verbose": -1,
    }
    base.update(overrides)
    return base


# =============================================================================
# Segmented thresholds
# =============================================================================
def segmented_thresholds_from_validation(valid_df, score_col, target_recall, segment_cols, min_pos=8):
    """
    Ideia 5: thresholds por segmento.
    Para cada segmento com positivos suficientes, usa o maior threshold que
    mantém recall >= target dentro do segmento. Segmentos pequenos usam global.
    """
    global_th = exact_threshold_max_recall(valid_df["is_fraud"].values, valid_df[score_col].values, target_recall)
    rules = []

    for keys, g in valid_df.groupby(segment_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n_pos = int(g["is_fraud"].sum())
        if n_pos >= min_pos:
            th = exact_threshold_max_recall(g["is_fraud"].values, g[score_col].values, target_recall)
            source = "segment"
        else:
            th = global_th
            source = "global_fallback"

        rule = {col: str(val) for col, val in zip(segment_cols, keys)}
        rule.update({"threshold": float(th), "source": source, "n": int(len(g)), "n_pos": n_pos})
        rules.append(rule)

    return pd.DataFrame(rules), float(global_th)


def apply_segmented_thresholds(df_part, score_col, segment_cols, rules_df, global_threshold):
    out = pd.DataFrame(index=df_part.index)
    tmp = df_part[segment_cols].copy()
    for c in segment_cols:
        tmp[c] = tmp[c].astype("string").fillna("<NA>").astype(str)

    rules = rules_df.copy()
    for c in segment_cols:
        rules[c] = rules[c].astype("string").fillna("<NA>").astype(str)

    tmp["_row_id"] = np.arange(len(tmp))
    merged = tmp.merge(rules[segment_cols + ["threshold"]], on=segment_cols, how="left")
    thresholds = merged["threshold"].fillna(global_threshold).astype(float).values
    pred = (df_part[score_col].astype(float).values >= thresholds).astype(int)
    return pred


# =============================================================================
# Main experiment
# =============================================================================
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--candidate-dir", default=str(CANDIDATE_DIR))
    parser.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)
    parser.add_argument("--fast", action="store_true", help="Executa menos configs para smoke test.")
    parser.add_argument("--no-categorical", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    candidate_dir = Path(args.candidate_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    train, valid, holdout_safe, holdout_full = split_dataset(df)
    feature_cols, numeric_cols, categorical_cols = infer_features(df, no_categorical=args.no_categorical)

    log.info("Dataset rows=%d fraud=%d normal=%d", len(df), int(df["is_fraud"].sum()), int((df["is_fraud"] == 0).sum()))
    log.info("Features: %d (%d numéricas, %d categóricas)", len(feature_cols), len(numeric_cols), len(categorical_cols))

    y_val = valid["is_fraud"].astype(int).values
    y_safe = holdout_safe["is_fraud"].astype(int).values
    y_full = holdout_full["is_fraud"].astype(int).values

    search_space = {
        "target_recall": args.target_recall,
        "ideas": [
            "exact_threshold",
            "topk_operational",
            "hard_negative_mining",
            "two_stage_lgbm",
            "segmented_thresholds",
            "sample_strategy_weights",
        ],
        "stage1_model": "HR01_pos8_base",
        "hard_negative_models": [],
        "two_stage_models": [],
        "segment_sets": [],
    }

    # -------------------------------------------------------------------------
    # Stage 1: base HR01 high-recall.
    # -------------------------------------------------------------------------
    stage1_params = lgbm_params_base(
        n_estimators=5000,
        learning_rate=0.012,
        num_leaves=63,
        max_depth=7,
        min_child_samples=45,
        subsample=0.90,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=1.5,
    )
    log.info("Treinando stage1 HR01_pos8_base...")
    stage1_model, stage1_prep = train_lgbm(
        "HR01_pos8_base",
        train,
        valid,
        feature_cols,
        numeric_cols,
        categorical_cols,
        params=stage1_params,
        pos_multiplier=8.0,
        strategy_profile="none",
    )

    train["score_stage1"] = predict_model(stage1_model, stage1_prep, train, feature_cols)
    valid["score_stage1"] = predict_model(stage1_model, stage1_prep, valid, feature_cols)
    holdout_safe["score_stage1"] = predict_model(stage1_model, stage1_prep, holdout_safe, feature_cols)
    holdout_full["score_stage1"] = predict_model(stage1_model, stage1_prep, holdout_full, feature_cols)

    p_val_s1 = valid["score_stage1"].values
    p_safe_s1 = holdout_safe["score_stage1"].values
    p_full_s1 = holdout_full["score_stage1"].values

    candidate_rows = []

    # Idea 1: exact threshold.
    th95 = exact_threshold_max_recall(y_val, p_val_s1, args.target_recall)
    pred_val = (p_val_s1 >= th95).astype(int)
    pred_safe = (p_safe_s1 >= th95).astype(int)
    pred_full = (p_full_s1 >= th95).astype(int)
    add_candidate(candidate_rows, "S1_EXACT_THR_RECALL95", "1_exact_threshold", "HR01_pos8_base", "max_threshold_for_val_recall_target", y_val, p_val_s1, y_safe, p_safe_s1, y_full, p_full_s1, pred_val, pred_safe, pred_full, threshold=th95)

    exact_df = exact_threshold_candidates(y_val, p_val_s1, recalls=(0.90, 0.92, 0.95, 0.97, 0.98, 0.99))
    exact_df.to_csv(output_dir / "03_exact_threshold_candidates.csv", index=False)

    # Idea 2: top-k.
    topk_rows = []
    for target in [0.90, 0.92, 0.95, 0.97, 0.98, 0.99]:
        k_val, alert_rate, min_score = topk_min_for_recall(y_val, p_val_s1, target)
        pred_val, kval2 = predict_topk(p_val_s1, alert_rate)
        pred_safe, ksafe = predict_topk(p_safe_s1, alert_rate)
        pred_full, kfull = predict_topk(p_full_s1, alert_rate)

        cid = f"S1_TOPK_RECALL{int(target*100)}"
        add_candidate(
            candidate_rows,
            cid,
            "2_topk_operational",
            "HR01_pos8_base",
            f"topk_min_for_val_recall_{target}",
            y_val, p_val_s1, y_safe, p_safe_s1, y_full, p_full_s1,
            pred_val, pred_safe, pred_full,
            threshold=None,
            alert_rate=alert_rate,
            extra={"k_validation": k_val, "k_holdout_safe": ksafe, "k_holdout_full": kfull, "target_recall_policy": target, "score_min_validation_k": min_score},
        )
        row = {"candidate_id": cid, "target_recall": target, "k_validation": k_val, "alert_rate": alert_rate, "score_min_validation_k": min_score}
        row.update({f"val_{k}": v for k, v in eval_binary(y_val, pred_val, p_val_s1).items()})
        row.update({f"safe_{k}": v for k, v in eval_binary(y_safe, pred_safe, p_safe_s1).items()})
        row.update({f"full_{k}": v for k, v in eval_binary(y_full, pred_full, p_full_s1).items()})
        topk_rows.append(row)

    pd.DataFrame(topk_rows).to_csv(output_dir / "04_topk_candidates.csv", index=False)

    # Idea 5: segmented thresholds on stage1.
    segment_sets = [
        ["value_band"],
        ["ds_tipo_chave_norm"],
        ["periodo_dia"],
        ["mbk_available_flag"],
        ["first_receiver_flag_real"],
        ["value_band", "ds_tipo_chave_norm"],
        ["first_receiver_flag_real", "value_band"],
        ["mbk_available_flag", "ds_tipo_chave_norm"],
    ]
    if args.fast:
        segment_sets = segment_sets[:3]

    segmented_rows = []
    segmented_rules_by_id = {}
    for i, seg_cols in enumerate(segment_sets, 1):
        missing = [c for c in seg_cols if c not in valid.columns]
        if missing:
            continue

        rules, global_th = segmented_thresholds_from_validation(valid, "score_stage1", args.target_recall, seg_cols, min_pos=8)
        pred_val = apply_segmented_thresholds(valid, "score_stage1", seg_cols, rules, global_th)
        pred_safe = apply_segmented_thresholds(holdout_safe, "score_stage1", seg_cols, rules, global_th)
        pred_full = apply_segmented_thresholds(holdout_full, "score_stage1", seg_cols, rules, global_th)

        cid = "S1_SEG_" + "_".join(seg_cols).upper()
        segmented_rules_by_id[cid] = rules
        add_candidate(candidate_rows, cid, "5_segmented_thresholds", "HR01_pos8_base", "segmented_thresholds_val_recall_target", y_val, p_val_s1, y_safe, p_safe_s1, y_full, p_full_s1, pred_val, pred_safe, pred_full, threshold=None, extra={"segment_cols": "|".join(seg_cols), "global_threshold": global_th})
        row = {"candidate_id": cid, "segment_cols": "|".join(seg_cols), "global_threshold": global_th}
        row.update({f"val_{k}": v for k, v in eval_binary(y_val, pred_val, p_val_s1).items()})
        row.update({f"safe_{k}": v for k, v in eval_binary(y_safe, pred_safe, p_safe_s1).items()})
        row.update({f"full_{k}": v for k, v in eval_binary(y_full, pred_full, p_full_s1).items()})
        segmented_rows.append(row)

    pd.DataFrame(segmented_rows).to_csv(output_dir / "05_segmented_threshold_candidates.csv", index=False)
    # Persist rules in JSON-friendly CSVs.
    for cid, rules in segmented_rules_by_id.items():
        rules.to_csv(output_dir / f"segmented_rules_{cid}.csv", index=False)

    # Hard negative masks from stage1 exact threshold and top-score norms.
    hn_threshold = th95
    hard_negative_base = ((train["is_fraud"].astype(int).values == 0) & (train["score_stage1"].values >= hn_threshold))
    # Fallback: top 5% normals by score if mask too small.
    if hard_negative_base.sum() < 100:
        normal_scores = train.loc[train["is_fraud"] == 0, "score_stage1"]
        q = normal_scores.quantile(0.95)
        hard_negative_base = ((train["is_fraud"].astype(int).values == 0) & (train["score_stage1"].values >= q))

    log.info("Hard negatives base: %d", int(hard_negative_base.sum()))

    # -------------------------------------------------------------------------
    # Ideas 3 + 6: hard-negative mining + sample strategy weights.
    # -------------------------------------------------------------------------
    hn_configs = [
        {
            "model_id": "HN01_pos8_hn5_strategy",
            "pos_multiplier": 8.0,
            "hard_negative_multiplier": 5.0,
            "strategy_profile": "hardneg_boost",
            "params": lgbm_params_base(n_estimators=5000, learning_rate=0.012, num_leaves=63, max_depth=7, min_child_samples=55, subsample=0.90, colsample_bytree=0.85, reg_alpha=0.75, reg_lambda=2.0),
        },
        {
            "model_id": "HN02_pos12_hn8_strategy",
            "pos_multiplier": 12.0,
            "hard_negative_multiplier": 8.0,
            "strategy_profile": "hardneg_boost",
            "params": lgbm_params_base(n_estimators=5500, learning_rate=0.010, num_leaves=63, max_depth=7, min_child_samples=55, subsample=0.90, colsample_bytree=0.85, reg_alpha=1.0, reg_lambda=2.5),
        },
        {
            "model_id": "HN03_pos15_hn12_aggressive",
            "pos_multiplier": 15.0,
            "hard_negative_multiplier": 12.0,
            "strategy_profile": "aggressive_hardneg",
            "params": lgbm_params_base(n_estimators=5500, learning_rate=0.010, num_leaves=95, max_depth=8, min_child_samples=45, subsample=0.90, colsample_bytree=0.80, reg_alpha=1.0, reg_lambda=3.0),
        },
    ]
    if args.fast:
        hn_configs = hn_configs[:1]

    search_space["hard_negative_models"] = hn_configs

    model_store = {
        "HR01_pos8_base": {
            "model": stage1_model,
            "preprocessor": stage1_prep,
            "scores": {
                "train": train["score_stage1"].values,
                "validation": p_val_s1,
                "holdout_safe": p_safe_s1,
                "holdout_full": p_full_s1,
            },
        }
    }

    hn_rows = []
    for cfg in hn_configs:
        mid = cfg["model_id"]
        log.info("Treinando hard-negative model %s...", mid)
        model, prep = train_lgbm(
            mid,
            train,
            valid,
            feature_cols,
            numeric_cols,
            categorical_cols,
            params=cfg["params"],
            pos_multiplier=cfg["pos_multiplier"],
            strategy_profile=cfg["strategy_profile"],
            hard_negative_mask=hard_negative_base,
            hard_negative_multiplier=cfg["hard_negative_multiplier"],
        )

        p_val = predict_model(model, prep, valid, feature_cols)
        p_safe = predict_model(model, prep, holdout_safe, feature_cols)
        p_full = predict_model(model, prep, holdout_full, feature_cols)
        p_train = predict_model(model, prep, train, feature_cols)

        model_store[mid] = {
            "model": model,
            "preprocessor": prep,
            "scores": {"train": p_train, "validation": p_val, "holdout_safe": p_safe, "holdout_full": p_full},
            "config": cfg,
        }

        th = exact_threshold_max_recall(y_val, p_val, args.target_recall)
        pred_val = (p_val >= th).astype(int)
        pred_safe = (p_safe >= th).astype(int)
        pred_full = (p_full >= th).astype(int)

        add_candidate(
            candidate_rows,
            f"{mid}_EXACT_RECALL95",
            "3_hard_negative_mining+6_strategy_weights+1_exact_threshold",
            mid,
            "hn_model_exact_threshold_val_recall_target",
            y_val, p_val, y_safe, p_safe, y_full, p_full,
            pred_val, pred_safe, pred_full,
            threshold=th,
            extra={k: v for k, v in cfg.items() if k != "params"},
        )
        row = {"candidate_id": f"{mid}_EXACT_RECALL95", "model_id": mid, "threshold": th}
        row.update({f"val_{k}": v for k, v in eval_binary(y_val, pred_val, p_val).items()})
        row.update({f"safe_{k}": v for k, v in eval_binary(y_safe, pred_safe, p_safe).items()})
        row.update({f"full_{k}": v for k, v in eval_binary(y_full, pred_full, p_full).items()})
        hn_rows.append(row)

        # Segmented thresholds on HN model too.
        if "first_receiver_flag_real" in valid.columns and "value_band" in valid.columns:
            tmp_valid = valid.copy()
            tmp_safe = holdout_safe.copy()
            tmp_full = holdout_full.copy()
            tmp_valid[f"score_{mid}"] = p_val
            tmp_safe[f"score_{mid}"] = p_safe
            tmp_full[f"score_{mid}"] = p_full
            seg_cols = ["first_receiver_flag_real", "value_band"]
            rules, global_th = segmented_thresholds_from_validation(tmp_valid, f"score_{mid}", args.target_recall, seg_cols, min_pos=8)
            pred_val_seg = apply_segmented_thresholds(tmp_valid, f"score_{mid}", seg_cols, rules, global_th)
            pred_safe_seg = apply_segmented_thresholds(tmp_safe, f"score_{mid}", seg_cols, rules, global_th)
            pred_full_seg = apply_segmented_thresholds(tmp_full, f"score_{mid}", seg_cols, rules, global_th)

            add_candidate(
                candidate_rows,
                f"{mid}_SEG_FIRST_VALUE",
                "3_hard_negative_mining+5_segmented_thresholds+6_strategy_weights",
                mid,
                "hn_model_segmented_first_receiver_value",
                y_val, p_val, y_safe, p_safe, y_full, p_full,
                pred_val_seg, pred_safe_seg, pred_full_seg,
                threshold=None,
                extra={"segment_cols": "|".join(seg_cols), "global_threshold": global_th},
            )
            rules.to_csv(output_dir / f"segmented_rules_{mid}_FIRST_VALUE.csv", index=False)

    pd.DataFrame(hn_rows).to_csv(output_dir / "06_hard_negative_candidates.csv", index=False)

    # -------------------------------------------------------------------------
    # Idea 4: Two-stage LGBM-only reranker.
    # -------------------------------------------------------------------------
    two_stage_rows = []

    # Stage1 pass threshold should be high recall. Use validation exact 95 threshold.
    stage1_pass_th = th95
    train_pass_mask = train["score_stage1"].values >= stage1_pass_th
    valid_pass_mask = valid["score_stage1"].values >= stage1_pass_th
    safe_pass_mask = holdout_safe["score_stage1"].values >= stage1_pass_th
    full_pass_mask = holdout_full["score_stage1"].values >= stage1_pass_th

    # Guard: ensure all positives needed can pass stage1. If not, stage2 cannot recover.
    log.info("Stage1 pass train=%d valid=%d safe=%d full=%d", int(train_pass_mask.sum()), int(valid_pass_mask.sum()), int(safe_pass_mask.sum()), int(full_pass_mask.sum()))

    train_s2 = train.loc[train_pass_mask].copy()
    valid_s2 = valid.loc[valid_pass_mask].copy()
    safe_s2 = holdout_safe.loc[safe_pass_mask].copy()
    full_s2 = holdout_full.loc[full_pass_mask].copy()

    # Add stage1 score as a feature for stage2.
    feature_cols_s2 = feature_cols + ["score_stage1"]
    numeric_cols_s2 = numeric_cols + ["score_stage1"]
    categorical_cols_s2 = categorical_cols

    two_stage_configs = [
        {
            "model_id": "S2A_pos8_hn_strategy",
            "pos_multiplier": 8.0,
            "strategy_profile": "hardneg_boost",
            "params": lgbm_params_base(n_estimators=3500, learning_rate=0.015, num_leaves=31, max_depth=6, min_child_samples=30, subsample=0.90, colsample_bytree=0.85, reg_alpha=1.0, reg_lambda=3.0),
        },
        {
            "model_id": "S2B_pos12_deeper",
            "pos_multiplier": 12.0,
            "strategy_profile": "aggressive_hardneg",
            "params": lgbm_params_base(n_estimators=4500, learning_rate=0.010, num_leaves=63, max_depth=7, min_child_samples=25, subsample=0.90, colsample_bytree=0.85, reg_alpha=1.0, reg_lambda=3.0),
        },
    ]
    if args.fast:
        two_stage_configs = two_stage_configs[:1]

    search_space["two_stage_models"] = two_stage_configs

    stage2_store = {}
    for cfg in two_stage_configs:
        mid = cfg["model_id"]
        log.info("Treinando two-stage model %s...", mid)

        model, prep = train_lgbm(
            mid,
            train_s2,
            valid_s2,
            feature_cols_s2,
            numeric_cols_s2,
            categorical_cols_s2,
            params=cfg["params"],
            pos_multiplier=cfg["pos_multiplier"],
            strategy_profile=cfg["strategy_profile"],
            hard_negative_mask=None,
        )

        # Predict only within pass set; outside stage1 pass is final negative.
        p_val_s2 = np.zeros(len(valid), dtype=float)
        p_safe_s2 = np.zeros(len(holdout_safe), dtype=float)
        p_full_s2 = np.zeros(len(holdout_full), dtype=float)

        if len(valid_s2):
            p_val_s2[valid_pass_mask] = predict_model(model, prep, valid_s2, feature_cols_s2)
        if len(safe_s2):
            p_safe_s2[safe_pass_mask] = predict_model(model, prep, safe_s2, feature_cols_s2)
        if len(full_s2):
            p_full_s2[full_pass_mask] = predict_model(model, prep, full_s2, feature_cols_s2)

        # Global cascade threshold chosen by validation global recall.
        th_s2 = exact_threshold_max_recall(y_val, p_val_s2, args.target_recall)
        pred_val = ((valid_pass_mask) & (p_val_s2 >= th_s2)).astype(int)
        pred_safe = ((safe_pass_mask) & (p_safe_s2 >= th_s2)).astype(int)
        pred_full = ((full_pass_mask) & (p_full_s2 >= th_s2)).astype(int)

        add_candidate(
            candidate_rows,
            f"{mid}_CASCADE_RECALL95",
            "4_two_stage_lgbm_only+1_exact_threshold",
            mid,
            "stage1_pass_plus_stage2_exact_threshold",
            y_val, p_val_s2, y_safe, p_safe_s2, y_full, p_full_s2,
            pred_val, pred_safe, pred_full,
            threshold=th_s2,
            extra={"stage1_pass_threshold": stage1_pass_th},
        )

        row = {"candidate_id": f"{mid}_CASCADE_RECALL95", "model_id": mid, "stage1_pass_threshold": stage1_pass_th, "stage2_threshold": th_s2}
        row.update({f"val_{k}": v for k, v in eval_binary(y_val, pred_val, p_val_s2).items()})
        row.update({f"safe_{k}": v for k, v in eval_binary(y_safe, pred_safe, p_safe_s2).items()})
        row.update({f"full_{k}": v for k, v in eval_binary(y_full, pred_full, p_full_s2).items()})
        two_stage_rows.append(row)

        stage2_store[mid] = {"model": model, "preprocessor": prep, "scores": {"validation": p_val_s2, "holdout_safe": p_safe_s2, "holdout_full": p_full_s2}, "config": cfg}

    pd.DataFrame(two_stage_rows).to_csv(output_dir / "07_two_stage_candidates.csv", index=False)

    # -------------------------------------------------------------------------
    # Candidate selection.
    # -------------------------------------------------------------------------
    comp = pd.DataFrame(candidate_rows)
    if comp.empty:
        raise RuntimeError("Nenhum candidato gerado.")

    comp["_sort"] = comp.apply(lambda r: champion_sort_key(r, args.target_recall), axis=1)
    comp = comp.sort_values("_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)
    comp.to_csv(output_dir / "01_candidate_policy_comparison.csv", index=False)

    champion = comp.iloc[0].to_dict()
    champion_id = champion["candidate_id"]
    champion_model_id = champion["model_id"]
    champion_idea = champion["idea"]

    # Reconstruct champion predictions for holdout safe/full.
    # Use candidate policy type.
    def get_scores_for_model(mid):
        if mid in model_store:
            s = model_store[mid]["scores"]
            return s["validation"], s["holdout_safe"], s["holdout_full"]
        if mid in stage2_store:
            s = stage2_store[mid]["scores"]
            return s["validation"], s["holdout_safe"], s["holdout_full"]
        raise RuntimeError(f"Modelo do campeão não encontrado: {mid}")

    p_val_ch, p_safe_ch, p_full_ch = get_scores_for_model(champion_model_id)

    # Final predictions based on champion candidate row.
    if "topk" in champion_idea:
        pred_val_ch, _ = predict_topk(p_val_ch, float(champion["alert_rate"]))
        pred_safe_ch, _ = predict_topk(p_safe_ch, float(champion["alert_rate"]))
        pred_full_ch, _ = predict_topk(p_full_ch, float(champion["alert_rate"]))
    elif "segmented" in champion_idea:
        # For segmented, use saved rules if simple stage1 or HN segment.
        # Fallback to exact threshold if no rules can be reconstructed.
        cid = champion_id
        rules_path = output_dir / f"segmented_rules_{cid}.csv"
        if not rules_path.exists() and champion_model_id != "HR01_pos8_base":
            rules_path = output_dir / f"segmented_rules_{champion_model_id}_FIRST_VALUE.csv"

        if rules_path.exists() and pd.notna(champion.get("segment_cols")):
            seg_cols = str(champion["segment_cols"]).split("|")
            rules = pd.read_csv(rules_path)
            global_th = float(champion.get("global_threshold", exact_threshold_max_recall(y_val, p_val_ch, args.target_recall)))
            score_col = "_champ_score"
            tmp_v = valid.copy(); tmp_v[score_col] = p_val_ch
            tmp_s = holdout_safe.copy(); tmp_s[score_col] = p_safe_ch
            tmp_f = holdout_full.copy(); tmp_f[score_col] = p_full_ch
            pred_val_ch = apply_segmented_thresholds(tmp_v, score_col, seg_cols, rules, global_th)
            pred_safe_ch = apply_segmented_thresholds(tmp_s, score_col, seg_cols, rules, global_th)
            pred_full_ch = apply_segmented_thresholds(tmp_f, score_col, seg_cols, rules, global_th)
        else:
            th = float(champion["threshold"])
            pred_val_ch = (p_val_ch >= th).astype(int)
            pred_safe_ch = (p_safe_ch >= th).astype(int)
            pred_full_ch = (p_full_ch >= th).astype(int)
    elif "two_stage" in champion_idea:
        th = float(champion["threshold"])
        # If stage2 scores are zero outside pass set, this is enough.
        pred_val_ch = (p_val_ch >= th).astype(int)
        pred_safe_ch = (p_safe_ch >= th).astype(int)
        pred_full_ch = (p_full_ch >= th).astype(int)
    else:
        th = float(champion["threshold"])
        pred_val_ch = (p_val_ch >= th).astype(int)
        pred_safe_ch = (p_safe_ch >= th).astype(int)
        pred_full_ch = (p_full_ch >= th).astype(int)

    metrics_rows = []
    for split, y, p, pred in [
        ("VALIDATION", y_val, p_val_ch, pred_val_ch),
        ("HOLDOUT_LABEL_SAFE", y_safe, p_safe_ch, pred_safe_ch),
        ("HOLDOUT_FULL", y_full, p_full_ch, pred_full_ch),
    ]:
        m = eval_binary(y, pred, y_prob=p, threshold=champion.get("threshold"))
        m.update({"temporal_split": split, "candidate_id": champion_id, "model_id": champion_model_id, "idea": champion_idea})
        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "02_champion_metrics_by_split.csv", index=False)

    # Predictions/errors.
    pred_safe_df = holdout_safe.copy()
    pred_safe_df["r4_score"] = p_safe_ch
    pred_safe_df["r4_pred"] = pred_safe_ch
    pred_safe_df.to_csv(output_dir / "08_champion_predictions_holdout_label_safe.csv", index=False)
    pred_safe_df[(pred_safe_df["is_fraud"] == 1) & (pred_safe_df["r4_pred"] == 0)].to_csv(output_dir / "09_champion_false_negatives_holdout_label_safe.csv", index=False)
    pred_safe_df[(pred_safe_df["is_fraud"] == 0) & (pred_safe_df["r4_pred"] == 1)].to_csv(output_dir / "10_champion_false_positives_holdout_label_safe.csv", index=False)

    # Feature importances/artifacts.
    stage1_names = get_feature_names(stage1_prep)
    pd.DataFrame({
        "feature": stage1_names,
        "importance_gain": stage1_model.booster_.feature_importance(importance_type="gain"),
        "importance_split": stage1_model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False).to_csv(output_dir / "11_stage1_feature_importance.csv", index=False)

    # If champion is a model we can persist.
    if champion_model_id in model_store:
        champ_model = model_store[champion_model_id]["model"]
        champ_prep = model_store[champion_model_id]["preprocessor"]
        champ_feature_cols = feature_cols
        champ_numeric_cols = numeric_cols
        champ_categorical_cols = categorical_cols
        champ_feature_names = get_feature_names(champ_prep)
        stage2_importance_path = output_dir / "12_stage2_feature_importance.csv"
        pd.DataFrame().to_csv(stage2_importance_path, index=False)
    elif champion_model_id in stage2_store:
        champ_model = stage2_store[champion_model_id]["model"]
        champ_prep = stage2_store[champion_model_id]["preprocessor"]
        champ_feature_cols = feature_cols + ["score_stage1"]
        champ_numeric_cols = numeric_cols + ["score_stage1"]
        champ_categorical_cols = categorical_cols
        champ_feature_names = get_feature_names(champ_prep)
        pd.DataFrame({
            "feature": champ_feature_names,
            "importance_gain": champ_model.booster_.feature_importance(importance_type="gain"),
            "importance_split": champ_model.booster_.feature_importance(importance_type="split"),
        }).sort_values("importance_gain", ascending=False).to_csv(output_dir / "12_stage2_feature_importance.csv", index=False)
    else:
        raise RuntimeError("Champion model object not found.")

    feature_schema = {
        "input_features_pre_transform": champ_feature_cols,
        "numeric_features": champ_numeric_cols,
        "categorical_features": champ_categorical_cols,
        "model_features_after_transform": champ_feature_names,
        "excluded_always": sorted(EXCLUDE_ALWAYS),
        "n_input_features": len(champ_feature_cols),
        "n_model_features_after_transform": len(champ_feature_names),
        "note": "Se campeão for two-stage, feature score_stage1 exige calcular stage1 antes do stage2.",
    }

    # Save candidate artifacts.
    model_final = candidate_dir / "model_lgbm_v3_r4_fp_squeeze_shadow.joblib"
    prep_final = candidate_dir / "preprocessor_lgbm_v3_r4_fp_squeeze_shadow.joblib"
    threshold_final = candidate_dir / "threshold_policy_exp012c_r4_fp_squeeze.json"
    features_final = candidate_dir / "features_lgbm_v3_r4_fp_squeeze_shadow.json"
    manifest_final = candidate_dir / "manifest_exp012c_r4_lgbm_fp_squeeze.json"
    stage1_model_final = candidate_dir / "stage1_model_lgbm_v3_r4.joblib"
    stage1_prep_final = candidate_dir / "stage1_preprocessor_lgbm_v3_r4.joblib"

    joblib.dump(champ_model, model_final)
    joblib.dump(champ_prep, prep_final)
    joblib.dump(stage1_model, stage1_model_final)
    joblib.dump(stage1_prep, stage1_prep_final)
    dump(champion, threshold_final)
    dump(feature_schema, features_final)
    dump(search_space, output_dir / "13_search_space.json")

    val_metrics = metrics_df[metrics_df["temporal_split"] == "VALIDATION"].iloc[0].to_dict()
    safe_metrics = metrics_df[metrics_df["temporal_split"] == "HOLDOUT_LABEL_SAFE"].iloc[0].to_dict()
    full_metrics = metrics_df[metrics_df["temporal_split"] == "HOLDOUT_FULL"].iloc[0].to_dict()

    objective_status = "VAL_RECALL_TARGET_MET" if val_metrics["recall"] >= args.target_recall else "VAL_RECALL_TARGET_NOT_MET"
    objective_status += "_SAFE_RECALL_TARGET_MET" if safe_metrics["recall"] >= args.target_recall else "_SAFE_RECALL_TARGET_NOT_MET"

    # Compare against R3 known FP baseline (2134) for convenience.
    r3_fp_baseline = 2134
    fp_delta_vs_r3 = int(safe_metrics["fp"]) - r3_fp_baseline

    summary = {
        "experiment": "EXP-012C-R4",
        "status": "DONE",
        "objective_status": objective_status,
        "target_recall": args.target_recall,
        "input_path": str(input_path),
        "input_md5": file_md5(input_path),
        "n_rows": int(len(df)),
        "n_fraud": int(df["is_fraud"].sum()),
        "n_normal": int((df["is_fraud"] == 0).sum()),
        "n_features_input": int(len(feature_cols)),
        "n_candidates": int(len(comp)),
        "champion_candidate_id": champion_id,
        "champion_model_id": champion_model_id,
        "champion_idea": champion_idea,
        "champion_policy": champion.get("policy"),
        "champion_threshold": champion.get("threshold"),
        "champion_alert_rate": champion.get("alert_rate"),
        "champion_validation": val_metrics,
        "champion_holdout_label_safe": safe_metrics,
        "champion_holdout_full": full_metrics,
        "safe_fp_delta_vs_exp012c_r3_hr01_threshold_0001": fp_delta_vs_r3,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "candidate_dir": str(candidate_dir),
        "model_path": str(model_final),
        "preprocessor_path": str(prep_final),
        "stage1_model_path": str(stage1_model_final),
        "stage1_preprocessor_path": str(stage1_prep_final),
    }
    dump(summary, output_dir / "00_run_summary.json")

    manifest = {
        "model_version": "exp012c_r4_lgbm_fp_squeeze_shadow",
        "status": "FP_SQUEEZE_SHADOW_CANDIDATE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "threshold_policy": champion,
        "notes": [
            "Artefato shadow. Não sobrescreve produção.",
            "Objetivo: reduzir FP mantendo recall >= 95% na validação.",
            "Explora exact threshold, top-k, hard negative mining, two-stage LGBM, segmented thresholds e strategy weights.",
            "Se recall no HOLDOUT_LABEL_SAFE ficar abaixo do alvo, não consolidar como baseline; usar como diagnóstico.",
        ],
    }
    dump(manifest, manifest_final)

    md = []
    md.append("# EXP-012C-R4 — LGBM High-Recall FP Squeeze")
    md.append("")
    md.append("## Champion")
    md.append(f"- candidate_id: `{champion_id}`")
    md.append(f"- model_id: `{champion_model_id}`")
    md.append(f"- idea: `{champion_idea}`")
    md.append(f"- policy: `{champion.get('policy')}`")
    md.append(f"- threshold: `{champion.get('threshold')}`")
    md.append(f"- alert_rate: `{champion.get('alert_rate')}`")
    md.append("")
    md.append("## Validation")
    for k in ["tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr", "roc_auc", "average_precision"]:
        md.append(f"- {k}: {val_metrics.get(k)}")
    md.append("")
    md.append("## Holdout label-safe")
    for k in ["tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr", "roc_auc", "average_precision"]:
        md.append(f"- {k}: {safe_metrics.get(k)}")
    md.append("")
    md.append("## Decisão sugerida")
    if val_metrics["recall"] >= args.target_recall and safe_metrics["recall"] >= args.target_recall:
        if safe_metrics["fp"] < r3_fp_baseline:
            md.append("APROVAR_COMO_NOVO_BASELINE_LGBM_HIGH_RECALL_PRE_MODULOS_EXTERNOS.")
        else:
            md.append("MANTER_EXP012C_R3_COMO_BASELINE_HIGH_RECALL; R4 não reduziu FP no holdout label-safe.")
    else:
        md.append("NÃO CONSOLIDAR COMO BASELINE; recall alvo não se manteve.")
    md.append("")
    md.append("## Próximo passo")
    md.append("Se aprovado, seguir para EXP-012D com módulos externos IF/BEH/SE usando este candidato como LGBM high-recall.")
    (output_dir / "14_recommendation.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
