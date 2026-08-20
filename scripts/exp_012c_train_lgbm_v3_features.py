#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-012C — Treino LGBM v3 com features históricas reais

Objetivo:
  Treinar um LightGBM shadow usando o dataset v3 criado no Big Data:
      hmo_ml.tb_pix_dataset_v3_features_180d_v1

Entrada default:
      dados/hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv

Pontos específicos do EXP-012C:
  - Não sobrescreve modelo produtivo.
  - Usa temporal_split do Hive: TRAIN, VALIDATION, HOLDOUT.
  - Usa sample_weight.
  - Seleciona threshold na VALIDATION.
  - Avalia HOLDOUT completo e HOLDOUT_LABEL_SAFE.
  - HOLDOUT_LABEL_SAFE considera somente datas do HOLDOUT até a última data
    com fraude confirmada no próprio HOLDOUT, para evitar cauda de normais ainda
    possivelmente não rotulada pelo MAF.

Saídas:
  resultados/experimentos/EXP-012C/
    00_run_summary.json
    01_metrics_by_split.csv
    02_threshold_sweep_validation.csv
    03_threshold_sweep_holdout_label_safe.csv
    04_threshold_sweep_holdout_full.csv
    05_feature_importance.csv
    06_predictions_validation.csv
    07_predictions_holdout_label_safe.csv
    08_predictions_holdout_full.csv
    09_false_negatives_holdout_label_safe.csv
    10_false_positives_holdout_label_safe.csv
    11_feature_schema.json
    12_score_distribution.csv
    13_metrics_by_strategy.csv
    14_training_config.json
    15_recommendation.md

  backend/artefatos_candidatos/exp012c_lgbm_v3/
    model_lgbm_v3_shadow.joblib
    preprocessor_lgbm_v3_shadow.joblib
    threshold_policy_exp012c.json
    features_lgbm_v3_shadow.json
    manifest_exp012c_lgbm_v3.json

Uso:
  python scripts/exp_012c_train_lgbm_v3_features.py

Teste rápido:
  python scripts/exp_012c_train_lgbm_v3_features.py --sample 30000
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


# =============================================================================
# Paths
# =============================================================================
SCRIPT_PATH = Path(__file__).resolve()

if (SCRIPT_PATH.parent.parent / "backend").exists():
    PROJECT_ROOT = SCRIPT_PATH.parent.parent
elif (SCRIPT_PATH.parent.parent.parent / "backend").exists():
    PROJECT_ROOT = SCRIPT_PATH.parent.parent.parent
else:
    PROJECT_ROOT = Path.cwd()

DADOS_DIR = PROJECT_ROOT / "dados"
RESULT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-012C"
CANDIDATE_DIR = PROJECT_ROOT / "backend" / "artefatos_candidatos" / "exp012c_lgbm_v3"

DEFAULT_INPUT = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv"


# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | EXP-012C | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EXP-012C")


# =============================================================================
# Feature policy
# =============================================================================
EXCLUDE_ALWAYS = {
    # label
    "is_fraud",

    # IDs/chaves
    "transaction_id",
    "cd_pix",
    "customer_id",
    "cd_cpf_pagador",
    "counterparty_id",
    "cd_cpf_cnpj_recebedor",
    "ds_chave_pix",
    "session_id",
    "ip_address",

    # datas cruas/metadata temporal direta
    "event_datetime",
    "dt_pix",
    "data_pix",
    "dt_carga",
    "dataset_created_at",
    "dataset_v3_created_at",
    "window_start_date",
    "window_end_date",
    "primeira_data_envio_recebedor_180d",

    # controle experimento/origem/split
    "temporal_split",
    "dataset_role",
    "source_dataset",
    "source_dataset_original",
    "sample_strategy",
    "sample_weight",
    "normal_sample_strategy",
    "normal_sample_source",

    # auditoria/labels MAF
    "label_status",
    "model_scope_status",
    "bank_direction",
    "triangulation_flag",
    "duplicate_conflict_flag",

    # raw MBK/retorno potencialmente instável ou com cardinalidade/semântica operacional
    "cd_retorno",
    "autcodret",
    "autdatref",
    "autdathorini",
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


# =============================================================================
# Helpers
# =============================================================================
def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def make_onehot_encoder() -> OneHotEncoder:
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
    if "data_pix" in df.columns:
        df["data_pix"] = pd.to_datetime(df["data_pix"], errors="coerce")
    else:
        df["data_pix"] = df["event_datetime"].dt.normalize()

    df = df[df["event_datetime"].notna()].copy()
    df = df[df["data_pix"].notna()].copy()

    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)

    if "temporal_split" not in df.columns:
        log.warning("temporal_split ausente. Criando split temporal 70/15/15 por event_datetime.")
        df = df.sort_values("event_datetime").reset_index(drop=True)
        n = len(df)
        train_end = int(n * 0.70)
        valid_end = int(n * 0.85)
        df["temporal_split"] = "HOLDOUT"
        df.loc[: train_end - 1, "temporal_split"] = "TRAIN"
        df.loc[train_end : valid_end - 1, "temporal_split"] = "VALIDATION"

    df["temporal_split"] = df["temporal_split"].astype(str).str.upper().str.strip()

    if "sample_weight" not in df.columns:
        df["sample_weight"] = 1.0
    df["sample_weight"] = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).clip(lower=0.05, upper=10.0)

    if "sample_strategy" not in df.columns:
        df["sample_strategy"] = "UNKNOWN"

    if "dataset_role" not in df.columns:
        df["dataset_role"] = np.where(df["is_fraud"] == 1, "POSITIVE_FRAUD", "NEGATIVE_NORMAL")

    if "transaction_id" in df.columns:
        df["transaction_id"] = df["transaction_id"].astype("string").str.strip()

    return df.sort_values(["event_datetime", "transaction_id"] if "transaction_id" in df.columns else ["event_datetime"]).reset_index(drop=True)


def infer_feature_columns(df: pd.DataFrame, no_categorical: bool = False) -> tuple[list[str], list[str], list[str]]:
    feature_cols: list[str] = []
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        if col in EXCLUDE_ALWAYS:
            continue

        lower = col.lower()
        if lower.endswith("_id") or "cpf" in lower or "cnpj" in lower:
            continue

        if pd.api.types.is_bool_dtype(df[col]):
            feature_cols.append(col)
            numeric_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            if int(df[col].notna().sum()) == 0:
                continue
            feature_cols.append(col)
            numeric_cols.append(col)
            continue

        if (not no_categorical) and col in CATEGORICAL_ALLOWLIST:
            nunique = int(df[col].astype("string").nunique(dropna=True))
            if 0 < nunique <= MAX_CATEGORICAL_CARDINALITY:
                feature_cols.append(col)
                categorical_cols.append(col)

    return feature_cols, numeric_cols, categorical_cols


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["temporal_split"] == "TRAIN"].copy()
    valid = df[df["temporal_split"] == "VALIDATION"].copy()
    holdout_full = df[df["temporal_split"] == "HOLDOUT"].copy()

    if train.empty or valid.empty or holdout_full.empty:
        raise RuntimeError(
            "Splits insuficientes. Esperado TRAIN, VALIDATION e HOLDOUT. "
            f"Encontrado: {df['temporal_split'].value_counts(dropna=False).to_dict()}"
        )

    max_holdout_fraud_date = holdout_full.loc[holdout_full["is_fraud"] == 1, "data_pix"].max()
    if pd.isna(max_holdout_fraud_date):
        raise RuntimeError("HOLDOUT não possui fraudes confirmadas; não é seguro avaliar.")

    holdout_label_safe = holdout_full[holdout_full["data_pix"] <= max_holdout_fraud_date].copy()

    for name, part in [("TRAIN", train), ("VALIDATION", valid), ("HOLDOUT_LABEL_SAFE", holdout_label_safe)]:
        n_pos = int(part["is_fraud"].sum())
        if n_pos == 0:
            raise RuntimeError(f"Split {name} não possui fraudes; não é seguro treinar/validar.")

    return train, valid, holdout_label_safe, holdout_full


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", make_onehot_encoder()),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names: list[str] = []
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


def sample_weights(y: pd.Series, base_weight: pd.Series, pos_multiplier: float) -> np.ndarray:
    y_arr = y.astype(int).values
    w = pd.to_numeric(base_weight, errors="coerce").fillna(1.0).clip(lower=0.05, upper=10.0).astype(float).values
    w = np.where(y_arr == 1, w * pos_multiplier, w)
    return w


def evaluate_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, label: str = "") -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both = len(np.unique(y_true)) > 1

    return {
        "label": label,
        "threshold": round(float(threshold), 8),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 8),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 8),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 8),
        "fpr": round(float(fp / max(fp + tn, 1)), 8),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 8) if has_both else None,
        "average_precision": round(float(average_precision_score(y_true, y_prob)), 8) if has_both else None,
    }


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in np.arange(0.005, 0.996, 0.005):
        rows.append(evaluate_threshold(y_true, y_prob, float(t)))
    return pd.DataFrame(rows)


def select_threshold_policy(y_valid: np.ndarray, p_valid: np.ndarray, target_recall: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    sweep = threshold_sweep(y_valid, p_valid)

    policies = []

    def add(name: str, selector: str, sub: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> None:
        if sub.empty:
            return
        row = sub.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()
        row["policy"] = name
        row["selector"] = selector
        policies.append(row)

    add(
        "BEST_F1_VALIDATION",
        "Maior F1 na validação",
        sweep,
        ["f1", "recall", "precision"],
        [False, False, False],
    )

    add(
        "PRECISION_GE_50_FPR_LE_1PCT",
        "Maior F1 com precision>=50% e FPR<=1%",
        sweep[(sweep["precision"] >= 0.50) & (sweep["fpr"] <= 0.01)],
        ["f1", "recall", "precision"],
        [False, False, False],
    )

    add(
        "PRECISION_GE_60_FPR_LE_1PCT",
        "Maior F1 com precision>=60% e FPR<=1%",
        sweep[(sweep["precision"] >= 0.60) & (sweep["fpr"] <= 0.01)],
        ["f1", "recall", "precision"],
        [False, False, False],
    )

    add(
        "FPR_LE_1PCT_MAX_RECALL",
        "Maior recall com FPR<=1%",
        sweep[sweep["fpr"] <= 0.01],
        ["recall", "f1", "precision"],
        [False, False, False],
    )

    add(
        "FPR_LE_05PCT_MAX_RECALL",
        "Maior recall com FPR<=0,5%",
        sweep[sweep["fpr"] <= 0.005],
        ["recall", "f1", "precision"],
        [False, False, False],
    )

    add(
        "TARGET_RECALL_VALIDATION",
        f"Maior threshold com recall>={target_recall}",
        sweep[sweep["recall"] >= target_recall],
        ["threshold", "precision", "f1"],
        [False, False, False],
    )

    policy_df = pd.DataFrame(policies)

    # Seleção oficial:
    # preferir policy com precision>=50 e FPR<=1; senão best_f1.
    preferred = policy_df[policy_df["policy"] == "PRECISION_GE_50_FPR_LE_1PCT"]
    if not preferred.empty:
        selected = preferred.iloc[0].to_dict()
        selected["selection_reason"] = "Selecionado por precision>=50%, FPR<=1% e maior F1 na validação."
    else:
        selected = policy_df[policy_df["policy"] == "BEST_F1_VALIDATION"].iloc[0].to_dict()
        selected["selection_reason"] = "Fallback para maior F1 na validação."

    return policy_df, selected


def save_predictions(df_part: pd.DataFrame, prob: np.ndarray, threshold: float, path: Path) -> pd.DataFrame:
    out = df_part.copy()
    out["lgbm_v3_score"] = prob
    out["lgbm_v3_pred"] = (prob >= threshold).astype(int)

    cols = [
        "transaction_id", "event_datetime", "data_pix", "temporal_split",
        "dataset_role", "sample_strategy", "is_fraud", "vl_pix",
        "lgbm_v3_score", "lgbm_v3_pred",
    ]
    cols = [c for c in cols if c in out.columns]
    out[cols].to_csv(path, index=False)
    return out


def metrics_by_strategy(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    group_cols = [c for c in ["temporal_split", "dataset_role", "sample_strategy"] if c in df.columns]
    if not group_cols:
        return pd.DataFrame()

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        y = g["is_fraud"].astype(int).values
        p = g["lgbm_v3_score"].astype(float).values
        m = evaluate_threshold(y, p, threshold)
        row = dict(zip(group_cols, keys))
        row.update({"n": len(g), **m})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, g in df.groupby("temporal_split", dropna=False):
        for is_fraud, gg in g.groupby("is_fraud", dropna=False):
            s = gg["lgbm_v3_score"].astype(float)
            rows.append({
                "temporal_split": split,
                "is_fraud": int(is_fraud),
                "n": len(gg),
                "min": float(s.min()),
                "p01": float(s.quantile(0.01)),
                "p05": float(s.quantile(0.05)),
                "p25": float(s.quantile(0.25)),
                "p50": float(s.quantile(0.50)),
                "p75": float(s.quantile(0.75)),
                "p95": float(s.quantile(0.95)),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
                "mean": float(s.mean()),
            })
    return pd.DataFrame(rows).sort_values(["temporal_split", "is_fraud"]).reset_index(drop=True)


def recommendation_md(summary: dict[str, Any]) -> str:
    h = summary["holdout_label_safe_operational"]
    lines = []
    lines.append("# EXP-012C — Recommendation")
    lines.append("")
    lines.append("## Status sugerido")
    if h["precision"] >= 0.50 and h["recall"] >= 0.50 and h["f1"] >= 0.50 and h["fpr"] <= 0.01:
        lines.append("CANDIDATO_FORTE_PARA_E2E_SHADOW")
    elif h["precision"] >= 0.50 and h["f1"] >= 0.40 and h["fpr"] <= 0.01:
        lines.append("CANDIDATO_PARA_E2E_SHADOW_COM_RESSALVAS")
    else:
        lines.append("DIAGNOSTICO_REQUER_NOVAS_FEATURES_OU_NOVO_SAMPLING")
    lines.append("")
    lines.append("## Holdout label-safe @ threshold operacional")
    for k in ["tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr", "roc_auc", "average_precision"]:
        lines.append(f"- {k}: {h.get(k)}")
    lines.append("")
    lines.append("## Observação sobre holdout")
    lines.append(
        f"O holdout completo vai até {summary['holdout_full_dt_max']}, mas a última fraude confirmada "
        f"no holdout ocorre em {summary['holdout_label_safe_dt_max']}. "
        "A decisão de modelo deve priorizar HOLDOUT_LABEL_SAFE para reduzir risco de avaliar normais ainda não maturados."
    )
    lines.append("")
    lines.append("## Próximo passo")
    lines.append("Executar EXP-012D — E2E Shadow v3 se as métricas forem superiores ao EXP-011C.")
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP-012C — Treino LGBM v3 com features históricas reais",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="CSV exportado da tabela v3.")
    parser.add_argument("--output-dir", default=str(RESULT_DIR), help="Diretório de resultados.")
    parser.add_argument("--candidate-dir", default=str(CANDIDATE_DIR), help="Diretório de artefatos candidatos.")
    parser.add_argument("--sample", type=int, default=None, help="Amostra opcional para teste rápido.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--pos-multiplier", type=float, default=4.0, help="Multiplicador de peso positivo.")
    parser.add_argument("--target-recall", type=float, default=0.50, help="Recall alvo opcional para política de threshold.")
    parser.add_argument("--no-categorical", action="store_true", help="Usar somente features numéricas.")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    candidate_dir = Path(args.candidate_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    print("=" * 80)
    print("EXP-012C — Treino LGBM v3 com features históricas reais")
    print("=" * 80)
    print(f"Input:         {input_path}")
    print(f"Output dir:    {output_dir}")
    print(f"Candidate dir: {candidate_dir}")
    print(f"MD5:           {file_md5(input_path)}")

    log.info("Carregando dataset v3...")
    df = pd.read_csv(input_path, low_memory=False)
    df = normalize_columns(df)

    if args.sample is not None and args.sample < len(df):
        log.info("Amostrando %d linhas...", args.sample)
        fraud = df[df["is_fraud"] == 1]
        normal = df[df["is_fraud"] == 0]
        n_fraud = min(len(fraud), max(1, int(args.sample * len(fraud) / len(df))))
        n_normal = args.sample - n_fraud
        df = pd.concat([
            fraud.sample(n=n_fraud, random_state=args.seed),
            normal.sample(n=min(n_normal, len(normal)), random_state=args.seed),
        ], axis=0).sort_values("event_datetime").reset_index(drop=True)

    train, valid, holdout_label_safe, holdout_full = split_dataset(df)

    log.info(
        "Dataset: rows=%d | fraud=%d | normal=%d | %s -> %s",
        len(df), int(df["is_fraud"].sum()), int((df["is_fraud"] == 0).sum()),
        df["data_pix"].min().date(), df["data_pix"].max().date(),
    )
    for name, part in [
        ("TRAIN", train),
        ("VALIDATION", valid),
        ("HOLDOUT_LABEL_SAFE", holdout_label_safe),
        ("HOLDOUT_FULL", holdout_full),
    ]:
        log.info(
            "%s: rows=%d | fraud=%d | normal=%d | %s -> %s",
            name, len(part), int(part["is_fraud"].sum()), int((part["is_fraud"] == 0).sum()),
            part["data_pix"].min().date(), part["data_pix"].max().date(),
        )

    feature_cols, numeric_cols, categorical_cols = infer_feature_columns(df, no_categorical=args.no_categorical)
    if not feature_cols:
        raise RuntimeError("Nenhuma feature selecionada.")

    log.info("Features selecionadas: %d (%d numéricas + %d categóricas)", len(feature_cols), len(numeric_cols), len(categorical_cols))

    X_train = train[feature_cols]
    y_train = train["is_fraud"].astype(int)
    X_valid = valid[feature_cols]
    y_valid = valid["is_fraud"].astype(int)
    X_hsafe = holdout_label_safe[feature_cols]
    y_hsafe = holdout_label_safe["is_fraud"].astype(int)
    X_hfull = holdout_full[feature_cols]
    y_hfull = holdout_full["is_fraud"].astype(int)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    log.info("Fit do preprocessor...")
    X_train_t = preprocessor.fit_transform(X_train)
    X_valid_t = preprocessor.transform(X_valid)
    X_hsafe_t = preprocessor.transform(X_hsafe)
    X_hfull_t = preprocessor.transform(X_hfull)

    model_feature_names = get_feature_names(preprocessor)

    w_train = sample_weights(y_train, train["sample_weight"], pos_multiplier=args.pos_multiplier)
    w_valid = pd.to_numeric(valid["sample_weight"], errors="coerce").fillna(1.0).astype(float).values

    config = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": 4500,
        "learning_rate": 0.015,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 60,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.75,
        "reg_lambda": 2.0,
        "random_state": args.seed,
        "n_jobs": -1,
        "scale_pos_weight": 1.0,
        "verbose": -1,
    }

    log.info("Treinando LGBM v3...")
    model = LGBMClassifier(**config)
    model.fit(
        X_train_t,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_valid_t, y_valid)],
        eval_sample_weight=[w_valid],
        eval_metric="average_precision",
        callbacks=[
            early_stopping(stopping_rounds=250, verbose=True),
            log_evaluation(period=200),
        ],
    )

    best_iter = getattr(model, "best_iteration_", None)
    log.info("Best iteration: %s", best_iter)

    p_train = model.predict_proba(X_train_t)[:, 1]
    p_valid = model.predict_proba(X_valid_t)[:, 1]
    p_hsafe = model.predict_proba(X_hsafe_t)[:, 1]
    p_hfull = model.predict_proba(X_hfull_t)[:, 1]

    policy_df, selected_policy = select_threshold_policy(y_valid.values, p_valid, args.target_recall)
    th = float(selected_policy["threshold"])
    log.info("Threshold operacional selecionado: %.6f | policy=%s", th, selected_policy.get("policy"))

    # Sweeps.
    sweep_valid = threshold_sweep(y_valid.values, p_valid)
    sweep_hsafe = threshold_sweep(y_hsafe.values, p_hsafe)
    sweep_hfull = threshold_sweep(y_hfull.values, p_hfull)

    sweep_valid.to_csv(output_dir / "02_threshold_sweep_validation.csv", index=False)
    sweep_hsafe.to_csv(output_dir / "03_threshold_sweep_holdout_label_safe.csv", index=False)
    sweep_hfull.to_csv(output_dir / "04_threshold_sweep_holdout_full.csv", index=False)

    # Métricas.
    metric_rows = []
    for split_name, y, p in [
        ("TRAIN", y_train.values, p_train),
        ("VALIDATION", y_valid.values, p_valid),
        ("HOLDOUT_LABEL_SAFE", y_hsafe.values, p_hsafe),
        ("HOLDOUT_FULL", y_hfull.values, p_hfull),
    ]:
        m = evaluate_threshold(y, p, th, label=split_name)
        m["temporal_split"] = split_name
        m["threshold_policy"] = selected_policy.get("policy")
        metric_rows.append(m)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "01_metrics_by_split.csv", index=False)

    # Predictions.
    pred_valid = save_predictions(valid, p_valid, th, output_dir / "06_predictions_validation.csv")
    pred_hsafe = save_predictions(holdout_label_safe, p_hsafe, th, output_dir / "07_predictions_holdout_label_safe.csv")
    pred_hfull = save_predictions(holdout_full, p_hfull, th, output_dir / "08_predictions_holdout_full.csv")

    fn_hsafe = pred_hsafe[(pred_hsafe["is_fraud"] == 1) & (pred_hsafe["lgbm_v3_pred"] == 0)].copy()
    fp_hsafe = pred_hsafe[(pred_hsafe["is_fraud"] == 0) & (pred_hsafe["lgbm_v3_pred"] == 1)].copy()
    if "vl_pix" in fn_hsafe.columns:
        fn_hsafe = fn_hsafe.sort_values("vl_pix", ascending=False)
    fp_hsafe = fp_hsafe.sort_values("lgbm_v3_score", ascending=False)

    fn_hsafe.to_csv(output_dir / "09_false_negatives_holdout_label_safe.csv", index=False)
    fp_hsafe.to_csv(output_dir / "10_false_positives_holdout_label_safe.csv", index=False)

    # Feature importance.
    importance = pd.DataFrame({
        "feature": model_feature_names,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    importance.to_csv(output_dir / "05_feature_importance.csv", index=False)

    feature_schema = {
        "input_features_pre_transform": feature_cols,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "model_features_after_transform": model_feature_names,
        "excluded_always": sorted(EXCLUDE_ALWAYS),
        "n_input_features": len(feature_cols),
        "n_model_features_after_transform": len(model_feature_names),
    }
    dump_json(feature_schema, output_dir / "11_feature_schema.json")

    # Distribuições.
    all_pred = pd.concat([
        train.assign(lgbm_v3_score=p_train, lgbm_v3_pred=(p_train >= th).astype(int)),
        valid.assign(lgbm_v3_score=p_valid, lgbm_v3_pred=(p_valid >= th).astype(int)),
        holdout_full.assign(lgbm_v3_score=p_hfull, lgbm_v3_pred=(p_hfull >= th).astype(int)),
    ], axis=0, ignore_index=True)

    score_distribution(all_pred).to_csv(output_dir / "12_score_distribution.csv", index=False)
    metrics_by_strategy(all_pred, th).to_csv(output_dir / "13_metrics_by_strategy.csv", index=False)
    dump_json(config, output_dir / "14_training_config.json")

    # Artefatos candidatos.
    model_path = candidate_dir / "model_lgbm_v3_shadow.joblib"
    preprocessor_path = candidate_dir / "preprocessor_lgbm_v3_shadow.joblib"
    threshold_path = candidate_dir / "threshold_policy_exp012c.json"
    features_path = candidate_dir / "features_lgbm_v3_shadow.json"
    manifest_path = candidate_dir / "manifest_exp012c_lgbm_v3.json"

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    dump_json(selected_policy, threshold_path)
    dump_json(feature_schema, features_path)

    hs_metrics = metrics_df[metrics_df["temporal_split"] == "HOLDOUT_LABEL_SAFE"].iloc[0].to_dict()
    hf_metrics = metrics_df[metrics_df["temporal_split"] == "HOLDOUT_FULL"].iloc[0].to_dict()

    elapsed = time.perf_counter() - t0
    summary = {
        "experiment": "EXP-012C",
        "status": "DONE",
        "input_path": str(input_path),
        "input_md5": file_md5(input_path),
        "output_dir": str(output_dir),
        "candidate_dir": str(candidate_dir),
        "n_rows": int(len(df)),
        "n_fraud": int(df["is_fraud"].sum()),
        "n_normal": int((df["is_fraud"] == 0).sum()),
        "n_train": int(len(train)),
        "n_validation": int(len(valid)),
        "n_holdout_label_safe": int(len(holdout_label_safe)),
        "n_holdout_full": int(len(holdout_full)),
        "holdout_label_safe_dt_min": str(holdout_label_safe["data_pix"].min().date()),
        "holdout_label_safe_dt_max": str(holdout_label_safe["data_pix"].max().date()),
        "holdout_full_dt_min": str(holdout_full["data_pix"].min().date()),
        "holdout_full_dt_max": str(holdout_full["data_pix"].max().date()),
        "n_features_input": int(len(feature_cols)),
        "n_model_features": int(len(model_feature_names)),
        "best_iteration": int(best_iter) if best_iter is not None else None,
        "pos_multiplier": float(args.pos_multiplier),
        "threshold_operational": th,
        "threshold_policy": selected_policy,
        "holdout_label_safe_operational": hs_metrics,
        "holdout_full_operational": hf_metrics,
        "elapsed_seconds": round(elapsed, 2),
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
    }

    dump_json(summary, output_dir / "00_run_summary.json")

    manifest = {
        "model_version": "exp012c_lgbm_v3_shadow",
        "status": "SHADOW_CANDIDATE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_md5": file_md5(input_path),
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "threshold_path": str(threshold_path),
        "features_path": str(features_path),
        "holdout_label_safe_operational": hs_metrics,
        "holdout_full_operational": hf_metrics,
        "notes": [
            "Artefato shadow. Não sobrescreve modelo produtivo.",
            "Decisão de modelo deve priorizar HOLDOUT_LABEL_SAFE por causa da cauda sem fraudes confirmadas no MAF.",
            "Requer EXP-012D E2E shadow antes de qualquer promoção.",
        ],
    }
    dump_json(manifest, manifest_path)

    (output_dir / "15_recommendation.md").write_text(recommendation_md(summary), encoding="utf-8")

    print("\n" + "=" * 80)
    print("EXP-012C CONCLUÍDO")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nArtefatos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_metrics_by_split.csv",
        output_dir / "05_feature_importance.csv",
        output_dir / "09_false_negatives_holdout_label_safe.csv",
        output_dir / "10_false_positives_holdout_label_safe.csv",
        output_dir / "15_recommendation.md",
        model_path,
        preprocessor_path,
        threshold_path,
        manifest_path,
    ]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
