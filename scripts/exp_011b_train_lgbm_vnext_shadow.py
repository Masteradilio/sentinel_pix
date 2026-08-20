#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-011B — Treino Shadow LGBM vNext no Dataset v2 Enriquecido

Objetivo:
  Treinar um LightGBM shadow, SEM sobrescrever o modelo produtivo atual,
  usando o dataset enriquecido do EXP-010G-R2.

Entrada default:
  dados/hmo_ml_tb_pix_dataset_v2_180d_v1_enriched.csv

Saídas:
  resultados/experimentos/EXP-011B/
    00_run_summary.json
    01_metrics_by_split.csv
    02_threshold_sweep_validation.csv
    03_threshold_sweep_holdout.csv
    04_feature_importance.csv
    05_predictions_validation.csv
    06_predictions_holdout.csv
    07_false_negatives_holdout.csv
    08_false_positives_holdout.csv
    09_feature_schema.json
    10_recommendation.md
    11_metrics_by_strategy.csv
    12_score_distribution.csv
    13_training_config.json

  backend/artefatos_candidatos/exp011b_lgbm_vnext/
    model_lgbm_vnext_shadow.joblib
    preprocessor_lgbm_vnext_shadow.joblib
    thresholds_lgbm_vnext_shadow.json
    features_lgbm_vnext_shadow.json
    manifest_lgbm_vnext_shadow.json

Uso:
  python scripts/exp_011b_train_lgbm_vnext_shadow.py

Teste rápido:
  python scripts/exp_011b_train_lgbm_vnext_shadow.py --sample 30000

Observação:
  Este script é SHADOW. Ele não escreve em backend/artefatos/model_lightgbm.joblib.
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# Windows UTF-8
if sys.platform == "win32":
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
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
RESULT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-011B"
CANDIDATE_DIR = PROJECT_ROOT / "backend" / "artefatos_candidatos" / "exp011b_lgbm_vnext"

DEFAULT_INPUT = DADOS_DIR / "hmo_ml_tb_pix_dataset_v2_180d_v1_enriched.csv"

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | EXP-011B | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EXP-011B")


# =============================================================================
# Feature policy
# =============================================================================

# Colunas que nunca entram como features por risco de leakage, ID puro ou metadata.
EXCLUDE_ALWAYS = {
    # Label / alvo
    "is_fraud",

    # IDs e chaves diretas
    "transaction_id",
    "cd_pix",
    "customer_id",
    "cd_cpf_pagador",
    "counterparty_id",
    "cd_cpf_cnpj_recebedor",
    "ds_chave_pix",
    "session_id",
    "ip_address",

    # Datas brutas / metadados temporais diretos
    "event_datetime",
    "dt_pix",
    "data_pix",
    "dt_carga",
    "dataset_created_at",
    "window_start_date",
    "window_end_date",

    # Controle de experimento / origem / split
    "temporal_split",
    "dataset_role",
    "source_dataset",
    "source_dataset_original",
    "sample_strategy",
    "sample_weight",
    "normal_sample_strategy",
    "normal_sample_source",

    # Campos de label MAF e auditoria
    "label_status",
    "model_scope_status",
    "bank_direction",
    "triangulation_flag",
    "duplicate_conflict_flag",

    # Campos que são essencialmente aliases pós-EXP010G
    "cd_retorno",
    "autcodret",
    "autdatref",
    "autdathorini",
}

# Categóricas de baixo risco que podem entrar via one-hot quando existirem.
CATEGORICAL_ALLOWLIST = {
    "ds_tipo_chave",
    "ds_tipo_chave_norm",
    "periodo_dia",
    "device_name",
    "app_version",
    "metodo_autenticacao",
    "is_agendamento_recorrente",
    "ds_sexo",
    "ds_estado_civil",
    "ds_segmento",
}

# Categóricas com cardinalidade alta são descartadas automaticamente.
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


def safe_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def make_onehot_encoder() -> OneHotEncoder:
    """Compatibilidade sklearn antigo/novo."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            min_frequency=None,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "event_datetime" not in df.columns and "dt_pix" in df.columns:
        df["event_datetime"] = df["dt_pix"]
    if "transaction_id" not in df.columns and "cd_pix" in df.columns:
        df["transaction_id"] = df["cd_pix"]

    df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
    df = df[df["event_datetime"].notna()].copy()

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

    return df.sort_values(["event_datetime", "transaction_id"] if "transaction_id" in df.columns else ["event_datetime"]).reset_index(drop=True)


def infer_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """
    Retorna:
      feature_cols, numeric_cols, categorical_cols
    """
    feature_cols: list[str] = []
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        if col in EXCLUDE_ALWAYS:
            continue

        # Excluir qualquer coluna textual ID-like não prevista.
        lower = col.lower()
        if lower.endswith("_id") or "cpf" in lower or "cnpj" in lower:
            continue

        # Numéricas entram por padrão.
        if pd.api.types.is_numeric_dtype(df[col]):
            # Evitar colunas com quase nenhum valor útil.
            non_null = int(df[col].notna().sum())
            if non_null == 0:
                continue
            feature_cols.append(col)
            numeric_cols.append(col)
            continue

        # Booleanas também entram.
        if pd.api.types.is_bool_dtype(df[col]):
            feature_cols.append(col)
            numeric_cols.append(col)
            continue

        # Categóricas só por allowlist e baixa cardinalidade.
        if col in CATEGORICAL_ALLOWLIST:
            nunique = int(df[col].astype("string").nunique(dropna=True))
            if 0 < nunique <= MAX_CATEGORICAL_CARDINALITY:
                feature_cols.append(col)
                categorical_cols.append(col)

    return feature_cols, numeric_cols, categorical_cols


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["temporal_split"] == "TRAIN"].copy()
    valid = df[df["temporal_split"] == "VALIDATION"].copy()
    holdout = df[df["temporal_split"] == "HOLDOUT"].copy()

    if train.empty or valid.empty or holdout.empty:
        raise RuntimeError(
            "Splits insuficientes. Esperado temporal_split com TRAIN, VALIDATION e HOLDOUT. "
            f"Encontrado: {df['temporal_split'].value_counts(dropna=False).to_dict()}"
        )

    for name, part in [("TRAIN", train), ("VALIDATION", valid), ("HOLDOUT", holdout)]:
        n_pos = int(part["is_fraud"].sum())
        if n_pos == 0:
            raise RuntimeError(f"Split {name} não possui fraudes; não é seguro treinar/validar.")

    return train, valid, holdout


def compute_class_adjusted_weights(y: pd.Series, base_weight: pd.Series) -> np.ndarray:
    """
    Combina sample_weight do dataset com ajuste moderado de classe.

    Em vez de usar scale_pos_weight extremo no modelo, multiplicamos pesos positivos
    por sqrt(n_neg/n_pos). Isso ajuda recall sem explodir FP.
    """
    y_arr = y.astype(int).values
    w = pd.to_numeric(base_weight, errors="coerce").fillna(1.0).astype(float).values
    n_pos = max(int((y_arr == 1).sum()), 1)
    n_neg = max(int((y_arr == 0).sum()), 1)
    pos_mult = float(np.sqrt(n_neg / n_pos))
    w = np.where(y_arr == 1, w * pos_mult, w)
    return w


def evaluate_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, label: str = "") -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    has_both = len(np.unique(y_true)) > 1
    return {
        "label": label,
        "threshold": round(float(threshold), 8),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": round(float(precision), 8),
        "recall": round(float(recall), 8),
        "f1": round(float(f1), 8),
        "fpr": round(float(fp / max(fp + tn, 1)), 8),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 8) if has_both else None,
        "average_precision": round(float(average_precision_score(y_true, y_prob)), 8) if has_both else None,
    }


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in np.arange(0.005, 0.996, 0.005):
        m = evaluate_threshold(y_true, y_prob, float(t))
        rows.append({
            "threshold": m["threshold"],
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tn": m["tn"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "fpr": m["fpr"],
        })
    return pd.DataFrame(rows)


def find_thresholds(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float) -> dict[str, Any]:
    sweep = threshold_sweep(y_true, y_prob)

    best_f1_row = sweep.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0].to_dict()

    # Threshold mais alto que atinge target_recall.
    eligible = sweep[sweep["recall"] >= target_recall].copy()
    if eligible.empty:
        recall_row = sweep.sort_values(["recall", "precision", "f1"], ascending=[False, False, False]).iloc[0].to_dict()
    else:
        recall_row = eligible.sort_values(["threshold", "precision", "f1"], ascending=[False, False, False]).iloc[0].to_dict()

    # Threshold operacional balanceado: maior F1 com recall >= 90% do target quando possível.
    relaxed_target = max(0.50, min(target_recall * 0.90, target_recall))
    eligible_relaxed = sweep[sweep["recall"] >= relaxed_target].copy()
    if eligible_relaxed.empty:
        operational_row = best_f1_row
    else:
        operational_row = eligible_relaxed.sort_values(["f1", "precision"], ascending=[False, False]).iloc[0].to_dict()

    return {
        "best_f1": best_f1_row,
        "target_recall": recall_row,
        "operational": operational_row,
        "target_recall_requested": target_recall,
        "relaxed_target_recall": relaxed_target,
    }


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


def save_predictions(df_part: pd.DataFrame, prob: np.ndarray, threshold: float, path: Path) -> pd.DataFrame:
    out = df_part.copy()
    out["lgbm_vnext_score"] = prob
    out["lgbm_vnext_pred"] = (prob >= threshold).astype(int)

    base_cols = [
        "transaction_id", "event_datetime", "temporal_split", "dataset_role",
        "sample_strategy", "is_fraud", "vl_pix", "lgbm_vnext_score", "lgbm_vnext_pred",
    ]
    cols = [c for c in base_cols if c in out.columns]
    extra = [c for c in ["customer_id", "cd_cpf_pagador", "counterparty_id", "cd_cpf_cnpj_recebedor"] if c in out.columns]
    cols = cols[:2] + extra + cols[2:]

    out[cols].to_csv(path, index=False)
    return out


def metrics_by_strategy(df: pd.DataFrame, prob_col: str, threshold: float) -> pd.DataFrame:
    rows = []
    group_cols = [c for c in ["temporal_split", "dataset_role", "sample_strategy"] if c in df.columns]
    if not group_cols:
        return pd.DataFrame()

    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        y = g["is_fraud"].astype(int).values
        p = g[prob_col].astype(float).values
        m = evaluate_threshold(y, p, threshold)
        row = dict(zip(group_cols, keys))
        row.update({
            "n": len(g),
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "fpr": m["fpr"],
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def score_distribution(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    rows = []
    group_cols = [c for c in ["temporal_split", "is_fraud"] if c in df.columns]
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        scores = g[prob_col].astype(float)
        row = dict(zip(group_cols, keys))
        row.update({
            "n": len(g),
            "min": float(scores.min()),
            "p01": float(scores.quantile(0.01)),
            "p05": float(scores.quantile(0.05)),
            "p25": float(scores.quantile(0.25)),
            "p50": float(scores.quantile(0.50)),
            "p75": float(scores.quantile(0.75)),
            "p95": float(scores.quantile(0.95)),
            "p99": float(scores.quantile(0.99)),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def recommendation_md(summary: dict[str, Any], holdout_metrics: dict[str, Any], threshold_info: dict[str, Any]) -> str:
    lines = []
    lines.append("# EXP-011B — LGBM vNext Shadow Recommendation")
    lines.append("")
    lines.append("## Status")
    if holdout_metrics["recall"] >= 0.90 and holdout_metrics["precision"] >= 0.50:
        status = "CANDIDATO_FORTE_PARA_E2E_SHADOW"
    elif holdout_metrics["recall"] >= 0.75:
        status = "CANDIDATO_PARA_E2E_SHADOW_COM_AJUSTES"
    else:
        status = "DIAGNOSTICO_MODELO_SHADOW_REQUER_AJUSTES"
    lines.append(status)
    lines.append("")
    lines.append("## Resumo")
    lines.append(f"- Input: `{summary['input_path']}`")
    lines.append(f"- Linhas: {summary['n_rows']}")
    lines.append(f"- Fraudes: {summary['n_fraud']}")
    lines.append(f"- Normais: {summary['n_normal']}")
    lines.append(f"- Features finais pós-preprocessor: {summary['n_model_features']}")
    lines.append(f"- Threshold operacional: {summary['threshold_operational']}")
    lines.append("")
    lines.append("## Holdout @ threshold operacional")
    for k in ["tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr", "roc_auc", "average_precision"]:
        lines.append(f"- {k}: {holdout_metrics.get(k)}")
    lines.append("")
    lines.append("## Decisão")
    lines.append(
        "Este modelo foi salvo apenas como candidato shadow. "
        "A próxima etapa é executar E2E com o DecisionEngine em modo candidato, "
        "sem sobrescrever o modelo produtivo."
    )
    lines.append("")
    lines.append("## Thresholds selecionados na validação")
    lines.append("```json")
    lines.append(json.dumps(threshold_info, ensure_ascii=False, indent=2, default=str))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP-011B — Treino Shadow LGBM vNext",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="CSV enriquecido do EXP-010G-R2.")
    parser.add_argument("--output-dir", default=str(RESULT_DIR), help="Diretório de resultados.")
    parser.add_argument("--candidate-dir", default=str(CANDIDATE_DIR), help="Diretório dos artefatos candidatos shadow.")
    parser.add_argument("--sample", type=int, default=None, help="Amostra opcional para teste rápido.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--target-recall", type=float, default=0.95, help="Recall alvo na validação para threshold operacional.")
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
    print("EXP-011B — Treino Shadow LGBM vNext")
    print("=" * 80)
    print(f"Input:         {input_path}")
    print(f"Output dir:    {output_dir}")
    print(f"Candidate dir: {candidate_dir}")
    print(f"MD5:           {file_md5(input_path)}")

    log.info("Carregando dataset...")
    df = pd.read_csv(input_path, low_memory=False)
    df = normalize_columns(df)

    if args.sample is not None and args.sample < len(df):
        log.info("Amostrando %d linhas com preservação temporal aproximada...", args.sample)
        # Garante presença de fraudes e preserva ordenação final.
        fraud = df[df["is_fraud"] == 1]
        normal = df[df["is_fraud"] == 0]
        n_fraud = min(len(fraud), max(1, int(args.sample * len(fraud) / len(df))))
        n_normal = args.sample - n_fraud
        df = pd.concat([
            fraud.sample(n=n_fraud, random_state=args.seed),
            normal.sample(n=min(n_normal, len(normal)), random_state=args.seed),
        ], axis=0).sort_values("event_datetime").reset_index(drop=True)

    train, valid, holdout = split_dataset(df)

    log.info(
        "Dataset: %d rows | %d fraudes | %d normais | %s -> %s",
        len(df), int(df["is_fraud"].sum()), int((df["is_fraud"] == 0).sum()),
        df["event_datetime"].min(), df["event_datetime"].max(),
    )
    for name, part in [("TRAIN", train), ("VALIDATION", valid), ("HOLDOUT", holdout)]:
        log.info(
            "%s: %d rows | %d fraudes | %d normais | %s -> %s",
            name, len(part), int(part["is_fraud"].sum()),
            int((part["is_fraud"] == 0).sum()),
            part["event_datetime"].min(), part["event_datetime"].max(),
        )

    feature_cols, numeric_cols, categorical_cols = infer_feature_columns(df)
    if args.no_categorical:
        feature_cols = numeric_cols
        categorical_cols = []

    if not feature_cols:
        raise RuntimeError("Nenhuma feature selecionada. Verifique o CSV enriquecido.")

    log.info("Features selecionadas: %d (%d numéricas + %d categóricas)", len(feature_cols), len(numeric_cols), len(categorical_cols))

    X_train = train[feature_cols]
    y_train = train["is_fraud"].astype(int)
    X_valid = valid[feature_cols]
    y_valid = valid["is_fraud"].astype(int)
    X_holdout = holdout[feature_cols]
    y_holdout = holdout["is_fraud"].astype(int)

    w_train = compute_class_adjusted_weights(y_train, train["sample_weight"])
    w_valid = pd.to_numeric(valid["sample_weight"], errors="coerce").fillna(1.0).astype(float).values

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    log.info("Fit do preprocessor...")
    X_train_t = preprocessor.fit_transform(X_train)
    X_valid_t = preprocessor.transform(X_valid)
    X_holdout_t = preprocessor.transform(X_holdout)

    model_feature_names = get_feature_names(preprocessor)

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    spw = float(n_neg / max(n_pos, 1))

    config = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": 4000,
        "learning_rate": 0.015,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": max(10, min(50, n_pos // 5)),
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.25,
        "reg_lambda": 1.25,
        "random_state": args.seed,
        "n_jobs": -1,
        "scale_pos_weight": 1.0,  # pesos de classe já estão em sample_weight
        "verbose": -1,
    }

    log.info("Treinando LightGBM shadow...")
    model = LGBMClassifier(**config)
    model.fit(
        X_train_t,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_valid_t, y_valid)],
        eval_sample_weight=[w_valid],
        eval_metric="average_precision",
        callbacks=[
            early_stopping(stopping_rounds=200, verbose=True),
            log_evaluation(period=200),
        ],
    )

    best_iter = getattr(model, "best_iteration_", None)
    log.info("Best iteration: %s", best_iter)

    p_train = model.predict_proba(X_train_t)[:, 1]
    p_valid = model.predict_proba(X_valid_t)[:, 1]
    p_holdout = model.predict_proba(X_holdout_t)[:, 1]

    threshold_info = find_thresholds(y_valid.values, p_valid, args.target_recall)
    th_best_f1 = float(threshold_info["best_f1"]["threshold"])
    th_target = float(threshold_info["target_recall"]["threshold"])
    th_operational = float(threshold_info["operational"]["threshold"])

    log.info("Threshold best F1 validation: %.6f", th_best_f1)
    log.info("Threshold target recall validation: %.6f", th_target)
    log.info("Threshold operational validation: %.6f", th_operational)

    # Métricas por split nos thresholds principais.
    rows = []
    for split_name, y, p in [
        ("TRAIN", y_train.values, p_train),
        ("VALIDATION", y_valid.values, p_valid),
        ("HOLDOUT", y_holdout.values, p_holdout),
    ]:
        for th_name, th in [
            ("best_f1_validation", th_best_f1),
            ("target_recall_validation", th_target),
            ("operational", th_operational),
        ]:
            m = evaluate_threshold(y, p, th, label=split_name)
            m["temporal_split"] = split_name
            m["threshold_name"] = th_name
            rows.append(m)

    metrics_by_split = pd.DataFrame(rows)
    metrics_by_split.to_csv(output_dir / "01_metrics_by_split.csv", index=False)

    sweep_valid = threshold_sweep(y_valid.values, p_valid)
    sweep_holdout = threshold_sweep(y_holdout.values, p_holdout)
    sweep_valid.to_csv(output_dir / "02_threshold_sweep_validation.csv", index=False)
    sweep_holdout.to_csv(output_dir / "03_threshold_sweep_holdout.csv", index=False)

    pred_valid = save_predictions(valid, p_valid, th_operational, output_dir / "05_predictions_validation.csv")
    pred_holdout = save_predictions(holdout, p_holdout, th_operational, output_dir / "06_predictions_holdout.csv")
    pred_train = train.copy()
    pred_train["lgbm_vnext_score"] = p_train
    pred_train["lgbm_vnext_pred"] = (p_train >= th_operational).astype(int)

    # Erros do holdout.
    fn_holdout = pred_holdout[(pred_holdout["is_fraud"] == 1) & (pred_holdout["lgbm_vnext_pred"] == 0)].copy()
    fp_holdout = pred_holdout[(pred_holdout["is_fraud"] == 0) & (pred_holdout["lgbm_vnext_pred"] == 1)].copy()
    if "vl_pix" in fn_holdout.columns:
        fn_holdout = fn_holdout.sort_values("vl_pix", ascending=False)
    fp_holdout = fp_holdout.sort_values("lgbm_vnext_score", ascending=False)
    fn_holdout.to_csv(output_dir / "07_false_negatives_holdout.csv", index=False)
    fp_holdout.to_csv(output_dir / "08_false_positives_holdout.csv", index=False)

    # Feature importance.
    booster = model.booster_
    importance = pd.DataFrame({
        "feature": model_feature_names,
        "importance_gain": booster.feature_importance(importance_type="gain"),
        "importance_split": booster.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    importance.to_csv(output_dir / "04_feature_importance.csv", index=False)

    # Schema.
    feature_schema = {
        "input_features_pre_transform": feature_cols,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "model_features_after_transform": model_feature_names,
        "excluded_always": sorted(EXCLUDE_ALWAYS),
        "n_input_features": len(feature_cols),
        "n_model_features_after_transform": len(model_feature_names),
    }
    safe_json_dump(feature_schema, output_dir / "09_feature_schema.json")

    # Métricas por estratégia.
    all_pred = pd.concat([pred_train, pred_valid, pred_holdout], axis=0, ignore_index=True)
    metrics_strategy = metrics_by_strategy(all_pred, "lgbm_vnext_score", th_operational)
    metrics_strategy.to_csv(output_dir / "11_metrics_by_strategy.csv", index=False)

    score_dist = score_distribution(all_pred, "lgbm_vnext_score")
    score_dist.to_csv(output_dir / "12_score_distribution.csv", index=False)

    safe_json_dump(config, output_dir / "13_training_config.json")

    holdout_operational = metrics_by_split[
        (metrics_by_split["temporal_split"] == "HOLDOUT") &
        (metrics_by_split["threshold_name"] == "operational")
    ].iloc[0].to_dict()

    # Artefatos candidatos shadow.
    model_path = candidate_dir / "model_lgbm_vnext_shadow.joblib"
    preprocessor_path = candidate_dir / "preprocessor_lgbm_vnext_shadow.joblib"
    thresholds_path = candidate_dir / "thresholds_lgbm_vnext_shadow.json"
    features_path = candidate_dir / "features_lgbm_vnext_shadow.json"
    manifest_path = candidate_dir / "manifest_lgbm_vnext_shadow.json"

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    safe_json_dump(threshold_info, thresholds_path)
    safe_json_dump(feature_schema, features_path)

    elapsed = time.perf_counter() - t0
    run_summary = {
        "experiment": "EXP-011B",
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
        "n_holdout": int(len(holdout)),
        "n_features_input": int(len(feature_cols)),
        "n_model_features": int(len(model_feature_names)),
        "best_iteration": int(best_iter) if best_iter is not None else None,
        "threshold_best_f1_validation": th_best_f1,
        "threshold_target_recall_validation": th_target,
        "threshold_operational": th_operational,
        "holdout_operational": holdout_operational,
        "elapsed_seconds": round(elapsed, 2),
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
    }
    safe_json_dump(run_summary, output_dir / "00_run_summary.json")

    manifest = {
        "model_version": "exp011b_lgbm_vnext_shadow",
        "status": "SHADOW_CANDIDATE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_md5": file_md5(input_path),
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "thresholds_path": str(thresholds_path),
        "features_path": str(features_path),
        "holdout_operational": holdout_operational,
        "notes": [
            "Artefato shadow. Não sobrescreve backend/artefatos/model_lightgbm.joblib.",
            "Requer EXP-011C para replay E2E no DecisionEngine antes de qualquer promoção.",
        ],
    }
    safe_json_dump(manifest, manifest_path)

    rec = recommendation_md(run_summary, holdout_operational, threshold_info)
    (output_dir / "10_recommendation.md").write_text(rec, encoding="utf-8")

    print("\n" + "=" * 80)
    print("EXP-011B CONCLUÍDO")
    print("=" * 80)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    print("\nArtefatos:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_metrics_by_split.csv",
        output_dir / "04_feature_importance.csv",
        output_dir / "10_recommendation.md",
        model_path,
        preprocessor_path,
        thresholds_path,
        features_path,
        manifest_path,
    ]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
