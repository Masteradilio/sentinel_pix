"""
experimentos/exp_005a_lgbm_recall/run_exp_005a_lgbm_recall.py

EXP-005A — LGBM Recall-Oriented v6.2

Versao robusta:
  - Nao chama PipelineOrquestrador.
  - Nao importa DecisionEngine.
  - Nao le scoring_config.json.
  - Nao faz swap de artefatos.
  - Treina variantes LightGBM e gera artefatos completos de treino/seleção.

Motivo:
  O E2E depende do runtime atual do projeto. Se decision_engine.py,
  scoring_config.json ou PipelineOrquestrador estiverem instaveis, o
  experimento de treino nao deve quebrar. O E2E completo fica para o
  EXP-005B, depois de consolidar os artefatos candidatos.

Entradas:
  dados/base_treino_final.csv
  backend/artefatos/lgbm_features.json

Saidas:
  resultados/experimentos/EXP-005A/
    01_tabela_modelos.csv
    02_threshold_sweep_lgbm.csv
    03_avaliacao_e2e_modelo_candidato.json
    04_validacao_cruzada.json
    05_shap_drift_report.md
    06_conclusao_executiva.md
    manifest_artefatos_candidatos.json

  backend/artefatos_candidatos/exp_005a_lgbm_v6_2_recall/
    lgbm_v6_2_recall_candidate.joblib
    lgbm_features_v6_2.json
    thresholds_lgbm_v6_2.json
    manifest_exp_005a.json

Uso:
  python experimentos\\exp_005a_lgbm_recall\\run_exp_005a_lgbm_recall.py

Opcoes:
  --precision-min 0.94
  --holdout-frac 0.20
  --validation-seed 123
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =========================================================
# PATHS
# =========================================================

EXP_DIR = Path(__file__).resolve().parent


def _find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists() and (p / "experimentos").exists():
            return p
    return start.parent.parent


PROJECT_ROOT = _find_project_root(EXP_DIR)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

BACKEND_DIR = PROJECT_ROOT / "backend"
ARTEFATOS_DIR = BACKEND_DIR / "artefatos"
CANDIDATE_DIR = BACKEND_DIR / "artefatos_candidatos" / "exp_005a_lgbm_v6_2_recall"
DADOS_DIR = PROJECT_ROOT / "dados"
DATASET_PATH = DADOS_DIR / "base_treino_final.csv"
OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-005A"


# =========================================================
# LOGGING
# =========================================================

EXP_ID = "EXP-005A"
EXP_TITLE = "LGBM Recall-Oriented v6.2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(EXP_ID)


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


# =========================================================
# DEPENDENCIAS ML
# =========================================================

try:
    import lightgbm as lgb
except Exception as exc:
    raise RuntimeError(
        "LightGBM nao esta instalado. Instale com: pip install lightgbm"
    ) from exc

try:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except Exception as exc:
    raise RuntimeError(
        "scikit-learn nao esta instalado. Instale com: pip install scikit-learn"
    ) from exc


# =========================================================
# BASELINE DOCUMENTAL DA FASE 2
# =========================================================

BASELINE_FASE2 = {
    "TP": 346,
    "FP": 15,
    "FN": 9,
    "Precision": 0.958449,
    "Recall": 0.974648,
    "F1": 0.9665,
    "FPR": 0.002657,
    "descricao": "Baseline pos-FASE 1 consolidado, apos EXP-004-FINAL V1_GUARD_CONTEXTUAL",
}


# =========================================================
# DATACLASSES
# =========================================================

@dataclass
class OperatingPoint:
    variant_id: str
    policy_id: str
    threshold: float
    precision: float
    recall: float
    f1: float
    fp: int
    fn: int
    tp: int
    tn: int
    fpr: float
    selected_reason: str


@dataclass
class ModelResult:
    variant_id: str
    label: str
    feature_set: str
    n_features: int
    train_rows: int
    holdout_rows: int
    holdout_frauds: int
    auc: float
    ap: float
    best_policy_id: str
    threshold_selected: float
    precision_selected: float
    recall_selected: float
    f1_selected: float
    fp_selected: int
    fn_selected: int
    tp_selected: int
    tn_selected: int
    fpr_selected: float
    selected_reason: str
    selected_for_candidate: bool
    uses_new_exp005_features: bool
    params: dict[str, Any]
    model_path: str
    features_path: str


# =========================================================
# JSON / SAFE HELPERS
# =========================================================

def _safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(x) for x in obj]
    if isinstance(obj, tuple):
        return [_safe_json(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return [_safe_json(x) for x in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if obj is None:
        return None
    return obj


def safe_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe_json(obj), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# =========================================================
# DATA HELPERS
# =========================================================

def _detect_datetime_col(df: pd.DataFrame) -> str | None:
    for col in ["event_datetime", "dt_pix", "data_hora", "timestamp"]:
        if col in df.columns:
            return col
    return None


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _prepare_label(df: pd.DataFrame) -> pd.Series:
    if "is_fraud" not in df.columns:
        raise ValueError("Dataset precisa conter coluna is_fraud.")
    return pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)


def _load_lgbm_feature_list() -> list[str]:
    path = ARTEFATOS_DIR / "lgbm_features.json"

    if not path.exists():
        raise FileNotFoundError(f"Nao encontrei {path}")

    obj = _read_json(path)

    if isinstance(obj, list):
        return [str(x) for x in obj]

    if isinstance(obj, dict):
        for key in ["features", "feature_names", "lgbm_features"]:
            if key in obj and isinstance(obj[key], list):
                return [str(x) for x in obj[key]]

    raise ValueError(f"Formato inesperado em {path}")


def _ensure_numeric_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)

    for col in feature_cols:
        if col in df.columns:
            x[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            x[col] = np.nan

    x = x.replace([np.inf, -np.inf], np.nan)
    med = x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = x.fillna(med).fillna(0.0)

    return x.astype(float)


# =========================================================
# FEATURE ENGINEERING EXP-005A
# =========================================================

def add_exp005a_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    vl = _num(df, "vl_pix", 0.0).clip(lower=0)
    rel = _num(df, "qt_tempo_relacionamento_mes", 999.0).clip(lower=0)
    idade = _num(df, "nr_idade", 0.0).clip(lower=0)
    first_receiver = _num(df, "first_receiver_flag", 0.0).clip(0, 1)
    pix_random = _num(df, "pix_key_random_flag", 0.0).clip(0, 1)
    burst = _num(df, "burst_30m_flag", 0.0).clip(0, 1)
    tx30 = _num(df, "tx_count_prev_30m", 0.0).clip(lower=0)
    distinct_receivers = _num(df, "distinct_receivers_so_far", 0.0).clip(lower=0)

    if "log_vl_pix" not in df.columns:
        df["log_vl_pix"] = np.log1p(vl)

    log_vl = _num(df, "log_vl_pix", 0.0)

    df["exp005_conta_nova_valor_alto_flag"] = (
        (rel <= 12)
        & (vl >= 5000)
        & (first_receiver == 1)
    ).astype(int)

    df["exp005_interaction_rel_valor"] = log_vl / (rel + 1.0)
    df["exp005_pix_random_x_first_receiver"] = pix_random * first_receiver
    df["exp005_valor_x_first_receiver"] = log_vl * first_receiver
    df["exp005_idade_x_valor_alto"] = idade * (vl >= 5000).astype(int)
    df["exp005_burst_x_valor"] = burst * log_vl
    df["exp005_tx30_x_valor"] = tx30 * log_vl
    df["exp005_first_receiver_x_rel_curto"] = first_receiver * (rel <= 12).astype(int)
    df["exp005_receiver_diversity_x_burst"] = distinct_receivers * burst

    # Aliases úteis caso não existam no dataset.
    if "valor_x_first_recv" not in df.columns:
        df["valor_x_first_recv"] = log_vl * first_receiver

    if "idade_x_first_recv" not in df.columns:
        df["idade_x_first_recv"] = idade * first_receiver

    if "valor_x_burst" not in df.columns:
        df["valor_x_burst"] = log_vl * burst

    if "burst_x_distinct_recv" not in df.columns:
        df["burst_x_distinct_recv"] = burst * distinct_receivers

    if "pix_random_x_first_receiver" not in df.columns:
        df["pix_random_x_first_receiver"] = pix_random * first_receiver

    return df


def build_feature_sets(base_features: list[str], df: pd.DataFrame) -> dict[str, list[str]]:
    existing_base = [f for f in base_features if f in df.columns]

    safe_interactions = [
        "valor_x_first_recv",
        "idade_x_first_recv",
        "valor_x_burst",
        "burst_x_distinct_recv",
        "valor_over_trimestre_avg",
        "pix_random_x_first_receiver",
    ]

    exp005_features = [
        "exp005_conta_nova_valor_alto_flag",
        "exp005_interaction_rel_valor",
        "exp005_pix_random_x_first_receiver",
        "exp005_valor_x_first_receiver",
        "exp005_idade_x_valor_alto",
        "exp005_burst_x_valor",
        "exp005_tx30_x_valor",
        "exp005_first_receiver_x_rel_curto",
        "exp005_receiver_diversity_x_burst",
    ]

    return {
        "baseline_features": sorted(set(existing_base)),
        "safe_interactions": sorted(set(existing_base + [f for f in safe_interactions if f in df.columns])),
        "exp005_interactions": sorted(set(existing_base + [f for f in safe_interactions + exp005_features if f in df.columns])),
    }


# =========================================================
# SPLIT
# =========================================================

def temporal_train_holdout_split(
    df: pd.DataFrame,
    holdout_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dt_col = _detect_datetime_col(df)

    if dt_col is None:
        logger.warning("Nenhuma coluna temporal detectada. Usando ordem atual do dataset.")
        ordered = df.copy()
    else:
        ordered = df.copy()
        ordered["_split_dt"] = pd.to_datetime(ordered[dt_col], errors="coerce")
        ordered = ordered.sort_values("_split_dt", kind="mergesort")
        ordered = ordered.drop(columns=["_split_dt"])

    n = len(ordered)
    split_idx = int(n * (1.0 - holdout_frac))
    split_idx = min(max(split_idx, 1), n - 1)

    train_df = ordered.iloc[:split_idx].copy()
    holdout_df = ordered.iloc[split_idx:].copy()

    if train_df["is_fraud"].sum() == 0 or holdout_df["is_fraud"].sum() == 0:
        logger.warning("Split temporal sem fraude em treino/holdout. Usando fallback estratificado.")

        fraud = df[df["is_fraud"].astype(int) == 1]
        normal = df[df["is_fraud"].astype(int) == 0]

        fraud_holdout = fraud.sample(frac=holdout_frac, random_state=42)
        normal_holdout = normal.sample(frac=holdout_frac, random_state=42)

        holdout_idx = list(set(fraud_holdout.index) | set(normal_holdout.index))
        holdout_df = df.loc[holdout_idx].copy()
        train_df = df.drop(index=holdout_idx).copy()

    return train_df, holdout_df


def stratified_eval_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    fraud = df[df["is_fraud"].astype(int) == 1]
    normal = df[df["is_fraud"].astype(int) == 0]

    n_fraud = len(fraud)
    n_normal = max(n - n_fraud, 0)
    n_normal = min(n_normal, len(normal))

    normal_sample = normal.sample(n=n_normal, random_state=seed)
    out = pd.concat([fraud, normal_sample], axis=0)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    logger.info(
        "Sample model-only (seed=%s): %d tx (%d fraudes + %d normais)",
        seed,
        len(out),
        int(out["is_fraud"].sum()),
        int((out["is_fraud"] == 0).sum()),
    )

    return out


# =========================================================
# METRICAS
# =========================================================

def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (scores >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / max(fp + tn, 1)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": round(float(precision), 6),
        "Recall": round(float(recall), 6),
        "F1": round(float(f1), 6),
        "FPR": round(float(fpr), 8),
    }


def threshold_sweep(
    y_true: np.ndarray,
    scores: np.ndarray,
    precision_min: float,
) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    rows = []

    for i, th in enumerate(thresholds):
        m = metrics_from_scores(y_true, scores, float(th))
        rows.append({
            "threshold": float(th),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(m["F1"]),
            "tp": int(m["TP"]),
            "fp": int(m["FP"]),
            "fn": int(m["FN"]),
            "tn": int(m["TN"]),
            "fpr": float(m["FPR"]),
            "eligible_precision_min": bool(float(precision[i]) >= precision_min),
            "eligible_precision_090": bool(float(precision[i]) >= 0.90),
            "eligible_precision_080": bool(float(precision[i]) >= 0.80),
            "eligible_recall_098": bool(float(recall[i]) >= 0.98),
            "eligible_recall_097": bool(float(recall[i]) >= 0.97),
            "eligible_recall_095": bool(float(recall[i]) >= 0.95),
        })

    return pd.DataFrame(rows)


def pick_operating_points(
    variant_id: str,
    sweep: pd.DataFrame,
    precision_min: float,
) -> list[OperatingPoint]:
    points: list[tuple[str, str, pd.DataFrame, list[str], list[bool]]] = []

    points.append((
        "P1_PRECISION_MIN",
        f"max_recall_at_precision_{precision_min:.2f}",
        sweep[sweep["precision"] >= precision_min].copy(),
        ["recall", "f1", "precision", "threshold"],
        [False, False, False, True],
    ))

    points.append((
        "P2_PRECISION_090",
        "max_recall_at_precision_0.90",
        sweep[sweep["precision"] >= 0.90].copy(),
        ["recall", "f1", "precision", "threshold"],
        [False, False, False, True],
    ))

    points.append((
        "P3_RECALL_098",
        "best_f1_at_recall_ge_0.98",
        sweep[sweep["recall"] >= 0.98].copy(),
        ["f1", "precision", "recall", "threshold"],
        [False, False, False, False],
    ))

    points.append((
        "P4_RECALL_095",
        "best_f1_at_recall_ge_0.95",
        sweep[sweep["recall"] >= 0.95].copy(),
        ["f1", "precision", "recall", "threshold"],
        [False, False, False, False],
    ))

    points.append((
        "P5_BEST_F1",
        "best_f1_global",
        sweep.copy(),
        ["f1", "recall", "precision", "threshold"],
        [False, False, False, False],
    ))

    out: list[OperatingPoint] = []
    seen_thresholds: set[float] = set()

    for policy_id, reason, df, sort_cols, ascending in points:
        if df.empty:
            continue

        chosen = df.sort_values(sort_cols, ascending=ascending).iloc[0]
        th = round(float(chosen["threshold"]), 10)

        # Permitir políticas duplicadas apenas uma vez.
        key = th
        if key in seen_thresholds:
            continue
        seen_thresholds.add(key)

        out.append(OperatingPoint(
            variant_id=variant_id,
            policy_id=policy_id,
            threshold=float(chosen["threshold"]),
            precision=float(chosen["precision"]),
            recall=float(chosen["recall"]),
            f1=float(chosen["f1"]),
            fp=int(chosen["fp"]),
            fn=int(chosen["fn"]),
            tp=int(chosen["tp"]),
            tn=int(chosen["tn"]),
            fpr=float(chosen["fpr"]),
            selected_reason=reason,
        ))

    return out


def select_best_operating_point(points_df: pd.DataFrame) -> pd.Series:
    """
    Seleciona candidato para EXP-005B.

    Regra:
      1. Preferir pontos com Precision >= 0.90 e Recall >= 0.90.
      2. Se nao houver, preferir Precision >= 0.80 e Recall >= 0.95.
      3. Se nao houver, usar maior F1.
    """
    eligible = points_df[
        (points_df["precision"] >= 0.90)
        & (points_df["recall"] >= 0.90)
    ].copy()

    if eligible.empty:
        eligible = points_df[
            (points_df["precision"] >= 0.80)
            & (points_df["recall"] >= 0.95)
        ].copy()

    if eligible.empty:
        eligible = points_df.copy()

    eligible = eligible.sort_values(
        ["recall", "precision", "f1", "fp"],
        ascending=[False, False, False, True],
    )

    return eligible.iloc[0]


# =========================================================
# TREINO
# =========================================================

def train_variant(
    variant_id: str,
    label: str,
    feature_set: str,
    features: list[str],
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    params: dict[str, Any],
    precision_min: float,
    output_dir: Path,
) -> tuple[ModelResult, pd.DataFrame, Any]:
    logger.info("Treinando %s — %s", variant_id, label)

    y_train = _prepare_label(train_df)
    y_holdout = _prepare_label(holdout_df)

    x_train = _ensure_numeric_matrix(train_df, features)
    x_holdout = _ensure_numeric_matrix(holdout_df, features)

    model = lgb.LGBMClassifier(**params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_holdout, y_holdout)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    scores = model.predict_proba(x_holdout)[:, 1]

    try:
        auc = float(roc_auc_score(y_holdout, scores))
    except Exception:
        auc = 0.0

    try:
        ap = float(average_precision_score(y_holdout, scores))
    except Exception:
        ap = 0.0

    sweep = threshold_sweep(y_holdout.values, scores, precision_min=precision_min)
    ops = pick_operating_points(variant_id, sweep, precision_min=precision_min)
    ops_df = pd.DataFrame([asdict(p) for p in ops])

    selected = select_best_operating_point(ops_df)

    model_path = output_dir / f"{variant_id}_model.joblib"
    features_path = output_dir / f"{variant_id}_features.json"

    joblib.dump(model, model_path)
    safe_json_dump({"features": features}, features_path)

    result = ModelResult(
        variant_id=variant_id,
        label=label,
        feature_set=feature_set,
        n_features=len(features),
        train_rows=len(train_df),
        holdout_rows=len(holdout_df),
        holdout_frauds=int(y_holdout.sum()),
        auc=round(auc, 6),
        ap=round(ap, 6),
        best_policy_id=str(selected["policy_id"]),
        threshold_selected=round(float(selected["threshold"]), 10),
        precision_selected=round(float(selected["precision"]), 6),
        recall_selected=round(float(selected["recall"]), 6),
        f1_selected=round(float(selected["f1"]), 6),
        fp_selected=int(selected["fp"]),
        fn_selected=int(selected["fn"]),
        tp_selected=int(selected["tp"]),
        tn_selected=int(selected["tn"]),
        fpr_selected=round(float(selected["fpr"]), 8),
        selected_reason=str(selected["selected_reason"]),
        selected_for_candidate=False,
        uses_new_exp005_features=(feature_set == "exp005_interactions"),
        params=params,
        model_path=str(model_path),
        features_path=str(features_path),
    )

    sweep.insert(0, "variant_id", variant_id)
    sweep.insert(1, "label", label)
    sweep.insert(2, "feature_set", feature_set)

    return result, sweep, model


def train_all(
    df: pd.DataFrame,
    precision_min: float,
    holdout_frac: float,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base_features = _load_lgbm_feature_list()

    df_feat = add_exp005a_features(df)
    feature_sets = build_feature_sets(base_features, df_feat)

    train_df, holdout_df = temporal_train_holdout_split(df_feat, holdout_frac=holdout_frac)

    y_train = _prepare_label(train_df)
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    base_spw = max(n_neg / max(n_pos, 1), 1.0)

    logger.info(
        "Split treino/holdout: train=%d fraud=%d | holdout=%d fraud=%d | scale_pos_weight_base=%.2f",
        len(train_df),
        int(train_df["is_fraud"].sum()),
        len(holdout_df),
        int(holdout_df["is_fraud"].sum()),
        base_spw,
    )

    common = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": 1200,
        "learning_rate": 0.015,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 0.25,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    variants = [
        {
            "variant_id": "LGBM_A_BASELINE_RETRAIN",
            "label": "Baseline retrain com feature set atual",
            "feature_set": "baseline_features",
            "spw": 1.0,
            "params": {**common, "scale_pos_weight": base_spw},
        },
        {
            "variant_id": "LGBM_B_SPW_1_5X",
            "label": "Scale_pos_weight 1.5x",
            "feature_set": "baseline_features",
            "spw": 1.5,
            "params": {**common, "scale_pos_weight": base_spw * 1.5},
        },
        {
            "variant_id": "LGBM_C_SPW_2_0X",
            "label": "Scale_pos_weight 2.0x",
            "feature_set": "baseline_features",
            "spw": 2.0,
            "params": {**common, "scale_pos_weight": base_spw * 2.0},
        },
        {
            "variant_id": "LGBM_D_SAFE_INTERACTIONS",
            "label": "Feature set atual + interacoes seguras existentes",
            "feature_set": "safe_interactions",
            "spw": 1.5,
            "params": {**common, "scale_pos_weight": base_spw * 1.5},
        },
        {
            "variant_id": "LGBM_E_EXP005_INTERACTIONS",
            "label": "Feature set com novas interacoes EXP-005A",
            "feature_set": "exp005_interactions",
            "spw": 1.5,
            "params": {**common, "scale_pos_weight": base_spw * 1.5},
        },
        {
            "variant_id": "LGBM_F_PRECISION_GUARD",
            "label": "Mais regularizado para proteger FP",
            "feature_set": "exp005_interactions",
            "spw": 1.25,
            "params": {
                **common,
                "scale_pos_weight": base_spw * 1.25,
                "num_leaves": 31,
                "min_child_samples": 40,
                "reg_alpha": 0.25,
                "reg_lambda": 1.0,
            },
        },
    ]

    results: list[dict[str, Any]] = []
    sweeps: list[pd.DataFrame] = []
    models: dict[str, Any] = {}
    feature_map: dict[str, list[str]] = {}

    for var in variants:
        features = feature_sets[var["feature_set"]]

        if not features:
            logger.warning("Pulando %s: feature set vazio.", var["variant_id"])
            continue

        result, sweep, model = train_variant(
            variant_id=var["variant_id"],
            label=var["label"],
            feature_set=var["feature_set"],
            features=features,
            train_df=train_df,
            holdout_df=holdout_df,
            params=var["params"],
            precision_min=precision_min,
            output_dir=output_dir,
        )

        row = asdict(result)
        row["scale_pos_weight_multiplier"] = var["spw"]
        results.append(row)
        sweeps.append(sweep)
        models[var["variant_id"]] = model
        feature_map[var["variant_id"]] = features

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise RuntimeError("Nenhuma variante foi treinada.")

    # Seleção final do candidato.
    # O EXP-005A nao promove runtime. Ele escolhe um candidato para EXP-005B.
    eligible = results_df[
        (results_df["precision_selected"] >= 0.90)
        & (results_df["recall_selected"] >= 0.90)
    ].copy()

    if eligible.empty:
        eligible = results_df[
            (results_df["precision_selected"] >= 0.80)
            & (results_df["recall_selected"] >= 0.95)
        ].copy()

    if eligible.empty:
        logger.warning(
            "Nenhum modelo atingiu os criterios operacionais no holdout. "
            "Selecionando por maior F1 como fallback."
        )
        eligible = results_df.copy()
        eligible = eligible.sort_values(
            ["f1_selected", "recall_selected", "precision_selected"],
            ascending=[False, False, False],
        )
    else:
        eligible = eligible.sort_values(
            ["recall_selected", "precision_selected", "f1_selected", "fp_selected"],
            ascending=[False, False, False, True],
        )

    winner_id = str(eligible.iloc[0]["variant_id"])
    results_df.loc[results_df["variant_id"] == winner_id, "selected_for_candidate"] = True

    sweep_df = pd.concat(sweeps, ignore_index=True)

    context = {
        "winner_id": winner_id,
        "models": models,
        "feature_map": feature_map,
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "holdout_frauds": int(holdout_df["is_fraud"].sum()),
        "feature_sets": feature_sets,
        "df_features": df_feat,
    }

    return results_df, sweep_df, context


# =========================================================
# CANDIDATE ARTIFACTS
# =========================================================

def prepare_candidate_artifacts(
    winner_id: str,
    results_df: pd.DataFrame,
    feature_map: dict[str, list[str]],
) -> dict[str, Any]:
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    winner = results_df.loc[results_df["variant_id"] == winner_id].iloc[0].to_dict()
    model_src = Path(winner["model_path"])

    model_dst = CANDIDATE_DIR / "lgbm_v6_2_recall_candidate.joblib"
    features_dst = CANDIDATE_DIR / "lgbm_features_v6_2.json"
    thresholds_dst = CANDIDATE_DIR / "thresholds_lgbm_v6_2.json"
    manifest_dst = CANDIDATE_DIR / "manifest_exp_005a.json"

    joblib.dump(joblib.load(model_src), model_dst)

    features = feature_map[winner_id]
    safe_json_dump({"features": features}, features_dst)

    thresholds = {
        "version": "v6.2-recall-candidate",
        "experiment_id": EXP_ID,
        "candidate_variant_id": winner_id,
        "model_only_threshold": float(winner["threshold_selected"]),
        "policy_id": winner["best_policy_id"],
        "selected_reason": winner["selected_reason"],
        "holdout_metrics": {
            "TP": int(winner["tp_selected"]),
            "FP": int(winner["fp_selected"]),
            "FN": int(winner["fn_selected"]),
            "TN": int(winner["tn_selected"]),
            "Precision": float(winner["precision_selected"]),
            "Recall": float(winner["recall_selected"]),
            "F1": float(winner["f1_selected"]),
            "FPR": float(winner["fpr_selected"]),
            "AUC": float(winner["auc"]),
            "AP": float(winner["ap"]),
        },
        "important_note": (
            "Este threshold e model-only. Nao promover diretamente para runtime. "
            "Usar EXP-005B para calibrar Decision Engine, thresholds e guard rail."
        ),
    }

    safe_json_dump(thresholds, thresholds_dst)

    manifest = {
        "experiment_id": EXP_ID,
        "status": "candidate_generated_model_only",
        "candidate_variant_id": winner_id,
        "candidate_dir": str(CANDIDATE_DIR),
        "model_file": str(model_dst),
        "features_file": str(features_dst),
        "thresholds_file": str(thresholds_dst),
        "uses_new_exp005_features": bool(winner["uses_new_exp005_features"]),
        "baseline_fase2_reference": BASELINE_FASE2,
        "warning": (
            "EXP-005A nao executa PipelineOrquestrador. "
            "E2E completo e calibracao de runtime pertencem ao EXP-005B."
        ),
    }

    safe_json_dump(manifest, manifest_dst)
    return manifest


# =========================================================
# MODEL-ONLY EVALUATION ARTIFACTS
# =========================================================

def evaluate_model_only_sample(
    df_feat: pd.DataFrame,
    winner_id: str,
    results_df: pd.DataFrame,
    context: dict[str, Any],
    sample_n: int,
    seed: int,
) -> dict[str, Any]:
    winner = results_df.loc[results_df["variant_id"] == winner_id].iloc[0].to_dict()
    model = context["models"][winner_id]
    features = context["feature_map"][winner_id]
    threshold = float(winner["threshold_selected"])

    sample = stratified_eval_sample(df_feat, n=sample_n, seed=seed)

    x = _ensure_numeric_matrix(sample, features)
    y = _prepare_label(sample).values
    scores = model.predict_proba(x)[:, 1]
    metrics = metrics_from_scores(y, scores, threshold)

    out = {
        "status": "MODEL_ONLY_NOT_PIPELINE_E2E",
        "reason": (
            "Esta avaliacao usa apenas o LightGBM candidato sobre sample estratificado. "
            "Nao chama PipelineOrquestrador, DecisionEngine nem scoring_config.json."
        ),
        "sample": {
            "n": int(len(sample)),
            "seed": seed,
            "fraudes": int(sample["is_fraud"].sum()),
            "normais": int((sample["is_fraud"] == 0).sum()),
        },
        "candidate": {
            "variant_id": winner_id,
            "threshold": threshold,
            "features": len(features),
        },
        "metrics": metrics,
        "baseline_fase2_reference": BASELINE_FASE2,
        "interpretation": (
            "Use este resultado apenas para selecionar candidato de modelo. "
            "A decisao de promover depende do EXP-005B com Decision Engine real."
        ),
    }

    # Top FNs e FPs model-only.
    pred = scores >= threshold
    fn_mask = (y == 1) & (~pred)
    fp_mask = (y == 0) & pred

    cols = [
        "cd_pix",
        "transaction_id",
        "cd_cpf_pagador",
        "customer_id",
        "vl_pix",
        "nr_idade",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "is_fraud",
    ]
    cols = [c for c in cols if c in sample.columns]

    detail = sample[cols].copy()
    detail["lgbm_score_candidate"] = scores
    detail["pred_candidate"] = pred.astype(int)

    out["top_fn_model_only"] = detail.loc[fn_mask].sort_values(
        "lgbm_score_candidate",
        ascending=False,
    ).head(30).to_dict(orient="records")

    out["top_fp_model_only"] = detail.loc[fp_mask].sort_values(
        "lgbm_score_candidate",
        ascending=False,
    ).head(30).to_dict(orient="records")

    return _safe_json(out)


def validation_report_model_only(
    df_feat: pd.DataFrame,
    winner_id: str,
    results_df: pd.DataFrame,
    context: dict[str, Any],
    sample_n: int,
    validation_seed: int,
) -> dict[str, Any]:
    return evaluate_model_only_sample(
        df_feat=df_feat,
        winner_id=winner_id,
        results_df=results_df,
        context=context,
        sample_n=sample_n,
        seed=validation_seed,
    )


# =========================================================
# REPORTS
# =========================================================

def write_shap_drift_report(
    path: Path,
    winner_id: str,
    results_df: pd.DataFrame,
    context: dict[str, Any],
) -> None:
    winner = results_df.loc[results_df["variant_id"] == winner_id].iloc[0].to_dict()
    model = context["models"][winner_id]
    features = context["feature_map"][winner_id]

    importances = getattr(model, "feature_importances_", None)

    lines = [
        "# EXP-005A — Feature Importance / Drift Report",
        "",
        f"- Vencedor: `{winner_id}`",
        f"- Feature set: `{winner['feature_set']}`",
        f"- Usa novas features EXP-005A: `{winner['uses_new_exp005_features']}`",
        f"- Numero de features: `{len(features)}`",
        "",
        "## Observacao",
        "",
        "Este relatorio usa `feature_importances_` do LightGBM como proxy rapido.",
        "SHAP completo deve ser feito posteriormente no model card, se o candidato avancar no EXP-005B.",
        "",
        "## Top features por importancia",
        "",
    ]

    if importances is None:
        lines.append("Modelo nao expoe `feature_importances_`.")
    else:
        imp = pd.DataFrame({
            "feature": features,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        lines.append("| Feature | Importance |")
        lines.append("|---|---:|")

        for _, row in imp.head(40).iterrows():
            lines.append(f"| `{row['feature']}` | {int(row['importance'])} |")

    lines.extend([
        "",
        "## Risco de deploy",
        "",
        "- Se o vencedor usa features `exp005_*`, elas precisam ser promovidas para `preprocessing.py` e `PipelineOrquestrador` antes de qualquer runtime.",
        "- EXP-005A nao e deployavel sozinho; ele gera candidato para EXP-005B.",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def write_conclusion(
    path: Path,
    winner_id: str,
    results_df: pd.DataFrame,
    model_eval: dict[str, Any],
    validation_eval: dict[str, Any],
) -> None:
    winner = results_df.loc[results_df["variant_id"] == winner_id].iloc[0].to_dict()
    metrics = model_eval.get("metrics", {})
    val_metrics = validation_eval.get("metrics", {})

    passed_model_min = (
        metrics.get("FN", 999) <= 7
        and metrics.get("Precision", 0) >= 0.80
        and metrics.get("Recall", 0) >= 0.95
    )

    status = "CANDIDATO_PARA_EXP_005B" if passed_model_min else "NAO_PROMOVER_DIRETO"

    lines = [
        "# EXP-005A — Conclusao Executiva",
        "",
        f"- Status: `{status}`",
        f"- Vencedor model-only: `{winner_id}`",
        "",
        "## Resultado holdout temporal",
        "",
        f"- Policy: `{winner['best_policy_id']}`",
        f"- Threshold selecionado: `{winner['threshold_selected']}`",
        f"- Precision: `{winner['precision_selected']}`",
        f"- Recall: `{winner['recall_selected']}`",
        f"- F1: `{winner['f1_selected']}`",
        f"- TP: `{winner['tp_selected']}`",
        f"- FP: `{winner['fp_selected']}`",
        f"- FN: `{winner['fn_selected']}`",
        f"- FPR: `{winner['fpr_selected']}`",
        f"- AUC: `{winner['auc']}`",
        f"- AP: `{winner['ap']}`",
        f"- Usa novas features EXP-005A: `{winner['uses_new_exp005_features']}`",
        "",
        "## Avaliacao model-only em sample estratificado",
        "",
        f"- TP: `{metrics.get('TP')}`",
        f"- FP: `{metrics.get('FP')}`",
        f"- FN: `{metrics.get('FN')}`",
        f"- Precision: `{metrics.get('Precision')}`",
        f"- Recall: `{metrics.get('Recall')}`",
        f"- F1: `{metrics.get('F1')}`",
        f"- FPR: `{metrics.get('FPR')}`",
        "",
        "## Validacao model-only",
        "",
        f"- TP: `{val_metrics.get('TP')}`",
        f"- FP: `{val_metrics.get('FP')}`",
        f"- FN: `{val_metrics.get('FN')}`",
        f"- Precision: `{val_metrics.get('Precision')}`",
        f"- Recall: `{val_metrics.get('Recall')}`",
        f"- F1: `{val_metrics.get('F1')}`",
        f"- FPR: `{val_metrics.get('FPR')}`",
        "",
        "## Baseline oficial FASE 2",
        "",
        f"- TP={BASELINE_FASE2['TP']}, FP={BASELINE_FASE2['FP']}, FN={BASELINE_FASE2['FN']}",
        f"- Precision={BASELINE_FASE2['Precision']:.4%}, Recall={BASELINE_FASE2['Recall']:.4%}, F1={BASELINE_FASE2['F1']:.4f}",
        "",
        "## Conclusao",
        "",
    ]

    if passed_model_min:
        lines.extend([
            "O EXP-005A gerou um candidato de LightGBM orientado a recall.",
            "Este resultado ainda nao e suficiente para promover runtime, porque nao passou pelo Decision Engine real.",
            "O proximo passo e o EXP-005B: calibrar thresholds, guard rail e avaliar E2E com artefatos estaveis.",
        ])
    else:
        lines.extend([
            "O EXP-005A nao gerou um candidato claramente promovivel.",
            "Ainda assim, os artefatos sao uteis para diagnosticar a curva precision-recall e orientar o EXP-005B ou uma nova rodada de treino.",
        ])

    lines.extend([
        "",
        "## Nota sobre E2E",
        "",
        "Esta versao do EXP-005A nao chama `PipelineOrquestrador`, `DecisionEngine` nem `scoring_config.json`.",
        "Isso foi feito para isolar o treino LGBM dos erros de runtime encontrados anteriormente.",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=f"{EXP_ID} — {EXP_TITLE}")
    parser.add_argument("--precision-min", type=float, default=0.94)
    parser.add_argument("--holdout-frac", type=float, default=0.20)
    parser.add_argument("--sample", type=int, default=6000)
    parser.add_argument("--validation-seed", type=int, default=123)
    args = parser.parse_args()

    t0 = time.perf_counter()

    print_section(f"{EXP_ID} — {EXP_TITLE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Output dir: %s", OUTPUT_DIR)
    logger.info("Candidate dir: %s", CANDIDATE_DIR)

    print_section("1. Carregar dataset")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    df["is_fraud"] = _prepare_label(df)

    logger.info("Dataset: %d linhas | fraudes=%d", len(df), int(df["is_fraud"].sum()))

    print_section("2. Treinar variantes LGBM recall-oriented")

    results_df, sweep_df, context = train_all(
        df=df,
        precision_min=args.precision_min,
        holdout_frac=args.holdout_frac,
        output_dir=OUTPUT_DIR,
    )

    winner_id = context["winner_id"]
    logger.info("Vencedor model-only: %s", winner_id)

    print_section("3. Salvar tabela de modelos e threshold sweep")

    results_path = OUTPUT_DIR / "01_tabela_modelos.csv"
    sweep_path = OUTPUT_DIR / "02_threshold_sweep_lgbm.csv"

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    sweep_df.to_csv(sweep_path, index=False, encoding="utf-8-sig")

    logger.info("Salvo: %s", results_path)
    logger.info("Salvo: %s", sweep_path)

    print_section("4. Preparar artefatos candidatos")

    manifest = prepare_candidate_artifacts(
        winner_id=winner_id,
        results_df=results_df,
        feature_map=context["feature_map"],
    )

    safe_json_dump(manifest, OUTPUT_DIR / "manifest_artefatos_candidatos.json")

    print_section("5. Avaliacao model-only do candidato")

    model_eval = evaluate_model_only_sample(
        df_feat=context["df_features"],
        winner_id=winner_id,
        results_df=results_df,
        context=context,
        sample_n=args.sample,
        seed=42,
    )

    safe_json_dump(model_eval, OUTPUT_DIR / "03_avaliacao_e2e_modelo_candidato.json")

    validation_eval = validation_report_model_only(
        df_feat=context["df_features"],
        winner_id=winner_id,
        results_df=results_df,
        context=context,
        sample_n=args.sample,
        validation_seed=args.validation_seed,
    )

    safe_json_dump(validation_eval, OUTPUT_DIR / "04_validacao_cruzada.json")

    print_section("6. Relatorios")

    write_shap_drift_report(
        path=OUTPUT_DIR / "05_shap_drift_report.md",
        winner_id=winner_id,
        results_df=results_df,
        context=context,
    )

    write_conclusion(
        path=OUTPUT_DIR / "06_conclusao_executiva.md",
        winner_id=winner_id,
        results_df=results_df,
        model_eval=model_eval,
        validation_eval=validation_eval,
    )

    winner = results_df.loc[results_df["variant_id"] == winner_id].iloc[0]

    logger.info("============================================================")
    logger.info("EXP-005A concluido SEM dependencias E2E")
    logger.info("Vencedor: %s", winner_id)
    logger.info(
        "Holdout: Precision=%.4f Recall=%.4f F1=%.4f FP=%d FN=%d threshold=%.6f",
        float(winner["precision_selected"]),
        float(winner["recall_selected"]),
        float(winner["f1_selected"]),
        int(winner["fp_selected"]),
        int(winner["fn_selected"]),
        float(winner["threshold_selected"]),
    )
    logger.info("Artefatos em: %s", OUTPUT_DIR)
    logger.info("Artefatos candidatos em: %s", CANDIDATE_DIR)
    logger.info("Tempo total: %.1fs", time.perf_counter() - t0)
    logger.info("============================================================")


if __name__ == "__main__":
    main()