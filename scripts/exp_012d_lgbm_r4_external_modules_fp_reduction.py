#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-012D — LGBM R4 + External Modules FP Reduction

Objetivo:
  Usar o baseline LGBM high-recall do EXP-012C-R4 como primeiro estágio e
  colocar IF / Behavioral Analytics / Social Engineering em jogo para reduzir
  falsos positivos, sem sacrificar o recall mínimo.

Contexto:
  EXP-012C-R4 consolidado:
    champion_candidate_id = S1_SEG_MBK_AVAILABLE_FLAG_DS_TIPO_CHAVE_NORM
    model_id = HR01_pos8_base
    policy = segmented_thresholds_val_recall_target
    HOLDOUT_LABEL_SAFE: TP=122, FP=1604, FN=2, recall=98.39%

Meta deste experimento:
  Minimizar FP na VALIDATION sujeito a recall >= target_recall.
  Confirmar no HOLDOUT_LABEL_SAFE.

Estratégia:
  1. Recalcula scores do LGBM R4 sobre o dataset v3 completo.
  2. Aplica a política segmentada campeã do R4.
  3. Para as transações positivas pelo LGBM R4, calcula:
     - SocialEngineeringDetector;
     - BehavioralAnalytics;
     - Isolation Forest percentile, quando artefatos do DecisionEngine estiverem disponíveis.
  4. Varre políticas de redução de FP:
     - veto de casos "quietos" com score baixo;
     - keep por evidência externa;
     - score de pontos LGBM + SE + BEH + IF.
  5. Escolhe o campeão por VALIDATION:
     recall >= target_recall, menor FP/FPR, maior precision/F1.
  6. Reporta HOLDOUT_LABEL_SAFE e HOLDOUT_FULL.

Uso:
  python scripts\\exp_012d_lgbm_r4_external_modules_fp_reduction.py

Smoke test:
  python scripts\\exp_012d_lgbm_r4_external_modules_fp_reduction.py --fast

Saídas:
  resultados/experimentos/EXP-012D/
    00_run_summary.json
    01_policy_comparison.csv
    02_champion_metrics_by_split.csv
    03_module_signal_coverage.csv
    04_module_scores_lgbm_r4_positives.csv
    05_champion_predictions_holdout_label_safe.csv
    06_champion_false_negatives_holdout_label_safe.csv
    07_champion_false_positives_holdout_label_safe.csv
    08_recommendation.md
    09_module_availability.json
    10_policy_search_space.json

Artefatos:
  backend/artefatos_candidatos/exp012d_r4_external_modules/
    policy_exp012d_external_modules.json
    manifest_exp012d_external_modules.json

Observação:
  Este script NÃO promove modelo nem altera artefatos produtivos.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
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
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

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
R4_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-012C-R4"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-012D"

R4_CANDIDATE_DIR = PROJECT_ROOT / "backend" / "artefatos_candidatos" / "exp012c_r4_lgbm_fp_squeeze"
CANDIDATE_DIR = PROJECT_ROOT / "backend" / "artefatos_candidatos" / "exp012d_r4_external_modules"
ARTEFATOS_DIR = PROJECT_ROOT / "backend" / "artefatos"

DEFAULT_INPUT = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | EXP-012D | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EXP-012D")


# =============================================================================
# Helpers
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


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().split(".")[-1] for c in df.columns]

    if "event_datetime" not in df.columns and "dt_pix" in df.columns:
        df["event_datetime"] = df["dt_pix"]
    if "transaction_id" not in df.columns and "cd_pix" in df.columns:
        df["transaction_id"] = df["cd_pix"]

    df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
    df["data_pix"] = pd.to_datetime(df["data_pix"] if "data_pix" in df.columns else df["event_datetime"], errors="coerce")
    df = df[df["event_datetime"].notna() & df["data_pix"].notna()].copy()

    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)
    df["temporal_split"] = df["temporal_split"].astype(str).str.upper().str.strip()
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


def eval_binary(y_true, y_pred, y_prob=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
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


def add_metrics_prefix(row: dict[str, Any], prefix: str, y, pred, score=None) -> None:
    metrics = eval_binary(y, pred, score)
    for k, v in metrics.items():
        row[f"{prefix}_{k}"] = v


def policy_sort_key(row: pd.Series, target_recall: float):
    pass_recall = int(row["val_recall"] >= target_recall)
    return (
        pass_recall,
        -float(row["val_fp"]),
        -float(row["val_fpr"]),
        float(row["val_precision"]),
        float(row["val_f1"]),
        float(row["safe_recall"]),
        -float(row["safe_fp"]),
    )


def make_candidate_row(candidate_id, family, policy_desc, y_val, pred_val, score_val, y_safe, pred_safe, score_safe, y_full, pred_full, score_full, extra=None):
    row = {
        "candidate_id": candidate_id,
        "family": family,
        "policy_desc": policy_desc,
    }
    add_metrics_prefix(row, "val", y_val, pred_val, score_val)
    add_metrics_prefix(row, "safe", y_safe, pred_safe, score_safe)
    add_metrics_prefix(row, "full", y_full, pred_full, score_full)
    if extra:
        row.update(extra)
    return row


# =============================================================================
# Module imports
# =============================================================================
def setup_import_paths() -> None:
    candidates = [
        PROJECT_ROOT,
        PROJECT_ROOT / "backend",
        PROJECT_ROOT / "backend" / "core",
        PROJECT_ROOT / "backend" / "app",
        PROJECT_ROOT / "backend" / "app" / "services",
    ]
    for p in candidates:
        if p.exists():
            sys.path.insert(0, str(p))


def import_optional_modules() -> dict[str, Any]:
    setup_import_paths()
    availability: dict[str, Any] = {
        "social_engineering": {"available": False, "error": None},
        "behavioral_analytics": {"available": False, "error": None},
        "decision_engine_if": {"available": False, "error": None},
    }

    modules: dict[str, Any] = {}

    # Social Engineering.
    for mod_name in ["core.social_engineering", "social_engineering"]:
        try:
            mod = importlib.import_module(mod_name)
            modules["SocialEngineeringDetector"] = getattr(mod, "SocialEngineeringDetector")
            availability["social_engineering"] = {"available": True, "module": mod_name, "version": getattr(modules["SocialEngineeringDetector"], "VERSION", None), "error": None}
            break
        except Exception as exc:
            availability["social_engineering"]["error"] = str(exc)

    # Behavioral Analytics.
    for mod_name in ["core.behavioral_analytics", "behavioral_analytics"]:
        try:
            mod = importlib.import_module(mod_name)
            modules["BehavioralAnalytics"] = getattr(mod, "BehavioralAnalytics")
            availability["behavioral_analytics"] = {"available": True, "module": mod_name, "version": getattr(modules["BehavioralAnalytics"], "VERSION", None), "error": None}
            break
        except Exception as exc:
            availability["behavioral_analytics"]["error"] = str(exc)

    # DecisionEngine for IF percentile only.
    for mod_name in ["core.decision_engine", "decision_engine"]:
        try:
            mod = importlib.import_module(mod_name)
            modules["PixDecisionEngine"] = getattr(mod, "PixDecisionEngine")
            modules["EngineConfig"] = getattr(mod, "EngineConfig")
            availability["decision_engine_if"] = {"available": True, "module": mod_name, "error": None}
            break
        except Exception as exc:
            availability["decision_engine_if"]["error"] = str(exc)

    return {"modules": modules, "availability": availability}


# =============================================================================
# R4 scoring
# =============================================================================
def load_feature_schema(candidate_dir: Path) -> dict[str, Any]:
    paths = [
        candidate_dir / "features_lgbm_v3_r4_fp_squeeze_shadow.json",
        candidate_dir / "features_lgbm_v3_r4_tuned_shadow.json",
    ]
    paths.extend(sorted(candidate_dir.glob("features*.json")))
    for p in paths:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Feature schema R4 não encontrado em {candidate_dir}")


def load_r4_artifacts(candidate_dir: Path):
    model_paths = [
        candidate_dir / "model_lgbm_v3_r4_fp_squeeze_shadow.joblib",
        candidate_dir / "stage1_model_lgbm_v3_r4.joblib",
    ]
    preprocessor_paths = [
        candidate_dir / "preprocessor_lgbm_v3_r4_fp_squeeze_shadow.joblib",
        candidate_dir / "stage1_preprocessor_lgbm_v3_r4.joblib",
    ]

    model_path = next((p for p in model_paths if p.exists()), None)
    preprocessor_path = next((p for p in preprocessor_paths if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(f"Modelo R4 não encontrado em {candidate_dir}")
    if preprocessor_path is None:
        raise FileNotFoundError(f"Preprocessor R4 não encontrado em {candidate_dir}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor, model_path, preprocessor_path


def score_r4_lgbm(df: pd.DataFrame, feature_cols: list[str], model, preprocessor) -> np.ndarray:
    work = df.copy()
    for c in feature_cols:
        if c not in work.columns:
            work[c] = np.nan
    X = work[feature_cols]
    return model.predict_proba(preprocessor.transform(X))[:, 1]


def load_r4_segmented_rules(r4_dir: Path, policy: dict[str, Any]) -> tuple[pd.DataFrame | None, list[str], float | None]:
    seg_cols_raw = policy.get("segment_cols")
    if seg_cols_raw is None or pd.isna(seg_cols_raw):
        return None, [], None

    seg_cols = str(seg_cols_raw).split("|")
    candidate_id = str(policy.get("candidate_id") or "S1_SEG_MBK_AVAILABLE_FLAG_DS_TIPO_CHAVE_NORM")
    paths = [
        r4_dir / f"segmented_rules_{candidate_id}.csv",
        r4_dir / "segmented_rules_S1_SEG_MBK_AVAILABLE_FLAG_DS_TIPO_CHAVE_NORM.csv",
    ]

    for p in paths:
        if p.exists():
            rules = pd.read_csv(p)
            global_threshold = safe_float(policy.get("global_threshold"), safe_float(rules["threshold"].min(), 0.001))
            return rules, seg_cols, global_threshold

    raise FileNotFoundError(
        "Regras segmentadas do R4 não encontradas. Esperado arquivo como "
        "resultados/experimentos/EXP-012C-R4/segmented_rules_S1_SEG_MBK_AVAILABLE_FLAG_DS_TIPO_CHAVE_NORM.csv"
    )


def apply_r4_policy(df: pd.DataFrame, scores: np.ndarray, r4_dir: Path, candidate_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    policy_path = candidate_dir / "threshold_policy_exp012c_r4_fp_squeeze.json"
    if not policy_path.exists():
        # Fallback para threshold R3 se não houver política R4, mas avisa no JSON.
        pred = (scores >= 0.001).astype(int)
        return pred, {"policy_source": "fallback_threshold_0.001", "threshold": 0.001}

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["policy_source"] = str(policy_path)

    # Segmented champion.
    if "segmented" in str(policy.get("idea", "")).lower() or policy.get("segment_cols"):
        rules, seg_cols, global_threshold = load_r4_segmented_rules(r4_dir, policy)

        tmp = df[seg_cols].copy()
        for c in seg_cols:
            tmp[c] = tmp[c].astype("string").fillna("<NA>").astype(str)

        rr = rules.copy()
        for c in seg_cols:
            rr[c] = rr[c].astype("string").fillna("<NA>").astype(str)

        merged = tmp.merge(rr[seg_cols + ["threshold"]], on=seg_cols, how="left")
        thresholds = merged["threshold"].fillna(global_threshold).astype(float).values
        pred = (scores >= thresholds).astype(int)

        policy["segment_cols"] = "|".join(seg_cols)
        policy["global_threshold_used"] = global_threshold
        policy["n_segment_rules"] = int(len(rules))
        return pred, policy

    # Scalar threshold fallback.
    threshold = safe_float(policy.get("threshold"), 0.001)
    pred = (scores >= threshold).astype(int)
    policy["threshold_used"] = threshold
    return pred, policy


# =============================================================================
# Module feature adapter and scoring
# =============================================================================
def add_alias_if_missing(features: dict[str, Any], target: str, candidates: list[str], default: Any = 0) -> None:
    if target in features and features[target] is not None and not (isinstance(features[target], float) and np.isnan(features[target])):
        return
    for c in candidates:
        val = features.get(c)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            features[target] = val
            return
    features[target] = default


def prepare_module_features(row: pd.Series) -> dict[str, Any]:
    f = row.to_dict()

    # Core aliases from v3 feature table to old module vocabulary.
    add_alias_if_missing(f, "cd_pix", ["transaction_id"], "")
    add_alias_if_missing(f, "dt_pix", ["event_datetime"], None)
    add_alias_if_missing(f, "cd_cpf_pagador", ["customer_id"], "")
    add_alias_if_missing(f, "cd_cpf_cnpj_recebedor", ["counterparty_id"], "")
    add_alias_if_missing(f, "first_receiver_flag", ["first_receiver_flag_real", "primeiro_envio_para_recebedor_180d"], 0)
    add_alias_if_missing(f, "qt_envio_recebedor_trimestre", ["qtd_pix_mesmo_recebedor_90d", "qtd_pix_mesmo_recebedor_180d"], 0)

    # Rolling / legacy aliases.
    add_alias_if_missing(f, "qt_total_pix_trimestre", ["qtd_pix_pagador_90d", "qtd_pix_pagador_180d"], 0)
    add_alias_if_missing(f, "qt_pix_dia_maximo_trimestre", ["max_qtd_pix_dia_pagador_90d", "max_qtd_pix_dia_pagador_30d"], 0)
    add_alias_if_missing(f, "ratio_valor_mediana", ["ratio_valor_media_pagador_90d", "ratio_valor_maximo_pagador_180d"], None)

    ratio = safe_float(f.get("ratio_valor_mediana"), 0.0)
    vl_pix = safe_float(f.get("vl_pix"), 0.0)
    if "vl_mediana_pix_trimestre" not in f or f.get("vl_mediana_pix_trimestre") is None:
        f["vl_mediana_pix_trimestre"] = vl_pix / ratio if ratio > 0 else 0.0

    # These features are not available in v3 daily aggregates unless explicitly exported.
    # Keep conservative defaults rather than inventing intraday velocity.
    add_alias_if_missing(f, "qt_intervalo_transacao_minuto", ["minutes_since_prev_tx"], None)
    add_alias_if_missing(f, "tx_count_prev_30m", ["tx_count_prev_30m"], 0)
    add_alias_if_missing(f, "burst_30m_flag", ["burst_30m_flag"], 0)
    add_alias_if_missing(f, "distinct_receivers_so_far", ["distinct_receivers_so_far"], 1)

    # Topaz aliases.
    add_alias_if_missing(f, "topaz_risk_score", ["topaz_score_filled"], 0)
    add_alias_if_missing(f, "topaz_score_filled", ["topaz_risk_score"], 0)
    add_alias_if_missing(f, "topaz_transacao_rejeitada", ["topaz_rejeitada_flag"], 0)
    add_alias_if_missing(f, "topaz_rejeitada_flag", ["topaz_transacao_rejeitada"], 0)

    # Key flags from ds_tipo_chave_norm.
    ds_tipo_norm = str(f.get("ds_tipo_chave_norm") or f.get("ds_tipo_chave") or "").upper()
    if "ds_tipo_chave" not in f or f.get("ds_tipo_chave") is None:
        if "CHAVE_ALEATORIA" in ds_tipo_norm or "ALEATORIA" in ds_tipo_norm:
            f["ds_tipo_chave"] = "CHAVE ALEATORIA"
        elif "EMAIL" in ds_tipo_norm:
            f["ds_tipo_chave"] = "EMAIL"
        elif "DOCUMENTO" in ds_tipo_norm or "TELEFONE" in ds_tipo_norm:
            f["ds_tipo_chave"] = "DOCUMENTO/TELEFONE"
        else:
            f["ds_tipo_chave"] = "OUTROS"

    f["pix_key_random_flag"] = int("ALEATORIA" in str(f.get("ds_tipo_chave", "")).upper())
    f["pix_key_email_flag"] = int("EMAIL" in str(f.get("ds_tipo_chave", "")).upper())
    f["pix_key_document_flag"] = int("DOCUMENTO" in str(f.get("ds_tipo_chave", "")).upper() or "TELEFONE" in str(f.get("ds_tipo_chave", "")).upper())

    # Defaults expected by modules.
    for col, default in {
        "nr_idade": 0,
        "qt_tempo_relacionamento_mes": 999,
        "vl_renda_cliente": 0,
        "ratio_pix_renda": None,
        "pix_over_50pct_renda_flag": 0,
        "pix_over_100pct_renda_flag": 0,
        "renda_missing_flag": 1,
        "is_sexo_feminino_flag": 0,
        "is_viuvo_flag": 0,
        "is_segmento_premium_flag": 0,
        "perfil_vulneravel_se_flag": 0,
        "qt_dependentes": 0,
        "is_login_senha_flag": 0,
        "is_agendamento_recorrente_flag": 0,
        "day_of_week": 0,
        "is_business_hours": 1,
        "hour": safe_int(f.get("hour"), 12),
    }.items():
        if col not in f or f[col] is None or (isinstance(f[col], float) and np.isnan(f[col])):
            f[col] = default

    return f


def instantiate_if_engine(modules: dict[str, Any], availability: dict[str, Any], artefatos_dir: Path):
    if not availability.get("decision_engine_if", {}).get("available"):
        return None, availability

    try:
        EngineConfig = modules["EngineConfig"]
        PixDecisionEngine = modules["PixDecisionEngine"]
        engine = PixDecisionEngine(EngineConfig(artefatos_dir=str(artefatos_dir)))
        if getattr(engine, "if_model", None) is None:
            availability["decision_engine_if"]["available"] = False
            availability["decision_engine_if"]["error"] = "DecisionEngine carregou, mas IF não está disponível."
            return None, availability
        availability["decision_engine_if"]["available"] = True
        availability["decision_engine_if"]["engine_version"] = getattr(engine, "ENGINE_VERSION", None)
        return engine, availability
    except Exception as exc:
        availability["decision_engine_if"]["available"] = False
        availability["decision_engine_if"]["error"] = str(exc)
        return None, availability


def score_external_modules(df: pd.DataFrame, positive_mask: np.ndarray, modules: dict[str, Any], availability: dict[str, Any], artefatos_dir: Path, fast: bool = False):
    out_cols = {
        "if_percentile": np.zeros(len(df), dtype=float),
        "if_raw": np.zeros(len(df), dtype=float),
        "if_active": np.zeros(len(df), dtype=int),
        "se_score": np.zeros(len(df), dtype=float),
        "se_pattern_count": np.zeros(len(df), dtype=float),
        "se_has_critico": np.zeros(len(df), dtype=float),
        "se_max_pattern_score": np.zeros(len(df), dtype=float),
        "behavioral_score": np.zeros(len(df), dtype=float),
        "behavioral_risk_factor_count": np.zeros(len(df), dtype=float),
        "behavioral_has_velocity_factor": np.zeros(len(df), dtype=float),
        "behavioral_has_dormancy_factor": np.zeros(len(df), dtype=float),
        "behavioral_has_age_value_factor": np.zeros(len(df), dtype=float),
        "behavioral_max_precision": np.zeros(len(df), dtype=float),
    }

    detail_rows = []

    # Instantiate modules.
    se_detector = None
    behavioral = None

    if availability.get("social_engineering", {}).get("available"):
        try:
            se_detector = modules["SocialEngineeringDetector"]()
        except Exception as exc:
            availability["social_engineering"]["available"] = False
            availability["social_engineering"]["error"] = f"Falha ao instanciar: {exc}"

    if availability.get("behavioral_analytics", {}).get("available"):
        try:
            behavioral = modules["BehavioralAnalytics"]()
        except Exception as exc:
            availability["behavioral_analytics"]["available"] = False
            availability["behavioral_analytics"]["error"] = f"Falha ao instanciar: {exc}"

    if_engine, availability = instantiate_if_engine(modules, availability, artefatos_dir)

    idxs = np.where(positive_mask)[0]
    if fast:
        # Smoke test: calcula módulos só nos primeiros positivos por split.
        selected = []
        for split in ["TRAIN", "VALIDATION", "HOLDOUT"]:
            split_idxs = [i for i in idxs if str(df.iloc[i]["temporal_split"]).upper() == split]
            selected.extend(split_idxs[:2000])
        idxs = np.array(sorted(set(selected)), dtype=int)

    log.info("Calculando módulos externos para %d transações positivas pelo LGBM R4...", len(idxs))

    # Sort chronologically for stateful Behavioral manager.
    idxs_sorted = sorted(idxs, key=lambda i: df.iloc[i]["event_datetime"])

    for n, idx in enumerate(idxs_sorted, 1):
        row = df.iloc[idx]
        features = prepare_module_features(row)

        # IF first because SE adapter can use if_percentile.
        if if_engine is not None:
            try:
                if_score, if_raw, if_active = if_engine._score_if(features)
                out_cols["if_percentile"][idx] = float(if_score)
                out_cols["if_raw"][idx] = float(if_raw)
                out_cols["if_active"][idx] = int(bool(if_active))
                features["if_percentile"] = float(if_score)
            except Exception:
                features["if_percentile"] = 0.0
        else:
            features["if_percentile"] = 0.0

        se_patterns = []
        if se_detector is not None:
            try:
                se_result = se_detector.detect_from_pipeline(features)
                sf = se_result.to_features()
                sd = se_result.to_dict()
                out_cols["se_score"][idx] = safe_float(sf.get("se_score"), 0.0)
                out_cols["se_pattern_count"][idx] = safe_float(sf.get("se_pattern_count"), 0.0)
                out_cols["se_has_critico"][idx] = safe_float(sf.get("se_has_critico"), 0.0)
                out_cols["se_max_pattern_score"][idx] = safe_float(sf.get("se_max_pattern_score"), 0.0)
                se_patterns = [p.get("pattern_name", "") for p in sd.get("patterns", [])]
            except Exception:
                se_patterns = []

        beh_factors = []
        if behavioral is not None:
            try:
                beh_result = behavioral.analyze(features)
                bf = beh_result.to_features()
                bd = beh_result.to_dict()
                out_cols["behavioral_score"][idx] = safe_float(bf.get("behavioral_score"), 0.0)
                out_cols["behavioral_risk_factor_count"][idx] = safe_float(bf.get("behavioral_risk_factor_count"), 0.0)
                out_cols["behavioral_has_velocity_factor"][idx] = safe_float(bf.get("behavioral_has_velocity_factor"), 0.0)
                out_cols["behavioral_has_dormancy_factor"][idx] = safe_float(bf.get("behavioral_has_dormancy_factor"), 0.0)
                out_cols["behavioral_has_age_value_factor"][idx] = safe_float(bf.get("behavioral_has_age_value_factor"), 0.0)
                out_cols["behavioral_max_precision"][idx] = safe_float(bf.get("behavioral_max_precision"), 0.0)
                beh_factors = [p.get("codigo", "") for p in bd.get("risk_factors", [])]
            except Exception:
                beh_factors = []

        detail_rows.append({
            "row_index": int(idx),
            "transaction_id": row.get("transaction_id"),
            "temporal_split": row.get("temporal_split"),
            "is_fraud": int(row.get("is_fraud", 0)),
            "lgbm_r4_score": safe_float(row.get("lgbm_r4_score"), 0.0),
            "if_percentile": out_cols["if_percentile"][idx],
            "se_score": out_cols["se_score"][idx],
            "se_pattern_count": out_cols["se_pattern_count"][idx],
            "se_patterns": "|".join(se_patterns),
            "behavioral_score": out_cols["behavioral_score"][idx],
            "behavioral_risk_factor_count": out_cols["behavioral_risk_factor_count"][idx],
            "behavioral_factors": "|".join(beh_factors),
        })

        if n % 1000 == 0:
            log.info("  módulos externos: %d/%d", n, len(idxs_sorted))

    for c, arr in out_cols.items():
        df[c] = arr

    df["external_any_signal"] = (
        (df["if_percentile"] >= 0.99)
        | (df["se_score"] >= 20)
        | (df["behavioral_score"] >= 15)
        | (df["se_pattern_count"] >= 1)
        | (df["behavioral_risk_factor_count"] >= 1)
    ).astype(int)

    return df, pd.DataFrame(detail_rows), availability


# =============================================================================
# Policy search
# =============================================================================
def get_split_arrays(df: pd.DataFrame, split_name: str):
    if split_name == "HOLDOUT_LABEL_SAFE":
        holdout = df[df["temporal_split"] == "HOLDOUT"].copy()
        max_fraud_dt = holdout.loc[holdout["is_fraud"] == 1, "data_pix"].max()
        part = holdout[holdout["data_pix"] <= max_fraud_dt].copy()
    elif split_name == "HOLDOUT_FULL":
        part = df[df["temporal_split"] == "HOLDOUT"].copy()
    else:
        part = df[df["temporal_split"] == split_name].copy()
    return part.index.values, part


def build_score_caps(valid_df: pd.DataFrame) -> list[float]:
    pos = valid_df[valid_df["lgbm_r4_pred"] == 1]["lgbm_r4_score"].astype(float)
    values = [0.001, 0.002, 0.005, 0.010, 0.020, 0.030, 0.050, 0.075, 0.100, 0.150, 0.200, 0.300, 0.500]
    if len(pos):
        for q in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
            values.append(float(pos.quantile(q)))
    return sorted(set(round(v, 8) for v in values if 0 <= v <= 1))


def policy_quiet_veto(df, score_cap, se_low, beh_low, if_low):
    quiet = (
        (df["se_score"].astype(float) <= se_low)
        & (df["behavioral_score"].astype(float) <= beh_low)
        & (df["if_percentile"].astype(float) < if_low)
    )
    veto = quiet & (df["lgbm_r4_score"].astype(float) < score_cap)
    return (df["lgbm_r4_pred"].astype(int).values == 1) & (~veto.values)


def policy_evidence_keep(df, score_high, se_thr, beh_thr, if_thr):
    keep = (
        (df["lgbm_r4_score"].astype(float) >= score_high)
        | (df["se_score"].astype(float) >= se_thr)
        | (df["behavioral_score"].astype(float) >= beh_thr)
        | (df["if_percentile"].astype(float) >= if_thr)
    )
    return (df["lgbm_r4_pred"].astype(int).values == 1) & keep.values


def policy_points(df, score_thr, min_points):
    points = np.zeros(len(df), dtype=float)

    points += np.where(df["lgbm_r4_score"].astype(float).values >= score_thr, 2.0, 0.0)
    points += np.where(df["se_score"].astype(float).values >= 40, 2.0, 0.0)
    points += np.where(df["se_score"].astype(float).values >= 20, 1.0, 0.0)
    points += np.where(df["behavioral_score"].astype(float).values >= 25, 2.0, 0.0)
    points += np.where(df["behavioral_score"].astype(float).values >= 15, 1.0, 0.0)
    points += np.where(df["if_percentile"].astype(float).values >= 0.995, 2.0, 0.0)
    points += np.where(df["if_percentile"].astype(float).values >= 0.990, 1.0, 0.0)
    points += np.where(df["se_has_critico"].astype(float).values >= 1, 1.0, 0.0)
    points += np.where(df["behavioral_max_precision"].astype(float).values >= 0.35, 1.0, 0.0)

    return (df["lgbm_r4_pred"].astype(int).values == 1) & (points >= min_points)


def run_policy_search(df: pd.DataFrame, target_recall: float, fast: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    idx_val, valid = get_split_arrays(df, "VALIDATION")
    idx_safe, safe = get_split_arrays(df, "HOLDOUT_LABEL_SAFE")
    idx_full, full = get_split_arrays(df, "HOLDOUT_FULL")

    y_val = valid["is_fraud"].astype(int).values
    y_safe = safe["is_fraud"].astype(int).values
    y_full = full["is_fraud"].astype(int).values

    p_val = valid["lgbm_r4_score"].astype(float).values
    p_safe = safe["lgbm_r4_score"].astype(float).values
    p_full = full["lgbm_r4_score"].astype(float).values

    rows = []

    # Baseline R4.
    pred_val = valid["lgbm_r4_pred"].astype(int).values
    pred_safe = safe["lgbm_r4_pred"].astype(int).values
    pred_full = full["lgbm_r4_pred"].astype(int).values
    rows.append(make_candidate_row(
        "BASELINE_R4_SEGMENTED",
        "baseline",
        "LGBM R4 segmented policy sem filtros externos",
        y_val, pred_val, p_val,
        y_safe, pred_safe, p_safe,
        y_full, pred_full, p_full,
        extra={},
    ))

    score_caps = build_score_caps(valid)
    if fast:
        score_caps = score_caps[:8]

    se_lows = [0, 10, 20, 40]
    beh_lows = [0, 10, 15, 20, 25, 40]
    if_lows = [0.90, 0.95, 0.99, 0.995, 1.01]

    # Quiet veto: remove casos baixos e sem evidência externa.
    for score_cap in score_caps:
        for se_low in se_lows:
            for beh_low in beh_lows:
                for if_low in if_lows:
                    pred_val = policy_quiet_veto(valid, score_cap, se_low, beh_low, if_low).astype(int)
                    # filtro rápido por recall validation antes de avaliar todos
                    if eval_binary(y_val, pred_val)["recall"] < target_recall:
                        continue
                    pred_safe = policy_quiet_veto(safe, score_cap, se_low, beh_low, if_low).astype(int)
                    pred_full = policy_quiet_veto(full, score_cap, se_low, beh_low, if_low).astype(int)
                    rows.append(make_candidate_row(
                        f"QUIET_VETO_score{score_cap}_se{se_low}_beh{beh_low}_if{if_low}",
                        "quiet_veto",
                        "Remove positivos LGBM com score baixo e módulos externos quietos",
                        y_val, pred_val, p_val,
                        y_safe, pred_safe, p_safe,
                        y_full, pred_full, p_full,
                        extra={"score_cap": score_cap, "se_low": se_low, "beh_low": beh_low, "if_low": if_low},
                    ))

    # Evidence keep: mantém score alto ou evidência externa.
    score_highs = score_caps + [0.40, 0.50, 0.60, 0.70, 0.80]
    score_highs = sorted(set(round(v, 8) for v in score_highs if 0 <= v <= 1))
    if fast:
        score_highs = score_highs[:8]

    se_thrs = [0, 20, 40, 60]
    beh_thrs = [0, 15, 20, 25, 40, 60]
    if_thrs = [0.90, 0.95, 0.99, 0.995, 0.999, 1.01]

    for score_high in score_highs:
        for se_thr in se_thrs:
            for beh_thr in beh_thrs:
                for if_thr in if_thrs:
                    pred_val = policy_evidence_keep(valid, score_high, se_thr, beh_thr, if_thr).astype(int)
                    if eval_binary(y_val, pred_val)["recall"] < target_recall:
                        continue
                    pred_safe = policy_evidence_keep(safe, score_high, se_thr, beh_thr, if_thr).astype(int)
                    pred_full = policy_evidence_keep(full, score_high, se_thr, beh_thr, if_thr).astype(int)
                    rows.append(make_candidate_row(
                        f"EVIDENCE_KEEP_score{score_high}_se{se_thr}_beh{beh_thr}_if{if_thr}",
                        "evidence_keep",
                        "Mantém positivos LGBM somente com score alto ou evidência IF/SE/BEH",
                        y_val, pred_val, p_val,
                        y_safe, pred_safe, p_safe,
                        y_full, pred_full, p_full,
                        extra={"score_high": score_high, "se_thr": se_thr, "beh_thr": beh_thr, "if_thr": if_thr},
                    ))

    # Points policy.
    min_points_values = [1, 2, 3, 4, 5]
    for score_thr in score_highs:
        for min_points in min_points_values:
            pred_val = policy_points(valid, score_thr, min_points).astype(int)
            if eval_binary(y_val, pred_val)["recall"] < target_recall:
                continue
            pred_safe = policy_points(safe, score_thr, min_points).astype(int)
            pred_full = policy_points(full, score_thr, min_points).astype(int)
            rows.append(make_candidate_row(
                f"POINTS_score{score_thr}_min{min_points}",
                "points",
                "Pontuação combinada LGBM + IF + SE + BEH",
                y_val, pred_val, p_val,
                y_safe, pred_safe, p_safe,
                y_full, pred_full, p_full,
                extra={"score_thr": score_thr, "min_points": min_points},
            ))

    comp = pd.DataFrame(rows)
    comp["_sort"] = comp.apply(lambda r: policy_sort_key(r, target_recall), axis=1)
    comp = comp.sort_values("_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)

    search_space = {
        "target_recall": target_recall,
        "families": ["baseline", "quiet_veto", "evidence_keep", "points"],
        "score_caps_count": len(score_caps),
        "score_highs_count": len(score_highs),
        "se_lows": se_lows,
        "beh_lows": beh_lows,
        "if_lows": if_lows,
        "se_thrs": se_thrs,
        "beh_thrs": beh_thrs,
        "if_thrs": if_thrs,
        "min_points_values": min_points_values,
        "n_candidates_after_recall_filter": int(len(comp)),
    }
    return comp, search_space


def apply_champion_policy(df_part: pd.DataFrame, champion: dict[str, Any]) -> np.ndarray:
    family = champion["family"]
    if family == "baseline":
        return df_part["lgbm_r4_pred"].astype(int).values
    if family == "quiet_veto":
        return policy_quiet_veto(
            df_part,
            safe_float(champion.get("score_cap")),
            safe_float(champion.get("se_low")),
            safe_float(champion.get("beh_low")),
            safe_float(champion.get("if_low")),
        ).astype(int)
    if family == "evidence_keep":
        return policy_evidence_keep(
            df_part,
            safe_float(champion.get("score_high")),
            safe_float(champion.get("se_thr")),
            safe_float(champion.get("beh_thr")),
            safe_float(champion.get("if_thr")),
        ).astype(int)
    if family == "points":
        return policy_points(
            df_part,
            safe_float(champion.get("score_thr")),
            safe_float(champion.get("min_points")),
        ).astype(int)
    raise ValueError(f"Family desconhecida: {family}")


def module_signal_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name in ["VALIDATION", "HOLDOUT_LABEL_SAFE", "HOLDOUT_FULL"]:
        _, part = get_split_arrays(df, split_name)
        lgbm_pos = part[part["lgbm_r4_pred"] == 1].copy()
        for is_fraud, g in lgbm_pos.groupby("is_fraud"):
            denom = max(len(g), 1)
            rows.append({
                "split": split_name,
                "is_fraud": int(is_fraud),
                "n_lgbm_r4_positive": int(len(g)),
                "avg_lgbm_r4_score": float(g["lgbm_r4_score"].mean()) if len(g) else 0.0,
                "se_score_ge_20": int((g["se_score"] >= 20).sum()),
                "se_score_ge_40": int((g["se_score"] >= 40).sum()),
                "se_score_ge_60": int((g["se_score"] >= 60).sum()),
                "beh_score_ge_15": int((g["behavioral_score"] >= 15).sum()),
                "beh_score_ge_25": int((g["behavioral_score"] >= 25).sum()),
                "if_ge_099": int((g["if_percentile"] >= 0.99).sum()),
                "if_ge_0995": int((g["if_percentile"] >= 0.995).sum()),
                "external_any_signal": int((g["external_any_signal"] == 1).sum()),
                "external_any_signal_rate": float((g["external_any_signal"] == 1).sum() / denom),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--r4-dir", default=str(R4_DIR))
    parser.add_argument("--r4-candidate-dir", default=str(R4_CANDIDATE_DIR))
    parser.add_argument("--candidate-dir", default=str(CANDIDATE_DIR))
    parser.add_argument("--artefatos-dir", default=str(ARTEFATOS_DIR))
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    r4_dir = Path(args.r4_dir)
    r4_candidate_dir = Path(args.r4_candidate_dir)
    candidate_dir = Path(args.candidate_dir)
    artefatos_dir = Path(args.artefatos_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXP-012D — LGBM R4 + IF/BEH/SE FP Reduction")
    print("=" * 80)
    print(f"Input:            {input_path}")
    print(f"R4 dir:           {r4_dir}")
    print(f"R4 candidate dir: {r4_candidate_dir}")
    print(f"Output dir:       {output_dir}")
    print(f"Target recall:    {args.target_recall}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    _, valid, holdout_safe, holdout_full = split_dataset(df)

    log.info("Dataset rows=%d fraud=%d normal=%d", len(df), int(df["is_fraud"].sum()), int((df["is_fraud"] == 0).sum()))
    log.info("Validation rows=%d fraud=%d", len(valid), int(valid["is_fraud"].sum()))
    log.info("Holdout label-safe rows=%d fraud=%d", len(holdout_safe), int(holdout_safe["is_fraud"].sum()))

    # R4 scoring.
    feature_schema = load_feature_schema(r4_candidate_dir)
    feature_cols = feature_schema["input_features_pre_transform"]
    model, preprocessor, model_path, preprocessor_path = load_r4_artifacts(r4_candidate_dir)

    log.info("Calculando score LGBM R4...")
    df["lgbm_r4_score"] = score_r4_lgbm(df, feature_cols, model, preprocessor)
    df["lgbm_r4_pred"], r4_policy = apply_r4_policy(df, df["lgbm_r4_score"].values, r4_dir, r4_candidate_dir)

    # Module scoring.
    import_info = import_optional_modules()
    modules = import_info["modules"]
    availability = import_info["availability"]

    df, module_details, availability = score_external_modules(
        df,
        df["lgbm_r4_pred"].astype(int).values == 1,
        modules,
        availability,
        artefatos_dir,
        fast=args.fast,
    )
    dump(availability, output_dir / "09_module_availability.json")

    module_details.to_csv(output_dir / "04_module_scores_lgbm_r4_positives.csv", index=False)
    coverage = module_signal_coverage(df)
    coverage.to_csv(output_dir / "03_module_signal_coverage.csv", index=False)

    # Policy search.
    comparison, search_space = run_policy_search(df, target_recall=args.target_recall, fast=args.fast)
    comparison.to_csv(output_dir / "01_policy_comparison.csv", index=False)
    dump(search_space, output_dir / "10_policy_search_space.json")

    champion = comparison.iloc[0].to_dict()

    # Apply champion to splits.
    metrics_rows = []
    pred_by_split = {}
    for split_name in ["VALIDATION", "HOLDOUT_LABEL_SAFE", "HOLDOUT_FULL"]:
        _, part = get_split_arrays(df, split_name)
        pred = apply_champion_policy(part, champion)
        pred_by_split[split_name] = (part, pred)
        metrics = eval_binary(part["is_fraud"].values, pred, part["lgbm_r4_score"].values)
        metrics.update({
            "temporal_split": split_name,
            "candidate_id": champion["candidate_id"],
            "family": champion["family"],
            "policy_desc": champion["policy_desc"],
        })
        metrics_rows.append(metrics)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "02_champion_metrics_by_split.csv", index=False)

    # Holdout predictions/errors.
    hsafe, hsafe_pred = pred_by_split["HOLDOUT_LABEL_SAFE"]
    hsafe_out = hsafe.copy()
    hsafe_out["exp012d_pred"] = hsafe_pred
    hsafe_out.to_csv(output_dir / "05_champion_predictions_holdout_label_safe.csv", index=False)
    hsafe_out[(hsafe_out["is_fraud"] == 1) & (hsafe_out["exp012d_pred"] == 0)].to_csv(output_dir / "06_champion_false_negatives_holdout_label_safe.csv", index=False)
    hsafe_out[(hsafe_out["is_fraud"] == 0) & (hsafe_out["exp012d_pred"] == 1)].to_csv(output_dir / "07_champion_false_positives_holdout_label_safe.csv", index=False)

    val_metrics = metrics_df[metrics_df["temporal_split"] == "VALIDATION"].iloc[0].to_dict()
    safe_metrics = metrics_df[metrics_df["temporal_split"] == "HOLDOUT_LABEL_SAFE"].iloc[0].to_dict()
    full_metrics = metrics_df[metrics_df["temporal_split"] == "HOLDOUT_FULL"].iloc[0].to_dict()

    # Compare against R4 established safe metrics.
    r4_safe_fp = 1604
    r4_safe_tp = 122
    r4_safe_fn = 2
    safe_fp_delta = int(safe_metrics["fp"]) - r4_safe_fp
    safe_tp_delta = int(safe_metrics["tp"]) - r4_safe_tp
    safe_fn_delta = int(safe_metrics["fn"]) - r4_safe_fn

    status = "VALIDATION_RECALL_TARGET_MET" if val_metrics["recall"] >= args.target_recall else "VALIDATION_RECALL_TARGET_NOT_MET"
    status += "_SAFE_RECALL_TARGET_MET" if safe_metrics["recall"] >= args.target_recall else "_SAFE_RECALL_TARGET_NOT_MET"
    if safe_metrics["recall"] >= args.target_recall and safe_fp_delta < 0:
        status += "_FP_REDUCED"
    else:
        status += "_FP_NOT_REDUCED_OR_UNSAFE"

    summary = {
        "experiment": "EXP-012D",
        "status": "DONE",
        "objective_status": status,
        "target_recall": args.target_recall,
        "input_path": str(input_path),
        "input_md5": file_md5(input_path),
        "n_rows": int(len(df)),
        "n_fraud": int(df["is_fraud"].sum()),
        "n_normal": int((df["is_fraud"] == 0).sum()),
        "r4_model_path": str(model_path),
        "r4_preprocessor_path": str(preprocessor_path),
        "r4_policy": r4_policy,
        "module_availability": availability,
        "n_policy_candidates": int(len(comparison)),
        "champion_candidate_id": champion["candidate_id"],
        "champion_family": champion["family"],
        "champion_policy_desc": champion["policy_desc"],
        "champion_details": champion,
        "champion_validation": val_metrics,
        "champion_holdout_label_safe": safe_metrics,
        "champion_holdout_full": full_metrics,
        "safe_fp_delta_vs_exp012c_r4": safe_fp_delta,
        "safe_tp_delta_vs_exp012c_r4": safe_tp_delta,
        "safe_fn_delta_vs_exp012c_r4": safe_fn_delta,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "candidate_dir": str(candidate_dir),
    }
    dump(summary, output_dir / "00_run_summary.json")

    # Candidate artifact.
    policy_artifact = {
        "experiment": "EXP-012D",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_recall": args.target_recall,
        "r4_policy": r4_policy,
        "external_policy": champion,
        "module_availability": availability,
        "feature_notes": {
            "lgbm_r4": "Score/predição do baseline R4 segmentado.",
            "external_modules": "Scores IF/SE/BEH calculados somente para positivos pelo LGBM R4.",
            "selection": "Campeão selecionado por validação: recall >= target, menor FP/FPR.",
        },
    }
    dump(policy_artifact, candidate_dir / "policy_exp012d_external_modules.json")
    dump({
        "model_version": "exp012d_lgbm_r4_external_modules_shadow",
        "status": "EXTERNAL_MODULES_FP_REDUCTION_CANDIDATE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "policy_path": str(candidate_dir / "policy_exp012d_external_modules.json"),
        "notes": [
            "Artefato shadow. Não sobrescreve produção.",
            "Usa LGBM R4 como primeiro estágio high-recall.",
            "Usa IF/SE/BEH como filtros de FP por política selecionada em validação.",
        ],
    }, candidate_dir / "manifest_exp012d_external_modules.json")

    # Recommendation.
    md = []
    md.append("# EXP-012D — LGBM R4 + External Modules FP Reduction")
    md.append("")
    md.append("## Champion")
    md.append(f"- candidate_id: `{champion['candidate_id']}`")
    md.append(f"- family: `{champion['family']}`")
    md.append(f"- policy: {champion['policy_desc']}")
    md.append("")
    md.append("## Validation")
    for k in ["tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr", "roc_auc", "average_precision"]:
        md.append(f"- {k}: {val_metrics.get(k)}")
    md.append("")
    md.append("## Holdout label-safe")
    for k in ["tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr", "roc_auc", "average_precision"]:
        md.append(f"- {k}: {safe_metrics.get(k)}")
    md.append("")
    md.append("## Delta vs EXP-012C-R4")
    md.append(f"- FP delta: {safe_fp_delta}")
    md.append(f"- TP delta: {safe_tp_delta}")
    md.append(f"- FN delta: {safe_fn_delta}")
    md.append("")
    md.append("## Module availability")
    for name, info in availability.items():
        md.append(f"- {name}: available={info.get('available')} error={info.get('error')}")
    md.append("")
    md.append("## Decisão sugerida")
    if safe_metrics["recall"] >= args.target_recall and safe_fp_delta < 0:
        md.append("APROVAR_COMO_BASELINE_CASCATA_LGBM_R4_IF_BEH_SE_SHADOW.")
    elif val_metrics["recall"] >= args.target_recall and safe_metrics["recall"] < args.target_recall:
        md.append("NÃO CONSOLIDAR: passou na validação, mas não manteve recall alvo no holdout label-safe.")
    elif safe_fp_delta >= 0:
        md.append("NÃO CONSOLIDAR: módulos externos não reduziram FP no holdout label-safe.")
    else:
        md.append("NÃO CONSOLIDAR: recall alvo não foi mantido.")
    md.append("")
    md.append("## Próximo passo")
    md.append("Se aprovado, executar EXP-012E E2E runtime shadow com a política campeã.")
    (output_dir / "08_recommendation.md").write_text("\n".join(md), encoding="utf-8")

    print("\n" + "=" * 80)
    print("EXP-012D CONCLUÍDO")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nArtefatos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_policy_comparison.csv",
        output_dir / "02_champion_metrics_by_split.csv",
        output_dir / "03_module_signal_coverage.csv",
        output_dir / "08_recommendation.md",
        candidate_dir / "policy_exp012d_external_modules.json",
        candidate_dir / "manifest_exp012d_external_modules.json",
    ]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
