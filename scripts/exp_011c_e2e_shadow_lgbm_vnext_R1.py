#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-011C — E2E Shadow com LGBM vNext R1/R2

Objetivo:
  Avaliar o candidato LGBM vNext dentro de um fluxo E2E shadow, sem sobrescrever
  os artefatos produtivos e sem alterar o DecisionEngine.

Como funciona:
  1. Lê o dataset enriquecido do EXP-010G-R2.
  2. Reaproveita, se existir, o replay baseline já gerado no EXP-011A_R2_FULL.
     Caso contrário, roda o baseline produtivo via simular_pipeline_e2e_v2.
  3. Carrega o modelo candidato R1:
       backend/artefatos_candidatos/exp011b_lgbm_vnext/
     e aplica o threshold R1 aprovado:
       threshold = 0.60
  4. Opcionalmente carrega o fallback R2:
       backend/artefatos_candidatos/exp011b_r2_lgbm_tuned/
  5. Compara:
       - BASELINE_PROD
       - R1_MODEL_ONLY
       - R1_ASSIST_BASELINE
       - R2_MODEL_ONLY, se --include-r2
       - R2_ASSIST_BASELINE, se --include-r2

Definição de ASSIST_BASELINE:
  - Se o baseline já decidiu CONFIRMAR/BLOQUEAR, preserva a decisão.
  - Se o baseline decidiu APROVAR e o candidato >= threshold, escala para CONFIRMAR.
  - Caso contrário, preserva APROVAR.

Entrada default:
  dados/hmo_ml_tb_pix_dataset_v2_180d_v1_enriched.csv

Saídas:
  resultados/experimentos/EXP-011C/
    00_run_summary.json
    01_metrics_comparison_by_split.csv
    02_confusion_comparison_by_split.csv
    03_shadow_delta_by_split.csv
    04_candidate_score_distribution.csv
    05_disagreements_holdout.csv
    06_new_true_positives_holdout.csv
    07_new_false_positives_holdout.csv
    08_remaining_false_negatives_holdout.csv
    09_rule_candidate_overlap.csv
    10_recommendation.md
    11_predictions_shadow.csv
    12_threshold_policies.json
    13_schema_check.json

Uso recomendado, reaproveitando o baseline já executado:
  python scripts/exp_011c_e2e_shadow_lgbm_vnext.py --include-r2

Forçar novo baseline E2E, se necessário:
  python scripts/exp_011c_e2e_shadow_lgbm_vnext.py --rerun-baseline --workers 1 --include-r2

Teste rápido:
  python scripts/exp_011c_e2e_shadow_lgbm_vnext.py --sample 5000 --rerun-baseline --workers 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# Windows UTF-8
# Patch EXP-011C-R1:
# Não reabrir sys.stdout/sys.stderr com TextIOWrapper.
# Em alguns ambientes Windows/PowerShell isso fecha o stream original e causa:
#   ValueError: I/O operation on closed file.
#   lost sys.stderr
# Usar reconfigure quando disponível; se não disponível, seguir sem alterar.
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

BACKEND_DIR = PROJECT_ROOT / "backend"
CORE_DIR = BACKEND_DIR / "core"
DADOS_DIR = PROJECT_ROOT / "dados"

sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT = DADOS_DIR / "hmo_ml_tb_pix_dataset_v2_180d_v1_enriched.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-011C"

DEFAULT_BASELINE_PRED = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-011A_R2_FULL" / "08_predictions.csv"

DEFAULT_R1_DIR = BACKEND_DIR / "artefatos_candidatos" / "exp011b_lgbm_vnext"
DEFAULT_R2_DIR = BACKEND_DIR / "artefatos_candidatos" / "exp011b_r2_lgbm_tuned"

# =============================================================================
# Imports do pipeline baseline
# =============================================================================
try:
    from backend.scripts.simular_pipeline_e2e_v2 import (
        process_batch_parallel,
        process_batch_sequential,
        validate_module_activations,
    )
except Exception:
    try:
        from scripts.simular_pipeline_e2e_v2 import (
            process_batch_parallel,
            process_batch_sequential,
            validate_module_activations,
        )
    except Exception:
        process_batch_parallel = None
        process_batch_sequential = None
        validate_module_activations = None

try:
    from core.pipeline_orquestrador import RAW_INPUT_COLUMNS
except Exception:
    RAW_INPUT_COLUMNS = []


# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | EXP-011C | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EXP-011C")


# =============================================================================
# Helpers gerais
# =============================================================================
FLAG_DECISIONS = {"CONFIRMAR", "BLOQUEAR"}


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def normalize_text_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def coalesce_columns(df: pd.DataFrame, target: str, sources: list[str]) -> pd.DataFrame:
    if target not in df.columns:
        df[target] = pd.Series(pd.NA, index=df.index, dtype="object")
    else:
        df[target] = df[target].astype("object")

    null_tokens = {"", "nan", "none", "null", "nat", "<na>"}
    for src in sources:
        if src not in df.columns:
            continue
        src_values = df[src].astype("object")
        target_text = df[target].astype("string").fillna("").str.strip().str.lower()
        mask = df[target].isna() | target_text.isin(null_tokens)
        df.loc[mask, target] = src_values.loc[mask].values
    return df


def adapt_dataset_for_baseline_replay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adapta o dataset enriquecido para o wrapper E2E baseline.
    Mantém todos os campos enriquecidos, mas garante aliases e defaults seguros.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    df = coalesce_columns(df, "cd_pix", ["transaction_id"])
    df = coalesce_columns(df, "transaction_id", ["cd_pix"])
    df = coalesce_columns(df, "cd_cpf_pagador", ["customer_id"])
    df = coalesce_columns(df, "customer_id", ["cd_cpf_pagador"])
    df = coalesce_columns(df, "cd_cpf_cnpj_recebedor", ["counterparty_id"])
    df = coalesce_columns(df, "dt_pix", ["event_datetime"])
    df = coalesce_columns(df, "event_datetime", ["dt_pix"])
    df = coalesce_columns(df, "ds_tipo_chave", ["ds_tipo_chave_norm"])
    df = coalesce_columns(df, "cd_retorno", ["autcodret"])

    for col in ["transaction_id", "cd_pix", "customer_id", "cd_cpf_pagador", "cd_cpf_cnpj_recebedor"]:
        if col in df.columns:
            df[col] = normalize_text_key(df[col])

    df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
    df["dt_pix"] = pd.to_datetime(df["dt_pix"], errors="coerce")
    if "data_pix" in df.columns:
        df["data_pix"] = pd.to_datetime(df["data_pix"], errors="coerce").dt.date
    else:
        df["data_pix"] = df["event_datetime"].dt.date

    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)

    for col in RAW_INPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Defaults necessários para evitar int(np.nan) no wrapper E2E.
    int_defaults = {
        "nr_idade": 0,
        "qt_tempo_relacionamento_mes": 0,
        "is_first_tx_trimestre": 0,
        "first_receiver_flag": 0,
        "burst_30m_flag": 0,
        "pix_key_random_flag": 0,
        "perfil_vulneravel_se_flag": 0,
    }
    for col, default in int_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default).astype(int)

    numeric_defaults = {
        "vl_pix": 0.0,
        "qt_total_pix_trimestre": 0.0,
        "vl_mediana_pix_trimestre": np.nan,
        "vl_desvio_padrao_pix_trimestre": np.nan,
        "qt_intervalo_transacao_minuto": np.nan,
        "qt_intervalo_mediana_trimestre": np.nan,
        "qt_intervalo_desvio_padrao_trimestre": np.nan,
        "qt_pix_dia_maximo_trimestre": 0.0,
        "latencia_rede_ms": np.nan,
        "vl_latencia_rede_media_trimestre": np.nan,
        "tempo_interacao_ms": np.nan,
        "vl_tempo_interacao_medio_trimestre": np.nan,
        "tempo_processamento_host_ms": np.nan,
        "topaz_risk_score": np.nan,
        "topaz_transacao_rejeitada": 0.0,
        "qt_aparelhos_distintos_trimestre": 0.0,
        "vl_renda_cliente": np.nan,
        "qt_dependentes": 0.0,
    }
    for col, default in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if not pd.isna(default):
            df[col] = df[col].fillna(default)

    text_defaults = {
        "ds_tipo_chave": "INFORMACAO_AUSENTE",
        "ds_chave_pix": np.nan,
        "device_name": np.nan,
        "app_version": np.nan,
        "ip_address": np.nan,
        "metodo_autenticacao": np.nan,
        "session_id": np.nan,
        "cd_retorno": np.nan,
        "is_agendamento_recorrente": np.nan,
        "ds_sexo": np.nan,
        "ds_estado_civil": np.nan,
        "ds_segmento": np.nan,
    }
    for col, default in text_defaults.items():
        if col not in df.columns:
            df[col] = default
        if pd.isna(default):
            df[col] = df[col].where(df[col].notna(), np.nan)
        else:
            df[col] = df[col].fillna(default)

    if "temporal_split" not in df.columns:
        df["temporal_split"] = "UNKNOWN"
    if "dataset_role" not in df.columns:
        df["dataset_role"] = np.where(df["is_fraud"] == 1, "POSITIVE_FRAUD", "NEGATIVE_NORMAL")
    if "sample_strategy" not in df.columns:
        df["sample_strategy"] = "UNKNOWN"

    df = df[df["transaction_id"].notna()].copy()
    df = df[df["event_datetime"].notna()].copy()
    return df.sort_values(["event_datetime", "transaction_id"]).reset_index(drop=True)


def load_input(path: Path, sample: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = adapt_dataset_for_baseline_replay(df)

    if sample is not None and sample < len(df):
        fraud = df[df["is_fraud"] == 1].copy()
        normal = df[df["is_fraud"] == 0].copy()
        # Preserva fraudes sempre que possível.
        if sample >= len(fraud):
            n_normal = sample - len(fraud)
            df = pd.concat([
                fraud,
                normal.sample(n=min(n_normal, len(normal)), random_state=seed),
            ], axis=0)
        else:
            df = fraud.sample(n=sample, random_state=seed)
        df = df.sort_values(["event_datetime", "transaction_id"]).reset_index(drop=True)

    return df


def normalize_baseline_predictions(input_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Une baseline ao input. Se pred_df vier do EXP-011A, normalmente já contém input + predição.
    Se vier diretamente do process_batch, contém idx + decisao.
    """
    pred = pred_df.copy()
    pred.columns = [str(c).strip() for c in pred.columns]

    if "decisao" not in pred.columns:
        raise RuntimeError("Baseline predictions sem coluna 'decisao'.")

    # Caso já tenha transaction_id, usar chave.
    if "transaction_id" in pred.columns:
        pred["transaction_id"] = normalize_text_key(pred["transaction_id"])
        merged = input_df.merge(
            pred.drop_duplicates("transaction_id", keep="first"),
            on="transaction_id",
            how="left",
            suffixes=("", "_baseline"),
        )
        return merged

    # Caso process_batch retorne idx.
    if "idx" in pred.columns:
        pred["idx"] = pd.to_numeric(pred["idx"], errors="coerce").astype("Int64")
        merged = input_df.reset_index().rename(columns={"index": "idx"}).merge(
            pred,
            on="idx",
            how="left",
            suffixes=("", "_baseline"),
        )
        return merged

    raise RuntimeError("Baseline predictions sem transaction_id nem idx.")


def run_or_load_baseline(input_df: pd.DataFrame, baseline_path: Path, rerun: bool, workers: int, output_dir: Path) -> pd.DataFrame:
    if (not rerun) and baseline_path.exists():
        log.info("Reaproveitando baseline existente: %s", baseline_path)
        pred = pd.read_csv(baseline_path, low_memory=False)
        merged = normalize_baseline_predictions(input_df, pred)
        return merged

    if process_batch_sequential is None:
        raise RuntimeError("Não foi possível importar simular_pipeline_e2e_v2. Forneça --baseline-predictions existente.")

    log.info("Rodando baseline produtivo via PipelineOrquestrador...")
    if workers > 1:
        pred_df = process_batch_parallel(input_df, n_workers=workers)
    else:
        pred_df = process_batch_sequential(input_df, progress_every=1000)

    pred_path = output_dir / "baseline_predictions_rerun.csv"
    pred_df.to_csv(pred_path, index=False)
    log.info("Baseline rerun salvo em: %s", pred_path)

    merged = normalize_baseline_predictions(input_df, pred_df)
    return merged


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# =============================================================================
# Candidate scoring
# =============================================================================
def load_candidate_r1(candidate_dir: Path) -> dict[str, Any]:
    model_path = candidate_dir / "model_lgbm_vnext_shadow.joblib"
    prep_path = candidate_dir / "preprocessor_lgbm_vnext_shadow.joblib"
    feature_path = candidate_dir / "features_lgbm_vnext_shadow.json"
    threshold_path = candidate_dir / "threshold_policy_exp011b_r1.json"

    missing = [p for p in [model_path, prep_path, feature_path, threshold_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Artefatos R1 ausentes: {missing}")

    threshold_policy = read_json(threshold_path)
    features = read_json(feature_path)
    return {
        "name": "R1",
        "model": joblib.load(model_path),
        "preprocessor": joblib.load(prep_path),
        "feature_schema": features,
        "threshold_policy": threshold_policy,
        "threshold": float(threshold_policy["threshold"]),
        "model_path": str(model_path),
        "preprocessor_path": str(prep_path),
    }


def load_candidate_r2(candidate_dir: Path) -> dict[str, Any]:
    model_path = candidate_dir / "model_lgbm_r2_tuned_shadow.joblib"
    prep_path = candidate_dir / "preprocessor_lgbm_r2_tuned_shadow.joblib"
    threshold_path = candidate_dir / "threshold_policy_exp011b_r2.json"

    missing = [p for p in [model_path, prep_path, threshold_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Artefatos R2 ausentes: {missing}")

    threshold_policy = read_json(threshold_path)
    # O R2 não precisa de features.json separado; usa o mesmo schema inferido no preprocessor.
    # Como o preprocessor foi treinado com colunas pré-transformação do EXP-011B, reaproveitamos
    # a lista do R1 se disponível no fluxo. Se não houver, tentaremos feature_names_in_.
    return {
        "name": "R2",
        "model": joblib.load(model_path),
        "preprocessor": joblib.load(prep_path),
        "feature_schema": None,
        "threshold_policy": threshold_policy,
        "threshold": float(threshold_policy["threshold"]),
        "model_path": str(model_path),
        "preprocessor_path": str(prep_path),
    }


def infer_candidate_feature_cols(candidate: dict[str, Any], fallback_schema: dict[str, Any] | None) -> list[str]:
    schema = candidate.get("feature_schema") or fallback_schema
    if schema and "input_features_pre_transform" in schema:
        return list(schema["input_features_pre_transform"])

    preprocessor = candidate["preprocessor"]
    if hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)

    raise RuntimeError(f"Não foi possível inferir features do candidato {candidate['name']}.")


def score_candidate(df: pd.DataFrame, candidate: dict[str, Any], feature_cols: list[str]) -> np.ndarray:
    X = df.copy()
    for col in feature_cols:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feature_cols]
    Xt = candidate["preprocessor"].transform(X)
    return candidate["model"].predict_proba(Xt)[:, 1]


# =============================================================================
# Metrics
# =============================================================================
def flagged_from_decision(decision: pd.Series) -> np.ndarray:
    return decision.astype(str).str.upper().isin(FLAG_DECISIONS).astype(int).values


def metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, Any]:
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
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 8)
        out["average_precision"] = round(float(average_precision_score(y_true, y_score)), 8)
    return out


def build_decision_cols(df: pd.DataFrame, decision_on_hit: str, include_r2: bool) -> pd.DataFrame:
    out = df.copy()
    out["decision_BASELINE_PROD"] = out["decisao"].astype(str).str.upper()
    baseline_flag = out["decision_BASELINE_PROD"].isin(FLAG_DECISIONS)

    # R1 model-only
    out["decision_R1_MODEL_ONLY"] = np.where(out["score_R1"] >= out.attrs["threshold_R1"], decision_on_hit, "APROVAR")

    # R1 assist
    out["decision_R1_ASSIST_BASELINE"] = out["decision_BASELINE_PROD"]
    hit_r1 = out["score_R1"] >= out.attrs["threshold_R1"]
    mask_r1 = (~baseline_flag) & hit_r1
    out.loc[mask_r1, "decision_R1_ASSIST_BASELINE"] = decision_on_hit

    if include_r2 and "score_R2" in out.columns:
        out["decision_R2_MODEL_ONLY"] = np.where(out["score_R2"] >= out.attrs["threshold_R2"], decision_on_hit, "APROVAR")
        out["decision_R2_ASSIST_BASELINE"] = out["decision_BASELINE_PROD"]
        hit_r2 = out["score_R2"] >= out.attrs["threshold_R2"]
        mask_r2 = (~baseline_flag) & hit_r2
        out.loc[mask_r2, "decision_R2_ASSIST_BASELINE"] = decision_on_hit

    return out


def comparison_metrics(df: pd.DataFrame, decision_cols: list[str]) -> pd.DataFrame:
    rows = []
    for split, g in df.groupby("temporal_split", dropna=False):
        y = g["is_fraud"].astype(int).values
        for col in decision_cols:
            y_pred = flagged_from_decision(g[col])
            score_col = None
            if "R1" in col and "score_R1" in g.columns:
                score_col = "score_R1"
            if "R2" in col and "score_R2" in g.columns:
                score_col = "score_R2"

            m = metrics(y, y_pred, g[score_col].values if score_col else None)
            m.update({
                "temporal_split": split,
                "policy": col.replace("decision_", ""),
                "n": len(g),
                "frauds": int(g["is_fraud"].sum()),
                "normals": int((g["is_fraud"] == 0).sum()),
            })
            rows.append(m)

    # global
    y = df["is_fraud"].astype(int).values
    for col in decision_cols:
        y_pred = flagged_from_decision(df[col])
        score_col = None
        if "R1" in col and "score_R1" in df.columns:
            score_col = "score_R1"
        if "R2" in col and "score_R2" in df.columns:
            score_col = "score_R2"
        m = metrics(y, y_pred, df[score_col].values if score_col else None)
        m.update({
            "temporal_split": "GLOBAL",
            "policy": col.replace("decision_", ""),
            "n": len(df),
            "frauds": int(df["is_fraud"].sum()),
            "normals": int((df["is_fraud"] == 0).sum()),
        })
        rows.append(m)

    cols = ["temporal_split", "policy", "n", "frauds", "normals", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr", "roc_auc", "average_precision"]
    out = pd.DataFrame(rows)
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols].sort_values(["temporal_split", "policy"]).reset_index(drop=True)


def confusion_distribution(df: pd.DataFrame, decision_cols: list[str]) -> pd.DataFrame:
    rows = []
    for split, g in df.groupby("temporal_split", dropna=False):
        for col in decision_cols:
            for dec, gd in g.groupby(col, dropna=False):
                rows.append({
                    "temporal_split": split,
                    "policy": col.replace("decision_", ""),
                    "decision": dec,
                    "n": len(gd),
                    "frauds": int(gd["is_fraud"].sum()),
                    "normals": int((gd["is_fraud"] == 0).sum()),
                    "fraud_rate": round(float(gd["is_fraud"].mean()), 8) if len(gd) else 0.0,
                })
    return pd.DataFrame(rows).sort_values(["temporal_split", "policy", "decision"]).reset_index(drop=True)


def delta_by_split(df: pd.DataFrame, candidate_cols: list[str]) -> pd.DataFrame:
    rows = []
    baseline_flag = df["decision_BASELINE_PROD"].isin(FLAG_DECISIONS)
    for split, g in df.groupby("temporal_split", dropna=False):
        bflag = g["decision_BASELINE_PROD"].isin(FLAG_DECISIONS)
        for col in candidate_cols:
            cflag = g[col].isin(FLAG_DECISIONS)
            new_flags = (~bflag) & cflag
            removed_flags = bflag & (~cflag)
            rows.append({
                "temporal_split": split,
                "policy": col.replace("decision_", ""),
                "new_flags": int(new_flags.sum()),
                "new_tp": int(((g["is_fraud"] == 1) & new_flags).sum()),
                "new_fp": int(((g["is_fraud"] == 0) & new_flags).sum()),
                "removed_flags": int(removed_flags.sum()),
                "removed_tp": int(((g["is_fraud"] == 1) & removed_flags).sum()),
                "removed_fp": int(((g["is_fraud"] == 0) & removed_flags).sum()),
            })
    return pd.DataFrame(rows).sort_values(["temporal_split", "policy"]).reset_index(drop=True)


def score_distribution(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    rows = []
    for split, g in df.groupby("temporal_split", dropna=False):
        for is_fraud, gg in g.groupby("is_fraud", dropna=False):
            for col in score_cols:
                s = pd.to_numeric(gg[col], errors="coerce")
                rows.append({
                    "temporal_split": split,
                    "is_fraud": int(is_fraud),
                    "score": col,
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
    return pd.DataFrame(rows).sort_values(["temporal_split", "score", "is_fraud"]).reset_index(drop=True)


def rule_candidate_overlap(df: pd.DataFrame) -> pd.DataFrame:
    if "cascade_rules" not in df.columns:
        return pd.DataFrame(columns=["rule", "n", "frauds", "candidate_r1_hits", "candidate_r1_hit_rate"])

    tmp = df.copy()
    tmp["rule"] = tmp["cascade_rules"].fillna("").astype(str).str.slice(0, 160)
    tmp = tmp[tmp["rule"].str.len() > 0]
    if tmp.empty:
        return pd.DataFrame(columns=["rule", "n", "frauds", "candidate_r1_hits", "candidate_r1_hit_rate"])

    rows = []
    th = df.attrs.get("threshold_R1", 0.6)
    for rule, g in tmp.groupby("rule"):
        hits = pd.to_numeric(g["score_R1"], errors="coerce") >= th
        rows.append({
            "rule": rule,
            "n": len(g),
            "frauds": int(g["is_fraud"].sum()),
            "normals": int((g["is_fraud"] == 0).sum()),
            "candidate_r1_hits": int(hits.sum()),
            "candidate_r1_hit_rate": round(float(hits.mean()), 8),
            "rule_precision": round(float(g["is_fraud"].mean()), 8),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def build_recommendation(summary: dict[str, Any], metrics_df: pd.DataFrame) -> str:
    def get(split: str, policy: str) -> dict[str, Any]:
        r = metrics_df[(metrics_df["temporal_split"] == split) & (metrics_df["policy"] == policy)]
        return r.iloc[0].to_dict() if not r.empty else {}

    base_h = get("HOLDOUT", "BASELINE_PROD")
    r1_h = get("HOLDOUT", "R1_ASSIST_BASELINE")
    r1_model_h = get("HOLDOUT", "R1_MODEL_ONLY")
    r2_h = get("HOLDOUT", "R2_ASSIST_BASELINE")

    lines = []
    lines.append("# EXP-011C — E2E Shadow LGBM vNext")
    lines.append("")
    lines.append("## Status sugerido")
    if r1_h and base_h and (r1_h.get("f1", 0) > base_h.get("f1", 0)) and (r1_h.get("fp", 999999) <= base_h.get("fp", 999999)):
        lines.append("APROVADO_PARA_ANALISE_DE_PATCH_SHADOW")
    else:
        lines.append("DIAGNOSTICO_E2E_SHADOW_REQUER_ANALISE")
    lines.append("")
    lines.append("## Resumo")
    lines.append(f"- Linhas: {summary['n_rows']}")
    lines.append(f"- Fraudes: {summary['n_fraud']}")
    lines.append(f"- Normais: {summary['n_normal']}")
    lines.append(f"- Threshold R1: {summary['threshold_r1']}")
    if summary.get("threshold_r2") is not None:
        lines.append(f"- Threshold R2: {summary['threshold_r2']}")
    lines.append("")
    lines.append("## HOLDOUT")
    for name, row in [
        ("Baseline produtivo", base_h),
        ("R1 model-only", r1_model_h),
        ("R1 assist baseline", r1_h),
        ("R2 assist baseline", r2_h),
    ]:
        if not row:
            continue
        lines.append(f"### {name}")
        for k in ["tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr"]:
            lines.append(f"- {k}: {row.get(k)}")
        lines.append("")
    lines.append("## Próxima decisão")
    lines.append(
        "Se o R1 assist baseline melhorar o F1 e reduzir/segurar FP versus baseline no holdout, "
        "seguir para EXP-011D com patch shadow controlado no DecisionEngine. "
        "Se o R1 model-only for melhor que o assist, avaliar substituição do score do modelo em shadow. "
        "Não promover automaticamente sem validar no dataset completo não truncado e sem regressão dos testes."
    )
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP-011C — E2E Shadow com LGBM vNext R1/R2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="CSV enriquecido do EXP-010G-R2.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Diretório de saída.")
    parser.add_argument("--baseline-predictions", default=str(DEFAULT_BASELINE_PRED), help="Predições baseline do EXP-011A_R2_FULL.")
    parser.add_argument("--rerun-baseline", action="store_true", help="Força rerun do baseline via PipelineOrquestrador.")
    parser.add_argument("--workers", type=int, default=1, help="Workers para rerun baseline.")
    parser.add_argument("--candidate-r1-dir", default=str(DEFAULT_R1_DIR), help="Diretório dos artefatos R1.")
    parser.add_argument("--candidate-r2-dir", default=str(DEFAULT_R2_DIR), help="Diretório dos artefatos R2.")
    parser.add_argument("--include-r2", action="store_true", help="Inclui fallback R2 na comparação.")
    parser.add_argument("--decision-on-hit", choices=["CONFIRMAR", "BLOQUEAR"], default="CONFIRMAR", help="Decisão shadow quando candidato bate threshold.")
    parser.add_argument("--sample", type=int, default=None, help="Amostra opcional para teste rápido.")
    parser.add_argument("--seed", type=int, default=42, help="Seed da amostra.")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    baseline_path = Path(args.baseline_predictions)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    print("=" * 80)
    print("EXP-011C — E2E Shadow LGBM vNext")
    print("=" * 80)
    print(f"Input:      {input_path}")
    print(f"Output:     {output_dir}")
    print(f"Baseline:   {baseline_path}")
    print(f"MD5 input:  {file_md5(input_path)}")

    log.info("Carregando input...")
    input_df = load_input(input_path, sample=args.sample, seed=args.seed)

    log.info("Carregando/rodando baseline...")
    baseline_df = run_or_load_baseline(
        input_df=input_df,
        baseline_path=baseline_path,
        rerun=args.rerun_baseline or args.sample is not None,
        workers=args.workers,
        output_dir=output_dir,
    )

    # Garantir colunas principais vindas do input caso o merge tenha criado sufixos.
    for col in ["is_fraud", "temporal_split", "dataset_role", "sample_strategy", "event_datetime"]:
        if col not in baseline_df.columns and f"{col}_baseline" in baseline_df.columns:
            baseline_df[col] = baseline_df[f"{col}_baseline"]

    if "decisao" not in baseline_df.columns:
        raise RuntimeError("Baseline final não possui coluna decisao.")

    # Se houve baseline reaproveitado com 100k e input 100k, OK. Se sample foi usado, rerun é obrigatório.
    baseline_df["transaction_id"] = normalize_text_key(baseline_df["transaction_id"])
    baseline_df["is_fraud"] = pd.to_numeric(baseline_df["is_fraud"], errors="coerce").fillna(0).astype(int)
    baseline_df["temporal_split"] = baseline_df["temporal_split"].astype(str).str.upper().str.strip()

    # Carregar candidatos.
    log.info("Carregando candidato R1...")
    r1 = load_candidate_r1(Path(args.candidate_r1_dir))
    r1_features = infer_candidate_feature_cols(r1, None)
    log.info("R1 threshold: %.6f | features=%d", r1["threshold"], len(r1_features))

    candidates = [r1]
    r2 = None
    r2_features = None
    if args.include_r2:
        log.info("Carregando candidato R2...")
        r2 = load_candidate_r2(Path(args.candidate_r2_dir))
        r2_features = infer_candidate_feature_cols(r2, r1["feature_schema"])
        log.info("R2 threshold: %.6f | features=%d", r2["threshold"], len(r2_features))
        candidates.append(r2)

    # Score dos candidatos no dataframe baseline, que contém todas as colunas do input.
    log.info("Scoring R1...")
    baseline_df["score_R1"] = score_candidate(baseline_df, r1, r1_features)

    baseline_df.attrs["threshold_R1"] = r1["threshold"]
    if args.include_r2 and r2 is not None and r2_features is not None:
        log.info("Scoring R2...")
        baseline_df["score_R2"] = score_candidate(baseline_df, r2, r2_features)
        baseline_df.attrs["threshold_R2"] = r2["threshold"]

    decision_df = build_decision_cols(
        baseline_df,
        decision_on_hit=args.decision_on_hit,
        include_r2=args.include_r2,
    )
    # Pandas attrs não persiste em cópia em todas as versões.
    decision_df.attrs["threshold_R1"] = r1["threshold"]
    if args.include_r2 and r2 is not None:
        decision_df.attrs["threshold_R2"] = r2["threshold"]

    decision_cols = ["decision_BASELINE_PROD", "decision_R1_MODEL_ONLY", "decision_R1_ASSIST_BASELINE"]
    if args.include_r2:
        decision_cols += ["decision_R2_MODEL_ONLY", "decision_R2_ASSIST_BASELINE"]

    metrics_df = comparison_metrics(decision_df, decision_cols)
    metrics_df.to_csv(output_dir / "01_metrics_comparison_by_split.csv", index=False)

    conf_df = confusion_distribution(decision_df, decision_cols)
    conf_df.to_csv(output_dir / "02_confusion_comparison_by_split.csv", index=False)

    candidate_delta_cols = [c for c in decision_cols if c != "decision_BASELINE_PROD"]
    delta_df = delta_by_split(decision_df, candidate_delta_cols)
    delta_df.to_csv(output_dir / "03_shadow_delta_by_split.csv", index=False)

    score_cols = ["score_R1"] + (["score_R2"] if args.include_r2 and "score_R2" in decision_df.columns else [])
    score_distribution(decision_df, score_cols).to_csv(output_dir / "04_candidate_score_distribution.csv", index=False)

    # Holdout-focused artifacts.
    holdout = decision_df[decision_df["temporal_split"] == "HOLDOUT"].copy()
    if holdout.empty:
        holdout = decision_df.copy()

    baseline_flag = holdout["decision_BASELINE_PROD"].isin(FLAG_DECISIONS)
    r1_flag = holdout["decision_R1_ASSIST_BASELINE"].isin(FLAG_DECISIONS)

    disagreements = holdout[baseline_flag != r1_flag].copy()
    disagreements.to_csv(output_dir / "05_disagreements_holdout.csv", index=False)

    new_flags = (~baseline_flag) & r1_flag
    new_tp = holdout[(holdout["is_fraud"] == 1) & new_flags].copy()
    new_fp = holdout[(holdout["is_fraud"] == 0) & new_flags].copy()
    remaining_fn = holdout[(holdout["is_fraud"] == 1) & (~r1_flag)].copy()

    new_tp.to_csv(output_dir / "06_new_true_positives_holdout.csv", index=False)
    new_fp.to_csv(output_dir / "07_new_false_positives_holdout.csv", index=False)
    remaining_fn.to_csv(output_dir / "08_remaining_false_negatives_holdout.csv", index=False)

    rule_candidate_overlap(decision_df).to_csv(output_dir / "09_rule_candidate_overlap.csv", index=False)

    # Predições completas.
    decision_df.to_csv(output_dir / "11_predictions_shadow.csv", index=False)

    threshold_policies = {
        "r1": {
            "threshold": r1["threshold"],
            "threshold_policy": r1["threshold_policy"],
            "model_path": r1["model_path"],
            "preprocessor_path": r1["preprocessor_path"],
        },
        "r2": None,
    }
    if args.include_r2 and r2 is not None:
        threshold_policies["r2"] = {
            "threshold": r2["threshold"],
            "threshold_policy": r2["threshold_policy"],
            "model_path": r2["model_path"],
            "preprocessor_path": r2["preprocessor_path"],
        }
    dump_json(threshold_policies, output_dir / "12_threshold_policies.json")

    schema_check = {
        "input_path": str(input_path),
        "input_md5": file_md5(input_path),
        "n_input": int(len(input_df)),
        "n_baseline_rows": int(len(baseline_df)),
        "missing_r1_features": [c for c in r1_features if c not in decision_df.columns],
        "missing_r2_features": [c for c in r2_features if c not in decision_df.columns] if r2_features else [],
        "decision_columns": decision_cols,
    }
    dump_json(schema_check, output_dir / "13_schema_check.json")

    elapsed = time.perf_counter() - t0
    summary = {
        "experiment": "EXP-011C",
        "status": "DONE",
        "input_path": str(input_path),
        "input_md5": file_md5(input_path),
        "output_dir": str(output_dir),
        "baseline_predictions": str(baseline_path),
        "baseline_rerun": bool(args.rerun_baseline or args.sample is not None),
        "sample_n": args.sample,
        "n_rows": int(len(decision_df)),
        "n_fraud": int(decision_df["is_fraud"].sum()),
        "n_normal": int((decision_df["is_fraud"] == 0).sum()),
        "threshold_r1": float(r1["threshold"]),
        "threshold_r2": float(r2["threshold"]) if r2 is not None else None,
        "decision_on_hit": args.decision_on_hit,
        "include_r2": bool(args.include_r2),
        "elapsed_seconds": round(elapsed, 2),
    }

    # Acrescentar métricas holdout principais.
    for policy in ["BASELINE_PROD", "R1_MODEL_ONLY", "R1_ASSIST_BASELINE", "R2_MODEL_ONLY", "R2_ASSIST_BASELINE"]:
        row = metrics_df[(metrics_df["temporal_split"] == "HOLDOUT") & (metrics_df["policy"] == policy)]
        if not row.empty:
            summary[f"holdout_{policy}"] = row.iloc[0].to_dict()

    dump_json(summary, output_dir / "00_run_summary.json")

    rec = build_recommendation(summary, metrics_df)
    (output_dir / "10_recommendation.md").write_text(rec, encoding="utf-8")

    print("\n" + "=" * 80)
    print("EXP-011C CONCLUÍDO")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nArtefatos:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_metrics_comparison_by_split.csv",
        output_dir / "03_shadow_delta_by_split.csv",
        output_dir / "05_disagreements_holdout.csv",
        output_dir / "10_recommendation.md",
        output_dir / "11_predictions_shadow.csv",
    ]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
