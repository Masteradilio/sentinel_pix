"""
experimentos/exp_005b_calibracao_engine/run_exp_005b_calibracao_engine.py

EXP-005B — Recalibracao pos-LGBM v6.2

Objetivo:
  Calibrar o threshold model-only do LGBM candidato gerado no EXP-005A
  antes de qualquer integracao E2E com DecisionEngine.

Esta versao e robusta:
  - Nao chama PipelineOrquestrador.
  - Nao importa DecisionEngine.
  - Nao altera backend/artefatos.
  - Apenas valida scoring_config.json e registra se o runtime esta apto.
  - Gera patch candidato de configuracao para futura avaliacao E2E.

Entradas esperadas:
  dados/base_treino_final.csv
  backend/artefatos_candidatos/exp_005a_lgbm_v6_2_recall/
    lgbm_v6_2_recall_candidate.joblib
    lgbm_features_v6_2.json
    thresholds_lgbm_v6_2.json

Saidas:
  resultados/experimentos/EXP-005B/
    00_preflight_runtime.json
    01_grid_thresholds_model_only.csv
    02_top_configs.json
    03_delta_fp_fn_melhor_config.json
    04_validacao_cruzada_model_only.json
    05_conclusao_executiva.md
    scoring_config_candidate_patch.json

Uso:
  python experimentos\\exp_005b_calibracao_engine\\run_exp_005b_calibracao_engine.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import f1_score, precision_score, recall_score
except Exception as exc:
    raise RuntimeError("Instale scikit-learn: pip install scikit-learn") from exc


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
OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-005B"


# =========================================================
# LOGGING
# =========================================================

EXP_ID = "EXP-005B"

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
# BASELINE
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
# JSON HELPERS
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
    return obj


def safe_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe_json(obj), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# =========================================================
# PREFLIGHT
# =========================================================

def validate_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "valid_json": False,
            "error": "arquivo nao encontrado",
        }

    try:
        obj = read_json(path)
        return {
            "path": str(path),
            "exists": True,
            "valid_json": True,
            "type": type(obj).__name__,
        }
    except Exception as exc:
        return {
            "path": str(path),
            "exists": True,
            "valid_json": False,
            "error": str(exc),
        }


def runtime_preflight() -> dict[str, Any]:
    scoring_path = ARTEFATOS_DIR / "scoring_config.json"
    lgbm_features_path = ARTEFATOS_DIR / "lgbm_features.json"

    checks = {
        "scoring_config": validate_json_file(scoring_path),
        "lgbm_features_atual": validate_json_file(lgbm_features_path),
        "candidate_model_exists": (CANDIDATE_DIR / "lgbm_v6_2_recall_candidate.joblib").exists(),
        "candidate_features_exists": (CANDIDATE_DIR / "lgbm_features_v6_2.json").exists(),
        "candidate_thresholds_exists": (CANDIDATE_DIR / "thresholds_lgbm_v6_2.json").exists(),
    }

    runtime_ready = (
        checks["scoring_config"]["valid_json"]
        and checks["lgbm_features_atual"]["valid_json"]
        and checks["candidate_model_exists"]
        and checks["candidate_features_exists"]
    )

    return {
        "runtime_ready_for_e2e": bool(runtime_ready),
        "checks": checks,
        "note": (
            "EXP-005B model-only pode rodar mesmo com runtime_ready_for_e2e=false. "
            "Para E2E real, corrija scoring_config.json antes."
        ),
    }


# =========================================================
# DATA / FEATURES
# =========================================================

def _prepare_label(df: pd.DataFrame) -> pd.Series:
    if "is_fraud" not in df.columns:
        raise ValueError("Dataset precisa conter coluna is_fraud.")
    return pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def add_exp005a_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Recria as features candidatas do EXP-005A caso o artefato candidato as use.
    O vencedor atual usa baseline_features, mas este helper deixa o script robusto.
    """
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


def load_candidate_features() -> list[str]:
    path = CANDIDATE_DIR / "lgbm_features_v6_2.json"
    obj = read_json(path)

    if isinstance(obj, list):
        return [str(x) for x in obj]

    if isinstance(obj, dict):
        for key in ["features", "feature_names", "lgbm_features"]:
            if key in obj:
                return [str(x) for x in obj[key]]

    raise ValueError(f"Formato inesperado em {path}")


def ensure_numeric_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)

    for col in features:
        if col in df.columns:
            x[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            x[col] = np.nan

    x = x.replace([np.inf, -np.inf], np.nan)
    med = x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = x.fillna(med).fillna(0.0)

    return x.astype(float)


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
        "Sample seed=%s: %d tx (%d fraudes + %d normais)",
        seed,
        len(out),
        int(out["is_fraud"].sum()),
        int((out["is_fraud"] == 0).sum()),
    )

    return out


# =========================================================
# METRICS / GRID
# =========================================================

def metrics_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
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


def evaluate_thresholds(
    sample: pd.DataFrame,
    scores: np.ndarray,
    thresholds: list[float],
    seed: int,
) -> pd.DataFrame:
    y_true = sample["is_fraud"].astype(int).values

    rows = []

    for th in thresholds:
        y_pred = (scores >= th).astype(int)
        m = metrics_from_pred(y_true, y_pred)

        row = {
            "seed": seed,
            "lgbm_threshold": float(th),
            **m,
            "delta_TP_vs_baseline": m["TP"] - BASELINE_FASE2["TP"],
            "delta_FP_vs_baseline": m["FP"] - BASELINE_FASE2["FP"],
            "delta_FN_vs_baseline": m["FN"] - BASELINE_FASE2["FN"],
            "delta_F1_vs_baseline": round(m["F1"] - BASELINE_FASE2["F1"], 6),
            "passes_min_fn": bool(m["FN"] <= 7),
            "passes_strong_fn": bool(m["FN"] <= 5),
            "passes_fp_22": bool(m["FP"] <= 22),
            "passes_fp_25": bool(m["FP"] <= 25),
            "passes_precision_94": bool(m["Precision"] >= 0.94),
            "passes_precision_935": bool(m["Precision"] >= 0.935),
            "passes_fpr_005": bool(m["FPR"] <= 0.005),
            "passes_f1_baseline": bool(m["F1"] >= BASELINE_FASE2["F1"]),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def select_best_config(grid_main: pd.DataFrame, grid_val: pd.DataFrame) -> dict[str, Any]:
    merged = grid_main.merge(
        grid_val,
        on="lgbm_threshold",
        suffixes=("_main", "_val"),
    )

    # Critério forte: precisa passar nos dois seeds.
    eligible = merged[
        (merged["FN_main"] <= 5)
        & (merged["FN_val"] <= 5)
        & (merged["FP_main"] <= 25)
        & (merged["FP_val"] <= 25)
        & (merged["Precision_main"] >= 0.935)
        & (merged["Precision_val"] >= 0.935)
        & (merged["FPR_main"] <= 0.005)
        & (merged["FPR_val"] <= 0.005)
    ].copy()

    selection_tier = "strong"

    # Critério mínimo.
    if eligible.empty:
        eligible = merged[
            (merged["FN_main"] <= 7)
            & (merged["FN_val"] <= 7)
            & (merged["FP_main"] <= 22)
            & (merged["FP_val"] <= 22)
            & (merged["Precision_main"] >= 0.94)
            & (merged["Precision_val"] >= 0.94)
        ].copy()
        selection_tier = "minimum"

    # Fallback controlado para EXP-005B: menor FN com FP/FPR controlados.
    if eligible.empty:
        eligible = merged[
            (merged["FP_main"] <= 25)
            & (merged["FP_val"] <= 30)
            & (merged["FPR_main"] <= 0.005)
            & (merged["FPR_val"] <= 0.006)
        ].copy()
        selection_tier = "controlled_fallback"

    # Fallback total: apenas melhor compromisso.
    if eligible.empty:
        eligible = merged.copy()
        selection_tier = "best_available"

    # Ordenação: menor pior FN, depois menor pior FP, depois maior F1 médio.
    eligible["worst_FN"] = eligible[["FN_main", "FN_val"]].max(axis=1)
    eligible["worst_FP"] = eligible[["FP_main", "FP_val"]].max(axis=1)
    eligible["avg_F1"] = eligible[["F1_main", "F1_val"]].mean(axis=1)
    eligible["avg_Precision"] = eligible[["Precision_main", "Precision_val"]].mean(axis=1)
    eligible["avg_Recall"] = eligible[["Recall_main", "Recall_val"]].mean(axis=1)

    eligible = eligible.sort_values(
        ["worst_FN", "worst_FP", "avg_F1", "avg_Precision"],
        ascending=[True, True, False, False],
    )

    chosen = eligible.iloc[0].to_dict()
    chosen["selection_tier"] = selection_tier

    return _safe_json(chosen)


def top_errors(
    sample: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    n: int = 30,
) -> dict[str, Any]:
    y_true = sample["is_fraud"].astype(int).values
    pred = scores >= threshold

    detail_cols = [
        "transaction_id",
        "cd_pix",
        "customer_id",
        "cd_cpf_pagador",
        "vl_pix",
        "nr_idade",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "is_fraud",
    ]
    detail_cols = [c for c in detail_cols if c in sample.columns]

    detail = sample[detail_cols].copy()
    detail["lgbm_score_candidate"] = scores
    detail["pred_candidate"] = pred.astype(int)

    fn_mask = (y_true == 1) & (~pred)
    fp_mask = (y_true == 0) & pred

    return {
        "top_fn": detail.loc[fn_mask].sort_values(
            "lgbm_score_candidate",
            ascending=False,
        ).head(n).to_dict(orient="records"),
        "top_fp": detail.loc[fp_mask].sort_values(
            "lgbm_score_candidate",
            ascending=False,
        ).head(n).to_dict(orient="records"),
    }


# =========================================================
# PATCH CANDIDATO
# =========================================================

def build_scoring_config_patch(best: dict[str, Any]) -> dict[str, Any]:
    th = float(best["lgbm_threshold"])

    return {
        "experiment_id": "EXP-005B",
        "candidate_from": "EXP-005A LGBM_C_SPW_2_0X",
        "status": "candidate_patch_not_runtime_deployed",
        "lgbm_candidate": {
            "model_path": str(CANDIDATE_DIR / "lgbm_v6_2_recall_candidate.joblib"),
            "features_path": str(CANDIDATE_DIR / "lgbm_features_v6_2.json"),
            "model_only_threshold_selected": th,
            "selection_tier": best.get("selection_tier"),
        },
        "suggested_engine_grid_for_e2e": {
            "threshold_confirmar": [60, 62, 65],
            "threshold_bloquear": [95],
            "lgbm_effective_threshold": sorted(set([
                round(max(th, 0.05), 6),
                0.07,
                0.10,
                0.15,
                0.20,
            ])),
            "lgbm_guard_threshold": [0.10, 0.20, 0.30],
            "lgbm_guard_enabled": [True],
            "guard_exception_alto_valor_se_beh_enabled": [True],
        },
        "recommended_first_e2e_config": {
            "threshold_confirmar": 62.0,
            "threshold_bloquear": 95.0,
            "lgbm_effective_threshold": th,
            "lgbm_guard_enabled": True,
            "lgbm_guard_threshold": 0.20,
            "guard_exception_alto_valor_se_beh_enabled": True,
        },
        "baseline_fase2_reference": BASELINE_FASE2,
        "important_note": (
            "Este patch nao deve ser copiado diretamente para producao. "
            "Ele orienta o EXP-005B E2E apos corrigir scoring_config.json."
        ),
    }


# =========================================================
# REPORT
# =========================================================

def write_conclusion(
    path: Path,
    best: dict[str, Any],
    preflight: dict[str, Any],
    grid_main: pd.DataFrame,
    grid_val: pd.DataFrame,
) -> None:
    th = float(best["lgbm_threshold"])

    main_row = grid_main.loc[grid_main["lgbm_threshold"].eq(th)].iloc[0].to_dict()
    val_row = grid_val.loc[grid_val["lgbm_threshold"].eq(th)].iloc[0].to_dict()

    runtime_ready = preflight["runtime_ready_for_e2e"]

    lines = [
        "# EXP-005B — Recalibracao pos-LGBM v6.2",
        "",
        f"- Status: `{'PRONTO_PARA_E2E' if runtime_ready else 'MODEL_ONLY_CONCLUIDO_RUNTIME_NAO_APTO'}`",
        f"- Threshold selecionado: `{th}`",
        f"- Selection tier: `{best.get('selection_tier')}`",
        "",
        "## Resultado seed 42",
        "",
        f"- TP: `{main_row['TP']}`",
        f"- FP: `{main_row['FP']}`",
        f"- FN: `{main_row['FN']}`",
        f"- Precision: `{main_row['Precision']}`",
        f"- Recall: `{main_row['Recall']}`",
        f"- F1: `{main_row['F1']}`",
        f"- FPR: `{main_row['FPR']}`",
        "",
        "## Resultado seed 123",
        "",
        f"- TP: `{val_row['TP']}`",
        f"- FP: `{val_row['FP']}`",
        f"- FN: `{val_row['FN']}`",
        f"- Precision: `{val_row['Precision']}`",
        f"- Recall: `{val_row['Recall']}`",
        f"- F1: `{val_row['F1']}`",
        f"- FPR: `{val_row['FPR']}`",
        "",
        "## Baseline FASE 2",
        "",
        f"- TP={BASELINE_FASE2['TP']}, FP={BASELINE_FASE2['FP']}, FN={BASELINE_FASE2['FN']}",
        f"- Precision={BASELINE_FASE2['Precision']:.4%}, Recall={BASELINE_FASE2['Recall']:.4%}, F1={BASELINE_FASE2['F1']:.4f}",
        "",
        "## Preflight runtime",
        "",
        f"- Runtime pronto para E2E: `{runtime_ready}`",
        f"- scoring_config valido: `{preflight['checks']['scoring_config']['valid_json']}`",
        "",
        "## Conclusao",
        "",
    ]

    if runtime_ready:
        lines.extend([
            "O EXP-005B model-only encontrou um threshold candidato e o runtime aparenta estar apto para E2E.",
            "Proximo passo: rodar avaliacao E2E com o patch candidato e grid de engine.",
        ])
    else:
        lines.extend([
            "O EXP-005B model-only encontrou um threshold candidato, mas o runtime ainda nao esta apto para E2E.",
            "Antes da avaliacao real, corrija `backend/artefatos/scoring_config.json`.",
            "O erro anterior apontou JSON invalido na linha 128; este problema precisa ser resolvido antes de chamar PipelineOrquestrador.",
        ])

    lines.extend([
        "",
        "## Observacao",
        "",
        "Este experimento ainda e model-only. A decisao de promover depende de E2E com DecisionEngine real.",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-005B — Recalibracao pos-LGBM v6.2")
    parser.add_argument("--sample", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=123)
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.03,0.04,0.05,0.0524338379,0.06,0.07,0.08,0.09,0.10,0.12,0.15,0.20,0.30",
        help="Lista separada por virgula de thresholds LGBM.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()

    print_section("EXP-005B — Recalibracao pos-LGBM v6.2")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Output dir: %s", OUTPUT_DIR)
    logger.info("Candidate dir: %s", CANDIDATE_DIR)

    print_section("0. Preflight runtime")

    preflight = runtime_preflight()
    safe_json_dump(preflight, OUTPUT_DIR / "00_preflight_runtime.json")

    if not preflight["runtime_ready_for_e2e"]:
        logger.warning("Runtime ainda nao esta apto para E2E. EXP-005B seguira em modo model-only.")
        logger.warning("Detalhes salvos em 00_preflight_runtime.json")

    print_section("1. Carregar dataset e candidato")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {DATASET_PATH}")

    model_path = CANDIDATE_DIR / "lgbm_v6_2_recall_candidate.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo candidato nao encontrado: {model_path}")

    df = pd.read_csv(DATASET_PATH)
    df["is_fraud"] = _prepare_label(df)
    df = add_exp005a_features(df)

    model = joblib.load(model_path)
    features = load_candidate_features()

    logger.info("Dataset: %d linhas | fraudes=%d", len(df), int(df["is_fraud"].sum()))
    logger.info("Modelo candidato: %s", model_path)
    logger.info("Features candidato: %d", len(features))

    thresholds = sorted(set(float(x.strip()) for x in args.thresholds.split(",") if x.strip()))

    print_section("2. Avaliar grid model-only seed principal")

    sample_main = stratified_eval_sample(df, n=args.sample, seed=args.seed)
    x_main = ensure_numeric_matrix(sample_main, features)
    scores_main = model.predict_proba(x_main)[:, 1]

    grid_main = evaluate_thresholds(
        sample=sample_main,
        scores=scores_main,
        thresholds=thresholds,
        seed=args.seed,
    )

    print_section("3. Validar grid model-only seed independente")

    sample_val = stratified_eval_sample(df, n=args.sample, seed=args.validation_seed)
    x_val = ensure_numeric_matrix(sample_val, features)
    scores_val = model.predict_proba(x_val)[:, 1]

    grid_val = evaluate_thresholds(
        sample=sample_val,
        scores=scores_val,
        thresholds=thresholds,
        seed=args.validation_seed,
    )

    grid_all = pd.concat([grid_main, grid_val], ignore_index=True)
    grid_path = OUTPUT_DIR / "01_grid_thresholds_model_only.csv"
    grid_all.to_csv(grid_path, index=False, encoding="utf-8-sig")

    logger.info("Grid salvo: %s", grid_path)

    print_section("4. Selecionar melhor configuracao")

    best = select_best_config(grid_main, grid_val)

    top_configs = {
        "best_config": best,
        "grid_main": grid_main.to_dict(orient="records"),
        "grid_validation": grid_val.to_dict(orient="records"),
        "thresholds_tested": thresholds,
    }

    safe_json_dump(top_configs, OUTPUT_DIR / "02_top_configs.json")

    best_threshold = float(best["lgbm_threshold"])
    logger.info("Melhor threshold selecionado: %.10f | tier=%s", best_threshold, best.get("selection_tier"))

    print_section("5. Detalhar deltas e erros")

    errors_main = top_errors(sample_main, scores_main, best_threshold)
    errors_val = top_errors(sample_val, scores_val, best_threshold)

    delta_payload = {
        "best_threshold": best_threshold,
        "selection_tier": best.get("selection_tier"),
        "baseline_fase2": BASELINE_FASE2,
        "main_seed": {
            "seed": args.seed,
            "metrics": grid_main.loc[grid_main["lgbm_threshold"].eq(best_threshold)].iloc[0].to_dict(),
            "errors": errors_main,
        },
        "validation_seed": {
            "seed": args.validation_seed,
            "metrics": grid_val.loc[grid_val["lgbm_threshold"].eq(best_threshold)].iloc[0].to_dict(),
            "errors": errors_val,
        },
    }

    safe_json_dump(delta_payload, OUTPUT_DIR / "03_delta_fp_fn_melhor_config.json")

    validation_payload = {
        "status": "MODEL_ONLY_NOT_PIPELINE_E2E",
        "seed": args.validation_seed,
        "best_threshold": best_threshold,
        "metrics": grid_val.loc[grid_val["lgbm_threshold"].eq(best_threshold)].iloc[0].to_dict(),
        "errors": errors_val,
    }

    safe_json_dump(validation_payload, OUTPUT_DIR / "04_validacao_cruzada_model_only.json")

    print_section("6. Gerar patch candidato")

    patch = build_scoring_config_patch(best)
    safe_json_dump(patch, OUTPUT_DIR / "scoring_config_candidate_patch.json")

    print_section("7. Conclusao executiva")

    write_conclusion(
        path=OUTPUT_DIR / "05_conclusao_executiva.md",
        best=best,
        preflight=preflight,
        grid_main=grid_main,
        grid_val=grid_val,
    )

    main_metrics = grid_main.loc[grid_main["lgbm_threshold"].eq(best_threshold)].iloc[0]
    val_metrics = grid_val.loc[grid_val["lgbm_threshold"].eq(best_threshold)].iloc[0]

    logger.info("============================================================")
    logger.info("EXP-005B concluido em modo model-only")
    logger.info("Threshold selecionado: %.10f", best_threshold)
    logger.info(
        "Seed %s: TP=%d FP=%d FN=%d Precision=%.4f Recall=%.4f F1=%.4f",
        args.seed,
        int(main_metrics["TP"]),
        int(main_metrics["FP"]),
        int(main_metrics["FN"]),
        float(main_metrics["Precision"]),
        float(main_metrics["Recall"]),
        float(main_metrics["F1"]),
    )
    logger.info(
        "Seed %s: TP=%d FP=%d FN=%d Precision=%.4f Recall=%.4f F1=%.4f",
        args.validation_seed,
        int(val_metrics["TP"]),
        int(val_metrics["FP"]),
        int(val_metrics["FN"]),
        float(val_metrics["Precision"]),
        float(val_metrics["Recall"]),
        float(val_metrics["F1"]),
    )
    logger.info("Runtime ready for E2E: %s", preflight["runtime_ready_for_e2e"])
    logger.info("Artefatos em: %s", OUTPUT_DIR)
    logger.info("Tempo total: %.1fs", time.perf_counter() - t0)
    logger.info("============================================================")


if __name__ == "__main__":
    main()