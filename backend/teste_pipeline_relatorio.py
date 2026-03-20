"""
teste_pipeline_relatorio.py — Teste Completo do Pipeline v2.0 + Relatório Executivo
=====================================================================================

Executa o pipeline híbrido otimizado nos dados de teste e gera:
  - Dashboard visual (PNG)
  - Relatório executivo (HTML) — para apresentar a diretores
  - Métricas detalhadas (JSON)
  - Resultados por transação (CSV)

Pipeline v2.0:
  LGBM Raw → IF (1ª tx) → Ensemble Raw → Mapeamento Híbrido (0-100) → Decisão
  🟢 APROVAR [0-60) | 🟡 CONFIRMAR [60-85) | 🔴 BLOQUEAR [85-100]

Artefatos necessários (em backend/artefatos/):
  - model_lightgbm.joblib (raw)
  - model_isolation_forest.joblib
  - preprocessing.joblib (PixPreprocessor)
  - scaler_isolation_forest.joblib
  - isolation_forest_config.json
  - if_ref_raw_train.npy
  - lgbm_features.json
  - scoring_config.json (mapeamento híbrido)
  - X_test.csv / y_test.csv

Saída (em relatorio/):
  - relatorio_executivo.html
  - dashboard_executivo.png
  - relatorio_metricas.json
  - resultados_detalhados.csv

Uso:
  python teste_pipeline_relatorio.py

Autor: Equipe Anomalia PIX
Data: Março 2026
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

warnings.filterwarnings("ignore")

# =========================================================
# PATHS
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "backend").exists() and (SCRIPT_DIR / "dados").exists():
    PROJECT_ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "backend").exists():
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

ARTEFATOS_DIR = PROJECT_ROOT / "backend" / "artefatos"
MODELOS_DIR = PROJECT_ROOT / "backend" / "modelos"
BACKEND_DIR = PROJECT_ROOT / "backend"
RELATORIO_DIR = PROJECT_ROOT / "relatorio"
RELATORIO_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(MODELOS_DIR))

try:
    from backend.core.preprocessing import PixPreprocessor
    print(f"✅ PixPreprocessor importado de {BACKEND_DIR / 'preprocessing.py'}")
except ImportError:
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from backend.core.preprocessing import PixPreprocessor
        print(f"✅ PixPreprocessor importado via backend.preprocessing")
    except ImportError as e:
        print(f"❌ Não foi possível importar PixPreprocessor: {e}")
        sys.exit(1)

# Artefatos
LGBM_RAW_PATH = ARTEFATOS_DIR / "model_lightgbm.joblib"
IF_MODEL_PATH = ARTEFATOS_DIR / "model_isolation_forest.joblib"
PREPROCESSOR_PATH = ARTEFATOS_DIR / "preprocessing.joblib"
IF_SCALER_PATH = ARTEFATOS_DIR / "scaler_isolation_forest.joblib"
IF_CONFIG_PATH = ARTEFATOS_DIR / "isolation_forest_config.json"
IF_REF_SCORES_PATH = ARTEFATOS_DIR / "if_ref_raw_train.npy"
LGBM_FEATURES_PATH = ARTEFATOS_DIR / "lgbm_features.json"
SCORING_CONFIG_PATH = ARTEFATOS_DIR / "scoring_config.json"
METRICAS_LGBM_PATH = ARTEFATOS_DIR / "metricas_lightgbm.json"
X_TEST_PATH = ARTEFATOS_DIR / "X_test.csv"
Y_TEST_PATH = ARTEFATOS_DIR / "y_test.csv"

# =========================================================
# CONFIGURAÇÃO DO PIPELINE v2.0
# =========================================================
# Faixas de decisão (score mapeado 0-100)
FAIXA_CONFIRMAR = 60.0
FAIXA_BLOQUEAR = 85.0

# Ensemble: LGBM Raw + IF
W_LGBM_WITH_IF = 0.75
W_IF = 0.25
IF_LGBM_RAW_LOW = 0.05
IF_LGBM_RAW_HIGH = 0.50

# =========================================================
# ESTILO DOS GRÁFICOS
# =========================================================
plt.rcParams.update(
    {
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#1a1d23",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#ffffff",
        "text.color": "#ffffff",
        "xtick.color": "#cccccc",
        "ytick.color": "#cccccc",
        "grid.color": "#333333",
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    }
)

COLORS = {
    "primary": "#00d4aa",
    "secondary": "#ff6b6b",
    "accent": "#4ecdc4",
    "warning": "#ffd93d",
    "info": "#6c5ce7",
    "bg_card": "#1a1d23",
    "text": "#ffffff",
    "text_muted": "#888888",
    "aprovar": "#00d4aa",
    "confirmar": "#ffd93d",
    "bloquear": "#ff6b6b",
}


# =========================================================
# 1. CARREGAR ARTEFATOS
# =========================================================
def load_artifacts() -> Dict[str, Any]:
    """Carrega todos os artefatos necessários."""
    print("\n" + "=" * 70)
    print("  CARREGAMENTO DOS ARTEFATOS")
    print("=" * 70)

    artifacts = {}

    # --- Preprocessor ---
    print(f"\n  Preprocessor: {PREPROCESSOR_PATH.name}...")
    artifacts["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
    print(f"    ✅ Tipo: {type(artifacts['preprocessor']).__name__}")
    print(f"    Colunas modelo: {len(artifacts['preprocessor'].model_columns_)}")

    # --- LightGBM Raw (base do pipeline v2.0) ---
    print(f"\n  LightGBM Raw: {LGBM_RAW_PATH.name}...")
    if not LGBM_RAW_PATH.exists():
        print(f"    ❌ Modelo LGBM Raw não encontrado!")
        sys.exit(1)
    artifacts["lgbm_raw"] = joblib.load(LGBM_RAW_PATH)
    print(f"    ✅ Tipo: {type(artifacts['lgbm_raw']).__name__}")

    # --- LGBM Features ---
    print(f"\n  Features: {LGBM_FEATURES_PATH.name}...")
    with open(LGBM_FEATURES_PATH, "r") as f:
        artifacts["lgbm_features"] = json.load(f)
    print(f"    ✅ {len(artifacts['lgbm_features'])} features")

    # --- Scoring Config (mapeamento híbrido) ---
    print(f"\n  Scoring Config: {SCORING_CONFIG_PATH.name}...")
    if SCORING_CONFIG_PATH.exists():
        with open(SCORING_CONFIG_PATH, "r", encoding="utf-8") as f:
            artifacts["scoring_config"] = json.load(f)
        mapeamento = artifacts["scoring_config"].get("mapeamento", {})
        artifacts["anchors_raw"] = np.array(mapeamento.get("anchors_raw", [0.0, 1.0]), dtype=np.float64)
        artifacts["anchors_out"] = np.array(mapeamento.get("anchors_out", [0.0, 100.0]), dtype=np.float64)
        metricas_scoring = artifacts["scoring_config"].get("metricas_teste", {})
        print(f"    ✅ {len(artifacts['anchors_raw'])} âncoras")
        print(f"    GAP: +{metricas_scoring.get('gap_fraud_min_vs_normal_p999', 'N/A')}")
        print(f"    Recall validação: {metricas_scoring.get('recall_bloquear', 'N/A')}")
    else:
        print(f"    ⚠️  Não encontrado — usando mapeamento linear")
        artifacts["scoring_config"] = {}
        artifacts["anchors_raw"] = np.array([0.0, 1.0])
        artifacts["anchors_out"] = np.array([0.0, 100.0])

    # --- Isolation Forest ---
    if IF_MODEL_PATH.exists():
        print(f"\n  Isolation Forest: {IF_MODEL_PATH.name}...")
        artifacts["if_model"] = joblib.load(IF_MODEL_PATH)
        print(f"    ✅ {artifacts['if_model'].n_estimators} trees")
    else:
        artifacts["if_model"] = None
        print(f"\n  ⚠️  Isolation Forest não encontrado")

    # --- IF Scaler ---
    if IF_SCALER_PATH.exists():
        artifacts["if_scaler"] = joblib.load(IF_SCALER_PATH)
        print(f"  IF Scaler: ✅ {type(artifacts['if_scaler']).__name__}")
    else:
        artifacts["if_scaler"] = None

    # --- IF Config ---
    if IF_CONFIG_PATH.exists():
        with open(IF_CONFIG_PATH, "r") as f:
            artifacts["if_config"] = json.load(f)
        print(f"  IF Config: ✅ (threshold={artifacts['if_config'].get('best_threshold', 'N/A')})")
    else:
        artifacts["if_config"] = None

    # --- IF Reference Scores ---
    if IF_REF_SCORES_PATH.exists():
        artifacts["if_ref_scores"] = np.load(IF_REF_SCORES_PATH)
        print(f"  IF Ref Scores: ✅ ({len(artifacts['if_ref_scores'])} scores)")
    else:
        artifacts["if_ref_scores"] = None

    # --- Métricas de treino ---
    if METRICAS_LGBM_PATH.exists():
        with open(METRICAS_LGBM_PATH, "r") as f:
            artifacts["metricas_treino"] = json.load(f)
        print(f"  Métricas treino: ✅")
    else:
        artifacts["metricas_treino"] = {}

    return artifacts


# =========================================================
# 2. CARREGAR DADOS DE TESTE
# =========================================================
def load_test_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Carrega X_test e y_test."""
    print("\n" + "=" * 70)
    print("  CARREGAMENTO DOS DADOS DE TESTE")
    print("=" * 70)

    if not X_TEST_PATH.exists() or not Y_TEST_PATH.exists():
        print(f"  ❌ Dados de teste não encontrados!")
        sys.exit(1)

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    print(f"  Fraudes: {y_test.sum()} ({y_test.mean() * 100:.2f}%)")
    print(f"  Normais: {(y_test == 0).sum()} ({(y_test == 0).mean() * 100:.2f}%)")

    return X_test, y_test


# =========================================================
# 3. EXECUTAR PIPELINE v2.0
# =========================================================
def run_pipeline(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    artifacts: Dict[str, Any],
) -> pd.DataFrame:
    """Executa o pipeline v2.0 (híbrido otimizado) no dataset de teste."""

    print("\n" + "=" * 70)
    print("  EXECUÇÃO DO PIPELINE v2.0 — Score Híbrido Otimizado")
    print("=" * 70)

    lgbm_raw = artifacts["lgbm_raw"]
    lgbm_features = artifacts["lgbm_features"]
    anchors_raw = artifacts["anchors_raw"]
    anchors_out = artifacts["anchors_out"]
    if_model = artifacts.get("if_model")
    if_scaler = artifacts.get("if_scaler")
    if_config = artifacts.get("if_config")
    if_ref_scores = artifacts.get("if_ref_scores")

    n_total = len(X_test)
    t_start = time.time()

    # ─── PASSO 1: Preparar features ───
    print(f"\n  [1/5] Preparando features ({len(lgbm_features)} features)...")

    missing_feats = [f for f in lgbm_features if f not in X_test.columns]
    if missing_feats:
        print(f"    ⚠️  {len(missing_feats)} features faltando — preenchendo com 0")
        for f in missing_feats:
            X_test[f] = 0

    X_lgbm = X_test[lgbm_features].copy()
    print(f"    ✅ X_lgbm: {X_lgbm.shape}")

    # ─── PASSO 2: Score LGBM Raw ───
    print(f"  [2/5] Calculando scores LGBM Raw...")

    lgbm_raw_proba = lgbm_raw.predict_proba(X_lgbm)[:, 1]
    print(f"    ✅ Raw scores: min={lgbm_raw_proba.min():.6f}, "
          f"max={lgbm_raw_proba.max():.6f}, median={np.median(lgbm_raw_proba):.6f}")

    # ─── PASSO 3: Isolation Forest ───
    print(f"  [3/5] Calculando scores Isolation Forest...")

    if_scores = np.zeros(n_total)
    if_active = np.zeros(n_total, dtype=bool)
    if_percentiles = np.zeros(n_total)

    has_if = (if_model is not None and if_config is not None)

    if has_if:
        if_features_list = if_config.get("features", [])
        if_medians = if_config.get("medians", {})

        if "is_first_tx_trimestre" in X_test.columns:
            first_tx_mask = X_test["is_first_tx_trimestre"] == 1
        else:
            first_tx_mask = pd.Series(False, index=X_test.index)

        n_first = first_tx_mask.sum()
        print(f"    Primeiras tx: {n_first:,} ({n_first / n_total * 100:.1f}%)")

        if n_first > 0 and if_features_list:
            X_if_data = pd.DataFrame(index=X_test[first_tx_mask].index)
            for feat in if_features_list:
                if feat in X_test.columns:
                    X_if_data[feat] = X_test.loc[first_tx_mask, feat].values
                else:
                    X_if_data[feat] = if_medians.get(feat, 0)

            for feat in if_features_list:
                median_val = if_medians.get(feat, 0)
                X_if_data[feat] = X_if_data[feat].fillna(median_val)

            if if_scaler is not None:
                X_if_scaled = if_scaler.transform(X_if_data[if_features_list])
            else:
                X_if_scaled = X_if_data[if_features_list].values

            raw_scores = if_model.decision_function(X_if_scaled)

            if if_ref_scores is not None and len(if_ref_scores) > 0:
                percentiles = np.array(
                    [np.mean(if_ref_scores <= s) for s in raw_scores]
                )
            else:
                percentiles = 1.0 / (1.0 + np.exp(raw_scores * 5))

            percentiles = np.clip(percentiles, 0, 1)

            first_idx = np.where(first_tx_mask.values)[0]
            if_scores[first_idx] = percentiles
            if_percentiles[first_idx] = percentiles

            # IF ativo apenas na zona cinzenta do LGBM raw
            lgbm_first = lgbm_raw_proba[first_idx]
            if_zone = (lgbm_first >= IF_LGBM_RAW_LOW) & (lgbm_first <= IF_LGBM_RAW_HIGH)
            active_idx = first_idx[if_zone]
            if_active[active_idx] = True

            print(f"    ✅ IF calculado para {n_first:,} tx")
            print(f"    IF ativo (zona cinzenta raw {IF_LGBM_RAW_LOW}-{IF_LGBM_RAW_HIGH}): "
                  f"{if_active.sum():,} tx")
    else:
        print(f"    ⚠️  IF desabilitado")

    # ─── PASSO 4: Ensemble Raw ───
    print(f"  [4/5] Calculando ensemble raw...")

    ensemble_raw = lgbm_raw_proba.copy()
    if has_if and if_active.any():
        ensemble_raw[if_active] = (
            W_LGBM_WITH_IF * lgbm_raw_proba[if_active]
            + W_IF * if_scores[if_active]
        )
    ensemble_raw = np.clip(ensemble_raw, 0.0, 1.0)

    # ─── PASSO 5: Mapeamento Híbrido → Score 0-100 → Decisão ───
    print(f"  [5/5] Mapeamento híbrido → Score 0-100 → Decisões...")

    scores_mapped = np.clip(
        np.interp(ensemble_raw, anchors_raw, anchors_out),
        0.0, 100.0,
    )

    # Decisões
    decisions = np.full(n_total, "APROVAR", dtype=object)
    decisions[scores_mapped >= FAIXA_CONFIRMAR] = "CONFIRMAR"
    decisions[scores_mapped >= FAIXA_BLOQUEAR] = "BLOQUEAR"

    # Rule scores
    rule_score_raw = X_test["rule_score_raw"].values if "rule_score_raw" in X_test.columns else np.zeros(n_total)
    rule_score_norm = X_test["rule_score_normalized"].values if "rule_score_normalized" in X_test.columns else np.zeros(n_total)

    # Montar resultados
    results = pd.DataFrame(
        {
            "y_true": y_test.values,
            "lgbm_raw_score": lgbm_raw_proba,
            "if_score": if_scores,
            "if_percentile": if_percentiles,
            "if_active": if_active,
            "is_first_tx": X_test["is_first_tx_trimestre"].values if "is_first_tx_trimestre" in X_test.columns else 0,
            "ensemble_raw": ensemble_raw,
            "score_mapped": np.round(scores_mapped, 2),
            "rule_score_raw": rule_score_raw,
            "rule_score_normalized": rule_score_norm,
            "decision": decisions,
            "w_lgbm": np.where(if_active, W_LGBM_WITH_IF, 1.0),
            "w_if": np.where(if_active, W_IF, 0.0),
        }
    )

    elapsed = time.time() - t_start
    print(f"\n  ✅ Pipeline completo: {n_total:,} tx em {elapsed:.1f}s "
          f"({n_total / elapsed:,.0f} tx/s)")

    # Resumo de decisões
    print(f"\n  ┌───────────────┬──────────┬─────────┬──────────┬───────────┐")
    print(f"  │   Decisão     │   Total  │    %    │ Fraudes  │ Taxa Fr.  │")
    print(f"  ├───────────────┼──────────┼─────────┼──────────┼───────────┤")
    for dec in ["APROVAR", "CONFIRMAR", "BLOQUEAR"]:
        mask = results["decision"] == dec
        count = mask.sum()
        fraud_in = results.loc[mask, "y_true"].sum()
        pct = count / n_total * 100
        fraud_rate = fraud_in / count * 100 if count > 0 else 0
        icon = {"APROVAR": "🟢", "CONFIRMAR": "🟡", "BLOQUEAR": "🔴"}[dec]
        print(f"  │ {icon} {dec:10s} │ {count:6,}  │ {pct:5.1f}%  │  {fraud_in:5.0f}   │ {fraud_rate:7.2f}%  │")
    print(f"  └───────────────┴──────────┴─────────┴──────────┴───────────┘")

    # Score stats
    print(f"\n  Scores mapeados (0-100):")
    print(f"    Normais:  min={scores_mapped[y_test == 0].min():.1f}, "
          f"P99.9={np.percentile(scores_mapped[y_test == 0], 99.9):.1f}, "
          f"max={scores_mapped[y_test == 0].max():.1f}")
    print(f"    Fraudes:  min={scores_mapped[y_test == 1].min():.1f}, "
          f"P5={np.percentile(scores_mapped[y_test == 1], 5):.1f}, "
          f"median={np.median(scores_mapped[y_test == 1]):.1f}")
    print(f"    GAP:      +{scores_mapped[y_test == 1].min() - np.percentile(scores_mapped[y_test == 0], 99.9):.1f} pontos")

    return results


# =========================================================
# 4. CALCULAR MÉTRICAS
# =========================================================
def calculate_metrics(df: pd.DataFrame, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula todas as métricas para o relatório."""
    print("\n" + "=" * 70)
    print("  CÁLCULO DAS MÉTRICAS")
    print("=" * 70)

    y_true = df["y_true"].values
    score_mapped = df["score_mapped"].values
    ensemble_raw = df["ensemble_raw"].values
    decisions = df["decision"].values

    n_total = len(y_true)
    n_fraudes = int(y_true.sum())
    n_normais = n_total - n_fraudes

    # --- Predição binária: BLOQUEAR = fraude detectada ---
    y_pred_bloquear = (decisions == "BLOQUEAR").astype(int)

    # --- Predição ampla: CONFIRMAR + BLOQUEAR = qualquer ação ---
    y_pred_qualquer_acao = np.isin(decisions, ["CONFIRMAR", "BLOQUEAR"]).astype(int)

    metrics = {
        "data_geracao": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "pipeline_version": "2.0",
        "scoring": "Mapeamento Híbrido Otimizado",
        "n_total": n_total,
        "n_fraudes": n_fraudes,
        "n_normais": n_normais,
        "taxa_fraude_pct": round(n_fraudes / n_total * 100, 4),
    }

    # ═══ Métricas: BLOQUEAR = fraude ═══
    cm = confusion_matrix(y_true, y_pred_bloquear)
    tn, fp, fn, tp = cm.ravel()

    metrics["pipeline_bloquear"] = {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": round(accuracy_score(y_true, y_pred_bloquear), 4),
        "precision": round(precision_score(y_true, y_pred_bloquear, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred_bloquear, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred_bloquear, zero_division=0), 4),
        "fpr": round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0,
        "fnr": round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0,
    }

    # ═══ Métricas: qualquer ação ═══
    cm2 = confusion_matrix(y_true, y_pred_qualquer_acao)
    tn2, fp2, fn2, tp2 = cm2.ravel()

    metrics["pipeline_qualquer_acao"] = {
        "tn": int(tn2), "fp": int(fp2), "fn": int(fn2), "tp": int(tp2),
        "accuracy": round(accuracy_score(y_true, y_pred_qualquer_acao), 4),
        "precision": round(precision_score(y_true, y_pred_qualquer_acao, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred_qualquer_acao, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred_qualquer_acao, zero_division=0), 4),
        "fpr": round(fp2 / (fp2 + tn2), 4) if (fp2 + tn2) > 0 else 0,
        "fnr": round(fn2 / (fn2 + tp2), 4) if (fn2 + tp2) > 0 else 0,
    }

    # ═══ AUC e AP (usando score raw para ROC, score mapeado para contexto) ═══
    metrics["auc_roc_ensemble"] = round(roc_auc_score(y_true, ensemble_raw), 4)
    metrics["auc_roc_lgbm_raw"] = round(roc_auc_score(y_true, df["lgbm_raw_score"].values), 4)
    metrics["ap_ensemble"] = round(average_precision_score(y_true, ensemble_raw), 4)
    metrics["ap_lgbm_raw"] = round(average_precision_score(y_true, df["lgbm_raw_score"].values), 4)

    # ═══ Best F1 no score mapeado ═══
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, score_mapped)
    f1_arr = 2 * (prec_arr * rec_arr) / (prec_arr + rec_arr + 1e-10)
    best_idx = np.argmax(f1_arr)
    metrics["best_f1"] = {
        "f1": round(float(f1_arr[best_idx]), 4),
        "threshold_mapped": round(float(thresh_arr[best_idx]) if best_idx < len(thresh_arr) else 85.0, 2),
        "precision": round(float(prec_arr[best_idx]), 4),
        "recall": round(float(rec_arr[best_idx]), 4),
    }

    # ═══ Decisões ═══
    dec_counts = pd.Series(decisions).value_counts()
    metrics["decisoes"] = {
        d: int(dec_counts.get(d, 0)) for d in ["APROVAR", "CONFIRMAR", "BLOQUEAR"]
    }

    n_intervencao = metrics["decisoes"]["CONFIRMAR"] + metrics["decisoes"]["BLOQUEAR"]
    metrics["taxa_intervencao_pct"] = round(n_intervencao / n_total * 100, 2)

    # ═══ Taxa de fraude por decisão ═══
    for dec in ["APROVAR", "CONFIRMAR", "BLOQUEAR"]:
        mask = decisions == dec
        if mask.sum() > 0:
            metrics[f"taxa_fraude_{dec.lower()}"] = round(float(y_true[mask].mean()) * 100, 4)
            metrics[f"n_fraude_{dec.lower()}"] = int(y_true[mask].sum())
        else:
            metrics[f"taxa_fraude_{dec.lower()}"] = 0.0
            metrics[f"n_fraude_{dec.lower()}"] = 0

    # ═══ Separação de scores ═══
    scores_fraude = score_mapped[y_true == 1]
    scores_normal = score_mapped[y_true == 0]
    metrics["separacao"] = {
        "fraud_min": round(float(scores_fraude.min()), 2) if len(scores_fraude) > 0 else None,
        "fraud_p5": round(float(np.percentile(scores_fraude, 5)), 2) if len(scores_fraude) > 0 else None,
        "fraud_median": round(float(np.median(scores_fraude)), 2) if len(scores_fraude) > 0 else None,
        "normal_p999": round(float(np.percentile(scores_normal, 99.9)), 2) if len(scores_normal) > 0 else None,
        "normal_max": round(float(scores_normal.max()), 2) if len(scores_normal) > 0 else None,
        "gap": round(float(scores_fraude.min() - np.percentile(scores_normal, 99.9)), 2) if len(scores_fraude) > 0 else None,
    }

    # ═══ IF stats ═══
    metrics["isolation_forest"] = {
        "first_tx_total": int(df["is_first_tx"].sum()),
        "if_active": int(df["if_active"].sum()),
        "if_active_pct": round(df["if_active"].mean() * 100, 2),
    }

    # ═══ Resumo executivo ═══
    p = metrics["pipeline_bloquear"]
    metrics["executivo"] = {
        "fraudes_detectadas_pct": round(p["recall"] * 100, 1),
        "fraudes_nao_detectadas_pct": round(p["fnr"] * 100, 1),
        "fraudes_nao_detectadas_n": p["fn"],
        "falsos_alarmes_pct": round(p["fpr"] * 100, 2),
        "falsos_alarmes_n": p["fp"],
        "precisao_alarmes_pct": round(p["precision"] * 100, 1),
        "taxa_intervencao_pct": metrics["taxa_intervencao_pct"],
        "auc_roc": metrics["auc_roc_ensemble"],
        "f1": p["f1"],
        "gap_separacao": metrics["separacao"]["gap"],
    }

    # Print resumo
    e = metrics["executivo"]
    sep = metrics["separacao"]
    print(f"\n  ═══ RESUMO ═══")
    print(f"  Fraudes detectadas (BLOQUEAR):  {e['fraudes_detectadas_pct']:.1f}%  ({p['tp']}/{n_fraudes})")
    print(f"  Fraudes perdidas:               {e['fraudes_nao_detectadas_pct']:.1f}%  ({p['fn']})")
    print(f"  Falsos alarmes (FPR):           {e['falsos_alarmes_pct']:.2f}%  ({p['fp']})")
    print(f"  Precisão dos alarmes:           {e['precisao_alarmes_pct']:.1f}%")
    print(f"  AUC-ROC:                        {e['auc_roc']:.4f}")
    print(f"  F1-Score (BLOQUEAR):            {e['f1']:.4f}")
    print(f"  GAP (fraud min - normal P99.9): +{sep['gap']:.1f} pontos")

    return metrics


# =========================================================
# 5. GRÁFICOS
# =========================================================
def plot_dashboard(df: pd.DataFrame, metrics: Dict[str, Any]) -> None:
    """Gera dashboard executivo v2.0."""
    print("\n" + "=" * 70)
    print("  GERANDO DASHBOARD")
    print("=" * 70)

    y_true = df["y_true"].values
    score_mapped = df["score_mapped"].values
    ensemble_raw = df["ensemble_raw"].values

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(
        "Relatório Executivo — Detecção de Fraude PIX | Pipeline v2.0\n"
        f"Score Híbrido Otimizado (0-100) | "
        f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')} | "
        f"{metrics['n_total']:,} transações | {metrics['n_fraudes']} fraudes",
        fontsize=16, fontweight="bold", color=COLORS["primary"], y=1.02,
    )

    # ─── 1. Matriz de Confusão (BLOQUEAR = fraude) ───
    ax = axes[0, 0]
    p = metrics["pipeline_bloquear"]
    cm = np.array([[p["tn"], p["fp"]], [p["fn"], p["tp"]]])
    cm_pct = cm / cm.sum() * 100

    ax.imshow(cm, cmap="RdYlGn_r", alpha=0.8)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal\n(predito)", "Fraude\n(predito)"])
    ax.set_yticklabels(["Normal\n(real)", "Fraude\n(real)"])
    ax.set_title("Matriz de Confusão\n(BLOQUEAR ≥85 = Fraude)", fontweight="bold", pad=15)

    labels = [
        [f"VN\n{p['tn']:,}\n({cm_pct[0, 0]:.2f}%)", f"FP\n{p['fp']:,}\n({cm_pct[0, 1]:.2f}%)"],
        [f"FN\n{p['fn']:,}\n({cm_pct[1, 0]:.2f}%)", f"VP\n{p['tp']:,}\n({cm_pct[1, 1]:.2f}%)"],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j], ha="center", va="center",
                    fontsize=12, fontweight="bold", color="white")

    # ─── 2. Curva ROC ───
    ax = axes[0, 1]
    fpr_ens, tpr_ens, _ = roc_curve(y_true, ensemble_raw)
    fpr_lgbm, tpr_lgbm, _ = roc_curve(y_true, df["lgbm_raw_score"].values)

    ax.plot(fpr_ens, tpr_ens, color=COLORS["primary"], lw=2.5,
            label=f"Ensemble (AUC={metrics['auc_roc_ensemble']:.4f})")
    ax.plot(fpr_lgbm, tpr_lgbm, color=COLORS["info"], lw=1.5, ls="--",
            label=f"LGBM Raw (AUC={metrics['auc_roc_lgbm_raw']:.4f})")
    ax.plot([0, 1], [0, 1], color=COLORS["text_muted"], lw=1, ls=":")
    ax.set_xlabel("Taxa de Falso Positivo")
    ax.set_ylabel("Taxa de Verdadeiro Positivo")
    ax.set_title("Curva ROC", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True)

    # ─── 3. Curva Precision-Recall ───
    ax = axes[0, 2]
    prec_ens, rec_ens, _ = precision_recall_curve(y_true, ensemble_raw)
    prec_lgbm, rec_lgbm, _ = precision_recall_curve(y_true, df["lgbm_raw_score"].values)

    ax.plot(rec_ens, prec_ens, color=COLORS["primary"], lw=2.5,
            label=f"Ensemble (AP={metrics['ap_ensemble']:.4f})")
    ax.plot(rec_lgbm, prec_lgbm, color=COLORS["info"], lw=1.5, ls="--",
            label=f"LGBM Raw (AP={metrics['ap_lgbm_raw']:.4f})")
    ax.set_xlabel("Recall (Fraudes Detectadas)")
    ax.set_ylabel("Precision (Precisão dos Alarmes)")
    ax.set_title("Curva Precision-Recall", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True)

    best = metrics["best_f1"]
    ax.scatter([best["recall"]], [best["precision"]], s=150, color=COLORS["warning"],
               zorder=5, marker="*")
    ax.annotate(f"Best F1={best['f1']:.3f}\n@score≥{best['threshold_mapped']:.0f}",
                xy=(best["recall"], best["precision"]),
                fontsize=9, color=COLORS["warning"],
                xytext=(best["recall"] - 0.15, best["precision"] - 0.1),
                arrowprops=dict(arrowstyle="->", color=COLORS["warning"]))

    # ─── 4. Distribuição de Decisões ───
    ax = axes[1, 0]
    dec_order = ["APROVAR", "CONFIRMAR", "BLOQUEAR"]
    dec_colors = [COLORS["aprovar"], COLORS["confirmar"], COLORS["bloquear"]]
    dec_counts = [metrics["decisoes"].get(d, 0) for d in dec_order]

    bars = ax.barh(dec_order, dec_counts, color=dec_colors, edgecolor="white", lw=0.5)
    ax.set_title("Distribuição de Decisões", fontweight="bold")
    ax.set_xlabel("Quantidade")
    for bar, count in zip(bars, dec_counts):
        pct = count / metrics["n_total"] * 100
        fraud_n = metrics.get(f"n_fraude_{dec_order[dec_counts.index(count)].lower()}", 0)
        ax.text(bar.get_width() + max(dec_counts) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({pct:.1f}%) | {fraud_n} fraudes",
                va="center", fontsize=11, color=COLORS["text"])
    ax.set_xlim(0, max(dec_counts) * 1.3)

    # ─── 5. Distribuição dos Scores Mapeados (0-100) ───
    ax = axes[1, 1]
    scores_normal = score_mapped[y_true == 0]
    scores_fraude = score_mapped[y_true == 1]

    ax.hist(scores_normal, bins=100, alpha=0.7, color=COLORS["primary"],
            label=f"Normal (n={len(scores_normal):,})", density=True)
    ax.hist(scores_fraude, bins=30, alpha=0.7, color=COLORS["secondary"],
            label=f"Fraude (n={len(scores_fraude):,})", density=True)

    ax.axvline(FAIXA_CONFIRMAR, color=COLORS["warning"], ls="--", lw=2, alpha=0.8)
    ax.text(FAIXA_CONFIRMAR + 1, ax.get_ylim()[1] * 0.85,
            f"CONFIRMAR\n≥{FAIXA_CONFIRMAR:.0f}", fontsize=9, color=COLORS["warning"])

    ax.axvline(FAIXA_BLOQUEAR, color=COLORS["secondary"], ls="--", lw=2, alpha=0.8)
    ax.text(FAIXA_BLOQUEAR + 1, ax.get_ylim()[1] * 0.7,
            f"BLOQUEAR\n≥{FAIXA_BLOQUEAR:.0f}", fontsize=9, color=COLORS["secondary"])

    ax.set_title("Distribuição dos Scores (0-100)", fontweight="bold")
    ax.set_xlabel("Score Mapeado (0-100)")
    ax.set_ylabel("Densidade")
    ax.legend(fontsize=9)

    # ─── 6. Zona de Separação (zoom) ───
    ax = axes[1, 2]
    sep = metrics["separacao"]

    # Histograma na zona relevante
    normal_high = scores_normal[scores_normal >= 10]
    ax.hist(normal_high, bins=50, alpha=0.6, color=COLORS["primary"],
            label=f"Normais ≥10 (n={len(normal_high):,})")
    ax.hist(scores_fraude, bins=20, alpha=0.6, color=COLORS["secondary"],
            label=f"Fraudes (n={len(scores_fraude):,})")

    ax.axvline(FAIXA_CONFIRMAR, color=COLORS["warning"], ls="--", lw=1.5)
    ax.axvline(FAIXA_BLOQUEAR, color=COLORS["secondary"], ls="--", lw=1.5)

    if sep["normal_p999"] is not None and sep["fraud_min"] is not None:
        ax.annotate(
            f"GAP: +{sep['gap']:.1f}",
            xy=((sep["normal_p999"] + sep["fraud_min"]) / 2, ax.get_ylim()[1] * 0.5),
            fontsize=14, fontweight="bold", color=COLORS["primary"],
            ha="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1d23", edgecolor=COLORS["primary"]),
        )

    ax.set_title("Zona de Separação (Zoom)", fontweight="bold")
    ax.set_xlabel("Score Mapeado (0-100)")
    ax.set_ylabel("Contagem")
    ax.legend(fontsize=9)

    plt.tight_layout()
    output = RELATORIO_DIR / "dashboard_executivo.png"
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ Dashboard salvo: {output}")


# =========================================================
# 6. RELATÓRIO HTML
# =========================================================
def generate_html_report(metrics: Dict[str, Any]) -> None:
    """Gera relatório executivo HTML v2.0."""
    print("\n" + "=" * 70)
    print("  GERANDO RELATÓRIO HTML")
    print("=" * 70)

    e = metrics["executivo"]
    p = metrics["pipeline_bloquear"]
    p2 = metrics["pipeline_qualquer_acao"]
    sep = metrics["separacao"]

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório Executivo — Detecção de Fraude PIX v2.0</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #0e1117; color: #e0e0e0; padding: 30px 50px; line-height: 1.6; }}
        .header {{ text-align: center; margin-bottom: 40px; border-bottom: 3px solid #00d4aa; padding-bottom: 20px; }}
        .header h1 {{ color: #00d4aa; font-size: 32px; margin-bottom: 8px; }}
        .header .subtitle {{ color: #888; font-size: 14px; }}
        .header .version {{ display: inline-block; background: rgba(0, 212, 170, 0.15); color: #00d4aa; padding: 4px 16px; border-radius: 20px; font-size: 13px; font-weight: bold; margin-top: 8px; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 40px; }}
        .kpi {{ background: #1a1d23; border-radius: 16px; padding: 28px; text-align: center; border: 2px solid #333; transition: border-color 0.3s; }}
        .kpi:hover {{ border-color: #00d4aa; }}
        .kpi.green {{ border-color: #00d4aa; }}
        .kpi.red {{ border-color: #ff6b6b; }}
        .kpi.yellow {{ border-color: #ffd93d; }}
        .kpi.blue {{ border-color: #6c5ce7; }}
        .kpi .value {{ font-size: 48px; font-weight: 800; line-height: 1.1; }}
        .kpi .label {{ font-size: 14px; color: #888; margin-top: 8px; text-transform: uppercase; letter-spacing: 1px; }}
        .kpi .detail {{ font-size: 12px; color: #666; margin-top: 6px; }}
        .section {{ background: #1a1d23; border-radius: 16px; padding: 28px; margin-bottom: 24px; border: 1px solid #2a2d33; }}
        .section h2 {{ color: #00d4aa; margin-bottom: 20px; font-size: 20px; padding-bottom: 10px; border-bottom: 1px solid #333; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th {{ color: #00d4aa; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; padding: 12px 16px; text-align: left; border-bottom: 2px solid #333; }}
        td {{ padding: 12px 16px; border-bottom: 1px solid #2a2d33; font-size: 14px; }}
        tr:hover {{ background: rgba(0, 212, 170, 0.05); }}
        .highlight {{ color: #00d4aa; font-weight: bold; }}
        .danger {{ color: #ff6b6b; font-weight: bold; }}
        .warning {{ color: #ffd93d; font-weight: bold; }}
        .info {{ color: #6c5ce7; font-weight: bold; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; }}
        .badge-green {{ background: rgba(0, 212, 170, 0.2); color: #00d4aa; }}
        .badge-red {{ background: rgba(255, 107, 107, 0.2); color: #ff6b6b; }}
        .badge-yellow {{ background: rgba(255, 217, 61, 0.2); color: #ffd93d; }}
        .img-container {{ text-align: center; margin: 20px 0; }}
        .img-container img {{ max-width: 100%; border-radius: 12px; border: 1px solid #333; }}
        .callout {{ background: rgba(0, 212, 170, 0.08); border-left: 4px solid #00d4aa; padding: 16px 20px; margin: 16px 0; border-radius: 0 8px 8px 0; }}
        .callout.warning {{ background: rgba(255, 217, 61, 0.08); border-left-color: #ffd93d; }}
        .callout.danger {{ background: rgba(255, 107, 107, 0.08); border-left-color: #ff6b6b; }}
        .callout.success {{ background: rgba(0, 212, 170, 0.12); border-left-color: #00d4aa; }}
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
        .score-bar {{ height: 32px; border-radius: 16px; background: linear-gradient(to right, #00d4aa 0%, #00d4aa 60%, #ffd93d 60%, #ffd93d 85%, #ff6b6b 85%, #ff6b6b 100%); margin: 16px 0; position: relative; }}
        .score-bar .label-left {{ position: absolute; left: 25%; top: 50%; transform: translate(-50%, -50%); font-size: 12px; font-weight: bold; color: #0e1117; }}
        .score-bar .label-mid {{ position: absolute; left: 72.5%; top: 50%; transform: translate(-50%, -50%); font-size: 12px; font-weight: bold; color: #0e1117; }}
        .score-bar .label-right {{ position: absolute; left: 92.5%; top: 50%; transform: translate(-50%, -50%); font-size: 12px; font-weight: bold; color: #0e1117; }}
        .footer {{ text-align: center; color: #555; font-size: 12px; margin-top: 50px; padding-top: 20px; border-top: 1px solid #333; }}
    </style>
</head>
<body>

<div class="header">
    <h1>🛡️ Sistema de Detecção de Fraude PIX</h1>
    <h2 style="color: #ccc; font-weight: 400; font-size: 18px; margin-top: 4px;">Relatório Executivo de Performance</h2>
    <div class="version">Pipeline v2.0 — Score Híbrido Otimizado (0-100)</div>
    <p class="subtitle" style="margin-top: 12px;">
        Gerado em {metrics['data_geracao']} |
        Base de teste: {metrics['n_total']:,} transações |
        {metrics['n_fraudes']} fraudes ({metrics['taxa_fraude_pct']:.2f}%)
    </p>
</div>

<!-- ═══════════ KPIs ═══════════ -->
<div class="kpi-grid">
    <div class="kpi green">
        <div class="value" style="color: #00d4aa;">{e['fraudes_detectadas_pct']:.1f}%</div>
        <div class="label">Fraudes Detectadas</div>
        <div class="detail">{p['tp']} de {metrics['n_fraudes']} fraudes capturadas (BLOQUEAR ≥85)</div>
    </div>
    <div class="kpi {'green' if e['fraudes_nao_detectadas_n'] == 0 else 'red'}">
        <div class="value" style="color: {'#00d4aa' if e['fraudes_nao_detectadas_n'] == 0 else '#ff6b6b'};">{e['fraudes_nao_detectadas_n']}</div>
        <div class="label">Fraudes Não Detectadas</div>
        <div class="detail">{'Nenhuma fraude escapou do sistema ✅' if e['fraudes_nao_detectadas_n'] == 0 else f"{e['fraudes_nao_detectadas_pct']:.1f}% passaram sem detecção"}</div>
    </div>
    <div class="kpi yellow">
        <div class="value" style="color: #ffd93d;">{e['falsos_alarmes_n']}</div>
        <div class="label">Falsos Alarmes</div>
        <div class="detail">{e['falsos_alarmes_pct']:.2f}% das tx legítimas sinalizadas ({e['falsos_alarmes_n']} revisões extras)</div>
    </div>
    <div class="kpi green">
        <div class="value" style="color: #00d4aa;">{e['precisao_alarmes_pct']:.1f}%</div>
        <div class="label">Precisão dos Alarmes</div>
        <div class="detail">De cada bloqueio, esta % é fraude real</div>
    </div>
    <div class="kpi blue">
        <div class="value" style="color: #6c5ce7;">{e['auc_roc']:.4f}</div>
        <div class="label">AUC-ROC</div>
        <div class="detail">Capacidade geral de separação (1.0 = perfeito)</div>
    </div>
    <div class="kpi green">
        <div class="value" style="color: #00d4aa;">+{sep['gap']:.1f}</div>
        <div class="label">GAP de Separação</div>
        <div class="detail">Pontos entre menor fraude ({sep['fraud_min']}) e P99.9 normal ({sep['normal_p999']})</div>
    </div>
</div>

<!-- ═══════════ Score Visual ═══════════ -->
<div class="section">
    <h2>📏 Escala de Score (0-100)</h2>
    <div class="score-bar">
        <span class="label-left">🟢 APROVAR<br>0 — 59</span>
        <span class="label-mid">🟡 CONFIRMAR<br>60 — 84</span>
        <span class="label-right">🔴 BLOQUEAR<br>85 — 100</span>
    </div>
    <table>
        <tr>
            <th>Faixa</th>
            <th>Score</th>
            <th>Ação</th>
            <th>Descrição</th>
        </tr>
        <tr>
            <td><span class="badge badge-green">🟢 APROVAR</span></td>
            <td>0 a 59</td>
            <td>Liberar automaticamente</td>
            <td>Transação com padrão normal — sem intervenção necessária</td>
        </tr>
        <tr>
            <td><span class="badge badge-yellow">🟡 CONFIRMAR</span></td>
            <td>60 a 84</td>
            <td>Autenticação adicional (2FA / biometria)</td>
            <td>Padrão levemente atípico — confirmar identidade do cliente</td>
        </tr>
        <tr>
            <td><span class="badge badge-red">🔴 BLOQUEAR</span></td>
            <td>85 a 100</td>
            <td>Parar para análise humana</td>
            <td>Alto risco de fraude — bloquear e encaminhar ao analista</td>
        </tr>
    </table>
</div>

<!-- ═══════════ Dashboard Visual ═══════════ -->
<div class="section">
    <h2>📊 Dashboard Visual</h2>
    <div class="img-container">
        <img src="dashboard_executivo.png" alt="Dashboard">
    </div>
</div>

<!-- ═══════════ Como Interpretar ═══════════ -->
<div class="section">
    <h2>📖 Guia de Interpretação para Executivos</h2>

    <div class="callout success">
        <strong>Destaque principal:</strong> O sistema detecta <strong>{e['fraudes_detectadas_pct']:.1f}%</strong>
        das fraudes com apenas <strong>{e['falsos_alarmes_n']}</strong> falsos alarmes em
        {metrics['n_total']:,} transações. {'Nenhuma fraude escapou do modelo.' if e['fraudes_nao_detectadas_n'] == 0 else ''}
    </div>

    <table>
        <tr>
            <th style="width: 25%">Métrica</th>
            <th style="width: 50%">O que significa</th>
            <th style="width: 25%">Resultado</th>
        </tr>
        <tr>
            <td><strong>Fraudes Detectadas</strong><br>(Recall)</td>
            <td>De cada 100 fraudes reais, quantas o sistema identifica e bloqueia</td>
            <td class="highlight" style="font-size: 18px;">{e['fraudes_detectadas_pct']:.1f}%</td>
        </tr>
        <tr>
            <td><strong>Fraudes Não Detectadas</strong><br>(False Negative Rate)</td>
            <td>Fraudes que passam pelo sistema sem serem barradas — <em>risco residual</em></td>
            <td class="{'highlight' if e['fraudes_nao_detectadas_n'] == 0 else 'danger'}" style="font-size: 18px;">{e['fraudes_nao_detectadas_n']} ({e['fraudes_nao_detectadas_pct']:.1f}%)</td>
        </tr>
        <tr>
            <td><strong>Falsos Alarmes</strong><br>(False Positive Rate)</td>
            <td>Transações legítimas erroneamente bloqueadas — <em>impacta experiência do cliente</em></td>
            <td class="warning" style="font-size: 18px;">{e['falsos_alarmes_pct']:.2f}% ({e['falsos_alarmes_n']})</td>
        </tr>
        <tr>
            <td><strong>Precisão dos Alarmes</strong><br>(Precision)</td>
            <td>Quando o sistema bloqueia, qual a chance de ser fraude real — <em>eficiência dos analistas</em></td>
            <td class="highlight" style="font-size: 18px;">{e['precisao_alarmes_pct']:.1f}%</td>
        </tr>
        <tr>
            <td><strong>GAP de Separação</strong></td>
            <td>Distância em pontos entre a fraude com menor score e o percentil 99.9 dos normais. Quanto maior, mais seguro o sistema</td>
            <td class="highlight" style="font-size: 18px;">+{sep['gap']:.1f} pontos</td>
        </tr>
        <tr>
            <td><strong>AUC-ROC</strong></td>
            <td>Nota geral do modelo: 0.50 = aleatório, 1.00 = perfeito. Acima de 0.95 é excelente</td>
            <td class="info" style="font-size: 18px;">{e['auc_roc']:.4f}</td>
        </tr>
    </table>
</div>

<!-- ═══════════ Decisões ═══════════ -->
<div class="section">
    <h2>🎯 Resultados por Faixa de Decisão</h2>

    <table>
        <tr>
            <th>Decisão</th>
            <th>Score</th>
            <th>Ação Operacional</th>
            <th>Quantidade</th>
            <th>% do Total</th>
            <th>Fraudes Reais</th>
            <th>Taxa Fraude</th>
        </tr>
        <tr>
            <td><span class="badge badge-green">🟢 APROVAR</span></td>
            <td>0 — 59</td>
            <td>Transação liberada automaticamente</td>
            <td>{metrics['decisoes']['APROVAR']:,}</td>
            <td>{metrics['decisoes']['APROVAR'] / metrics['n_total'] * 100:.1f}%</td>
            <td class="{'highlight' if metrics.get('n_fraude_aprovar', 0) == 0 else 'danger'}">{metrics.get('n_fraude_aprovar', 0)}</td>
            <td>{metrics.get('taxa_fraude_aprovar', 0):.3f}%</td>
        </tr>
        <tr>
            <td><span class="badge badge-yellow">🟡 CONFIRMAR</span></td>
            <td>60 — 84</td>
            <td>Solicitar 2FA / biometria</td>
            <td>{metrics['decisoes']['CONFIRMAR']:,}</td>
            <td>{metrics['decisoes']['CONFIRMAR'] / metrics['n_total'] * 100:.1f}%</td>
            <td>{metrics.get('n_fraude_confirmar', 0)}</td>
            <td>{metrics.get('taxa_fraude_confirmar', 0):.2f}%</td>
        </tr>
        <tr>
            <td><span class="badge badge-red">🔴 BLOQUEAR</span></td>
            <td>85 — 100</td>
            <td>Bloquear + enviar ao analista</td>
            <td>{metrics['decisoes']['BLOQUEAR']:,}</td>
            <td>{metrics['decisoes']['BLOQUEAR'] / metrics['n_total'] * 100:.1f}%</td>
            <td class="danger">{metrics.get('n_fraude_bloquear', 0)}</td>
            <td class="danger">{metrics.get('taxa_fraude_bloquear', 0):.2f}%</td>
        </tr>
    </table>

    <div class="callout">
        <strong>Leitura-chave:</strong> A faixa APROVAR deve ter taxa de fraude próxima de 0% —
        significa que nenhuma fraude escapa. A faixa BLOQUEAR deve ter taxa alta —
        os bloqueios são precisos e o analista não perde tempo com falsos alarmes.
    </div>
</div>

<!-- ═══════════ Separação de Scores ═══════════ -->
<div class="section">
    <h2>🔬 Qualidade da Separação</h2>

    <div class="callout success">
        <strong>GAP de +{sep['gap']:.1f} pontos</strong> — Existe uma enorme distância entre os scores das
        fraudes e os das transações normais. Isso significa que o modelo tem alta confiança nas suas classificações.
    </div>

    <div class="two-col">
        <div>
            <h3 style="color: #00d4aa; margin-bottom: 12px;">Transações Normais</h3>
            <table>
                <tr><td>P99.9 (99.9% estão abaixo de)</td><td class="highlight">{sep['normal_p999']}</td></tr>
                <tr><td>Máximo observado</td><td>{sep['normal_max']}</td></tr>
            </table>
        </div>
        <div>
            <h3 style="color: #ff6b6b; margin-bottom: 12px;">Fraudes</h3>
            <table>
                <tr><td>Mínimo (fraude mais "fraca")</td><td class="danger">{sep['fraud_min']}</td></tr>
                <tr><td>Percentil 5</td><td>{sep['fraud_p5']}</td></tr>
                <tr><td>Mediana</td><td>{sep['fraud_median']}</td></tr>
            </table>
        </div>
    </div>
</div>

<!-- ═══════════ Duas perspectivas ═══════════ -->
<div class="section">
    <h2>📐 Duas Perspectivas de Classificação</h2>

    <div class="callout warning">
        <strong>Por que duas?</strong> Na visão conservadora, apenas BLOQUEAR conta como "fraude detectada".
        Na visão ampla, CONFIRMAR também ajuda — se o fraudador não consegue passar a biometria, a fraude é impedida.
    </div>

    <div class="two-col">
        <div>
            <h3 style="color: #ff6b6b; margin-bottom: 12px;">Visão Conservadora (só BLOQUEAR)</h3>
            <table>
                <tr><td>Recall</td><td class="highlight">{p['recall'] * 100:.1f}%</td></tr>
                <tr><td>Precision</td><td>{p['precision'] * 100:.1f}%</td></tr>
                <tr><td>F1-Score</td><td>{p['f1']:.4f}</td></tr>
                <tr><td>Falsos positivos</td><td>{p['fp']:,}</td></tr>
                <tr><td>Falsos negativos</td><td class="{'highlight' if p['fn'] == 0 else 'danger'}">{p['fn']}</td></tr>
            </table>
        </div>
        <div>
            <h3 style="color: #4ecdc4; margin-bottom: 12px;">Visão Ampla (CONFIRMAR + BLOQUEAR)</h3>
            <table>
                <tr><td>Recall</td><td class="highlight">{p2['recall'] * 100:.1f}%</td></tr>
                <tr><td>Precision</td><td>{p2['precision'] * 100:.1f}%</td></tr>
                <tr><td>F1-Score</td><td>{p2['f1']:.4f}</td></tr>
                <tr><td>Falsos positivos</td><td>{p2['fp']:,}</td></tr>
                <tr><td>Falsos negativos</td><td class="{'highlight' if p2['fn'] == 0 else 'danger'}">{p2['fn']}</td></tr>
            </table>
        </div>
    </div>
</div>

<!-- ═══════════ IF Stats ═══════════ -->
<div class="section">
    <h2>🔍 Isolation Forest — Especialista em Primeiras Transações</h2>
    <table>
        <tr><td>Primeiras transações do trimestre</td><td>{metrics['isolation_forest']['first_tx_total']:,}</td></tr>
        <tr><td>IF efetivamente ativo (zona cinzenta do LGBM raw)</td><td>{metrics['isolation_forest']['if_active']:,} ({metrics['isolation_forest']['if_active_pct']:.1f}%)</td></tr>
    </table>
    <div class="callout">
        O Isolation Forest atua <strong>apenas</strong> em primeiras transações do trimestre
        <strong>e somente</strong> quando o LGBM está indeciso (score raw entre {IF_LGBM_RAW_LOW} e {IF_LGBM_RAW_HIGH}).
    </div>
</div>

<!-- ═══════════ Arquitetura ═══════════ -->
<div class="section">
    <h2>⚙️ Arquitetura do Sistema v2.0</h2>
    <table>
        <tr><th>Componente</th><th>Tipo</th><th>Papel</th></tr>
        <tr>
            <td><strong>LightGBM v3 (Raw)</strong></td>
            <td>Gradient Boosting (1500 trees, depth=7)</td>
            <td>Modelo principal — gera score raw (0-1) com 62 features</td>
        </tr>
        <tr>
            <td><strong>Motor de Regras</strong></td>
            <td>6 regras determinísticas</td>
            <td>Features para o LGBM: idade, relacionamento, conta mula, chave aleatória, velocidade, topaz</td>
        </tr>
        <tr>
            <td><strong>Isolation Forest</strong></td>
            <td>300 árvores, contaminação=1%</td>
            <td>Especialista em primeiras transações — detecta anomalias sem histórico</td>
        </tr>
        <tr>
            <td><strong>Mapeamento Híbrido</strong></td>
            <td>Interpolação não-linear (12 âncoras)</td>
            <td>Converte score raw (0-1) → score intuitivo (0-100) com máxima separação</td>
        </tr>
        <tr>
            <td><strong>Preprocessor</strong></td>
            <td>PixPreprocessor (custom)</td>
            <td>Limpeza, imputação de nulos e preparação das features</td>
        </tr>
    </table>

    <div class="callout">
        <strong>Fluxo:</strong> Transação → Features → LGBM Raw (0-1) → IF (1ª tx) → Ensemble Raw
        → Mapeamento Híbrido → Score 0-100 → Decisão (APROVAR / CONFIRMAR / BLOQUEAR)
    </div>
</div>

<!-- ═══════════ Comparação de Modelos ═══════════ -->
<div class="section">
    <h2>📈 Performance do Modelo</h2>
    <table>
        <tr>
            <th>Modelo</th>
            <th>AUC-ROC</th>
            <th>Average Precision</th>
        </tr>
        <tr>
            <td>LGBM Raw</td>
            <td>{metrics['auc_roc_lgbm_raw']:.4f}</td>
            <td>{metrics['ap_lgbm_raw']:.4f}</td>
        </tr>
        <tr>
            <td><strong>Ensemble (LGBM Raw + IF)</strong></td>
            <td class="highlight"><strong>{metrics['auc_roc_ensemble']:.4f}</strong></td>
            <td class="highlight"><strong>{metrics['ap_ensemble']:.4f}</strong></td>
        </tr>
    </table>
</div>
"""
    
    # Benchmark section (condicional)
    bench_html = ""
    if "benchmark_latencia" in metrics:
        b = metrics["benchmark_latencia"]
        sla_status = "✅ APROVADO" if b["sla_ok"] else "❌ REPROVADO"
        sla_color = "#00d4aa" if b["sla_ok"] else "#ff6b6b"
        bench_html = f"""
<!-- ═══════════ Benchmark de Performance ═══════════ -->
<div class="section">
    <h2>⚡ Performance — Impacto no SLA da Transação PIX</h2>

    <div class="callout {'success' if b['sla_ok'] else 'danger'}">
        <strong>SLA: <span style="color: {sla_color};">{sla_status}</span></strong> —
        A inferência leva em média <strong>{b['media_ms']:.1f}ms</strong> por transação,
        muito abaixo do limite de {b['sla_limite_ms']}ms.
        O modelo processa <strong>{b['throughput_por_segundo']:.0f} transações/segundo</strong> em CPU.
    </div>

    <div class="kpi-grid">
        <div class="kpi green">
            <div class="value" style="color: #00d4aa; font-size: 40px;">{b['media_ms']:.1f}ms</div>
            <div class="label">Latência Média</div>
            <div class="detail">Tempo médio de inferência por transação</div>
        </div>
        <div class="kpi green">
            <div class="value" style="color: #00d4aa; font-size: 40px;">{b['p99_ms']:.1f}ms</div>
            <div class="label">Latência P99</div>
            <div class="detail">99% das transações são processadas abaixo deste valor</div>
        </div>
        <div class="kpi green">
            <div class="value" style="color: #00d4aa; font-size: 40px;">{b['throughput_por_segundo']:.0f}</div>
            <div class="label">Transações/Segundo</div>
            <div class="detail">Capacidade de processamento em CPU local</div>
        </div>
    </div>

    <table>
        <tr>
            <th>Métrica</th>
            <th>Valor</th>
            <th>Observação</th>
        </tr>
        <tr>
            <td>Latência média</td>
            <td class="highlight">{b['media_ms']:.2f} ms</td>
            <td>Tempo end-to-end: features + modelo + mapeamento + decisão</td>
        </tr>
        <tr>
            <td>Latência mediana</td>
            <td>{b['mediana_ms']:.2f} ms</td>
            <td>50% das inferências são mais rápidas que isso</td>
        </tr>
        <tr>
            <td>Latência P95</td>
            <td>{b['p95_ms']:.2f} ms</td>
            <td>95% das inferências</td>
        </tr>
        <tr>
            <td>Latência P99</td>
            <td class="highlight">{b['p99_ms']:.2f} ms</td>
            <td>Métrica principal para SLA — 99% das inferências</td>
        </tr>
        <tr>
            <td>Latência máxima</td>
            <td>{b['max_ms']:.2f} ms</td>
            <td>Pior caso observado (cold cache, GC, etc.)</td>
        </tr>
        <tr>
            <td>Desvio padrão</td>
            <td>{b['std_ms']:.2f} ms</td>
            <td>Variabilidade — quanto menor, mais previsível</td>
        </tr>
        <tr>
            <td>Throughput</td>
            <td class="highlight">{b['throughput_por_segundo']:.0f} tx/s</td>
            <td>Inferências por segundo em CPU local</td>
        </tr>
        <tr>
            <td>SLA &lt; {b['sla_limite_ms']}ms</td>
            <td style="color: {sla_color}; font-weight: bold; font-size: 16px;">{sla_status}</td>
            <td>{'Modelo não impacta o tempo da transação PIX' if b['sla_ok'] else 'Necessita otimização'}</td>
        </tr>
    </table>

    <div class="callout">
        <strong>Contexto:</strong> Uma transação PIX tem SLA regulatório de até 10 segundos (BACEN).
        O modelo de fraude adiciona apenas <strong>{b['media_ms']:.1f}ms</strong> — representando
        <strong>{b['media_ms'] / 10000 * 100:.3f}%</strong> do tempo total permitido.
        Em produção com servidor dedicado, a latência tende a ser <strong>ainda menor</strong>.
    </div>
</div>
"""

    html += f"""
{bench_html}

<div class="footer">
    <p>🔒 Documento confidencial — uso interno BRB</p>
    <p>Sistema Anomalia PIX v2.0 | Pipeline Híbrido Otimizado | {datetime.now().strftime('%d/%m/%Y')}</p>
</div>

</body>
</html>"""


    output = RELATORIO_DIR / "relatorio_executivo.html"
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✅ Relatório HTML: {output}")


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n")
    print("█" * 70)
    print("  TESTE COMPLETO DO PIPELINE v2.0 + RELATÓRIO EXECUTIVO")
    print("  Score Híbrido Otimizado (0-100)")
    print("  🟢 APROVAR [0-60) | 🟡 CONFIRMAR [60-85) | 🔴 BLOQUEAR [85-100]")
    print("█" * 70)

    t0 = time.time()

    # 1. Carregar artefatos
    artifacts = load_artifacts()

    # 2. Carregar dados de teste
    X_test, y_test = load_test_data()

    # 3. Executar pipeline
    df_results = run_pipeline(X_test, y_test, artifacts)

    # 4. Calcular métricas
    metrics = calculate_metrics(df_results, artifacts)

    # 5. Salvar métricas JSON
    metrics_path = RELATORIO_DIR / "relatorio_metricas.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  ✅ Métricas JSON: {metrics_path}")

    # 6. Gráficos
    plot_dashboard(df_results, metrics)

    # 7. Relatório HTML
    generate_html_report(metrics)

    # 8. Resultados detalhados
    results_path = RELATORIO_DIR / "resultados_detalhados.csv"
    df_results.to_csv(results_path, index=False)
    print(f"  ✅ Resultados CSV: {results_path}")
    

    # 9. Benchmark de Latência (inferência individual) 
    print(f"\n{'=' * 70}")
    print("  BENCHMARK DE LATÊNCIA — INFERÊNCIA INDIVIDUAL")
    print(f"{'=' * 70}")

    # Importar o pipeline de inferência
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from pipeline_inferencia import PipelineInferencia

        pipeline = PipelineInferencia(artefatos_dir=str(ARTEFATOS_DIR))

        # Transação de teste realista
        tx_bench = {
            "cd_pix": "E00000208202603181530001234567890",
            "dt_pix": "2026-03-18 15:30:00",
            "cd_cpf_pagador": "12345678901",
            "cd_cpf_cnpj_recebedor": "98765432100",
            "ds_chave_pix": "98765432100",
            "ds_tipo_chave": "DOCUMENTO/TELEFONE",
            "vl_pix": 150.00,
            "qt_total_pix_trimestre": 25,
            "vl_mediana_pix_trimestre": 120.00,
            "vl_desvio_padrao_pix_trimestre": 80.00,
            "qt_intervalo_transacao_minuto": 1440,
            "qt_intervalo_mediana_trimestre": 1200,
            "qt_intervalo_desvio_padrao_trimestre": 600,
            "qt_pix_dia_maximo_trimestre": 3,
            "device_name": "Samsung Galaxy S23",
            "app_version": "7.12.0",
            "ip_address": "192.168.1.1",
            "latencia_rede_ms": 45.0,
            "vl_latencia_rede_media_trimestre": 42.0,
            "tempo_interacao_ms": None,
            "vl_tempo_interacao_medio_trimestre": None,
            "tempo_processamento_host_ms": 120.0,
            "metodo_autenticacao": "biometria",
            "session_id": "sess_abc123",
            "cd_retorno": "00",
            "topaz_risk_score": 1.5,
            "topaz_transacao_rejeitada": 0,
            "topaz_transacao_habilitada": 1,
            "is_agendamento_recorrente": "false",
            "topaz_sync_id": None,
            "qt_aparelhos_distintos_trimestre": 1,
            "nr_idade": 35,
            "qt_tempo_relacionamento_mes": 120,
        }

        # Warmup (3 chamadas para estabilizar)
        for _ in range(3):
            pipeline.predict(tx_bench)

        # Benchmark: N iterações
        N_ITER = 200
        latencias = []

        for i in range(N_ITER):
            t0 = time.perf_counter()
            _ = pipeline.predict(tx_bench)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencias.append(elapsed_ms)

        latencias = np.array(latencias)

        bench = {
            "n_iteracoes": N_ITER,
            "media_ms": round(float(np.mean(latencias)), 2),
            "mediana_ms": round(float(np.median(latencias)), 2),
            "p95_ms": round(float(np.percentile(latencias, 95)), 2),
            "p99_ms": round(float(np.percentile(latencias, 99)), 2),
            "min_ms": round(float(np.min(latencias)), 2),
            "max_ms": round(float(np.max(latencias)), 2),
            "std_ms": round(float(np.std(latencias)), 2),
            "throughput_por_segundo": round(1000.0 / float(np.mean(latencias)), 1),
        }

        # SLA check
        SLA_LIMITE_MS = 100  # Limite máximo aceitável por transação
        sla_ok = bench["p99_ms"] < SLA_LIMITE_MS

        print(f"\n  Configuração: {N_ITER} inferências individuais (pipeline completo)")
        print(f"  Hardware: CPU local (sem GPU)")
        print(f"\n  ┌──────────────────────────────────────────────┐")
        print(f"  │        LATÊNCIA POR TRANSAÇÃO                │")
        print(f"  ├──────────────────────────────────────────────┤")
        print(f"  │  Média:              {bench['media_ms']:8.2f} ms            │")
        print(f"  │  Mediana:            {bench['mediana_ms']:8.2f} ms            │")
        print(f"  │  P95:                {bench['p95_ms']:8.2f} ms            │")
        print(f"  │  P99:                {bench['p99_ms']:8.2f} ms            │")
        print(f"  │  Min:                {bench['min_ms']:8.2f} ms            │")
        print(f"  │  Max:                {bench['max_ms']:8.2f} ms            │")
        print(f"  │  Desvio padrão:      {bench['std_ms']:8.2f} ms            │")
        print(f"  ├──────────────────────────────────────────────┤")
        print(f"  │  Throughput:    {bench['throughput_por_segundo']:8.1f} tx/s             │")
        print(f"  │  SLA < {SLA_LIMITE_MS}ms:       {'✅ OK' if sla_ok else '❌ ATENÇÃO'}                     │")
        print(f"  └──────────────────────────────────────────────┘")

        # Adicionar ao metrics
        metrics["benchmark_latencia"] = bench
        metrics["benchmark_latencia"]["sla_limite_ms"] = SLA_LIMITE_MS
        metrics["benchmark_latencia"]["sla_ok"] = sla_ok

        # Re-salvar métricas com benchmark
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

    except ImportError:
        print(f"  ⚠️  pipeline_inferencia.py não encontrado — benchmark pulado")
        bench = None


    # Resumo final
    elapsed = time.time() - t0
    e = metrics["executivo"]
    sep = metrics["separacao"]

    print(f"\n\n{'█' * 70}")
    print("  RESUMO FINAL — PIPELINE v2.0")
    print(f"{'█' * 70}")
    print(f"""
  📊 BASE DE TESTE
     Transações:         {metrics['n_total']:,}
     Fraudes:             {metrics['n_fraudes']} ({metrics['taxa_fraude_pct']:.2f}%)

  🎯 PERFORMANCE (Score Híbrido 0-100)
     Fraudes detectadas:  {e['fraudes_detectadas_pct']:.1f}% ({metrics['pipeline_bloquear']['tp']}/{metrics['n_fraudes']})
     Fraudes perdidas:    {e['fraudes_nao_detectadas_n']} ({e['fraudes_nao_detectadas_pct']:.1f}%)
     Falsos alarmes:      {e['falsos_alarmes_n']} ({e['falsos_alarmes_pct']:.2f}%)
     Precisão alarmes:    {e['precisao_alarmes_pct']:.1f}%
     AUC-ROC:             {e['auc_roc']:.4f}
     F1-Score:            {e['f1']:.4f}
     GAP separação:       +{sep['gap']:.1f} pontos
""")

    if "benchmark_latencia" in metrics:
        b = metrics["benchmark_latencia"]
        print(f"""
  ⚡ PERFORMANCE (LATÊNCIA POR TRANSAÇÃO)
     Média:               {b['media_ms']:.2f} ms
     Mediana:             {b['mediana_ms']:.2f} ms
     P95:                 {b['p95_ms']:.2f} ms
     P99:                 {b['p99_ms']:.2f} ms
     Throughput:          {b['throughput_por_segundo']:.0f} inferências/segundo
     SLA < {b['sla_limite_ms']}ms:          {'✅ APROVADO' if b['sla_ok'] else '❌ REPROVADO'}""")

    print(f"""
  📏 FAIXAS DE DECISÃO
     🟢 APROVAR   [0-60):   {metrics['decisoes']['APROVAR']:,} tx | {metrics.get('n_fraude_aprovar', 0)} fraudes
     🟡 CONFIRMAR [60-85):  {metrics['decisoes']['CONFIRMAR']:,} tx | {metrics.get('n_fraude_confirmar', 0)} fraudes
     🔴 BLOQUEAR  [85-100]: {metrics['decisoes']['BLOQUEAR']:,} tx | {metrics.get('n_fraude_bloquear', 0)} fraudes

  📁 ARQUIVOS GERADOS
     📄 {RELATORIO_DIR}/relatorio_executivo.html
     📊 {RELATORIO_DIR}/dashboard_executivo.png
     📋 {RELATORIO_DIR}/relatorio_metricas.json
     📑 {RELATORIO_DIR}/resultados_detalhados.csv

  ⏱️  Tempo total: {elapsed:.1f}s
    """)


if __name__ == "__main__":
    main()
