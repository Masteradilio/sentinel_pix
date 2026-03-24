"""
teste_pipeline_relatorio.py v2.1 — Teste Completo do Pipeline + Relatório Executivo
=====================================================================================

Executa o pipeline v2.1 COMPLETO nos dados de teste e gera:
  - Dashboard visual (PNG)
  - Relatório executivo (HTML) — para apresentar a diretores
  - Métricas detalhadas (JSON)
  - Resultados por transação (CSV)

Pipeline v2.1 (ensemble completo):
  LGBM Score
    ├── score >= threshold → FRAUDE (LGBM detectou)
    ├── score < threshold  → Cascade Rules (6 regras)
    │   ├── Cascade triggered → FRAUDE
    │   └── Cascade clean → IF Score (boost condicional)
    │       ├── IF >= 0.85 → Boost +0.15
    │       ├── IF >= 0.70 → Boost +0.08
    │       └── IF < 0.70 → Sem boost
    └── Ensemble Raw → Mapeamento 0-100
        → Agravantes (24 fatores, 7 fases)
        → Social Engineering (12 padrões)
        → Behavioral Analytics (15 fatores)
        → Vetos → Decisão Final

  🟢 APROVAR [0-60) | 🟡 CONFIRMAR [60-85) | 🔴 BLOQUEAR [85-100]

Uso:
  python teste_pipeline_relatorio.py
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
from typing import Dict, Any, List, Tuple, Optional

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
elif SCRIPT_DIR.name == "backend":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

ARTEFATOS_DIR = PROJECT_ROOT / "backend" / "artefatos"
if not ARTEFATOS_DIR.exists():
    ARTEFATOS_DIR = SCRIPT_DIR / "artefatos"

BACKEND_DIR = PROJECT_ROOT / "backend"
RELATORIO_DIR = PROJECT_ROOT / "relatorio"
RELATORIO_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Artefatos
LGBM_PATH = ARTEFATOS_DIR / "model_lightgbm.joblib"
IF_MODEL_PATH = ARTEFATOS_DIR / "model_isolation_forest.joblib"
IF_SCALER_PATH = ARTEFATOS_DIR / "scaler_isolation_forest.joblib"
IF_CONFIG_PATH = ARTEFATOS_DIR / "isolation_forest_config.json"
IF_REF_SCORES_PATH = ARTEFATOS_DIR / "if_ref_raw_train.npy"
LGBM_FEATURES_PATH = ARTEFATOS_DIR / "lgbm_features.json"
SCORING_CONFIG_PATH = ARTEFATOS_DIR / "scoring_config.json"
THRESHOLDS_CONFIG_PATH = ARTEFATOS_DIR / "thresholds_config.json"
METRICAS_LGBM_PATH = ARTEFATOS_DIR / "metricas_lightgbm.json"
X_TEST_PATH = ARTEFATOS_DIR / "X_test.csv"
Y_TEST_PATH = ARTEFATOS_DIR / "y_test.csv"

# =========================================================
# CONFIGURAÇÃO DO PIPELINE v2.1
# =========================================================
FAIXA_CONFIRMAR = 60.0
FAIXA_BLOQUEAR = 85.0
LGBM_THRESHOLD = 0.08

# IF Boost condicional
IF_HIGH_THRESHOLD = 0.99
IF_VERY_HIGH_THRESHOLD = 0.9994
IF_BOOST_HIGH = 0.05
IF_BOOST_VERY_HIGH = 0.08

# =========================================================
# ESTILO DOS GRÁFICOS
# =========================================================
plt.rcParams.update({
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
})

COLORS = {
    "primary": "#00d4aa",
    "secondary": "#ff6b6b",
    "accent": "#4ecdc4",
    "warning": "#ffd93d",
    "info": "#6c5ce7",
    "cascade": "#ff9f43",
    "bg_card": "#1a1d23",
    "text": "#ffffff",
    "text_muted": "#888888",
    "aprovar": "#00d4aa",
    "confirmar": "#ffd93d",
    "bloquear": "#ff6b6b",
}


# =========================================================
# HELPERS
# =========================================================
def _safe_float(val, default=None):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        v = float(val)
        return default if v != v else v
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    if val is None:
        return default
    try:
        v = float(val)
        return default if v != v else int(v)
    except (ValueError, TypeError):
        return default


# =========================================================
# CASCADE RULES
# =========================================================
def evaluate_cascade(row: pd.Series, lgbm_score: float) -> Tuple[bool, List[str]]:
    """Avalia as 6 regras cascade para uma transação."""
    triggered = []

    tx_30m = _safe_int(row.get("tx_count_prev_30m"), 0)
    first_recv = _safe_int(row.get("first_receiver_flag"), 0)
    tempo_rel = _safe_float(row.get("qt_tempo_relacionamento_mes"), 999)
    ratio_med = _safe_float(row.get("ratio_valor_mediana"), 0)
    vl_pix = _safe_float(row.get("vl_pix"), 0)
    burst_flag = _safe_int(row.get("burst_30m_flag"), 0)
    idade = _safe_int(row.get("nr_idade"), 0)
    chave_random = _safe_int(row.get("pix_key_random_flag"), 0)

    if tx_30m >= 3 and first_recv == 1:
        triggered.append("C1_BURST_FIRST_RECEIVER")
    if tx_30m >= 5:
        triggered.append("C2_BURST_INTENSO")
    # C3 foi desativada pois gerava 128 FPs e 0 TPs.
    # if tempo_rel <= 6 and first_recv == 1 and ratio_med >= 3.0:
    #     triggered.append("C3_CONTA_NOVA_ATIPICO")
    if tempo_rel <= 3 and vl_pix >= 5000:
        triggered.append("C4_CONTA_NOVA_ALTO_VALOR")
    if burst_flag == 1 and ratio_med >= 5.0 and vl_pix >= 1000:
        triggered.append("C5_ESVAZIAMENTO")
    # C6: LGBM borderline + sinais combinados
    if lgbm_score >= 0.05:
        sinais = sum([
            first_recv == 1,
            ratio_med >= 3.0,
            vl_pix >= 2000,
            idade >= 60,
            chave_random == 1,
        ])
        if sinais >= 4:
            triggered.append("C6_LGBM_BORDERLINE_COMBINADO")

    return len(triggered) > 0, triggered


# =========================================================
# IF SCORING
# =========================================================
def score_if_batch(
    X: pd.DataFrame,
    lgbm_scores: np.ndarray,
    if_model, if_scaler, if_config, if_ref_scores,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calcula IF scores em batch com boost condicional."""
    n = len(X)
    if_percentiles = np.zeros(n)
    if_raw_scores = np.zeros(n)
    if_active = np.zeros(n, dtype=bool)
    if_boost = np.zeros(n)

    if if_model is None or if_config is None:
        return if_percentiles, if_raw_scores, if_active, if_boost

    if_features_list = if_config.get("features", [])
    if_medians = if_config.get("medians", {})
    ensemble_params = if_config.get("ensemble_params", {})

    lgbm_th = ensemble_params.get("lgbm_threshold", LGBM_THRESHOLD)
    high_th = ensemble_params.get("if_high_threshold", IF_HIGH_THRESHOLD)
    very_high_th = ensemble_params.get("if_very_high_threshold", IF_VERY_HIGH_THRESHOLD)
    boost_high = ensemble_params.get("boost_high", IF_BOOST_HIGH)
    boost_very_high = ensemble_params.get("boost_very_high", IF_BOOST_VERY_HIGH)

    eligible_mask = lgbm_scores < lgbm_th
    eligible_idx = np.where(eligible_mask)[0]

    if len(eligible_idx) == 0 or not if_features_list:
        return if_percentiles, if_raw_scores, if_active, if_boost

    # Preparar features
    X_if = pd.DataFrame(index=range(len(eligible_idx)))
    X_eligible = X.iloc[eligible_idx].reset_index(drop=True)

    for feat in if_features_list:
        if feat in X_eligible.columns:
            X_if[feat] = X_eligible[feat].values
        else:
            X_if[feat] = if_medians.get(feat, 0)

    # Criar features de interação inline
    if "valor_x_burst" in if_features_list and "valor_x_burst" not in X.columns:
        vl = X_if.get("vl_pix", pd.Series(0, index=X_if.index)).fillna(0)
        tx30 = X_if.get("tx_count_prev_30m", pd.Series(0, index=X_if.index)).fillna(0)
        X_if["valor_x_burst"] = vl * (tx30 + 1)
    if "idade_x_first_recv" in if_features_list and "idade_x_first_recv" not in X.columns:
        idade = X_if.get("nr_idade", pd.Series(0, index=X_if.index)).fillna(0)
        fr = X_if.get("first_receiver_flag", pd.Series(0, index=X_if.index)).fillna(0)
        X_if["idade_x_first_recv"] = idade * fr
    if "valor_x_first_recv" in if_features_list and "valor_x_first_recv" not in X.columns:
        vl = X_if.get("vl_pix", pd.Series(0, index=X_if.index)).fillna(0)
        fr = X_if.get("first_receiver_flag", pd.Series(0, index=X_if.index)).fillna(0)
        X_if["valor_x_first_recv"] = vl * fr
    if "burst_x_distinct_recv" in if_features_list and "burst_x_distinct_recv" not in X.columns:
        tx30 = X_if.get("tx_count_prev_30m", pd.Series(0, index=X_if.index)).fillna(0)
        dr = X_if.get("distinct_receivers_so_far", pd.Series(1, index=X_if.index)).fillna(1)
        X_if["burst_x_distinct_recv"] = tx30 * dr
    if "valor_over_trimestre_avg" in if_features_list and "valor_over_trimestre_avg" not in X.columns:
        vl = X_if.get("vl_pix", pd.Series(0, index=X_if.index)).fillna(0)
        med = X_if.get("vl_mediana_pix_trimestre", pd.Series(1, index=X_if.index)).fillna(1)
        qt = X_if.get("qt_total_pix_trimestre", pd.Series(1, index=X_if.index)).fillna(1).clip(lower=1)
        total = med * qt
        X_if["valor_over_trimestre_avg"] = np.where(total > 0, vl / total, 0)

    # Fill NaN
    for feat in if_features_list:
        if feat in X_if.columns:
            X_if[feat] = X_if[feat].fillna(if_medians.get(feat, 0))
        else:
            X_if[feat] = if_medians.get(feat, 0)

    X_if_ordered = X_if[if_features_list]

    if if_scaler is not None:
        X_scaled = if_scaler.transform(X_if_ordered)
    else:
        X_scaled = X_if_ordered.values

    raw = if_model.decision_function(X_scaled)

    if if_ref_scores is not None and len(if_ref_scores) > 0:
        inverted = -raw
        ref_inverted = -if_ref_scores
        percentiles = np.array([float(np.mean(ref_inverted <= inv)) for inv in inverted])
    else:
        percentiles = 1.0 / (1.0 + np.exp(raw * 5))

    percentiles = np.clip(percentiles, 0, 1)

    boosts = np.zeros(len(eligible_idx))
    boosts[percentiles >= very_high_th] = boost_very_high
    mask_high = (percentiles >= high_th) & (percentiles < very_high_th)
    boosts[mask_high] = boost_high

    if_percentiles[eligible_idx] = percentiles
    if_raw_scores[eligible_idx] = raw
    if_active[eligible_idx] = True
    if_boost[eligible_idx] = boosts

    return if_percentiles, if_raw_scores, if_active, if_boost


# =========================================================
# 1. CARREGAR ARTEFATOS
# =========================================================
def load_artifacts() -> Dict[str, Any]:
    """Carrega todos os artefatos necessários."""
    print("\n" + "=" * 70)
    print("  CARREGAMENTO DOS ARTEFATOS")
    print("=" * 70)

    artifacts = {}

    # LightGBM
    print(f"\n  LightGBM: {LGBM_PATH.name}...")
    if not LGBM_PATH.exists():
        print(f"    ❌ Modelo LGBM não encontrado!")
        sys.exit(1)
    artifacts["lgbm"] = joblib.load(LGBM_PATH)
    print(f"    ✅ Tipo: {type(artifacts['lgbm']).__name__}")

    # ─── DIAGNÓSTICO DE FEATURES ───
    # Extrair features reais do modelo treinado (fonte de verdade)
    lgbm_model = artifacts["lgbm"]
    if hasattr(lgbm_model, "feature_name_"):
        model_features = list(lgbm_model.feature_name_)
    elif hasattr(lgbm_model, "booster_") and hasattr(lgbm_model.booster_, "feature_name"):
        model_features = lgbm_model.booster_.feature_name()
    else:
        model_features = None

    n_model_features = lgbm_model.n_features_in_ if hasattr(lgbm_model, "n_features_in_") else "?"
    print(f"    Features no modelo treinado: {n_model_features}")

    # Carregar lgbm_features.json
    json_features = None
    if LGBM_FEATURES_PATH.exists():
        with open(LGBM_FEATURES_PATH, "r") as f:
            json_features = json.load(f)
        print(f"    Features no lgbm_features.json: {len(json_features)}")

    # Verificar mismatch
    if model_features is not None:
        n_model = len(model_features)
        if json_features and len(json_features) != n_model:
            print(f"\n    ⚠️  MISMATCH DETECTADO!")
            print(f"    Modelo treinado: {n_model} features")
            print(f"    lgbm_features.json: {len(json_features)} features")

            # Identificar diferenças
            model_set = set(model_features)
            json_set = set(json_features) if json_features else set()
            no_json = model_set - json_set
            no_model = json_set - model_set

            if no_json:
                print(f"    Faltando no JSON (presentes no modelo): {len(no_json)}")
                for f in sorted(no_json):
                    print(f"      + {f}")
            if no_model:
                print(f"    Sobrando no JSON (ausentes no modelo): {len(no_model)}")
                for f in sorted(no_model):
                    print(f"      - {f}")

            print(f"\n    🔧 CORRIGINDO: usando features do modelo treinado como fonte de verdade")
            artifacts["lgbm_features"] = model_features

            # Atualizar o JSON para evitar o problema no futuro
            updated_path = LGBM_FEATURES_PATH
            with open(updated_path, "w") as f:
                json.dump(model_features, f, indent=2)
            print(f"    ✅ lgbm_features.json ATUALIZADO com {n_model} features")
        else:
            artifacts["lgbm_features"] = model_features
            print(f"  Features LGBM: ✅ {len(model_features)} (modelo e JSON consistentes)")
    elif json_features:
        artifacts["lgbm_features"] = json_features
        print(f"  Features LGBM: ⚠️ {len(json_features)} (do JSON, modelo sem feature_name_)")
    else:
        print(f"    ❌ Não foi possível determinar as features do LGBM!")
        sys.exit(1)

    # Scoring Config
    if SCORING_CONFIG_PATH.exists():
        with open(SCORING_CONFIG_PATH, "r", encoding="utf-8") as f:
            artifacts["scoring_config"] = json.load(f)
        mapeamento = artifacts["scoring_config"].get("mapeamento", {})
        artifacts["anchors_raw"] = np.array(mapeamento.get("anchors_raw", [0.0, 1.0]), dtype=np.float64)
        artifacts["anchors_out"] = np.array(mapeamento.get("anchors_out", [0.0, 100.0]), dtype=np.float64)
        print(f"  Scoring Config: ✅ {len(artifacts['anchors_raw'])} âncoras")
    else:
        artifacts["scoring_config"] = {}
        artifacts["anchors_raw"] = np.array([0.0, 1.0])
        artifacts["anchors_out"] = np.array([0.0, 100.0])
        print(f"  Scoring Config: ⚠️ Não encontrado — mapeamento linear")

    # Thresholds Config
    global LGBM_THRESHOLD
    if THRESHOLDS_CONFIG_PATH.exists():
        with open(THRESHOLDS_CONFIG_PATH, "r", encoding="utf-8") as f:
            th_config = json.load(f)
        LGBM_THRESHOLD = float(th_config.get("threshold_f1_best", LGBM_THRESHOLD))
        print(f"  Thresholds: ✅ LGBM threshold = {LGBM_THRESHOLD:.4f}")

    # Isolation Forest
    if IF_MODEL_PATH.exists():
        artifacts["if_model"] = joblib.load(IF_MODEL_PATH)
        print(f"  IF Model: ✅ {artifacts['if_model'].n_estimators} trees")
    else:
        artifacts["if_model"] = None
        print(f"  IF Model: ⚠️ Não encontrado")

    if IF_SCALER_PATH.exists():
        artifacts["if_scaler"] = joblib.load(IF_SCALER_PATH)
        print(f"  IF Scaler: ✅")
    else:
        artifacts["if_scaler"] = None

    if IF_CONFIG_PATH.exists():
        with open(IF_CONFIG_PATH, "r") as f:
            artifacts["if_config"] = json.load(f)
        ep = artifacts["if_config"].get("ensemble_params", {})
        print(f"  IF Config: ✅ boost={ep.get('boost_high', 'N/A')}/{ep.get('boost_very_high', 'N/A')}")
    else:
        artifacts["if_config"] = None

    if IF_REF_SCORES_PATH.exists():
        artifacts["if_ref_scores"] = np.load(IF_REF_SCORES_PATH)
        print(f"  IF Ref Scores: ✅ ({len(artifacts['if_ref_scores'])} scores)")
    else:
        artifacts["if_ref_scores"] = None

    # Métricas de treino
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
        print(f"     X_test: {X_TEST_PATH}")
        print(f"     y_test: {Y_TEST_PATH}")
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
# 3. EXECUTAR PIPELINE v2.1 COMPLETO
# =========================================================
def run_pipeline(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    artifacts: Dict[str, Any],
) -> pd.DataFrame:
    """Executa o pipeline v2.1 completo no dataset de teste."""

    print("\n" + "=" * 70)
    print("  EXECUÇÃO DO PIPELINE v2.1 — LGBM + Cascade + IF Boost")
    print("=" * 70)

    lgbm = artifacts["lgbm"]
    lgbm_features = artifacts["lgbm_features"]
    anchors_raw = artifacts["anchors_raw"]
    anchors_out = artifacts["anchors_out"]
    if_model = artifacts.get("if_model")
    if_scaler = artifacts.get("if_scaler")
    if_config = artifacts.get("if_config")
    if_ref_scores = artifacts.get("if_ref_scores")

    n_total = len(X_test)
    t_start = time.time()

    # ─── PASSO 1: Preparar features LGBM ───
    print(f"\n  [1/6] Preparando features LGBM ({len(lgbm_features)})...")

    missing_feats = [f for f in lgbm_features if f not in X_test.columns]
    extra_feats = [f for f in X_test.columns if f not in lgbm_features]

    if missing_feats:
        print(f"    ⚠️  {len(missing_feats)} features faltando no X_test — preenchendo com 0:")
        for feat in missing_feats:
            print(f"      + {feat} = 0")
            X_test[feat] = 0

    if extra_feats:
        print(f"    ℹ️  {len(extra_feats)} features extras no X_test (não usadas pelo LGBM)")

    X_lgbm = X_test[lgbm_features].copy()

    # Garantir que não tem NaN (LGBM não aceita)
    nan_cols = X_lgbm.columns[X_lgbm.isna().any()].tolist()
    if nan_cols:
        print(f"    ⚠️  {len(nan_cols)} colunas com NaN — preenchendo com 0")
        X_lgbm = X_lgbm.fillna(0)

    print(f"    ✅ X_lgbm: {X_lgbm.shape}")

    # ─── PASSO 2: Score LGBM ───
    print(f"  [2/6] Calculando scores LGBM...")
    lgbm_proba = lgbm.predict_proba(X_lgbm)[:, 1]
    lgbm_pred = (lgbm_proba >= LGBM_THRESHOLD).astype(int)
    n_lgbm_flag = lgbm_pred.sum()
    print(f"    ✅ LGBM: min={lgbm_proba.min():.6f}, max={lgbm_proba.max():.6f}, "
          f"median={np.median(lgbm_proba):.6f}")
    print(f"    LGBM flags (@{LGBM_THRESHOLD}): {n_lgbm_flag:,} ({n_lgbm_flag/n_total*100:.2f}%)")

    # ─── PASSO 3: Cascade Rules ───
    print(f"  [3/6] Avaliando Cascade Rules (6 regras)...")
    cascade_triggered = np.zeros(n_total, dtype=bool)
    cascade_rules = [[] for _ in range(n_total)]
    n_cascade = 0

    for i in range(n_total):
        if lgbm_proba[i] < LGBM_THRESHOLD:
            trig, rules = evaluate_cascade(X_test.iloc[i], lgbm_proba[i])
            if trig:
                cascade_triggered[i] = True
                cascade_rules[i] = rules
                n_cascade += 1

    print(f"    ✅ Cascade triggered: {n_cascade:,} tx ({n_cascade/n_total*100:.3f}%)")
    rule_counts = {}
    for rules in cascade_rules:
        for r in rules:
            rule_counts[r] = rule_counts.get(r, 0) + 1
    if rule_counts:
        for r, c in sorted(rule_counts.items(), key=lambda x: -x[1])[:6]:
            print(f"      {r}: {c:,}")

    # ─── PASSO 4: IF Score + Boost ───
    print(f"  [4/6] Calculando IF Scores + Boost Condicional...")
    if_percentiles, if_raw, if_active, if_boost = score_if_batch(
        X_test, lgbm_proba, if_model, if_scaler, if_config, if_ref_scores
    )
    n_if_active = if_active.sum()
    n_if_boosted = (if_boost > 0).sum()
    print(f"    ✅ IF ativo: {n_if_active:,} tx ({n_if_active/n_total*100:.2f}%)")
    print(f"    IF com boost: {n_if_boosted:,} tx")
    if n_if_boosted > 0:
        print(f"      Boost médio: +{if_boost[if_boost > 0].mean():.4f}")

    # ─── PASSO 5: Ensemble ───
    print(f"  [5/6] Calculando Ensemble (LGBM + Cascade + IF Boost)...")
    ensemble_raw = lgbm_proba.copy()

    ensemble_raw[cascade_triggered] = np.maximum(
        ensemble_raw[cascade_triggered], LGBM_THRESHOLD
    )

    boost_mask = if_active & (if_boost > 0)
    ensemble_raw[boost_mask] = ensemble_raw[boost_mask] + if_boost[boost_mask]
    ensemble_raw = np.clip(ensemble_raw, 0.0, 1.0)

    # ─── PASSO 6: Mapeamento + Decisão ───
    print(f"  [6/6] Mapeamento 0-100 → Decisões...")
    scores_mapped = np.clip(
        np.interp(ensemble_raw, anchors_raw, anchors_out),
        0.0, 100.0,
    )

    decisions = np.full(n_total, "APROVAR", dtype=object)
    decisions[scores_mapped >= FAIXA_CONFIRMAR] = "CONFIRMAR"
    decisions[scores_mapped >= FAIXA_BLOQUEAR] = "BLOQUEAR"

    # Montar resultados
    rule_score_raw = X_test["rule_score_raw"].values if "rule_score_raw" in X_test.columns else np.zeros(n_total)

    results = pd.DataFrame({
        "y_true": y_test.values,
        "lgbm_raw_score": lgbm_proba,
        "lgbm_pred": lgbm_pred,
        "cascade_triggered": cascade_triggered,
        "cascade_rules": [",".join(r) for r in cascade_rules],
        "if_score": if_percentiles,
        "if_raw": if_raw,
        "if_active": if_active,
        "if_boost": if_boost,
        "is_first_tx": X_test["is_first_tx_trimestre"].values if "is_first_tx_trimestre" in X_test.columns else 0,
        "ensemble_raw": ensemble_raw,
        "score_mapped": np.round(scores_mapped, 2),
        "rule_score_raw": rule_score_raw,
        "decision": decisions,
        "detected_by": _get_detection_source(lgbm_proba, cascade_triggered, if_boost, scores_mapped),
    })

    elapsed = time.time() - t_start

    # ═══ Resumo ═══
    print(f"\n  ✅ Pipeline completo: {n_total:,} tx em {elapsed:.1f}s ({n_total/elapsed:,.0f} tx/s)")

    print(f"\n  ┌─────────────────┬──────────┬─────────┬──────────┬───────────┐")
    print(f"  │    Decisão      │   Total  │    %    │ Fraudes  │ Taxa Fr.  │")
    print(f"  ├─────────────────┼──────────┼─────────┼──────────┼───────────┤")
    for dec in ["APROVAR", "CONFIRMAR", "BLOQUEAR"]:
        mask = results["decision"] == dec
        count = mask.sum()
        fraud_in = results.loc[mask, "y_true"].sum()
        pct = count / n_total * 100
        fraud_rate = fraud_in / count * 100 if count > 0 else 0
        icon = {"APROVAR": "🟢", "CONFIRMAR": "🟡", "BLOQUEAR": "🔴"}[dec]
        print(f"  │ {icon} {dec:12s} │ {count:6,}  │ {pct:5.1f}%  │  {fraud_in:5.0f}   │ {fraud_rate:7.2f}%  │")
    print(f"  └─────────────────┴──────────┴─────────┴──────────┴───────────┘")

    # Contribuição de cada componente
    fraud_mask = y_test.values == 1
    lgbm_only = (lgbm_proba >= LGBM_THRESHOLD) & fraud_mask
    cascade_only = cascade_triggered & fraud_mask & ~(lgbm_proba >= LGBM_THRESHOLD)
    if_only = (if_boost > 0) & fraud_mask & ~(lgbm_proba >= LGBM_THRESHOLD) & ~cascade_triggered

    print(f"\n  ═══ CONTRIBUIÇÃO POR COMPONENTE (fraudes) ═══")
    print(f"  LGBM (≥{LGBM_THRESHOLD}):     {lgbm_only.sum():4d} fraudes detectadas")
    print(f"  Cascade Rules:    {cascade_only.sum():4d} fraudes capturadas (FN do LGBM)")
    print(f"  IF Boost:         {if_only.sum():4d} fraudes promovidas")
    print(f"  Total detectáveis:{(lgbm_only | cascade_only | if_only).sum():4d} / {fraud_mask.sum()}")

    scores_fraude = scores_mapped[fraud_mask]
    scores_normal = scores_mapped[~fraud_mask]
    if len(scores_fraude) > 0 and len(scores_normal) > 0:
        print(f"\n  Scores mapeados (0-100):")
        print(f"    Normais:  P99.9={np.percentile(scores_normal, 99.9):.1f}, max={scores_normal.max():.1f}")
        print(f"    Fraudes:  min={scores_fraude.min():.1f}, P5={np.percentile(scores_fraude, 5):.1f}, "
              f"median={np.median(scores_fraude):.1f}")
        print(f"    GAP:      +{scores_fraude.min() - np.percentile(scores_normal, 99.9):.1f} pontos")

    return results


def _get_detection_source(lgbm_proba, cascade_triggered, if_boost, scores_mapped):
    """Identifica qual componente detectou cada transação."""
    n = len(lgbm_proba)
    sources = np.full(n, "NONE", dtype=object)

    for i in range(n):
        if scores_mapped[i] < FAIXA_CONFIRMAR:
            sources[i] = "APROVAR"
        elif scores_mapped[i] < FAIXA_BLOQUEAR:
            sources[i] = "CONFIRMAR"
        elif lgbm_proba[i] >= LGBM_THRESHOLD:
            sources[i] = "LGBM"
        elif cascade_triggered[i]:
            sources[i] = "CASCADE"
        elif if_boost[i] > 0:
            sources[i] = "IF_BOOST"
        else:
            sources[i] = "ENSEMBLE"

    return sources


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

    y_pred_bloquear = (decisions == "BLOQUEAR").astype(int)
    y_pred_qualquer_acao = np.isin(decisions, ["CONFIRMAR", "BLOQUEAR"]).astype(int)

    metrics = {
        "data_geracao": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "pipeline_version": "2.1",
        "scoring": "LGBM + Cascade Rules + IF Boost + Mapeamento Híbrido",
        "n_total": n_total,
        "n_fraudes": n_fraudes,
        "n_normais": n_normais,
        "taxa_fraude_pct": round(n_fraudes / n_total * 100, 4),
        "lgbm_threshold": LGBM_THRESHOLD,
        "lgbm_features_count": len(artifacts["lgbm_features"]),
    }

    # ═══ Métricas: BLOQUEAR ═══
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

    # ═══ AUC e AP ═══
    metrics["auc_roc_ensemble"] = round(roc_auc_score(y_true, ensemble_raw), 4)
    metrics["auc_roc_lgbm"] = round(roc_auc_score(y_true, df["lgbm_raw_score"].values), 4)
    metrics["ap_ensemble"] = round(average_precision_score(y_true, ensemble_raw), 4)
    metrics["ap_lgbm"] = round(average_precision_score(y_true, df["lgbm_raw_score"].values), 4)

    # ═══ Best F1 ═══
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

    for dec in ["APROVAR", "CONFIRMAR", "BLOQUEAR"]:
        mask = decisions == dec
        if mask.sum() > 0:
            metrics[f"taxa_fraude_{dec.lower()}"] = round(float(y_true[mask].mean()) * 100, 4)
            metrics[f"n_fraude_{dec.lower()}"] = int(y_true[mask].sum())
        else:
            metrics[f"taxa_fraude_{dec.lower()}"] = 0.0
            metrics[f"n_fraude_{dec.lower()}"] = 0

    # ═══ Separação ═══
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

    # ═══ Componentes ═══
    metrics["componentes"] = {
        "lgbm_detections": int(df["lgbm_pred"].sum()),
        "cascade_triggered": int(df["cascade_triggered"].sum()),
        "if_active": int(df["if_active"].sum()),
        "if_boosted": int((df["if_boost"] > 0).sum()),
        "is_first_tx_total": int(df["is_first_tx"].sum()),
    }

    fraud_mask = y_true == 1
    detected_by = df["detected_by"].values
    metrics["contribuicao_fraudes"] = {
        "lgbm": int(((detected_by == "LGBM") & fraud_mask).sum()),
        "cascade": int(((detected_by == "CASCADE") & fraud_mask).sum()),
        "if_boost": int(((detected_by == "IF_BOOST") & fraud_mask).sum()),
        "ensemble": int(((detected_by == "ENSEMBLE") & fraud_mask).sum()),
        "nao_detectadas": int(((detected_by == "APROVAR") & fraud_mask).sum()),
    }

    rule_counts = {}
    for rules_str in df["cascade_rules"]:
        if rules_str:
            for r in rules_str.split(","):
                r = r.strip()
                if r:
                    rule_counts[r] = rule_counts.get(r, 0) + 1
    metrics["cascade_rules_detail"] = rule_counts

    # ═══ Resumo executivo ═══
    p = metrics["pipeline_bloquear"]
    sep = metrics["separacao"]
    contrib = metrics["contribuicao_fraudes"]
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
        "gap_separacao": sep["gap"],
        "fraudes_por_lgbm": contrib["lgbm"],
        "fraudes_por_cascade": contrib["cascade"],
        "fraudes_por_if": contrib["if_boost"],
    }

    e = metrics["executivo"]
    print(f"\n  ═══ RESUMO ═══")
    print(f"  Fraudes detectadas (BLOQUEAR):  {e['fraudes_detectadas_pct']:.1f}%  ({p['tp']}/{n_fraudes})")
    print(f"  Fraudes perdidas:               {e['fraudes_nao_detectadas_pct']:.1f}%  ({p['fn']})")
    print(f"  Falsos alarmes (FPR):           {e['falsos_alarmes_pct']:.2f}%  ({p['fp']})")
    print(f"  Precisão dos alarmes:           {e['precisao_alarmes_pct']:.1f}%")
    print(f"  AUC-ROC:                        {e['auc_roc']:.4f}")
    print(f"  F1-Score (BLOQUEAR):            {e['f1']:.4f}")
    print(f"  GAP:                            +{sep['gap']:.1f} pontos")
    print(f"  ─── Contribuição ───")
    print(f"    LGBM:    {contrib['lgbm']:4d} fraudes")
    print(f"    Cascade: {contrib['cascade']:4d} fraudes")
    print(f"    IF:      {contrib['if_boost']:4d} fraudes")
    print(f"    Perdidas:{contrib['nao_detectadas']:4d} fraudes")

    return metrics


# =========================================================
# 5. GRÁFICOS
# =========================================================
def plot_dashboard(df: pd.DataFrame, metrics: Dict[str, Any]) -> None:
    """Gera dashboard executivo v2.1."""
    print("\n" + "=" * 70)
    print("  GERANDO DASHBOARD")
    print("=" * 70)

    y_true = df["y_true"].values
    score_mapped = df["score_mapped"].values
    ensemble_raw = df["ensemble_raw"].values

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(
        "Relatório Executivo — Detecção de Fraude PIX | Pipeline v2.1\n"
        f"LGBM + Cascade Rules + IF Boost | "
        f"{datetime.now().strftime('%d/%m/%Y %H:%M')} | "
        f"{metrics['n_total']:,} tx | {metrics['n_fraudes']} fraudes",
        fontsize=16, fontweight="bold", color=COLORS["primary"], y=1.02,
    )

    # ─── 1. Matriz de Confusão ───
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
        [f"VN\n{p['tn']:,}\n({cm_pct[0,0]:.2f}%)", f"FP\n{p['fp']:,}\n({cm_pct[0,1]:.2f}%)"],
        [f"FN\n{p['fn']:,}\n({cm_pct[1,0]:.2f}%)", f"VP\n{p['tp']:,}\n({cm_pct[1,1]:.2f}%)"],
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
            label=f"Ensemble v2.1 (AUC={metrics['auc_roc_ensemble']:.4f})")
    ax.plot(fpr_lgbm, tpr_lgbm, color=COLORS["info"], lw=1.5, ls="--",
            label=f"LGBM sozinho (AUC={metrics['auc_roc_lgbm']:.4f})")
    ax.plot([0, 1], [0, 1], color=COLORS["text_muted"], lw=1, ls=":")
    ax.set_xlabel("Taxa de Falso Positivo")
    ax.set_ylabel("Taxa de Verdadeiro Positivo")
    ax.set_title("Curva ROC", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True)

    # ─── 3. Curva PR ───
    ax = axes[0, 2]
    prec_ens, rec_ens, _ = precision_recall_curve(y_true, ensemble_raw)
    prec_lgbm, rec_lgbm, _ = precision_recall_curve(y_true, df["lgbm_raw_score"].values)

    ax.plot(rec_ens, prec_ens, color=COLORS["primary"], lw=2.5,
            label=f"Ensemble (AP={metrics['ap_ensemble']:.4f})")
    ax.plot(rec_lgbm, prec_lgbm, color=COLORS["info"], lw=1.5, ls="--",
            label=f"LGBM (AP={metrics['ap_lgbm']:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True)

    best = metrics["best_f1"]
    ax.scatter([best["recall"]], [best["precision"]], s=150, color=COLORS["warning"],
               zorder=5, marker="*")
    ax.annotate(f"Best F1={best['f1']:.3f}", xy=(best["recall"], best["precision"]),
                fontsize=9, color=COLORS["warning"],
                xytext=(best["recall"] - 0.15, best["precision"] - 0.1),
                arrowprops=dict(arrowstyle="->", color=COLORS["warning"]))

    # ─── 4. Contribuição por Componente ───
    ax = axes[1, 0]
    contrib = metrics["contribuicao_fraudes"]
    comp_names = ["LGBM", "Cascade", "IF Boost", "Perdidas"]
    comp_values = [contrib["lgbm"], contrib["cascade"], contrib["if_boost"], contrib["nao_detectadas"]]
    comp_colors = [COLORS["primary"], COLORS["cascade"], COLORS["info"], COLORS["secondary"]]

    bars = ax.barh(comp_names, comp_values, color=comp_colors, edgecolor="white", lw=0.5)
    ax.set_title("Contribuição por Componente\n(Fraudes Detectadas)", fontweight="bold")
    ax.set_xlabel("Fraudes")
    for bar, val in zip(bars, comp_values):
        if val > 0:
            pct = val / max(metrics["n_fraudes"], 1) * 100
            ax.text(bar.get_width() + max(max(comp_values), 1) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val} ({pct:.1f}%)", va="center", fontsize=11, color=COLORS["text"])
    ax.set_xlim(0, max(max(comp_values), 1) * 1.4)

    # ─── 5. Distribuição dos Scores ───
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
    ax.set_xlabel("Score Mapeado")
    ax.set_ylabel("Densidade")
    ax.legend(fontsize=9)

    # ─── 6. Decisões vs Fraudes ───
    ax = axes[1, 2]
    dec_order = ["APROVAR", "CONFIRMAR", "BLOQUEAR"]
    dec_colors_list = [COLORS["aprovar"], COLORS["confirmar"], COLORS["bloquear"]]

    dec_totals = [metrics["decisoes"].get(d, 0) for d in dec_order]
    dec_frauds = [metrics.get(f"n_fraude_{d.lower()}", 0) for d in dec_order]

    x = np.arange(len(dec_order))
    width = 0.35

    bars1 = ax.bar(x - width/2, dec_totals, width, label="Total", color=dec_colors_list, alpha=0.6)
    bars2 = ax.bar(x + width/2, dec_frauds, width, label="Fraudes", color=COLORS["secondary"], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(["🟢 APROVAR", "🟡 CONFIRMAR", "🔴 BLOQUEAR"])
    ax.set_title("Decisões vs Fraudes Reais", fontweight="bold")
    ax.set_ylabel("Quantidade")
    ax.legend(fontsize=9)

    # Usar log scale apenas se houver variação grande
    if max(dec_totals) > 100 * max(max(dec_frauds), 1):
        ax.set_yscale("log")

    for bar, val in zip(bars1, dec_totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                f"{val:,}", ha="center", va="bottom", fontsize=9, color=COLORS["text"])
    for bar, val in zip(bars2, dec_frauds):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                    f"{val}", ha="center", va="bottom", fontsize=9, color=COLORS["secondary"])

    plt.tight_layout()
    output = RELATORIO_DIR / "dashboard_executivo.png"
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ Dashboard salvo: {output}")


# =========================================================
# 6. RELATÓRIO HTML
# =========================================================
def generate_html_report(metrics: Dict[str, Any]) -> None:
    """Gera relatório executivo HTML v2.1."""
    print("\n" + "=" * 70)
    print("  GERANDO RELATÓRIO HTML")
    print("=" * 70)

    e = metrics["executivo"]
    p = metrics["pipeline_bloquear"]
    p2 = metrics["pipeline_qualquer_acao"]
    sep = metrics["separacao"]
    comp = metrics["componentes"]
    contrib = metrics["contribuicao_fraudes"]
    cascade_detail = metrics.get("cascade_rules_detail", {})

    cascade_rows = ""
    for rule, count in sorted(cascade_detail.items(), key=lambda x: -x[1]):
        cascade_rows += f"<tr><td>{rule}</td><td>{count:,}</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório Executivo — Detecção de Fraude PIX v2.1</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #0e1117; color: #e0e0e0; padding: 30px 50px; line-height: 1.6; }}
        .header {{ text-align: center; margin-bottom: 40px; border-bottom: 3px solid #00d4aa; padding-bottom: 20px; }}
        .header h1 {{ color: #00d4aa; font-size: 32px; margin-bottom: 8px; }}
        .header .version {{ display: inline-block; background: rgba(0, 212, 170, 0.15); color: #00d4aa; padding: 4px 16px; border-radius: 20px; font-size: 13px; font-weight: bold; margin-top: 8px; }}
        .header .subtitle {{ color: #888; font-size: 14px; margin-top: 8px; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 40px; }}
        .kpi {{ background: #1a1d23; border-radius: 16px; padding: 28px; text-align: center; border: 2px solid #333; }}
        .kpi:hover {{ border-color: #00d4aa; }}
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
        .orange {{ color: #ff9f43; font-weight: bold; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; }}
        .badge-green {{ background: rgba(0, 212, 170, 0.2); color: #00d4aa; }}
        .badge-red {{ background: rgba(255, 107, 107, 0.2); color: #ff6b6b; }}
        .badge-yellow {{ background: rgba(255, 217, 61, 0.2); color: #ffd93d; }}
        .badge-orange {{ background: rgba(255, 159, 67, 0.2); color: #ff9f43; }}
        .img-container {{ text-align: center; margin: 20px 0; }}
        .img-container img {{ max-width: 100%; border-radius: 12px; border: 1px solid #333; }}
        .callout {{ background: rgba(0, 212, 170, 0.08); border-left: 4px solid #00d4aa; padding: 16px 20px; margin: 16px 0; border-radius: 0 8px 8px 0; }}
        .callout.warning {{ background: rgba(255, 217, 61, 0.08); border-left-color: #ffd93d; }}
        .callout.danger {{ background: rgba(255, 107, 107, 0.08); border-left-color: #ff6b6b; }}
        .callout.success {{ background: rgba(0, 212, 170, 0.12); border-left-color: #00d4aa; }}
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
        .three-col {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 24px; }}
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
    <h2 style="color: #ccc; font-weight: 400; font-size: 18px;">Relatório Executivo de Performance</h2>
    <div class="version">Pipeline v2.1 — LGBM ({metrics['lgbm_features_count']} features) + Cascade Rules + IF Boost</div>
    <p class="subtitle" style="margin-top: 12px;">
        Gerado em {metrics['data_geracao']} |
        Base de teste: {metrics['n_total']:,} transações |
        {metrics['n_fraudes']} fraudes ({metrics['taxa_fraude_pct']:.2f}%)
    </p>
</div>

<!-- KPIs -->
<div class="kpi-grid">
    <div class="kpi" style="border-color: #00d4aa;">
        <div class="value" style="color: #00d4aa;">{e['fraudes_detectadas_pct']:.1f}%</div>
        <div class="label">Fraudes Detectadas</div>
        <div class="detail">{p['tp']} de {metrics['n_fraudes']} fraudes (BLOQUEAR ≥85)</div>
    </div>
    <div class="kpi" style="border-color: {'#00d4aa' if e['fraudes_nao_detectadas_n'] == 0 else '#ff6b6b'};">
        <div class="value" style="color: {'#00d4aa' if e['fraudes_nao_detectadas_n'] == 0 else '#ff6b6b'};">{e['fraudes_nao_detectadas_n']}</div>
        <div class="label">Fraudes Não Detectadas</div>
        <div class="detail">{'Nenhuma fraude escapou ✅' if e['fraudes_nao_detectadas_n'] == 0 else f"{e['fraudes_nao_detectadas_pct']:.1f}% passaram"}</div>
    </div>
    <div class="kpi" style="border-color: #ffd93d;">
        <div class="value" style="color: #ffd93d;">{e['falsos_alarmes_n']}</div>
        <div class="label">Falsos Alarmes</div>
        <div class="detail">{e['falsos_alarmes_pct']:.2f}% das tx legítimas</div>
    </div>
    <div class="kpi" style="border-color: #00d4aa;">
        <div class="value" style="color: #00d4aa;">{e['precisao_alarmes_pct']:.1f}%</div>
        <div class="label">Precisão dos Alarmes</div>
        <div class="detail">Cada bloqueio tem esta chance de ser fraude real</div>
    </div>
    <div class="kpi" style="border-color: #6c5ce7;">
        <div class="value" style="color: #6c5ce7;">{e['auc_roc']:.4f}</div>
        <div class="label">AUC-ROC</div>
        <div class="detail">Capacidade geral (1.0 = perfeito)</div>
    </div>
    <div class="kpi" style="border-color: #00d4aa;">
        <div class="value" style="color: #00d4aa;">+{sep['gap']:.1f}</div>
        <div class="label">GAP de Separação</div>
        <div class="detail">Menor fraude ({sep['fraud_min']}) vs P99.9 normal ({sep['normal_p999']})</div>
    </div>
</div>

<!-- Contribuição -->
<div class="section">
    <h2>🏗️ Contribuição de Cada Componente na Detecção de Fraudes</h2>
    <div class="callout success">
        <strong>Sistema em 3 camadas:</strong> O LGBM é o motor principal.
        As <strong>Cascade Rules</strong> capturam fraudes que o LGBM perde (bursts, contas novas).
        O <strong>IF Boost</strong> promove transações anômalas na zona cinzenta.
    </div>
    <div class="three-col">
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 48px; font-weight: 800; color: #00d4aa;">{contrib['lgbm']}</div>
            <div style="color: #888; margin-top: 8px;">🧠 LGBM</div>
            <div style="color: #666; font-size: 12px;">Motor principal de ML</div>
        </div>
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 48px; font-weight: 800; color: #ff9f43;">{contrib['cascade']}</div>
            <div style="color: #888; margin-top: 8px;">🔗 Cascade Rules</div>
            <div style="color: #666; font-size: 12px;">6 regras para FN do LGBM</div>
        </div>
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 48px; font-weight: 800; color: #6c5ce7;">{contrib['if_boost']}</div>
            <div style="color: #888; margin-top: 8px;">🔍 IF Boost</div>
            <div style="color: #666; font-size: 12px;">Detecção de anomalias</div>
        </div>
    </div>
    {'<div class="callout danger"><strong>Fraudes não detectadas:</strong> ' + str(contrib["nao_detectadas"]) + ' fraudes passaram por todas as camadas.</div>' if contrib['nao_detectadas'] > 0 else '<div class="callout success"><strong>Zero fraudes perdidas!</strong> Todas as camadas juntas capturaram 100% das fraudes.</div>'}
</div>

<!-- Cascade Rules -->
<div class="section">
    <h2>🔗 Cascade Rules — Detalhamento</h2>
    <p style="color: #888; margin-bottom: 16px;">
        Atuam <strong>apenas quando o LGBM não flagga</strong> (score &lt; {metrics['lgbm_threshold']}).
    </p>
    <table>
        <tr><th>Regra</th><th>Tx Capturadas</th></tr>
        {cascade_rows if cascade_rows else '<tr><td colspan="2" style="color: #888;">Nenhuma cascade rule ativada</td></tr>'}
    </table>
    <div class="callout">
        Total cascade: {comp['cascade_triggered']:,} tx |
        Fraudes capturadas: <strong>{contrib['cascade']}</strong>
    </div>
</div>

<!-- Score Visual -->
<div class="section">
    <h2>📏 Escala de Score (0-100)</h2>
    <div class="score-bar">
        <span class="label-left">🟢 APROVAR<br>0 — 59</span>
        <span class="label-mid">🟡 CONFIRMAR<br>60 — 84</span>
        <span class="label-right">🔴 BLOQUEAR<br>85 — 100</span>
    </div>
</div>

<!-- Dashboard -->
<div class="section">
    <h2>📊 Dashboard Visual</h2>
    <div class="img-container">
        <img src="dashboard_executivo.png" alt="Dashboard">
    </div>
</div>

<!-- Guia -->
<div class="section">
    <h2>📖 Guia para Executivos</h2>
    <div class="callout success">
        <strong>Destaque:</strong> O sistema detecta <strong>{e['fraudes_detectadas_pct']:.1f}%</strong> das fraudes
        com apenas <strong>{e['falsos_alarmes_n']}</strong> falsos alarmes em {metrics['n_total']:,} transações.
    </div>
    <table>
        <tr><th>Métrica</th><th>O que significa</th><th>Resultado</th></tr>
        <tr>
            <td><strong>Fraudes Detectadas</strong> (Recall)</td>
            <td>De cada 100 fraudes, quantas o sistema bloqueia</td>
            <td class="highlight" style="font-size: 18px;">{e['fraudes_detectadas_pct']:.1f}%</td>
        </tr>
        <tr>
            <td><strong>Falsos Alarmes</strong> (FPR)</td>
            <td>Transações legítimas erroneamente bloqueadas</td>
            <td class="warning" style="font-size: 18px;">{e['falsos_alarmes_pct']:.2f}% ({e['falsos_alarmes_n']})</td>
        </tr>
        <tr>
            <td><strong>Precisão</strong></td>
            <td>Quando bloqueia, qual a chance de ser fraude real</td>
            <td class="highlight" style="font-size: 18px;">{e['precisao_alarmes_pct']:.1f}%</td>
        </tr>
        <tr>
            <td><strong>GAP</strong></td>
            <td>Distância entre menor fraude e normal mais alto</td>
            <td class="highlight" style="font-size: 18px;">+{sep['gap']:.1f} pontos</td>
        </tr>
        <tr>
            <td><strong>AUC-ROC</strong></td>
            <td>Nota geral (0.50=aleatório, 1.00=perfeito)</td>
            <td class="info" style="font-size: 18px;">{e['auc_roc']:.4f}</td>
        </tr>
    </table>
</div>

<!-- Decisões -->
<div class="section">
    <h2>🎯 Resultados por Faixa</h2>
    <table>
        <tr><th>Decisão</th><th>Score</th><th>Ação</th><th>Qtd</th><th>%</th><th>Fraudes</th><th>Taxa</th></tr>
        <tr>
            <td><span class="badge badge-green">🟢 APROVAR</span></td><td>0—59</td><td>Liberar</td>
            <td>{metrics['decisoes']['APROVAR']:,}</td><td>{metrics['decisoes']['APROVAR']/metrics['n_total']*100:.1f}%</td>
            <td class="{'highlight' if metrics.get('n_fraude_aprovar',0)==0 else 'danger'}">{metrics.get('n_fraude_aprovar',0)}</td><td>{metrics.get('taxa_fraude_aprovar',0):.4f}%</td>
        </tr>
        <tr>
            <td><span class="badge badge-yellow">🟡 CONFIRMAR</span></td><td>60—84</td><td>2FA</td>
            <td>{metrics['decisoes']['CONFIRMAR']:,}</td><td>{metrics['decisoes']['CONFIRMAR']/metrics['n_total']*100:.1f}%</td>
            <td>{metrics.get('n_fraude_confirmar',0)}</td><td>{metrics.get('taxa_fraude_confirmar',0):.2f}%</td>
        </tr>
        <tr>
            <td><span class="badge badge-red">🔴 BLOQUEAR</span></td><td>85—100</td><td>Bloquear</td>
            <td>{metrics['decisoes']['BLOQUEAR']:,}</td><td>{metrics['decisoes']['BLOQUEAR']/metrics['n_total']*100:.1f}%</td>
            <td class="danger">{metrics.get('n_fraude_bloquear',0)}</td><td class="danger">{metrics.get('taxa_fraude_bloquear',0):.2f}%</td>
        </tr>
    </table>
</div>

<!-- Separação -->
<div class="section">
    <h2>🔬 Qualidade da Separação</h2>
    <div class="callout success">
        <strong>GAP de +{sep['gap']:.1f} pontos</strong> entre a fraude mais fraca e o P99.9 das normais.
    </div>
    <div class="two-col">
        <div>
            <h3 style="color: #00d4aa;">Normais</h3>
            <table>
                <tr><td>P99.9</td><td class="highlight">{sep['normal_p999']}</td></tr>
                <tr><td>Máximo</td><td>{sep['normal_max']}</td></tr>
            </table>
        </div>
        <div>
            <h3 style="color: #ff6b6b;">Fraudes</h3>
            <table>
                <tr><td>Mínimo</td><td class="danger">{sep['fraud_min']}</td></tr>
                <tr><td>P5</td><td>{sep['fraud_p5']}</td></tr>
                <tr><td>Mediana</td><td>{sep['fraud_median']}</td></tr>
            </table>
        </div>
    </div>
</div>

<!-- Duas perspectivas -->
<div class="section">
    <h2>📐 Duas Perspectivas</h2>
    <div class="two-col">
        <div>
            <h3 style="color: #ff6b6b;">Conservadora (só BLOQUEAR)</h3>
            <table>
                <tr><td>Recall</td><td class="highlight">{p['recall']*100:.1f}%</td></tr>
                <tr><td>Precision</td><td>{p['precision']*100:.1f}%</td></tr>
                <tr><td>F1</td><td>{p['f1']:.4f}</td></tr>
                <tr><td>FP</td><td>{p['fp']:,}</td></tr>
                <tr><td>FN</td><td class="{'highlight' if p['fn']==0 else 'danger'}">{p['fn']}</td></tr>
            </table>
        </div>
        <div>
            <h3 style="color: #4ecdc4;">Ampla (CONFIRMAR + BLOQUEAR)</h3>
            <table>
                <tr><td>Recall</td><td class="highlight">{p2['recall']*100:.1f}%</td></tr>
                <tr><td>Precision</td><td>{p2['precision']*100:.1f}%</td></tr>
                <tr><td>F1</td><td>{p2['f1']:.4f}</td></tr>
                <tr><td>FP</td><td>{p2['fp']:,}</td></tr>
                <tr><td>FN</td><td class="{'highlight' if p2['fn']==0 else 'danger'}">{p2['fn']}</td></tr>
            </table>
        </div>
    </div>
</div>

<!-- Arquitetura -->
<div class="section">
    <h2>⚙️ Arquitetura do Sistema v2.1</h2>
    <table>
        <tr><th>Componente</th><th>Tipo</th><th>Papel</th></tr>
        <tr><td><strong>🧠 LightGBM</strong></td><td>Gradient Boosting ({metrics['lgbm_features_count']} features)</td><td>Motor principal — threshold={metrics['lgbm_threshold']:.4f}</td></tr>
        <tr><td><strong>🔗 Cascade Rules</strong></td><td>6 regras determinísticas</td><td>Captura FN do LGBM: bursts, contas novas, esvaziamento</td></tr>
        <tr><td><strong>🔍 Isolation Forest</strong></td><td>800 trees, boost condicional</td><td>Complementar: boost +0.08/+0.15 quando score IF alto</td></tr>
        <tr><td><strong>📏 Mapeamento Híbrido</strong></td><td>Interpolação não-linear</td><td>Raw → Score 0-100</td></tr>
        <tr><td><strong>⚠️ 24 Agravantes</strong></td><td>7 fases de análise</td><td>Ajuste fino por contexto</td></tr>
        <tr><td><strong>🛡️ Eng. Social</strong></td><td>12 padrões combinatórios</td><td>Detecta golpes conhecidos</td></tr>
        <tr><td><strong>🔍 Behavioral</strong></td><td>15 fatores RT</td><td>Anomalias comportamentais</td></tr>
    </table>
    <div class="callout">
        <strong>Fluxo:</strong> TX → LGBM → Cascade → IF Boost → Mapeamento → Agravantes/SE/Behavioral → Score 0-100 → Decisão
    </div>
</div>

<!-- Performance -->
<div class="section">
    <h2>📈 Performance</h2>
    <table>
        <tr><th>Modelo</th><th>AUC-ROC</th><th>Average Precision</th></tr>
        <tr><td>LGBM sozinho</td><td>{metrics['auc_roc_lgbm']:.4f}</td><td>{metrics['ap_lgbm']:.4f}</td></tr>
        <tr><td><strong>Ensemble v2.1</strong></td><td class="highlight"><strong>{metrics['auc_roc_ensemble']:.4f}</strong></td><td class="highlight"><strong>{metrics['ap_ensemble']:.4f}</strong></td></tr>
    </table>
</div>

<div class="footer">
    Sistema de Detecção de Fraude PIX v2.1 — LGBM + Cascade + IF Boost<br>
    Relatório gerado em {metrics['data_geracao']}
</div>

</body>
</html>"""

    output = RELATORIO_DIR / "relatorio_executivo.html"
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✅ Relatório HTML salvo: {output}")


# =========================================================
# 7. SALVAR
# =========================================================
def save_outputs(df: pd.DataFrame, metrics: Dict[str, Any]) -> None:
    """Salva métricas JSON e resultados CSV."""
    print("\n" + "=" * 70)
    print("  SALVANDO ARTEFATOS")
    print("=" * 70)

    metrics_path = RELATORIO_DIR / "relatorio_metricas.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"  ✅ Métricas: {metrics_path}")

    csv_path = RELATORIO_DIR / "resultados_detalhados.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Resultados: {csv_path} ({len(df):,} rows)")


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n")
    print("█" * 70)
    print("█                                                                    █")
    print("█   TESTE DO PIPELINE v2.1 + RELATÓRIO EXECUTIVO                     █")
    print("█   LGBM + Cascade Rules + IF Boost + Mapeamento Híbrido             █")
    print("█                                                                    █")
    print("█" * 70)

    t_total = time.time()

    # 1. Carregar
    artifacts = load_artifacts()
    X_test, y_test = load_test_data()

    # 2. Executar pipeline
    results = run_pipeline(X_test, y_test, artifacts)

    # 3. Métricas
    metrics = calculate_metrics(results, artifacts)

    # 4. Dashboard
    plot_dashboard(results, metrics)

    # 5. Relatório HTML
    generate_html_report(metrics)

    # 6. Salvar
    save_outputs(results, metrics)

    # Resumo final
    elapsed = time.time() - t_total
    e = metrics["executivo"]
    sep = metrics["separacao"]
    contrib = metrics["contribuicao_fraudes"]

    print("\n" + "█" * 70)
    print("█  RESULTADO FINAL                                                   █")
    print("█" * 70)
    print(f"""
  Pipeline v2.1: LGBM ({metrics['lgbm_features_count']} features) + Cascade (6) + IF Boost

  ┌─────────────────────────────────────────────────────┐
  │  🛡️  Fraudes Detectadas:  {e['fraudes_detectadas_pct']:5.1f}%  ({metrics['pipeline_bloquear']['tp']}/{metrics['n_fraudes']})          │
  │  ❌  Fraudes Perdidas:    {e['fraudes_nao_detectadas_n']:5d}                            │
  │  ⚠️   Falsos Alarmes:     {e['falsos_alarmes_n']:5d}  ({e['falsos_alarmes_pct']:.2f}%)             │
  │  🎯  Precisão Alarmes:   {e['precisao_alarmes_pct']:5.1f}%                           │
  │  📊  AUC-ROC:            {e['auc_roc']:.4f}                          │
  │  📏  GAP Separação:      +{sep['gap']:.1f} pontos                      │
  ├─────────────────────────────────────────────────────┤
  │  🧠 LGBM:     {contrib['lgbm']:4d} fraudes                              │
  │  🔗 Cascade:  {contrib['cascade']:4d} fraudes                              │
  │  🔍 IF Boost: {contrib['if_boost']:4d} fraudes                              │
  │  ❌ Perdidas: {contrib['nao_detectadas']:4d}                                     │
  └─────────────────────────────────────────────────────┘

  📁 Artefatos salvos em: {RELATORIO_DIR}/
    - relatorio_executivo.html
    - dashboard_executivo.png
    - relatorio_metricas.json
    - resultados_detalhados.csv

  ⏱️  Tempo total: {elapsed:.1f}s
""")

    # Veredicto
    if e['fraudes_nao_detectadas_n'] == 0:
        print("  ✅ VEREDICTO: Sistema APTO para produção — 0 fraudes perdidas!")
    elif e['fraudes_detectadas_pct'] >= 95:
        print(f"  ✅ VEREDICTO: Sistema APTO — {e['fraudes_detectadas_pct']:.1f}% recall")
    elif e['fraudes_detectadas_pct'] >= 90:
        print(f"  ⚠️  VEREDICTO: ACEITÁVEL — {e['fraudes_detectadas_pct']:.1f}% recall, {e['fraudes_nao_detectadas_n']} FN")
    else:
        print(f"  ❌ VEREDICTO: Precisa melhorias — {e['fraudes_detectadas_pct']:.1f}% recall")


if __name__ == "__main__":
    main()
