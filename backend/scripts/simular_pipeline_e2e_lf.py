# scripts/simular_pipeline_e2e_lf.py — PATCH v3.0.3 → v3.0.5
# Atualiza simulação E2E para espelhar decision_engine.py v3.0.5
#
# Mudanças:
#   1. threshold_confirmar: 80.0 → 77.0
#   2. Veto #8 (VETO_SE_BEH_VALOR_NOVO) adicionado
#   3. Cascade C3 com LGBM guard (>= 0.35)
#   4. Fast-Approve override (FA-05): LGBM < 0.25 + SE=0 + BEH=0
#   5. IF extremo LGBM gate: 0.05 → 0.25
#   6. engine_version → "3.0.5"

"""
scripts/simular_pipeline_e2e_lf.py — Simulação End-to-End do Pipeline Antifraude

Roda o pipeline COMPLETO no dataset leakage-free (100.355 tx, 355 fraudes),
reproduzindo exatamente o fluxo do decision_engine.py v3.0.5 com:
  - LGBM v5.1 (predict real via .joblib)
  - IF v3 (predict real via .joblib)
  - SE v3.4 (rule-based, roda direto)
  - BEH v3.1 (rule-based, roda direto)
  - Engine v3.0.5 (cascade v3 + fast-approve + vetos + agravantes)

Mudanças v3.0.3 → v3.0.5:
  - threshold_confirmar: 80 → 77 (v3.0.4)
  - Veto VETO_SE_BEH_VALOR_NOVO adicionado (v3.0.4)
  - Cascade C3 com LGBM guard >= 0.35 (v3.0.5)
  - Fast-Approve override: LGBM < 0.25 + SE=0 + BEH=0 (v3.0.5)
  - IF extremo LGBM gate reforçado: 0.05 → 0.25 (v3.0.5)

NÃO depende do pipeline_orquestrador.py — roda independente para evitar
efeitos colaterais do cache de histórico por cliente.

Análises geradas:
  A. Métricas globais (P/R/F1 por decisão e por componente)
  B. Contribuição marginal (cada componente sozinho vs ensemble)
  C. Overlap matrix (Venn: quem pega quem)
  D. FN analysis (fraudes invisíveis — perfil)
  E. FP analysis (falsos positivos — quem gera)
  F. Threshold sweep (precision-recall tradeoff)

Uso:
    cd E:/Projetos/rebuild_pix/backend
    python scripts/simular_pipeline_e2e_lf.py

Autor: AI Engineer + Adilio
Data: 2026-04-12
Versão: 2.0 (sync com decision_engine v3.0.5)
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ─── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Paths ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # backend/
BACKEND_DIR = PROJECT_ROOT
ARTEFATOS_DIR = BACKEND_DIR / "artefatos"
DADOS_DIR = PROJECT_ROOT.parent / "dados"
DATASET_PATH = DADOS_DIR / "base_mvp_model_ready_leakage_free.csv"

# Garantir imports do projeto
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Output
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = PROJECT_ROOT.parent / "resultados" / f"simulacao_e2e_lf_{TIMESTAMP}"

# ─── Versões (SYNC com decision_engine.py) ─────────────────
ENGINE_VERSION = "3.0.5"
SE_VERSION = "3.4"
BEH_VERSION = "3.1"

# ─── Thresholds (SYNC com EngineConfig v3.0.5) ────────────
THRESHOLD_CONFIRMAR = 77.0   # v3.0.4: era 80.0
THRESHOLD_BLOQUEAR = 95.0
VETO_THRESHOLD = 0.90

# ─── Cascade v3 constants (SYNC com v3.0.5) ───────────────
CASCADE_BURST_MIN_TX = 3
CASCADE_IF_THRESHOLD = 0.995
CASCADE_C3_LGBM_MIN = 0.35   # NOVO v3.0.5: LGBM guard para C3

# ─── Fast-Approve constants (NOVO v3.0.5) ─────────────────
FA_LGBM_MAX = 0.25
LGBM_MIN_FOR_IF_VETO = 0.10
LGBM_MIN_FOR_IF_EXTREME = 0.25  # v3.0.5: era 0.05


# =========================================================
# 1. CARREGAMENTO DE ARTEFATOS
# =========================================================
class ArtefatosLoader:
    """Carrega modelos e configs do disco."""

    def __init__(self, art_dir: Path = ARTEFATOS_DIR) -> None:
        self.art_dir = art_dir
        self.lgbm_model = None
        self.lgbm_features: list[str] = []
        self.if_model = None
        self.if_scaler = None
        self.if_features: list[str] = []
        self.if_medians: dict[str, float] = {}
        self.if_ref_scores: np.ndarray | None = None
        self.anchors_raw: np.ndarray = np.array([0.0, 1.0])
        self.anchors_out: np.ndarray = np.array([0.0, 100.0])
        self.lgbm_threshold: float = 0.35
        self.lgbm_effective_threshold: float = 0.40

    def load_all(self) -> None:
        """Carrega todos os artefatos necessários."""
        t0 = time.perf_counter()
        art = self.art_dir

        # LGBM
        lgbm_path = art / "model_lightgbm.joblib"
        if lgbm_path.exists():
            self.lgbm_model = joblib.load(lgbm_path)
            logger.info(f"LGBM carregado: {type(self.lgbm_model).__name__}")
        else:
            raise FileNotFoundError(f"LGBM não encontrado: {lgbm_path}")

        # LGBM Features
        feat_path = art / "lgbm_features.json"
        if feat_path.exists():
            with open(feat_path) as f:
                data = json.load(f)
            self.lgbm_features = (
                data["features"] if isinstance(data, dict) else data
            )
            logger.info(f"LGBM features: {len(self.lgbm_features)}")

        # Scoring Config
        scoring_path = art / "scoring_config.json"
        if scoring_path.exists():
            with open(scoring_path, encoding="utf-8") as f:
                sc = json.load(f)
            mapeamento = sc.get("mapeamento", {})
            self.anchors_raw = np.array(
                mapeamento.get("anchors_raw", [0.0, 1.0]), dtype=np.float64
            )
            self.anchors_out = np.array(
                mapeamento.get("anchors_out", [0.0, 100.0]), dtype=np.float64
            )

        # Thresholds
        th_path = art / "thresholds_config.json"
        if th_path.exists():
            with open(th_path, encoding="utf-8") as f:
                th = json.load(f)
            self.lgbm_threshold = float(
                th.get("threshold_f1_best", self.lgbm_threshold)
            )
            self.lgbm_effective_threshold = max(
                self.lgbm_effective_threshold, self.lgbm_threshold + 0.05
            )

        # Isolation Forest
        if_path = art / "model_isolation_forest.joblib"
        if if_path.exists():
            self.if_model = joblib.load(if_path)
            logger.info(f"IF carregado: {self.if_model.n_estimators} trees")

        scaler_path = art / "scaler_isolation_forest.joblib"
        if scaler_path.exists():
            self.if_scaler = joblib.load(scaler_path)

        config_path = art / "isolation_forest_config.json"
        if config_path.exists():
            with open(config_path) as f:
                ifc = json.load(f)
            self.if_features = ifc.get("features", [])
            self.if_medians = ifc.get("medians", {})

        ref_path = art / "if_ref_raw_train.npy"
        if ref_path.exists():
            self.if_ref_scores = np.load(ref_path)
            logger.info(f"IF ref scores: {len(self.if_ref_scores)}")

        # v3.0.5: Usar features do scaler como fonte de verdade
        if self.if_scaler is not None and hasattr(self.if_scaler, "feature_names_in_"):
            scaler_features = list(self.if_scaler.feature_names_in_)
            if set(scaler_features) != set(self.if_features):
                logger.warning(
                    "⚠️ Features do scaler DIFEREM do config! "
                    "Usando scaler como fonte de verdade."
                )
                logger.info(f"IF scaler features ({len(scaler_features)}): {scaler_features}")
                self.if_features = scaler_features

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"Artefatos carregados em {elapsed:.0f}ms")


# =========================================================
# 2. SCORING FUNCTIONS
# =========================================================
def score_lgbm_batch(
    df: pd.DataFrame, loader: ArtefatosLoader,
) -> np.ndarray:
    """Calcula scores LGBM para todo o DataFrame."""
    features = loader.lgbm_features
    X = df[features].copy().fillna(0.0)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    proba = loader.lgbm_model.predict_proba(X)[:, 1]
    return np.clip(proba, 0.0, 1.0)


def create_if_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de interação para IF v3 (batch)."""
    tx_30m = df["tx_count_prev_30m"].fillna(0)
    vl_pix = df["vl_pix"].fillna(0).clip(lower=0)
    nr_idade = df["nr_idade"].fillna(0)
    first_recv = df["first_receiver_flag"].fillna(0)
    distinct_recv = df["distinct_receivers_so_far"].fillna(1)

    return pd.DataFrame({
        "valor_x_burst": vl_pix * (tx_30m + 1),
        "idade_x_first_recv": nr_idade * first_recv,
        "burst_x_distinct_recv": tx_30m * distinct_recv,
        "log_vl_pix": np.log1p(vl_pix),
    }, index=df.index)


def score_if_batch(
    df: pd.DataFrame, loader: ArtefatosLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula IF percentile e raw score para todo o DataFrame."""
    if loader.if_model is None:
        n = len(df)
        return np.zeros(n), np.zeros(n)

    interaction = create_if_interaction_features(df)
    df_combined = pd.concat([df, interaction], axis=1)

    X = pd.DataFrame(index=df.index)
    for feat in loader.if_features:
        if feat in df_combined.columns:
            vals = pd.to_numeric(df_combined[feat], errors="coerce")
            median = loader.if_medians.get(feat, 0)
            vals = vals.fillna(median).replace([np.inf, -np.inf], median)
            X[feat] = vals.astype(float)
        else:
            median = loader.if_medians.get(feat, 0)
            X[feat] = median
            logger.warning(
                f"  IF feature '{feat}' não encontrada — usando median={median}"
            )

    if loader.if_scaler is not None:
        X_scaled = loader.if_scaler.transform(X)
    else:
        X_scaled = X.values

    raw_scores = loader.if_model.decision_function(X_scaled)

    if loader.if_ref_scores is not None and len(loader.if_ref_scores) > 0:
        inverted = -raw_scores
        ref_inverted = -loader.if_ref_scores
        ref_sorted = np.sort(ref_inverted)
        percentiles = np.searchsorted(ref_sorted, inverted) / len(ref_sorted)
    else:
        percentiles = 1.0 / (1.0 + np.exp(raw_scores * 5))

    percentiles = np.clip(percentiles, 0, 1)
    return percentiles, raw_scores


# =========================================================
# 3. CASCADE RULES v3 (com LGBM guard)
# =========================================================
def apply_cascade_batch(
    df: pd.DataFrame,
    if_percentiles: np.ndarray,
    lgbm_raw: np.ndarray,
) -> pd.DataFrame:
    """
    Aplica cascade rules v3 em batch.

    Mudanças v2 → v3:
      C1: INALTERADO (burst >= 3 → BLOQUEAR)
      C3: ADICIONADO lgbm >= 0.35 guard (v3.0.5)

    Returns:
        DataFrame com colunas: cascade_c1, cascade_c3, cascade_action
    """
    tx_30m = df["tx_count_prev_30m"].fillna(0).astype(int)
    burst = df["burst_30m_flag"].fillna(0).astype(int)

    # C1: burst >= 3 → BLOQUEAR (inalterado)
    c1 = tx_30m >= CASCADE_BURST_MIN_TX

    # C3: burst=1 AND tx_count < 3 AND IF >= threshold AND LGBM >= 0.35
    c3 = (
        (burst == 1)
        & (tx_30m < CASCADE_BURST_MIN_TX)
        & (if_percentiles >= CASCADE_IF_THRESHOLD)
        & (lgbm_raw >= CASCADE_C3_LGBM_MIN)  # NOVO v3.0.5
    )

    action = pd.Series("NONE", index=df.index)
    action[c3] = "CONFIRMAR"
    action[c1] = "BLOQUEAR"  # C1 tem prioridade

    return pd.DataFrame({
        "cascade_c1": c1,
        "cascade_c3": c3,
        "cascade_action": action,
    }, index=df.index)


# =========================================================
# 4. ENSEMBLE v3.0.5
# =========================================================
def calculate_ensemble_batch(
    lgbm_raw: np.ndarray,
    cascade_action: pd.Series,
    effective_threshold: float = 0.40,
) -> np.ndarray:
    """Calcula ensemble raw (0-1) em batch."""
    ensemble = lgbm_raw.copy()

    cascade_mask = cascade_action.isin(["BLOQUEAR", "CONFIRMAR"]).values
    below_threshold = lgbm_raw < effective_threshold

    promote_mask = cascade_mask & below_threshold
    ensemble[promote_mask] = effective_threshold

    return ensemble


def map_to_score_batch(
    raw: np.ndarray,
    anchors_raw: np.ndarray,
    anchors_out: np.ndarray,
) -> np.ndarray:
    """Mapeamento não-linear: raw (0-1) → score (0-100)."""
    return np.clip(np.interp(raw, anchors_raw, anchors_out), 0.0, 100.0)


# =========================================================
# 5. VETOS v3.0.5 (com Fast-Approve)
# =========================================================
def apply_vetos_batch(
    df: pd.DataFrame,
    score_mapped: np.ndarray,
    lgbm_raw: np.ndarray,
    if_percentiles: np.ndarray,
    se_scores: np.ndarray,
    beh_scores: np.ndarray,
    beh_has_velocity: np.ndarray,
    beh_has_age_value: np.ndarray,
    cascade_action: pd.Series,
) -> tuple[np.ndarray, list[str | None]]:
    """
    Aplica vetos em batch v3.0.5.

    Hierarquia:
      0. Fast-Approve: LGBM < 0.25 + SE=0 + BEH=0 → skip IF-based vetos
      1. Cascade C1 → BLOQUEAR (NÃO afetado por FA)
      2. LGBM + IF convergência → BLOQUEAR
      3. IF extremo + score alto + LGBM >= 0.25 → BLOQUEAR
      4. SE CRITICO + BEH convergência → BLOQUEAR (NÃO afetado por FA)
      5. Cascade C3 → CONFIRMAR (já tem LGBM guard)
      6. LGBM solo >= veto_threshold → CONFIRMAR
      7. SE CRITICO → CONFIRMAR
      8. VETO_SE_BEH_VALOR_NOVO → CONFIRMAR (v3.0.4)
      9. BEH velocity + age_value >= 40 → CONFIRMAR
    """
    n = len(df)
    scores = score_mapped.copy()
    veto_descs: list[str | None] = [None] * n

    # Pré-calcular arrays para veto #8
    vl_pix_arr = df["vl_pix"].fillna(0).values
    rel_mes_arr = df["qt_tempo_relacionamento_mes"].fillna(999).values

    for i in range(n):
        # ── 0. FAST-APPROVE OVERRIDE (v3.0.5) ──
        fast_approve_active = (
            lgbm_raw[i] < FA_LGBM_MAX
            and se_scores[i] == 0
            and beh_scores[i] == 0
        )

        # ── 1. Cascade C1 → BLOQUEAR ──
        if cascade_action.iloc[i] == "BLOQUEAR":
            scores[i] = max(scores[i], THRESHOLD_BLOQUEAR)
            veto_descs[i] = "VETO BLOQUEAR: Cascade BURST_GTE3"
            continue

        # ── Sinais base ──
        lgbm_veto = lgbm_raw[i] >= VETO_THRESHOLD
        if_is_high = if_percentiles[i] >= VETO_THRESHOLD
        # v3.0.5: IF veto desabilitado se Fast-Approve ativo
        if_veto_eligible = (
            if_is_high
            and lgbm_raw[i] >= LGBM_MIN_FOR_IF_VETO
            and not fast_approve_active
        )

        # ── 2. LGBM + IF convergência → BLOQUEAR ──
        if lgbm_veto and if_veto_eligible:
            scores[i] = max(scores[i], THRESHOLD_BLOQUEAR)
            veto_descs[i] = (
                f"VETO BLOQUEAR: LGBM={lgbm_raw[i]*100:.1f}% + "
                f"IF={if_percentiles[i]*100:.1f}%"
            )
            continue

        # ── 3. IF extremo + score alto → BLOQUEAR ──
        # v3.0.5: LGBM gate reforçado (0.05 → 0.25) + Fast-Approve
        if (
            if_percentiles[i] >= 0.995
            and score_mapped[i] >= 50
            and lgbm_raw[i] >= LGBM_MIN_FOR_IF_EXTREME
            and not fast_approve_active
        ):
            scores[i] = max(scores[i], THRESHOLD_BLOQUEAR)
            veto_descs[i] = (
                f"VETO BLOQUEAR: IF extremo={if_percentiles[i]*100:.1f}%"
            )
            continue

        # ── 4. SE CRITICO + BEH convergência → BLOQUEAR ──
        if se_scores[i] >= 60 and beh_scores[i] >= 25:
            scores[i] = max(scores[i], THRESHOLD_BLOQUEAR)
            veto_descs[i] = (
                f"VETO BLOQUEAR: SE={se_scores[i]:.0f} + "
                f"BEH={beh_scores[i]:.0f}"
            )
            continue

        # ── 5. Cascade C3 → CONFIRMAR ──
        if cascade_action.iloc[i] == "CONFIRMAR":
            scores[i] = max(scores[i], THRESHOLD_CONFIRMAR)
            veto_descs[i] = "VETO CONFIRMAR: Cascade IF999_BURST"
            continue

        # ── 6. LGBM solo >= veto_threshold → CONFIRMAR ──
        if lgbm_veto:
            scores[i] = max(scores[i], THRESHOLD_CONFIRMAR)
            veto_descs[i] = (
                f"VETO CONFIRMAR: LGBM={lgbm_raw[i]*100:.1f}%"
            )
            continue

        # ── 7. SE CRITICO → CONFIRMAR ──
        if se_scores[i] >= 60:
            scores[i] = max(scores[i], THRESHOLD_CONFIRMAR)
            veto_descs[i] = (
                f"VETO CONFIRMAR: SE CRITICO ({se_scores[i]:.0f})"
            )
            continue

        # ── 8. VETO_SE_BEH_VALOR_NOVO (v3.0.4) → CONFIRMAR ──
        if (
            se_scores[i] >= 40
            and beh_scores[i] >= 15
            and vl_pix_arr[i] >= 15000
            and rel_mes_arr[i] <= 12
        ):
            scores[i] = max(scores[i], THRESHOLD_CONFIRMAR)
            veto_descs[i] = (
                f"VETO CONFIRMAR: SE({se_scores[i]:.0f}) + "
                f"BEH({beh_scores[i]:.0f}) + "
                f"valor(R${vl_pix_arr[i]:,.0f}) + "
                f"conta_nova({rel_mes_arr[i]:.0f}m)"
            )
            continue

        # ── 9. BEH velocity + age_value >= 40 → CONFIRMAR ──
        if beh_scores[i] >= 40 and (
            beh_has_velocity[i] and beh_has_age_value[i]
        ):
            scores[i] = max(scores[i], THRESHOLD_CONFIRMAR)
            veto_descs[i] = f"VETO CONFIRMAR: BEH={beh_scores[i]:.0f}"
            continue

    return scores, veto_descs


# =========================================================
# 6. SE v3.4 + BEH v3.1 (usando módulos reais)
# =========================================================
def run_se_batch(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[list[str]], list[list[dict]]]:
    """Roda SE v3.4 em cada row."""
    from core.social_engineering import SocialEngineeringDetector

    detector = SocialEngineeringDetector()
    n = len(df)
    scores = np.zeros(n)
    all_pattern_names: list[list[str]] = []
    all_raw_patterns: list[list[dict]] = []

    for idx, (_, row) in enumerate(df.iterrows()):
        features = row.to_dict()
        features = {
            k: (None if pd.isna(v) else v) for k, v in features.items()
        }
        result = detector.detect_from_pipeline(features)
        scores[idx] = result.se_score
        all_pattern_names.append(
            [p.pattern_name for p in result.patterns]
        )
        all_raw_patterns.append([
            {
                "pattern_name": p.pattern_name,
                "severity": p.severity,
                "score": p.score,
                "description": p.description,
                "matched_indicators": p.matched_indicators,
            }
            for p in result.patterns
        ])

        if (idx + 1) % 10000 == 0:
            logger.info(f"  SE: {idx + 1}/{n} processadas")

    return scores, all_pattern_names, all_raw_patterns


def run_beh_batch(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[list[dict]], np.ndarray, np.ndarray]:
    """Roda BEH v3.1 em cada row."""
    from core.behavioral_analytics import BehavioralAnalytics

    engine = BehavioralAnalytics()
    n = len(df)
    scores = np.zeros(n)
    has_velocity = np.zeros(n, dtype=bool)
    has_age_value = np.zeros(n, dtype=bool)
    all_factors: list[list[dict]] = []

    for idx, (_, row) in enumerate(df.iterrows()):
        features = row.to_dict()
        features = {
            k: (None if pd.isna(v) else v) for k, v in features.items()
        }
        result = engine.analyze(features)
        scores[idx] = result.behavioral_score
        factors = [
            {
                "codigo": rf.codigo,
                "source": rf.source,
                "score_add": rf.score_add,
                "precision": rf.precision,
            }
            for rf in result.risk_factors
        ]
        all_factors.append(factors)
        has_velocity[idx] = any(
            rf.source == "velocity" for rf in result.risk_factors
        )
        has_age_value[idx] = any(
            rf.source == "age_value" for rf in result.risk_factors
        )

        if (idx + 1) % 10000 == 0:
            logger.info(f"  BEH: {idx + 1}/{n} processadas")

    return scores, all_factors, has_velocity, has_age_value


# =========================================================
# 7. CLASSIFICAÇÃO FINAL
# =========================================================
def classify_batch(scores: np.ndarray) -> np.ndarray:
    """Classifica decisão baseado no score final."""
    decisions = np.full(len(scores), "APROVAR", dtype=object)
    decisions[scores >= THRESHOLD_CONFIRMAR] = "CONFIRMAR"
    decisions[scores >= THRESHOLD_BLOQUEAR] = "BLOQUEAR"
    return decisions


# =========================================================
# 8. ANÁLISES
# =========================================================
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> dict[str, Any]:
    """Calcula TP, FP, FN, TN, Precision, Recall, F1."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    fpr = fp / max(fp + tn, 1)

    return {
        "label": label,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "FPR": round(fpr, 6),
    }


def analise_a_metricas_globais(df: pd.DataFrame) -> dict[str, Any]:
    """A. Métricas globais por decisão."""
    y = df["is_fraud"].values
    results = {}

    flagged = df["decisao"].isin(["CONFIRMAR", "BLOQUEAR"]).astype(int).values
    results["pipeline_total"] = compute_metrics(y, flagged, "Pipeline (C+B)")

    blocked = (df["decisao"] == "BLOQUEAR").astype(int).values
    results["pipeline_bloquear"] = compute_metrics(y, blocked, "Pipeline (B)")

    eff_th = df["lgbm_effective_threshold"].iloc[0]
    lgbm_flag = (df["lgbm_raw"] >= eff_th).astype(int).values
    results["lgbm_solo"] = compute_metrics(y, lgbm_flag, "LGBM solo")

    cascade_flag = df["cascade_action"].isin(["BLOQUEAR", "CONFIRMAR"]).astype(int).values
    results["cascade_solo"] = compute_metrics(y, cascade_flag, "Cascade solo")

    se_flag = (df["se_score"] > 0).astype(int).values
    results["se_solo"] = compute_metrics(y, se_flag, "SE solo (score>0)")

    beh_flag = (df["beh_score"] > 0).astype(int).values
    results["beh_solo"] = compute_metrics(y, beh_flag, "BEH solo (score>0)")

    return results


def analise_b_contribuicao_marginal(df: pd.DataFrame) -> dict[str, Any]:
    """B. Contribuição marginal."""
    y = df["is_fraud"].values
    fraud_idx = set(df.index[y == 1])
    eff_th = df["lgbm_effective_threshold"].iloc[0]

    lgbm_det = set(df.index[(y == 1) & (df["lgbm_raw"] >= eff_th)])
    cascade_det = set(df.index[
        (y == 1) & df["cascade_action"].isin(["BLOQUEAR", "CONFIRMAR"])
    ])
    se_det = set(df.index[(y == 1) & (df["se_score"] > 0)])
    beh_det = set(df.index[(y == 1) & (df["beh_score"] > 0)])

    pipeline_det = set(df.index[
        (y == 1) & df["decisao"].isin(["CONFIRMAR", "BLOQUEAR"])
    ])

    all_det = lgbm_det | cascade_det | se_det | beh_det

    return {
        "total_fraudes": len(fraud_idx),
        "pipeline_detected": len(pipeline_det),
        "pipeline_missed": len(fraud_idx - pipeline_det),
        "components": {
            "LGBM": {
                "detected": len(lgbm_det),
                "exclusive": len(lgbm_det - cascade_det - se_det - beh_det),
                "incremental_over_lgbm": 0,
            },
            "Cascade": {
                "detected": len(cascade_det),
                "exclusive": len(cascade_det - lgbm_det - se_det - beh_det),
                "incremental_over_lgbm": len(cascade_det - lgbm_det),
            },
            "SE": {
                "detected": len(se_det),
                "exclusive": len(se_det - lgbm_det - cascade_det - beh_det),
                "incremental_over_lgbm": len(se_det - lgbm_det),
            },
            "BEH": {
                "detected": len(beh_det),
                "exclusive": len(beh_det - lgbm_det - cascade_det - se_det),
                "incremental_over_lgbm": len(beh_det - lgbm_det),
            },
        },
        "any_component": len(all_det),
        "invisible_to_all": len(fraud_idx - all_det),
    }


def analise_c_overlap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """C. Overlap matrix."""
    y = df["is_fraud"].values
    eff_th = df["lgbm_effective_threshold"].iloc[0]

    detections = {
        "LGBM": set(df.index[(y == 1) & (df["lgbm_raw"] >= eff_th)]),
        "Cascade": set(df.index[
            (y == 1) & df["cascade_action"].isin(["BLOQUEAR", "CONFIRMAR"])
        ]),
        "SE": set(df.index[(y == 1) & (df["se_score"] > 0)]),
        "BEH": set(df.index[(y == 1) & (df["beh_score"] > 0)]),
    }

    components = list(detections.keys())
    matrix = pd.DataFrame(0, index=components, columns=components)
    for ci in components:
        for cj in components:
            matrix.loc[ci, cj] = len(detections[ci] & detections[cj])

    return matrix


def analise_d_fn_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """D. FN analysis."""
    y = df["is_fraud"].values
    missed = df[(y == 1) & (df["decisao"] == "APROVAR")].copy()

    if len(missed) == 0:
        logger.info("🎉 Zero FN — pipeline pegou TODAS as fraudes!")
        return pd.DataFrame()

    cols = [
        "vl_pix", "nr_idade", "qt_tempo_relacionamento_mes",
        "qt_total_pix_trimestre", "tx_count_prev_30m", "burst_30m_flag",
        "first_receiver_flag", "pix_key_random_flag",
        "lgbm_raw", "if_percentile", "se_score", "beh_score",
        "score_final", "decisao",
    ]
    available = [c for c in cols if c in missed.columns]
    return missed[available].sort_values("lgbm_raw", ascending=False)


def analise_e_fp_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """E. FP analysis."""
    y = df["is_fraud"].values
    flagged_normal = df[
        (y == 0) & df["decisao"].isin(["CONFIRMAR", "BLOQUEAR"])
    ]
    eff_th = df["lgbm_effective_threshold"].iloc[0]

    fp_sources = {
        "LGBM (>=eff_th)": int(((y == 0) & (df["lgbm_raw"] >= eff_th)).sum()),
        "Cascade C1": int(((y == 0) & df["cascade_c1"]).sum()),
        "Cascade C3": int(((y == 0) & df["cascade_c3"]).sum()),
        "SE (score>0)": int(((y == 0) & (df["se_score"] > 0)).sum()),
        "BEH (score>0)": int(((y == 0) & (df["beh_score"] > 0)).sum()),
        "Veto BLOQUEAR": int(
            ((y == 0) & df["veto_desc"].notna()
             & df["veto_desc"].str.contains("BLOQUEAR", na=False)).sum()
        ),
        "Veto CONFIRMAR": int(
            ((y == 0) & df["veto_desc"].notna()
             & df["veto_desc"].str.contains("CONFIRMAR", na=False)).sum()
        ),
    }

    return {
        "total_fp_pipeline": len(flagged_normal),
        "fp_by_source": fp_sources,
        "fp_decisao": {
            "CONFIRMAR": int(((y == 0) & (df["decisao"] == "CONFIRMAR")).sum()),
            "BLOQUEAR": int(((y == 0) & (df["decisao"] == "BLOQUEAR")).sum()),
        },
    }


def analise_f_threshold_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """F. Threshold sweep."""
    y = df["is_fraud"].values
    score = df["score_final"].values

    rows = []
    for th_conf in range(40, 81, 5):
        for th_block in range(max(th_conf + 10, 70), 96, 5):
            flagged = (score >= th_conf).astype(int)
            blocked = (score >= th_block).astype(int)
            m_flag = compute_metrics(y, flagged, f"C>={th_conf}")
            m_block = compute_metrics(y, blocked, f"B>={th_block}")
            rows.append({
                "th_confirmar": th_conf,
                "th_bloquear": th_block,
                "flagged_prec": m_flag["Precision"],
                "flagged_rec": m_flag["Recall"],
                "flagged_f1": m_flag["F1"],
                "flagged_fp": m_flag["FP"],
                "blocked_prec": m_block["Precision"],
                "blocked_rec": m_block["Recall"],
                "blocked_f1": m_block["F1"],
                "blocked_fp": m_block["FP"],
            })

    return pd.DataFrame(rows).sort_values("flagged_f1", ascending=False)


# =========================================================
# 9. PRINT FORMATADO
# =========================================================
def print_section(title: str) -> None:
    """Imprime separador de seção."""
    width = 70
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def print_metrics_table(metrics: dict[str, dict]) -> None:
    """Imprime tabela de métricas."""
    header = (
        f"{'Componente':<25} {'TP':>5} {'FP':>6} {'FN':>5} "
        f"{'Prec':>7} {'Rec':>7} {'F1':>7} {'FPR':>8}"
    )
    print(header)
    print("-" * len(header))
    for m in metrics.values():
        print(
            f"{m['label']:<25} {m['TP']:>5} {m['FP']:>6} {m['FN']:>5} "
            f"{m['Precision']:>7.1%} {m['Recall']:>7.1%} {m['F1']:>7.3f} "
            f"{m['FPR']:>8.4%}"
        )


# =========================================================
# 10. MAIN
# =========================================================
def main() -> None:
    """Executa simulação end-to-end v3.0.5."""
    t_start = time.perf_counter()

    print_section(f"SIMULAÇÃO END-TO-END — Pipeline Antifraude v{ENGINE_VERSION}")
    print(f"  Dataset: {DATASET_PATH.name}")
    print(f"  Output:  {OUTPUT_DIR}")
    print(f"  Timestamp: {TIMESTAMP}")
    print(f"  Thresholds: CONFIRMAR={THRESHOLD_CONFIRMAR} BLOQUEAR={THRESHOLD_BLOQUEAR}")
    print(f"  Fast-Approve: LGBM < {FA_LGBM_MAX} + SE=0 + BEH=0")
    print(f"  Cascade C3 LGBM guard: >= {CASCADE_C3_LGBM_MIN}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Carregar artefatos ─────────────────────────────────
    print_section("1. CARREGAMENTO DE ARTEFATOS")
    loader = ArtefatosLoader()
    loader.load_all()

    # ─── Carregar dataset ───────────────────────────────────
    print_section("2. CARREGAMENTO DO DATASET")
    df = pd.read_csv(DATASET_PATH)
    n_total = len(df)
    n_fraud = int(df["is_fraud"].sum())
    n_normal = n_total - n_fraud
    logger.info(
        f"Dataset: {n_total:,} tx | {n_fraud} fraudes ({n_fraud/n_total:.2%}) | "
        f"{n_normal:,} normais"
    )

    missing_feats = [f for f in loader.lgbm_features if f not in df.columns]
    if missing_feats:
        logger.warning(
            f"⚠️ {len(missing_feats)} features LGBM faltando: "
            f"{missing_feats[:5]}... — preenchidas com 0"
        )
        for f in missing_feats:
            df[f] = 0

    # ─── LGBM Scoring ──────────────────────────────────────
    print_section("3. LGBM SCORING")
    t0 = time.perf_counter()
    df["lgbm_raw"] = score_lgbm_batch(df, loader)
    df["lgbm_effective_threshold"] = loader.lgbm_effective_threshold
    lgbm_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"LGBM: {lgbm_ms:.0f}ms | threshold={loader.lgbm_threshold:.4f} | "
        f"effective={loader.lgbm_effective_threshold:.4f}"
    )
    logger.info(
        f"  Fraudes >= eff_th: "
        f"{((df['is_fraud']==1) & (df['lgbm_raw'] >= loader.lgbm_effective_threshold)).sum()}"
        f"/{n_fraud}"
    )

    # ─── IF Scoring ────────────────────────────────────────
    print_section("4. ISOLATION FOREST SCORING")
    t0 = time.perf_counter()
    df["if_percentile"], df["if_raw"] = score_if_batch(df, loader)
    if_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"IF: {if_ms:.0f}ms")
    logger.info(
        f"  IF >= 99.5%: {int((df['if_percentile'] >= 0.995).sum()):,} tx "
        f"({((df['is_fraud']==1) & (df['if_percentile'] >= 0.995)).sum()} fraudes)"
    )

    # ─── Cascade Rules v3 ─────────────────────────────────
    print_section("5. CASCADE RULES v3")
    cascade_df = apply_cascade_batch(
        df, df["if_percentile"].values, df["lgbm_raw"].values,
    )
    df["cascade_c1"] = cascade_df["cascade_c1"]
    df["cascade_c3"] = cascade_df["cascade_c3"]
    df["cascade_action"] = cascade_df["cascade_action"]

    c1_count = int(df["cascade_c1"].sum())
    c3_count = int(df["cascade_c3"].sum())
    c1_tp = int(((df["is_fraud"] == 1) & df["cascade_c1"]).sum())
    c3_tp = int(((df["is_fraud"] == 1) & df["cascade_c3"]).sum())
    c3_fp = c3_count - c3_tp
    logger.info(
        f"C1 (burst>=3): {c1_count} ativações ({c1_tp} TP) | "
        f"C3 (IF+burst+LGBM): {c3_count} ativações ({c3_tp} TP, {c3_fp} FP)"
    )

    # ─── SE v3.4 ───────────────────────────────────────────
    print_section("6. SOCIAL ENGINEERING v3.4")
    t0 = time.perf_counter()
    df["se_score"], se_patterns, se_raw_patterns = run_se_batch(df)
    se_ms = (time.perf_counter() - t0) * 1000
    se_detected = int((df["se_score"] > 0).sum())
    se_tp = int(((df["is_fraud"] == 1) & (df["se_score"] > 0)).sum())
    logger.info(
        f"SE: {se_ms:.0f}ms | {se_detected} ativações "
        f"({se_tp} TP / {se_detected - se_tp} FP)"
    )

    # ─── BEH v3.1 ─────────────────────────────────────────
    print_section("7. BEHAVIORAL ANALYTICS v3.1")
    t0 = time.perf_counter()
    (
        df["beh_score"],
        beh_factors,
        df["beh_has_velocity"],
        df["beh_has_age_value"],
    ) = run_beh_batch(df)
    beh_ms = (time.perf_counter() - t0) * 1000
    beh_detected = int((df["beh_score"] > 0).sum())
    beh_tp = int(((df["is_fraud"] == 1) & (df["beh_score"] > 0)).sum())
    logger.info(
        f"BEH: {beh_ms:.0f}ms | {beh_detected} ativações "
        f"({beh_tp} TP / {beh_detected - beh_tp} FP)"
    )

    # ─── Ensemble ──────────────────────────────────────────
    print_section("8. ENSEMBLE + MAPEAMENTO")
    df["ensemble_raw"] = calculate_ensemble_batch(
        df["lgbm_raw"].values,
        df["cascade_action"],
        effective_threshold=loader.lgbm_effective_threshold,
    )
    df["score_mapped"] = map_to_score_batch(
        df["ensemble_raw"].values, loader.anchors_raw, loader.anchors_out,
    )

    # ─── Vetos v3.0.5 ─────────────────────────────────────
    print_section("9. VETOS v3.0.5 (Fast-Approve + SE/BEH)")
    df["score_final"], veto_descs = apply_vetos_batch(
        df,
        df["score_mapped"].values,
        df["lgbm_raw"].values,
        df["if_percentile"].values,
        df["se_score"].values,
        df["beh_score"].values,
        df["beh_has_velocity"].values,
        df["beh_has_age_value"].values,
        df["cascade_action"],
    )
    df["veto_desc"] = veto_descs
    n_vetos = int(df["veto_desc"].notna().sum())
    logger.info(f"Vetos aplicados: {n_vetos}")

    # Fast-Approve stats
    fa_mask = (
        (df["lgbm_raw"] < FA_LGBM_MAX)
        & (df["se_score"] == 0)
        & (df["beh_score"] == 0)
    )
    fa_count = int(fa_mask.sum())
    fa_fraud = int((fa_mask & (df["is_fraud"] == 1)).sum())
    logger.info(
        f"Fast-Approve ativo: {fa_count:,} tx ({fa_fraud} fraudes — "
        f"{'⚠️ RISCO' if fa_fraud > 0 else '✅ seguro'})"
    )

    # ─── Decisão ───────────────────────────────────────────
    df["decisao"] = classify_batch(df["score_final"].values)

    # ─── ANÁLISES ──────────────────────────────────────────
    print_section("A. MÉTRICAS GLOBAIS")
    metrics_a = analise_a_metricas_globais(df)
    print_metrics_table(metrics_a)

    print_section("B. CONTRIBUIÇÃO MARGINAL")
    contrib = analise_b_contribuicao_marginal(df)
    print(f"\n  Total fraudes: {contrib['total_fraudes']}")
    print(f"  Pipeline detectou: {contrib['pipeline_detected']}")
    print(f"  Pipeline perdeu:   {contrib['pipeline_missed']}")
    print(f"  Invisíveis (nenhum componente): {contrib['invisible_to_all']}")
    print()
    header = f"  {'Componente':<12} {'Detectou':>10} {'Exclusivo':>10} {'Incr/LGBM':>10}"
    print(header)
    print(f"  {'-'*42}")
    for name, data in contrib["components"].items():
        print(
            f"  {name:<12} {data['detected']:>10} "
            f"{data['exclusive']:>10} {data['incremental_over_lgbm']:>10}"
        )

    print_section("C. OVERLAP MATRIX (TPs)")
    overlap = analise_c_overlap_matrix(df)
    print(overlap.to_string())

    print_section("D. FN ANALYSIS (Fraudes não detectadas)")
    fn_df = analise_d_fn_analysis(df)
    if len(fn_df) > 0:
        print(f"\n  {len(fn_df)} fraudes não detectadas:")
        print(fn_df.head(20).to_string())
    else:
        print("\n  🎉 ZERO FN — todas as fraudes foram detectadas!")

    print_section("E. FP ANALYSIS")
    fp_analysis = analise_e_fp_analysis(df)
    print(f"\n  Total FP do pipeline: {fp_analysis['total_fp_pipeline']}")
    print(f"\n  FP por fonte:")
    for source, count in fp_analysis["fp_by_source"].items():
        print(f"    {source:<25} {count:>6}")
    print(f"\n  FP por decisão:")
    for dec, count in fp_analysis["fp_decisao"].items():
        print(f"    {dec:<25} {count:>6}")

    print_section("F. THRESHOLD SWEEP")
    sweep = analise_f_threshold_sweep(df)
    print("\n  Top 10 configurações por F1 (flagged):")
    print(
        sweep.head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # ─── SALVAR RESULTADOS ─────────────────────────────────
    print_section("SALVANDO RESULTADOS")

    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        raise TypeError(f"Tipo não serializável: {type(obj)}")

    try:
        with open(OUTPUT_DIR / "metricas_globais.json", "w", encoding="utf-8") as f:
            json.dump(metrics_a, f, indent=2, ensure_ascii=False, default=_json_safe)
        logger.info("  ✅ metricas_globais.json")

        with open(OUTPUT_DIR / "contribuicao_marginal.json", "w", encoding="utf-8") as f:
            json.dump(contrib, f, indent=2, ensure_ascii=False, default=_json_safe)
        logger.info("  ✅ contribuicao_marginal.json")

        overlap.to_csv(OUTPUT_DIR / "overlap_matrix.csv")
        logger.info("  ✅ overlap_matrix.csv")

        if len(fn_df) > 0:
            fn_df.to_csv(OUTPUT_DIR / "fn_analysis.csv", index=False)
            logger.info(f"  ✅ fn_analysis.csv ({len(fn_df)} rows)")

        with open(OUTPUT_DIR / "fp_analysis.json", "w", encoding="utf-8") as f:
            json.dump(fp_analysis, f, indent=2, ensure_ascii=False, default=_json_safe)
        logger.info("  ✅ fp_analysis.json")

        sweep.to_csv(OUTPUT_DIR / "threshold_sweep.csv", index=False)
        logger.info(f"  ✅ threshold_sweep.csv ({len(sweep)} rows)")

    except Exception as e:
        logger.error(f"  ❌ Erro ao salvar: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # Decisões completas
    export_cols = [
        "is_fraud", "vl_pix", "nr_idade", "qt_tempo_relacionamento_mes",
        "tx_count_prev_30m", "burst_30m_flag", "first_receiver_flag",
        "lgbm_raw", "if_percentile", "if_raw",
        "cascade_c1", "cascade_c3", "cascade_action",
        "se_score", "beh_score",
        "beh_has_velocity", "beh_has_age_value",
        "ensemble_raw", "score_mapped", "score_final",
        "veto_desc", "decisao",
    ]
    available_cols = [c for c in export_cols if c in df.columns]
    df[available_cols].to_csv(OUTPUT_DIR / "decisoes_completas.csv", index=False)

    # Config
    config_info = {
        "engine_version": ENGINE_VERSION,
        "se_version": SE_VERSION,
        "beh_version": BEH_VERSION,
        "lgbm_threshold": loader.lgbm_threshold,
        "lgbm_effective_threshold": loader.lgbm_effective_threshold,
        "if_cascade_threshold": CASCADE_IF_THRESHOLD,
        "cascade_c3_lgbm_min": CASCADE_C3_LGBM_MIN,
        "fast_approve_lgbm_max": FA_LGBM_MAX,
        "lgbm_min_for_if_extreme": LGBM_MIN_FOR_IF_EXTREME,
        "threshold_confirmar": THRESHOLD_CONFIRMAR,
        "threshold_bloquear": THRESHOLD_BLOQUEAR,
        "veto_threshold": VETO_THRESHOLD,
        "dataset": str(DATASET_PATH),
        "n_total": n_total,
        "n_fraud": n_fraud,
        "timestamp": TIMESTAMP,
    }
    with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)

    # ─── SUMMARY ───────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_start
    pm = metrics_a["pipeline_total"]

    print_section("RESUMO FINAL")
    print(f"""
  Dataset:     {n_total:,} tx ({n_fraud} fraudes, {n_fraud/n_total:.2%})
  Pipeline:    Engine v{ENGINE_VERSION} + SE v{SE_VERSION} + BEH v{BEH_VERSION}
  Thresholds:  CONFIRMAR={THRESHOLD_CONFIRMAR} | BLOQUEAR={THRESHOLD_BLOQUEAR}
  Fast-Approve: LGBM < {FA_LGBM_MAX} + SE=0 + BEH=0
  Cascade C3:  LGBM guard >= {CASCADE_C3_LGBM_MIN}

  ┌─────────────────────────────────────────────────┐
  │  RESULTADO DO PIPELINE COMPLETO                  │
  │                                                   │
  │  TP = {pm['TP']:<6}  FP = {pm['FP']:<6}               │
  │  FN = {pm['FN']:<6}  TN = {pm['TN']:<6}               │
  │                                                   │
  │  Precision = {pm['Precision']:.1%}                           │
  │  Recall    = {pm['Recall']:.1%}                           │
  │  F1        = {pm['F1']:.3f}                            │
  │  FPR       = {pm['FPR']:.4%}                          │
  └─────────────────────────────────────────────────┘

  Fraudes invisíveis: {contrib['invisible_to_all']}
  Tempo total: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)
  Output: {OUTPUT_DIR}
    """)

    logger.info("✅ Simulação concluída com sucesso!")


if __name__ == "__main__":
    main()
