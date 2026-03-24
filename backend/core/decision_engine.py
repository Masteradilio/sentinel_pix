"""
core/decision_engine.py v2.1 — Motor de Decisão para Transações PIX

Mudanças v2.0 → v2.1:
  1. IF v4 COMPLEMENTAR: IF ativa para QUALQUER tx quando LGBM < threshold
     (antes: só primeiras tx do trimestre)
  2. CASCADE RULES: 6 regras integradas para capturar bursts que LGBM perde
  3. FEATURES DE INTERAÇÃO: cria valor_x_burst, etc. inline para o IF
  4. ENSEMBLE BOOST CONDICIONAL: IF adiciona boost ao score quando LGBM incerto
     (antes: weighted average com zona cinzenta)
  5. IF CONFIG v4: carrega ensemble_params do isolation_forest_config.json
  6. THRESHOLDS CONFIG: carrega thresholds_config.json com threshold_f1 do LGBM
  7. NOVO AGRAVANTE: CASCADE_RULE quando cascade detecta padrão
  8. LGBM THRESHOLD: usa threshold_f1 (0.08) em vez de fixo 0.5

Responsabilidades deste módulo (e SOMENTE estas):
  1. Carregar modelos (LGBM + IF) e artefatos de scoring
  2. Calcular scores dos modelos
  3. Aplicar cascade rules nos FN do LGBM
  4. Calcular ensemble com boost condicional do IF
  5. Mapeamento 0-100
  6. Calcular 24 agravantes (7 fases)
  7. Aplicar regras de veto
  8. Integrar scores de SE e Behavioral (recebidos prontos)
  9. Retornar DecisionResult completo

Integração:
  O orquestrador chama:
    engine = PixDecisionEngine(config)
    result = engine.decide(features_dict, se_result, behavioral_result)
"""

from __future__ import annotations

import json
import logging
import time
import numpy as np
import joblib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =========================================================
# CONFIGURATION DEFAULTS
# =========================================================
@dataclass
class EngineConfig:
    """Configuração do motor de decisão. Pode ser overridden pelo orquestrador."""

    # --- Paths dos artefatos ---
    artefatos_dir: str = "backend/artefatos"

    # --- Thresholds de decisão (score 0-100) ---
    threshold_confirmar: float = 60.0
    threshold_bloquear: float = 85.0

    # --- Thresholds de veto (score raw 0-1) ---
    veto_threshold: float = 0.85

    # --- LGBM threshold (melhor F1 do treino v4.1) ---
    lgbm_threshold: float = 0.08

    # --- Faixas de agravante para scores de modelo ---
    faixa_1_max: int = 50       # 0-50%: peso 0
    faixa_2_max: int = 69       # 51-69%: peso 1
    faixa_3_max: int = 85       # 70-85%: peso 2, 86%+: peso 3

    # --- Pesos dos agravantes ---
    peso_chave_aleatoria: int = 2
    peso_intervalo_30min_1tx: int = 1
    peso_intervalo_30min_2tx: int = 2

    # --- Ensemble IF v4 (boost condicional) ---
    if_high_threshold: float = 0.99
    if_very_high_threshold: float = 0.9994
    if_boost_high: float = 0.05
    if_boost_very_high: float = 0.08

    # --- Cascade rules ---
    cascade_enabled: bool = True

    # --- Peso máximo teórico dos agravantes ---
    peso_maximo: int = 70

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineConfig":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


# =========================================================
# DATA CLASSES
# =========================================================
@dataclass
class AgravanteFator:
    """Representa um fator agravante aplicado à decisão."""
    codigo: str
    descricao: str
    peso: int
    valor_original: Any = None


@dataclass
class CascadeResult:
    """Resultado de uma regra cascade."""
    triggered: bool
    rule_id: str = ""
    rule_name: str = ""
    reason: str = ""


@dataclass
class DecisionResult:
    """Resultado completo da decisão para uma transação."""
    # Decisão
    decisao: str                    # "APROVAR", "CONFIRMAR", "BLOQUEAR"
    score_final: float              # 0-100 (mapeado)
    score_raw: float                # 0-1 (ensemble raw)

    # Componentes de scoring
    score_lgbm_raw: float           # 0-1
    score_lgbm_mapped: float        # 0-100
    score_if: float                 # 0-1 (percentile)
    score_if_raw: float             # Decision function raw
    if_active: bool                 # Se IF contribuiu
    if_boost_applied: float         # Boost adicionado pelo IF

    # Cascade
    cascade_triggered: bool = False
    cascade_rules: List[str] = field(default_factory=list)

    # Regras do pipeline
    rule_score_raw: float = 0.0
    rule_score_normalized: float = 0.0

    # Agravantes
    peso_total: int = 0
    peso_maximo: int = 70
    agravantes: List[AgravanteFator] = field(default_factory=list)

    # SE + Behavioral (recebidos do orquestrador)
    se_score: float = 0.0
    se_patterns: List[str] = field(default_factory=list)
    behavioral_score: float = 0.0
    behavioral_factors: List[Dict[str, Any]] = field(default_factory=list)

    # Vetos e atenuantes
    veto_aplicado: Optional[str] = None
    atenuantes: List[str] = field(default_factory=list)

    # Metadata
    latency_ms: float = 0.0
    transaction_id: Optional[str] = None
    customer_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializa para dict (usado pela API)."""
        return {
            "decisao": self.decisao,
            "score_final": round(self.score_final, 2),
            "score_raw": round(self.score_raw, 8),
            "componentes": {
                "lgbm_raw": round(self.score_lgbm_raw, 8),
                "lgbm_mapped": round(self.score_lgbm_mapped, 2),
                "if_score": round(self.score_if, 6),
                "if_raw": round(self.score_if_raw, 6),
                "if_active": self.if_active,
                "if_boost_applied": round(self.if_boost_applied, 4),
                "rule_score_raw": round(self.rule_score_raw, 2),
                "rule_score_normalized": round(self.rule_score_normalized, 4),
            },
            "cascade": {
                "triggered": self.cascade_triggered,
                "rules": self.cascade_rules,
            },
            "agravantes": [asdict(a) for a in self.agravantes if a.peso > 0],
            "peso_total": self.peso_total,
            "peso_maximo": self.peso_maximo,
            "social_engineering": {
                "se_score": round(self.se_score, 2),
                "patterns": self.se_patterns,
            },
            "behavioral": {
                "behavioral_score": round(self.behavioral_score, 2),
                "risk_factors": self.behavioral_factors,
            },
            "veto_aplicado": self.veto_aplicado,
            "atenuantes": self.atenuantes,
            "metadata": {
                "latency_ms": round(self.latency_ms, 2),
                "transaction_id": self.transaction_id,
                "customer_id": self.customer_id,
            },
            "faixas": {
                "aprovar": f"[0, {self.score_final:.0f})" if self.decisao == "APROVAR" else None,
                "confirmar": f"Score ≥ confirmar threshold" if self.decisao == "CONFIRMAR" else None,
                "bloquear": f"Score ≥ bloquear threshold" if self.decisao == "BLOQUEAR" else None,
            },
        }


# =========================================================
# CASCADE RULES (integradas do train_lgbm v4.1)
# =========================================================
class CascadeRuleEngine:
    """
    Regras cascade para capturar fraudes que o LGBM perde.

    6 regras baseadas na análise dos FN da v4.0/v4.1:
      C1: BURST_FIRST_RECEIVER — 3+ tx em 30min para recebedor novo
      C2: BURST_INTENSO — 5+ tx em 30min
      C3: CONTA_NOVA_ATIPICO — conta nova + recebedor novo + valor 3x mediana
      C4: CONTA_NOVA_ALTO_VALOR — conta < 3 meses + PIX > R$5000
      C5: ESVAZIAMENTO — burst + valor 5x mediana + > R$1000
      C6: LGBM_BORDERLINE_COMBINADO — LGBM > 0.01 + 3 sinais combinados
    """

    @staticmethod
    def evaluate(features: Dict[str, Any], lgbm_score: float) -> List[CascadeResult]:
        """Avalia todas as 6 regras cascade. Retorna lista de resultados."""
        results = []
        f = features

        tx_30m = _safe_int(f.get("tx_count_prev_30m"), 0)
        first_recv = _safe_int(f.get("first_receiver_flag"), 0)
        tempo_rel = _safe_float(f.get("qt_tempo_relacionamento_mes"), 999)
        ratio_med = _safe_float(f.get("ratio_valor_mediana"), 0)
        vl_pix = _safe_float(f.get("vl_pix"), 0)
        burst_flag = _safe_int(f.get("burst_30m_flag"), 0)
        idade = _safe_int(f.get("nr_idade"), 0)
        chave_random = _safe_int(f.get("pix_key_random_flag"), 0)

        # C1: Burst + primeiro recebedor
        if tx_30m >= 3 and first_recv == 1:
            results.append(CascadeResult(
                triggered=True, rule_id="C1",
                rule_name="BURST_FIRST_RECEIVER",
                reason=f"Burst de {tx_30m + 1} tx em 30min para recebedor novo",
            ))

        # C2: Burst intenso
        if tx_30m >= 5:
            results.append(CascadeResult(
                triggered=True, rule_id="C2",
                rule_name="BURST_INTENSO",
                reason=f"{tx_30m + 1} transações em 30 minutos (esvaziamento)",
            ))

        # C3 foi desativada (gerava muitos FPs sem capturar FNs adicionais)
        # if tempo_rel <= 6 and first_recv == 1 and ratio_med >= 3.0:
        #     results.append(CascadeResult(
        #         triggered=True, rule_id="C3",
        #         rule_name="CONTA_NOVA_ATIPICO",
        #         reason=f"Conta nova ({tempo_rel:.0f}m) + recebedor novo + valor {ratio_med:.1f}x mediana",
        #     ))

        # C4: Conta muito nova + alto valor
        if tempo_rel <= 3 and vl_pix >= 5000:
            results.append(CascadeResult(
                triggered=True, rule_id="C4",
                rule_name="CONTA_NOVA_ALTO_VALOR",
                reason=f"Conta < 3 meses + PIX R${vl_pix:,.0f}",
            ))

        # C5: Esvaziamento (burst + valor extremo)
        if burst_flag == 1 and ratio_med >= 5.0 and vl_pix >= 1000:
            results.append(CascadeResult(
                triggered=True, rule_id="C5",
                rule_name="ESVAZIAMENTO",
                reason=f"Burst + valor {ratio_med:.1f}x mediana (R${vl_pix:,.0f})",
            ))

        # C6: LGBM borderline + sinais combinados
        if lgbm_score >= 0.05:
            sinais = 0
            if first_recv == 1:
                sinais += 1
            if ratio_med >= 3.0:
                sinais += 1
            if vl_pix >= 2000:
                sinais += 1
            if idade >= 60:
                sinais += 1
            if chave_random == 1:
                sinais += 1
            if sinais >= 4:
                results.append(CascadeResult(
                    triggered=True, rule_id="C6",
                    rule_name="LGBM_BORDERLINE_COMBINADO",
                    reason=f"Score limítrofe ({lgbm_score:.3f}) com {sinais} sinais de risco",
                ))

        return results


# =========================================================
# MOTOR DE DECISÃO
# =========================================================
class PixDecisionEngine:
    """
    Motor de decisão para transações PIX v2.1.

    Recebe features processadas + resultados de SE/Behavioral
    → Retorna DecisionResult com score 0-100 e decisão.

    Fluxo interno v2.1:
      features_dict → LGBM Score
        ├── score >= lgbm_threshold → FRAUDE (LGBM detectou)
        ├── score < lgbm_threshold → Cascade Rules
        │   ├── Cascade triggered → FRAUDE (regras capturaram)
        │   └── Cascade clean → IF Score (boost condicional)
        │       ├── IF >= very_high → Boost +0.15
        │       ├── IF >= high → Boost +0.08
        │       └── IF < high → Sem boost
        └── Ensemble Raw → Mapeamento 0-100
            → Agravantes → Vetos → Decisão Final
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.art_dir = Path(self.config.artefatos_dir)

        # Modelos
        self.lgbm_model = None
        self.lgbm_features: List[str] = []
        self.if_model = None
        self.if_scaler = None
        self.if_features: List[str] = []
        self.if_medians: Dict[str, float] = {}
        self.if_ref_scores: Optional[np.ndarray] = None
        self.if_threshold: float = 0.95
        self.if_ensemble_params: Dict[str, Any] = {}

        # Cascade
        self.cascade_engine = CascadeRuleEngine()
        self.cascade_rules_config: List[Dict] = []

        # Scoring config (mapeamento híbrido)
        self.anchors_raw: np.ndarray = np.array([0.0, 1.0])
        self.anchors_out: np.ndarray = np.array([0.0, 100.0])
        self.scoring_config: Dict = {}

        # Status
        self.available = False
        self._load_time: Optional[float] = None

        # Carregar
        self._load_all()

    # ==========================================================
    # LOADING
    # ==========================================================
    def _load_all(self):
        """Carrega todos os artefatos do disco."""
        t0 = time.perf_counter()
        art = self.art_dir

        # 1. LightGBM
        lgbm_path = art / "model_lightgbm.joblib"
        if lgbm_path.exists():
            self.lgbm_model = joblib.load(lgbm_path)
            logger.info(f"LGBM carregado: {type(self.lgbm_model).__name__}")
        else:
            logger.warning(f"LGBM não encontrado em {lgbm_path}")

        # 2. LGBM Features
        features_path = art / "lgbm_features.json"
        if features_path.exists():
            with open(features_path, "r") as f:
                self.lgbm_features = json.load(f)
            logger.info(f"LGBM features: {len(self.lgbm_features)}")
        elif self.lgbm_model and hasattr(self.lgbm_model, "feature_name_"):
            self.lgbm_features = list(self.lgbm_model.feature_name_)

        # 3. Scoring Config
        scoring_path = art / "scoring_config.json"
        if scoring_path.exists():
            with open(scoring_path, "r", encoding="utf-8") as f:
                self.scoring_config = json.load(f)
            mapeamento = self.scoring_config.get("mapeamento", {})
            self.anchors_raw = np.array(
                mapeamento.get("anchors_raw", [0.0, 1.0]), dtype=np.float64
            )
            self.anchors_out = np.array(
                mapeamento.get("anchors_out", [0.0, 100.0]), dtype=np.float64
            )
            logger.info(f"Scoring config: {len(self.anchors_raw)} âncoras")

        # 4. Thresholds Config (v4.1)
        thresholds_path = art / "thresholds_config.json"
        if thresholds_path.exists():
            with open(thresholds_path, "r", encoding="utf-8") as f:
                th_config = json.load(f)
            lgbm_th = th_config.get("threshold_f1_best", self.config.lgbm_threshold)
            self.config.lgbm_threshold = float(lgbm_th)
            logger.info(f"LGBM threshold loaded: {self.config.lgbm_threshold:.4f}")

        # 5. Isolation Forest
        if_path = art / "model_isolation_forest.joblib"
        if if_path.exists():
            self.if_model = joblib.load(if_path)
            logger.info(f"IF carregado: {self.if_model.n_estimators} trees")

        # 6. IF Scaler
        scaler_path = art / "scaler_isolation_forest.joblib"
        if scaler_path.exists():
            self.if_scaler = joblib.load(scaler_path)

        # 7. IF Config (v4)
        config_path = art / "isolation_forest_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                if_config = json.load(f)
            self.if_features = if_config.get("features", [])
            self.if_medians = if_config.get("medians", {})
            self.if_threshold = if_config.get("best_threshold", 0.95)
            self.if_ensemble_params = if_config.get("ensemble_params", {})

            # Sobrescrever config com params do IF v4
            ep = self.if_ensemble_params
            if ep:
                self.config.if_high_threshold = float(ep.get("if_high_threshold", 0.70))
                self.config.if_very_high_threshold = float(ep.get("if_very_high_threshold", 0.85))
                self.config.if_boost_high = float(ep.get("boost_high", 0.08))
                self.config.if_boost_very_high = float(ep.get("boost_very_high", 0.15))

            logger.info(
                f"IF config v4: {len(self.if_features)} features | "
                f"boost: +{self.config.if_boost_high}/{self.config.if_boost_very_high}"
            )

        # 8. IF Reference Scores
        ref_path = art / "if_ref_raw_train.npy"
        if ref_path.exists():
            self.if_ref_scores = np.load(ref_path)
            logger.info(f"IF ref scores: {len(self.if_ref_scores)}")

        # 9. Cascade Rules Config
        cascade_path = art / "cascade_rules.json"
        if cascade_path.exists():
            with open(cascade_path, "r", encoding="utf-8") as f:
                cascade_config = json.load(f)
            self.cascade_rules_config = cascade_config.get("rules", [])
            logger.info(f"Cascade rules: {len(self.cascade_rules_config)}")

        self.available = self.lgbm_model is not None
        self._load_time = (time.perf_counter() - t0) * 1000

        logger.info(
            f"PixDecisionEngine v2.1 inicializado em {self._load_time:.1f}ms | "
            f"LGBM={'OK' if self.lgbm_model else 'MISSING'} | "
            f"IF={'OK' if self.if_model else 'OFF'} | "
            f"Cascade={'ON' if self.config.cascade_enabled else 'OFF'} | "
            f"Features: LGBM={len(self.lgbm_features)} IF={len(self.if_features)}"
        )

    # ==========================================================
    # SCORING — LGBM
    # ==========================================================
    def _score_lgbm(self, features: Dict[str, Any]) -> float:
        """Calcula score raw do LGBM (0-1)."""
        if self.lgbm_model is None:
            return 0.0

        import pandas as pd
        row = {}
        for feat in self.lgbm_features:
            val = features.get(feat, 0)
            try:
                row[feat] = float(val) if val is not None and str(val) != "nan" else 0.0
            except (ValueError, TypeError):
                row[feat] = 0.0

        X = pd.DataFrame([row])[self.lgbm_features]
        proba = self.lgbm_model.predict_proba(X)[:, 1]
        return float(np.clip(proba[0], 0.0, 1.0))

    # ==========================================================
    # SCORING — IF v4 (complementar, não restrito a 1ª tx)
    # ==========================================================
    def _create_if_interaction_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Cria features de interação necessárias pelo IF v4 inline."""
        tx_30m = _safe_float(features.get("tx_count_prev_30m"), 0)
        vl_pix = _safe_float(features.get("vl_pix"), 0)
        nr_idade = _safe_float(features.get("nr_idade"), 0)
        first_recv = _safe_float(features.get("first_receiver_flag"), 0)
        distinct_recv = _safe_float(features.get("distinct_receivers_so_far"), 1)
        mediana = _safe_float(features.get("vl_mediana_pix_trimestre"), 0)
        qt_total = max(_safe_float(features.get("qt_total_pix_trimestre"), 1), 1)

        total_esperado = mediana * qt_total

        return {
            "valor_x_burst": vl_pix * (tx_30m + 1),
            "idade_x_first_recv": nr_idade * first_recv,
            "valor_x_first_recv": vl_pix * first_recv,
            "burst_x_distinct_recv": tx_30m * distinct_recv,
            "valor_over_trimestre_avg": (vl_pix / total_esperado) if total_esperado > 0 else 0,
        }

    def _score_if(self, features: Dict[str, Any], lgbm_raw: float) -> Tuple[float, float, bool, float]:
        """
        Calcula score do Isolation Forest v4.

        O IF v4 ativa quando LGBM < lgbm_threshold (complementar).
        Retorna boost condicional baseado no score IF.

        Returns:
            (percentile_score, raw_score, is_active, boost_amount)
        """
        if self.if_model is None or not self.if_features:
            return 0.0, 0.0, False, 0.0

        # IF só ativa quando LGBM não confia (abaixo do threshold)
        if lgbm_raw >= self.config.lgbm_threshold:
            return 0.0, 0.0, False, 0.0

        import pandas as pd

        # Criar features de interação
        interaction_features = self._create_if_interaction_features(features)

        # Montar row com todas as features do IF
        row = {}
        for feat in self.if_features:
            # Primeiro tentar features diretas, depois interações
            if feat in interaction_features:
                row[feat] = interaction_features[feat]
            else:
                val = features.get(feat)
                try:
                    fval = float(val) if val is not None and str(val) != "nan" else None
                    if fval is None or np.isinf(fval):
                        row[feat] = self.if_medians.get(feat, 0)
                    else:
                        row[feat] = fval
                except (ValueError, TypeError):
                    row[feat] = self.if_medians.get(feat, 0)

        X = pd.DataFrame([row])[self.if_features]

        # Scaler
        if self.if_scaler is not None:
            X_scaled = self.if_scaler.transform(X)
        else:
            X_scaled = X.values

        # Score raw (decision_function)
        raw = float(self.if_model.decision_function(X_scaled)[0])

        # Percentile scoring (invertido: mais negativo = mais anômalo)
        if self.if_ref_scores is not None and len(self.if_ref_scores) > 0:
            inverted = -raw
            ref_inverted = -self.if_ref_scores
            percentile = float(np.mean(ref_inverted <= inverted))
        else:
            percentile = float(1.0 / (1.0 + np.exp(raw * 5)))

        percentile = float(np.clip(percentile, 0, 1))

        # Boost condicional baseado no score IF
        cfg = self.config
        if percentile >= cfg.if_very_high_threshold:
            boost = cfg.if_boost_very_high
        elif percentile >= cfg.if_high_threshold:
            boost = cfg.if_boost_high
        else:
            boost = 0.0

        return percentile, raw, True, boost

    # ==========================================================
    # ENSEMBLE v2.1 — Boost condicional
    # ==========================================================
    def _calculate_ensemble(
        self,
        lgbm_raw: float,
        if_score: float,
        if_active: bool,
        if_boost: float,
        cascade_triggered: bool,
    ) -> float:
        """
        Calcula ensemble raw (0-1) com boost condicional.

        Estratégia v2.1:
          - Se cascade triggered: força score mínimo = lgbm_threshold
          - Se IF ativo com boost: lgbm_raw + boost
          - Senão: lgbm_raw puro
        """
        score = lgbm_raw

        # Cascade: garante que a tx seja flaggada
        if cascade_triggered:
            score = max(score, self.config.lgbm_threshold)

        # IF boost: adiciona ao score quando LGBM incerto
        if if_active and if_boost > 0:
            score = score + if_boost

        return float(np.clip(score, 0.0, 1.0))

    def _map_to_score(self, raw: float) -> float:
        """Mapeamento não-linear: raw (0-1) → score (0-100)."""
        return float(np.clip(
            np.interp(raw, self.anchors_raw, self.anchors_out),
            0.0, 100.0,
        ))

    # ==========================================================
    # AGRAVANTES (7 fases)
    # ==========================================================
    def _score_to_peso(self, score: float) -> int:
        """Converte score 0-1 em peso de agravante."""
        pct = score * 100
        if pct <= self.config.faixa_1_max:
            return 0
        if pct <= self.config.faixa_2_max:
            return 1
        if pct <= self.config.faixa_3_max:
            return 2
        return 3

    def _calcular_agravantes(
        self,
        features: Dict[str, Any],
        lgbm_raw: float,
        if_score: float,
        if_active: bool,
        cascade_results: List[CascadeResult],
        se_score: float,
        se_patterns: List[str],
        behavioral_score: float,
    ) -> List[AgravanteFator]:
        """Calcula todos os agravantes (Fases 1-7 + SE + Behavioral)."""
        agravantes = []
        f = features

        # ─── FASE 1: Scores de Modelo ────────────────────────
        peso_lgbm = self._score_to_peso(lgbm_raw)
        if peso_lgbm > 0:
            agravantes.append(AgravanteFator(
                "LGBM_SCORE",
                f"Score LGBM: {lgbm_raw*100:.1f}%",
                peso_lgbm, lgbm_raw,
            ))

        if if_active:
            peso_if = self._score_to_peso(if_score)
            if peso_if > 0:
                agravantes.append(AgravanteFator(
                    "IF_SCORE",
                    f"Score anomalia IF: {if_score*100:.1f}%",
                    peso_if, if_score,
                ))

        # ─── FASE 1b: Cascade Rules ─────────────────────────
        cascade_triggered = [c for c in cascade_results if c.triggered]
        if cascade_triggered:
            worst = cascade_triggered[0]
            peso_cascade = min(4, len(cascade_triggered) + 1)
            names = ", ".join(c.rule_name for c in cascade_triggered[:3])
            agravantes.append(AgravanteFator(
                f"CASCADE_{worst.rule_name}",
                f"Regra cascade: {names} — {worst.reason}",
                peso_cascade,
                {"rules": [c.rule_name for c in cascade_triggered]},
            ))

        # ─── FASE 2: Regras Clássicas ───────────────────────
        intervalo = _safe_float(f.get("qt_intervalo_transacao_minuto"))
        qt_total = _safe_int(f.get("qt_total_pix_trimestre"), 0)
        if intervalo is not None and 0 <= intervalo <= 30:
            if qt_total >= 3:
                agravantes.append(AgravanteFator(
                    "INTERVALO_30MIN",
                    f"Múltiplas transações em {intervalo:.0f} minutos",
                    self.config.peso_intervalo_30min_2tx, intervalo,
                ))
            else:
                agravantes.append(AgravanteFator(
                    "INTERVALO_30MIN",
                    f"Segunda transação em {intervalo:.0f} minutos",
                    self.config.peso_intervalo_30min_1tx, intervalo,
                ))

        idade = _safe_int(f.get("nr_idade"), 0)
        if idade >= 76:
            agravantes.append(AgravanteFator("IDADE", f"Cliente idoso vulnerável ({idade} anos)", 3, idade))
        elif idade >= 66:
            agravantes.append(AgravanteFator("IDADE", f"Cliente idoso ({idade} anos)", 2, idade))
        elif idade >= 60:
            agravantes.append(AgravanteFator("IDADE", f"Cliente sênior ({idade} anos)", 1, idade))

        tempo_rel = _safe_float(f.get("qt_tempo_relacionamento_mes"), 999)
        if tempo_rel <= 1:
            agravantes.append(AgravanteFator("RELACIONAMENTO", f"Cliente muito novo ({tempo_rel:.1f} meses)", 3, tempo_rel))
        elif tempo_rel <= 2:
            agravantes.append(AgravanteFator("RELACIONAMENTO", f"Cliente novo ({tempo_rel:.1f} meses)", 2, tempo_rel))
        elif tempo_rel <= 3:
            agravantes.append(AgravanteFator("RELACIONAMENTO", f"Cliente recente ({tempo_rel:.1f} meses)", 1, tempo_rel))

        chave_random = _safe_int(f.get("pix_key_random_flag"), 0)
        if chave_random == 1:
            agravantes.append(AgravanteFator(
                "CHAVE_ALEATORIA", "Transação para chave PIX aleatória",
                self.config.peso_chave_aleatoria, 1,
            ))

        hour = _safe_int(f.get("hour"), -1)
        if hour >= 0 and (hour >= 22 or hour < 6):
            agravantes.append(AgravanteFator(
                "HORARIO_NOTURNO",
                f"Transação fora do horário comercial ({hour}h)",
                3, hour,
            ))

        # ─── FASE 3: Topaz ──────────────────────────────────
        topaz_rejeitada = _safe_int(f.get("topaz_rejeitada_flag"), 0)
        topaz_score = _safe_float(f.get("topaz_score_filled"), 0)

        if topaz_rejeitada == 1:
            agravantes.append(AgravanteFator(
                "TOPAZ_REJEITADA",
                "VETO: Transação previamente rejeitada pelo Topaz",
                5, 1,
            ))
        elif topaz_score >= 4:
            agravantes.append(AgravanteFator("TOPAZ_RISCO_CRITICO", f"Score Topaz crítico: {topaz_score:.0f}/5", 4, topaz_score))
        elif topaz_score >= 3:
            agravantes.append(AgravanteFator("TOPAZ_RISCO_ALTO", f"Score Topaz alto: {topaz_score:.0f}/5", 3, topaz_score))
        elif topaz_score >= 2:
            agravantes.append(AgravanteFator("TOPAZ_RISCO_MODERADO", f"Score Topaz moderado: {topaz_score:.0f}/5", 2, topaz_score))

        # ─── FASE 4: Velocity ────────────────────────────────
        burst = _safe_int(f.get("burst_30m_flag"), 0)
        tx_30m = _safe_int(f.get("tx_count_prev_30m"), 0)
        qt_dia_max = _safe_int(f.get("qt_pix_dia_maximo_trimestre"), 0)
        ratio_valor = _safe_float(f.get("ratio_valor_mediana"))

        if (
            intervalo is not None and intervalo <= 5
            and ratio_valor is not None and ratio_valor >= 5.0
            and qt_dia_max >= 3
        ):
            agravantes.append(AgravanteFator(
                "VELOCITY_ESVAZIAMENTO_CRITICO",
                f"CRÍTICO: Esvaziamento — {qt_dia_max} PIX/dia máx, intervalo {intervalo:.0f}min, valor {ratio_valor:.1f}x mediana",
                4, {"intervalo": intervalo, "max_dia": qt_dia_max, "ratio": ratio_valor},
            ))
        elif (
            intervalo is not None and intervalo <= 10
            and ratio_valor is not None and ratio_valor >= 3.0
        ):
            agravantes.append(AgravanteFator(
                "VELOCITY_RAPIDA_ALTO_VALOR",
                f"PIX rápido ({intervalo:.0f}min) com valor alto ({ratio_valor:.1f}x mediana)",
                3, {"intervalo": intervalo, "ratio": ratio_valor},
            ))
        elif qt_dia_max >= 5 and burst == 1:
            media_diaria = qt_total / 90.0 if qt_total > 0 else 1
            if qt_dia_max >= media_diaria * 3:
                agravantes.append(AgravanteFator(
                    "VELOCITY_FREQUENCIA_ANORMAL",
                    f"Frequência anormal: {qt_dia_max} PIX/dia máx vs média {media_diaria:.1f}/dia",
                    2, {"max_dia": qt_dia_max, "media": media_diaria},
                ))

        # ─── FASE 5: Agravantes Estratégicos ─────────────────
        vl_pix = _safe_float(f.get("vl_pix"), 0)
        mediana = _safe_float(f.get("vl_mediana_pix_trimestre"), 0)

        if ratio_valor is not None and mediana > 0:
            if ratio_valor >= 10.0:
                agravantes.append(AgravanteFator(
                    "VALOR_ATIPICO",
                    f"CRÍTICO: Valor {ratio_valor:.1f}x acima da mediana (R${vl_pix:,.2f} vs R${mediana:,.2f})",
                    4, ratio_valor,
                ))
            elif ratio_valor >= 5.0:
                agravantes.append(AgravanteFator(
                    "VALOR_ATIPICO",
                    f"Valor muito alto: {ratio_valor:.1f}x acima da mediana",
                    3, ratio_valor,
                ))
            elif ratio_valor >= 3.0:
                agravantes.append(AgravanteFator(
                    "VALOR_ATIPICO",
                    f"Valor alto: {ratio_valor:.1f}x acima da mediana",
                    2, ratio_valor,
                ))

        first_receiver = _safe_int(f.get("first_receiver_flag"), 0)
        if first_receiver == 1 and ratio_valor is not None:
            if ratio_valor >= 5.0:
                agravantes.append(AgravanteFator(
                    "PRIMEIRO_ENVIO_ALTO",
                    f"CRÍTICO: Primeiro envio para recebedor com valor {ratio_valor:.1f}x mediana",
                    4, ratio_valor,
                ))
            elif ratio_valor >= 3.0:
                agravantes.append(AgravanteFator(
                    "PRIMEIRO_ENVIO_ALTO",
                    f"Primeiro envio com valor alto ({ratio_valor:.1f}x mediana)",
                    3, ratio_valor,
                ))
            elif ratio_valor >= 1.5:
                agravantes.append(AgravanteFator(
                    "PRIMEIRO_ENVIO_ALTO",
                    "Primeiro envio (valor acima do normal)",
                    2, ratio_valor,
                ))
            else:
                agravantes.append(AgravanteFator(
                    "PRIMEIRO_ENVIO",
                    "Primeiro envio para recebedor desconhecido",
                    1, ratio_valor,
                ))

        if qt_dia_max >= 10:
            media_diaria = qt_total / 90.0 if qt_total > 0 else 1
            if qt_dia_max >= media_diaria * 5:
                agravantes.append(AgravanteFator(
                    "VOLUME_TRIMESTRAL",
                    f"Volume anômalo: {qt_dia_max} PIX em 1 dia (média {media_diaria:.1f}/dia)",
                    3, qt_dia_max,
                ))
            elif qt_dia_max >= media_diaria * 3:
                agravantes.append(AgravanteFator(
                    "VOLUME_TRIMESTRAL",
                    f"Volume elevado: {qt_dia_max} PIX em 1 dia",
                    2, qt_dia_max,
                ))

        mediana_intervalo = _safe_float(f.get("qt_intervalo_mediana_trimestre"), 0)
        if intervalo is not None and mediana_intervalo > 0 and intervalo >= 0:
            razao_int = intervalo / mediana_intervalo
            if razao_int <= 0.05:
                agravantes.append(AgravanteFator(
                    "INTERVALO_RELATIVO",
                    f"Velocidade crítica: {intervalo:.0f}min vs mediana {mediana_intervalo:.0f}min",
                    3, razao_int,
                ))
            elif razao_int <= 0.15:
                agravantes.append(AgravanteFator(
                    "INTERVALO_RELATIVO",
                    f"Velocidade atípica: {intervalo:.0f}min vs mediana {mediana_intervalo:.0f}min",
                    2, razao_int,
                ))
            elif razao_int <= 0.30:
                agravantes.append(AgravanteFator(
                    "INTERVALO_RELATIVO",
                    f"Intervalo abaixo do padrão pessoal",
                    1, razao_int,
                ))

        # ─── FASE 6: Renda e Perfil (v2.1b) ─────────────────
        pix_over_100 = _safe_int(f.get("pix_over_100pct_renda_flag"), 0)
        ratio_renda = _safe_float(f.get("ratio_pix_renda"))
        vl_renda = _safe_float(f.get("vl_renda_cliente"), 0)
        if pix_over_100 == 1 and ratio_renda is not None:
            agravantes.append(AgravanteFator(
                "RENDA_INCOMPATIVEL",
                f"PIX de R${vl_pix:,.2f} = {ratio_renda:.0%} da renda (R${vl_renda:,.2f})",
                4, ratio_renda,
            ))
        elif _safe_int(f.get("pix_over_50pct_renda_flag"), 0) == 1 and ratio_renda is not None:
            agravantes.append(AgravanteFator(
                "RENDA_METADE_COMPROMETIDA",
                f"PIX compromete {ratio_renda:.0%} da renda mensal",
                2, ratio_renda,
            ))

        if _safe_int(f.get("perfil_vulneravel_se_flag"), 0) == 1:
            agravantes.append(AgravanteFator(
                "PERFIL_VULNERAVEL",
                f"Alta vulnerabilidade: viúvo(a), {idade} anos, sem dependentes",
                3, 1,
            ))

        latencia = _safe_float(f.get("latencia_rede_ms_final"))
        if latencia is not None:
            if latencia >= 5000:
                agravantes.append(AgravanteFator(
                    "LATENCIA_REDE_ALTA",
                    f"CRÍTICO: Latência {latencia:.0f}ms — possível acesso remoto",
                    4, latencia,
                ))
            elif latencia >= 2000:
                agravantes.append(AgravanteFator(
                    "LATENCIA_REDE_ALTA",
                    f"Latência suspeita ({latencia:.0f}ms) — possível VPN",
                    3, latencia,
                ))
            elif latencia >= 1000:
                agravantes.append(AgravanteFator(
                    "LATENCIA_REDE_ALTA",
                    f"Latência acima do normal ({latencia:.0f}ms)",
                    2, latencia,
                ))

        tempo_interacao = _safe_float(f.get("tempo_interacao_ms_final"))
        if tempo_interacao is not None:
            if tempo_interacao < 60:
                agravantes.append(AgravanteFator(
                    "INTERACAO_AUTOMATIZADA",
                    f"CRÍTICO: Automatizada ({tempo_interacao:.0f}ms)",
                    4, tempo_interacao,
                ))
            elif tempo_interacao < 80:
                agravantes.append(AgravanteFator(
                    "INTERACAO_AUTOMATIZADA",
                    f"Interação suspeita ({tempo_interacao:.0f}ms — padrão bot)",
                    3, tempo_interacao,
                ))

        login_senha = _safe_int(f.get("is_login_senha_flag"), 0)
        if login_senha == 1 and idade >= 60 and vl_pix >= 1000:
            agravantes.append(AgravanteFator(
                "LOGIN_SENHA_IDOSO",
                f"Idoso ({idade}a) usando senha em PIX R${vl_pix:,.2f}",
                2, {"idade": idade, "vl_pix": vl_pix},
            ))

        # ─── FASE 7: SE + BEHAVIORAL ────────────────────────
        if se_score >= 40 and se_patterns:
            peso_se = 4 if se_score >= 60 else 3
            agravantes.append(AgravanteFator(
                f"ENG_SOCIAL_{se_patterns[0]}",
                f"Engenharia social: {', '.join(se_patterns[:2])}",
                peso_se,
                {"se_score": se_score, "patterns": se_patterns},
            ))

        if behavioral_score >= 20:
            peso_beh = 3 if behavioral_score >= 50 else 2 if behavioral_score >= 30 else 1
            agravantes.append(AgravanteFator(
                "BEHAVIORAL_ANOMALO",
                f"Risco comportamental: {behavioral_score:.0f}/100",
                peso_beh,
                behavioral_score,
            ))

        # ─── ATENUANTE ──────────────────────────────────────
        if _safe_int(f.get("is_agendamento_recorrente_flag"), 0) == 1:
            agravantes.append(AgravanteFator(
                "AGENDAMENTO_RECORRENTE",
                "PIX recorrente agendado — risco atenuado",
                -1, 1,
            ))

        return agravantes

    # ==========================================================
    # VETO
    # ==========================================================
    def _aplicar_veto(
        self,
        score_mapped: float,
        lgbm_raw: float,
        if_score: float,
        if_active: bool,
        cascade_triggered: bool,
    ) -> Tuple[float, Optional[str]]:
        """
        Aplica regras de veto.

        - Cascade triggered: mínimo CONFIRMAR
        - 1 modelo ≥ veto_threshold: mínimo CONFIRMAR
        - 2 modelos ≥ veto_threshold: mínimo BLOQUEAR
        """
        threshold = self.config.veto_threshold
        vetos = []

        if lgbm_raw >= threshold:
            vetos.append(f"LGBM={lgbm_raw*100:.1f}%")
        if if_active and if_score >= threshold:
            vetos.append(f"IF={if_score*100:.1f}%")

        if len(vetos) >= 2:
            score_ajustado = max(score_mapped, self.config.threshold_bloquear)
            desc = f"VETO BLOQUEAR: {' + '.join(vetos)} ≥ {threshold*100:.0f}%"
            logger.info(f"{desc} | Score: {score_mapped:.1f} → {score_ajustado:.1f}")
            return score_ajustado, desc

        if len(vetos) == 1:
            score_ajustado = max(score_mapped, self.config.threshold_confirmar)
            desc = f"VETO CONFIRMAR: {vetos[0]} ≥ {threshold*100:.0f}%"
            logger.info(f"{desc} | Score: {score_mapped:.1f} → {score_ajustado:.1f}")
            return score_ajustado, desc

        # Cascade veto: garante ao menos CONFIRMAR
        if cascade_triggered:
            score_ajustado = max(score_mapped, self.config.threshold_confirmar)
            desc = "VETO CONFIRMAR: Cascade rule triggered"
            logger.info(f"{desc} | Score: {score_mapped:.1f} → {score_ajustado:.1f}")
            return score_ajustado, desc

        return score_mapped, None

    # ==========================================================
    # DECISÃO
    # ==========================================================
    def _classificar(self, score: float) -> str:
        if score >= self.config.threshold_bloquear:
            return "BLOQUEAR"
        if score >= self.config.threshold_confirmar:
            return "CONFIRMAR"
        return "APROVAR"

    # ==========================================================
    # API PRINCIPAL
    # ==========================================================
    def decide(
        self,
        features: Dict[str, Any],
        se_result: Optional[Dict[str, Any]] = None,
        behavioral_result: Optional[Dict[str, Any]] = None,
    ) -> DecisionResult:
        """
        Executa a decisão completa para uma transação.

        Fluxo v2.1:
          1. LGBM Score
          2. Cascade Rules (se LGBM < threshold)
          3. IF Score + Boost (se LGBM < threshold)
          4. Ensemble (LGBM + boost + cascade)
          5. Mapeamento 0-100
          6. Agravantes (24 fatores, 7 fases)
          7. Vetos
          8. Decisão final
        """
        t0 = time.perf_counter()

        # --- Extrair SE + Behavioral ---
        se_score = 0.0
        se_patterns: List[str] = []
        if se_result:
            se_score = float(se_result.get("se_score", 0))
            se_patterns = se_result.get("patterns", [])
            if se_patterns and isinstance(se_patterns[0], dict):
                se_patterns = [p.get("pattern_name", "") for p in se_patterns]

        behavioral_score = 0.0
        behavioral_factors: List[Dict[str, Any]] = []
        if behavioral_result:
            behavioral_score = float(behavioral_result.get("behavioral_score", 0))
            behavioral_factors = behavioral_result.get("risk_factors", [])

        # --- 1. LGBM Scoring ---
        lgbm_raw = self._score_lgbm(features)

        # --- 2. Cascade Rules (quando LGBM não flagga) ---
        cascade_results: List[CascadeResult] = []
        cascade_triggered = False
        cascade_rule_names: List[str] = []

        if self.config.cascade_enabled and lgbm_raw < self.config.lgbm_threshold:
            cascade_results = self.cascade_engine.evaluate(features, lgbm_raw)
            cascade_triggered = any(c.triggered for c in cascade_results)
            cascade_rule_names = [c.rule_name for c in cascade_results if c.triggered]

        # --- 3. IF Score + Boost (complementar) ---
        if_score, if_raw, if_active, if_boost = self._score_if(features, lgbm_raw)

        # --- 4. Ensemble ---
        ensemble_raw = self._calculate_ensemble(
            lgbm_raw, if_score, if_active, if_boost, cascade_triggered
        )

        # --- 5. Mapeamento 0-100 ---
        score_mapped = self._map_to_score(ensemble_raw)
        lgbm_mapped = self._map_to_score(lgbm_raw)

        # --- 6. Agravantes ---
        agravantes = self._calcular_agravantes(
            features, lgbm_raw, if_score, if_active,
            cascade_results, se_score, se_patterns, behavioral_score
        )

        atenuantes = []
        agravantes_positivos = []
        for a in agravantes:
            if a.peso < 0:
                atenuantes.append(a.descricao)
            else:
                agravantes_positivos.append(a)

        peso_total = sum(a.peso for a in agravantes_positivos)

        # Agravantes adicionam até +15 pontos ao score
        peso_normalizado = peso_total / max(self.config.peso_maximo, 1)
        bonus_agravantes = peso_normalizado * 15.0
        score_com_agravantes = min(100.0, score_mapped + bonus_agravantes)

        if atenuantes:
            score_com_agravantes = max(0.0, score_com_agravantes - 5.0)

        # --- 7. Veto ---
        score_final, veto_desc = self._aplicar_veto(
            score_com_agravantes, lgbm_raw, if_score, if_active, cascade_triggered
        )

        # --- 8. Decisão ---
        decisao = self._classificar(score_final)

        latency = (time.perf_counter() - t0) * 1000

        return DecisionResult(
            decisao=decisao,
            score_final=score_final,
            score_raw=ensemble_raw,
            score_lgbm_raw=lgbm_raw,
            score_lgbm_mapped=lgbm_mapped,
            score_if=if_score,
            score_if_raw=if_raw,
            if_active=if_active,
            if_boost_applied=if_boost,
            cascade_triggered=cascade_triggered,
            cascade_rules=cascade_rule_names,
            rule_score_raw=_safe_float(features.get("rule_score_raw"), 0),
            rule_score_normalized=_safe_float(features.get("rule_score_normalized"), 0),
            peso_total=peso_total,
            peso_maximo=self.config.peso_maximo,
            agravantes=agravantes_positivos,
            se_score=se_score,
            se_patterns=se_patterns,
            behavioral_score=behavioral_score,
            behavioral_factors=behavioral_factors,
            veto_aplicado=veto_desc,
            atenuantes=atenuantes,
            latency_ms=latency,
            transaction_id=str(features.get("transaction_id", "")),
            customer_id=str(features.get("customer_id", "")),
        )

    # ==========================================================
    # STATUS
    # ==========================================================
    def get_status(self) -> Dict[str, Any]:
        """Retorna status do engine para health check."""
        metricas = self.scoring_config.get("metricas_teste", {})
        return {
            "engine_version": "2.1",
            "available": self.available,
            "load_time_ms": round(self._load_time, 1) if self._load_time else None,
            "lgbm": self.lgbm_model is not None,
            "lgbm_features": len(self.lgbm_features),
            "lgbm_threshold": self.config.lgbm_threshold,
            "if_enabled": self.if_model is not None,
            "if_features": len(self.if_features),
            "if_ensemble": {
                "high_threshold": self.config.if_high_threshold,
                "very_high_threshold": self.config.if_very_high_threshold,
                "boost_high": self.config.if_boost_high,
                "boost_very_high": self.config.if_boost_very_high,
            },
            "cascade_enabled": self.config.cascade_enabled,
            "cascade_rules": len(self.cascade_rules_config),
            "scoring_anchors": len(self.anchors_raw),
            "scoring_version": self.scoring_config.get("versao", "N/A"),
            "thresholds": {
                "confirmar": self.config.threshold_confirmar,
                "bloquear": self.config.threshold_bloquear,
                "veto": self.config.veto_threshold,
            },
            "metricas_validacao": {
                "recall_bloquear": metricas.get("recall_bloquear", "N/A"),
                "precision_bloquear": metricas.get("precision_bloquear", "N/A"),
                "f1_bloquear": metricas.get("f1_bloquear", "N/A"),
                "gap": metricas.get("gap_fraud_min_vs_normal_p999", "N/A"),
            },
        }


# =========================================================
# MODULE-LEVEL HELPERS
# =========================================================
def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    if val is None:
        return default
    try:
        v = float(val)
        return default if v != v else v
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    try:
        v = float(val)
        return default if v != v else int(v)
    except (ValueError, TypeError):
        return default
