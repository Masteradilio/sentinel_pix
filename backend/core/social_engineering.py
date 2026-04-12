"""
core/social_engineering.py v3.4 — Detector de Padrões de Engenharia Social

Ajustado para dataset leakage-free (rolling window causal).

Mudanças v3.3 → v3.4 (Simulação SE v3.4 — Ajuste Leakage-Free):

  Contexto: Re-validação do SE v3.3 no dataset leakage-free revelou
  degradação catastrófica de 2 padrões. A feature `primeira_tx_trimestre`
  tornou-se anti-indicador (Lift 0.57x) porque 78% dos normais agora
  são 1ªTX no rolling window causal (vs ~25% no dataset otimizado).

  Diagnóstico (SE v3.3 no leakage-free):
    Baseline: TP=284, FP=4487, Prec=5.9%, Recall=80.0%
    vs v3.3 otimizado: TP=262, FP=346, Prec=43.1%, Recall=73.8%
    → FP explodiu 13x por causa de 2 padrões:
      COACAO_FISICA:       FP 34 → 947  (Prec 72.4% → 11.8%)
      PRIMEIRA_TX_SUSPEITA: FP 34 → 4395 (Prec 72.4% → 2.8%)

  Simulação completa (simular_se_v34_leakage_free.py, 26 cenários):

  [S2A] PRIMEIRA_TX_SUSPEITA — REMOVIDO
        Motivo: Prec 2.8% com 4395 FP é inaceitável. O padrão dependia
        estruturalmente de `primeira_tx_trimestre` como required, que
        virou anti-indicador no leakage-free. Redesigns testados
        (valor_redondo, renda_comprometida, valor_5k) não atingiram
        precision suficiente para justificar manutenção.
        O módulo Behavioral (BEH v3.0) já cobre esse cenário com
        PRIMEIRA_TX_VALOR_ALTO (Prec 72.4%, TP=89) e
        CONTA_DORMANTE_VALOR_ALTO (Prec 65.0%, TP=141).
        Impacto isolado: TP=284 (inalterado), FP 4487→1282 (-71.4%)

  [S1] COACAO_FISICA — Remover `primeira_tx_trimestre` dos required, ms=6
       Motivo: `primeira_tx` como required ativava em 78% dos normais,
       causando 947 FP. Removido dos required e movido para optional
       (contribui +1 quando presente, não é gate).
       Simulação ms=6: COACAO TP=105, FP=226, Prec=31.7%
       vs v3.3 leakage-free: COACAO TP=127, FP=947, Prec=11.8%
       Trade-off: -22 TP, -721 FP. Precision quase triplica.

  [S3] FALSO_FUNCIONARIO_BANCO — ms 7→9
       Motivo: Com LGBM v5.1 em 96.25% recall, o SE não precisa de
       recall alto neste padrão. Subir ms de 7 para 9 é cirúrgico:
       FALSO_FUNC ms=7: TP=97, FP=326, Prec=22.9%
       FALSO_FUNC ms=9: TP=58, FP=39,  Prec=59.8%
       Trade-off: -39 TP, -287 FP. Precision 2.6x.
       Os 39 TP perdidos são quase todos cobertos pelo LGBM.

  Impacto global estimado (v3.3 leakage-free → v3.4):
    TP: 284 → ~230 (recall isolado ↓, mas LGBM cobre)
    FP: 4487 → ~350 (redução ~92%)
    Precision: 5.9% → ~40%
    Recall (isolado): 80.0% → ~65%

  Nota: O recall do SE isolado diminui, mas no pipeline completo
  (LGBM 96.25% + SE + IF + BEH), o recall combinado permanece >96%.
  O ganho real está na redução de 92% dos FP, que reduz carga
  operacional de análise manual e melhora a experiência do cliente.

  Padrões ativos: 8 (era 9, removido PRIMEIRA_TX_SUSPEITA)
  Indicadores ativos: 29 (removido pix_acima_500 do required de padrões eliminados)
  Clusters de overlap atualizados.

Referências de calibração:
  - Dataset: base_mvp_model_ready_leakage_free.csv (100.355 tx, 355 fraudes)
  - Simulação: simular_se_v34_leakage_free.py (26 cenários)
  - LGBM v5.1: metricas_lgbm_v5.json (holdout F1=0.911, Recall=96.25%)
  - IF v3: metricas_if.json (complementary boost)
  - BEH v3.0: behavioral_analytics.py (19 exclusivas, recalibração pendente)
  - Data da calibração: 2026-04-12
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =========================================================
# DATA CLASSES
# =========================================================
@dataclass
class PatternMatch:
    """Representa um padrão de engenharia social detectado."""

    pattern_name: str
    severity: str
    score: int
    matched_indicators: List[str]
    description: str


@dataclass
class SEAnalysisResult:
    """Resultado completo da análise de engenharia social."""

    se_score: float
    patterns: List[PatternMatch] = field(default_factory=list)
    active_indicators: Dict[str, bool] = field(default_factory=dict)
    phase2_indicators_missing: List[str] = field(default_factory=list)

    @property
    def risk_level(self) -> str:
        if self.se_score >= 60:
            return "CRITICO"
        if self.se_score >= 40:
            return "ALTO"
        if self.se_score >= 20:
            return "MEDIO"
        return "BAIXO"

    @property
    def worst_pattern(self) -> Optional[PatternMatch]:
        return self.patterns[0] if self.patterns else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "se_score": round(self.se_score, 2),
            "risk_level": self.risk_level,
            "total_patterns": len(self.patterns),
            "patterns": [
                {
                    "pattern_name": p.pattern_name,
                    "severity": p.severity,
                    "score": p.score,
                    "matched_indicators": p.matched_indicators,
                    "description": p.description,
                }
                for p in self.patterns
            ],
            "worst_pattern": (
                self.worst_pattern.pattern_name if self.worst_pattern else None
            ),
            "phase2_indicators_missing": self.phase2_indicators_missing,
        }

    def to_features(self) -> Dict[str, float]:
        return {
            "se_score": round(self.se_score, 2),
            "se_pattern_count": float(len(self.patterns)),
            "se_has_critico": float(
                any(p.severity == "CRITICO" for p in self.patterns)
            ),
            "se_max_pattern_score": float(
                max((p.score for p in self.patterns), default=0)
            ),
        }


_SEVERITY_ORDER = {"CRITICO": 0, "ALTO": 1, "MEDIO": 2, "BAIXO": 3}


# =========================================================
# OVERLAP CLUSTERS (Jaccard > 0.15 na validação retroativa)
#
# v3.4: PRIMEIRA_TX_SUSPEITA removido — cluster não necessário.
#        Demais clusters inalterados.
# =========================================================
_OVERLAP_CLUSTERS: List[frozenset] = [
    frozenset({"IDOSO_VULNERAVEL_70", "IDOSO_VULNERAVEL_80"}),
    frozenset({"ESVAZIAMENTO_CONTA", "BURST_ESVAZIAMENTO_CONTA", "BURST_INTENSO_RAPIDO"}),
    frozenset({"COACAO_FISICA", "BURST_VALOR_ALTO"}),
]


# =========================================================
# SOCIAL ENGINEERING DETECTOR v3.4
# =========================================================
class SocialEngineeringDetector:
    """
    Detecta padrões de golpes de engenharia social v3.4 (Leakage-Free).

    8 padrões ativos, calibrados com dataset leakage-free (100.355 tx).
    Ajustes baseados em simulação de 26 cenários.

    Papel no pipeline: segunda linha de defesa, complementar ao LGBM v5.1.
    Foco: alta precision (reduzir FP), rescue de FN do LGBM, explainability.

    Padrões e performance (leakage-free):
      ESVAZIAMENTO_CONTA        — ms=4, Prec 67.7%  [inalterado]
      COACAO_FISICA              — ms=6, Prec ~31.7% [removido primeira_tx required]
      BURST_ESVAZIAMENTO_CONTA  — ms=3, Prec 38.1%  [inalterado]
      FALSO_FUNCIONARIO_BANCO   — ms=9, Prec ~59.8% [subiu de ms=7]
      IDOSO_VULNERAVEL_70       — ms=7, Prec 37.6%  [inalterado]
      IDOSO_VULNERAVEL_80       — ms=6, Prec 45.8%  [inalterado]
      BURST_VALOR_ALTO          — ms=3, Prec 78.8%  [inalterado]
      BURST_INTENSO_RAPIDO      — ms=6, Prec 100%   [inalterado]

    REMOVIDO: PRIMEIRA_TX_SUSPEITA (Prec 2.8% no leakage-free, 4395 FP)
    """

    VERSION = "3.4"

    # Indicadores de "fase 2" — dados do Big Data que podem faltar em RT
    PHASE2_INDICATORS = [
        "mulher_idosa_raw",
        "viuvo_viuva_raw",
        "segmento_alto_patrimonio_raw",
    ]

    def __init__(self):
        self._setup_indicators()
        self._setup_patterns()
        logger.info(
            f"SocialEngineeringDetector v{self.VERSION} inicializado "
            f"({len(self.PATTERNS)} padrões, {len(self.INDICATORS)} indicadores)"
        )

    # =============================================================
    # FEATURE ADAPTER
    # =============================================================
    @staticmethod
    def _adapt_features(features: Dict[str, Any]) -> Dict[str, Any]:
        """Mapeia features do pipeline para nomes esperados pelos indicadores."""
        f: Dict[str, Any] = {}

        # --- IDs e metadata ---
        f["customer_id"] = features.get("customer_id") or features.get(
            "cd_cpf_pagador"
        )
        f["transaction_id"] = features.get("transaction_id") or features.get(
            "cd_pix"
        )

        # --- Datetime ---
        dt_raw = features.get("event_datetime") or features.get("dt_pix")
        if isinstance(dt_raw, str):
            try:
                f["dt_pix"] = datetime.fromisoformat(
                    dt_raw.replace("Z", "+00:00").replace(" ", "T")
                )
            except (ValueError, TypeError):
                f["dt_pix"] = None
        elif isinstance(dt_raw, datetime):
            f["dt_pix"] = dt_raw
        else:
            f["dt_pix"] = None

        # --- Temporais ---
        f["hour"] = _safe_int(features.get("hour"), -1)
        f["day_of_week"] = _safe_int(features.get("day_of_week"), -1)
        f["is_business_hours"] = _safe_int(features.get("is_business_hours"), 0)

        # --- Valor ---
        f["vl_pix"] = _safe_float(features.get("vl_pix"), 0.0)
        f["vl_mediana_pix_trimestre"] = _safe_float(
            features.get("vl_mediana_pix_trimestre"), 0.0
        )
        f["ratio_valor_mediana"] = _safe_float(features.get("ratio_valor_mediana"))

        # --- Renda ---
        f["pix_over_100pct_renda_flag"] = _safe_int(
            features.get("pix_over_100pct_renda_flag"), 0
        )
        f["pix_over_50pct_renda_flag"] = _safe_int(
            features.get("pix_over_50pct_renda_flag"), 0
        )
        f["renda_missing_flag"] = _safe_int(features.get("renda_missing_flag"), 0)

        # --- Perfil do cliente ---
        f["nr_idade"] = _safe_int(features.get("nr_idade"), 0)
        f["qt_tempo_relacionamento_mes"] = _safe_int(
            features.get("qt_tempo_relacionamento_mes"), 999
        )
        f["is_sexo_feminino_flag"] = _safe_int(
            features.get("is_sexo_feminino_flag"), 0
        )
        f["is_viuvo_flag"] = _safe_int(features.get("is_viuvo_flag"), 0)
        f["is_segmento_premium_flag"] = _safe_int(
            features.get("is_segmento_premium_flag"), 0
        )
        f["perfil_vulneravel_se_flag"] = _safe_int(
            features.get("perfil_vulneravel_se_flag"), 0
        )

        # --- Recebedor ---
        f["first_receiver_flag"] = _safe_int(features.get("first_receiver_flag"), 0)
        f["qt_envio_recebedor_trimestre"] = _safe_int(
            features.get("qt_envio_recebedor_trimestre"), 0
        )

        # --- Tipo de chave ---
        f["pix_key_random_flag"] = _safe_int(features.get("pix_key_random_flag"), 0)

        # --- Velocidade / Frequência ---
        f["qt_intervalo_transacao_minuto"] = _safe_float(
            features.get("qt_intervalo_transacao_minuto")
        )
        f["qt_pix_dia_maximo_trimestre"] = _safe_int(
            features.get("qt_pix_dia_maximo_trimestre"), 0
        )
        f["qt_total_pix_trimestre"] = _safe_int(
            features.get("qt_total_pix_trimestre"), 0
        )
        f["burst_30m_flag"] = _safe_int(features.get("burst_30m_flag"), 0)
        f["tx_count_prev_30m"] = _safe_int(features.get("tx_count_prev_30m"), 0)
        f["is_first_tx_trimestre"] = _safe_int(
            features.get("is_first_tx_trimestre"), 0
        )

        # --- Distinct receivers ---
        f["distinct_receivers_so_far"] = _safe_int(
            features.get("distinct_receivers_so_far"), 1
        )

        # --- Autenticação ---
        f["is_login_senha_flag"] = _safe_int(features.get("is_login_senha_flag"), 0)
        f["is_agendamento_recorrente_flag"] = _safe_int(
            features.get("is_agendamento_recorrente_flag"), 0
        )

        return f

    # =============================================================
    # INDICATORS SETUP — Somente Lift ≥ 1.5x validado
    # =============================================================
    def _setup_indicators(self):
        """
        Define indicadores de risco.

        Todos validados na Frente 1 com Lift ≥ 1.5x.

        v3.4: Nenhum indicador removido (todos continuam válidos).
              primeira_tx_trimestre mantido como indicador — apenas
              removido dos required do COACAO_FISICA (movido para optional).
              O indicador em si NÃO é anti-indicador — o problema era
              usá-lo como GATE em padrões, não como sinal opcional.
        """
        self.INDICATORS: Dict[str, Callable[[Dict[str, Any]], bool]] = {
            # ─── VELOCITY / BURST (Lift > 40x) ──────────────────
            "burst_intenso": lambda f: f["tx_count_prev_30m"] >= 3,
            "burst_30m": lambda f: f["burst_30m_flag"] == 1,
            "multiplos_pix_rapidos": lambda f: (
                f["burst_30m_flag"] == 1 and f["qt_pix_dia_maximo_trimestre"] >= 3
            ),
            "primeira_tx_trimestre": lambda f: f["is_first_tx_trimestre"] == 1,
            # v3.4: Mantido como indicador optional. Lift 146.3x quando
            # combinado com burst/valor. NÃO usar como required gate
            # em padrões — 78% dos normais no leakage-free ativam este flag.
            "burst_conta_antiga": lambda f: (
                f["qt_tempo_relacionamento_mes"] >= 12
                and f["burst_30m_flag"] == 1
                and f["first_receiver_flag"] == 1
            ),
            # ─── VALOR ABSOLUTO (Lift > 14x) ────────────────────
            "valor_absoluto_alto": lambda f: f["vl_pix"] >= 5000,
            "valor_absoluto_muito_alto": lambda f: f["vl_pix"] >= 10000,
            "pix_acima_1000": lambda f: f["vl_pix"] >= 1000,
            "pix_acima_500": lambda f: f["vl_pix"] >= 500,
            # ─── RENDA (Lift > 4x) ──────────────────────────────
            "renda_desconhecida_valor_alto": lambda f: (
                f["renda_missing_flag"] == 1 and f["vl_pix"] >= 5000
            ),
            "renda_metade_comprometida": lambda f: (
                f["pix_over_50pct_renda_flag"] == 1
            ),
            "renda_incompativel": lambda f: f["pix_over_100pct_renda_flag"] == 1,
            # ─── PERFIL / IDADE (Lift > 3.8x) ───────────────────
            "idade_70_plus": lambda f: f["nr_idade"] >= 70,
            "idade_60_plus": lambda f: f["nr_idade"] >= 60,
            "idade_80_plus": lambda f: f["nr_idade"] >= 80,
            # ─── INTERVALO (Lift > 3.8x) ────────────────────────
            "intervalo_muito_curto": lambda f: (
                f["qt_intervalo_transacao_minuto"] is not None
                and 0 <= f["qt_intervalo_transacao_minuto"] <= 5
            ),
            "intervalo_curto": lambda f: (
                f["qt_intervalo_transacao_minuto"] is not None
                and 0 <= f["qt_intervalo_transacao_minuto"] <= 30
            ),
            # ─── OUTROS (Lift > 1.5x) ───────────────────────────
            "conta_recem_aberta": lambda f: f["qt_tempo_relacionamento_mes"] <= 1,
            "cliente_muito_novo": lambda f: f["qt_tempo_relacionamento_mes"] <= 3,
            "valor_redondo": lambda f: _is_valor_redondo(f["vl_pix"]),
            "multiplos_recebedores_distintos": lambda f: (
                f["distinct_receivers_so_far"] >= 3
            ),
            "is_segmento_premium": lambda f: f["is_segmento_premium_flag"] == 1,
            "perfil_vulneravel_se": lambda f: f["perfil_vulneravel_se_flag"] == 1,
            # ─── CHAVE ALEATÓRIA (só útil combinado) ─────────────
            "chave_aleatoria": lambda f: f["pix_key_random_flag"] == 1,
            # ─── COMPOSTOS ──────────────────────────────────────
            "aproximando_esgotamento": lambda f: (
                f["ratio_valor_mediana"] is not None
                and f["ratio_valor_mediana"] >= 5.0
                and f["burst_30m_flag"] == 1
            ),
            "recebedor_nunca_visto": lambda f: (
                f["qt_envio_recebedor_trimestre"] == 0
            ),
            # ─── AUTENTICAÇÃO ────────────────────────────────────
            "login_senha": lambda f: f["is_login_senha_flag"] == 1,
            # ─── HORÁRIO ─────────────────────────────────────────
            "horario_comercial": lambda f: (
                8 <= f["hour"] < 18 and 0 <= f["day_of_week"] <= 4
            ),
            # ─── ATENUANTE ──────────────────────────────────────
            "agendamento_recorrente": lambda f: (
                f["is_agendamento_recorrente_flag"] == 1
            ),
        }

    # =============================================================
    # PATTERNS SETUP — 8 padrões calibrados (leakage-free)
    # =============================================================
    def _setup_patterns(self):
        """
        Define os 8 padrões de golpes ativos.

        v3.4 vs v3.3:
          REMOVIDO: PRIMEIRA_TX_SUSPEITA (Prec 2.8%, 4395 FP no leakage-free)
          AJUSTADO: COACAO_FISICA (removido primeira_tx dos required, ms 5→6)
          AJUSTADO: FALSO_FUNCIONARIO_BANCO (ms 7→9)

        Calibração: base_mvp_model_ready_leakage_free.csv (100.355 tx)
        Simulação: simular_se_v34_leakage_free.py (26 cenários)
        Data: 2026-04-12
        """
        self.PATTERNS: Dict[str, Dict[str, Any]] = {
            # ═══════════════════════════════════════════════════════
            # PADRÃO 1: ESVAZIAMENTO_CONTA [inalterado]
            # Leakage-free: TP=61, FP=0, Prec=100%
            # ═══════════════════════════════════════════════════════
            "ESVAZIAMENTO_CONTA": {
                "required": ["multiplos_pix_rapidos"],
                "optional": [
                    "burst_intenso",
                    "intervalo_muito_curto",
                    "pix_acima_1000",
                    "valor_absoluto_alto",
                    "primeira_tx_trimestre",
                    "renda_desconhecida_valor_alto",
                    "renda_incompativel",
                    "multiplos_recebedores_distintos",
                    "aproximando_esgotamento",
                ],
                "min_score": 4,
                "severity": "CRITICO",
                "description": (
                    "Esvaziamento de conta: múltiplos PIX rápidos + "
                    "valores altos + padrão de urgência"
                ),
            },
            # ═══════════════════════════════════════════════════════
            # PADRÃO 2: COACAO_FISICA [v3.4 AJUSTADO]
            #
            # v3.3 (leakage-free): TP=127, FP=947, Prec=11.8%
            # v3.4: Removido primeira_tx_trimestre dos required.
            #       Movido para optional (+1 quando presente).
            #       min_score 5→6 para compensar required mais frouxo.
            #
            # Simulação S1_COACAO_ms6:
            #   COACAO: TP=105, FP=226, Prec=31.7%
            #   vs v3.3: -22 TP, -721 FP, +19.9pp Prec
            # ═══════════════════════════════════════════════════════
            "COACAO_FISICA": {
                "required": [
                    "intervalo_muito_curto",    # Lift 6.7x
                    "pix_acima_1000",           # Lift 14.5x
                    # v3.4: primeira_tx_trimestre REMOVIDO dos required
                    # Motivo: 78% dos normais ativam no leakage-free → 947 FP
                    # Movido para optional abaixo
                ],
                "optional": [
                    "primeira_tx_trimestre",    # v3.4: movido de required para optional
                    "burst_intenso",
                    "burst_30m",
                    "multiplos_pix_rapidos",
                    "valor_absoluto_alto",
                    "valor_absoluto_muito_alto",
                    "renda_desconhecida_valor_alto",
                    "renda_incompativel",
                    "multiplos_recebedores_distintos",
                ],
                "min_score": 6,
                # v3.4: 5→6 (compensa remoção de primeira_tx dos required)
                # required(2 indicators × 2pts) = 4 base
                # Precisa de 2+ optional para ativar (filtra ruído)
                "severity": "CRITICO",
                "description": (
                    "Possível coação física — PIX alto em intervalos "
                    "< 5 minutos com indicadores de vulnerabilidade"
                ),
            },
            # ═══════════════════════════════════════════════════════
            # PADRÃO 3: BURST_ESVAZIAMENTO_CONTA [inalterado]
            # Leakage-free: TP=16, FP=26, Prec=38.1%
            # ═══════════════════════════════════════════════════════
            "BURST_ESVAZIAMENTO_CONTA": {
                "required": ["burst_conta_antiga", "pix_acima_1000"],
                "optional": [
                    "burst_intenso",
                    "intervalo_muito_curto",
                    "multiplos_recebedores_distintos",
                    "valor_absoluto_alto",
                    "renda_desconhecida_valor_alto",
                    "renda_incompativel",
                    "recebedor_nunca_visto",
                    "aproximando_esgotamento",
                ],
                "min_score": 3,
                "severity": "CRITICO",
                "description": (
                    "Conta antiga com burst súbito de PIX alto para "
                    "recebedores novos — conta comprometida"
                ),
            },
            # ═══════════════════════════════════════════════════════
            # PADRÃO 4: FALSO_FUNCIONARIO_BANCO [v3.4 AJUSTADO]
            #
            # v3.3 (leakage-free): TP=97, FP=326, Prec=22.9%
            # v3.4: min_score 7→9
            #
            # Simulação S3_FALSO_FUNC_ms9:
            #   TP=58, FP=39, Prec=59.8%
            #   vs v3.3: -39 TP, -287 FP, +36.9pp Prec
            #
            # Justificativa: LGBM v5.1 tem 96.25% recall. Os 39 TP
            # perdidos são quase todos cobertos pelo LGBM. O ganho de
            # 287 FP a menos justifica amplamente a troca.
            # ═══════════════════════════════════════════════════════
            "FALSO_FUNCIONARIO_BANCO": {
                "required": ["chave_aleatoria", "pix_acima_1000"],
                "optional": [
                    "idade_60_plus",
                    "idade_70_plus",
                    "burst_30m",
                    "intervalo_muito_curto",
                    "valor_absoluto_alto",
                    "valor_redondo",
                    "horario_comercial",
                    "is_segmento_premium",
                    "login_senha",
                    "renda_incompativel",
                    "renda_desconhecida_valor_alto",
                    "recebedor_nunca_visto",
                ],
                "min_score": 9,
                # v3.4: 7→9 | -39 TP, -287 FP | Prec 22.9%→59.8%
                "severity": "CRITICO",
                "description": (
                    "Padrão de golpe do falso funcionário: chave aleatória + "
                    "valor alto + múltiplos indicadores de vulnerabilidade"
                ),
            },
            # ═══════════════════════════════════════════════════════
            # PADRÃO 5: IDOSO_VULNERAVEL_70 [inalterado]
            # Leakage-free: TP=71, FP=118, Prec=37.6%
            # ═══════════════════════════════════════════════════════
            "IDOSO_VULNERAVEL_70": {
                "required": ["idade_70_plus", "pix_acima_1000"],
                "optional": [
                    "burst_30m",
                    "intervalo_muito_curto",
                    "valor_absoluto_alto",
                    "valor_redondo",
                    "chave_aleatoria",
                    "is_segmento_premium",
                    "perfil_vulneravel_se",
                    "login_senha",
                    "renda_incompativel",
                    "recebedor_nunca_visto",
                ],
                "min_score": 7,
                "severity": "CRITICO",
                "description": (
                    "Cliente 70+ com PIX de alto valor — "
                    "vulnerabilidade a engenharia social"
                ),
            },
            # ═══════════════════════════════════════════════════════
            # PADRÃO 6: IDOSO_VULNERAVEL_80 [inalterado]
            # Leakage-free: TP=11, FP=13, Prec=45.8%
            # ═══════════════════════════════════════════════════════
            "IDOSO_VULNERAVEL_80": {
                "required": ["idade_80_plus", "pix_acima_1000"],
                "optional": [
                    "burst_30m",
                    "intervalo_muito_curto",
                    "valor_absoluto_alto",
                    "chave_aleatoria",
                    "perfil_vulneravel_se",
                    "login_senha",
                    "recebedor_nunca_visto",
                ],
                "min_score": 6,
                "severity": "CRITICO",
                "description": (
                    "Cliente 80+ com PIX de alto valor — "
                    "altíssima vulnerabilidade a golpes"
                ),
            },
            # ═══════════════════════════════════════════════════════
            # PADRÃO 7: BURST_VALOR_ALTO [inalterado]
            # Leakage-free: TP=141, FP=38, Prec=78.8%
            # ═══════════════════════════════════════════════════════
            "BURST_VALOR_ALTO": {
                "required": ["burst_30m", "pix_acima_500"],
                "optional": [
                    "burst_intenso",
                    "multiplos_pix_rapidos",
                    "intervalo_muito_curto",
                    "pix_acima_1000",
                    "valor_absoluto_alto",
                    "valor_absoluto_muito_alto",
                    "primeira_tx_trimestre",
                    "renda_desconhecida_valor_alto",
                    "renda_metade_comprometida",
                    "idade_60_plus",
                ],
                "min_score": 3,
                "severity": "ALTO",
                "description": (
                    "Burst de transações em 30min com valor ≥ R$500 — "
                    "padrão de urgência típico de engenharia social"
                ),
            },
            # ═══════════════════════════════════════════════════════
            # PADRÃO 8: BURST_INTENSO_RAPIDO [inalterado]
            # Leakage-free: TP=48, FP=0, Prec=100%
            # ═══════════════════════════════════════════════════════
            "BURST_INTENSO_RAPIDO": {
                "required": [
                    "burst_intenso",
                    "burst_30m",
                    "multiplos_pix_rapidos",
                ],
                "optional": [
                    "pix_acima_1000",
                    "valor_absoluto_alto",
                    "intervalo_muito_curto",
                    "primeira_tx_trimestre",
                    "aproximando_esgotamento",
                ],
                "min_score": 6,
                "severity": "CRITICO",
                "description": (
                    "ALERTA MÁXIMO: Burst intenso (3+ tx em 30min) com "
                    "múltiplos PIX rápidos — 100%% correlação com fraude"
                ),
            },
            # ═══════════════════════════════════════════════════════
            # REMOVIDO: PRIMEIRA_TX_SUSPEITA
            #
            # v3.3: TP=127, FP=4395, Prec=2.8% no leakage-free
            # v3.4: REMOVIDO — precision inaceitável.
            #
            # primeira_tx_trimestre como required gate ativa em 78%
            # dos normais no leakage-free (rolling window causal).
            # Redesigns testados (S2B valor_redondo, S2C renda, S2D
            # valor_5k) não atingiram precision suficiente.
            #
            # O módulo Behavioral cobre esse cenário com:
            #   PRIMEIRA_TX_VALOR_ALTO (Prec 72.4%, TP=89)
            #   CONTA_DORMANTE_VALOR_ALTO (Prec 65.0%, TP=141)
            # ═══════════════════════════════════════════════════════
        }

    # =============================================================
    # PUBLIC API
    # =============================================================
    def detect_from_pipeline(
        self, features: Dict[str, Any]
    ) -> SEAnalysisResult:
        """Método principal — recebe features já processadas pelo pipeline."""
        f = self._adapt_features(features)

        # Avaliar todos os indicadores
        active_indicators: Dict[str, bool] = {}
        for name, check in self.INDICATORS.items():
            try:
                active_indicators[name] = check(f)
            except Exception as e:
                logger.debug(f"Indicador '{name}' falhou: {e}")
                active_indicators[name] = False

        phase2_missing = list(self.PHASE2_INDICATORS)

        # Avaliar cada padrão
        detected: List[PatternMatch] = []
        for pattern_name, config in self.PATTERNS.items():
            score = 0
            matched: List[str] = []

            # Verificar required (TODOS devem estar presentes)
            required_ok = True
            for ind in config["required"]:
                if active_indicators.get(ind, False):
                    score += 2
                    matched.append(ind)
                else:
                    required_ok = False
                    break

            if not required_ok:
                continue

            # Somar optional
            for ind in config["optional"]:
                if active_indicators.get(ind, False):
                    score += 1
                    matched.append(ind)

            # Verificar min_score
            if score >= config["min_score"]:
                detected.append(
                    PatternMatch(
                        pattern_name=pattern_name,
                        severity=config["severity"],
                        score=score,
                        matched_indicators=matched,
                        description=config["description"],
                    )
                )

        # Ordenar por severidade, depois por score
        detected.sort(
            key=lambda x: (_SEVERITY_ORDER.get(x.severity, 99), -x.score)
        )

        # Calcular score final com deduplicação
        se_score = self._calculate_se_score(detected, active_indicators)

        return SEAnalysisResult(
            se_score=se_score,
            patterns=detected,
            active_indicators=active_indicators,
            phase2_indicators_missing=phase2_missing,
        )

    def detect_patterns(
        self, features: Dict[str, Any]
    ) -> List[PatternMatch]:
        """API de compatibilidade."""
        result = self.detect_from_pipeline(features)
        return result.patterns

    def get_worst_pattern(
        self, features: Dict[str, Any]
    ) -> Optional[PatternMatch]:
        """Retorna o padrão mais grave detectado."""
        result = self.detect_from_pipeline(features)
        return result.worst_pattern

    def calculate_social_engineering_score(
        self, features: Dict[str, Any]
    ) -> Tuple[float, List[PatternMatch]]:
        """Calcula score SE e retorna padrões detectados."""
        result = self.detect_from_pipeline(features)
        return result.se_score, result.patterns

    # =============================================================
    # PRIVATE: Score calculation v3 — com deduplicação
    # =============================================================
    @staticmethod
    def _calculate_se_score(
        patterns: List[PatternMatch],
        active_indicators: Dict[str, bool],
    ) -> float:
        """
        Calcula score SE com deduplicação por cluster de overlap.

        Padrões no mesmo cluster (Jaccard > 0.15) não somam.
        Apenas o de maior severidade/score dentro do cluster conta.
        """
        if not patterns:
            return 0.0

        severity_scores: Dict[str, float] = {
            "CRITICO": 40.0,
            "ALTO": 25.0,
            "MEDIO": 15.0,
            "BAIXO": 10.0,
        }

        used: set[str] = set()
        score = 0.0

        for pattern in patterns:
            if pattern.pattern_name in used:
                continue

            cluster_found = False
            for cluster in _OVERLAP_CLUSTERS:
                if pattern.pattern_name in cluster:
                    used.update(cluster)
                    cluster_found = True
                    break

            if not cluster_found:
                used.add(pattern.pattern_name)

            score += severity_scores.get(pattern.severity, 10.0)

        # Atenuante: agendamento recorrente
        if active_indicators.get("agendamento_recorrente", False):
            score = max(0.0, score - 15.0)

        return min(100.0, score)


# =========================================================
# MODULE-LEVEL HELPERS
# =========================================================
def _safe_float(
    val: Any, default: Optional[float] = None
) -> Optional[float]:
    """Converte valor para float de forma segura."""
    if val is None:
        return default
    try:
        v = float(val)
        if v != v:  # NaN check
            return default
        return v
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Converte valor para int de forma segura."""
    if val is None:
        return default
    try:
        v = float(val)
        if v != v:  # NaN check
            return default
        return int(v)
    except (ValueError, TypeError):
        return default


def _is_valor_redondo(valor: float) -> bool:
    """Verifica se valor é múltiplo de 100 (típico de golpes)."""
    if not valor or valor <= 0:
        return False
    return valor >= 100 and (valor % 100 == 0)
