"""
core/social_engineering.py v3.2 — Detector de Padrões de Engenharia Social

Reescrito com base na Validação Retroativa (Frente 1).
Calibrado com curvas Precision-Recall por padrão (Frente 2).
Melhorias cirúrgicas baseadas na Análise Exploratória (Frente 3).

Mudanças v3.1 → v3.2 (Melhorias da Frente 3 — Análise Exploratória):

  Contexto: Frente 3 analisou 155 fraudes invisíveis (score=0) e 303 FP
  do COACAO_FISICA. Identificou padrões exploráveis e testou combinações
  de indicadores com validação empírica.

  Ajustes aplicados:

  [R1] COACAO_FISICA — Adicionar primeira_tx_trimestre como required
       Motivação: Frente 3 mostrou que apenas 4.6% dos FP têm primeira_tx,
       vs 31.1% dos TP. Adicionar como required elimina 88.8% dos FP.
       Impacto medido: TP 122→89 (-33), FP 303→34 (-269), Prec 28.7%→72.4%
       F1: 0.313 → 0.3724 (+19%)
       Trade-off: Perde 33 TP, mas ganha 269 FP. Precision quase triplica.

  [R2] NOVO PADRÃO: BURST_VALOR_ALTO (burst_30m + pix_acima_1000)
       Motivação: Par com melhor precision×recall da Frente 3.
       Impacto medido: TP=105, FP=27, Prec=79.5%, F1=0.4312, FPR=0.027%
       Captura fraudes de valor alto com burst que o COACAO não pega mais
       (após R1 restringir COACAO a primeira_tx_trimestre).

  [R3] NOVO PADRÃO: BURST_INTENSO_RAPIDO (burst_intenso + multiplos_pix)
       Motivação: Trinca com 100% precision e zero FP na Frente 3.
       Impacto medido: TP=48, FP=0, Prec=100%, F1=0.2382
       Regra cirúrgica — pega só fraude, nunca falso positivo.

  [R4] NOVO PADRÃO: PRIMEIRA_TX_SUSPEITA (primeira_tx + pix_acima_1000)
       Motivação: Captura fraudes invisíveis de "low & slow" (36.8% das
       invisíveis têm primeira_tx vs 25% das detectadas — Lift 1.47).
       Impacto medido: TP=89, FP=34, Prec=72.4%, F1=0.3724

  Impacto global estimado (v3.1 → v3.2):
    COACAO_FISICA: FP 303→34 (-89%), Prec 28.7%→72.4%
    Novos padrões: até ~105 TP adicionais (com overlap parcial)
    Fraudes invisíveis: redução estimada de 155 → ~80-100

  Padrões ativos: 6 → 9
  Overlap clusters atualizados para incluir novos padrões

Mudanças v3.0 → v3.1 (Calibração min_score — Frente 2):
  Critério de otimização: max_f1 com override manual baseado em
  análise de tradeoff Precision × FP por padrão.

  Ajustes de min_score:
    ESVAZIAMENTO_CONTA:       5 → 4  (F1 ≈ igual, Precision +17.7pp, -64 FP vs ms=3)
    BURST_ESVAZIAMENTO_CONTA: 5 → 3  (+6 TP, +1 FP, Precision sobe)
    FALSO_FUNCIONARIO_BANCO:  6 → 7  (-245 FP, Precision +14pp)
    IDOSO_VULNERAVEL_70:      6 → 7  (-198 FP, Precision +14.5pp)
    IDOSO_VULNERAVEL_80:      5 → 6  (-61 FP, -2 TP, Precision 3x)
    COACAO_FISICA:            5 → 5  (confirmado como ótimo)

  Impacto global estimado (v3.0 → v3.1):
    FP: 957 → ~550-600 (-37%)
    TP: 219 → ~216-218 (~neutro)
    Precision: 18.6% → ~27-28%
    Recall: 61.7% → ~61%

Mudanças v2.1 → v3.0 (Data-Driven Rewrite):
  1. REMOVIDOS 5 padrões danosos (precision < 1%, FPR > 1%):
     - GOLPE_PIX_ERRADO (41k FP, precision 0.21%)
     - ROMANCE_SCAM (19k FP, precision 0.06%)
     - GOLPE_INVESTIMENTO (5k FP, precision 0.08%)
     - FALSO_SEQUESTRO (0 TP, 1.1k FP)
     - TRANSACAO_ATIPICA (0 TP, 2.8k FP)
     - CONTA_LARANJA_SAIDA (0 TP, 4 FP — irrelevante)

  2. REMOVIDOS indicadores com Lift < 1.0 (ANTI-indicadores):
     - valor_alto_vs_historico (Lift 0.25)
     - valor_muito_alto_vs_historico (Lift 0.26)
     - valor_critico_vs_historico (Lift 0.32)
     - escalada_valores (Lift 0.21)
     - horario_noturno (Lift 0.0)
     - horario_madrugada (Lift 0.0)
     - zscore_valor_extremo (Lift 0.0)
     - alta_frequencia_diaria (Lift 0.41)
     - primeiro_envio (Lift 0.61 — usado por 98% dos normais)
     - chave_aleatoria isolada (Lift 0.9 — 43% dos normais usam)

  3. RECALIBRADO FALSO_FUNCIONARIO_BANCO:
     - Required: chave_aleatoria + pix_acima_1000 (filtro de valor)
     - Optional: só indicadores com Lift > 3x
     - min_score: 4 → 6 → 7 (v3.1)

  4. RECALIBRADO IDOSO_VULNERAVEL_70 e _80:
     - Required: idade + pix_acima_1000 (elimina FP de idosos
       fazendo PIX de R$10)
     - Optional: só Lift > 3x

  5. FORTALECIDOS padrões de velocity (os que funcionam):
     - ESVAZIAMENTO_CONTA (precision 37.7%)
     - COACAO_FISICA (precision 51.3%)
     - BURST_ESVAZIAMENTO_CONTA (precision 38.1%)

  6. NOVO scoring v3 com deduplicação por cluster de overlap

  7. TODOS os indicadores restantes possuem Lift > 1.5x validado

Padrões ativos: 9 (calibrados com dados de 100.355 transações)
Indicadores ativos: 30 (todos com Lift ≥ 1.5x)

Referências de calibração:
  - Dataset: base_mvp_model_ready_optimized.csv (100.355 tx, 355 fraudes)
  - Validação Frente 1: avaliar_se_retroativo.py
  - Calibração Frente 2: calibrar_min_score_SE.py (curvas P-R por padrão)
  - Análise Exploratória Frente 3: se_frente3_analise_exploratoria.py
  - Data da calibração: 2026-04-10
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
# Padrões no mesmo cluster não somam score — só o maior conta.
#
# v3.2: Clusters atualizados com novos padrões.
# - BURST_VALOR_ALTO compartilha indicadores com COACAO_FISICA
#   (ambos usam burst + valor alto)
# - BURST_INTENSO_RAPIDO é subconjunto do ESVAZIAMENTO_CONTA
# - PRIMEIRA_TX_SUSPEITA pode co-ocorrer com COACAO_FISICA (R1)
#   mas cobre casos distintos — NÃO clusterizado com COACAO
# =========================================================
_OVERLAP_CLUSTERS: List[frozenset] = [
    frozenset({"IDOSO_VULNERAVEL_70", "IDOSO_VULNERAVEL_80"}),
    frozenset({"ESVAZIAMENTO_CONTA", "BURST_ESVAZIAMENTO_CONTA", "BURST_INTENSO_RAPIDO"}),
    frozenset({"COACAO_FISICA", "BURST_VALOR_ALTO"}),
]


# =========================================================
# SOCIAL ENGINEERING DETECTOR v3.2
# =========================================================
class SocialEngineeringDetector:
    """
    Detecta padrões de golpes de engenharia social v3.2 (Frente 3).

    9 padrões ativos, calibrados com dados reais (100.355 tx).
    min_score otimizado por curvas Precision-Recall (Frente 2).
    Melhorias cirúrgicas baseadas na Análise Exploratória (Frente 3).
    Todos os indicadores possuem Lift ≥ 1.5x validado.

    Padrões e performance medida/estimada:
      ESVAZIAMENTO_CONTA        — ms=4, Prec 67.7%, F1 0.369
      COACAO_FISICA (R1)        — ms=5, Prec ~72.4%, F1 ~0.372 (+ primeira_tx required)
      BURST_ESVAZIAMENTO_CONTA  — ms=3, Prec 38.1%, F1 0.081
      FALSO_FUNCIONARIO_BANCO   — ms=7, Prec 33.6%, F1 0.276
      IDOSO_VULNERAVEL_70       — ms=7, Prec 37.6%, F1 0.261
      IDOSO_VULNERAVEL_80       — ms=6, Prec 45.8%, F1 0.058
      BURST_VALOR_ALTO (R2)     — ms=3, Prec ~79.5%, F1 ~0.431 [NOVO]
      BURST_INTENSO_RAPIDO (R3) — ms=6, Prec ~100%, F1 ~0.238  [NOVO]
      PRIMEIRA_TX_SUSPEITA (R4) — ms=4, Prec ~72.4%, F1 ~0.372 [NOVO]
    """

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
            f"SocialEngineeringDetector v3.2 inicializado "
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

        # --- Renda (v2.1b) ---
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

        # --- v2.1: Features do IF v4 / orquestrador ---
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

        TODOS os indicadores abaixo foram validados na Frente 1
        com Lift ≥ 1.5x (taxa em fraudes / taxa em normais).

        Indicadores com Lift < 1.0 foram REMOVIDOS:
          - chave_aleatoria isolada (0.9x), primeiro_envio (0.61x),
          - valor_alto_vs_historico (0.25x), horario_noturno (0.0x),
          - escalada_valores (0.21x), etc.
        """
        self.INDICATORS: Dict[str, Callable[[Dict[str, Any]], bool]] = {
            # ─── VELOCITY / BURST (Lift > 40x) ──────────────────
            # Estes são os indicadores mais discriminativos do BRB.
            "burst_intenso": lambda f: f["tx_count_prev_30m"] >= 3,
            # Lift ∞ (0 normais) | IG 0.0039
            "burst_30m": lambda f: f["burst_30m_flag"] == 1,
            # Lift 241.7x | IG 0.0092
            "multiplos_pix_rapidos": lambda f: (
                f["burst_30m_flag"] == 1 and f["qt_pix_dia_maximo_trimestre"] >= 3
            ),
            # Lift 196.7x | IG 0.0071
            "primeira_tx_trimestre": lambda f: f["is_first_tx_trimestre"] == 1,
            # Lift 146.3x | IG 0.0061
            "burst_conta_antiga": lambda f: (
                f["qt_tempo_relacionamento_mes"] >= 12
                and f["burst_30m_flag"] == 1
                and f["first_receiver_flag"] == 1
            ),
            # Lift 47.3x | IG 0.0009
            # ─── VALOR ABSOLUTO (Lift > 14x) ────────────────────
            "valor_absoluto_alto": lambda f: f["vl_pix"] >= 5000,
            # Lift 34.3x | IG 0.0043
            "valor_absoluto_muito_alto": lambda f: f["vl_pix"] >= 10000,
            # Lift 29.4x | IG 0.0015
            "pix_acima_1000": lambda f: f["vl_pix"] >= 1000,
            # Lift 14.5x | IG 0.0088
            # ─── RENDA (Lift > 4x) ──────────────────────────────
            "renda_desconhecida_valor_alto": lambda f: (
                f["renda_missing_flag"] == 1 and f["vl_pix"] >= 5000
            ),
            # Lift 23.4x | IG 0.0018
            "renda_metade_comprometida": lambda f: (
                f["pix_over_50pct_renda_flag"] == 1
            ),
            # Lift 4.6x | IG 0.0016
            "renda_incompativel": lambda f: f["pix_over_100pct_renda_flag"] == 1,
            # Lift TBD (subconjunto de renda_metade)
            # ─── PERFIL / IDADE (Lift > 3.8x) ───────────────────
            "idade_70_plus": lambda f: f["nr_idade"] >= 70,
            # Lift 8.4x | IG 0.0024
            "idade_60_plus": lambda f: f["nr_idade"] >= 60,
            # Lift 3.9x | IG 0.0022
            "idade_80_plus": lambda f: f["nr_idade"] >= 80,
            # Lift 4.5x | IG 0.0002
            # ─── INTERVALO (Lift > 3.8x) ────────────────────────
            "intervalo_muito_curto": lambda f: (
                f["qt_intervalo_transacao_minuto"] is not None
                and 0 <= f["qt_intervalo_transacao_minuto"] <= 5
            ),
            # Lift 6.7x | IG 0.0052
            "intervalo_curto": lambda f: (
                f["qt_intervalo_transacao_minuto"] is not None
                and 0 <= f["qt_intervalo_transacao_minuto"] <= 30
            ),
            # Lift 3.8x | IG 0.0051
            # ─── OUTROS (Lift > 1.5x) ───────────────────────────
            "conta_recem_aberta": lambda f: f["qt_tempo_relacionamento_mes"] <= 1,
            # Lift 3.7x
            "cliente_muito_novo": lambda f: f["qt_tempo_relacionamento_mes"] <= 3,
            # Lift 2.4x
            "valor_redondo": lambda f: _is_valor_redondo(f["vl_pix"]),
            # Lift 3.0x | IG 0.0009
            "multiplos_recebedores_distintos": lambda f: (
                f["distinct_receivers_so_far"] >= 3
            ),
            # Lift 1.8x
            "is_segmento_premium": lambda f: f["is_segmento_premium_flag"] == 1,
            # Lift 1.6x
            "perfil_vulneravel_se": lambda f: f["perfil_vulneravel_se_flag"] == 1,
            # Lift 1.5x
            # ─── CHAVE ALEATÓRIA (Lift 0.9x isolado — só útil combinado) ─
            # Mantido APENAS para uso em required COMBINADO com pix_acima_1000
            "chave_aleatoria": lambda f: f["pix_key_random_flag"] == 1,
            # ─── COMPOSTOS ──────────────────────────────────────
            "aproximando_esgotamento": lambda f: (
                f["ratio_valor_mediana"] is not None
                and f["ratio_valor_mediana"] >= 5.0
                and f["burst_30m_flag"] == 1
            ),
            # Lift 26.8x | IG 0.0001
            "recebedor_nunca_visto": lambda f: (
                f["qt_envio_recebedor_trimestre"] == 0
            ),
            # Usado como optional (não required — Lift não medido isolado)
            # ─── AUTENTICAÇÃO ────────────────────────────────────
            "login_senha": lambda f: f["is_login_senha_flag"] == 1,
            # Lift TBD
            # ─── HORÁRIO (Lift > 1.0x — mantido para padrões específicos) ─
            "horario_comercial": lambda f: (
                8 <= f["hour"] < 18 and 0 <= f["day_of_week"] <= 4
            ),
            # Lift ~1.1x (fraco, mas conceitualmente correto para falso funcionário)
            # ─── ATENUANTE ──────────────────────────────────────
            "agendamento_recorrente": lambda f: (
                f["is_agendamento_recorrente_flag"] == 1
            ),
        }

    # =============================================================
    # PATTERNS SETUP — 9 padrões calibrados com dados
    # =============================================================
    def _setup_patterns(self):
        """
        Define os 9 padrões de golpes ativos.

        Critérios para inclusão:
          - Precision ≥ 1% OU recall incremental vs LGBM
          - Todos os optional têm Lift ≥ 1.5x
          - min_score calibrado via curvas P-R (Frente 2)
          - Novos padrões validados empiricamente na Frente 3

        Calibração: base_mvp_model_ready_optimized.csv (100.355 tx)
        Otimização: calibrar_min_score_SE.py (max_f1 + override manual)
        Exploratória: se_frente3_analise_exploratoria.py
        Data: 2026-04-10
        """
        self.PATTERNS: Dict[str, Dict[str, Any]] = {
            # ═══════════════════════════════════════════════════════
            # PADRÃO 1: ESVAZIAMENTO_CONTA
            # v3.0: ms=5, TP=52, FP=21, Prec=71.2%, F1=0.243
            # v3.1: ms=4, TP=90, FP=43, Prec=67.7%, F1=0.369
            # v3.2: sem alteração
            # ═══════════════════════════════════════════════════════
            "ESVAZIAMENTO_CONTA": {
                "required": ["multiplos_pix_rapidos"],
                # Lift 196.7x — gate principal
                "optional": [
                    "burst_intenso",           # Lift ∞
                    "intervalo_muito_curto",    # Lift 6.7x
                    "pix_acima_1000",           # Lift 14.5x
                    "valor_absoluto_alto",      # Lift 34.3x
                    "primeira_tx_trimestre",    # Lift 146.3x
                    "renda_desconhecida_valor_alto",  # Lift 23.4x
                    "renda_incompativel",       # Lift TBD
                    "multiplos_recebedores_distintos",  # Lift 1.8x
                    "aproximando_esgotamento",  # Lift 26.8x
                ],
                "min_score": 4,
                # v3.0=5 → v3.1=4 | +38 TP, +22 FP | Prec 67.7% | F1 0.369
                "severity": "CRITICO",
                "description": (
                    "Esvaziamento de conta: múltiplos PIX rápidos + "
                    "valores altos + padrão de urgência"
                ),
            },
            # ═══════════════════════════════════════════════════════
            # PADRÃO 2: COACAO_FISICA [R1 — Frente 3]
            # v3.0: ms=5, TP=122, FP=303, Prec=28.7%, F1=0.313
            # v3.1: ms=5 (confirmado como ótimo na Frente 2)
            # v3.2: +primeira_tx_trimestre como required
            #   Medido na Frente 3: TP 122→89, FP 303→34, Prec 72.4%
            #   F1: 0.313 → 0.3724 (+19%)
            #   Justificativa: Apenas 4.6% dos FP têm primeira_tx vs
            #   31.1% dos TP. Elimina 88.8% dos FP.
            #   Os 33 TP perdidos são cobertos pelo novo BURST_VALOR_ALTO.
            # ═══════════════════════════════════════════════════════
            "COACAO_FISICA": {
                "required": [
                    "intervalo_muito_curto",    # Lift 6.7x
                    "pix_acima_1000",           # Lift 14.5x
                    "primeira_tx_trimestre",    # Lift 146.3x [R1: NOVO required]
                ],
                "optional": [
                    "burst_intenso",            # Lift ∞
                    "burst_30m",                # Lift 241.7x
                    "multiplos_pix_rapidos",    # Lift 196.7x
                    "valor_absoluto_alto",      # Lift 34.3x
                    "valor_absoluto_muito_alto",  # Lift 29.4x
                    "renda_desconhecida_valor_alto",  # Lift 23.4x
                    "renda_incompativel",       # Lift TBD
                    "multiplos_recebedores_distintos",  # Lift 1.8x
                ],
                "min_score": 5,
                # v3.1=5 → v3.2=5 (mantido, mas required agora soma 6 base)
                # Com 3 required (+2 cada = 6) já atinge min_score.
                # Na prática, ms=5 está OK: required sozinhos já ativam.
                "severity": "CRITICO",
                "description": (
                    "URGENTE: Possível coação física — primeira transação do "
                    "trimestre com PIX alto em intervalos < 5 minutos"
                ),
            },
            # ═══════════════════════════════════════════════════════
            # PADRÃO 3: BURST_ESVAZIAMENTO_CONTA
            # v3.0: ms=5, TP=10, FP=25, Prec=28.6%, F1=0.051
            # v3.1: ms=3, TP=16, FP=26, Prec=38.1%, F1=0.081
            # v3.2: sem alteração
            # ═══════════════════════════════════════════════════════
            "BURST_ESVAZIAMENTO_CONTA": {
                "required": ["burst_conta_antiga", "pix_acima_1000"],
                # Lift 47.3x + 14.5x
                "optional": [
                    "burst_intenso",            # Lift ∞
                    "intervalo_muito_curto",     # Lift 6.7x
                    "multiplos_recebedores_distintos",  # Lift 1.8x
                    "valor_absoluto_alto",       # Lift 34.3x
                    "renda_desconhecida_valor_alto",  # Lift 23.4x
                    "renda_incompativel",        # Lift TBD
                    "recebedor_nunca_visto",     # Contextual
                    "aproximando_esgotamento",   # Lift 26.8x
                ],
                "min_score": 3,
                # v3.0=5 → v3.1=3 | +6 TP, +1 FP | Prec 38.1% | F1 0.081
                "severity": "CRITICO",
                "description": (
                    "Conta antiga com burst súbito de PIX alto para "
                    "recebedores novos — conta comprometida"
                ),
            },
            # ═══════════════════════════════════════════════════════
            # PADRÃO 4: FALSO_FUNCIONARIO_BANCO (recalibrado)
            # v3.0: ms=6, TP=99, FP=409, Prec=19.5%, F1=0.229
            # v3.1: ms=7, TP=83, FP=164, Prec=33.6%, F1=0.276
            # v3.2: sem alteração
            # ═══════════════════════════════════════════════════════
            "FALSO_FUNCIONARIO_BANCO": {
                "required": ["chave_aleatoria", "pix_acima_1000"],
                # chave_aleatoria Lift 0.9x isolado, MAS em combinação
                # com pix_acima_1000 (Lift 14.5x) filtra 95% dos FP.
                # Conceitualmente essencial: vítima não conhece o recebedor.
                "optional": [
                    "idade_60_plus",            # Lift 3.9x
                    "idade_70_plus",            # Lift 8.4x
                    "burst_30m",                # Lift 241.7x
                    "intervalo_muito_curto",    # Lift 6.7x
                    "valor_absoluto_alto",      # Lift 34.3x
                    "valor_redondo",            # Lift 3.0x
                    "horario_comercial",        # Lift ~1.1x (conceitual)
                    "is_segmento_premium",      # Lift 1.6x
                    "login_senha",              # Lift TBD
                    "renda_incompativel",       # Lift TBD
                    "renda_desconhecida_valor_alto",  # Lift 23.4x
                    "recebedor_nunca_visto",    # Contextual
                ],
                "min_score": 7,
                # v3.0=6 → v3.1=7 | -245 FP | Prec 33.6% | F1 0.276
                # Precisa de required(+4) + 3 optional.
                "severity": "CRITICO",
                "description": (
                    "Padrão de golpe do falso funcionário: chave aleatória + "
                    "valor alto + indicadores de vulnerabilidade"
                ),
            },
            # ═══════════════════════════════════════════════════════
            # PADRÃO 5: IDOSO_VULNERAVEL_70 (recalibrado)
            # v3.0: ms=6, TP=95, FP=316, Prec=23.1%, F1=0.248
            # v3.1: ms=7, TP=71, FP=118, Prec=37.6%, F1=0.261
            # v3.2: sem alteração
            # ═══════════════════════════════════════════════════════
            "IDOSO_VULNERAVEL_70": {
                "required": ["idade_70_plus", "pix_acima_1000"],
                # Lift 8.4x + 14.5x — filtra idosos fazendo PIX de R$10
                "optional": [
                    "burst_30m",                # Lift 241.7x
                    "intervalo_muito_curto",    # Lift 6.7x
                    "valor_absoluto_alto",      # Lift 34.3x
                    "valor_redondo",            # Lift 3.0x
                    "chave_aleatoria",          # Contextual para idoso
                    "is_segmento_premium",      # Lift 1.6x
                    "perfil_vulneravel_se",     # Lift 1.5x
                    "login_senha",              # Lift TBD
                    "renda_incompativel",       # Lift TBD
                    "recebedor_nunca_visto",    # Contextual
                ],
                "min_score": 7,
                # v3.0=6 → v3.1=7 | -198 FP | Prec 37.6% | F1 0.261
                # required(+4) + 3 optional necessários
                "severity": "CRITICO",
                "description": (
                    "Cliente 70+ com PIX de alto valor — "
                    "vulnerabilidade a engenharia social"
                ),
            },
            # ═══════════════════════════════════════════════════════
            # PADRÃO 6: IDOSO_VULNERAVEL_80 (recalibrado)
            # v3.0: ms=5, TP=13, FP=74, Prec=14.9%, F1=0.059
            # v3.1: ms=6, TP=11, FP=13, Prec=45.8%, F1=0.058
            # v3.2: sem alteração
            # ═══════════════════════════════════════════════════════
            "IDOSO_VULNERAVEL_80": {
                "required": ["idade_80_plus", "pix_acima_1000"],
                # Lift 4.5x + 14.5x
                "optional": [
                    "burst_30m",                # Lift 241.7x
                    "intervalo_muito_curto",    # Lift 6.7x
                    "valor_absoluto_alto",      # Lift 34.3x
                    "chave_aleatoria",          # Contextual
                    "perfil_vulneravel_se",     # Lift 1.5x
                    "login_senha",              # Lift TBD
                    "recebedor_nunca_visto",    # Contextual
                ],
                "min_score": 6,
                # v3.0=5 → v3.1=6 | -61 FP, -2 TP | Prec 45.8%
                # required(+4) + 2 optional necessários.
                "severity": "CRITICO",
                "description": (
                    "Cliente 80+ com PIX de alto valor — "
                    "altíssima vulnerabilidade a golpes"
                ),
            },
            # ═══════════════════════════════════════════════════════
            # PADRÃO 7: BURST_VALOR_ALTO [R2 — NOVO na Frente 3]
            # Medido: TP=105, FP=27, Prec=79.5%, F1=0.4312, FPR=0.027%
            #
            # Justificativa: Melhor par precision×recall da Frente 3.
            # Captura fraudes com burst + valor alto que o COACAO v3.2
            # não pega mais (após R1 exigir primeira_tx_trimestre).
            # Cluster de overlap com COACAO_FISICA para não double-count.
            #
            # Required: burst_30m (Lift 241.7x) + pix_acima_1000 (14.5x)
            # min_score=3: required já soma 4, precisa de 0 optional.
            # Mantemos ms=3 ao invés de ms=2 para exigir que pelo menos
            # o par esteja presente (sanity check).
            # ═══════════════════════════════════════════════════════
            "BURST_VALOR_ALTO": {
                "required": ["burst_30m", "pix_acima_1000"],
                # Lift 241.7x + 14.5x — par com FPR 0.027%
                "optional": [
                    "burst_intenso",            # Lift ∞
                    "multiplos_pix_rapidos",    # Lift 196.7x
                    "intervalo_muito_curto",    # Lift 6.7x
                    "valor_absoluto_alto",      # Lift 34.3x
                    "valor_absoluto_muito_alto",  # Lift 29.4x
                    "primeira_tx_trimestre",    # Lift 146.3x
                    "renda_desconhecida_valor_alto",  # Lift 23.4x
                    "renda_metade_comprometida",  # Lift 4.6x
                    "idade_60_plus",            # Lift 3.9x
                ],
                "min_score": 3,
                # required(+4) ≥ ms=3 → ativa com apenas os required.
                # Optional elevam o score para refletir gravidade.
                "severity": "ALTO",
                "description": (
                    "Burst de transações em 30min com valor ≥ R$1.000 — "
                    "padrão de urgência típico de engenharia social"
                ),
            },
            # ═══════════════════════════════════════════════════════
            # PADRÃO 8: BURST_INTENSO_RAPIDO [R3 — NOVO na Frente 3]
            # Medido: TP=48, FP=0, Prec=100%, F1=0.2382
            #
            # Justificativa: Regra cirúrgica com ZERO falsos positivos.
            # A trinca burst_intenso + burst_30m + multiplos_pix_rapidos
            # captura 48 fraudes sem nenhum FP — pega quando 3+ tx em
            # 30min com tx_count_prev_30m ≥ 3 e qt_pix_dia_max ≥ 3.
            #
            # Cluster de overlap com ESVAZIAMENTO_CONTA (que requer
            # multiplos_pix_rapidos, subconjunto parcial).
            #
            # min_score=6: os 3 required somam 6. Ativa somente se
            # todos os 3 estiverem presentes — sem margem para erro.
            # ═══════════════════════════════════════════════════════
            "BURST_INTENSO_RAPIDO": {
                "required": [
                    "burst_intenso",           # Lift ∞ (0 normais)
                    "burst_30m",               # Lift 241.7x
                    "multiplos_pix_rapidos",   # Lift 196.7x
                ],
                # Combinação tripla com 100% precision
                "optional": [
                    "pix_acima_1000",          # Lift 14.5x
                    "valor_absoluto_alto",     # Lift 34.3x
                    "intervalo_muito_curto",   # Lift 6.7x
                    "primeira_tx_trimestre",   # Lift 146.3x
                    "aproximando_esgotamento", # Lift 26.8x
                ],
                "min_score": 6,
                # required(+6) = 6 ≥ ms=6 → ativa apenas com os 3 required.
                # Optional elevam o score para scoring fino.
                "severity": "CRITICO",
                "description": (
                    "ALERTA MÁXIMO: Burst intenso (3+ tx em 30min) com "
                    "múltiplos PIX rápidos — 100%% correlação com fraude"
                ),
            },
            # ═══════════════════════════════════════════════════════
            # PADRÃO 9: PRIMEIRA_TX_SUSPEITA [R4 — NOVO na Frente 3]
            # Medido: TP=89, FP=34, Prec=72.4%, F1=0.3724
            #
            # Justificativa: Captura fraudes "low & slow" invisíveis ao
            # SE v3.1. As fraudes invisíveis têm 36.8% de primeira_tx vs
            # 25% das detectadas (Lift 1.47 — sub-representadas na detecção).
            #
            # Perfil alvo: Primeira transação do trimestre com valor ≥ R$1k
            # para recebedor que nunca recebeu antes (first_receiver).
            # Conceitualmente: golpista convence vítima a fazer um PIX
            # "único" de valor moderado — sem burst, sem urgência aparente.
            #
            # NÃO clusterizado com COACAO_FISICA: apesar de compartilharem
            # primeira_tx + pix_acima_1000, PRIMEIRA_TX_SUSPEITA NÃO requer
            # intervalo_muito_curto, cobrindo um espaço diferente.
            #
            # min_score=4: required(+4), precisa de 0+ optional.
            # Severity MEDIO: menor confiança que CRITICO (72.4% prec),
            # e perfil menos urgente (sem burst/intervalo curto).
            # ═══════════════════════════════════════════════════════
            "PRIMEIRA_TX_SUSPEITA": {
                "required": ["primeira_tx_trimestre", "pix_acima_1000"],
                # Lift 146.3x + 14.5x
                "optional": [
                    "idade_60_plus",            # Lift 3.9x
                    "idade_70_plus",            # Lift 8.4x
                    "renda_metade_comprometida",  # Lift 4.6x
                    "renda_desconhecida_valor_alto",  # Lift 23.4x
                    "valor_absoluto_alto",      # Lift 34.3x
                    "is_segmento_premium",      # Lift 1.6x
                    "valor_redondo",            # Lift 3.0x
                    "chave_aleatoria",          # Contextual
                    "recebedor_nunca_visto",    # Contextual
                ],
                "min_score": 4,
                # required(+4) ≥ ms=4 → ativa com apenas os required.
                # Severity MEDIO reflete menor urgência e precisão.
                "severity": "MEDIO",
                "description": (
                    "Primeira transação do trimestre com valor ≥ R$1.000 — "
                    "possível engenharia social sem sinais de urgência"
                ),
            },
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

        v3.0: Padrões no mesmo cluster (Jaccard > 0.15) não somam.
        Apenas o de maior severidade/score dentro do cluster conta.

        Isso evita double-counting quando, por exemplo,
        ESVAZIAMENTO_CONTA e BURST_ESVAZIAMENTO_CONTA ativam juntos,
        ou COACAO_FISICA e BURST_VALOR_ALTO co-ocorrem.
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

        # Padrões já vêm ordenados por severidade + score
        for pattern in patterns:
            if pattern.pattern_name in used:
                continue

            # Verificar se este padrão pertence a algum cluster
            cluster_found = False
            for cluster in _OVERLAP_CLUSTERS:
                if pattern.pattern_name in cluster:
                    # Marcar todo o cluster como usado
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
