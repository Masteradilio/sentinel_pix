"""
core/social_engineering.py v2.0 — Detector de Padrões de Engenharia Social

Reescrita completa v2.0:
  1. CRIADO _adapt_features() — mapeia features do pipeline para nomes esperados
  2. REMOVIDOS indicadores sem dados reais: mulher_idosa (ds_sexo nem sempre
     disponível no RT), viuvo_viuva, segmento_alto_patrimonio, recebedor_nunca_visto
     → Marcados como "fase 2" (serão ativados quando cobertura > 80%)
  3. SUBSTITUÍDO vl_razao_pix_limite → ratio_valor_mediana em todos os indicadores
     (vl_razao_pix_limite não existe no pipeline; ratio_valor_mediana sim)
  4. SUBSTITUÍDO tp_primeiro_envio_recebedor_trimestre → first_receiver_flag
     (já calculado pelo pipeline sequencial, mais confiável)
  5. IMPLEMENTADO escalada_valores usando cache do pipeline
     (compara vl_pix com mediana + desvio padrão, sem depender de vl_transacao_anterior)
  6. ATUALIZADOS padrões removendo indicadores indisponíveis dos optional
  7. ADICIONADO detect_from_pipeline() — recebe dados já processados
  8. MANTIDOS os 11 padrões mas ajustados min_score onde indicadores foram removidos

Novos indicadores v2.0 (com dados v2.1b do Big Data):
  - is_viuvo_flag (do pipeline, quando disponível)
  - is_segmento_premium_flag (do pipeline)
  - is_sexo_feminino_flag (do pipeline)
  - perfil_vulneravel_se_flag (viúvo + idoso + sem dependentes)
  - pix_over_100pct_renda_flag (PIX > renda mensal)
  - pix_over_50pct_renda_flag (PIX > 50% da renda)
  - renda_missing_flag (renda desconhecida)
  - ratio_pix_renda (PIX / renda)
  - burst_30m_flag (do pipeline)
  - tx_count_prev_30m (do pipeline)

Integração com pipeline:
  - detect_from_pipeline(features_dict) → método principal
  - detect_patterns(features_dict) → compatibilidade com orquestrador existente
  - Todos os indicadores usam APENAS campos que existem no preprocessing.py v3.1
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
    severity: str           # "CRITICO", "ALTO", "MEDIO", "BAIXO"
    score: int              # Pontuação interna do padrão
    matched_indicators: List[str]
    description: str


@dataclass
class SEAnalysisResult:
    """Resultado completo da análise de engenharia social."""
    se_score: float                                     # 0-100
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
        """Serializa para dict (usado pelo orquestrador/API)."""
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
            "worst_pattern": self.worst_pattern.pattern_name if self.worst_pattern else None,
            "phase2_indicators_missing": self.phase2_indicators_missing,
        }

    def to_features(self) -> Dict[str, float]:
        """Exporta features numéricas para o score final do orquestrador."""
        return {
            "se_score": round(self.se_score, 2),
            "se_pattern_count": float(len(self.patterns)),
            "se_has_critico": float(any(p.severity == "CRITICO" for p in self.patterns)),
            "se_max_pattern_score": float(max((p.score for p in self.patterns), default=0)),
        }


# =========================================================
# SEVERITY ORDER (usado para ordenação)
# =========================================================
_SEVERITY_ORDER = {"CRITICO": 0, "ALTO": 1, "MEDIO": 2, "BAIXO": 3}


# =========================================================
# SOCIAL ENGINEERING DETECTOR
# =========================================================
class SocialEngineeringDetector:
    """
    Detecta padrões típicos de golpes de engenharia social.

    11 padrões detectados:
    ─────────────────────────────────────────────────────────
    #   Padrão                      Severidade  Min Score
    1   FALSO_FUNCIONARIO_BANCO     CRITICO     4
    2   FALSO_SEQUESTRO             CRITICO     5
    3   ESVAZIAMENTO_CONTA          CRITICO     5
    4   GOLPE_PIX_ERRADO            ALTO        4
    5   ROMANCE_SCAM                ALTO        4
    6   IDOSO_VULNERAVEL_70         CRITICO     4
    7   IDOSO_VULNERAVEL_80         CRITICO     3
    8   CONTA_LARANJA_SAIDA         CRITICO     4
    9   GOLPE_INVESTIMENTO          ALTO        4
    10  COACAO_FISICA               CRITICO     5
    11  TRANSACAO_ATIPICA           MEDIO       4
    ─────────────────────────────────────────────────────────

    Todos os indicadores usam APENAS campos disponíveis no
    preprocessing.py v3.1 (sem dependências externas).
    """

    # Indicadores de "fase 2" — serão ativados quando cobertura > 80%
    PHASE2_INDICATORS = [
        "mulher_idosa_raw",             # depende de ds_sexo com alta cobertura
        "viuvo_viuva_raw",              # depende de ds_estado_civil com alta cobertura
        "segmento_alto_patrimonio_raw", # depende de ds_segmento com alta cobertura
        "recebedor_nunca_visto",        # qt_envio_recebedor_trimestre == 0 (validar cobertura)
    ]

    def __init__(self):
        """Inicializa o detector com indicadores e padrões configurados."""
        self._value_cache: Dict[str, List[float]] = {}  # cpf -> últimos valores
        self._max_cache_size = 10
        self._setup_indicators()
        self._setup_patterns()
        logger.info("SocialEngineeringDetector v2.0 inicializado")

    # =============================================================
    # FEATURE ADAPTER
    # =============================================================
    @staticmethod
    def _adapt_features(features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mapeia features do pipeline para nomes esperados pelos indicadores.

        Garante compatibilidade entre:
        - Nomes do preprocessing.py v3.1 (snake_case)
        - Nomes legados do orquestrador
        - Features derivadas que podem vir com nomes diferentes

        Returns:
            Dict normalizado com todos os campos necessários.
        """
        f = {}

        # --- IDs e metadata ---
        f["customer_id"] = features.get("customer_id") or features.get("cd_cpf_pagador")
        f["transaction_id"] = features.get("transaction_id") or features.get("cd_pix")

        # --- Datetime ---
        dt_raw = features.get("event_datetime") or features.get("dt_pix")
        if isinstance(dt_raw, str):
            try:
                f["dt_pix"] = datetime.fromisoformat(dt_raw.replace("Z", "+00:00").replace(" ", "T"))
            except (ValueError, TypeError):
                f["dt_pix"] = None
        elif isinstance(dt_raw, datetime):
            f["dt_pix"] = dt_raw
        else:
            f["dt_pix"] = None

        # --- Temporais do pipeline ---
        f["hour"] = _safe_int(features.get("hour"), -1)
        f["day_of_week"] = _safe_int(features.get("day_of_week"), -1)
        f["is_business_hours"] = _safe_int(features.get("is_business_hours"), 0)

        # --- Valor ---
        f["vl_pix"] = _safe_float(features.get("vl_pix"), 0.0)
        f["vl_mediana_pix_trimestre"] = _safe_float(features.get("vl_mediana_pix_trimestre"), 0.0)
        f["vl_desvio_padrao_pix_trimestre"] = _safe_float(features.get("vl_desvio_padrao_pix_trimestre"), 0.0)
        f["ratio_valor_mediana"] = _safe_float(features.get("ratio_valor_mediana"))
        f["zscore_valor_aprox"] = _safe_float(features.get("zscore_valor_aprox"))
        f["log_vl_pix"] = _safe_float(features.get("log_vl_pix"), 0.0)

        # --- Renda (v2.1b) ---
        f["vl_renda_cliente"] = _safe_float(features.get("vl_renda_cliente"), 0.0)
        f["ratio_pix_renda"] = _safe_float(features.get("ratio_pix_renda"))
        f["pix_over_50pct_renda_flag"] = _safe_int(features.get("pix_over_50pct_renda_flag"), 0)
        f["pix_over_100pct_renda_flag"] = _safe_int(features.get("pix_over_100pct_renda_flag"), 0)
        f["renda_missing_flag"] = _safe_int(features.get("renda_missing_flag"), 0)

        # --- Perfil do cliente ---
        f["nr_idade"] = _safe_int(features.get("nr_idade"), 0)
        f["qt_tempo_relacionamento_mes"] = _safe_int(features.get("qt_tempo_relacionamento_mes"), 999)
        f["is_sexo_feminino_flag"] = _safe_int(features.get("is_sexo_feminino_flag"), 0)
        f["is_viuvo_flag"] = _safe_int(features.get("is_viuvo_flag"), 0)
        f["is_segmento_premium_flag"] = _safe_int(features.get("is_segmento_premium_flag"), 0)
        f["perfil_vulneravel_se_flag"] = _safe_int(features.get("perfil_vulneravel_se_flag"), 0)
        f["qt_dependentes"] = _safe_int(features.get("qt_dependentes"), 0)

        # --- Recebedor ---
        f["first_receiver_flag"] = _safe_int(features.get("first_receiver_flag"), 0)
        f["receiver_document_same_as_customer_flag"] = _safe_int(
            features.get("receiver_document_same_as_customer_flag"), 0
        )
        f["tp_primeiro_envio_recebedor_trimestre"] = _safe_int(
            features.get("tp_primeiro_envio_recebedor_trimestre"), 0
        )
        f["qt_envio_recebedor_trimestre"] = _safe_int(
            features.get("qt_envio_recebedor_trimestre"), 0
        )
        f["cd_cpf_cnpj_recebedor"] = str(features.get("cd_cpf_cnpj_recebedor") or "")

        # --- Tipo de chave (flags do pipeline) ---
        f["pix_key_random_flag"] = _safe_int(features.get("pix_key_random_flag"), 0)
        f["pix_key_email_flag"] = _safe_int(features.get("pix_key_email_flag"), 0)
        f["pix_key_document_flag"] = _safe_int(features.get("pix_key_document_flag"), 0)
        f["pix_key_other_flag"] = _safe_int(features.get("pix_key_other_flag"), 0)
        f["pix_key_missing_flag_derived"] = _safe_int(features.get("pix_key_missing_flag_derived"), 0)
        # Fallback para texto bruto (quando disponível)
        f["ds_tipo_chave"] = str(features.get("ds_tipo_chave") or "")

        # --- Velocidade / Frequência (do pipeline) ---
        f["qt_intervalo_transacao_minuto"] = _safe_float(features.get("qt_intervalo_transacao_minuto"))
        f["qt_pix_dia_maximo_trimestre"] = _safe_int(features.get("qt_pix_dia_maximo_trimestre"), 0)
        f["qt_total_pix_trimestre"] = _safe_int(features.get("qt_total_pix_trimestre"), 0)
        f["burst_30m_flag"] = _safe_int(features.get("burst_30m_flag"), 0)
        f["tx_count_prev_30m"] = _safe_int(features.get("tx_count_prev_30m"), 0)
        f["minutes_since_prev_tx"] = _safe_float(features.get("minutes_since_prev_tx"))
        f["is_first_tx_trimestre"] = _safe_int(features.get("is_first_tx_trimestre"), 0)

        # --- Interação / Auth ---
        f["tempo_interacao_ms_final"] = _safe_float(features.get("tempo_interacao_ms_final"))
        f["ratio_tempo_interacao_cliente"] = _safe_float(features.get("ratio_tempo_interacao_cliente"))
        f["metodo_auth_encoded"] = _safe_int(features.get("metodo_auth_encoded"), 3)
        f["is_login_senha_flag"] = _safe_int(features.get("is_login_senha_flag"), 0)
        f["is_login_biometria_flag"] = _safe_int(features.get("is_login_biometria_flag"), 0)
        f["is_agendamento_recorrente_flag"] = _safe_int(features.get("is_agendamento_recorrente_flag"), 0)

        # --- Device ---
        f["device_missing_flag"] = _safe_int(features.get("device_missing_flag"), 0)
        f["qt_aparelhos_distintos_trimestre"] = _safe_int(
            features.get("qt_aparelhos_distintos_trimestre"), 0
        )

        # --- Regras do pipeline ---
        f["rule_age_score"] = _safe_int(features.get("rule_age_score"), 0)
        f["rule_velocity_score"] = _safe_int(features.get("rule_velocity_score"), 0)
        f["rule_score_raw"] = _safe_float(features.get("rule_score_raw"), 0.0)

        # --- Topaz (informativo, não usado no score SE) ---
        f["topaz_score_filled"] = _safe_float(features.get("topaz_score_filled"), 0.0)
        f["topaz_rejeitada_flag"] = _safe_int(features.get("topaz_rejeitada_flag"), 0)

        return f

    # =============================================================
    # INDICATORS SETUP
    # =============================================================
    def _setup_indicators(self):
        """
        Define os indicadores de risco.

        REGRA: Cada indicador usa APENAS campos do _adapt_features().
        Nenhum indicador faz I/O externo.
        """
        self.INDICATORS: Dict[str, Callable[[Dict[str, Any]], bool]] = {

            # ─── PERFIL DO CLIENTE ───────────────────────────────
            "idade_60_plus": lambda f: f["nr_idade"] >= 60,
            "idade_70_plus": lambda f: f["nr_idade"] >= 70,
            "idade_80_plus": lambda f: f["nr_idade"] >= 80,
            "cliente_novo": lambda f: f["qt_tempo_relacionamento_mes"] <= 6,
            "cliente_muito_novo": lambda f: f["qt_tempo_relacionamento_mes"] <= 3,
            "conta_recem_aberta": lambda f: f["qt_tempo_relacionamento_mes"] <= 1,

            # Indicadores do pipeline v2.1b (flags prontas)
            "perfil_vulneravel_se": lambda f: f["perfil_vulneravel_se_flag"] == 1,
            "is_viuvo": lambda f: f["is_viuvo_flag"] == 1,
            "is_segmento_premium": lambda f: f["is_segmento_premium_flag"] == 1,
            "is_sexo_feminino": lambda f: f["is_sexo_feminino_flag"] == 1,

            # ─── HORÁRIO ─────────────────────────────────────────
            # Usa hour/day_of_week do pipeline (já calculados)
            "horario_noturno": lambda f: f["hour"] >= 22 or (0 <= f["hour"] < 6),
            "horario_madrugada": lambda f: 0 <= f["hour"] < 5,
            "horario_comercial": lambda f: 8 <= f["hour"] < 18 and 0 <= f["day_of_week"] <= 4,
            "horario_almoco": lambda f: 11 <= f["hour"] < 14,
            "fim_de_semana": lambda f: f["day_of_week"] >= 5,

            # ─── RECEBEDOR ───────────────────────────────────────
            # Adaptação #4: first_receiver_flag do pipeline (mais confiável)
            "primeiro_envio": lambda f: f["first_receiver_flag"] == 1,
            "recebedor_pj": lambda f: len(f["cd_cpf_cnpj_recebedor"]) >= 14,
            "recebedor_mesmo_cpf": lambda f: f["receiver_document_same_as_customer_flag"] == 1,

            # ─── TIPO DE CHAVE ───────────────────────────────────
            # Usa flags do pipeline (já calculadas no preprocessing)
            "chave_aleatoria": lambda f: f["pix_key_random_flag"] == 1,
            "chave_email": lambda f: f["pix_key_email_flag"] == 1,
            "chave_documento_telefone": lambda f: f["pix_key_document_flag"] == 1,

            # ─── VALOR ───────────────────────────────────────────
            # Adaptação #3: ratio_valor_mediana em vez de vl_razao_pix_limite
            "valor_alto_vs_historico": lambda f: (
                f["ratio_valor_mediana"] is not None and f["ratio_valor_mediana"] >= 3.0
            ),
            "valor_muito_alto_vs_historico": lambda f: (
                f["ratio_valor_mediana"] is not None and f["ratio_valor_mediana"] >= 5.0
            ),
            "valor_critico_vs_historico": lambda f: (
                f["ratio_valor_mediana"] is not None and f["ratio_valor_mediana"] >= 8.0
            ),
            "valor_absoluto_alto": lambda f: f["vl_pix"] >= 5000,
            "valor_absoluto_muito_alto": lambda f: f["vl_pix"] >= 10000,
            "valor_redondo": lambda f: _is_valor_redondo(f["vl_pix"]),
            "zscore_valor_extremo": lambda f: (
                f["zscore_valor_aprox"] is not None and f["zscore_valor_aprox"] >= 3.0
            ),
            "pix_acima_1000": lambda f: f["vl_pix"] >= 1000,

            # Indicadores de renda (v2.1b)
            "renda_incompativel": lambda f: f["pix_over_100pct_renda_flag"] == 1,
            "renda_metade_comprometida": lambda f: f["pix_over_50pct_renda_flag"] == 1,
            "renda_desconhecida_valor_alto": lambda f: (
                f["renda_missing_flag"] == 1 and f["vl_pix"] >= 5000
            ),

            # ─── VELOCIDADE / FREQUÊNCIA ─────────────────────────
            "intervalo_curto": lambda f: (
                f["qt_intervalo_transacao_minuto"] is not None
                and 0 <= f["qt_intervalo_transacao_minuto"] <= 30
            ),
            "intervalo_muito_curto": lambda f: (
                f["qt_intervalo_transacao_minuto"] is not None
                and 0 <= f["qt_intervalo_transacao_minuto"] <= 5
            ),
            "burst_30m": lambda f: f["burst_30m_flag"] == 1,
            "burst_intenso": lambda f: f["tx_count_prev_30m"] >= 3,
            "alta_frequencia_diaria": lambda f: f["qt_pix_dia_maximo_trimestre"] >= 5,
            "primeira_tx_trimestre": lambda f: f["is_first_tx_trimestre"] == 1,

            # ─── COMPOSTOS ───────────────────────────────────────
            "multiplos_pix_rapidos": lambda f: (
                f["burst_30m_flag"] == 1 and f["qt_pix_dia_maximo_trimestre"] >= 3
            ),
            "escalada_valores": lambda f: self._detectar_escalada_valores(f),
            "aproximando_esgotamento": lambda f: (
                f["ratio_valor_mediana"] is not None
                and f["ratio_valor_mediana"] >= 5.0
                and f["burst_30m_flag"] == 1
            ),

            # ─── AUTENTICAÇÃO ────────────────────────────────────
            "login_senha": lambda f: f["is_login_senha_flag"] == 1,
            "login_biometria": lambda f: f["is_login_biometria_flag"] == 1,

            # ─── ATENUANTES ──────────────────────────────────────
            "agendamento_recorrente": lambda f: f["is_agendamento_recorrente_flag"] == 1,
        }

    # =============================================================
    # PATTERNS SETUP
    # =============================================================
    def _setup_patterns(self):
        """
        Define os 11 padrões de golpes.

        Adaptação #8: min_score ajustado onde indicadores de fase 2 foram removidos.
        """
        self.PATTERNS = {
            # ─── 1. FALSO FUNCIONÁRIO DO BANCO ───────────────────
            "FALSO_FUNCIONARIO_BANCO": {
                "required": ["chave_aleatoria"],
                "optional": [
                    "idade_60_plus", "horario_comercial",
                    "valor_alto_vs_historico", "valor_redondo", "primeiro_envio",
                    "is_segmento_premium", "login_senha",
                    "renda_incompativel",
                ],
                "min_score": 4,   # Ajustado: era 5, removido mulher_idosa dos optional
                "severity": "CRITICO",
                "description": (
                    "Padrão de golpe do falso funcionário do banco: "
                    "chave aleatória + idoso + horário comercial + valor atípico"
                ),
            },

            # ─── 2. FALSO SEQUESTRO ──────────────────────────────
            "FALSO_SEQUESTRO": {
                "required": ["horario_noturno", "valor_alto_vs_historico"],
                "optional": [
                    "horario_madrugada", "multiplos_pix_rapidos",
                    "intervalo_muito_curto", "chave_aleatoria",
                    "primeiro_envio", "aproximando_esgotamento",
                    "burst_intenso",
                ],
                "min_score": 5,  # Ajustado: era 6, chave_celular removida
                "severity": "CRITICO",
                "description": (
                    "Padrão de golpe do falso sequestro: "
                    "madrugada + valores altos + múltiplos PIX rápidos"
                ),
            },

            # ─── 3. ESVAZIAMENTO DE CONTA ────────────────────────
            "ESVAZIAMENTO_CONTA": {
                "required": ["multiplos_pix_rapidos"],
                "optional": [
                    "intervalo_muito_curto", "valor_critico_vs_historico",
                    "primeiro_envio", "chave_aleatoria", "horario_noturno",
                    "escalada_valores", "burst_intenso",
                    "renda_incompativel",
                ],
                "min_score": 5,   # Ajustado: era 6, removido aproximando_limite dos required
                "severity": "CRITICO",
                "description": (
                    "Esvaziamento de conta: múltiplos PIX rápidos + "
                    "valores críticos + destinos desconhecidos"
                ),
            },

            # ─── 4. GOLPE DO PIX ERRADO ──────────────────────────
            "GOLPE_PIX_ERRADO": {
                "required": ["primeiro_envio", "chave_aleatoria"],
                "optional": [
                    "valor_redondo", "intervalo_curto", "horario_comercial",
                    "valor_alto_vs_historico",
                ],
                "min_score": 4,
                "severity": "ALTO",
                "description": (
                    "Possível golpe do PIX errado: transferência para "
                    "chave aleatória de destinatário desconhecido"
                ),
            },

            # ─── 5. ROMANCE SCAM ─────────────────────────────────
            "ROMANCE_SCAM": {
                "required": ["primeiro_envio", "valor_alto_vs_historico"],
                "optional": [
                    "idade_60_plus", "is_viuvo",
                    "chave_email", "valor_muito_alto_vs_historico",
                    "fim_de_semana", "escalada_valores",
                    "perfil_vulneravel_se", "renda_metade_comprometida",
                ],
                "min_score": 4,  # Ajustado: era 5, removido mulher_idosa e chave_celular
                "severity": "ALTO",
                "description": (
                    "Possível golpe do amor: primeiro envio + valor alto + "
                    "perfil de vulnerabilidade"
                ),
            },

            # ─── 6. IDOSO VULNERÁVEL 70+ ─────────────────────────
            "IDOSO_VULNERAVEL_70": {
                "required": ["idade_70_plus", "primeiro_envio"],
                "optional": [
                    "valor_alto_vs_historico", "chave_aleatoria",
                    "horario_comercial", "valor_redondo",
                    "is_segmento_premium", "login_senha",
                    "perfil_vulneravel_se", "renda_incompativel",
                ],
                "min_score": 4,  # Ajustado: era 5, removido mulher_idosa
                "severity": "CRITICO",
                "description": (
                    "Cliente 70+ enviando para destino desconhecido — "
                    "padrão de vítima de engenharia social"
                ),
            },

            # ─── 7. IDOSO VULNERÁVEL 80+ ─────────────────────────
            "IDOSO_VULNERAVEL_80": {
                "required": ["idade_80_plus"],
                "optional": [
                    "valor_alto_vs_historico", "primeiro_envio",
                    "chave_aleatoria", "perfil_vulneravel_se",
                    "renda_incompativel", "login_senha",
                ],
                "min_score": 3,
                "severity": "CRITICO",
                "description": (
                    "Cliente 80+ — alta vulnerabilidade a golpes de "
                    "engenharia social"
                ),
            },

            # ─── 8. CONTA LARANJA (SAÍDA) ────────────────────────
            "CONTA_LARANJA_SAIDA": {
                "required": ["conta_recem_aberta", "valor_absoluto_alto"],
                "optional": [
                    "multiplos_pix_rapidos", "alta_frequencia_diaria",
                    "chave_aleatoria", "primeiro_envio",
                    "burst_intenso", "renda_desconhecida_valor_alto",
                ],
                "min_score": 4,  # Ajustado: era 5, removido frequencia_anormal
                "severity": "CRITICO",
                "description": (
                    "Padrão de conta laranja: conta nova + alto volume + "
                    "destinos múltiplos"
                ),
            },

            # ─── 9. GOLPE DE INVESTIMENTO ────────────────────────
            "GOLPE_INVESTIMENTO": {
                "required": ["escalada_valores"],
                "optional": [
                    "primeiro_envio", "chave_aleatoria",
                    "recebedor_pj", "valor_alto_vs_historico",
                    "valor_absoluto_alto", "renda_metade_comprometida",
                ],
                "min_score": 4,   # Ajustado: era 5, removido pix_acima_maximo_historico
                "severity": "ALTO",
                "description": (
                    "Possível golpe de investimento: padrão de escalada de "
                    "valores para destino desconhecido"
                ),
            },

            # ─── 10. COAÇÃO FÍSICA ───────────────────────────────
            "COACAO_FISICA": {
                "required": ["intervalo_muito_curto", "valor_absoluto_muito_alto"],
                "optional": [
                    "horario_noturno", "horario_madrugada",
                    "multiplos_pix_rapidos", "chave_aleatoria",
                    "burst_intenso", "renda_incompativel",
                ],
                "min_score": 5,
                "severity": "CRITICO",
                "description": (
                    "URGENTE: Possível coação física — múltiplos PIX de "
                    "alto valor em intervalos < 5 minutos"
                ),
            },

            # ─── 11. TRANSAÇÃO ATÍPICA ───────────────────────────
            "TRANSACAO_ATIPICA": {
                "required": ["zscore_valor_extremo", "primeiro_envio"],
                "optional": [
                    "chave_aleatoria", "horario_noturno",
                    "valor_absoluto_alto", "renda_metade_comprometida",
                ],
                "min_score": 4,
                "severity": "MEDIO",
                "description": (
                    "Transação com valor extremo (z-score ≥ 3) para "
                    "destinatário desconhecido"
                ),
            },
        }

    # =============================================================
    # ESCALADA DE VALORES (usando cache do pipeline)
    # =============================================================
    def _detectar_escalada_valores(self, f: Dict[str, Any]) -> bool:
        """
        Detecta padrão de escalada de valores.

        Adaptação #5: Usa cache interno + zscore + ratio_valor_mediana
        em vez de depender de vl_transacao_anterior (que não existe no pipeline).

        Lógica:
        - Se ratio_valor_mediana >= 3 E zscore >= 2 → possível escalada
        - Se há cache de valores anteriores para o CPF → compara sequência
        """
        cpf = str(f.get("customer_id") or "")
        vl_pix = f.get("vl_pix", 0)
        ratio = f.get("ratio_valor_mediana")
        zscore = f.get("zscore_valor_aprox")

        # Método 1: Comparar com cache de transações anteriores
        if cpf and cpf != "None":
            history = self._value_cache.get(cpf, [])

            # Registrar valor atual no cache
            if cpf not in self._value_cache:
                self._value_cache[cpf] = []
            self._value_cache[cpf].append(vl_pix)
            if len(self._value_cache[cpf]) > self._max_cache_size:
                self._value_cache[cpf] = self._value_cache[cpf][-self._max_cache_size:]

            # Se tem pelo menos 2 valores anteriores, verificar escalada
            if len(history) >= 2:
                # Escalada: os 3 últimos valores são crescentes
                recent = history[-2:] + [vl_pix]
                if recent[0] < recent[1] < recent[2]:
                    mediana = f.get("vl_mediana_pix_trimestre", 0)
                    if mediana > 0 and recent[2] > mediana * 2:
                        return True

        # Método 2: Fallback estatístico (ratio + zscore combinados)
        if ratio is not None and zscore is not None:
            if ratio >= 3.0 and zscore >= 2.0:
                return True

        return False

    # =============================================================
    # PUBLIC API: detect_from_pipeline (método principal)
    # =============================================================
    def detect_from_pipeline(self, features: Dict[str, Any]) -> SEAnalysisResult:
        """
        Método principal — recebe features já processadas pelo pipeline.

        Args:
            features: Dict com features do preprocessing.py v3.1
                      (pode conter nomes do pipeline ou legados).

        Returns:
            SEAnalysisResult com score 0-100, padrões e indicadores ativos.
        """
        # Adaptar features para formato interno
        f = self._adapt_features(features)

        # Avaliar todos os indicadores
        active_indicators: Dict[str, bool] = {}
        for name, check in self.INDICATORS.items():
            try:
                active_indicators[name] = check(f)
            except Exception as e:
                logger.debug(f"Indicador '{name}' falhou: {e}")
                active_indicators[name] = False

        # Verificar indicadores de fase 2
        phase2_missing = []
        for ind in self.PHASE2_INDICATORS:
            phase2_missing.append(ind)

        # Avaliar padrões
        detected: List[PatternMatch] = []
        for pattern_name, config in self.PATTERNS.items():
            score = 0
            matched = []

            # Required: TODOS devem estar ativos (2 pontos cada)
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

            # Optional: 1 ponto cada
            for ind in config["optional"]:
                if active_indicators.get(ind, False):
                    score += 1
                    matched.append(ind)

            # Verificar min_score
            if score >= config["min_score"]:
                detected.append(PatternMatch(
                    pattern_name=pattern_name,
                    severity=config["severity"],
                    score=score,
                    matched_indicators=matched,
                    description=config["description"],
                ))

        # Ordenar por severidade e score
        detected.sort(key=lambda x: (_SEVERITY_ORDER.get(x.severity, 99), -x.score))

        # Calcular score SE
        se_score = self._calculate_se_score(detected, active_indicators)

        return SEAnalysisResult(
            se_score=se_score,
            patterns=detected,
            active_indicators=active_indicators,
            phase2_indicators_missing=phase2_missing,
        )

    # =============================================================
    # PUBLIC API: detect_patterns (compatibilidade com orquestrador)
    # =============================================================
    def detect_patterns(self, features: Dict[str, Any]) -> List[PatternMatch]:
        """
        API de compatibilidade — retorna apenas a lista de padrões.
        Usado pelo PixDecisionEngine existente.
        """
        result = self.detect_from_pipeline(features)
        return result.patterns

    def get_worst_pattern(self, features: Dict[str, Any]) -> Optional[PatternMatch]:
        """Retorna o padrão mais grave detectado, se houver."""
        result = self.detect_from_pipeline(features)
        return result.worst_pattern

    def calculate_social_engineering_score(
        self, features: Dict[str, Any]
    ) -> Tuple[float, List[PatternMatch]]:
        """
        API de compatibilidade — retorna (score, patterns).
        Usado pelo PixDecisionEngine existente.
        """
        result = self.detect_from_pipeline(features)
        return result.se_score, result.patterns

    # =============================================================
    # PRIVATE: Score calculation
    # =============================================================
    @staticmethod
    def _calculate_se_score(
        patterns: List[PatternMatch],
        active_indicators: Dict[str, bool],
    ) -> float:
        """
        Calcula score de engenharia social 0-100.

        Lógica:
        - Base: soma de severidades dos padrões detectados
        - Atenuante: agendamento_recorrente reduz -15
        """
        if not patterns:
            return 0.0

        score = 0.0
        severity_scores = {
            "CRITICO": 40,
            "ALTO": 25,
            "MEDIO": 15,
            "BAIXO": 10,
        }

        for pattern in patterns:
            score += severity_scores.get(pattern.severity, 10)

        # Atenuante: agendamento recorrente
        if active_indicators.get("agendamento_recorrente", False):
            score = max(0.0, score - 15)

        return min(100.0, score)


# =========================================================
# MODULE-LEVEL HELPERS
# =========================================================
def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    if val is None:
        return default
    try:
        v = float(val)
        if v != v:  # NaN
            return default
        return v
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    try:
        v = float(val)
        if v != v:  # NaN
            return default
        return int(v)
    except (ValueError, TypeError):
        return default


def _is_valor_redondo(valor: float) -> bool:
    """Valores redondos são típicos de golpes (R$1000, R$5000, R$500)."""
    if not valor or valor <= 0:
        return False
    return valor >= 100 and (valor % 100 == 0)
