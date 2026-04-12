"""
core/behavioral_analytics.py v3.1 — Recalibrado para Leakage-Free

Evolução v3.0 → v3.1 baseada na validação com dataset leakage-free
(base_mvp_model_ready_leakage_free.csv, 100.355 tx, rolling window causal).

Problema identificado:
  No v3.0, os fatores dormancy usavam gates calibrados no dataset "optimized"
  que tinha data leakage temporal (qt_total_pix_trimestre calculado com visão
  completa do trimestre). No dataset leakage-free (rolling window causal):
  - 97.7% dos normais têm qt_total_pix_trimestre ≤ 2 → gate inútil
  - is_first_tx_trimestre Lift 0.568x → ANTI-INDICADOR
  - FP explodiu: 1.797 → 7.544 (+5.747)
  - PERFIL_VULNERAVEL_SE: TP=5, FP=11.787, Prec=0.04%

Mudanças v3.0 → v3.1:
  ─────────────────────────────────────────────────────────────────────────
  REMOVIDOS (2 fatores — mortos no leakage-free):
    ✗ PRIMEIRA_TX_VALOR_ALTO    is_first_tx_trimestre Lift 0.568x (anti-indicador)
    ✗ PERFIL_VULNERAVEL_SE      TP=5, FP=11.787, Prec=0.04% (lixo puro)

  RECALIBRADOS (2 fatores — gates ajustados para leakage-free):
    ⟳ CONTA_DORMANTE_VALOR_ALTO  Gate: qt_pix ≤ 2 & vl ≥ 1000
                                   → qt_pix == 0 & vl ≥ 5000
                                 LF: TP=58, FP=738, Prec=7.3%, Lift=22.1x
                                 Score: 20 → 15 (precision caiu)

    ⟳ CONTA_DORMANTE_IDOSO       Gate: qt_pix ≤ 2 & idade ≥ 60 & vl ≥ 500
                                   → idade ≥ 65 & vl ≥ 2000 (SEM gate dormancy)
                                 Renomeado: IDOSO_VALOR_ALTO
                                 LF: TP=101, FP=304, Prec=24.9%, Lift=93.6x
                                 Score: 25 → 20

  NOVOS (1 fator — descoberto na exploração B2-LF):
    ★ IDOSO_VALOR_CRITICO        idade ≥ 70 & vl ≥ 5000
                                 LF: TP=45, FP=65, Prec=40.9%, Lift=195.0x
                                 Score: +10 (boost sobre IDOSO_VALOR_ALTO)

  MANTIDOS (3 fatores — velocity intactos, Lift > 100x):
    ✓ FREQUENCIA_BURST           Prec 97.4%, TP=75,  FP=2   (idêntico)
    ✓ BURST_CONTA_COMPROMETIDA   Prec 80.0%, TP=8,   FP=2   (idêntico)
    ✓ MULTIPLOS_RECEBEDORES_BURST Prec 35.9%, TP=28, FP=50  (idêntico)

  ─────────────────────────────────────────────────────────────────────────
  Fatores v3.1 (6 total):
  #   Código                          Source       Score  Prec(LF)   Origem
  1   FREQUENCIA_BURST                velocity     +25    97.4%      B1
  2   BURST_CONTA_COMPROMETIDA        velocity     +20    80.0%      B1
  3   MULTIPLOS_RECEBEDORES_BURST     velocity     +20    35.9%      B1
  4   CONTA_DORMANTE_VALOR_EXTREMO    dormancy     +15     7.3%      B2-LF ⟳
  5   IDOSO_VALOR_ALTO                age+value    +20    24.9%      B2-LF ⟳
  6   IDOSO_VALOR_CRITICO             age+value    +10    40.9%      B2-LF ★
  ─────────────────────────────────────────────────────────────────────────
  Nota: Fator 6 é boost condicional sobre fator 5 (co-ativam).
  Um idoso de 72 anos com PIX de R$8.000 recebe:
    IDOSO_VALOR_ALTO (+20) + IDOSO_VALOR_CRITICO (+10) = 30 pontos

  Atenuante: AGENDAMENTO_RECORRENTE   session      -10
  Máximo teórico: 25+20+20+15+20+10 = 110 (capped em 100)

  Impacto estimado v3.0(LF) → v3.1(LF):
    FP (score > 0):     7.544  →  ~1.100  (redução ~85%)
    Precision (score>0): 3.76% →  ~15%    (melhoria ~4x)
    Recall (score>0):   83.1%  →  ~50%    (tradeoff aceitável)
    FPR:                7.54%  →  ~1.1%   (redução ~85%)
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


logger = logging.getLogger(__name__)


# =========================================================
# DATA CLASSES
# =========================================================
@dataclass
class BehavioralRiskFactor:
    """Um fator de risco comportamental detectado em tempo real."""

    codigo: str
    descricao: str
    peso: int          # 1-5 (severidade)
    score_add: float   # pontos adicionados ao behavioral_score
    source: str        # "velocity", "dormancy", "age_value"
    precision: float   # precision empírica validada (leakage-free)
    origin: str        # "B1", "B2", "B2-LF"


@dataclass
class DeviceInfo:
    """Informações do dispositivo (mantido para compatibilidade da API)."""

    device_id: str
    device_model: str
    device_type: str
    app_version: str | None
    is_known: bool
    first_seen: datetime | None = None
    last_seen: datetime | None = None


@dataclass
class SessionMetrics:
    """Métricas da sessão extraídas das features do pipeline."""

    tempo_interacao_ms: float | None
    latencia_rede_ms: float | None
    metodo_login: str
    is_agendamento_recorrente: bool
    duration_estimate_seconds: int | None


@dataclass
class BehavioralAnalysisResult:
    """Resultado completo da análise comportamental."""

    behavioral_score: float = 0.0
    risk_factors: list[BehavioralRiskFactor] = field(default_factory=list)
    device_info: DeviceInfo | None = None
    session_metrics: SessionMetrics | None = None
    fatores_atenuantes: list[str] = field(default_factory=list)

    @property
    def risk_level(self) -> str:
        if self.behavioral_score >= 60:
            return "CRITICO"
        if self.behavioral_score >= 40:
            return "ALTO"
        if self.behavioral_score >= 20:
            return "MEDIO"
        return "BAIXO"

    @property
    def total_fatores(self) -> int:
        return len(self.risk_factors)

    def to_dict(self) -> dict[str, Any]:
        """Serializa para dict (usado pelo orquestrador/API)."""
        return {
            "behavioral_score": round(self.behavioral_score, 2),
            "risk_level": self.risk_level,
            "total_fatores": self.total_fatores,
            "risk_factors": [
                {
                    "codigo": rf.codigo,
                    "descricao": rf.descricao,
                    "peso": rf.peso,
                    "score_add": rf.score_add,
                    "source": rf.source,
                    "precision": rf.precision,
                    "origin": rf.origin,
                }
                for rf in self.risk_factors
            ],
            "fatores_atenuantes": self.fatores_atenuantes,
            "device_info": {
                "device_id": self.device_info.device_id,
                "device_model": self.device_info.device_model,
                "device_type": self.device_info.device_type,
                "app_version": self.device_info.app_version,
                "is_known": self.device_info.is_known,
            } if self.device_info else None,
            "session_metrics": {
                "tempo_interacao_ms": self.session_metrics.tempo_interacao_ms,
                "latencia_rede_ms": self.session_metrics.latencia_rede_ms,
                "metodo_login": self.session_metrics.metodo_login,
                "is_agendamento_recorrente": (
                    self.session_metrics.is_agendamento_recorrente
                ),
                "duration_estimate_seconds": (
                    self.session_metrics.duration_estimate_seconds
                ),
            } if self.session_metrics else None,
        }

    def to_features(self) -> dict[str, float]:
        """Exporta features numéricas para alimentar o score final."""
        return {
            "behavioral_score": round(self.behavioral_score, 2),
            "behavioral_risk_factor_count": float(self.total_fatores),
            "behavioral_has_velocity_factor": float(
                any(rf.source == "velocity" for rf in self.risk_factors)
            ),
            "behavioral_has_dormancy_factor": float(
                any(rf.source == "dormancy" for rf in self.risk_factors)
            ),
            "behavioral_has_age_value_factor": float(
                any(rf.source == "age_value" for rf in self.risk_factors)
            ),
            "behavioral_max_precision": float(
                max((rf.precision for rf in self.risk_factors), default=0.0)
            ),
        }


# =========================================================
# INLINE PROFILE MANAGER (cache em memória)
# =========================================================
class _InlineProfileManager:
    """
    Gerenciador de perfis comportamentais em memória.

    Mantém histórico leve por CPF para comparação intra-sessão.
    Em produção, substituir por Redis/DynamoDB.
    """

    def __init__(
        self,
        max_tx_history: int = 50,
        max_profiles: int = 100_000,
    ) -> None:
        self._profiles: dict[str, dict[str, Any]] = {}
        self._tx_history: dict[str, list[datetime]] = defaultdict(list)
        self._max_tx_history = max_tx_history
        self._max_profiles = max_profiles

    def get_or_create(self, cpf: str) -> dict[str, Any]:
        """Retorna ou cria profile para o CPF."""
        if cpf not in self._profiles:
            if len(self._profiles) >= self._max_profiles:
                oldest_cpf = next(iter(self._profiles))
                del self._profiles[oldest_cpf]
                self._tx_history.pop(oldest_cpf, None)

            self._profiles[cpf] = {
                "devices_conhecidos": set(),
                "metodo_login_principal": "desconhecido",
                "login_counts": defaultdict(int),
                "total_tx": 0,
            }
        return self._profiles[cpf]

    def is_device_known(self, cpf: str, device_id: str) -> bool:
        """Verifica se device já foi visto para este CPF."""
        profile = self.get_or_create(cpf)
        return device_id in profile["devices_conhecidos"]

    def register_device(self, cpf: str, device_id: str) -> None:
        """Registra device no histórico do CPF."""
        profile = self.get_or_create(cpf)
        profile["devices_conhecidos"].add(device_id)

    def register_login(self, cpf: str, metodo: str) -> None:
        """Registra método de login no profile."""
        profile = self.get_or_create(cpf)
        profile["login_counts"][metodo] += 1
        if profile["login_counts"]:
            profile["metodo_login_principal"] = max(
                profile["login_counts"],
                key=profile["login_counts"].get,  # type: ignore[arg-type]
            )

    def register_tx(self, cpf: str, dt: datetime) -> None:
        """Registra transação no histórico temporal."""
        profile = self.get_or_create(cpf)
        profile["total_tx"] += 1
        history = self._tx_history[cpf]
        history.append(dt)
        if len(history) > self._max_tx_history:
            self._tx_history[cpf] = history[-self._max_tx_history:]


# =========================================================
# BEHAVIORAL ANALYTICS ENGINE v3.1
# =========================================================
class BehavioralAnalytics:
    """
    Motor de análise comportamental v3.1 — Leakage-Free Validated.

    6 fatores de risco, todos validados com dados causalmente corretos
    (rolling window, sem data leakage temporal).

    Categorias:
      - velocity  (3): padrões de burst/frequência (Tier 1, B1)
      - dormancy  (1): conta dormante + valor extremo (Tier 2, B2-LF recalibrado)
      - age_value (2): idoso + valor alto (Tier 2, B2-LF novo)
    """

    VERSION = "3.1"

    def __init__(self) -> None:
        self._profile_mgr = _InlineProfileManager()
        self._device_history: dict[str, DeviceInfo] = {}
        logger.info(
            "BehavioralAnalytics v%s inicializado "
            "(6 fatores leakage-free validated: "
            "3 velocity + 1 dormancy + 2 age_value)",
            self.VERSION,
        )

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------
    def analyze(self, features: dict[str, Any]) -> BehavioralAnalysisResult:
        """
        Analisa comportamento da transação em tempo real.

        Args:
            features: Dict com features processadas pelo pipeline.

        Returns:
            BehavioralAnalysisResult com score 0-100 e fatores de risco.
        """
        cpf = str(
            features.get("cd_cpf_pagador")
            or features.get("customer_id")
            or ""
        )
        now = datetime.utcnow()

        # --- Extrair dados ---
        device_info = self._build_device_info(cpf, features, now)
        session_metrics = self._build_session_metrics(features)
        ext = self._extract_features(features)

        # --- Avaliar fatores de risco ---
        risk_factors: list[BehavioralRiskFactor] = []
        atenuantes: list[str] = []
        score = 0.0

        # ─────────────────────────────────────────────────────
        # TIER 1 — VELOCITY (B1, intactos — à prova de leakage)
        # ─────────────────────────────────────────────────────

        # 1. FREQUENCIA_BURST
        #    Condição: burst_30m_flag=1 AND tx_count_prev_30m >= 2
        #    LF: Prec 97.4%, TP=75, FP=2, Lift 10.563x
        if ext["burst_30m_flag"] and ext["tx_count_prev_30m"] >= 2:
            rf = BehavioralRiskFactor(
                codigo="FREQUENCIA_BURST",
                descricao=(
                    f"{ext['tx_count_prev_30m'] + 1} transações em 30 minutos "
                    f"(PIX de R${ext['vl_pix']:,.2f}) — padrão de burst confirmado"
                ),
                peso=5,
                score_add=25,
                source="velocity",
                precision=0.974,
                origin="B1",
            )
            risk_factors.append(rf)
            score += rf.score_add

        # 2. BURST_CONTA_COMPROMETIDA
        #    Condição: conta >= 12m + tx_count >= 2 + first_receiver + valor >= 500
        #    LF: Prec 80.0%, TP=8, FP=2, Lift 1.127x
        if (
            ext["qt_tempo_relacionamento_mes"] >= 12
            and ext["tx_count_prev_30m"] >= 2
            and ext["first_receiver_flag"]
            and ext["vl_pix"] >= 500
        ):
            rf = BehavioralRiskFactor(
                codigo="BURST_CONTA_COMPROMETIDA",
                descricao=(
                    f"Conta antiga ({ext['qt_tempo_relacionamento_mes']} meses) "
                    f"com burst ({ext['tx_count_prev_30m'] + 1} tx em 30min) "
                    f"para recebedor novo — R${ext['vl_pix']:,.2f}"
                ),
                peso=4,
                score_add=20,
                source="velocity",
                precision=0.80,
                origin="B1",
            )
            risk_factors.append(rf)
            score += rf.score_add

        # 3. MULTIPLOS_RECEBEDORES_BURST
        #    Condição: burst_30m_flag + distinct_receivers >= 3
        #    LF: Prec 35.9%, TP=28, FP=50, Lift 158x
        if ext["burst_30m_flag"] and ext["distinct_receivers_so_far"] >= 3:
            rf = BehavioralRiskFactor(
                codigo="MULTIPLOS_RECEBEDORES_BURST",
                descricao=(
                    f"Burst com {ext['distinct_receivers_so_far']} recebedores "
                    f"distintos — esvaziamento pulverizado"
                ),
                peso=4,
                score_add=20,
                source="velocity",
                precision=0.359,
                origin="B1",
            )
            risk_factors.append(rf)
            score += rf.score_add

        # ─────────────────────────────────────────────────────
        # TIER 2 — DORMANCY (recalibrado para leakage-free)
        # ─────────────────────────────────────────────────────

        # 4. CONTA_DORMANTE_VALOR_EXTREMO
        #    Condição: qt_total_pix_trimestre == 0 AND vl_pix >= 5000
        #    LF: TP=58, FP=738, Prec=7.3%, Lift=22.1x
        #    Mudança v3.0→v3.1: gate restrito (qt_pix==0, vl>=5000)
        #    Semântica: conta sem NENHUMA tx no trimestre que faz PIX >= R$5k
        #    Score reduzido para 15 (precision caiu vs v3.0)
        if ext["qt_total_pix_trimestre"] == 0 and ext["vl_pix"] >= 5000:
            rf = BehavioralRiskFactor(
                codigo="CONTA_DORMANTE_VALOR_EXTREMO",
                descricao=(
                    f"Conta sem transações no trimestre fazendo PIX de "
                    f"R${ext['vl_pix']:,.2f} — padrão de conta dormante comprometida"
                ),
                peso=3,
                score_add=15,
                source="dormancy",
                precision=0.073,
                origin="B2-LF",
            )
            risk_factors.append(rf)
            score += rf.score_add

        # ─────────────────────────────────────────────────────
        # TIER 2 — AGE + VALUE (novo no B2-LF)
        # ─────────────────────────────────────────────────────

        # 5. IDOSO_VALOR_ALTO
        #    Condição: nr_idade >= 65 AND vl_pix >= 2000
        #    LF: TP=101, FP=304, Prec=24.9%, Lift=93.6x, F1=0.266
        #    Insight: funciona MELHOR sem gate de dormancy — idosos fraudados
        #    são alvo independente de atividade da conta
        #    Substitui CONTA_DORMANTE_IDOSO do v3.0
        if ext["nr_idade"] >= 65 and ext["vl_pix"] >= 2000:
            rf = BehavioralRiskFactor(
                codigo="IDOSO_VALOR_ALTO",
                descricao=(
                    f"Idoso ({ext['nr_idade']} anos) com PIX de "
                    f"R${ext['vl_pix']:,.2f} — alto risco de engenharia social"
                ),
                peso=4,
                score_add=20,
                source="age_value",
                precision=0.249,
                origin="B2-LF",
            )
            risk_factors.append(rf)
            score += rf.score_add

        # 6. IDOSO_VALOR_CRITICO (boost condicional)
        #    Condição: nr_idade >= 70 AND vl_pix >= 5000
        #    LF: TP=45, FP=65, Prec=40.9%, Lift=195.0x
        #    Semântica: boost adicional — idoso 70+ com valor muito alto
        #    Co-ativa com IDOSO_VALOR_ALTO (score aditivo)
        #    Idoso de 72 com R$8k recebe: 20 (fator 5) + 10 (fator 6) = 30
        if ext["nr_idade"] >= 70 and ext["vl_pix"] >= 5000:
            rf = BehavioralRiskFactor(
                codigo="IDOSO_VALOR_CRITICO",
                descricao=(
                    f"Idoso 70+ ({ext['nr_idade']} anos) com PIX crítico "
                    f"R${ext['vl_pix']:,.2f} — risco máximo de engenharia social"
                ),
                peso=5,
                score_add=10,
                source="age_value",
                precision=0.409,
                origin="B2-LF",
            )
            risk_factors.append(rf)
            score += rf.score_add

        # ─────────────────────────────────────────────────────
        # ATENUANTE
        # ─────────────────────────────────────────────────────

        if session_metrics.is_agendamento_recorrente:
            atenuantes.append(
                "AGENDAMENTO_RECORRENTE: PIX recorrente agendado — risco atenuado"
            )
            score = max(0.0, score - 10)

        # --- Registrar no profile manager ---
        self._register_transaction(cpf, features, device_info, session_metrics, now)

        # --- Normalizar score ---
        score = min(100.0, max(0.0, score))

        return BehavioralAnalysisResult(
            behavioral_score=score,
            risk_factors=risk_factors,
            device_info=device_info,
            session_metrics=session_metrics,
            fatores_atenuantes=atenuantes,
        )

    # ---------------------------------------------------------
    # PRIVATE: Registrar transação no profile manager
    # ---------------------------------------------------------
    def _register_transaction(
        self,
        cpf: str,
        features: dict[str, Any],
        device_info: DeviceInfo,
        session_metrics: SessionMetrics,
        now: datetime,
    ) -> None:
        """Registra device, login e timestamp no profile manager."""
        self._profile_mgr.register_device(cpf, device_info.device_id)
        self._profile_mgr.register_login(cpf, session_metrics.metodo_login)
        try:
            dt_pix = features.get("dt_pix") or features.get("event_datetime")
            if isinstance(dt_pix, str):
                dt_obj = datetime.fromisoformat(dt_pix.replace("Z", "+00:00"))
            elif isinstance(dt_pix, datetime):
                dt_obj = dt_pix
            else:
                dt_obj = now
            self._profile_mgr.register_tx(cpf, dt_obj)
        except Exception:
            self._profile_mgr.register_tx(cpf, now)

    # ---------------------------------------------------------
    # PRIVATE: Build device info
    # ---------------------------------------------------------
    def _build_device_info(
        self,
        cpf: str,
        features: dict[str, Any],
        now: datetime,
    ) -> DeviceInfo:
        """Constrói DeviceInfo a partir das features (mantido para API)."""
        device_name = (
            features.get("device_name")
            or features.get("device_name_normalized")
            or ""
        )
        app_version = features.get("app_version")
        device_model = str(device_name).strip() if device_name else "Desconhecido"
        device_type = self._infer_device_type(device_model)

        device_id_seed = f"{cpf}|{device_model}|{app_version or ''}"
        device_id = hashlib.sha256(device_id_seed.encode()).hexdigest()[:16]

        is_known = self._profile_mgr.is_device_known(cpf, device_id)

        if device_id in self._device_history:
            existing = self._device_history[device_id]
            existing.last_seen = now
            if app_version:
                existing.app_version = str(app_version)
        else:
            self._device_history[device_id] = DeviceInfo(
                device_id=device_id,
                device_model=device_model,
                device_type=device_type,
                app_version=str(app_version) if app_version else None,
                is_known=is_known,
                first_seen=now,
                last_seen=now,
            )

        return DeviceInfo(
            device_id=device_id,
            device_model=device_model,
            device_type=device_type,
            app_version=str(app_version) if app_version else None,
            is_known=is_known,
            first_seen=self._device_history[device_id].first_seen,
            last_seen=now,
        )

    # ---------------------------------------------------------
    # PRIVATE: Build session metrics
    # ---------------------------------------------------------
    def _build_session_metrics(
        self,
        features: dict[str, Any],
    ) -> SessionMetrics:
        """Constrói SessionMetrics a partir das features."""
        tempo_raw = (
            features.get("tempo_interacao_ms")
            or features.get("tempo_interacao_ms_final")
        )
        tempo_ms = self._safe_float(tempo_raw)

        latencia_raw = (
            features.get("latencia_rede_ms")
            or features.get("latencia_rede_ms_final")
        )
        latencia_ms = self._safe_float(latencia_raw)

        metodo_raw = features.get("metodo_autenticacao")
        metodo = self._normalize_login_method(metodo_raw)

        # Agendamento recorrente: verifica flag int e fallback string
        is_recorrente_raw = features.get("is_agendamento_recorrente_flag")
        if is_recorrente_raw is not None:
            is_recorrente = bool(self._safe_int(is_recorrente_raw, 0))
        else:
            is_recorrente_str = features.get("is_agendamento_recorrente", "")
            is_recorrente = str(is_recorrente_str).strip().lower() in (
                "true", "1", "sim",
            )

        duration_s = (
            int(tempo_ms / 1000) if tempo_ms and tempo_ms > 0 else None
        )

        return SessionMetrics(
            tempo_interacao_ms=tempo_ms,
            latencia_rede_ms=latencia_ms,
            metodo_login=metodo,
            is_agendamento_recorrente=is_recorrente,
            duration_estimate_seconds=duration_s,
        )

    # ---------------------------------------------------------
    # PRIVATE: Extract features
    # ---------------------------------------------------------
    def _extract_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """Extrai e normaliza features relevantes do dict do pipeline."""
        return {
            # Valor da transação
            "vl_pix": self._safe_float(features.get("vl_pix"), 0.0),

            # Dados do cliente
            "nr_idade": self._safe_int(features.get("nr_idade"), 0),
            "qt_tempo_relacionamento_mes": self._safe_int(
                features.get("qt_tempo_relacionamento_mes"), 999,
            ),

            # Velocity / Burst
            "burst_30m_flag": bool(
                self._safe_int(features.get("burst_30m_flag"), 0),
            ),
            "tx_count_prev_30m": self._safe_int(
                features.get("tx_count_prev_30m"), 0,
            ),
            "distinct_receivers_so_far": self._safe_int(
                features.get("distinct_receivers_so_far"), 1,
            ),
            "first_receiver_flag": bool(
                self._safe_int(features.get("first_receiver_flag"), 0),
            ),

            # Dormancy (recalibrado v3.1 — só qt_total_pix_trimestre)
            "qt_total_pix_trimestre": self._safe_int(
                features.get("qt_total_pix_trimestre"), 999,
            ),
            # NOTA: is_first_tx_trimestre REMOVIDO (anti-indicador, Lift 0.568x)
            # NOTA: perfil_vulneravel_se_flag REMOVIDO (Prec 0.04%, TP=5, FP=11787)
        }

    # ---------------------------------------------------------
    # STATIC HELPERS
    # ---------------------------------------------------------
    @staticmethod
    def _infer_device_type(device_model: str) -> str:
        """Infere tipo de device a partir do model name."""
        model = device_model.lower()
        if any(kw in model for kw in ("iphone", "ipad", "ios", "apple")):
            return "iOS"
        if any(
            kw in model
            for kw in (
                "samsung", "xiaomi", "motorola", "pixel",
                "oneplus", "huawei", "lg", "galaxy", "redmi",
            )
        ):
            return "Android"
        return "Desconhecido"

    @staticmethod
    def _normalize_login_method(raw: Any) -> str:
        """Normaliza string de método de autenticação."""
        if raw is None:
            return "desconhecido"
        val = str(raw).strip().lower()
        if val in ("1", "bio", "biometria", "biometric"):
            return "biometria"
        if val in ("2", "senha", "password"):
            return "senha"
        if val in ("3", "pin"):
            return "pin"
        if "bio" in val:
            return "biometria"
        if "senha" in val:
            return "senha"
        if "pin" in val:
            return "pin"
        return "desconhecido"

    @staticmethod
    def _safe_float(
        val: Any,
        default: float | None = None,
    ) -> float | None:
        """Converte para float de forma segura."""
        if val is None:
            return default
        try:
            v = float(val)
            if v != v:  # NaN check
                return default
            return v
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_int(val: Any, default: int = 0) -> int:
        """Converte para int de forma segura."""
        if val is None:
            return default
        try:
            v = float(val)
            if v != v:  # NaN check
                return default
            return int(v)
        except (ValueError, TypeError):
            return default

    # ---------------------------------------------------------
    # PUBLIC: Summary
    # ---------------------------------------------------------
    def get_risk_summary(self, result: BehavioralAnalysisResult) -> str:
        """Retorna resumo textual para logs e API."""
        if result.behavioral_score == 0:
            return "Comportamento normal — sem anomalias detectadas"

        level = result.risk_level
        count = result.total_fatores
        top_factors = ", ".join(rf.codigo for rf in result.risk_factors[:3])

        sources = {rf.source for rf in result.risk_factors}
        source_str = "+".join(sorted(sources))

        summary = (
            f"Risco comportamental {level} "
            f"({result.behavioral_score:.0f}/100): "
            f"{count} fator(es) [{source_str}] — {top_factors}"
        )

        if result.fatores_atenuantes:
            summary += f" | Atenuantes: {len(result.fatores_atenuantes)}"

        return summary

    # ---------------------------------------------------------
    # PUBLIC: Factor catalog (para documentação/testes)
    # ---------------------------------------------------------
    @staticmethod
    def get_factor_catalog() -> list[dict[str, Any]]:
        """Retorna catálogo dos 6 fatores com métricas de validação LF."""
        return [
            {
                "codigo": "FREQUENCIA_BURST",
                "tier": 1, "source": "velocity", "origin": "B1",
                "score_add": 25, "precision_lf": 0.974,
                "tp_lf": 75, "fp_lf": 2,
                "condition": "burst_30m_flag=1 AND tx_count_prev_30m >= 2",
                "status": "intacto",
            },
            {
                "codigo": "BURST_CONTA_COMPROMETIDA",
                "tier": 1, "source": "velocity", "origin": "B1",
                "score_add": 20, "precision_lf": 0.80,
                "tp_lf": 8, "fp_lf": 2,
                "condition": (
                    "qt_tempo_relacionamento_mes >= 12 AND "
                    "tx_count_prev_30m >= 2 AND "
                    "first_receiver_flag=1 AND vl_pix >= 500"
                ),
                "status": "intacto",
            },
            {
                "codigo": "MULTIPLOS_RECEBEDORES_BURST",
                "tier": 1, "source": "velocity", "origin": "B1",
                "score_add": 20, "precision_lf": 0.359,
                "tp_lf": 28, "fp_lf": 50,
                "condition": (
                    "burst_30m_flag=1 AND distinct_receivers_so_far >= 3"
                ),
                "status": "intacto",
            },
            {
                "codigo": "CONTA_DORMANTE_VALOR_EXTREMO",
                "tier": 2, "source": "dormancy", "origin": "B2-LF",
                "score_add": 15, "precision_lf": 0.073,
                "tp_lf": 58, "fp_lf": 738,
                "condition": (
                    "qt_total_pix_trimestre == 0 AND vl_pix >= 5000"
                ),
                "v30_gate": "qt_total_pix_trimestre <= 2 AND vl_pix >= 1000",
                "status": "recalibrado",
                "rationale": (
                    "97.7% dos normais têm qt_pix <= 2 no LF. "
                    "Restringir para ==0 e vl>=5000 reduz FP de 5215 para 738."
                ),
            },
            {
                "codigo": "IDOSO_VALOR_ALTO",
                "tier": 2, "source": "age_value", "origin": "B2-LF",
                "score_add": 20, "precision_lf": 0.249,
                "tp_lf": 101, "fp_lf": 304,
                "condition": "nr_idade >= 65 AND vl_pix >= 2000",
                "v30_gate": (
                    "qt_total_pix_trimestre <= 2 AND "
                    "nr_idade >= 60 AND vl_pix >= 500"
                ),
                "status": "recalibrado",
                "rationale": (
                    "Remover gate dormancy MELHORA performance. "
                    "Idosos são alvo independente de atividade da conta. "
                    "Prec: 6.9% → 24.9%, com mais TP (137 → 101 é por "
                    "faixa etária mais restrita, mas FP: 1850 → 304)."
                ),
            },
            {
                "codigo": "IDOSO_VALOR_CRITICO",
                "tier": 2, "source": "age_value", "origin": "B2-LF",
                "score_add": 10, "precision_lf": 0.409,
                "tp_lf": 45, "fp_lf": 65,
                "condition": "nr_idade >= 70 AND vl_pix >= 5000",
                "status": "novo",
                "rationale": (
                    "Boost condicional sobre IDOSO_VALOR_ALTO. "
                    "Prec 40.9%, Lift 195x. Co-ativa com fator 5."
                ),
            },
        ]
