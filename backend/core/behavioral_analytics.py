"""
core/behavioral_analytics.py v2.0 — Análise Comportamental Real para MVP Fraude PIX

Mudanças v2.0 (reescrita completa):
  1. ELIMINADO mock mode — só análise real
  2. REMOVIDAS dependências externas: app.core.ip_service, app.core.user_profile_manager
  3. UserProfileManager EMBUTIDO inline (cache em memória com dict)
  4. REMOVIDOS fatores Topaz (já contabilizados no LGBM via topaz_score_filled,
     topaz_rejeitada_flag, rule_topaz_score — evita contagem dupla)
  5. REMOVIDO touch_pressure_score (dado não existe na base real)
  6. REMOVIDO GeoIP lookup (latência de rede externa inaceitável em RT;
     mover para fase de investigação pós-alerta)
  7. REAPROVEITADAS features do pipeline: rule_age_score, burst_30m_flag,
     first_receiver_flag, ratio_tempo_interacao_cliente, etc.
  8. FOCADO nos 12 fatores viáveis em RT com dados reais disponíveis
  9. OUTPUT PADRONIZADO para integração com pipeline (to_dict + to_features)

Novos fatores v2.0 (com dados v2.1b do Big Data):
  - PERFIL_VULNERAVEL_SE: viúvo + idoso + sem dependentes
  - RENDA_INCOMPATIVEL: PIX > 100% da renda mensal
  - SEGMENTO_PREMIUM_DEVICE_NOVO: cliente premium em device desconhecido
  - LOGIN_SENHA_IDOSO: idoso autenticando por senha (não biometria)

Integração com pipeline:
  - Recebe dict de features já processadas pelo preprocessing.py
  - Retorna BehavioralAnalysisResult com score, fatores e dict para o orquestrador
  - Não faz I/O externo (sem rede, sem banco) — tudo em memória
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

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
    source: str        # "device", "session", "profile", "behavioral", "value"


@dataclass
class DeviceInfo:
    """Informações do dispositivo inferidas dos dados da transação."""
    device_id: str
    device_model: str
    device_type: str            # "Android", "iOS", "Desconhecido"
    app_version: Optional[str]
    is_known: bool              # Se já foi visto para este CPF
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None


@dataclass
class SessionMetrics:
    """Métricas da sessão extraídas das features do pipeline."""
    tempo_interacao_ms: Optional[float]
    latencia_rede_ms: Optional[float]
    metodo_login: str                       # "biometria", "senha", "pin", "desconhecido"
    is_agendamento_recorrente: bool
    duration_estimate_seconds: Optional[int]


@dataclass
class BehavioralAnalysisResult:
    """Resultado completo da análise comportamental."""
    behavioral_score: float                                 # 0-100
    risk_factors: List[BehavioralRiskFactor] = field(default_factory=list)
    device_info: Optional[DeviceInfo] = None
    session_metrics: Optional[SessionMetrics] = None
    fatores_atenuantes: List[str] = field(default_factory=list)

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

    def to_dict(self) -> Dict[str, Any]:
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
                "is_agendamento_recorrente": self.session_metrics.is_agendamento_recorrente,
                "duration_estimate_seconds": self.session_metrics.duration_estimate_seconds,
            } if self.session_metrics else None,
        }

    def to_features(self) -> Dict[str, float]:
        """Exporta features numéricas para alimentar o score final do orquestrador."""
        return {
            "behavioral_score": round(self.behavioral_score, 2),
            "behavioral_risk_factor_count": float(self.total_fatores),
            "behavioral_device_is_known": float(self.device_info.is_known) if self.device_info else 0.0,
            "behavioral_login_senha_flag": float(
                self.session_metrics.metodo_login == "senha"
            ) if self.session_metrics else 0.0,
            "behavioral_agendamento_recorrente_flag": float(
                self.session_metrics.is_agendamento_recorrente
            ) if self.session_metrics else 0.0,
        }


# =========================================================
# INLINE PROFILE MANAGER (cache em memória)
# =========================================================
class _InlineProfileManager:
    """
    Gerenciador de perfis comportamentais em memória.

    Mantém histórico leve por CPF para comparação intra-sessão:
    - Devices conhecidos
    - Método de login habitual
    - Timestamps de transações recentes (para burst detection)

    Em produção, substituir por Redis/DynamoDB.
    """

    def __init__(self, max_tx_history: int = 50, max_profiles: int = 100_000):
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self._tx_history: Dict[str, List[datetime]] = defaultdict(list)
        self._max_tx_history = max_tx_history
        self._max_profiles = max_profiles

    def get_or_create(self, cpf: str) -> Dict[str, Any]:
        if cpf not in self._profiles:
            if len(self._profiles) >= self._max_profiles:
                # Evict mais antigo (LRU simplificado)
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
        profile = self.get_or_create(cpf)
        return device_id in profile["devices_conhecidos"]

    def register_device(self, cpf: str, device_id: str) -> None:
        profile = self.get_or_create(cpf)
        profile["devices_conhecidos"].add(device_id)

    def is_metodo_login_diferente(self, cpf: str, metodo_atual: str) -> bool:
        profile = self.get_or_create(cpf)
        principal = profile["metodo_login_principal"]
        if principal == "desconhecido" or profile["total_tx"] < 3:
            return False  # Sem histórico suficiente
        return metodo_atual != principal

    def register_login(self, cpf: str, metodo: str) -> None:
        profile = self.get_or_create(cpf)
        profile["login_counts"][metodo] += 1
        # Atualizar principal
        if profile["login_counts"]:
            profile["metodo_login_principal"] = max(
                profile["login_counts"], key=profile["login_counts"].get
            )

    def register_tx(self, cpf: str, dt: datetime) -> None:
        profile = self.get_or_create(cpf)
        profile["total_tx"] += 1
        history = self._tx_history[cpf]
        history.append(dt)
        if len(history) > self._max_tx_history:
            self._tx_history[cpf] = history[-self._max_tx_history:]

    def count_recent_tx(self, cpf: str, dt: datetime, window_minutes: int = 5) -> int:
        cutoff = dt - timedelta(minutes=window_minutes)
        return sum(1 for t in self._tx_history.get(cpf, []) if t >= cutoff)


# =========================================================
# BEHAVIORAL ANALYTICS ENGINE
# =========================================================
class BehavioralAnalytics:
    """
    Motor de análise comportamental em tempo real.

    12 fatores de risco viáveis com dados reais disponíveis:
    ─────────────────────────────────────────────────────────
    #   Código                          Source       Score
    1   DEVICE_NOVO                     device       +25
    2   DEVICE_NOVO_IDOSO               device       +20
    3   DEVICE_NOVO_PREMIUM             device       +15
    4   LOGIN_SENHA_ALTO_VALOR          session      +10
    5   LOGIN_SENHA_IDOSO               session      +15
    6   LOGIN_METODO_DIFERENTE          session      +15
    7   SESSAO_RAPIDA_ALTO_VALOR        session      +15
    8   TEMPO_INTERACAO_ANORMAL         behavioral   +15
    9   FREQUENCIA_BURST                behavioral   +20
    10  PERFIL_VULNERAVEL_SE            profile      +20
    11  RENDA_INCOMPATIVEL              value        +25
    12  PRIMEIRO_PIX_CLIENTE_NOVO       profile      +30
    ─────────────────────────────────────────────────────────
    Atenuante: AGENDAMENTO_RECORRENTE   session      -10

    Máximo teórico: ~225 (capped em 100)
    """

    def __init__(self):
        self._profile_mgr = _InlineProfileManager()
        self._device_history: Dict[str, DeviceInfo] = {}
        logger.info("BehavioralAnalytics v2.0 inicializado (modo real, sem mock)")

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------
    def analyze(self, features: Dict[str, Any]) -> BehavioralAnalysisResult:
        """
        Analisa comportamento da transação em tempo real.

        Args:
            features: Dict com features processadas pelo pipeline.
                      Espera campos do preprocessing.py v3.1.

        Returns:
            BehavioralAnalysisResult com score 0-100 e fatores de risco.
        """
        cpf = str(features.get("cd_cpf_pagador") or features.get("customer_id") or "")
        now = datetime.utcnow()

        # --- Extrair dados ---
        device_info = self._build_device_info(cpf, features, now)
        session_metrics = self._build_session_metrics(features)
        extracted = self._extract_features(features)

        # --- Avaliar fatores de risco ---
        risk_factors: List[BehavioralRiskFactor] = []
        atenuantes: List[str] = []
        score = 0.0

        # 1. DEVICE_NOVO
        if not device_info.is_known:
            rf = BehavioralRiskFactor(
                codigo="DEVICE_NOVO",
                descricao=f"Primeiro acesso deste dispositivo ({device_info.device_model})",
                peso=3, score_add=25, source="device",
            )
            risk_factors.append(rf)
            score += rf.score_add

        # 2. DEVICE_NOVO_IDOSO (combinação)
        if not device_info.is_known and extracted["nr_idade"] >= 65:
            rf = BehavioralRiskFactor(
                codigo="DEVICE_NOVO_IDOSO",
                descricao=f"Cliente idoso ({extracted['nr_idade']} anos) em dispositivo novo — alto risco de engenharia social",
                peso=4, score_add=20, source="device",
            )
            risk_factors.append(rf)
            score += rf.score_add

        # 3. DEVICE_NOVO_PREMIUM (combinação)
        if not device_info.is_known and extracted["is_segmento_premium"]:
            rf = BehavioralRiskFactor(
                codigo="DEVICE_NOVO_PREMIUM",
                descricao="Cliente de segmento premium acessando de dispositivo desconhecido",
                peso=3, score_add=15, source="device",
            )
            risk_factors.append(rf)
            score += rf.score_add

        # 4. LOGIN_SENHA_ALTO_VALOR
        if session_metrics.metodo_login == "senha" and extracted["vl_pix"] >= 1000:
            rf = BehavioralRiskFactor(
                codigo="LOGIN_SENHA_ALTO_VALOR",
                descricao=f"Login por senha (não biometria) em PIX de R${extracted['vl_pix']:,.2f}",
                peso=2, score_add=10, source="session",
            )
            risk_factors.append(rf)
            score += rf.score_add

        # 5. LOGIN_SENHA_IDOSO
        if session_metrics.metodo_login == "senha" and extracted["nr_idade"] >= 60:
            rf = BehavioralRiskFactor(
                codigo="LOGIN_SENHA_IDOSO",
                descricao=f"Idoso ({extracted['nr_idade']} anos) autenticando por senha — possível coação",
                peso=3, score_add=15, source="session",
            )
            risk_factors.append(rf)
            score += rf.score_add

        # 6. LOGIN_METODO_DIFERENTE
        if self._profile_mgr.is_metodo_login_diferente(cpf, session_metrics.metodo_login):
            rf = BehavioralRiskFactor(
                codigo="LOGIN_METODO_DIFERENTE",
                descricao=f"Método de login mudou para '{session_metrics.metodo_login}' (diferente do habitual)",
                peso=2, score_add=15, source="session",
            )
            risk_factors.append(rf)
            score += rf.score_add

        # 7. SESSAO_RAPIDA_ALTO_VALOR
        if (
            session_metrics.tempo_interacao_ms is not None
            and session_metrics.tempo_interacao_ms < 30_000  # < 30 segundos
            and extracted["vl_pix"] >= 1000
        ):
            tempo_s = session_metrics.tempo_interacao_ms / 1000
            rf = BehavioralRiskFactor(
                codigo="SESSAO_RAPIDA_ALTO_VALOR",
                descricao=f"Interação muito rápida ({tempo_s:.1f}s) para PIX de R${extracted['vl_pix']:,.2f}",
                peso=3, score_add=15, source="session",
            )
            risk_factors.append(rf)
            score += rf.score_add

        # 8. TEMPO_INTERACAO_ANORMAL
        ratio_tempo = extracted["ratio_tempo_interacao"]
        if ratio_tempo is not None:
            if ratio_tempo < 0.3:
                rf = BehavioralRiskFactor(
                    codigo="TEMPO_INTERACAO_ANORMAL",
                    descricao=f"Interação {ratio_tempo:.1%} do tempo médio — possível automação",
                    peso=3, score_add=15, source="behavioral",
                )
                risk_factors.append(rf)
                score += rf.score_add
            elif ratio_tempo > 3.0:
                rf = BehavioralRiskFactor(
                    codigo="TEMPO_INTERACAO_ANORMAL",
                    descricao=f"Interação {ratio_tempo:.1f}x o tempo médio — possível hesitação/coação",
                    peso=2, score_add=15, source="behavioral",
                )
                risk_factors.append(rf)
                score += rf.score_add

        # 9. FREQUENCIA_BURST
        burst_flag = extracted["burst_30m_flag"]
        tx_count_30m = extracted["tx_count_prev_30m"]
        if burst_flag and tx_count_30m >= 2:
            rf = BehavioralRiskFactor(
                codigo="FREQUENCIA_BURST",
                descricao=f"{tx_count_30m + 1} transações em 30 minutos — padrão de burst",
                peso=3, score_add=20, source="behavioral",
            )
            risk_factors.append(rf)
            score += rf.score_add

        # 10. PERFIL_VULNERAVEL_SE
        if extracted["perfil_vulneravel_se"]:
            rf = BehavioralRiskFactor(
                codigo="PERFIL_VULNERAVEL_SE",
                descricao=f"Perfil de alta vulnerabilidade: viúvo(a), {extracted['nr_idade']} anos, sem dependentes",
                peso=4, score_add=20, source="profile",
            )
            risk_factors.append(rf)
            score += rf.score_add

        # 11. RENDA_INCOMPATIVEL
        if extracted["pix_over_100pct_renda"]:
            ratio_renda = extracted["ratio_pix_renda"]
            rf = BehavioralRiskFactor(
                codigo="RENDA_INCOMPATIVEL",
                descricao=f"PIX de R${extracted['vl_pix']:,.2f} equivale a {ratio_renda:.0%} da renda mensal (R${extracted['vl_renda']:,.2f})",
                peso=4, score_add=25, source="value",
            )
            risk_factors.append(rf)
            score += rf.score_add

        # 12. PRIMEIRO_PIX_CLIENTE_NOVO
        if (
            extracted["qt_tempo_relacionamento_mes"] < 3
            and extracted["first_receiver_flag"]
            and extracted["vl_pix"] >= 1000
        ):
            rf = BehavioralRiskFactor(
                codigo="PRIMEIRO_PIX_CLIENTE_NOVO",
                descricao=f"Cliente novo ({extracted['qt_tempo_relacionamento_mes']} meses), primeiro envio ao destinatário, PIX de R${extracted['vl_pix']:,.2f}",
                peso=5, score_add=30, source="profile",
            )
            risk_factors.append(rf)
            score += rf.score_add

        # --- Atenuante: AGENDAMENTO_RECORRENTE ---
        if session_metrics.is_agendamento_recorrente:
            atenuantes.append("AGENDAMENTO_RECORRENTE: PIX recorrente agendado — risco atenuado")
            score = max(0.0, score - 10)

        # --- Registrar no profile manager ---
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
    # PRIVATE: Build device info
    # ---------------------------------------------------------
    def _build_device_info(
        self, cpf: str, features: Dict[str, Any], now: datetime
    ) -> DeviceInfo:
        device_name = features.get("device_name") or features.get("device_name_normalized") or ""
        app_version = features.get("app_version")
        device_model = str(device_name).strip() if device_name else "Desconhecido"
        device_type = self._infer_device_type(device_model)

        # Gerar device_id determinístico
        device_id_seed = f"{cpf}|{device_model}|{app_version or ''}"
        device_id = hashlib.sha256(device_id_seed.encode()).hexdigest()[:16]

        # Verificar se é conhecido
        is_known = self._profile_mgr.is_device_known(cpf, device_id)

        # Atualizar histórico global de devices
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
    def _build_session_metrics(self, features: Dict[str, Any]) -> SessionMetrics:
        # Tempo de interação
        tempo_raw = features.get("tempo_interacao_ms") or features.get("tempo_interacao_ms_final")
        tempo_ms = self._safe_float(tempo_raw)

        # Latência
        latencia_raw = features.get("latencia_rede_ms") or features.get("latencia_rede_ms_final")
        latencia_ms = self._safe_float(latencia_raw)

        # Método de login
        metodo_raw = features.get("metodo_autenticacao")
        metodo = self._normalize_login_method(metodo_raw)

        # Agendamento recorrente
        is_recorrente_raw = features.get("is_agendamento_recorrente", "")
        is_recorrente = str(is_recorrente_raw).strip().lower() == "true"

        # Estimativa de duração da sessão
        duration_s = int(tempo_ms / 1000) if tempo_ms and tempo_ms > 0 else None

        return SessionMetrics(
            tempo_interacao_ms=tempo_ms,
            latencia_rede_ms=latencia_ms,
            metodo_login=metodo,
            is_agendamento_recorrente=is_recorrente,
            duration_estimate_seconds=duration_s,
        )

    # ---------------------------------------------------------
    # PRIVATE: Extract features from pipeline dict
    # ---------------------------------------------------------
    def _extract_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Extrai e normaliza features relevantes do dict do pipeline."""
        return {
            "vl_pix": self._safe_float(features.get("vl_pix"), 0.0),
            "nr_idade": self._safe_int(features.get("nr_idade"), 0),
            "qt_tempo_relacionamento_mes": self._safe_int(
                features.get("qt_tempo_relacionamento_mes"), 999
            ),
            "is_segmento_premium": bool(
                self._safe_int(features.get("is_segmento_premium_flag"), 0)
            ),
            "is_viuvo": bool(self._safe_int(features.get("is_viuvo_flag"), 0)),
            "perfil_vulneravel_se": bool(
                self._safe_int(features.get("perfil_vulneravel_se_flag"), 0)
            ),
            "vl_renda": self._safe_float(features.get("vl_renda_cliente"), 0.0),
            "ratio_pix_renda": self._safe_float(features.get("ratio_pix_renda")),
            "pix_over_100pct_renda": bool(
                self._safe_int(features.get("pix_over_100pct_renda_flag"), 0)
            ),
            "first_receiver_flag": bool(
                self._safe_int(features.get("first_receiver_flag"), 0)
            ),
            "burst_30m_flag": bool(
                self._safe_int(features.get("burst_30m_flag"), 0)
            ),
            "tx_count_prev_30m": self._safe_int(features.get("tx_count_prev_30m"), 0),
            "ratio_tempo_interacao": self._safe_float(
                features.get("ratio_tempo_interacao_cliente")
            ),
        }

    # ---------------------------------------------------------
    # STATIC HELPERS
    # ---------------------------------------------------------
    @staticmethod
    def _infer_device_type(device_model: str) -> str:
        model = device_model.lower()
        if any(kw in model for kw in ("iphone", "ipad", "ios", "apple")):
            return "iOS"
        if any(kw in model for kw in ("samsung", "xiaomi", "motorola", "pixel", "oneplus", "huawei", "lg", "galaxy", "redmi")):
            return "Android"
        return "Desconhecido"

    @staticmethod
    def _normalize_login_method(raw: Any) -> str:
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
    def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
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

        summary = f"Risco comportamental {level} ({result.behavioral_score:.0f}/100): "
        summary += f"{count} fator(es) — {top_factors}"

        if result.fatores_atenuantes:
            summary += f" | Atenuantes: {len(result.fatores_atenuantes)}"

        return summary
