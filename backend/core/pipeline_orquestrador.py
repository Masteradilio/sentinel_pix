"""
pipeline_orquestrador.py v1.4 — Orquestrador de Inferência PIX Antifraude

Mudanças v1.3 → v1.4:
  1. Integração com Decision Engine v3.0.5 (SE + BEH na Fase 7 e vetos)
  2. Integração com Behavioral Analytics v3.1 (6 fatores leakage-free validated)
  3. Integração com Social Engineering v3.4 (8 padrões calibrados)
  4. _build_response: inclui precision/origin nos risk_factors do behavioral
  5. _build_response: pipeline_version 1.3, engine_version 2.2
  6. get_status: reporta versões de SE v3.4, BEH v3.1, Engine v3.0.5
  7. Removido DEBUG print temporário
  8. Diagrama de fluxo atualizado

Mudanças v1.1 → v1.2:
  1. SHAP TreeExplainer para explicabilidade por transação (top-10 features)
  2. _build_response otimizado: remove redundâncias, omite blocos vazios
  3. score_raw removido (duplicata de componentes.lgbm_raw)
  4. faixas e peso_maximo movidos para metadata (são constantes)
  5. cascade e atenuantes omitidos quando vazios
  6. Compatível com DecisionResult v2.1

Mudanças v1.0 → v1.1:
  1. _build_response: inclui cascade_triggered, cascade_rules, if_boost_applied
  2. _print_result: exibe cascade rules e IF boost no output
  3. get_status: engine_version 2.1, cascade info, IF ensemble params
  4. _create_features: cria qt_envio_recebedor_trimestre (feature LGBM)
  5. _create_sequential_features: cria qt_envio_recebedor_trimestre do cache
  6. Compatível com DecisionResult v2.1 (if_boost_applied, cascade_*)

Ponto de entrada único para inferência em tempo real e batch.
Coordena todas as camadas sem duplicar lógica:

  Dados Brutos (dict/DataFrame)
       │
       ▼
  ┌─────────────────────────────────┐
  │  1. Feature Engineering         │  ← preprocessing.py (PixPreprocessor)
  │     Limpeza + derivadas + flags │
  └──────────────┬──────────────────┘
                 │ features_dict
       ┌─────────┼─────────┐
       ▼         ▼         ▼
  ┌─────────┐ ┌────────┐ ┌────────┐
  │ Decision│ │ Social │ │Behav.  │  ← 3 engines independentes
  │ Engine  │ │ Eng.   │ │Analyt. │
  │ v2.2    │ │ v3.3   │ │ v3.0   │
  └────┬────┘ └───┬────┘ └───┬────┘
       │          │          │
       └──────────┼──────────┘
                  ▼
  ┌─────────────────────────────────┐
  │  2. Consolidação                │
  │     LGBM + Cascade + IF Boost   │
  │     SE patterns (granular)      │  ← v2.2: Fase 7 reescrita
  │     BEH factors (by category)   │  ← v2.2: velocity/dormancy/profile
  │     Vetos SE/BEH                │  ← v2.2: novos vetos
  │     Score final + Decisão       │
  │     SHAP Explicabilidade        │
  └─────────────────────────────────┘

Uso:
    from pipeline_orquestrador import PipelineOrquestrador

    pipeline = PipelineOrquestrador()

    # Transação individual (tempo real)
    resultado = pipeline.analisar(dados_transacao_dict)

    # Lote (batch)
    resultados = pipeline.analisar_batch(lista_de_dicts)

    # Health check
    status = pipeline.get_status()
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# =========================================================
# IMPORTS DO PROJETO
# =========================================================

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent.parent

# Detectar raiz do projeto
if (SCRIPT_DIR / "backend").exists():
    PROJECT_ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "backend").exists():
    PROJECT_ROOT = SCRIPT_DIR.parent
elif SCRIPT_DIR.name == "backend":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

BACKEND_DIR = PROJECT_ROOT / "backend"
ARTEFATOS_DIR = BACKEND_DIR / "artefatos"

# Garantir imports
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Preprocessing (feature engineering)
from preprocessing import (
    PixPreprocessor,
    standardize_columns,
    ensure_column,
    normalize_text_value,
    clean_text_columns,
    safe_to_numeric,
    safe_to_datetime,
    replace_sentinels_with_nan,
    replace_zero_with_nan,
    normalize_transaction_key,
    normalize_device_name,
    extract_app_version_minor,
    classify_key_flags,
    robust_divide,
    map_topaz_rule,
)

# Engines
from core.decision_engine import PixDecisionEngine, EngineConfig, DecisionResult
from core.behavioral_analytics import BehavioralAnalytics, BehavioralAnalysisResult
from core.social_engineering import SocialEngineeringDetector, SEAnalysisResult

# SHAP — explicabilidade v1.2+
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap não instalado — explicabilidade SHAP desabilitada")


# =========================================================
# VERSÃO DO PIPELINE
# =========================================================
PIPELINE_VERSION = "1.4"


# =========================================================
# CONFIGURAÇÃO DE COLUNAS (única fonte de verdade)
# =========================================================

RAW_INPUT_COLUMNS = [
    "cd_pix", "dt_pix", "cd_cpf_pagador", "cd_cpf_cnpj_recebedor",
    "ds_chave_pix", "ds_tipo_chave", "vl_pix",
    "qt_total_pix_trimestre", "vl_mediana_pix_trimestre",
    "vl_desvio_padrao_pix_trimestre", "qt_intervalo_transacao_minuto",
    "qt_intervalo_mediana_trimestre", "qt_intervalo_desvio_padrao_trimestre",
    "qt_pix_dia_maximo_trimestre",
    "device_name", "app_version", "ip_address",
    "latencia_rede_ms", "vl_latencia_rede_media_trimestre",
    "tempo_interacao_ms", "vl_tempo_interacao_medio_trimestre",
    "tempo_processamento_host_ms", "metodo_autenticacao", "session_id",
    "cd_retorno", "topaz_risk_score", "topaz_transacao_rejeitada",
    "is_agendamento_recorrente",
    "qt_aparelhos_distintos_trimestre", "nr_idade", "qt_tempo_relacionamento_mes",
    # v2.1b — novos campos do Big Data
    "vl_renda_cliente", "ds_sexo", "ds_estado_civil",
    "ds_segmento", "qt_dependentes",
]

NUMERIC_COLUMNS = [
    "vl_pix", "qt_total_pix_trimestre", "vl_mediana_pix_trimestre",
    "vl_desvio_padrao_pix_trimestre", "qt_intervalo_transacao_minuto",
    "qt_intervalo_mediana_trimestre", "qt_intervalo_desvio_padrao_trimestre",
    "qt_pix_dia_maximo_trimestre", "latencia_rede_ms",
    "vl_latencia_rede_media_trimestre", "tempo_interacao_ms",
    "vl_tempo_interacao_medio_trimestre", "tempo_processamento_host_ms",
    "topaz_risk_score", "topaz_transacao_rejeitada",
    "qt_aparelhos_distintos_trimestre",
    "nr_idade", "qt_tempo_relacionamento_mes",
    "vl_renda_cliente", "qt_dependentes",
]

TEXT_COLUMNS = [
    "cd_pix", "cd_cpf_pagador", "cd_cpf_cnpj_recebedor",
    "ds_chave_pix", "ds_tipo_chave", "device_name", "app_version",
    "ip_address", "metodo_autenticacao", "session_id", "cd_retorno",
    "is_agendamento_recorrente",
    "ds_sexo", "ds_estado_civil", "ds_segmento",
]

SENTINEL_COLUMNS = [
    "latencia_rede_ms", "vl_latencia_rede_media_trimestre",
    "tempo_interacao_ms", "vl_tempo_interacao_medio_trimestre",
    "tempo_processamento_host_ms", "topaz_risk_score",
    "topaz_transacao_rejeitada",
]

# Features LGBM que têm tradução humana para SHAP
SHAP_FEATURE_LABELS = {
    "vl_pix": "Valor do PIX (R$)",
    "log_vl_pix": "Log do valor do PIX",
    "ratio_valor_mediana": "Razão valor / mediana trimestral",
    "diff_valor_mediana": "Diferença valor − mediana (R$)",
    "ratio_valor_desvio_padrao": "Razão valor / desvio padrão",
    "zscore_valor_aprox": "Z-score aproximado do valor",
    "nr_idade": "Idade do cliente (anos)",
    "qt_tempo_relacionamento_mes": "Tempo de relacionamento (meses)",
    "qt_total_pix_trimestre": "Total de PIX no trimestre",
    "vl_mediana_pix_trimestre": "Mediana PIX trimestral (R$)",
    "vl_desvio_padrao_pix_trimestre": "Desvio padrão PIX trimestral",
    "qt_intervalo_transacao_minuto": "Intervalo desde última tx (min)",
    "qt_intervalo_mediana_trimestre": "Mediana do intervalo trimestral (min)",
    "qt_intervalo_desvio_padrao_trimestre": "Desvio padrão do intervalo",
    "ratio_intervalo_vs_mediana": "Razão intervalo / mediana intervalo",
    "diff_intervalo_vs_mediana": "Diferença intervalo − mediana (min)",
    "zscore_intervalo_aprox": "Z-score do intervalo",
    "qt_pix_dia_maximo_trimestre": "Máx PIX/dia no trimestre",
    "latencia_rede_ms_final": "Latência de rede (ms)",
    "vl_latencia_rede_media_trimestre": "Latência média trimestral (ms)",
    "ratio_latencia_cliente": "Razão latência / média cliente",
    "diff_latencia_cliente": "Diferença latência − média (ms)",
    "latencia_host_ratio": "Razão latência rede / host",
    "tempo_interacao_ms_final": "Tempo de interação (ms)",
    "vl_tempo_interacao_medio_trimestre": "Tempo interação médio trimestral",
    "ratio_tempo_interacao_cliente": "Razão tempo interação / média",
    "tempo_processamento_host_ms": "Tempo processamento host (ms)",
    "topaz_score_filled": "Score Topaz (0-5)",
    "topaz_rejeitada_flag": "Topaz rejeitou a transação",
    "pix_key_random_flag": "Chave PIX aleatória",
    "pix_key_email_flag": "Chave PIX é email",
    "pix_key_document_flag": "Chave PIX é documento/telefone",
    "receiver_document_same_as_customer_flag": "Recebedor = Pagador (mesmo CPF)",
    "metodo_auth_encoded": "Método autenticação (1=bio, 2=senha, 3=pin)",
    "is_agendamento_recorrente_flag": "É agendamento recorrente",
    "hour": "Hora da transação",
    "day_of_week": "Dia da semana (0=seg)",
    "is_business_hours": "Horário comercial (8h-18h)",
    "device_missing_flag": "Device não informado",
    "app_version_missing_flag": "Versão app não informada",
    "auth_method_missing_flag": "Método auth não informado",
    "topaz_missing_flag": "Score Topaz ausente",
    "host_time_missing_flag": "Tempo host ausente",
    "latencia_missing_flag": "Latência ausente",
    "renda_missing_flag": "Renda não informada",
    "tempo_interacao_missing_flag": "Tempo interação ausente",
    "app_version_minor": "Versão minor do app",
    "vl_pix_over_1000_flag": "Valor ≥ R$1.000",
    "is_first_tx_trimestre": "Primeira tx do trimestre",
    "qt_aparelhos_distintos_trimestre": "Aparelhos distintos no trimestre",
    "minutes_since_prev_tx": "Minutos desde tx anterior",
    "tx_count_prev_30m": "Transações nos últimos 30min",
    "receiver_tx_count_prev": "Envios anteriores p/ este recebedor",
    "qt_envio_recebedor_trimestre": "Envios p/ recebedor no trimestre",
    "first_receiver_flag": "Primeiro envio p/ este recebedor",
    "key_tx_count_prev": "Usos anteriores desta chave",
    "first_key_flag": "Primeira vez usando esta chave",
    "distinct_receivers_so_far": "Recebedores distintos até agora",
    "distinct_keys_so_far": "Chaves distintas até agora",
    "tp_primeiro_envio_recebedor_trimestre": "Primeiro envio ao recebedor (trimestre)",
    "burst_30m_flag": "Burst: ≥1 tx nos últimos 30min",
    "rule_age_score": "Score regra: idade",
    "rule_relationship_score": "Score regra: relacionamento",
    "rule_mule_account_score": "Score regra: conta laranja",
    "rule_random_key_score": "Score regra: chave aleatória",
    "rule_velocity_score": "Score regra: velocidade",
    "rule_topaz_score": "Score regra: Topaz",
    "rule_score_raw": "Score total de regras (soma)",
    "rule_score_normalized": "Score de regras normalizado",
    "is_sexo_feminino_flag": "Sexo feminino",
    "is_viuvo_flag": "Estado civil: viúvo(a)",
    "is_segmento_premium_flag": "Segmento premium/exclusivo",
    "qt_dependentes": "Quantidade de dependentes",
    "ratio_pix_renda": "Razão PIX / renda mensal",
    "pix_over_50pct_renda_flag": "PIX > 50% da renda",
    "pix_over_100pct_renda_flag": "PIX > 100% da renda",
    "perfil_vulneravel_se_flag": "Perfil vulnerável (viúvo+idoso+s/ dependentes)",
    "vl_renda_cliente": "Renda mensal do cliente (R$)",
}


# =========================================================
# ORQUESTRADOR v1.3
# =========================================================
class PipelineOrquestrador:
    """
    Orquestrador de inferência para detecção de fraude PIX v1.3.

    Responsabilidades:
        1. Feature engineering (usa preprocessing.py — sem duplicação)
        2. Coordenar os 3 engines:
           - Decision Engine v3.0.5 (LGBM + IF + Cascade + Agravantes + Vetos)
           - Social Engineering Detector v3.3 (8 padrões calibrados)
           - Behavioral Analytics v3.1 (6 fatores leakage-free validated)
        3. SHAP explicabilidade por transação
        4. Consolidar resposta final padronizada
        5. Manter cache de histórico por cliente (features sequenciais)

    O que NÃO faz:
        - Servir HTTP (→ api.py)
        - Treinar modelos (→ notebooks/scripts de treino)
        - Carregar dados em massa (→ scripts de ingestão)
    """

    def __init__(
        self,
        artefatos_dir: Optional[str] = None,
        engine_config: Optional[EngineConfig] = None,
        shap_enabled: bool = True,
        shap_top_n: int = 10,
    ):
        self.artefatos_dir = Path(artefatos_dir) if artefatos_dir else ARTEFATOS_DIR

        t0 = time.perf_counter()

        # --- 1. Preprocessor ---
        self.preprocessor: Optional[PixPreprocessor] = None
        self._load_preprocessor()

        # --- 2. Decision Engine v3.0.5 ---
        config = engine_config or EngineConfig(artefatos_dir=str(self.artefatos_dir))
        self.engine = PixDecisionEngine(config)

        # --- 3. Social Engineering Detector v3.3 ---
        self.se_detector = SocialEngineeringDetector(
            pattern_config={
                "se_pattern_residual_enabled": config.se_pattern_residual_enabled,
                "se_pattern_residual_age_young_max": config.se_pattern_residual_age_young_max,
                "se_pattern_residual_age_old_min": config.se_pattern_residual_age_old_min,
                "se_pattern_residual_value_min": config.se_pattern_residual_value_min,
                "se_pattern_residual_value_max": config.se_pattern_residual_value_max,
                "se_pattern_residual_rel_max": config.se_pattern_residual_rel_max,
                "se_pattern_residual_if_min": config.se_pattern_residual_if_min,
            }
        )

        # --- 4. Behavioral Analytics v3.1 ---
        self.behavioral = BehavioralAnalytics()

        # --- 5. SHAP Explainer ---
        self.shap_enabled = shap_enabled and SHAP_AVAILABLE
        self.shap_top_n = shap_top_n
        self._shap_explainer = None
        if self.shap_enabled and self.engine.lgbm_model is not None:
            try:
                self._shap_explainer = shap.TreeExplainer(self.engine.lgbm_model)
                logger.info("SHAP TreeExplainer inicializado com sucesso")
            except Exception as e:
                logger.warning(f"Falha ao inicializar SHAP: {e}")
                self._shap_explainer = None
                self.shap_enabled = False

        # --- 6. Cache de histórico por cliente ---
        self._customer_history: Dict[str, Dict[str, Any]] = {}

        # Status
        self._load_time_ms = (time.perf_counter() - t0) * 1000
        self.available = self.engine.available

        # Versões dos módulos
        self._engine_version = getattr(self.engine, "ENGINE_VERSION", "2.2")
        self._behavioral_version = getattr(self.behavioral, "VERSION", "3.0")
        self._se_version = getattr(self.se_detector, "VERSION", "3.4")

        logger.info(
            f"PipelineOrquestrador v{PIPELINE_VERSION} inicializado em "
            f"{self._load_time_ms:.0f}ms | "
            f"Engine v{self._engine_version}={'OK' if self.engine.available else 'DEGRADED'} | "
            f"Preprocessor={'OK' if self.preprocessor else 'PASSTHROUGH'} | "
            f"Cascade={'ON' if self.engine.config.cascade_enabled else 'OFF'} | "
            f"IF={'OK' if self.engine.if_model else 'OFF'} | "
            f"SHAP={'OK' if self._shap_explainer else 'OFF'} | "
            f"SE v{self._se_version}=OK | "
            f"BEH v{self._behavioral_version}=OK"
        )

    # ==========================================================
    # LOADING
    # ==========================================================
    def _load_preprocessor(self):
        """Carrega o PixPreprocessor treinado."""
        path = self.artefatos_dir / "preprocessing.joblib"
        if path.exists():
            try:
                from preprocessing import PixPreprocessor
                import __main__
                if not hasattr(__main__, 'PixPreprocessor'):
                    __main__.PixPreprocessor = PixPreprocessor
                import core.preprocessing as _prep_mod
                sys.modules['preprocessing'] = _prep_mod

                self.preprocessor = joblib.load(ARTEFATOS_DIR / "preprocessing.joblib")
                logger.info("Preprocessor carregado com sucesso")
            except Exception as e:
                logger.warning(f"Erro ao carregar preprocessor: {e} — usando passthrough")
                self.preprocessor = None
        else:
            logger.warning(f"Preprocessor não encontrado em {path} — usando passthrough")

    # ==========================================================
    # SHAP EXPLICABILIDADE
    # ==========================================================
    def _compute_shap_explanation(
        self,
        X_row: pd.DataFrame,
        decisao: str,
        top_n: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Calcula SHAP values para uma transação individual.

        Usa TreeExplainer (otimizado para LightGBM, ~2-5ms por row).
        Só é calculado para decisões CONFIRMAR e BLOQUEAR.
        """
        if decisao == "APROVAR":
            return None

        if self._shap_explainer is None:
            return None

        if top_n is None:
            top_n = self.shap_top_n

        try:
            shap_values = self._shap_explainer.shap_values(X_row)

            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0]

            base_value = self._shap_explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(base_value[1])

            feature_names = X_row.columns.tolist()
            indices = np.argsort(np.abs(sv))[::-1][:top_n]

            top_features = []
            total_abs_shap = float(np.abs(sv).sum())

            for i in indices:
                feat_name = feature_names[i]
                shap_val = float(sv[i])
                feat_value = X_row.iloc[0, i]

                if isinstance(feat_value, (np.integer,)):
                    feat_value = int(feat_value)
                elif isinstance(feat_value, (np.floating,)):
                    feat_value = round(float(feat_value), 4) if not np.isnan(feat_value) else None
                elif pd.isna(feat_value):
                    feat_value = None

                direction = "aumenta_risco" if shap_val > 0 else "diminui_risco"
                human_label = SHAP_FEATURE_LABELS.get(feat_name, feat_name)
                impact_pct = round(
                    abs(shap_val) / max(total_abs_shap, 1e-9) * 100, 1
                )

                top_features.append({
                    "feature": feat_name,
                    "label": human_label,
                    "shap_value": round(shap_val, 6),
                    "feature_value": feat_value,
                    "direction": direction,
                    "impact_pct": impact_pct,
                })

            return {
                "method": "TreeSHAP",
                "base_value": round(float(base_value), 6),
                "top_features": top_features,
                "sum_shap": round(float(sv.sum()), 6),
                "prediction_logodds": round(float(base_value + sv.sum()), 6),
                "note": (
                    "SHAP values indicam a contribuição de cada feature "
                    "para a probabilidade de fraude (log-odds). "
                    "Valores positivos aumentam o risco, negativos diminuem."
                ),
            }

        except Exception as e:
            logger.warning(f"Erro ao calcular SHAP: {e}")
            return {"error": str(e), "top_features": []}

    # ==========================================================
    # FEATURE ENGINEERING (fonte única — sem duplicação)
    # ==========================================================
    def _prepare_raw(self, data: Union[Dict, pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Converte input bruto em DataFrame padronizado de 1 linha."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, pd.Series):
            df = data.to_frame().T.reset_index(drop=True)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            raise ValueError(f"Tipo de entrada não suportado: {type(data)}")

        df = standardize_columns(df)
        for col in RAW_INPUT_COLUMNS:
            df = ensure_column(df, col)
        return df

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria TODAS as features para inferência.

        Fonte única de verdade para feature engineering em RT.
        Usa funções do preprocessing.py (sem duplicar lógica).
        """
        df = df.copy()

        # ─── Limpeza ────────────────────────────────────────
        for col in TEXT_COLUMNS:
            if col in df.columns and df[col].dtype != object:
                df[col] = df[col].apply(lambda v: str(v) if pd.notna(v) else None)
        df = clean_text_columns(df, TEXT_COLUMNS)
        df = safe_to_numeric(df, NUMERIC_COLUMNS)
        df = safe_to_datetime(df, ["dt_pix"])
        df = replace_sentinels_with_nan(df, SENTINEL_COLUMNS)
        df = replace_zero_with_nan(df, ["vl_tempo_interacao_medio_trimestre"])

        # ─── IDs ────────────────────────────────────────────
        df["transaction_id"] = normalize_transaction_key(df["cd_pix"])
        df["customer_id"] = df["cd_cpf_pagador"]
        df["event_datetime"] = df["dt_pix"]

        # ─── Missing Flags ──────────────────────────────────
        df["device_missing_flag"] = df["device_name"].isna().astype(int)
        df["app_version_missing_flag"] = df["app_version"].isna().astype(int)
        df["auth_method_missing_flag"] = df["metodo_autenticacao"].isna().astype(int)
        df["topaz_missing_flag"] = df["topaz_risk_score"].isna().astype(int)
        df["host_time_missing_flag"] = df["tempo_processamento_host_ms"].isna().astype(int)
        df["latencia_missing_flag"] = df["latencia_rede_ms"].isna().astype(int)
        df["renda_missing_flag"] = df["vl_renda_cliente"].isna().astype(int)
        df["tempo_interacao_missing_flag"] = df["tempo_interacao_ms"].isna().astype(int)

        # ─── Latência final ─────────────────────────────────
        df["latencia_rede_ms_final"] = df["latencia_rede_ms"]
        df["tempo_interacao_ms_final"] = df["tempo_interacao_ms"]

        # ─── Device / App ───────────────────────────────────
        df["device_name_normalized"] = df["device_name"].apply(normalize_device_name)
        df["app_version_minor"] = df["app_version"].apply(extract_app_version_minor)

        # ─── Key Flags ──────────────────────────────────────
        ds_tipo = df["ds_tipo_chave"].apply(normalize_text_value).fillna("Informação ausente")
        key_flags = classify_key_flags(ds_tipo)
        for c in key_flags.columns:
            df[c] = key_flags[c].values

        # ─── Receiver Flags ─────────────────────────────────
        df["receiver_document_same_as_customer_flag"] = (
            df["customer_id"].notna()
            & df["cd_cpf_cnpj_recebedor"].notna()
            & (df["customer_id"].astype(str) == df["cd_cpf_cnpj_recebedor"].astype(str))
        ).astype(int)

        # ─── Temporal ───────────────────────────────────────
        df["hour"] = df["event_datetime"].dt.hour.fillna(12).astype(int)
        df["day_of_week"] = df["event_datetime"].dt.dayofweek.fillna(0).astype(int)
        df["is_business_hours"] = df["hour"].between(8, 18, inclusive="both").astype(int)

        # ─── Core Derived ───────────────────────────────────
        df["log_vl_pix"] = np.log1p(df["vl_pix"].clip(lower=0))
        df["ratio_valor_mediana"] = robust_divide(df["vl_pix"], df["vl_mediana_pix_trimestre"])
        df["diff_valor_mediana"] = df["vl_pix"] - df["vl_mediana_pix_trimestre"]
        df["ratio_valor_desvio_padrao"] = robust_divide(
            df["vl_pix"], df["vl_desvio_padrao_pix_trimestre"]
        )
        df["zscore_valor_aprox"] = robust_divide(
            df["vl_pix"] - df["vl_mediana_pix_trimestre"],
            df["vl_desvio_padrao_pix_trimestre"],
        )

        df["ratio_intervalo_vs_mediana"] = robust_divide(
            df["qt_intervalo_transacao_minuto"], df["qt_intervalo_mediana_trimestre"]
        )
        df["diff_intervalo_vs_mediana"] = (
            df["qt_intervalo_transacao_minuto"] - df["qt_intervalo_mediana_trimestre"]
        )
        df["zscore_intervalo_aprox"] = robust_divide(
            df["qt_intervalo_transacao_minuto"] - df["qt_intervalo_mediana_trimestre"],
            df["qt_intervalo_desvio_padrao_trimestre"],
        )

        df["ratio_latencia_cliente"] = robust_divide(
            df["latencia_rede_ms_final"], df["vl_latencia_rede_media_trimestre"]
        )
        df["diff_latencia_cliente"] = (
            df["latencia_rede_ms_final"] - df["vl_latencia_rede_media_trimestre"]
        )
        df["latencia_host_ratio"] = robust_divide(
            df["latencia_rede_ms_final"], df["tempo_processamento_host_ms"]
        )

        # ─── Tempo de interação ─────────────────────────────
        df["ratio_tempo_interacao_cliente"] = robust_divide(
            df["tempo_interacao_ms_final"], df["vl_tempo_interacao_medio_trimestre"]
        )

        # ─── Topaz ──────────────────────────────────────────
        df["topaz_score_filled"] = df["topaz_risk_score"].fillna(0)
        df["topaz_rejeitada_flag"] = df["topaz_transacao_rejeitada"].fillna(0).astype(int)

        # ─── Autenticação ───────────────────────────────────
        auth_map = {"biometria": 1, "senha": 2, "pin": 3}
        _auth = pd.Series(
            [str(v).strip().lower() if pd.notna(v) else "" for v in df["metodo_autenticacao"]],
            index=df.index,
        )
        df["metodo_auth_encoded"] = _auth.map(auth_map).fillna(0).astype(int)

        # ─── Login Flags ────────────────────────────────────
        df["is_login_biometria_flag"] = (_auth == "biometria").astype(int)
        df["is_login_senha_flag"] = (_auth.isin(["senha", "pin"])).astype(int)

        # ─── Agendamento ────────────────────────────────────
        _agend = df["is_agendamento_recorrente"].fillna("false")
        _agend = pd.Series([str(v).strip().lower() for v in _agend], index=df.index)
        df["is_agendamento_recorrente_flag"] = _agend.isin(
            ["true", "1", "1.0", "sim", "yes"]
        ).astype(int)

        # ─── Flags v3 ──────────────────────────────────────
        df["vl_pix_over_1000_flag"] = (df["vl_pix"] >= 1000).astype(int)
        df["is_first_tx_trimestre"] = (df["qt_total_pix_trimestre"] == 1).astype(int)

        # ─── v2.1b: Renda e Perfil ─────────────────────────
        df = self._create_profile_features(df)

        # ─── Features Sequenciais (cache por cliente) ───────
        df = self._create_sequential_features(df)

        # ─── Rule Scores ────────────────────────────────────
        df = self._create_rule_scores(df)

        return df

    def _create_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features de perfil do cliente (dados v2.1b do Big Data)."""
        sexo_upper = df["ds_sexo"].fillna("").astype(str).str.upper()
        df["is_sexo_feminino_flag"] = sexo_upper.isin(["F", "FEMININO", "FEMALE"]).astype(int)

        estado_civil_upper = df["ds_estado_civil"].fillna("").astype(str).str.upper()
        df["is_viuvo_flag"] = estado_civil_upper.str.contains("VIUV", na=False).astype(int)

        segmento_upper = df["ds_segmento"].fillna("").astype(str).str.upper()
        df["is_segmento_premium_flag"] = segmento_upper.isin(
            ["EXCLUSIVO", "PRIVATE", "MILLENIUM", "PREMIUM", "VIP"]
        ).astype(int)

        df["qt_dependentes"] = pd.to_numeric(df["qt_dependentes"], errors="coerce").fillna(0)

        renda = pd.to_numeric(df["vl_renda_cliente"], errors="coerce")
        vl_pix = df["vl_pix"]

        df["ratio_pix_renda"] = np.where(
            renda.notna() & (renda > 0), vl_pix / renda, np.nan,
        )
        df["pix_over_50pct_renda_flag"] = np.where(
            renda.notna() & (renda > 0), (vl_pix > renda * 0.5).astype(int), 0,
        )
        df["pix_over_100pct_renda_flag"] = np.where(
            renda.notna() & (renda > 0), (vl_pix > renda).astype(int), 0,
        )

        df["perfil_vulneravel_se_flag"] = (
            (df["is_viuvo_flag"] == 1)
            & (df["nr_idade"] >= 65)
            & (df["qt_dependentes"] == 0)
        ).astype(int)

        return df

    def _create_sequential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features sequenciais usando cache de histórico por cliente.

        v1.4.1: Correção de bug com df.loc em DataFrames de 1 linha.
        Usa .at[] para atribuição escalar (mais estável que .loc em pandas >= 2.0).
        """
        # Inicializar colunas com valores default (vetorizado, sem loop)
        default_values = {
            "minutes_since_prev_tx": np.nan,
            "tx_count_prev_30m": 0,
            "receiver_tx_count_prev": 0,
            "qt_envio_recebedor_trimestre": 0,
            "first_receiver_flag": 1,
            "key_tx_count_prev": 0,
            "first_key_flag": 1,
            "distinct_receivers_so_far": 1,
            "distinct_keys_so_far": 1,
            "tp_primeiro_envio_recebedor_trimestre": 1,
        }
        for col, default in default_values.items():
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)

        # Loop por linha para consultar histórico
        for idx in df.index:
            customer_id = str(df.at[idx, "customer_id"])
            event_time = df.at[idx, "event_datetime"]
            receiver = str(df.at[idx, "cd_cpf_cnpj_recebedor"])
            pix_key = str(df.at[idx, "ds_chave_pix"])

            hist = self._customer_history.get(customer_id)

            if hist is None or not pd.notna(event_time):
                # Sem histórico ou datetime inválido → mantém defaults
                continue

            # --- minutes_since_prev_tx ---
            last_time = hist.get("last_event_time")
            if last_time is not None and pd.notna(last_time):
                diff_min = (event_time - last_time).total_seconds() / 60.0
                df.at[idx, "minutes_since_prev_tx"] = max(diff_min, 0)

            # --- tx_count_prev_30m ---
            recent_times = hist.get("recent_times", [])
            count_30m = sum(
                1 for t in recent_times
                if pd.notna(t)
                and (event_time - t).total_seconds() / 60.0 <= 30
            )
            df.at[idx, "tx_count_prev_30m"] = count_30m

            # --- Receiver counts ---
            receiver_counts = hist.get("receiver_counts", {})
            rcv_count = receiver_counts.get(receiver, 0)
            df.at[idx, "receiver_tx_count_prev"] = rcv_count
            df.at[idx, "qt_envio_recebedor_trimestre"] = rcv_count
            df.at[idx, "first_receiver_flag"] = 1 if rcv_count == 0 else 0
            df.at[idx, "tp_primeiro_envio_recebedor_trimestre"] = (
                1 if rcv_count == 0 else 0
            )

            # --- Key counts ---
            key_counts = hist.get("key_counts", {})
            key_count = key_counts.get(pix_key, 0)
            df.at[idx, "key_tx_count_prev"] = key_count
            df.at[idx, "first_key_flag"] = 1 if key_count == 0 else 0

            # --- Distinct counts ---
            df.at[idx, "distinct_receivers_so_far"] = len(receiver_counts) + (
                1 if receiver not in receiver_counts else 0
            )
            df.at[idx, "distinct_keys_so_far"] = len(key_counts) + (
                1 if pix_key not in key_counts else 0
            )

        # burst_30m_flag é derivado de tx_count_prev_30m (vetorizado)
        df["burst_30m_flag"] = (df["tx_count_prev_30m"] >= 1).astype(int)

        return df


    def _update_customer_history(self, df: pd.DataFrame):
        """Atualiza o cache de histórico após inferência bem-sucedida."""
        for idx in df.index:
            customer_id = str(df.loc[idx, "customer_id"])
            event_time = df.loc[idx, "event_datetime"]
            receiver = str(df.loc[idx, "cd_cpf_cnpj_recebedor"])
            pix_key = str(df.loc[idx, "ds_chave_pix"])

            if customer_id in ("nan", "None", ""):
                continue

            if customer_id not in self._customer_history:
                self._customer_history[customer_id] = {
                    "last_event_time": None,
                    "recent_times": [],
                    "receiver_counts": {},
                    "key_counts": {},
                }

            hist = self._customer_history[customer_id]
            hist["last_event_time"] = event_time

            if pd.notna(event_time):
                hist["recent_times"].append(event_time)
                cutoff = event_time - pd.Timedelta(minutes=60)
                hist["recent_times"] = [
                    t for t in hist["recent_times"] if pd.notna(t) and t >= cutoff
                ]

            if receiver not in ("nan", "None", ""):
                hist["receiver_counts"][receiver] = (
                    hist["receiver_counts"].get(receiver, 0) + 1
                )

            if pix_key not in ("nan", "None", ""):
                hist["key_counts"][pix_key] = (
                    hist["key_counts"].get(pix_key, 0) + 1
                )

    def _create_rule_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria rule scores do pipeline (usados como features pelo LGBM)."""
        df["rule_age_score"] = np.select(
            [
                df["nr_idade"].between(60, 65),
                df["nr_idade"].between(66, 75),
                df["nr_idade"] >= 76,
            ],
            [1, 2, 3], default=0,
        )

        df["rule_relationship_score"] = np.select(
            [
                df["qt_tempo_relacionamento_mes"].between(61, 90),
                df["qt_tempo_relacionamento_mes"].between(31, 60),
                df["qt_tempo_relacionamento_mes"].between(0, 30),
            ],
            [1, 2, 3], default=0,
        )

        df["rule_mule_account_score"] = np.select(
            [
                df["first_receiver_flag"] == 1,
                df["receiver_document_same_as_customer_flag"] == 1,
            ],
            [2, 1], default=0,
        )

        df["rule_random_key_score"] = np.select(
            [
                df["pix_key_random_flag"] == 1,
                df["pix_key_email_flag"] == 1,
                df["pix_key_document_flag"] == 1,
            ],
            [2, 1, 0], default=0,
        )

        df["rule_velocity_score"] = np.select(
            [
                df["tx_count_prev_30m"] == 0,
                df["tx_count_prev_30m"] == 1,
                df["tx_count_prev_30m"] == 2,
                df["tx_count_prev_30m"] >= 3,
            ],
            [0, 2, 3, 4], default=0,
        )

        df["rule_topaz_score"] = df["topaz_risk_score"].apply(map_topaz_rule)

        rule_cols = [
            "rule_age_score", "rule_relationship_score",
            "rule_mule_account_score", "rule_random_key_score",
            "rule_velocity_score", "rule_topaz_score",
        ]
        df["rule_score_raw"] = df[rule_cols].fillna(0).sum(axis=1)

        MAX_RULE_SCORE = 21
        df["rule_score_normalized"] = df["rule_score_raw"] / MAX_RULE_SCORE

        return df

    # ==========================================================
    # PREPROCESSOR TRANSFORM
    # ==========================================================
    def _transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica o PixPreprocessor treinado para imputação e transformação."""
        if self.preprocessor is None:
            return df

        id_cols = [
            "transaction_id", "customer_id", "event_datetime",
            "source_dataset", "is_fraud",
        ]
        df_for_transform = df.drop(
            columns=[c for c in id_cols if c in df.columns],
            errors="ignore",
        )

        try:
            X_transformed = self.preprocessor.transform(df_for_transform)
            return X_transformed
        except Exception as e:
            logger.warning(f"Preprocessor transform falhou: {e} — usando passthrough")
            return df

    # ==========================================================
    # CONVERSÃO DF → DICT (para os engines)
    # ==========================================================
    @staticmethod
    def _row_to_dict(df: pd.DataFrame, idx: int = 0) -> Dict[str, Any]:
        """Converte uma linha do DataFrame em dict para os engines."""
        row = df.iloc[idx]
        result = {}
        for col in df.columns:
            val = row[col]
            if isinstance(val, (np.integer,)):
                result[col] = int(val)
            elif isinstance(val, (np.floating,)):
                result[col] = float(val) if not np.isnan(val) else None
            elif isinstance(val, (np.bool_,)):
                result[col] = bool(val)
            elif isinstance(val, pd.Timestamp):
                result[col] = val.to_pydatetime()
            elif pd.isna(val):
                result[col] = None
            else:
                result[col] = val
        return result

    # ==========================================================
    # API PRINCIPAL: analisar (tempo real — 1 transação)
    # ==========================================================
    def analisar(self, data: Union[Dict, pd.Series, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analisa UMA transação PIX e retorna resultado completo.

        Fluxo v1.3:
          1. Preparar input → DataFrame padronizado
          2. Feature engineering → todas as features derivadas
          3. Preprocessor → imputação/transformação
          4. Converter para dict
          5. SE v3.4 → padrões de engenharia social
          6. Behavioral v3.0 → fatores comportamentais
          7. Decision Engine v3.0.5 → LGBM + Cascade + IF + Agravantes + Vetos
          8. SHAP → explicabilidade (CONFIRMAR/BLOQUEAR)
          9. Atualizar histórico do cliente
          10. Montar resposta final
        """
        t0 = time.perf_counter()
        timings: Dict[str, float] = {}

        # ─── 1. Preparar input ──────────────────────────────
        t1 = time.perf_counter()
        df_raw = self._prepare_raw(data)
        timings["prepare_ms"] = (time.perf_counter() - t1) * 1000

        # ─── 2. Feature Engineering ─────────────────────────
        t1 = time.perf_counter()
        df_features = self._create_features(df_raw)
        timings["features_ms"] = (time.perf_counter() - t1) * 1000

        # ─── 3. Preprocessor Transform ──────────────────────
        t1 = time.perf_counter()
        df_transformed = self._transform_features(df_features)
        timings["transform_ms"] = (time.perf_counter() - t1) * 1000

        # ─── 4. Converter para dict ─────────────────────────
        features_dict = self._row_to_dict(df_features)
        if isinstance(df_transformed, pd.DataFrame) and len(df_transformed) > 0:
            transformed_dict = self._row_to_dict(df_transformed)
            for k, v in transformed_dict.items():
                if k not in ("transaction_id", "customer_id", "event_datetime"):
                    features_dict[k] = v

        # ─── 4b. Pre-compute IF Percentile for SE ───────────
        if_score, _, _ = self.engine._score_if(features_dict)
        features_dict["if_percentile"] = if_score

        # ─── 5. Social Engineering v3.4 ─────────────────────
        t1 = time.perf_counter()
        se_result: SEAnalysisResult = self.se_detector.detect_from_pipeline(features_dict)
        timings["se_ms"] = (time.perf_counter() - t1) * 1000

        # ─── 6. Behavioral Analytics v3.1 ───────────────────
        t1 = time.perf_counter()
        behavioral_result: BehavioralAnalysisResult = self.behavioral.analyze(features_dict)
        timings["behavioral_ms"] = (time.perf_counter() - t1) * 1000

        # ─── 7. Decision Engine v3.0.5 ────────────────────────
        t1 = time.perf_counter()
        decision: DecisionResult = self.engine.decide(
            features=features_dict,
            se_result=se_result.to_dict(),
            behavioral_result=behavioral_result.to_dict(),
        )
        timings["engine_ms"] = (time.perf_counter() - t1) * 1000

        # ─── 8. SHAP Explicabilidade ────────────────────────
        t1 = time.perf_counter()
        shap_explanation = None
        if self.shap_enabled and decision.decisao in ("CONFIRMAR", "BLOQUEAR"):
            try:
                lgbm_features = self.engine.lgbm_features
                available_feats = [f for f in lgbm_features if f in df_features.columns]
                if len(available_feats) == len(lgbm_features):
                    X_for_shap = df_features[lgbm_features].copy().fillna(0)
                    shap_explanation = self._compute_shap_explanation(
                        X_for_shap, decision.decisao
                    )
                else:
                    missing = set(lgbm_features) - set(available_feats)
                    logger.warning(
                        f"SHAP: {len(missing)} features faltando: {list(missing)[:5]}"
                    )
            except Exception as e:
                logger.warning(f"SHAP falhou: {e}")
        timings["shap_ms"] = (time.perf_counter() - t1) * 1000

        # ─── 9. Atualizar histórico ─────────────────────────
        self._update_customer_history(df_features)

        # ─── 10. Montar resposta final ──────────────────────
        total_ms = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = total_ms

        response = self._build_response(
            decision=decision,
            se_result=se_result,
            behavioral_result=behavioral_result,
            features_dict=features_dict,
            timings=timings,
            shap_explanation=shap_explanation,
        )

        # Log resumido
        cascade_info = (
            f" CASCADE={','.join(decision.cascade_rules)}"
            if decision.cascade_triggered else ""
        )
        if_info = (
            f" IF_boost={decision.if_boost_applied:.2f}" if decision.if_active else ""
        )
        shap_info = " SHAP=OK" if shap_explanation and "error" not in shap_explanation else ""
        veto_info = f" VETO" if decision.veto_aplicado else ""

        logger.info(
            f"TX {decision.transaction_id} | "
            f"{decision.decisao} | "
            f"Score={decision.score_final:.1f} | "
            f"LGBM={decision.score_lgbm_raw:.4f} | "
            f"SE={se_result.se_score:.0f} | "
            f"BEH={behavioral_result.behavioral_score:.0f} | "
            f"Agr={decision.peso_total}/{decision.peso_maximo}"
            f"{cascade_info}{if_info}{veto_info}{shap_info} | "
            f"{total_ms:.0f}ms"
        )

        return response

    # ==========================================================
    # API BATCH
    # ==========================================================
    def analisar_batch(
        self,
        data: Union[List[Dict], pd.DataFrame],
        max_workers: int = 1,
    ) -> List[Dict[str, Any]]:
        """Analisa múltiplas transações em sequência."""
        if isinstance(data, pd.DataFrame):
            rows = [data.iloc[i] for i in range(len(data))]
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError(f"Tipo não suportado para batch: {type(data)}")

        results = []
        t0 = time.perf_counter()

        for i, row in enumerate(rows):
            try:
                result = self.analisar(row)
                results.append(result)
            except Exception as e:
                logger.error(f"Erro na transação {i}: {e}")
                results.append({
                    "error": str(e),
                    "index": i,
                    "decisao": "ERRO",
                    "score_final": -1,
                })

        total_ms = (time.perf_counter() - t0) * 1000
        avg_ms = total_ms / max(len(rows), 1)
        logger.info(
            f"Batch concluído: {len(rows)} transações em {total_ms:.0f}ms "
            f"(média {avg_ms:.1f}ms/tx)"
        )

        return results

    # ==========================================================
    # BUILD RESPONSE v1.3
    # ==========================================================
    def _build_response(
        self,
        decision: DecisionResult,
        se_result: SEAnalysisResult,
        behavioral_result: BehavioralAnalysisResult,
        features_dict: Dict[str, Any],
        timings: Dict[str, float],
        shap_explanation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Monta resposta padronizada para API/consumidor.

        v1.3 — Mudanças vs v1.2:
          - Behavioral risk_factors inclui precision e origin (v3.0)
          - Metadata inclui versões de todos os módulos
          - Veto info inclui contexto SE/BEH quando aplicável
        """
        response: Dict[str, Any] = {}

        # ─── Decisão ────────────────────────────────────────
        response["decisao"] = decision.decisao
        response["score_final"] = round(decision.score_final, 2)

        # ─── Identificação ──────────────────────────────────
        response["transaction_id"] = decision.transaction_id
        response["customer_id"] = decision.customer_id
        response["timestamp"] = str(features_dict.get("event_datetime", ""))
        response["vl_pix"] = features_dict.get("vl_pix")

        # ─── Componentes de Score ───────────────────────────
        componentes = {
            "lgbm_raw": round(decision.score_lgbm_raw, 8),
            "lgbm_mapped": round(decision.score_lgbm_mapped, 2),
            "rule_score_raw": round(decision.rule_score_raw, 2),
            "rule_score_normalized": round(decision.rule_score_normalized, 4),
        }
        if decision.if_active:
            componentes["if_score"] = round(decision.score_if, 6)
            componentes["if_raw"] = round(decision.score_if_raw, 6)
            componentes["if_boost_applied"] = round(decision.if_boost_applied, 4)
        response["componentes"] = componentes

        # ─── Cascade — só incluir quando triggered ──────────
        if decision.cascade_triggered:
            response["cascade"] = {
                "triggered": True,
                "rules": decision.cascade_rules,
            }

        # ─── Agravantes ─────────────────────────────────────
        agravantes_ativos = [a for a in decision.agravantes if a.peso > 0]
        if agravantes_ativos:
            response["agravantes"] = [
                {
                    "codigo": a.codigo,
                    "descricao": a.descricao,
                    "peso": a.peso,
                }
                for a in agravantes_ativos
            ]
            response["peso_total"] = decision.peso_total

        # ─── SHAP Explicabilidade ───────────────────────────
        if shap_explanation is not None:
            response["explicabilidade"] = shap_explanation

        # ─── Social Engineering v3.4 ────────────────────────
        if se_result.se_score > 0 or se_result.patterns:
            response["social_engineering"] = {
                "se_score": round(se_result.se_score, 2),
                "risk_level": se_result.risk_level,
                "patterns": [
                    {
                        "pattern_name": p.pattern_name,
                        "severity": p.severity,
                        "score": p.score,
                        "matched_indicators": p.matched_indicators,
                        "description": p.description,
                    }
                    for p in se_result.patterns
                ],
                "worst_pattern": (
                    se_result.worst_pattern.pattern_name
                    if se_result.worst_pattern
                    else None
                ),
            }

        # ─── Behavioral v3.0 ───────────────────────────────
        if behavioral_result.behavioral_score > 0 or behavioral_result.risk_factors:
            beh_block: Dict[str, Any] = {
                "behavioral_score": round(behavioral_result.behavioral_score, 2),
                "risk_level": behavioral_result.risk_level,
                "risk_factors": [
                    {
                        "codigo": rf.codigo,
                        "descricao": rf.descricao,
                        "peso": rf.peso,
                        "source": rf.source,
                        "precision": getattr(rf, "precision", None),
                        "origin": getattr(rf, "origin", None),
                    }
                    for rf in behavioral_result.risk_factors
                ],
            }
            if behavioral_result.device_info:
                beh_block["device_info"] = {
                    "device_model": behavioral_result.device_info.device_model,
                    "device_type": behavioral_result.device_info.device_type,
                    "is_known": behavioral_result.device_info.is_known,
                }
            response["behavioral"] = beh_block

        # ─── Veto ──────────────────────────────────────────
        if decision.veto_aplicado:
            response["veto_aplicado"] = decision.veto_aplicado
            response["veto_reason"] = decision.veto_reason
        if decision.veto_suppressed_reason:
            response["veto_suppressed_reason"] = decision.veto_suppressed_reason

        # ─── Atenuantes — só incluir quando presentes ───────
        if decision.atenuantes:
            response["atenuantes"] = decision.atenuantes

        # ─── Metadata ──────────────────────────────────────
        response["metadata"] = {
            "pipeline_version": PIPELINE_VERSION,
            "engine_version": self._engine_version,
            "se_version": self._se_version,
            "behavioral_version": self._behavioral_version,
            "scoring_version": self.engine.scoring_config.get("versao", "N/A"),
            "cascade_enabled": self.engine.config.cascade_enabled,
            "shap_enabled": self.shap_enabled,
            "lgbm_threshold": self.engine.config.lgbm_threshold,
            "faixas": {
                "aprovar": f"[0, {self.engine.config.threshold_confirmar:.0f})",
                "confirmar": (
                    f"[{self.engine.config.threshold_confirmar:.0f}, "
                    f"{self.engine.config.threshold_bloquear:.0f})"
                ),
                "bloquear": f"[{self.engine.config.threshold_bloquear:.0f}, 100]",
            },
            "timings": {k: round(v, 1) for k, v in timings.items()},
            "timestamp_inferencia": datetime.utcnow().isoformat() + "Z",
        }

        return response

    # ==========================================================
    # HEALTH CHECK v1.3
    # ==========================================================
    def get_status(self) -> Dict[str, Any]:
        """Retorna status completo do pipeline para health check."""
        engine_status = self.engine.get_status()

        return {
            "pipeline_version": PIPELINE_VERSION,
            "available": self.available,
            "load_time_ms": round(self._load_time_ms, 1),
            "modules": {
                "engine": {
                    "version": self._engine_version,
                    "available": self.engine.available,
                    "lgbm": engine_status.get("lgbm", False),
                    "lgbm_features": engine_status.get("lgbm_features", 0),
                    "lgbm_threshold": engine_status.get("lgbm_threshold", "N/A"),
                    "if_enabled": engine_status.get("if_enabled", False),
                    "cascade_enabled": engine_status.get("cascade_enabled", False),
                },
                "social_engineering": {
                    "version": self._se_version,
                    "available": True,
                    "patterns": len(self.se_detector.PATTERNS),
                    "indicators": len(self.se_detector.INDICATORS),
                },
                "behavioral": {
                    "version": self._behavioral_version,
                    "available": True,
                    "factors": len(BehavioralAnalytics.get_factor_catalog()),
                    "categories": ["velocity", "dormancy", "profile"],
                },
                "preprocessor": {
                    "available": self.preprocessor is not None,
                },
                "shap": {
                    "available": self._shap_explainer is not None,
                    "enabled": self.shap_enabled,
                    "top_n": self.shap_top_n,
                },
            },
            "thresholds": engine_status.get("thresholds", {}),
            "integration": engine_status.get("integration", {}),
            "scoring_version": engine_status.get("scoring_version", "N/A"),
        }
