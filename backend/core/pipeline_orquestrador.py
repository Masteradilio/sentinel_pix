"""
pipeline_orquestrador.py v1.0 — Orquestrador de Inferência PIX Antifraude

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
  └────┬────┘ └───┬────┘ └───┬────┘
       │          │          │
       └──────────┼──────────┘
                  ▼
  ┌─────────────────────────────────┐
  │  2. Consolidação                │
  │     Score final + Decisão       │
  │     Agravantes + Vetos          │
  │     Resposta padronizada        │
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

Dependências:
    - preprocessing.py          → PixPreprocessor + funções de FE
    - core/decision_engine.py   → PixDecisionEngine (scoring + agravantes + vetos)
    - core/behavioral_analytics.py → BehavioralAnalytics (12 fatores RT)
    - core/social_engineering.py   → SocialEngineeringDetector (11 padrões)
    - artefatos/                → Modelos treinados (.joblib, .json)
"""

from __future__ import annotations

import logging
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# =========================================================
# IMPORTS DO PROJETO
# =========================================================

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent

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
import sys
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


# =========================================================
# CONFIGURAÇÃO DE COLUNAS (única fonte de verdade)
# =========================================================

# Colunas que chegam do banco/API (dados brutos)
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


# =========================================================
# ORQUESTRADOR
# =========================================================
class PipelineOrquestrador:
    """
    Orquestrador de inferência para detecção de fraude PIX.

    Responsabilidades:
        1. Feature engineering (usa preprocessing.py — sem duplicação)
        2. Coordenar os 3 engines (Decision, SE, Behavioral)
        3. Consolidar resposta final padronizada
        4. Manter cache de histórico por cliente (features sequenciais)

    O que NÃO faz:
        - Servir HTTP (→ api.py)
        - Treinar modelos (→ notebooks/scripts de treino)
        - Carregar dados em massa (→ scripts de ingestão)
    """

    def __init__(
        self,
        artefatos_dir: Optional[str] = None,
        engine_config: Optional[EngineConfig] = None,
    ):
        """
        Inicializa o orquestrador carregando todos os componentes.

        Args:
            artefatos_dir: Caminho para pasta de artefatos.
            engine_config: Config do decision engine (thresholds, pesos).
        """
        self.artefatos_dir = Path(artefatos_dir) if artefatos_dir else ARTEFATOS_DIR

        # Timers
        t0 = time.perf_counter()

        # --- 1. Preprocessor ---
        self.preprocessor: Optional[PixPreprocessor] = None
        self._load_preprocessor()

        # --- 2. Decision Engine ---
        config = engine_config or EngineConfig(artefatos_dir=str(self.artefatos_dir))
        self.engine = PixDecisionEngine(config)

        # --- 3. Social Engineering Detector ---
        self.se_detector = SocialEngineeringDetector()

        # --- 4. Behavioral Analytics ---
        self.behavioral = BehavioralAnalytics()

        # --- 5. Cache de histórico por cliente ---
        self._customer_history: Dict[str, Dict[str, Any]] = {}

        # Status
        self._load_time_ms = (time.perf_counter() - t0) * 1000
        self.available = self.engine.available

        logger.info(
            f"PipelineOrquestrador v1.0 inicializado em {self._load_time_ms:.0f}ms | "
            f"Engine={'OK' if self.engine.available else 'DEGRADED'} | "
            f"Preprocessor={'OK' if self.preprocessor else 'PASSTHROUGH'} | "
            f"SE=OK | Behavioral=OK"
        )

    # ==========================================================
    # LOADING
    # ==========================================================
    def _load_preprocessor(self):
        """Carrega o PixPreprocessor treinado."""
        path = self.artefatos_dir / "preprocessing.joblib"
        if path.exists():
            try:
                self.preprocessor = joblib.load(path)
                n_cols = len(getattr(self.preprocessor, "model_columns_", []))
                logger.info(f"Preprocessor carregado: {n_cols} colunas")
            except Exception as e:
                logger.warning(f"Erro ao carregar preprocessor: {e}")
                self.preprocessor = None
        else:
            logger.warning(f"Preprocessor não encontrado em {path} — usando passthrough")

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
        df["metodo_auth_encoded"] = (
            df["metodo_autenticacao"]
            .str.strip().str.lower()
            .map(auth_map)
            .fillna(0)
            .astype(int)
        )
        df["is_login_biometria_flag"] = (df["metodo_auth_encoded"] == 1).astype(int)
        df["is_login_senha_flag"] = (df["metodo_auth_encoded"] == 2).astype(int)

        # ─── Agendamento ────────────────────────────────────
        df["is_agendamento_recorrente_flag"] = (
            df["is_agendamento_recorrente"]
            .fillna("false")
            .astype(str).str.strip().str.lower()
            .isin(["true", "1", "sim", "yes"])
            .astype(int)
        )

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

        # Sexo
        sexo_upper = df["ds_sexo"].fillna("").astype(str).str.upper()
        df["is_sexo_feminino_flag"] = sexo_upper.isin(["F", "FEMININO", "FEMALE"]).astype(int)

        # Estado civil
        estado_civil_upper = df["ds_estado_civil"].fillna("").astype(str).str.upper()
        df["is_viuvo_flag"] = estado_civil_upper.str.contains("VIUV", na=False).astype(int)

        # Segmento
        segmento_upper = df["ds_segmento"].fillna("").astype(str).str.upper()
        df["is_segmento_premium_flag"] = segmento_upper.isin(
            ["EXCLUSIVO", "PRIVATE", "MILLENIUM", "PREMIUM", "VIP"]
        ).astype(int)

        # Dependentes
        df["qt_dependentes"] = pd.to_numeric(df["qt_dependentes"], errors="coerce").fillna(0)

        # Renda
        renda = pd.to_numeric(df["vl_renda_cliente"], errors="coerce")
        vl_pix = df["vl_pix"]

        df["ratio_pix_renda"] = np.where(
            renda.notna() & (renda > 0),
            vl_pix / renda,
            np.nan,
        )
        df["pix_over_50pct_renda_flag"] = np.where(
            renda.notna() & (renda > 0),
            (vl_pix > renda * 0.5).astype(int),
            0,
        )
        df["pix_over_100pct_renda_flag"] = np.where(
            renda.notna() & (renda > 0),
            (vl_pix > renda).astype(int),
            0,
        )

        # Perfil vulnerável: viúvo + idoso (65+) + sem dependentes
        df["perfil_vulneravel_se_flag"] = (
            (df["is_viuvo_flag"] == 1)
            & (df["nr_idade"] >= 65)
            & (df["qt_dependentes"] == 0)
        ).astype(int)

        return df

    def _create_sequential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features sequenciais usando cache de histórico por cliente.

        Mantém em memória: último timestamp, contadores de recebedores,
        contadores de chaves, timestamps recentes (para burst detection).
        """
        for idx in df.index:
            customer_id = str(df.loc[idx, "customer_id"])
            event_time = df.loc[idx, "event_datetime"]
            receiver = str(df.loc[idx, "cd_cpf_cnpj_recebedor"])
            pix_key = str(df.loc[idx, "ds_chave_pix"])

            hist = self._customer_history.get(customer_id)

            if hist is not None and pd.notna(event_time):
                # Minutos desde última transação
                last_time = hist.get("last_event_time")
                if last_time is not None and pd.notna(last_time):
                    diff_min = (event_time - last_time).total_seconds() / 60.0
                    df.loc[idx, "minutes_since_prev_tx"] = max(diff_min, 0)
                else:
                    df.loc[idx, "minutes_since_prev_tx"] = np.nan

                # Contagem de tx nos últimos 30 minutos
                recent_times = hist.get("recent_times", [])
                count_30m = sum(
                    1 for t in recent_times
                    if pd.notna(t) and (event_time - t).total_seconds() / 60.0 <= 30
                )
                df.loc[idx, "tx_count_prev_30m"] = count_30m

                # Contagem de tx para este recebedor
                receiver_counts = hist.get("receiver_counts", {})
                df.loc[idx, "receiver_tx_count_prev"] = receiver_counts.get(receiver, 0)
                df.loc[idx, "first_receiver_flag"] = 1 if receiver_counts.get(receiver, 0) == 0 else 0

                # Contagem de tx com esta chave
                key_counts = hist.get("key_counts", {})
                df.loc[idx, "key_tx_count_prev"] = key_counts.get(pix_key, 0)
                df.loc[idx, "first_key_flag"] = 1 if key_counts.get(pix_key, 0) == 0 else 0

                # Distintos acumulados
                df.loc[idx, "distinct_receivers_so_far"] = len(receiver_counts) + (
                    1 if receiver not in receiver_counts else 0
                )
                df.loc[idx, "distinct_keys_so_far"] = len(key_counts) + (
                    1 if pix_key not in key_counts else 0
                )
            else:
                # Primeira transação conhecida deste cliente
                df.loc[idx, "minutes_since_prev_tx"] = np.nan
                df.loc[idx, "tx_count_prev_30m"] = 0
                df.loc[idx, "receiver_tx_count_prev"] = 0
                df.loc[idx, "first_receiver_flag"] = 1
                df.loc[idx, "key_tx_count_prev"] = 0
                df.loc[idx, "first_key_flag"] = 1
                df.loc[idx, "distinct_receivers_so_far"] = 1
                df.loc[idx, "distinct_keys_so_far"] = 1

        # Burst flag
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
                # Manter apenas últimos 60 minutos
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
            [1, 2, 3],
            default=0,
        )

        df["rule_relationship_score"] = np.select(
            [
                df["qt_tempo_relacionamento_mes"].between(61, 90),
                df["qt_tempo_relacionamento_mes"].between(31, 60),
                df["qt_tempo_relacionamento_mes"].between(0, 30),
            ],
            [1, 2, 3],
            default=0,
        )

        df["rule_mule_account_score"] = np.select(
            [
                df["first_receiver_flag"] == 1,
                df["receiver_document_same_as_customer_flag"] == 1,
            ],
            [2, 1],
            default=0,
        )

        df["rule_random_key_score"] = np.select(
            [
                df["pix_key_random_flag"] == 1,
                df["pix_key_email_flag"] == 1,
                df["pix_key_document_flag"] == 1,
            ],
            [2, 1, 0],
            default=0,
        )

        df["rule_velocity_score"] = np.select(
            [
                df["tx_count_prev_30m"] == 0,
                df["tx_count_prev_30m"] == 1,
                df["tx_count_prev_30m"] == 2,
                df["tx_count_prev_30m"] >= 3,
            ],
            [0, 2, 3, 4],
            default=0,
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
        """
        Aplica o PixPreprocessor treinado para imputação e transformação.

        Se preprocessor não disponível, retorna o DataFrame como está
        (modo degradado — features brutas vão direto para os modelos).
        """
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
            # Converter numpy types para Python native
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

        Este é o método principal para inferência em tempo real.

        Args:
            data: Dados brutos da transação (dict, Series ou DataFrame de 1 linha).

        Returns:
            Dict padronizado com decisão, scores, agravantes e metadata.

        Exemplo:
            resultado = pipeline.analisar({
                "cd_pix": "E00000208...",
                "dt_pix": "2026-03-19 14:30:00",
                "vl_pix": 5000.00,
                "nr_idade": 72,
                ...
            })
            print(resultado["decisao"])       # "BLOQUEAR"
            print(resultado["score_final"])   # 89.3
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

        # ─── 4. Converter para dict (para os engines) ───────
        # Merge: features originais + transformadas
        features_dict = self._row_to_dict(df_features)
        if isinstance(df_transformed, pd.DataFrame) and len(df_transformed) > 0:
            transformed_dict = self._row_to_dict(df_transformed)
            # Features transformadas sobrescrevem (mais tratadas)
            for k, v in transformed_dict.items():
                if k not in ("transaction_id", "customer_id", "event_datetime"):
                    features_dict[k] = v

        # ─── 5. Social Engineering ──────────────────────────
        t1 = time.perf_counter()
        se_result: SEAnalysisResult = self.se_detector.detect_from_pipeline(features_dict)
        timings["se_ms"] = (time.perf_counter() - t1) * 1000

        # ─── 6. Behavioral Analytics ────────────────────────
        t1 = time.perf_counter()
        behavioral_result: BehavioralAnalysisResult = self.behavioral.analyze(features_dict)
        timings["behavioral_ms"] = (time.perf_counter() - t1) * 1000

        # ─── 7. Decision Engine ─────────────────────────────
        t1 = time.perf_counter()
        decision: DecisionResult = self.engine.decide(
            features=features_dict,
            se_result=se_result.to_dict(),
            behavioral_result=behavioral_result.to_dict(),
        )
        timings["engine_ms"] = (time.perf_counter() - t1) * 1000

        # ─── 8. Atualizar histórico ─────────────────────────
        self._update_customer_history(df_features)

        # ─── 9. Montar resposta final ───────────────────────
        total_ms = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = total_ms

        response = self._build_response(
            decision=decision,
            se_result=se_result,
            behavioral_result=behavioral_result,
            features_dict=features_dict,
            timings=timings,
        )

        # Log resumido
        logger.info(
            f"TX {decision.transaction_id} | "
            f"{decision.decisao} | "
            f"Score={decision.score_final:.1f} | "
            f"SE={se_result.se_score:.0f} | "
            f"BEH={behavioral_result.behavioral_score:.0f} | "
            f"Agr={decision.peso_total}/{decision.peso_maximo} | "
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
        """
        Analisa múltiplas transações em sequência.

        Args:
            data: Lista de dicts ou DataFrame com múltiplas linhas.
            max_workers: Reservado para paralelismo futuro (atualmente sequencial).

        Returns:
            Lista de resultados (um por transação).
        """
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
    # BUILD RESPONSE
    # ==========================================================
    def _build_response(
        self,
        decision: DecisionResult,
        se_result: SEAnalysisResult,
        behavioral_result: BehavioralAnalysisResult,
        features_dict: Dict[str, Any],
        timings: Dict[str, float],
    ) -> Dict[str, Any]:
        """Monta resposta padronizada para API/consumidor."""
        return {
            # ─── Decisão ────────────────────────────────────
            "decisao": decision.decisao,
            "score_final": round(decision.score_final, 2),
            "score_raw": round(decision.score_raw, 8),

            # ─── Identificação ──────────────────────────────
            "transaction_id": decision.transaction_id,
            "customer_id": decision.customer_id,
            "timestamp": str(features_dict.get("event_datetime", "")),
            "vl_pix": features_dict.get("vl_pix"),

            # ─── Componentes de Score ───────────────────────
            "componentes": {
                "lgbm_raw": round(decision.score_lgbm_raw, 8),
                "lgbm_mapped": round(decision.score_lgbm_mapped, 2),
                "if_score": round(decision.score_if, 6),
                "if_active": decision.if_active,
                "rule_score_raw": round(decision.rule_score_raw, 2),
                "rule_score_normalized": round(decision.rule_score_normalized, 4),
            },

            # ─── Agravantes ─────────────────────────────────
            "agravantes": [
                {
                    "codigo": a.codigo,
                    "descricao": a.descricao,
                    "peso": a.peso,
                }
                for a in decision.agravantes
            ],
            "peso_total": decision.peso_total,
            "peso_maximo": decision.peso_maximo,

            # ─── Social Engineering ─────────────────────────
            "social_engineering": {
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
                    if se_result.worst_pattern else None
                ),
            },

            # ─── Behavioral ─────────────────────────────────
            "behavioral": {
                "behavioral_score": round(behavioral_result.behavioral_score, 2),
                "risk_level": behavioral_result.risk_level,
                "risk_factors": [
                    {
                        "codigo": rf.codigo,
                        "descricao": rf.descricao,
                        "peso": rf.peso,
                        "source": rf.source,
                    }
                    for rf in behavioral_result.risk_factors
                ],
                "device_info": {
                    "device_model": behavioral_result.device_info.device_model,
                    "device_type": behavioral_result.device_info.device_type,
                    "is_known": behavioral_result.device_info.is_known,
                } if behavioral_result.device_info else None,
            },

            # ─── Veto / Atenuantes ──────────────────────────
            "veto_aplicado": decision.veto_aplicado,
            "atenuantes": decision.atenuantes,

            # ─── Faixas de decisão ──────────────────────────
            "faixas": {
                "aprovar": f"[0, {self.engine.config.threshold_confirmar:.0f})",
                "confirmar": f"[{self.engine.config.threshold_confirmar:.0f}, {self.engine.config.threshold_bloquear:.0f})",
                "bloquear": f"[{self.engine.config.threshold_bloquear:.0f}, 100]",
            },

            # ─── Metadata ──────────────────────────────────
            "metadata": {
                "pipeline_version": "1.0",
                "engine_version": "2.0",
                "scoring_version": self.engine.scoring_config.get("versao", "N/A"),
                "timings": {k: round(v, 1) for k, v in timings.items()},
                "timestamp_inferencia": datetime.utcnow().isoformat() + "Z",
            },
        }

    # ==========================================================
    # HEALTH CHECK / STATUS
    # ==========================================================
    def get_status(self) -> Dict[str, Any]:
        """Retorna status completo do pipeline para health checks."""
        engine_status = self.engine.get_status()
        return {
            "status": "healthy" if self.available else "degraded",
            "pipeline_version": "1.0",
            "load_time_ms": round(self._load_time_ms, 1),
            "components": {
                "preprocessor": self.preprocessor is not None,
                "decision_engine": engine_status,
                "social_engineering": True,
                "behavioral_analytics": True,
            },
            "cache": {
                "customers_tracked": len(self._customer_history),
            },
            "thresholds": {
                "confirmar": self.engine.config.threshold_confirmar,
                "bloquear": self.engine.config.threshold_bloquear,
                "veto": self.engine.config.veto_threshold,
            },
        }

    def reset_cache(self):
        """Limpa cache de histórico (útil para testes)."""
        self._customer_history.clear()
        logger.info("Cache de histórico resetado")


# =========================================================
# TESTE STANDALONE
# =========================================================
def main():
    """Teste completo do pipeline orquestrador."""
    print("\n")
    print("█" * 70)
    print("  TESTE DO PIPELINE ORQUESTRADOR v1.0")
    print("  Feature Engineering → SE + Behavioral + Engine → Decisão")
    print("█" * 70)

    pipeline = PipelineOrquestrador()
    status = pipeline.get_status()
    print(f"\n  📊 Status: {status['status']}")
    print(f"     Load time: {status['load_time_ms']:.0f}ms")
    print(f"     Preprocessor: {'✅' if status['components']['preprocessor'] else '⚠️'}")
    engine = status['components']['decision_engine']
    print(f"     LGBM: {'✅' if engine['lgbm'] else '❌'} ({engine['lgbm_features']} features)")
    print(f"     IF: {'✅' if engine['if_enabled'] else '—'} ({engine['if_features']} features)")

    # ─── Teste 1: Transação Normal ───
    tx_normal = {
        "cd_pix": "E00000208202603191430001234567890",
        "dt_pix": "2026-03-19 14:30:00",
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
        "tempo_interacao_ms": 5200.0,
        "vl_tempo_interacao_medio_trimestre": 4800.0,
        "tempo_processamento_host_ms": 120.0,
        "metodo_autenticacao": "biometria",
        "session_id": "sess_abc123",
        "cd_retorno": "00",
        "topaz_risk_score": 1.5,
        "topaz_transacao_rejeitada": 0,
        "is_agendamento_recorrente": "false",
        "qt_aparelhos_distintos_trimestre": 1,
        "nr_idade": 35,
        "qt_tempo_relacionamento_mes": 120,
        "vl_renda_cliente": 8000.00,
        "ds_sexo": "M",
        "ds_estado_civil": "CASADO",
        "ds_segmento": "VAREJO",
        "qt_dependentes": 2,
    }

    print(f"\n{'─' * 60}")
    print(f"  🧪 TESTE 1: Transação Normal")
    print(f"     R$150 | 14h30 | Biometria | Cliente 10 anos | Renda R$8k")
    print(f"{'─' * 60}")
    r1 = pipeline.analisar(tx_normal)
    _print_result(r1)

    # ─── Teste 2: Transação Suspeita (Idoso + Chave Aleatória) ───
    tx_suspeita = {
        "cd_pix": "E00000208202603190315009876543210",
        "dt_pix": "2026-03-19 03:15:00",
        "cd_cpf_pagador": "99887766554",
        "cd_cpf_cnpj_recebedor": "11223344556",
        "ds_chave_pix": "abc123-def456-ghi789",
        "ds_tipo_chave": "CHAVE ALEATORIA",
        "vl_pix": 4999.00,
        "qt_total_pix_trimestre": 1,
        "vl_mediana_pix_trimestre": 0,
        "vl_desvio_padrao_pix_trimestre": 0,
        "qt_intervalo_transacao_minuto": 0,
        "qt_intervalo_mediana_trimestre": 0,
        "qt_intervalo_desvio_padrao_trimestre": 0,
        "qt_pix_dia_maximo_trimestre": 1,
        "device_name": None,
        "app_version": "7.10.0",
        "ip_address": None,
        "latencia_rede_ms": None,
        "vl_latencia_rede_media_trimestre": None,
        "tempo_interacao_ms": None,
        "vl_tempo_interacao_medio_trimestre": None,
        "tempo_processamento_host_ms": None,
        "metodo_autenticacao": "senha",
        "session_id": None,
        "cd_retorno": "00",
        "topaz_risk_score": 4.0,
        "topaz_transacao_rejeitada": 0,
        "is_agendamento_recorrente": None,
        "qt_aparelhos_distintos_trimestre": 1,
        "nr_idade": 78,
        "qt_tempo_relacionamento_mes": 2,
        "vl_renda_cliente": 3200.00,
        "ds_sexo": "F",
        "ds_estado_civil": "VIUVA",
        "ds_segmento": "VAREJO",
        "qt_dependentes": 0,
    }

    print(f"\n{'─' * 60}")
    print(f"  🧪 TESTE 2: Transação Suspeita")
    print(f"     R$4.999 | 3h15 | Senha | Viúva 78 anos | Chave aleatória | 2 meses")
    print(f"{'─' * 60}")
    r2 = pipeline.analisar(tx_suspeita)
    _print_result(r2)

    # ─── Teste 3: Transação Intermediária ───
    tx_inter = {
        "cd_pix": "E00000208202603191800005551234567",
        "dt_pix": "2026-03-19 18:00:00",
        "cd_cpf_pagador": "55512345678",
        "cd_cpf_cnpj_recebedor": "66698765432",
        "ds_chave_pix": "66698765432",
        "ds_tipo_chave": "DOCUMENTO/TELEFONE",
        "vl_pix": 2500.00,
        "qt_total_pix_trimestre": 5,
        "vl_mediana_pix_trimestre": 800.00,
        "vl_desvio_padrao_pix_trimestre": 400.00,
        "qt_intervalo_transacao_minuto": 60,
        "qt_intervalo_mediana_trimestre": 2880,
        "qt_intervalo_desvio_padrao_trimestre": 1440,
        "qt_pix_dia_maximo_trimestre": 2,
        "device_name": "iPhone 14",
        "app_version": "7.11.0",
        "ip_address": "10.0.0.1",
        "latencia_rede_ms": 80.0,
        "vl_latencia_rede_media_trimestre": 50.0,
        "tempo_interacao_ms": 3500.0,
        "vl_tempo_interacao_medio_trimestre": 4000.0,
        "tempo_processamento_host_ms": 200.0,
        "metodo_autenticacao": "senha",
        "session_id": "sess_xyz789",
        "cd_retorno": "00",
        "topaz_risk_score": 2.0,
        "topaz_transacao_rejeitada": 0,
        "is_agendamento_recorrente": "false",
        "qt_aparelhos_distintos_trimestre": 2,
        "nr_idade": 45,
        "qt_tempo_relacionamento_mes": 36,
        "vl_renda_cliente": 5500.00,
        "ds_sexo": "M",
        "ds_estado_civil": "CASADO",
        "ds_segmento": "EXCLUSIVO",
        "qt_dependentes": 1,
    }

    print(f"\n{'─' * 60}")
    print(f"  🧪 TESTE 3: Transação Intermediária")
    print(f"     R$2.500 | 18h | Senha | 45 anos | Segmento Exclusivo | 3x mediana")
    print(f"{'─' * 60}")
    r3 = pipeline.analisar(tx_inter)
    _print_result(r3)

    # ─── Resumo ───
    print(f"\n{'═' * 60}")
    print(f"  📋 RESUMO DOS TESTES")
    print(f"{'═' * 60}")
    icons = {"APROVAR": "🟢", "CONFIRMAR": "🟡", "BLOQUEAR": "🔴", "ERRO": "❌"}
    for i, (label, r) in enumerate([
        ("Normal", r1), ("Suspeita", r2), ("Intermediária", r3)
    ], 1):
        icon = icons.get(r["decisao"], "❓")
        se = r["social_engineering"]["se_score"]
        beh = r["behavioral"]["behavioral_score"]
        agr = r["peso_total"]
        ms = r["metadata"]["timings"]["total_ms"]
        print(
            f"  {i}. {label:15} → {icon} {r['decisao']:10} | "
            f"Score={r['score_final']:5.1f} | "
            f"SE={se:4.0f} | BEH={beh:4.0f} | "
            f"Agr={agr:2d}/{r['peso_maximo']} | "
            f"{ms:.0f}ms"
        )

    print(f"\n  ✅ Pipeline Orquestrador v1.0 funcionando!")
    print(f"  💡 Próximo: criar api.py (FastAPI)")


def _print_result(r: Dict):
    """Imprime resultado formatado."""
    icons = {"APROVAR": "🟢", "CONFIRMAR": "🟡", "BLOQUEAR": "🔴", "ERRO": "❌"}
    icon = icons.get(r["decisao"], "❓")

    print(f"\n  {icon} DECISÃO: {r['decisao']}")
    print(f"  ┌─────────────────────────────────────────────────┐")
    print(f"  │ Score Final:     {r['score_final']:6.2f} / 100                  │")
    print(f"  │ Score Raw:       {r['score_raw']:.8f}                   │")
    print(f"  ├─────────────────────────────────────────────────┤")

    comp = r["componentes"]
    print(f"  │ LGBM Raw:        {comp['lgbm_raw']:.8f} → {comp['lgbm_mapped']:5.1f}   │")
    print(f"  │ IF Score:        {comp['if_score']:.6f}  "
          f"({'✅' if comp['if_active'] else '—':2})              │")
    print(f"  │ Rules:           {comp['rule_score_raw']:.0f}/21 "
          f"({comp['rule_score_normalized']:.1%})                  │")

    # Agravantes
    agr_ativos = [a for a in r["agravantes"] if a["peso"] > 0]
    if agr_ativos:
        print(f"  ├─────────────────────────────────────────────────┤")
        print(f"  │ AGRAVANTES ({r['peso_total']}/{r['peso_maximo']}):                            │")
        for a in agr_ativos[:5]:
            codigo = a["codigo"][:30]
            print(f"  │   [{a['peso']}] {codigo:30}            │")
        if len(agr_ativos) > 5:
            print(f"  │   ... +{len(agr_ativos)-5} agravantes                         │")

    # SE
    se = r["social_engineering"]
    if se["se_score"] > 0:
        print(f"  ├─────────────────────────────────────────────────┤")
        print(f"  │ ENG. SOCIAL: {se['risk_level']:8} (score={se['se_score']:.0f})           │")
        for p in se["patterns"][:2]:
            print(f"  │   ⚠️  {p['pattern_name'][:35]:35}        │")

    # Behavioral
    beh = r["behavioral"]
    if beh["behavioral_score"] > 0:
        print(f"  ├─────────────────────────────────────────────────┤")
        print(f"  │ BEHAVIORAL: {beh['risk_level']:8} (score={beh['behavioral_score']:.0f})          │")
        for rf in beh["risk_factors"][:2]:
            print(f"  │   🔍 {rf['codigo'][:35]:35}        │")

    # Veto
    if r.get("veto_aplicado"):
        print(f"  ├─────────────────────────────────────────────────┤")
        print(f"  │ 🚫 VETO: {r['veto_aplicado'][:40]:40}│")

    # Timing
    t = r["metadata"]["timings"]
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │ Latência: {t['total_ms']:.0f}ms total "
          f"(FE={t['features_ms']:.0f} SE={t['se_ms']:.0f} "
          f"BEH={t['behavioral_ms']:.0f} ENG={t['engine_ms']:.0f}) │")
    print(f"  └─────────────────────────────────────────────────┘")


if __name__ == "__main__":
    main()
