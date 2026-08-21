"""
preprocessing.py v4.1 — Graph features com grafo temporal incremental

Correção sobre v4.0:
  - Graph features calculadas com grafo incremental (cada tx só vê tx anteriores)
  - Community IDs substituídos por community_size (feature numérica com semântica)
  - Removidas: sender_community, receiver_community, same_community (noise/NZV)
  - Features de valor normalizadas: ratios em vez de absolutos
  - Novas features derivadas: sender_value_ratio_to_receiver, degree_ratio

Pipeline:
  [1] Load & merge (normais + fraudes)
  [2] Standardize columns + clean + dedup
  [3] Feature engineering completa
  [4] Fix leakage temporal (rolling 90d por CPF)
  [5] Graph Feature Engineering TEMPORAL (grafo incremental)
  [6] Seleção de colunas finais
  [7] PixPreprocessor fit/transform
  [8] Salvar CSV + joblib

Autor: AI Engineer + Adilio
Data: 2026-04-12
"""

import logging
import os
import re
import time

import joblib
import networkx as nx
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DADOS_DIR = os.path.join(PROJECT_ROOT, "dados")
ARTEFACT_DIR = os.path.join(PROJECT_ROOT, "backend", "artefatos")

PATH_PIX_NORMAL = os.path.join(DADOS_DIR, "dados_pix_normais_optimized.csv")
PATH_PIX_FRAUD = os.path.join(DADOS_DIR, "dados_pix_fraudes_optimized.csv")

OUTPUT_MODEL_READY = os.path.join(DADOS_DIR, "base_treino_final.csv")
OUTPUT_PREPROCESSOR = os.path.join(ARTEFACT_DIR, "preprocessing.joblib")
OUTPUT_DIAGNOSTICO = os.path.join(ARTEFACT_DIR, "diagnostico_features.csv")

os.makedirs(DADOS_DIR, exist_ok=True)
os.makedirs(ARTEFACT_DIR, exist_ok=True)

RANDOM_STATE = 42
NULL_THRESHOLD = 0.95
ROLLING_WINDOW_SECONDS = 90 * 86400  # 90 dias em segundos
GRAPH_WINDOW_SECONDS = 90 * 86400    # 90 dias para janela do grafo

# ═══════════════════════════════════════════════════════════════════════
# FEATURES A REMOVER DO MODELO LGBM
# ═══════════════════════════════════════════════════════════════════════
FEATURES_TO_DROP_FROM_MODEL = [
    # --- GRUPO ORIGINAL: Placeholders, redundantes, gain=0 ---
    "rule_ratio_pix_limite_score",
    "autorizacao_previa_flag",
    "rule_pre_authorization_discount",
    "is_elderly_flag",
    "is_new_customer_flag",
    "rule_pix_30m_score",
    "rule_night_score",
    "receiver_missing_flag",
    "pix_key_missing_flag",
    "pix_key_type_missing_flag",
    "session_missing_flag",
    "ip_missing_flag",
    "app_version_major",
    "is_weekend",
    "is_night",
    "processamento_host_alto_flag",
    "pix_freq_high_flag",
    "period_of_day",
    # --- Duplicatas exatas (correlação Spearman = 1.0) ---
    "log_vl_pix",
    "topaz_score_filled",
    "rule_score_normalized",
    "latencia_missing_flag",
    "rule_velocity_score",
    "rule_mule_account_score",
    # --- Score = 0.000 em todos os testes ---
    "qt_dependentes",
    "is_login_biometria_flag",
    "topaz_rejeitada_flag",
    # --- Near-Zero Variance + Permutation ≤ 0 ---
    "is_login_senha_flag",
    "is_agendamento_recorrente_flag",
    "pix_key_missing_flag_derived",
    "metodo_auth_encoded",
    "tempo_interacao_missing_flag",
    "app_version_missing_flag",
    "app_version_minor",
    "auth_method_missing_flag",
    "pix_key_email_flag",
    "pix_key_document_flag",
    "pix_key_other_flag",
    "day_of_week",
    "is_business_hours",
    "pix_over_100pct_renda_flag",
    "latencia_rede_ms_final",
    "tempo_processamento_host_ms",
    "latencia_host_ratio",
    "receiver_document_same_as_customer_flag",
    "is_sexo_feminino_flag",
]


# ═══════════════════════════════════════════════════════════════════════
# HELPERS — Limpeza e normalização
# ═══════════════════════════════════════════════════════════════════════
NULL_STRINGS = {"", "null", "none", "nan", "nat", "missing", "informação ausente"}


def normalize_colname(col: str) -> str:
    """Remove prefixo de tabela e normaliza nome de coluna."""
    col = str(col).strip()
    if "." in col:
        col = col.split(".")[-1]
    return col.strip().lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de colunas do DataFrame."""
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]
    return df


def ensure_column(df: pd.DataFrame, col: str, default=np.nan) -> pd.DataFrame:
    """Garante que coluna existe no DataFrame."""
    if col not in df.columns:
        df[col] = default
    return df


def normalize_text_value(x):
    """Normaliza valor textual — converte strings nulas para NaN."""
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x.lower() in NULL_STRINGS:
        return np.nan
    return x


def clean_text_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Aplica normalização textual em colunas especificadas."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(normalize_text_value)
    return df


def safe_to_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Converte colunas para numérico de forma segura."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def safe_to_datetime(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Converte colunas para datetime de forma segura."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def replace_sentinels_with_nan(
    df: pd.DataFrame, cols: list, sentinels: list | None = None,
) -> pd.DataFrame:
    """Substitui valores sentinela (-1) por NaN."""
    if sentinels is None:
        sentinels = [-1, -1.0]
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(sentinels, np.nan)
    return df


def replace_zero_with_nan(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Substitui zeros por NaN em colunas especificadas."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)
    return df


def robust_divide(a, b):
    """Divisão segura — retorna NaN quando denominador é 0 ou NaN."""
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.where((pd.isna(a)) | (pd.isna(b)) | (b == 0), np.nan, a / b)


def _numeric_feature(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """Retorna coluna numérica alinhada ao índice, com fallback para default."""
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _safe_log1p(series: pd.Series) -> pd.Series:
    """log1p para valores não negativos, preservando estabilidade de NaN."""
    return np.log1p(series.fillna(0).clip(lower=0))


def create_trust_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria scores explicáveis de confiança/reputação pagador-recebedor.

    As entradas usadas são históricas e pré-evento no dataset v3. Quando uma
    coluna ainda não existe no contrato runtime, a feature cai para fallback
    conservador sem quebrar inferência.
    """
    df = df.copy()

    payer_count_180 = _numeric_feature(df, "qtd_pix_pagador_180d")
    payer_value_180 = _numeric_feature(df, "valor_total_pagador_180d")
    payer_max_180 = _numeric_feature(df, "valor_maximo_pix_pagador_180d")
    receiver_count_180 = _numeric_feature(df, "qtd_pix_recebidos_180d")
    receiver_value_180 = _numeric_feature(df, "valor_total_recebido_180d")
    receiver_distinct_payers = _numeric_feature(df, "soma_pagadores_distintos_dia_recebedor_180d")
    pair_count_180 = _numeric_feature(df, "qtd_pix_mesmo_recebedor_180d")
    pair_value_180 = _numeric_feature(df, "valor_total_para_recebedor_180d")
    pair_days = _numeric_feature(df, "dias_desde_primeiro_envio_recebedor")
    value = _numeric_feature(df, "vl_pix")
    ratio_payer_mean = _numeric_feature(df, "ratio_valor_media_pagador_90d")
    lgbm = _numeric_feature(df, "lgbm_raw")

    if "first_receiver_flag_real" in df.columns:
        first_receiver = _numeric_feature(df, "first_receiver_flag_real").fillna(1)
    else:
        first_receiver = _numeric_feature(df, "first_receiver_flag").fillna(1)

    df["payer_history_strength_score"] = (
        _safe_log1p(payer_count_180) * 12.0
        + _safe_log1p(payer_value_180) * 4.0
        + _safe_log1p(payer_max_180) * 3.0
    ).clip(0, 100)

    df["receiver_reputation_score"] = (
        _safe_log1p(receiver_count_180) * 14.0
        + _safe_log1p(receiver_value_180) * 4.0
        + _safe_log1p(receiver_distinct_payers) * 12.0
    ).clip(0, 100)

    df["relationship_strength_score"] = (
        _safe_log1p(pair_count_180) * 22.0
        + _safe_log1p(pair_value_180) * 4.0
        + np.minimum(pair_days.fillna(0).clip(lower=0), 180.0) / 180.0 * 30.0
    ).clip(0, 100)

    df["receiver_novelty_risk_score"] = (
        (first_receiver == 1).astype(float) * 35.0
        + (receiver_count_180.fillna(0) <= 0).astype(float) * 30.0
        + (receiver_value_180.fillna(0) <= 0).astype(float) * 20.0
        + (pair_count_180.fillna(0) <= 0).astype(float) * 15.0
    ).clip(0, 100)

    df["transaction_normality_score"] = (
        100.0
        - np.minimum(ratio_payer_mean.fillna(0).clip(lower=0), 25.0) * 2.4
        - np.minimum(value.fillna(0).clip(lower=0) / 1000.0, 30.0)
        - np.minimum(lgbm.fillna(0).clip(lower=0) * 300.0, 60.0)
    ).clip(0, 100)

    df["payer_receiver_trust_score"] = (
        df["payer_history_strength_score"] * 0.25
        + df["receiver_reputation_score"] * 0.30
        + df["relationship_strength_score"] * 0.30
        + df["transaction_normality_score"] * 0.15
        - df["receiver_novelty_risk_score"] * 0.35
    ).clip(0, 100)

    df["trust_bucket"] = pd.cut(
        df["payer_receiver_trust_score"],
        bins=[-0.01, 20, 40, 60, 80, 100],
        labels=["trust_00_20", "trust_20_40", "trust_40_60", "trust_60_80", "trust_80_100"],
    ).astype(str)
    df["receiver_rep_bucket"] = pd.cut(
        df["receiver_reputation_score"],
        bins=[-0.01, 20, 40, 60, 80, 100],
        labels=["rep_00_20", "rep_20_40", "rep_40_60", "rep_60_80", "rep_80_100"],
    ).astype(str)
    df["relationship_bucket"] = pd.cut(
        df["relationship_strength_score"],
        bins=[-0.01, 20, 40, 60, 80, 100],
        labels=["rel_00_20", "rel_20_40", "rel_40_60", "rel_60_80", "rel_80_100"],
    ).astype(str)
    df["novelty_bucket"] = pd.cut(
        df["receiver_novelty_risk_score"],
        bins=[-0.01, 20, 40, 60, 80, 100],
        labels=["nov_00_20", "nov_20_40", "nov_40_60", "nov_60_80", "nov_80_100"],
    ).astype(str)

    return df


def normalize_device_name(x):
    """Normaliza nome de dispositivo."""
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    if x in NULL_STRINGS:
        return np.nan
    return re.sub(r"\s+", "_", x)


def extract_app_version_minor(version):
    """Extrai versão minor do app (ex: '5.3.1' → 3.0)."""
    if pd.isna(version):
        return np.nan
    parts = str(version).strip().split(".")
    if len(parts) >= 2 and parts[1].isdigit():
        return float(parts[1])
    return np.nan


def map_topaz_rule(score):
    """Mapeia topaz_risk_score para score discreto."""
    if pd.isna(score):
        return 0
    s = float(score)
    if s <= 1:
        return 0
    elif s <= 2:
        return 2
    elif s <= 3:
        return 3
    elif s <= 4:
        return 4
    elif s <= 5:
        return 5
    elif s < 20:
        return 2
    elif s < 40:
        return 3
    elif s < 60:
        return 4
    return 5


def normalize_transaction_key(series: pd.Series) -> pd.Series:
    """Normaliza chave de transação (cd_pix)."""
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\s+", "", regex=True)
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return s


def count_non_null_priority(df: pd.DataFrame, cols: list) -> pd.Series:
    """Conta colunas não-nulas para priorização na dedup."""
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return pd.Series(0, index=df.index)
    return df[valid_cols].notna().sum(axis=1)


def deduplicate_by_key(
    df: pd.DataFrame,
    key_col: str,
    priority_cols: list,
    extra_priority_cols: list | None = None,
) -> pd.DataFrame:
    """Deduplica por chave, priorizando linhas com mais dados preenchidos."""
    df = df.copy()
    if extra_priority_cols is None:
        extra_priority_cols = []
    df["_nn"] = df.notna().sum(axis=1)
    df["_pr"] = count_non_null_priority(df, priority_cols)
    for c in extra_priority_cols:
        if c not in df.columns:
            df[c] = 0
    sort_cols = [key_col, "_nn", "_pr"] + extra_priority_cols
    ascending = [True, False, False] + [False] * len(extra_priority_cols)
    df = df.sort_values(sort_cols, ascending=ascending)
    df = df.drop_duplicates(subset=[key_col], keep="first").copy()
    df = df.drop(columns=["_nn", "_pr"], errors="ignore")
    return df


def classify_key_flags(ds_tipo_chave_series: pd.Series) -> pd.DataFrame:
    """Classifica tipo de chave PIX em flags binárias."""
    s = ds_tipo_chave_series.astype(str).str.upper()
    out = pd.DataFrame(index=ds_tipo_chave_series.index)
    out["pix_key_random_flag"] = (s == "CHAVE ALEATORIA").astype(int)
    out["pix_key_email_flag"] = (s == "EMAIL").astype(int)
    out["pix_key_document_flag"] = (s == "DOCUMENTO/TELEFONE").astype(int)
    out["pix_key_other_flag"] = (s == "OUTROS").astype(int)
    out["pix_key_missing_flag_derived"] = s.isin(
        ["NAN", "INFORMAÇÃO AUSENTE"]
    ).astype(int)
    return out


def cumulative_distinct_count(series: pd.Series) -> pd.Series:
    """Contagem cumulativa de valores distintos."""
    seen: set = set()
    counts = []
    for val in series:
        if pd.notna(val):
            seen.add(val)
        counts.append(len(seen))
    return pd.Series(counts, index=series.index)


def encode_metodo_autenticacao(x) -> int:
    """Codifica método de autenticação como inteiro."""
    if pd.isna(x):
        return 3
    val = str(x).strip().lower()
    if val in ("1", "bio", "biometria", "biometric"):
        return 0
    if val in ("2", "senha", "password"):
        return 1
    if val in ("3", "pin"):
        return 2
    if "bio" in val:
        return 0
    if "senha" in val:
        return 1
    if "pin" in val:
        return 2
    return 3


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1 — LOAD & MERGE
# ═══════════════════════════════════════════════════════════════════════
def load_and_prepare_pix(path_normal: str, path_fraud: str) -> pd.DataFrame:
    """Carrega CSVs brutos, unifica, limpa e deduplica.

    Args:
        path_normal: Caminho para CSV de transações normais.
        path_fraud: Caminho para CSV de transações fraudulentas.

    Returns:
        DataFrame unificado, limpo e deduplicado por cd_pix.
    """
    logger.info("Carregando PIX normais: %s", path_normal)
    pix_normal = standardize_columns(
        pd.read_csv(path_normal, low_memory=False, encoding="latin-1")
    )
    logger.info("  → %d linhas, %d colunas", len(pix_normal), len(pix_normal.columns))

    logger.info("Carregando PIX fraudes: %s", path_fraud)
    pix_fraud = standardize_columns(
        pd.read_csv(path_fraud, low_memory=False, encoding="latin-1")
    )
    logger.info("  → %d linhas, %d colunas", len(pix_fraud), len(pix_fraud.columns))

    pix_normal["is_fraud"] = 0
    pix_normal["source_dataset"] = "normal"

    if "is_fraud" in pix_fraud.columns:
        pix_fraud["is_fraud"] = pd.to_numeric(
            pix_fraud["is_fraud"], errors="coerce"
        ).fillna(1).astype(int)
    elif "tp_fraude" in pix_fraud.columns:
        pix_fraud["is_fraud"] = pd.to_numeric(
            pix_fraud["tp_fraude"], errors="coerce"
        ).fillna(1).astype(int)
    else:
        pix_fraud["is_fraud"] = 1
    pix_fraud["source_dataset"] = "fraud"

    required_cols = [
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
        "qt_aparelhos_distintos_trimestre", "nr_idade", "qt_tempo_relacionamento_mes",
        "ds_sexo", "ds_estado_civil", "ds_segmento",
        "tp_primeiro_envio_recebedor_trimestre", "qt_envio_recebedor_trimestre",
        "vl_renda_cliente",
        "is_fraud", "source_dataset", "dt_carga",
    ]

    for c in required_cols:
        pix_normal = ensure_column(pix_normal, c)
        pix_fraud = ensure_column(pix_fraud, c)

    pix_all = pd.concat(
        [pix_normal[required_cols], pix_fraud[required_cols]],
        ignore_index=True, sort=False,
    )

    text_cols = [
        "cd_pix", "cd_cpf_pagador", "cd_cpf_cnpj_recebedor",
        "ds_chave_pix", "ds_tipo_chave", "device_name", "app_version",
        "ip_address", "metodo_autenticacao", "session_id", "cd_retorno",
        "source_dataset", "ds_sexo", "ds_estado_civil", "ds_segmento",
    ]
    pix_all = clean_text_columns(pix_all, text_cols)

    num_cols = [
        "vl_pix", "qt_total_pix_trimestre", "vl_mediana_pix_trimestre",
        "vl_desvio_padrao_pix_trimestre", "qt_intervalo_transacao_minuto",
        "qt_intervalo_mediana_trimestre", "qt_intervalo_desvio_padrao_trimestre",
        "qt_pix_dia_maximo_trimestre", "latencia_rede_ms",
        "vl_latencia_rede_media_trimestre", "tempo_interacao_ms",
        "vl_tempo_interacao_medio_trimestre", "tempo_processamento_host_ms",
        "topaz_risk_score", "topaz_transacao_rejeitada",
        "qt_aparelhos_distintos_trimestre",
        "nr_idade", "qt_tempo_relacionamento_mes", "is_fraud",
        "tp_primeiro_envio_recebedor_trimestre", "qt_envio_recebedor_trimestre",
        "vl_renda_cliente",
    ]
    pix_all = safe_to_numeric(pix_all, num_cols)
    pix_all = safe_to_datetime(pix_all, ["dt_pix", "dt_carga"])

    pix_all["cd_pix"] = normalize_transaction_key(pix_all["cd_pix"])
    pix_all["cd_cpf_pagador"] = pix_all["cd_cpf_pagador"].astype("object")
    pix_all["cd_cpf_cnpj_recebedor"] = pix_all["cd_cpf_cnpj_recebedor"].astype("object")

    sentinel_cols = [
        "latencia_rede_ms", "vl_latencia_rede_media_trimestre",
        "tempo_interacao_ms", "vl_tempo_interacao_medio_trimestre",
        "tempo_processamento_host_ms", "topaz_risk_score",
        "topaz_transacao_rejeitada",
    ]
    pix_all = replace_sentinels_with_nan(pix_all, sentinel_cols)
    pix_all = replace_zero_with_nan(pix_all, ["vl_tempo_interacao_medio_trimestre"])

    logger.info("Deduplicando PIX por cd_pix...")
    pix_priority = [
        "cd_cpf_cnpj_recebedor", "ds_chave_pix", "ds_tipo_chave",
        "device_name", "app_version", "ip_address", "latencia_rede_ms",
        "tempo_interacao_ms", "tempo_processamento_host_ms",
        "metodo_autenticacao", "session_id", "cd_retorno",
        "topaz_risk_score", "topaz_transacao_rejeitada",
        "ds_sexo", "ds_estado_civil", "ds_segmento",
        "tp_primeiro_envio_recebedor_trimestre",
    ]
    pix_all = deduplicate_by_key(pix_all, "cd_pix", pix_priority)
    logger.info("  → %d transações únicas", len(pix_all))
    return pix_all


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════
def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria TODAS as features derivadas a partir dos dados brutos limpos.

    Args:
        df: DataFrame limpo e deduplicado.

    Returns:
        DataFrame com todas as features criadas.
    """
    df = df.copy()

    df["transaction_id"] = df["cd_pix"]
    df["customer_id"] = df["cd_cpf_pagador"]
    df["event_datetime"] = df["dt_pix"]
    df = df[df["transaction_id"].notna()].copy()

    df["latencia_rede_ms_final"] = df["latencia_rede_ms"]

    # --- MISSING FLAGS ---
    df["device_missing_flag"] = df["device_name"].isna().astype(int)
    df["app_version_missing_flag"] = df["app_version"].isna().astype(int)
    df["auth_method_missing_flag"] = df["metodo_autenticacao"].isna().astype(int)
    df["topaz_missing_flag"] = df["topaz_risk_score"].isna().astype(int)
    df["host_time_missing_flag"] = df["tempo_processamento_host_ms"].isna().astype(int)
    df["latencia_missing_flag"] = df["latencia_rede_ms_final"].isna().astype(int)
    df["tempo_interacao_missing_flag"] = df["tempo_interacao_ms"].isna().astype(int)

    # --- DEVICE / APP ---
    df["device_name_normalized"] = df["device_name"].apply(normalize_device_name)
    df["app_version_minor"] = df["app_version"].apply(extract_app_version_minor)

    # --- KEY FLAGS ---
    df["ds_tipo_chave"] = df["ds_tipo_chave"].apply(normalize_text_value)
    df["ds_chave_pix"] = df["ds_chave_pix"].apply(normalize_text_value)
    key_flags = classify_key_flags(df["ds_tipo_chave"].fillna("Informação ausente"))
    for c in key_flags.columns:
        df[c] = key_flags[c].values

    df["receiver_document_same_as_customer_flag"] = (
        df["customer_id"].notna()
        & df["cd_cpf_cnpj_recebedor"].notna()
        & (df["customer_id"].astype(str) == df["cd_cpf_cnpj_recebedor"].astype(str))
    ).astype(int)

    # --- TEMPORAL ---
    df["hour"] = df["event_datetime"].dt.hour
    df["day_of_week"] = df["event_datetime"].dt.dayofweek
    df["is_business_hours"] = (
        df["hour"].between(8, 18, inclusive="both").fillna(False).astype(int)
    )

    # --- CORE DERIVED (valor) ---
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

    # --- CORE DERIVED (intervalo) ---
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

    # --- LATÊNCIA ---
    df["ratio_latencia_cliente"] = robust_divide(
        df["latencia_rede_ms_final"], df["vl_latencia_rede_media_trimestre"]
    )
    df["diff_latencia_cliente"] = (
        df["latencia_rede_ms_final"] - df["vl_latencia_rede_media_trimestre"]
    )
    df["latencia_host_ratio"] = robust_divide(
        df["latencia_rede_ms_final"], df["tempo_processamento_host_ms"]
    )

    # --- TOPAZ ---
    df["topaz_score_filled"] = df["topaz_risk_score"].fillna(0)

    # --- FLAGS ---
    df["vl_pix_over_1000_flag"] = (df["vl_pix"] >= 1000).astype(int)
    df["is_first_tx_trimestre"] = (df["qt_total_pix_trimestre"] == 1).astype(int)

    # --- RENDA ---
    df["vl_renda_cliente"] = pd.to_numeric(df["vl_renda_cliente"], errors="coerce").fillna(0)
    df["qt_dependentes"] = 0

    df["ratio_pix_renda"] = np.where(
        df["vl_renda_cliente"] > 0,
        df["vl_pix"] / df["vl_renda_cliente"],
        np.nan,
    )
    df["pix_over_50pct_renda_flag"] = np.where(
        df["vl_renda_cliente"] > 0,
        (df["vl_pix"] > df["vl_renda_cliente"] * 0.5).astype(int),
        0,
    )
    df["pix_over_100pct_renda_flag"] = np.where(
        df["vl_renda_cliente"] > 0,
        (df["vl_pix"] > df["vl_renda_cliente"]).astype(int),
        0,
    )
    df["renda_missing_flag"] = (df["vl_renda_cliente"] <= 0).astype(int)

    # --- TEMPO DE INTERAÇÃO ---
    df["tempo_interacao_ms_final"] = df["tempo_interacao_ms"]
    df["ratio_tempo_interacao_cliente"] = robust_divide(
        df["tempo_interacao_ms_final"], df["vl_tempo_interacao_medio_trimestre"]
    )
    df["diff_tempo_interacao_cliente"] = (
        df["tempo_interacao_ms_final"] - df["vl_tempo_interacao_medio_trimestre"]
    )

    # --- MÉTODO DE AUTENTICAÇÃO ---
    df["metodo_auth_encoded"] = df["metodo_autenticacao"].apply(encode_metodo_autenticacao)
    df["is_login_senha_flag"] = (df["metodo_auth_encoded"] == 1).astype(int)
    df["is_login_biometria_flag"] = (df["metodo_auth_encoded"] == 0).astype(int)

    # --- AGENDAMENTO ---
    df["is_agendamento_recorrente"] = np.nan
    df["is_agendamento_recorrente_flag"] = 0

    # --- TOPAZ REJEITADA ---
    df["topaz_rejeitada_flag"] = (
        pd.to_numeric(df["topaz_transacao_rejeitada"], errors="coerce").fillna(0) == 1
    ).astype(int)

    # --- SEXO ---
    df["is_sexo_feminino_flag"] = (
        df["ds_sexo"].astype(str).str.strip().str.upper() == "F"
    ).astype(int)

    # --- ESTADO CIVIL ---
    df["is_viuvo_flag"] = (
        df["ds_estado_civil"].astype(str).str.strip().str.upper().str.contains(
            "VIUV", na=False,
        )
    ).astype(int)

    # --- SEGMENTO ---
    _seg = df["ds_segmento"].astype(str).str.strip().str.upper()
    df["is_segmento_premium_flag"] = _seg.isin([
        "EXCLUSIVO", "PRIVATE", "MILLENIUM", "MILLENIUM CAPIT", "PREMIUM", "VIP",
    ]).astype(int)

    # --- PRIMEIRO ENVIO AO RECEBEDOR ---
    df["tp_primeiro_envio_recebedor_trimestre"] = (
        pd.to_numeric(df["tp_primeiro_envio_recebedor_trimestre"], errors="coerce")
        .fillna(0).astype(int)
    )

    # --- QT ENVIO RECEBEDOR ---
    df["qt_envio_recebedor_trimestre"] = (
        pd.to_numeric(df["qt_envio_recebedor_trimestre"], errors="coerce").fillna(0)
    )

    # --- PERFIL VULNERÁVEL ---
    df["perfil_vulneravel_se_flag"] = (
        (df["is_viuvo_flag"] == 1)
        & (df["nr_idade"] >= 60)
        & (df["qt_dependentes"] == 0)
    ).astype(int)

    # --- SEQUENCIAIS POR CLIENTE ---
    logger.info("Criando features sequenciais por cliente...")
    df = df.sort_values(
        ["customer_id", "event_datetime", "transaction_id"]
    ).reset_index(drop=True)

    df["prev_event_datetime"] = df.groupby("customer_id")["event_datetime"].shift(1)
    df["minutes_since_prev_tx"] = (
        (df["event_datetime"] - df["prev_event_datetime"]).dt.total_seconds() / 60.0
    )

    logger.info("  Calculando tx_count_prev_30m (janela 30min)...")
    df["tx_count_prev_30m"] = 0
    for _, group in df.groupby("customer_id", sort=False):
        idx = group.index.to_list()
        times = group["event_datetime"].tolist()
        counts = []
        for i, ct in enumerate(times):
            if pd.isna(ct):
                counts.append(0)
                continue
            c = 0
            j = i - 1
            while j >= 0:
                if pd.isna(times[j]):
                    j -= 1
                    continue
                if (ct - times[j]).total_seconds() / 60.0 <= 30:
                    c += 1
                    j -= 1
                else:
                    break
            counts.append(c)
        df.loc[idx, "tx_count_prev_30m"] = counts

    df["burst_30m_flag"] = (df["tx_count_prev_30m"] >= 1).astype(int)

    df["receiver_tx_count_prev"] = df.groupby(
        ["customer_id", "cd_cpf_cnpj_recebedor"]
    ).cumcount()
    df["first_receiver_flag"] = (df["receiver_tx_count_prev"] == 0).astype(int)

    df["key_tx_count_prev"] = df.groupby(
        ["customer_id", "ds_chave_pix"]
    ).cumcount()
    df["first_key_flag"] = (df["key_tx_count_prev"] == 0).astype(int)

    df["distinct_receivers_so_far"] = (
        df.groupby("customer_id", sort=False)["cd_cpf_cnpj_recebedor"]
        .transform(cumulative_distinct_count)
    )
    df["distinct_keys_so_far"] = (
        df.groupby("customer_id", sort=False)["ds_chave_pix"]
        .transform(cumulative_distinct_count)
    )

    # --- REGRAS HEURÍSTICAS ---
    df = create_trust_features(df)

    df["rule_age_score"] = np.select(
        [df["nr_idade"].between(60, 65), df["nr_idade"].between(66, 75), df["nr_idade"] >= 76],
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
        [df["first_receiver_flag"] == 1, df["receiver_document_same_as_customer_flag"] == 1],
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
            df["tx_count_prev_30m"] == 0, df["tx_count_prev_30m"] == 1,
            df["tx_count_prev_30m"] == 2, df["tx_count_prev_30m"] >= 3,
        ],
        [0, 2, 3, 4], default=0,
    )
    df["rule_topaz_score"] = df["topaz_risk_score"].apply(map_topaz_rule)

    rule_components = [
        "rule_age_score", "rule_relationship_score",
        "rule_mule_account_score", "rule_random_key_score",
        "rule_velocity_score", "rule_topaz_score",
    ]
    df["rule_score_raw"] = df[rule_components].fillna(0).sum(axis=1)
    max_rs = df["rule_score_raw"].max(skipna=True)
    df["rule_score_normalized"] = (
        df["rule_score_raw"] / max_rs if (not pd.isna(max_rs) and max_rs > 0) else 0.0
    )

    df = df.drop(columns=["prev_event_datetime"], errors="ignore")
    return df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 — FIX LEAKAGE TEMPORAL (rolling 90d por CPF)
# ═══════════════════════════════════════════════════════════════════════
def compute_rolling_for_group(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    intervals: np.ndarray,
    dates: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calcula features rolling para um CPF já ordenado por timestamp.

    Para cada tx_i, usa apenas tx [0..i-1] dentro de 90 dias.

    Args:
        timestamps_s: Timestamps em epoch seconds (int64).
        values: vl_pix por transação.
        intervals: qt_intervalo_transacao_minuto por transação.
        dates: Datas (datetime64[D]) para cálculo de daily max.

    Returns:
        Dict com arrays de mesma length para cada feature corrigida.
    """
    n = len(timestamps_s)

    qt_total = np.zeros(n, dtype=np.int64)
    vl_mediana = np.full(n, np.nan, dtype=np.float64)
    vl_desvio = np.full(n, np.nan, dtype=np.float64)
    qt_dia_max = np.zeros(n, dtype=np.int64)
    intervalo_med = np.full(n, 0.0, dtype=np.float64)
    intervalo_dev = np.full(n, 0.0, dtype=np.float64)

    for i in range(1, n):
        time_diffs = timestamps_s[i] - timestamps_s[:i]
        mask = time_diffs <= ROLLING_WINDOW_SECONDS

        if not mask.any():
            continue

        w_vals = values[:i][mask]
        count = len(w_vals)

        qt_total[i] = count
        vl_mediana[i] = np.median(w_vals)

        if count >= 2:
            vl_desvio[i] = np.std(w_vals, ddof=1)

        w_dates = dates[:i][mask]
        unique_dates, date_counts = np.unique(w_dates, return_counts=True)
        if len(date_counts) > 0:
            qt_dia_max[i] = date_counts.max()

        w_intervals = intervals[:i][mask]
        valid = w_intervals[w_intervals > 0]

        if len(valid) >= 1:
            intervalo_med[i] = np.median(valid)
        if len(valid) >= 2:
            intervalo_dev[i] = np.std(valid, ddof=1)

    return {
        "qt_total_pix_trimestre": qt_total,
        "vl_mediana_pix_trimestre": vl_mediana,
        "vl_desvio_padrao_pix_trimestre": vl_desvio,
        "qt_pix_dia_maximo_trimestre": qt_dia_max,
        "qt_intervalo_mediana_trimestre": intervalo_med,
        "qt_intervalo_desvio_padrao_trimestre": intervalo_dev,
    }


def fix_leakage_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige leakage temporal nas 6 features trimestrais + derivadas.

    Args:
        df: DataFrame com features já criadas.

    Returns:
        DataFrame com features temporais corrigidas.
    """
    logger.info("Iniciando fix de leakage temporal (rolling 90d por CPF)...")
    df = df.copy()
    df = df.sort_values(["customer_id", "event_datetime"]).reset_index(drop=True)

    df["_ts_epoch"] = (
        df["event_datetime"].astype("int64") // 10**9
    ).astype("int64")
    df["_date"] = df["event_datetime"].dt.date

    result_cols = [
        "qt_total_pix_trimestre",
        "vl_mediana_pix_trimestre",
        "vl_desvio_padrao_pix_trimestre",
        "qt_pix_dia_maximo_trimestre",
        "qt_intervalo_mediana_trimestre",
        "qt_intervalo_desvio_padrao_trimestre",
    ]

    t0 = time.time()
    groups = df.groupby("customer_id", sort=False)
    n_groups = groups.ngroups
    processed = 0

    for cid, idx in groups.groups.items():
        group = df.loc[idx].sort_values("event_datetime")
        gi = group.index.values

        ts = df.loc[gi, "_ts_epoch"].values
        vals = df.loc[gi, "vl_pix"].values.astype(np.float64)
        intv = df.loc[gi, "qt_intervalo_transacao_minuto"].values.astype(np.float64)
        dates = df.loc[gi, "_date"].values

        results = compute_rolling_for_group(ts, vals, intv, dates)

        for col_name, arr in results.items():
            df.loc[gi, col_name] = arr

        processed += 1
        if processed % 5000 == 0 or processed == n_groups:
            elapsed = time.time() - t0
            rate = processed / max(elapsed, 0.01)
            eta = (n_groups - processed) / max(rate, 0.01)
            logger.info(
                "  %d/%d CPFs (%d%%) — %.0fs, ~%.0fs restantes",
                processed, n_groups, int(processed / n_groups * 100),
                elapsed, eta,
            )

    elapsed_total = time.time() - t0
    logger.info("  ✅ Rolling features calculadas em %.1fs", elapsed_total)

    df = df.drop(columns=["_ts_epoch", "_date"], errors="ignore")

    # --- Recalcular features derivadas ---
    logger.info("Recalculando features derivadas pós-leakage fix...")

    vl = df["vl_pix"].values
    med = df["vl_mediana_pix_trimestre"].values
    dev = df["vl_desvio_padrao_pix_trimestre"].values
    intv = df["qt_intervalo_transacao_minuto"].values
    intv_med = df["qt_intervalo_mediana_trimestre"].values
    intv_dev = df["qt_intervalo_desvio_padrao_trimestre"].values

    df["is_first_tx_trimestre"] = (df["qt_total_pix_trimestre"] == 0).astype(int)

    with np.errstate(divide="ignore", invalid="ignore"):
        df["ratio_valor_mediana"] = np.where(
            (med == 0) | np.isnan(med), np.nan, vl / med,
        )
        df["diff_valor_mediana"] = vl - med
        df["ratio_valor_desvio_padrao"] = np.where(
            (dev == 0) | np.isnan(dev), np.nan, vl / dev,
        )
        df["zscore_valor_aprox"] = np.where(
            (dev == 0) | np.isnan(dev), np.nan, (vl - med) / dev,
        )
        df["ratio_intervalo_vs_mediana"] = np.where(
            (intv_med == 0) | np.isnan(intv_med), np.nan, intv / intv_med,
        )
        df["diff_intervalo_vs_mediana"] = intv - intv_med
        df["zscore_intervalo_aprox"] = np.where(
            (intv_dev == 0) | np.isnan(intv_dev), np.nan,
            (intv - intv_med) / intv_dev,
        )

    for col in result_cols:
        df[col] = df[col].fillna(0)

    return df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4 — GRAPH FEATURE ENGINEERING (TEMPORAL INCREMENTAL)
# ═══════════════════════════════════════════════════════════════════════
def compute_temporal_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula graph features com grafo incremental temporal.

    Para cada transação t_i, constrói o grafo usando APENAS transações
    t_0..t_{i-1} dentro de uma janela de 90 dias. Sem leakage.

    Features calculadas (13 total):
      Sender (pagador):
        - sender_out_degree: nº recebedores distintos nas tx anteriores
        - sender_tx_count: nº de tx anteriores do pagador na janela
        - sender_avg_value: valor médio por tx anterior do pagador
        - sender_value_zscore: z-score do valor vs histórico do pagador
      Receiver (recebedor):
        - receiver_in_degree: nº pagadores distintos que enviaram pro recebedor
        - receiver_tx_count: nº tx recebidas pelo recebedor na janela
        - receiver_avg_value: valor médio recebido por tx
      Pair (par pagador-recebedor):
        - pair_tx_count: nº tx anteriores deste par
        - is_new_edge: par nunca transacionou antes na janela (0/1)
      Ratios (derivadas sem leakage):
        - sender_value_ratio: vl_pix / sender_avg_value
        - receiver_value_ratio: vl_pix / receiver_avg_value
        - receiver_concentration_hhi: Herfindahl do pagador
        - degree_ratio: sender_out_degree / receiver_in_degree

    Args:
        df: DataFrame ordenado por event_datetime com colunas:
            transaction_id, customer_id, cd_cpf_cnpj_recebedor,
            vl_pix, event_datetime.

    Returns:
        DataFrame com transaction_id + 13 graph features.
    """
    logger.info("Calculando graph features TEMPORAIS (grafo incremental)...")
    t0 = time.perf_counter()

    df = df.sort_values("event_datetime").reset_index(drop=True)
    n = len(df)

    ts_epoch = (df["event_datetime"].astype("int64") // 10**9).values
    tx_ids = df["transaction_id"].values
    senders = df["customer_id"].values
    receivers = df["cd_cpf_cnpj_recebedor"].values
    values = df["vl_pix"].values.astype(np.float64)

    # Pré-alocar arrays de resultado
    sender_out_degree = np.zeros(n, dtype=np.int32)
    sender_tx_count = np.zeros(n, dtype=np.int32)
    sender_avg_value = np.zeros(n, dtype=np.float64)
    sender_value_zscore = np.zeros(n, dtype=np.float64)

    receiver_in_degree = np.zeros(n, dtype=np.int32)
    receiver_tx_count = np.zeros(n, dtype=np.int32)
    receiver_avg_value = np.zeros(n, dtype=np.float64)

    pair_tx_count = np.zeros(n, dtype=np.int32)
    is_new_edge = np.ones(n, dtype=np.int32)  # default: novo par

    receiver_concentration_hhi = np.zeros(n, dtype=np.float64)

    # Estruturas incrementais
    # sender → list of (timestamp, value, receiver)
    sender_history: dict[str, list[tuple[int, float, str]]] = {}
    # receiver → list of (timestamp, value, sender)
    receiver_history: dict[str, list[tuple[int, float, str]]] = {}
    # (sender, receiver) → list of timestamps
    pair_history: dict[tuple[str, str], list[int]] = {}

    log_interval = max(1, n // 20)

    for i in range(n):
        s = senders[i]
        r = receivers[i]
        v = values[i]
        ts = ts_epoch[i]
        window_start = ts - GRAPH_WINDOW_SECONDS

        if pd.isna(s) or pd.isna(r):
            continue

        s_str = str(s)
        r_str = str(r)
        pair_key = (s_str, r_str)

        # === Consultar histórico ANTES da tx atual ===

        # Sender history (filtrar por janela)
        s_hist = sender_history.get(s_str, [])
        s_in_window = [(t, val, recv) for t, val, recv in s_hist if t >= window_start]

        if s_in_window:
            s_vals = [val for _, val, _ in s_in_window]
            s_recvs = set(recv for _, _, recv in s_in_window)

            sender_out_degree[i] = len(s_recvs)
            sender_tx_count[i] = len(s_in_window)
            s_mean = np.mean(s_vals)
            sender_avg_value[i] = s_mean

            if len(s_vals) >= 2:
                s_std = np.std(s_vals, ddof=1)
                sender_value_zscore[i] = (v - s_mean) / s_std if s_std > 0 else 0.0

            # HHI: concentração por recebedor
            recv_counts: dict[str, int] = {}
            for _, _, recv in s_in_window:
                recv_counts[recv] = recv_counts.get(recv, 0) + 1
            total_s = len(s_in_window)
            if total_s > 0:
                hhi = sum((c / total_s) ** 2 for c in recv_counts.values())
                receiver_concentration_hhi[i] = hhi

        # Receiver history (filtrar por janela)
        r_hist = receiver_history.get(r_str, [])
        r_in_window = [(t, val, sndr) for t, val, sndr in r_hist if t >= window_start]

        if r_in_window:
            r_vals = [val for _, val, _ in r_in_window]
            r_sndrs = set(sndr for _, _, sndr in r_in_window)

            receiver_in_degree[i] = len(r_sndrs)
            receiver_tx_count[i] = len(r_in_window)
            receiver_avg_value[i] = np.mean(r_vals)

        # Pair history
        p_hist = pair_history.get(pair_key, [])
        p_in_window = [t for t in p_hist if t >= window_start]
        pair_tx_count[i] = len(p_in_window)
        is_new_edge[i] = 1 if len(p_in_window) == 0 else 0

        # === Registrar tx atual no histórico ===
        if s_str not in sender_history:
            sender_history[s_str] = []
        sender_history[s_str].append((ts, v, r_str))

        if r_str not in receiver_history:
            receiver_history[r_str] = []
        receiver_history[r_str].append((ts, v, s_str))

        if pair_key not in pair_history:
            pair_history[pair_key] = []
        pair_history[pair_key].append(ts)

        if (i + 1) % log_interval == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / max(elapsed, 0.01)
            eta = (n - i - 1) / max(rate, 0.01)
            logger.info(
                "  Graph: %d/%d tx (%d%%) — %.0fs, ~%.0fs restantes",
                i + 1, n, int((i + 1) / n * 100), elapsed, eta,
            )

    # === Features derivadas (ratios) ===
    with np.errstate(divide="ignore", invalid="ignore"):
        sender_value_ratio = np.where(
            sender_avg_value > 0, values / sender_avg_value, 0.0,
        )
        receiver_value_ratio = np.where(
            receiver_avg_value > 0, values / receiver_avg_value, 0.0,
        )
        degree_ratio = np.where(
            receiver_in_degree > 0,
            sender_out_degree.astype(np.float64) / receiver_in_degree,
            0.0,
        )

    df_graph = pd.DataFrame({
        "transaction_id": tx_ids,
        # Sender features
        "graph_sender_out_degree": sender_out_degree,
        "graph_sender_tx_count": sender_tx_count,
        "graph_sender_avg_value": sender_avg_value,
        "graph_sender_value_zscore": sender_value_zscore,
        # Receiver features
        "graph_receiver_in_degree": receiver_in_degree,
        "graph_receiver_tx_count": receiver_tx_count,
        "graph_receiver_avg_value": receiver_avg_value,
        # Pair features
        "graph_pair_tx_count": pair_tx_count,
        "graph_is_new_edge": is_new_edge,
        # Derived ratios
        "graph_sender_value_ratio": sender_value_ratio,
        "graph_receiver_value_ratio": receiver_value_ratio,
        "graph_receiver_concentration_hhi": receiver_concentration_hhi,
        "graph_degree_ratio": degree_ratio,
    })

    elapsed = time.perf_counter() - t0
    logger.info(
        "  ✅ Graph features temporais: %d tx × %d features (%.1fs)",
        len(df_graph), len(df_graph.columns) - 1, elapsed,
    )

    # Estatísticas de cobertura
    has_history = (sender_tx_count > 0).sum()
    logger.info(
        "  Cobertura: %d/%d tx (%.1f%%) com histórico de sender",
        has_history, n, 100 * has_history / n,
    )
    has_recv_hist = (receiver_tx_count > 0).sum()
    logger.info(
        "  Cobertura: %d/%d tx (%.1f%%) com histórico de receiver",
        has_recv_hist, n, 100 * has_recv_hist / n,
    )
    n_new = is_new_edge.sum()
    logger.info(
        "  New edges: %d/%d (%.1f%%)",
        n_new, n, 100 * n_new / n,
    )

    return df_graph


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5 — SELEÇÃO DE COLUNAS FINAIS
# ═══════════════════════════════════════════════════════════════════════
def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona TODAS as colunas para o CSV intermediário.

    Args:
        df: DataFrame com todas as features + graph features.

    Returns:
        DataFrame com colunas selecionadas e ordenadas.
    """
    final_cols = [
        # IDs
        "transaction_id", "customer_id", "event_datetime", "source_dataset", "is_fraud",
        # Texto bruto (excluído no fit pelo exclude_from_model_)
        "cd_cpf_cnpj_recebedor", "ds_chave_pix", "ds_tipo_chave",
        "device_name", "device_name_normalized", "app_version",
        "ip_address", "metodo_autenticacao", "session_id", "cd_retorno",
        "ds_sexo", "ds_estado_civil", "ds_segmento",
        "is_agendamento_recorrente",
        # Flags chave/recebedor
        "receiver_document_same_as_customer_flag",
        "pix_key_random_flag", "pix_key_email_flag",
        "pix_key_document_flag", "pix_key_other_flag", "pix_key_missing_flag_derived",
        # Valor / histórico
        "vl_pix", "log_vl_pix", "vl_pix_over_1000_flag",
        "qt_total_pix_trimestre", "is_first_tx_trimestre",
        "vl_mediana_pix_trimestre", "vl_desvio_padrao_pix_trimestre",
        "qt_intervalo_transacao_minuto", "qt_intervalo_mediana_trimestre",
        "qt_intervalo_desvio_padrao_trimestre", "qt_pix_dia_maximo_trimestre",
        "qt_aparelhos_distintos_trimestre", "nr_idade", "qt_tempo_relacionamento_mes",
        # Latência / device
        "latencia_rede_ms_final", "vl_latencia_rede_media_trimestre",
        "tempo_processamento_host_ms",
        # Tempo de interação
        "tempo_interacao_ms_final", "vl_tempo_interacao_medio_trimestre",
        "ratio_tempo_interacao_cliente", "diff_tempo_interacao_cliente",
        # Ratios / desvios
        "ratio_valor_mediana", "diff_valor_mediana",
        "ratio_valor_desvio_padrao", "zscore_valor_aprox",
        "ratio_intervalo_vs_mediana", "diff_intervalo_vs_mediana",
        "zscore_intervalo_aprox",
        "ratio_latencia_cliente", "diff_latencia_cliente",
        "latencia_host_ratio",
        # Sequenciais
        "minutes_since_prev_tx", "tx_count_prev_30m", "burst_30m_flag",
        "receiver_tx_count_prev", "first_receiver_flag",
        "key_tx_count_prev", "first_key_flag",
        "distinct_receivers_so_far", "distinct_keys_so_far",
        # Temporais
        "hour", "day_of_week", "is_business_hours",
        # Device / mobile
        "app_version_minor",
        "topaz_risk_score", "topaz_score_filled",
        # Topaz rejeitada flag
        "topaz_rejeitada_flag",
        # Método de autenticação encoded
        "metodo_auth_encoded", "is_login_senha_flag", "is_login_biometria_flag",
        # Agendamento recorrente flag
        "is_agendamento_recorrente_flag",
        # Perfil do cliente
        "is_sexo_feminino_flag", "is_viuvo_flag", "is_segmento_premium_flag",
        # Primeiro envio e contagem recebedor
        "tp_primeiro_envio_recebedor_trimestre", "qt_envio_recebedor_trimestre",
        # Missing flags
        "device_missing_flag", "app_version_missing_flag",
        "auth_method_missing_flag", "topaz_missing_flag",
        "host_time_missing_flag", "latencia_missing_flag",
        "tempo_interacao_missing_flag",
        # Renda e perfil de vulnerabilidade
        "vl_renda_cliente", "qt_dependentes",
        "ratio_pix_renda", "pix_over_50pct_renda_flag", "pix_over_100pct_renda_flag",
        "renda_missing_flag", "perfil_vulneravel_se_flag",
        # Regras
        "rule_age_score", "rule_relationship_score",
        "rule_mule_account_score", "rule_random_key_score",
        "rule_velocity_score", "rule_topaz_score",
        "rule_score_raw", "rule_score_normalized",
        # === GRAPH FEATURES TEMPORAIS (13 novas — prefixo graph_) ===
        "graph_sender_out_degree",
        "graph_sender_tx_count",
        "graph_sender_avg_value",
        "graph_sender_value_zscore",
        "graph_receiver_in_degree",
        "graph_receiver_tx_count",
        "graph_receiver_avg_value",
        "graph_pair_tx_count",
        "graph_is_new_edge",
        "graph_sender_value_ratio",
        "graph_receiver_value_ratio",
        "graph_receiver_concentration_hhi",
        "graph_degree_ratio",
        # === FEATURES DE RELACIONAMENTO V3.1 (5 novas — Fase 2) ===
        "qtd_pix_mesmo_recebedor_7d",
        "valor_medio_para_recebedor_180d",
        "dias_desde_ultima_transacao_recebedor",
        "ratio_valor_pix_vs_max_recebedor_180d",
        "is_recebedor_recorrente_180d",
        # === FEATURES DE TRUST / REPUTACAO (R5B5) ===
        "payer_history_strength_score",
        "receiver_reputation_score",
        "relationship_strength_score",
        "receiver_novelty_risk_score",
        "transaction_normality_score",
        "payer_receiver_trust_score",
        "trust_bucket",
        "receiver_rep_bucket",
        "relationship_bucket",
        "novelty_bucket",
    ]

    for c in final_cols:
        df = ensure_column(df, c)
    return df[final_cols].copy()


# ═══════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════
def diagnose_features(
    df: pd.DataFrame, id_cols: list, label_col: str = "is_fraud",
) -> pd.DataFrame:
    """Gera diagnóstico de features: null%, nunique, stats por classe.

    Args:
        df: DataFrame com features + label.
        id_cols: Colunas de ID a ignorar.
        label_col: Nome da coluna de label.

    Returns:
        DataFrame de diagnóstico.
    """
    feature_cols = [c for c in df.columns if c not in id_cols]
    rows = []
    for c in feature_cols:
        null_pct = df[c].isna().mean()
        nunique = df[c].nunique(dropna=True)
        dtype = str(df[c].dtype)
        std_fraud = std_normal = mean_fraud = mean_normal = np.nan
        if dtype not in ("object", "category"):
            vals = pd.to_numeric(df[c], errors="coerce")
            if label_col in df.columns:
                fraud_mask = df[label_col] == 1
                normal_mask = df[label_col] == 0
                std_fraud = vals[fraud_mask].std(skipna=True)
                std_normal = vals[normal_mask].std(skipna=True)
                mean_fraud = vals[fraud_mask].mean(skipna=True)
                mean_normal = vals[normal_mask].mean(skipna=True)
        rows.append({
            "feature": c, "dtype": dtype,
            "null_pct": round(null_pct, 4), "nunique": nunique,
            "std_fraud": round(std_fraud, 4) if not pd.isna(std_fraud) else np.nan,
            "std_normal": round(std_normal, 4) if not pd.isna(std_normal) else np.nan,
            "mean_fraud": round(mean_fraud, 4) if not pd.isna(mean_fraud) else np.nan,
            "mean_normal": round(mean_normal, 4) if not pd.isna(mean_normal) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("null_pct", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
# PREPROCESSOR CLASS
# ═══════════════════════════════════════════════════════════════════════
class PixPreprocessor:
    """Preprocessador para transformar features em formato model-ready."""

    def __init__(self):
        self.numeric_imputer_: dict[str, float] = {}
        self.categorical_fill_value_: str = "__MISSING__"
        self.categorical_columns_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.model_columns_: list[str] = []
        self.categorical_levels_: dict[str, list[str]] = {}
        self.host_time_median_: float = 0.0
        self.dropped_high_null_: list[tuple[str, float]] = []
        self.dropped_explicit_: list[str] = []

        self.id_columns_ = [
            "transaction_id", "customer_id", "event_datetime",
            "source_dataset", "is_fraud",
        ]

        self.exclude_from_model_ = [
            "cd_cpf_cnpj_recebedor", "ds_chave_pix", "ds_tipo_chave",
            "device_name", "device_name_normalized", "app_version",
            "ip_address", "metodo_autenticacao", "session_id",
            "cd_retorno",
            "ds_sexo", "ds_estado_civil", "ds_segmento",
            "is_agendamento_recorrente",
        ]

    def fit(
        self,
        df: pd.DataFrame,
        null_threshold: float = 0.95,
        explicit_drop: list[str] | None = None,
    ) -> "PixPreprocessor":
        """Aprende parâmetros de transformação."""
        df = df.copy()
        if explicit_drop is None:
            explicit_drop = []

        cols_to_drop = [c for c in self.exclude_from_model_ if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors="ignore")

        self.dropped_explicit_ = [c for c in explicit_drop if c in df.columns]
        if self.dropped_explicit_:
            df = df.drop(columns=self.dropped_explicit_, errors="ignore")

        feature_cols = [c for c in df.columns if c not in self.id_columns_]

        self.dropped_high_null_ = []
        for c in feature_cols:
            null_pct = df[c].isna().mean()
            if null_pct > null_threshold:
                self.dropped_high_null_.append((c, round(null_pct, 4)))

        drop_null_names = [x[0] for x in self.dropped_high_null_]
        if drop_null_names:
            logger.info(
                "  ⚠ Removendo %d features com >%.0f%% null",
                len(drop_null_names), null_threshold * 100,
            )
            for name, pct in self.dropped_high_null_:
                logger.info("    - %s: %.1f%% null", name, pct * 100)
            df = df.drop(columns=drop_null_names, errors="ignore")
            feature_cols = [c for c in feature_cols if c not in drop_null_names]

        self.categorical_columns_ = [
            c for c in feature_cols
            if str(df[c].dtype) == "object" or str(df[c].dtype).startswith("category")
        ]
        self.numeric_columns_ = [
            c for c in feature_cols if c not in self.categorical_columns_
        ]

        self.numeric_imputer_ = {}
        for c in self.numeric_columns_:
            med = pd.to_numeric(df[c], errors="coerce").median(skipna=True)
            self.numeric_imputer_[c] = med if not pd.isna(med) else 0.0

        if "tempo_processamento_host_ms" in df.columns:
            v = pd.to_numeric(
                df["tempo_processamento_host_ms"], errors="coerce"
            ).median(skipna=True)
            self.host_time_median_ = v if not pd.isna(v) else 0.0

        self.categorical_levels_ = {}
        for c in self.categorical_columns_:
            vals = sorted(
                df[c].fillna(self.categorical_fill_value_).astype(str).unique().tolist()
            )
            self.categorical_levels_[c] = vals

        model_cols = list(self.numeric_columns_)
        for c in self.categorical_columns_:
            for level in self.categorical_levels_[c]:
                model_cols.append(f"{c}__{level}")
        self.model_columns_ = model_cols
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica transformação usando parâmetros aprendidos no fit."""
        df = df.copy()
        df = df.drop(
            columns=[c for c in self.exclude_from_model_ if c in df.columns],
            errors="ignore",
        )
        df = df.drop(
            columns=[c for c in self.dropped_explicit_ if c in df.columns],
            errors="ignore",
        )
        drop_null_names = [x[0] for x in self.dropped_high_null_]
        df = df.drop(
            columns=[c for c in drop_null_names if c in df.columns],
            errors="ignore",
        )

        for c in self.numeric_columns_:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(self.numeric_imputer_[c])

        for c in self.categorical_columns_:
            if c not in df.columns:
                df[c] = self.categorical_fill_value_
            df[c] = df[c].fillna(self.categorical_fill_value_).astype(str)

        parts = [df[self.numeric_columns_].copy()]
        for c in self.categorical_columns_:
            ohe_dict = {}
            for level in self.categorical_levels_[c]:
                ohe_dict[f"{c}__{level}"] = (df[c] == level).astype(int)
            parts.append(pd.DataFrame(ohe_dict, index=df.index))

        out = pd.concat(parts, axis=1)
        for c in self.model_columns_:
            if c not in out.columns:
                out[c] = 0
        return out[self.model_columns_].copy()

    def fit_transform(
        self,
        df: pd.DataFrame,
        null_threshold: float = 0.95,
        explicit_drop: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fit + transform em uma chamada."""
        self.fit(df, null_threshold=null_threshold, explicit_drop=explicit_drop)
        return self.transform(df)


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════
def main() -> None:
    """Pipeline principal — dados brutos → base de treino + artefato."""
    t_start = time.time()

    print("=" * 70)
    print("PREPROCESSING v4.1 — Pipeline com Graph Temporal Incremental")
    print("  Dados brutos → Feature Engineering → Leakage Fix")
    print("  → Graph Features TEMPORAIS → Base de Treino Final")
    print("=" * 70)

    # ── [1/8] LOAD & MERGE ──
    print("\n[1/8] Carregando e unificando dados brutos...")
    df = load_and_prepare_pix(PATH_PIX_NORMAL, PATH_PIX_FRAUD)

    n_fraud = int(df["is_fraud"].sum())
    n_normal = int((df["is_fraud"] == 0).sum())
    print(f"  Fraudes: {n_fraud:,} | Normais: {n_normal:,}")
    print(f"  Proporção fraude: {n_fraud / (n_fraud + n_normal) * 100:.3f}%")

    # ── [2/8] FEATURE ENGINEERING ──
    print("\n[2/8] Feature engineering completa...")
    df = create_all_features(df)

    # ── [3/8] DEDUPLICAÇÃO FINAL ──
    print("\n[3/8] Deduplicação final por transaction_id...")
    before = len(df)
    priority_cols = [
        "latencia_rede_ms_final", "tempo_processamento_host_ms",
        "device_name", "app_version", "ip_address", "metodo_autenticacao",
        "cd_retorno", "topaz_risk_score", "cd_cpf_cnpj_recebedor",
        "ds_chave_pix", "ds_tipo_chave", "session_id",
        "tempo_interacao_ms", "ds_sexo", "ds_estado_civil",
    ]
    df = deduplicate_by_key(df, "transaction_id", priority_cols)
    after = len(df)
    print(f"  {before:,} → {after:,} ({before - after:,} duplicatas removidas)")

    # ── [4/8] FIX LEAKAGE TEMPORAL ──
    print("\n[4/8] Corrigindo leakage temporal (rolling 90d por CPF)...")
    df = fix_leakage_temporal(df)
    n_first = int(df["is_first_tx_trimestre"].sum())
    print(f"  is_first_tx_trimestre: {n_first:,} transações sem histórico")

    # ── [5/8] GRAPH FEATURE ENGINEERING TEMPORAL ──
    print("\n[5/8] Graph Feature Engineering TEMPORAL (incremental)...")
    df_graph = compute_temporal_graph_features(df)

    # Merge graph features
    df = df.merge(df_graph, on="transaction_id", how="left")
    graph_cols = [c for c in df_graph.columns if c != "transaction_id"]
    df[graph_cols] = df[graph_cols].fillna(0)
    print(f"  {len(graph_cols)} graph features temporais adicionadas")

    # ── [6/8] SELEÇÃO DE COLUNAS ──
    print("\n[6/8] Seleção de colunas finais...")
    df_features = select_final_columns(df)
    print(f"  {len(df_features.columns)} colunas selecionadas")

    # ── [7/8] DIAGNÓSTICO + PREPROCESSOR ──
    print("\n[7/8] Diagnóstico e fit/transform do PixPreprocessor...")
    id_cols = [
        "transaction_id", "customer_id", "event_datetime",
        "source_dataset", "is_fraud",
    ]

    diag = diagnose_features(df_features, id_cols)
    diag.to_csv(OUTPUT_DIAGNOSTICO, index=False)
    logger.info("  Diagnóstico salvo: %s", OUTPUT_DIAGNOSTICO)

    preprocessor = PixPreprocessor()
    feature_only = df_features.drop(columns=preprocessor.id_columns_, errors="ignore")
    X_ready = preprocessor.fit_transform(
        feature_only,
        null_threshold=NULL_THRESHOLD,
        explicit_drop=FEATURES_TO_DROP_FROM_MODEL,
    )

    if "tempo_processamento_host_ms" in feature_only.columns:
        v = pd.to_numeric(
            feature_only["tempo_processamento_host_ms"], errors="coerce"
        ).median(skipna=True)
        preprocessor.host_time_median_ = v if not pd.isna(v) else 0.0

    model_ready = pd.concat(
        [
            df_features[preprocessor.id_columns_].reset_index(drop=True),
            X_ready.reset_index(drop=True),
        ],
        axis=1,
    )

    # ── [8/8] SALVAR ──
    print(f"\n[8/8] Salvando artefatos...")

    print(f"  CSV: {OUTPUT_MODEL_READY}")
    model_ready.to_csv(OUTPUT_MODEL_READY, index=False)

    print(f"  Joblib: {OUTPUT_PREPROCESSOR}")
    joblib.dump(preprocessor, OUTPUT_PREPROCESSOR)

    # ── RESUMO FINAL ──
    elapsed = time.time() - t_start

    print("\n" + "=" * 70)
    print("✅ PREPROCESSING v4.1 CONCLUÍDO")
    print("=" * 70)
    print(f"\n  ⏱️  Tempo total: {elapsed:.1f}s")
    print(f"\n  📊 Dataset final:")
    print(f"     Shape:              {model_ready.shape}")
    print(f"     Fraudes:            {n_fraud:,}")
    print(f"     Normais:            {n_normal:,}")
    print(f"     Proporção fraude:   {n_fraud / (n_fraud + n_normal) * 100:.3f}%")
    print(f"\n  🔧 Preprocessor:")
    print(f"     Colunas do modelo:  {len(preprocessor.model_columns_)}")
    print(f"     Numéricas:          {len(preprocessor.numeric_columns_)}")
    print(f"     Categóricas:        {len(preprocessor.categorical_columns_)}")
    print(f"     Excluídas (texto):  {len(preprocessor.exclude_from_model_)}")
    print(f"     Excluídas (drop):   {len(preprocessor.dropped_explicit_)}")
    print(f"     Excluídas (null):   {len(preprocessor.dropped_high_null_)}")
    print(f"\n  📊 Graph features temporais: {len(graph_cols)}")

    # Effect size das graph features
    if n_fraud > 0:
        fraud_mask = df_features["is_fraud"] == 1
        normal_mask = df_features["is_fraud"] == 0
        graph_in_final = [c for c in graph_cols if c in df_features.columns]
        if graph_in_final:
            print(f"\n  🔥 Graph Features (effect size fraude vs normal):")
            effects = []
            for col in graph_in_final:
                vals = pd.to_numeric(df_features[col], errors="coerce")
                mf = vals[fraud_mask].mean()
                mn = vals[normal_mask].mean()
                std = vals.std()
                es = abs(mf - mn) / std if std > 0 else 0.0
                effects.append((col, mn, mf, es))
            effects.sort(key=lambda x: x[3], reverse=True)
            for col, mn, mf, es in effects:
                emoji = "🔥" if es > 0.5 else "✅" if es > 0.2 else "  "
                print(
                    f"     {emoji} {col:<40} "
                    f"normal={mn:.4f}  fraude={mf:.4f}  ES={es:.4f}"
                )

    print(f"\n  📁 Artefatos salvos:")
    print(f"     {OUTPUT_MODEL_READY}")
    print(f"     {OUTPUT_PREPROCESSOR}")
    print(f"     {OUTPUT_DIAGNOSTICO}")
    print("=" * 70)


if __name__ == "__main__":
    main()
