"""
preprocessing.py — Script unificado de preprocessing para MVP de detecção de fraude PIX.

Lê:
  - /dados/dados_pix_normais.csv
  - /dados/dados_fraudes_pix.csv

Faz:
  - padronização de colunas e tipos
  - limpeza de texto, sentinelas, zeros
  - deduplicação robusta por cd_pix
  - consolidação normais + fraudes
  - balanceamento 50/50 (100% fraudes + amostra de normais)
  - criação de todas as features derivadas do MVP
  - flags de missing, chave PIX, device, regras
  - features sequenciais por cliente
  - fit/transform do PixPreprocessor (imputação + one-hot)
  - geração de artefato joblib para inferência

Salva:
  - /dados/base_mvp_model_ready.csv    (pronta para treino, após imputação + encoding)
  - /backend/artefatos/preprocessing.joblib
"""

import os
import re
import joblib
import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DADOS_DIR = os.path.join(PROJECT_ROOT, "dados")
ARTEFACT_DIR = os.path.join(PROJECT_ROOT, "backend", "artefatos")

PATH_PIX_NORMAL = os.path.join(DADOS_DIR, "dados_pix_normais.csv")
PATH_PIX_FRAUD = os.path.join(DADOS_DIR, "dados_pix_fraudes.csv")

OUTPUT_MODEL_READY = os.path.join(DADOS_DIR, "base_mvp_model_ready.csv")
OUTPUT_PREPROCESSOR = os.path.join(ARTEFACT_DIR, "preprocessing.joblib")

os.makedirs(DADOS_DIR, exist_ok=True)
os.makedirs(ARTEFACT_DIR, exist_ok=True)

RANDOM_STATE = 42


# =========================================================
# HELPERS
# =========================================================
NULL_STRINGS = {"", "null", "none", "nan", "nat", "missing", "informação ausente"}


def normalize_colname(col: str) -> str:
    col = str(col).strip()
    if "." in col:
        col = col.split(".")[-1]
    return col.strip().lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]
    return df


def ensure_column(df: pd.DataFrame, col: str, default=np.nan) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = default
    return df


def normalize_text_value(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x.lower() in NULL_STRINGS:
        return np.nan
    return x


def clean_text_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(normalize_text_value)
    return df


def safe_to_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def safe_to_datetime(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def replace_sentinels_with_nan(df: pd.DataFrame, cols: list, sentinels=None) -> pd.DataFrame:
    if sentinels is None:
        sentinels = [-1, -1.0]
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(sentinels, np.nan)
    return df


def replace_zero_with_nan(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)
    return df


def robust_divide(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.where((pd.isna(a)) | (pd.isna(b)) | (b == 0), np.nan, a / b)


def normalize_device_name(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    if x in NULL_STRINGS:
        return np.nan
    return re.sub(r"\s+", "_", x)


def extract_app_version_major(version):
    if pd.isna(version):
        return np.nan
    m = re.match(r"^(\d+)", str(version).strip())
    return float(m.group(1)) if m else np.nan


def extract_app_version_minor(version):
    if pd.isna(version):
        return np.nan
    parts = str(version).strip().split(".")
    if len(parts) >= 2 and parts[1].isdigit():
        return float(parts[1])
    return np.nan


def period_of_day(hour):
    if pd.isna(hour):
        return np.nan
    hour = int(hour)
    if 0 <= hour < 6:
        return "madrugada"
    elif 6 <= hour < 12:
        return "manha"
    elif 12 <= hour < 18:
        return "tarde"
    return "noite"


def map_topaz_rule(score):
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
    s = series.astype(str).str.strip()
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    s = s.str.replace(r"\s+", "", regex=True)
    return s


def count_non_null_priority(df: pd.DataFrame, cols: list) -> pd.Series:
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return pd.Series(0, index=df.index)
    return df[valid_cols].notna().sum(axis=1)


def deduplicate_by_key(
    df: pd.DataFrame,
    key_col: str,
    priority_cols: list,
    extra_priority_cols: list = None,
) -> pd.DataFrame:
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
    s = ds_tipo_chave_series.astype(str).str.upper()
    out = pd.DataFrame(index=ds_tipo_chave_series.index)
    out["pix_key_random_flag"] = (s == "CHAVE ALEATORIA").astype(int)
    out["pix_key_email_flag"] = (s == "EMAIL").astype(int)
    out["pix_key_document_flag"] = (s == "DOCUMENTO/TELEFONE").astype(int)
    out["pix_key_other_flag"] = (s == "OUTROS").astype(int)
    out["pix_key_missing_flag_derived"] = s.isin(["NAN", "INFORMAÇÃO AUSENTE"]).astype(int)
    return out


def cumulative_distinct_count(series: pd.Series) -> pd.Series:
    seen = set()
    counts = []
    for val in series:
        if pd.notna(val):
            seen.add(val)
        counts.append(len(seen))
    return pd.Series(counts, index=series.index)


# =========================================================
# PREPROCESSOR CLASS
# =========================================================
class PixPreprocessor:
    """
    Artefato reutilizável para inferência.

    - fit():  aprende medianas de imputação e níveis categóricos
    - transform():  aplica imputação + one-hot encoding consistente
    - Salvo via joblib para uso em produção
    """

    def __init__(self):
        self.numeric_imputer_ = {}
        self.categorical_fill_value_ = "__MISSING__"
        self.categorical_columns_ = []
        self.numeric_columns_ = []
        self.model_columns_ = []
        self.categorical_levels_ = {}
        self.host_time_median_ = 0.0

        self.id_columns_ = [
            "transaction_id",
            "customer_id",
            "event_datetime",
            "source_dataset",
            "is_fraud",
        ]

        # Colunas de texto bruto que NÃO devem entrar no modelo.
        # Elas já foram usadas para derivar flags e features numéricas.
        self.exclude_from_model_ = [
            "cd_cpf_cnpj_recebedor",
            "ds_chave_pix",
            "ds_tipo_chave",
            "device_name",
            "device_name_normalized",
            "app_version",
            "ip_address",
            "metodo_autenticacao",
            "session_id",
            "cd_retorno",
            "is_agendamento_recorrente",
            "topaz_sync_id",
        ]

    def fit(self, df: pd.DataFrame):
        df = df.copy()

        # Remover colunas brutas de texto que não devem virar features
        cols_to_drop = [c for c in self.exclude_from_model_ if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors="ignore")

        feature_cols = [c for c in df.columns if c not in self.id_columns_]

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
            v = pd.to_numeric(df["tempo_processamento_host_ms"], errors="coerce").median(skipna=True)
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
        df = df.copy()

        # Remover colunas brutas de texto
        df = df.drop(
            columns=[c for c in self.exclude_from_model_ if c in df.columns],
            errors="ignore",
        )

        # Garantir e imputar numéricas
        for c in self.numeric_columns_:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(self.numeric_imputer_[c])

        # Garantir e imputar categóricas
        for c in self.categorical_columns_:
            if c not in df.columns:
                df[c] = self.categorical_fill_value_
            df[c] = df[c].fillna(self.categorical_fill_value_).astype(str)

        # Construir DataFrame de saída via pd.concat (sem fragmentação)
        parts = [df[self.numeric_columns_].copy()]

        for c in self.categorical_columns_:
            ohe_dict = {}
            for level in self.categorical_levels_[c]:
                ohe_dict[f"{c}__{level}"] = (df[c] == level).astype(int)
            parts.append(pd.DataFrame(ohe_dict, index=df.index))

        out = pd.concat(parts, axis=1)

        # Garantir todas as colunas na ordem correta
        for c in self.model_columns_:
            if c not in out.columns:
                out[c] = 0

        return out[self.model_columns_].copy()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)


# =========================================================
# PIPELINE FUNCTIONS
# =========================================================

def load_and_prepare_pix(path_normal: str, path_fraud: str) -> pd.DataFrame:
    """Carrega, padroniza e concatena bases PIX normais + fraudes."""
    print("  Carregando PIX normais...")
    pix_normal = standardize_columns(pd.read_csv(path_normal, low_memory=False))
    print(f"    → {len(pix_normal)} linhas")

    print("  Carregando PIX fraudes...")
    pix_fraud = standardize_columns(pd.read_csv(path_fraud, low_memory=False))
    print(f"    → {len(pix_fraud)} linhas")

    # labels
    pix_normal["is_fraud"] = 0
    pix_normal["source_dataset"] = "normal"

    if "tp_fraude" in pix_fraud.columns:
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
        "topaz_transacao_habilitada", "is_agendamento_recorrente", "topaz_sync_id",
        "qt_aparelhos_distintos_trimestre", "nr_idade", "qt_tempo_relacionamento_mes",
        "is_fraud", "source_dataset", "dt_carga",
    ]

    for c in required_cols:
        pix_normal = ensure_column(pix_normal, c)
        pix_fraud = ensure_column(pix_fraud, c)

    pix_all = pd.concat(
        [pix_normal[required_cols], pix_fraud[required_cols]],
        ignore_index=True, sort=False,
    )

    # limpeza texto
    text_cols = [
        "cd_pix", "cd_cpf_pagador", "cd_cpf_cnpj_recebedor",
        "ds_chave_pix", "ds_tipo_chave", "device_name", "app_version",
        "ip_address", "metodo_autenticacao", "session_id", "cd_retorno",
        "is_agendamento_recorrente", "topaz_sync_id", "source_dataset",
    ]
    pix_all = clean_text_columns(pix_all, text_cols)

    # tipos numéricos
    num_cols = [
        "vl_pix", "qt_total_pix_trimestre", "vl_mediana_pix_trimestre",
        "vl_desvio_padrao_pix_trimestre", "qt_intervalo_transacao_minuto",
        "qt_intervalo_mediana_trimestre", "qt_intervalo_desvio_padrao_trimestre",
        "qt_pix_dia_maximo_trimestre", "latencia_rede_ms",
        "vl_latencia_rede_media_trimestre", "tempo_interacao_ms",
        "vl_tempo_interacao_medio_trimestre", "tempo_processamento_host_ms",
        "topaz_risk_score", "topaz_transacao_rejeitada",
        "topaz_transacao_habilitada", "qt_aparelhos_distintos_trimestre",
        "nr_idade", "qt_tempo_relacionamento_mes", "is_fraud",
    ]
    pix_all = safe_to_numeric(pix_all, num_cols)
    pix_all = safe_to_datetime(pix_all, ["dt_pix", "dt_carga"])

    # chave normalizada
    pix_all["cd_pix"] = normalize_transaction_key(pix_all["cd_pix"])
    pix_all["cd_cpf_pagador"] = pix_all["cd_cpf_pagador"].astype("object")
    pix_all["cd_cpf_cnpj_recebedor"] = pix_all["cd_cpf_cnpj_recebedor"].astype("object")

    # sentinelas
    sentinel_cols = [
        "latencia_rede_ms", "vl_latencia_rede_media_trimestre",
        "tempo_interacao_ms", "vl_tempo_interacao_medio_trimestre",
        "tempo_processamento_host_ms", "topaz_risk_score",
        "topaz_transacao_rejeitada", "topaz_transacao_habilitada",
    ]
    pix_all = replace_sentinels_with_nan(pix_all, sentinel_cols)
    pix_all = replace_zero_with_nan(pix_all, ["vl_tempo_interacao_medio_trimestre"])

    # dedup
    print("  Deduplicando PIX por cd_pix...")
    pix_priority = [
        "cd_cpf_cnpj_recebedor", "ds_chave_pix", "ds_tipo_chave",
        "device_name", "app_version", "ip_address", "latencia_rede_ms",
        "tempo_interacao_ms", "tempo_processamento_host_ms",
        "metodo_autenticacao", "session_id", "cd_retorno",
        "topaz_risk_score", "topaz_transacao_rejeitada",
        "topaz_transacao_habilitada", "is_agendamento_recorrente", "topaz_sync_id",
    ]
    pix_all = deduplicate_by_key(pix_all, "cd_pix", pix_priority)
    print(f"    → {len(pix_all)} transações únicas")

    return pix_all


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Balanceia 50/50: 100% fraudes + amostra aleatória de normais."""
    df_fraud = df[df["is_fraud"] == 1].copy()
    df_normal = df[df["is_fraud"] == 0].copy()

    n_fraud = len(df_fraud)
    n_normal = len(df_normal)

    print(f"  Fraudes: {n_fraud} | Normais: {n_normal}")

    if n_fraud == 0:
        print("  ⚠ Nenhuma fraude encontrada — retornando base completa sem balanceamento.")
        return df

    if n_normal <= n_fraud:
        print(f"  Normais ({n_normal}) <= Fraudes ({n_fraud}) — usando todas as normais.")
        df_normal_sample = df_normal
    else:
        print(f"  Amostrando {n_fraud} normais para balancear com {n_fraud} fraudes.")
        df_normal_sample = df_normal.sample(n=n_fraud, random_state=RANDOM_STATE)

    df_balanced = pd.concat([df_fraud, df_normal_sample], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"  Base balanceada: {len(df_balanced)} linhas "
          f"({len(df_fraud)} fraudes + {len(df_normal_sample)} normais)")
    return df_balanced


def create_all_features(df: pd.DataFrame, host_time_median: float = None) -> pd.DataFrame:
    """Cria todas as features derivadas do MVP."""
    df = df.copy()

    # --- IDs ---
    df["transaction_id"] = df["cd_pix"]
    df["customer_id"] = df["cd_cpf_pagador"]
    df["event_datetime"] = df["dt_pix"]

    df = df[df["transaction_id"].notna()].copy()

    # latência e interação já vêm direto das bases unificadas
    df["latencia_rede_ms_final"] = df["latencia_rede_ms"]
    df["tempo_interacao_ms_final"] = df["tempo_interacao_ms"]

    # --- MISSING FLAGS ---
    df["device_missing_flag"] = df["device_name"].isna().astype(int)
    df["ip_missing_flag"] = df["ip_address"].isna().astype(int)
    df["app_version_missing_flag"] = df["app_version"].isna().astype(int)
    df["auth_method_missing_flag"] = df["metodo_autenticacao"].isna().astype(int)
    df["topaz_missing_flag"] = df["topaz_risk_score"].isna().astype(int)
    df["host_time_missing_flag"] = df["tempo_processamento_host_ms"].isna().astype(int)
    df["latencia_missing_flag"] = df["latencia_rede_ms_final"].isna().astype(int)
    df["tempo_interacao_missing_flag"] = df["tempo_interacao_ms_final"].isna().astype(int)
    df["receiver_missing_flag"] = df["cd_cpf_cnpj_recebedor"].isna().astype(int)
    df["pix_key_missing_flag"] = df["ds_chave_pix"].isna().astype(int)
    df["pix_key_type_missing_flag"] = df["ds_tipo_chave"].isna().astype(int)
    df["session_missing_flag"] = df["session_id"].isna().astype(int)
    df["topaz_sync_missing_flag"] = df["topaz_sync_id"].isna().astype(int)

    # --- DEVICE / APP ---
    df["device_name_normalized"] = df["device_name"].apply(normalize_device_name)
    df["app_version_major"] = df["app_version"].apply(extract_app_version_major)
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
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df["is_business_hours"] = df["hour"].between(8, 18, inclusive="both").fillna(False).astype(int)
    df["period_of_day"] = df["hour"].apply(period_of_day)

    # --- CORE DERIVED ---
    df["log_vl_pix"] = np.log1p(df["vl_pix"].clip(lower=0))
    df["ratio_valor_mediana"] = robust_divide(df["vl_pix"], df["vl_mediana_pix_trimestre"])
    df["diff_valor_mediana"] = df["vl_pix"] - df["vl_mediana_pix_trimestre"]
    df["ratio_valor_desvio_padrao"] = robust_divide(df["vl_pix"], df["vl_desvio_padrao_pix_trimestre"])
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
    df["diff_latencia_cliente"] = df["latencia_rede_ms_final"] - df["vl_latencia_rede_media_trimestre"]

    df["ratio_tempo_interacao_cliente"] = robust_divide(
        df["tempo_interacao_ms_final"], df["vl_tempo_interacao_medio_trimestre"]
    )
    df["diff_tempo_interacao_cliente"] = (
        df["tempo_interacao_ms_final"] - df["vl_tempo_interacao_medio_trimestre"]
    )

    df["tempo_interacao_baixo_flag"] = (
        df["ratio_tempo_interacao_cliente"] < 0.5
    ).fillna(False).astype(int)
    df["tempo_interacao_alto_flag"] = (
        df["ratio_tempo_interacao_cliente"] > 2.0
    ).fillna(False).astype(int)

    df["latencia_host_ratio"] = robust_divide(
        df["latencia_rede_ms_final"], df["tempo_processamento_host_ms"]
    )

    if host_time_median is None:
        host_time_median = pd.to_numeric(
            df["tempo_processamento_host_ms"], errors="coerce"
        ).median(skipna=True)
        if pd.isna(host_time_median):
            host_time_median = 0.0

    df["processamento_host_alto_flag"] = (
        df["tempo_processamento_host_ms"] > host_time_median
    ).fillna(False).astype(int)

    df["topaz_score_filled"] = df["topaz_risk_score"].fillna(-1)

    # --- SEQUENCIAIS ---
    print("  Criando features sequenciais por cliente...")
    df = df.sort_values(["customer_id", "event_datetime", "transaction_id"]).reset_index(drop=True)

    df["prev_event_datetime"] = df.groupby("customer_id")["event_datetime"].shift(1)
    df["minutes_since_prev_tx"] = (
        (df["event_datetime"] - df["prev_event_datetime"]).dt.total_seconds() / 60.0
    )

    # tx_count_prev_30m
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
    df["pix_freq_high_flag"] = (
        (df["ratio_intervalo_vs_mediana"] < 0.5) | (df["tx_count_prev_30m"] >= 2)
    ).fillna(False).astype(int)

    # receiver / key sequenciais
    df["receiver_tx_count_prev"] = df.groupby(["customer_id", "cd_cpf_cnpj_recebedor"]).cumcount()
    df["first_receiver_flag"] = (df["receiver_tx_count_prev"] == 0).astype(int)

    df["key_tx_count_prev"] = df.groupby(["customer_id", "ds_chave_pix"]).cumcount()
    df["first_key_flag"] = (df["key_tx_count_prev"] == 0).astype(int)

    df["distinct_receivers_so_far"] = (
        df.groupby("customer_id", sort=False)["cd_cpf_cnpj_recebedor"]
        .transform(cumulative_distinct_count)
    )
    df["distinct_keys_so_far"] = (
        df.groupby("customer_id", sort=False)["ds_chave_pix"]
        .transform(cumulative_distinct_count)
    )

    # --- REGRAS ---
    df["rule_pix_30m_score"] = np.select(
        [df["tx_count_prev_30m"] == 0, df["tx_count_prev_30m"] == 1, df["tx_count_prev_30m"] >= 2],
        [0, 1, 2], default=0,
    )
    df["rule_ratio_pix_limite_score"] = np.nan

    df["rule_age_score"] = np.select(
        [df["nr_idade"].between(60, 65), df["nr_idade"].between(66, 75), df["nr_idade"] >= 76],
        [1, 2, 3], default=0,
    )
    df["is_elderly_flag"] = (df["nr_idade"] >= 60).fillna(False).astype(int)

    df["rule_relationship_score"] = np.select(
        [
            df["qt_tempo_relacionamento_mes"].between(61, 90),
            df["qt_tempo_relacionamento_mes"].between(31, 60),
            df["qt_tempo_relacionamento_mes"].between(0, 30),
        ],
        [1, 2, 3], default=0,
    )
    df["is_new_customer_flag"] = (
        df["qt_tempo_relacionamento_mes"].between(0, 30)
    ).fillna(False).astype(int)

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

    df["rule_night_score"] = np.where(df["is_night"] == 1, 3, 0)
    df["rule_velocity_score"] = np.select(
        [
            df["tx_count_prev_30m"] == 0, df["tx_count_prev_30m"] == 1,
            df["tx_count_prev_30m"] == 2, df["tx_count_prev_30m"] >= 3,
        ],
        [0, 2, 3, 4], default=0,
    )
    df["rule_topaz_score"] = df["topaz_risk_score"].apply(map_topaz_rule)

    df["autorizacao_previa_flag"] = 0
    df["rule_pre_authorization_discount"] = 0.0

    rule_components = [
        "rule_pix_30m_score", "rule_age_score", "rule_relationship_score",
        "rule_mule_account_score", "rule_random_key_score",
        "rule_night_score", "rule_velocity_score", "rule_topaz_score",
    ]
    df["rule_score_raw"] = df[rule_components].fillna(0).sum(axis=1)
    max_rs = df["rule_score_raw"].max(skipna=True)
    df["rule_score_normalized"] = (
        df["rule_score_raw"] / max_rs if (not pd.isna(max_rs) and max_rs > 0) else 0.0
    )

    # limpar temporários
    df = df.drop(columns=["prev_event_datetime"], errors="ignore")

    return df


def deduplicate_final(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicação final por transaction_id."""
    df = df.copy()
    priority_cols = [
        "latencia_rede_ms_final", "tempo_interacao_ms_final",
        "tempo_processamento_host_ms", "device_name", "app_version",
        "ip_address", "metodo_autenticacao", "cd_retorno", "topaz_risk_score",
        "topaz_transacao_rejeitada", "topaz_transacao_habilitada",
        "is_agendamento_recorrente", "cd_cpf_cnpj_recebedor",
        "ds_chave_pix", "ds_tipo_chave", "session_id", "topaz_sync_id",
    ]
    df = deduplicate_by_key(df, "transaction_id", priority_cols)
    return df


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona e ordena as colunas finais para o modelo."""
    final_cols = [
        # IDs + label
        "transaction_id", "customer_id", "event_datetime",
        "source_dataset", "is_fraud",
        # recebedor / chave (texto bruto — será excluído no fit/transform)
        "cd_cpf_cnpj_recebedor", "ds_chave_pix", "ds_tipo_chave",
        # flags derivadas de recebedor / chave
        "receiver_document_same_as_customer_flag",
        "pix_key_random_flag", "pix_key_email_flag",
        "pix_key_document_flag", "pix_key_other_flag", "pix_key_missing_flag_derived",
        # valor / histórico
        "vl_pix", "log_vl_pix", "qt_total_pix_trimestre",
        "vl_mediana_pix_trimestre", "vl_desvio_padrao_pix_trimestre",
        "qt_intervalo_transacao_minuto", "qt_intervalo_mediana_trimestre",
        "qt_intervalo_desvio_padrao_trimestre", "qt_pix_dia_maximo_trimestre",
        "qt_aparelhos_distintos_trimestre", "nr_idade", "qt_tempo_relacionamento_mes",
        # latência / interação
        "latencia_rede_ms_final", "vl_latencia_rede_media_trimestre",
        "tempo_interacao_ms_final", "vl_tempo_interacao_medio_trimestre",
        "tempo_processamento_host_ms",
        # ratios / desvios
        "ratio_valor_mediana", "diff_valor_mediana",
        "ratio_valor_desvio_padrao", "zscore_valor_aprox",
        "ratio_intervalo_vs_mediana", "diff_intervalo_vs_mediana",
        "zscore_intervalo_aprox",
        "ratio_latencia_cliente", "diff_latencia_cliente",
        "ratio_tempo_interacao_cliente", "diff_tempo_interacao_cliente",
        "tempo_interacao_baixo_flag", "tempo_interacao_alto_flag",
        "latencia_host_ratio", "processamento_host_alto_flag",
        # sequenciais
        "minutes_since_prev_tx", "tx_count_prev_30m",
        "burst_30m_flag", "pix_freq_high_flag",
        "receiver_tx_count_prev", "first_receiver_flag",
        "key_tx_count_prev", "first_key_flag",
        "distinct_receivers_so_far", "distinct_keys_so_far",
        # temporais
        "hour", "day_of_week", "is_weekend", "is_night",
        "is_business_hours", "period_of_day",
        # device / mobile (texto bruto — será excluído no fit/transform)
        "device_name", "device_name_normalized",
        "app_version", "app_version_major", "app_version_minor",
        "ip_address", "metodo_autenticacao", "session_id",
        "cd_retorno", "topaz_risk_score", "topaz_score_filled",
        "topaz_transacao_rejeitada", "topaz_transacao_habilitada",
        "is_agendamento_recorrente", "topaz_sync_id",
        # missing flags
        "device_missing_flag", "ip_missing_flag",
        "app_version_missing_flag", "auth_method_missing_flag",
        "topaz_missing_flag", "host_time_missing_flag",
        "latencia_missing_flag", "tempo_interacao_missing_flag",
        "receiver_missing_flag", "pix_key_missing_flag",
        "pix_key_type_missing_flag", "session_missing_flag",
        "topaz_sync_missing_flag",
        # regras
        "rule_pix_30m_score", "rule_ratio_pix_limite_score",
        "rule_age_score", "is_elderly_flag",
        "rule_relationship_score", "is_new_customer_flag",
        "rule_mule_account_score", "rule_random_key_score",
        "rule_night_score", "rule_velocity_score", "rule_topaz_score",
        "autorizacao_previa_flag", "rule_pre_authorization_discount",
        "rule_score_raw", "rule_score_normalized",
    ]
    for c in final_cols:
        df = ensure_column(df, c)
    return df[final_cols].copy()


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 70)
    print("PREPROCESSING UNIFICADO — MVP Fraude PIX")
    print("=" * 70)

    # --- 1. CARGA ---
    print("\n[1/6] Carregando dados...")
    pix_all = load_and_prepare_pix(PATH_PIX_NORMAL, PATH_PIX_FRAUD)

    # --- 2. BALANCEAMENTO ---
    print("\n[2/6] Balanceamento 50/50...")
    df = balance_dataset(pix_all)

    # --- 3. FEATURES ---
    print("\n[3/6] Feature engineering...")
    df = create_all_features(df)

    # --- 4. DEDUP FINAL ---
    print("\n[4/6] Deduplicação final...")
    before = len(df)
    df = deduplicate_final(df)
    after = len(df)
    print(f"  {before} → {after} ({before - after} duplicatas removidas)")

    # --- 5. SELEÇÃO + FIT/TRANSFORM ---
    print("\n[5/6] Seleção de colunas + fit/transform do preprocessor...")
    df_features = select_final_columns(df)

    # diagnósticos
    n_fraud = int(df_features["is_fraud"].sum())
    n_normal = int((df_features["is_fraud"] == 0).sum())
    n_unique = df_features["transaction_id"].nunique()
    n_dups = df_features["transaction_id"].duplicated().sum()

    print(f"\n  Diagnósticos:")
    print(f"    Fraudes:       {n_fraud}")
    print(f"    Normais:       {n_normal}")
    print(f"    Proporção:     {n_fraud / (n_fraud + n_normal) * 100:.1f}% fraude")
    print(f"    Tx únicas:     {n_unique}")
    print(f"    Duplicatas:    {n_dups}")

    preprocessor = PixPreprocessor()

    feature_only = df_features.drop(columns=preprocessor.id_columns_, errors="ignore")
    X_ready = preprocessor.fit_transform(feature_only)

    # salvar mediana do host time no artefato
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

    print(f"\n  Salvando model-ready em: {OUTPUT_MODEL_READY}")
    model_ready.to_csv(OUTPUT_MODEL_READY, index=False)

    # --- 6. ARTEFATO ---
    print(f"\n[6/6] Salvando artefato em: {OUTPUT_PREPROCESSOR}")
    joblib.dump(preprocessor, OUTPUT_PREPROCESSOR)

    print("\n" + "=" * 70)
    print("CONCLUÍDO")
    print(f"  base_mvp_model_ready:  {model_ready.shape}")
    print(f"  Colunas do modelo:     {len(preprocessor.model_columns_)}")
    print(f"  Numéricas:             {len(preprocessor.numeric_columns_)}")
    print(f"  Categóricas:           {len(preprocessor.categorical_columns_)}")
    print(f"  Excluídas (texto):     {len(preprocessor.exclude_from_model_)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
