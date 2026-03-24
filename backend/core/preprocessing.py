"""
preprocessing.py v3.1 — Preprocessing otimizado para MVP de detecção de fraude PIX.

Mudanças v3.1 (sobre v3):
  - RESGATADOS: tempo_interacao_ms, vl_tempo_interacao_medio_trimestre,
    metodo_autenticacao, is_agendamento_recorrente, topaz_transacao_rejeitada
    (antes descartados — agora geram features para LGBM + Behavioral + SE)
  - ADICIONADOS novos campos da ingestão v2.1:
    ds_sexo, ds_estado_civil, ds_segmento,
    tp_primeiro_envio_recebedor_trimestre, qt_envio_recebedor_trimestre
  - ADICIONADOS campos v2.1b:
    vl_renda_cliente, qt_dependentes (da aoxb17 — pessoa física)
  - NOVAS FEATURES derivadas:
    metodo_auth_encoded, is_login_senha_flag, is_login_biometria_flag,
    is_agendamento_recorrente_flag, topaz_rejeitada_flag,
    is_sexo_feminino_flag, is_viuvo_flag, is_segmento_premium_flag,
    ratio_tempo_interacao_cliente, diff_tempo_interacao_cliente,
    tempo_interacao_missing_flag,
    ratio_pix_renda, pix_over_50pct_renda_flag, pix_over_100pct_renda_flag,
    renda_missing_flag, perfil_vulneravel_se_flag
  - REMOVIDOS do drop list: topaz_transacao_rejeitada, is_agendamento_recorrente,
    tempo_interacao_ms_final, vl_tempo_interacao_medio_trimestre,
    ratio_tempo_interacao_cliente, diff_tempo_interacao_cliente
  - REMOVIDOS da ingestão: topaz_sync_id, topaz_transacao_habilitada
  - ATUALIZADO exclude_from_model_ para não excluir metodo_autenticacao
    e is_agendamento_recorrente (agora geram flags numéricas)

Correções v3.1 consolidado:
  - FIX #1: Removida duplicação em required_cols
  - FIX #2: Indentação do bloco de renda corrigida (4 espaços)
  - FIX #3: perfil_vulneravel_se_flag movido para depois de is_viuvo_flag
  - FIX #4: Comentários de seção adicionados em select_final_columns
"""

import os
import re
import joblib
import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DADOS_DIR = os.path.join(PROJECT_ROOT, "dados")
ARTEFACT_DIR = os.path.join(PROJECT_ROOT, "backend", "artefatos")

PATH_PIX_NORMAL = os.path.join(DADOS_DIR, "dados_pix_normais.csv")
PATH_PIX_FRAUD = os.path.join(DADOS_DIR, "dados_pix_fraudes.csv")

OUTPUT_MODEL_READY = os.path.join(DADOS_DIR, "base_mvp_model_ready.csv")
OUTPUT_PREPROCESSOR = os.path.join(ARTEFACT_DIR, "preprocessing.joblib")
OUTPUT_DIAGNOSTICO = os.path.join(ARTEFACT_DIR, "diagnostico_features.csv")

os.makedirs(DADOS_DIR, exist_ok=True)
os.makedirs(ARTEFACT_DIR, exist_ok=True)

RANDOM_STATE = 42
NULL_THRESHOLD = 0.95

# Features a serem EXPLICITAMENTE removidas do modelo
FEATURES_TO_DROP_FROM_MODEL = [
    # Placeholders sem dado real
    "rule_ratio_pix_limite_score",
    "autorizacao_previa_flag",
    "rule_pre_authorization_discount",
    # Redundantes — informação já capturada por features contínuas
    "is_elderly_flag",           # nr_idade já cobre
    "is_new_customer_flag",      # qt_tempo_relacionamento_mes já cobre
    "rule_pix_30m_score",        # tx_count_prev_30m + rule_velocity_score já cobrem
    "rule_night_score",          # hour captura
    # Missing flags sem variância ou contribuição
    "receiver_missing_flag",         # 100% = 0 (constante)
    "pix_key_missing_flag",          # duplica pix_key_missing_flag_derived
    "pix_key_type_missing_flag",     # duplica pix_key_missing_flag_derived
    "session_missing_flag",          # correlaciona com device_missing_flag
    "ip_missing_flag",               # correlaciona com device_missing_flag
    # Gain=0 e SHAP=0 consistentemente
    "app_version_major",        # quase sem variância (99.99% = 7)
    "is_weekend",               # gain=0, shap=0
    "is_night",                 # gain=0 no v2 (hour captura)
    "processamento_host_alto_flag",  # gain=0, shap=0
    "pix_freq_high_flag",       # gain=0, shap=0
    "period_of_day",            # gain=0, shap=0 (hour já captura)
]


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


def extract_app_version_minor(version):
    if pd.isna(version):
        return np.nan
    parts = str(version).strip().split(".")
    if len(parts) >= 2 and parts[1].isdigit():
        return float(parts[1])
    return np.nan


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


def deduplicate_by_key(df, key_col, priority_cols, extra_priority_cols=None):
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


def encode_metodo_autenticacao(x):
    """Converte método de autenticação em valor numérico ordinal.
    biometria=0 (mais seguro), senha=1, pin=2, desconhecido=3
    """
    if pd.isna(x):
        return 3  # desconhecido
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
    return 3  # desconhecido


# =========================================================
# DIAGNOSTICS
# =========================================================
def diagnose_features(df, id_cols, label_col="is_fraud"):
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


# =========================================================
# PREPROCESSOR CLASS
# =========================================================
class PixPreprocessor:
    def __init__(self):
        self.numeric_imputer_ = {}
        self.categorical_fill_value_ = "__MISSING__"
        self.categorical_columns_ = []
        self.numeric_columns_ = []
        self.model_columns_ = []
        self.categorical_levels_ = {}
        self.host_time_median_ = 0.0
        self.dropped_high_null_ = []
        self.dropped_explicit_ = []

        self.id_columns_ = [
            "transaction_id", "customer_id", "event_datetime",
            "source_dataset", "is_fraud",
        ]

        self.exclude_from_model_ = [
            "cd_cpf_cnpj_recebedor", "ds_chave_pix", "ds_tipo_chave",
            "device_name", "device_name_normalized", "app_version",
            "ip_address", "metodo_autenticacao", "session_id",
            "cd_retorno",
            # Textos brutos dos novos campos (flags derivadas vão ao modelo)
            "ds_sexo", "ds_estado_civil", "ds_segmento",
            "is_agendamento_recorrente",  # texto bruto — flag numérica vai ao modelo
        ]

    def fit(self, df, null_threshold=0.95, explicit_drop=None):
        df = df.copy()
        if explicit_drop is None:
            explicit_drop = []

        cols_to_drop = [c for c in self.exclude_from_model_ if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors="ignore")

        # Drop explicit features
        self.dropped_explicit_ = [c for c in explicit_drop if c in df.columns]
        if self.dropped_explicit_:
            df = df.drop(columns=self.dropped_explicit_, errors="ignore")

        feature_cols = [c for c in df.columns if c not in self.id_columns_]

        # Drop high null
        self.dropped_high_null_ = []
        for c in feature_cols:
            null_pct = df[c].isna().mean()
            if null_pct > null_threshold:
                self.dropped_high_null_.append((c, round(null_pct, 4)))

        drop_null_names = [x[0] for x in self.dropped_high_null_]
        if drop_null_names:
            print(f"  ⚠ Removendo {len(drop_null_names)} features com >{null_threshold*100:.0f}% null:")
            for name, pct in self.dropped_high_null_:
                print(f"    - {name}: {pct*100:.1f}% null")
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
            v = pd.to_numeric(df["tempo_processamento_host_ms"], errors="coerce").median(skipna=True)
            self.host_time_median_ = v if not pd.isna(v) else 0.0

        self.categorical_levels_ = {}
        for c in self.categorical_columns_:
            vals = sorted(df[c].fillna(self.categorical_fill_value_).astype(str).unique().tolist())
            self.categorical_levels_[c] = vals

        model_cols = list(self.numeric_columns_)
        for c in self.categorical_columns_:
            for level in self.categorical_levels_[c]:
                model_cols.append(f"{c}__{level}")
        self.model_columns_ = model_cols
        return self

    def transform(self, df):
        df = df.copy()
        df = df.drop(columns=[c for c in self.exclude_from_model_ if c in df.columns], errors="ignore")
        df = df.drop(columns=[c for c in self.dropped_explicit_ if c in df.columns], errors="ignore")
        drop_null_names = [x[0] for x in self.dropped_high_null_]
        df = df.drop(columns=[c for c in drop_null_names if c in df.columns], errors="ignore")

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

    def fit_transform(self, df, null_threshold=0.95, explicit_drop=None):
        self.fit(df, null_threshold=null_threshold, explicit_drop=explicit_drop)
        return self.transform(df)


# =========================================================
# PIPELINE FUNCTIONS
# =========================================================
def load_and_prepare_pix(path_normal, path_fraud):
    print("  Carregando PIX normais...")
    pix_normal = standardize_columns(pd.read_csv(path_normal, low_memory=False))
    print(f"    → {len(pix_normal)} linhas")

    print("  Carregando PIX fraudes...")
    pix_fraud = standardize_columns(pd.read_csv(path_fraud, low_memory=False))
    print(f"    → {len(pix_fraud)} linhas")

    pix_normal["is_fraud"] = 0
    pix_normal["source_dataset"] = "normal"

    if "tp_fraude" in pix_fraud.columns:
        pix_fraud["is_fraud"] = pd.to_numeric(pix_fraud["tp_fraude"], errors="coerce").fillna(1).astype(int)
    else:
        pix_fraud["is_fraud"] = 1
    pix_fraud["source_dataset"] = "fraud"

    # FIX #1: required_cols SEM duplicação
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
        "is_agendamento_recorrente",
        "qt_aparelhos_distintos_trimestre", "nr_idade", "qt_tempo_relacionamento_mes",
        # Campos da ingestão v2.1
        "ds_sexo", "ds_estado_civil", "ds_segmento",
        "tp_primeiro_envio_recebedor_trimestre", "qt_envio_recebedor_trimestre",
        # Campos v2.1b: renda e dependentes (da aoxb17)
        "vl_renda_cliente", "qt_dependentes",
        # Controle
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
        "is_agendamento_recorrente", "source_dataset",
        "ds_sexo", "ds_estado_civil", "ds_segmento",
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
        "vl_renda_cliente", "qt_dependentes",
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

    print("  Deduplicando PIX por cd_pix...")
    pix_priority = [
        "cd_cpf_cnpj_recebedor", "ds_chave_pix", "ds_tipo_chave",
        "device_name", "app_version", "ip_address", "latencia_rede_ms",
        "tempo_interacao_ms", "tempo_processamento_host_ms",
        "metodo_autenticacao", "session_id", "cd_retorno",
        "topaz_risk_score", "topaz_transacao_rejeitada",
        "is_agendamento_recorrente",
        "ds_sexo", "ds_estado_civil", "ds_segmento",
        "tp_primeiro_envio_recebedor_trimestre",
    ]
    pix_all = deduplicate_by_key(pix_all, "cd_pix", pix_priority)
    print(f"    → {len(pix_all)} transações únicas")
    return pix_all


def create_all_features(df, host_time_median=None):
    df = df.copy()

    df["transaction_id"] = df["cd_pix"]
    df["customer_id"] = df["cd_cpf_pagador"]
    df["event_datetime"] = df["dt_pix"]
    df = df[df["transaction_id"].notna()].copy()

    df["latencia_rede_ms_final"] = df["latencia_rede_ms"]

    # --- MISSING FLAGS (apenas as úteis) ---
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
    df["is_business_hours"] = df["hour"].between(8, 18, inclusive="both").fillna(False).astype(int)

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

    df["latencia_host_ratio"] = robust_divide(
        df["latencia_rede_ms_final"], df["tempo_processamento_host_ms"]
    )

    if host_time_median is None:
        host_time_median = pd.to_numeric(
            df["tempo_processamento_host_ms"], errors="coerce"
        ).median(skipna=True)
        if pd.isna(host_time_median):
            host_time_median = 0.0

    # --- TOPAZ: usar 0 quando missing (não -1) ---
    df["topaz_score_filled"] = df["topaz_risk_score"].fillna(0)

    # --- FLAGS v3 ---
    df["vl_pix_over_1000_flag"] = (df["vl_pix"] >= 1000).astype(int)
    df["is_first_tx_trimestre"] = (df["qt_total_pix_trimestre"] == 1).astype(int)

    # =========================================================
    # v2.1b: Features derivadas de RENDA
    # FIX #2: Indentação corrigida (4 espaços, nível da função)
    # =========================================================
    df["vl_renda_cliente"] = pd.to_numeric(df["vl_renda_cliente"], errors="coerce").fillna(0)
    df["qt_dependentes"] = pd.to_numeric(df["qt_dependentes"], errors="coerce").fillna(0)

    # Ratio PIX / Renda — extremamente discriminativo
    df["ratio_pix_renda"] = np.where(
        df["vl_renda_cliente"] > 0,
        df["vl_pix"] / df["vl_renda_cliente"],
        np.nan  # Renda desconhecida = NaN (model trata via missing)
    )

    # Flag: PIX > 50% da renda mensal
    df["pix_over_50pct_renda_flag"] = np.where(
        df["vl_renda_cliente"] > 0,
        (df["vl_pix"] > df["vl_renda_cliente"] * 0.5).astype(int),
        0  # Conservador quando renda desconhecida
    )

    # Flag: PIX > 100% da renda mensal (alerta máximo)
    df["pix_over_100pct_renda_flag"] = np.where(
        df["vl_renda_cliente"] > 0,
        (df["vl_pix"] > df["vl_renda_cliente"]).astype(int),
        0
    )

    # Missing flag para renda
    df["renda_missing_flag"] = (df["vl_renda_cliente"] <= 0).astype(int)

    # =========================================================
    # v3.1: Features resgatadas e novas
    # =========================================================

    # --- TEMPO DE INTERAÇÃO (antes descartado — agora feature real) ---
    df["tempo_interacao_ms_final"] = df["tempo_interacao_ms"]
    df["ratio_tempo_interacao_cliente"] = robust_divide(
        df["tempo_interacao_ms_final"], df["vl_tempo_interacao_medio_trimestre"]
    )
    df["diff_tempo_interacao_cliente"] = (
        df["tempo_interacao_ms_final"] - df["vl_tempo_interacao_medio_trimestre"]
    )

    # --- MÉTODO DE AUTENTICAÇÃO (antes descartado — agora encoded) ---
    df["metodo_auth_encoded"] = df["metodo_autenticacao"].apply(encode_metodo_autenticacao)
    df["is_login_senha_flag"] = (df["metodo_auth_encoded"] == 1).astype(int)
    df["is_login_biometria_flag"] = (df["metodo_auth_encoded"] == 0).astype(int)

    # --- AGENDAMENTO RECORRENTE (antes descartado — agora flag) ---
    df["is_agendamento_recorrente_flag"] = (
        df["is_agendamento_recorrente"].astype(str).str.strip().str.lower() == "true"
    ).astype(int)

    # --- TOPAZ REJEITADA (antes descartado — agora flag) ---
    df["topaz_rejeitada_flag"] = (
        pd.to_numeric(df["topaz_transacao_rejeitada"], errors="coerce").fillna(0) == 1
    ).astype(int)

    # --- SEXO (campo v2.1) ---
    df["is_sexo_feminino_flag"] = (
        df["ds_sexo"].astype(str).str.strip().str.upper() == "F"
    ).astype(int)

    # --- ESTADO CIVIL (campo v2.1) ---
    df["is_viuvo_flag"] = (
        df["ds_estado_civil"].astype(str).str.strip().str.upper().str.contains("VIUV", na=False)
    ).astype(int)

    # --- SEGMENTO (campo v2.1) ---
    _seg = df["ds_segmento"].astype(str).str.strip().str.upper()
    df["is_segmento_premium_flag"] = _seg.isin([
        "EXCLUSIVO", "PRIVATE", "MILLENIUM", "MILLENIUM CAPIT", "PREMIUM", "VIP"
    ]).astype(int)

    # --- PRIMEIRO ENVIO AO RECEBEDOR (campo v2.1) ---
    df["tp_primeiro_envio_recebedor_trimestre"] = (
        pd.to_numeric(df["tp_primeiro_envio_recebedor_trimestre"], errors="coerce").fillna(0).astype(int)
    )

    # --- QT ENVIO RECEBEDOR TRIMESTRE (campo v2.1) ---
    df["qt_envio_recebedor_trimestre"] = (
        pd.to_numeric(df["qt_envio_recebedor_trimestre"], errors="coerce").fillna(0)
    )

    # =========================================================
    # FIX #3: perfil_vulneravel_se_flag APÓS criação de is_viuvo_flag
    # Depende de: is_viuvo_flag, nr_idade, qt_dependentes
    # =========================================================
    df["perfil_vulneravel_se_flag"] = (
        (df["is_viuvo_flag"] == 1) &
        (df["nr_idade"] >= 60) &
        (df["qt_dependentes"] == 0)
    ).astype(int)

    # =========================================================
    # FIM features v3.1
    # =========================================================

    # --- SEQUENCIAIS ---
    print("  Criando features sequenciais por cliente...")
    df = df.sort_values(["customer_id", "event_datetime", "transaction_id"]).reset_index(drop=True)

    df["prev_event_datetime"] = df.groupby("customer_id")["event_datetime"].shift(1)
    df["minutes_since_prev_tx"] = (
        (df["event_datetime"] - df["prev_event_datetime"]).dt.total_seconds() / 60.0
    )

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

    # --- REGRAS (apenas as que contribuem) ---
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
        [df["pix_key_random_flag"] == 1, df["pix_key_email_flag"] == 1, df["pix_key_document_flag"] == 1],
        [2, 1, 0], default=0,
    )

    df["rule_velocity_score"] = np.select(
        [df["tx_count_prev_30m"] == 0, df["tx_count_prev_30m"] == 1,
         df["tx_count_prev_30m"] == 2, df["tx_count_prev_30m"] >= 3],
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


def deduplicate_final(df):
    priority_cols = [
        "latencia_rede_ms_final", "tempo_processamento_host_ms",
        "device_name", "app_version", "ip_address", "metodo_autenticacao",
        "cd_retorno", "topaz_risk_score", "cd_cpf_cnpj_recebedor",
        "ds_chave_pix", "ds_tipo_chave", "session_id",
        "tempo_interacao_ms", "ds_sexo", "ds_estado_civil",
    ]
    return deduplicate_by_key(df.copy(), "transaction_id", priority_cols)


def select_final_columns(df):
    final_cols = [
        # IDs
        "transaction_id", "customer_id", "event_datetime", "source_dataset", "is_fraud",
        # Texto bruto (excluído no fit)
        "cd_cpf_cnpj_recebedor", "ds_chave_pix", "ds_tipo_chave",
        "device_name", "device_name_normalized", "app_version",
        "ip_address", "metodo_autenticacao", "session_id", "cd_retorno",
        # Texto bruto dos novos campos (excluído no fit, usado por SE/Behavioral)
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
        # Tempo de interação (resgatado v3.1)
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
        # Topaz rejeitada flag (v3.1)
        "topaz_rejeitada_flag",
        # Método de autenticação encoded (v3.1)
        "metodo_auth_encoded", "is_login_senha_flag", "is_login_biometria_flag",
        # Agendamento recorrente flag (v3.1)
        "is_agendamento_recorrente_flag",
        # Perfil do cliente (v2.1)
        "is_sexo_feminino_flag", "is_viuvo_flag", "is_segmento_premium_flag",
        # Primeiro envio e contagem recebedor (v2.1)
        "tp_primeiro_envio_recebedor_trimestre", "qt_envio_recebedor_trimestre",
        # Missing flags (úteis)
        "device_missing_flag", "app_version_missing_flag",
        "auth_method_missing_flag", "topaz_missing_flag",
        "host_time_missing_flag", "latencia_missing_flag",
        "tempo_interacao_missing_flag",
        # FIX #4: Renda e perfil de vulnerabilidade (v2.1b) — seção organizada
        "vl_renda_cliente", "qt_dependentes",
        "ratio_pix_renda", "pix_over_50pct_renda_flag", "pix_over_100pct_renda_flag",
        "renda_missing_flag", "perfil_vulneravel_se_flag",
        # Regras
        "rule_age_score", "rule_relationship_score",
        "rule_mule_account_score", "rule_random_key_score",
        "rule_velocity_score", "rule_topaz_score",
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
    print("PREPROCESSING v3.1 — MVP Fraude PIX")
    print("  Features resgatadas | Novos campos v2.1 | SE/Behavioral ready")
    print("=" * 70)

    print("\n[1/7] Carregando dados...")
    pix_all = load_and_prepare_pix(PATH_PIX_NORMAL, PATH_PIX_FRAUD)

    print("\n[2/7] Usando TODOS os dados...")
    df = pix_all.copy()
    n_fraud = int(df["is_fraud"].sum())
    n_normal = int((df["is_fraud"] == 0).sum())
    print(f"  Fraudes: {n_fraud} | Normais: {n_normal}")
    print(f"  Proporção fraude: {n_fraud / (n_fraud + n_normal) * 100:.2f}%")

    print("\n[3/7] Feature engineering...")
    df = create_all_features(df)

    print("\n[4/7] Deduplicação final...")
    before = len(df)
    df = deduplicate_final(df)
    after = len(df)
    print(f"  {before} → {after} ({before - after} duplicatas removidas)")

    print("\n[5/7] Diagnóstico de features...")
    df_features = select_final_columns(df)
    id_cols = ["transaction_id", "customer_id", "event_datetime", "source_dataset", "is_fraud"]
    diag = diagnose_features(df_features, id_cols)
    diag.to_csv(OUTPUT_DIAGNOSTICO, index=False)
    print(f"  Salvo em: {OUTPUT_DIAGNOSTICO}")

    n_fraud = int(df_features["is_fraud"].sum())
    n_normal = int((df_features["is_fraud"] == 0).sum())
    print(f"\n  Resumo:")
    print(f"    Fraudes:       {n_fraud}")
    print(f"    Normais:       {n_normal}")
    print(f"    Proporção:     {n_fraud / (n_fraud + n_normal) * 100:.2f}% fraude")
    print(f"    Tx únicas:     {df_features['transaction_id'].nunique()}")

    # Log de cobertura dos novos campos
    print(f"\n  Cobertura dos novos campos v3.1:")
    total = len(df_features)
    novos_campos = [
        "ds_sexo", "ds_estado_civil", "ds_segmento",
        "tp_primeiro_envio_recebedor_trimestre", "qt_envio_recebedor_trimestre",
        "tempo_interacao_ms_final", "metodo_auth_encoded",
        "is_agendamento_recorrente_flag", "topaz_rejeitada_flag",
        "is_sexo_feminino_flag", "is_viuvo_flag", "is_segmento_premium_flag",
        "vl_renda_cliente", "qt_dependentes",
        "ratio_pix_renda", "pix_over_50pct_renda_flag", "pix_over_100pct_renda_flag",
        "renda_missing_flag", "perfil_vulneravel_se_flag",
    ]
    for col_name in novos_campos:
        if col_name in df_features.columns:
            if df_features[col_name].dtype == "object":
                not_null = df_features[col_name].notna().sum()
                not_null -= (df_features[col_name].astype(str).str.lower().isin(
                    ["nan", "informação ausente", ""]
                )).sum()
            else:
                not_null = df_features[col_name].notna().sum()
            pct = round((not_null / total) * 100, 2) if total > 0 else 0
            print(f"    {col_name}: {int(not_null)}/{total} ({pct}%)")

    print(f"\n  Features explicitamente removidas ({len(FEATURES_TO_DROP_FROM_MODEL)}):")
    for f in sorted(FEATURES_TO_DROP_FROM_MODEL):
        print(f"    ✗ {f}")

    print("\n[6/7] Fit/transform do preprocessor...")
    preprocessor = PixPreprocessor()
    feature_only = df_features.drop(columns=preprocessor.id_columns_, errors="ignore")
    X_ready = preprocessor.fit_transform(
        feature_only,
        null_threshold=NULL_THRESHOLD,
        explicit_drop=FEATURES_TO_DROP_FROM_MODEL,
    )

    if "tempo_processamento_host_ms" in feature_only.columns:
        v = pd.to_numeric(feature_only["tempo_processamento_host_ms"], errors="coerce").median(skipna=True)
        preprocessor.host_time_median_ = v if not pd.isna(v) else 0.0

    model_ready = pd.concat(
        [df_features[preprocessor.id_columns_].reset_index(drop=True), X_ready.reset_index(drop=True)],
        axis=1,
    )

    print(f"\n  Salvando model-ready em: {OUTPUT_MODEL_READY}")
    model_ready.to_csv(OUTPUT_MODEL_READY, index=False)

    print(f"\n[7/7] Salvando artefato em: {OUTPUT_PREPROCESSOR}")
    joblib.dump(preprocessor, OUTPUT_PREPROCESSOR)

    print("\n" + "=" * 70)
    print("CONCLUÍDO — PREPROCESSING v3.1")
    print(f"  base_mvp_model_ready:  {model_ready.shape}")
    print(f"  Colunas do modelo:     {len(preprocessor.model_columns_)}")
    print(f"  Numéricas:             {len(preprocessor.numeric_columns_)}")
    print(f"  Categóricas:           {len(preprocessor.categorical_columns_)}")
    print(f"  Excluídas (texto):     {len(preprocessor.exclude_from_model_)}")
    print(f"  Excluídas (explícito): {len(preprocessor.dropped_explicit_)}")
    print(f"  Excluídas (alta null): {len(preprocessor.dropped_high_null_)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
