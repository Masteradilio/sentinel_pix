"""
    
    O que o script faz
lê:
/dados/dados_pix_normais.csv
/dados/dados_fraudes_pix.csv
/dados/dados_features_mobile.csv
padroniza os nomes das colunas
cria is_fraud
concatena normais + fraudes
tenta juntar com mobile via:
cd_pix ↔ end_to_end_id
cria as features derivadas do MVP

salva:
/dados/base_mvp_features.csv
"""



import os
import re
import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DADOS_DIR = os.path.join(PROJECT_ROOT, "dados")

PATH_PIX_NORMAL = os.path.join(DADOS_DIR, "dados_pix_normais.csv")
PATH_PIX_FRAUD = os.path.join(DADOS_DIR, "dados_fraudes_pix.csv")
PATH_MOBILE = os.path.join(DADOS_DIR, "dados_features_mobile.csv")

OUTPUT_PATH = os.path.join(DADOS_DIR, "base_mvp_features.csv")


# =========================================================
# HELPERS
# =========================================================
NULL_STRINGS = {"", "null", "none", "nan", "nat", "missing"}


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
    x = re.sub(r"\s+", "_", x)
    return x


def extract_app_version_major(version):
    if pd.isna(version):
        return np.nan
    version = str(version).strip()
    m = re.match(r"^(\d+)", version)
    return float(m.group(1)) if m else np.nan


def extract_app_version_minor(version):
    if pd.isna(version):
        return np.nan
    version = str(version).strip()
    parts = version.split(".")
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
    # remove espaços extras invisíveis
    s = s.str.replace(r"\s+", "", regex=True)
    return s


def count_non_null_priority(df: pd.DataFrame, cols: list) -> pd.Series:
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return pd.Series(0, index=df.index)
    return df[valid_cols].notna().sum(axis=1)


def deduplicate_by_key(df: pd.DataFrame, key_col: str, priority_cols: list, extra_priority_cols=None) -> pd.DataFrame:
    df = df.copy()

    if extra_priority_cols is None:
        extra_priority_cols = []

    df["non_null_count"] = df.notna().sum(axis=1)
    df["priority_count"] = count_non_null_priority(df, priority_cols)

    for c in extra_priority_cols:
        if c not in df.columns:
            df[c] = 0

    sort_cols = [key_col, "non_null_count", "priority_count"] + extra_priority_cols
    ascending = [True, False, False] + [False] * len(extra_priority_cols)

    df = df.sort_values(sort_cols, ascending=ascending)
    df = df.drop_duplicates(subset=[key_col], keep="first").copy()

    df = df.drop(columns=["non_null_count", "priority_count"], errors="ignore")
    return df


def deduplicate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    priority_cols = [
        "latencia_rede_ms_final",
        "tempo_interacao_ms_final",
        "tempo_processamento_host_ms",
        "device_name",
        "app_version",
        "ip_address",
        "metodo_autenticacao",
        "cd_retorno",
        "topaz_risk_score",
        "topaz_transacao_rejeitada",
        "topaz_transacao_habilitada",
        "is_agendamento_recorrente",
    ]

    df["has_mobile_join"] = df["join_status_mobile"].fillna(0).astype(int)
    df["has_latencia_final"] = df["latencia_rede_ms_final"].notna().astype(int)
    df["has_tempo_interacao_final"] = df["tempo_interacao_ms_final"].notna().astype(int)
    df["has_host_time"] = df["tempo_processamento_host_ms"].notna().astype(int)

    df = deduplicate_by_key(
        df,
        key_col="transaction_id",
        priority_cols=priority_cols,
        extra_priority_cols=[
            "has_mobile_join",
            "has_latencia_final",
            "has_tempo_interacao_final",
            "has_host_time",
        ]
    )

    df = df.drop(
        columns=[
            "has_mobile_join",
            "has_latencia_final",
            "has_tempo_interacao_final",
            "has_host_time",
        ],
        errors="ignore"
    )

    return df


# =========================================================
# LOAD RAW DATA
# =========================================================
print("Carregando arquivos brutos...")

pix_normal = pd.read_csv(PATH_PIX_NORMAL, low_memory=False)
pix_fraud = pd.read_csv(PATH_PIX_FRAUD, low_memory=False)
mobile = pd.read_csv(PATH_MOBILE, low_memory=False)

pix_normal = standardize_columns(pix_normal)
pix_fraud = standardize_columns(pix_fraud)
mobile = standardize_columns(mobile)


# =========================================================
# PIX BASES
# =========================================================
pix_normal["is_fraud"] = 0
pix_normal["source_dataset"] = "normal"

pix_fraud["is_fraud"] = 1
pix_fraud["source_dataset"] = "fraud"

required_pix_cols = [
    "cd_pix",
    "dt_pix",
    "cd_cpf_pagador",
    "vl_pix",
    "qt_total_pix_trimestre",
    "vl_mediana_pix_trimestre",
    "vl_desvio_padrao_pix_trimestre",
    "qt_intervalo_transacao_minuto",
    "qt_intervalo_mediana_trimestre",
    "qt_intervalo_desvio_padrao_trimestre",
    "qt_pix_dia_maximo_trimestre",
    "latencia_rede_ms",
    "vl_latencia_rede_media_trimestre",
    "tempo_interacao_ms",
    "vl_tempo_interacao_medio_trimestre",
    "qt_aparelhos_distintos_trimestre",
    "nr_idade",
    "qt_tempo_relacionamento_mes",
    "dt_carga",
    "is_fraud",
    "source_dataset",
]

for c in required_pix_cols:
    pix_normal = ensure_column(pix_normal, c, np.nan)
    pix_fraud = ensure_column(pix_fraud, c, np.nan)

pix_all = pd.concat(
    [pix_normal[required_pix_cols], pix_fraud[required_pix_cols]],
    ignore_index=True,
    sort=False
)

pix_numeric_cols = [
    "vl_pix",
    "qt_total_pix_trimestre",
    "vl_mediana_pix_trimestre",
    "vl_desvio_padrao_pix_trimestre",
    "qt_intervalo_transacao_minuto",
    "qt_intervalo_mediana_trimestre",
    "qt_intervalo_desvio_padrao_trimestre",
    "qt_pix_dia_maximo_trimestre",
    "latencia_rede_ms",
    "vl_latencia_rede_media_trimestre",
    "tempo_interacao_ms",
    "vl_tempo_interacao_medio_trimestre",
    "qt_aparelhos_distintos_trimestre",
    "nr_idade",
    "qt_tempo_relacionamento_mes",
    "is_fraud",
]

pix_all = safe_to_numeric(pix_all, pix_numeric_cols)
pix_all = safe_to_datetime(pix_all, ["dt_pix", "dt_carga"])

pix_all["cd_pix"] = normalize_transaction_key(pix_all["cd_pix"])
pix_all["cd_cpf_pagador"] = pd.to_numeric(pix_all["cd_cpf_pagador"], errors="coerce")

pix_all = replace_sentinels_with_nan(
    pix_all,
    [
        "latencia_rede_ms",
        "vl_latencia_rede_media_trimestre",
        "tempo_interacao_ms",
        "vl_tempo_interacao_medio_trimestre",
    ],
    sentinels=[-1, -1.0]
)

# zeros que provavelmente representam ausência em campos de interação
pix_all = replace_zero_with_nan(
    pix_all,
    [
        "vl_tempo_interacao_medio_trimestre",
    ]
)


# =========================================================
# MOBILE BASE
# =========================================================
mobile_required_cols = [
    "end_to_end_id",
    "data_hora_inicio",
    "nr_conta",
    "valor_transacao",
    "cd_tipo_transacao",
    "cd_retorno",
    "device_name",
    "app_version",
    "ip_address",
    "latencia_rede_ms",
    "tempo_interacao_ms",
    "tempo_processamento_host_ms",
    "metodo_autenticacao",
    "session_id",
    "topaz_risk_score",
    "topaz_transacao_rejeitada",
    "topaz_transacao_habilitada",
    "is_agendamento_recorrente",
]

for c in mobile_required_cols:
    mobile = ensure_column(mobile, c, np.nan)

mobile = mobile[mobile_required_cols].copy()

mobile_text_cols = [
    "end_to_end_id",
    "device_name",
    "app_version",
    "ip_address",
    "metodo_autenticacao",
    "session_id",
    "cd_retorno",
]

mobile = clean_text_columns(mobile, mobile_text_cols)

mobile_numeric_cols = [
    "valor_transacao",
    "cd_tipo_transacao",
    "latencia_rede_ms",
    "tempo_interacao_ms",
    "tempo_processamento_host_ms",
    "topaz_risk_score",
    "topaz_transacao_rejeitada",
    "topaz_transacao_habilitada",
    "is_agendamento_recorrente",
]

mobile = safe_to_numeric(mobile, mobile_numeric_cols)
mobile = safe_to_datetime(mobile, ["data_hora_inicio"])

mobile["end_to_end_id"] = normalize_transaction_key(mobile["end_to_end_id"])

mobile = replace_sentinels_with_nan(
    mobile,
    [
        "latencia_rede_ms",
        "tempo_interacao_ms",
        "tempo_processamento_host_ms",
        "topaz_risk_score",
        "topaz_transacao_rejeitada",
        "topaz_transacao_habilitada",
    ],
    sentinels=[-1, -1.0]
)

mobile = mobile.rename(columns={
    "end_to_end_id": "cd_pix_mobile",
    "latencia_rede_ms": "mobile_latencia_rede_ms",
    "tempo_interacao_ms": "mobile_tempo_interacao_ms",
    "valor_transacao": "mobile_valor_transacao",
})

# deduplicar mobile por chave antes do merge
print("Deduplicando base mobile por end_to_end_id...")
mobile_priority_cols = [
    "mobile_latencia_rede_ms",
    "mobile_tempo_interacao_ms",
    "tempo_processamento_host_ms",
    "device_name",
    "app_version",
    "ip_address",
    "metodo_autenticacao",
    "cd_retorno",
    "topaz_risk_score",
    "topaz_transacao_rejeitada",
    "topaz_transacao_habilitada",
    "is_agendamento_recorrente",
]
mobile_before = len(mobile)
mobile = deduplicate_by_key(
    mobile,
    key_col="cd_pix_mobile",
    priority_cols=mobile_priority_cols
)
mobile_after = len(mobile)
print(f"Linhas mobile antes: {mobile_before}")
print(f"Linhas mobile após deduplicação: {mobile_after}")


# =========================================================
# JOIN PIX + MOBILE
# =========================================================
print("Realizando join PIX + mobile...")

df = pix_all.merge(
    mobile,
    how="left",
    left_on="cd_pix",
    right_on="cd_pix_mobile"
)

df["join_status_mobile"] = np.where(df["cd_pix_mobile"].notna(), 1, 0)

df["transaction_id"] = df["cd_pix"]
df["customer_id"] = df["cd_cpf_pagador"]

df["event_datetime"] = df["dt_pix"]
mask_dt = df["event_datetime"].isna()
df.loc[mask_dt, "event_datetime"] = df.loc[mask_dt, "data_hora_inicio"]

df["latencia_rede_ms_final"] = df["latencia_rede_ms"]
mask_lat = df["latencia_rede_ms_final"].isna()
df.loc[mask_lat, "latencia_rede_ms_final"] = df.loc[mask_lat, "mobile_latencia_rede_ms"]

df["tempo_interacao_ms_final"] = df["tempo_interacao_ms"]
mask_tempo = df["tempo_interacao_ms_final"].isna()
df.loc[mask_tempo, "tempo_interacao_ms_final"] = df.loc[mask_tempo, "mobile_tempo_interacao_ms"]


# =========================================================
# BASIC CLEANING
# =========================================================
df = df[df["transaction_id"].notna()].copy()

print("Deduplicando transações...")
before_dedup = len(df)
df = deduplicate_transactions(df)
after_dedup = len(df)

df = df.sort_values(["customer_id", "event_datetime", "transaction_id"]).reset_index(drop=True)

print(f"Linhas antes da deduplicação final: {before_dedup}")
print(f"Linhas após deduplicação final: {after_dedup}")
print(f"Duplicatas removidas: {before_dedup - after_dedup}")


# =========================================================
# MISSING FLAGS
# =========================================================
df["device_missing_flag"] = df["device_name"].isna().astype(int)
df["ip_missing_flag"] = df["ip_address"].isna().astype(int)
df["app_version_missing_flag"] = df["app_version"].isna().astype(int)
df["auth_method_missing_flag"] = df["metodo_autenticacao"].isna().astype(int)
df["topaz_missing_flag"] = df["topaz_risk_score"].isna().astype(int)
df["host_time_missing_flag"] = df["tempo_processamento_host_ms"].isna().astype(int)

df["latencia_missing_flag"] = df["latencia_rede_ms_final"].isna().astype(int)
df["tempo_interacao_missing_flag"] = df["tempo_interacao_ms_final"].isna().astype(int)
df["mobile_join_missing_flag"] = (df["join_status_mobile"] == 0).astype(int)


# =========================================================
# DEVICE / APP FEATURES
# =========================================================
df["device_name_normalized"] = df["device_name"].apply(normalize_device_name)
df["app_version_major"] = df["app_version"].apply(extract_app_version_major)
df["app_version_minor"] = df["app_version"].apply(extract_app_version_minor)


# =========================================================
# TIME FEATURES
# =========================================================
df["hour"] = df["event_datetime"].dt.hour
df["day_of_week"] = df["event_datetime"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
df["is_business_hours"] = df["hour"].between(8, 18, inclusive="both").fillna(False).astype(int)
df["period_of_day"] = df["hour"].apply(period_of_day)


# =========================================================
# FEATURE ENGINEERING
# =========================================================
print("Criando features derivadas...")

df["log_vl_pix"] = np.log1p(df["vl_pix"].clip(lower=0))

df["ratio_valor_mediana"] = robust_divide(df["vl_pix"], df["vl_mediana_pix_trimestre"])
df["diff_valor_mediana"] = df["vl_pix"] - df["vl_mediana_pix_trimestre"]

df["ratio_valor_desvio_padrao"] = robust_divide(df["vl_pix"], df["vl_desvio_padrao_pix_trimestre"])
df["zscore_valor_aprox"] = robust_divide(
    df["vl_pix"] - df["vl_mediana_pix_trimestre"],
    df["vl_desvio_padrao_pix_trimestre"]
)

df["ratio_intervalo_vs_mediana"] = robust_divide(
    df["qt_intervalo_transacao_minuto"],
    df["qt_intervalo_mediana_trimestre"]
)
df["diff_intervalo_vs_mediana"] = (
    df["qt_intervalo_transacao_minuto"] - df["qt_intervalo_mediana_trimestre"]
)
df["zscore_intervalo_aprox"] = robust_divide(
    df["qt_intervalo_transacao_minuto"] - df["qt_intervalo_mediana_trimestre"],
    df["qt_intervalo_desvio_padrao_trimestre"]
)

df["ratio_latencia_cliente"] = robust_divide(
    df["latencia_rede_ms_final"],
    df["vl_latencia_rede_media_trimestre"]
)
df["diff_latencia_cliente"] = (
    df["latencia_rede_ms_final"] - df["vl_latencia_rede_media_trimestre"]
)

df["ratio_tempo_interacao_cliente"] = robust_divide(
    df["tempo_interacao_ms_final"],
    df["vl_tempo_interacao_medio_trimestre"]
)
df["diff_tempo_interacao_cliente"] = (
    df["tempo_interacao_ms_final"] - df["vl_tempo_interacao_medio_trimestre"]
)

df["tempo_interacao_baixo_flag"] = (
    (df["ratio_tempo_interacao_cliente"] < 0.5)
).fillna(False).astype(int)

df["tempo_interacao_alto_flag"] = (
    (df["ratio_tempo_interacao_cliente"] > 2.0)
).fillna(False).astype(int)

df["latencia_host_ratio"] = robust_divide(
    df["latencia_rede_ms_final"],
    df["tempo_processamento_host_ms"]
)

host_median = df["tempo_processamento_host_ms"].median(skipna=True)
df["processamento_host_alto_flag"] = (
    df["tempo_processamento_host_ms"] > host_median
).fillna(False).astype(int)


# =========================================================
# CUSTOMER SEQUENCE FEATURES
# =========================================================
print("Criando features sequenciais...")

df["prev_event_datetime"] = df.groupby("customer_id")["event_datetime"].shift(1)
df["minutes_since_prev_tx"] = (
    (df["event_datetime"] - df["prev_event_datetime"]).dt.total_seconds() / 60.0
)

df["tx_count_prev_30m"] = 0

for cust_id, group in df.groupby("customer_id", sort=False):
    idx = group.index.to_list()
    times = group["event_datetime"].tolist()

    counts = []
    for i, current_time in enumerate(times):
        if pd.isna(current_time):
            counts.append(0)
            continue

        c = 0
        j = i - 1
        while j >= 0:
            prev_time = times[j]
            if pd.isna(prev_time):
                j -= 1
                continue

            diff_min = (current_time - prev_time).total_seconds() / 60.0
            if 0 <= diff_min <= 30:
                c += 1
                j -= 1
            else:
                break
        counts.append(c)

    df.loc[idx, "tx_count_prev_30m"] = counts

df["burst_30m_flag"] = (df["tx_count_prev_30m"] >= 1).astype(int)

df["pix_freq_high_flag"] = (
    (df["ratio_intervalo_vs_mediana"] < 0.5) |
    (df["tx_count_prev_30m"] >= 2)
).fillna(False).astype(int)


# =========================================================
# RULE FEATURES
# =========================================================
df["rule_pix_30m_score"] = np.select(
    [
        df["tx_count_prev_30m"] == 0,
        df["tx_count_prev_30m"] == 1,
        df["tx_count_prev_30m"] >= 2
    ],
    [0, 1, 2],
    default=0
)

df["rule_ratio_pix_limite_score"] = np.nan

df["rule_age_score"] = np.select(
    [
        df["nr_idade"].between(60, 65, inclusive="both"),
        df["nr_idade"].between(66, 75, inclusive="both"),
        df["nr_idade"] >= 76
    ],
    [1, 2, 3],
    default=0
)

df["is_elderly_flag"] = (df["nr_idade"] >= 60).fillna(False).astype(int)

df["rule_relationship_score"] = np.select(
    [
        df["qt_tempo_relacionamento_mes"].between(61, 90, inclusive="both"),
        df["qt_tempo_relacionamento_mes"].between(31, 60, inclusive="both"),
        df["qt_tempo_relacionamento_mes"].between(0, 30, inclusive="both"),
    ],
    [1, 2, 3],
    default=0
)

df["is_new_customer_flag"] = (
    df["qt_tempo_relacionamento_mes"].between(0, 30, inclusive="both")
).fillna(False).astype(int)

df["rule_mule_account_score"] = np.nan
df["rule_random_key_score"] = np.nan

df["rule_night_score"] = np.where(df["is_night"] == 1, 3, 0)

df["rule_velocity_score"] = np.select(
    [
        df["tx_count_prev_30m"] == 0,
        df["tx_count_prev_30m"] == 1,
        df["tx_count_prev_30m"] == 2,
        df["tx_count_prev_30m"] >= 3
    ],
    [0, 2, 3, 4],
    default=0
)

df["rule_topaz_score"] = df["topaz_risk_score"].apply(map_topaz_rule)

df["autorizacao_previa_flag"] = 0
df["rule_pre_authorization_discount"] = 0.0

rule_components = [
    "rule_pix_30m_score",
    "rule_age_score",
    "rule_relationship_score",
    "rule_night_score",
    "rule_velocity_score",
    "rule_topaz_score",
]

df["rule_score_raw"] = df[rule_components].fillna(0).sum(axis=1)

max_rule_score = df["rule_score_raw"].max(skipna=True)
if pd.isna(max_rule_score) or max_rule_score == 0:
    df["rule_score_normalized"] = 0.0
else:
    df["rule_score_normalized"] = df["rule_score_raw"] / max_rule_score


# =========================================================
# FINAL FEATURE SELECTION
# =========================================================
final_cols = [
    "transaction_id",
    "customer_id",
    "event_datetime",
    "source_dataset",
    "join_status_mobile",
    "is_fraud",

    "vl_pix",
    "log_vl_pix",
    "qt_total_pix_trimestre",
    "vl_mediana_pix_trimestre",
    "vl_desvio_padrao_pix_trimestre",
    "qt_intervalo_transacao_minuto",
    "qt_intervalo_mediana_trimestre",
    "qt_intervalo_desvio_padrao_trimestre",
    "qt_pix_dia_maximo_trimestre",
    "qt_aparelhos_distintos_trimestre",
    "nr_idade",
    "qt_tempo_relacionamento_mes",

    "latencia_rede_ms_final",
    "vl_latencia_rede_media_trimestre",
    "tempo_interacao_ms_final",
    "vl_tempo_interacao_medio_trimestre",
    "tempo_processamento_host_ms",

    "ratio_valor_mediana",
    "diff_valor_mediana",
    "ratio_valor_desvio_padrao",
    "zscore_valor_aprox",
    "ratio_intervalo_vs_mediana",
    "diff_intervalo_vs_mediana",
    "zscore_intervalo_aprox",
    "ratio_latencia_cliente",
    "diff_latencia_cliente",
    "ratio_tempo_interacao_cliente",
    "diff_tempo_interacao_cliente",
    "tempo_interacao_baixo_flag",
    "tempo_interacao_alto_flag",
    "latencia_host_ratio",
    "processamento_host_alto_flag",

    "minutes_since_prev_tx",
    "tx_count_prev_30m",
    "burst_30m_flag",
    "pix_freq_high_flag",

    "hour",
    "day_of_week",
    "is_weekend",
    "is_night",
    "is_business_hours",
    "period_of_day",

    "device_name",
    "device_name_normalized",
    "app_version",
    "app_version_major",
    "app_version_minor",
    "ip_address",
    "metodo_autenticacao",
    "cd_retorno",
    "topaz_risk_score",
    "topaz_transacao_rejeitada",
    "topaz_transacao_habilitada",
    "is_agendamento_recorrente",

    "device_missing_flag",
    "ip_missing_flag",
    "app_version_missing_flag",
    "auth_method_missing_flag",
    "topaz_missing_flag",
    "host_time_missing_flag",
    "latencia_missing_flag",
    "tempo_interacao_missing_flag",
    "mobile_join_missing_flag",

    "rule_pix_30m_score",
    "rule_ratio_pix_limite_score",
    "rule_age_score",
    "is_elderly_flag",
    "rule_relationship_score",
    "is_new_customer_flag",
    "rule_mule_account_score",
    "rule_random_key_score",
    "rule_night_score",
    "rule_velocity_score",
    "rule_topaz_score",
    "autorizacao_previa_flag",
    "rule_pre_authorization_discount",
    "rule_score_raw",
    "rule_score_normalized",
]

for c in final_cols:
    df = ensure_column(df, c, np.nan)

final_df = df[final_cols].copy()


# =========================================================
# DIAGNOSTIC LOGS
# =========================================================
fraud_source_incorrect = (
    (final_df["source_dataset"] == "fraud") &
    (final_df["is_fraud"] != 1)
).sum()

duplicated_transactions = final_df["transaction_id"].duplicated().sum()

print("\nDiagnósticos:")
print(f"Fraudes com label incorreto: {fraud_source_incorrect}")
print(f"Duplicatas restantes de transaction_id: {duplicated_transactions}")
print(f"Transações únicas: {final_df['transaction_id'].nunique()}")
print(f"Linhas com join mobile: {int(final_df['join_status_mobile'].sum())}")
print(f"Fraudes: {int(final_df['is_fraud'].sum())}")
print(f"Normais: {int((final_df['is_fraud'] == 0).sum())}")


# =========================================================
# SAVE
# =========================================================
final_df.to_csv(OUTPUT_PATH, index=False)

print("\nFeature engineering concluído com sucesso.")
print(f"Arquivo salvo em: {OUTPUT_PATH}")
print(f"Shape final: {final_df.shape}")
print("\nColunas finais:")
print(final_df.columns.tolist())


