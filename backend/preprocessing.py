"""
Esse script vai:

Ler os CSVs em /dados
Fazer a consolidação e feature engineering do MVP
Limpar duplicidades
Tratar missing
Codificar variáveis categóricas

Salvar:
/dados/base_mvp_features.csv
/dados/base_mvp_model_ready.csv

Gerar um artefato:
/backend/artefatos/preprocessing.joblib

Esse artefato vai guardar:

lista final de features
medians de imputação
colunas categóricas
categorias vistas no treino
colunas finais do modelo
lógica de transformação reutilizável

"""

import os
import re
import joblib
import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
# project root one level up from backend/ directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DADOS_DIR = os.path.join(PROJECT_ROOT, "dados")
ARTEFACT_DIR = os.path.join(PROJECT_ROOT, "backend", "artefatos")

PATH_PIX_NORMAL = os.path.join(DADOS_DIR, "dados_pix_normais.csv")
PATH_PIX_FRAUD = os.path.join(DADOS_DIR, "dados_fraudes_pix.csv")
PATH_MOBILE = os.path.join(DADOS_DIR, "dados_features_mobile.csv")

OUTPUT_FEATURES = os.path.join(DADOS_DIR, "base_mvp_features.csv")
OUTPUT_MODEL_READY = os.path.join(DADOS_DIR, "base_mvp_model_ready.csv")
OUTPUT_PREPROCESSOR = os.path.join(ARTEFACT_DIR, "preprocessing.joblib")

os.makedirs(DADOS_DIR, exist_ok=True)
os.makedirs(ARTEFACT_DIR, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================
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


def robust_divide(a, b):
    return np.where((pd.isna(b)) | (b == 0), np.nan, a / b)


def extract_app_version_major(version):
    if pd.isna(version):
        return np.nan
    version = str(version)
    m = re.match(r"^\s*(\d+)", version)
    return float(m.group(1)) if m else np.nan


def extract_app_version_minor(version):
    if pd.isna(version):
        return np.nan
    version = str(version)
    parts = version.split(".")
    if len(parts) >= 2 and parts[1].isdigit():
        return float(parts[1])
    return np.nan


def normalize_device_name(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    x = re.sub(r"\s+", "_", x)
    return x


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

        self.id_columns_ = [
            "transaction_id",
            "customer_id",
            "event_datetime",
            "source_dataset",
            "join_status_mobile",
            "is_fraud",
        ]

    def build_base_from_raw_files(self, pix_normal, pix_fraud, mobile):
        pix_normal = standardize_columns(pix_normal)
        pix_fraud = standardize_columns(pix_fraud)
        mobile = standardize_columns(mobile)

        pix_normal["is_fraud"] = 0
        pix_normal["source_dataset"] = "normal"

        if "tp_fraude" in pix_fraud.columns:
            pix_fraud["is_fraud"] = pd.to_numeric(
                pix_fraud["tp_fraude"], errors="coerce"
            ).fillna(1).astype(int)
        else:
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

        mobile_numeric_cols = [
            "valor_transacao",
            "cd_tipo_transacao",
            "cd_retorno",
            "latencia_rede_ms",
            "tempo_interacao_ms",
            "tempo_processamento_host_ms",
            "metodo_autenticacao",
            "topaz_risk_score",
            "topaz_transacao_rejeitada",
            "topaz_transacao_habilitada",
            "is_agendamento_recorrente",
        ]

        pix_all = safe_to_numeric(pix_all, pix_numeric_cols)
        mobile = safe_to_numeric(mobile, mobile_numeric_cols)

        pix_all = safe_to_datetime(pix_all, ["dt_pix", "dt_carga"])
        mobile = safe_to_datetime(mobile, ["data_hora_inicio"])

        mobile = mobile.rename(columns={
            "end_to_end_id": "cd_pix_mobile",
            "latencia_rede_ms": "mobile_latencia_rede_ms",
            "tempo_interacao_ms": "mobile_tempo_interacao_ms",
            "valor_transacao": "mobile_valor_transacao",
        })

        df = pix_all.merge(
            mobile,
            how="left",
            left_on="cd_pix",
            right_on="cd_pix_mobile"
        )

        df["join_status_mobile"] = np.where(df["cd_pix_mobile"].notna(), 1, 0)

        df["latencia_rede_ms_final"] = df["latencia_rede_ms"]
        df.loc[df["latencia_rede_ms_final"].isna(), "latencia_rede_ms_final"] = df.loc[
            df["latencia_rede_ms_final"].isna(), "mobile_latencia_rede_ms"
        ]

        df["tempo_interacao_ms_final"] = df["tempo_interacao_ms"]
        df.loc[df["tempo_interacao_ms_final"].isna(), "tempo_interacao_ms_final"] = df.loc[
            df["tempo_interacao_ms_final"].isna(), "mobile_tempo_interacao_ms"
        ]

        df["event_datetime"] = df["dt_pix"]
        df.loc[df["event_datetime"].isna(), "event_datetime"] = df.loc[
            df["event_datetime"].isna(), "data_hora_inicio"
        ]

        df["transaction_id"] = df["cd_pix"]
        df["customer_id"] = df["cd_cpf_pagador"]

        df = df[df["transaction_id"].notna()].copy()
        df = df.sort_values(["customer_id", "event_datetime", "transaction_id"]).reset_index(drop=True)

        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # missing flags
        df["device_missing_flag"] = df["device_name"].isna().astype(int)
        df["ip_missing_flag"] = df["ip_address"].isna().astype(int)
        df["app_version_missing_flag"] = df["app_version"].isna().astype(int)
        df["auth_method_missing_flag"] = df["metodo_autenticacao"].isna().astype(int)
        df["topaz_missing_flag"] = df["topaz_risk_score"].isna().astype(int)
        df["host_time_missing_flag"] = df["tempo_processamento_host_ms"].isna().astype(int)

        # device/app
        df["device_name_normalized"] = df["device_name"].apply(normalize_device_name)
        df["app_version_major"] = df["app_version"].apply(extract_app_version_major)
        df["app_version_minor"] = df["app_version"].apply(extract_app_version_minor)

        # time
        df["hour"] = df["event_datetime"].dt.hour
        df["day_of_week"] = df["event_datetime"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        df["is_business_hours"] = df["hour"].between(8, 18, inclusive="both").fillna(False).astype(int)
        df["period_of_day"] = df["hour"].apply(period_of_day)

        # core derived
        df["log_vl_pix"] = np.log1p(df["vl_pix"])
        df["ratio_valor_mediana"] = robust_divide(df["vl_pix"], df["vl_mediana_pix_trimestre"])
        df["diff_valor_mediana"] = df["vl_pix"] - df["vl_mediana_pix_trimestre"]
        df["ratio_valor_desvio_padrao"] = robust_divide(df["vl_pix"], df["vl_desvio_padrao_pix_trimestre"])
        df["zscore_valor_aprox"] = robust_divide(
            (df["vl_pix"] - df["vl_mediana_pix_trimestre"]),
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
            (df["qt_intervalo_transacao_minuto"] - df["qt_intervalo_mediana_trimestre"]),
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

        df["processamento_host_alto_flag"] = (
            df["tempo_processamento_host_ms"] >
            df["tempo_processamento_host_ms"].median(skipna=True)
        ).fillna(False).astype(int)

        # sequence
        df = df.sort_values(["customer_id", "event_datetime", "transaction_id"]).reset_index(drop=True)
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
                    if pd.isna(times[j]):
                        j -= 1
                        continue
                    diff_min = (current_time - times[j]).total_seconds() / 60.0
                    if diff_min <= 30:
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

        # rules
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

        df["topaz_score_filled"] = df["topaz_risk_score"].fillna(-1)

        return df

    def deduplicate_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicação por transaction_id priorizando:
        1. maior quantidade de campos preenchidos
        2. join com mobile
        3. maior topaz disponível
        """
        df = df.copy()

        df["non_null_count"] = df.notna().sum(axis=1)

        if "topaz_risk_score" not in df.columns:
            df["topaz_risk_score"] = np.nan

        if "join_status_mobile" not in df.columns:
            df["join_status_mobile"] = 0

        df = df.sort_values(
            by=["transaction_id", "non_null_count", "join_status_mobile", "topaz_risk_score"],
            ascending=[True, False, False, False]
        )

        df = df.drop_duplicates(subset=["transaction_id"], keep="first").copy()
        df = df.drop(columns=["non_null_count"], errors="ignore")
        return df

    def select_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
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
            "topaz_score_filled",
            "topaz_transacao_rejeitada",
            "topaz_transacao_habilitada",
            "is_agendamento_recorrente",

            "device_missing_flag",
            "ip_missing_flag",
            "app_version_missing_flag",
            "auth_method_missing_flag",
            "topaz_missing_flag",
            "host_time_missing_flag",

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

        return df[final_cols].copy()

    def fit(self, df: pd.DataFrame):
        df = df.copy()

        # separar colunas
        feature_cols = [c for c in df.columns if c not in self.id_columns_]

        # identificar tipos
        self.categorical_columns_ = [
            c for c in feature_cols
            if str(df[c].dtype) == "object" or str(df[c].dtype).startswith("category")
        ]

        self.numeric_columns_ = [
            c for c in feature_cols
            if c not in self.categorical_columns_
        ]

        # imputação numérica
        self.numeric_imputer_ = {}
        for c in self.numeric_columns_:
            median_value = pd.to_numeric(df[c], errors="coerce").median(skipna=True)
            if pd.isna(median_value):
                median_value = 0.0
            self.numeric_imputer_[c] = median_value

        # níveis categóricos vistos no treino
        self.categorical_levels_ = {}
        for c in self.categorical_columns_:
            vals = df[c].fillna(self.categorical_fill_value_).astype(str).unique().tolist()
            vals = sorted(vals)
            self.categorical_levels_[c] = vals

        # construir colunas finais do modelo com one-hot consistente
        model_cols = []

        for c in self.numeric_columns_:
            model_cols.append(c)

        for c in self.categorical_columns_:
            for level in self.categorical_levels_[c]:
                model_cols.append(f"{c}__{level}")

        self.model_columns_ = model_cols
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # garantir todas as colunas conhecidas
        for c in self.numeric_columns_:
            if c not in df.columns:
                df[c] = np.nan

        for c in self.categorical_columns_:
            if c not in df.columns:
                df[c] = self.categorical_fill_value_

        # imputação numérica
        for c in self.numeric_columns_:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(self.numeric_imputer_[c])

        # imputação categórica
        for c in self.categorical_columns_:
            df[c] = df[c].fillna(self.categorical_fill_value_).astype(str)

        # one-hot manual consistente
        out = pd.DataFrame(index=df.index)

        for c in self.numeric_columns_:
            out[c] = df[c]

        for c in self.categorical_columns_:
            known_levels = self.categorical_levels_[c]
            for level in known_levels:
                out[f"{c}__{level}"] = (df[c] == level).astype(int)

        # garantir ordem e colunas
        for c in self.model_columns_:
            if c not in out.columns:
                out[c] = 0

        out = out[self.model_columns_].copy()
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)


# =========================================================
# MAIN
# =========================================================
def main():
    print("Carregando arquivos...")
    pix_normal = pd.read_csv(PATH_PIX_NORMAL)
    pix_fraud = pd.read_csv(PATH_PIX_FRAUD)
    mobile = pd.read_csv(PATH_MOBILE)

    preprocessor = PixPreprocessor()

    print("Montando base consolidada...")
    df_raw = preprocessor.build_base_from_raw_files(pix_normal, pix_fraud, mobile)

    print("Criando features...")
    df_features = preprocessor.create_features(df_raw)

    print("Deduplicando transações...")
    df_features = preprocessor.deduplicate_transactions(df_features)

    print("Selecionando colunas finais de features...")
    df_features = preprocessor.select_feature_columns(df_features)

    print(f"Salvando base de features em: {OUTPUT_FEATURES}")
    df_features.to_csv(OUTPUT_FEATURES, index=False)

    print("Ajustando preprocessing para treino/inferência...")
    df_model_input = df_features.copy()

    # base para o modelo: remove ids e label na transformação
    feature_only_df = df_model_input.drop(
        columns=["transaction_id", "customer_id", "event_datetime", "source_dataset", "join_status_mobile", "is_fraud"],
        errors="ignore"
    )

    X_ready = preprocessor.fit_transform(feature_only_df)

    # recoloca alguns ids e label para inspeção e treino
    model_ready_df = pd.concat(
        [
            df_model_input[["transaction_id", "customer_id", "event_datetime", "source_dataset", "join_status_mobile", "is_fraud"]].reset_index(drop=True),
            X_ready.reset_index(drop=True)
        ],
        axis=1
    )

    print(f"Salvando base model-ready em: {OUTPUT_MODEL_READY}")
    model_ready_df.to_csv(OUTPUT_MODEL_READY, index=False)

    print(f"Salvando artefato de preprocessing em: {OUTPUT_PREPROCESSOR}")
    joblib.dump(preprocessor, OUTPUT_PREPROCESSOR)

    print("\nProcessamento concluído com sucesso.")
    print(f"Base features shape: {df_features.shape}")
    print(f"Base model-ready shape: {model_ready_df.shape}")
    print(f"Nº colunas finais do modelo: {len(preprocessor.model_columns_)}")


if __name__ == "__main__":
    main()
