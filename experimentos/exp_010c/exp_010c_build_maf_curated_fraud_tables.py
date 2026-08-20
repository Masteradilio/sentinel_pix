# -*- coding: utf-8 -*-
"""
EXP-010C — Build MAF Curated Fraud Tables

Objetivo:
  Criar tabelas definitivas derivadas da fonte textual MAF:

    1. hmo_ml.tb_pix_fraude_labels_maf_curated_v1
       - tabela curada de labels pós-evento;
       - deduplicada por transaction_id/E2E ID;
       - preserva status, confiança, tipo, direção e flags de conflito;
       - textos ficam apenas para auditoria.

    2. hmo_ml.tb_pix_fraudes_maf_hidratadas_v1
       - tabela final de fraudes hidratadas para o modelo atual;
       - usa apenas POSITIVE_FOR_CURRENT_MODEL;
       - hidrata com PIX + cliente;
       - opcionalmente hidrata mobile, se ENABLE_MOBILE=True;
       - calcula features rolling leakage-free;
       - gera saída compatível com preprocessing.py.

Observações:
  - Textos da MAF NÃO entram como features do modelo.
  - BRB_CREDITADO_RECEBEDOR fica segregado.
  - TRIANGULACAO fica segregada.
  - Casos com conflito de label/direção ficam fora do positivo forte.
  - O script é idempotente: sobrescreve as tabelas v1.

Execução:
  python exp_010c_build_maf_curated_fraud_tables.py

Saídas locais:
  /home/cdsw/Adilio/rebuild_pix/Artefatos/EXP-010C/
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel


# Compatibilidade PySpark antigo x NumPy novo.
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]


# ============================================================
# CONFIG
# ============================================================

SOURCE_TABLE = "landing_brb_oracle_maf.tb_infracao_pix"

PIX_EXTRATO_TABLE = "landing_brb_oracle_blk.tb_extrato_pix"
PIX_REGISTRO_TABLE = "landing_brb_oracle_blk.tb_registro_pix"
MBK_TABLE = "landing_brb_oracle_mbk.aut"

LABEL_TABLE = "hmo_ml.tb_pix_fraude_labels_maf_curated_v1"
HYDRATED_TABLE = "hmo_ml.tb_pix_fraudes_maf_hidratadas_v1"

BRB_ISPB = "00000208"

OUTPUT_BASE_DIR = "/home/cdsw/Adilio/rebuild_pix/Artefatos"
EXP_NAME = "EXP-010C"
OUTPUT_DIR = Path(OUTPUT_BASE_DIR) / EXP_NAME

CSV_LIMIT = 1000
LOOKUP_DAYS_BACK = 1460
NINETY_DAYS_SECONDS = 90 * 86400

# Mobile pode ficar pesado. Deixe False nesta primeira execução definitiva.
ENABLE_MOBILE = False

# Gera CSV local completo da tabela hidratada. Deve ficar perto de dezenas de milhares de linhas.
EXPORT_FULL_HYDRATED_CSV = True

# Se True, derruba e recria as tabelas destino.
OVERWRITE_TABLES = True


# ============================================================
# SPARK
# ============================================================

def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("EXP-010C - Build MAF Curated Fraud Tables")
        .config("spark.driver.memory", "8g")
        .config("spark.driver.maxResultSize", "3g")
        .config("spark.executor.memory", "10g")
        .config("spark.executor.cores", "2")
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.dynamicAllocation.minExecutors", "2")
        .config("spark.dynamicAllocation.initialExecutors", "2")
        .config("spark.dynamicAllocation.maxExecutors", "10")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.shuffle.partitions", "96")
        .config("spark.default.parallelism", "96")
        .config("spark.sql.autoBroadcastJoinThreshold", "104857600")
        .config("spark.sql.broadcastTimeout", "1200")
        .config("spark.network.timeout", "1200s")
        .config("spark.executor.heartbeatInterval", "60s")
        # Necessário por causa de inconsistências físicas/schema na MAF.
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.hive.convertMetastoreParquet", "false")
        .config("spark.sql.parquet.mergeSchema", "false")
        .config("spark.yarn.executor.memoryOverhead", "2048")
        .enableHiveSupport()
        .getOrCreate()
    )


# ============================================================
# HELPERS
# ============================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(x) for x in obj]
    if isinstance(obj, tuple):
        return [safe_json(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe_json(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_completeness_score(df, cols):
    expr = F.lit(0)
    for c in cols:
        if c in df.columns:
            expr = expr + F.when(
                F.col(c).isNotNull()
                & (F.col(c).cast("string") != "")
                & (F.col(c).cast("string") != "Informação ausente"),
                F.lit(1),
            ).otherwise(F.lit(0))
    return expr


def safe_count(df, label: str) -> int:
    try:
        n = int(df.count())
        print(f"[COUNT] {label}: {n}")
        return n
    except Exception as exc:
        print(f"[WARN] Falha ao contar {label}: {exc}")
        return -1


def clean_ispb_col(colname: str):
    return F.lpad(
        F.regexp_replace(F.coalesce(F.col(colname).cast("string"), F.lit("")), r"[^0-9]", ""),
        8,
        "0",
    )


def normalize_text_expr(colname: str):
    expr = F.lower(F.coalesce(F.col(colname).cast("string"), F.lit("")))
    expr = F.translate(
        expr,
        "áàãâäéèêëíìîïóòõôöúùûüçñÁÀÃÂÄÉÈÊËÍÌÎÏÓÒÕÔÖÚÙÛÜÇÑ",
        "aaaaaeeeeiiiiooooouuuucnAAAAAEEEEIIIIOOOOOUUUUCN",
    )
    expr = F.regexp_replace(expr, r"[^a-zA-Z0-9 ]+", " ")
    expr = F.regexp_replace(expr, r"\s+", " ")
    return F.trim(expr)


def write_limited_csv(df, filename: str, limit: int = CSV_LIMIT, order_cols: list[str] | None = None) -> Path:
    path = OUTPUT_DIR / filename

    out = df

    if order_cols:
        valid = [c for c in order_cols if c in out.columns]
        if valid:
            out = out.orderBy(*[F.col(c).desc_nulls_last() for c in valid])

    out = out.limit(limit)

    for col_name, dtype in out.dtypes:
        dtype_l = str(dtype).lower()
        if dtype_l in {"date", "timestamp"} or dtype_l.startswith("timestamp"):
            out = out.withColumn(col_name, F.date_format(F.col(col_name), "yyyy-MM-dd HH:mm:ss"))
        elif dtype_l == "boolean":
            out = out.withColumn(
                col_name,
                F.when(F.col(col_name) == True, F.lit("true"))
                 .when(F.col(col_name) == False, F.lit("false"))
                 .otherwise(F.lit(None).cast("string")),
            )

    try:
        pdf = out.toPandas()
    except Exception as exc:
        msg = str(exc)
        if "np.bool" in msg or "datetime64" in msg or "numpy" in msg:
            out2 = out.select([F.col(c).cast("string").alias(c) for c in out.columns])
            pdf = out2.toPandas()
        else:
            raise

    pdf.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def export_full_csv(df, filename: str) -> Path:
    path = OUTPUT_DIR / filename

    out = df

    for col_name, dtype in out.dtypes:
        dtype_l = str(dtype).lower()
        if dtype_l in {"date", "timestamp"} or dtype_l.startswith("timestamp"):
            out = out.withColumn(col_name, F.date_format(F.col(col_name), "yyyy-MM-dd HH:mm:ss"))
        elif dtype_l == "boolean":
            out = out.withColumn(
                col_name,
                F.when(F.col(col_name) == True, F.lit("true"))
                 .when(F.col(col_name) == False, F.lit("false"))
                 .otherwise(F.lit(None).cast("string")),
            )

    pdf = out.toPandas()
    pdf.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_table(df, table_name: str) -> None:
    if OVERWRITE_TABLES:
        print(f"[TABLE] Drop se existir: {table_name}")
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    print(f"[TABLE] Salvando: {table_name}")
    df.write.mode("overwrite").format("parquet").saveAsTable(table_name)
    print(f"[TABLE] OK: {table_name}")


# ============================================================
# MAF LABEL CURATION
# ============================================================

def add_direction_and_standard_columns(df):
    return (
        df
        .withColumnRenamed("cd_identificador_fim_transacao", "transaction_id_raw")
        .withColumn("transaction_id", F.trim(F.col("transaction_id_raw").cast("string")))
        .withColumn("cd_ispb_creditado_norm", clean_ispb_col("cd_ispb_creditado"))
        .withColumn("cd_ispb_debitado_norm", clean_ispb_col("cd_ispb_debitado"))
        .withColumn(
            "bank_direction",
            F.when(
                (F.col("cd_ispb_debitado_norm") == BRB_ISPB)
                & (F.col("cd_ispb_creditado_norm") != BRB_ISPB),
                F.lit("BRB_DEBITADO_PAGADOR"),
            )
            .when(
                (F.col("cd_ispb_creditado_norm") == BRB_ISPB)
                & (F.col("cd_ispb_debitado_norm") != BRB_ISPB),
                F.lit("BRB_CREDITADO_RECEBEDOR"),
            )
            .when(
                (F.col("cd_ispb_creditado_norm") == BRB_ISPB)
                & (F.col("cd_ispb_debitado_norm") == BRB_ISPB),
                F.lit("BRB_BOTH"),
            )
            .otherwise(F.lit("OUTROS")),
        )
        .withColumn("transaction_id_valid_flag", F.col("transaction_id").rlike(r"^E[A-Za-z0-9]{20,}$"))
        .withColumn("dt_infracao_pix_ts", F.to_timestamp("dt_infracao_pix"))
        .withColumn("dt_ultima_alteracao_ts", F.to_timestamp("dt_ultima_alteracao"))
    )


def add_text_classification(df):
    df = (
        df
        .withColumn("tx_analise_detalhe_norm", normalize_text_expr("tx_analise_detalhe"))
        .withColumn("tx_analise_infracao_norm", normalize_text_expr("tx_analise_infracao"))
        .withColumn("all_text_norm", F.concat_ws(" ", F.col("tx_analise_detalhe_norm"), F.col("tx_analise_infracao_norm")))
    )

    positive_detail_pattern = (
        "denuncia acatada|denuncia procedente|procedente|"
        "fraude confirmada|confirmada fraude|transacao fraudulenta|"
        "conta em processo de encerramento|devolucao realizada|"
        "med procedente|relato procedente"
    )

    negative_detail_pattern = (
        "sem indicios|sem indicio|nao identificamos|nao identificado|"
        "nao havera devolucao|nao houve fraude|"
        "improcedente|denuncia nao acatada|nao acatada|rejeitada|"
        "alegado nao se enquadra|nao se enquadra"
    )

    triang_pattern = "triangulacao|triang"

    df = (
        df
        .withColumn("result_code_int", F.col("cd_resultado_analise_infracao").cast("int"))
        .withColumn("positive_by_result", F.col("result_code_int") == F.lit(1))
        .withColumn("negative_by_result", F.col("result_code_int") == F.lit(2))
        .withColumn("positive_by_text", F.col("tx_analise_detalhe_norm").rlike(positive_detail_pattern))
        .withColumn("negative_by_text", F.col("tx_analise_detalhe_norm").rlike(negative_detail_pattern))
        .withColumn("triangulation_flag", F.col("all_text_norm").rlike(triang_pattern))
        .withColumn(
            "label_conflict_flag",
            (F.col("positive_by_result") | F.col("positive_by_text"))
            & (F.col("negative_by_result") | F.col("negative_by_text")),
        )
        .withColumn(
            "label_status",
            F.when(F.col("label_conflict_flag"), F.lit("REVIEW_CONFLICT"))
             .when(F.col("positive_by_result") | F.col("positive_by_text"), F.lit("CONFIRMED_FRAUD_CANDIDATE"))
             .when(F.col("negative_by_result") | F.col("negative_by_text"), F.lit("NOT_FRAUD_OR_REJECTED"))
             .otherwise(F.lit("REVIEW_REQUIRED")),
        )
        .withColumn(
            "label_confidence",
            F.when(F.col("label_conflict_flag"), F.lit("CONFLICT"))
             .when(F.col("positive_by_result") & F.col("positive_by_text"), F.lit("STRONG"))
             .when(F.col("positive_by_result"), F.lit("MEDIUM_STRUCTURED"))
             .when(F.col("positive_by_text"), F.lit("MEDIUM_TEXT"))
             .when(F.col("negative_by_result") & F.col("negative_by_text"), F.lit("STRONG_NEGATIVE"))
             .when(F.col("negative_by_result") | F.col("negative_by_text"), F.lit("MEDIUM_NEGATIVE"))
             .otherwise(F.lit("LOW_REVIEW")),
        )
        .withColumn(
            "fraud_type",
            F.when(F.col("triangulation_flag"), F.lit("TRIANGULACAO"))
             .when(F.col("all_text_norm").rlike("investimento|cripto|criptomoeda|app|aplicativo|corretagem"), F.lit("GOLPE_INVESTIMENTO_APP"))
             .when(F.col("all_text_norm").rlike("whatsapp|telefone|ligacao|falso atendente|central|suporte"), F.lit("ENGENHARIA_SOCIAL"))
             .when(F.col("all_text_norm").rlike("sequestro|coacao|ameaca"), F.lit("COACAO_AMEACA"))
             .when(F.col("all_text_norm").rlike("compra|produto|mercadoria|venda|marketplace"), F.lit("GOLPE_COMPRA_VENDA"))
             .when(F.col("all_text_norm").rlike("boleto|fatura|pagamento falso"), F.lit("GOLPE_PAGAMENTO"))
             .otherwise(F.lit("NAO_CLASSIFICADO")),
        )
    )

    return df


def build_maf_source(spark: SparkSession):
    print("[1/8] Lendo e normalizando fonte MAF...")

    raw = spark.table(SOURCE_TABLE)

    # Normaliza nomes sem ler colunas problemáticas.
    for old in raw.columns:
        new = old.split(".")[-1].lower()
        if old != new:
            raw = raw.withColumnRenamed(old, new)

    expected_cols = [
        "sq_infracao_pix",
        "cd_resultado_analise_infracao",
        "cd_reporte_infracao",
        "dt_infracao_pix",
        "cd_identificador_fim_transacao",
        "tx_analise_infracao",
        "sq_transacao_pagamento",
        "cd_ispb_creditado",
        "cd_ispb_debitado",
        "cd_sequencial_gpi",
        "dt_ultima_alteracao",
        "cd_status_infracao_pix",
        "tx_analise_detalhe",
        "vl_relato_infracao",
        "cd_infracao_med_pix",
        "sq_transacao_bloqueio_pix",
        "ts_carga_landing",
    ]

    safe_select_exprs = []
    for c in expected_cols:
        if c in raw.columns:
            safe_select_exprs.append(F.col(c).cast("string").alias(c))
        else:
            safe_select_exprs.append(F.lit(None).cast("string").alias(c))

    base = raw.select(*safe_select_exprs)

    tx_not_empty = (
        F.col("cd_identificador_fim_transacao").isNotNull()
        & (F.length(F.trim(F.col("cd_identificador_fim_transacao").cast("string"))) > 0)
    )

    detalhe_not_empty = (
        F.col("tx_analise_detalhe").isNotNull()
        & (F.length(F.trim(F.col("tx_analise_detalhe").cast("string"))) > 0)
    )

    infracao_not_empty = (
        F.col("tx_analise_infracao").isNotNull()
        & (F.length(F.trim(F.col("tx_analise_infracao").cast("string"))) > 0)
    )

    df = (
        base
        .filter(tx_not_empty)
        .filter(detalhe_not_empty | infracao_not_empty)
    )

    df = add_direction_and_standard_columns(df)
    df = add_text_classification(df)

    return df


def build_curated_labels(source_df):
    print("[2/8] Curando e deduplicando labels MAF...")

    dup_stats = (
        source_df
        .groupBy("transaction_id")
        .agg(
            F.count("*").alias("n_raw_rows_per_transaction"),
            F.countDistinct("label_status").alias("n_distinct_label_status"),
            F.countDistinct("bank_direction").alias("n_distinct_bank_direction"),
            F.max(F.when(F.col("label_conflict_flag"), 1).otherwise(0)).alias("any_label_conflict_flag"),
            F.max("dt_ultima_alteracao_ts").alias("max_dt_ultima_alteracao_ts"),
        )
    )

    label_priority = (
        F.when(F.col("label_status") == "CONFIRMED_FRAUD_CANDIDATE", F.lit(100))
         .when(F.col("label_status") == "REVIEW_CONFLICT", F.lit(80))
         .when(F.col("label_status") == "REVIEW_REQUIRED", F.lit(50))
         .when(F.col("label_status") == "NOT_FRAUD_OR_REJECTED", F.lit(10))
         .otherwise(F.lit(0))
    )

    confidence_priority = (
        F.when(F.col("label_confidence") == "STRONG", F.lit(100))
         .when(F.col("label_confidence") == "MEDIUM_STRUCTURED", F.lit(80))
         .when(F.col("label_confidence") == "MEDIUM_TEXT", F.lit(70))
         .when(F.col("label_confidence") == "CONFLICT", F.lit(60))
         .otherwise(F.lit(10))
    )

    direction_priority = (
        F.when(F.col("bank_direction") == "BRB_DEBITADO_PAGADOR", F.lit(100))
         .when(F.col("bank_direction") == "BRB_CREDITADO_RECEBEDOR", F.lit(60))
         .when(F.col("bank_direction") == "BRB_BOTH", F.lit(40))
         .otherwise(F.lit(10))
    )

    w = (
        Window.partitionBy("transaction_id")
        .orderBy(
            label_priority.desc(),
            confidence_priority.desc(),
            direction_priority.desc(),
            F.col("transaction_id_valid_flag").desc(),
            F.col("dt_ultima_alteracao_ts").desc_nulls_last(),
            F.col("sq_infracao_pix").desc_nulls_last(),
        )
    )

    labels = (
        source_df
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
        .join(dup_stats, on="transaction_id", how="left")
        .withColumn(
            "duplicate_conflict_flag",
            (F.col("n_distinct_label_status") > 1)
            | (F.col("n_distinct_bank_direction") > 1)
            | (F.col("any_label_conflict_flag") > 0),
        )
        .withColumn(
            "model_scope_status",
            F.when(F.col("duplicate_conflict_flag"), F.lit("REVIEW_OR_CONFLICT"))
             .when(
                (F.col("bank_direction") == "BRB_DEBITADO_PAGADOR")
                & (F.col("label_status") == "CONFIRMED_FRAUD_CANDIDATE")
                & (F.col("transaction_id_valid_flag") == True)
                & (F.col("triangulation_flag") == False),
                F.lit("POSITIVE_FOR_CURRENT_MODEL"),
             )
             .when(
                (F.col("bank_direction") == "BRB_DEBITADO_PAGADOR")
                & (F.col("label_status") == "CONFIRMED_FRAUD_CANDIDATE")
                & (F.col("transaction_id_valid_flag") == True)
                & (F.col("triangulation_flag") == True),
                F.lit("TRIANGULATION_SEGREGATED"),
             )
             .when(
                (F.col("bank_direction") == "BRB_CREDITADO_RECEBEDOR")
                & (F.col("label_status") == "CONFIRMED_FRAUD_CANDIDATE"),
                F.lit("RECEIVER_SCOPE_SEGREGATED"),
             )
             .when(F.col("label_status") == "NOT_FRAUD_OR_REJECTED", F.lit("NOT_FRAUD_OR_REJECTED"))
             .otherwise(F.lit("REVIEW_REQUIRED")),
        )
        .withColumn("label_source", F.lit("MAF_TB_INFRACAO_PIX"))
        .withColumn("curated_at", F.current_timestamp())
    )

    label_cols = [
        "transaction_id",
        "transaction_id_raw",
        "transaction_id_valid_flag",
        "model_scope_status",
        "label_status",
        "label_confidence",
        "fraud_type",
        "bank_direction",
        "triangulation_flag",
        "duplicate_conflict_flag",
        "label_conflict_flag",
        "positive_by_result",
        "positive_by_text",
        "negative_by_result",
        "negative_by_text",
        "result_code_int",
        "cd_resultado_analise_infracao",
        "cd_status_infracao_pix",
        "cd_ispb_creditado_norm",
        "cd_ispb_debitado_norm",
        "dt_infracao_pix",
        "dt_infracao_pix_ts",
        "dt_ultima_alteracao",
        "dt_ultima_alteracao_ts",
        "max_dt_ultima_alteracao_ts",
        "n_raw_rows_per_transaction",
        "n_distinct_label_status",
        "n_distinct_bank_direction",
        "any_label_conflict_flag",
        "sq_infracao_pix",
        "cd_reporte_infracao",
        "sq_transacao_pagamento",
        "cd_sequencial_gpi",
        "vl_relato_infracao",
        "cd_infracao_med_pix",
        "sq_transacao_bloqueio_pix",
        "tx_analise_infracao",
        "tx_analise_detalhe",
        "label_source",
        "curated_at",
    ]

    return labels.select(*[c for c in label_cols if c in labels.columns])


# ============================================================
# HYDRATION — PIX, CLIENTE, MOBILE opcional
# ============================================================

def build_fraud_keys(labels_df):
    return (
        labels_df
        .filter(F.col("model_scope_status") == "POSITIVE_FOR_CURRENT_MODEL")
        .select(F.col("transaction_id").alias("cd_pix"))
        .dropDuplicates()
        .persist(StorageLevel.MEMORY_AND_DISK)
    )


def build_pix_base(spark: SparkSession, fraud_keys):
    print("[4/8] Hidratando PIX base...")

    df_pix_raw = (
        spark.table(PIX_EXTRATO_TABLE).alias("t")
        .join(
            spark.table(PIX_REGISTRO_TABLE).alias("r"),
            F.col("t.ds_id_pix") == F.col("r.ds_id_pix"),
            "inner",
        )
        .select(
            F.trim(F.col("t.ds_id_pix")).alias("cd_pix"),
            F.lpad(F.regexp_replace(F.col("t.nr_cpf_cnpj_origem").cast("string"), r"[^0-9]", ""), 14, "0").alias("cd_cpf_pagador"),
            F.lpad(F.regexp_replace(F.col("t.nr_cpf_cnpj_destino").cast("string"), r"[^0-9]", ""), 14, "0").alias("cd_cpf_cnpj_recebedor"),
            F.col("t.vl_pix").cast("double").alias("vl_pix"),
            F.col("t.dt_pix").cast("timestamp").alias("dt_pix"),
            F.col("t.dt_pix").cast("date").alias("data_pix"),
            F.coalesce(F.col("t.ds_chave_pix"), F.lit("Informação ausente")).alias("ds_chave_pix"),
            F.when(F.col("t.ds_chave_pix").isNull(), F.lit("Informação ausente"))
             .when(F.length(F.col("t.ds_chave_pix")) >= 32, F.lit("CHAVE ALEATORIA"))
             .when(F.col("t.ds_chave_pix").like("%@%"), F.lit("EMAIL"))
             .when(F.col("t.ds_chave_pix").rlike(r"^[0-9]+$") & (F.length(F.col("t.ds_chave_pix")) >= 11), F.lit("DOCUMENTO/TELEFONE"))
             .otherwise(F.lit("OUTROS")).alias("ds_tipo_chave"),
            F.col("r.st_processamento_retorno").alias("st_processamento_retorno"),
            F.lpad(F.regexp_replace(F.col("t.cd_ispb_origem").cast("string"), r"[^0-9]", ""), 8, "0").alias("cd_ispb_origem_norm"),
            F.lpad(F.regexp_replace(F.col("t.cd_ispb_destino").cast("string"), r"[^0-9]", ""), 8, "0").alias("cd_ispb_destino_norm"),
        )
        .filter(F.col("cd_pix").isNotNull())
        .filter(F.col("cd_ispb_origem_norm") == BRB_ISPB)
        .filter(F.col("st_processamento_retorno") != "RJCT")
    )

    if LOOKUP_DAYS_BACK is not None and int(LOOKUP_DAYS_BACK) > 0:
        df_pix_raw = df_pix_raw.filter(
            F.to_date("dt_pix") >= F.date_sub(F.current_date(), int(LOOKUP_DAYS_BACK))
        )

    df_pix = (
        df_pix_raw
        .join(F.broadcast(fraud_keys), on="cd_pix", how="inner")
        .drop("st_processamento_retorno", "cd_ispb_origem_norm", "cd_ispb_destino_norm")
        .dropDuplicates(["cd_pix"])
        .withColumn("is_fraud", F.lit(1))
    )

    return df_pix.persist(StorageLevel.MEMORY_AND_DISK)


def build_cliente(spark: SparkSession):
    print("[5/8] Carregando perfil de clientes...")

    df_cliente = spark.sql("""
        SELECT
            c.X0100_CLTCOD as cd_cliente,
            concat(
                LPAD(CAST(cast(c.X0100_CLTCGC as BIGINT) AS STRING), 12, '0'),
                LPAD(CAST(cast(c.X0100_CLTCGCDIG as BIGINT) AS STRING), 2, '0')
            ) AS cd_cpf_pagador,

            COALESCE(trim(segmento.ds_segmento), 'Informação ausente') as ds_segmento,

            COALESCE(
                cast(
                    datediff(
                        current_date(),
                        cast(
                            date_format(
                                concat(
                                    cast(substr(cast(cast(c.X0100_CLTDATNAS AS INT) as STRING),1,4) as STRING),
                                    '-',
                                    cast(substr(cast(cast(c.X0100_CLTDATNAS AS INT) as STRING),5,2) as STRING),
                                    '-',
                                    cast(substr(cast(cast(c.X0100_CLTDATNAS AS INT) as STRING),7,2) as STRING)
                                ),
                                'yyyy-MM-dd'
                            ) as date
                        )
                    ) / 365.25 as int
                ),
                0
            ) as nr_idade,

            c.X0100_CLTDATPCAD as dt_cadastro_raw,

            CASE
                WHEN pf.X1700_FISSEX = 1 THEN 'M'
                WHEN pf.X1700_FISSEX = 2 THEN 'F'
                ELSE 'Informação ausente'
            END as ds_sexo,

            CASE
                WHEN pf.X1700_FISESTCVL = 1 THEN 'SOLTEIRO'
                WHEN pf.X1700_FISESTCVL = 2 THEN 'CASADO'
                WHEN pf.X1700_FISESTCVL = 3 THEN 'VIUVO'
                WHEN pf.X1700_FISESTCVL = 4 THEN 'DIVORCIADO'
                WHEN pf.X1700_FISESTCVL = 5 THEN 'UNIAO_ESTAVEL'
                WHEN pf.X1700_FISESTCVL = 6 THEN 'SEPARADO'
                WHEN pf.X1700_FISESTCVL = 7 THEN 'OUTRO'
                ELSE 'Informação ausente'
            END as ds_estado_civil,

            COALESCE(cast(pf.X1700_FISVALRENDA as double), 0) as vl_renda_cliente,
            COALESCE(cast(pf.X1700_FISNUMDEP as int), 0) as qt_dependentes

        FROM landing_brb_db2_aox.aoxb01 c

        LEFT JOIN (
            SELECT
                X1700_CLTCOD,
                X1700_FISSEX,
                X1700_FISESTCVL,
                X1700_FISVALRENDA,
                X1700_FISNUMDEP,
                ROW_NUMBER() OVER (
                    PARTITION BY X1700_CLTCOD
                    ORDER BY X1700_HDRDATA DESC, X1700_HDRHORA DESC
                ) as rn
            FROM landing_brb_db2_aox.aoxb17
        ) pf
            ON pf.X1700_CLTCOD = c.X0100_CLTCOD
            AND pf.rn = 1

        LEFT JOIN (
            SELECT cd_segmento, ds_segmento
            FROM (
                SELECT
                    trim(a0100_segcodsgm) cd_segmento,
                    a0100_segdessgm ds_segmento,
                    RANK() OVER (
                        PARTITION BY trim(a0100_segcodsgm)
                        ORDER BY A0100_HDRDATA ASC, A0100_HDRHORA DESC
                    ) rank
                FROM landing_brb_db2_dna.dnab01
            ) rk
            WHERE rank = 1
        ) segmento
            ON trim(c.X0100_SGMCODSEG) = trim(segmento.cd_segmento)
    """)

    df_cliente = (
        df_cliente
        .withColumn(
            "dt_inicio_relacionamento",
            F.to_date(
                F.concat(
                    F.substring(F.col("dt_cadastro_raw").cast("string"), 1, 4),
                    F.lit("-"),
                    F.substring(F.col("dt_cadastro_raw").cast("string"), 5, 2),
                    F.lit("-"),
                    F.substring(F.col("dt_cadastro_raw").cast("string"), 7, 2),
                ),
                "yyyy-MM-dd",
            ),
        )
        .withColumn(
            "qt_tempo_relacionamento_mes",
            F.coalesce(
                F.floor(F.months_between(F.current_date(), F.col("dt_inicio_relacionamento"))),
                F.lit(0),
            ).cast("int"),
        )
        .drop("dt_cadastro_raw", "dt_inicio_relacionamento")
    )

    w = Window.partitionBy("cd_cpf_pagador").orderBy(F.col("cd_cliente").desc_nulls_last())

    df_cliente = (
        df_cliente
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn", "cd_cliente")
    )

    return df_cliente.persist(StorageLevel.MEMORY_AND_DISK)


def build_mobile(spark: SparkSession, fraud_keys):
    print("[6/8] Extraindo mobile MBK para chaves MAF...")

    cutoff_date = spark.sql(
        f"SELECT cast(date_sub(current_date(), {int(LOOKUP_DAYS_BACK)}) as string)"
    ).collect()[0][0]

    df_mobile = spark.sql(f"""
        SELECT
            trim(
                COALESCE(
                    REGEXP_EXTRACT(auttrn, '<BRB__IdFimAFimOriginalPix.*?>(.*?)</BRB__IdFimAFimOriginalPix>', 1),
                    REGEXP_EXTRACT(auttrn, '<FTN__IdFimAFimOriginalPix.*?>(.*?)</FTN__IdFimAFimOriginalPix>', 1),
                    REGEXP_EXTRACT(auttrn, '<idFimAFim.*?>(.*?)</idFimAFim>', 1)
                )
            ) AS end_to_end_id,
            autdathorini AS data_hora_inicio,
            autdatref AS data_referencia,
            autcodret AS cd_retorno,
            REGEXP_EXTRACT(auttrn, '<FTN__NomeDispositivo.*?>(.*?)</FTN__NomeDispositivo>', 1) AS device_name,
            REGEXP_EXTRACT(auttrn, '<BRB__UserAgentTopaz.*?>(.*?)</BRB__UserAgentTopaz>', 1) AS app_version,
            COALESCE(
                REGEXP_EXTRACT(auttrn, '<FTN__IpUsuario.*?>(.*?)</FTN__IpUsuario>', 1),
                REGEXP_EXTRACT(auttrn, '<ip>(.*?)</ip>', 1)
            ) AS ip_address,
            CAST(NULLIF(REGEXP_EXTRACT(auttrn, '<tempoRede.*?>(.*?)</tempoRede>', 1), '') AS INT) AS latencia_rede_ms,
            CAST(NULLIF(REGEXP_EXTRACT(auttrn, '<tempoAtendimento.*?>(.*?)</tempoAtendimento>', 1), '') AS INT) AS tempo_interacao_ms,
            CAST(NULLIF(REGEXP_EXTRACT(auttrn, '<tempoAutorizacao.*?>(.*?)</tempoAutorizacao>', 1), '') AS INT) AS tempo_processamento_host_ms,
            REGEXP_EXTRACT(auttrn, '<BRB__AuthenticationMethodTopaz.*?>(.*?)</BRB__AuthenticationMethodTopaz>', 1) AS metodo_autenticacao,
            REGEXP_EXTRACT(auttrn, '<BRB__IdentificadorSessao.*?>(.*?)</BRB__IdentificadorSessao>', 1) AS session_id,
            CAST(NULLIF(COALESCE(
                REGEXP_EXTRACT(auttrn, '<BRB__ResultadoConsultaScoreTopaz[^>]*tipo="java.lang.Integer"[^>]*>(.*?)</BRB__ResultadoConsultaScoreTopaz>', 1),
                REGEXP_EXTRACT(auttrn, '<BRB__ResultadoConsultaScoreTopaz>(\\\\d+)</BRB__ResultadoConsultaScoreTopaz>', 1)
            ), '') AS INT) AS topaz_risk_score,
            CAST(NULLIF(REGEXP_EXTRACT(auttrn, '<BRB__TopazTransacaoRejeitada[^>]*>(.*?)</BRB__TopazTransacaoRejeitada>', 1), '') AS INT) AS topaz_transacao_rejeitada,
            REGEXP_EXTRACT(auttrn, '<BRB__IsAgendamentoRecorrenteForTopaz[^>]*>(.*?)</BRB__IsAgendamentoRecorrenteForTopaz>', 1) AS is_agendamento_recorrente
        FROM {MBK_TABLE}
        WHERE autdatref >= '{cutoff_date}'
          AND auttrn LIKE '%<transacao%'
    """)

    df_mobile = (
        df_mobile
        .filter(F.col("end_to_end_id").isNotNull())
        .filter(F.length(F.trim(F.col("end_to_end_id"))) > 0)
        .join(
            F.broadcast(fraud_keys.select(F.col("cd_pix").alias("_ref"))),
            F.col("end_to_end_id") == F.col("_ref"),
            "inner",
        )
        .drop("_ref")
    )

    mobile_score_cols = [
        "device_name", "app_version", "ip_address", "latencia_rede_ms",
        "tempo_interacao_ms", "tempo_processamento_host_ms", "metodo_autenticacao",
        "session_id", "cd_retorno", "topaz_risk_score", "topaz_transacao_rejeitada",
        "is_agendamento_recorrente",
    ]

    df_mobile = df_mobile.withColumn("mobile_score", build_completeness_score(df_mobile, mobile_score_cols))

    w = Window.partitionBy("end_to_end_id").orderBy(
        F.col("mobile_score").desc(),
        F.col("data_referencia").desc_nulls_last(),
        F.col("data_hora_inicio").desc_nulls_last(),
    )

    df_mobile = (
        df_mobile
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn", "mobile_score")
    )

    return df_mobile.persist(StorageLevel.MEMORY_AND_DISK)


def add_mobile_placeholders(df):
    placeholders = {
        "end_to_end_id": "string",
        "data_hora_inicio": "string",
        "data_referencia": "string",
        "cd_retorno": "string",
        "device_name": "string",
        "app_version": "string",
        "ip_address": "string",
        "latencia_rede_ms": "int",
        "tempo_interacao_ms": "int",
        "tempo_processamento_host_ms": "int",
        "metodo_autenticacao": "string",
        "session_id": "string",
        "topaz_risk_score": "int",
        "topaz_transacao_rejeitada": "int",
        "is_agendamento_recorrente": "string",
    }

    out = df
    for col_name, dtype in placeholders.items():
        if col_name not in out.columns:
            out = out.withColumn(col_name, F.lit(None).cast(dtype))

    return out


def build_enriched_base(df_pix, df_cliente, df_mobile=None):
    print("[7/8] Montando base enriquecida...")

    if df_mobile is not None:
        df_base = (
            df_pix
            .join(df_mobile, df_pix.cd_pix == df_mobile.end_to_end_id, "left")
            .join(F.broadcast(df_cliente), on="cd_cpf_pagador", how="left")
        )
    else:
        df_base = df_pix.join(F.broadcast(df_cliente), on="cd_cpf_pagador", how="left")
        df_base = add_mobile_placeholders(df_base)

    base_score_cols = [
        "device_name", "app_version", "ip_address", "latencia_rede_ms",
        "tempo_interacao_ms", "tempo_processamento_host_ms", "metodo_autenticacao",
        "session_id", "cd_retorno", "topaz_risk_score", "topaz_transacao_rejeitada",
        "is_agendamento_recorrente",
        "nr_idade", "qt_tempo_relacionamento_mes", "ds_segmento",
        "ds_sexo", "ds_estado_civil",
    ]

    df_base = df_base.withColumn("base_score", build_completeness_score(df_base, base_score_cols))

    w = Window.partitionBy("cd_pix").orderBy(
        F.col("base_score").desc(),
        F.col("dt_pix").desc_nulls_last(),
    )

    df_base = (
        df_base
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn", "base_score")
    )

    return df_base.persist(StorageLevel.MEMORY_AND_DISK)


# ============================================================
# ROLLING FEATURES
# ============================================================

def add_rolling_features(df_base):
    print("[8/8] Calculando rolling features leakage-free...")

    df_base = df_base.withColumn("dt_pix_long", F.col("dt_pix").cast("long"))

    w_rolling_90d = (
        Window.partitionBy("cd_cpf_pagador")
        .orderBy("dt_pix_long")
        .rangeBetween(-NINETY_DAYS_SECONDS, -1)
    )

    df_features = (
        df_base
        .withColumn("qt_total_pix_trimestre", F.count("cd_pix").over(w_rolling_90d))
        .withColumn("vl_mediana_pix_trimestre", F.percentile_approx("vl_pix", 0.5).over(w_rolling_90d))
        .withColumn("vl_desvio_padrao_pix_trimestre", F.stddev("vl_pix").over(w_rolling_90d))
        .withColumn("qt_aparelhos_distintos_trimestre", F.size(F.collect_set("device_name").over(w_rolling_90d)))
        .withColumn("vl_latencia_rede_media_trimestre", F.avg("latencia_rede_ms").over(w_rolling_90d))
        .withColumn("vl_tempo_interacao_medio_trimestre", F.avg("tempo_interacao_ms").over(w_rolling_90d))
    )

    df_daily_counts = (
        df_base
        .groupBy("cd_cpf_pagador", "data_pix")
        .agg(F.count("cd_pix").alias("daily_count"))
    )

    df_daily_max_per_tx = (
        df_features.select("cd_pix", "cd_cpf_pagador", "data_pix", "dt_pix")
        .withColumnRenamed("data_pix", "data_pix_tx")
        .join(
            df_daily_counts.withColumnRenamed("data_pix", "data_pix_daily"),
            on="cd_cpf_pagador",
            how="inner",
        )
        .filter(
            (F.col("data_pix_daily") < F.col("data_pix_tx"))
            & (F.col("data_pix_daily") >= F.date_sub(F.col("data_pix_tx"), 90))
        )
        .groupBy("cd_pix")
        .agg(F.max("daily_count").alias("qt_pix_dia_maximo_trimestre"))
    )

    df_features = (
        df_features
        .join(df_daily_max_per_tx, on="cd_pix", how="left")
        .withColumn("qt_pix_dia_maximo_trimestre", F.coalesce(F.col("qt_pix_dia_maximo_trimestre"), F.lit(0)))
    )

    w_user_order = Window.partitionBy("cd_cpf_pagador").orderBy("dt_pix")
    w_receiver = Window.partitionBy("cd_cpf_pagador", "cd_cpf_cnpj_recebedor").orderBy("dt_pix")
    w_receiver_count = (
        Window.partitionBy("cd_cpf_pagador", "cd_cpf_cnpj_recebedor")
        .orderBy(F.col("dt_pix").cast("long"))
        .rangeBetween(-NINETY_DAYS_SECONDS, 0)
    )

    df_features = (
        df_features
        .withColumn("dt_transacao_anterior", F.lag("dt_pix").over(w_user_order))
        .withColumn("delta_pix_segundos", F.col("dt_pix").cast("long") - F.col("dt_transacao_anterior").cast("long"))
        .withColumn("qt_intervalo_transacao_minuto", F.coalesce(F.round(F.col("delta_pix_segundos") / 60, 4), F.lit(0.0)))
        .withColumn("tp_primeiro_envio_recebedor_trimestre", F.when(F.row_number().over(w_receiver) == 1, 1).otherwise(0))
        .withColumn("qt_envio_recebedor_trimestre", F.count("cd_pix").over(w_receiver_count))
        .drop("dt_transacao_anterior", "delta_pix_segundos")
    )

    w_rolling_intervalo = (
        Window.partitionBy("cd_cpf_pagador")
        .orderBy("dt_pix_long")
        .rangeBetween(-NINETY_DAYS_SECONDS, -1)
    )

    df_features = (
        df_features
        .withColumn(
            "qt_intervalo_mediana_trimestre",
            F.coalesce(F.percentile_approx("qt_intervalo_transacao_minuto", 0.5).over(w_rolling_intervalo), F.lit(0.0)),
        )
        .withColumn(
            "qt_intervalo_desvio_padrao_trimestre",
            F.coalesce(F.stddev("qt_intervalo_transacao_minuto").over(w_rolling_intervalo), F.lit(0.0)),
        )
    )

    df_features = (
        df_features
        .withColumn("qt_aparelhos_distintos_trimestre", F.coalesce(F.col("qt_aparelhos_distintos_trimestre"), F.lit(0)))
        .withColumn("qt_pix_dia_maximo_trimestre", F.coalesce(F.col("qt_pix_dia_maximo_trimestre"), F.lit(0)))
        .withColumn("vl_mediana_pix_trimestre", F.coalesce(F.col("vl_mediana_pix_trimestre"), F.lit(0.0)))
        .withColumn("vl_desvio_padrao_pix_trimestre", F.coalesce(F.col("vl_desvio_padrao_pix_trimestre"), F.lit(0.0)))
        .withColumn("vl_latencia_rede_media_trimestre", F.coalesce(F.col("vl_latencia_rede_media_trimestre"), F.lit(0.0)))
        .withColumn("vl_tempo_interacao_medio_trimestre", F.coalesce(F.col("vl_tempo_interacao_medio_trimestre"), F.lit(0.0)))
        .withColumn("qt_total_pix_trimestre", F.coalesce(F.col("qt_total_pix_trimestre"), F.lit(0)))
    )

    return df_features


def select_final_hydrated(df_features, labels_df):
    label_meta = (
        labels_df
        .select(
            F.col("transaction_id").alias("cd_pix"),
            "model_scope_status",
            "label_status",
            "label_confidence",
            "fraud_type",
            "bank_direction",
            "triangulation_flag",
            "duplicate_conflict_flag",
        )
    )

    df = df_features.join(label_meta, on="cd_pix", how="left")

    final_columns = [
        "cd_pix", "dt_pix", "cd_cpf_pagador", "cd_cpf_cnpj_recebedor",
        "ds_chave_pix", "ds_tipo_chave", "vl_pix",
        "qt_total_pix_trimestre", "vl_mediana_pix_trimestre",
        "vl_desvio_padrao_pix_trimestre", "qt_intervalo_transacao_minuto",
        "qt_intervalo_mediana_trimestre", "qt_intervalo_desvio_padrao_trimestre",
        "qt_pix_dia_maximo_trimestre",
        "device_name", "app_version", "ip_address",
        "latencia_rede_ms", "vl_latencia_rede_media_trimestre",
        "tempo_interacao_ms", "vl_tempo_interacao_medio_trimestre",
        "tempo_processamento_host_ms",
        "metodo_autenticacao", "session_id", "cd_retorno",
        "topaz_risk_score", "topaz_transacao_rejeitada",
        "is_agendamento_recorrente",
        "qt_aparelhos_distintos_trimestre",
        "nr_idade", "qt_tempo_relacionamento_mes",
        "ds_sexo", "ds_estado_civil", "ds_segmento",
        "vl_renda_cliente", "qt_dependentes",
        "tp_primeiro_envio_recebedor_trimestre",
        "qt_envio_recebedor_trimestre",
        "is_fraud",
        "model_scope_status",
        "label_status",
        "label_confidence",
        "fraud_type",
        "bank_direction",
        "triangulation_flag",
        "duplicate_conflict_flag",
    ]

    for c in final_columns:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast("string"))

    df_final = (
        df
        .select(*final_columns)
        .dropDuplicates(["cd_pix"])
        .withColumn("dt_carga", F.current_date())
        .withColumn("source_dataset", F.lit("fraud_maf"))
        .withColumn("source_label_table", F.lit(LABEL_TABLE))
    )

    return df_final


# ============================================================
# REPORTS
# ============================================================

def write_reports(labels_df, hydrated_df, counts: dict[str, Any]) -> None:
    print("[REPORT] Gerando artefatos locais...")

    write_limited_csv(
        labels_df.groupBy("model_scope_status", "label_status", "bank_direction", "fraud_type").agg(
            F.count("*").alias("n_rows"),
            F.countDistinct("transaction_id").alias("n_transactions"),
        ).orderBy(F.col("n_rows").desc()),
        "01_label_scope_distribution.csv",
    )

    write_limited_csv(
        labels_df.filter(F.col("model_scope_status") == "POSITIVE_FOR_CURRENT_MODEL"),
        "02_positive_labels_sample.csv",
        order_cols=["dt_ultima_alteracao_ts"],
    )

    write_limited_csv(
        labels_df.filter(F.col("model_scope_status").isin("REVIEW_OR_CONFLICT", "REVIEW_REQUIRED")),
        "03_review_or_conflict_labels_sample.csv",
        order_cols=["dt_ultima_alteracao_ts"],
    )

    write_limited_csv(
        labels_df.filter(F.col("model_scope_status") == "TRIANGULATION_SEGREGATED"),
        "04_triangulation_labels_sample.csv",
        order_cols=["dt_ultima_alteracao_ts"],
    )

    write_limited_csv(
        hydrated_df,
        "05_hydrated_frauds_sample.csv",
        order_cols=["dt_pix"],
    )

    coverage_cols = [
        "ds_sexo", "ds_estado_civil", "ds_segmento",
        "vl_renda_cliente", "qt_dependentes",
        "qt_tempo_relacionamento_mes",
        "tp_primeiro_envio_recebedor_trimestre",
        "qt_envio_recebedor_trimestre",
        "device_name", "app_version", "ip_address",
        "tempo_interacao_ms", "metodo_autenticacao",
        "topaz_risk_score", "topaz_transacao_rejeitada",
    ]

    rows = []
    total = hydrated_df.count()

    for c in coverage_cols:
        if c in hydrated_df.columns:
            val = hydrated_df.agg(
                F.sum(
                    F.when(
                        F.col(c).isNotNull()
                        & (F.col(c).cast("string") != "")
                        & (F.col(c).cast("string") != "Informação ausente")
                        & (F.col(c).cast("string") != "0"),
                        1,
                    ).otherwise(0)
                ).alias("not_null")
            ).collect()[0]["not_null"]

            rows.append({
                "column": c,
                "not_null_or_informative": int(val or 0),
                "total": int(total),
                "coverage_pct": round((float(val or 0) / max(float(total), 1.0)) * 100, 4),
            })

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "06_hydrated_coverage.csv", index=False, encoding="utf-8-sig")

    if EXPORT_FULL_HYDRATED_CSV:
        export_full_csv(hydrated_df, "dados_pix_fraudes_maf_hidratadas_v1.csv")

    summary = {
        "generated_at": now_iso(),
        "experiment": EXP_NAME,
        "source_table": SOURCE_TABLE,
        "label_table": LABEL_TABLE,
        "hydrated_table": HYDRATED_TABLE,
        "enable_mobile": ENABLE_MOBILE,
        "lookup_days_back": LOOKUP_DAYS_BACK,
        "counts": counts,
        "artifacts_dir": str(OUTPUT_DIR),
        "artifacts": [
            "01_label_scope_distribution.csv",
            "02_positive_labels_sample.csv",
            "03_review_or_conflict_labels_sample.csv",
            "04_triangulation_labels_sample.csv",
            "05_hydrated_frauds_sample.csv",
            "06_hydrated_coverage.csv",
            "dados_pix_fraudes_maf_hidratadas_v1.csv" if EXPORT_FULL_HYDRATED_CSV else None,
        ],
        "notes": [
            "Textos da MAF foram usados apenas para curadoria de labels, nao como features.",
            "BRB_CREDITADO_RECEBEDOR ficou segregado.",
            "Triangulacao ficou segregada.",
            "Tabela hidratada usa apenas POSITIVE_FOR_CURRENT_MODEL.",
            "Mobile fica desabilitado por padrao para reduzir custo; campos mobile podem ficar nulos.",
        ],
    }

    write_json(OUTPUT_DIR / "00_run_summary.json", summary)

    recommendation = f"""# EXP-010C — Build MAF Curated Fraud Tables

Gerado em: `{now_iso()}`

## Resultado

- Label table: `{LABEL_TABLE}`
- Hydrated fraud table: `{HYDRATED_TABLE}`
- ENABLE_MOBILE: `{ENABLE_MOBILE}`
- Labels curados: `{counts.get("labels_curated")}`
- POSITIVE_FOR_CURRENT_MODEL: `{counts.get("positive_for_current_model")}`
- Hidratados finais: `{counts.get("hydrated_final")}`

## Decisão esperada

Se a tabela hidratada tiver volume coerente e cobertura suficiente de cliente/relacionamento, o próximo passo é copiar/exportar `dados_pix_fraudes_maf_hidratadas_v1.csv` para o ambiente local do projeto e rodar o `preprocessing.py` junto com a base de normais.

## Observações

- Textos pós-evento permanecem fora das features.
- Casos de triangulação permanecem segregados.
- Casos BRB_CREDITADO_RECEBEDOR permanecem segregados.
- Casos com conflito de label/direção não entram na tabela hidratada final.
"""

    (OUTPUT_DIR / "07_recommendation.md").write_text(recommendation, encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    global spark

    t0 = time.time()
    ensure_output_dir()

    spark = create_spark_session()

    print("=" * 80)
    print("EXP-010C — Build MAF Curated Fraud Tables")
    print("=" * 80)
    print(f"SOURCE_TABLE: {SOURCE_TABLE}")
    print(f"LABEL_TABLE: {LABEL_TABLE}")
    print(f"HYDRATED_TABLE: {HYDRATED_TABLE}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"ENABLE_MOBILE: {ENABLE_MOBILE}")
    print(f"LOOKUP_DAYS_BACK: {LOOKUP_DAYS_BACK}")
    print("=" * 80)

    source_df = None
    labels_df = None
    fraud_keys = None
    df_pix = None
    df_cliente = None
    df_mobile = None
    df_base = None
    df_features = None
    df_final = None

    try:
        source_df = build_maf_source(spark).persist(StorageLevel.MEMORY_AND_DISK)
        source_count = safe_count(source_df, "source_df")

        labels_df = build_curated_labels(source_df).persist(StorageLevel.MEMORY_AND_DISK)
        labels_count = safe_count(labels_df, "labels_curated")

        write_limited_csv(
            labels_df.groupBy("model_scope_status").agg(
                F.count("*").alias("n_rows"),
                F.countDistinct("transaction_id").alias("n_transactions"),
            ).orderBy(F.col("n_rows").desc()),
            "00a_model_scope_status_distribution.csv",
        )

        positive_count = safe_count(
            labels_df.filter(F.col("model_scope_status") == "POSITIVE_FOR_CURRENT_MODEL"),
            "positive_for_current_model",
        )

        triang_count = safe_count(
            labels_df.filter(F.col("model_scope_status") == "TRIANGULATION_SEGREGATED"),
            "triangulation_segregated",
        )

        receiver_count = safe_count(
            labels_df.filter(F.col("model_scope_status") == "RECEIVER_SCOPE_SEGREGATED"),
            "receiver_scope_segregated",
        )

        review_count = safe_count(
            labels_df.filter(F.col("model_scope_status").isin("REVIEW_OR_CONFLICT", "REVIEW_REQUIRED")),
            "review_or_conflict",
        )

        save_table(labels_df, LABEL_TABLE)

        fraud_keys = build_fraud_keys(labels_df)
        fraud_key_count = safe_count(fraud_keys, "fraud_keys_for_hydration")

        df_pix = build_pix_base(spark, fraud_keys)
        pix_count = safe_count(df_pix, "pix_hydrated_base")

        df_cliente = build_cliente(spark)

        if ENABLE_MOBILE:
            df_mobile = build_mobile(spark, fraud_keys)
            mobile_count = safe_count(df_mobile, "mobile_dedup")
        else:
            print("[INFO] ENABLE_MOBILE=False. Campos mobile serão preenchidos como nulos.")
            df_mobile = None
            mobile_count = None

        df_base = build_enriched_base(df_pix, df_cliente, df_mobile)
        base_count = safe_count(df_base, "base_enriched")

        df_features = add_rolling_features(df_base).persist(StorageLevel.MEMORY_AND_DISK)

        df_final = select_final_hydrated(df_features, labels_df).persist(StorageLevel.MEMORY_AND_DISK)
        final_count = safe_count(df_final, "hydrated_final")

        save_table(df_final, HYDRATED_TABLE)

        counts = {
            "source_rows": source_count,
            "labels_curated": labels_count,
            "positive_for_current_model": positive_count,
            "triangulation_segregated": triang_count,
            "receiver_scope_segregated": receiver_count,
            "review_or_conflict": review_count,
            "fraud_keys_for_hydration": fraud_key_count,
            "pix_hydrated_base": pix_count,
            "mobile_dedup": mobile_count,
            "base_enriched": base_count,
            "hydrated_final": final_count,
        }

        write_reports(labels_df, df_final, counts)

        elapsed = round((time.time() - t0) / 60, 2)

        print()
        print("=" * 80)
        print("[OK] EXP-010C concluído")
        print(f"[OK] Tempo total: {elapsed} min")
        print(f"[OK] Label table: {LABEL_TABLE}")
        print(f"[OK] Hydrated table: {HYDRATED_TABLE}")
        print(f"[OK] Artefatos: {OUTPUT_DIR}")
        print("=" * 80)

    finally:
        for obj_name, obj in [
            ("source_df", source_df),
            ("labels_df", labels_df),
            ("fraud_keys", fraud_keys),
            ("df_pix", df_pix),
            ("df_cliente", df_cliente),
            ("df_mobile", df_mobile),
            ("df_base", df_base),
            ("df_features", df_features),
            ("df_final", df_final),
        ]:
            try:
                if obj is not None:
                    obj.unpersist()
            except Exception as exc:
                print(f"[WARN] Falha ao unpersist {obj_name}: {exc}")


if __name__ == "__main__":
    main()