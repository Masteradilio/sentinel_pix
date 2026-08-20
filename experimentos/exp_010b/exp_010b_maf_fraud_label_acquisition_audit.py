# -*- coding: utf-8 -*-
"""
EXP-010B — MAF Fraud Label Acquisition Audit

Objetivo:
  Auditar a nova tabela textual do departamento de fraudes antes de criar
  tabela intermediaria definitiva ou tabela final de fraudes hidratadas.

Este script:
  - roda no CML com acesso ao Hive/Hue;
  - le a tabela landing_brb_oracle_maf.tb_infracao_pix;
  - classifica labels candidatos usando campos estruturados + texto de conclusao;
  - separa direcao BRB_DEBITADO_PAGADOR vs BRB_CREDITADO_RECEBEDOR;
  - mede cobertura de join com PIX, mobile e cliente;
  - gera artefatos CSV pequenos, sempre limitados a 1000 linhas;
  - nao altera modelo, scoring_config, DecisionEngine ou tabelas definitivas.

Saida:
  /cdsw/home/Adilio/rebuild_pix/Artefatos/EXP-010B/

Observacao:
  Este experimento e apenas de auditoria. O script definitivo de gestao
  de tabelas sera gerado depois da avaliacao dos artefatos.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

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
CLIENTE_TABLE = "landing_brb_db2_aox.aoxb01"

BRB_ISPB = "00000208"

# Caminho solicitado pelo usuario.
OUTPUT_BASE_DIR = "/home/cdsw/Adilio/rebuild_pix/Artefatos"
EXP_NAME = "EXP-010B"
OUTPUT_DIR = Path(OUTPUT_BASE_DIR) / EXP_NAME

# Para manter o experimento rapido.
CSV_LIMIT = 1000

SOURCE_DAYS_BACK = None
LOOKUP_DAYS_BACK = 1460
DAYS_BACK = SOURCE_DAYS_BACK

ENABLE_PIX_COVERAGE = True
ENABLE_MOBILE_COVERAGE = False
ENABLE_CLIENT_COVERAGE = False

# Para auditoria rápida, mede join PIX só no universo que pode alimentar o modelo atual.
PIX_COVERAGE_CONFIRMED_ONLY = True


# ============================================================
# SPARK
# ============================================================

def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("EXP-010B - MAF Fraud Label Acquisition Audit")
        .config("spark.driver.memory", "6g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.executor.memory", "8g")
        .config("spark.executor.cores", "2")
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.dynamicAllocation.minExecutors", "2")
        .config("spark.dynamicAllocation.initialExecutors", "2")
        .config("spark.dynamicAllocation.maxExecutors", "10")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.shuffle.partitions", "80")
        .config("spark.default.parallelism", "80")
        .config("spark.sql.autoBroadcastJoinThreshold", "104857600")
        .config("spark.sql.broadcastTimeout", "1200")
        .config("spark.network.timeout", "1200s")
        .config("spark.executor.heartbeatInterval", "60s")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.hive.convertMetastoreParquet", "false")
        .config("spark.sql.parquet.mergeSchema", "false")
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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_limited_csv(df, filename: str, limit: int = CSV_LIMIT, order_cols: list[str] | None = None) -> Path:
    """
    Escreve no maximo `limit` linhas em CSV local via pandas.

    Correções:
      - evita erro pandas datetime64 sem precisão;
      - evita erro PySpark antigo com np.bool removido do NumPy novo;
      - converte date/timestamp/boolean para string antes do toPandas();
      - lida com DataFrame vazio sem quebrar.
    """
    path = OUTPUT_DIR / filename

    out = df

    if order_cols:
        valid = [c for c in order_cols if c in out.columns]
        if valid:
            out = out.orderBy(*[F.col(c).desc_nulls_last() for c in valid])

    out = out.limit(limit)

    # Cast defensivo antes do toPandas.
    for col_name, dtype in out.dtypes:
        dtype_l = str(dtype).lower()

        if dtype_l in {"date", "timestamp"} or dtype_l.startswith("timestamp"):
            out = out.withColumn(
                col_name,
                F.date_format(F.col(col_name), "yyyy-MM-dd HH:mm:ss"),
            )

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
        # Fallback final: força tudo para string.
        msg = str(exc)
        if "np.bool" in msg or "datetime64" in msg or "numpy" in msg:
            out2 = out.select([F.col(c).cast("string").alias(c) for c in out.columns])
            pdf = out2.toPandas()
        else:
            raise

    pdf.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_pandas_csv(pdf: pd.DataFrame, filename: str, limit: int = CSV_LIMIT) -> Path:
    path = OUTPUT_DIR / filename
    pdf.head(limit).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def safe_count(df, label: str) -> int:
    try:
        n = df.count()
        print(f"[COUNT] {label}: {n}")
        return int(n)
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
    """
    Normaliza texto para classificacao simples:
      - lower
      - remove acentos comuns
      - troca pontuacao por espaco
      - colapsa espacos
    """
    expr = F.lower(F.coalesce(F.col(colname).cast("string"), F.lit("")))
    expr = F.translate(
        expr,
        "áàãâäéèêëíìîïóòõôöúùûüçñÁÀÃÂÄÉÈÊËÍÌÎÏÓÒÕÔÖÚÙÛÜÇÑ",
        "aaaaaeeeeiiiiooooouuuucnAAAAAEEEEIIIIOOOOOUUUUCN",
    )
    expr = F.regexp_replace(expr, r"[^a-zA-Z0-9 ]+", " ")
    expr = F.regexp_replace(expr, r"\s+", " ")
    return F.trim(expr)


def add_text_classification(df):
    """
    Cria colunas de classificacao textual conservadora.
    A conclusao operacional fica em tx_analise_detalhe.
    O relato do cliente ajuda em fraud_type, mas nao deve sozinho confirmar fraude.
    """
    df = (
        df
        .withColumn("tx_analise_detalhe_norm", normalize_text_expr("tx_analise_detalhe"))
        .withColumn("tx_analise_infracao_norm", normalize_text_expr("tx_analise_infracao"))
        .withColumn(
            "all_text_norm",
            F.concat_ws(
                " ",
                F.col("tx_analise_detalhe_norm"),
                F.col("tx_analise_infracao_norm"),
            ),
        )
    )

    positive_detail_pattern = (
        "denuncia acatada|denuncia procedente|procedente|"
        "fraude confirmada|confirmada fraude|transacao fraudulenta|"
        "conta em processo de encerramento|devolucao realizada|"
        "med procedente|relato procedente"
    )

    negative_detail_pattern = (
        "sem indicios|sem indicio|nao identificamos|nao identificado|"
        "nao havera devolucao|nao haverá devolucao|nao houve fraude|"
        "improcedente|denuncia nao acatada|nao acatada|rejeitada|"
        "alegado nao se enquadra|nao se enquadra"
    )

    triang_pattern = "triangulacao|triangulacao|triang"

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
    )

    fraud_type = (
        F.when(F.col("triangulation_flag"), F.lit("TRIANGULACAO"))
         .when(F.col("all_text_norm").rlike("investimento|cripto|criptomoeda|app|aplicativo|corretagem"), F.lit("GOLPE_INVESTIMENTO_APP"))
         .when(F.col("all_text_norm").rlike("whatsapp|telefone|ligacao|falso atendente|central|suporte"), F.lit("ENGENHARIA_SOCIAL"))
         .when(F.col("all_text_norm").rlike("sequestro|coacao|ameaça|ameaca"), F.lit("COACAO_AMEACA"))
         .when(F.col("all_text_norm").rlike("compra|produto|mercadoria|venda|marketplace"), F.lit("GOLPE_COMPRA_VENDA"))
         .when(F.col("all_text_norm").rlike("boleto|fatura|pagamento falso"), F.lit("GOLPE_PAGAMENTO"))
         .otherwise(F.lit("NAO_CLASSIFICADO"))
    )

    df = df.withColumn("fraud_type", fraud_type)

    return df


def add_direction_and_standard_columns(df):
    df = (
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

    return df


def deduplicate_labels(df):
    confidence_priority = (
        F.when(F.col("label_status") == "CONFIRMED_FRAUD_CANDIDATE", F.lit(100))
         .when(F.col("label_status") == "REVIEW_CONFLICT", F.lit(80))
         .when(F.col("label_status") == "REVIEW_REQUIRED", F.lit(50))
         .when(F.col("label_status") == "NOT_FRAUD_OR_REJECTED", F.lit(10))
         .otherwise(F.lit(0))
    )

    direction_priority = (
        F.when(F.col("bank_direction") == "BRB_DEBITADO_PAGADOR", F.lit(100))
         .when(F.col("bank_direction") == "BRB_CREDITADO_RECEBEDOR", F.lit(60))
         .otherwise(F.lit(10))
    )

    w = (
        Window.partitionBy("transaction_id")
        .orderBy(
            confidence_priority.desc(),
            direction_priority.desc(),
            F.col("dt_ultima_alteracao_ts").desc_nulls_last(),
            F.col("sq_infracao_pix").desc_nulls_last(),
        )
    )

    return (
        df
        .withColumn("_rn_label", F.row_number().over(w))
        .withColumn("_n_per_transaction", F.count("*").over(Window.partitionBy("transaction_id")))
        .filter(F.col("_rn_label") == 1)
        .drop("_rn_label")
    )


def build_source_df(spark: SparkSession):
    print("[1/8] Lendo fonte textual MAF...")

    raw = spark.table(SOURCE_TABLE)

    # Normaliza nomes: remove prefixo e joga para lowercase.
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

    for c in expected_cols:
        if c not in raw.columns:
            raw = raw.withColumn(c, F.lit(None).cast("string"))

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

    # Diagnóstico antes dos filtros.
    diag = (
        base
        .agg(
            F.count("*").alias("raw_rows"),
            F.sum(F.when(tx_not_empty, 1).otherwise(0)).alias("rows_with_transaction_id"),
            F.sum(F.when(detalhe_not_empty, 1).otherwise(0)).alias("rows_with_tx_analise_detalhe"),
            F.sum(F.when(infracao_not_empty, 1).otherwise(0)).alias("rows_with_tx_analise_infracao"),
            F.sum(F.when(tx_not_empty & (detalhe_not_empty | infracao_not_empty), 1).otherwise(0)).alias("rows_after_basic_filters"),
            F.date_format(F.min(F.to_timestamp("dt_ultima_alteracao")), "yyyy-MM-dd HH:mm:ss").alias("min_dt_ultima_alteracao"),
            F.date_format(F.max(F.to_timestamp("dt_ultima_alteracao")), "yyyy-MM-dd HH:mm:ss").alias("max_dt_ultima_alteracao"),
        )
    )

    write_limited_csv(diag, "00a_source_prefilter_diagnostics.csv", limit=1)

    df = (
        base
        .filter(tx_not_empty)
        .filter(detalhe_not_empty | infracao_not_empty)
    )

    # Filtro temporal opcional. Na auditoria inicial, DAYS_BACK=None.
    if DAYS_BACK is not None and int(DAYS_BACK) > 0:
        df_with_date = df.withColumn("_dt_ultima_alteracao_date", F.to_date(F.to_timestamp("dt_ultima_alteracao")))

        filtered = df_with_date.filter(
            F.col("_dt_ultima_alteracao_date") >= F.date_sub(F.current_date(), int(DAYS_BACK))
        )

        # Se o filtro temporal zerar a base, mantém sem filtro para não matar a auditoria.
        if filtered.limit(1).count() > 0:
            df = filtered.drop("_dt_ultima_alteracao_date")
        else:
            print("[WARN] Filtro temporal zerou a base. Mantendo fonte sem filtro temporal nesta auditoria.")

    df = add_direction_and_standard_columns(df)
    df = add_text_classification(df)

    return df


def build_pix_lookup(spark: SparkSession, labels_df):
    print("[5/8] Medindo cobertura com extrato PIX...")

    tx_ids = labels_df.select("transaction_id").dropDuplicates()

    pix = (
        spark.table(PIX_EXTRATO_TABLE)
        .select(
            F.trim(F.col("ds_id_pix")).alias("transaction_id"),
            F.col("vl_pix").cast("double").alias("pix_vl_pix"),
            F.col("dt_pix").cast("timestamp").alias("pix_dt_pix"),
            F.lpad(F.regexp_replace(F.col("nr_cpf_cnpj_origem").cast("string"), r"[^0-9]", ""), 14, "0").alias("pix_cd_cpf_pagador"),
            F.lpad(F.regexp_replace(F.col("nr_cpf_cnpj_destino").cast("string"), r"[^0-9]", ""), 14, "0").alias("pix_cd_cpf_cnpj_recebedor"),
            F.lpad(F.regexp_replace(F.col("cd_ispb_origem").cast("string"), r"[^0-9]", ""), 8, "0").alias("pix_cd_ispb_origem"),
            F.lpad(F.regexp_replace(F.col("cd_ispb_destino").cast("string"), r"[^0-9]", ""), 8, "0").alias("pix_cd_ispb_destino"),
        )
        .filter(F.col("transaction_id").isNotNull())
    )

    if LOOKUP_DAYS_BACK is not None and int(LOOKUP_DAYS_BACK) > 0:
        pix = pix.filter(
            F.to_date("pix_dt_pix") >= F.date_sub(F.current_date(), int(LOOKUP_DAYS_BACK))
        )

    pix = tx_ids.join(pix, on="transaction_id", how="left")

    return pix


def build_mobile_lookup(spark: SparkSession, labels_df):
    print("[6/8] Medindo cobertura com mobile MBK...")

    tx_ids = labels_df.select("transaction_id").dropDuplicates()

    mbk = spark.table(MBK_TABLE)

    mobile = (
        mbk
        .select(
            F.trim(
                F.coalesce(
                    F.regexp_extract("auttrn", r"<BRB__IdFimAFimOriginalPix.*?>(.*?)</BRB__IdFimAFimOriginalPix>", 1),
                    F.regexp_extract("auttrn", r"<FTN__IdFimAFimOriginalPix.*?>(.*?)</FTN__IdFimAFimOriginalPix>", 1),
                    F.regexp_extract("auttrn", r"<idFimAFim.*?>(.*?)</idFimAFim>", 1),
                )
            ).alias("transaction_id"),
            F.col("autdatref").alias("mobile_data_referencia"),
            F.col("autdathorini").alias("mobile_data_hora_inicio"),
            F.regexp_extract("auttrn", r"<FTN__NomeDispositivo.*?>(.*?)</FTN__NomeDispositivo>", 1).alias("mobile_device_name"),
            F.regexp_extract("auttrn", r"<BRB__UserAgentTopaz.*?>(.*?)</BRB__UserAgentTopaz>", 1).alias("mobile_app_version"),
            F.coalesce(
                F.regexp_extract("auttrn", r"<FTN__IpUsuario.*?>(.*?)</FTN__IpUsuario>", 1),
                F.regexp_extract("auttrn", r"<ip>(.*?)</ip>", 1),
            ).alias("mobile_ip_address"),
            F.regexp_extract("auttrn", r"<BRB__ResultadoConsultaScoreTopaz[^>]*tipo=\"java.lang.Integer\"[^>]*>(.*?)</BRB__ResultadoConsultaScoreTopaz>", 1).alias("mobile_topaz_score_raw"),
        )
        .filter(F.col("transaction_id").isNotNull())
        .filter(F.length(F.trim(F.col("transaction_id"))) > 0)
    )

    if LOOKUP_DAYS_BACK is not None and int(LOOKUP_DAYS_BACK) > 0:
        mobile = mobile.filter(
            F.col("autdatref") >= F.date_format(
                F.date_sub(F.current_date(), int(LOOKUP_DAYS_BACK)),
                "yyyy-MM-dd",
            )
        )

    # Dedup por transaction_id, priorizando linhas com mais campos.
    score_cols = [
        "mobile_device_name",
        "mobile_app_version",
        "mobile_ip_address",
        "mobile_topaz_score_raw",
    ]

    expr = F.lit(0)
    for c in score_cols:
        expr = expr + F.when(F.col(c).isNotNull() & (F.col(c) != ""), F.lit(1)).otherwise(F.lit(0))

    mobile = mobile.withColumn("_mobile_score", expr)

    w = (
        Window.partitionBy("transaction_id")
        .orderBy(
            F.col("_mobile_score").desc(),
            F.col("mobile_data_referencia").desc_nulls_last(),
            F.col("mobile_data_hora_inicio").desc_nulls_last(),
        )
    )

    mobile = (
        mobile
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn", "_mobile_score")
    )

    return tx_ids.join(mobile, on="transaction_id", how="left")


def build_client_lookup(spark: SparkSession, pix_join_df):
    print("[7/8] Medindo cobertura com cliente...")

    cpf_ids = (
        pix_join_df
        .select(F.col("pix_cd_cpf_pagador").alias("cd_cpf_pagador"))
        .filter(F.col("cd_cpf_pagador").isNotNull())
        .dropDuplicates()
    )

    cliente = (
        spark.table(CLIENTE_TABLE)
        .select(
            F.concat(
                F.lpad(F.col("X0100_CLTCGC").cast("bigint").cast("string"), 12, "0"),
                F.lpad(F.col("X0100_CLTCGCDIG").cast("bigint").cast("string"), 2, "0"),
            ).alias("cd_cpf_pagador"),
            F.col("X0100_CLTCOD").alias("cd_cliente"),
            F.col("X0100_CLTDATPCAD").alias("dt_cadastro_raw"),
        )
        .filter(F.col("cd_cpf_pagador").isNotNull())
    )

    w = Window.partitionBy("cd_cpf_pagador").orderBy(F.col("cd_cliente").desc_nulls_last())

    cliente = (
        cliente
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    return cpf_ids.join(cliente, on="cd_cpf_pagador", how="left")


def build_schema_profile(df):
    rows = []
    total = df.count()

    for name, dtype in df.dtypes:
        nulls = df.filter(F.col(name).isNull()).count()
        rows.append(
            {
                "column": name,
                "dtype": dtype,
                "n_rows": total,
                "n_nulls": nulls,
                "null_rate": round(nulls / max(total, 1), 6),
            }
        )

    return pd.DataFrame(rows)


def build_distribution(df, group_cols: list[str], count_col_name: str = "n_rows"):
    return (
        df
        .groupBy(*group_cols)
        .agg(
            F.count("*").alias(count_col_name),
            F.countDistinct("transaction_id").alias("n_distinct_transactions"),
        )
        .orderBy(F.col(count_col_name).desc())
    )


def build_join_coverage(labels_df, joined_df, join_flag_col: str, group_cols: list[str]):
    base = (
        joined_df
        .withColumn("_matched", F.when(F.col(join_flag_col).isNotNull(), F.lit(1)).otherwise(F.lit(0)))
        .groupBy(*group_cols)
        .agg(
            F.count("*").alias("n_labels"),
            F.sum("_matched").alias("n_matched"),
        )
        .withColumn("pct_matched", F.round(F.col("n_matched") / F.col("n_labels"), 6))
        .orderBy(F.col("n_labels").desc())
    )
    return base


def write_recommendation(
    total_raw: int,
    total_dedup: int,
    n_confirmed_brb_payer: int,
    n_review: int,
    n_conflict: int,
    pix_match_count: int,
    mobile_match_count: int,
    client_match_count: int | None,
):
    lines = [
        "# EXP-010B — MAF Fraud Label Acquisition Audit",
        "",
        f"Gerado em: `{now_iso()}`",
        "",
        "## Objetivo",
        "",
        "Auditar a nova tabela textual de fraudes antes de criar tabela intermediária ou tabela final de fraudes hidratadas.",
        "",
        "## Resultado executivo",
        "",
        f"- Linhas brutas auditadas: `{total_raw}`",
        f"- Transações deduplicadas por E2E ID: `{total_dedup}`",
        f"- Candidatos fortes para o modelo atual (`BRB_DEBITADO_PAGADOR` + `CONFIRMED_FRAUD_CANDIDATE`): `{n_confirmed_brb_payer}`",
        f"- Casos para revisão: `{n_review}`",
        f"- Conflitos de label: `{n_conflict}`",
        f"- Match PIX: `{pix_match_count}`",
        f"- Match mobile: `{mobile_match_count}`",
    ]

    if client_match_count is not None:
        lines.append(f"- Match cliente: `{client_match_count}`")
    else:
        lines.append("- Match cliente: `NOT_RUN`")

    lines.extend(
        [
            "",
            "## Decisão preliminar",
            "",
            "A nova fonte deve ser tratada como fonte de labels pós-evento. Os textos de relato e conclusão não devem entrar como features do modelo.",
            "",
            "Casos recomendados para próxima etapa:",
            "",
            "```text",
            "bank_direction = BRB_DEBITADO_PAGADOR",
            "label_status = CONFIRMED_FRAUD_CANDIDATE",
            "transaction_id_valid_flag = true",
            "triangulation_flag = false, salvo se decidirmos modelar triangulação separadamente",
            "```",
            "",
            "## Próximo passo",
            "",
            "Após avaliar os artefatos, gerar o script definitivo para:",
            "",
            "1. criar tabela intermediária curada de labels MAF;",
            "2. criar tabela final de fraudes hidratadas via join com PIX/mobile/cliente;",
            "3. produzir CSV compatível com o `preprocessing.py` e com o contrato v1.1.",
            "",
        ]
    )

    (OUTPUT_DIR / "12_recommendation.md").write_text("\n".join(lines), encoding="utf-8")


def write_next_script_spec():
    lines = [
        "# Próximo script recomendado",
        "",
        "## EXP-010C — Build MAF Curated Fraud Tables",
        "",
        "## Objetivo",
        "",
        "Criar as tabelas definitivas derivadas da fonte textual MAF.",
        "",
        "## Tabelas sugeridas",
        "",
        "```text",
        "hmo_ml.tb_pix_fraude_labels_maf_curated_v1",
        "hmo_ml.tb_pix_fraudes_maf_hidratadas_v1",
        "```",
        "",
        "## Regras sugeridas",
        "",
        "- A tabela de labels deve conter transaction_id, label_status, label_confidence, fraud_type e bank_direction.",
        "- A tabela final de fraudes deve conter apenas casos hidratados com features compatíveis com o pipeline atual.",
        "- Textos pós-evento devem ficar apenas para auditoria, não como feature.",
        "- Casos BRB_CREDITADO_RECEBEDOR devem ficar segregados até decisão de escopo.",
        "- Casos de triangulação devem ficar segregados ou tipados explicitamente.",
        "- CSVs finais para treino devem passar pelo `preprocessing.py`.",
        "",
    ]

    (OUTPUT_DIR / "13_next_script_spec.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    ensure_output_dir()
    pix_lookup = None
    mobile_lookup = None
    labels_dedup = None
    source_df = None

    spark = create_spark_session()

    print("=" * 80)
    print("EXP-010B — MAF Fraud Label Acquisition Audit")
    print("=" * 80)
    print(f"Fonte: {SOURCE_TABLE}")
    print(f"Saida: {OUTPUT_DIR}")
    print(f"SOURCE_DAYS_BACK: {SOURCE_DAYS_BACK}")
    print(f"LOOKUP_DAYS_BACK: {LOOKUP_DAYS_BACK}")
    print(f"ENABLE_PIX_COVERAGE: {ENABLE_PIX_COVERAGE}")
    print(f"ENABLE_MOBILE_COVERAGE: {ENABLE_MOBILE_COVERAGE}")
    print(f"ENABLE_CLIENT_COVERAGE: {ENABLE_CLIENT_COVERAGE}")
    print(f"PIX_COVERAGE_CONFIRMED_ONLY: {PIX_COVERAGE_CONFIRMED_ONLY}")
    print(f"CSV_LIMIT: {CSV_LIMIT}")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Fonte e classificacao
    # --------------------------------------------------------
    source_df = build_source_df(spark).persist()

    total_raw = safe_count(source_df, "source_df")

    # --------------------------------------------------------
    # 2. Schema profile
    # --------------------------------------------------------
    print("[2/8] Gerando perfil de schema...")
    schema_profile = build_schema_profile(source_df)
    write_pandas_csv(schema_profile, "01_schema_profile.csv", limit=CSV_LIMIT)

    # --------------------------------------------------------
    # 3. Dedup e distribuicoes
    # --------------------------------------------------------
    print("[3/8] Deduplicando por transaction_id e gerando distribuicoes...")
    labels_dedup = deduplicate_labels(source_df).persist()
    total_dedup = safe_count(labels_dedup, "labels_dedup")

    label_dist = build_distribution(labels_dedup, ["label_status", "label_confidence"])
    write_limited_csv(label_dist, "02_label_status_distribution.csv")

    direction_dist = build_distribution(labels_dedup, ["bank_direction", "label_status"])
    write_limited_csv(direction_dist, "03_bank_direction_distribution.csv")

    fraud_type_dist = build_distribution(labels_dedup, ["fraud_type", "label_status"])
    write_limited_csv(fraud_type_dist, "04_fraud_type_distribution.csv")

    dup_audit = (
        source_df
        .groupBy("transaction_id")
        .agg(
            F.count("*").alias("n_rows"),
            F.countDistinct("label_status").alias("n_label_status"),
            F.countDistinct("bank_direction").alias("n_bank_direction"),
            F.max("dt_ultima_alteracao_ts").alias("max_dt_ultima_alteracao"),
        )
        .filter(F.col("n_rows") > 1)
        .orderBy(F.col("n_rows").desc(), F.col("max_dt_ultima_alteracao").desc_nulls_last())
    )
    
    write_limited_csv(
    dup_audit,
    "11_e2e_duplicate_audit.csv",
    order_cols=["n_rows", "max_dt_ultima_alteracao"],
    )

    # --------------------------------------------------------
    # 4. Amostras de avaliacao
    # --------------------------------------------------------
    print("[4/8] Salvando amostras limitadas...")

    columns_for_samples = [
        "sq_infracao_pix",
        "transaction_id",
        "transaction_id_valid_flag",
        "dt_infracao_pix",
        "dt_ultima_alteracao",
        "cd_resultado_analise_infracao",
        "cd_status_infracao_pix",
        "cd_ispb_creditado_norm",
        "cd_ispb_debitado_norm",
        "bank_direction",
        "label_status",
        "label_confidence",
        "fraud_type",
        "triangulation_flag",
        "label_conflict_flag",
        "tx_analise_infracao",
        "tx_analise_detalhe",
    ]

    confirmed_brb_payer = (
        labels_dedup
        .filter(F.col("bank_direction") == "BRB_DEBITADO_PAGADOR")
        .filter(F.col("label_status") == "CONFIRMED_FRAUD_CANDIDATE")
        .filter(F.col("transaction_id_valid_flag") == True)
    )

    review_required = labels_dedup.filter(F.col("label_status").isin("REVIEW_REQUIRED", "REVIEW_CONFLICT"))

    rejected = labels_dedup.filter(F.col("label_status") == "NOT_FRAUD_OR_REJECTED")

    write_limited_csv(
        confirmed_brb_payer.select(*[c for c in columns_for_samples if c in confirmed_brb_payer.columns]),
        "05_confirmed_fraud_brb_payer_sample.csv",
        order_cols=["dt_ultima_alteracao_ts"],
    )
    write_limited_csv(
        review_required.select(*[c for c in columns_for_samples if c in review_required.columns]),
        "06_review_required_sample.csv",
        order_cols=["dt_ultima_alteracao_ts"],
    )
    write_limited_csv(
        rejected.select(*[c for c in columns_for_samples if c in rejected.columns]),
        "07_rejected_sample.csv",
        order_cols=["dt_ultima_alteracao_ts"],
    )

    n_confirmed_brb_payer = safe_count(confirmed_brb_payer, "confirmed_brb_payer")
    n_review = safe_count(review_required, "review_required")
    n_conflict = safe_count(labels_dedup.filter(F.col("label_status") == "REVIEW_CONFLICT"), "review_conflict")

    # --------------------------------------------------------
    # 5. PIX coverage
    # --------------------------------------------------------
    pix_match_count = None
    pix_lookup = None

    if ENABLE_PIX_COVERAGE:
        print("[5/8] Medindo cobertura com extrato PIX...")

        pix_scope = confirmed_brb_payer if PIX_COVERAGE_CONFIRMED_ONLY else labels_dedup
        pix_scope = pix_scope.persist()

        pix_lookup = build_pix_lookup(spark, pix_scope).persist()

        pix_join = (
            pix_scope
            .select(
                "transaction_id",
                "label_status",
                "label_confidence",
                "bank_direction",
                "fraud_type",
                "triangulation_flag",
            )
            .join(pix_lookup, on="transaction_id", how="left")
        )

        pix_coverage = build_join_coverage(
            labels_df=pix_scope,
            joined_df=pix_join,
            join_flag_col="pix_dt_pix",
            group_cols=["bank_direction", "label_status"],
        )

        write_limited_csv(pix_coverage, "08_join_coverage_pix.csv")

        pix_match_count = safe_count(
            pix_join.filter(F.col("pix_dt_pix").isNotNull()),
            "pix_match_count",
        )

        write_limited_csv(
            pix_join.filter(F.col("pix_dt_pix").isNotNull()),
            "08a_pix_matched_sample.csv",
            order_cols=["pix_dt_pix"],
        )

        write_limited_csv(
            pix_join.filter(F.col("pix_dt_pix").isNull()),
            "08b_pix_unmatched_sample.csv",
        )

        pix_scope.unpersist()

    else:
        pd.DataFrame(
            [{"status": "NOT_RUN_DISABLED", "message": "ENABLE_PIX_COVERAGE=False"}]
        ).to_csv(OUTPUT_DIR / "08_join_coverage_pix.csv", index=False, encoding="utf-8-sig")

        pix_join = (
            labels_dedup
            .select(
                "transaction_id",
                "label_status",
                "label_confidence",
                "bank_direction",
                "fraud_type",
                "triangulation_flag",
            )
            .join(pix_lookup, on="transaction_id", how="left")
        )

        pix_coverage = build_join_coverage(
            labels_df=labels_dedup,
            joined_df=pix_join,
            join_flag_col="pix_dt_pix",
            group_cols=["bank_direction", "label_status"],
        )

        write_limited_csv(pix_coverage, "08_join_coverage_pix.csv")

        pix_match_count = safe_count(pix_join.filter(F.col("pix_dt_pix").isNotNull()), "pix_match_count")

    # --------------------------------------------------------
    # 6. Mobile coverage
    # --------------------------------------------------------
    mobile_match_count = None

    if ENABLE_MOBILE_COVERAGE:
        mobile_lookup = build_mobile_lookup(spark, confirmed_brb_payer).persist()

        mobile_join = (
            confirmed_brb_payer
            .select(
                "transaction_id",
                "label_status",
                "bank_direction",
                "fraud_type",
            )
            .join(mobile_lookup, on="transaction_id", how="left")
        )

        mobile_coverage = build_join_coverage(
            labels_df=confirmed_brb_payer,
            joined_df=mobile_join,
            join_flag_col="mobile_data_referencia",
            group_cols=["bank_direction", "label_status"],
        )

        write_limited_csv(mobile_coverage, "09_join_coverage_mobile.csv")

        mobile_match_count = safe_count(
            mobile_join.filter(F.col("mobile_data_referencia").isNotNull()),
            "mobile_match_count",
        )

        mobile_lookup.unpersist()

    else:
        print("[6/8] Mobile coverage pulado no modo fast.")
        pd.DataFrame(
            [{
                "status": "NOT_RUN_FAST_MODE",
                "message": "Cobertura mobile foi pulada para evitar varredura massiva da MBK no EXP-010B."
            }]
        ).to_csv(OUTPUT_DIR / "09_join_coverage_mobile.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 7. Cliente coverage
    # --------------------------------------------------------
    client_match_count = None

    if ENABLE_CLIENT_COVERAGE and pix_lookup is not None:
        try:
            client_lookup = build_client_lookup(spark, pix_lookup).persist()

            client_join = (
                pix_join
                .select(
                    "transaction_id",
                    "label_status",
                    "bank_direction",
                    "fraud_type",
                    "pix_cd_cpf_pagador",
                )
                .join(
                    client_lookup,
                    pix_join.pix_cd_cpf_pagador == client_lookup.cd_cpf_pagador,
                    "left",
                )
                .drop(client_lookup.cd_cpf_pagador)
            )

            client_coverage = build_join_coverage(
                labels_df=confirmed_brb_payer,
                joined_df=client_join,
                join_flag_col="cd_cliente",
                group_cols=["bank_direction", "label_status"],
            )

            write_limited_csv(client_coverage, "10_join_coverage_cliente.csv")

            client_match_count = safe_count(
                client_join.filter(F.col("cd_cliente").isNotNull()),
                "client_match_count",
            )

            client_lookup.unpersist()

        except Exception as exc:
            print(f"[WARN] Falha na cobertura de cliente: {exc}")
            pd.DataFrame(
                [{"status": "NOT_RUN_ERROR", "error": str(exc)}]
            ).to_csv(OUTPUT_DIR / "10_join_coverage_cliente.csv", index=False, encoding="utf-8-sig")

    else:
        print("[7/8] Cliente coverage pulado no modo fast.")
        pd.DataFrame(
            [{
                "status": "NOT_RUN_FAST_MODE",
                "message": "Cobertura cliente foi pulada no EXP-010B fast. Será tratada no script final de hidratação."
            }]
        ).to_csv(OUTPUT_DIR / "10_join_coverage_cliente.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 8. Sumario e relatorios
    # --------------------------------------------------------
    print("[8/8] Gerando sumario e recomendacao...")

    summary = {
        "generated_at": now_iso(),
        "experiment": EXP_NAME,
        "source_table": SOURCE_TABLE,
        "output_dir": str(OUTPUT_DIR),
        "days_back": DAYS_BACK,
        "csv_limit": CSV_LIMIT,
        "counts": {
            "source_rows": total_raw,
            "dedup_transactions": total_dedup,
            "confirmed_brb_payer_candidates": n_confirmed_brb_payer,
            "review_required_or_conflict": n_review,
            "review_conflict": n_conflict,
            "pix_match_count": pix_match_count,
            "mobile_match_count": mobile_match_count,
            "client_match_count": client_match_count,
        },
        "label_rule_summary": {
            "confirmed_fraud_candidate": "cd_resultado_analise_infracao=1 ou texto conclusivo positivo em tx_analise_detalhe",
            "not_fraud_or_rejected": "cd_resultado_analise_infracao=2 ou texto conclusivo negativo em tx_analise_detalhe",
            "review_conflict": "sinais positivos e negativos simultaneos",
            "review_required": "sem conclusao clara",
        },
        "recommended_positive_scope_for_current_model": {
            "bank_direction": "BRB_DEBITADO_PAGADOR",
            "label_status": "CONFIRMED_FRAUD_CANDIDATE",
            "transaction_id_valid_flag": True,
            "notes": [
                "BRB_CREDITADO_RECEBEDOR deve ficar segregado ate decisao de escopo.",
                "Triangulacao deve ser segregada ou tipada explicitamente.",
                "Textos pos-evento nao devem entrar como features.",
            ],
        },
        "artifacts": [
            "01_schema_profile.csv",
            "02_label_status_distribution.csv",
            "03_bank_direction_distribution.csv",
            "04_fraud_type_distribution.csv",
            "05_confirmed_fraud_brb_payer_sample.csv",
            "06_review_required_sample.csv",
            "07_rejected_sample.csv",
            "08_join_coverage_pix.csv",
            "09_join_coverage_mobile.csv",
            "10_join_coverage_cliente.csv",
            "11_e2e_duplicate_audit.csv",
            "12_recommendation.md",
            "13_next_script_spec.md",
        ],
    }

    write_json(OUTPUT_DIR / "00_run_summary.json", summary)

    write_recommendation(
        total_raw=total_raw,
        total_dedup=total_dedup,
        n_confirmed_brb_payer=n_confirmed_brb_payer,
        n_review=n_review,
        n_conflict=n_conflict,
        pix_match_count=pix_match_count,
        mobile_match_count=mobile_match_count,
        client_match_count=client_match_count,
    )

    write_next_script_spec()

    for obj_name, obj in [
        ("labels_dedup", labels_dedup),
        ("source_df", source_df),
        ("pix_lookup", pix_lookup),
        ("mobile_lookup", mobile_lookup),
    ]:
        try:
            if obj is not None:
                obj.unpersist()
        except Exception as exc:
            print(f"[WARN] Falha ao unpersist {obj_name}: {exc}")

    elapsed = round((time.time() - t0) / 60, 2)

    print()
    print("=" * 80)
    print("[OK] EXP-010B concluido")
    print(f"[OK] Artefatos em: {OUTPUT_DIR}")
    print(f"[OK] Tempo total: {elapsed} min")
    print("=" * 80)

    print()
    print("Arquivos principais:")
    for name in summary["artifacts"]:
        print(f"  {OUTPUT_DIR / name}")
    print(f"  {OUTPUT_DIR / '00_run_summary.json'}")


if __name__ == "__main__":
    main()