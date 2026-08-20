# -*- coding: utf-8 -*-
"""
EXP-010F — Normal Sampling v2 / 180 dias

Objetivo:
  Criar uma amostra controlada de transações PIX normais na mesma janela
  de 180 dias usada nas fraudes MAF recentes.

Este experimento:
  - NÃO treina modelo;
  - NÃO calcula ainda todas as features rolling;
  - NÃO varre MBK bruta;
  - usa apenas PIX bruto + compact index MBK já criado, quando houver;
  - exclui fraudes MAF conhecidas;
  - salva uma tabela normal amostrada para o EXP-010G.

Entradas:
  landing_brb_oracle_blk.tb_extrato_pix
  landing_brb_oracle_blk.tb_registro_pix
  hmo_ml.tb_pix_fraudes_maf_hidratadas_v1
  hmo_ml.tb_pix_mbk_compact_180d_v1, se existir

Saídas Hive:
  hmo_ml.tb_pix_normais_sample_180d_v1
  hmo_ml.tb_pix_normais_sample_mbk_180d_v1

Saídas locais:
  /home/cdsw/Adilio/rebuild_pix/Artefatos/EXP-010F/

Observação:
  A etapa de feature engineering leakage-free fica para o EXP-010G.
"""

from __future__ import annotations

import json
import logging
import math
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel


if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]


# ============================================================
# CONFIG — ALTERE AQUI NO CML
# ============================================================

MAF_HYDRATED_TABLE = "hmo_ml.tb_pix_fraudes_maf_hidratadas_v1"
MBK_COMPACT_TABLE = "hmo_ml.tb_pix_mbk_compact_180d_v1"

OUTPUT_NORMAL_TABLE = "hmo_ml.tb_pix_normais_sample_180d_v1"
OUTPUT_NORMAL_MBK_TABLE = "hmo_ml.tb_pix_normais_sample_mbk_180d_v1"

OUTPUT_BASE_DIR = "/home/cdsw/Adilio/rebuild_pix/Artefatos"
EXP_NAME = "EXP-010F"
OUTPUT_DIR = Path(OUTPUT_BASE_DIR) / EXP_NAME

WINDOW_DAYS = 180

# Amostragem determinística por hash.
# Em média, 12.000 ppm = 1,2% da base elegível.
BASE_SAMPLE_PPM = 12000

# Oversampling para transações normais mais informativas.
HIGH_VALUE_SAMPLE_PPM = 60000       # 6%
MID_VALUE_SAMPLE_PPM = 25000        # 2,5%
RANDOM_KEY_SAMPLE_PPM = 25000       # 2,5%

# Cap diário para evitar explodir a amostra.
DAILY_CAP = 3000

# True na primeira execução ou quando quiser recriar a amostra.
OVERWRITE_TABLES = True

# Mantém output CSVs pequenos.
CSV_LIMIT = 1000


# ============================================================
# LOGGING
# ============================================================

warnings.filterwarnings("ignore")
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)


# ============================================================
# SPARK
# ============================================================

def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("EXP-010F - Normal Sampling v2 180d")

        # Recursos conservadores para CML/K8s
        .config("spark.driver.memory", "6g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.executor.memory", "8g")
        .config("spark.executor.cores", "1")
        .config("spark.dynamicAllocation.enabled", "false")
        .config("spark.executor.instances", "4")

        # Paralelismo compatível com poucos executors
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.default.parallelism", "64")

        # Otimizações
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.autoBroadcastJoinThreshold", "104857600")
        .config("spark.sql.broadcastTimeout", "1200")
        .config("spark.network.timeout", "1200s")
        .config("spark.executor.heartbeatInterval", "60s")
        .config("spark.yarn.executor.memoryOverhead", "2048")

        # Console silencioso
        .config("spark.ui.showConsoleProgress", "false")

        .enableHiveSupport()
        .getOrCreate()
    )


def quiet_spark_logs(spark: SparkSession) -> None:
    try:
        spark.sparkContext.setLogLevel("ERROR")
    except Exception:
        pass

    try:
        log4j = spark._jvm.org.apache.log4j
        log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)

        noisy_loggers = [
            "org.apache.spark",
            "org.apache.spark.scheduler",
            "org.apache.spark.scheduler.cluster.k8s",
            "org.apache.spark.scheduler.cluster.k8s.ExecutorPodsAllocator",
            "org.apache.spark.scheduler.cluster.k8s.ExecutorPodsSnapshotsStoreImpl",
            "org.apache.hadoop",
            "org.apache.hive",
            "org.sparkproject",
            "io.fabric8.kubernetes.client",
        ]

        for name in noisy_loggers:
            log4j.LogManager.getLogger(name).setLevel(log4j.Level.ERROR)

    except Exception:
        pass


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
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe_json(obj), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_pandas_csv(pdf: pd.DataFrame, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    pdf.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def safe_count(df, label: str) -> int:
    try:
        n = int(df.count())
        print(f"[COUNT] {label}: {n}")
        return n
    except Exception as exc:
        print(f"[WARN] Falha ao contar {label}: {exc}")
        return -1


def table_exists(spark: SparkSession, table_name: str) -> bool:
    try:
        spark.table(table_name).limit(1).count()
        return True
    except Exception:
        return False


def non_empty_col(colname: str):
    return (
        F.col(colname).isNotNull()
        & (F.length(F.trim(F.col(colname).cast("string"))) > 0)
        & (~F.lower(F.trim(F.col(colname).cast("string"))).isin(
            "", "nan", "none", "null", "<na>", "informação ausente", "informacao ausente"
        ))
    )


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
    except Exception:
        out2 = out.select([F.col(c).cast("string").alias(c) for c in out.columns])
        pdf = out2.toPandas()

    pdf.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def infer_window(spark: SparkSession) -> tuple[str, str]:
    row = (
        spark.table(MAF_HYDRATED_TABLE)
        .select(F.max(F.to_date("dt_pix")).alias("max_date"))
        .collect()[0]
    )

    max_date = row["max_date"]

    if max_date is None:
        raise RuntimeError("Não foi possível inferir max(dt_pix) da base MAF.")

    end_date = pd.to_datetime(str(max_date)).date()
    start_date = end_date - timedelta(days=WINDOW_DAYS - 1)

    return str(start_date), str(end_date)


def save_table(spark: SparkSession, df, table_name: str) -> None:
    if OVERWRITE_TABLES:
        print(f"[TABLE] DROP IF EXISTS {table_name}")
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    print(f"[TABLE] Salvando {table_name}")
    df.write.mode("overwrite").format("parquet").saveAsTable(table_name)
    print(f"[TABLE] OK {table_name}")


# ============================================================
# BUILD NORMAL SAMPLE
# ============================================================

def build_known_fraud_keys(spark: SparkSession):
    maf = (
        spark.table(MAF_HYDRATED_TABLE)
        .select(F.trim(F.col("cd_pix").cast("string")).alias("cd_pix"))
        .filter(F.col("cd_pix").isNotNull())
        .filter(F.length(F.col("cd_pix")) > 0)
        .dropDuplicates(["cd_pix"])
    )

    # Pode existir por causa do EXP-010E-R2B.
    if table_exists(spark, "hmo_ml.tb_pix_maf_mbk_hydration_180d_v2"):
        hydration = (
            spark.table("hmo_ml.tb_pix_maf_mbk_hydration_180d_v2")
            .select(F.trim(F.col("cd_pix").cast("string")).alias("cd_pix"))
            .filter(F.col("cd_pix").isNotNull())
            .filter(F.length(F.col("cd_pix")) > 0)
            .dropDuplicates(["cd_pix"])
        )
        return maf.unionByName(hydration).dropDuplicates(["cd_pix"])

    return maf


def build_normal_candidates(spark: SparkSession, start_date: str, end_date: str):
    print("[1/6] Lendo candidatos normais PIX...")

    df = spark.sql(f"""
        SELECT
            trim(t.ds_id_pix) as cd_pix,
            LPAD(t.nr_cpf_cnpj_origem, 14, '0') as cd_cpf_pagador,
            LPAD(t.nr_cpf_cnpj_destino, 14, '0') as cd_cpf_cnpj_recebedor,
            cast(t.vl_pix as double) as vl_pix,
            cast(t.dt_pix as timestamp) as dt_pix,
            cast(t.dt_pix as date) as data_pix,
            COALESCE(t.ds_chave_pix, 'Informação ausente') as ds_chave_pix,
            CASE
                WHEN t.ds_chave_pix IS NULL THEN 'Informação ausente'
                WHEN length(t.ds_chave_pix) >= 32 THEN 'CHAVE ALEATORIA'
                WHEN t.ds_chave_pix LIKE '%@%' THEN 'EMAIL'
                WHEN regexp_like(t.ds_chave_pix, '^[0-9]+$') AND length(t.ds_chave_pix) >= 11 THEN 'DOCUMENTO/TELEFONE'
                ELSE 'OUTROS'
            END as ds_tipo_chave
        FROM landing_brb_oracle_blk.tb_extrato_pix t
        INNER JOIN landing_brb_oracle_blk.tb_registro_pix r
            ON t.ds_id_pix = r.ds_id_pix
        WHERE cast(t.cd_ispb_origem as int) = 208
          AND r.st_processamento_retorno <> 'RJCT'
          AND t.nr_cpf_cnpj_origem <> t.nr_cpf_cnpj_destino
          AND cast(t.dt_pix as date) >= date('{start_date}')
          AND cast(t.dt_pix as date) <= date('{end_date}')
          AND t.ds_id_pix IS NOT NULL
          AND length(trim(t.ds_id_pix)) > 0
    """)

    df = (
        df
        .filter(F.col("cd_pix").rlike(r"^E[A-Za-z0-9]{20,}$"))
        .dropDuplicates(["cd_pix"])
    )

    return df


def add_sampling_columns(df):
    print("[2/6] Criando estratos e hash de amostragem...")

    df = (
        df
        .withColumn("hour", F.hour("dt_pix"))
        .withColumn(
            "periodo_dia",
            F.when(F.col("hour").between(0, 5), F.lit("madrugada"))
             .when(F.col("hour").between(6, 11), F.lit("manha"))
             .when(F.col("hour").between(12, 17), F.lit("tarde"))
             .otherwise(F.lit("noite"))
        )
        .withColumn(
            "value_band",
            F.when(F.col("vl_pix") < 100, F.lit("A_000_100"))
             .when(F.col("vl_pix") < 500, F.lit("B_100_500"))
             .when(F.col("vl_pix") < 1000, F.lit("C_500_1000"))
             .when(F.col("vl_pix") < 5000, F.lit("D_1000_5000"))
             .when(F.col("vl_pix") < 10000, F.lit("E_5000_10000"))
             .otherwise(F.lit("F_10000_PLUS"))
        )
        .withColumn(
            "sample_ppm",
            F.when(F.col("vl_pix") >= 10000, F.lit(HIGH_VALUE_SAMPLE_PPM))
             .when(F.col("vl_pix") >= 1000, F.lit(MID_VALUE_SAMPLE_PPM))
             .when(F.col("ds_tipo_chave") == "CHAVE ALEATORIA", F.lit(RANDOM_KEY_SAMPLE_PPM))
             .otherwise(F.lit(BASE_SAMPLE_PPM))
        )
        .withColumn("_sample_hash", F.pmod(F.xxhash64(F.col("cd_pix")), F.lit(1000000)))
        .withColumn(
            "sample_priority",
            F.when(F.col("vl_pix") >= 10000, F.lit(4))
             .when(F.col("vl_pix") >= 1000, F.lit(3))
             .when(F.col("ds_tipo_chave") == "CHAVE ALEATORIA", F.lit(2))
             .otherwise(F.lit(1))
        )
        .withColumn(
            "sample_reason",
            F.when(F.col("vl_pix") >= 10000, F.lit("high_value_10k_plus"))
             .when(F.col("vl_pix") >= 1000, F.lit("mid_high_value_1k_plus"))
             .when(F.col("ds_tipo_chave") == "CHAVE ALEATORIA", F.lit("random_key_oversample"))
             .otherwise(F.lit("base_hash_sample"))
        )
    )

    return df


def sample_normals(df):
    print("[3/6] Aplicando amostragem hash + cap diário...")

    sampled = df.filter(F.col("_sample_hash") < F.col("sample_ppm"))

    w_day = (
        Window.partitionBy("data_pix")
        .orderBy(
            F.col("sample_priority").desc(),
            F.col("_sample_hash").asc(),
        )
    )

    sampled = (
        sampled
        .withColumn("_rn_day", F.row_number().over(w_day))
        .filter(F.col("_rn_day") <= F.lit(DAILY_CAP))
        .drop("_rn_day")
        .withColumn("is_fraud", F.lit(0))
        .withColumn("source_dataset", F.lit("normal_180d_sample_v2"))
        .withColumn("normal_sample_version", F.lit("EXP-010F"))
        .withColumn("dt_carga", F.current_timestamp())
    )

    return sampled


def join_mbk_if_available(spark: SparkSession, sampled):
    print("[5/6] Enriquecendo com MBK compacta quando disponível...")

    if not table_exists(spark, MBK_COMPACT_TABLE):
        print(f"[WARN] Tabela MBK compacta não encontrada: {MBK_COMPACT_TABLE}")
        return sampled

    compact = spark.table(MBK_COMPACT_TABLE)

    score_col = F.coalesce(F.col("mbk_completeness_score"), F.lit(-1))

    w_mbk = (
        Window.partitionBy("end_to_end_id")
        .orderBy(
            score_col.desc(),
            F.col("autdatref").desc_nulls_last(),
            F.col("autdathorini").desc_nulls_last(),
        )
    )

    mbk_dedup = (
        compact
        .withColumn("_rn_mbk", F.row_number().over(w_mbk))
        .filter(F.col("_rn_mbk") == 1)
        .drop("_rn_mbk")
    )

    enriched = (
        sampled
        .join(
            mbk_dedup,
            sampled.cd_pix == mbk_dedup.end_to_end_id,
            "left",
        )
        .drop("end_to_end_id")
    )

    return enriched


def build_reports(spark: SparkSession, start_date: str, end_date: str, sampled, enriched):
    print("[6/6] Gerando relatórios...")

    n_sampled = safe_count(sampled, "normais_sample")
    n_enriched = safe_count(enriched, "normais_sample_mbk")

    matched_mbk = (
        enriched
        .filter(F.col("autdatref").isNotNull())
    )

    n_matched_mbk = safe_count(matched_mbk, "normais_sample_mbk_matched")
    mbk_coverage = round(n_matched_mbk / max(n_enriched, 1), 6)

    by_day = (
        sampled
        .groupBy("data_pix")
        .agg(
            F.count("*").alias("n_normais_sample"),
            F.sum(F.when(F.col("vl_pix") >= 1000, 1).otherwise(0)).alias("n_vl_1000_plus"),
            F.sum(F.when(F.col("ds_tipo_chave") == "CHAVE ALEATORIA", 1).otherwise(0)).alias("n_random_key"),
        )
        .orderBy("data_pix")
    )
    write_limited_csv(by_day, "01_distribution_by_day.csv", limit=10000)

    by_value = (
        sampled
        .groupBy("value_band")
        .agg(
            F.count("*").alias("n_rows"),
            F.round(F.avg("vl_pix"), 2).alias("avg_vl_pix"),
            F.round(F.expr("percentile_approx(vl_pix, 0.5)"), 2).alias("median_vl_pix"),
            F.round(F.max("vl_pix"), 2).alias("max_vl_pix"),
        )
        .orderBy("value_band")
    )
    write_limited_csv(by_value, "02_distribution_by_value_band.csv", limit=1000)

    by_key = (
        sampled
        .groupBy("ds_tipo_chave")
        .agg(F.count("*").alias("n_rows"))
        .orderBy(F.col("n_rows").desc())
    )
    write_limited_csv(by_key, "03_distribution_by_key_type.csv", limit=1000)

    by_reason = (
        sampled
        .groupBy("sample_reason")
        .agg(F.count("*").alias("n_rows"))
        .orderBy(F.col("n_rows").desc())
    )
    write_limited_csv(by_reason, "04_distribution_by_sample_reason.csv", limit=1000)

    mbk_summary = pd.DataFrame(
        [
            {
                "n_sampled": n_sampled,
                "n_enriched": n_enriched,
                "n_matched_mbk": n_matched_mbk,
                "mbk_coverage": mbk_coverage,
            }
        ]
    )
    write_pandas_csv(mbk_summary, "05_mbk_coverage_summary.csv")

    write_limited_csv(
        enriched,
        "06_sample_preview.csv",
        limit=CSV_LIMIT,
        order_cols=["dt_pix"],
    )

    recommendation_lines = [
        "# EXP-010F — Normal Sampling v2 / 180 dias",
        "",
        f"Gerado em: `{now_iso()}`",
        "",
        "## Resultado executivo",
        "",
        f"- Janela: `{start_date}` a `{end_date}`",
        f"- Tabela normal: `{OUTPUT_NORMAL_TABLE}`",
        f"- Tabela normal + MBK compacta: `{OUTPUT_NORMAL_MBK_TABLE}`",
        f"- Normais amostrados: `{n_sampled}`",
        f"- Normais enriquecidos: `{n_enriched}`",
        f"- Matches MBK compacta: `{n_matched_mbk}`",
        f"- Cobertura MBK na amostra normal: `{mbk_coverage}`",
        "",
        "## Interpretação",
        "",
        "Este experimento gera a amostra normal de 180 dias para uso no EXP-010G.",
        "A amostra exclui fraudes MAF conhecidas e usa amostragem determinística por hash, com oversampling de valores altos e chave aleatória.",
        "",
        "A cobertura MBK nos normais depende do compact index já existente. Se a cobertura vier parcial, isso não bloqueia o EXP-010G; a feature de missingness deverá ser mantida.",
        "",
        "## Próximo passo",
        "",
        "Seguir para EXP-010G — Unified Dataset Builder v2, unificando fraudes MAF hidratadas, normais amostrados e features finais leakage-free.",
        "",
    ]

    (OUTPUT_DIR / "07_recommendation.md").write_text(
        "\n".join(recommendation_lines),
        encoding="utf-8",
    )

    return {
        "n_sampled": n_sampled,
        "n_enriched": n_enriched,
        "n_matched_mbk": n_matched_mbk,
        "mbk_coverage": mbk_coverage,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    ensure_output_dir()

    spark = create_spark_session()
    quiet_spark_logs(spark)

    print("=" * 80)
    print("EXP-010F — Normal Sampling v2 / 180 dias")
    print("=" * 80)
    print(f"MAF_HYDRATED_TABLE: {MAF_HYDRATED_TABLE}")
    print(f"MBK_COMPACT_TABLE: {MBK_COMPACT_TABLE}")
    print(f"OUTPUT_NORMAL_TABLE: {OUTPUT_NORMAL_TABLE}")
    print(f"OUTPUT_NORMAL_MBK_TABLE: {OUTPUT_NORMAL_MBK_TABLE}")
    print(f"WINDOW_DAYS: {WINDOW_DAYS}")
    print(f"BASE_SAMPLE_PPM: {BASE_SAMPLE_PPM}")
    print(f"DAILY_CAP: {DAILY_CAP}")
    print(f"OVERWRITE_TABLES: {OVERWRITE_TABLES}")
    print("=" * 80)

    start_date, end_date = infer_window(spark)
    print(f"[WINDOW] {start_date} a {end_date}")

    fraud_keys = build_known_fraud_keys(spark).persist(StorageLevel.MEMORY_AND_DISK)
    n_fraud_keys = safe_count(fraud_keys, "known_fraud_keys")

    candidates = build_normal_candidates(spark, start_date, end_date)

    print("[1b/6] Excluindo fraudes conhecidas...")
    normal_candidates = (
        candidates
        .join(F.broadcast(fraud_keys), on="cd_pix", how="left_anti")
    )

    normal_candidates = add_sampling_columns(normal_candidates)

    sampled = sample_normals(normal_candidates).persist(StorageLevel.MEMORY_AND_DISK)

    n_sampled = safe_count(sampled, "sampled_normals_before_save")

    if n_sampled == 0:
        raise RuntimeError("A amostra normal ficou vazia. Ajuste SAMPLE_PPM/DAILY_CAP ou revise filtros.")

    print("[4/6] Salvando tabela de normais amostrados...")
    save_table(spark, sampled, OUTPUT_NORMAL_TABLE)

    sampled_from_table = spark.table(OUTPUT_NORMAL_TABLE)

    enriched = join_mbk_if_available(spark, sampled_from_table).persist(StorageLevel.MEMORY_AND_DISK)

    save_table(spark, enriched, OUTPUT_NORMAL_MBK_TABLE)

    summary_counts = build_reports(
        spark=spark,
        start_date=start_date,
        end_date=end_date,
        sampled=sampled_from_table,
        enriched=spark.table(OUTPUT_NORMAL_MBK_TABLE),
    )

    elapsed_min = round((time.time() - t0) / 60, 2)

    run_summary = {
        "generated_at": now_iso(),
        "experiment": "EXP-010F",
        "status": "DONE",
        "elapsed_min": elapsed_min,
        "window_start_date": start_date,
        "window_end_date": end_date,
        "window_days": WINDOW_DAYS,
        "base_sample_ppm": BASE_SAMPLE_PPM,
        "high_value_sample_ppm": HIGH_VALUE_SAMPLE_PPM,
        "mid_value_sample_ppm": MID_VALUE_SAMPLE_PPM,
        "random_key_sample_ppm": RANDOM_KEY_SAMPLE_PPM,
        "daily_cap": DAILY_CAP,
        "known_fraud_keys": n_fraud_keys,
        "output_normal_table": OUTPUT_NORMAL_TABLE,
        "output_normal_mbk_table": OUTPUT_NORMAL_MBK_TABLE,
        "summary_counts": summary_counts,
        "artifacts": [
            "01_distribution_by_day.csv",
            "02_distribution_by_value_band.csv",
            "03_distribution_by_key_type.csv",
            "04_distribution_by_sample_reason.csv",
            "05_mbk_coverage_summary.csv",
            "06_sample_preview.csv",
            "07_recommendation.md",
        ],
    }

    write_json(OUTPUT_DIR / "00_run_summary.json", run_summary)

    print()
    print("=" * 80)
    print("[OK] EXP-010F finalizado")
    print(f"[OK] Tempo total: {elapsed_min} min")
    print(f"[OK] Normais amostrados: {summary_counts.get('n_sampled')}")
    print(f"[OK] Cobertura MBK: {summary_counts.get('mbk_coverage')}")
    print(f"[OK] Artefatos: {OUTPUT_DIR}")
    print("=" * 80)

    try:
        fraud_keys.unpersist()
        sampled.unpersist()
        enriched.unpersist()
    except Exception:
        pass


if __name__ == "__main__":
    main()