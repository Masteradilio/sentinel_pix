# -*- coding: utf-8 -*-
"""
EXP-010E — MBK Keyed Hydration Audit

Objetivo:
  Hidratar, de forma controlada, os dados MBK/mobile das fraudes MAF já
  curadas e hidratadas no EXP-010C.

Entrada principal:
  hmo_ml.tb_pix_fraudes_maf_hidratadas_v1

Fonte MBK:
  landing_brb_oracle_mbk.aut

Saídas Hive:
  hmo_ml.tb_pix_maf_mbk_target_keys_v1
  hmo_ml.tb_pix_maf_mbk_hydration_audit_v1

Saídas locais:
  /home/cdsw/Adilio/rebuild_pix/Artefatos/EXP-010E/

Estratégia:
  - não varrer MBK inteira;
  - criar tabela pequena de chaves alvo;
  - processar MBK por mês de dt_pix;
  - filtrar MBK por autdatref entre min/max do mês alvo com pequena folga;
  - extrair E2E ID do XML/texto da MBK;
  - fazer join com broadcast das chaves daquele mês;
  - deduplicar por transaction_id priorizando completude;
  - gerar cobertura por mês e por campo.

Execução recomendada:
  1) piloto:
     python exp_010e_mbk_keyed_hydration_audit.py --max-months 1

  2) execução completa:
     python exp_010e_mbk_keyed_hydration_audit.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timedelta
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

MAF_HYDRATED_TABLE = "hmo_ml.tb_pix_fraudes_maf_hidratadas_v1"
MBK_TABLE = "landing_brb_oracle_mbk.aut"

TARGET_KEYS_TABLE = "hmo_ml.tb_pix_maf_mbk_target_keys_v1"
MBK_AUDIT_TABLE = "hmo_ml.tb_pix_maf_mbk_hydration_audit_v1"

OUTPUT_BASE_DIR = "/home/cdsw/Adilio/rebuild_pix/Artefatos"
EXP_NAME = "EXP-010E"
OUTPUT_DIR = Path(OUTPUT_BASE_DIR) / EXP_NAME

CSV_LIMIT = 1000

# Folga ao buscar MBK ao redor da data da transação.
# Normalmente autdatref deve bater com dt_pix, mas a folga protege variação de timezone/carga.
DATE_PAD_DAYS = 2

# Se True, recria tabelas destino.
OVERWRITE_TABLES = True

# Modo seguro: não contar MBK bruta filtrada por mês, pois isso pode custar caro.
COUNT_RAW_MBK_CANDIDATES = False


MBK_OUTPUT_COLUMNS = [
    "transaction_id",
    "target_dt_pix",
    "target_date",
    "target_month",
    "autdatref",
    "autdathorini",
    "autcodret",
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
    "is_agendamento_recorrente",
    "mbk_completeness_score",
    "hydrated_at",
]


# ============================================================
# SPARK
# ============================================================

def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("EXP-010E - MBK Keyed Hydration Audit")
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
    path.write_text(
        json.dumps(safe_json(obj), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def safe_count(df, label: str) -> int:
    try:
        n = int(df.count())
        print(f"[COUNT] {label}: {n}")
        return n
    except Exception as exc:
        print(f"[WARN] Falha ao contar {label}: {exc}")
        return -1


def non_empty_col(colname: str):
    return (
        F.col(colname).isNotNull()
        & (F.length(F.trim(F.col(colname).cast("string"))) > 0)
        & (~F.lower(F.trim(F.col(colname).cast("string"))).isin(
            "", "nan", "none", "null", "<na>", "informação ausente", "informacao ausente"
        ))
    )


def nonempty_expr(expr):
    return F.when(F.length(F.trim(expr.cast("string"))) > 0, F.trim(expr.cast("string")))


def first_non_empty_expr(*exprs):
    cleaned = [nonempty_expr(e) for e in exprs]
    return F.coalesce(*cleaned)


def build_completeness_score(df, cols: list[str]):
    score = F.lit(0)
    for c in cols:
        if c in df.columns:
            score = score + F.when(non_empty_col(c), F.lit(1)).otherwise(F.lit(0))
    return score


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


def write_pandas_csv(pdf: pd.DataFrame, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    pdf.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def table_exists(spark: SparkSession, table_name: str) -> bool:
    try:
        spark.table(table_name).limit(1).count()
        return True
    except Exception:
        return False


def save_overwrite_table(spark: SparkSession, df, table_name: str) -> None:
    if OVERWRITE_TABLES:
        print(f"[TABLE] DROP IF EXISTS {table_name}")
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    print(f"[TABLE] Salvando {table_name}")
    df.write.mode("overwrite").format("parquet").saveAsTable(table_name)
    print(f"[TABLE] OK {table_name}")


def append_or_create_table(spark: SparkSession, df, table_name: str, first_write: bool) -> bool:
    mode = "overwrite" if first_write else "append"

    if first_write and OVERWRITE_TABLES:
        print(f"[TABLE] DROP IF EXISTS {table_name}")
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    print(f"[TABLE] {mode.upper()} {table_name}")
    df.write.mode(mode).format("parquet").saveAsTable(table_name)
    print(f"[TABLE] OK {table_name}")

    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EXP-010E MBK Keyed Hydration Audit")
    parser.add_argument(
        "--max-months",
        type=int,
        default=0,
        help="Limita quantidade de meses processados. 0 = todos.",
    )
    parser.add_argument(
        "--months",
        type=str,
        default="",
        help="Lista opcional de meses YYYY-MM separados por vírgula. Ex: 2026-04,2026-05",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Não derruba tabelas destino antes de executar.",
    )
    return parser.parse_args()


# ============================================================
# TARGET KEYS
# ============================================================

def build_target_keys(spark: SparkSession):
    print("[1/7] Criando tabela de chaves alvo MAF...")

    src = spark.table(MAF_HYDRATED_TABLE)

    target = (
        src
        .select(
            F.trim(F.col("cd_pix").cast("string")).alias("cd_pix"),
            F.col("dt_pix").cast("timestamp").alias("dt_pix"),
            F.to_date("dt_pix").alias("target_date"),
            F.date_format(F.to_date("dt_pix"), "yyyy-MM").alias("target_month"),
            F.col("cd_cpf_pagador").cast("string").alias("cd_cpf_pagador"),
            F.col("cd_cpf_cnpj_recebedor").cast("string").alias("cd_cpf_cnpj_recebedor"),
            F.col("vl_pix").cast("double").alias("vl_pix"),
        )
        .filter(F.col("cd_pix").isNotNull())
        .filter(F.length(F.trim(F.col("cd_pix"))) > 0)
        .filter(F.col("cd_pix").rlike(r"^E[A-Za-z0-9]{20,}$"))
        .dropDuplicates(["cd_pix"])
    )

    save_overwrite_table(spark, target, TARGET_KEYS_TABLE)

    return target.persist(StorageLevel.MEMORY_AND_DISK)


def get_month_plan(target_keys, args: argparse.Namespace) -> list[dict[str, Any]]:
    month_df = (
        target_keys
        .groupBy("target_month")
        .agg(
            F.count("*").alias("n_keys"),
            F.min("target_date").alias("min_target_date"),
            F.max("target_date").alias("max_target_date"),
        )
        .orderBy("target_month")
    )

    month_pdf = month_df.toPandas()

    if args.months.strip():
        wanted = {m.strip() for m in args.months.split(",") if m.strip()}
        month_pdf = month_pdf[month_pdf["target_month"].isin(wanted)].copy()

    if args.max_months and args.max_months > 0:
        month_pdf = month_pdf.head(int(args.max_months)).copy()

    write_pandas_csv(month_pdf, "01_target_key_month_distribution.csv")

    plan: list[dict[str, Any]] = []

    for _, row in month_pdf.iterrows():
        min_date = pd.to_datetime(row["min_target_date"]).date()
        max_date = pd.to_datetime(row["max_target_date"]).date()

        start_date = min_date - timedelta(days=DATE_PAD_DAYS)
        end_date = max_date + timedelta(days=DATE_PAD_DAYS)

        plan.append(
            {
                "target_month": str(row["target_month"]),
                "n_keys": int(row["n_keys"]),
                "min_target_date": str(min_date),
                "max_target_date": str(max_date),
                "lookup_start_date": str(start_date),
                "lookup_end_date": str(end_date),
            }
        )

    return plan


# ============================================================
# MBK EXTRACTION
# ============================================================

def extract_e2e_expr():
    auttrn = F.col("auttrn").cast("string")

    return first_non_empty_expr(
        F.regexp_extract(auttrn, r"<BRB__IdFimAFimOriginalPix.*?>(.*?)</BRB__IdFimAFimOriginalPix>", 1),
        F.regexp_extract(auttrn, r"<FTN__IdFimAFimOriginalPix.*?>(.*?)</FTN__IdFimAFimOriginalPix>", 1),
        F.regexp_extract(auttrn, r"<idFimAfim.*?>(.*?)</idFimAfim>", 1),
        F.regexp_extract(auttrn, r"<idFimAFim.*?>(.*?)</idFimAFim>", 1),
        F.regexp_extract(auttrn, r"<endToEndId.*?>(.*?)</endToEndId>", 1),
        F.regexp_extract(auttrn, r"(E[0-9A-Za-z]{20,})", 1),
    )


def build_mbk_candidates(spark: SparkSession, lookup_start_date: str, lookup_end_date: str):
    print(f"[MBK] Lendo MBK entre {lookup_start_date} e {lookup_end_date}")

    mbk = spark.table(MBK_TABLE)

    auttrn = F.col("auttrn").cast("string")

    # Filtro barato antes dos regex.
    candidate_filter = (
        auttrn.rlike(r"E[0-9A-Za-z]{20,}")
        | auttrn.like("%FimAfim%")
        | auttrn.like("%FimAFim%")
        | auttrn.like("%IdFim%")
        | auttrn.like("%idFim%")
        | auttrn.like("%Pix%")
        | auttrn.like("%PIX%")
    )

    df = (
        mbk
        .filter(F.col("autdatref") >= F.lit(lookup_start_date))
        .filter(F.col("autdatref") <= F.lit(lookup_end_date))
        .filter(F.col("auttrn").isNotNull())
        .filter(candidate_filter)
        .select(
            extract_e2e_expr().alias("transaction_id"),
            F.col("autdatref").cast("string").alias("autdatref"),
            F.col("autdathorini").cast("string").alias("autdathorini"),
            F.col("autcodret").cast("string").alias("autcodret"),
            F.regexp_extract(auttrn, r"<FTN__NomeDispositivo.*?>(.*?)</FTN__NomeDispositivo>", 1).alias("device_name"),
            F.regexp_extract(auttrn, r"<BRB__UserAgentTopaz.*?>(.*?)</BRB__UserAgentTopaz>", 1).alias("app_version"),
            first_non_empty_expr(
                F.regexp_extract(auttrn, r"<FTN__IpUsuario.*?>(.*?)</FTN__IpUsuario>", 1),
                F.regexp_extract(auttrn, r"<ip>(.*?)</ip>", 1),
            ).alias("ip_address"),
            F.regexp_extract(auttrn, r"<tempoRede.*?>(.*?)</tempoRede>", 1).cast("int").alias("latencia_rede_ms"),
            F.regexp_extract(auttrn, r"<tempoAtendimento.*?>(.*?)</tempoAtendimento>", 1).cast("int").alias("tempo_interacao_ms"),
            F.regexp_extract(auttrn, r"<tempoAutorizacao.*?>(.*?)</tempoAutorizacao>", 1).cast("int").alias("tempo_processamento_host_ms"),
            F.regexp_extract(auttrn, r"<BRB__AuthenticationMethodTopaz.*?>(.*?)</BRB__AuthenticationMethodTopaz>", 1).alias("metodo_autenticacao"),
            F.regexp_extract(auttrn, r"<BRB__IdentificadorSessao.*?>(.*?)</BRB__IdentificadorSessao>", 1).alias("session_id"),
            first_non_empty_expr(
                F.regexp_extract(
                    auttrn,
                    r"<BRB__ResultadoConsultaScoreTopaz[^>]*tipo=\"java.lang.Integer\"[^>]*>(.*?)</BRB__ResultadoConsultaScoreTopaz>",
                    1,
                ),
                F.regexp_extract(auttrn, r"<BRB__ResultadoConsultaScoreTopaz>(\d+)</BRB__ResultadoConsultaScoreTopaz>", 1),
            ).cast("int").alias("topaz_risk_score"),
            F.regexp_extract(auttrn, r"<BRB__TopazTransacaoRejeitada[^>]*>(.*?)</BRB__TopazTransacaoRejeitada>", 1).cast("int").alias("topaz_transacao_rejeitada"),
            F.regexp_extract(auttrn, r"<BRB__IsAgendamentoRecorrenteForTopaz[^>]*>(.*?)</BRB__IsAgendamentoRecorrenteForTopaz>", 1).alias("is_agendamento_recorrente"),
        )
        .filter(F.col("transaction_id").isNotNull())
        .filter(F.length(F.trim(F.col("transaction_id"))) > 0)
    )

    return df


def join_and_dedup_month(spark: SparkSession, target_keys, month_info: dict[str, Any]):
    month = month_info["target_month"]

    print("=" * 80)
    print(f"[2/7] Processando mês {month} | keys={month_info['n_keys']}")
    print("=" * 80)

    keys_month = (
        target_keys
        .filter(F.col("target_month") == F.lit(month))
        .select(
            F.col("cd_pix"),
            F.col("dt_pix").alias("target_dt_pix"),
            F.col("target_date"),
            F.col("target_month"),
        )
        .dropDuplicates(["cd_pix"])
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    mbk_candidates = build_mbk_candidates(
        spark,
        lookup_start_date=month_info["lookup_start_date"],
        lookup_end_date=month_info["lookup_end_date"],
    )

    if COUNT_RAW_MBK_CANDIDATES:
        safe_count(mbk_candidates, f"mbk_candidates_{month}")

    joined = (
        mbk_candidates
        .join(
            F.broadcast(keys_month),
            mbk_candidates.transaction_id == keys_month.cd_pix,
            "inner",
        )
        .drop("cd_pix")
    )

    score_cols = [
        "device_name",
        "app_version",
        "ip_address",
        "latencia_rede_ms",
        "tempo_interacao_ms",
        "tempo_processamento_host_ms",
        "metodo_autenticacao",
        "session_id",
        "autcodret",
        "topaz_risk_score",
        "topaz_transacao_rejeitada",
        "is_agendamento_recorrente",
    ]

    joined = joined.withColumn("mbk_completeness_score", build_completeness_score(joined, score_cols))

    w = (
        Window.partitionBy("transaction_id")
        .orderBy(
            F.col("mbk_completeness_score").desc(),
            F.col("autdatref").desc_nulls_last(),
            F.col("autdathorini").desc_nulls_last(),
        )
    )

    dedup = (
        joined
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
        .withColumn("hydrated_at", F.current_timestamp())
    )

    for c in MBK_OUTPUT_COLUMNS:
        if c not in dedup.columns:
            dedup = dedup.withColumn(c, F.lit(None).cast("string"))

    dedup = dedup.select(*MBK_OUTPUT_COLUMNS)

    n_join_rows = safe_count(joined, f"joined_rows_{month}")
    n_matched = safe_count(dedup, f"matched_transactions_{month}")

    keys_month.unpersist()

    return dedup, {
        "target_month": month,
        "n_keys": int(month_info["n_keys"]),
        "lookup_start_date": month_info["lookup_start_date"],
        "lookup_end_date": month_info["lookup_end_date"],
        "n_join_rows": n_join_rows,
        "n_matched_transactions": n_matched,
        "coverage_pct": round(n_matched / max(int(month_info["n_keys"]), 1), 6),
    }


# ============================================================
# REPORTS
# ============================================================

def build_coverage_reports(spark: SparkSession, target_keys):
    print("[5/7] Gerando relatórios de cobertura...")

    if not table_exists(spark, MBK_AUDIT_TABLE):
        print("[WARN] Tabela MBK audit ainda não existe.")
        return None, None, None

    mbk = spark.table(MBK_AUDIT_TABLE)

    matched_ids = (
        mbk
        .select(F.col("transaction_id").alias("cd_pix"))
        .dropDuplicates()
        .withColumn("matched_mbk", F.lit(1))
    )

    coverage_by_month = (
        target_keys
        .join(matched_ids, on="cd_pix", how="left")
        .withColumn("matched_mbk", F.coalesce(F.col("matched_mbk"), F.lit(0)))
        .groupBy("target_month")
        .agg(
            F.count("*").alias("n_target_keys"),
            F.sum("matched_mbk").alias("n_matched_mbk"),
        )
        .withColumn("coverage_pct", F.round(F.col("n_matched_mbk") / F.col("n_target_keys"), 6))
        .orderBy("target_month")
    )

    write_limited_csv(coverage_by_month, "03_mbk_coverage_by_month.csv", limit=10000)

    total_target = safe_count(target_keys, "total_target_keys")
    total_matched = safe_count(matched_ids, "total_mbk_matched_keys")

    field_rows = []

    mbk_count = safe_count(mbk, "mbk_audit_table_rows")

    for c in [
        "device_name",
        "app_version",
        "ip_address",
        "latencia_rede_ms",
        "tempo_interacao_ms",
        "tempo_processamento_host_ms",
        "metodo_autenticacao",
        "session_id",
        "autcodret",
        "topaz_risk_score",
        "topaz_transacao_rejeitada",
        "is_agendamento_recorrente",
    ]:
        if c not in mbk.columns:
            field_rows.append(
                {
                    "field": c,
                    "n_non_empty": 0,
                    "n_rows": mbk_count,
                    "coverage_pct": 0.0,
                }
            )
            continue

        n_non_empty = (
            mbk
            .agg(F.sum(F.when(non_empty_col(c), F.lit(1)).otherwise(F.lit(0))).alias("n"))
            .collect()[0]["n"]
        )

        n_non_empty = int(n_non_empty or 0)

        field_rows.append(
            {
                "field": c,
                "n_non_empty": n_non_empty,
                "n_rows": mbk_count,
                "coverage_pct": round(n_non_empty / max(mbk_count, 1), 6),
            }
        )

    field_pdf = pd.DataFrame(field_rows).sort_values("coverage_pct", ascending=False)
    write_pandas_csv(field_pdf, "04_mbk_field_coverage.csv")

    write_limited_csv(
        mbk,
        "05_mbk_matched_sample.csv",
        order_cols=["target_dt_pix"],
    )

    unmatched = (
        target_keys
        .join(matched_ids, on="cd_pix", how="left")
        .filter(F.col("matched_mbk").isNull())
        .select(
            "cd_pix",
            "dt_pix",
            "target_date",
            "target_month",
            "cd_cpf_pagador",
            "cd_cpf_cnpj_recebedor",
            "vl_pix",
        )
    )

    write_limited_csv(
        unmatched,
        "06_mbk_unmatched_sample.csv",
        order_cols=["dt_pix"],
    )

    return {
        "total_target_keys": total_target,
        "total_mbk_matched_keys": total_matched,
        "overall_coverage_pct": round(total_matched / max(total_target, 1), 6),
        "mbk_audit_table_rows": mbk_count,
    }, coverage_by_month, field_pdf


def write_recommendation(
    summary_counts: dict[str, Any],
    batch_results: list[dict[str, Any]],
    field_coverage: pd.DataFrame | None,
) -> None:
    coverage = summary_counts.get("overall_coverage_pct") if summary_counts else None
    matched = summary_counts.get("total_mbk_matched_keys") if summary_counts else None
    total = summary_counts.get("total_target_keys") if summary_counts else None

    lines = [
        "# EXP-010E — MBK Keyed Hydration Audit",
        "",
        f"Gerado em: `{now_iso()}`",
        "",
        "## Objetivo",
        "",
        "Auditar a hidratação MBK/mobile por chave para as fraudes MAF fortes do EXP-010C.",
        "",
        "## Resultado executivo",
        "",
        f"- Target keys: `{total}`",
        f"- Matched MBK: `{matched}`",
        f"- Coverage geral: `{coverage}`",
        f"- Tabela alvo: `{TARGET_KEYS_TABLE}`",
        f"- Tabela MBK audit: `{MBK_AUDIT_TABLE}`",
        "",
        "## Interpretação",
        "",
        "Este experimento não treina modelo. Ele apenas mede se conseguimos recuperar sinais mobile/Topaz de forma controlada, sem varrer a MBK inteira.",
        "",
    ]

    if coverage is None:
        lines.extend(
            [
                "## Decisão",
                "",
                "Não houve cobertura calculada. Verificar execução e tabela MBK audit.",
                "",
            ]
        )
    elif coverage >= 0.50:
        lines.extend(
            [
                "## Decisão",
                "",
                "Cobertura MBK preliminar suficiente para avançar para integração controlada no dataset unificado.",
                "",
            ]
        )
    elif coverage >= 0.20:
        lines.extend(
            [
                "## Decisão",
                "",
                "Cobertura MBK parcial. Avançar com cautela: usar MBK como enriquecimento opcional e investigar meses/campos de baixa cobertura.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Decisão",
                "",
                "Cobertura MBK baixa. Antes de usar no treino, revisar regex de extração, janela de busca, tags XML e filtros da tabela MBK.",
                "",
            ]
        )

    lines.extend(
        [
            "## Próximo passo",
            "",
            "Após avaliar os artefatos, decidir entre:",
            "",
            "1. promover a tabela `tb_pix_maf_mbk_hydration_audit_v1` como fonte de enrichment;",
            "2. ajustar regex/filtros e rerodar o EXP-010E;",
            "3. seguir para EXP-010F — Normal Sampling v2 com MBK opcional.",
            "",
        ]
    )

    (OUTPUT_DIR / "07_recommendation.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    global OVERWRITE_TABLES

    args = parse_args()

    if args.no_overwrite:
        OVERWRITE_TABLES = False

    t0 = time.time()
    ensure_output_dir()

    spark = create_spark_session()

    print("=" * 80)
    print("EXP-010E — MBK Keyed Hydration Audit")
    print("=" * 80)
    print(f"MAF_HYDRATED_TABLE: {MAF_HYDRATED_TABLE}")
    print(f"MBK_TABLE: {MBK_TABLE}")
    print(f"TARGET_KEYS_TABLE: {TARGET_KEYS_TABLE}")
    print(f"MBK_AUDIT_TABLE: {MBK_AUDIT_TABLE}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"DATE_PAD_DAYS: {DATE_PAD_DAYS}")
    print(f"MAX_MONTHS: {args.max_months}")
    print(f"MONTHS: {args.months or 'ALL'}")
    print("=" * 80)

    target_keys = None

    batch_results: list[dict[str, Any]] = []
    first_write = True

    try:
        target_keys = build_target_keys(spark)
        n_target = safe_count(target_keys, "target_keys")

        month_plan = get_month_plan(target_keys, args)

        print(f"[PLAN] Meses a processar: {len(month_plan)}")
        write_json(OUTPUT_DIR / "00a_month_plan.json", month_plan)

        if OVERWRITE_TABLES:
            print(f"[TABLE] DROP IF EXISTS {MBK_AUDIT_TABLE}")
            spark.sql(f"DROP TABLE IF EXISTS {MBK_AUDIT_TABLE}")

        for idx, month_info in enumerate(month_plan, start=1):
            print(f"[BATCH] {idx}/{len(month_plan)} — {month_info['target_month']}")

            dedup_month, batch_summary = join_and_dedup_month(
                spark=spark,
                target_keys=target_keys,
                month_info=month_info,
            )

            if batch_summary["n_matched_transactions"] > 0:
                first_write = append_or_create_table(
                    spark,
                    dedup_month,
                    MBK_AUDIT_TABLE,
                    first_write=first_write,
                )
            else:
                print(f"[BATCH] {month_info['target_month']} sem matches. Nada a gravar.")

            batch_results.append(batch_summary)
            write_pandas_csv(pd.DataFrame(batch_results), "02_batch_results.csv")

        summary_counts, coverage_by_month, field_coverage = build_coverage_reports(spark, target_keys)

        write_recommendation(summary_counts or {}, batch_results, field_coverage)

        elapsed_min = round((time.time() - t0) / 60, 2)

        run_summary = {
            "generated_at": now_iso(),
            "experiment": "EXP-010E",
            "status": "DONE",
            "elapsed_min": elapsed_min,
            "maf_hydrated_table": MAF_HYDRATED_TABLE,
            "mbk_table": MBK_TABLE,
            "target_keys_table": TARGET_KEYS_TABLE,
            "mbk_audit_table": MBK_AUDIT_TABLE,
            "date_pad_days": DATE_PAD_DAYS,
            "n_target_keys": n_target,
            "n_months_planned": len(month_plan),
            "n_months_processed": len(batch_results),
            "summary_counts": summary_counts,
            "batch_results": batch_results,
            "artifacts": [
                "00a_month_plan.json",
                "01_target_key_month_distribution.csv",
                "02_batch_results.csv",
                "03_mbk_coverage_by_month.csv",
                "04_mbk_field_coverage.csv",
                "05_mbk_matched_sample.csv",
                "06_mbk_unmatched_sample.csv",
                "07_recommendation.md",
            ],
        }

        write_json(OUTPUT_DIR / "00_run_summary.json", run_summary)

        print()
        print("=" * 80)
        print("[OK] EXP-010E concluído")
        print(f"[OK] Tempo total: {elapsed_min} min")
        print(f"[OK] Artefatos em: {OUTPUT_DIR}")
        print(f"[OK] Coverage: {(summary_counts or {}).get('overall_coverage_pct')}")
        print("=" * 80)

    finally:
        try:
            if target_keys is not None:
                target_keys.unpersist()
        except Exception as exc:
            print(f"[WARN] Falha ao unpersist target_keys: {exc}")


if __name__ == "__main__":
    main()