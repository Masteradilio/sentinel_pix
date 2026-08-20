# -*- coding: utf-8 -*-
"""
EXP-010E-R2B — MBK Compact Index 180d

Objetivo:
  Criar um índice compacto da MBK para a janela recente de 180 dias,
  processando a tabela landing_brb_oracle_mbk.aut por partição diária
  autdatref, em vez de procurar chaves por regex em lotes/mês.

Por que R2B:
  - EXP-010E original tentou 49 meses: inviável.
  - EXP-010E-R1 tentou keyed hydration por janela de 180 dias: >18h.
  - Esta versão faz scan diário de partições, uma vez por dia, usando filtro simples:
      autdatref = 'YYYY-MM-DD'
      auttrn LIKE '%<transacao%'
    Depois extrai E2E e faz join com as chaves MAF daquele dia.

Entrada:
  hmo_ml.tb_pix_fraudes_maf_hidratadas_v1
  landing_brb_oracle_mbk.aut

Saídas Hive:
  hmo_ml.tb_pix_maf_mbk_target_keys_180d_v2
  hmo_ml.tb_pix_mbk_compact_180d_v1
  hmo_ml.tb_pix_maf_mbk_hydration_180d_v2

Saídas locais:
  /home/cdsw/Adilio/rebuild_pix/Artefatos/EXP-010E-R2B/

Como usar no CML:
  - Para piloto: deixe MAX_DAYS_TO_PROCESS = 3.
  - Para execução completa: altere MAX_DAYS_TO_PROCESS = 0.
  - O script para automaticamente ao atingir TIME_BUDGET_MIN.
"""

from __future__ import annotations

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


if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
    
import logging
import warnings

warnings.filterwarnings("ignore")

logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)


# ============================================================
# CONFIG — ALTERE AQUI NO CML
# ============================================================

MAF_HYDRATED_TABLE = "hmo_ml.tb_pix_fraudes_maf_hidratadas_v1"
MBK_TABLE = "landing_brb_oracle_mbk.aut"

TARGET_KEYS_TABLE = "hmo_ml.tb_pix_maf_mbk_target_keys_180d_v2"
MBK_COMPACT_TABLE = "hmo_ml.tb_pix_mbk_compact_180d_v1"
HYDRATION_TABLE = "hmo_ml.tb_pix_maf_mbk_hydration_180d_v2"

OUTPUT_BASE_DIR = "/home/cdsw/Adilio/rebuild_pix/Artefatos"
EXP_NAME = "EXP-010E-R2B"
OUTPUT_DIR = Path(OUTPUT_BASE_DIR) / EXP_NAME

WINDOW_DAYS = 180

# PILOTO: 3
# COMPLETO: 0
MAX_DAYS_TO_PROCESS = 30

# Orçamento duro de tempo. O script salva parcial e para com status PARTIAL_TIMEOUT.
TIME_BUDGET_MIN = 300

# Folga de data para MBK. Para performance, o padrão agora é 0.
# Se cobertura vier baixa, podemos testar 1 depois.
DATE_PAD_DAYS = 0

CSV_LIMIT = 1000

# True no primeiro run. Em rerun/resume, coloque False para não apagar o índice parcial.
OVERWRITE_TABLES = False

# Se True, pula partições já existentes no MBK_COMPACT_TABLE.
RESUME_SKIP_PROCESSED_DATES = True

# Evita contagens caras na MBK bruta.
DEBUG_COUNTS = False


COMPACT_NONPART_COLS = [
    "end_to_end_id",
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
    "compact_created_at",
]

HYDRATION_COLS = [
    "cd_pix",
    "dt_pix",
    "target_date",
    "cd_cpf_pagador",
    "cd_cpf_cnpj_recebedor",
    "vl_pix",
    "end_to_end_id",
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
        .appName("EXP-010E-R2B - MBK Compact Index 180d")

        # Recursos conservadores para não estourar quota
        .config("spark.driver.memory", "6g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.executor.memory", "8g")
        .config("spark.executor.cores", "1")

        # IMPORTANTE: desliga dynamic allocation para parar spam de criação de pods
        .config("spark.dynamicAllocation.enabled", "false")
        .config("spark.executor.instances", "4")

        # Reduz paralelismo para combinar com poucos executors
        .config("spark.sql.shuffle.partitions", "48")
        .config("spark.default.parallelism", "48")

        # Mantém otimizações úteis
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.autoBroadcastJoinThreshold", "104857600")
        .config("spark.sql.broadcastTimeout", "1200")
        .config("spark.network.timeout", "1200s")
        .config("spark.executor.heartbeatInterval", "60s")
        .config("spark.yarn.executor.memoryOverhead", "2048")

        # Reduz saída no console
        .config("spark.ui.showConsoleProgress", "false")

        .enableHiveSupport()
        .getOrCreate()
    )


# ============================================================
# HELPERS
# ============================================================

def quiet_spark_logs(spark: SparkSession) -> None:
    """
    Reduz logs Spark/Kubernetes no console do CML.
    """
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


def nullif_empty(expr):
    return F.when(F.length(F.trim(expr.cast("string"))) > 0, F.trim(expr.cast("string")))


def first_non_empty_expr(*exprs):
    return F.coalesce(*[nullif_empty(e) for e in exprs])


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
    except Exception:
        out2 = out.select([F.col(c).cast("string").alias(c) for c in out.columns])
        pdf = out2.toPandas()

    pdf.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def elapsed_min(t0: float) -> float:
    return round((time.time() - t0) / 60, 2)
  
  
def write_recommendation(summary_counts: dict[str, Any], day_results: list[dict[str, Any]]) -> None:
    total = summary_counts.get("target_keys_total_180d", 0)
    matched_total = summary_counts.get("matched_mbk_total_180d", 0)
    coverage_total = summary_counts.get("coverage_total_180d", 0.0)

    processed_target = summary_counts.get("target_keys_exact_processed_target_dates", 0)
    processed_matched = summary_counts.get("matched_mbk_exact_processed_target_dates", 0)
    coverage_processed = summary_counts.get("coverage_exact_target_dates_processed", 0.0)

    processed_days = summary_counts.get("n_processed_lookup_dates", 0)

    field_matched = pd.DataFrame(summary_counts.get("field_coverage_matched_only", []))
    best_fields = []

    if not field_matched.empty:
        best_fields = (
            field_matched
            .sort_values("coverage_pct", ascending=False)
            .head(8)
            .to_dict(orient="records")
        )

    total_compact_rows = 0
    if day_results:
        for row in day_results:
            total_compact_rows += int(row.get("n_compact_rows") or 0)

    lines = [
        "# EXP-010E-R2B — MBK Compact Index 180d",
        "",
        f"Gerado em: `{now_iso()}`",
        "",
        "## Resultado executivo corrigido",
        "",
        f"- Target keys total 180d: `{total}`",
        f"- Dias/processamentos realizados neste run: `{processed_days}`",
        f"- Registros compactos MBK gerados neste run: `{total_compact_rows}`",
        f"- Matched MBK total contra 180d: `{matched_total}`",
        f"- Coverage total 180d parcial: `{coverage_total}`",
        f"- Target keys nos dias alvo processados: `{processed_target}`",
        f"- Matched MBK nos dias alvo processados: `{processed_matched}`",
        f"- Coverage nos dias alvo processados: `{coverage_processed}`",
        f"- Compact table: `{MBK_COMPACT_TABLE}`",
        f"- Hydration table: `{HYDRATION_TABLE}`",
        "",
        "## Como interpretar",
        "",
        "A cobertura total 180d ainda é parcial, porque o piloto processou apenas parte das partições MBK.",
        "",
        "Portanto, `coverage_total_180d` não deve ser usado para reprovar a estratégia enquanto o compact index ainda não cobrir todos os dias da janela.",
        "",
        "A métrica mais útil para o piloto é:",
        "",
        "```text",
        "coverage_exact_target_dates_processed",
        "```",
        "",
        "Além disso, o arquivo `03b_mbk_match_lag_distribution.csv` deve ser usado para verificar se `autdatref` bate com `dt_pix` ou se precisamos aumentar `DATE_PAD_DAYS`.",
        "",
        "## Campos MBK entre registros encontrados",
        "",
    ]

    if not best_fields:
        lines.append("- Nenhum campo MBK encontrado.")
    else:
        for row in best_fields:
            lines.append(
                f"- `{row['field']}`: coverage=`{row['coverage_pct']}` "
                f"({row['n_non_empty']}/{row['n_rows']})"
            )

    lines.extend(
        [
            "",
            "## Decisão preliminar",
            "",
        ]
    )

    if processed_days == 0:
        decision = "RELATORIO_SEM_DIAS_PROCESSADOS"
        text = "Nenhum dia foi processado; não há decisão operacional."
    elif coverage_processed >= 0.50:
        decision = "PILOTO_APROVADO_PROCESSAR_EM_BLOCOS"
        text = "A estratégia por partição diária é viável e a cobertura nos dias processados é boa. Prosseguir em blocos controlados."
    elif coverage_processed >= 0.20:
        decision = "PILOTO_APROVADO_COM_COBERTURA_PARCIAL"
        text = "A estratégia operacional é viável, mas a cobertura dos dias processados é parcial. Avaliar lag de datas e regex antes de processar muitos dias."
    else:
        decision = "PILOTO_OPERACIONAL_OK_MAS_COBERTURA_BAIXA"
        text = "O processamento diário é viável, mas a cobertura exata nos dias alvo ainda está baixa. Investigar defasagem autdatref vs dt_pix, tags XML e filtros."

    lines.extend(
        [
            f"`{decision}`",
            "",
            text,
            "",
            "## Próximo passo recomendado",
            "",
            "Antes de processar mais blocos, verificar:",
            "",
            "1. `03b_mbk_match_lag_distribution.csv` para decidir se `DATE_PAD_DAYS` deve ser 0, 1 ou 2;",
            "2. `04c_mbk_field_coverage_matched_only.csv` para medir qualidade dos campos entre matches;",
            "3. `03_mbk_coverage_by_day.csv` para ver cobertura real nos dias alvo processados;",
            "4. se aprovado, continuar em blocos de 12–15 dias com `OVERWRITE_TABLES=False` e `RESUME_SKIP_PROCESSED_DATES=True`.",
            "",
        ]
    )

    (OUTPUT_DIR / "07_recommendation.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# TABLE DDL
# ============================================================

def create_compact_table_if_needed(spark: SparkSession) -> None:
    if OVERWRITE_TABLES:
        print(f"[TABLE] DROP IF EXISTS {MBK_COMPACT_TABLE}")
        spark.sql(f"DROP TABLE IF EXISTS {MBK_COMPACT_TABLE}")

    if not table_exists(spark, MBK_COMPACT_TABLE):
        print(f"[TABLE] CREATE {MBK_COMPACT_TABLE}")
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {MBK_COMPACT_TABLE} (
                end_to_end_id STRING,
                autdathorini STRING,
                autcodret STRING,
                device_name STRING,
                app_version STRING,
                ip_address STRING,
                latencia_rede_ms INT,
                tempo_interacao_ms INT,
                tempo_processamento_host_ms INT,
                metodo_autenticacao STRING,
                session_id STRING,
                topaz_risk_score INT,
                topaz_transacao_rejeitada INT,
                is_agendamento_recorrente STRING,
                mbk_completeness_score INT,
                compact_created_at TIMESTAMP
            )
            PARTITIONED BY (autdatref STRING)
            STORED AS PARQUET
        """)


def create_hydration_table_if_needed(spark: SparkSession) -> None:
    if OVERWRITE_TABLES:
        print(f"[TABLE] DROP IF EXISTS {HYDRATION_TABLE}")
        spark.sql(f"DROP TABLE IF EXISTS {HYDRATION_TABLE}")

    if not table_exists(spark, HYDRATION_TABLE):
        print(f"[TABLE] CREATE {HYDRATION_TABLE}")
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {HYDRATION_TABLE} (
                cd_pix STRING,
                dt_pix TIMESTAMP,
                target_date DATE,
                cd_cpf_pagador STRING,
                cd_cpf_cnpj_recebedor STRING,
                vl_pix DOUBLE,
                end_to_end_id STRING,
                autdatref STRING,
                autdathorini STRING,
                autcodret STRING,
                device_name STRING,
                app_version STRING,
                ip_address STRING,
                latencia_rede_ms INT,
                tempo_interacao_ms INT,
                tempo_processamento_host_ms INT,
                metodo_autenticacao STRING,
                session_id STRING,
                topaz_risk_score INT,
                topaz_transacao_rejeitada INT,
                is_agendamento_recorrente STRING,
                mbk_completeness_score INT,
                hydrated_at TIMESTAMP
            )
            STORED AS PARQUET
        """)


def save_target_keys(spark: SparkSession, df) -> None:
    if OVERWRITE_TABLES:
        print(f"[TABLE] DROP IF EXISTS {TARGET_KEYS_TABLE}")
        spark.sql(f"DROP TABLE IF EXISTS {TARGET_KEYS_TABLE}")

    print(f"[TABLE] Salvando {TARGET_KEYS_TABLE}")
    df.write.mode("overwrite").format("parquet").saveAsTable(TARGET_KEYS_TABLE)
    print(f"[TABLE] OK {TARGET_KEYS_TABLE}")


def insert_compact_partition(spark: SparkSession, df, lookup_date: str) -> None:
    """
    Insere a partição diária usando partição estática Hive.

    Necessário porque o ambiente Hive está com:
      hive.exec.dynamic.partition.mode=strict

    Como processamos um autdatref por vez, a partição estática é mais segura.
    """
    tmp_view = "tmp_exp010e_r2b_compact_partition"

    out = df.select(*COMPACT_NONPART_COLS)
    out.createOrReplaceTempView(tmp_view)

    cols_sql = ",\n                ".join(COMPACT_NONPART_COLS)

    spark.sql(f"""
        INSERT INTO TABLE {MBK_COMPACT_TABLE}
        PARTITION (autdatref='{lookup_date}')
        SELECT
                {cols_sql}
        FROM {tmp_view}
    """)


def append_hydration(df) -> None:
    df.select(*HYDRATION_COLS).write.mode("append").insertInto(HYDRATION_TABLE)


# ============================================================
# TARGET KEYS AND PLAN
# ============================================================

def infer_window(spark: SparkSession) -> tuple[str, str]:
    row = (
        spark.table(MAF_HYDRATED_TABLE)
        .select(F.max(F.to_date("dt_pix")).alias("max_date"))
        .collect()[0]
    )

    max_date = row["max_date"]

    if max_date is None:
        raise RuntimeError("Não foi possível inferir max(dt_pix).")

    end_date = pd.to_datetime(str(max_date)).date()
    start_date = end_date - timedelta(days=WINDOW_DAYS - 1)

    return str(start_date), str(end_date)


def build_target_keys(spark: SparkSession, start_date: str, end_date: str):
    print("[1/7] Criando target keys 180d...")

    df = (
        spark.table(MAF_HYDRATED_TABLE)
        .select(
            F.trim(F.col("cd_pix").cast("string")).alias("cd_pix"),
            F.col("dt_pix").cast("timestamp").alias("dt_pix"),
            F.to_date("dt_pix").alias("target_date"),
            F.col("cd_cpf_pagador").cast("string").alias("cd_cpf_pagador"),
            F.col("cd_cpf_cnpj_recebedor").cast("string").alias("cd_cpf_cnpj_recebedor"),
            F.col("vl_pix").cast("double").alias("vl_pix"),
        )
        .filter(F.col("target_date") >= F.lit(start_date))
        .filter(F.col("target_date") <= F.lit(end_date))
        .filter(F.col("cd_pix").isNotNull())
        .filter(F.length(F.trim(F.col("cd_pix"))) > 0)
        .filter(F.col("cd_pix").rlike(r"^E[A-Za-z0-9]{20,}$"))
        .dropDuplicates(["cd_pix"])
    )

    save_target_keys(spark, df)

    return df.persist(StorageLevel.MEMORY_AND_DISK)


def get_available_mbk_partitions(spark: SparkSession) -> set[str]:
    print("[2/7] Lendo partições MBK disponíveis...")

    try:
        parts = spark.sql(f"SHOW PARTITIONS {MBK_TABLE}").toPandas()
        col = parts.columns[0]
        values = []
        for x in parts[col].astype(str).tolist():
            if "=" in x:
                values.append(x.split("=")[-1])
            else:
                values.append(x)
        return {v for v in values if len(v) == 10}
    except Exception as exc:
        print(f"[WARN] SHOW PARTITIONS falhou: {exc}")
        print("[WARN] Continuando sem filtro de partição disponível.")
        return set()


def get_processed_compact_dates(spark: SparkSession) -> set[str]:
    if not RESUME_SKIP_PROCESSED_DATES:
        return set()

    if not table_exists(spark, MBK_COMPACT_TABLE):
        return set()

    try:
        rows = spark.table(MBK_COMPACT_TABLE).select("autdatref").distinct().collect()
        return {r["autdatref"] for r in rows}
    except Exception as exc:
        print(f"[WARN] Falha ao ler datas processadas: {exc}")
        return set()


def build_day_plan(spark: SparkSession, target_keys, available_partitions: set[str], processed_dates: set[str]) -> list[dict[str, Any]]:
    print("[3/7] Montando plano diário...")

    day_rows = (
        target_keys
        .groupBy("target_date")
        .agg(
            F.count("*").alias("n_keys"),
            F.date_format(F.min("dt_pix"), "yyyy-MM-dd HH:mm:ss").alias("min_dt_pix"),
            F.date_format(F.max("dt_pix"), "yyyy-MM-dd HH:mm:ss").alias("max_dt_pix"),
        )
        .select(
            F.date_format(F.col("target_date"), "yyyy-MM-dd").alias("target_date"),
            F.col("n_keys").cast("int").alias("n_keys"),
            F.col("min_dt_pix"),
            F.col("max_dt_pix"),
        )
        .orderBy("target_date")
        .collect()
    )

    plan = []

    for row in day_rows:
        target_date = str(row["target_date"])

        lookup_dates = []
        base_dt = pd.to_datetime(target_date).date()

        for delta in range(-DATE_PAD_DAYS, DATE_PAD_DAYS + 1):
            d = str(base_dt + timedelta(days=delta))
            lookup_dates.append(d)

        lookup_dates = sorted(set(lookup_dates))

        for lookup_date in lookup_dates:
            if available_partitions and lookup_date not in available_partitions:
                continue

            if lookup_date in processed_dates:
                continue

            plan.append(
                {
                    "target_date": target_date,
                    "lookup_date": lookup_date,
                    "n_keys_target_date": int(row["n_keys"]),
                    "min_dt_pix": str(row["min_dt_pix"]),
                    "max_dt_pix": str(row["max_dt_pix"]),
                }
            )

    if MAX_DAYS_TO_PROCESS and MAX_DAYS_TO_PROCESS > 0:
        plan = plan[:MAX_DAYS_TO_PROCESS]

    write_pandas_csv(pd.DataFrame(plan), "01_day_plan.csv")

    return plan


# ============================================================
# MBK EXTRACTION
# ============================================================

def extract_e2e_expr():
    auttrn = F.col("auttrn").cast("string")

    return first_non_empty_expr(
        F.regexp_extract(auttrn, r"<BRB__IdFimAFimOriginalPix.*?>(.*?)</BRB__IdFimAFimOriginalPix>", 1),
        F.regexp_extract(auttrn, r"<FTN__IdFimAfimOriginalPix.*?>(.*?)</FTN__IdFimAfimOriginalPix>", 1),
        F.regexp_extract(auttrn, r"<FTN__IdFimAFimOriginalPix.*?>(.*?)</FTN__IdFimAFimOriginalPix>", 1),
        F.regexp_extract(auttrn, r"<idFimAfim.*?>(.*?)</idFimAfim>", 1),
        F.regexp_extract(auttrn, r"<idFimAFim.*?>(.*?)</idFimAFim>", 1),
        F.regexp_extract(auttrn, r"<endToEndId.*?>(.*?)</endToEndId>", 1),
        F.regexp_extract(auttrn, r"(E[0-9A-Za-z]{20,})", 1),
    )


def build_mbk_compact_for_date(spark: SparkSession, lookup_date: str):
    print(f"[MBK] Processando partição autdatref={lookup_date}")

    auttrn = F.col("auttrn").cast("string")

    df = (
        spark.table(MBK_TABLE)
        .filter(F.col("autdatref") == F.lit(lookup_date))
        .filter(F.col("auttrn").isNotNull())
        .filter(auttrn.like("%<transacao%"))
        .select(
            extract_e2e_expr().alias("end_to_end_id"),
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
        .filter(F.col("end_to_end_id").isNotNull())
        .filter(F.length(F.trim(F.col("end_to_end_id"))) > 0)
        .filter(F.col("end_to_end_id").rlike(r"^E[A-Za-z0-9]{20,}$"))
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

    df = df.withColumn("mbk_completeness_score", build_completeness_score(df, score_cols))

    w = (
        Window.partitionBy("end_to_end_id")
        .orderBy(
            F.col("mbk_completeness_score").desc(),
            F.col("autdathorini").desc_nulls_last(),
        )
    )

    dedup = (
        df
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
        .withColumn("compact_created_at", F.current_timestamp())
    )

    for c in COMPACT_NONPART_COLS + ["autdatref"]:
        if c not in dedup.columns:
            dedup = dedup.withColumn(c, F.lit(None).cast("string"))

    return dedup.select(*(COMPACT_NONPART_COLS + ["autdatref"]))


# ============================================================
# HYDRATION AND REPORTS
# ============================================================

def hydrate_maf_from_compact(spark: SparkSession, target_keys):
    print("[6/7] Hidratando MAF a partir do compact index...")

    compact = spark.table(MBK_COMPACT_TABLE)

    joined = (
        target_keys
        .join(
            compact,
            target_keys.cd_pix == compact.end_to_end_id,
            "left",
        )
        .withColumn("hydrated_at", F.current_timestamp())
    )

    score_col = F.coalesce(F.col("mbk_completeness_score"), F.lit(-1))

    w = (
        Window.partitionBy("cd_pix")
        .orderBy(
            score_col.desc(),
            F.col("autdatref").desc_nulls_last(),
            F.col("autdathorini").desc_nulls_last(),
        )
    )

    dedup = (
        joined
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    for c in HYDRATION_COLS:
        if c not in dedup.columns:
            dedup = dedup.withColumn(c, F.lit(None).cast("string"))

    dedup = dedup.select(*HYDRATION_COLS)

    print(f"[TABLE] Salvando {HYDRATION_TABLE}")
    dedup.write.mode("overwrite").format("parquet").saveAsTable(HYDRATION_TABLE)
    print(f"[TABLE] OK {HYDRATION_TABLE}")

    return dedup.persist(StorageLevel.MEMORY_AND_DISK)


def build_reports(spark: SparkSession, target_keys, hydration_df, day_results: list[dict[str, Any]]):
    print("[7/7] Gerando relatórios corrigidos...")

    target_count = safe_count(target_keys, "target_keys_total_180d")
    hydration_count = safe_count(hydration_df, "hydration_rows_total_180d")

    matched_df = hydration_df.filter(F.col("end_to_end_id").isNotNull())
    matched_count = safe_count(matched_df, "matched_mbk_total_180d")

    coverage_total_180d = round(matched_count / max(target_count, 1), 6)

    processed_target_dates = sorted(
        {
            str(x.get("target_date"))
            for x in day_results
            if x.get("target_date")
        }
    )

    processed_lookup_dates = sorted(
        {
            str(x.get("lookup_date"))
            for x in day_results
            if x.get("lookup_date")
        }
    )

    if processed_target_dates:
        processed_target_df = (
            target_keys
            .withColumn("target_date_str", F.date_format(F.col("target_date"), "yyyy-MM-dd"))
            .filter(F.col("target_date_str").isin(processed_target_dates))
            .drop("target_date_str")
        )

        processed_hydration_df = (
            hydration_df
            .withColumn("target_date_str", F.date_format(F.col("target_date"), "yyyy-MM-dd"))
            .filter(F.col("target_date_str").isin(processed_target_dates))
            .drop("target_date_str")
        )
    else:
        processed_target_df = target_keys.filter(F.lit(False))
        processed_hydration_df = hydration_df.filter(F.lit(False))

    processed_target_count = safe_count(processed_target_df, "target_keys_exact_processed_target_dates")
    processed_matched_count = safe_count(
        processed_hydration_df.filter(F.col("end_to_end_id").isNotNull()),
        "matched_mbk_exact_processed_target_dates",
    )

    coverage_exact_target_dates_processed = round(
        processed_matched_count / max(processed_target_count, 1),
        6,
    )

    # Cobertura por dia alvo, incluindo flag se aquele dia foi efetivamente processado no piloto.
    processed_dates_df = spark.createDataFrame(
        [(d,) for d in processed_target_dates],
        ["processed_target_date_str"],
    ) if processed_target_dates else spark.createDataFrame([], "processed_target_date_str string")

    coverage_by_day = (
        hydration_df
        .withColumn("target_date_str", F.date_format(F.col("target_date"), "yyyy-MM-dd"))
        .withColumn("matched_mbk", F.when(F.col("end_to_end_id").isNotNull(), F.lit(1)).otherwise(F.lit(0)))
        .groupBy("target_date_str")
        .agg(
            F.count("*").alias("n_target_keys"),
            F.sum("matched_mbk").alias("n_matched_mbk"),
        )
        .withColumn("coverage_pct", F.round(F.col("n_matched_mbk") / F.col("n_target_keys"), 6))
        .join(
            processed_dates_df,
            F.col("target_date_str") == F.col("processed_target_date_str"),
            "left",
        )
        .withColumn(
            "processed_in_current_run",
            F.when(F.col("processed_target_date_str").isNotNull(), F.lit(1)).otherwise(F.lit(0)),
        )
        .drop("processed_target_date_str")
        .orderBy("target_date_str")
    )

    write_limited_csv(coverage_by_day, "03_mbk_coverage_by_day.csv", limit=10000)

    # Distribuição de defasagem entre data da transação e autdatref MBK.
    # Ajuda a decidir se DATE_PAD_DAYS=0 é suficiente.
    lag_df = (
        matched_df
        .withColumn("autdatref_date", F.to_date(F.col("autdatref")))
        .withColumn("lag_days", F.datediff(F.col("autdatref_date"), F.col("target_date")))
        .groupBy("lag_days")
        .agg(
            F.count("*").alias("n_matches"),
            F.countDistinct("cd_pix").alias("n_distinct_cd_pix"),
        )
        .orderBy("lag_days")
    )

    write_limited_csv(lag_df, "03b_mbk_match_lag_distribution.csv", limit=10000)

    fields = [
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

    def collect_field_coverage(df_scope, denominator: int, scope_name: str) -> list[dict[str, Any]]:
        rows = []

        for c in fields:
            if c not in df_scope.columns:
                n_non_empty = 0
            else:
                n_non_empty = (
                    df_scope
                    .agg(F.sum(F.when(non_empty_col(c), F.lit(1)).otherwise(F.lit(0))).alias("n"))
                    .collect()[0]["n"]
                )
                n_non_empty = int(n_non_empty or 0)

            rows.append(
                {
                    "scope": scope_name,
                    "field": c,
                    "n_non_empty": n_non_empty,
                    "n_rows": denominator,
                    "coverage_pct": round(n_non_empty / max(denominator, 1), 6),
                }
            )

        return rows

    total_field_rows = collect_field_coverage(
        hydration_df,
        hydration_count,
        "total_180d",
    )

    processed_field_rows = collect_field_coverage(
        processed_hydration_df,
        safe_count(processed_hydration_df, "hydration_rows_exact_processed_target_dates"),
        "exact_processed_target_dates",
    )

    matched_field_rows = collect_field_coverage(
        matched_df,
        matched_count,
        "matched_only",
    )

    total_field_pdf = pd.DataFrame(total_field_rows).sort_values(["scope", "coverage_pct"], ascending=[True, False])
    processed_field_pdf = pd.DataFrame(processed_field_rows).sort_values(["scope", "coverage_pct"], ascending=[True, False])
    matched_field_pdf = pd.DataFrame(matched_field_rows).sort_values(["scope", "coverage_pct"], ascending=[True, False])

    # Mantém o nome antigo apontando para matched_only, que é o que mede qualidade dos campos entre registros achados.
    write_pandas_csv(matched_field_pdf, "04_mbk_field_coverage.csv")
    write_pandas_csv(total_field_pdf, "04a_mbk_field_coverage_total_180d.csv")
    write_pandas_csv(processed_field_pdf, "04b_mbk_field_coverage_exact_processed_target_dates.csv")
    write_pandas_csv(matched_field_pdf, "04c_mbk_field_coverage_matched_only.csv")

    write_limited_csv(
        matched_df,
        "05_mbk_matched_sample.csv",
        order_cols=["target_date"],
    )

    write_limited_csv(
        hydration_df.filter(F.col("end_to_end_id").isNull()),
        "06_mbk_unmatched_sample.csv",
        order_cols=["target_date"],
    )

    summary_counts = {
        "target_keys_total_180d": target_count,
        "hydration_rows_total_180d": hydration_count,
        "matched_mbk_total_180d": matched_count,
        "coverage_total_180d": coverage_total_180d,

        "processed_target_dates": processed_target_dates,
        "processed_lookup_dates": processed_lookup_dates,
        "n_processed_target_dates": len(processed_target_dates),
        "n_processed_lookup_dates": len(processed_lookup_dates),

        "target_keys_exact_processed_target_dates": processed_target_count,
        "matched_mbk_exact_processed_target_dates": processed_matched_count,
        "coverage_exact_target_dates_processed": coverage_exact_target_dates_processed,

        # Compatibilidade com versões anteriores do summary.
        "target_keys": target_count,
        "hydration_rows": hydration_count,
        "matched_mbk": matched_count,
        "coverage_pct": coverage_total_180d,

        "field_coverage_total_180d": total_field_rows,
        "field_coverage_exact_processed_target_dates": processed_field_rows,
        "field_coverage_matched_only": matched_field_rows,
    }

    return summary_counts


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    ensure_output_dir()

    spark = create_spark_session()
    quiet_spark_logs(spark)

    print("=" * 80)
    print("EXP-010E-R2B — MBK Compact Index 180d")
    print("=" * 80)
    print(f"MAF_HYDRATED_TABLE: {MAF_HYDRATED_TABLE}")
    print(f"MBK_TABLE: {MBK_TABLE}")
    print(f"TARGET_KEYS_TABLE: {TARGET_KEYS_TABLE}")
    print(f"MBK_COMPACT_TABLE: {MBK_COMPACT_TABLE}")
    print(f"HYDRATION_TABLE: {HYDRATION_TABLE}")
    print(f"WINDOW_DAYS: {WINDOW_DAYS}")
    print(f"DATE_PAD_DAYS: {DATE_PAD_DAYS}")
    print(f"MAX_DAYS_TO_PROCESS: {MAX_DAYS_TO_PROCESS}")
    print(f"TIME_BUDGET_MIN: {TIME_BUDGET_MIN}")
    print(f"OVERWRITE_TABLES: {OVERWRITE_TABLES}")
    print(f"RESUME_SKIP_PROCESSED_DATES: {RESUME_SKIP_PROCESSED_DATES}")
    print("=" * 80)

    target_keys = None
    hydration_df = None
    day_results: list[dict[str, Any]] = []
    status = "DONE"

    try:
        start_date, end_date = infer_window(spark)
        print(f"[WINDOW] {start_date} a {end_date}")

        target_keys = build_target_keys(spark, start_date, end_date)

        n_target = safe_count(target_keys, "target_keys_180d")

        if n_target == 0:
            raise RuntimeError("Nenhuma chave MAF encontrada na janela de 180 dias.")

        available_partitions = get_available_mbk_partitions(spark)

        create_compact_table_if_needed(spark)
        create_hydration_table_if_needed(spark)

        processed_dates = get_processed_compact_dates(spark)

        day_plan = build_day_plan(
            spark=spark,
            target_keys=target_keys,
            available_partitions=available_partitions,
            processed_dates=processed_dates,
        )

        write_json(OUTPUT_DIR / "00a_day_plan.json", day_plan)

        print(f"[PLAN] Dias a processar: {len(day_plan)}")

        for idx, day in enumerate(day_plan, start=1):
            current_elapsed = elapsed_min(t0)

            if current_elapsed >= TIME_BUDGET_MIN:
                print(f"[TIMEOUT] Orçamento de {TIME_BUDGET_MIN} min atingido antes do próximo dia.")
                status = "PARTIAL_TIMEOUT"
                break

            lookup_date = day["lookup_date"]
            target_date = day["target_date"]

            print("=" * 80)
            print(f"[DAY {idx}/{len(day_plan)}] target_date={target_date} lookup_date={lookup_date}")
            print(f"[TIME] elapsed={current_elapsed} min")
            print("=" * 80)

            day_start = time.time()

            compact_day = build_mbk_compact_for_date(spark, lookup_date).persist(StorageLevel.MEMORY_AND_DISK)

            try:
                # Materializa o resultado uma vez. O insert deve reaproveitar o cache.
                n_compact = safe_count(compact_day, f"compact_{lookup_date}")

                if n_compact > 0:
                    insert_compact_partition(spark, compact_day, lookup_date)
                    print(f"[DAY] Compact inserido para {lookup_date}")
                else:
                    print(f"[DAY] Sem registros compactos para {lookup_date}")

            finally:
                try:
                    compact_day.unpersist()
                except Exception as exc:
                    print(f"[WARN] Falha ao unpersist compact_day {lookup_date}: {exc}")

            day_results.append(
                {
                    "idx": idx,
                    "target_date": target_date,
                    "lookup_date": lookup_date,
                    "n_keys_target_date": int(day["n_keys_target_date"]),
                    "n_compact_rows": n_compact,
                    "elapsed_min_day": elapsed_min(day_start),
                    "elapsed_min_total": elapsed_min(t0),
                }
            )

            write_pandas_csv(pd.DataFrame(day_results), "02_day_results.csv")

        hydration_df = hydrate_maf_from_compact(spark, target_keys)
        summary_counts = build_reports(spark, target_keys, hydration_df, day_results)
        write_recommendation(summary_counts, day_results)

        run_summary = {
            "generated_at": now_iso(),
            "experiment": "EXP-010E-R2B",
            "status": status,
            "elapsed_min": elapsed_min(t0),
            "window_start_date": start_date,
            "window_end_date": end_date,
            "window_days": WINDOW_DAYS,
            "date_pad_days": DATE_PAD_DAYS,
            "max_days_to_process": MAX_DAYS_TO_PROCESS,
            "time_budget_min": TIME_BUDGET_MIN,
            "overwrite_tables": OVERWRITE_TABLES,
            "resume_skip_processed_dates": RESUME_SKIP_PROCESSED_DATES,
            "target_keys_table": TARGET_KEYS_TABLE,
            "mbk_compact_table": MBK_COMPACT_TABLE,
            "hydration_table": HYDRATION_TABLE,
            "n_target_keys_180d": n_target,
            "n_days_planned": len(day_plan),
            "n_days_processed": len(day_results),
            "summary_counts": summary_counts,
            "artifacts": [
              "00a_day_plan.json",
              "01_day_plan.csv",
              "02_day_results.csv",
              "03_mbk_coverage_by_day.csv",
              "03b_mbk_match_lag_distribution.csv",
              "04_mbk_field_coverage.csv",
              "04a_mbk_field_coverage_total_180d.csv",
              "04b_mbk_field_coverage_exact_processed_target_dates.csv",
              "04c_mbk_field_coverage_matched_only.csv",
              "05_mbk_matched_sample.csv",
              "06_mbk_unmatched_sample.csv",
              "07_recommendation.md",
          ],
        }

        write_json(OUTPUT_DIR / "00_run_summary.json", run_summary)

        print()
        print("=" * 80)
        print("[OK] EXP-010E-R2B finalizado")
        print(f"[OK] Status: {status}")
        print(f"[OK] Tempo total: {elapsed_min(t0)} min")
        print(f"[OK] Coverage: {summary_counts.get('coverage_pct')}")
        print(f"[OK] Artefatos: {OUTPUT_DIR}")
        print("=" * 80)

    finally:
        try:
            if target_keys is not None:
                target_keys.unpersist()
        except Exception as exc:
            print(f"[WARN] Falha ao unpersist target_keys: {exc}")

        try:
            if hydration_df is not None:
                hydration_df.unpersist()
        except Exception as exc:
            print(f"[WARN] Falha ao unpersist hydration_df: {exc}")


if __name__ == "__main__":
    main()