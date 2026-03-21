"""
01_ingestao_unificada_mobile_normais.py
========================================
POC Anomalia PIX — Ingestão Normais Trimestre V2.3 (PIPELINE MATERIALIZADO)

Alterações v2.3:
  ══════════════════════════════════════════════════════════════
  PROBLEMA RESOLVIDO: DAG monolítico causava 6h+ de execução
    porque o Spark tentava planejar/executar tudo de uma vez,
    incluindo o parsing de REGEX na MBK (centenas de milhões de rows).

  SOLUÇÃO: Pipeline em etapas materializadas via tabelas temporárias
    - Cada etapa pesada é salva como tabela Hive (Parquet)
    - Lida de volta com schema otimizado (sem recomputação)
    - Progresso visível etapa por etapa
    - DAG curto em cada etapa → planejamento rápido

  ETAPAS:
    1. Clientes → persist (pequeno, cabe em memória)
    2. PIX bruto → tabela tmp (milhões de rows, dedup pesada)
    3. Mobile MBK → tabela tmp (REGEX pesado, maior gargalo)
    4. JOIN PIX + Mobile + Cliente + skew filter → tabela tmp
    5. Pré-agregações por CPF → persist (1 row por CPF)
    6. Features por transação + merge → tabela final

  Mantém TODAS as colunas v2.1/v2.2 (mesma saída).
  ══════════════════════════════════════════════════════════════
"""

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel
import time


# =========================================================
# FLAGS DE CONTROLE
# =========================================================
DEBUG_COUNTS = False
SKEW_THRESHOLD = 5000
CLEANUP_TEMP_TABLES = True  # Limpar tabelas temporárias ao final


def create_spark_session():
    return (
        SparkSession.builder
        .appName("POC - PIX Anomalia Normais Trimestre V2.3")
        .config("spark.driver.memory", "8g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.executor.memory", "12g")
        .config("spark.executor.cores", "3")
        # Dynamic allocation
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.dynamicAllocation.minExecutors", "2")
        .config("spark.dynamicAllocation.initialExecutors", "4")
        .config("spark.dynamicAllocation.maxExecutors", "20")
        # AQE
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
        .config("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256m")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m")
        # Shuffle
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.default.parallelism", "200")
        # Broadcast conservador
        .config("spark.sql.autoBroadcastJoinThreshold", "10485760")  # 10MB
        .config("spark.sql.broadcastTimeout", "600")
        # Timeout
        .config("spark.network.timeout", "1200s")
        .config("spark.executor.heartbeatInterval", "60s")
        # Parquet
        .config("spark.sql.parquet.enableVectorizedReader", "true")
        .config("spark.sql.hive.convertMetastoreParquet", "false")
        # Memory overhead
        .config("spark.yarn.executor.memoryOverhead", "3072")
        .enableHiveSupport()
        .getOrCreate()
    )


def build_completeness_score(df, cols):
    expr = F.lit(0)
    for c in cols:
        if c in df.columns:
            expr = expr + F.when(F.col(c).isNotNull(), F.lit(1)).otherwise(F.lit(0))
    return expr


def safe_show(df, title, n=5):
    print(f"\n--- {title} ---")
    try:
        df.show(n, truncate=False)
    except Exception as e:
        print(f"Não foi possível exibir '{title}': {e}")


def save_temp(df, spark, table_name, partition_col=None):
    """Salva DataFrame como tabela Hive temporária e retorna lido de volta."""
    t0 = time.time()
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    writer = df.write.mode("overwrite").format("parquet")
    if partition_col:
        writer = writer.partitionBy(partition_col)
    writer.saveAsTable(table_name)

    count = spark.table(table_name).count()
    elapsed = round((time.time() - t0) / 60, 1)
    print(f"    ✅ {table_name}: {count:,} rows ({elapsed} min)")
    return spark.table(table_name), count


def main():
    spark = create_spark_session()

    output_table = "hmo_ml.tb_pix_anomalia_normais_trim_poc_v2"
    mbk_source_table = "landing_brb_oracle_mbk.aut"

    # Tabelas temporárias
    TMP_PIX = "hmo_ml.tmp_v23_pix_dedup"
    TMP_MOBILE = "hmo_ml.tmp_v23_mobile_dedup"
    TMP_BASE = "hmo_ml.tmp_v23_base_joined"

    t_start = time.time()

    print("=" * 80)
    print("PROCESSAMENTO UNIFICADO PIX + MOBILE V2.3 (PIPELINE MATERIALIZADO)")
    print("=" * 80)

    spark.sql(f"DROP TABLE IF EXISTS {output_table}")

    days_back = 90
    cutoff_date = spark.sql(
        f"SELECT cast(date_sub(current_date(), {days_back}) as string)"
    ).collect()[0][0]

    print(f"Data de corte: {cutoff_date}")
    print(f"Tabela destino: {output_table}")

    # =============================================================
    # ETAPA 1: CLIENTES (persist em memória — tabela de dimensão)
    # =============================================================
    print("\n" + "=" * 60)
    print("ETAPA 1/6: CLIENTES")
    print("=" * 60)

    t1 = time.time()

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
        df_cliente.withColumn(
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

    # Dedup clientes
    window_cli = Window.partitionBy("cd_cpf_pagador").orderBy(F.col("cd_cliente").desc())
    df_cliente = (
        df_cliente
        .withColumn("_rn", F.row_number().over(window_cli))
        .filter(F.col("_rn") == 1)
        .drop("_rn", "cd_cliente")
    )

    df_cliente = df_cliente.persist(StorageLevel.MEMORY_AND_DISK)
    total_clientes = df_cliente.count()
    print(f"    ✅ Clientes: {total_clientes:,} rows ({round((time.time() - t1) / 60, 1)} min)")

    # =============================================================
    # ETAPA 2: PIX BRUTO → TABELA TEMPORÁRIA
    # =============================================================
    print("\n" + "=" * 60)
    print("ETAPA 2/6: PIX BRUTO (dedup + salvar)")
    print("=" * 60)

    df_pix_raw = spark.sql("""
        SELECT
            trim(t.ds_id_pix) as cd_pix,
            LPAD(t.nr_cpf_cnpj_origem, 14, '0') as cd_cpf_pagador,
            LPAD(t.nr_cpf_cnpj_destino, 14, '0') as cd_cpf_cnpj_recebedor,
            t.vl_pix,
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
          AND t.dt_pix >= date_sub(current_date(), 90)
    """)

    df_pix_raw = (
        df_pix_raw
        .filter(F.col("cd_pix").isNotNull())
        .dropDuplicates(["cd_pix"])
    )

    df_pix, total_pix = save_temp(df_pix_raw, spark, TMP_PIX)

    # =============================================================
    # ETAPA 3: MOBILE MBK → TABELA TEMPORÁRIA
    # =============================================================
    print("\n" + "=" * 60)
    print("ETAPA 3/6: MOBILE MBK (regex + dedup + salvar)")
    print("=" * 60)

    df_mobile_raw = spark.sql(f"""
        SELECT
            autnsu AS nsu_transacao,
            autnsuaut AS nsu_autorizadora,
            ctanum AS nr_conta,

            trim(
                COALESCE(
                    REGEXP_EXTRACT(auttrn, '<BRB__IdFimAFimOriginalPix.*?>(.*?)</BRB__IdFimAFimOriginalPix>', 1),
                    REGEXP_EXTRACT(auttrn, '<FTN__IdFimAFimOriginalPix.*?>(.*?)</FTN__IdFimAFimOriginalPix>', 1),
                    REGEXP_EXTRACT(auttrn, '<idFimAFim.*?>(.*?)</idFimAFim>', 1)
                )
            ) AS end_to_end_id,

            autdathorini AS data_hora_inicio,
            autdatref AS data_referencia,
            ttrcod AS cd_tipo_transacao,
            autval AS valor_transacao,
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
                REGEXP_EXTRACT(auttrn, '<BRB__ResultadoConsultaScoreTopaz>(\\d+)</BRB__ResultadoConsultaScoreTopaz>', 1)
            ), '') AS INT) AS topaz_risk_score,

            CAST(NULLIF(REGEXP_EXTRACT(auttrn, '<BRB__TopazTransacaoRejeitada[^>]*>(.*?)</BRB__TopazTransacaoRejeitada>', 1), '') AS INT) AS topaz_transacao_rejeitada,

            REGEXP_EXTRACT(auttrn, '<BRB__IsAgendamentoRecorrenteForTopaz[^>]*>(.*?)</BRB__IsAgendamentoRecorrenteForTopaz>', 1) AS is_agendamento_recorrente

        FROM {mbk_source_table}
        WHERE autdatref >= '{cutoff_date}'
          AND auttrn LIKE '%<transacao%'
    """)

    # Filtrar + dedup mobile
    df_mobile = (
        df_mobile_raw
        .filter(F.col("end_to_end_id").isNotNull())
        .filter(F.length(F.trim(F.col("end_to_end_id"))) > 0)
    )

    mobile_score_cols = [
        "device_name", "app_version", "ip_address",
        "latencia_rede_ms", "tempo_interacao_ms",
        "tempo_processamento_host_ms", "metodo_autenticacao",
        "session_id", "cd_retorno", "topaz_risk_score",
        "topaz_transacao_rejeitada", "is_agendamento_recorrente",
    ]

    df_mobile = df_mobile.withColumn(
        "mobile_score",
        build_completeness_score(df_mobile, mobile_score_cols)
    )

    w_mobile = Window.partitionBy("end_to_end_id").orderBy(
        F.col("mobile_score").desc(),
        F.col("data_referencia").desc_nulls_last(),
        F.col("data_hora_inicio").desc_nulls_last()
    )

    df_mobile = (
        df_mobile
        .withColumn("rn_mobile", F.row_number().over(w_mobile))
        .filter(F.col("rn_mobile") == 1)
        .drop("rn_mobile", "mobile_score")
    )

    df_mobile, total_mobile = save_temp(df_mobile, spark, TMP_MOBILE)

    # =============================================================
    # ETAPA 4: JOIN PIX + MOBILE + CLIENTE + SKEW FILTER → TMP
    # =============================================================
    print("\n" + "=" * 60)
    print("ETAPA 4/6: JOIN + SKEW FILTER (salvar base enriquecida)")
    print("=" * 60)

    # Ler das tabelas tmp (DAG limpo, sem recomputação)
    df_pix = spark.table(TMP_PIX)
    df_mobile = spark.table(TMP_MOBILE)

    df_base = (
        df_pix
        .join(df_mobile, df_pix.cd_pix == df_mobile.end_to_end_id, "left")
        .join(df_cliente, on="cd_cpf_pagador", how="inner")
    )

    # Unicidade pós-join
    base_score_cols = [
        "device_name", "app_version", "ip_address",
        "latencia_rede_ms", "tempo_interacao_ms",
        "tempo_processamento_host_ms", "metodo_autenticacao",
        "session_id", "cd_retorno", "topaz_risk_score",
        "topaz_transacao_rejeitada", "is_agendamento_recorrente",
        "nr_idade", "qt_tempo_relacionamento_mes",
        "ds_sexo", "ds_estado_civil", "ds_segmento",
    ]

    df_base = df_base.withColumn(
        "base_score",
        build_completeness_score(df_base, base_score_cols)
    )

    w_base = Window.partitionBy("cd_pix").orderBy(
        F.col("base_score").desc(),
        F.col("dt_pix").desc_nulls_last()
    )

    df_base = (
        df_base
        .withColumn("rn_base", F.row_number().over(w_base))
        .filter(F.col("rn_base") == 1)
        .drop("rn_base", "base_score")
    )

    # Skew filter integrado (semi-join via subquery)
    df_cpf_validos = (
        df_base
        .groupBy("cd_cpf_pagador")
        .agg(F.count("cd_pix").alias("qt_pix_skew"))
        .filter(F.col("qt_pix_skew") <= SKEW_THRESHOLD)
        .select("cd_cpf_pagador")
    )

    df_base = df_base.join(df_cpf_validos, on="cd_cpf_pagador", how="inner")

    # Salvar base como tabela tmp — CORTA O DAG completamente
    df_base, total_base = save_temp(df_base, spark, TMP_BASE)

    # Liberar clientes da memória
    df_cliente.unpersist()

    # =============================================================
    # ETAPA 5: PRÉ-AGREGAÇÕES POR CPF
    # =============================================================
    print("\n" + "=" * 60)
    print("ETAPA 5/6: PRÉ-AGREGAÇÕES POR CPF")
    print("=" * 60)

    t5 = time.time()

    # Ler base da tabela tmp (DAG limpo)
    df_base = spark.table(TMP_BASE)

    # 5a. Contagem diária máxima por CPF
    df_daily_max = (
        df_base
        .groupBy("cd_cpf_pagador", "data_pix")
        .agg(F.count("cd_pix").alias("daily_count"))
        .groupBy("cd_cpf_pagador")
        .agg(F.max("daily_count").alias("qt_pix_dia_maximo_trimestre"))
    )

    # 5b. Métricas trimestrais por CPF
    df_cpf_stats = (
        df_base
        .groupBy("cd_cpf_pagador")
        .agg(
            F.count("cd_pix").alias("_qt_total_pix_trimestre"),
            F.percentile_approx("vl_pix", 0.5).alias("_vl_mediana_pix_trimestre"),
            F.stddev("vl_pix").alias("_vl_desvio_padrao_pix_trimestre"),
            F.countDistinct("device_name").alias("_qt_aparelhos_distintos_trimestre"),
            F.avg("latencia_rede_ms").alias("_vl_latencia_rede_media_trimestre"),
            F.avg("tempo_interacao_ms").alias("_vl_tempo_interacao_medio_trimestre"),
        )
    )

    # 5c. Merge
    df_cpf_agg = df_cpf_stats.join(df_daily_max, on="cd_cpf_pagador", how="left")
    df_cpf_agg = df_cpf_agg.persist(StorageLevel.MEMORY_AND_DISK)
    total_cpfs_agg = df_cpf_agg.count()
    print(f"    ✅ CPFs pré-agregados: {total_cpfs_agg:,} ({round((time.time() - t5) / 60, 1)} min)")

    # Broadcast condicional
    if total_cpfs_agg <= 500_000:
        print(f"    → Broadcast habilitado ({total_cpfs_agg:,} CPFs)")
        df_cpf_agg_join = F.broadcast(df_cpf_agg)
    else:
        print(f"    → SortMergeJoin ({total_cpfs_agg:,} CPFs)")
        df_cpf_agg_join = df_cpf_agg

    # =============================================================
    # ETAPA 6: FEATURES + MERGE + SAVE FINAL
    # =============================================================
    print("\n" + "=" * 60)
    print("ETAPA 6/6: FEATURES + MERGE + SAVE")
    print("=" * 60)

    t6 = time.time()

    # Relê base da tabela tmp
    df_base = spark.table(TMP_BASE)

    # Windows leves
    w_user_order = Window.partitionBy("cd_cpf_pagador").orderBy("dt_pix")
    w_receiver = Window.partitionBy("cd_cpf_pagador", "cd_cpf_cnpj_recebedor").orderBy("dt_pix")
    w_receiver_count = (
        Window.partitionBy("cd_cpf_pagador", "cd_cpf_cnpj_recebedor")
        .orderBy(F.col("dt_pix").cast("long"))
        .rangeBetween(-90 * 86400, 0)
    )

    df_features = (
        df_base
        .withColumn("dt_transacao_anterior", F.lag("dt_pix").over(w_user_order))
        .withColumn(
            "delta_pix_segundos",
            F.col("dt_pix").cast("long") - F.col("dt_transacao_anterior").cast("long"),
        )
        .withColumn(
            "qt_intervalo_transacao_minuto",
            F.coalesce(F.round(F.col("delta_pix_segundos") / 60, 4), F.lit(0.0)),
        )
        .withColumn(
            "tp_primeiro_envio_recebedor_trimestre",
            F.when(F.row_number().over(w_receiver) == 1, 1).otherwise(0),
        )
        .withColumn(
            "qt_envio_recebedor_trimestre",
            F.count("cd_pix").over(w_receiver_count),
        )
        .drop("dt_transacao_anterior", "delta_pix_segundos")
    )

    # JOIN com pré-agregações
    df_features = (
        df_features
        .join(df_cpf_agg_join, on="cd_cpf_pagador", how="left")
        .withColumnRenamed("_qt_total_pix_trimestre", "qt_total_pix_trimestre")
        .withColumnRenamed("_vl_mediana_pix_trimestre", "vl_mediana_pix_trimestre")
        .withColumnRenamed("_vl_desvio_padrao_pix_trimestre", "vl_desvio_padrao_pix_trimestre")
        .withColumnRenamed("_qt_aparelhos_distintos_trimestre", "qt_aparelhos_distintos_trimestre")
        .withColumnRenamed("_vl_latencia_rede_media_trimestre", "vl_latencia_rede_media_trimestre")
        .withColumnRenamed("_vl_tempo_interacao_medio_trimestre", "vl_tempo_interacao_medio_trimestre")
    )

    # Mediana e desvio de intervalos
    df_intervalo_stats = (
        df_features
        .filter(F.col("qt_intervalo_transacao_minuto") > 0)
        .groupBy("cd_cpf_pagador")
        .agg(
            F.percentile_approx("qt_intervalo_transacao_minuto", 0.5).alias("qt_intervalo_mediana_trimestre"),
            F.stddev("qt_intervalo_transacao_minuto").alias("qt_intervalo_desvio_padrao_trimestre"),
        )
    )

    if total_cpfs_agg <= 500_000:
        df_intervalo_stats = F.broadcast(df_intervalo_stats)

    df_features = df_features.join(df_intervalo_stats, on="cd_cpf_pagador", how="left")

    # Coalesces
    df_features = (
        df_features
        .withColumn("qt_aparelhos_distintos_trimestre", F.coalesce(F.col("qt_aparelhos_distintos_trimestre"), F.lit(0)))
        .withColumn("qt_pix_dia_maximo_trimestre", F.coalesce(F.col("qt_pix_dia_maximo_trimestre"), F.lit(0)))
        .withColumn("qt_intervalo_mediana_trimestre", F.coalesce(F.col("qt_intervalo_mediana_trimestre"), F.lit(0.0)))
        .withColumn("qt_intervalo_desvio_padrao_trimestre", F.coalesce(F.col("qt_intervalo_desvio_padrao_trimestre"), F.lit(0.0)))
    )

    # Seleção final
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
    ]

    df_final = df_features.select(*final_columns)
    df_final = df_final.dropDuplicates(["cd_pix"])
    df_final = df_final.withColumn("dt_carga", F.current_date())

    # SAVE
    print(f"    Salvando {output_table}...")
    df_final.write.mode("overwrite").format("parquet").saveAsTable(output_table)

    # Validação
    df_check = spark.table(output_table)
    total_final = df_check.count()
    print(f"    ✅ {output_table}: {total_final:,} rows ({round((time.time() - t6) / 60, 1)} min)")

    # Cobertura (1 único agg)
    print("\n    --- COBERTURA DOS CAMPOS ---")
    coverage_cols = [
        "ds_sexo", "ds_estado_civil", "ds_segmento",
        "vl_renda_cliente", "qt_dependentes",
        "tp_primeiro_envio_recebedor_trimestre",
        "qt_envio_recebedor_trimestre",
        "tempo_interacao_ms", "metodo_autenticacao",
        "is_agendamento_recorrente", "topaz_transacao_rejeitada",
    ]

    coverage_exprs = []
    for col_name in coverage_cols:
        if col_name in df_check.columns:
            coverage_exprs.append(
                F.sum(
                    F.when(
                        F.col(col_name).isNotNull()
                        & (F.col(col_name) != "Informação ausente")
                        & (F.col(col_name) != "")
                        & (F.col(col_name) != 0),
                        1,
                    ).otherwise(0)
                ).alias(f"cov_{col_name}")
            )

    if coverage_exprs:
        coverage_row = df_check.agg(*coverage_exprs).collect()[0]
        for col_name in coverage_cols:
            alias = f"cov_{col_name}"
            if alias in coverage_row.asDict():
                not_null = coverage_row[alias]
                pct = round((not_null / total_final) * 100, 2) if total_final > 0 else 0
                print(f"    {col_name}: {not_null:,}/{total_final:,} ({pct}%)")

    safe_show(df_check.limit(5), "Amostra do dataset final", n=5)

    # =============================================================
    # CLEANUP
    # =============================================================
    df_cpf_agg.unpersist()

    if CLEANUP_TEMP_TABLES:
        print("\n    Limpando tabelas temporárias...")
        for t in [TMP_PIX, TMP_MOBILE, TMP_BASE]:
            spark.sql(f"DROP TABLE IF EXISTS {t}")
            print(f"    ✗ {t} removida")

    total_elapsed = round((time.time() - t_start) / 60, 1)

    print("\n" + "=" * 80)
    print(f"CONCLUÍDO — V2.3 (PIPELINE MATERIALIZADO)")
    print(f"Total final: {total_final:,} rows")
    print(f"Tempo total: {total_elapsed} min")
    print("=" * 80)


if __name__ == "__main__":
    main()
