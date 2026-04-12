"""
02_ingestao_unificada_mobile_fraudes.py
========================================
POC Anomalia PIX — Ingestão Fraudes Trimestre V3.0 (LEAKAGE-FREE)

Alterações v3.0 (sobre v2.2):
  ══════════════════════════════════════════════════════════════
  MESMA CORREÇÃO do script de normais:
    Features trimestrais calculadas via ROLLING WINDOW
    com rangeBetween(-90d, -1s) em vez de groupBy CPF.

  DIFERENÇA DO SCRIPT DE NORMAIS:
    - Dataset menor (~centenas de fraudes) → rolling é rápido
    - Persist em memória suficiente (sem tabelas tmp)
    - Broadcast agressivo mantido

  Mantém TODAS as etapas e otimizações da v2.2.
  Mesma saída (mesmas colunas + is_fraud).
  ══════════════════════════════════════════════════════════════
"""

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel


# =========================================================
# FLAGS DE CONTROLE
# =========================================================
DEBUG_COUNTS = False

# Constante: 90 dias em segundos
NINETY_DAYS_SECONDS = 90 * 86400


def create_spark_session():
    return (
        SparkSession.builder
        .appName("POC - PIX Fraudes Trimestre Unificado V3.0 (Leakage-Free)")
        .config("spark.driver.memory", "6g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.executor.memory", "8g")
        .config("spark.executor.cores", "2")
        # Dynamic allocation
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.dynamicAllocation.minExecutors", "2")
        .config("spark.dynamicAllocation.initialExecutors", "2")
        .config("spark.dynamicAllocation.maxExecutors", "10")
        # AQE
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
        .config("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256m")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m")
        # Shuffle
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.default.parallelism", "64")
        # Broadcast — 100MB (fraudes = dataset pequeno)
        .config("spark.sql.autoBroadcastJoinThreshold", "104857600")
        .config("spark.sql.broadcastTimeout", "1200")
        # Timeout
        .config("spark.network.timeout", "1200s")
        .config("spark.executor.heartbeatInterval", "60s")
        # Parquet
        .config("spark.sql.parquet.enableVectorizedReader", "true")
        .config("spark.sql.hive.convertMetastoreParquet", "false")
        # Memory overhead
        .config("spark.yarn.executor.memoryOverhead", "2048")
        .enableHiveSupport()
        .getOrCreate()
    )


def build_completeness_score(df, cols):
    """Score de completude: conta campos não-nulos."""
    expr = F.lit(0)
    for c in cols:
        if c in df.columns:
            expr = expr + F.when(F.col(c).isNotNull(), F.lit(1)).otherwise(F.lit(0))
    return expr


def safe_show(df, title, n=10):
    print(f"\n--- {title} ---")
    try:
        df.show(n, truncate=False)
    except Exception as e:
        print(f"Não foi possível exibir amostra de '{title}': {e}")


def main():
    spark = create_spark_session()

    output_table = "hmo_ml.tb_pix_anomalia_fraudes_trim_poc_v3"
    mbk_source_table = "landing_brb_oracle_mbk.aut"

    days_back = 90
    cutoff_date = spark.sql(
        f"SELECT cast(date_sub(current_date(), {days_back}) as string)"
    ).collect()[0][0]

    print("=" * 80)
    print("PROCESSAMENTO UNIFICADO PIX FRAUDES + MOBILE V3.0 (LEAKAGE-FREE)")
    print(f"Data de corte: {cutoff_date}")
    print(f"Tabela destino: {output_table}")
    print(f"Debug counts: {DEBUG_COUNTS}")
    print("=" * 80)

    spark.sql(f"DROP TABLE IF EXISTS {output_table}")

    # =========================================================
    # 1. CHAVES DE FRAUDE (idêntico à v2.2)
    # =========================================================
    print("\n[1/8] Gerando chaves de fraude PIX...")

    df_fraud_keys = spark.sql("""
        SELECT DISTINCT
            TRIM(B.idpix) AS cd_pix
        FROM landing_brb_oracle_gesei.protocolo_enviado A
        JOIN landing_brb_oracle_gesei.protocolo_enviado_transacao B
            ON A.COD = B.ID
        JOIN landing_brb_oracle_maf.tb_infracao_pix C
            ON C.cd_identificador_fim_transacao = B.idpix
        WHERE B.data_trans >= date_sub(CURRENT_TIMESTAMP(), 90)
          AND B.desc_trans = 'PIX'
          AND (
                C.cd_resultado_analise_infracao = 1
                OR (C.cd_resultado_analise_infracao = 2 AND B.valor > 500)
          )
          AND NOT UPPER(A.DESCRICAO) LIKE '%TRIANG%'
          AND NOT UPPER(A.oc_vinculada) LIKE '%TRIANG%'
          AND B.idpix IS NOT NULL
          AND LENGTH(TRIM(B.idpix)) > 0
    """).cache()

    fraud_count = df_fraud_keys.count()
    print(f"    Total chaves de fraude: {fraud_count}")

    # =========================================================
    # 2. CLIENTES (idêntico à v2.2)
    # =========================================================
    print("\n[2/8] Carregando perfil de clientes...")

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

    window_cli = Window.partitionBy("cd_cpf_pagador").orderBy(F.col("cd_cliente").desc())
    df_cliente = (
        df_cliente
        .withColumn("_rn", F.row_number().over(window_cli))
        .filter(F.col("_rn") == 1)
        .drop("_rn", "cd_cliente")
    )

    df_cliente = df_cliente.persist(StorageLevel.MEMORY_AND_DISK)

    if DEBUG_COUNTS:
        print(f"    → Clientes únicos: {df_cliente.count()}")
    else:
        safe_show(df_cliente.limit(3), "Amostra clientes", n=3)

    # =========================================================
    # 3. BASE PIX FRAUDADA (idêntico à v2.2)
    # =========================================================
    print("\n[3/8] Gerando base PIX fraudada...")

    df_pix_base = spark.sql("""
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
                WHEN t.ds_chave_pix LIKE '%@%' THEN 'EMAIL'
                WHEN regexp_like(t.ds_chave_pix, '^[0-9]+$') AND length(t.ds_chave_pix) >= 11 THEN 'DOCUMENTO/TELEFONE'
                WHEN length(t.ds_chave_pix) >= 32 THEN 'CHAVE ALEATORIA'
                ELSE 'OUTROS'
            END as ds_tipo_chave
        FROM landing_brb_oracle_blk.tb_extrato_pix t
        INNER JOIN landing_brb_oracle_blk.tb_registro_pix r
            ON t.ds_id_pix = r.ds_id_pix
        WHERE cast(t.cd_ispb_origem as int) = 208
          AND t.dt_pix >= date_sub(current_date(), 90)
    """)

    df_pix_base = (
        df_pix_base
        .join(F.broadcast(df_fraud_keys), on="cd_pix", how="inner")
        .dropDuplicates(["cd_pix"])
        .withColumn("is_fraud", F.lit(1))
    )

    df_pix_base = df_pix_base.persist(StorageLevel.MEMORY_AND_DISK)
    pix_fraud_count = df_pix_base.count()
    print(f"    Total PIX fraude (dedup): {pix_fraud_count}")

    # =========================================================
    # 4. MOBILE REDUZIDO (idêntico à v2.2)
    # =========================================================
    print("\n[4/8] Extraindo mobile (somente fraudes)...")

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
                REGEXP_EXTRACT(auttrn, '<BRB__ResultadoConsultaScoreTopaz>(\\d+)</BRB__ResultadoConsultaScoreTopaz>', 1)
            ), '') AS INT) AS topaz_risk_score,
            CAST(NULLIF(REGEXP_EXTRACT(auttrn, '<BRB__TopazTransacaoRejeitada[^>]*>(.*?)</BRB__TopazTransacaoRejeitada>', 1), '') AS INT) AS topaz_transacao_rejeitada,
            REGEXP_EXTRACT(auttrn, '<BRB__IsAgendamentoRecorrenteForTopaz[^>]*>(.*?)</BRB__IsAgendamentoRecorrenteForTopaz>', 1) AS is_agendamento_recorrente
        FROM {mbk_source_table}
        WHERE autdatref >= '{cutoff_date}'
          AND auttrn LIKE '%<transacao%'
    """)

    df_mobile = (
        df_mobile
        .filter(F.col("end_to_end_id").isNotNull())
        .filter(F.length(F.trim(F.col("end_to_end_id"))) > 0)
        .join(
            F.broadcast(df_fraud_keys.select(F.col("cd_pix").alias("_ref"))),
            F.col("end_to_end_id") == F.col("_ref"),
            "inner",
        )
        .drop("_ref")
    )

    mobile_score_cols = [
        "device_name", "app_version", "ip_address", "latencia_rede_ms",
        "tempo_interacao_ms", "tempo_processamento_host_ms", "metodo_autenticacao",
        "session_id", "cd_retorno", "topaz_risk_score",
        "topaz_transacao_rejeitada", "is_agendamento_recorrente",
    ]

    df_mobile = df_mobile.withColumn(
        "mobile_score",
        build_completeness_score(df_mobile, mobile_score_cols),
    )

    w_mobile = Window.partitionBy("end_to_end_id").orderBy(
        F.col("mobile_score").desc(),
        F.col("data_referencia").desc_nulls_last(),
        F.col("data_hora_inicio").desc_nulls_last(),
    )

    df_mobile = (
        df_mobile
        .withColumn("rn_mobile", F.row_number().over(w_mobile))
        .filter(F.col("rn_mobile") == 1)
        .drop("rn_mobile", "mobile_score")
    )

    df_mobile = df_mobile.persist(StorageLevel.MEMORY_AND_DISK)

    if DEBUG_COUNTS:
        print(f"    Total mobile fraude (dedup): {df_mobile.count()}")
    else:
        print("    Mobile fraude carregado e deduplicado")

    # =========================================================
    # 5. JOIN PIX + MOBILE + CLIENTE (idêntico à v2.2)
    # =========================================================
    print("\n[5/8] JOIN PIX + Mobile + Cliente...")

    df_base = (
        df_pix_base
        .join(df_mobile, df_pix_base.cd_pix == df_mobile.end_to_end_id, "left")
        .join(F.broadcast(df_cliente), on="cd_cpf_pagador", how="left")
    )

    base_score_cols = [
        "device_name", "app_version", "ip_address", "latencia_rede_ms",
        "tempo_interacao_ms", "tempo_processamento_host_ms", "metodo_autenticacao",
        "session_id", "cd_retorno", "topaz_risk_score", "topaz_transacao_rejeitada",
        "is_agendamento_recorrente",
        "nr_idade", "qt_tempo_relacionamento_mes", "ds_segmento",
        "ds_sexo", "ds_estado_civil",
    ]

    df_base = df_base.withColumn(
        "base_score",
        build_completeness_score(df_base, base_score_cols),
    )

    w_base = Window.partitionBy("cd_pix").orderBy(
        F.col("base_score").desc(),
        F.col("dt_pix").desc_nulls_last(),
    )

    df_base = (
        df_base
        .withColumn("rn_base", F.row_number().over(w_base))
        .filter(F.col("rn_base") == 1)
        .drop("rn_base", "base_score")
    )

    df_base = df_base.persist(StorageLevel.MEMORY_AND_DISK)

    df_pix_base.unpersist()
    df_mobile.unpersist()

    base_count = df_base.count()
    print(f"    Total base enriquecida: {base_count}")

    # =========================================================
    # 6. FEATURES ROLLING (LEAKAGE-FREE) — SUBSTITUI groupBy
    # =========================================================
    # ══════════════════════════════════════════════════════════════
    # MUDANÇA PRINCIPAL v3.0: Mesma lógica do script de normais.
    # Para fraudes, o dataset é pequeno (~centenas de rows),
    # então rolling windows são rápidas — sem necessidade de
    # tabelas temporárias para daily_counts.
    # ══════════════════════════════════════════════════════════════
    print("\n[6/8] Calculando features rolling (leakage-free)...")

    df_base = df_base.withColumn("dt_pix_long", F.col("dt_pix").cast("long"))

    # Window: 90 dias antes, excluindo tx atual
    w_rolling_90d = (
        Window.partitionBy("cd_cpf_pagador")
        .orderBy("dt_pix_long")
        .rangeBetween(-NINETY_DAYS_SECONDS, -1)
    )

    df_features = (
        df_base
        .withColumn(
            "qt_total_pix_trimestre",
            F.count("cd_pix").over(w_rolling_90d)
        )
        .withColumn(
            "vl_mediana_pix_trimestre",
            F.percentile_approx("vl_pix", 0.5).over(w_rolling_90d)
        )
        .withColumn(
            "vl_desvio_padrao_pix_trimestre",
            F.stddev("vl_pix").over(w_rolling_90d)
        )
        .withColumn(
            "qt_aparelhos_distintos_trimestre",
            F.size(F.collect_set("device_name").over(w_rolling_90d))
        )
        .withColumn(
            "vl_latencia_rede_media_trimestre",
            F.avg("latencia_rede_ms").over(w_rolling_90d)
        )
        .withColumn(
            "vl_tempo_interacao_medio_trimestre",
            F.avg("tempo_interacao_ms").over(w_rolling_90d)
        )
    )

    # qt_pix_dia_maximo_trimestre via self-join
    # (dataset pequeno — sem materialização intermediária necessária)
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
            how="inner"
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
        .withColumn(
            "qt_pix_dia_maximo_trimestre",
            F.coalesce(F.col("qt_pix_dia_maximo_trimestre"), F.lit(0))
        )
    )

    # =========================================================
    # 7. FEATURES POR TRANSAÇÃO (intervalos + primeiro envio)
    # =========================================================
    print("\n[7/8] Calculando features por transação...")

    w_user_order = Window.partitionBy("cd_cpf_pagador").orderBy("dt_pix")
    w_receiver = (
        Window.partitionBy("cd_cpf_pagador", "cd_cpf_cnpj_recebedor")
        .orderBy("dt_pix")
    )
    w_receiver_count = (
        Window.partitionBy("cd_cpf_pagador", "cd_cpf_cnpj_recebedor")
        .orderBy(F.col("dt_pix").cast("long"))
        .rangeBetween(-NINETY_DAYS_SECONDS, 0)
    )

    df_features = (
        df_features
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

    # Mediana e desvio de intervalos rolling
    w_rolling_intervalo = (
        Window.partitionBy("cd_cpf_pagador")
        .orderBy("dt_pix_long")
        .rangeBetween(-NINETY_DAYS_SECONDS, -1)
    )

    df_features = (
        df_features
        .withColumn(
            "qt_intervalo_mediana_trimestre",
            F.coalesce(
                F.percentile_approx("qt_intervalo_transacao_minuto", 0.5)
                .over(w_rolling_intervalo),
                F.lit(0.0)
            )
        )
        .withColumn(
            "qt_intervalo_desvio_padrao_trimestre",
            F.coalesce(
                F.stddev("qt_intervalo_transacao_minuto")
                .over(w_rolling_intervalo),
                F.lit(0.0)
            )
        )
    )

    # Coalesces
    df_features = (
        df_features
        .withColumn("qt_aparelhos_distintos_trimestre",
                     F.coalesce(F.col("qt_aparelhos_distintos_trimestre"), F.lit(0)))
        .withColumn("qt_pix_dia_maximo_trimestre",
                     F.coalesce(F.col("qt_pix_dia_maximo_trimestre"), F.lit(0)))
        .withColumn("vl_mediana_pix_trimestre",
                     F.coalesce(F.col("vl_mediana_pix_trimestre"), F.lit(0.0)))
        .withColumn("vl_desvio_padrao_pix_trimestre",
                     F.coalesce(F.col("vl_desvio_padrao_pix_trimestre"), F.lit(0.0)))
        .withColumn("vl_latencia_rede_media_trimestre",
                     F.coalesce(F.col("vl_latencia_rede_media_trimestre"), F.lit(0.0)))
        .withColumn("vl_tempo_interacao_medio_trimestre",
                     F.coalesce(F.col("vl_tempo_interacao_medio_trimestre"), F.lit(0.0)))
        .withColumn("qt_total_pix_trimestre",
                     F.coalesce(F.col("qt_total_pix_trimestre"), F.lit(0)))
    )

    # =========================================================
    # 8. SELEÇÃO FINAL + DEDUP + SAVE
    # =========================================================
    print("\n[8/8] Seleção final, deduplicação e salvamento...")

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
    ]

    df_final = df_features.select(*final_columns)
    df_final = df_final.dropDuplicates(["cd_pix"])
    df_final = df_final.withColumn("dt_carga", F.current_date())

    # Save
    df_final.write.mode("overwrite").format("parquet").saveAsTable(output_table)
    print(f"    ✅ Tabela {output_table} salva com sucesso!")

    # =========================================================
    # VALIDAÇÃO + COBERTURA
    # =========================================================
    print("\n    --- VALIDAÇÃO FINAL ---")

    df_check = spark.table(output_table)
    total_final = df_check.count()
    print(f"    Total final: {total_final}")

    # ── VALIDAÇÃO DE LEAKAGE: qt_total deve CRESCER por CPF ──
    print("\n    --- VALIDAÇÃO LEAKAGE-FREE (amostra) ---")
    print("    Para um CPF com múltiplas tx, qt_total deve CRESCER:")
    df_check_sample = (
        df_check
        .groupBy("cd_cpf_pagador")
        .agg(F.count("*").alias("n_tx"))
        .filter(F.col("n_tx") >= 3)
        .limit(1)
    )
    sample_cpf_rows = df_check_sample.collect()
    if sample_cpf_rows:
        sample_cpf = sample_cpf_rows[0]["cd_cpf_pagador"]
        (
            df_check
            .filter(F.col("cd_cpf_pagador") == sample_cpf)
            .select("cd_pix", "dt_pix", "vl_pix",
                    "qt_total_pix_trimestre", "vl_mediana_pix_trimestre")
            .orderBy("dt_pix")
            .show(20, truncate=False)
        )
    else:
        print("    (Nenhum CPF com 3+ tx encontrado para amostra)")

    # Cobertura
    coverage_cols = [
        "ds_sexo", "ds_estado_civil", "ds_segmento",
        "vl_renda_cliente", "qt_dependentes",
        "tp_primeiro_envio_recebedor_trimestre",
        "qt_envio_recebedor_trimestre",
        "tempo_interacao_ms", "metodo_autenticacao",
        "is_agendamento_recorrente", "topaz_transacao_rejeitada",
    ]

    print("\n    --- COBERTURA DOS CAMPOS v3.0 (FRAUDES) ---")

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
                print(f"    {col_name}: {not_null}/{total_final} ({pct}%)")

    safe_show(df_check.limit(5), "Amostra do dataset final de fraudes", n=5)

    # Cleanup
    df_base.unpersist()
    df_cliente.unpersist()
    df_fraud_keys.unpersist()

    print("\n" + "=" * 80)
    print("CONCLUÍDO — V3.0 (LEAKAGE-FREE)")
    print(f"Total fraudes: {total_final}")
    print("=" * 80)


if __name__ == "__main__":
    main()
