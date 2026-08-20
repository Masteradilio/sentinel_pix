from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel


def create_spark_session():
    return (
        SparkSession.builder
        .appName("POC - PIX Anomalia Normais Trimestre Unificado V2")
        .config("spark.driver.memory", "8g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.executor.memory", "12g")
        .config("spark.executor.cores", "3")
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.dynamicAllocation.minExecutors", "2")
        .config("spark.dynamicAllocation.initialExecutors", "4")
        .config("spark.dynamicAllocation.maxExecutors", "20")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.shuffle.partitions", "300")
        .config("spark.default.parallelism", "300")
        .config("spark.sql.autoBroadcastJoinThreshold", "52428800")  # 50MB
        .config("spark.sql.broadcastTimeout", "1200")
        .config("spark.network.timeout", "1200s")
        .config("spark.executor.heartbeatInterval", "60s")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.hive.convertMetastoreParquet", "false")
        .config("spark.yarn.executor.memoryOverhead", "3072")
        .enableHiveSupport()
        .getOrCreate()
    )


def build_completeness_score(df, cols):
    expr = None
    for c in cols:
        if c in df.columns:
            term = F.when(F.col(c).isNotNull(), F.lit(1)).otherwise(F.lit(0))
            expr = term if expr is None else expr + term
    if expr is None:
        expr = F.lit(0)
    return expr


def safe_show(df, title, n=10):
    print(f"\n--- {title} ---")
    try:
        df.show(n, truncate=False)
    except Exception as e:
        print(f"Não foi possível exibir amostra de '{title}': {e}")


def main():
    spark = create_spark_session()

    output_table = "hmo_ml.tb_pix_anomalia_normais_trim_poc_v2"
    mbk_source_table = "landing_brb_oracle_mbk.aut"

    print(f"--- Iniciando POC Normais Trimestre UNIFICADO V2 (Destino: {output_table}) ---")
    spark.sql(f"DROP TABLE IF EXISTS {output_table}")

    days_back = 90
    cutoff_date = spark.sql(
        f"SELECT cast(date_sub(current_date(), {days_back}) as string)"
    ).collect()[0][0]

    print("=" * 80)
    print("PROCESSAMENTO UNIFICADO PIX + MOBILE V2")
    print(f"Data de corte: {cutoff_date}")
    print(f"Tabela destino: {output_table}")
    print("=" * 80)

    # =========================================================
    # 1. CLIENTES
    # =========================================================
    print("1. Carregando perfil de clientes...")

    df_cliente = spark.sql("""
        SELECT
            c.x0100_cltcod as cd_cliente,
            concat(
                LPAD(CAST(cast(X0100_CLTCGC as BIGINT) AS STRING), 12, '0'),
                LPAD(CAST(cast(X0100_CLTCGCDIG as BIGINT) AS STRING), 2, '0')
            ) AS cd_cpf_pagador,
            COALESCE(trim(segmento.ds_segmento), 'Informação ausente') as ds_segmento,
            COALESCE(
                cast(
                    datediff(
                        current_date(),
                        cast(
                            date_format(
                                (
                                    concat(
                                        cast(substr(cast(X0100_CLTDATNAS AS INT),1,4) as STRING),
                                        '-',
                                        cast(substr(cast(X0100_CLTDATNAS AS INT),5,2) as STRING),
                                        '-',
                                        cast(substr(cast(X0100_CLTDATNAS AS INT),7,2) as STRING)
                                    )
                                ),
                                'yyyy-MM-dd'
                            ) as date
                        )
                    ) / 365.25 as int
                ),
                0
            ) as nr_idade,
            c.X0100_CLTDATPCAD as dt_cadastro_raw
        FROM landing_brb_db2_aox.aoxb01 c
        LEFT JOIN (
            SELECT cd_segmento, ds_segmento
            FROM (
                SELECT
                    trim(a0100_segcodsgm) cd_segmento,
                    a0100_segdessgm ds_segmento,
                    rank() OVER (
                        PARTITION BY trim(a0100_segcodsgm)
                        ORDER BY A0100_HDRDATA, A0100_HDRHORA DESC
                    ) rank
                FROM landing_brb_db2_dna.dnab01
            ) rk
            WHERE rank = 1
        ) segmento
            ON trim(X0100_SGMCODSEG) = trim(segmento.cd_segmento)
    """)

    df_cliente = (
        df_cliente.withColumn(
            "dt_inicio_relacionamento",
            F.to_date(
                F.concat(
                    F.substring(F.col("dt_cadastro_raw").cast("string"), 1, 4), F.lit("-"),
                    F.substring(F.col("dt_cadastro_raw").cast("string"), 5, 2), F.lit("-"),
                    F.substring(F.col("dt_cadastro_raw").cast("string"), 7, 2)
                )
            )
        )
        .withColumn(
            "qt_tempo_relacionamento_mes",
            F.coalesce(
                F.round(F.months_between(F.current_date(), F.col("dt_inicio_relacionamento")), 4),
                F.lit(0)
            )
        )
        .drop("dt_cadastro_raw")
    )

    print("1.1 Deduplicando clientes por cd_cpf_pagador...")
    cliente_score_cols = [
        "cd_cliente", "ds_segmento", "nr_idade",
        "dt_inicio_relacionamento", "qt_tempo_relacionamento_mes"
    ]
    df_cliente = df_cliente.withColumn(
        "cliente_score",
        build_completeness_score(df_cliente, cliente_score_cols)
    )

    w_cliente = Window.partitionBy("cd_cpf_pagador").orderBy(
        F.col("cliente_score").desc(),
        F.col("qt_tempo_relacionamento_mes").desc_nulls_last(),
        F.col("nr_idade").desc_nulls_last()
    )

    df_cliente = (
        df_cliente
        .withColumn("rn_cliente", F.row_number().over(w_cliente))
        .filter(F.col("rn_cliente") == 1)
        .drop("rn_cliente", "cliente_score")
    )

    # =========================================================
    # 2. PIX BRUTO
    # =========================================================
    print("2. Carregando PIX bruto trimestral...")

    df_pix_raw = spark.sql("""
        SELECT
            trim(t.ds_id_pix) as cd_pix,
            t.autnsublk,
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

    df_pix_raw = df_pix_raw.filter(F.col("cd_pix").isNotNull())
    print("2.1 Deduplicando PIX bruto por cd_pix...")
    df_pix_raw = df_pix_raw.dropDuplicates(["cd_pix"])

    total_pix = df_pix_raw.count()
    total_pix_distinct = df_pix_raw.select("cd_pix").distinct().count()

    print(f"Total PIX bruto: {total_pix}")
    print(f"Total PIX distintos: {total_pix_distinct}")

    # =========================================================
    # 3. MOBILE DIRETO DA MBK
    # =========================================================
    print("3. Extraindo mobile diretamente da MBK...")

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
            CAST(NULLIF(REGEXP_EXTRACT(auttrn, '<BRB__TopazTransacaoHabilitada[^>]*>(.*?)</BRB__TopazTransacaoHabilitada>', 1), '') AS INT) AS topaz_transacao_habilitada,

            REGEXP_EXTRACT(auttrn, '<BRB__IsAgendamentoRecorrenteForTopaz[^>]*>(.*?)</BRB__IsAgendamentoRecorrenteForTopaz>', 1) AS is_agendamento_recorrente,
            REGEXP_EXTRACT(auttrn, '<BRB__SyncIdTopaz[^>]*>(.*?)</BRB__SyncIdTopaz>', 1) AS topaz_sync_id

        FROM {mbk_source_table}
        WHERE autdatref >= '{cutoff_date}'
          AND auttrn LIKE '%<transacao%'
    """)

    total_mobile_raw = df_mobile_raw.count()
    print(f"Total mobile bruto extraído: {total_mobile_raw}")

    df_mobile = (
        df_mobile_raw
        .filter(F.col("end_to_end_id").isNotNull())
        .filter(F.length(F.trim(F.col("end_to_end_id"))) > 0)
    )

    total_mobile_with_key = df_mobile.count()
    total_mobile_key_distinct = df_mobile.select("end_to_end_id").distinct().count()

    print(f"Total mobile com end_to_end_id válido: {total_mobile_with_key}")
    print(f"Total end_to_end_id distintos no mobile: {total_mobile_key_distinct}")

    print("3.1 Deduplicando mobile por end_to_end_id...")

    mobile_score_cols = [
        "device_name",
        "app_version",
        "ip_address",
        "latencia_rede_ms",
        "tempo_interacao_ms",
        "tempo_processamento_host_ms",
        "metodo_autenticacao",
        "session_id",
        "cd_retorno",
        "topaz_risk_score",
        "topaz_transacao_rejeitada",
        "topaz_transacao_habilitada",
        "is_agendamento_recorrente",
        "topaz_sync_id",
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

    total_mobile_dedup = df_mobile.count()
    print(f"Total mobile após deduplicação: {total_mobile_dedup}")

    # =========================================================
    # 3.2 DIAGNÓSTICO DE CHAVES
    # =========================================================
    print("3.2 Diagnóstico de chaves PIX x mobile...")

    df_pix_keys = df_pix_raw.select("cd_pix").distinct()
    df_mobile_keys = df_mobile.select("end_to_end_id").distinct()

    total_intersection = (
        df_pix_keys.alias("p")
        .join(df_mobile_keys.alias("m"), F.col("p.cd_pix") == F.col("m.end_to_end_id"), "inner")
        .count()
    )

    print(f"Interseção exata de chaves PIX x mobile: {total_intersection}")

    safe_show(
        df_pix_raw
        .select("cd_pix", F.length("cd_pix").alias("len_cd_pix"))
        .groupBy("len_cd_pix")
        .count()
        .orderBy("len_cd_pix"),
        "Distribuição de tamanho de cd_pix"
    )

    safe_show(
        df_mobile
        .select("end_to_end_id", F.length("end_to_end_id").alias("len_end_to_end_id"))
        .groupBy("len_end_to_end_id")
        .count()
        .orderBy("len_end_to_end_id"),
        "Distribuição de tamanho de end_to_end_id"
    )

    safe_show(
        df_pix_keys.alias("p")
        .join(df_mobile_keys.alias("m"), F.col("p.cd_pix") == F.col("m.end_to_end_id"), "left_anti")
        .limit(20),
        "Amostra de cd_pix sem match no mobile",
        n=20
    )

    safe_show(
        df_mobile_keys.alias("m")
        .join(df_pix_keys.alias("p"), F.col("m.end_to_end_id") == F.col("p.cd_pix"), "left_anti")
        .limit(20),
        "Amostra de end_to_end_id sem match no PIX",
        n=20
    )

    # =========================================================
    # 4. JOIN PIX + MOBILE + CLIENTE
    # =========================================================
    print("4. Enriquecendo PIX com mobile e cliente...")

    # re-partition moderado antes do join pesado
    df_pix_raw = df_pix_raw.repartition(200, "cd_pix")
    df_mobile = df_mobile.repartition(200, "end_to_end_id")

    df_base = (
        df_pix_raw
        .join(df_mobile, df_pix_raw.cd_pix == df_mobile.end_to_end_id, "left")
        .join(F.broadcast(df_cliente), on="cd_cpf_pagador", how="inner")
    )

    segmentos_validos = ['BRASILIA', 'EXCLUSIVO', 'MILLENIUM', 'MILLENIUM CAPIT', 'RURAL PF']
    df_base = df_base.filter(F.col("ds_segmento").isin(segmentos_validos))

    print("4.1 Garantindo unicidade pós-join por cd_pix...")

    base_score_cols = [
        "device_name",
        "app_version",
        "ip_address",
        "latencia_rede_ms",
        "tempo_interacao_ms",
        "tempo_processamento_host_ms",
        "metodo_autenticacao",
        "session_id",
        "cd_retorno",
        "topaz_risk_score",
        "topaz_transacao_rejeitada",
        "topaz_transacao_habilitada",
        "is_agendamento_recorrente",
        "topaz_sync_id",
        "cd_cliente",
        "nr_idade",
        "qt_tempo_relacionamento_mes",
        "ds_segmento"
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

    # persist em disco para reduzir pressão de memória
    df_base = df_base.persist(StorageLevel.DISK_ONLY)

    # =========================================================
    # 4.2 LOG DE COBERTURA MOBILE
    # =========================================================
    print("4.2 Calculando cobertura mobile...")
    total_base = df_base.count()
    total_mobile_match = df_base.filter(F.col("end_to_end_id").isNotNull()).count()

    cobertura = round((total_mobile_match / total_base) * 100, 4) if total_base > 0 else 0.0

    print(f"Total base após join: {total_base}")
    print(f"Total com match mobile: {total_mobile_match}")
    print(f"Cobertura mobile final: {cobertura}%")

    safe_show(
        df_base
        .select("cd_pix", "end_to_end_id", "device_name", "app_version", "ip_address")
        .filter(F.col("end_to_end_id").isNotNull())
        .limit(20),
        "Amostra de registros com match mobile",
        n=20
    )

    # =========================================================
    # 4.3 TRATAMENTO DE SKEW
    # =========================================================
    print("4.3 Tratando distorções: Filtrando CPFs com volume anômalo (> 5.000 transações no trimestre)...")
    df_skew_filter = df_base.groupBy("cd_cpf_pagador").agg(F.count("cd_pix").alias("qt_pix_skew"))
    df_base = df_base.join(df_skew_filter, on="cd_cpf_pagador", how="inner")
    df_base = df_base.filter(F.col("qt_pix_skew") <= 5000).drop("qt_pix_skew")

    # =========================================================
    # 5. FEATURES HISTÓRICAS EM 90 DIAS
    # =========================================================
    print("5. Calculando features históricas (90 dias)...")

    w_user_window = (
        Window.partitionBy("cd_cpf_pagador")
        .orderBy(F.col("dt_pix").cast("long"))
        .rangeBetween(-90 * 86400, 0)
    )

    w_user_order = Window.partitionBy("cd_cpf_pagador").orderBy("dt_pix")

    df_features = (
        df_base
        .withColumn("qt_total_pix_trimestre", F.count("cd_pix").over(w_user_window))
        .withColumn("vl_mediana_pix_trimestre", F.percentile_approx("vl_pix", 0.5).over(w_user_window))
        .withColumn("vl_desvio_padrao_pix_trimestre", F.stddev("vl_pix").over(w_user_window))
        .withColumn("dt_transacao_anterior", F.lag("dt_pix").over(w_user_order))
        .withColumn("delta_pix_segundos", F.col("dt_pix").cast("long") - F.col("dt_transacao_anterior").cast("long"))
        .withColumn("qt_intervalo_transacao_minuto", F.round(F.col("delta_pix_segundos") / 60, 4))
        .withColumn("qt_aparelhos_distintos_trimestre", F.size(F.collect_set("device_name").over(w_user_window)))
        .withColumn("vl_latencia_rede_media_trimestre", F.avg("latencia_rede_ms").over(w_user_window))
        .withColumn("vl_tempo_interacao_medio_trimestre", F.avg("tempo_interacao_ms").over(w_user_window))
        .withColumn("qt_intervalo_mediana_trimestre", F.percentile_approx("qt_intervalo_transacao_minuto", 0.5).over(w_user_window))
        .withColumn("qt_intervalo_desvio_padrao_trimestre", F.stddev("qt_intervalo_transacao_minuto").over(w_user_window))
    )

    df_features = (
        df_features
        .withColumn("qt_aparelhos_distintos_trimestre", F.coalesce(F.col("qt_aparelhos_distintos_trimestre"), F.lit(0)))
        .withColumn("qt_intervalo_transacao_minuto", F.coalesce(F.col("qt_intervalo_transacao_minuto"), F.lit(0.0)))
    )

    # =========================================================
    # 6. qt_pix_dia_maximo_trimestre
    # =========================================================
    print("6. Consolidando contagem diária máxima...")

    df_daily_counts = (
        df_base.groupBy("cd_cpf_pagador", "data_pix")
        .agg(F.count("cd_pix").alias("daily_count"))
    )

    df_features.createOrReplaceTempView("features_base")
    df_daily_counts.createOrReplaceTempView("daily_counts")

    df_final = spark.sql("""
        SELECT
            f.cd_pix,
            f.dt_pix,
            f.cd_cpf_pagador,
            f.cd_cpf_cnpj_recebedor,
            f.ds_chave_pix,
            f.ds_tipo_chave,
            f.vl_pix,
            f.qt_total_pix_trimestre,
            f.vl_mediana_pix_trimestre,
            f.vl_desvio_padrao_pix_trimestre,
            f.qt_intervalo_transacao_minuto,
            f.qt_intervalo_mediana_trimestre,
            f.qt_intervalo_desvio_padrao_trimestre,
            COALESCE(MAX(d.daily_count), 0) as qt_pix_dia_maximo_trimestre,
            f.device_name,
            f.app_version,
            f.ip_address,
            f.latencia_rede_ms,
            f.vl_latencia_rede_media_trimestre,
            f.tempo_interacao_ms,
            f.vl_tempo_interacao_medio_trimestre,
            f.tempo_processamento_host_ms,
            f.metodo_autenticacao,
            f.session_id,
            f.cd_retorno,
            f.topaz_risk_score,
            f.topaz_transacao_rejeitada,
            f.topaz_transacao_habilitada,
            f.is_agendamento_recorrente,
            f.topaz_sync_id,
            f.qt_aparelhos_distintos_trimestre,
            f.nr_idade,
            f.qt_tempo_relacionamento_mes,
            current_date() as dt_carga
        FROM features_base f
        LEFT JOIN daily_counts d
            ON f.cd_cpf_pagador = d.cd_cpf_pagador
           AND d.data_pix BETWEEN date_sub(f.data_pix, 90) AND f.data_pix
        GROUP BY
            f.cd_pix,
            f.dt_pix,
            f.cd_cpf_pagador,
            f.cd_cpf_cnpj_recebedor,
            f.ds_chave_pix,
            f.ds_tipo_chave,
            f.vl_pix,
            f.qt_total_pix_trimestre,
            f.vl_mediana_pix_trimestre,
            f.vl_desvio_padrao_pix_trimestre,
            f.qt_intervalo_transacao_minuto,
            f.qt_intervalo_mediana_trimestre,
            f.qt_intervalo_desvio_padrao_trimestre,
            f.device_name,
            f.app_version,
            f.ip_address,
            f.latencia_rede_ms,
            f.vl_latencia_rede_media_trimestre,
            f.tempo_interacao_ms,
            f.vl_tempo_interacao_medio_trimestre,
            f.tempo_processamento_host_ms,
            f.metodo_autenticacao,
            f.session_id,
            f.cd_retorno,
            f.topaz_risk_score,
            f.topaz_transacao_rejeitada,
            f.topaz_transacao_habilitada,
            f.is_agendamento_recorrente,
            f.topaz_sync_id,
            f.qt_aparelhos_distintos_trimestre,
            f.nr_idade,
            f.qt_tempo_relacionamento_mes,
            f.data_pix
    """)

    # =========================================================
    # 7. DEDUPLICAÇÃO FINAL
    # =========================================================
    print("7. Aplicando deduplicação final por cd_pix...")

    final_score_cols = [
        "device_name",
        "app_version",
        "ip_address",
        "latencia_rede_ms",
        "tempo_interacao_ms",
        "tempo_processamento_host_ms",
        "metodo_autenticacao",
        "session_id",
        "cd_retorno",
        "topaz_risk_score",
        "topaz_transacao_rejeitada",
        "topaz_transacao_habilitada",
        "is_agendamento_recorrente",
        "topaz_sync_id",
        "nr_idade",
        "qt_tempo_relacionamento_mes"
    ]

    df_final = df_final.withColumn(
        "completude_score",
        build_completeness_score(df_final, final_score_cols)
    )

    w_final = Window.partitionBy("cd_pix").orderBy(
        F.col("completude_score").desc(),
        F.col("dt_pix").desc_nulls_last()
    )

    df_final = (
        df_final
        .withColumn("rn_dedup", F.row_number().over(w_final))
        .filter(F.col("rn_dedup") == 1)
        .drop("rn_dedup", "completude_score")
    )

    total_final = df_final.count()
    total_unique = df_final.select("cd_pix").distinct().count()

    print(f"Total final de linhas: {total_final}")
    print(f"Total final de cd_pix únicos: {total_unique}")
    print(f"Diferença (deve ser 0): {total_final - total_unique}")

    # =========================================================
    # 8. SAVE
    # =========================================================
    print("8. Salvando tabela final...")
    df_final.write.mode("overwrite").format("parquet").saveAsTable(output_table)
    print(f"Concluído! Tabela {output_table} gerada com sucesso.")

    df_base.unpersist()


if __name__ == "__main__":
    main()
