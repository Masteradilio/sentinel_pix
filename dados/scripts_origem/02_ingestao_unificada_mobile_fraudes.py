from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel


def create_spark_session():
    return (
        SparkSession.builder
        .appName("POC - PIX Fraudes Trimestre Unificado V5 - Otimizado")
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
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.default.parallelism", "64")
        .config("spark.sql.autoBroadcastJoinThreshold", "104857600")  # 100MB
        .config("spark.sql.broadcastTimeout", "1200")
        .config("spark.network.timeout", "1200s")
        .config("spark.executor.heartbeatInterval", "60s")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.hive.convertMetastoreParquet", "false")
        .config("spark.yarn.executor.memoryOverhead", "2048")
        .enableHiveSupport()
        .getOrCreate()
    )


def build_completeness_score(df, cols):
    expr = None
    for c in cols:
        if c in df.columns:
            term = F.when(F.col(c).isNotNull(), F.lit(1)).otherwise(F.lit(0))
            expr = term if expr is None else expr + term
    return expr if expr is not None else F.lit(0)


def main():
    spark = create_spark_session()

    output_table = "hmo_ml.tb_pix_anomalia_fraudes_trim_poc_v2"
    mbk_source_table = "landing_brb_oracle_mbk.aut"

    days_back = 90
    cutoff_date = spark.sql(
        f"SELECT cast(date_sub(current_date(), {days_back}) as string)"
    ).collect()[0][0]

    print("=" * 80)
    print("PROCESSAMENTO UNIFICADO PIX FRAUDES + MOBILE V5 (OTIMIZADO)")
    print(f"Data de corte: {cutoff_date}")
    print(f"Tabela destino: {output_table}")
    print("=" * 80)

    spark.sql(f"DROP TABLE IF EXISTS {output_table}")

    # =========================================================
    # 1. CHAVES DE FRAUDE (pequeno — broadcast seguro)
    # =========================================================
    print("1. Gerando chaves de fraude PIX...")

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
    print(f"Total chaves de fraude: {fraud_count}")

    # =========================================================
    # 2. CLIENTES
    # =========================================================
    print("2. Carregando clientes...")

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
                                concat(
                                    cast(substr(cast(X0100_CLTDATNAS AS INT),1,4) as STRING),
                                    '-',
                                    cast(substr(cast(X0100_CLTDATNAS AS INT),5,2) as STRING),
                                    '-',
                                    cast(substr(cast(X0100_CLTDATNAS AS INT),7,2) as STRING)
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
    # 3. BASE PIX FRAUDADA (filtrada cedo — reduz volume)
    # =========================================================
    print("3. Gerando base PIX fraudada...")

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

    # JOIN com fraud_keys via broadcast (são poucos registros)
    df_pix_base = (
        df_pix_base
        .join(F.broadcast(df_fraud_keys), on="cd_pix", how="inner")
        .dropDuplicates(["cd_pix"])
        .withColumn("is_fraud", F.lit(1))
    )

    # Persiste em disco — base pequena, mas reutilizada várias vezes
    df_pix_base = df_pix_base.persist(StorageLevel.DISK_ONLY)
    pix_fraud_count = df_pix_base.count()
    print(f"Total PIX fraude (dedup): {pix_fraud_count}")

    # =========================================================
    # 4. MOBILE REDUZIDO (só fraudes — filtro precoce)
    # =========================================================
    print("4. Gerando base mobile reduzida (só fraudes)...")

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
            CAST(NULLIF(REGEXP_EXTRACT(auttrn, '<BRB__TopazTransacaoHabilitada[^>]*>(.*?)</BRB__TopazTransacaoHabilitada>', 1), '') AS INT) AS topaz_transacao_habilitada,
            REGEXP_EXTRACT(auttrn, '<BRB__IsAgendamentoRecorrenteForTopaz[^>]*>(.*?)</BRB__IsAgendamentoRecorrenteForTopaz>', 1) AS is_agendamento_recorrente,
            REGEXP_EXTRACT(auttrn, '<BRB__SyncIdTopaz[^>]*>(.*?)</BRB__SyncIdTopaz>', 1) AS topaz_sync_id
        FROM {mbk_source_table}
        WHERE autdatref >= '{cutoff_date}'
          AND auttrn LIKE '%<transacao%'
    """)

    # Filtra só os end_to_end_id que existem nas fraudes (broadcast)
    df_mobile = (
        df_mobile
        .filter(F.col("end_to_end_id").isNotNull())
        .filter(F.length(F.trim(F.col("end_to_end_id"))) > 0)
        .join(
            F.broadcast(df_fraud_keys.select(F.col("cd_pix").alias("_ref"))),
            F.col("end_to_end_id") == F.col("_ref"),
            "inner"
        )
        .drop("_ref")
    )

    mobile_score_cols = [
        "device_name", "app_version", "ip_address", "latencia_rede_ms",
        "tempo_interacao_ms", "tempo_processamento_host_ms", "metodo_autenticacao",
        "session_id", "cd_retorno", "topaz_risk_score",
        "topaz_transacao_rejeitada", "topaz_transacao_habilitada",
        "is_agendamento_recorrente", "topaz_sync_id"
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

    df_mobile = df_mobile.persist(StorageLevel.DISK_ONLY)
    mobile_count = df_mobile.count()
    print(f"Total mobile fraude (dedup): {mobile_count}")

    # =========================================================
    # 5. JOIN PIX + MOBILE + CLIENTE
    # =========================================================
    print("5. Montando base enriquecida...")

    df_base = (
        df_pix_base
        .join(df_mobile, df_pix_base.cd_pix == df_mobile.end_to_end_id, "left")
        .join(F.broadcast(df_cliente), on="cd_cpf_pagador", how="left")
    )

    # Dedup pós-join
    base_score_cols = [
        "device_name", "app_version", "ip_address", "latencia_rede_ms",
        "tempo_interacao_ms", "tempo_processamento_host_ms", "metodo_autenticacao",
        "session_id", "cd_retorno", "topaz_risk_score", "topaz_transacao_rejeitada",
        "topaz_transacao_habilitada", "is_agendamento_recorrente", "topaz_sync_id",
        "cd_cliente", "nr_idade", "qt_tempo_relacionamento_mes", "ds_segmento"
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

    # =====================================================
    # CHECKPOINT: salvar em tabela temporária para quebrar
    # o DAG do Spark e liberar memória dos stages anteriores
    # =====================================================
    tmp_checkpoint = "hmo_ml.tmp_pix_fraud_checkpoint_v5"
    spark.sql(f"DROP TABLE IF EXISTS {tmp_checkpoint}")

    print("5.1 Salvando checkpoint intermediário...")
    df_base.write.mode("overwrite").format("parquet").saveAsTable(tmp_checkpoint)

    # Libera tudo que veio antes
    df_pix_base.unpersist()
    df_mobile.unpersist()
    df_fraud_keys.unpersist()

    # Recarrega do disco — DAG limpo
    df_base = spark.table(tmp_checkpoint)
    base_count = df_base.count()
    print(f"Total base pós-checkpoint: {base_count}")

    # =========================================================
    # 6. FEATURES HISTÓRICAS — WINDOW FUNCTIONS
    #    (mesma abordagem do script 01 de normais)
    # =========================================================
    print("6. Calculando features históricas com window functions...")

    w_user_window = (
        Window.partitionBy("cd_cpf_pagador")
        .orderBy(F.col("dt_pix").cast("long"))
        .rangeBetween(-90 * 86400, 0)
    )

    w_user_order = Window.partitionBy("cd_cpf_pagador").orderBy("dt_pix")

    df_features = (
        df_base
        .withColumn("qt_total_pix_trimestre",
                     F.count("cd_pix").over(w_user_window))
        .withColumn("vl_mediana_pix_trimestre",
                     F.percentile_approx("vl_pix", 0.5).over(w_user_window))
        .withColumn("vl_desvio_padrao_pix_trimestre",
                     F.stddev("vl_pix").over(w_user_window))
        .withColumn("dt_transacao_anterior",
                     F.lag("dt_pix").over(w_user_order))
        .withColumn("delta_pix_segundos",
                     F.col("dt_pix").cast("long") - F.col("dt_transacao_anterior").cast("long"))
        .withColumn("qt_intervalo_transacao_minuto",
                     F.round(F.col("delta_pix_segundos") / 60, 4))
        # -------------------------------------------------------
        # MUDANÇA-CHAVE: qt_aparelhos_distintos via collect_set
        # em window function (como no script 01 de normais)
        # ELIMINA o join explosivo da seção 7 original
        # -------------------------------------------------------
        .withColumn("qt_aparelhos_distintos_trimestre",
                     F.size(F.collect_set("device_name").over(w_user_window)))
        .withColumn("vl_latencia_rede_media_trimestre",
                     F.avg("latencia_rede_ms").over(w_user_window))
        .withColumn("vl_tempo_interacao_medio_trimestre",
                     F.avg("tempo_interacao_ms").over(w_user_window))
        .withColumn("qt_intervalo_mediana_trimestre",
                     F.percentile_approx("qt_intervalo_transacao_minuto", 0.5).over(w_user_window))
        .withColumn("qt_intervalo_desvio_padrao_trimestre",
                     F.stddev("qt_intervalo_transacao_minuto").over(w_user_window))
    )

    # Coalesce de nulos
    df_features = (
        df_features
        .withColumn("qt_aparelhos_distintos_trimestre",
                     F.coalesce(F.col("qt_aparelhos_distintos_trimestre"), F.lit(0)))
        .withColumn("qt_intervalo_transacao_minuto",
                     F.coalesce(F.col("qt_intervalo_transacao_minuto"), F.lit(0.0)))
    )

    # =========================================================
    # 7. qt_pix_dia_maximo_trimestre
    #    (mesma abordagem do script 01: LEFT JOIN com daily_counts
    #     via SQL com GROUP BY — mais eficiente que o join do V4)
    # =========================================================
    print("7. Consolidando contagem diária máxima...")

    df_daily_counts = (
        df_base.groupBy("cd_cpf_pagador", "data_pix")
        .agg(F.count("cd_pix").alias("daily_count"))
    )

    df_features.createOrReplaceTempView("features_fraud")
    df_daily_counts.createOrReplaceTempView("daily_counts_fraud")

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
            f.is_fraud,
            current_date() as dt_carga
        FROM features_fraud f
        LEFT JOIN daily_counts_fraud d
            ON f.cd_cpf_pagador = d.cd_cpf_pagador
           AND d.data_pix BETWEEN date_sub(f.data_pix, 90) AND f.data_pix
        GROUP BY
            f.cd_pix, f.dt_pix, f.cd_cpf_pagador, f.cd_cpf_cnpj_recebedor,
            f.ds_chave_pix, f.ds_tipo_chave, f.vl_pix,
            f.qt_total_pix_trimestre, f.vl_mediana_pix_trimestre,
            f.vl_desvio_padrao_pix_trimestre, f.qt_intervalo_transacao_minuto,
            f.qt_intervalo_mediana_trimestre, f.qt_intervalo_desvio_padrao_trimestre,
            f.device_name, f.app_version, f.ip_address,
            f.latencia_rede_ms, f.vl_latencia_rede_media_trimestre,
            f.tempo_interacao_ms, f.vl_tempo_interacao_medio_trimestre,
            f.tempo_processamento_host_ms, f.metodo_autenticacao,
            f.session_id, f.cd_retorno, f.topaz_risk_score,
            f.topaz_transacao_rejeitada, f.topaz_transacao_habilitada,
            f.is_agendamento_recorrente, f.topaz_sync_id,
            f.qt_aparelhos_distintos_trimestre, f.nr_idade,
            f.qt_tempo_relacionamento_mes, f.is_fraud, f.data_pix
    """)

    # =========================================================
    # 8. DEDUPLICAÇÃO FINAL
    # =========================================================
    print("8. Deduplicação final por cd_pix...")

    final_score_cols = [
        "device_name", "app_version", "ip_address", "latencia_rede_ms",
        "tempo_interacao_ms", "tempo_processamento_host_ms", "metodo_autenticacao",
        "session_id", "cd_retorno", "topaz_risk_score", "topaz_transacao_rejeitada",
        "topaz_transacao_habilitada", "is_agendamento_recorrente", "topaz_sync_id",
        "nr_idade", "qt_tempo_relacionamento_mes"
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
        .withColumn("rn_final", F.row_number().over(w_final))
        .filter(F.col("rn_final") == 1)
        .drop("rn_final", "completude_score", "dt_transacao_anterior",
               "delta_pix_segundos", "data_pix")
    )

    # =========================================================
    # 9. SAVE
    # =========================================================
    print("9. Salvando tabela final...")
    df_final.write.mode("overwrite").format("parquet").saveAsTable(output_table)

    total_final = spark.table(output_table).count()
    print(f"Concluído! Tabela {output_table} gerada com {total_final} registros.")

    # Limpeza de temporárias
    spark.sql(f"DROP TABLE IF EXISTS {tmp_checkpoint}")


if __name__ == "__main__":
    main()
