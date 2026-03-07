from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window


def create_spark_session():
    return (
        SparkSession.builder
        .appName("POC - PIX Anomalia Normais Trimestre")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.executor.cores", "4")
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.hive.convertMetastoreParquet", "false")
        .enableHiveSupport()
        .getOrCreate()
    )


def main():
    spark = create_spark_session()

    output_table = "hmo_ml.tb_pix_anomalia_normais_trim_poc"
    mobile_table = "hmo_ml.tb_features_mobile_trim_poc"

    print(f"--- Iniciando POC Normais Trimestre (Destino: {output_table}) ---")
    spark.sql(f"DROP TABLE IF EXISTS {output_table}")

    # 1. Clientes
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
                    rank() OVER(partition by trim(a0100_segcodsgm) order by A0100_HDRDATA, A0100_HDRHORA desc) rank
                from landing_brb_db2_dna.dnab01
            ) rk
            where rank = 1
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
            F.coalesce(F.round(F.months_between(F.current_date(), F.col("dt_inicio_relacionamento")), 4), F.lit(0))
        )
        .drop("dt_cadastro_raw")
    )

    # 2. Mobile
    print(f"2. Carregando mobile ({mobile_table})...")
    df_mobile = spark.table(mobile_table).select(
        "end_to_end_id",
        "nsu_autorizadora",
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
        "topaz_sync_id"
    )

    # deduplicar mobile por end_to_end_id
    w_mobile = Window.partitionBy("end_to_end_id").orderBy(F.col("data_referencia").desc_nulls_last())
    if "data_referencia" in spark.table(mobile_table).columns:
        df_mobile = spark.table(mobile_table).withColumn("rn", F.row_number().over(w_mobile)).filter("rn = 1").drop("rn")
    else:
        df_mobile = df_mobile.dropDuplicates(["end_to_end_id"])

    # 3. PIX bruto
    print("3. Carregando PIX bruto trimestral...")
    df_pix_raw = spark.sql("""
        SELECT
            t.ds_id_pix as cd_pix,
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

    # 4. Join mobile + cliente
    print("4. Enriquecendo PIX com mobile e cliente...")
    df_base = (
        df_pix_raw
        .join(df_mobile, df_pix_raw.cd_pix == df_mobile.end_to_end_id, "left")
        .join(df_cliente, on="cd_cpf_pagador", how="inner")
    )

    segmentos_validos = ['BRASILIA', 'EXCLUSIVO', 'MILLENIUM', 'MILLENIUM CAPIT', 'RURAL PF']
    df_base = df_base.filter(F.col("ds_segmento").isin(segmentos_validos))

    # 5. Features históricas em 90 dias
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

    # não preencher agressivamente com -1 aqui
    df_features = (
        df_features
        .withColumn("qt_aparelhos_distintos_trimestre", F.coalesce(F.col("qt_aparelhos_distintos_trimestre"), F.lit(0)))
        .withColumn("qt_intervalo_transacao_minuto", F.coalesce(F.col("qt_intervalo_transacao_minuto"), F.lit(0.0)))
    )

    # 6. qt_pix_dia_maximo_trimestre
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
            f.cd_pix, f.dt_pix, f.cd_cpf_pagador, f.cd_cpf_cnpj_recebedor,
            f.ds_chave_pix, f.ds_tipo_chave,
            f.vl_pix, f.qt_total_pix_trimestre, f.vl_mediana_pix_trimestre, f.vl_desvio_padrao_pix_trimestre,
            f.qt_intervalo_transacao_minuto, f.qt_intervalo_mediana_trimestre, f.qt_intervalo_desvio_padrao_trimestre,
            f.device_name, f.app_version, f.ip_address,
            f.latencia_rede_ms, f.vl_latencia_rede_media_trimestre,
            f.tempo_interacao_ms, f.vl_tempo_interacao_medio_trimestre,
            f.tempo_processamento_host_ms, f.metodo_autenticacao, f.session_id, f.cd_retorno,
            f.topaz_risk_score, f.topaz_transacao_rejeitada, f.topaz_transacao_habilitada,
            f.is_agendamento_recorrente, f.topaz_sync_id,
            f.qt_aparelhos_distintos_trimestre,
            f.nr_idade, f.qt_tempo_relacionamento_mes, f.data_pix
    """)

    df_final.write.mode("overwrite").format("parquet").saveAsTable(output_table)
    print(f"Concluído! Tabela {output_table} gerada com sucesso.")


if __name__ == "__main__":
    main()
