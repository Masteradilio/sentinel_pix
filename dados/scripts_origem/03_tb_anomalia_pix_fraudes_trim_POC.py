from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window


def create_spark_session():
    return (
        SparkSession.builder
        .appName("POC - PIX Fraudes Trimestre")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.network.timeout", "800s")
        .config("spark.executor.heartbeatInterval", "60s")
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


def ensure_column(df, col_name, data_type="string"):
    if col_name not in df.columns:
        df = df.withColumn(col_name, F.lit(None).cast(data_type))
    return df


def main():
    spark = create_spark_session()

    csv_fraudes_path = "hdfs:///modelos_ml/nudan/nudan_prd/ris/anomalia_comportamental/dados_fraude_pix.csv"
    mobile_table = "hmo_ml.tb_features_mobile_trim_poc"
    output_table = "hmo_ml.tb_fraudes_pix_trim_poc"

    print(f"--- Iniciando POC Fraudes Trimestre (Destino: {output_table}) ---")
    spark.sql(f"DROP TABLE IF EXISTS {output_table}")

    # =========================================================
    # 1. LEITURA CSV FRAUDES
    # =========================================================
    print("1. Lendo CSV histórico de fraudes...")
    try:
        df_fraudes = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(csv_fraudes_path)
        )
    except Exception as e:
        print(f"ERRO CRÍTICO: {e}")
        return

    # =========================================================
    # 2. GARANTIA DE COLUNAS E TIPAGEM
    # =========================================================
    print("2. Padronizando colunas e tipos...")

    string_cols = [
        "cd_pix",
        "cd_cpf_pagador",
        "cd_cpf_cnpj_recebedor",
        "ds_chave_pix",
        "ds_tipo_chave",
        "device_name",
        "app_version",
        "ip_address",
        "metodo_autenticacao",
        "session_id",
        "cd_retorno",
        "is_agendamento_recorrente",
        "topaz_sync_id",
    ]

    double_cols = [
        "vl_pix",
        "qt_total_pix_trimestre",
        "vl_mediana_pix_trimestre",
        "vl_desvio_padrao_pix_trimestre",
        "qt_intervalo_transacao_minuto",
        "qt_intervalo_mediana_trimestre",
        "qt_intervalo_desvio_padrao_trimestre",
        "qt_pix_dia_maximo_trimestre",
        "qt_aparelhos_distintos_trimestre",
        "nr_idade",
        "qt_tempo_relacionamento_mes",
        "latencia_rede_ms",
        "tempo_interacao_ms",
        "tempo_processamento_host_ms",
        "topaz_risk_score",
        "topaz_transacao_rejeitada",
        "topaz_transacao_habilitada",
        "vl_latencia_rede_media_trimestre",
        "vl_tempo_interacao_medio_trimestre",
    ]

    for c in string_cols:
        df_fraudes = ensure_column(df_fraudes, c, "string")

    for c in double_cols:
        df_fraudes = ensure_column(df_fraudes, c, "double")

    if "dt_pix" not in df_fraudes.columns:
        df_fraudes = df_fraudes.withColumn("dt_pix", F.lit(None).cast("timestamp"))
    else:
        df_fraudes = df_fraudes.withColumn("dt_pix", F.to_timestamp(F.col("dt_pix")))

    for c in string_cols:
        if c in df_fraudes.columns:
            df_fraudes = df_fraudes.withColumn(c, F.col(c).cast("string"))

    for c in double_cols:
        if c in df_fraudes.columns:
            df_fraudes = df_fraudes.withColumn(c, F.col(c).cast("double"))

    df_fraudes = df_fraudes.withColumn("tp_fraude", F.lit(1))
    df_fraudes = df_fraudes.withColumn("dt_carga", F.current_date())
    df_fraudes = df_fraudes.filter(F.col("cd_pix").isNotNull())

    # =========================================================
    # 3. DEDUPLICAÇÃO DA PRÓPRIA BASE DE FRAUDES
    # =========================================================
    print("3. Deduplicando base de fraudes por cd_pix...")

    fraude_score_cols = [
        "cd_cpf_pagador",
        "cd_cpf_cnpj_recebedor",
        "ds_chave_pix",
        "ds_tipo_chave",
        "vl_pix",
        "dt_pix",
        "nr_idade",
        "qt_tempo_relacionamento_mes",
        "qt_total_pix_trimestre",
        "vl_mediana_pix_trimestre",
        "vl_desvio_padrao_pix_trimestre",
        "qt_intervalo_transacao_minuto",
        "qt_intervalo_mediana_trimestre",
        "qt_intervalo_desvio_padrao_trimestre",
        "qt_pix_dia_maximo_trimestre",
        "qt_aparelhos_distintos_trimestre",
        "vl_latencia_rede_media_trimestre",
        "vl_tempo_interacao_medio_trimestre",
    ]

    df_fraudes = df_fraudes.withColumn(
        "fraude_score",
        build_completeness_score(df_fraudes, fraude_score_cols)
    )

    w_fraude = Window.partitionBy("cd_pix").orderBy(
        F.col("fraude_score").desc(),
        F.col("dt_pix").desc_nulls_last()
    )

    df_fraudes = (
        df_fraudes
        .withColumn("rn_fraude", F.row_number().over(w_fraude))
        .filter(F.col("rn_fraude") == 1)
        .drop("rn_fraude", "fraude_score")
    )

    # =========================================================
    # 4. MOBILE TRIMESTRAL
    # =========================================================
    print("4. Carregando e deduplicando mobile trimestral...")
    df_mobile_raw = spark.table(mobile_table)

    mobile_cols = [
        "end_to_end_id",
        "data_hora_inicio",
        "data_referencia",
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

    existing_mobile_cols = [c for c in mobile_cols if c in df_mobile_raw.columns]
    df_mobile = df_mobile_raw.select(*existing_mobile_cols)

    df_mobile = df_mobile.filter(F.col("end_to_end_id").isNotNull())

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

    if "data_referencia" in df_mobile.columns:
        w_mobile = Window.partitionBy("end_to_end_id").orderBy(
            F.col("mobile_score").desc(),
            F.col("data_referencia").desc_nulls_last(),
            F.col("data_hora_inicio").desc_nulls_last()
        )
    else:
        w_mobile = Window.partitionBy("end_to_end_id").orderBy(
            F.col("mobile_score").desc(),
            F.col("data_hora_inicio").desc_nulls_last()
        )

    df_mobile = (
        df_mobile
        .withColumn("rn_mobile", F.row_number().over(w_mobile))
        .filter(F.col("rn_mobile") == 1)
        .drop("rn_mobile", "mobile_score")
    )

    # RENOMEAR CAMPOS DO MOBILE PARA EVITAR AMBIGUIDADE
    df_mobile = df_mobile.select(
        F.col("end_to_end_id").alias("mob_id"),
        F.col("device_name").alias("mobile_device_name"),
        F.col("app_version").alias("mobile_app_version"),
        F.col("ip_address").alias("mobile_ip_address"),
        F.col("latencia_rede_ms").alias("mobile_latencia_rede_ms"),
        F.col("tempo_interacao_ms").alias("mobile_tempo_interacao_ms"),
        F.col("tempo_processamento_host_ms").alias("mobile_tempo_processamento_host_ms"),
        F.col("metodo_autenticacao").alias("mobile_metodo_autenticacao"),
        F.col("session_id").alias("mobile_session_id"),
        F.col("cd_retorno").alias("mobile_cd_retorno"),
        F.col("topaz_risk_score").alias("mobile_topaz_risk_score"),
        F.col("topaz_transacao_rejeitada").alias("mobile_topaz_transacao_rejeitada"),
        F.col("topaz_transacao_habilitada").alias("mobile_topaz_transacao_habilitada"),
        F.col("is_agendamento_recorrente").alias("mobile_is_agendamento_recorrente"),
        F.col("topaz_sync_id").alias("mobile_topaz_sync_id")
    )

    # =========================================================
    # 5. JOIN FRAUDES + MOBILE
    # =========================================================
    print("5. Enriquecendo fraudes com mobile...")
    df_fraudes_enriched = (
        df_fraudes
        .join(df_mobile, df_fraudes.cd_pix == df_mobile.mob_id, "left")
        .drop("mob_id")
    )

    # =========================================================
    # 6. CONSOLIDAÇÃO DE CAMPOS FINAL
    # =========================================================
    print("6. Consolidando colunas finais de mobile...")

    df_fraudes_enriched = df_fraudes_enriched.withColumn(
        "device_name",
        F.coalesce(F.col("device_name"), F.col("mobile_device_name"))
    ).withColumn(
        "app_version",
        F.coalesce(F.col("app_version"), F.col("mobile_app_version"))
    ).withColumn(
        "ip_address",
        F.coalesce(F.col("ip_address"), F.col("mobile_ip_address"))
    ).withColumn(
        "latencia_rede_ms",
        F.coalesce(F.col("latencia_rede_ms"), F.col("mobile_latencia_rede_ms").cast("double"))
    ).withColumn(
        "tempo_interacao_ms",
        F.coalesce(F.col("tempo_interacao_ms"), F.col("mobile_tempo_interacao_ms").cast("double"))
    ).withColumn(
        "tempo_processamento_host_ms",
        F.coalesce(F.col("tempo_processamento_host_ms"), F.col("mobile_tempo_processamento_host_ms").cast("double"))
    ).withColumn(
        "metodo_autenticacao",
        F.coalesce(F.col("metodo_autenticacao"), F.col("mobile_metodo_autenticacao"))
    ).withColumn(
        "session_id",
        F.coalesce(F.col("session_id"), F.col("mobile_session_id"))
    ).withColumn(
        "cd_retorno",
        F.coalesce(F.col("cd_retorno"), F.col("mobile_cd_retorno"))
    ).withColumn(
        "topaz_risk_score",
        F.coalesce(F.col("topaz_risk_score"), F.col("mobile_topaz_risk_score").cast("double"))
    ).withColumn(
        "topaz_transacao_rejeitada",
        F.coalesce(F.col("topaz_transacao_rejeitada"), F.col("mobile_topaz_transacao_rejeitada").cast("double"))
    ).withColumn(
        "topaz_transacao_habilitada",
        F.coalesce(F.col("topaz_transacao_habilitada"), F.col("mobile_topaz_transacao_habilitada").cast("double"))
    ).withColumn(
        "is_agendamento_recorrente",
        F.coalesce(F.col("is_agendamento_recorrente"), F.col("mobile_is_agendamento_recorrente"))
    ).withColumn(
        "topaz_sync_id",
        F.coalesce(F.col("topaz_sync_id"), F.col("mobile_topaz_sync_id"))
    )

    # colunas derivadas
    df_fraudes_enriched = df_fraudes_enriched.withColumn(
        "vl_latencia_rede_media_trimestre",
        F.coalesce(
            F.col("vl_latencia_rede_media_trimestre"),
            F.col("latencia_rede_ms").cast("double")
        )
    )

    df_fraudes_enriched = df_fraudes_enriched.withColumn(
        "vl_tempo_interacao_medio_trimestre",
        F.coalesce(
            F.col("vl_tempo_interacao_medio_trimestre"),
            F.col("tempo_interacao_ms").cast("double")
        )
    )

    if "qt_aparelhos_distintos_trimestre" not in df_fraudes_enriched.columns:
        df_fraudes_enriched = df_fraudes_enriched.withColumn(
            "qt_aparelhos_distintos_trimestre",
            F.lit(None).cast("double")
        )

    # remove colunas auxiliares mobile
    cols_drop_mobile = [
        "mobile_device_name",
        "mobile_app_version",
        "mobile_ip_address",
        "mobile_latencia_rede_ms",
        "mobile_tempo_interacao_ms",
        "mobile_tempo_processamento_host_ms",
        "mobile_metodo_autenticacao",
        "mobile_session_id",
        "mobile_cd_retorno",
        "mobile_topaz_risk_score",
        "mobile_topaz_transacao_rejeitada",
        "mobile_topaz_transacao_habilitada",
        "mobile_is_agendamento_recorrente",
        "mobile_topaz_sync_id",
    ]

    for c in cols_drop_mobile:
        if c in df_fraudes_enriched.columns:
            df_fraudes_enriched = df_fraudes_enriched.drop(c)

    # =========================================================
    # 7. DEDUPLICAÇÃO FINAL PÓS-JOIN POR cd_pix
    # =========================================================
    print("7. Aplicando deduplicação final por cd_pix...")

    final_score_cols = [
        "cd_cpf_pagador",
        "cd_cpf_cnpj_recebedor",
        "ds_chave_pix",
        "ds_tipo_chave",
        "vl_pix",
        "dt_pix",
        "qt_total_pix_trimestre",
        "vl_mediana_pix_trimestre",
        "vl_desvio_padrao_pix_trimestre",
        "qt_intervalo_transacao_minuto",
        "qt_intervalo_mediana_trimestre",
        "qt_intervalo_desvio_padrao_trimestre",
        "qt_pix_dia_maximo_trimestre",
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
        "qt_aparelhos_distintos_trimestre",
        "nr_idade",
        "qt_tempo_relacionamento_mes",
        "vl_latencia_rede_media_trimestre",
        "vl_tempo_interacao_medio_trimestre",
    ]

    df_fraudes_enriched = df_fraudes_enriched.withColumn(
        "completude_score",
        build_completeness_score(df_fraudes_enriched, final_score_cols)
    )

    w_final = Window.partitionBy("cd_pix").orderBy(
        F.col("completude_score").desc(),
        F.col("dt_pix").desc_nulls_last()
    )

    df_fraudes_enriched = (
        df_fraudes_enriched
        .withColumn("rn_dedup", F.row_number().over(w_final))
        .filter(F.col("rn_dedup") == 1)
        .drop("rn_dedup", "completude_score")
    )

    # =========================================================
    # 8. LOGS
    # =========================================================
    total_final = df_fraudes_enriched.count()
    total_unique = df_fraudes_enriched.select("cd_pix").distinct().count()

    print(f"Total final de linhas: {total_final}")
    print(f"Total final de cd_pix únicos: {total_unique}")
    print(f"Diferença (deve ser 0): {total_final - total_unique}")

    # =========================================================
    # 9. SAVE
    # =========================================================
    print("9. Salvando tabela trimestral de fraudes...")
    df_fraudes_enriched.write.mode("overwrite").format("parquet").saveAsTable(output_table)

    print(f"✅ SUCESSO! Tabela {output_table} criada.")


if __name__ == "__main__":
    main()
