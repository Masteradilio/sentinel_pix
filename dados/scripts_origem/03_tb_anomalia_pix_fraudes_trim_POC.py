from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window


def create_spark_session():
    return (
        SparkSession.builder
        .appName("POC - PIX Fraudes Trimestre")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .enableHiveSupport()
        .getOrCreate()
    )


def main():
    spark = create_spark_session()

    csv_fraudes_path = "hdfs:///modelos_ml/nudan/nudan_prd/ris/anomalia_comportamental/dados_fraude_pix.csv"
    mobile_table = "hmo_ml.tb_features_mobile_trim_poc"
    output_table = "hmo_ml.tb_fraudes_pix_trim_poc"

    print(f"--- Iniciando POC Fraudes Trimestre (Destino: {output_table}) ---")
    spark.sql(f"DROP TABLE IF EXISTS {output_table}")

    # 1. CSV fraudes
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

    # garantir tipos
    cols_to_cast = [
        "vl_pix",
        "qt_total_pix_trimestre",
        "vl_mediana_pix_trimestre",
        "vl_desvio_padrao_pix_trimestre",
        "qt_intervalo_transacao_minuto",
        "qt_intervalo_mediana_trimestre",
        "qt_intervalo_desvio_padrao_trimestre",
        "qt_pix_dia_maximo_trimestre",
        "nr_idade",
        "qt_tempo_relacionamento_mes"
    ]

    for col in cols_to_cast:
        if col in df_fraudes.columns:
            df_fraudes = df_fraudes.withColumn(col, F.col(col).cast("double"))
        else:
            df_fraudes = df_fraudes.withColumn(col, F.lit(None).cast("double"))

    if "cd_pix" not in df_fraudes.columns:
        df_fraudes = df_fraudes.withColumn("cd_pix", F.lit(None).cast("string"))

    # força label
    df_fraudes = df_fraudes.withColumn("tp_fraude", F.lit(1))

    # garantir colunas novas de chave/recebedor
    for c in ["cd_cpf_cnpj_recebedor", "ds_chave_pix", "ds_tipo_chave"]:
        if c not in df_fraudes.columns:
            df_fraudes = df_fraudes.withColumn(c, F.lit(None).cast("string"))

    # 2. mobile trimestral
    print("2. Enriquecendo fraudes com mobile trimestral...")
    df_mobile_raw = spark.table(mobile_table)

    if "data_referencia" in df_mobile_raw.columns:
        w_mobile = Window.partitionBy("end_to_end_id").orderBy(F.col("data_referencia").desc_nulls_last())
        df_mobile = (
            df_mobile_raw
            .withColumn("rn", F.row_number().over(w_mobile))
            .filter("rn = 1")
            .drop("rn")
        )
    else:
        df_mobile = df_mobile_raw.dropDuplicates(["end_to_end_id"])

    df_mobile = df_mobile.select(
        F.col("end_to_end_id").alias("mob_id"),
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

    df_fraudes_enriched = df_fraudes.join(
        df_mobile,
        df_fraudes.cd_pix == df_mobile.mob_id,
        "left"
    ).drop("mob_id")

    # defaults moderados só onde realmente necessário
    if "qt_aparelhos_distintos_trimestre" not in df_fraudes_enriched.columns:
        df_fraudes_enriched = df_fraudes_enriched.withColumn("qt_aparelhos_distintos_trimestre", F.lit(None).cast("double"))

    if "vl_latencia_rede_media_trimestre" not in df_fraudes_enriched.columns:
        df_fraudes_enriched = df_fraudes_enriched.withColumn("vl_latencia_rede_media_trimestre", F.col("latencia_rede_ms").cast("double"))

    if "vl_tempo_interacao_medio_trimestre" not in df_fraudes_enriched.columns:
        df_fraudes_enriched = df_fraudes_enriched.withColumn("vl_tempo_interacao_medio_trimestre", F.col("tempo_interacao_ms").cast("double"))

    # dt_carga
    df_fraudes_enriched = df_fraudes_enriched.withColumn("dt_carga", F.current_date())

    # salvar direto enriquecido
    print("3. Salvando tabela trimestral de fraudes...")
    df_fraudes_enriched.write.mode("overwrite").format("parquet").saveAsTable(output_table)

    print(f"✅ SUCESSO! Tabela {output_table} criada.")


if __name__ == "__main__":
    main()
