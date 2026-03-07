from pyspark.sql import SparkSession


def main():
    spark = (
        SparkSession.builder
        .appName("POC - ETL Mobile Trimestre")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.executor.cores", "4")
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.hive.convertMetastoreParquet", "false")
        .enableHiveSupport()
        .getOrCreate()
    )

    target_table = "hmo_ml.tb_features_mobile_trim_poc"
    source_table = "landing_brb_oracle_mbk.aut"

    days_back = 90
    date_query = f"SELECT cast(date_sub(current_date(), {days_back}) as string)"
    cutoff_date = spark.sql(date_query).collect()[0][0]

    print("=" * 70)
    print("PROCESSAMENTO MOBILE - TRIMESTRE")
    print(f"Data de corte: {cutoff_date}")
    print(f"Tabela destino: {target_table}")
    print("=" * 70)

    spark.sql(f"DROP TABLE IF EXISTS {target_table}")

    query = f"""
    CREATE TABLE {target_table}
    STORED AS PARQUET
    AS
    SELECT
        autnsu AS nsu_transacao,
        autnsuaut AS nsu_autorizadora,
        ctanum AS nr_conta,

        COALESCE(
            REGEXP_EXTRACT(auttrn, '<BRB__IdFimAFimOriginalPix.*?>(.*?)</BRB__IdFimAFimOriginalPix>', 1),
            REGEXP_EXTRACT(auttrn, '<FTN__IdFimAFimOriginalPix.*?>(.*?)</FTN__IdFimAFimOriginalPix>', 1),
            REGEXP_EXTRACT(auttrn, '<idFimAFim.*?>(.*?)</idFimAFim>', 1)
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

    FROM {source_table}
    WHERE autdatref >= '{cutoff_date}'
      AND auttrn LIKE '%<transacao%'
    """

    print("Executando criação da tabela mobile trimestral...")
    spark.sql(query)

    count = spark.sql(f"SELECT count(*) FROM {target_table}").collect()[0][0]
    print(f"Sucesso! Total de linhas extraídas: {count}")


if __name__ == "__main__":
    main()
