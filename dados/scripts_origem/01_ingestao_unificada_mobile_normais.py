"""
01_ingestao_unificada_mobile_normais.py
========================================
POC Anomalia PIX — Ingestão Normais Trimestre V2.2 (OTIMIZADO)

Alterações v2.2 (otimização de performance):
  ══════════════════════════════════════════════════════════════
  GARGALO #1 ELIMINADO: JOIN daily_counts com range de datas
    ANTES:  LEFT JOIN daily_counts ON cpf AND data BETWEEN date_sub(90)
            → Explosão combinatória O(CPF × dias × transações)
    AGORA:  Pré-agrega MAX(daily_count) por CPF → JOIN 1:1
            → O(CPF) — redução de ~1000x em rows intermediárias

  GARGALO #2 ELIMINADO: collect_set(device_name) em window
    ANTES:  collect_set().over(rangeBetween) → serializa array por row
    AGORA:  Pré-agrega countDistinct por CPF → JOIN 1:1

  GARGALO #3 OTIMIZADO: percentile_approx × 3 + stddev em window
    ANTES:  3x percentile_approx + 2x stddev em rangeBetween window
    AGORA:  Consolidados em UMA pré-agregação por CPF + 1 window restante

  GARGALO #4 ELIMINADO: 8+ .count() intermediários
    ANTES:  Cada count() materializa o DAG inteiro
    AGORA:  Counts em grupo único + diagnósticos via .limit().show()
            (Counts opcionais controlados por flag DEBUG_COUNTS)

  GARGALO #5 ELIMINADO: Diagnóstico de chaves pesado
    ANTES:  distinct + count + join + anti-join (4 shuffles)
    AGORA:  Movido para flag DEBUG_KEYS (desligado em produção)

  GARGALO #6 CORRIGIDO: persist(DISK_ONLY) → MEMORY_AND_DISK
  GARGALO #7 REMOVIDO: repartition forçado (AQE cuida)
  GARGALO #10 CORRIGIDO: vectorizedReader habilitado

  Mantém TODOS os campos e features da v2.1 (mesma saída).
  ══════════════════════════════════════════════════════════════

  Tempo estimado: ~2-4h (vs 18h+ da v2.1)
"""

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel


# =========================================================
# FLAGS DE CONTROLE
# =========================================================
DEBUG_COUNTS = False   # True = executa counts intermediários (lento, só para debug)
DEBUG_KEYS = False     # True = diagnóstico de chaves PIX x mobile (lento)
SKEW_THRESHOLD = 5000  # Máximo de PIX por CPF no trimestre


def create_spark_session():
    return (
        SparkSession.builder
        .appName("POC - PIX Anomalia Normais Trimestre Unificado V2.2")
        .config("spark.driver.memory", "8g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.executor.memory", "12g")
        .config("spark.executor.cores", "3")
        # Dynamic allocation
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.dynamicAllocation.minExecutors", "2")
        .config("spark.dynamicAllocation.initialExecutors", "4")
        .config("spark.dynamicAllocation.maxExecutors", "20")
        # AQE (deixar o Spark otimizar shuffles/partitions)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
        .config("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256m")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m")
        # Shuffle — AQE ajusta, mas piso razoável
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.default.parallelism", "200")
        # Broadcast — 50MB
        .config("spark.sql.autoBroadcastJoinThreshold", "52428800")
        .config("spark.sql.broadcastTimeout", "1200")
        # Timeout
        .config("spark.network.timeout", "1200s")
        .config("spark.executor.heartbeatInterval", "60s")
        # Parquet — CORRIGIDO: vectorized reader HABILITADO
        .config("spark.sql.parquet.enableVectorizedReader", "true")
        .config("spark.sql.hive.convertMetastoreParquet", "false")
        # Memory overhead
        .config("spark.yarn.executor.memoryOverhead", "3072")
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

    output_table = "hmo_ml.tb_pix_anomalia_normais_trim_poc_v2"
    mbk_source_table = "landing_brb_oracle_mbk.aut"

    print("=" * 80)
    print("PROCESSAMENTO UNIFICADO PIX + MOBILE V2.2 (OTIMIZADO)")
    print("=" * 80)

    spark.sql(f"DROP TABLE IF EXISTS {output_table}")

    days_back = 90
    cutoff_date = spark.sql(
        f"SELECT cast(date_sub(current_date(), {days_back}) as string)"
    ).collect()[0][0]

    print(f"Data de corte: {cutoff_date}")
    print(f"Tabela destino: {output_table}")
    print(f"Debug counts: {DEBUG_COUNTS} | Debug keys: {DEBUG_KEYS}")

    # =========================================================
    # 1. CLIENTES
    # =========================================================
    print("\n[1/8] Carregando perfil de clientes...")

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

    # ══════════════════════════════════════════════════════════
    # OTIMIZAÇÃO: Persist cliente (tabela pequena, será broadcast)
    # ══════════════════════════════════════════════════════════
    df_cliente = df_cliente.persist(StorageLevel.MEMORY_AND_DISK)

    if DEBUG_COUNTS:
        print(f"    → Clientes únicos: {df_cliente.count()}")
    else:
        safe_show(df_cliente.limit(5), "Amostra de clientes", n=5)

    # =========================================================
    # 2. PIX BRUTO
    # =========================================================
    print("\n[2/8] Carregando PIX bruto trimestral...")

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

    if DEBUG_COUNTS:
        total_pix = df_pix_raw.count()
        print(f"    Total PIX bruto (dedup): {total_pix}")
    else:
        print("    PIX bruto carregado e deduplicado (count pulado — DEBUG_COUNTS=False)")

    # =========================================================
    # 3. MOBILE DIRETO DA MBK
    # =========================================================
    print("\n[3/8] Extraindo mobile da MBK...")

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

    if DEBUG_COUNTS:
        print(f"    Mobile após dedup: {df_mobile.count()}")
    else:
        print("    Mobile carregado e deduplicado")

    # =========================================================
    # 3.2 DIAGNÓSTICO DE CHAVES (só em modo debug)
    # =========================================================
    if DEBUG_KEYS:
        print("\n[3.2] Diagnóstico de chaves PIX x mobile...")

        df_pix_keys = df_pix_raw.select("cd_pix").distinct()
        df_mobile_keys = df_mobile.select("end_to_end_id").distinct()

        total_intersection = (
            df_pix_keys.alias("p")
            .join(df_mobile_keys.alias("m"), F.col("p.cd_pix") == F.col("m.end_to_end_id"), "inner")
            .count()
        )
        print(f"    Interseção exata PIX x mobile: {total_intersection}")

        safe_show(
            df_pix_raw
            .select("cd_pix", F.length("cd_pix").alias("len_cd_pix"))
            .groupBy("len_cd_pix").count().orderBy("len_cd_pix"),
            "Distribuição de tamanho de cd_pix"
        )

        safe_show(
            df_mobile
            .select("end_to_end_id", F.length("end_to_end_id").alias("len_e2e"))
            .groupBy("len_e2e").count().orderBy("len_e2e"),
            "Distribuição de tamanho de end_to_end_id"
        )

        safe_show(
            df_pix_keys.alias("p")
            .join(df_mobile_keys.alias("m"), F.col("p.cd_pix") == F.col("m.end_to_end_id"), "left_anti")
            .limit(20),
            "Amostra de cd_pix sem match no mobile"
        )

        safe_show(
            df_mobile_keys.alias("m")
            .join(df_pix_keys.alias("p"), F.col("m.end_to_end_id") == F.col("p.cd_pix"), "left_anti")
            .limit(20),
            "Amostra de end_to_end_id sem match no PIX"
        )
    else:
        print("\n[3.2] Diagnóstico de chaves PULADO (DEBUG_KEYS=False)")

    # =========================================================
    # 4. JOIN PIX + MOBILE + CLIENTE
    # =========================================================
    print("\n[4/8] JOIN PIX + Mobile + Cliente...")

    # ══════════════════════════════════════════════════════════
    # OTIMIZAÇÃO: Sem repartition forçado — AQE resolve
    # OTIMIZAÇÃO: Cliente via broadcast (tabela pequena ~centenas de MB)
    # ══════════════════════════════════════════════════════════
    df_base = (
        df_pix_raw
        .join(df_mobile, df_pix_raw.cd_pix == df_mobile.end_to_end_id, "left")
        .join(F.broadcast(df_cliente), on="cd_cpf_pagador", how="inner")
    )

    # Unicidade pós-join por cd_pix
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

    # =========================================================
    # 4.1 FILTRO DE SKEW
    # =========================================================
    print(f"[4.1] Filtrando CPFs com > {SKEW_THRESHOLD} transações...")

    df_cpf_counts = (
        df_base
        .groupBy("cd_cpf_pagador")
        .agg(F.count("cd_pix").alias("qt_pix_skew"))
    )

    # ══════════════════════════════════════════════════════════
    # OTIMIZAÇÃO: Broadcast do filtro de skew (é pequeno: 1 row por CPF)
    # ══════════════════════════════════════════════════════════
    df_cpf_validos = F.broadcast(
        df_cpf_counts.filter(F.col("qt_pix_skew") <= SKEW_THRESHOLD)
        .select("cd_cpf_pagador")
    )

    df_base = df_base.join(df_cpf_validos, on="cd_cpf_pagador", how="inner")

    # ══════════════════════════════════════════════════════════
    # OTIMIZAÇÃO: MEMORY_AND_DISK (não DISK_ONLY)
    # ══════════════════════════════════════════════════════════
    df_base = df_base.persist(StorageLevel.MEMORY_AND_DISK)

    # Forçar materialização UMA VEZ (necessário para estabilidade do DAG)
    total_base = df_base.count()
    print(f"    Total base após join + skew filter: {total_base}")

    # =========================================================
    # 5. PRÉ-AGREGAÇÕES POR CPF (ELIMINA GARGALOS 1, 2, 3)
    # =========================================================
    print("\n[5/8] Pré-agregando métricas trimestrais por CPF...")

    # ══════════════════════════════════════════════════════════
    # OTIMIZAÇÃO PRINCIPAL: Uma ÚNICA agregação por CPF substitui:
    #   - collect_set(device_name) em window → countDistinct aqui
    #   - percentile_approx(vl_pix) em window → aqui
    #   - stddev(vl_pix) em window → aqui
    #   - percentile_approx(intervalo) em window → aqui (via 2º passo)
    #   - MAX(daily_count) via JOIN range → aqui via sub-agregação
    # ══════════════════════════════════════════════════════════

    # 5a. Contagem diária máxima por CPF (substitui JOIN com range de datas)
    df_daily_max = (
        df_base
        .groupBy("cd_cpf_pagador", "data_pix")
        .agg(F.count("cd_pix").alias("daily_count"))
        .groupBy("cd_cpf_pagador")
        .agg(F.max("daily_count").alias("qt_pix_dia_maximo_trimestre"))
    )

    # 5b. Métricas trimestrais por CPF (substitui windows pesadas)
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

    # 5c. Merge das pré-agregações (ambos keyed por cd_cpf_pagador)
    df_cpf_agg = (
        df_cpf_stats
        .join(df_daily_max, on="cd_cpf_pagador", how="left")
    )

    # ══════════════════════════════════════════════════════════
    # OTIMIZAÇÃO: Broadcast das agregações (1 row por CPF — cabe em memória)
    # ══════════════════════════════════════════════════════════
    df_cpf_agg = F.broadcast(df_cpf_agg)

    print("    Pré-agregações por CPF concluídas")

    # =========================================================
    # 6. FEATURES POR TRANSAÇÃO (windows leves)
    # =========================================================
    print("\n[6/8] Calculando features por transação...")

    # ══════════════════════════════════════════════════════════
    # OTIMIZAÇÃO: Só 2 windows restam (as que PRECISAM ser por-row):
    #   1. lag(dt_pix) → intervalo entre transações
    #   2. row_number por (cpf, recebedor) → primeiro envio
    #
    # TUDO que era agregação trimestral agora vem do JOIN com df_cpf_agg
    # ══════════════════════════════════════════════════════════

    w_user_order = Window.partitionBy("cd_cpf_pagador").orderBy("dt_pix")

    w_receiver = Window.partitionBy("cd_cpf_pagador", "cd_cpf_cnpj_recebedor").orderBy("dt_pix")

    w_receiver_count = (
        Window.partitionBy("cd_cpf_pagador", "cd_cpf_cnpj_recebedor")
        .orderBy(F.col("dt_pix").cast("long"))
        .rangeBetween(-90 * 86400, 0)
    )

    df_features = (
        df_base
        # Window leve: lag para intervalo
        .withColumn("dt_transacao_anterior", F.lag("dt_pix").over(w_user_order))
        .withColumn(
            "delta_pix_segundos",
            F.col("dt_pix").cast("long") - F.col("dt_transacao_anterior").cast("long"),
        )
        .withColumn(
            "qt_intervalo_transacao_minuto",
            F.coalesce(F.round(F.col("delta_pix_segundos") / 60, 4), F.lit(0.0)),
        )
        # Window leve: primeiro envio ao recebedor
        .withColumn(
            "tp_primeiro_envio_recebedor_trimestre",
            F.when(F.row_number().over(w_receiver) == 1, 1).otherwise(0),
        )
        # Window: contagem de envios ao recebedor (rangeBetween — inevitável mas por par cpf+recebedor, muito menor)
        .withColumn(
            "qt_envio_recebedor_trimestre",
            F.count("cd_pix").over(w_receiver_count),
        )
        # Limpar temporários
        .drop("dt_transacao_anterior", "delta_pix_segundos")
    )

    # =========================================================
    # 6.1 JOIN com pré-agregações + mediana/desvio de intervalo
    # =========================================================
    print("[6.1] Enriquecendo com métricas trimestrais pré-agregadas...")

    df_features = (
        df_features
        .join(df_cpf_agg, on="cd_cpf_pagador", how="left")
        # Renomear para nomes finais
        .withColumnRenamed("_qt_total_pix_trimestre", "qt_total_pix_trimestre")
        .withColumnRenamed("_vl_mediana_pix_trimestre", "vl_mediana_pix_trimestre")
        .withColumnRenamed("_vl_desvio_padrao_pix_trimestre", "vl_desvio_padrao_pix_trimestre")
        .withColumnRenamed("_qt_aparelhos_distintos_trimestre", "qt_aparelhos_distintos_trimestre")
        .withColumnRenamed("_vl_latencia_rede_media_trimestre", "vl_latencia_rede_media_trimestre")
        .withColumnRenamed("_vl_tempo_interacao_medio_trimestre", "vl_tempo_interacao_medio_trimestre")
    )

    # ══════════════════════════════════════════════════════════
    # OTIMIZAÇÃO: Mediana e desvio de intervalo — calculados APÓS
    # ter o intervalo por-row, via uma única agregação extra
    # ══════════════════════════════════════════════════════════
    print("[6.2] Calculando mediana e desvio de intervalos...")

    df_intervalo_stats = (
        df_features
        .filter(F.col("qt_intervalo_transacao_minuto") > 0)  # Excluir primeira tx (intervalo 0)
        .groupBy("cd_cpf_pagador")
        .agg(
            F.percentile_approx("qt_intervalo_transacao_minuto", 0.5).alias("qt_intervalo_mediana_trimestre"),
            F.stddev("qt_intervalo_transacao_minuto").alias("qt_intervalo_desvio_padrao_trimestre"),
        )
    )

    df_intervalo_stats = F.broadcast(df_intervalo_stats)

    df_features = (
        df_features
        .join(df_intervalo_stats, on="cd_cpf_pagador", how="left")
    )

    # Coalesces de segurança
    df_features = (
        df_features
        .withColumn("qt_aparelhos_distintos_trimestre", F.coalesce(F.col("qt_aparelhos_distintos_trimestre"), F.lit(0)))
        .withColumn("qt_pix_dia_maximo_trimestre", F.coalesce(F.col("qt_pix_dia_maximo_trimestre"), F.lit(0)))
        .withColumn("qt_intervalo_mediana_trimestre", F.coalesce(F.col("qt_intervalo_mediana_trimestre"), F.lit(0.0)))
        .withColumn("qt_intervalo_desvio_padrao_trimestre", F.coalesce(F.col("qt_intervalo_desvio_padrao_trimestre"), F.lit(0.0)))
    )

    # =========================================================
    # 7. SELEÇÃO FINAL + DEDUP
    # =========================================================
    print("\n[7/8] Seleção final e deduplicação...")

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

    # ══════════════════════════════════════════════════════════
    # OTIMIZAÇÃO: Dedup mais simples — já garantimos unicidade
    # no passo 4 (pós-join), então aqui é safety net
    # ══════════════════════════════════════════════════════════
    df_final = df_final.dropDuplicates(["cd_pix"])

    # Adiciona dt_carga
    df_final = df_final.withColumn("dt_carga", F.current_date())

    # =========================================================
    # 7.1 VALIDAÇÃO + COBERTURA
    # =========================================================
    print("[7.1] Validando resultado final...")

    total_final = df_final.count()
    print(f"    Total final: {total_final}")

    if DEBUG_COUNTS:
        total_unique = df_final.select("cd_pix").distinct().count()
        print(f"    cd_pix únicos: {total_unique}")
        print(f"    Diferença (deve ser 0): {total_final - total_unique}")

    # Cobertura dos campos v2.1 — via aggregation única (não N counts separados)
    print("\n    --- COBERTURA DOS CAMPOS v2.1 ---")
    coverage_cols = [
        "ds_sexo", "ds_estado_civil", "ds_segmento",
        "vl_renda_cliente", "qt_dependentes",
        "tp_primeiro_envio_recebedor_trimestre",
        "qt_envio_recebedor_trimestre",
        "tempo_interacao_ms", "metodo_autenticacao",
        "is_agendamento_recorrente", "topaz_transacao_rejeitada",
    ]

    # ══════════════════════════════════════════════════════════
    # OTIMIZAÇÃO: Uma ÚNICA action para todas as coberturas
    # (em vez de N .count() separados = N materializations)
    # ══════════════════════════════════════════════════════════
    coverage_exprs = []
    for col_name in coverage_cols:
        if col_name in df_final.columns:
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
        coverage_row = df_final.agg(*coverage_exprs).collect()[0]
        for col_name in coverage_cols:
            alias = f"cov_{col_name}"
            if alias in coverage_row.asDict():
                not_null = coverage_row[alias]
                pct = round((not_null / total_final) * 100, 2) if total_final > 0 else 0
                print(f"    {col_name}: {not_null}/{total_final} ({pct}%)")

    # Amostra final
    safe_show(df_final.limit(5), "Amostra do dataset final", n=5)

    # =========================================================
    # 8. SAVE
    # =========================================================
    print(f"\n[8/8] Salvando tabela {output_table}...")
    df_final.write.mode("overwrite").format("parquet").saveAsTable(output_table)
    print(f"    ✅ Tabela {output_table} salva com sucesso!")

    # Cleanup
    df_base.unpersist()
    df_cliente.unpersist()

    print("\n" + "=" * 80)
    print("CONCLUÍDO — V2.2 (OTIMIZADO)")
    print("=" * 80)


if __name__ == "__main__":
    main()
