-- =====================================================================
-- Tabela: hmo_ml.tb_pix_normais_dataset_ready_v1
-- EXP-010F-R2 — etapa 04
-- Objetivo:
--   Unir normais R2 qualificados com normais parciais aproveitaveis R1.
--   Versao rolling:
--     - sempre limita R1 e R2 aos ultimos WINDOW_DAYS;
--     - normaliza ds_tipo_chave;
--     - remove variacao de 191 dias observada na primeira execucao historica.
-- =====================================================================

SET hive.execution.engine=tez;
SET hive.vectorized.execution.enabled=false;
SET hive.vectorized.execution.reduce.enabled=false;
SET hive.exec.compress.output=true;

DROP TABLE IF EXISTS ${hivevar:DB_WORK}.tb_pix_normais_dataset_ready_v1;

CREATE TABLE ${hivevar:DB_WORK}.tb_pix_normais_dataset_ready_v1
STORED AS PARQUET
AS
WITH params AS (
    SELECT
        date_sub(current_date, cast('${hivevar:WINDOW_LAG_DAYS}' AS int)) AS dt_fim,
        date_sub(
            current_date,
            cast('${hivevar:WINDOW_LAG_DAYS}' AS int) + cast('${hivevar:WINDOW_DAYS}' AS int) - 1
        ) AS dt_ini
),

r2 AS (
    SELECT
        cd_pix,
        cd_cpf_pagador,
        cd_cpf_cnpj_recebedor,
        cast(vl_pix AS double) AS vl_pix,
        cast(dt_pix AS timestamp) AS dt_pix,
        cast(data_pix AS date) AS data_pix,
        ds_chave_pix,

        CASE
            WHEN upper(trim(cast(ds_tipo_chave AS string))) LIKE 'CHAVE%ALEATORIA%' THEN 'CHAVE_ALEATORIA'
            WHEN upper(trim(cast(ds_tipo_chave AS string))) LIKE 'DOCUMENTO%TELEFONE%' THEN 'DOCUMENTO_TELEFONE'
            WHEN upper(trim(cast(ds_tipo_chave AS string))) = 'EMAIL' THEN 'EMAIL'
            WHEN upper(trim(cast(ds_tipo_chave AS string))) LIKE 'INFORMA%AUSENTE%' THEN 'INFORMACAO_AUSENTE'
            WHEN ds_tipo_chave IS NULL THEN 'INFORMACAO_AUSENTE'
            ELSE 'OUTROS'
        END AS ds_tipo_chave,

        hour,
        periodo_dia,

        CASE
            WHEN cast(vl_pix AS double) < 100 THEN 'A_000_100'
            WHEN cast(vl_pix AS double) < 500 THEN 'B_100_500'
            WHEN cast(vl_pix AS double) < 1000 THEN 'C_500_1000'
            WHEN cast(vl_pix AS double) < 5000 THEN 'D_1000_5000'
            WHEN cast(vl_pix AS double) < 10000 THEN 'E_5000_10000'
            ELSE 'F_10000_PLUS'
        END AS value_band,

        normal_sample_source,
        normal_sample_strategy,
        cast(normal_sample_weight AS double) AS normal_sample_weight,
        matched_control_flag,
        hard_negative_flag,
        recent_normal_flag,
        mbk_available_flag,
        autdatref,
        autdathorini,
        autcodret,
        device_name,
        app_version,
        ip_address,
        latencia_rede_ms,
        tempo_interacao_ms,
        tempo_processamento_host_ms,
        metodo_autenticacao,
        session_id,
        topaz_risk_score,
        topaz_transacao_rejeitada,
        is_agendamento_recorrente,
        mbk_completeness_score,
        is_fraud,
        dt_carga
    FROM ${hivevar:DB_WORK}.tb_pix_normais_qualified_sample_mbk_180d_v1 n
    JOIN params p ON 1 = 1
    WHERE cast(n.data_pix AS date) BETWEEN p.dt_ini AND p.dt_fim
),

r1_partial AS (
    SELECT
        cd_pix,
        cd_cpf_pagador,
        cd_cpf_cnpj_recebedor,
        cast(vl_pix AS double) AS vl_pix,
        cast(dt_pix AS timestamp) AS dt_pix,
        cast(data_pix AS date) AS data_pix,
        ds_chave_pix,

        CASE
            WHEN upper(trim(cast(ds_tipo_chave AS string))) LIKE 'CHAVE%ALEATORIA%' THEN 'CHAVE_ALEATORIA'
            WHEN upper(trim(cast(ds_tipo_chave AS string))) LIKE 'DOCUMENTO%TELEFONE%' THEN 'DOCUMENTO_TELEFONE'
            WHEN upper(trim(cast(ds_tipo_chave AS string))) = 'EMAIL' THEN 'EMAIL'
            WHEN upper(trim(cast(ds_tipo_chave AS string))) LIKE 'INFORMA%AUSENTE%' THEN 'INFORMACAO_AUSENTE'
            WHEN ds_tipo_chave IS NULL THEN 'INFORMACAO_AUSENTE'
            ELSE 'OUTROS'
        END AS ds_tipo_chave,

        hour,
        periodo_dia,

        CASE
            WHEN cast(vl_pix AS double) < 100 THEN 'A_000_100'
            WHEN cast(vl_pix AS double) < 500 THEN 'B_100_500'
            WHEN cast(vl_pix AS double) < 1000 THEN 'C_500_1000'
            WHEN cast(vl_pix AS double) < 5000 THEN 'D_1000_5000'
            WHEN cast(vl_pix AS double) < 10000 THEN 'E_5000_10000'
            ELSE 'F_10000_PLUS'
        END AS value_band,

        'EXP010F_R1_PARTIAL' AS normal_sample_source,
        'N0_R1_PARTIAL_REUSE' AS normal_sample_strategy,
        cast(1.0 AS double) AS normal_sample_weight,
        cast(0 AS int) AS matched_control_flag,
        cast(0 AS int) AS hard_negative_flag,
        cast(0 AS int) AS recent_normal_flag,
        CASE WHEN autdatref IS NOT NULL THEN 1 ELSE 0 END AS mbk_available_flag,
        autdatref,
        autdathorini,
        autcodret,
        device_name,
        app_version,
        ip_address,
        latencia_rede_ms,
        tempo_interacao_ms,
        tempo_processamento_host_ms,
        metodo_autenticacao,
        session_id,
        topaz_risk_score,
        topaz_transacao_rejeitada,
        is_agendamento_recorrente,
        mbk_completeness_score,
        cast(0 AS int) AS is_fraud,
        dt_carga
    FROM ${hivevar:DB_WORK}.tb_pix_normais_sample_mbk_180d_v2 r
    JOIN params p ON 1 = 1
    WHERE cast(r.data_pix AS date) BETWEEN p.dt_ini AND p.dt_fim
),

unioned AS (
    SELECT * FROM r2
    UNION ALL
    SELECT * FROM r1_partial
),

dedup AS (
    SELECT
        u.*,
        row_number() OVER (
            PARTITION BY cd_pix
            ORDER BY
                CASE normal_sample_source
                    WHEN 'EXP010F_R2_HUE_HIVE_ROLLING_180D' THEN 1
                    WHEN 'EXP010F_R2_HUE_HIVE' THEN 2
                    ELSE 3
                END,
                mbk_available_flag DESC,
                normal_sample_weight DESC
        ) AS rn
    FROM unioned u
)

SELECT
    cd_pix,
    cd_cpf_pagador,
    cd_cpf_cnpj_recebedor,
    vl_pix,
    dt_pix,
    data_pix,
    ds_chave_pix,
    ds_tipo_chave,
    hour,
    periodo_dia,
    value_band,
    normal_sample_source,
    normal_sample_strategy,
    normal_sample_weight,
    matched_control_flag,
    hard_negative_flag,
    recent_normal_flag,
    mbk_available_flag,
    autdatref,
    autdathorini,
    autcodret,
    device_name,
    app_version,
    ip_address,
    latencia_rede_ms,
    tempo_interacao_ms,
    tempo_processamento_host_ms,
    metodo_autenticacao,
    session_id,
    topaz_risk_score,
    topaz_transacao_rejeitada,
    is_agendamento_recorrente,
    mbk_completeness_score,
    is_fraud,
    dt_carga
FROM dedup
WHERE rn = 1
;
