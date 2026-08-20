-- =====================================================================
-- Tabela: hmo_ml.tb_pix_dataset_v2_180d_v1
-- EXP-010G — Unified Dataset Builder v2
-- Objetivo:
--   Criar dataset unificado model-ready para replay/treino shadow:
--     - positivos MAF confirmados dentro da janela movel de 180 dias;
--     - normais qualificados EXP-010F-R2 dentro da mesma janela;
--     - flags de origem/estrategia/cobertura MBK;
--     - pesos de amostragem;
--     - split temporal leakage-free.
--
-- Parametros esperados:
--   DB_WORK
--   WINDOW_DAYS
--   WINDOW_LAG_DAYS
-- =====================================================================

SET hive.execution.engine=tez;
SET hive.vectorized.execution.enabled=false;
SET hive.vectorized.execution.reduce.enabled=false;
SET hive.exec.compress.output=true;

DROP TABLE IF EXISTS ${hivevar:DB_WORK}.tb_pix_dataset_v2_180d_v1;

CREATE TABLE ${hivevar:DB_WORK}.tb_pix_dataset_v2_180d_v1
STORED AS PARQUET
AS
WITH params AS (
    SELECT
        date_sub(current_date, cast('${hivevar:WINDOW_LAG_DAYS}' AS int)) AS dt_fim,
        date_sub(
            current_date,
            cast('${hivevar:WINDOW_LAG_DAYS}' AS int) + cast('${hivevar:WINDOW_DAYS}' AS int) - 1
        ) AS dt_ini,
        date_sub(current_date, cast('${hivevar:WINDOW_LAG_DAYS}' AS int) + 60) AS dt_train_fim,
        date_sub(current_date, cast('${hivevar:WINDOW_LAG_DAYS}' AS int) + 59) AS dt_valid_ini,
        date_sub(current_date, cast('${hivevar:WINDOW_LAG_DAYS}' AS int) + 30) AS dt_valid_fim,
        date_sub(current_date, cast('${hivevar:WINDOW_LAG_DAYS}' AS int) + 29) AS dt_holdout_ini
),

maf_mbk_dedup AS (
    SELECT *
    FROM (
        SELECT
            h.*,
            row_number() OVER (
                PARTITION BY cd_pix
                ORDER BY
                    coalesce(mbk_completeness_score, -1) DESC,
                    autdatref DESC,
                    autdathorini DESC
            ) AS rn_mbk
        FROM ${hivevar:DB_WORK}.tb_pix_maf_mbk_hydration_180d_v2 h
        JOIN params p ON 1 = 1
        WHERE cast(h.dt_pix AS date) BETWEEN p.dt_ini AND p.dt_fim
    ) x
    WHERE rn_mbk = 1
),

positivos_maf AS (
    SELECT
        trim(cast(m.cd_pix AS string)) AS transaction_id,
        trim(cast(m.cd_pix AS string)) AS cd_pix,
        lpad(cast(m.cd_cpf_pagador AS string), 14, '0') AS customer_id,
        lpad(cast(m.cd_cpf_pagador AS string), 14, '0') AS cd_cpf_pagador,
        lpad(cast(m.cd_cpf_cnpj_recebedor AS string), 14, '0') AS counterparty_id,
        lpad(cast(m.cd_cpf_cnpj_recebedor AS string), 14, '0') AS cd_cpf_cnpj_recebedor,
        cast(cast(m.vl_pix AS string) AS double) AS vl_pix,
        cast(m.dt_pix AS timestamp) AS event_datetime,
        cast(m.dt_pix AS timestamp) AS dt_pix,
        cast(m.dt_pix AS date) AS data_pix,

        coalesce(cast(m.ds_chave_pix AS string), 'INFORMACAO_AUSENTE') AS ds_chave_pix,

        CASE
            WHEN m.ds_chave_pix IS NULL THEN 'INFORMACAO_AUSENTE'
            WHEN length(cast(m.ds_chave_pix AS string)) >= 32 THEN 'CHAVE_ALEATORIA'
            WHEN cast(m.ds_chave_pix AS string) LIKE '%@%' THEN 'EMAIL'
            WHEN cast(m.ds_chave_pix AS string) RLIKE '^[0-9]+$'
                 AND length(cast(m.ds_chave_pix AS string)) >= 11 THEN 'DOCUMENTO_TELEFONE'
            ELSE 'OUTROS'
        END AS ds_tipo_chave_norm,

        hour(cast(m.dt_pix AS timestamp)) AS hour,

        CASE
            WHEN hour(cast(m.dt_pix AS timestamp)) BETWEEN 0 AND 5 THEN 'madrugada'
            WHEN hour(cast(m.dt_pix AS timestamp)) BETWEEN 6 AND 11 THEN 'manha'
            WHEN hour(cast(m.dt_pix AS timestamp)) BETWEEN 12 AND 17 THEN 'tarde'
            ELSE 'noite'
        END AS periodo_dia,

        CASE
            WHEN cast(cast(m.vl_pix AS string) AS double) < 100 THEN 'A_000_100'
            WHEN cast(cast(m.vl_pix AS string) AS double) < 500 THEN 'B_100_500'
            WHEN cast(cast(m.vl_pix AS string) AS double) < 1000 THEN 'C_500_1000'
            WHEN cast(cast(m.vl_pix AS string) AS double) < 5000 THEN 'D_1000_5000'
            WHEN cast(cast(m.vl_pix AS string) AS double) < 10000 THEN 'E_5000_10000'
            ELSE 'F_10000_PLUS'
        END AS value_band,

        cast(1 AS int) AS is_fraud,
        'POSITIVE_FRAUD' AS dataset_role,
        'MAF_CONFIRMED_FRAUD_180D' AS source_dataset,
        'P1_MAF_RECENTE_180D' AS sample_strategy,
        cast(1.00 AS double) AS sample_weight,

        cast(0 AS int) AS matched_control_flag,
        cast(0 AS int) AS hard_negative_flag,
        cast(0 AS int) AS recent_normal_flag,
        cast(1 AS int) AS maf_recent_flag,
        cast(0 AS int) AS maf_historical_flag,

        CASE WHEN h.cd_pix IS NOT NULL AND h.autdatref IS NOT NULL THEN 1 ELSE 0 END AS mbk_available_flag,

        h.autdatref,
        h.autdathorini,
        h.autcodret,
        h.device_name,
        h.app_version,
        h.ip_address,
        h.latencia_rede_ms,
        h.tempo_interacao_ms,
        h.tempo_processamento_host_ms,
        h.metodo_autenticacao,
        h.session_id,
        h.topaz_risk_score,
        h.topaz_transacao_rejeitada,
        h.is_agendamento_recorrente,
        h.mbk_completeness_score,

        cast(m.label_status AS string) AS label_status,
        cast(m.model_scope_status AS string) AS model_scope_status,
        cast(m.bank_direction AS string) AS bank_direction,
        cast(m.triangulation_flag AS string) AS triangulation_flag,
        cast(m.duplicate_conflict_flag AS string) AS duplicate_conflict_flag,

        CASE
            WHEN cast(m.dt_pix AS date) <= p.dt_train_fim THEN 'TRAIN'
            WHEN cast(m.dt_pix AS date) BETWEEN p.dt_valid_ini AND p.dt_valid_fim THEN 'VALIDATION'
            WHEN cast(m.dt_pix AS date) BETWEEN p.dt_holdout_ini AND p.dt_fim THEN 'HOLDOUT'
            ELSE 'OUT_OF_WINDOW'
        END AS temporal_split,

        p.dt_ini AS window_start_date,
        p.dt_fim AS window_end_date,
        current_timestamp() AS dataset_created_at

    FROM ${hivevar:DB_WORK}.tb_pix_fraudes_maf_hidratadas_v1 m
    JOIN params p ON 1 = 1
    LEFT JOIN maf_mbk_dedup h
        ON trim(cast(m.cd_pix AS string)) = trim(cast(h.cd_pix AS string))
    WHERE cast(m.dt_pix AS date) BETWEEN p.dt_ini AND p.dt_fim
      AND trim(cast(m.cd_pix AS string)) IS NOT NULL
      AND length(trim(cast(m.cd_pix AS string))) > 0
      AND cast(m.is_fraud AS int) = 1
      AND cast(m.model_scope_status AS string) = 'POSITIVE_FOR_CURRENT_MODEL'
      AND cast(m.label_status AS string) = 'CONFIRMED_FRAUD_CANDIDATE'
),

normais AS (
    SELECT
        trim(cast(n.cd_pix AS string)) AS transaction_id,
        trim(cast(n.cd_pix AS string)) AS cd_pix,
        lpad(cast(n.cd_cpf_pagador AS string), 14, '0') AS customer_id,
        lpad(cast(n.cd_cpf_pagador AS string), 14, '0') AS cd_cpf_pagador,
        lpad(cast(n.cd_cpf_cnpj_recebedor AS string), 14, '0') AS counterparty_id,
        lpad(cast(n.cd_cpf_cnpj_recebedor AS string), 14, '0') AS cd_cpf_cnpj_recebedor,
        cast(n.vl_pix AS double) AS vl_pix,
        cast(n.dt_pix AS timestamp) AS event_datetime,
        cast(n.dt_pix AS timestamp) AS dt_pix,
        cast(n.data_pix AS date) AS data_pix,

        coalesce(cast(n.ds_chave_pix AS string), 'INFORMACAO_AUSENTE') AS ds_chave_pix,

        CASE
            WHEN upper(trim(cast(n.ds_tipo_chave AS string))) LIKE 'CHAVE%ALEATORIA%' THEN 'CHAVE_ALEATORIA'
            WHEN upper(trim(cast(n.ds_tipo_chave AS string))) LIKE 'DOCUMENTO%TELEFONE%' THEN 'DOCUMENTO_TELEFONE'
            WHEN upper(trim(cast(n.ds_tipo_chave AS string))) = 'EMAIL' THEN 'EMAIL'
            WHEN upper(trim(cast(n.ds_tipo_chave AS string))) LIKE 'INFORMA%AUSENTE%' THEN 'INFORMACAO_AUSENTE'
            WHEN n.ds_tipo_chave IS NULL THEN 'INFORMACAO_AUSENTE'
            ELSE 'OUTROS'
        END AS ds_tipo_chave_norm,

        n.hour,
        n.periodo_dia,
        n.value_band,

        cast(0 AS int) AS is_fraud,
        'NEGATIVE_NORMAL' AS dataset_role,
        cast(n.normal_sample_source AS string) AS source_dataset,
        cast(n.normal_sample_strategy AS string) AS sample_strategy,

        CASE
            WHEN n.normal_sample_strategy = 'N2_MATCHED_CONTROLS' THEN cast(0.80 AS double)
            WHEN n.normal_sample_strategy = 'N3_HARD_NEGATIVES' THEN cast(1.50 AS double)
            WHEN n.normal_sample_strategy = 'N4_RECENT_NORMALS' THEN cast(1.10 AS double)
            ELSE coalesce(cast(n.normal_sample_weight AS double), cast(1.00 AS double))
        END AS sample_weight,

        cast(n.matched_control_flag AS int) AS matched_control_flag,
        cast(n.hard_negative_flag AS int) AS hard_negative_flag,
        cast(n.recent_normal_flag AS int) AS recent_normal_flag,
        cast(0 AS int) AS maf_recent_flag,
        cast(0 AS int) AS maf_historical_flag,

        cast(n.mbk_available_flag AS int) AS mbk_available_flag,

        n.autdatref,
        n.autdathorini,
        n.autcodret,
        n.device_name,
        n.app_version,
        n.ip_address,
        n.latencia_rede_ms,
        n.tempo_interacao_ms,
        n.tempo_processamento_host_ms,
        n.metodo_autenticacao,
        n.session_id,
        n.topaz_risk_score,
        n.topaz_transacao_rejeitada,
        n.is_agendamento_recorrente,
        n.mbk_completeness_score,

        cast(NULL AS string) AS label_status,
        cast(NULL AS string) AS model_scope_status,
        cast(NULL AS string) AS bank_direction,
        cast(NULL AS string) AS triangulation_flag,
        cast(NULL AS string) AS duplicate_conflict_flag,

        CASE
            WHEN cast(n.data_pix AS date) <= p.dt_train_fim THEN 'TRAIN'
            WHEN cast(n.data_pix AS date) BETWEEN p.dt_valid_ini AND p.dt_valid_fim THEN 'VALIDATION'
            WHEN cast(n.data_pix AS date) BETWEEN p.dt_holdout_ini AND p.dt_fim THEN 'HOLDOUT'
            ELSE 'OUT_OF_WINDOW'
        END AS temporal_split,

        p.dt_ini AS window_start_date,
        p.dt_fim AS window_end_date,
        current_timestamp() AS dataset_created_at

    FROM ${hivevar:DB_WORK}.tb_pix_normais_dataset_ready_v1 n
    JOIN params p ON 1 = 1
    WHERE cast(n.data_pix AS date) BETWEEN p.dt_ini AND p.dt_fim
),

unioned AS (
    SELECT * FROM positivos_maf
    UNION ALL
    SELECT * FROM normais
),

dedup AS (
    SELECT
        u.*,
        row_number() OVER (
            PARTITION BY transaction_id
            ORDER BY
                is_fraud DESC,
                mbk_available_flag DESC,
                sample_weight DESC,
                dataset_created_at DESC
        ) AS rn
    FROM unioned u
)

SELECT
    transaction_id,
    cd_pix,
    customer_id,
    cd_cpf_pagador,
    counterparty_id,
    cd_cpf_cnpj_recebedor,
    vl_pix,
    event_datetime,
    dt_pix,
    data_pix,
    ds_chave_pix,
    ds_tipo_chave_norm,
    hour,
    periodo_dia,
    value_band,
    is_fraud,
    dataset_role,
    source_dataset,
    sample_strategy,
    sample_weight,
    matched_control_flag,
    hard_negative_flag,
    recent_normal_flag,
    maf_recent_flag,
    maf_historical_flag,
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
    label_status,
    model_scope_status,
    bank_direction,
    triangulation_flag,
    duplicate_conflict_flag,
    temporal_split,
    window_start_date,
    window_end_date,
    dataset_created_at
FROM dedup
WHERE rn = 1
;
