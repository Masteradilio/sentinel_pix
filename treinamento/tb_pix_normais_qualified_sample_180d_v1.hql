-- =====================================================================
-- Tabela: hmo_ml.tb_pix_normais_qualified_sample_180d_v1
-- EXP-010F-R2 — etapa 02
-- Objetivo:
--   Deduplicar e aplicar caps por estrategia/estrato.
--   Versao rolling: consome RAW ja limitado aos ultimos WINDOW_DAYS.
-- =====================================================================

SET hive.execution.engine=tez;
SET hive.vectorized.execution.enabled=false;
SET hive.vectorized.execution.reduce.enabled=false;
SET hive.exec.compress.output=true;

DROP TABLE IF EXISTS ${hivevar:DB_WORK}.tb_pix_normais_qualified_sample_180d_v1;

CREATE TABLE ${hivevar:DB_WORK}.tb_pix_normais_qualified_sample_180d_v1
STORED AS PARQUET
AS
WITH ranked AS (
    SELECT
        q.*,

        row_number() OVER (
            PARTITION BY primary_candidate_strategy, value_band, periodo_dia, ds_tipo_chave
            ORDER BY sample_hash ASC
        ) AS rn_strategy_stratum,

        row_number() OVER (
            PARTITION BY cd_pix
            ORDER BY
                CASE primary_candidate_strategy
                    WHEN 'N2_MATCHED_CONTROL_CANDIDATE' THEN 1
                    WHEN 'N3_HARD_NEGATIVE_CANDIDATE' THEN 2
                    WHEN 'N4_RECENT_NORMAL_CANDIDATE' THEN 3
                    ELSE 4
                END,
                sample_hash ASC
        ) AS rn_cd_pix

    FROM ${hivevar:DB_WORK}.tb_pix_normais_qualified_raw_180d_v1 q
),

dedup AS (
    SELECT *
    FROM ranked
    WHERE rn_cd_pix = 1
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
    sample_hash,

    CASE primary_candidate_strategy
        WHEN 'N2_MATCHED_CONTROL_CANDIDATE' THEN 'N2_MATCHED_CONTROLS'
        WHEN 'N3_HARD_NEGATIVE_CANDIDATE' THEN 'N3_HARD_NEGATIVES'
        WHEN 'N4_RECENT_NORMAL_CANDIDATE' THEN 'N4_RECENT_NORMALS'
        ELSE 'N1_BACKGROUND_NORMAL'
    END AS normal_sample_strategy,

    CASE
        WHEN primary_candidate_strategy = 'N2_MATCHED_CONTROL_CANDIDATE' THEN 1
        ELSE 0
    END AS matched_control_flag,

    CASE
        WHEN primary_candidate_strategy = 'N3_HARD_NEGATIVE_CANDIDATE' THEN 1
        ELSE 0
    END AS hard_negative_flag,

    CASE
        WHEN primary_candidate_strategy = 'N4_RECENT_NORMAL_CANDIDATE' THEN 1
        ELSE 0
    END AS recent_normal_flag,

    CASE
        -- N2 domina naturalmente a amostra; peso menor evita super-representacao no treino.
        WHEN primary_candidate_strategy = 'N2_MATCHED_CONTROL_CANDIDATE' THEN cast(0.80 AS double)
        WHEN primary_candidate_strategy = 'N3_HARD_NEGATIVE_CANDIDATE' THEN cast(1.50 AS double)
        WHEN primary_candidate_strategy = 'N4_RECENT_NORMAL_CANDIDATE' THEN cast(1.10 AS double)
        ELSE cast(1.00 AS double)
    END AS normal_sample_weight,

    'EXP010F_R2_HUE_HIVE_ROLLING_180D' AS normal_sample_source,
    0 AS is_fraud,
    current_timestamp() AS dt_carga

FROM dedup
WHERE
    (
        primary_candidate_strategy = 'N1_BACKGROUND_CANDIDATE'
        AND rn_strategy_stratum <= 5000
    )
    OR
    (
        primary_candidate_strategy = 'N2_MATCHED_CONTROL_CANDIDATE'
        AND rn_strategy_stratum <= 1600
    )
    OR
    (
        primary_candidate_strategy = 'N3_HARD_NEGATIVE_CANDIDATE'
        AND rn_strategy_stratum <= 3500
    )
    OR
    (
        primary_candidate_strategy = 'N4_RECENT_NORMAL_CANDIDATE'
        AND rn_strategy_stratum <= 2500
    )
;
