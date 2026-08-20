-- =====================================================================
-- EXP-010F-R2 — Normal Sampling Qualificado via Hue/Hive
-- Objetivo:
--   Criar uma base normal menor, estratificada e útil para treino,
--   evitando backfill exaustivo de 180 dias via CML/Spark.
-- =====================================================================

SET hive.execution.engine=tez;
SET hive.vectorized.execution.enabled=false;
SET hive.vectorized.execution.reduce.enabled=false;
SET hive.exec.dynamic.partition=true;
SET hive.exec.dynamic.partition.mode=nonstrict;

-- Ajustar quando a janela MAF mudar.
SET hivevar:DT_INI=2025-11-13;
SET hivevar:DT_FIM=2026-05-11;

-- =====================================================================
-- 0. Limpeza das tabelas do experimento
-- =====================================================================

DROP TABLE IF EXISTS hmo_ml.tb_pix_normais_qualified_raw_180d_v1;
DROP TABLE IF EXISTS hmo_ml.tb_pix_normais_qualified_sample_180d_v1;
DROP TABLE IF EXISTS hmo_ml.tb_pix_normais_qualified_sample_mbk_180d_v1;
DROP TABLE IF EXISTS hmo_ml.tb_pix_normais_dataset_ready_v1;
DROP TABLE IF EXISTS hmo_ml.tb_pix_normais_qualified_audit_v1;

-- =====================================================================
-- 1. Amostra bruta qualificada
--
-- Esta etapa já amostra cedo, no SQL.
-- Não materializa todos os normais de 180 dias.
-- =====================================================================

CREATE TABLE hmo_ml.tb_pix_normais_qualified_raw_180d_v1
STORED AS PARQUET
AS
WITH maf_keys AS (
    SELECT DISTINCT
        trim(cast(cd_pix AS string)) AS cd_pix
    FROM hmo_ml.tb_pix_fraudes_maf_hidratadas_v1
    WHERE cd_pix IS NOT NULL
),

maf_strata AS (
    SELECT
        CASE
            WHEN cast(vl_pix AS double) < 100 THEN 'A_000_100'
            WHEN cast(vl_pix AS double) < 500 THEN 'B_100_500'
            WHEN cast(vl_pix AS double) < 1000 THEN 'C_500_1000'
            WHEN cast(vl_pix AS double) < 5000 THEN 'D_1000_5000'
            WHEN cast(vl_pix AS double) < 10000 THEN 'E_5000_10000'
            ELSE 'F_10000_PLUS'
        END AS value_band,

        CASE
            WHEN hour(cast(dt_pix AS timestamp)) BETWEEN 0 AND 5 THEN 'madrugada'
            WHEN hour(cast(dt_pix AS timestamp)) BETWEEN 6 AND 11 THEN 'manha'
            WHEN hour(cast(dt_pix AS timestamp)) BETWEEN 12 AND 17 THEN 'tarde'
            ELSE 'noite'
        END AS periodo_dia,

        CASE
            WHEN ds_chave_pix IS NULL THEN 'INFORMACAO_AUSENTE'
            WHEN length(cast(ds_chave_pix AS string)) >= 32 THEN 'CHAVE_ALEATORIA'
            WHEN cast(ds_chave_pix AS string) LIKE '%@%' THEN 'EMAIL'
            WHEN cast(ds_chave_pix AS string) RLIKE '^[0-9]+$'
                 AND length(cast(ds_chave_pix AS string)) >= 11 THEN 'DOCUMENTO_TELEFONE'
            ELSE 'OUTROS'
        END AS ds_tipo_chave,

        count(*) AS n_frauds
    FROM hmo_ml.tb_pix_fraudes_maf_hidratadas_v1
    WHERE cast(dt_pix AS date) BETWEEN '${hivevar:DT_INI}' AND '${hivevar:DT_FIM}'
    GROUP BY
        CASE
            WHEN cast(vl_pix AS double) < 100 THEN 'A_000_100'
            WHEN cast(vl_pix AS double) < 500 THEN 'B_100_500'
            WHEN cast(vl_pix AS double) < 1000 THEN 'C_500_1000'
            WHEN cast(vl_pix AS double) < 5000 THEN 'D_1000_5000'
            WHEN cast(vl_pix AS double) < 10000 THEN 'E_5000_10000'
            ELSE 'F_10000_PLUS'
        END,
        CASE
            WHEN hour(cast(dt_pix AS timestamp)) BETWEEN 0 AND 5 THEN 'madrugada'
            WHEN hour(cast(dt_pix AS timestamp)) BETWEEN 6 AND 11 THEN 'manha'
            WHEN hour(cast(dt_pix AS timestamp)) BETWEEN 12 AND 17 THEN 'tarde'
            ELSE 'noite'
        END,
        CASE
            WHEN ds_chave_pix IS NULL THEN 'INFORMACAO_AUSENTE'
            WHEN length(cast(ds_chave_pix AS string)) >= 32 THEN 'CHAVE_ALEATORIA'
            WHEN cast(ds_chave_pix AS string) LIKE '%@%' THEN 'EMAIL'
            WHEN cast(ds_chave_pix AS string) RLIKE '^[0-9]+$'
                 AND length(cast(ds_chave_pix AS string)) >= 11 THEN 'DOCUMENTO_TELEFONE'
            ELSE 'OUTROS'
        END
),

registro_ok AS (
    SELECT DISTINCT
        trim(cast(ds_id_pix AS string)) AS cd_pix
    FROM landing_brb_oracle_blk.tb_registro_pix
    WHERE coalesce(cast(st_processamento_retorno AS string), '') <> 'RJCT'
),

base AS (
    SELECT
        trim(cast(t.ds_id_pix AS string)) AS cd_pix,
        lpad(cast(t.nr_cpf_cnpj_origem AS string), 14, '0') AS cd_cpf_pagador,
        lpad(cast(t.nr_cpf_cnpj_destino AS string), 14, '0') AS cd_cpf_cnpj_recebedor,
        cast(t.vl_pix AS double) AS vl_pix,
        cast(t.dt_pix AS timestamp) AS dt_pix,
        cast(t.dt_pix AS date) AS data_pix,
        coalesce(cast(t.ds_chave_pix AS string), 'INFORMACAO_AUSENTE') AS ds_chave_pix,

        CASE
            WHEN t.ds_chave_pix IS NULL THEN 'INFORMACAO_AUSENTE'
            WHEN length(cast(t.ds_chave_pix AS string)) >= 32 THEN 'CHAVE_ALEATORIA'
            WHEN cast(t.ds_chave_pix AS string) LIKE '%@%' THEN 'EMAIL'
            WHEN cast(t.ds_chave_pix AS string) RLIKE '^[0-9]+$'
                 AND length(cast(t.ds_chave_pix AS string)) >= 11 THEN 'DOCUMENTO_TELEFONE'
            ELSE 'OUTROS'
        END AS ds_tipo_chave,

        hour(cast(t.dt_pix AS timestamp)) AS hour,

        CASE
            WHEN hour(cast(t.dt_pix AS timestamp)) BETWEEN 0 AND 5 THEN 'madrugada'
            WHEN hour(cast(t.dt_pix AS timestamp)) BETWEEN 6 AND 11 THEN 'manha'
            WHEN hour(cast(t.dt_pix AS timestamp)) BETWEEN 12 AND 17 THEN 'tarde'
            ELSE 'noite'
        END AS periodo_dia,

        CASE
            WHEN cast(t.vl_pix AS double) < 100 THEN 'A_000_100'
            WHEN cast(t.vl_pix AS double) < 500 THEN 'B_100_500'
            WHEN cast(t.vl_pix AS double) < 1000 THEN 'C_500_1000'
            WHEN cast(t.vl_pix AS double) < 5000 THEN 'D_1000_5000'
            WHEN cast(t.vl_pix AS double) < 10000 THEN 'E_5000_10000'
            ELSE 'F_10000_PLUS'
        END AS value_band,

        pmod(crc32(trim(cast(t.ds_id_pix AS string))), 1000000) AS sample_hash

    FROM landing_brb_oracle_blk.tb_extrato_pix t
    INNER JOIN registro_ok r
        ON trim(cast(t.ds_id_pix AS string)) = r.cd_pix
    LEFT JOIN maf_keys f
        ON trim(cast(t.ds_id_pix AS string)) = f.cd_pix

    WHERE cast(t.cd_ispb_origem AS int) = 208
      AND t.nr_cpf_cnpj_origem <> t.nr_cpf_cnpj_destino
      AND cast(t.dt_pix AS date) BETWEEN '${hivevar:DT_INI}' AND '${hivevar:DT_FIM}'
      AND t.ds_id_pix IS NOT NULL
      AND length(trim(cast(t.ds_id_pix AS string))) > 0
      AND trim(cast(t.ds_id_pix AS string)) RLIKE '^E[A-Za-z0-9]{20,}$'
      AND f.cd_pix IS NULL
),

qualified AS (
    SELECT
        b.*,
        fs.n_frauds,

        CASE WHEN fs.n_frauds IS NOT NULL THEN 1 ELSE 0 END AS matched_control_candidate_flag,

        CASE
            WHEN b.vl_pix >= 10000 THEN 1
            WHEN b.ds_tipo_chave = 'CHAVE_ALEATORIA' THEN 1
            WHEN b.periodo_dia = 'madrugada' AND b.vl_pix >= 500 THEN 1
            WHEN b.value_band IN ('D_1000_5000', 'E_5000_10000', 'F_10000_PLUS') THEN 1
            ELSE 0
        END AS hard_negative_candidate_flag,

        CASE
            WHEN b.data_pix >= date_sub('${hivevar:DT_FIM}', 60) THEN 1
            ELSE 0
        END AS recent_candidate_flag

    FROM base b
    LEFT JOIN maf_strata fs
        ON b.value_band = fs.value_band
       AND b.periodo_dia = fs.periodo_dia
       AND b.ds_tipo_chave = fs.ds_tipo_chave
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
    n_frauds,
    matched_control_candidate_flag,
    hard_negative_candidate_flag,
    recent_candidate_flag,

    CASE
        WHEN matched_control_candidate_flag = 1 THEN 'N2_MATCHED_CONTROL_CANDIDATE'
        WHEN hard_negative_candidate_flag = 1 THEN 'N3_HARD_NEGATIVE_CANDIDATE'
        WHEN recent_candidate_flag = 1 THEN 'N4_RECENT_NORMAL_CANDIDATE'
        ELSE 'N1_BACKGROUND_CANDIDATE'
    END AS primary_candidate_strategy,

    current_timestamp() AS created_at

FROM qualified

WHERE
      -- Background leve: aproximadamente 0,03% da base elegível.
      sample_hash < 300

   OR -- Matched controls: mais agressivo, pois só entra em estratos parecidos com fraude.
      (matched_control_candidate_flag = 1 AND sample_hash < 80000)

   OR -- Hard negatives.
      (hard_negative_candidate_flag = 1 AND sample_hash < 30000)

   OR -- Reforço de recentes.
      (recent_candidate_flag = 1 AND sample_hash >= 300 AND sample_hash < 1200)
;

-- =====================================================================
-- 2. Amostra qualificada final, com cap por estratégia/estrato
-- =====================================================================

CREATE TABLE hmo_ml.tb_pix_normais_qualified_sample_180d_v1
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
    FROM hmo_ml.tb_pix_normais_qualified_raw_180d_v1 q
),

dedup AS (
    SELECT *
    FROM ranked
    WHERE rn_cd_pix = 1
),

final_sample AS (
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
            WHEN primary_candidate_strategy = 'N2_MATCHED_CONTROL_CANDIDATE' THEN 1.25
            WHEN primary_candidate_strategy = 'N3_HARD_NEGATIVE_CANDIDATE' THEN 1.50
            WHEN primary_candidate_strategy = 'N4_RECENT_NORMAL_CANDIDATE' THEN 1.10
            ELSE 1.00
        END AS normal_sample_weight,

        'EXP010F_R2_HUE_HIVE' AS normal_sample_source,
        0 AS is_fraud,
        current_timestamp() AS dt_carga

    FROM dedup

    WHERE
        -- Caps por estrato/estratégia. Ajustáveis após auditoria.
        (
            primary_candidate_strategy = 'N1_BACKGROUND_CANDIDATE'
            AND rn_strategy_stratum <= 2500
        )
        OR
        (
            primary_candidate_strategy = 'N2_MATCHED_CONTROL_CANDIDATE'
            AND rn_strategy_stratum <= 4000
        )
        OR
        (
            primary_candidate_strategy = 'N3_HARD_NEGATIVE_CANDIDATE'
            AND rn_strategy_stratum <= 3000
        )
        OR
        (
            primary_candidate_strategy = 'N4_RECENT_NORMAL_CANDIDATE'
            AND rn_strategy_stratum <= 2000
        )
)

SELECT *
FROM final_sample
;

-- =====================================================================
-- 3. Enriquecimento com MBK compacta
-- =====================================================================

CREATE TABLE hmo_ml.tb_pix_normais_qualified_sample_mbk_180d_v1
STORED AS PARQUET
AS
WITH mbk_dedup AS (
    SELECT *
    FROM (
        SELECT
            m.*,
            row_number() OVER (
                PARTITION BY end_to_end_id
                ORDER BY
                    coalesce(mbk_completeness_score, -1) DESC,
                    autdatref DESC,
                    autdathorini DESC
            ) AS rn_mbk
        FROM hmo_ml.tb_pix_mbk_compact_180d_v1 m
    ) x
    WHERE rn_mbk = 1
)

SELECT
    n.*,

    CASE WHEN m.end_to_end_id IS NOT NULL THEN 1 ELSE 0 END AS mbk_available_flag,

    m.autdatref,
    m.autdathorini,
    m.autcodret,
    m.device_name,
    m.app_version,
    m.ip_address,
    m.latencia_rede_ms,
    m.tempo_interacao_ms,
    m.tempo_processamento_host_ms,
    m.metodo_autenticacao,
    m.session_id,
    m.topaz_risk_score,
    m.topaz_transacao_rejeitada,
    m.is_agendamento_recorrente,
    m.mbk_completeness_score

FROM hmo_ml.tb_pix_normais_qualified_sample_180d_v1 n
LEFT JOIN mbk_dedup m
    ON n.cd_pix = m.end_to_end_id
;

-- =====================================================================
-- 4. Dataset normal ready: combina EXP-010F-R1 parcial + R2 qualificado
-- =====================================================================

CREATE TABLE hmo_ml.tb_pix_normais_dataset_ready_v1
STORED AS PARQUET
AS
WITH r2 AS (
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
    FROM hmo_ml.tb_pix_normais_qualified_sample_mbk_180d_v1
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
        ds_tipo_chave,
        hour,
        periodo_dia,
        value_band,
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
    FROM hmo_ml.tb_pix_normais_sample_mbk_180d_v2
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
                    WHEN 'EXP010F_R2_HUE_HIVE' THEN 1
                    ELSE 2
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

-- =====================================================================
-- 5. Auditoria consolidada
-- =====================================================================

CREATE TABLE hmo_ml.tb_pix_normais_qualified_audit_v1
STORED AS PARQUET
AS
SELECT
    'TOTAL_READY' AS metric_group,
    'ALL' AS metric_name,
    count(*) AS n_rows
FROM hmo_ml.tb_pix_normais_dataset_ready_v1

UNION ALL

SELECT
    'BY_STRATEGY' AS metric_group,
    normal_sample_strategy AS metric_name,
    count(*) AS n_rows
FROM hmo_ml.tb_pix_normais_dataset_ready_v1
GROUP BY normal_sample_strategy

UNION ALL

SELECT
    'BY_VALUE_BAND' AS metric_group,
    value_band AS metric_name,
    count(*) AS n_rows
FROM hmo_ml.tb_pix_normais_dataset_ready_v1
GROUP BY value_band

UNION ALL

SELECT
    'BY_KEY_TYPE' AS metric_group,
    ds_tipo_chave AS metric_name,
    count(*) AS n_rows
FROM hmo_ml.tb_pix_normais_dataset_ready_v1
GROUP BY ds_tipo_chave

UNION ALL

SELECT
    'BY_PERIOD' AS metric_group,
    periodo_dia AS metric_name,
    count(*) AS n_rows
FROM hmo_ml.tb_pix_normais_dataset_ready_v1
GROUP BY periodo_dia

UNION ALL

SELECT
    'MBK_COVERAGE' AS metric_group,
    cast(mbk_available_flag AS string) AS metric_name,
    count(*) AS n_rows
FROM hmo_ml.tb_pix_normais_dataset_ready_v1
GROUP BY mbk_available_flag
;

-- =====================================================================
-- 6. Auditoria anti-overlap com MAF
-- =====================================================================

DROP TABLE IF EXISTS hmo_ml.tb_pix_normais_qualified_overlap_maf_audit_v1;

CREATE TABLE hmo_ml.tb_pix_normais_qualified_overlap_maf_audit_v1
STORED AS PARQUET
AS
SELECT
    n.cd_pix,
    n.dt_pix,
    n.vl_pix,
    n.normal_sample_source,
    n.normal_sample_strategy,
    m.label_status,
    m.model_scope_status
FROM hmo_ml.tb_pix_normais_dataset_ready_v1 n
INNER JOIN hmo_ml.tb_pix_fraudes_maf_hidratadas_v1 m
    ON n.cd_pix = m.cd_pix
;

-- Esperado: 0 linhas.