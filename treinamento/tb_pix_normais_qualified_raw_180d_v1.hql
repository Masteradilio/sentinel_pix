-- =====================================================================
-- Tabela: hmo_ml.tb_pix_normais_qualified_raw_180d_v1
-- EXP-010F-R2 — etapa 01
-- Objetivo:
--   Criar candidatos normais qualificados com amostragem cedo no Hive.
--   Versao rolling: sempre usa os ultimos WINDOW_DAYS, com fim em D-WINDOW_LAG_DAYS.
-- Parametros esperados:
--   DB_WORK
--   WINDOW_DAYS       default recomendado no workflow: 180
--   WINDOW_LAG_DAYS   default recomendado no workflow: 1
-- =====================================================================

SET hive.execution.engine=tez;
SET hive.vectorized.execution.enabled=false;
SET hive.vectorized.execution.reduce.enabled=false;
SET hive.exec.dynamic.partition=true;
SET hive.exec.dynamic.partition.mode=nonstrict;
SET hive.exec.compress.output=true;

DROP TABLE IF EXISTS ${hivevar:DB_WORK}.tb_pix_normais_qualified_raw_180d_v1;

CREATE TABLE ${hivevar:DB_WORK}.tb_pix_normais_qualified_raw_180d_v1
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

maf_keys AS (
    SELECT DISTINCT
        trim(cast(cd_pix AS string)) AS cd_pix
    FROM ${hivevar:DB_WORK}.tb_pix_fraudes_maf_hidratadas_v1
    WHERE cd_pix IS NOT NULL
      AND length(trim(cast(cd_pix AS string))) > 0
),

maf_strata AS (
    SELECT
        CASE
            WHEN cast(cast(m.vl_pix AS string) AS double) < 100 THEN 'A_000_100'
            WHEN cast(cast(m.vl_pix AS string) AS double) < 500 THEN 'B_100_500'
            WHEN cast(cast(m.vl_pix AS string) AS double) < 1000 THEN 'C_500_1000'
            WHEN cast(cast(m.vl_pix AS string) AS double) < 5000 THEN 'D_1000_5000'
            WHEN cast(cast(m.vl_pix AS string) AS double) < 10000 THEN 'E_5000_10000'
            ELSE 'F_10000_PLUS'
        END AS value_band,

        CASE
            WHEN hour(cast(m.dt_pix AS timestamp)) BETWEEN 0 AND 5 THEN 'madrugada'
            WHEN hour(cast(m.dt_pix AS timestamp)) BETWEEN 6 AND 11 THEN 'manha'
            WHEN hour(cast(m.dt_pix AS timestamp)) BETWEEN 12 AND 17 THEN 'tarde'
            ELSE 'noite'
        END AS periodo_dia,

        CASE
            WHEN m.ds_chave_pix IS NULL THEN 'INFORMACAO_AUSENTE'
            WHEN length(cast(m.ds_chave_pix AS string)) >= 32 THEN 'CHAVE_ALEATORIA'
            WHEN cast(m.ds_chave_pix AS string) LIKE '%@%' THEN 'EMAIL'
            WHEN cast(m.ds_chave_pix AS string) RLIKE '^[0-9]+$'
                 AND length(cast(m.ds_chave_pix AS string)) >= 11 THEN 'DOCUMENTO_TELEFONE'
            ELSE 'OUTROS'
        END AS ds_tipo_chave,

        count(*) AS n_frauds
    FROM ${hivevar:DB_WORK}.tb_pix_fraudes_maf_hidratadas_v1 m
    JOIN params p ON 1 = 1
    WHERE cast(m.dt_pix AS date) BETWEEN p.dt_ini AND p.dt_fim
    GROUP BY
        CASE
            WHEN cast(cast(m.vl_pix AS string) AS double) < 100 THEN 'A_000_100'
            WHEN cast(cast(m.vl_pix AS string) AS double) < 500 THEN 'B_100_500'
            WHEN cast(cast(m.vl_pix AS string) AS double) < 1000 THEN 'C_500_1000'
            WHEN cast(cast(m.vl_pix AS string) AS double) < 5000 THEN 'D_1000_5000'
            WHEN cast(cast(m.vl_pix AS string) AS double) < 10000 THEN 'E_5000_10000'
            ELSE 'F_10000_PLUS'
        END,
        CASE
            WHEN hour(cast(m.dt_pix AS timestamp)) BETWEEN 0 AND 5 THEN 'madrugada'
            WHEN hour(cast(m.dt_pix AS timestamp)) BETWEEN 6 AND 11 THEN 'manha'
            WHEN hour(cast(m.dt_pix AS timestamp)) BETWEEN 12 AND 17 THEN 'tarde'
            ELSE 'noite'
        END,
        CASE
            WHEN m.ds_chave_pix IS NULL THEN 'INFORMACAO_AUSENTE'
            WHEN length(cast(m.ds_chave_pix AS string)) >= 32 THEN 'CHAVE_ALEATORIA'
            WHEN cast(m.ds_chave_pix AS string) LIKE '%@%' THEN 'EMAIL'
            WHEN cast(m.ds_chave_pix AS string) RLIKE '^[0-9]+$'
                 AND length(cast(m.ds_chave_pix AS string)) >= 11 THEN 'DOCUMENTO_TELEFONE'
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
        cast(cast(t.vl_pix AS string) AS double) AS vl_pix,
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
            WHEN cast(cast(t.vl_pix AS string) AS double) < 100 THEN 'A_000_100'
            WHEN cast(cast(t.vl_pix AS string) AS double) < 500 THEN 'B_100_500'
            WHEN cast(cast(t.vl_pix AS string) AS double) < 1000 THEN 'C_500_1000'
            WHEN cast(cast(t.vl_pix AS string) AS double) < 5000 THEN 'D_1000_5000'
            WHEN cast(cast(t.vl_pix AS string) AS double) < 10000 THEN 'E_5000_10000'
            ELSE 'F_10000_PLUS'
        END AS value_band,

        pmod(crc32(trim(cast(t.ds_id_pix AS string))), 1000000) AS sample_hash

    FROM landing_brb_oracle_blk.tb_extrato_pix t
    JOIN params p ON 1 = 1
    INNER JOIN registro_ok r
        ON trim(cast(t.ds_id_pix AS string)) = r.cd_pix
    LEFT JOIN maf_keys f
        ON trim(cast(t.ds_id_pix AS string)) = f.cd_pix
    WHERE cast(t.cd_ispb_origem AS int) = 208
      AND t.nr_cpf_cnpj_origem <> t.nr_cpf_cnpj_destino
      AND cast(t.dt_pix AS date) BETWEEN p.dt_ini AND p.dt_fim
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
            WHEN b.data_pix >= date_sub(p.dt_fim, 60) THEN 1
            ELSE 0
        END AS recent_candidate_flag

    FROM base b
    JOIN params p ON 1 = 1
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
      -- Background leve.
      sample_hash < 1000

   OR -- Matched controls: reduzido em relacao ao primeiro historico para evitar dominancia excessiva.
      (matched_control_candidate_flag = 1 AND sample_hash < 35000)

   OR -- Hard negatives: um pouco mais forte para preservar casos dificeis.
      (hard_negative_candidate_flag = 1 AND sample_hash < 60000)

   OR -- Reforco de recentes.
      (recent_candidate_flag = 1 AND sample_hash >= 1000 AND sample_hash < 5000)
;
