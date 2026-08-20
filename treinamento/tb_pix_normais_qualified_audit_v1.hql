-- =====================================================================
-- Tabela: hmo_ml.tb_pix_normais_qualified_audit_v1
-- EXP-010F-R2 — etapa 05
-- Objetivo:
--   Gerar auditoria consolidada da base normal final rolling 180d.
-- =====================================================================

SET hive.execution.engine=tez;
SET hive.vectorized.execution.enabled=false;
SET hive.vectorized.execution.reduce.enabled=false;
SET hive.exec.compress.output=true;

DROP TABLE IF EXISTS ${hivevar:DB_WORK}.tb_pix_normais_qualified_audit_v1;

CREATE TABLE ${hivevar:DB_WORK}.tb_pix_normais_qualified_audit_v1
STORED AS PARQUET
AS
SELECT
    'TOTAL_READY' AS metric_group,
    'ALL' AS metric_name,
    count(*) AS n_rows
FROM ${hivevar:DB_WORK}.tb_pix_normais_dataset_ready_v1

UNION ALL

SELECT
    'DISTINCT_CD_PIX' AS metric_group,
    'ALL' AS metric_name,
    count(DISTINCT cd_pix) AS n_rows
FROM ${hivevar:DB_WORK}.tb_pix_normais_dataset_ready_v1

UNION ALL

SELECT
    'DUPLICATED_CD_PIX' AS metric_group,
    'ALL' AS metric_name,
    count(*) - count(DISTINCT cd_pix) AS n_rows
FROM ${hivevar:DB_WORK}.tb_pix_normais_dataset_ready_v1

UNION ALL

SELECT
    'DATE_MIN' AS metric_group,
    cast(min(data_pix) AS string) AS metric_name,
    cast(0 AS bigint) AS n_rows
FROM ${hivevar:DB_WORK}.tb_pix_normais_dataset_ready_v1

UNION ALL

SELECT
    'DATE_MAX' AS metric_group,
    cast(max(data_pix) AS string) AS metric_name,
    cast(0 AS bigint) AS n_rows
FROM ${hivevar:DB_WORK}.tb_pix_normais_dataset_ready_v1

UNION ALL

SELECT
    'DAYS_WITH_DATA' AS metric_group,
    'ALL' AS metric_name,
    count(DISTINCT data_pix) AS n_rows
FROM ${hivevar:DB_WORK}.tb_pix_normais_dataset_ready_v1

UNION ALL

SELECT
    'BY_SOURCE' AS metric_group,
    normal_sample_source AS metric_name,
    count(*) AS n_rows
FROM ${hivevar:DB_WORK}.tb_pix_normais_dataset_ready_v1
GROUP BY normal_sample_source

UNION ALL

SELECT
    'BY_STRATEGY' AS metric_group,
    normal_sample_strategy AS metric_name,
    count(*) AS n_rows
FROM ${hivevar:DB_WORK}.tb_pix_normais_dataset_ready_v1
GROUP BY normal_sample_strategy

UNION ALL

SELECT
    'BY_VALUE_BAND' AS metric_group,
    value_band AS metric_name,
    count(*) AS n_rows
FROM ${hivevar:DB_WORK}.tb_pix_normais_dataset_ready_v1
GROUP BY value_band

UNION ALL

SELECT
    'BY_KEY_TYPE' AS metric_group,
    ds_tipo_chave AS metric_name,
    count(*) AS n_rows
FROM ${hivevar:DB_WORK}.tb_pix_normais_dataset_ready_v1
GROUP BY ds_tipo_chave

UNION ALL

SELECT
    'BY_PERIOD' AS metric_group,
    periodo_dia AS metric_name,
    count(*) AS n_rows
FROM ${hivevar:DB_WORK}.tb_pix_normais_dataset_ready_v1
GROUP BY periodo_dia

UNION ALL

SELECT
    'MBK_COVERAGE' AS metric_group,
    cast(mbk_available_flag AS string) AS metric_name,
    count(*) AS n_rows
FROM ${hivevar:DB_WORK}.tb_pix_normais_dataset_ready_v1
GROUP BY mbk_available_flag

UNION ALL

SELECT
    'FLAGS_MATCHED_CONTROL' AS metric_group,
    cast(matched_control_flag AS string) AS metric_name,
    count(*) AS n_rows
FROM ${hivevar:DB_WORK}.tb_pix_normais_dataset_ready_v1
GROUP BY matched_control_flag

UNION ALL

SELECT
    'FLAGS_HARD_NEGATIVE' AS metric_group,
    cast(hard_negative_flag AS string) AS metric_name,
    count(*) AS n_rows
FROM ${hivevar:DB_WORK}.tb_pix_normais_dataset_ready_v1
GROUP BY hard_negative_flag
;
