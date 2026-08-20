-- =====================================================================
-- Tabela: hmo_ml.tb_pix_normais_qualified_overlap_maf_audit_v1
-- EXP-010F-R2 — etapa 06
-- Objetivo:
--   Auditar se algum normal final tem overlap com fraude MAF.
-- Esperado:
--   0 linhas.
-- =====================================================================

SET hive.execution.engine=tez;
SET hive.vectorized.execution.enabled=false;
SET hive.vectorized.execution.reduce.enabled=false;
SET hive.exec.compress.output=true;

DROP TABLE IF EXISTS ${hivevar:DB_WORK}.tb_pix_normais_qualified_overlap_maf_audit_v1;

CREATE TABLE ${hivevar:DB_WORK}.tb_pix_normais_qualified_overlap_maf_audit_v1
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
FROM ${hivevar:DB_WORK}.tb_pix_normais_dataset_ready_v1 n
INNER JOIN ${hivevar:DB_WORK}.tb_pix_fraudes_maf_hidratadas_v1 m
    ON trim(cast(n.cd_pix AS string)) = trim(cast(m.cd_pix AS string))
;
