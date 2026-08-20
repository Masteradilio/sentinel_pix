-- =====================================================================
-- EXP-012A-R1 Stage 2/3
-- Tabela: ${DB_WORK}.tb_pix_dataset_v3_daily_agg_180d_v1
-- Agregados diarios reais em 180d por pagador, par e recebedor.
-- =====================================================================

SET hive.execution.engine=tez;
SET hive.vectorized.execution.enabled=false;
SET hive.vectorized.execution.reduce.enabled=false;
SET hive.exec.compress.output=true;
SET hive.exec.parallel=true;

DROP TABLE IF EXISTS ${hivevar:DB_WORK}.tb_pix_dataset_v3_daily_agg_180d_v1;

CREATE TABLE ${hivevar:DB_WORK}.tb_pix_dataset_v3_daily_agg_180d_v1
STORED AS PARQUET
AS
WITH params AS (
    SELECT
        date_sub(current_date, cast('${hivevar:WINDOW_LAG_DAYS}' AS int)) AS dt_fim,
        date_sub(current_date, cast('${hivevar:WINDOW_LAG_DAYS}' AS int) + cast('${hivevar:WINDOW_DAYS}' AS int) - 1) AS dt_ini
),
target_customers AS (
    SELECT DISTINCT customer_id AS id_doc
    FROM ${hivevar:DB_WORK}.tb_pix_dataset_v3_target_180d_v1
    WHERE customer_id IS NOT NULL AND length(trim(customer_id)) > 0
),
target_receivers AS (
    SELECT DISTINCT counterparty_id AS id_doc
    FROM ${hivevar:DB_WORK}.tb_pix_dataset_v3_target_180d_v1
    WHERE counterparty_id IS NOT NULL AND length(trim(counterparty_id)) > 0
),
pix_base AS (
    SELECT
        trim(cast(t.ds_id_pix AS string)) AS cd_pix,
        lpad(cast(t.nr_cpf_cnpj_origem AS string), 14, '0') AS payer_id,
        lpad(cast(t.nr_cpf_cnpj_destino AS string), 14, '0') AS receiver_id,
        cast(cast(t.vl_pix AS string) AS double) AS vl_pix,
        cast(t.dt_pix AS date) AS data_pix
    FROM landing_brb_oracle_blk.tb_extrato_pix t
    JOIN params p ON 1=1
    WHERE cast(t.cd_ispb_origem AS int) = 208
      AND t.nr_cpf_cnpj_origem <> t.nr_cpf_cnpj_destino
      AND cast(t.dt_pix AS date) BETWEEN p.dt_ini AND p.dt_fim
      AND t.ds_id_pix IS NOT NULL
      AND trim(cast(t.ds_id_pix AS string)) RLIKE '^E[A-Za-z0-9]{20,}$'
),
pix_filtered AS (
    SELECT u.* FROM pix_base u LEFT SEMI JOIN target_customers c ON u.payer_id = c.id_doc
    UNION ALL
    SELECT u.* FROM pix_base u LEFT SEMI JOIN target_receivers r ON u.receiver_id = r.id_doc
),
pix_dedup AS (
    SELECT cd_pix, payer_id, receiver_id, vl_pix, data_pix
    FROM (
        SELECT u.*, row_number() OVER (PARTITION BY cd_pix ORDER BY data_pix DESC) AS rn
        FROM pix_filtered u
    ) x
    WHERE rn = 1
)
SELECT
    'PAYER' AS feature_scope,
    payer_id AS entity_key_1,
    cast(NULL AS string) AS entity_key_2,
    data_pix,
    count(*) AS qtd_pix,
    sum(vl_pix) AS valor_total,
    avg(vl_pix) AS valor_medio,
    max(vl_pix) AS valor_maximo,
    count(DISTINCT receiver_id) AS qtd_distintos
FROM pix_dedup
GROUP BY payer_id, data_pix

UNION ALL

SELECT
    'PAIR' AS feature_scope,
    payer_id AS entity_key_1,
    receiver_id AS entity_key_2,
    data_pix,
    count(*) AS qtd_pix,
    sum(vl_pix) AS valor_total,
    avg(vl_pix) AS valor_medio,
    max(vl_pix) AS valor_maximo,
    cast(0 AS bigint) AS qtd_distintos
FROM pix_dedup
GROUP BY payer_id, receiver_id, data_pix

UNION ALL

SELECT
    'RECEIVER' AS feature_scope,
    receiver_id AS entity_key_1,
    cast(NULL AS string) AS entity_key_2,
    data_pix,
    count(*) AS qtd_pix,
    sum(vl_pix) AS valor_total,
    avg(vl_pix) AS valor_medio,
    max(vl_pix) AS valor_maximo,
    count(DISTINCT payer_id) AS qtd_distintos
FROM pix_dedup
GROUP BY receiver_id, data_pix
;
