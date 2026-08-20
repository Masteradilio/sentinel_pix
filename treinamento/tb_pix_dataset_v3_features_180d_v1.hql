-- =====================================================================
-- EXP-012A-R1 Stage 3/3
-- Tabela final: ${DB_WORK}.tb_pix_dataset_v3_features_180d_v1
-- Junta target + agregados diarios. Histórico até D-1, leakage-safe.
-- =====================================================================

SET hive.execution.engine=tez;
SET hive.vectorized.execution.enabled=false;
SET hive.vectorized.execution.reduce.enabled=false;
SET hive.exec.compress.output=true;
SET hive.exec.parallel=true;

DROP TABLE IF EXISTS ${hivevar:DB_WORK}.tb_pix_dataset_v3_features_180d_v1;

CREATE TABLE ${hivevar:DB_WORK}.tb_pix_dataset_v3_features_180d_v1
STORED AS PARQUET
AS
WITH target AS (
    SELECT * FROM ${hivevar:DB_WORK}.tb_pix_dataset_v3_target_180d_v1
),
payer_roll AS (
    SELECT
        t.transaction_id,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 7 THEN d.qtd_pix ELSE 0 END) AS qtd_pix_pagador_7d,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 30 THEN d.qtd_pix ELSE 0 END) AS qtd_pix_pagador_30d,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 90 THEN d.qtd_pix ELSE 0 END) AS qtd_pix_pagador_90d,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 180 THEN d.qtd_pix ELSE 0 END) AS qtd_pix_pagador_180d,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 7 THEN d.valor_total ELSE 0 END) AS valor_total_pagador_7d,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 30 THEN d.valor_total ELSE 0 END) AS valor_total_pagador_30d,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 90 THEN d.valor_total ELSE 0 END) AS valor_total_pagador_90d,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 180 THEN d.valor_total ELSE 0 END) AS valor_total_pagador_180d,
        max(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 7 THEN d.qtd_pix ELSE 0 END) AS max_qtd_pix_dia_pagador_7d,
        max(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 30 THEN d.qtd_pix ELSE 0 END) AS max_qtd_pix_dia_pagador_30d,
        max(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 180 THEN d.valor_maximo ELSE NULL END) AS valor_maximo_pix_pagador_180d,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 180 THEN d.qtd_distintos ELSE 0 END) AS soma_recebedores_distintos_dia_180d
    FROM target t
    LEFT JOIN ${hivevar:DB_WORK}.tb_pix_dataset_v3_daily_agg_180d_v1 d
        ON d.feature_scope = 'PAYER'
       AND d.entity_key_1 = t.customer_id
       AND d.data_pix BETWEEN date_sub(t.data_pix, 180) AND date_sub(t.data_pix, 1)
    GROUP BY t.transaction_id
),
pair_roll AS (
    SELECT
        t.transaction_id,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 30 THEN d.qtd_pix ELSE 0 END) AS qtd_pix_mesmo_recebedor_30d,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 90 THEN d.qtd_pix ELSE 0 END) AS qtd_pix_mesmo_recebedor_90d,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 180 THEN d.qtd_pix ELSE 0 END) AS qtd_pix_mesmo_recebedor_180d,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 30 THEN d.valor_total ELSE 0 END) AS valor_total_para_recebedor_30d,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 90 THEN d.valor_total ELSE 0 END) AS valor_total_para_recebedor_90d,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 180 THEN d.valor_total ELSE 0 END) AS valor_total_para_recebedor_180d,
        min(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 180 THEN d.data_pix ELSE NULL END) AS primeira_data_envio_recebedor_180d
    FROM target t
    LEFT JOIN ${hivevar:DB_WORK}.tb_pix_dataset_v3_daily_agg_180d_v1 d
        ON d.feature_scope = 'PAIR'
       AND d.entity_key_1 = t.customer_id
       AND d.entity_key_2 = t.counterparty_id
       AND d.data_pix BETWEEN date_sub(t.data_pix, 180) AND date_sub(t.data_pix, 1)
    GROUP BY t.transaction_id
),
receiver_roll AS (
    SELECT
        t.transaction_id,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 30 THEN d.qtd_pix ELSE 0 END) AS qtd_pix_recebidos_30d,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 90 THEN d.qtd_pix ELSE 0 END) AS qtd_pix_recebidos_90d,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 180 THEN d.qtd_pix ELSE 0 END) AS qtd_pix_recebidos_180d,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 30 THEN d.valor_total ELSE 0 END) AS valor_total_recebido_30d,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 90 THEN d.valor_total ELSE 0 END) AS valor_total_recebido_90d,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 180 THEN d.valor_total ELSE 0 END) AS valor_total_recebido_180d,
        sum(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 180 THEN d.qtd_distintos ELSE 0 END) AS soma_pagadores_distintos_dia_recebedor_180d,
        max(CASE WHEN datediff(t.data_pix, d.data_pix) BETWEEN 1 AND 180 THEN d.qtd_pix ELSE 0 END) AS max_qtd_pix_recebidos_dia_180d
    FROM target t
    LEFT JOIN ${hivevar:DB_WORK}.tb_pix_dataset_v3_daily_agg_180d_v1 d
        ON d.feature_scope = 'RECEIVER'
       AND d.entity_key_1 = t.counterparty_id
       AND d.data_pix BETWEEN date_sub(t.data_pix, 180) AND date_sub(t.data_pix, 1)
    GROUP BY t.transaction_id
)
SELECT
    t.*,
    coalesce(p.qtd_pix_pagador_7d, 0) AS qtd_pix_pagador_7d,
    coalesce(p.qtd_pix_pagador_30d, 0) AS qtd_pix_pagador_30d,
    coalesce(p.qtd_pix_pagador_90d, 0) AS qtd_pix_pagador_90d,
    coalesce(p.qtd_pix_pagador_180d, 0) AS qtd_pix_pagador_180d,
    coalesce(p.valor_total_pagador_7d, 0.0) AS valor_total_pagador_7d,
    coalesce(p.valor_total_pagador_30d, 0.0) AS valor_total_pagador_30d,
    coalesce(p.valor_total_pagador_90d, 0.0) AS valor_total_pagador_90d,
    coalesce(p.valor_total_pagador_180d, 0.0) AS valor_total_pagador_180d,
    coalesce(p.max_qtd_pix_dia_pagador_7d, 0) AS max_qtd_pix_dia_pagador_7d,
    coalesce(p.max_qtd_pix_dia_pagador_30d, 0) AS max_qtd_pix_dia_pagador_30d,
    p.valor_maximo_pix_pagador_180d,
    coalesce(p.soma_recebedores_distintos_dia_180d, 0) AS soma_recebedores_distintos_dia_180d,
    coalesce(pair.qtd_pix_mesmo_recebedor_30d, 0) AS qtd_pix_mesmo_recebedor_30d,
    coalesce(pair.qtd_pix_mesmo_recebedor_90d, 0) AS qtd_pix_mesmo_recebedor_90d,
    coalesce(pair.qtd_pix_mesmo_recebedor_180d, 0) AS qtd_pix_mesmo_recebedor_180d,
    coalesce(pair.valor_total_para_recebedor_30d, 0.0) AS valor_total_para_recebedor_30d,
    coalesce(pair.valor_total_para_recebedor_90d, 0.0) AS valor_total_para_recebedor_90d,
    coalesce(pair.valor_total_para_recebedor_180d, 0.0) AS valor_total_para_recebedor_180d,
    pair.primeira_data_envio_recebedor_180d,
    CASE WHEN coalesce(pair.qtd_pix_mesmo_recebedor_180d, 0) = 0 THEN 1 ELSE 0 END AS primeiro_envio_para_recebedor_180d,
    CASE WHEN pair.primeira_data_envio_recebedor_180d IS NULL THEN NULL ELSE datediff(t.data_pix, pair.primeira_data_envio_recebedor_180d) END AS dias_desde_primeiro_envio_recebedor,
    coalesce(r.qtd_pix_recebidos_30d, 0) AS qtd_pix_recebidos_30d,
    coalesce(r.qtd_pix_recebidos_90d, 0) AS qtd_pix_recebidos_90d,
    coalesce(r.qtd_pix_recebidos_180d, 0) AS qtd_pix_recebidos_180d,
    coalesce(r.valor_total_recebido_30d, 0.0) AS valor_total_recebido_30d,
    coalesce(r.valor_total_recebido_90d, 0.0) AS valor_total_recebido_90d,
    coalesce(r.valor_total_recebido_180d, 0.0) AS valor_total_recebido_180d,
    coalesce(r.soma_pagadores_distintos_dia_recebedor_180d, 0) AS soma_pagadores_distintos_dia_recebedor_180d,
    coalesce(r.max_qtd_pix_recebidos_dia_180d, 0) AS max_qtd_pix_recebidos_dia_180d,
    CASE WHEN coalesce(p.max_qtd_pix_dia_pagador_7d, 0) >= 3 THEN 1 ELSE 0 END AS burst_daily_7d_flag,
    CASE WHEN coalesce(pair.qtd_pix_mesmo_recebedor_180d, 0) = 0 THEN 1 ELSE 0 END AS first_receiver_flag_real,
    CASE WHEN p.valor_total_pagador_90d IS NULL OR p.qtd_pix_pagador_90d IS NULL OR p.qtd_pix_pagador_90d = 0 THEN NULL ELSE t.vl_pix / (p.valor_total_pagador_90d / p.qtd_pix_pagador_90d) END AS ratio_valor_media_pagador_90d,
    CASE WHEN p.valor_maximo_pix_pagador_180d IS NULL OR p.valor_maximo_pix_pagador_180d = 0 THEN NULL ELSE t.vl_pix / p.valor_maximo_pix_pagador_180d END AS ratio_valor_maximo_pagador_180d,
    current_timestamp() AS dataset_v3_created_at
FROM target t
LEFT JOIN payer_roll p ON t.transaction_id = p.transaction_id
LEFT JOIN pair_roll pair ON t.transaction_id = pair.transaction_id
LEFT JOIN receiver_roll r ON t.transaction_id = r.transaction_id
;
