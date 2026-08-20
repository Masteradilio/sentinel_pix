-- =====================================================================
-- Tabela: hmo_ml.tb_pix_normais_qualified_sample_mbk_180d_v1
-- EXP-010F-R2 — etapa 03
-- Objetivo:
--   Enriquecer normais qualificados com MBK compacta quando disponivel.
--   Versao rolling: filtra MBK compacta para a mesma janela movel.
-- =====================================================================

SET hive.execution.engine=tez;
SET hive.vectorized.execution.enabled=false;
SET hive.vectorized.execution.reduce.enabled=false;
SET hive.exec.compress.output=true;

DROP TABLE IF EXISTS ${hivevar:DB_WORK}.tb_pix_normais_qualified_sample_mbk_180d_v1;

CREATE TABLE ${hivevar:DB_WORK}.tb_pix_normais_qualified_sample_mbk_180d_v1
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

mbk_dedup AS (
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
        FROM ${hivevar:DB_WORK}.tb_pix_mbk_compact_180d_v1 m
        JOIN params p ON 1 = 1
        WHERE cast(m.autdatref AS date) BETWEEN p.dt_ini AND p.dt_fim
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

FROM ${hivevar:DB_WORK}.tb_pix_normais_qualified_sample_180d_v1 n
LEFT JOIN mbk_dedup m
    ON n.cd_pix = m.end_to_end_id
;
