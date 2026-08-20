# EXP-010E-R2B — MBK Compact Index 180d

Gerado em: `2026-05-15T15:38:05`

## Resultado executivo corrigido

- Target keys total 180d: `1594`
- Dias/processamentos realizados neste run: `0`
- Registros compactos MBK gerados neste run: `0`
- Matched MBK total contra 180d: `1391`
- Coverage total 180d parcial: `0.872647`
- Target keys nos dias alvo processados: `0`
- Matched MBK nos dias alvo processados: `0`
- Coverage nos dias alvo processados: `0.0`
- Compact table: `hmo_ml.tb_pix_mbk_compact_180d_v1`
- Hydration table: `hmo_ml.tb_pix_maf_mbk_hydration_180d_v2`

## Como interpretar

A cobertura total 180d ainda é parcial, porque o piloto processou apenas parte das partições MBK.

Portanto, `coverage_total_180d` não deve ser usado para reprovar a estratégia enquanto o compact index ainda não cobrir todos os dias da janela.

A métrica mais útil para o piloto é:

```text
coverage_exact_target_dates_processed
```

Além disso, o arquivo `03b_mbk_match_lag_distribution.csv` deve ser usado para verificar se `autdatref` bate com `dt_pix` ou se precisamos aumentar `DATE_PAD_DAYS`.

## Campos MBK entre registros encontrados

- `autcodret`: coverage=`1.0` (1391/1391)
- `latencia_rede_ms`: coverage=`0.99353` (1382/1391)
- `tempo_processamento_host_ms`: coverage=`0.99353` (1382/1391)
- `ip_address`: coverage=`0.99353` (1382/1391)
- `device_name`: coverage=`0.989935` (1377/1391)
- `app_version`: coverage=`0.989935` (1377/1391)
- `session_id`: coverage=`0.989935` (1377/1391)
- `topaz_risk_score`: coverage=`0.919482` (1279/1391)

## Decisão preliminar

`RELATORIO_SEM_DIAS_PROCESSADOS`

Nenhum dia foi processado; não há decisão operacional.

## Próximo passo recomendado

Antes de processar mais blocos, verificar:

1. `03b_mbk_match_lag_distribution.csv` para decidir se `DATE_PAD_DAYS` deve ser 0, 1 ou 2;
2. `04c_mbk_field_coverage_matched_only.csv` para medir qualidade dos campos entre matches;
3. `03_mbk_coverage_by_day.csv` para ver cobertura real nos dias alvo processados;
4. se aprovado, continuar em blocos de 12–15 dias com `OVERWRITE_TABLES=False` e `RESUME_SKIP_PROCESSED_DATES=True`.
