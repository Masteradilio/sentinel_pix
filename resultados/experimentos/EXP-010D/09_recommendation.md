# EXP-010D — MAF Hydrated Fraud Compatibility Audit

Gerado em: `2026-05-13T15:49:24`

## Status

**Status geral:** `WARN`

## Resumo executivo

- Linhas no CSV MAF hidratado: `13558`
- `cd_pix` únicos: `13558`
- Colunas: `49`

- Menor `dt_pix`: `2022-05-13 06:33:46`
- Maior `dt_pix`: `2026-05-11 13:53:02`

- Valor mediano: `1170.25`
- Valor p95: `12903.45`
- Valor máximo: `275000.00`

## Checks com falha ou warning

- `mbk/mbk_any_coverage`: `WARN` — MBK ausente ou com baixa cobertura | esperado=`>=0.20 em pelo menos uma coluna MBK` atual=`{'device_name': 0.0, 'app_version': 0.0, 'ip_address': 0.0, 'latencia_rede_ms': 0.0, 'tempo_interacao_ms': 0.0, 'tempo_processamento_host_ms': 0.0, 'metodo_autenticacao': 0.0, 'session_id': 0.0, 'cd_retorno': 0.0, 'topaz_risk_score': 0.0, 'topaz_transacao_rejeitada': 0.0, 'is_agendamento_recorrente': 0.0}`

## Overlap com fraudes antigas

- Status: `NO_ID_COLUMN`

## Interpretação

A base MAF hidratada deve ser tratada como nova fonte positiva forte, mas ainda não deve ser usada sozinha para treino.

Ela precisa ser combinada com:

1. hidratação MBK por chave no EXP-010E;
2. amostragem de normais em 90/180 dias no EXP-010F;
3. construção de dataset unificado no EXP-010G.

## Decisão

Avançar para o EXP-010E — MBK Keyed Hydration Audit, mantendo o EXP-010D como validação local da base MAF.
