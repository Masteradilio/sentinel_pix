# DATA_INTAKE_CONTRACT — Antifraude PIX

Gerado em: `2026-05-12T18:28:23`

## Versão

- `schema_version`: `1.1`
- `model_version`: `post_fase2_c1`
- `status`: `ACTIVE_BASELINE`

## Objetivo

Definir o contrato de entrada para novos dados de transações PIX e novos labels de fraude.

Esta versão separa explicitamente dois modos de uso:

1. **scoring/inferência**: novas transações podem não ter `is_fraud`.
2. **avaliação supervisionada**: labels vêm em arquivo/tabela separada ligada por `transaction_id`.

## Modo 1 — Scoring / inferência

Usado quando o objetivo é aplicar o baseline `post_fase2_c1` a uma nova janela ainda sem labels completos.

### Colunas obrigatórias de transações

- `transaction_id`
- `customer_id`
- `vl_pix`
- `qt_tempo_relacionamento_mes`
- `first_receiver_flag`
- `pix_key_random_flag`

### Coluna temporal obrigatória

O arquivo deve conter pelo menos uma das colunas abaixo:

- `event_datetime` **ou** `dt_transacao`

### Observação sobre `is_fraud`

`is_fraud` **não é obrigatório** no arquivo de transações em modo scoring/inferência.

## Modo 2 — Avaliação supervisionada

Usado quando há labels confirmados e o objetivo é calcular métricas como TP, FP, FN, Precision, Recall e F1.

Neste modo, as transações seguem o mesmo contrato do modo scoring, mas os labels devem vir em arquivo ou tabela separada.

### Colunas obrigatórias de labels

- `transaction_id`
- `is_fraud`

### Colunas recomendadas de labels

- `fraud_type`
- `label_source`
- `label_created_at`
- `label_confidence`
- `chargeback_flag`
- `confirmed_by_human`
- `contestacao_id`
- `motivo_fraude`
- `canal_confirmacao`

## Colunas necessárias para a C1

- `vl_pix`
- `qt_tempo_relacionamento_mes`
- `first_receiver_flag`
- `pix_key_random_flag`

## Colunas recomendadas para enriquecer novas extrações

- `event_datetime`
- `dt_transacao`
- `idade_cliente`
- `nr_idade`
- `tipo_pessoa`
- `canal`
- `chave_pix_tipo`
- `uf_origem`
- `uf_destino`
- `device_id`
- `ip`
- `merchant_category`
- `vl_renda_cliente`
- `topaz_risk_score`
- `qt_total_pix_trimestre`
- `qt_envio_recebedor_trimestre`
- `qt_aparelhos_distintos_trimestre`

## Regras mínimas

- `transaction_id` deve ser único no arquivo de transações.
- `transaction_id` deve permitir join com labels quando labels existirem.
- O arquivo deve ter uma data transacional explícita: `event_datetime` ou `dt_transacao`.
- Colunas obrigatórias do modo scoring não devem vir nulas.
- `is_fraud` deve usar valores `0` ou `1` quando existir em labels.
- Não assumir que ausência de fraude confirmada significa normalidade sem janela de maturação.
- Mudanças grandes de distribuição em `vl_pix`, relacionamento ou flags devem ser tratadas como drift.
- Novos dados não devem sobrescrever o baseline oficial; devem ser avaliados em diretório próprio.

## Arquivos técnicos

- `resultados/experimentos/EXP-010A-R1/01_transaction_schema_contract_v1_1.json`
- `resultados/experimentos/EXP-010A-R1/02_label_schema_contract_v1_1.json`
- `resultados/experimentos/EXP-010A-R1/03_DATA_INTAKE_CONTRACT_v1_1.md`
- `resultados/experimentos/EXP-010A-R1/04_REVALUATION_HARNESS_SPEC_v1_1.md`
