# Proposta Técnica: Arquitetura de Produção com Feature Store HBase

## Motor Híbrido de Detecção de Fraudes PIX — BRB

**Versão:** 1.5.0-r5b22
**Data:** 12 de junho de 2026
**Autor:** Squad IA — Prevenção a Fraudes
**Status:** Baseline Operacional Oficial (R5B22_OFFICIAL_CONSTRAINED_BASELINE)

---

## 1. Contexto e Objetivo

### 1.1 O Problema e o Baseline R5B22

O modelo oficial de detecção de fraudes PIX opera com o baseline restritivo `R5B22_OFFICIAL_CONSTRAINED_BASELINE` (FPR global de 0,957474%). Para sustentar sua performance sem violar restrições, o classificador LightGBM aluno consome exatas **78 features**. 

No momento em que uma transação PIX é realizada, os dados chegam fracionados. Precisamos enriquecer a transação em **tempo real** respeitando um SLA rígido (inferior a 150ms). Como consultas diretas a bancos legados ou Hadoop demoram segundos, o uso de uma **Feature Store baseada no Apache HBase** torna-se essencial para pré-materializar as variáveis complexas em milissegundos.

### 1.2 A Solução: Feature Store no HBase

O Apache HBase foi selecionado pelas seguintes capacidades:
- **Latência de leitura:** 1-10ms (crucial para o pipeline online).
- **Lookup flexível por Row Key:** Busca isolada do pagador, recebedor ou relação mútua.
- **Integração:** Ingestão massiva em lote por Spark (horária/diária).

### 1.3 SLA Alvo e Impacto

A meta principal é que o acréscimo das consultas ao HBase, somadas ao pipeline destilado (LGBM + Engenharia Social + Behavioral Analytics), não estoure o limiar de SLA:

| Etapa do Pipeline | Latência Alvo | Latência Máxima (Hard Limit) |
|---|:---:|:---:|
| Recebimento da transação (Online) | 0ms | 0ms |
| Lookup HBase (pagador + recebedor + relação + contrato) | **5-15ms** | 30ms |
| Features derivadas (Preprocessing runtime) | ~5ms | 10ms |
| Inferência do Pipeline (LGBM v6.x, SE, BEH, Regras R5B14/R5B22) | ~80ms | 150ms |
| **Total do Pipeline Transacional** | **~100ms** | **190ms** |
| **SLA do Banco Central** | — | **10.000ms** |

*Nota sobre Grafo:* O módulo investigativo de grafos (Graph Engineering) é assíncrono e pós-decisão. **Ele não entra no cálculo de SLA transacional** do pipeline principal.

---

## 2. Reclassificação das 78 Features por Origem

As 78 variáveis do aluno destilado (R5B22) e suas ramificações estão estratificadas pelas suas origens de obtenção/produção:

### 2.1 Transação / Evento Online (Embutidas no Payload)
- `vl_pix`
- `ds_tipo_chave_norm`
- `hour`
- `periodo_dia`
- `value_band`
- `autcodret`
- Identificadores brutos (`transaction_id`, `customer_id`, `counterparty_id`, `event_datetime`)

### 2.2 Mobile / Device / Host (Online ou Enriquecimento Rápido)
- `latencia_rede_ms`
- `tempo_processamento_host_ms`
- `topaz_risk_score`
- `mbk_completeness_score`
- `mbk_available_flag`

### 2.3 Feature Store HBase: Histórico do Pagador
- `qtd_pix_pagador_7d`, `qtd_pix_pagador_30d`, `qtd_pix_pagador_90d`, `qtd_pix_pagador_180d`
- `valor_total_pagador_7d`, `valor_total_pagador_30d`, `valor_total_pagador_90d`, `valor_total_pagador_180d`
- `max_qtd_pix_dia_pagador_7d`, `max_qtd_pix_dia_pagador_30d`
- `valor_maximo_pix_pagador_180d`
- `soma_recebedores_distintos_dia_180d`

### 2.4 Feature Store HBase: Relação Pagador-Recebedor
- `qtd_pix_mesmo_recebedor_7d`, `qtd_pix_mesmo_recebedor_30d`, `qtd_pix_mesmo_recebedor_90d`, `qtd_pix_mesmo_recebedor_180d`
- `valor_total_para_recebedor_30d`, `valor_total_para_recebedor_90d`, `valor_total_para_recebedor_180d`
- `primeiro_envio_para_recebedor_180d`, `dias_desde_primeiro_envio_recebedor`
- `valor_medio_para_recebedor_180d`
- `dias_desde_ultima_transacao_recebedor`
- `ratio_valor_pix_vs_max_recebedor_180d`
- `is_recebedor_recorrente_180d`

### 2.5 Feature Store HBase: Histórico do Recebedor
- `qtd_pix_recebidos_30d`, `qtd_pix_recebidos_90d`, `qtd_pix_recebidos_180d`
- `valor_total_recebido_30d`, `valor_total_recebido_90d`, `valor_total_recebido_180d`
- `soma_pagadores_distintos_dia_recebedor_180d`
- `max_qtd_pix_recebidos_dia_180d`
- `first_receiver_flag_real`

### 2.6 Runtime Derivadas (Calculadas Online)
- Bins gerados: `score_bin`, `lgbm_bin`, `if_bin`, `ratio_bin`, `qtd_rec_bin`, `valor_rec_bin`
- `ratio_valor_media_pagador_90d`, `ratio_valor_maximo_pagador_180d`, `burst_daily_7d_flag`
- Componentes consultivos: `lgbm_raw`, `lgbm_mapped`, `if_percentile`, `se_score`, `beh_score`, `score_final`

### 2.7 Contrato Congelado / Sinais do Professor (HBase ou Distilados)
- Decisão base: `r4g_fast_frozen_decisao_recommended`
- Políticas ativas originais: `r5b14_rule_applied`, `r5b14_layer_applied`
- Bins e Features estáticas com sufixo `_frozen`: `ds_tipo_chave_norm_frozen`, `value_band_frozen`, `periodo_dia_frozen`, `mbk_available_flag_frozen`, `first_receiver_flag_real_frozen`, `ratio_valor_maximo_pagador_180d_frozen`, `ratio_valor_media_pagador_90d_frozen`, `vl_pix_frozen`, `qtd_pix_pagador_180d_frozen`, `valor_total_pagador_180d_frozen`, `qtd_pix_mesmo_recebedor_180d_frozen`, `valor_total_para_recebedor_180d_frozen`.

---

## 3. Modelagem do HBase — Tabelas, Famílias e TTL

Para satisfazer o ecossistema destilado sem estourar SLAs, expandimos o modelo original:

### 3.1 `fraud_detection:perfil_cliente` e `fraud_detection:historico_trimestral`
- **Conteúdo:** Perfil demográfico e estatísticas históricas do pagador.
- **Row Key:** `customer_id` (Pagador) formatado.
- **Atualização:** Horária/Diária.
- **TTL:** 48 horas (perfil) e 24 horas (histórico).

### 3.2 `fraud_detection:receiver_history`
- **Conteúdo:** Histórico de fan-in e totais financeiros do recebedor global (independente de quem pagou).
- **Row Key:** `counterparty_id` (Recebedor).
- **Atualização:** Horária.
- **TTL:** 7 dias.

### 3.3 `fraud_detection:historico_recebedores` (Relação A -> B)
- **Conteúdo:** Recorrência, valores acumulados de um pagador para um destinatário específico (Aresta do grafo).
- **Row Key:** `{customer_id}#{counterparty_id}` (Composição).
- **Atualização:** Horária.
- **TTL:** 180 dias.

### 3.4 `fraud_detection:r5b22_contract_features`
- **Conteúdo:** As variáveis estáticas (`_frozen`), decisões históricas do professor e aplicação de regras, ancoradas pela versão do baseline R5B22.
- **Row Key:** `transaction_id` ou chave primária da transação passada se necessitar de reavaliação.
- **Atualização:** Versionado por baseline (gerado durante a promoção de modelos).
- **TTL:** Permanente/Longo prazo (enquanto durar o Baseline R5B22).

### 3.5 `fraud_detection:graph_investigation_features`
- **Conteúdo:** Tabela consultiva/secundária gerada assincronamente pelo Graph Investigation Engine para salvar os atributos e scores topológicos complexos.
- **Row Key:** `transaction_id` ou `{payer_id}#{receiver_id}`.
- **Atualização:** Incremental/Event-Driven.
- **TTL:** 30 dias (para fins de auditoria).

---

## 4. Design das Row Keys

A arquitetura de chaveamento do HBase deve mitigar *hot-spotting* e permitir as três esferas de busca transacional de maneira instantânea e determinística:

1. **Pagador:** `LPAD(customer_id, 14, '0')` — usado em perfil e histórico do usuário origem.
2. **Recebedor:** `LPAD(counterparty_id, 14, '0')` — usado em features isoladas de destino (receiver_history).
3. **Relação (Par):** `LPAD(customer_id, 14, '0') + "#" + LPAD(counterparty_id, 14, '0')` — usado na tabela mútua para consultar "esse pagador conhece esse recebedor?".
4. **Grafo / Evento:** Para investigações e auditoria dos artefatos R5B22, a base será centrada em `transaction_id`.

---

## 5. Validação e Homologação da Proposta

Para garantir a operação íntegra das 78 features em ambiente transacional, a engenharia deve submeter a arquitetura a três frentes de prova:

1. **Teste de Completude (Feature Parity):** Verificar se as chamadas de API Get (multigets) preenchem 100% das matrizes de dicionário requeridas pelo `PipelineOrquestrador`. O modelo recusa scoring se features do professor (R5B16/R5B22 contract) estiverem ausentes.
2. **Teste de Freshness (Atualização Temporal):** Certificar que os batches do Spark garantem que as métricas temporais (como histórico de 24h, 7d, 30d, etc.) tenham defasagem aceitável (tolerância máxima recomendada: 4 horas).
3. **Teste de Fallback e Degradação:** Assegurar que, diante da falta esporádica de um nó (row não encontrada) na tabela `fraud_detection:receiver_history`, o preprocessor impute o valor para a categoria "Missing" documentada no metadata do LGBM, sem corromper a pontuação.
4. **Teste de Drift Semanal:** Analisar as frequências dos bins e categorias enviadas via API contra os metadados oficias do `scoring_config.json` para observar vazamentos temporais.
