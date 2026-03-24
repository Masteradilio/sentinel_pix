Adilio, fiz a análise completa cruzando todas as fontes. Aqui está o relatório detalhado:

---

## Relatório: Refinamento da Ingestão de Dados do Big Data

### 1. Panorama Atual

#### O que o LGBM usa (62 features)

Analisando o `base_mvp_model_ready.csv` e o `lgbm_features.json`, as 62 features que o modelo efetivamente usa são:

| Grupo                           | Features                                                                                                                                                                                                                                                                                           | Qt           |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| Valor/Desvio                    | `vl_pix`, `log_vl_pix`, `vl_pix_over_1000_flag`, `vl_mediana_pix_trimestre`, `vl_desvio_padrao_pix_trimestre`, `ratio_valor_mediana`, `diff_valor_mediana`, `ratio_valor_desvio_padrao`, `zscore_valor_aprox`                                                                    | 9            |
| Frequência                     | `qt_total_pix_trimestre`, `is_first_tx_trimestre`, `qt_intervalo_transacao_minuto`, `qt_intervalo_mediana_trimestre`, `qt_intervalo_desvio_padrao_trimestre`, `qt_pix_dia_maximo_trimestre`, `ratio_intervalo_vs_mediana`, `diff_intervalo_vs_mediana`, `zscore_intervalo_aprox` | 9            |
| Perfil cliente                  | `nr_idade`, `qt_tempo_relacionamento_mes`, `qt_aparelhos_distintos_trimestre`                                                                                                                                                                                                                | 3            |
| Rede/Sessão                    | `latencia_rede_ms_final`, `vl_latencia_rede_media_trimestre`, `tempo_processamento_host_ms`, `ratio_latencia_cliente`, `diff_latencia_cliente`, `latencia_host_ratio`                                                                                                                  | 6            |
| Chave PIX                       | `pix_key_random_flag`, `pix_key_email_flag`, `pix_key_document_flag`, `pix_key_other_flag`, `pix_key_missing_flag_derived`                                                                                                                                                               | 5            |
| Flags recebedor                 | `receiver_document_same_as_customer_flag`, `first_receiver_flag`                                                                                                                                                                                                                               | 2            |
| Sequenciais                     | `minutes_since_prev_tx`, `tx_count_prev_30m`, `burst_30m_flag`, `receiver_tx_count_prev`, `key_tx_count_prev`, `first_key_flag`, `distinct_receivers_so_far`, `distinct_keys_so_far`                                                                                               | 8            |
| Temporal                        | `hour`, `day_of_week`, `is_business_hours`                                                                                                                                                                                                                                                   | 3            |
| Device/App                      | `app_version_minor`, `device_missing_flag`, `app_version_missing_flag`, `auth_method_missing_flag`, `topaz_missing_flag`, `host_time_missing_flag`, `latencia_missing_flag`                                                                                                          | 7            |
| Topaz                           | `topaz_score_filled` (`topaz_risk_score` preenchido)                                                                                                                                                                                                                                           | 1            |
| Regras                          | `rule_age_score`, `rule_relationship_score`, `rule_mule_account_score`, `rule_random_key_score`, `rule_velocity_score`, `rule_topaz_score`, `rule_score_raw`, `rule_score_normalized`                                                                                              | 8            |
| **Não usados no treino** | `topaz_risk_score` (original c/ NaN)                                                                                                                                                                                                                                                             | 1            |
| **Total**                 |                                                                                                                                                                                                                                                                                                    | **62** |

#### O que o Isolation Forest usa (10 features)

Segundo o `isolation_forest_config.json`, apenas:

```
log_vl_pix, qt_total_pix_trimestre, qt_intervalo_transacao_minuto,
qt_pix_dia_maximo_trimestre, nr_idade, qt_tempo_relacionamento_mes,
ratio_valor_mediana, zscore_valor_aprox, ratio_intervalo_vs_mediana,
zscore_intervalo_aprox
```

---

### 2. O que é ingerido do Big Data mas NÃO é usado por nenhum componente

Estes campos são extraídos nos scripts Spark (01/02) mas **não chegam ao modelo** nem seriam usados pelos módulos SE/Behavioral:

| Campo ingerido                         | Motivo de descarte                                                             | Recomendação                                                                            |
| -------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| `cd_retorno`                         | Nunca vira feature (sempre "0" ou nulo)                                        | **MANTER** — útil para filtro de qualidade, mas não precisa ir pro CSV de treino |
| `topaz_transacao_habilitada`         | Sempre 1 ou NULL, sem variância                                               | **REMOVER do CSV** — valor constante                                               |
| `topaz_sync_id`                      | ID interno Topaz, nunca usado                                                  | **REMOVER do CSV** — só metadata                                                  |
| `ds_segmento`                        | Filtrado no Spark (`BRASILIA`, `EXCLUSIVO`, etc.) mas não chega ao modelo | ⚠️**MANTER E PASSAR AO CSV** (ver seção 3)                                      |
| `ds_chave_pix` (texto bruto)         | Já decomposto em flags (`pix_key_random_flag`, etc.)                        | **MANTER na ingestão** — necessário para derivar flags                           |
| `cd_cpf_cnpj_recebedor`              | Usado só para `receiver_document_same_as_customer_flag`                     | **MANTER na ingestão** — necessário para derivação                             |
| `session_id`                         | Nunca vira feature                                                             | **MANTER na ingestão** — útil para Behavioral (identificar sessão)              |
| `tempo_interacao_ms`                 | Presente no Spark mas**não chega como feature** ao modelo               | ⚠️**PROBLEMA** (ver seção 4)                                                    |
| `vl_tempo_interacao_medio_trimestre` | Idem                                                                           | ⚠️**PROBLEMA** (ver seção 4)                                                    |
| `metodo_autenticacao`                | Presente no Spark mas não vira feature                                        | ⚠️**PROBLEMA** (ver seção 4)                                                    |
| `ip_address`                         | Presente no Spark mas não vira feature                                        | Manter para Behavioral (GeoIP futuro)                                                     |
| `device_name`                        | Presente, gera `device_missing_flag` e `device_name_normalized`            | Manter                                                                                    |

---

### 3. Campos que FALTAM na ingestão e são necessários

Estes campos **existem no Big Data** mas não estão sendo extraídos pelos scripts Spark:

| Campo necessário                                             | Onde existe                                                                   | Quem precisa                                                                     | Impacto                                                                                              |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **`ds_sexo`** (gênero)                               | `landing_brb_db2_aox.aoxb01` — campo `X0100_CLTSEXO` ou similar          | SE Detector:`mulher_idosa` (2.3x mais vítima)                                 | 🔴**ALTO** — ativa indicador crítico para golpe do falso funcionário                        |
| **`ds_estado_civil`**                                 | `landing_brb_db2_aox.aoxb01` — campo de estado civil                       | SE Detector:`viuvo_viuva` (romance scam)                                       | 🟡**MÉDIO** — ativa 1 indicador                                                              |
| **`ds_segmento`**                                     | Já é extraído no Spark! Mas é**filtrado e descartado** antes do CSV | SE Detector:`segmento_alto_patrimonio`                                         | 🔴**ALTO** — já temos o dado, só precisa passar ao CSV                                      |
| **`vl_limite_pix`** ou **`vl_limite_diario`** | Tabela de limites do cliente (possivelmente em outra landing)                 | SE Detector: indicadores `valor_alto`, `valor_critico`, `valor_muito_alto` | 🟡**MÉDIO** — hoje usamos `ratio_valor_mediana` como proxy                                 |
| **`tp_primeiro_envio_recebedor_trimestre`**           | Derivável no Spark com window function                                       | SE Detector:`primeiro_envio` (89% das fraudes!)                                | 🔴**ALTO** — campo crítico. Hoje só existe no cache do pipeline via `first_receiver_flag` |

---

### 4. Campos ingeridos mas DESPERDIÇADOS (não viram features)

Estes campos são extraídos do Big Data, chegam ao CSV, mas **o preprocessing.py os descarta** e nunca viram features para o modelo:

| Campo                                            | Valor nos dados                               | Por que é desperdiçado                            | Deveria virar feature?                                                                         |
| ------------------------------------------------ | --------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **`tempo_interacao_ms`**                 | Presente em ~70% das fraudes, ~0% das normais | É dropado no preprocessing (muitos NaN em normais) | ✅**SIM** — Behavioral usa intensamente (SESSAO_RAPIDA, SESSAO_LONGA, TYPING_DEVIATION) |
| **`vl_tempo_interacao_medio_trimestre`** | Idem                                          | Idem                                                | ✅**SIM** — base de comparação para anomalias                                         |
| **`metodo_autenticacao`**                | "1" = biometria, "2" = senha, etc.            | Nunca foi convertido em features                    | ✅**SIM** — `LOGIN_SENHA_ALTO_VALOR`, `LOGIN_METHOD_CHANGE`                         |
| **`is_agendamento_recorrente`**          | "true"/"false"                                | Nunca vira feature                                  | ✅**SIM** — atenuante no Behavioral (reduz FP)                                          |
| **`topaz_transacao_rejeitada`**          | 0 ou 1                                        | Feature existe mas não entra no modelo             | ✅**SIM** — VETO do Topaz (peso 5 no Behavioral)                                        |

**Resumo**: Temos dados valiosos sendo ingeridos do Big Data que são jogados fora antes de chegar ao modelo. Isso é o maior desperdício atual.

---

### 5. Recomendação: O que ADICIONAR à ingestão Spark

#### 5.1 Novos campos do cadastro de clientes (seção 1 dos scripts Spark)

```sql
-- Adicionar ao SELECT de df_cliente:
c.X0100_CLTSEXO as cd_sexo_raw,    -- Para derivar ds_sexo (M/F)
c.X0100_CLTESTCIV as cd_estado_civil_raw  -- Para derivar ds_estado_civil
-- ds_segmento já é extraído — só precisa NÃO filtrar antes de salvar
```

**Features derivadas no CSV:**

| Feature nova            | Derivação                                                             | Quem usa                        |
| ----------------------- | ----------------------------------------------------------------------- | ------------------------------- |
| `ds_sexo`             | Mapeamento do código raw → "M"/"F"                                    | SE:`mulher_idosa`             |
| `ds_estado_civil`     | Mapeamento do código → "SOLTEIRO"/"CASADO"/"VIUVO"/etc.               | SE:`viuvo_viuva`              |
| `ds_segmento`         | Já extraído —**parar de filtrar antes do CSV**                 | SE:`segmento_alto_patrimonio` |
| `is_segmento_premium` | `ds_segmento IN ('EXCLUSIVO','PRIVATE','MILLENIUM','PREMIUM')` → 0/1 | LGBM (feature nova)             |

#### 5.2 Derivar `tp_primeiro_envio_recebedor_trimestre` no Spark

```sql
-- Na seção de window functions:
.withColumn("tp_primeiro_envio_recebedor_trimestre",
    F.when(
        F.row_number().over(
            Window.partitionBy("cd_cpf_pagador", "cd_cpf_cnpj_recebedor")
            .orderBy("dt_pix")
        ) == 1, 1
    ).otherwise(0)
)
```

**Impacto**: Esta é provavelmente a feature mais valiosa que falta. 89% das fraudes são primeiro envio ao recebedor. Hoje só existe via cache do pipeline (`first_receiver_flag`), mas para treino e para o SE Detector, ter isso direto do Spark é muito melhor.

#### 5.3 Garantir que campos existentes cheguem ao CSV final

O script Spark já extrai mas o CSV de treino perde:

```
✅ Passar ao CSV: tempo_interacao_ms
✅ Passar ao CSV: vl_tempo_interacao_medio_trimestre  
✅ Passar ao CSV: metodo_autenticacao
✅ Passar ao CSV: is_agendamento_recorrente
✅ Passar ao CSV: topaz_transacao_rejeitada
✅ Passar ao CSV: ip_address (para Behavioral futuro)
✅ Passar ao CSV: device_name (texto — para Behavioral)
✅ Passar ao CSV: app_version (texto — para Behavioral)
✅ Passar ao CSV: session_id (para Behavioral)
```

---

### 6. Recomendação: O que REMOVER da ingestão/CSV

| Campo                          | Motivo                                         | Ação                                              |
| ------------------------------ | ---------------------------------------------- | --------------------------------------------------- |
| `topaz_sync_id`              | ID interno, nunca usado                        | **Parar de extrair**                          |
| `topaz_transacao_habilitada` | Sempre 1 ou NULL                               | **Parar de extrair**                          |
| `autnsublk` (script normais) | Nunca usado                                    | **Já não aparece no CSV** ✅                |
| `cd_retorno`                 | Útil para filtro mas não para modelo         | **Extrair mas não incluir no CSV de treino** |
| `cd_cliente`                 | ID interno BRB                                 | **Extrair mas não incluir no CSV de treino** |
| `dt_inicio_relacionamento`   | Redundante com `qt_tempo_relacionamento_mes` | **Extrair mas não incluir no CSV**           |

---

### 7. Impacto esperado nos componentes

#### LGBM (62 → ~67 features)

| Feature nova                                          | Tipo      | Impacto esperado                                      |
| ----------------------------------------------------- | --------- | ----------------------------------------------------- |
| `tp_primeiro_envio_recebedor_trimestre`             | Flag 0/1  | 🔴**Muito alto** — 89% correlação com fraude |
| `is_segmento_premium`                               | Flag 0/1  | 🟡 Médio — ajuda em segmentação                   |
| `metodo_auth_encoded` (biometria=0, senha=1, pin=2) | Numérico | 🟡 Médio — login por senha em valor alto é sinal   |
| `is_agendamento_recorrente_flag`                    | Flag 0/1  | 🟢 Baixo — atenuante                                 |
| `topaz_rejeitada_flag`                              | Flag 0/1  | 🔴 Alto — sinal forte de fraude                      |

#### Isolation Forest (~10 → ~14 features)

Com os novos dados, o IF pode ser retreinado com features adicionais:

| Feature candidata                         | Motivo                                      |
| ----------------------------------------- | ------------------------------------------- |
| `tp_primeiro_envio_recebedor_trimestre` | Primeiro envio + 1ª tx = altíssimo risco  |
| `pix_key_random_flag`                   | Chave aleatória em 1ª tx é sinal forte   |
| `metodo_auth_encoded`                   | Sessão sem biometria em 1ª tx é suspeito |
| `hour`                                  | Madrugada + 1ª tx = risco                  |

#### Social Engineering Detector

| Indicador que ATIVA               | Campo necessário                         | Status                                |
| --------------------------------- | ----------------------------------------- | ------------------------------------- |
| `mulher_idosa`                  | `ds_sexo`                               | 🆕 Precisa ingerir                    |
| `viuvo_viuva`                   | `ds_estado_civil`                       | 🆕 Precisa ingerir                    |
| `segmento_alto_patrimonio`      | `ds_segmento`                           | ✅ Já ingerido, só parar de filtrar |
| `primeiro_envio`                | `tp_primeiro_envio_recebedor_trimestre` | 🆕 Derivar no Spark                   |
| `valor_alto/muito_alto/critico` | `ratio_valor_mediana` como proxy        | ✅ Já existe                         |

#### Behavioral Analytics

| Fator que ATIVA               | Campo necessário                                               | Status                              |
| ----------------------------- | --------------------------------------------------------------- | ----------------------------------- |
| `TYPING_ANORMAL`            | `tempo_interacao_ms`                                          | ✅ Já ingerido, parar de descartar |
| `SESSAO_MUITO_RAPIDA`       | `tempo_interacao_ms`                                          | ✅ Idem                             |
| `SESSAO_MUITO_LONGA`        | `tempo_interacao_ms` + `vl_tempo_interacao_medio_trimestre` | ✅ Idem                             |
| `LOGIN_SENHA_ALTO_VALOR`    | `metodo_autenticacao`                                         | ✅ Já ingerido, parar de descartar |
| `LOGIN_METHOD_CHANGE`       | `metodo_autenticacao` (precisa de perfil)                     | ✅ Já ingerido                     |
| `AGENDAMENTO_RECORRENTE`    | `is_agendamento_recorrente`                                   | ✅ Já ingerido, parar de descartar |
| `APP_VERSION_DESATUALIZADA` | `app_version`                                                 | ✅ Já existe                       |

---

### 8. Resumo das alterações nos scripts Spark

#### `01_ingestao_unificada_mobile_normais.py`

```diff
# Seção 1 - CLIENTES: Adicionar campos
+ c.X0100_CLTSEXO as cd_sexo_raw,
+ c.X0100_CLTESTCIV as cd_estado_civil_raw

# Seção 4 - JOIN: NÃO filtrar segmento antes de salvar
- segmentos_validos = ['BRASILIA', 'EXCLUSIVO', ...]
- df_base = df_base.filter(F.col("ds_segmento").isin(segmentos_validos))
+ # Manter ds_segmento no CSV — filtro pode ser feito depois no treino

# Seção 5 - FEATURES: Adicionar primeiro envio
+ .withColumn("tp_primeiro_envio_recebedor_trimestre", ...)

# Seção 6 - OUTPUT: Adicionar campos ao SELECT final
+ f.ds_sexo,
+ f.ds_estado_civil,  
+ f.ds_segmento,
+ f.tp_primeiro_envio_recebedor_trimestre,
+ f.tempo_interacao_ms,        -- JÁ EXISTE, garantir que não é dropado
+ f.vl_tempo_interacao_medio_trimestre,  -- IDEM
+ f.metodo_autenticacao,       -- IDEM
+ f.is_agendamento_recorrente, -- IDEM

# REMOVER do SELECT final:
- f.topaz_sync_id,
- f.topaz_transacao_habilitada,
```

#### `02_ingestao_unificada_mobile_fraudes.py`

Mesmas alterações, adaptadas ao script de fraudes.

---

### 9. Plano de ação

| Passo       | O que fazer                                                                                              | Prioridade  |
| ----------- | -------------------------------------------------------------------------------------------------------- | ----------- |
| **1** | Verificar no Big Data se `X0100_CLTSEXO` e `X0100_CLTESTCIV` existem e qual o dicionário de valores | 🔴 Primeiro |
| **2** | Alterar os 2 scripts Spark conforme seção 8                                                            | 🔴 Segundo  |
| **3** | Rodar nova ingestão e gerar CSVs atualizados                                                            | 🔴 Terceiro |
| **4** | Eu adapto o `preprocessing.py` para não descartar os campos novos                                     | 🔴 Quarto   |
| **5** | Retreinar LGBM com ~67 features (incluindo as novas)                                                     | 🟡 Quinto   |
| **6** | Retreinar IF com ~14 features                                                                            | 🟡 Sexto    |
| **7** | Adaptar e integrar SE + Behavioral com dados reais                                                       | 🟡 Sétimo  |
| **8** | Novo `teste_pipeline_relatorio.py` com métricas v2.1                                                  | 🟢 Oitavo   |

Quer que eu prepare as alterações exatas nos scripts Spark para você rodar, ou prefere primeiro verificar a disponibilidade dos campos `X0100_CLTSEXO` e `X0100_CLTESTCIV` no Big Data?
