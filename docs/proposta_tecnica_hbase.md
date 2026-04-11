

# Proposta Técnica: Arquitetura de Produção com Feature Store HBase

## Motor Híbrido de Detecção de Fraudes PIX — BRB

**Versão:** 1.0
**Data:** 27 de março de 2026
**Autor:** Squad IA — Prevenção a Fraudes
**Status:** Proposta para avaliação

---

## 1. Contexto e Objetivo

### 1.1 O Problema

O modelo de detecção de fraudes PIX foi treinado com **52 features** construídas a partir de dados de **5 sistemas fonte** (BLK, MBK, AOX, DNA, GESEI/MAF). No entanto, no momento em que uma transação PIX chega para análise em tempo real, **apenas 7 campos estão disponíveis instantaneamente** — os demais precisam ser buscados ou calculados.

| Grupo de Features | Qtd no Modelo | Disponível na transação? | Fonte Real | Latência (query direta) |
|---|:---:|:---:|---|---|
| **Transação** (vl_pix, cd_pix, dt_pix, chave PIX) | 7 | ✅ Sim | Evento PIX | 0ms |
| **Perfil do Cliente** (idade, relacionamento, sexo, renda) | 7 | ❌ Não | AOX/DNA (DB2) | 200-500ms |
| **Histórico Trimestral** (mediana, desvio, frequência) | 8 | ❌ Não | BLK (Hadoop) | 3-10s |
| **Device/Sessão** (latência, topaz, aparelhos) | 7 | ⚠️ Parcial | MBK (Hadoop) | 3-10s |
| **Sequenciais** (burst, first_receiver, distinct_receivers) | 12 | ❌ Não | Cálculo em tempo real | Depende |
| **Derivadas** (ratios, zscores, rule_scores) | 11 | ❌ Não | Preprocessamento local | ~5ms |

**Conclusão:** Consultar os sistemas fonte em tempo real é **inviável** para o SLA do PIX. Uma query no Hadoop leva 3-10 segundos; uma query no DB2 leva 200-500ms. Precisamos de uma camada intermediária que pré-materialize esses dados com latência de milissegundos.

### 1.2 A Solução: Feature Store no HBase

O **Apache HBase** é um banco de dados NoSQL distribuído, orientado a colunas, construído sobre o HDFS (Hadoop Distributed File System). É a escolha ideal para este caso de uso porque:

| Característica do HBase | Relevância para o Projeto |
|--------------------------|--------------------------|
| **Latência de leitura: 1-10ms** | Compatível com o SLA de 150ms do pipeline de inferência |
| **Lookup por chave primária (row key)** | Nosso caso é exatamente isso: buscar perfil por CPF |
| **Integração nativa com Hadoop/Spark** | O banco já possui infraestrutura Hadoop; o Spark job de ingestão pode escrever diretamente no HBase |
| **Column Families** | Permite organizar os dados por grupo funcional (perfil, histórico, recebedores) com TTLs diferentes |
| **Escalabilidade horizontal** | Suporta milhões de rows sem degradação de performance |
| **Versionamento de células** | Permite manter histórico de atualizações (útil para auditoria) |

### 1.3 SLA Alvo

| Etapa do Pipeline | Latência Alvo | Latência Máxima |
|---|:---:|:---:|
| Recebimento da transação | 0ms | 0ms |
| Lookup HBase (perfil + histórico) | **5-10ms** | 20ms |
| Features sequenciais (cache memória) | <1ms | 5ms |
| Preprocessing (ratios, zscores, rules) | ~5ms | 10ms |
| Inferência LGBM + IF + SE + Behavioral | ~134ms | 200ms |
| **Total do pipeline** | **~150ms** | **235ms** |
| **SLA do Banco Central** | — | **10.000ms** |
| **Margem de folga** | — | **~42×** |

---

## 2. Arquitetura Geral

### 2.1 Visão Macro

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                    FLUXO BATCH (a cada hora)                        │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐       │
│  │   BLK    │   │   AOX    │   │   DNA    │   │   MBK    │       │
│  │(Extrato  │   │(Cadastro │   │(Segmento)│   │(Mobile   │       │
│  │  PIX)    │   │ Cliente) │   │          │   │ Banking) │       │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘       │
│       │               │               │               │             │
│       └───────────────┼───────────────┼───────────────┘             │
│                       ▼                                             │
│              ┌────────────────┐                                     │
│              │   Spark Job    │  ← 01_ingestao_unificada adaptado  │
│              │   (Horário)    │                                     │
│              └───────┬────────┘                                     │
│                      │                                              │
│                      ▼                                              │
│              ┌────────────────┐                                     │
│              │     HBASE      │  ← Feature Store materializada     │
│              │  (por CPF)     │     ~800K rows, ~640MB             │
│              └────────────────┘                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                  FLUXO TEMPO REAL (por transação)                   │
│                                                                     │
│  Transação PIX ──────────────────────────────────┐                  │
│  (7 campos)                                      │                  │
│       │                                          │                  │
│       ▼                                          ▼                  │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────┐            │
│  │ Features │    │  HBase GET   │    │    Cache de    │            │
│  │ Online   │    │  (por CPF)   │    │   Histórico    │            │
│  │ (7)      │    │  5-10ms      │    │  (memória)     │            │
│  └────┬─────┘    └──────┬───────┘    └───────┬────────┘            │
│       │                 │                    │                      │
│       └────────┬────────┘────────────────────┘                      │
│                ▼                                                     │
│       ┌────────────────┐                                            │
│       │ Preprocessing  │  ← PixPreprocessor (joblib)               │
│       │ + Derivadas    │    Calcula ratios, zscores, rules         │
│       │ (~10ms)        │                                            │
│       └───────┬────────┘                                            │
│               ▼                                                     │
│       ┌────────────────┐                                            │
│       │  52 Features   │                                            │
│       │  Completas     │                                            │
│       └───────┬────────┘                                            │
│               ▼                                                     │
│       ┌────────────────────────────────────────┐                    │
│       │         Motor de Decisão               │                    │
│       │  LGBM → Cascade → IF → SE → Behav.    │                    │
│       │  (~134ms)                              │                    │
│       └───────┬────────────────────────────────┘                    │
│               ▼                                                     │
│       DECISÃO: APROVAR │ CONFIRMAR │ BLOQUEAR                       │
│       + Explicabilidade SHAP + Mensagem ao Cliente                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Componentes do Sistema

| Componente | Tecnologia | Responsabilidade |
|-----------|-----------|-----------------|
| **Feature Store** | HBase | Armazena perfil + histórico pré-calculado por CPF |
| **Spark Job** | PySpark (adaptação do `01_ingestao_unificada`) | Materializa dados no HBase a cada hora |
| **Cache de Histórico** | Memória da aplicação (dict Python / Redis futuro) | Mantém últimas N transações por CPF para features sequenciais |
| **API Antifraude** | FastAPI (Python) | Orquestra o pipeline de inferência em tempo real |
| **Preprocessor** | PixPreprocessor (joblib) | Transforma dados brutos em 52 features do modelo |
| **Motor de Decisão** | LGBM + Cascade + IF + SE + Behavioral | Classifica a transação e gera decisão + explicabilidade |

---

## 3. Modelagem do HBase — Tabelas e Column Families

### 3.1 Design da Row Key

A **row key** de todas as tabelas é o **CPF do pagador** (14 caracteres, com LPAD de zeros). Isso garante:

- **Lookup O(1)** por CPF na hora da transação
- **Distribuição uniforme** (CPFs são naturalmente distribuídos)
- **Sem hot-spotting** (CPFs não têm prefixo comum problemático)

```
Row Key: LPAD(cd_cpf_pagador, 14, '0')
Exemplo: "00012345678900"
```

> **Alternativa para evitar hot-spotting em ambientes muito grandes:** Usar hash do CPF como prefixo: `MD5(cpf)[0:4] + cpf`. Isso distribui as rows uniformemente entre os RegionServers do HBase. Avaliar com a equipe de infraestrutura se necessário.

### 3.2 Tabela: `fraud_detection:perfil_cliente`

**Atualização:** Diária (dados mudam raramente)
**Fonte:** AOX (`aoxb01` + `aoxb17`) + DNA (`dnab01`)
**TTL:** 48 horas

| Column Family | Qualifier | Tipo | Exemplo | Feature(s) do Modelo |
|:---:|-----------|:----:|---------|----------------------|
| `demo` | `nr_idade` | int | 67 | `nr_idade` → `rule_age_score` |
| `demo` | `qt_tempo_relacionamento_mes` | int | 240 | `qt_tempo_relacionamento_mes` → `rule_relationship_score` |
| `demo` | `ds_sexo` | string | F | → `is_sexo_feminino_flag` |
| `demo` | `ds_estado_civil` | string | VIUVO | → `is_viuvo_flag` → `perfil_vulneravel_se_flag` |
| `demo` | `ds_segmento` | string | VAREJO | → `is_segmento_premium_flag` |
| `renda` | `vl_renda_cliente` | double | 3500.00 | `vl_renda_cliente` → `ratio_pix_renda` → `pix_over_50pct_renda_flag` → `renda_missing_flag` |
| `renda` | `qt_dependentes` | int | 0 | → `perfil_vulneravel_se_flag` |
| `meta` | `dt_ultima_atualizacao` | timestamp | 2026-03-27T06:00:00Z | Controle de freshness |

**Comando HBase Shell para criação:**

```bash
create 'fraud_detection:perfil_cliente',
  {NAME => 'demo', VERSIONS => 2, TTL => 172800, COMPRESSION => 'SNAPPY'},
  {NAME => 'renda', VERSIONS => 2, TTL => 172800, COMPRESSION => 'SNAPPY'},
  {NAME => 'meta', VERSIONS => 1, TTL => 172800}
```

### 3.3 Tabela: `fraud_detection:historico_trimestral`

**Atualização:** Horária (reflete transações recentes)
**Fonte:** BLK (`tb_extrato_pix`) — agregações Spark sobre 90 dias
**TTL:** 24 horas

| Column Family | Qualifier | Tipo | Exemplo | Feature(s) do Modelo |
|:---:|-----------|:----:|---------|----------------------|
| `valor` | `vl_mediana_pix_trimestre` | double | 150.00 | `vl_mediana_pix_trimestre` → `ratio_valor_mediana` → `diff_valor_mediana` → `zscore_valor_aprox` |
| `valor` | `vl_desvio_padrao_pix_trimestre` | double | 89.50 | `vl_desvio_padrao_pix_trimestre` → `ratio_valor_desvio_padrao` |
| `freq` | `qt_total_pix_trimestre` | int | 47 | `qt_total_pix_trimestre` → `is_first_tx_trimestre` |
| `freq` | `qt_pix_dia_maximo_trimestre` | int | 5 | `qt_pix_dia_maximo_trimestre` |
| `freq` | `qt_intervalo_mediana_trimestre` | double | 4320.0 | `qt_intervalo_mediana_trimestre` → `ratio_intervalo_vs_mediana` → `diff_intervalo_vs_mediana` |
| `freq` | `qt_intervalo_desvio_padrao_trimestre` | double | 3200.0 | `qt_intervalo_desvio_padrao_trimestre` → `zscore_intervalo_aprox` |
| `device` | `qt_aparelhos_distintos_trimestre` | int | 2 | `qt_aparelhos_distintos_trimestre` |
| `device` | `vl_latencia_rede_media_trimestre` | double | 320.0 | `vl_latencia_rede_media_trimestre` → `ratio_latencia_cliente` → `diff_latencia_cliente` |
| `meta` | `dt_ultima_atualizacao` | timestamp | 2026-03-27T14:00:00Z | Controle de freshness |

```bash
create 'fraud_detection:historico_trimestral',
  {NAME => 'valor', VERSIONS => 3, TTL => 86400, COMPRESSION => 'SNAPPY'},
  {NAME => 'freq', VERSIONS => 3, TTL => 86400, COMPRESSION => 'SNAPPY'},
  {NAME => 'device', VERSIONS => 3, TTL => 86400, COMPRESSION => 'SNAPPY'},
  {NAME => 'meta', VERSIONS => 1, TTL => 86400}
```

### 3.4 Tabela: `fraud_detection:historico_recebedores`

**Atualização:** Horária (ou incremental a cada transação, se possível)
**Fonte:** BLK — relação pagador ↔ recebedor
**TTL:** 24 horas

| Column Family | Qualifier | Tipo | Exemplo | Feature(s) do Modelo |
|:---:|-----------|:----:|---------|----------------------|
| `recv` | `recebedores_conhecidos` | JSON/bytes | ["cpf1","cpf2",...] | → `first_receiver_flag` → `distinct_receivers_so_far` |
| `recv` | `contagem:{cpf_recebedor}` | int | 5 | → `receiver_tx_count_prev` → `qt_envio_recebedor_trimestre` → `tp_primeiro_envio_recebedor_trimestre` |
| `chave` | `chaves_usadas` | JSON/bytes | ["ch1","ch2",...] | → `first_key_flag` → `key_tx_count_prev` → `distinct_keys_so_far` |
| `meta` | `dt_ultima_atualizacao` | timestamp | 2026-03-27T14:00:00Z | Controle de freshness |

```bash
create 'fraud_detection:historico_recebedores',
  {NAME => 'recv', VERSIONS => 3, TTL => 86400, COMPRESSION => 'SNAPPY'},
  {NAME => 'chave', VERSIONS => 3, TTL => 86400, COMPRESSION => 'SNAPPY'},
  {NAME => 'meta', VERSIONS => 1, TTL => 86400}
```

### 3.5 Tabela: `fraud_detection:sessao_device`

**Atualização:** Horária (dados do último login/transação)
**Fonte:** MBK (`aut`) — parsing XML
**TTL:** 24 horas

| Column Family | Qualifier | Tipo | Exemplo | Feature(s) do Modelo |
|:---:|-----------|:----:|---------|----------------------|
| `topaz` | `topaz_risk_score` | int | 0 | `topaz_risk_score` → `topaz_missing_flag` → `rule_topaz_score` |
| `device` | `device_name` | string | Samsung Galaxy S23 | → `device_missing_flag` |
| `device` | `ultimo_ip` | string | 189.10.x.x | Referência para behavioral analytics |
| `latencia` | `ultima_latencia_rede_ms` | int | 280 | Referência para features de latência |
| `latencia` | `ultimo_tempo_host_ms` | int | 45 | → `host_time_missing_flag` |
| `auth` | `ultimo_metodo_autenticacao` | string | BIOMETRIA | → `is_login_senha_flag` |
| `meta` | `dt_ultima_atualizacao` | timestamp | 2026-03-27T14:00:00Z | Controle de freshness |

```bash
create 'fraud_detection:sessao_device',
  {NAME => 'topaz', VERSIONS => 3, TTL => 86400, COMPRESSION => 'SNAPPY'},
  {NAME => 'device', VERSIONS => 3, TTL => 86400, COMPRESSION => 'SNAPPY'},
  {NAME => 'latencia', VERSIONS => 3, TTL => 86400, COMPRESSION => 'SNAPPY'},
  {NAME => 'auth', VERSIONS => 3, TTL => 86400, COMPRESSION => 'SNAPPY'},
  {NAME => 'meta', VERSIONS => 1, TTL => 86400}
```

### 3.6 Resumo do Volume no HBase

| Tabela | Rows Estimadas | Tamanho por Row | Total | Atualização |
|--------|:--------------:|:---------------:|:-----:|:-----------:|
| `perfil_cliente` | ~800.000 | ~200 bytes | ~160 MB | Diária |
| `historico_trimestral` | ~800.000 | ~300 bytes | ~240 MB | Horária |
| `historico_recebedores` | ~800.000 | ~500 bytes¹ | ~400 MB | Horária |
| `sessao_device` | ~800.000 | ~250 bytes | ~200 MB | Horária |
| **Total** | — | — | **~1 GB** | — |

¹ *Varia conforme a quantidade de recebedores distintos por CPF. Clientes com muitos recebedores terão rows maiores.*

---

## 4. Spark Job de Materialização

### 4.1 Adaptação do Script de Ingestão Existente

O script `01_ingestao_unificada_mobile_normais.py` já realiza todas as agregações necessárias. A adaptação consiste em:

1. **Manter** toda a lógica de ETL existente (etapas 1 a 5)
2. **Substituir** a etapa 6 (save como tabela Hive) por **write no HBase**
3. **Adicionar** etapa de escrita incremental dos recebedores

### 4.2 Pseudocódigo do Job Adaptado

```python
"""
02_materializa_feature_store_hbase.py
Adaptação do 01_ingestao_unificada para escrever no HBase
"""

from pyspark.sql import SparkSession
import pyspark.sql.functions as F


def create_spark_with_hbase():
    """Spark session com suporte a HBase via Spark-HBase Connector."""
    return (
        SparkSession.builder
        .appName("Feature Store HBase - Fraude PIX")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "12g")
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.dynamicAllocation.maxExecutors", "20")
        .config("spark.sql.adaptive.enabled", "true")
        # HBase configs
        .config(
            "spark.jars",
            "/path/to/shc-core-1.1.3.jar,"
            "/path/to/hbase-client.jar,"
            "/path/to/hbase-common.jar"
        )
        .enableHiveSupport()
        .getOrCreate()
    )


def write_to_hbase(df, table_name, catalog):
    """
    Escreve DataFrame no HBase usando o
    Spark-HBase Connector (SHC).
    """
    df.write \
        .options(catalog=catalog) \
        .format("org.apache.spark.sql.execution.datasources.hbase") \
        .save()


def main():
    spark = create_spark_with_hbase()

    # =========================================================
    # ETAPA 1-5: Reutilizar lógica do 01_ingestao_unificada
    # (clientes, PIX, MBK, JOIN, pré-agregações)
    # =========================================================

    # ... (código existente das etapas 1-5) ...

    # =========================================================
    # ETAPA 6A: Materializar PERFIL_CLIENTE no HBase
    # =========================================================
    print("Materializando perfil_cliente no HBase...")

    catalog_perfil = """{
        "table": {"namespace": "fraud_detection", "name": "perfil_cliente"},
        "rowkey": "cpf",
        "columns": {
            "cpf":                          {"cf":"rowkey", "col":"cpf",                          "type":"string"},
            "nr_idade":                     {"cf":"demo",   "col":"nr_idade",                     "type":"int"},
            "qt_tempo_relacionamento_mes":  {"cf":"demo",   "col":"qt_tempo_relacionamento_mes",  "type":"int"},
            "ds_sexo":                      {"cf":"demo",   "col":"ds_sexo",                      "type":"string"},
            "ds_estado_civil":              {"cf":"demo",   "col":"ds_estado_civil",              "type":"string"},
            "ds_segmento":                  {"cf":"demo",   "col":"ds_segmento",                  "type":"string"},
            "vl_renda_cliente":             {"cf":"renda",  "col":"vl_renda_cliente",             "type":"double"},
            "qt_dependentes":               {"cf":"renda",  "col":"qt_dependentes",               "type":"int"}
        }
    }"""

    df_perfil = (
        df_cliente
        .withColumnRenamed("cd_cpf_pagador", "cpf")
        .select(
            "cpf", "nr_idade", "qt_tempo_relacionamento_mes",
            "ds_sexo", "ds_estado_civil", "ds_segmento",
            "vl_renda_cliente", "qt_dependentes"
        )
    )

    write_to_hbase(df_perfil, "perfil_cliente", catalog_perfil)

    # =========================================================
    # ETAPA 6B: Materializar HISTORICO_TRIMESTRAL no HBase
    # =========================================================
    print("Materializando historico_trimestral no HBase...")

    catalog_historico = """{
        "table": {"namespace": "fraud_detection", "name": "historico_trimestral"},
        "rowkey": "cpf",
        "columns": {
            "cpf":                                  {"cf":"rowkey", "col":"cpf",                                  "type":"string"},
            "vl_mediana_pix_trimestre":              {"cf":"valor",  "col":"vl_mediana_pix_trimestre",              "type":"double"},
            "vl_desvio_padrao_pix_trimestre":        {"cf":"valor",  "col":"vl_desvio_padrao_pix_trimestre",        "type":"double"},
            "qt_total_pix_trimestre":                {"cf":"freq",   "col":"qt_total_pix_trimestre",                "type":"int"},
            "qt_pix_dia_maximo_trimestre":            {"cf":"freq",   "col":"qt_pix_dia_maximo_trimestre",            "type":"int"},
            "qt_intervalo_mediana_trimestre":         {"cf":"freq",   "col":"qt_intervalo_mediana_trimestre",         "type":"double"},
            "qt_intervalo_desvio_padrao_trimestre":   {"cf":"freq",   "col":"qt_intervalo_desvio_padrao_trimestre",   "type":"double"},
            "qt_aparelhos_distintos_trimestre":       {"cf":"device", "col":"qt_aparelhos_distintos_trimestre",       "type":"int"},
            "vl_latencia_rede_media_trimestre":       {"cf":"device", "col":"vl_latencia_rede_media_trimestre",       "type":"double"}
        }
    }"""

    df_hist = (
        df_cpf_agg
        .withColumnRenamed("cd_cpf_pagador", "cpf")
        .select(
            "cpf",
            "vl_mediana_pix_trimestre",
            "vl_desvio_padrao_pix_trimestre",
            "qt_total_pix_trimestre",
            "qt_pix_dia_maximo_trimestre",
            "qt_intervalo_mediana_trimestre",
            "qt_intervalo_desvio_padrao_trimestre",
            "qt_aparelhos_distintos_trimestre",
            "vl_latencia_rede_media_trimestre",
        )
    )

    write_to_hbase(df_hist, "historico_trimestral", catalog_historico)

    # =========================================================
    # ETAPA 6C: Materializar HISTORICO_RECEBEDORES no HBase
    # =========================================================
    print("Materializando historico_recebedores no HBase...")

    # Agrega recebedores e chaves por CPF
    df_recv = (
        df_base
        .groupBy("cd_cpf_pagador")
        .agg(
            F.collect_set("cd_cpf_cnpj_recebedor").alias("recebedores_conhecidos"),
            F.collect_set("ds_chave_pix").alias("chaves_usadas"),
        )
        .withColumn(
            "recebedores_json",
            F.to_json(F.col("recebedores_conhecidos"))
        )
        .withColumn(
            "chaves_json",
            F.to_json(F.col("chaves_usadas"))
        )
        .withColumnRenamed("cd_cpf_pagador", "cpf")
    )

    # Contagem por par (pagador, recebedor)
    df_recv_count = (
        df_base
        .groupBy("cd_cpf_pagador", "cd_cpf_cnpj_recebedor")
        .agg(F.count("cd_pix").alias("qt_envios"))
    )

    # Escrita via HBase API para qualifiers dinâmicos
    # (ver seção 4.3 para implementação detalhada)
    write_recv_to_hbase(df_recv, df_recv_count)

    # =========================================================
    # ETAPA 6D: Materializar SESSAO_DEVICE no HBase
    # =========================================================
    print("Materializando sessao_device no HBase...")

    # Pega o registro MBK mais recente por CPF
    from pyspark.sql.window import Window

    w_recent = Window.partitionBy("cd_cpf_pagador").orderBy(
        F.col("dt_pix").desc()
    )

    df_device = (
        df_base
        .withColumn("rn", F.row_number().over(w_recent))
        .filter(F.col("rn") == 1)
        .drop("rn")
        .withColumnRenamed("cd_cpf_pagador", "cpf")
        .select(
            "cpf",
            "topaz_risk_score",
            "device_name",
            "ip_address",
            "latencia_rede_ms",
            "tempo_processamento_host_ms",
            "metodo_autenticacao",
        )
    )

    write_to_hbase(df_device, "sessao_device", catalog_device)

    print("Feature Store HBase materializada com sucesso.")


if __name__ == "__main__":
    main()
```

### 4.3 Escrita de Recebedores com Qualifiers Dinâmicos

Para a tabela de recebedores, cada par (pagador, recebedor) precisa de um qualifier dinâmico. Isso requer a API Java do HBase via PySpark:

```python
def write_recv_to_hbase(df_recv, df_recv_count):
    """
    Escreve recebedores no HBase com qualifiers dinâmicos.
    Usa a API HBase via JVM gateway do PySpark.
    """
    spark = df_recv.sparkSession
    sc = spark.sparkContext

    conf = sc._jvm.org.apache.hadoop.hbase.HBaseConfiguration.create()
    connection = sc._jvm.org.apache.hadoop.hbase.client \
        .ConnectionFactory.createConnection(conf)
    table = connection.getTable(
        sc._jvm.org.apache.hadoop.hbase.TableName
        .valueOf("fraud_detection", "historico_recebedores")
    )

    # Batch de puts
    rows = df_recv_count.collect()

    batch = []
    for row in rows:
        cpf = str(row["cd_cpf_pagador"]).zfill(14)
        recv = str(row["cd_cpf_cnpj_recebedor"])
        count = int(row["qt_envios"])

        put = sc._jvm.org.apache.hadoop.hbase.client.Put(
            cpf.encode("utf-8")
        )
        put.addColumn(
            b"recv",
            f"contagem:{recv}".encode("utf-8"),
            str(count).encode("utf-8"),
        )
        batch.append(put)

        if len(batch) >= 10000:
            table.put(batch)
            batch = []

    if batch:
        table.put(batch)

    table.close()
    connection.close()
```

### 4.4 Agendamento do Job

| Job | Frequência | Horário | Duração Estimada | Dependências |
|-----|:---------:|:-------:|:----------------:|-------------|
| `perfil_cliente` | Diária | 06:00 | ~15 min | AOX/DNA atualizados |
| `historico_trimestral` | Horária | :00 de cada hora | ~20 min | BLK atualizado |
| `historico_recebedores` | Horária | :05 de cada hora | ~25 min | BLK atualizado |
| `sessao_device` | Horária | :10 de cada hora | ~15 min | MBK atualizado |

**Orquestração sugerida:** Apache Airflow (se disponível) ou cron job no edge node do Hadoop.

---

## 5. Integração com a API de Inferência

### 5.1 Cliente HBase na API Python

```python
"""
hbase_feature_store.py
Cliente HBase para a API de inferência em tempo real.
"""

import happybase
import json
import logging
import time
from typing import Dict, Optional


logger = logging.getLogger(__name__)


class HBaseFeatureStore:
    """
    Cliente para buscar features pré-materializadas no HBase.
    Usado pela API de inferência em tempo real.
    """

    def __init__(
        self,
        host: str = "hbase-master.brb.internal",
        port: int = 9090,
        timeout: int = 5000,
        pool_size: int = 10,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout

        # Pool de conexões para alta concorrência
        self.pool = happybase.ConnectionPool(
            size=pool_size,
            host=host,
            port=port,
            timeout=timeout,
            transport="buffered",
        )

        # Nomes das tabelas
        self.TABLE_PERFIL = b"fraud_detection:perfil_cliente"
        self.TABLE_HISTORICO = b"fraud_detection:historico_trimestral"
        self.TABLE_RECEBEDORES = b"fraud_detection:historico_recebedores"
        self.TABLE_DEVICE = b"fraud_detection:sessao_device"

        # Defaults conservadores (usados quando HBase não responde)
        self.DEFAULTS = {
            # Perfil
            "nr_idade": 40,
            "qt_tempo_relacionamento_mes": 58,
            "ds_sexo": "Informação ausente",
            "ds_estado_civil": "Informação ausente",
            "ds_segmento": "Informação ausente",
            "vl_renda_cliente": 0.0,
            "qt_dependentes": 0,
            # Histórico trimestral
            "qt_total_pix_trimestre": 1,
            "vl_mediana_pix_trimestre": 0.0,
            "vl_desvio_padrao_pix_trimestre": 0.0,
            "qt_intervalo_mediana_trimestre": 0.0,
            "qt_intervalo_desvio_padrao_trimestre": 0.0,
            "qt_pix_dia_maximo_trimestre": 1,
            "qt_aparelhos_distintos_trimestre": 1,
            "vl_latencia_rede_media_trimestre": 0.0,
            # Device/Sessão
            "topaz_risk_score": None,
            "device_name": None,
            "latencia_rede_ms": None,
            "tempo_processamento_host_ms": None,
            "metodo_autenticacao": None,
        }

    def _make_row_key(self, cpf: str) -> bytes:
        """Cria row key padronizada a partir do CPF."""
        return cpf.strip().zfill(14).encode("utf-8")

    def get_perfil(self, cpf: str) -> Dict:
        """Busca perfil do cliente no HBase."""
        row_key = self._make_row_key(cpf)
        t0 = time.time()

        try:
            with self.pool.connection() as conn:
                table = conn.table(self.TABLE_PERFIL)
                row = table.row(row_key)

            if not row:
                logger.warning(f"Perfil não encontrado: {cpf[:4]}***")
                return {k: self.DEFAULTS[k] for k in [
                    "nr_idade", "qt_tempo_relacionamento_mes",
                    "ds_sexo", "ds_estado_civil", "ds_segmento",
                    "vl_renda_cliente", "qt_dependentes",
                ]}

            return {
                "nr_idade": int(row.get(b"demo:nr_idade", b"40")),
                "qt_tempo_relacionamento_mes": int(
                    row.get(b"demo:qt_tempo_relacionamento_mes", b"58")
                ),
                "ds_sexo": row.get(
                    b"demo:ds_sexo", b"Informacao ausente"
                ).decode("utf-8"),
                "ds_estado_civil": row.get(
                    b"demo:ds_estado_civil", b"Informacao ausente"
                ).decode("utf-8"),
                "ds_segmento": row.get(
                    b"demo:ds_segmento", b"Informacao ausente"
                ).decode("utf-8"),
                "vl_renda_cliente": float(
                    row.get(b"renda:vl_renda_cliente", b"0.0")
                ),
                "qt_dependentes": int(
                    row.get(b"renda:qt_dependentes", b"0")
                ),
            }

        except Exception as e:
            logger.error(f"Erro HBase perfil: {e}")
            return {k: self.DEFAULTS[k] for k in [
                "nr_idade", "qt_tempo_relacionamento_mes",
                "ds_sexo", "ds_estado_civil", "ds_segmento",
                "vl_renda_cliente", "qt_dependentes",
            ]}

        finally:
            elapsed = (time.time() - t0) * 1000
            logger.debug(f"HBase perfil: {elapsed:.1f}ms")

    def get_historico_trimestral(self, cpf: str) -> Dict:
        """Busca histórico trimestral no HBase."""
        row_key = self._make_row_key(cpf)
        t0 = time.time()

        try:
            with self.pool.connection() as conn:
                table = conn.table(self.TABLE_HISTORICO)
                row = table.row(row_key)

            if not row:
                logger.warning(
                    f"Histórico não encontrado: {cpf[:4]}***"
                )
                return {k: self.DEFAULTS[k] for k in [
                    "qt_total_pix_trimestre",
                    "vl_mediana_pix_trimestre",
                    "vl_desvio_padrao_pix_trimestre",
                    "qt_intervalo_mediana_trimestre",
                    "qt_intervalo_desvio_padrao_trimestre",
                    "qt_pix_dia_maximo_trimestre",
                    "qt_aparelhos_distintos_trimestre",
                    "vl_latencia_rede_media_trimestre",
                ]}

            return {
                "qt_total_pix_trimestre": int(
                    row.get(b"freq:qt_total_pix_trimestre", b"1")
                ),
                "vl_mediana_pix_trimestre": float(
                    row.get(b"valor:vl_mediana_pix_trimestre", b"0.0")
                ),
                "vl_desvio_padrao_pix_trimestre": float(
                    row.get(
                        b"valor:vl_desvio_padrao_pix_trimestre", b"0.0"
                    )
                ),
                "qt_intervalo_mediana_trimestre": float(
                    row.get(
                        b"freq:qt_intervalo_mediana_trimestre", b"0.0"
                    )
                ),
                "qt_intervalo_desvio_padrao_trimestre": float(
                    row.get(
                        b"freq:qt_intervalo_desvio_padrao_trimestre",
                        b"0.0",
                    )
                ),
                "qt_pix_dia_maximo_trimestre": int(
                    row.get(b"freq:qt_pix_dia_maximo_trimestre", b"1")
                ),
                "qt_aparelhos_distintos_trimestre": int(
                    row.get(
                        b"device:qt_aparelhos_distintos_trimestre", b"1"
                    )
                ),
                "vl_latencia_rede_media_trimestre": float(
                    row.get(
                        b"device:vl_latencia_rede_media_trimestre",
                        b"0.0",
                    )
                ),
            }

        except Exception as e:
            logger.error(f"Erro HBase histórico: {e}")
            return {k: self.DEFAULTS[k] for k in [
                "qt_total_pix_trimestre",
                "vl_mediana_pix_trimestre",
                "vl_desvio_padrao_pix_trimestre",
                "qt_intervalo_mediana_trimestre",
                "qt_intervalo_desvio_padrao_trimestre",
                "qt_pix_dia_maximo_trimestre",
                "qt_aparelhos_distintos_trimestre",
                "vl_latencia_rede_media_trimestre",
            ]}

        finally:
            elapsed = (time.time() - t0) * 1000
            logger.debug(f"HBase historico: {elapsed:.1f}ms")

    def get_recebedores(self, cpf: str) -> Dict:
        """Busca histórico de recebedores no HBase."""
        row_key = self._make_row_key(cpf)
        t0 = time.time()

        try:
            with self.pool.connection() as conn:
                table = conn.table(self.TABLE_RECEBEDORES)
                row = table.row(row_key)

            if not row:
                return {
                    "recebedores_conhecidos": set(),
                    "chaves_usadas": set(),
                    "contagem_por_recebedor": {},
                }

            # Parse recebedores
            recv_json = row.get(
                b"recv:recebedores_conhecidos", b"[]"
            ).decode("utf-8")
            recebedores = set(json.loads(recv_json))

            # Parse chaves
            chaves_json = row.get(
                b"chave:chaves_usadas", b"[]"
            ).decode("utf-8")
            chaves = set(json.loads(chaves_json))

            # Parse contagens (qualifiers dinâmicos)
            contagem = {}
            for key, value in row.items():
                if key.startswith(b"recv:contagem:"):
                    recv_cpf = key.decode("utf-8").split(
                        "contagem:"
                    )[1]
                    contagem[recv_cpf] = int(value)

            return {
                "recebedores_conhecidos": recebedores,
                "chaves_usadas": chaves,
                "contagem_por_recebedor": contagem,
            }

        except Exception as e:
            logger.error(f"Erro HBase recebedores: {e}")
            return {
                "recebedores_conhecidos": set(),
                "chaves_usadas": set(),
                "contagem_por_recebedor": {},
            }

        finally:
            elapsed = (time.time() - t0) * 1000
            logger.debug(f"HBase recebedores: {elapsed:.1f}ms")

    def get_device(self, cpf: str) -> Dict:
        """Busca dados de device/sessão no HBase."""
        row_key = self._make_row_key(cpf)
        t0 = time.time()

        try:
            with self.pool.connection() as conn:
                table = conn.table(self.TABLE_DEVICE)
                row = table.row(row_key)

            if not row:
                return {
                    "topaz_risk_score": None,
                    "device_name": None,
                    "latencia_rede_ms": None,
                    "tempo_processamento_host_ms": None,
                    "metodo_autenticacao": None,
                }

            def safe_int(val, default=None):
                if val is None:
                    return default
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return default

            def safe_str(val, default=None):
                if val is None:
                    return default
                decoded = val.decode("utf-8")
                return decoded if decoded else default

            return {
                "topaz_risk_score": safe_int(
                    row.get(b"topaz:topaz_risk_score")
                ),
                "device_name": safe_str(
                    row.get(b"device:device_name")
                ),
                "latencia_rede_ms": safe_int(
                    row.get(b"latencia:ultima_latencia_rede_ms")
                ),
                "tempo_processamento_host_ms": safe_int(
                    row.get(b"latencia:ultimo_tempo_host_ms")
                ),
                "metodo_autenticacao": safe_str(
                    row.get(b"auth:ultimo_metodo_autenticacao")
                ),
            }

        except Exception as e:
            logger.error(f"Erro HBase device: {e}")
            return {
                "topaz_risk_score": None,
                "device_name": None,
                "latencia_rede_ms": None,
                "tempo_processamento_host_ms": None,
                "metodo_autenticacao": None,
            }

        finally:
            elapsed = (time.time() - t0) * 1000
            logger.debug(f"HBase device: {elapsed:.1f}ms")

    def get_all_features(self, cpf: str) -> Dict:
        """
        Busca TODAS as features de um CPF em um único call.
        Faz os 4 GETs em sequência (otimizável com multi-get).
        """
        t0 = time.time()

        perfil = self.get_perfil(cpf)
        historico = self.get_historico_trimestral(cpf)
        recebedores = self.get_recebedores(cpf)
        device = self.get_device(cpf)

        result = {}
        result.update(perfil)
        result.update(historico)
        result["_recebedores"] = recebedores
        result.update(device)

        elapsed = (time.time() - t0) * 1000
        logger.info(
            f"HBase total ({cpf[:4]}***): {elapsed:.1f}ms "
            f"(4 tabelas)"
        )

        return result

    def health_check(self) -> bool:
        """Verifica se o HBase está acessível."""
        try:
            with self.pool.connection() as conn:
                conn.tables()
            return True
        except Exception:
            return False
```

### 5.2 Integração com o Orquestrador Existente

A integração no `decision_engine.py` (orquestrador) requer mudanças mínimas:

```python
"""
Trecho de integração no orquestrador existente.
Substitui os dados mockados/estáticos por lookup no HBase.
"""

from hbase_feature_store import HBaseFeatureStore


class PixFraudOrchestrator:

    def __init__(self):
        # ... (carregamento de modelos existente) ...

        # Nova: Feature Store HBase
        self.feature_store = HBaseFeatureStore(
            host="hbase-master.brb.internal",
            port=9090,
            pool_size=10,
        )

    def _enrich_transaction(self, tx: dict) -> dict:
        """
        Enriquece a transação com features do HBase.
        Substitui o método anterior que usava dados estáticos.
        """
        cpf = tx["cd_cpf_pagador"]
        cpf_recebedor = tx["cd_cpf_cnpj_recebedor"]
        chave_pix = tx.get("ds_chave_pix", "")

        # 1. Buscar features do HBase (~10-20ms total)
        hbase_data = self.feature_store.get_all_features(cpf)

        # 2. Mesclar dados da transação + HBase
        enriched = {**tx, **hbase_data}

        # 3. Calcular features de recebedor
        recebedores = hbase_data.get("_recebedores", {})
        recv_set = recebedores.get("recebedores_conhecidos", set())
        recv_count = recebedores.get(
            "contagem_por_recebedor", {}
        )
        chaves_set = recebedores.get("chaves_usadas", set())

        enriched["first_receiver_flag"] = (
            0 if cpf_recebedor in recv_set else 1
        )
        enriched["receiver_tx_count_prev"] = recv_count.get(
            cpf_recebedor, 0
        )
        enriched["distinct_receivers_so_far"] = len(recv_set)
        enriched["first_key_flag"] = (
            0 if chave_pix in chaves_set else 1
        )
        enriched["key_tx_count_prev"] = sum(
            1 for c in chaves_set if c == chave_pix
        )
        enriched["distinct_keys_so_far"] = len(chaves_set)

        # 4. Calcular features sequenciais (cache em memória)
        seq_features = self._compute_sequential_features(
            cpf, tx["dt_pix"]
        )
        enriched.update(seq_features)

        # 5. Remover campo interno
        enriched.pop("_recebedores", None)

        return enriched

    def _compute_sequential_features(
        self, cpf: str, dt_pix
    ) -> dict:
        """
        Calcula features sequenciais usando o cache
        em memória (já existente no orquestrador).
        """
        history = self._customer_history.get(cpf, [])

        # minutes_since_prev_tx
        if history:
            last_ts = history[-1]
            delta = (dt_pix - last_ts).total_seconds() / 60
            minutes_since_prev = max(0, delta)
        else:
            minutes_since_prev = 0

        # tx_count_prev_30m
        cutoff_30m = dt_pix - timedelta(minutes=30)
        tx_in_30m = sum(1 for ts in history if ts >= cutoff_30m)

        # burst_30m_flag
        burst_flag = 1 if tx_in_30m >= 1 else 0

        # qt_intervalo_transacao_minuto
        interval = minutes_since_prev

        # Atualizar cache
        history.append(dt_pix)
        # Manter apenas últimas 2 horas
        cutoff_2h = dt_pix - timedelta(hours=2)
        self._customer_history[cpf] = [
            ts for ts in history if ts >= cutoff_2h
        ]

        return {
            "minutes_since_prev_tx": minutes_since_prev,
            "tx_count_prev_30m": tx_in_30m,
            "burst_30m_flag": burst_flag,
            "qt_intervalo_transacao_minuto": interval,
        }
```

---

## 6. Estratégia de Degradação Graciosa

### 6.1 Cenários de Falha e Respostas

O sistema deve continuar operando **mesmo quando o HBase estiver indisponível**. A estratégia é de **degradação graciosa em 3 níveis**:

```
Transação PIX chega
        │
        ▼
   HBase disponível?
        │
       SIM ──────────────────────────► NÍVEL 1: Dados completos
        │                               (52 features, performance máxima)
       NÃO
        │
        ▼
   Cache local tem dados
   recentes desse CPF?
        │
       SIM ──────────────────────────► NÍVEL 2: Cache stale
        │                               (dados de até 24h atrás)
       NÃO
        │
        ▼
   Banco transacional
   (DB2/Oracle) responde
   em < 100ms?
        │
       SIM ──────────────────────────► NÍVEL 3: Dados parciais
        │                               (só perfil estático, sem histórico)
       NÃO
        │
        ▼
   NÍVEL 4: Defaults conservadores ──► Modelo ainda funciona
                                        (missing_flags ativadas,
                                         Cascade Rules ativas,
                                         qualidade reduzida)
```

### 6.2 Impacto Estimado por Nível de Degradação

| Nível | Dados Disponíveis | Features Afetadas | Impacto no AUC (estimado) | Impacto no Recall |
|:-----:|-------------------|:-----------------:|:-------------------------:|:-----------------:|
| 1 | Tudo (HBase ok) | 0 de 52 | 0% (baseline) | 98,75% |
| 2 | Cache stale (24h) | ~3 (histórico desatualizado) | ~0,1% | ~98% |
| 3 | Só perfil estático | ~20 (sem histórico/device) | ~2-5% | ~90-95% |
| 4 | Defaults | ~30 (quase tudo ausente) | ~10-20% | ~75-85% |

**No nível 4, o sistema ainda oferece proteção porque:**
- As **missing_flags** (`device_missing_flag`, `topaz_missing_flag`, `host_time_missing_flag`, `renda_missing_flag`) são features do modelo — ele aprendeu que dados ausentes são um sinal de risco
- As **Cascade Rules** funcionam mesmo sem dados completos (C1 e C2 usam apenas features sequenciais do cache em memória)
- O **Isolation Forest** detecta anomalias estruturais independentemente de dados completos
- Os módulos de **Engenharia Social** e **Behavioral Analytics** usam features da transação + cache

### 6.3 Implementação da Degradação

```python
class HBaseFeatureStoreWithFallback(HBaseFeatureStore):
    """
    Feature Store com degradação graciosa em 4 níveis.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._local_cache = {}  # CPF → {dados, timestamp}
        self._cache_ttl = 86400  # 24 horas
        self._metrics = {
            "level_1": 0,
            "level_2": 0,
            "level_3": 0,
            "level_4": 0,
        }

    def get_all_features_with_fallback(
        self, cpf: str
    ) -> tuple[Dict, int]:
        """
        Retorna (features, nível_degradação).
        """

        # NÍVEL 1: HBase
        try:
            data = self.get_all_features(cpf)
            # Atualizar cache local
            self._local_cache[cpf] = {
                "data": data,
                "timestamp": time.time(),
            }
            self._metrics["level_1"] += 1
            return data, 1

        except Exception as e:
            logger.warning(f"HBase falhou: {e}")

        # NÍVEL 2: Cache local
        cached = self._local_cache.get(cpf)
        if cached:
            age = time.time() - cached["timestamp"]
            if age < self._cache_ttl:
                logger.info(
                    f"Usando cache local "
                    f"({age/3600:.1f}h): {cpf[:4]}***"
                )
                self._metrics["level_2"] += 1
                return cached["data"], 2

        # NÍVEL 3: Query direta ao DB2 (só perfil)
        try:
            perfil = self._query_db2_perfil(cpf)
            if perfil:
                data = {**self.DEFAULTS, **perfil}
                self._metrics["level_3"] += 1
                return data, 3
        except Exception:
            pass

        # NÍVEL 4: Defaults conservadores
        logger.warning(
            f"Usando defaults: {cpf[:4]}*** "
            f"(todas as fontes indisponíveis)"
        )
        self._metrics["level_4"] += 1
        return dict(self.DEFAULTS), 4

    def _query_db2_perfil(
        self, cpf: str
    ) -> Optional[Dict]:
        """
        Query direta ao DB2 para dados estáticos.
        Timeout agressivo de 100ms.
        """
        # Implementar conforme driver DB2 do banco
        # (jaydebeapi ou ibm_db)
        pass

    def get_degradation_metrics(self) -> Dict:
        """Retorna métricas de degradação para monitoramento."""
        total = sum(self._metrics.values()) or 1
        return {
            "total_requests": total,
            "level_1_pct": self._metrics["level_1"] / total * 100,
            "level_2_pct": self._metrics["level_2"] / total * 100,
            "level_3_pct": self._metrics["level_3"] / total * 100,
            "level_4_pct": self._metrics["level_4"] / total * 100,
        }
```

---

## 7. Mapeamento Completo: Features × Fonte em Produção

Esta tabela é o **contrato de dados** entre o batch (Spark → HBase) e o tempo real (API → HBase):

| # | Feature do Modelo | Fonte em Produção | Tabela HBase | Column Family:Qualifier | Calculada? |
|:-:|-------------------|:-----------------:|:------------:|:-----------------------:|:----------:|
| 1 | `vl_pix` | Transação online | — | — | Não |
| 2 | `vl_pix_over_1000_flag` | Derivada | — | — | `vl_pix >= 1000` |
| 3 | `vl_mediana_pix_trimestre` | HBase | `historico_trimestral` | `valor:vl_mediana_pix_trimestre` | Não |
| 4 | `vl_desvio_padrao_pix_trimestre` | HBase | `historico_trimestral` | `valor:vl_desvio_padrao_pix_trimestre` | Não |
| 5 | `ratio_valor_mediana` | Derivada | — | — | `vl_pix / mediana` |
| 6 | `diff_valor_mediana` | Derivada | — | — | `vl_pix - mediana` |
| 7 | `ratio_valor_desvio_padrao` | Derivada | — | — | `vl_pix / desvio` |
| 8 | `zscore_valor_aprox` | Derivada | — | — | `(vl_pix - mediana) / desvio` |
| 9 | `qt_total_pix_trimestre` | HBase | `historico_trimestral` | `freq:qt_total_pix_trimestre` | Não |
| 10 | `is_first_tx_trimestre` | Derivada | — | — | `qt_total == 0` |
| 11 | `qt_intervalo_transacao_minuto` | Cache memória | — | — | Sequencial |
| 12 | `qt_intervalo_mediana_trimestre` | HBase | `historico_trimestral` | `freq:qt_intervalo_mediana_trimestre` | Não |
| 13 | `qt_intervalo_desvio_padrao_trimestre` | HBase | `historico_trimestral` | `freq:qt_intervalo_desvio_padrao_trimestre` | Não |
| 14 | `qt_pix_dia_maximo_trimestre` | HBase | `historico_trimestral` | `freq:qt_pix_dia_maximo_trimestre` | Não |
| 15 | `ratio_intervalo_vs_mediana` | Derivada | — | — | `intervalo / mediana_intervalo` |
| 16 | `diff_intervalo_vs_mediana` | Derivada | — | — | `intervalo - mediana_intervalo` |
| 17 | `zscore_intervalo_aprox` | Derivada | — | — | Calculada |
| 18 | `minutes_since_prev_tx` | Cache memória | — | — | Sequencial |
| 19 | `tx_count_prev_30m` | Cache memória | — | — | Sequencial |
| 20 | `burst_30m_flag` | Cache memória | — | — | `tx_count_30m >= 1` |
| 21 | `receiver_tx_count_prev` | HBase | `historico_recebedores` | `recv:contagem:{cpf_recv}` | Não |
| 22 | `first_receiver_flag` | HBase + lógica | `historico_recebedores` | `recv:recebedores_conhecidos` | `recv not in set` |
| 23 | `distinct_receivers_so_far` | HBase | `historico_recebedores` | `recv:recebedores_conhecidos` | `len(set)` |
| 24 | `tp_primeiro_envio_recebedor_trimestre` | HBase | `historico_recebedores` | `recv:contagem:{cpf_recv}` | `count == 0` |
| 25 | `qt_envio_recebedor_trimestre` | HBase | `historico_recebedores` | `recv:contagem:{cpf_recv}` | Não |
| 26 | `pix_key_random_flag` | Transação online | — | — | `len(chave) >= 32` |
| 27 | `key_tx_count_prev` | HBase | `historico_recebedores` | `chave:chaves_usadas` | Lógica |
| 28 | `first_key_flag` | HBase + lógica | `historico_recebedores` | `chave:chaves_usadas` | `chave not in set` |
| 29 | `distinct_keys_so_far` | HBase | `historico_recebedores` | `chave:chaves_usadas` | `len(set)` |
| 30 | `hour` | Transação online | — | — | `dt_pix.hour` |
| 31 | `nr_idade` | HBase | `perfil_cliente` | `demo:nr_idade` | Não |
| 32 | `qt_tempo_relacionamento_mes` | HBase | `perfil_cliente` | `demo:qt_tempo_relacionamento_mes` | Não |
| 33 | `qt_aparelhos_distintos_trimestre` | HBase | `historico_trimestral` | `device:qt_aparelhos_distintos_trimestre` | Não |
| 34 | `vl_latencia_rede_media_trimestre` | HBase | `historico_trimestral` | `device:vl_latencia_rede_media_trimestre` | Não |
| 35 | `ratio_latencia_cliente` | Derivada | — | — | `latencia_atual / media_trim` |
| 36 | `diff_latencia_cliente` | Derivada | — | — | `latencia_atual - media_trim` |
| 37 | `device_missing_flag` | Derivada | — | — | `device_name is None` |
| 38 | `host_time_missing_flag` | Derivada | — | — | `host_ms is None` |
| 39 | `topaz_missing_flag` | Derivada | — | — | `topaz_score is None` |
| 40 | `topaz_risk_score` | HBase | `sessao_device` | `topaz:topaz_risk_score` | Não |
| 41 | `rule_age_score` | Derivada | — | — | Regra de negócio |
| 42 | `rule_relationship_score` | Derivada | — | — | Regra de negócio |
| 43 | `rule_random_key_score` | Derivada | — | — | Regra de negócio |
| 44 | `rule_topaz_score` | Derivada | — | — | Regra de negócio |
| 45 | `rule_score_raw` | Derivada | — | — | Soma das rules |
| 46 | `ratio_pix_renda` | Derivada | — | — | `vl_pix / renda` |
| 47 | `vl_renda_cliente` | HBase | `perfil_cliente` | `renda:vl_renda_cliente` | Não |
| 48 | `pix_over_50pct_renda_flag` | Derivada | — | — | `vl_pix > renda * 0.5` |
| 49 | `renda_missing_flag` | Derivada | — | — | `renda == 0` |
| 50 | `perfil_vulneravel_se_flag` | Derivada | — | — | Lógica combinada |
| 51 | `is_viuvo_flag` | Derivada | — | — | `estado_civil == VIUVO` |
| 52 | `is_segmento_premium_flag` | Derivada | — | — | `segmento in (EXCLUSIVO, PRIVATE)` |

**Resumo por fonte:**

| Fonte | Qtd Features |
|-------|:---:|
| Transação online (7 campos brutos) | 4 features diretas + 2 derivadas |
| HBase (4 tabelas) | 18 features diretas |
| Cache memória (sequenciais) | 4 features |
| Derivadas (calculadas pelo preprocessing) | 26 features |
| **Total** | **52 features** |

---

## 8. Monitoramento e Observabilidade

### 8.1 Métricas a Coletar

| Métrica | Onde Coletar | Alerta Se |
|---------|:------------|----------|
| **Latência HBase (p50, p95, p99)** | API | p99 > 20ms |
| **Latência total do pipeline** | API | p99 > 300ms |
| **Taxa de degradação (%)** | API | Nível 3+4 > 5% |
| **Freshness do HBase** | Spark job | Última atualização > 3 horas |
| **Volume de rows no HBase** | HBase metrics | Queda > 10% vs dia anterior |
| **Score distribution (drift)** | API logs | Mediana de scores muda > 20% |
| **Taxa de BLOQUEAR** | API logs | > 2% das transações (vs baseline 1%) |
| **Taxa de CONFIRMAR** | API logs | > 3% das transações |
| **FP reportados (clientes reclamam)** | Mesa de Fraude | > 20/dia |

### 8.2 Dashboard Sugerido

```
┌─────────────────────────────────────────────────────────┐
│           MOTOR ANTIFRAUDE PIX — DASHBOARD               │
├──────────────────┬──────────────────┬────────────────────┤
│  LATÊNCIA (ms)   │  DEGRADAÇÃO      │  DECISÕES HOJE     │
│  p50: 142ms ✅   │  Nível 1: 99.2% │  Aprovadas: 45.231 │
│  p95: 168ms ✅   │  Nível 2: 0.6%  │  Confirmar:    312 │
│  p99: 203ms ✅   │  Nível 3: 0.1%  │  Bloquear:      87 │
│  max: 312ms ✅   │  Nível 4: 0.1%  │  Taxa interv: 0.87%│
├──────────────────┴──────────────────┴────────────────────┤
│  HBASE FRESHNESS                                         │
│  perfil_cliente:       06:15 (hoje) ✅                   │
│  historico_trimestral: 14:02 (hoje) ✅                   │
│  historico_recebedores:14:05 (hoje) ✅                   │
│  sessao_device:        14:10 (hoje) ✅                   │
├──────────────────────────────────────────────────────────┤
│  SCORE DISTRIBUTION (últimas 24h)                        │
│  Normal: mediana=0.00004  p95=0.0012  max=0.844         │
│  Status: ✅ Sem drift significativo                      │
└──────────────────────────────────────────────────────────┘
```

---

## 9. Plano de Implementação — Fases

### Fase 1: Infraestrutura HBase (Semanas 1-2)

| Tarefa | Responsável | Entregável |
|--------|:-----------:|-----------|
| Provisionar namespace `fraud_detection` no HBase | Infra/DBA | Namespace criado |
| Criar as 4 tabelas com Column Families | Infra/DBA | Tabelas criadas |
| Configurar RegionServers e replicação | Infra | HBase operacional |
| Testar latência de leitura/escrita | Squad IA | Relatório de benchmark |
| Configurar monitoramento (HBase metrics) | Infra | Alertas configurados |

### Fase 2: Job de Materialização (Semanas 2-4)

| Tarefa | Responsável | Entregável |
|--------|:-----------:|-----------|
| Adaptar `01_ingestao_unificada` para write no HBase | Squad IA | Script `02_materializa_feature_store_hbase.py` |
| Testar materialização completa (800K CPFs) | Squad IA | HBase populado + relatório de cobertura |
| Configurar agendamento (Airflow/cron) | Squad IA + Infra | Jobs rodando horariamente |
| Validar freshness e TTL | Squad IA | Dados atualizados < 2h |

### Fase 3: Integração com API (Semanas 4-6)

| Tarefa | Responsável | Entregável |
|--------|:-----------:|-----------|
| Implementar `HBaseFeatureStore` | Squad IA | Cliente HBase funcional |
| Implementar degradação graciosa | Squad IA | 4 níveis de fallback |
| Integrar com orquestrador existente | Squad IA | API usando HBase em vez de dados estáticos |
| Testes de integração (15 cenários) | Squad IA | Todos os testes passando |
| Teste de carga (100 req/s) | Squad IA + Infra | Relatório de performance |

### Fase 4: Shadow Mode (Semanas 6-10)

| Tarefa | Responsável | Entregável |
|--------|:-----------:|-----------|
| Deploy em produção (shadow) | Squad IA + Infra | API rodando em paralelo |
| Coletar logs de decisão | Squad IA | Dataset de shadow mode |
| Comparar decisões com fraudes reais | Squad IA + GEPFRA | Relatório de performance real |
| Ajustar thresholds se necessário | Squad IA | Thresholds calibrados |
| Dashboard de monitoramento | Squad IA | Dashboard operacional |

### Fase 5: Produção Plena (Semana 10+)

| Tarefa | Responsável | Entregável |
|--------|:-----------:|-----------|
| Ativar bloqueio automático | Gestão + Squad IA | Motor ativo |
| Integrar feedback da GEPFRA | Squad IA | Feedback Loop funcional |
| Primeiro ciclo de retreino | Squad IA | Modelo atualizado |
| Documentação operacional final | Squad IA | Runbook completo |

---

## 10. Riscos e Mitigações

| # | Risco | Probabilidade | Impacto | Mitigação |
|:-:|-------|:------------:|:-------:|-----------|
| 1 | HBase indisponível por manutenção | Média | Alto | Degradação graciosa nível 2-4; cache local com TTL 24h |
| 2 | Dados no HBase desatualizados (job Spark falha) | Média | Médio | Monitoramento de freshness; alerta se > 3h; modelo funciona com dados de até 24h |
| 3 | Hot-spotting no HBase (CPFs concentrados) | Baixa | Médio | Prefixo hash na row key se necessário; monitorar distribuição de requests |
| 4 | Latência do HBase acima do esperado sob carga | Baixa | Alto | Pool de conexões; benchmark prévio; fallback para cache local |
| 5 | Inconsistência entre features de treino e produção (train-serve skew) | Média | Alto | Validação periódica: comparar distribuição de features do HBase com a base de treino |
| 6 | Volume de recebedores cresce demais para alguns CPFs | Baixa | Baixo | Limitar a 500 recebedores mais recentes; prune no job Spark |

---

## 11. Resumo Executivo

| Aspecto | Detalhe |
|---------|---------|
| **Objetivo** | Fornecer as 52 features do modelo em tempo real via Feature Store HBase |
| **Tecnologia** | Apache HBase (4 tabelas, ~1 GB, ~800K rows) |
| **Latência de lookup** | 5-10ms (vs 3-10s no Hadoop direto) |
| **Latência total do pipeline** | ~150ms (vs 10.000ms do SLA BACEN) |
| **Materialização** | Spark job horário (adaptação do script existente) |
| **Resiliência** | 4 níveis de degradação graciosa |
| **Prazo estimado** | 10 semanas até produção plena |
| **Dependências** | Namespace HBase provisionado; Thrift server ativo; rede entre API e HBase |