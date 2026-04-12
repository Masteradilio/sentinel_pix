PROMPT DO USUÁRIO:

"""
Ok, agora parece que chegamos a um ponto ótimo para levar o modelo para produção, mas eu ouvi falar que nesse tipo de modelo, um módulo de Grafos teria oportunidade de capturar coisas que o LGBM, IF, SE e BEH não são capazes de capturar, ou seja, iria enriquecer o modelo com outras nuances. O problema atual é que no banco onde trabalho eu não tenho um banco de dados de grafo disponível, então você acha que, com tudo o que temos disponível, seria possível agregar um módulo de Grafos e salvar os resultados em um banco de dados tipo Hive, Impala ou Hbase?

Considere o contexto que você tem disponível, e analise a possibilidade de que um módulo de Grafos possa agregar valor real no atual modelo, se ele contribuir com a melhoria das métricas atuais seria bom, desde que não aumente os FN, mas será que poderia agregar valor de outras formas sem elevar o SLA da transação, sem adicionar demais tempo de execução para fazer a inferência em tempo real? O que você acha disso? Pode pesquisar na internet sobre o tema pra embasar melhor sua resposta
"""

RESPOSTA DO LLM:



Boa pergunta, Adilio. Deixa eu pesquisar o estado da arte antes de opinar.Adilio, excelente pergunta. Pesquisei bastante e tenho uma opinião bem formada agora. Vou ser direto.

---

## TL;DR

**Sim, grafos podem agregar valor real ao seu pipeline — mas NÃO como GNN em tempo real.** A abordagem correta para o seu contexto é **Graph Feature Engineering (GFE)**: extrair features de grafo em batch/near-real-time e alimentá-las como colunas adicionais no seu LGBM existente. Sem banco de grafos, sem GNN, sem impacto no SLA.

---

## 1. O que os grafos capturam que seu pipeline atual NÃO captura

Seu pipeline v3.0.5 opera **transação por transação** — cada tx é um vetor de features independente. Isso é ótimo para fraude individual, mas **cego para fraude em rede**:

| Padrão de fraude | Seu pipeline v3.0.5 | Com features de grafo |
|---|---|---|
| **Contas mula** (mesma conta recebe de N vítimas) | ❌ Não vê — cada envio é tx isolada | ✅ `in_degree_destinatario` alto |
| **Fan-out** (1 conta envia para N destinos novos) | ⚠️ BEH pega parcialmente (velocity) | ✅ `out_degree_24h` + `distinct_targets_7d` |
| **Triângulos suspeitos** (A→B→C→A, laundering) | ❌ Impossível sem grafo | ✅ `cycle_count`, `triangles` |
| **Comunidade de fraude** (cluster de contas interligadas) | ❌ Impossível | ✅ `community_fraud_rate`, `pagerank` |
| **Conta ponte** (recebe e repassa rapidamente) | ⚠️ burst_30m pega parcialmente | ✅ `avg_time_recv_to_send`, `betweenness` |

O artigo da **Thoughtworks (março/2026)** confirma exatamente isso para pagamentos real-time:

> *"Fraudsters increasingly operate in networks, while most financial services organizations still defend in rows and columns."*

E o paper **"BI-GBDT" (jan/2026)** propõe exatamente a abordagem que eu vou recomendar — features de interação comportamental baseadas em grafo, mas usando GBDT (como nosso LGBM) em vez de GNN:

> *"A Graph-Free Behavioral Interaction-Aware Gradient Boosting Framework for Fraud Detection in Large-Scale Payment Systems"*

---

## 2. Por que NÃO usar GNN em tempo real no seu contexto

| Restrição | Impacto |
|---|---|
| **Sem banco de grafos (Neo4j, Neptune, etc.)** | GNN precisa de adjacência em tempo real — inviável sem graph DB |
| **SLA de tempo real (< 100ms)** | GNN inference com GraphSAGE/GAT é 20-50ms **sozinha**, mais o overhead de construir o subgrafo on-the-fly |
| **Complexidade operacional** | GNN requer pipeline separado de treinamento, serving com DGL/PyG, versioning de grafos |
| **Dados disponíveis** | Seu dataset tem `origem` implícita (features por tx) mas **não tem ID do destinatário** — sem ele, não há aresta real |

O paper da AWS mostra que é possível GNN sem graph DB externo, mas exige **FAISS + Redis + DGL endpoint** — infraestrutura que você não tem.

---

## 3. A Abordagem Correta: Graph Feature Engineering (GFE) em Batch

A ideia é simples e poderosa:

```
┌─────────────────────────────────────────────────────────────┐
│                    BATCH (diário/horário)                     │
│                                                               │
│  1. Construir grafo de transações (NetworkX/DuckDB)           │
│  2. Calcular features de grafo por conta                      │
│  3. Salvar em Hive/Impala/HBase como tabela de features       │
│  4. Features ficam pré-computadas para lookup                 │
└──────────────────────────────┬────────────────────────────────┘
                               │
                               ▼ JOIN por conta
┌─────────────────────────────────────────────────────────────┐
│                    REAL-TIME (< 100ms)                        │
│                                                               │
│  1. Transação chega                                           │
│  2. Lookup das graph features pré-computadas (HBase/cache)    │
│  3. Features de grafo entram como colunas no LGBM             │
│  4. Pipeline v3.0.5 roda normalmente + features extras        │
│  5. Decisão: APROVAR / CONFIRMAR / BLOQUEAR                  │
└─────────────────────────────────────────────────────────────┘
```

**Zero impacto no SLA** — é só um lookup a mais de features pré-computadas, como já fazemos com qualquer feature store.

### Features de Grafo Candidatas

| Feature | Descrição | Janela | Onde captura valor |
|---|---|---|---|
| `in_degree_dest_7d` | Quantas contas distintas enviaram PIX para o destinatário nos últimos 7 dias | 7d | **Conta mula** (muitos remetentes) |
| `out_degree_orig_24h` | Quantos destinos distintos a conta de origem enviou nas últimas 24h | 24h | **Fan-out** (lavagem) |
| `dest_fraud_rate_30d` | Taxa de fraude histórica do destinatário | 30d | **Destino tóxico** |
| `dest_avg_recv_amt_7d` | Valor médio recebido pelo destinatário | 7d | Anomalia de valor no destino |
| `pagerank_orig` | PageRank da conta de origem no grafo de transações | 7d | **Centralidade** (hub de fraude) |
| `community_fraud_rate` | Taxa de fraude da comunidade (Louvain/Label Propagation) da conta | 7d | **Cluster de fraude** |
| `triangles_orig` | Número de triângulos envolvendo a conta de origem | 7d | **Ciclo de lavagem** |
| `betweenness_dest` | Betweenness centrality do destinatário | 7d | **Conta ponte** |
| `time_since_first_recv` | Tempo desde o primeiro PIX recebido pelo destinatário | all-time | **Conta nova recebendo** |
| `n_unique_senders_to_dest_1h` | Senders distintos para o destino na última hora | 1h | **Mula em atividade** |

---

## 4. Compatibilidade com Hive/Impala/HBase

| Storage | Papel | Viabilidade |
|---|---|---|
| **Hive** | Armazenar tabela de graph features calculadas em batch (Spark job diário) | ✅ Perfeito para batch |
| **Impala** | Query rápida para análise/debug das features de grafo | ✅ Bom para exploração |
| **HBase** | Lookup em tempo real por `account_id` → graph features | ✅ **Ideal para serving** (< 5ms p99) |

A arquitetura natural é:

```
Spark Job (batch, 1-4x/dia)
  ├─ Lê transações dos últimos 7 dias do Hive
  ├─ Constrói grafo com NetworkX/GraphFrames
  ├─ Calcula features (PageRank, degree, communities, etc.)
  ├─ Salva em Hive (histórico, auditoria)
  └─ Publica em HBase (serving real-time, key=account_id)

Pipeline Real-Time
  ├─ Transação chega
  ├─ Lookup HBase: graph_features[account_id] → ~5ms
  ├─ Merge com features transacionais
  ├─ LGBM v6 (retreinado com graph features) → predict
  └─ Pipeline v3.0.5 continua normalmente
```

---

## 5. Impacto Esperado nas Métricas

Sendo honesto sobre o que esperar:

### Onde grafos provavelmente **VÃO** ajudar

| Área | Mecanismo | Impacto estimado |
|---|---|---|
| **Redução de FP** | Conta com `dest_fraud_rate = 0%` + `in_degree normal` = evidência de legitimidade | **-10 a -30 FP** |
| **Novos padrões** | Detecção de contas mula (alto in_degree + conta nova) que hoje escapam | **Possível +1-5 TP** em cenários futuros |
| **Explicabilidade** | "Destino recebeu PIX de 47 contas distintas em 24h" — sinal claro para analista | Qualitativo |
| **Robustez** | Features de grafo são mais difíceis de manipular por fraudadores | Estratégico |

### Onde grafos provavelmente **NÃO** vão ajudar (neste dataset)

| Limitação | Motivo |
|---|---|
| **Os 3 FN irredutíveis** | São transações isoladas (tx_count=0, sem burst, sem padrão relacional). SE=0, BEH=0. Grafo não ajuda em transação sem contexto de rede. |
| **Recall (já 99,15%)** | Pouco espaço para subir. Os FN são irredutíveis por falta de sinal, não por falta de método. |
| **Overhead dos 2 FN invisíveis** | As 2 fraudes invisíveis a todos os componentes provavelmente continuam invisíveis — não há padrão relacional detectável. |

### Estimativa conservadora

| Métrica | v3.0.5 atual | Com GFE (estimativa) | Delta |
|---|---|---|---|
| TP | 352 | 352-353 | 0 a +1 |
| FP | 159 | 130-145 | **-15 a -30** |
| FN | 3 | 3 | 0 |
| Precision | 68,9% | **71-73%** | +2-4pp |
| F1 | 0,813 | **0,83-0,85** | +0,02-0,04 |

**O valor principal não é recall — é precision e robustez a novos padrões de fraude em produção.**

---

## 6. Restrições e Riscos

| Risco | Mitigação |
|---|---|
| **Sem ID de destinatário no dataset atual** | Precisa de `dest_account_id` ou `chave_pix_destino` para construir o grafo. **Se não tiver, GFE é inviável.** |
| **Latência do batch** | Features de grafo ficam defasadas (1-24h). Para mula que opera em minutos, features de 1h atrás podem ser insuficientes |
| **Cold start** | Conta nova sem histórico de grafo → features zeradas → LGBM precisa funcionar sem elas (já funciona hoje) |
| **Feature leakage** | Cuidado com `dest_fraud_rate` — se calculada no mesmo período do label, é leakage puro |
| **Custo de Spark** | Job diário com NetworkX/GraphFrames sobre milhões de tx pode ser pesado. Precisa de sizing |

---

## 7. Minha Recomendação

### Fase 1: Validação de Viabilidade (1-2 semanas)

Antes de qualquer código:

1. **Verificar se existe `dest_account_id`** (ou chave PIX de destino) nos dados brutos. Sem isso, não há grafo.
2. **Calcular features de grafo no dataset histórico** com NetworkX/pandas puro.
3. **Treinar LGBM v6** com as graph features adicionais e comparar com v5.1.
4. **Se F1 melhorar**: seguir para Fase 2.
5. **Se F1 não melhorar**: as fraudes neste dataset não têm padrão relacional — economize o esforço.

### Fase 2: Pipeline Batch (2-4 semanas)

Se Fase 1 validar:

1. Spark job para construir grafo e calcular features.
2. Salvar em Hive (histórico) + HBase (serving).
3. Retreinar LGBM com graph features.
4. Integrar lookup HBase no pipeline real-time.
5. Validar que SLA não foi afetado.

### O que eu **NÃO** recomendaria agora

- ❌ GNN (GraphSAGE, GAT) — complexidade desproporcional sem graph DB
- ❌ Neo4j/Neptune — custo e complexidade operacional sem necessidade imediata
- ❌ Graph features em real-time (on-the-fly) — SLA killer

---

## Resumo Final

| Pergunta | Resposta |
|---|---|
| Grafos podem agregar valor? | **Sim**, especialmente em precision e detecção de mulas/redes |
| Precisa de graph DB? | **Não** — Graph Feature Engineering com NetworkX + Hive/HBase resolve |
| Aumenta SLA? | **Não** — features pré-computadas em batch, lookup < 5ms |
| Vai melhorar recall? | **Provavelmente não** (já é 99,15%, FN são irredutíveis) |
| Vai reduzir FP? | **Provavelmente sim** (-15 a -30 FP estimado) |
| Prerequisito crítico? | **ID do destinatário nos dados** — sem ele, não há grafo |
| Roda em Hive/HBase? | **Sim**, é a arquitetura ideal |

A pergunta chave: **você tem acesso ao ID do destinatário (conta ou chave PIX) nos dados brutos?** Se sim, vale muito a pena explorar. Se não, o módulo de grafos fica bloqueado até ter esse dado.


PROMPT DO USUARIO:

"""
Verifique os dados brutos que tenho disponível (uma amostra das transações normais e a totalidade dos dados de fraude) e mais o dataset que o LGBM e IF usam pra treinar, e analise também a proposta de Feature Store que temos para usar em produção com o HBase e como ela poderia se integrar com esse modelo de grafos, e depois vocÊ me informa se a proposta é viável ou não pra começarmos a modelar os scripts e testar resultados.
"""

RESPOSTA DO LLM:



Adilio, boa. Deixa eu fazer uma análise completa dos dados antes de te dar o veredito.

---

## Análise de Viabilidade: Módulo de Grafos

### 1. Inventário dos Dados Disponíveis para Grafo

Olhando os dados brutos (`amostra_dados_pix.csv` e `dados_pix_fraudes_optimized.csv`), a resposta é **sim, temos o que precisamos**:

| Campo | Presente? | Exemplo | Papel no Grafo |
|---|---|---|---|
| `cd_cpf_pagador` | ✅ | `8169802172` | **Nó origem** |
| `cd_cpf_cnpj_recebedor` | ✅ | `9746310356` | **Nó destino** |
| `ds_chave_pix` | ✅ | `09746310356`, UUID, email | **Atributo da aresta** |
| `ds_tipo_chave` | ✅ | DOCUMENTO, CHAVE ALEATORIA, EMAIL | **Atributo da aresta** |
| `dt_pix` | ✅ | `2025-12-22 13:07:09` | **Timestamp da aresta** |
| `vl_pix` | ✅ | `3948.05` | **Peso da aresta** |
| `is_fraud` | ✅ (fraudes) | `1` | **Label para treino** |

**Isso é ouro.** Temos o grafo bipartido completo: `pagador → recebedor` com timestamp, valor, tipo de chave e label de fraude.

### 2. Padrões de Grafo Visíveis nos Dados de Fraude

Analisando `dados_pix_fraudes_optimized.csv`, os padrões são **gritantes**:

#### Padrão 1: Fan-out para mesmo recebedor (conta mula)

```
CPF 6865224191 (idade 79, F, MILLENIUM):
  → 9746310356  (R$ 3.948 + R$ 2.992)     ← 2 tx, mesmo dest
  → 10499815700 (R$ 48.999 + R$ 999 + R$ 6.099 + R$ 32.999 + R$ 2.999)  ← 5 tx, mesmo dest
  
  Total: 7 transações fraudulentas, 2 destinos distintos, R$ 99.036
```

O `cd_cpf_cnpj_recebedor = 10499815700` recebeu **5 transferências** da mesma vítima. Com grafo, veríamos que esse recebedor provavelmente recebe de **outras vítimas também**.

#### Padrão 2: Múltiplas vítimas → mesmo destino (rede mula)

```
Recebedor 16001077703 aparece recebendo de:
  - CPF 18556400159 (R$ 4.999 + R$ 650 + R$ 3.600 + R$ 2.400)
  - CPF 15395111115 (R$ 3.500 + R$ 3.390 + R$ 2.800)
  
  Duas vítimas distintas, mesmo destino, mesmo dia (2026-01-20)
```

```
Recebedor 20583432719 recebendo de:
  - CPF 15395111115 (R$ 5.000 + R$ 4.995 + R$ 3.000 + R$ 800)
  - CPF 18556400159 (R$ 3.600 + R$ 2.400)
  
  Novamente, duas vítimas, mesmo destino, mesmo dia
```

**Isso é uma rede de mulas clássica.** O LGBM não vê isso — ele vê cada tx isolada. Com `in_degree(16001077703) = 2 vítimas distintas em 24h`, um feature de grafo pegaria isso trivialmente.

#### Padrão 3: Recebedor com chave aleatória + múltiplos remetentes

```
Recebedor 38372267000182 (chave: a8eda32b-cca2-444d-9859-1bc7f67c780a):
  Nos dados NORMAIS: recebe de ~15 CPFs distintos (parece loja/CNPJ legítimo)
  Nos dados de FRAUDE:
    - CPF 88932168172 → R$ 1.489
    - CPF 23926643153 → R$ 5.089 + R$ 5.149
    - CPF 10213260115 → R$ 9.986 + R$ 6.736 + R$ 9.456 + ... (9 tx, R$ 59.600!)
    - CPF 18474837120 → R$ 672
```

Esse CNPJ recebe tanto tx normais quanto fraudes. O `in_degree` total é alto (normal para loja), mas a **concentração de valor por remetente novo** é anômala. Feature: `max_single_sender_amount_24h / avg_sender_amount_30d`.

#### Padrão 4: Burst de tx para destinos sequenciais (lavagem rápida)

```
CPF 82989818120 (idade 48, F, MILLENIUM, R$ 1.716 renda):
  18:05 → 11665794763 (R$ 780)
  18:06 → 11665794763 (R$ 950)
  18:09 → 6684091181  (R$ 920)     ← novo destino
  18:09 → 6684091181  (R$ 802)
  18:10 → 6684091181  (R$ 960)
  18:12 → 6684091181  (R$ 860)
  18:16 → 6684091181  (R$ 999)
  18:30 → 2310870129  (R$ 913)     ← novo destino
  18:31 → 2310870129  (R$ 911)
  18:31 → 2310870129  (R$ 820)
  18:33 → 2310870129  (R$ 819)
  18:34 → 2310870129  (R$ 990)
  18:34 → 2310870129  (R$ 958)
  18:36 → 11665794763 (R$ 850)     ← volta ao primeiro
  18:37 → 86680730104 (R$ 1.100)   ← 4º destino
  
  16 transações, 4 destinos, ~30 min, R$ 14.633
```

O burst_30m_flag e tx_count_prev_30m já pegam isso parcialmente. Mas o grafo adicionaria: **`distinct_targets_30m = 4`** e **`fan_out_velocity = 4 destinos novos em 30min`** — features que não existem hoje.

### 3. Features de Grafo Viáveis com os Dados

Com base nos dados reais, aqui estão as features que podemos calcular:

| Feature | Cálculo | Janela | Valor para fraude (estimado) | Onde salvar |
|---|---|---|---|---|
| **`dest_in_degree_24h`** | Nº de CPFs distintos que enviaram para o destino nas últimas 24h | 24h | Alto para mulas | HBase: nova CF `graph` |
| **`dest_in_degree_7d`** | Idem, 7 dias | 7d | Alto para mulas persistentes | HBase |
| **`dest_total_recv_amt_24h`** | Valor total recebido pelo destino em 24h | 24h | Alto para mulas | HBase |
| **`dest_fraud_rate_hist`** | % de fraudes históricas envolvendo esse destino como recebedor | All-time | > 0 para mulas conhecidas | HBase |
| **`dest_is_cnpj`** | Flag se destino é CNPJ (len > 11) | — | Padrão misto | Derivada |
| **`dest_avg_recv_interval_min`** | Intervalo médio entre recebimentos no destino | 24h | Muito baixo para mulas | HBase |
| **`orig_distinct_targets_24h`** | Nº de destinos distintos do pagador nas últimas 24h | 24h | Alto em fan-out | HBase |
| **`orig_distinct_targets_30m`** | Nº de destinos distintos em 30 min | 30m | ≥ 3 = muito suspeito | Cache memória |
| **`shared_dest_flag`** | Flag se outro CPF enviou para o mesmo destino na última 1h | 1h | Indicador de mula ativa | Cache/HBase |
| **`dest_new_sender_ratio_24h`** | % de remetentes novos (first_receiver) no destino em 24h | 24h | Alto para mulas novas | HBase |

### 4. Integração com Feature Store HBase

A proposta técnica do HBase já tem a tabela `fraud_detection:historico_recebedores` com a estrutura perfeita. A integração é natural:

#### Nova tabela: `fraud_detection:graph_features`

```bash
create 'fraud_detection:graph_features',
  {NAME => 'dest', VERSIONS => 3, TTL => 86400, COMPRESSION => 'SNAPPY'},
  {NAME => 'orig', VERSIONS => 3, TTL => 86400, COMPRESSION => 'SNAPPY'},
  {NAME => 'meta', VERSIONS => 1, TTL => 86400}
```

**Row key dupla:** Precisamos de lookup **tanto por pagador quanto por recebedor**:

| Tabela | Row Key | Uso |
|---|---|---|
| `graph_features_by_dest` | `LPAD(cpf_recebedor, 14, '0')` | Buscar perfil do destino quando tx chega |
| `graph_features_by_orig` | `LPAD(cpf_pagador, 14, '0')` | Buscar perfil de grafo do pagador |

#### Spark Job de Graph Features

```python
"""
03_materializa_graph_features.py
Calcula features de grafo e materializa no HBase.
Roda a cada hora, após o job de historico_recebedores.
"""

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from datetime import datetime, timedelta


def compute_graph_features(spark: SparkSession, dt_ref: datetime):
    """
    Calcula features de grafo a partir do extrato PIX.

    Args:
        spark: SparkSession com HiveSupport.
        dt_ref: Data/hora de referência para janelas temporais.
    """
    dt_24h = dt_ref - timedelta(hours=24)
    dt_7d = dt_ref - timedelta(days=7)
    dt_30d = dt_ref - timedelta(days=30)

    # =========================================================
    # ETAPA 1: Carregar transações PIX do período
    # =========================================================
    df_pix = (
        spark.sql(f"""
            SELECT
                cd_cpf_pagador,
                cd_cpf_cnpj_recebedor,
                ds_chave_pix,
                ds_tipo_chave,
                vl_pix,
                dt_pix,
                CASE WHEN LENGTH(cd_cpf_cnpj_recebedor) > 11
                     THEN 1 ELSE 0 END AS dest_is_cnpj
            FROM blk.tb_extrato_pix
            WHERE dt_pix >= '{dt_30d.strftime("%Y-%m-%d %H:%M:%S")}'
        """)
    )

    df_pix.cache()

    # =========================================================
    # ETAPA 2: Features do DESTINO (recebedor)
    # =========================================================

    # 2a. In-degree (senders distintos) por janela
    df_dest_24h = (
        df_pix
        .filter(F.col("dt_pix") >= F.lit(dt_24h))
        .groupBy("cd_cpf_cnpj_recebedor")
        .agg(
            F.countDistinct("cd_cpf_pagador")
                .alias("dest_in_degree_24h"),
            F.sum("vl_pix")
                .alias("dest_total_recv_amt_24h"),
            F.count("*")
                .alias("dest_tx_count_24h"),
            F.avg("vl_pix")
                .alias("dest_avg_recv_amt_24h"),
        )
    )

    df_dest_7d = (
        df_pix
        .filter(F.col("dt_pix") >= F.lit(dt_7d))
        .groupBy("cd_cpf_cnpj_recebedor")
        .agg(
            F.countDistinct("cd_cpf_pagador")
                .alias("dest_in_degree_7d"),
            F.sum("vl_pix")
                .alias("dest_total_recv_amt_7d"),
        )
    )

    # 2b. Intervalo médio entre recebimentos (24h)
    w_dest = Window.partitionBy("cd_cpf_cnpj_recebedor").orderBy("dt_pix")

    df_dest_intervals = (
        df_pix
        .filter(F.col("dt_pix") >= F.lit(dt_24h))
        .withColumn("prev_dt", F.lag("dt_pix").over(w_dest))
        .withColumn(
            "interval_min",
            (
                F.unix_timestamp("dt_pix")
                - F.unix_timestamp("prev_dt")
            ) / 60,
        )
        .filter(F.col("interval_min").isNotNull())
        .groupBy("cd_cpf_cnpj_recebedor")
        .agg(
            F.avg("interval_min")
                .alias("dest_avg_recv_interval_min"),
            F.min("interval_min")
                .alias("dest_min_recv_interval_min"),
        )
    )

    # 2c. % de remetentes novos (first_receiver) no destino
    # Primeiro, pegar remetentes históricos (> 7d atrás)
    df_hist_senders = (
        df_pix
        .filter(F.col("dt_pix") < F.lit(dt_7d))
        .select("cd_cpf_cnpj_recebedor", "cd_cpf_pagador")
        .distinct()
    )

    df_recent_senders = (
        df_pix
        .filter(F.col("dt_pix") >= F.lit(dt_24h))
        .select("cd_cpf_cnpj_recebedor", "cd_cpf_pagador")
        .distinct()
    )

    df_new_senders = (
        df_recent_senders
        .join(
            df_hist_senders,
            on=["cd_cpf_cnpj_recebedor", "cd_cpf_pagador"],
            how="left_anti",
        )
        .groupBy("cd_cpf_cnpj_recebedor")
        .agg(F.count("*").alias("dest_new_senders_24h"))
    )

    # 2d. Join todas as features de destino
    df_dest_features = (
        df_dest_24h
        .join(df_dest_7d, on="cd_cpf_cnpj_recebedor", how="left")
        .join(df_dest_intervals, on="cd_cpf_cnpj_recebedor", how="left")
        .join(df_new_senders, on="cd_cpf_cnpj_recebedor", how="left")
        .withColumn(
            "dest_new_sender_ratio_24h",
            F.coalesce(
                F.col("dest_new_senders_24h")
                / F.col("dest_in_degree_24h"),
                F.lit(0.0),
            ),
        )
        .fillna(0)
        .withColumnRenamed("cd_cpf_cnpj_recebedor", "cpf")
    )

    # =========================================================
    # ETAPA 3: Features da ORIGEM (pagador)
    # =========================================================
    df_orig_24h = (
        df_pix
        .filter(F.col("dt_pix") >= F.lit(dt_24h))
        .groupBy("cd_cpf_pagador")
        .agg(
            F.countDistinct("cd_cpf_cnpj_recebedor")
                .alias("orig_distinct_targets_24h"),
            F.sum("vl_pix")
                .alias("orig_total_sent_amt_24h"),
            F.count("*")
                .alias("orig_tx_count_24h"),
        )
    )

    df_orig_features = (
        df_orig_24h
        .withColumnRenamed("cd_cpf_pagador", "cpf")
    )

    # =========================================================
    # ETAPA 4: Materializar no HBase
    # =========================================================
    write_to_hbase(df_dest_features, "graph_features_by_dest", catalog_dest)
    write_to_hbase(df_orig_features, "graph_features_by_orig", catalog_orig)

    df_pix.unpersist()
```

#### Integração no pipeline real-time

```python
# No _enrich_transaction() do orquestrador:

def _enrich_with_graph_features(self, tx: dict) -> dict:
    """Adiciona graph features ao vetor de features."""
    cpf_pagador = tx["cd_cpf_pagador"]
    cpf_recebedor = tx["cd_cpf_cnpj_recebedor"]

    # Lookup por destino (~5ms)
    dest_features = self.feature_store.get_graph_by_dest(
        cpf_recebedor
    )

    # Lookup por origem (~5ms)
    orig_features = self.feature_store.get_graph_by_orig(
        cpf_pagador
    )

    tx.update(dest_features)
    tx.update(orig_features)

    return tx
```

### 5. Impacto no SLA

| Etapa | Antes | Depois | Delta |
|---|---|---|---|
| HBase GET (4 tabelas) | ~10-20ms | ~15-30ms (6 tabelas) | **+5-10ms** |
| Preprocessing | ~5ms | ~7ms (mais features) | +2ms |
| LGBM inference | ~20ms | ~22ms (mais features) | +2ms |
| **Total pipeline** | **~150ms** | **~165ms** | **+15ms** |
| **SLA BCB** | 10.000ms | 10.000ms | **60× margem** |

**Zero risco de SLA.** O overhead é de ~15ms — irrelevante.

### 6. Plano de Validação (Fase 1)

Antes de codar o Spark job, podemos validar **offline** com os dados que já temos:

```python
"""
Pseudocódigo do experimento de validação offline.
Roda no notebook, sem HBase, sem produção.
"""

# 1. Construir grafo com NetworkX a partir dos dados brutos
import networkx as nx
import pandas as pd

df_normal = pd.read_csv("amostra_dados_pix.csv")
df_fraud = pd.read_csv("dados_pix_fraudes_optimized.csv")

# Renomear colunas (remover prefixo da tabela)
# ...

df_all = pd.concat([df_normal, df_fraud])

# 2. Calcular features de grafo por destino
dest_features = (
    df_all
    .groupby("cd_cpf_cnpj_recebedor")
    .agg(
        dest_in_degree_7d=("cd_cpf_pagador", "nunique"),
        dest_total_recv_amt_7d=("vl_pix", "sum"),
        dest_tx_count_7d=("cd_pix", "count"),
    )
)

# 3. Calcular features de grafo por origem
orig_features = (
    df_all
    .groupby("cd_cpf_pagador")
    .agg(
        orig_distinct_targets_7d=(
            "cd_cpf_cnpj_recebedor", "nunique"
        ),
        orig_total_sent_7d=("vl_pix", "sum"),
    )
)

# 4. Merge com dataset de treino do LGBM
df_train = pd.read_csv("base_mvp_model_ready_leakage_free.csv")
# JOIN por customer_id ↔ cd_cpf_pagador
# JOIN por recebedor (precisa do campo no dataset de treino)

# 5. Retreinar LGBM com features extras
# 6. Comparar F1, Precision, Recall vs v3.0.5
```

### 7. Veredito

| Critério | Avaliação |
|---|---|
| **Dados disponíveis?** | ✅ `cd_cpf_pagador` + `cd_cpf_cnpj_recebedor` + timestamp + valor |
| **Padrões de fraude em rede visíveis?** | ✅ Mulas recebendo de múltiplas vítimas, fan-out, burst multi-destino |
| **Integrável com HBase?** | ✅ Nova tabela `graph_features`, +5-10ms no lookup |
| **Impacto no SLA?** | ✅ ~165ms total, 60× abaixo do limite do BCB |
| **Aumenta FN?** | ❌ Não — features aditivas, não substituem nada |
| **Pode reduzir FP?** | ✅ `dest_fraud_rate = 0%` + `dest_in_degree normal` = evidência de legitimidade |
| **Pode pegar novos padrões?** | ✅ Redes de mulas que hoje são invisíveis ao LGBM |
| **Requer graph DB (Neo4j)?** | ❌ Não — NetworkX no batch, HBase no serving |

---

## Proposta: Duas Fases

### Fase 1: Validação Offline (1-2 semanas)

1. **Script `04_graph_feature_engineering.py`** — calcula features de grafo offline com NetworkX/pandas
2. **JOIN com `base_mvp_model_ready_leakage_free.csv`** — precisa mapear `customer_id` ↔ `cd_cpf_pagador` e adicionar `cd_cpf_cnpj_recebedor` ao dataset
3. **Treinar LGBM v6** com graph features
4. **Comparar métricas** — se F1 subir e FN não aumentar, seguir para Fase 2

> ⚠️ **Bloqueio potencial:** O `base_mvp_model_ready_leakage_free.csv` (dataset de treino do LGBM) não contém `cd_cpf_cnpj_recebedor`. Preciso confirmar: **o `customer_id` do dataset de treino é mapeável ao `cd_cpf_pagador` dos dados brutos?** E temos como recuperar o recebedor de cada transação?

### Fase 2: Produção (2-4 semanas, se Fase 1 validar)

1. **Spark job `03_materializa_graph_features.py`** — roda horário
2. **Nova tabela HBase** — `graph_features_by_dest` + `graph_features_by_orig`
3. **Integrar no orquestrador** — lookup + merge com features existentes
4. **Retreinar LGBM** com features de grafo
5. **Simulação E2E** com pipeline completo

Adilio, a proposta é **viável e promissora**. Os dados de fraude mostram padrões de rede claros que o LGBM não captura hoje. O custo de implementação é baixo (batch com NetworkX/Spark, serving com HBase), e o overhead no SLA é desprezível.

A pergunta-chave pra destravar: **o `customer_id` na `amostra_base_mvp.csv` é o `cd_cpf_pagador`?** Olhando os dados, parece que sim (ambos são numéricos com ~11 dígitos), mas preciso da sua confirmação pra desenhar o JOIN correto.


PROMPT DO USUARIO:
"""
eu acho que sim, customer_id é o mesmo cd_cpf_pagador, mas você pode fazer um script rápido para verificar o match entre essas duas features pra confirmar com certeza e podermos prosseguir pra você desenhar o join perfeitamente.
"""

RESPOSTA DO LLM:



**100% de match em todos os testes.** Resultado perfeito, Adilio. 🎯

Resumo do que confirmamos:

| Teste | Resultado |
|---|---|
| `transaction_id == cd_pix` | **999/999 (100%)** |
| `customer_id == cd_cpf_pagador` | **999/999 (100%)** |
| `vl_pix` match cruzado | **999/999 (100%)** |
| `cd_cpf_cnpj_recebedor` disponível | **999/999 (100%)** |
| IDs MVP ⊂ IDs brutos | **994/994 (100%)** — zero ID órfão |

Os 174 IDs "só nos brutos" são as fraudes adicionais que não estão na amostra MVP (esperado — a base de fraude tem mais registros).

Agora temos o contrato de dados firmado:

- **JOIN key:** `transaction_id` ↔ `cd_pix` (chave primária, 1:1)
- **ID de cliente:** `customer_id` == `cd_cpf_pagador` (confirmado)
- **Recebedor:** `cd_cpf_cnpj_recebedor` disponível em 100% das transações
- **844 recebedores únicos** em 999 transações — densidade boa para o grafo

Vou desenhar o script de Graph Feature Engineering agora. Quer que eu:

1. **Comece pelo script offline** (`04_graph_feature_engineering.py`) que calcula as features de grafo, faz o JOIN com a base MVP e gera um novo dataset pronto pra treino?
2. Ou prefere que eu **primeiro especifique formalmente** (mini-SPEC) as features de grafo e critérios de validação antes de codar?

Minha recomendação: opção 1 direto — já temos a análise feita, os padrões validados e os dados confirmados. Posso incluir um header com a spec das features no próprio script. Menos cerimônia, mais resultado.



PROMPT DO USUARIO:

"""
Pergunta: esse script já será o do módulo GFE? Ele é um modelo que precisa de treino, ou você está se referindo a adaptar o dataset que treinou o LGBM para treiná-lo novamente? Esse script vai constar os 4 padrões que você viu no dataset de fraudes e os algoritmos de grafo que você indicou antes?
"""

RESPOSTA DO LLM:



Excelentes perguntas, Adilio. Vou esclarecer cada ponto:

## GFE não é um modelo — é um feature engineering pipeline

**Graph Feature Engineering (GFE) não treina nada.** Ele é um módulo de **transformação de dados** que:

1. **Constrói um grafo** bipartido `pagador → recebedor` a partir dos dados brutos
2. **Calcula métricas de grafo** (os algoritmos que mapeamos)
3. **Gera novas colunas** (features) para cada transação
4. **Enriquece a base MVP** via JOIN — produzindo um novo dataset

O fluxo é:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐
│ dados_brutos │ ──→ │  GFE Module  │ ──→ │ base_mvp_com_grafos  │
│  (PIX raw)   │     │ (sem treino) │     │  (dataset enriquecido)│
└──────────────┘     └──────────────┘     └──────────────────────┘
                      Puro cálculo!              │
                      networkx + math            ▼
                                          ┌──────────────┐
                                          │ LGBM retrain │
                                          │ (com + feats)│
                                          └──────────────┘
```

## O que vai ter no script

Sim — os **4 padrões de fraude** mapeados viram features calculadas com os **algoritmos de grafo** que indiquei:

| Padrão de Fraude | Feature(s) de Grafo | Algoritmo |
|---|---|---|
| **Fan-out** (1 pagador → N recebedores) | `sender_out_degree`, `sender_unique_receivers_24h` | Degree centrality |
| **Fan-in** (N pagadores → 1 recebedor) | `receiver_in_degree`, `receiver_unique_senders_24h` | Degree centrality |
| **Mula/ponte** (intermediário conectando clusters) | `sender_betweenness`, `receiver_betweenness` | Betweenness centrality |
| **Anel/ciclo** (A→B→C→A) | `sender_in_cycle`, `cycle_length_min` | Cycle detection (DFS) |

**Mais features derivadas:**
- `sender_pagerank` / `receiver_pagerank` — importância global no grafo
- `receiver_concentration` — % do valor total que vai pro top recebedor (Herfindahl)
- `is_new_edge` — pagador nunca transacionou com esse recebedor antes
- `community_id` — detecção de comunidades (Louvain) para clustering implícito
- `transaction_velocity_zscore` — velocidade anômala no subgrafo local

## Depois do GFE

Aí sim — **retreinamos o LGBM** com o dataset enriquecido. A expectativa é que features de grafo capturem padrões relacionais que features tabulares isoladas não conseguem (o LGBM vê cada linha independente, o grafo vê a rede).

## Resumo da pipeline

```
04_graph_feature_engineering.py    → calcula features de grafo, gera novo dataset
05_retrain_lgbm_with_graph.py      → retreina LGBM com features originais + grafo
06_evaluate_improvement.py         → compara métricas antes vs depois
```

Faz sentido? Parto pro `04_graph_feature_engineering.py` com esses 4 padrões + algoritmos?


PROMPT DO USUARIO:

"""
ah sim, então esse script que você vai fazer é só um script preparatório dos dados, ainda não é o script do módulo principal de GFE, não é mesmo?
"""

RESPOSTA DO LLM:



Exato, Adilio. Vou ser preciso aqui:

## O que eu ia fazer vs. o que precisa existir

| Artefato | O que é | Onde vive |
|---|---|---|
| **Script preparatório** (`04_graph_feature_engineering.py`) | Script one-shot em `backend/scripts/` que roda uma vez, gera o dataset enriquecido e salva CSV | `backend/scripts/` |
| **Módulo GFE** (o de verdade) | Módulo de produção em `backend/core/` ou `backend/modelos/` que recebe uma transação (ou batch) e calcula as features de grafo **on-the-fly** ou contra um grafo pré-computado | `backend/core/graph/` |

O script preparatório é um **protótipo offline** — serve pra:
1. Validar que as features de grafo **realmente melhoram** o LGBM
2. Definir **quais features** têm signal (nem todas vão ajudar)
3. Calibrar **janelas temporais** e thresholds

Se as features provarem valor no retreino, aí sim a gente **promove a lógica** pra um módulo de produção com:
- Grafo mantido em memória (ou banco de grafos)
- Atualização incremental a cada transação nova
- API que a `api.py` chama antes de passar pro modelo

## A sequência correta

```
Script preparatório (exploração)     →  validou?  →  Módulo GFE (produção)
       ↓                                                    ↓
  Dataset enriquecido                              Cálculo real-time/batch
       ↓                                                    ↓
  Retrain LGBM (offline)                           Serve na API com LGBM
       ↓
  Features de grafo ajudaram?
  SIM → promove pra módulo
  NÃO → pivotar abordagem
```

**Ou seja: sim, é preparatório.** Primeiro a gente prova que funciona, depois produtiza. Cargo-cultar um módulo de produção sem saber se as features têm signal seria engenharia prematura.

Parto pro script preparatório então?