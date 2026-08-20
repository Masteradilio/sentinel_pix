# Módulo de Graph Engineering Investigativo (Baseline R5B22)

## Ficha Técnica do Módulo

```
Nome:            Graph Investigation Engine (GFE Investigativo)
Versão:          v1.0 (Baseline Operacional R5B22)
Tipo:            Módulo de investigação pós-decisão (Near Real-Time / Batch)
Foco:            Comunidades, contas laranja (mulas), fan-in/fan-out, bridge accounts
SLA:             Assíncrono (não impacta SLA da transação)
Módulo:          core/graph_engineering.py
Status:          Opt-in por configuração
```

---

## 1. Propósito e Justificativa

### 1.1 O papel do grafo no baseline atual
O pipeline R5B22 (Machine Learning Distilado + Engenharia Social + Behavioral Analytics) avalia a transação *no momento do evento*. Ele toma decisões altamente precisas para os rótulos de tempo real, operando num cenário "cego para topologias complexas de rede".

O **Graph Investigation Engine** atua de forma complementar e **pós-decisão**. Seu objetivo não é decidir se uma transação será bloqueada ou aprovada em milissegundos, mas sim:
1. Analisar transações que já sofreram intervenção (`CONFIRMAR` ou `BLOQUEAR`).
2. Identificar clusters de fraudadores e contas laranja (mulas) relacionando CPF de origem, destino e padrões topológicos.
3. Gerar insumos ricos e exportáveis (CSV investigativo incremental) para os analistas de fraude e inteligência de ameaças financeiras.

### 1.2 Por que operar de forma investigativa?
Avaliar subgrafos com dezenas de arestas e profundidades N *on-the-fly* dentro dos 100ms de limite da API é custoso e pode induzir timeouts severos em produção, exigindo infraestruturas massivas de Grafos (ex. Neo4J) em memória. Operando em modo **Assíncrono Pós-Decisão / Relatório**, não dependemos de um banco de grafos caríssimo e conseguimos entregar 99% do valor estratégico investigando fraudes latentes.

---

## 2. Padrões Topológicos Capturados

| Padrão | Descrição Topológica | Indicadores / Features Geradas |
|---|---|---|
| **Contas Mula (Fan-in)** | Mesma conta recebe de inúmeras vítimas diferentes em uma janela curta. | `suspected_mule_score`, `graph_in_degree_receiver_24h`, `graph_unique_payers_to_receiver_24h` |
| **Lavagem Rápida (Fan-out)** | Uma conta envia valores fracionados para múltiplos destinos novos após receber fundos. | `fanout_score`, `graph_out_degree_payer_24h`, `graph_unique_receivers_from_payer_24h` |
| **Conta Ponte (Bridge)** | Conta atua como repasse instantâneo na mesma janela temporal. Recebe A e joga para B. | `bridge_account_score`, tempo entre in/out |
| **Reciprocidade e Idade** | Análise de tempo de vínculo ou ausência total de interações bidirecionais. | `graph_is_new_receiver`, `graph_receiver_first_seen_age_hours`, `graph_reciprocity_flag` |

---

## 3. Estrutura de Integração (Opt-In)

O fluxo da API no R5B22 é preservado de ponta a ponta. A chamada para a Graph Investigation ocorre num fire-and-forget:

```text
API retorna decisao (APROVAR/CONFIRMAR/BLOQUEAR)
  │
  ├─ Se decisão em {CONFIRMAR, BLOQUEAR}:
  │    └─ Verifica ENV: GRAPH_INVESTIGATION_ENABLED=true ?
  │         └─ SIM: Acionar GraphInvestigationEngine.process_transaction() de forma isolada
  │         └─ Engine atualiza memória local (janelas)
  │         └─ Calcula features do subgrafo
  │         └─ Grava linha em CSV incremental append-only
  │
  └─ Se decisão for APROVAR:
       └─ Nenhuma ação por padrão (reduz custo de processamento p/ normais)
```

---

## 4. Pré-requisitos e Saída (Relatório CSV)

### 4.1 Pré-requisitos de Dados Mínimos
Para construir o nó do grafo o módulo necessita, em cada transação interpelada:
- **Aresta/Transação:** `transaction_id`, `event_datetime`, `vl_pix`
- **Nó Origem:** `customer_id` (Pagador)
- **Nó Destino:** `counterparty_id` (Recebedor)
- **Atributos:** `decisao`, `score_final`

> *Nota de Resiliência: Se um campo estiver ausente, o módulo continua operando, inserindo métricas `null` e logando um warning, jamais estourando uma Exception para o orquestrador transacional.*

### 4.2 Saída do Relatório Investigativo
Os dados são expostos em `resultados/investigacao/graph_investigation_report.csv` em formato append-only.
Cada intervenção de bloqueio gera uma linha contendo as métricas de tempo real somadas às inferências do grafo:

*Exemplo de colunas no relatório:*
`transaction_id, payer_id, receiver_id, decisao, rule_applied, graph_in_degree_receiver_24h, graph_suspected_mule_score, graph_bridge_account_score...`
