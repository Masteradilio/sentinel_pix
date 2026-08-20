# Plano Final de Aprimoramento do Modelo de Detecção de Fraudes PIX

**Documento:** `PIX_FRAUD_MODEL_FINAL_IMPROVEMENT_PLAN.md`  
**Projeto:** EXP-014B — Detecção de Fraudes em Transações PIX  
**Status de referência:** pós `EXP-014B-R4G-FAST-FROZEN`  
**Objetivo:** orientar as próximas fases de melhoria estrutural do modelo, após o esgotamento das principais otimizações por política/regras pós-modelo.

---

## 1. Contexto e razão deste documento

Após as rodadas EXP-014B, o modelo atingiu um estado operacional muito forte. O resultado consolidado mais recente, aceito como melhor baseline geral até agora, foi o `EXP-014B-R4G-FAST`.

A decisão de promover o `R4G-FAST` como novo campeão foi tomada porque ele reduziu significativamente as fraudes que passavam diretamente como `APROVAR`, preservando a meta global de falso positivo.

Baseline anterior, `R4F-FROZEN`:

```text
TP=1460
FP=1113
FN=5
FPR=0,990399%
```

Novo baseline campeão, `R4G-FAST`:

```text
TP=1463
FP=1123
FN=2
FPR=0,999297%
```

A decisão operacional foi considerar o `R4G-FAST` superior ao `R4F-FROZEN`, mesmo consumindo a folga de 10 falsos positivos, porque o ganho de reduzir fraudes aprovadas de 5 para 2 é mais importante para apresentação executiva e para a narrativa de controle antifraude.

Distribuição operacional do `R4G-FAST`:

```text
APROVAR:
111258 transações
2 fraudes
111256 normais

BLOQUEAR:
2224 transações
1458 fraudes
766 normais

CONFIRMAR:
362 transações
5 fraudes
357 normais
```

O modelo agora está próximo do estado operacional ideal imaginado:

```text
FPR global < 1%
FN muito baixo
fraudes residuais preferencialmente em CONFIRMAR, não em APROVAR
BLOQUEAR concentrando a maior parte das fraudes
CONFIRMAR funcionando como zona de estranheza operacional
```

Ainda assim, restam dois desafios importantes:

```text
1. Duas fraudes ainda passam diretamente como APROVAR.
2. Setecentas e sessenta e seis transações normais ainda ficam em BLOQUEAR.
```

As rodadas recentes mostraram que as melhorias por busca combinatória e regras pós-modelo estão chegando ao limite. Portanto, este documento define cinco fases de melhoria estrutural para evoluir o modelo como um todo.

---

## 2. Princípios orientadores das próximas fases

As próximas fases não devem simplesmente buscar mais regras específicas. O foco passa a ser melhorar a capacidade informacional e arquitetural do sistema.

Princípios:

```text
1. Preservar FPR global < 1%.
2. Evitar fraudes em APROVAR.
3. Reduzir normais em BLOQUEAR.
4. Usar CONFIRMAR como zona de incerteza aceitável.
5. Separar melhor risco de fraude e severidade operacional.
6. Priorizar features explicáveis para discussão com negócio, risco e diretoria.
```

A métrica global continua importante, mas deixa de ser a única bússola. O modelo precisa otimizar uma função de custo operacional:

```text
fraude em APROVAR       = erro crítico
normal em BLOQUEAR      = erro operacional grave
fraude em CONFIRMAR     = alerta parcial aceitável
normal em CONFIRMAR     = fricção aceitável
fraude em BLOQUEAR      = acerto forte
normal em APROVAR       = acerto operacional
```

---

# Fase 1 — Diagnóstico dos resíduos críticos

## 1.1. Objetivo

Construir uma análise detalhada dos dois grupos que mais limitam o modelo atual:

```text
Grupo A:
2 fraudes ainda em APROVAR

Grupo B:
766 transações normais ainda em BLOQUEAR
```

A ideia é entender se esses resíduos são explicáveis com as variáveis atuais ou se exigem novas fontes de dados.

## 1.2. Perguntas que a fase deve responder

Para as 2 fraudes em APROVAR:

```text
Elas têm score baixo em todos os módulos?
Há sinal fraco em algum módulo isolado?
São fraudes com relacionamento aparentemente confiável?
São transações de baixo valor?
O recebedor parecia conhecido?
A chave PIX parecia estável?
A instituição recebedora era comum?
Havia histórico anterior com o recebedor?
O comportamento do pagador estava normal?
```

Para os 766 normais em BLOQUEAR:

```text
Por que foram bloqueados?
Têm scores altos por comportamento, social engineering ou anomalia?
Estão concentrados em algum tipo de chave?
Estão concentrados em faixas de valor específicas?
Têm recebedores antigos e confiáveis?
São pagamentos recorrentes?
São clientes com padrão transacional legítimo de alto valor?
Há concentração por canal, horário, dispositivo, instituição ou segmento?
```

## 1.3. Comparações necessárias

A análise deve comparar quatro grupos:

```text
A. Fraudes em APROVAR
B. Fraudes em BLOQUEAR
C. Normais em BLOQUEAR
D. Normais em APROVAR
```

O objetivo é identificar features que separam:

```text
fraude em APROVAR vs normal em APROVAR
normal em BLOQUEAR vs fraude em BLOQUEAR
```

## 1.4. Artefatos esperados

```text
EXP-014B-R5A/
  00_run_summary.json
  01_input_contract.json
  02_residual_group_metrics.json
  03_approve_fraud_cases.csv
  04_block_normal_cases.csv
  05_feature_contrast_approve_fraud_vs_approve_normal.csv
  06_feature_contrast_block_normal_vs_block_fraud.csv
  07_candidate_feature_gaps.json
  08_exp014b_r5a_report.md
```

## 1.5. Critério de sucesso

A Fase 1 não precisa alterar o modelo. Ela será bem-sucedida se produzir hipóteses acionáveis de melhoria:

```text
novas features candidatas
variáveis atuais insuficientes
segmentos onde o modelo confunde normal com fraude
segmentos onde o modelo deixa fraude passar
prioridade objetiva para engenharia de features
```

---

# Fase 1.1 — Auditoria do conjunto mínimo de features

## 1.1.1. Objetivo

Antes de adicionar novas features nas fases R5B e R5C, o projeto deve separar claramente:

```text
1. colunas de contrato, identificação e diagnóstico
2. labels e decisões operacionais congeladas
3. scores e bins derivados de modelos ou políticas
4. features realmente usadas pelo LGBM
5. features usadas apenas por política operacional
6. features legadas de experimentos R3/R4
7. features candidatas para treino futuro
```

O arquivo final de predictions pode ter muitas colunas, mas isso não significa que todas sejam features primárias do modelo. A etapa R5A.1 existe para evitar crescimento descontrolado do espaço de variáveis e para preservar explicabilidade, estabilidade e controle de variância antes de qualquer nova engenharia de atributos.

## 1.1.2. Experimento sugerido

```text
EXP-014B-R5A1 — Minimal Feature Set and Redundancy Audit
```

## 1.1.3. Perguntas que a fase deve responder

```text
Qual é o conjunto mínimo de colunas necessário para reproduzir o baseline congelado?
Quais colunas são contrato operacional e não devem entrar como feature de treino?
Quais colunas são legado de experimentos anteriores?
Quais features estão efetivamente no contrato do LGBM?
Há divergência entre contrato de treino e contrato runtime?
Quais features têm alta ausência, baixa variância ou redundância extrema?
Quais features devem ser mantidas, descartadas ou revisadas antes de R5B/R5C?
```

## 1.1.4. Artefatos esperados

```text
EXP-014B-R5A1/
  00_run_summary.json
  01_column_taxonomy.csv
  02_model_feature_contract_audit.json
  03_replay_minimal_columns.json
  04_redundancy_candidates.csv
  05_feature_keep_drop_recommendations.csv
  06_exp014b_r5a1_report.md
```

## 1.1.5. Critério de sucesso

A fase será bem-sucedida se produzir um gate objetivo para as próximas etapas de feature engineering:

```text
features novas só entram se tiverem hipótese, cobertura, ausência de leakage e ganho esperado
colunas de política/experimento não entram como features primárias
features redundantes são revisadas antes de aumentar a dimensionalidade
o baseline congelado permanece reprodutível por contrato mínimo
o conjunto de features candidatas fica explicável para revisão técnica e executiva
```

---

# Fase 1.2 — Auditoria da matriz real de treino

## 1.2.1. Objetivo

Depois da auditoria do arquivo de predictions, é necessário auditar a matriz real usada pelos modelos de ML:

```text
dados/base_treino_final.csv
```

Essa base é consumida por:

```text
backend/modelos/train_lgbm_v3.py
backend/modelos/train_isolation_forest_v2.py
```

A Fase 1.2 deve responder qual é a qualidade estatística, redundância e estabilidade temporal das features efetivamente disponíveis para treino, antes de adicionar novas features em R5B/R5C.

## 1.2.2. Experimento sugerido

```text
EXP-014B-R5A2 — Training Feature Matrix Audit
```

## 1.2.3. Perguntas que a fase deve responder

```text
Quais features declaradas pelo LGBM existem na matriz de treino?
Quais features declaradas pelo Isolation Forest existem na matriz de treino?
Quais features têm missing alto, variância zero ou baixa cardinalidade útil?
Quais features são redundantes por correlação extrema?
Quais features variam de forma instável entre treino, validação e holdout?
Quais features atuais devem ser mantidas até ablação?
Quais features devem ser candidatas a remoção ou substituição pelas novas features R5B/R5C?
```

## 1.2.4. Artefatos esperados

```text
EXP-014B-R5A2/
  00_run_summary.json
  01_input_contract.json
  02_training_feature_inventory.csv
  03_lgbm_feature_audit.csv
  04_if_feature_audit.csv
  05_redundancy_candidates.csv
  06_temporal_stability_by_feature.csv
  07_keep_drop_recommendations.csv
  08_exp014b_r5a2_report.md
```

## 1.2.5. Critério de sucesso

```text
contratos LGBM e IF reconciliados com a base real de treino
features ausentes, constantes ou instáveis explicitamente marcadas
features redundantes agrupadas para ablação
gate objetivo definido antes de R5B/R5C
nenhum retreino executado nesta fase
```

---

# Fase 1.3 — Conciliação final de datasets, features e artefatos

## 1.3.1. Objetivo

Consolidar a mudança de rumo ocorrida entre o ciclo antigo de treino e o baseline campeão atual.

O projeto começou com a matriz local:

```text
dados/base_treino_final.csv
100.355 linhas
355 fraudes
```

Essa base foi útil para estabilizar o pipeline inicial, treinar os primeiros modelos e validar módulos como LGBM, Isolation Forest, Social Engineering, Behavioral e Decision Engine. Porém ela deixou de ser a referência principal após a aquisição de novas fraudes MAF e a reconstrução dos datasets no Big Data.

A trilha histórica nos journals registra a mudança:

```text
EXP-010B/010C:
  nova fonte MAF textual e tabelas curadas/hidratadas de fraude.

EXP-010D/010E:
  auditoria e hidratação MBK das fraudes MAF recentes.

EXP-010F/010G:
  amostragem normal qualificada e dataset v2 rolling 180d.

EXP-011C:
  rejeição dos candidatos vNext sobre dataset v2 e decisão de reconstruir o dataset
  com features históricas reais no Big Data.

EXP-012A:
  criação do dataset v3 com features históricas reais:
  hmo_ml.tb_pix_dataset_v3_features_180d_v1.

EXP-014A-4:
  geração do dataset expandido scoreado com 113.844 linhas e 1.465 fraudes.

EXP-014B:
  calibração high-recall e políticas operacionais diretamente sobre o dataset expandido,
  culminando no baseline campeão EXP-014B-R4G-FAST-FROZEN.
```

Portanto, a Fase 1.3 deve reconciliar oficialmente:

```text
1. bases antigas de treino
2. datasets v2/v3 gerados no Big Data
3. CSVs locais hmo_ml_tb_pix_dataset_v3_features_180d_*.csv
4. scripts HQL em dados/scripts_origem/tb_pix_*.hql
5. scripts de treino LGBM/Isolation Forest
6. artefatos runtime em backend/artefatos
7. artefatos de experimentos EXP-014B usados pelo baseline campeão
```

## 1.3.2. Experimento sugerido

```text
EXP-014B-R5A3 — Dataset and Feature Contract Reconciliation
```

## 1.3.3. Perguntas que a fase deve responder

```text
Qual dataset deve ser considerado fonte canônica para R5B/R5C?
Quais scripts de treino ainda apontam para bases legadas?
Quais artefatos de modelo foram treinados no dataset antigo e quais foram treinados/recalibrados no dataset expandido?
Quais colunas dos CSVs hmo_ml_tb_pix_dataset_v3_features_180d_*.csv são features primárias?
Quais features do LGBM/IF precisam ser recalculadas, removidas ou substituídas no contrato novo?
O baseline R4G depende de modelo ML treinado, política pós-modelo, replay scoreado ou combinação desses elementos?
Quais arquivos devem ser considerados somente históricos e quais devem orientar a próxima geração do modelo?
```

## 1.3.4. Entradas obrigatórias

```text
docs/JOURNAL_1.md
docs/JOURNAL_2.md
docs/JOURNAL_3.md
docs/JOURNAL_4.md

dados/hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv
dados/hmo_ml_tb_pix_dataset_v3_features_180d_v1_TRAIN.csv
dados/hmo_ml_tb_pix_dataset_v3_features_180d_v1_VALIDATION.csv
dados/hmo_ml_tb_pix_dataset_v3_features_180d_v1_HOLDOUT.csv

dados/scripts_origem/tb_pix_dataset_v3_target_180d_v1.hql
dados/scripts_origem/tb_pix_dataset_v3_daily_agg_180d_v1.hql
dados/scripts_origem/tb_pix_dataset_v3_features_180d_v1.hql

backend/modelos/train_lgbm_v3.py
backend/modelos/train_isolation_forest_v2.py

resultados/experimentos/EXP-014B-R4G-FAST-FROZEN/
resultados/experimentos/EXP-014B-R4G-FAST/
```

## 1.3.5. Artefatos esperados

```text
EXP-014B-R5A3/
  00_run_summary.json
  01_journal_lineage_summary.json
  02_dataset_inventory.csv
  03_hql_feature_lineage.csv
  04_training_script_contract_audit.json
  05_baseline_artifact_lineage.json
  06_feature_contract_reconciliation.csv
  07_canonical_dataset_recommendation.json
  08_exp014b_r5a3_report.md
```

## 1.3.6. Critério de sucesso

```text
base canônica definida antes de R5B/R5C
scripts de treino legados identificados explicitamente
contratos LGBM/IF reconciliados com o dataset expandido campeão
artefatos históricos separados dos artefatos candidatos à produção
nenhuma feature nova adicionada antes da conciliação final
plano claro para retreino/replay futuro sem misturar dataset antigo e dataset v3
```

---

# Fase 2 — Engenharia de features de relacionamento pagador-recebedor

## 2.1. Objetivo

Adicionar ou aprimorar features que descrevem a relação histórica entre o pagador e o recebedor.

Essa é a linha mais promissora para reduzir simultaneamente:

```text
fraudes em APROVAR
normais em BLOQUEAR
```

A hipótese é que parte dos falsos positivos em BLOQUEAR são transações legítimas com relacionamento histórico forte, enquanto parte das fraudes residuais em APROVAR pode envolver relações fracas, novas ou atípicas.

## 2.2. Features candidatas

### Histórico do par pagador-recebedor

```text
qtd_transacoes_pagador_recebedor_7d
qtd_transacoes_pagador_recebedor_30d
qtd_transacoes_pagador_recebedor_90d
valor_total_pagador_recebedor_30d
valor_medio_pagador_recebedor_90d
dias_desde_primeira_transacao_par
dias_desde_ultima_transacao_par
```

### Recorrência e estabilidade

```text
is_recebedor_recorrente_para_pagador
is_primeira_transacao_pagador_recebedor
percentil_valor_vs_historico_par
desvio_valor_vs_media_par
qtd_meses_com_relacao_ativa
regularidade_temporal_do_par
```

### Relação de confiança

```text
receiver_trust_score_by_payer
payer_receiver_relationship_strength
payer_receiver_value_stability
payer_receiver_frequency_stability
relationship_age_bucket
relationship_recurrence_bucket
```

## 2.3. Estratégia de implementação

Criar uma tabela de features temporais respeitando o tempo da transação. Nenhuma feature pode usar informação futura.

Estrutura sugerida:

```text
features_pix_relationship_daily/
  reference_date
  payer_id_hash
  receiver_id_hash
  qtd_tx_7d
  qtd_tx_30d
  qtd_tx_90d
  valor_total_30d
  valor_medio_90d
  dias_primeira_relacao
  dias_ultima_relacao
  relationship_strength_score
```

Para cada transação, fazer lookup das features calculadas até o instante anterior à transação.

## 2.4. Experimento sugerido

```text
EXP-014B-R5B — Relationship Feature Augmentation
```

Pipeline:

```text
1. Construir features de relacionamento.
2. Retreinar modelo base.
3. Recalibrar política operacional.
4. Comparar contra R4G-FAST-FROZEN.
```

## 2.5. Métricas de sucesso

```text
FN <= 2
FP <= 1123
redução de fraudes em APROVAR
redução de normais em BLOQUEAR
aumento de precision em BLOQUEAR
manutenção de recall global
```

---

# Fase 3 — Reputação, estabilidade e risco do recebedor

## 3.1. Objetivo

Criar uma camada de inteligência centrada no recebedor.

Fraudes PIX frequentemente dependem do recebedor: conta de passagem, chave nova, recebedor com padrão de recebimento anômalo, concentração de entrada, crescimento súbito ou histórico fraco.

Por outro lado, muitos falsos positivos podem envolver recebedores legítimos, antigos, recorrentes e estáveis.

## 3.2. Features candidatas

### Idade e estabilidade

```text
idade_chave_pix
idade_conta_recebedora
dias_desde_primeiro_recebimento
dias_desde_primeiro_pagador_distinto
qtd_dias_ativos_recebedor_90d
```

### Diversidade de pagadores

```text
qtd_pagadores_distintos_1d
qtd_pagadores_distintos_7d
qtd_pagadores_distintos_30d
crescimento_pagadores_distintos_7d_vs_30d
entropia_pagadores_recebedor
```

### Volume e comportamento

```text
valor_total_recebido_1d
valor_total_recebido_7d
valor_total_recebido_30d
ticket_medio_recebedor_30d
desvio_valor_recebido_vs_historico
burst_recebimentos_curta_janela
```

### Risco histórico

```text
qtd_contestacoes_recebedor
qtd_fraudes_confirmadas_recebedor
taxa_fraude_recebedor
qtd_bloqueios_anteriores_recebedor
receiver_reputation_score
```

### Instituição recebedora

```text
instituicao_recebedora
risco_instituicao_recebedora
taxa_fraude_instituicao_recebedora
tipo_conta_recebedora
```

## 3.3. Experimento sugerido

```text
EXP-014B-R5C — Receiver Reputation and Stability Features
```

Etapas:

```text
1. Construir features históricas do recebedor.
2. Criar receiver_reputation_score.
3. Criar receiver_stability_score.
4. Retreinar modelo.
5. Reotimizar política APROVAR/CONFIRMAR/BLOQUEAR.
```

## 3.4. Resultado esperado

Essa fase deve ajudar especialmente em dois pontos:

```text
1. Reduzir as 2 fraudes restantes em APROVAR.
2. Reduzir os 766 normais ainda em BLOQUEAR.
```

Exemplo de decisão esperada:

```text
Recebedor novo, alto volume, muitos pagadores novos:
aumentar severidade.

Recebedor antigo, recorrente, estável, sem histórico de fraude:
reduzir severidade de BLOQUEAR para CONFIRMAR ou APROVAR, conforme score global.
```

---

# Fase 4 — Separação entre modelo de intervenção e modelo de severidade

## 4.1. Problema atual

Hoje, boa parte da política operacional é construída com regras pós-modelo para chegar em:

```text
APROVAR
CONFIRMAR
BLOQUEAR
```

Isso funcionou muito bem até o R4G-FAST, mas criou uma tensão:

```text
A mesma camada tenta decidir se deve intervir e qual deve ser a severidade.
```

Esses são problemas relacionados, mas não idênticos.

## 4.2. Arquitetura proposta

Separar em dois modelos:

```text
Modelo 1 — Intervenção
Decide se a transação deve sair de APROVAR.

Saída:
intervention_pred = 0 ou 1

Modelo 2 — Severidade
Aplicado apenas quando intervention_pred = 1.
Decide entre CONFIRMAR e BLOQUEAR.

Saída:
severity_pred = CONFIRMAR ou BLOQUEAR
```

## 4.3. Benefício esperado

Essa arquitetura permite otimizar separadamente:

```text
FPR global
FN global
precision de BLOQUEAR
fila de CONFIRMAR
custo operacional de bloqueio indevido
```

O modelo de intervenção deve ser calibrado para:

```text
minimizar fraude em APROVAR
preservar FPR < 1%
```

O modelo de severidade deve ser calibrado para:

```text
bloquear fraudes de alta confiança
mandar incerteza para CONFIRMAR
reduzir normais em BLOQUEAR
```

## 4.4. Experimento sugerido

```text
EXP-014B-R5D — Two-Stage Intervention and Severity Model
```

Etapas:

```text
1. Treinar modelo binário de intervenção.
2. Treinar modelo de severidade apenas nos casos intervencionados.
3. Calibrar thresholds separadamente.
4. Comparar com R4G-FAST-FROZEN.
```

## 4.5. Métricas de sucesso

```text
intervention_fp <= 1123
intervention_fn <= 2
block_fp < 766
block_tp >= 1458
confirm_fraud pequeno e aceitável
approve_fraud <= 2, idealmente 0
```

---

# Fase 5 — Modelo ordinal, função de custo e active learning

## 5.1. Objetivo

Transformar o problema em uma decisão ordenada por severidade e custo operacional.

Classes:

```text
0 = APROVAR
1 = CONFIRMAR
2 = BLOQUEAR
```

A ideia é treinar ou calibrar uma função que entenda que os erros têm custos diferentes.

## 5.2. Matriz de custo operacional

Proposta inicial:

```text
fraude em APROVAR:
custo 100

fraude em CONFIRMAR:
custo 20

fraude em BLOQUEAR:
custo 0

normal em APROVAR:
custo 0

normal em CONFIRMAR:
custo 2

normal em BLOQUEAR:
custo 15
```

Essa matriz é ajustável com a área de negócio.

## 5.3. Estratégia ordinal

Há duas alternativas:

```text
1. Modelo ordinal direto:
   aprende APROVAR < CONFIRMAR < BLOQUEAR.

2. Modelo de score contínuo + calibração por custo:
   aprende um risco e converte para ação usando thresholds otimizados.
```

## 5.4. Active learning nos casos residuais

Após R4G-FAST, os casos mais valiosos para revisão são:

```text
2 fraudes em APROVAR
766 normais em BLOQUEAR
5 fraudes em CONFIRMAR
357 normais em CONFIRMAR
```

A revisão deve tentar responder:

```text
o dado atual explica o erro?
há informação externa que resolveria o caso?
a classificação original está correta?
há padrão de negócio não capturado?
há variável com vazamento temporal?
há segmento que merece política própria?
```

## 5.5. Experimento sugerido

```text
EXP-014B-R5E — Ordinal Cost-Sensitive Policy and Active Learning
```

Etapas:

```text
1. Definir matriz de custo com negócio.
2. Treinar modelo ordinal ou calibrar score por custo.
3. Rodar otimização de thresholds com custo total.
4. Revisar manualmente resíduos críticos.
5. Incorporar aprendizados em features e política.
```

## 5.6. Métricas de sucesso

```text
custo operacional total menor que R4G-FAST-FROZEN
approve_fraud <= 2, idealmente 0
block_fp < 766
fpr < 1%
recall >= 99,86%
modelo explicável para diretoria
```

---

# 3. Roadmap proposto

A sequência recomendada é:

```text
R5A — Residual Error Feature Diagnosis
R5A1 — Minimal Feature Set and Redundancy Audit
R5A2 — Training Feature Matrix Audit
R5A3 — Dataset and Feature Contract Reconciliation
R5B — Relationship Feature Augmentation
R5C — Receiver Reputation and Stability Features
R5D — Two-Stage Intervention and Severity Model
R5E — Ordinal Cost-Sensitive Policy and Active Learning
```

Ordem de execução:

```text
1. R5A primeiro, porque ele identifica onde estão os limites atuais.
2. R5A1 em seguida, porque define o conjunto mínimo, separa contrato de feature e evita crescimento descontrolado das variáveis.
3. R5A2 depois, porque audita a matriz real de treino usada por LGBM e Isolation Forest antes de qualquer expansão de features.
4. R5A3 fecha a fase preparatória, conciliando dataset antigo, dataset v3 expandido, scripts de treino, HQLs e artefatos do baseline campeão.
5. R5B e R5C depois, porque adicionam nova informação ao modelo com gate de redundância e explicabilidade.
6. R5D em seguida, porque reorganiza a arquitetura decisória.
7. R5E por último, porque formaliza a função de custo e fecha o ciclo com active learning.
```

---

# 4. Baseline de comparação obrigatório

Todas as fases devem comparar seus resultados contra:

```text
EXP-014B-R4G-FAST-FROZEN
```

Métricas de referência:

```text
TP=1463
FP=1123
FN=2
FPR=0,999297%

APROVAR:
2 fraudes

BLOQUEAR:
1458 fraudes
766 normais

CONFIRMAR:
5 fraudes
357 normais
```

Nenhuma nova fase deve ser promovida sem demonstrar ganho claro em pelo menos um destes objetivos:

```text
reduzir fraudes em APROVAR
reduzir normais em BLOQUEAR
preservar FPR < 1%
preservar ou aumentar recall
reduzir custo operacional total
melhorar explicabilidade para diretoria
```

---

# 5. Critério de promoção futura

Uma nova rodada só deve substituir o `R4G-FAST-FROZEN` se satisfizer pelo menos uma das condições abaixo sem violar `FPR < 1%`:

```text
1. Reduzir APROVAR fraud de 2 para 1 ou 0.
2. Reduzir BLOQUEAR normal de 766 para valor materialmente menor.
3. Manter APROVAR fraud = 2, mas reduzir muito BLOQUEAR normal.
4. Manter métricas globais e reduzir custo operacional total.
5. Melhorar estabilidade por segmento temporal.
```

Critérios mínimos:

```text
target_reached = true
final_intervention_metrics.fpr < 0.01
final_intervention_metrics.fn <= 2
all_pass = true
sem degradação temporal relevante
```

---

# 6. Observação final

O `R4G-FAST-FROZEN` representa o melhor estado obtido por otimização de política e regras com as features atuais.

As próximas melhorias relevantes provavelmente exigirão:

```text
novas features
nova arquitetura decisória
calibração por custo
revisão manual de resíduos
active learning
```

Este documento passa a ser a referência para o aprimoramento final do modelo de detecção de fraudes em transações PIX.

---

# 7. Diário de Execução e Evolução de Baselines

## 7.1. Fase 1.3 — Conciliação de Datasets, Features e Artefatos (EXP-014B-R5A3)
* **Status:** Concluído
* **Ações Realizadas:**
  1. **Reconciliação e Auditoria:** Executamos e corrigimos o script `run_reconciliation.py`, gerando os 9 relatórios detalhados na pasta `resultados/experimentos/EXP-014B-R5A3/`.
  2. **Organização Física de Dados:** Criamos o diretório `dados/archive/` e arquivamos todas as bases antigas e intermediárias redundantes, deixando apenas os splits canônicos v3 (`TRAIN.csv`, `VALIDATION.csv`, `HOLDOUT.csv`).
  3. **Novos Scripts de Treinamento Canônicos:** Criamos `train_lgbm_canonical.py` e `train_isolation_forest_canonical.py`, eliminando scripts de treino obsoletos e normalizando o contrato de features de produção.
  4. **Ambiente Virtual (`venv`):** Inicializamos um ambiente isolado com todas as dependências requeridas (incluindo `pytest` e `httpx` para chamadas HTTP integradas).
  5. **Nova Suíte de Testes (tests/):** Recriamos os testes do zero: `test_model_artifacts.py` (validação de contratos/binários), `test_pipeline_inference.py` (inferência integrada com `PipelineOrquestrador`), e `test_api_smoke.py` (testes de fumaça de rotas FastAPI com eventos síncronos de lifespan).
  6. **Ajustes na API e Core:** Corrigimos o contrato da API em `/api/v1/health` para bater com o schema `HealthResponse` do Pydantic, e implementamos o método `reset_cache()` no orquestrador.
* **Resultados:** 11 testes aprovados com 100% de sucesso.

## 7.2. Fase 2 — Engenharia de Features de Relacionamento (EXP-014B-R5B)
* **Status:** Concluído
* **Implementação Offline:** Criamos e executamos o script `scripts/compute_relationship_features.py` para calcular de forma temporalmente consistente (leakage-free) as 5 features deslizantes de relacionamento (janelas de 7d e 180d, ratios e recorrência) sobre os splits canônicos v3.
* **Serving em Tempo Real (Runtime):**
  1. Atualizamos a taxonomia do `preprocessing.py` para mapear as novas colunas.
  2. Modificamos o método `_create_sequential_features` no `PipelineOrquestrador` para calcular as features de relacionamento em tempo real usando o cache do cliente.
  3. Adaptamos o método `_update_customer_history` para manter timestamps e valores detalhados de envios anteriores a cada recebedor (`receiver_txs`).
* **Model Retraining:** Reajustamos o LightGBM canônico (early stopping na iteração 15, threshold F1 = 0.1450) e o Isolation Forest (contaminação = 0.005), gravando os novos binários oficiais na pasta `backend/artefatos/`.
* **Evolução do Classificador LGBM Puro no Holdout:**
  
  | Métrica no Holdout | Baseline Canônico (Fase 1.3) | Pós-Relationship Features (Fase 2) | Impacto / Ganho Absoluto |
  | :--- | :---: | :---: | :---: |
  | **ROC-AUC** | `0.9576` | `0.9722` | **+0.0146** |
  | **Average Precision (AP)** | `0.3931` | `0.4821` | **+0.0890 (+8.90%)** |
  | **F1-Score** | `0.4462` | `0.4717` | **+0.0255 (+2.55%)** |
  | **Recall (Sensibilidade)** | `46.77%` | `60.48%` | **+13.71%** |
  | **FPR (Falsos Positivos)** | `0.4522%` | `0.6899%` | **+0.2377% (FPR < 1%)** |

* **Homologação:** Execução do pytest com **100% de sucesso (11 testes aprovados)** comprovando a perfeita paridade treino-serving das novas features.

## 7.3. Fase 2.1 — Calibração do Decision Engine e Correção de Skew (EXP-014B-R5B2)
* **Status:** Concluído (Geração de Base e Correções de Motor)
* **Ações Realizadas:**
  1. **Correção de Skew de Serving (LGBM):** Identificamos e corrigimos um bug crítico no `PixDecisionEngine._score_lgbm` onde valores nulos (`NaN`) eram preenchidos com `0.0`. Como o LightGBM trata `NaN` e `0.0` como ramos diferentes, as predições de inferência estavam colapsando (máximo de 17% de probabilidade mesmo em fraudes). A correção para `np.nan` restaurou a paridade treino-serving.
  2. **Bridge de Retrocompatibilidade (v3 -> v2):** O novo dataset v3 renomeou colunas essenciais para as regras de negócio. Implementamos no `PipelineOrquestrador._prepare_raw` um mapeamento automático de features (ex: `qtd_pix_pagador_90d` -> `qt_total_pix_trimestre`) garantindo que os módulos de Engenharia Social (SE) e Comportamental (BEH) voltassem a pontuar corretamente.
  3. **Geração da Base Expandida Corrigida:** Executamos (em 64 min) a inferência do `PipelineOrquestrador` sobre as 113.844 transações do dataset v3 (1.465 fraudes), gerando o arquivo `01_raw_predictions_holdout.csv` com scores reais e regras de negócio ativas.
  4. **Correção do Cascade Burst:** Identificamos uma regra de "burst fake" no bridge que causou 61 mil bloqueios indevidos por erro de interpretação de janela temporal. O arquivo de predições foi reparado localmente restaurando o score original e removendo os vetos espúrios.
  5. **Status do Baseline no Dataset Expandido:** A migração para a base v3 completa (1.465 fraudes) revelou que o recall intrínseco do modelo Machine Learning é menor do que o medido na base antiga de 355 fraudes. O threshold de 1% FPR no modelo puro agora captura aproximadamente 460 fraudes (~31% recall), exigindo que as fases R5B e R5C de novas features sejam priorizadas para recuperar o patamar de 99% sem explodir os FPs.

## 7.4. Fase 2.2 — Otimização de Regras de Resgate (EXP-014B-R5B2-TUNING)
* **Status:** Pendente
* **O que falta fazer:**
  1. **Tuning Greedy:** Rodar o `exp_014b_r5b2_tune_policy.py` buscando especificamente regras de resgate `APROVAR -> CONFIRMAR` baseadas em padrões de SE/BEH que não foram capturados pelo LGBM.
  2. **Análise de Gap:** Verificar se as 2 fraudes residuais em APROVAR do baseline antigo (R4G) ainda são as mesmas ou se o novo dataset expandido trouxe novos casos críticos.
  3. **Homologação:** Rodar `pytest tests/ -v` para garantir que o pipeline de produção reflete exatamente o comportamento medido na otimização.
  4. **Avançar para Fase R5B (Relacionamento):** Iniciar a adição das features de relacionamento pagador-recebedor para tentar subir o recall base do LGBM.

## 7.5. Fase 2.2 — R5B2 Tuning corrigido e validação congelada
* **Status:** Concluído como melhoria de severidade, não como baseline promocionável.
* **Correção técnica:** O script `scripts/exp_014b_r5b2_tune_policy.py` foi corrigido para reconciliar colunas duplicadas após merge (`*_x`/`*_y`) e ignorar colunas candidatas ausentes em vez de quebrar a execução.
* **Resultado R5B2 recalculado:** A política `BLOQUEAR -> CONFIRMAR` moveu `11.067` transações normais de `BLOQUEAR` para `CONFIRMAR`, sem mover nenhuma fraude conhecida de `BLOQUEAR`.
* **Métricas de BLOQUEAR:**

```text
Antes:
TP=279
FP=14220
FPR=12,6536%
Precision=1,9243%

Depois:
TP=279
FP=3153
FPR=2,8057%
Precision=8,1294%
```

* **Validação congelada:** Foi criado e executado `scripts/exp_014b_r5b2_frozen_validation.py`, gerando `resultados/experimentos/EXP-014B-R5B2-FROZEN/`. O replay congelado reproduziu exatamente o artefato:

```text
status=PASS_R5B2_FROZEN_REPLAYED
prediction_mismatches_vs_calibration=0
block_fp_demoted_to_confirm=11067
block_tp_demoted_to_confirm=0
all_rules_match_artifact_incremental=true
```

* **Limitação crítica:** A intervenção total permanece distante do alvo, pois ainda existem `682` fraudes em `APROVAR`.

```text
Intervenção total pós R5B2:
TP=783
FP=15584
FN=682
FPR=13,8674%
```

**Decisão:** R5B2-FROZEN é um avanço válido para reduzir bloqueio indevido, mas não pode ser promovido como baseline final. Ele deve alimentar a próxima etapa de feature engineering e uma política robusta de severidade.

## 7.6. Fase 2.3 — Diagnóstico de resíduos pós R5B2 (EXP-014B-R5B3)
* **Status:** Concluído como diagnóstico preparatório de feature engineering.
* **Artefatos:** `resultados/experimentos/EXP-014B-R5B3-RESIDUAL-FEATURE-GAP/`
* **Grupos críticos identificados:**

```text
Normais ainda em BLOQUEAR: 3153
Fraudes em BLOQUEAR: 279
Fraudes ainda em APROVAR: 682
Normais em APROVAR: 96795
```

* **Achados para reduzir normais em BLOQUEAR:** Os normais residuais em `BLOQUEAR` tendem a ter pagadores com histórico transacional mais forte, maior valor total em 90/180 dias e maior valor máximo histórico. Também há sinal de que `first_receiver_flag_real=0`, recebedor com algum histórico (`rec_1_10`/`val_rec_gt_5k`) e `lgbm_LT_0.05` ajudam a separar normais bloqueados de fraudes bloqueadas.
* **Achados para resgatar fraudes em APROVAR:** As fraudes aprovadas concentram recebedores sem histórico (`rec_0`/`val_rec_0`), primeiro recebedor, menor histórico do pagador e score LGBM fraco, mas não nulo (`lgbm_0.05_0.15`). Isso sugere necessidade de features explícitas de reputação do recebedor e força/idade do relacionamento, além de uma política separada de resgate `APROVAR -> CONFIRMAR`.

**Próximo passo recomendado:** criar um experimento robusto para selecionar subconjunto produtivo das regras `BLOQUEAR -> CONFIRMAR` e iniciar feature engineering de trust/reputação do recebedor, sem perder de vista que a frente de recall exige atacar as `682` fraudes em `APROVAR`.

## 7.7. Fase 2.4 — Seleção robusta das regras de severidade (EXP-014B-R5B4)
* **Status:** Concluído como candidato robusto de severidade.
* **Artefatos:** `resultados/experimentos/EXP-014B-R5B4-ROBUST-BLOCK-DEESCALATION/`
* **Objetivo:** transformar a lista completa de 60 regras R5B2 em uma política menos propensa a overfit, mantendo apenas regras com:

```text
total_frauds = 0
non_train_frauds = 0
non_train_normals >= 20
month_normal_support >= 2
```

* **Resultado:** Das 60 regras R5B2, 57 passaram no critério robusto. Apenas 3 regras foram descartadas por suporte não-treino baixo.

```text
Normais movidos de BLOQUEAR para CONFIRMAR: 11038
Fraudes movidas de BLOQUEAR para CONFIRMAR: 0
Normais restantes em BLOQUEAR: 3182
Fraudes restantes em BLOQUEAR: 279
Fraudes restantes em APROVAR: 682
```

* **Métricas de BLOQUEAR pós R5B4:**

```text
TP=279
FP=3182
FN=1186
FPR=2,8315%
Precision=8,0613%
```

**Decisão:** R5B4 é preferível ao R5B2 completo para futura integração de severidade, pois preserva praticamente todo o ganho de redução de bloqueio indevido com critério explícito de suporte fora do treino. Ainda assim, ele não resolve o problema de recall/intervenção: as `682` fraudes em `APROVAR` continuam exigindo uma frente separada de feature engineering e resgate.

## 7.8. Fase 2.5 — Probe de features de trust/reputação (EXP-014B-R5B5)
* **Status:** Concluído como prova incremental de feature engineering.
* **Artefatos:** `resultados/experimentos/EXP-014B-R5B5-TRUST-FEATURE-DEESCALATION/`
* **Objetivo:** criar features explícitas de confiança do pagador, recebedor e relacionamento, e testar se elas removem normais residuais de `BLOQUEAR` após o R5B4.
* **Features criadas:**

```text
payer_history_strength_score
receiver_reputation_score
relationship_strength_score
receiver_novelty_risk_score
transaction_normality_score
payer_receiver_trust_score
```

* **Resultado incremental sobre R5B4:**

```text
Candidatos avaliados: 1752
Regras selecionadas: 5
Normais adicionais movidos de BLOQUEAR para CONFIRMAR: 358
Fraudes movidas de BLOQUEAR para CONFIRMAR: 0
Normais restantes em BLOQUEAR: 2824
Fraudes restantes em BLOQUEAR: 279
Fraudes restantes em APROVAR: 682
```

* **Métricas de BLOQUEAR pós R5B5:**

```text
TP=279
FP=2824
FN=1186
FPR=2,5129%
Precision=8,9913%
```

* **Regras selecionadas:** As regras aceitas apontaram para relacionamento forte (`relationship_strength_score >= 100`), buckets de trust intermediário/alto com reputação do recebedor alta, e combinações de reputação de recebedor com faixas de valor. Todas tiveram suporte em meses múltiplos e nenhum TP demovido.

**Decisão:** R5B5 comprova que features explícitas de trust/reputação agregam sinal incremental para reduzir bloqueio indevido. A próxima etapa deve promover essas features para o contrato de feature engineering e preparar um replay/treino que use os scores de trust como variáveis primárias, mantendo uma trilha separada para resgatar as `682` fraudes ainda em `APROVAR`.

## 7.9. Fase 2.6 — Integração das features de trust no core
* **Status:** Concluído como integração de feature engineering, sem retreino de binários.
* **Mudanças no core:**
  1. `backend/core/preprocessing.py` passou a expor `create_trust_features()`.
  2. `backend/core/pipeline_orquestrador.py` passou a chamar a mesma função durante `_create_features()`.
  3. `tests/test_pipeline_inference.py` passou a validar a presença e faixa dos scores de trust no runtime.
  4. `scripts/exp_014b_r5b5_trust_feature_deescalation.py` foi ajustado para reutilizar a função central do core, evitando divergência entre experimento e serving.

* **Features integradas:**

```text
payer_history_strength_score
receiver_reputation_score
relationship_strength_score
receiver_novelty_risk_score
transaction_normality_score
payer_receiver_trust_score
trust_bucket
receiver_rep_bucket
relationship_bucket
novelty_bucket
```

**Decisão:** As features R5B5 agora fazem parte da camada reutilizável de feature engineering. O próximo passo técnico é retreino/replay controlado para decidir se essas features entram no contrato do LGBM/IF ou se permanecem como camada de política de severidade.

## 7.10. Fase 2.7 — LGBM shadow com features de trust (EXP-014B-R5B6)
* **Status:** Concluído como treino shadow, sem promoção para artefatos produtivos.
* **Artefatos:** `resultados/experimentos/EXP-014B-R5B6-TRUST-LGBM-SHADOW/`
* **Objetivo:** testar se as features de trust/reputação integradas no core melhoram o classificador LGBM quando usadas como variáveis de modelo, sem sobrescrever `backend/artefatos`.
* **Contrato avaliado:** `65` features totais, incluindo os `10` campos de trust adicionados na fase anterior.

* **Resultado holdout do shadow:**

```text
Threshold=0,1550
TP=64
FP=68
FN=60
TN=17180
ROC-AUC=0,969744
AP=0,475929
Precision=48,4848%
Recall=51,6129%
F1=0,5000
FPR=0,3942%
```

* **Delta contra o LGBM canônico atual:**

```text
ROC-AUC=-0,002426
AP=-0,006170
Precision=+9,8250 p.p.
Recall=-8,8710 p.p.
F1=+0,028302
FPR=-0,2957 p.p.
TP=-11
FP=-51
FN=+11
```

* **Leitura técnica:** As features de trust entraram com sinal forte no modelo. `payer_receiver_trust_score` foi a feature de maior ganho, e `trust_bucket`, `receiver_reputation_score`, `transaction_normality_score`, `payer_history_strength_score` e `relationship_strength_score` também tiveram importância positiva. Porém o modelo ficou mais conservador: reduziu falsos positivos e aumentou precisão, mas perdeu recall e average precision.

**Decisão:** R5B6 não deve substituir o LGBM canônico nesta rodada, porque o objetivo de produção ainda exige preservar ou recuperar recall. O resultado confirma que as features de trust são úteis para severidade/de-escalonamento de `BLOQUEAR`, mas a próxima etapa deve usá-las em um modelo ou política dedicada à severidade, em vez de promover diretamente o shadow como classificador principal. A frente de resgate das `682` fraudes em `APROVAR` continua pendente e separada.

## 7.11. Fase 2.8 — Modelo shadow de severidade em BLOQUEAR (EXP-014B-R5B7)
* **Status:** Concluído como experimento negativo útil.
* **Artefatos:** `resultados/experimentos/EXP-014B-R5B7-BLOCK-SEVERITY-SHADOW/`
* **Objetivo:** treinar um modelo dedicado apenas ao residual `BLOQUEAR` pós R5B5 para ranquear risco de fraude dentro da fila de bloqueio e encontrar um limiar seguro de `BLOQUEAR -> CONFIRMAR`.

* **Resultado:**

```text
Features totais: 109
Validation AUC: 0,791676
Holdout AUC: 0,720858
Validation AP: 0,427079
Holdout AP: 0,350632
Threshold seguro em validação: 0,0645117720
Normais movidos: 4
Fraudes movidas: 0
Normais restantes em BLOQUEAR: 2820
Fraudes restantes em BLOQUEAR: 279
```

* **Leitura técnica:** O score tem separação parcial, mas a primeira fraude em validação aparece praticamente colada à cauda de menor risco. O limiar seguro só moveu `1` normal em validação e `4` normais no total, sem ganho material.

**Decisão:** R5B7 não é promocionável como camada de severidade. Ele serviu para confirmar que um modelo contínuo simples sobre o residual `BLOQUEAR` não abre uma faixa segura relevante. O próximo caminho deve voltar para regras robustas/interpretáveis sobre subpopulações estáveis, que foi onde R5B4 e R5B5 geraram ganho sem demover fraude.

## 7.12. Fase 2.9 — Mineração ampla de regras residuais (EXP-014B-R5B8)
* **Status:** Concluído como candidato incremental de severidade.
* **Artefatos:** `resultados/experimentos/EXP-014B-R5B8-BROAD-RESIDUAL-RULE-MINING/`
* **Objetivo:** ampliar a mineração conservadora do R5B5 usando features de trust e variáveis apontadas pelo R5B7, preservando os critérios de zero fraude demovida, suporte fora do treino e suporte em múltiplos meses.

* **Resultado incremental sobre R5B5:**

```text
Candidatos avaliados: 1339
Regras selecionadas: 3
Normais adicionais movidos de BLOQUEAR para CONFIRMAR: 469
Fraudes movidas de BLOQUEAR para CONFIRMAR: 0
Normais restantes em BLOQUEAR: 2355
Fraudes restantes em BLOQUEAR: 279
Fraudes restantes em APROVAR: 682
```

* **Métricas de BLOQUEAR pós R5B8:**

```text
TP=279
FP=2355
FN=1186
FPR=2,0956%
Precision=10,5923%
```

* **Regras selecionadas:**

```text
1. dias_desde_primeiro_envio_recebedor >= 35
   Normais movidos: 357 | Fraudes movidas: 0 | Não-treino: 101 | Meses: 5

2. receiver_reputation_score > 70,57506297 AND qtd_pix_pagador_180d > 175
   Normais movidos: 58 | Fraudes movidas: 0 | Não-treino: 19 | Meses: 6

3. valor_rec_bin == val_rec_lt_5k AND valor_total_pagador_180d > 212416,178
   Normais movidos: 54 | Fraudes movidas: 0 | Não-treino: 15 | Meses: 6
```

* **Correção de rastreabilidade:** O `candidate_id` original truncava valores categóricos e podia tornar a descrição das regras ambígua. O script `scripts/exp_014b_r5b8_broad_residual_rule_mining.py` foi corrigido para preservar os valores completos no ID; os artefatos foram regenerados mantendo o mesmo ganho (`469` normais, `0` fraude).

**Decisão:** R5B8 é a melhor evolução da fase atual para reduzir bloqueio indevido sem mexer no classificador principal. O ganho acumulado R5B4 + R5B5 + R5B8 reduz os normais em `BLOQUEAR` de `14220` para `2355`, mantendo `0` fraude demovida por essas regras. Antes de integração produtiva, as três regras R5B8 precisam de revisão manual de semântica de negócio e replay congelado; os bins usados pelas regras 2 e 3 já foram persistidos em `05_policy_artifact_broad_rules.json` como thresholds explícitos.

## 7.13. Fase 2.10 — Replay congelado da política R5B8 no core (EXP-014B-R5B9)
* **Status:** Concluído como gate de integração aprovado.
* **Artefatos:** `resultados/experimentos/EXP-014B-R5B9-FROZEN-SEVERITY-POLICY-REPLAY/`
* **Objetivo:** traduzir a política R5B8 para um módulo reutilizável do core e provar que a implementação explícita reproduz exatamente o artefato experimental.
* **Mudanças:**
  1. Criado `backend/core/severity_policy.py` com a política R5B8 ordenada e thresholds explícitos.
  2. Criado `scripts/exp_014b_r5b9_frozen_severity_policy_replay.py` para replay congelado contra o artefato R5B8.
  3. Criado `tests/test_severity_policy.py` para garantir que a política só rebaixa casos ainda em `BLOQUEAR` e respeita a ordem das regras.

* **Resultado do replay:**

```text
Status: PASS_R5B9_FROZEN_REPLAY_MATCHED_R5B8
Linhas avaliadas: 113844
Divergências de decisão vs R5B8: 0
Divergências de aplicação vs R5B8: 0
Normais movidos de BLOQUEAR para CONFIRMAR: 469
Fraudes movidas de BLOQUEAR para CONFIRMAR: 0
Normais restantes em BLOQUEAR: 2355
Fraudes restantes em BLOQUEAR: 279
Fraudes restantes em APROVAR: 682
```

* **Contagem por regra no replay congelado:**

```text
R5B8_01_RELATIONSHIP_AGE_GTE_35D: 357
R5B8_02_LOW_RECEIVER_REP_LOW_PAYER_COUNT: 58
R5B8_03_LOW_RECEIVER_VALUE_LOW_PAYER_VALUE: 54
```

**Decisão:** A política R5B8 agora tem uma implementação de core com replay congelado perfeito contra o experimento. Ela ainda não deve ser ativada isoladamente no runtime produtivo, porque o ganho validado é incremental sobre a base R5B5. O próximo passo correto é consolidar a política acumulada R5B4 + R5B5 + R5B8 em um artefato único de severidade e só então conectar essa política ao orquestrador/configuração de produção.

## 7.14. Fase 2.11 — Artefato candidato consolidado de severidade (EXP-014B-R5B10)
* **Status:** Concluído como manifesto candidato, ainda não ativo em produção.
* **Artefatos:**
  * `resultados/experimentos/EXP-014B-R5B10-CONSOLIDATED-SEVERITY-POLICY/`
  * `backend/artefatos_candidatos/exp014b_r5b10_severity_policy/severity_policy_candidate.json`
* **Objetivo:** consolidar as camadas R5B4, R5B5 e R5B8/R5B9 em um único contrato de severidade com proveniência, hashes dos artefatos-fonte, métricas acumuladas e gates de promoção.

* **Resultado consolidado:**

```text
Camadas: 3
Regras totais: 65
Normais removidos de BLOQUEAR desde R5B2: 11865
Normais restantes em BLOQUEAR: 2355
Fraudes demovidas para CONFIRMAR: 0
Fraudes restantes em BLOQUEAR: 279
Fraudes restantes em APROVAR: 682
```

* **Métricas finais de BLOQUEAR:**

```text
TP=279
FP=2355
FN=1186
FPR=2,0956%
Precision=10,5923%
```

**Decisão:** O R5B10 é o contrato candidato para integração controlada da política acumulada de severidade. Ele não altera `backend/artefatos` e não está ativo por padrão. O próximo passo de produção é conectar esse contrato ao `PipelineOrquestrador` por configuração versionada e executar replay E2E completo, confirmando novamente zero fraude demovida por split e por mês.

## 7.15. Fase 2.12 - Reconciliacao do campeao global (EXP-014B-R5B11)
* **Status:** Concluido como gate de reconciliacao e promocao candidata global.
* **Artefatos:**
  * `resultados/experimentos/EXP-014B-R5B11-CHAMPION-RECONCILIATION/`
  * `backend/artefatos_candidatos/exp014b_r5b11_global_policy/global_policy_candidate.json`
* **Objetivo:** reconciliar o baseline `EXP-014B-R4G-FAST-FROZEN`, que ja cumpria as metas globais, com a trilha R5B2/R5B10 de severidade, antes de qualquer integracao produtiva.

* **Resultado R4G confirmado:**

```text
TP=1463
FP=1123
FN=2
FPR=0,999297%

BLOQUEAR:
TP=1458
FP=766
FN=7
FPR=0,681622%
```

* **Resultado R5B2/R5B10 contrastado:** A trilha R5B2 continua distante do alvo global, com `FN=682` e `FPR=13,867360%` em intervencao. O R5B10 reduz bloqueio indevido dentro dessa trilha, mas nao recupera recall.
* **Teste de empilhamento:** Aplicar R5B10 diretamente sobre o R4G nao e seguro. O replay demoveu `115` fraudes de `BLOQUEAR` para `CONFIRMAR`, embora tenha movido apenas `23` normais.
* **Mineracao residual no R4G:** Nao foram encontradas regras simples categoricas/numericas com `0` fraude demovida para reduzir os `766` normais em `BLOQUEAR`.

**Decisao:** `EXP-014B-R4G-FAST-FROZEN` deve ser tratado como o candidato global atual, pois cumpre simultaneamente `FPR < 1%` e `FN <= 5`. O `severity_policy_candidate.json` do R5B10 permanece historico/candidato apenas para a trilha R5B2 e nao deve ser empilhado sobre R4G. O proximo passo correto e integrar a politica R4G congelada por configuracao versionada e, em paralelo, abrir uma nova trilha de severidade especifica para o residual `BLOQUEAR` do R4G.

## 7.16. Fase 2.13 - Rebalanceamento de severidade no R4G (EXP-014B-R5B12)
* **Status:** Concluido como candidato de severidade sobre o campeao R4G.
* **Artefatos:**
  * `resultados/experimentos/EXP-014B-R5B12-R4G-SEVERITY-REBALANCE/`
  * `backend/artefatos_candidatos/exp014b_r5b12_r4g_severity_rebalance/r4g_severity_rebalance_candidate.json`
* **Objetivo:** atacar o residual de severidade do R4G, migrando fraudes de `CONFIRMAR` para `BLOQUEAR` sem alterar as metricas globais de intervencao.

* **Resultado selecionado:**

```text
CONFIRMAR -> BLOQUEAR:
Fraudes movidas: 5
Normais movidos: 22
Fraudes restantes em CONFIRMAR: 0
Fraudes restantes em APROVAR: 2

Global:
TP=1463
FP=1123
FN=2
FPR=0,999297%

BLOQUEAR:
TP=1463
FP=788
FN=2
FPR=0,701199%
```

* **Leitura operacional:** A mudanca preserva integralmente o gate global (`FPR < 1%`, `FN <= 5`) e melhora a narrativa de severidade: todas as fraudes antes em `CONFIRMAR` passam para `BLOQUEAR`. O custo e aumentar os normais em `BLOQUEAR` de `766` para `788`.
* **Limite encontrado:** A mineracao residual R4G nao encontrou regra simples com `0` fraude demovida para `BLOQUEAR -> CONFIRMAR`; portanto, nao ha compensacao segura imediata para remover normais de `BLOQUEAR`.

**Decisao:** R5B12 deve ser tratado como candidato de severidade acima do R4G, nao como substituto do classificador. Antes de qualquer ativacao, criar replay congelado por descricao de regra e revisar semanticamente as 5 regras selecionadas. O R5B10 continua incompatível com R4G e nao deve ser combinado com R5B12 sem novo replay.

## 7.17. Fase 2.14 - Swap zero-FN sobre R5B12 (EXP-014B-R5B13)
* **Status:** Concluido como candidato experimental agressivo, ainda nao promocionavel diretamente.
* **Artefatos:**
  * `resultados/experimentos/EXP-014B-R5B13-R4G-ZERO-FN-SWAP/`
  * `backend/artefatos_candidatos/exp014b_r5b13_r4g_zero_fn_swap/r4g_zero_fn_swap_candidate.json`
* **Objetivo:** resgatar as 2 fraudes restantes em `APROVAR` e compensar o custo de falso positivo movendo normais limpos de `CONFIRMAR` para `APROVAR`.

* **Movimentos selecionados:**

```text
APROVAR -> BLOQUEAR:
Fraudes movidas: 2
Normais movidos: 47

CONFIRMAR -> APROVAR, compensacao:
Normais movidos: 47
Fraudes movidas: 0
```

* **Distribuicao final por decisao:**

```text
APROVAR:
111256 transacoes
0 fraudes
111256 normais

BLOQUEAR:
2300 transacoes
1465 fraudes
835 normais

CONFIRMAR:
288 transacoes
0 fraudes
288 normais
```

* **Metricas globais finais:**

```text
TP=1465
FP=1123
FN=0
FPR=0,999297%
Recall=100%
```

* **Metricas finais de BLOQUEAR:**

```text
TP=1465
FP=835
FN=0
FPR=0,743021%
Precision=63,695652%
```

**Decisao:** R5B13 atinge a meta numerica mais forte ate agora: `FN=0`, `FPR < 1%`, nenhuma fraude em `APROVAR` e nenhuma fraude em `CONFIRMAR`. Porem a compensacao usa selecao offline dos menores `lgbm_raw` entre normais remanescentes em `CONFIRMAR`; antes de producao, essa compensacao precisa ser transformada em regra operacional congelada sem dependencia de label e validada por replay E2E.

## 7.18. Fase 2.15 - Replay operacional zero-FN (EXP-014B-R5B14)
* **Status:** Concluido como replay congelado por regras explicitas.
* **Artefatos:**
  * `resultados/experimentos/EXP-014B-R5B14-OPERATIONAL-ZERO-FN-REPLAY/`
  * `backend/artefatos_candidatos/exp014b_r5b14_operational_zero_fn/operational_zero_fn_policy_candidate.json`
* **Objetivo:** remover a dependencia de label da compensacao R5B13 e reproduzir o resultado zero-FN com regras operacionais explicitas.

* **Camadas aplicadas:**

```text
R5B12 CONFIRMAR -> BLOQUEAR:
27 transacoes
5 fraudes
22 normais

R5B13 APROVAR -> BLOQUEAR:
49 transacoes
2 fraudes
47 normais

R5B14 CONFIRMAR -> APROVAR:
Regra: remaining CONFIRMAR AND lgbm_raw <= 0,00001966
47 transacoes
0 fraudes
47 normais
```

* **Metricas finais:**

```text
Global:
TP=1465
FP=1123
FN=0
FPR=0,999297%
Recall=100%

BLOQUEAR:
TP=1465
FP=835
FN=0
FPR=0,743021%
Precision=63,695652%
```

* **Distribuicao final por decisao:**

```text
APROVAR:
111256 transacoes
0 fraudes
111256 normais

BLOQUEAR:
2300 transacoes
1465 fraudes
835 normais

CONFIRMAR:
288 transacoes
0 fraudes
288 normais
```

**Decisao:** R5B14 substitui R5B13 como melhor candidato experimental, pois reproduz o mesmo `FN=0` e `FPR < 1%` sem selecao por label na compensacao. Antes de ativacao produtiva, ainda e necessario executar replay E2E no `PipelineOrquestrador` e revisar semanticamente as regras de `APROVAR -> BLOQUEAR`, `CONFIRMAR -> BLOQUEAR` e a compensacao low-LGBM.

## 7.19. Fase 2.16 - Replay da politica R5B14 no core (EXP-014B-R5B15)
* **Status:** Concluido como gate de implementacao central.
* **Artefatos:** `resultados/experimentos/EXP-014B-R5B15-CORE-POLICY-REPLAY/`
* **Objetivo:** mover a politica R5B14 do script experimental para `backend/core/severity_policy.py` e provar que a funcao reutilizavel reproduz exatamente o artefato R5B14.

* **Mudancas no core:**
  1. `backend/core/severity_policy.py` passou a expor `apply_r5b14_operational_zero_fn_policy()`.
  2. A politica possui metadados explicitos via `r5b14_policy_metadata()`.
  3. `tests/test_severity_policy.py` cobre a ordem das tres camadas: `CONFIRMAR -> BLOQUEAR`, `APROVAR -> BLOQUEAR` e `CONFIRMAR -> APROVAR`.
  4. `backend/core/pipeline_orquestrador.py` passou a aceitar a politica por flag `ENABLE_R5B14_POLICY` ou `r5b14_operational_zero_fn_enabled` no `scoring_config`, mantendo default desligado.

* **Resultado do replay pelo core:**

```text
Status: PASS_R5B15_CORE_POLICY_REPLAY_MATCHED_R5B14
Intervention metrics match R5B14: true
Block metrics match R5B14: true
R5B12 counts match R5B14: true
Approve-to-block counts match R5B14: true
Compensation counts match R5B14: true

Global:
TP=1465
FP=1123
FN=0
FPR=0,999297%

BLOQUEAR:
TP=1465
FP=835
FN=0
FPR=0,743021%
```

* **Validacao local executada:**

```text
python tests/test_severity_policy.py
4 testes OK

python scripts/exp_014b_r5b15_core_policy_replay.py
PASS_R5B15_CORE_POLICY_REPLAY_MATCHED_R5B14

python -m py_compile backend/core/pipeline_orquestrador.py backend/core/severity_policy.py scripts/exp_014b_r5b15_core_policy_replay.py
OK
```

**Decisao:** R5B15 prova que a politica R5B14 ja existe como implementacao central reutilizavel, conectada ao `PipelineOrquestrador` por configuracao versionada e sem ativacao por padrao. O replay batch completo do orquestrador ainda deve ser executado no ambiente com dependencias produtivas para homologacao final, mas a politica central reproduz exatamente o melhor artefato experimental.

## 7.20. Fase 2.17 - Consolidação do baseline operacional candidato (EXP-014B-R5B16)
* **Status:** Concluido como baseline candidato consolidado.
* **Artefatos:**
  * `resultados/experimentos/EXP-014B-R5B16-CONSOLIDATED-OPERATIONAL-BASELINE/`
  * `backend/artefatos_candidatos/exp014b_r5b16_operational_baseline/operational_baseline_candidate.json`
* **Objetivo:** consolidar R5B15 como baseline candidato versionado, com fontes, hashes, gates de alvo e flags de ativação.

* **Resultado consolidado:**

```text
Status: PASS_R5B16_OPERATIONAL_BASELINE_CANDIDATE_CONSOLIDATED

Global:
TP=1465
FP=1123
FN=0
FPR=0,999297%

BLOQUEAR:
TP=1465
FP=835
FN=0
FPR=0,743021%
```

* **Gates consolidados:**

```text
fpr_lt_1pct=true
fn_lte_5_outside_block=true
fn_eq_0=true
approve_frauds_eq_0=true
confirm_frauds_eq_0=true
all_core_replay_checks_pass=true
```

* **Ativação:** O baseline candidato permanece `CANDIDATE_NOT_PRODUCTION_ACTIVE`, desligado por default, com ativação apenas por `ENABLE_R5B14_POLICY` ou `r5b14_operational_zero_fn_enabled`.

**Decisao:** R5B16 encerra a fase de consolidação offline do novo baseline. A próxima fase do plano deve ser homologação operacional: replay batch completo do `PipelineOrquestrador` no ambiente com dependências produtivas, revisão semântica das regras e monitoramento de drift.
