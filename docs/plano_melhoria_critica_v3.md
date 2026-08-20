# Plano de Melhoria Crítica 3 — Viabilização do Modelo Antifraude PIX com MAF, MBK, Grafos Leves e Feature Store

**Status:** Plano oficial a partir de 2026-05-13  
**Substitui:** `plano_melhoria_critica_1.md` e `plano_melhoria_crítica_2.md`  
**Baseline de entrada:** `post_fase2_c1`  
**Diretriz principal:** abandonar a busca por novas regras sobre os mesmos sinais e evoluir o modelo com novos dados, hidratação MBK, amostragem controlada de normais, validação de grafos leves e preparação da Feature Store HBase.

---

## 1. Decisão de mudança de rumo

Os planos anteriores cumpriram seu papel:

- o **Plano 1** conduziu a investigação inicial, estabilização de métricas, otimização cirúrgica, promoção da V1 Guard Contextual e posterior FASE 2 com retreino/modelagem shadow;
- o **Plano 2** consolidou o baseline pós-C1, governança, regressão, observabilidade, drift, dashboard operacional, data contract e preparação para novos dados.

A partir de agora, ambos ficam **encerrados como planos ativos**. Elementos válidos deles passam a ser incorporados como **gates operacionais permanentes** neste Plano 3.

A mudança de rumo ocorre porque o projeto saiu de uma fase de ajuste fino em cima de poucos sinais e entrou em uma fase de **engenharia de dados + revalidação supervisionada ampliada**.

A nova realidade é:

```text
Baseline pós-C1 estável:
  seed 42:  TP=347, FP=14, FN=8, F1≈0,9693
  seed 123: TP=347, FP=12, FN=8, F1≈0,9720

Nova fonte MAF:
  labels_curated=134.599
  positive_for_current_model=15.564
  hydrated_final=13.558 fraudes fortes

Novos desafios:
  1. não é viável coletar normais de 2022–2026;
  2. normais devem ficar limitados a 90/180 dias;
  3. as fraudes MAF históricas são valiosas demais para serem descartadas;
  4. MBK precisa ser hidratada com estratégia inteligente, sem varredura massiva;
  5. grafos podem ajudar, mas apenas como features leves e batch;
  6. Feature Store HBase passa a ser parte da estratégia de produção.
```

---

## 2. Baseline oficial e ativos preservados

### 2.1 Baseline do motor de decisão

O baseline oficial continua sendo `post_fase2_c1`.

| Seed | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 347 | 14 | 8 | 96,1219% | 97,7465% | 0,9693 |
| 123 | 347 | 12 | 8 | 96,6574% | 97,7465% | 0,9720 |

Configuração conceitual ativa:

```json
{
  "threshold_confirmar": 62.0,
  "threshold_bloquear": 95.0,
  "lgbm_guard_enabled": true,
  "lgbm_guard_threshold": 0.30,
  "guard_exception_alto_valor_se_beh_enabled": true,
  "exp006f_c1_enabled": true,
  "exp006f_c1_min_score": 58.0,
  "exp006f_c1_max_score": 62.0,
  "exp006f_c1_min_valor": 100.0,
  "exp006f_c1_max_valor": 500.0,
  "exp006f_c1_max_rel_meses": 12.0,
  "exp006f_c1_min_lgbm_raw": 0.06,
  "exp006f_c1_max_lgbm_raw": 0.10,
  "exp006f_c1_require_first_receiver": true,
  "exp006f_c1_require_not_pix_random": true,
  "exp006f_c1_max_se_score": 0.0,
  "exp006f_c1_max_beh_score": 0.0,
  "se_pattern_residual_enabled": false,
  "exp003_residual_confirm_enabled": false
}
```

### 2.2 Decisões preservadas dos planos anteriores

Permanecem válidas:

1. **V1 Guard Contextual ativa** como exceção cirúrgica ao guard rail.
2. **C1 Near-Threshold ativa** e protegida por regressão.
3. **LGBM v6.2 rejeitado para runtime**.
4. **Meta-learner shadow rejeitado para promoção**.
5. **First receiver não deve virar regra hardcoded generalista**.
6. **R2 rejeitada**.
7. **EXP-003 residual desligado**.
8. **Decision logging, drift monitor, fila de revisão e dashboard operacional permanecem como camadas de governança**.
9. **Governance Smoke Test EXP-009E é gate obrigatório antes de qualquer promoção**.
10. **Data Intake Contract v1.1 é o contrato oficial para novas transações e labels**.

---

## 3. Princípios operacionais do Plano 3

### 3.1 Produtividade

Não produzir experimentos que matem produtividade.

Regras:

```text
- Preferir artifact-only, audit-only e model-only antes de qualquer E2E.
- E2E completo só quando houver candidato real.
- Todo script deve ter escopo claro, outputs pequenos de auditoria e modo fast.
- CSVs de auditoria devem ser limitados quando forem apenas para leitura/análise.
- Jobs Spark pesados devem ser quebrados por data, por chave ou por partição.
- Se um stage indicar explosão de tasks, interromper e redesenhar.
```

### 3.2 Segurança metodológica

```text
- Nenhum texto pós-evento da MAF pode entrar como feature.
- Nenhuma feature futura ao momento da transação pode entrar no treino.
- Labels MAF são pós-evento e servem para supervisão, não para inferência.
- Normais não rotulados precisam de janela de maturação.
- Avaliação oficial deve respeitar janela temporal coerente.
- Retreino só pode ser promovido após quick-E2E, regressão e governance smoke.
```

### 3.3 Estratégia de uso da MAF

As 13.558 fraudes MAF hidratadas não serão descartadas, mas serão usadas com separação:

```text
MAF_RECENTE_ALINHADA:
  fraudes dentro da janela dos normais coletáveis, 90/180 dias.
  Uso: treino/validação principal.

MAF_HISTORICA_FORA_JANELA:
  fraudes fora da janela de normais.
  Uso: treino auxiliar ponderado, análise de padrões, stress test e avaliação de recall,
  mas não como única base de validação oficial contra normais recentes.

MAF_RESERVA_TEMPORAL:
  subconjunto reservado para holdout temporal e avaliação de generalização.
```

### 3.4 Estratégia de normais

Não coletar 2022–2026 completo.

Limite aceito:

```text
janela normal mínima: 90 dias
janela normal máxima: 180 dias
```

Amostragem recomendada:

```text
- estratificada por dia;
- estratificada por faixa de valor;
- estratificada por tipo de chave;
- estratificada por horário;
- incluir oversampling de zonas de risco:
  - valores altos;
  - first_receiver;
  - chave aleatória;
  - relacionamento curto;
  - IF/SE/BEH potencialmente altos;
  - comportamento próximo dos FPs conhecidos.
```

---

## 4. Ordem oficial de execução

## FASE 0 — Gates herdados e estabilização permanente

**Status:** parcialmente concluída nos planos anteriores.  
**Objetivo:** manter os artefatos de governança como pré-requisito de qualquer nova mudança.

### Gates obrigatórios

Antes de promover qualquer alteração:

```text
python -m py_compile backend/core/decision_engine.py
python -m py_compile backend/core/pipeline_orquestrador.py
python -m py_compile backend/scripts/simular_pipeline_e2e_v2.py
python -m pytest tests/test_regression_post_fase2.py -q
python -m pytest tests/test_regression_post_fase2.py -q -m slow
python scripts/governance_smoke_test.py
```

### Artefatos oficiais mantidos

```text
docs/VALIDATION_REPORT_POST_FASE2.md
docs/RULES_CATALOG.md
docs/DECISION_TRACE_SPEC.md
backend/artefatos/MANIFEST_MODEL.json
resultados/experimentos/EXPERIMENT_INDEX.md
docs/JOURNAL.md
```

### Critério de aceite

Nenhum experimento do Plano 3 pode promover artefato se a regressão pós-C1 ou o governance smoke estiver vermelho.

---

# FASE 1 — EXP-010D: MAF Hydrated Fraud Compatibility Audit

**Objetivo:** validar localmente o CSV `dados_pix_fraudes_maf_hidratadas_v1.csv` antes de qualquer treino.

### Perguntas que o EXP-010D deve responder

1. O CSV novo é compatível com `preprocessing.py`?
2. `cd_pix` vira `transaction_id` sem perda?
3. `cd_cpf_pagador` vira `customer_id` sem inconsistência?
4. `dt_pix` vira `event_datetime` corretamente?
5. Há duplicados?
6. Há overlap com as 355 fraudes antigas?
7. O período temporal está coerente?
8. Quantas fraudes caem dentro de 90/180 dias?
9. Quais colunas críticas estão ausentes?
10. Qual o impacto de `ENABLE_MOBILE=False` no CSV atual?

### Saídas esperadas

```text
resultados/experimentos/EXP-010D/
├── 00_run_summary.json
├── 01_schema_compatibility.csv
├── 02_temporal_distribution.csv
├── 03_duplicate_audit.csv
├── 04_overlap_legacy_frauds.csv
├── 05_contract_v1_1_check.csv
├── 06_missing_critical_features.csv
├── 07_preprocessing_dry_run_report.md
└── 08_recommendation.md
```

### Critério de aceite

```text
- cd_pix único ou deduplicável;
- is_fraud=1 em 100%;
- nenhuma coluna textual MAF em features;
- preprocessing dry-run sem erro;
- contrato v1.1 atendido ou lacunas explicitadas;
- recomendação clara para EXP-010E/010F.
```

---

# FASE 2 — EXP-010E: MBK Keyed Hydration Audit

**Objetivo:** enriquecer as fraudes MAF com MBK sem varredura massiva e sem OOM.

### Estratégia

O erro a evitar é ler MBK inteira. A estratégia correta é keyed hydration:

```text
1. Criar tabela pequena de chaves alvo:
   transaction_id, dt_pix, cd_cpf_pagador, cd_cpf_cnpj_recebedor.

2. Dividir por data:
   rodar por dia, semana ou mês, conforme particionamento real da MBK.

3. Filtrar MBK por autdatref:
   usar apenas datas necessárias.

4. Filtrar por tags PIX antes de regex pesada:
   auttrn LIKE '%IdFimAfim%'
   ou tags equivalentes usadas no MBK.

5. Extrair E2E ID após reduzir o universo.

6. Fazer left-semi join com chaves alvo.

7. Deduplicar por transaction_id priorizando completude.

8. Salvar tabela reutilizável.
```

### Tabela esperada

```text
hmo_ml.tb_pix_maf_mbk_hydrated_v1
```

### Campos críticos

```text
transaction_id
device_name
app_version
ip_address
latencia_rede_ms
tempo_interacao_ms
tempo_processamento_host_ms
metodo_autenticacao
session_id
topaz_risk_score
topaz_transacao_rejeitada
is_agendamento_recorrente
data_referencia
data_hora_inicio
coverage_score
```

### Saídas esperadas

```text
Artefatos/EXP-010E/
├── 00_run_summary.json
├── 01_monthly_key_counts.csv
├── 02_mbk_coverage_by_month.csv
├── 03_mbk_field_coverage.csv
├── 04_unmatched_keys_sample.csv
├── 05_matched_keys_sample.csv
├── 06_dedup_strategy_report.md
└── 07_recommendation.md
```

### Critério de aceite

```text
- job roda sem stage explosivo;
- cobertura MBK medida por mês;
- campos críticos têm cobertura conhecida;
- tabela intermediária deduplicada por transaction_id;
- se cobertura for baixa, documentar se é limitação da fonte ou do filtro.
```

---

# FASE 3 — EXP-010F-R2: Normal Sampling Qualificado via Hue/Sqoop/Hive

**Objetivo:** construir uma base de transações normais menor, qualificada e estratificada para treino do dataset v2, evitando backfill exaustivo e custoso no CML.

## Mudança de estratégia

O EXP-010F-R1 mostrou que a captura diária de normais via CML/Spark é operacionalmente cara demais para o objetivo de treinamento. Apesar de ter gerado aproximadamente 42.000 normais aproveitáveis em 14 dias, a abordagem não deve continuar como principal estratégia de coleta.

A partir do EXP-010F-R2, a coleta de normais será feita via workflow Hue/Sqoop/Hive, com amostragem cedo no SQL e criação de tabelas intermediárias auditáveis.

## Estratégia

A amostra normal deve ser qualificada, não exaustiva.

A coleta deve combinar:

1. Normais parciais já coletados no EXP-010F-R1
   - aproximadamente 42.000 linhas;
   - preservar origem como EXP010F_R1_PARTIAL;
   - usar como parte da base normal.

2. Background normals
   - amostra aleatória leve;
   - distribuída na janela recente;
   - usada para representar comportamento operacional comum.

3. Matched controls
   - normais pareados com fraudes MAF;
   - por faixa de valor;
   - por horário/período do dia;
   - por tipo de chave;
   - por proximidade temporal;
   - objetivo: criar negativos comparáveis aos positivos.

4. Hard negatives
   - transações normais com aparência suspeita;
   - alto valor;
   - chave aleatória;
   - madrugada;
   - recebedor novo ou raro;
   - relacionamento curto quando disponível;
   - padrões próximos aos FPs/FNs conhecidos.

5. MBK-covered normals
   - normais com match no compact index MBK;
   - usados para treinar melhor sinais mobile/device/IP/topaz;
   - manter flag de missingness quando MBK não existir.

Janela temporal

A janela de referência continua sendo até 180 dias, alinhada ao EXP-010E e à janela aceita para normais recentes.

janela normal mínima: 90 dias
janela normal máxima: 180 dias

Porém, não é obrigatório coletar volume uniforme em todos os dias. A coleta deve priorizar representatividade e cobertura dos estratos.

Volumes alvo revisados
normais já disponíveis do EXP-010F-R1:
  aproximadamente 42.000

meta adicional inicial:
  40.000 a 80.000 normais qualificados

meta total recomendada para dataset v2:
  80.000 a 120.000 normais

A meta anterior de 200.000 a 800.000 normais deixa de ser obrigatória. Ela poderá ser reconsiderada somente se a coleta via Hue/Sqoop/Hive se mostrar muito barata e estável.

Critérios de exclusão
- excluir cd_pix presente em hmo_ml.tb_pix_fraudes_maf_hidratadas_v1;
- excluir labels positivos curados da MAF;
- excluir transações com retorno rejeitado, quando aplicável;
- excluir origem igual a destino;
- manter janela de maturação para tratar ausência de fraude conhecida como normalidade;
- não usar texto MAF ou qualquer campo pós-evento como feature.

Tabelas esperadas

Os nomes finais serão ajustados após revisão dos scripts genéricos de workflow Hue/Sqoop, mas a estrutura conceitual será:

hmo_ml.tb_pix_normais_qualified_candidates_180d_v1
hmo_ml.tb_pix_normais_qualified_sample_180d_v1
hmo_ml.tb_pix_normais_qualified_sample_mbk_180d_v1
hmo_ml.tb_pix_normais_dataset_ready_v1
Campos mínimos esperados
cd_pix
cd_cpf_pagador
cd_cpf_cnpj_recebedor
vl_pix
dt_pix
data_pix
ds_tipo_chave
periodo_dia
value_band
normal_sample_source
normal_sample_strategy
normal_sample_weight
matched_control_flag
hard_negative_flag
mbk_available_flag
is_fraud=0

Campos MBK, quando disponíveis:

device_name
app_version
ip_address
latencia_rede_ms
tempo_interacao_ms
tempo_processamento_host_ms
metodo_autenticacao
session_id
topaz_risk_score
topaz_transacao_rejeitada
is_agendamento_recorrente
mbk_completeness_score

Saídas esperadas
Artefatos/EXP-010F-R2/
├── 00_run_summary.json
├── 01_sampling_strategy_counts.csv
├── 02_value_band_distribution.csv
├── 03_temporal_distribution.csv
├── 04_key_type_distribution.csv
├── 05_mbk_coverage_report.csv
├── 06_overlap_with_maf_audit.csv
├── 07_sample_preview.csv
└── 08_recommendation.md

Critério de aceite
- amostra normal final possui volume suficiente para treino vNext;
- positivos MAF foram excluídos da base normal;
- distribuição por valor, horário e tipo de chave é auditável;
- há hard negatives suficientes;
- há matched controls suficientes;
- cobertura MBK é conhecida e explicitamente marcada;
- ausência de MBK é representada por flags de missingness;
- tabelas são geradas por workflow reproduzível no Hue/Sqoop/Hive;
- EXP-010G consegue consumir a base sem ajustes manuais.

Decisão operacional

O EXP-010F-R2 substitui a estratégia de backfill exaustivo via CML. O CML poderá ser usado para auditoria, validação e treinamento, mas não deve ser o mecanismo principal de extração massiva de normais.

---

# FASE 4 — EXP-010G: Unified Dataset Builder v2

**Objetivo:** construir dataset unificado para treino e avaliação.

### Entradas


- fraudes MAF hidratadas;
- MBK hidratada por transaction_id;
- normais parciais aproveitáveis do EXP-010F-R1;
- normais qualificados/estratificados do EXP-010F-R2;
- fraudes antigas do baseline;
- labels separados conforme contrato v1.1.


### Regras de composição


1. Separar MAF recente de MAF histórica.
2. Não misturar textos MAF como features.
3. Deduplicar por transaction_id.
4. Garantir event_datetime confiável.
5. Recalcular features leakage-free.
6. Criar folds temporais.
7. Criar holdout temporal final.
8. Registrar manifesto do dataset.
9. Preservar origem e estratégia de amostragem dos normais.
10. Criar sample_weight para diferenciar background normals, matched controls e hard negatives.


### Saídas esperadas


dados/processed/base_treino_pix_v2.parquet
dados/processed/base_treino_pix_v2_sample.csv
dados/manifests/dataset_pix_v2.json

resultados/experimentos/EXP-010G/
├── 00_run_summary.json
├── 01_dataset_profile.csv
├── 02_label_distribution.csv
├── 03_temporal_split_report.csv
├── 04_feature_compatibility.csv
├── 05_train_serve_parity_report.md
└── 06_recommendation.md


### Critério de aceite


- preprocessing gera dataset model-ready;
- todas as features do LGBM/IF/SE/BEH estão presentes ou com fallback explícito;
- MAF recente e histórica são identificáveis;
- splits temporais estão definidos;
- dataset tem manifesto.


---

# FASE 5 — EXP-011: Treino Shadow vNext

**Objetivo:** testar novos modelos sem comprometer o baseline oficial.

## EXP-011A — Baseline Replay no dataset v2

Rodar o baseline `post_fase2_c1` no dataset v2, sem retreino.

Perguntas:

```text
- baseline generaliza?
- C1/V1 continuam úteis?
- SE/BEH disparam de forma coerente?
- quais FNs/FPs surgem na base nova?
```

## EXP-011B — LGBM vNext com MAF recente

Treinar LGBM com:

```text
- normais 90/180 dias;
- MAF_RECENTE_ALINHADA;
- fraudes antigas como complemento;
- features MBK quando disponíveis;
- validação temporal.
```

## EXP-011C — LGBM vNext com MAF histórica ponderada

Testar uso da MAF histórica com peso controlado.

Possibilidades:

```text
- sample_weight menor para MAF histórica;
- treino com MAF histórica, avaliação oficial só em janela recente;
- ablation com e sem MAF histórica.
```

## EXP-011D — IF Recalibration vNext

Recalibrar Isolation Forest com dataset v2, mantendo papel complementar.

## EXP-011E — Engine Quick-E2E vNext

Só rodar depois que houver candidato real.

Regras:

```text
- E2E rápido;
- amostra estratificada;
- timeout e limite de workers;
- nunca repetir o E2E pesado de mais de uma hora sem necessidade.
```

### Critério de promoção

Nenhum modelo vNext será promovido se:

```text
- FN aumentar;
- FP explodir;
- regressão pós-C1 falhar;
- governance smoke falhar;
- seed/janela temporal não confirmar;
- explicabilidade SHAP for incoerente;
- train/serve parity estiver fraca.
```

---

# FASE 6 — EXP-012: Lightweight Graph Feature Engineering

**Objetivo:** testar se grafos leves agregam valor sem grande esforço e sem impacto no tempo real.

### Decisão de escopo

Não implementar GNN. Não implementar banco de grafo.

Implementar apenas Graph Feature Engineering em Spark SQL/batch.

### Features candidatas

```text
dest_in_degree_24h
dest_in_degree_7d
dest_total_recv_amt_24h
dest_tx_count_24h
dest_avg_recv_amt_24h
dest_new_sender_ratio_24h
orig_distinct_targets_24h
orig_total_sent_amt_24h
shared_dest_recent_flag
dest_prior_fraud_count_30d
dest_prior_fraud_flag
```

### Regra leakage-free

Para cada transação T:

```text
usar somente eventos com dt_pix < T.dt_pix
```

### Experimentos

```text
EXP-012A — Graph Features SQL/Spark Feasibility
EXP-012B — Dataset v2 + Graph Features
EXP-012C — LGBM vNext ablation: sem grafo vs com grafo
EXP-012D — decisão de promover, manter shadow ou descartar
```

### Critério de continuidade

Prosseguir com grafos apenas se:

```text
- F1 melhora;
- FN não aumenta;
- FP cai ou permanece controlado;
- features são baratas de materializar;
- não há leakage;
- explicabilidade é útil para conta-laranja/mula.
```

---

# FASE 7 — EXP-013: Feature Store HBase para Produção

**Objetivo:** viabilizar serving realista do modelo, já que a transação online entrega apenas parte das features.

### Racional

O modelo e os módulos SE/BEH dependem de features que não chegam diretamente na transação PIX. A Feature Store em HBase passa a ser o caminho para entregar features de perfil, histórico, recebedores, sessão/device e, futuramente, grafo em baixa latência.

### Fases

## EXP-013A — Feature Store Contract v1

Definir contrato de leitura/escrita das tabelas:

```text
fraud_detection:perfil_cliente
fraud_detection:historico_trimestral
fraud_detection:historico_recebedores
fraud_detection:sessao_device
fraud_detection:graph_features_by_dest
fraud_detection:graph_features_by_orig
```

## EXP-013B — HBase DDL via Hue Shell

Criar scripts idempotentes para:

```text
- criar namespace;
- criar tabelas;
- criar column families;
- configurar TTL;
- configurar compression;
- validar existência.
```

## EXP-013C — Spark Materialization Dry Run

Antes de escrever no HBase, materializar em Hive/Parquet:

```text
hmo_ml.fs_perfil_cliente_v1
hmo_ml.fs_historico_trimestral_v1
hmo_ml.fs_historico_recebedores_v1
hmo_ml.fs_sessao_device_v1
hmo_ml.fs_graph_features_v1
```

## EXP-013D — HBase Write/Read Smoke Test

Escrever amostra pequena e validar GET por CPF/recebedor.

## EXP-013E — API Adapter Shadow

Criar adaptador no orquestrador para buscar HBase em modo shadow, sem alterar decisão final.

### Critério de aceite

```text
- GET por CPF funcional;
- latência medida;
- fallback definido;
- freshness monitorada;
- train/serve parity testável;
- sem dependência de HBase para rodar batch offline.
```

---

# FASE 8 — EXP-014: Shadow Production Package e Governança vNext

**Objetivo:** preparar pacote de produção shadow antes de qualquer promoção.

### Componentes

```text
EXP-014A — Governance Smoke pós-vNext
EXP-014B — Decision Logging v2
EXP-014C — Drift + Feature Freshness Monitor
EXP-014D — Dashboard Operacional v2
EXP-014E — Runbook de Produção/Shadow
EXP-014F — Model Card vNext
```

### Itens obrigatórios

```text
- versionamento de dataset;
- versionamento de artefatos;
- MANIFEST_MODEL atualizado;
- JOURNAL atualizado;
- validação pós-C1;
- comparação baseline vs vNext;
- plano de rollback;
- métricas de latência;
- métricas de degradação HBase;
- alertas de freshness.
```

---

# FASE 9 — Decisão de Promoção

**Objetivo:** decidir se o baseline `post_fase2_c1` será mantido ou substituído.

### Possíveis decisões

```text
A. Manter baseline atual.
B. Promover novo dataset/modelo LGBM.
C. Promover apenas thresholds/regras recalibradas.
D. Promover Feature Store em shadow sem trocar modelo.
E. Promover grafos como features shadow.
F. Aguardar mais dados antes de nova promoção.
```

### Critério mínimo de promoção

```text
- FN não aumenta;
- FP não aumenta de forma operacionalmente inaceitável;
- F1 melhora ou há ganho claro de risco/custo;
- validação temporal confirma;
- quick-E2E confirma;
- regression suite passa;
- governance smoke passa;
- documentação e Journal atualizados;
- rollback definido.
```

---

## 5. Ordem imediata de trabalho

A ordem prática a partir de agora é:

```text
1. EXP-010D — MAF Hydrated Fraud Compatibility Audit local.
2. EXP-010E — MBK Keyed Hydration Audit no CML.
3. EXP-010F-R2 — Normal Sampling Qualificado via Hue/Sqoop/Hive.
  3.1. Revisar scripts genéricos de workflow Sqoop/Hue.
  3.2. Criar .hql de amostragem normal qualificada.
  3.3. Gerar tabelas intermediárias e finais de normais estratificados.
  3.4. Auditar distribuição e overlap com MAF antes do EXP-010G.
4. EXP-010G — Unified Dataset Builder v2.
5. EXP-011A — Replay do baseline no dataset v2.
6. EXP-011B/C/D — treino shadow LGBM/IF vNext.
7. EXP-011E — quick-E2E apenas se houver candidato real.
8. EXP-012A — grafos leves somente após dataset v2 existir.
9. EXP-013A/B — HBase contract e DDL podem rodar em paralelo.
10. EXP-014 — pacote shadow/governança antes de promoção.
```

---

## 6. Encerramento dos planos anteriores

A partir deste documento:

```text
plano_melhoria_critica_1.md = encerrado como plano ativo
plano_melhoria_crítica_2.md = encerrado como plano ativo
plano_melhoria_critica3.md = plano oficial vigente
```

Os planos anteriores continuam como histórico e fonte de racional técnico, mas a execução daqui em diante deve seguir exclusivamente este Plano 3.

---

## 7. Próximo passo concreto

O próximo passo recomendado é gerar e executar:

```text
EXP-010D — MAF Hydrated Fraud Compatibility Audit
```

Ele é local, rápido e desbloqueia as decisões sobre MBK, normais e dataset v2.
