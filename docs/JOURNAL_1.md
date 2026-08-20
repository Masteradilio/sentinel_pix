# Journal — Decisões Técnicas do Pipeline Antifraude PIX

Este journal registra decisões técnicas relevantes do projeto, em formato cronológico e append-only.

## Objetivo

- evitar que o `plano_melhoria_critica.md` fique excessivamente grande;
- preservar o racional de decisões importantes;
- registrar trade-offs, restrições e motivos de promoção/rejeição;
- facilitar auditoria futura;
- manter histórico de mudanças entre fases.

## Regra de uso

```text
Não apagar decisões antigas.
Se uma decisão mudar, adicionar uma nova entrada explicando a mudança.
```

---

<!-- decision_id: FASE2_BASELINE_POS_C1 -->

## FASE 2 encerrada com baseline pós-C1

**Data:** `2026-05-09T15:30:00`  
**ID:** `FASE2_BASELINE_POS_C1`

**Decisão:** encerrar a FASE 2 com sucesso mínimo validado.

Baseline oficial:

| Seed | TP | FP | FN | F1 |
|---:|---:|---:|---:|---:|
| 42 | 347 | 14 | 8 | 0,9693 |
| 123 | 347 | 12 | 8 | 0,9720 |

**Racional:** a C1 recuperou 1 FN nos dois seeds, adicionou 0 FP e não perdeu TP. O EXP-007A não encontrou candidato seguro adicional com os sinais atuais.

**Consequência:** novas reduções relevantes de FN provavelmente dependem de novos sinais/dados, não de novas regras sobre os mesmos sinais.

---

<!-- decision_id: EXP008A_REGRESSION_SUITE_POS_C1 -->

## EXP-008A aprovado — suíte de regressão pós-C1

**Data:** `2026-05-09T15:30:00`  
**ID:** `EXP008A_REGRESSION_SUITE_POS_C1`

**Decisão:** tornar a regressão pós-C1 obrigatória antes de qualquer mudança futura.

Validação conhecida:

```text
pytest normal: 6 passed
pytest slow: 1 passed, 5 deselected
```

**Racional:** a regressão protege a C1, o baseline pós-FASE 2, o scoring_config e a validação runtime da transação alvo.

---

<!-- decision_id: EXP008B_VALIDATION_REPORT -->

## EXP-008B aprovado — Validation Report Pós-FASE 2

**Data:** `2026-05-09T15:30:00`  
**ID:** `EXP008B_VALIDATION_REPORT`

**Decisão:** aceitar `docs/VALIDATION_REPORT_POST_FASE2.md` como relatório oficial da versão pós-FASE 2.

**Racional:** o relatório consolida baseline pós-C1, métricas oficiais, deltas da C1, FNs residuais, decisões promovidas/rejeitadas e comandos obrigatórios de regressão.

---

<!-- decision_id: EXP008C_RULES_TRACE -->

## EXP-008C aprovado — Rules Catalog e Decision Trace

**Data:** `2026-05-09T15:30:00`  
**ID:** `EXP008C_RULES_TRACE`

**Decisão:** aceitar `docs/RULES_CATALOG.md`, `docs/DECISION_TRACE_SPEC.md` e `docs/DECISION_TRACE_EXAMPLE.json`.

**Racional:** os artefatos documentam regras ativas, regras rejeitadas, thresholds, guard rails, campos mínimos de rastreabilidade e exemplo de decisão C1.

---

<!-- decision_id: EXP008D_CLEANUP_RESTRICAO -->

## EXP-008D aprovado com restrição — cleanup técnico dos patches

**Data:** `2026-05-09T15:30:00`  
**ID:** `EXP008D_CLEANUP_RESTRICAO`

**Decisão:** aprovar o cleanup técnico com restrição.

**Removido:**
- resíduos órfãos do wrapper runtime antigo no `decision_engine.py`.

**Reposto:**
- binding defensivo `_hydrate_config_from_scoring_config`, pois o runtime do `PixDecisionEngine` ainda depende dele na inicialização.

**Mantido:**
- wrapper efetivo em `backend/scripts/simular_pipeline_e2e_v2.py`;
- wrapper redundante em `backend/core/pipeline_orquestrador.py`, por segurança;
- campos C1 no `EngineConfig`;
- configuração C1 no `scoring_config.json`.

**Racional:** a tentativa de remoção automática do wrapper do `pipeline_orquestrador.py` apresentou risco de quebra. A prioridade da FASE 3 é estabilidade e regressão verde, não limpeza estética.

Validação final:

```text
py_compile decision_engine.py: OK
py_compile pipeline_orquestrador.py: OK
py_compile simular_pipeline_e2e_v2.py: OK
hydrate OK True
C1 field OK True
pytest normal: 6 passed
pytest slow: 1 passed, 5 deselected
```

---

<!-- decision_id: EXP008E_JOURNAL_CRIADO -->

## EXP-008E — criação do Journal de decisões

**Data:** `2026-05-09T15:30:00`  
**ID:** `EXP008E_JOURNAL_CRIADO`

**Decisão:** criar `docs/JOURNAL.md` como registro cronológico e append-only de decisões técnicas importantes.

**Racional:** o `plano_melhoria_critica.md` deve permanecer estratégico. Decisões detalhadas, restrições, trade-offs e resultados de validação devem ser registrados no journal para evitar que o plano fique excessivamente grande.

**Uso esperado:** toda decisão promovida, rejeitada ou aprovada com restrição deve ganhar uma entrada no journal.

---

<!-- decision_id: EXP009A_DECISION_LOGGING_APROVADO -->

## EXP-009A aprovado — Decision Logging Estruturado

**Decisão:** aprovar o decision logging estruturado do baseline `post_fase2_c1`.

O experimento gerou logs estruturados para 12.000 decisões, cobrindo os seeds 42 e 123, com validação de schema bem-sucedida:

```text
ok=true
missing_fields=[]
duplicate_decision_ids=0
invalid_decision_values=[]

---

<!-- decision_id: EXP009B_DRIFT_MONITOR_APROVADO -->

## EXP-009B aprovado — Drift Monitor Offline

**Decisão:** aprovar o monitor offline de drift do baseline `post_fase2_c1`.

O experimento comparou os logs estruturados do EXP-009A entre `seed_42_reference` e `seed_123_current`, com 6.000 decisões em cada base.

Resultado:

```text
status=OK
n_alerts=0
n_warnings=0
schema_reference.ok=true
schema_current.ok=true

---

### EXP-009C — Zona Cinza e Fila de Revisão Humana

**Status:** Aprovado com observação.

O experimento criou uma fila offline de revisão humana a partir dos logs estruturados do EXP-009A, reduzindo 11.280 decisões `APROVAR` para 749 linhas de fila e 730 transações deduplicadas.

A fila capturou 2 fraudes conhecidas no total com seeds e 1 fraude conhecida deduplicada, equivalente a 12,5% dos FNs conhecidos em `APROVAR`. A fraude capturada apareceu apenas no top 200 deduplicado, não no top 25, 50 ou 100.

**Decisão:** a fila é válida para auditoria humana, active learning, coleta de evidências e monitoramento operacional. Ela não deve ser usada para promoção automática de regra, pois os casos de maior prioridade ficaram dominados por sinais de IF extremo, alto valor e V1 quase acionada, mas não concentraram fraudes conhecidas.

**Conclusão:** EXP-009C entrega uma camada útil de operação e observabilidade, mas não resolve os FNs residuais. Os FNs restantes continuam próximos do limite dos sinais atuais.

---

<!-- decision_id: EXP009D_OPERATIONAL_DASHBOARD_APROVADO -->

## EXP-009D aprovado — Painel de Métricas Operacionais

**Decisão:** aprovar o EXP-009D como camada offline de métricas operacionais e bases para Power BI do baseline `post_fase2_c1`.

O experimento gerou bases consolidadas a partir dos logs estruturados do EXP-009A e da fila de revisão humana do EXP-009C, incluindo:

- KPIs gerais;
- distribuição de decisões;
- métricas por regra aplicada;
- faixas de `score_final`, `lgbm_raw` e `if_percentile`;
- métricas diárias;
- métricas da fila de revisão;
- fato principal para Power BI;
- fato da fila de revisão para Power BI.

Os KPIs principais ficaram consistentes com o baseline combinado pós-C1:

```text
decisions=12000
unique_transactions=11330
fraudes_conhecidas=710
APROVAR=11280
CONFIRMAR=131
BLOQUEAR=589
C1_aplicada=2
Precision=0.963889
Recall=0.977465
F1=0.970629
FPR=0.002303

Durante a validação, foi identificado e corrigido um problema na extração de datas a partir do transaction_id. A extração passou a considerar corretamente o padrão do E2E ID PIX:

E + ISPB(8 dígitos) + YYYYMMDD + restante

Após a correção, a validação de datas passou:

decision_fact rows: 12000
transaction_date vazias: 0
transaction_date strings vazias: 0
anos decision_fact: [2025, 2026]

daily rows: 170
anos daily: [2025, 2026]

review_queue rows: 730
review_queue transaction_date vazias: 0
review_queue transaction_date strings vazias: 0
anos review_queue: [2025, 2026]

STATUS: OK

Racional: o projeto passa a ter uma camada operacional de BI offline, permitindo acompanhar volume de decisões, métricas de performance, acionamento de regras, taxa da C1, distribuição de scores e fila de revisão humana sem alterar o modelo ou o motor de decisão.

Uso esperado: alimentar Power BI ou dashboard offline para acompanhamento operacional, auditoria, governança e comunicação executiva.

Próximo passo: EXP-009E — Smoke Test de Reprodutibilidade e Pacote de Governança.

---

<!-- decision_id: EXP009E_GOVERNANCE_SMOKE_TEST_APROVADO -->

## EXP-009E aprovado — Governance Smoke Test

**Decisão:** aprovar o pacote de governança mínima do baseline `post_fase2_c1`.

Após instalar `pytest` no ambiente `.venv` e atualizar os hashes do `MANIFEST_MODEL.json`, o smoke test de governança passou sem falhas e sem warnings.

Resultado final:

```text
overall_status=PASS
n_total_checks=58
n_pass=58
n_warn=0
n_fail=0

Todos os grupos de validação passaram:

commands=PASS
config=PASS
critical_code=PASS
docs=PASS
exp009a=PASS
exp009b=PASS
exp009c=PASS
exp009d=PASS
hash=PASS
journal=PASS
manifest=PASS
scoring_config=PASS
tests=PASS

Os comandos críticos também passaram:

py_compile decision_engine.py: PASS
py_compile pipeline_orquestrador.py: PASS
py_compile simular_pipeline_e2e_v2.py: PASS
hydrate OK True
C1 field OK True
pytest normal: 6 passed
pytest slow: 1 passed, 5 deselected

Racional: o projeto agora possui um portão único de qualidade para validar código crítico, regressão pós-C1, artefatos oficiais, hashes do Manifest, logs estruturados, drift monitor, fila de revisão humana, dashboard operacional e journal antes de qualquer nova rodada experimental.

Decisão operacional: antes de qualquer nova mudança relevante no modelo, regras, scoring_config, engine, artefatos ou documentação oficial, executar o EXP-009E como smoke test de governança.

Próximo passo: EXP-010A — Data Intake Contract e Harness de Reavaliação com Novos Dados.


## Situação atual do projeto

Com isso, a parte de governança e observabilidade ficou muito sólida:

```text
EXP-008A: regressão pós-C1 aprovada
EXP-008B: validation report aprovado
EXP-008C: rules catalog e decision trace aprovados
EXP-008D: cleanup técnico aprovado com restrição
EXP-008E: manifest, index e journal aprovados
EXP-009A: decision logging estruturado aprovado
EXP-009B: drift monitor offline aprovado
EXP-009C: fila de revisão humana aprovada com observação
EXP-009D: painel operacional aprovado após correção de datas
EXP-009E: governance smoke test aprovado

---

<!-- decision_id: EXP010A_DATA_INTAKE_CONTRACT_APROVADO -->

## EXP-010A aprovado — Data Intake Contract e Harness de Reavaliação

**Decisão:** aprovar o EXP-010A como preparação para reavaliação futura do baseline `post_fase2_c1` com novos dados.

O experimento gerou contrato de entrada de transações, contrato de labels, perfil da base de referência, relatório de compatibilidade de features e especificação do harness de reavaliação.

A base de referência usada foi carregada via `experimentos.utils_experimentos.load_dataset()`, com 100.355 linhas e 70 colunas. A validação de novos dados ficou como `NOT_RUN`, pois ainda não havia novo arquivo de transações/labels nesta rodada.

Artefatos principais:

- `01_transaction_schema_contract.csv`
- `02_transaction_schema_contract.json`
- `03_label_schema_contract.json`
- `04_reference_data_profile.csv`
- `05_feature_compatibility_report.csv`
- `06_DATA_INTAKE_CONTRACT.md`
- `07_REVALUATION_HARNESS_SPEC.md`
- `10_next_experiment_spec.md`

**Observação:** antes de usar dados reais do Big Data, o contrato deve separar dois modos:
1. modo scoring/inferência, no qual `is_fraud` não é obrigatório no arquivo de transações;
2. modo avaliação supervisionada, no qual `is_fraud` deve vir em tabela de labels ligada por `transaction_id`.

Também é recomendado tornar `event_datetime` ou `dt_transacao` obrigatório operacionalmente para extrações incrementais, drift, janelas temporais e painéis.

**Próximo passo:** discutir estratégia de extração de novos dados de fraude e transações normais diretamente do ambiente Big Data da empresa.

---

<!-- decision_id: EXP010A_R1_DATA_INTAKE_CONTRACT_V1_1_APROVADO -->

## EXP-010A-R1 aprovado — Data Intake Contract v1.1

**Decisão:** aprovar o ajuste v1.1 do contrato de entrada de dados e do harness de reavaliação.

O ajuste separou explicitamente dois modos de uso:

1. **scoring/inferência:** novas transações podem ser pontuadas sem `is_fraud`;
2. **avaliação supervisionada:** métricas de performance exigem labels separados ligados por `transaction_id`.

Principais mudanças:

```text
is_fraud deixou de ser obrigatório no arquivo de transações em modo scoring/inferência.
labels supervisionados passaram a ter contrato separado.
event_datetime ou dt_transacao virou requisito operacional obrigatório.
harness separou dry run sem labels de reavaliação supervisionada com labels.
foi criado checklist de preparação para extração no Big Data.

O contrato v1.1 exige, para transações em modo scoring:

transaction_id
customer_id
vl_pix
qt_tempo_relacionamento_mes
first_receiver_flag
pix_key_random_flag
event_datetime ou dt_transacao

Para avaliação supervisionada, os labels devem conter:

transaction_id
is_fraud

O contrato de labels também passou a diferenciar confirmed_fraud_only de full_supervised_window, evitando tratar ausência de fraude confirmada como normalidade automática sem janela de maturação.

Observação técnica: o JSON do contrato preserva campos legados v1.0, como required e contract_role. Validadores futuros devem usar os campos v1.1 (required_for_scoring, required_one_of_group, required_for_supervised_evaluation e label_column) para evitar exigir is_fraud no arquivo de transações.

Racional: o projeto agora está preparado para discutir aquisição de novos dados diretamente no Big Data sem misturar transações, labels, features pós-evento e risco de leakage.

Próximo passo: definir estratégia de extração de novos dados de fraude e transações normais no ambiente Big Data.

---

<!-- decision_id: EXP010B_MAF_FRAUD_LABEL_ACQUISITION_STRATEGY -->

## EXP-010B definido — MAF Fraud Label Acquisition Audit

**Decisão:** usar a nova tabela textual do departamento de fraudes como fonte de curadoria de labels, não como fonte direta de features do modelo.

A nova fonte possui relatos, textos de análise, conclusões operacionais e o identificador único da transação PIX/E2E ID. Como esses textos são produzidos após o evento transacional, eles não devem entrar como variáveis preditivas do modelo para evitar leakage.

A estratégia aprovada é criar uma trilha em duas etapas:

```text
1. Curadoria de labels:
   nova tabela textual MAF
   → transaction_id / E2E ID
   → label_status
   → label_confidence
   → fraud_type
   → bank_direction

2. Hidratação transacional:
   transaction_id confirmado
   → join nas tabelas PIX/mobile/cliente já usadas pelo pipeline atual
   → geração das features compatíveis com o modelo

O experimento inicial será o EXP-010B — MAF Fraud Label Acquisition Audit, com objetivo de auditar a nova fonte antes de criar qualquer tabela intermediária ou final.

Critérios da auditoria:

identificar transaction_id/E2E ID válido;
separar casos em que o BRB é pagador/debitado dos casos em que o BRB é recebedor/creditado;
classificar conclusões como CONFIRMED_FRAUD_CANDIDATE, NOT_FRAUD_OR_REJECTED, REVIEW_REQUIRED ou REVIEW_CONFLICT;
isolar triangulação e casos ambíguos;
medir cobertura de join com extrato PIX, mobile MBK e cadastro de cliente;
limitar artefatos CSV a 1000 linhas para análise rápida;
não promover mudança de modelo nesta etapa.

Racional: a nova fonte pode aumentar significativamente o volume de fraudes confirmadas, mas precisa ser tratada como fonte de label pós-evento. As features continuarão vindo das tabelas transacionais já usadas pelo pipeline leakage-free.

Próximo passo: executar o EXP-010B no CML e avaliar os artefatos antes de gerar o script definitivo de gestão da tabela intermediária de labels e da tabela final de fraudes hidratadas.

---

<!-- decision_id: EXP010C_MAF_CURATED_FRAUD_TABLES_APROVADO -->

## EXP-010C aprovado — MAF Curated Fraud Tables

**Decisão:** aprovar a criação das tabelas curadas e hidratadas derivadas da fonte textual MAF.

O experimento criou com sucesso:

```text
hmo_ml.tb_pix_fraude_labels_maf_curated_v1
hmo_ml.tb_pix_fraudes_maf_hidratadas_v1

Resultado executivo:

source_rows=135262
labels_curated=134599
positive_for_current_model=15564
triangulation_segregated=20242
receiver_scope_segregated=38077
review_or_conflict=22519
fraud_keys_for_hydration=15564
pix_hydrated_base=13558
hydrated_final=13558

A tabela final hidratada contém 13.558 fraudes fortes, sem duplicidade de cd_pix, todas com:

is_fraud=1
model_scope_status=POSITIVE_FOR_CURRENT_MODEL
label_status=CONFIRMED_FRAUD_CANDIDATE
bank_direction=BRB_DEBITADO_PAGADOR
triangulation_flag=false
duplicate_conflict_flag=false

A curadoria preservou a decisão estratégica definida no EXP-010B: textos pós-evento da MAF foram usados apenas para curadoria de labels e auditoria, não como features do modelo. Casos BRB_CREDITADO_RECEBEDOR, triangulação e conflitos ficaram segregados.

Observação: o mobile ficou desabilitado nesta execução para evitar varredura pesada da MBK. Portanto, campos mobile/topaz foram mantidos nulos e devem ser tratados em etapa posterior, se necessário.

Decisão operacional: não treinar modelo ainda diretamente com o novo CSV. Antes, executar uma etapa local de compatibilidade e validação com o preprocessing.py.

Próximo passo: EXP-010D — MAF Hydrated Fraud Compatibility Audit.

---

<!-- decision_id: PLANO_MELHORIA_CRITICA3_MUDANCA_DE_RUMO -->

## Plano de Melhoria Crítica 3 aprovado — mudança de rumo pós-MAF/HBase/MBK

**Data:** `2026-05-13T00:00:00-03:00`  
**ID:** `PLANO_MELHORIA_CRITICA3_MUDANCA_DE_RUMO`

**Decisão:** encerrar `plano_melhoria_critica_1.md` e `plano_melhoria_crítica_2.md` como planos ativos e adotar `plano_melhoria_critica3.md` como plano oficial vigente para viabilização do modelo antifraude PIX.

A mudança de rumo foi motivada por quatro fatos técnicos:

1. o baseline `post_fase2_c1` já está estabilizado, governado e protegido por regressão;
2. novas reduções relevantes de FN usando apenas os sinais antigos se mostraram limitadas;
3. a fonte MAF adicionou 13.558 fraudes fortes hidratadas, mudando o patamar de dados positivos disponíveis;
4. a evolução segura agora depende de hidratação MBK, amostragem controlada de normais, dataset v2, grafos leves e Feature Store HBase.

**Planos encerrados como ativos:**

```text
plano_melhoria_critica_1.md
plano_melhoria_crítica_2.md
```

Esses documentos permanecem como histórico e fonte de racional técnico, mas não devem mais orientar a ordem de execução.

**Plano oficial vigente:**

```text
plano_melhoria_critica3.md
```

**Diretrizes preservadas dos planos anteriores:**

- manter baseline `post_fase2_c1` como referência oficial;
- manter V1 Guard Contextual e C1 Near-Threshold como regras ativas;
- manter LGBM v6.2 rejeitado para runtime;
- manter meta-learner shadow como diagnóstico, sem promoção;
- não promover regras hardcoded generalistas baseadas em `first_receiver_flag`;
- executar regressão pós-C1 e governance smoke antes de qualquer promoção;
- manter decision logging, drift monitor, fila de revisão, dashboard operacional, Manifest e Journal como gates de governança;
- preservar Data Intake Contract v1.1 como contrato oficial de novas transações e labels.

**Nova ordem de execução aprovada:**

```text
1. EXP-010D — MAF Hydrated Fraud Compatibility Audit
2. EXP-010E — MBK Keyed Hydration Audit
3. EXP-010F — Normal Sampling v2, limitado a 90/180 dias
4. EXP-010G — Unified Dataset Builder v2
5. EXP-011 — treino shadow vNext com dataset v2
6. EXP-012 — Lightweight Graph Feature Engineering
7. EXP-013 — Feature Store HBase
8. EXP-014 — Shadow Production Package e Governança vNext
9. Decisão de promoção, manutenção ou rollback
```

**Racional:** as 13.558 fraudes MAF são valiosas demais para serem descartadas, mas não podem ser usadas de forma ingênua contra normais recentes. A nova estratégia separa MAF recente, MAF histórica, normais amostrados em 90/180 dias e validação temporal. Além disso, reconhece que MBK é indispensável para preservar a utilidade dos módulos `social_engineering` e `behavioral_analytics`, e que HBase é o caminho natural para reduzir train/serve skew em produção.

**Decisão operacional:** nenhum novo treino ou promoção de modelo deve ocorrer antes de concluir pelo menos EXP-010D, EXP-010E, EXP-010F e EXP-010G.

**Próximo passo:** gerar e executar o EXP-010D local para validar a compatibilidade do CSV `dados_pix_fraudes_maf_hidratadas_v1.csv` com o `preprocessing.py` e com o contrato v1.1.

---

<!-- decision_id: EXP010D_MAF_HYDRATED_COMPATIBILITY_APROVADO -->

## EXP-010D aprovado com warning esperado — MAF Hydrated Fraud Compatibility Audit

**Decisão:** aprovar o EXP-010D como validação local da base `dados_pix_fraudes_maf_hidratadas_v1.csv`.

Resultado executivo:

```text
status=WARN
n_rows=13558
n_columns=49
n_cd_pix_unique=13558
n_checks=15
n_pass=14
n_warn=1
n_fail=0

A base passou nos checks críticos de identidade, label, escopo, direção, datas e valores:

cd_pix único
is_fraud=1
model_scope_status=POSITIVE_FOR_CURRENT_MODEL
label_status=CONFIRMED_FRAUD_CANDIDATE
bank_direction=BRB_DEBITADO_PAGADOR
triangulation_flag=false
duplicate_conflict_flag=false
dt_pix parseável
vl_pix numérico e positivo

O único warning foi a ausência de cobertura MBK, já esperada porque o EXP-010C foi executado com ENABLE_MOBILE=False para evitar varredura massiva da tabela MBK.

A base cobre transações de 2022-05-13 a 2026-05-11, com valor mediano de R$ 1.170,25, p95 de R$ 12.903,45 e valor máximo de R$ 275.000,00.

Observação: o overlap com fraudes antigas não foi calculado porque o CSV antigo possui nomes de colunas prefixados, como tb_pix_anomalia_fraudes_trim_poc_v2.cd_pix. Isso não bloqueia a evolução.

Decisão operacional: a base MAF hidratada é válida como nova fonte positiva forte, mas ainda não deve ser usada diretamente para treino. Antes, deve passar por hidratação MBK por chave e posterior montagem de dataset unificado com normais amostrados.

Próximo passo: EXP-010E — MBK Keyed Hydration Audit.

---

<!-- decision_id: EXP010E_PRODUCAO_INCREMENTAL_FEATURE_STORE_TREINO_CONTINUO -->

## EXP-010E — Decisão de arquitetura para produção incremental, Feature Store e retreino contínuo

**Decisão:** adotar uma arquitetura incremental para viabilizar o modelo de anomalia e fraude PIX em produção, combinando atualização diária de dados, descoberta contínua de novos casos de fraude, Feature Store e retreino semanal automatizado.

A partir dos resultados do EXP-010E-R2B, foi confirmado que o processamento retroativo da MBK é viável quando feito por partição diária, usando uma tabela compacta intermediária. Na rodada de 15 dias, o processo concluiu em 89,91 minutos, processou 15 partições diárias, gerou 8.779.546 registros compactos MBK e atingiu cobertura de 85,71% nos dias efetivamente processados:

```text
target_keys_total_180d=1594
n_days_processed=15
target_keys_exact_processed_target_dates=217
matched_mbk_exact_processed_target_dates=186
coverage_exact_target_dates_processed=0.857143
matched_mbk_total_180d=261
coverage_total_180d_parcial=0.163739
elapsed_min=89.91

A métrica coverage_total_180d ainda é parcial, pois o compact index não cobria toda a janela no momento da análise. A métrica correta para avaliar a viabilidade operacional da estratégia é coverage_exact_target_dates_processed.

Decisão operacional: após o backfill inicial de 180 dias, não reprocessar a janela inteira diariamente. O desenho correto será:

1. Backfill inicial:
   - carregar os 180 dias históricos uma única vez;
   - criar/atualizar o compact index MBK particionado por autdatref;
   - hidratar fraudes MAF e bases de treino com os dados disponíveis.

2. Rotina diária:
   - processar apenas o novo dia fechado da MBK, preferencialmente D-1;
   - opcionalmente reprocessar D-2 e D-3 para capturar atraso de carga;
   - inserir/atualizar a partição diária do compact index;
   - remover ou expirar partições fora da janela móvel de 180 dias;
   - atualizar features agregadas rolling;
   - publicar snapshot na Feature Store.

3. Descoberta diária de novas fraudes:
   - consultar as fontes de fraude já mapeadas, incluindo a fonte textual MAF;
   - curar novos labels confirmados;
   - deduplicar por transaction_id/E2E ID;
   - separar positivos fortes, triangulação, BRB recebedor e casos de revisão/conflito;
   - inserir os novos casos em tabela curada de labels;
   - hidratar os novos positivos com PIX, cliente e MBK quando disponível.

4. Feature Store:
   - usar Hive/Parquet como offline store para compact index e datasets históricos;
   - usar HBase como serving store para features online;
   - manter rowkeys e famílias de features para cliente, cliente-recebedor, recebedor, device/IP e reputação;
   - garantir paridade treino/serving.

5. Retreino semanal:
   - montar dataset semanal com normais recentes e fraudes novas/curadas;
   - treinar modelos candidatos LGBM/Isolation Forest;
   - registrar métricas, parâmetros, artefatos e versões;
   - comparar contra baseline ativo;
   - promover apenas se passar nos gates de regressão, FP/FN, drift e estabilidade.

6. MLflow:
   - usar MLflow para tracking de experimentos, métricas, parâmetros, artefatos e model registry;
   - não tratar MLflow como orquestrador principal;
   - orquestração deve ficar em Airflow, CML Jobs, Oozie ou workflow equivalente;
   - MLflow será a camada de governança de experimentos/modelos, não o scheduler do pipeline.

Racional: o projeto não deve depender de reprocessamentos retroativos caros para operar em produção. A estratégia correta é transformar o custo pesado de 180 dias em um backfill único e, depois disso, operar incrementalmente com janela móvel. Isso reduz custo, melhora previsibilidade operacional e permite manter o modelo atualizado com novos padrões de fraude.

Implicações para o plano de melhoria crítica 3:

O EXP-010E passa a ter dois objetivos:
concluir o backfill inicial de 180 dias da MBK compacta;
definir o workflow incremental diário de atualização da MBK e descoberta de fraudes.
O EXP-010F de amostragem de normais deve usar a mesma janela móvel de 180 dias.
O EXP-010G deve criar o dataset unificado v2 com:
fraudes MAF recentes;
novos casos confirmados descobertos diariamente;
normais amostrados;
MBK compacta quando disponível;
flags explícitas de missingness.
As fases de Feature Store HBase e produção shadow deixam de ser opcionais e passam a ser componentes necessários para serving online.

Decisão final: seguir com arquitetura incremental diária + retreino semanal automatizado, mantendo MLflow como camada de rastreabilidade/model registry e usando orquestração externa para executar os workflows.

---

<!-- decision_id: EXP010E_R2B_MBK_MAF_HYDRATION_CONCLUIDO -->

## EXP-010E-R2B concluído para hidratação MBK das fraudes MAF 180d

O EXP-010E-R2B foi concluído para o objetivo de hidratar as fraudes MAF recentes com dados MBK. Na janela de 180 dias (`2025-11-13` a `2026-05-11`), foram identificadas 1.594 fraudes MAF fortes e 1.391 tiveram correspondência MBK no compact index, resultando em cobertura de 87,26%.

Resultado:

```text
target_keys_total_180d=1594
hydration_rows_total_180d=1594
matched_mbk_total_180d=1391
coverage_total_180d=0.872647

O plano de processamento retornou Dias a processar: 0, indicando que todas as datas-alvo com fraudes MAF na janela já haviam sido compactadas anteriormente. A execução final apenas refez a hidratação e os relatórios.

Observação importante: o script R2B processa datas-alvo com fraudes MAF, não necessariamente todos os dias corridos da janela de 180 dias. Portanto, o resultado aprova a hidratação MBK das fraudes MAF recentes, mas não substitui uma compactação completa por calendário para hidratação futura de transações normais e Feature Store.

Decisão: encerrar o EXP-010E-R2B como aprovado para MAF e seguir para EXP-010F — Normal Sampling v2. A compactação MBK completa por calendário deverá ser tratada em experimento separado, se necessária.

---

<!-- decision_id: EXP010F_R2_NORMAL_SAMPLING_QUALIFICADO_HUE_SQOOP -->

## EXP-010F-R2 definido — Normal Sampling Qualificado via Hue/Sqoop

**Decisão:** encerrar a estratégia de backfill exaustivo de normais por CML/Spark dia a dia e substituir por uma estratégia de amostragem normal qualificada, estratificada e operacionalizada via workflow Hue/Sqoop/Hive.

O EXP-010F-R1 demonstrou que a estratégia de coletar 3.000 normais por dia ao longo de 180 dias é operacionalmente custosa demais para o objetivo de treino. A execução parcial conseguiu gerar aproximadamente 42.000 normais úteis em 14 dias, mas cada dia levou tempo elevado e a sessão Hive/Metastore se tornou instável em execução longa. Portanto, a abordagem é aproveitável como fonte parcial, mas não deve continuar como desenho principal.

Resultado aproveitável do EXP-010F-R1:

```text
dias_ok=14
normais_aproveitaveis≈42000
fonte=hmo_ml.tb_pix_normais_sample_180d_v2 / hmo_ml.tb_pix_normais_sample_mbk_180d_v2
status=PARCIAL_APROVEITAVEL

Nova estratégia: usar workflow Hue/Sqoop/Hive para construir uma base normal menor, porém mais útil para treinamento, com amostragem estratificada e qualificada.

A amostra normal não precisa cobrir mecanicamente todos os dias da janela com o mesmo volume. Ela precisa representar bem a operação recente e incluir casos normais difíceis, parecidos com fraude, para melhorar a capacidade de separação do modelo.

A amostragem normal será composta por estratos:

N1_BACKGROUND_NORMAL:
  amostra aleatória leve de transações normais recentes.

N2_MATCHED_CONTROLS:
  normais pareados com fraudes MAF por janela temporal, faixa de valor, horário e tipo de chave.

N3_HARD_NEGATIVES:
  normais com aparência suspeita, como alto valor, chave aleatória, madrugada, recebedor novo ou padrões próximos aos FPs/FNs conhecidos.

N4_RECENT_NORMALS:
  reforço de normais dos últimos 30/60 dias para capturar comportamento operacional atual.

N5_MBK_COVERED_NORMALS:
  normais com correspondência na MBK compacta, para treinar melhor sinais mobile/device/IP quando disponíveis.

Fontes previstas:

- transações PIX normais via tabelas operacionais já usadas no Big Data;
- fraudes MAF curadas para exclusão de positivos conhecidos;
- MBK compact index para marcação de cobertura mobile/topaz quando disponível;
- amostra parcial do EXP-010F-R1 como base aproveitável;
- futuras tabelas intermediárias criadas pelo workflow Hue/Sqoop/Hive.

Critérios metodológicos:

- excluir qualquer cd_pix presente nas fraudes MAF curadas;
- manter labels MAF como supervisão, nunca como feature;
- preservar janela de maturação para tratar transações sem fraude conhecida como normais;
- evitar leitura massiva desnecessária no CML;
- amostrar cedo no SQL/Hive, antes de qualquer processamento pesado;
- gerar tabelas intermediárias auditáveis;
- limitar artefatos de inspeção a amostras pequenas;
- preparar saída compatível com o EXP-010G.

Meta de volume revisada:

normais já coletados: ~42.000
meta adicional inicial: 40.000 a 80.000 normais qualificados
meta total recomendada para dataset v2: 80.000 a 120.000 normais

A meta anterior de centenas de milhares de normais deixa de ser obrigatória. Para o treino vNext, a prioridade passa a ser qualidade, cobertura de estratos e presença de hard negatives, não volume bruto.

Racional: o dataset final precisa ser forte para treino supervisionado, não necessariamente exaustivo. Uma amostra normal qualificada, combinada com as fraudes MAF recentes/históricas e com MBK quando disponível, deve produzir melhor relação custo-benefício do que continuar o backfill diário pesado via CML.

Impacto no Plano 3:

EXP-010F deixa de ser "Normal Sampling v2 90/180 dias no CML"
e passa a ser:
EXP-010F-R2 — Normal Sampling Qualificado via Hue/Sqoop/Hive.

EXP-010G deverá consumir:
- fraudes MAF hidratadas;
- fraudes MAF com MBK;
- normais parciais do EXP-010F-R1;
- normais qualificados do EXP-010F-R2;
- flags de origem/estrato;
- flags de cobertura MBK;
- features leakage-free recalculadas.

Decisão operacional: antes de escrever o .hql definitivo, revisar os scripts genéricos de workflow Sqoop/Hue usados no ambiente do banco e adaptar o padrão institucional para criar tabelas intermediárias e finais de normais qualificados.

---

---

<!-- decision_id: EXP010F_R2_NORMAL_SAMPLING_HUE_HIVE_APROVADO_COM_AJUSTES -->

## EXP-010F-R2 aprovado com ajustes — Normal Sampling Qualificado via Hue/Hive

**Decisão:** aprovar o EXP-010F-R2 como fonte normal qualificada para o dataset v2, com ajustes obrigatórios no EXP-010G antes de qualquer treino.

O workflow Hue/Hive executou com sucesso e gerou a tabela final:

```text
hmo_ml.tb_pix_normais_dataset_ready_v1

Resultado da auditoria:

total_normais=297015
cd_pix_unicos=297015
duplicados=0
overlap_maf=0
mbk_available=214716
mbk_coverage=72.29%
dt_min=2025-11-13
dt_max=2026-05-22
dias_com_dados=191

Distribuição por estratégia:

N2_MATCHED_CONTROLS=252762 (85.10%)
N0_R1_PARTIAL_REUSE=41891 (14.10%)
N3_HARD_NEGATIVES=2158 (0.73%)
N4_RECENT_NORMALS=147 (0.05%)
N1_BACKGROUND_NORMAL=57 (0.02%)

Racional: a base normal gerada é suficientemente grande, não possui duplicidade e não apresenta overlap com fraudes MAF conhecidas. A cobertura MBK de 72,29% é adequada para treinar sinais mobile/device/topaz com flag explícita de missingness.

Ajustes obrigatórios para o EXP-010G:

1. Normalizar ds_tipo_chave, pois há variantes como CHAVE_ALEATORIA/CHAVE ALEATORIA e DOCUMENTO_TELEFONE/DOCUMENTO/TELEFONE.
2. Definir janela oficial de treino, preferencialmente alinhada à MAF recente validada.
3. Evitar tratar transações muito recentes como normais sem janela de maturação.
4. Controlar o excesso de N2_MATCHED_CONTROLS por sample_weight e/ou downsampling.
5. Preservar origem, estratégia de amostragem e mbk_available_flag no dataset final.

Decisão operacional: não treinar modelo diretamente sobre tb_pix_normais_dataset_ready_v1. A próxima etapa é o EXP-010G — Unified Dataset Builder v2, que deverá criar dataset final versionado, com splits temporais, pesos, manifesto e compatibilidade com o pipeline de treino.

Próximo passo: executar EXP-010G.

---

<!-- decision_id: EXP010G_DATASET_V2_180D_APROVADO -->

## EXP-010G aprovado — Unified Dataset Builder v2 rolling 180d

**Decisão:** aprovar o EXP-010G como dataset v2 oficial para replay e treinamento shadow do modelo antifraude PIX.

O workflow Hue/Hive foi atualizado para executar o EXP-010G após a consolidação dos normais qualificados do EXP-010F-R2. A nova tabela final criada foi:

```text
hmo_ml.tb_pix_dataset_v2_180d_v1

A tabela passou a ser construída em modo rolling 180 dias, sem datas hardcoded, usando:

WINDOW_DAYS=180
WINDOW_LAG_DAYS=1

Resultado da validação:

total_linhas=116988
transaction_id_unicos=116988
duplicados=0
total_fraudes=1491
total_normais=115497
dt_min=2025-11-26
dt_max=2026-05-24
dias_com_dados=180
window_start_date=2025-11-26
window_end_date=2026-05-24

Validações anti-leakage:

DUPLICATED_TRANSACTION_ID=0
LABEL_CONFLICT_TRANSACTION_ID=0
NORMAL_OVERLAP_WITH_MAF=0
OUT_OF_WINDOW_ROWS=0

Distribuição temporal:

TRAIN:
  POSITIVE_FRAUD=1115
  NEGATIVE_NORMAL=80079

VALIDATION:
  POSITIVE_FRAUD=239
  NEGATIVE_NORMAL=17967

HOLDOUT:
  POSITIVE_FRAUD=137
  NEGATIVE_NORMAL=17451

Cobertura MBK:

mbk_available_total=89834
mbk_missing_total=27154
mbk_coverage_total≈76.79%

mbk_available_positivos=1303/1491≈87.39%
mbk_available_normais=88531/115497≈76.65%

Qualidade de features críticas:

missing_vl_pix=0
missing_event_datetime=0
missing_customer_id=0
missing_counterparty_id=0
missing_tipo_chave=0

Estratégias presentes no dataset:

P1_MAF_RECENTE_180D=1491
N2_MATCHED_CONTROLS=107412
N3_HARD_NEGATIVES=4173
N0_R1_PARTIAL_REUSE=2992
N4_RECENT_NORMALS=687
N1_BACKGROUND_NORMAL=233

Racional: o dataset v2 possui janela temporal correta, ausência de duplicidade, ausência de conflito de labels, ausência de overlap entre normais e MAF, cobertura MBK forte e splits temporais adequados para replay/treinamento sem vazamento temporal. A predominância de N2_MATCHED_CONTROLS é aceita neste momento porque a estratégia recebeu sample_weight=0.8, enquanto hard negatives recebem peso maior.

Status: APROVADO_PARA_EXP011.

Próximo passo: iniciar EXP-011A — Replay Baseline no Dataset v2, antes de qualquer promoção de modelo.

---

<!-- decision_id: EXP011B_R1_R2_LGBM_VNEXT_THRESHOLD_TUNING -->

## EXP-011B-R1/R2 — Ajuste de métricas do LGBM vNext Shadow

**Decisão:** aprovar o EXP-011B-R1 como candidato principal para a próxima etapa E2E shadow e manter o EXP-011B-R2 como fallback conservador.

Após o EXP-011B inicial, foi identificado que o modelo LGBM vNext possuía sinal preditivo, mas o threshold operacional inicialmente escolhido (`0.075`) era agressivo demais e inviável operacionalmente.

Resultado original do EXP-011B no holdout com threshold `0.075`:

```text
TP=46
FP=1912
FN=12
TN=6852
precision=0.02349336
recall=0.79310345
f1=0.04563492
fpr=0.21816522
roc_auc=0.89759734
average_precision=0.29720644

Conclusão: o threshold 0.075 foi rejeitado por gerar excesso de falsos positivos.

EXP-011B-R1 — Threshold Policy Sweep

O EXP-011B-R1 não retreinou o modelo; apenas reavaliou políticas de threshold sobre o modelo treinado no EXP-011B.

Política selecionada:

policy=PRECISION_GE_50_FPR_LE_1PCT
threshold=0.60
selection_reason=Selecionado por equilíbrio: precision>=50%, FPR<=1% e maior F1 na validação.

Resultado no holdout:

TP=15
FP=14
FN=43
TN=8750
precision=0.51724138
recall=0.25862069
f1=0.34482759
fpr=0.00159744

Decisão: aprovar o R1 como candidato principal, pois ele melhora o baseline enriquecido no holdout, aumenta o F1 e reduz drasticamente falsos positivos.

EXP-011B-R2 — Retreino conservador

O EXP-011B-R2 testou 5 configurações LightGBM mais conservadoras, variando regularização, complexidade e peso positivo.

Configuração campeã:

config_id=C03_balanced_weight
policy=FP_LE_100_BEST_F1
threshold=0.53

Resultado no holdout:

TP=13
FP=7
FN=45
TN=8757
precision=0.65
recall=0.22413793
f1=0.33333333
fpr=0.00079872
roc_auc=0.89770849
average_precision=0.29810059

Decisão: manter o R2 como fallback conservador. Apesar de ter precision maior e menos falsos positivos, ele perde TP, recall e F1 em relação ao R1.

Comparação final no holdout
Baseline EXP-011A enriquecido:
TP=12, FP=73, FN=46, TN=8691, precision=0.1412, recall=0.2069, f1=0.1678

EXP-011B-R1:
TP=15, FP=14, FN=43, TN=8750, precision=0.5172, recall=0.2586, f1=0.3448

EXP-011B-R2:
TP=13, FP=7, FN=45, TN=8757, precision=0.6500, recall=0.2241, f1=0.3333

Racional: o R1 apresenta o melhor equilíbrio geral para a próxima etapa: melhora TP, reduz FP, aumenta precision, aumenta recall e praticamente dobra o F1 em relação ao baseline enriquecido. O R2 permanece útil como política alternativa caso a operação priorize redução máxima de falsos positivos.

Status:

EXP-011B-R1 = APROVADO_COMO_CANDIDATO_PRINCIPAL
EXP-011B-R2 = APROVADO_COMO_FALLBACK_CONSERVADOR
threshold 0.075 = REJEITADO
threshold R1 0.60 = APROVADO_PARA_E2E_SHADOW

Próximo passo: iniciar EXP-011C — E2E Shadow com LGBM vNext R1, integrando o candidato ao DecisionEngine em modo shadow, sem sobrescrever o modelo produtivo.
