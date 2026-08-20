# CHANGELOG

Todas as mudancas notaveis deste projeto sao documentadas neste arquivo.

O formato segue a ideia de [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/), adaptada para os experimentos internos do projeto.

## [1.5.0-r5b22] - 2026-06-12

### Ajustes Fase 7 (Proposta Técnica HBase)
- Atualizado o documento `docs/proposta_tecnica_hbase.md` para suportar o baseline `1.5.0-r5b22`.
- Expansão do catálogo de tabelas sugeridas para incluir o contrato congelado (`fraud_detection:r5b22_contract_features`) e o histórico unificado de relacionamento (`fraud_detection:receiver_history` e arestas HBase).
- Definição explícita das 78 features separadas por origem (transação, enrich, preprocessor, HBase e sinais de metadados frozen) compatibilizando TTL e SLA estrito.

### Ajustes Fase 6 (Graph Engineering Investigativo)
- Atualizado arquivo `docs/modulo_graph_feature_engineering.md` para refletir o status investigativo e *opt-in* assíncrono (evitando impacto no SLA transacional do Baseline).
- Implementado módulo de produção `backend/core/graph_engineering.py` (`GraphInvestigationEngine`) que exporta, para CSV, os scores analíticos dos topológicos suspeitos (fan-in/fan-out, contas ponte, is_new_receiver, suspected_mule_score).
- Adicionados testes de sanidade robustos (`tests/test_graph_engineering.py`) confirmando resiliência à ausência de chaves, integridade com transações do tipo "APROVAR", e escrita isolada com locks e thread safety.

### Ajustes Fase 5 (Catálogo de Regras R5B22)
- Criado o documento `docs/catalogo_regras_r5b22.md` com a relação exaustiva das regras vigentes.
- Criado extrator dinâmico `scripts/export_r5b22_rule_catalog.py` que compila regras estruturais do código e métricas das heurísticas JSON ativas, originando sumários em Markdown, CSV e JSON.
- Consolidação transparente das regras de baixa tolerância (R5B14) e do controle de falso positivo (R5B22), garantindo a auditoria demandada pelas frentes de negócio.

### Ajustes Fase 4 (Atualização Documental de Componentes e Motor)
- Atualizada documentação `docs/modulo_comportamental.md` para separar a avaliação especialista (runtime) da integração do `beh_score` pelo aluno preditivo, atualizando a calibração para 113.844 transações e classificando os antigos thresholds v3.0 como aviso histórico.
- Atualizada documentação `docs/modulo_engenharia_social.md`, ressaltando o status de explicabilidade do SE no R5B22 e sua comunicação por `se_score` e `se_worst_pattern` para o Baseline Oficial, rebaixando as antigas taxas e Lifts para controle histórico.
- Atualizado o manual mestre `docs/motor_decisao_modelo.md`, formalizando a versão 1.5.0-r5b22, orquestrada via destilação do contrato (R5B16/R5B18) atrelada às políticas seguras R5B14 e finalizando com a suavização via DEMOTIONS das regras R5B22. Métricas, diagramas e listagem oficial de overrides adaptadas para o modelo final.

### Ajustes Fase 3 (Contrato Documental de Features)
- Atualizado arquivo `docs/lista_de_features.md` para abranger as exatas 78 features catalogadas e extraídas no metadado do aluno R5B22, abandonando o escopo antigo (MVP/52 features). 
- Categorização explícita entre features transacionais de tempo real, dados do Feature Store (HBase), derivações em tempo real e sinais de telemetria/telemetria frozen do baseline do professor.

### Ajustes Fase 2 (Scripts de Treino R5B22)
- Criado `backend/modelos/train_lgbm_distilled_r5b22.py` para reproduzir semanalmente o aluno destilado (intervenção e bloqueio) a partir das targets do professor (contrato congelado R5B16/R5B18 e políticas).
- Implementado gate de segurança para mapear categorias ausentes e garantir uso exclusivo das 78 features catalogadas em `model_lgbm_distilled_r5b22_metadata.json`.
- Script legado `train_lgbm_canonical.py` atualizado com aviso de depreciação indicando uso do novo modelo.
- Documentação de `train_isolation_forest_canonical.py` atualizada ressaltando seu caráter consultivo sob as novas regras do baseline.

### Ajustes Fase 1 (API e Explicabilidade)
- Endpoint `/api/v1/analyze` atualizado para receber as flags opcionais `explain` e `debug` visando reduzir o payload em produção.
- `AnalyzeResponse` atualizado para incluir condicionalmente campos do baseline R5B22: `r5b22_policy_applied`, `r5b22_rule_applied` e `decisao_original_r5b22`.
- Informações de metadados da aplicação, versionamento (`1.5.0-r5b22`) e descrição atualizados na definição principal do FastAPI.

### Adicionado

- Baseline oficial `R5B22_OFFICIAL_CONSTRAINED_BASELINE`.
- Politica oficial em `backend/artefatos/r5b22_official_baseline_policy.json`.
- Sumario oficial em `backend/artefatos/r5b22_official_baseline_summary.json`.
- LGBM aluno distilado do contrato R5B16/R5B18:
  - `backend/artefatos/model_lgbm_distilled_r5b22_intervention.joblib`
  - `backend/artefatos/model_lgbm_distilled_r5b22_block.joblib`
  - `backend/artefatos/model_lgbm_distilled_r5b22_metadata.json`
- Documento executivo atualizado em `docs/apresentacao_mvp_v2.md`.
- Suporte runtime a R5B22 no `PipelineOrquestrador`, aplicado apos R5B16/R5B14.
- Campos de auditoria R5B22 no E2E:
  - `decisao_original_r5b22`
  - `r5b22_policy_applied`
  - `r5b22_rule_applied`

### Alterado

- `backend/artefatos/scoring_config.json` atualizado para `1.5.0-r5b22`.
- Flags oficiais ativadas:
  - `r5b14_operational_zero_fn_enabled=true`
  - `r5b16_frozen_contract_enabled=true`
  - `r5b22_official_baseline_enabled=true`
  - `official_baseline_policy=R5B22_OFFICIAL_CONSTRAINED_BASELINE`
- `backend/core/decision_engine.py` passou a reconhecer as flags oficiais no `EngineConfig`.
- `backend/core/pipeline_orquestrador.py` passou a carregar e aplicar a politica oficial R5B22.
- `backend/scripts/simular_pipeline_e2e_v2.py` passou a usar a base MAF v3 como dataset padrao e anexar colunas do contrato R4G frozen para reproduzir R5B16/R5B22.
- `README.md` atualizado para refletir o estado operacional atual do projeto.

### Metricas oficiais

Distribuicao operacional R5B22:

| Decisao | Transacoes | Fraudes | Normais |
|---|---:|---:|---:|
| APROVAR | 111.305 | 2 | 111.303 |
| CONFIRMAR | 326 | 10 | 316 |
| BLOQUEAR | 2.213 | 1.453 | 760 |

Metricas globais:

| Metrica | Valor |
|---|---:|
| TP | 1.463 |
| FP | 1.076 |
| FN | 2 |
| TN | 111.303 |
| Precision | 57,621111% |
| Recall | 99,863481% |
| F1 | 0,73076923 |
| FPR | 0,957474% |

Metricas de `BLOQUEAR`:

| Metrica | Valor |
|---|---:|
| TP | 1.453 |
| FP | 760 |
| FN fora de BLOQUEAR | 12 |
| TN | 111.619 |
| Precision | 65,657479% |
| Recall | 99,180887% |
| F1 | 0,79010332 |
| FPR | 0,676283% |

### Validacao

- `python -m py_compile backend/core/decision_engine.py backend/core/pipeline_orquestrador.py backend/scripts/simular_pipeline_e2e_v2.py`
- Carga do `PipelineOrquestrador` confirmou R5B14, R5B16, R5B22 e policy JSON ativos.
- Carga amostral do E2E confirmou anexacao das colunas frozen R4G usadas por R5B16/R5B22.

## [1.4.x-r5b16-r5b18] - 2026-06-12

### Adicionado

- Homologacao E2E do contrato congelado R5B16/R5B18.
- Reproducao operacional do baseline congelado usando `r4g_fast_frozen_decisao_recommended`.
- Politica R5B14 para restricoes de baixo falso negativo.
- Gates de validacao para FPR global menor que 1% e controle de falso negativo fora de `BLOQUEAR`.

### Resultado

- Baseline anterior concentrava todas as fraudes conhecidas fora de `APROVAR`, mas mantinha mais normais em `BLOQUEAR`.
- R5B22 substituiu esse baseline ao reduzir normais bloqueadas e melhorar a precisao de `BLOQUEAR`, aceitando teto controlado de fraude em `APROVAR` e `CONFIRMAR`.

## [3.1.0] - 2026-04-20

### Adicionado

- EXP-003: novo padrao residual no `SocialEngineeringDetector` para perfis vulneraveis com transferencia atipica moderada e alta anomalia.
- EXP-002: guard rail LGBM para suprimir vetos do Isolation Forest quando o LGBM indica baixa probabilidade de fraude.

### Alterado

- Threshold global de `CONFIRMAR` ajustado em `scoring_config.json`.
- `pipeline_orquestrador.py` passou a precomputar `if_percentile` para uso pelo detector de engenharia social.

## [3.0.5] - 2026-04-12

### Adicionado

- Graph Feature Engineering com features temporais incrementais sem leakage.
- LightGBM v6.1 experimental com graph features.
- Fast-Approve Override.
- Cascade v3 com LGBM guard.
- Simulacao E2E leakage-free.

### Alterado

- Decision Engine v3.0.5 melhorou precision e reduziu falsos positivos no baseline antigo.
- Preprocessing v4.1 passou a gerar dataset processado com fases de leakage fix, graph features e selecao.
- Estrutura do projeto foi reorganizada com limpeza de artefatos intermediarios.

## [3.0.0] - 2026-04-11

### Adicionado

- Behavioral Analytics v3.0.
- Social Engineering v3.3.

### Alterado

- Correcao de leakage temporal nas features trimestrais.
- Retreino do LightGBM v5.1 com features leakage-free.
- Retreino do Isolation Forest v3 com features reduzidas.

### Removido

- Fatores BEH e indicadores SE com performance negativa.
- `rule_engine.py`, substituido pelo Cascade integrado ao Decision Engine.

## [2.1.1] - 2026-03-22

### Alterado

- Ajuste de precisao da regra C6 do Cascade.
- Calibragem de ancoras em `scoring_config.json`.

## [2.1.0] - 2026-03-22

### Adicionado

- API com explicabilidade.
- Decision Engine v2.1.
- Modulos iniciais de Social Engineering e Behavioral Analytics.

### Alterado

- Ajustes iniciais para reduzir falso negativo e FPR no dataset antigo.

## [0.0.1] - 2026-03-06

### Adicionado

- Documento inicial de requisitos e arquitetura.
- Lista de features para MVP.
- README, CHANGELOG e `requirements.txt`.
- Scripts de backend e ingestao de dados.

