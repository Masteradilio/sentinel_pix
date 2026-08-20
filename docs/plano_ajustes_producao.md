# Plano de ajustes para producao - baseline R5B22

## 0. Contexto obrigatorio antes de implementar

Este plano deve ser executado por um LLM/coding agent que ainda nao conhece toda a jornada do projeto. Antes de alterar codigo ou documentacao, leia os arquivos abaixo nesta ordem:

1. `README.md`
   - Estado oficial do projeto.
   - Baseline atual: `R5B22_OFFICIAL_CONSTRAINED_BASELINE`.
   - Metricas oficiais, artefatos oficiais e arquitetura operacional.

2. `CHANGELOG.md`
   - Mudancas da versao `1.5.0-r5b22`.
   - Arquivos adicionados/alterados para R5B22.
   - Validacoes ja executadas.

3. `docs/JOURNAL_4.md`
   - Historico dos experimentos R4/R5.
   - Motivo da evolucao para contrato congelado, politica R5B14 e baseline oficial R5B22.

4. `docs/PIX_FRAUD_MODEL_FINAL_IMPROVEMENT_PLAN.md`
   - Plano experimental completo que levou ao baseline final.
   - Especialmente as secoes finais R5B16, R5B18, R5B22 e qualquer anexo posterior.

5. `docs/apresentacao_mvp_v2.md`
   - Narrativa executiva correta do MVP v2.
   - Metricas oficiais, regras ativas e leitura operacional do modelo.

6. Artefatos oficiais:
   - `backend/artefatos/scoring_config.json`
   - `backend/artefatos/r5b22_official_baseline_policy.json`
   - `backend/artefatos/r5b22_official_baseline_summary.json`
   - `backend/artefatos/model_lgbm_distilled_r5b22_metadata.json`
   - Nao e necessario abrir os arquivos `.joblib` diretamente.

7. Experimento oficial:
   - `resultados/experimentos/EXP-014B-R5B22-OFFICIAL-CONSTRAINED-BASELINE/`

8. Codigo runtime principal:
   - `backend/api.py`
   - `backend/core/pipeline_orquestrador.py`
   - `backend/core/decision_engine.py`
   - `backend/core/r5b14_operational_policy.py`
   - `backend/core/severity_policy.py`
   - `backend/core/social_engineering.py`
   - `backend/core/behavioral_analytics.py`

9. Scripts de treino atuais:
   - `backend/modelos/train_lgbm_canonical.py`
   - `backend/modelos/train_isolation_forest_canonical.py`

10. Documentos a serem atualizados neste plano:
   - `docs/lista_de_features.md`
   - `docs/modulo_comportamental.md`
   - `docs/modulo_engenharia_social.md`
   - `docs/motor_decisao_modelo.md`
   - `docs/modulo_graph_feature_engineering.md`
   - `docs/proposta_tecnica_hbase.md`

## 1. Estado alvo do baseline

O baseline de producao atual e:

```text
Baseline: R5B22_OFFICIAL_CONSTRAINED_BASELINE
Versao de scoring: 1.5.0-r5b22
Base de validacao: 113.844 transacoes
Fraudes confirmadas: 1.465
Normais: 112.379
```

Metricas globais considerando intervencao como `CONFIRMAR` ou `BLOQUEAR`:

```text
TP=1463
FP=1076
FN=2
TN=111303
Precision=57,621111%
Recall=99,863481%
F1=0,73076923
FPR=0,957474%
```

Metricas especificas de `BLOQUEAR`:

```text
TP=1453
FP=760
FN fora de BLOQUEAR=12
TN=111619
Precision=65,657479%
Recall=99,180887%
F1=0,79010332
FPR=0,676283%
```

Distribuicao final:

```text
APROVAR:   111.305 transacoes | 2 fraudes    | 111.303 normais
CONFIRMAR:     326 transacoes | 10 fraudes   | 316 normais
BLOQUEAR:    2.213 transacoes | 1.453 fraudes| 760 normais
```

Gates que nao podem ser violados:

```text
FPR global < 1%
Fraudes em APROVAR <= 5
Fraudes em CONFIRMAR <= 10
Precisao de BLOQUEAR deve permanecer melhor que o baseline R5B16/R5B18
```

Arquitetura operacional correta:

```text
PipelineOrquestrador
  -> PixDecisionEngine
  -> contrato congelado R5B16/R5B18 baseado em r4g_fast_frozen_decisao_recommended
  -> politica R5B14 de baixo FN
  -> politica oficial R5B22
  -> decisao final: APROVAR / CONFIRMAR / BLOQUEAR
```

Importante: o LGBM aluno R5B22 e uma destilacao do contrato R5B16/R5B18. Ele nao deve ser descrito nem tratado como um LGBM puro treinado apenas em features brutas. Ele usa sinais do professor, incluindo:

```text
r4g_fast_frozen_decisao_recommended
r5b14_rule_applied
r5b14_layer_applied
```

## Fase 1 - Adequar a API para producao R5B22 e explicabilidade SHAP (Concluído)

### Objetivo

Adequar `backend/api.py` para expor o baseline R5B22 de forma enxuta, estavel e explicavel, preservando boas praticas de API. A API deve continuar sendo uma camada HTTP fina sobre `PipelineOrquestrador`; ela nao deve virar local de feature engineering, regra antifraude ou regra de modelo.

### Fontes de verdade

- `backend/api.py`
- `backend/core/pipeline_orquestrador.py`
- `backend/core/decision_engine.py`
- `backend/artefatos/scoring_config.json`
- `backend/artefatos/r5b22_official_baseline_policy.json`
- `backend/artefatos/r5b22_official_baseline_summary.json`
- `docs/apresentacao_mvp_v2.md`

### Tarefas

1. Mapear a resposta atual de `/api/v1/analyze`.
   - Identificar campos retornados hoje.
   - Separar campos indispensaveis de debug interno.
   - Verificar se campos R5B22 aparecem quando a politica e aplicada:
     - `decisao_original_r5b22`
     - `r5b22_policy_applied`
     - `r5b22_rule_applied`

2. Definir contrato de resposta enxuto.
   - Campos obrigatorios sugeridos:
     - `transaction_id`
     - `customer_id`
     - `decisao`
     - `score_final`
     - `timestamp`
     - `vl_pix`
     - `componentes`
     - `explicabilidade`
     - `politicas_aplicadas`
     - `mensagem_cliente` ou bloco `cx`
   - Campos de debug pesado devem ficar opcionais e controlados por flag/query param, por exemplo `include_debug=true`.

3. Reconciliar explicabilidade SHAP.
   - Verificar como `PipelineOrquestrador` gera SHAP hoje.
   - Garantir que a API preserve a explicabilidade quando ela existir.
   - Se SHAP estiver indisponivel para alguma transacao, retornar explicacao degradada, mas estruturada:
     - principais componentes de score;
     - regra/politica aplicada;
     - fatores SE/BEH;
     - motivo da decisao.
   - Nao calcular SHAP dentro de `backend/api.py`; o calculo deve permanecer no orquestrador ou em modulo proprio.

4. Atualizar modelos Pydantic.
   - Alinhar `AnalyzeResponse`, `BatchResponse` e modelos auxiliares com R5B22.
   - Evitar `Dict[str, Any]` onde houver estrutura estavel.
   - Manter flexibilidade apenas em blocos naturalmente variaveis, como `explicabilidade`.

5. Adicionar modo de explicabilidade.
   - `explain=false`: resposta minima para baixa latencia.
   - `explain=true`: incluir SHAP e explicabilidade completa.
   - `debug=true`: incluir campos internos, traces e politicas.

6. Atualizar `/api/v1/health` e `/api/v1/status`.
   - Expor:
     - `scoring_version=1.5.0-r5b22`
     - flags R5B14/R5B16/R5B22;
     - policy id oficial;
     - presenca dos artefatos oficiais.

7. Testar endpoints.
   - Criar/atualizar testes em `tests/test_api_smoke.py`.
   - Testar:
     - transacao unica;
     - lote;
     - health/status;
     - resposta com e sem explicabilidade;
     - presenca de campos R5B22 quando politica aplicada.

### Gates de aceite

- `python -m pytest tests/test_api_smoke.py -q`
- `python -m pytest tests/test_pipeline_inference.py -q`
- A API nao deve importar `.joblib` diretamente; isso continua responsabilidade do pipeline/engine.
- A API nao deve conter regra antifraude hardcoded.
- O contrato deve ser documentado em docstring ou em `docs/api_contract_r5b22.md` se necessario.

## Fase 2 - Adequar scripts de treino para reproduzir o aluno R5B22 (Concluído)

### Objetivo

Atualizar os scripts em `backend/modelos` para treinar semanalmente modelos compativeis com o baseline oficial R5B22. O treino deve reproduzir o LGBM aluno destilado do contrato R5B16/R5B18 e preservar thresholds, features, categoricos e metricas de validacao.

### Problema atual

Os scripts atuais `train_lgbm_canonical.py` e `train_isolation_forest_canonical.py` ainda refletem o ciclo canônico R5A3. Eles treinam modelos diretamente sobre `is_fraud` e nao reproduzem integralmente o aluno R5B22, que depende do contrato congelado e de sinais do professor.

### Fontes de verdade

- `backend/artefatos/model_lgbm_distilled_r5b22_metadata.json`
- `backend/artefatos/model_lgbm_distilled_r5b22_intervention.joblib`
- `backend/artefatos/model_lgbm_distilled_r5b22_block.joblib`
- `backend/artefatos/r5b22_official_baseline_summary.json`
- `backend/artefatos/r5b22_official_baseline_policy.json`
- `backend/scripts/simular_pipeline_e2e_v2.py`
- `resultados/experimentos/EXP-014B-R5B22-OFFICIAL-CONSTRAINED-BASELINE/`

### Tarefas

1. Criar script de treino destilado R5B22.
   - Nome sugerido: `backend/modelos/train_lgbm_distilled_r5b22.py`.
   - Treinar dois modelos:
     - `model_lgbm_distilled_r5b22_intervention.joblib`
     - `model_lgbm_distilled_r5b22_block.joblib`
   - Targets:
     - intervencao: `APROVAR` vs `CONFIRMAR/BLOQUEAR`;
     - bloqueio: `BLOQUEAR` vs `APROVAR/CONFIRMAR`.
   - O target do aluno deve vir do contrato professor, nao diretamente de `is_fraud`.

2. Congelar contrato de features.
   - Usar exatamente as 78 features de `model_lgbm_distilled_r5b22_metadata.json`.
   - Validar presenca, ordem e tipo.
   - Falhar explicitamente se alguma feature obrigatoria estiver ausente.

3. Reproduzir codificacao categorica.
   - Usar as categorias de `category_encoders` no metadata R5B22.
   - Definir tratamento para categorias novas:
     - mapear para `<MISSING>` quando houver;
     - ou mapear para categoria fallback documentada.
   - Nao fazer `LabelEncoder.fit` livre em producao sem preservar mapeamento.

4. Reproduzir thresholds oficiais.
   - `intervention_threshold = 0.10908055367264094`
   - `block_threshold = 0.0807211383972785`
   - Salvar thresholds no metadata e em arquivo de metricas.

5. Atualizar script canônico antigo.
   - `train_lgbm_canonical.py` pode:
     - chamar o novo script;
     - ou ser mantido como treino legado com aviso claro.
   - O README/doc deve apontar o treino semanal para o script R5B22, nao para R5A3.

6. Revisar `train_isolation_forest_canonical.py`.
   - Confirmar se IF ainda e usado como componente consultivo no runtime R5B22.
   - Se sim, manter treino semanal, mas documentar que ele nao define sozinho o baseline R5B22.
   - Garantir que `if_features.json`, scaler e referencia de scores sejam salvos com versionamento.

7. Criar relatorio semanal de treino.
   - Saida sugerida:
     - `backend/modelos/resultado_treino_r5b22/metricas_r5b22_distilled.json`
     - `backend/modelos/resultado_treino_r5b22/feature_importance_intervention.csv`
     - `backend/modelos/resultado_treino_r5b22/feature_importance_block.csv`
     - `backend/modelos/resultado_treino_r5b22/validation_report.md`

8. Implementar gates de reproducao.
   - Comparar contra metricas oficiais:
     - FPR global < 1%;
     - APROVAR fraud <= 5;
     - CONFIRMAR fraud <= 10;
     - BLOQUEAR precision nao pode cair sem aprovacao explicita.
   - Comparar mimic do professor:
     - precision/recall/F1 de mimic do contrato;
     - divergencias por split.

### Gates de aceite

- O treino semanal deve gerar artefatos versionados sem sobrescrever automaticamente os oficiais.
- Promocao para `backend/artefatos` deve exigir etapa explicita de validacao.
- O metadata gerado deve conter:
  - features;
  - categoricos;
  - thresholds;
  - metricas por split;
  - hash dos datasets usados;
  - data/hora;
  - policy id professor.

## Fase 3 - Atualizar `docs/lista_de_features.md` (Concluído)

### Objetivo

Atualizar a lista de features para representar corretamente as features MAF e sinais derivados usados no baseline R5B22.

### Fonte de verdade principal

`backend/artefatos/model_lgbm_distilled_r5b22_metadata.json`, campo `feature_columns`.

### Tarefas

1. Reescrever a introducao.
   - Remover narrativa antiga de MVP inicial com 52 features.
   - Explicar que o baseline R5B22 usa 78 features no LGBM aluno destilado.

2. Separar features por grupo:
   - Identificacao e controle, nao preditoras:
     - `transaction_id`, `customer_id`, `event_datetime`, `is_fraud`, splits.
   - Transacao atual:
     - `vl_pix`, `ds_tipo_chave_norm`, `hour`, `periodo_dia`, `value_band`, `autcodret`.
   - Mobile/device/host:
     - `latencia_rede_ms`, `tempo_processamento_host_ms`, `topaz_risk_score`, `mbk_completeness_score`, `mbk_available_flag`.
   - Historico pagador:
     - `qtd_pix_pagador_7d`, `qtd_pix_pagador_30d`, `qtd_pix_pagador_90d`, `qtd_pix_pagador_180d`;
     - `valor_total_pagador_7d`, `valor_total_pagador_30d`, `valor_total_pagador_90d`, `valor_total_pagador_180d`;
     - `max_qtd_pix_dia_pagador_7d`, `max_qtd_pix_dia_pagador_30d`, `valor_maximo_pix_pagador_180d`;
     - `soma_recebedores_distintos_dia_180d`.
   - Relacao pagador-recebedor:
     - `qtd_pix_mesmo_recebedor_7d`, `qtd_pix_mesmo_recebedor_30d`, `qtd_pix_mesmo_recebedor_90d`, `qtd_pix_mesmo_recebedor_180d`;
     - `valor_total_para_recebedor_30d`, `valor_total_para_recebedor_90d`, `valor_total_para_recebedor_180d`;
     - `primeiro_envio_para_recebedor_180d`, `dias_desde_primeiro_envio_recebedor`;
     - `valor_medio_para_recebedor_180d`, `dias_desde_ultima_transacao_recebedor`, `ratio_valor_pix_vs_max_recebedor_180d`, `is_recebedor_recorrente_180d`.
   - Historico recebedor:
     - `qtd_pix_recebidos_30d`, `qtd_pix_recebidos_90d`, `qtd_pix_recebidos_180d`;
     - `valor_total_recebido_30d`, `valor_total_recebido_90d`, `valor_total_recebido_180d`;
     - `soma_pagadores_distintos_dia_recebedor_180d`, `max_qtd_pix_recebidos_dia_180d`;
     - `first_receiver_flag_real`.
   - Ratios e flags:
     - `burst_daily_7d_flag`, `ratio_valor_media_pagador_90d`, `ratio_valor_maximo_pagador_180d`.
   - Sinais congelados/professor:
     - `r4g_fast_frozen_decisao_recommended`;
     - `r5b14_rule_applied`;
     - `r5b14_layer_applied`;
     - `ds_tipo_chave_norm_frozen`, `value_band_frozen`, `periodo_dia_frozen`;
     - `score_bin`, `lgbm_bin`, `if_bin`, `ratio_bin`, `qtd_rec_bin`, `valor_rec_bin`;
     - `mbk_available_flag_frozen`, `first_receiver_flag_real_frozen`;
     - `ratio_valor_maximo_pagador_180d_frozen`, `ratio_valor_media_pagador_90d_frozen`, `vl_pix_frozen`, `qtd_pix_pagador_180d_frozen`, `valor_total_pagador_180d_frozen`, `qtd_pix_mesmo_recebedor_180d_frozen`, `valor_total_para_recebedor_180d_frozen`.
   - Scores de componentes:
     - `module_quiet`, `se_worst_pattern`, `lgbm_raw`, `lgbm_r4_score`, `score_final`, `lgbm_mapped`, `if_percentile`, `se_score`, `beh_score`.

3. Marcar features por origem operacional.
   - Evento da transacao.
   - Feature store HBase.
   - Cache/estado online.
   - Derivada no runtime.
   - Derivada offline/contrato professor.

4. Indicar features categoricas.
   - Usar `categorical_features` do metadata:
     - `ds_tipo_chave_norm`
     - `periodo_dia`
     - `value_band`
     - `r4g_fast_frozen_decisao_recommended`
     - `r5b14_rule_applied`
     - `r5b14_layer_applied`
     - `ds_tipo_chave_norm_frozen`
     - `value_band_frozen`
     - `periodo_dia_frozen`
     - `score_bin`
     - `lgbm_bin`
     - `if_bin`
     - `ratio_bin`
     - `qtd_rec_bin`
     - `valor_rec_bin`
     - `module_quiet`
     - `se_worst_pattern`

### Gates de aceite

- O documento nao deve mais afirmar que o modelo oficial usa 52 features.
- O documento deve explicar claramente que algumas features sao sinais do professor e nao campos brutos.
- O total de features do aluno deve bater com `n_features=78`.

## Fase 4 - Atualizar documentacao dos modulos BEH, SE e Motor de Decisao (Concluído)

### Objetivo

Atualizar `modulo_comportamental.md`, `modulo_engenharia_social.md` e `motor_decisao_modelo.md` para refletirem o funcionamento real do baseline R5B22, sem apagar o valor historico dos modulos.

### Tarefas para `docs/modulo_comportamental.md`

1. Atualizar ficha tecnica.
   - Dataset de referencia antigo: 100.355 transacoes / 355 fraudes.
   - Dataset atual do baseline: 113.844 transacoes / 1.465 fraudes.
   - Explicar se o BEH ainda opera como componente runtime e como seu score entra no contrato R5B22.

2. Separar duas camadas:
   - BEH como modulo especialista runtime (`behavioral_analytics.py`);
   - `beh_score` como feature/sinal usado pelo aluno R5B22.

3. Revisar claims de performance.
   - Nao usar metricas antigas como se fossem metricas oficiais atuais.
   - Manter metricas antigas em secao "historico" se ainda forem uteis.

4. Documentar integracao com API.
   - Como fatores BEH aparecem na explicabilidade.
   - Como devem ser exibidos em `motivos`/`explicabilidade`.

### Tarefas para `docs/modulo_engenharia_social.md`

1. Atualizar ficha tecnica para estado R5B22.
   - SE permanece como modulo especialista e como sinal `se_score`/`se_worst_pattern`.

2. Separar:
   - regras/padroes SE historicos;
   - papel atual no baseline R5B22;
   - uso na explicabilidade e auditoria.

3. Revisar metricas antigas.
   - Nao apresentar validacao de 355 fraudes como validacao oficial atual.
   - Referenciar R5B22 como baseline oficial com 1.465 fraudes.

4. Documentar saida esperada.
   - `se_score`
   - `patterns`
   - `active_indicators`
   - `risk_level`
   - `se_worst_pattern`

### Tarefas para `docs/motor_decisao_modelo.md`

1. Atualizar versao do motor documentado.
   - De v3.0.5 para baseline operacional `1.5.0-r5b22`.

2. Trocar narrativa do fluxo.
   - Fluxo antigo: LGBM + IF + Cascade + SE + BEH + thresholds.
   - Fluxo atual:
     - `PipelineOrquestrador`;
     - `PixDecisionEngine`;
     - contrato congelado R5B16/R5B18;
     - politica R5B14;
     - politica oficial R5B22;
     - decisao final.

3. Documentar flags oficiais.
   - `r5b14_operational_zero_fn_enabled`
   - `r5b16_frozen_contract_enabled`
   - `r5b22_official_baseline_enabled`
   - `official_baseline_policy`

4. Documentar regras ativas.
   - Regras R5B14:
     - `R5B14_CTB_01_LGBM_RAW_HIGH`
     - `R5B14_CTB_02_SCORE_2_3_LGBM_R4_HIGH`
     - `R5B14_CTB_03_SCORE_2_3_LGBM_R4_MED`
     - `R5B14_CTB_04_DOC_PHONE_HIGH_PAYER_COUNT`
     - `R5B14_CTB_05_OUTROS_RATIO_MAX_HIGH`
     - `R5B14_ATB_01_DOC_PHONE_MORNING_SCORE_HIGH`
     - `R5B14_ATB_02_NIGHT_SCORE_1_2_RATIO_HIGH`
     - `R5B14_CTA_01_LOW_LGBM_RAW_COMPENSATION`
   - Regras R5B22:
     - `DEMOTE_LAYER_APPROVE_TO_BLOCK_TO_APROVAR`
     - `DEMOTE_LAYER_CONFIRM_TO_BLOCK_TO_CONFIRMAR`
     - `DEMOTE_CAT2_ds_tipo_chave_norm_OUTROS__lgbm_bin_lgbm_0.05_0.1`
     - `DEMOTE_CAT2_value_band_E_5000_10000__lgbm_bin_lgbm_0.05_0.1`

5. Atualizar metricas.
   - Usar as metricas oficiais R5B22 do README/apresentacao/summary.

### Gates de aceite

- Nenhum dos tres docs deve vender v3.0.5 como baseline atual.
- Todos devem apontar R5B22 como baseline oficial.
- Claims historicos devem ser rotulados como historicos.

## Fase 5 - Extrair e documentar todas as regras ativas do modelo (Concluído)

### Objetivo

Criar uma lista completa e auditavel das regras/politicas que ajudam o modelo a reconhecer quase todas as fraudes, consolidando o experimento R5B22, artefatos oficiais, apresentacao e `backend/core/severity_policy.py`.

### Fontes obrigatorias

- `resultados/experimentos/EXP-014B-R5B22-OFFICIAL-CONSTRAINED-BASELINE/`
- `backend/artefatos/r5b22_official_baseline_policy.json`
- `backend/artefatos/r5b22_official_baseline_summary.json`
- `backend/artefatos/scoring_config.json`
- `backend/core/r5b14_operational_policy.py`
- `backend/core/severity_policy.py`
- `docs/apresentacao_mvp_v2.md`

### Tarefas

1. Criar documento de regras.
   - Nome sugerido: `docs/catalogo_regras_r5b22.md`.

2. Consolidar regras R5B14.
   - Para cada regra:
     - id;
     - camada;
     - condicao;
     - acao;
     - objetivo;
     - impacto quando disponivel.

3. Consolidar regras R5B22.
   - Fonte primaria: `r5b22_official_baseline_policy.json`.
   - Incluir:
     - target action;
     - linhas;
     - fraudes;
     - normais;
     - incremental_n_rows;
     - incremental_n_frauds;
     - incremental_n_normals.

4. Incluir contrato congelado R5B16/R5B18.
   - Explicar que `r4g_fast_frozen_decisao_recommended` e uma decisao-base congelada, nao uma regra simples.
   - Listar os campos de contrato congelado usados pelo aluno.

5. Incluir `backend/core/severity_policy.py`.
   - Explicar se esta politica esta ativa no baseline oficial ou se e candidato/apoio.
   - Nao misturar politica candidata com baseline oficial sem verificar flags em `scoring_config.json`.

6. Criar extrator automatico.
   - Nome sugerido: `scripts/export_r5b22_rule_catalog.py`.
   - Entradas:
     - policy JSON;
     - summary JSON;
     - arquivos de policy core quando possivel.
   - Saidas:
     - `docs/catalogo_regras_r5b22.md`;
     - `resultados/r5b22_rule_catalog.csv`;
     - `resultados/r5b22_rule_catalog.json`.

7. Validar completude.
   - A lista deve conter pelo menos:
     - 8 regras R5B14;
     - 4 regras R5B22;
     - descricao do contrato R5B16/R5B18;
     - referencia a `severity_policy.py` com status correto.

### Gates de aceite

- Toda regra oficial deve ter fonte rastreavel.
- Nao misturar regra historica, candidata e oficial sem coluna/status.
- O catalogo deve ser util para auditoria, apresentacao e implementacao.

## Fase 6 - Atualizar e implementar modulo de Graph Engineering investigativo (Concluído)

### Objetivo

Atualizar `docs/modulo_graph_feature_engineering.md` para o contexto R5B22 e implementar `backend/core/graph_engineering.py` como modulo investigativo pos-decisao. Este modulo nao deve atuar no tempo real da decisao transacional. Ele deve ser acionado apenas para transacoes intervencionadas (`CONFIRMAR` ou `BLOQUEAR`) e gerar relatorios incrementais para investigacao.

### Principio arquitetural

O modulo de grafo nao deve aumentar o SLA da API. O fluxo correto e:

```text
API retorna decisao
  se decisao em {CONFIRMAR, BLOQUEAR}:
      acionar GraphInvestigationEngine de forma pos-decisao
      gerar/anexar linha em CSV de investigacao
  se APROVAR:
      nao acionar por padrao
```

Na primeira versao, o acionamento pode ser sincrono opcional ou assíncrono simples. Em producao, preferir fila/evento.

### Tarefas de documentacao

1. Atualizar `docs/modulo_graph_feature_engineering.md`.
   - Remover/atualizar referencias a metricas antigas v3.0.5.
   - Explicar o papel atual:
     - modulo investigativo pos-decisao;
     - nao feature realtime obrigatoria;
     - foco em contas laranja, comunidades, fan-in/fan-out, ponte e relacoes suspeitas.

2. Documentar pre-requisitos de dados.
   - Origem/pagador:
     - `customer_id` ou `cd_cpf_pagador`.
   - Destino/recebedor:
     - `counterparty_id`, `cd_cpf_cnpj_recebedor` ou identificador equivalente.
   - Aresta:
     - `transaction_id`, `event_datetime`, `vl_pix`, `ds_chave_pix`, `ds_tipo_chave_norm`, decisao, label quando disponivel.

3. Documentar features investigativas.
   - Exemplos:
     - `graph_in_degree_receiver_1h`
     - `graph_in_degree_receiver_24h`
     - `graph_in_degree_receiver_7d`
     - `graph_out_degree_payer_24h`
     - `graph_unique_payers_to_receiver_24h`
     - `graph_unique_receivers_from_payer_24h`
     - `graph_receiver_total_amount_24h`
     - `graph_payer_total_amount_24h`
     - `graph_receiver_first_seen_age_hours`
     - `graph_is_new_receiver`
     - `graph_reciprocity_flag`
     - `graph_two_hop_shared_receiver_count`
     - `graph_suspected_mule_score`
     - `graph_bridge_account_score`
     - `graph_component_size`
     - `graph_known_fraud_neighbor_count`

4. Documentar saida do relatorio CSV.
   - Caminho sugerido:
     - `resultados/investigacao/graph_investigation_report.csv`
   - Uma linha por transacao intervencionada.
   - O arquivo deve ser incremental/apend-only.

### Tarefas de implementacao

1. Criar `backend/core/graph_engineering.py`.

2. Implementar classe principal.
   - Nome sugerido: `GraphInvestigationEngine`.
   - Responsabilidades:
     - receber transacao e decisao final;
     - atualizar memoria local opcional de arestas recentes;
     - calcular features de grafo possiveis com os dados disponiveis;
     - gerar linha de relatorio;
     - apendar em CSV.

3. Implementar entrada tolerante.
   - Aceitar `dict` da transacao original e `dict` de resultado da API/pipeline.
   - Normalizar campos:
     - `transaction_id`
     - `customer_id`
     - `counterparty_id`
     - `event_datetime`
     - `vl_pix`
     - `decisao`
     - `score_final`
     - `r5b22_rule_applied`
     - `r5b14_rule_applied`

4. Implementar calculos iniciais sem dependencias pesadas.
   - Usar `pandas`, `collections.deque`, `defaultdict`.
   - Evitar NetworkX no caminho inicial se nao for necessario.
   - Calcular janelas simples:
     - 1h;
     - 24h;
     - 7d quando houver historico suficiente.

5. Implementar scores investigativos simples.
   - `suspected_mule_score`:
     - alto numero de pagadores distintos para o mesmo recebedor;
     - recebedor novo;
     - valor acumulado alto;
     - alta taxa de intervencao associada ao recebedor.
   - `bridge_account_score`:
     - conta recebe e envia em janela curta, se houver dados de ambos os lados.
   - `fanout_score`:
     - pagador enviando para muitos recebedores novos.

6. Implementar persistencia CSV incremental.
   - Criar diretorio se nao existir.
   - Escrever header apenas se arquivo nao existir.
   - Usar lock simples ou escrita atomica se houver concorrencia.
   - Em caso de erro, nao quebrar a decisao da API; apenas logar falha investigativa.

7. Integrar opcionalmente ao pipeline/API.
   - Primeira opcao segura:
     - adicionar chamada opcional controlada por env var:
       - `GRAPH_INVESTIGATION_ENABLED=true`
       - `GRAPH_INVESTIGATION_REPORT_PATH=...`
   - Acionar apenas se `decisao in {"CONFIRMAR", "BLOQUEAR"}`.
   - Nao bloquear resposta da API se o modulo falhar.

8. Criar testes.
   - `tests/test_graph_engineering.py`.
   - Casos:
     - APROVAR nao gera linha quando configurado assim;
     - CONFIRMAR gera linha;
     - BLOQUEAR gera linha;
     - arquivo CSV e incremental;
     - sem `counterparty_id`, modulo gera relatorio com campos nulos e warning, sem exception fatal.

### Gates de aceite

- `backend/core/graph_engineering.py` nao deve alterar a decisao final.
- O modulo deve ser opt-in por configuracao.
- A API nao pode falhar por erro no relatorio de grafo.
- O relatorio deve conter informacao suficiente para analista investigar relacoes.

## Fase 7 - Atualizar proposta HBase para o baseline R5B22 (Concluído)

### Objetivo

Atualizar `docs/proposta_tecnica_hbase.md` para refletir exatamente as features usadas no baseline R5B22 e separar:

1. Features que chegam com a transacao.
2. Features geradas em tempo real.
3. Features que devem vir do Feature Store em HBase.
4. Features/sinais do contrato congelado e politicas.
5. Features investigativas de grafo, que podem ir para HBase/Hive mas nao sao obrigatorias para decisao realtime na primeira versao.

### Problema atual

O documento fala em 52 features e arquitetura anterior. O baseline R5B22 usa 78 features no aluno, politicas R5B14/R5B22 e sinais congelados do professor.

### Tarefas

1. Atualizar resumo executivo.
   - Versao alvo: `1.5.0-r5b22`.
   - Baseline oficial: `R5B22_OFFICIAL_CONSTRAINED_BASELINE`.
   - FPR global oficial: `0,957474%`.

2. Reclassificar features por origem.

   **Transacao/evento online:**
   - `vl_pix`
   - `ds_tipo_chave_norm`
   - `hour`
   - `periodo_dia`
   - `value_band`
   - `autcodret`
   - identificadores e timestamp.

   **Mobile/device/host online ou enriquecimento rapido:**
   - `latencia_rede_ms`
   - `tempo_processamento_host_ms`
   - `topaz_risk_score`
   - `mbk_completeness_score`
   - `mbk_available_flag`

   **Feature Store HBase - historico do pagador:**
   - contagens PIX 7d/30d/90d/180d;
   - valores totais 7d/30d/90d/180d;
   - maximos por dia;
   - maximo PIX 180d;
   - recebedores distintos.

   **Feature Store HBase - relacao pagador-recebedor:**
   - quantidade para mesmo recebedor por janela;
   - valor total para recebedor por janela;
   - primeiro envio;
   - dias desde primeiro envio;
   - dias desde ultima transacao;
   - recorrencia.

   **Feature Store HBase - historico do recebedor:**
   - quantidade recebida por janela;
   - valor recebido por janela;
   - pagadores distintos;
   - maximo de recebimentos por dia;
   - first receiver flag.

   **Runtime derivadas:**
   - ratios;
   - bins;
   - `lgbm_raw`;
   - `lgbm_mapped`;
   - `if_percentile`;
   - `se_score`;
   - `beh_score`;
   - `score_final`.

   **Contrato congelado/professor:**
   - `r4g_fast_frozen_decisao_recommended`;
   - `r5b14_rule_applied`;
   - `r5b14_layer_applied`;
   - campos `_frozen`;
   - bins do contrato.

3. Atualizar tabelas HBase propostas.
   - Manter ou revisar:
     - `fraud_detection:perfil_cliente`;
     - `fraud_detection:historico_trimestral`;
     - `fraud_detection:historico_recebedores`.
   - Adicionar ou revisar:
     - `fraud_detection:r5b22_contract_features`;
     - `fraud_detection:receiver_history`;
     - `fraud_detection:graph_investigation_features`.

4. Definir row keys.
   - Pagador:
     - `customer_id`/CPF normalizado.
   - Relacao pagador-recebedor:
     - `customer_id#counterparty_id`.
   - Recebedor:
     - `counterparty_id`.
   - Grafo:
     - por investigacao: `transaction_id`;
     - por no: `node_id`;
     - por par: `payer_id#receiver_id`.

5. Definir freshness/TTL.
   - Features transacionais: evento.
   - Historico 7d/30d/90d/180d: atualizacao horaria ou diaria conforme custo.
   - Contrato congelado: versionado por baseline.
   - Grafo investigativo: TTL maior para auditoria, ou armazenamento em Hive e sumario em HBase.

6. Atualizar SLA.
   - Lookup HBase deve ser tratado como componente de latencia.
   - Graph investigation nao entra no SLA da resposta transacional se for pos-decisao/async.

7. Adicionar secao de validacao.
   - Teste de completude de features.
   - Teste de freshness.
   - Teste de fallback quando HBase nao retorna linha.
   - Teste de drift semanal.

### Gates de aceite

- O documento nao deve mais afirmar que o modelo atual usa 52 features.
- Deve explicar claramente as 78 features do aluno R5B22.
- Deve separar realtime, HBase, runtime derivado, contrato congelado e grafo investigativo.
- Deve ser compativel com treino semanal e serving de baixa latencia.

## Ordem recomendada de implementacao

1. Fase 3 primeiro: atualizar feature contract documental.
2. Fase 5 em seguida: criar catalogo de regras oficial.
3. Fase 4: atualizar docs dos modulos com base nos dois contratos acima.
4. Fase 2: migrar treino semanal R5B22.
5. Fase 1: ajustar API e explicabilidade em cima do contrato ja estabilizado.
6. Fase 7: atualizar proposta HBase com feature contract definitivo.
7. Fase 6: implementar graph engineering investigativo como modulo opt-in.

## Validacao final do plano completo

Ao final das sete fases, executar no minimo:

```powershell
python -m py_compile backend\api.py backend\core\pipeline_orquestrador.py backend\core\decision_engine.py backend\core\graph_engineering.py backend\modelos\train_lgbm_distilled_r5b22.py
python -m pytest tests -q
python backend\scripts\simular_pipeline_e2e_v2.py --sample 2000
```

Se houver alteracao em politica, contrato de features, treino ou artefatos candidatos, tambem executar:

```powershell
python backend\scripts\simular_pipeline_e2e_v2.py --full --workers 4
```

Critérios de sucesso finais:

```text
scoring_config continua em 1.5.0-r5b22 ou versao posterior documentada
FPR global < 1%
Fraudes em APROVAR <= 5
Fraudes em CONFIRMAR <= 10
API retorna decisao e explicabilidade coerentes
Scripts de treino reproduzem aluno R5B22 ou geram candidato com gate claro
Docs refletem o baseline R5B22, nao versoes antigas
Graph engineering e opt-in e nao altera a decisao em tempo real
HBase doc separa corretamente origem das features
```
