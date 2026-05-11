# 🛠️ FASE 3 — Consolidação Operacional Pós-FASE 2

> **Objetivo:** transformar o baseline pós-C1 em uma versão estável, testável, reproduzível e segura para evolução futura, sem depender de novos casos de fraude.

## 1. Contexto

A FASE 2 reduziu os FNs de 9 para 8 sem aumentar FP, promovendo apenas a regra cirúrgica `C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER`. O LGBM v6.2 foi rejeitado para runtime, a regra R2 foi rejeitada, e o meta-learner shadow não encontrou candidato seguro adicional.

Portanto, o foco imediato não deve ser criar novas regras ou treinar novos modelos com os mesmos sinais. A prioridade passa a ser consolidar tecnicamente o sistema, garantir que as melhorias não regridam e preparar o pipeline para absorver novos dados quando eles estiverem disponíveis.

## 2. Baseline oficial da FASE 3

O baseline oficial de entrada da FASE 3 é o baseline pós-C1:

| Seed | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 347 | 14 | 8 | 96,1219% | 97,7465% | 0,9693 |
| 123 | 347 | 12 | 8 | 96,6574% | 97,7465% | 0,9720 |

Configuração conceitual:

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
````

## 3. Objetivos da FASE 3

A FASE 3 não busca reduzir FN diretamente. Ela busca garantir que o modelo atual seja confiável, auditável e pronto para evolução.

Objetivos:

1. consolidar o patch C1 de forma limpa no runtime;
2. criar testes de regressão para impedir perda de performance;
3. criar uma fonte única da verdade para métricas oficiais;
4. documentar as regras, thresholds e decisões rejeitadas;
5. preparar o projeto para reavaliação futura com novos dados;
6. reduzir risco operacional antes de qualquer nova modelagem.

## 4. EXP-008A — Regression Suite Pós-C1

**Categoria:** Testes / Validação
**Complexidade:** 🟡 Média
**Prioridade:** 🔴 Alta
**Objetivo:** criar uma suíte de regressão automática para garantir que o baseline pós-C1 não seja quebrado por mudanças futuras.

### Escopo

Criar testes automatizados para:

* `threshold_confirmar = 62`;
* `threshold_bloquear = 95`;
* `lgbm_guard_enabled = true`;
* `lgbm_guard_threshold = 0.30`;
* `guard_exception_alto_valor_se_beh_enabled = true`;
* `exp006f_c1_enabled = true`;
* rejeição do EXP-003 residual;
* rejeição do LGBM v6.2 para runtime;
* rejeição da R2;
* consistência seed 42 e seed 123.

### Testes mínimos

1. **Teste da C1**

   * Transação `E0000020820260205003505340630525` deve sair como `CONFIRMAR`.
   * Deve conter `exp006f_c1_applied = True`.
   * Deve preservar `decisao_original_exp006f_c1 = APROVAR`.

2. **Teste da V1 Guard Contextual**

   * Caso de alto valor recuperado pelo EXP-004-FINAL deve continuar como `CONFIRMAR`.

3. **Teste anti-R2**

   * Casos que eram FP no EXP-006C/R2 não devem ser promovidos.

4. **Teste de métricas oficiais**

   * Seed 42 deve manter `TP=347`, `FP=14`, `FN=8`.
   * Seed 123 deve manter `TP=347`, `FP=12`, `FN=8`.

### Critério de aceite

```text
pytest tests/test_regression_post_fase2.py
```

deve passar integralmente.

A FASE 3 não pode ser considerada concluída enquanto a suíte de regressão não proteger C1, V1, guard rail e thresholds oficiais.

## 5. EXP-008B — Validation Report Pós-FASE 2

**Categoria:** Documentação / Governança
**Complexidade:** 🟢 Baixa
**Prioridade:** 🔴 Alta
**Objetivo:** criar um relatório oficial de validação do baseline pós-FASE 2.

### Artefato

Criar:

```text
docs/VALIDATION_REPORT_POST_FASE2.md
```

### Conteúdo obrigatório

1. dataset utilizado;
2. número de transações;
3. número de fraudes;
4. baseline pré-FASE 2;
5. baseline pós-C1;
6. deltas por experimento;
7. experimentos promovidos;
8. experimentos rejeitados;
9. métricas seed 42 e seed 123;
10. lista dos 8 FNs residuais;
11. interpretação dos FNs como provavelmente limitados pelos dados atuais;
12. próximos passos dependentes de novos sinais.

### Decisão oficial documentada

```text
Modelo oficial: baseline pós-FASE 2 com C1 habilitada.
LGBM v6.2: rejeitado para runtime.
Meta-learner: diagnóstico apenas, sem promoção.
Próxima melhoria de recall: depende de novos dados/sinais.
```

## 6. EXP-008C — Rules Catalog e Decision Trace

**Categoria:** Explicabilidade / Auditoria
**Complexidade:** 🟡 Média
**Prioridade:** 🟠 Média-Alta
**Objetivo:** documentar todas as regras, exceções e guard rails ativos no motor de decisão.

### Artefato

Criar:

```text
docs/RULES_CATALOG.md
```

### Conteúdo obrigatório

Para cada regra ativa:

* nome;
* objetivo;
* condição;
* ação;
* origem experimental;
* status;
* risco operacional;
* flag de configuração;
* critério de desligamento.

### Regras que devem constar

1. threshold de confirmação;
2. threshold de bloqueio;
3. guard rail LGBM;
4. V1 Guard Contextual;
5. C1 Near-Threshold;
6. regras SE ativas;
7. regras BEH ativas;
8. regras rejeitadas e motivo da rejeição.

### Exemplo de entrada

```markdown
## C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER

Status: Ativa  
Origem: EXP-006E / EXP-006F  
Flag: exp006f_c1_enabled  
Ação: APROVAR → CONFIRMAR

Condição:
- decisão atual = APROVAR
- first_receiver_flag = 1
- pix_key_random_flag = 0
- relacionamento <= 12 meses
- 100 <= valor < 500
- 0.06 <= lgbm_raw < 0.10
- 58 <= score_final < 62
- se_score <= 0
- beh_score <= 0

Evidência:
- recuperou 1 FN nos seeds 42 e 123
- adicionou 0 FP
- perdeu 0 TP

Risco:
- regra estreita, mas dependente de score near-threshold
- deve ser monitorada em produção
```

## 7. EXP-008D — Cleanup Técnico dos Patches

**Categoria:** Refatoração / Manutenibilidade
**Complexidade:** 🟡 Média
**Prioridade:** 🔴 Alta
**Objetivo:** remover wrappers temporários e consolidar C1/V1 no ponto correto do runtime.

Durante a FASE 2, alguns patches foram aplicados de forma incremental para validar hipóteses rapidamente. A FASE 3 deve limpar essa implementação.

### Objetivo técnico

Mover a lógica definitiva da C1 para um ponto estável e controlado do `DecisionEngine` ou do pipeline final, evitando wrappers defensivos espalhados.

### Resultado esperado

A C1 deve existir como:

* método claro;
* função testável;
* flag configurável;
* logging explícito;
* motivo de decisão rastreável;
* sem duplicação em múltiplos arquivos.

### Critério de aceite

1. `decision_engine.py` compila;
2. `pipeline_orquestrador.py` compila;
3. `simular_pipeline_e2e_v2.py` compila;
4. teste da transação C1 passa;
5. seed 42 e seed 123 mantêm métricas pós-C1;
6. nenhum wrapper temporário desnecessário permanece ativo.

## 8. EXP-008E — Experiment Manifest e Versionamento de Artefatos

**Categoria:** MLOps Light
**Complexidade:** 🟢 Baixa
**Prioridade:** 🟠 Média
**Objetivo:** versionar artefatos e decisões sem exigir infraestrutura pesada.

### Artefatos

Criar:

```text
backend/artefatos/MANIFEST_MODEL.json
resultados/experimentos/EXPERIMENT_INDEX.md
```

### MANIFEST_MODEL.json

Deve conter:

```json
{
  "model_version": "post_fase2_c1",
  "decision_engine_version": "v3.0.5_post_c1",
  "active_lgbm": "baseline_producao",
  "rejected_lgbm_candidates": [
    "LGBM_C_SPW_2_0X"
  ],
  "active_rules": [
    "V1_GUARD_CONTEXTUAL",
    "C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER"
  ],
  "disabled_rules": [
    "EXP003_RESIDUAL",
    "R2_LOW_VALUE_GRAY_FIRST_RECEIVER"
  ],
  "official_metrics": {
    "seed_42": {
      "TP": 347,
      "FP": 14,
      "FN": 8,
      "F1": 0.9693
    },
    "seed_123": {
      "TP": 347,
      "FP": 12,
      "FN": 8,
      "F1": 0.9720
    }
  }
}
```

### EXPERIMENT_INDEX.md

Deve resumir:

* EXP-001;
* EXP-002;
* EXP-003;
* EXP-004-FINAL;
* EXP-005A;
* EXP-005B;
* EXP-006 a EXP-006F;
* EXP-007A;
* decisão de promoção/rejeição de cada experimento.

## 9. Critério de aceite da FASE 3

A FASE 3 é concluída quando:

* C1 está consolidada sem wrappers experimentais desnecessários;
* há suíte de regressão pós-C1;
* há relatório oficial de validação;
* há catálogo de regras;
* há manifesto de artefatos;
* seed 42 e seed 123 reproduzem o baseline oficial;
* qualquer mudança futura no engine passa por teste automatizado.

---

# 📈 FASE 4 — Observabilidade, Drift e Feedback Operacional

> **Objetivo:** preparar o sistema para operação contínua e coleta de evidências futuras, sem depender ainda de novos casos de fraude rotulados.

## 1. Contexto

Como os 8 FNs residuais parecem depender de novos sinais, a próxima evolução precisa ser orientada por observabilidade: entender como o modelo se comporta em novas transações, quais decisões ficam em zona cinza e quais sinais sofrem drift.

A FASE 4 não busca treinar novo modelo. Ela cria a infraestrutura para detectar quando o modelo começa a envelhecer ou quando surge oportunidade de melhoria.

## 2. EXP-009A — Decision Logging Estruturado

**Categoria:** Observabilidade
**Complexidade:** 🟡 Média
**Prioridade:** 🔴 Alta
**Objetivo:** registrar, para cada decisão, os sinais e motivos que levaram a `APROVAR`, `CONFIRMAR` ou `BLOQUEAR`.

### Campos mínimos

Para cada transação, registrar:

* transaction_id;
* timestamp;
* customer_id anonimizado;
* valor;
* decisão final;
* score_final;
* lgbm_raw;
* lgbm_mapped;
* if_percentile;
* se_score;
* beh_score;
* regras aplicadas;
* guard rails aplicados;
* flags como `exp006f_c1_applied`;
* motivo textual da decisão;
* versão do modelo;
* versão do scoring_config.

### Critério de aceite

Para uma execução E2E curta, 100% das decisões devem possuir:

```text
decision_id
model_version
score_final
decisao
decision_reason
```

## 3. EXP-009B — Drift Monitor Offline

**Categoria:** Monitoramento
**Complexidade:** 🟡 Média
**Prioridade:** 🟠 Média-Alta
**Objetivo:** criar um monitor de drift offline comparando a distribuição dos dados recentes com o dataset de validação.

### Features monitoradas

* vl_pix;
* log_vl_pix;
* qt_tempo_relacionamento_mes;
* nr_idade;
* lgbm_raw;
* if_percentile;
* se_score;
* beh_score;
* score_final;
* first_receiver_flag;
* pix_key_random_flag;
* proporção de APROVAR/CONFIRMAR/BLOQUEAR;
* taxa de acionamento da V1;
* taxa de acionamento da C1.

### Saída

```text
resultados/monitoramento/drift_report_YYYYMMDD.md
```

### Alertas

Gerar alerta quando:

* score_final tiver drift relevante;
* lgbm_raw mudar distribuição;
* C1 disparar muito acima do esperado;
* V1 disparar muito acima do esperado;
* taxa de CONFIRMAR subir sem justificativa;
* taxa de APROVAR subir em segmentos historicamente suspeitos.

## 4. EXP-009C — Zona Cinza e Fila de Revisão Humana

**Categoria:** Active Learning / Operação
**Complexidade:** 🟡 Média
**Prioridade:** 🟠 Média
**Objetivo:** criar uma fila de casos informativos para revisão humana futura, sem alterar o modelo ainda.

### Critérios para fila

Entram na fila:

* score_final entre 55 e 62;
* LGBM em zona cinza;
* IF alto isolado;
* first_receiver com valor fora do padrão;
* casos com discordância entre LGBM, IF, SE e BEH;
* casos com C1 quase acionada;
* casos com V1 quase acionada;
* casos APROVAR com score shadow alto no EXP-007A.

### Saída

```text
resultados/active_learning/review_queue.csv
```

### Campos

* transaction_id;
* decisão atual;
* score_final;
* score_gap_to_confirmar;
* motivo de entrada na fila;
* sinais principais;
* recomendação de revisão;
* prioridade.

### Uso futuro

Essa fila servirá como fonte de amostras para:

* auditoria humana;
* coleta de labels;
* seleção de novos casos para retreino;
* análise de novos padrões de fraude.

## 5. EXP-009D — Painel de Métricas Operacionais

**Categoria:** Dashboard / Governança
**Complexidade:** 🟢 Baixa-Média
**Prioridade:** 🟠 Média
**Objetivo:** criar um painel offline com métricas operacionais do modelo.

### Métricas

* volume total de transações;
* taxa de APROVAR;
* taxa de CONFIRMAR;
* taxa de BLOQUEAR;
* distribuição de score_final;
* top motivos de decisão;
* taxa de acionamento da C1;
* taxa de acionamento da V1;
* distribuição por valor;
* distribuição por idade de relacionamento;
* evolução de drift;
* volume em fila de revisão.

### Artefatos

```text
resultados/dashboard_operacional/model_monitoring.parquet
resultados/dashboard_operacional/model_monitoring_summary.md
```

## 6. Critério de aceite da FASE 4

A FASE 4 é concluída quando:

* decisões são logadas com versão e motivo;
* drift report offline é gerado;
* fila de revisão humana é criada;
* painel operacional possui métricas mínimas;
* existe mecanismo para capturar evidências futuras sem alterar o modelo.

---

# 🧪 FASE 5 — Robustez, Simulação e Testes de Segurança

> **Objetivo:** testar a estabilidade do modelo atual contra cenários adversos e variações plausíveis, sem exigir novos casos reais de fraude.

## 1. Contexto

Mesmo sem novos casos, ainda é possível avaliar a robustez do pipeline com simulações controladas. A ideia não é inventar fraude sintética para treinar modelo, mas testar se o engine reage de forma coerente a variações plausíveis.

## 2. EXP-010A — Perturbation Testing dos FNs e TPs

**Categoria:** Robustez / Sensibilidade
**Complexidade:** 🟡 Média
**Prioridade:** 🟠 Média
**Objetivo:** aplicar pequenas perturbações nos FNs, TPs e FPs atuais para entender a sensibilidade do engine.

### Perturbações

Testar variações em:

* valor da transação;
* tempo de relacionamento;
* first_receiver_flag;
* pix_key_random_flag;
* if_percentile;
* lgbm_raw;
* se_score;
* beh_score;
* score_final.

### Perguntas

* Quanto o score precisa mudar para um FN virar CONFIRMAR?
* Quais FNs estão mais próximos da fronteira?
* Quais TPs são frágeis e poderiam virar FN?
* Quais FPs são causados por margem pequena?

### Saída

```text
resultados/experimentos/EXP-010A/perturbation_report.md
```

## 3. EXP-010B — Adversarial Scenario Testing

**Categoria:** Segurança / Red Team
**Complexidade:** 🟡 Média-Alta
**Prioridade:** 🟠 Média
**Objetivo:** simular alterações plausíveis que um fraudador poderia tentar para reduzir o score.

### Exemplos

* dividir valor em múltiplas transações menores;
* usar recebedor não aleatório;
* reduzir valor para fugir de regra;
* esperar alguns dias para aumentar relacionamento;
* usar padrões que não acionem SE/BEH;
* escolher valores próximos aos FNs residuais.

### Importante

Esses cenários não devem ser usados para treinar o modelo como fraude real. Eles servem para identificar fragilidade e criar requisitos para novas features.

## 4. EXP-010C — Cost Model Offline

**Categoria:** Negócio / Risco
**Complexidade:** 🟢 Baixa-Média
**Prioridade:** 🟠 Média
**Objetivo:** criar um modelo de custo para comparar impacto de FN e FP.

### Métricas

* valor total dos FNs;
* valor total dos FPs;
* custo estimado de revisão;
* custo estimado de fraude perdida;
* benefício líquido de cada regra;
* custo por ponto de recall;
* custo operacional de confirmar transação legítima.

### Saída

```text
resultados/cost_model/cost_report.md
```

## 5. EXP-010D — Explainability Pack

**Categoria:** Explicabilidade
**Complexidade:** 🟡 Média
**Prioridade:** 🟠 Média
**Objetivo:** gerar um pacote de explicabilidade para auditoria interna.

### Conteúdo

* explicação dos TPs;
* explicação dos FPs;
* explicação dos FNs;
* regras que mais impactam decisão;
* exemplos de C1 e V1;
* exemplos de casos data-limited;
* limitações conhecidas.

## 6. Critério de aceite da FASE 5

A FASE 5 é concluída quando:

* há relatório de sensibilidade dos FNs/FPs;
* há cenários adversariais documentados;
* há modelo de custo offline;
* há pacote de explicabilidade;
* o sistema tem riscos conhecidos documentados antes de nova coleta de dados.

---

# 🧱 FASE 6 — Preparação para Reavaliação com Novos Dados

> **Objetivo:** preparar o pipeline para receber novos dados normais e fraudulentos sem improviso.

## 1. Contexto

Antes de trazer novos dados, é preciso definir exatamente como eles serão ingeridos, validados, versionados e comparados com o baseline atual. Essa fase ainda pode ser feita sem novos casos.

## 2. EXP-011A — Data Contract para Novas Transações

**Categoria:** Engenharia de Dados
**Complexidade:** 🟡 Média
**Prioridade:** 🔴 Alta
**Objetivo:** criar um contrato de dados para garantir que novas bases possam ser comparadas com a base atual.

### O contrato deve definir

* colunas obrigatórias;
* tipos esperados;
* campos de data/hora;
* identificadores anonimizados;
* label de fraude;
* status de contestação;
* status de MED;
* dados do recebedor;
* dados de dispositivo/sessão, se disponíveis;
* critérios de exclusão;
* regras de deduplicação;
* janela temporal.

### Artefato

```text
docs/DATA_CONTRACT_NOVOS_DADOS.md
```

## 3. EXP-011B — Labeling Protocol

**Categoria:** Governança de Label
**Complexidade:** 🟡 Média
**Prioridade:** 🔴 Alta
**Objetivo:** definir o que será considerado fraude e o que será considerado transação normal na próxima rodada.

### Regras de label

Documentar:

* fraude confirmada;
* suspeita não confirmada;
* contestação;
* MED;
* devolução;
* falso positivo operacional;
* decisão humana;
* tempo mínimo de maturação do label;
* conflitos entre fontes.

### Risco principal

Não treinar com labels imaturos. Uma transação aparentemente normal hoje pode virar contestação dias depois.

## 4. EXP-011C — Dataset Versioning

**Categoria:** MLOps / Dados
**Complexidade:** 🟢 Baixa-Média
**Prioridade:** 🟠 Média
**Objetivo:** definir versionamento para datasets antigos e novos.

### Estrutura sugerida

```text
dados/
├── raw/
│   ├── pix_2026Q1/
│   └── pix_2026Q2/
├── processed/
│   ├── base_treino_final_v1.parquet
│   ├── base_treino_final_post_fase2.parquet
│   └── base_treino_final_novos_dados_v1.parquet
└── manifests/
    ├── dataset_v1.json
    └── dataset_novos_dados_v1.json
```

## 5. EXP-011D — Backtest Harness

**Categoria:** Validação
**Complexidade:** 🟡 Média
**Prioridade:** 🔴 Alta
**Objetivo:** criar um harness único para rodar o modelo em qualquer nova base e comparar contra o baseline pós-FASE 2.

### Entrada

```text
python scripts/run_backtest_modelo_pix.py --dataset dados/processed/nova_base.parquet
```

### Saída

```text
resultados/backtests/<dataset_version>/
├── metrics_global.json
├── metrics_by_seed.csv
├── fn_census.csv
├── fp_census.csv
├── decision_trace.parquet
└── recommendation.md
```

## 6. Critério de aceite da FASE 6

A FASE 6 é concluída quando:

* existe contrato de dados;
* existe protocolo de labels;
* existe versionamento de datasets;
* existe harness de backtest;
* o projeto está pronto para receber novos casos sem quebrar metodologia.

---

# 🔁 FASE 7 — Reavaliação Completa com Novos Casos de Fraude e Novas Transações Normais

> **Objetivo:** reavaliar o modelo inteiro com novos dados reais, do treinamento aos ajustes finos, considerando novas transações normais e novos casos confirmados de fraude.

## 1. Contexto

Esta é a única fase que depende de novos dados. Ela deve começar apenas quando houver nova base com:

* novas transações normais;
* novos casos confirmados de fraude;
* labels suficientemente maduros;
* preferencialmente novos sinais de recebedor, grafo, dispositivo, sessão ou MED/contestação.

A FASE 7 não é apenas um retreino. É uma revalidação completa do pipeline.

## 2. Objetivos

1. medir se o baseline pós-FASE 2 generaliza em dados novos;
2. identificar novos FNs e FPs reais;
3. verificar se C1 e V1 continuam úteis;
4. recalibrar thresholds se houver drift;
5. retreinar LGBM somente se houver ganho real;
6. reavaliar IF, SE, BEH e meta-learner;
7. decidir se um novo modelo deve substituir o baseline atual.

## 3. EXP-012A — Backtest do Baseline Pós-FASE 2 em Dados Novos

**Categoria:** Validação externa
**Complexidade:** 🟡 Média
**Prioridade:** 🔴 Alta
**Objetivo:** rodar o baseline atual em novos dados sem qualquer retreino.

### Perguntas

* O recall se mantém?
* O FP aumenta?
* C1 dispara em frequência aceitável?
* V1 continua útil?
* Os FNs novos parecem similares aos antigos?
* Há novos padrões de fraude?

### Critério de aceite

Se o baseline pós-FASE 2 generalizar bem:

```text
Recall próximo ao validado
FP controlado
sem explosão de C1/V1
F1 não degrada materialmente
```

Se não generalizar, iniciar recalibração.

## 4. EXP-012B — Censo dos Novos FNs e FPs

**Categoria:** Diagnóstico
**Complexidade:** 🟡 Média
**Prioridade:** 🔴 Alta
**Objetivo:** repetir a cartografia de erros com os novos dados.

### Saídas

```text
resultados/experimentos/EXP-012B/
├── fn_census_new_data.csv
├── fp_census_new_data.csv
├── fn_clusters_new_data.csv
├── fp_clusters_new_data.csv
└── diagnosis.md
```

### Classificações

* erro por drift;
* erro por ausência de sinal;
* erro por recebedor novo;
* erro por dispositivo/sessão;
* erro por valor baixo;
* erro por fraude comportamental;
* erro por regra excessivamente conservadora;
* erro por label problemático.

## 5. EXP-012C — Retreino LGBM com Dados Novos

**Categoria:** Modelagem supervisionada
**Complexidade:** 🔴 Alta
**Prioridade:** 🟠 Média-Alta
**Objetivo:** retreinar o LGBM apenas se o backtest mostrar oportunidade real.

### Regras

Não retreinar automaticamente. Retreinar somente se:

* houver novos FNs com sinal tabular;
* houver aumento relevante de drift;
* o baseline pós-FASE 2 perder recall;
* os novos dados aumentarem diversidade de fraude.

### Critério de seleção

O novo LGBM só pode ser promovido se, no `DecisionEngine` real:

* reduzir FN;
* não aumentar FP acima do limite;
* validar em janela temporal posterior;
* preservar ou melhorar F1;
* não depender de leakage;
* não piorar segmentos sensíveis.

## 6. EXP-012D — Recalibração do DecisionEngine

**Categoria:** Calibração
**Complexidade:** 🟡 Média
**Prioridade:** 🔴 Alta
**Objetivo:** recalibrar thresholds e guard rails com os novos dados.

### Itens a reavaliar

* `threshold_confirmar`;
* `threshold_bloquear`;
* `lgbm_guard_threshold`;
* V1 Guard Contextual;
* C1 Near-Threshold;
* Fast Approve;
* pesos de LGBM/IF/SE/BEH;
* thresholds internos de SE/BEH.

### Regra

Qualquer recalibração deve passar por:

```text
artifact-only → model-only → quick-E2E → final-E2E
```

Sem grids longos no E2E.

## 7. EXP-012E — Meta-Learner com Novos Dados

**Categoria:** Meta-modelagem
**Complexidade:** 🔴 Alta
**Prioridade:** 🟠 Média
**Objetivo:** reavaliar o meta-learner apenas se houver novos FNs com separabilidade nos sinais.

### Critério

O meta-learner só avança se:

* recuperar FN novo;
* FP adicional for controlado;
* validação temporal confirmar;
* explicabilidade for coerente;
* não substituir o engine explicável sem fallback.

## 8. EXP-012F — Decisão de Promoção

**Categoria:** Governança
**Complexidade:** 🟡 Média
**Prioridade:** 🔴 Alta
**Objetivo:** decidir se o baseline pós-FASE 2 será mantido ou substituído.

### Possíveis decisões

1. manter baseline pós-FASE 2;
2. apenas recalibrar thresholds;
3. promover novo LGBM;
4. promover nova regra cirúrgica;
5. promover novo meta-learner como shadow ou componente auxiliar;
6. exigir novas fontes de dados antes de nova tentativa.

### Critério mínimo de promoção

Qualquer nova versão deve:

* reduzir FN ou reduzir FP sem piorar o outro;
* validar em janela temporal independente;
* manter explicabilidade;
* preservar rastreabilidade;
* passar na suíte de regressão;
* atualizar `VALIDATION_REPORT`;
* atualizar `MANIFEST_MODEL`.

## 9. Critério de encerramento da FASE 7

A FASE 7 é concluída quando houver uma destas decisões:

### Cenário A — Baseline atual generaliza bem

Manter baseline pós-FASE 2 e continuar monitoramento.

### Cenário B — Recalibração suficiente

Promover novos thresholds/regras, sem retreino.

### Cenário C — Novo modelo superior

Promover nova versão do LGBM/meta-modelo, com validação temporal robusta.

### Cenário D — Dados ainda insuficientes

Documentar que os novos casos continuam não separáveis com os sinais atuais e priorizar novas fontes.

## 10. Resultado esperado da FASE 7

Resultado realista:

* confirmar robustez do baseline pós-FASE 2;
* reduzir mais 1 a 3 FNs se os novos dados trouxerem sinal útil;
* manter FP dentro do limite operacional;
* atualizar o pipeline com base em evidência real.

Resultado ideal:

* FN ≤ 5;
* FP controlado;
* Recall ≥ 98,6%;
* F1 ≥ 0,9720;
* melhoria validada em dados temporalmente novos.

