# Workflow Oozie de Treinamento Contínuo (CT) - Baseline R5B22

Este documento descreve a infraestrutura End-to-End de **Integração Contínua de Machine Learning (CT/MLOps)** para o motor antifraude PIX, configurado para o ambiente Apache Oozie. A esteira engloba desde a ingestão de dados brutos e feature engineering no Data Lake até a validação rigorosa e publicação dos artefatos do modelo destilado no HDFS.

Todos os scripts pertencentes a este workflow estão centralizados no diretório `/treinamento`.

Diferente de modelos de ML convencionais que treinam rotulando diretamente se uma transação é fraude (`is_fraud`), o motor oficial R5B22 utiliza uma técnica de **Distillation Training**. O "Aluno" (LGBM) aprende a imitar as decisões consolidadas de um "Professor" (Baseline R5B18 combinado com a política rígida de Segurança R5B14).

---

## 1. Frequência e Mecânica Geral

- **Frequência do Workflow Oozie:** Semanal (ex: todos os domingos à meia-noite).
- **Frequência da API (Client):** Diária (sincronização de binários do HDFS).
- **Objetivos:** 
  1. Extrair os dados da semana mais recentes de normais e fraudes.
  2. Processar e enriquecer as features (HBase/HDFS features em janela de 180 dias).
  3. Re-destilar o modelo preservando o rigor do Baseline para lidar com fraudes evolutivas.
  4. Avaliar métricas em holdout.
  5. Promover os artefatos se passarem pelos gates de segurança.

---

## 2. Diagrama Macro do Workflow Oozie

```text
  ┌─────────────────────────────────────────────────────────┐
  │                   WORKFLOW OOZIE (SEMANAL)              │
  │                                                         │
  │  [FASE 1] Ingestão e Preparação de Dados                │
  │    ├─ 1.1 Extração Qualificada de Normais (HQL)         │
  │    ├─ 1.2 Amostragem e Enriquecimento MBK (HQL/PySpark) │
  │    ├─ 1.3 Merge com Fraudes e Dataset V2                │
  │    └─ 1.4 Feature Engineering V3 (Agregações 180d)      │
  │                                                         │
  │                          ▼                              │
  │                                                         │
  │  [FASE 2] Teacher Annotation (PySpark Action)           │
  │    └─ 01_anotar_contrato_professor.py                   │
  │       (Anota o dataset com labels de política rigorosa) │
  │                                                         │
  │                          ▼                              │
  │                                                         │
  │  [FASE 3] Treino Distilado (Shell/PySpark Action)       │
  │    └─ 02_train_distilled_r5b22.py                       │
  │       (Treina Intervention e Block Models)              │
  │                                                         │
  │                          ▼                              │
  │                                                         │
  │  [FASE 4] Gatekeeper de Qualidade (Shell Action)        │
  │    └─ 03_validar_promocao_r5b22.py                      │
  │       (Valida Holdout: Recall >= 97% | FPR <= 2%)       │
  │       [Exit Code 0 = SUCESSO] | [Exit Code 1 = FALHA]   │
  │                                                         │
  │                          ▼ (Se SUCESSO)                 │
  │                                                         │
  │  [FASE 5] HDFS Publishing (Shell Action)                │
  │    └─ 04_publish_hdfs.sh                                │
  │       (Envia .joblibs e .json para o HDFS oficial)      │
  └─────────────────────────────────────────────────────────┘
```

---

## 3. Detalhamento das Etapas (Nodes)

### FASE 1: Ingestão e Preparação de Dados (Feature Engineering)
Esta é a fase inicial pesada em dados, composta por diversos scripts SQL (Hive) orquestrados sequencialmente:
*   **A. Preparação da Base Normal:**
    *   `tb_pix_normais_qualified_raw_180d_v1.hql`: Faz o filtro sanitário excluindo transações nulas, devolvidas ou de teste.
    *   `tb_pix_normais_qualified_sample_180d_v1.hql`: Realiza a amostragem estratificada visando manter o balanceamento da infraestrutura de treino (economizando RAM e CPU).
    *   `tb_pix_normais_qualified_sample_mbk_180d_v1.hql`: Efetua o Join de dados de telemetria mobile (latência, ip, device) do sistema Mobile Banking (MBK).
    *   `tb_pix_normais_dataset_ready_v1.hql`: Consolida a base final de normais limpa.
*   **B. Consolidação (Dataset V2):**
    *   `tb_pix_dataset_v2_180d_v1.hql`: Empilha as tabelas de fraudes certificadas (`ingestao_fraudes_corrigida.py`) com os normais (do item A).
*   **C. Engenharia de Features Avançadas (Dataset V3):**
    *   `tb_pix_dataset_v3_daily_agg_180d_v1.hql` e `tb_pix_dataset_v3_features_180d_v1.hql`: Calculam médias, somas (fan-out, receiver velocity) e contagens em janelas deslizantes usando partições de 180 dias.
    *   `tb_pix_dataset_v3_target_180d_v1.hql`: Centraliza a saída contendo as 78 features exigidas, gravando a tabela no diretório alvo final do HDFS para ser consumido pelo modelo.

### FASE 2: Teacher Annotation (`01_anotar_contrato_professor.py`)
Lê a tabela HDFS V3 gerada na Fase 1 e submete essas transações (já enriquecidas com as features completas) à política mestre histórica. O script injeta os rótulos pseudo-reais que o LGBM Aluno deve emular: `contract_intervention` e `contract_block`. Esta etapa obriga o modelo a replicar o baseline das políticas R5B18 e mitigadoras do Falso Positivo (R5B14/R5B22).

### FASE 3: Treinamento Distilado (`02_train_distilled_r5b22.py`)
Utiliza os splits gerados pelas partições. Consome estritamente as 78 features homologadas.
Esta etapa treina simultaneamente os dois classificadores LGBM:
1. Modelo de Intervenção (`APROVAR` vs `CONFIRMAR/BLOQUEAR`)
2. Modelo de Bloqueio Rígido (`BLOQUEAR` vs `APROVAR/CONFIRMAR`)
A saída gera os binários `.joblib`, um CSV com Feature Importances, e o crucial arquivo JSON (`metricas_r5b22_distilled.json`) contendo as métricas de inferência na fatia de Holdout.

### FASE 4: Gatekeeper de Qualidade (`03_validar_promocao_r5b22.py`)
**A Caixa de Segurança.** Lê o JSON gerado no passo anterior e valida dinamicamente contra limiares flexibilizados que permitem absorver novos comportamentos sem perder controle de negócio:
*   **Limiares exigidos:** Recall (Intervenção) >= 97% e FPR (Falsos Positivos) <= 2%.
*   Se os números ficarem piores do que essas barreiras de contenção (ex. FPR estourar pra 5%), o script encerra com `exit 1`. O Oozie lê o status de falha, aciona o Kill Node, interrompe a promoção e pode enviar um alerta (e.g., e-mail) à equipe de Prevenção de Fraudes e Cientistas de Dados (MLOps) sobre a ocorrência de drástico Data Drift.

### FASE 5: Publicação no HDFS (`04_publish_hdfs.sh`)
Se a Fase 4 rodar sem erros (`exit 0`), este Shell Script faz um `hdfs dfs -put -f` enviando os arquivos compilados (`.joblib`, `.json`, `.csv`) para o path oficial de hospedagem:
`hdfs:///modelos_ml/nudan/nudan_hmo/anomalia_pix/artefatos/`

---

## 4. O Fluxo de Atualização Transacional da API (Consumer)

O ambiente transacional de scoring (a API rodando em Kubernetes/VMS na porta onde ocorrem as validações do PIX) nunca faz o download dos modelos na requisição de 100ms. 

A sincronização de atualização ocorre offline por um Cron nativo no servidor da API, seguindo esta rotina diária:

1. **Agendamento Cron (ex. 04:00 AM):** Executa localmente o `treinamento/sync_api_artifacts.sh`.
2. **Download Rápido:** A máquina realiza `hdfs dfs -get -f` baixando da pasta de artefatos do HDFS direto para `/backend/artefatos/`.
3. **Reinicialização Lógica:** O script finaliza solicitando o graceful restart do servidor ASGI (ex. uvicorn/gunicorn `sudo systemctl restart ...`).
4. **Hot-Swap em Memória:** Ao reiniciar, o evento `startup` da API recarrega os Joblibs em RAM, mantendo a operação daquele dia protegida pelo modelo atualizado com os últimos padrões de fraude, mas segurado pelo Baseline R5B22.
