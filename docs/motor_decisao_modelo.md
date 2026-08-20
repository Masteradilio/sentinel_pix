

# Relatório Técnico — Motor de Decisão Antifraude PIX 1.5.0-r5b22

**Projeto:** rebuild_pix | **Versão:** 1.5.0-r5b22 (Baseline Oficial R5B22)
**Data:** 12 de junho de 2026 | **Autor:** Adilio + AI Engineer
**Classificação:** Documento técnico interno

---

## 1. Resumo Executivo

O motor de decisão antifraude PIX é a engrenagem central coordenada pelo `PipelineOrquestrador`. No seu estado atual (R5B22), ele consolida o modelo de aprendizado de máquina destilado, um conjunto de regras estritas de mitigação de falso negativo (R5B14) e as heurísticas refinadas oficiais R5B22. O sistema é regido por flags ativáveis via `scoring_config.json`, permitindo escalabilidade determinística de `APROVAR`, `CONFIRMAR` e `BLOQUEAR`.

### Resultados Finais do Baseline Oficial (R5B22)

Base de validação: 113.844 transações (1.465 fraudes confirmadas).

| Métrica Global (Intervenção) | Valor |
|---|---|
| **TP** | 1.463 |
| **FP** | 1.076 |
| **FN** | 2 |
| **TN** | 111.303 |
| **Precision** | 57,62% |
| **Recall (Sensibilidade)** | 99,86% |
| **F1-Score** | 0,73 |
| **False Positive Rate** | 0,957% |

| Métrica Específica (Faixa BLOQUEAR) | Valor |
|---|---|
| **TP** | 1.453 |
| **FP** | 760 |
| **Precision** | 65,65% |
| **Recall** | 99,18% |
| **FN fora de BLOQUEAR** | 12 |

**Distribuição Operacional:**
- **APROVAR:** 111.305 transações | 2 fraudes | 111.303 normais
- **CONFIRMAR:** 326 transações | 10 fraudes | 316 normais
- **BLOQUEAR:** 2.213 transações | 1.453 fraudes| 760 normais

Esses números respeitam ativamente os "gates" de não regressão do baseline: FPR global abaixo de 1%, máximo de 5 fraudes escapando em `APROVAR` e 10 fraudes toleráveis em `CONFIRMAR`.

---

## 2. Arquitetura do Motor de Decisão

O pipeline de decisão (gerenciado via `PipelineOrquestrador` e `PixDecisionEngine`) obedece a um encadeamento determinístico onde flags oficiais e politicas atuam em cascata. O estado atual é pautado pela destilação do aprendizado.

```text
┌─────────────────────────────────────────────────────────────┐
│                    TRANSAÇÃO PIX (features_dict)             │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
              ┌──────────────────┐
              │ 1. Orquestrador  │  Enriquecimento de Sinais, 
              │    (Feature Eng) │  BEH Score, SE Score, IF, LGBM
              └───────┬──────────┘
                      ▼
              ┌──────────────────┐
              │ 2. Contrato      │  Avalia output e produz a base congelada:
              │    Professor     │  r4g_fast_frozen_decisao_recommended
              │    (R5B16/18)    │
              └───────┬──────────┘
                      ▼
              ┌──────────────────┐
              │ 3. Política      │  Aplica as 8 regras R5B14 de baixa
              │    R5B14         │  tolerância a falso negativo
              └───────┬──────────┘
                      ▼
              ┌──────────────────┐
              │ 4. Modelo Aluno  │  Utilizando os dados, o LGBM destilado
              │    Destilado     │  aproxima a decisão ideal (baseada no 
              │    (R5B22)       │  professor)
              └───────┬──────────┘
                      ▼
              ┌──────────────────┐
              │ 5. Política      │  O motor aplica as heurísticas finais
              │    Oficial R5B22 │  R5B22 (ex: DEMOTE_LAYER...) para gerir FPR
              └───────┬──────────┘
                      ▼
              ┌──────────────────┐
              │ 6. Decisão       │  APROVAR / CONFIRMAR / BLOQUEAR
              │    Final         │
              └──────────────────┘
```

A arquitetura oficial de validação em código (`backend/core/decision_engine.py`) obedece ativamente as seguintes flags operacionais globais em `scoring_config.json`:
- `r5b14_operational_zero_fn_enabled=true`
- `r5b16_frozen_contract_enabled=true`
- `r5b22_official_baseline_enabled=true`
- `official_baseline_policy=R5B22_OFFICIAL_CONSTRAINED_BASELINE`

---

## 3. Componentes do Motor

### 3.1 LightGBM (LGBM) — Modelo Supervisionado

**Papel:** Backbone do sistema. Gera a probabilidade base de fraude a partir de features transacionais. Na v3.0.5, o LGBM assumiu papel ainda mais central como **árbitro de confiança** — quando o LGBM discorda fortemente do IF, o LGBM prevalece.

| Parâmetro | Valor |
|---|---|
| Tipo | `LGBMClassifier` |
| Estimadores | 958 |
| Total de features | 52 |
| Output | Probabilidade raw $$p \in [0, 1]$$ |
| Threshold efetivo | $$p \geq 0.40$$ para flag inicial |
| Threshold original (F1-best) | $$p \geq 0.27$$ |

**Top 10 Features por Importância:**

| Rank | Feature | Importância | % Acumulado |
|---|---|---|---|
| 1 | `version` | 2.799 | 10,7% |
| 2 | `f30` | 2.605 | 20,6% |
| 3 | `f33` | 2.095 | 28,6% |
| 4 | `f34` | 1.988 | 36,2% |
| 5 | `f29` | 1.915 | 43,5% |
| 6 | `f31` | 1.766 | 50,2% |
| 7 | `f35` | 1.751 | 56,9% |
| 8 | `f44` | 1.337 | 62,0% |
| 9 | `f10` | 1.089 | 66,2% |
| 10 | `f17` | 822 | 69,3% |

> **Nota:** As features estão anonimizadas (`f29`–`f46`) por questões de segurança do dataset. As top-7 features concentram **56,9%** da importância total, indicando que o modelo se apoia fortemente em um subconjunto bem definido de sinais transacionais.

**Performance solo:**

$$
\text{LGBM}_{solo}: \text{TP}=352, \text{FP}=189, \text{FN}=3 \implies \text{Recall}=99{,}15\%, \text{Precision}=65{,}06\%
$$

O LGBM sozinho já é responsável pela vasta maioria da capacidade de detecção. As camadas subsequentes refinam a precisão e adicionam cobertura em edge cases. Na v3.0.5, o pipeline completo **supera o LGBM solo** em precision (68,87% vs 65,06%) enquanto mantém o mesmo recall — demonstrando que as camadas adicionais agora **adicionam valor líquido positivo**.

---

### 3.2 Isolation Forest (IF) — Modelo Não-Supervisionado

**Papel:** Detector de anomalias estruturais. Na v3.0.5, o IF opera em **modo consultivo** — seus sinais são considerados apenas quando o LGBM (que viu labels de treino) concorda ou não discorda fortemente.

| Parâmetro | Valor |
|---|---|
| Árvores | 800 |
| Features do scaler | 13 (fonte de verdade) |
| Output | Percentil de anomalia $$\in [0, 1]$$ |
| Papel no ensemble | Consultivo (veto-only para extremos, com gate LGBM) |
| Correlação com LGBM (geral) | 0,121 (baixa — sinais complementares) |
| Correlação com LGBM (fraudes) | 0,276 (moderada — convergem em fraudes claras) |

**Poder Discriminativo por Faixa de Percentil:**

| Percentil IF | Total | Fraudes | Taxa Fraude | Interpretação |
|---|---|---|---|---|
| < 50% | 50.164 | 6 | 0,012% | Normal — sem risco |
| 50–90% | 40.681 | 35 | 0,086% | Normal — risco residual |
| 90–95% | 4.945 | 28 | 0,57% | Levemente anômalo |
| 95–99% | 3.887 | 81 | 2,08% | Anômalo — **6× a taxa base** |
| 99–99,5% | 258 | 21 | 8,14% | **Muito anômalo** |
| 99,5–99,9% | 231 | 50 | 21,6% | **Altamente anômalo** |
| **99,9%+** | **189** | **134** | **70,9%** | **Extremamente anômalo** |

**Mudança de filosofia na v3.0.5:**

O IF tem excelente poder discriminativo nas faixas extremas, mas na v1.3 ele gerava FP ao ativar vetos em transações onde o LGBM discordava (score < 0,25). A nova regra é:

$$
\text{IF veto ativo} \iff \text{IF}_{percentile} \geq \theta_{veto} \land \neg\text{FastApprove}
$$

Onde Fast-Approve está ativo quando:

$$
\text{FastApprove} = (\text{LGBM}_{raw} < 0.25) \land (\text{SE} = 0) \land (\text{BEH} = 0)
$$

Isso preserva o poder discriminativo do IF para fraudes reais (onde LGBM concorda) e elimina os FP onde IF detectava anomalia estatística sem evidência supervisionada.

---

### 3.3 Cascade v3 — Regras de Alta Precisão

**Papel:** Regras determinísticas que capturam padrões de fraude com alta confiança. Na v3.0.5, a C3 recebeu um **LGBM guard** que eliminou 56 FP sem perder nenhum TP.

**Regras implementadas:**

| Regra | Condição | Ação | Ativações | TP | FP | Precision |
|---|---|---|---|---|---|---|
| **C1** — Burst ≥ 3 | $$\text{tx\_count\_30m} \geq 3$$ | BLOQUEAR | 48 | 48 | 0 | **100%** |
| **C3** — IF999 + Burst + LGBM | $$IF \geq 99{,}5\% \land \text{burst} \geq 1 \land \text{LGBM} \geq 0{,}35$$ | CONFIRMAR | 100 | 95 | 5 | **95,0%** |

**Evolução da Cascade C3:**

| Versão | Condição C3 | Ativações | TP | FP | Precision |
|---|---|---|---|---|---|
| v1.3 (Cascade v2) | IF ≥ 99,5% + burst | 156 | 95 | 61 | 60,9% |
| **v3.0.5 (Cascade v3)** | IF ≥ 99,5% + burst + **LGBM ≥ 0,35** | **100** | **95** | **5** | **95,0%** |

O LGBM guard eliminou **56 FP** (todas transações com LGBM < 0,35 onde o IF detectava anomalia estatística) sem perder nenhum dos 95 TP. A C3 passou de uma regra conservadora com precision de 61% para uma regra de **alta confiança** com precision de 95%.

A Cascade solo agora tem performance standalone impressionante:

$$
\text{Cascade}_{solo}: \text{TP}=143, \text{FP}=5, \text{Precision}=96{,}6\%, \text{Recall}=40{,}3\%
$$

---

### 3.4 Ensemble Score — Combinação LGBM + Cascade

**Papel:** Combina as saídas do LGBM e Cascade em um score unificado de 0 a 100. Na v3.0.5, o IF **não participa** do ensemble como boost aditivo — é consultado apenas pelas regras de veto (com gate de Fast-Approve).

$$
\text{score}_{ensemble} = \begin{cases}
\text{LGBM}_{raw} & \text{se } \text{LGBM}_{raw} \geq 0.40 \\
0.40 & \text{se Cascade triggered e } \text{LGBM}_{raw} < 0.40 \\
\text{LGBM}_{raw} & \text{caso contrário (sem boost IF)}
\end{cases}
$$

A distribuição final de scores mostra excelente **calibração**:

| Faixa | Total | Fraudes | FraudRate | Decisão |
|---|---|---|---|---|
| 0–10 | 96.624 | 0 | 0,000% | APROVAR |
| 10–30 | 2.253 | 0 | 0,000% | APROVAR |
| 30–50 | 523 | 2 | 0,382% | APROVAR |
| 50–60 | 43 | 0 | 0,000% | APROVAR |
| 60–70 | 290 | 1 | 0,345% | APROVAR |
| 70–77 | 70 | 0 | 0,000% | APROVAR |
| **77–85** | **108** | **5** | **4,63%** | **CONFIRMAR** |
| 85–95 | 40 | 0 | 0,000% | CONFIRMAR |
| **95–100** | **404** | **347** | **85,9%** | **BLOQUEAR** |

O score é **fortemente bimodal**: 96,3% das transações ficam abaixo de 10 (aprovação segura) e as fraudes se concentram massivamente acima de 95. A zona de CONFIRMAR (77–95) tem apenas ~148 transações, demonstrando que o modelo é decisivo — raramente deixa transações em zona cinza.

---

### 3.5 Social Engineering Score (SE)

**Papel:** Detecta padrões consistentes com golpes de engenharia social (ex: vítima sendo manipulada para fazer PIX).

| Métrica | Valor |
|---|---|
| Transações com SE > 0 | 579 (0,58% do total) |
| Fraudes com SE > 0 | **220 de 355** (62,0%) |
| Normais com SE > 0 | 359 |
| Média SE (fraudes) | 37,59 |
| Média SE (normais) | 0,16 |

**Distribuição por Faixa:**

| Faixa SE | Total | Fraudes | Normais | Precision |
|---|---|---|---|---|
| Zero | 99.776 | 135 | 99.641 | 0,14% |
| 1–39 | 59 | 47 | 12 | **79,7%** |
| 40–59 | 354 | 50 | 304 | 14,1% |
| 60+ | 166 | 123 | 43 | **74,1%** |

**Insight crítico:** O SE score é um discriminador fortíssimo nas extremidades. A faixa 1–39 tem **79,7% de precision** e a 60+ tem **74,1%**. A faixa 40–59 tem mais ruído (14,1%), mas é usada no veto cirúrgico (SE ≥ 40 + BEH ≥ 15 + valor alto + conta nova) com bom resultado.

$$
P(\text{fraude} | SE \geq 60) = 74{,}1\% \quad \text{vs} \quad P(\text{fraude} | SE = 0) = 0{,}14\%
$$

O SE score é ativado em apenas **0,58%** das transações, mas cobre **62%** das fraudes — um sinal extremamente eficiente. Na v3.0.5, o SE participa do mecanismo de Fast-Approve: quando SE = 0, é um sinal forte de que a transação **não** é engenharia social, reforçando a confiança na decisão do LGBM.

---

### 3.6 Behavioral Score (BEH)

**Papel:** Detecta comportamento anômalo baseado em dois sub-scores:
- **Velocity:** Frequência anormal de transações (explosão de atividade)
- **Age/Value:** Combinação de conta nova + valor alto (indicativo de conta mula ou vítima recente)

| Métrica | Valor |
|---|---|
| Transações com BEH > 0 | 1.208 (1,2% do total) |
| Fraudes com BEH > 0 | **198 de 355** (55,8%) |
| Normais com BEH > 0 | 1.010 |
| Média BEH (fraudes) | 16,72 |
| Média BEH (normais) | 0,19 |

**Distribuição por Faixa:**

| Faixa BEH | Total | Fraudes | Normais | Precision |
|---|---|---|---|---|
| Zero | 99.147 | 157 | 98.990 | 0,16% |
| 1–14 | 0 | 0 | 0 | — |
| 15–24 | 993 | 75 | 918 | 7,6% |
| 25–39 | 100 | 67 | 33 | **67,0%** |
| 40+ | 115 | 56 | 59 | **48,7%** |

**Fontes comportamentais:**

| Fonte | Ativações | Fraudes | Precision |
|---|---|---|---|
| Velocity | 130 | 80 | **61,5%** |
| Age/Value | 405 | 101 | **24,9%** |
| Ambos | 13 | 13 | **100%** |

**Insight:** Quando velocity **E** age/value disparam juntos, a precision é de **100%** — sinal inequívoco de fraude. Na v3.0.5, o BEH participa do Fast-Approve: quando BEH = 0, é evidência adicional de que o IF está detectando anomalia estatística, não fraude comportamental.

---

### 3.7 Fast-Approve Override (NOVO na v3.0.5)

**Papel:** Mecanismo de **desescalada** que impede o Isolation Forest de gerar vetos quando o modelo supervisionado (LGBM) discorda fortemente e nenhum sinal comportamental (SE, BEH) está presente.

**Condição de ativação:**

$$
\text{FastApprove} = (\text{LGBM}_{raw} < 0{,}25) \land (\text{SE} = 0) \land (\text{BEH} = 0)
$$

**Efeito:** Quando ativo, os seguintes vetos são suprimidos:
- Veto #2: LGBM + IF convergência → BLOQUEAR
- Veto #3: IF extremo + agravantes → BLOQUEAR

**Estatísticas validadas:**

| Métrica | Valor |
|---|---|
| Transações com Fast-Approve ativo | 98.680 (98,3% do total) |
| Fraudes no escopo | 2 (ambas **invisíveis a todos os componentes**) |
| FP evitados | ~20 (via supressão de vetos IF-based) |
| TP perdidos | **0** |

**Justificativa técnica:** O LGBM viu os labels de treino; o IF não. Quando o LGBM diz "não é fraude" com alta confiança (p < 0,25) e nenhum sinal independente (SE, BEH) contradiz, o IF não deve ter autoridade para override. As 2 fraudes no escopo do Fast-Approve são as mesmas 2 fraudes **invisíveis a todos os componentes** (`invisible_to_all = 2`) — elas seriam APROVAR com ou sem o Fast-Approve.

---

### 3.8 Sistema de Regras — Contrato R5B14 e R5B22

**Papel:** O R5B22 orquestra um mecanismo de override com heurísticas seguras para prevenir falso negativo (via R5B14) e controlar falso positivo sem perder bloqueios genuínos (via demotions R5B22).

**Regras R5B14 (Prevenção de Falso Negativo Ativa):**

| Regra R5B14 | Ação/Intervenção |
|---|---|
| `R5B14_CTB_01_LGBM_RAW_HIGH` | Elevação severa ao ultrapassar limiar preditivo puro |
| `R5B14_CTB_02_SCORE_2_3_LGBM_R4_HIGH` | Contenção baseada em score final R4 vs Raw High |
| `R5B14_CTB_03_SCORE_2_3_LGBM_R4_MED` | Contenção baseada em score final R4 vs Raw Med |
| `R5B14_CTB_04_DOC_PHONE_HIGH_PAYER_COUNT` | Gatilho via telefone/documento + fan-out pagador alto |
| `R5B14_CTB_05_OUTROS_RATIO_MAX_HIGH` | Gatilho de proporção de valor vs máximo histórico |
| `R5B14_ATB_01_DOC_PHONE_MORNING_SCORE_HIGH` | Gatilho matutino para telefone/documento score alto |
| `R5B14_ATB_02_NIGHT_SCORE_1_2_RATIO_HIGH` | Gatilho noturno vs ratio outlier |
| `R5B14_CTA_01_LOW_LGBM_RAW_COMPENSATION` | Desescalonamento para score puro ínfimo |

**Regras Oficiais R5B22 (Redução de Falsos Positivos Ativa):**

| Regra R5B22 | Ação de Demotion (Suavização) |
|---|---|
| `DEMOTE_LAYER_APPROVE_TO_BLOCK_TO_APROVAR` | Suaviza severidade restrita não respaldada |
| `DEMOTE_LAYER_CONFIRM_TO_BLOCK_TO_CONFIRMAR` | Reverte o escalonamento severo sem indícios |
| `DEMOTE_CAT2_ds_tipo_chave_norm_OUTROS__lgbm_bin_lgbm_0.05_0.1` | Desescalona para CONFIRMAR na zona cinza estatística |
| `DEMOTE_CAT2_value_band_E_5000_10000__lgbm_bin_lgbm_0.05_0.1` | Desescalona valor E / zona cinza estatística para CONFIRMAR |

Essas heurísticas combinadas promovem a calibragem fina que a decisão algorítmica destilada herda.

---

## 4. Thresholds de Decisão

| Threshold | Score | Decisão | Ação |
|---|---|---|---|
| **BLOQUEAR** | $$\text{score} \geq 95$$ | Rejeição automática | Transação bloqueada sem intervenção humana |
| **CONFIRMAR** | $$77 \leq \text{score} < 95$$ | Revisão manual | Transação retida para análise de um operador |
| **APROVAR** | $$\text{score} < 77$$ | Aprovação automática | Transação liberada |

**Justificativa dos thresholds:**
- O threshold de BLOQUEAR em 95 garante precision de **85,89%** — a cada ~6 bloqueios, ~5 são fraudes reais.
- O threshold de CONFIRMAR em 77 recupera fraudes borderline (scores 77-80) ao custo moderado de FP direcionados para revisão manual — um tradeoff aceitável dado que CONFIRMAR não rejeita, apenas direciona para humano.

**Alternativa analisada:** Com threshold CONFIRMAR em 80, o pipeline teria precision de **77,85%** com apenas 99 FP, mas perderia 4 fraudes (FN=7 vs FN=3). A decisão de manter 77 prioriza recall.

---

## 5. Distribuição Final de Decisões

| Decisão | Total | % do Total | Fraudes | Normais | FraudRate |
|---|---|---|---|---|---|
| **APROVAR** | 99.844 | 99,49% | 3 | 99.841 | 0,003% |
| **CONFIRMAR** | 107 | 0,11% | 5 | 102 | 4,67% |
| **BLOQUEAR** | 404 | 0,40% | 347 | 57 | 85,89% |

**Impacto financeiro:**
- **Fraude bloqueada automaticamente:** ~R$ 2.170.187 (347 transações)
- **Fraude direcionada para revisão:** ~R$ 24.690 (5 transações)
- **Fraude perdida (FN):** R$ 3.166,76 (3 transações)
- **Taxa de recuperação financeira:**

$$
\frac{2.170.187 + 24.690}{2.170.187 + 24.690 + 3.167} = 99{,}86\%
$$

O sistema intercepta **99,86% do valor financeiro** das fraudes.

---

## 6. Evolução entre Versões

| Versão | Descrição | TP | FP | FN | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|
| **v0** | LGBM solo (th=0.35) | 352 | 211 | 3 | 99,15% | 62,52% | 0,7669 |
| **v1** | LGBM solo (th=0.40) | 352 | 189 | 3 | 99,15% | 65,06% | 0,7857 |
| **v1.2** | Ensemble + Cascade v2 + Vetos (th=80) | 348 | 171 | 7 | 98,03% | 67,05% | 0,7963 |
| **v1.3** | Ensemble + th=77 + veto cirúrgico | 352 | 207 | 3 | 99,15% | 62,97% | 0,7702 |
| **v3.0.5** | **Cascade v3 + Fast-Approve + th=77** | **352** | **159** | **3** | **99,15%** | **68,87%** | **0,8129** |

**Análise da evolução:**

1. **v0 → v1:** Ajuste de threshold LGBM de 0.35→0.40 eliminou 22 FP sem perder recall. Melhoria pura de precision.

2. **v1 → v1.2:** Ensemble completo com threshold 80 trocou recall por precision: perdeu 4 fraudes (FN de 3→7) mas ganhou melhor precision (67%) e F1 (0.7963). Revelou que threshold 80 era conservador demais.

3. **v1.2 → v1.3:** Redução do threshold para 77 + veto cirúrgico recuperou as 4 fraudes: FN voltou de 7→3, recall subiu para 99,15%. Custo: +36 FP (171→207) para fila CONFIRMAR.

4. **v1.3 → v3.0.5:** Duas mudanças cirúrgicas eliminaram 48 FP sem perder recall:
   - **Cascade C3 com LGBM guard (≥ 0,35):** Eliminou 56 FP da C3 (de 61→5), precision C3 de 61%→95%.
   - **Fast-Approve override:** Suprimiu ~20 vetos IF-based em transações onde LGBM < 0,25 + SE=0 + BEH=0.
   - Resultado líquido: **-48 FP, 0 FN**, F1 subiu de 0,770 para **0,813**.

**A v3.0.5 é a melhor versão em todas as métricas relevantes simultaneamente** — é a primeira versão que supera o LGBM solo em precision (68,9% vs 65,1%) enquanto mantém recall idêntico (99,15%).

---

## 7. Análise dos Falsos Negativos (Fraudes Perdidas)

Três fraudes permanecem indetectáveis pelo motor — classificadas como **FN irredutíveis**:

| # | Índice | Valor | Idade | Relacionamento | LGBM | IF | SE | BEH | Score Final |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 13814 | R$ 188,82 | 27 | 41 meses | 0,023 | 28,9% | 0 | 0 | 36,5 |
| 2 | 90197 | R$ 498,96 | 45 | 317 meses | 0,455 | 66,7% | 0 | 0 | 71,3 |
| 3 | 98992 | R$ 2.478,98 | 4 | 27 meses | 0,148 | 88,3% | 0 | 0 | 62,3 |

**Por que são irredutíveis:**

- **Todas têm SE=0 e BEH=0:** Nenhum sinal comportamental ou de engenharia social foi ativado.
- **Nenhuma tem burst:** `tx_count_prev_30m = 0` em todas — transação única e isolada.
- **Valores moderados a baixos:** R$ 189 a R$ 2.479 — dentro da faixa normal de PIX.
- **FN #90197** é o mais "próximo" — score 71,3, apenas 5,7 pontos abaixo do threshold. LGBM deu 0,455 (moderado). Reduzir threshold para 71 adicionaria dezenas de FP.
- **FN #13814** é o mais "invisível" — LGBM deu apenas 0,023 (praticamente zero), IF está no percentil 28,9%. Indistinguível de transação legítima com features disponíveis.
- **FN #98992** tem idade=4 anos, o que é incomum, mas sem outros sinais não é suficiente para flag.

**Valor total perdido:** R$ 3.166,76 — representando apenas **0,14%** do valor total de fraudes no dataset.

**Caminho para melhoria:** Essas fraudes provavelmente requerem features adicionais não disponíveis neste dataset (ex: device fingerprint, geolocalização, histórico de destino) ou mecanismos de feedback em tempo real.

---

## 8. Análise dos Falsos Positivos

| Métrica | v1.3 | v3.0.5 | Delta |
|---|---|---|---|
| Total de FP | 207 | **159** | **-48** |
| FP em CONFIRMAR | 130 | **102** | -28 |
| FP em BLOQUEAR | 77 | **57** | **-20** |

**Fontes de FP (v3.0.5):**

| Origem do FP | Quantidade | Mudança vs v1.3 |
|---|---|---|
| LGBM (≥ effective threshold) | 189 | inalterado |
| Cascade C1 | 0 | inalterado |
| Cascade C3 | **5** | **-56** 🔥 |
| SE (score > 0) | 359 | inalterado |
| BEH (score > 0) | 1.010 | inalterado |
| Veto BLOQUEAR | **46** | -20 |
| Veto CONFIRMAR | **53** | -22 |

> **Nota:** As contagens por fonte representam transações normais onde cada componente ativou; não somam ao total de FP do pipeline porque uma transação pode ter múltiplos sinais mas só conta como 1 FP se a decisão final for CONFIRMAR ou BLOQUEAR.

**Impacto das mudanças v3.0.5:**

| Mudança | FP eliminados | Mecanismo |
|---|---|---|
| **C3 LGBM guard ≥ 0,35** | **56** (de 61→5) | Suprime C3 quando LGBM < 0,35 |
| **Fast-Approve override** | **~20** (via vetos) | Suprime vetos IF-based quando LGBM < 0,25 + SE=0 + BEH=0 |
| **IF extremo gate 0,05→0,25** | incluso no FA | Reforça consistência do gate LGBM |
| **Net (com overlap)** | **-48** | — |

**Interpretação:** Os FP restantes (159) têm perfil de **transações genuinamente anômalas** — são transações legítimas com padrão estatisticamente semelhante ao de fraudes. O sistema erra "do lado seguro", e a v3.0.5 reduziu significativamente os erros causados por **divergência entre modelos** (IF flaggando anomalias que o LGBM sabia serem legítimas).

---

## 9. Métricas Operacionais

### Carga de Revisão Manual

$$
\text{Taxa CONFIRMAR} = \frac{107}{100.355} = 0{,}107\%
$$

A cada **10.000 transações**, apenas **~11** são direcionadas para revisão humana (vs ~13 na v1.3). Desses 11:
- ~0,5 são fraudes reais
- ~10,5 são falsos alertas

Isso representa uma carga operacional **extremamente baixa** e viável para uma equipe de análise enxuta. A redução de 135→107 transações em CONFIRMAR na v3.0.5 representa **-21% de carga operacional**.

### Taxa de Bloqueio Automático

$$
\text{Taxa BLOQUEAR} = \frac{404}{100.355} = 0{,}40\%
$$

A cada 10.000 transações, **~40 são bloqueadas automaticamente** com precision de **85,89%** (vs 81,84% na v1.3).

### Comparativo Operacional

| Métrica | v1.3 | v3.0.5 | Melhoria |
|---|---|---|---|
| Transações em CONFIRMAR | 135 | **107** | -21% |
| Transações em BLOQUEAR | 424 | **404** | -5% |
| Precision BLOQUEAR | 81,84% | **85,89%** | +4,1pp |
| FP em BLOQUEAR (bloqueios indevidos) | 77 | **57** | **-26%** |

---

## 10. Contribuição Marginal dos Componentes

| Componente | Detectou | Exclusivo | Incremental sobre LGBM |
|---|---|---|---|
| **LGBM** | 352 | 106 | — |
| **Cascade** | 143 | 0 | 0 |
| **SE** | 220 | 0 | 1 |
| **BEH** | 198 | 0 | 1 |
| **Qualquer componente** | 353 | — | — |
| **Invisíveis a todos** | 2 | — | — |

**Interpretação:** O LGBM é o backbone absoluto do sistema — detecta 352 das 355 fraudes sozinho. Cascade, SE e BEH são **redundantes com o LGBM** para a maioria dos casos, mas agregam valor como:

1. **Rede de segurança:** Se o LGBM falhar em produção (drift, adversarial), as outras camadas mantêm cobertura parcial.
2. **Agravantes e vetos:** Mesmo sem TP exclusivo, os sinais de SE e BEH enriquecem a decisão para o analista humano em CONFIRMAR.
3. **Veto cirúrgico:** O veto SE ≥ 40 + BEH ≥ 15 + valor alto + conta nova recuperou 1 FN específico na v1.3.
4. **Fast-Approve evidence:** SE=0 e BEH=0 são sinais de ausência de fraude que o Fast-Approve usa para calibrar a confiança.

---

## 11. Conclusões e Recomendações

### Pontos Fortes

1. **Recall excepcional (99,15%):** O sistema captura praticamente todas as fraudes, com apenas 3 FN irredutíveis de baixo valor.
2. **Precision recorde (68,87%):** A v3.0.5 é a primeira versão onde o pipeline completo supera o LGBM solo em precision.
3. **F1 = 0,813:** Melhor F1 de todas as versões, demonstrando que recall e precision foram otimizados simultaneamente.
4. **Arquitetura com checks-and-balances:** O Fast-Approve impede que o IF (não-supervisionado) override o LGBM (supervisionado) sem evidência corroborativa.
5. **Cascade C3 com 95% precision:** O LGBM guard transformou uma regra ruidosa em regra de alta confiança.
6. **Carga operacional mínima:** Apenas 0,107% das transações requerem revisão manual (-21% vs v1.3).
7. **Recuperação financeira de 99,86%:** O valor perdido (R$ 3.167) é insignificante frente ao total interceptado (~R$ 2,2M).

### Limitações Conhecidas

1. **Feature names anonimizadas:** As features f29–f46 dificultam interpretabilidade para stakeholders não-técnicos.
2. **FN irredutíveis sem sinais comportamentais:** Os 3 FN têm SE=0 e BEH=0, sugerindo que features adicionais seriam necessárias para capturá-los.
3. **Dataset de validação único:** Métricas baseadas em um snapshot; validação contínua em produção é essencial.
4. **Fast-Approve scope amplo:** 98,3% das transações são cobertas pelo FA — é correto (LGBM < 0,25 + SE=0 + BEH=0 = transações claramente legítimas), mas requer monitoramento em produção.

### Próximos Passos Recomendados

1. **Aplicar patch v3.0.5 no `decision_engine.py`:** Sincronizar o motor de decisão de produção com a simulação validada.
2. **Monitoramento de drift em produção:** Implementar tracking de distribuição de scores, taxa de Fast-Approve, e taxa de vetos ao longo do tempo.
3. **Avaliar features adicionais:** Device fingerprint, geolocalização, e grafo de destinatários podem capturar os FN irredutíveis.
4. **Feedback loop:** Usar os resultados de revisão manual (CONFIRMAR) para retreino do LGBM.
5. **Stress test adversarial:** Validar que o Fast-Approve não cria vulnerabilidades exploráveis (ex: atacante que mantém LGBM < 0,25 deliberadamente).

---

*Relatório gerado em 12/04/2026 — Motor de Decisão Antifraude PIX v3.0.5*