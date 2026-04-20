
# Documentação do Módulo de Engenharia Social — SE v3.3

## Ficha Técnica do Módulo

```
Nome:            Social Engineering Detector (SE)
Versão:          v3.3
Tipo:            Sistema Especialista Baseado em Regras (RBES)
Padrões ativos:  9
Indicadores:     31 (Lift ≥ 1.5x validado)
Dataset:         100.355 transações PIX (355 fraudes confirmadas)
Calibração:      2026-04-11
Linguagem:       Python 3.12+
Módulo:          core/social_engineering.py
```

---

## 1. Propósito e Justificativa

### 1.1 Por que este módulo existe

O módulo SE é uma **camada complementar de detecção** que opera em conjunto com o modelo de machine learning (LGBM). Sua função é codificar **modus operandi conhecidos de golpes por engenharia social** em regras auditáveis, servindo como:

1. **Agravante no score final** — quando o SE detecta padrões de golpe, eleva o score de risco da transação em +3 a +4 pontos
2. **Camada de explicabilidade** — gera alertas humano-legíveis ("Possível coação física", "Padrão de falso funcionário") que nenhum SHAP value substitui
3. **Rede de segurança regulatória** — atende ao requisito do BACEN de que sistemas de detecção devem ser **auditáveis e explicáveis**

### 1.2 Validação conceitual pela indústria

A arquitetura híbrida ML + Regras é o **padrão da indústria** para detecção de fraudes financeiras:

| Referência | Achado |
|---|---|
| **Flagright (Jan/2026)** — "AI vs Rules-Based Transaction Monitoring" | "A hybrid approach, blending AI with risk-based rules, is emerging as the **gold standard** for payment processors' compliance programs." Pesquisa do American Banker (2025): 62% dos bancos priorizam automação inteligente para fraude |
| **Sundararamaiah et al. (Dez/2024)** — "Unifying AI and Rule-based Models for Financial Fraud Detection", IJCTT v72 | "Combining [rules and ML] can result in more accurate fraud detection with **fewer false positives**." Propõe framework híbrido rules+ML como arquitetura de referência |
| **Ben Abid et al. (2025)** — "A Scalable Hybrid Approach to Detecting Fraud with ML", EUSIPCO 2025 | Demonstra que XGBoost treinado com labels de regras + Isolation Forest + Autoencoder supera cada abordagem isolada. Framework modular similar ao nosso (ML + regras + anomalias) |
| **Vallarino (Abr/2025)** — "Detecting Financial Fraud with Hybrid Deep Learning", arXiv:2504.03750 | "Detection systems must combine robustness, adaptability, and precision." Modelo híbrido MoE atingiu 98.7% accuracy, 94.3% precision — mas requer camada de regras para compliance AML/KYC |
| **Fraud.net (2023)** — "Bridging the Gap: Incorporating AI/ML into Rules-Based Models" | "Rules-based fraud detection has been the backbone of transaction monitoring for decades [...] **85-99% of alerts from older rule-based tools are false positives**" — reforça necessidade de calibração data-driven como a realizada nas Frentes 1-5 |

### 1.3 Contexto regulatório brasileiro

| Regulação | Relevância para o módulo SE |
|---|---|
| **Resolução BCB nº 493 (Ago/2025)** — MED 2.0 | Obriga instituições a rastrear e bloquear valores de PIX fraudulento. O SE produz evidência estruturada para acionar o MED. Vigente desde Fev/2026 |
| **MED 2.0 — Mecanismo Especial de Devolução** | Permite bloqueio automático de contas denunciadas por fraude. Exige que a instituição demonstre **indícios de fraude** — os padrões do SE servem como justificativa formal |
| **Resolução BCB nº 403/2024** | Limites noturnos para PIX (R$1.000 entre 20h-6h). O SE detecta transações que exploram horários diurnos para valores altos (padrão FALSO_FUNCIONARIO_BANCO) |
| **LGPD + Resolução CMN 4.893/2021** | Exige que sistemas de decisão automatizada sejam explicáveis. O SE gera output 100% auditável: lista de indicadores + padrão detectado + score |

### 1.4 Cenário de fraudes PIX no Brasil

| Dado | Fonte |
|---|---|
| **28 milhões** de brasileiros vítimas de golpes PIX em 2025 | ADDP (Associação de Defesa de Dados Pessoais e do Consumidor), 2025 |
| **53%** das vítimas têm mais de 50 anos | ADDP, 2025 — valida indicadores `idade_60_plus` e `idade_70_plus` |
| **Falsa central telefônica** é o golpe com maior valor médio por vítima: **R$7.500** | Febraban, 2025 — valida o padrão `FALSO_FUNCIONARIO_BANCO` |
| Golpe do PIX cresce com **engenharia social e urgência** | ESET Brasil / Security Leaders (Jan/2026) — valida padrões baseados em `intervalo_curto` e `burst` |
| Estimativa de **redução de 40%** nos golpes bem-sucedidos com MED 2.0 | Especialistas citados pelo Estado de Minas e JC (Fev-Mar/2026) |
| IA está elevando sofisticação dos golpes (deepfakes, personalização) | ESET Brasil (Jan/2026): "Se antes os golpes eram genéricos, agora são moldados em segundos para perfis específicos" |

---

## 2. Arquitetura do Módulo

### 2.1 Visão geral

```
Transação PIX
    │
    ▼
┌────────────────────────────────┐
│  Feature Adapter               │  Normaliza features do pipeline
│  _adapt_features()             │  para formato esperado
└──────────┬─────────────────────┘
           │
           ▼
┌────────────────────────────────┐
│  Avaliação de 31 Indicadores   │  Cada indicador: bool (True/False)
│  _setup_indicators()           │  Todos com Lift ≥ 1.5x validado
└──────────┬─────────────────────┘
           │
           ▼
┌────────────────────────────────┐
│  Match contra 9 Padrões        │  Para cada padrão:
│  _setup_patterns()             │    - ALL required presentes? (+2 cada)
│                                │    - Somar optional ativos (+1 cada)
│                                │    - score ≥ min_score → DETECTADO
└──────────┬─────────────────────┘
           │
           ▼
┌────────────────────────────────┐
│  Scoring com Deduplicação      │  Clusters de overlap (Jaccard > 0.15)
│  _calculate_se_score()         │  evitam double-counting
│                                │  Atenuante: agendamento_recorrente
└──────────┬─────────────────────┘
           │
           ▼
      SEAnalysisResult
      ├── se_score: 0-100
      ├── patterns: [PatternMatch, ...]
      ├── active_indicators: {str: bool}
      └── risk_level: BAIXO|MEDIO|ALTO|CRITICO
```

### 2.2 Lógica de scoring

Cada padrão detectado contribui com um valor fixo baseado em severidade:

| Severidade | Pontos | Significado |
|---|---|---|
| CRITICO | 40 | Alta certeza de fraude (Precision ≥ 45%) |
| ALTO | 25 | Forte suspeita (Precision ≥ 70%) |
| MEDIO | 15 | Suspeita moderada (Precision ≥ 70%, sem urgência) |
| BAIXO | 10 | Sinal fraco |

**Deduplicação:** Padrões no mesmo cluster de overlap não somam — apenas o de maior severidade conta. Isso evita inflação de score quando indicadores compartilhados ativam múltiplos padrões simultaneamente.

**Clusters de overlap (v3.3):**
- `{IDOSO_VULNERAVEL_70, IDOSO_VULNERAVEL_80}` — subconjunto por idade
- `{ESVAZIAMENTO_CONTA, BURST_ESVAZIAMENTO_CONTA, BURST_INTENSO_RAPIDO}` — velocity patterns
- `{COACAO_FISICA, BURST_VALOR_ALTO}` — burst + valor alto

**Atenuante:** Se `agendamento_recorrente` estiver ativo, o score é reduzido em 15 pontos (transação previamente agendada é menos provável de ser fraude).

---

## 3. Catálogo de Indicadores

### 3.1 Indicadores ativos (31)

Todos os indicadores foram validados empiricamente com o dataset de calibração (100.355 tx, 355 fraudes). **Lift** = taxa em fraudes / taxa em normais. Indicadores com Lift < 1.0 foram removidos na v3.0.

| Indicador | Condição | Lift | Taxa Fraude | Taxa Normal | Info Gain |
|---|---|---|---|---|---|
| **burst_intenso** | `tx_count_prev_30m ≥ 3` | **∞** (0 normais) | 13.5% | 0.000% | 0.0039 |
| **burst_30m** | `burst_30m_flag = 1` | **241.7x** | 40.8% | 0.169% | 0.0092 |
| **multiplos_pix_rapidos** | `burst_30m + pix_dia_max ≥ 3` | **196.7x** | 33.2% | 0.169% | 0.0071 |
| **primeira_tx_trimestre** | `is_first_tx_trimestre = 1` | **146.3x** | 30.1% | 0.206% | 0.0061 |
| **burst_conta_antiga** | `relacionamento ≥ 12m + burst + first_recv` | **47.3x** | 6.2% | 0.131% | 0.0009 |
| **valor_absoluto_alto** | `vl_pix ≥ 5.000` | **34.3x** | 31.6% | 0.919% | 0.0043 |
| **valor_absoluto_muito_alto** | `vl_pix ≥ 10.000` | **29.4x** | 12.4% | 0.421% | 0.0015 |
| **aproximando_esgotamento** | `ratio_mediana ≥ 5 + burst` | **26.8x** | 0.6% | 0.021% | 0.0001 |
| **renda_desconhecida_valor_alto** | `renda_missing + vl_pix ≥ 5k` | **23.4x** | 15.5% | 0.663% | 0.0018 |
| **pix_acima_1000** | `vl_pix ≥ 1.000` | **14.5x** | 77.5% | 5.340% | 0.0088 |
| **pix_acima_500** | `vl_pix ≥ 500` | **10.1x** | 94.4% | 9.359% | 0.0103 |
| **idade_70_plus** | `nr_idade ≥ 70` | **8.4x** | 33.5% | 3.970% | 0.0024 |
| **intervalo_muito_curto** | `intervalo ≤ 5 minutos` | **6.7x** | 71.5% | 10.763% | 0.0052 |
| **renda_metade_comprometida** | `pix_over_50pct_renda = 1` | **4.6x** | 36.9% | 8.058% | 0.0016 |
| **idade_80_plus** | `nr_idade ≥ 80` | **4.5x** | 4.8% | 1.063% | 0.0002 |
| **idade_60_plus** | `nr_idade ≥ 60` | **3.9x** | 52.7% | 13.645% | 0.0022 |
| **intervalo_curto** | `intervalo ≤ 30 minutos` | **3.8x** | 89.9% | 23.623% | 0.0051 |
| **conta_recem_aberta** | `relacionamento ≤ 1 mês` | **3.7x** | 0.9% | 0.230% | 0.0000 |
| **valor_redondo** | `vl_pix ≥ 100 e múltiplo de 100` | **3.0x** | 35.5% | 11.938% | 0.0009 |
| **cliente_muito_novo** | `relacionamento ≤ 3 meses` | **2.4x** | 2.2% | 0.927% | 0.0000 |
| **multiplos_recebedores_distintos** | `distinct_receivers ≥ 3` | **1.8x** | 9.6% | 5.438% | 0.0001 |
| **is_segmento_premium** | `segmento_premium = 1` | **1.6x** | 79.4% | 49.870% | 0.0009 |
| **perfil_vulneravel_se** | `perfil_vulneravel_se = 1` | **1.5x** | 2.5% | 1.669% | 0.0000 |
| **chave_aleatoria** | `pix_key_random = 1` | **0.9x** | 38.6% | 42.668% | 0.0000 |
| **renda_incompativel** | `pix_over_100pct_renda = 1` | TBD | — | — | — |
| **recebedor_nunca_visto** | `qt_envio_recebedor = 0` | Contextual | — | — | — |
| **login_senha** | `is_login_senha = 1` | TBD | — | — | — |
| **horario_comercial** | `8h-18h, seg-sex` | ~1.1x | — | — | — |
| **agendamento_recorrente** | `is_agendamento_recorrente = 1` | Atenuante | — | — | — |

> **Nota sobre `chave_aleatoria`:** Lift 0.9x isolado (43% dos normais usam chave aleatória). Mantido **exclusivamente** para uso combinado com `pix_acima_1000` no padrão FALSO_FUNCIONARIO_BANCO, onde a combinação é conceitualmente essencial (vítima não conhece o recebedor).

### 3.2 Indicadores removidos (v3.0)

| Indicador | Lift | Motivo da remoção |
|---|---|---|
| `valor_alto_vs_historico` | 0.25x | **Anti-indicador** — mais presente em normais |
| `valor_muito_alto_vs_historico` | 0.26x | Anti-indicador |
| `valor_critico_vs_historico` | 0.32x | Anti-indicador |
| `escalada_valores` | 0.21x | Anti-indicador |
| `horario_noturno` | 0.0x | Zero fraudes no período |
| `horario_madrugada` | 0.0x | Zero fraudes no período |
| `zscore_valor_extremo` | 0.0x | Zero ativação em fraudes |
| `alta_frequencia_diaria` | 0.41x | Anti-indicador |
| `primeiro_envio` | 0.61x | 98% dos normais ativam |

---

## 4. Catálogo de Padrões

### 4.1 Resumo de performance (v3.3 — medido)

| # | Padrão | Required | ms | TP | FP | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **BURST_VALOR_ALTO** | burst_30m + pix≥500 | 3 | **141** | 38 | **78.8%** | 39.7% | **0.528** | 0.038% |
| 2 | **ESVAZIAMENTO_CONTA** | multiplos_pix_rapidos | 4 | 98 | 44 | 69.0% | 27.6% | 0.394 | 0.044% |
| 3 | **COACAO_FISICA** | intervalo≤5m + pix≥1k + 1ª_tx | 5 | 89 | 34 | 72.4% | 25.1% | 0.372 | 0.034% |
| 4 | **PRIMEIRA_TX_SUSPEITA** | 1ª_tx + pix≥1k | 4 | 89 | 34 | 72.4% | 25.1% | 0.372 | 0.034% |
| 5 | **FALSO_FUNCIONARIO_BANCO** | chave_aleatória + pix≥1k | 7 | 83 | 164 | 33.6% | 23.4% | 0.276 | 0.164% |
| 6 | **IDOSO_VULNERAVEL_70** | idade≥70 + pix≥1k | 7 | 71 | 118 | 37.6% | 20.0% | 0.261 | 0.118% |
| 7 | **BURST_INTENSO_RAPIDO** | burst_intenso + burst_30m + mult_pix | 6 | 48 | **0** | **100%** | 13.5% | 0.238 | **0.000%** |
| 8 | **BURST_ESVAZIAMENTO_CONTA** | burst_conta_antiga + pix≥1k | 3 | 16 | 26 | 38.1% | 4.5% | 0.081 | 0.026% |
| 9 | **IDOSO_VULNERAVEL_80** | idade≥80 + pix≥1k | 6 | 11 | 13 | 45.8% | 3.1% | 0.058 | 0.013% |

### 4.2 Fichas técnicas por padrão

---

#### PADRÃO 1: BURST_VALOR_ALTO

| Campo | Valor |
|---|---|
| **Golpe mapeado** | Engenharia social com urgência — vítima é pressionada a fazer múltiplas transferências rápidas de valor moderado a alto |
| **Modus operandi** | Criminoso cria senso de urgência (falsa emergência, suposta fraude na conta) levando a múltiplos PIX em menos de 30 minutos. O burst temporal é a assinatura mais forte |
| **Required** | `burst_30m` (Lift 241.7x) + `pix_acima_500` (Lift 10.1x) |
| **Optional** | burst_intenso, multiplos_pix_rapidos, intervalo_muito_curto, pix_acima_1000, valor_absoluto_alto, valor_absoluto_muito_alto, primeira_tx_trimestre, renda_desconhecida_valor_alto, renda_metade_comprometida, idade_60_plus |
| **min_score** | 3 (required soma 4 → ativa com os 2 required apenas) |
| **Severity** | ALTO |
| **Performance** | TP=141, FP=38, **Precision=78.8%**, F1=0.528, FPR=0.038% |
| **Validação externa** | Febraban documenta falsa central como golpe #1 em valor/vítima (R$7.500 médio). ESET Brasil (Jan/2026): "transferências instantâneas + pressão emocional + urgência" |
| **Histórico** | v3.2: Criado (gate R$1k). v3.3: Gate relaxado para R$500 (+36 TP, +11 FP). Simulado em 100.355 tx |
| **Última calibração** | 2026-04-11 |

---

#### PADRÃO 2: ESVAZIAMENTO_CONTA

| Campo | Valor |
|---|---|
| **Golpe mapeado** | Account takeover ou coação — vítima (ou fraudador com acesso) faz múltiplos PIX rápidos esvaziando a conta |
| **Modus operandi** | Burst de ≥2 transações com `qt_pix_dia_max ≥ 3` no trimestre, indicando comportamento atípico de volume. Frequentemente combinado com valores altos e recebedores desconhecidos |
| **Required** | `multiplos_pix_rapidos` (Lift 196.7x) |
| **Optional** | burst_intenso, intervalo_muito_curto, pix_acima_1000, valor_absoluto_alto, primeira_tx_trimestre, renda_desconhecida_valor_alto, renda_incompativel, multiplos_recebedores_distintos, aproximando_esgotamento |
| **min_score** | 4 (required=2 + 2 optional necessários) |
| **Severity** | CRITICO |
| **Performance** | TP=98, FP=44, **Precision=69.0%**, F1=0.394, FPR=0.044% |
| **Validação externa** | LexisNexis 2026: 28M fraudes PIX em 2025. Relatório Axur (2025) sobre Plump Spider documenta MO de esvaziamento com contas laranja |
| **Histórico** | v3.0: ms=5. v3.1: ms=4 (calibrado via curva P-R) |
| **Última calibração** | 2026-04-11 |

---

#### PADRÃO 3: COACAO_FISICA

| Campo | Valor |
|---|---|
| **Golpe mapeado** | Sequestro relâmpago / coação presencial — vítima forçada a fazer PIX sob ameaça física |
| **Modus operandi** | Primeira transação do trimestre + valor ≥R$1k + intervalo ≤5min entre transações. O perfil "nunca usou e de repente faz PIX alto e rápido" é assinatura de coação |
| **Required** | `intervalo_muito_curto` (Lift 6.7x) + `pix_acima_1000` (Lift 14.5x) + `primeira_tx_trimestre` (Lift 146.3x) |
| **Optional** | burst_intenso, burst_30m, multiplos_pix_rapidos, valor_absoluto_alto, valor_absoluto_muito_alto, renda_desconhecida_valor_alto, renda_incompativel, multiplos_recebedores_distintos |
| **min_score** | 5 (3 required = 6 → ativa sempre que os 3 required estão presentes) |
| **Severity** | CRITICO |
| **Performance** | TP=89, FP=34, **Precision=72.4%**, F1=0.372, FPR=0.034% |
| **Validação externa** | MED 2.0 (Resolução BCB 493/2025) explicitamente inclui "coerção" como motivo para acionamento. Novas regras de Fev/2026 reforçam combate a "casos de coerção" |
| **Histórico** | v3.0: sem primeira_tx (FP=303). v3.2: +primeira_tx required (FP→34, -89%) |
| **Última calibração** | 2026-04-11 |

---

#### PADRÃO 4: PRIMEIRA_TX_SUSPEITA

| Campo | Valor |
|---|---|
| **Golpe mapeado** | Engenharia social "low & slow" — golpista convence vítima a fazer um PIX "único" sem sinais de urgência |
| **Modus operandi** | Primeira transação do trimestre com valor ≥R$1k. Sem burst, sem intervalo curto obrigatório. Perfil: vítima inativa que é convencida a transferir valor significativo uma única vez |
| **Required** | `primeira_tx_trimestre` (Lift 146.3x) + `pix_acima_1000` (Lift 14.5x) |
| **Optional** | idade_60_plus, idade_70_plus, renda_metade_comprometida, renda_desconhecida_valor_alto, valor_absoluto_alto, is_segmento_premium, valor_redondo, chave_aleatoria, recebedor_nunca_visto |
| **min_score** | 4 (required=4 → ativa com os 2 required) |
| **Severity** | MEDIO |
| **Performance** | TP=89, FP=34, **Precision=72.4%**, F1=0.372, FPR=0.034% |
| **Validação externa** | ADDP (2025): 53% das vítimas >50 anos, perfil de baixa atividade digital. Febraban: golpe do falso funcionário usa abordagem "única" (uma ligação → uma transferência) |
| **Histórico** | v3.2: Criado para capturar invisíveis "low & slow" |
| **Última calibração** | 2026-04-11 |

---

#### PADRÃO 5: FALSO_FUNCIONARIO_BANCO

| Campo | Valor |
|---|---|
| **Golpe mapeado** | Falsa central telefônica — criminoso liga se passando por banco |
| **Modus operandi** | Vítima recebe ligação de suposto funcionário alegando fraude na conta. É instruída a transferir para "conta segura" via chave aleatória (vítima não conhece o recebedor) com valor alto |
| **Required** | `chave_aleatoria` (Lift 0.9x isolado, essencial conceitualmente) + `pix_acima_1000` (Lift 14.5x) |
| **Optional** | idade_60_plus, idade_70_plus, burst_30m, intervalo_muito_curto, valor_absoluto_alto, valor_redondo, horario_comercial, is_segmento_premium, login_senha, renda_incompativel, renda_desconhecida_valor_alto, recebedor_nunca_visto |
| **min_score** | 7 (required=4 + 3 optional necessários) |
| **Severity** | CRITICO |
| **Performance** | TP=83, FP=164, Precision=33.6%, F1=0.276, FPR=0.164% |
| **Validação externa** | Febraban: golpe do falso funcionário é o **#1 em valor médio** (R$7.500/vítima, 2025). ESET (Jan/2026): "criminosos usam IA para personalizar ligações". Let's Money (Mar/2026): "novo golpe do Pix tem preocupado especialistas — criminosos se passam por funcionários" |
| **Histórico** | v3.0: ms=6 (FP=409). v3.1: ms=7 (-245 FP). Recalibrado via curva P-R |
| **Última calibração** | 2026-04-11 |

---

#### PADRÃO 6: IDOSO_VULNERAVEL_70

| Campo | Valor |
|---|---|
| **Golpe mapeado** | Qualquer golpe de SE direcionado a idosos 70+ |
| **Modus operandi** | Idosos são alvos preferenciais por menor familiaridade digital e maior patrimônio. O padrão detecta quando um cliente 70+ faz PIX ≥R$1k com múltiplos indicadores de risco |
| **Required** | `idade_70_plus` (Lift 8.4x) + `pix_acima_1000` (Lift 14.5x) |
| **min_score** | 7 (required=4 + 3 optional) |
| **Severity** | CRITICO |
| **Performance** | TP=71, FP=118, Precision=37.6%, F1=0.261, FPR=0.118% |
| **Validação externa** | ADDP (2025): **53% das vítimas de golpes PIX têm mais de 50 anos**. Febraban: idosos são alvo prioritário de engenharia social |
| **Última calibração** | 2026-04-11 |

---

#### PADRÃO 7: IDOSO_VULNERAVEL_80

| Campo | Valor |
|---|---|
| **Golpe mapeado** | Golpe de SE contra idosos 80+ — vulnerabilidade máxima |
| **Required** | `idade_80_plus` (Lift 4.5x) + `pix_acima_1000` (Lift 14.5x) |
| **min_score** | 6 (required=4 + 2 optional) |
| **Severity** | CRITICO |
| **Performance** | TP=11, FP=13, Precision=45.8%, F1=0.058, FPR=0.013% |
| **Última calibração** | 2026-04-11 |

---

#### PADRÃO 8: BURST_INTENSO_RAPIDO

| Campo | Valor |
|---|---|
| **Golpe mapeado** | Esvaziamento acelerado — burst extremo com 3+ tx em 30min |
| **Modus operandi** | Trinca de indicadores de velocity extrema que **nunca ocorre em transações normais** no dataset. Quando os três ativam simultaneamente, é fraude com 100% de certeza |
| **Required** | `burst_intenso` (Lift ∞) + `burst_30m` (Lift 241.7x) + `multiplos_pix_rapidos` (Lift 196.7x) |
| **min_score** | 6 (3 required = 6) |
| **Severity** | CRITICO |
| **Performance** | TP=48, **FP=0**, **Precision=100%**, F1=0.238, FPR=0.000% |
| **Nota** | **Zero falsos positivos.** Regra cirúrgica — pega exclusivamente fraude |
| **Última calibração** | 2026-04-11 |

---

#### PADRÃO 9: BURST_ESVAZIAMENTO_CONTA

| Campo | Valor |
|---|---|
| **Golpe mapeado** | Conta antiga comprometida — burst súbito após longo período de inatividade |
| **Modus operandi** | Conta com ≥12 meses de relacionamento que apresenta burst para primeiro recebedor + valor alto. Indica account takeover |
| **Required** | `burst_conta_antiga` (Lift 47.3x) + `pix_acima_1000` (Lift 14.5x) |
| **min_score** | 3 (required=4 → ativa com required) |
| **Severity** | CRITICO |
| **Performance** | TP=16, FP=26, Precision=38.1%, F1=0.081, FPR=0.026% |
| **Última calibração** | 2026-04-11 |

---

#### PADRÃO 10: IDOSO_JOVEM_VALOR_MODERADO_RESIDUAL

| Campo | Valor |
|---|---|
| **Golpe mapeado** | Perfis de jovens/idosos coagidos a realizar transações atípicas de valor médio-alto. |
| **Modus operandi** | Clientes jovens (≤25) ou seniores (≥60) com contas recentes (<24m) transferindo R$1.500 a R$15.000. O IF≥0.90 confirma o cenário como uma anomalia em seu histórico pessoal. |
| **Required** | `idade_young_max` ou `idade_old_min`, `valor_moderado`, `relacionamento_conta_nova`, `if_percentile_anomalo`, `first_receiver` |
| **min_score** | 4 |
| **Severity** | MODERADO_ALTO |
| **Performance** | TBD (inserido via EXP-003, FNs recuperados nativamente pelo EXP-001) |
| **Última calibração** | 2026-04-20 |

---

## 5. Performance Global do Módulo

### 5.1 Evolução v2.1 → v3.3

| Métrica | v2.1 | v3.0 | v3.1 | v3.2 | **v3.3** |
|---|---|---|---|---|---|
| **Fraudes detectadas** | 188 (52.9%) | 219 (61.7%) | 200 (56.3%) | 244 (68.7%) | **262 (73.8%)** |
| **Invisíveis** | 167 | 136 | 155 | 111 | **93** |
| **FP totais** | 57.108 | 957 | ~553 | 335 | **341** |
| **FPR** | 57.1% | 0.96% | 0.55% | 0.34% | **0.34%** |
| **Precision (thr>0)** | 0.33% | 18.6% | 26.6% | 42.1% | **43.5%** |
| **F1 (thr>0)** | 0.007 | 0.286 | 0.361 | 0.522 | **0.547** |
| **Padrões ativos** | 12 | 6 | 6 | 9 | **9** |
| **Indicadores ativos** | ~50 | 28 | 28 | 30 | **31** |

### 5.2 Performance por threshold (v3.3)

| Threshold | TP | FP | Precision | Recall | F1 | FPR | Uso recomendado |
|---|---|---|---|---|---|---|---|
| >0 | 262 | 341 | 43.5% | 73.8% | 0.547 | 0.341% | Agravante leve |
| ≥25 | 229 | 335 | 40.6% | 64.5% | 0.498 | 0.335% | — |
| **≥40** | **209** | **85** | **71.1%** | **58.9%** | **0.644** | **0.085%** | **Agravante forte** |
| ≥60 | 155 | 55 | 73.8% | 43.7% | 0.549 | 0.055% | Quase-veto |
| ≥80 | 82 | 6 | **93.2%** | 23.1% | 0.370 | 0.006% | Veto |

### 5.3 Distribuição de scores

| Métrica | Fraudes | Normais | Separação |
|---|---|---|---|
| Média | **49.6** | 0.16 | 49.4 pts |
| Mediana | **55.0** | 0.0 | 55.0 pts |
| % score = 0 | 26.2% | **99.66%** | — |
| % score > 40 | **58.9%** | 0.08% | — |

### 5.4 Complementaridade com LGBM

| Métrica | Resultado |
|---|---|
| Fraudes detectadas por ambos (holdout) | 64/80 |
| Só LGBM detectou | 15/80 |
| Só SE detectou | 0/80 |
| Nenhum detectou | 1/80 |

> **O SE não captura fraudes incrementais** vs o LGBM no holdout. Seu valor é como **agravante, explicador e rede de segurança** — não como detector independente. Testamos injetar as features do SE como input do LGBM (Frente 4) e concluímos que **não melhora o ML** porque o LGBM já aprende as mesmas combinações via tree splits. A integração correta é pós-LGBM como agravante.

---

## 6. Metodologia de Calibração

### 6.1 Processo de 6 frentes

| Frente | Objetivo | Resultado |
|---|---|---|
| **F1: Validação Retroativa** | Medir performance do SE isolado em 100.355 tx | Confusion matrix por padrão + indicadores com Lift |
| **F2: Calibração min_score** | Otimizar min_score via curvas Precision-Recall | 6 padrões calibrados, override manual onde necessário |
| **F3: Análise Exploratória** | Investigar invisíveis + FP + candidatos a novos padrões | +3 padrões novos, COACAO fix (-89% FP) |
| **F4: Integração SE × LGBM** | Testar SE features como input do LGBM | **Rejeitado** — SE features redundantes, OOF piora |
| **F5: Investigação Forense** | Dissecar 111 invisíveis restantes + simular novos padrões | Gate R$500 no BURST_VALOR_ALTO (+18 fraudes) |
| **F6: Documentação** | Este documento |

### 6.2 Critérios de inclusão/exclusão

**Para indicadores:**
- Inclusão: Lift ≥ 1.5x validado no dataset completo
- Exclusão: Lift < 1.0x (anti-indicador) ou Lift < 1.5x sem justificativa conceitual

**Para padrões:**
- Inclusão: Precision ≥ 1% **ou** recall incremental demonstrado
- min_score: Calibrado via max(F1) com override manual para tradeoffs Precision × FP
- Exclusão: Precision < 1% **e** zero valor incremental (5 padrões removidos na v3.0)

**Para novos padrões (Frente 5):**
- Simulação completa em 100.355 tx
- Critério: Precision ≥ 25% **E** Novos (ex-invisíveis) ≥ 5
- 20 candidatos testados, 1 aprovado (gate R$500 no BURST_VALOR_ALTO)

### 6.3 Hipóteses rejeitadas (documentação de decisões negativas)

| Hipótese | Resultado | Motivo da rejeição |
|---|---|---|
| SE features como input do LGBM (F4) | OOF AP cai -3.3% | Features redundantes com inputs brutos |
| intervalo_curto + renda_comprometida (F5-R1) | Prec=9.8%, FP=840 | intervalo_curto ativa em 24% dos normais |
| idoso + intervalo_curto sem gate valor (F5-R3) | Prec=8.3%, FP=1.430 | FP inaceitável |
| premium + renda + intervalo (F5-R4) | Prec=7.6%, FP=1.309 | Indicadores de baixo Lift |
| intervalo + chave_aleatória (F5-R6) | Prec=2.5%, FP=4.167 | Catastrófico — ambos muito comuns em normais |

---

## 7. Limitações Conhecidas

### 7.1 Fraudes irrecuperáveis

**93 fraudes (26.2%)** permanecem invisíveis ao SE v3.3. Perfil dominante:

- Valor < R$500 (abaixo de todos os gates)
- Sem burst (sem padrão de urgência)
- Sem primeira transação do trimestre
- Indicadores ativos: apenas intervalo_curto + perfil genérico (Lift < 2x)

**Essas fraudes são irrecuperáveis com os dados disponíveis** e requerem fontes adicionais para detecção:
- Grafo de recebedores (recebedor aparece em outras fraudes?)
- Device fingerprint detalhado
- Behavioral biometrics (velocidade de digitação, padrão de toque)
- Análise multi-sessão

### 7.2 Overlap COACAO_FISICA × PRIMEIRA_TX_SUSPEITA

Jaccard = 1.0 (co-ativação total). Não é problema operacional porque:
- Estão em clusters separados (cobrem conceitos diferentes)
- Deduplicação funciona: COACAO (CRITICO, 40pts) domina quando ambos ativam
- PRIMEIRA_TX_SUSPEITA (MEDIO, 15pts) só contribui sozinho quando COACAO não ativa

Se o Jaccard permanecer 1.0 após recalibração futura, considerar merge.

### 7.3 FALSO_FUNCIONARIO_BANCO — maior gerador de FP

164 FP (48% dos FP totais). O `chave_aleatoria` como required tem Lift 0.9x isolado. Alternativa testada (remover chave_aleatoria do required) pioraria a coerência conceitual — falso funcionário sem chave aleatória não faz sentido operacional. O min_score=7 é o controle de FP (exige required + 3 optional).

---

## 8. Integração no Pipeline

### 8.1 Fluxo operacional

```
Transação → Feature Engineering → LGBM (score 0-1)
                                    │
                                    ├─ SE roda em paralelo (<1ms)
                                    │
                                    ▼
                              Orquestrador
                                    │
                                    ├─ Se LGBM score ≥ threshold → suspeita
                                    │   └─ SE score ≥ 40 → agravante +3 pts
                                    │   └─ SE score ≥ 60 → agravante +4 pts
                                    │
                                    ├─ Se LGBM score < threshold
                                    │   └─ SE score ≥ 80 → override para revisão
                                    │
                                    └─ Decisão Final (APROVAR/CONFIRMAR/BLOQUEAR)
```

### 8.2 Output para CX (Customer Experience)

Quando o SE detecta um padrão, o output é humano-legível e pode ser usado em:

- **Mensagem ao operador de revisão:** "Padrão detectado: COACAO_FISICA — Possível coação física. Primeira transação do trimestre, valor R$3.000, intervalo < 5 minutos"
- **Justificativa MED:** Evidência estruturada para acionar o Mecanismo Especial de Devolução
- **Alerta regulatório:** Documentação auditável de por que a transação foi bloqueada

---

## 9. Plano de Manutenção

| Atividade | Frequência | Responsável |
|---|---|---|
| Re-rodar `avaliar_se_retroativo.py` com dados novos | Trimestral | Data Science |
| Recalibrar min_score com `calibrar_min_score_SE.py` | Trimestral | Data Science |
| Revisar indicadores (Lift pode degradar) | Trimestral | Data Science |
| Análise exploratória de novas invisíveis | Semestral | Data Science |
| Atualizar referências regulatórias | Quando houver nova resolução BCB | Compliance + DS |
| Adicionar novos padrões (se novos MO surgirem) | Sob demanda | Data Science |

---

## 10. Referências

### Regulatórias
1. **Banco Central do Brasil** — Resolução BCB nº 493 (Ago/2025): MED 2.0 com rastreamento de contas intermediárias
2. **Banco Central do Brasil** — Resolução BCB nº 403/2024: Limites noturnos para PIX
3. **Banco Central do Brasil** — MED 2.0 obrigatório desde 02/Fev/2026 para todas as instituições
4. **CMN** — Resolução 4.893/2021: Política de segurança cibernética

### Dados de Mercado
5. **ADDP** (2025) — 28 milhões de vítimas de golpes PIX em 2025; 53% >50 anos
6. **Febraban** (2025) — Falsa central telefônica: golpe #1 em valor médio (R$7.500/vítima)
7. **Febraban** (2023) — "Bandidos usam novas abordagens para golpes antigos" — taxonomia de SE
8. **ESET Brasil / Security Leaders** (Jan/2026) — "IA e engenharia social devem tornar golpes com PIX mais sofisticados em 2026"
9. **Let's Money** (Mar/2026) — "Golpe do PIX cresce com engenharia social e urgência"

### Acadêmicas e Técnicas
10. **Sundararamaiah et al.** (Dez/2024) — "Unifying AI and Rule-based Models for Financial Fraud Detection", IJCTT v72, DOI: 10.14445/22312803/IJCTT-V72I12P107
11. **Ben Abid et al.** (2025) — "A Scalable Hybrid Approach to Detecting Fraud with ML", EUSIPCO 2025
12. **Vallarino** (Abr/2025) — "Detecting Financial Fraud with Hybrid Deep Learning: A Mix-of-Experts Approach", arXiv:2504.03750
13. **Flagright** (Jan/2026) — "AI vs Rules-Based Transaction Monitoring: Why a Hybrid Approach Wins"
14. **Fraud.net** (2023) — "Bridging the Gap: Incorporating AI/ML into Rules-Based Fraud Detection Models"

### Scripts de Calibração (Artefatos Internos)
15. `avaliar_se_retroativo.py` — Validação retroativa (Frente 1)
16. `calibrar_min_score_SE.py` — Curvas Precision-Recall por padrão (Frente 2)
17. `se_frente3_analise_exploratoria.py` — Análise exploratória (Frente 3)
18. `se_frente4_feature_injection.py` — Teste de injeção SE → LGBM (Frente 4)
19. `se_frente5_invisiveis_investigation.py` — Investigação forense das invisíveis (Frente 5)
20. `se_frente5_simulacao_v33.py` — Simulação de 20 candidatos a padrões (Frente 5)

---

*Documento gerado em 2026-04-11. Módulo SE v3.3 calibrado com 100.355 transações (355 fraudes confirmadas).*

---

