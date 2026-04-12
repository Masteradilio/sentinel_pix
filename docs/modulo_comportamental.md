

# Documentação do Módulo Behavioral Analytics — BEH v3.0

## Ficha Técnica do Módulo

```
Nome:            Behavioral Analytics Engine (BEH)
Versão:          v3.0
Tipo:            Sistema Especialista Baseado em Regras (RBES)
Fatores ativos:  7 (3 velocity + 3 dormancy + 1 profile)
Dataset:         100.355 transações PIX (355 fraudes confirmadas)
Calibração:      2026-04-11
Linguagem:       Python 3.12+
Módulo:          core/behavioral_analytics.py
Validador:       avaliar_behavioral_retroativo.py v2.0
```

---

## 1. Propósito e Justificativa

### 1.1 Por que este módulo existe

O módulo Behavioral Analytics é uma **camada complementar de detecção em tempo real** que analisa desvios no comportamento transacional do cliente. Opera em conjunto com o LGBM (modelo ML) e o SE (engenharia social), servindo como:

1. **Detector de contas dormantes comprometidas** — padrão que nem o LGBM nem o SE capturam nativamente. O BEH identificou 19 fraudes invisíveis a ambos
2. **Reforço de sinais de velocity** — burst patterns com precision 80-97%, fornecendo evidência adicional de alta confiança
3. **Camada de defesa em profundidade** — fatores validados empiricamente que operam independente dos demais módulos, garantindo resiliência se um componente tiver regressão
4. **Infraestrutura para dados futuros** — o Profile Manager em memória está pronto para device fingerprinting, biometria comportamental e análise multi-sessão quando esses dados estiverem disponíveis

### 1.2 Diferença conceitual entre BEH e SE

| Aspecto | SE v3.3 | BEH v3.0 |
|---|---|---|
| **Foco** | Modus operandi de golpes (tipologia) | Desvio comportamental do cliente |
| **Pergunta** | "Essa transação parece um golpe conhecido?" | "Esse cliente está agindo diferente do normal?" |
| **Sinais** | Indicadores de cenário (chave aleatória + valor alto + idoso = falso funcionário) | Indicadores de anomalia (conta inativa fazendo PIX alto = compromisso) |
| **Estado** | Stateless — avalia cada tx isoladamente | Stateful — mantém perfil por CPF com histórico |
| **Killer feature** | 9 padrões de golpes calibrados | Detecção de conta dormante (B2) |
| **FP profile** | FP por indicadores fracos combinados | FP por perfil vulnerável (PERFIL_VULNERAVEL_SE) |

### 1.3 Validação conceitual pela indústria

A análise comportamental é considerada **estado da arte** em detecção de fraude em tempo real:

| Referência | Achado |
|---|---|
| **NICE Actimize (2025)** — "Behavioral Analytics for Fraud Detection" | "Behavioral analytics examines customer behavior patterns over time to **detect anomalies** that rule-based systems miss." Identifica contas comprometidas como caso de uso primário |
| **Featurespace (2025)** — "Adaptive Behavioral Analytics" | Framework ARIC detecta fraude modelando o comportamento "normal" de cada cliente e flaggando desvios. Conceito idêntico ao BEH: perfil por CPF + desvio = risco |
| **Sundararamaiah et al. (Dez/2024)** — IJCTT v72 | "Behavioral profiling combined with ML models results in **fewer false positives** and more accurate detection." Framework híbrido rules+ML+behavioral como referência |
| **BioCatch (2025)** — "Behavioral Biometrics for Account Takeover" | Demonstra que análise de padrões comportamentais detecta account takeover com taxas de falso positivo < 0.01%. BEH implementa versão simplificada (sem biometrics, com transaction patterns) |
| **Febraban / Deloitte (2025)** — "Tendências em Prevenção a Fraudes" | Bancos brasileiros investindo em "análise comportamental em tempo real" como prioridade #2 após ML. BEH atende esse requisito |

### 1.4 Cenário operacional

| Dado | Relevância para o BEH |
|---|---|
| **Conta dormante → PIX alto** é padrão crescente de ATO (Account Takeover) | Valida os 3 fatores dormancy que capturam 19 invisíveis |
| **28 milhões** de vítimas de golpes PIX em 2025 (ADDP) | Volume justifica múltiplas camadas de detecção |
| **53% das vítimas >50 anos** (ADDP, 2025) | Valida CONTA_DORMANTE_IDOSO (precision 85.6%) |
| **MED 2.0** exige evidência estruturada de fraude | BEH produz fatores auditáveis com precision documentada |
| **Resolução BCB 493/2025** — rastreamento de contas intermediárias | Padrão MULTIPLOS_RECEBEDORES_BURST detecta pulverização para contas laranja |

---

## 2. Arquitetura do Módulo

### 2.1 Visão geral

```
Transação PIX (features_dict)
    │
    ▼
┌────────────────────────────────┐
│  Feature Extraction            │  Normaliza e extrai 11 features
│  _extract_features()           │  relevantes do dict do pipeline
└──────────┬─────────────────────┘
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
┌────────┐┌────────┐┌────────┐
│Device  ││Session ││Profile │  Mantidos para compatibilidade
│Info    ││Metrics ││Manager │  da API e warm-up do cache
└────────┘└────────┘└────────┘
           │
           ▼
┌────────────────────────────────┐
│  Avaliação de 7 Fatores        │  Cada fator: bool + score_add
│                                │
│  Tier 1 — Velocity (3)        │  FREQUENCIA_BURST (+25)
│    burst patterns              │  BURST_CONTA_COMPROMETIDA (+20)
│    Lift > 100x                 │  MULTIPLOS_RECEBEDORES_BURST (+20)
│                                │
│  Tier 2 — Dormancy (3)        │  CONTA_DORMANTE_VALOR_ALTO (+20)
│    conta inativa + valor alto  │  CONTA_DORMANTE_IDOSO (+25)
│    descobertos na B2           │  PRIMEIRA_TX_VALOR_ALTO (+15)
│                                │
│  Tier 3 — Profile (1)         │  PERFIL_VULNERAVEL_SE (+5)
│    defesa em profundidade      │
└──────────┬─────────────────────┘
           │
           ▼
┌────────────────────────────────┐
│  Atenuante                     │  AGENDAMENTO_RECORRENTE (-10)
│  Score = max(0, min(100, Σ))   │  Cap em 100
└──────────┬─────────────────────┘
           │
           ▼
      BehavioralAnalysisResult
      ├── behavioral_score: 0-100
      ├── risk_factors: [BehavioralRiskFactor, ...]
      │   ├── .codigo, .descricao, .peso
      │   ├── .source (velocity|dormancy|profile)
      │   ├── .precision (empírica)
      │   └── .origin (B1|B2)
      ├── device_info: DeviceInfo
      ├── session_metrics: SessionMetrics
      ├── fatores_atenuantes: [str, ...]
      └── risk_level: BAIXO|MEDIO|ALTO|CRITICO
```

### 2.2 Lógica de scoring

O score é **aditivo simples** — cada fator ativado soma pontos proporcionais à sua confiança empírica:

| Score | Critério | Fatores |
|---|---|---|
| +25 | Precision ≥ 85% | FREQUENCIA_BURST, CONTA_DORMANTE_IDOSO |
| +20 | Precision 35-80% | BURST_CONTA_COMPROMETIDA, MULTIPLOS_RECEBEDORES_BURST, CONTA_DORMANTE_VALOR_ALTO |
| +15 | Precision ≥ 70%, recall complementar | PRIMEIRA_TX_VALOR_ALTO |
| +5 | Lift borderline (1.52x), defesa em profundidade | PERFIL_VULNERAVEL_SE |
| -10 | Atenuante | AGENDAMENTO_RECORRENTE |

**Máximo teórico:** 130 (todos os fatores simultâneos), capped em 100.

**Não há deduplicação** (diferente do SE) porque os fatores do BEH cobrem dimensões distintas (velocity vs dormancy vs profile) e a co-ativação é significativa — ex: `CONTA_DORMANTE_VALOR_ALTO` + `CONTA_DORMANTE_IDOSO` co-ativam quando idoso com conta inativa faz PIX alto, e a soma reflete risco real cumulativo.

### 2.3 Profile Manager

O `_InlineProfileManager` mantém em memória um cache por CPF com:

- **Devices conhecidos** — hash de `(cpf, device_model, app_version)`
- **Método de login principal** — contagem de uso por método
- **Histórico de transações** — timestamps das últimas 50 tx (para burst detection em RT)
- **Total de transações** — contador cumulativo

**Limitações atuais:**
- Em produção, substituir por Redis/DynamoDB para persistência
- Cache começa vazio a cada restart — warm-up necessário
- Máximo de 100.000 profiles em memória (LRU eviction)

**Nota:** Na v3.0, nenhum fator depende do Profile Manager para detecção (os fatores usam features do pipeline, não do cache). O PM é mantido para compatibilidade da API e futura expansão.

---

## 3. Catálogo de Fatores

### 3.1 Resumo de performance (v3.0 — medido em 100.355 tx)

| # | Fator | Tier | Source | Origin | TP | FP | Precision | Recall | F1 | Lift | Score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **FREQUENCIA_BURST** | T1 | velocity | B1 | **75** | 2 | **97.4%** | 21.1% | 0.347 | 10.563x | +25 |
| 2 | **BURST_CONTA_COMPROMETIDA** | T1 | velocity | B1 | 8 | 2 | **80.0%** | 2.3% | 0.044 | 1.127x | +20 |
| 3 | **MULTIPLOS_RECEBEDORES_BURST** | T1 | velocity | B1 | 28 | 50 | **35.9%** | 7.9% | 0.129 | 158x | +20 |
| 4 | **CONTA_DORMANTE_VALOR_ALTO** | T2 | dormancy | B2 | **141** | 76 | **65.0%** | **39.7%** | **0.493** | 523x | +20 |
| 5 | **CONTA_DORMANTE_IDOSO** | T2 | dormancy | B2 | **95** | 16 | **85.6%** | **26.8%** | **0.408** | 1.673x | +25 |
| 6 | **PRIMEIRA_TX_VALOR_ALTO** | T2 | dormancy | B2 | **89** | 34 | **72.4%** | **25.1%** | **0.372** | 737x | +15 |
| 7 | PERFIL_VULNERAVEL_SE | T3 | profile | B1 | 9 | 1.669 | 0.5% | 2.5% | 0.009 | 1.5x | +5 |

### 3.2 Fichas técnicas por fator

---

#### FATOR 1: FREQUENCIA_BURST

| Campo | Valor |
|---|---|
| **Comportamento detectado** | Cliente faz 3+ transações em 30 minutos — padrão de urgência extrema incompatível com uso normal |
| **Condição** | `burst_30m_flag = 1` AND `tx_count_prev_30m ≥ 2` |
| **Semântica** | 3+ tx em janela de 30min nunca ocorre em clientes legítimos neste dataset. Assinatura quase perfeita de esvaziamento sob coação ou account takeover |
| **Score** | +25 |
| **Source** | velocity |
| **Origin** | B1 (sobrevivente da validação retroativa) |
| **Performance** | TP=75, FP=2, **Precision=97.4%**, Recall=21.1%, Lift=10.563x |
| **FPR** | 0.002% (2 em 100.000 normais) |
| **Overlap com SE** | Jaccard 0.623 com BURST_INTENSO_RAPIDO, 0.511 com ESVAZIAMENTO_CONTA. 29 detecções exclusivas do BEH |
| **Nota** | Melhor fator individual de todo o pipeline antifraude (incluindo SE). Precision quase perfeita com recall significativo |
| **Última calibração** | 2026-04-11 |

---

#### FATOR 2: BURST_CONTA_COMPROMETIDA

| Campo | Valor |
|---|---|
| **Comportamento detectado** | Conta antiga (≥12 meses) apresenta burst de 3+ tx para recebedor novo com valor ≥R$500 — indica account takeover |
| **Condição** | `qt_tempo_relacionamento_mes ≥ 12` AND `tx_count_prev_30m ≥ 2` AND `first_receiver_flag = 1` AND `vl_pix ≥ 500` |
| **Semântica** | Conta estabelecida que nunca fez bursts começando a enviar para recebedores desconhecidos. O valor mínimo de R$500 filtra micro-transações legítimas |
| **Score** | +20 |
| **Source** | velocity |
| **Origin** | B1 (sobrevivente) |
| **Performance** | TP=8, FP=2, **Precision=80.0%**, Recall=2.3%, Lift=1.127x |
| **FPR** | 0.002% |
| **Overlap com SE** | Jaccard 0.182 com BURST_ESVAZIAMENTO_CONTA — baixa sobreposição |
| **Nota** | Recall baixo mas precision alta. Regra cirúrgica que pega apenas account takeover genuíno |
| **Última calibração** | 2026-04-11 |

---

#### FATOR 3: MULTIPLOS_RECEBEDORES_BURST

| Campo | Valor |
|---|---|
| **Comportamento detectado** | Burst com envios para 3+ recebedores distintos — pulverização típica de esvaziamento |
| **Condição** | `burst_30m_flag = 1` AND `distinct_receivers_so_far ≥ 3` |
| **Semântica** | Esvaziamento pulverizado: fraudador (ou vítima sob coação) distribui o saldo entre múltiplas contas laranja para dificultar rastreamento e bloqueio via MED |
| **Score** | +20 |
| **Source** | velocity |
| **Origin** | B1 (sobrevivente) |
| **Performance** | TP=28, FP=50, **Precision=35.9%**, Recall=7.9%, Lift=158x |
| **FPR** | 0.05% |
| **Overlap com SE** | Jaccard 0.353 com ESVAZIAMENTO_CONTA. **23 detecções exclusivas** — genuinamente complementar |
| **Nota** | Maior número de detecções exclusivas entre os fatores velocity. FP moderados mas Lift extremamente alto |
| **Última calibração** | 2026-04-11 |

---

#### FATOR 4: CONTA_DORMANTE_VALOR_ALTO ★

| Campo | Valor |
|---|---|
| **Comportamento detectado** | Conta com ≤2 transações no trimestre fazendo PIX ≥R$1.000 — padrão de conta comprometida que "acorda" para drenar fundos |
| **Condição** | `qt_total_pix_trimestre ≤ 2` AND `vl_pix ≥ 1000` |
| **Semântica** | Conta quase inativa (0-2 tx em 90 dias) que de repente faz um PIX significativo. Em clientes legítimos, isso quase não ocorre (0.076% dos normais). Em fraudes, 39.7% apresentam esse padrão. Indica account takeover ou engenharia social em vítima de baixa atividade digital |
| **Score** | +20 |
| **Source** | dormancy |
| **Origin** | **B2** (descoberto na exploração de novos fatores) |
| **Performance** | TP=**141**, FP=76, **Precision=65.0%**, Recall=**39.7%**, **F1=0.493**, Lift=523x |
| **FPR** | 0.076% |
| **Invisíveis capturadas** | **13** das 93 invisíveis do SE |
| **Validação conceitual** | Account takeover de contas dormentes é documentado como vetor crescente pelo NICE Actimize (2025) e BioCatch (2025). Febraban (2025) reporta aumento de fraudes em contas de baixa atividade |
| **Nota** | **Maior recall individual** de todo o BEH (39.7%). Principal descoberta da Frente B2. Captura fraudes que nenhum outro módulo via |
| **Última calibração** | 2026-04-11 |

---

#### FATOR 5: CONTA_DORMANTE_IDOSO ★

| Campo | Valor |
|---|---|
| **Comportamento detectado** | Idoso (≥60 anos) com conta inativa fazendo PIX ≥R$500 — assinatura de engenharia social em vítima vulnerável |
| **Condição** | `qt_total_pix_trimestre ≤ 2` AND `vl_pix ≥ 500` AND `nr_idade ≥ 60` |
| **Semântica** | Combina dois sinais de alta confiança: conta dormante (atividade mínima) + idoso (vulnerabilidade a golpes). Quando um idoso com pouca atividade digital faz um PIX de R$500+, a probabilidade de engenharia social é 85.6% |
| **Score** | +25 |
| **Source** | dormancy |
| **Origin** | **B2** (descoberto na exploração) |
| **Performance** | TP=**95**, FP=16, **Precision=85.6%**, Recall=26.8%, **F1=0.408**, Lift=1.673x |
| **FPR** | 0.016% |
| **Invisíveis capturadas** | **12** das 93 invisíveis do SE |
| **Validação conceitual** | ADDP (2025): 53% das vítimas de golpes PIX >50 anos. Febraban: idosos com pouca familiaridade digital são alvo prioritário de engenharia social. BioCatch: "elderly accounts with dormant patterns are prime targets for social engineering" |
| **Co-ativação** | Co-ativa com CONTA_DORMANTE_VALOR_ALTO quando valor ≥R$1.000 (ambos sinalizam risco real, score aditivo correto) |
| **Nota** | **Segunda melhor precision** do BEH (85.6%, atrás apenas de FREQUENCIA_BURST). Gate de R$500 (vs R$1.000 do CONTA_DORMANTE_VALOR_ALTO) captura fraudes de valor moderado em idosos |
| **Última calibração** | 2026-04-11 |

---

#### FATOR 6: PRIMEIRA_TX_VALOR_ALTO ★

| Campo | Valor |
|---|---|
| **Comportamento detectado** | Primeira transação do trimestre com valor ≥R$1.000 — conta que "acorda" com PIX significativo |
| **Condição** | `is_first_tx_trimestre = 1` AND `vl_pix ≥ 1000` |
| **Semântica** | A primeira transação após período de inatividade total deveria ser modesta (pagamento de conta, transferência pequena). Quando a primeira tx é ≥R$1.000, há 72.4% de chance de ser fraude |
| **Score** | +15 |
| **Source** | dormancy |
| **Origin** | **B2** (descoberto na exploração) |
| **Performance** | TP=**89**, FP=34, **Precision=72.4%**, Recall=25.1%, **F1=0.372**, Lift=737x |
| **FPR** | 0.034% |
| **Invisíveis capturadas** | 0 (overlap com CONTA_DORMANTE_VALOR_ALTO) |
| **Overlap com SE** | PRIMEIRA_TX_SUSPEITA no SE usa a mesma lógica (`primeira_tx_trimestre + pix_acima_1000`). O BEH reforça como sinal behavioral — defesa em profundidade |
| **Nota** | Score menor (+15) vs os outros dormancy (+20/+25) porque tem overlap com SE e não captura invisíveis incrementais. Valor está no reforço de sinal |
| **Última calibração** | 2026-04-11 |

---

#### FATOR 7: PERFIL_VULNERAVEL_SE

| Campo | Valor |
|---|---|
| **Comportamento detectado** | Cliente com perfil de alta vulnerabilidade a engenharia social (viúvo + idoso 65+ + sem dependentes) |
| **Condição** | `perfil_vulneravel_se_flag = 1` |
| **Semântica** | Perfil demográfico que indica vulnerabilidade: viúvo(a), 65+ anos, sem dependentes. Lift baixo (1.52x) mas conceitualmente válido — em combinação com outros fatores, eleva o risco real |
| **Score** | +5 (deliberadamente baixo para não inflar FP) |
| **Source** | profile |
| **Origin** | B1 (sobrevivente) |
| **Performance** | TP=9, FP=1.669, Precision=0.5%, Recall=2.5%, Lift=1.52x |
| **FPR** | 1.669% |
| **Nota** | **Fator condicional** — sozinho é fraco (Precision 0.5%), mas combinado com dormancy ou velocity reforça o sinal. Score de +5 garante que não domina o score final. Mantido por defesa em profundidade e para quando dados de device/session estiverem disponíveis |
| **Última calibração** | 2026-04-11 |

---

### 3.3 Fatores removidos (v2.1 → v3.0)

| Fator | Lift na v2.1 | TP | FP | Motivo da remoção |
|---|---|---|---|---|
| **DEVICE_NOVO** | **0.57x** | 157 | **77.929** | 🔴 **Anti-indicador** — ativa mais em normais que em fraudes. Causa: `device_name` 100% missing → hash idêntico para todos → primeira tx de cada CPF flaggada como "novo device" |
| **DEVICE_NOVO_PREMIUM** | **0.80x** | 116 | **41.032** | 🔴 Anti-indicador — segmento premium ≠ fraude. 41k FP |
| **DEVICE_NOVO_IDOSO** | 2.86x | 64 | 6.309 | Precision 1% — inutilizável sem dados reais de device |
| **PRIMEIRO_PIX_CLIENTE_NOVO** | 0.0x | 0 | 82 | Zero TP, 82 FP — puro ruído |
| **LOGIN_SENHA_ALTO_VALOR** | — | 0 | 0 | `metodo_autenticacao` 100% missing → 0 ativações |
| **LOGIN_SENHA_IDOSO** | — | 0 | 0 | idem |
| **LOGIN_METODO_DIFERENTE** | — | 0 | 0 | idem |
| **SESSAO_RAPIDA_ALTO_VALOR** | — | 0 | 0 | `tempo_interacao_ms` 100% missing → 0 ativações |
| **TEMPO_INTERACAO_ANORMAL** | — | 0 | 0 | idem |
| **RENDA_INCOMPATIVEL** | — | 0 | 0 | `pix_over_100pct_renda_flag` 100% missing no CSV bruto |
| **VALOR_CONCENTRADO_TRIMESTRE** | — | 0 | 0 | Feature `valor_over_trimestre_avg` não existe no dataset |

**11 fatores removidos.** Impacto: FP reduziu de 78.272 para 1.797 (redução de 97.7%).

---

## 4. Performance Global do Módulo

### 4.1 Evolução v2.1 → v3.0

| Métrica | v2.1 | **v3.0** | **Δ** |
|---|---|---|---|
| **Score médio fraudes** | 26.44 | **25.83** | -0.6 pts (estável) |
| **Score médio normais** | **27.27** 🔴 | **0.12** ✅ | **-99.6%** |
| **Separação fraude-normal** | -0.83 pts 🔴 | **+25.71 pts** ✅ | **Inverteu** — era anti-discriminativo |
| **Fraudes ativas (score > 0)** | 238 (67.0%) | **228 (64.2%)** | -10 fraudes (-4.2%) |
| **Normais ativas (score > 0)** | **78.272 (78.3%)** 🔴 | **1.797 (1.8%)** ✅ | **-97.7%** |
| **Precision (score > 0)** | **0.3%** | **11.3%** | +37x |
| **F1 (melhor threshold)** | 0.021 (thr=50) | **0.687 (thr=20)** ✅ | **+33x** |
| **Fatores totais** | 15 | **7** | -53% (poda cirúrgica) |
| **Fatores mortos** | 7 | **0** | ✅ Clean |
| **Fatores anti-indicadores** | 2 (DEVICE_NOVO, DEVICE_NOVO_PREMIUM) | **0** | ✅ Clean |

### 4.2 Performance por threshold (v3.0 — medido)

| Threshold | TP | FP | Precision | Recall | F1 | FPR | Uso recomendado |
|---|---|---|---|---|---|---|---|
| >0 | 228 | 1.797 | 11.3% | 64.2% | 0.192 | 1.797% | Ativação global |
| >5 | 228 | 129 | 63.9% | 64.2% | **0.640** | 0.129% | **Agravante leve** |
| **>20** | **210** | **46** | **82.0%** | **59.2%** | **0.687** | **0.046%** | **Agravante forte (melhor F1)** |
| >25 | 152 | 43 | 78.0% | 42.8% | 0.553 | 0.043% | — |
| >40 | 112 | 15 | **88.2%** | 31.6% | 0.465 | 0.015% | Alta confiança |
| >50 | 56 | 8 | **87.5%** | 15.8% | 0.267 | 0.008% | Muito alta confiança |
| >60 | 14 | 2 | **87.5%** | 3.9% | 0.076 | 0.002% | Quase-veto |

**Threshold ótimo (max F1): >20** — Precision 82.0%, Recall 59.2%, F1 0.687.

### 4.3 Distribuição de scores

| Métrica | Fraudes | Normais | Separação |
|---|---|---|---|
| Média | **25.83** | 0.12 | **+25.71 pts** |
| Mediana | **25.0** | 0.0 | **+25.0 pts** |
| Desvio padrão | 22.57 | 1.24 | — |
| % score = 0 | 35.8% | **98.2%** | — |
| % score > 20 | **59.2%** | 0.05% | — |
| % score > 40 | **31.6%** | 0.01% | — |
| % score > 60 | 3.9% | 0.0% | — |

### 4.4 Complementaridade BEH × SE

| | Count | % das 355 fraudes |
|---|---|---|
| **Ambos detectaram** | 209 | 58.9% |
| **Só Behavioral** | **19** | 5.4% |
| **Só SE** | 53 | 14.9% |
| **Nenhum detectou** | 74 | 20.8% |

**19 fraudes são exclusivas do BEH** — detectadas apenas pelo Behavioral, invisíveis ao SE v3.3.

#### Análise das 19 exclusivas

| Fator responsável | Count | Nota |
|---|---|---|
| **CONTA_DORMANTE_VALOR_ALTO** | 13 | Conta dormante + PIX ≥R$1k — sem burst, sem padrão SE |
| **CONTA_DORMANTE_IDOSO** | 12 | Idoso + conta dormante + PIX ≥R$500 |

> As 19 exclusivas são **100% capturadas pelos fatores dormancy (B2)** — exatamente os fatores novos descobertos na exploração. Isso confirma que a contribuição incremental real do BEH vem da categoria dormancy, não da velocity (que tem alto overlap com SE).

#### Cobertura combinada

| Cenário | Fraudes cobertas | % |
|---|---|---|
| Só SE | 262 | 73.8% |
| Só BEH | 228 | 64.2% |
| **SE ∪ BEH** | **281** | **79.2%** |
| Nenhum | 74 | 20.8% |

A união SE + BEH cobre **+19 fraudes** vs SE sozinho (73.8% → 79.2%). As 74 restantes são irrecuperáveis com dados atuais.

---

## 5. Integração no Pipeline

### 5.1 Fluxo operacional (Pipeline v1.3 / Engine v2.2)

```
Transação → Feature Engineering → features_dict
                                       │
                        ┌──────────────┼──────────────┐
                        ▼              ▼              ▼
                   SE v3.3        BEH v3.0       LGBM v4.1
                   (<1ms)         (<1ms)          (~2ms)
                        │              │              │
                        └──────────────┼──────────────┘
                                       ▼
                              Decision Engine v2.2
                                       │
                  ┌────────────────────┼────────────────────┐
                  ▼                    ▼                    ▼
           Fase 7: SE              Fase 7b: BEH        Vetos v2.2
           Agravantes              Agravantes           SE ≥ 60 → CONFIRMAR
           granulares              por categoria         BEH velocity ≥ 40 → CONFIRMAR
           por padrão              velocity=4            SE≥60 + BEH≥25 → BLOQUEAR
           (max 3)                 dormancy=3
                                   profile=1
                  │                    │                    │
                  └────────────────────┼────────────────────┘
                                       ▼
                                 Score Final → Decisão
                                 (APROVAR/CONFIRMAR/BLOQUEAR)
```

### 5.2 Integração como agravante (Decision Engine v2.2)

O BEH contribui para o score final via agravantes na Fase 7b, com peso proporcional à categoria dos fatores:

| Cenário | Peso agravante | Exemplo |
|---|---|---|
| **velocity + score ≥ 40** | 4-5 pts | FREQUENCIA_BURST ativou (score=25) + MULTIPLOS_RECEBEDORES_BURST (score=45 total) |
| **dormancy + score ≥ 25** | 3 pts | CONTA_DORMANTE_VALOR_ALTO + CONTA_DORMANTE_IDOSO (score=45) |
| **score 20-24** | 2 pts | CONTA_DORMANTE_VALOR_ALTO sozinho (score=20) |
| **score 15-19** | 1 pt | PRIMEIRA_TX_VALOR_ALTO sozinho (score=15) |
| **profile only (score 5)** | 0 pts | PERFIL_VULNERAVEL_SE abaixo do threshold (não gera agravante) |

### 5.3 Regras de veto do BEH (Engine v2.2)

| Regra | Condição | Ação |
|---|---|---|
| **Convergência SE + BEH** | SE ≥ 60 AND BEH ≥ 25 | → mínimo BLOQUEAR |
| **BEH velocity alto** | BEH ≥ 40 AND tem fator velocity | → mínimo CONFIRMAR |

### 5.4 Output para analista/CX

Quando o BEH detecta fatores, o output é humano-legível:

```json
{
  "behavioral": {
    "behavioral_score": 45,
    "risk_level": "ALTO",
    "risk_factors": [
      {
        "codigo": "CONTA_DORMANTE_VALOR_ALTO",
        "descricao": "Conta com apenas 1 tx no trimestre fazendo PIX de R$3.500,00 — padrão de conta dormante comprometida",
        "peso": 4,
        "source": "dormancy",
        "precision": 0.65,
        "origin": "B2"
      },
      {
        "codigo": "CONTA_DORMANTE_IDOSO",
        "descricao": "Idoso (72 anos) com conta inativa (1 tx/trim) fazendo PIX de R$3.500,00 — alto risco de engenharia social",
        "peso": 5,
        "source": "dormancy",
        "precision": 0.856,
        "origin": "B2"
      }
    ]
  }
}
```

---

## 6. Metodologia de Calibração

### 6.1 Processo de frentes

| Frente | Objetivo | Resultado |
|---|---|---|
| **B1: Validação Retroativa** | Rodar BEH v2.1 em 100.355 tx e medir tudo | Diagnóstico devastador: anti-discriminativo, 78k FP, 7 fatores mortos |
| **B2: Exploração + Redesign** | Explorar novos indicadores + candidatos a fatores | Descoberta de `baixa_freq_trim` (Lift 118.6x) e 3 fatores dormancy |
| **B2b: Overlap** | Medir Jaccard BEH × SE por fator | Confirmou complementaridade dos velocity e unicidade dos dormancy |
| **B3: Implementação v3.0** | Implementar os 7 fatores validados | behavioral_analytics.py v3.0 — código limpo, dataclasses atualizadas |
| **B4: Re-validação** | Rodar validação retroativa com v3.0 | Números confirmados: Prec 82% (thr>20), 19 exclusivas, 0 anti-indicadores |
| **B5: Integração** | Integrar no Engine v2.2 e Pipeline v1.3 | Fase 7b granular + novos vetos SE/BEH |
| **B6: Documentação** | Este documento |

### 6.2 Critérios de inclusão/exclusão

**Para fatores sobreviventes (B1):**
- Inclusão: Lift ≥ 1.5x AND Precision > 0% AND TP > 0
- Exclusão: Lift < 1.0 (anti-indicador), 0 ativações (morto), 0 TP (ruído puro)

**Para fatores novos (B2):**
- Exploração exaustiva: 35 indicadores booleanos × 231 pares × 648 trincas
- Critério: Precision ≥ 35% E F1 ≥ 0.30 E NewInvisible ≥ 0
- 3 candidatos aprovados (os 3 dormancy)
- Validados com re-execução completa em 100.355 tx

**Para scoring:**
- Score proporcional à precision empírica: +25 (Prec ≥ 85%), +20 (Prec 35-80%), +15 (Prec ≥ 70% sem incremento), +5 (condicional)
- Sem deduplicação entre fatores (dimensões distintas)
- Threshold ótimo: >20 (max F1 = 0.687)

### 6.3 Hipóteses rejeitadas (documentação de decisões negativas)

| Hipótese | Resultado | Motivo da rejeição |
|---|---|---|
| DEVICE_NOVO como fator útil | Lift **0.57x** — anti-indicador | `device_name` 100% missing → hash idêntico para todos, 78k FP |
| DEVICE_NOVO_PREMIUM | Lift **0.80x** — anti-indicador | Segmento premium ≠ device novo. 41k FP |
| DEVICE_NOVO_IDOSO | Precision 1% | Sem dados reais de device, não discrimina |
| PRIMEIRO_PIX_CLIENTE_NOVO | 0 TP, 82 FP | Puro ruído — cliente novo não implica fraude |
| 7 fatores session/login/renda | 0 ativações | Features 100% missing no dataset |
| Desativar módulo completamente | — | **Rejeitado** — 19 exclusivas dos dormancy justificam manutenção |
| BEH features como input do LGBM | Não testado | SE features já foram testadas (Frente SE-F4) e rejeitadas por redundância. BEH tem sinais similares |

---

## 7. Limitações Conhecidas

### 7.1 Dependência de dados ausentes

6 de 19 features monitoradas estão 100% missing no dataset atual:

| Feature | Status | Impacto |
|---|---|---|
| `device_name` | ❌ 100% missing | Impossibilita detecção de device novo real |
| `app_version` | ❌ 100% missing | Impossibilita análise de versão de app |
| `metodo_autenticacao` | ❌ 100% missing | Impossibilita análise de método de login |
| `tempo_interacao_ms` | ❌ 100% missing | Impossibilita detecção de sessão rápida/automatizada |
| `latencia_rede_ms` | ❌ 100% missing | Impossibilita detecção de VPN/acesso remoto |
| `ip_address` | ❌ 100% missing | Impossibilita geolocalização |

**Quando esses dados estiverem disponíveis**, os fatores de device/session podem ser reativados com calibração empírica. A infraestrutura (Profile Manager, DeviceInfo, SessionMetrics) já está implementada.

### 7.2 Fraudes irrecuperáveis

**74 fraudes (20.8%)** permanecem invisíveis a SE + BEH combinados. Perfil dominante:

- Valor < R$500 (abaixo dos gates)
- Sem burst (comportamento "normal")
- Conta com atividade regular (>2 tx/trimestre — não dormante)
- Sem padrão SE reconhecível

Requerem dados adicionais: grafo de recebedores, device fingerprint, behavioral biometrics.

### 7.3 PERFIL_VULNERAVEL_SE como gerador de FP

1.669 FP (93% dos FP totais do BEH) vêm exclusivamente deste fator. O score baixo (+5) minimiza o impacto, mas se o FPR global do BEH for crítico, considerar remoção. Mantido por:
- Score +5 não muda decisões sozinho
- Funciona como "boost" quando combinado com dormancy/velocity
- Será mais útil quando dados de device/session estiverem disponíveis

### 7.4 Overlap parcial com SE

| Fator BEH | SE equivalente | Jaccard | Comentário |
|---|---|---|---|
| FREQUENCIA_BURST | BURST_INTENSO_RAPIDO | 0.623 | Alto overlap, mas 29 exclusivas BEH |
| FREQUENCIA_BURST | ESVAZIAMENTO_CONTA | 0.511 | Alto overlap |
| BURST_CONTA_COMPROMETIDA | BURST_ESVAZIAMENTO_CONTA | 0.182 | Baixo overlap |
| MULTIPLOS_RECEBEDORES_BURST | ESVAZIAMENTO_CONTA | 0.353 | Moderado, 23 exclusivas |
| PRIMEIRA_TX_VALOR_ALTO | PRIMEIRA_TX_SUSPEITA | ~1.0 | **Overlap quase total** |

**PRIMEIRA_TX_VALOR_ALTO** é o mais redundante (Jaccard ~1.0 com SE). Mantido por defesa em profundidade e porque o score (+15) é moderado. Se simplificação for prioridade, é o primeiro candidato a remoção.

### 7.5 Profile Manager começa vazio

Em restart do serviço, o cache é zerado. Não afeta os 7 fatores atuais (todos usam features do pipeline, não do cache), mas afetará fatores futuros de device/session. Solução: persistir profiles em Redis/DynamoDB.

---

## 8. Plano de Manutenção

| Atividade | Frequência | Responsável |
|---|---|---|
| Re-rodar `avaliar_behavioral_retroativo.py` com dados novos | Trimestral | Data Science |
| Verificar se Lift dos fatores degradou | Trimestral | Data Science |
| Reavaliar features 100% missing (device, session) | Quando dados estiverem disponíveis | Data Science + Engenharia |
| Explorar novos fatores com `behavioral_b2_exploracao.py` | Semestral | Data Science |
| Atualizar thresholds no Decision Engine | Quando BEH for recalibrado | Data Science |
| Monitorar FPR do PERFIL_VULNERAVEL_SE | Trimestral | Data Science |
| Persistir Profile Manager em Redis | Quando migrar para produção | Engenharia |

---

## 9. Dados de Calibração (Referência)

### 9.1 Dataset

```
Arquivo:    base_mvp_model_ready_optimized.csv
Registros:  100.355 transações PIX
Fraudes:    355 (0.354%)
Normais:    100.000
Período:    Dez/2025 — Mar/2026
Fonte:      BRB (Banco de Brasília)
```

### 9.2 Features utilizadas pelos fatores v3.0

| Feature | Disponibilidade | Usado por |
|---|---|---|
| `burst_30m_flag` | ✅ 100% | FREQUENCIA_BURST, MULTIPLOS_RECEBEDORES_BURST |
| `tx_count_prev_30m` | ✅ 100% | FREQUENCIA_BURST, BURST_CONTA_COMPROMETIDA |
| `vl_pix` | ✅ 100% | BURST_CONTA_COMPROMETIDA, CONTA_DORMANTE_*, PRIMEIRA_TX_* |
| `qt_total_pix_trimestre` | ✅ 100% | CONTA_DORMANTE_VALOR_ALTO, CONTA_DORMANTE_IDOSO |
| `is_first_tx_trimestre` | ✅ 100% | PRIMEIRA_TX_VALOR_ALTO |
| `nr_idade` | ✅ 100% | CONTA_DORMANTE_IDOSO |
| `qt_tempo_relacionamento_mes` | ✅ 100% | BURST_CONTA_COMPROMETIDA |
| `first_receiver_flag` | ✅ 100% | BURST_CONTA_COMPROMETIDA |
| `distinct_receivers_so_far` | ✅ 100% | MULTIPLOS_RECEBEDORES_BURST |
| `perfil_vulneravel_se_flag` | ✅ 100% | PERFIL_VULNERAVEL_SE |

**Todas as features dos 7 fatores têm 100% de disponibilidade.** Não há risco de ativação parcial por dados faltantes.

---

## 10. Referências

### Acadêmicas e Técnicas
1. **Sundararamaiah et al.** (Dez/2024) — "Unifying AI and Rule-based Models for Financial Fraud Detection", IJCTT v72, DOI: 10.14445/22312803/IJCTT-V72I12P107
2. **Ben Abid et al.** (2025) — "A Scalable Hybrid Approach to Detecting Fraud with ML", EUSIPCO 2025
3. **Vallarino** (Abr/2025) — "Detecting Financial Fraud with Hybrid Deep Learning: A Mix-of-Experts Approach", arXiv:2504.03750

### Indústria de Fraude
4. **NICE Actimize** (2025) — "Behavioral Analytics for Fraud Detection: From Rule-Based to Adaptive Systems"
5. **Featurespace** (2025) — "Adaptive Behavioral Analytics (ARIC)" — Framework de referência para detecção por desvio de perfil
6. **BioCatch** (2025) — "Behavioral Biometrics for Account Takeover Detection" — Validação conceitual de conta dormante como vetor de ATO
7. **Flagright** (Jan/2026) — "AI vs Rules-Based Transaction Monitoring: Why a Hybrid Approach Wins"

### Dados de Mercado e Regulatórias
8. **ADDP** (2025) — 28 milhões de vítimas de golpes PIX em 2025; 53% >50 anos
9. **Febraban** (2025) — Falsa central telefônica: golpe #1 em valor médio (R$7.500/vítima)
10. **Febraban / Deloitte** (2025) — "Tendências em Prevenção a Fraudes" — análise comportamental como prioridade #2 após ML
11. **Banco Central do Brasil** — Resolução BCB nº 493 (Ago/2025): MED 2.0
12. **Banco Central do Brasil** — Resolução BCB nº 403/2024: Limites noturnos para PIX

### Scripts de Calibração (Artefatos Internos)
13. `avaliar_behavioral_retroativo.py` v2.0 — Validação retroativa (Frente B1 + B4)
14. `behavioral_b2_exploracao.py` — Exploração de novos fatores (Frente B2)
15. `behavioral_b2_candidatos_pares.csv` — 231 pares testados
16. `behavioral_b2_candidatos_trincas.csv` — 648 trincas testadas
17. `behavioral_validacao_metricas.json` — Métricas oficiais v3.0
18. `behavioral_validacao_por_fator.csv` — Performance por fator
19. `behavioral_validacao_relatorio.html` — Relatório visual

---

*Documento gerado em 2026-04-11. Módulo BEH v3.0 calibrado com 100.355 transações (355 fraudes confirmadas). Todos os fatores empiricamente validados.*

---