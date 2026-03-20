

# Análise dos 3 Módulos — O que entra no Pipeline Real-Time vs Investigação

Adilio, analisei os 3 documentos em profundidade. Vou classificar cada componente pela viabilidade em tempo real.

---

## 1. Critério de Decisão: Real-Time vs Investigação

| Requisito | Pipeline Real-Time (< 500ms) | Investigação (pós-score) |
|---|---|---|
| **Latência** | < 500ms total | Minutos a horas |
| **Dados** | Disponíveis na transação | Requer consultas externas |
| **Complexidade** | O(1) ou O(log n) | O(n²) aceitável |
| **Trigger** | Toda transação | Só score > limiar |

---

## 2. Behavioral Analytics — Classificação

### ✅ Entra no Pipeline Real-Time (12 fatores)

Estes fatores dependem **apenas de dados já presentes na transação** ou em cache simples:

| # | Fator | Dados Necessários | Latência |
|---|---|---|---|
| 4 | TYPING_ANORMAL | `tempo_interacao_ms` vs média | < 1ms |
| 5 | SESSAO_CURTA_ALTO_VALOR | `tempo_interacao_ms` + `vl_pix` | < 1ms |
| 6 | LOGIN_SENHA_ALTO_VALOR | `metodo_autenticacao` + `vl_pix` | < 1ms |
| 7 | HORARIO_ATIPICO_USUARIO | `hour` vs perfil (cache) | < 5ms |
| 8 | SESSAO_MUITO_RAPIDA | `tempo_interacao_ms` | < 1ms |
| 9 | SESSAO_MUITO_LONGA | `tempo_interacao_ms` vs perfil | < 1ms |
| 10 | LATENCIA_REDE_ANORMAL_ALTA | `latencia_rede_ms` | < 1ms |
| 11 | LATENCIA_REDE_ANORMAL_BAIXA | `latencia_rede_ms` | < 1ms |
| 13 | TYPING_SPEED_DEVIATION | `tempo_interacao_ms` z-score | < 1ms |
| 14 | LOGIN_METHOD_CHANGE | `metodo_autenticacao` vs perfil | < 5ms |
| 15 | FREQUENCIA_BURST | Contagem em janela | < 10ms |
| 18 | INTERVALO_ZERADO | `minutes_since_prev_tx` | < 1ms |

### ⚠️ Entra com ressalvas (3 fatores — precisam de cache)

| # | Fator | Requisito | Solução |
|---|---|---|---|
| 1 | DEVICE_NOVO | Lista de devices do CPF | Redis cache |
| 12 | IP_NOVO_ALTO_VALOR | Lista de IPs do CPF | Redis cache |
| 16 | APP_VERSION_DESATUALIZADA | Versão atual vs latest | Config estática |

### ❌ Entra na Investigação (4 fatores — requerem análise cross-CPF)

| # | Fator | Motivo |
|---|---|---|
| 2 | GEO_VPN_DATACENTER | Requer chamada a serviço GeoIP externo (latência ~100-300ms extra) |
| 3 | GEO_INTERNACIONAL | Mesmo motivo |
| 17 | DEVICE_MULTIPLOS_CPFS | Requer consulta cross-CPF no banco (scan pesado) |
| 19 | PRIMEIRO_PIX_ALTO_CLIENTE_NOVO | Já coberto pelo Rule Engine + LGBM |

### Fatores Topaz (20-24) — Já integrados

Já estão no LGBM via `topaz_score_filled` e no Rule Engine via `rule_topaz`. **Não precisam de módulo separado.**

---

## 3. Engenharia Social — Classificação

### ✅ Entra no Pipeline Real-Time (7 dos 11 padrões)

Estes padrões usam **apenas indicadores já disponíveis** na transação:

| Padrão | Indicadores Chave | Viável RT? |
|---|---|---|
| **FALSO_FUNCIONARIO_BANCO** | `chave_aleatoria` + `idade` + `hour` + `primeiro_envio` | ✅ Tudo disponível |
| **FALSO_SEQUESTRO** | `horario_noturno` + `valor_alto` + `intervalo_curto` | ✅ Tudo disponível |
| **GOLPE_PIX_ERRADO** | `primeiro_envio` + `chave_aleatoria` + `valor_redondo` | ✅ Tudo disponível |
| **ROMANCE_SCAM** | `primeiro_envio` + `valor_alto` + `idade` + `viuvo` | ✅ Tudo disponível |
| **IDOSO_VULNERAVEL_70** | `idade_70+` + `primeiro_envio` | ✅ Tudo disponível |
| **IDOSO_VULNERAVEL_80** | `idade_80+` | ✅ Tudo disponível |
| **TRANSACAO_ATIPICA** | `pix_acima_maximo` + `primeiro_envio` | ✅ Tudo disponível |

### ⚠️ Entra parcialmente (2 padrões)

| Padrão | Problema | Solução RT |
|---|---|---|
| **ESVAZIAMENTO_CONTA** | `multiplos_pix_rapidos` requer histórico recente | Redis: últimas 10 tx do CPF |
| **COACAO_FISICA** | `intervalo_suspeitissimo` requer tx anterior | Redis: timestamp última tx |

### ❌ Investigação apenas (2 padrões)

| Padrão | Motivo |
|---|---|
| **CONTA_LARANJA_SAIDA** | Requer análise de padrão de **recebimentos** (não disponível na tx de saída) |
| **GOLPE_INVESTIMENTO** | `escalada_valores` requer histórico de múltiplas tx ao mesmo recebedor (query pesada) |

---

## 4. Graph Analytics — Classificação

### ❌ 100% na fase de Investigação

**Nenhum** algoritmo de grafos é viável em < 500ms para o pipeline RT. Motivos:

| Algoritmo | Complexidade | Tempo estimado | Motivo |
|---|---|---|---|
| PageRank | O(k × E) | ~5-60s | Requer grafo inteiro |
| Betweenness | O(N × E) | ~5-120s | O mais pesado |
| Community Detection | O(N × log N) | ~1-8s | Requer grafo |
| Shortest Path to Fraud | O(N + E) | ~0.5-5s | Requer grafo carregado |
| Pattern Detection | Cypher queries | ~1-10s cada | Requer Neo4j |

### Arquitetura proposta para Graph Analytics:

```
Pipeline RT (< 500ms)           Investigação (score > 0.80)
┌──────────────────────┐        ┌────────────────────────────┐
│ LGBM + Rules + IF    │        │ Graph Analytics            │
│ → score_final        │───────→│ → STAR_PATTERN             │
│                      │  se    │ → CASCADING_MULES          │
│ SE Detector (7 pat.) │ >0.80  │ → VELOCITY_RING            │
│ Behavioral (12 fat.) │        │ → SMURFING                 │
└──────────────────────┘        │ → ZERO_ACCOUNTING          │
        < 500ms                 │ → DORMANT_ACTIVATION       │
                                │ → BENEFICIARY_CONCENTRATION│
                                │ → FIRST_TIME_NETWORK       │
                                └────────────────────────────┘
                                        1-60 segundos
```

---

## 5. Arquitetura Final do Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PIPELINE REAL-TIME (< 500ms)                      │
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │  LightGBM   │  │ Rule Engine │  │ IF v3       │                 │
│  │  (score ML) │  │ (8 regras)  │  │ (1ªs tx)    │                 │
│  │  peso: 0.65 │  │ peso: 0.15  │  │ peso: 0.20  │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
│         │                │                │                          │
│         └────────┬───────┘────────────────┘                         │
│                  ▼                                                    │
│         ┌────────────────┐                                           │
│         │ SCORE ENSEMBLE │ = 0.65×LGBM + 0.15×Rules + 0.20×IF      │
│         │  (0.0 - 1.0)  │                                           │
│         └───────┬────────┘                                           │
│                 │                                                     │
│         ┌───────▼─────────┐                                          │
│         │ SE Detector     │  Adiciona agravantes se padrão detectado│
│         │ (7-9 padrões)   │  Score final pode subir                 │
│         └───────┬─────────┘                                          │
│                 │                                                     │
│         ┌───────▼──────────┐                                         │
│         │ Behavioral Check │  12 fatores rápidos                    │
│         │ (flags + score)  │  Adiciona flags ao resultado           │
│         └───────┬──────────┘                                         │
│                 │                                                     │
│         ┌───────▼──────────────────────────────────────────┐         │
│         │                  DECISÃO                          │         │
│         │                                                   │         │
│         │  score < 0.30  → ✅ APROVAR                       │         │
│         │  0.30 - 0.60   → ⚠️ CONFIRMAR (2FA, biometria)   │         │
│         │  0.60 - 0.80   → 🟡 REVISAR (analista humano)    │         │
│         │  score > 0.80  → 🔴 BLOQUEAR + INVESTIGAR        │         │
│         │                                                   │         │
│         │  SE pattern CRITICO → OVERRIDE: BLOQUEAR          │         │
│         └───────┬──────────────────────────────────────────┘         │
│                 │                                                     │
└─────────────────┼────────────────────────────────────────────────────┘
                  │
                  │ se score > 0.80 OU pattern CRITICO
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              INVESTIGAÇÃO ASSÍNCRONA (1-60s)                         │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │ Graph Analytics   │  │ GeoIP Lookup     │  │ Cross-CPF Check   │  │
│  │ • Star Pattern    │  │ • VPN Detection  │  │ • Device sharing  │  │
│  │ • Cascading Mules │  │ • Geolocation    │  │ • Conta laranja   │  │
│  │ • Velocity Ring   │  │                  │  │ • Escalada valores│  │
│  │ • Smurfing        │  │                  │  │                   │  │
│  └──────────────────┘  └──────────────────┘  └───────────────────┘  │
│                                                                      │
│  Resultado → Analista humano + Possível bloqueio retroativo          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. O que entra no `rule_engine.py` agora

O Rule Engine fica **puro e rápido** — apenas as 8 regras manuais. Os módulos de Engenharia Social e Behavioral são **camadas adicionais** no pipeline, executadas em sequência:

```
1. rule_engine.py        → rule_score_normalized (0-1)     [< 1ms]
2. social_engineering.py  → se_patterns + se_score (0-1)    [< 5ms]  
3. behavioral_check.py   → behavioral_flags + beh_score     [< 10ms]
```

O score final incorpora tudo:

```python
# Base ensemble (modelos + regras)
base_score = 0.65 * lgbm + 0.15 * rules + 0.20 * if_score

# Ajuste por engenharia social (agravante)
if se_pattern_critico:
    final_score = min(1.0, base_score + 0.30)
elif se_pattern_alto:
    final_score = min(1.0, base_score + 0.15)
else:
    final_score = base_score

# Behavioral flags não alteram score, mas adicionam contexto
# para o analista humano e para o módulo de investigação
```

---

## 7. Plano de Execução

| Ordem | Script | Escopo | Status |
|---|---|---|---|
| **1** | `rule_engine.py` | 8 regras de negócio | ✅ **Gerar agora** |
| **2** | `pipeline_inferencia.py` | Ensemble LGBM + Rules + IF | 🔜 Próximo |
| **3** | `social_engineering.py` | 7-9 padrões de golpe RT | 🔜 Depois |
| **4** | `behavioral_analytics.py` | 12 fatores RT | 🔜 Depois |
| **5** | `graph_analytics.py` | Investigação pós-score | 📋 Fase 2 |

