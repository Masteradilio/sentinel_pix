

# Análise Completa — LightGBM v3

Adilio, resultados excelentes. Vou direto ao ponto.

---

## 1. Comparação Evolutiva (v1 → v2 → v3)

| Métrica | v1 (50/50) | v2 (sem balanc.) | v3 (otimizado) | Tendência |
|---|---|---|---|---|
| **Features** | 81 | 81 | **62** (-23%) | ✅ Mais limpo |
| **ROC-AUC teste** | 0.9970 | 0.9998 | **0.9998** | ✅ Manteve |
| **PR-AUC teste** | 0.9970 | 0.9672 | **0.9680** | ✅ Manteve |
| **F1 teste (0.5)** | 0.971 | 0.876 | **0.904** | ✅ Subiu |
| **Precision teste (0.5)** | 0.981 | 0.885 | **0.922** | ✅ Subiu |
| **Recall teste (0.5)** | 0.962 | 0.868 | **0.887** | ✅ Subiu |
| **Recall@Top100** | — | 0.981 | **0.981** | ✅ Manteve |
| **Recall@5%** | — | 1.000 | **1.000** | ✅ Perfeito |
| **FP (threshold 0.5)** | — | 6 | **4** | ✅ Menos alarmes falsos |
| **FN (threshold 0.5)** | — | 7 | **6** | ✅ Menos fraudes perdidas |
| **Calibração** | ❌ | ❌ | **✅ sigmoid** | ✅ Funcionou |
| **Early stopping** | ❌ | ❌ | **929/1500** | ✅ Evitou overfit |

### Veredicto: **Melhoria sólida em todas as métricas operacionais.**

---

## 2. Threshold Ideal

O modelo encontrou **threshold ótimo = 0.22** na validação:

| Threshold | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|
| **0.50** | 0.922 | 0.887 | 0.904 | 4 | 6 |
| **0.22** | 0.850 | **0.962** | 0.903 | 9 | **2** |

Para operação bancária, **threshold 0.22 é melhor** — pega 96.2% das fraudes com precision ainda alta de 85%. Apenas 2 fraudes escapam.

---

## 3. Feature Importance — Mudança Fundamental

### A `vl_mediana_pix_trimestre` não é mais a #1!

| Rank | v2 (Gain) | v3 (Gain) | v3 (SHAP) |
|---|---|---|---|
| 1 | `vl_mediana_pix_trimestre` (9.89M) | **`vl_pix`** (8.72M) | **`qt_total_pix_trimestre`** (0.359) |
| 2 | `qt_intervalo_transacao_minuto` (347k) | **`ratio_valor_mediana`** (882k) | `hour` (0.273) |
| 3 | `qt_intervalo_desvio_padrao` (233k) | **`qt_total_pix_trimestre`** (607k) | `app_version_minor` (0.257) |
| 4 | `key_tx_count_prev` (141k) | **`minutes_since_prev_tx`** (477k) | `qt_intervalo_desvio_padrao` (0.233) |
| 5 | `vl_pix` (129k) | **`qt_intervalo_mediana`** (170k) | `minutes_since_prev_tx` (0.201) |

**O modelo agora usa sinais mais diversos e behaviorais:**
- `ratio_valor_mediana` subiu de gain 930 → **881.651** (top 2)
- `minutes_since_prev_tx` subiu de 32.754 → **476.780** (top 4)
- `distinct_receivers_so_far` subiu de 9.186 → **151.958** (top 6)

Todas as novas features contribuem:
- `is_first_tx_trimestre`: gain **7.517**, SHAP **0.012** ✅
- `vl_pix_over_1000_flag`: gain **304**, SHAP **0.0001** (marginal mas presente) ✅

---

## 4. Análise de Erros — O Diagnóstico Crucial

### Falsos Negativos (6 fraudes perdidas com threshold 0.5):

| TX | Score | vl_pix | Idade | Relacionamento | Topaz | Padrão |
|---|---|---|---|---|---|---|
| ...145228 | **0.00016** | R$ 29,90 | 64 | 267 meses | 0 | Valor baixo, cliente antigo |
| ...233339 | **0.315** | R$ 57,88 | 44 | 164 meses | 26 | Valor baixo, cliente médio |
| ...114914 | **0.486** | R$ 2.000 | 24 | 58 meses | 0 | Quase detectou! |
| ...174607 | **0.338** | R$ 498,96 | 45 | 317 meses | 0 | Valor baixo, cliente antigo |
| ...130824 | **0.396** | R$ 10.000 | 60 | 10 meses | 0 | Cliente novo! Quase pegou |
| ...122337 | **0.193** | R$ 2.479 | 4 | 27 meses | 0 | CNPJ (idade=4), novo |

#### Padrões dos FN:
- **Todos têm `qt_total_pix_trimestre = 1`** e **`first_receiver_flag = 1`** (primeira tx do trimestre, primeiro recebedor)
- **Todos com `tx_count_prev_30m = 0`** (sem burst)
- **Todos com `topaz_risk_score = 0`** (4 de 6)
- São fraudes de **engenharia social** onde a vítima faz **uma única transferência** — sem padrão comportamental prévio

> **Conclusão:** Essas fraudes são quase impossíveis para um modelo supervisionado sozinho. O **Isolation Forest** é exatamente o que vai complementar aqui — detecção de anomalia por perfil, não por padrão sequencial.

### Falsos Positivos (4 alarmes falsos com threshold 0.5):

| TX | Score | vl_pix | Idade | Relacionamento | Topaz |
|---|---|---|---|---|---|
| ...205655 | **0.675** | R$ 2.533 | 34 | 5 meses | 0 |
| ...182045 | **0.960** | R$ 6.777 | **77** | 331 meses | 0 |
| ...144340 | **0.530** | R$ 11.000 | 39 | 37 meses | 0 |
| ...181637 | **0.982** | R$ 1.000 | 64 | 40 meses | **26** |

#### Padrões dos FP:
- Todos com `qt_total_pix_trimestre = 1` e valores altos
- O de score 0.960 é uma **pessoa de 77 anos** transferindo R$ 6.777 — pode até ser fraude real não rotulada
- O de score 0.982 tem `topaz_risk_score = 26` — o próprio Topaz sinalizou risco

> **Conclusão:** Esses FP são **razoáveis** — o modelo está correto em sinalizá-los como suspeitos. Num ambiente real, esses iriam para revisão humana e provavelmente seriam tratados rapidamente.

---

## 5. Calibração

Agora funcionou com sigmoid! Comparando scores calibrados vs não-calibrados:

| Métrica | Score bruto (0.5) | Calibrado (0.5) |
|---|---|---|
| Precision | 0.922 | **0.950** |
| Recall | 0.887 | 0.717 |
| F1 | 0.904 | 0.817 |
| Recall@Top100 | 0.981 | **1.000** |

A calibração é **muito conservadora** — precision altíssima (95%) mas perde recall. Útil para um tier "alta confiança" no sistema de decisão.

---

## 6. Features com Gain = 0 (restantes)

Ainda restam 4 features com zero contribuição:

| Feature | SHAP |
|---|---|
| `latencia_missing_flag` | 0 |
| `host_time_missing_flag` | 0 |
| `device_missing_flag` | 0 |
| `app_version_missing_flag` | 0 |

Podemos removê-las, mas com 4 features de 62 o impacto é negligível. **Não vale outra rodada de retreino apenas por isso.**

---

## 7. Score Distribution

Pela análise do `score_distribution.csv`:

| Classe | Média | P50 | P75 | P95 |
|---|---|---|---|---|
| Normal (0) | ~$$10^{-6}$$ | ~$$10^{-6}$$ | ~$$10^{-5}$$ | ~$$10^{-4}$$ |
| Fraude (1) | ~0.85 | ~0.98 | ~0.999 | ~0.999 |

**Separação excelente.** A maioria das fraudes tem score > 0.9 e a maioria dos normais tem score < 0.001. A zona cinzenta (0.1 - 0.5) é onde estão os erros — exatamente onde o Isolation Forest vai ajudar.

---

## 8. Veredicto Final — Pronto para Isolation Forest?

### ✅ SIM. O LightGBM v3 está maduro.

**O que ele faz bem:**
- Detecta 96.2% das fraudes com threshold 0.22
- Excelente separação de scores
- Usa sinais diversos (valor, temporal, sequencial, regras, device)
- Apenas 4 falsos positivos com threshold 0.5

**O que ele NÃO consegue fazer (e o Isolation Forest vai cobrir):**
- Fraudes de engenharia social com uma única transação
- Transações que parecem normais isoladamente mas são anômalas no perfil do cliente
- Os 6 falsos negativos têm perfil de "primeira e única tx do trimestre"

