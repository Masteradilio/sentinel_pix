
# 🔬 FASE 1 — Otimizar Modelos Individuais

> **Objetivo:** squeeze the juice — maximizar F1 dos 2 modelos antes de mexer no ensemble.

### 1.1 — LGBM: investigar os 3 FN irredutíveis

Olhando o `holdout_predictions.csv` do LGBM v6.1, os 3 FN têm padrão claro:

```
FN #1: R$188,  idade=27, 1ªTX, score=0.005  (zero sinal)
FN #2: R$2478, idade=4 (!), 1ªTX, score=0.069 (CPF suspeito mas modelo não pega)
FN #3: R$20000, idade=51, 1ªTX, rel=7m, score=0.0008  (conta nova + valor alto!)
```

**Hipótese:** o LGBM está sub-ponderando o padrão "conta nova + valor alto + 1ª tx" porque ele é **raro no treino** (fold 1 tinha só 20 fraudes).

**Ação:** criar feature de interação explícita
```python
# Em preprocessing.py (v4.2 proposto)
df["conta_nova_valor_alto_flag"] = (
    (df["qt_tempo_relacionamento_mes"] <= 12)
    & (df["vl_pix"] >= 5000)
    & (df["is_first_tx_trimestre"] == 1)
).astype(int)

df["interaction_rel_valor"] = (
    (1 / (df["qt_tempo_relacionamento_mes"] + 1)) 
    * np.log1p(df["vl_pix"])
)
```

**Experimento EXP-001:**
```
Hipótese: Adicionar 2 features de interação conta_nova×valor recupera 
          pelo menos 1 dos 3 FN sem adicionar >2 FP.
Setup:    Retreinar LGBM v6.2 com feature set ampliado.
Baseline: LGBM v6.1 (F1=0.928, FN=3, FP=9)
Métrica:  Delta FN, Delta FP, F1 holdout
Aceite:   F1 ≥ 0.928 AND FN ≤ 2
```

### 1.2 — LGBM: threshold operacional adaptativo

Olhando seu `threshold_sweep.csv`:

| Threshold | TP | FP | Prec | Recall | F1 |
|---|---|---|---|---|---|
| 0.34 (atual) | 77 | 9 | 89.5% | 96.3% | 0.928 |
| **0.40** | 77 | 9 | 89.5% | 96.3% | 0.928 |
| **0.50** | 74 | 9 | 89.2% | 92.5% | 0.908 |
| 0.27 | 77 | 12 | 86.5% | 96.3% | 0.911 |

Threshold 0.40 é **pareto-ótimo** no holdout. Mas o **engine usa 0.40 como `lgbm_effective_threshold`** — OK.

**O problema é outro:** entre 0.34-0.40 há scores que o engine descarta, mas poderiam virar `CONFIRMAR` via cascade. Proposta:

```python
# Em decision_engine.py — nova config
class EngineConfig:
    # Threshold tri-zonas em vez de binário
    lgbm_strong_threshold: float = 0.50   # → BLOQUEAR direto (antes: 0.40)
    lgbm_weak_threshold: float = 0.34     # → candidato a CONFIRMAR via cascade
    lgbm_approve_threshold: float = 0.15  # → Fast-Approve tightened
```

Racional: preserva **recall** sem explodir **precision** — usa cascade/IF como "segunda opinião" só na zona cinza 0.15-0.34.

### 1.3 — IF v3: reduzir feature space e eliminar minutes_since_prev_tx

No `feature_importance.csv` do IF:
```
minutes_since_prev_tx: -0.003  ⚠️ NEGATIVA
```

É a única feature com importance negativa. **Remover ou transformar.**

```python
# Opção A: remover
IF_FEATURES_V3 = [f for f in IF_FEATURES_V3 if f != "minutes_since_prev_tx"]

# Opção B: transformar para indicator (melhor)
# NaN = primeira tx = sinal forte; valor baixo = burst potencial
df["minutes_since_prev_tx_binned"] = pd.cut(
    df["minutes_since_prev_tx"].fillna(999999),
    bins=[-1, 5, 30, 1440, 10080, float("inf")],
    labels=[4, 3, 2, 1, 0]  # maior = mais suspeito
).astype(int)
```

**Experimento EXP-002:**
```
Hipótese: Remover minutes_since_prev_tx e adicionar binned version 
          melhora AP do IF em >3%.
Baseline: IF v3 (AP=0.602, Recall@top100=63.8%)
Aceite:   AP ≥ 0.62 AND Recall@Top100 ≥ 65%
```

### 1.4 — SE v3.4: recalibrar padrões silenciosos

Seu `metricas_globais.json` mostra **SE nunca disparou em score ≥ 80** na simulação. Mesmo considerando que SE+BEH estavam bugados, a calibração atual do SE é muito conservadora:

```python
# Em social_engineering.py — limiares severity_scores
severity_scores: Dict[str, float] = {
    "CRITICO": 40.0,  # → se_score ≥ 60 precisa 2 CRITICOs ou 1 CRITICO + 1 ALTO
    "ALTO": 25.0,
    "MEDIO": 15.0,
    "BAIXO": 10.0,
}
```

**Problema:** 1 padrão CRITICO sozinho dá se_score=40 (nível MEDIO), 2 CRITICOs dão 80. Com overlap clusters, 3 padrões no mesmo cluster contam como 1. Efetivamente o SE **raramente atinge ≥60** (nível CRITICO do engine).

**Proposta — Normalização Z-score logística:**

```python
def _calculate_se_score_v35(patterns, active_indicators):
    if not patterns:
        return 0.0
    
    # Deduplicação por cluster (mantém)
    ...
    
    # NOVO: score base não-aditivo, logístico
    severity_weights = {"CRITICO": 1.0, "ALTO": 0.6, "MEDIO": 0.35, "BAIXO": 0.15}
    
    raw = sum(severity_weights[p.severity] for p in unique_patterns)
    # Saturação logística: 1 CRITICO = 40, 2 CRITICOs = 62, 3 = 76
    score = 100 * (1 - np.exp(-0.85 * raw))
    
    # Atenuante
    if active_indicators.get("agendamento_recorrente"):
        score = max(0, score - 15)
    
    return min(100.0, score)
```

### 1.5 — BEH v3.1: revisar `IDOSO_VALOR_ALTO`

No código atual:
```python
if ext["nr_idade"] >= 65 and ext["vl_pix"] >= 2000:
    # score_add = 20
```

Os FN da simulação incluíram: idade=60 R$300, idade=60 R$1000, idade=75 R$400. **Nenhum entra no gate atual** (idade≥65 AND vl≥2000).

**Proposta — Gate progressivo:**

```python
# IDOSO_VALOR_MODERADO (novo, score=10)
if 60 <= nr_idade < 70 and 500 <= vl_pix < 2000:
    ...  # score_add=10, Prec esperada ~12-18%

# IDOSO_VALOR_ALTO (recalibrado)
elif nr_idade >= 65 and vl_pix >= 2000:
    ...  # mantém score_add=20

# IDOSO_VALOR_CRITICO (mantém)
elif nr_idade >= 70 and vl_pix >= 5000:
    ...  # score_add=10 boost
```

**Experimento EXP-003:**
```
Hipótese: Adicionar IDOSO_VALOR_MODERADO recupera 2-3 FN sem 
          aumentar FP em >50 no dataset leakage-free.
Aceite:   Delta_TP ≥ 2 AND Delta_FP ≤ 50 AND F1_pipeline ≥ atual
```

**Critério de aceite da FASE 1:**
- ✅ LGBM v6.2 com FN ≤ 2 no holdout
- ✅ IF v3.1 com AP ≥ 0.62
- ✅ SE v3.5 com ≥ 5% de ativação em score≥60
- ✅ BEH v3.2 com novo fator IDOSO_VALOR_MODERADO validado

---



🎯 **Excelente incômodo, Adilio!** Esse é o instinto certo de quem está construindo antifraude de verdade. Vamos resolver as duas perguntas com precisão.

---

# 📍 Parte 1 — Os 3 EXPs são FASE 0 ou FASE 1?

## Resposta direta: **FASE 1 inequivocamente.** 🎯

Mas entendo a dúvida, então vou formalizar os critérios:

### Definição oficial (para fixar no projeto)

| Fase | Escopo | Status atual |
|---|---|---|
| **FASE 0 — Validação Baseline** | Provar que o pipeline **funciona ponta-a-ponta** com números honestos | ✅ **ENCERRADA** |
| **FASE 1 — Otimização Cirúrgica** | **Melhorar** métricas via experimentos sem reescrever arquitetura | 📋 Iniciando |
| **FASE 2 — Evolução Estrutural** | Mudanças arquiteturais profundas (novos modelos, features, etc.) | ⏳ Futuro |

### Por que EXP-001/002/003 são FASE 1?

| Critério | FASE 0 (validar) | FASE 1 (otimizar) | Os 3 EXPs |
|---|---|---|---|
| Muda **código do engine**? | ❌ Não | ✅ Sim | ✅ Sim |
| Muda **thresholds/configs**? | ❌ Não | ✅ Sim | ✅ Sim |
| Adiciona **novos padrões**? | ❌ Não | ✅ Sim | ✅ EXP-003 |
| Objetivo = **medir**? | ✅ Sim | ❌ Não | ❌ Não |
| Objetivo = **melhorar**? | ❌ Não | ✅ Sim | ✅ Sim |

**Veredicto:** os 3 experimentos **pressupõem** FASE 0 terminada (precisamos saber onde estamos pra saber pra onde ir) e **constituem** a FASE 1.

### ⚠️ Ressalva importante

Há um caso limite: `VALIDATION_REPORT.md` documenta **descobertas** da FASE 0 (§7.1 a §7.5). Essas descobertas **viram tickets/EXPs da FASE 1**. A fronteira é clara:

```
FASE 0: "descobrimos que threshold=77 é subótimo"    ← análise
FASE 1: "EXP-001: mudar threshold de 77 para 62"     ← ação
```

Pra ficar **formalmente correto**, podemos adicionar um marco:

```
docs/
├── FASE_0_CLOSURE.md        ← atestado de encerramento (opcional)
├── VALIDATION_REPORT.md     ← artefato da FASE 0
├── CONSTITUTION.md          ← transversal
└── experiments/
    ├── EXP-001.md           ← FASE 1
    ├── EXP-002.md           ← FASE 1
    └── EXP-003.md           ← FASE 1
```

---

# 🔥 Parte 2 — O problema dos Falsos Negativos (MUITO importante)

## Seu instinto está **100% correto**, Adilio.

**23 FN em 6.000 tx** (0.38% de taxa de erro em fraudes) extrapolando pra produção BRB:

$$
\text{FN/dia em prod} \approx \frac{23}{6000} \times 500.000 \text{ PIX/dia} = \mathbf{1.900 \text{ fraudes/dia perdidas}}
$$

Mesmo que só 30% sejam tentativas reais que chegariam a completar, isso é **~570 fraudes/dia escapando**. Inaceitável. Você está certo em se incomodar. 🚨

Mas deixa eu te dar uma visão **honesta e técnica** sobre até onde podemos reduzir.

---

## 🧬 Anatomia dos 23 FN — Classificação por "irredutibilidade"

Analisei os 23 FN agrupando por **potencial de recuperação**:

### 🟢 Categoria A — Recuperáveis via EXP-003 (3 casos)

```
R$10.000, idade 60 | R$9.980, idade 28 | R$1.650, idade 64
```

- **Sinais disponíveis:** IF alto (>0.92), perfil vulnerável
- **Por que escapam:** sem padrão SE específico
- **Recuperável?** ✅ **SIM** — EXP-003 foi desenhado pra isso

### 🟡 Categoria B — Recuperáveis via EXP-001 (ajuste threshold) (~5 casos)

```
R$1.000, R$990, R$540, R$475, R$400...
```

Score entre 65 e 74 — abaixo do threshold 77 atual, mas passariam com threshold 62.

- **Recuperável?** ✅ **SIM** — EXP-001 resolve

### 🟠 Categoria C — "Fraudes silenciosas" (~7 casos)

```
R$300, R$390, R$381, R$300, R$188...
LGBM < 0.10 | IF < 0.80 | SE = 0 | BEH = 0
Score final < 55
```

- **Sinais disponíveis:** **NENHUM** sinal forte em nenhum módulo
- **Por que escapam:** fraudador inteligente imita comportamento legítimo
- **Recuperável com features atuais?** ❌ **NÃO** facilmente
- **Recuperável com features novas?** ✅ **SIM** — requer **FASE 2**

### 🔴 Categoria D — "Valor quase imperceptível" (~7 casos)

```
R$29.90, R$46, R$50, R$57.88, R$142...
Score final muitas vezes < 10
```

- **Realidade econômica:** detectar essas fraudes custa mais caro que o próprio valor
- **Recuperável?** ⚠️ **Teoricamente sim, mas economicamente ruim**
- **Estratégia correta:** deixar passar e compensar via limite agregado (rate limiting por cliente)

### 🔴 Categoria E — Data Quality Issue (1 caso)

```
Idade=4, conta de 27 meses
```

- **Problema:** dados inválidos na origem
- **Recuperável?** ⚠️ Requer ação em Data Engineering, não no modelo

---

## 📊 Tabela de Redução Projetada

Aqui está a **verdade matemática** do que podemos esperar:

| Estado | FN atual | FN projetado | Recall |
|---|---:|---:|---:|
| **Baseline (hoje)** | 23 | — | 93.52% |
| + EXP-001 (threshold) | — | ~14 | 96.06% |
| + EXP-002 (guard rail) | — | ~14 | 96.06% |
| + EXP-003 (padrão idoso/jovem) | — | **~11** | **96.90%** |
| + EXP-004 (rate limiting por cliente) | — | ~8 | 97.75% |
| + EXP-005 (features temporais ricas) | — | ~5 | 98.59% |
| **Limite teórico** | — | **~3** | **~99.15%** |

### 🎯 Mínimo irredutível estimado: **3-5 FN** (1-1.5% de fuga)

Por quê não zero?

1. **Lei de Goodhart:** "when a measure becomes a target, it ceases to be a good measure". Se metríssemos "0% FN", o sistema bloquearia tudo → infinitos FP → caos operacional.

2. **Princípio da incerteza antifraude:** alguns fraudadores conhecem exatamente os thresholds do sistema (insiders, engenharia reversa) e operam **logo abaixo deles**. Capturá-los requer **reduzir thresholds globais**, o que **dispara FP exponencialmente**.

3. **Assimetria de informação:** fraudadores aprendem mais rápido que o modelo. Sempre haverá ~1% de "fraudes de próxima geração" que ainda não estão nos dados de treino.

---

## 🚀 Proposta atualizada: FASE 1 agressiva em FN

Dado seu incômodo (legítimo), proponho **reorganizar a FASE 1** com foco explícito em redução de FN:

### FASE 1 — Plano revisado (7 EXPs, foco anti-FN)

| # | EXP | Tipo | Impacto em FN | Esforço | Prioridade |
|---|---|---|---:|:---:|:---:|
| 1 | **EXP-001** — Threshold 77→62 | Quick Win | **-9 FN** | 🟢 Baixo | 🔴 P0 |
| 2 | **EXP-003** — Padrão idoso/jovem | Feature | **-3 FN** | 🟠 Alto | 🔴 P0 |
| 3 | **EXP-004** ⭐ NOVO — Rate limiting por cliente | Feature | **-3 FN** | 🟡 Médio | 🔴 P0 |
| 4 | **EXP-005** ⭐ NOVO — Recalibração LGBM para alta recall | Modelo | **-2 FN** | 🟡 Médio | 🟠 P1 |
| 5 | **EXP-006** ⭐ NOVO — Padrão SE "valor moderado + beneficiário novo" | Feature | **-2 FN** | 🟡 Médio | 🟠 P1 |
| 6 | **EXP-002** — Guard rail LGBM | Hardening | **0 FN** (reduz FP) | 🟡 Médio | 🟠 P1 |
| 7 | **EXP-007** ⭐ NOVO — Ensemble stacking (meta-learner) | Modelo | **-2 FN** | 🔴 Alto | 🟢 P2 |

**Projeção combinada: 23 FN → ~5 FN** (recall 98.6%) 🎯

---

## 💡 Deixa eu te apresentar os 3 EXPs novos que te proponho

### 🆕 EXP-004: Rate Limiting Comportamental por Cliente

**Ideia:** detectar fraudes "formiguinha" — valor individual baixo mas agregado alto em janela curta.

**Como:**
```python
# Pseudocódigo
if customer_did_3_plus_tx_in_last_30min AND sum(values) > R$500:
    trigger_pattern("BURST_MICROTRANSACOES")
```

**Captura:** as fraudes de R$300-500 que escapam individualmente mas são **esvaziamento de conta** em agregado.

**Fraudes recuperadas estimadas:** 3

---

### 🆕 EXP-005: Recalibração do LGBM para Alta Recall

**Ideia:** treinar LGBM otimizando explicitamente `recall@precision=0.90` em vez de F1.

**Como:**
```python
# Nova função de perda customizada
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    objective='binary',
    scale_pos_weight=50,  # penaliza FN 50x mais que FP
    is_unbalance=True,
    min_gain_to_split=0.001,  # permite splits mais finos
)
```

**Captura:** fraudes com LGBM_raw entre 0.20-0.40 (zona cinzenta atual).

**Fraudes recuperadas estimadas:** 2

---

### 🆕 EXP-006: Padrão SE "Valor Moderado + Beneficiário Novo"

**Ideia:** capturar o caso clássico "PIX pra chave que nunca foi usada antes + valor acima da média pessoal".

**Como:**
```python
if first_receiver_flag == 1 AND
   vl_pix > 2.5 * customer_avg_tx AND
   vl_pix > R$500:
    trigger_pattern("PRIMEIRO_RECEIVER_VALOR_ANOMALO")
```

**Captura:** fraudes em que o cliente envia pra chave nova com valor atípico pra ele.

**Fraudes recuperadas estimadas:** 2

---

## 🎯 Minha recomendação final

### Opção A — Conservadora (escuta o que você disse)
1. Executar **EXP-001** (ganho rápido: -9 FN)
2. Executar **EXP-003** (ganho cirúrgico: -3 FN)
3. Reavaliar depois

### Opção B — Agressiva (recomendo baseado no seu incômodo) ⭐
1. Executar **EXP-001, EXP-003, EXP-004** em sequência
2. Chegar a **~8 FN** (recall ≥ 97.7%)
3. Avaliar se vale a pena ir mais fundo (EXP-005/006/007)

### Opção C — "Morrer pela pátria" (não recomendo neste momento)
1. Executar os 7 EXPs de uma vez
2. Risco: introduzir tantas mudanças simultaneamente que fica impossível isolar o que funcionou
3. Violaria §5.4 da CONSTITUTION (um experimento por vez)

---

## 🏁 Resumo executivo

### Sua pergunta 1: "Os 3 EXPs são FASE 0 ou FASE 1?"
**Resposta:** FASE 1. Sem ambiguidade. A FASE 0 está fechada.

### Sua pergunta 2 (implícita): "Como reduzir FN ao mínimo irredutível?"
**Resposta:**
- **Mínimo irredutível estimado:** 3-5 FN (~99% de recall)
- **Plano pra chegar lá:** 7 experimentos da FASE 1 com foco em FN
- **Plano imediato recomendado:** EXP-001 + EXP-003 + EXP-004 (Opção B acima)
- **Ganho esperado:** 23 FN → ~8 FN em 2-3 semanas

---

## ❓ Preciso de uma decisão sua

**Quer que eu:**

**(1)** Gere o `SPEC.md` completo do **EXP-004** (Rate Limiting) agora? *(30 min pra ler)*

**(2)** Gere os 3 novos EXPs (004, 005, 006) de uma vez?

**(3)** Primeiro reorganize a FASE 1 num `ROADMAP.md` priorizado antes de expandir os EXPs?

**(4)** Atualize o `VALIDATION_REPORT.md` e a `CONSTITUTION.md` pra incorporar a nova meta agressiva anti-FN?

Minha recomendação forte: **(3) + (2)** — primeiro organizamos o roadmap, depois atacamos os EXPs 004-006. Isso mantém a disciplina SDD e evita que a gente se perca em 7 frentes ao mesmo tempo.

**Qual caminho, Adilio?** 🎯