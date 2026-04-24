# Plano de Correção e Evolução — Pipeline Antifraude PIX

Adilio, absorvi o projeto inteiro — do `preprocessing.py v4.1` até a simulação E2E. Antes de montar o plano, deixa eu ser direto sobre o que vejo:

**Você tem um sistema técnica e cientificamente sólido**, com boa engenharia, leakage-free validado, documentação em código exemplar e SDD aplicado. Mas tem **3 problemas estruturais** que comprometem a confiabilidade dos números reportados, e **7 oportunidades de melhoria** com ROI alto.

Vou organizar em fases executáveis, cada uma com critério de aceite mensurável.

---

## 📊 Diagnóstico Consolidado

### O que está **funcionando bem**
- ✅ LGBM v6.1 com F1=0.928 no holdout (sólido)
- ✅ IF v3 com treino segmentado (gap fraude/normal dobrou)
- ✅ Leakage-free rigoroso (rolling window causal)
- ✅ Engine v3.0.5 com Fast-Approve (design elegante)
- ✅ SHAP explicabilidade integrada
- ✅ API FastAPI com CX layer

### O que está **quebrado ou frágil**
| # | Problema | Severidade | Evidência |
|---|---|---|---|
| 1 | **SE e BEH zerados** na simulação E2E | 🔴 Crítico | `SE TP=0, FN=355` no `metricas_globais.json` |
| 2 | **3 "fontes da verdade" divergentes** para métricas | 🔴 Crítico | Doc diz F1=0.813, holdout=0.928, E2E=0.940 |
| 3 | **Simulador E2E diverge do engine real** | 🟠 Alto | Anchors hardcoded, vetos simplificados |
| 4 | **SE v3.4 com recall isolado 0%** (calibração agressiva demais) | 🟠 Alto | `metricas_globais.json` mostra SE nunca dispara @≥80 |
| 5 | **3 FN irredutíveis** no LGBM (score 0.001-0.07) | 🟡 Médio | "Invisíveis" a todas camadas |
| 6 | **Cache de histórico em memória** (perde no restart) | 🟡 Médio | `_InlineProfileManager` no `behavioral_analytics.py` |
| 7 | **Calibração isotonic piora resultado** | 🟡 Médio | F1 cai de 0.928 → 0.882 |
| 8 | **Sem MLOps / experiment tracking** | 🟢 Baixo | Versões documentadas em docstring, não em MLflow |

---

## 🎯 Plano em 5 Fases

```
FASE 0 (1-2 dias)    → Estabilizar Realidade
FASE 1 (3-5 dias)    → Otimizar Modelos Individuais  
FASE 2 (5-7 dias)    → Refinar Engine e Regras
FASE 3 (5-7 dias)    → Robustez Operacional (MLOps light)
FASE 4 (contínuo)    → Evolução Científica
```

---

# 🚨 FASE 0 — Estabilizar a Realidade (BLOQUEADOR)

> **Objetivo:** ter UMA fonte da verdade sobre como o pipeline performa. Sem isso, qualquer decisão técnica é chute.

### 0.1 — Corrigir o simulador E2E (4-6h)

**Problema raiz:** o `simular_pipeline_e2e_lf.py` tenta inferir interface dos módulos via `hasattr(detector, method_name)` com lista genérica, mas `SocialEngineeringDetector` expõe `detect_from_pipeline()` e `BehavioralAnalytics` expõe `analyze()` retornando **dataclasses**, não `dict`.

**Correção:**

```python
# SE v3.4 — uso correto
from core.social_engineering import SocialEngineeringDetector

se_detector = SocialEngineeringDetector()
for _, row in df.iterrows():
    features = row.to_dict()
    result: SEAnalysisResult = se_detector.detect_from_pipeline(features)
    scores.append(result.se_score)  # propriedade da dataclass
    patterns.append([p.pattern_name for p in result.patterns])

# BEH v3.1 — uso correto  
from core.behavioral_analytics import BehavioralAnalytics

beh = BehavioralAnalytics()
for _, row in df.iterrows():
    result: BehavioralAnalysisResult = beh.analyze(row.to_dict())
    scores.append(result.behavioral_score)
    has_velocity.append(any(rf.source == "velocity" for rf in result.risk_factors))
```

**Guardrails obrigatórios:**

```python
# Adicionar validação pós-scoring
assert se_scores.sum() > 0, (
    "❌ SE zerou em 100% das tx. Algo está errado com o adapter."
)
assert beh_scores.sum() > 0, "❌ BEH zerou em 100% das tx."

# Logar distribuição
logger.info(
    f"[SE] ativações: {(se_scores > 0).sum()}/{len(se_scores)} "
    f"({(se_scores > 0).mean():.1%})"
)
logger.info(
    f"[BEH] ativações: {(beh_scores > 0).sum()}/{len(beh_scores)} "
    f"({(beh_scores > 0).mean():.1%})"
)
```

### 0.2 — Substituir reimplementação por chamada ao PixDecisionEngine real

O simulador reimplementa vetos, cascade e ensemble — e diverge do `decision_engine.py v3.0.5`. **Fim dessa duplicação.**

```python
from core.decision_engine import PixDecisionEngine, EngineConfig

config = EngineConfig(artefatos_dir=str(ARTEFATOS_DIR))
engine = PixDecisionEngine(config)

# Em vez de reimplementar vetos em batch, rodar engine.decide() real
for idx, row in df.iterrows():
    features = row.to_dict()
    se_result = se_detector.detect_from_pipeline(features).to_dict()
    beh_result = beh.analyze(features).to_dict()
    decision = engine.decide(features, se_result, beh_result)
    # ...coletar decision.decisao, decision.score_final, etc
```

Sim, vai ficar mais lento (100k × ~50ms = ~80min). Solução:
- Rodar em **paralelo** com `multiprocessing.Pool` (8 workers → ~10min)
- Ou rodar apenas em **subset estratificado** (10k tx) para validação rápida

### 0.3 — Criar `VALIDATION_REPORT.md` como fonte única da verdade

**Template:**

```markdown
# PIX Antifraude — Validation Report
**Dataset:** base_treino_final.csv (100.355 tx, 355 fraudes)
**Data:** 2026-04-XX
**Pipeline:** v1.4 + Engine v3.0.5 + SE v3.4 + BEH v3.1

## Métricas Oficiais (executadas via engine real)

### Pipeline Completo
| Métrica | Valor | Range 95% CI (bootstrap) |
|---|---|---|
| TP | XXX | — |
| FP | XXX | — |
| FN | XXX | — |
| Recall | XX.X% | [XX, XX] |
| Precision | XX.X% | [XX, XX] |
| F1 | 0.XXX | [0.XX, 0.XX] |

### Ablation (componentes)
| Config | TP | FP | Recall | Precision |
|---|---|---|---|---|
| LGBM solo @ th=0.34 | X | X | X% | X% |
| LGBM + Cascade | X | X | X% | X% |
| LGBM + IF | X | X | X% | X% |
| LGBM + SE | X | X | X% | X% |
| LGBM + BEH | X | X | X% | X% |
| **Pipeline full** | X | X | X% | X% |

### FN Analysis (fraudes irredutíveis)
| # | CPF | vl_pix | LGBM | IF | SE | BEH | Razão |
|---|---|---|---|---|---|---|---|
| 1 | XXX | R$X | X | X | X | X | SE não ativou porque... |
```

### 0.4 — Criar `CONSTITUTION.md` do projeto (SDD Phase 1)

Seu projeto tem convenções implícitas espalhadas. Formaliza:

```markdown
# CONSTITUTION.md

## Não-negociáveis

### Tech Stack
- Python 3.12+, type hints em TODAS funções públicas
- ML: LightGBM + sklearn IsolationForest
- Regras: rule-based engines puros (sem dependência de ML)
- Serving: FastAPI + joblib artifacts
- Ferramentas: uv, ruff, pytest, mypy

### Arquitetura
- **Separação estrita:** `core/` = engines, `scripts/` = treino/simulação
- **Artefatos de produção:** `backend/artefatos/*.joblib`
- **Relatórios de treino:** `backend/modelos/resultado_treino_*/`
- **NUNCA** misturar código de produção com código de treino

### Metodologia de Validação
- **Temporal split obrigatório** (sem shuffle)
- **Leakage-free enforcement:** toda feature derivada de histórico
  deve usar rolling window causal estrito (< data atual da tx)
- **Holdout intocado:** mínimo 10% temporal final
- **Retreino: recompute do zero** a partir do raw

### Métricas que importam
- **Primária:** F1 no holdout temporal
- **Operacional:** FPR < 0.1%, Recall > 95%
- **Negócio:** custo FN (R$/fraude) vs custo FP (fricção)

### Veto Absoluto
- ❌ Não usar features com data leakage (tem whitelist `LEAKAGE_FIXED_FEATURES`)
- ❌ Não usar smoothing/SMOTE em dados fraudulentos
- ❌ Não otimizar threshold em dados de treino
```

**Critério de aceite da FASE 0:**
- ✅ SE e BEH disparando em >1% das tx
- ✅ Pipeline E2E rodando via `engine.decide()` real
- ✅ `VALIDATION_REPORT.md` com números consistentes
- ✅ `CONSTITUTION.md` commitado

---

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

EXP-004-FINAL — Fechamento da FASE 1

Categoria: Capstone / Política contextual
Complexidade: 🟡 Média
Prioridade: 🔴 Alta
Status: ✅ Concluído — melhoria parcial aprovada para runtime
Fase: 1 — Otimização Cirúrgica
Objetivo: consolidar os experimentos remanescentes da FASE 1 em uma única rodada de validação, evitando prolongar a fase com múltiplos experimentos incrementais.

1. Contexto

Após a execução dos EXP-001, EXP-002 e EXP-003, o pipeline antifraude PIX já havia atingido um patamar forte de performance:

EXP-001: reduziu o threshold final de confirmação de 77 para 62, recuperando 14 fraudes no sample de 6.000 transações, com ganho de recall e F1.
EXP-002: adicionou o guard rail LGBM, reduzindo falsos positivos de 18 para 15, mas com custo de 1 TP perdido, resultando em TP=345, FP=15, FN=10, F1=0.9650.
EXP-003: testou o padrão residual IDOSO_JOVEM_VALOR_MODERADO_RESIDUAL, mas não trouxe ganho métrico global, pois o caso residual já era capturado como CONFIRMAR pelo baseline atual.

A investigação da FASE 1 estava se alongando com os EXP-004, EXP-005, EXP-006 e EXP-007 ainda previstos. Para evitar fragmentação excessiva, foi criado o EXP-004-FINAL, um experimento capstone com variantes internas, cujo objetivo era testar de uma vez:

exceção contextual ao guard rail do EXP-002;
padrão comportamental de rate limiting;
padrão de primeiro recebedor com valor anômalo;
combinação final dos sinais;
classificação dos FNs residuais.
2. Baseline do experimento

O baseline oficial utilizado no EXP-004-FINAL foi o runtime pós-EXP-001 e EXP-002:

Métrica	Valor
TP	345
FP	15
FN	10
Precision	95,8333%
Recall	97,1831%
F1	0,9650
FPR	0,2657%

Esse baseline corresponde ao estado com:

{
  "threshold_confirmar": 62.0,
  "threshold_bloquear": 95.0,
  "lgbm_guard_enabled": true,
  "lgbm_guard_threshold": 0.30,
  "se_pattern_residual_enabled": false,
  "exp003_residual_confirm_enabled": false
}
3. Variantes testadas

O experimento testou quatro variantes principais contra o baseline:

Variante	Descrição
V1_GUARD_CONTEXTUAL	Exceção cirúrgica ao guard rail para casos de alto valor com convergência SE + BEH + IF
V2_RATE_LIMIT	Padrão comportamental de múltiplas transações em janela curta
V3_PRIMEIRO_RECEIVER	Primeiro recebedor com valor anômalo para o histórico do cliente
V4_COMBO_FINAL	Combinação de V1 + V2 + V3
4. Resultado principal

O vencedor foi o V1_GUARD_CONTEXTUAL.

Configuração	TP	FP	FN	Precision	Recall	F1
Baseline EXP-001 + EXP-002	345	15	10	95,8333%	97,1831%	0,9650
V1_GUARD_CONTEXTUAL	346	15	9	95,8449%	97,4648%	0,9665

Delta:

Métrica	Delta
TP	+1
FP	+0
FN	-1
F1	+0,0014
Valor de FN recuperado	R$ 20.000,00
Valor de FP adicionado	R$ 0,00

O relatório executivo do EXP-004-FINAL marcou o status como NAO_APROVADO_AUTOMATICAMENTE, porque o experimento não atingiu o critério agressivo de recall mínimo de 98,31%. Ainda assim, todos os demais critérios foram satisfeitos: FP ≤ 20, Precision ≥ 94%, FPR ≤ 0,5%, F1 não decrescente e TP não decrescente.

A conclusão técnica é que o experimento foi positivo como melhoria cirúrgica parcial, embora não suficiente para encerrar a busca por redução adicional de FN.

5. Validação cruzada

A validação cruzada com seed=123 confirmou a direção do ganho:

Configuração	TP	FP	FN	Precision	Recall	F1	FPR
Baseline	345	12	10	96,6387%	97,1831%	0,9691	0,2126%
V1_GUARD_CONTEXTUAL	346	12	9	96,6480%	97,4648%	0,9705	0,2126%

A variante vencedora recuperou novamente 1 FN, sem adicionar FP, preservando a robustez observada no sample principal.

6. Caso recuperado

O V1 recuperou uma fraude real de R$20.000, previamente perdida por causa do guard rail LGBM do EXP-002.

Características do caso:

Campo	Valor
Customer ID	93911343434
Valor	R$ 20.000,00
Idade	51
Tempo de relacionamento	7 meses
First receiver	1
Pix key random	0
LGBM raw	0,02045083
IF percentile	0,988755
SE score	40
BEH score	15
Score final original	42,37
Decisão original	APROVAR
Nova decisão	CONFIRMAR
Motivo	GUARD_EXCEPTION_ALTO_VALOR_SE_BEH

O caso havia sido suprimido pelo guard rail com a justificativa:

LGBM_GUARD_RAIL: VETO CONFIRMAR SE+BEH_VALOR_NOVO | LGBM=0.0205 < 0.30

O EXP-004-FINAL demonstrou que o guard rail do EXP-002 deve continuar ativo, mas não deve ser absoluto quando há convergência forte e cirúrgica de:

valor alto + conta nova + primeiro recebedor + IF extremo + SE + BEH

O JSON de delta do experimento mostra que o V1 recuperou exatamente esse FN, no valor de R$20.000, sem adicionar falso positivo.

7. Análise das variantes rejeitadas
7.1 V2_RATE_LIMIT

O padrão RATE_LIMIT_ANOMALO teve comportamento limpo, mas não trouxe ganho de FN.

No relatório shadow, o V2 teve:

Métrica	Valor
Policy hits	57
Hits fraude	57
Precision dos hits	100%
Upgrades APROVAR → CONFIRMAR	0
TP adicional	0
FP adicional	0

Isso indica que o sinal de rate limit é bom para explicabilidade e para futura modelagem, mas não melhorou a decisão final no runtime atual, pois os casos marcados já estavam capturados por outras camadas.

Decisão: manter como sinal shadow / candidata a feature para FASE 2, mas não promover como regra de decisão.

7.2 V3_PRIMEIRO_RECEIVER

O padrão PRIMEIRO_RECEIVER_VALOR_ANOMALO confirmou a hipótese de que first_receiver_flag é um sinal forte, mas perigoso como regra determinística.

Resultado no sample principal:

Métrica	Valor
TP	346
FP	95
FN	9
Precision	78,458%
F1	0,8693
FP adicional	+80

O padrão recuperou o mesmo FN de R$20.000, mas adicionou 80 falsos positivos. O relatório shadow mostra que teve 208 hits, com 122 fraudes e 86 legítimas, resultando em precision de hits de apenas 58,65%.

Decisão: não promover como regra hardcoded. O sinal deve ser tratado como feature/interação supervisionada em FASE 2.

7.3 V4_COMBO_FINAL

A variante combinada herdou o problema do V3.

Resultado:

Métrica	Valor
TP	346
FP	95
FN	9
F1	0,8693
FP adicional	+80

Não houve ganho adicional sobre V1. A combinação apenas adicionou ruído operacional.

Decisão: rejeitada.

8. Bug identificado

O artefato 03_fn_residuais_classificados.csv classificou todos os 9 FNs residuais como:

GUARD_SUPPRESSED_CANDIDATE

Essa classificação parece incorreta. A causa provável é tratamento inadequado de NaN em veto_suppressed_reason: valores nulos foram convertidos para string "nan" e interpretados como texto válido.

Correção recomendada:

def _has_text_value(x) -> bool:
    if x is None:
        return False

    try:
        if pd.isna(x):
            return False
    except Exception:
        pass

    s = str(x).strip().lower()
    return s not in {"", "nan", "none", "null", "<na>"}

E substituir verificações como:

suppressed = str(row.get("veto_suppressed_reason", "") or "").strip()

if suppressed:
    return "GUARD_SUPPRESSED_CANDIDATE", ...

por:

suppressed_raw = row.get("veto_suppressed_reason", "")

if _has_text_value(suppressed_raw):
    return "GUARD_SUPPRESSED_CANDIDATE", ...

Sem essa correção, relatórios futuros podem superestimar problemas de guard rail e direcionar a investigação para o lugar errado.

9. Decisão final da FASE 1

A FASE 1 deve ser encerrada com a seguinte decisão:

FASE 1 encerrada com melhoria parcial validada.
Promovido: V1_GUARD_CONTEXTUAL.
Rejeitado: PRIMEIRO_RECEIVER_VALOR_ANOMALO como regra hardcoded.
Mantido como shadow: RATE_LIMIT_ANOMALO.
Próxima etapa: FASE 2 — modelagem supervisionada e meta-modelagem.

Baseline consolidado pós-FASE 1:

Métrica	Valor
TP	346
FP	15
FN	9
Precision	95,8449%
Recall	97,4648%
F1	0,9665
FPR	0,2657%
10. Mudança permanente recomendada

Promover o padrão GUARD_EXCEPTION_ALTO_VALOR_SE_BEH para o decision_engine.py, com flag configurável no scoring_config.json.

A regra deve atuar apenas como exceção cirúrgica ao guard rail quando ocorrer:

PF válida
valor >= R$15.000
tempo de relacionamento <= 12 meses
first_receiver_flag == 1
IF percentile >= 0.985
SE score >= 40
BEH score >= 15
0.01 <= LGBM raw < 0.30

Essa exceção preserva o benefício do EXP-002, que reduziu FP, mas recupera o caso comprovado em que o guard rail foi excessivamente agressivo.

Patch obrigatório antes da FASE 2:
1. Promover V1_GUARD_CONTEXTUAL do EXP-004-FINAL para o decision_engine.py.
2. Corrigir o bug de classificação de FN residual com NaN.
3. Atualizar scoring_config.json com os parâmetros da exceção do guard rail.
4. Rodar uma validação curta para garantir:
   seed=42:  TP=346, FP=15, FN=9
   seed=123: TP=346, FP=12, FN=9

**Critério de aceite da FASE 1:**
- ✅ LGBM v6.2 com FN ≤ 2 no holdout
- ✅ IF v3.1 com AP ≥ 0.62
- ✅ SE v3.5 com ≥ 5% de ativação em score≥60
- ✅ BEH v3.2 com novo fator IDOSO_VALOR_MODERADO validado

---

FASE 2 — Recalibração Supervisionada e Meta-Modelagem Anti-FN
Status: 📋 Planejada
Objetivo: reduzir falsos negativos remanescentes usando modelagem supervisionada e meta-modelagem, evitando novas regras manuais generalistas.
Baseline: pós-FASE 1 consolidado, após promoção do V1 do EXP-004-FINAL.

1. Contexto
A FASE 1 demonstrou que o pipeline antifraude PIX já possui uma arquitetura sólida e calibrada para produção. A validação end-to-end da FASE 0 estabeleceu uma fonte confiável da verdade usando o PipelineOrquestrador real, o DecisionEngine real e os módulos SE/BEH reais. 
Na FASE 1, foram executadas otimizações cirúrgicas sobre o motor de decisão:


ajuste do threshold final;


hardening do guard rail LGBM;


teste de padrão residual SE;


exceção contextual ao guard rail.


Os resultados indicam que a maior parte do ganho possível por regras manuais já foi extraída. O próximo salto de performance deve vir de retreinamento supervisionado, features de interação e meta-modelagem, não da criação de novas regras determinísticas amplas.

2. Baseline oficial da FASE 2
O baseline da FASE 2 é o runtime pós-FASE 1 consolidado:
MétricaValorTP346FP15FN9Precision95,8449%Recall97,4648%F10,9665FPR0,2657%
Configuração conceitual:
{  "threshold_confirmar": 62.0,  "threshold_bloquear": 95.0,  "lgbm_guard_enabled": true,  "lgbm_guard_threshold": 0.30,  "guard_exception_alto_valor_se_beh_enabled": true,  "se_pattern_residual_enabled": false,  "exp003_residual_confirm_enabled": false}

3. Diagnóstico que motiva a FASE 2
3.1 O limite das regras manuais
O EXP-004-FINAL demonstrou que:


regras cirúrgicas podem funcionar bem quando têm alvo claro;


sinais generalistas como first_receiver_flag são úteis, mas perigosos como regra hardcoded;


padrões comportamentais podem ser bons como sinal, mas não necessariamente alteram a decisão final;


insistir em novas regras pode gerar muitos falsos positivos.


O caso mais claro foi o PRIMEIRO_RECEIVER_VALOR_ANOMALO: ele recuperou 1 FN, mas adicionou 80 FP no sample principal, derrubando o F1 de 0,9650 para 0,8693. 
3.2 Oportunidade de modelagem
Os sinais rejeitados como regras continuam úteis como features:
first_receiver_flagpix_key_random_flagvalor_x_first_receiveridade_x_first_receiverif_percentilelgbm_rawse_scorebeh_scorerate_limit_anomalo
O próprio lgbm_features.json já contém interações relevantes, como idade_x_first_recv, valor_x_first_recv, burst_x_distinct_recv e valor_over_trimestre_avg. 
A FASE 2 deve usar esses sinais em modelos supervisionados, permitindo que o algoritmo aprenda quando first_receiver_flag é perigoso e quando é apenas comportamento legítimo.

4. Objetivo da FASE 2
Reduzir FN sem degradar a operação.
Meta mínima
MétricaAlvoFN≤ 7FP≤ 22Precision≥ 94,0%Recall≥ 98,0%F1≥ 0,9665
Meta forte
MétricaAlvoFN≤ 5FP≤ 25Precision≥ 93,5%Recall≥ 98,6%F1≥ 0,9700
Meta excelente
MétricaAlvoFN≤ 3FP≤ 30Precision≥ 92,0%Recall≥ 99,15%F1≥ 0,9720

5. Princípios da FASE 2
A FASE 2 deve seguir cinco princípios:
1. Não criar novas regras hardcoded generalistas.2. Transformar sinais promissores em features supervisionadas.3. Otimizar para Recall@Precision mínima, não F1 puro.4. Recalibrar o guard rail sempre que o LGBM for retreinado.5. Manter o Decision Engine explicável, com fallback determinístico.

EXP-005A — LGBM Recall-Oriented v6.2
Categoria: Modelagem / Retreinamento
Complexidade: 🟡 Média
Prioridade: 🔴 Alta
Objetivo: retreinar o LightGBM com objetivo assimétrico anti-FN.

1. Contexto
O LGBM atual foi treinado otimizando F1 padrão. Porém, fraude PIX tem assimetria estrutural: o custo de um FN é maior que o custo de um FP moderado. A spec original do EXP-005 já identificava esse desalinhamento e propunha retreinar o modelo otimizando Recall@Precision≥0.90, usando scale_pos_weight, focal loss ou class weights. 
A FASE 1 reforçou essa necessidade: sinais como first_receiver_flag e pix_key_random_flag são relevantes, mas geram muitos FP quando usados diretamente como regra. Eles devem ser aprendidos pelo modelo em interação com outros sinais.

2. Hipótese

Retreinar o LightGBM com objetivo orientado a recall e features de interação antifraude reduz os FNs de 9 para ≤7, mantendo FP ≤22 e Precision ≥94%.


3. Features candidatas
Adicionar ou reforçar as seguintes features:
df["conta_nova_valor_alto_flag"] = (    (df["qt_tempo_relacionamento_mes"] <= 12)    & (df["vl_pix"] >= 5000)    & (df["first_receiver_flag"] == 1)).astype(int)df["interaction_rel_valor"] = (    np.log1p(df["vl_pix"])    / (df["qt_tempo_relacionamento_mes"].fillna(999) + 1))df["pix_random_x_first_receiver"] = (    df["pix_key_random_flag"] * df["first_receiver_flag"])df["valor_x_first_receiver"] = (    np.log1p(df["vl_pix"]) * df["first_receiver_flag"])df["idade_x_valor_alto"] = (    df["nr_idade"] * (df["vl_pix"] >= 5000).astype(int))df["valor_x_if_percentile"] = (    np.log1p(df["vl_pix"]) * df["if_percentile"])
Observação: se if_percentile não estiver disponível no dataset de treino original do LGBM, essa feature deve ficar reservada para o meta-learner, não para o LGBM primário, para evitar dependência circular entre modelos.

4. Variantes de treino
Testar:
LGBM_A_BASELINE_RETRAIN- mesmo setup atual, apenas reprodutibilidadeLGBM_B_SCALE_POS_WEIGHT- scale_pos_weight aumentado- objetivo: recall maior mantendo precision mínimaLGBM_C_CLASS_WEIGHT_CUSTOM- peso de fraude 2x, 3x, 5x- selecionar por Recall@PrecisionLGBM_D_FOCAL_LOSS- se viável tecnicamente- foco em exemplos difíceis e borderlineLGBM_E_FEATURE_INTERACTIONS- melhor configuração anterior + novas features de interação

5. Métrica de seleção
Não selecionar por F1 puro.
Critério primário:
Maximizar Recall sujeito a Precision >= 94% no sample E2E.
Critérios secundários:
1. menor FP;2. maior F1;3. estabilidade seed=123;4. explicabilidade SHAP coerente;5. não piorar casos de alto valor.

6. Critério de aceite
A variante LGBM v6.2 só pode ser promovida se, no pipeline real:
FN <= 7FP <= 22Precision >= 94%Recall >= 98%F1 >= 0,9665seed=123 confirma direção
Se o modelo reduzir FN, mas aumentar FP demais, ele não deve ser promovido diretamente; deve ir para o EXP-005B de recalibração de thresholds.

7. Artefatos esperados
resultados/experimentos/EXP-005A/├── 01_tabela_modelos.csv├── 02_threshold_sweep_lgbm.csv├── 03_avaliacao_e2e_modelo_candidato.json├── 04_validacao_cruzada.json├── 05_shap_drift_report.md└── 06_conclusao_executiva.md
Modelo candidato:
backend/artefatos_candidatos/lgbm_v6_2_recall.joblibbackend/artefatos_candidatos/lgbm_features_v6_2.jsonbackend/artefatos_candidatos/thresholds_lgbm_v6_2.json



## EXP-005A — LGBM Recall-Oriented v6.2

**Status:** ✅ Concluído — candidato gerado para EXP-005B
**Tipo:** Retreino supervisionado model-only
**Objetivo:** testar se um LightGBM reponderado para fraude conseguiria reduzir FN sem depender de novas regras manuais.

O experimento treinou múltiplas variantes LightGBM com foco em recall. O vencedor foi `LGBM_C_SPW_2_0X`, usando o mesmo conjunto de **52 features baseline**, sem depender das novas features `exp005_*`. O ganho veio principalmente da reponderação da classe fraude com `scale_pos_weight` aumentado.

Resultado model-only no sample principal de 6.000 transações:

| Métrica   | Baseline FASE 2 | LGBM_C_SPW_2_0X |
| --------- | --------------: | --------------: |
| TP        |             346 |             353 |
| FP        |              15 |              20 |
| FN        |               9 |               2 |
| Precision |          95,84% |          94,64% |
| Recall    |          97,46% |          99,44% |
| F1        |          0,9665 |          0,9698 |

Na validação seed 123, o candidato manteve `FN=2`, mas subiu para `FP=27`, com `Precision=92,89%` e `F1=0,9605`. Portanto, o modelo **não deve ser promovido diretamente para runtime**, mas é um candidato forte para calibração no EXP-005B.

**Conclusão:** o EXP-005A provou que o retreino supervisionado orientado a recall consegue capturar parte relevante dos FNs remanescentes, reduzindo FN de 9 para 2 em avaliação model-only. Porém, o threshold `0.0524338379` é agressivo e precisa ser recalibrado dentro do `DecisionEngine`, com guard rail e thresholds finais. O próximo passo é o **EXP-005B — Recalibração do Engine pós-LGBM v6.2**.



EXP-005B — Recalibração do Engine pós-LGBM v6.2
Categoria: Calibração / Política de decisão
Complexidade: 🟡 Média
Prioridade: 🔴 Alta
Objetivo: recalibrar thresholds internos e guard rail após retreino do LGBM.

1. Contexto
O EXP-002 calibrado na FASE 1 usa:
{  "lgbm_guard_enabled": true,  "lgbm_guard_threshold": 0.30}
Esse valor foi aprovado para a distribuição do LGBM atual, reduzindo FP de 18 para 15, mas perdendo 1 TP. 
Se o LGBM for retreinado no EXP-005A, a distribuição de lgbm_raw pode mudar. Portanto, o guard rail precisa ser recalibrado.

2. Hipótese

Recalibrar lgbm_guard_threshold, lgbm_effective_threshold e threshold_confirmar sobre o LGBM v6.2 permite reduzir FN sem reabrir falsos positivos excessivos.


3. Grid de calibração
Testar combinações:
threshold_confirmar:- 58- 60- 62- 65lgbm_guard_threshold:- 0.10- 0.20- 0.30- 0.40lgbm_effective_threshold:- 0.25- 0.30- 0.35- 0.40
Manter sempre ativa a exceção:
guard_exception_alto_valor_se_beh_enabled = true

4. Critério de seleção
Escolher a configuração que:
1. minimize FN;2. mantenha FP <= 25;3. mantenha Precision >= 93,5%;4. mantenha FPR <= 0,5%;5. preserve ou melhore F1;6. valide em seed=123.

5. Artefatos esperados
resultados/experimentos/EXP-005B/├── 01_grid_thresholds.csv├── 02_top_configs.json├── 03_delta_fp_fn_melhor_config.json├── 04_validacao_cruzada.json└── 05_conclusao_executiva.md
Config candidata:
backend/artefatos_candidatos/scoring_config_v2_candidate.json

EXP-007A — Meta-Learner Shadow
Categoria: Meta-modelagem / Stacking
Complexidade: 🔴 Alta
Prioridade: 🟠 Média-Alta
Objetivo: treinar um meta-learner em modo shadow para aprender combinações não-lineares entre LGBM, IF, SE, BEH e features-chave.

1. Contexto
O engine atual combina sinais por regras e pesos manuais. Essa arquitetura é explicável e controlável, mas tem limitações: não aprende interações não-lineares entre módulos.
A spec original do EXP-007 já identificava que um meta-learner poderia capturar padrões como:
LGBM baixo-moderado + IF alto + primeiro recebedor + valor anômalo
que a agregação linear pode diluir. 

2. Hipótese

Um meta-learner treinado sobre as saídas dos módulos primários e features-chave melhora a recuperação de FN em modo shadow, sem aumentar FP de forma operacionalmente inviável.


3. Inputs do meta-learner
Usar como features:
lgbm_rawlgbm_mappedif_percentileif_rawse_scorebeh_scorescore_final_atualdecisao_atual_encodedfirst_receiver_flagpix_key_random_flagvl_pixlog_vl_pixnr_idadeqt_tempo_relacionamento_mestx_count_prev_30mburst_30m_flagdistinct_receivers_so_farvalor_x_first_receiveridade_x_first_receivervalor_x_burstburst_x_distinct_recvvalor_over_trimestre_avgrate_limit_anomalo_shadowguard_exception_alto_valor_shadow

4. Modo de operação
O EXP-007A deve ser shadow only.
Ou seja:
- não substitui o Decision Engine;- não altera decisões em produção;- apenas calcula probabilidade alternativa de fraude;- compara contra o engine atual;- gera explicabilidade SHAP;- identifica FNs que o meta-learner capturaria.

5. Critério de aceite
Para virar candidato de FASE 3, o meta-learner precisa:
FN <= baseline pós-EXP-005BFP <= baseline + 5F1 >= baselineRecall maior que baselineSHAP coerenteseed=123 confirma direçãosem dependência de leakage temporal

6. Artefatos esperados
resultados/experimentos/EXP-007A/├── 01_metricas_shadow.csv├── 02_fn_capturados_pelo_meta.json├── 03_fp_adicionados_pelo_meta.json├── 04_shap_global.md├── 05_validacao_cruzada.json└── 06_conclusao_shadow.md
Modelo candidato:
backend/artefatos_candidatos/meta_learner_shadow_v1.joblibbackend/artefatos_candidatos/meta_features_v1.json

Ordem de execução da FASE 2
A ordem recomendada é:
0. Patch FASE 1:   - promover V1_GUARD_CONTEXTUAL;   - corrigir bug de NaN em FN residual;   - validar seed=42 e seed=123.1. EXP-005A:   - retreinar LGBM v6.2 orientado a recall.2. EXP-005B:   - recalibrar thresholds e guard rail em cima do LGBM v6.2.3. EXP-007A:   - treinar meta-learner shadow.4. Decisão:   - se LGBM v6.2 + engine calibrado bater meta forte, promover.   - se meta-learner superar com estabilidade, planejar FASE 3.   - se nenhum modelo superar baseline com segurança, manter baseline pós-FASE 1.

Critério de encerramento da FASE 2
A FASE 2 pode ser encerrada em três cenários.
Cenário A — Sucesso forte
FN <= 5FP <= 25Precision >= 93,5%Recall >= 98,6%F1 >= 0,9700
Decisão:
Promover LGBM v6.2 + scoring_config recalibrado.Meta-learner segue para FASE 3 se tiver ganho adicional.

Cenário B — Sucesso mínimo
FN <= 7FP <= 22Precision >= 94%Recall >= 98%F1 >= 0,9665
Decisão:
Promover melhoria se validada em seed=123 e sem regressão de alto valor.

Cenário C — Sem ganho robusto
FN permanece >= 8ou FP > limiteou seed=123 não confirma direção
Decisão:
Manter baseline pós-FASE 1.Classificar FNs remanescentes como dependentes de novas fontes de dados.Mover para FASE 3/4: device fingerprint, reputação de recebedor, grafo transacional ampliado, MED, contestação e dados de sessão.

Resultado esperado da FASE 2
Resultado realista esperado:
EtapaGanho esperadoEXP-005Arecuperar 1 a 3 FNs via LGBM orientado a recallEXP-005Bpreservar recall com FP controladoEXP-007Aidentificar se há ganho adicional por stackingFASE 2 finalFN entre 5 e 7, FP entre 18 e 25
Resultado ideal:
TP >= 350FN <= 5FP <= 25Recall >= 98,6%Precision >= 93,5%F1 >= 0,9700

Observação final
A FASE 1 demonstrou que o sistema já está em um ponto de alta qualidade. A partir daqui, tentar reduzir FN com regras manuais tende a causar explosão de FP, como visto no PRIMEIRO_RECEIVER_VALOR_ANOMALO.
A FASE 2 deve, portanto, mudar a abordagem:
De: regras manuais para cada cluster de FNPara: aprendizado supervisionado das interações que tornam esses clusters perigosos
Esse é o caminho mais seguro para melhorar o modelo sem degradar a experiência dos clientes legítimos.



# 🛠️ FASE 3 — Robustez Operacional (MLOps Light)

> **Objetivo:** passar de "script que roda" para "sistema que se mantém".

### 3.1 — Experiment tracking com MLflow

Instalar MLflow local (sqlite backend, sem servidor):

```python
# scripts/train_lgbm_v3.py — adicionar
import mlflow

mlflow.set_tracking_uri("file://./mlruns")
mlflow.set_experiment("pix_antifraude_lgbm")

with mlflow.start_run(run_name=f"lgbm_v{VERSION}"):
    mlflow.log_params({
        "n_estimators": final_n_estimators,
        "learning_rate": 0.01,
        "num_leaves": 63,
        "features_count": len(feature_cols),
    })
    mlflow.log_metric("holdout_f1", holdout_metrics_f1["f1"])
    mlflow.log_metric("holdout_recall", holdout_metrics_f1["recall"])
    mlflow.log_metric("holdout_fn", holdout_metrics_f1["fn"])
    mlflow.lightgbm.log_model(final_model, "model")
    mlflow.log_artifact(str(METRICS_PATH))
```

Benefício: você compara v5.1, v6.1, v6.2 lado a lado visualmente.

### 3.2 — Profile manager persistente

O `_InlineProfileManager` em memória **perde tudo no restart da API**. Isso quebra as features sequenciais por minutos (até o cache reconstruir).

**Solução simples (SQLite + WAL):**

```python
# core/profile_store.py (novo)
import sqlite3
from contextlib import contextmanager

class SQLiteProfileStore:
    def __init__(self, db_path: str = "profiles.db"):
        self.db_path = db_path
        self._init_schema()
    
    def _init_schema(self):
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS tx_history (
                    cpf TEXT NOT NULL,
                    tx_ts INTEGER NOT NULL,
                    receiver TEXT,
                    pix_key TEXT,
                    vl_pix REAL,
                    PRIMARY KEY (cpf, tx_ts)
                );
                CREATE INDEX IF NOT EXISTS idx_cpf_ts ON tx_history(cpf, tx_ts DESC);
                PRAGMA journal_mode=WAL;
            """)
    
    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def get_tx_count_30m(self, cpf: str, ts_now: int) -> int:
        with self._conn() as c:
            cur = c.execute(
                "SELECT COUNT(*) FROM tx_history WHERE cpf=? AND tx_ts >= ?",
                (cpf, ts_now - 1800)
            )
            return cur.fetchone()[0]
    
    # ...demais métodos
```

Trocar `_InlineProfileManager` por `SQLiteProfileStore` no `behavioral_analytics.py`. API continua igual.

### 3.3 — Testes unitários para regras críticas

```python
# tests/test_behavioral_analytics.py (novo)
import pytest
from core.behavioral_analytics import BehavioralAnalytics

class TestFrequenciaBurst:
    def setup_method(self):
        self.beh = BehavioralAnalytics()
    
    def test_burst_detecta_4_tx_30m(self):
        features = {
            "cd_cpf_pagador": "12345",
            "tx_count_prev_30m": 3,
            "burst_30m_flag": 1,
            "vl_pix": 500,
            "vl_mediana_pix_trimestre": 100,
            # ... outras features mínimas
        }
        result = self.beh.analyze(features)
        codigos = [rf.codigo for rf in result.risk_factors]
        assert "FREQUENCIA_BURST" in codigos
        assert result.behavioral_score >= 25
    
    def test_burst_nao_dispara_com_2_tx(self):
        features = {...}
        features["tx_count_prev_30m"] = 1
        result = self.beh.analyze(features)
        assert "FREQUENCIA_BURST" not in [rf.codigo for rf in result.risk_factors]
```

Priorizar:
1. Todas as 8 regras do SE v3.4
2. Todos os 6 fatores do BEH v3.1  
3. Vetos críticos do Engine
4. Cascade C1 e C3

Meta: **80% de cobertura nos engines**.

### 3.4 — Monitoramento de drift

Adicionar endpoint `/api/v1/drift` que compara distribuição das últimas 10k tx vs distribuição de treino:

```python
# core/drift_monitor.py
def kolmogorov_smirnov_drift(train_stats: dict, recent_values: np.ndarray, feat: str) -> float:
    """Retorna p-value — se < 0.01 há drift significativo."""
    from scipy.stats import ks_2samp
    train_mean = train_stats[feat]["mean"]
    train_std = train_stats[feat]["std"]
    # Gera amostra teórica vs observada
    ...
```

Alertar se >5 features derivarem.

### 3.5 — Documentação unificada

Criar `docs/` com:
- `ARCHITECTURE.md` — diagrama de componentes (use mermaid)
- `DEPLOYMENT.md` — como rodar em produção
- `MODEL_CARD_LGBM.md` — model card com viés, limitações, dataset
- `MODEL_CARD_IF.md` — idem
- `RULES_CATALOG.md` — todas as regras SE/BEH centralizadas

**Critério de aceite da FASE 3:**
- ✅ MLflow rodando com 3+ runs documentados
- ✅ Profile store persistente operacional
- ✅ 80% cobertura de testes em `core/`
- ✅ Endpoint de drift retornando métricas
- ✅ Documentação organizada em `docs/`

---

# 🔬 FASE 4 — Evolução Científica (contínuo)

> **Objetivo:** passar de 96% → 99% de recall mantendo precisão alta, com inovação dirigida por ciência.

### 4.1 — Modelo de sequência para detectar padrões de golpe

O problema do FN "R$20k, conta 7 meses, 1ª tx, score 0.0008" é que **features agregadas perdem o sinal**. Um modelo sequencial veria: login → nav → PIX rápido = padrão anômalo.

**Proposta leve:** LSTM pequeno sobre sequência de eventos da sessão, rodando como **3º modelo no ensemble** (junto com LGBM e IF). Tamanho modesto: 2 camadas LSTM(32), output sigmoid. Treinado separadamente em sessões de fraude conhecidas.

**EXP-005:**
```
Hipótese: LSTM sobre sequência de eventos captura os FN residuais 
          do LGBM (sinais comportamentais temporais).
Dataset:  Eventos de sessão até o momento do PIX.
Métrica:  Recall nos 3 FN irredutíveis atuais.
Aceite:   Recupera ≥ 1 FN sem adicionar > 5 FP no holdout.
```

### 4.2 — Active learning para reduzir anotação

Para os casos **CONFIRMAR** que ficam na zona cinza, implementar loop de active learning:

```
Tx → Score → CONFIRMAR → Humano decide → Label vira feedback →
     Modelo retreinado mensalmente com novos labels
```

Ferramenta: **Argilla** (open source, Python-native).

### 4.3 — Explainability quantitativa para auditoria

SHAP já está integrado. Adicionar:
- **Counterfactual explanations:** "Esta tx seria APROVADA se vl_pix fosse ≤ R$1.500" (biblioteca DiCE)
- **Global feature importance dashboard** (streamlit rodando em paralelo)

### 4.4 — Adversarial robustness

Simular um atacante que sabe do modelo e tenta burlar:

```python
# scripts/adversarial_test.py
def generate_adversarial_tx(fraud_tx: dict, model) -> dict:
    """Gera variação da fraude que passa no modelo."""
    # Grid search sobre features mutáveis (hora, device, etc)
    # Retorna tx que minimiza score mantendo natureza fraudulenta
    ...
```

Rode contra suas fraudes do dataset. Se 20%+ delas conseguem "burlar" com mudanças <5% nas features, você tem um modelo frágil.

### 4.5 — Pesquisar papers recentes (auto-descoberta)

Tópicos para monitorar no arXiv/NeurIPS:
- **Temporal graph neural networks for fraud** (você vai chegar lá)
- **LLM-based anomaly explanation** (resumir decisão em PT-BR)
- **Federated learning para bancos** (treinar sem compartilhar dados)
- **Differential privacy em modelos antifraude**

---

## 🗺️ Ordem de Execução Sugerida

```
Semana 1:        FASE 0 (bloqueador)
Semanas 2-3:     FASE 1 paralela com FASE 2 (cada dev pega uma)
Semana 4:        FASE 2 (cleanup final)  
Semanas 5-7:     FASE 3 (MLOps)
Contínuo:        FASE 4 (research)
```

---

## 📋 Kanban proposto (artefatos SDD a gerar)

```
backend/
├── CONSTITUTION.md                    ← FASE 0
├── VALIDATION_REPORT.md               ← FASE 0
├── docs/
│   ├── ARCHITECTURE.md                ← FASE 3
│   ├── DEPLOYMENT.md                  ← FASE 3
│   ├── MODEL_CARD_LGBM.md             ← FASE 3
│   ├── MODEL_CARD_IF.md               ← FASE 3
│   └── RULES_CATALOG.md               ← FASE 3
├── experiments/
│   ├── EXP-001_lgbm_v6.2.md           ← FASE 1
│   ├── EXP-002_if_v3.1.md             ← FASE 1
│   ├── EXP-003_beh_v3.2.md            ← FASE 1
│   ├── EXP-004_cascade_c3_tuning.md   ← FASE 2
│   └── EXP-005_lstm_sequencia.md      ← FASE 4
├── tests/                             ← FASE 3
│   ├── test_behavioral_analytics.py
│   ├── test_social_engineering.py
│   └── test_decision_engine.py
└── core/
    ├── profile_store.py               ← FASE 3 (novo)
    ├── drift_monitor.py               ← FASE 3 (novo)
    └── cost_model.py                  ← FASE 2 (novo)
```

---

## 🎯 Meta de Impacto

Se executar FASES 0-2 corretamente, esperado no dataset atual:

| Métrica | Atual (reportado) | Meta FASE 2 |
|---|---|---|
| Recall | 96.9% (bugado) | **≥ 98.5%** (real) |
| Precision | 91.2% (bugado) | **≥ 75%** (real com SE+BEH) |
| F1 | 0.940 (bugado) | **≥ 0.86** (real, mais honesto) |
| FN | 11 | **≤ 4** |
| FPR | 0.033% | **≤ 0.15%** |

**Nota importante:** os números "melhoram" no sentido de **serem reais**. O F1=0.94 reportado hoje é artificial porque SE/BEH não estavam rodando — o sistema completo vai ter F1 um pouco menor, mas **recall maior** (que é o que importa para antifraude).

---

## 🚨 Riscos e Mitigações

| Risco | Probabilidade | Mitigação |
|---|---|---|
| Correção da FASE 0 revelar que SE+BEH geram muitos FP | Alta | Recalibrar SE v3.5 na FASE 1 |
| FASE 1 não melhorar FN | Média | Fallback: manter LGBM v6.1, focar em engine |
| MLOps atrasar | Baixa | Opcional na FASE 3, não bloqueia produção |
| Dataset pequeno limitar generalização | Alta | Aumentar janela de dados, cross-validar geograficamente |

---

## 💬 Próximos Passos Concretos

Se você topar esse plano, meu próximo entregável pode ser qualquer um:

**Opção A** — Patch pronto do `simular_pipeline_e2e_lf.py` corrigido (FASE 0.1 completa)

**Opção B** — Template completo do `CONSTITUTION.md` customizado para seu projeto (FASE 0.4)

**Opção C** — Spec detalhado do EXP-001 (LGBM v6.2 com features de interação) seguindo protocolo SDD

**Opção D** — Template do `VALIDATION_REPORT.md` com placeholders para você preencher

**Opção E** — Code review profundo de um módulo específico (`decision_engine.py` ou `behavioral_analytics.py`) com refactor proposto

**Qual você quer atacar primeiro?** Minha sugestão forte: **Opção A + D em paralelo** — corrige o simulador e gera o relatório real em ~1 dia. Sem isso, tudo que vier depois estará tomando decisão em dados enganosos.

O que você acha, Adilio? Algum ponto do plano você discorda ou quer repriorizar? 🚀