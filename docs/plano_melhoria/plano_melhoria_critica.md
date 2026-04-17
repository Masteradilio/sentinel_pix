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

**Critério de aceite da FASE 1:**
- ✅ LGBM v6.2 com FN ≤ 2 no holdout
- ✅ IF v3.1 com AP ≥ 0.62
- ✅ SE v3.5 com ≥ 5% de ativação em score≥60
- ✅ BEH v3.2 com novo fator IDOSO_VALOR_MODERADO validado

---

# ⚙️ FASE 2 — Refinar Engine e Regras

> **Objetivo:** o engine v3.0.5 está quase lá. Faltam ajustes cirúrgicos e eliminação de dead code.

### 2.1 — Threshold tuning baseado em custo de negócio

Você não tem função de custo formalizada. Proponho:

```python
# Em backend/config/cost_model.py (novo)
COST_FN = 2500.0   # R$ médio perdido por fraude não detectada
COST_FP_CONFIRMAR = 5.0    # custo de incomodar cliente com 2FA
COST_FP_BLOQUEAR = 50.0    # custo de bloquear + atender + reabrir

def pipeline_cost(tp, fp_confirmar, fp_bloquear, fn) -> float:
    return (
        fn * COST_FN
        + fp_confirmar * COST_FP_CONFIRMAR  
        + fp_bloquear * COST_FP_BLOQUEAR
    )

def optimal_thresholds(scores, y_true, grid_step=1.0) -> dict:
    """Busca (th_confirmar, th_bloquear) que minimizam custo."""
    best = {"cost": float("inf")}
    for th_c in np.arange(50, 91, grid_step):
        for th_b in np.arange(th_c + 5, 100, grid_step):
            tp_b = np.sum((scores >= th_b) & (y_true == 1))
            tp_c = np.sum((scores >= th_c) & (scores < th_b) & (y_true == 1))
            fp_b = np.sum((scores >= th_b) & (y_true == 0))
            fp_c = np.sum((scores >= th_c) & (scores < th_b) & (y_true == 0))
            fn = np.sum((scores < th_c) & (y_true == 1))
            cost = pipeline_cost(tp_b + tp_c, fp_c, fp_b, fn)
            if cost < best["cost"]:
                best = {"cost": cost, "th_c": th_c, "th_b": th_b, ...}
    return best
```

Rode isso sobre o holdout pós-correção FASE 0. Vai provavelmente dizer que `th_confirmar=72` é melhor que 77, porque cada FN custa mais do que 10 FP em CONFIRMAR.

### 2.2 — Cascade C3: validar se LGBM guard 0.35 ainda é ótimo

No engine, `CASCADE_C3_LGBM_MIN = 0.35`. Esse número veio da calibração v3.0.5 **sem SE/BEH corrigidos**. Com SE e BEH funcionando, é possível que 0.30 ou 0.40 seja melhor.

**EXP-004:** grid search {0.25, 0.30, 0.35, 0.40, 0.45} no dataset full.

### 2.3 — Fast-Approve: revisar agressividade

`FA_LGBM_MAX = 0.25` + SE=0 + BEH=0. Com SE/BEH funcionais, a condição `SE=0 AND BEH=0` vai ser **muito mais rara** (pois SE v3.5 dispara mais). Isso significa que Fast-Approve vai cobrir **menos transações**, e o custo operacional sobe.

**Compromisso proposto:**

```python
# v3.0.6 — Fast-Approve com lógica OR-graduada
fast_approve_tier_1 = (
    lgbm_raw < 0.15 and se_score == 0 and beh_score == 0
)  # ~90% das tx — aprovação instantânea

fast_approve_tier_2 = (
    lgbm_raw < 0.25 
    and se_score < 20 
    and beh_score < 15
    and not fast_approve_tier_1
)  # ~8% das tx — aprovação com log para auditoria posterior
```

### 2.4 — Remover vetos mortos

Auditando `_aplicar_veto()`, o veto **"IF extremo + 8 agravantes"** parece nunca disparar no dataset atual. Verificar com log e, se confirmado, **remover**.

### 2.5 — Adicionar regra de proteção para perfil vulnerável (sem ML)

Regra determinística acima de qualquer modelo:

```python
# Em decision_engine.py — antes dos vetos ML
if (
    features.get("perfil_vulneravel_se_flag") == 1
    and features.get("first_receiver_flag") == 1
    and features.get("vl_pix", 0) >= 1000
    and features.get("pix_key_random_flag") == 1
):
    # Viúvo, idoso, sem dependentes, primeira tx, chave aleatória
    # Independente de ML, pede confirmação
    return score_ajustado, "VETO CONFIRMAR: perfil vulnerável alto risco"
```

Isso é **anti-golpe engineering social** puro, não depende de threshold e é auditável pelo jurídico.

**Critério de aceite da FASE 2:**
- ✅ Thresholds otimizados por custo validados em holdout
- ✅ Cascade C3 confirmado ou recalibrado
- ✅ Fast-Approve em 2 tiers
- ✅ Vetos mortos removidos
- ✅ Regra de perfil vulnerável implementada

---

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