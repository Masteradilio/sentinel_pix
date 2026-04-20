# EXP-001 — Ajuste do Threshold Final (77 → 62)

> Experimento da **FASE 1 — Otimização Cirúrgica** do projeto BRB Antifraude PIX.
> Especificação completa em [`docs/experiments/EXP-001.md`](../../docs/experiments/EXP-001.md).

---

## 🎯 Objetivo

Validar se reduzir o `score_final_threshold_confirmar` de **77 → 62** aumenta o recall do pipeline sem degradar F1 nem causar explosão de falsos positivos.

**Hipótese central:**
> Threshold 62 captura ~14 fraudes adicionais (Recall 93.52% → 97.46%) com custo aceitável de ~9 FP novos, resultando em F1 superior (0.9554 → 0.9638).

---

## 📂 Estrutura

```
exp_001_threshold_final/
├── __init__.py
├── README.md                    ← você está aqui
├── config_variantes.json        ← thresholds testados (baseline + 3 variantes)
└── run_exp_001.py               ← script principal
```

---

## 🚀 Como rodar

### Pré-requisitos

- Estar na raiz do projeto (`rebuild_pix/`)
- Python 3.12+ com dependências do projeto instaladas
- Artefatos do pipeline disponíveis em `backend/artefatos/`
- Dataset em `dados/base_treino_final.csv`
- Script `backend/scripts/simular_pipeline_e2e_v2.py` funcional

### Execução padrão (recomendada)

```powershell
python experimentos\exp_001_threshold_final\run_exp_001.py --workers 4
```

**Tempo estimado:** 6-10 minutos (inclui validação cruzada em seed=123).

### Execução rápida (sem validação cruzada)

Útil para debug ou iteração rápida:

```powershell
python experimentos\exp_001_threshold_final\run_exp_001.py --workers 4 --skip-validation
```

**Tempo estimado:** 3-5 minutos.

### Execução sequencial (sem paralelismo)

Para máquinas com pouca RAM (cada worker consome ~200MB):

```powershell
python experimentos\exp_001_threshold_final\run_exp_001.py --workers 1
```

**Tempo estimado:** ~20-30 minutos.

### Customizar tamanho do sample

```powershell
python experimentos\exp_001_threshold_final\run_exp_001.py --sample 10000 --workers 4
```

---

## ⚙️ Parâmetros CLI

| Argumento | Tipo | Default | Descrição |
|---|---|---|---|
| `--sample` | `int` | `6000` (do config) | Override do tamanho do sample estratificado |
| `--workers` | `int` | `1` | Workers paralelos (1=sequencial, 2-4 recomendado) |
| `--skip-validation` | `flag` | `False` | Pula validação cruzada em seed=123 |

---

## 🧠 Estratégia de execução

Este experimento **NÃO reprocessa** o sample a cada variante. Isso seria desperdício computacional, porque o threshold de decisão é aplicado **depois** do `score_final` ser calculado pelo pipeline.

### Fluxo otimizado

```
1. Carrega dataset  →  2. Gera sample estratificado (seed=42)
                                    ↓
3. Processa UMA vez via PipelineOrquestrador  →  predictions_df
                                    ↓
4. Aplica 4 thresholds post-hoc (baseline=77, V1=62, V2=65, V3=70)
                                    ↓
5. Sweep fino 40-90 (step=1)  →  identifica ótimo absoluto
                                    ↓
6. Análise qualitativa FP/FN  →  perfil demográfico, alertas de bias
                                    ↓
7. Validação cruzada (seed=123)  →  guarda contra overfit
                                    ↓
8. Gera 5 artefatos + relatório executivo
```

**Ganho de performance:** ~4x mais rápido que rodar 4 execuções separadas.

### Preservação de vetos

Vetos cirúrgicos do engine (`VETO BLOQUEAR`, `VETO CONFIRMAR`) são **preservados** em todas as variantes. O threshold varia apenas para transações que passaram pela rota score-based. Isso garante paridade com o comportamento de produção.

---

## 📦 Artefatos gerados

Todos em `resultados/experimentos/EXP-001/` (caminho fixo, sem timestamp).

| # | Arquivo | Formato | Pergunta que responde |
|---|---|---|---|
| 1 | `01_tabela_comparativa.csv` | CSV | Qual variante teve melhor F1/Recall/Precision? |
| 2 | `02_threshold_sweep_fino.csv` | CSV | Existe threshold ainda melhor entre os testados? |
| 3 | `03_analise_fp_fn.json` | JSON | Quem são os novos FP? Quais FN foram recuperados? |
| 4 | `04_validacao_cruzada.json` | JSON | O resultado generaliza em sample independente? |
| 5 | `05_conclusao_executiva.md` | Markdown | **Aprovar? Qual variante? Próximos passos?** ⭐ |

### Ordem recomendada de leitura

1. **`05_conclusao_executiva.md`** — comece aqui. Veredicto + recomendação.
2. `01_tabela_comparativa.csv` — confirma os números brutos.
3. `03_analise_fp_fn.json` — entende perfis afetados (auditoria de bias).
4. `02_threshold_sweep_fino.csv` — explora alternativas próximas ao vencedor.
5. `04_validacao_cruzada.json` — audita generalização estatística.

---

## ✅ Critérios de aceitação

Definidos em [`config_variantes.json`](./config_variantes.json):

| Critério | Meta |
|---|---|
| Delta F1 ≥ | +0.005 |
| Recall mínimo | 95% |
| Precision mínima | 90% |
| FPR máximo | 0.5% |

A variante vencedora é aprovada **se e somente se** todos os 4 critérios forem atendidos **E** a validação cruzada confirmar a direção do ganho.

---

## 🚦 Possíveis resultados

### ✅ Cenário "Aprovado + Validado"

Todos critérios atendidos E F1 na seed=123 > F1 baseline.

**Ação:**
1. Atualizar `backend/artefatos/scoring_config.json` com threshold novo.
2. Incrementar `engine_version`: 3.0.5 → 3.0.6.
3. Abrir PR com link para `05_conclusao_executiva.md`.
4. Monitorar métricas por 48h pós-deploy.
5. Seguir para próximo experimento (EXP-004 — Rate Limiting).

### ⚠️ Cenário "Aprovado com Ressalva"

Critérios atendidos mas validação cruzada não confirmou direção.

**Ação:** rodar o experimento em sample maior (12-20k) antes de deploy.

### ❌ Cenário "Reprovado"

Algum critério primário falhou.

**Ação:** investigar causa raiz. Considerar pular direto para EXP-004 (novos padrões) que atacam FN diferentes dos capturáveis por threshold.

---

## 🔄 Rollback

Se após deploy as métricas em produção degradarem:

### Triggers automáticos de rollback

- Recall < 85% em janela de 24h
- FPR > 2% em janela de 24h
- Reclamações de clientes aumentam >50% vs baseline

### Procedimento

1. Reverter `backend/artefatos/scoring_config.json` para threshold=77
2. Abrir post-mortem documentado em `docs/retrospectives/`
3. Replanejar com aprendizados

---

## 🐛 Troubleshooting

### Erro: `FileNotFoundError: base_treino_final.csv`

**Causa:** script sendo rodado de diretório errado.
**Solução:** rode sempre da raiz `rebuild_pix/`.

### Erro: `ImportError: cannot import name 'PipelineOrquestrador'`

**Causa:** `sys.path` não configurado corretamente ou artefatos ausentes.
**Solução:** verifique se `backend/core/pipeline_orquestrador.py` existe e se `backend/artefatos/` contém os modelos LGBM/IF.

### Workers travando / OOM

**Causa:** cada worker consome ~200MB RAM.
**Solução:** reduzir `--workers`. Em máquina com 8GB RAM, use no máximo `--workers 2`.

### Tempo muito lento mesmo com workers=4

**Causa:** I/O do dataset ou SHAP ativo.
**Solução:** o script já desabilita SHAP (`shap_enabled=False`). Se ainda lento, investigue gargalo com `py-spy`.

---

## 📚 Referências

- **Spec completa:** [`docs/experiments/EXP-001.md`](../../docs/experiments/EXP-001.md)
- **Motivação:** `docs/VALIDATION_REPORT.md` §7.1 (threshold sweep da FASE 0)
- **Guardrails:** `docs/CONSTITUTION.md` §2.3 (metas operacionais)
- **Script base:** `backend/scripts/simular_pipeline_e2e_v2.py` (v2 da FASE 0)

---

## 📜 Histórico

| Data | Versão | Mudança |
|---|---|---|
| 2026-04-17 | 1.0.0 | Criação inicial do experimento |

---

*Parte da FASE 1 — Otimização Cirúrgica do projeto BRB Antifraude PIX*
