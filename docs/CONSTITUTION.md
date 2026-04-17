# CONSTITUTION — BRB Rebuild PIX | Sistema Antifraude

> **"Specifications are the source of truth; code is derived and transient."**

**Projeto:** BRB Rebuild PIX — Sistema Antifraude PIX
**Versão:** 1.0
**Data de ratificação:** 2026-04-17
**Status:** 🔒 ATIVO — IMUTÁVEL SEM AMENDMENT FORMAL

---

## Preâmbulo

Este documento define as **leis não-negociáveis** que governam o projeto BRB Rebuild PIX. Qualquer código, especificação, plano ou tarefa que contrarie esta Constituição é **inválido por definição** e deve ser corrigido ou rejeitado.

Alterações nesta Constituição requerem:
1. Proposta formal (amendment) com justificativa técnica
2. Aprovação explícita de Adilio
3. Bump de versão major
4. Migração documentada de artefatos dependentes

---

## Artigo I — Filosofia de Engenharia

### §1.1 Princípios Fundamentais

1. **Specification-First:** nenhuma linha de código produtivo é escrita sem especificação formal (SPEC.md ou ticket SDD equivalente).
2. **Honest Engineering:** métricas, baselines e resultados são sempre reais. **Cherry-picking, hardcoded scores, ou métricas fabricadas são violações constitucionais graves.**
3. **Reproducibility-by-Default:** todo experimento é reproduzível — seed, versão de código, versão de dados, config.
4. **No Cargo-Culting:** toda decisão técnica tem justificativa baseada em evidência, paper, ou experimento. "Porque é moda" não é justificativa.
5. **Fail Loud, Fail Early:** erros silenciosos são proibidos. Guardrails automáticos são obrigatórios.

### §1.2 Hierarquia de Autoridade (Reality Tunnel)

Em caso de conflito, a ordem de prevalência é **imutável**:
CONSTITUTION.md (esta) ↓ SPEC.md (requisitos funcionais/não-funcionais) ↓ PLAN.md (arquitetura e tecnologia) ↓ TASKS.md (decomposição) ↓ Código (artefato transitório)




Código que contradiz SPEC é **código errado**, não spec errada.

---

## Artigo II — Domínio de Negócio (Antifraude PIX)

### §2.1 Assimetria de Custos (Lei Fundamental)

Antifraude PIX tem **assimetria estrutural**:

$$
\text{Custo}(FN) \gg \text{Custo}(FP)
$$

**Consequência operacional:**
- FN (fraude não detectada) = perda financeira direta + risco regulatório + dano reputacional
- FP (legítimo flagrado) = fricção recuperável via 2FA ou contato humano
- **Recall é prioritário sobre Precision** na função de perda.

### §2.2 Hierarquia de Decisões

O sistema produz **exatamente 3 decisões**:

| Decisão | Significado | Ação Operacional |
|---|---|---|
| `APROVAR` | Baixo risco | Transação segue sem fricção |
| `CONFIRMAR` | Risco médio | 2FA ou notificação ao cliente |
| `BLOQUEAR` | Alto risco | Transação impedida + investigação |

**Não existem outras decisões.** Adicionar novas decisões requer amendment constitucional.

### §2.3 Metas Operacionais Mínimas

O sistema em produção deve sustentar:

| Métrica | Mínimo Aceitável | Alvo | Atual (FASE 0) |
|---|---:|---:|---:|
| Recall | ≥ 85% | ≥ 95% | 93.52% |
| Precision | ≥ 70% | ≥ 90% | 97.65% |
| F1 | ≥ 0.80 | ≥ 0.92 | 0.9554 |
| FPR | ≤ 2.0% | ≤ 0.5% | 0.14% |
| Latência (p95) | ≤ 500ms | ≤ 200ms | TBD |

**Regressão abaixo do mínimo aceitável bloqueia deploy.**

---

## Artigo III — Tecnologia Mandatória

### §3.1 Stack Obrigatório

| Camada | Tecnologia | Razão |
|---|---|---|
| Linguagem | **Python 3.12+** | Tipagem moderna (`str \| None`, `list[int]`) |
| Package manager | **uv** | 10-100x mais rápido que pip |
| Linter + formatter | **ruff** | Substitui black+isort+flake8 |
| Type checker | **mypy** | Validação estática obrigatória |
| Testing | **pytest** | Padrão de facto |
| Data (tabular) | **pandas** (legacy) / **polars** (novo) | Pandas permitido; polars preferido em código novo |
| ML core | **scikit-learn, LightGBM** | Estabilidade, interpretabilidade |
| Anomaly | **IsolationForest (sklearn)** | Simplicidade, explicabilidade |
| Logging | **logging** (stdlib) | `print` é proibido fora de CLI |

### §3.2 Stack Proibido

- ❌ `print()` em código de produção (exceto CLI output explícito)
- ❌ `pickle` para dados persistentes (usar `joblib` ou JSON)
- ❌ `eval()` / `exec()` sobre input externo
- ❌ Dependências com licenças GPL/AGPL sem aprovação jurídica
- ❌ Modelos em formato não-serializável reproduzível

### §3.3 Estrutura de Diretórios (imutável)
rebuild_pix/ ├── backend/ │ ├── core/ # Engines (decision, behavioral, social, pipeline) │ ├── preprocessing/ # Feature engineering │ ├── artefatos/ # Modelos treinados (.joblib) │ ├── scripts/ # CLI executáveis │ └── tests/ # pytest ├── dados/ # Datasets (gitignored se >50MB) ├── resultados/ # Outputs de experimentos (gitignored) ├── docs/ │ ├── CONSTITUTION.md # Este arquivo │ ├── SPEC.md # Requisitos │ ├── PLAN.md # Arquitetura │ ├── TASKS.md # Sprint atual │ └── experiments/ # EXP-*.md └── pyproject.toml # uv




---

## Artigo IV — Padrões de Código

### §4.1 Type Hints (Obrigatório)

Toda função pública **deve** ter type hints completos:

```python
# ✅ CORRETO
def score_transaction(
    tx: dict[str, Any],
    threshold: float = 0.5,
) -> DecisionResult:
    ...

# ❌ PROIBIDO
def score_transaction(tx, threshold=0.5):
    ...
Usar sintaxe moderna:

str | None em vez de Optional[str]
list[int] em vez de List[int]
dict[str, Any] em vez de Dict[str, Any]
§4.2 Docstrings (Google-style)
python


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula métricas de classificação binária.

    Args:
        y_true: Ground truth (shape: [N]).
        y_pred: Predições binárias (shape: [N]).

    Returns:
        Dict com keys: precision, recall, f1, fpr.

    Raises:
        ValueError: Se shapes incompatíveis.
    """
§4.3 Logging
python


# ✅ CORRETO
logger.info(f"Pipeline iniciado: {len(df)} tx")
logger.warning(f"SE ativou em apenas {rate:.2%} — verificar calibração")

# ❌ PROIBIDO (fora de CLI)
print(f"Pipeline iniciado: {len(df)} tx")
§4.4 Exception Handling
python


# ✅ CORRETO — específico e informativo
try:
    result = orquestrador.analisar(tx)
except KeyError as e:
    logger.error(f"Campo obrigatório ausente: {e}")
    raise ValidationError(f"Transação inválida: {e}") from e

# ❌ PROIBIDO — genérico e silencioso
try:
    result = orquestrador.analisar(tx)
except:
    pass
§4.5 Configuração vs Lógica
Thresholds, pesos, limiares: em scoring_config.json ou equivalente
Nunca hardcoded em código
Configs versionadas no Git
Artigo V — Machine Learning e IA
§5.1 Reprodutibilidade
Todo experimento ML deve:

Fixar seed em: numpy, random, sklearn, pytorch (se aplicável)
Registrar versão do dataset (hash ou timestamp)
Registrar versão do código (git commit SHA)
Salvar config completa junto com resultado
Logar métricas via estrutura padrão (ver §5.4)
§5.2 Zero Data Leakage
Split temporal obrigatório (não shuffle aleatório em dados temporais)
Features derivadas de customer_id não podem usar informação futura
Target encoding feito apenas com dados de treino
Validação holdout intocada até experimento final
§5.3 Baselines Obrigatórios
Todo modelo novo deve reportar métricas contra:

Baseline trivial (classe majoritária ou random)
Baseline atual em produção (se existir)
Baseline simples (regra heurística de negócio)
Sem baselines, resultados não são interpretáveis.

§5.4 Protocolo de Experimento (obrigatório)
markdown


# EXP-XXX: [Título]

**Hipótese:** [O que esperamos provar/refutar]
**Baseline:** [Métricas do sistema atual]
**Setup:**
  - Dataset: [caminho + hash]
  - Seed: [valor]
  - Config: [link para config]
**Resultados:** [métricas com CI quando possível]
**Análise:** [por que obtivemos esses resultados]
**Conclusão:** [hipótese confirmada/refutada + próximos passos]
Artigo VI — Segurança e Compliance
§6.1 Dados Sensíveis
CPFs, nomes, emails: nunca logados em texto plano
Logs com PII devem usar hash ou mask
Outputs de debug nunca incluem PII
Datasets com PII não versionados no Git
§6.2 LGPD/BACEN
Decisões de bloqueio devem ser explicáveis (SHAP, regras nomeadas)
Cliente tem direito a contestar decisão automatizada
Retenção de dados conforme política BRB
Auditoria completa: toda decisão registrada com timestamp + razão
§6.3 Ataques Adversariais Reconhecidos
O sistema deve ser resiliente a:

Evasão: fraudadores tentando parecer legítimos
Envenenamento: injeção de rótulos errados no treino
Extração de modelo: queries massivas para reverse-engineering
Mitigações são detalhadas em PLAN.md.

Artigo VII — Processo de Desenvolvimento
§7.1 Metodologia
Spec-Driven Development (SDD) é obrigatório. As 6 fases:

CONSTITUTION (este documento) — leis
SPEC — requisitos (EARS syntax)
PLAN — arquitetura
TASKS — decomposição
IMPLEMENT — código + testes
VALIDATE & EVOLVE — experimentação
§7.2 Git
Branch principal: main (protegido)
Branches de feature: feat/<ticket>-<descrição>
Branches de experimento: exp/<exp-id>-<descrição>
Commits: Conventional Commits (feat:, fix:, exp:, docs:, refactor:)
PRs obrigatórios para main
§7.3 Code Review
Obrigatório para código em backend/core/
Opcional para scripts de análise exploratória
Revisor valida: type hints, docstrings, testes, aderência ao SPEC
§7.4 Testes



Tipo	Obrigatório para	Cobertura mínima
Unit	backend/core/	80%
Integration	Pipeline E2E	100% dos fluxos
Regression	Após mudanças em engine	Sample de 1000 tx
§7.5 Artefatos Versionados
Sempre no Git:

Código-fonte
Specs, plans, tasks, experiments
Configs (scoring_config.json, etc.)
pyproject.toml, uv.lock
Nunca no Git (usar DVC ou S3):

Datasets brutos
Modelos treinados >50MB
Resultados de experimentos
Artigo VIII — Guardrails Automáticos
§8.1 Pre-commit
ruff format + ruff check
mypy sem erros
pytest dos testes unitários relevantes
§8.2 CI/CD
Testes unitários em toda PR
Testes de regressão em mudanças no engine
Validação de métricas mínimas antes de merge em main
§8.3 Produção
Monitoramento contínuo de: recall, precision, FPR, latência
Alertas automáticos se métricas caem abaixo de §2.3
Dashboard de drift (features e predições)
Circuit breaker: rollback automático se degradação detectada
Artigo IX — Éticos e Responsabilidade
§9.1 Bias e Justiça
O sistema não pode:

Discriminar por gênero, raça, religião, orientação sexual
Usar features proxy de atributos protegidos sem justificativa documentada
Produzir taxas de FP substancialmente maiores para grupos específicos
Auditoria de bias é obrigatória trimestralmente.

§9.2 Transparência
Clientes bloqueados recebem razão humana-inteligível
Processo de contestação disponível
Relatórios periódicos ao BACEN conforme regulação
§9.3 Limite de Automação
Decisões de bloqueio permanente ou encerramento de conta requerem revisão humana. O sistema apenas recomenda — humano decide.

Artigo X — Amendment
Esta Constituição pode ser alterada via:

Proposta de Amendment (formato: docs/amendments/AMD-XXX.md)
Justificativa técnica/operacional
Impacto em SPEC/PLAN/TASKS
Plano de migração
Discussão e revisão (mínimo 48h)
Aprovação explícita de Adilio
Merge com bump de versão major
Notificação a todos os stakeholders
Amendments retroativos são proibidos — todas as decisões passadas permanecem válidas sob sua versão constitucional.

Apêndice A — Glossário



Termo	Definição
FN	False Negative — fraude não detectada
FP	False Positive — legítima flagrada
LGBM	LightGBM (gradient boosting)
IF	Isolation Forest (anomaly detection)
SE	Social Engineering (módulo de detecção de engenharia social)
BEH	Behavioral Analytics (análise comportamental)
Veto	Override do score final por regra cirúrgica
Cascade	Sequência de regras de bloqueio
PIX	Sistema de pagamentos instantâneos do Brasil (BACEN)
Ratificado em: 2026-04-17 Ratificado por: Adilio (Product Owner) + AI Engineer (Technical Lead) Versão: 1.0 Próxima revisão agendada: 2026-10-17 (6 meses)