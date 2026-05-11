# Journal — Decisões Técnicas do Pipeline Antifraude PIX

Este journal registra decisões técnicas relevantes do projeto, em formato cronológico e append-only.

## Objetivo

- evitar que o `plano_melhoria_critica.md` fique excessivamente grande;
- preservar o racional de decisões importantes;
- registrar trade-offs, restrições e motivos de promoção/rejeição;
- facilitar auditoria futura;
- manter histórico de mudanças entre fases.

## Regra de uso

```text
Não apagar decisões antigas.
Se uma decisão mudar, adicionar uma nova entrada explicando a mudança.
```

---

<!-- decision_id: FASE2_BASELINE_POS_C1 -->

## FASE 2 encerrada com baseline pós-C1

**Data:** `2026-05-09T15:30:00`  
**ID:** `FASE2_BASELINE_POS_C1`

**Decisão:** encerrar a FASE 2 com sucesso mínimo validado.

Baseline oficial:

| Seed | TP | FP | FN | F1 |
|---:|---:|---:|---:|---:|
| 42 | 347 | 14 | 8 | 0,9693 |
| 123 | 347 | 12 | 8 | 0,9720 |

**Racional:** a C1 recuperou 1 FN nos dois seeds, adicionou 0 FP e não perdeu TP. O EXP-007A não encontrou candidato seguro adicional com os sinais atuais.

**Consequência:** novas reduções relevantes de FN provavelmente dependem de novos sinais/dados, não de novas regras sobre os mesmos sinais.

---

<!-- decision_id: EXP008A_REGRESSION_SUITE_POS_C1 -->

## EXP-008A aprovado — suíte de regressão pós-C1

**Data:** `2026-05-09T15:30:00`  
**ID:** `EXP008A_REGRESSION_SUITE_POS_C1`

**Decisão:** tornar a regressão pós-C1 obrigatória antes de qualquer mudança futura.

Validação conhecida:

```text
pytest normal: 6 passed
pytest slow: 1 passed, 5 deselected
```

**Racional:** a regressão protege a C1, o baseline pós-FASE 2, o scoring_config e a validação runtime da transação alvo.

---

<!-- decision_id: EXP008B_VALIDATION_REPORT -->

## EXP-008B aprovado — Validation Report Pós-FASE 2

**Data:** `2026-05-09T15:30:00`  
**ID:** `EXP008B_VALIDATION_REPORT`

**Decisão:** aceitar `docs/VALIDATION_REPORT_POST_FASE2.md` como relatório oficial da versão pós-FASE 2.

**Racional:** o relatório consolida baseline pós-C1, métricas oficiais, deltas da C1, FNs residuais, decisões promovidas/rejeitadas e comandos obrigatórios de regressão.

---

<!-- decision_id: EXP008C_RULES_TRACE -->

## EXP-008C aprovado — Rules Catalog e Decision Trace

**Data:** `2026-05-09T15:30:00`  
**ID:** `EXP008C_RULES_TRACE`

**Decisão:** aceitar `docs/RULES_CATALOG.md`, `docs/DECISION_TRACE_SPEC.md` e `docs/DECISION_TRACE_EXAMPLE.json`.

**Racional:** os artefatos documentam regras ativas, regras rejeitadas, thresholds, guard rails, campos mínimos de rastreabilidade e exemplo de decisão C1.

---

<!-- decision_id: EXP008D_CLEANUP_RESTRICAO -->

## EXP-008D aprovado com restrição — cleanup técnico dos patches

**Data:** `2026-05-09T15:30:00`  
**ID:** `EXP008D_CLEANUP_RESTRICAO`

**Decisão:** aprovar o cleanup técnico com restrição.

**Removido:**
- resíduos órfãos do wrapper runtime antigo no `decision_engine.py`.

**Reposto:**
- binding defensivo `_hydrate_config_from_scoring_config`, pois o runtime do `PixDecisionEngine` ainda depende dele na inicialização.

**Mantido:**
- wrapper efetivo em `backend/scripts/simular_pipeline_e2e_v2.py`;
- wrapper redundante em `backend/core/pipeline_orquestrador.py`, por segurança;
- campos C1 no `EngineConfig`;
- configuração C1 no `scoring_config.json`.

**Racional:** a tentativa de remoção automática do wrapper do `pipeline_orquestrador.py` apresentou risco de quebra. A prioridade da FASE 3 é estabilidade e regressão verde, não limpeza estética.

Validação final:

```text
py_compile decision_engine.py: OK
py_compile pipeline_orquestrador.py: OK
py_compile simular_pipeline_e2e_v2.py: OK
hydrate OK True
C1 field OK True
pytest normal: 6 passed
pytest slow: 1 passed, 5 deselected
```

---

<!-- decision_id: EXP008E_JOURNAL_CRIADO -->

## EXP-008E — criação do Journal de decisões

**Data:** `2026-05-09T15:30:00`  
**ID:** `EXP008E_JOURNAL_CRIADO`

**Decisão:** criar `docs/JOURNAL.md` como registro cronológico e append-only de decisões técnicas importantes.

**Racional:** o `plano_melhoria_critica.md` deve permanecer estratégico. Decisões detalhadas, restrições, trade-offs e resultados de validação devem ser registrados no journal para evitar que o plano fique excessivamente grande.

**Uso esperado:** toda decisão promovida, rejeitada ou aprovada com restrição deve ganhar uma entrada no journal.

---
