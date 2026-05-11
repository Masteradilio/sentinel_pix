# Experiment Index — Pipeline Antifraude PIX

**Gerado em:** `2026-05-09T15:30:00`

Este índice resume as principais rodadas experimentais e decisões de promoção/rejeição.

## Estado oficial atual

```text
Versão ativa: post_fase2_c1
Fase atual: FASE 3 — Consolidação Operacional
Baseline oficial:
  seed 42:  TP=347, FP=14, FN=8, F1≈0,9693
  seed 123: TP=347, FP=12, FN=8, F1≈0,9720
```

## Índice de experimentos

| Experimento | Status | Decisão | Observação |
|---|---|---|---|
| EXP-001 | Concluído | Diagnóstico | Base inicial de melhoria |
| EXP-002 | Concluído | Diagnóstico | Avaliação incremental |
| EXP-003 | Rejeitado/desligado | Não promover | Residual com risco de FP |
| EXP-004-FINAL | Promovido | Promover V1 | `V1_GUARD_CONTEXTUAL` recuperou FN sem FP |
| EXP-005A | Concluído | Não promover direto | LGBM v6.2 promissor model-only |
| EXP-005B | Rejeitado | Não promover LGBM v6.2 | Engine real não teve ganho líquido |
| EXP-006 | Concluído | Diagnóstico | Cartografia de erros residuais |
| EXP-006B | Concluído | Diagnóstico | Contrafactuais do engine |
| EXP-006C/R2 | Rejeitado | Não promover R2 | 0 FN recuperado, FP adicionado |
| EXP-006D | Concluído | Diagnóstico | Censo dos FNs residuais |
| EXP-006E | Aprovado para quick-E2E | Testar C1 | C1 artifact-only positiva |
| EXP-006F | Promovido | Promover C1 | 1 FN recuperado, 0 FP adicionado |
| EXP-007A | Diagnóstico | Não promover meta-learner | Sem candidato seguro |
| EXP-008A | Aprovado | Regressão pós-C1 | `6 passed`; slow `1 passed` |
| EXP-008B | Aprovado | Validation Report | Baseline pós-FASE 2 formalizado |
| EXP-008C | Aprovado | Rules Catalog / Trace | Regras e rastreabilidade documentadas |
| EXP-008D | Aprovado com restrição | Cleanup parcial seguro | Estabilidade priorizada |
| EXP-008E | Aprovado | Manifest / Index / Journal | Versionamento e registro de decisões |

## Experimentos rejeitados formalmente

- `LGBM_C_SPW_2_0X`
- `R2_LOW_VALUE_GRAY_FIRST_RECEIVER`
- `META_LEARNER_SHADOW` como componente de decisão
- regra ampla baseada apenas em `first_receiver_flag`
- `EXP003_RESIDUAL`

## Artefatos oficiais

- `docs/VALIDATION_REPORT_POST_FASE2.md`
- `docs/RULES_CATALOG.md`
- `docs/DECISION_TRACE_SPEC.md`
- `docs/DECISION_TRACE_EXAMPLE.json`
- `docs/JOURNAL.md`
- `backend/artefatos/MANIFEST_MODEL.json`
- `tests/test_regression_post_fase2.py`

## Procedimento obrigatório antes de qualquer mudança

```powershell
python -m pytest tests\test_regression_post_fase2.py -q
python -m pytest tests\test_regression_post_fase2.py -q -m slow
```
