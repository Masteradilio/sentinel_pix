# EXP-008E — Manifest, Experiment Index e Journal

Gerado em: `2026-05-09T15:30:00`

## Objetivo

Criar artefatos mínimos de versionamento e governança do baseline pós-FASE 2 / FASE 3.

## Artefatos gerados

- `backend/artefatos/MANIFEST_MODEL.json`
- `resultados/experimentos/EXPERIMENT_INDEX.md`
- `docs/JOURNAL.md`
- `resultados/experimentos/EXP-008E/README.md`

## Journal

Entradas novas adicionadas: `6`

- `FASE2_BASELINE_POS_C1`
- `EXP008A_REGRESSION_SUITE_POS_C1`
- `EXP008B_VALIDATION_REPORT`
- `EXP008C_RULES_TRACE`
- `EXP008D_CLEANUP_RESTRICAO`
- `EXP008E_JOURNAL_CRIADO`

## Regressão

- Executada em: `2026-05-09T15:30:00`
- Tudo passou: `True`

### `E:\Projetos\rebuild_pix\venv\Scripts\python.exe -m py_compile backend/core/decision_engine.py`

```text
(sem saída)
```

### `E:\Projetos\rebuild_pix\venv\Scripts\python.exe -m py_compile backend/core/pipeline_orquestrador.py`

```text
(sem saída)
```

### `E:\Projetos\rebuild_pix\venv\Scripts\python.exe -m py_compile backend/scripts/simular_pipeline_e2e_v2.py`

```text
(sem saída)
```

### `E:\Projetos\rebuild_pix\venv\Scripts\python.exe -m pytest tests/test_regression_post_fase2.py -q`

```text
......                                                                   [100%]
6 passed in 2.51s
```

### `E:\Projetos\rebuild_pix\venv\Scripts\python.exe -m pytest tests/test_regression_post_fase2.py -q -m slow`

```text
.                                                                        [100%]
1 passed, 5 deselected in 2.32s
```
