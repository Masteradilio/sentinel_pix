"""
scripts/build_exp008e_manifest_and_journal.py

EXP-008E — Experiment Manifest, Experiment Index e Journal

Gera/atualiza:
  backend/artefatos/MANIFEST_MODEL.json
  resultados/experimentos/EXPERIMENT_INDEX.md
  docs/JOURNAL.md
  resultados/experimentos/EXP-008E/README.md

Objetivo:
  Versionar o estado oficial pós-FASE 2 / FASE 3 e criar um journal
  append-only para registrar decisões importantes sem inflar o plano de melhoria.

Uso:
  python scripts\\build_exp008e_manifest_and_journal.py

Opcional:
  python scripts\\build_exp008e_manifest_and_journal.py --run-regression
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists():
            return p
    return start.parent


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = find_project_root(SCRIPT_DIR)

DOCS_DIR = ROOT / "docs"
ARTEFATOS_DIR = ROOT / "backend" / "artefatos"
RESULTADOS_DIR = ROOT / "resultados" / "experimentos"
EXP008E_DIR = RESULTADOS_DIR / "EXP-008E"

SCORING_PATH = ARTEFATOS_DIR / "scoring_config.json"
MANIFEST_PATH = ARTEFATOS_DIR / "MANIFEST_MODEL.json"
EXPERIMENT_INDEX_PATH = RESULTADOS_DIR / "EXPERIMENT_INDEX.md"
JOURNAL_PATH = DOCS_DIR / "JOURNAL.md"
EXP008E_README_PATH = EXP008E_DIR / "README.md"


OFFICIAL_METRICS = {
    "seed_42": {
        "TP": 347,
        "FP": 14,
        "FN": 8,
        "Precision": 0.961219,
        "Recall": 0.977465,
        "F1": 0.9693,
    },
    "seed_123": {
        "TP": 347,
        "FP": 12,
        "FN": 8,
        "Precision": 0.966574,
        "Recall": 0.977465,
        "F1": 0.9720,
    },
}


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_regression() -> dict[str, Any]:
    commands = [
        [sys.executable, "-m", "py_compile", "backend/core/decision_engine.py"],
        [sys.executable, "-m", "py_compile", "backend/core/pipeline_orquestrador.py"],
        [sys.executable, "-m", "py_compile", "backend/scripts/simular_pipeline_e2e_v2.py"],
        [sys.executable, "-m", "pytest", "tests/test_regression_post_fase2.py", "-q"],
        [sys.executable, "-m", "pytest", "tests/test_regression_post_fase2.py", "-q", "-m", "slow"],
    ]

    results = []

    for cmd in commands:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
            timeout=180,
        )

        results.append(
            {
                "command": " ".join(cmd),
                "returncode": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        )

    return {
        "ran_at": datetime.now().isoformat(timespec="seconds"),
        "all_passed": all(r["returncode"] == 0 for r in results),
        "results": results,
    }


def build_manifest(config: dict[str, Any], regression: dict[str, Any] | None) -> dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")

    active_config_keys = [
        "threshold_confirmar",
        "threshold_bloquear",
        "lgbm_guard_enabled",
        "lgbm_guard_threshold",
        "guard_exception_alto_valor_se_beh_enabled",
        "guard_exception_alto_valor_min",
        "guard_exception_alto_valor_rel_max",
        "guard_exception_alto_valor_if_min",
        "guard_exception_alto_valor_lgbm_min",
        "guard_exception_alto_valor_require_first_receiver",
        "guard_exception_alto_valor_require_pf",
        "exp006f_c1_enabled",
        "exp006f_c1_min_score",
        "exp006f_c1_max_score",
        "exp006f_c1_min_valor",
        "exp006f_c1_max_valor",
        "exp006f_c1_max_rel_meses",
        "exp006f_c1_min_lgbm_raw",
        "exp006f_c1_max_lgbm_raw",
        "exp006f_c1_require_first_receiver",
        "exp006f_c1_require_not_pix_random",
        "exp006f_c1_max_se_score",
        "exp006f_c1_max_beh_score",
        "se_pattern_residual_enabled",
        "exp003_residual_confirm_enabled",
    ]

    active_config = {key: config.get(key) for key in active_config_keys if key in config}

    return {
        "manifest_schema_version": "1.0",
        "generated_at": now,
        "model_version": "post_fase2_c1",
        "decision_engine_version": "v3.0.5_post_c1_exp008d",
        "project_phase": "FASE_3_CONSOLIDACAO_OPERACIONAL",
        "status": "ACTIVE_BASELINE",
        "active_lgbm": "baseline_producao_pre_lgbm_v6_2",
        "active_rules": [
            "V1_GUARD_CONTEXTUAL",
            "C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER",
        ],
        "active_guardrails": [
            "LGBM_GUARD_RAIL",
        ],
        "rejected_candidates": [
            {
                "id": "LGBM_C_SPW_2_0X",
                "source": "EXP-005A/EXP-005B",
                "reason": "Promissor model-only, mas não reduziu FN líquido no engine real e aumentou FP.",
            },
            {
                "id": "R2_LOW_VALUE_GRAY_FIRST_RECEIVER",
                "source": "EXP-006C",
                "reason": "Recuperou 0 FN e adicionou FP.",
            },
            {
                "id": "META_LEARNER_SHADOW",
                "source": "EXP-007A",
                "reason": "Não encontrou candidato seguro para overlay adicional.",
            },
        ],
        "disabled_rules": [
            "EXP003_RESIDUAL",
            "SE_PATTERN_RESIDUAL",
        ],
        "official_metrics": OFFICIAL_METRICS,
        "known_limitations": [
            "Restam 8 FNs residuais após C1.",
            "EXP-007A não encontrou overlay seguro com os sinais atuais.",
            "Melhoria adicional de recall depende provavelmente de novas fontes de dados/sinais.",
        ],
        "active_config": active_config,
        "artifact_hashes": {
            "scoring_config_sha256": sha256_file(SCORING_PATH),
            "validation_report_sha256": sha256_file(DOCS_DIR / "VALIDATION_REPORT_POST_FASE2.md"),
            "rules_catalog_sha256": sha256_file(DOCS_DIR / "RULES_CATALOG.md"),
            "decision_trace_spec_sha256": sha256_file(DOCS_DIR / "DECISION_TRACE_SPEC.md"),
            "decision_trace_example_sha256": sha256_file(DOCS_DIR / "DECISION_TRACE_EXAMPLE.json"),
            "regression_test_sha256": sha256_file(ROOT / "tests" / "test_regression_post_fase2.py"),
        },
        "regression": regression,
        "required_regression_commands": [
            "python -m pytest tests\\test_regression_post_fase2.py -q",
            "python -m pytest tests\\test_regression_post_fase2.py -q -m slow",
        ],
        "notes": [
            "EXP-008D aprovado com restrição.",
            "Wrapper efetivo da C1 permanece em backend/scripts/simular_pipeline_e2e_v2.py.",
            "Wrapper redundante no pipeline_orquestrador.py foi mantido por segurança.",
            "Binding defensivo _hydrate_config_from_scoring_config foi reposto no decision_engine.py.",
            "Journal criado no EXP-008E para registrar decisões técnicas sem inflar o plano de melhoria.",
        ],
    }


def build_experiment_index() -> str:
    now = datetime.now().isoformat(timespec="seconds")

    return f"""# Experiment Index — Pipeline Antifraude PIX

**Gerado em:** `{now}`

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
| EXP-008E | Executado | Manifest / Index / Journal | Versionamento e registro de decisões |

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
python -m pytest tests\\test_regression_post_fase2.py -q
python -m pytest tests\\test_regression_post_fase2.py -q -m slow
```
"""


def journal_header() -> str:
    return """# Journal — Decisões Técnicas do Pipeline Antifraude PIX

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
"""


def decision_entry(decision_id: str, title: str, body: str) -> str:
    now = datetime.now().isoformat(timespec="seconds")

    return f"""
<!-- decision_id: {decision_id} -->

## {title}

**Data:** `{now}`  
**ID:** `{decision_id}`

{body.strip()}

---
"""


def append_journal_entry_if_missing(path: Path, decision_id: str, title: str, body: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = journal_header()

    marker = f"<!-- decision_id: {decision_id} -->"

    if marker in text:
        return False

    text = text.rstrip() + "\n" + decision_entry(decision_id, title, body)
    path.write_text(text, encoding="utf-8")

    return True


def update_journal() -> list[str]:
    added: list[str] = []

    entries = [
        (
            "FASE2_BASELINE_POS_C1",
            "FASE 2 encerrada com baseline pós-C1",
            """
**Decisão:** encerrar a FASE 2 com sucesso mínimo validado.

Baseline oficial:

| Seed | TP | FP | FN | F1 |
|---:|---:|---:|---:|---:|
| 42 | 347 | 14 | 8 | 0,9693 |
| 123 | 347 | 12 | 8 | 0,9720 |

**Racional:** a C1 recuperou 1 FN nos dois seeds, adicionou 0 FP e não perdeu TP. O EXP-007A não encontrou candidato seguro adicional com os sinais atuais.

**Consequência:** novas reduções relevantes de FN provavelmente dependem de novos sinais/dados, não de novas regras sobre os mesmos sinais.
""",
        ),
        (
            "EXP008A_REGRESSION_SUITE_POS_C1",
            "EXP-008A aprovado — suíte de regressão pós-C1",
            """
**Decisão:** tornar a regressão pós-C1 obrigatória antes de qualquer mudança futura.

Validação conhecida:

```text
pytest normal: 6 passed
pytest slow: 1 passed, 5 deselected
```

**Racional:** a regressão protege a C1, o baseline pós-FASE 2, o scoring_config e a validação runtime da transação alvo.
""",
        ),
        (
            "EXP008B_VALIDATION_REPORT",
            "EXP-008B aprovado — Validation Report Pós-FASE 2",
            """
**Decisão:** aceitar `docs/VALIDATION_REPORT_POST_FASE2.md` como relatório oficial da versão pós-FASE 2.

**Racional:** o relatório consolida baseline pós-C1, métricas oficiais, deltas da C1, FNs residuais, decisões promovidas/rejeitadas e comandos obrigatórios de regressão.
""",
        ),
        (
            "EXP008C_RULES_TRACE",
            "EXP-008C aprovado — Rules Catalog e Decision Trace",
            """
**Decisão:** aceitar `docs/RULES_CATALOG.md`, `docs/DECISION_TRACE_SPEC.md` e `docs/DECISION_TRACE_EXAMPLE.json`.

**Racional:** os artefatos documentam regras ativas, regras rejeitadas, thresholds, guard rails, campos mínimos de rastreabilidade e exemplo de decisão C1.
""",
        ),
        (
            "EXP008D_CLEANUP_RESTRICAO",
            "EXP-008D aprovado com restrição — cleanup técnico dos patches",
            """
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
""",
        ),
        (
            "EXP008E_JOURNAL_CRIADO",
            "EXP-008E — criação do Journal de decisões",
            """
**Decisão:** criar `docs/JOURNAL.md` como registro cronológico e append-only de decisões técnicas importantes.

**Racional:** o `plano_melhoria_critica.md` deve permanecer estratégico. Decisões detalhadas, restrições, trade-offs e resultados de validação devem ser registrados no journal para evitar que o plano fique excessivamente grande.

**Uso esperado:** toda decisão promovida, rejeitada ou aprovada com restrição deve ganhar uma entrada no journal.
""",
        ),
    ]

    for decision_id, title, body in entries:
        did_add = append_journal_entry_if_missing(JOURNAL_PATH, decision_id, title, body)
        if did_add:
            added.append(decision_id)

    return added


def build_exp008e_readme(
    regression: dict[str, Any] | None,
    added_journal_entries: list[str],
) -> str:
    now = datetime.now().isoformat(timespec="seconds")

    lines = [
        "# EXP-008E — Manifest, Experiment Index e Journal",
        "",
        f"Gerado em: `{now}`",
        "",
        "## Objetivo",
        "",
        "Criar artefatos mínimos de versionamento e governança do baseline pós-FASE 2 / FASE 3.",
        "",
        "## Artefatos gerados",
        "",
        "- `backend/artefatos/MANIFEST_MODEL.json`",
        "- `resultados/experimentos/EXPERIMENT_INDEX.md`",
        "- `docs/JOURNAL.md`",
        "- `resultados/experimentos/EXP-008E/README.md`",
        "",
        "## Journal",
        "",
        f"Entradas novas adicionadas: `{len(added_journal_entries)}`",
        "",
    ]

    if added_journal_entries:
        for item in added_journal_entries:
            lines.append(f"- `{item}`")
    else:
        lines.append("- Nenhuma entrada nova; entradas já existiam.")

    lines.extend(
        [
            "",
            "## Regressão",
            "",
        ]
    )

    if regression:
        lines.append(f"- Executada em: `{regression['ran_at']}`")
        lines.append(f"- Tudo passou: `{regression['all_passed']}`")
        lines.append("")

        for result in regression["results"]:
            lines.append(f"### `{result['command']}`")
            lines.append("")
            lines.append("```text")
            lines.append(result["stdout"] or result["stderr"] or "(sem saída)")
            lines.append("```")
            lines.append("")
    else:
        lines.append("Regressão não executada automaticamente nesta geração.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EXP-008E governance artifacts")
    parser.add_argument("--run-regression", action="store_true")
    args = parser.parse_args()

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    ARTEFATOS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    EXP008E_DIR.mkdir(parents=True, exist_ok=True)

    if not SCORING_PATH.exists():
        raise FileNotFoundError(f"scoring_config.json não encontrado: {SCORING_PATH}")

    config = read_json(SCORING_PATH, {})
    regression = run_regression() if args.run_regression else None

    manifest = build_manifest(config, regression)
    write_json(MANIFEST_PATH, manifest)

    EXPERIMENT_INDEX_PATH.write_text(build_experiment_index(), encoding="utf-8")

    added_journal_entries = update_journal()

    EXP008E_README_PATH.write_text(
        build_exp008e_readme(regression, added_journal_entries),
        encoding="utf-8",
    )

    print(f"[OK] Manifest gerado: {MANIFEST_PATH}")
    print(f"[OK] Experiment index gerado: {EXPERIMENT_INDEX_PATH}")
    print(f"[OK] Journal atualizado: {JOURNAL_PATH}")
    print(f"[OK] EXP-008E README gerado: {EXP008E_README_PATH}")
    print(f"[OK] Entradas novas no journal: {len(added_journal_entries)}")

    if regression:
        print(f"[OK] Regressão executada. all_passed={regression['all_passed']}")
        if not regression["all_passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()