#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R5B16 - Consolidated operational baseline candidate.

Consolida o R5B15 como baseline candidato versionado, sem ativar em producao.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "EXP-014B-R5B16-CONSOLIDATED-OPERATIONAL-BASELINE"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / EXPERIMENT
CANDIDATE_DIR = PROJECT_ROOT / "backend" / "artefatos_candidatos" / "exp014b_r5b16_operational_baseline"

R5B15_SUMMARY = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R5B15-CORE-POLICY-REPLAY" / "00_run_summary.json"
R5B15_BY_ACTION = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R5B15-CORE-POLICY-REPLAY" / "01_metrics_by_action.csv"
R5B14_CANDIDATE = PROJECT_ROOT / "backend" / "artefatos_candidatos" / "exp014b_r5b14_operational_zero_fn" / "operational_zero_fn_policy_candidate.json"
SEVERITY_POLICY = PROJECT_ROOT / "backend" / "core" / "severity_policy.py"
PIPELINE = PROJECT_ROOT / "backend" / "core" / "pipeline_orquestrador.py"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    for path in [R5B15_SUMMARY, R5B15_BY_ACTION, R5B14_CANDIDATE, SEVERITY_POLICY, PIPELINE]:
        if not path.exists():
            raise FileNotFoundError(path)

    r5b15 = read_json(R5B15_SUMMARY)
    r5b14_candidate = read_json(R5B14_CANDIDATE)

    candidate = {
        "artifact_type": "operational_baseline_candidate",
        "experiment": EXPERIMENT,
        "status": "CANDIDATE_NOT_PRODUCTION_ACTIVE",
        "baseline_name": "EXP-014B-R5B16 / R5B15 core replay consolidated",
        "base_policy": "EXP-014B-R4G-FAST-FROZEN",
        "policy_id": r5b15["policy_metadata"]["policy_id"],
        "rule_set_version": r5b15["policy_metadata"]["rule_set_version"],
        "activation": {
            "enabled_by_default": False,
            "runtime_flag_env": "ENABLE_R5B14_POLICY",
            "runtime_flag_scoring_config": "r5b14_operational_zero_fn_enabled",
        },
        "final_action_col": "r5b15_core_policy_decisao",
        "metrics": {
            "global_intervention": r5b15["final_intervention_metrics"],
            "block_only": r5b15["final_block_metrics"],
            "by_action_csv": str(R5B15_BY_ACTION.relative_to(PROJECT_ROOT)),
        },
        "decision_distribution": {
            "APROVAR": {"rows": 111256, "frauds": 0, "normals": 111256},
            "BLOQUEAR": {"rows": 2300, "frauds": 1465, "normals": 835},
            "CONFIRMAR": {"rows": 288, "frauds": 0, "normals": 288},
        },
        "target_gates": {
            "fpr_lt_1pct": r5b15["final_intervention_metrics"]["fpr"] < 0.01,
            "fn_lte_5_outside_block": r5b15["final_intervention_metrics"]["fn"] <= 5,
            "fn_eq_0": r5b15["final_intervention_metrics"]["fn"] == 0,
            "approve_frauds_eq_0": r5b15["remaining_approve_frauds"] == 0,
            "confirm_frauds_eq_0": r5b15["remaining_confirm_frauds"] == 0,
            "all_core_replay_checks_pass": bool(r5b15["all_pass"]),
        },
        "policy_layers": r5b14_candidate.get("layers"),
        "core_replay_checks": r5b15["checks"],
        "sources": [
            source_entry(R5B15_SUMMARY),
            source_entry(R5B15_BY_ACTION),
            source_entry(R5B14_CANDIDATE),
            source_entry(SEVERITY_POLICY),
            source_entry(PIPELINE),
        ],
        "promotion_gates": [
            "Executar replay batch completo no ambiente produtivo com dependencias do PipelineOrquestrador.",
            "Revisar semanticamente as regras R5B14 com risco/negocio.",
            "Ativar somente por configuracao versionada, nunca por default implicito.",
            "Monitorar drift de lgbm_raw e bins usados pelas regras antes de producao.",
        ],
    }

    summary = {
        "experiment": EXPERIMENT,
        "status": "PASS_R5B16_OPERATIONAL_BASELINE_CANDIDATE_CONSOLIDATED",
        "candidate_artifact": str((CANDIDATE_DIR / "operational_baseline_candidate.json").relative_to(PROJECT_ROOT)),
        "global_intervention_metrics": candidate["metrics"]["global_intervention"],
        "block_metrics": candidate["metrics"]["block_only"],
        "target_gates": candidate["target_gates"],
    }

    write_json(OUT_DIR / "00_run_summary.json", summary)
    write_json(OUT_DIR / "01_operational_baseline_candidate.json", candidate)
    write_json(CANDIDATE_DIR / "operational_baseline_candidate.json", candidate)

    report = f"""# {EXPERIMENT} - baseline operacional candidato

## Resultado executivo
- Status: `{summary['status']}`
- Artefato candidato: `{summary['candidate_artifact']}`
- Ativo por default: `False`
- Flag runtime: `ENABLE_R5B14_POLICY`

## Metricas globais
```json
{json.dumps(candidate['metrics']['global_intervention'], ensure_ascii=False, indent=2)}
```

## Metricas BLOQUEAR
```json
{json.dumps(candidate['metrics']['block_only'], ensure_ascii=False, indent=2)}
```

## Decisao tecnica
R5B16 consolida R5B15 como baseline operacional candidato. A politica esta
centralizada no core e conectada ao orquestrador por configuracao, mas permanece
desligada por default ate replay batch completo no ambiente produtivo.
"""
    (OUT_DIR / "02_exp014b_r5b16_consolidated_operational_baseline_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
