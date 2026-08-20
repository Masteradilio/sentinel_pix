#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R5B10 — Consolidated Severity Policy Candidate.

Consolida as camadas R5B4 + R5B5 + R5B8/R5B9 em um único manifesto candidato
de política de severidade. O script não altera backend/artefatos produtivo.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "EXP-014B-R5B10-CONSOLIDATED-SEVERITY-POLICY"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / EXPERIMENT
CANDIDATE_DIR = PROJECT_ROOT / "backend" / "artefatos_candidatos" / "exp014b_r5b10_severity_policy"

R5B4_ARTIFACT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R5B4-ROBUST-BLOCK-DEESCALATION" / "03_policy_artifact_robust.json"
R5B5_ARTIFACT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R5B5-TRUST-FEATURE-DEESCALATION" / "06_policy_artifact_trust.json"
R5B8_ARTIFACT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R5B8-BROAD-RESIDUAL-RULE-MINING" / "05_policy_artifact_broad_rules.json"
R5B9_SUMMARY = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R5B9-FROZEN-SEVERITY-POLICY-REPLAY" / "00_run_summary.json"


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

    for path in [R5B4_ARTIFACT, R5B5_ARTIFACT, R5B8_ARTIFACT, R5B9_SUMMARY]:
        if not path.exists():
            raise FileNotFoundError(path)

    r5b4 = read_json(R5B4_ARTIFACT)
    r5b5 = read_json(R5B5_ARTIFACT)
    r5b8 = read_json(R5B8_ARTIFACT)
    r5b9 = read_json(R5B9_SUMMARY)

    r5b4_rules = r5b4.get("selected_block_to_confirm_rules", [])
    r5b5_rules = r5b5.get("selected_rules", [])
    r5b8_rules = r5b9.get("policy", {}).get("rules", [])

    final_metrics = r5b9["final_block_metrics"]
    base_r5b2_block_fp = 14220
    final_block_fp = int(r5b9["remaining_block_normals"])
    fp_removed_total = base_r5b2_block_fp - final_block_fp

    candidate = {
        "artifact_type": "severity_policy_candidate",
        "experiment": EXPERIMENT,
        "status": "CANDIDATE_NOT_PRODUCTION_ACTIVE",
        "activation": {
            "enabled_by_default": False,
            "reason": (
                "Política consolidada validada offline. Requer conexão controlada "
                "ao orquestrador e replay E2E antes de produção."
            ),
        },
        "scope": {
            "base_action": "BLOQUEAR",
            "target_action": "CONFIRMAR",
            "does_not_promote_approve": True,
            "does_not_overwrite_model_binaries": True,
        },
        "layers": [
            {
                "layer": "R5B4",
                "source_experiment": r5b4.get("experiment"),
                "base_action_col": r5b4.get("base_action_col"),
                "final_action_col": r5b4.get("final_action_col"),
                "move_col": r5b4.get("move_col"),
                "n_rules": len(r5b4_rules),
                "rules": r5b4_rules,
                "run_summary": r5b4.get("run_summary"),
            },
            {
                "layer": "R5B5",
                "source_experiment": r5b5.get("experiment"),
                "base_action_col": r5b5.get("base_action_col"),
                "final_action_col": r5b5.get("final_action_col"),
                "move_col": r5b5.get("move_col"),
                "n_rules": len(r5b5_rules),
                "rules": r5b5_rules,
                "run_summary": r5b5.get("run_summary"),
            },
            {
                "layer": "R5B8_CORE_REPLAYED_AS_R5B9",
                "source_experiment": r5b8.get("experiment"),
                "base_action_col": r5b8.get("base_action_col"),
                "final_action_col": r5b8.get("final_action_col"),
                "move_col": r5b8.get("move_col"),
                "n_rules": len(r5b8_rules),
                "rules": r5b8_rules,
                "frozen_replay_summary": {
                    "experiment": r5b9.get("experiment"),
                    "status": r5b9.get("status"),
                    "action_mismatches_vs_r5b8": r5b9.get("action_mismatches_vs_r5b8"),
                    "move_mismatches_vs_r5b8": r5b9.get("move_mismatches_vs_r5b8"),
                    "rule_counts": r5b9.get("rule_counts"),
                },
            },
        ],
        "metrics": {
            "base_r5b2_block_fp": base_r5b2_block_fp,
            "final_block_fp": final_block_fp,
            "fp_removed_total": fp_removed_total,
            "fp_removed_ratio": round(fp_removed_total / base_r5b2_block_fp, 8),
            "block_tp_demoted_to_confirm": int(r5b9["block_tp_demoted_to_confirm"]),
            "remaining_block_frauds": int(r5b9["remaining_block_frauds"]),
            "remaining_approve_frauds": int(r5b9["remaining_approve_frauds"]),
            "final_block_metrics": final_metrics,
            "final_intervention_metrics": r5b9["final_intervention_metrics"],
        },
        "sources": [
            source_entry(R5B4_ARTIFACT),
            source_entry(R5B5_ARTIFACT),
            source_entry(R5B8_ARTIFACT),
            source_entry(R5B9_SUMMARY),
        ],
        "promotion_gates": [
            "Integrar a política acumulada ao orquestrador por configuração versionada.",
            "Executar replay E2E completo do PipelineOrquestrador na base v3.",
            "Confirmar zero fraude demovida por split e por mês.",
            "Revisar semanticamente regras de alto valor/histórico com área de negócio.",
            "Manter trilha separada para recuperar 682 fraudes ainda em APROVAR.",
        ],
    }

    write_json(OUT_DIR / "00_consolidated_severity_policy_candidate.json", candidate)
    write_json(CANDIDATE_DIR / "severity_policy_candidate.json", candidate)

    summary = {
        "experiment": EXPERIMENT,
        "status": "PASS_R5B10_CONSOLIDATED_CANDIDATE_WRITTEN",
        "candidate_artifact": str((CANDIDATE_DIR / "severity_policy_candidate.json").relative_to(PROJECT_ROOT)),
        "n_layers": len(candidate["layers"]),
        "n_rules_total": len(r5b4_rules) + len(r5b5_rules) + len(r5b8_rules),
        "fp_removed_total": fp_removed_total,
        "final_block_fp": final_block_fp,
        "block_tp_demoted_to_confirm": int(r5b9["block_tp_demoted_to_confirm"]),
        "remaining_block_frauds": int(r5b9["remaining_block_frauds"]),
        "remaining_approve_frauds": int(r5b9["remaining_approve_frauds"]),
        "final_block_metrics": final_metrics,
    }
    write_json(OUT_DIR / "01_run_summary.json", summary)

    report = f"""# {EXPERIMENT} — Política consolidada candidata

## Resultado executivo
- Status: `{summary['status']}`
- Artefato candidato: `{summary['candidate_artifact']}`
- Camadas: `{summary['n_layers']}`
- Regras totais: `{summary['n_rules_total']}`
- Normais removidos de BLOQUEAR desde R5B2: `{summary['fp_removed_total']}`
- Normais restantes em BLOQUEAR: `{summary['final_block_fp']}`
- Fraudes demovidas para CONFIRMAR: `{summary['block_tp_demoted_to_confirm']}`
- Fraudes restantes em BLOQUEAR: `{summary['remaining_block_frauds']}`
- Fraudes restantes em APROVAR: `{summary['remaining_approve_frauds']}`

## Métricas finais de BLOQUEAR
```json
{json.dumps(final_metrics, ensure_ascii=False, indent=2)}
```

## Decisão técnica
O artefato consolidado é candidato e não está ativo em produção. Ele preserva a
proveniência das camadas R5B4, R5B5 e R5B8/R5B9 e serve como contrato de entrada
para o próximo passo: integração configurável ao orquestrador e replay E2E.
"""
    (OUT_DIR / "02_exp014b_r5b10_consolidated_severity_policy_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
