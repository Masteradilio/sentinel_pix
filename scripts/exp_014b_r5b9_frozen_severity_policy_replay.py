#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R5B9 — Frozen Severity Policy Replay.

Reexecuta a política R5B8 traduzida para código de core e compara o resultado
linha a linha com o artefato experimental R5B8. Este replay é a ponte entre
mineração offline e integração controlada da política de severidade.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.severity_policy import apply_r5b8_block_deescalation, r5b8_policy_metadata
from exp_014b_r5b2_tune_policy import LABELS, find_col, ints, metrics, pred_block, pred_intervention


EXPERIMENT = "EXP-014B-R5B9-FROZEN-SEVERITY-POLICY-REPLAY"
SOURCE_R5B5 = "EXP-014B-R5B5-TRUST-FEATURE-DEESCALATION"
SOURCE_R5B8 = "EXP-014B-R5B8-BROAD-RESIDUAL-RULE-MINING"

INPUT_R5B5 = PROJECT_ROOT / "resultados" / "experimentos" / SOURCE_R5B5 / "05_predictions_trust.csv"
REFERENCE_R5B8 = PROJECT_ROOT / "resultados" / "experimentos" / SOURCE_R5B8 / "04_predictions_broad_rules.csv"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / EXPERIMENT

BASE_ACTION_COL = "r5b5_trust_decisao"
REFERENCE_ACTION_COL = "r5b8_broad_rules_decisao"
REFERENCE_MOVE_COL = "exp014b_r5b8_broad_rules_block_to_confirm"
FINAL_ACTION_COL = "r5b9_frozen_severity_decisao"
MOVE_COL = "exp014b_r5b9_frozen_severity_block_to_confirm"


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_action(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().str.strip()


def by_action(df: pd.DataFrame, action_col: str, label_col: str) -> pd.DataFrame:
    out = df.groupby(action_col).agg(n_rows=(label_col, "size"), n_frauds=(label_col, "sum")).reset_index()
    out["n_normals"] = out["n_rows"] - out["n_frauds"]
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_R5B5.exists():
        raise FileNotFoundError(INPUT_R5B5)
    if not REFERENCE_R5B8.exists():
        raise FileNotFoundError(REFERENCE_R5B8)

    df = pd.read_csv(INPUT_R5B5, low_memory=False)
    reference = pd.read_csv(
        REFERENCE_R5B8,
        usecols=["transaction_id", REFERENCE_ACTION_COL, REFERENCE_MOVE_COL],
        low_memory=False,
    )
    label_col = find_col(df, LABELS)
    y = ints(df[label_col])

    final_action, trace = apply_r5b8_block_deescalation(df, df[BASE_ACTION_COL])
    replay = df.copy()
    replay[MOVE_COL] = trace["r5b8_any_rule_applied"].astype(int)
    replay["exp014b_r5b9_frozen_severity_rule"] = trace["r5b8_rule_applied"]
    replay[FINAL_ACTION_COL] = final_action
    replay["exp014b_r5b9_intervention_pred"] = pred_intervention(final_action)
    replay["exp014b_r5b9_block_pred"] = pred_block(final_action)

    merged = replay.merge(reference, on="transaction_id", how="left", validate="one_to_one")
    action_mismatch = normalize_action(merged[FINAL_ACTION_COL]) != normalize_action(merged[REFERENCE_ACTION_COL])
    move_mismatch = merged[MOVE_COL].astype(int) != pd.to_numeric(merged[REFERENCE_MOVE_COL], errors="coerce").fillna(-1).astype(int)
    mismatch_cols = [
        "transaction_id",
        label_col,
        "temporal_split",
        BASE_ACTION_COL,
        FINAL_ACTION_COL,
        REFERENCE_ACTION_COL,
        MOVE_COL,
        REFERENCE_MOVE_COL,
        "exp014b_r5b9_frozen_severity_rule",
    ]
    mismatches = merged.loc[action_mismatch | move_mismatch, mismatch_cols].copy()

    move_mask = replay[MOVE_COL].astype(bool).to_numpy()
    y_np = y.to_numpy()
    final_block_metrics = metrics(replay[label_col], replay["exp014b_r5b9_block_pred"])
    summary = {
        "experiment": EXPERIMENT,
        "source_r5b5": SOURCE_R5B5,
        "source_r5b8": SOURCE_R5B8,
        "status": "PASS_R5B9_FROZEN_REPLAY_MATCHED_R5B8" if len(mismatches) == 0 else "FAIL_R5B9_REPLAY_MISMATCH",
        "all_pass": bool(len(mismatches) == 0),
        "policy": r5b8_policy_metadata(),
        "n_rows": int(len(replay)),
        "action_mismatches_vs_r5b8": int(action_mismatch.sum()),
        "move_mismatches_vs_r5b8": int(move_mismatch.sum()),
        "total_mismatches_vs_r5b8": int(len(mismatches)),
        "block_fp_demoted_to_confirm": int((move_mask & (y_np == 0)).sum()),
        "block_tp_demoted_to_confirm": int((move_mask & (y_np == 1)).sum()),
        "remaining_block_normals": int(((final_action == "BLOQUEAR") & (y_np == 0)).sum()),
        "remaining_block_frauds": int(((final_action == "BLOQUEAR") & (y_np == 1)).sum()),
        "remaining_approve_frauds": int(((final_action == "APROVAR") & (y_np == 1)).sum()),
        "base_block_metrics": metrics(replay[label_col], pred_block(replay[BASE_ACTION_COL])),
        "final_block_metrics": final_block_metrics,
        "final_intervention_metrics": metrics(replay[label_col], replay["exp014b_r5b9_intervention_pred"]),
        "rule_counts": trace["r5b8_rule_applied"].replace("", pd.NA).dropna().value_counts().to_dict(),
    }

    write_json(OUT_DIR / "00_run_summary.json", summary)
    write_json(OUT_DIR / "01_policy_metadata.json", r5b8_policy_metadata())
    by_action(replay, FINAL_ACTION_COL, label_col).to_csv(OUT_DIR / "02_metrics_by_action.csv", index=False)
    trace.to_csv(OUT_DIR / "03_policy_trace.csv", index=False)
    mismatches.to_csv(OUT_DIR / "04_mismatches_vs_r5b8.csv", index=False)
    replay.to_csv(OUT_DIR / "05_predictions_frozen_severity.csv", index=False)

    report = f"""# {EXPERIMENT} — Replay congelado da política de severidade

## Resultado executivo
- Status: `{summary['status']}`
- Linhas avaliadas: `{summary['n_rows']}`
- Divergências de decisão vs R5B8: `{summary['action_mismatches_vs_r5b8']}`
- Divergências de aplicação vs R5B8: `{summary['move_mismatches_vs_r5b8']}`
- Normais movidos de BLOQUEAR para CONFIRMAR: `{summary['block_fp_demoted_to_confirm']}`
- Fraudes movidas de BLOQUEAR para CONFIRMAR: `{summary['block_tp_demoted_to_confirm']}`
- Normais restantes em BLOQUEAR: `{summary['remaining_block_normals']}`
- Fraudes restantes em BLOQUEAR: `{summary['remaining_block_frauds']}`
- Fraudes restantes em APROVAR: `{summary['remaining_approve_frauds']}`

## Métricas finais de BLOQUEAR
```json
{json.dumps(final_block_metrics, ensure_ascii=False, indent=2)}
```

## Contagem por regra
```json
{json.dumps(summary['rule_counts'], ensure_ascii=False, indent=2)}
```

## Decisão técnica
Este replay valida que a implementação explícita em `backend.core.severity_policy`
reproduz o artefato R5B8. Ele ainda não ativa a política automaticamente no
runtime produtivo; serve como gate antes de conectar a política ao orquestrador
ou a um arquivo de configuração versionado.
"""
    (OUT_DIR / "06_exp014b_r5b9_frozen_severity_policy_replay_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
