#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R5B15 - Core policy replay.

Reaplica a politica R5B14 a partir de backend.core.severity_policy e valida
paridade com o artefato experimental R5B14.
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

from backend.core.severity_policy import apply_r5b14_operational_zero_fn_policy, r5b14_policy_metadata


EXPERIMENT = "EXP-014B-R5B15-CORE-POLICY-REPLAY"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / EXPERIMENT

INPUT_FILE = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R4G-FAST-FROZEN" / "06_predictions_frozen.csv"
R5B14_SUMMARY = (
    PROJECT_ROOT
    / "resultados"
    / "experimentos"
    / "EXP-014B-R5B14-OPERATIONAL-ZERO-FN-REPLAY"
    / "00_run_summary.json"
)

LABEL_COL = "is_fraud"
BASE_ACTION_COL = "r4g_fast_frozen_decisao_recommended"
FINAL_ACTION_COL = "r5b15_core_policy_decisao"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ints(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def actions(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().str.strip()


def intervention_pred(action: pd.Series) -> pd.Series:
    return actions(action).isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)


def block_pred(action: pd.Series) -> pd.Series:
    return actions(action).eq("BLOQUEAR").astype(int)


def metrics(y_true: pd.Series, pred: pd.Series) -> dict[str, Any]:
    y = ints(y_true)
    p = ints(pred)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(float(precision), 8),
        "recall": round(float(recall), 8),
        "f1": round(float(f1), 8),
        "fpr": round(float(fpr), 8),
    }


def action_table(df: pd.DataFrame, action_col: str) -> pd.DataFrame:
    out = df.groupby(action_col, dropna=False).agg(n_rows=(LABEL_COL, "size"), n_frauds=(LABEL_COL, "sum")).reset_index()
    out["n_normals"] = out["n_rows"] - out["n_frauds"]
    out["precision_within_action"] = (out["n_frauds"] / out["n_rows"]).round(8)
    return out.sort_values(action_col)


def count_layer(trace: pd.DataFrame, y: pd.Series, layer: str) -> dict[str, int]:
    mask = trace["r5b14_layer_applied"].eq(layer)
    return {
        "rows": int(mask.sum()),
        "frauds": int((mask & (y == 1)).sum()),
        "normals": int((mask & (y == 0)).sum()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in [INPUT_FILE, R5B14_SUMMARY]:
        if not path.exists():
            raise FileNotFoundError(path)

    expected = read_json(R5B14_SUMMARY)
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    y = ints(df[LABEL_COL])

    final_actions, trace = apply_r5b14_operational_zero_fn_policy(df, df[BASE_ACTION_COL])
    df[FINAL_ACTION_COL] = final_actions
    df["exp014b_r5b15_intervention_pred"] = intervention_pred(final_actions)
    df["exp014b_r5b15_block_pred"] = block_pred(final_actions)

    final_intervention = metrics(df[LABEL_COL], df["exp014b_r5b15_intervention_pred"])
    final_block = metrics(df[LABEL_COL], df["exp014b_r5b15_block_pred"])
    by_action = action_table(df, FINAL_ACTION_COL)

    layer_counts = {
        "r5b12_confirm_to_block": count_layer(trace, y, "CONFIRM_TO_BLOCK"),
        "approve_to_block": count_layer(trace, y, "APPROVE_TO_BLOCK"),
        "confirm_to_approve_compensation": count_layer(trace, y, "CONFIRM_TO_APPROVE"),
    }

    checks = {
        "intervention_metrics_match_r5b14": final_intervention == expected["final_intervention_metrics"],
        "block_metrics_match_r5b14": final_block == expected["final_block_metrics"],
        "r5b12_counts_match_r5b14": layer_counts["r5b12_confirm_to_block"] == expected["r5b12_confirm_to_block"],
        "approve_to_block_counts_match_r5b14": layer_counts["approve_to_block"] == expected["approve_to_block"],
        "compensation_counts_match_r5b14": layer_counts["confirm_to_approve_compensation"] == {
            k: expected["confirm_to_approve_compensation"][k] for k in ["rows", "frauds", "normals"]
        },
    }
    all_pass = all(checks.values())

    summary = {
        "experiment": EXPERIMENT,
        "status": "PASS_R5B15_CORE_POLICY_REPLAY_MATCHED_R5B14" if all_pass else "FAIL_R5B15_CORE_POLICY_REPLAY",
        "input_file": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
        "policy_metadata": r5b14_policy_metadata(),
        "checks": checks,
        "layer_counts": layer_counts,
        "final_intervention_metrics": final_intervention,
        "final_block_metrics": final_block,
        "remaining_approve_frauds": int((final_actions.eq("APROVAR") & (y == 1)).sum()),
        "remaining_confirm_frauds": int((final_actions.eq("CONFIRMAR") & (y == 1)).sum()),
        "all_pass": all_pass,
    }

    write_json(OUT_DIR / "00_run_summary.json", summary)
    by_action.to_csv(OUT_DIR / "01_metrics_by_action.csv", index=False)
    trace.to_csv(OUT_DIR / "02_policy_trace.csv", index=False)

    report = f"""# {EXPERIMENT} - Core policy replay

## Resultado executivo
- Status: `{summary['status']}`
- All pass: `{all_pass}`
- Fraudes restantes em APROVAR: `{summary['remaining_approve_frauds']}`
- Fraudes restantes em CONFIRMAR: `{summary['remaining_confirm_frauds']}`

## Checks
```json
{json.dumps(checks, ensure_ascii=False, indent=2)}
```

## Intervencao global
```json
{json.dumps(final_intervention, ensure_ascii=False, indent=2)}
```

## BLOQUEAR
```json
{json.dumps(final_block, ensure_ascii=False, indent=2)}
```
"""
    (OUT_DIR / "03_exp014b_r5b15_core_policy_replay_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
