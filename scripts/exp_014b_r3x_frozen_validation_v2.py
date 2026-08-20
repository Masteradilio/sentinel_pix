# -*- coding: utf-8 -*-
"""
EXP-014B-R3X-FROZEN — Frozen validation of reconstructed decision policy

Objetivo:
  Validar, sem nova mineração, a política operacional reconstruída no R3X:

    exp014b_r3q_frozen_pred=0 -> APROVAR
    exp014b_r3q_frozen_pred=1 e block_policy=true -> BLOQUEAR
    exp014b_r3q_frozen_pred=1 e block_policy=false -> CONFIRMAR

Entrada padrão:
  resultados/experimentos/EXP-014B-R3X/09_predictions_reconstructed.csv
  resultados/experimentos/EXP-014B-R3X/08_policy_artifact_recommended.json

Saídas:
  resultados/experimentos/EXP-014B-R3X-FROZEN/
    00_run_summary.json
    01_input_contract.json
    02_frozen_validation.json
    03_decision_metrics_by_action.csv
    04_prediction_mismatches.csv
    05_policy_artifact_frozen.json
    06_predictions_frozen.csv
    07_exp014b_r3x_frozen_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT = "EXP-014B-R3X-FROZEN"

LABEL_CANDIDATES = ["is_fraud", "fraude", "target", "label", "tp_fraude"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, default=None)
    parser.add_argument("--artifact", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def default_paths() -> tuple[Path, Path, Path]:
    root = Path.cwd()
    pred = root / "resultados" / "experimentos" / "EXP-014B-R3X" / "09_predictions_reconstructed.csv"
    artifact = root / "resultados" / "experimentos" / "EXP-014B-R3X" / "08_policy_artifact_recommended.json"
    out = root / "resultados" / "experimentos" / EXPERIMENT
    return pred, artifact, out


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(f"Nenhuma coluna encontrada entre: {candidates}")
    return None


def safe_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def metrics(y_true: pd.Series, pred: pd.Series) -> dict[str, Any]:
    y = safe_int_series(y_true)
    p = safe_int_series(pred)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
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


def action_to_intervention(action: pd.Series) -> pd.Series:
    return action.astype(str).str.upper().isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)


def action_to_block(action: pd.Series) -> pd.Series:
    return action.astype(str).str.upper().eq("BLOQUEAR").astype(int)


def parse_policy(policy: str, prefix: str) -> tuple[str, float]:
    body = policy[len(prefix):]
    score_col, th_str = body.rsplit("_", 1)
    return score_col, float(th_str)


def apply_block_policy(df: pd.DataFrame, base_col: str, policy_name: str) -> pd.Series:
    base_alert = safe_int_series(df[base_col]).eq(1)

    if policy_name == "no_block_all_r3q_alerts_confirmar":
        return pd.Series(False, index=df.index)

    if policy_name == "block_all_r3q_alerts":
        return base_alert

    if policy_name.startswith("block_hi_"):
        score_col, th = parse_policy(policy_name, "block_hi_")
        if score_col not in df.columns:
            raise KeyError(f"Score col ausente para policy {policy_name}: {score_col}")
        return base_alert & pd.to_numeric(df[score_col], errors="coerce").ge(th)

    if policy_name.startswith("block_lo_"):
        score_col, th = parse_policy(policy_name, "block_lo_")
        if score_col not in df.columns:
            raise KeyError(f"Score col ausente para policy {policy_name}: {score_col}")
        return base_alert & pd.to_numeric(df[score_col], errors="coerce").le(th)

    raise ValueError(f"Policy não suportada: {policy_name}")


def build_action(df: pd.DataFrame, base_col: str, block_mask: pd.Series) -> pd.Series:
    base_alert = safe_int_series(df[base_col]).eq(1)
    block = block_mask.fillna(False)
    return pd.Series(
        np.select(
            [~base_alert, base_alert & block, base_alert & ~block],
            ["APROVAR", "BLOQUEAR", "CONFIRMAR"],
            default="APROVAR",
        ),
        index=df.index,
    )


def metrics_by_action(df: pd.DataFrame, label_col: str, action_col: str) -> pd.DataFrame:
    rows = []
    y = safe_int_series(df[label_col])
    for action, idx in df.groupby(action_col, dropna=False).groups.items():
        idx = list(idx)
        yy = y.loc[idx]
        n = int(len(idx))
        frauds = int((yy == 1).sum())
        normals = int((yy == 0).sum())
        rows.append({
            "action": str(action),
            "n_rows": n,
            "n_frauds": frauds,
            "n_normals": normals,
            "precision_within_action": round(float(frauds / n), 8) if n else 0.0,
        })
    return pd.DataFrame(rows).sort_values("action")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    default_pred, default_artifact, default_out = default_paths()
    pred_path = Path(args.predictions) if args.predictions else default_pred
    artifact_path = Path(args.artifact) if args.artifact else default_artifact
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions não encontrado: {pred_path}")

    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact não encontrado: {artifact_path}")

    df = pd.read_csv(pred_path, low_memory=False)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    label_col = find_col(df, LABEL_CANDIDATES)
    base_col = artifact["base_col"]
    final_action_col_source = artifact.get("final_action_col", "r3x_decisao_pos_policy")
    intervention_source = artifact.get("intervention_pred_col", "exp014b_r3x_intervention_pred")
    block_source = artifact.get("block_pred_col", "exp014b_r3x_block_pred")
    policy_name = artifact["selected_block_policy"]["policy_name"]

    missing = []
    for c in [label_col, base_col]:
        if c is None or c not in df.columns:
            missing.append(c)

    if missing:
        raise KeyError(f"Colunas obrigatórias ausentes: {missing}")

    block_mask = apply_block_policy(df, base_col, policy_name)

    df["r3x_frozen_decisao_pos_policy"] = build_action(df, base_col, block_mask)
    df["exp014b_r3x_frozen_intervention_pred"] = action_to_intervention(df["r3x_frozen_decisao_pos_policy"])
    df["exp014b_r3x_frozen_block_pred"] = action_to_block(df["r3x_frozen_decisao_pos_policy"])

    source_action_available = final_action_col_source in df.columns
    source_intervention_available = intervention_source in df.columns
    source_block_available = block_source in df.columns

    action_mismatches = pd.Series(False, index=df.index)
    intervention_mismatches = pd.Series(False, index=df.index)
    block_mismatches = pd.Series(False, index=df.index)

    if source_action_available:
        action_mismatches = (
            df[final_action_col_source].astype(str).str.upper()
            != df["r3x_frozen_decisao_pos_policy"].astype(str).str.upper()
        )

    if source_intervention_available:
        intervention_mismatches = (
            safe_int_series(df[intervention_source])
            != safe_int_series(df["exp014b_r3x_frozen_intervention_pred"])
        )

    if source_block_available:
        block_mismatches = (
            safe_int_series(df[block_source])
            != safe_int_series(df["exp014b_r3x_frozen_block_pred"])
        )

    any_mismatch = action_mismatches | intervention_mismatches | block_mismatches

    base_metrics = metrics(df[label_col], df[base_col])
    frozen_intervention_metrics = metrics(df[label_col], df["exp014b_r3x_frozen_intervention_pred"])
    frozen_block_metrics = metrics(df[label_col], df["exp014b_r3x_frozen_block_pred"])

    expected_intervention = artifact["aligned_intervention_metrics"]
    expected_block = artifact["aligned_block_metrics"]

    intervention_match_expected = frozen_intervention_metrics == expected_intervention
    block_match_expected = frozen_block_metrics == expected_block
    no_prediction_mismatches = int(any_mismatch.sum()) == 0

    validation = {
        "status": (
            "PASS_R3X_FROZEN_VALIDATED_ALIGNMENT_REPLAY_OK"
            if intervention_match_expected and block_match_expected and no_prediction_mismatches
            else "FAIL_R3X_FROZEN_VALIDATION_MISMATCH"
        ),
        "policy_name": policy_name,
        "base_col": base_col,
        "source_action_col": final_action_col_source,
        "frozen_action_col": "r3x_frozen_decisao_pos_policy",
        "source_intervention_col": intervention_source,
        "frozen_intervention_col": "exp014b_r3x_frozen_intervention_pred",
        "source_block_col": block_source,
        "frozen_block_col": "exp014b_r3x_frozen_block_pred",
        "base_metrics": base_metrics,
        "expected_intervention_metrics": expected_intervention,
        "frozen_intervention_metrics": frozen_intervention_metrics,
        "expected_block_metrics": expected_block,
        "frozen_block_metrics": frozen_block_metrics,
        "intervention_match_expected": intervention_match_expected,
        "block_match_expected": block_match_expected,
        "n_action_mismatches": int(action_mismatches.sum()) if source_action_available else None,
        "n_intervention_mismatches": int(intervention_mismatches.sum()) if source_intervention_available else None,
        "n_block_mismatches": int(block_mismatches.sum()) if source_block_available else None,
        "n_any_mismatches": int(any_mismatch.sum()),
        "no_prediction_mismatches": no_prediction_mismatches,
        "all_pass": bool(intervention_match_expected and block_match_expected and no_prediction_mismatches),
    }

    by_action = metrics_by_action(df, label_col, "r3x_frozen_decisao_pos_policy")

    mismatch_cols = [
        c for c in [
            "transaction_id", "cd_pix", label_col, base_col,
            final_action_col_source, "r3x_frozen_decisao_pos_policy",
            intervention_source, "exp014b_r3x_frozen_intervention_pred",
            block_source, "exp014b_r3x_frozen_block_pred",
            "score_final", "lgbm_r4_score", "lgbm_raw", "lgbm_mapped",
        ]
        if c in df.columns
    ]
    df.loc[any_mismatch, mismatch_cols].to_csv(
        out_dir / "04_prediction_mismatches.csv",
        index=False,
        encoding="utf-8",
    )

    frozen_artifact = {
        **artifact,
        "experiment": EXPERIMENT,
        "frozen_validation_status": validation["status"],
        "frozen_action_col": "r3x_frozen_decisao_pos_policy",
        "frozen_intervention_pred_col": "exp014b_r3x_frozen_intervention_pred",
        "frozen_block_pred_col": "exp014b_r3x_frozen_block_pred",
        "frozen_intervention_metrics": frozen_intervention_metrics,
        "frozen_block_metrics": frozen_block_metrics,
        "validation": validation,
    }

    summary = {
        "experiment": EXPERIMENT,
        "status": "DONE",
        "objective_status": validation["status"],
        "n_rows": int(len(df)),
        "n_frauds": int(safe_int_series(df[label_col]).sum()),
        "n_normals": int((safe_int_series(df[label_col]) == 0).sum()),
        "predictions_path": str(pred_path),
        "artifact_path": str(artifact_path),
        "policy_name": policy_name,
        "base_col": base_col,
        "frozen_intervention_metrics": frozen_intervention_metrics,
        "frozen_block_metrics": frozen_block_metrics,
        "n_any_mismatches": int(any_mismatch.sum()),
        "all_pass": validation["all_pass"],
        "output_dir": str(out_dir),
    }

    contract = {
        "predictions_path": str(pred_path),
        "artifact_path": str(artifact_path),
        "label_col": label_col,
        "base_col": base_col,
        "policy_name": policy_name,
        "missing": missing,
        "contract_ok": len(missing) == 0,
    }

    write_json(out_dir / "00_run_summary.json", summary)
    write_json(out_dir / "01_input_contract.json", contract)
    write_json(out_dir / "02_frozen_validation.json", validation)
    by_action.to_csv(out_dir / "03_decision_metrics_by_action.csv", index=False, encoding="utf-8")
    write_json(out_dir / "05_policy_artifact_frozen.json", frozen_artifact)
    df.to_csv(out_dir / "06_predictions_frozen.csv", index=False, encoding="utf-8")

    report = f"""# {EXPERIMENT} - Frozen validation

## Resultado executivo
- Status: `{validation["status"]}`
- Policy: `{policy_name}`
- All pass: `{validation["all_pass"]}`
- Prediction mismatches: `{validation["n_any_mismatches"]}`

## Métricas de intervenção congelada
```json
{json.dumps(frozen_intervention_metrics, ensure_ascii=False, indent=2)}
```

## Métricas de BLOQUEAR congelado
```json
{json.dumps(frozen_block_metrics, ensure_ascii=False, indent=2)}
```

## Métricas por ação
{by_action.to_markdown(index=False)}

## Validação
```json
{json.dumps(validation, ensure_ascii=False, indent=2)}
```

## Decisão sugerida
Se PASS, consolidar R3X-FROZEN como baseline operacional alinhado.
A próxima rodada deve melhorar a fila CONFIRMAR, mantendo BLOQUEAR e alinhamento fixos.
"""
    (out_dir / "07_exp014b_r3x_frozen_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
