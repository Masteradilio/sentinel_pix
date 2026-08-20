# -*- coding: utf-8 -*-
"""
EXP-014B-R3Y-FROZEN - Frozen validation of Confirm Queue Reduction.

This script replays the R3Y selected demotions on top of the R3X-FROZEN
baseline and validates that the replay reproduces the R3Y recommended artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT = "EXP-014B-R3Y-FROZEN"
LABEL_CANDIDATES = ["is_fraud", "fraude", "target", "label", "tp_fraude"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-predictions", type=str, default=None)
    p.add_argument("--artifact", type=str, default=None)
    p.add_argument("--r3y-predictions", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def default_paths() -> tuple[Path, Path, Path, Path]:
    root = Path.cwd()
    artifact = root / "resultados" / "experimentos" / "EXP-014B-R3Y" / "08_policy_artifact_recommended.json"
    base_pred = root / "resultados" / "experimentos" / "EXP-014B-R3X-FROZEN" / "06_predictions_frozen.csv"
    r3y_pred = root / "resultados" / "experimentos" / "EXP-014B-R3Y" / "09_predictions_recommended.csv"
    out = root / "resultados" / "experimentos" / EXPERIMENT
    return base_pred, artifact, r3y_pred, out


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(f"No column found among: {candidates}")


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


def normalize_action(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def action_to_intervention(action: pd.Series) -> pd.Series:
    return action.astype(str).str.upper().isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)


def action_to_block(action: pd.Series) -> pd.Series:
    return action.astype(str).str.upper().eq("BLOQUEAR").astype(int)


def mask_from_description(df: pd.DataFrame, description: str) -> pd.Series:
    prefix = "Demover CONFIRMAR com "
    if not description.startswith(prefix):
        raise ValueError(f"Unsupported description: {description}")
    expr = description[len(prefix):]
    if " <= " in expr and " AND " not in expr:
        col, th = expr.split(" <= ", 1)
        return pd.to_numeric(df[col], errors="coerce").le(float(th))
    if " >= " in expr and " AND " not in expr:
        col, th = expr.split(" >= ", 1)
        return pd.to_numeric(df[col], errors="coerce").ge(float(th))
    if " AND " in expr:
        mask = pd.Series(True, index=df.index)
        for part in expr.split(" AND "):
            col, val = part.split(" == ", 1)
            mask &= df[col].fillna("<MISSING>").astype(str).eq(str(val))
        return mask
    if " == " in expr:
        col, val = expr.split(" == ", 1)
        return df[col].fillna("<MISSING>").astype(str).eq(str(val))
    raise ValueError(f"Unsupported expression: {expr}")


def apply_demotions(df: pd.DataFrame, base_action_col: str, selected: list[dict[str, Any]]) -> pd.Series:
    action = normalize_action(df[base_action_col])
    confirm = action.eq("CONFIRMAR")
    demote = pd.Series(False, index=df.index)
    for rule in selected:
        desc = str(rule["description"])
        demote |= confirm & mask_from_description(df, desc)
    return demote


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
    default_base, default_artifact, default_r3y_pred, default_out = default_paths()
    artifact_path = Path(args.artifact) if args.artifact else default_artifact
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    base_path = Path(args.base_predictions) if args.base_predictions else Path(artifact.get("input_predictions_path") or default_base)
    if not base_path.exists():
        base_path = default_base
    if not base_path.exists():
        raise FileNotFoundError(f"Base predictions not found: {base_path}")

    ref_path = Path(args.r3y_predictions) if args.r3y_predictions else default_r3y_pred
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(base_path, low_memory=False)
    label_col = find_col(df, LABEL_CANDIDATES)
    base_action_col = artifact["base_action_col"]
    selected = artifact.get("selected_demotions", [])
    if base_action_col not in df.columns:
        raise KeyError(f"Base action column not found: {base_action_col}")

    demote = apply_demotions(df, base_action_col, selected)
    base_action = normalize_action(df[base_action_col])
    final_action = base_action.copy()
    final_action[base_action.eq("CONFIRMAR") & demote] = "APROVAR"

    df["exp014b_r3y_frozen_demote_confirm_to_approve"] = demote.astype(int)
    df["r3y_frozen_decisao_recommended"] = final_action
    df["exp014b_r3y_frozen_intervention_pred"] = action_to_intervention(final_action)
    df["exp014b_r3y_frozen_block_pred"] = action_to_block(final_action)

    y = safe_int_series(df[label_col])
    frozen_intervention = metrics(y, df["exp014b_r3y_frozen_intervention_pred"])
    frozen_block = metrics(y, df["exp014b_r3y_frozen_block_pred"])
    expected_intervention = artifact["final_intervention_metrics"]
    expected_block = artifact["final_block_metrics"]

    intervention_match = frozen_intervention == expected_intervention
    block_match = frozen_block == expected_block

    n_any_mismatches = 0
    mismatch_df = pd.DataFrame()
    ref_available = ref_path.exists()
    n_action_mismatches = None
    n_intervention_mismatches = None
    n_block_mismatches = None
    if ref_available:
        ref = pd.read_csv(ref_path, low_memory=False)
        if len(ref) != len(df):
            raise ValueError(f"Reference rows={len(ref)} but base rows={len(df)}")
        src_action = artifact.get("final_action_col", "r3y_decisao_recommended")
        src_intervention = artifact.get("intervention_pred_col", "exp014b_r3y_intervention_pred")
        src_block = artifact.get("block_pred_col", "exp014b_r3y_block_pred")
        action_m = pd.Series(False, index=df.index)
        intervention_m = pd.Series(False, index=df.index)
        block_m = pd.Series(False, index=df.index)
        if src_action in ref.columns:
            action_m = normalize_action(ref[src_action]) != normalize_action(df["r3y_frozen_decisao_recommended"])
        if src_intervention in ref.columns:
            intervention_m = safe_int_series(ref[src_intervention]) != safe_int_series(df["exp014b_r3y_frozen_intervention_pred"])
        if src_block in ref.columns:
            block_m = safe_int_series(ref[src_block]) != safe_int_series(df["exp014b_r3y_frozen_block_pred"])
        any_m = action_m | intervention_m | block_m
        n_action_mismatches = int(action_m.sum())
        n_intervention_mismatches = int(intervention_m.sum())
        n_block_mismatches = int(block_m.sum())
        n_any_mismatches = int(any_m.sum())
        cols = [c for c in [label_col, base_action_col, "r3y_frozen_decisao_recommended", "exp014b_r3y_frozen_intervention_pred", "exp014b_r3y_frozen_block_pred", "score_final", "lgbm_r4_score"] if c in df.columns]
        mismatch_df = df.loc[any_m, cols].copy()

    validation = {
        "status": "PASS_R3Y_FROZEN_VALIDATED_REPLAY_OK" if intervention_match and block_match and n_any_mismatches == 0 else "FAIL_R3Y_FROZEN_VALIDATION_MISMATCH",
        "base_predictions_path": str(base_path),
        "artifact_path": str(artifact_path),
        "reference_predictions_path": str(ref_path) if ref_available else None,
        "base_action_col": base_action_col,
        "frozen_action_col": "r3y_frozen_decisao_recommended",
        "frozen_intervention_col": "exp014b_r3y_frozen_intervention_pred",
        "frozen_block_col": "exp014b_r3y_frozen_block_pred",
        "expected_intervention_metrics": expected_intervention,
        "frozen_intervention_metrics": frozen_intervention,
        "expected_block_metrics": expected_block,
        "frozen_block_metrics": frozen_block,
        "intervention_match_expected": intervention_match,
        "block_match_expected": block_match,
        "n_action_mismatches": n_action_mismatches,
        "n_intervention_mismatches": n_intervention_mismatches,
        "n_block_mismatches": n_block_mismatches,
        "n_any_mismatches": n_any_mismatches,
        "all_pass": bool(intervention_match and block_match and n_any_mismatches == 0),
    }

    frozen_artifact = {
        **artifact,
        "experiment": EXPERIMENT,
        "frozen_validation_status": validation["status"],
        "frozen_action_col": "r3y_frozen_decisao_recommended",
        "frozen_demote_col": "exp014b_r3y_frozen_demote_confirm_to_approve",
        "frozen_intervention_pred_col": "exp014b_r3y_frozen_intervention_pred",
        "frozen_block_pred_col": "exp014b_r3y_frozen_block_pred",
        "frozen_intervention_metrics": frozen_intervention,
        "frozen_block_metrics": frozen_block,
        "validation": validation,
    }

    by_action = metrics_by_action(df, label_col, "r3y_frozen_decisao_recommended")
    summary = {
        "experiment": EXPERIMENT,
        "status": "DONE",
        "objective_status": validation["status"],
        "n_rows": int(len(df)),
        "n_frauds": int((y == 1).sum()),
        "n_normals": int((y == 0).sum()),
        "base_predictions_path": str(base_path),
        "artifact_path": str(artifact_path),
        "frozen_intervention_metrics": frozen_intervention,
        "frozen_block_metrics": frozen_block,
        "n_selected_demotions": int(len(selected)),
        "n_any_mismatches": n_any_mismatches,
        "all_pass": validation["all_pass"],
        "output_dir": str(out_dir),
    }
    contract = {
        "base_predictions_path": str(base_path),
        "artifact_path": str(artifact_path),
        "label_col": label_col,
        "base_action_col": base_action_col,
        "n_selected_demotions": int(len(selected)),
        "contract_ok": True,
        "missing": [],
    }

    write_json(out_dir / "00_run_summary.json", summary)
    write_json(out_dir / "01_input_contract.json", contract)
    write_json(out_dir / "02_frozen_validation.json", validation)
    by_action.to_csv(out_dir / "03_decision_metrics_by_action.csv", index=False, encoding="utf-8")
    mismatch_df.to_csv(out_dir / "04_prediction_mismatches.csv", index=False, encoding="utf-8")
    write_json(out_dir / "05_policy_artifact_frozen.json", frozen_artifact)
    df.to_csv(out_dir / "06_predictions_frozen.csv", index=False, encoding="utf-8")
    report = f"""# {EXPERIMENT} - Frozen validation\n\n## Resultado executivo\n- Status: `{validation['status']}`\n- All pass: `{validation['all_pass']}`\n- Prediction mismatches: `{validation['n_any_mismatches']}`\n- Regras congeladas: `{len(selected)}`\n\n## Metricas de intervencao congelada\n```json\n{json.dumps(frozen_intervention, ensure_ascii=False, indent=2)}\n```\n\n## Metricas de BLOQUEAR congelado\n```json\n{json.dumps(frozen_block, ensure_ascii=False, indent=2)}\n```\n\n## Metricas por acao\n{by_action.to_markdown(index=False)}\n\n## Validacao\n```json\n{json.dumps(validation, ensure_ascii=False, indent=2)}\n```\n\n## Decisao sugerida\nSe PASS, consolidar R3Y-FROZEN como novo baseline operacional e executar R3Z.\n"""
    (out_dir / "07_exp014b_r3y_frozen_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
