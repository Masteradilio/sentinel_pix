# -*- coding: utf-8 -*-
"""
EXP-014B-R4C-FROZEN — Frozen validation do R4C.

Reaplica o artifact recomendado do R4C sobre o baseline R4A-FROZEN e valida
se o replay reproduz as métricas do R4C.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


EXPERIMENT = "EXP-014B-R4C-FROZEN"
LABELS = ["is_fraud", "fraude", "target", "label", "tp_fraude"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-predictions", default=None)
    p.add_argument("--artifact", default=None)
    p.add_argument("--r4c-predictions", default=None)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def defaults() -> tuple[Path, Path, Path, Path]:
    root = Path.cwd()
    return (
        root / "resultados" / "experimentos" / "EXP-014B-R4A-FROZEN" / "06_predictions_frozen.csv",
        root / "resultados" / "experimentos" / "EXP-014B-R4C" / "08_policy_artifact_recommended.json",
        root / "resultados" / "experimentos" / "EXP-014B-R4C" / "09_predictions_recommended.csv",
        root / "resultados" / "experimentos" / EXPERIMENT,
    )


def find_col(df: pd.DataFrame, names: list[str]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        if n.lower() in lower:
            return lower[n.lower()]
    raise KeyError(f"Coluna não encontrada entre: {names}")


def ints(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def norm_action(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def pred_intervention(action: pd.Series) -> pd.Series:
    return norm_action(action).isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)


def pred_block(action: pd.Series) -> pd.Series:
    return norm_action(action).eq("BLOQUEAR").astype(int)


def metrics(y_true: pd.Series, pred: pd.Series) -> dict[str, Any]:
    y = ints(y_true)
    p = ints(pred)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
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


def mask_from_description(df: pd.DataFrame, desc: str) -> pd.Series:
    prefix = "Demover BLOQUEAR R4C com "
    if not desc.startswith(prefix):
        raise ValueError(f"Descrição não suportada: {desc}")
    expr = desc[len(prefix):]
    mask = pd.Series(True, index=df.index)
    for part in expr.split(" AND "):
        if " == " in part:
            col, val = part.split(" == ", 1)
            mask &= df[col].fillna("<MISSING>").astype(str).eq(str(val))
        elif " <= " in part:
            col, val = part.split(" <= ", 1)
            mask &= pd.to_numeric(df[col], errors="coerce").le(float(val))
        elif " >= " in part:
            col, val = part.split(" >= ", 1)
            mask &= pd.to_numeric(df[col], errors="coerce").ge(float(val))
        else:
            raise ValueError(f"Parte não suportada: {part}")
    return mask


def action_metrics(df: pd.DataFrame, label_col: str, action_col: str) -> pd.DataFrame:
    y = ints(df[label_col])
    rows = []
    for action, idx in df.groupby(action_col, dropna=False).groups.items():
        idx = list(idx)
        yy = y.loc[idx]
        n = len(idx)
        frauds = int((yy == 1).sum())
        normals = int((yy == 0).sum())
        rows.append({
            "action": str(action),
            "n_rows": int(n),
            "n_frauds": frauds,
            "n_normals": normals,
            "precision_within_action": round(float(frauds / n), 8) if n else 0.0,
        })
    return pd.DataFrame(rows).sort_values("action")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    a = parse_args()
    base_default, artifact_default, ref_default, out_default = defaults()

    artifact_path = Path(a.artifact) if a.artifact else artifact_default
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact R4C não encontrado: {artifact_path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    base_path = Path(a.base_predictions) if a.base_predictions else Path(artifact.get("input_predictions_path") or base_default)
    if not base_path.exists():
        base_path = base_default
    if not base_path.exists():
        raise FileNotFoundError(f"Base predictions não encontrado: {base_path}")

    ref_path = Path(a.r4c_predictions) if a.r4c_predictions else ref_default
    out = Path(a.output_dir) if a.output_dir else out_default
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(base_path, low_memory=False).copy()
    label_col = find_col(df, LABELS)
    base_action_col = artifact["base_action_col"]

    action = norm_action(df[base_action_col])
    block_mask = action.eq("BLOQUEAR")
    demote = pd.Series(False, index=df.index)

    for rule in artifact.get("selected_demotions", []):
        demote |= block_mask & mask_from_description(df, str(rule["description"]))

    final_action = action.copy()
    final_action[demote] = "APROVAR"

    df["exp014b_r4c_frozen_demote_block_to_approve"] = demote.astype(int)
    df["r4c_frozen_decisao_recommended"] = final_action
    df["exp014b_r4c_frozen_intervention_pred"] = pred_intervention(final_action)
    df["exp014b_r4c_frozen_block_pred"] = pred_block(final_action)

    frozen_intervention = metrics(df[label_col], df["exp014b_r4c_frozen_intervention_pred"])
    frozen_block = metrics(df[label_col], df["exp014b_r4c_frozen_block_pred"])

    intervention_ok = frozen_intervention == artifact["final_intervention_metrics"]
    block_ok = frozen_block == artifact["final_block_metrics"]

    n_mismatch = 0
    mismatch_df = pd.DataFrame()
    if ref_path.exists():
        ref = pd.read_csv(ref_path, low_memory=False)
        mismatch = pd.Series(False, index=df.index)
        src_action = artifact.get("final_action_col", "r4c_decisao_recommended")
        src_inter = artifact.get("intervention_pred_col", "exp014b_r4c_intervention_pred")
        src_block = artifact.get("block_pred_col", "exp014b_r4c_block_pred")
        if src_action in ref.columns:
            mismatch |= norm_action(ref[src_action]).ne(norm_action(df["r4c_frozen_decisao_recommended"]))
        if src_inter in ref.columns:
            mismatch |= ints(ref[src_inter]).ne(ints(df["exp014b_r4c_frozen_intervention_pred"]))
        if src_block in ref.columns:
            mismatch |= ints(ref[src_block]).ne(ints(df["exp014b_r4c_frozen_block_pred"]))
        n_mismatch = int(mismatch.sum())
        cols = [c for c in [label_col, base_action_col, "r4c_frozen_decisao_recommended", "lgbm_r4_score", "score_final"] if c in df.columns]
        mismatch_df = df.loc[mismatch, cols].copy()

    status = "PASS_R4C_FROZEN_VALIDATED_REPLAY_OK" if intervention_ok and block_ok and n_mismatch == 0 else "FAIL_R4C_FROZEN_VALIDATION_MISMATCH"

    validation = {
        "status": status,
        "base_predictions_path": str(base_path),
        "artifact_path": str(artifact_path),
        "reference_predictions_path": str(ref_path) if ref_path.exists() else None,
        "base_action_col": base_action_col,
        "frozen_action_col": "r4c_frozen_decisao_recommended",
        "frozen_intervention_col": "exp014b_r4c_frozen_intervention_pred",
        "frozen_block_col": "exp014b_r4c_frozen_block_pred",
        "expected_intervention_metrics": artifact["final_intervention_metrics"],
        "frozen_intervention_metrics": frozen_intervention,
        "expected_block_metrics": artifact["final_block_metrics"],
        "frozen_block_metrics": frozen_block,
        "intervention_match_expected": intervention_ok,
        "block_match_expected": block_ok,
        "n_any_mismatches": n_mismatch,
        "all_pass": status.startswith("PASS"),
    }

    by_action = action_metrics(df, label_col, "r4c_frozen_decisao_recommended")

    frozen_artifact = {
        **artifact,
        "experiment": EXPERIMENT,
        "frozen_validation_status": status,
        "frozen_action_col": "r4c_frozen_decisao_recommended",
        "frozen_demote_col": "exp014b_r4c_frozen_demote_block_to_approve",
        "frozen_intervention_pred_col": "exp014b_r4c_frozen_intervention_pred",
        "frozen_block_pred_col": "exp014b_r4c_frozen_block_pred",
        "frozen_intervention_metrics": frozen_intervention,
        "frozen_block_metrics": frozen_block,
        "validation": validation,
    }

    summary = {
        "experiment": EXPERIMENT,
        "status": "DONE",
        "objective_status": status,
        "n_rows": int(len(df)),
        "frozen_intervention_metrics": frozen_intervention,
        "frozen_block_metrics": frozen_block,
        "n_selected_demotions": int(len(artifact.get("selected_demotions", []))),
        "n_any_mismatches": n_mismatch,
        "all_pass": validation["all_pass"],
        "output_dir": str(out),
    }

    contract = {
        "base_predictions_path": str(base_path),
        "artifact_path": str(artifact_path),
        "label_col": label_col,
        "base_action_col": base_action_col,
        "n_selected_demotions": int(len(artifact.get("selected_demotions", []))),
        "contract_ok": True,
        "missing": [],
    }

    write_json(out / "00_run_summary.json", summary)
    write_json(out / "01_input_contract.json", contract)
    write_json(out / "02_frozen_validation.json", validation)
    by_action.to_csv(out / "03_decision_metrics_by_action.csv", index=False, encoding="utf-8")
    mismatch_df.to_csv(out / "04_prediction_mismatches.csv", index=False, encoding="utf-8")
    write_json(out / "05_policy_artifact_frozen.json", frozen_artifact)
    df.to_csv(out / "06_predictions_frozen.csv", index=False, encoding="utf-8")

    try:
        by_action_md = by_action.to_markdown(index=False)
    except Exception:
        by_action_md = by_action.to_string(index=False)

    report = f"""# {EXPERIMENT} - Frozen validation

## Resultado executivo
- Status: `{status}`
- All pass: `{validation["all_pass"]}`
- Prediction mismatches: `{n_mismatch}`
- Regras congeladas: `{len(artifact.get("selected_demotions", []))}`

## Métricas de intervenção congelada
```json
{json.dumps(frozen_intervention, ensure_ascii=False, indent=2)}
```

## Métricas de BLOQUEAR congelado
```json
{json.dumps(frozen_block, ensure_ascii=False, indent=2)}
```

## Métricas por ação
{by_action_md}

## Validação
```json
{json.dumps(validation, ensure_ascii=False, indent=2)}
```

## Decisão sugerida
Se PASS, consolidar R4C-FROZEN como baseline agressivo/operacional.
A próxima rodada deve rebalancear severidade: promover fraudes em CONFIRMAR para BLOQUEAR e mover normais de BLOQUEAR para CONFIRMAR.
"""
    (out / "07_exp014b_r4c_frozen_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
