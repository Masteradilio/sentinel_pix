# -*- coding: utf-8 -*-
"""
EXP-014B-R4D-FROZEN — Frozen validation do R4D conservador.

Reaplica o artifact recomendado do R4D sobre o baseline R4C-FROZEN e valida:
  - replay exato das métricas de intervenção;
  - replay exato das métricas de BLOQUEAR;
  - mismatch zero contra predictions recomendadas, se disponíveis.

Saídas:
  resultados/experimentos/EXP-014B-R4D-FROZEN/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


EXPERIMENT = "EXP-014B-R4D-FROZEN"
LABELS = ["is_fraud", "fraude", "target", "label", "tp_fraude"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-predictions", default=None)
    p.add_argument("--artifact", default=None)
    p.add_argument("--r4d-predictions", default=None)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def defaults() -> tuple[Path, Path, Path, Path]:
    root = Path.cwd()
    return (
        root / "resultados" / "experimentos" / "EXP-014B-R4C-FROZEN" / "06_predictions_frozen.csv",
        root / "resultados" / "experimentos" / "EXP-014B-R4D" / "10_policy_artifact_recommended.json",
        root / "resultados" / "experimentos" / "EXP-014B-R4D" / "11_predictions_recommended.csv",
        root / "resultados" / "experimentos" / EXPERIMENT,
    )


def find_col(df: pd.DataFrame, names: list[str]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise KeyError(f"Coluna não encontrada entre: {names}")


def ints(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def actions(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def intervention_from_action(action: pd.Series) -> pd.Series:
    return actions(action).isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)


def block_from_action(action: pd.Series) -> pd.Series:
    return actions(action).eq("BLOQUEAR").astype(int)


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


def action_table(df: pd.DataFrame, label_col: str, action_col: str) -> pd.DataFrame:
    y = ints(df[label_col])
    rows = []
    for action, idx in df.groupby(action_col, dropna=False).groups.items():
        idx = list(idx)
        yy = y.loc[idx]
        n = int(len(idx))
        frauds = int((yy == 1).sum())
        normals = int((yy == 0).sum())
        rows.append(
            {
                "action": str(action),
                "n_rows": n,
                "n_frauds": frauds,
                "n_normals": normals,
                "precision_within_action": round(float(frauds / n), 8) if n else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("action")


def mask_from_description(df: pd.DataFrame, desc: str) -> pd.Series:
    prefixes = [
        "Mover BLOQUEAR para CONFIRMAR R4D com ",
        "Mover CONFIRMAR para BLOQUEAR R4D com ",
    ]
    prefix = None
    for p in prefixes:
        if desc.startswith(p):
            prefix = p
            break
    if prefix is None:
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


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_default, artifact_default, ref_default, out_default = defaults()

    artifact_path = Path(args.artifact) if args.artifact else artifact_default
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact R4D não encontrado: {artifact_path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    base_path = Path(args.base_predictions) if args.base_predictions else Path(artifact.get("input_predictions_path") or base_default)
    if not base_path.exists():
        base_path = base_default
    if not base_path.exists():
        raise FileNotFoundError(f"Base predictions não encontrado: {base_path}")

    ref_path = Path(args.r4d_predictions) if args.r4d_predictions else ref_default
    out = Path(args.output_dir) if args.output_dir else out_default
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(base_path, low_memory=False).copy()
    label_col = find_col(df, LABELS)
    base_action_col = artifact["base_action_col"]

    base_action = actions(df[base_action_col])
    final_action = base_action.copy()

    block_to_confirm = pd.Series(False, index=df.index)
    confirm_to_block = pd.Series(False, index=df.index)

    for rule in artifact.get("selected_block_to_confirm_rules", []):
        block_to_confirm |= base_action.eq("BLOQUEAR") & mask_from_description(df, str(rule["description"]))

    for rule in artifact.get("selected_confirm_to_block_rules", []):
        confirm_to_block |= base_action.eq("CONFIRMAR") & mask_from_description(df, str(rule["description"]))

    final_action.loc[block_to_confirm] = "CONFIRMAR"
    final_action.loc[confirm_to_block] = "BLOQUEAR"

    df["exp014b_r4d_frozen_block_to_confirm"] = block_to_confirm.astype(int)
    df["exp014b_r4d_frozen_confirm_to_block"] = confirm_to_block.astype(int)
    df["r4d_frozen_decisao_recommended"] = final_action
    df["exp014b_r4d_frozen_intervention_pred"] = intervention_from_action(final_action)
    df["exp014b_r4d_frozen_block_pred"] = block_from_action(final_action)

    frozen_intervention = metrics(df[label_col], df["exp014b_r4d_frozen_intervention_pred"])
    frozen_block = metrics(df[label_col], df["exp014b_r4d_frozen_block_pred"])

    intervention_ok = frozen_intervention == artifact["final_intervention_metrics"]
    block_ok = frozen_block == artifact["final_block_metrics"]

    n_mismatch = 0
    mismatch_df = pd.DataFrame()
    if ref_path.exists():
        ref = pd.read_csv(ref_path, low_memory=False)
        mismatch = pd.Series(False, index=df.index)
        src_action = artifact.get("final_action_col", "r4d_decisao_recommended")
        src_intervention = artifact.get("intervention_pred_col", "exp014b_r4d_intervention_pred")
        src_block = artifact.get("block_pred_col", "exp014b_r4d_block_pred")
        if src_action in ref.columns:
            mismatch |= actions(ref[src_action]).ne(actions(df["r4d_frozen_decisao_recommended"]))
        if src_intervention in ref.columns:
            mismatch |= ints(ref[src_intervention]).ne(ints(df["exp014b_r4d_frozen_intervention_pred"]))
        if src_block in ref.columns:
            mismatch |= ints(ref[src_block]).ne(ints(df["exp014b_r4d_frozen_block_pred"]))
        n_mismatch = int(mismatch.sum())
        cols = [c for c in [label_col, base_action_col, "r4d_frozen_decisao_recommended", "lgbm_r4_score", "score_final"] if c in df.columns]
        mismatch_df = df.loc[mismatch, cols].copy()

    status = "PASS_R4D_FROZEN_VALIDATED_REPLAY_OK" if intervention_ok and block_ok and n_mismatch == 0 else "FAIL_R4D_FROZEN_VALIDATION_MISMATCH"

    by_action = action_table(df, label_col, "r4d_frozen_decisao_recommended")

    validation = {
        "status": status,
        "base_predictions_path": str(base_path),
        "artifact_path": str(artifact_path),
        "reference_predictions_path": str(ref_path) if ref_path.exists() else None,
        "base_action_col": base_action_col,
        "frozen_action_col": "r4d_frozen_decisao_recommended",
        "frozen_intervention_col": "exp014b_r4d_frozen_intervention_pred",
        "frozen_block_col": "exp014b_r4d_frozen_block_pred",
        "expected_intervention_metrics": artifact["final_intervention_metrics"],
        "frozen_intervention_metrics": frozen_intervention,
        "expected_block_metrics": artifact["final_block_metrics"],
        "frozen_block_metrics": frozen_block,
        "intervention_match_expected": intervention_ok,
        "block_match_expected": block_ok,
        "n_any_mismatches": n_mismatch,
        "all_pass": status.startswith("PASS"),
    }

    frozen_artifact = {
        **artifact,
        "experiment": EXPERIMENT,
        "frozen_validation_status": status,
        "frozen_action_col": "r4d_frozen_decisao_recommended",
        "frozen_block_to_confirm_col": "exp014b_r4d_frozen_block_to_confirm",
        "frozen_confirm_to_block_col": "exp014b_r4d_frozen_confirm_to_block",
        "frozen_intervention_pred_col": "exp014b_r4d_frozen_intervention_pred",
        "frozen_block_pred_col": "exp014b_r4d_frozen_block_pred",
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
        "n_block_to_confirm": int(block_to_confirm.sum()),
        "n_confirm_to_block": int(confirm_to_block.sum()),
        "n_any_mismatches": n_mismatch,
        "all_pass": validation["all_pass"],
        "output_dir": str(out),
    }

    contract = {
        "base_predictions_path": str(base_path),
        "artifact_path": str(artifact_path),
        "label_col": label_col,
        "base_action_col": base_action_col,
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
- BLOQUEAR -> CONFIRMAR: `{int(block_to_confirm.sum())}`
- CONFIRMAR -> BLOQUEAR: `{int(confirm_to_block.sum())}`

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
Se PASS, consolidar R4D-FROZEN como baseline operacional de severidade.
A próxima rodada deve reduzir FPR global abaixo de 1% e continuar rebalanceando severidade.
"""
    (out / "07_exp014b_r4d_frozen_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
