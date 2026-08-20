# -*- coding: utf-8 -*-
"""
EXP-014B-R3Z-FROZEN — Frozen validation of residual confirm FP reduction.

Reaplica o artifact recomendado do R3Z sobre o baseline R3Y-FROZEN e valida
se as predicoes/metricas reproduzem exatamente a rodada R3Z.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

EXPERIMENT = "EXP-014B-R3Z-FROZEN"
LABEL_CANDIDATES = ["is_fraud", "fraude", "target", "label", "tp_fraude"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-predictions", default=None)
    p.add_argument("--artifact", default=None)
    p.add_argument("--r3z-predictions", default=None)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def default_paths() -> tuple[Path, Path, Path, Path]:
    root = Path.cwd()
    artifact = root / "resultados" / "experimentos" / "EXP-014B-R3Z" / "08_policy_artifact_recommended.json"
    base_pred = root / "resultados" / "experimentos" / "EXP-014B-R3Y-FROZEN" / "06_predictions_frozen.csv"
    r3z_pred = root / "resultados" / "experimentos" / "EXP-014B-R3Z" / "09_predictions_recommended.csv"
    out = root / "resultados" / "experimentos" / EXPERIMENT
    return base_pred, artifact, r3z_pred, out


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(f"Nenhuma coluna encontrada entre: {candidates}")


def safe_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def metrics(y_true: pd.Series, pred: pd.Series) -> dict[str, Any]:
    y = safe_int(y_true)
    p = safe_int(pred)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(float(precision), 8),
        "recall": round(float(recall), 8),
        "f1": round(float(f1), 8),
        "fpr": round(float(fpr), 8),
    }


def action_to_intervention(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)


def action_to_block(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().eq("BLOQUEAR").astype(int)


def mask_from_description(df: pd.DataFrame, description: str) -> pd.Series:
    prefix = "Demover CONFIRMAR residual com "
    if not description.startswith(prefix):
        raise ValueError(f"Descricao nao suportada: {description}")
    expr = description[len(prefix):]
    if " <= " in expr and " AND " not in expr:
        col, th = expr.split(" <= ", 1)
        return pd.to_numeric(df[col], errors="coerce").le(float(th))
    if " >= " in expr and " AND " not in expr:
        col, th = expr.split(" >= ", 1)
        return pd.to_numeric(df[col], errors="coerce").ge(float(th))
    mask = pd.Series(True, index=df.index)
    for part in expr.split(" AND "):
        if " == " not in part:
            raise ValueError(f"Parte da regra nao suportada: {part}")
        col, val = part.split(" == ", 1)
        mask &= df[col].fillna("<MISSING>").astype(str).eq(str(val))
    return mask


def apply_demotions(df: pd.DataFrame, base_action_col: str, selected: list[dict[str, Any]]) -> pd.Series:
    base_action = df[base_action_col].astype(str).str.upper()
    confirm = base_action.eq("CONFIRMAR")
    demote = pd.Series(False, index=df.index)
    for rule in selected:
        demote |= confirm & mask_from_description(df, str(rule["description"]))
    return demote


def final_action(df: pd.DataFrame, base_action_col: str, demote: pd.Series) -> pd.Series:
    out = df[base_action_col].astype(str).str.upper().copy()
    out[out.eq("CONFIRMAR") & demote.fillna(False)] = "APROVAR"
    return out


def by_action(df: pd.DataFrame, label_col: str, action_col: str) -> pd.DataFrame:
    y = safe_int(df[label_col])
    rows = []
    for action, idx in df.groupby(action_col, dropna=False).groups.items():
        idx = list(idx)
        yy = y.loc[idx]
        n = int(len(idx))
        frauds = int((yy == 1).sum())
        normals = int((yy == 0).sum())
        rows.append({
            "action": str(action), "n_rows": n, "n_frauds": frauds,
            "n_normals": normals,
            "precision_within_action": round(float(frauds / n), 8) if n else 0.0,
        })
    return pd.DataFrame(rows).sort_values("action")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    default_base, default_artifact, default_ref, default_out = default_paths()
    artifact_path = Path(args.artifact) if args.artifact else default_artifact
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact R3Z nao encontrado: {artifact_path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    base_path = Path(args.base_predictions) if args.base_predictions else Path(artifact.get("input_predictions_path") or default_base)
    if not base_path.exists():
        base_path = default_base
    if not base_path.exists():
        raise FileNotFoundError(f"Base predictions nao encontrado: {base_path}")

    ref_path = Path(args.r3z_predictions) if args.r3z_predictions else default_ref
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(base_path, low_memory=False)
    label_col = find_col(df, LABEL_CANDIDATES)
    base_action_col = artifact["base_action_col"]
    if base_action_col not in df.columns:
        raise KeyError(f"Coluna base ausente: {base_action_col}")

    demote = apply_demotions(df, base_action_col, artifact.get("selected_demotions", []))
    df["exp014b_r3z_frozen_demote_confirm_to_approve"] = demote.astype(int)
    df["r3z_frozen_decisao_recommended"] = final_action(df, base_action_col, demote)
    df["exp014b_r3z_frozen_intervention_pred"] = action_to_intervention(df["r3z_frozen_decisao_recommended"])
    df["exp014b_r3z_frozen_block_pred"] = action_to_block(df["r3z_frozen_decisao_recommended"])

    y = safe_int(df[label_col])
    frozen_intervention = metrics(y, df["exp014b_r3z_frozen_intervention_pred"])
    frozen_block = metrics(y, df["exp014b_r3z_frozen_block_pred"])
    expected_intervention = artifact["final_intervention_metrics"]
    expected_block = artifact["final_block_metrics"]

    n_any_mismatches = 0
    mismatch_df = pd.DataFrame()
    if ref_path.exists():
        ref = pd.read_csv(ref_path, low_memory=False)
        if len(ref) != len(df):
            raise ValueError(f"Referencia R3Z tem {len(ref)} linhas, base tem {len(df)}")
        act_col = artifact.get("final_action_col", "r3z_decisao_recommended")
        int_col = artifact.get("intervention_pred_col", "exp014b_r3z_intervention_pred")
        blk_col = artifact.get("block_pred_col", "exp014b_r3z_block_pred")
        action_mis = ref[act_col].astype(str).str.upper().ne(df["r3z_frozen_decisao_recommended"].astype(str).str.upper()) if act_col in ref.columns else pd.Series(False, index=df.index)
        int_mis = safe_int(ref[int_col]).ne(safe_int(df["exp014b_r3z_frozen_intervention_pred"])) if int_col in ref.columns else pd.Series(False, index=df.index)
        blk_mis = safe_int(ref[blk_col]).ne(safe_int(df["exp014b_r3z_frozen_block_pred"])) if blk_col in ref.columns else pd.Series(False, index=df.index)
        any_mis = action_mis | int_mis | blk_mis
        n_any_mismatches = int(any_mis.sum())
        cols = [c for c in [label_col, base_action_col, "r3z_frozen_decisao_recommended", "exp014b_r3z_frozen_intervention_pred", "exp014b_r3z_frozen_block_pred", "score_final", "lgbm_r4_score"] if c in df.columns]
        mismatch_df = df.loc[any_mis, cols].copy()

    validation = {
        "status": "PASS_R3Z_FROZEN_VALIDATED_REPLAY_OK" if frozen_intervention == expected_intervention and frozen_block == expected_block and n_any_mismatches == 0 else "FAIL_R3Z_FROZEN_VALIDATION_MISMATCH",
        "base_predictions_path": str(base_path),
        "artifact_path": str(artifact_path),
        "r3z_reference_predictions_path": str(ref_path) if ref_path.exists() else None,
        "base_action_col": base_action_col,
        "frozen_action_col": "r3z_frozen_decisao_recommended",
        "frozen_intervention_col": "exp014b_r3z_frozen_intervention_pred",
        "frozen_block_col": "exp014b_r3z_frozen_block_pred",
        "expected_intervention_metrics": expected_intervention,
        "frozen_intervention_metrics": frozen_intervention,
        "expected_block_metrics": expected_block,
        "frozen_block_metrics": frozen_block,
        "intervention_match_expected": frozen_intervention == expected_intervention,
        "block_match_expected": frozen_block == expected_block,
        "n_any_mismatches": n_any_mismatches,
        "all_pass": bool(frozen_intervention == expected_intervention and frozen_block == expected_block and n_any_mismatches == 0),
    }

    action_metrics = by_action(df, label_col, "r3z_frozen_decisao_recommended")
    frozen_artifact = {
        **artifact,
        "experiment": EXPERIMENT,
        "frozen_validation_status": validation["status"],
        "frozen_action_col": "r3z_frozen_decisao_recommended",
        "frozen_demote_col": "exp014b_r3z_frozen_demote_confirm_to_approve",
        "frozen_intervention_pred_col": "exp014b_r3z_frozen_intervention_pred",
        "frozen_block_pred_col": "exp014b_r3z_frozen_block_pred",
        "frozen_intervention_metrics": frozen_intervention,
        "frozen_block_metrics": frozen_block,
        "validation": validation,
    }
    summary = {
        "experiment": EXPERIMENT,
        "status": "DONE",
        "objective_status": validation["status"],
        "n_rows": int(len(df)),
        "n_frauds": int((y == 1).sum()),
        "n_normals": int((y == 0).sum()),
        "frozen_intervention_metrics": frozen_intervention,
        "frozen_block_metrics": frozen_block,
        "n_selected_demotions": int(len(artifact.get("selected_demotions", []))),
        "n_any_mismatches": n_any_mismatches,
        "all_pass": validation["all_pass"],
        "output_dir": str(out_dir),
    }
    contract = {"base_predictions_path": str(base_path), "artifact_path": str(artifact_path), "label_col": label_col, "base_action_col": base_action_col, "contract_ok": True, "missing": []}

    write_json(out_dir / "00_run_summary.json", summary)
    write_json(out_dir / "01_input_contract.json", contract)
    write_json(out_dir / "02_frozen_validation.json", validation)
    action_metrics.to_csv(out_dir / "03_decision_metrics_by_action.csv", index=False, encoding="utf-8")
    mismatch_df.to_csv(out_dir / "04_prediction_mismatches.csv", index=False, encoding="utf-8")
    write_json(out_dir / "05_policy_artifact_frozen.json", frozen_artifact)
    df.to_csv(out_dir / "06_predictions_frozen.csv", index=False, encoding="utf-8")
    report = f"""# {EXPERIMENT} - Frozen validation

## Resultado executivo
- Status: `{validation['status']}`
- All pass: `{validation['all_pass']}`
- Prediction mismatches: `{n_any_mismatches}`
- Regras congeladas: `{len(artifact.get('selected_demotions', []))}`

## Metricas de intervencao congelada
```json
{json.dumps(frozen_intervention, ensure_ascii=False, indent=2)}
```

## Metricas de BLOQUEAR congelado
```json
{json.dumps(frozen_block, ensure_ascii=False, indent=2)}
```

## Metricas por acao
{action_metrics.to_markdown(index=False)}

## Validacao
```json
{json.dumps(validation, ensure_ascii=False, indent=2)}
```

## Decisao sugerida
Se PASS, consolidar R3Z-FROZEN como baseline operacional. A proxima rodada deve buscar FPR 1,5% sem aumentar FN alem de 5.
"""
    (out_dir / "07_exp014b_r3z_frozen_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
