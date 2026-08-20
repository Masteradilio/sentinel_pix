# -*- coding: utf-8 -*-
"""
EXP-014B-R4F-FROZEN — replay/validação congelada do R4F.

Entrada default:
  resultados/experimentos/EXP-014B-R4E-FROZEN/06_predictions_frozen.csv
  resultados/experimentos/EXP-014B-R4F/08_policy_artifact_recommended.json

Saída:
  resultados/experimentos/EXP-014B-R4F-FROZEN/
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
import pandas as pd

LABELS = ["is_fraud", "fraude", "target", "label", "tp_fraude"]
EXPERIMENT = "EXP-014B-R4F-FROZEN"

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-predictions", default=None)
    p.add_argument("--artifact", default=None)
    p.add_argument("--r4f-predictions", default=None)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()

def defaults():
    root = Path.cwd()
    return (
        root / "resultados/experimentos/EXP-014B-R4E-FROZEN/06_predictions_frozen.csv",
        root / "resultados/experimentos/EXP-014B-R4F/08_policy_artifact_recommended.json",
        root / "resultados/experimentos/EXP-014B-R4F/09_predictions_recommended.csv",
        root / "resultados/experimentos/EXP-014B-R4F-FROZEN",
    )

def find_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n in df.columns: return n
        if n.lower() in lower: return lower[n.lower()]
    raise KeyError(f"Coluna não encontrada: {names}")

def ints(s):
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

def act(s):
    return s.astype(str).str.strip().str.upper()

def inter_from_action(a):
    return act(a).isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)

def block_from_action(a):
    return act(a).eq("BLOQUEAR").astype(int)

def metrics(y, pred):
    y, p = ints(y), ints(pred)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(float(pr), 8), "recall": round(float(rc), 8),
            "f1": round(float(f1), 8), "fpr": round(float(fpr), 8)}

def action_table(df, label_col, action_col):
    y = ints(df[label_col])
    rows = []
    for a, idx in df.groupby(action_col, dropna=False).groups.items():
        idx = list(idx); yy = y.loc[idx]; n = len(idx)
        rows.append({
            "action": str(a), "n_rows": int(n),
            "n_frauds": int((yy == 1).sum()),
            "n_normals": int((yy == 0).sum()),
            "precision_within_action": round(float((yy == 1).sum() / n), 8) if n else 0.0,
        })
    return pd.DataFrame(rows).sort_values("action")

def mask_desc(df, desc):
    prefix = "Mover BLOQUEAR para CONFIRMAR R4F com "
    if not desc.startswith(prefix):
        raise ValueError(f"Descrição R4F não suportada: {desc}")
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

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    a = args()
    base_d, art_d, ref_d, out_d = defaults()
    art_path = Path(a.artifact) if a.artifact else art_d
    if not art_path.exists(): raise FileNotFoundError(art_path)
    artifact = json.loads(art_path.read_text(encoding="utf-8"))

    base_path = Path(a.base_predictions) if a.base_predictions else Path(artifact.get("input_predictions_path") or base_d)
    if not base_path.exists(): base_path = base_d
    if not base_path.exists(): raise FileNotFoundError(base_path)

    ref_path = Path(a.r4f_predictions) if a.r4f_predictions else ref_d
    out = Path(a.output_dir) if a.output_dir else out_d
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(base_path, low_memory=False).copy()
    label_col = find_col(df, LABELS)
    base_action_col = artifact["base_action_col"]
    base_action = act(df[base_action_col])

    move = pd.Series(False, index=df.index)
    for rule in artifact.get("selected_block_to_confirm_rules", []):
        move |= base_action.eq("BLOQUEAR") & mask_desc(df, str(rule["description"]))

    final_action = base_action.copy()
    final_action.loc[move] = "CONFIRMAR"

    df["exp014b_r4f_frozen_block_to_confirm"] = move.astype(int)
    df["r4f_frozen_decisao_recommended"] = final_action
    df["exp014b_r4f_frozen_intervention_pred"] = inter_from_action(final_action)
    df["exp014b_r4f_frozen_block_pred"] = block_from_action(final_action)

    frozen_i = metrics(df[label_col], df["exp014b_r4f_frozen_intervention_pred"])
    frozen_b = metrics(df[label_col], df["exp014b_r4f_frozen_block_pred"])
    i_ok = frozen_i == artifact["final_intervention_metrics"]
    b_ok = frozen_b == artifact["final_block_metrics"]

    n_mismatch = 0
    mismatch_df = pd.DataFrame()
    if ref_path.exists():
        ref = pd.read_csv(ref_path, low_memory=False)
        mismatch = pd.Series(False, index=df.index)
        for src, dst, fn in [
            (artifact.get("final_action_col", "r4f_decisao_recommended"), "r4f_frozen_decisao_recommended", act),
            (artifact.get("intervention_pred_col", "exp014b_r4f_intervention_pred"), "exp014b_r4f_frozen_intervention_pred", ints),
            (artifact.get("block_pred_col", "exp014b_r4f_block_pred"), "exp014b_r4f_frozen_block_pred", ints),
        ]:
            if src in ref.columns:
                mismatch |= fn(ref[src]).ne(fn(df[dst]))
        n_mismatch = int(mismatch.sum())
        cols = [c for c in [label_col, base_action_col, "r4f_frozen_decisao_recommended", "lgbm_r4_score", "score_final"] if c in df.columns]
        mismatch_df = df.loc[mismatch, cols].copy()

    status = "PASS_R4F_FROZEN_VALIDATED_REPLAY_OK" if i_ok and b_ok and n_mismatch == 0 else "FAIL_R4F_FROZEN_VALIDATION_MISMATCH"
    by_action = action_table(df, label_col, "r4f_frozen_decisao_recommended")

    validation = {
        "status": status, "base_predictions_path": str(base_path), "artifact_path": str(art_path),
        "reference_predictions_path": str(ref_path) if ref_path.exists() else None,
        "base_action_col": base_action_col,
        "frozen_action_col": "r4f_frozen_decisao_recommended",
        "frozen_intervention_col": "exp014b_r4f_frozen_intervention_pred",
        "frozen_block_col": "exp014b_r4f_frozen_block_pred",
        "expected_intervention_metrics": artifact["final_intervention_metrics"],
        "frozen_intervention_metrics": frozen_i,
        "expected_block_metrics": artifact["final_block_metrics"],
        "frozen_block_metrics": frozen_b,
        "intervention_match_expected": i_ok,
        "block_match_expected": b_ok,
        "n_any_mismatches": n_mismatch,
        "all_pass": status.startswith("PASS"),
    }

    frozen_artifact = dict(artifact)
    frozen_artifact.update({
        "experiment": EXPERIMENT,
        "frozen_validation_status": status,
        "frozen_action_col": "r4f_frozen_decisao_recommended",
        "frozen_block_to_confirm_col": "exp014b_r4f_frozen_block_to_confirm",
        "frozen_intervention_pred_col": "exp014b_r4f_frozen_intervention_pred",
        "frozen_block_pred_col": "exp014b_r4f_frozen_block_pred",
        "frozen_intervention_metrics": frozen_i,
        "frozen_block_metrics": frozen_b,
        "validation": validation,
    })

    summary = {
        "experiment": EXPERIMENT, "status": "DONE", "objective_status": status,
        "n_rows": int(len(df)), "frozen_intervention_metrics": frozen_i,
        "frozen_block_metrics": frozen_b, "n_block_to_confirm": int(move.sum()),
        "n_any_mismatches": n_mismatch, "all_pass": validation["all_pass"], "output_dir": str(out),
    }

    write_json(out / "00_run_summary.json", summary)
    write_json(out / "01_input_contract.json", {"base_predictions_path": str(base_path), "artifact_path": str(art_path), "label_col": label_col, "base_action_col": base_action_col, "contract_ok": True, "missing": []})
    write_json(out / "02_frozen_validation.json", validation)
    by_action.to_csv(out / "03_decision_metrics_by_action.csv", index=False, encoding="utf-8")
    mismatch_df.to_csv(out / "04_prediction_mismatches.csv", index=False, encoding="utf-8")
    write_json(out / "05_policy_artifact_frozen.json", frozen_artifact)
    df.to_csv(out / "06_predictions_frozen.csv", index=False, encoding="utf-8")

    try: action_md = by_action.to_markdown(index=False)
    except Exception: action_md = by_action.to_string(index=False)
    report = f"""# {EXPERIMENT}

## Resultado executivo
- Status: `{status}`
- All pass: `{validation['all_pass']}`
- Prediction mismatches: `{n_mismatch}`

## Intervenção congelada
```json
{json.dumps(frozen_i, ensure_ascii=False, indent=2)}
```

## BLOQUEAR congelado
```json
{json.dumps(frozen_b, ensure_ascii=False, indent=2)}
```

## Métricas por ação
{action_md}
"""
    (out / "07_exp014b_r4f_frozen_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
