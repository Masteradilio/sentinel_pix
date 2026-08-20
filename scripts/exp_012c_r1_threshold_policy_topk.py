#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-012C-R1 — Threshold Policy + Top-K Analysis

Objetivo:
  Reaproveitar as predições/sweeps já geradas no EXP-012C completo e produzir
  uma análise rápida de políticas de threshold e top-k, sem retreinar.

Uso:
  python scripts\exp_012c_r1_threshold_policy_topk.py

Entradas:
  resultados/experimentos/EXP-012C/
    02_threshold_sweep_validation.csv
    03_threshold_sweep_holdout_label_safe.csv
    04_threshold_sweep_holdout_full.csv
    06_predictions_validation.csv
    07_predictions_holdout_label_safe.csv
    08_predictions_holdout_full.csv

Saídas:
  resultados/experimentos/EXP-012C-R1/
    00_summary.json
    01_threshold_policy_comparison.csv
    02_holdout_label_safe_diagnostic_thresholds.csv
    03_topk_validation.csv
    04_topk_holdout_label_safe.csv
    05_topk_holdout_full.csv
    06_recommendation.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-012C"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-012C-R1"


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def nearest(df: pd.DataFrame, threshold: float) -> pd.Series:
    return df.loc[(df["threshold"] - threshold).abs().idxmin()]


def row_to_prefixed(row: pd.Series, prefix: str) -> dict:
    keys = ["threshold", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr", "roc_auc", "average_precision"]
    return {f"{prefix}_{k}": row.get(k) for k in keys if k in row.index}


def build_policy_comparison(val: pd.DataFrame, hs: pd.DataFrame, hf: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(policy: str, selector: str, sub: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> None:
        if sub.empty:
            return
        v = sub.sort_values(sort_cols, ascending=ascending).iloc[0]
        hsafe = nearest(hs, float(v["threshold"]))
        hfull = nearest(hf, float(v["threshold"]))
        row = {"policy": policy, "selector": selector, "threshold": float(v["threshold"])}
        row.update(row_to_prefixed(v, "val"))
        row.update(row_to_prefixed(hsafe, "holdout_safe"))
        row.update(row_to_prefixed(hfull, "holdout_full"))
        rows.append(row)

    add("BEST_F1_VALIDATION", "Maior F1 na validação", val, ["f1", "recall", "precision"], [False, False, False])
    add("PRECISION_GE_50_BEST_F1_VALIDATION", "Precision>=50%, maior F1 na validação", val[val["precision"] >= 0.50], ["f1", "recall"], [False, False])
    add("PRECISION_GE_50_MAX_RECALL_VALIDATION", "Precision>=50%, maior recall na validação", val[val["precision"] >= 0.50], ["recall", "f1"], [False, False])
    add("PRECISION_GE_45_FPR_LE_1PCT_BEST_F1", "Precision>=45%, FPR<=1%, maior F1", val[(val["precision"] >= 0.45) & (val["fpr"] <= 0.01)], ["f1", "recall"], [False, False])
    add("RECALL_GE_50_FPR_LE_1PCT_BEST_F1", "Recall>=50%, FPR<=1%, maior F1", val[(val["recall"] >= 0.50) & (val["fpr"] <= 0.01)], ["f1", "precision"], [False, False])
    add("FPR_LE_1PCT_MAX_RECALL", "FPR<=1%, maior recall", val[val["fpr"] <= 0.01], ["recall", "f1"], [False, False])
    add("FPR_LE_05PCT_MAX_RECALL", "FPR<=0,5%, maior recall", val[val["fpr"] <= 0.005], ["recall", "f1"], [False, False])

    return pd.DataFrame(rows)


def diagnostic_holdout_thresholds(hs: pd.DataFrame, hf: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(policy: str, sub: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> None:
        if sub.empty:
            return
        h = sub.sort_values(sort_cols, ascending=ascending).iloc[0]
        hf_row = nearest(hf, float(h["threshold"]))
        row = {"diagnostic_policy": policy, "threshold": float(h["threshold"])}
        row.update(row_to_prefixed(h, "holdout_safe"))
        row.update(row_to_prefixed(hf_row, "holdout_full"))
        rows.append(row)

    add("SAFE_BEST_F1", hs, ["f1", "recall", "precision"], [False, False, False])
    add("SAFE_PRECISION_GE_50_BEST_F1", hs[hs["precision"] >= 0.50], ["f1", "recall"], [False, False])
    add("SAFE_RECALL_GE_50_FPR_LE_1PCT", hs[(hs["recall"] >= 0.50) & (hs["fpr"] <= 0.01)], ["f1", "precision"], [False, False])
    add("SAFE_FPR_LE_1PCT_MAX_RECALL", hs[hs["fpr"] <= 0.01], ["recall", "f1"], [False, False])
    return pd.DataFrame(rows)


def topk_metrics(pred: pd.DataFrame, split_name: str, ks=(25, 50, 75, 100, 150, 200, 250, 500, 1000)) -> pd.DataFrame:
    if "lgbm_v3_score" not in pred.columns:
        raise RuntimeError(f"{split_name}: coluna lgbm_v3_score ausente.")
    if "is_fraud" not in pred.columns:
        raise RuntimeError(f"{split_name}: coluna is_fraud ausente.")

    df = pred.copy()
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)
    df["lgbm_v3_score"] = pd.to_numeric(df["lgbm_v3_score"], errors="coerce").fillna(-1.0)
    df = df.sort_values("lgbm_v3_score", ascending=False).reset_index(drop=True)

    n_fraud = int(df["is_fraud"].sum())
    rows = []
    for k in ks:
        k2 = min(k, len(df))
        top = df.head(k2)
        tp = int(top["is_fraud"].sum())
        fp = int(k2 - tp)
        rows.append({
            "split": split_name,
            "k": int(k2),
            "tp_at_k": tp,
            "fp_at_k": fp,
            "precision_at_k": tp / max(k2, 1),
            "recall_at_k": tp / max(n_fraud, 1),
            "frauds_total": n_fraud,
            "score_min_at_k": float(top["lgbm_v3_score"].min()) if k2 else None,
            "score_max": float(df["lgbm_v3_score"].max()) if len(df) else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    val = pd.read_csv(IN_DIR / "02_threshold_sweep_validation.csv")
    hs = pd.read_csv(IN_DIR / "03_threshold_sweep_holdout_label_safe.csv")
    hf = pd.read_csv(IN_DIR / "04_threshold_sweep_holdout_full.csv")

    policies = build_policy_comparison(val, hs, hf)
    diagnostics = diagnostic_holdout_thresholds(hs, hf)

    policies.to_csv(OUT_DIR / "01_threshold_policy_comparison.csv", index=False)
    diagnostics.to_csv(OUT_DIR / "02_holdout_label_safe_diagnostic_thresholds.csv", index=False)

    pred_val = pd.read_csv(IN_DIR / "06_predictions_validation.csv")
    pred_hs = pd.read_csv(IN_DIR / "07_predictions_holdout_label_safe.csv")
    pred_hf = pd.read_csv(IN_DIR / "08_predictions_holdout_full.csv")

    top_val = topk_metrics(pred_val, "VALIDATION")
    top_hs = topk_metrics(pred_hs, "HOLDOUT_LABEL_SAFE")
    top_hf = topk_metrics(pred_hf, "HOLDOUT_FULL")

    top_val.to_csv(OUT_DIR / "03_topk_validation.csv", index=False)
    top_hs.to_csv(OUT_DIR / "04_topk_holdout_label_safe.csv", index=False)
    top_hf.to_csv(OUT_DIR / "05_topk_holdout_full.csv", index=False)

    best_validation_policy = policies.sort_values(["val_f1", "val_recall"], ascending=[False, False]).iloc[0].to_dict()
    best_safe_diag = diagnostics.sort_values(["holdout_safe_f1", "holdout_safe_recall"], ascending=[False, False]).iloc[0].to_dict()

    summary = {
        "experiment": "EXP-012C-R1",
        "status": "DONE",
        "source_dir": str(IN_DIR),
        "best_validation_policy": best_validation_policy,
        "best_holdout_safe_diagnostic": best_safe_diag,
        "note": "O diagnóstico por holdout não deve ser usado sozinho para promoção; serve para guiar o R2 tuning grid.",
    }
    dump(summary, OUT_DIR / "00_summary.json")

    md = []
    md.append("# EXP-012C-R1 — Threshold Policy + Top-K")
    md.append("")
    md.append("## Decisão")
    md.append("Usar esta etapa como diagnóstico rápido antes do retreino R2.")
    md.append("")
    md.append("## Melhor política por validação")
    for k in ["policy", "threshold", "val_precision", "val_recall", "val_f1", "val_fpr", "holdout_safe_precision", "holdout_safe_recall", "holdout_safe_f1", "holdout_safe_fpr"]:
        md.append(f"- {k}: {best_validation_policy.get(k)}")
    md.append("")
    md.append("## Melhor diagnóstico no holdout label-safe")
    for k in ["diagnostic_policy", "threshold", "holdout_safe_precision", "holdout_safe_recall", "holdout_safe_f1", "holdout_safe_fpr"]:
        md.append(f"- {k}: {best_safe_diag.get(k)}")
    md.append("")
    md.append("## Próximo passo")
    md.append("Executar EXP-012C-R2 com tuning grid, removendo `rn` e variando peso positivo/regularização.")
    (OUT_DIR / "06_recommendation.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
