#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-011B-R1 — Threshold Policy Sweep

Objetivo:
  Reavaliar os thresholds do EXP-011B sem retreinar o modelo, escolhendo
  políticas operacionais mais realistas que maximizem F1/recall dentro de
  limites de precision, FPR ou orçamento de FP.

Entrada default:
  resultados/experimentos/EXP-011B/02_threshold_sweep_validation.csv
  resultados/experimentos/EXP-011B/03_threshold_sweep_holdout.csv
  resultados/experimentos/EXP-011B/00_run_summary.json

Saídas:
  resultados/experimentos/EXP-011B-R1/
    00_threshold_policy_summary.json
    01_policy_comparison.csv
    02_selected_threshold_policy.json
    03_recommendation.md

Também copia o threshold selecionado para:
  backend/artefatos_candidatos/exp011b_lgbm_vnext/threshold_policy_exp011b_r1.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP011B_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-011B"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-011B-R1"
CANDIDATE_DIR = PROJECT_ROOT / "backend" / "artefatos_candidatos" / "exp011b_lgbm_vnext"


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def nearest_threshold(df: pd.DataFrame, threshold: float) -> pd.Series:
    idx = (df["threshold"] - threshold).abs().idxmin()
    return df.loc[idx]


def select_policy(validation: pd.DataFrame, holdout: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    policies = []

    def add_policy(name: str, selector_desc: str, sub: pd.DataFrame, sort_cols: list[str], ascending: list[bool]):
        if sub.empty:
            return
        row_val = sub.sort_values(sort_cols, ascending=ascending).iloc[0]
        row_hold = nearest_threshold(holdout, float(row_val["threshold"]))

        rec = {
            "policy": name,
            "selector": selector_desc,
            "threshold": float(row_val["threshold"]),

            "val_tp": int(row_val["tp"]),
            "val_fp": int(row_val["fp"]),
            "val_fn": int(row_val["fn"]),
            "val_tn": int(row_val["tn"]),
            "val_precision": float(row_val["precision"]),
            "val_recall": float(row_val["recall"]),
            "val_f1": float(row_val["f1"]),
            "val_fpr": float(row_val["fpr"]),

            "holdout_tp": int(row_hold["tp"]),
            "holdout_fp": int(row_hold["fp"]),
            "holdout_fn": int(row_hold["fn"]),
            "holdout_tn": int(row_hold["tn"]),
            "holdout_precision": float(row_hold["precision"]),
            "holdout_recall": float(row_hold["recall"]),
            "holdout_f1": float(row_hold["f1"]),
            "holdout_fpr": float(row_hold["fpr"]),
        }
        policies.append(rec)

    add_policy(
        "BEST_F1_VALIDATION",
        "Maior F1 na validação",
        validation,
        ["f1", "recall", "precision"],
        [False, False, False],
    )

    add_policy(
        "PRECISION_GE_50_FPR_LE_1PCT",
        "Maior F1 com precision>=50% e FPR<=1%",
        validation[(validation["precision"] >= 0.50) & (validation["fpr"] <= 0.01)],
        ["f1", "recall", "precision"],
        [False, False, False],
    )

    add_policy(
        "PRECISION_GE_70_FPR_LE_1PCT",
        "Maior F1 com precision>=70% e FPR<=1%",
        validation[(validation["precision"] >= 0.70) & (validation["fpr"] <= 0.01)],
        ["f1", "recall", "precision"],
        [False, False, False],
    )

    add_policy(
        "FPR_LE_05PCT_MAX_RECALL",
        "Maior recall com FPR<=0,5%",
        validation[validation["fpr"] <= 0.005],
        ["recall", "f1", "precision"],
        [False, False, False],
    )

    add_policy(
        "FPR_LE_1PCT_MAX_RECALL",
        "Maior recall com FPR<=1%",
        validation[validation["fpr"] <= 0.01],
        ["recall", "f1", "precision"],
        [False, False, False],
    )

    add_policy(
        "FP_BUDGET_100_BEST_F1",
        "Maior F1 com até 100 FP na validação",
        validation[validation["fp"] <= 100],
        ["f1", "recall", "precision"],
        [False, False, False],
    )

    add_policy(
        "FP_BUDGET_250_BEST_F1",
        "Maior F1 com até 250 FP na validação",
        validation[validation["fp"] <= 250],
        ["f1", "recall", "precision"],
        [False, False, False],
    )

    policy_df = pd.DataFrame(policies).drop_duplicates("policy")

    # Política oficial R1:
    # 1) precision>=50 e FPR<=1, se disponível;
    # 2) senão, melhor F1 validação.
    preferred = policy_df[policy_df["policy"] == "PRECISION_GE_50_FPR_LE_1PCT"]
    if not preferred.empty:
        selected = preferred.iloc[0].to_dict()
        selected["selection_reason"] = "Selecionado por equilíbrio: precision>=50%, FPR<=1% e maior F1 na validação."
    else:
        selected = policy_df[policy_df["policy"] == "BEST_F1_VALIDATION"].iloc[0].to_dict()
        selected["selection_reason"] = "Fallback: melhor F1 na validação."

    return policy_df, selected


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    val_path = EXP011B_DIR / "02_threshold_sweep_validation.csv"
    hold_path = EXP011B_DIR / "03_threshold_sweep_holdout.csv"
    summary_path = EXP011B_DIR / "00_run_summary.json"

    if not val_path.exists() or not hold_path.exists():
        raise FileNotFoundError("Artefatos do EXP-011B não encontrados. Rode primeiro o EXP-011B.")

    validation = pd.read_csv(val_path)
    holdout = pd.read_csv(hold_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    policy_df, selected = select_policy(validation, holdout)
    policy_df.to_csv(OUT_DIR / "01_policy_comparison.csv", index=False)

    out_summary = {
        "experiment": "EXP-011B-R1",
        "status": "DONE",
        "source_exp011b": str(EXP011B_DIR),
        "n_policies": int(len(policy_df)),
        "selected_policy": selected,
        "original_operational_threshold": summary.get("threshold_operational"),
        "original_holdout_operational": summary.get("holdout_operational"),
    }

    dump(out_summary, OUT_DIR / "00_threshold_policy_summary.json")
    dump(selected, OUT_DIR / "02_selected_threshold_policy.json")
    dump(selected, CANDIDATE_DIR / "threshold_policy_exp011b_r1.json")

    md = []
    md.append("# EXP-011B-R1 — Threshold Policy Sweep")
    md.append("")
    md.append("## Decisão")
    md.append("APROVADO — substituir o threshold operacional agressivo do EXP-011B pela política R1.")
    md.append("")
    md.append("## Política selecionada")
    md.append(f"- policy: `{selected['policy']}`")
    md.append(f"- threshold: `{selected['threshold']}`")
    md.append(f"- motivo: {selected['selection_reason']}")
    md.append("")
    md.append("## Holdout esperado com a política selecionada")
    for k in ["holdout_tp", "holdout_fp", "holdout_fn", "holdout_tn", "holdout_precision", "holdout_recall", "holdout_f1", "holdout_fpr"]:
        md.append(f"- {k}: {selected[k]}")
    md.append("")
    md.append("## Observação")
    md.append(
        "Essa etapa não retreina o modelo; apenas corrige a política de threshold. "
        "O próximo passo é EXP-011B-R2 para tentar melhorar o trade-off precision/recall por retreinamento conservador."
    )
    (OUT_DIR / "03_recommendation.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(out_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
