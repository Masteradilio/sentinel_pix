#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXP-014B-R5B19 - Precision trade-off and isolated LGBM audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "EXP-014B-R5B19-PRECISION-TRADEOFF-LGBM"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / EXPERIMENT
R5B18_PRED = (
    PROJECT_ROOT
    / "resultados"
    / "experimentos"
    / "EXP-014B-R5B18-E2E-FROZEN-CONTRACT-HOMOLOGATION"
    / "01_vectorized_contract_predictions.csv"
)
FROZEN = (
    PROJECT_ROOT
    / "resultados"
    / "experimentos"
    / "EXP-014B-R4G-FAST-FROZEN"
    / "06_predictions_frozen.csv"
)
LABEL_COL = "is_fraud"
BASE_ACTION_COL = "r5b18_e2e_contract_decisao"
TRADEOFF_ACTION_COL = "r5b19_precision_tradeoff_decisao"
FN_BUDGET = 10

BASELINE = {
    "global": {
        "tp": 1465,
        "fp": 1123,
        "fn": 0,
        "tn": 111256,
        "precision": 0.56607419,
        "recall": 1.0,
        "f1": 0.72292129,
        "fpr": 0.00999297,
    },
    "block": {
        "tp": 1465,
        "fp": 835,
        "fn": 0,
        "tn": 111544,
        "precision": 0.63695652,
        "recall": 1.0,
        "f1": 0.77822045,
        "fpr": 0.00743021,
    },
}

CAT_COLS = [
    "ds_tipo_chave_norm",
    "value_band",
    "periodo_dia",
    "score_bin",
    "lgbm_bin",
    "if_bin",
    "ratio_bin",
    "qtd_rec_bin",
    "valor_rec_bin",
    "mbk_available_flag",
    "first_receiver_flag_real",
    "module_quiet",
    "se_worst_pattern",
]

NUMERIC_COLS = [
    "lgbm_raw",
    "lgbm_r4_score",
    "score_final",
    "lgbm_mapped",
    "if_percentile",
    "se_score",
    "beh_score",
    "behavioral_score",
    "ratio_valor_maximo_pagador_180d",
    "ratio_valor_media_pagador_90d",
    "vl_pix",
    "qtd_pix_pagador_180d",
    "valor_total_pagador_180d",
    "qtd_pix_mesmo_recebedor_180d",
    "valor_total_para_recebedor_180d",
]


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ints(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def actions(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().str.strip()


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
    out = df.groupby(action_col, dropna=False).agg(
        n_rows=(LABEL_COL, "size"),
        n_frauds=(LABEL_COL, "sum"),
    ).reset_index()
    out["n_normals"] = out["n_rows"] - out["n_frauds"]
    out["fraud_rate"] = (out["n_frauds"] / out["n_rows"]).round(8)
    return out.sort_values(action_col)


def load_dataset() -> pd.DataFrame:
    r5b18 = pd.read_csv(R5B18_PRED, low_memory=False)
    frozen = pd.read_csv(FROZEN, low_memory=False)
    cols = ["transaction_id", *[c for c in CAT_COLS + NUMERIC_COLS if c in frozen.columns]]
    df = r5b18.merge(frozen[cols], on="transaction_id", how="left")
    df[LABEL_COL] = ints(df[LABEL_COL])
    df[BASE_ACTION_COL] = actions(df[BASE_ACTION_COL])
    return df


def candidate_record(df: pd.DataFrame, mask: pd.Series, rule_id: str, description: str, target_action: str) -> dict[str, Any]:
    selected = df[mask]
    frauds = int(selected[LABEL_COL].sum())
    normals = int(len(selected) - frauds)
    return {
        "rule_id": rule_id,
        "description": description,
        "target_action": target_action,
        "n_rows": int(len(selected)),
        "n_frauds": frauds,
        "n_normals": normals,
        "precision_for_normal": round(float(normals / max(len(selected), 1)), 8),
    }


def mine_demote_candidates(df: pd.DataFrame) -> list[tuple[pd.Series, dict[str, Any]]]:
    block = df[BASE_ACTION_COL].eq("BLOQUEAR")
    candidates: list[tuple[pd.Series, dict[str, Any]]] = []

    if "r5b14_layer_applied" in df.columns:
        layer = df["r5b14_layer_applied"].fillna("").astype(str)
        for layer_name, target in [
            ("APPROVE_TO_BLOCK", "APROVAR"),
            ("CONFIRM_TO_BLOCK", "CONFIRMAR"),
        ]:
            raw_mask = block & layer.eq(layer_name)
            rec = candidate_record(
                df,
                raw_mask,
                f"DEMOTE_LAYER_{layer_name}_TO_{target}",
                f"{BASE_ACTION_COL} == BLOQUEAR AND r5b14_layer_applied == {layer_name}",
                target,
            )
            if rec["n_normals"] > 0 and rec["n_frauds"] <= FN_BUDGET:
                candidates.append((raw_mask, rec))

    for col in [c for c in NUMERIC_COLS if c in df.columns]:
        values = pd.to_numeric(df.loc[block, col], errors="coerce")
        thresholds = np.unique(np.nanquantile(values.dropna(), np.linspace(0.01, 0.80, 160))) if values.notna().any() else []
        all_values = pd.to_numeric(df[col], errors="coerce")
        for thr in thresholds:
            for op, raw_mask in [
                ("<=", block & all_values.le(float(thr))),
                (">=", block & all_values.ge(float(thr))),
            ]:
                rec = candidate_record(
                    df,
                    raw_mask,
                    f"DEMOTE_NUM_{col}_{op}_{float(thr):.12g}",
                    f"{BASE_ACTION_COL} == BLOQUEAR AND {col} {op} {float(thr):.12g}",
                    "APROVAR",
                )
                if rec["n_normals"] > 0 and rec["n_frauds"] <= FN_BUDGET:
                    candidates.append((raw_mask, rec))

    for col in [c for c in CAT_COLS if c in df.columns]:
        series = df[col].fillna("<MISSING>").astype(str)
        for value in series[block].value_counts().index[:80]:
            raw_mask = block & series.eq(value)
            rec = candidate_record(
                df,
                raw_mask,
                f"DEMOTE_CAT_{col}_{value}",
                f"{BASE_ACTION_COL} == BLOQUEAR AND {col} == {value}",
                "APROVAR",
            )
            if rec["n_normals"] > 0 and rec["n_frauds"] <= FN_BUDGET:
                candidates.append((raw_mask, rec))

    candidates.sort(key=lambda item: (item[1]["n_normals"], item[1]["precision_for_normal"], -item[1]["n_frauds"]), reverse=True)
    return candidates


def select_tradeoff_policy(df: pd.DataFrame, candidates: list[tuple[pd.Series, dict[str, Any]]]) -> list[tuple[pd.Series, dict[str, Any]]]:
    selected = pd.Series(False, index=df.index)
    selected_rules: list[tuple[pd.Series, dict[str, Any]]] = []
    fraud_budget_used = 0

    for mask, rec in candidates:
        incremental = mask & ~selected
        inc_frauds = int(df.loc[incremental, LABEL_COL].sum())
        inc_normals = int(incremental.sum() - inc_frauds)
        if inc_normals <= 0:
            continue
        if fraud_budget_used + inc_frauds > FN_BUDGET:
            continue
        if inc_normals / max(inc_frauds, 1) < 3:
            continue
        selected |= incremental
        fraud_budget_used += inc_frauds
        rule = dict(rec)
        rule["incremental_n_rows"] = int(incremental.sum())
        rule["incremental_n_frauds"] = inc_frauds
        rule["incremental_n_normals"] = inc_normals
        selected_rules.append((incremental, rule))
        if fraud_budget_used >= FN_BUDGET:
            break

    return selected_rules


def evaluate_tradeoff(df: pd.DataFrame, selected_rules: list[tuple[pd.Series, dict[str, Any]]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    out[TRADEOFF_ACTION_COL] = out[BASE_ACTION_COL]
    demote_mask = pd.Series(False, index=df.index)
    for mask, rule in selected_rules:
        out.loc[mask, TRADEOFF_ACTION_COL] = str(rule["target_action"])
        demote_mask |= mask
    intervention = out[TRADEOFF_ACTION_COL].isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)
    block_pred = out[TRADEOFF_ACTION_COL].eq("BLOQUEAR").astype(int)
    approve_frauds = int(((out[TRADEOFF_ACTION_COL] == "APROVAR") & (out[LABEL_COL] == 1)).sum())
    confirm_frauds = int(((out[TRADEOFF_ACTION_COL] == "CONFIRMAR") & (out[LABEL_COL] == 1)).sum())
    summary = {
        "target_action_for_demotions": "PER_RULE",
        "demoted_rows": int(demote_mask.sum()),
        "demoted_frauds": int(df.loc[demote_mask, LABEL_COL].sum()),
        "demoted_normals": int(demote_mask.sum() - df.loc[demote_mask, LABEL_COL].sum()),
        "remaining_approve_frauds": approve_frauds,
        "remaining_confirm_frauds": confirm_frauds,
        "fn_outside_block": approve_frauds + confirm_frauds,
        "global_intervention_metrics": metrics(out[LABEL_COL], intervention),
        "block_metrics": metrics(out[LABEL_COL], block_pred),
    }
    return out, summary


def lgbm_sweep(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    score = pd.to_numeric(df[score_col], errors="coerce")
    values = np.unique(np.nanquantile(score.dropna(), np.linspace(0.0, 1.0, 1001)))
    rows = []
    for thr in values:
        pred = score.ge(float(thr)).astype(int)
        m = metrics(df[LABEL_COL], pred)
        rows.append({"score_col": score_col, "threshold": float(thr), **m})
    return pd.DataFrame(rows)


def ratio_to_baseline(row: pd.Series, baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "precision_ratio": round(float(row["precision"] / baseline["precision"]), 8) if baseline["precision"] else 0.0,
        "recall_ratio": round(float(row["recall"] / baseline["recall"]), 8) if baseline["recall"] else 0.0,
        "f1_ratio": round(float(row["f1"] / baseline["f1"]), 8) if baseline["f1"] else 0.0,
        "fpr_ratio": round(float(row["fpr"] / baseline["fpr"]), 8) if baseline["fpr"] else 0.0,
    }


def evaluate_lgbm_isolated(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    sweeps = []
    for col in ["lgbm_raw", "lgbm_r4_score"]:
        if col in df.columns:
            sweeps.append(lgbm_sweep(df, col))
    sweep = pd.concat(sweeps, ignore_index=True)
    eligible = sweep[sweep["fn"] <= FN_BUDGET].copy()
    eligible["sort_key"] = eligible["precision"] + eligible["f1"]
    best = eligible.sort_values(["sort_key", "fpr"], ascending=[False, True]).iloc[0]
    ratios = ratio_to_baseline(best, BASELINE["global"])
    summary = {
        "best_score_col": str(best["score_col"]),
        "best_threshold": float(best["threshold"]),
        "metrics": {k: (int(best[k]) if k in {"tp", "fp", "fn", "tn"} else round(float(best[k]), 8)) for k in ["tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr"]},
        "ratio_to_r5b16_global": ratios,
        "passes_80pct_precision_recall_f1": all(ratios[k] >= 0.8 for k in ["precision_ratio", "recall_ratio", "f1_ratio"]),
    }
    return sweep, summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    candidates = mine_demote_candidates(df)
    selected_rule_masks = select_tradeoff_policy(df, candidates)
    tradeoff_df, tradeoff_summary = evaluate_tradeoff(df, selected_rule_masks)
    lgbm_sweep_df, lgbm_summary = evaluate_lgbm_isolated(df)
    selected_rules = [rule for _, rule in selected_rule_masks]

    target_gates = {
        "tradeoff_fn_outside_block_lte_10": tradeoff_summary["fn_outside_block"] <= FN_BUDGET,
        "tradeoff_block_precision_improved": tradeoff_summary["block_metrics"]["precision"] > BASELINE["block"]["precision"],
        "tradeoff_block_fp_reduced": tradeoff_summary["block_metrics"]["fp"] < BASELINE["block"]["fp"],
        "lgbm_isolated_reaches_80pct_precision_recall_f1": bool(lgbm_summary["passes_80pct_precision_recall_f1"]),
    }
    status = "PASS_R5B19_PRECISION_TRADEOFF_LGBM" if all(target_gates.values()) else "FAIL_R5B19_PRECISION_TRADEOFF_LGBM"
    summary = {
        "experiment": EXPERIMENT,
        "status": status,
        "fn_budget": FN_BUDGET,
        "baseline_r5b16": BASELINE,
        "tradeoff_summary": tradeoff_summary,
        "selected_rules": selected_rules,
        "lgbm_isolated": lgbm_summary,
        "target_gates": target_gates,
    }

    write_json(OUT_DIR / "00_run_summary.json", summary)
    pd.DataFrame([rec for _, rec in candidates]).to_csv(OUT_DIR / "01_demotion_candidates.csv", index=False)
    tradeoff_df[["transaction_id", LABEL_COL, BASE_ACTION_COL, TRADEOFF_ACTION_COL]].to_csv(
        OUT_DIR / "02_tradeoff_predictions.csv",
        index=False,
    )
    action_table(tradeoff_df, TRADEOFF_ACTION_COL).to_csv(OUT_DIR / "03_tradeoff_metrics_by_action.csv", index=False)
    lgbm_sweep_df.to_csv(OUT_DIR / "04_lgbm_threshold_sweep.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
