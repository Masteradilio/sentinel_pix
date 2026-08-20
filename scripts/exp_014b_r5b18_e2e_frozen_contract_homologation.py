#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXP-014B-R5B18 - Homologacao do contrato frozen R4G/R5B16 no E2E."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
CORE_DIR = BACKEND_DIR / "core"
BACKEND_SCRIPTS_DIR = BACKEND_DIR / "scripts"

for path in [str(CORE_DIR), str(BACKEND_DIR), str(PROJECT_ROOT), str(BACKEND_SCRIPTS_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ["ENABLE_R5B14_POLICY"] = "1"
os.environ["ENABLE_R5B16_FROZEN_CONTRACT"] = "1"
os.environ["USE_PRECOMPUTED_FEATURES"] = "1"

import simular_pipeline_e2e_v2 as sim  # noqa: E402
from backend.core.severity_policy import apply_r5b16_frozen_contract_policy, r5b16_policy_metadata  # noqa: E402


EXPERIMENT = "EXP-014B-R5B18-E2E-FROZEN-CONTRACT-HOMOLOGATION"
DEFAULT_INPUT = PROJECT_ROOT / "dados" / "hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv"
FROZEN_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R4G-FAST-FROZEN" / "06_predictions_frozen.csv"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / EXPERIMENT
LABEL_COL = "is_fraud"
FROZEN_ACTION_COL = "r4g_fast_frozen_decisao_recommended"
FINAL_ACTION_COL = "r5b18_e2e_contract_decisao"


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


def prepare_input(input_path: Path, frozen_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    frozen = pd.read_csv(frozen_path, low_memory=False)
    if "transaction_id" not in df.columns:
        raise KeyError("transaction_id ausente na base de entrada.")
    if FROZEN_ACTION_COL not in frozen.columns:
        raise KeyError(f"{FROZEN_ACTION_COL} ausente no frozen.")

    merge_cols = [
        c
        for c in frozen.columns
        if c not in df.columns or c in {"transaction_id", FROZEN_ACTION_COL, "score_bin", "lgbm_bin", "ratio_bin", "lgbm_r4_score"}
    ]
    merge_cols = list(dict.fromkeys(["transaction_id", FROZEN_ACTION_COL, *merge_cols]))
    if FROZEN_ACTION_COL not in df.columns:
        df = df.merge(frozen[merge_cols], on="transaction_id", how="left")
    df[LABEL_COL] = ints(df[LABEL_COL])
    missing_frozen = int(df[FROZEN_ACTION_COL].isna().sum())
    if missing_frozen:
        raise RuntimeError(f"{missing_frozen} linhas sem acao frozen apos merge.")
    return df


def limited_stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    fraud = df[df[LABEL_COL] == 1]
    normal = df[df[LABEL_COL] == 0]
    n_fraud = min(len(fraud), max(1, round(n * len(fraud) / max(len(df), 1))))
    n_normal = min(len(normal), n - n_fraud)
    return pd.concat(
        [
            fraud.sample(n=n_fraud, random_state=42) if n_fraud else fraud.head(0),
            normal.sample(n=n_normal, random_state=42) if n_normal else normal.head(0),
        ],
        axis=0,
    ).sort_index().reset_index(drop=True)


def run_vectorized(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    final_actions, trace = apply_r5b16_frozen_contract_policy(df)
    out = df[["transaction_id", LABEL_COL, FROZEN_ACTION_COL]].copy()
    out[FINAL_ACTION_COL] = actions(final_actions)
    out["r5b14_rule_applied"] = trace["r5b14_rule_applied"]
    out["r5b14_layer_applied"] = trace["r5b14_layer_applied"]
    intervention = out[FINAL_ACTION_COL].isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)
    block = out[FINAL_ACTION_COL].eq("BLOQUEAR").astype(int)
    approve_frauds = int(((out[FINAL_ACTION_COL] == "APROVAR") & (out[LABEL_COL] == 1)).sum())
    confirm_frauds = int(((out[FINAL_ACTION_COL] == "CONFIRMAR") & (out[LABEL_COL] == 1)).sum())
    global_metrics = metrics(out[LABEL_COL], intervention)
    block_metrics = metrics(out[LABEL_COL], block)
    summary = {
        "n_rows": int(len(out)),
        "n_frauds": int(out[LABEL_COL].sum()),
        "remaining_approve_frauds": approve_frauds,
        "remaining_confirm_frauds": confirm_frauds,
        "fn_outside_block": approve_frauds + confirm_frauds,
        "global_intervention_metrics": global_metrics,
        "block_metrics": block_metrics,
    }
    return out, summary


def run_e2e_sample(
    df: pd.DataFrame,
    expected: pd.DataFrame,
    sample_n: int,
    progress_every: int,
    workers: int,
) -> dict[str, Any]:
    sample = limited_stratified_sample(df, sample_n)
    if workers > 1:
        predictions = sim.process_batch_parallel(sample, n_workers=workers)
    else:
        predictions = sim.process_batch_sequential(sample, progress_every=progress_every)
    predictions["decisao"] = actions(predictions["decisao"])
    expected_map = expected.set_index("transaction_id")[FINAL_ACTION_COL]
    predictions["expected_contract_decisao"] = predictions["transaction_id"].map(expected_map)
    mismatch = predictions["decisao"].ne(predictions["expected_contract_decisao"])
    return {
        "sample_n": int(len(predictions)),
        "sample_frauds": int(predictions[LABEL_COL].sum()),
        "mismatches": int(mismatch.sum()),
        "csv": "02_e2e_sample_predictions.csv",
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--frozen", type=Path, default=FROZEN_INPUT)
    parser.add_argument("--sample-e2e", type=int, default=200)
    parser.add_argument("--e2e-workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-fn-outside-block", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    t0 = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_input(args.input, args.frozen)
    vectorized, vectorized_summary = run_vectorized(df)
    by_action = action_table(vectorized, FINAL_ACTION_COL)
    by_layer = action_table(vectorized, "r5b14_layer_applied")

    e2e_summary: dict[str, Any] | None = None
    if args.sample_e2e:
        e2e = run_e2e_sample(df, vectorized, args.sample_e2e, args.progress_every, args.e2e_workers)
        predictions = e2e.pop("predictions")
        predictions.to_csv(args.output_dir / "02_e2e_sample_predictions.csv", index=False)
        e2e_summary = e2e

    target_gates = {
        "fpr_lt_1pct": vectorized_summary["global_intervention_metrics"]["fpr"] < 0.01,
        "fn_outside_block_lte_budget": vectorized_summary["fn_outside_block"] <= args.max_fn_outside_block,
        "e2e_sample_matches_vectorized_contract": (e2e_summary or {}).get("mismatches", 0) == 0,
    }
    status = "PASS_R5B18_E2E_FROZEN_CONTRACT_HOMOLOGATION" if all(target_gates.values()) else "FAIL_R5B18_E2E_FROZEN_CONTRACT_HOMOLOGATION"
    summary = {
        "experiment": EXPERIMENT,
        "status": status,
        "policy_metadata": r5b16_policy_metadata(),
        "input_file": str(args.input.relative_to(PROJECT_ROOT)),
        "frozen_file": str(args.frozen.relative_to(PROJECT_ROOT)),
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "runtime_flags": {
            "ENABLE_R5B16_FROZEN_CONTRACT": os.environ.get("ENABLE_R5B16_FROZEN_CONTRACT"),
            "ENABLE_R5B14_POLICY": os.environ.get("ENABLE_R5B14_POLICY"),
            "USE_PRECOMPUTED_FEATURES": os.environ.get("USE_PRECOMPUTED_FEATURES"),
        },
        "fn_budget": args.max_fn_outside_block,
        "vectorized_contract": vectorized_summary,
        "e2e_sample": e2e_summary,
        "target_gates": target_gates,
    }

    write_json(args.output_dir / "00_run_summary.json", summary)
    vectorized.to_csv(args.output_dir / "01_vectorized_contract_predictions.csv", index=False)
    by_action.to_csv(args.output_dir / "03_metrics_by_action.csv", index=False)
    by_layer.to_csv(args.output_dir / "04_metrics_by_r5b14_layer.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
