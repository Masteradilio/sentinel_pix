#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R5B17 - PipelineOrquestrador operational homologation.

Executa o PipelineOrquestrador real com a politica R5B14/R5B16 ativada por
flag e mede as decisoes finais na base MAF.
"""

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
os.environ["USE_PRECOMPUTED_FEATURES"] = "1"

import simular_pipeline_e2e_v2 as sim  # noqa: E402


EXPERIMENT = "EXP-014B-R5B17-PIPELINE-HOMOLOGATION"
DEFAULT_INPUT = PROJECT_ROOT / "dados" / "hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / EXPERIMENT
LABEL_COL = "is_fraud"


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ints(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def actions(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().str.strip()


def binary_metrics(y_true: pd.Series, pred: pd.Series) -> dict[str, Any]:
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


def action_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("decisao", dropna=False).agg(
        n_rows=(LABEL_COL, "size"),
        n_frauds=(LABEL_COL, "sum"),
    ).reset_index()
    out["n_normals"] = out["n_rows"] - out["n_frauds"]
    out["fraud_rate"] = (out["n_frauds"] / out["n_rows"]).round(8)
    return out.sort_values("decisao")


def layer_table(df: pd.DataFrame) -> pd.DataFrame:
    if "r5b14_layer_applied" not in df.columns:
        return pd.DataFrame(columns=["r5b14_layer_applied", "n_rows", "n_frauds", "n_normals"])
    applied = df[df.get("r5b14_policy_applied", False) == True].copy()  # noqa: E712
    if applied.empty:
        return pd.DataFrame(columns=["r5b14_layer_applied", "n_rows", "n_frauds", "n_normals"])
    out = applied.groupby("r5b14_layer_applied", dropna=False).agg(
        n_rows=(LABEL_COL, "size"),
        n_frauds=(LABEL_COL, "sum"),
    ).reset_index()
    out["n_normals"] = out["n_rows"] - out["n_frauds"]
    return out.sort_values("r5b14_layer_applied")


def load_input(path: Path, sample: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    if LABEL_COL not in df.columns:
        raise KeyError(f"Coluna obrigatoria ausente: {LABEL_COL}")
    df[LABEL_COL] = ints(df[LABEL_COL])
    if "event_datetime" in df.columns:
        df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
        df = df.sort_values("event_datetime").reset_index(drop=True)
    if sample and sample < len(df):
        fraud = df[df[LABEL_COL] == 1]
        normal = df[df[LABEL_COL] == 0]
        fraud_share = len(fraud) / max(len(df), 1)
        n_fraud = min(len(fraud), max(1, round(sample * fraud_share)))
        n_normal = min(len(normal), sample - n_fraud)
        sampled = [
            fraud.sample(n=n_fraud, random_state=42) if n_fraud else fraud.head(0),
            normal.sample(n=n_normal, random_state=42) if n_normal else normal.head(0),
        ]
        df = pd.concat(sampled, axis=0).sort_index().reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample", type=int, help="Executa amostra estratificada com N transacoes.")
    mode.add_argument("--full", action="store_true", help="Executa a base completa.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--frauds-only", action="store_true", help="Filtra a entrada para is_fraud=1 antes do processamento.")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    t0 = time.perf_counter()
    out_dir = args.output_dir
    if args.sample:
        out_dir = out_dir / f"sample_{args.sample}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_input(args.input, args.sample)
    if args.frauds_only:
        df = df[df[LABEL_COL] == 1].reset_index(drop=True)
        out_dir = out_dir / "frauds_only"
        out_dir.mkdir(parents=True, exist_ok=True)
    if args.workers > 1:
        predictions = sim.process_batch_parallel(df, n_workers=args.workers)
    else:
        predictions = sim.process_batch_sequential(df, progress_every=args.progress_every)

    predictions["decisao"] = actions(predictions["decisao"])
    predictions[LABEL_COL] = ints(predictions[LABEL_COL])
    intervention_pred = predictions["decisao"].isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)
    block_pred = predictions["decisao"].eq("BLOQUEAR").astype(int)

    global_metrics = binary_metrics(predictions[LABEL_COL], intervention_pred)
    block_metrics = binary_metrics(predictions[LABEL_COL], block_pred)
    by_action = action_table(predictions)
    by_layer = layer_table(predictions)
    error_count = int(predictions["decisao"].eq("ERRO").sum())
    approve_frauds = int(((predictions["decisao"] == "APROVAR") & (predictions[LABEL_COL] == 1)).sum())
    confirm_frauds = int(((predictions["decisao"] == "CONFIRMAR") & (predictions[LABEL_COL] == 1)).sum())

    target_gates = {
        "no_pipeline_errors": error_count == 0,
        "fpr_lt_1pct": global_metrics["fpr"] < 0.01,
        "fn_lte_5_outside_block": global_metrics["fn"] <= 5,
        "approve_frauds_eq_0": approve_frauds == 0,
        "confirm_frauds_eq_0": confirm_frauds == 0,
    }
    all_pass = all(target_gates.values())
    status = "PASS_R5B17_PIPELINE_HOMOLOGATION" if all_pass else "FAIL_R5B17_PIPELINE_HOMOLOGATION"

    predictions.to_csv(out_dir / "01_pipeline_predictions.csv", index=False)
    by_action.to_csv(out_dir / "02_metrics_by_action.csv", index=False)
    by_layer.to_csv(out_dir / "03_r5b14_layers.csv", index=False)

    summary = {
        "experiment": EXPERIMENT,
        "status": status,
        "mode": "sample" if args.sample else "full",
        "sample": args.sample,
        "frauds_only": bool(args.frauds_only),
        "input_file": str(args.input.relative_to(PROJECT_ROOT)),
        "n_rows": int(len(predictions)),
        "n_frauds": int(predictions[LABEL_COL].sum()),
        "workers": args.workers,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "runtime_flags": {
            "ENABLE_R5B14_POLICY": os.environ.get("ENABLE_R5B14_POLICY"),
            "USE_PRECOMPUTED_FEATURES": os.environ.get("USE_PRECOMPUTED_FEATURES"),
        },
        "error_count": error_count,
        "remaining_approve_frauds": approve_frauds,
        "remaining_confirm_frauds": confirm_frauds,
        "global_intervention_metrics": global_metrics,
        "block_metrics": block_metrics,
        "target_gates": target_gates,
    }
    write_json(out_dir / "00_run_summary.json", summary)

    report = f"""# {EXPERIMENT} - pipeline homologation

## Resultado executivo
- Status: `{status}`
- Modo: `{summary['mode']}`
- Linhas: `{summary['n_rows']}`
- Fraudes: `{summary['n_frauds']}`
- Erros de pipeline: `{error_count}`

## Metricas globais
```json
{json.dumps(global_metrics, ensure_ascii=False, indent=2)}
```

## Metricas BLOQUEAR
```json
{json.dumps(block_metrics, ensure_ascii=False, indent=2)}
```

## Gates
```json
{json.dumps(target_gates, ensure_ascii=False, indent=2)}
```
"""
    (out_dir / "04_exp014b_r5b17_pipeline_homologation_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
