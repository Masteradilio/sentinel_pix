#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3H-FROZEN — Frozen Validation do novo benchmark expandido

Objetivo:
  Validar de forma congelada o policy artifact do EXP-014B-R3H, sem nova
  mineração, sem beam search e sem recalibrar thresholds.

Benchmark esperado do R3H:

    TP=1409
    FP=4935
    FN=56
    recall=96,177%
    precision=22,210%
    FPR=4,391%
    Wilson low=95,069%

Entrada preferencial:
  resultados/experimentos/EXP-014B-R3G/09_predictions.csv

Policy congelada:
  resultados/experimentos/EXP-014B-R3H/12_policy_artifact.json

Coluna base:
  exp014b_r3g_balanced_final_pred

Saída final:
  exp014b_r3h_frozen_pred

Uso:
  python scripts/exp_014b_r3h_frozen_validation.py

Se quiser validar usando outro CSV com a coluna base:
  python scripts/exp_014b_r3h_frozen_validation.py --input caminho\\arquivo.csv

Critério de aprovação:
  - reproduzir exatamente TP=1409, FP=4935, FN=56;
  - TP_loss total = 0;
  - FP removidos total = 585;
  - Wilson low >= 0.95;
  - schema mínimo OK;
  - sem regras não aplicadas por coluna ausente.

Saídas:
  resultados/experimentos/EXP-014B-R3H-FROZEN/
    00_run_summary.json
    01_input_contract.json
    02_base_metrics.csv
    03_rule_replay_impact.csv
    04_frozen_metrics.csv
    05_time_block_metrics.csv
    06_wilson_recall_ci.csv
    07_false_negatives.csv
    08_false_positives_sample.csv
    09_policy_replay_artifact.json
    10_predictions.csv
    11_exp014b_r3h_frozen_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "dados").exists() else Path.cwd()

DEFAULT_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3G" / "09_predictions.csv"
DEFAULT_POLICY = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3H" / "12_policy_artifact.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3H-FROZEN"

DEFAULT_BASE_COL = "exp014b_r3g_balanced_final_pred"
FINAL_COL = "exp014b_r3h_frozen_pred"

EXPECTED = {
    "tp": 1409,
    "fp": 4935,
    "fn": 56,
    "recall": 0.96177474,
    "precision": 0.22209962,
    "fpr": 0.0439139,
    "fp_removed_vs_base": 585,
    "tp_loss_vs_base": 0,
    "wilson_low_min": 0.95,
}

FEATURE_COLS_FOR_CONTRACT = [
    "value_band",
    "ds_tipo_chave_norm",
    "periodo_dia",
    "first_receiver_flag_real",
    "mbk_available_flag",
    "lgbm_bin",
    "if_bin",
    "score_bin",
    "ratio_bin",
    "qtd_rec_bin",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().split(".")[-1] for c in df.columns]

    if "transaction_id" not in df.columns and "cd_pix" in df.columns:
        df["transaction_id"] = df["cd_pix"]
    if "event_datetime" not in df.columns and "dt_pix" in df.columns:
        df["event_datetime"] = df["dt_pix"]

    if "is_fraud" not in df.columns:
        raise RuntimeError("Coluna obrigatória ausente: is_fraud")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)

    if "transaction_id" in df.columns:
        df["transaction_id"] = df["transaction_id"].astype("string").str.strip()

    for c in ["event_datetime", "data_pix"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    return df.reset_index(drop=True)


def pick_col(df: pd.DataFrame, names: str | list[str]) -> str | None:
    if isinstance(names, str):
        names = [names]
    for n in names:
        if n in df.columns:
            return n
    return None


def num(df: pd.DataFrame, names: str | list[str], default: float = 0.0) -> pd.Series:
    col = pick_col(df, names)
    if col is None:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def qbin_series(s: pd.Series, name: str, bins: list[float]) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    labels = []
    edges = [-np.inf] + bins + [np.inf]
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        if np.isneginf(left):
            labels.append(f"{name}_LT_{right:g}")
        elif np.isposinf(right):
            labels.append(f"{name}_GE_{left:g}")
        else:
            labels.append(f"{name}_{left:g}_{right:g}")
    return pd.cut(vals, bins=edges, labels=labels, include_lowest=True).astype("string").fillna(f"{name}_MISSING").astype(str)


def add_bins_and_guards(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recria bins somente quando ausentes. Como a entrada preferencial é o
    predictions.csv do R3G, normalmente as colunas já estarão prontas.
    """
    df = df.copy()

    if "lgbm_bin" not in df.columns and pick_col(df, ["lgbm_r4_score", "lgbm_mapped", "lgbm_raw"]):
        df["lgbm_bin"] = qbin_series(num(df, ["lgbm_r4_score", "lgbm_mapped", "lgbm_raw"], 0.0), "lgbm", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1])
    if "if_bin" not in df.columns and pick_col(df, ["if_percentile", "if_percentile_x", "if_percentile_y"]):
        df["if_bin"] = qbin_series(num(df, ["if_percentile", "if_percentile_x", "if_percentile_y"], 0.0), "if", [0.32, 0.5, 0.7, 0.85, 0.95])
    if "score_bin" not in df.columns and "score_final" in df.columns:
        df["score_bin"] = qbin_series(num(df, "score_final", 0.0), "score", [0.5, 1, 2, 3, 5, 10])
    if "ratio_bin" not in df.columns and "ratio_valor_media_pagador_90d" in df.columns:
        df["ratio_bin"] = qbin_series(num(df, "ratio_valor_media_pagador_90d", 0.0), "ratio", [0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    if "qtd_rec_bin" not in df.columns and "qtd_pix_recebidos_180d" in df.columns:
        df["qtd_rec_bin"] = qbin_series(num(df, "qtd_pix_recebidos_180d", 0.0), "qtdrec", [0, 1, 2, 5, 10, 20, 50, 100])
    if "vl_bin" not in df.columns and "vl_pix" in df.columns:
        df["vl_bin"] = qbin_series(num(df, "vl_pix", 0.0), "vl", [20, 50, 100, 250, 500, 1000, 5000, 10000])

    if "module_quiet" not in df.columns:
        se_score = num(df, ["se_score_x", "se_score_y", "se_score"], 0.0)
        se_count = num(df, ["se_patterns_count", "se_pattern_count"], 0.0)
        beh_score = num(df, ["beh_score", "behavioral_score"], 0.0)
        beh_count = num(df, ["beh_factors_count", "behavioral_risk_factor_count"], 0.0)
        runtime = num(df, "runtime_flagged", 0.0)
        module_strong = (
            (se_score >= 40)
            | (se_count >= 2)
            | (beh_score >= 25)
            | (beh_count >= 2)
            | (runtime >= 1)
        )
        df["module_quiet"] = np.where(module_strong, "module_strong", "module_quiet")

    return df


def compute_metrics(y_true, y_pred) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 8),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 8),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 8),
        "fpr": round(float(fp / max(fp + tn, 1)), 8),
    }


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) / n) + (z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def make_time_blocks(df: pd.DataFrame, n_blocks: int) -> pd.Series:
    if "data_pix" in df.columns and df["data_pix"].notna().any():
        dates = pd.to_datetime(df["data_pix"], errors="coerce")
    elif "event_datetime" in df.columns and df["event_datetime"].notna().any():
        dates = pd.to_datetime(df["event_datetime"], errors="coerce")
    else:
        return pd.qcut(np.arange(len(df)), q=min(n_blocks, len(df)), labels=False, duplicates="drop").astype(int)

    tmp = pd.DataFrame({"date": dates, "_idx": np.arange(len(df))}).sort_values(["date", "_idx"])
    tmp["block"] = pd.qcut(np.arange(len(tmp)), q=min(n_blocks, len(tmp)), labels=False, duplicates="drop")
    out = pd.Series(index=tmp["_idx"].values, data=tmp["block"].values).sort_index()
    return out.astype(int)


def block_metrics(df: pd.DataFrame, pred: np.ndarray, blocks: pd.Series, policy_name: str) -> pd.DataFrame:
    y = df["is_fraud"].to_numpy(dtype=int)
    bvals = blocks.to_numpy()
    rows = []
    for b in sorted(blocks.dropna().unique()):
        idx = bvals == b
        part = df.loc[idx]
        rows.append({
            "policy_name": policy_name,
            "block": int(b),
            "n_rows": int(len(part)),
            "n_frauds": int(part["is_fraud"].sum()),
            **compute_metrics(y[idx], pred[idx]),
        })
    return pd.DataFrame(rows)


def parse_params(rule: dict[str, Any]) -> dict[str, Any]:
    raw = rule.get("params_json") or rule.get("params") or "{}"
    if isinstance(raw, dict):
        return raw
    # Handles Infinity if it appears in JSON-like artifacts.
    return json.loads(str(raw).replace("Infinity", "1e999"))


def rule_mask(df: pd.DataFrame, rule: dict[str, Any], current_pred: np.ndarray) -> tuple[np.ndarray, list[str]]:
    params = parse_params(rule)
    missing = []
    mask = np.ones(len(df), dtype=bool)
    rtype = params.get("type")

    if rtype == "combo":
        cols = params.get("combo_cols", [])
        vals = params.get("combo_values", [])
        for c, v in zip(cols, vals):
            if c not in df.columns:
                missing.append(c)
                return np.zeros(len(df), dtype=bool), missing
            mask = mask & (df[c].astype("string").fillna("<MISSING>").astype(str).to_numpy() == str(v))
    elif rtype == "numeric_threshold":
        c = params.get("col")
        if c not in df.columns:
            missing.append(str(c))
            return np.zeros(len(df), dtype=bool), missing
        vals = num(df, c, 0.0).to_numpy(dtype=float)
        cut = float(params.get("cut"))
        direction = params.get("direction")
        if direction == "le":
            mask = mask & (vals <= cut)
        else:
            mask = mask & (vals >= cut)
    else:
        missing.append(f"unsupported_rule_type:{rtype}")
        return np.zeros(len(df), dtype=bool), missing

    if params.get("require_module_quiet", False):
        if "module_quiet" not in df.columns:
            missing.append("module_quiet")
            return np.zeros(len(df), dtype=bool), missing
        mask = mask & (df["module_quiet"].astype(str).to_numpy() == "module_quiet")

    return mask & (current_pred.astype(int) == 1), missing


def apply_frozen_rules(df: pd.DataFrame, base_pred: np.ndarray, selected_rules: list[dict[str, Any]]) -> tuple[np.ndarray, pd.DataFrame]:
    y = df["is_fraud"].to_numpy(dtype=int)
    pred = base_pred.copy().astype(int)
    rows = []
    missing_any = []

    for i, rule in enumerate(selected_rules):
        mask, missing = rule_mask(df, rule, pred)
        tp_loss = int(((y == 1) & mask).sum())
        fp_removed = int(((y == 0) & mask).sum())
        n_removed = int(mask.sum())

        expected_tp_loss = rule.get("tp_loss")
        expected_fp_removed = rule.get("fp_removed")

        pred[mask] = 0

        rows.append({
            "rule_index": i,
            "rule_id": rule.get("rule_id"),
            "family": rule.get("family"),
            "description": rule.get("description"),
            "n_removed": n_removed,
            "tp_loss": tp_loss,
            "fp_removed": fp_removed,
            "expected_tp_loss": expected_tp_loss,
            "expected_fp_removed": expected_fp_removed,
            "tp_loss_match_expected": (expected_tp_loss is None) or int(expected_tp_loss) == tp_loss,
            "fp_removed_match_expected": (expected_fp_removed is None) or int(expected_fp_removed) == fp_removed,
            "missing_columns": "|".join(missing),
            "params_json": rule.get("params_json"),
        })
        missing_any.extend(missing)

    return pred, pd.DataFrame(rows)


def approx_equal(a: float, b: float, tol: float = 1e-8) -> bool:
    return abs(float(a) - float(b)) <= tol


def make_contract(df: pd.DataFrame, policy: dict[str, Any], base_col: str) -> dict[str, Any]:
    required = ["is_fraud", base_col]
    selected_rules = policy.get("selected_rules", [])
    rule_cols = set()
    unsupported = []
    for rule in selected_rules:
        params = parse_params(rule)
        if params.get("type") == "combo":
            rule_cols.update(params.get("combo_cols", []))
        elif params.get("type") == "numeric_threshold":
            rule_cols.add(params.get("col"))
        else:
            unsupported.append(params.get("type"))
        if params.get("require_module_quiet", False):
            rule_cols.add("module_quiet")

    all_required = required + sorted(str(c) for c in rule_cols if c)
    missing = [c for c in all_required if c not in df.columns]
    return {
        "n_rows": int(len(df)),
        "n_frauds": int(df["is_fraud"].sum()) if "is_fraud" in df.columns else None,
        "base_col": base_col,
        "n_selected_rules": int(len(selected_rules)),
        "required_columns": all_required,
        "missing_columns": missing,
        "unsupported_rule_types": unsupported,
        "feature_cols_present": [c for c in FEATURE_COLS_FOR_CONTRACT if c in df.columns],
        "contract_ok": not missing and not unsupported,
    }


def make_report(summary: dict[str, Any], rule_df: pd.DataFrame, metrics_df: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014B-R3H-FROZEN — Frozen Validation")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Input: `{summary['input_path']}`")
    lines.append(f"- Policy: `{summary['policy_path']}`")
    lines.append(f"- Base col: `{summary['base_col']}`")
    lines.append("")
    lines.append("## Métricas")
    lines.append(metrics_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Checks principais")
    lines.append(f"- Expected metrics matched: `{summary['expected_metrics_matched']}`")
    lines.append(f"- TP loss vs base: `{summary['tp_loss_vs_base']}`")
    lines.append(f"- FP removed vs base: `{summary['fp_removed_vs_base']}`")
    lines.append(f"- Wilson low: `{summary['wilson_recall_low']}`")
    lines.append(f"- Schema OK: `{summary['schema_ok']}`")
    lines.append(f"- Rule replay OK: `{summary['rule_replay_ok']}`")
    lines.append("")
    lines.append("## Impacto por regra")
    show_cols = [
        "rule_id", "description", "tp_loss", "fp_removed",
        "expected_tp_loss", "expected_fp_removed",
        "tp_loss_match_expected", "fp_removed_match_expected", "missing_columns",
    ]
    if rule_df.empty:
        lines.append("Nenhuma regra aplicada.")
    else:
        lines.append(rule_df[[c for c in show_cols if c in rule_df.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Decisão")
    if summary["objective_status"].startswith("PASS"):
        lines.append("Policy artifact do R3H validado de forma congelada. Próximo passo: EXP-014B-R3I — auditoria dos 56 FNs residuais.")
    else:
        lines.append("Validação congelada falhou. Não promover benchmark antes de corrigir divergências.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base-col", default=None)
    parser.add_argument("--expected-tp", type=int, default=EXPECTED["tp"])
    parser.add_argument("--expected-fp", type=int, default=EXPECTED["fp"])
    parser.add_argument("--expected-fn", type=int, default=EXPECTED["fn"])
    parser.add_argument("--expected-fp-removed", type=int, default=EXPECTED["fp_removed_vs_base"])
    parser.add_argument("--expected-tp-loss", type=int, default=EXPECTED["tp_loss_vs_base"])
    parser.add_argument("--wilson-low-min", type=float, default=EXPECTED["wilson_low_min"])
    parser.add_argument("--time-blocks", type=int, default=10)
    parser.add_argument("--no-write-predictions", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    policy_path = Path(args.policy)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014B-R3H-FROZEN — Frozen Validation")
    log("=" * 80)
    log(f"Input: {input_path}")
    log(f"Policy: {policy_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy artifact não encontrado: {policy_path}")

    df = add_bins_and_guards(normalize_columns(pd.read_csv(input_path, low_memory=False)))
    policy = load_json(policy_path)
    base_col = args.base_col or policy.get("base_col") or DEFAULT_BASE_COL

    contract = make_contract(df, policy, base_col)
    dump_json(contract, output_dir / "01_input_contract.json")

    if not contract["contract_ok"]:
        summary = {
            "experiment": "EXP-014B-R3H-FROZEN",
            "status": "DONE",
            "objective_status": "FAIL_CONTRACT_NOT_OK",
            "contract": contract,
            "input_path": str(input_path),
            "policy_path": str(policy_path),
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
            "output_dir": str(output_dir),
        }
        dump_json(summary, output_dir / "00_run_summary.json")
        log(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    df[base_col] = pd.to_numeric(df[base_col], errors="coerce").fillna(0).astype(int)
    base_pred = df[base_col].to_numpy(dtype=int)
    y = df["is_fraud"].to_numpy(dtype=int)

    base_metrics = compute_metrics(y, base_pred)
    pd.DataFrame([{"policy_name": "R3H_FROZEN_BASE", **base_metrics}]).to_csv(output_dir / "02_base_metrics.csv", index=False)

    frozen_pred, rule_df = apply_frozen_rules(df, base_pred, policy.get("selected_rules", []))
    df[FINAL_COL] = frozen_pred.astype(int)
    final_metrics = compute_metrics(y, frozen_pred)

    rule_df.to_csv(output_dir / "03_rule_replay_impact.csv", index=False)

    fp_removed_vs_base = base_metrics["fp"] - final_metrics["fp"]
    tp_loss_vs_base = base_metrics["tp"] - final_metrics["tp"]

    metrics_df = pd.DataFrame([
        {"policy_name": "R3H_FROZEN_BASE", **base_metrics},
        {"policy_name": "EXP014B_R3H_FROZEN_FINAL", **final_metrics},
    ])
    metrics_df["fp_delta_vs_expected_final"] = metrics_df["fp"] - int(args.expected_fp)
    metrics_df["fn_delta_vs_expected_final"] = metrics_df["fn"] - int(args.expected_fn)
    metrics_df.to_csv(output_dir / "04_frozen_metrics.csv", index=False)

    blocks = make_time_blocks(df, args.time_blocks)
    block_df = pd.concat([
        block_metrics(df, base_pred, blocks, "R3H_FROZEN_BASE"),
        block_metrics(df, frozen_pred, blocks, "EXP014B_R3H_FROZEN_FINAL"),
    ], ignore_index=True)
    block_df.to_csv(output_dir / "05_time_block_metrics.csv", index=False)

    wl, wh = wilson_ci(final_metrics["tp"], int(y.sum()))
    wilson_df = pd.DataFrame([{
        "policy_name": "EXP014B_R3H_FROZEN_FINAL",
        "tp": final_metrics["tp"],
        "n_frauds": int(y.sum()),
        "recall": final_metrics["recall"],
        "wilson_low": wl,
        "wilson_high": wh,
        "wilson_low_min": args.wilson_low_min,
        "wilson_pass": bool(wl >= args.wilson_low_min),
    }])
    wilson_df.to_csv(output_dir / "06_wilson_recall_ci.csv", index=False)

    df[(df["is_fraud"] == 1) & (df[FINAL_COL] == 0)].to_csv(output_dir / "07_false_negatives.csv", index=False)
    fp_df = df[(df["is_fraud"] == 0) & (df[FINAL_COL] == 1)].copy()
    if len(fp_df) > 5000:
        fp_df = fp_df.sample(5000, random_state=42)
    fp_df.to_csv(output_dir / "08_false_positives_sample.csv", index=False)

    expected_metrics_matched = (
        final_metrics["tp"] == int(args.expected_tp)
        and final_metrics["fp"] == int(args.expected_fp)
        and final_metrics["fn"] == int(args.expected_fn)
    )
    expected_delta_matched = (
        int(fp_removed_vs_base) == int(args.expected_fp_removed)
        and int(tp_loss_vs_base) == int(args.expected_tp_loss)
    )
    schema_ok = bool(contract["contract_ok"])
    rule_replay_ok = bool(
        rule_df["missing_columns"].fillna("").eq("").all()
        and rule_df["tp_loss_match_expected"].all()
        and rule_df["fp_removed_match_expected"].all()
    ) if not rule_df.empty else True
    wilson_pass = bool(wl >= args.wilson_low_min)

    all_pass = expected_metrics_matched and expected_delta_matched and schema_ok and rule_replay_ok and wilson_pass

    objective_status = "PASS_R3H_FROZEN_VALIDATED" if all_pass else "FAIL_R3H_FROZEN_DIVERGENCE"
    if expected_metrics_matched:
        objective_status += "_METRICS_MATCH"
    else:
        objective_status += "_METRICS_MISMATCH"
    if rule_replay_ok:
        objective_status += "_RULES_MATCH"
    else:
        objective_status += "_RULES_MISMATCH"
    if wilson_pass:
        objective_status += "_WILSON_PASS"
    else:
        objective_status += "_WILSON_FAIL"

    replay_artifact = {
        "experiment": "EXP-014B-R3H-FROZEN",
        "policy_name": "r3h_frozen_replay_validation",
        "source_policy_artifact": str(policy_path),
        "input_path": str(input_path),
        "base_col": base_col,
        "final_col": FINAL_COL,
        "base_metrics": base_metrics,
        "final_metrics": final_metrics,
        "fp_removed_vs_base": int(fp_removed_vs_base),
        "tp_loss_vs_base": int(tp_loss_vs_base),
        "wilson_low": wl,
        "wilson_high": wh,
        "checks": {
            "schema_ok": schema_ok,
            "expected_metrics_matched": expected_metrics_matched,
            "expected_delta_matched": expected_delta_matched,
            "rule_replay_ok": rule_replay_ok,
            "wilson_pass": wilson_pass,
            "all_pass": all_pass,
        },
        "selected_rules_replayed": rule_df.to_dict(orient="records") if not rule_df.empty else [],
        "notes": [
            "Frozen validation only: no mining, no threshold recalibration, no beam search.",
            "R3H is promotion-candidate benchmark only if all checks pass.",
            "Next step after PASS: EXP-014B-R3I false-negative residual audit."
        ],
    }
    dump_json(replay_artifact, output_dir / "09_policy_replay_artifact.json")

    if not args.no_write_predictions:
        df.to_csv(output_dir / "10_predictions.csv", index=False)

    summary = {
        "experiment": "EXP-014B-R3H-FROZEN",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "policy_path": str(policy_path),
        "base_col": base_col,
        "final_col": FINAL_COL,
        "n_rows": int(len(df)),
        "n_frauds": int(y.sum()),
        "base_metrics": base_metrics,
        "final_metrics": final_metrics,
        "fp_removed_vs_base": int(fp_removed_vs_base),
        "tp_loss_vs_base": int(tp_loss_vs_base),
        "wilson_recall_low": wl,
        "wilson_recall_high": wh,
        "schema_ok": schema_ok,
        "expected_metrics_matched": expected_metrics_matched,
        "expected_delta_matched": expected_delta_matched,
        "rule_replay_ok": rule_replay_ok,
        "wilson_pass": wilson_pass,
        "all_pass": all_pass,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, rule_df, metrics_df)
    (output_dir / "11_exp014b_r3h_frozen_report.md").write_text(report, encoding="utf-8")

    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_input_contract.json",
        output_dir / "03_rule_replay_impact.csv",
        output_dir / "04_frozen_metrics.csv",
        output_dir / "05_time_block_metrics.csv",
        output_dir / "06_wilson_recall_ci.csv",
        output_dir / "09_policy_replay_artifact.json",
        output_dir / "11_exp014b_r3h_frozen_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
