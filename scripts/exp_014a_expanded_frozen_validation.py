#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014A — Expanded Frozen Validation

Objetivo:
  Validar em escala maior a política congelada vencedora do EXP-013K/013L:

      residual_fp_mined_tp0_policy
      validação pequena: TP=118, FP=199, FN=6, recall=95.16%, precision=37.22%

Motivação:
  O EXP-013L explicou que os avisos do bootstrap vinham de suporte positivo
  insuficiente: apenas 124 fraudes e TP exatamente no piso necessário para
  recall >= 95%.

  O EXP-014A resolve isso do jeito correto:
    - aplica a política congelada em um dataset expandido;
    - usa mais fraudes e não-fraudes representativos;
    - não minera regras novas;
    - mede recall, TP buffer, Wilson CI, bootstrap e FPR em escala.

Entrada esperada:
  Um CSV expandido JÁ SCOREADO com:
    - is_fraud
    - event_datetime ou data_pix
    - uma predição base antes da política EXP-013K:
        pred_STRICT_RECALL95_SAFE_ONLY
        ou exp013k_base_pred
        ou exp013h_frozen_pred
        ou exp013g_micro_pred
    - colunas brutas/bins usadas pelas 10 regras EXP-013K:
        ds_tipo_chave_norm
        first_receiver_flag_real
        mbk_available_flag
        value_band
        ratio_valor_media_pagador_90d ou ratio_bin
        lgbm_r4_score/r4_score/lgbm_mapped/lgbm_raw ou lgbm_bin
        score_final ou score_bin
        vl_pix ou vl_bin
        if_percentile/if_percentile_x/if_percentile_y ou if_bin

Default input:
  dados/exp014a_expanded_scored_input.csv

Exemplos:
  python scripts/exp_014a_expanded_frozen_validation.py

  python scripts/exp_014a_expanded_frozen_validation.py ^
    --input dados/hmo_ml_tb_pix_dataset_v3_features_180d_v1_scored.csv

  python scripts/exp_014a_expanded_frozen_validation.py --preflight-only

Saídas:
  resultados/experimentos/EXP-014A/
    00_run_summary.json
    01_input_contract_report.json
    02_global_metrics.csv
    03_rule_impact.csv
    04_time_block_metrics.csv
    05_standard_bootstrap.csv
    06_stratified_bootstrap.csv
    07_temporal_block_bootstrap.csv
    08_wilson_recall_ci.csv
    09_positive_support_diagnostics.csv
    10_false_negatives.csv
    11_false_positives_sample.csv
    12_policy_used.json
    13_exp014a_report.md
    14_frozen_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "backend").exists() else Path.cwd()

DEFAULT_INPUT = PROJECT_ROOT / "dados" / "exp014a_expanded_scored_input.csv"
DEFAULT_POLICY = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013K" / "12_policy_artifact.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014A"

BASE_PRED_CANDIDATES = [
    "pred_STRICT_RECALL95_SAFE_ONLY",
    "exp013k_base_pred",
    "exp013h_frozen_pred",
    "exp013g_micro_pred",
    "pred_HIGH_RECALL_95",
]

FINAL_PRED_CANDIDATES = [
    "exp013k_residual_fp_pred",
    "exp013l_frozen_pred",
]

REQUIRED_BUSINESS_COLS = [
    "is_fraud",
]

DATE_CANDIDATES = ["event_datetime", "data_pix", "dt_pix"]

RULE_RAW_REQUIREMENTS = {
    "ratio_bin": [["ratio_bin"], ["ratio_valor_media_pagador_90d"]],
    "lgbm_bin": [["lgbm_bin"], ["lgbm_r4_score"], ["r4_score"], ["lgbm_mapped"], ["lgbm_raw"]],
    "score_bin": [["score_bin"], ["score_final"]],
    "vl_bin": [["vl_bin"], ["vl_pix"]],
    "if_bin": [["if_bin"], ["if_percentile"], ["if_percentile_x"], ["if_percentile_y"]],
}


def log(msg: str) -> None:
    print(msg, flush=True)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().split(".")[-1] for c in df.columns]

    if "transaction_id" not in df.columns and "cd_pix" in df.columns:
        df["transaction_id"] = df["cd_pix"]
    if "event_datetime" not in df.columns and "dt_pix" in df.columns:
        df["event_datetime"] = df["dt_pix"]

    if "is_fraud" in df.columns:
        df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)

    for c in BASE_PRED_CANDIDATES + FINAL_PRED_CANDIDATES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    for c in ["event_datetime", "data_pix"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if "transaction_id" in df.columns:
        df["transaction_id"] = df["transaction_id"].astype("string").str.strip()

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


def text(df: pd.DataFrame, names: str | list[str], default: str = "<MISSING>") -> pd.Series:
    col = pick_col(df, names)
    if col is None:
        return pd.Series([default] * len(df), index=df.index, dtype="string")
    return df[col].astype("string").fillna(default).astype(str)


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


def compute_single_bin_if_needed(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col in df.columns:
        return text(df, col)

    if col == "lgbm_bin":
        return qbin_series(num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0), "lgbm", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1])
    if col == "if_bin":
        return qbin_series(num(df, ["if_percentile_x", "if_percentile_y", "if_percentile"], 0.0), "if", [0.32, 0.5, 0.7, 0.85, 0.95])
    if col == "vl_bin":
        return qbin_series(num(df, "vl_pix", 0.0), "vl", [20, 50, 100, 250, 500, 1000, 5000, 10000])
    if col == "score_bin":
        return qbin_series(num(df, "score_final", 0.0), "score", [0.5, 1, 2, 3, 5, 10])
    if col == "qtd_rec_bin":
        return qbin_series(num(df, "qtd_pix_recebidos_180d", 0.0), "qtdrec", [0, 1, 2, 5, 10, 20, 50, 100])
    if col == "valor_rec_bin":
        return qbin_series(num(df, "valor_total_recebido_180d", 0.0), "valrec", [0, 100, 500, 1000, 5000, 10000, 25000])
    if col == "ratio_bin":
        return qbin_series(num(df, "ratio_valor_media_pagador_90d", 0.0), "ratio", [0.05, 0.1, 0.2, 0.5, 1, 2, 5])

    return None


def parse_params_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    try:
        return json.loads(str(raw))
    except Exception:
        return {}


def load_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Policy artifact não encontrado: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "selected_rules" not in obj:
        raise RuntimeError("Policy artifact não contém selected_rules.")
    return obj


def selected_rules_from_policy(policy: dict[str, Any]) -> list[dict[str, Any]]:
    rules = policy.get("selected_rules", [])
    if not isinstance(rules, list) or not rules:
        raise RuntimeError("Policy artifact selected_rules vazio.")
    return rules


def infer_base_col(df: pd.DataFrame, requested: str | None, allow_final_direct: bool) -> tuple[str | None, str]:
    if requested:
        if requested not in df.columns:
            raise RuntimeError(f"--base-pred-col informado, mas coluna não existe: {requested}")
        return requested, "base_col_requested"

    for c in BASE_PRED_CANDIDATES:
        if c in df.columns:
            return c, "base_col_auto"

    if allow_final_direct:
        for c in FINAL_PRED_CANDIDATES:
            if c in df.columns:
                return c, "final_pred_direct"

    return None, "missing"


def contract_report(df: pd.DataFrame, policy: dict[str, Any], base_col: str | None, base_mode: str) -> dict[str, Any]:
    columns = set(df.columns)
    missing_required = [c for c in REQUIRED_BUSINESS_COLS if c not in columns]

    has_date = any(c in columns for c in DATE_CANDIDATES)
    if not has_date:
        missing_required.append("event_datetime_or_data_pix")

    if base_col is None:
        missing_required.append("base_prediction_column")

    selected_rules = selected_rules_from_policy(policy)

    rule_requirements = []
    missing_rule_features = []

    for i, rule in enumerate(selected_rules):
        params = parse_params_json(rule.get("params_json", {}))
        if not params and isinstance(rule.get("params"), dict):
            params = rule["params"]

        combo_cols = params.get("combo_cols", [])
        combo_values = params.get("combo_values", [])

        for c in combo_cols:
            ok = False
            alternatives = []

            if c in RULE_RAW_REQUIREMENTS:
                alternatives = RULE_RAW_REQUIREMENTS[c]
                ok = any(all(a in columns for a in alt) for alt in alternatives)
            else:
                alternatives = [[c]]
                ok = c in columns

            rec = {
                "rule_index": i,
                "description": rule.get("description"),
                "required_logical_col": c,
                "accepted_alternatives": alternatives,
                "ok": ok,
            }
            rule_requirements.append(rec)

            if not ok:
                missing_rule_features.append(rec)

    return {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "base_col": base_col,
        "base_mode": base_mode,
        "has_is_fraud": "is_fraud" in columns,
        "has_date": has_date,
        "missing_required": missing_required,
        "n_policy_rules": len(selected_rules),
        "rule_requirements": rule_requirements,
        "missing_rule_features": missing_rule_features,
        "contract_ok": (len(missing_required) == 0 and len(missing_rule_features) == 0),
    }


def apply_selected_rule_mask(df: pd.DataFrame, rule: dict[str, Any], current_pred: np.ndarray) -> np.ndarray:
    params = parse_params_json(rule.get("params_json", {}))
    if not params and isinstance(rule.get("params"), dict):
        params = rule["params"]

    cols = params.get("combo_cols", [])
    vals = params.get("combo_values", [])

    if not cols:
        desc = str(rule.get("description", ""))
        cols, vals = [], []
        for part in desc.split(" AND "):
            part = part.strip()
            if "=" in part:
                c, v = part.split("=", 1)
                cols.append(c.strip())
                vals.append(v.strip())

    if not cols:
        raise RuntimeError(f"Não consegui parsear regra congelada: {rule}")

    mask = np.ones(len(df), dtype=bool)

    for c, v in zip(cols, vals):
        series = compute_single_bin_if_needed(df, c)
        if series is None:
            raise RuntimeError(f"Coluna necessária para regra congelada ausente e não computável: {c}")
        mask = mask & (series.astype(str) == str(v))

    return mask & (current_pred == 1)


def apply_frozen_policy(df: pd.DataFrame, policy: dict[str, Any], base_col: str, base_mode: str) -> tuple[np.ndarray, pd.DataFrame]:
    # If user supplied/auto detected final prediction directly, do not re-apply rules.
    if base_mode == "final_pred_direct":
        return df[base_col].to_numpy(dtype=int), pd.DataFrame([{
            "rule_index": None,
            "mode": "used_final_prediction_directly",
            "description": f"Using existing final prediction column: {base_col}",
            "tp_loss": None,
            "fp_removed": None,
            "n_removed": None,
        }])

    y = df["is_fraud"].to_numpy(dtype=int)
    pred = df[base_col].to_numpy(dtype=int).copy()
    rows = []

    for idx, rule in enumerate(selected_rules_from_policy(policy)):
        mask = apply_selected_rule_mask(df, rule, pred)
        tp_loss = int(((y == 1) & mask).sum())
        fp_removed = int(((y == 0) & mask).sum())
        n_removed = int(mask.sum())

        pred[mask] = 0

        rows.append({
            "rule_index": idx,
            "mode": "applied_rule",
            "description": rule.get("description"),
            "tp_loss": tp_loss,
            "fp_removed": fp_removed,
            "n_removed": n_removed,
            "params_json": rule.get("params_json", "{}"),
        })

    return pred, pd.DataFrame(rows)


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
    rows = []
    for b in sorted(blocks.dropna().unique()):
        idx = blocks.to_numpy() == b
        part = df.loc[idx].copy()
        m = compute_metrics(part["is_fraud"].to_numpy(dtype=int), pred[idx])
        m.update({
            "policy_name": policy_name,
            "block": int(b),
            "n_rows": int(len(part)),
            "n_frauds": int(part["is_fraud"].sum()),
            "dt_min": str(part["data_pix"].min().date()) if "data_pix" in part.columns and part["data_pix"].notna().any() else (
                str(part["event_datetime"].min().date()) if "event_datetime" in part.columns and part["event_datetime"].notna().any() else None
            ),
            "dt_max": str(part["data_pix"].max().date()) if "data_pix" in part.columns and part["data_pix"].notna().any() else (
                str(part["event_datetime"].max().date()) if "event_datetime" in part.columns and part["event_datetime"].notna().any() else None
            ),
        })
        rows.append(m)
    return pd.DataFrame(rows)


def bootstrap_summary(rows: list[dict[str, Any]], target_recall: float, method: str) -> pd.DataFrame:
    boot = pd.DataFrame(rows)
    out = []
    for metric in ["tp", "fp", "fn", "precision", "recall", "f1", "fpr"]:
        vals = boot[metric].astype(float)
        out.append({
            "method": method,
            "metric": metric,
            "mean": float(vals.mean()),
            "p025": float(vals.quantile(0.025)),
            "p050": float(vals.quantile(0.50)),
            "p975": float(vals.quantile(0.975)),
            "target_recall": target_recall if metric == "recall" else None,
            "p_below_target_recall": float((boot["recall"] < target_recall).mean()) if metric == "recall" else None,
        })
    return pd.DataFrame(out)


def standard_bootstrap(df: pd.DataFrame, pred: np.ndarray, iters: int, seed: int, target_recall: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_all = df["is_fraud"].to_numpy(dtype=int)
    n = len(df)
    rows = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        rows.append(compute_metrics(y_all[idx], pred[idx]))
    return bootstrap_summary(rows, target_recall, "standard_rows")


def stratified_bootstrap(df: pd.DataFrame, pred: np.ndarray, iters: int, seed: int, target_recall: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_all = df["is_fraud"].to_numpy(dtype=int)
    pos_idx = np.where(y_all == 1)[0]
    neg_idx = np.where(y_all == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return pd.DataFrame()

    rows = []
    for _ in range(iters):
        s_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        s_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([s_pos, s_neg])
        rows.append(compute_metrics(y_all[idx], pred[idx]))
    return bootstrap_summary(rows, target_recall, "stratified_class")


def temporal_block_bootstrap(df: pd.DataFrame, pred: np.ndarray, blocks: pd.Series, iters: int, seed: int, target_recall: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_all = df["is_fraud"].to_numpy(dtype=int)
    block_values = sorted(blocks.dropna().unique())
    block_indices = [np.where(blocks.to_numpy() == b)[0] for b in block_values]

    rows = []
    for _ in range(iters):
        selected_blocks = rng.choice(np.arange(len(block_indices)), size=len(block_indices), replace=True)
        idx = np.concatenate([block_indices[i] for i in selected_blocks])
        rows.append(compute_metrics(y_all[idx], pred[idx]))
    return bootstrap_summary(rows, target_recall, "temporal_block")


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) / n) + (z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def min_successes_for_wilson_lower(n: int, target: float, z: float = 1.959963984540054) -> int | None:
    for x in range(0, n + 1):
        lo, _ = wilson_ci(x, n, z)
        if lo >= target:
            return x
    return None


def positive_support_diagnostics(metrics: dict[str, Any], total_frauds: int, target_recall: float, z: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_tp_required = int(math.ceil(target_recall * total_frauds))
    tp_buffer = metrics["tp"] - min_tp_required
    fn_allowed = total_frauds - min_tp_required
    fn_buffer = fn_allowed - metrics["fn"]
    wilson_low, wilson_high = wilson_ci(metrics["tp"], total_frauds, z)
    min_tp_wilson = min_successes_for_wilson_lower(total_frauds, target_recall, z)

    support = pd.DataFrame([{
        "total_frauds": total_frauds,
        "observed_tp": metrics["tp"],
        "observed_fn": metrics["fn"],
        "observed_recall": metrics["recall"],
        "target_recall": target_recall,
        "min_tp_required_for_target": min_tp_required,
        "max_fn_allowed_for_target": fn_allowed,
        "tp_buffer_vs_target": tp_buffer,
        "fn_buffer_vs_target": fn_buffer,
        "wilson_recall_low": wilson_low,
        "wilson_recall_high": wilson_high,
        "min_tp_needed_for_wilson_low_ge_target": min_tp_wilson,
        "additional_tp_needed_for_wilson_low_ge_target": None if min_tp_wilson is None else max(0, min_tp_wilson - metrics["tp"]),
        "positive_support_status": (
            "STRONG" if wilson_low >= target_recall else
            "BUFFERED" if tp_buffer > 0 else
            "ZERO_BUFFER"
        ),
    }])

    wilson = pd.DataFrame([{
        "metric": "recall",
        "successes_tp": metrics["tp"],
        "n_frauds": total_frauds,
        "point_estimate": metrics["recall"],
        "wilson_low": wilson_low,
        "wilson_high": wilson_high,
        "target_recall": target_recall,
        "wilson_low_ge_target": bool(wilson_low >= target_recall),
        "confidence_z": z,
    }])

    return support, wilson


def gate_decision(metrics: dict[str, Any], support: pd.DataFrame, std: pd.DataFrame, strat: pd.DataFrame, block: pd.DataFrame, target_recall: float, reference_fpr: float | None, reference_fpr_multiplier: float) -> dict[str, Any]:
    hard_risks = []
    warnings = []
    diagnostics = []

    if metrics["recall"] < target_recall:
        hard_risks.append("GLOBAL_RECALL_BELOW_TARGET")

    max_fpr = None
    if reference_fpr is not None:
        max_fpr = reference_fpr * reference_fpr_multiplier
        if metrics["fpr"] > max_fpr:
            hard_risks.append("FPR_ABOVE_REFERENCE_MULTIPLIER")

    tp_buffer = int(support["tp_buffer_vs_target"].iloc[0])
    wilson_low = float(support["wilson_recall_low"].iloc[0])
    add_tp_wilson = support["additional_tp_needed_for_wilson_low_ge_target"].iloc[0]

    if tp_buffer <= 0:
        warnings.append("ZERO_TP_BUFFER_AGAINST_TARGET")
    elif tp_buffer < 5:
        warnings.append("LOW_TP_BUFFER_AGAINST_TARGET")

    if wilson_low < target_recall:
        warnings.append("WILSON_RECALL_LOWER_BELOW_TARGET")
        diagnostics.append("Statistical evidence is not yet strong enough for Wilson lower bound >= target.")

    def p_below(df: pd.DataFrame, method: str) -> float | None:
        if df is None or df.empty:
            return None
        r = df[(df["method"] == method) & (df["metric"] == "recall")]
        if r.empty:
            return None
        return float(r["p_below_target_recall"].iloc[0])

    probs = {
        "standard": p_below(std, "standard_rows"),
        "stratified": p_below(strat, "stratified_class"),
        "temporal_block": p_below(block, "temporal_block"),
    }

    for k, v in probs.items():
        if v is not None and v > 0.35:
            warnings.append(f"{k.upper()}_BOOTSTRAP_PROB_BELOW_TARGET_HIGH")
        elif v is not None and v > 0.15:
            warnings.append(f"{k.upper()}_BOOTSTRAP_PROB_BELOW_TARGET_MEDIUM")

    operational_gate = "FAIL" if hard_risks else "PASS"
    statistical_evidence_gate = "PASS" if wilson_low >= target_recall else "INSUFFICIENT_POSITIVE_SUPPORT"

    if hard_risks:
        final_gate = "FAIL"
    elif warnings:
        final_gate = "PASS_WITH_DIAGNOSTIC_WARNINGS"
    else:
        final_gate = "PASS"

    return {
        "final_gate": final_gate,
        "operational_gate": operational_gate,
        "statistical_evidence_gate": statistical_evidence_gate,
        "hard_risks": hard_risks,
        "warnings": warnings,
        "diagnostics": diagnostics,
        "tp_buffer_vs_target": tp_buffer,
        "wilson_recall_low": wilson_low,
        "additional_tp_needed_for_wilson_low_ge_target": None if pd.isna(add_tp_wilson) else int(add_tp_wilson),
        "bootstrap_prob_below_target": probs,
        "reference_fpr": reference_fpr,
        "reference_fpr_multiplier": reference_fpr_multiplier,
        "max_allowed_fpr": max_fpr,
    }


def make_report(summary: dict[str, Any], global_df: pd.DataFrame, support: pd.DataFrame, wilson: pd.DataFrame, blocks: pd.DataFrame, std: pd.DataFrame, strat: pd.DataFrame, block_boot: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014A — Expanded Frozen Validation")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Final gate: `{summary['gate']['final_gate']}`")
    lines.append(f"- Operational gate: `{summary['gate']['operational_gate']}`")
    lines.append(f"- Statistical evidence gate: `{summary['gate']['statistical_evidence_gate']}`")
    lines.append(f"- Objective status: `{summary['objective_status']}`")
    lines.append("")
    lines.append("## Métricas globais")
    lines.append(global_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Suporte positivo")
    lines.append(support.to_markdown(index=False))
    lines.append("")
    lines.append("## Wilson CI")
    lines.append(wilson.to_markdown(index=False))
    lines.append("")
    lines.append("## Blocos temporais")
    lines.append(blocks.to_markdown(index=False))
    lines.append("")
    lines.append("## Bootstrap recall")
    if not std.empty:
        lines.append("### Standard")
        lines.append(std[std["metric"] == "recall"].to_markdown(index=False))
    if not strat.empty:
        lines.append("### Stratified")
        lines.append(strat[strat["metric"] == "recall"].to_markdown(index=False))
    if not block_boot.empty:
        lines.append("### Temporal block")
        lines.append(block_boot[block_boot["metric"] == "recall"].to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretação")
    lines.append("Este experimento não minera novas regras. Ele mede se a política EXP-013K congelada se sustenta com mais suporte positivo e distribuição ampliada.")
    if summary["gate"]["operational_gate"] == "PASS":
        lines.append("A política passou no gate operacional da validação expandida.")
    else:
        lines.append("A política falhou no gate operacional da validação expandida.")
    if summary["gate"]["statistical_evidence_gate"] == "PASS":
        lines.append("O problema de suporte positivo/bootstrap foi resolvido pelo dataset expandido.")
    else:
        lines.append("O suporte positivo ainda é insuficiente para prova estatística forte; usar os novos FNs para EXP-014B Recall Buffer Recovery.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--policy-artifact", default=str(DEFAULT_POLICY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base-pred-col", default=None)
    parser.add_argument("--allow-final-direct", action="store_true", help="Permite usar exp013k_residual_fp_pred direto se não houver base.")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--reference-fpr", type=float, default=0.02017437, help="FPR de referência EXP-013K. Use -1 para ignorar.")
    parser.add_argument("--reference-fpr-multiplier", type=float, default=1.25)
    parser.add_argument("--time-blocks", type=int, default=10)
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-z", type=float, default=1.959963984540054)
    parser.add_argument("--false-positive-sample", type=int, default=5000)
    parser.add_argument("--no-write-predictions", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    policy_path = Path(args.policy_artifact)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014A — Expanded Frozen Validation")
    log("=" * 80)
    log(f"Input: {input_path}")
    log(f"Policy: {policy_path}")
    log(f"Output: {output_dir}")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input não encontrado: {input_path}\n"
            "Crie um CSV expandido scoreado em dados/exp014a_expanded_scored_input.csv "
            "ou informe --input caminho\\arquivo.csv"
        )

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    policy = load_policy(policy_path)
    dump_json(policy, output_dir / "12_policy_used.json")

    base_col, base_mode = infer_base_col(df, args.base_pred_col, args.allow_final_direct)
    contract = contract_report(df, policy, base_col, base_mode)
    dump_json(contract, output_dir / "01_input_contract_report.json")

    if not contract["contract_ok"]:
        log("CONTRATO DE ENTRADA FALHOU.")
        log(json.dumps(contract, ensure_ascii=False, indent=2))
        if args.preflight_only:
            return
        raise RuntimeError("Input não atende ao contrato do EXP-014A. Veja 01_input_contract_report.json.")

    if args.preflight_only:
        log("Preflight OK.")
        return

    pred, rule_impacts = apply_frozen_policy(df, policy, base_col, base_mode)
    y = df["is_fraud"].to_numpy(dtype=int)
    metrics = compute_metrics(y, pred)
    total_frauds = int(y.sum())

    # Optional baseline before EXP-013K rules.
    global_rows = []
    if base_col and base_mode != "final_pred_direct":
        base_metrics = compute_metrics(y, df[base_col].to_numpy(dtype=int))
        global_rows.append({"policy_name": "BASE_BEFORE_EXP013K_RULES", "pred_col": base_col, **base_metrics})
    global_rows.append({"policy_name": "FROZEN_EXP013K_ON_EXPANDED", "pred_col": "exp014a_frozen_pred", **metrics})

    global_df = pd.DataFrame(global_rows)
    global_df.to_csv(output_dir / "02_global_metrics.csv", index=False)

    rule_impacts.to_csv(output_dir / "03_rule_impact.csv", index=False)

    predictions = df.copy()
    predictions["exp014a_frozen_pred"] = pred

    if not args.no_write_predictions:
        predictions.to_csv(output_dir / "14_frozen_predictions.csv", index=False)

    predictions[(predictions["is_fraud"] == 1) & (predictions["exp014a_frozen_pred"] == 0)].to_csv(output_dir / "10_false_negatives.csv", index=False)

    fps = predictions[(predictions["is_fraud"] == 0) & (predictions["exp014a_frozen_pred"] == 1)]
    if len(fps) > args.false_positive_sample:
        fps = fps.sample(args.false_positive_sample, random_state=args.seed)
    fps.to_csv(output_dir / "11_false_positives_sample.csv", index=False)

    blocks = make_time_blocks(df, args.time_blocks)
    block_df = block_metrics(df, pred, blocks, "FROZEN_EXP013K_ON_EXPANDED")
    block_df.to_csv(output_dir / "04_time_block_metrics.csv", index=False)

    log("[1/4] Bootstrap padrão...")
    std_boot = standard_bootstrap(df, pred, args.bootstrap_iters, args.seed, args.target_recall)
    std_boot.to_csv(output_dir / "05_standard_bootstrap.csv", index=False)

    log("[2/4] Bootstrap estratificado...")
    strat_boot = stratified_bootstrap(df, pred, args.bootstrap_iters, args.seed + 101, args.target_recall)
    strat_boot.to_csv(output_dir / "06_stratified_bootstrap.csv", index=False)

    log("[3/4] Bootstrap temporal por blocos...")
    block_boot = temporal_block_bootstrap(df, pred, blocks, args.bootstrap_iters, args.seed + 202, args.target_recall)
    block_boot.to_csv(output_dir / "07_temporal_block_bootstrap.csv", index=False)

    log("[4/4] Wilson e suporte positivo...")
    support_df, wilson_df = positive_support_diagnostics(metrics, total_frauds, args.target_recall, args.confidence_z)
    wilson_df.to_csv(output_dir / "08_wilson_recall_ci.csv", index=False)
    support_df.to_csv(output_dir / "09_positive_support_diagnostics.csv", index=False)

    reference_fpr = None if args.reference_fpr is None or args.reference_fpr < 0 else args.reference_fpr
    gate = gate_decision(metrics, support_df, std_boot, strat_boot, block_boot, args.target_recall, reference_fpr, args.reference_fpr_multiplier)

    objective_status = f"{gate['final_gate']}_TARGET_RECALL_" + ("MET" if metrics["recall"] >= args.target_recall else "NOT_MET")
    objective_status += "_STAT_SUPPORT_" + gate["statistical_evidence_gate"]

    summary = {
        "experiment": "EXP-014A",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "policy_path": str(policy_path),
        "base_col": base_col,
        "base_mode": base_mode,
        "n_rows": int(len(df)),
        "total_frauds": total_frauds,
        "target_recall": args.target_recall,
        "reference_fpr": reference_fpr,
        "reference_fpr_multiplier": args.reference_fpr_multiplier,
        "metrics": metrics,
        "gate": gate,
        "support_diagnostics": support_df.to_dict(orient="records")[0],
        "rule_impact_summary": rule_impacts.to_dict(orient="records"),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, global_df, support_df, wilson_df, block_df, std_boot, strat_boot, block_boot)
    (output_dir / "13_exp014a_report.md").write_text(report, encoding="utf-8")

    log("")
    log("=" * 80)
    log("EXP-014A CONCLUÍDO")
    log("=" * 80)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_input_contract_report.json",
        output_dir / "02_global_metrics.csv",
        output_dir / "03_rule_impact.csv",
        output_dir / "04_time_block_metrics.csv",
        output_dir / "05_standard_bootstrap.csv",
        output_dir / "06_stratified_bootstrap.csv",
        output_dir / "07_temporal_block_bootstrap.csv",
        output_dir / "08_wilson_recall_ci.csv",
        output_dir / "09_positive_support_diagnostics.csv",
        output_dir / "13_exp014a_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
