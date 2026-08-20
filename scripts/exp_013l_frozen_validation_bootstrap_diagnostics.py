#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-013L — Frozen Validation + Bootstrap Diagnostics

Objetivo:
  Confirmar a política vencedora do EXP-013K sem nova mineração/reotimização:

      residual_fp_mined_tp0_policy
      TP=118, FP=199, FN=6, recall=95.16%, precision=37.22%

  E tratar tecnicamente os avisos repetidos do bootstrap.

Problema do bootstrap:
  Com apenas 124 fraudes e TP=118, o recall global fica exatamente acima do piso:
      ceil(0.95 * 124) = 118
  Portanto, o buffer de TP é 0. Qualquer incerteza amostral tende a gerar
  probabilidade relevante de recall < 95% em bootstrap. Isso não significa
  necessariamente que a política piorou; significa que a política opera sem
  folga estatística.

O que este script faz além do bootstrap padrão:
  1. Validação congelada da política EXP-013K.
  2. Bootstrap padrão, para comparabilidade com rodadas anteriores.
  3. Bootstrap estratificado por classe, preservando quantidade de fraudes/não-fraudes.
  4. Bootstrap temporal por blocos.
  5. Intervalo de confiança Wilson para recall.
  6. Diagnóstico de suporte positivo:
       - TP buffer contra target
       - FN buffer
       - mínimo de TP necessário
       - quantos acertos seriam necessários para ter Wilson lower >= 95%
  7. Gate separado:
       - hard gate operacional: métricas observadas passam ou não;
       - statistical evidence gate: há ou não evidência estatística forte;
       - bootstrap warnings viram diagnóstico explícito, não ruído repetido.

Entradas default:
  resultados/experimentos/EXP-013K/07_selected_predictions.csv
  resultados/experimentos/EXP-013K/12_policy_artifact.json

Uso:
  python scripts/exp_013l_frozen_validation_bootstrap_diagnostics.py

Execução mais rápida:
  python scripts/exp_013l_frozen_validation_bootstrap_diagnostics.py --bootstrap-iters 200

Validação externa:
  python scripts/exp_013l_frozen_validation_bootstrap_diagnostics.py --input caminho\\novo_arquivo.csv

Saídas:
  resultados/experimentos/EXP-013L/
    00_run_summary.json
    01_global_metrics.csv
    02_time_block_metrics.csv
    03_standard_bootstrap.csv
    04_stratified_bootstrap.csv
    05_block_bootstrap.csv
    06_wilson_recall_ci.csv
    07_positive_support_diagnostics.csv
    08_frozen_predictions.csv
    09_false_negatives.csv
    10_false_positives.csv
    11_gate_report.md
    12_policy_used.json
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

DEFAULT_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013K" / "07_selected_predictions.csv"
DEFAULT_POLICY = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013K" / "12_policy_artifact.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013L"

PRED_COL_CANDIDATES = [
    "exp013k_residual_fp_pred",
    "pred_STRICT_RECALL95_SAFE_ONLY",
    "exp013h_frozen_pred",
    "exp013g_micro_pred",
]


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

    if "is_fraud" not in df.columns:
        raise RuntimeError("Coluna is_fraud ausente.")

    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)

    for c in PRED_COL_CANDIDATES + ["exp013k_base_pred"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

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


def load_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Policy artifact não encontrado: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "selected_rules" not in obj:
        raise RuntimeError("Policy artifact não contém selected_rules.")
    return obj


def parse_params_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    try:
        return json.loads(str(raw))
    except Exception:
        return {}


def apply_selected_rule_mask(df: pd.DataFrame, rule: dict[str, Any], base_pred: np.ndarray) -> np.ndarray:
    params = parse_params_json(rule.get("params_json", {}))
    if not params:
        params = rule.get("params", {}) if isinstance(rule.get("params"), dict) else {}

    cols = params.get("combo_cols", [])
    vals = params.get("combo_values", [])

    if not cols:
        # Fallback parse "a=b AND c=d" from description.
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
        if c not in df.columns:
            # Try to compute known bins if missing.
            computed = compute_single_bin_if_needed(df, c)
            if computed is None:
                raise RuntimeError(f"Coluna necessária para regra congelada ausente: {c}")
            series = computed
        else:
            series = text(df, c)

        mask = mask & (series.astype(str) == str(v))

    return mask & (base_pred == 1)


def compute_single_bin_if_needed(df: pd.DataFrame, col: str) -> pd.Series | None:
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


def apply_frozen_policy(df: pd.DataFrame, policy: dict[str, Any], pred_col: str | None) -> tuple[np.ndarray, pd.DataFrame, str]:
    # Prefer ready-made EXP-013K prediction if it exists.
    if pred_col and pred_col in df.columns:
        return df[pred_col].to_numpy(dtype=int), pd.DataFrame(), pred_col

    if "exp013k_residual_fp_pred" in df.columns:
        return df["exp013k_residual_fp_pred"].to_numpy(dtype=int), pd.DataFrame(), "exp013k_residual_fp_pred"

    # Reapply on base if needed.
    base_col = policy.get("base_pred_col") or "pred_STRICT_RECALL95_SAFE_ONLY"
    if base_col not in df.columns:
        if "exp013k_base_pred" in df.columns:
            base_col = "exp013k_base_pred"
        else:
            raise RuntimeError("Input não contém exp013k_residual_fp_pred nem coluna base para reaplicar a política.")

    y = df["is_fraud"].to_numpy(dtype=int)
    pred = df[base_col].to_numpy(dtype=int).copy()
    rows = []

    for idx, rule in enumerate(policy.get("selected_rules", [])):
        mask = apply_selected_rule_mask(df, rule, pred)
        tp_loss = int(((y == 1) & mask).sum())
        fp_removed = int(((y == 0) & mask).sum())
        pred[mask] = 0

        rows.append({
            "rule_index": idx,
            "description": rule.get("description"),
            "tp_loss": tp_loss,
            "fp_removed": fp_removed,
            "n_removed": int(mask.sum()),
        })

    return pred, pd.DataFrame(rows), f"reapplied_from_{base_col}"


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
            "dt_min": str(part["data_pix"].min().date()) if "data_pix" in part.columns and part["data_pix"].notna().any() else None,
            "dt_max": str(part["data_pix"].max().date()) if "data_pix" in part.columns and part["data_pix"].notna().any() else None,
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
    rows = []

    for _ in range(iters):
        s_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        s_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([s_pos, s_neg])
        rows.append(compute_metrics(y_all[idx], pred[idx]))

    return bootstrap_summary(rows, target_recall, "stratified_class")


def block_bootstrap(df: pd.DataFrame, pred: np.ndarray, blocks: pd.Series, iters: int, seed: int, target_recall: float) -> pd.DataFrame:
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


def positive_support_diagnostics(metrics: dict[str, Any], total_frauds: int, target_recall: float, confidence_z: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_tp_required = int(math.ceil(target_recall * total_frauds))
    tp_buffer = metrics["tp"] - min_tp_required
    fn_allowed = total_frauds - min_tp_required
    fn_buffer = fn_allowed - metrics["fn"]
    wilson_low, wilson_high = wilson_ci(metrics["tp"], total_frauds, confidence_z)
    min_tp_wilson = min_successes_for_wilson_lower(total_frauds, target_recall, confidence_z)

    diag = pd.DataFrame([{
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
        "interpretation": (
            "Observed policy passes target but has zero TP buffer; repeated bootstrap warnings are expected."
            if tp_buffer == 0 else
            "Observed policy has positive TP buffer."
        ),
    }])

    wilson_df = pd.DataFrame([{
        "metric": "recall",
        "successes_tp": metrics["tp"],
        "n_frauds": total_frauds,
        "point_estimate": metrics["recall"],
        "wilson_low": wilson_low,
        "wilson_high": wilson_high,
        "target_recall": target_recall,
        "wilson_low_ge_target": bool(wilson_low >= target_recall),
        "confidence_z": confidence_z,
    }])

    return diag, wilson_df


def gate_decision(
    metrics: dict[str, Any],
    total_frauds: int,
    target_recall: float,
    reference_fp: int | None,
    std_boot: pd.DataFrame,
    strat_boot: pd.DataFrame,
    block_boot: pd.DataFrame,
    support_diag: pd.DataFrame,
) -> dict[str, Any]:
    min_tp_required = int(math.ceil(target_recall * total_frauds))
    hard_risks = []
    warnings = []
    diagnostics = []

    if metrics["tp"] < min_tp_required or metrics["recall"] < target_recall:
        hard_risks.append("GLOBAL_RECALL_BELOW_TARGET")

    if reference_fp is not None and metrics["fp"] > reference_fp:
        hard_risks.append("FP_ABOVE_REFERENCE")

    tp_buffer = int(support_diag["tp_buffer_vs_target"].iloc[0])
    wilson_low = float(support_diag["wilson_recall_low"].iloc[0])
    add_tp_wilson = support_diag["additional_tp_needed_for_wilson_low_ge_target"].iloc[0]

    if tp_buffer <= 0:
        warnings.append("ZERO_TP_BUFFER_AGAINST_TARGET")
        diagnostics.append("Bootstrap recall warnings are expected because observed TP is exactly the minimum required for recall>=target.")

    if wilson_low < target_recall:
        warnings.append("WILSON_RECALL_LOWER_BELOW_TARGET")
        diagnostics.append("Current positive support is insufficient to statistically prove recall>=target at 95% confidence.")

    def boot_prob(df: pd.DataFrame, method: str) -> float | None:
        r = df[(df["method"] == method) & (df["metric"] == "recall")]
        if r.empty:
            return None
        return float(r["p_below_target_recall"].iloc[0])

    p_std = boot_prob(std_boot, "standard_rows")
    p_strat = boot_prob(strat_boot, "stratified_class")
    p_block = boot_prob(block_boot, "temporal_block")

    for name, p in [("STANDARD_BOOTSTRAP", p_std), ("STRATIFIED_BOOTSTRAP", p_strat), ("BLOCK_BOOTSTRAP", p_block)]:
        if p is not None and p > 0.35:
            warnings.append(f"{name}_PROB_BELOW_TARGET_HIGH")

    operational_gate = "FAIL" if hard_risks else "PASS"
    statistical_evidence_gate = "PASS" if (wilson_low >= target_recall and tp_buffer > 0) else "INSUFFICIENT_POSITIVE_SUPPORT"

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
        "standard_bootstrap_prob_below_target": p_std,
        "stratified_bootstrap_prob_below_target": p_strat,
        "block_bootstrap_prob_below_target": p_block,
    }


def make_report(summary: dict[str, Any], global_df: pd.DataFrame, support_df: pd.DataFrame, wilson_df: pd.DataFrame, std: pd.DataFrame, strat: pd.DataFrame, block: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-013L — Frozen Validation + Bootstrap Diagnostics")
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
    lines.append("## Diagnóstico do bootstrap")
    lines.append("Os avisos de bootstrap não foram ignorados. Eles foram decompostos em suporte positivo, Wilson CI e três formas de bootstrap.")
    lines.append("")
    lines.append("### Suporte positivo")
    lines.append(support_df.to_markdown(index=False))
    lines.append("")
    lines.append("### Wilson CI")
    lines.append(wilson_df.to_markdown(index=False))
    lines.append("")
    lines.append("### Bootstrap padrão")
    lines.append(std[std["metric"] == "recall"].to_markdown(index=False))
    lines.append("")
    lines.append("### Bootstrap estratificado por classe")
    lines.append(strat[strat["metric"] == "recall"].to_markdown(index=False))
    lines.append("")
    lines.append("### Bootstrap temporal por blocos")
    lines.append(block[block["metric"] == "recall"].to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretação")
    if summary["gate"]["operational_gate"] == "PASS":
        lines.append("A política congelada passou operacionalmente: mantém recall observado >= alvo e FP <= referência.")
    else:
        lines.append("A política congelada falhou operacionalmente; não promover.")
    lines.append("")
    if summary["gate"]["statistical_evidence_gate"] == "INSUFFICIENT_POSITIVE_SUPPORT":
        lines.append("Os alertas de bootstrap persistem porque a política opera exatamente no piso de TP necessário. Com 124 fraudes e TP=118, o buffer de TP é zero. Portanto, o bootstrap está informando falta de folga estatística, não necessariamente piora da política.")
        lines.append("Para transformar esse alerta em PASS estatístico forte, há dois caminhos: recuperar TPs/FNs ou validar em uma janela externa com mais fraudes mantendo recall observado acima do alvo.")
    else:
        lines.append("O suporte estatístico é suficiente pelo critério Wilson configurado.")
    lines.append("")
    lines.append("## Decisão recomendada")
    lines.append("Usar este resultado como validação congelada para avançar a patch shadow configurável, mas manter monitoramento explícito de recall por janela e TP buffer.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--policy-artifact", default=str(DEFAULT_POLICY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pred-col", default=None, help="Se informado e existir, usa esta predição diretamente.")
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--reference-fp", type=int, default=199, help="FP máximo esperado. Use -1 para ignorar.")
    parser.add_argument("--time-blocks", type=int, default=5)
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-z", type=float, default=1.959963984540054)
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    policy_path = Path(args.policy_artifact)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-013L — Frozen Validation + Bootstrap Diagnostics")
    log("=" * 80)
    log(f"Input: {input_path}")
    log(f"Policy: {policy_path}")
    log(f"Output: {output_dir}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    policy = load_policy(policy_path)
    dump_json(policy, output_dir / "12_policy_used.json")

    pred, rule_impacts, applied_mode = apply_frozen_policy(df, policy, args.pred_col)
    y = df["is_fraud"].to_numpy(dtype=int)
    metrics = compute_metrics(y, pred)
    total_frauds = int(y.sum())
    reference_fp = None if args.reference_fp is None or args.reference_fp < 0 else args.reference_fp

    global_df = pd.DataFrame([{
        "policy_name": "FROZEN_EXP013K_RESIDUAL_FP_MINED",
        "applied_mode": applied_mode,
        **metrics,
    }])
    global_df.to_csv(output_dir / "01_global_metrics.csv", index=False)

    predictions = df.copy()
    predictions["exp013l_frozen_pred"] = pred
    predictions.to_csv(output_dir / "08_frozen_predictions.csv", index=False)
    predictions[(predictions["is_fraud"] == 1) & (predictions["exp013l_frozen_pred"] == 0)].to_csv(output_dir / "09_false_negatives.csv", index=False)
    predictions[(predictions["is_fraud"] == 0) & (predictions["exp013l_frozen_pred"] == 1)].to_csv(output_dir / "10_false_positives.csv", index=False)

    if not rule_impacts.empty:
        rule_impacts.to_csv(output_dir / "rule_reapplication_impacts.csv", index=False)

    blocks = make_time_blocks(df, args.time_blocks)
    block_df = block_metrics(df, pred, blocks, "FROZEN_EXP013K_RESIDUAL_FP_MINED")
    block_df.to_csv(output_dir / "02_time_block_metrics.csv", index=False)

    log("[1/4] Bootstrap padrão...")
    std_boot = standard_bootstrap(df, pred, args.bootstrap_iters, args.seed, args.target_recall)
    std_boot.to_csv(output_dir / "03_standard_bootstrap.csv", index=False)

    log("[2/4] Bootstrap estratificado por classe...")
    strat_boot = stratified_bootstrap(df, pred, args.bootstrap_iters, args.seed + 101, args.target_recall)
    strat_boot.to_csv(output_dir / "04_stratified_bootstrap.csv", index=False)

    log("[3/4] Bootstrap temporal por blocos...")
    block_boot = block_bootstrap(df, pred, blocks, args.bootstrap_iters, args.seed + 202, args.target_recall)
    block_boot.to_csv(output_dir / "05_block_bootstrap.csv", index=False)

    log("[4/4] Diagnóstico Wilson/suporte positivo...")
    support_df, wilson_df = positive_support_diagnostics(metrics, total_frauds, args.target_recall, args.confidence_z)
    wilson_df.to_csv(output_dir / "06_wilson_recall_ci.csv", index=False)
    support_df.to_csv(output_dir / "07_positive_support_diagnostics.csv", index=False)

    gate = gate_decision(
        metrics=metrics,
        total_frauds=total_frauds,
        target_recall=args.target_recall,
        reference_fp=reference_fp,
        std_boot=std_boot,
        strat_boot=strat_boot,
        block_boot=block_boot,
        support_diag=support_df,
    )

    objective_status = f"{gate['final_gate']}_TARGET_RECALL_" + ("MET" if metrics["recall"] >= args.target_recall else "NOT_MET")
    objective_status += "_REFERENCE_FP_" + ("MET" if reference_fp is None or metrics["fp"] <= reference_fp else "NOT_MET")
    objective_status += "_BOOTSTRAP_DIAGNOSTIC_EXPLAINED"

    summary = {
        "experiment": "EXP-013L",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "policy_path": str(policy_path),
        "applied_mode": applied_mode,
        "n_rows": int(len(df)),
        "total_frauds": total_frauds,
        "target_recall": args.target_recall,
        "reference_fp": reference_fp,
        "metrics": metrics,
        "gate": gate,
        "support_diagnostics": support_df.to_dict(orient="records")[0],
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, global_df, support_df, wilson_df, std_boot, strat_boot, block_boot)
    (output_dir / "11_gate_report.md").write_text(report, encoding="utf-8")

    log("")
    log("=" * 80)
    log("EXP-013L CONCLUÍDO")
    log("=" * 80)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_global_metrics.csv",
        output_dir / "02_time_block_metrics.csv",
        output_dir / "03_standard_bootstrap.csv",
        output_dir / "04_stratified_bootstrap.csv",
        output_dir / "05_block_bootstrap.csv",
        output_dir / "06_wilson_recall_ci.csv",
        output_dir / "07_positive_support_diagnostics.csv",
        output_dir / "11_gate_report.md",
        output_dir / "12_policy_used.json",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
