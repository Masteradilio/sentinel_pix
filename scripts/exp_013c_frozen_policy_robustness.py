#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-013C — Frozen Statistical Veto Robustness Validation

Objetivo:
  Validar se a política campeã do EXP-013B-R1 é aprendizado estatístico
  verdadeiro ou se há indício de overfitting na amostra usada para buscá-la.

Princípios:
  1. Política CONGELADA: não faz nova busca e não reotimiza thresholds.
  2. Aplica exatamente as regras do EXP-013B-R1.
  3. Mede robustez por:
     - métricas globais;
     - validação por blocos temporais;
     - bootstrap de confiança;
     - ablação leave-one-rule-out;
     - impacto individual/sequencial das regras;
     - stress test leve de thresholds;
     - relatório de risco de overfitting.
  4. Pode ser usado em outra amostra/split no futuro:
       --input caminho/para/outro_comparison_by_transaction.csv

Entradas default:
  resultados/experimentos/EXP-012E/04_comparison_by_transaction.csv
  resultados/experimentos/EXP-013B-R1/10_policy_artifact.json

Execução:
  python scripts/exp_013c_frozen_policy_robustness.py

Execução mais rápida:
  python scripts/exp_013c_frozen_policy_robustness.py --bootstrap-iters 200

Saídas:
  resultados/experimentos/EXP-013C/
    00_run_summary.json
    01_global_metrics.csv
    02_time_block_metrics.csv
    03_bootstrap_confidence_intervals.csv
    04_rule_individual_impact.csv
    05_rule_leave_one_out.csv
    06_rule_sequential_impact.csv
    07_threshold_stress_test.csv
    08_frozen_policy_predictions.csv
    09_frozen_policy_false_negatives.csv
    10_frozen_policy_false_positives.csv
    11_overfitting_risk_report.md
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

DEFAULT_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-012E" / "04_comparison_by_transaction.csv"
DEFAULT_POLICY = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013B-R1" / "10_policy_artifact.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013C"

FLAGGED_DECISIONS = {"CONFIRMAR", "BLOQUEAR"}


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

    if "shadow_exp012d_flagged" not in df.columns:
        for c in ["exp012d_pred", "r4_pred", "lgbm_r4_pred"]:
            if c in df.columns:
                df["shadow_exp012d_flagged"] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
                break

    if "shadow_exp012d_flagged" not in df.columns:
        raise RuntimeError("Não encontrei shadow_exp012d_flagged/exp012d_pred/r4_pred/lgbm_r4_pred.")

    df["shadow_exp012d_flagged"] = pd.to_numeric(df["shadow_exp012d_flagged"], errors="coerce").fillna(0).astype(int)

    if "runtime_flagged" not in df.columns:
        if "decisao" in df.columns:
            df["runtime_flagged"] = df["decisao"].astype(str).str.upper().isin(FLAGGED_DECISIONS).astype(int)
        else:
            df["runtime_flagged"] = 0
    df["runtime_flagged"] = pd.to_numeric(df["runtime_flagged"], errors="coerce").fillna(0).astype(int)

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


def boolish(df: pd.DataFrame, names: str | list[str], default: bool = False) -> pd.Series:
    col = pick_col(df, names)
    if col is None:
        return pd.Series([default] * len(df), index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(default)
    return s.astype(str).str.upper().isin({"1", "1.0", "TRUE", "T", "SIM", "YES", "Y"})


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


def strong_preserve_mask(df: pd.DataFrame) -> np.ndarray:
    se_score = num(df, ["se_score_x", "se_score_y", "se_score"], 0.0)
    se_count = num(df, ["se_patterns_count", "se_pattern_count"], 0.0)
    beh_score = num(df, ["beh_score", "behavioral_score"], 0.0)
    beh_count = num(df, ["beh_factors_count", "behavioral_risk_factor_count"], 0.0)
    runtime = num(df, "runtime_flagged", 0.0)
    cascade = boolish(df, "cascade_triggered", False)
    decisao = text(df, "decisao", "").str.upper()

    return (
        (se_score >= 65)
        | (se_count >= 2)
        | (beh_score >= 45)
        | (beh_count >= 2)
        | (runtime >= 1)
        | decisao.isin(FLAGGED_DECISIONS)
        | cascade
    ).to_numpy(dtype=bool)


def load_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Policy artifact não encontrado: {path}")

    obj = json.loads(path.read_text(encoding="utf-8"))
    if "rules" not in obj or not obj["rules"]:
        raise RuntimeError("Policy artifact sem regras.")
    return obj


def parse_params(rule: dict[str, Any]) -> dict[str, Any]:
    raw = rule.get("params_json", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def get_lgbm_score(df: pd.DataFrame) -> pd.Series:
    return num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0)


def get_if_percentile(df: pd.DataFrame) -> pd.Series:
    return num(df, ["if_percentile_x", "if_percentile_y", "if_percentile"], 0.0)


def apply_rule_mask(df: pd.DataFrame, rule: dict[str, Any], threshold_multiplier: float = 1.0) -> np.ndarray:
    family = str(rule.get("family", ""))
    params = parse_params(rule)
    preserve = strong_preserve_mask(df)

    # All rules are vetoes applied only after shadow positive; caller enforces this too.
    mask = np.zeros(len(df), dtype=bool)

    if family == "numeric":
        feature = params.get("feature")
        op = params.get("op")
        th = float(params.get("threshold"))
        th_adj = th * threshold_multiplier
        vals = num(df, feature, np.nan)
        if op == "lt":
            mask = (vals < th_adj).to_numpy(dtype=bool)
        elif op == "gt":
            mask = (vals > th_adj).to_numpy(dtype=bool)
        else:
            raise ValueError(f"Operador numeric desconhecido: {op}")

        if bool(params.get("preserve", False)):
            mask = mask & (~preserve)

    elif family == "segment":
        cols = params.get("segment_cols", [])
        values = params.get("segment_values", [])
        mask = np.ones(len(df), dtype=bool)
        for c, v in zip(cols, values):
            mask = mask & (text(df, c) == str(v)).to_numpy(dtype=bool)
        if bool(params.get("preserve", False)):
            mask = mask & (~preserve)

    elif family == "segment_lgbm":
        cols = params.get("segment_cols", [])
        values = params.get("segment_values", [])
        lgbm_lt = float(params.get("lgbm_lt")) * threshold_multiplier
        mask = np.ones(len(df), dtype=bool)
        for c, v in zip(cols, values):
            mask = mask & (text(df, c) == str(v)).to_numpy(dtype=bool)
        mask = mask & (get_lgbm_score(df) < lgbm_lt).to_numpy(dtype=bool)
        if bool(params.get("preserve", False)):
            mask = mask & (~preserve)

    elif family == "receiver_value_established":
        lgbm_lt = float(params.get("lgbm_lt")) * threshold_multiplier
        receiver_value_gt = float(params.get("receiver_value_gt")) * threshold_multiplier
        mask = (
            (get_lgbm_score(df) < lgbm_lt)
            & (num(df, "valor_total_recebido_180d", 0.0) > receiver_value_gt)
        ).to_numpy(dtype=bool)
        # The R1 rule generation used strong-preserve for this family.
        mask = mask & (~preserve)

    elif family == "quiet_veto":
        lgbm_lt = float(params.get("lgbm_lt")) * threshold_multiplier
        if_lt = float(params.get("if_lt")) * threshold_multiplier
        se_score = num(df, ["se_score_x", "se_score_y", "se_score"], 0.0)
        se_count = num(df, ["se_patterns_count", "se_pattern_count"], 0.0)
        beh_score = num(df, ["beh_score", "behavioral_score"], 0.0)
        beh_count = num(df, ["beh_factors_count", "behavioral_risk_factor_count"], 0.0)

        mask = (
            (get_lgbm_score(df) < lgbm_lt)
            & (get_if_percentile(df) < if_lt)
            & (se_score <= 20)
            & (se_count < 2)
            & (beh_score <= 25)
            & (beh_count < 2)
        ).to_numpy(dtype=bool)
        mask = mask & (~preserve)

    else:
        raise ValueError(f"Família de regra não implementada: {family}")

    return mask.astype(bool)


def apply_policy(df: pd.DataFrame, policy: dict[str, Any], excluded_rule_ids: set[str] | None = None, only_rule_id: str | None = None, threshold_multiplier: float = 1.0) -> tuple[np.ndarray, pd.DataFrame]:
    excluded_rule_ids = excluded_rule_ids or set()
    shadow = df["shadow_exp012d_flagged"].to_numpy(dtype=int).astype(bool)

    veto_any = np.zeros(len(df), dtype=bool)
    rule_rows = []

    for idx, rule in enumerate(policy["rules"]):
        rule_id = str(rule.get("rule_id", f"rule_{idx}"))
        if rule_id in excluded_rule_ids:
            continue
        if only_rule_id is not None and rule_id != only_rule_id:
            continue

        mask = apply_rule_mask(df, rule, threshold_multiplier=threshold_multiplier)
        mask = mask & shadow

        y = df["is_fraud"].to_numpy(dtype=int)
        tp_loss = int(((y == 1) & mask).sum())
        fp_removed = int(((y == 0) & mask).sum())

        rule_rows.append({
            "rule_id": rule_id,
            "family": rule.get("family"),
            "description": rule.get("description"),
            "tp_loss": tp_loss,
            "fp_removed": fp_removed,
            "n_removed": int(mask.sum()),
        })

        veto_any = veto_any | mask

    pred = shadow.astype(int)
    pred[veto_any] = 0
    return pred, pd.DataFrame(rule_rows)


def bootstrap_ci(df: pd.DataFrame, policy: dict[str, Any], iters: int, seed: int, target_recall: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(df)
    y = df["is_fraud"].to_numpy(dtype=int)

    rows = []
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        sample = df.iloc[idx].reset_index(drop=True)
        pred, _ = apply_policy(sample, policy)
        m = compute_metrics(sample["is_fraud"].to_numpy(dtype=int), pred)
        rows.append(m)

    boot = pd.DataFrame(rows)
    ci_rows = []
    for metric in ["tp", "fp", "fn", "precision", "recall", "f1", "fpr"]:
        vals = boot[metric].astype(float)
        ci_rows.append({
            "metric": metric,
            "mean": float(vals.mean()),
            "p025": float(vals.quantile(0.025)),
            "p050": float(vals.quantile(0.50)),
            "p975": float(vals.quantile(0.975)),
            "target_recall": target_recall if metric == "recall" else None,
            "p_below_target_recall": float((boot["recall"] < target_recall).mean()) if metric == "recall" else None,
        })
    return pd.DataFrame(ci_rows)


def make_time_blocks(df: pd.DataFrame, n_blocks: int) -> pd.Series:
    # Prefer real date if available.
    if "data_pix" in df.columns and df["data_pix"].notna().any():
        dates = pd.to_datetime(df["data_pix"], errors="coerce")
    elif "event_datetime" in df.columns and df["event_datetime"].notna().any():
        dates = pd.to_datetime(df["event_datetime"], errors="coerce")
    else:
        # Stable fallback by row order.
        return pd.qcut(np.arange(len(df)), q=min(n_blocks, len(df)), labels=False, duplicates="drop").astype(int)

    tmp = pd.DataFrame({"date": dates, "_idx": np.arange(len(df))}).sort_values(["date", "_idx"])
    tmp["block"] = pd.qcut(np.arange(len(tmp)), q=min(n_blocks, len(tmp)), labels=False, duplicates="drop")
    out = pd.Series(index=tmp["_idx"].values, data=tmp["block"].values).sort_index()
    return out.astype(int)


def time_block_metrics(df: pd.DataFrame, policy: dict[str, Any], n_blocks: int) -> pd.DataFrame:
    blocks = make_time_blocks(df, n_blocks)
    rows = []
    for b in sorted(blocks.dropna().unique()):
        part = df.loc[blocks == b].copy()
        pred, _ = apply_policy(part, policy)
        m = compute_metrics(part["is_fraud"].to_numpy(dtype=int), pred)
        m.update({
            "block": int(b),
            "n_rows": int(len(part)),
            "n_frauds": int(part["is_fraud"].sum()),
            "dt_min": str(part["data_pix"].min().date()) if "data_pix" in part.columns and part["data_pix"].notna().any() else None,
            "dt_max": str(part["data_pix"].max().date()) if "data_pix" in part.columns and part["data_pix"].notna().any() else None,
        })
        rows.append(m)
    return pd.DataFrame(rows)


def stress_test(df: pd.DataFrame, policy: dict[str, Any], multipliers: list[float]) -> pd.DataFrame:
    rows = []
    for mult in multipliers:
        pred, _ = apply_policy(df, policy, threshold_multiplier=mult)
        m = compute_metrics(df["is_fraud"].to_numpy(dtype=int), pred)
        m["threshold_multiplier"] = mult
        rows.append(m)
    return pd.DataFrame(rows)


def risk_assessment(global_metrics: dict[str, Any], time_blocks: pd.DataFrame, boot: pd.DataFrame, loo: pd.DataFrame, stress: pd.DataFrame, target_recall: float) -> dict[str, Any]:
    risks = []

    if global_metrics["recall"] < target_recall:
        risks.append("GLOBAL_RECALL_BELOW_TARGET")

    if not time_blocks.empty:
        min_block_recall = float(time_blocks["recall"].min())
        if min_block_recall < target_recall:
            risks.append("SOME_TIME_BLOCK_RECALL_BELOW_TARGET")
    else:
        min_block_recall = None

    recall_ci = boot[boot["metric"] == "recall"]
    if not recall_ci.empty:
        recall_p025 = float(recall_ci["p025"].iloc[0])
        p_below = float(recall_ci["p_below_target_recall"].iloc[0])
        if recall_p025 < target_recall:
            risks.append("BOOTSTRAP_RECALL_CI_LOWER_BELOW_TARGET")
        if p_below > 0.10:
            risks.append("BOOTSTRAP_TARGET_FAILURE_PROB_GT_10PCT")
    else:
        recall_p025 = None
        p_below = None

    if not loo.empty:
        # If removing one rule improves recall with modest FP cost, champion may be too aggressive.
        safe_loo = loo[loo["recall"] >= target_recall]
        if len(safe_loo) > 0:
            best_conservative = safe_loo.sort_values(["fn", "fp"], ascending=[True, True]).iloc[0].to_dict()
        else:
            best_conservative = None
    else:
        best_conservative = None

    if not stress.empty:
        bad_stress = int((stress["recall"] < target_recall).sum())
        if bad_stress > 0:
            risks.append("THRESHOLD_STRESS_HAS_RECALL_FAILURE")
    else:
        bad_stress = None

    if not risks:
        risk_level = "LOW"
    elif "GLOBAL_RECALL_BELOW_TARGET" in risks:
        risk_level = "HIGH"
    elif "BOOTSTRAP_RECALL_CI_LOWER_BELOW_TARGET" in risks or "SOME_TIME_BLOCK_RECALL_BELOW_TARGET" in risks:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW_MEDIUM"

    return {
        "risk_level": risk_level,
        "risks": risks,
        "min_time_block_recall": min_block_recall,
        "bootstrap_recall_p025": recall_p025,
        "bootstrap_prob_recall_below_target": p_below,
        "threshold_stress_failures": bad_stress,
        "best_conservative_leave_one_out": best_conservative,
    }


def make_report(summary: dict[str, Any], global_metrics_df: pd.DataFrame, time_blocks: pd.DataFrame, loo: pd.DataFrame, stress: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-013C — Frozen Statistical Veto Robustness Validation")
    lines.append("")
    lines.append("## Decisão preliminar")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Overfitting risk level: `{summary['overfitting_risk']['risk_level']}`")
    lines.append(f"- Risks: `{', '.join(summary['overfitting_risk']['risks']) or 'none'}`")
    lines.append("")
    lines.append("## Métricas globais")
    lines.append(global_metrics_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Blocos temporais")
    if time_blocks.empty:
        lines.append("Sem blocos temporais.")
    else:
        lines.append(time_blocks.to_markdown(index=False))
    lines.append("")
    lines.append("## Leave-one-rule-out")
    if loo.empty:
        lines.append("Sem ablação.")
    else:
        use = ["excluded_rule_id", "tp", "fp", "fn", "precision", "recall", "f1", "fpr"]
        lines.append(loo[use].to_markdown(index=False))
    lines.append("")
    lines.append("## Stress test de thresholds")
    if stress.empty:
        lines.append("Sem stress test.")
    else:
        lines.append(stress.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretação")
    if summary["overfitting_risk"]["risk_level"] in {"LOW", "LOW_MEDIUM"}:
        lines.append("A política congelada mostra sinais aceitáveis de robustez nesta validação. Ainda assim, para promoção, recomenda-se testar em uma segunda janela temporal ou outro split gerado pelo mesmo pipeline.")
    else:
        lines.append("A política congelada tem sinais de risco. Recomenda-se usar uma variante conservadora, possivelmente removendo a regra que mais consome TP, e repetir o EXP-013C.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--time-blocks", type=int, default=5)
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    policy_path = Path(args.policy)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-013C — Frozen Statistical Veto Robustness Validation")
    log("=" * 80)
    log(f"Input: {input_path}")
    log(f"Policy: {policy_path}")
    log(f"Output: {output_dir}")
    log(f"Target recall: {args.target_recall}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    policy = load_policy(policy_path)
    dump_json(policy, output_dir / "12_policy_used.json")

    y = df["is_fraud"].to_numpy(dtype=int)
    baseline_pred = df["shadow_exp012d_flagged"].to_numpy(dtype=int)
    frozen_pred, rule_impacts = apply_policy(df, policy)

    baseline_metrics = compute_metrics(y, baseline_pred)
    frozen_metrics = compute_metrics(y, frozen_pred)

    global_rows = [
        {"policy": "BASELINE_SHADOW_EXP012D", **baseline_metrics},
        {"policy": "FROZEN_EXP013B_R1_POLICY", **frozen_metrics},
    ]
    global_metrics_df = pd.DataFrame(global_rows)
    global_metrics_df.to_csv(output_dir / "01_global_metrics.csv", index=False)

    log("[1/6] Métricas globais calculadas.")
    log(f"      Baseline: TP={baseline_metrics['tp']} FP={baseline_metrics['fp']} FN={baseline_metrics['fn']} recall={baseline_metrics['recall']}")
    log(f"      Frozen:   TP={frozen_metrics['tp']} FP={frozen_metrics['fp']} FN={frozen_metrics['fn']} recall={frozen_metrics['recall']}")

    log("[2/6] Validação por blocos temporais...")
    tb = time_block_metrics(df, policy, args.time_blocks)
    tb.to_csv(output_dir / "02_time_block_metrics.csv", index=False)

    log("[3/6] Bootstrap confidence intervals...")
    boot = bootstrap_ci(df, policy, args.bootstrap_iters, args.seed, args.target_recall)
    boot.to_csv(output_dir / "03_bootstrap_confidence_intervals.csv", index=False)

    log("[4/6] Impacto individual e ablação leave-one-rule-out...")
    individual_rows = []
    loo_rows = []

    for rule in policy["rules"]:
        rid = str(rule.get("rule_id"))
        pred_single, impact_single = apply_policy(df, policy, only_rule_id=rid)
        # For single-rule, compare removed from baseline.
        m_single = compute_metrics(y, pred_single)
        individual_rows.append({
            "rule_id": rid,
            "family": rule.get("family"),
            "description": rule.get("description"),
            **m_single,
            "fp_removed_vs_baseline": baseline_metrics["fp"] - m_single["fp"],
            "tp_lost_vs_baseline": baseline_metrics["tp"] - m_single["tp"],
        })

        pred_loo, _ = apply_policy(df, policy, excluded_rule_ids={rid})
        m_loo = compute_metrics(y, pred_loo)
        loo_rows.append({
            "excluded_rule_id": rid,
            "excluded_family": rule.get("family"),
            "excluded_description": rule.get("description"),
            **m_loo,
            "fp_delta_vs_full_policy": m_loo["fp"] - frozen_metrics["fp"],
            "tp_delta_vs_full_policy": m_loo["tp"] - frozen_metrics["tp"],
        })

    individual_df = pd.DataFrame(individual_rows)
    loo_df = pd.DataFrame(loo_rows)
    individual_df.to_csv(output_dir / "04_rule_individual_impact.csv", index=False)
    loo_df.to_csv(output_dir / "05_rule_leave_one_out.csv", index=False)

    log("[5/6] Impacto sequencial das regras...")
    seq_rows = []
    current_excluded = set(str(r.get("rule_id")) for r in policy["rules"])
    # Add rules in the artifact order.
    included: set[str] = set()
    for rule in policy["rules"]:
        rid = str(rule.get("rule_id"))
        included.add(rid)
        excluded = set(str(r.get("rule_id")) for r in policy["rules"]) - included
        pred_seq, _ = apply_policy(df, policy, excluded_rule_ids=excluded)
        m_seq = compute_metrics(y, pred_seq)
        seq_rows.append({
            "step": len(included),
            "added_rule_id": rid,
            "added_family": rule.get("family"),
            "added_description": rule.get("description"),
            **m_seq,
            "fp_removed_vs_baseline": baseline_metrics["fp"] - m_seq["fp"],
            "tp_lost_vs_baseline": baseline_metrics["tp"] - m_seq["tp"],
        })
    seq_df = pd.DataFrame(seq_rows)
    seq_df.to_csv(output_dir / "06_rule_sequential_impact.csv", index=False)

    log("[6/6] Stress test de thresholds...")
    stress = stress_test(df, policy, multipliers=[0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20])
    stress.to_csv(output_dir / "07_threshold_stress_test.csv", index=False)

    predictions = df.copy()
    predictions["exp013c_frozen_pred"] = frozen_pred
    predictions["exp013c_removed_by_veto"] = ((baseline_pred == 1) & (frozen_pred == 0)).astype(int)
    predictions.to_csv(output_dir / "08_frozen_policy_predictions.csv", index=False)
    predictions[(predictions["is_fraud"] == 1) & (predictions["exp013c_frozen_pred"] == 0)].to_csv(output_dir / "09_frozen_policy_false_negatives.csv", index=False)
    predictions[(predictions["is_fraud"] == 0) & (predictions["exp013c_frozen_pred"] == 1)].to_csv(output_dir / "10_frozen_policy_false_positives.csv", index=False)

    risk = risk_assessment(frozen_metrics, tb, boot, loo_df, stress, args.target_recall)

    objective_status = "TARGET_RECALL_MET" if frozen_metrics["recall"] >= args.target_recall else "TARGET_RECALL_NOT_MET"
    objective_status += "_FP_REDUCED" if frozen_metrics["fp"] < baseline_metrics["fp"] else "_FP_NOT_REDUCED"
    objective_status += f"_OVERFIT_RISK_{risk['risk_level']}"

    summary = {
        "experiment": "EXP-013C",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "policy_path": str(policy_path),
        "n_rows": int(len(df)),
        "total_frauds": int(y.sum()),
        "target_recall": args.target_recall,
        "baseline_metrics": baseline_metrics,
        "frozen_policy_metrics": frozen_metrics,
        "fp_removed_vs_baseline": int(baseline_metrics["fp"] - frozen_metrics["fp"]),
        "tp_lost_vs_baseline": int(baseline_metrics["tp"] - frozen_metrics["tp"]),
        "rule_count": int(len(policy["rules"])),
        "overfitting_risk": risk,
        "bootstrap_iters": args.bootstrap_iters,
        "time_blocks": args.time_blocks,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, global_metrics_df, tb, loo_df, stress)
    (output_dir / "11_overfitting_risk_report.md").write_text(report, encoding="utf-8")

    log("")
    log("=" * 80)
    log("EXP-013C CONCLUÍDO")
    log("=" * 80)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_global_metrics.csv",
        output_dir / "02_time_block_metrics.csv",
        output_dir / "03_bootstrap_confidence_intervals.csv",
        output_dir / "05_rule_leave_one_out.csv",
        output_dir / "07_threshold_stress_test.csv",
        output_dir / "11_overfitting_risk_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
