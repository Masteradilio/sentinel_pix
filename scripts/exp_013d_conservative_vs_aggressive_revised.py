#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-013D — Conservative vs Aggressive-Revised Frozen Policy Duel

Objetivo:
  Testar, lado a lado, duas políticas congeladas derivadas do EXP-013C:

  A) Conservadora:
     Política EXP-013B-R1 sem a regra receiver_value_established:
       lgbm<0.02 AND receiver_value_180d>2000

  B) Agressiva revisada:
     Política EXP-013B-R1 completa com threshold_multiplier=1.05,
     conforme hipótese observada no stress test do EXP-013C.

Princípios:
  - Não há nova busca de regras.
  - Não há retuning por métrica.
  - As duas políticas são congeladas e auditáveis.
  - A decisão é por robustez:
      recall global >= 95%
      recall por blocos o mais estável possível
      bootstrap com menor risco possível de recall abaixo de 95%
      menor FP entre políticas aprováveis

Entradas default:
  resultados/experimentos/EXP-012E/04_comparison_by_transaction.csv
  resultados/experimentos/EXP-013B-R1/10_policy_artifact.json

Uso:
  python scripts/exp_013d_conservative_vs_aggressive_revised.py

Uso mais rápido:
  python scripts/exp_013d_conservative_vs_aggressive_revised.py --bootstrap-iters 200

Saídas:
  resultados/experimentos/EXP-013D/
    00_run_summary.json
    01_global_metrics.csv
    02_time_block_metrics.csv
    03_bootstrap_confidence_intervals.csv
    04_policy_rule_impacts.csv
    05_policy_predictions.csv
    06_false_negatives_by_policy.csv
    07_false_positives_by_policy.csv
    08_policy_selection_report.md
    09_selected_policy_artifact.json
    10_policy_variants_tested.json
"""

from __future__ import annotations

import argparse
import json
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013D"

FLAGGED_DECISIONS = {"CONFIRMAR", "BLOQUEAR"}
RECEIVER_VALUE_FAMILY = "receiver_value_established"


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

    if family == "numeric":
        feature = params.get("feature")
        op = params.get("op")
        th = float(params.get("threshold")) * threshold_multiplier
        vals = num(df, feature, np.nan)
        if op == "lt":
            mask = (vals < th).to_numpy(dtype=bool)
        elif op == "gt":
            mask = (vals > th).to_numpy(dtype=bool)
        else:
            raise ValueError(f"Operador numeric desconhecido: {op}")
        if bool(params.get("preserve", False)):
            mask = mask & (~preserve)
        return mask

    if family == "segment":
        cols = params.get("segment_cols", [])
        values = params.get("segment_values", [])
        mask = np.ones(len(df), dtype=bool)
        for c, v in zip(cols, values):
            mask = mask & (text(df, c) == str(v)).to_numpy(dtype=bool)
        if bool(params.get("preserve", False)):
            mask = mask & (~preserve)
        return mask

    if family == "segment_lgbm":
        cols = params.get("segment_cols", [])
        values = params.get("segment_values", [])
        lgbm_lt = float(params.get("lgbm_lt")) * threshold_multiplier
        mask = np.ones(len(df), dtype=bool)
        for c, v in zip(cols, values):
            mask = mask & (text(df, c) == str(v)).to_numpy(dtype=bool)
        mask = mask & (get_lgbm_score(df) < lgbm_lt).to_numpy(dtype=bool)
        if bool(params.get("preserve", False)):
            mask = mask & (~preserve)
        return mask

    if family == "receiver_value_established":
        lgbm_lt = float(params.get("lgbm_lt")) * threshold_multiplier
        receiver_value_gt = float(params.get("receiver_value_gt")) * threshold_multiplier
        mask = (
            (get_lgbm_score(df) < lgbm_lt)
            & (num(df, "valor_total_recebido_180d", 0.0) > receiver_value_gt)
        ).to_numpy(dtype=bool)
        # Na geração original, esta família foi usada com strong preserve.
        mask = mask & (~preserve)
        return mask

    if family == "quiet_veto":
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
        return mask

    raise ValueError(f"Família de regra não implementada: {family}")


def build_policy_variants(base_policy: dict[str, Any]) -> list[dict[str, Any]]:
    receiver_rule_ids = [
        str(r.get("rule_id"))
        for r in base_policy["rules"]
        if str(r.get("family")) == RECEIVER_VALUE_FAMILY
        or "receiver_value_180d>2000" in str(r.get("description", ""))
        or "receiver_value_gt" in str(r.get("params_json", ""))
    ]

    return [
        {
            "policy_name": "BASELINE_SHADOW_EXP012D",
            "kind": "baseline",
            "description": "Baseline shadow EXP-012D sem veto estatístico.",
            "excluded_rule_ids": [],
            "threshold_multiplier": 1.0,
        },
        {
            "policy_name": "AGGRESSIVE_ORIGINAL_EXP013B_R1",
            "kind": "frozen_policy",
            "description": "Política agressiva original EXP-013B-R1.",
            "excluded_rule_ids": [],
            "threshold_multiplier": 1.0,
        },
        {
            "policy_name": "CONSERVATIVE_NO_RECEIVER_VALUE",
            "kind": "frozen_policy",
            "description": "Variante conservadora sem a regra receiver_value_established.",
            "excluded_rule_ids": receiver_rule_ids,
            "threshold_multiplier": 1.0,
        },
        {
            "policy_name": "AGGRESSIVE_REVISED_MULT_1_05",
            "kind": "frozen_policy",
            "description": "Política agressiva revisada com threshold_multiplier=1.05.",
            "excluded_rule_ids": [],
            "threshold_multiplier": 1.05,
        },
    ]


def apply_policy_variant(df: pd.DataFrame, base_policy: dict[str, Any], variant: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame]:
    shadow = df["shadow_exp012d_flagged"].to_numpy(dtype=int).astype(bool)

    if variant["kind"] == "baseline":
        return shadow.astype(int), pd.DataFrame()

    excluded = set(variant.get("excluded_rule_ids", []))
    mult = float(variant.get("threshold_multiplier", 1.0))

    veto_any = np.zeros(len(df), dtype=bool)
    rule_rows = []
    y = df["is_fraud"].to_numpy(dtype=int)

    for idx, rule in enumerate(base_policy["rules"]):
        rule_id = str(rule.get("rule_id", f"rule_{idx}"))
        if rule_id in excluded:
            continue

        mask = apply_rule_mask(df, rule, threshold_multiplier=mult)
        mask = mask & shadow

        tp_loss = int(((y == 1) & mask).sum())
        fp_removed = int(((y == 0) & mask).sum())

        rule_rows.append({
            "policy_name": variant["policy_name"],
            "rule_id": rule_id,
            "family": rule.get("family"),
            "description": rule.get("description"),
            "tp_loss": tp_loss,
            "fp_removed": fp_removed,
            "n_removed": int(mask.sum()),
            "threshold_multiplier": mult,
            "excluded": False,
        })

        veto_any = veto_any | mask

    # Record excluded rules for audit.
    for idx, rule in enumerate(base_policy["rules"]):
        rule_id = str(rule.get("rule_id", f"rule_{idx}"))
        if rule_id in excluded:
            rule_rows.append({
                "policy_name": variant["policy_name"],
                "rule_id": rule_id,
                "family": rule.get("family"),
                "description": rule.get("description"),
                "tp_loss": None,
                "fp_removed": None,
                "n_removed": None,
                "threshold_multiplier": mult,
                "excluded": True,
            })

    pred = shadow.astype(int)
    pred[veto_any] = 0
    return pred, pd.DataFrame(rule_rows)


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


def evaluate_policies(df: pd.DataFrame, base_policy: dict[str, Any], variants: list[dict[str, Any]], time_blocks: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df["is_fraud"].to_numpy(dtype=int)
    blocks = make_time_blocks(df, time_blocks)

    global_rows = []
    block_rows = []
    impact_rows = []
    predictions = df.copy()

    for variant in variants:
        pname = variant["policy_name"]
        pred, impacts = apply_policy_variant(df, base_policy, variant)
        predictions[f"pred_{pname}"] = pred

        m = compute_metrics(y, pred)
        m.update({
            "policy_name": pname,
            "kind": variant["kind"],
            "description": variant["description"],
            "threshold_multiplier": variant.get("threshold_multiplier", 1.0),
            "excluded_rule_ids": "|".join(variant.get("excluded_rule_ids", [])),
        })
        global_rows.append(m)

        if not impacts.empty:
            impact_rows.append(impacts)

        for b in sorted(blocks.dropna().unique()):
            part = df.loc[blocks == b].copy()
            pred_b = pred[blocks == b]
            mb = compute_metrics(part["is_fraud"].to_numpy(dtype=int), pred_b)
            mb.update({
                "policy_name": pname,
                "block": int(b),
                "n_rows": int(len(part)),
                "n_frauds": int(part["is_fraud"].sum()),
                "dt_min": str(part["data_pix"].min().date()) if "data_pix" in part.columns and part["data_pix"].notna().any() else None,
                "dt_max": str(part["data_pix"].max().date()) if "data_pix" in part.columns and part["data_pix"].notna().any() else None,
            })
            block_rows.append(mb)

    global_df = pd.DataFrame(global_rows)
    block_df = pd.DataFrame(block_rows)
    impact_df = pd.concat(impact_rows, ignore_index=True) if impact_rows else pd.DataFrame()
    return global_df, block_df, impact_df, predictions


def bootstrap_policy(df: pd.DataFrame, base_policy: dict[str, Any], variants: list[dict[str, Any]], iters: int, seed: int, target_recall: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(df)
    rows = []

    for variant in variants:
        if variant["kind"] == "baseline":
            # Ainda avalia baseline para comparação.
            pass

        metrics_list = []
        for _ in range(iters):
            idx = rng.integers(0, n, size=n)
            sample = df.iloc[idx].reset_index(drop=True)
            pred, _ = apply_policy_variant(sample, base_policy, variant)
            metrics_list.append(compute_metrics(sample["is_fraud"].to_numpy(dtype=int), pred))

        boot = pd.DataFrame(metrics_list)
        for metric in ["tp", "fp", "fn", "precision", "recall", "f1", "fpr"]:
            vals = boot[metric].astype(float)
            rows.append({
                "policy_name": variant["policy_name"],
                "metric": metric,
                "mean": float(vals.mean()),
                "p025": float(vals.quantile(0.025)),
                "p050": float(vals.quantile(0.50)),
                "p975": float(vals.quantile(0.975)),
                "target_recall": target_recall if metric == "recall" else None,
                "p_below_target_recall": float((boot["recall"] < target_recall).mean()) if metric == "recall" else None,
            })

    return pd.DataFrame(rows)


def select_policy(global_df: pd.DataFrame, block_df: pd.DataFrame, boot_df: pd.DataFrame, target_recall: float) -> dict[str, Any]:
    rows = []

    # Do not select baseline or original aggressive as final unless others fail.
    for _, g in global_df.iterrows():
        pname = g["policy_name"]
        if pname == "BASELINE_SHADOW_EXP012D":
            continue

        blocks = block_df[block_df["policy_name"] == pname]
        min_block_recall = float(blocks["recall"].min()) if not blocks.empty else None

        recall_ci = boot_df[(boot_df["policy_name"] == pname) & (boot_df["metric"] == "recall")]
        if not recall_ci.empty:
            recall_p025 = float(recall_ci["p025"].iloc[0])
            p_below = float(recall_ci["p_below_target_recall"].iloc[0])
        else:
            recall_p025 = None
            p_below = None

        global_ok = float(g["recall"]) >= target_recall
        # Strong approval is conservative; usable candidate can still be medium risk.
        robust_ok = (
            global_ok
            and (min_block_recall is not None and min_block_recall >= target_recall)
            and (recall_p025 is not None and recall_p025 >= target_recall)
            and (p_below is not None and p_below <= 0.10)
        )

        medium_ok = (
            global_ok
            and (p_below is not None and p_below <= 0.45)
        )

        rows.append({
            "policy_name": pname,
            "global_recall": float(g["recall"]),
            "global_fp": int(g["fp"]),
            "global_tp": int(g["tp"]),
            "global_fn": int(g["fn"]),
            "min_block_recall": min_block_recall,
            "bootstrap_recall_p025": recall_p025,
            "bootstrap_prob_below_target": p_below,
            "global_ok": global_ok,
            "robust_ok": robust_ok,
            "medium_ok": medium_ok,
        })

    score = pd.DataFrame(rows)

    # Prefer robust policies. If none, prefer medium_ok conservative/no receiver over aggressive if both similar.
    robust = score[score["robust_ok"]].copy()
    if not robust.empty:
        chosen = robust.sort_values(["global_fp", "global_fn"], ascending=[True, True]).iloc[0].to_dict()
        rationale = "Selected lowest FP among robust policies."
    else:
        medium = score[score["medium_ok"]].copy()
        if not medium.empty:
            # Penalize policies with min block recall far below target and favor conservative.
            medium["risk_sort"] = (
                (target_recall - medium["min_block_recall"].fillna(0)).clip(lower=0)
                + (target_recall - medium["bootstrap_recall_p025"].fillna(0)).clip(lower=0)
                + medium["bootstrap_prob_below_target"].fillna(1)
            )
            chosen = medium.sort_values(["risk_sort", "global_fp"], ascending=[True, True]).iloc[0].to_dict()
            rationale = "No fully robust policy; selected best medium-risk policy."
        else:
            chosen = score.sort_values(["global_recall", "global_fp"], ascending=[False, True]).iloc[0].to_dict()
            rationale = "No policy met target safely; selected diagnostic best by recall then FP."

    return {
        "selection_table": rows,
        "selected_policy_name": chosen["policy_name"],
        "selected_policy": chosen,
        "selection_rationale": rationale,
    }


def make_false_files(predictions: pd.DataFrame, selected_policy: str, output_dir: Path) -> None:
    col = f"pred_{selected_policy}"
    if col not in predictions.columns:
        return
    predictions[(predictions["is_fraud"] == 1) & (predictions[col] == 0)].to_csv(output_dir / "06_false_negatives_selected_policy.csv", index=False)
    predictions[(predictions["is_fraud"] == 0) & (predictions[col] == 1)].to_csv(output_dir / "07_false_positives_selected_policy.csv", index=False)


def make_report(summary: dict[str, Any], global_df: pd.DataFrame, block_df: pd.DataFrame, boot_df: pd.DataFrame, selection: dict[str, Any]) -> str:
    lines = []
    lines.append("# EXP-013D — Conservative vs Aggressive-Revised Frozen Policy Duel")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Selected policy: `{summary['selected_policy_name']}`")
    lines.append(f"- Rationale: {selection['selection_rationale']}")
    lines.append("")
    lines.append("## Métricas globais")
    lines.append(global_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Seleção")
    lines.append(pd.DataFrame(selection["selection_table"]).to_markdown(index=False))
    lines.append("")
    lines.append("## Blocos temporais")
    lines.append(block_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Bootstrap recall")
    rec = boot_df[boot_df["metric"] == "recall"].copy()
    lines.append(rec.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretação")
    selected = summary["selected_policy_name"]
    if selected == "CONSERVATIVE_NO_RECEIVER_VALUE":
        lines.append("A variante conservadora foi favorecida. Isso indica que recuperar folga de recall é mais importante do que espremer o último bloco de FP neste momento.")
    elif selected == "AGGRESSIVE_REVISED_MULT_1_05":
        lines.append("A variante agressiva revisada foi favorecida. Antes de promover, ela ainda deve passar por validação E2E/temporal externa.")
    else:
        lines.append("Nenhuma das duas variantes resolveu completamente o risco; usar este resultado para nova rodada conservadora.")
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
    log("EXP-013D — Conservative vs Aggressive-Revised Frozen Policy Duel")
    log("=" * 80)
    log(f"Input: {input_path}")
    log(f"Policy: {policy_path}")
    log(f"Output: {output_dir}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    policy = load_policy(policy_path)
    variants = build_policy_variants(policy)

    dump_json({"variants": variants, "base_policy_path": str(policy_path)}, output_dir / "10_policy_variants_tested.json")

    log("[1/4] Avaliando políticas globalmente e por blocos...")
    global_df, block_df, impact_df, predictions = evaluate_policies(df, policy, variants, args.time_blocks)
    global_df.to_csv(output_dir / "01_global_metrics.csv", index=False)
    block_df.to_csv(output_dir / "02_time_block_metrics.csv", index=False)
    impact_df.to_csv(output_dir / "04_policy_rule_impacts.csv", index=False)
    predictions.to_csv(output_dir / "05_policy_predictions.csv", index=False)

    log("[2/4] Bootstrap...")
    boot_df = bootstrap_policy(df, policy, variants, args.bootstrap_iters, args.seed, args.target_recall)
    boot_df.to_csv(output_dir / "03_bootstrap_confidence_intervals.csv", index=False)

    log("[3/4] Selecionando política...")
    selection = select_policy(global_df, block_df, boot_df, args.target_recall)
    selected_name = selection["selected_policy_name"]

    make_false_files(predictions, selected_name, output_dir)

    selected_global = global_df[global_df["policy_name"] == selected_name].iloc[0].to_dict()
    baseline_global = global_df[global_df["policy_name"] == "BASELINE_SHADOW_EXP012D"].iloc[0].to_dict()

    objective_status = "TARGET_RECALL_MET" if selected_global["recall"] >= args.target_recall else "TARGET_RECALL_NOT_MET"
    objective_status += "_FP_REDUCED" if selected_global["fp"] < baseline_global["fp"] else "_FP_NOT_REDUCED"

    sel_table = pd.DataFrame(selection["selection_table"])
    selected_sel = sel_table[sel_table["policy_name"] == selected_name].iloc[0].to_dict()
    if selected_sel.get("robust_ok"):
        objective_status += "_ROBUST_OK"
    elif selected_sel.get("medium_ok"):
        objective_status += "_MEDIUM_RISK"
    else:
        objective_status += "_HIGH_RISK"

    # Save selected policy artifact.
    selected_variant = next(v for v in variants if v["policy_name"] == selected_name)
    selected_artifact = {
        "experiment": "EXP-013D",
        "selected_policy_name": selected_name,
        "target_recall": args.target_recall,
        "objective_status": objective_status,
        "selected_variant": selected_variant,
        "selected_global_metrics": selected_global,
        "selection": selection,
        "base_policy": policy,
        "notes": [
            "No new search was performed; this experiment compares frozen variants.",
            "Conservative policy excludes receiver_value_established rule.",
            "Aggressive revised policy uses threshold_multiplier=1.05.",
            "Before production patch, validate selected policy on external temporal sample/E2E.",
        ],
    }
    dump_json(selected_artifact, output_dir / "09_selected_policy_artifact.json")

    summary = {
        "experiment": "EXP-013D",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "policy_path": str(policy_path),
        "n_rows": int(len(df)),
        "total_frauds": int(df["is_fraud"].sum()),
        "target_recall": args.target_recall,
        "bootstrap_iters": args.bootstrap_iters,
        "time_blocks": args.time_blocks,
        "global_metrics": global_df.to_dict(orient="records"),
        "selection": selection,
        "selected_policy_name": selected_name,
        "selected_global_metrics": selected_global,
        "fp_removed_vs_baseline_shadow": int(baseline_global["fp"] - selected_global["fp"]),
        "tp_lost_vs_baseline_shadow": int(baseline_global["tp"] - selected_global["tp"]),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    log("[4/4] Gerando relatório...")
    report = make_report(summary, global_df, block_df, boot_df, selection)
    (output_dir / "08_policy_selection_report.md").write_text(report, encoding="utf-8")

    log("")
    log("=" * 80)
    log("EXP-013D CONCLUÍDO")
    log("=" * 80)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_global_metrics.csv",
        output_dir / "02_time_block_metrics.csv",
        output_dir / "03_bootstrap_confidence_intervals.csv",
        output_dir / "08_policy_selection_report.md",
        output_dir / "09_selected_policy_artifact.json",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
