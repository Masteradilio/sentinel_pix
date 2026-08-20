# -*- coding: utf-8 -*-
"""
EXP-014B-R3Y — Confirm Queue Reduction / Low-Risk Confirmation Demotion

Objetivo:
  Melhorar a fila CONFIRMAR consolidada no R3X-FROZEN.

Baseline congelado:
  - BLOQUEAR deve permanecer intocado.
  - APROVAR deve permanecer intocado, exceto pelos CONFIRMAR demovidos.
  - Apenas casos atualmente em CONFIRMAR podem ser rebaixados para APROVAR.
  - Orçamento rígido: FN adicional <= 5.

Contexto R3X-FROZEN:
  APROVAR   = 108305 linhas, 0 fraudes
  BLOQUEAR  = 2493 linhas, 1293 fraudes, 1200 normais, FPR=1.068%
  CONFIRMAR = 3046 linhas, 172 fraudes, 2874 normais, precision=5.65%

Meta R3Y:
  Reduzir normais em CONFIRMAR sem mexer em BLOQUEAR e sem perder mais que
  5 fraudes conhecidas para APROVAR.

Saídas:
  resultados/experimentos/EXP-014B-R3Y/
    00_run_summary.json
    01_input_contract.json
    02_base_frozen_metrics.json
    03_demote_candidates.csv
    04_selection_frontier.csv
    05_selected_demotions.csv
    06_decision_metrics_by_action.csv
    07_robustness_by_segment.csv
    08_policy_artifact_recommended.json
    09_predictions_recommended.csv
    10_exp014b_r3y_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT = "EXP-014B-R3Y"

LABEL_CANDIDATES = ["is_fraud", "fraude", "target", "label", "tp_fraude"]

ACTION_CANDIDATES = [
    "r3x_frozen_decisao_pos_policy",
    "r3x_decisao_pos_policy",
    "decisao",
]

INTERVENTION_CANDIDATES = [
    "exp014b_r3x_frozen_intervention_pred",
    "exp014b_r3x_intervention_pred",
]

BLOCK_CANDIDATES = [
    "exp014b_r3x_frozen_block_pred",
    "exp014b_r3x_block_pred",
]

SCORE_COLS = [
    "lgbm_r4_score",
    "score_final",
    "lgbm_raw",
    "lgbm_mapped",
    "peso_total",
    "if_percentile",
    "se_score",
    "beh_score",
    "behavioral_score",
    "topaz_risk_score",
    "exp014b_r3s_second_stage_score",
    "exp014b_r3u_receiver_relationship_trust_score",
]

CATEGORICAL_COLS = [
    "ds_tipo_chave_norm",
    "value_band",
    "periodo_dia",
    "score_bin",
    "lgbm_bin",
    "if_bin",
    "ratio_bin",
    "qtd_rec_bin",
    "valor_rec_bin",
    "module_quiet",
    "se_worst_pattern",
    "mbk_available_flag",
    "first_receiver_flag_real",
    "r3u_missing_receiver_history_flag",
    "r3u_receiver_known_flag",
    "r3u_receiver_reputable_flag",
    "r3u_receiver_strong_flag",
    "r3u_relationship_known_flag",
    "r3u_relationship_recurrent_flag",
    "r3u_relationship_strong_flag",
    "r3u_first_receiver_flag",
    "r3u_module_quiet_flag",
    "r3u_se_missing_flag",
    "r3u_ratio_lt_005_flag",
    "r3u_mbk_quality_flag",
    "r3u_receiver_trust_bucket",
    "r3u_relationship_bucket",
]

SEGMENT_COLS = [
    "temporal_split",
    "event_month",
    "ds_tipo_chave_norm",
    "value_band",
    "periodo_dia",
    "score_bin",
    "lgbm_bin",
    "if_bin",
    "ratio_bin",
    "qtd_rec_bin",
    "valor_rec_bin",
    "module_quiet",
    "se_worst_pattern",
    "mbk_available_flag",
    "first_receiver_flag_real",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, default=None)
    parser.add_argument("--artifact", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-fn-additional", type=int, default=5)
    parser.add_argument("--min-fp-removed", type=int, default=10)
    parser.add_argument("--max-rules", type=int, default=25)
    parser.add_argument("--max-candidates", type=int, default=2000)
    return parser.parse_args()


def default_paths() -> tuple[Path, Path, Path]:
    root = Path.cwd()
    pred = root / "resultados" / "experimentos" / "EXP-014B-R3X-FROZEN" / "06_predictions_frozen.csv"
    artifact = root / "resultados" / "experimentos" / "EXP-014B-R3X-FROZEN" / "05_policy_artifact_frozen.json"
    out = root / "resultados" / "experimentos" / EXPERIMENT
    return pred, artifact, out


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(f"Nenhuma coluna encontrada entre: {candidates}")
    return None


def safe_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def metrics(y_true: pd.Series, pred: pd.Series) -> dict[str, Any]:
    y = safe_int_series(y_true)
    p = safe_int_series(pred)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
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


def action_to_intervention(action: pd.Series) -> pd.Series:
    return action.astype(str).str.upper().isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)


def action_to_block(action: pd.Series) -> pd.Series:
    return action.astype(str).str.upper().eq("BLOQUEAR").astype(int)


def normalize_action_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def build_candidate_row(
    df: pd.DataFrame,
    label_col: str,
    confirm_mask: pd.Series,
    demote_mask: pd.Series,
    candidate_id: str,
    rule_type: str,
    description: str,
) -> dict[str, Any] | None:
    y = safe_int_series(df[label_col])
    mask = confirm_mask & demote_mask.fillna(False)
    n = int(mask.sum())
    if n == 0:
        return None

    fp_removed = int(((y == 0) & mask).sum())
    tp_loss = int(((y == 1) & mask).sum())
    precision_demoted = tp_loss / n if n else 0.0

    return {
        "candidate_id": candidate_id,
        "rule_type": rule_type,
        "description": description,
        "n_demoted": n,
        "fp_removed": fp_removed,
        "tp_loss": tp_loss,
        "precision_demoted": round(float(precision_demoted), 8),
        "fp_per_tp_loss": round(float(fp_removed / max(tp_loss, 1)), 8),
    }


def build_candidates(
    df: pd.DataFrame,
    label_col: str,
    action_col: str,
    min_fp_removed: int,
    max_fn_additional: int,
    max_candidates: int,
) -> pd.DataFrame:
    y = safe_int_series(df[label_col])
    action = normalize_action_series(df[action_col])
    confirm_mask = action.eq("CONFIRMAR")
    rows: list[dict[str, Any]] = []

    # Score thresholds, both directions.
    score_cols = [c for c in SCORE_COLS if c in df.columns]
    quantiles = [
        0.01, 0.02, 0.05, 0.08, 0.10,
        0.12, 0.15, 0.18, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.92, 0.95,
        0.98, 0.99,
    ]

    for col in score_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s[confirm_mask & s.notna()]
        if valid.empty:
            continue
        thresholds = sorted(set(float(valid.quantile(q)) for q in quantiles if pd.notna(valid.quantile(q))))
        for th in thresholds:
            row = build_candidate_row(
                df,
                label_col,
                confirm_mask,
                s.le(th),
                f"score_lo__{col}__{th:.12g}",
                "score_threshold",
                f"Demover CONFIRMAR com {col} <= {th:.12g}",
            )
            if row:
                rows.append(row)

            row = build_candidate_row(
                df,
                label_col,
                confirm_mask,
                s.ge(th),
                f"score_hi__{col}__{th:.12g}",
                "score_threshold",
                f"Demover CONFIRMAR com {col} >= {th:.12g}",
            )
            if row:
                rows.append(row)

    # Single categorical segments.
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    for col in cat_cols:
        vals = df.loc[confirm_mask, col].fillna("<MISSING>").astype(str)
        value_counts = vals.value_counts(dropna=False)
        for val, count in value_counts.items():
            if int(count) == 0:
                continue
            cond = df[col].fillna("<MISSING>").astype(str).eq(str(val))
            safe_val = str(val).replace("|", "_").replace(" ", "_")[:80]
            row = build_candidate_row(
                df,
                label_col,
                confirm_mask,
                cond,
                f"cat__{col}__{safe_val}",
                "categorical_segment",
                f"Demover CONFIRMAR com {col} == {val}",
            )
            if row:
                rows.append(row)

    # Pair categorical intersections, restricted to useful support to avoid explosion.
    useful_cat_cols = [c for c in cat_cols if df.loc[confirm_mask, c].nunique(dropna=False) <= 30]
    for i, c1 in enumerate(useful_cat_cols):
        for c2 in useful_cat_cols[i + 1:]:
            tmp = df.loc[confirm_mask, [c1, c2]].fillna("<MISSING>").astype(str)
            pair_counts = tmp.value_counts(dropna=False).head(200)
            for (v1, v2), count in pair_counts.items():
                if int(count) < min_fp_removed:
                    continue
                cond = (
                    df[c1].fillna("<MISSING>").astype(str).eq(str(v1))
                    & df[c2].fillna("<MISSING>").astype(str).eq(str(v2))
                )
                sv1 = str(v1).replace("|", "_").replace(" ", "_")[:40]
                sv2 = str(v2).replace("|", "_").replace(" ", "_")[:40]
                row = build_candidate_row(
                    df,
                    label_col,
                    confirm_mask,
                    cond,
                    f"cat2__{c1}={sv1}__{c2}={sv2}",
                    "categorical_pair",
                    f"Demover CONFIRMAR com {c1} == {v1} AND {c2} == {v2}",
                )
                if row:
                    rows.append(row)

    if not rows:
        return pd.DataFrame()

    cand = pd.DataFrame(rows).drop_duplicates(subset=["description"])
    cand = cand[
        (cand["fp_removed"] >= int(min_fp_removed))
        & (cand["tp_loss"] <= int(max_fn_additional))
    ].copy()

    if cand.empty:
        return cand

    cand = cand.sort_values(
        ["tp_loss", "fp_removed", "fp_per_tp_loss", "n_demoted"],
        ascending=[True, False, False, False],
    ).head(max_candidates)

    return cand.reset_index(drop=True)


def mask_from_candidate(df: pd.DataFrame, candidate_id: str) -> pd.Series:
    if candidate_id.startswith("score_lo__") or candidate_id.startswith("score_hi__"):
        direction, col, th = candidate_id.split("__", 2)
        th_val = float(th)
        s = pd.to_numeric(df[col], errors="coerce")
        if direction == "score_lo":
            return s.le(th_val)
        return s.ge(th_val)

    if candidate_id.startswith("cat__"):
        _, col, val = candidate_id.split("__", 2)
        # For exact reconstruction, parse from description when possible is hard with truncated IDs.
        # candidate_id is only used for candidates whose val was stringified safely.
        # Use description-based mask in select_greedy via full candidate row instead.
        raise ValueError("Use mask_from_description for categorical candidates.")

    if candidate_id.startswith("cat2__"):
        raise ValueError("Use mask_from_description for categorical pair candidates.")

    raise ValueError(f"candidate_id não suportado: {candidate_id}")


def mask_from_description(df: pd.DataFrame, description: str) -> pd.Series:
    prefix = "Demover CONFIRMAR com "
    if not description.startswith(prefix):
        raise ValueError(f"Descrição não suportada: {description}")

    expr = description[len(prefix):]

    if " <= " in expr and " AND " not in expr:
        col, th = expr.split(" <= ", 1)
        return pd.to_numeric(df[col], errors="coerce").le(float(th))

    if " >= " in expr and " AND " not in expr:
        col, th = expr.split(" >= ", 1)
        return pd.to_numeric(df[col], errors="coerce").ge(float(th))

    if " AND " in expr:
        parts = expr.split(" AND ")
        mask = pd.Series(True, index=df.index)
        for p in parts:
            col, val = p.split(" == ", 1)
            mask &= df[col].fillna("<MISSING>").astype(str).eq(str(val))
        return mask

    if " == " in expr:
        col, val = expr.split(" == ", 1)
        return df[col].fillna("<MISSING>").astype(str).eq(str(val))

    raise ValueError(f"Expressão não suportada: {expr}")


def select_greedy(
    df: pd.DataFrame,
    label_col: str,
    action_col: str,
    candidates: pd.DataFrame,
    max_fn_additional: int,
    max_rules: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    y = safe_int_series(df[label_col])
    action = normalize_action_series(df[action_col])
    confirm_mask = action.eq("CONFIRMAR")

    selected_rows = []
    frontier_rows = []
    cumulative_demote = pd.Series(False, index=df.index)
    cumulative_fn = 0
    cumulative_fp = 0

    remaining = candidates.copy().reset_index(drop=True)

    for step in range(1, int(max_rules) + 1):
        best = None
        best_mask = None
        best_gain = None

        for _, row in remaining.iterrows():
            try:
                mask = confirm_mask & mask_from_description(df, str(row["description"])) & (~cumulative_demote)
            except Exception:
                continue

            n = int(mask.sum())
            if n == 0:
                continue

            fp_gain = int(((y == 0) & mask).sum())
            fn_gain = int(((y == 1) & mask).sum())

            if fp_gain <= 0:
                continue
            if cumulative_fn + fn_gain > int(max_fn_additional):
                continue

            # Prefer zero FN, then best FP per FN, then absolute FP.
            score = (
                1 if fn_gain == 0 else 0,
                fp_gain / max(fn_gain, 1),
                fp_gain,
                -fn_gain,
                n,
            )

            if best is None or score > best_gain:
                best = row.copy()
                best_mask = mask
                best_gain = score
                best["incremental_n_demoted"] = n
                best["incremental_fp_removed"] = fp_gain
                best["incremental_tp_loss"] = fn_gain

        if best is None or best_mask is None:
            break

        cumulative_demote |= best_mask
        cumulative_fn += int(best["incremental_tp_loss"])
        cumulative_fp += int(best["incremental_fp_removed"])

        best["selection_step"] = step
        best["cumulative_fp_removed"] = cumulative_fp
        best["cumulative_tp_loss"] = cumulative_fn
        selected_rows.append(best)

        frontier_rows.append({
            "selection_step": step,
            "selected_candidate_id": best["candidate_id"],
            "selected_description": best["description"],
            "incremental_n_demoted": int(best["incremental_n_demoted"]),
            "incremental_fp_removed": int(best["incremental_fp_removed"]),
            "incremental_tp_loss": int(best["incremental_tp_loss"]),
            "cumulative_n_demoted": int(cumulative_demote.sum()),
            "cumulative_fp_removed": int(cumulative_fp),
            "cumulative_tp_loss": int(cumulative_fn),
        })

        # Remove selected candidate from future consideration.
        remaining = remaining[remaining["description"] != best["description"]].reset_index(drop=True)

    selected = pd.DataFrame(selected_rows)
    frontier = pd.DataFrame(frontier_rows)
    return selected, frontier, cumulative_demote


def apply_demotions(
    df: pd.DataFrame,
    action_col: str,
    demote_mask: pd.Series,
) -> pd.Series:
    action = normalize_action_series(df[action_col]).copy()
    final = action.copy()
    final[action.eq("CONFIRMAR") & demote_mask.fillna(False)] = "APROVAR"
    return final


def metrics_by_action(df: pd.DataFrame, label_col: str, action_col: str) -> pd.DataFrame:
    rows = []
    y = safe_int_series(df[label_col])
    for action, idx in df.groupby(action_col, dropna=False).groups.items():
        idx = list(idx)
        yy = y.loc[idx]
        n = int(len(idx))
        frauds = int((yy == 1).sum())
        normals = int((yy == 0).sum())
        rows.append({
            "action": str(action),
            "n_rows": n,
            "n_frauds": frauds,
            "n_normals": normals,
            "precision_within_action": round(float(frauds / n), 8) if n else 0.0,
        })
    return pd.DataFrame(rows).sort_values("action")


def robustness(df: pd.DataFrame, label_col: str, before_col: str, after_col: str) -> pd.DataFrame:
    rows = []
    y = safe_int_series(df[label_col])
    before_intervention = action_to_intervention(df[before_col])
    after_intervention = action_to_intervention(df[after_col])

    for col in SEGMENT_COLS:
        if col not in df.columns:
            continue
        for val, idx in df.groupby(col, dropna=False).groups.items():
            idx = list(idx)
            yy = y.loc[idx]
            before_m = metrics(yy, before_intervention.loc[idx])
            after_m = metrics(yy, after_intervention.loc[idx])
            rows.append({
                "segment_col": col,
                "segment_value": str(val),
                "n_rows": int(len(idx)),
                "n_frauds": int((yy == 1).sum()),
                "fp_removed": int(before_m["fp"] - after_m["fp"]),
                "tp_loss": int(before_m["tp"] - after_m["tp"]),
                "before_fp": before_m["fp"],
                "after_fp": after_m["fp"],
                "before_tp": before_m["tp"],
                "after_tp": after_m["tp"],
                "after_fn": after_m["fn"],
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["fp_removed", "n_rows"], ascending=[False, False])


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    default_pred, default_artifact, default_out = default_paths()
    pred_path = Path(args.predictions) if args.predictions else default_pred
    artifact_path = Path(args.artifact) if args.artifact else default_artifact
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions não encontrado: {pred_path}")

    df = pd.read_csv(pred_path, low_memory=False)
    artifact = None
    if artifact_path.exists():
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    label_col = find_col(df, LABEL_CANDIDATES)
    action_col = find_col(df, ACTION_CANDIDATES)
    block_col = find_col(df, BLOCK_CANDIDATES, required=False)
    intervention_col = find_col(df, INTERVENTION_CANDIDATES, required=False)

    original_action = normalize_action_series(df[action_col])
    if block_col is not None:
        original_block = safe_int_series(df[block_col])
    else:
        original_block = action_to_block(original_action)

    if intervention_col is not None:
        original_intervention = safe_int_series(df[intervention_col])
    else:
        original_intervention = action_to_intervention(original_action)

    y = safe_int_series(df[label_col])
    baseline_intervention_metrics = metrics(y, original_intervention)
    baseline_block_metrics = metrics(y, original_block)
    baseline_by_action = metrics_by_action(df.assign(_action=original_action), label_col, "_action")

    candidates = build_candidates(
        df,
        label_col,
        action_col,
        int(args.min_fp_removed),
        int(args.max_fn_additional),
        int(args.max_candidates),
    )

    if candidates.empty:
        selected = pd.DataFrame()
        frontier = pd.DataFrame()
        demote_mask = pd.Series(False, index=df.index)
    else:
        selected, frontier, demote_mask = select_greedy(
            df,
            label_col,
            action_col,
            candidates,
            int(args.max_fn_additional),
            int(args.max_rules),
        )

    df["exp014b_r3y_demote_confirm_to_approve"] = (
        normalize_action_series(df[action_col]).eq("CONFIRMAR")
        & demote_mask.fillna(False)
    ).astype(int)

    df["r3y_decisao_recommended"] = apply_demotions(
        df,
        action_col,
        df["exp014b_r3y_demote_confirm_to_approve"].eq(1),
    )

    df["exp014b_r3y_intervention_pred"] = action_to_intervention(df["r3y_decisao_recommended"])
    df["exp014b_r3y_block_pred"] = action_to_block(df["r3y_decisao_recommended"])

    final_intervention_metrics = metrics(y, df["exp014b_r3y_intervention_pred"])
    final_block_metrics = metrics(y, df["exp014b_r3y_block_pred"])
    final_by_action = metrics_by_action(df, label_col, "r3y_decisao_recommended")
    rob = robustness(df, label_col, action_col, "r3y_decisao_recommended")

    # Verify BLOQUEAR unchanged.
    block_unchanged = bool((safe_int_series(original_block) == safe_int_series(df["exp014b_r3y_block_pred"])).all())
    n_block_mismatches = int((safe_int_series(original_block) != safe_int_series(df["exp014b_r3y_block_pred"])).sum())

    fp_removed_total = int(baseline_intervention_metrics["fp"] - final_intervention_metrics["fp"])
    fn_added_total = int(final_intervention_metrics["fn"] - baseline_intervention_metrics["fn"])

    # Confirm queue before/after.
    before_confirm = baseline_by_action[baseline_by_action["action"].eq("CONFIRMAR")]
    after_confirm = final_by_action[final_by_action["action"].eq("CONFIRMAR")]

    def get_action_value(table: pd.DataFrame, col: str) -> int | float:
        if table.empty:
            return 0
        value = table.iloc[0][col]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        return value

    summary = {
        "experiment": EXPERIMENT,
        "status": "DONE",
        "objective_status": (
            "DONE_R3Y_CONFIRM_QUEUE_REDUCED_WITHIN_FN_BUDGET_BLOCK_UNCHANGED"
            if block_unchanged and fn_added_total <= int(args.max_fn_additional) and fp_removed_total > 0
            else "DONE_R3Y_NO_SAFE_CONFIRM_REDUCTION"
        ),
        "n_rows": int(len(df)),
        "n_frauds": int((y == 1).sum()),
        "n_normals": int((y == 0).sum()),
        "predictions_path": str(pred_path),
        "artifact_path": str(artifact_path) if artifact_path.exists() else None,
        "action_col": action_col,
        "block_col": block_col,
        "intervention_col": intervention_col,
        "baseline_intervention_metrics": baseline_intervention_metrics,
        "baseline_block_metrics": baseline_block_metrics,
        "final_intervention_metrics": final_intervention_metrics,
        "final_block_metrics": final_block_metrics,
        "fp_removed_total": fp_removed_total,
        "fn_added_total": fn_added_total,
        "block_unchanged": block_unchanged,
        "n_block_mismatches": n_block_mismatches,
        "confirm_before_n": int(get_action_value(before_confirm, "n_rows")),
        "confirm_before_frauds": int(get_action_value(before_confirm, "n_frauds")),
        "confirm_before_normals": int(get_action_value(before_confirm, "n_normals")),
        "confirm_after_n": int(get_action_value(after_confirm, "n_rows")),
        "confirm_after_frauds": int(get_action_value(after_confirm, "n_frauds")),
        "confirm_after_normals": int(get_action_value(after_confirm, "n_normals")),
        "n_candidates_evaluated": int(len(candidates)),
        "n_selected_rules": int(len(selected)),
        "max_fn_additional": int(args.max_fn_additional),
        "min_fp_removed": int(args.min_fp_removed),
        "all_pass": bool(block_unchanged and fn_added_total <= int(args.max_fn_additional)),
        "output_dir": str(out_dir),
    }

    contract = {
        "predictions_path": str(pred_path),
        "artifact_path": str(artifact_path) if artifact_path.exists() else None,
        "label_col": label_col,
        "action_col": action_col,
        "block_col": block_col,
        "intervention_col": intervention_col,
        "max_fn_additional": int(args.max_fn_additional),
        "min_fp_removed": int(args.min_fp_removed),
        "max_rules": int(args.max_rules),
        "missing": [],
        "contract_ok": True,
    }

    base_metrics = {
        "baseline_intervention_metrics": baseline_intervention_metrics,
        "baseline_block_metrics": baseline_block_metrics,
        "baseline_by_action": baseline_by_action.to_dict(orient="records"),
        "frozen_artifact_status": artifact.get("frozen_validation_status") if isinstance(artifact, dict) else None,
    }

    recommended_artifact = {
        "experiment": EXPERIMENT,
        "input_predictions_path": str(pred_path),
        "base_action_col": action_col,
        "final_action_col": "r3y_decisao_recommended",
        "demote_col": "exp014b_r3y_demote_confirm_to_approve",
        "intervention_pred_col": "exp014b_r3y_intervention_pred",
        "block_pred_col": "exp014b_r3y_block_pred",
        "baseline_intervention_metrics": baseline_intervention_metrics,
        "baseline_block_metrics": baseline_block_metrics,
        "final_intervention_metrics": final_intervention_metrics,
        "final_block_metrics": final_block_metrics,
        "fp_removed_total": fp_removed_total,
        "fn_added_total": fn_added_total,
        "block_unchanged": block_unchanged,
        "selected_demotions": selected.to_dict(orient="records") if not selected.empty else [],
        "notes": [
            "Only CONFIRMAR rows are eligible for demotion to APROVAR.",
            "BLOQUEAR must remain unchanged.",
            "Promotion requires frozen validation if the gain is accepted.",
        ],
    }

    write_json(out_dir / "00_run_summary.json", summary)
    write_json(out_dir / "01_input_contract.json", contract)
    write_json(out_dir / "02_base_frozen_metrics.json", base_metrics)
    candidates.to_csv(out_dir / "03_demote_candidates.csv", index=False, encoding="utf-8")
    frontier.to_csv(out_dir / "04_selection_frontier.csv", index=False, encoding="utf-8")
    selected.to_csv(out_dir / "05_selected_demotions.csv", index=False, encoding="utf-8")
    final_by_action.to_csv(out_dir / "06_decision_metrics_by_action.csv", index=False, encoding="utf-8")
    rob.to_csv(out_dir / "07_robustness_by_segment.csv", index=False, encoding="utf-8")
    write_json(out_dir / "08_policy_artifact_recommended.json", recommended_artifact)
    df.to_csv(out_dir / "09_predictions_recommended.csv", index=False, encoding="utf-8")

    selected_md = selected.to_markdown(index=False) if not selected.empty else "Nenhuma regra selecionada."
    frontier_md = frontier.to_markdown(index=False) if not frontier.empty else "Nenhuma seleção possível."
    final_by_action_md = final_by_action.to_markdown(index=False)

    report = f"""# {EXPERIMENT} - Confirm Queue Reduction

## Resultado executivo
- Status: `{summary["objective_status"]}`
- All pass: `{summary["all_pass"]}`
- BLOQUEAR unchanged: `{block_unchanged}`
- FP removidos total: `{fp_removed_total}`
- FN adicionais: `{fn_added_total}`

## Baseline intervenção
```json
{json.dumps(baseline_intervention_metrics, ensure_ascii=False, indent=2)}
```

## Final intervenção
```json
{json.dumps(final_intervention_metrics, ensure_ascii=False, indent=2)}
```

## Baseline BLOQUEAR
```json
{json.dumps(baseline_block_metrics, ensure_ascii=False, indent=2)}
```

## Final BLOQUEAR
```json
{json.dumps(final_block_metrics, ensure_ascii=False, indent=2)}
```

## Fila CONFIRMAR
```text
Antes:  n={summary["confirm_before_n"]}, fraudes={summary["confirm_before_frauds"]}, normais={summary["confirm_before_normals"]}
Depois: n={summary["confirm_after_n"]}, fraudes={summary["confirm_after_frauds"]}, normais={summary["confirm_after_normals"]}
```

## Métricas por ação final
{final_by_action_md}

## Regras selecionadas
{selected_md}

## Frontier de seleção
{frontier_md}

## Decisão sugerida
Se houver redução relevante de CONFIRMAR dentro do orçamento de FN e com BLOQUEAR intacto,
executar R3Y-FROZEN. Caso contrário, o gargalo exige novas features para distinguir
CONFIRMAR benigno de fraude residual.
"""
    (out_dir / "10_exp014b_r3y_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
