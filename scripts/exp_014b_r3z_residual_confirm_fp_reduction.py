# -*- coding: utf-8 -*-
"""
EXP-014B-R3Z - Residual Confirm FP Reduction after R3Y.

Starts from R3Y-FROZEN when available. Keeps BLOQUEAR unchanged and only
allows residual CONFIRMAR rows to be demoted to APROVAR.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT = "EXP-014B-R3Z"
LABEL_CANDIDATES = ["is_fraud", "fraude", "target", "label", "tp_fraude"]
ACTION_CANDIDATES = [
    "r3y_frozen_decisao_recommended",
    "r3y_decisao_recommended",
    "r3x_frozen_decisao_pos_policy",
    "r3x_decisao_pos_policy",
]
BLOCK_CANDIDATES = [
    "exp014b_r3y_frozen_block_pred",
    "exp014b_r3y_block_pred",
    "exp014b_r3x_frozen_block_pred",
    "exp014b_r3x_block_pred",
]
INTERVENTION_CANDIDATES = [
    "exp014b_r3y_frozen_intervention_pred",
    "exp014b_r3y_intervention_pred",
    "exp014b_r3x_frozen_intervention_pred",
    "exp014b_r3x_intervention_pred",
]
SCORE_COLS = [
    "lgbm_r4_score", "score_final", "lgbm_raw", "lgbm_mapped", "peso_total",
    "if_percentile", "se_score", "beh_score", "behavioral_score", "topaz_risk_score",
    "exp014b_r3s_second_stage_score", "exp014b_r3u_receiver_relationship_trust_score",
]
CATEGORICAL_COLS = [
    "ds_tipo_chave_norm", "value_band", "periodo_dia", "score_bin", "lgbm_bin",
    "if_bin", "ratio_bin", "qtd_rec_bin", "valor_rec_bin", "module_quiet",
    "se_worst_pattern", "mbk_available_flag", "first_receiver_flag_real",
    "r3u_missing_receiver_history_flag", "r3u_receiver_known_flag", "r3u_receiver_reputable_flag",
    "r3u_receiver_strong_flag", "r3u_relationship_known_flag", "r3u_relationship_recurrent_flag",
    "r3u_relationship_strong_flag", "r3u_first_receiver_flag", "r3u_module_quiet_flag",
    "r3u_se_missing_flag", "r3u_ratio_lt_005_flag", "r3u_mbk_quality_flag",
    "r3u_receiver_trust_bucket", "r3u_relationship_bucket",
]
SEGMENT_COLS = [
    "temporal_split", "event_month", "ds_tipo_chave_norm", "value_band", "periodo_dia",
    "score_bin", "lgbm_bin", "if_bin", "ratio_bin", "qtd_rec_bin", "valor_rec_bin",
    "module_quiet", "se_worst_pattern", "mbk_available_flag", "first_receiver_flag_real",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=str, default=None)
    p.add_argument("--artifact", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--max-fn-additional", type=int, default=5)
    p.add_argument("--min-incremental-fp", type=int, default=5)
    p.add_argument("--max-rules", type=int, default=40)
    p.add_argument("--max-candidates", type=int, default=4000)
    p.add_argument("--enable-triplets", action="store_true")
    return p.parse_args()


def default_paths() -> tuple[Path, Path | None, Path]:
    root = Path.cwd()
    frozen_pred = root / "resultados" / "experimentos" / "EXP-014B-R3Y-FROZEN" / "06_predictions_frozen.csv"
    rec_pred = root / "resultados" / "experimentos" / "EXP-014B-R3Y" / "09_predictions_recommended.csv"
    pred = frozen_pred if frozen_pred.exists() else rec_pred
    artifact = root / "resultados" / "experimentos" / "EXP-014B-R3Y-FROZEN" / "05_policy_artifact_frozen.json"
    if not artifact.exists():
        artifact = root / "resultados" / "experimentos" / "EXP-014B-R3Y" / "08_policy_artifact_recommended.json"
    out = root / "resultados" / "experimentos" / EXPERIMENT
    return pred, artifact if artifact.exists() else None, out


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(f"No column found among: {candidates}")
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
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(float(precision), 8),
        "recall": round(float(recall), 8),
        "f1": round(float(f1), 8),
        "fpr": round(float(fpr), 8),
    }


def normalize_action(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def action_to_intervention(action: pd.Series) -> pd.Series:
    return action.astype(str).str.upper().isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)


def action_to_block(action: pd.Series) -> pd.Series:
    return action.astype(str).str.upper().eq("BLOQUEAR").astype(int)


def candidate_row(df: pd.DataFrame, label_col: str, confirm: pd.Series, cond: pd.Series,
                  candidate_id: str, rule_type: str, description: str) -> dict[str, Any] | None:
    y = safe_int_series(df[label_col])
    mask = confirm & cond.fillna(False)
    n = int(mask.sum())
    if n == 0:
        return None
    fp = int(((y == 0) & mask).sum())
    tp_loss = int(((y == 1) & mask).sum())
    return {
        "candidate_id": candidate_id,
        "rule_type": rule_type,
        "description": description,
        "n_demoted": n,
        "fp_removed": fp,
        "tp_loss": tp_loss,
        "precision_demoted": round(float(tp_loss / n), 8) if n else 0.0,
        "fp_per_tp_loss": round(float(fp / max(tp_loss, 1)), 8),
    }


def mask_from_description(df: pd.DataFrame, description: str) -> pd.Series:
    prefix = "Demover CONFIRMAR residual com "
    if not description.startswith(prefix):
        raise ValueError(f"Unsupported description: {description}")
    expr = description[len(prefix):]
    if " <= " in expr and " AND " not in expr:
        col, th = expr.split(" <= ", 1)
        return pd.to_numeric(df[col], errors="coerce").le(float(th))
    if " >= " in expr and " AND " not in expr:
        col, th = expr.split(" >= ", 1)
        return pd.to_numeric(df[col], errors="coerce").ge(float(th))
    if " AND " in expr:
        mask = pd.Series(True, index=df.index)
        for part in expr.split(" AND "):
            col, val = part.split(" == ", 1)
            mask &= df[col].fillna("<MISSING>").astype(str).eq(str(val))
        return mask
    if " == " in expr:
        col, val = expr.split(" == ", 1)
        return df[col].fillna("<MISSING>").astype(str).eq(str(val))
    raise ValueError(f"Unsupported expression: {expr}")


def build_candidates(df: pd.DataFrame, label_col: str, action_col: str,
                     min_fp: int, max_fn: int, max_candidates: int,
                     enable_triplets: bool) -> pd.DataFrame:
    action = normalize_action(df[action_col])
    confirm = action.eq("CONFIRMAR")
    rows: list[dict[str, Any]] = []
    score_cols = [c for c in SCORE_COLS if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    quantiles = [0.005,0.01,0.02,0.03,0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.88,0.90,0.92,0.95,0.97,0.98,0.99,0.995]

    for col in score_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s[confirm & s.notna()]
        if valid.empty:
            continue
        thresholds = sorted(set(float(valid.quantile(q)) for q in quantiles if pd.notna(valid.quantile(q))))
        for th in thresholds:
            for direction, cond, op in [("score_lo", s.le(th), "<="), ("score_hi", s.ge(th), ">=")]:
                row = candidate_row(df, label_col, confirm, cond, f"{direction}__{col}__{th:.12g}", "score_threshold", f"Demover CONFIRMAR residual com {col} {op} {th:.12g}")
                if row:
                    rows.append(row)

    useful_cats = [c for c in cat_cols if df.loc[confirm, c].nunique(dropna=False) <= 40]

    for col in cat_cols:
        vals = df.loc[confirm, col].fillna("<MISSING>").astype(str)
        for val, count in vals.value_counts(dropna=False).items():
            if int(count) < min_fp:
                continue
            cond = df[col].fillna("<MISSING>").astype(str).eq(str(val))
            row = candidate_row(df, label_col, confirm, cond, f"cat__{col}__{str(val)[:80]}", "categorical_segment", f"Demover CONFIRMAR residual com {col} == {val}")
            if row:
                rows.append(row)

    for c1, c2 in itertools.combinations(useful_cats, 2):
        tmp = df.loc[confirm, [c1, c2]].fillna("<MISSING>").astype(str)
        for (v1, v2), count in tmp.value_counts(dropna=False).head(300).items():
            if int(count) < min_fp:
                continue
            cond = df[c1].fillna("<MISSING>").astype(str).eq(str(v1)) & df[c2].fillna("<MISSING>").astype(str).eq(str(v2))
            row = candidate_row(df, label_col, confirm, cond, f"cat2__{c1}={str(v1)[:30]}__{c2}={str(v2)[:30]}", "categorical_pair", f"Demover CONFIRMAR residual com {c1} == {v1} AND {c2} == {v2}")
            if row:
                rows.append(row)

    for cat_col in useful_cats[:20]:
        vals = df.loc[confirm, cat_col].fillna("<MISSING>").astype(str).value_counts(dropna=False).head(20)
        for score_col in score_cols:
            s = pd.to_numeric(df[score_col], errors="coerce")
            valid = s[confirm & s.notna()]
            if valid.empty:
                continue
            thresholds = sorted(set(float(valid.quantile(q)) for q in [0.10,0.20,0.30,0.70,0.80,0.90,0.95]))
            for val, count in vals.items():
                if int(count) < min_fp:
                    continue
                base_cond = df[cat_col].fillna("<MISSING>").astype(str).eq(str(val))
                for th in thresholds:
                    for direction, score_cond, op in [("scorecat_lo", s.le(th), "<="), ("scorecat_hi", s.ge(th), ">=")]:
                        cond = base_cond & score_cond
                        row = candidate_row(df, label_col, confirm, cond, f"{direction}__{cat_col}={str(val)[:30]}__{score_col}__{th:.12g}", "score_categorical_intersection", f"Demover CONFIRMAR residual com {cat_col} == {val} AND {score_col} {op} {th:.12g}")
                        if row:
                            rows.append(row)

    if enable_triplets:
        for c1, c2, c3 in itertools.combinations(useful_cats[:12], 3):
            tmp = df.loc[confirm, [c1, c2, c3]].fillna("<MISSING>").astype(str)
            for (v1, v2, v3), count in tmp.value_counts(dropna=False).head(150).items():
                if int(count) < min_fp:
                    continue
                cond = (df[c1].fillna("<MISSING>").astype(str).eq(str(v1)) &
                        df[c2].fillna("<MISSING>").astype(str).eq(str(v2)) &
                        df[c3].fillna("<MISSING>").astype(str).eq(str(v3)))
                row = candidate_row(df, label_col, confirm, cond, f"cat3__{c1}={str(v1)[:20]}__{c2}={str(v2)[:20]}__{c3}={str(v3)[:20]}", "categorical_triplet", f"Demover CONFIRMAR residual com {c1} == {v1} AND {c2} == {v2} AND {c3} == {v3}")
                if row:
                    rows.append(row)

    if not rows:
        return pd.DataFrame()
    cand = pd.DataFrame(rows).drop_duplicates(subset=["description"])
    cand = cand[(cand["fp_removed"] >= int(min_fp)) & (cand["tp_loss"] <= int(max_fn))].copy()
    if cand.empty:
        return cand
    cand = cand.sort_values(["tp_loss", "fp_removed", "fp_per_tp_loss", "n_demoted"], ascending=[True, False, False, False]).head(max_candidates)
    return cand.reset_index(drop=True)


def select_greedy(df: pd.DataFrame, label_col: str, action_col: str,
                  candidates: pd.DataFrame, max_fn: int, max_rules: int,
                  min_fp: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    y = safe_int_series(df[label_col])
    action = normalize_action(df[action_col])
    confirm = action.eq("CONFIRMAR")
    selected_rows = []
    frontier_rows = []
    cumulative = pd.Series(False, index=df.index)
    cum_fn = 0
    cum_fp = 0
    remaining = candidates.copy().reset_index(drop=True)

    for step in range(1, int(max_rules) + 1):
        best = None
        best_mask = None
        best_score = None
        for _, row in remaining.iterrows():
            try:
                mask = confirm & mask_from_description(df, str(row["description"])) & (~cumulative)
            except Exception:
                continue
            n = int(mask.sum())
            if n == 0:
                continue
            fp_gain = int(((y == 0) & mask).sum())
            fn_gain = int(((y == 1) & mask).sum())
            if fp_gain < int(min_fp):
                continue
            if cum_fn + fn_gain > int(max_fn):
                continue
            score = (1 if fn_gain == 0 else 0, fp_gain / max(fn_gain, 1), fp_gain, -fn_gain, n)
            if best is None or score > best_score:
                best = row.copy()
                best_mask = mask
                best_score = score
                best["incremental_n_demoted"] = n
                best["incremental_fp_removed"] = fp_gain
                best["incremental_tp_loss"] = fn_gain
        if best is None or best_mask is None:
            break
        cumulative |= best_mask
        cum_fn += int(best["incremental_tp_loss"])
        cum_fp += int(best["incremental_fp_removed"])
        best["selection_step"] = step
        best["cumulative_fp_removed"] = cum_fp
        best["cumulative_tp_loss"] = cum_fn
        selected_rows.append(best)
        frontier_rows.append({
            "selection_step": step,
            "selected_candidate_id": best["candidate_id"],
            "selected_description": best["description"],
            "incremental_n_demoted": int(best["incremental_n_demoted"]),
            "incremental_fp_removed": int(best["incremental_fp_removed"]),
            "incremental_tp_loss": int(best["incremental_tp_loss"]),
            "cumulative_n_demoted": int(cumulative.sum()),
            "cumulative_fp_removed": int(cum_fp),
            "cumulative_tp_loss": int(cum_fn),
        })
        remaining = remaining[remaining["description"] != best["description"]].reset_index(drop=True)
    return pd.DataFrame(selected_rows), pd.DataFrame(frontier_rows), cumulative


def apply_demotions(df: pd.DataFrame, action_col: str, demote: pd.Series) -> pd.Series:
    action = normalize_action(df[action_col]).copy()
    final = action.copy()
    final[action.eq("CONFIRMAR") & demote.fillna(False)] = "APROVAR"
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
        rows.append({"action": str(action), "n_rows": n, "n_frauds": frauds, "n_normals": normals, "precision_within_action": round(float(frauds / n), 8) if n else 0.0})
    return pd.DataFrame(rows).sort_values("action")


def robustness(df: pd.DataFrame, label_col: str, before_col: str, after_col: str) -> pd.DataFrame:
    rows = []
    y = safe_int_series(df[label_col])
    before = action_to_intervention(df[before_col])
    after = action_to_intervention(df[after_col])
    for col in SEGMENT_COLS:
        if col not in df.columns:
            continue
        for val, idx in df.groupby(col, dropna=False).groups.items():
            idx = list(idx)
            yy = y.loc[idx]
            bm = metrics(yy, before.loc[idx])
            am = metrics(yy, after.loc[idx])
            rows.append({
                "segment_col": col,
                "segment_value": str(val),
                "n_rows": int(len(idx)),
                "n_frauds": int((yy == 1).sum()),
                "fp_removed": int(bm["fp"] - am["fp"]),
                "tp_loss": int(bm["tp"] - am["tp"]),
                "before_fp": bm["fp"],
                "after_fp": am["fp"],
                "before_tp": bm["tp"],
                "after_tp": am["tp"],
                "after_fn": am["fn"],
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
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    df = pd.read_csv(pred_path, low_memory=False)
    artifact = None
    if artifact_path and Path(artifact_path).exists():
        artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    label_col = find_col(df, LABEL_CANDIDATES)
    action_col = find_col(df, ACTION_CANDIDATES)
    block_col = find_col(df, BLOCK_CANDIDATES, required=False)
    intervention_col = find_col(df, INTERVENTION_CANDIDATES, required=False)

    action = normalize_action(df[action_col])
    original_block = safe_int_series(df[block_col]) if block_col else action_to_block(action)
    original_intervention = safe_int_series(df[intervention_col]) if intervention_col else action_to_intervention(action)
    y = safe_int_series(df[label_col])
    base_intervention = metrics(y, original_intervention)
    base_block = metrics(y, original_block)
    base_by_action = metrics_by_action(df.assign(_action=action), label_col, "_action")

    candidates = build_candidates(df, label_col, action_col, int(args.min_incremental_fp), int(args.max_fn_additional), int(args.max_candidates), bool(args.enable_triplets))
    if candidates.empty:
        selected = pd.DataFrame()
        frontier = pd.DataFrame()
        demote = pd.Series(False, index=df.index)
    else:
        selected, frontier, demote = select_greedy(df, label_col, action_col, candidates, int(args.max_fn_additional), int(args.max_rules), int(args.min_incremental_fp))

    df["exp014b_r3z_demote_confirm_to_approve"] = (normalize_action(df[action_col]).eq("CONFIRMAR") & demote.fillna(False)).astype(int)
    df["r3z_decisao_recommended"] = apply_demotions(df, action_col, df["exp014b_r3z_demote_confirm_to_approve"].eq(1))
    df["exp014b_r3z_intervention_pred"] = action_to_intervention(df["r3z_decisao_recommended"])
    df["exp014b_r3z_block_pred"] = action_to_block(df["r3z_decisao_recommended"])

    final_intervention = metrics(y, df["exp014b_r3z_intervention_pred"])
    final_block = metrics(y, df["exp014b_r3z_block_pred"])
    final_by_action = metrics_by_action(df, label_col, "r3z_decisao_recommended")
    rob = robustness(df, label_col, action_col, "r3z_decisao_recommended")

    block_unchanged = bool((safe_int_series(original_block) == safe_int_series(df["exp014b_r3z_block_pred"])).all())
    n_block_mismatches = int((safe_int_series(original_block) != safe_int_series(df["exp014b_r3z_block_pred"])).sum())
    fp_removed = int(base_intervention["fp"] - final_intervention["fp"])
    fn_added = int(final_intervention["fn"] - base_intervention["fn"])

    before_confirm = base_by_action[base_by_action["action"].eq("CONFIRMAR")]
    after_confirm = final_by_action[final_by_action["action"].eq("CONFIRMAR")]
    def aval(table: pd.DataFrame, col: str) -> int:
        return 0 if table.empty else int(table.iloc[0][col])

    summary = {
        "experiment": EXPERIMENT,
        "status": "DONE",
        "objective_status": "DONE_R3Z_RESIDUAL_CONFIRM_REDUCED_WITHIN_FN_BUDGET_BLOCK_UNCHANGED" if block_unchanged and fn_added <= int(args.max_fn_additional) and fp_removed > 0 else "DONE_R3Z_NO_SAFE_RESIDUAL_CONFIRM_REDUCTION",
        "n_rows": int(len(df)),
        "n_frauds": int((y == 1).sum()),
        "n_normals": int((y == 0).sum()),
        "predictions_path": str(pred_path),
        "artifact_path": str(artifact_path) if artifact_path else None,
        "action_col": action_col,
        "block_col": block_col,
        "intervention_col": intervention_col,
        "baseline_intervention_metrics": base_intervention,
        "baseline_block_metrics": base_block,
        "final_intervention_metrics": final_intervention,
        "final_block_metrics": final_block,
        "fp_removed_total": fp_removed,
        "fn_added_total": fn_added,
        "block_unchanged": block_unchanged,
        "n_block_mismatches": n_block_mismatches,
        "confirm_before_n": aval(before_confirm, "n_rows"),
        "confirm_before_frauds": aval(before_confirm, "n_frauds"),
        "confirm_before_normals": aval(before_confirm, "n_normals"),
        "confirm_after_n": aval(after_confirm, "n_rows"),
        "confirm_after_frauds": aval(after_confirm, "n_frauds"),
        "confirm_after_normals": aval(after_confirm, "n_normals"),
        "n_candidates_evaluated": int(len(candidates)),
        "n_selected_rules": int(len(selected)),
        "max_fn_additional": int(args.max_fn_additional),
        "min_incremental_fp": int(args.min_incremental_fp),
        "max_rules": int(args.max_rules),
        "enable_triplets": bool(args.enable_triplets),
        "all_pass": bool(block_unchanged and fn_added <= int(args.max_fn_additional)),
        "output_dir": str(out_dir),
    }
    contract = {
        "predictions_path": str(pred_path),
        "artifact_path": str(artifact_path) if artifact_path else None,
        "label_col": label_col,
        "action_col": action_col,
        "block_col": block_col,
        "intervention_col": intervention_col,
        "max_fn_additional": int(args.max_fn_additional),
        "min_incremental_fp": int(args.min_incremental_fp),
        "max_rules": int(args.max_rules),
        "enable_triplets": bool(args.enable_triplets),
        "contract_ok": True,
        "missing": [],
    }
    base_metrics = {
        "baseline_intervention_metrics": base_intervention,
        "baseline_block_metrics": base_block,
        "baseline_by_action": base_by_action.to_dict(orient="records"),
        "artifact_status": artifact.get("frozen_validation_status") if isinstance(artifact, dict) else None,
    }
    recommended_artifact = {
        "experiment": EXPERIMENT,
        "input_predictions_path": str(pred_path),
        "base_action_col": action_col,
        "final_action_col": "r3z_decisao_recommended",
        "demote_col": "exp014b_r3z_demote_confirm_to_approve",
        "intervention_pred_col": "exp014b_r3z_intervention_pred",
        "block_pred_col": "exp014b_r3z_block_pred",
        "baseline_intervention_metrics": base_intervention,
        "baseline_block_metrics": base_block,
        "final_intervention_metrics": final_intervention,
        "final_block_metrics": final_block,
        "fp_removed_total": fp_removed,
        "fn_added_total": fn_added,
        "block_unchanged": block_unchanged,
        "selected_demotions": selected.to_dict(orient="records") if not selected.empty else [],
        "notes": [
            "Only residual CONFIRMAR rows are eligible for demotion to APROVAR.",
            "BLOQUEAR must remain unchanged.",
            "R3Y demotions are treated as frozen in the input baseline.",
            "Promotion requires R3Z-FROZEN validation if the gain is accepted.",
        ],
    }

    write_json(out_dir / "00_run_summary.json", summary)
    write_json(out_dir / "01_input_contract.json", contract)
    write_json(out_dir / "02_base_metrics.json", base_metrics)
    candidates.to_csv(out_dir / "03_residual_candidates.csv", index=False, encoding="utf-8")
    frontier.to_csv(out_dir / "04_selection_frontier.csv", index=False, encoding="utf-8")
    selected.to_csv(out_dir / "05_selected_demotions.csv", index=False, encoding="utf-8")
    final_by_action.to_csv(out_dir / "06_decision_metrics_by_action.csv", index=False, encoding="utf-8")
    rob.to_csv(out_dir / "07_robustness_by_segment.csv", index=False, encoding="utf-8")
    write_json(out_dir / "08_policy_artifact_recommended.json", recommended_artifact)
    df.to_csv(out_dir / "09_predictions_recommended.csv", index=False, encoding="utf-8")

    selected_md = selected.to_markdown(index=False) if not selected.empty else "Nenhuma regra selecionada."
    frontier_md = frontier.to_markdown(index=False) if not frontier.empty else "Nenhuma selecao possivel."
    final_by_action_md = final_by_action.to_markdown(index=False)
    report = f"""# {EXPERIMENT} - Residual Confirm FP Reduction\n\n## Resultado executivo\n- Status: `{summary['objective_status']}`\n- All pass: `{summary['all_pass']}`\n- BLOQUEAR unchanged: `{block_unchanged}`\n- FP removidos total: `{fp_removed}`\n- FN adicionais: `{fn_added}`\n\n## Baseline intervencao\n```json\n{json.dumps(base_intervention, ensure_ascii=False, indent=2)}\n```\n\n## Final intervencao\n```json\n{json.dumps(final_intervention, ensure_ascii=False, indent=2)}\n```\n\n## Baseline BLOQUEAR\n```json\n{json.dumps(base_block, ensure_ascii=False, indent=2)}\n```\n\n## Final BLOQUEAR\n```json\n{json.dumps(final_block, ensure_ascii=False, indent=2)}\n```\n\n## Fila CONFIRMAR residual\n```text\nAntes:  n={summary['confirm_before_n']}, fraudes={summary['confirm_before_frauds']}, normais={summary['confirm_before_normals']}\nDepois: n={summary['confirm_after_n']}, fraudes={summary['confirm_after_frauds']}, normais={summary['confirm_after_normals']}\n```\n\n## Metricas por acao final\n{final_by_action_md}\n\n## Regras selecionadas\n{selected_md}\n\n## Frontier de selecao\n{frontier_md}\n\n## Decisao sugerida\nSe houver reducao relevante dentro do orcamento e com BLOQUEAR intacto, executar R3Z-FROZEN. Caso contrario, consolidar R3Y-FROZEN e partir para novas features.\n"""
    (out_dir / "10_exp014b_r3z_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
