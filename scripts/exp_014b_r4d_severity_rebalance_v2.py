# -*- coding: utf-8 -*-
"""
EXP-014B-R4D — Severity Rebalance after R4C v2

Objetivo:
  Partir do R4C-FROZEN e rebalancear a severidade operacional sem alterar
  a intervenção total:
    - mover normais prováveis de BLOQUEAR -> CONFIRMAR;
    - mover fraudes prováveis de CONFIRMAR -> BLOQUEAR;
    - manter APROVAR intocado;
    - manter FN/intervenção total do R4C.

Uso recomendado:
  python scripts\exp_014b_r4d_severity_rebalance_v2.py --enable-quads --max-block-tp-demoted 0 --max-confirm-fp-promoted 80 --max-rules-block 120 --max-rules-confirm 120 --min-support 1

Saídas:
  resultados/experimentos/EXP-014B-R4D/
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT = "EXP-014B-R4D"

LABELS = ["is_fraud", "fraude", "target", "label", "tp_fraude"]

ACTION_COLS = [
    "r4c_frozen_decisao_recommended",
    "r4c_decisao_recommended",
    "r4a_frozen_decisao_recommended",
    "r4a_decisao_recommended",
]

INTERVENTION_COLS = [
    "exp014b_r4c_frozen_intervention_pred",
    "exp014b_r4c_intervention_pred",
    "exp014b_r4a_frozen_intervention_pred",
    "exp014b_r4a_intervention_pred",
]

BLOCK_COLS = [
    "exp014b_r4c_frozen_block_pred",
    "exp014b_r4c_block_pred",
    "exp014b_r4a_frozen_block_pred",
    "exp014b_r4a_block_pred",
]

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

ROBUSTNESS_COLS = [
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
    "mbk_available_flag",
    "first_receiver_flag_real",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default=None)
    p.add_argument("--artifact", default=None)
    p.add_argument("--output-dir", default=None)

    p.add_argument("--max-block-tp-demoted", type=int, default=0)
    p.add_argument("--max-confirm-fp-promoted", type=int, default=80)
    p.add_argument("--target-confirm-tp-promoted", type=int, default=172)

    p.add_argument("--max-rules-block", type=int, default=120)
    p.add_argument("--max-rules-confirm", type=int, default=120)
    p.add_argument("--max-candidates", type=int, default=12000)

    p.add_argument("--min-support", type=int, default=1)
    p.add_argument("--min-incremental-good", type=int, default=1)
    p.add_argument("--enable-quads", action="store_true")
    p.add_argument("--combo-topn", type=int, default=500)
    p.add_argument("--score-cat-top-values", type=int, default=30)
    return p.parse_args()


def default_paths() -> tuple[Path, Path | None, Path]:
    root = Path.cwd()

    pred = root / "resultados" / "experimentos" / "EXP-014B-R4C-FROZEN" / "06_predictions_frozen.csv"
    if not pred.exists():
        pred = root / "resultados" / "experimentos" / "EXP-014B-R4C" / "09_predictions_recommended.csv"

    artifact = root / "resultados" / "experimentos" / "EXP-014B-R4C-FROZEN" / "05_policy_artifact_frozen.json"
    if not artifact.exists():
        artifact = root / "resultados" / "experimentos" / "EXP-014B-R4C" / "08_policy_artifact_recommended.json"

    out = root / "resultados" / "experimentos" / EXPERIMENT
    return pred, artifact if artifact.exists() else None, out


def find_col(df: pd.DataFrame, names: list[str], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    if required:
        raise KeyError(f"Nenhuma coluna encontrada entre: {names}")
    return None


def ints(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def actions(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def intervention_from_action(action: pd.Series) -> pd.Series:
    return actions(action).isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)


def block_from_action(action: pd.Series) -> pd.Series:
    return actions(action).eq("BLOQUEAR").astype(int)


def metrics(y_true: pd.Series, pred: pd.Series) -> dict[str, Any]:
    y = ints(y_true)
    p = ints(pred)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
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


def action_table(df: pd.DataFrame, label_col: str, action_col: str) -> pd.DataFrame:
    y = ints(df[label_col])
    rows = []
    for action, idx in df.groupby(action_col, dropna=False).groups.items():
        idx = list(idx)
        yy = y.loc[idx]
        n = int(len(idx))
        frauds = int((yy == 1).sum())
        normals = int((yy == 0).sum())
        rows.append(
            {
                "action": str(action),
                "n_rows": n,
                "n_frauds": frauds,
                "n_normals": normals,
                "precision_within_action": round(float(frauds / n), 8) if n else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("action")


def add_candidate(
    rows: list[dict[str, Any]],
    masks: dict[str, np.ndarray],
    seen: set[str],
    candidate_id: str,
    rule_type: str,
    description: str,
    local_mask: np.ndarray,
    y_local: np.ndarray,
    good_label: int,
    min_support: int,
    min_good: int,
) -> None:
    if description in seen:
        return

    n = int(local_mask.sum())
    if n < min_support:
        return

    yy = y_local[local_mask]
    good = int((yy == good_label).sum())
    bad = int((yy != good_label).sum())

    if good < min_good:
        return

    seen.add(description)
    rows.append(
        {
            "candidate_id": candidate_id,
            "rule_type": rule_type,
            "description": description,
            "n_affected": n,
            "good_count": good,
            "bad_count": bad,
            "precision_for_goal": round(float(good / n), 8) if n else 0.0,
            "good_per_bad": round(float(good / max(bad, 1)), 8),
        }
    )
    masks[candidate_id] = local_mask.copy()


def mine_candidates(
    df: pd.DataFrame,
    eligible_idx: np.ndarray,
    y_all: np.ndarray,
    prefix: str,
    phase_name: str,
    good_label: int,
    cat_cols: list[str],
    score_cols: list[str],
    min_support: int,
    min_good: int,
    max_candidates: int,
    enable_quads: bool,
    combo_topn: int,
    score_cat_top_values: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    local = df.iloc[eligible_idx].copy()
    y_local = y_all[eligible_idx]

    rows: list[dict[str, Any]] = []
    masks: dict[str, np.ndarray] = {}
    seen: set[str] = set()

    # 1) Score thresholds.
    quantiles = [
        0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10,
        0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60,
        0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97,
        0.98, 0.99, 0.995,
    ]

    for score_col in score_cols:
        s = pd.to_numeric(local[score_col], errors="coerce")
        valid = s.dropna()
        if valid.empty:
            continue

        values = s.to_numpy()
        thresholds = sorted(set(float(valid.quantile(q)) for q in quantiles if pd.notna(valid.quantile(q))))

        for th in thresholds:
            for op, mask in [
                ("<=", np.isfinite(values) & (values <= th)),
                (">=", np.isfinite(values) & (values >= th)),
            ]:
                add_candidate(
                    rows,
                    masks,
                    seen,
                    f"{phase_name}_score__{score_col}__{op}{th:.12g}",
                    "score_threshold",
                    f"{prefix} com {score_col} {op} {th:.12g}",
                    mask,
                    y_local,
                    good_label,
                    min_support,
                    min_good,
                )

    # 2) Categorical combinations.
    combo_plan = [(1, 28, combo_topn), (2, 24, combo_topn), (3, 16, combo_topn)]
    if enable_quads:
        combo_plan.append((4, 12, max(150, combo_topn // 2)))

    for size, max_cols, topn in combo_plan:
        for cols in itertools.combinations(cat_cols[:max_cols], size):
            tmp = local[list(cols)].fillna("<MISSING>").astype(str)
            vc = tmp.value_counts(dropna=False).head(topn)

            for vals, support in vc.items():
                if int(support) < min_support:
                    continue

                vals_tuple = vals if isinstance(vals, tuple) else (vals,)
                mask = np.ones(len(local), dtype=bool)
                parts = []
                safe_parts = []

                for col, val in zip(cols, vals_tuple):
                    val = str(val)
                    mask &= tmp[col].to_numpy() == val
                    parts.append(f"{col} == {val}")
                    safe_parts.append(f"{col}={val[:20]}")

                add_candidate(
                    rows,
                    masks,
                    seen,
                    f"{phase_name}_cat{size}__" + "__".join(safe_parts),
                    f"categorical_{size}",
                    f"{prefix} com " + " AND ".join(parts),
                    mask,
                    y_local,
                    good_label,
                    min_support,
                    min_good,
                )

    # 3) Score + one categorical value.
    score_qs = [0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90, 0.95]

    for score_col in score_cols[:8]:
        s = pd.to_numeric(local[score_col], errors="coerce")
        valid = s.dropna()
        if valid.empty:
            continue

        values = s.to_numpy()
        thresholds = sorted(set(float(valid.quantile(q)) for q in score_qs if pd.notna(valid.quantile(q))))

        for cat_col in cat_cols[:18]:
            cat = local[cat_col].fillna("<MISSING>").astype(str)
            cat_values = cat.to_numpy()
            top_values = cat.value_counts(dropna=False).head(score_cat_top_values)

            for cat_val, support in top_values.items():
                if int(support) < min_support:
                    continue

                base = cat_values == str(cat_val)

                for th in thresholds:
                    for op, smask in [
                        ("<=", np.isfinite(values) & (values <= th)),
                        (">=", np.isfinite(values) & (values >= th)),
                    ]:
                        add_candidate(
                            rows,
                            masks,
                            seen,
                            f"{phase_name}_scorecat__{cat_col}={str(cat_val)[:20]}__{score_col}__{op}{th:.12g}",
                            "score_cat_1",
                            f"{prefix} com {cat_col} == {cat_val} AND {score_col} {op} {th:.12g}",
                            base & smask,
                            y_local,
                            good_label,
                            min_support,
                            min_good,
                        )

    if not rows:
        return pd.DataFrame(), {}

    candidates = pd.DataFrame(rows).drop_duplicates(subset=["description"])
    candidates = candidates.sort_values(
        ["bad_count", "good_count", "good_per_bad", "n_affected"],
        ascending=[True, False, False, False],
    ).head(max_candidates).reset_index(drop=True)

    keep = set(candidates["candidate_id"].astype(str))
    masks = {k: v for k, v in masks.items() if k in keep}

    return candidates, masks


def greedy_select(
    candidates: pd.DataFrame,
    masks: dict[str, np.ndarray],
    y_local: np.ndarray,
    good_label: int,
    max_bad_total: int,
    max_rules: int,
    min_good: int,
    target_good: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    selected_mask = np.zeros(len(y_local), dtype=bool)
    selected_rows: list[pd.Series] = []
    frontier: list[dict[str, Any]] = []

    cumulative_good = 0
    cumulative_bad = 0
    remaining = candidates.copy().reset_index(drop=True)

    for step in range(1, int(max_rules) + 1):
        if target_good is not None and cumulative_good >= target_good:
            break

        best = None
        best_mask = None
        best_score = None

        for _, row in remaining.iterrows():
            cid = str(row["candidate_id"])
            mask = masks.get(cid)
            if mask is None:
                continue

            incremental = mask & (~selected_mask)
            n = int(incremental.sum())
            if n == 0:
                continue

            yy = y_local[incremental]
            good = int((yy == good_label).sum())
            bad = int((yy != good_label).sum())

            if good < min_good:
                continue
            if cumulative_bad + bad > max_bad_total:
                continue

            score = (
                1 if bad == 0 else 0,
                good / max(bad, 1),
                good,
                -bad,
                -n,
            )

            if best is None or score > best_score:
                best = row.copy()
                best_mask = incremental
                best_score = score
                best["incremental_n"] = n
                best["incremental_good"] = good
                best["incremental_bad"] = bad

        if best is None or best_mask is None:
            break

        selected_mask |= best_mask
        cumulative_good += int(best["incremental_good"])
        cumulative_bad += int(best["incremental_bad"])

        best["selection_step"] = step
        best["cumulative_n"] = int(selected_mask.sum())
        best["cumulative_good"] = int(cumulative_good)
        best["cumulative_bad"] = int(cumulative_bad)
        selected_rows.append(best)

        frontier.append(
            {
                "selection_step": step,
                "selected_candidate_id": str(best["candidate_id"]),
                "selected_description": str(best["description"]),
                "incremental_n": int(best["incremental_n"]),
                "incremental_good": int(best["incremental_good"]),
                "incremental_bad": int(best["incremental_bad"]),
                "cumulative_n": int(selected_mask.sum()),
                "cumulative_good": int(cumulative_good),
                "cumulative_bad": int(cumulative_bad),
            }
        )

        remaining = remaining[remaining["candidate_id"].astype(str) != str(best["candidate_id"])].reset_index(drop=True)

    selected = pd.DataFrame(selected_rows) if selected_rows else pd.DataFrame()
    frontier_df = pd.DataFrame(frontier) if frontier else pd.DataFrame()
    return selected, frontier_df, selected_mask


def robustness_by_segment(df: pd.DataFrame, label_col: str, before_action_col: str, after_action_col: str) -> pd.DataFrame:
    y = ints(df[label_col])
    before_block = block_from_action(df[before_action_col])
    after_block = block_from_action(df[after_action_col])

    rows = []
    for col in ROBUSTNESS_COLS:
        if col not in df.columns:
            continue

        for val, idx in df.groupby(col, dropna=False).groups.items():
            idx = list(idx)
            yy = y.loc[idx]
            before_m = metrics(yy, before_block.loc[idx])
            after_m = metrics(yy, after_block.loc[idx])

            rows.append(
                {
                    "segment_col": col,
                    "segment_value": str(val),
                    "n_rows": int(len(idx)),
                    "block_tp_delta": int(after_m["tp"] - before_m["tp"]),
                    "block_fp_delta": int(after_m["fp"] - before_m["fp"]),
                    "before_block_tp": before_m["tp"],
                    "after_block_tp": after_m["tp"],
                    "before_block_fp": before_m["fp"],
                    "after_block_fp": after_m["fp"],
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["block_tp_delta", "block_fp_delta", "n_rows"],
        ascending=[False, True, False],
    )


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def table_md(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df is None or df.empty:
        return "Nenhuma linha."
    d = df.head(max_rows).copy()
    try:
        return d.to_markdown(index=False)
    except Exception:
        return d.to_string(index=False)


def main() -> None:
    args = parse_args()
    default_pred, default_artifact, default_out = default_paths()

    pred_path = Path(args.predictions) if args.predictions else default_pred
    artifact_path = Path(args.artifact) if args.artifact else default_artifact
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions não encontrado: {pred_path}")

    artifact = None
    if artifact_path and Path(artifact_path).exists():
        artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))

    df = pd.read_csv(pred_path, low_memory=False).copy()

    label_col = find_col(df, LABELS)
    action_col = find_col(df, ACTION_COLS)
    intervention_col = find_col(df, INTERVENTION_COLS, required=False)
    block_col = find_col(df, BLOCK_COLS, required=False)

    y = ints(df[label_col]).to_numpy()
    base_action = actions(df[action_col])

    if intervention_col:
        base_intervention = ints(df[intervention_col])
    else:
        base_intervention = intervention_from_action(base_action)

    if block_col:
        base_block = ints(df[block_col])
    else:
        base_block = block_from_action(base_action)

    base_intervention_metrics = metrics(pd.Series(y), base_intervention)
    base_block_metrics = metrics(pd.Series(y), base_block)
    base_by_action = action_table(df.assign(_base_action=base_action), label_col, "_base_action")

    cat_cols = [
        c for c in CAT_COLS
        if c in df.columns and 1 < df[c].fillna("<MISSING>").astype(str).nunique(dropna=False) <= 100
    ]
    score_cols = [c for c in SCORE_COLS if c in df.columns]

    # Phase 1: BLOQUEAR -> CONFIRMAR. Good = normal (0), bad = fraud (1).
    block_idx = np.flatnonzero(base_action.eq("BLOQUEAR").to_numpy())

    block_candidates, block_masks = mine_candidates(
        df=df,
        eligible_idx=block_idx,
        y_all=y,
        prefix="Mover BLOQUEAR para CONFIRMAR R4D",
        phase_name="block_to_confirm",
        good_label=0,
        cat_cols=cat_cols,
        score_cols=score_cols,
        min_support=int(args.min_support),
        min_good=int(args.min_incremental_good),
        max_candidates=int(args.max_candidates),
        enable_quads=bool(args.enable_quads),
        combo_topn=int(args.combo_topn),
        score_cat_top_values=int(args.score_cat_top_values),
    )

    selected_block, frontier_block, local_block_move = greedy_select(
        candidates=block_candidates,
        masks=block_masks,
        y_local=y[block_idx],
        good_label=0,
        max_bad_total=int(args.max_block_tp_demoted),
        max_rules=int(args.max_rules_block),
        min_good=int(args.min_incremental_good),
        target_good=None,
    )

    # Phase 2: CONFIRMAR -> BLOQUEAR. Good = fraud (1), bad = normal (0).
    confirm_idx = np.flatnonzero(base_action.eq("CONFIRMAR").to_numpy())

    confirm_candidates, confirm_masks = mine_candidates(
        df=df,
        eligible_idx=confirm_idx,
        y_all=y,
        prefix="Mover CONFIRMAR para BLOQUEAR R4D",
        phase_name="confirm_to_block",
        good_label=1,
        cat_cols=cat_cols,
        score_cols=score_cols,
        min_support=int(args.min_support),
        min_good=int(args.min_incremental_good),
        max_candidates=int(args.max_candidates),
        enable_quads=bool(args.enable_quads),
        combo_topn=int(args.combo_topn),
        score_cat_top_values=int(args.score_cat_top_values),
    )

    selected_confirm, frontier_confirm, local_confirm_move = greedy_select(
        candidates=confirm_candidates,
        masks=confirm_masks,
        y_local=y[confirm_idx],
        good_label=1,
        max_bad_total=int(args.max_confirm_fp_promoted),
        max_rules=int(args.max_rules_confirm),
        min_good=int(args.min_incremental_good),
        target_good=int(args.target_confirm_tp_promoted),
    )

    block_to_confirm = np.zeros(len(df), dtype=bool)
    block_to_confirm[block_idx] = local_block_move

    confirm_to_block = np.zeros(len(df), dtype=bool)
    confirm_to_block[confirm_idx] = local_confirm_move

    final_action = base_action.copy()
    final_action.loc[block_to_confirm] = "CONFIRMAR"
    final_action.loc[confirm_to_block] = "BLOQUEAR"

    df["exp014b_r4d_block_to_confirm"] = block_to_confirm.astype(int)
    df["exp014b_r4d_confirm_to_block"] = confirm_to_block.astype(int)
    df["r4d_decisao_recommended"] = final_action
    df["exp014b_r4d_intervention_pred"] = intervention_from_action(final_action)
    df["exp014b_r4d_block_pred"] = block_from_action(final_action)

    final_intervention_metrics = metrics(pd.Series(y), df["exp014b_r4d_intervention_pred"])
    final_block_metrics = metrics(pd.Series(y), df["exp014b_r4d_block_pred"])
    final_by_action = action_table(df, label_col, "r4d_decisao_recommended")

    block_fp_demoted_to_confirm = int(((y == 0) & block_to_confirm).sum())
    block_tp_demoted_to_confirm = int(((y == 1) & block_to_confirm).sum())
    confirm_tp_promoted_to_block = int(((y == 1) & confirm_to_block).sum())
    confirm_fp_promoted_to_block = int(((y == 0) & confirm_to_block).sum())

    intervention_pred_unchanged = bool(
        (ints(base_intervention).to_numpy() == ints(df["exp014b_r4d_intervention_pred"]).to_numpy()).all()
    )
    approval_fraud_delta = int(final_intervention_metrics["fn"] - base_intervention_metrics["fn"])

    robustness = robustness_by_segment(df, label_col, action_col, "r4d_decisao_recommended")

    objective_status = (
        "DONE_R4D_SEVERITY_REBALANCED_INTERVENTION_UNCHANGED"
        if intervention_pred_unchanged and approval_fraud_delta == 0 and (
            block_fp_demoted_to_confirm > 0 or confirm_tp_promoted_to_block > 0
        )
        else "DONE_R4D_NO_SAFE_SEVERITY_REBALANCE"
    )

    summary = {
        "experiment": EXPERIMENT,
        "status": "DONE",
        "objective_status": objective_status,
        "n_rows": int(len(df)),
        "n_frauds": int((y == 1).sum()),
        "n_normals": int((y == 0).sum()),
        "predictions_path": str(pred_path),
        "artifact_path": str(artifact_path) if artifact_path else None,
        "label_col": label_col,
        "action_col": action_col,
        "intervention_col": intervention_col,
        "block_col": block_col,
        "baseline_intervention_metrics": base_intervention_metrics,
        "baseline_block_metrics": base_block_metrics,
        "final_intervention_metrics": final_intervention_metrics,
        "final_block_metrics": final_block_metrics,
        "intervention_pred_unchanged": intervention_pred_unchanged,
        "approval_fraud_delta": approval_fraud_delta,
        "block_fp_demoted_to_confirm": block_fp_demoted_to_confirm,
        "block_tp_demoted_to_confirm": block_tp_demoted_to_confirm,
        "confirm_tp_promoted_to_block": confirm_tp_promoted_to_block,
        "confirm_fp_promoted_to_block": confirm_fp_promoted_to_block,
        "net_block_tp_delta": int(final_block_metrics["tp"] - base_block_metrics["tp"]),
        "net_block_fp_delta": int(final_block_metrics["fp"] - base_block_metrics["fp"]),
        "n_block_candidates": int(len(block_candidates)),
        "n_confirm_candidates": int(len(confirm_candidates)),
        "n_selected_block_rules": int(len(selected_block)),
        "n_selected_confirm_rules": int(len(selected_confirm)),
        "max_block_tp_demoted": int(args.max_block_tp_demoted),
        "max_confirm_fp_promoted": int(args.max_confirm_fp_promoted),
        "target_confirm_tp_promoted": int(args.target_confirm_tp_promoted),
        "all_pass": bool(intervention_pred_unchanged and approval_fraud_delta == 0),
        "output_dir": str(out_dir),
    }

    contract = {
        "predictions_path": str(pred_path),
        "artifact_path": str(artifact_path) if artifact_path else None,
        "label_col": label_col,
        "action_col": action_col,
        "intervention_col": intervention_col,
        "block_col": block_col,
        "contract_ok": True,
        "missing": [],
    }

    base_metrics = {
        "baseline_intervention_metrics": base_intervention_metrics,
        "baseline_block_metrics": base_block_metrics,
        "baseline_by_action": base_by_action.to_dict(orient="records"),
        "artifact_status": artifact.get("frozen_validation_status") if isinstance(artifact, dict) else None,
    }

    policy = {
        "experiment": EXPERIMENT,
        "input_predictions_path": str(pred_path),
        "base_action_col": action_col,
        "final_action_col": "r4d_decisao_recommended",
        "block_to_confirm_col": "exp014b_r4d_block_to_confirm",
        "confirm_to_block_col": "exp014b_r4d_confirm_to_block",
        "intervention_pred_col": "exp014b_r4d_intervention_pred",
        "block_pred_col": "exp014b_r4d_block_pred",
        "baseline_intervention_metrics": base_intervention_metrics,
        "baseline_block_metrics": base_block_metrics,
        "final_intervention_metrics": final_intervention_metrics,
        "final_block_metrics": final_block_metrics,
        "block_fp_demoted_to_confirm": block_fp_demoted_to_confirm,
        "block_tp_demoted_to_confirm": block_tp_demoted_to_confirm,
        "confirm_tp_promoted_to_block": confirm_tp_promoted_to_block,
        "confirm_fp_promoted_to_block": confirm_fp_promoted_to_block,
        "selected_block_to_confirm_rules": selected_block.to_dict(orient="records") if not selected_block.empty else [],
        "selected_confirm_to_block_rules": selected_confirm.to_dict(orient="records") if not selected_confirm.empty else [],
        "notes": [
            "R4D changes only severity between CONFIRMAR and BLOQUEAR.",
            "Intervention total should remain unchanged.",
            "APROVAR should remain unchanged.",
            "Promotion requires frozen replay and business review.",
        ],
    }

    write_json(out_dir / "00_run_summary.json", summary)
    write_json(out_dir / "01_input_contract.json", contract)
    write_json(out_dir / "02_base_metrics.json", base_metrics)
    block_candidates.to_csv(out_dir / "03_block_to_confirm_candidates.csv", index=False, encoding="utf-8")
    confirm_candidates.to_csv(out_dir / "04_confirm_to_block_candidates.csv", index=False, encoding="utf-8")
    selected_block.to_csv(out_dir / "05_selected_block_to_confirm_rules.csv", index=False, encoding="utf-8")
    selected_confirm.to_csv(out_dir / "06_selected_confirm_to_block_rules.csv", index=False, encoding="utf-8")

    frontier = pd.concat(
        [
            frontier_block.assign(phase="block_to_confirm") if not frontier_block.empty else pd.DataFrame(),
            frontier_confirm.assign(phase="confirm_to_block") if not frontier_confirm.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    frontier.to_csv(out_dir / "07_selection_frontier.csv", index=False, encoding="utf-8")
    final_by_action.to_csv(out_dir / "08_decision_metrics_by_action.csv", index=False, encoding="utf-8")
    robustness.to_csv(out_dir / "09_robustness_by_segment.csv", index=False, encoding="utf-8")
    write_json(out_dir / "10_policy_artifact_recommended.json", policy)
    df.to_csv(out_dir / "11_predictions_recommended.csv", index=False, encoding="utf-8")

    report = f"""# {EXPERIMENT} - Severity Rebalance after R4C

## Resultado executivo
- Status: `{objective_status}`
- All pass: `{summary["all_pass"]}`
- Intervention unchanged: `{intervention_pred_unchanged}`
- Approval fraud delta: `{approval_fraud_delta}`

## Movimentos de severidade
- BLOQUEAR -> CONFIRMAR, normais movidos: `{block_fp_demoted_to_confirm}`
- BLOQUEAR -> CONFIRMAR, fraudes movidas: `{block_tp_demoted_to_confirm}`
- CONFIRMAR -> BLOQUEAR, fraudes promovidas: `{confirm_tp_promoted_to_block}`
- CONFIRMAR -> BLOQUEAR, normais promovidos: `{confirm_fp_promoted_to_block}`

## Baseline intervenção
```json
{json.dumps(base_intervention_metrics, ensure_ascii=False, indent=2)}
```

## Final intervenção
```json
{json.dumps(final_intervention_metrics, ensure_ascii=False, indent=2)}
```

## Baseline BLOQUEAR
```json
{json.dumps(base_block_metrics, ensure_ascii=False, indent=2)}
```

## Final BLOQUEAR
```json
{json.dumps(final_block_metrics, ensure_ascii=False, indent=2)}
```

## Métricas por ação final
{table_md(final_by_action)}

## Regras BLOQUEAR -> CONFIRMAR selecionadas
{table_md(selected_block)}

## Regras CONFIRMAR -> BLOQUEAR selecionadas
{table_md(selected_confirm)}

## Decisão sugerida
Se o resultado melhorar a severidade sem alterar intervenção total, promover para R4D-FROZEN.
"""
    (out_dir / "12_exp014b_r4d_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
