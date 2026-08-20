# -*- coding: utf-8 -*-
"""
EXP-014B-R4F — Block-to-Confirm Fine Tune after R4E

Objetivo:
  Partir do R4E-FROZEN e reduzir a severidade indevida:
    - mover normais prováveis de BLOQUEAR -> CONFIRMAR;
    - por padrão, não mover nenhuma fraude de BLOQUEAR;
    - não alterar APROVAR;
    - não alterar intervenção global, TP, FP global, FN ou FPR global.

Por que funciona:
  BLOQUEAR e CONFIRMAR são ambos intervenção. Então mover BLOQUEAR -> CONFIRMAR
  reduz severidade, mas mantém a transação dentro da intervenção.

Saídas:
  resultados/experimentos/EXP-014B-R4F/
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT = "EXP-014B-R4F"

LABELS = ["is_fraud", "fraude", "target", "label", "tp_fraude"]

ACTION_COLS = [
    "r4e_frozen_decisao_recommended",
    "r4e_decisao_recommended",
    "r4d_frozen_decisao_recommended",
    "r4d_decisao_recommended",
]

INTERVENTION_COLS = [
    "exp014b_r4e_frozen_intervention_pred",
    "exp014b_r4e_intervention_pred",
    "exp014b_r4d_frozen_intervention_pred",
    "exp014b_r4d_intervention_pred",
]

BLOCK_COLS = [
    "exp014b_r4e_frozen_block_pred",
    "exp014b_r4e_block_pred",
    "exp014b_r4d_frozen_block_pred",
    "exp014b_r4d_block_pred",
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
    p.add_argument("--target-block-fp-demoted", type=int, default=None)
    p.add_argument("--max-rules", type=int, default=160)
    p.add_argument("--max-candidates", type=int, default=16000)
    p.add_argument("--min-support", type=int, default=1)
    p.add_argument("--min-incremental-fp", type=int, default=1)
    p.add_argument("--enable-quads", action="store_true")
    p.add_argument("--enable-score-cat-pairs", action="store_true")
    p.add_argument("--combo-topn", type=int, default=800)
    p.add_argument("--score-cat-top-values", type=int, default=50)
    return p.parse_args()


def default_paths() -> tuple[Path, Path | None, Path]:
    root = Path.cwd()
    pred = root / "resultados" / "experimentos" / "EXP-014B-R4E-FROZEN" / "06_predictions_frozen.csv"
    if not pred.exists():
        pred = root / "resultados" / "experimentos" / "EXP-014B-R4E" / "13_predictions_recommended.csv"

    artifact = root / "resultados" / "experimentos" / "EXP-014B-R4E-FROZEN" / "05_policy_artifact_frozen.json"
    if not artifact.exists():
        artifact = root / "resultados" / "experimentos" / "EXP-014B-R4E" / "12_policy_artifact_recommended.json"

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
    min_support: int,
    min_fp: int,
) -> None:
    if description in seen:
        return

    n = int(local_mask.sum())
    if n < min_support:
        return

    yy = y_local[local_mask]
    fp_removed = int((yy == 0).sum())
    tp_demoted = int((yy == 1).sum())

    if fp_removed < min_fp:
        return

    seen.add(description)
    rows.append(
        {
            "candidate_id": candidate_id,
            "rule_type": rule_type,
            "description": description,
            "n_demoted": n,
            "block_fp_demoted": fp_removed,
            "block_tp_demoted": tp_demoted,
            "precision_demoted_is_normal": round(float(fp_removed / n), 8) if n else 0.0,
            "fp_per_tp_demoted": round(float(fp_removed / max(tp_demoted, 1)), 8),
        }
    )
    masks[candidate_id] = local_mask.copy()


def mine_candidates(
    df: pd.DataFrame,
    block_idx: np.ndarray,
    y_all: np.ndarray,
    cat_cols: list[str],
    score_cols: list[str],
    min_support: int,
    min_fp: int,
    max_candidates: int,
    enable_quads: bool,
    enable_score_cat_pairs: bool,
    combo_topn: int,
    score_cat_top_values: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    local = df.iloc[block_idx].copy()
    y_local = y_all[block_idx]
    rows: list[dict[str, Any]] = []
    masks: dict[str, np.ndarray] = {}
    seen: set[str] = set()

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
                    f"block_to_confirm_score__{score_col}__{op}{th:.12g}",
                    "score_threshold",
                    f"Mover BLOQUEAR para CONFIRMAR R4F com {score_col} {op} {th:.12g}",
                    mask,
                    y_local,
                    min_support,
                    min_fp,
                )

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
                    f"block_to_confirm_cat{size}__" + "__".join(safe_parts),
                    f"categorical_{size}",
                    "Mover BLOQUEAR para CONFIRMAR R4F com " + " AND ".join(parts),
                    mask,
                    y_local,
                    min_support,
                    min_fp,
                )

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
                            f"block_to_confirm_scorecat__{cat_col}={str(cat_val)[:20]}__{score_col}__{op}{th:.12g}",
                            "score_cat_1",
                            f"Mover BLOQUEAR para CONFIRMAR R4F com {cat_col} == {cat_val} AND {score_col} {op} {th:.12g}",
                            base & smask,
                            y_local,
                            min_support,
                            min_fp,
                        )

        if enable_score_cat_pairs:
            for cat1, cat2 in itertools.combinations(cat_cols[:12], 2):
                c1 = local[cat1].fillna("<MISSING>").astype(str)
                c2 = local[cat2].fillna("<MISSING>").astype(str)
                pair_df = pd.DataFrame({cat1: c1, cat2: c2})
                vc = pair_df.value_counts(dropna=False).head(max(120, score_cat_top_values * 3))
                c1v = c1.to_numpy()
                c2v = c2.to_numpy()
                for vals, support in vc.items():
                    if int(support) < min_support:
                        continue
                    v1, v2 = vals if isinstance(vals, tuple) else (vals, "")
                    base = (c1v == str(v1)) & (c2v == str(v2))
                    for th in thresholds:
                        for op, smask in [
                            ("<=", np.isfinite(values) & (values <= th)),
                            (">=", np.isfinite(values) & (values >= th)),
                        ]:
                            add_candidate(
                                rows,
                                masks,
                                seen,
                                f"block_to_confirm_scorecat2__{cat1}={str(v1)[:14]}__{cat2}={str(v2)[:14]}__{score_col}__{op}{th:.12g}",
                                "score_cat_2",
                                f"Mover BLOQUEAR para CONFIRMAR R4F com {cat1} == {v1} AND {cat2} == {v2} AND {score_col} {op} {th:.12g}",
                                base & smask,
                                y_local,
                                min_support,
                                min_fp,
                            )

    if not rows:
        return pd.DataFrame(), {}

    candidates = pd.DataFrame(rows).drop_duplicates(subset=["description"])
    candidates = candidates.sort_values(
        ["block_tp_demoted", "block_fp_demoted", "fp_per_tp_demoted", "n_demoted"],
        ascending=[True, False, False, False],
    ).head(max_candidates).reset_index(drop=True)

    keep = set(candidates["candidate_id"].astype(str))
    masks = {k: v for k, v in masks.items() if k in keep}

    return candidates, masks


def greedy_select(
    candidates: pd.DataFrame,
    masks: dict[str, np.ndarray],
    y_local: np.ndarray,
    max_tp_demoted: int,
    max_rules: int,
    min_fp: int,
    target_fp_demoted: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    selected_mask = np.zeros(len(y_local), dtype=bool)
    selected_rows = []
    frontier = []

    cum_fp = 0
    cum_tp = 0
    remaining = candidates.copy().reset_index(drop=True)

    for step in range(1, int(max_rules) + 1):
        if target_fp_demoted is not None and cum_fp >= target_fp_demoted:
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
            fp_gain = int((yy == 0).sum())
            tp_loss = int((yy == 1).sum())

            if fp_gain < min_fp:
                continue
            if cum_tp + tp_loss > max_tp_demoted:
                continue

            score = (
                1 if tp_loss == 0 else 0,
                fp_gain / max(tp_loss, 1),
                fp_gain,
                -tp_loss,
                -n,
            )

            if best is None or score > best_score:
                best = row.copy()
                best_mask = incremental
                best_score = score
                best["incremental_n"] = n
                best["incremental_block_fp_demoted"] = fp_gain
                best["incremental_block_tp_demoted"] = tp_loss

        if best is None or best_mask is None:
            break

        selected_mask |= best_mask
        cum_fp += int(best["incremental_block_fp_demoted"])
        cum_tp += int(best["incremental_block_tp_demoted"])

        best["selection_step"] = step
        best["cumulative_n"] = int(selected_mask.sum())
        best["cumulative_block_fp_demoted"] = int(cum_fp)
        best["cumulative_block_tp_demoted"] = int(cum_tp)
        selected_rows.append(best)

        frontier.append(
            {
                "selection_step": step,
                "selected_candidate_id": str(best["candidate_id"]),
                "selected_description": str(best["description"]),
                "incremental_n": int(best["incremental_n"]),
                "incremental_block_fp_demoted": int(best["incremental_block_fp_demoted"]),
                "incremental_block_tp_demoted": int(best["incremental_block_tp_demoted"]),
                "cumulative_n": int(selected_mask.sum()),
                "cumulative_block_fp_demoted": int(cum_fp),
                "cumulative_block_tp_demoted": int(cum_tp),
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
    before_inter = intervention_from_action(df[before_action_col])
    after_inter = intervention_from_action(df[after_action_col])

    rows = []
    for col in ROBUSTNESS_COLS:
        if col not in df.columns:
            continue

        for val, idx in df.groupby(col, dropna=False).groups.items():
            idx = list(idx)
            yy = y.loc[idx]
            before_b = metrics(yy, before_block.loc[idx])
            after_b = metrics(yy, after_block.loc[idx])
            before_i = metrics(yy, before_inter.loc[idx])
            after_i = metrics(yy, after_inter.loc[idx])
            rows.append(
                {
                    "segment_col": col,
                    "segment_value": str(val),
                    "n_rows": int(len(idx)),
                    "block_fp_delta": int(after_b["fp"] - before_b["fp"]),
                    "block_tp_delta": int(after_b["tp"] - before_b["tp"]),
                    "intervention_fp_delta": int(after_i["fp"] - before_i["fp"]),
                    "intervention_tp_delta": int(after_i["tp"] - before_i["tp"]),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["block_fp_delta", "block_tp_delta", "n_rows"],
        ascending=[True, False, False],
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

    block_idx = np.flatnonzero(base_action.eq("BLOQUEAR").to_numpy())

    candidates, masks = mine_candidates(
        df=df,
        block_idx=block_idx,
        y_all=y,
        cat_cols=cat_cols,
        score_cols=score_cols,
        min_support=int(args.min_support),
        min_fp=int(args.min_incremental_fp),
        max_candidates=int(args.max_candidates),
        enable_quads=bool(args.enable_quads),
        enable_score_cat_pairs=bool(args.enable_score_cat_pairs),
        combo_topn=int(args.combo_topn),
        score_cat_top_values=int(args.score_cat_top_values),
    )

    selected, frontier, local_move = greedy_select(
        candidates=candidates,
        masks=masks,
        y_local=y[block_idx],
        max_tp_demoted=int(args.max_block_tp_demoted),
        max_rules=int(args.max_rules),
        min_fp=int(args.min_incremental_fp),
        target_fp_demoted=args.target_block_fp_demoted,
    )

    block_to_confirm = np.zeros(len(df), dtype=bool)
    block_to_confirm[block_idx] = local_move

    final_action = base_action.copy()
    final_action.loc[block_to_confirm] = "CONFIRMAR"

    df["exp014b_r4f_block_to_confirm"] = block_to_confirm.astype(int)
    df["r4f_decisao_recommended"] = final_action
    df["exp014b_r4f_intervention_pred"] = intervention_from_action(final_action)
    df["exp014b_r4f_block_pred"] = block_from_action(final_action)

    final_intervention_metrics = metrics(pd.Series(y), df["exp014b_r4f_intervention_pred"])
    final_block_metrics = metrics(pd.Series(y), df["exp014b_r4f_block_pred"])
    final_by_action = action_table(df, label_col, "r4f_decisao_recommended")

    block_fp_demoted = int(((y == 0) & block_to_confirm).sum())
    block_tp_demoted = int(((y == 1) & block_to_confirm).sum())
    intervention_unchanged = bool(
        (ints(base_intervention).to_numpy() == ints(df["exp014b_r4f_intervention_pred"]).to_numpy()).all()
    )
    approval_fraud_delta = int(final_intervention_metrics["fn"] - base_intervention_metrics["fn"])

    robustness = robustness_by_segment(df, label_col, action_col, "r4f_decisao_recommended")

    objective_status = (
        "DONE_R4F_BLOCK_NORMALS_DEMOTED_WITH_GLOBAL_METRICS_UNCHANGED"
        if block_fp_demoted > 0 and intervention_unchanged and approval_fraud_delta == 0 and block_tp_demoted <= int(args.max_block_tp_demoted)
        else "DONE_R4F_NO_SAFE_BLOCK_TO_CONFIRM_GAIN"
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
        "intervention_unchanged": intervention_unchanged,
        "approval_fraud_delta": approval_fraud_delta,
        "block_fp_demoted_to_confirm": block_fp_demoted,
        "block_tp_demoted_to_confirm": block_tp_demoted,
        "net_block_fp_delta": int(final_block_metrics["fp"] - base_block_metrics["fp"]),
        "net_block_tp_delta": int(final_block_metrics["tp"] - base_block_metrics["tp"]),
        "n_candidates": int(len(candidates)),
        "n_selected_rules": int(len(selected)),
        "max_block_tp_demoted": int(args.max_block_tp_demoted),
        "all_pass": bool(intervention_unchanged and approval_fraud_delta == 0 and block_tp_demoted <= int(args.max_block_tp_demoted)),
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
        "final_action_col": "r4f_decisao_recommended",
        "block_to_confirm_col": "exp014b_r4f_block_to_confirm",
        "intervention_pred_col": "exp014b_r4f_intervention_pred",
        "block_pred_col": "exp014b_r4f_block_pred",
        "baseline_intervention_metrics": base_intervention_metrics,
        "baseline_block_metrics": base_block_metrics,
        "final_intervention_metrics": final_intervention_metrics,
        "final_block_metrics": final_block_metrics,
        "intervention_unchanged": intervention_unchanged,
        "approval_fraud_delta": approval_fraud_delta,
        "block_fp_demoted_to_confirm": block_fp_demoted,
        "block_tp_demoted_to_confirm": block_tp_demoted,
        "selected_block_to_confirm_rules": selected.to_dict(orient="records") if not selected.empty else [],
        "notes": [
            "R4F only moves BLOQUEAR to CONFIRMAR.",
            "Global intervention should remain unchanged.",
            "Default policy allows zero fraud demotion from BLOQUEAR.",
            "Promotion requires frozen replay and business review.",
        ],
    }

    write_json(out_dir / "00_run_summary.json", summary)
    write_json(out_dir / "01_input_contract.json", contract)
    write_json(out_dir / "02_base_metrics.json", base_metrics)
    candidates.to_csv(out_dir / "03_block_to_confirm_candidates.csv", index=False, encoding="utf-8")
    selected.to_csv(out_dir / "04_selected_block_to_confirm_rules.csv", index=False, encoding="utf-8")
    frontier.to_csv(out_dir / "05_selection_frontier.csv", index=False, encoding="utf-8")
    final_by_action.to_csv(out_dir / "06_decision_metrics_by_action.csv", index=False, encoding="utf-8")
    robustness.to_csv(out_dir / "07_robustness_by_segment.csv", index=False, encoding="utf-8")
    write_json(out_dir / "08_policy_artifact_recommended.json", policy)
    df.to_csv(out_dir / "09_predictions_recommended.csv", index=False, encoding="utf-8")

    report = f"""# {EXPERIMENT} - Block-to-Confirm Fine Tune after R4E

## Resultado executivo
- Status: `{objective_status}`
- All pass: `{summary["all_pass"]}`
- Intervention unchanged: `{intervention_unchanged}`
- Approval fraud delta: `{approval_fraud_delta}`
- BLOQUEAR -> CONFIRMAR, normais movidos: `{block_fp_demoted}`
- BLOQUEAR -> CONFIRMAR, fraudes movidas: `{block_tp_demoted}`

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

## Regras selecionadas
{table_md(selected)}

## Decisão sugerida
Se houver redução relevante de FP em BLOQUEAR com `block_tp_demoted_to_confirm=0`, promover para R4F-FROZEN.
Se o ganho for pequeno, manter R4E-FROZEN como baseline final e considerar alternativas de feature engineering.
"""
    (out_dir / "10_exp014b_r4f_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
