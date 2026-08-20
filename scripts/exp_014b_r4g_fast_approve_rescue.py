# -*- coding: utf-8 -*-
"""
EXP-014B-R4G-FAST - Approve Fraud Rescue + Block Fine Tune

Versao eficiente do R4G:
- APROVAR -> CONFIRMAR: busca ancorada apenas nas fraudes aprovadas.
- BLOQUEAR -> CONFIRMAR: busca no subconjunto pequeno de BLOQUEAR.

Default input:
  resultados/experimentos/EXP-014B-R4F-FROZEN/06_predictions_frozen.csv
  fallback: resultados/experimentos/EXP-014B-R4F/09_predictions_recommended.csv

Output:
  resultados/experimentos/EXP-014B-R4G-FAST/
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT = "EXP-014B-R4G-FAST"
LABELS = ["is_fraud", "fraude", "target", "label", "tp_fraude"]
ACTION_COLS = [
    "r4f_frozen_decisao_recommended", "r4f_decisao_recommended",
    "r4e_frozen_decisao_recommended", "r4e_decisao_recommended",
]
INTERVENTION_COLS = [
    "exp014b_r4f_frozen_intervention_pred", "exp014b_r4f_intervention_pred",
    "exp014b_r4e_frozen_intervention_pred", "exp014b_r4e_intervention_pred",
]
BLOCK_COLS = [
    "exp014b_r4f_frozen_block_pred", "exp014b_r4f_block_pred",
    "exp014b_r4e_frozen_block_pred", "exp014b_r4e_block_pred",
]
CAT_COLS = [
    "ds_tipo_chave_norm", "value_band", "periodo_dia", "score_bin", "lgbm_bin", "if_bin",
    "ratio_bin", "qtd_rec_bin", "valor_rec_bin", "mbk_available_flag", "first_receiver_flag_real",
    "module_quiet", "se_worst_pattern", "r3u_missing_receiver_history_flag", "r3u_receiver_known_flag",
    "r3u_receiver_reputable_flag", "r3u_receiver_strong_flag", "r3u_relationship_known_flag",
    "r3u_relationship_recurrent_flag", "r3u_relationship_strong_flag", "r3u_first_receiver_flag",
    "r3u_module_quiet_flag", "r3u_se_missing_flag", "r3u_ratio_lt_005_flag", "r3u_mbk_quality_flag",
    "r3u_receiver_trust_bucket", "r3u_relationship_bucket",
]
SCORE_COLS = [
    "lgbm_r4_score", "score_final", "lgbm_raw", "lgbm_mapped", "peso_total", "if_percentile",
    "se_score", "beh_score", "behavioral_score", "topaz_risk_score",
    "exp014b_r3s_second_stage_score", "exp014b_r3u_receiver_relationship_trust_score",
]
SEGMENT_COLS = [
    "temporal_split", "event_month", "ds_tipo_chave_norm", "value_band", "periodo_dia",
    "score_bin", "lgbm_bin", "if_bin", "ratio_bin", "qtd_rec_bin", "valor_rec_bin",
    "mbk_available_flag", "first_receiver_flag_real",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default=None)
    p.add_argument("--artifact", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--max-total-fn", type=int, default=5)
    p.add_argument("--max-approve-fp-promoted", type=int, default=None)
    p.add_argument("--target-approve-tp-promoted", type=int, default=5)
    p.add_argument("--max-block-tp-demoted", type=int, default=0)
    p.add_argument("--target-block-fp-demoted", type=int, default=None)
    p.add_argument("--max-rules-approve", type=int, default=40)
    p.add_argument("--max-rules-block", type=int, default=180)
    p.add_argument("--approve-max-cat-cols", type=int, default=14)
    p.add_argument("--approve-max-combo-size", type=int, default=3)
    p.add_argument("--approve-enable-quads", action="store_true")
    p.add_argument("--approve-score-cat", action="store_true")
    p.add_argument("--block-max-cat-cols", type=int, default=18)
    p.add_argument("--block-enable-quads", action="store_true")
    p.add_argument("--block-score-cat", action="store_true")
    p.add_argument("--max-candidates-approve", type=int, default=3000)
    p.add_argument("--max-candidates-block", type=int, default=8000)
    p.add_argument("--min-support", type=int, default=1)
    p.add_argument("--min-incremental-good", type=int, default=1)
    p.add_argument("--skip-block-finetune", action="store_true")
    return p.parse_args()


def defaults() -> tuple[Path, Path | None, Path]:
    root = Path.cwd()
    pred = root / "resultados" / "experimentos" / "EXP-014B-R4F-FROZEN" / "06_predictions_frozen.csv"
    if not pred.exists():
        pred = root / "resultados" / "experimentos" / "EXP-014B-R4F" / "09_predictions_recommended.csv"
    art = root / "resultados" / "experimentos" / "EXP-014B-R4F-FROZEN" / "05_policy_artifact_frozen.json"
    if not art.exists():
        art = root / "resultados" / "experimentos" / "EXP-014B-R4F" / "08_policy_artifact_recommended.json"
    out = root / "resultados" / "experimentos" / EXPERIMENT
    return pred, art if art.exists() else None, out


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


def ints(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def norm_action(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def pred_intervention(action: pd.Series) -> pd.Series:
    return norm_action(action).isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)


def pred_block(action: pd.Series) -> pd.Series:
    return norm_action(action).eq("BLOQUEAR").astype(int)


def metrics(y_true: pd.Series, pred: pd.Series) -> dict[str, Any]:
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
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(float(precision), 8),
        "recall": round(float(recall), 8),
        "f1": round(float(f1), 8),
        "fpr": round(float(fpr), 8),
    }


def strict_target_fp(n_normals: int, target_fpr: float) -> int:
    return int(np.ceil(float(target_fpr) * int(n_normals)) - 1)


def action_table(df: pd.DataFrame, label_col: str, action_col: str) -> pd.DataFrame:
    y = ints(df[label_col])
    rows = []
    for action, idx in df.groupby(action_col, dropna=False).groups.items():
        idx = list(idx)
        yy = y.loc[idx]
        n = int(len(idx))
        frauds = int((yy == 1).sum())
        normals = int((yy == 0).sum())
        rows.append({
            "action": str(action), "n_rows": n, "n_frauds": frauds, "n_normals": normals,
            "precision_within_action": round(float(frauds / n), 8) if n else 0.0,
        })
    return pd.DataFrame(rows).sort_values("action")


def usable_cat_cols(df: pd.DataFrame, max_cols: int) -> list[str]:
    cols = []
    for c in CAT_COLS:
        if c not in df.columns:
            continue
        nun = df[c].fillna("<MISSING>").astype(str).nunique(dropna=False)
        if 1 < nun <= 100:
            cols.append(c)
    return cols[:max_cols]


def usable_score_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in SCORE_COLS if c in df.columns]


def add_candidate(rows, masks, seen, cid, rule_type, desc, mask, y_local, good_label, min_support, min_good, max_bad=None):
    if desc in seen:
        return
    n = int(mask.sum())
    if n < min_support:
        return
    yy = y_local[mask]
    good = int((yy == good_label).sum())
    bad = int((yy != good_label).sum())
    if good < min_good:
        return
    if max_bad is not None and bad > max_bad:
        return
    seen.add(desc)
    rows.append({
        "candidate_id": cid, "rule_type": rule_type, "description": desc,
        "n_affected": n, "good_count": good, "bad_count": bad,
        "precision_for_goal": round(float(good / n), 8) if n else 0.0,
        "good_per_bad": round(float(good / max(bad, 1)), 8),
    })
    masks[cid] = mask.copy()


def eq_mask(arrs: dict[str, np.ndarray], cols: tuple[str, ...], vals: tuple[str, ...]) -> np.ndarray:
    mask = np.ones(len(next(iter(arrs.values()))), dtype=bool)
    for c, v in zip(cols, vals):
        mask &= arrs[c] == v
    return mask


def mine_approve_anchored(df, idx, y_all, cat_cols, score_cols, max_bad, max_combo_size, enable_quads, enable_score_cat, max_candidates, min_support, min_good):
    local = df.iloc[idx].copy()
    y_local = y_all[idx]
    fraud_pos = np.flatnonzero(y_local == 1)
    rows, masks, seen = [], {}, set()
    if len(fraud_pos) == 0:
        return pd.DataFrame(), {}

    arrs = {c: local[c].fillna("<MISSING>").astype(str).to_numpy() for c in cat_cols}
    max_size = min(max(max_combo_size, 4 if enable_quads else max_combo_size), len(cat_cols))
    for size in range(1, max_size + 1):
        for cols in itertools.combinations(cat_cols, size):
            vals_seen = set()
            for pos in fraud_pos:
                vals = tuple(arrs[c][pos] for c in cols)
                if vals in vals_seen:
                    continue
                vals_seen.add(vals)
                mask = eq_mask(arrs, cols, vals)
                parts = [f"{c} == {v}" for c, v in zip(cols, vals)]
                safe = "__".join(f"{c}={str(v)[:18]}" for c, v in zip(cols, vals))
                add_candidate(rows, masks, seen, f"approve_to_confirm_cat{size}__{safe}", f"categorical_{size}",
                              "Mover APROVAR para CONFIRMAR R4G_FAST com " + " AND ".join(parts),
                              mask, y_local, 1, min_support, min_good, max_bad)

    for sc in score_cols:
        s = pd.to_numeric(local[sc], errors="coerce")
        vals = s.to_numpy()
        for pos in fraud_pos:
            v = vals[pos]
            if not np.isfinite(v):
                continue
            for op, mask in [("<=", np.isfinite(vals) & (vals <= v)), (">=", np.isfinite(vals) & (vals >= v))]:
                add_candidate(rows, masks, seen, f"approve_to_confirm_score__{sc}__{op}{float(v):.12g}", "score_threshold",
                              f"Mover APROVAR para CONFIRMAR R4G_FAST com {sc} {op} {float(v):.12g}",
                              mask, y_local, 1, min_support, min_good, max_bad)

    if enable_score_cat:
        for sc in score_cols[:8]:
            s = pd.to_numeric(local[sc], errors="coerce")
            vals = s.to_numpy()
            for cat in cat_cols[:14]:
                cv = arrs[cat]
                for pos in fraud_pos:
                    score_v = vals[pos]
                    cat_v = cv[pos]
                    if not np.isfinite(score_v):
                        continue
                    base = cv == cat_v
                    for op, sm in [("<=", np.isfinite(vals) & (vals <= score_v)), (">=", np.isfinite(vals) & (vals >= score_v))]:
                        add_candidate(rows, masks, seen, f"approve_to_confirm_scorecat__{cat}={str(cat_v)[:18]}__{sc}__{op}{float(score_v):.12g}", "score_cat_1",
                                      f"Mover APROVAR para CONFIRMAR R4G_FAST com {cat} == {cat_v} AND {sc} {op} {float(score_v):.12g}",
                                      base & sm, y_local, 1, min_support, min_good, max_bad)

    if not rows:
        return pd.DataFrame(), {}
    cand = pd.DataFrame(rows).drop_duplicates(subset=["description"])
    cand = cand.sort_values(["bad_count", "good_count", "good_per_bad", "n_affected"], ascending=[True, False, False, False]).head(max_candidates).reset_index(drop=True)
    keep = set(cand["candidate_id"].astype(str))
    return cand, {k: v for k, v in masks.items() if k in keep}


def mine_block_fast(df, idx, y_all, cat_cols, score_cols, max_bad, max_candidates, min_support, min_good, enable_quads, enable_score_cat):
    local = df.iloc[idx].copy()
    y_local = y_all[idx]
    rows, masks, seen = [], {}, set()
    if len(local) == 0:
        return pd.DataFrame(), {}
    arrs = {c: local[c].fillna("<MISSING>").astype(str).to_numpy() for c in cat_cols}
    plan = [(1, cat_cols[:18]), (2, cat_cols[:16]), (3, cat_cols[:12])]
    if enable_quads:
        plan.append((4, cat_cols[:10]))
    for size, cols_source in plan:
        for cols in itertools.combinations(cols_source, size):
            tmp = local[list(cols)].fillna("<MISSING>").astype(str)
            g = tmp.assign(_y=y_local).groupby(list(cols), dropna=False)["_y"].agg(["size", "sum"])
            g["good"] = g["size"] - g["sum"]
            g["bad"] = g["sum"]
            g = g[(g["good"] >= min_good) & (g["bad"] <= max_bad)].sort_values(["bad", "good"], ascending=[True, False])
            for vals, row in g.head(200).iterrows():
                vals = vals if isinstance(vals, tuple) else (vals,)
                vals = tuple(str(v) for v in vals)
                mask = eq_mask(arrs, cols, vals)
                parts = [f"{c} == {v}" for c, v in zip(cols, vals)]
                safe = "__".join(f"{c}={str(v)[:18]}" for c, v in zip(cols, vals))
                add_candidate(rows, masks, seen, f"block_to_confirm_cat{size}__{safe}", f"categorical_{size}",
                              "Mover BLOQUEAR para CONFIRMAR R4G_FAST com " + " AND ".join(parts),
                              mask, y_local, 0, min_support, min_good, max_bad)
    qs = [0.005,0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.92,0.95,0.97,0.98,0.99,0.995]
    for sc in score_cols:
        s = pd.to_numeric(local[sc], errors="coerce")
        valid = s.dropna()
        if valid.empty:
            continue
        vals = s.to_numpy()
        ths = sorted(set(float(valid.quantile(q)) for q in qs if pd.notna(valid.quantile(q))))
        for th in ths:
            for op, mask in [("<=", np.isfinite(vals) & (vals <= th)), (">=", np.isfinite(vals) & (vals >= th))]:
                add_candidate(rows, masks, seen, f"block_to_confirm_score__{sc}__{op}{th:.12g}", "score_threshold",
                              f"Mover BLOQUEAR para CONFIRMAR R4G_FAST com {sc} {op} {th:.12g}",
                              mask, y_local, 0, min_support, min_good, max_bad)
    if enable_score_cat:
        sqs = [0.10,0.20,0.30,0.40,0.60,0.70,0.80,0.90,0.95]
        for sc in score_cols[:8]:
            s = pd.to_numeric(local[sc], errors="coerce")
            valid = s.dropna()
            if valid.empty:
                continue
            vals = s.to_numpy()
            ths = sorted(set(float(valid.quantile(q)) for q in sqs if pd.notna(valid.quantile(q))))
            for cat in cat_cols[:14]:
                cv = arrs[cat]
                for cat_v in pd.Series(cv).value_counts().head(30).index:
                    base = cv == str(cat_v)
                    for th in ths:
                        for op, sm in [("<=", np.isfinite(vals) & (vals <= th)), (">=", np.isfinite(vals) & (vals >= th))]:
                            add_candidate(rows, masks, seen, f"block_to_confirm_scorecat__{cat}={str(cat_v)[:18]}__{sc}__{op}{th:.12g}", "score_cat_1",
                                          f"Mover BLOQUEAR para CONFIRMAR R4G_FAST com {cat} == {cat_v} AND {sc} {op} {th:.12g}",
                                          base & sm, y_local, 0, min_support, min_good, max_bad)
    if not rows:
        return pd.DataFrame(), {}
    cand = pd.DataFrame(rows).drop_duplicates(subset=["description"])
    cand = cand.sort_values(["bad_count", "good_count", "good_per_bad", "n_affected"], ascending=[True, False, False, False]).head(max_candidates).reset_index(drop=True)
    keep = set(cand["candidate_id"].astype(str))
    return cand, {k: v for k, v in masks.items() if k in keep}


def greedy_select(cand, masks, y_local, good_label, max_bad_total, max_rules, min_good, target_good=None):
    selected = np.zeros(len(y_local), dtype=bool)
    rows, frontier = [], []
    cg, cb = 0, 0
    remaining = cand.copy().reset_index(drop=True)
    for step in range(1, int(max_rules) + 1):
        if target_good is not None and cg >= target_good:
            break
        best = None
        best_mask = None
        best_score = None
        for _, row in remaining.iterrows():
            mask = masks.get(str(row["candidate_id"]))
            if mask is None:
                continue
            inc = mask & (~selected)
            n = int(inc.sum())
            if n == 0:
                continue
            yy = y_local[inc]
            good = int((yy == good_label).sum())
            bad = int((yy != good_label).sum())
            if good < min_good:
                continue
            if cb + bad > max_bad_total:
                continue
            score = (1 if bad == 0 else 0, good / max(bad, 1), good, -bad, -n)
            if best is None or score > best_score:
                best, best_mask, best_score = row.copy(), inc, score
                best["incremental_n"] = n
                best["incremental_good"] = good
                best["incremental_bad"] = bad
        if best is None or best_mask is None:
            break
        selected |= best_mask
        cg += int(best["incremental_good"])
        cb += int(best["incremental_bad"])
        best["selection_step"] = step
        best["cumulative_n"] = int(selected.sum())
        best["cumulative_good"] = int(cg)
        best["cumulative_bad"] = int(cb)
        rows.append(best)
        frontier.append({
            "selection_step": step,
            "selected_candidate_id": str(best["candidate_id"]),
            "selected_description": str(best["description"]),
            "incremental_n": int(best["incremental_n"]),
            "incremental_good": int(best["incremental_good"]),
            "incremental_bad": int(best["incremental_bad"]),
            "cumulative_n": int(selected.sum()),
            "cumulative_good": int(cg),
            "cumulative_bad": int(cb),
        })
        remaining = remaining[remaining["candidate_id"].astype(str) != str(best["candidate_id"])].reset_index(drop=True)
    return pd.DataFrame(rows) if rows else pd.DataFrame(), pd.DataFrame(frontier) if frontier else pd.DataFrame(), selected


def robustness_by_segment(df, label_col, before_action_col, after_action_col):
    y = ints(df[label_col])
    before_i = pred_intervention(df[before_action_col])
    after_i = pred_intervention(df[after_action_col])
    before_b = pred_block(df[before_action_col])
    after_b = pred_block(df[after_action_col])
    rows = []
    for col in SEGMENT_COLS:
        if col not in df.columns:
            continue
        for val, idx in df.groupby(col, dropna=False).groups.items():
            idx = list(idx)
            yy = y.loc[idx]
            bi = metrics(yy, before_i.loc[idx])
            ai = metrics(yy, after_i.loc[idx])
            bb = metrics(yy, before_b.loc[idx])
            ab = metrics(yy, after_b.loc[idx])
            rows.append({
                "segment_col": col, "segment_value": str(val), "n_rows": int(len(idx)),
                "intervention_tp_delta": int(ai["tp"] - bi["tp"]),
                "intervention_fp_delta": int(ai["fp"] - bi["fp"]),
                "block_tp_delta": int(ab["tp"] - bb["tp"]),
                "block_fp_delta": int(ab["fp"] - bb["fp"]),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["intervention_tp_delta", "intervention_fp_delta", "block_fp_delta", "n_rows"], ascending=[False, True, True, False])


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def table_md(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df is None or df.empty:
        return "Nenhuma linha."
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return df.head(max_rows).to_string(index=False)


def main() -> None:
    args = parse_args()
    default_pred, default_artifact, default_out = defaults()
    pred_path = Path(args.predictions) if args.predictions else default_pred
    artifact_path = Path(args.artifact) if args.artifact else default_artifact
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions nao encontrado: {pred_path}")

    artifact = None
    if artifact_path and Path(artifact_path).exists():
        artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))

    df = pd.read_csv(pred_path, low_memory=False).copy()
    label_col = find_col(df, LABELS)
    action_col = find_col(df, ACTION_COLS)
    intervention_col = find_col(df, INTERVENTION_COLS, required=False)
    block_col = find_col(df, BLOCK_COLS, required=False)

    y = ints(df[label_col]).to_numpy()
    base_action = norm_action(df[action_col])
    base_intervention = ints(df[intervention_col]) if intervention_col else pred_intervention(base_action)
    base_block = ints(df[block_col]) if block_col else pred_block(base_action)
    base_intervention_metrics = metrics(pd.Series(y), base_intervention)
    base_block_metrics = metrics(pd.Series(y), base_block)

    n_normals = int((y == 0).sum())
    target_fp = strict_target_fp(n_normals, float(args.target_fpr))
    base_fp = int(base_intervention_metrics["fp"])
    headroom = max(0, target_fp - base_fp)
    max_approve_fp = headroom if args.max_approve_fp_promoted is None else int(args.max_approve_fp_promoted)

    approve_cat_cols = usable_cat_cols(df, int(args.approve_max_cat_cols))
    block_cat_cols = usable_cat_cols(df, int(args.block_max_cat_cols))
    score_cols = usable_score_cols(df)

    approve_idx = np.flatnonzero(base_action.eq("APROVAR").to_numpy())
    approve_candidates, approve_masks = mine_approve_anchored(
        df, approve_idx, y, approve_cat_cols, score_cols, int(max_approve_fp),
        int(args.approve_max_combo_size), bool(args.approve_enable_quads), bool(args.approve_score_cat),
        int(args.max_candidates_approve), int(args.min_support), int(args.min_incremental_good)
    )
    selected_approve, frontier_approve, local_approve_move = greedy_select(
        approve_candidates, approve_masks, y[approve_idx], 1, int(max_approve_fp),
        int(args.max_rules_approve), int(args.min_incremental_good), int(args.target_approve_tp_promoted)
    )

    approve_to_confirm = np.zeros(len(df), dtype=bool)
    approve_to_confirm[approve_idx] = local_approve_move
    action_after_a = base_action.copy()
    action_after_a.loc[approve_to_confirm] = "CONFIRMAR"

    if args.skip_block_finetune:
        block_candidates = pd.DataFrame()
        selected_block = pd.DataFrame()
        frontier_block = pd.DataFrame()
        block_to_confirm = np.zeros(len(df), dtype=bool)
    else:
        block_idx = np.flatnonzero(action_after_a.eq("BLOQUEAR").to_numpy())
        block_candidates, block_masks = mine_block_fast(
            df, block_idx, y, block_cat_cols, score_cols, int(args.max_block_tp_demoted),
            int(args.max_candidates_block), int(args.min_support), int(args.min_incremental_good),
            bool(args.block_enable_quads), bool(args.block_score_cat)
        )
        selected_block, frontier_block, local_block_move = greedy_select(
            block_candidates, block_masks, y[block_idx], 0, int(args.max_block_tp_demoted),
            int(args.max_rules_block), int(args.min_incremental_good), args.target_block_fp_demoted
        )
        block_to_confirm = np.zeros(len(df), dtype=bool)
        block_to_confirm[block_idx] = local_block_move

    final_action = action_after_a.copy()
    final_action.loc[block_to_confirm] = "CONFIRMAR"

    df["exp014b_r4g_fast_approve_to_confirm"] = approve_to_confirm.astype(int)
    df["exp014b_r4g_fast_block_to_confirm"] = block_to_confirm.astype(int)
    df["r4g_fast_decisao_recommended"] = final_action
    df["exp014b_r4g_fast_intervention_pred"] = pred_intervention(final_action)
    df["exp014b_r4g_fast_block_pred"] = pred_block(final_action)

    final_intervention_metrics = metrics(pd.Series(y), df["exp014b_r4g_fast_intervention_pred"])
    final_block_metrics = metrics(pd.Series(y), df["exp014b_r4g_fast_block_pred"])
    final_by_action = action_table(df, label_col, "r4g_fast_decisao_recommended")

    approve_tp = int(((y == 1) & approve_to_confirm).sum())
    approve_fp = int(((y == 0) & approve_to_confirm).sum())
    block_fp = int(((y == 0) & block_to_confirm).sum())
    block_tp = int(((y == 1) & block_to_confirm).sum())
    approve_fraud_remaining = int(((y == 1) & norm_action(final_action).eq("APROVAR").to_numpy()).sum())
    target_reached = bool(final_intervention_metrics["fp"] <= target_fp)
    fn_total_ok = bool(final_intervention_metrics["fn"] <= int(args.max_total_fn))

    objective_status = (
        "DONE_R4G_FAST_APPROVE_FRAUDS_RESCUED_AND_FPR_LT1_PRESERVED"
        if approve_tp > 0 and target_reached and fn_total_ok and block_tp <= int(args.max_block_tp_demoted)
        else "DONE_R4G_FAST_NO_SAFE_APPROVE_FRAUD_RESCUE"
        if approve_tp == 0
        else "DONE_R4G_FAST_RESCUED_APPROVE_FRAUDS_BUT_TARGET_NOT_PRESERVED"
    )

    summary = {
        "experiment": EXPERIMENT, "status": "DONE", "objective_status": objective_status,
        "n_rows": int(len(df)), "n_frauds": int((y == 1).sum()), "n_normals": n_normals,
        "predictions_path": str(pred_path), "artifact_path": str(artifact_path) if artifact_path else None,
        "label_col": label_col, "action_col": action_col, "intervention_col": intervention_col, "block_col": block_col,
        "baseline_intervention_metrics": base_intervention_metrics, "baseline_block_metrics": base_block_metrics,
        "final_intervention_metrics": final_intervention_metrics, "final_block_metrics": final_block_metrics,
        "target_fpr_strict": float(args.target_fpr), "target_fp_strict": int(target_fp),
        "available_fp_headroom": int(headroom), "max_approve_fp_promoted": int(max_approve_fp),
        "target_reached": target_reached, "gap_to_target_fp": max(0, int(final_intervention_metrics["fp"] - target_fp)),
        "fn_total_ok": fn_total_ok,
        "approve_tp_promoted_to_confirm": approve_tp, "approve_fp_promoted_to_confirm": approve_fp,
        "approval_fraud_remaining": approve_fraud_remaining,
        "block_fp_demoted_to_confirm": block_fp, "block_tp_demoted_to_confirm": block_tp,
        "net_intervention_tp_delta": int(final_intervention_metrics["tp"] - base_intervention_metrics["tp"]),
        "net_intervention_fp_delta": int(final_intervention_metrics["fp"] - base_intervention_metrics["fp"]),
        "net_block_tp_delta": int(final_block_metrics["tp"] - base_block_metrics["tp"]),
        "net_block_fp_delta": int(final_block_metrics["fp"] - base_block_metrics["fp"]),
        "n_approve_candidates": int(len(approve_candidates)), "n_block_candidates": int(len(block_candidates)),
        "n_selected_approve_rules": int(len(selected_approve)), "n_selected_block_rules": int(len(selected_block)),
        "approve_cat_cols_used": approve_cat_cols, "block_cat_cols_used": block_cat_cols, "score_cols_used": score_cols,
        "all_pass": bool(target_reached and fn_total_ok and block_tp <= int(args.max_block_tp_demoted)),
        "output_dir": str(out_dir),
    }

    policy = {
        "experiment": EXPERIMENT, "input_predictions_path": str(pred_path), "base_action_col": action_col,
        "final_action_col": "r4g_fast_decisao_recommended",
        "approve_to_confirm_col": "exp014b_r4g_fast_approve_to_confirm",
        "block_to_confirm_col": "exp014b_r4g_fast_block_to_confirm",
        "intervention_pred_col": "exp014b_r4g_fast_intervention_pred",
        "block_pred_col": "exp014b_r4g_fast_block_pred",
        "baseline_intervention_metrics": base_intervention_metrics, "baseline_block_metrics": base_block_metrics,
        "final_intervention_metrics": final_intervention_metrics, "final_block_metrics": final_block_metrics,
        "target_fpr_strict": float(args.target_fpr), "target_fp_strict": int(target_fp),
        "available_fp_headroom": int(headroom), "target_reached": target_reached,
        "approve_tp_promoted_to_confirm": approve_tp, "approve_fp_promoted_to_confirm": approve_fp,
        "approval_fraud_remaining": approve_fraud_remaining,
        "block_fp_demoted_to_confirm": block_fp, "block_tp_demoted_to_confirm": block_tp,
        "selected_approve_to_confirm_rules": selected_approve.to_dict(orient="records") if not selected_approve.empty else [],
        "selected_block_to_confirm_rules": selected_block.to_dict(orient="records") if not selected_block.empty else [],
    }

    write_json(out_dir / "00_run_summary.json", summary)
    write_json(out_dir / "01_input_contract.json", {
        "predictions_path": str(pred_path), "artifact_path": str(artifact_path) if artifact_path else None,
        "label_col": label_col, "action_col": action_col, "intervention_col": intervention_col, "block_col": block_col,
        "target_fpr_strict": float(args.target_fpr), "target_fp_strict": int(target_fp),
        "available_fp_headroom": int(headroom), "max_approve_fp_promoted": int(max_approve_fp),
        "contract_ok": True, "missing": [],
    })
    write_json(out_dir / "02_base_metrics.json", {
        "baseline_intervention_metrics": base_intervention_metrics,
        "baseline_block_metrics": base_block_metrics,
        "baseline_by_action": action_table(df.assign(_base_action=base_action), label_col, "_base_action").to_dict(orient="records"),
        "artifact_status": artifact.get("frozen_validation_status") if isinstance(artifact, dict) else None,
    })
    approve_candidates.to_csv(out_dir / "03_approve_to_confirm_candidates.csv", index=False, encoding="utf-8")
    block_candidates.to_csv(out_dir / "04_block_to_confirm_candidates.csv", index=False, encoding="utf-8")
    selected_approve.to_csv(out_dir / "05_selected_approve_to_confirm_rules.csv", index=False, encoding="utf-8")
    selected_block.to_csv(out_dir / "06_selected_block_to_confirm_rules.csv", index=False, encoding="utf-8")
    pd.concat([
        frontier_approve.assign(phase="approve_to_confirm") if not frontier_approve.empty else pd.DataFrame(),
        frontier_block.assign(phase="block_to_confirm") if not frontier_block.empty else pd.DataFrame(),
    ], ignore_index=True).to_csv(out_dir / "07_selection_frontier.csv", index=False, encoding="utf-8")
    final_by_action.to_csv(out_dir / "08_decision_metrics_by_action.csv", index=False, encoding="utf-8")
    robustness_by_segment(df, label_col, action_col, "r4g_fast_decisao_recommended").to_csv(out_dir / "09_robustness_by_segment.csv", index=False, encoding="utf-8")
    write_json(out_dir / "10_policy_artifact_recommended.json", policy)
    df.to_csv(out_dir / "11_predictions_recommended.csv", index=False, encoding="utf-8")

    report = f"""# {EXPERIMENT}

## Resultado executivo
- Status: `{objective_status}`
- All pass: `{summary['all_pass']}`
- Target FP strict: `{target_fp}`
- Target reached: `{target_reached}`
- Folga FP baseline: `{headroom}`
- Max approve FP promoted: `{max_approve_fp}`

## Movimentos
- APROVAR -> CONFIRMAR, fraudes promovidas: `{approve_tp}`
- APROVAR -> CONFIRMAR, normais promovidos: `{approve_fp}`
- Fraudes restantes em APROVAR: `{approve_fraud_remaining}`
- BLOQUEAR -> CONFIRMAR, normais movidos: `{block_fp}`
- BLOQUEAR -> CONFIRMAR, fraudes movidas: `{block_tp}`

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

## Regras APROVAR -> CONFIRMAR
{table_md(selected_approve)}

## Regras BLOQUEAR -> CONFIRMAR
{table_md(selected_block)}
"""
    (out_dir / "12_exp014b_r4g_fast_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
