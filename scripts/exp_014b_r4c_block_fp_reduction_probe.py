# -*- coding: utf-8 -*-
"""
EXP-014B-R4C — Block FP Reduction Probe

Objetivo:
  Tentar atingir a meta final FPR < 1% partindo do baseline campeão R4A-FROZEN,
  atacando especificamente o gargalo BLOQUEAR.

Motivação:
  R4A-FROZEN atingiu FPR <= 1,5% com FN=0:
    TP=1465, FP=1677, FN=0, FPR=1,492%
  Porém BLOQUEAR sozinho ainda tem cerca de 1200 FP.
  Para FPR < 1%, o alvo estrito é FP <= 1123.
  Portanto, é necessário reduzir FP dentro do BLOQUEAR.

Regra de segurança:
  - Base padrão: EXP-014B-R4A-FROZEN.
  - Apenas linhas BLOQUEAR são elegíveis para demotion.
  - Demotion significa BLOQUEAR -> APROVAR.
  - FN total máximo default: 5.
  - Não mexe em CONFIRMAR.
  - Busca eficiente por groupby/poda; evita quints exaustivos.

Saídas:
  resultados/experimentos/EXP-014B-R4C/
    00_run_summary.json
    01_input_contract.json
    02_base_metrics.json
    03_block_candidates.csv
    04_selection_frontier.csv
    05_selected_block_demotions.csv
    06_decision_metrics_by_action.csv
    07_robustness_by_segment.csv
    08_policy_artifact_recommended.json
    09_predictions_recommended.csv
    10_exp014b_r4c_report.md
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT = "EXP-014B-R4C"

LABEL_CANDIDATES = ["is_fraud", "fraude", "target", "label", "tp_fraude"]

ACTION_CANDIDATES = [
    "r4a_frozen_decisao_recommended",
    "r4a_decisao_recommended",
    "r3z_frozen_decisao_recommended",
    "r3z_decisao_recommended",
]

BLOCK_CANDIDATES = [
    "exp014b_r4a_frozen_block_pred",
    "exp014b_r4a_block_pred",
    "exp014b_r3z_frozen_block_pred",
    "exp014b_r3z_block_pred",
]

INTERVENTION_CANDIDATES = [
    "exp014b_r4a_frozen_intervention_pred",
    "exp014b_r4a_intervention_pred",
    "exp014b_r3z_frozen_intervention_pred",
    "exp014b_r3z_intervention_pred",
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

CAT_COLS_PRIORITY = [
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
    "mbk_available_flag",
    "first_receiver_flag_real",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default=None)
    p.add_argument("--artifact", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--max-total-fn", type=int, default=5)
    p.add_argument("--min-support", type=int, default=2)
    p.add_argument("--min-incremental-fp", type=int, default=1)
    p.add_argument("--max-rules", type=int, default=80)
    p.add_argument("--max-candidates", type=int, default=8000)
    p.add_argument("--enable-quads", action="store_true")
    p.add_argument("--continue-after-target", action="store_true")
    p.add_argument("--score-cat-top-values", type=int, default=30)
    p.add_argument("--combo-topn", type=int, default=500)
    return p.parse_args()


def default_paths() -> tuple[Path, Path | None, Path]:
    root = Path.cwd()
    frozen = root / "resultados" / "experimentos" / "EXP-014B-R4A-FROZEN" / "06_predictions_frozen.csv"
    rec = root / "resultados" / "experimentos" / "EXP-014B-R4A" / "09_predictions_recommended.csv"
    pred = frozen if frozen.exists() else rec

    artifact = root / "resultados" / "experimentos" / "EXP-014B-R4A-FROZEN" / "05_policy_artifact_frozen.json"
    if not artifact.exists():
        artifact = root / "resultados" / "experimentos" / "EXP-014B-R4A" / "08_policy_artifact_recommended.json"

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
        raise KeyError(f"Nenhuma coluna encontrada entre: {candidates}")
    return None


def int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def norm_action(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def pred_intervention(action: pd.Series) -> pd.Series:
    return norm_action(action).isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)


def pred_block(action: pd.Series) -> pd.Series:
    return norm_action(action).eq("BLOQUEAR").astype(int)


def metrics(y_true: pd.Series, pred: pd.Series) -> dict[str, Any]:
    y = int_series(y_true)
    p = int_series(pred)
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


def strict_target_fp(n_normals: int, target_fpr: float) -> int:
    # FPR < target_fpr. For n_normals=112379 and target=0.01, this returns 1123.
    return int(np.ceil(float(target_fpr) * int(n_normals)) - 1)


def make_candidate(
    candidate_id: str,
    rule_type: str,
    description: str,
    local_mask: np.ndarray,
    y_eligible: np.ndarray,
    min_support: int,
    min_fp: int,
) -> tuple[dict[str, Any], np.ndarray] | None:
    n = int(local_mask.sum())
    if n < min_support:
        return None

    yy = y_eligible[local_mask]
    fp_removed = int((yy == 0).sum())
    tp_loss = int((yy == 1).sum())

    if fp_removed < min_fp:
        return None

    row = {
        "candidate_id": candidate_id,
        "rule_type": rule_type,
        "description": description,
        "n_demoted": n,
        "fp_removed": fp_removed,
        "tp_loss": tp_loss,
        "precision_demoted": round(float(tp_loss / n), 8) if n else 0.0,
        "fp_per_tp_loss": round(float(fp_removed / max(tp_loss, 1)), 8),
    }
    return row, local_mask.copy()


def add_candidate(
    rows: list[dict[str, Any]],
    masks: dict[str, np.ndarray],
    seen: set[str],
    candidate_id: str,
    rule_type: str,
    description: str,
    local_mask: np.ndarray,
    y_eligible: np.ndarray,
    min_support: int,
    min_fp: int,
) -> None:
    if description in seen:
        return
    item = make_candidate(candidate_id, rule_type, description, local_mask, y_eligible, min_support, min_fp)
    if item is None:
        return
    row, mask = item
    seen.add(description)
    rows.append(row)
    masks[row["candidate_id"]] = mask


def build_group_candidates(
    eligible: pd.DataFrame,
    y_eligible: np.ndarray,
    cols: list[str],
    rows: list[dict[str, Any]],
    masks: dict[str, np.ndarray],
    seen: set[str],
    min_support: int,
    min_fp: int,
    combo_topn: int,
) -> None:
    if not cols:
        return

    tmp = eligible[cols].fillna("<MISSING>").astype(str)
    vc = tmp.value_counts(dropna=False).head(combo_topn)

    for vals, support in vc.items():
        vals_tuple = vals if isinstance(vals, tuple) else (vals,)
        if int(support) < min_support:
            continue

        mask = np.ones(len(eligible), dtype=bool)
        parts = []
        safe_parts = []
        for c, v in zip(cols, vals_tuple):
            v_str = str(v)
            mask &= tmp[c].to_numpy() == v_str
            parts.append(f"{c} == {v_str}")
            safe_parts.append(f"{c}={v_str[:24]}")

        cid = f"cat{len(cols)}__" + "__".join(safe_parts)
        desc = "Demover BLOQUEAR R4C com " + " AND ".join(parts)
        add_candidate(rows, masks, seen, cid, f"categorical_{len(cols)}", desc, mask, y_eligible, min_support, min_fp)


def build_score_candidates(
    eligible: pd.DataFrame,
    y_eligible: np.ndarray,
    score_cols: list[str],
    rows: list[dict[str, Any]],
    masks: dict[str, np.ndarray],
    seen: set[str],
    min_support: int,
    min_fp: int,
) -> None:
    qs = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25,
          0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95,
          0.97, 0.98, 0.99, 0.995]

    for c in score_cols:
        s = pd.to_numeric(eligible[c], errors="coerce")
        valid = s.dropna()
        if valid.empty:
            continue
        values = s.to_numpy()
        thresholds = sorted(set(float(valid.quantile(q)) for q in qs if pd.notna(valid.quantile(q))))
        for th in thresholds:
            lo = np.isfinite(values) & (values <= th)
            hi = np.isfinite(values) & (values >= th)
            add_candidate(rows, masks, seen, f"score_lo__{c}__{th:.12g}", "score_threshold",
                          f"Demover BLOQUEAR R4C com {c} <= {th:.12g}", lo, y_eligible, min_support, min_fp)
            add_candidate(rows, masks, seen, f"score_hi__{c}__{th:.12g}", "score_threshold",
                          f"Demover BLOQUEAR R4C com {c} >= {th:.12g}", hi, y_eligible, min_support, min_fp)


def build_score_cat_candidates(
    eligible: pd.DataFrame,
    y_eligible: np.ndarray,
    score_cols: list[str],
    cat_cols: list[str],
    rows: list[dict[str, Any]],
    masks: dict[str, np.ndarray],
    seen: set[str],
    min_support: int,
    min_fp: int,
    top_values: int,
) -> None:
    qs = [0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90, 0.95]

    for score_col in score_cols:
        s = pd.to_numeric(eligible[score_col], errors="coerce")
        valid = s.dropna()
        if valid.empty:
            continue
        s_values = s.to_numpy()
        thresholds = sorted(set(float(valid.quantile(q)) for q in qs if pd.notna(valid.quantile(q))))

        for cat_col in cat_cols:
            cat = eligible[cat_col].fillna("<MISSING>").astype(str)
            top = cat.value_counts(dropna=False).head(top_values)
            cat_values = cat.to_numpy()

            for cat_val, support in top.items():
                if int(support) < min_support:
                    continue
                base = cat_values == str(cat_val)

                for th in thresholds:
                    lo = base & np.isfinite(s_values) & (s_values <= th)
                    hi = base & np.isfinite(s_values) & (s_values >= th)

                    safe = str(cat_val)[:24]
                    add_candidate(rows, masks, seen,
                                  f"scorecat_lo__{cat_col}={safe}__{score_col}__{th:.12g}",
                                  "score_cat_1",
                                  f"Demover BLOQUEAR R4C com {cat_col} == {cat_val} AND {score_col} <= {th:.12g}",
                                  lo, y_eligible, min_support, min_fp)
                    add_candidate(rows, masks, seen,
                                  f"scorecat_hi__{cat_col}={safe}__{score_col}__{th:.12g}",
                                  "score_cat_1",
                                  f"Demover BLOQUEAR R4C com {cat_col} == {cat_val} AND {score_col} >= {th:.12g}",
                                  hi, y_eligible, min_support, min_fp)


def mine_candidates(
    df: pd.DataFrame,
    eligible_idx: np.ndarray,
    y: np.ndarray,
    cat_cols: list[str],
    score_cols: list[str],
    min_support: int,
    min_fp: int,
    max_candidates: int,
    enable_quads: bool,
    combo_topn: int,
    score_cat_top_values: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    eligible = df.iloc[eligible_idx].copy()
    y_eligible = y[eligible_idx]

    rows: list[dict[str, Any]] = []
    masks: dict[str, np.ndarray] = {}
    seen: set[str] = set()

    # Score thresholds.
    build_score_candidates(eligible, y_eligible, score_cols, rows, masks, seen, min_support, min_fp)

    # Categorical groups.
    for size, max_cols, topn in [(1, 28, combo_topn), (2, 24, combo_topn), (3, 16, combo_topn)]:
        for cols in itertools.combinations(cat_cols[:max_cols], size):
            build_group_candidates(eligible, y_eligible, list(cols), rows, masks, seen, min_support, min_fp, topn)

    if enable_quads:
        for cols in itertools.combinations(cat_cols[:12], 4):
            build_group_candidates(eligible, y_eligible, list(cols), rows, masks, seen, min_support, min_fp, max(100, combo_topn // 2))

    # Score + categorical.
    build_score_cat_candidates(
        eligible, y_eligible,
        score_cols[:8], cat_cols[:18],
        rows, masks, seen,
        min_support, min_fp, score_cat_top_values
    )

    if not rows:
        return pd.DataFrame(), {}

    cand = pd.DataFrame(rows).drop_duplicates(subset=["description"])
    # Align masks to kept ids.
    kept = set(cand["candidate_id"].astype(str))
    masks = {k: v for k, v in masks.items() if k in kept}

    cand = cand.sort_values(
        ["tp_loss", "fp_removed", "fp_per_tp_loss", "n_demoted"],
        ascending=[True, False, False, False],
    ).head(max_candidates).reset_index(drop=True)
    kept = set(cand["candidate_id"].astype(str))
    masks = {k: v for k, v in masks.items() if k in kept}

    return cand, masks


def select_greedy(
    cand: pd.DataFrame,
    masks: dict[str, np.ndarray],
    y_eligible: np.ndarray,
    base_fn: int,
    base_fp: int,
    target_fp: int,
    max_total_fn: int,
    max_rules: int,
    min_incremental_fp: int,
    continue_after_target: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    selected_mask = np.zeros_like(y_eligible, dtype=bool)
    selected_rows = []
    frontier_rows = []
    cumulative_fp = 0
    cumulative_fn_added = 0
    remaining = cand.copy().reset_index(drop=True)

    for step in range(1, max_rules + 1):
        current_fp = base_fp - cumulative_fp
        if current_fp <= target_fp and not continue_after_target:
            break

        best = None
        best_mask = None
        best_score = None

        for _, row in remaining.iterrows():
            cid = str(row["candidate_id"])
            mask = masks.get(cid)
            if mask is None:
                continue
            inc_mask = mask & (~selected_mask)
            n = int(inc_mask.sum())
            if n == 0:
                continue
            yy = y_eligible[inc_mask]
            fp_gain = int((yy == 0).sum())
            tp_loss = int((yy == 1).sum())

            if fp_gain < min_incremental_fp:
                continue
            if base_fn + cumulative_fn_added + tp_loss > max_total_fn:
                continue

            # Score:
            # prioritize zero TP loss, then FP/TP, then absolute FP.
            score = (
                1 if tp_loss == 0 else 0,
                fp_gain / max(tp_loss, 1),
                fp_gain,
                -tp_loss,
                -n,
            )

            if best is None or score > best_score:
                best = row.copy()
                best_mask = inc_mask
                best_score = score
                best["incremental_n_demoted"] = n
                best["incremental_fp_removed"] = fp_gain
                best["incremental_tp_loss"] = tp_loss

        if best is None or best_mask is None:
            break

        selected_mask |= best_mask
        cumulative_fp += int(best["incremental_fp_removed"])
        cumulative_fn_added += int(best["incremental_tp_loss"])

        result_fp = base_fp - cumulative_fp
        result_fn = base_fn + cumulative_fn_added

        best["selection_step"] = step
        best["cumulative_n_demoted"] = int(selected_mask.sum())
        best["cumulative_fp_removed"] = int(cumulative_fp)
        best["cumulative_fn_added"] = int(cumulative_fn_added)
        best["result_fp"] = int(result_fp)
        best["result_fn"] = int(result_fn)
        best["target_reached"] = bool(result_fp <= target_fp)
        selected_rows.append(best)

        frontier_rows.append({
            "selection_step": step,
            "selected_candidate_id": str(best["candidate_id"]),
            "selected_description": str(best["description"]),
            "incremental_n_demoted": int(best["incremental_n_demoted"]),
            "incremental_fp_removed": int(best["incremental_fp_removed"]),
            "incremental_tp_loss": int(best["incremental_tp_loss"]),
            "cumulative_n_demoted": int(selected_mask.sum()),
            "cumulative_fp_removed": int(cumulative_fp),
            "cumulative_fn_added": int(cumulative_fn_added),
            "result_fp": int(result_fp),
            "result_fn": int(result_fn),
            "target_reached": bool(result_fp <= target_fp),
        })

        remaining = remaining[remaining["candidate_id"].astype(str) != str(best["candidate_id"])].reset_index(drop=True)

    return pd.DataFrame(selected_rows), pd.DataFrame(frontier_rows), selected_mask


def metrics_by_action(df: pd.DataFrame, label_col: str, action_col: str) -> pd.DataFrame:
    y = int_series(df[label_col])
    rows = []
    for action, idx in df.groupby(action_col, dropna=False).groups.items():
        idx = list(idx)
        yy = y.loc[idx]
        n = int(len(yy))
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
    y = int_series(df[label_col])
    before = pred_intervention(df[before_col])
    after = pred_intervention(df[after_col])

    for col in SEGMENT_COLS:
        if col not in df.columns:
            continue
        for val, idx in df.groupby(col, dropna=False).groups.items():
            idx = list(idx)
            before_m = metrics(y.loc[idx], before.loc[idx])
            after_m = metrics(y.loc[idx], after.loc[idx])
            rows.append({
                "segment_col": col,
                "segment_value": str(val),
                "n_rows": int(len(idx)),
                "n_frauds": int((y.loc[idx] == 1).sum()),
                "fp_removed": int(before_m["fp"] - after_m["fp"]),
                "tp_loss": int(before_m["tp"] - after_m["tp"]),
                "before_fp": before_m["fp"],
                "after_fp": after_m["fp"],
                "before_tp": before_m["tp"],
                "after_tp": after_m["tp"],
                "after_fn": after_m["fn"],
            })
    return pd.DataFrame(rows).sort_values(["fp_removed", "n_rows"], ascending=[False, False]) if rows else pd.DataFrame()


def df_to_md(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df is None or df.empty:
        return "Nenhuma linha."
    d = df.head(max_rows).copy()
    try:
        return d.to_markdown(index=False)
    except Exception:
        return d.to_string(index=False)


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
    df = df.copy()  # defragmenta DataFrame de entrada

    artifact = None
    if artifact_path and Path(artifact_path).exists():
        artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))

    label_col = find_col(df, LABEL_CANDIDATES)
    action_col = find_col(df, ACTION_CANDIDATES)
    block_col = find_col(df, BLOCK_CANDIDATES, required=False)
    intervention_col = find_col(df, INTERVENTION_CANDIDATES, required=False)

    y = int_series(df[label_col]).to_numpy()
    action = norm_action(df[action_col])

    if block_col:
        base_block = int_series(df[block_col])
    else:
        base_block = pred_block(action)

    if intervention_col:
        base_intervention = int_series(df[intervention_col])
    else:
        base_intervention = pred_intervention(action)

    base_intervention_metrics = metrics(pd.Series(y), base_intervention)
    base_block_metrics = metrics(pd.Series(y), base_block)

    n_normals = int((y == 0).sum())
    target_fp = strict_target_fp(n_normals, float(args.target_fpr))
    base_fn = int(base_intervention_metrics["fn"])
    base_fp = int(base_intervention_metrics["fp"])

    # R4C elegível somente BLOQUEAR.
    eligible_mask = action.eq("BLOQUEAR").to_numpy()
    eligible_idx = np.flatnonzero(eligible_mask)

    cat_cols = [c for c in CAT_COLS_PRIORITY if c in df.columns]
    # Keep categorical columns with useful variability inside BLOQUEAR.
    cat_cols = [
        c for c in cat_cols
        if 1 < df.loc[eligible_mask, c].fillna("<MISSING>").astype(str).nunique(dropna=False) <= 80
    ]
    score_cols = [c for c in SCORE_COLS if c in df.columns]

    candidates, masks = mine_candidates(
        df=df,
        eligible_idx=eligible_idx,
        y=y,
        cat_cols=cat_cols,
        score_cols=score_cols,
        min_support=int(args.min_support),
        min_fp=int(args.min_incremental_fp),
        max_candidates=int(args.max_candidates),
        enable_quads=bool(args.enable_quads),
        combo_topn=int(args.combo_topn),
        score_cat_top_values=int(args.score_cat_top_values),
    )

    if candidates.empty:
        selected = pd.DataFrame()
        frontier = pd.DataFrame()
        selected_local = np.zeros(len(eligible_idx), dtype=bool)
    else:
        selected, frontier, selected_local = select_greedy(
            cand=candidates,
            masks=masks,
            y_eligible=y[eligible_idx],
            base_fn=base_fn,
            base_fp=base_fp,
            target_fp=target_fp,
            max_total_fn=int(args.max_total_fn),
            max_rules=int(args.max_rules),
            min_incremental_fp=int(args.min_incremental_fp),
            continue_after_target=bool(args.continue_after_target),
        )

    full_demote = np.zeros(len(df), dtype=bool)
    full_demote[eligible_idx] = selected_local

    final_action = action.copy()
    final_action.loc[full_demote] = "APROVAR"

    df["exp014b_r4c_demote_block_to_approve"] = full_demote.astype(int)
    df["r4c_decisao_recommended"] = final_action
    df["exp014b_r4c_intervention_pred"] = pred_intervention(final_action)
    df["exp014b_r4c_block_pred"] = pred_block(final_action)

    final_intervention_metrics = metrics(pd.Series(y), df["exp014b_r4c_intervention_pred"])
    final_block_metrics = metrics(pd.Series(y), df["exp014b_r4c_block_pred"])

    fp_removed_total = int(base_intervention_metrics["fp"] - final_intervention_metrics["fp"])
    fn_added_total = int(final_intervention_metrics["fn"] - base_intervention_metrics["fn"])
    block_fp_removed = int(base_block_metrics["fp"] - final_block_metrics["fp"])
    block_tp_loss = int(base_block_metrics["tp"] - final_block_metrics["tp"])
    target_reached = bool(final_intervention_metrics["fp"] <= target_fp)
    gap = max(0, int(final_intervention_metrics["fp"] - target_fp))

    by_action = metrics_by_action(df, label_col, "r4c_decisao_recommended")
    rob = robustness(df, label_col, action_col, "r4c_decisao_recommended")

    summary = {
        "experiment": EXPERIMENT,
        "status": "DONE",
        "objective_status": (
            "DONE_R4C_FPR_LT1_TARGET_REACHED_WITHIN_FN_BUDGET"
            if target_reached and final_intervention_metrics["fn"] <= int(args.max_total_fn)
            else "DONE_R4C_BLOCK_FP_REDUCED_TARGET_NOT_REACHED"
            if fp_removed_total > 0 and final_intervention_metrics["fn"] <= int(args.max_total_fn)
            else "DONE_R4C_NO_SAFE_BLOCK_FP_REDUCTION"
        ),
        "n_rows": int(len(df)),
        "n_frauds": int((y == 1).sum()),
        "n_normals": n_normals,
        "predictions_path": str(pred_path),
        "artifact_path": str(artifact_path) if artifact_path else None,
        "label_col": label_col,
        "action_col": action_col,
        "block_col": block_col,
        "intervention_col": intervention_col,
        "eligible_action": "BLOQUEAR",
        "baseline_intervention_metrics": base_intervention_metrics,
        "baseline_block_metrics": base_block_metrics,
        "final_intervention_metrics": final_intervention_metrics,
        "final_block_metrics": final_block_metrics,
        "target_fpr_strict": float(args.target_fpr),
        "target_fp_strict": int(target_fp),
        "target_reached": target_reached,
        "gap_to_target_fp": int(gap),
        "fp_removed_total": fp_removed_total,
        "fn_added_total": fn_added_total,
        "block_fp_removed": block_fp_removed,
        "block_tp_loss": block_tp_loss,
        "n_eligible_block_rows": int(len(eligible_idx)),
        "n_candidates_evaluated": int(len(candidates)),
        "n_selected_rules": int(len(selected)),
        "cat_cols_used": cat_cols,
        "score_cols_used": score_cols,
        "min_support": int(args.min_support),
        "min_incremental_fp": int(args.min_incremental_fp),
        "max_rules": int(args.max_rules),
        "max_total_fn": int(args.max_total_fn),
        "enable_quads": bool(args.enable_quads),
        "all_pass": bool(final_intervention_metrics["fn"] <= int(args.max_total_fn)),
        "output_dir": str(out_dir),
    }

    contract = {
        "predictions_path": str(pred_path),
        "artifact_path": str(artifact_path) if artifact_path else None,
        "label_col": label_col,
        "action_col": action_col,
        "block_col": block_col,
        "intervention_col": intervention_col,
        "eligible_action": "BLOQUEAR",
        "target_fpr_strict": float(args.target_fpr),
        "target_fp_strict": int(target_fp),
        "max_total_fn": int(args.max_total_fn),
        "contract_ok": True,
        "missing": [],
    }

    base_metrics = {
        "baseline_intervention_metrics": base_intervention_metrics,
        "baseline_block_metrics": base_block_metrics,
        "artifact_status": artifact.get("frozen_validation_status") if isinstance(artifact, dict) else None,
    }

    rec_artifact = {
        "experiment": EXPERIMENT,
        "input_predictions_path": str(pred_path),
        "base_action_col": action_col,
        "final_action_col": "r4c_decisao_recommended",
        "demote_col": "exp014b_r4c_demote_block_to_approve",
        "intervention_pred_col": "exp014b_r4c_intervention_pred",
        "block_pred_col": "exp014b_r4c_block_pred",
        "baseline_intervention_metrics": base_intervention_metrics,
        "baseline_block_metrics": base_block_metrics,
        "final_intervention_metrics": final_intervention_metrics,
        "final_block_metrics": final_block_metrics,
        "target_fpr_strict": float(args.target_fpr),
        "target_fp_strict": int(target_fp),
        "target_reached": target_reached,
        "gap_to_target_fp": int(gap),
        "fp_removed_total": fp_removed_total,
        "fn_added_total": fn_added_total,
        "block_fp_removed": block_fp_removed,
        "block_tp_loss": block_tp_loss,
        "selected_demotions": selected.to_dict(orient="records") if not selected.empty else [],
        "notes": [
            "R4C only demotes BLOQUEAR rows to APROVAR.",
            "This is a diagnostic search for the strict FPR < 1% target.",
            "Promotion requires frozen replay and business review because BLOQUEAR semantics changed.",
        ],
    }

    write_json(out_dir / "00_run_summary.json", summary)
    write_json(out_dir / "01_input_contract.json", contract)
    write_json(out_dir / "02_base_metrics.json", base_metrics)
    candidates.to_csv(out_dir / "03_block_candidates.csv", index=False, encoding="utf-8")
    frontier.to_csv(out_dir / "04_selection_frontier.csv", index=False, encoding="utf-8")
    selected.to_csv(out_dir / "05_selected_block_demotions.csv", index=False, encoding="utf-8")
    by_action.to_csv(out_dir / "06_decision_metrics_by_action.csv", index=False, encoding="utf-8")
    rob.to_csv(out_dir / "07_robustness_by_segment.csv", index=False, encoding="utf-8")
    write_json(out_dir / "08_policy_artifact_recommended.json", rec_artifact)
    df.to_csv(out_dir / "09_predictions_recommended.csv", index=False, encoding="utf-8")

    report = f"""# {EXPERIMENT} - Block FP Reduction Probe

## Resultado executivo
- Status: `{summary["objective_status"]}`
- All pass: `{summary["all_pass"]}`
- Target FPR strict: `{args.target_fpr}`
- Target FP strict: `{target_fp}`
- Target reached: `{target_reached}`
- Gap FP to target: `{gap}`
- FP removidos total: `{fp_removed_total}`
- FN adicionados: `{fn_added_total}`
- Block FP removidos: `{block_fp_removed}`
- Block TP loss: `{block_tp_loss}`
- Candidatos avaliados: `{len(candidates)}`
- Regras selecionadas: `{len(selected)}`

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
{df_to_md(by_action)}

## Regras selecionadas
{df_to_md(selected)}

## Frontier
{df_to_md(frontier)}

## Decisão sugerida
Se `target_reached=true`, executar R4C-FROZEN e revisão semântica das regras que mexeram em BLOQUEAR.
Se `target_reached=false`, o resultado indica que não há bloco benigno grande o bastante dentro de BLOQUEAR com as features atuais.
"""
    (out_dir / "10_exp014b_r4c_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
