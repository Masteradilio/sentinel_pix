# -*- coding: utf-8 -*-
"""
EXP-014B-R4A — FPR 1.5 Pathfinder after R3Z-long

Objetivo:
  Buscar o caminho restante até FPR <= 1.5% após o melhor R3Z disponível.

Entrada padrão:
  1) resultados/experimentos/EXP-014B-R3Z-FROZEN/06_predictions_frozen.csv
  2) fallback: resultados/experimentos/EXP-014B-R3Z/09_predictions_recommended.csv

Regras:
  - BLOQUEAR deve permanecer intocado.
  - Somente CONFIRMAR residual pode virar APROVAR.
  - Orçamento default: FN total <= 5.
  - Meta default: FPR <= 1.5%.
  - Se a meta for atingida, a seleção para por padrão.

Saídas:
  resultados/experimentos/EXP-014B-R4A/
    00_run_summary.json
    01_input_contract.json
    02_base_metrics.json
    03_candidates.csv
    04_selection_frontier.csv
    05_selected_demotions.csv
    06_decision_metrics_by_action.csv
    07_robustness_by_segment.csv
    08_policy_artifact_recommended.json
    09_predictions_recommended.csv
    10_exp014b_r4a_report.md
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT = "EXP-014B-R4A"

LABEL_CANDIDATES = ["is_fraud", "fraude", "target", "label", "tp_fraude"]

ACTION_CANDIDATES = [
    "r3z_frozen_decisao_recommended",
    "r3z_decisao_recommended",
    "r3y_frozen_decisao_recommended",
    "r3y_decisao_recommended",
]

BLOCK_CANDIDATES = [
    "exp014b_r3z_frozen_block_pred",
    "exp014b_r3z_block_pred",
    "exp014b_r3y_frozen_block_pred",
    "exp014b_r3y_block_pred",
]

INTERVENTION_CANDIDATES = [
    "exp014b_r3z_frozen_intervention_pred",
    "exp014b_r3z_intervention_pred",
    "exp014b_r3y_frozen_intervention_pred",
    "exp014b_r3y_intervention_pred",
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
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=str, default=None)
    p.add_argument("--artifact", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--target-fpr", type=float, default=0.015)
    p.add_argument("--max-total-fn", type=int, default=5)
    p.add_argument("--min-incremental-fp", type=int, default=2)
    p.add_argument("--max-rules", type=int, default=80)
    p.add_argument("--max-candidates", type=int, default=6000)
    p.add_argument("--enable-quads", action="store_true")
    p.add_argument("--continue-after-target", action="store_true")
    return p.parse_args()


def default_paths() -> tuple[Path, Path | None, Path]:
    root = Path.cwd()
    frozen = root / "resultados" / "experimentos" / "EXP-014B-R3Z-FROZEN" / "06_predictions_frozen.csv"
    rec = root / "resultados" / "experimentos" / "EXP-014B-R3Z" / "09_predictions_recommended.csv"
    pred = frozen if frozen.exists() else rec

    artifact = root / "resultados" / "experimentos" / "EXP-014B-R3Z-FROZEN" / "05_policy_artifact_frozen.json"
    if not artifact.exists():
        artifact = root / "resultados" / "experimentos" / "EXP-014B-R3Z" / "08_policy_artifact_recommended.json"
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


def safe_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def normalize_action_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def action_to_intervention(action: pd.Series) -> pd.Series:
    return action.astype(str).str.upper().isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)


def action_to_block(action: pd.Series) -> pd.Series:
    return action.astype(str).str.upper().eq("BLOQUEAR").astype(int)


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
    return {
        "candidate_id": candidate_id,
        "rule_type": rule_type,
        "description": description,
        "n_demoted": n,
        "fp_removed": fp_removed,
        "tp_loss": tp_loss,
        "precision_demoted": round(float(tp_loss / n), 8) if n else 0.0,
        "fp_per_tp_loss": round(float(fp_removed / max(tp_loss, 1)), 8),
    }


def mask_from_description(df: pd.DataFrame, description: str) -> pd.Series:
    prefix = "Demover CONFIRMAR R4A com "
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
        mask = pd.Series(True, index=df.index)
        for part in expr.split(" AND "):
            if " == " in part:
                col, val = part.split(" == ", 1)
                mask &= df[col].fillna("<MISSING>").astype(str).eq(str(val))
            elif " <= " in part:
                col, th = part.split(" <= ", 1)
                mask &= pd.to_numeric(df[col], errors="coerce").le(float(th))
            elif " >= " in part:
                col, th = part.split(" >= ", 1)
                mask &= pd.to_numeric(df[col], errors="coerce").ge(float(th))
            else:
                raise ValueError(f"Parte não suportada: {part}")
        return mask

    if " == " in expr:
        col, val = expr.split(" == ", 1)
        return df[col].fillna("<MISSING>").astype(str).eq(str(val))

    raise ValueError(f"Expressão não suportada: {expr}")


def build_candidates(
    df: pd.DataFrame,
    label_col: str,
    action_col: str,
    min_incremental_fp: int,
    max_candidates: int,
    enable_quads: bool,
) -> pd.DataFrame:
    action = normalize_action_series(df[action_col])
    confirm_mask = action.eq("CONFIRMAR")
    rows: list[dict[str, Any]] = []

    score_cols = [c for c in SCORE_COLS if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    useful_cat_cols = [
        c for c in cat_cols
        if 1 < df.loc[confirm_mask, c].fillna("<MISSING>").astype(str).nunique(dropna=False) <= 50
    ]

    quantiles = [
        0.005, 0.01, 0.02, 0.03, 0.05,
        0.08, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.40, 0.50, 0.60, 0.70,
        0.80, 0.85, 0.90, 0.92, 0.95,
        0.97, 0.98, 0.99, 0.995,
    ]

    # Score thresholds.
    for col in score_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s[confirm_mask & s.notna()]
        if valid.empty:
            continue
        thresholds = sorted(set(float(valid.quantile(q)) for q in quantiles if pd.notna(valid.quantile(q))))
        for th in thresholds:
            for direction, cond, op in [
                ("score_lo", s.le(th), "<="),
                ("score_hi", s.ge(th), ">="),
            ]:
                row = build_candidate_row(
                    df, label_col, confirm_mask, cond,
                    f"{direction}__{col}__{th:.12g}",
                    "score_threshold",
                    f"Demover CONFIRMAR R4A com {col} {op} {th:.12g}",
                )
                if row:
                    rows.append(row)

    # Categorical singles.
    for c in useful_cat_cols:
        vals = df.loc[confirm_mask, c].fillna("<MISSING>").astype(str).value_counts(dropna=False)
        for v, cnt in vals.items():
            if int(cnt) < min_incremental_fp:
                continue
            cond = df[c].fillna("<MISSING>").astype(str).eq(str(v))
            row = build_candidate_row(
                df, label_col, confirm_mask, cond,
                f"cat__{c}={str(v)[:40]}",
                "categorical_single",
                f"Demover CONFIRMAR R4A com {c} == {v}",
            )
            if row:
                rows.append(row)

    # Categorical pairs/triplets.
    for size, limit_cols, topn in [(2, useful_cat_cols[:24], 300), (3, useful_cat_cols[:16], 250)]:
        for cols in itertools.combinations(limit_cols, size):
            tmp = df.loc[confirm_mask, list(cols)].fillna("<MISSING>").astype(str)
            counts = tmp.value_counts(dropna=False).head(topn)
            for vals, cnt in counts.items():
                vals_tuple = vals if isinstance(vals, tuple) else (vals,)
                if int(cnt) < min_incremental_fp:
                    continue
                cond = pd.Series(True, index=df.index)
                parts = []
                for col, val in zip(cols, vals_tuple):
                    cond &= df[col].fillna("<MISSING>").astype(str).eq(str(val))
                    parts.append(f"{col} == {val}")
                row = build_candidate_row(
                    df, label_col, confirm_mask, cond,
                    f"cat{size}__" + "__".join(f"{c}={str(v)[:20]}" for c, v in zip(cols, vals_tuple)),
                    f"categorical_{size}",
                    "Demover CONFIRMAR R4A com " + " AND ".join(parts),
                )
                if row:
                    rows.append(row)

    # Score + 1 categorical and score + 2 categoricals.
    score_quantiles = [0.10, 0.20, 0.30, 0.70, 0.80, 0.90, 0.95]
    for score_col in score_cols[:8]:
        s = pd.to_numeric(df[score_col], errors="coerce")
        valid = s[confirm_mask & s.notna()]
        if valid.empty:
            continue
        thresholds = sorted(set(float(valid.quantile(q)) for q in score_quantiles if pd.notna(valid.quantile(q))))

        for size, limit_cols, topn in [(1, useful_cat_cols[:16], 100), (2, useful_cat_cols[:12], 120)]:
            for cols in itertools.combinations(limit_cols, size):
                tmp = df.loc[confirm_mask, list(cols)].fillna("<MISSING>").astype(str)
                counts = tmp.value_counts(dropna=False).head(topn)
                for vals, cnt in counts.items():
                    vals_tuple = vals if isinstance(vals, tuple) else (vals,)
                    if int(cnt) < min_incremental_fp:
                        continue
                    base_cond = pd.Series(True, index=df.index)
                    cat_parts = []
                    for col, val in zip(cols, vals_tuple):
                        base_cond &= df[col].fillna("<MISSING>").astype(str).eq(str(val))
                        cat_parts.append(f"{col} == {val}")
                    for th in thresholds:
                        for direction, score_cond, op in [
                            ("scorecat_lo", s.le(th), "<="),
                            ("scorecat_hi", s.ge(th), ">="),
                        ]:
                            cond = base_cond & score_cond
                            parts = cat_parts + [f"{score_col} {op} {th:.12g}"]
                            row = build_candidate_row(
                                df, label_col, confirm_mask, cond,
                                f"{direction}__{score_col}__{th:.12g}__" + "__".join(str(v)[:20] for v in vals_tuple),
                                f"score_cat_{size}",
                                "Demover CONFIRMAR R4A com " + " AND ".join(parts),
                            )
                            if row:
                                rows.append(row)

    # Optional quads, heavy but residual queue is small.
    if enable_quads:
        for cols in itertools.combinations(useful_cat_cols[:12], 4):
            tmp = df.loc[confirm_mask, list(cols)].fillna("<MISSING>").astype(str)
            counts = tmp.value_counts(dropna=False).head(150)
            for vals, cnt in counts.items():
                vals_tuple = vals if isinstance(vals, tuple) else (vals,)
                if int(cnt) < min_incremental_fp:
                    continue
                cond = pd.Series(True, index=df.index)
                parts = []
                for col, val in zip(cols, vals_tuple):
                    cond &= df[col].fillna("<MISSING>").astype(str).eq(str(val))
                    parts.append(f"{col} == {val}")
                row = build_candidate_row(
                    df, label_col, confirm_mask, cond,
                    "cat4__" + "__".join(f"{c}={str(v)[:15]}" for c, v in zip(cols, vals_tuple)),
                    "categorical_4",
                    "Demover CONFIRMAR R4A com " + " AND ".join(parts),
                )
                if row:
                    rows.append(row)

    if not rows:
        return pd.DataFrame()

    cand = pd.DataFrame(rows).drop_duplicates(subset=["description"])
    cand = cand[cand["fp_removed"] >= int(min_incremental_fp)].copy()
    if cand.empty:
        return cand

    cand = cand.sort_values(
        ["tp_loss", "fp_removed", "fp_per_tp_loss", "n_demoted"],
        ascending=[True, False, False, False],
    ).head(max_candidates)
    return cand.reset_index(drop=True)


def select_greedy(
    df: pd.DataFrame,
    label_col: str,
    action_col: str,
    candidates: pd.DataFrame,
    base_fn: int,
    max_total_fn: int,
    target_fp: int,
    max_rules: int,
    min_incremental_fp: int,
    continue_after_target: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    y = safe_int_series(df[label_col])
    action = normalize_action_series(df[action_col])
    confirm_mask = action.eq("CONFIRMAR")

    cumulative_demote = pd.Series(False, index=df.index)
    cumulative_fp = 0
    cumulative_fn_added = 0
    selected_rows = []
    frontier_rows = []
    remaining = candidates.copy().reset_index(drop=True)

    for step in range(1, int(max_rules) + 1):
        current_pred = action_to_intervention(apply_demotions(df, action_col, cumulative_demote))
        current_m = metrics(y, current_pred)
        if current_m["fp"] <= target_fp and not continue_after_target:
            break

        best = None
        best_mask = None
        best_score = None

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

            if fp_gain < min_incremental_fp:
                continue
            if base_fn + cumulative_fn_added + fn_gain > max_total_fn:
                continue

            # Prefer:
            # 1) FN zero;
            # 2) FP/FN;
            # 3) absolute FP;
            # 4) smaller support to reduce overly broad collateral.
            score = (
                1 if fn_gain == 0 else 0,
                fp_gain / max(fn_gain, 1),
                fp_gain,
                -fn_gain,
                -n,
            )

            if best is None or score > best_score:
                best = row.copy()
                best_mask = mask
                best_score = score
                best["incremental_n_demoted"] = n
                best["incremental_fp_removed"] = fp_gain
                best["incremental_tp_loss"] = fn_gain

        if best is None or best_mask is None:
            break

        cumulative_demote |= best_mask
        cumulative_fp += int(best["incremental_fp_removed"])
        cumulative_fn_added += int(best["incremental_tp_loss"])

        selected_action = apply_demotions(df, action_col, cumulative_demote)
        selected_m = metrics(y, action_to_intervention(selected_action))

        best["selection_step"] = step
        best["cumulative_fp_removed"] = cumulative_fp
        best["cumulative_fn_added"] = cumulative_fn_added
        best["result_fp"] = selected_m["fp"]
        best["result_fn"] = selected_m["fn"]
        best["result_fpr"] = selected_m["fpr"]
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
            "cumulative_fn_added": int(cumulative_fn_added),
            "result_fp": selected_m["fp"],
            "result_fn": selected_m["fn"],
            "result_fpr": selected_m["fpr"],
            "target_reached": bool(selected_m["fp"] <= target_fp),
        })

        remaining = remaining[remaining["description"] != best["description"]].reset_index(drop=True)

    return pd.DataFrame(selected_rows), pd.DataFrame(frontier_rows), cumulative_demote


def apply_demotions(df: pd.DataFrame, action_col: str, demote_mask: pd.Series) -> pd.Series:
    action = normalize_action_series(df[action_col]).copy()
    out = action.copy()
    out[action.eq("CONFIRMAR") & demote_mask.fillna(False)] = "APROVAR"
    return out


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
    before_pred = action_to_intervention(df[before_col])
    after_pred = action_to_intervention(df[after_col])
    for col in SEGMENT_COLS:
        if col not in df.columns:
            continue
        for val, idx in df.groupby(col, dropna=False).groups.items():
            idx = list(idx)
            before_m = metrics(y.loc[idx], before_pred.loc[idx])
            after_m = metrics(y.loc[idx], after_pred.loc[idx])
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
    if artifact_path and Path(artifact_path).exists():
        artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))

    label_col = find_col(df, LABEL_CANDIDATES)
    action_col = find_col(df, ACTION_CANDIDATES)
    block_col = find_col(df, BLOCK_CANDIDATES, required=False)
    intervention_col = find_col(df, INTERVENTION_CANDIDATES, required=False)

    y = safe_int_series(df[label_col])
    action = normalize_action_series(df[action_col])
    base_intervention = safe_int_series(df[intervention_col]) if intervention_col else action_to_intervention(action)
    base_block = safe_int_series(df[block_col]) if block_col else action_to_block(action)

    base_intervention_metrics = metrics(y, base_intervention)
    base_block_metrics = metrics(y, base_block)
    base_fn = int(base_intervention_metrics["fn"])
    n_normals = int((y == 0).sum())
    target_fp = int(np.floor(float(args.target_fpr) * n_normals))

    candidates = build_candidates(
        df=df,
        label_col=label_col,
        action_col=action_col,
        min_incremental_fp=int(args.min_incremental_fp),
        max_candidates=int(args.max_candidates),
        enable_quads=bool(args.enable_quads),
    )

    if candidates.empty:
        selected = pd.DataFrame()
        frontier = pd.DataFrame()
        demote_mask = pd.Series(False, index=df.index)
    else:
        selected, frontier, demote_mask = select_greedy(
            df=df,
            label_col=label_col,
            action_col=action_col,
            candidates=candidates,
            base_fn=base_fn,
            max_total_fn=int(args.max_total_fn),
            target_fp=target_fp,
            max_rules=int(args.max_rules),
            min_incremental_fp=int(args.min_incremental_fp),
            continue_after_target=bool(args.continue_after_target),
        )

    df["exp014b_r4a_demote_confirm_to_approve"] = (
        normalize_action_series(df[action_col]).eq("CONFIRMAR") & demote_mask.fillna(False)
    ).astype(int)
    df["r4a_decisao_recommended"] = apply_demotions(df, action_col, demote_mask)
    df["exp014b_r4a_intervention_pred"] = action_to_intervention(df["r4a_decisao_recommended"])
    df["exp014b_r4a_block_pred"] = action_to_block(df["r4a_decisao_recommended"])

    final_intervention_metrics = metrics(y, df["exp014b_r4a_intervention_pred"])
    final_block_metrics = metrics(y, df["exp014b_r4a_block_pred"])

    block_unchanged = bool((safe_int_series(base_block) == safe_int_series(df["exp014b_r4a_block_pred"])).all())
    n_block_mismatches = int((safe_int_series(base_block) != safe_int_series(df["exp014b_r4a_block_pred"])).sum())

    fp_removed_total = int(base_intervention_metrics["fp"] - final_intervention_metrics["fp"])
    fn_added_total = int(final_intervention_metrics["fn"] - base_intervention_metrics["fn"])
    target_reached = bool(final_intervention_metrics["fp"] <= target_fp)
    gap_to_target_fp = max(0, int(final_intervention_metrics["fp"] - target_fp))

    final_by_action = metrics_by_action(df, label_col, "r4a_decisao_recommended")
    base_by_action = metrics_by_action(df.assign(_action=action), label_col, "_action")
    rob = robustness(df, label_col, action_col, "r4a_decisao_recommended")

    def action_val(table: pd.DataFrame, action_name: str, col: str) -> int:
        t = table[table["action"].eq(action_name)]
        return int(t.iloc[0][col]) if not t.empty else 0

    summary = {
        "experiment": EXPERIMENT,
        "status": "DONE",
        "objective_status": (
            "DONE_R4A_FPR15_TARGET_REACHED_WITHIN_FN_BUDGET_BLOCK_UNCHANGED"
            if target_reached and block_unchanged and final_intervention_metrics["fn"] <= int(args.max_total_fn)
            else "DONE_R4A_FPR15_TARGET_NOT_REACHED_BUT_IMPROVED"
            if fp_removed_total > 0 and block_unchanged and final_intervention_metrics["fn"] <= int(args.max_total_fn)
            else "DONE_R4A_NO_SAFE_IMPROVEMENT"
        ),
        "n_rows": int(len(df)),
        "n_frauds": int((y == 1).sum()),
        "n_normals": n_normals,
        "predictions_path": str(pred_path),
        "artifact_path": str(artifact_path) if artifact_path else None,
        "action_col": action_col,
        "block_col": block_col,
        "intervention_col": intervention_col,
        "baseline_intervention_metrics": base_intervention_metrics,
        "baseline_block_metrics": base_block_metrics,
        "final_intervention_metrics": final_intervention_metrics,
        "final_block_metrics": final_block_metrics,
        "target_fpr": float(args.target_fpr),
        "target_fp": target_fp,
        "target_reached": target_reached,
        "gap_to_target_fp": gap_to_target_fp,
        "fp_removed_total": fp_removed_total,
        "fn_added_total": fn_added_total,
        "max_total_fn": int(args.max_total_fn),
        "block_unchanged": block_unchanged,
        "n_block_mismatches": n_block_mismatches,
        "confirm_before_n": action_val(base_by_action, "CONFIRMAR", "n_rows"),
        "confirm_before_frauds": action_val(base_by_action, "CONFIRMAR", "n_frauds"),
        "confirm_before_normals": action_val(base_by_action, "CONFIRMAR", "n_normals"),
        "confirm_after_n": action_val(final_by_action, "CONFIRMAR", "n_rows"),
        "confirm_after_frauds": action_val(final_by_action, "CONFIRMAR", "n_frauds"),
        "confirm_after_normals": action_val(final_by_action, "CONFIRMAR", "n_normals"),
        "n_candidates_evaluated": int(len(candidates)),
        "n_selected_rules": int(len(selected)),
        "min_incremental_fp": int(args.min_incremental_fp),
        "max_rules": int(args.max_rules),
        "enable_quads": bool(args.enable_quads),
        "all_pass": bool(block_unchanged and final_intervention_metrics["fn"] <= int(args.max_total_fn)),
        "output_dir": str(out_dir),
    }

    contract = {
        "predictions_path": str(pred_path),
        "artifact_path": str(artifact_path) if artifact_path else None,
        "label_col": label_col,
        "action_col": action_col,
        "block_col": block_col,
        "intervention_col": intervention_col,
        "target_fpr": float(args.target_fpr),
        "target_fp": target_fp,
        "max_total_fn": int(args.max_total_fn),
        "min_incremental_fp": int(args.min_incremental_fp),
        "max_rules": int(args.max_rules),
        "enable_quads": bool(args.enable_quads),
        "contract_ok": True,
        "missing": [],
    }

    base_metrics = {
        "baseline_intervention_metrics": base_intervention_metrics,
        "baseline_block_metrics": base_block_metrics,
        "baseline_by_action": base_by_action.to_dict(orient="records"),
        "artifact_status": artifact.get("frozen_validation_status") if isinstance(artifact, dict) else None,
    }

    rec_artifact = {
        "experiment": EXPERIMENT,
        "input_predictions_path": str(pred_path),
        "base_action_col": action_col,
        "final_action_col": "r4a_decisao_recommended",
        "demote_col": "exp014b_r4a_demote_confirm_to_approve",
        "intervention_pred_col": "exp014b_r4a_intervention_pred",
        "block_pred_col": "exp014b_r4a_block_pred",
        "baseline_intervention_metrics": base_intervention_metrics,
        "baseline_block_metrics": base_block_metrics,
        "final_intervention_metrics": final_intervention_metrics,
        "final_block_metrics": final_block_metrics,
        "target_fpr": float(args.target_fpr),
        "target_fp": target_fp,
        "target_reached": target_reached,
        "gap_to_target_fp": gap_to_target_fp,
        "fp_removed_total": fp_removed_total,
        "fn_added_total": fn_added_total,
        "block_unchanged": block_unchanged,
        "selected_demotions": selected.to_dict(orient="records") if not selected.empty else [],
        "notes": [
            "Only CONFIRMAR residual rows are eligible for APROVAR demotion.",
            "BLOQUEAR is fixed.",
            "Default selection stops when FPR target is reached.",
            "Promotion requires R4A-FROZEN validation.",
        ],
    }

    write_json(out_dir / "00_run_summary.json", summary)
    write_json(out_dir / "01_input_contract.json", contract)
    write_json(out_dir / "02_base_metrics.json", base_metrics)
    candidates.to_csv(out_dir / "03_candidates.csv", index=False, encoding="utf-8")
    frontier.to_csv(out_dir / "04_selection_frontier.csv", index=False, encoding="utf-8")
    selected.to_csv(out_dir / "05_selected_demotions.csv", index=False, encoding="utf-8")
    final_by_action.to_csv(out_dir / "06_decision_metrics_by_action.csv", index=False, encoding="utf-8")
    rob.to_csv(out_dir / "07_robustness_by_segment.csv", index=False, encoding="utf-8")
    write_json(out_dir / "08_policy_artifact_recommended.json", rec_artifact)
    df.to_csv(out_dir / "09_predictions_recommended.csv", index=False, encoding="utf-8")

    selected_md = selected.to_markdown(index=False) if not selected.empty else "Nenhuma regra selecionada."
    frontier_md = frontier.to_markdown(index=False) if not frontier.empty else "Nenhuma seleção possível."
    by_action_md = final_by_action.to_markdown(index=False)

    report = f"""# {EXPERIMENT} - FPR 1.5 Pathfinder

## Resultado executivo
- Status: `{summary["objective_status"]}`
- All pass: `{summary["all_pass"]}`
- Target FPR: `{args.target_fpr}`
- Target FP: `{target_fp}`
- Target reached: `{target_reached}`
- Gap FP to target: `{gap_to_target_fp}`
- BLOQUEAR unchanged: `{block_unchanged}`
- FP removidos total: `{fp_removed_total}`
- FN adicionados: `{fn_added_total}`

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

## Fila CONFIRMAR
```text
Antes:  n={summary["confirm_before_n"]}, fraudes={summary["confirm_before_frauds"]}, normais={summary["confirm_before_normals"]}
Depois: n={summary["confirm_after_n"]}, fraudes={summary["confirm_after_frauds"]}, normais={summary["confirm_after_normals"]}
```

## Métricas por ação final
{by_action_md}

## Regras selecionadas
{selected_md}

## Frontier
{frontier_md}

## Decisão sugerida
Se `target_reached=true`, executar R4A-FROZEN.
Se não, avaliar o gap residual e decidir entre busca mais ampla (`--enable-quads`) ou criação de novas features.
"""
    (out_dir / "10_exp014b_r4a_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
