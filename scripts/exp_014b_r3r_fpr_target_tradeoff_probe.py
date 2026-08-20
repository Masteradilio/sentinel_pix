#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3R - R3Q-FROZEN consolidation + FPR target tradeoff probe

Goal:
  Consolidate EXP-014B-R3Q by frozen replay, then test whether FP can be
  reduced toward FPR <= 1% under controlled FN budgets.

Budgets:
  - scenario_fn0: FN must remain 0
  - scenario_fn1: FN <= 1
  - scenario_fn2: FN <= 2

Hard constraints:
  - no rescues
  - no threshold changes
  - no runtime calls
  - only demotion/veto rules over current alerts
  - do not use temporal_split/event_month/source/sample as rule predicates

Default input:
  resultados/experimentos/EXP-014B-R3Q/08_predictions_recommended.csv
  resultados/experimentos/EXP-014B-R3Q/07_policy_artifact_recommended.json

Outputs:
  resultados/experimentos/EXP-014B-R3R/
    00_run_summary.json
    01_input_contract.json
    02_r3q_frozen_validation.json
    03_target_scenarios.csv
    04_tradeoff_candidates.csv
    05_selected_rules_by_scenario.csv
    06_selection_frontier.csv
    07_robustness_by_segment.csv
    08_policy_artifact_recommended.json
    09_predictions_recommended.csv
    10_exp014b_r3r_report.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
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
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "dados").exists() else Path.cwd()

DEFAULT_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3Q" / "08_predictions_recommended.csv"
DEFAULT_ARTIFACT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3Q" / "07_policy_artifact_recommended.json"
DEFAULT_OUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3R"

BASE_COL = "exp014b_r3p_frozen_pred"
EXISTING_R3Q_COL = "exp014b_r3q_recommended_pred"
R3Q_FROZEN_COL = "exp014b_r3q_frozen_pred"
R3R_FINAL_COL = "exp014b_r3r_recommended_pred"

FEATURE_COLS = [
    "ds_tipo_chave_norm",
    "value_band",
    "mbk_available_flag",
    "first_receiver_flag_real",
    "periodo_dia",
    "module_quiet",
    "lgbm_bin",
    "if_bin",
    "score_bin",
    "ratio_bin",
    "qtd_rec_bin",
    "vl_bin",
    "valor_rec_bin",
]

ANCHOR_COLS = {
    "lgbm_bin",
    "if_bin",
    "score_bin",
    "ratio_bin",
    "qtd_rec_bin",
    "vl_bin",
    "value_band",
}

NUMERIC_COLS = [
    "lgbm_r4_score",
    "lgbm_mapped",
    "lgbm_raw",
    "score_final",
    "if_percentile",
    "vl_pix",
    "ratio_valor_media_pagador_90d",
    "ratio_valor_maximo_pagador_180d",
    "qtd_pix_recebidos_180d",
    "valor_total_recebido_180d",
    "se_score",
    "se_patterns_count",
    "beh_score",
    "beh_factors_count",
    "runtime_flagged",
]

SEGMENT_COLS = [
    "temporal_split",
    "event_month",
    "ds_tipo_chave_norm",
    "value_band",
    "mbk_available_flag",
    "periodo_dia",
    "sample_strategy",
    "source_dataset",
]

@dataclass
class Candidate:
    rule_id: str
    family: str
    description: str
    mask: np.ndarray
    tp_loss: int
    fp_removed: int
    n_removed: int
    combo_size: int
    n_temporal_splits_with_fp_removed: int
    n_months_with_fp_removed: int
    has_nontrain_support: bool
    has_validation_or_holdout_support: bool
    params: dict[str, Any]


def log(msg: str) -> None:
    print(msg, flush=True)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().split(".")[-1] for c in df.columns]
    if "transaction_id" not in df.columns and "cd_pix" in df.columns:
        df["transaction_id"] = df["cd_pix"]
    if "event_datetime" not in df.columns and "dt_pix" in df.columns:
        df["event_datetime"] = df["dt_pix"]
    if "is_fraud" not in df.columns:
        raise RuntimeError("Coluna obrigatoria ausente: is_fraud")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)
    if "event_datetime" in df.columns:
        dt = pd.to_datetime(df["event_datetime"], errors="coerce")
        if "event_month" not in df.columns:
            df["event_month"] = dt.dt.to_period("M").astype(str)
    for c in [BASE_COL, EXISTING_R3Q_COL, R3Q_FROZEN_COL, R3R_FINAL_COL]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    if "module_quiet" not in df.columns:
        se = num(df, ["se_score_x", "se_score_y", "se_score"], 0.0)
        sec = num(df, ["se_patterns_count", "se_pattern_count"], 0.0)
        beh = num(df, ["beh_score", "behavioral_score"], 0.0)
        behc = num(df, ["beh_factors_count", "behavioral_risk_factor_count"], 0.0)
        runtime = num(df, "runtime_flagged", 0.0)
        strong = (se >= 40) | (sec >= 2) | (beh >= 25) | (behc >= 2) | (runtime >= 1)
        df["module_quiet"] = np.where(strong, "module_strong", "module_quiet")
    return df.reset_index(drop=True)


def pick(df: pd.DataFrame, names: str | list[str]) -> str | None:
    if isinstance(names, str):
        names = [names]
    for n in names:
        if n in df.columns:
            return n
    return None


def num(df: pd.DataFrame, names: str | list[str], default: float = 0.0) -> pd.Series:
    c = pick(df, names)
    if c is None:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
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


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n) + (z * z / (4 * n * n))) ** 0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def parse_params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    return json.loads(str(raw).replace("Infinity", "1e999"))


def rule_mask(df: pd.DataFrame, current_pred: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    rtype = params.get("type")
    if rtype in {"numeric_headroom", "numeric_retighten", "numeric_threshold"}:
        c = params.get("col")
        if c not in df.columns:
            return np.zeros(len(df), dtype=bool)
        vals = num(df, c, 0.0).to_numpy(dtype=float)
        cut = float(params.get("cut"))
        direction = params.get("direction")
        mask = vals >= cut if direction == "ge" else vals <= cut
    elif rtype in {"combo_headroom", "combo_retighten", "combo"}:
        mask = np.ones(len(df), dtype=bool)
        for c, v in zip(params.get("combo_cols", []), params.get("combo_values", [])):
            if c not in df.columns:
                return np.zeros(len(df), dtype=bool)
            mask &= df[c].astype("string").fillna("<MISSING>").astype(str).to_numpy() == str(v)
    else:
        return np.zeros(len(df), dtype=bool)
    mask &= current_pred.astype(int) == 1
    return mask


def apply_rule_rows(df: pd.DataFrame, pred: np.ndarray, rules: list[dict[str, Any]], y: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    cur = pred.copy().astype(int)
    rows = []
    for i, rule in enumerate(rules):
        params = parse_params(rule.get("params_json") or rule.get("params") or "{}")
        mask = rule_mask(df, cur, params)
        tp_loss = int(((y == 1) & mask).sum())
        fp_removed = int(((y == 0) & mask).sum())
        cur[mask] = 0
        m = metrics(y, cur)
        rows.append({
            "rule_index": i,
            "rule_id": rule.get("rule_id"),
            "family": rule.get("family"),
            "description": rule.get("description"),
            "tp_loss": tp_loss,
            "fp_removed": fp_removed,
            "n_removed": int(mask.sum()),
            "params_json": json.dumps(params, ensure_ascii=False),
            **m,
        })
    return cur, pd.DataFrame(rows)


def validate_r3q_frozen(df: pd.DataFrame, artifact: dict[str, Any], output_dir: Path) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    y = df["is_fraud"].to_numpy(dtype=int)
    base = df[BASE_COL].to_numpy(dtype=int)
    rules = artifact.get("selected_fp_rules", [])
    frozen_pred, replay = apply_rule_rows(df, base, rules, y)
    base_m = metrics(y, base)
    frozen_m = metrics(y, frozen_pred)
    existing_m = metrics(y, df[EXISTING_R3Q_COL].to_numpy(dtype=int)) if EXISTING_R3Q_COL in df.columns else None
    mismatches = int((frozen_pred != df[EXISTING_R3Q_COL].to_numpy(dtype=int)).sum()) if EXISTING_R3Q_COL in df.columns else None
    exp_base = artifact.get("base_r3p_frozen_metrics", {})
    exp_final = artifact.get("recommended_metrics", {})
    wl, wh = wilson(frozen_m["tp"], int(y.sum()))
    validation = {
        "expected_base_metrics": exp_base,
        "actual_base_metrics": base_m,
        "expected_final_metrics": exp_final,
        "actual_frozen_metrics": frozen_m,
        "existing_r3q_metrics": existing_m,
        "prediction_mismatches_vs_existing": mismatches,
        "fp_removed_vs_r3p": int(base_m["fp"] - frozen_m["fp"]),
        "tp_loss_vs_r3p": int(base_m["tp"] - frozen_m["tp"]),
        "fn_delta_vs_r3p": int(frozen_m["fn"] - base_m["fn"]),
        "wilson_low": wl,
        "wilson_high": wh,
        "base_metrics_match_artifact": bool(all(base_m.get(k) == exp_base.get(k) for k in ["tp", "fp", "fn"])),
        "final_metrics_match_artifact": bool(all(frozen_m.get(k) == exp_final.get(k) for k in ["tp", "fp", "fn"])),
        "existing_prediction_match": bool(mismatches == 0) if mismatches is not None else None,
        "fn_zero_preserved": bool(frozen_m["fn"] == 0),
        "fp_reduced": bool(frozen_m["fp"] < base_m["fp"]),
        "rule_tp_loss_zero": bool((replay["tp_loss"].sum() if not replay.empty else 0) == 0),
    }
    validation["all_pass"] = bool(
        validation["base_metrics_match_artifact"]
        and validation["final_metrics_match_artifact"]
        and validation["fn_zero_preserved"]
        and validation["fp_reduced"]
        and validation["rule_tp_loss_zero"]
        and (validation["existing_prediction_match"] is not False)
    )
    validation["status"] = "PASS_R3Q_FROZEN_VALIDATED" if validation["all_pass"] else "FAIL_R3Q_FROZEN_DIVERGENCE"
    dump_json(validation, output_dir / "02_r3q_frozen_validation.json")
    replay.to_csv(output_dir / "02b_r3q_frozen_rule_replay.csv", index=False)
    return frozen_pred, validation, replay


def support_stats(df: pd.DataFrame, mask: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    fp_mask = mask & (y == 0)
    if fp_mask.sum() == 0:
        return {
            "n_temporal_splits_with_fp_removed": 0,
            "n_months_with_fp_removed": 0,
            "has_nontrain_support": False,
            "has_validation_or_holdout_support": False,
        }
    splits = set()
    if "temporal_split" in df.columns:
        splits = set(df.loc[fp_mask, "temporal_split"].astype(str).dropna().unique().tolist())
    months = set()
    if "event_month" in df.columns:
        months = set(df.loc[fp_mask, "event_month"].astype(str).dropna().unique().tolist())
    return {
        "n_temporal_splits_with_fp_removed": len(splits),
        "n_months_with_fp_removed": len(months),
        "has_nontrain_support": bool(any(s in {"VALIDATION", "HOLDOUT"} for s in splits)),
        "has_validation_or_holdout_support": bool(any(s in {"VALIDATION", "HOLDOUT"} for s in splits)),
    }


def add_candidate(
    out: list[Candidate],
    df: pd.DataFrame,
    y: np.ndarray,
    family: str,
    desc: str,
    mask: np.ndarray,
    min_fp_removed: int,
    max_candidate_tp_loss: int,
    combo_size: int,
    params: dict[str, Any],
    require_nontrain_support: bool,
    require_holdout_or_validation_support: bool,
) -> None:
    if not mask.any():
        return
    tp_loss = int(((y == 1) & mask).sum())
    fp_removed = int(((y == 0) & mask).sum())
    if tp_loss > max_candidate_tp_loss or fp_removed < min_fp_removed:
        return
    sup = support_stats(df, mask, y)
    if require_nontrain_support and not sup["has_nontrain_support"]:
        return
    if require_holdout_or_validation_support and not sup["has_validation_or_holdout_support"]:
        return
    out.append(Candidate(
        rule_id=f"r3r_tmp_{len(out):05d}",
        family=family,
        description=desc,
        mask=mask,
        tp_loss=tp_loss,
        fp_removed=fp_removed,
        n_removed=int(mask.sum()),
        combo_size=combo_size,
        params=params,
        **sup,
    ))


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in FEATURE_COLS:
        if c in df.columns:
            out[c] = df[c].astype("string").fillna("<MISSING>").astype(str)
    return out


def mine_tradeoff_candidates(
    df: pd.DataFrame,
    pred: np.ndarray,
    min_fp_removed: int,
    max_candidate_tp_loss: int,
    max_combo_size: int,
    top_groups_per_combo: int,
    require_nontrain_support: bool,
    require_holdout_or_validation_support: bool,
    max_seconds: int,
) -> list[Candidate]:
    y = df["is_fraud"].to_numpy(dtype=int)
    alerted = pred.astype(int) == 1
    out: list[Candidate] = []
    t0 = time.perf_counter()

    # Numeric demotion candidates.
    for c in NUMERIC_COLS:
        if time.perf_counter() - t0 >= max_seconds:
            break
        if c not in df.columns:
            continue
        vals = num(df, c, 0.0).to_numpy(dtype=float)
        active = vals[alerted]
        if len(active) == 0:
            continue
        qs = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
        try:
            cuts = sorted(set(float(x) for x in np.quantile(active, qs) if np.isfinite(x)))
        except Exception:
            cuts = []
        for cut in cuts:
            for direction in ["le", "ge"]:
                mask = alerted & ((vals <= cut) if direction == "le" else (vals >= cut))
                op = "<=" if direction == "le" else ">="
                desc = f"alert AND {c}{op}{cut:g}"
                add_candidate(
                    out, df, y, "r3r_numeric_tradeoff", desc, mask,
                    min_fp_removed, max_candidate_tp_loss, 1,
                    {"type": "numeric_headroom", "col": c, "direction": direction, "cut": cut},
                    require_nontrain_support, require_holdout_or_validation_support,
                )

    # Categorical deployable combos. Do not use event_month/temporal_split/source/sample as predicates.
    feat = feature_frame(df)
    cols = list(feat.columns)
    idx = np.where(alerted)[0]
    for r in range(1, max_combo_size + 1):
        if time.perf_counter() - t0 >= max_seconds:
            break
        for combo in itertools.combinations(cols, r):
            if time.perf_counter() - t0 >= max_seconds:
                break
            combo = list(combo)
            if r == 1 and combo[0] not in ANCHOR_COLS.union({"ds_tipo_chave_norm", "module_quiet"}):
                continue
            if r >= 2 and not any(c in ANCHOR_COLS for c in combo):
                continue
            if r >= 3 and not any(c in {"lgbm_bin", "if_bin", "score_bin", "ratio_bin", "qtd_rec_bin", "vl_bin"} for c in combo):
                continue
            sub = feat.iloc[idx][combo]
            if sub.empty:
                continue
            groups = []
            for key, rel in sub.groupby(combo, dropna=False).indices.items():
                rows = sub.iloc[list(rel)].index.to_numpy(dtype=int)
                if len(rows) < min_fp_removed:
                    continue
                mask = np.zeros(len(df), dtype=bool)
                mask[rows] = True
                mask &= alerted
                tp_loss = int(((y == 1) & mask).sum())
                fp_removed = int(((y == 0) & mask).sum())
                if tp_loss <= max_candidate_tp_loss and fp_removed >= min_fp_removed:
                    groups.append((-fp_removed, tp_loss, key, mask))
            groups.sort()
            for _, _, key, mask in groups[:top_groups_per_combo]:
                vals = key if isinstance(key, tuple) else (key,)
                vals = [str(v) for v in vals]
                desc = "alert AND " + " AND ".join([f"{c}={v}" for c, v in zip(combo, vals)])
                add_candidate(
                    out, df, y, "r3r_combo_tradeoff", desc, mask,
                    min_fp_removed, max_candidate_tp_loss, r,
                    {"type": "combo_headroom", "combo_cols": combo, "combo_values": vals},
                    require_nontrain_support, require_holdout_or_validation_support,
                )

    # Deduplicate by exact affected rows.
    best: dict[bytes, Candidate] = {}
    for c in out:
        key = np.packbits(c.mask).tobytes()
        old = best.get(key)
        if old is None or (c.fp_removed, -c.tp_loss, -c.combo_size, -len(c.description)) > (old.fp_removed, -old.tp_loss, -old.combo_size, -len(old.description)):
            best[key] = c
    out = list(best.values())
    out.sort(key=lambda c: (-c.fp_removed, c.tp_loss, c.combo_size, len(c.description)))
    for i, c in enumerate(out):
        c.rule_id = f"r3r_cand_{i:05d}"
    return out


def candidates_to_df(cands: list[Candidate], scenario: str = "") -> pd.DataFrame:
    return pd.DataFrame([{
        "scenario": scenario,
        "candidate_index": i,
        "rule_id": c.rule_id,
        "family": c.family,
        "description": c.description,
        "tp_loss": c.tp_loss,
        "fp_removed": c.fp_removed,
        "n_removed": c.n_removed,
        "combo_size": c.combo_size,
        "n_temporal_splits_with_fp_removed": c.n_temporal_splits_with_fp_removed,
        "n_months_with_fp_removed": c.n_months_with_fp_removed,
        "has_nontrain_support": c.has_nontrain_support,
        "has_validation_or_holdout_support": c.has_validation_or_holdout_support,
        "params_json": json.dumps(c.params, ensure_ascii=False),
    } for i, c in enumerate(cands)])


def select_for_budget(
    cands: list[Candidate],
    df: pd.DataFrame,
    base_pred: np.ndarray,
    fn_budget: int,
    max_rules: int,
    target_fp: int,
    tp_loss_penalty: float,
) -> tuple[np.ndarray, list[Candidate], pd.DataFrame, str]:
    y = df["is_fraud"].to_numpy(dtype=int)
    cur = base_pred.copy().astype(int)
    selected: list[Candidate] = []
    used: set[int] = set()
    frontier = []
    stop_reason = "completed"
    base_m = metrics(y, cur)
    cumulative_tp_loss = 0

    for depth in range(1, max_rules + 1):
        current_alerted = cur.astype(int) == 1
        best = None
        for i, cand in enumerate(cands[:5000]):
            if i in used:
                continue
            mask = cand.mask & current_alerted
            if not mask.any():
                continue
            tp_loss = int(((y == 1) & mask).sum())
            fp_removed = int(((y == 0) & mask).sum())
            if fp_removed <= 0:
                continue
            if cumulative_tp_loss + tp_loss > fn_budget:
                continue
            # Prefer high FP, but charge a TP-loss penalty so FN budget is used only when worthwhile.
            score = fp_removed - tp_loss_penalty * tp_loss
            rank = (score, fp_removed, -tp_loss, -cand.combo_size, -len(cand.description))
            if best is None or rank > best[0]:
                best = (rank, i, cand, mask, tp_loss, fp_removed)
        if best is None:
            stop_reason = f"no_more_candidates_at_depth_{depth}"
            break
        _, i, cand, mask, tp_loss, fp_removed = best
        cur[mask] = 0
        used.add(i)
        cumulative_tp_loss += tp_loss
        chosen = Candidate(
            rule_id=cand.rule_id,
            family=cand.family,
            description=cand.description,
            mask=mask.copy(),
            tp_loss=tp_loss,
            fp_removed=fp_removed,
            n_removed=int(mask.sum()),
            combo_size=cand.combo_size,
            n_temporal_splits_with_fp_removed=support_stats(df, mask, y)["n_temporal_splits_with_fp_removed"],
            n_months_with_fp_removed=support_stats(df, mask, y)["n_months_with_fp_removed"],
            has_nontrain_support=support_stats(df, mask, y)["has_nontrain_support"],
            has_validation_or_holdout_support=support_stats(df, mask, y)["has_validation_or_holdout_support"],
            params=cand.params,
        )
        selected.append(chosen)
        m = metrics(y, cur)
        frontier.append({
            "scenario": f"fn_budget_{fn_budget}",
            "depth": depth,
            "rule_id": chosen.rule_id,
            "description": chosen.description,
            "marginal_fp_removed": fp_removed,
            "marginal_tp_loss": tp_loss,
            "cumulative_fp_removed": int(base_m["fp"] - m["fp"]),
            "cumulative_tp_loss": int(base_m["tp"] - m["tp"]),
            "target_fp": int(target_fp),
            "target_gap_fp": int(max(m["fp"] - target_fp, 0)),
            "target_fpr_reached": bool(m["fp"] <= target_fp),
            **m,
        })
        if m["fp"] <= target_fp:
            stop_reason = f"target_fpr_reached_at_depth_{depth}"
            break
    if not frontier:
        m = metrics(y, cur)
        frontier.append({
            "scenario": f"fn_budget_{fn_budget}",
            "depth": 0,
            "rule_id": "",
            "description": "",
            "marginal_fp_removed": 0,
            "marginal_tp_loss": 0,
            "cumulative_fp_removed": 0,
            "cumulative_tp_loss": 0,
            "target_fp": int(target_fp),
            "target_gap_fp": int(max(m["fp"] - target_fp, 0)),
            "target_fpr_reached": bool(m["fp"] <= target_fp),
            **m,
        })
    return cur, selected, pd.DataFrame(frontier), stop_reason


def robustness_by_segment(df: pd.DataFrame, base_pred: np.ndarray, final_pred: np.ndarray) -> pd.DataFrame:
    y = df["is_fraud"].to_numpy(dtype=int)
    rows = []
    for col in SEGMENT_COLS:
        if col not in df.columns:
            continue
        vals = df[col].astype("string").fillna("<MISSING>").astype(str)
        for v in sorted(vals.unique()):
            mask = vals.to_numpy() == v
            if mask.sum() == 0:
                continue
            b = metrics(y[mask], base_pred[mask])
            f = metrics(y[mask], final_pred[mask])
            rows.append({
                "segment_col": col,
                "segment_value": v,
                "n_rows": int(mask.sum()),
                "n_frauds": int(y[mask].sum()),
                "fp_removed": int(b["fp"] - f["fp"]),
                "tp_loss": int(b["tp"] - f["tp"]),
                "fn_delta": int(f["fn"] - b["fn"]),
                "final_tp": f["tp"],
                "final_fp": f["fp"],
                "final_fn": f["fn"],
                "final_recall": f["recall"],
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["tp_loss", "fn_delta", "fp_removed"], ascending=[False, False, False]).reset_index(drop=True)


def make_report(summary: dict[str, Any], validation: dict[str, Any], scenarios: pd.DataFrame, frontier: pd.DataFrame, selected: pd.DataFrame, target_fp: int) -> str:
    lines = []
    lines.append("# EXP-014B-R3R - FPR target tradeoff probe")
    lines.append("")
    lines.append("## Resultado executivo")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- All pass: `{summary['all_pass']}`")
    lines.append(f"- Base R3Q-FROZEN: `{summary['base_r3q_frozen_metrics']}`")
    lines.append(f"- Cenario recomendado: `{summary['recommended_scenario']}`")
    lines.append(f"- Metricas recomendadas: `{summary['recommended_metrics']}`")
    lines.append(f"- FP removidos vs R3Q: `{summary['fp_removed_vs_r3q']}`")
    lines.append(f"- FN delta vs R3Q: `{summary['fn_delta_vs_r3q']}`")
    lines.append(f"- FPR alvo: `{summary['target_fpr']}` | FP max alvo: `{target_fp}`")
    lines.append(f"- Gap ate FPR alvo: `{summary['target_gap_fp']}` FP")
    lines.append("")
    lines.append("## Validacao R3Q congelada")
    lines.append("```json")
    lines.append(json.dumps(validation, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Cenarios por orcamento de FN")
    if scenarios.empty:
        lines.append("Nenhum cenario gerado.")
    else:
        show = ["scenario", "fn_budget", "n_selected_rules", "tp", "fp", "fn", "precision", "recall", "fpr", "fp_removed_vs_r3q", "fn_delta_vs_r3q", "target_fpr_reached", "target_gap_fp", "stop_reason"]
        lines.append(scenarios[[c for c in show if c in scenarios.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Fronteira de selecao")
    if frontier.empty:
        lines.append("Fronteira vazia.")
    else:
        show = ["scenario", "depth", "marginal_fp_removed", "marginal_tp_loss", "cumulative_fp_removed", "cumulative_tp_loss", "tp", "fp", "fn", "fpr", "target_gap_fp", "description"]
        lines.append(frontier[[c for c in show if c in frontier.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Regras selecionadas")
    if selected.empty:
        lines.append("Nenhuma regra selecionada.")
    else:
        show = ["scenario", "rule_id", "description", "fp_removed", "tp_loss", "combo_size", "n_temporal_splits_with_fp_removed", "n_months_with_fp_removed"]
        lines.append(selected[[c for c in show if c in selected.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Decisao sugerida")
    if summary.get("target_fpr_reached"):
        lines.append("O alvo FPR <= 1% foi atingido dentro do orcamento de FN. Proximo passo: frozen validation do cenario recomendado.")
    elif summary.get("recommended_metrics", {}).get("fn", 99) <= 2 and summary.get("fp_removed_vs_r3q", 0) >= 250:
        lines.append("Houve reducao material, mas o alvo FPR <= 1% ainda nao foi atingido. Proximo passo: congelar apenas se a reducao for operacionalmente relevante; caso contrario, migrar para segundo estagio/modelo ranker.")
    else:
        lines.append("O probe nao chegou perto do alvo FPR <= 1%. Isso indica que regras tabulares simples provavelmente atingiram limite pratico; proxima etapa deve ser segundo estagio/ranker ou novas features, nao mais microvetos.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--fn-budgets", default="0,1,2")
    ap.add_argument("--min-fp-removed", type=int, default=20)
    ap.add_argument("--max-candidate-tp-loss", type=int, default=2)
    ap.add_argument("--max-combo-size", type=int, default=4)
    ap.add_argument("--top-groups-per-combo", type=int, default=120)
    ap.add_argument("--max-rules", type=int, default=50)
    ap.add_argument("--max-seconds", type=int, default=240)
    ap.add_argument("--tp-loss-penalty", type=float, default=25.0)
    ap.add_argument("--require-nontrain-support", action="store_true")
    ap.add_argument("--require-holdout-or-validation-support", action="store_true")
    ap.add_argument("--no-write-predictions", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)
    artifact_path = Path(args.artifact)
    if not input_path.exists():
        raise FileNotFoundError(f"input nao encontrado: {input_path}")
    if not artifact_path.exists():
        raise FileNotFoundError(f"artifact nao encontrado: {artifact_path}")

    log("=" * 80)
    log("EXP-014B-R3R - R3Q-FROZEN consolidation + FPR target tradeoff probe")
    log("=" * 80)

    df = normalize(pd.read_csv(input_path, low_memory=False))
    artifact = load_json(artifact_path)
    missing = []
    for c in ["is_fraud", BASE_COL]:
        if c not in df.columns:
            missing.append(c)
    if "selected_fp_rules" not in artifact:
        missing.append("artifact.selected_fp_rules")
    contract = {
        "n_rows": int(len(df)),
        "n_frauds": int(df["is_fraud"].sum()) if "is_fraud" in df.columns else None,
        "input_path": str(input_path),
        "artifact_path": str(artifact_path),
        "base_col": BASE_COL,
        "existing_r3q_col_present": bool(EXISTING_R3Q_COL in df.columns),
        "n_selected_rules_in_artifact": int(len(artifact.get("selected_fp_rules", []))),
        "missing": missing,
        "contract_ok": not missing,
        "constraints": {
            "target_fpr": args.target_fpr,
            "fn_budgets": args.fn_budgets,
            "min_fp_removed": args.min_fp_removed,
            "max_candidate_tp_loss": args.max_candidate_tp_loss,
            "max_combo_size": args.max_combo_size,
            "max_rules": args.max_rules,
            "tp_loss_penalty": args.tp_loss_penalty,
            "require_nontrain_support": args.require_nontrain_support,
            "require_holdout_or_validation_support": args.require_holdout_or_validation_support,
            "max_seconds": args.max_seconds,
        },
    }
    dump_json(contract, out / "01_input_contract.json")
    if missing:
        raise RuntimeError(f"Contrato falhou: {missing}")

    y = df["is_fraud"].to_numpy(dtype=int)
    n_neg = int((y == 0).sum())
    target_fp = int(math.floor(args.target_fpr * n_neg))

    log("[A] Validando R3Q congelado por replay...")
    r3q_pred, validation, replay = validate_r3q_frozen(df, artifact, out)
    df[R3Q_FROZEN_COL] = r3q_pred.astype(int)
    base_m = metrics(y, r3q_pred)
    log(f"R3Q frozen metrics: {base_m}")
    log(f"FPR target {args.target_fpr:.4f}: max FP={target_fp}; current gap={max(base_m['fp'] - target_fp, 0)}")

    log("[B] Minerando candidatos tradeoff FN<=2...")
    cands = mine_tradeoff_candidates(
        df=df,
        pred=r3q_pred,
        min_fp_removed=args.min_fp_removed,
        max_candidate_tp_loss=args.max_candidate_tp_loss,
        max_combo_size=args.max_combo_size,
        top_groups_per_combo=args.top_groups_per_combo,
        require_nontrain_support=args.require_nontrain_support,
        require_holdout_or_validation_support=args.require_holdout_or_validation_support,
        max_seconds=args.max_seconds,
    )
    cand_df = candidates_to_df(cands)
    cand_df.to_csv(out / "04_tradeoff_candidates.csv", index=False)
    log(f"Candidatos tradeoff: {len(cands)}")

    scenario_rows = []
    selected_rows = []
    frontier_rows = []
    pred_by_scenario: dict[str, np.ndarray] = {}
    rules_by_scenario: dict[str, list[Candidate]] = {}
    fn_budgets = [int(x.strip()) for x in str(args.fn_budgets).split(",") if x.strip()]

    for budget in fn_budgets:
        pred, selected, frontier, stop = select_for_budget(
            cands=cands,
            df=df,
            base_pred=r3q_pred,
            fn_budget=budget,
            max_rules=args.max_rules,
            target_fp=target_fp,
            tp_loss_penalty=args.tp_loss_penalty,
        )
        scenario = f"r3r_fn_budget_{budget}"
        pred_by_scenario[scenario] = pred
        rules_by_scenario[scenario] = selected
        m = metrics(y, pred)
        scenario_rows.append({
            "scenario": scenario,
            "fn_budget": int(budget),
            "n_selected_rules": int(len(selected)),
            "fp_removed_vs_r3q": int(base_m["fp"] - m["fp"]),
            "tp_loss_vs_r3q": int(base_m["tp"] - m["tp"]),
            "fn_delta_vs_r3q": int(m["fn"] - base_m["fn"]),
            "target_fp": int(target_fp),
            "target_fpr": float(args.target_fpr),
            "target_fpr_reached": bool(m["fp"] <= target_fp),
            "target_gap_fp": int(max(m["fp"] - target_fp, 0)),
            "stop_reason": stop,
            **m,
        })
        if not frontier.empty:
            frontier_rows.append(frontier)
        if selected:
            sdf = candidates_to_df(selected, scenario=scenario)
            selected_rows.append(sdf)
        log(f"  {scenario}: {m} | removed={base_m['fp'] - m['fp']} | gap_target={max(m['fp'] - target_fp, 0)} | stop={stop}")

    scenarios = pd.DataFrame(scenario_rows)
    scenarios.to_csv(out / "03_target_scenarios.csv", index=False)
    frontier_df = pd.concat(frontier_rows, ignore_index=True) if frontier_rows else pd.DataFrame()
    frontier_df.to_csv(out / "06_selection_frontier.csv", index=False)
    selected_df = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    selected_df.to_csv(out / "05_selected_rules_by_scenario.csv", index=False)

    if scenarios.empty:
        raise RuntimeError("Nenhum cenario gerado")
    # Recommendation: first any scenario that reaches target, then lowest FP under FN<=2, then best FP reduction.
    target_hit = scenarios[scenarios["target_fpr_reached"] == True].copy()
    if not target_hit.empty:
        rec_row = target_hit.sort_values(["fn", "fp", "n_selected_rules"], ascending=[True, True, True]).iloc[0]
    else:
        rec_row = scenarios.sort_values(["fp", "fn", "n_selected_rules"], ascending=[True, True, True]).iloc[0]
    rec_scenario = str(rec_row["scenario"])
    rec_pred = pred_by_scenario[rec_scenario]
    rec_rules = rules_by_scenario.get(rec_scenario, [])
    rec_m = metrics(y, rec_pred)
    robustness = robustness_by_segment(df, r3q_pred, rec_pred)
    robustness.to_csv(out / "07_robustness_by_segment.csv", index=False)

    wl, wh = wilson(rec_m["tp"], int(y.sum()))
    df[R3R_FINAL_COL] = rec_pred.astype(int)
    df["exp014b_r3r_recommended_scenario"] = rec_scenario

    target_reached = bool(rec_m["fp"] <= target_fp)
    fp_removed = int(base_m["fp"] - rec_m["fp"])
    fn_delta = int(rec_m["fn"] - base_m["fn"])
    if target_reached:
        status = "DONE_R3Q_FROZEN_BASE_VALIDATED_FPR_TARGET_REACHED"
    elif fn_delta <= 2 and fp_removed >= 250:
        status = "DONE_R3Q_FROZEN_BASE_VALIDATED_FP_REDUCED_MATERIAL_BUT_FPR_TARGET_NOT_REACHED"
    elif fn_delta <= 2 and fp_removed > 0:
        status = "DONE_R3Q_FROZEN_BASE_VALIDATED_FP_REDUCED_MICRO_BUT_FPR_TARGET_NOT_REACHED"
    else:
        status = "DONE_R3Q_FROZEN_BASE_VALIDATED_NO_ACCEPTABLE_TARGET_PROGRESS"
    status += "_FN_BUDGET_OK" if fn_delta <= 2 else "_FN_BUDGET_BROKEN"

    artifact_out = {
        "experiment": "EXP-014B-R3R",
        "policy_name": "r3r_fpr_target_tradeoff_probe",
        "objective_status": status,
        "source_artifact": str(artifact_path),
        "input_path": str(input_path),
        "base_col": R3Q_FROZEN_COL,
        "final_pred_col": R3R_FINAL_COL,
        "target_fpr": float(args.target_fpr),
        "target_fp": int(target_fp),
        "base_r3q_frozen_metrics": base_m,
        "recommended_scenario": rec_scenario,
        "recommended_metrics": rec_m,
        "fp_removed_vs_r3q": fp_removed,
        "tp_loss_vs_r3q": int(base_m["tp"] - rec_m["tp"]),
        "fn_delta_vs_r3q": fn_delta,
        "target_fpr_reached": target_reached,
        "target_gap_fp": int(max(rec_m["fp"] - target_fp, 0)),
        "wilson_low": wl,
        "wilson_high": wh,
        "r3q_frozen_validation": validation,
        "scenario_metrics": scenarios.to_dict(orient="records"),
        "selected_fp_rules": candidates_to_df(rec_rules, scenario=rec_scenario).to_dict(orient="records") if rec_rules else [],
        "constraints": contract["constraints"],
        "notes": [
            "This probe allows controlled FN budgets up to 2 to test whether FPR <= 1% is reachable with demotion rules.",
            "No rescues, no threshold changes, no runtime calls.",
            "If target is not reached, consider second-stage/ranker or new features instead of continuing micro-veto mining.",
        ],
    }
    dump_json(artifact_out, out / "08_policy_artifact_recommended.json")

    if not args.no_write_predictions:
        df.to_csv(out / "09_predictions_recommended.csv", index=False)

    all_pass = bool(validation["all_pass"] and fn_delta <= 2 and rec_m["fp"] < base_m["fp"])
    summary = {
        "experiment": "EXP-014B-R3R",
        "status": "DONE",
        "objective_status": status,
        "n_rows": int(len(df)),
        "n_frauds": int(y.sum()),
        "n_normals": int(n_neg),
        "input_path": str(input_path),
        "artifact_path": str(artifact_path),
        "base_validation_status": validation["status"],
        "base_r3q_frozen_metrics": base_m,
        "target_fpr": float(args.target_fpr),
        "target_fp": int(target_fp),
        "target_gap_fp_from_base": int(max(base_m["fp"] - target_fp, 0)),
        "recommended_scenario": rec_scenario,
        "recommended_metrics": rec_m,
        "fp_removed_vs_r3q": fp_removed,
        "tp_loss_vs_r3q": int(base_m["tp"] - rec_m["tp"]),
        "fn_delta_vs_r3q": fn_delta,
        "target_fpr_reached": target_reached,
        "target_gap_fp": int(max(rec_m["fp"] - target_fp, 0)),
        "n_tradeoff_candidates": int(len(cands)),
        "n_selected_rules": int(len(rec_rules)),
        "wilson_low": wl,
        "wilson_high": wh,
        "all_pass": all_pass,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(out),
    }
    dump_json(summary, out / "00_run_summary.json")

    report = make_report(summary, validation, scenarios, frontier_df, selected_df, target_fp)
    (out / "10_exp014b_r3r_report.md").write_text(report, encoding="utf-8")

    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        out / "00_run_summary.json",
        out / "01_input_contract.json",
        out / "02_r3q_frozen_validation.json",
        out / "03_target_scenarios.csv",
        out / "04_tradeoff_candidates.csv",
        out / "05_selected_rules_by_scenario.csv",
        out / "06_selection_frontier.csv",
        out / "07_robustness_by_segment.csv",
        out / "08_policy_artifact_recommended.json",
        out / "10_exp014b_r3r_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
