# -*- coding: utf-8 -*-
"""
EXP-014B-R3U - Receiver / Relationship Trust Feature Probe

Objetivo:
- Diagnosticar se features de confianca do recebedor e do relacionamento pagador-recebedor
  explicam parte relevante dos falsos positivos do benchmark R3Q.
- Testar politicas de democao baseadas em sinais de confianca, sem promover nada sem frozen.

Padrao:
- Entrada default: resultados/experimentos/EXP-014B-R3S/08_predictions_recommended.csv
  com fallback para R3R/R3Q/R3P.
- Base default: exp014b_r3q_frozen_pred, com fallback para exp014b_r3q_recommended_pred.
- Meta comercial atual: FN <= 5, recall >= 95%, FPR <= 1.5%.

Saidas:
  resultados/experimentos/EXP-014B-R3U/
    00_run_summary.json
    01_input_contract.json
    02_base_metrics.json
    03_trust_feature_diagnostics.csv
    04_trust_demote_candidates.csv
    05_selected_policy.json
    06_policy_frontier.csv
    07_robustness_by_segment.csv
    08_policy_artifact_recommended.json
    09_predictions_recommended.csv
    10_exp014b_r3u_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


EXP_NAME = "EXP-014B-R3U"
OUT_COL = "exp014b_r3u_recommended_pred"
SCORE_COL = "exp014b_r3u_receiver_relationship_trust_score"

DEFAULT_TARGET_FPR = 0.015
DEFAULT_MAX_FN = 5
DEFAULT_MIN_RECALL = 0.95
DEFAULT_MAX_RULES = 10
DEFAULT_MIN_FP_REMOVED = 10


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if math.isnan(float(obj)) or math.isinf(float(obj)):
            return None
        return float(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if pd.isna(obj) if not isinstance(obj, (list, dict, tuple)) else False:
        return None
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(json_safe(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def find_input_path(repo_root: Path) -> Optional[Path]:
    candidates = [
        repo_root / "resultados" / "experimentos" / "EXP-014B-R3S" / "08_predictions_recommended.csv",
        repo_root / "resultados" / "experimentos" / "EXP-014B-R3R" / "09_predictions_recommended.csv",
        repo_root / "resultados" / "experimentos" / "EXP-014B-R3Q" / "08_predictions_recommended.csv",
        repo_root / "resultados" / "experimentos" / "EXP-014B-R3P-FROZEN" / "08_predictions_frozen.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def choose_base_col(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    if requested and requested in df.columns:
        return requested
    candidates = [
        "exp014b_r3q_frozen_pred",
        "exp014b_r3q_recommended_pred",
        "exp014b_r3p_frozen_pred",
        "exp014b_r3p_recommended_pred",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    pred_cols = [c for c in df.columns if c.startswith("exp014b_") and c.endswith("_pred")]
    return pred_cols[-1] if pred_cols else None


def bin_to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def cat(df: pd.DataFrame, col: str, default: str = "<MISSING>") -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="object")
    return df[col].astype("object").where(df[col].notna(), default).astype(str)


def metrics_from_pred(y: pd.Series, pred: pd.Series) -> Dict[str, Any]:
    yv = bin_to_int(y)
    pv = bin_to_int(pred)
    tp = int(((pv == 1) & (yv == 1)).sum())
    fp = int(((pv == 1) & (yv == 0)).sum())
    fn = int(((pv == 0) & (yv == 1)).sum())
    tn = int(((pv == 0) & (yv == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
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


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def rank_auc(y_true: pd.Series, score: pd.Series) -> float:
    y = bin_to_int(y_true)
    s = pd.to_numeric(score, errors="coerce")
    mask = s.notna()
    y = y[mask]
    s = s[mask]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = s.rank(method="average")
    sum_pos = float(ranks[y == 1].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def add_trust_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    q_same_30 = num(out, "qtd_pix_mesmo_recebedor_30d").fillna(0)
    q_same_90 = num(out, "qtd_pix_mesmo_recebedor_90d").fillna(0)
    q_same_180 = num(out, "qtd_pix_mesmo_recebedor_180d").fillna(0)
    v_same_30 = num(out, "valor_total_para_recebedor_30d").fillna(0)
    v_same_90 = num(out, "valor_total_para_recebedor_90d").fillna(0)
    v_same_180 = num(out, "valor_total_para_recebedor_180d").fillna(0)
    days_rel = num(out, "dias_desde_primeiro_envio_recebedor").fillna(0)

    q_rec_30 = num(out, "qtd_pix_recebidos_30d").fillna(0)
    q_rec_90 = num(out, "qtd_pix_recebidos_90d").fillna(0)
    q_rec_180 = num(out, "qtd_pix_recebidos_180d").fillna(0)
    v_rec_30 = num(out, "valor_total_recebido_30d").fillna(0)
    v_rec_90 = num(out, "valor_total_recebido_90d").fillna(0)
    v_rec_180 = num(out, "valor_total_recebido_180d").fillna(0)
    distinct_payers = num(out, "soma_pagadores_distintos_dia_recebedor_180d").fillna(0)
    max_rec_day = num(out, "max_qtd_pix_recebidos_dia_180d").fillna(0)

    qtd_rec_bin = cat(out, "qtd_rec_bin")
    valor_rec_bin = cat(out, "valor_rec_bin")
    first_receiver = num(out, "first_receiver_flag_real").fillna(num(out, "first_receiver_flag").fillna(0))
    mbk_flag = num(out, "mbk_available_flag").fillna(0)
    mbk_comp = num(out, "mbk_completeness_score").fillna(0)
    module_quiet = cat(out, "module_quiet")
    se_worst = cat(out, "se_worst_pattern")
    ratio_bin = cat(out, "ratio_bin")

    missing_receiver_history = (
        qtd_rec_bin.eq("qtdrec_LT_0")
        | valor_rec_bin.eq("valrec_LT_0")
        | ((q_rec_180 <= 0) & (v_rec_180 <= 0) & (distinct_payers <= 0))
    )

    relationship_known = (q_same_180 > 0) | (v_same_180 > 0) | (days_rel > 0)
    relationship_recurrent = (q_same_180 >= 2) | (q_same_90 >= 2) | (v_same_180 >= 500) | (days_rel >= 30)
    relationship_strong = (q_same_180 >= 5) | (v_same_180 >= 2000) | (days_rel >= 90)

    receiver_known = (q_rec_180 > 0) | (v_rec_180 > 0) | (distinct_payers > 0)
    receiver_reputable = (q_rec_180 >= 10) | (v_rec_180 >= 5000) | (distinct_payers >= 5)
    receiver_strong = (q_rec_180 >= 30) | (v_rec_180 >= 25000) | (distinct_payers >= 10) | (max_rec_day >= 5)

    out["r3u_missing_receiver_history_flag"] = missing_receiver_history.astype(int)
    out["r3u_receiver_known_flag"] = receiver_known.astype(int)
    out["r3u_receiver_reputable_flag"] = receiver_reputable.astype(int)
    out["r3u_receiver_strong_flag"] = receiver_strong.astype(int)
    out["r3u_relationship_known_flag"] = relationship_known.astype(int)
    out["r3u_relationship_recurrent_flag"] = relationship_recurrent.astype(int)
    out["r3u_relationship_strong_flag"] = relationship_strong.astype(int)
    out["r3u_first_receiver_flag"] = (first_receiver == 1).astype(int)
    out["r3u_module_quiet_flag"] = module_quiet.eq("module_quiet").astype(int)
    out["r3u_se_missing_flag"] = se_worst.eq("<MISSING>").astype(int)
    out["r3u_ratio_lt_005_flag"] = ratio_bin.eq("ratio_LT_0.05").astype(int)
    out["r3u_mbk_quality_flag"] = ((mbk_flag == 1) & (mbk_comp >= 0.7)).astype(int)

    trust_score = (
        out["r3u_relationship_known_flag"]
        + out["r3u_relationship_recurrent_flag"]
        + out["r3u_relationship_strong_flag"]
        + out["r3u_receiver_known_flag"]
        + out["r3u_receiver_reputable_flag"]
        + out["r3u_receiver_strong_flag"]
        + out["r3u_mbk_quality_flag"]
        - out["r3u_missing_receiver_history_flag"]
        - out["r3u_first_receiver_flag"]
    )
    out[SCORE_COL] = trust_score.astype(float)

    conditions = [
        missing_receiver_history,
        receiver_strong,
        receiver_reputable,
        receiver_known,
    ]
    choices = ["missing_history", "strong_reputation", "medium_reputation", "some_history"]
    out["r3u_receiver_trust_bucket"] = np.select(conditions, choices, default="unknown")

    rel_conditions = [relationship_strong, relationship_recurrent, relationship_known]
    rel_choices = ["strong_relationship", "recurrent_relationship", "known_relationship"]
    out["r3u_relationship_bucket"] = np.select(rel_conditions, rel_choices, default="no_relationship")

    return out


def segment_rows(df: pd.DataFrame, base_mask: pd.Series, y: pd.Series, feature: str) -> List[Dict[str, Any]]:
    rows = []
    base_precision = float(y[base_mask].mean()) if int(base_mask.sum()) else 0.0
    total_fp = int(((base_mask) & (y == 0)).sum())
    total_tp = int(((base_mask) & (y == 1)).sum())
    s = cat(df, feature) if not pd.api.types.is_numeric_dtype(df.get(feature, pd.Series(dtype=float))) else df[feature].fillna(-999999).astype(str)
    for val, idx in s[base_mask].groupby(s[base_mask]).groups.items():
        mask = base_mask & s.eq(val)
        alert_count = int(mask.sum())
        tp = int((mask & (y == 1)).sum())
        fp = int((mask & (y == 0)).sum())
        if alert_count == 0:
            continue
        precision = tp / alert_count
        fp_share = fp / total_fp if total_fp else 0.0
        tp_share = tp / total_tp if total_tp else 0.0
        fp_to_tp = fp / tp if tp else float("inf")
        diagnostic_score = fp_share * 100.0 + max(0.0, base_precision - precision) * 100.0
        rows.append({
            "feature": feature,
            "value": str(val),
            "alert_count": alert_count,
            "tp": tp,
            "fp": fp,
            "precision": round(float(precision), 8),
            "base_precision": round(base_precision, 8),
            "precision_gap_vs_base": round(float(precision - base_precision), 8),
            "fp_share": round(float(fp_share), 8),
            "tp_share": round(float(tp_share), 8),
            "fp_to_tp_ratio": None if math.isinf(fp_to_tp) else round(float(fp_to_tp), 8),
            "diagnostic_score": round(float(diagnostic_score), 8),
        })
    return rows


def numeric_separation(df: pd.DataFrame, base_mask: pd.Series, y: pd.Series, features: Iterable[str]) -> pd.DataFrame:
    rows = []
    alert_y = y[base_mask]
    for feature in features:
        if feature not in df.columns:
            continue
        s = num(df, feature)
        s_alert = s[base_mask]
        if s_alert.notna().sum() < 20:
            continue
        auc = rank_auc(alert_y, s_alert)
        tp_vals = s_alert[alert_y == 1].dropna()
        fp_vals = s_alert[alert_y == 0].dropna()
        if len(tp_vals) == 0 or len(fp_vals) == 0:
            continue
        rows.append({
            "feature": feature,
            "n_alerts_non_null": int(s_alert.notna().sum()),
            "n_tp": int(len(tp_vals)),
            "n_fp": int(len(fp_vals)),
            "auc_fraud_high": round(float(auc), 8) if not math.isnan(auc) else None,
            "auc_fp_high": round(float(1 - auc), 8) if not math.isnan(auc) else None,
            "tp_median": float(tp_vals.median()),
            "fp_median": float(fp_vals.median()),
            "tp_p10": float(tp_vals.quantile(0.10)),
            "tp_p90": float(tp_vals.quantile(0.90)),
            "fp_p10": float(fp_vals.quantile(0.10)),
            "fp_p90": float(fp_vals.quantile(0.90)),
            "abs_auc_distance_from_random": round(float(abs(auc - 0.5)), 8) if not math.isnan(auc) else None,
        })
    return pd.DataFrame(rows).sort_values("abs_auc_distance_from_random", ascending=False)


def eval_candidate(
    df: pd.DataFrame,
    y: pd.Series,
    base_pred: pd.Series,
    mask: pd.Series,
    base_metrics: Dict[str, Any],
    target_fpr: float,
    target_fp: int,
    max_fn: int,
    min_recall: float,
    name: str,
    description: str,
    family: str,
) -> Dict[str, Any]:
    demote_mask = mask & (base_pred == 1)
    new_pred = base_pred.copy()
    new_pred[demote_mask] = 0
    m = metrics_from_pred(y, new_pred)
    fp_removed = int(base_metrics["fp"] - m["fp"])
    fn_delta = int(m["fn"] - base_metrics["fn"])
    tp_loss = int(base_metrics["tp"] - m["tp"])
    return {
        "candidate_id": name,
        "family": family,
        "description": description,
        "demoted_alerts": int(demote_mask.sum()),
        "fp_removed_vs_base": fp_removed,
        "tp_loss_vs_base": tp_loss,
        "fn_delta_vs_base": fn_delta,
        "tp": m["tp"],
        "fp": m["fp"],
        "fn": m["fn"],
        "tn": m["tn"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "fpr": m["fpr"],
        "target_fpr_reached": bool(m["fpr"] <= target_fpr),
        "target_gap_fp": max(0, int(m["fp"] - target_fp)),
        "within_fn_budget": bool(m["fn"] <= max_fn),
        "recall_ok": bool(m["recall"] >= min_recall),
        "policy_ok": bool(m["fn"] <= max_fn and m["recall"] >= min_recall),
    }


def candidate_masks(df: pd.DataFrame, base_mask: pd.Series) -> List[Tuple[str, str, str, pd.Series]]:
    rows: List[Tuple[str, str, str, pd.Series]] = []

    lgbm = num(df, "lgbm_r4_score")
    score_final = num(df, "score_final")
    if_perc = num(df, "if_percentile")
    trust = num(df, SCORE_COL).fillna(-99)
    module_quiet = cat(df, "module_quiet").eq("module_quiet")
    ratio_low = cat(df, "ratio_bin").eq("ratio_LT_0.05")

    def add(cid: str, family: str, desc: str, mask: pd.Series) -> None:
        rows.append((cid, family, desc, mask.fillna(False).astype(bool)))

    # Quantile thresholds from current alerts. Keep limited to avoid expensive grids.
    alert_lgbm = lgbm[base_mask & lgbm.notna()]
    alert_score = score_final[base_mask & score_final.notna()]
    alert_if = if_perc[base_mask & if_perc.notna()]
    lgbm_thresholds = sorted(set([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30] + [float(alert_lgbm.quantile(q)) for q in [0.05, 0.10, 0.20, 0.30] if len(alert_lgbm)]))
    score_thresholds = sorted(set([1.0, 2.0, 3.0, 5.0] + [float(alert_score.quantile(q)) for q in [0.05, 0.10, 0.20] if len(alert_score)]))
    if_thresholds = sorted(set([0.50, 0.70, 0.85] + [float(alert_if.quantile(q)) for q in [0.10, 0.20, 0.30] if len(alert_if)]))

    # Trust-driven demotions.
    for t in [1, 2, 3, 4]:
        trust_mask = trust >= t
        add(f"trust_ge_{t}", "trust_score_only", f"alert AND {SCORE_COL}>={t}", trust_mask)
        for thr in lgbm_thresholds:
            add(
                f"trust_ge_{t}_lgbm_le_{thr:.6g}",
                "trust_plus_low_lgbm",
                f"alert AND {SCORE_COL}>={t} AND lgbm_r4_score<={thr:.6g}",
                trust_mask & (lgbm <= thr),
            )
        for thr in score_thresholds:
            add(
                f"trust_ge_{t}_score_le_{thr:.6g}",
                "trust_plus_low_score",
                f"alert AND {SCORE_COL}>={t} AND score_final<={thr:.6g}",
                trust_mask & (score_final <= thr),
            )

    # Relationship / receiver reputation candidates.
    for flag_col in [
        "r3u_relationship_recurrent_flag",
        "r3u_relationship_strong_flag",
        "r3u_receiver_reputable_flag",
        "r3u_receiver_strong_flag",
        "r3u_mbk_quality_flag",
    ]:
        flag = num(df, flag_col).fillna(0).eq(1)
        add(flag_col, "single_trust_flag", f"alert AND {flag_col}=1", flag)
        for thr in lgbm_thresholds:
            add(
                f"{flag_col}_lgbm_le_{thr:.6g}",
                "trust_flag_plus_low_lgbm",
                f"alert AND {flag_col}=1 AND lgbm_r4_score<={thr:.6g}",
                flag & (lgbm <= thr),
            )

    # Missing-history diagnostic candidates: only useful if low score makes them safe.
    missing_hist = num(df, "r3u_missing_receiver_history_flag").fillna(0).eq(1)
    first_receiver = num(df, "r3u_first_receiver_flag").fillna(0).eq(1)
    for thr in lgbm_thresholds:
        add(
            f"missing_history_lgbm_le_{thr:.6g}",
            "missing_history_low_lgbm_probe",
            f"alert AND r3u_missing_receiver_history_flag=1 AND lgbm_r4_score<={thr:.6g}",
            missing_hist & (lgbm <= thr),
        )
    for thr in score_thresholds:
        add(
            f"missing_history_score_le_{thr:.6g}",
            "missing_history_low_score_probe",
            f"alert AND r3u_missing_receiver_history_flag=1 AND score_final<={thr:.6g}",
            missing_hist & (score_final <= thr),
        )

    # Module quiet + low risk probes.
    for thr in lgbm_thresholds:
        add(
            f"module_quiet_lgbm_le_{thr:.6g}",
            "module_quiet_low_lgbm_probe",
            f"alert AND module_quiet=module_quiet AND lgbm_r4_score<={thr:.6g}",
            module_quiet & (lgbm <= thr),
        )
    for thr in if_thresholds:
        add(
            f"ratio_low_if_le_{thr:.6g}",
            "ratio_low_if_probe",
            f"alert AND ratio_bin=ratio_LT_0.05 AND if_percentile<={thr:.6g}",
            ratio_low & (if_perc <= thr),
        )

    # Non-first receiver trust candidates.
    non_first = ~first_receiver
    for thr in lgbm_thresholds:
        add(
            f"non_first_lgbm_le_{thr:.6g}",
            "non_first_low_lgbm_probe",
            f"alert AND first_receiver_flag_real=0 AND lgbm_r4_score<={thr:.6g}",
            non_first & (lgbm <= thr),
        )

    # Deduplicate by id; avoid massive list.
    dedup: Dict[str, Tuple[str, str, str, pd.Series]] = {}
    for cid, fam, desc, mask in rows:
        if cid not in dedup:
            dedup[cid] = (cid, fam, desc, mask)
    return list(dedup.values())


def greedy_select(
    df: pd.DataFrame,
    y: pd.Series,
    base_pred: pd.Series,
    candidate_table: pd.DataFrame,
    mask_lookup: Dict[str, pd.Series],
    base_metrics: Dict[str, Any],
    target_fpr: float,
    target_fp: int,
    max_fn: int,
    min_recall: float,
    max_rules: int,
    min_fp_removed: int,
) -> Tuple[pd.Series, List[Dict[str, Any]], pd.DataFrame]:
    current_pred = base_pred.copy()
    selected: List[Dict[str, Any]] = []
    frontier_rows: List[Dict[str, Any]] = []
    used: set = set()

    viable = candidate_table[(candidate_table["policy_ok"] == True) & (candidate_table["fp_removed_vs_base"] >= min_fp_removed)].copy()
    if viable.empty:
        return current_pred, selected, pd.DataFrame(frontier_rows)

    viable["utility"] = viable["fp_removed_vs_base"] - 250 * viable["tp_loss_vs_base"] - 0.01 * viable["target_gap_fp"]
    viable = viable.sort_values(["target_fpr_reached", "utility", "fp_removed_vs_base"], ascending=[False, False, False])

    for _, row in viable.iterrows():
        if len(selected) >= max_rules:
            break
        cid = row["candidate_id"]
        if cid in used:
            continue
        mask = mask_lookup[cid] & (current_pred == 1)
        if int(mask.sum()) == 0:
            continue
        new_pred = current_pred.copy()
        new_pred[mask] = 0
        m = metrics_from_pred(y, new_pred)
        marginal_fp_removed = int(((mask) & (y == 0)).sum())
        marginal_tp_loss = int(((mask) & (y == 1)).sum())
        if marginal_fp_removed < min_fp_removed:
            continue
        if m["fn"] > max_fn or m["recall"] < min_recall:
            continue
        selected_row = {
            "rule_index": len(selected),
            "candidate_id": cid,
            "family": row["family"],
            "description": row["description"],
            "marginal_demoted_alerts": int(mask.sum()),
            "marginal_fp_removed": marginal_fp_removed,
            "marginal_tp_loss": marginal_tp_loss,
            "cumulative_fp_removed": int(base_metrics["fp"] - m["fp"]),
            "cumulative_tp_loss": int(base_metrics["tp"] - m["tp"]),
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "precision": m["precision"],
            "recall": m["recall"],
            "fpr": m["fpr"],
            "target_gap_fp": max(0, int(m["fp"] - target_fp)),
            "target_fpr_reached": bool(m["fpr"] <= target_fpr),
        }
        selected.append(selected_row)
        frontier_rows.append(selected_row.copy())
        current_pred = new_pred
        used.add(cid)
        if m["fpr"] <= target_fpr:
            break
    return current_pred, selected, pd.DataFrame(frontier_rows)


def robustness_by_segment(df: pd.DataFrame, y: pd.Series, base_pred: pd.Series, final_pred: pd.Series) -> pd.DataFrame:
    seg_cols = [
        "temporal_split",
        "event_month",
        "ds_tipo_chave_norm",
        "value_band",
        "periodo_dia",
        "r3u_receiver_trust_bucket",
        "r3u_relationship_bucket",
        "r3u_missing_receiver_history_flag",
        "r3u_module_quiet_flag",
        "r3u_first_receiver_flag",
        "mbk_available_flag",
    ]
    rows = []
    for col in seg_cols:
        if col not in df.columns:
            continue
        s = cat(df, col) if not pd.api.types.is_numeric_dtype(df[col]) else df[col].fillna(-999999).astype(str)
        for val in sorted(s.dropna().unique()):
            mask = s.eq(val)
            if int(mask.sum()) == 0:
                continue
            before = metrics_from_pred(y[mask], base_pred[mask])
            after = metrics_from_pred(y[mask], final_pred[mask])
            rows.append({
                "segment_col": col,
                "segment_value": str(val),
                "n_rows": int(mask.sum()),
                "n_frauds": int((y[mask] == 1).sum()),
                "base_tp": before["tp"],
                "base_fp": before["fp"],
                "base_fn": before["fn"],
                "final_tp": after["tp"],
                "final_fp": after["fp"],
                "final_fn": after["fn"],
                "fp_removed": int(before["fp"] - after["fp"]),
                "fn_delta": int(after["fn"] - before["fn"]),
                "final_recall": after["recall"],
                "final_fpr": after["fpr"],
            })
    return pd.DataFrame(rows).sort_values(["fp_removed", "fn_delta"], ascending=[False, False])


def build_report(summary: Dict[str, Any], top_diag: pd.DataFrame, selected: List[Dict[str, Any]], selected_policy: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# {EXP_NAME} - Receiver / Relationship Trust Feature Probe")
    lines.append("")
    lines.append("## Resultado executivo")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- All pass: `{summary['all_pass']}`")
    lines.append(f"- Base: `{summary['base_metrics']}`")
    lines.append(f"- Meta comercial: FN<={summary['max_fn']}, recall>={summary['min_recall']}, FPR<={summary['target_fpr']}")
    lines.append(f"- FP max alvo: `{summary['target_fp']}`")
    lines.append(f"- Gap base ate alvo: `{summary['target_gap_fp_from_base']}` FP")
    lines.append(f"- Politica selecionada: `{selected_policy.get('policy_name')}`")
    lines.append(f"- Metricas recomendadas: `{summary['recommended_metrics']}`")
    lines.append(f"- FP removidos vs base: `{summary['fp_removed_vs_base']}`")
    lines.append(f"- FN delta vs base: `{summary['fn_delta_vs_base']}`")
    lines.append(f"- Target comercial atingido: `{summary['commercial_target_reached']}`")
    lines.append("")
    lines.append("## Top diagnosticos de confianca")
    if not top_diag.empty:
        cols = ["feature", "value", "alert_count", "tp", "fp", "precision", "fp_share", "diagnostic_score"]
        lines.append(top_diag[cols].head(20).to_markdown(index=False))
    else:
        lines.append("Sem diagnosticos suficientes.")
    lines.append("")
    lines.append("## Regras selecionadas")
    if selected:
        lines.append(pd.DataFrame(selected).to_markdown(index=False))
    else:
        lines.append("Nenhuma regra de confianca atingiu ganho minimo dentro do orcamento.")
    lines.append("")
    lines.append("## Decisao sugerida")
    if summary["commercial_target_reached"]:
        lines.append("Candidato atingiu a meta comercial. Proximo passo: R3U-FROZEN, sem nova mineracao.")
    elif summary["fp_removed_vs_base"] > 0:
        lines.append("Houve ganho diagnostico, mas meta comercial nao foi atingida. Usar achados para decidir se vale frozen ou enriquecimento de dados.")
    else:
        lines.append("Nao houve ganho seguro com as features existentes. Priorizar enriquecimento de dados de reputacao/relacionamento antes de nova otimizacao.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="CSV de predicoes de entrada")
    parser.add_argument("--base-col", default=None, help="Coluna base de predicao")
    parser.add_argument("--output-dir", default=None, help="Diretorio de saida")
    parser.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    parser.add_argument("--max-fn", type=int, default=DEFAULT_MAX_FN)
    parser.add_argument("--min-recall", type=float, default=DEFAULT_MIN_RECALL)
    parser.add_argument("--max-rules", type=int, default=DEFAULT_MAX_RULES)
    parser.add_argument("--min-fp-removed", type=int, default=DEFAULT_MIN_FP_REMOVED)
    args = parser.parse_args()

    t0 = time.time()
    repo_root = Path.cwd()
    input_path = Path(args.input) if args.input else find_input_path(repo_root)
    if input_path is None or not input_path.exists():
        raise FileNotFoundError("Nao encontrei CSV de entrada. Use --input caminho/do/arquivo.csv")

    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "resultados" / "experimentos" / EXP_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)
    if "is_fraud" not in df.columns:
        raise ValueError("Coluna obrigatoria ausente: is_fraud")
    y = bin_to_int(df["is_fraud"])
    base_col = choose_base_col(df, args.base_col)
    if not base_col:
        raise ValueError("Nao encontrei coluna de predicao base")
    base_pred = bin_to_int(df[base_col])

    df = add_trust_features(df)
    base_metrics = metrics_from_pred(y, base_pred)
    n_rows = int(len(df))
    n_frauds = int((y == 1).sum())
    n_normals = int((y == 0).sum())
    target_fp = int(math.floor(args.target_fpr * n_normals))
    target_gap_base = max(0, int(base_metrics["fp"] - target_fp))
    wilson_low, wilson_high = wilson_interval(base_metrics["tp"], n_frauds)

    contract = {
        "n_rows": n_rows,
        "n_frauds": n_frauds,
        "n_normals": n_normals,
        "input_path": str(input_path),
        "base_col": base_col,
        "target_fpr": args.target_fpr,
        "target_fp": target_fp,
        "max_fn": args.max_fn,
        "min_recall": args.min_recall,
        "missing": [],
        "contract_ok": True,
    }
    required_cols = ["is_fraud", base_col]
    contract["missing"] = [c for c in required_cols if c not in df.columns]
    contract["contract_ok"] = len(contract["missing"]) == 0

    base_summary = {
        "base_metrics": base_metrics,
        "target_fpr_ok": bool(base_metrics["fpr"] <= args.target_fpr),
        "fn_budget_ok": bool(base_metrics["fn"] <= args.max_fn),
        "recall_ok": bool(base_metrics["recall"] >= args.min_recall),
        "target_fp": target_fp,
        "target_gap_fp": target_gap_base,
        "wilson_low": wilson_low,
        "wilson_high": wilson_high,
    }

    base_mask = base_pred == 1
    diag_features = [
        "r3u_receiver_trust_bucket",
        "r3u_relationship_bucket",
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
        "qtd_rec_bin",
        "valor_rec_bin",
        "ratio_bin",
        "score_bin",
        "if_bin",
        "lgbm_bin",
        "value_band",
        "ds_tipo_chave_norm",
    ]
    diag_rows: List[Dict[str, Any]] = []
    for f in diag_features:
        if f in df.columns:
            diag_rows.extend(segment_rows(df, base_mask, y, f))
    diag_df = pd.DataFrame(diag_rows)
    if not diag_df.empty:
        diag_df = diag_df.sort_values("diagnostic_score", ascending=False)

    numeric_features = [
        SCORE_COL,
        "lgbm_r4_score",
        "score_final",
        "if_percentile",
        "qtd_pix_mesmo_recebedor_30d",
        "qtd_pix_mesmo_recebedor_90d",
        "qtd_pix_mesmo_recebedor_180d",
        "valor_total_para_recebedor_30d",
        "valor_total_para_recebedor_90d",
        "valor_total_para_recebedor_180d",
        "dias_desde_primeiro_envio_recebedor",
        "qtd_pix_recebidos_30d",
        "qtd_pix_recebidos_90d",
        "qtd_pix_recebidos_180d",
        "valor_total_recebido_30d",
        "valor_total_recebido_90d",
        "valor_total_recebido_180d",
        "soma_pagadores_distintos_dia_recebedor_180d",
        "max_qtd_pix_recebidos_dia_180d",
        "mbk_completeness_score",
        "topaz_risk_score",
    ]
    numsep_df = numeric_separation(df, base_mask, y, numeric_features)

    masks = candidate_masks(df, base_mask)
    mask_lookup: Dict[str, pd.Series] = {cid: mask for cid, _, _, mask in masks}
    cand_rows = []
    for cid, fam, desc, mask in masks:
        cand_rows.append(eval_candidate(
            df=df,
            y=y,
            base_pred=base_pred.copy(),
            mask=mask,
            base_metrics=base_metrics,
            target_fpr=args.target_fpr,
            target_fp=target_fp,
            max_fn=args.max_fn,
            min_recall=args.min_recall,
            name=cid,
            description=desc,
            family=fam,
        ))
    cand_df = pd.DataFrame(cand_rows)
    if not cand_df.empty:
        cand_df = cand_df.sort_values(["policy_ok", "target_fpr_reached", "fp_removed_vs_base", "tp_loss_vs_base"], ascending=[False, False, False, True])

    final_pred, selected, frontier_df = greedy_select(
        df=df,
        y=y,
        base_pred=base_pred.copy(),
        candidate_table=cand_df,
        mask_lookup=mask_lookup,
        base_metrics=base_metrics,
        target_fpr=args.target_fpr,
        target_fp=target_fp,
        max_fn=args.max_fn,
        min_recall=args.min_recall,
        max_rules=args.max_rules,
        min_fp_removed=args.min_fp_removed,
    )

    recommended_metrics = metrics_from_pred(y, final_pred)
    fp_removed_vs_base = int(base_metrics["fp"] - recommended_metrics["fp"])
    fn_delta_vs_base = int(recommended_metrics["fn"] - base_metrics["fn"])
    target_gap_fp = max(0, int(recommended_metrics["fp"] - target_fp))
    commercial_target_reached = bool(
        recommended_metrics["fn"] <= args.max_fn
        and recommended_metrics["recall"] >= args.min_recall
        and recommended_metrics["fpr"] <= args.target_fpr
    )

    if commercial_target_reached:
        objective_status = "DONE_R3U_TRUST_FEATURE_POLICY_TARGET_REACHED_NEEDS_FROZEN"
    elif fp_removed_vs_base > 0:
        objective_status = "DONE_R3U_TRUST_FEATURE_POLICY_FP_REDUCED_TARGET_NOT_REACHED"
    else:
        objective_status = "DONE_R3U_TRUST_FEATURE_DIAGNOSTIC_NO_SAFE_GAIN"

    df[OUT_COL] = final_pred.astype(int)

    selected_policy = {
        "policy_name": "r3u_receiver_relationship_trust_probe_policy" if selected else "r3u_no_safe_policy_selected",
        "base_col": base_col,
        "final_pred_col": OUT_COL,
        "score_col": SCORE_COL,
        "target_fpr": args.target_fpr,
        "target_fp": target_fp,
        "max_fn": args.max_fn,
        "min_recall": args.min_recall,
        "base_metrics": base_metrics,
        "recommended_metrics": recommended_metrics,
        "fp_removed_vs_base": fp_removed_vs_base,
        "fn_delta_vs_base": fn_delta_vs_base,
        "target_gap_fp": target_gap_fp,
        "commercial_target_reached": commercial_target_reached,
        "selected_rules": selected,
        "notes": [
            "This is a receiver/relationship trust feature probe.",
            "No external data is required; derived features are built from current prediction columns.",
            "Promotion requires frozen validation and business review if FN > 0.",
        ],
    }

    robustness_df = robustness_by_segment(df, y, base_pred, final_pred)

    summary = {
        "experiment": EXP_NAME,
        "status": "DONE",
        "objective_status": objective_status,
        "n_rows": n_rows,
        "n_frauds": n_frauds,
        "n_normals": n_normals,
        "input_path": str(input_path),
        "base_col": base_col,
        "base_metrics": base_metrics,
        "target_fpr": args.target_fpr,
        "target_fp": target_fp,
        "target_gap_fp_from_base": target_gap_base,
        "max_fn": args.max_fn,
        "min_recall": args.min_recall,
        "recommended_policy_name": selected_policy["policy_name"],
        "recommended_metrics": recommended_metrics,
        "fp_removed_vs_base": fp_removed_vs_base,
        "fn_delta_vs_base": fn_delta_vs_base,
        "target_gap_fp": target_gap_fp,
        "commercial_target_reached": commercial_target_reached,
        "n_candidates_evaluated": int(len(cand_df)),
        "n_selected_rules": int(len(selected)),
        "all_pass": bool(contract["contract_ok"]),
        "elapsed_seconds": round(time.time() - t0, 2),
        "output_dir": str(output_dir),
    }

    write_json(output_dir / "00_run_summary.json", summary)
    write_json(output_dir / "01_input_contract.json", contract)
    write_json(output_dir / "02_base_metrics.json", base_summary)
    diag_df.to_csv(output_dir / "03_trust_feature_diagnostics.csv", index=False)
    cand_df.to_csv(output_dir / "04_trust_demote_candidates.csv", index=False)
    write_json(output_dir / "05_selected_policy.json", selected_policy)
    frontier_df.to_csv(output_dir / "06_policy_frontier.csv", index=False)
    robustness_df.to_csv(output_dir / "07_robustness_by_segment.csv", index=False)
    write_json(output_dir / "08_policy_artifact_recommended.json", selected_policy)
    df.to_csv(output_dir / "09_predictions_recommended.csv", index=False)
    report = build_report(summary, diag_df, selected, selected_policy)
    (output_dir / "10_exp014b_r3u_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
