#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3N — Consolidate R3M-FROZEN + Irreducible FN Search

Objetivo:
  Continuar a estratégia FN First / FP Second até aproximar os FNs irredutíveis.

Parte A — R3M-FROZEN:
  Reaplica o artifact recomendado do EXP-014B-R3M, sem nova mineração, e valida:

    EXP014B_R3M_CAP5100
    TP=1444
    FP=4816
    FN=21
    recall=98,567%
    precision=23,067%
    FPR=4,285%
    Wilson low≈97,819%

Parte B — Irreducible FN Search:
  Usa o R3M congelado como base e tenta reduzir os 21 FNs residuais usando:
    1. novo headroom TP0 sobre alertas atuais;
    2. reuso das bibliotecas de rescues R3M/R3L/R3I;
    3. novos rescues gerados diretamente dos FNs residuais;
    4. re-tightening curto apenas nos alertas adicionados;
    5. avaliação por caps absolutos de FP.

Uso recomendado:
  python scripts/exp_014b_r3n_consolidate_r3m_and_irreducible_fn_search.py --fp-caps 5000,5050,5100,5200 --preferred-fp-cap 5000 --max-new-combo-size 4 --max-seconds-headroom 120 --max-seconds-retighten 90

Execução mais agressiva para buscar FN irredutível:
  python scripts/exp_014b_r3n_consolidate_r3m_and_irreducible_fn_search.py --fp-caps 5000,5100,5250,5500 --preferred-fp-cap 5100 --max-new-combo-size 5 --max-fp-added-candidate 900 --max-seconds-headroom 180 --max-seconds-retighten 120

Saídas:
  resultados/experimentos/EXP-014B-R3N/
    00_run_summary.json
    01_input_contract.json
    02_r3m_frozen_validation.json
    03_r3m_frozen_metrics.csv
    04_r3m_frozen_rule_impact.csv
    05_r3n_base_metrics.csv
    06_r3n_headroom_candidates.csv
    07_r3n_headroom_selected_rules.csv
    08_r3n_headroom_metrics.csv
    09_r3n_rescue_candidates.csv
    10_r3n_scenarios_before_retightening.csv
    11_r3n_scenario_metrics_after_retightening.csv
    12_r3n_selected_rules_by_scenario.csv
    13_r3n_residual_fns_recommended.csv
    14_policy_artifact_recommended.json
    15_predictions_recommended.csv
    16_exp014b_r3n_report.md
"""

from __future__ import annotations

import argparse
import itertools
import json
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

DEFAULT_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3M" / "14_predictions_recommended.csv"
DEFAULT_R3M_ARTIFACT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3M" / "13_policy_artifact_recommended.json"
DEFAULT_R3M_RESCUES = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3M" / "09_r3m_rescue_candidates.csv"
DEFAULT_R3L_RESCUES = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3L" / "06_rescue_candidates.csv"
DEFAULT_R3I_RESCUES = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3I" / "07_rescue_candidates.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3N"

R3L_BASE_COL = "exp014b_r3l_frozen_pred"
R3M_EXISTING_COL = "exp014b_r3m_recommended_pred"
R3M_FROZEN_COL = "exp014b_r3m_frozen_pred"
R3N_FINAL_COL = "exp014b_r3n_recommended_pred"

EXPECTED_R3M = {
    "scenario": "r3m_cap_5100",
    "tp": 1444,
    "fp": 4816,
    "fn": 21,
    "headroom_fp_removed": 183,
    "rescue_fn_recovered": 8,
    "rescue_fp_added": 357,
    "retightening_fp_removed": 279,
    "retightening_tp_loss": 0,
    "wilson_low_min": 0.95,
}

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
]

NUMERIC_COLS = [
    "lgbm_r4_score",
    "lgbm_mapped",
    "lgbm_raw",
    "score_final",
    "if_percentile",
    "if_percentile_x",
    "if_percentile_y",
    "vl_pix",
    "ratio_valor_media_pagador_90d",
    "qtd_pix_recebidos_180d",
    "valor_total_recebido_180d",
]


@dataclass
class Candidate:
    rule_id: str
    family: str
    description: str
    mask: np.ndarray
    tp_loss: int = 0
    fp_effect: int = 0
    n_effect: int = 0
    ratio: float = float("inf")
    params: dict[str, Any] | None = None


def log(msg: str) -> None:
    print(msg, flush=True)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().split(".")[-1] for c in df.columns]

    if "transaction_id" not in df.columns and "cd_pix" in df.columns:
        df["transaction_id"] = df["cd_pix"]
    if "event_datetime" not in df.columns and "dt_pix" in df.columns:
        df["event_datetime"] = df["dt_pix"]

    if "is_fraud" not in df.columns:
        raise RuntimeError("Coluna obrigatória ausente: is_fraud")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)

    if "transaction_id" in df.columns:
        df["transaction_id"] = df["transaction_id"].astype("string").str.strip()

    for c in ["event_datetime", "data_pix"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    for c in [R3L_BASE_COL, R3M_EXISTING_COL, R3M_FROZEN_COL, R3N_FINAL_COL]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

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


def qbin_series(s: pd.Series, name: str, bins: list[float]) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    labels = []
    edges = [-np.inf] + bins + [np.inf]
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        if np.isneginf(left):
            labels.append(f"{name}_LT_{right:g}")
        elif np.isposinf(right):
            labels.append(f"{name}_GE_{left:g}")
        else:
            labels.append(f"{name}_{left:g}_{right:g}")
    return pd.cut(vals, bins=edges, labels=labels, include_lowest=True).astype("string").fillna(f"{name}_MISSING").astype(str)


def add_bins_and_guards(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "lgbm_bin" not in df.columns and pick_col(df, ["lgbm_r4_score", "lgbm_mapped", "lgbm_raw"]):
        df["lgbm_bin"] = qbin_series(num(df, ["lgbm_r4_score", "lgbm_mapped", "lgbm_raw"], 0.0), "lgbm", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1])
    if "if_bin" not in df.columns and pick_col(df, ["if_percentile", "if_percentile_x", "if_percentile_y"]):
        df["if_bin"] = qbin_series(num(df, ["if_percentile", "if_percentile_x", "if_percentile_y"], 0.0), "if", [0.32, 0.5, 0.7, 0.85, 0.95])
    if "score_bin" not in df.columns and "score_final" in df.columns:
        df["score_bin"] = qbin_series(num(df, "score_final", 0.0), "score", [0.5, 1, 2, 3, 5, 10])
    if "ratio_bin" not in df.columns and "ratio_valor_media_pagador_90d" in df.columns:
        df["ratio_bin"] = qbin_series(num(df, "ratio_valor_media_pagador_90d", 0.0), "ratio", [0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    if "qtd_rec_bin" not in df.columns and "qtd_pix_recebidos_180d" in df.columns:
        df["qtd_rec_bin"] = qbin_series(num(df, "qtd_pix_recebidos_180d", 0.0), "qtdrec", [0, 1, 2, 5, 10, 20, 50, 100])
    if "vl_bin" not in df.columns and "vl_pix" in df.columns:
        df["vl_bin"] = qbin_series(num(df, "vl_pix", 0.0), "vl", [20, 50, 100, 250, 500, 1000, 5000, 10000])

    if "module_quiet" not in df.columns:
        se_score = num(df, ["se_score_x", "se_score_y", "se_score"], 0.0)
        se_count = num(df, ["se_patterns_count", "se_pattern_count"], 0.0)
        beh_score = num(df, ["beh_score", "behavioral_score"], 0.0)
        beh_count = num(df, ["beh_factors_count", "behavioral_risk_factor_count"], 0.0)
        runtime = num(df, "runtime_flagged", 0.0)
        strong = (se_score >= 40) | (se_count >= 2) | (beh_score >= 25) | (beh_count >= 2) | (runtime >= 1)
        df["module_quiet"] = np.where(strong, "module_strong", "module_quiet")

    return df


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


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n) + (z**2 / (4 * n**2))) ** 0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def parse_params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    return json.loads(str(raw).replace("Infinity", "1e999"))


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in FEATURE_COLS:
        if c in df.columns:
            out[c] = df[c].astype("string").fillna("<MISSING>").astype(str)
    return out


def rule_mask(df: pd.DataFrame, current_pred: np.ndarray, params: dict[str, Any], mode: str, scope_mask: np.ndarray | None = None) -> np.ndarray:
    rtype = params.get("type")
    if rtype in ["numeric_headroom", "numeric_threshold_rescue", "numeric_retighten"]:
        c = params.get("col")
        if c not in df.columns:
            return np.zeros(len(df), dtype=bool)
        vals = num(df, c, 0.0).to_numpy(dtype=float)
        direction = params.get("direction")
        cut = float(params.get("cut"))
        mask = (vals >= cut) if direction == "ge" else (vals <= cut)
    elif rtype in ["combo_headroom", "combo_rescue", "combo_retighten"]:
        mask = np.ones(len(df), dtype=bool)
        for c, v in zip(params.get("combo_cols", []), params.get("combo_values", [])):
            if c not in df.columns:
                return np.zeros(len(df), dtype=bool)
            mask = mask & (df[c].astype("string").fillna("<MISSING>").astype(str).to_numpy() == str(v))
    else:
        return np.zeros(len(df), dtype=bool)

    if params.get("require_module_quiet", False):
        if "module_quiet" not in df.columns:
            return np.zeros(len(df), dtype=bool)
        mask = mask & (df["module_quiet"].astype(str).to_numpy() == "module_quiet")

    if mode in ["headroom", "retighten"]:
        mask = mask & (current_pred.astype(int) == 1)
    elif mode == "rescue":
        mask = mask & (current_pred.astype(int) == 0)
    else:
        raise RuntimeError(f"mode inválido: {mode}")

    if scope_mask is not None:
        mask = mask & scope_mask

    return mask


def apply_rules(df: pd.DataFrame, pred: np.ndarray, rules: list[dict[str, Any]], mode: str, y: np.ndarray, scope_mask: np.ndarray | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    current = pred.copy().astype(int)
    rows = []

    for i, rule in enumerate(rules):
        params = parse_params(rule.get("params_json") or rule.get("params") or "{}")
        mask = rule_mask(df, current, params, mode=mode, scope_mask=scope_mask)
        tp_loss = int(((y == 1) & mask).sum()) if mode in ["headroom", "retighten"] else 0
        fp_removed = int(((y == 0) & mask).sum()) if mode in ["headroom", "retighten"] else 0
        fn_recovered = int(((y == 1) & mask).sum()) if mode == "rescue" else 0
        fp_added = int(((y == 0) & mask).sum()) if mode == "rescue" else 0

        if mode in ["headroom", "retighten"]:
            current[mask] = 0
        elif mode == "rescue":
            current[mask] = 1

        rows.append({
            "phase": mode,
            "rule_index": i,
            "rule_id": rule.get("rule_id") or rule.get("candidate_id"),
            "family": rule.get("family"),
            "description": rule.get("description"),
            "tp_loss": tp_loss,
            "fp_removed": fp_removed,
            "fn_recovered": fn_recovered,
            "fp_added": fp_added,
            "n_effect": int(mask.sum()),
            "params_json": json.dumps(params, ensure_ascii=False),
        })

    return current, pd.DataFrame(rows)


def apply_rescue_ids(df: pd.DataFrame, pred: np.ndarray, rescue_df: pd.DataFrame, ids: list[str], y: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    selected = rescue_df[rescue_df["candidate_id"].astype(str).isin(set(ids))].copy()
    order = {cid: i for i, cid in enumerate(ids)}
    selected["_order"] = selected["candidate_id"].astype(str).map(order)
    selected = selected.sort_values("_order")
    rules = selected.to_dict(orient="records")
    return apply_rules(df, pred, rules, mode="rescue", y=y)


def validate_r3m_frozen(df: pd.DataFrame, artifact: dict[str, Any], rescue_df: pd.DataFrame, output_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    y = df["is_fraud"].to_numpy(dtype=int)
    scenario_name = artifact.get("recommended_scenario") or EXPECTED_R3M["scenario"]
    scenario = artifact["scenario_artifacts"][scenario_name]

    if R3L_BASE_COL not in df.columns:
        raise RuntimeError(f"Coluna base do R3M ausente: {R3L_BASE_COL}")
    base_pred = df[R3L_BASE_COL].to_numpy(dtype=int)
    base_metrics = compute_metrics(y, base_pred)

    head_rules = scenario.get("selected_headroom_rules", [])
    pred_after_head, head_impact = apply_rules(df, base_pred, head_rules, mode="headroom", y=y)
    head_metrics = compute_metrics(y, pred_after_head)

    rescue_ids = [str(x) for x in scenario.get("selected_rescue_candidate_ids", [])]
    pred_after_rescue, rescue_impact = apply_rescue_ids(df, pred_after_head, rescue_df, rescue_ids, y)
    rescue_metrics = compute_metrics(y, pred_after_rescue)

    added_by_rescue = (pred_after_rescue.astype(int) == 1) & (pred_after_head.astype(int) == 0)
    ret_rules = scenario.get("selected_retighten_rules", [])
    pred_final, ret_impact = apply_rules(df, pred_after_rescue, ret_rules, mode="retighten", y=y, scope_mask=added_by_rescue)
    final_metrics = compute_metrics(y, pred_final)

    all_impact = pd.concat([head_impact, rescue_impact, ret_impact], ignore_index=True)
    all_impact.to_csv(output_dir / "04_r3m_frozen_rule_impact.csv", index=False)

    metrics_df = pd.DataFrame([
        {"policy_name": "R3L_BASE", **base_metrics},
        {"policy_name": "R3M_AFTER_HEADROOM", **head_metrics},
        {"policy_name": "R3M_AFTER_RESCUE_BEFORE_RETIGHTEN", **rescue_metrics},
        {"policy_name": "EXP014B_R3M_FROZEN_FINAL", **final_metrics},
    ])
    metrics_df.to_csv(output_dir / "03_r3m_frozen_metrics.csv", index=False)

    headroom_fp_removed = base_metrics["fp"] - head_metrics["fp"]
    rescue_fn_recovered = head_metrics["fn"] - rescue_metrics["fn"]
    rescue_fp_added = rescue_metrics["fp"] - head_metrics["fp"]
    retightening_fp_removed = rescue_metrics["fp"] - final_metrics["fp"]
    retightening_tp_loss = rescue_metrics["tp"] - final_metrics["tp"]
    wl, wh = wilson_ci(final_metrics["tp"], int(y.sum()))

    expected_metrics_match = (
        final_metrics["tp"] == EXPECTED_R3M["tp"]
        and final_metrics["fp"] == EXPECTED_R3M["fp"]
        and final_metrics["fn"] == EXPECTED_R3M["fn"]
    )
    expected_phases_match = (
        int(headroom_fp_removed) == EXPECTED_R3M["headroom_fp_removed"]
        and int(rescue_fn_recovered) == EXPECTED_R3M["rescue_fn_recovered"]
        and int(rescue_fp_added) == EXPECTED_R3M["rescue_fp_added"]
        and int(retightening_fp_removed) == EXPECTED_R3M["retightening_fp_removed"]
        and int(retightening_tp_loss) == EXPECTED_R3M["retightening_tp_loss"]
    )
    wilson_pass = wl >= EXPECTED_R3M["wilson_low_min"]
    all_pass = expected_metrics_match and expected_phases_match and wilson_pass

    validation = {
        "scenario_name": scenario_name,
        "base_metrics": base_metrics,
        "headroom_metrics": head_metrics,
        "rescue_metrics_before_retightening": rescue_metrics,
        "final_metrics": final_metrics,
        "headroom_fp_removed": int(headroom_fp_removed),
        "rescue_fn_recovered": int(rescue_fn_recovered),
        "rescue_fp_added": int(rescue_fp_added),
        "retightening_fp_removed": int(retightening_fp_removed),
        "retightening_tp_loss": int(retightening_tp_loss),
        "wilson_low": wl,
        "wilson_high": wh,
        "expected_metrics_match": bool(expected_metrics_match),
        "expected_phases_match": bool(expected_phases_match),
        "wilson_pass": bool(wilson_pass),
        "all_pass": bool(all_pass),
        "status": "PASS_R3M_FROZEN_VALIDATED" if all_pass else "FAIL_R3M_FROZEN_DIVERGENCE",
    }
    dump_json(validation, output_dir / "02_r3m_frozen_validation.json")
    return pred_final, validation


def candidates_df(cands: list[Candidate], scenario: str = "") -> pd.DataFrame:
    return pd.DataFrame([{
        "scenario": scenario,
        "candidate_index": i,
        "rule_id": c.rule_id,
        "family": c.family,
        "description": c.description,
        "tp_loss": c.tp_loss,
        "fp_effect": c.fp_effect,
        "n_effect": c.n_effect,
        "ratio": c.ratio,
        "params_json": json.dumps(c.params or {}, ensure_ascii=False),
    } for i, c in enumerate(cands)])


def add_tp0_candidate(out, family, desc, mask, y, min_fp_removed, params):
    if not mask.any():
        return
    tp_loss = int(((y == 1) & mask).sum())
    fp_removed = int(((y == 0) & mask).sum())
    if tp_loss != 0 or fp_removed < min_fp_removed:
        return
    out.append(Candidate(
        rule_id=f"c_{len(out):05d}",
        family=family,
        description=desc,
        mask=mask,
        tp_loss=0,
        fp_effect=fp_removed,
        n_effect=int(mask.sum()),
        ratio=float("inf"),
        params=params,
    ))


def mine_tp0_veto_candidates(df, pred, min_fp_removed, max_combo_size, top_groups_per_combo, prefix_family):
    y = df["is_fraud"].to_numpy(dtype=int)
    alerted = pred.astype(int) == 1
    out = []

    for c in NUMERIC_COLS:
        if c not in df.columns:
            continue
        vals = num(df, c, 0.0).to_numpy(dtype=float)
        active = vals[alerted]
        if len(active) == 0:
            continue
        try:
            cuts = sorted(set(float(x) for x in np.quantile(active, [0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]) if np.isfinite(x)))
        except Exception:
            cuts = []
        for cut in cuts:
            for direction in ["le", "ge"]:
                mask = alerted & ((vals <= cut) if direction == "le" else (vals >= cut))
                desc = f"alert AND {c}<={cut:g}" if direction == "le" else f"alert AND {c}>={cut:g}"
                add_tp0_candidate(out, f"{prefix_family}_numeric_tp0", desc, mask, y, min_fp_removed, {"type": "numeric_headroom", "col": c, "direction": direction, "cut": cut})

    feat = feature_frame(df)
    cols = list(feat.columns)
    bins = [c for c in cols if c.endswith("_bin") or c == "module_quiet"]
    important = ["ds_tipo_chave_norm", "value_band", "mbk_available_flag", "first_receiver_flag_real", "periodo_dia"]
    idx = np.where(alerted)[0]

    for r in range(1, max_combo_size + 1):
        for combo in itertools.combinations(cols, r):
            combo = list(combo)
            if r == 1 and combo[0] not in bins + ["ds_tipo_chave_norm", "value_band"]:
                continue
            if r >= 2 and not any(c in combo for c in important + bins):
                continue
            sub = feat.iloc[idx][combo]
            if sub.empty:
                continue
            group_rows = []
            for key, rel_idxs in sub.groupby(combo, dropna=False).indices.items():
                idxs = sub.iloc[list(rel_idxs)].index.to_numpy(dtype=int)
                if len(idxs) < min_fp_removed:
                    continue
                mask = np.zeros(len(df), dtype=bool)
                mask[idxs] = True
                mask = mask & alerted
                tp_loss = int(((y == 1) & mask).sum())
                fp_removed = int(((y == 0) & mask).sum())
                if tp_loss == 0 and fp_removed >= min_fp_removed:
                    group_rows.append((-fp_removed, key, mask, fp_removed))
            group_rows.sort()
            for _, key, mask, fp_removed in group_rows[:top_groups_per_combo]:
                vals = key if isinstance(key, tuple) else (key,)
                vals = [str(v) for v in vals]
                desc = "alert AND " + " AND ".join([f"{c}={v}" for c, v in zip(combo, vals)])
                add_tp0_candidate(out, f"{prefix_family}_combo_tp0", desc, mask, y, min_fp_removed, {"type": "combo_headroom", "combo_cols": combo, "combo_values": vals})

    best = {}
    for c in out:
        k = np.packbits(c.mask).tobytes()
        old = best.get(k)
        if old is None or c.fp_effect > old.fp_effect:
            best[k] = c
    out = list(best.values())
    out.sort(key=lambda c: (-c.fp_effect, len(c.description)))
    for i, c in enumerate(out):
        c.rule_id = f"{prefix_family}_{i:05d}"
    return out


def greedy_select_tp0(cands: list[Candidate], pred: np.ndarray, y: np.ndarray, max_rules: int, max_seconds: int) -> tuple[np.ndarray, list[Candidate]]:
    t0 = time.perf_counter()
    current = pred.copy()
    selected = []
    used = set()

    for _ in range(max_rules):
        if time.perf_counter() - t0 >= max_seconds:
            break
        best = None
        current_alerted = current.astype(int) == 1
        for i, c in enumerate(cands[:1500]):
            if i in used:
                continue
            mask = c.mask & current_alerted
            tp_loss = int(((y == 1) & mask).sum())
            fp_removed = int(((y == 0) & mask).sum())
            if tp_loss != 0 or fp_removed <= 0:
                continue
            rank = (fp_removed, -int(mask.sum()))
            if best is None or rank > best[0]:
                best = (rank, i, c, mask, fp_removed)
        if best is None:
            break
        _, i, c, mask, fp_removed = best
        current[mask] = 0
        used.add(i)
        selected.append(Candidate(
            rule_id=c.rule_id,
            family=c.family,
            description=c.description,
            mask=mask.copy(),
            tp_loss=0,
            fp_effect=fp_removed,
            n_effect=int(mask.sum()),
            ratio=float("inf"),
            params=c.params,
        ))
    return current, selected


def add_rescue_candidate(rows, df, pred, mask, family, desc, params, min_fn_recovered, max_fp_added):
    y = df["is_fraud"].to_numpy(dtype=int)
    mask = mask & (pred.astype(int) == 0)
    fn_recovered = int(((y == 1) & mask).sum())
    fp_added = int(((y == 0) & mask).sum())
    if fn_recovered < min_fn_recovered or fp_added > max_fp_added:
        return
    rows.append({
        "candidate_id": f"r3n_rescue_{len(rows):05d}",
        "family": family,
        "description": desc,
        "fn_recovered": fn_recovered,
        "fp_added": fp_added,
        "n_added": int(mask.sum()),
        "fp_per_fn": fp_added / max(fn_recovered, 1),
        "params_json": json.dumps(params, ensure_ascii=False),
    })


def build_rescue_candidates(df, pred, rescue_libraries, min_fn_recovered, max_fp_added, max_new_combo_size, top_groups_per_combo):
    rows = []
    y = df["is_fraud"].to_numpy(dtype=int)
    not_alerted = pred.astype(int) == 0
    fn_idx = np.where((y == 1) & not_alerted)[0]

    for lib_name, lib_df in rescue_libraries:
        if lib_df is None or lib_df.empty or "params_json" not in lib_df.columns:
            continue
        for _, row in lib_df.iterrows():
            params = parse_params(row["params_json"])
            mask = rule_mask(df, pred, params, mode="rescue")
            add_rescue_candidate(rows, df, pred, mask, f"reused_{lib_name}", str(row.get("description")), params, min_fn_recovered, max_fp_added)

    # New numeric candidates from residual FNs.
    for c in NUMERIC_COLS:
        if c not in df.columns:
            continue
        vals = num(df, c, 0.0).to_numpy(dtype=float)
        fn_vals = vals[fn_idx]
        if len(fn_vals) == 0:
            continue
        try:
            cuts = sorted(set(float(x) for x in np.quantile(fn_vals, [0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]) if np.isfinite(x)))
        except Exception:
            cuts = []
        for cut in cuts:
            for direction in ["ge", "le"]:
                mask = not_alerted & ((vals >= cut) if direction == "ge" else (vals <= cut))
                desc = f"{c}>={cut:g}" if direction == "ge" else f"{c}<={cut:g}"
                add_rescue_candidate(rows, df, pred, mask, "new_numeric_residual_fn_rescue", desc, {"type": "numeric_threshold_rescue", "col": c, "direction": direction, "cut": cut}, min_fn_recovered, max_fp_added)

    # Targeted combo candidates: only values present in residual FNs.
    feat = feature_frame(df)
    cols = list(feat.columns)
    for r in range(1, max_new_combo_size + 1):
        for combo in itertools.combinations(cols, r):
            combo = list(combo)
            if r == 1 and combo[0] not in ["lgbm_bin", "if_bin", "score_bin", "ds_tipo_chave_norm", "value_band", "ratio_bin", "qtd_rec_bin", "periodo_dia"]:
                continue

            fn_keys = feat.iloc[fn_idx][combo].drop_duplicates()
            if fn_keys.empty:
                continue

            group_rows = []
            for _, keyrow in fn_keys.iterrows():
                mask = not_alerted.copy()
                vals = []
                for c in combo:
                    v = str(keyrow[c])
                    vals.append(v)
                    mask = mask & (feat[c].to_numpy() == v)
                fn_recovered = int(((y == 1) & mask).sum())
                fp_added = int(((y == 0) & mask).sum())
                if fn_recovered >= min_fn_recovered and fp_added <= max_fp_added:
                    group_rows.append((fp_added / max(fn_recovered, 1), -fn_recovered, vals, mask))
            group_rows.sort()
            for _, _, vals, mask in group_rows[:top_groups_per_combo]:
                desc = " AND ".join([f"{c}={v}" for c, v in zip(combo, vals)])
                add_rescue_candidate(rows, df, pred, mask, "new_combo_residual_fn_rescue", desc, {"type": "combo_rescue", "combo_cols": combo, "combo_values": vals}, min_fn_recovered, max_fp_added)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["params_json"]).reset_index(drop=True)
    out = out.sort_values(["fp_per_fn", "fp_added", "fn_recovered"], ascending=[True, True, False]).reset_index(drop=True)
    out["candidate_id"] = [f"r3n_rescue_{i:05d}" for i in range(len(out))]
    return out


def apply_rescue_row(df, current_pred, row):
    params = parse_params(row["params_json"])
    mask = rule_mask(df, current_pred, params, mode="rescue")
    y = df["is_fraud"].to_numpy(dtype=int)
    fn_recovered = int(((y == 1) & mask).sum())
    fp_added = int(((y == 0) & mask).sum())
    new_pred = current_pred.copy()
    new_pred[mask] = 1
    return new_pred, {"fn_recovered": fn_recovered, "fp_added": fp_added, "n_added": int(mask.sum())}


def greedy_rescue_for_caps(df, base_pred, rescue_candidates, fp_caps):
    y = df["is_fraud"].to_numpy(dtype=int)
    base_metrics = compute_metrics(y, base_pred)
    rows = []
    pred_by_scenario = {}
    candidates = rescue_candidates.head(2500).to_dict(orient="records") if not rescue_candidates.empty else []

    for cap in fp_caps:
        current = base_pred.copy()
        selected = []
        while True:
            current_metrics = compute_metrics(y, current)
            current_fp = current_metrics["fp"]
            best = None
            for cand in candidates:
                if cand["candidate_id"] in selected:
                    continue
                new_pred, gain = apply_rescue_row(df, current, cand)
                if gain["fn_recovered"] <= 0:
                    continue
                if current_fp + gain["fp_added"] > cap:
                    continue
                ratio = gain["fn_recovered"] / max(gain["fp_added"], 1)
                rank = (ratio, gain["fn_recovered"], -gain["fp_added"])
                if best is None or rank > best[0]:
                    best = (rank, cand, new_pred, gain)
            if best is None:
                break
            _, cand, new_pred, gain = best
            current = new_pred
            selected.append(cand["candidate_id"])

        m = compute_metrics(y, current)
        rows.append({
            "scenario": f"r3n_cap_{cap}",
            "fp_cap": int(cap),
            "n_selected_rescues": int(len(selected)),
            "selected_rescue_candidate_ids": "|".join(selected),
            "fn_recovered_vs_headroom": int(base_metrics["fn"] - m["fn"]),
            "fp_added_vs_headroom": int(m["fp"] - base_metrics["fp"]),
            **m,
        })
        pred_by_scenario[f"r3n_cap_{cap}"] = current

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["fn", "fp"], ascending=[True, True]).reset_index(drop=True)
    return out, pred_by_scenario


def run_retighten(df, headroom_pred, scenario_pred, min_fp_removed, max_combo_size, top_groups_per_combo, max_rules, max_seconds):
    y = df["is_fraud"].to_numpy(dtype=int)
    added = (scenario_pred.astype(int) == 1) & (headroom_pred.astype(int) == 0)
    cands = []

    for c in NUMERIC_COLS:
        if c not in df.columns:
            continue
        vals = num(df, c, 0.0).to_numpy(dtype=float)
        active = vals[added]
        if len(active) == 0:
            continue
        try:
            cuts = sorted(set(float(x) for x in np.quantile(active, [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]) if np.isfinite(x)))
        except Exception:
            cuts = []
        for cut in cuts:
            for direction in ["le", "ge"]:
                mask = added & ((vals <= cut) if direction == "le" else (vals >= cut))
                desc = f"added_only AND {c}<={cut:g}" if direction == "le" else f"added_only AND {c}>={cut:g}"
                add_tp0_candidate(cands, "retighten_numeric_added_only", desc, mask, y, min_fp_removed, {"type": "numeric_retighten", "scope": "rescue_added_only", "col": c, "direction": direction, "cut": cut})

    feat = feature_frame(df)
    cols = list(feat.columns)
    bins = [c for c in cols if c.endswith("_bin") or c == "module_quiet"]
    important = ["ds_tipo_chave_norm", "value_band", "mbk_available_flag", "first_receiver_flag_real", "periodo_dia"]
    idx = np.where(added)[0]
    for r in range(1, max_combo_size + 1):
        for combo in itertools.combinations(cols, r):
            combo = list(combo)
            if r == 1 and combo[0] not in bins + ["ds_tipo_chave_norm", "value_band"]:
                continue
            if r >= 2 and not any(c in combo for c in important + bins):
                continue
            sub = feat.iloc[idx][combo]
            if sub.empty:
                continue
            group_rows = []
            for key, rel_idxs in sub.groupby(combo, dropna=False).indices.items():
                idxs = sub.iloc[list(rel_idxs)].index.to_numpy(dtype=int)
                if len(idxs) < min_fp_removed:
                    continue
                mask = np.zeros(len(df), dtype=bool)
                mask[idxs] = True
                mask = mask & added
                tp_loss = int(((y == 1) & mask).sum())
                fp_removed = int(((y == 0) & mask).sum())
                if tp_loss == 0 and fp_removed >= min_fp_removed:
                    group_rows.append((-fp_removed, key, mask, fp_removed))
            group_rows.sort()
            for _, key, mask, fp_removed in group_rows[:top_groups_per_combo]:
                vals = key if isinstance(key, tuple) else (key,)
                vals = [str(v) for v in vals]
                desc = "added_only AND " + " AND ".join([f"{c}={v}" for c, v in zip(combo, vals)])
                add_tp0_candidate(cands, "retighten_combo_added_only", desc, mask, y, min_fp_removed, {"type": "combo_retighten", "scope": "rescue_added_only", "combo_cols": combo, "combo_values": vals})

    best = {}
    for c in cands:
        k = np.packbits(c.mask).tobytes()
        old = best.get(k)
        if old is None or c.fp_effect > old.fp_effect:
            best[k] = c
    cands = list(best.values())
    cands.sort(key=lambda c: (-c.fp_effect, len(c.description)))
    for i, c in enumerate(cands):
        c.rule_id = f"ret_{i:05d}"

    final_pred, selected = greedy_select_tp0(cands, scenario_pred, y, max_rules=max_rules, max_seconds=max_seconds)
    return final_pred, cands, selected


def select_recommended(after_df: pd.DataFrame, preferred_cap: int):
    if after_df.empty:
        raise RuntimeError("after_df vazio.")
    under = after_df[after_df["fp"] <= preferred_cap].copy()
    if not under.empty:
        return str(under.sort_values(["fn", "fp"], ascending=[True, True]).iloc[0]["scenario"])
    return str(after_df.sort_values(["fn", "fp"], ascending=[True, True]).iloc[0]["scenario"])


def make_report(summary, validation, base_metrics, headroom_metrics, scenario_df, after_df, head_rules_df, selected_rules_df):
    lines = []
    lines.append("# EXP-014B-R3N — Consolidate R3M-FROZEN + Irreducible FN Search")
    lines.append("")
    lines.append("## Parte A — Consolidação R3M")
    lines.append(f"- Status R3M frozen: `{validation['status']}`")
    lines.append(f"- Métricas R3M frozen: `{validation['final_metrics']}`")
    lines.append("")
    lines.append("## Parte B — Busca por FN irredutível")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Cenário recomendado: `{summary['recommended_scenario']}`")
    lines.append(f"- Métricas recomendadas: `{summary['recommended_metrics']}`")
    lines.append("")
    lines.append("## Base e headroom")
    lines.append(f"- Base R3M frozen: `{base_metrics}`")
    lines.append(f"- Após headroom: `{headroom_metrics}`")
    lines.append("")
    lines.append("## Cenários antes do re-tightening")
    show = ["scenario", "fp_cap", "fn_recovered_vs_headroom", "fp_added_vs_headroom", "tp", "fp", "fn", "precision", "recall", "fpr"]
    lines.append(scenario_df[[c for c in show if c in scenario_df.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Cenários após re-tightening")
    show2 = ["scenario", "net_fn_recovered_vs_base", "net_fp_delta_vs_base", "headroom_fp_removed", "retightening_fp_removed", "tp", "fp", "fn", "precision", "recall", "fpr"]
    lines.append(after_df[[c for c in show2 if c in after_df.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Headroom selecionado")
    if head_rules_df.empty:
        lines.append("Nenhuma regra de headroom selecionada.")
    else:
        show3 = ["rule_id", "family", "description", "fp_effect", "tp_loss"]
        lines.append(head_rules_df[[c for c in show3 if c in head_rules_df.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Retightening selecionado por cenário")
    if selected_rules_df.empty:
        lines.append("Nenhuma regra de re-tightening selecionada.")
    else:
        show4 = ["scenario", "rule_id", "family", "description", "fp_effect", "tp_loss"]
        lines.append(selected_rules_df[[c for c in show4 if c in selected_rules_df.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Decisão sugerida")
    lines.append("Se o R3N reduzir FN abaixo de 21 mantendo FP dentro do cap, executar validação congelada. Se o ganho começar a ficar pequeno ou exigir FP alto, iniciar auditoria dos FNs residuais e considerar hard-negative mining/segundo estágio.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--r3m-artifact", default=str(DEFAULT_R3M_ARTIFACT))
    parser.add_argument("--r3m-rescues", default=str(DEFAULT_R3M_RESCUES))
    parser.add_argument("--r3l-rescues", default=str(DEFAULT_R3L_RESCUES))
    parser.add_argument("--r3i-rescues", default=str(DEFAULT_R3I_RESCUES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fp-caps", default="5000,5050,5100,5200")
    parser.add_argument("--preferred-fp-cap", type=int, default=5000)
    parser.add_argument("--min-fp-removed-headroom", type=int, default=10)
    parser.add_argument("--max-headroom-rules", type=int, default=5)
    parser.add_argument("--max-seconds-headroom", type=int, default=120)
    parser.add_argument("--max-headroom-combo-size", type=int, default=3)
    parser.add_argument("--max-fp-added-candidate", type=int, default=650)
    parser.add_argument("--min-fn-recovered", type=int, default=1)
    parser.add_argument("--max-new-combo-size", type=int, default=4)
    parser.add_argument("--top-groups-per-combo", type=int, default=60)
    parser.add_argument("--min-fp-removed-retighten", type=int, default=5)
    parser.add_argument("--max-retighten-rules", type=int, default=5)
    parser.add_argument("--max-seconds-retighten", type=int, default=90)
    parser.add_argument("--no-write-predictions", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    artifact_path = Path(args.r3m_artifact)
    r3m_rescues_path = Path(args.r3m_rescues)
    r3l_rescues_path = Path(args.r3l_rescues)
    r3i_rescues_path = Path(args.r3i_rescues)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014B-R3N — Consolidate R3M-FROZEN + Irreducible FN Search")
    log("=" * 80)

    for p, name in [(input_path, "input"), (artifact_path, "r3m_artifact"), (r3m_rescues_path, "r3m_rescues")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} não encontrado: {p}")

    df = add_bins_and_guards(normalize_columns(pd.read_csv(input_path, low_memory=False)))
    artifact = load_json(artifact_path)
    r3m_rescues = pd.read_csv(r3m_rescues_path)
    r3l_rescues = pd.read_csv(r3l_rescues_path) if r3l_rescues_path.exists() else pd.DataFrame()
    r3i_rescues = pd.read_csv(r3i_rescues_path) if r3i_rescues_path.exists() else pd.DataFrame()

    missing = []
    if "is_fraud" not in df.columns:
        missing.append("is_fraud")
    if R3L_BASE_COL not in df.columns:
        missing.append(R3L_BASE_COL)
    if "params_json" not in r3m_rescues.columns:
        missing.append("r3m_rescues.params_json")
    contract = {
        "n_rows": int(len(df)),
        "n_frauds": int(df["is_fraud"].sum()) if "is_fraud" in df.columns else None,
        "r3l_base_col": R3L_BASE_COL,
        "r3m_artifact_path": str(artifact_path),
        "r3m_rescues_rows": int(len(r3m_rescues)),
        "r3l_rescues_rows": int(len(r3l_rescues)),
        "r3i_rescues_rows": int(len(r3i_rescues)),
        "missing": missing,
        "contract_ok": not missing,
    }
    dump_json(contract, output_dir / "01_input_contract.json")
    if missing:
        raise RuntimeError(f"Contrato falhou: {missing}")

    y = df["is_fraud"].to_numpy(dtype=int)

    log("[A] Validando R3M congelado...")
    r3m_pred, validation = validate_r3m_frozen(df, artifact, r3m_rescues, output_dir)
    df[R3M_FROZEN_COL] = r3m_pred.astype(int)

    # B) New FN search.
    base_pred = r3m_pred.copy()
    base_metrics = compute_metrics(y, base_pred)
    pd.DataFrame([{"policy_name": "R3M_FROZEN_BASE_FOR_R3N", **base_metrics}]).to_csv(output_dir / "05_r3n_base_metrics.csv", index=False)
    log(f"Base R3N metrics: {base_metrics}")

    log("[B1] Minerando headroom TP0...")
    head_cands = mine_tp0_veto_candidates(
        df=df,
        pred=base_pred,
        min_fp_removed=args.min_fp_removed_headroom,
        max_combo_size=args.max_headroom_combo_size,
        top_groups_per_combo=args.top_groups_per_combo,
        prefix_family="headn",
    )
    candidates_df(head_cands).to_csv(output_dir / "06_r3n_headroom_candidates.csv", index=False)

    head_pred, head_selected = greedy_select_tp0(
        head_cands,
        base_pred,
        y,
        max_rules=args.max_headroom_rules,
        max_seconds=args.max_seconds_headroom,
    )
    head_rules_df = candidates_df(head_selected)
    head_rules_df.to_csv(output_dir / "07_r3n_headroom_selected_rules.csv", index=False)
    head_metrics = compute_metrics(y, head_pred)
    pd.DataFrame([{"policy_name": "R3N_HEADROOM_BASE", **head_metrics}]).to_csv(output_dir / "08_r3n_headroom_metrics.csv", index=False)
    headroom_fp_removed = base_metrics["fp"] - head_metrics["fp"]
    log(f"Headroom metrics: {head_metrics}")

    log("[B2] Gerando rescues sobre FNs residuais...")
    rescue_candidates = build_rescue_candidates(
        df=df,
        pred=head_pred,
        rescue_libraries=[("r3m", r3m_rescues), ("r3l", r3l_rescues), ("r3i", r3i_rescues)],
        min_fn_recovered=args.min_fn_recovered,
        max_fp_added=args.max_fp_added_candidate,
        max_new_combo_size=args.max_new_combo_size,
        top_groups_per_combo=args.top_groups_per_combo,
    )
    rescue_candidates.to_csv(output_dir / "09_r3n_rescue_candidates.csv", index=False)
    log(f"Rescue candidates R3N: {len(rescue_candidates)}")

    fp_caps = [int(x.strip()) for x in str(args.fp_caps).split(",") if x.strip()]
    scenario_df, scenario_preds = greedy_rescue_for_caps(df, head_pred, rescue_candidates, fp_caps)
    scenario_df.to_csv(output_dir / "10_r3n_scenarios_before_retightening.csv", index=False)

    log("[B3] Re-tightening curto por cenário...")
    after_rows = []
    all_selected = []
    scenario_artifacts = {}

    for _, row in scenario_df.iterrows():
        scenario = str(row["scenario"])
        scenario_pred = scenario_preds[scenario]
        final_pred, ret_cands, ret_selected = run_retighten(
            df=df,
            headroom_pred=head_pred,
            scenario_pred=scenario_pred,
            min_fp_removed=args.min_fp_removed_retighten,
            max_combo_size=args.max_new_combo_size,
            top_groups_per_combo=args.top_groups_per_combo,
            max_rules=args.max_retighten_rules,
            max_seconds=args.max_seconds_retighten,
        )
        final_metrics = compute_metrics(y, final_pred)
        rescue_metrics = compute_metrics(y, scenario_pred)
        retightening_fp_removed = rescue_metrics["fp"] - final_metrics["fp"]
        retightening_tp_loss = rescue_metrics["tp"] - final_metrics["tp"]

        sel_df = candidates_df(ret_selected, scenario)
        all_selected.append(sel_df)

        selected_ids = str(row["selected_rescue_candidate_ids"]).split("|") if pd.notna(row.get("selected_rescue_candidate_ids")) and str(row.get("selected_rescue_candidate_ids")) else []

        after_rows.append({
            "scenario": scenario,
            "fp_cap": int(row["fp_cap"]),
            "headroom_fp_removed": int(headroom_fp_removed),
            "fn_recovered_before_retighten": int(row["fn_recovered_vs_headroom"]),
            "fp_added_before_retighten": int(row["fp_added_vs_headroom"]),
            "retightening_fp_removed": int(retightening_fp_removed),
            "retightening_tp_loss": int(retightening_tp_loss),
            "net_fn_recovered_vs_base": int(base_metrics["fn"] - final_metrics["fn"]),
            "net_fp_delta_vs_base": int(final_metrics["fp"] - base_metrics["fp"]),
            **final_metrics,
            "selected_rescue_candidate_ids": "|".join(selected_ids),
            "selected_retighten_rule_ids": "|".join(r.rule_id for r in ret_selected),
        })

        scenario_artifacts[scenario] = {
            "scenario": scenario,
            "fp_cap": int(row["fp_cap"]),
            "selected_headroom_rules": head_rules_df.to_dict(orient="records") if not head_rules_df.empty else [],
            "selected_rescue_candidate_ids": selected_ids,
            "selected_retighten_rules": sel_df.to_dict(orient="records") if not sel_df.empty else [],
            "metrics_after_headroom": head_metrics,
            "metrics_before_retightening": rescue_metrics,
            "final_metrics": final_metrics,
        }
        log(f"  {scenario}: final={final_metrics}")

    after_df = pd.DataFrame(after_rows)
    after_df.to_csv(output_dir / "11_r3n_scenario_metrics_after_retightening.csv", index=False)

    selected_rules_df = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    selected_rules_df.to_csv(output_dir / "12_r3n_selected_rules_by_scenario.csv", index=False)

    recommended_scenario = select_recommended(after_df, args.preferred_fp_cap)
    rec_scenario_pred = scenario_preds[recommended_scenario]
    rec_final_pred, _, _ = run_retighten(
        df=df,
        headroom_pred=head_pred,
        scenario_pred=rec_scenario_pred,
        min_fp_removed=args.min_fp_removed_retighten,
        max_combo_size=args.max_new_combo_size,
        top_groups_per_combo=args.top_groups_per_combo,
        max_rules=args.max_retighten_rules,
        max_seconds=args.max_seconds_retighten,
    )
    rec_metrics = compute_metrics(y, rec_final_pred)
    wl, wh = wilson_ci(rec_metrics["tp"], int(y.sum()))

    df[R3N_FINAL_COL] = rec_final_pred.astype(int)
    df["exp014b_r3n_recommended_scenario"] = recommended_scenario
    df[(df["is_fraud"] == 1) & (df[R3N_FINAL_COL] == 0)].to_csv(output_dir / "13_r3n_residual_fns_recommended.csv", index=False)

    objective_status = "DONE_R3M_FROZEN_CONSOLIDATED"
    objective_status += "_R3N_FN_IMPROVED_VS_R3M" if rec_metrics["fn"] < base_metrics["fn"] else "_R3N_FN_NOT_IMPROVED_VS_R3M"
    objective_status += "_FP_WITHIN_PREFERRED_CAP" if rec_metrics["fp"] <= args.preferred_fp_cap else "_FP_ABOVE_PREFERRED_CAP"
    objective_status += "_R3M_FROZEN_PASS" if validation["all_pass"] else "_R3M_FROZEN_NOT_PASS"

    artifact_out = {
        "experiment": "EXP-014B-R3N",
        "policy_name": "consolidate_r3m_and_irreducible_fn_search",
        "objective_status": objective_status,
        "r3m_frozen_validation": validation,
        "base_r3m_frozen_metrics": base_metrics,
        "headroom_metrics": head_metrics,
        "recommended_scenario": recommended_scenario,
        "recommended_metrics": rec_metrics,
        "wilson_low": wl,
        "wilson_high": wh,
        "scenario_artifacts": scenario_artifacts,
        "constraints": {
            "fp_caps": fp_caps,
            "preferred_fp_cap": args.preferred_fp_cap,
            "max_headroom_rules": args.max_headroom_rules,
            "max_retighten_rules": args.max_retighten_rules,
            "max_fp_added_candidate": args.max_fp_added_candidate,
            "max_new_combo_size": args.max_new_combo_size,
        },
        "notes": [
            "Part A consolidates R3M-FROZEN without new mining.",
            "Part B searches for residual FN reduction using headroom + rescue + retighten.",
            "If FN improvement becomes marginal or FP cost rises, stop optimization and audit residual FNs for irreducibility."
        ],
    }
    dump_json(artifact_out, output_dir / "14_policy_artifact_recommended.json")

    if not args.no_write_predictions:
        df.to_csv(output_dir / "15_predictions_recommended.csv", index=False)

    summary = {
        "experiment": "EXP-014B-R3N",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "r3m_artifact_path": str(artifact_path),
        "r3m_rescues_path": str(r3m_rescues_path),
        "r3l_rescues_path": str(r3l_rescues_path),
        "r3i_rescues_path": str(r3i_rescues_path),
        "n_rows": int(len(df)),
        "n_frauds": int(df["is_fraud"].sum()),
        "r3m_frozen_validation_status": validation["status"],
        "r3m_frozen_metrics": validation["final_metrics"],
        "base_r3m_frozen_metrics": base_metrics,
        "headroom_metrics": head_metrics,
        "headroom_fp_removed": int(headroom_fp_removed),
        "n_headroom_candidates": int(len(head_cands)),
        "n_headroom_selected": int(len(head_selected)),
        "n_rescue_candidates": int(len(rescue_candidates)),
        "fp_caps": fp_caps,
        "preferred_fp_cap": args.preferred_fp_cap,
        "recommended_scenario": recommended_scenario,
        "recommended_metrics": rec_metrics,
        "recommended_wilson_low": wl,
        "recommended_wilson_high": wh,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, validation, base_metrics, head_metrics, scenario_df, after_df, head_rules_df, selected_rules_df)
    (output_dir / "16_exp014b_r3n_report.md").write_text(report, encoding="utf-8")

    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_input_contract.json",
        output_dir / "02_r3m_frozen_validation.json",
        output_dir / "03_r3m_frozen_metrics.csv",
        output_dir / "04_r3m_frozen_rule_impact.csv",
        output_dir / "09_r3n_rescue_candidates.csv",
        output_dir / "11_r3n_scenario_metrics_after_retightening.csv",
        output_dir / "12_r3n_selected_rules_by_scenario.csv",
        output_dir / "13_r3n_residual_fns_recommended.csv",
        output_dir / "14_policy_artifact_recommended.json",
        output_dir / "16_exp014b_r3n_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
