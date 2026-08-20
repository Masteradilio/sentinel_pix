#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3J — Frozen Rescue Frontier + FP Re-tightening

Objetivo:
  Transformar a auditoria diagnóstica do EXP-014B-R3I em cenários congelados
  reprodutíveis, sem nova mineração longa de FN.

  O R3I mostrou que os 56 FNs residuais do R3H não são irredutíveis:

    R3H frozen:
      TP=1409, FP=4935, FN=56, recall=96,177%, precision=22,210%, FPR=4,391%

    Fronteira diagnóstica R3I:
      +100 FP  -> +13 TP, FN=43, FP=5025
      +250 FP  -> +19 TP, FN=37, FP=5181
      +500 FP  -> +27 TP, FN=29, FP=5418
      +1000 FP -> +38 TP, FN=18, FP=5908
      +2000 FP -> +51 TP, FN=5,  FP=6895

Este R3J faz:
  1. Lê as predições congeladas do R3H.
  2. Lê os candidatos de resgate e a fronteira greedy do R3I.
  3. Reaplica, sem nova mineração de resgate, os candidate_ids de cada cenário.
  4. Confirma as métricas congeladas de cada cenário.
  5. Para cada cenário, executa uma etapa curta de FP re-tightening:
       - atua somente sobre alertas adicionados pelo rescue;
       - por padrão remove apenas FPs adicionados com TP_loss=0;
       - tenta recuperar parte dos FPs adicionados sem perder FNs recuperados.
  6. Escolhe um candidato recomendado por política:
       - balanced: menor FN com FP aceitável;
       - rescue100_priority: prioriza o cenário +100 FP;
       - fp_lt_5000: tenta voltar abaixo de 5000 FP, se possível.

Uso padrão:
  python scripts/exp_014b_r3j_frozen_rescue_frontier_fp_retightening.py

Rodada curta recomendada:
  python scripts/exp_014b_r3j_frozen_rescue_frontier_fp_retightening.py --scenario-budgets 100,250,500 --max-seconds-per-scenario 120 --max-rules 5

Rodada mais exploratória, ainda curta:
  python scripts/exp_014b_r3j_frozen_rescue_frontier_fp_retightening.py --scenario-budgets 100,250,500,1000 --max-seconds-per-scenario 180 --max-rules 8 --max-combo-size 4

Somente reaplicar a fronteira congelada, sem re-tightening:
  python scripts/exp_014b_r3j_frozen_rescue_frontier_fp_retightening.py --skip-retightening

Saídas:
  resultados/experimentos/EXP-014B-R3J/
    00_run_summary.json
    01_input_contract.json
    02_r3i_frontier_replayed.csv
    03_scenario_metrics_before_retightening.csv
    04_retightening_candidate_summary.csv
    05_all_retightening_frontiers.csv
    06_scenario_metrics_after_retightening.csv
    07_selected_rules_by_scenario.csv
    08_policy_artifact_recommended.json
    09_predictions_recommended.csv
    10_exp014b_r3j_report.md
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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "dados").exists() else Path.cwd()

DEFAULT_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3H-FROZEN" / "10_predictions.csv"
DEFAULT_R3I_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3I"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3J"

BASE_COL = "exp014b_r3h_frozen_pred"
RECOMMENDED_COL = "exp014b_r3j_recommended_pred"

R3H_BENCHMARK = {
    "source": "EXP014B_R3H_FROZEN_FINAL",
    "tp": 1409,
    "fp": 4935,
    "fn": 56,
    "precision": 0.22209962,
    "recall": 0.96177474,
    "fpr": 0.0439139,
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
class VetoCandidate:
    rule_id: str
    family: str
    description: str
    mask: np.ndarray
    tp_loss: int
    fp_removed: int
    n_removed: int
    fp_per_tp: float
    params: dict[str, Any]


@dataclass
class State:
    mask: np.ndarray
    rule_indices: tuple[int, ...]
    tp_loss: int
    fp_removed: int


def log(msg: str) -> None:
    print(msg, flush=True)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


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

    for c in [BASE_COL]:
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
    margin = z * math.sqrt((phat * (1 - phat) / n) + (z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def make_contract(df: pd.DataFrame, rescue: pd.DataFrame, frontier: pd.DataFrame) -> dict[str, Any]:
    missing = []
    if "is_fraud" not in df.columns:
        missing.append("is_fraud")
    if BASE_COL not in df.columns:
        missing.append(BASE_COL)
    if rescue.empty:
        missing.append("r3i_rescue_candidates_nonempty")
    if frontier.empty:
        missing.append("r3i_rescue_frontier_nonempty")

    return {
        "n_rows": int(len(df)),
        "n_frauds": int(df["is_fraud"].sum()) if "is_fraud" in df.columns else None,
        "base_col": BASE_COL,
        "missing": missing,
        "feature_cols_present": [c for c in FEATURE_COLS if c in df.columns],
        "numeric_cols_present": [c for c in NUMERIC_COLS if c in df.columns],
        "n_rescue_candidates": int(len(rescue)),
        "n_frontier_rows": int(len(frontier)),
        "contract_ok": not missing,
    }


def parse_params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    return json.loads(str(raw).replace("Infinity", "1e999"))


def rescue_mask_from_params(df: pd.DataFrame, current_pred: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    not_alerted = current_pred.astype(int) == 0
    if params.get("type") == "numeric_threshold_rescue":
        c = params.get("col")
        if c not in df.columns:
            return np.zeros(len(df), dtype=bool)
        vals = num(df, c, 0.0).to_numpy(dtype=float)
        if params.get("direction") == "ge":
            return not_alerted & (vals >= float(params["cut"]))
        return not_alerted & (vals <= float(params["cut"]))

    if params.get("type") == "combo_rescue":
        mask = not_alerted.copy()
        for c, v in zip(params.get("combo_cols", []), params.get("combo_values", [])):
            if c not in df.columns:
                return np.zeros(len(df), dtype=bool)
            mask = mask & (df[c].astype("string").fillna("<MISSING>").astype(str).to_numpy() == str(v))
        return mask

    return np.zeros(len(df), dtype=bool)


def apply_rescue_ids(df: pd.DataFrame, base_pred: np.ndarray, rescue_df: pd.DataFrame, selected_ids: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    y = df["is_fraud"].to_numpy(dtype=int)
    pred = base_pred.copy().astype(int)
    selected_rows = rescue_df[rescue_df["candidate_id"].astype(str).isin(set(selected_ids))].copy()
    impact_rows = []

    for _, row in selected_rows.iterrows():
        params = parse_params(row["params_json"])
        mask = rescue_mask_from_params(df, pred, params)
        fn_recovered = int(((y == 1) & mask).sum())
        fp_added = int(((y == 0) & mask).sum())
        pred[mask] = 1
        impact_rows.append({
            "candidate_id": row["candidate_id"],
            "description": row.get("description"),
            "family": row.get("family"),
            "fn_recovered_replay": fn_recovered,
            "fp_added_replay": fp_added,
            "n_added_replay": int(mask.sum()),
            "expected_fn_recovered": row.get("fn_recovered"),
            "expected_fp_added": row.get("fp_added"),
            "params_json": row["params_json"],
        })

    return pred, pd.DataFrame(impact_rows)


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in FEATURE_COLS:
        if c in df.columns:
            out[c] = df[c].astype("string").fillna("<MISSING>").astype(str)
    return out


def add_veto_candidate(
    out: list[VetoCandidate],
    prefix: str,
    family: str,
    description: str,
    mask: np.ndarray,
    y: np.ndarray,
    max_tp_loss: int,
    min_fp_removed: int,
    min_fp_per_tp: float,
    params: dict[str, Any],
) -> None:
    if not mask.any():
        return

    tp_loss = int(((y == 1) & mask).sum())
    fp_removed = int(((y == 0) & mask).sum())
    if fp_removed < min_fp_removed:
        return
    if tp_loss > max_tp_loss:
        return
    fp_per_tp = float("inf") if tp_loss == 0 else fp_removed / max(tp_loss, 1)
    if tp_loss > 0 and fp_per_tp < min_fp_per_tp:
        return

    out.append(VetoCandidate(
        rule_id=f"{prefix}_{len(out):05d}",
        family=family,
        description=description,
        mask=mask,
        tp_loss=tp_loss,
        fp_removed=fp_removed,
        n_removed=int(mask.sum()),
        fp_per_tp=fp_per_tp,
        params=params,
    ))


def mine_retightening_candidates(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    scenario_pred: np.ndarray,
    max_tp_loss: int,
    min_fp_removed: int,
    max_combo_size: int,
    top_groups_per_combo: int,
    min_fp_per_tp: float,
) -> list[VetoCandidate]:
    """
    Mine vetoes only on rows added by the rescue layer:
      scenario_pred == 1 and base_pred == 0

    Default max_tp_loss=0 means it will only remove added FPs and preserve
    every recovered FN.
    """
    y = df["is_fraud"].to_numpy(dtype=int)
    added_alerts = (scenario_pred.astype(int) == 1) & (base_pred.astype(int) == 0)
    out: list[VetoCandidate] = []

    # Numeric vetoes on added alerts.
    for c in NUMERIC_COLS:
        if c not in df.columns:
            continue
        vals = num(df, c, 0.0).to_numpy(dtype=float)
        active_vals = vals[added_alerts]
        if len(active_vals) == 0:
            continue

        cuts = []
        try:
            cuts.extend([float(x) for x in np.quantile(active_vals, [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90])])
        except Exception:
            pass
        cuts = sorted(set(x for x in cuts if np.isfinite(x)))

        for cut in cuts:
            # Both directions, because added FP regions may be low-score or high-history.
            for direction in ["le", "ge"]:
                mask = added_alerts & ((vals <= cut) if direction == "le" else (vals >= cut))
                desc = f"added_only AND {c}<={cut:g}" if direction == "le" else f"added_only AND {c}>={cut:g}"
                add_veto_candidate(
                    out, "r3j_num", "retighten_numeric_added_only", desc, mask, y,
                    max_tp_loss, min_fp_removed, min_fp_per_tp,
                    {"type": "numeric_retighten", "scope": "rescue_added_only", "col": c, "direction": direction, "cut": cut},
                )

    # Combo vetoes on added alerts.
    feat = feature_frame(df)
    cols = list(feat.columns)
    important = ["ds_tipo_chave_norm", "value_band", "mbk_available_flag", "first_receiver_flag_real", "periodo_dia"]
    bins = [c for c in cols if c.endswith("_bin") or c == "module_quiet"]

    added_idx = np.where(added_alerts)[0]
    for r in range(1, max_combo_size + 1):
        for combo in itertools.combinations(cols, r):
            combo = list(combo)
            if r == 1 and combo[0] not in bins + ["ds_tipo_chave_norm", "value_band"]:
                continue
            if r >= 2 and not any(c in combo for c in important + bins):
                continue

            subset = feat.iloc[added_idx][combo]
            if subset.empty:
                continue

            group_rows = []
            grouped = subset.groupby(combo, dropna=False).indices
            for key, rel_idxs in grouped.items():
                idxs = subset.iloc[list(rel_idxs)].index.to_numpy(dtype=int)
                if len(idxs) < min_fp_removed:
                    continue
                mask = np.zeros(len(df), dtype=bool)
                mask[idxs] = True
                mask = mask & added_alerts

                tp_loss = int(((y == 1) & mask).sum())
                fp_removed = int(((y == 0) & mask).sum())
                if fp_removed < min_fp_removed or tp_loss > max_tp_loss:
                    continue
                fp_per_tp = float("inf") if tp_loss == 0 else fp_removed / max(tp_loss, 1)
                if tp_loss > 0 and fp_per_tp < min_fp_per_tp:
                    continue
                group_rows.append((tp_loss, -fp_removed, key, mask, fp_removed, fp_per_tp))

            if not group_rows:
                continue

            group_rows.sort()
            for tp_loss, neg_fp, key, mask, fp_removed, fp_per_tp in group_rows[:top_groups_per_combo]:
                vals = key if isinstance(key, tuple) else (key,)
                vals = [str(v) for v in vals]
                desc = "added_only AND " + " AND ".join([f"{c}={v}" for c, v in zip(combo, vals)])
                add_veto_candidate(
                    out, "r3j_combo", "retighten_microsegment_added_only", desc, mask, y,
                    max_tp_loss, min_fp_removed, min_fp_per_tp,
                    {"type": "combo_retighten", "scope": "rescue_added_only", "combo_cols": combo, "combo_values": vals},
                )

    # Deduplicate by mask.
    best: dict[bytes, VetoCandidate] = {}
    for c in out:
        key = np.packbits(c.mask).tobytes()
        old = best.get(key)
        if old is None or (c.fp_removed, -c.tp_loss, -len(c.description)) > (old.fp_removed, -old.tp_loss, -len(old.description)):
            best[key] = c

    out = list(best.values())
    out.sort(key=lambda c: (c.tp_loss > 0, c.tp_loss, -c.fp_removed, -c.fp_per_tp))
    return out


def candidates_df(cands: list[VetoCandidate], scenario: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "scenario": scenario,
        "candidate_index": i,
        "rule_id": c.rule_id,
        "family": c.family,
        "description": c.description,
        "tp_loss": c.tp_loss,
        "fp_removed": c.fp_removed,
        "n_removed": c.n_removed,
        "fp_per_tp": c.fp_per_tp,
        "params_json": json.dumps(c.params, ensure_ascii=False),
    } for i, c in enumerate(cands)])


def search_vetos(
    cands: list[VetoCandidate],
    scenario_pred: np.ndarray,
    y: np.ndarray,
    max_tp_loss: int,
    max_candidates: int,
    beam_width: int,
    max_rules: int,
    max_seconds: int,
    scenario: str,
) -> tuple[pd.DataFrame, State, list[VetoCandidate], str]:
    t0 = time.perf_counter()
    usable = [c for c in cands if c.tp_loss <= max_tp_loss]
    usable.sort(key=lambda c: (c.tp_loss > 0, -c.fp_removed if c.tp_loss == 0 else -c.fp_per_tp, -c.fp_removed))
    usable = usable[:max_candidates]

    fraud_idx = np.where(y == 1)[0]
    zero_loss_mode = max_tp_loss == 0 and all(c.tp_loss == 0 for c in usable)

    def rank(s: State):
        return (s.fp_removed, -s.tp_loss, -len(s.rule_indices))

    pending_limit = max(beam_width * 8, 1000)
    pending_keep = max(beam_width * 4, 500)

    def prune(d):
        if len(d) <= pending_keep:
            return d
        return dict(sorted(d.items(), key=lambda kv: rank(kv[1]), reverse=True)[:pending_keep])

    initial = State(np.zeros(len(y), dtype=bool), tuple(), 0, 0)
    states = [initial]
    best = initial
    rows = []
    stop_reason = "completed"

    for depth in range(1, max_rules + 1):
        if time.perf_counter() - t0 >= max_seconds:
            stop_reason = f"max_seconds_before_depth_{depth}"
            break

        next_states: dict[bytes, State] = {}
        for state in states:
            last = state.rule_indices[-1] if state.rule_indices else -1
            old_total = state.tp_loss + state.fp_removed

            for i in range(last + 1, len(usable)):
                new_mask = state.mask | usable[i].mask
                total = int(new_mask.sum())
                if total <= old_total:
                    continue

                if zero_loss_mode:
                    tp_loss = 0
                    fp_removed = total
                else:
                    tp_loss = int(new_mask[fraud_idx].sum()) if len(fraud_idx) else 0
                    if tp_loss > max_tp_loss:
                        continue
                    fp_removed = total - tp_loss

                if fp_removed <= state.fp_removed:
                    continue

                ns = State(new_mask, state.rule_indices + (i,), tp_loss, fp_removed)
                key = np.packbits(new_mask).tobytes()
                old = next_states.get(key)
                if old is None or rank(ns) > rank(old):
                    next_states[key] = ns

                if len(next_states) > pending_limit:
                    next_states = prune(next_states)

            if time.perf_counter() - t0 >= max_seconds:
                stop_reason = f"max_seconds_during_depth_{depth}"
                break

        if not next_states:
            if not stop_reason.startswith("max_seconds"):
                stop_reason = f"no_next_states_at_depth_{depth}"
            break

        states = sorted(next_states.values(), key=rank, reverse=True)[:beam_width]
        if rank(states[0]) > rank(best):
            best = states[0]

        for s in states[:50]:
            pred = scenario_pred.copy()
            pred[s.mask] = 0
            rows.append({
                "scenario": scenario,
                "depth": depth,
                "tp_loss": s.tp_loss,
                "fp_removed": s.fp_removed,
                "n_rules": len(s.rule_indices),
                **compute_metrics(y, pred),
                "rule_ids": "|".join(usable[i].rule_id for i in s.rule_indices),
                "rule_descriptions": " || ".join(usable[i].description for i in s.rule_indices),
            })

        if time.perf_counter() - t0 >= max_seconds:
            stop_reason = f"max_seconds_after_depth_{depth}"
            break

    if not rows:
        rows = [{
            "scenario": scenario,
            "depth": 0,
            "tp_loss": 0,
            "fp_removed": 0,
            "n_rules": 0,
            **compute_metrics(y, scenario_pred),
            "rule_ids": "",
            "rule_descriptions": "",
        }]

    frontier = pd.DataFrame(rows).sort_values(["fn", "fp"], ascending=[True, True]).reset_index(drop=True)
    selected = [usable[i] for i in best.rule_indices]
    return frontier, best, selected, stop_reason


def apply_retighten_rules(pred: np.ndarray, rules: list[VetoCandidate]) -> np.ndarray:
    out = pred.copy()
    if rules:
        mask = np.zeros(len(pred), dtype=bool)
        for r in rules:
            mask = mask | r.mask
        out[mask] = 0
    return out


def extract_scenarios(frontier: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    out = frontier[frontier["fp_budget"].astype(int).isin(set(budgets))].copy()
    if out.empty:
        return out
    out = out.sort_values(["fp_budget"]).reset_index(drop=True)
    return out


def select_recommended(metrics_after: pd.DataFrame, selection_policy: str) -> str:
    if metrics_after.empty:
        raise RuntimeError("metrics_after vazio.")

    df = metrics_after.copy()
    if selection_policy == "rescue100_priority":
        s = df[df["scenario"].astype(str) == "rescue_budget_100"]
        if not s.empty:
            return str(s.sort_values(["fp", "fn"]).iloc[0]["scenario"])
    if selection_policy == "fp_lt_5000":
        s = df[df["fp"] < 5000]
        if not s.empty:
            # Among policies under 5000 FP, maximize TP/FN recovery.
            return str(s.sort_values(["fn", "fp"], ascending=[True, True]).iloc[0]["scenario"])
    # balanced: prefer meaningful FN reduction, then reasonable FP.
    # Sort by: fn, fp, precision desc.
    return str(df.sort_values(["fn", "fp", "precision"], ascending=[True, True, False]).iloc[0]["scenario"])


def make_report(summary: dict[str, Any], before: pd.DataFrame, after: pd.DataFrame, selected_rules: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014B-R3J — Frozen Rescue Frontier + FP Re-tightening")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Cenário recomendado: `{summary['recommended_scenario']}`")
    lines.append(f"- Métricas recomendadas: `{summary['recommended_metrics']}`")
    lines.append("")
    lines.append("## Cenários antes do re-tightening")
    show = ["scenario", "fp_budget", "fn_recovered_vs_r3h", "fp_added_vs_r3h", "tp", "fp", "fn", "precision", "recall", "fpr"]
    lines.append(before[[c for c in show if c in before.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Cenários após re-tightening")
    show_after = ["scenario", "retightening_fp_removed", "retightening_tp_loss", "tp", "fp", "fn", "precision", "recall", "fpr", "stop_reason"]
    lines.append(after[[c for c in show_after if c in after.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Regras selecionadas de re-tightening")
    if selected_rules.empty:
        lines.append("Nenhuma regra selecionada.")
    else:
        show_rules = ["scenario", "rule_id", "family", "description", "tp_loss", "fp_removed", "fp_per_tp"]
        lines.append(selected_rules[[c for c in show_rules if c in selected_rules.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretação")
    lines.append("O R3J é congelado quanto aos resgates: ele reaplica candidate_ids do R3I e só tenta re-tightening curto sobre os alertas adicionados. Qualquer cenário recomendado ainda deve passar por validação congelada antes de promoção.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--r3i-dir", default=str(DEFAULT_R3I_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scenario-budgets", default="100,250,500,1000,2000")
    parser.add_argument("--skip-retightening", action="store_true")
    parser.add_argument("--max-tp-loss-retighten", type=int, default=0)
    parser.add_argument("--min-fp-removed-retighten", type=int, default=10)
    parser.add_argument("--max-combo-size", type=int, default=3)
    parser.add_argument("--top-groups-per-combo", type=int, default=40)
    parser.add_argument("--min-fp-per-tp", type=float, default=500.0)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--beam-width", type=int, default=140)
    parser.add_argument("--max-rules", type=int, default=5)
    parser.add_argument("--max-seconds-per-scenario", type=int, default=120)
    parser.add_argument("--selection-policy", choices=["balanced", "rescue100_priority", "fp_lt_5000"], default="rescue100_priority")
    parser.add_argument("--no-write-predictions", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    r3i_dir = Path(args.r3i_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014B-R3J — Frozen Rescue Frontier + FP Re-tightening")
    log("=" * 80)

    rescue_path = r3i_dir / "07_rescue_candidates.csv"
    frontier_path = r3i_dir / "08_rescue_frontier_greedy.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")
    if not rescue_path.exists():
        raise FileNotFoundError(f"Rescue candidates não encontrado: {rescue_path}")
    if not frontier_path.exists():
        raise FileNotFoundError(f"Frontier R3I não encontrada: {frontier_path}")

    df = add_bins_and_guards(normalize_columns(pd.read_csv(input_path, low_memory=False)))
    rescue = pd.read_csv(rescue_path)
    frontier_r3i = pd.read_csv(frontier_path)

    contract = make_contract(df, rescue, frontier_r3i)
    dump_json(contract, output_dir / "01_input_contract.json")
    if not contract["contract_ok"]:
        raise RuntimeError(f"Contrato falhou: {contract['missing']}")

    y = df["is_fraud"].to_numpy(dtype=int)
    base_pred = df[BASE_COL].to_numpy(dtype=int)
    base_metrics = compute_metrics(y, base_pred)

    budgets = [int(x.strip()) for x in str(args.scenario_budgets).split(",") if x.strip()]
    scenarios = extract_scenarios(frontier_r3i, budgets)
    if scenarios.empty:
        raise RuntimeError(f"Nenhum cenário encontrado para budgets={budgets} em {frontier_path}")

    scenarios.to_csv(output_dir / "02_r3i_frontier_replayed.csv", index=False)

    before_rows = []
    after_rows = []
    all_frontiers = []
    all_candidates_summary = []
    all_selected_rules = []
    scenario_predictions: dict[str, np.ndarray] = {}
    scenario_artifacts: dict[str, Any] = {}

    for _, row in scenarios.iterrows():
        budget = int(row["fp_budget"])
        scenario_name = f"rescue_budget_{budget}"
        log(f"\n[Scenario] {scenario_name}")

        selected_ids = str(row["selected_candidate_ids"]).split("|") if pd.notna(row.get("selected_candidate_ids")) else []
        rescue_pred, rescue_impact = apply_rescue_ids(df, base_pred, rescue, selected_ids)
        rescue_metrics = compute_metrics(y, rescue_pred)
        rescue_impact.to_csv(output_dir / f"impact_{scenario_name}.csv", index=False)

        fn_recovered = base_metrics["fn"] - rescue_metrics["fn"]
        fp_added = rescue_metrics["fp"] - base_metrics["fp"]

        before_rows.append({
            "scenario": scenario_name,
            "fp_budget": budget,
            "n_selected_rescue_candidates": len(selected_ids),
            "fn_recovered_vs_r3h": int(fn_recovered),
            "fp_added_vs_r3h": int(fp_added),
            **rescue_metrics,
            "selected_rescue_candidate_ids": "|".join(selected_ids),
        })

        if args.skip_retightening:
            final_pred = rescue_pred.copy()
            final_metrics = rescue_metrics
            selected_retighten = []
            retighten_frontier = pd.DataFrame([{
                "scenario": scenario_name,
                "depth": 0,
                "tp_loss": 0,
                "fp_removed": 0,
                "n_rules": 0,
                **final_metrics,
                "rule_ids": "",
                "rule_descriptions": "",
            }])
            stop_reason = "skipped"
            cands = []
        else:
            cands = mine_retightening_candidates(
                df=df,
                base_pred=base_pred,
                scenario_pred=rescue_pred,
                max_tp_loss=args.max_tp_loss_retighten,
                min_fp_removed=args.min_fp_removed_retighten,
                max_combo_size=args.max_combo_size,
                top_groups_per_combo=args.top_groups_per_combo,
                min_fp_per_tp=args.min_fp_per_tp,
            )
            cand_df = candidates_df(cands, scenario_name)
            cand_df.to_csv(output_dir / f"candidates_{scenario_name}.csv", index=False)

            retighten_frontier, best, selected_retighten, stop_reason = search_vetos(
                cands=cands,
                scenario_pred=rescue_pred,
                y=y,
                max_tp_loss=args.max_tp_loss_retighten,
                max_candidates=args.max_candidates,
                beam_width=args.beam_width,
                max_rules=args.max_rules,
                max_seconds=args.max_seconds_per_scenario,
                scenario=scenario_name,
            )
            final_pred = rescue_pred.copy()
            final_pred[best.mask] = 0
            final_metrics = compute_metrics(y, final_pred)

        retighten_frontier.to_csv(output_dir / f"frontier_{scenario_name}.csv", index=False)
        all_frontiers.append(retighten_frontier)

        rules_df = candidates_df(selected_retighten, scenario_name)
        all_selected_rules.append(rules_df)

        retightening_fp_removed = rescue_metrics["fp"] - final_metrics["fp"]
        retightening_tp_loss = rescue_metrics["tp"] - final_metrics["tp"]

        after_rows.append({
            "scenario": scenario_name,
            "fp_budget": budget,
            "n_selected_rescue_candidates": len(selected_ids),
            "fn_recovered_vs_r3h_before_retighten": int(fn_recovered),
            "fp_added_vs_r3h_before_retighten": int(fp_added),
            "retightening_fp_removed": int(retightening_fp_removed),
            "retightening_tp_loss": int(retightening_tp_loss),
            "net_fn_recovered_vs_r3h": int(base_metrics["fn"] - final_metrics["fn"]),
            "net_fp_added_vs_r3h": int(final_metrics["fp"] - base_metrics["fp"]),
            "n_selected_retighten_rules": int(len(selected_retighten)),
            "stop_reason": stop_reason,
            **final_metrics,
            "selected_rescue_candidate_ids": "|".join(selected_ids),
            "selected_retighten_rule_ids": "|".join(r.rule_id for r in selected_retighten),
        })

        all_candidates_summary.append({
            "scenario": scenario_name,
            "n_retightening_candidates": int(len(cands)),
            "n_selected_retighten_rules": int(len(selected_retighten)),
            "stop_reason": stop_reason,
        })

        scenario_predictions[scenario_name] = final_pred
        scenario_artifacts[scenario_name] = {
            "scenario": scenario_name,
            "fp_budget": budget,
            "selected_rescue_candidate_ids": selected_ids,
            "rescue_metrics_before_retightening": rescue_metrics,
            "final_metrics": final_metrics,
            "selected_retighten_rules": rules_df.to_dict(orient="records") if not rules_df.empty else [],
            "stop_reason": stop_reason,
        }

        log(f"  before={rescue_metrics}")
        log(f"  after={final_metrics}")

    before_df = pd.DataFrame(before_rows)
    before_df.to_csv(output_dir / "03_scenario_metrics_before_retightening.csv", index=False)

    pd.DataFrame(all_candidates_summary).to_csv(output_dir / "04_retightening_candidate_summary.csv", index=False)

    all_frontiers_df = pd.concat(all_frontiers, ignore_index=True) if all_frontiers else pd.DataFrame()
    all_frontiers_df.to_csv(output_dir / "05_all_retightening_frontiers.csv", index=False)

    after_df = pd.DataFrame(after_rows)
    after_df.to_csv(output_dir / "06_scenario_metrics_after_retightening.csv", index=False)

    selected_rules_df = pd.concat(all_selected_rules, ignore_index=True) if all_selected_rules else pd.DataFrame()
    selected_rules_df.to_csv(output_dir / "07_selected_rules_by_scenario.csv", index=False)

    recommended_scenario = select_recommended(after_df, args.selection_policy)
    recommended_pred = scenario_predictions[recommended_scenario]
    recommended_metrics = compute_metrics(y, recommended_pred)
    wl, wh = wilson_ci(recommended_metrics["tp"], int(y.sum()))

    df[RECOMMENDED_COL] = recommended_pred.astype(int)
    df["exp014b_r3j_recommended_scenario"] = recommended_scenario

    recommended_artifact = {
        "experiment": "EXP-014B-R3J",
        "policy_name": "frozen_rescue_frontier_fp_retightening",
        "selection_policy": args.selection_policy,
        "recommended_scenario": recommended_scenario,
        "base_r3h_metrics": base_metrics,
        "recommended_metrics": recommended_metrics,
        "wilson_low": wl,
        "wilson_high": wh,
        "scenario_artifacts": scenario_artifacts,
        "constraints": {
            "scenario_budgets": budgets,
            "skip_retightening": args.skip_retightening,
            "max_tp_loss_retighten": args.max_tp_loss_retighten,
            "min_fp_removed_retighten": args.min_fp_removed_retighten,
            "max_seconds_per_scenario": args.max_seconds_per_scenario,
        },
        "notes": [
            "Rescue layer is frozen from R3I selected_candidate_ids.",
            "Re-tightening only applies to rescue-added alerts.",
            "Diagnostic/promotion-candidate only. Needs frozen validation before promotion."
        ],
    }
    dump_json(recommended_artifact, output_dir / "08_policy_artifact_recommended.json")

    if not args.no_write_predictions:
        df.to_csv(output_dir / "09_predictions_recommended.csv", index=False)

    objective_status = "DONE"
    objective_status += "_RESCUE_FRONTIER_REPLAYED"
    objective_status += "_RETIGHTEN_SKIPPED" if args.skip_retightening else "_RETIGHTEN_DONE"
    if recommended_metrics["fn"] < base_metrics["fn"]:
        objective_status += "_FN_IMPROVED_VS_R3H"
    else:
        objective_status += "_FN_NOT_IMPROVED_VS_R3H"
    if recommended_metrics["fp"] <= 5000:
        objective_status += "_FP_LE_5000"
    else:
        objective_status += "_FP_GT_5000"

    summary = {
        "experiment": "EXP-014B-R3J",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "r3i_dir": str(r3i_dir),
        "n_rows": int(len(df)),
        "n_frauds": int(y.sum()),
        "base_metrics": base_metrics,
        "scenario_budgets": budgets,
        "selection_policy": args.selection_policy,
        "recommended_scenario": recommended_scenario,
        "recommended_metrics": recommended_metrics,
        "recommended_wilson_low": wl,
        "recommended_wilson_high": wh,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, before_df, after_df, selected_rules_df)
    (output_dir / "10_exp014b_r3j_report.md").write_text(report, encoding="utf-8")

    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_input_contract.json",
        output_dir / "02_r3i_frontier_replayed.csv",
        output_dir / "03_scenario_metrics_before_retightening.csv",
        output_dir / "04_retightening_candidate_summary.csv",
        output_dir / "05_all_retightening_frontiers.csv",
        output_dir / "06_scenario_metrics_after_retightening.csv",
        output_dir / "07_selected_rules_by_scenario.csv",
        output_dir / "08_policy_artifact_recommended.json",
        output_dir / "10_exp014b_r3j_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
