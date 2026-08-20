#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3K — Reuse Rescue Library on R3J-FROZEN + short FP Second

Objetivo:
  Continuar a estratégia FN First / FP Second sem repetir mineração longa.

Base congelada:
  EXP014B_R3J_FROZEN_FINAL
  TP=1422
  FP=4965
  FN=43
  recall=97,065%
  precision=22,264%
  FPR=4,418%
  Wilson low≈96,07%

Ideia:
  O R3I já criou uma biblioteca de 2.851 candidatos de resgate.
  Em vez de minerar tudo de novo, este R3K reaproveita essa biblioteca sobre
  a nova base R3J-FROZEN, reavalia o ganho marginal dos candidatos ainda úteis
  e monta cenários curtos por orçamento de FP.

Fluxo:
  1. Carrega R3J-FROZEN/10_predictions.csv.
  2. Usa exp014b_r3j_frozen_pred como base.
  3. Carrega R3I/07_rescue_candidates.csv.
  4. Reavalia candidatos sobre os 43 FNs residuais.
  5. Monta fronteira greedy com budgets pequenos.
  6. Executa re-tightening curto somente nos alertas adicionados.
  7. Seleciona candidato recomendado, preferindo FP <= cap.

Uso recomendado:
  python scripts/exp_014b_r3k_reuse_rescue_library_microevolution.py --fp-budgets 25,50,100,250 --fp-cap 5000 --max-seconds-per-scenario 90

Execução somente replay, sem re-tightening:
  python scripts/exp_014b_r3k_reuse_rescue_library_microevolution.py --skip-retightening

Saídas:
  resultados/experimentos/EXP-014B-R3K/
    00_run_summary.json
    01_input_contract.json
    02_base_metrics.csv
    03_rescue_library_replayed.csv
    04_rescue_frontier_before_retightening.csv
    05_retightening_candidate_summary.csv
    06_all_retightening_frontiers.csv
    07_scenario_metrics_after_retightening.csv
    08_selected_rules_by_scenario.csv
    09_policy_artifact_recommended.json
    10_predictions_recommended.csv
    11_exp014b_r3k_report.md
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

DEFAULT_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3J-FROZEN" / "10_predictions.csv"
DEFAULT_RESCUE_CANDIDATES = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3I" / "07_rescue_candidates.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3K"

BASE_COL = "exp014b_r3j_frozen_pred"
FINAL_COL = "exp014b_r3k_recommended_pred"

R3J_FROZEN_BENCHMARK = {
    "tp": 1422,
    "fp": 4965,
    "fn": 43,
    "precision": 0.22263974,
    "recall": 0.97064846,
    "fpr": 0.04418085,
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
    margin = z * ((phat * (1 - phat) / n) + (z**2 / (4 * n**2))) ** 0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


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
        return not_alerted & ((vals >= float(params["cut"])) if params.get("direction") == "ge" else (vals <= float(params["cut"])))

    if params.get("type") == "combo_rescue":
        mask = not_alerted.copy()
        for c, v in zip(params.get("combo_cols", []), params.get("combo_values", [])):
            if c not in df.columns:
                return np.zeros(len(df), dtype=bool)
            mask = mask & (df[c].astype("string").fillna("<MISSING>").astype(str).to_numpy() == str(v))
        return mask

    return np.zeros(len(df), dtype=bool)


def evaluate_rescue_library(df: pd.DataFrame, base_pred: np.ndarray, rescue_df: pd.DataFrame, max_fp_added_candidate: int, min_fn_recovered: int) -> pd.DataFrame:
    y = df["is_fraud"].to_numpy(dtype=int)
    rows = []
    for _, row in rescue_df.iterrows():
        params = parse_params(row["params_json"])
        mask = rescue_mask_from_params(df, base_pred, params)
        fn_recovered = int(((y == 1) & mask).sum())
        fp_added = int(((y == 0) & mask).sum())
        if fn_recovered < min_fn_recovered:
            continue
        if fp_added > max_fp_added_candidate:
            continue
        rows.append({
            "candidate_id": str(row["candidate_id"]),
            "family": row.get("family"),
            "description": row.get("description"),
            "fn_recovered": fn_recovered,
            "fp_added": fp_added,
            "n_added": int(mask.sum()),
            "fp_per_fn": fp_added / max(fn_recovered, 1),
            "source_fn_recovered": row.get("fn_recovered"),
            "source_fp_added": row.get("fp_added"),
            "params_json": row["params_json"],
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["params_json"]).reset_index(drop=True)
    out = out.sort_values(["fp_per_fn", "fp_added", "fn_recovered"], ascending=[True, True, False]).reset_index(drop=True)
    return out


def apply_rescue_candidate(df: pd.DataFrame, current_pred: np.ndarray, cand: dict[str, Any]) -> tuple[np.ndarray, dict[str, int]]:
    y = df["is_fraud"].to_numpy(dtype=int)
    params = parse_params(cand["params_json"])
    mask = rescue_mask_from_params(df, current_pred, params)
    fn_recovered = int(((y == 1) & mask).sum())
    fp_added = int(((y == 0) & mask).sum())
    new_pred = current_pred.copy()
    new_pred[mask] = 1
    return new_pred, {"fn_recovered": fn_recovered, "fp_added": fp_added, "n_added": int(mask.sum())}


def greedy_rescue_frontier(df: pd.DataFrame, base_pred: np.ndarray, candidates: pd.DataFrame, fp_budgets: list[int]) -> pd.DataFrame:
    rows = []
    y = df["is_fraud"].to_numpy(dtype=int)
    cand_rows = candidates.head(1000).to_dict(orient="records")

    for budget in fp_budgets:
        current = base_pred.copy()
        selected = []
        total_fp_added = 0
        total_fn_recovered = 0

        while True:
            best = None
            for cand in cand_rows:
                if cand["candidate_id"] in selected:
                    continue
                _, gain = apply_rescue_candidate(df, current, cand)
                if gain["fn_recovered"] <= 0:
                    continue
                if total_fp_added + gain["fp_added"] > budget:
                    continue
                # Favor low FP/FN, then absolute FN recovered, then lower FP.
                ratio = gain["fn_recovered"] / max(gain["fp_added"], 1)
                rank = (ratio, gain["fn_recovered"], -gain["fp_added"])
                if best is None or rank > best[0]:
                    best = (rank, cand, gain)
            if best is None:
                break
            _, cand, gain = best
            current, gain2 = apply_rescue_candidate(df, current, cand)
            selected.append(cand["candidate_id"])
            total_fp_added += gain2["fp_added"]
            total_fn_recovered += gain2["fn_recovered"]

        m = compute_metrics(y, current)
        rows.append({
            "scenario": f"r3k_rescue_budget_{budget}",
            "fp_budget": int(budget),
            "fn_recovered_vs_base": int(total_fn_recovered),
            "fp_added_vs_base": int(total_fp_added),
            "n_selected_rescue_candidates": int(len(selected)),
            "selected_rescue_candidate_ids": "|".join(selected),
            **m,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["fn", "fp"], ascending=[True, True]).reset_index(drop=True)
    return out


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in FEATURE_COLS:
        if c in df.columns:
            out[c] = df[c].astype("string").fillna("<MISSING>").astype(str)
    return out


def add_veto_candidate(out, prefix, family, description, mask, y, max_tp_loss, min_fp_removed, min_fp_per_tp, params):
    if not mask.any():
        return
    tp_loss = int(((y == 1) & mask).sum())
    fp_removed = int(((y == 0) & mask).sum())
    if fp_removed < min_fp_removed or tp_loss > max_tp_loss:
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


def mine_retightening_candidates(df, base_pred, scenario_pred, max_tp_loss, min_fp_removed, max_combo_size, top_groups_per_combo, min_fp_per_tp):
    y = df["is_fraud"].to_numpy(dtype=int)
    added_alerts = (scenario_pred.astype(int) == 1) & (base_pred.astype(int) == 0)
    out = []

    # Numeric candidates, both directions.
    for c in NUMERIC_COLS:
        if c not in df.columns:
            continue
        vals = num(df, c, 0.0).to_numpy(dtype=float)
        active = vals[added_alerts]
        if len(active) == 0:
            continue
        try:
            cuts = sorted(set(float(x) for x in np.quantile(active, [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]) if np.isfinite(x)))
        except Exception:
            cuts = []
        for cut in cuts:
            for direction in ["le", "ge"]:
                mask = added_alerts & ((vals <= cut) if direction == "le" else (vals >= cut))
                desc = f"added_only AND {c}<={cut:g}" if direction == "le" else f"added_only AND {c}>={cut:g}"
                add_veto_candidate(
                    out, "r3k_num", "retighten_numeric_added_only", desc, mask, y,
                    max_tp_loss, min_fp_removed, min_fp_per_tp,
                    {"type": "numeric_retighten", "scope": "rescue_added_only", "col": c, "direction": direction, "cut": cut},
                )

    # Combo candidates.
    feat = feature_frame(df)
    cols = list(feat.columns)
    bins = [c for c in cols if c.endswith("_bin") or c == "module_quiet"]
    important = ["ds_tipo_chave_norm", "value_band", "mbk_available_flag", "first_receiver_flag_real", "periodo_dia"]
    added_idx = np.where(added_alerts)[0]

    for r in range(1, max_combo_size + 1):
        for combo in itertools.combinations(cols, r):
            combo = list(combo)
            if r == 1 and combo[0] not in bins + ["ds_tipo_chave_norm", "value_band"]:
                continue
            if r >= 2 and not any(c in combo for c in important + bins):
                continue

            sub = feat.iloc[added_idx][combo]
            if sub.empty:
                continue
            group_rows = []
            for key, rel_idxs in sub.groupby(combo, dropna=False).indices.items():
                idxs = sub.iloc[list(rel_idxs)].index.to_numpy(dtype=int)
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
            group_rows.sort()
            for tp_loss, neg_fp, key, mask, fp_removed, fp_per_tp in group_rows[:top_groups_per_combo]:
                vals = key if isinstance(key, tuple) else (key,)
                vals = [str(v) for v in vals]
                desc = "added_only AND " + " AND ".join([f"{c}={v}" for c, v in zip(combo, vals)])
                add_veto_candidate(
                    out, "r3k_combo", "retighten_microsegment_added_only", desc, mask, y,
                    max_tp_loss, min_fp_removed, min_fp_per_tp,
                    {"type": "combo_retighten", "scope": "rescue_added_only", "combo_cols": combo, "combo_values": vals},
                )

    best = {}
    for c in out:
        k = np.packbits(c.mask).tobytes()
        old = best.get(k)
        if old is None or (c.fp_removed, -c.tp_loss, -len(c.description)) > (old.fp_removed, -old.tp_loss, -len(old.description)):
            best[k] = c
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


def search_vetos(cands, scenario_pred, y, max_tp_loss, max_candidates, beam_width, max_rules, max_seconds, scenario):
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

    best = State(np.zeros(len(y), dtype=bool), tuple(), 0, 0)
    states = [best]
    rows = []
    stop_reason = "completed"

    for depth in range(1, max_rules + 1):
        if time.perf_counter() - t0 >= max_seconds:
            stop_reason = f"max_seconds_before_depth_{depth}"
            break
        next_states = {}
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
                    tp_loss = int(new_mask[fraud_idx].sum())
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
        rows = [{"scenario": scenario, "depth": 0, "tp_loss": 0, "fp_removed": 0, "n_rules": 0, **compute_metrics(y, scenario_pred), "rule_ids": "", "rule_descriptions": ""}]
    return pd.DataFrame(rows).sort_values(["fn", "fp"], ascending=[True, True]).reset_index(drop=True), best, [usable[i] for i in best.rule_indices], stop_reason


def select_recommended(after_df: pd.DataFrame, fp_cap: int) -> str:
    if after_df.empty:
        raise RuntimeError("after_df vazio.")
    under = after_df[after_df["fp"] <= fp_cap].copy()
    if not under.empty:
        # Within FP cap, choose lowest FN; then lowest FP.
        return str(under.sort_values(["fn", "fp"], ascending=[True, True]).iloc[0]["scenario"])
    # Otherwise choose maximum FN improvement with lowest FP.
    return str(after_df.sort_values(["fn", "fp"], ascending=[True, True]).iloc[0]["scenario"])


def apply_selected_rescues(df: pd.DataFrame, base_pred: np.ndarray, replayed: pd.DataFrame, selected_ids: list[str]) -> np.ndarray:
    current = base_pred.copy()
    by_id = {str(r["candidate_id"]): r for r in replayed.to_dict(orient="records")}
    for cid in selected_ids:
        row = by_id.get(str(cid))
        if row is None:
            continue
        current, _ = apply_rescue_candidate(df, current, row)
    return current


def make_report(summary, before_df, after_df, rules_df):
    lines = []
    lines.append("# EXP-014B-R3K — Reuse Rescue Library on R3J-FROZEN")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Cenário recomendado: `{summary['recommended_scenario']}`")
    lines.append(f"- Métricas recomendadas: `{summary['recommended_metrics']}`")
    lines.append("")
    lines.append("## Fronteira antes do re-tightening")
    show = ["scenario", "fp_budget", "fn_recovered_vs_base", "fp_added_vs_base", "tp", "fp", "fn", "precision", "recall", "fpr"]
    lines.append(before_df[[c for c in show if c in before_df.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Fronteira após re-tightening")
    show2 = ["scenario", "retightening_fp_removed", "retightening_tp_loss", "net_fn_recovered_vs_base", "net_fp_added_vs_base", "tp", "fp", "fn", "precision", "recall", "fpr", "stop_reason"]
    lines.append(after_df[[c for c in show2 if c in after_df.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Regras selecionadas")
    if rules_df.empty:
        lines.append("Nenhuma regra selecionada.")
    else:
        show3 = ["scenario", "rule_id", "family", "description", "tp_loss", "fp_removed", "fp_per_tp"]
        lines.append(rules_df[[c for c in show3 if c in rules_df.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretação")
    lines.append("O R3K reaproveita a biblioteca R3I sobre a base R3J-FROZEN. Se melhorar FN mantendo FP dentro do cap, o próximo passo é validação congelada do artifact recomendado.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--rescue-candidates", default=str(DEFAULT_RESCUE_CANDIDATES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fp-budgets", default="25,50,100,250,500")
    parser.add_argument("--fp-cap", type=int, default=5000)
    parser.add_argument("--max-fp-added-candidate", type=int, default=750)
    parser.add_argument("--min-fn-recovered", type=int, default=1)
    parser.add_argument("--skip-retightening", action="store_true")
    parser.add_argument("--max-tp-loss-retighten", type=int, default=0)
    parser.add_argument("--min-fp-removed-retighten", type=int, default=5)
    parser.add_argument("--max-combo-size", type=int, default=3)
    parser.add_argument("--top-groups-per-combo", type=int, default=40)
    parser.add_argument("--min-fp-per-tp", type=float, default=500.0)
    parser.add_argument("--max-candidates", type=int, default=400)
    parser.add_argument("--beam-width", type=int, default=120)
    parser.add_argument("--max-rules", type=int, default=5)
    parser.add_argument("--max-seconds-per-scenario", type=int, default=90)
    parser.add_argument("--no-write-predictions", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    rescue_path = Path(args.rescue_candidates)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014B-R3K — Reuse Rescue Library on R3J-FROZEN")
    log("=" * 80)

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")
    if not rescue_path.exists():
        raise FileNotFoundError(f"Rescue candidates não encontrado: {rescue_path}")

    df = add_bins_and_guards(normalize_columns(pd.read_csv(input_path, low_memory=False)))
    rescue_source = pd.read_csv(rescue_path)

    missing = []
    if BASE_COL not in df.columns:
        missing.append(BASE_COL)
    if "is_fraud" not in df.columns:
        missing.append("is_fraud")
    if "params_json" not in rescue_source.columns:
        missing.append("rescue_candidates.params_json")
    if missing:
        raise RuntimeError(f"Contrato falhou: {missing}")

    y = df["is_fraud"].to_numpy(dtype=int)
    base_pred = df[BASE_COL].to_numpy(dtype=int)
    base_metrics = compute_metrics(y, base_pred)
    pd.DataFrame([{"policy_name": "R3J_FROZEN_BASE", **base_metrics}]).to_csv(output_dir / "02_base_metrics.csv", index=False)

    contract = {
        "n_rows": int(len(df)),
        "n_frauds": int(df["is_fraud"].sum()),
        "base_col": BASE_COL,
        "base_metrics": base_metrics,
        "n_rescue_candidates_source": int(len(rescue_source)),
        "missing": missing,
        "contract_ok": not missing,
    }
    dump_json(contract, output_dir / "01_input_contract.json")

    log(f"Base metrics: {base_metrics}")

    log("[1/4] Reavaliando biblioteca R3I sobre R3J-FROZEN...")
    replayed = evaluate_rescue_library(
        df=df,
        base_pred=base_pred,
        rescue_df=rescue_source,
        max_fp_added_candidate=args.max_fp_added_candidate,
        min_fn_recovered=args.min_fn_recovered,
    )
    replayed.to_csv(output_dir / "03_rescue_library_replayed.csv", index=False)
    log(f"Candidatos úteis: {len(replayed)}")

    budgets = [int(x.strip()) for x in str(args.fp_budgets).split(",") if x.strip()]
    before_df = greedy_rescue_frontier(df, base_pred, replayed, budgets)
    before_df.to_csv(output_dir / "04_rescue_frontier_before_retightening.csv", index=False)

    after_rows = []
    all_frontiers = []
    all_selected_rules = []
    candidate_summary = []
    scenario_predictions = {}
    scenario_artifacts = {}

    log("[2/4] Re-tightening curto por cenário...")
    for _, row in before_df.iterrows():
        scenario = str(row["scenario"])
        selected_ids = str(row["selected_rescue_candidate_ids"]).split("|") if pd.notna(row.get("selected_rescue_candidate_ids")) and str(row.get("selected_rescue_candidate_ids")) else []
        rescue_pred = apply_selected_rescues(df, base_pred, replayed, selected_ids)
        rescue_metrics = compute_metrics(y, rescue_pred)

        if args.skip_retightening:
            final_pred = rescue_pred.copy()
            final_metrics = rescue_metrics
            selected_rules = []
            frontier = pd.DataFrame([{"scenario": scenario, "depth": 0, "tp_loss": 0, "fp_removed": 0, "n_rules": 0, **final_metrics, "rule_ids": "", "rule_descriptions": ""}])
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
            cand_df = candidates_df(cands, scenario)
            cand_df.to_csv(output_dir / f"candidates_{scenario}.csv", index=False)

            frontier, best, selected_rules, stop_reason = search_vetos(
                cands=cands,
                scenario_pred=rescue_pred,
                y=y,
                max_tp_loss=args.max_tp_loss_retighten,
                max_candidates=args.max_candidates,
                beam_width=args.beam_width,
                max_rules=args.max_rules,
                max_seconds=args.max_seconds_per_scenario,
                scenario=scenario,
            )
            final_pred = rescue_pred.copy()
            final_pred[best.mask] = 0
            final_metrics = compute_metrics(y, final_pred)

        frontier.to_csv(output_dir / f"frontier_{scenario}.csv", index=False)
        all_frontiers.append(frontier)

        rules_df = candidates_df(selected_rules, scenario)
        all_selected_rules.append(rules_df)

        retightening_fp_removed = rescue_metrics["fp"] - final_metrics["fp"]
        retightening_tp_loss = rescue_metrics["tp"] - final_metrics["tp"]

        after_rows.append({
            "scenario": scenario,
            "fp_budget": int(row["fp_budget"]),
            "n_selected_rescue_candidates": int(len(selected_ids)),
            "fn_recovered_before_retighten": int(base_metrics["fn"] - rescue_metrics["fn"]),
            "fp_added_before_retighten": int(rescue_metrics["fp"] - base_metrics["fp"]),
            "retightening_fp_removed": int(retightening_fp_removed),
            "retightening_tp_loss": int(retightening_tp_loss),
            "net_fn_recovered_vs_base": int(base_metrics["fn"] - final_metrics["fn"]),
            "net_fp_added_vs_base": int(final_metrics["fp"] - base_metrics["fp"]),
            "n_selected_retighten_rules": int(len(selected_rules)),
            "stop_reason": stop_reason,
            **final_metrics,
            "selected_rescue_candidate_ids": "|".join(selected_ids),
            "selected_retighten_rule_ids": "|".join(r.rule_id for r in selected_rules),
        })

        candidate_summary.append({
            "scenario": scenario,
            "n_retightening_candidates": int(len(cands)),
            "n_selected_retighten_rules": int(len(selected_rules)),
            "stop_reason": stop_reason,
        })

        scenario_predictions[scenario] = final_pred
        scenario_artifacts[scenario] = {
            "scenario": scenario,
            "fp_budget": int(row["fp_budget"]),
            "selected_rescue_candidate_ids": selected_ids,
            "rescue_metrics_before_retightening": rescue_metrics,
            "final_metrics": final_metrics,
            "selected_retighten_rules": rules_df.to_dict(orient="records") if not rules_df.empty else [],
            "stop_reason": stop_reason,
        }

        log(f"  {scenario}: before={rescue_metrics}, after={final_metrics}")

    pd.DataFrame(candidate_summary).to_csv(output_dir / "05_retightening_candidate_summary.csv", index=False)
    all_frontiers_df = pd.concat(all_frontiers, ignore_index=True) if all_frontiers else pd.DataFrame()
    all_frontiers_df.to_csv(output_dir / "06_all_retightening_frontiers.csv", index=False)

    after_df = pd.DataFrame(after_rows)
    after_df.to_csv(output_dir / "07_scenario_metrics_after_retightening.csv", index=False)

    selected_rules_df = pd.concat(all_selected_rules, ignore_index=True) if all_selected_rules else pd.DataFrame()
    selected_rules_df.to_csv(output_dir / "08_selected_rules_by_scenario.csv", index=False)

    recommended_scenario = select_recommended(after_df, args.fp_cap)
    recommended_pred = scenario_predictions[recommended_scenario]
    recommended_metrics = compute_metrics(y, recommended_pred)
    wl, wh = wilson_ci(recommended_metrics["tp"], int(y.sum()))

    df[FINAL_COL] = recommended_pred.astype(int)
    df["exp014b_r3k_recommended_scenario"] = recommended_scenario

    objective_status = "DONE_RESCUE_LIBRARY_REPLAYED"
    objective_status += "_RETIGHTEN_SKIPPED" if args.skip_retightening else "_RETIGHTEN_DONE"
    objective_status += "_FN_IMPROVED_VS_R3J" if recommended_metrics["fn"] < base_metrics["fn"] else "_FN_NOT_IMPROVED_VS_R3J"
    objective_status += "_FP_WITHIN_CAP" if recommended_metrics["fp"] <= args.fp_cap else "_FP_ABOVE_CAP"

    artifact = {
        "experiment": "EXP-014B-R3K",
        "policy_name": "reuse_rescue_library_on_r3j_frozen_microevolution",
        "objective_status": objective_status,
        "base_r3j_frozen_metrics": base_metrics,
        "recommended_scenario": recommended_scenario,
        "recommended_metrics": recommended_metrics,
        "wilson_low": wl,
        "wilson_high": wh,
        "scenario_artifacts": scenario_artifacts,
        "constraints": {
            "fp_budgets": budgets,
            "fp_cap": args.fp_cap,
            "skip_retightening": args.skip_retightening,
            "max_tp_loss_retighten": args.max_tp_loss_retighten,
            "max_seconds_per_scenario": args.max_seconds_per_scenario,
        },
        "notes": [
            "Short microevolution only.",
            "No new FN mining; reuses R3I rescue library.",
            "Needs frozen validation before promotion."
        ],
    }
    dump_json(artifact, output_dir / "09_policy_artifact_recommended.json")

    if not args.no_write_predictions:
        df.to_csv(output_dir / "10_predictions_recommended.csv", index=False)

    summary = {
        "experiment": "EXP-014B-R3K",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "rescue_candidates_path": str(rescue_path),
        "n_rows": int(len(df)),
        "n_frauds": int(df["is_fraud"].sum()),
        "base_metrics": base_metrics,
        "n_replayed_rescue_candidates": int(len(replayed)),
        "fp_budgets": budgets,
        "fp_cap": args.fp_cap,
        "recommended_scenario": recommended_scenario,
        "recommended_metrics": recommended_metrics,
        "recommended_wilson_low": wl,
        "recommended_wilson_high": wh,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, before_df, after_df, selected_rules_df)
    (output_dir / "11_exp014b_r3k_report.md").write_text(report, encoding="utf-8")

    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_input_contract.json",
        output_dir / "03_rescue_library_replayed.csv",
        output_dir / "04_rescue_frontier_before_retightening.csv",
        output_dir / "07_scenario_metrics_after_retightening.csv",
        output_dir / "08_selected_rules_by_scenario.csv",
        output_dir / "09_policy_artifact_recommended.json",
        output_dir / "11_exp014b_r3k_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
