#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3G — Consolidate R3F States + Quick FP-Second Iteration

Objetivo:
  Consolidar, sem refazer a execução profunda, os dois estados descobertos no
  EXP-014B-R3F parcial:

    1) BALANCED_FN_FIRST:
       TP alvo 1409, FN ~=56, FP ~=6267 na fronteira DP.
       Este é o novo candidato operacional balanceado.

    2) EXTREME_FN_FIRST:
       TP=1464, FN=1, FP=19725 após 8 vetos TP0 já encontrados.
       Este é o candidato de recall extremo / fila ampliada.

  Depois, executar uma rodada curta de FP Second sobre o BALANCED_FN_FIRST,
  com limite de tempo baixo e apenas vetos seguros dentro do orçamento de TP.

Por que existe:
  A execução profunda do R3F foi interrompida, mas os artefatos parciais já
  provaram a fronteira FN First. Este script evita rerun longo:
    - reaproveita 04_point_result.json para consolidar EXTREME_FN_FIRST;
    - recalcula somente a DP rápida para reconstruir o BALANCED_FN_FIRST;
    - roda uma busca curta de vetos sobre BALANCED_FN_FIRST.

Uso padrão:
  python scripts/exp_014b_r3g_consolidate_r3f_states_quick_fp_second.py

Uso rápido e seguro:
  python scripts/exp_014b_r3g_consolidate_r3f_states_quick_fp_second.py --quick-veto-seconds 180 --max-rules 5 --max-candidates 400 --beam-width 120

Somente consolidar, sem nova busca de vetos:
  python scripts/exp_014b_r3g_consolidate_r3f_states_quick_fp_second.py --skip-quick-vetos

Saídas:
  resultados/experimentos/EXP-014B-R3G/
    00_run_summary.json
    01_reused_r3f_frontier_rows.csv
    02_consolidated_states.csv
    03_balanced_quick_candidates.csv
    04_balanced_quick_frontier.csv
    05_balanced_selected_rules_quick.csv
    06_extreme_selected_rules_reused.csv
    07_policy_artifact_balanced.json
    08_policy_artifact_extreme.json
    09_predictions.csv
    10_exp014b_r3g_report.md
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

DEFAULT_INPUT = PROJECT_ROOT / "dados" / "exp014a_expanded_scored_input.csv"
DEFAULT_R3F_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3F"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3G"

SMALL_BENCHMARK = {
    "source": "SMALL_REAPPLIED_EXP013K_POLICY",
    "tp": 118,
    "fp": 199,
    "fn": 6,
    "recall": 0.9516,
    "precision": 0.3722,
    "fpr": 0.0202,
}

R3E_BENCHMARK = {
    "source": "EXP-014B-R3E",
    "tp": 1393,
    "fp": 6403,
    "fn": 72,
    "recall": 0.95085,
    "precision": 0.17868,
    "fpr": 0.05698,
}

SCORE_COLS_DEFAULT = [
    "lgbm_r4_score",
    "lgbm_mapped",
    "lgbm_raw",
    "score_final",
]

FEATURE_COLS = [
    "value_band",
    "ds_tipo_chave_norm",
    "periodo_dia",
    "first_receiver_flag_real",
    "mbk_available_flag",
    "module_quiet",
    "lgbm_bin",
    "if_bin",
    "score_bin",
    "vl_bin",
    "ratio_bin",
    "qtd_rec_bin",
    "valor_rec_bin",
]

NUMERIC_VETO_COLS = {
    "lgbm_r4_score": "le",
    "lgbm_mapped": "le",
    "lgbm_raw": "le",
    "score_final": "le",
    "if_percentile": "le",
    "if_percentile_x": "le",
    "if_percentile_y": "le",
    "vl_pix": "le",
    "ratio_valor_media_pagador_90d": "le",
    "qtd_pix_recebidos_180d": "ge",
    "valor_total_recebido_180d": "ge",
}


@dataclass
class SegmentInfo:
    segment_id: int
    segment_values: dict[str, str]
    idxs: np.ndarray
    n_rows: int
    n_pos: int
    n_neg: int


@dataclass
class VetoCandidate:
    rule_id: str
    family: str
    description: str
    mask: np.ndarray
    tp_loss: int
    fp_removed: int
    n_removed: int
    block_tp_loss_max: int
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
        raise RuntimeError("Coluna is_fraud ausente.")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)

    if "decisao" in df.columns and "runtime_flagged" not in df.columns:
        df["runtime_flagged"] = df["decisao"].astype(str).str.upper().isin({"CONFIRMAR", "BLOQUEAR"}).astype(int)
    if "runtime_flagged" not in df.columns:
        df["runtime_flagged"] = 0

    for c in ["runtime_flagged", "exp014a_frozen_pred", "exp013k_residual_fp_pred", "exp014b_r3d_final_pred", "exp014b_r3e_final_pred"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "transaction_id" in df.columns:
        df["transaction_id"] = df["transaction_id"].astype("string").str.strip()

    for c in ["event_datetime", "data_pix"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

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
    if "vl_bin" not in df.columns and "vl_pix" in df.columns:
        df["vl_bin"] = qbin_series(num(df, "vl_pix", 0.0), "vl", [20, 50, 100, 250, 500, 1000, 5000, 10000])
    if "ratio_bin" not in df.columns and "ratio_valor_media_pagador_90d" in df.columns:
        df["ratio_bin"] = qbin_series(num(df, "ratio_valor_media_pagador_90d", 0.0), "ratio", [0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    if "qtd_rec_bin" not in df.columns and "qtd_pix_recebidos_180d" in df.columns:
        df["qtd_rec_bin"] = qbin_series(num(df, "qtd_pix_recebidos_180d", 0.0), "qtdrec", [0, 1, 2, 5, 10, 20, 50, 100])
    if "valor_rec_bin" not in df.columns and "valor_total_recebido_180d" in df.columns:
        df["valor_rec_bin"] = qbin_series(num(df, "valor_total_recebido_180d", 0.0), "valrec", [0, 100, 500, 1000, 5000, 10000, 25000])

    se_score = num(df, ["se_score_x", "se_score_y", "se_score"], 0.0)
    se_count = num(df, ["se_patterns_count", "se_pattern_count"], 0.0)
    beh_score = num(df, ["beh_score", "behavioral_score"], 0.0)
    beh_count = num(df, ["beh_factors_count", "behavioral_risk_factor_count"], 0.0)
    runtime = num(df, "runtime_flagged", 0.0)

    module_strong = (
        (se_score >= 40)
        | (se_count >= 2)
        | (beh_score >= 25)
        | (beh_count >= 2)
        | (runtime >= 1)
    )
    df["module_quiet"] = np.where(module_strong, "module_strong", "module_quiet")
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


def threshold_grid(values: np.ndarray, n_quantiles: int, include_all_none: bool = True) -> list[float]:
    vals = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if vals.empty:
        return []
    grid = sorted(set(float(x) for x in vals.quantile(np.linspace(0, 1, n_quantiles)).to_numpy()))
    grid.extend([float(vals.min()), float(vals.max())])
    if include_all_none:
        grid.append(float(vals.min()) - 1e-12)
        grid.append(float(vals.max()) + 1e-12)
    return sorted(set(grid))


def build_segments(df: pd.DataFrame, segment_cols: list[str]) -> list[SegmentInfo]:
    y = df["is_fraud"].to_numpy(dtype=int)
    seg_df = pd.DataFrame(index=df.index)
    for c in segment_cols:
        if c not in df.columns:
            raise RuntimeError(f"Coluna de segmento ausente: {c}")
        seg_df[c] = df[c].astype("string").fillna("<MISSING>").astype(str)

    segments = []
    grouped = seg_df.groupby(segment_cols, dropna=False).indices
    for sid, (key, rel_idxs) in enumerate(grouped.items()):
        idxs = np.asarray(list(rel_idxs), dtype=int)
        vals = key if isinstance(key, tuple) else (key,)
        seg_vals = {c: str(v) for c, v in zip(segment_cols, vals)}
        segments.append(SegmentInfo(
            segment_id=sid,
            segment_values=seg_vals,
            idxs=idxs,
            n_rows=int(len(idxs)),
            n_pos=int(y[idxs].sum()),
            n_neg=int(len(idxs) - y[idxs].sum()),
        ))
    return segments


def pareto_options(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_tp = {}
    for opt in options:
        tp = int(opt["tp"])
        old = best_by_tp.get(tp)
        if old is None or int(opt["fp"]) < int(old["fp"]):
            best_by_tp[tp] = opt
    opts = sorted(best_by_tp.values(), key=lambda o: (-int(o["tp"]), int(o["fp"])))
    kept = []
    best_fp = float("inf")
    for opt in opts:
        fp = int(opt["fp"])
        if fp < best_fp:
            kept.append(opt)
            best_fp = fp
    return sorted(kept, key=lambda o: (int(o["tp"]), int(o["fp"])))


def build_segment_options(
    df: pd.DataFrame,
    segments: list[SegmentInfo],
    score_cols: list[str],
    n_quantiles: int,
    max_options_per_segment: int,
) -> list[list[dict[str, Any]]]:
    y = df["is_fraud"].to_numpy(dtype=int)
    scores_by_col = {c: num(df, c, 0.0).to_numpy(dtype=float) for c in score_cols if c in df.columns}
    all_options = []

    for seg in segments:
        opts = [
            {"segment_id": seg.segment_id, "score_col": None, "threshold": None, "direction": "none", "tp": 0, "fp": 0, "n_alerts": 0},
            {"segment_id": seg.segment_id, "score_col": None, "threshold": None, "direction": "all", "tp": seg.n_pos, "fp": seg.n_neg, "n_alerts": seg.n_rows},
        ]
        for col, scores in scores_by_col.items():
            s = scores[seg.idxs]
            grid = threshold_grid(s, n_quantiles=n_quantiles, include_all_none=True)
            y_seg = y[seg.idxs]
            for th in grid:
                pred = s >= th
                tp = int(y_seg[pred].sum())
                fp = int(pred.sum() - tp)
                opts.append({
                    "segment_id": seg.segment_id,
                    "score_col": col,
                    "threshold": float(th),
                    "direction": "ge",
                    "tp": tp,
                    "fp": fp,
                    "n_alerts": int(pred.sum()),
                })
        opts = pareto_options(opts)
        if len(opts) > max_options_per_segment:
            opts_df = pd.DataFrame(opts).sort_values(["tp", "fp"]).reset_index(drop=True)
            idx_keep = set(np.linspace(0, len(opts_df) - 1, max_options_per_segment, dtype=int).tolist())
            idx_keep.update(opts_df.sort_values(["fp", "tp"], ascending=[True, False]).head(max_options_per_segment // 4).index.tolist())
            opts = opts_df.iloc[sorted(idx_keep)].to_dict(orient="records")
        all_options.append(opts)
    return all_options


def prune_dp(dp: dict[int, tuple[int, list[int]]]) -> dict[int, tuple[int, list[int]]]:
    items = sorted(dp.items(), key=lambda kv: (-kv[0], kv[1][0]))
    pruned = {}
    best_fp = float("inf")
    for tp, val in items:
        fp = val[0]
        if fp < best_fp:
            pruned[tp] = val
            best_fp = fp
    return pruned


def solve_target_tp(all_options: list[list[dict[str, Any]]], total_frauds: int, target_tp: int) -> tuple[list[int], dict[str, Any]]:
    dp = {0: (0, [])}
    for opts in all_options:
        new_dp = {}
        for tp_cur, (fp_cur, choices) in dp.items():
            for opt_idx, opt in enumerate(opts):
                tp_new = min(total_frauds, tp_cur + int(opt["tp"]))
                fp_new = fp_cur + int(opt["fp"])
                old = new_dp.get(tp_new)
                if old is None or fp_new < old[0]:
                    new_dp[tp_new] = (fp_new, choices + [opt_idx])
        dp = prune_dp(new_dp)

    eligible = [tp for tp in dp.keys() if tp >= target_tp]
    if not eligible:
        raise RuntimeError(f"Nenhuma solução DP para target_tp={target_tp}.")
    best_tp = min(eligible, key=lambda tp: (dp[tp][0], tp))
    fp, choices = dp[best_tp]
    return choices, {
        "target_tp": int(target_tp),
        "achieved_tp": int(best_tp),
        "achieved_fn": int(total_frauds - best_tp),
        "base_fp": int(fp),
    }


def build_prediction_from_choices(df: pd.DataFrame, segments: list[SegmentInfo], all_options: list[list[dict[str, Any]]], choices: list[int]) -> np.ndarray:
    pred = np.zeros(len(df), dtype=int)
    scores_cache = {}
    for seg, opt_idx in zip(segments, choices):
        opt = all_options[seg.segment_id][opt_idx]
        idxs = seg.idxs
        if opt["direction"] == "none":
            continue
        if opt["direction"] == "all":
            pred[idxs] = 1
            continue
        col = opt["score_col"]
        if col not in scores_cache:
            scores_cache[col] = num(df, col, 0.0).to_numpy(dtype=float)
        pred[idxs] = (scores_cache[col][idxs] >= float(opt["threshold"])).astype(int)
    return pred


def choice_recipe(segments: list[SegmentInfo], all_options: list[list[dict[str, Any]]], choices: list[int], segment_cols: list[str]) -> dict[str, Any]:
    out = []
    for seg, opt_idx in zip(segments, choices):
        opt = dict(all_options[seg.segment_id][opt_idx])
        opt["segment_values"] = seg.segment_values
        out.append(opt)
    return {"type": "global_recall_budget_dp", "segment_cols": segment_cols, "segments": out}


def apply_recipe(df: pd.DataFrame, recipe: dict[str, Any]) -> np.ndarray:
    if recipe.get("type") != "global_recall_budget_dp":
        raise RuntimeError(f"Recipe não suportada: {recipe.get('type')}")
    pred = np.zeros(len(df), dtype=int)
    segment_cols = recipe["segment_cols"]
    scores_cache = {}
    for seg in recipe["segments"]:
        mask = np.ones(len(df), dtype=bool)
        for c in segment_cols:
            val = str(seg["segment_values"][c])
            mask = mask & (df[c].astype("string").fillna("<MISSING>").astype(str).to_numpy() == val)
        direction = seg.get("direction")
        if direction == "none":
            continue
        if direction == "all":
            pred[mask] = 1
            continue
        col = seg["score_col"]
        if col not in scores_cache:
            scores_cache[col] = num(df, col, 0.0).to_numpy(dtype=float)
        pred[mask] = (scores_cache[col][mask] >= float(seg["threshold"])).astype(int)
    return pred


def rule_mask(df: pd.DataFrame, rule: dict[str, Any], current_pred: np.ndarray) -> np.ndarray:
    params_raw = rule.get("params_json") or rule.get("params") or "{}"
    params = params_raw if isinstance(params_raw, dict) else json.loads(str(params_raw).replace("Infinity", "1e999"))
    mask = np.ones(len(df), dtype=bool)

    if params.get("type") == "combo":
        cols = params.get("combo_cols", [])
        vals = params.get("combo_values", [])
        for c, v in zip(cols, vals):
            if c not in df.columns:
                return np.zeros(len(df), dtype=bool)
            mask = mask & (df[c].astype("string").fillna("<MISSING>").astype(str).to_numpy() == str(v))
    elif params.get("type") == "numeric_threshold":
        c = params["col"]
        direction = params["direction"]
        cut = float(params["cut"])
        if c not in df.columns:
            return np.zeros(len(df), dtype=bool)
        vals = num(df, c, 0.0).to_numpy(dtype=float)
        mask = mask & ((vals <= cut) if direction == "le" else (vals >= cut))
    else:
        # fallback from description is intentionally conservative
        return np.zeros(len(df), dtype=bool)

    if params.get("require_module_quiet", False) and "module_quiet" in df.columns:
        mask = mask & (df["module_quiet"].astype(str).to_numpy() == "module_quiet")
    return mask & (current_pred.astype(int) == 1)


def apply_rules(df: pd.DataFrame, base_pred: np.ndarray, rules: list[dict[str, Any]]) -> tuple[np.ndarray, pd.DataFrame]:
    y = df["is_fraud"].to_numpy(dtype=int)
    pred = base_pred.copy().astype(int)
    rows = []
    for i, rule in enumerate(rules):
        m = rule_mask(df, rule, pred)
        tp_loss = int(((y == 1) & m).sum())
        fp_removed = int(((y == 0) & m).sum())
        pred[m] = 0
        rows.append({
            "rule_index": i,
            "rule_id": rule.get("rule_id"),
            "family": rule.get("family"),
            "description": rule.get("description"),
            "tp_loss": tp_loss,
            "fp_removed": fp_removed,
            "n_removed": int(m.sum()),
            "params_json": rule.get("params_json"),
        })
    return pred, pd.DataFrame(rows)


def make_time_blocks(df: pd.DataFrame, n_blocks: int) -> pd.Series:
    if "data_pix" in df.columns and df["data_pix"].notna().any():
        dates = pd.to_datetime(df["data_pix"], errors="coerce")
    elif "event_datetime" in df.columns and df["event_datetime"].notna().any():
        dates = pd.to_datetime(df["event_datetime"], errors="coerce")
    else:
        return pd.qcut(np.arange(len(df)), q=min(n_blocks, len(df)), labels=False, duplicates="drop").astype(int)
    tmp = pd.DataFrame({"date": dates, "_idx": np.arange(len(df))}).sort_values(["date", "_idx"])
    tmp["block"] = pd.qcut(np.arange(len(tmp)), q=min(n_blocks, len(tmp)), labels=False, duplicates="drop")
    return pd.Series(index=tmp["_idx"].values, data=tmp["block"].values).sort_index().astype(int)


def candidate_stats(mask: np.ndarray, y: np.ndarray, blocks: pd.Series) -> tuple[int, int, int]:
    tp_loss = int(mask[y == 1].sum())
    fp_removed = int(mask[y == 0].sum())
    bmax = 0
    bvals = blocks.to_numpy()
    pos_mask = y == 1
    for b in sorted(blocks.dropna().unique()):
        bm = mask & (bvals == b)
        bmax = max(bmax, int((bm & pos_mask).sum()))
    return tp_loss, fp_removed, bmax


def add_candidate(out, prefix, family, description, mask, y, blocks, min_fp_removed, allowed_tp_loss, max_block_tp_loss, min_fp_per_tp, params):
    if not mask.any():
        return
    tp_loss, fp_removed, bmax = candidate_stats(mask, y, blocks)
    if fp_removed < min_fp_removed or tp_loss > allowed_tp_loss or bmax > max_block_tp_loss:
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
        block_tp_loss_max=bmax,
        fp_per_tp=fp_per_tp,
        params=params,
    ))


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in FEATURE_COLS:
        if c in df.columns:
            out[c] = df[c].astype("string").fillna("<MISSING>").astype(str)
    return out


def mine_quick_candidates(df, base_pred, blocks, target_tp, min_fp_removed, max_combo_size, top_groups_per_combo, max_block_tp_loss, min_fp_per_tp, require_module_quiet):
    y = df["is_fraud"].to_numpy(dtype=int)
    base_tp = int(((base_pred == 1) & (y == 1)).sum())
    allowed_tp_loss = max(0, base_tp - target_tp)
    pred_pos = base_pred.astype(bool)
    if require_module_quiet and "module_quiet" in df.columns:
        pred_pos = pred_pos & (df["module_quiet"].astype(str).to_numpy() == "module_quiet")

    out = []

    # Numeric quick rules
    for col, direction in NUMERIC_VETO_COLS.items():
        if col not in df.columns:
            continue
        vals = num(df, col, 0.0).to_numpy(dtype=float)
        pp = vals[pred_pos]
        if len(pp) == 0:
            continue
        cuts = []
        try:
            cuts.extend([float(x) for x in np.quantile(pp, [0.03, 0.05, 0.10, 0.20, 0.30, 0.50])])
        except Exception:
            pass
        cuts = sorted(set(float(x) for x in cuts if np.isfinite(x)))
        for cut in cuts:
            mask = pred_pos & ((vals <= cut) if direction == "le" else (vals >= cut))
            desc = f"{col}<={cut:g}" if direction == "le" else f"{col}>={cut:g}"
            add_candidate(out, "r3g_num", "quick_numeric_veto", desc, mask, y, blocks, min_fp_removed, allowed_tp_loss, max_block_tp_loss, min_fp_per_tp, {"type": "numeric_threshold", "col": col, "direction": direction, "cut": cut, "require_module_quiet": require_module_quiet})

    # Combo quick rules
    feat = feature_frame(df)
    cols = list(feat.columns)
    base_cols = [c for c in ["value_band", "ds_tipo_chave_norm", "periodo_dia", "first_receiver_flag_real", "mbk_available_flag"] if c in cols]
    bin_cols = [c for c in cols if c.endswith("_bin") or c == "module_quiet"]
    idx_pos = np.where(pred_pos)[0]

    for r in range(2, max_combo_size + 1):
        for combo in itertools.combinations(cols, r):
            combo = list(combo)
            if not any(c in base_cols for c in combo) or not any(c in bin_cols for c in combo):
                continue
            subset = feat.iloc[idx_pos][combo]
            if subset.empty:
                continue
            grouped = subset.groupby(combo, dropna=False).indices
            group_rows = []
            for key, rel_idxs in grouped.items():
                idxs = subset.iloc[list(rel_idxs)].index.to_numpy(dtype=int)
                if len(idxs) < min_fp_removed:
                    continue
                mask = np.zeros(len(df), dtype=bool)
                mask[idxs] = True
                mask = mask & pred_pos
                tp_loss, fp_removed, bmax = candidate_stats(mask, y, blocks)
                if fp_removed < min_fp_removed or tp_loss > allowed_tp_loss or bmax > max_block_tp_loss:
                    continue
                fp_per_tp = float("inf") if tp_loss == 0 else fp_removed / max(tp_loss, 1)
                if tp_loss > 0 and fp_per_tp < min_fp_per_tp:
                    continue
                group_rows.append((tp_loss, -fp_removed, key, mask, fp_removed, bmax, fp_per_tp))
            group_rows.sort()
            for tp_loss, neg_fp, key, mask, fp_removed, bmax, fp_per_tp in group_rows[:top_groups_per_combo]:
                vals = key if isinstance(key, tuple) else (key,)
                vals = [str(v) for v in vals]
                desc = " AND ".join([f"{c}={v}" for c, v in zip(combo, vals)])
                out.append(VetoCandidate(
                    rule_id=f"r3g_combo_{len(out):05d}",
                    family="quick_microsegment_veto",
                    description=desc,
                    mask=mask,
                    tp_loss=tp_loss,
                    fp_removed=fp_removed,
                    n_removed=int(mask.sum()),
                    block_tp_loss_max=bmax,
                    fp_per_tp=fp_per_tp,
                    params={"type": "combo", "combo_cols": combo, "combo_values": vals, "require_module_quiet": require_module_quiet},
                ))

    # Dedupe
    best = {}
    for c in out:
        k = np.packbits(c.mask).tobytes()
        old = best.get(k)
        if old is None or (c.fp_removed, -c.tp_loss) > (old.fp_removed, -old.tp_loss):
            best[k] = c
    out = list(best.values())
    out.sort(key=lambda c: (c.tp_loss > 0, c.tp_loss, c.block_tp_loss_max, -c.fp_removed))
    return out


def candidates_df(cands: list[VetoCandidate]) -> pd.DataFrame:
    return pd.DataFrame([{
        "candidate_index": i,
        "rule_id": c.rule_id,
        "family": c.family,
        "description": c.description,
        "tp_loss": c.tp_loss,
        "fp_removed": c.fp_removed,
        "n_removed": c.n_removed,
        "block_tp_loss_max": c.block_tp_loss_max,
        "fp_per_tp": c.fp_per_tp,
        "params_json": json.dumps(c.params, ensure_ascii=False),
    } for i, c in enumerate(cands)])


def search_quick_vetos(cands, base_pred, y, target_tp, max_candidates, beam_width, max_rules, max_seconds):
    t0 = time.perf_counter()
    base_tp = int(((base_pred == 1) & (y == 1)).sum())
    tp_budget = max(0, base_tp - target_tp)

    usable = [c for c in cands if c.tp_loss <= tp_budget]
    usable.sort(key=lambda c: (c.tp_loss > 0, -c.fp_removed if c.tp_loss == 0 else -c.fp_per_tp, -c.fp_removed))
    usable = usable[:max_candidates]
    fraud_idx = np.where(y == 1)[0]
    zero_loss_mode = (tp_budget == 0 and all(c.tp_loss == 0 for c in usable))

    pending_limit = max(beam_width * 8, 1000)
    pending_keep = max(beam_width * 4, 500)

    def rank(s: State):
        return (s.fp_removed, -s.tp_loss, -len(s.rule_indices))

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
                    if tp_loss > tp_budget:
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
            pred = base_pred.copy()
            pred[s.mask] = 0
            rows.append({"depth": depth, "tp_loss": s.tp_loss, "fp_removed": s.fp_removed, "n_rules": len(s.rule_indices), **compute_metrics(y, pred), "rule_ids": "|".join(usable[i].rule_id for i in s.rule_indices), "rule_descriptions": " || ".join(usable[i].description for i in s.rule_indices)})
        if time.perf_counter() - t0 >= max_seconds:
            stop_reason = f"max_seconds_after_depth_{depth}"
            break

    if not rows:
        rows = [{"depth": 0, "tp_loss": 0, "fp_removed": 0, "n_rules": 0, **compute_metrics(y, base_pred), "rule_ids": "", "rule_descriptions": ""}]
    selected = [usable[i] for i in best.rule_indices]
    return pd.DataFrame(rows).sort_values(["fp", "fn"], ascending=[True, True]).reset_index(drop=True), best, selected, stop_reason


def make_report(summary, states_df, balanced_rules_df):
    lines = []
    lines.append("# EXP-014B-R3G — Consolidação R3F + rodada curta FP Second")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Balanced final: `{summary['balanced_final_metrics']}`")
    lines.append(f"- Extreme final: `{summary.get('extreme_final_metrics')}`")
    lines.append("")
    lines.append("## Estados consolidados")
    lines.append(states_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Regras rápidas selecionadas para o Balanced")
    if balanced_rules_df.empty:
        lines.append("Nenhuma regra rápida selecionada.")
    else:
        cols = ["rule_id", "family", "description", "tp_loss", "fp_removed", "block_tp_loss_max", "fp_per_tp"]
        lines.append(balanced_rules_df[[c for c in cols if c in balanced_rules_df.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Próximo passo")
    lines.append("Usar o estado BALANCED_R3G como benchmark operacional FN First / FP Second e rodar apenas microexperimentos curtos sobre ele. O estado EXTREME_R3F fica como referência de recall máximo/fila ampliada.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--r3f-dir", default=str(DEFAULT_R3F_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--balanced-target-tp", type=int, default=1409)
    parser.add_argument("--segment-cols", default="ds_tipo_chave_norm,mbk_available_flag,value_band")
    parser.add_argument("--score-cols", default=",".join(SCORE_COLS_DEFAULT))
    parser.add_argument("--threshold-quantiles", type=int, default=80)
    parser.add_argument("--max-options-per-segment", type=int, default=60)
    parser.add_argument("--quick-veto-seconds", type=int, default=180)
    parser.add_argument("--skip-quick-vetos", action="store_true")
    parser.add_argument("--min-fp-removed", type=int, default=30)
    parser.add_argument("--max-combo-size", type=int, default=3)
    parser.add_argument("--top-groups-per-combo", type=int, default=40)
    parser.add_argument("--max-candidates", type=int, default=400)
    parser.add_argument("--beam-width", type=int, default=120)
    parser.add_argument("--max-rules", type=int, default=5)
    parser.add_argument("--max-block-tp-loss", type=int, default=1)
    parser.add_argument("--min-fp-per-tp", type=float, default=200.0)
    parser.add_argument("--time-blocks", type=int, default=10)
    parser.add_argument("--require-module-quiet", action="store_true", default=True)
    parser.add_argument("--no-write-predictions", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    r3f_dir = Path(args.r3f_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014B-R3G — Consolidate R3F States + Quick FP-Second")
    log("=" * 80)

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    df = add_bins_and_guards(normalize_columns(pd.read_csv(input_path, low_memory=False)))
    y = df["is_fraud"].to_numpy(dtype=int)
    total_frauds = int(y.sum())
    total_negatives = int(len(df) - total_frauds)
    blocks = make_time_blocks(df, args.time_blocks)

    segment_cols = [x.strip() for x in args.segment_cols.split(",") if x.strip()]
    score_cols = [x.strip() for x in args.score_cols.split(",") if x.strip() and x.strip() in df.columns]

    # Reuse frontier row if present.
    frontier_path = r3f_dir / "03_global_recall_budget_frontier.csv"
    reused_rows = pd.DataFrame()
    if frontier_path.exists():
        frontier = pd.read_csv(frontier_path)
        reused_rows = frontier[(frontier["target_tp"].astype(int) == int(args.balanced_target_tp)) | (frontier["achieved_tp"].astype(int) == int(args.balanced_target_tp))].copy()
        reused_rows.to_csv(output_dir / "01_reused_r3f_frontier_rows.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "01_reused_r3f_frontier_rows.csv", index=False)

    log("[1/5] Recalculando DP apenas para BALANCED target...")
    segments = build_segments(df, segment_cols)
    all_options = build_segment_options(df, segments, score_cols, args.threshold_quantiles, args.max_options_per_segment)
    choices, dp_info = solve_target_tp(all_options, total_frauds, args.balanced_target_tp)
    balanced_recipe = choice_recipe(segments, all_options, choices, segment_cols)
    balanced_base_pred = build_prediction_from_choices(df, segments, all_options, choices)
    balanced_base_metrics = compute_metrics(y, balanced_base_pred)

    log(f"Balanced base: {balanced_base_metrics}")

    log("[2/5] Rodada curta FP Second sobre BALANCED...")
    if args.skip_quick_vetos:
        balanced_final_pred = balanced_base_pred.copy()
        balanced_rules_df = pd.DataFrame()
        balanced_frontier = pd.DataFrame([{"depth": 0, **balanced_base_metrics}])
        quick_stop_reason = "skipped"
        quick_candidates = []
    else:
        quick_candidates = mine_quick_candidates(
            df=df,
            base_pred=balanced_base_pred,
            blocks=blocks,
            target_tp=args.balanced_target_tp,
            min_fp_removed=args.min_fp_removed,
            max_combo_size=args.max_combo_size,
            top_groups_per_combo=args.top_groups_per_combo,
            max_block_tp_loss=args.max_block_tp_loss,
            min_fp_per_tp=args.min_fp_per_tp,
            require_module_quiet=args.require_module_quiet,
        )
        candidates_df(quick_candidates).to_csv(output_dir / "03_balanced_quick_candidates.csv", index=False)
        log(f"Quick candidates: {len(quick_candidates)}")
        balanced_frontier, best, selected, quick_stop_reason = search_quick_vetos(
            quick_candidates,
            balanced_base_pred,
            y,
            args.balanced_target_tp,
            args.max_candidates,
            args.beam_width,
            args.max_rules,
            args.quick_veto_seconds,
        )
        balanced_frontier.to_csv(output_dir / "04_balanced_quick_frontier.csv", index=False)
        balanced_final_pred = balanced_base_pred.copy()
        balanced_final_pred[best.mask] = 0
        balanced_rules_df = candidates_df(selected)
    balanced_rules_df.to_csv(output_dir / "05_balanced_selected_rules_quick.csv", index=False)
    balanced_final_metrics = compute_metrics(y, balanced_final_pred)

    log(f"Balanced final: {balanced_final_metrics}")

    log("[3/5] Consolidando EXTREME a partir do point_result, se existir...")
    extreme_path = r3f_dir / "per_point" / "point_01_tp1464" / "04_point_result.json"
    if not extreme_path.exists():
        extreme_path = r3f_dir / "04_point_result.json"

    extreme_base_metrics = None
    extreme_final_metrics = None
    extreme_recipe = None
    extreme_rules_df = pd.DataFrame()
    extreme_final_pred = None

    if extreme_path.exists():
        extreme_obj = load_json(extreme_path)
        extreme_recipe = extreme_obj.get("base_recipe")
        if extreme_recipe:
            extreme_base_pred = apply_recipe(df, extreme_recipe)
            extreme_rules = extreme_obj.get("selected_rules", [])
            extreme_final_pred, extreme_rules_df = apply_rules(df, extreme_base_pred, extreme_rules)
            extreme_base_metrics = compute_metrics(y, extreme_base_pred)
            extreme_final_metrics = compute_metrics(y, extreme_final_pred)
            extreme_rules_df.to_csv(output_dir / "06_extreme_selected_rules_reused.csv", index=False)
        else:
            pd.DataFrame().to_csv(output_dir / "06_extreme_selected_rules_reused.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "06_extreme_selected_rules_reused.csv", index=False)

    df["exp014b_r3g_balanced_base_pred"] = balanced_base_pred.astype(int)
    df["exp014b_r3g_balanced_final_pred"] = balanced_final_pred.astype(int)
    if extreme_final_pred is not None:
        df["exp014b_r3g_extreme_final_pred"] = extreme_final_pred.astype(int)

    states = [
        {"state": "BALANCED_R3G_BASE", **balanced_base_metrics},
        {"state": "BALANCED_R3G_QUICK_FP_SECOND", **balanced_final_metrics},
    ]
    if extreme_base_metrics:
        states.append({"state": "EXTREME_R3F_BASE_RECONSTRUCTED", **extreme_base_metrics})
    if extreme_final_metrics:
        states.append({"state": "EXTREME_R3F_REUSED_VETOS", **extreme_final_metrics})
    states_df = pd.DataFrame(states)
    states_df["fp_gap_vs_r3e"] = states_df["fp"] - R3E_BENCHMARK["fp"]
    states_df["fn_gap_vs_r3e"] = states_df["fn"] - R3E_BENCHMARK["fn"]
    states_df["fpr_gap_vs_small"] = states_df["fpr"] - SMALL_BENCHMARK["fpr"]
    states_df["precision_gap_vs_small"] = states_df["precision"] - SMALL_BENCHMARK["precision"]
    states_df.to_csv(output_dir / "02_consolidated_states.csv", index=False)

    for pred_col in ["exp014b_r3g_balanced_final_pred", "exp014b_r3g_extreme_final_pred"]:
        if pred_col in df.columns:
            m = compute_metrics(y, df[pred_col].to_numpy(dtype=int))
            wl, wh = wilson_ci(m["tp"], total_frauds)
            log(f"{pred_col}: metrics={m}, wilson=({wl:.6f}, {wh:.6f})")

    balanced_wl, balanced_wh = wilson_ci(balanced_final_metrics["tp"], total_frauds)
    extreme_wl, extreme_wh = (wilson_ci(extreme_final_metrics["tp"], total_frauds) if extreme_final_metrics else (None, None))

    balanced_artifact = {
        "experiment": "EXP-014B-R3G",
        "policy_name": "balanced_fn_first_quick_fp_second",
        "source": "R3F balanced target reconstructed by scoped DP",
        "target_tp": args.balanced_target_tp,
        "dp_info": dp_info,
        "recipe": balanced_recipe,
        "base_metrics": balanced_base_metrics,
        "final_metrics": balanced_final_metrics,
        "wilson_low": balanced_wl,
        "wilson_high": balanced_wh,
        "quick_stop_reason": quick_stop_reason,
        "selected_rules": balanced_rules_df.to_dict(orient="records") if not balanced_rules_df.empty else [],
    }
    dump_json(balanced_artifact, output_dir / "07_policy_artifact_balanced.json")

    extreme_artifact = {
        "experiment": "EXP-014B-R3G",
        "policy_name": "extreme_fn_first_reused_from_r3f_partial",
        "source": str(extreme_path) if extreme_path.exists() else None,
        "recipe": extreme_recipe,
        "base_metrics": extreme_base_metrics,
        "final_metrics": extreme_final_metrics,
        "wilson_low": extreme_wl,
        "wilson_high": extreme_wh,
        "selected_rules": extreme_rules_df.to_dict(orient="records") if not extreme_rules_df.empty else [],
    }
    dump_json(extreme_artifact, output_dir / "08_policy_artifact_extreme.json")

    if not args.no_write_predictions:
        df.to_csv(output_dir / "09_predictions.csv", index=False)

    objective_status = "DONE"
    objective_status += "_BALANCED_FN_IMPROVED_VS_R3E" if balanced_final_metrics["fn"] < R3E_BENCHMARK["fn"] else "_BALANCED_FN_NOT_IMPROVED_VS_R3E"
    objective_status += "_BALANCED_FP_IMPROVED_VS_R3E" if balanced_final_metrics["fp"] < R3E_BENCHMARK["fp"] else "_BALANCED_FP_NOT_IMPROVED_VS_R3E"
    objective_status += "_EXTREME_CONSOLIDATED" if extreme_final_metrics else "_EXTREME_NOT_FOUND"

    summary = {
        "experiment": "EXP-014B-R3G",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "r3f_dir": str(r3f_dir),
        "n_rows": int(len(df)),
        "n_frauds": total_frauds,
        "n_negatives": total_negatives,
        "balanced_target_tp": args.balanced_target_tp,
        "balanced_base_metrics": balanced_base_metrics,
        "balanced_final_metrics": balanced_final_metrics,
        "balanced_wilson_low": balanced_wl,
        "balanced_wilson_high": balanced_wh,
        "balanced_quick_stop_reason": quick_stop_reason,
        "balanced_n_quick_candidates": int(len(quick_candidates)),
        "balanced_n_selected_rules": int(len(balanced_rules_df)),
        "extreme_base_metrics": extreme_base_metrics,
        "extreme_final_metrics": extreme_final_metrics,
        "extreme_wilson_low": extreme_wl,
        "extreme_wilson_high": extreme_wh,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, states_df, balanced_rules_df)
    (output_dir / "10_exp014b_r3g_report.md").write_text(report, encoding="utf-8")

    log("[5/5] Concluído.")
    log(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
