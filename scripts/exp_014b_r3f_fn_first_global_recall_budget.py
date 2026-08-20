#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3F — FN-First Global Recall Budget Optimizer

Objetivo:
  Mudar a estratégia para FN First e FP Second no dataset expandido.

  Em vez de fixar recall >=95% e tentar reduzir FP com pouco/zero buffer,
  este experimento constrói uma fronteira global:

      menor FP possível para cada orçamento de FN/TP

  Depois, para cada ponto selecionado da fronteira, aplica vetos residuais
  tentando reduzir FP sem ultrapassar o orçamento de FN daquele ponto.

Perguntas respondidas:
  1. Qual é o mínimo de FN alcançável com os scores/features atuais?
  2. Quanto FP custa reduzir FN para 0, 5, 10, 20, etc.?
  3. Para cada patamar de FN, quanto FP conseguimos recuperar sem aumentar FN?
  4. Onde parece estar o ponto de parada: FN irredutível primeiro, FP irredutível depois?

Estratégia:
  - Otimizador global por segmentos:
      segment_cols default = ds_tipo_chave_norm, mbk_available_flag, value_band
  - Para cada segmento, gera opções de threshold por score.
  - Resolve por programação dinâmica:
      minimizar FP total sujeito a TP >= target_tp.
  - Targets incluem:
      TP mínimo 95%, Wilson aproximado, e patamares FN-first até TP total.
  - Depois aplica microvetos/numeric vetos preservando o target_tp do ponto.

Uso padrão:
  python scripts/exp_014b_r3f_fn_first_global_recall_budget.py

Execução rápida:
  python scripts/exp_014b_r3f_fn_first_global_recall_budget.py --max-frontier-points 4 --max-candidates 500 --beam-width 120 --max-rules 6 --max-seconds-per-point 240 --bootstrap-iters 50

Execução profunda:
  python scripts/exp_014b_r3f_fn_first_global_recall_budget.py --max-frontier-points 8 --max-candidates 900 --beam-width 260 --max-rules 10 --max-seconds-per-point 600 --bootstrap-iters 100

Saídas:
  resultados/experimentos/EXP-014B-R3F/
    00_run_summary.json
    01_input_contract.json
    02_segment_options_summary.csv
    03_global_recall_budget_frontier.csv
    04_selected_frontier_points.csv
    05_point_results.json
    06_candidate_summary.csv
    07_frontier_after_vetos_all.csv
    08_selected_rules_best.csv
    09_policy_metrics.csv
    10_time_block_metrics.csv
    11_wilson_recall_ci.csv
    12_bootstrap_summary.csv
    13_false_negatives.csv
    14_false_positives_sample.csv
    15_policy_artifact.json
    16_predictions.csv
    17_exp014b_r3f_report.md
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3F"

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
    "r4_score",
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
    "r4_score": "le",
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

    if "lgbm_bin" not in df.columns and pick_col(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"]):
        df["lgbm_bin"] = qbin_series(num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0), "lgbm", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1])
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


def contract(df: pd.DataFrame, score_cols: list[str]) -> dict[str, Any]:
    missing = []
    if "is_fraud" not in df.columns:
        missing.append("is_fraud")
    if not any(c in df.columns for c in score_cols):
        missing.append("score_column")
    if not any(c in df.columns for c in ["event_datetime", "data_pix", "dt_pix"]):
        missing.append("date_column")
    return {
        "contract_ok": not missing,
        "missing": missing,
        "n_rows": int(len(df)),
        "n_frauds": int(df["is_fraud"].sum()) if "is_fraud" in df.columns else None,
        "score_cols_present": [c for c in score_cols if c in df.columns],
        "segment_cols_present": [c for c in ["ds_tipo_chave_norm", "mbk_available_flag", "value_band", "first_receiver_flag_real", "periodo_dia"] if c in df.columns],
    }


def make_time_blocks(df: pd.DataFrame, n_blocks: int) -> pd.Series:
    if "data_pix" in df.columns and df["data_pix"].notna().any():
        dates = pd.to_datetime(df["data_pix"], errors="coerce")
    elif "event_datetime" in df.columns and df["event_datetime"].notna().any():
        dates = pd.to_datetime(df["event_datetime"], errors="coerce")
    else:
        return pd.qcut(np.arange(len(df)), q=min(n_blocks, len(df)), labels=False, duplicates="drop").astype(int)

    tmp = pd.DataFrame({"date": dates, "_idx": np.arange(len(df))}).sort_values(["date", "_idx"])
    tmp["block"] = pd.qcut(np.arange(len(tmp)), q=min(n_blocks, len(tmp)), labels=False, duplicates="drop")
    out = pd.Series(index=tmp["_idx"].values, data=tmp["block"].values).sort_index()
    return out.astype(int)


def block_metrics(df: pd.DataFrame, pred: np.ndarray, blocks: pd.Series, policy_name: str) -> pd.DataFrame:
    y = df["is_fraud"].to_numpy(dtype=int)
    rows = []
    bvals = blocks.to_numpy()
    for b in sorted(blocks.dropna().unique()):
        idx = bvals == b
        part = df.loc[idx]
        rows.append({
            "policy_name": policy_name,
            "block": int(b),
            "n_rows": int(len(part)),
            "n_frauds": int(part["is_fraud"].sum()),
            **compute_metrics(y[idx], pred[idx]),
        })
    return pd.DataFrame(rows)


def threshold_grid(values: np.ndarray, n_quantiles: int, include_all_none: bool = True) -> list[float]:
    vals = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if vals.empty:
        return []
    qs = np.linspace(0, 1, n_quantiles)
    grid = sorted(set(float(x) for x in vals.quantile(qs).to_numpy()))
    grid.extend([float(vals.min()), float(vals.max())])
    if include_all_none:
        # threshold below min => alert all; above max => alert none
        grid.append(float(vals.min()) - 1e-12)
        grid.append(float(vals.max()) + 1e-12)
    return sorted(set(grid))


def build_segments(df: pd.DataFrame, segment_cols: list[str]) -> list[SegmentInfo]:
    y = df["is_fraud"].to_numpy(dtype=int)
    if not segment_cols:
        return [SegmentInfo(0, {"GLOBAL": "ALL"}, np.arange(len(df)), len(df), int(y.sum()), int(len(df) - y.sum()))]

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
    # Keep min FP for each TP, then remove dominated points.
    best_by_tp: dict[int, dict[str, Any]] = {}
    for opt in options:
        tp = int(opt["tp"])
        old = best_by_tp.get(tp)
        if old is None or int(opt["fp"]) < int(old["fp"]):
            best_by_tp[tp] = opt

    opts = sorted(best_by_tp.values(), key=lambda o: (-int(o["tp"]), int(o["fp"])))
    kept = []
    best_fp_so_far = float("inf")
    for opt in opts:
        fp = int(opt["fp"])
        if fp < best_fp_so_far:
            kept.append(opt)
            best_fp_so_far = fp

    # Sort ascending TP for DP readability.
    kept = sorted(kept, key=lambda o: (int(o["tp"]), int(o["fp"])))
    return kept


def build_segment_options(
    df: pd.DataFrame,
    segments: list[SegmentInfo],
    score_cols: list[str],
    n_quantiles: int,
    max_options_per_segment: int,
) -> tuple[list[list[dict[str, Any]]], pd.DataFrame]:
    y = df["is_fraud"].to_numpy(dtype=int)
    all_options: list[list[dict[str, Any]]] = []
    summary_rows = []

    scores_by_col = {c: num(df, c, 0.0).to_numpy(dtype=float) for c in score_cols if c in df.columns}

    for seg in segments:
        opts = []
        # none option
        opts.append({
            "segment_id": seg.segment_id,
            "score_col": None,
            "threshold": None,
            "direction": "none",
            "tp": 0,
            "fp": 0,
            "n_alerts": 0,
        })
        # all option
        opts.append({
            "segment_id": seg.segment_id,
            "score_col": None,
            "threshold": None,
            "direction": "all",
            "tp": seg.n_pos,
            "fp": seg.n_neg,
            "n_alerts": seg.n_rows,
        })

        for col, scores in scores_by_col.items():
            s = scores[seg.idxs]
            grid = threshold_grid(s, n_quantiles=n_quantiles, include_all_none=True)
            for th in grid:
                pred = s >= th
                tp = int(y[seg.idxs][pred].sum())
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
        # Keep extremes and most useful points if too many.
        if len(opts) > max_options_per_segment:
            # retain evenly across TP, plus low FP options.
            opts_df = pd.DataFrame(opts).sort_values(["tp", "fp"]).reset_index(drop=True)
            idx_keep = set(np.linspace(0, len(opts_df) - 1, max_options_per_segment, dtype=int).tolist())
            idx_keep.update(opts_df.sort_values(["fp", "tp"], ascending=[True, False]).head(max_options_per_segment // 4).index.tolist())
            opts = opts_df.iloc[sorted(idx_keep)].to_dict(orient="records")

        all_options.append(opts)
        summary_rows.append({
            "segment_id": seg.segment_id,
            **seg.segment_values,
            "n_rows": seg.n_rows,
            "n_pos": seg.n_pos,
            "n_neg": seg.n_neg,
            "n_options": len(opts),
            "max_tp": max(int(o["tp"]) for o in opts),
            "min_fp": min(int(o["fp"]) for o in opts),
            "max_fp": max(int(o["fp"]) for o in opts),
        })

    return all_options, pd.DataFrame(summary_rows)


def prune_dp(dp: dict[int, tuple[int, list[int]]]) -> dict[int, tuple[int, list[int]]]:
    # Remove states dominated by another state with >=TP and <=FP.
    items = sorted(dp.items(), key=lambda kv: (-kv[0], kv[1][0]))
    pruned: dict[int, tuple[int, list[int]]] = {}
    best_fp = float("inf")
    for tp, val in items:
        fp = val[0]
        if fp < best_fp:
            pruned[tp] = val
            best_fp = fp
    return pruned


def solve_global_budget(
    all_options: list[list[dict[str, Any]]],
    total_frauds: int,
    target_tps: list[int],
) -> tuple[pd.DataFrame, dict[int, list[int]]]:
    t0 = time.perf_counter()
    dp: dict[int, tuple[int, list[int]]] = {0: (0, [])}

    for seg_idx, opts in enumerate(all_options):
        new_dp: dict[int, tuple[int, list[int]]] = {}
        for tp_cur, (fp_cur, choices) in dp.items():
            for opt_idx, opt in enumerate(opts):
                tp_new = min(total_frauds, tp_cur + int(opt["tp"]))
                fp_new = fp_cur + int(opt["fp"])
                old = new_dp.get(tp_new)
                if old is None or fp_new < old[0]:
                    new_dp[tp_new] = (fp_new, choices + [opt_idx])
        dp = prune_dp(new_dp)

    rows = []
    recipes: dict[int, list[int]] = {}
    possible_tps = sorted(dp.keys())
    for target in sorted(set(target_tps)):
        eligible = [tp for tp in possible_tps if tp >= target]
        if not eligible:
            continue
        best_tp = min(eligible, key=lambda tp: (dp[tp][0], tp))
        fp, choices = dp[best_tp]
        recipes[int(target)] = choices
        rows.append({
            "target_tp": int(target),
            "target_fn": int(total_frauds - target),
            "achieved_tp": int(best_tp),
            "achieved_fn": int(total_frauds - best_tp),
            "base_fp": int(fp),
            "base_precision": float(best_tp / max(best_tp + fp, 1)),
            "base_recall": float(best_tp / max(total_frauds, 1)),
            "base_fpr": None,
        })

    frontier = pd.DataFrame(rows)
    if not frontier.empty:
        frontier = frontier.sort_values(["achieved_fn", "base_fp"], ascending=[True, True]).reset_index(drop=True)
    log(f"DP concluído em {time.perf_counter()-t0:.1f}s com {len(dp)} estados finais.")
    return frontier, recipes


def build_prediction_from_choices(
    df: pd.DataFrame,
    segments: list[SegmentInfo],
    all_options: list[list[dict[str, Any]]],
    choices: list[int],
) -> np.ndarray:
    pred = np.zeros(len(df), dtype=int)
    scores_cache: dict[str, np.ndarray] = {}

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


def choice_recipe(
    segments: list[SegmentInfo],
    all_options: list[list[dict[str, Any]]],
    choices: list[int],
    segment_cols: list[str],
) -> dict[str, Any]:
    out = []
    for seg, opt_idx in zip(segments, choices):
        opt = dict(all_options[seg.segment_id][opt_idx])
        opt["segment_values"] = seg.segment_values
        out.append(opt)
    return {
        "type": "global_recall_budget_dp",
        "segment_cols": segment_cols,
        "segments": out,
    }


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in FEATURE_COLS:
        if c in df.columns:
            out[c] = df[c].astype("string").fillna("<MISSING>").astype(str)
    return out


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


def add_candidate(
    out: list[VetoCandidate],
    prefix: str,
    family: str,
    description: str,
    mask: np.ndarray,
    y: np.ndarray,
    blocks: pd.Series,
    min_fp_removed: int,
    allowed_tp_loss: int,
    max_block_tp_loss: int,
    min_fp_per_tp: float,
    params: dict[str, Any],
) -> None:
    if not mask.any():
        return
    tp_loss, fp_removed, bmax = candidate_stats(mask, y, blocks)
    if fp_removed < min_fp_removed:
        return
    if tp_loss > allowed_tp_loss:
        return
    if bmax > max_block_tp_loss:
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


def mine_numeric_candidates(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    blocks: pd.Series,
    allowed_tp_loss: int,
    min_fp_removed: int,
    max_block_tp_loss: int,
    min_fp_per_tp: float,
    require_module_quiet: bool,
) -> list[VetoCandidate]:
    y = df["is_fraud"].to_numpy(dtype=int)
    pred_pos = base_pred.astype(bool)
    if require_module_quiet and "module_quiet" in df.columns:
        pred_pos = pred_pos & (df["module_quiet"].astype(str).to_numpy() == "module_quiet")

    out: list[VetoCandidate] = []
    for col, direction in NUMERIC_VETO_COLS.items():
        if col not in df.columns:
            continue
        vals = num(df, col, 0.0).to_numpy(dtype=float)
        pp = vals[pred_pos]
        if len(pp) == 0:
            continue
        cuts = []
        try:
            cuts.extend([float(x) for x in np.quantile(pp, [0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70])])
        except Exception:
            pass
        fixed = {
            "lgbm_r4_score": [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1],
            "score_final": [0.5, 1, 2, 3, 5, 10],
            "if_percentile": [0.32, 0.5, 0.7, 0.85],
            "vl_pix": [20, 50, 100, 250],
            "ratio_valor_media_pagador_90d": [0.05, 0.1, 0.2, 0.5, 1],
        }
        cuts.extend(fixed.get(col, []))
        cuts = sorted(set(float(x) for x in cuts if np.isfinite(x)))

        for cut in cuts:
            if direction == "le":
                mask = pred_pos & (vals <= cut)
                desc = f"{col}<={cut:g}"
            else:
                mask = pred_pos & (vals >= cut)
                desc = f"{col}>={cut:g}"
            add_candidate(
                out, "r3f_num", "numeric_precision_veto", desc, mask, y, blocks,
                min_fp_removed, allowed_tp_loss, max_block_tp_loss, min_fp_per_tp,
                {"type": "numeric_threshold", "col": col, "direction": direction, "cut": cut, "require_module_quiet": require_module_quiet},
            )
    return out


def mine_combo_candidates(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    blocks: pd.Series,
    allowed_tp_loss: int,
    min_fp_removed: int,
    max_combo_size: int,
    top_groups_per_combo: int,
    max_block_tp_loss: int,
    min_fp_per_tp: float,
    require_module_quiet: bool,
) -> list[VetoCandidate]:
    y = df["is_fraud"].to_numpy(dtype=int)
    pred_pos = base_pred.astype(bool)
    if require_module_quiet and "module_quiet" in df.columns:
        pred_pos = pred_pos & (df["module_quiet"].astype(str).to_numpy() == "module_quiet")

    feat = feature_frame(df)
    cols = list(feat.columns)
    base_cols = [c for c in ["value_band", "ds_tipo_chave_norm", "periodo_dia", "first_receiver_flag_real", "mbk_available_flag"] if c in cols]
    bin_cols = [c for c in cols if c.endswith("_bin") or c == "module_quiet"]

    combos = []
    for r in range(2, max_combo_size + 1):
        for combo in itertools.combinations(cols, r):
            combo = list(combo)
            if not any(c in base_cols for c in combo):
                continue
            if not any(c in bin_cols for c in combo):
                continue
            combos.append(combo)

    out: list[VetoCandidate] = []
    idx_pos = np.where(pred_pos)[0]
    if len(idx_pos) == 0:
        return out

    for combo in combos:
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

        if not group_rows:
            continue

        group_rows.sort()
        for tp_loss, neg_fp, key, mask, fp_removed, bmax, fp_per_tp in group_rows[:top_groups_per_combo]:
            key_tuple = key if isinstance(key, tuple) else (key,)
            vals = [str(v) for v in key_tuple]
            desc = " AND ".join([f"{c}={v}" for c, v in zip(combo, vals)])
            out.append(VetoCandidate(
                rule_id=f"r3f_combo_{len(out):05d}",
                family="microsegment_precision_veto",
                description=desc,
                mask=mask,
                tp_loss=tp_loss,
                fp_removed=fp_removed,
                n_removed=int(mask.sum()),
                block_tp_loss_max=bmax,
                fp_per_tp=fp_per_tp,
                params={"type": "combo", "combo_cols": combo, "combo_values": vals, "require_module_quiet": require_module_quiet},
            ))
    return out


def dedupe_candidates(cands: list[VetoCandidate]) -> list[VetoCandidate]:
    best: dict[bytes, VetoCandidate] = {}
    for c in cands:
        key = np.packbits(c.mask).tobytes()
        old = best.get(key)
        if old is None or (c.fp_removed, -c.tp_loss, -len(c.description)) > (old.fp_removed, -old.tp_loss, -len(old.description)):
            best[key] = c
    out = list(best.values())
    out.sort(key=lambda c: (c.tp_loss > 0, c.tp_loss, c.block_tp_loss_max, -c.fp_removed, -c.fp_per_tp))
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


def search_best_vetos(
    cands: list[VetoCandidate],
    base_pred: np.ndarray,
    y: np.ndarray,
    target_tp: int,
    max_candidates: int,
    beam_width: int,
    max_rules: int,
    max_seconds: int,
    output_dir: Path,
    point_slug: str,
) -> tuple[pd.DataFrame, State, list[VetoCandidate], str]:
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

    def rank_state(s: State):
        return (s.fp_removed, -s.tp_loss, -len(s.rule_indices))

    def prune_pending(d: dict[bytes, State], keep: int) -> dict[bytes, State]:
        if len(d) <= keep:
            return d
        items = sorted(d.items(), key=lambda kv: rank_state(kv[1]), reverse=True)[:keep]
        return dict(items)

    zero = np.zeros(len(y), dtype=bool)
    initial = State(zero, tuple(), 0, 0)
    states = [initial]
    best = initial
    rows = []
    stop_reason = "completed"

    try:
        for depth in range(1, max_rules + 1):
            if time.perf_counter() - t0 >= max_seconds:
                stop_reason = f"max_seconds_before_depth_{depth}"
                break

            next_states: dict[bytes, State] = {}
            expansions = 0
            prunes = 0
            depth_t0 = time.perf_counter()

            for state in states:
                last = state.rule_indices[-1] if state.rule_indices else -1
                old_total = state.tp_loss + state.fp_removed

                for i in range(last + 1, len(usable)):
                    c = usable[i]
                    new_mask = state.mask | c.mask
                    total = int(new_mask.sum())
                    if total <= old_total:
                        continue

                    if zero_loss_mode:
                        tp_loss = 0
                        fp_removed = total
                    else:
                        tp_loss = int(new_mask[fraud_idx].sum()) if len(fraud_idx) else 0
                        if tp_loss > tp_budget:
                            continue
                        fp_removed = total - tp_loss

                    if fp_removed <= state.fp_removed:
                        continue

                    key = np.packbits(new_mask).tobytes()
                    ns = State(new_mask, state.rule_indices + (i,), tp_loss, fp_removed)
                    old = next_states.get(key)
                    if old is None or rank_state(ns) > rank_state(old):
                        next_states[key] = ns

                    expansions += 1
                    if len(next_states) > pending_limit:
                        next_states = prune_pending(next_states, pending_keep)
                        prunes += 1

                if time.perf_counter() - t0 >= max_seconds:
                    stop_reason = f"max_seconds_during_depth_{depth}"
                    break

            if not next_states:
                if stop_reason.startswith("max_seconds"):
                    break
                stop_reason = f"no_next_states_at_depth_{depth}"
                break

            states = sorted(next_states.values(), key=rank_state, reverse=True)[:beam_width]
            if rank_state(states[0]) > rank_state(best):
                best = states[0]

            for s in states[:50]:
                pred = base_pred.copy()
                pred[s.mask] = 0
                m = compute_metrics(y, pred)
                rows.append({
                    "point_slug": point_slug,
                    "target_tp": target_tp,
                    "depth": depth,
                    "tp_loss": s.tp_loss,
                    "fp_removed": s.fp_removed,
                    "n_rules": len(s.rule_indices),
                    **m,
                    "rule_ids": "|".join(usable[i].rule_id for i in s.rule_indices),
                    "rule_descriptions": " || ".join(usable[i].description for i in s.rule_indices),
                })

            pd.DataFrame(rows).to_csv(output_dir / f"checkpoint_frontier_{point_slug}_depth_{depth:02d}.csv", index=False)
            log(f"    {point_slug} depth={depth}: best_fp_removed={best.fp_removed}, tp_loss={best.tp_loss}/{tp_budget}, states={len(states)}, expansions={expansions}, prunes={prunes}, depth_s={time.perf_counter()-depth_t0:.1f}")

            if time.perf_counter() - t0 >= max_seconds:
                stop_reason = f"max_seconds_after_depth_{depth}"
                break

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt_saved_best"
        log(f"KeyboardInterrupt em {point_slug}; salvando melhor estado.")

    if not rows:
        m = compute_metrics(y, base_pred)
        rows = [{
            "point_slug": point_slug,
            "target_tp": target_tp,
            "depth": 0,
            "tp_loss": 0,
            "fp_removed": 0,
            "n_rules": 0,
            **m,
            "rule_ids": "",
            "rule_descriptions": "",
        }]

    frontier = pd.DataFrame(rows).sort_values(["fn", "fp"], ascending=[True, True]).reset_index(drop=True)
    selected = [usable[i] for i in best.rule_indices]
    return frontier, best, selected, stop_reason


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) / n) + (z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_summary(df: pd.DataFrame, pred_col: str, iters: int, seed: int, target_recall: float) -> pd.DataFrame:
    if iters <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    y_all = df["is_fraud"].to_numpy(dtype=int)
    pred_all = df[pred_col].to_numpy(dtype=int)
    pos_idx = np.where(y_all == 1)[0]
    neg_idx = np.where(y_all == 0)[0]
    rows = []
    for _ in range(iters):
        s_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        s_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([s_pos, s_neg])
        rows.append(compute_metrics(y_all[idx], pred_all[idx]))
    boot = pd.DataFrame(rows)
    out = []
    for metric in ["tp", "fp", "fn", "precision", "recall", "f1", "fpr"]:
        vals = boot[metric].astype(float)
        out.append({
            "method": "stratified_class",
            "metric": metric,
            "mean": float(vals.mean()),
            "p025": float(vals.quantile(0.025)),
            "p050": float(vals.quantile(0.50)),
            "p975": float(vals.quantile(0.975)),
            "target_recall": target_recall if metric == "recall" else None,
            "p_below_target_recall": float((boot["recall"] < target_recall).mean()) if metric == "recall" else None,
        })
    return pd.DataFrame(out)


def parse_target_tps(raw: str | None, total_frauds: int, recall_floor: float) -> list[int]:
    base = set()
    if raw:
        for p in str(raw).split(","):
            p = p.strip()
            if p:
                base.add(int(p))
    else:
        base.update([
            math.ceil(recall_floor * total_frauds),
            math.ceil(0.951 * total_frauds),
            math.ceil(0.955 * total_frauds),
            math.ceil(0.96 * total_frauds),
            math.ceil(0.97 * total_frauds),
            math.ceil(0.98 * total_frauds),
            math.ceil(0.985 * total_frauds),
            math.ceil(0.99 * total_frauds),
            math.ceil(0.995 * total_frauds),
            total_frauds,
        ])
        # Wilson-ish target observed from previous discussion around 1409 for n=1465.
        base.add(min(total_frauds, 1409))
        # explicit FN budgets
        for fn in [0, 1, 2, 5, 10, 20, 30, 50, 72]:
            base.add(total_frauds - fn)

    return sorted(set(x for x in base if 0 <= x <= total_frauds))


def select_frontier_points(frontier: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if frontier.empty:
        return frontier

    rows = []
    # Always include lowest FN points first.
    for _, row in frontier.sort_values(["achieved_fn", "base_fp"]).iterrows():
        rows.append(row)
        if len(rows) >= max_points:
            break

    # Include important 95/Wilson-ish points if not already.
    for _, row in frontier.sort_values(["target_tp"]).iterrows():
        if row["target_tp"] in [1392, 1409] and not any(int(r["target_tp"]) == int(row["target_tp"]) for r in rows):
            rows.append(row)
        if len(rows) >= max_points:
            break

    out = pd.DataFrame(rows).drop_duplicates(subset=["target_tp"]).reset_index(drop=True)
    return out.head(max_points)


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


def make_report(summary: dict[str, Any], frontier: pd.DataFrame, metrics: pd.DataFrame, rules: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014B-R3F — FN-First Global Recall Budget Optimizer")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Seleção: `{summary['selection_policy']}`")
    lines.append(f"- Melhor ponto: `{summary['best_point_slug']}`")
    lines.append(f"- TP/FN final: `{summary['final_metrics']['tp']}` / `{summary['final_metrics']['fn']}`")
    lines.append(f"- FP final: `{summary['final_metrics']['fp']}`")
    lines.append(f"- Recall final: `{summary['final_metrics']['recall']}`")
    lines.append(f"- Precision final: `{summary['final_metrics']['precision']}`")
    lines.append(f"- FPR final: `{summary['final_metrics']['fpr']}`")
    lines.append("")
    lines.append("## Fronteira global FN/FP antes dos vetos")
    show = ["target_tp", "target_fn", "achieved_tp", "achieved_fn", "base_fp", "base_precision", "base_recall"]
    lines.append(frontier[[c for c in show if c in frontier.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Métricas comparativas")
    lines.append(metrics.to_markdown(index=False))
    lines.append("")
    lines.append("## Regras selecionadas no melhor ponto")
    if rules.empty:
        lines.append("Nenhuma regra selecionada.")
    else:
        show_rules = ["rule_id", "family", "description", "tp_loss", "fp_removed", "block_tp_loss_max", "fp_per_tp"]
        lines.append(rules[[c for c in show_rules if c in rules.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretação")
    lines.append("Este experimento não escolhe automaticamente a política de produção; ele revela a fronteira FN First / FP Second. Se o ponto de menor FN tiver FP inviável, o Journal deve registrar esse custo e escolher o ponto de parada operacional.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--segment-cols", default="ds_tipo_chave_norm,mbk_available_flag,value_band")
    parser.add_argument("--score-cols", default=",".join(SCORE_COLS_DEFAULT))
    parser.add_argument("--target-tps", default=None, help="Lista explícita de TPs alvo. Default gera alvos de FN-first.")
    parser.add_argument("--recall-floor-reference", type=float, default=0.95)
    parser.add_argument("--threshold-quantiles", type=int, default=80)
    parser.add_argument("--max-options-per-segment", type=int, default=60)
    parser.add_argument("--max-frontier-points", type=int, default=6)
    parser.add_argument("--time-blocks", type=int, default=10)
    parser.add_argument("--min-fp-removed", type=int, default=30)
    parser.add_argument("--max-combo-size", type=int, default=4)
    parser.add_argument("--top-groups-per-combo", type=int, default=50)
    parser.add_argument("--max-candidate-tp-loss", type=int, default=6)
    parser.add_argument("--max-block-tp-loss", type=int, default=2)
    parser.add_argument("--min-fp-per-tp", type=float, default=150.0)
    parser.add_argument("--max-candidates", type=int, default=700)
    parser.add_argument("--beam-width", type=int, default=180)
    parser.add_argument("--max-rules", type=int, default=8)
    parser.add_argument("--max-seconds-per-point", type=int, default=450)
    parser.add_argument("--bootstrap-iters", type=int, default=100)
    parser.add_argument("--require-module-quiet", action="store_true", default=True)
    parser.add_argument("--selection-policy", choices=["fn_first", "fp_under_95", "balanced"], default="fn_first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-write-predictions", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_point_dir = output_dir / "per_point"
    per_point_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014B-R3F — FN-First Global Recall Budget Optimizer")
    log("=" * 80)
    log(f"Input: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    segment_cols = [c.strip() for c in str(args.segment_cols).split(",") if c.strip()]
    score_cols = [c.strip() for c in str(args.score_cols).split(",") if c.strip()]

    df = add_bins_and_guards(normalize_columns(pd.read_csv(input_path, low_memory=False)))
    ct = contract(df, score_cols)
    dump_json(ct, output_dir / "01_input_contract.json")
    if not ct["contract_ok"]:
        raise RuntimeError(f"Contrato falhou: {ct['missing']}")

    y = df["is_fraud"].to_numpy(dtype=int)
    total_frauds = int(y.sum())
    total_neg = int(len(df) - total_frauds)
    target_tps = parse_target_tps(args.target_tps, total_frauds, args.recall_floor_reference)

    blocks = make_time_blocks(df, args.time_blocks)

    log("[1/8] Construindo segmentos e opções de threshold...")
    segments = build_segments(df, segment_cols)
    all_options, seg_summary = build_segment_options(
        df=df,
        segments=segments,
        score_cols=score_cols,
        n_quantiles=args.threshold_quantiles,
        max_options_per_segment=args.max_options_per_segment,
    )
    seg_summary.to_csv(output_dir / "02_segment_options_summary.csv", index=False)
    log(f"Segmentos={len(segments)}, target_tps={target_tps}")

    log("[2/8] Resolvendo otimizador global de orçamento de recall...")
    frontier, choice_recipes = solve_global_budget(all_options, total_frauds, target_tps)
    if frontier.empty:
        raise RuntimeError("Fronteira DP vazia.")
    frontier["base_fpr"] = frontier["base_fp"] / max(total_neg, 1)
    frontier["fp_gap_vs_r3e"] = frontier["base_fp"] - R3E_BENCHMARK["fp"]
    frontier["fn_gap_vs_r3e"] = frontier["achieved_fn"] - R3E_BENCHMARK["fn"]
    frontier.to_csv(output_dir / "03_global_recall_budget_frontier.csv", index=False)

    selected_points = select_frontier_points(frontier, args.max_frontier_points)
    selected_points.to_csv(output_dir / "04_selected_frontier_points.csv", index=False)

    all_results = []
    all_frontiers = []
    candidate_summary = []
    best_global = None

    log("[3/8] Aplicando vetos por ponto selecionado...")
    for ix, row in selected_points.iterrows():
        point_slug = f"point_{ix:02d}_tp{int(row['target_tp'])}"
        point_dir = per_point_dir / point_slug
        point_dir.mkdir(parents=True, exist_ok=True)

        choices = choice_recipes[int(row["target_tp"])]
        recipe = choice_recipe(segments, all_options, choices, segment_cols)
        base_pred = build_prediction_from_choices(df, segments, all_options, choices)
        base_metrics = compute_metrics(y, base_pred)
        target_tp = int(row["target_tp"])
        tp_budget = max(0, base_metrics["tp"] - target_tp)
        allowed_tp_loss = min(args.max_candidate_tp_loss, tp_budget)

        log("")
        log(f"--- {point_slug}: target_tp={target_tp}, base={base_metrics}, tp_budget={tp_budget}")

        num_cands = mine_numeric_candidates(
            df=df,
            base_pred=base_pred,
            blocks=blocks,
            allowed_tp_loss=allowed_tp_loss,
            min_fp_removed=args.min_fp_removed,
            max_block_tp_loss=args.max_block_tp_loss,
            min_fp_per_tp=args.min_fp_per_tp,
            require_module_quiet=args.require_module_quiet,
        )
        combo_cands = mine_combo_candidates(
            df=df,
            base_pred=base_pred,
            blocks=blocks,
            allowed_tp_loss=allowed_tp_loss,
            min_fp_removed=args.min_fp_removed,
            max_combo_size=args.max_combo_size,
            top_groups_per_combo=args.top_groups_per_combo,
            max_block_tp_loss=args.max_block_tp_loss,
            min_fp_per_tp=args.min_fp_per_tp,
            require_module_quiet=args.require_module_quiet,
        )
        cands = dedupe_candidates(num_cands + combo_cands)
        cdf = candidates_df(cands)
        cdf.to_csv(point_dir / "01_candidates.csv", index=False)
        candidate_summary.append({
            "point_slug": point_slug,
            "target_tp": target_tp,
            "base_tp": base_metrics["tp"],
            "base_fp": base_metrics["fp"],
            "base_fn": base_metrics["fn"],
            "tp_budget": tp_budget,
            "allowed_tp_loss": allowed_tp_loss,
            "n_numeric_candidates": len(num_cands),
            "n_combo_candidates": len(combo_cands),
            "n_candidates": len(cands),
        })

        frontier_after, best, selected_rules, stop_reason = search_best_vetos(
            cands=cands,
            base_pred=base_pred,
            y=y,
            target_tp=target_tp,
            max_candidates=args.max_candidates,
            beam_width=args.beam_width,
            max_rules=args.max_rules,
            max_seconds=args.max_seconds_per_point,
            output_dir=point_dir,
            point_slug=point_slug,
        )
        frontier_after.to_csv(point_dir / "02_frontier_after_vetos.csv", index=False)
        rules_df = candidates_df(selected_rules)
        rules_df.to_csv(point_dir / "03_selected_rules.csv", index=False)

        final_pred = base_pred.copy()
        final_pred[best.mask] = 0
        final_metrics = compute_metrics(y, final_pred)

        fp_removed = base_metrics["fp"] - final_metrics["fp"]
        tp_loss = base_metrics["tp"] - final_metrics["tp"]
        wl, wh = wilson_ci(final_metrics["tp"], total_frauds)

        result = {
            "point_slug": point_slug,
            "target_tp": target_tp,
            "target_fn": total_frauds - target_tp,
            "base_recipe": recipe,
            "base_metrics": base_metrics,
            "final_metrics": final_metrics,
            "tp_budget": tp_budget,
            "tp_loss_vs_base": int(tp_loss),
            "fp_removed_vs_base": int(fp_removed),
            "n_candidates": len(cands),
            "n_selected_rules": len(selected_rules),
            "stop_reason": stop_reason,
            "wilson_low": wl,
            "wilson_high": wh,
            "selected_rules": rules_df.to_dict(orient="records") if not rules_df.empty else [],
        }
        dump_json(result, point_dir / "04_point_result.json")
        all_results.append(result)
        all_frontiers.append(frontier_after)

        if args.selection_policy == "fn_first":
            rank = (final_metrics["fn"], final_metrics["fp"], -final_metrics["precision"])
        elif args.selection_policy == "fp_under_95":
            # best FP among policies still >= reference 95% recall
            valid = final_metrics["recall"] >= args.recall_floor_reference
            rank = (0 if valid else 1, final_metrics["fp"], final_metrics["fn"])
        else:
            # balanced: FN-first but punishes extreme FP
            rank = (final_metrics["fn"], final_metrics["fpr"], final_metrics["fp"])

        if best_global is None or rank < best_global["rank"]:
            best_global = {
                "rank": rank,
                "point_slug": point_slug,
                "target_tp": target_tp,
                "base_pred": base_pred,
                "final_pred": final_pred,
                "base_metrics": base_metrics,
                "final_metrics": final_metrics,
                "tp_budget": tp_budget,
                "tp_loss_vs_base": int(tp_loss),
                "fp_removed_vs_base": int(fp_removed),
                "selected_rules": selected_rules,
                "selected_rules_df": rules_df,
                "stop_reason": stop_reason,
                "base_recipe": recipe,
                "wilson_low": wl,
                "wilson_high": wh,
            }

    if best_global is None:
        raise RuntimeError("Nenhum ponto global foi produzido.")

    pd.DataFrame(all_results).to_json(output_dir / "05_point_results.json", orient="records", force_ascii=False, indent=2)
    pd.DataFrame(candidate_summary).to_csv(output_dir / "06_candidate_summary.csv", index=False)
    if all_frontiers:
        pd.concat(all_frontiers, ignore_index=True).to_csv(output_dir / "07_frontier_after_vetos_all.csv", index=False)

    log("[4/8] Consolidando melhor política FN-first...")
    df["exp014b_r3f_selected_base_pred"] = best_global["base_pred"].astype(int)
    df["exp014b_r3f_final_pred"] = best_global["final_pred"].astype(int)

    selected_rules_df = best_global["selected_rules_df"]
    selected_rules_df.to_csv(output_dir / "08_selected_rules_best.csv", index=False)

    log("[5/8] Métricas comparativas...")
    policy_rows = []
    for c in ["exp014a_frozen_pred", "exp013k_residual_fp_pred", "exp014b_r3d_final_pred", "exp014b_r3e_final_pred"]:
        if c in df.columns:
            policy_rows.append({"policy_name": c, **compute_metrics(y, df[c].to_numpy(dtype=int))})
    policy_rows.append({"policy_name": "EXP014B_R3F_SELECTED_BASE", **best_global["base_metrics"]})
    policy_rows.append({"policy_name": "EXP014B_R3F_FINAL", **best_global["final_metrics"]})

    metrics_df = pd.DataFrame(policy_rows)
    metrics_df["fn_gap_vs_r3e"] = metrics_df["fn"] - R3E_BENCHMARK["fn"]
    metrics_df["fp_gap_vs_r3e"] = metrics_df["fp"] - R3E_BENCHMARK["fp"]
    metrics_df["fpr_gap_vs_small"] = metrics_df["fpr"] - SMALL_BENCHMARK["fpr"]
    metrics_df["precision_gap_vs_small"] = metrics_df["precision"] - SMALL_BENCHMARK["precision"]
    metrics_df.to_csv(output_dir / "09_policy_metrics.csv", index=False)

    block_df = pd.concat([
        block_metrics(df, df["exp014b_r3f_selected_base_pred"].to_numpy(dtype=int), blocks, "EXP014B_R3F_SELECTED_BASE"),
        block_metrics(df, df["exp014b_r3f_final_pred"].to_numpy(dtype=int), blocks, "EXP014B_R3F_FINAL"),
    ], ignore_index=True)
    block_df.to_csv(output_dir / "10_time_block_metrics.csv", index=False)

    log("[6/8] Wilson, bootstrap, erros...")
    final_metrics = best_global["final_metrics"]
    wl, wh = wilson_ci(final_metrics["tp"], total_frauds)
    wilson_df = pd.DataFrame([{
        "metric": "recall",
        "successes_tp": final_metrics["tp"],
        "n_frauds": total_frauds,
        "point_estimate": final_metrics["recall"],
        "wilson_low": wl,
        "wilson_high": wh,
        "reference_recall_floor": args.recall_floor_reference,
        "tp_vs_95_floor": final_metrics["tp"] - math.ceil(args.recall_floor_reference * total_frauds),
        "wilson_low_ge_95_floor": bool(wl >= args.recall_floor_reference),
    }])
    wilson_df.to_csv(output_dir / "11_wilson_recall_ci.csv", index=False)

    boot_df = bootstrap_summary(df, "exp014b_r3f_final_pred", args.bootstrap_iters, args.seed, args.recall_floor_reference)
    boot_df.to_csv(output_dir / "12_bootstrap_summary.csv", index=False)

    df[(df["is_fraud"] == 1) & (df["exp014b_r3f_final_pred"] == 0)].to_csv(output_dir / "13_false_negatives.csv", index=False)
    fp = df[(df["is_fraud"] == 0) & (df["exp014b_r3f_final_pred"] == 1)].copy()
    if len(fp) > 5000:
        fp = fp.sample(5000, random_state=args.seed)
    fp.to_csv(output_dir / "14_false_positives_sample.csv", index=False)

    if not args.no_write_predictions:
        df.to_csv(output_dir / "16_predictions.csv", index=False)

    objective_status = "DONE"
    objective_status += "_FN_IMPROVED_VS_R3E" if final_metrics["fn"] < R3E_BENCHMARK["fn"] else "_FN_NOT_IMPROVED_VS_R3E"
    objective_status += "_FP_REDUCED_VS_BASE" if best_global["fp_removed_vs_base"] > 0 else "_FP_NOT_REDUCED_VS_BASE"
    objective_status += "_WILSON_PASS_95" if wl >= args.recall_floor_reference else "_WILSON_NOT_PASS_95"
    objective_status += "_FN_ZERO" if final_metrics["fn"] == 0 else "_FN_GT_ZERO"

    artifact = {
        "experiment": "EXP-014B-R3F",
        "policy_name": "fn_first_global_recall_budget_optimizer",
        "objective_status": objective_status,
        "selection_policy": args.selection_policy,
        "small_benchmark": SMALL_BENCHMARK,
        "r3e_benchmark": R3E_BENCHMARK,
        "total_frauds": total_frauds,
        "target_tps": target_tps,
        "best_point_slug": best_global["point_slug"],
        "best_target_tp": best_global["target_tp"],
        "best_base_recipe": best_global["base_recipe"],
        "base_metrics": best_global["base_metrics"],
        "final_metrics": final_metrics,
        "tp_budget": best_global["tp_budget"],
        "tp_loss_vs_base": best_global["tp_loss_vs_base"],
        "fp_removed_vs_base": best_global["fp_removed_vs_base"],
        "wilson": wilson_df.to_dict(orient="records")[0],
        "selected_rules": selected_rules_df.to_dict(orient="records") if not selected_rules_df.empty else [],
        "notes": [
            "FN First, FP Second.",
            "Global DP optimizer minimizes FP for each target TP.",
            "Veto phase preserves the selected target TP budget.",
            "This is frontier discovery, not automatic promotion."
        ],
    }
    dump_json(artifact, output_dir / "15_policy_artifact.json")

    summary = {
        "experiment": "EXP-014B-R3F",
        "status": "DONE",
        "objective_status": objective_status,
        "selection_policy": args.selection_policy,
        "input_path": str(input_path),
        "n_rows": int(len(df)),
        "n_frauds": total_frauds,
        "total_negatives": total_neg,
        "segment_cols": segment_cols,
        "score_cols": [c for c in score_cols if c in df.columns],
        "target_tps": target_tps,
        "best_point_slug": best_global["point_slug"],
        "best_target_tp": best_global["target_tp"],
        "base_metrics": best_global["base_metrics"],
        "final_metrics": final_metrics,
        "tp_budget": best_global["tp_budget"],
        "tp_loss_vs_base": best_global["tp_loss_vs_base"],
        "fp_removed_vs_base": best_global["fp_removed_vs_base"],
        "n_selected_rules": int(len(best_global["selected_rules"])),
        "stop_reason": best_global["stop_reason"],
        "wilson_recall_low": wl,
        "wilson_recall_high": wh,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, frontier, metrics_df, selected_rules_df)
    (output_dir / "17_exp014b_r3f_report.md").write_text(report, encoding="utf-8")

    log("[8/8] Concluído.")
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "02_segment_options_summary.csv",
        output_dir / "03_global_recall_budget_frontier.csv",
        output_dir / "04_selected_frontier_points.csv",
        output_dir / "05_point_results.json",
        output_dir / "06_candidate_summary.csv",
        output_dir / "07_frontier_after_vetos_all.csv",
        output_dir / "08_selected_rules_best.csv",
        output_dir / "09_policy_metrics.csv",
        output_dir / "10_time_block_metrics.csv",
        output_dir / "11_wilson_recall_ci.csv",
        output_dir / "12_bootstrap_summary.csv",
        output_dir / "15_policy_artifact.json",
        output_dir / "17_exp014b_r3f_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
