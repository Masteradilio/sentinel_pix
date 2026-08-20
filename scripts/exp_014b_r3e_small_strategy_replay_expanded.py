#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3E — Small-Strategy Replay on Expanded Dataset

Objetivo:
  Reaplicar no dataset expandido a lógica que levou ao benchmark campeão no
  dataset pequeno, agora sem depender da coluna original pred_STRICT_RECALL95_SAFE_ONLY.

Estratégia inspirada no journal:
  1. Primeiro estágio high-recall LGBM.
     - No dataset pequeno, o R4 venceu usando thresholds segmentados por
       mbk_available_flag + ds_tipo_chave_norm.
     - Aqui testamos bases globais e segmentadas, com targets 95%, 95.5%, 96% e 97%.

  2. Criar buffer de recall antes de reduzir FP.
     - O R3D ficou TP=1392, exatamente no mínimo para recall >=95%.
     - Sem buffer, qualquer perda de TP quebra o alvo.
     - O R3E testa bases um pouco mais largas para permitir vetos TP1/TP2
       somente quando a troca FP/TP for excelente.

  3. Reaplicar a camada estatística/microveto:
     - vetos por sinais estatísticos: LGBM, IF, score_final, valor, ratio e histórico;
     - microsegmentos: value_band, tipo chave, período, first_receiver, MBK, bins;
     - preservar module_strong: SE/BEH/runtime fortes não devem ser vetados.

  4. Escolher a menor quantidade de FP com recall final >=95%.

Uso padrão:
  python scripts/exp_014b_r3e_small_strategy_replay_expanded.py

Execução rápida:
  python scripts/exp_014b_r3e_small_strategy_replay_expanded.py --max-bases 4 --max-candidates 500 --beam-width 120 --max-rules 6 --max-seconds-per-base 240 --bootstrap-iters 50

Execução profunda:
  python scripts/exp_014b_r3e_small_strategy_replay_expanded.py --max-bases 10 --max-combo-size 4 --max-candidates 900 --beam-width 260 --max-rules 10 --max-seconds-per-base 600 --bootstrap-iters 100
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3E"

SMALL_BENCHMARK = {
    "source": "SMALL_REAPPLIED_EXP013K_POLICY",
    "tp": 118,
    "fp": 199,
    "fn": 6,
    "recall": 0.9516,
    "precision": 0.3722,
    "fpr": 0.0202,
}

SCORE_COLS = [
    "lgbm_r4_score",
    "r4_score",
    "lgbm_mapped",
    "lgbm_raw",
    "score_final",
]

BASE_EXISTING_COLS = [
    "pred_STRICT_RECALL95_SAFE_ONLY",
    "exp013k_base_pred",
    "exp013h_frozen_pred",
    "exp013g_micro_pred",
    "pred_HIGH_RECALL_95",
    "exp014b_r3d_selected_base_pred",
]

SEGMENT_SPECS = [
    ("GLOBAL", []),
    ("SEG_MBK", ["mbk_available_flag"]),
    ("SEG_CHAVE", ["ds_tipo_chave_norm"]),
    ("SEG_MBK_CHAVE", ["mbk_available_flag", "ds_tipo_chave_norm"]),
    ("SEG_VALUE", ["value_band"]),
    ("SEG_MBK_VALUE", ["mbk_available_flag", "value_band"]),
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
    "if_percentile": "le",
    "if_percentile_x": "le",
    "if_percentile_y": "le",
    "score_final": "le",
    "vl_pix": "le",
    "ratio_valor_media_pagador_90d": "le",
    "qtd_pix_recebidos_180d": "ge",
    "valor_total_recebido_180d": "ge",
}


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

    for c in BASE_EXISTING_COLS + ["runtime_flagged", "exp014a_frozen_pred", "exp013k_residual_fp_pred"]:
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


def contract(df: pd.DataFrame) -> dict[str, Any]:
    missing = []
    if "is_fraud" not in df.columns:
        missing.append("is_fraud")
    if not any(c in df.columns for c in SCORE_COLS):
        missing.append("score_column")
    if not any(c in df.columns for c in ["event_datetime", "data_pix", "dt_pix"]):
        missing.append("date_column")
    return {
        "contract_ok": not missing,
        "missing": missing,
        "n_rows": int(len(df)),
        "n_frauds": int(df["is_fraud"].sum()) if "is_fraud" in df.columns else None,
        "score_cols_present": [c for c in SCORE_COLS if c in df.columns],
        "existing_pred_cols_present": [c for c in BASE_EXISTING_COLS if c in df.columns],
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


def threshold_grid(s: pd.Series, n: int) -> list[float]:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if vals.empty:
        return []
    qs = np.linspace(0, 1, n)
    vals2 = sorted(set(float(x) for x in vals.quantile(qs).to_numpy()))
    vals2.extend([float(vals.min()), float(vals.max())])
    return sorted(set(vals2))


def select_threshold_ge(scores: np.ndarray, y: np.ndarray, mask: np.ndarray, target_recall: float, grid: list[float]) -> tuple[float, dict[str, Any]]:
    idx = np.asarray(mask, dtype=bool)
    y_sub = y[idx]
    s_sub = scores[idx]
    n_pos = int(y_sub.sum())

    if len(y_sub) == 0:
        return float("inf"), {"tp": 0, "fp": 0, "fn": 0, "recall": 1.0, "reason": "empty_segment"}

    if n_pos == 0:
        th = float(np.nanmax(s_sub) + 1e-12) if len(s_sub) else float("inf")
        return th, {"tp": 0, "fp": 0, "fn": 0, "recall": 1.0, "reason": "no_positives"}

    best = None
    for th in grid:
        pred = (s_sub >= th).astype(int)
        m = compute_metrics(y_sub, pred)
        if m["recall"] >= target_recall:
            rank = (m["fp"], -m["precision"], -m["tp"], th)
            if best is None or rank < best[0]:
                best = (rank, float(th), m)

    if best is None:
        # Maximum recall fallback.
        th = min(grid) if grid else float(np.nanmin(s_sub))
        pred = (s_sub >= th).astype(int)
        return float(th), {**compute_metrics(y_sub, pred), "reason": "target_not_met_min_threshold"}

    return best[1], {**best[2], "reason": "target_met"}


def build_global_base(df: pd.DataFrame, score_col: str, target_recall: float, n_grid: int) -> tuple[np.ndarray, dict[str, Any]]:
    y = df["is_fraud"].to_numpy(dtype=int)
    scores = num(df, score_col, 0.0).to_numpy(dtype=float)
    grid = threshold_grid(pd.Series(scores), n_grid)
    th, info = select_threshold_ge(scores, y, np.ones(len(df), dtype=bool), target_recall, grid)
    pred = (scores >= th).astype(int)
    recipe = {
        "type": "global_threshold",
        "score_col": score_col,
        "direction": "ge",
        "threshold": th,
        "target_recall": target_recall,
        "threshold_info": info,
    }
    return pred, recipe


def build_segmented_base(
    df: pd.DataFrame,
    score_col: str,
    segment_cols: list[str],
    target_recall: float,
    n_grid: int,
    min_frauds_per_segment: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    y = df["is_fraud"].to_numpy(dtype=int)
    scores = num(df, score_col, 0.0).to_numpy(dtype=float)
    grid = threshold_grid(pd.Series(scores), n_grid)
    fallback_th, fallback_info = select_threshold_ge(scores, y, np.ones(len(df), dtype=bool), target_recall, grid)

    if not segment_cols:
        pred = (scores >= fallback_th).astype(int)
        return pred, {
            "type": "global_threshold",
            "score_col": score_col,
            "threshold": fallback_th,
            "target_recall": target_recall,
            "threshold_info": fallback_info,
        }

    seg_df = pd.DataFrame(index=df.index)
    for c in segment_cols:
        if c not in df.columns:
            raise RuntimeError(f"Coluna de segmento ausente: {c}")
        seg_df[c] = df[c].astype("string").fillna("<MISSING>").astype(str)

    pred = np.zeros(len(df), dtype=int)
    segments = []
    grouped = seg_df.groupby(segment_cols, dropna=False).indices

    for key, rel_idxs in grouped.items():
        idxs = np.asarray(list(rel_idxs), dtype=int)
        mask = np.zeros(len(df), dtype=bool)
        mask[idxs] = True
        n_pos = int(y[mask].sum())

        if n_pos >= min_frauds_per_segment:
            th, info = select_threshold_ge(scores, y, mask, target_recall, grid)
            source = "segment_threshold"
        else:
            th, info = fallback_th, {"reason": "fallback_low_positive_segment", "n_pos": n_pos}
            source = "fallback_global"

        pred[mask] = (scores[mask] >= th).astype(int)
        vals = key if isinstance(key, tuple) else (key,)
        segments.append({
            "segment_cols": segment_cols,
            "segment_values": [str(x) for x in vals],
            "threshold": float(th),
            "n_rows": int(mask.sum()),
            "n_frauds": n_pos,
            "source": source,
            "info": info,
        })

    recipe = {
        "type": "segmented_thresholds",
        "score_col": score_col,
        "direction": "ge",
        "segment_cols": segment_cols,
        "target_recall": target_recall,
        "fallback_threshold": float(fallback_th),
        "fallback_info": fallback_info,
        "segments": segments,
    }
    return pred, recipe


def temporal_validation_masks(blocks: pd.Series, validation_blocks: int) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(blocks.dropna().unique())
    validation_blocks = min(max(1, validation_blocks), max(1, len(unique) - 1))
    val_set = set(unique[-validation_blocks:])
    val = blocks.isin(val_set).to_numpy(dtype=bool)
    return ~val, val


def evaluate_base_pool(
    df: pd.DataFrame,
    blocks: pd.Series,
    base_targets: list[float],
    n_grid: int,
    min_frauds_per_segment: int,
    validation_blocks: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    y = df["is_fraud"].to_numpy(dtype=int)
    _, val_mask = temporal_validation_masks(blocks, validation_blocks)

    rows = []
    preds = {}
    recipes = {}

    # Existing columns.
    for col in [c for c in BASE_EXISTING_COLS if c in df.columns]:
        pred = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).to_numpy()
        key = f"EXISTING__{col}"
        preds[key] = pred
        recipes[key] = {"type": "existing_column", "column": col}
        m = compute_metrics(y, pred)
        mv = compute_metrics(y[val_mask], pred[val_mask])
        mb = block_metrics(df, pred, blocks, key)
        rows.append({
            "base_key": key,
            "family": "EXISTING",
            "score_col": None,
            "segment_spec": "EXISTING",
            "base_target_recall": None,
            **{f"full_{k}": v for k, v in m.items()},
            **{f"val_{k}": v for k, v in mv.items()},
            "min_block_recall": float(mb["recall"].min()) if not mb.empty else np.nan,
        })

    for score_col in [c for c in SCORE_COLS if c in df.columns]:
        for target in base_targets:
            for spec_name, seg_cols in SEGMENT_SPECS:
                missing = [c for c in seg_cols if c not in df.columns]
                if missing:
                    continue

                pred, recipe = build_segmented_base(
                    df=df,
                    score_col=score_col,
                    segment_cols=seg_cols,
                    target_recall=target,
                    n_grid=n_grid,
                    min_frauds_per_segment=min_frauds_per_segment,
                )
                key = f"{spec_name}__{score_col}__target_{target:.4f}"
                preds[key] = pred
                recipes[key] = recipe

                m = compute_metrics(y, pred)
                mv = compute_metrics(y[val_mask], pred[val_mask])
                mb = block_metrics(df, pred, blocks, key)
                rows.append({
                    "base_key": key,
                    "family": spec_name,
                    "score_col": score_col,
                    "segment_spec": "+".join(seg_cols) if seg_cols else "GLOBAL",
                    "base_target_recall": target,
                    **{f"full_{k}": v for k, v in m.items()},
                    **{f"val_{k}": v for k, v in mv.items()},
                    "min_block_recall": float(mb["recall"].min()) if not mb.empty else np.nan,
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out, preds, recipes

    out["passes_global_target"] = out["full_recall"] >= min(base_targets)
    out["fpr_gap_vs_small"] = out["full_fpr"] - SMALL_BENCHMARK["fpr"]
    out["precision_gap_vs_small"] = out["full_precision"] - SMALL_BENCHMARK["precision"]
    out["type_rank"] = out["family"].map({"SEG_MBK_CHAVE": 0, "GLOBAL": 1, "SEG_MBK": 2, "SEG_CHAVE": 3, "SEG_VALUE": 4, "SEG_MBK_VALUE": 5, "EXISTING": 6}).fillna(9)
    out = out.sort_values(["passes_global_target", "full_fp", "full_recall", "type_rank"], ascending=[False, True, False, True]).reset_index(drop=True)
    return out, preds, recipes


def select_base_candidates(base_df: pd.DataFrame, max_bases: int) -> list[str]:
    if base_df.empty:
        return []

    selected = []
    # Always include top global low-FP.
    for _, row in base_df[base_df["passes_global_target"]].head(max_bases * 2).iterrows():
        key = str(row["base_key"])
        if key not in selected:
            selected.append(key)
        if len(selected) >= max_bases:
            break

    # Force inclusion of best per family/target when possible.
    for _, grp in base_df[base_df["passes_global_target"]].groupby(["family", "base_target_recall"], dropna=False):
        key = str(grp.sort_values(["full_fp", "full_recall"], ascending=[True, False]).iloc[0]["base_key"])
        if key not in selected:
            selected.append(key)
        if len(selected) >= max_bases:
            break

    return selected[:max_bases]


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in FEATURE_COLS:
        if c in df.columns:
            out[c] = df[c].astype("string").fillna("<MISSING>").astype(str)
    return out


def block_tp_loss_max(mask: np.ndarray, y: np.ndarray, blocks: pd.Series) -> int:
    bvals = blocks.to_numpy()
    max_loss = 0
    for b in sorted(blocks.dropna().unique()):
        bm = mask & (bvals == b)
        max_loss = max(max_loss, int(mask[bm & (y == 1)].sum()) if False else int(((y == 1) & bm).sum()))
    return max_loss


def candidate_stats(mask: np.ndarray, y: np.ndarray, blocks: pd.Series) -> tuple[int, int, int]:
    tp_loss = int(mask[y == 1].sum())
    fp_removed = int(mask[y == 0].sum())
    bmax = 0
    bvals = blocks.to_numpy()
    for b in sorted(blocks.dropna().unique()):
        bm = mask & (bvals == b)
        bmax = max(bmax, int(bm[y == 1].sum()))
    return tp_loss, fp_removed, bmax


def add_candidate(
    out: list[VetoCandidate],
    rule_id_prefix: str,
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
        rule_id=f"{rule_id_prefix}_{len(out):05d}",
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

    fixed_cuts = {
        "lgbm_r4_score": [0.02, 0.03, 0.04, 0.05, 0.075, 0.1],
        "r4_score": [0.02, 0.03, 0.04, 0.05, 0.075, 0.1],
        "lgbm_mapped": [0.02, 0.03, 0.04, 0.05, 0.075, 0.1],
        "lgbm_raw": [0.02, 0.03, 0.04, 0.05, 0.075, 0.1],
        "if_percentile": [0.32, 0.5, 0.7, 0.85],
        "if_percentile_x": [0.32, 0.5, 0.7, 0.85],
        "if_percentile_y": [0.32, 0.5, 0.7, 0.85],
        "score_final": [0.5, 0.76, 1, 2, 3, 5, 10],
        "vl_pix": [20, 50, 100, 250],
        "ratio_valor_media_pagador_90d": [0.05, 0.1, 0.2, 0.5, 1.0],
        "qtd_pix_recebidos_180d": [5, 10, 20, 50, 100],
        "valor_total_recebido_180d": [1000, 2000, 5000, 10000, 25000],
    }

    for col, direction in NUMERIC_VETO_COLS.items():
        if col not in df.columns:
            continue
        vals = num(df, col, 0.0).to_numpy(dtype=float)
        pp = vals[pred_pos]
        if len(pp) == 0:
            continue

        cuts = list(fixed_cuts.get(col, []))
        try:
            cuts.extend([float(x) for x in np.quantile(pp, [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70])])
        except Exception:
            pass
        cuts = sorted(set([float(x) for x in cuts if np.isfinite(x)]))

        for cut in cuts:
            if direction == "le":
                mask = pred_pos & (vals <= cut)
                desc = f"{col}<={cut:g}"
            else:
                mask = pred_pos & (vals >= cut)
                desc = f"{col}>={cut:g}"

            add_candidate(
                out, "r3e_num", "statistical_numeric_veto", desc, mask, y, blocks,
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

        group_rows = []
        grouped = subset.groupby(combo, dropna=False).indices
        for key, rel_idxs in grouped.items():
            idxs = subset.iloc[list(rel_idxs)].index.to_numpy(dtype=int)
            if len(idxs) < min_fp_removed:
                continue
            mask = np.zeros(len(df), dtype=bool)
            mask[idxs] = True
            mask = mask & pred_pos

            tp_loss, fp_removed, bmax = candidate_stats(mask, y, blocks)
            if fp_removed < min_fp_removed:
                continue
            if tp_loss > allowed_tp_loss:
                continue
            if bmax > max_block_tp_loss:
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
                rule_id=f"r3e_combo_{len(out):05d}",
                family="microsegment_combo_veto",
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
    tp_budget: int,
    max_candidates: int,
    beam_width: int,
    max_rules: int,
    max_seconds: int,
    output_dir: Path,
    base_slug: str,
) -> tuple[pd.DataFrame, State, list[VetoCandidate], str]:
    t0 = time.perf_counter()

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
                    "base_key": base_slug,
                    "depth": depth,
                    "tp_loss": s.tp_loss,
                    "fp_removed": s.fp_removed,
                    "n_rules": len(s.rule_indices),
                    **m,
                    "rule_ids": "|".join(usable[i].rule_id for i in s.rule_indices),
                    "rule_descriptions": " || ".join(usable[i].description for i in s.rule_indices),
                })

            pd.DataFrame(rows).to_csv(output_dir / f"checkpoint_frontier_{base_slug}_depth_{depth:02d}.csv", index=False)
            log(f"    {base_slug} depth={depth}: best_fp_removed={best.fp_removed}, tp_loss={best.tp_loss}/{tp_budget}, states={len(states)}, expansions={expansions}, prunes={prunes}, depth_s={time.perf_counter()-depth_t0:.1f}")

            if time.perf_counter() - t0 >= max_seconds:
                stop_reason = f"max_seconds_after_depth_{depth}"
                break

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt_saved_best"
        log(f"KeyboardInterrupt em {base_slug}; salvando melhor estado.")

    if not rows:
        m = compute_metrics(y, base_pred)
        rows = [{
            "base_key": base_slug,
            "depth": 0,
            "tp_loss": 0,
            "fp_removed": 0,
            "n_rules": 0,
            **m,
            "rule_ids": "",
            "rule_descriptions": "",
        }]

    frontier = pd.DataFrame(rows).sort_values(["fp", "tp"], ascending=[True, False]).reset_index(drop=True)
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


def make_report(summary: dict[str, Any], base_df: pd.DataFrame, metrics_df: pd.DataFrame, selected_rules: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014B-R3E — Small-Strategy Replay on Expanded Dataset")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Melhor base: `{summary['best_base_key']}`")
    lines.append(f"- FP final: `{summary['final_metrics']['fp']}`")
    lines.append(f"- Recall final: `{summary['final_metrics']['recall']}`")
    lines.append(f"- Precision final: `{summary['final_metrics']['precision']}`")
    lines.append(f"- FPR final: `{summary['final_metrics']['fpr']}`")
    lines.append(f"- Wilson low: `{summary['wilson_recall_low']}`")
    lines.append("")
    lines.append("## Benchmark pequeno")
    lines.append("```json")
    lines.append(json.dumps(SMALL_BENCHMARK, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Métricas comparativas")
    lines.append(metrics_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Top bases")
    show_cols = ["base_key", "family", "score_col", "segment_spec", "base_target_recall", "full_tp", "full_fp", "full_fn", "full_precision", "full_recall", "full_fpr", "val_recall", "min_block_recall"]
    lines.append(base_df[[c for c in show_cols if c in base_df.columns]].head(30).to_markdown(index=False))
    lines.append("")
    lines.append("## Regras selecionadas")
    if selected_rules.empty:
        lines.append("Nenhuma regra selecionada.")
    else:
        cols = ["rule_id", "family", "description", "tp_loss", "fp_removed", "block_tp_loss_max", "fp_per_tp"]
        lines.append(selected_rules[[c for c in cols if c in selected_rules.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretação")
    if summary["final_metrics"]["recall"] >= summary["target_recall"]:
        lines.append("A política mantém recall >=95%. Comparar FP/FPR/precision contra o R3D e contra o benchmark pequeno antes de promover.")
    else:
        lines.append("A política não mantém recall >=95%; não promover.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--base-targets", default="0.95,0.955,0.96,0.97")
    parser.add_argument("--threshold-grid", type=int, default=1000)
    parser.add_argument("--min-frauds-per-segment", type=int, default=20)
    parser.add_argument("--max-bases", type=int, default=8)
    parser.add_argument("--time-blocks", type=int, default=10)
    parser.add_argument("--validation-blocks", type=int, default=3)
    parser.add_argument("--min-fp-removed", type=int, default=30)
    parser.add_argument("--max-combo-size", type=int, default=4)
    parser.add_argument("--top-groups-per-combo", type=int, default=50)
    parser.add_argument("--max-candidate-tp-loss", type=int, default=4)
    parser.add_argument("--max-block-tp-loss", type=int, default=2)
    parser.add_argument("--min-fp-per-tp", type=float, default=120.0)
    parser.add_argument("--max-candidates", type=int, default=700)
    parser.add_argument("--beam-width", type=int, default=200)
    parser.add_argument("--max-rules", type=int, default=10)
    parser.add_argument("--max-seconds-per-base", type=int, default=450)
    parser.add_argument("--bootstrap-iters", type=int, default=100)
    parser.add_argument("--require-module-quiet", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-write-predictions", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_base_dir = output_dir / "per_base"
    per_base_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014B-R3E — Small-Strategy Replay on Expanded Dataset")
    log("=" * 80)
    log(f"Input: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    df = add_bins_and_guards(normalize_columns(pd.read_csv(input_path, low_memory=False)))
    ct = contract(df)
    dump_json(ct, output_dir / "01_input_contract.json")
    if not ct["contract_ok"]:
        raise RuntimeError(f"Contrato falhou: {ct['missing']}")

    y = df["is_fraud"].to_numpy(dtype=int)
    total_frauds = int(df["is_fraud"].sum())
    min_tp_required = int(math.ceil(args.target_recall * total_frauds))
    blocks = make_time_blocks(df, args.time_blocks)
    base_targets = sorted(set(float(x.strip()) for x in str(args.base_targets).split(",") if x.strip()))

    log("[1/7] Construindo bases globais e segmentadas...")
    base_df, base_preds, base_recipes = evaluate_base_pool(
        df=df,
        blocks=blocks,
        base_targets=base_targets,
        n_grid=args.threshold_grid,
        min_frauds_per_segment=args.min_frauds_per_segment,
        validation_blocks=args.validation_blocks,
    )
    base_df.to_csv(output_dir / "02_base_candidates.csv", index=False)
    dump_json(base_recipes, output_dir / "03_base_recipes.json")

    base_keys = select_base_candidates(base_df, args.max_bases)
    if not base_keys:
        raise RuntimeError("Nenhuma base candidata foi selecionada.")
    pd.DataFrame({"base_key": base_keys}).to_csv(output_dir / "04_selected_base_pool.csv", index=False)
    log(f"Bases no pool: {len(base_keys)}")

    global_results = []
    all_frontiers = []
    all_candidate_summaries = []
    best_global = None

    log("[2/7] Rodando mineração e beam por base...")
    for base_key in base_keys:
        base_slug = "base_" + str(len(global_results)).zfill(2)
        base_out = per_base_dir / base_slug
        base_out.mkdir(parents=True, exist_ok=True)

        base_pred = base_preds[base_key].copy()
        base_metrics = compute_metrics(y, base_pred)
        tp_budget = max(0, base_metrics["tp"] - min_tp_required)

        log("")
        log(f"--- {base_slug}: {base_key}")
        log(f"    base_metrics={base_metrics}, tp_budget={tp_budget}")

        allowed_tp_loss = min(args.max_candidate_tp_loss, tp_budget)

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
        cdf.to_csv(base_out / "01_candidates.csv", index=False)
        all_candidate_summaries.append({
            "base_slug": base_slug,
            "base_key": base_key,
            "base_tp": base_metrics["tp"],
            "base_fp": base_metrics["fp"],
            "base_fn": base_metrics["fn"],
            "base_recall": base_metrics["recall"],
            "base_precision": base_metrics["precision"],
            "base_fpr": base_metrics["fpr"],
            "tp_budget": tp_budget,
            "n_numeric_candidates": len(num_cands),
            "n_combo_candidates": len(combo_cands),
            "n_candidates_after_dedupe": len(cands),
        })
        log(f"    candidatos={len(cands)}")

        frontier, best, selected_rules, stop_reason = search_best_vetos(
            cands=cands,
            base_pred=base_pred,
            y=y,
            tp_budget=tp_budget,
            max_candidates=args.max_candidates,
            beam_width=args.beam_width,
            max_rules=args.max_rules,
            max_seconds=args.max_seconds_per_base,
            output_dir=base_out,
            base_slug=base_slug,
        )
        frontier.to_csv(base_out / "02_frontier.csv", index=False)
        rules_df = candidates_df(selected_rules)
        rules_df.to_csv(base_out / "03_selected_rules.csv", index=False)

        final_pred = base_pred.copy()
        final_pred[best.mask] = 0
        final_metrics = compute_metrics(y, final_pred)

        fp_removed = base_metrics["fp"] - final_metrics["fp"]
        tp_loss = base_metrics["tp"] - final_metrics["tp"]

        result = {
            "base_slug": base_slug,
            "base_key": base_key,
            "stop_reason": stop_reason,
            "base_recipe": base_recipes.get(base_key),
            "base_metrics": base_metrics,
            "final_metrics": final_metrics,
            "tp_budget": tp_budget,
            "tp_loss_vs_base": int(tp_loss),
            "fp_removed_vs_base": int(fp_removed),
            "n_candidates": len(cands),
            "n_selected_rules": len(selected_rules),
            "selected_rules": rules_df.to_dict(orient="records") if not rules_df.empty else [],
        }
        dump_json(result, base_out / "04_base_result.json")

        all_frontiers.append(frontier)
        global_results.append(result)

        valid = final_metrics["recall"] >= args.target_recall
        wl, _ = wilson_ci(final_metrics["tp"], total_frauds)
        rank = (0 if valid else 1, final_metrics["fp"], -final_metrics["recall"], -final_metrics["precision"], 0 if wl >= args.target_recall else 1)
        if best_global is None or rank < best_global["rank"]:
            best_global = {
                "rank": rank,
                "base_slug": base_slug,
                "base_key": base_key,
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
                "base_recipe": base_recipes.get(base_key),
            }

    if best_global is None:
        raise RuntimeError("Nenhum resultado global foi produzido.")

    pd.DataFrame(global_results).to_json(output_dir / "05_base_results.json", orient="records", force_ascii=False, indent=2)
    pd.DataFrame(all_candidate_summaries).to_csv(output_dir / "06_candidate_summary.csv", index=False)
    if all_frontiers:
        pd.concat(all_frontiers, ignore_index=True).to_csv(output_dir / "07_frontier_all.csv", index=False)

    log("[3/7] Consolidando melhor política...")
    df["exp014b_r3e_selected_base_pred"] = best_global["base_pred"].astype(int)
    df["exp014b_r3e_final_pred"] = best_global["final_pred"].astype(int)

    selected_rules_df = best_global["selected_rules_df"]
    selected_rules_df.to_csv(output_dir / "08_selected_rules.csv", index=False)

    log("[4/7] Métricas finais...")
    policy_rows = []
    for c in ["exp014a_frozen_pred", "exp013k_residual_fp_pred", "exp014b_r3d_final_pred"]:
        if c in df.columns:
            policy_rows.append({"policy_name": c, **compute_metrics(y, df[c].to_numpy(dtype=int))})
    policy_rows.append({"policy_name": "EXP014B_R3E_SELECTED_BASE", **best_global["base_metrics"]})
    policy_rows.append({"policy_name": "EXP014B_R3E_FINAL", **best_global["final_metrics"]})
    metrics_df = pd.DataFrame(policy_rows)
    metrics_df["fpr_gap_vs_small"] = metrics_df["fpr"] - SMALL_BENCHMARK["fpr"]
    metrics_df["precision_gap_vs_small"] = metrics_df["precision"] - SMALL_BENCHMARK["precision"]
    metrics_df["recall_gap_vs_small"] = metrics_df["recall"] - SMALL_BENCHMARK["recall"]
    metrics_df.to_csv(output_dir / "09_policy_metrics.csv", index=False)

    block_df = pd.concat([
        block_metrics(df, df["exp014b_r3e_selected_base_pred"].to_numpy(dtype=int), blocks, "EXP014B_R3E_SELECTED_BASE"),
        block_metrics(df, df["exp014b_r3e_final_pred"].to_numpy(dtype=int), blocks, "EXP014B_R3E_FINAL"),
    ], ignore_index=True)
    block_df.to_csv(output_dir / "10_time_block_metrics.csv", index=False)

    log("[5/7] Wilson, bootstrap, amostras de erro...")
    final_metrics = best_global["final_metrics"]
    wl, wh = wilson_ci(final_metrics["tp"], total_frauds)
    wilson_df = pd.DataFrame([{
        "metric": "recall",
        "successes_tp": final_metrics["tp"],
        "n_frauds": total_frauds,
        "point_estimate": final_metrics["recall"],
        "wilson_low": wl,
        "wilson_high": wh,
        "target_recall": args.target_recall,
        "min_tp_required": min_tp_required,
        "tp_buffer_vs_target": final_metrics["tp"] - min_tp_required,
        "wilson_low_ge_target": bool(wl >= args.target_recall),
    }])
    wilson_df.to_csv(output_dir / "11_wilson_recall_ci.csv", index=False)

    boot_df = bootstrap_summary(df, "exp014b_r3e_final_pred", args.bootstrap_iters, args.seed, args.target_recall)
    boot_df.to_csv(output_dir / "12_bootstrap_summary.csv", index=False)

    df[(df["is_fraud"] == 1) & (df["exp014b_r3e_final_pred"] == 0)].to_csv(output_dir / "13_false_negatives.csv", index=False)
    fp = df[(df["is_fraud"] == 0) & (df["exp014b_r3e_final_pred"] == 1)].copy()
    if len(fp) > 5000:
        fp = fp.sample(5000, random_state=args.seed)
    fp.to_csv(output_dir / "14_false_positives_sample.csv", index=False)

    if not args.no_write_predictions:
        df.to_csv(output_dir / "16_predictions.csv", index=False)

    objective_status = "DONE"
    objective_status += "_TARGET_RECALL_MET" if final_metrics["recall"] >= args.target_recall else "_TARGET_RECALL_NOT_MET"
    objective_status += "_FP_REDUCED" if best_global["fp_removed_vs_base"] > 0 else "_FP_NOT_REDUCED"
    objective_status += "_TP_BUDGET_USED" if best_global["tp_loss_vs_base"] > 0 else "_TPLOSS0"
    objective_status += "_WILSON_PASS" if wl >= args.target_recall else "_WILSON_NOT_PASS"
    objective_status += "_FPR_NEAR_SMALL_BENCHMARK" if final_metrics["fpr"] <= SMALL_BENCHMARK["fpr"] * 1.25 else "_FPR_ABOVE_SMALL_BENCHMARK"

    artifact = {
        "experiment": "EXP-014B-R3E",
        "policy_name": "small_strategy_replay_expanded",
        "objective_status": objective_status,
        "small_benchmark": SMALL_BENCHMARK,
        "target_recall": args.target_recall,
        "min_tp_required": min_tp_required,
        "best_base_key": best_global["base_key"],
        "best_base_recipe": best_global["base_recipe"],
        "base_metrics": best_global["base_metrics"],
        "final_metrics": final_metrics,
        "tp_budget": best_global["tp_budget"],
        "tp_loss_vs_base": best_global["tp_loss_vs_base"],
        "fp_removed_vs_base": best_global["fp_removed_vs_base"],
        "wilson": wilson_df.to_dict(orient="records")[0],
        "selected_rules": selected_rules_df.to_dict(orient="records") if not selected_rules_df.empty else [],
        "notes": [
            "No runtime call.",
            "Replays the small-dataset strategy: segmented LGBM base + statistical vetos + microsegment vetos + module preservation.",
            "Explores buffered bases so TP budget can be used only if final recall remains >=95%.",
            "Requires frozen validation before promotion."
        ],
    }
    dump_json(artifact, output_dir / "15_policy_artifact.json")

    summary = {
        "experiment": "EXP-014B-R3E",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "n_rows": int(len(df)),
        "n_frauds": total_frauds,
        "target_recall": args.target_recall,
        "min_tp_required": min_tp_required,
        "small_benchmark": SMALL_BENCHMARK,
        "base_targets": base_targets,
        "best_base_slug": best_global["base_slug"],
        "best_base_key": best_global["base_key"],
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

    report = make_report(summary, base_df, metrics_df, selected_rules_df)
    (output_dir / "17_exp014b_r3e_report.md").write_text(report, encoding="utf-8")

    log("[7/7] Concluído.")
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "02_base_candidates.csv",
        output_dir / "04_selected_base_pool.csv",
        output_dir / "06_candidate_summary.csv",
        output_dir / "07_frontier_all.csv",
        output_dir / "08_selected_rules.csv",
        output_dir / "09_policy_metrics.csv",
        output_dir / "10_time_block_metrics.csv",
        output_dir / "11_wilson_recall_ci.csv",
        output_dir / "12_bootstrap_summary.csv",
        output_dir / "15_policy_artifact.json",
        output_dir / "17_exp014b_r3e_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
