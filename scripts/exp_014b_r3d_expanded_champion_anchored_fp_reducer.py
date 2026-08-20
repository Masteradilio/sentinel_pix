#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3D — Expanded Champion-Anchored FP Reducer

Objetivo:
  Fazer o melhor avanço possível no dataset expandido com 1.465 fraudes,
  mantendo os marcos já conquistados:

    - recall final >= 95%;
    - comparar contra o benchmark operacional EXP-013K/013L:
        recall=95,16%, precision=37,22%, FPR=2,02%;
    - buscar o menor FP possível no expandido;
    - evitar políticas largas como EXP-014B-R1/R2 quando houver base melhor.

Contexto:
  - EXP-014B-R3A reproduziu o champion pequeno, mas não conseguiu aplicar no
    expandido porque faltava pred_STRICT_RECALL95_SAFE_ONLY.
  - EXP-014B-R3B mostrou que reconstruir a base por surrogate não era seguro.
  - EXP-014B-R3C mostrou que EXP-013J não gera a base; ele consome
    exp013h_frozen_pred / exp013g_micro_pred / pred_HIGH_RECALL_95.
  - R3D, portanto, usa uma estratégia pragmática e auditável:
      1. procurar qualquer base high-recall já existente no expandido;
      2. se não existir, calibrar uma base minimal-FP com recall >=95%;
      3. aplicar mineração residual de FP no expandido, preservando recall>=95%;
      4. comparar métricas finais com os marcos do projeto.

Este experimento NÃO chama runtime e NÃO depende de EXP-013J rodar.

Ele é uma rodada de fronteira prática:
  "Dado o dataset expandido scoreado atual, qual o menor FP que conseguimos
   obter mantendo recall >=95%?"

Uso:
  python scripts/exp_014b_r3d_expanded_champion_anchored_fp_reducer.py

Mais rápido:
  python scripts/exp_014b_r3d_expanded_champion_anchored_fp_reducer.py --bootstrap-iters 100 --max-seconds 300

Mais profundo:
  python scripts/exp_014b_r3d_expanded_champion_anchored_fp_reducer.py --max-combo-size 4 --max-candidates 800 --beam-width 250 --max-rules 10 --max-seconds 900

Saídas:
  resultados/experimentos/EXP-014B-R3D/
    00_run_summary.json
    01_input_contract.json
    02_base_candidates.csv
    03_selected_base_metrics.csv
    04_residual_veto_candidates.csv
    05_frontier.csv
    06_selected_rules.csv
    07_policy_metrics.csv
    08_time_block_metrics.csv
    09_wilson_recall_ci.csv
    10_bootstrap_summary.csv
    11_false_negatives.csv
    12_false_positives_sample.csv
    13_policy_artifact.json
    14_predictions.csv
    15_exp014b_r3d_report.md
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3D"

BENCHMARK_SMALL = {
    "source": "EXP-013K/EXP-013L pequeno",
    "tp": 118,
    "fp": 199,
    "fn": 6,
    "recall": 0.9516,
    "precision": 0.3722,
    "fpr": 0.0202,
}

BASE_PRED_COLS = [
    "pred_STRICT_RECALL95_SAFE_ONLY",
    "exp013k_base_pred",
    "exp013h_frozen_pred",
    "exp013g_micro_pred",
    "pred_HIGH_RECALL_95",
    "exp014b_r3a_base_pred",
    "exp014b_r1_base_high_recall_pred",
    "exp014b_base_high_recall_pred",
    "exp014a_frozen_pred",
]

SCORE_COLS = [
    "lgbm_r4_score",
    "r4_score",
    "lgbm_mapped",
    "lgbm_raw",
    "score_final",
    "if_percentile",
    "if_percentile_x",
    "if_percentile_y",
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


@dataclass
class VetoCandidate:
    rule_id: str
    family: str
    description: str
    cols: list[str]
    vals: list[str]
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

    for c in BASE_PRED_COLS + ["runtime_flagged"]:
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
        "base_pred_cols_present": [c for c in BASE_PRED_COLS if c in df.columns],
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


def threshold_values(s: pd.Series, n_quantiles: int) -> list[float]:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if vals.empty:
        return []
    qs = np.linspace(0, 1, n_quantiles)
    qv = sorted(set(float(x) for x in vals.quantile(qs).to_numpy()))
    qv.extend([float(vals.min()), float(vals.max())])
    return sorted(set(qv))


def temporal_validation_mask(blocks: pd.Series, validation_blocks: int) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(blocks.dropna().unique())
    validation_blocks = min(max(1, validation_blocks), max(1, len(unique) - 1))
    val_set = set(unique[-validation_blocks:])
    val = blocks.isin(val_set).to_numpy(dtype=bool)
    return ~val, val


def evaluate_base_candidates(
    df: pd.DataFrame,
    blocks: pd.Series,
    target_recall: float,
    n_thresholds: int,
    validation_blocks: int,
    min_validation_recall: float | None,
    min_block_recall: float | None,
) -> pd.DataFrame:
    y = df["is_fraud"].to_numpy(dtype=int)
    disc_mask, val_mask = temporal_validation_mask(blocks, validation_blocks)
    rows = []

    # Existing base columns first.
    for c in [x for x in BASE_PRED_COLS if x in df.columns]:
        pred = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).to_numpy()
        m = compute_metrics(y, pred)
        mv = compute_metrics(y[val_mask], pred[val_mask])
        mb = block_metrics(df, pred, blocks, c)
        min_block = float(mb["recall"].min()) if not mb.empty else np.nan
        rows.append({
            "candidate_type": "existing_column",
            "name": c,
            "score_col": None,
            "direction": None,
            "threshold": None,
            **{f"full_{k}": v for k, v in m.items()},
            **{f"val_{k}": v for k, v in mv.items()},
            "min_block_recall": min_block,
            "passes_target": m["recall"] >= target_recall,
            "passes_validation_floor": True if min_validation_recall is None else mv["recall"] >= min_validation_recall,
            "passes_block_floor": True if min_block_recall is None else min_block >= min_block_recall,
        })

    # Thresholds: select minimal FP for recall >= target.
    for c in [x for x in SCORE_COLS if x in df.columns]:
        scores = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        for direction in ["ge", "le"]:
            for th in threshold_values(scores, n_thresholds):
                pred = (scores >= th).astype(int).to_numpy() if direction == "ge" else (scores <= th).astype(int).to_numpy()
                m = compute_metrics(y, pred)
                if m["recall"] < target_recall:
                    # keep near misses? no, not useful as base for final recall.
                    continue
                mv = compute_metrics(y[val_mask], pred[val_mask])
                mb = block_metrics(df, pred, blocks, f"{c}_{direction}_{th}")
                min_block = float(mb["recall"].min()) if not mb.empty else np.nan
                rows.append({
                    "candidate_type": "threshold",
                    "name": f"{c}_{direction}_{th:.12g}",
                    "score_col": c,
                    "direction": direction,
                    "threshold": float(th),
                    **{f"full_{k}": v for k, v in m.items()},
                    **{f"val_{k}": v for k, v in mv.items()},
                    "min_block_recall": min_block,
                    "passes_target": True,
                    "passes_validation_floor": True if min_validation_recall is None else mv["recall"] >= min_validation_recall,
                    "passes_block_floor": True if min_block_recall is None else min_block >= min_block_recall,
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["fpr_gap_vs_small_benchmark"] = out["full_fpr"] - BENCHMARK_SMALL["fpr"]
    out["precision_gap_vs_small_benchmark"] = out["full_precision"] - BENCHMARK_SMALL["precision"]
    out["passes_all_floors"] = out["passes_target"] & out["passes_validation_floor"] & out["passes_block_floor"]

    # Champion-anchored ranking:
    # 1. must pass global recall.
    # 2. prefer floors if configured.
    # 3. lowest FP/FPR.
    # 4. higher precision.
    # 5. existing champion-style columns over generated thresholds.
    out["type_rank"] = out["candidate_type"].map({"existing_column": 0, "threshold": 1}).fillna(9)
    out = out.sort_values(
        ["passes_all_floors", "full_fp", "full_fpr", "full_precision", "type_rank"],
        ascending=[False, True, True, False, True],
    ).reset_index(drop=True)
    return out


def apply_base(df: pd.DataFrame, row: pd.Series) -> np.ndarray:
    if row["candidate_type"] == "existing_column":
        return pd.to_numeric(df[str(row["name"])], errors="coerce").fillna(0).astype(int).to_numpy()

    s = pd.to_numeric(df[str(row["score_col"])], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    th = float(row["threshold"])
    if row["direction"] == "ge":
        return (s >= th).astype(int).to_numpy()
    return (s <= th).astype(int).to_numpy()


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
        max_loss = max(max_loss, int(((y == 1) & bm).sum()))
    return max_loss


def mine_veto_candidates(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    blocks: pd.Series,
    tp_budget: int,
    min_fp_removed: int,
    max_combo_size: int,
    top_groups_per_combo: int,
    max_candidate_tp_loss: int,
    max_block_tp_loss: int,
    min_fp_per_tp: float,
    require_module_quiet: bool,
) -> list[VetoCandidate]:
    y = df["is_fraud"].to_numpy(dtype=int)
    pred_pos = base_pred.astype(bool)
    feat = feature_frame(df)
    cols = list(feat.columns)

    if "module_quiet" not in feat.columns:
        require_module_quiet = False

    allowed_tp_loss = min(max_candidate_tp_loss, tp_budget)
    candidates: list[VetoCandidate] = []

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

    work_idx = np.where(pred_pos)[0]
    if len(work_idx) == 0:
        return []

    for combo in combos:
        subset = feat.iloc[work_idx][combo]
        if subset.empty:
            continue

        grouped = subset.groupby(combo, dropna=False).indices
        stats = []
        for key, rel_idxs in grouped.items():
            idxs = subset.iloc[list(rel_idxs)].index.to_numpy(dtype=int)
            if len(idxs) < min_fp_removed:
                continue
            mask = np.zeros(len(df), dtype=bool)
            mask[idxs] = True
            mask = mask & pred_pos

            if require_module_quiet:
                mask = mask & (df["module_quiet"].astype(str).to_numpy() == "module_quiet")

            if not mask.any():
                continue

            tp_loss = int(((y == 1) & mask).sum())
            fp_removed = int(((y == 0) & mask).sum())
            if fp_removed < min_fp_removed:
                continue
            if tp_loss > allowed_tp_loss:
                continue

            fp_per_tp = float("inf") if tp_loss == 0 else fp_removed / tp_loss
            if tp_loss > 0 and fp_per_tp < min_fp_per_tp:
                continue

            bmax = block_tp_loss_max(mask, y, blocks)
            if bmax > max_block_tp_loss:
                continue

            stats.append((fp_removed, tp_loss, key, mask, bmax, fp_per_tp))

        if not stats:
            continue

        stats.sort(key=lambda x: (x[1], -x[0]))
        for fp_removed, tp_loss, key, mask, bmax, fp_per_tp in stats[:top_groups_per_combo]:
            key_tuple = key if isinstance(key, tuple) else (key,)
            vals = [str(v) for v in key_tuple]
            desc = " AND ".join([f"{c}={v}" for c, v in zip(combo, vals)])
            candidates.append(VetoCandidate(
                rule_id=f"r3d_veto_{len(candidates):05d}",
                family="expanded_residual_veto",
                description=desc,
                cols=combo,
                vals=vals,
                mask=mask,
                tp_loss=tp_loss,
                fp_removed=fp_removed,
                n_removed=int(mask.sum()),
                block_tp_loss_max=bmax,
                fp_per_tp=fp_per_tp,
                params={"combo_cols": combo, "combo_values": vals, "require_module_quiet": require_module_quiet},
            ))

    # Deduplicate by exact mask.
    best: dict[bytes, VetoCandidate] = {}
    for c in candidates:
        key = np.packbits(c.mask).tobytes()
        old = best.get(key)
        if old is None or (c.fp_removed, -c.tp_loss, -len(c.description)) > (old.fp_removed, -old.tp_loss, -len(old.description)):
            best[key] = c

    out = list(best.values())
    out.sort(key=lambda c: (c.tp_loss, c.block_tp_loss_max, -c.fp_removed, -c.fp_per_tp))
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
) -> tuple[pd.DataFrame, State, list[VetoCandidate], str]:
    """
    Beam search memory-safe.

    Diferenças contra a versão original:
      - poda next_states durante o depth;
      - evita alocações do tipo ((y == 1) & new_mask);
      - quando tp_budget=0 e candidatos são TP0, union também é TP0.
    """
    t0 = time.perf_counter()

    usable = [c for c in cands if c.tp_loss <= tp_budget]
    usable.sort(key=lambda c: (c.tp_loss > 0, -c.fp_removed if c.tp_loss == 0 else -c.fp_per_tp, -c.fp_removed))
    usable = usable[:max_candidates]

    fraud_idx = np.where(y == 1)[0]
    zero_loss_mode = (tp_budget == 0 and all(c.tp_loss == 0 for c in usable))

    # Limites internos para impedir explosão de memória.
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
            elapsed = time.perf_counter() - t0
            if elapsed >= max_seconds:
                stop_reason = f"max_seconds_before_depth_{depth}"
                break

            next_states: dict[bytes, State] = {}
            depth_t0 = time.perf_counter()
            expansions = 0
            prunes = 0

            for state in states:
                last = state.rule_indices[-1] if state.rule_indices else -1
                old_total_removed = state.tp_loss + state.fp_removed

                for i in range(last + 1, len(usable)):
                    c = usable[i]
                    new_mask = state.mask | c.mask
                    new_total_removed = int(new_mask.sum())

                    # Se a união não adicionou nada, descarte.
                    if new_total_removed <= old_total_removed:
                        continue

                    if zero_loss_mode:
                        tp_loss = 0
                        fp_removed = new_total_removed
                    else:
                        # Computa TP loss só nos índices positivos para evitar array temporário grande.
                        tp_loss = int(new_mask[fraud_idx].sum()) if len(fraud_idx) else 0
                        if tp_loss > tp_budget:
                            continue
                        fp_removed = new_total_removed - tp_loss

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

                # Checagem de tempo também dentro do loop externo.
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
                    "depth": depth,
                    "tp_loss": s.tp_loss,
                    "fp_removed": s.fp_removed,
                    "n_rules": len(s.rule_indices),
                    **m,
                    "rule_ids": "|".join(usable[i].rule_id for i in s.rule_indices),
                    "rule_descriptions": " || ".join(usable[i].description for i in s.rule_indices),
                })

            pd.DataFrame(rows).to_csv(output_dir / f"checkpoint_frontier_depth_{depth:02d}.csv", index=False)

            log(
                f"  depth={depth}: best_fp_removed={best.fp_removed}, "
                f"tp_loss={best.tp_loss}/{tp_budget}, states={len(states)}, "
                f"expansions={expansions}, prunes={prunes}, "
                f"depth_s={time.perf_counter()-depth_t0:.1f}"
            )

            if time.perf_counter() - t0 >= max_seconds:
                stop_reason = f"max_seconds_after_depth_{depth}"
                break

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt_saved_best"
        log("KeyboardInterrupt capturado; salvando melhor estado.")

    if not rows:
        m = compute_metrics(y, base_pred)
        rows = [{
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


def make_report(summary: dict[str, Any], base_df: pd.DataFrame, metrics_df: pd.DataFrame, rules_df: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014B-R3D — Expanded Champion-Anchored FP Reducer")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Base selecionada: `{summary['selected_base_name']}`")
    lines.append(f"- FP final: `{summary['final_metrics']['fp']}`")
    lines.append(f"- Recall final: `{summary['final_metrics']['recall']}`")
    lines.append(f"- Precision final: `{summary['final_metrics']['precision']}`")
    lines.append(f"- FPR final: `{summary['final_metrics']['fpr']}`")
    lines.append("")
    lines.append("## Marco de comparação EXP-013K/013L pequeno")
    lines.append(f"- Recall: `{BENCHMARK_SMALL['recall']}`")
    lines.append(f"- Precision: `{BENCHMARK_SMALL['precision']}`")
    lines.append(f"- FPR: `{BENCHMARK_SMALL['fpr']}`")
    lines.append("")
    lines.append("## Métricas")
    lines.append(metrics_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Top bases candidatas")
    show_cols = ["candidate_type", "name", "full_tp", "full_fp", "full_fn", "full_precision", "full_recall", "full_fpr", "val_recall", "min_block_recall"]
    show_cols = [c for c in show_cols if c in base_df.columns]
    lines.append(base_df[show_cols].head(20).to_markdown(index=False))
    lines.append("")
    lines.append("## Regras selecionadas")
    if rules_df.empty:
        lines.append("Nenhuma regra selecionada.")
    else:
        show_rules = ["rule_id", "description", "tp_loss", "fp_removed", "block_tp_loss_max", "fp_per_tp"]
        lines.append(rules_df[[c for c in show_rules if c in rules_df.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Decisão sugerida")
    if summary["final_metrics"]["recall"] >= summary["target_recall"]:
        if summary["final_metrics"]["fpr"] <= BENCHMARK_SMALL["fpr"] * 1.25:
            lines.append("A política aproximou o marco de FPR do benchmark pequeno no dataset expandido. Próximo passo: validação congelada e governança.")
        else:
            lines.append("A política mantém recall >=95%, mas o FPR ainda está acima do marco operacional pequeno. Usar como fronteira expandida atual e decidir se há mais uma rodada focada nos FPs residuais ou se a limitação vem do score/modelo.")
    else:
        lines.append("A política não mantém recall >=95%; não promover.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--thresholds", type=int, default=1200)
    parser.add_argument("--time-blocks", type=int, default=10)
    parser.add_argument("--validation-blocks", type=int, default=3)
    parser.add_argument("--min-validation-recall", type=float, default=None, help="Opcional. Ex: 0.90 ou 0.95. Default não filtra por validação temporal.")
    parser.add_argument("--min-block-recall", type=float, default=None, help="Opcional. Ex: 0.80. Default não filtra por bloco.")
    parser.add_argument("--min-fp-removed", type=int, default=30)
    parser.add_argument("--max-combo-size", type=int, default=3)
    parser.add_argument("--top-groups-per-combo", type=int, default=50)
    parser.add_argument("--max-candidate-tp-loss", type=int, default=6)
    parser.add_argument("--max-block-tp-loss", type=int, default=2)
    parser.add_argument("--min-fp-per-tp", type=float, default=80.0)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--beam-width", type=int, default=180)
    parser.add_argument("--max-rules", type=int, default=8)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    parser.add_argument("--require-module-quiet", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-write-predictions", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014B-R3D — Expanded Champion-Anchored FP Reducer")
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

    log("[1/6] Avaliando bases candidatas...")
    base_candidates = evaluate_base_candidates(
        df=df,
        blocks=blocks,
        target_recall=args.target_recall,
        n_thresholds=args.thresholds,
        validation_blocks=args.validation_blocks,
        min_validation_recall=args.min_validation_recall,
        min_block_recall=args.min_block_recall,
    )
    base_candidates.to_csv(output_dir / "02_base_candidates.csv", index=False)
    if base_candidates.empty:
        raise RuntimeError("Nenhuma base candidata com recall >= target foi encontrada.")

    selected_base = base_candidates.iloc[0]
    base_pred = apply_base(df, selected_base)
    df["exp014b_r3d_selected_base_pred"] = base_pred
    base_metrics = compute_metrics(y, base_pred)
    tp_budget = max(0, base_metrics["tp"] - min_tp_required)

    pd.DataFrame([{"policy_name": "EXP014B_R3D_SELECTED_BASE", **base_metrics}]).to_csv(output_dir / "03_selected_base_metrics.csv", index=False)

    log(f"Base selecionada: {selected_base['name']} {base_metrics}, tp_budget={tp_budget}")

    log("[2/6] Minerando vetos residuais no expandido...")
    candidates = mine_veto_candidates(
        df=df,
        base_pred=base_pred,
        blocks=blocks,
        tp_budget=tp_budget,
        min_fp_removed=args.min_fp_removed,
        max_combo_size=args.max_combo_size,
        top_groups_per_combo=args.top_groups_per_combo,
        max_candidate_tp_loss=args.max_candidate_tp_loss,
        max_block_tp_loss=args.max_block_tp_loss,
        min_fp_per_tp=args.min_fp_per_tp,
        require_module_quiet=args.require_module_quiet,
    )
    cdf = candidates_df(candidates)
    cdf.to_csv(output_dir / "04_residual_veto_candidates.csv", index=False)
    log(f"Candidatos gerados: {len(candidates)}")

    log("[3/6] Buscando melhor combinação de vetos...")
    frontier, best, selected_rules, stop_reason = search_best_vetos(
        cands=candidates,
        base_pred=base_pred,
        y=y,
        tp_budget=tp_budget,
        max_candidates=args.max_candidates,
        beam_width=args.beam_width,
        max_rules=args.max_rules,
        max_seconds=args.max_seconds,
        output_dir=output_dir,
    )
    frontier.to_csv(output_dir / "05_frontier.csv", index=False)

    rules_df = candidates_df(selected_rules)
    rules_df.to_csv(output_dir / "06_selected_rules.csv", index=False)

    final_pred = base_pred.copy()
    final_pred[best.mask] = 0
    df["exp014b_r3d_final_pred"] = final_pred
    final_metrics = compute_metrics(y, final_pred)

    fp_removed_vs_base = base_metrics["fp"] - final_metrics["fp"]
    tp_loss_vs_base = base_metrics["tp"] - final_metrics["tp"]

    log("[4/6] Métricas e comparação...")
    policy_rows = []
    # Compare with known runtime / broad policies if present.
    for c in ["exp014a_frozen_pred", "exp013k_residual_fp_pred", "exp014b_r1_safe_beam_pred", "exp014b_r2_irreducible_fp_pred"]:
        if c in df.columns:
            policy_rows.append({"policy_name": c, **compute_metrics(y, df[c].to_numpy(dtype=int))})
    policy_rows.append({"policy_name": "EXP014B_R3D_SELECTED_BASE", **base_metrics})
    policy_rows.append({"policy_name": "EXP014B_R3D_FINAL", **final_metrics})

    metrics_df = pd.DataFrame(policy_rows)
    metrics_df["fpr_gap_vs_exp013k_l"] = metrics_df["fpr"] - BENCHMARK_SMALL["fpr"]
    metrics_df["precision_gap_vs_exp013k_l"] = metrics_df["precision"] - BENCHMARK_SMALL["precision"]
    metrics_df["recall_gap_vs_exp013k_l"] = metrics_df["recall"] - BENCHMARK_SMALL["recall"]
    metrics_df.to_csv(output_dir / "07_policy_metrics.csv", index=False)

    block_df = pd.concat([
        block_metrics(df, base_pred, blocks, "EXP014B_R3D_SELECTED_BASE"),
        block_metrics(df, final_pred, blocks, "EXP014B_R3D_FINAL"),
    ], ignore_index=True)
    block_df.to_csv(output_dir / "08_time_block_metrics.csv", index=False)

    log("[5/6] Wilson, bootstrap, FNs/FPs...")
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
    wilson_df.to_csv(output_dir / "09_wilson_recall_ci.csv", index=False)

    boot_df = bootstrap_summary(df, "exp014b_r3d_final_pred", args.bootstrap_iters, args.seed, args.target_recall)
    boot_df.to_csv(output_dir / "10_bootstrap_summary.csv", index=False)

    df[(df["is_fraud"] == 1) & (df["exp014b_r3d_final_pred"] == 0)].to_csv(output_dir / "11_false_negatives.csv", index=False)
    fp = df[(df["is_fraud"] == 0) & (df["exp014b_r3d_final_pred"] == 1)].copy()
    if len(fp) > 5000:
        fp = fp.sample(5000, random_state=args.seed)
    fp.to_csv(output_dir / "12_false_positives_sample.csv", index=False)

    if not args.no_write_predictions:
        df.to_csv(output_dir / "14_predictions.csv", index=False)

    objective_status = "DONE"
    objective_status += "_TARGET_RECALL_MET" if final_metrics["recall"] >= args.target_recall else "_TARGET_RECALL_NOT_MET"
    objective_status += "_FP_REDUCED" if fp_removed_vs_base > 0 else "_FP_NOT_REDUCED"
    objective_status += "_TP_BUDGET_USED" if tp_loss_vs_base > 0 else "_TPLOSS0"
    objective_status += "_WILSON_PASS" if wl >= args.target_recall else "_WILSON_NOT_PASS"
    objective_status += "_FPR_NEAR_SMALL_BENCHMARK" if final_metrics["fpr"] <= BENCHMARK_SMALL["fpr"] * 1.25 else "_FPR_ABOVE_SMALL_BENCHMARK"

    artifact = {
        "experiment": "EXP-014B-R3D",
        "policy_name": "expanded_champion_anchored_fp_reducer",
        "objective_status": objective_status,
        "benchmark_small": BENCHMARK_SMALL,
        "selected_base": selected_base.to_dict(),
        "base_metrics": base_metrics,
        "final_metrics": final_metrics,
        "target_recall": args.target_recall,
        "min_tp_required": min_tp_required,
        "tp_budget": tp_budget,
        "tp_loss_vs_base": int(tp_loss_vs_base),
        "fp_removed_vs_base": int(fp_removed_vs_base),
        "wilson": wilson_df.to_dict(orient="records")[0],
        "selected_rules": rules_df.to_dict(orient="records") if not rules_df.empty else [],
        "notes": [
            "No runtime call.",
            "Uses expanded scored dataset with 1465 frauds.",
            "First tries existing champion-style base columns; if missing, selects minimal-FP threshold base with recall>=95%.",
            "Residual veto search preserves final recall>=95% and compares against EXP-013K/013L small benchmark."
        ],
    }
    dump_json(artifact, output_dir / "13_policy_artifact.json")

    summary = {
        "experiment": "EXP-014B-R3D",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "n_rows": int(len(df)),
        "n_frauds": total_frauds,
        "target_recall": args.target_recall,
        "min_tp_required": min_tp_required,
        "benchmark_small": BENCHMARK_SMALL,
        "selected_base_name": str(selected_base["name"]),
        "selected_base_type": str(selected_base["candidate_type"]),
        "base_metrics": base_metrics,
        "final_metrics": final_metrics,
        "tp_budget": tp_budget,
        "tp_loss_vs_base": int(tp_loss_vs_base),
        "fp_removed_vs_base": int(fp_removed_vs_base),
        "n_candidates": int(len(candidates)),
        "n_selected_rules": int(len(selected_rules)),
        "stop_reason": stop_reason,
        "wilson_recall_low": wl,
        "wilson_recall_high": wh,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, base_candidates, metrics_df, rules_df)
    (output_dir / "15_exp014b_r3d_report.md").write_text(report, encoding="utf-8")

    log("[6/6] Concluído.")
    log("")
    log("=" * 80)
    log("EXP-014B-R3D CONCLUÍDO")
    log("=" * 80)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "02_base_candidates.csv",
        output_dir / "03_selected_base_metrics.csv",
        output_dir / "04_residual_veto_candidates.csv",
        output_dir / "05_frontier.csv",
        output_dir / "06_selected_rules.csv",
        output_dir / "07_policy_metrics.csv",
        output_dir / "08_time_block_metrics.csv",
        output_dir / "09_wilson_recall_ci.csv",
        output_dir / "10_bootstrap_summary.csv",
        output_dir / "13_policy_artifact.json",
        output_dir / "15_exp014b_r3d_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
