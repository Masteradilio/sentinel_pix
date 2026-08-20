#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B — Expanded High-Recall Threshold + Safe Veto Mining

Contexto:
  EXP-014A-4 criou o dataset expandido scoreado:
      dados/exp014a_expanded_scored_input.csv
  com 113.844 linhas e 1.465 fraudes.

  EXP-014A-5 mostrou que não devemos transportar a política pequena:
    - o surrogate da base high-recall ficou com recall ~84,98%;
    - as 10 regras EXP-013K perderam 205 TPs no expandido;
    - recall final ficou ~70,99%.

Objetivo do EXP-014B:
  Criar uma nova política high-recall diretamente no dataset expandido:
    1. Calibrar uma base high-recall por threshold em lgbm_r4_score/score_final.
    2. Exigir recall >= 95% no expandido.
    3. Minerar novos microvetos seguros no expandido.
    4. Aceitar apenas vetos com TP_loss=0 global e por bloco temporal.
    5. Rodar rápido, sem chamar runtime.

Importante:
  - Este experimento USA labels do dataset expandido para calibração/diagnóstico.
  - Se encontrar uma boa política, o próximo passo será EXP-014C Frozen Validation,
    sem nova mineração.

Uso:
  python scripts/exp_014b_expanded_high_recall_threshold_safe_veto.py

Mais conservador:
  python scripts/exp_014b_expanded_high_recall_threshold_safe_veto.py --max-combo-size 3 --max-rules 6

Mais profundo:
  python scripts/exp_014b_expanded_high_recall_threshold_safe_veto.py --max-combo-size 4 --max-candidates 500 --beam-width 250 --max-rules 10

Saídas:
  resultados/experimentos/EXP-014B/
    00_run_summary.json
    01_input_contract.json
    02_threshold_sweep.csv
    03_base_policy_metrics.csv
    04_veto_candidates.csv
    05_frontier.csv
    06_selected_rules.csv
    07_policy_metrics.csv
    08_time_block_metrics.csv
    09_wilson_recall_ci.csv
    10_bootstrap_summary.csv
    11_false_negatives.csv
    12_false_positives_sample.csv
    13_policy_artifact.json
    14_exp014b_report.md
    15_predictions.csv
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B"

SCORE_COL_CANDIDATES = [
    "lgbm_r4_score",
    "r4_score",
    "lgbm_mapped",
    "lgbm_raw",
    "score_final",
]

VETO_FEATURES_BASE = [
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
]


@dataclass
class Candidate:
    rule_id: str
    description: str
    cols: list[str]
    vals: list[str]
    mask: np.ndarray
    tp_loss: int
    fp_removed: int
    n_removed: int
    block_tp_loss_max: int


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

    for c in ["event_datetime", "data_pix"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if "transaction_id" in df.columns:
        df["transaction_id"] = df["transaction_id"].astype("string").str.strip()

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
            labels.append(f"{name}_{left:g:g}_{right:g}")
    return pd.cut(vals, bins=edges, labels=labels, include_lowest=True).astype("string").fillna(f"{name}_MISSING").astype(str)


def ensure_bins_and_guards(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "lgbm_bin" not in df.columns and pick_col(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"]):
        df["lgbm_bin"] = qbin_series(num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0), "lgbm", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1])

    if "if_bin" not in df.columns and pick_col(df, ["if_percentile", "if_percentile_x", "if_percentile_y"]):
        df["if_bin"] = qbin_series(num(df, ["if_percentile", "if_percentile_x", "if_percentile_y"], 0.0), "if", [0.32, 0.5, 0.7, 0.85, 0.95])

    if "score_bin" not in df.columns and "score_final" in df.columns:
        df["score_bin"] = qbin_series(num(df, "score_final", 0.0), "score", [0.5, 1, 2, 3, 5, 10])

    if "ratio_bin" not in df.columns and "ratio_valor_media_pagador_90d" in df.columns:
        df["ratio_bin"] = qbin_series(num(df, "ratio_valor_media_pagador_90d", 0.0), "ratio", [0.05, 0.1, 0.2, 0.5, 1, 2, 5])

    if "vl_bin" not in df.columns and "vl_pix" in df.columns:
        df["vl_bin"] = qbin_series(num(df, "vl_pix", 0.0), "vl", [20, 50, 100, 250, 500, 1000, 5000, 10000])

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


def contract_report(df: pd.DataFrame) -> dict[str, Any]:
    missing = []
    if "is_fraud" not in df.columns:
        missing.append("is_fraud")
    if not any(c in df.columns for c in ["event_datetime", "data_pix", "dt_pix"]):
        missing.append("event_datetime_or_data_pix")
    if not any(c in df.columns for c in SCORE_COL_CANDIDATES):
        missing.append("risk_score_column")

    for logical, alternatives in {
        "lgbm_bin": [["lgbm_bin"], ["lgbm_r4_score"], ["r4_score"], ["lgbm_mapped"], ["lgbm_raw"]],
        "if_bin": [["if_bin"], ["if_percentile"], ["if_percentile_x"], ["if_percentile_y"]],
        "score_bin": [["score_bin"], ["score_final"]],
        "ratio_bin": [["ratio_bin"], ["ratio_valor_media_pagador_90d"]],
        "vl_bin": [["vl_bin"], ["vl_pix"]],
        "value_band": [["value_band"]],
        "ds_tipo_chave_norm": [["ds_tipo_chave_norm"]],
        "first_receiver_flag_real": [["first_receiver_flag_real"]],
        "mbk_available_flag": [["mbk_available_flag"]],
    }.items():
        if not any(all(c in df.columns for c in alt) for alt in alternatives):
            missing.append(f"feature_or_bin:{logical}")

    return {
        "contract_ok": len(missing) == 0,
        "missing": missing,
        "n_rows": int(len(df)),
        "n_frauds": int(df["is_fraud"].sum()) if "is_fraud" in df.columns else None,
        "score_cols_present": [c for c in SCORE_COL_CANDIDATES if c in df.columns],
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


def temporal_discovery_validation_split(blocks: pd.Series, validation_blocks: int) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(blocks.dropna().unique())
    validation_blocks = min(max(1, validation_blocks), max(1, len(unique) - 1))
    val_set = set(unique[-validation_blocks:])
    val = blocks.isin(val_set).to_numpy(dtype=bool)
    disc = ~val
    return disc, val


def threshold_values_from_scores(scores: pd.Series, n: int) -> list[float]:
    vals = pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if vals.empty:
        return []
    qs = np.linspace(0.0, 1.0, n)
    thresholds = sorted(set(float(x) for x in vals.quantile(qs).to_numpy()))
    # Add exact min/max borders.
    thresholds.extend([float(vals.min()), float(vals.max())])
    return sorted(set(thresholds))


def sweep_thresholds(df: pd.DataFrame, blocks: pd.Series, target_recall: float, n_thresholds: int, validation_blocks: int) -> pd.DataFrame:
    y = df["is_fraud"].to_numpy(dtype=int)
    disc_mask, val_mask = temporal_discovery_validation_split(blocks, validation_blocks)
    rows = []

    for col in [c for c in SCORE_COL_CANDIDATES if c in df.columns]:
        scores = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        thresholds = threshold_values_from_scores(scores, n_thresholds)
        if not thresholds:
            continue

        for direction in ["ge", "le"]:
            for th in thresholds:
                pred = (scores >= th).to_numpy(dtype=bool) if direction == "ge" else (scores <= th).to_numpy(dtype=bool)
                pred_i = pred.astype(int)

                m_full = compute_metrics(y, pred_i)
                m_disc = compute_metrics(y[disc_mask], pred_i[disc_mask])
                m_val = compute_metrics(y[val_mask], pred_i[val_mask])

                rows.append({
                    "score_col": col,
                    "direction": direction,
                    "threshold": th,
                    "full_tp": m_full["tp"],
                    "full_fp": m_full["fp"],
                    "full_fn": m_full["fn"],
                    "full_precision": m_full["precision"],
                    "full_recall": m_full["recall"],
                    "full_f1": m_full["f1"],
                    "full_fpr": m_full["fpr"],
                    "disc_tp": m_disc["tp"],
                    "disc_fp": m_disc["fp"],
                    "disc_fn": m_disc["fn"],
                    "disc_precision": m_disc["precision"],
                    "disc_recall": m_disc["recall"],
                    "disc_fpr": m_disc["fpr"],
                    "val_tp": m_val["tp"],
                    "val_fp": m_val["fp"],
                    "val_fn": m_val["fn"],
                    "val_precision": m_val["precision"],
                    "val_recall": m_val["recall"],
                    "val_fpr": m_val["fpr"],
                    "target_recall_met_full": m_full["recall"] >= target_recall,
                    "target_recall_met_disc": m_disc["recall"] >= target_recall,
                    "target_recall_met_val": m_val["recall"] >= target_recall,
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Ranking:
    # 1. Prefer candidates with full and validation recall >= target.
    # 2. Lowest full FP.
    # 3. Higher precision.
    out["passes_strict"] = out["target_recall_met_full"] & out["target_recall_met_val"]
    out["passes_full_only"] = out["target_recall_met_full"]
    out = out.sort_values(
        ["passes_strict", "passes_full_only", "full_fp", "full_precision", "val_recall"],
        ascending=[False, False, True, False, False],
    ).reset_index(drop=True)
    return out


def select_base_policy(sweep: pd.DataFrame, target_recall: float) -> dict[str, Any]:
    if sweep.empty:
        raise RuntimeError("Threshold sweep vazio.")

    strict = sweep[(sweep["full_recall"] >= target_recall) & (sweep["val_recall"] >= target_recall)].copy()
    if not strict.empty:
        row = strict.sort_values(["full_fp", "val_fp", "full_precision"], ascending=[True, True, False]).iloc[0]
        status = "STRICT_FULL_AND_VALIDATION_RECALL_MET"
    else:
        full = sweep[sweep["full_recall"] >= target_recall].copy()
        if not full.empty:
            row = full.sort_values(["full_fp", "val_recall", "full_precision"], ascending=[True, False, False]).iloc[0]
            status = "FULL_RECALL_MET_VALIDATION_WARNING"
        else:
            row = sweep.sort_values(["full_recall", "full_fp"], ascending=[False, True]).iloc[0]
            status = "TARGET_NOT_MET_BEST_AVAILABLE"

    return {
        "selection_status": status,
        "score_col": str(row["score_col"]),
        "direction": str(row["direction"]),
        "threshold": float(row["threshold"]),
        "full_metrics": {
            "tp": int(row["full_tp"]),
            "fp": int(row["full_fp"]),
            "fn": int(row["full_fn"]),
            "precision": float(row["full_precision"]),
            "recall": float(row["full_recall"]),
            "fpr": float(row["full_fpr"]),
        },
        "validation_metrics": {
            "tp": int(row["val_tp"]),
            "fp": int(row["val_fp"]),
            "fn": int(row["val_fn"]),
            "precision": float(row["val_precision"]),
            "recall": float(row["val_recall"]),
            "fpr": float(row["val_fpr"]),
        },
        "discovery_metrics": {
            "tp": int(row["disc_tp"]),
            "fp": int(row["disc_fp"]),
            "fn": int(row["disc_fn"]),
            "precision": float(row["disc_precision"]),
            "recall": float(row["disc_recall"]),
            "fpr": float(row["disc_fpr"]),
        },
    }


def apply_threshold(df: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    scores = pd.to_numeric(df[spec["score_col"]], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    if spec["direction"] == "ge":
        return (scores >= float(spec["threshold"])).astype(int).to_numpy()
    return (scores <= float(spec["threshold"])).astype(int).to_numpy()


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in VETO_FEATURES_BASE:
        if c in df.columns:
            out[c] = df[c].astype("string").fillna("<MISSING>").astype(str)
    return out


def block_tp_loss_max(mask: np.ndarray, y: np.ndarray, blocks: pd.Series) -> int:
    out = 0
    block_values = sorted(blocks.dropna().unique())
    bvals = blocks.to_numpy()
    for b in block_values:
        bm = mask & (bvals == b)
        out = max(out, int(((y == 1) & bm).sum()))
    return out


def candidate_df(cands: list[Candidate]) -> pd.DataFrame:
    return pd.DataFrame([{
        "candidate_index": i,
        "rule_id": c.rule_id,
        "description": c.description,
        "cols": "|".join(c.cols),
        "vals": "|".join(c.vals),
        "tp_loss": c.tp_loss,
        "fp_removed": c.fp_removed,
        "n_removed": c.n_removed,
        "block_tp_loss_max": c.block_tp_loss_max,
    } for i, c in enumerate(cands)])


def mine_veto_candidates(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    blocks: pd.Series,
    min_fp_removed: int,
    max_combo_size: int,
    top_groups_per_combo: int,
) -> list[Candidate]:
    y = df["is_fraud"].to_numpy(dtype=int)
    pred_pos = base_pred.astype(bool)
    feat = feature_frame(df)
    cols_available = list(feat.columns)

    combos = []
    for r in range(2, max_combo_size + 1):
        for combo in itertools.combinations(cols_available, r):
            combo = list(combo)
            # Avoid weak combos that are only bins without business segment.
            if r >= 3:
                combos.append(combo)
            else:
                if any(c in combo for c in ["value_band", "ds_tipo_chave_norm", "periodo_dia", "first_receiver_flag_real", "mbk_available_flag"]):
                    combos.append(combo)

    candidates: list[Candidate] = []

    for combo in combos:
        subset = feat.loc[pred_pos, combo]
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

            tp_loss = int(((y == 1) & mask).sum())
            fp_removed = int(((y == 0) & mask).sum())
            if tp_loss != 0 or fp_removed < min_fp_removed:
                continue

            bmax = block_tp_loss_max(mask, y, blocks)
            if bmax != 0:
                continue

            group_rows.append((fp_removed, key, mask, tp_loss, bmax))

        if not group_rows:
            continue

        group_rows.sort(key=lambda x: x[0], reverse=True)
        for fp_removed, key, mask, tp_loss, bmax in group_rows[:top_groups_per_combo]:
            key_tuple = key if isinstance(key, tuple) else (key,)
            vals = [str(v) for v in key_tuple]
            desc = " AND ".join([f"{c}={v}" for c, v in zip(combo, vals)])
            rid = f"exp014b_veto_{len(candidates):05d}"
            candidates.append(Candidate(
                rule_id=rid,
                description=desc,
                cols=combo,
                vals=vals,
                mask=mask,
                tp_loss=tp_loss,
                fp_removed=fp_removed,
                n_removed=int(mask.sum()),
                block_tp_loss_max=bmax,
            ))

    # Dedupe by mask.
    best: dict[bytes, Candidate] = {}
    for c in candidates:
        key = np.packbits(c.mask).tobytes()
        old = best.get(key)
        if old is None or (c.fp_removed, -len(c.description)) > (old.fp_removed, -len(old.description)):
            best[key] = c

    out = list(best.values())
    out.sort(key=lambda c: (c.tp_loss, c.block_tp_loss_max, -c.fp_removed))
    return out


def search_best_vetos(candidates: list[Candidate], base_pred: np.ndarray, y: np.ndarray, max_candidates: int, beam_width: int, max_rules: int):
    usable = [c for c in candidates if c.tp_loss == 0 and c.block_tp_loss_max == 0]
    usable.sort(key=lambda c: (c.fp_removed, -len(c.description)), reverse=True)
    usable = usable[:max_candidates]

    zero = np.zeros(len(y), dtype=bool)
    initial = State(mask=zero, rule_indices=tuple(), tp_loss=0, fp_removed=0)
    states = [initial]
    best = initial
    rows = []

    for depth in range(1, max_rules + 1):
        next_states: dict[bytes, State] = {}

        for state in states:
            last = state.rule_indices[-1] if state.rule_indices else -1
            for i in range(last + 1, len(usable)):
                c = usable[i]
                new_mask = state.mask | c.mask
                if np.array_equal(new_mask, state.mask):
                    continue
                tp_loss = int(((y == 1) & new_mask).sum())
                if tp_loss != 0:
                    continue
                fp_removed = int(((y == 0) & new_mask).sum())
                if fp_removed <= state.fp_removed:
                    continue

                key = np.packbits(new_mask).tobytes()
                ns = State(new_mask, state.rule_indices + (i,), tp_loss, fp_removed)
                old = next_states.get(key)
                if old is None or (ns.fp_removed, -len(ns.rule_indices)) > (old.fp_removed, -len(old.rule_indices)):
                    next_states[key] = ns

        if not next_states:
            break

        states = sorted(next_states.values(), key=lambda s: (s.fp_removed, -len(s.rule_indices)), reverse=True)[:beam_width]
        if states[0].fp_removed > best.fp_removed:
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

        log(f"  beam depth={depth}: best_fp_removed={best.fp_removed}, states={len(states)}")

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

    frontier = pd.DataFrame(rows).sort_values(["fp", "n_rules"], ascending=[True, True]).reset_index(drop=True)
    selected = [usable[i] for i in best.rule_indices]
    return frontier, best, selected


def block_metrics(df: pd.DataFrame, pred: np.ndarray, blocks: pd.Series, policy_name: str) -> pd.DataFrame:
    rows = []
    y = df["is_fraud"].to_numpy(dtype=int)
    bvals = blocks.to_numpy()
    for b in sorted(blocks.dropna().unique()):
        idx = bvals == b
        part = df.loc[idx]
        m = compute_metrics(y[idx], pred[idx])
        rows.append({
            "policy_name": policy_name,
            "block": int(b),
            "n_rows": int(len(part)),
            "n_frauds": int(part["is_fraud"].sum()),
            **m,
        })
    return pd.DataFrame(rows)


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


def make_report(summary: dict[str, Any], base_metrics: pd.DataFrame, policy_metrics: pd.DataFrame, selected_rules: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014B — Expanded High-Recall Threshold + Safe Veto Mining")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Base selection: `{summary['base_policy']['selection_status']}`")
    lines.append(f"- Score: `{summary['base_policy']['score_col']}` `{summary['base_policy']['direction']}` `{summary['base_policy']['threshold']}`")
    lines.append("")
    lines.append("## Base high-recall")
    lines.append(base_metrics.to_markdown(index=False))
    lines.append("")
    lines.append("## Política final")
    lines.append(policy_metrics.to_markdown(index=False))
    lines.append("")
    lines.append("## Regras selecionadas")
    if selected_rules.empty:
        lines.append("Nenhuma regra selecionada.")
    else:
        lines.append(selected_rules.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretação")
    lines.append("Este experimento calibra a base e minera vetos diretamente no dataset expandido. Se a política final mantiver recall >=95% e reduzir FP com TP_loss=0, o próximo passo é EXP-014C Frozen Validation sem nova mineração.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--thresholds", type=int, default=400)
    parser.add_argument("--time-blocks", type=int, default=10)
    parser.add_argument("--validation-blocks", type=int, default=3)
    parser.add_argument("--min-fp-removed", type=int, default=25)
    parser.add_argument("--max-combo-size", type=int, default=3)
    parser.add_argument("--top-groups-per-combo", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--beam-width", type=int, default=160)
    parser.add_argument("--max-rules", type=int, default=8)
    parser.add_argument("--bootstrap-iters", type=int, default=300)
    parser.add_argument("--false-positive-sample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-write-predictions", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014B — Expanded High-Recall Threshold + Safe Veto Mining")
    log("=" * 80)
    log(f"Input: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    df = ensure_bins_and_guards(df)
    contract = contract_report(df)
    dump_json(contract, output_dir / "01_input_contract.json")
    if not contract["contract_ok"]:
        raise RuntimeError(f"Contrato de input falhou: {contract['missing']}")

    y = df["is_fraud"].to_numpy(dtype=int)
    blocks = make_time_blocks(df, args.time_blocks)

    log("[1/5] Threshold sweep...")
    sweep = sweep_thresholds(df, blocks, args.target_recall, args.thresholds, args.validation_blocks)
    sweep.to_csv(output_dir / "02_threshold_sweep.csv", index=False)

    base_policy = select_base_policy(sweep, args.target_recall)
    base_pred = apply_threshold(df, base_policy)
    df["exp014b_base_high_recall_pred"] = base_pred

    base_metrics = pd.DataFrame([{"policy_name": "EXP014B_BASE_HIGH_RECALL", **compute_metrics(y, base_pred)}])
    base_metrics.to_csv(output_dir / "03_base_policy_metrics.csv", index=False)

    log(f"Base: {base_policy}")

    log("[2/5] Minerando vetos seguros...")
    candidates = mine_veto_candidates(
        df=df,
        base_pred=base_pred,
        blocks=blocks,
        min_fp_removed=args.min_fp_removed,
        max_combo_size=args.max_combo_size,
        top_groups_per_combo=args.top_groups_per_combo,
    )
    cdf = candidate_df(candidates)
    cdf.to_csv(output_dir / "04_veto_candidates.csv", index=False)
    log(f"Candidatos seguros: {len(candidates)}")

    log("[3/5] Beam search de vetos...")
    frontier, best, selected = search_best_vetos(
        candidates=candidates,
        base_pred=base_pred,
        y=y,
        max_candidates=args.max_candidates,
        beam_width=args.beam_width,
        max_rules=args.max_rules,
    )
    frontier.to_csv(output_dir / "05_frontier.csv", index=False)

    final_pred = base_pred.copy()
    final_pred[best.mask] = 0
    df["exp014b_high_recall_safe_veto_pred"] = final_pred

    selected_df = candidate_df(selected)
    selected_df.to_csv(output_dir / "06_selected_rules.csv", index=False)

    policy_rows = [
        {"policy_name": "EXP014B_BASE_HIGH_RECALL", **compute_metrics(y, base_pred)},
        {"policy_name": "EXP014B_HIGH_RECALL_SAFE_VETO", **compute_metrics(y, final_pred)},
    ]

    # Include runtime final if present for comparison.
    for runtime_col in ["exp014a_frozen_pred", "exp013k_residual_fp_pred"]:
        if runtime_col in df.columns:
            policy_rows.insert(0, {"policy_name": f"RUNTIME_FINAL_{runtime_col}", **compute_metrics(y, df[runtime_col].to_numpy(dtype=int))})
            break

    policy_metrics = pd.DataFrame(policy_rows)
    policy_metrics.to_csv(output_dir / "07_policy_metrics.csv", index=False)

    log("[4/5] Blocos, Wilson e Bootstrap...")
    block_parts = []
    block_parts.append(block_metrics(df, base_pred, blocks, "EXP014B_BASE_HIGH_RECALL"))
    block_parts.append(block_metrics(df, final_pred, blocks, "EXP014B_HIGH_RECALL_SAFE_VETO"))
    block_df = pd.concat(block_parts, ignore_index=True)
    block_df.to_csv(output_dir / "08_time_block_metrics.csv", index=False)

    final_metrics = compute_metrics(y, final_pred)
    total_frauds = int(df["is_fraud"].sum())
    min_tp_required = int(math.ceil(args.target_recall * total_frauds))
    wilson_low, wilson_high = wilson_ci(final_metrics["tp"], total_frauds)
    wilson_df = pd.DataFrame([{
        "metric": "recall",
        "successes_tp": final_metrics["tp"],
        "n_frauds": total_frauds,
        "point_estimate": final_metrics["recall"],
        "wilson_low": wilson_low,
        "wilson_high": wilson_high,
        "target_recall": args.target_recall,
        "min_tp_required": min_tp_required,
        "tp_buffer_vs_target": final_metrics["tp"] - min_tp_required,
        "wilson_low_ge_target": bool(wilson_low >= args.target_recall),
    }])
    wilson_df.to_csv(output_dir / "09_wilson_recall_ci.csv", index=False)

    boot_df = bootstrap_summary(df, "exp014b_high_recall_safe_veto_pred", args.bootstrap_iters, args.seed, args.target_recall)
    boot_df.to_csv(output_dir / "10_bootstrap_summary.csv", index=False)

    log("[5/5] FNs/FPs e artefato...")
    fn = df[(df["is_fraud"] == 1) & (df["exp014b_high_recall_safe_veto_pred"] == 0)].copy()
    fp = df[(df["is_fraud"] == 0) & (df["exp014b_high_recall_safe_veto_pred"] == 1)].copy()
    fn.to_csv(output_dir / "11_false_negatives.csv", index=False)
    if len(fp) > args.false_positive_sample:
        fp = fp.sample(args.false_positive_sample, random_state=args.seed)
    fp.to_csv(output_dir / "12_false_positives_sample.csv", index=False)

    if not args.no_write_predictions:
        df.to_csv(output_dir / "15_predictions.csv", index=False)

    fp_removed_vs_base = int(compute_metrics(y, base_pred)["fp"] - final_metrics["fp"])
    tp_loss_vs_base = int(compute_metrics(y, base_pred)["tp"] - final_metrics["tp"])

    objective_status = "DONE"
    objective_status += "_TARGET_RECALL_MET" if final_metrics["recall"] >= args.target_recall else "_TARGET_RECALL_NOT_MET"
    objective_status += "_TPLOSS0" if tp_loss_vs_base == 0 else "_TPLOSS_GT0"
    objective_status += "_FP_REDUCED" if fp_removed_vs_base > 0 else "_FP_NOT_REDUCED"
    objective_status += "_WILSON_PASS" if wilson_low >= args.target_recall else "_WILSON_NOT_PASS"

    policy_artifact = {
        "experiment": "EXP-014B",
        "policy_name": "expanded_high_recall_threshold_safe_veto",
        "objective_status": objective_status,
        "base_policy": base_policy,
        "selected_metrics": final_metrics,
        "base_metrics": compute_metrics(y, base_pred),
        "fp_removed_vs_base": fp_removed_vs_base,
        "tp_loss_vs_base": tp_loss_vs_base,
        "wilson": wilson_df.to_dict(orient="records")[0],
        "selected_rules": selected_df.to_dict(orient="records") if not selected_df.empty else [],
        "notes": [
            "No runtime call.",
            "Base threshold calibrated directly on expanded scored input.",
            "Selected veto rules require TP_loss=0 globally and block_tp_loss_max=0.",
            "If accepted, next step is EXP-014C Frozen Validation without mining."
        ],
    }
    dump_json(policy_artifact, output_dir / "13_policy_artifact.json")

    summary = {
        "experiment": "EXP-014B",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "n_rows": int(len(df)),
        "n_frauds": total_frauds,
        "target_recall": args.target_recall,
        "base_policy": base_policy,
        "base_metrics": compute_metrics(y, base_pred),
        "selected_metrics": final_metrics,
        "fp_removed_vs_base": fp_removed_vs_base,
        "tp_loss_vs_base": tp_loss_vs_base,
        "n_candidates": int(len(candidates)),
        "n_selected_rules": int(len(selected)),
        "wilson_recall_low": wilson_low,
        "wilson_recall_high": wilson_high,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, base_metrics, policy_metrics, selected_df)
    (output_dir / "14_exp014b_report.md").write_text(report, encoding="utf-8")

    log("")
    log("=" * 80)
    log("EXP-014B CONCLUÍDO")
    log("=" * 80)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "02_threshold_sweep.csv",
        output_dir / "03_base_policy_metrics.csv",
        output_dir / "04_veto_candidates.csv",
        output_dir / "06_selected_rules.csv",
        output_dir / "07_policy_metrics.csv",
        output_dir / "08_time_block_metrics.csv",
        output_dir / "09_wilson_recall_ci.csv",
        output_dir / "10_bootstrap_summary.csv",
        output_dir / "13_policy_artifact.json",
        output_dir / "14_exp014b_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
