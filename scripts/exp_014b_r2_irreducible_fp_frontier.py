#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R2 — Expanded Irreducible FP Frontier

Objetivo:
  Encontrar a fronteira de menor FP possível no dataset expandido, usando as
  1.465 fraudes, mantendo recall final >= 95%.

Contexto:
  - EXP-013K/013L consolidou um benchmark excelente em conjunto pequeno:
      TP=118, FP=199, FN=6, recall=95,16%
    mas com apenas 124 fraudes.
  - EXP-014B-R1 confirmou suporte estatístico no expandido:
      TP=1448, FP=20706, FN=17, recall=98,84%, Wilson PASS
    porém FP residual alto demais.
  - Agora o foco é a fronteira irredutível:
      reduzir FP ao máximo, aceitando gastar buffer de TP
      desde que recall final nunca fique abaixo de 95%.

Estratégia:
  1. Carregar dados/exp014a_expanded_scored_input.csv.
  2. Fazer sweep de thresholds em lgbm_r4_score/score_final.
  3. Selecionar bases em faixas de recall:
       95%, 95,5%, 96%, 96,5%, 97%, 98%.
  4. Para cada base:
       - calcular TP buffer contra recall >= 95%;
       - minerar vetos candidatos com TP_loss controlado;
       - executar beam search com orçamento de TP;
       - escolher menor FP final respeitando recall >= 95%.
  5. Escolher a melhor política global: menor FP final com recall >= 95%.
  6. Gerar artefato congelável para EXP-014C.

Importante:
  - Não chama runtime.
  - Usa labels para mineração, portanto se aprovado precisa de EXP-014C
    Frozen Validation sem nova mineração.
  - Tem limite de tempo por base e checkpoints, para evitar travamento.

Uso padrão:
  python scripts/exp_014b_r2_irreducible_fp_frontier.py

Mais rápido:
  python scripts/exp_014b_r2_irreducible_fp_frontier.py --bootstrap-iters 100 --max-seconds-per-base 180

Mais profundo:
  python scripts/exp_014b_r2_irreducible_fp_frontier.py --max-combo-size 4 --max-candidates 600 --beam-width 200 --max-rules 9
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R2"

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
class VetoCandidate:
    rule_id: str
    description: str
    cols: list[str]
    vals: list[str]
    mask: np.ndarray
    tp_loss: int
    fp_removed: int
    n_removed: int
    block_tp_loss_max: int
    fp_per_tp: float


@dataclass
class BeamState:
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
            labels.append(f"{name}_{left:g}_{right:g}")
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
    return ~val, val


def threshold_values_from_scores(scores: pd.Series, n: int) -> list[float]:
    vals = pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if vals.empty:
        return []
    qs = np.linspace(0.0, 1.0, n)
    thresholds = sorted(set(float(x) for x in vals.quantile(qs).to_numpy()))
    thresholds.extend([float(vals.min()), float(vals.max())])
    return sorted(set(thresholds))


def sweep_thresholds(df: pd.DataFrame, blocks: pd.Series, n_thresholds: int, validation_blocks: int) -> pd.DataFrame:
    y = df["is_fraud"].to_numpy(dtype=int)
    disc_mask, val_mask = temporal_discovery_validation_split(blocks, validation_blocks)
    rows = []

    for col in [c for c in SCORE_COL_CANDIDATES if c in df.columns]:
        scores = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        thresholds = threshold_values_from_scores(scores, n_thresholds)
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
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["full_recall", "full_fp"], ascending=[False, True]).reset_index(drop=True)


def parse_base_targets(raw: str) -> list[float]:
    vals = []
    for part in str(raw).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return sorted(set(vals))


def select_base_policies(sweep: pd.DataFrame, base_targets: list[float], validation_recall_floor: float, final_target_recall: float) -> pd.DataFrame:
    rows = []
    for target in base_targets:
        strict = sweep[(sweep["full_recall"] >= target) & (sweep["val_recall"] >= validation_recall_floor)].copy()
        if not strict.empty:
            row = strict.sort_values(["full_fp", "val_fp", "full_precision"], ascending=[True, True, False]).iloc[0]
            status = "STRICT_FULL_AND_VALIDATION_RECALL_MET"
        else:
            full = sweep[sweep["full_recall"] >= target].copy()
            if not full.empty:
                row = full.sort_values(["full_fp", "val_recall", "full_precision"], ascending=[True, False, False]).iloc[0]
                status = "FULL_RECALL_MET_VALIDATION_WARNING"
            else:
                row = sweep.sort_values(["full_recall", "full_fp"], ascending=[False, True]).iloc[0]
                status = "TARGET_NOT_MET_BEST_AVAILABLE"

        rec = row.to_dict()
        rec["base_target_recall"] = target
        rec["selection_status"] = status
        rec["final_target_recall"] = final_target_recall
        rows.append(rec)

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["score_col", "direction", "threshold"]).reset_index(drop=True)
    out["base_id"] = [f"base_{i:03d}" for i in range(len(out))]
    return out


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


def block_tp_loss_max_from_mask(mask: np.ndarray, y: np.ndarray, blocks: pd.Series) -> int:
    out = 0
    bvals = blocks.to_numpy()
    for b in sorted(blocks.dropna().unique()):
        bm = mask & (bvals == b)
        out = max(out, int(((y == 1) & bm).sum()))
    return out


def mine_candidates_for_base(
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
) -> list[VetoCandidate]:
    y = df["is_fraud"].to_numpy(dtype=int)
    pred_pos = base_pred.astype(bool)
    feat = feature_frame(df)
    cols_available = list(feat.columns)

    if tp_budget <= 0:
        allowed_candidate_tp_loss = 0
    else:
        allowed_candidate_tp_loss = min(tp_budget, max_candidate_tp_loss)

    combos = []
    for r in range(2, max_combo_size + 1):
        for combo in itertools.combinations(cols_available, r):
            combo = list(combo)
            if r >= 3:
                combos.append(combo)
            else:
                if any(c in combo for c in ["value_band", "ds_tipo_chave_norm", "periodo_dia", "first_receiver_flag_real", "mbk_available_flag"]):
                    combos.append(combo)

    candidates: list[VetoCandidate] = []

    # Work only over predicted positives.
    base_idx = np.where(pred_pos)[0]
    if len(base_idx) == 0:
        return []

    work = feat.iloc[base_idx].copy()
    work["_is_fraud"] = y[base_idx]
    work["_block"] = blocks.iloc[base_idx].to_numpy()

    for combo in combos:
        if not all(c in work.columns for c in combo):
            continue

        grouped = work.groupby(combo, dropna=False)
        stats = grouped["_is_fraud"].agg(["count", "sum"]).reset_index()
        stats = stats.rename(columns={"count": "n_removed", "sum": "tp_loss"})
        stats["fp_removed"] = stats["n_removed"] - stats["tp_loss"]

        stats = stats[
            (stats["fp_removed"] >= min_fp_removed)
            & (stats["tp_loss"] <= allowed_candidate_tp_loss)
        ].copy()
        if stats.empty:
            continue

        stats["fp_per_tp"] = np.where(stats["tp_loss"] > 0, stats["fp_removed"] / stats["tp_loss"], np.inf)
        stats = stats[(stats["tp_loss"] == 0) | (stats["fp_per_tp"] >= min_fp_per_tp)].copy()
        if stats.empty:
            continue

        # Compute block max TP loss only for promising groups.
        stats = stats.sort_values(["tp_loss", "fp_removed"], ascending=[True, False]).head(top_groups_per_combo)

        for _, row in stats.iterrows():
            vals = [str(row[c]) for c in combo]
            mask = pred_pos.copy()
            for c, v in zip(combo, vals):
                mask = mask & (feat[c].astype(str).to_numpy() == v)

            tp_loss = int(((y == 1) & mask).sum())
            fp_removed = int(((y == 0) & mask).sum())
            n_removed = int(mask.sum())
            if tp_loss > allowed_candidate_tp_loss or fp_removed < min_fp_removed:
                continue

            fp_per_tp = float(fp_removed / tp_loss) if tp_loss > 0 else float("inf")
            if tp_loss > 0 and fp_per_tp < min_fp_per_tp:
                continue

            bmax = block_tp_loss_max_from_mask(mask, y, blocks)
            if bmax > max_block_tp_loss:
                continue

            rid = f"veto_{len(candidates):05d}"
            desc = " AND ".join([f"{c}={v}" for c, v in zip(combo, vals)])
            candidates.append(VetoCandidate(
                rule_id=rid,
                description=desc,
                cols=combo,
                vals=vals,
                mask=mask,
                tp_loss=tp_loss,
                fp_removed=fp_removed,
                n_removed=n_removed,
                block_tp_loss_max=bmax,
                fp_per_tp=fp_per_tp,
            ))

    # Dedupe by mask, keep strongest/simple rule.
    best: dict[bytes, VetoCandidate] = {}
    for c in candidates:
        key = np.packbits(c.mask).tobytes()
        old = best.get(key)
        if old is None:
            best[key] = c
            continue
        if (c.fp_removed, -c.tp_loss, -len(c.description)) > (old.fp_removed, -old.tp_loss, -len(old.description)):
            best[key] = c

    out = list(best.values())
    out.sort(key=lambda c: (c.tp_loss, -c.fp_removed, len(c.description)))
    return out


def candidates_to_df(cands: list[VetoCandidate]) -> pd.DataFrame:
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
        "fp_per_tp": c.fp_per_tp,
    } for i, c in enumerate(cands)])


def selected_rules_df(cands: list[VetoCandidate]) -> pd.DataFrame:
    return pd.DataFrame([{
        "selected_order": i,
        "rule_id": c.rule_id,
        "description": c.description,
        "cols": "|".join(c.cols),
        "vals": "|".join(c.vals),
        "tp_loss_individual": c.tp_loss,
        "fp_removed_individual": c.fp_removed,
        "block_tp_loss_max_individual": c.block_tp_loss_max,
        "fp_per_tp_individual": c.fp_per_tp,
    } for i, c in enumerate(cands)])


def search_best_with_budget(
    cands: list[VetoCandidate],
    base_pred: np.ndarray,
    y: np.ndarray,
    tp_budget: int,
    beam_width: int,
    max_rules: int,
    max_candidates: int,
    max_seconds: int,
    output_dir: Path,
    base_id: str,
) -> tuple[pd.DataFrame, BeamState, list[VetoCandidate], str]:
    t0 = time.perf_counter()

    # Candidate ordering: zero-loss first, then high FP/TP ratio.
    usable = [c for c in cands if c.tp_loss <= tp_budget]
    usable.sort(key=lambda c: (c.tp_loss > 0, -c.fp_removed if c.tp_loss == 0 else -c.fp_per_tp, -c.fp_removed))
    usable = usable[:max_candidates]

    zero = np.zeros(len(y), dtype=bool)
    initial = BeamState(mask=zero, rule_indices=tuple(), tp_loss=0, fp_removed=0)
    states = [initial]
    best = initial
    rows = []
    stop_reason = "completed"

    try:
        for depth in range(1, max_rules + 1):
            if time.perf_counter() - t0 >= max_seconds:
                stop_reason = f"max_seconds_before_depth_{depth}"
                break

            depth_t0 = time.perf_counter()
            next_states: dict[bytes, BeamState] = {}

            for state in states:
                last = state.rule_indices[-1] if state.rule_indices else -1
                for i in range(last + 1, len(usable)):
                    c = usable[i]
                    new_mask = state.mask | c.mask
                    if np.array_equal(new_mask, state.mask):
                        continue

                    # Exact union loss, required because positive TP_loss candidates are allowed.
                    tp_loss = int(((y == 1) & new_mask).sum())
                    if tp_loss > tp_budget:
                        continue

                    fp_removed = int(((y == 0) & new_mask).sum())
                    if fp_removed <= state.fp_removed:
                        continue

                    key = np.packbits(new_mask).tobytes()
                    ns = BeamState(new_mask, state.rule_indices + (i,), tp_loss, fp_removed)
                    old = next_states.get(key)
                    if old is None or (ns.fp_removed, -ns.tp_loss, -len(ns.rule_indices)) > (old.fp_removed, -old.tp_loss, -len(old.rule_indices)):
                        next_states[key] = ns

            if not next_states:
                stop_reason = f"no_next_states_at_depth_{depth}"
                break

            states = sorted(next_states.values(), key=lambda s: (s.fp_removed, -s.tp_loss, -len(s.rule_indices)), reverse=True)[:beam_width]
            if (states[0].fp_removed, -states[0].tp_loss) > (best.fp_removed, -best.tp_loss):
                best = states[0]

            for s in states[:50]:
                pred = base_pred.copy()
                pred[s.mask] = 0
                m = compute_metrics(y, pred)
                rows.append({
                    "base_id": base_id,
                    "depth": depth,
                    "tp_loss": s.tp_loss,
                    "fp_removed": s.fp_removed,
                    "n_rules": len(s.rule_indices),
                    **m,
                    "rule_ids": "|".join(usable[i].rule_id for i in s.rule_indices),
                    "rule_descriptions": " || ".join(usable[i].description for i in s.rule_indices),
                })

            pd.DataFrame(rows).to_csv(output_dir / f"checkpoint_frontier_{base_id}_depth_{depth:02d}.csv", index=False)
            selected_rules_df([usable[i] for i in best.rule_indices]).to_csv(output_dir / f"checkpoint_selected_{base_id}_depth_{depth:02d}.csv", index=False)

            elapsed = time.perf_counter() - t0
            log(f"    {base_id} depth={depth}: best_fp_removed={best.fp_removed}, tp_loss={best.tp_loss}/{tp_budget}, states={len(states)}, depth_s={time.perf_counter()-depth_t0:.1f}, total_s={elapsed:.1f}")

            if elapsed >= max_seconds:
                stop_reason = f"max_seconds_after_depth_{depth}"
                break

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt_saved_best"
        log(f"KeyboardInterrupt em {base_id}; salvando melhor estado.")

    if not rows:
        m = compute_metrics(y, base_pred)
        rows = [{
            "base_id": base_id,
            "depth": 0,
            "tp_loss": 0,
            "fp_removed": 0,
            "n_rules": 0,
            **m,
            "rule_ids": "",
            "rule_descriptions": "",
        }]

    frontier = pd.DataFrame(rows)
    selected = [usable[i] for i in best.rule_indices]
    return frontier, best, selected, stop_reason


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


def make_report(summary: dict[str, Any], selected_metrics: pd.DataFrame, selected_rules: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014B-R2 — Expanded Irreducible FP Frontier")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Best base: `{summary['best_base_id']}`")
    lines.append(f"- FP final: `{summary['selected_metrics']['fp']}`")
    lines.append(f"- Recall final: `{summary['selected_metrics']['recall']}`")
    lines.append(f"- TP loss vs base: `{summary['tp_loss_vs_base']}`")
    lines.append(f"- FP removed vs base: `{summary['fp_removed_vs_base']}`")
    lines.append("")
    lines.append("## Métricas selecionadas")
    lines.append(selected_metrics.to_markdown(index=False))
    lines.append("")
    lines.append("## Regras selecionadas")
    if selected_rules.empty:
        lines.append("Nenhuma regra selecionada.")
    else:
        lines.append(selected_rules.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretação")
    lines.append("Este experimento procura a fronteira de menor FP no dataset expandido, permitindo gasto controlado do buffer de TP desde que o recall final permaneça >=95%.")
    if summary["selected_metrics"]["recall"] >= summary["target_recall"]:
        lines.append("A política respeita o piso de recall. Se o FP residual for operacionalmente aceitável, o próximo passo é EXP-014C Frozen Validation sem nova mineração.")
    else:
        lines.append("A política não respeitou o piso de recall; não promover.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--base-targets", default="0.95,0.955,0.96,0.965,0.97,0.98")
    parser.add_argument("--validation-recall-floor", type=float, default=0.95)
    parser.add_argument("--thresholds", type=int, default=500)
    parser.add_argument("--time-blocks", type=int, default=10)
    parser.add_argument("--validation-blocks", type=int, default=3)
    parser.add_argument("--min-fp-removed", type=int, default=40)
    parser.add_argument("--max-combo-size", type=int, default=3)
    parser.add_argument("--top-groups-per-combo", type=int, default=40)
    parser.add_argument("--max-candidate-tp-loss", type=int, default=8)
    parser.add_argument("--max-block-tp-loss", type=int, default=2)
    parser.add_argument("--min-fp-per-tp", type=float, default=80.0)
    parser.add_argument("--max-candidates", type=int, default=450)
    parser.add_argument("--beam-width", type=int, default=160)
    parser.add_argument("--max-rules", type=int, default=8)
    parser.add_argument("--max-seconds-per-base", type=int, default=300)
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    parser.add_argument("--false-positive-sample", type=int, default=5000)
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
    log("EXP-014B-R2 — Expanded Irreducible FP Frontier")
    log("=" * 80)
    log(f"Input: {input_path}")
    log(f"Output: {output_dir}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    df = ensure_bins_and_guards(df)
    contract = contract_report(df)
    dump_json(contract, output_dir / "01_input_contract.json")
    if not contract["contract_ok"]:
        raise RuntimeError(f"Contrato de input falhou: {contract['missing']}")

    y = df["is_fraud"].to_numpy(dtype=int)
    total_frauds = int(df["is_fraud"].sum())
    min_tp_required = int(math.ceil(args.target_recall * total_frauds))
    blocks = make_time_blocks(df, args.time_blocks)

    log("[1/6] Threshold sweep...")
    sweep = sweep_thresholds(df, blocks, args.thresholds, args.validation_blocks)
    sweep.to_csv(output_dir / "02_threshold_sweep.csv", index=False)

    base_targets = parse_base_targets(args.base_targets)
    bases = select_base_policies(sweep, base_targets, args.validation_recall_floor, args.target_recall)
    bases.to_csv(output_dir / "03_base_policies.csv", index=False)
    log(f"Bases selecionadas: {len(bases)}")

    all_frontiers = []
    all_candidates_summary = []
    best_global = None

    log("[2/6] Minerando e buscando por base...")
    for _, base_row in bases.iterrows():
        base_spec = base_row.to_dict()
        base_id = str(base_spec["base_id"])
        log("")
        log(f"--- {base_id}: target={base_spec['base_target_recall']}, {base_spec['score_col']} {base_spec['direction']} {base_spec['threshold']} ---")

        base_pred = apply_threshold(df, base_spec)
        base_metrics = compute_metrics(y, base_pred)
        tp_budget = max(0, base_metrics["tp"] - min_tp_required)
        log(f"  base_metrics={base_metrics}, tp_budget={tp_budget}")

        base_outdir = per_base_dir / base_id
        base_outdir.mkdir(parents=True, exist_ok=True)

        cand_t0 = time.perf_counter()
        cands = mine_candidates_for_base(
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
        )
        cand_df = candidates_to_df(cands)
        cand_df.to_csv(base_outdir / "candidates.csv", index=False)
        all_candidates_summary.append({
            "base_id": base_id,
            "n_candidates": int(len(cands)),
            "candidate_mining_seconds": round(time.perf_counter() - cand_t0, 2),
            "tp_budget": tp_budget,
            **{f"base_{k}": v for k, v in base_metrics.items()},
        })
        log(f"  candidatos={len(cands)} em {time.perf_counter()-cand_t0:.1f}s")

        if not cands:
            frontier = pd.DataFrame([{
                "base_id": base_id,
                "depth": 0,
                "tp_loss": 0,
                "fp_removed": 0,
                "n_rules": 0,
                **base_metrics,
                "rule_ids": "",
                "rule_descriptions": "",
            }])
            best = BeamState(np.zeros(len(y), dtype=bool), tuple(), 0, 0)
            selected = []
            stop_reason = "no_candidates"
        else:
            frontier, best, selected, stop_reason = search_best_with_budget(
                cands=cands,
                base_pred=base_pred,
                y=y,
                tp_budget=tp_budget,
                beam_width=args.beam_width,
                max_rules=args.max_rules,
                max_candidates=args.max_candidates,
                max_seconds=args.max_seconds_per_base,
                output_dir=base_outdir,
                base_id=base_id,
            )

        frontier.to_csv(base_outdir / "frontier.csv", index=False)
        selected_df = selected_rules_df(selected)
        selected_df.to_csv(base_outdir / "selected_rules.csv", index=False)
        all_frontiers.append(frontier)

        final_pred = base_pred.copy()
        final_pred[best.mask] = 0
        final_metrics = compute_metrics(y, final_pred)
        fp_removed_vs_base = base_metrics["fp"] - final_metrics["fp"]
        tp_loss_vs_base = base_metrics["tp"] - final_metrics["tp"]

        base_result = {
            "base_id": base_id,
            "stop_reason": stop_reason,
            "base_spec": base_spec,
            "base_metrics": base_metrics,
            "selected_metrics": final_metrics,
            "tp_budget": tp_budget,
            "tp_loss_vs_base": int(tp_loss_vs_base),
            "fp_removed_vs_base": int(fp_removed_vs_base),
            "n_selected_rules": int(len(selected)),
            "selected_rules": selected_df.to_dict(orient="records") if not selected_df.empty else [],
        }
        dump_json(base_result, base_outdir / "base_result.json")

        valid = final_metrics["recall"] >= args.target_recall
        if valid:
            candidate_rank = (final_metrics["fp"], -final_metrics["recall"], -final_metrics["precision"], tp_loss_vs_base)
        else:
            candidate_rank = (10**18, -final_metrics["recall"], final_metrics["fp"], tp_loss_vs_base)

        if best_global is None or candidate_rank < best_global["rank"]:
            best_global = {
                "rank": candidate_rank,
                "base_id": base_id,
                "base_spec": base_spec,
                "base_metrics": base_metrics,
                "final_pred": final_pred,
                "selected_metrics": final_metrics,
                "best_state": best,
                "selected_rules": selected,
                "selected_rules_df": selected_df,
                "stop_reason": stop_reason,
                "tp_budget": tp_budget,
                "tp_loss_vs_base": int(tp_loss_vs_base),
                "fp_removed_vs_base": int(fp_removed_vs_base),
            }

    if best_global is None:
        raise RuntimeError("Nenhuma base produziu resultado.")

    all_frontier_df = pd.concat(all_frontiers, ignore_index=True) if all_frontiers else pd.DataFrame()
    all_frontier_df.to_csv(output_dir / "04_frontier_all.csv", index=False)
    pd.DataFrame(all_candidates_summary).to_csv(output_dir / "05_candidate_summary.csv", index=False)

    log("[3/6] Consolidando melhor política global...")
    selected_pred = best_global["final_pred"]
    df["exp014b_r2_irreducible_fp_pred"] = selected_pred

    selected_rules = best_global["selected_rules_df"]
    selected_rules.to_csv(output_dir / "06_selected_rules.csv", index=False)

    metrics_rows = []
    # Runtime comparison if present.
    for runtime_col in ["exp014a_frozen_pred", "exp013k_residual_fp_pred"]:
        if runtime_col in df.columns:
            metrics_rows.append({"policy_name": f"RUNTIME_FINAL_{runtime_col}", **compute_metrics(y, df[runtime_col].to_numpy(dtype=int))})
            break
    metrics_rows.append({"policy_name": "EXP014B_R2_SELECTED_BASE", **best_global["base_metrics"]})
    metrics_rows.append({"policy_name": "EXP014B_R2_IRREDUCIBLE_FP_FRONTIER", **best_global["selected_metrics"]})
    selected_metrics_df = pd.DataFrame(metrics_rows)
    selected_metrics_df.to_csv(output_dir / "07_selected_policy_metrics.csv", index=False)

    log("[4/6] Blocos e estatística...")
    block_df = pd.concat([
        block_metrics(df, apply_threshold(df, best_global["base_spec"]), blocks, "EXP014B_R2_SELECTED_BASE"),
        block_metrics(df, selected_pred, blocks, "EXP014B_R2_IRREDUCIBLE_FP_FRONTIER"),
    ], ignore_index=True)
    block_df.to_csv(output_dir / "08_time_block_metrics.csv", index=False)

    selected_metrics = best_global["selected_metrics"]
    wilson_low, wilson_high = wilson_ci(selected_metrics["tp"], total_frauds)
    wilson_df = pd.DataFrame([{
        "metric": "recall",
        "successes_tp": selected_metrics["tp"],
        "n_frauds": total_frauds,
        "point_estimate": selected_metrics["recall"],
        "wilson_low": wilson_low,
        "wilson_high": wilson_high,
        "target_recall": args.target_recall,
        "min_tp_required": min_tp_required,
        "tp_buffer_vs_target": selected_metrics["tp"] - min_tp_required,
        "wilson_low_ge_target": bool(wilson_low >= args.target_recall),
    }])
    wilson_df.to_csv(output_dir / "09_wilson_recall_ci.csv", index=False)

    boot_df = bootstrap_summary(df, "exp014b_r2_irreducible_fp_pred", args.bootstrap_iters, args.seed, args.target_recall)
    boot_df.to_csv(output_dir / "10_bootstrap_summary.csv", index=False)

    log("[5/6] FNs, FPs e artefatos...")
    fn = df[(df["is_fraud"] == 1) & (df["exp014b_r2_irreducible_fp_pred"] == 0)].copy()
    fp = df[(df["is_fraud"] == 0) & (df["exp014b_r2_irreducible_fp_pred"] == 1)].copy()
    fn.to_csv(output_dir / "11_false_negatives.csv", index=False)
    if len(fp) > args.false_positive_sample:
        fp = fp.sample(args.false_positive_sample, random_state=args.seed)
    fp.to_csv(output_dir / "12_false_positives_sample.csv", index=False)

    if not args.no_write_predictions:
        df.to_csv(output_dir / "14_predictions.csv", index=False)

    objective_status = "DONE"
    objective_status += "_TARGET_RECALL_MET" if selected_metrics["recall"] >= args.target_recall else "_TARGET_RECALL_NOT_MET"
    objective_status += "_FP_FRONTIER_SELECTED"
    objective_status += "_WILSON_PASS" if wilson_low >= args.target_recall else "_WILSON_NOT_PASS"
    if best_global["tp_loss_vs_base"] > 0:
        objective_status += "_TP_BUDGET_USED"
    else:
        objective_status += "_TPLOSS0"

    artifact = {
        "experiment": "EXP-014B-R2",
        "policy_name": "expanded_irreducible_fp_frontier",
        "objective_status": objective_status,
        "best_base_id": best_global["base_id"],
        "base_policy": best_global["base_spec"],
        "base_metrics": best_global["base_metrics"],
        "selected_metrics": selected_metrics,
        "target_recall": args.target_recall,
        "min_tp_required": min_tp_required,
        "tp_budget": best_global["tp_budget"],
        "tp_loss_vs_base": best_global["tp_loss_vs_base"],
        "fp_removed_vs_base": best_global["fp_removed_vs_base"],
        "wilson": wilson_df.to_dict(orient="records")[0],
        "selected_rules": selected_rules.to_dict(orient="records") if not selected_rules.empty else [],
        "notes": [
            "No runtime call.",
            "Uses expanded scored dataset with 1465 frauds.",
            "Searches minimum FP frontier with final recall >= 95%.",
            "Uses labels for mining; if accepted, requires EXP-014C Frozen Validation."
        ],
    }
    dump_json(artifact, output_dir / "13_policy_artifact.json")

    summary = {
        "experiment": "EXP-014B-R2",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "n_rows": int(len(df)),
        "n_frauds": total_frauds,
        "target_recall": args.target_recall,
        "min_tp_required": min_tp_required,
        "base_targets": base_targets,
        "best_base_id": best_global["base_id"],
        "best_stop_reason": best_global["stop_reason"],
        "base_policy": best_global["base_spec"],
        "base_metrics": best_global["base_metrics"],
        "selected_metrics": selected_metrics,
        "tp_budget": best_global["tp_budget"],
        "tp_loss_vs_base": best_global["tp_loss_vs_base"],
        "fp_removed_vs_base": best_global["fp_removed_vs_base"],
        "n_selected_rules": int(len(best_global["selected_rules"])),
        "wilson_recall_low": wilson_low,
        "wilson_recall_high": wilson_high,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, selected_metrics_df, selected_rules)
    (output_dir / "15_exp014b_r2_report.md").write_text(report, encoding="utf-8")

    log("[6/6] Concluído.")
    log("")
    log("=" * 80)
    log("EXP-014B-R2 CONCLUÍDO")
    log("=" * 80)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "03_base_policies.csv",
        output_dir / "04_frontier_all.csv",
        output_dir / "06_selected_rules.csv",
        output_dir / "07_selected_policy_metrics.csv",
        output_dir / "08_time_block_metrics.csv",
        output_dir / "09_wilson_recall_ci.csv",
        output_dir / "10_bootstrap_summary.csv",
        output_dir / "13_policy_artifact.json",
        output_dir / "15_exp014b_r2_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
