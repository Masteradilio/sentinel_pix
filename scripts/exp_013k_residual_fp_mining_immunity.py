#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-013K — Residual FP Mining with TP/FN Immunity

Objetivo:
  Partir do benchmark vencedor EXP-013J STRICT_RECALL95_SAFE_ONLY:
      TP=118
      FP=349
      FN=6
      recall=95.16%

  E minerar os 349 FPs residuais procurando novos microvetos
  com restrição rígida:
      TP_loss global = 0
      TP_loss por bloco temporal = 0
      TP >= 118
      recall >= 95%

Estratégia:
  1. Usa pred_STRICT_RECALL95_SAFE_ONLY como base.
  2. Cria guardas de imunidade:
       - não vetar sinais fortes SE/BEH/runtime;
       - não vetar padrões de TP atuais;
       - registrar assinatura dos FNs atuais para governança.
  3. Minera segmentos residuais FP-only usando combinações:
       - value_band
       - ds_tipo_chave_norm
       - periodo_dia
       - first_receiver_flag_real
       - mbk_available_flag
       - bins de lgbm, IF, valor, score_final e histórico.
  4. Seleciona combinações com TP_loss=0 e suporte mínimo.
  5. Faz beam search pequeno para maximizar FP removidos.
  6. Gera política congelável apenas se mantiver recall >=95%.

Entradas default:
  resultados/experimentos/EXP-013J/06_predictions_by_scenario.csv

Uso:
  python scripts/exp_013k_residual_fp_mining_immunity.py

Execução mais profunda:
  python scripts/exp_013k_residual_fp_mining_immunity.py --max-combo-size 5 --beam-width 250 --max-rules 10

Saídas:
  resultados/experimentos/EXP-013K/
    00_run_summary.json
    01_base_metrics.csv
    02_residual_fp_profile.csv
    03_immunity_guards.csv
    04_residual_fp_candidates.csv
    05_frontier.csv
    06_selected_rules.csv
    07_selected_predictions.csv
    08_false_negatives.csv
    09_false_positives.csv
    10_time_block_metrics.csv
    11_bootstrap_confidence_intervals.csv
    12_policy_artifact.json
    13_exp013k_report.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
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
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "backend").exists() else Path.cwd()

DEFAULT_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013J" / "06_predictions_by_scenario.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013K"

BASE_PRED_CANDIDATES = [
    "pred_STRICT_RECALL95_SAFE_ONLY",
    "exp013h_frozen_pred",
    "exp013g_micro_pred",
    "pred_HIGH_RECALL_95",
]


@dataclass
class Candidate:
    rule_id: str
    family: str
    description: str
    mask: np.ndarray
    tp_loss: int
    fp_removed: int
    n_removed: int
    block_tp_loss_max: int
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

    for c in BASE_PRED_CANDIDATES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "runtime_flagged" not in df.columns:
        if "decisao" in df.columns:
            df["runtime_flagged"] = df["decisao"].astype(str).str.upper().isin({"CONFIRMAR", "BLOQUEAR"}).astype(int)
        else:
            df["runtime_flagged"] = 0
    df["runtime_flagged"] = pd.to_numeric(df["runtime_flagged"], errors="coerce").fillna(0).astype(int)

    if "transaction_id" in df.columns:
        df["transaction_id"] = df["transaction_id"].astype("string").str.strip()

    for c in ["event_datetime", "data_pix"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    return df.reset_index(drop=True)


def pick_pred_col(df: pd.DataFrame, requested: str | None) -> str:
    if requested and requested in df.columns:
        return requested
    for c in BASE_PRED_CANDIDATES:
        if c in df.columns:
            return c
    raise RuntimeError("Não encontrei coluna de predição base. Use --pred-col.")


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


def text(df: pd.DataFrame, names: str | list[str], default: str = "<MISSING>") -> pd.Series:
    col = pick_col(df, names)
    if col is None:
        return pd.Series([default] * len(df), index=df.index, dtype="string")
    return df[col].astype("string").fillna(default).astype(str)


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


def add_error_labels(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    out = df.copy()
    y = out["is_fraud"].to_numpy(dtype=int)
    p = out[pred_col].to_numpy(dtype=int)
    out["exp013k_error_type"] = np.select(
        [
            (y == 1) & (p == 1),
            (y == 0) & (p == 1),
            (y == 1) & (p == 0),
            (y == 0) & (p == 0),
        ],
        ["TP", "FP", "FN", "TN"],
        default="UNK",
    )
    return out


def strong_module_preserve(df: pd.DataFrame) -> np.ndarray:
    se_score = num(df, ["se_score_x", "se_score_y", "se_score"], 0.0)
    se_count = num(df, ["se_patterns_count", "se_pattern_count"], 0.0)
    beh_score = num(df, ["beh_score", "behavioral_score"], 0.0)
    beh_count = num(df, ["beh_factors_count", "behavioral_risk_factor_count"], 0.0)
    runtime = num(df, "runtime_flagged", 0.0)

    return (
        (se_score >= 40)
        | (se_count >= 2)
        | (beh_score >= 25)
        | (beh_count >= 2)
        | (runtime >= 1)
    ).to_numpy(dtype=bool)


def qbin_series(s: pd.Series, name: str, bins: list[float] | None = None) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if bins is None:
        try:
            out = pd.qcut(vals.rank(method="first"), q=5, labels=[f"{name}_Q{i}" for i in range(1, 6)], duplicates="drop")
            return out.astype("string").fillna(f"{name}_MISSING").astype(str)
        except Exception:
            return pd.Series([f"{name}_MISSING"] * len(vals), index=vals.index)

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


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    for c in ["value_band", "ds_tipo_chave_norm", "periodo_dia", "first_receiver_flag_real", "mbk_available_flag"]:
        if c in df.columns:
            feat[c] = text(df, c)

    lgbm = num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0)
    ifp = num(df, ["if_percentile_x", "if_percentile_y", "if_percentile"], 0.0)
    vl = num(df, "vl_pix", 0.0)
    score_final = num(df, "score_final", 0.0)
    qtd_rec = num(df, "qtd_pix_recebidos_180d", 0.0)
    valor_rec = num(df, "valor_total_recebido_180d", 0.0)
    ratio = num(df, "ratio_valor_media_pagador_90d", 0.0)

    feat["lgbm_bin"] = qbin_series(lgbm, "lgbm", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1])
    feat["if_bin"] = qbin_series(ifp, "if", [0.32, 0.5, 0.7, 0.85, 0.95])
    feat["vl_bin"] = qbin_series(vl, "vl", [20, 50, 100, 250, 500, 1000, 5000, 10000])
    feat["score_bin"] = qbin_series(score_final, "score", [0.5, 1, 2, 3, 5, 10])
    feat["qtd_rec_bin"] = qbin_series(qtd_rec, "qtdrec", [0, 1, 2, 5, 10, 20, 50, 100])
    feat["valor_rec_bin"] = qbin_series(valor_rec, "valrec", [0, 100, 500, 1000, 5000, 10000, 25000])
    feat["ratio_bin"] = qbin_series(ratio, "ratio", [0.05, 0.1, 0.2, 0.5, 1, 2, 5])

    # Useful binary module guards/absence flags.
    preserve = strong_module_preserve(df)
    feat["module_quiet"] = np.where(preserve, "module_strong", "module_quiet")

    return feat


def sanitize_id(s: str, max_len: int = 140) -> str:
    t = re.sub(r"[^A-Za-z0-9_]+", "_", str(s))
    t = re.sub(r"_+", "_", t).strip("_")
    return t[:max_len] or "rule"


def candidate_dataframe(candidates: list[Candidate]) -> pd.DataFrame:
    return pd.DataFrame([{
        "candidate_index": i,
        "rule_id": c.rule_id,
        "family": c.family,
        "description": c.description,
        "tp_loss": c.tp_loss,
        "fp_removed": c.fp_removed,
        "n_removed": c.n_removed,
        "block_tp_loss_max": c.block_tp_loss_max,
        "params_json": json.dumps(c.params, ensure_ascii=False),
    } for i, c in enumerate(candidates)])


def dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    best: dict[bytes, Candidate] = {}
    for c in candidates:
        key = np.packbits(c.mask).tobytes()
        old = best.get(key)
        if old is None:
            best[key] = c
        else:
            new_key = (c.fp_removed, -c.tp_loss, -len(c.description))
            old_key = (old.fp_removed, -old.tp_loss, -len(old.description))
            if new_key > old_key:
                best[key] = c

    out = list(best.values())
    out.sort(key=lambda c: (c.tp_loss, c.block_tp_loss_max, -c.fp_removed, len(c.description)))
    return out


def mine_candidates(
    df: pd.DataFrame,
    pred_col: str,
    blocks: pd.Series,
    min_fp_removed: int,
    max_combo_size: int,
    top_groups_per_combo: int,
    require_module_quiet: bool,
) -> list[Candidate]:
    y = df["is_fraud"].to_numpy(dtype=int)
    pred_pos = df[pred_col].to_numpy(dtype=int).astype(bool)
    preserve = strong_module_preserve(df)
    feat = build_feature_frame(df)

    candidate_cols = list(feat.columns)
    # Keep combos meaningful and avoid too many large, redundant combinations.
    base_cols = [c for c in ["value_band", "ds_tipo_chave_norm", "periodo_dia", "first_receiver_flag_real", "mbk_available_flag"] if c in feat.columns]
    bin_cols = [c for c in candidate_cols if c.endswith("_bin") or c == "module_quiet"]

    combos = []

    # Segment + one/two score bins.
    for r in range(2, max_combo_size + 1):
        for combo in itertools.combinations(candidate_cols, r):
            combo = list(combo)
            # At least one business categorical and one score/bin/module dimension.
            if not any(c in base_cols for c in combo):
                continue
            if not any(c in bin_cols for c in combo):
                continue
            # Avoid too many pure numeric-bin combos.
            if len([c for c in combo if c in bin_cols]) > 3:
                continue
            combos.append(combo)

    log(f"  combos gerados={len(combos)}")

    candidates: list[Candidate] = []

    for ci, combo in enumerate(combos):
        subset = feat.loc[pred_pos, combo].copy()
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

            if require_module_quiet:
                mask = mask & (~preserve)

            mask = mask & pred_pos
            if not mask.any():
                continue

            tp_loss = int(((y == 1) & mask).sum())
            fp_removed = int(((y == 0) & mask).sum())

            if fp_removed < min_fp_removed:
                continue

            # This experiment only keeps global TP-loss zero candidates.
            if tp_loss != 0:
                continue

            block_tp_losses = []
            for b in sorted(blocks.dropna().unique()):
                bm = mask & (blocks.to_numpy() == b)
                block_tp_losses.append(int(((y == 1) & bm).sum()))
            block_tp_loss_max = max(block_tp_losses) if block_tp_losses else 0
            if block_tp_loss_max != 0:
                continue

            group_rows.append((fp_removed, key, mask, tp_loss, block_tp_loss_max))

        if not group_rows:
            continue

        group_rows.sort(key=lambda x: x[0], reverse=True)

        for rank, (fp_removed, key, mask, tp_loss, block_tp_loss_max) in enumerate(group_rows[:top_groups_per_combo]):
            key_tuple = key if isinstance(key, tuple) else (key,)
            desc_parts = [f"{c}={v}" for c, v in zip(combo, key_tuple)]
            desc = " AND ".join(desc_parts)
            rid = sanitize_id(f"residual_{len(candidates):05d}_{desc}")
            candidates.append(Candidate(
                rule_id=rid,
                family="residual_combo_veto",
                description=desc,
                mask=mask,
                tp_loss=tp_loss,
                fp_removed=fp_removed,
                n_removed=int(mask.sum()),
                block_tp_loss_max=block_tp_loss_max,
                params={"combo_cols": combo, "combo_values": [str(v) for v in key_tuple], "require_module_quiet": require_module_quiet},
            ))

    out = dedupe_candidates(candidates)
    log(f"  candidatos TP0/blocoTP0 após dedupe={len(out)}")
    return out


def search_best(
    candidates: list[Candidate],
    base_pred: np.ndarray,
    y: np.ndarray,
    max_candidates: int,
    beam_width: int,
    max_rules: int,
) -> tuple[pd.DataFrame, State, list[Candidate]]:
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
                ns = State(mask=new_mask, rule_indices=state.rule_indices + (i,), tp_loss=tp_loss, fp_removed=fp_removed)
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
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "rule_ids": "|".join(usable[i].rule_id for i in s.rule_indices),
                "rule_descriptions": " || ".join(usable[i].description for i in s.rule_indices),
            })

        log(f"  depth={depth}: states={len(states)}, best_fp_removed={best.fp_removed}")

    if not rows:
        pred = base_pred.copy()
        m = compute_metrics(y, pred)
        rows = [{
            "depth": 0,
            "tp_loss": 0,
            "fp_removed": 0,
            "n_rules": 0,
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "rule_ids": "",
            "rule_descriptions": "",
        }]

    frontier = pd.DataFrame(rows).sort_values(["fp", "n_rules"], ascending=[True, True]).reset_index(drop=True)
    return frontier, best, usable


def block_metrics(df: pd.DataFrame, pred: np.ndarray, blocks: pd.Series, policy_name: str) -> pd.DataFrame:
    rows = []
    for b in sorted(blocks.dropna().unique()):
        part = df.loc[blocks == b].copy()
        pred_b = pred[blocks == b]
        m = compute_metrics(part["is_fraud"].to_numpy(dtype=int), pred_b)
        m.update({
            "policy_name": policy_name,
            "block": int(b),
            "n_rows": int(len(part)),
            "n_frauds": int(part["is_fraud"].sum()),
            "dt_min": str(part["data_pix"].min().date()) if "data_pix" in part.columns and part["data_pix"].notna().any() else None,
            "dt_max": str(part["data_pix"].max().date()) if "data_pix" in part.columns and part["data_pix"].notna().any() else None,
        })
        rows.append(m)
    return pd.DataFrame(rows)


def bootstrap_eval(df: pd.DataFrame, pred_col: str, iters: int, seed: int, target_recall: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n = len(df)

    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        y = df.iloc[idx]["is_fraud"].to_numpy(dtype=int)
        pred = df.iloc[idx][pred_col].to_numpy(dtype=int)
        rows.append(compute_metrics(y, pred))

    boot = pd.DataFrame(rows)
    out = []
    for metric in ["tp", "fp", "fn", "precision", "recall", "f1", "fpr"]:
        vals = boot[metric].astype(float)
        out.append({
            "metric": metric,
            "mean": float(vals.mean()),
            "p025": float(vals.quantile(0.025)),
            "p050": float(vals.quantile(0.50)),
            "p975": float(vals.quantile(0.975)),
            "target_recall": target_recall if metric == "recall" else None,
            "p_below_target_recall": float((boot["recall"] < target_recall).mean()) if metric == "recall" else None,
        })
    return pd.DataFrame(out)


def immunity_guards(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    et = add_error_labels(df, pred_col)["exp013k_error_type"]
    rows = []

    signals = {
        "module_strong_preserve": strong_module_preserve(df),
        "se_score_ge_40": (num(df, ["se_score_x", "se_score_y", "se_score"], 0.0) >= 40).to_numpy(dtype=bool),
        "beh_score_ge_25": (num(df, ["beh_score", "behavioral_score"], 0.0) >= 25).to_numpy(dtype=bool),
        "runtime_flagged": (num(df, "runtime_flagged", 0.0) >= 1).to_numpy(dtype=bool),
        "lgbm_fn_low_range_approx": (
            (num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0) >= 0.00019755638)
            & (num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0) <= 0.007845118)
        ).to_numpy(dtype=bool),
    }

    for name, mask in signals.items():
        rows.append({
            "guard": name,
            "n_total": int(mask.sum()),
            "n_tp": int(((et == "TP") & mask).sum()),
            "n_fp": int(((et == "FP") & mask).sum()),
            "n_fn": int(((et == "FN") & mask).sum()),
            "n_tn": int(((et == "TN") & mask).sum()),
            "role": "preserve_guard",
        })

    return pd.DataFrame(rows)


def residual_fp_profile(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    d = add_error_labels(df, pred_col)
    fp = d[d["exp013k_error_type"] == "FP"].copy()

    rows = []
    cols = [c for c in ["value_band", "ds_tipo_chave_norm", "periodo_dia", "first_receiver_flag_real", "mbk_available_flag"] if c in fp.columns]
    for c in cols:
        vc = fp[c].astype("string").fillna("<MISSING>").value_counts().reset_index()
        vc.columns = ["value", "n_fp"]
        vc["feature"] = c
        rows.append(vc[["feature", "value", "n_fp"]])

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True).sort_values(["feature", "n_fp"], ascending=[True, False]).reset_index(drop=True)


def make_report(summary: dict[str, Any], selected_rules: pd.DataFrame, frontier: pd.DataFrame, blocks: pd.DataFrame, boot: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-013K — Residual FP Mining with TP/FN Immunity")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Base EXP-013J: TP={summary['base_metrics']['tp']}, FP={summary['base_metrics']['fp']}, FN={summary['base_metrics']['fn']}, recall={summary['base_metrics']['recall']}")
    lines.append(f"- Selecionado EXP-013K: TP={summary['selected_metrics']['tp']}, FP={summary['selected_metrics']['fp']}, FN={summary['selected_metrics']['fn']}, recall={summary['selected_metrics']['recall']}, precision={summary['selected_metrics']['precision']}")
    lines.append(f"- FP removidos vs base: {summary['fp_removed_vs_base']}")
    lines.append(f"- TP loss vs base: {summary['tp_loss_vs_base']}")
    lines.append("")
    lines.append("## Regras selecionadas")
    if selected_rules.empty:
        lines.append("Nenhuma regra selecionada.")
    else:
        lines.append(selected_rules[["family", "description", "tp_loss", "fp_removed", "block_tp_loss_max"]].to_markdown(index=False))
    lines.append("")
    lines.append("## Top fronteira")
    if frontier.empty:
        lines.append("Sem fronteira.")
    else:
        lines.append(frontier[["depth", "tp", "fp", "fn", "precision", "recall", "fp_removed", "n_rules"]].head(20).to_markdown(index=False))
    lines.append("")
    lines.append("## Blocos temporais")
    lines.append(blocks.to_markdown(index=False))
    lines.append("")
    lines.append("## Bootstrap recall")
    lines.append(boot[boot["metric"] == "recall"].to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretação")
    if summary["fp_removed_vs_base"] > 0:
        lines.append("A mineração residual encontrou microvetos adicionais com TP_loss=0. Esta política pode substituir o EXP-013J como novo benchmark apenas após validação congelada.")
    else:
        lines.append("Nenhum ganho adicional seguro foi encontrado. Manter EXP-013J como benchmark.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pred-col", default=None)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--time-blocks", type=int, default=5)
    parser.add_argument("--bootstrap-iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-fp-removed", type=int, default=3)
    parser.add_argument("--max-combo-size", type=int, default=4)
    parser.add_argument("--top-groups-per-combo", type=int, default=30)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--beam-width", type=int, default=200)
    parser.add_argument("--max-rules", type=int, default=8)
    parser.add_argument("--allow-module-strong-veto", action="store_true", help="Por padrão, vetos só pegam module_quiet. Use isto para permitir module_strong.")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-013K — Residual FP Mining with TP/FN Immunity")
    log("=" * 80)
    log(f"Input: {input_path}")
    log(f"Output: {output_dir}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    pred_col = pick_pred_col(df, args.pred_col)
    df = add_error_labels(df, pred_col)

    y = df["is_fraud"].to_numpy(dtype=int)
    base_pred = df[pred_col].to_numpy(dtype=int)
    base_metrics = compute_metrics(y, base_pred)
    total_frauds = int(y.sum())
    min_tp_required = int(math.ceil(args.target_recall * total_frauds))

    if base_metrics["tp"] < min_tp_required:
        raise RuntimeError(f"Base não cumpre recall target. TP={base_metrics['tp']} min={min_tp_required}")

    log(f"Base {pred_col}: TP={base_metrics['tp']} FP={base_metrics['fp']} FN={base_metrics['fn']} recall={base_metrics['recall']}")

    pd.DataFrame([{"pred_col": pred_col, **base_metrics}]).to_csv(output_dir / "01_base_metrics.csv", index=False)

    residual_fp_profile(df, pred_col).to_csv(output_dir / "02_residual_fp_profile.csv", index=False)
    guards = immunity_guards(df, pred_col)
    guards.to_csv(output_dir / "03_immunity_guards.csv", index=False)

    blocks = make_time_blocks(df, args.time_blocks)

    log("[1/3] Minerando candidatos residuais TP0/blocoTP0...")
    candidates = mine_candidates(
        df=df,
        pred_col=pred_col,
        blocks=blocks,
        min_fp_removed=args.min_fp_removed,
        max_combo_size=args.max_combo_size,
        top_groups_per_combo=args.top_groups_per_combo,
        require_module_quiet=not args.allow_module_strong_veto,
    )
    cand_df = candidate_dataframe(candidates)
    cand_df.to_csv(output_dir / "04_residual_fp_candidates.csv", index=False)

    log("[2/3] Beam search...")
    frontier, best, usable = search_best(
        candidates=candidates,
        base_pred=base_pred,
        y=y,
        max_candidates=args.max_candidates,
        beam_width=args.beam_width,
        max_rules=args.max_rules,
    )
    frontier.to_csv(output_dir / "05_frontier.csv", index=False)

    selected = [usable[i] for i in best.rule_indices]
    selected_df = candidate_dataframe(selected)
    selected_df.to_csv(output_dir / "06_selected_rules.csv", index=False)

    selected_pred = base_pred.copy()
    selected_pred[best.mask] = 0
    selected_metrics = compute_metrics(y, selected_pred)

    predictions = df.copy()
    predictions["exp013k_residual_fp_pred"] = selected_pred
    predictions["exp013k_base_pred"] = base_pred
    predictions["exp013k_changed_vs_base"] = (selected_pred != base_pred).astype(int)
    predictions.to_csv(output_dir / "07_selected_predictions.csv", index=False)
    predictions[(predictions["is_fraud"] == 1) & (predictions["exp013k_residual_fp_pred"] == 0)].to_csv(output_dir / "08_false_negatives.csv", index=False)
    predictions[(predictions["is_fraud"] == 0) & (predictions["exp013k_residual_fp_pred"] == 1)].to_csv(output_dir / "09_false_positives.csv", index=False)

    log("[3/3] Blocos e bootstrap...")
    block_df = pd.concat([
        block_metrics(df, base_pred, blocks, "BASE_EXP013J_STRICT"),
        block_metrics(df, selected_pred, blocks, "EXP013K_RESIDUAL_FP_MINED"),
    ], ignore_index=True)
    block_df.to_csv(output_dir / "10_time_block_metrics.csv", index=False)

    boot_input = predictions.copy()
    boot_df = bootstrap_eval(boot_input, "exp013k_residual_fp_pred", args.bootstrap_iters, args.seed, args.target_recall)
    boot_df.to_csv(output_dir / "11_bootstrap_confidence_intervals.csv", index=False)

    fp_removed_vs_base = int(base_metrics["fp"] - selected_metrics["fp"])
    tp_loss_vs_base = int(base_metrics["tp"] - selected_metrics["tp"])

    objective_status = "TARGET_RECALL_MET" if selected_metrics["tp"] >= min_tp_required and selected_metrics["recall"] >= args.target_recall else "TARGET_RECALL_NOT_MET"
    objective_status += "_TPLOSS0" if tp_loss_vs_base == 0 else "_TPLOSS_GT0"
    objective_status += "_FP_REDUCED" if fp_removed_vs_base > 0 else "_FP_NOT_REDUCED"

    policy_artifact = {
        "experiment": "EXP-013K",
        "policy_name": "residual_fp_mined_tp0_policy",
        "objective_status": objective_status,
        "base_pred_col": pred_col,
        "target_recall": args.target_recall,
        "min_tp_required": min_tp_required,
        "base_metrics": base_metrics,
        "selected_metrics": selected_metrics,
        "fp_removed_vs_base": fp_removed_vs_base,
        "tp_loss_vs_base": tp_loss_vs_base,
        "immunity_guards": guards.to_dict(orient="records"),
        "selected_rules": selected_df.to_dict(orient="records") if not selected_df.empty else [],
        "notes": [
            "Residual FP mining after EXP-013J STRICT_RECALL95_SAFE_ONLY.",
            "Hard constraints: global TP_loss=0 and block TP_loss=0 for candidate rules.",
            "Default mining excludes strong SE/BEH/runtime/module preserve cases.",
            "Validate frozen before production/shadow patch."
        ],
    }
    dump_json(policy_artifact, output_dir / "12_policy_artifact.json")

    summary = {
        "experiment": "EXP-013K",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "pred_col": pred_col,
        "n_rows": int(len(df)),
        "total_frauds": total_frauds,
        "target_recall": args.target_recall,
        "min_tp_required": min_tp_required,
        "base_metrics": base_metrics,
        "selected_metrics": selected_metrics,
        "fp_removed_vs_base": fp_removed_vs_base,
        "tp_loss_vs_base": tp_loss_vs_base,
        "n_candidates": int(len(candidates)),
        "n_selected_rules": int(len(selected)),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, selected_df, frontier, block_df, boot_df)
    (output_dir / "13_exp013k_report.md").write_text(report, encoding="utf-8")

    log("")
    log("=" * 80)
    log("EXP-013K CONCLUÍDO")
    log("=" * 80)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "04_residual_fp_candidates.csv",
        output_dir / "05_frontier.csv",
        output_dir / "06_selected_rules.csv",
        output_dir / "10_time_block_metrics.csv",
        output_dir / "11_bootstrap_confidence_intervals.csv",
        output_dir / "12_policy_artifact.json",
        output_dir / "13_exp013k_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
