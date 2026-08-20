#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-013F — Recall vs False Positive Benchmark

Objetivo:
  Comparar três cenários de recall mínimo e medir quanto FP conseguimos reduzir
  partindo da política refinada do EXP-013E.

Pergunta:
  Se relaxarmos o recall aceito de >=95% para >=92.5% ou >=90%,
  quantos falsos positivos conseguimos reduzir?

Cenários:
  1. HIGH_RECALL_95: recall >= 95.0%
  2. MID_RECALL_92_5: recall >= 92.5%
  3. LOW_RECALL_90: recall >= 90.0%

Base:
  EXP-013E:
    TP=119, FP=712, FN=5, recall=95.97%

Entrada default:
  resultados/experimentos/EXP-013E/06_selected_predictions.csv

Uso:
  python scripts/exp_013f_recall_fp_tradeoff_benchmark.py

Execução mais profunda:
  python scripts/exp_013f_recall_fp_tradeoff_benchmark.py --max-candidates 600 --beam-width 120 --max-depth 8

Saídas:
  resultados/experimentos/EXP-013F/
    00_run_summary.json
    01_base_metrics.csv
    02_veto_candidates.csv
    03_benchmark_scenarios.csv
    04_scenario_rules.csv
    05_scenario_predictions.csv
    06_false_negatives_by_scenario.csv
    07_false_positives_by_scenario.csv
    08_recall_fp_tradeoff_report.md
    09_policy_artifacts_by_scenario.json
"""

from __future__ import annotations

import argparse
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

DEFAULT_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013E" / "06_selected_predictions.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013F"

FLAGGED_DECISIONS = {"CONFIRMAR", "BLOQUEAR"}


@dataclass
class VetoRule:
    rule_id: str
    family: str
    description: str
    mask: np.ndarray
    tp_loss: int
    fp_removed: int
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

    if "exp013e_refined_pred" not in df.columns:
        # Fallback para nome eventual de política refinada.
        for c in ["exp013e_pred", "exp013d_pred", "shadow_exp012d_flagged", "exp012d_pred"]:
            if c in df.columns:
                df["exp013e_refined_pred"] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
                break

    if "exp013e_refined_pred" not in df.columns:
        raise RuntimeError("Não encontrei exp013e_refined_pred no input.")

    df["exp013e_refined_pred"] = pd.to_numeric(df["exp013e_refined_pred"], errors="coerce").fillna(0).astype(int)

    if "shadow_exp012d_flagged" not in df.columns:
        for c in ["exp012d_pred", "r4_pred", "lgbm_r4_pred"]:
            if c in df.columns:
                df["shadow_exp012d_flagged"] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
                break

    if "runtime_flagged" not in df.columns:
        if "decisao" in df.columns:
            df["runtime_flagged"] = df["decisao"].astype(str).str.upper().isin(FLAGGED_DECISIONS).astype(int)
        else:
            df["runtime_flagged"] = 0
    df["runtime_flagged"] = pd.to_numeric(df["runtime_flagged"], errors="coerce").fillna(0).astype(int)

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


def text(df: pd.DataFrame, names: str | list[str], default: str = "<MISSING>") -> pd.Series:
    col = pick_col(df, names)
    if col is None:
        return pd.Series([default] * len(df), index=df.index, dtype="string")
    return df[col].astype("string").fillna(default).astype(str)


def boolish(df: pd.DataFrame, names: str | list[str], default: bool = False) -> pd.Series:
    col = pick_col(df, names)
    if col is None:
        return pd.Series([default] * len(df), index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(default)
    return s.astype(str).str.upper().isin({"1", "1.0", "TRUE", "T", "SIM", "YES", "Y"})


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


def strong_preserve_mask(df: pd.DataFrame) -> np.ndarray:
    se_score = num(df, ["se_score_x", "se_score_y", "se_score"], 0.0)
    se_count = num(df, ["se_patterns_count", "se_pattern_count"], 0.0)
    beh_score = num(df, ["beh_score", "behavioral_score"], 0.0)
    beh_count = num(df, ["beh_factors_count", "behavioral_risk_factor_count"], 0.0)
    runtime = num(df, "runtime_flagged", 0.0)
    cascade = boolish(df, "cascade_triggered", False)
    decisao = text(df, "decisao", "").str.upper()

    return (
        (se_score >= 65)
        | (se_count >= 2)
        | (beh_score >= 45)
        | (beh_count >= 2)
        | (runtime >= 1)
        | decisao.isin(FLAGGED_DECISIONS)
        | cascade
    ).to_numpy(dtype=bool)


def sanitize_id(s: str, max_len: int = 120) -> str:
    t = re.sub(r"[^A-Za-z0-9_]+", "_", str(s))
    t = re.sub(r"_+", "_", t).strip("_")
    return t[:max_len] or "rule"


def add_rule(rules: list[VetoRule], df: pd.DataFrame, y: np.ndarray, base_pred: np.ndarray, family: str, description: str, mask: np.ndarray, params: dict[str, Any], min_fp_removed: int) -> None:
    mask = np.asarray(mask, dtype=bool) & (base_pred == 1)
    if not mask.any():
        return

    tp_loss = int(((y == 1) & mask).sum())
    fp_removed = int(((y == 0) & mask).sum())

    if fp_removed < min_fp_removed:
        return

    rid = sanitize_id(f"{family}_{len(rules):04d}_{description}")
    rules.append(VetoRule(
        rule_id=rid,
        family=family,
        description=description,
        mask=mask,
        tp_loss=tp_loss,
        fp_removed=fp_removed,
        params=params,
    ))


def dedupe_rules(rules: list[VetoRule]) -> list[VetoRule]:
    best: dict[bytes, VetoRule] = {}
    for r in rules:
        key = np.packbits(r.mask).tobytes()
        old = best.get(key)
        if old is None:
            best[key] = r
        else:
            new_key = (r.fp_removed, -r.tp_loss, -len(r.description))
            old_key = (old.fp_removed, -old.tp_loss, -len(old.description))
            if new_key > old_key:
                best[key] = r

    out = list(best.values())
    out.sort(key=lambda r: (r.tp_loss, -r.fp_removed, -r.fp_removed / max(r.tp_loss, 1)))
    return out


def generate_veto_candidates(df: pd.DataFrame, base_pred: np.ndarray, min_fp_removed: int) -> list[VetoRule]:
    y = df["is_fraud"].to_numpy(dtype=int)
    preserve = strong_preserve_mask(df)

    lgbm = num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0)
    score_final = num(df, "score_final", np.nan)
    ifp = num(df, ["if_percentile_x", "if_percentile_y", "if_percentile"], 0.0)
    se_score = num(df, ["se_score_x", "se_score_y", "se_score"], 0.0)
    se_count = num(df, ["se_patterns_count", "se_pattern_count"], 0.0)
    beh_score = num(df, ["beh_score", "behavioral_score"], 0.0)
    beh_count = num(df, ["beh_factors_count", "behavioral_risk_factor_count"], 0.0)

    vl = num(df, "vl_pix", 0.0)
    ratio = num(df, "ratio_valor_media_pagador_90d", 0.0)
    ratio_max = num(df, "ratio_valor_maximo_pagador_180d", 0.0)
    qtd_rec_180 = num(df, "qtd_pix_recebidos_180d", 0.0)
    qtd_rec_90 = num(df, "qtd_pix_recebidos_90d", 0.0)
    valor_rec_180 = num(df, "valor_total_recebido_180d", 0.0)
    pagadores_dist = num(df, "soma_pagadores_distintos_dia_recebedor_180d", 0.0)

    rules: list[VetoRule] = []

    log("[1/4] Gerando vetos numéricos e compostos...")

    # Low-confidence score vetos.
    for th in [0.0019429789, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10]:
        add_rule(rules, df, y, base_pred, "lgbm_threshold", f"lgbm_r4_score<{th}", (lgbm < th).to_numpy(dtype=bool), {"feature": "lgbm_r4_score", "op": "lt", "threshold": th}, min_fp_removed)
        add_rule(rules, df, y, base_pred, "lgbm_threshold_preserve", f"lgbm_r4_score<{th} AND NOT strong_preserve", (lgbm < th).to_numpy(dtype=bool) & (~preserve), {"feature": "lgbm_r4_score", "op": "lt", "threshold": th, "preserve": True}, min_fp_removed)

    for th in [0.50, 0.76, 1.0, 2.0, 3.0, 4.0, 5.0]:
        add_rule(rules, df, y, base_pred, "score_final_threshold", f"score_final<{th}", (score_final < th).to_numpy(dtype=bool), {"feature": "score_final", "op": "lt", "threshold": th}, min_fp_removed)

    # IF/LGBM combined.
    for lth in [0.01, 0.02, 0.03, 0.05, 0.10]:
        for ifth in [0.32, 0.50, 0.70, 0.85, 0.95]:
            mask = ((lgbm < lth) & (ifp < ifth)).to_numpy(dtype=bool) & (~preserve)
            add_rule(rules, df, y, base_pred, "if_lgbm_veto", f"lgbm<{lth} AND if<{ifth} AND NOT strong_preserve", mask, {"lgbm_lt": lth, "if_lt": ifth}, min_fp_removed)

    # Quiet veto.
    for lth in [0.02, 0.03, 0.05, 0.10]:
        for ifth in [0.70, 0.85, 0.95]:
            mask = (
                (lgbm < lth)
                & (ifp < ifth)
                & (se_score <= 20)
                & (se_count < 2)
                & (beh_score <= 25)
                & (beh_count < 2)
            ).to_numpy(dtype=bool) & (~preserve)
            add_rule(rules, df, y, base_pred, "quiet_veto", f"lgbm<{lth} IF<{ifth} SE/BEH quiet", mask, {"lgbm_lt": lth, "if_lt": ifth}, min_fp_removed)

    # Low value/ratio.
    for lth in [0.02, 0.05, 0.10]:
        for vlth in [20, 50, 100, 250, 500, 1000]:
            mask = ((lgbm < lth) & (vl < vlth)).to_numpy(dtype=bool) & (~preserve)
            add_rule(rules, df, y, base_pred, "low_value_veto", f"lgbm<{lth} AND vl_pix<{vlth}", mask, {"lgbm_lt": lth, "vl_lt": vlth}, min_fp_removed)

        for rth in [0.068208507, 0.10726481, 0.19765786, 0.5, 1.0]:
            mask = ((lgbm < lth) & (ratio < rth)).to_numpy(dtype=bool) & (~preserve)
            add_rule(rules, df, y, base_pred, "low_ratio_veto", f"lgbm<{lth} AND ratio_valor_media_pagador_90d<{rth}", mask, {"lgbm_lt": lth, "ratio_lt": rth}, min_fp_removed)

    # Receiver-established vetos.
    for lth in [0.01, 0.02, 0.05, 0.10]:
        for qth in [5, 10, 20, 50, 100]:
            mask = ((lgbm < lth) & ((qtd_rec_180 > qth) | (qtd_rec_90 > qth))).to_numpy(dtype=bool) & (~preserve)
            add_rule(rules, df, y, base_pred, "receiver_history_veto", f"lgbm<{lth} AND receiver_qtd>{qth}", mask, {"lgbm_lt": lth, "receiver_qtd_gt": qth}, min_fp_removed)

        for vth in [500, 1000, 2000, 5000, 10000, 25000]:
            mask = ((lgbm < lth) & (valor_rec_180 > vth)).to_numpy(dtype=bool) & (~preserve)
            add_rule(rules, df, y, base_pred, "receiver_value_veto", f"lgbm<{lth} AND receiver_value_180d>{vth}", mask, {"lgbm_lt": lth, "receiver_value_gt": vth}, min_fp_removed)

        for pth in [2, 5, 10, 20, 50]:
            mask = ((lgbm < lth) & (pagadores_dist > pth)).to_numpy(dtype=bool) & (~preserve)
            add_rule(rules, df, y, base_pred, "receiver_many_payers_veto", f"lgbm<{lth} AND payers_distinct>{pth}", mask, {"lgbm_lt": lth, "payers_gt": pth}, min_fp_removed)

    log("[2/4] Gerando vetos segmentados...")
    segment_sets = [
        ["value_band"],
        ["ds_tipo_chave_norm"],
        ["periodo_dia"],
        ["first_receiver_flag_real"],
        ["mbk_available_flag"],
        ["value_band", "ds_tipo_chave_norm"],
        ["periodo_dia", "value_band"],
        ["first_receiver_flag_real", "value_band"],
        ["first_receiver_flag_real", "ds_tipo_chave_norm"],
        ["mbk_available_flag", "ds_tipo_chave_norm"],
        ["value_band", "ds_tipo_chave_norm", "periodo_dia"],
    ]

    current_pos = base_pred == 1
    for cols in segment_sets:
        if any(c not in df.columns for c in cols):
            continue

        key_df = pd.DataFrame(index=df.index)
        for c in cols:
            key_df[c] = text(df, c)

        grouped = key_df[current_pos].groupby(cols, dropna=False).indices
        for key, idxs_rel in grouped.items():
            idxs = key_df[current_pos].iloc[list(idxs_rel)].index.to_numpy(dtype=int)
            if len(idxs) < min_fp_removed:
                continue

            mask = np.zeros(len(df), dtype=bool)
            mask[idxs] = True
            fp = int(((y == 0) & mask).sum())
            tp = int(((y == 1) & mask).sum())
            if fp < min_fp_removed:
                continue

            key_tuple = key if isinstance(key, tuple) else (key,)
            desc = " AND ".join([f"{c}={v}" for c, v in zip(cols, key_tuple)])
            add_rule(rules, df, y, base_pred, "segment_veto", desc, mask & (~preserve), {"segment_cols": cols, "segment_values": [str(v) for v in key_tuple], "preserve": True}, min_fp_removed)

            # Segment + LGBM low for wider recall tradeoff.
            for lth in [0.02, 0.05, 0.10]:
                add_rule(
                    rules, df, y, base_pred,
                    "segment_lgbm_veto",
                    f"{desc} AND lgbm<{lth}",
                    mask & (lgbm < lth).to_numpy(dtype=bool) & (~preserve),
                    {"segment_cols": cols, "segment_values": [str(v) for v in key_tuple], "lgbm_lt": lth, "preserve": True},
                    min_fp_removed,
                )

    rules = dedupe_rules(rules)
    log(f"    candidatos após dedupe: {len(rules)}")
    return rules


def rules_df(rules: list[VetoRule]) -> pd.DataFrame:
    return pd.DataFrame([{
        "rule_index": i,
        "rule_id": r.rule_id,
        "family": r.family,
        "description": r.description,
        "tp_loss": r.tp_loss,
        "fp_removed": r.fp_removed,
        "fp_per_tp_loss": r.fp_removed / max(r.tp_loss, 1),
        "params_json": json.dumps(r.params, ensure_ascii=False),
    } for i, r in enumerate(rules)])


def search_for_target(
    rules: list[VetoRule],
    base_pred: np.ndarray,
    y: np.ndarray,
    target_recall: float,
    max_candidates: int,
    beam_width: int,
    max_depth: int,
) -> tuple[pd.DataFrame, State]:
    total_frauds = int(y.sum())
    base_metrics = compute_metrics(y, base_pred)
    min_tp_required = int(math.ceil(target_recall * total_frauds))
    max_tp_loss = max(0, base_metrics["tp"] - min_tp_required)

    candidates = [r for r in rules if r.tp_loss <= max_tp_loss]
    candidates.sort(key=lambda r: (r.fp_removed / max(r.tp_loss, 1), r.fp_removed, -r.tp_loss), reverse=True)
    candidates = candidates[:max_candidates]

    log(f"    target={target_recall:.3f}: min_tp={min_tp_required}, max_tp_loss={max_tp_loss}, candidates={len(candidates)}")

    zero = np.zeros(len(y), dtype=bool)
    base_state = State(mask=zero, rule_indices=tuple(), tp_loss=0, fp_removed=0)
    states = [base_state]
    best = base_state
    rows = []

    for depth in range(1, max_depth + 1):
        next_states: dict[bytes, State] = {}

        for state in states:
            last = state.rule_indices[-1] if state.rule_indices else -1
            for i in range(last + 1, len(candidates)):
                r = candidates[i]
                new_mask = state.mask | r.mask
                if np.array_equal(new_mask, state.mask):
                    continue

                tp_loss = int(((y == 1) & new_mask).sum())
                if tp_loss > max_tp_loss:
                    continue

                fp_removed = int(((y == 0) & new_mask).sum())
                if fp_removed <= state.fp_removed:
                    continue

                key = np.packbits(new_mask).tobytes()
                ns = State(mask=new_mask, rule_indices=state.rule_indices + (i,), tp_loss=tp_loss, fp_removed=fp_removed)
                old = next_states.get(key)
                if old is None or (ns.fp_removed, -ns.tp_loss, -len(ns.rule_indices)) > (old.fp_removed, -old.tp_loss, -len(old.rule_indices)):
                    next_states[key] = ns

        if not next_states:
            break

        states = sorted(next_states.values(), key=lambda s: (s.fp_removed, -s.tp_loss, -len(s.rule_indices)), reverse=True)[:beam_width]

        if (states[0].fp_removed, -states[0].tp_loss) > (best.fp_removed, -best.tp_loss):
            best = states[0]

        for s in states[:50]:
            pred = base_pred.copy()
            pred[s.mask] = 0
            m = compute_metrics(y, pred)
            rows.append({
                "target_recall": target_recall,
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
                "rule_ids": "|".join(candidates[i].rule_id for i in s.rule_indices),
                "rule_descriptions": " || ".join(candidates[i].description for i in s.rule_indices),
            })

    # Convert rule_indices back to rule_ids by storing candidate descriptions in rows;
    # best keeps candidate-relative indices and is interpreted by helper.
    frontier = pd.DataFrame(rows)
    if frontier.empty:
        m = compute_metrics(y, base_pred)
        frontier = pd.DataFrame([{
            "target_recall": target_recall,
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
        }])

    # Attach candidate-relative best; caller receives candidates via hidden in rule_ids? Easier:
    # return best plus a frontier row containing selected rule_ids. Reconstruct using candidates.
    best.rule_indices = tuple(rules.index(candidates[i]) for i in best.rule_indices)  # type: ignore
    return frontier, best


def bootstrap_metrics(df: pd.DataFrame, pred_col: str, iters: int, seed: int, target_recall: float) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(df)
    rows = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        y = df.iloc[idx]["is_fraud"].to_numpy(dtype=int)
        pred = df.iloc[idx][pred_col].to_numpy(dtype=int)
        rows.append(compute_metrics(y, pred))
    boot = pd.DataFrame(rows)
    return {
        "recall_mean": float(boot["recall"].mean()),
        "recall_p025": float(boot["recall"].quantile(0.025)),
        "recall_p050": float(boot["recall"].quantile(0.50)),
        "recall_p975": float(boot["recall"].quantile(0.975)),
        "prob_recall_below_target": float((boot["recall"] < target_recall).mean()),
        "fp_mean": float(boot["fp"].mean()),
        "fp_p050": float(boot["fp"].quantile(0.50)),
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


def block_summary(df: pd.DataFrame, pred_col: str, n_blocks: int) -> dict[str, Any]:
    blocks = make_time_blocks(df, n_blocks)
    recalls = []
    fps = []
    for b in sorted(blocks.dropna().unique()):
        part = df.loc[blocks == b]
        pred = part[pred_col].to_numpy(dtype=int)
        m = compute_metrics(part["is_fraud"].to_numpy(dtype=int), pred)
        recalls.append(m["recall"])
        fps.append(m["fp"])
    return {
        "min_block_recall": float(min(recalls)) if recalls else None,
        "max_block_fp": int(max(fps)) if fps else None,
    }


def make_report(summary: dict[str, Any], scenarios: pd.DataFrame, rule_df: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-013F — Recall vs False Positive Benchmark")
    lines.append("")
    lines.append("## Objetivo")
    lines.append("Comparar três metas mínimas de recall: >=95%, >=92.5% e >=90%, medindo o menor FP encontrado em cada cenário.")
    lines.append("")
    lines.append("## Resultado dos cenários")
    show_cols = [
        "scenario", "target_recall", "tp", "fp", "fn", "precision", "recall",
        "fp_reduction_vs_exp013e", "fp_reduction_pct_vs_exp013e",
        "tp_loss_vs_exp013e", "min_block_recall", "bootstrap_prob_recall_below_target",
    ]
    lines.append(scenarios[show_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Regras selecionadas por cenário")
    if rule_df.empty:
        lines.append("Nenhuma regra adicional selecionada.")
    else:
        lines.append(rule_df[["scenario", "family", "description", "tp_loss", "fp_removed"]].to_markdown(index=False))
    lines.append("")
    lines.append("## Leitura")
    lines.append("Use este benchmark para decidir se a redução adicional de FP compensa a perda de recall. A seleção final ainda deve considerar risco temporal/bootstrap, não apenas FP global.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-candidates", type=int, default=400)
    parser.add_argument("--beam-width", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--min-fp-removed", type=int, default=15)
    parser.add_argument("--bootstrap-iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-blocks", type=int, default=5)
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-013F — Recall vs False Positive Benchmark")
    log("=" * 80)
    log(f"Input: {input_path}")
    log(f"Output: {output_dir}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    y = df["is_fraud"].to_numpy(dtype=int)
    base_pred = df["exp013e_refined_pred"].to_numpy(dtype=int)
    base_metrics = compute_metrics(y, base_pred)

    log(f"EXP-013E base: TP={base_metrics['tp']} FP={base_metrics['fp']} FN={base_metrics['fn']} recall={base_metrics['recall']}")
    pd.DataFrame([{"policy": "EXP013E_REFINED_BASE", **base_metrics}]).to_csv(output_dir / "01_base_metrics.csv", index=False)

    rules = generate_veto_candidates(df, base_pred, args.min_fp_removed)
    rdf = rules_df(rules)
    rdf.to_csv(output_dir / "02_veto_candidates.csv", index=False)

    scenarios = [
        ("HIGH_RECALL_95", 0.95),
        ("MID_RECALL_92_5", 0.925),
        ("LOW_RECALL_90", 0.90),
    ]

    all_frontiers = []
    scenario_rows = []
    selected_rule_rows = []
    predictions = df.copy()
    policy_artifacts = {}

    log("[3/4] Buscando cenários...")
    for scenario_name, target in scenarios:
        frontier, best = search_for_target(
            rules=rules,
            base_pred=base_pred,
            y=y,
            target_recall=target,
            max_candidates=args.max_candidates,
            beam_width=args.beam_width,
            max_depth=args.max_depth,
        )
        frontier["scenario"] = scenario_name
        all_frontiers.append(frontier)

        pred = base_pred.copy()
        pred[best.mask] = 0
        col = f"pred_{scenario_name}"
        predictions[col] = pred

        m = compute_metrics(y, pred)
        boot = bootstrap_metrics(predictions, col, args.bootstrap_iters, args.seed, target)
        block = block_summary(predictions, col, args.time_blocks)

        selected = [rules[i] for i in best.rule_indices]
        for r in selected:
            selected_rule_rows.append({
                "scenario": scenario_name,
                "target_recall": target,
                "rule_id": r.rule_id,
                "family": r.family,
                "description": r.description,
                "tp_loss": r.tp_loss,
                "fp_removed": r.fp_removed,
                "params_json": json.dumps(r.params, ensure_ascii=False),
            })

        fp_reduction = base_metrics["fp"] - m["fp"]
        tp_loss = base_metrics["tp"] - m["tp"]

        row = {
            "scenario": scenario_name,
            "target_recall": target,
            **m,
            "tp_loss_vs_exp013e": int(tp_loss),
            "fp_reduction_vs_exp013e": int(fp_reduction),
            "fp_reduction_pct_vs_exp013e": round(float(fp_reduction / max(base_metrics["fp"], 1)), 6),
            "precision_lift_vs_exp013e": round(float(m["precision"] / max(base_metrics["precision"], 1e-9)), 6),
            "n_rules": len(selected),
            "rule_ids": "|".join(r.rule_id for r in selected),
            **block,
            "bootstrap_recall_mean": boot["recall_mean"],
            "bootstrap_recall_p025": boot["recall_p025"],
            "bootstrap_recall_p050": boot["recall_p050"],
            "bootstrap_recall_p975": boot["recall_p975"],
            "bootstrap_prob_recall_below_target": boot["prob_recall_below_target"],
            "bootstrap_fp_mean": boot["fp_mean"],
            "bootstrap_fp_p050": boot["fp_p050"],
        }
        scenario_rows.append(row)

        policy_artifacts[scenario_name] = {
            "scenario": scenario_name,
            "target_recall": target,
            "metrics": m,
            "rules": [{
                "rule_id": r.rule_id,
                "family": r.family,
                "description": r.description,
                "tp_loss": r.tp_loss,
                "fp_removed": r.fp_removed,
                "params": r.params,
            } for r in selected],
            "base_policy": "EXP-013E conservative_refined_local_policy",
        }

        log(f"    {scenario_name}: TP={m['tp']} FP={m['fp']} FN={m['fn']} recall={m['recall']} precision={m['precision']} rules={len(selected)}")

    frontier_df = pd.concat(all_frontiers, ignore_index=True) if all_frontiers else pd.DataFrame()
    frontier_df.to_csv(output_dir / "04_frontiers_by_scenario.csv", index=False)

    scenarios_df = pd.DataFrame(scenario_rows)
    scenarios_df.to_csv(output_dir / "03_benchmark_scenarios.csv", index=False)

    selected_rules_df = pd.DataFrame(selected_rule_rows)
    selected_rules_df.to_csv(output_dir / "04_scenario_rules.csv", index=False)

    predictions.to_csv(output_dir / "05_scenario_predictions.csv", index=False)

    # Error files.
    fn_rows = []
    fp_rows = []
    for scenario_name, _ in scenarios:
        col = f"pred_{scenario_name}"
        fn_part = predictions[(predictions["is_fraud"] == 1) & (predictions[col] == 0)].copy()
        fp_part = predictions[(predictions["is_fraud"] == 0) & (predictions[col] == 1)].copy()
        fn_part["scenario"] = scenario_name
        fp_part["scenario"] = scenario_name
        fn_rows.append(fn_part)
        fp_rows.append(fp_part)

    pd.concat(fn_rows, ignore_index=True).to_csv(output_dir / "06_false_negatives_by_scenario.csv", index=False)
    pd.concat(fp_rows, ignore_index=True).to_csv(output_dir / "07_false_positives_by_scenario.csv", index=False)

    dump_json(policy_artifacts, output_dir / "09_policy_artifacts_by_scenario.json")

    # Simple recommendation.
    low = scenarios_df[scenarios_df["scenario"] == "LOW_RECALL_90"].iloc[0].to_dict()
    high = scenarios_df[scenarios_df["scenario"] == "HIGH_RECALL_95"].iloc[0].to_dict()
    mid = scenarios_df[scenarios_df["scenario"] == "MID_RECALL_92_5"].iloc[0].to_dict()

    summary = {
        "experiment": "EXP-013F",
        "status": "DONE",
        "question": "Does relaxing recall from >=95% to >=92.5% or >=90% reduce FP enough to justify promotion?",
        "input_path": str(input_path),
        "n_rows": int(len(df)),
        "total_frauds": int(y.sum()),
        "base_exp013e_metrics": base_metrics,
        "scenarios": scenarios_df.to_dict(orient="records"),
        "comparison": {
            "high_95_fp": int(high["fp"]),
            "mid_92_5_fp": int(mid["fp"]),
            "low_90_fp": int(low["fp"]),
            "low_vs_high_fp_reduction": int(high["fp"] - low["fp"]),
            "low_vs_high_fp_reduction_pct": float((high["fp"] - low["fp"]) / max(high["fp"], 1)),
            "low_vs_exp013e_fp_reduction_pct": float((base_metrics["fp"] - low["fp"]) / max(base_metrics["fp"], 1)),
        },
        "decision_hint": "Evaluate whether LOW_RECALL_90 achieved >=50% FP reduction vs EXP-013E while keeping recall>=90%. Also inspect bootstrap and block recall.",
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, scenarios_df, selected_rules_df)
    (output_dir / "08_recall_fp_tradeoff_report.md").write_text(report, encoding="utf-8")

    log("")
    log("=" * 80)
    log("EXP-013F CONCLUÍDO")
    log("=" * 80)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "03_benchmark_scenarios.csv",
        output_dir / "04_scenario_rules.csv",
        output_dir / "05_scenario_predictions.csv",
        output_dir / "08_recall_fp_tradeoff_report.md",
        output_dir / "09_policy_artifacts_by_scenario.json",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
