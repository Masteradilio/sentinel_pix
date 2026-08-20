#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-013G — High Recall 95 Micro-Refinement

Objetivo:
  Consolidar o cenário HIGH_RECALL_95 do EXP-013F como novo benchmark vencedor
  e tentar microevoluções SEM deixar recall cair abaixo de 95%.

Benchmark base:
  HIGH_RECALL_95:
    TP=118
    FP=494
    FN=6
    recall=95.16%

Regra de ouro:
  Não aceitar nenhum candidato com recall < 95%.
  Como o dataset tem 124 fraudes, recall >=95% exige TP >=118.
  O HIGH_RECALL_95 já está exatamente no piso de TP=118, então:
    - vetos adicionais precisam ter TP loss = 0, OU
    - uma restauração precisa recuperar TP antes de permitir algum veto com TP loss.

Estratégia:
  1. Carrega resultados do EXP-013F/05_scenario_predictions.csv.
  2. Usa pred_HIGH_RECALL_95 como política base.
  3. Gera candidatos de restauração sobre casos removidos pelo HIGH_RECALL_95.
  4. Gera candidatos de veto sobre FPs remanescentes do HIGH_RECALL_95.
  5. Faz busca local pequena, exigindo TP >= 118 e recall >= 95%.
  6. Mede robustez por blocos temporais e bootstrap.

Entradas default:
  resultados/experimentos/EXP-013F/05_scenario_predictions.csv
  resultados/experimentos/EXP-013F/09_policy_artifacts_by_scenario.json

Uso:
  python scripts/exp_013g_high_recall95_micro_refinement.py

Uso mais rápido:
  python scripts/exp_013g_high_recall95_micro_refinement.py --bootstrap-iters 200

Uso mais profundo:
  python scripts/exp_013g_high_recall95_micro_refinement.py --veto-beam 150 --restore-beam 80 --max-veto-rules 6

Saídas:
  resultados/experimentos/EXP-013G/
    00_run_summary.json
    01_base_metrics.csv
    02_restore_candidates.csv
    03_veto_candidates.csv
    04_micro_frontier.csv
    05_selected_micro_rules.csv
    06_selected_predictions.csv
    07_selected_false_negatives.csv
    08_selected_false_positives.csv
    09_time_block_metrics.csv
    10_bootstrap_confidence_intervals.csv
    11_micro_refinement_report.md
    12_policy_artifact.json
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

DEFAULT_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013F" / "05_scenario_predictions.csv"
DEFAULT_POLICY = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013F" / "09_policy_artifacts_by_scenario.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013G"

FLAGGED_DECISIONS = {"CONFIRMAR", "BLOQUEAR"}


@dataclass
class Action:
    action_id: str
    action_type: str  # restore or veto
    family: str
    description: str
    mask: np.ndarray
    tp_delta: int
    fp_delta: int
    params: dict[str, Any]


@dataclass
class State:
    pred: np.ndarray
    restore_ids: tuple[int, ...]
    veto_ids: tuple[int, ...]
    metrics: dict[str, Any]


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

    if "pred_HIGH_RECALL_95" not in df.columns:
        raise RuntimeError("Coluna pred_HIGH_RECALL_95 ausente. Use resultados/experimentos/EXP-013F/05_scenario_predictions.csv.")

    df["pred_HIGH_RECALL_95"] = pd.to_numeric(df["pred_HIGH_RECALL_95"], errors="coerce").fillna(0).astype(int)

    if "exp013e_refined_pred" in df.columns:
        df["exp013e_refined_pred"] = pd.to_numeric(df["exp013e_refined_pred"], errors="coerce").fillna(0).astype(int)

    if "shadow_exp012d_flagged" not in df.columns:
        for c in ["exp012d_pred", "r4_pred", "lgbm_r4_pred"]:
            if c in df.columns:
                df["shadow_exp012d_flagged"] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
                break
    if "shadow_exp012d_flagged" not in df.columns:
        df["shadow_exp012d_flagged"] = df["pred_HIGH_RECALL_95"]

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
    return t[:max_len] or "action"


def add_action(
    actions: list[Action],
    action_type: str,
    family: str,
    description: str,
    mask: np.ndarray,
    base_pred: np.ndarray,
    y: np.ndarray,
    min_fp_effect: int,
    max_fp_restore_per_tp: float,
) -> None:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return

    if action_type == "restore":
        effective = mask & (base_pred == 0)
        tp_delta = int(((y == 1) & effective).sum())
        fp_delta = int(((y == 0) & effective).sum())
        if tp_delta <= 0:
            return
        if fp_delta / max(tp_delta, 1) > max_fp_restore_per_tp:
            return
    elif action_type == "veto":
        effective = mask & (base_pred == 1)
        tp_delta = -int(((y == 1) & effective).sum())
        fp_delta = -int(((y == 0) & effective).sum())
        if abs(fp_delta) < min_fp_effect:
            return
    else:
        raise ValueError(action_type)

    if not effective.any():
        return

    aid = sanitize_id(f"{action_type}_{family}_{len(actions):04d}_{description}")
    actions.append(Action(
        action_id=aid,
        action_type=action_type,
        family=family,
        description=description,
        mask=effective,
        tp_delta=tp_delta,
        fp_delta=fp_delta,
        params={},
    ))


def dedupe_actions(actions: list[Action]) -> list[Action]:
    best: dict[tuple[bytes, str], Action] = {}
    for a in actions:
        key = (np.packbits(a.mask).tobytes(), a.action_type)
        old = best.get(key)
        if old is None:
            best[key] = a
        else:
            if a.action_type == "restore":
                new_key = (a.tp_delta, -a.fp_delta, -len(a.description))
                old_key = (old.tp_delta, -old.fp_delta, -len(old.description))
            else:
                new_key = (-abs(a.tp_delta), abs(a.fp_delta), -len(a.description))
                old_key = (-abs(old.tp_delta), abs(old.fp_delta), -len(old.description))
            if new_key > old_key:
                best[key] = a
    return list(best.values())


def action_df(actions: list[Action]) -> pd.DataFrame:
    return pd.DataFrame([{
        "action_index": i,
        "action_id": a.action_id,
        "action_type": a.action_type,
        "family": a.family,
        "description": a.description,
        "tp_delta": a.tp_delta,
        "fp_delta": a.fp_delta,
        "n_affected": int(a.mask.sum()),
    } for i, a in enumerate(actions)])


def generate_restore_candidates(df: pd.DataFrame, base_pred: np.ndarray, y: np.ndarray, max_fp_restore_per_tp: float) -> list[Action]:
    actions: list[Action] = []

    shadow = df["shadow_exp012d_flagged"].to_numpy(dtype=int).astype(bool)
    exp013e = df["exp013e_refined_pred"].to_numpy(dtype=int).astype(bool) if "exp013e_refined_pred" in df.columns else shadow
    vetoed = ((shadow | exp013e) & (base_pred == 0))
    current_fns = (y == 1) & (base_pred == 0)

    if not current_fns.any():
        return actions

    lgbm = num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0)
    ifp = num(df, ["if_percentile_x", "if_percentile_y", "if_percentile"], 0.0)
    vl = num(df, "vl_pix", 0.0)
    se_score = num(df, ["se_score_x", "se_score_y", "se_score"], 0.0)
    beh_score = num(df, ["beh_score", "behavioral_score"], 0.0)

    # Restore cases removed by HIGH but kept by EXP-013E, likely the 1 TP lost in HIGH.
    add_action(
        actions, "restore", "restore_exp013e_kept",
        "restore cases kept by EXP013E but removed by HIGH_RECALL_95",
        (exp013e & (base_pred == 0)),
        base_pred, y, min_fp_effect=0, max_fp_restore_per_tp=max_fp_restore_per_tp
    )

    specs = [
        ("lgbm", lgbm, [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]),
        ("if_percentile", ifp, [0.70, 0.80, 0.90, 0.95]),
        ("vl_pix", vl, [500, 1000, 2000, 5000, 10000]),
        ("se_score", se_score, [20, 40, 65]),
        ("behavioral_score", beh_score, [15, 25, 45]),
    ]

    for feat, vals, thresholds in specs:
        for th in thresholds:
            mask = vetoed & (vals >= th).to_numpy(dtype=bool)
            add_action(
                actions, "restore", "numeric_preserve",
                f"restore vetoed where {feat}>={th}",
                mask, base_pred, y, min_fp_effect=0, max_fp_restore_per_tp=max_fp_restore_per_tp
            )

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
    ]

    for cols in segment_sets:
        if any(c not in df.columns for c in cols):
            continue

        keys = pd.DataFrame(index=df.index)
        for c in cols:
            keys[c] = text(df, c)

        fn_keys = keys.loc[current_fns].drop_duplicates()
        for _, row in fn_keys.iterrows():
            mask = vetoed.copy()
            parts = []
            for c in cols:
                val = str(row[c])
                parts.append(f"{c}={val}")
                mask = mask & (keys[c] == val).to_numpy(dtype=bool)
            add_action(
                actions, "restore", "segment_preserve",
                "restore vetoed segment " + " AND ".join(parts),
                mask, base_pred, y, min_fp_effect=0, max_fp_restore_per_tp=max_fp_restore_per_tp
            )

    return dedupe_actions(actions)


def generate_veto_candidates(df: pd.DataFrame, base_pred: np.ndarray, y: np.ndarray, min_fp_removed: int, max_tp_loss_per_veto: int) -> list[Action]:
    actions: list[Action] = []
    preserve = strong_preserve_mask(df)

    lgbm = num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0)
    ifp = num(df, ["if_percentile_x", "if_percentile_y", "if_percentile"], 0.0)
    vl = num(df, "vl_pix", 0.0)
    ratio = num(df, "ratio_valor_media_pagador_90d", 0.0)
    qtd_rec = num(df, "qtd_pix_recebidos_180d", 0.0)
    qtd_rec_90 = num(df, "qtd_pix_recebidos_90d", 0.0)
    valor_rec = num(df, "valor_total_recebido_180d", 0.0)
    pagadores_dist = num(df, "soma_pagadores_distintos_dia_recebedor_180d", 0.0)
    se_score = num(df, ["se_score_x", "se_score_y", "se_score"], 0.0)
    beh_score = num(df, ["beh_score", "behavioral_score"], 0.0)

    def maybe(desc: str, family: str, mask: np.ndarray):
        effective = mask & (base_pred == 1)
        tp_loss = int(((y == 1) & effective).sum())
        if tp_loss > max_tp_loss_per_veto:
            return
        add_action(
            actions, "veto", family, desc, effective, base_pred, y,
            min_fp_effect=min_fp_removed, max_fp_restore_per_tp=9999
        )

    # Prefer zero-loss / tiny-loss additional vetos.
    for th in [0.0019429789, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05]:
        maybe(f"veto current positives where lgbm<{th} AND NOT strong_preserve", "lgbm_veto", (lgbm < th).to_numpy(dtype=bool) & (~preserve))

    for lth in [0.02, 0.03, 0.05, 0.075, 0.10]:
        for ifth in [0.50, 0.70, 0.85, 0.95]:
            maybe(f"veto current positives where lgbm<{lth} IF<{ifth} NOT strong_preserve", "if_lgbm_veto", ((lgbm < lth) & (ifp < ifth)).to_numpy(dtype=bool) & (~preserve))

        for vlth in [20, 50, 100, 250, 500, 1000]:
            maybe(f"veto current positives where lgbm<{lth} vl_pix<{vlth}", "low_value_veto", ((lgbm < lth) & (vl < vlth)).to_numpy(dtype=bool) & (~preserve))

        for rth in [0.068208507, 0.10726481, 0.19765786, 0.5, 1.0]:
            maybe(f"veto current positives where lgbm<{lth} ratio<{rth}", "low_ratio_veto", ((lgbm < lth) & (ratio < rth)).to_numpy(dtype=bool) & (~preserve))

        for qth in [5, 10, 20, 50, 100]:
            maybe(f"veto current positives where lgbm<{lth} receiver_qtd>{qth}", "receiver_history_veto", ((lgbm < lth) & ((qtd_rec > qth) | (qtd_rec_90 > qth))).to_numpy(dtype=bool) & (~preserve))

        for vth in [5000, 10000, 25000, 50000]:
            maybe(f"veto current positives where lgbm<{lth} receiver_value_180d>{vth}", "receiver_value_veto_light", ((lgbm < lth) & (valor_rec > vth)).to_numpy(dtype=bool) & (~preserve))

        for pth in [5, 10, 20, 50]:
            maybe(f"veto current positives where lgbm<{lth} payers_distinct>{pth}", "receiver_many_payers_veto", ((lgbm < lth) & (pagadores_dist > pth)).to_numpy(dtype=bool) & (~preserve))

    for lth in [0.03, 0.05, 0.075]:
        maybe(f"quiet veto current positives where lgbm<{lth} IF<0.7 SE/BEH quiet", "quiet_veto", ((lgbm < lth) & (ifp < 0.7) & (se_score <= 20) & (beh_score <= 25)).to_numpy(dtype=bool) & (~preserve))

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

        keys = pd.DataFrame(index=df.index)
        for c in cols:
            keys[c] = text(df, c)

        grouped = keys[current_pos].groupby(cols, dropna=False).indices
        for key, idxs_rel in grouped.items():
            idxs = keys[current_pos].iloc[list(idxs_rel)].index.to_numpy(dtype=int)
            if len(idxs) < min_fp_removed:
                continue

            mask = np.zeros(len(df), dtype=bool)
            mask[idxs] = True
            key_tuple = key if isinstance(key, tuple) else (key,)
            desc = " AND ".join([f"{c}={v}" for c, v in zip(cols, key_tuple)])
            maybe(f"veto current positive segment {desc}", "segment_veto", mask & (~preserve))

            for lth in [0.02, 0.05, 0.10]:
                maybe(f"veto current positive segment {desc} AND lgbm<{lth}", "segment_lgbm_veto", mask & (lgbm < lth).to_numpy(dtype=bool) & (~preserve))

    return dedupe_actions(actions)


def apply_actions(base_pred: np.ndarray, actions: list[Action], restore_ids: tuple[int, ...], veto_ids: tuple[int, ...]) -> np.ndarray:
    pred = base_pred.copy()
    for i in restore_ids:
        pred[actions[i].mask] = 1
    for i in veto_ids:
        pred[actions[i].mask] = 0
    return pred


def score_state(m: dict[str, Any], min_tp_required: int, base_fp: int) -> tuple:
    return (
        int(m["tp"] >= min_tp_required),
        -m["fp"],
        m["tp"],
        m["precision"],
        m["f1"],
    )


def beam_search(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    restore_actions: list[Action],
    veto_actions: list[Action],
    target_recall: float,
    min_tp_required: int,
    restore_beam: int,
    veto_beam: int,
    max_restore_rules: int,
    max_veto_rules: int,
) -> tuple[pd.DataFrame, State]:
    y = df["is_fraud"].to_numpy(dtype=int)
    base_metrics = compute_metrics(y, base_pred)
    all_actions = restore_actions + veto_actions
    veto_offset = len(restore_actions)

    # First, build restore states. Include base state.
    base_state = State(base_pred.copy(), tuple(), tuple(), base_metrics)
    restore_states = [base_state]
    all_restore_states = [base_state]

    for depth in range(1, max_restore_rules + 1):
        next_states: dict[bytes, State] = {}
        for state in restore_states:
            last = state.restore_ids[-1] if state.restore_ids else -1
            for ridx in range(last + 1, len(restore_actions)):
                pred = state.pred.copy()
                pred[restore_actions[ridx].mask] = 1
                m = compute_metrics(y, pred)

                # Don't allow huge FP restoration.
                if m["fp"] > base_metrics["fp"] + 150:
                    continue

                key = np.packbits(pred.astype(bool)).tobytes()
                ns = State(pred, state.restore_ids + (ridx,), tuple(), m)
                old = next_states.get(key)
                if old is None or (m["tp"], -m["fp"], m["precision"]) > (old.metrics["tp"], -old.metrics["fp"], old.metrics["precision"]):
                    next_states[key] = ns

        if not next_states:
            break

        restore_states = sorted(next_states.values(), key=lambda s: (s.metrics["tp"], -s.metrics["fp"]), reverse=True)[:restore_beam]
        all_restore_states.extend(restore_states)
        log(f"  restore depth={depth}: states={len(restore_states)}, best_tp={restore_states[0].metrics['tp']}, best_fp={restore_states[0].metrics['fp']}")

    all_restore_states = sorted(all_restore_states, key=lambda s: (s.metrics["tp"], -s.metrics["fp"]), reverse=True)[:restore_beam]

    best = base_state
    rows = []

    for rs_idx, rs in enumerate(all_restore_states):
        states = [rs]
        for depth in range(1, max_veto_rules + 1):
            next_states: dict[bytes, State] = {}
            for state in states:
                last = state.veto_ids[-1] - veto_offset if state.veto_ids else -1
                for vidx_rel in range(last + 1, len(veto_actions)):
                    vidx = veto_offset + vidx_rel
                    pred = state.pred.copy()
                    pred[veto_actions[vidx_rel].mask] = 0
                    m = compute_metrics(y, pred)

                    # Hard no-go: never below target recall/TP.
                    if m["tp"] < min_tp_required or m["recall"] < target_recall:
                        continue
                    if m["fp"] > base_metrics["fp"]:
                        continue

                    key = np.packbits(pred.astype(bool)).tobytes()
                    ns = State(pred, state.restore_ids, state.veto_ids + (vidx,), m)
                    old = next_states.get(key)
                    if old is None or score_state(m, min_tp_required, base_metrics["fp"]) > score_state(old.metrics, min_tp_required, base_metrics["fp"]):
                        next_states[key] = ns

            if not next_states:
                break

            states = sorted(next_states.values(), key=lambda s: score_state(s.metrics, min_tp_required, base_metrics["fp"]), reverse=True)[:veto_beam]

            if score_state(states[0].metrics, min_tp_required, base_metrics["fp"]) > score_state(best.metrics, min_tp_required, base_metrics["fp"]):
                best = states[0]

            for s in states[:50]:
                rows.append({
                    "restore_state_idx": rs_idx,
                    "depth": depth,
                    "tp": s.metrics["tp"],
                    "fp": s.metrics["fp"],
                    "fn": s.metrics["fn"],
                    "precision": s.metrics["precision"],
                    "recall": s.metrics["recall"],
                    "f1": s.metrics["f1"],
                    "n_restore_rules": len(s.restore_ids),
                    "n_veto_rules": len(s.veto_ids),
                    "restore_action_ids": "|".join(all_actions[i].action_id for i in s.restore_ids),
                    "veto_action_ids": "|".join(all_actions[i].action_id for i in s.veto_ids),
                })

    if not rows:
        rows = [{
            "restore_state_idx": 0,
            "depth": 0,
            "tp": best.metrics["tp"],
            "fp": best.metrics["fp"],
            "fn": best.metrics["fn"],
            "precision": best.metrics["precision"],
            "recall": best.metrics["recall"],
            "f1": best.metrics["f1"],
            "n_restore_rules": len(best.restore_ids),
            "n_veto_rules": len(best.veto_ids),
            "restore_action_ids": "",
            "veto_action_ids": "",
        }]

    frontier = pd.DataFrame(rows).sort_values(["fp", "tp"], ascending=[True, False]).reset_index(drop=True)
    return frontier, best


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


def block_metrics(df: pd.DataFrame, pred: np.ndarray, n_blocks: int, policy_name: str) -> pd.DataFrame:
    blocks = make_time_blocks(df, n_blocks)
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
    n = len(df)
    rows = []
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


def make_report(summary: dict[str, Any], selected: pd.DataFrame, blocks: pd.DataFrame, boot: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-013G — High Recall 95 Micro-Refinement")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Base HIGH_RECALL_95: TP={summary['base_high_recall_metrics']['tp']}, FP={summary['base_high_recall_metrics']['fp']}, FN={summary['base_high_recall_metrics']['fn']}, recall={summary['base_high_recall_metrics']['recall']}")
    lines.append(f"- Micro-refinada: TP={summary['selected_metrics']['tp']}, FP={summary['selected_metrics']['fp']}, FN={summary['selected_metrics']['fn']}, recall={summary['selected_metrics']['recall']}, precision={summary['selected_metrics']['precision']}")
    lines.append(f"- FP delta vs HIGH_RECALL_95: {summary['fp_delta_vs_high_recall']}")
    lines.append(f"- TP delta vs HIGH_RECALL_95: {summary['tp_delta_vs_high_recall']}")
    lines.append("")
    lines.append("## Ações selecionadas")
    if selected.empty:
        lines.append("Nenhuma microação selecionada; HIGH_RECALL_95 permanece o benchmark.")
    else:
        lines.append(selected[["action_type", "family", "description", "tp_delta", "fp_delta"]].to_markdown(index=False))
    lines.append("")
    lines.append("## Blocos temporais")
    lines.append(blocks.to_markdown(index=False))
    lines.append("")
    lines.append("## Bootstrap recall")
    rec = boot[boot["metric"] == "recall"]
    lines.append(rec.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretação")
    if summary["selected_metrics"]["fp"] < summary["base_high_recall_metrics"]["fp"]:
        lines.append("A microevolução reduziu FP mantendo recall >=95%. Ela deve substituir o HIGH_RECALL_95 como novo benchmark apenas após revisão dos FNs e validação E2E/temporal.")
    else:
        lines.append("Não houve microevolução segura. Manter HIGH_RECALL_95 como benchmark vencedor.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--policy-artifact", default=str(DEFAULT_POLICY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-blocks", type=int, default=5)
    parser.add_argument("--max-fp-restore-per-tp", type=float, default=60.0)
    parser.add_argument("--min-fp-removed", type=int, default=10)
    parser.add_argument("--max-tp-loss-per-veto", type=int, default=1)
    parser.add_argument("--restore-beam", type=int, default=50)
    parser.add_argument("--veto-beam", type=int, default=100)
    parser.add_argument("--max-restore-rules", type=int, default=2)
    parser.add_argument("--max-veto-rules", type=int, default=5)
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-013G — High Recall 95 Micro-Refinement")
    log("=" * 80)
    log(f"Input: {input_path}")
    log(f"Output: {output_dir}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    y = df["is_fraud"].to_numpy(dtype=int)
    base_pred = df["pred_HIGH_RECALL_95"].to_numpy(dtype=int)

    total_frauds = int(y.sum())
    min_tp_required = int(math.ceil(args.target_recall * total_frauds))
    base_metrics = compute_metrics(y, base_pred)

    log(f"HIGH_RECALL_95 base: TP={base_metrics['tp']} FP={base_metrics['fp']} FN={base_metrics['fn']} recall={base_metrics['recall']}")
    log(f"Target recall={args.target_recall}; min_tp_required={min_tp_required}")

    if base_metrics["tp"] < min_tp_required:
        raise RuntimeError("Base HIGH_RECALL_95 não cumpre TP mínimo; abortando.")

    pd.DataFrame([{"policy_name": "HIGH_RECALL_95", **base_metrics}]).to_csv(output_dir / "01_base_metrics.csv", index=False)

    log("[1/4] Gerando candidatos de restauração...")
    restore_actions = generate_restore_candidates(df, base_pred, y, args.max_fp_restore_per_tp)
    restore_df = action_df(restore_actions).sort_values(["tp_delta", "fp_delta"], ascending=[False, True]) if restore_actions else pd.DataFrame()
    restore_df.to_csv(output_dir / "02_restore_candidates.csv", index=False)
    log(f"    restore candidates={len(restore_actions)}")

    log("[2/4] Gerando candidatos de veto...")
    veto_actions = generate_veto_candidates(df, base_pred, y, args.min_fp_removed, args.max_tp_loss_per_veto)
    veto_df = action_df(veto_actions).sort_values(["tp_delta", "fp_delta"], ascending=[False, True]) if veto_actions else pd.DataFrame()
    veto_df.to_csv(output_dir / "03_veto_candidates.csv", index=False)
    log(f"    veto candidates={len(veto_actions)}")

    log("[3/4] Busca local com recall hard floor...")
    frontier, best = beam_search(
        df=df,
        base_pred=base_pred,
        restore_actions=restore_actions,
        veto_actions=veto_actions,
        target_recall=args.target_recall,
        min_tp_required=min_tp_required,
        restore_beam=args.restore_beam,
        veto_beam=args.veto_beam,
        max_restore_rules=args.max_restore_rules,
        max_veto_rules=args.max_veto_rules,
    )
    frontier.to_csv(output_dir / "04_micro_frontier.csv", index=False)

    all_actions = restore_actions + veto_actions
    selected_indices = list(best.restore_ids) + list(best.veto_ids)
    selected_df = action_df([all_actions[i] for i in selected_indices]) if selected_indices else pd.DataFrame()
    selected_df.to_csv(output_dir / "05_selected_micro_rules.csv", index=False)

    selected_pred = best.pred
    selected_metrics = compute_metrics(y, selected_pred)

    predictions = df.copy()
    predictions["exp013g_micro_pred"] = selected_pred
    predictions["exp013g_high_recall_base_pred"] = base_pred
    predictions["exp013g_changed_vs_high_recall"] = (selected_pred != base_pred).astype(int)
    predictions.to_csv(output_dir / "06_selected_predictions.csv", index=False)
    predictions[(predictions["is_fraud"] == 1) & (predictions["exp013g_micro_pred"] == 0)].to_csv(output_dir / "07_selected_false_negatives.csv", index=False)
    predictions[(predictions["is_fraud"] == 0) & (predictions["exp013g_micro_pred"] == 1)].to_csv(output_dir / "08_selected_false_positives.csv", index=False)

    log("[4/4] Blocos temporais e bootstrap...")
    blocks = pd.concat([
        block_metrics(df, base_pred, args.time_blocks, "HIGH_RECALL_95"),
        block_metrics(df, selected_pred, args.time_blocks, "EXP013G_MICRO_REFINED"),
    ], ignore_index=True)
    blocks.to_csv(output_dir / "09_time_block_metrics.csv", index=False)

    boot_input = predictions.copy()
    boot_df = bootstrap_eval(boot_input, "exp013g_micro_pred", args.bootstrap_iters, args.seed, args.target_recall)
    boot_df.to_csv(output_dir / "10_bootstrap_confidence_intervals.csv", index=False)

    min_block_selected = float(blocks[blocks["policy_name"] == "EXP013G_MICRO_REFINED"]["recall"].min()) if not blocks.empty else None
    boot_recall = boot_df[boot_df["metric"] == "recall"].iloc[0].to_dict() if not boot_df.empty else {}

    objective_status = "TARGET_RECALL_MET" if selected_metrics["recall"] >= args.target_recall and selected_metrics["tp"] >= min_tp_required else "TARGET_RECALL_NOT_MET"
    objective_status += "_FP_REDUCED" if selected_metrics["fp"] < base_metrics["fp"] else "_FP_NOT_REDUCED"
    objective_status += "_TP_NOT_WORSE" if selected_metrics["tp"] >= base_metrics["tp"] else "_TP_LOWER_THAN_BASE"

    policy_artifact = {
        "experiment": "EXP-013G",
        "policy_name": "high_recall95_micro_refined_policy",
        "objective_status": objective_status,
        "base_policy": "EXP-013F HIGH_RECALL_95",
        "target_recall": args.target_recall,
        "min_tp_required": min_tp_required,
        "base_high_recall_metrics": base_metrics,
        "selected_metrics": selected_metrics,
        "selected_actions": selected_df.to_dict(orient="records") if not selected_df.empty else [],
        "notes": [
            "Starts from EXP-013F HIGH_RECALL_95.",
            "Hard constraint: recall >=95% and TP >= min_tp_required.",
            "Microevolution only; no broad retuning.",
            "Validate externally before production."
        ],
    }
    dump_json(policy_artifact, output_dir / "12_policy_artifact.json")

    summary = {
        "experiment": "EXP-013G",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "n_rows": int(len(df)),
        "total_frauds": total_frauds,
        "target_recall": args.target_recall,
        "min_tp_required": min_tp_required,
        "base_high_recall_metrics": base_metrics,
        "selected_metrics": selected_metrics,
        "tp_delta_vs_high_recall": int(selected_metrics["tp"] - base_metrics["tp"]),
        "fp_delta_vs_high_recall": int(selected_metrics["fp"] - base_metrics["fp"]),
        "fn_delta_vs_high_recall": int(selected_metrics["fn"] - base_metrics["fn"]),
        "n_restore_candidates": int(len(restore_actions)),
        "n_veto_candidates": int(len(veto_actions)),
        "n_selected_restore_actions": int(len(best.restore_ids)),
        "n_selected_veto_actions": int(len(best.veto_ids)),
        "min_block_recall_selected": min_block_selected,
        "bootstrap_recall": boot_recall,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, selected_df, blocks, boot_df)
    (output_dir / "11_micro_refinement_report.md").write_text(report, encoding="utf-8")

    log("")
    log("=" * 80)
    log("EXP-013G CONCLUÍDO")
    log("=" * 80)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "04_micro_frontier.csv",
        output_dir / "05_selected_micro_rules.csv",
        output_dir / "06_selected_predictions.csv",
        output_dir / "09_time_block_metrics.csv",
        output_dir / "10_bootstrap_confidence_intervals.csv",
        output_dir / "11_micro_refinement_report.md",
        output_dir / "12_policy_artifact.json",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
