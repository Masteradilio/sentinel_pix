#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3P — Residual FP-only Conservative Reducer sobre R3O-FROZEN

Objetivo:
  Reduzir falsos positivos residuais após o R3O-FROZEN, preservando integralmente:
    TP=1465
    FN=0
    recall=100%

Princípios:
  - não aplicar novos rescues;
  - não alterar threshold;
  - não chamar runtime;
  - minerar apenas vetos sobre alertas atuais do R3O-FROZEN;
  - aceitar somente regras com TP_loss=0;
  - exigir estabilidade temporal mínima para reduzir risco de overfitting.

Uso recomendado:
  python scripts/exp_014b_r3p_residual_fp_only_conservative_reducer.py

Uso um pouco mais exploratório, ainda conservador:
  python scripts/exp_014b_r3p_residual_fp_only_conservative_reducer.py --max-rules 15 --min-fp-removed 12 --max-combo-size 3 --top-groups-per-combo 30

Saídas:
  resultados/experimentos/EXP-014B-R3P/
    00_run_summary.json
    01_input_contract.json
    02_base_validation.json
    03_fp_candidates.csv
    04_selection_frontier.csv
    05_selected_fp_rules.csv
    06_rule_stability_audit.csv
    07_policy_artifact_recommended.json
    08_predictions_recommended.csv
    09_exp014b_r3p_report.md
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

DEFAULT_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3O-FROZEN" / "09_predictions_frozen.csv"
DEFAULT_ARTIFACT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3O-FROZEN" / "08_policy_artifact_frozen.json"
DEFAULT_OUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3P"

BASE_COL = "exp014b_r3o_frozen_pred"
FINAL_COL = "exp014b_r3p_recommended_pred"

EXPECTED_BASE = {
    "tp": 1465,
    "fp": 4252,
    "fn": 0,
    "precision": 0.25625328,
    "recall": 1.0,
    "fpr": 0.03783625,
    "wilson_low_min": 0.99,
}

# Mantemos somente features operacionais/diagnosticas já usadas nas rodadas FP-only.
# Evitar source_dataset, sample_strategy, dataset_role e qualquer label.
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
class Candidate:
    rule_id: str
    family: str
    description: str
    mask: np.ndarray
    tp_loss: int
    fp_removed: int
    n_removed: int
    n_temporal_splits_with_fp_removed: int
    n_months_with_fp_removed: int
    has_nontrain_support: bool
    combo_size: int
    params: dict[str, Any]


def log(msg: str) -> None:
    print(msg, flush=True)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
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


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n) + (z * z / (4 * n * n))) ** 0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().split(".")[-1] for c in df.columns]
    if "transaction_id" not in df.columns and "cd_pix" in df.columns:
        df["transaction_id"] = df["cd_pix"]
    if "event_datetime" not in df.columns and "dt_pix" in df.columns:
        df["event_datetime"] = df["dt_pix"]
    if "is_fraud" not in df.columns:
        raise RuntimeError("Coluna obrigatoria ausente: is_fraud")

    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)
    for c in [BASE_COL, FINAL_COL, "exp014b_r3o_recommended_pred", "exp014b_r3n_frozen_pred"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "event_datetime" in df.columns:
        df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
    if "event_month" not in df.columns and "event_datetime" in df.columns:
        df["event_month"] = df["event_datetime"].dt.to_period("M").astype(str).replace("NaT", "MISSING")
    elif "event_month" not in df.columns:
        df["event_month"] = "MISSING"

    if "temporal_split" not in df.columns:
        df["temporal_split"] = "UNKNOWN"

    return df.reset_index(drop=True)


def pick(df: pd.DataFrame, names: str | list[str]) -> str | None:
    if isinstance(names, str):
        names = [names]
    for n in names:
        if n in df.columns:
            return n
    return None


def num(df: pd.DataFrame, names: str | list[str], default: float = 0.0) -> pd.Series:
    c = pick(df, names)
    if c is None:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def qbin(s: pd.Series, name: str, bins: list[float]) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    edges = [-np.inf] + bins + [np.inf]
    labels: list[str] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if np.isneginf(a):
            labels.append(f"{name}_LT_{b:g}")
        elif np.isposinf(b):
            labels.append(f"{name}_GE_{a:g}")
        else:
            labels.append(f"{name}_{a:g}_{b:g}")
    return pd.cut(vals, bins=edges, labels=labels, include_lowest=True).astype("string").fillna(f"{name}_MISSING").astype(str)


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "lgbm_bin" not in df.columns and pick(df, ["lgbm_r4_score", "lgbm_mapped", "lgbm_raw"]):
        df["lgbm_bin"] = qbin(num(df, ["lgbm_r4_score", "lgbm_mapped", "lgbm_raw"]), "lgbm", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1])
    if "if_bin" not in df.columns and pick(df, ["if_percentile", "if_percentile_x", "if_percentile_y"]):
        df["if_bin"] = qbin(num(df, ["if_percentile", "if_percentile_x", "if_percentile_y"]), "if", [0.32, 0.5, 0.7, 0.85, 0.95])
    if "score_bin" not in df.columns and "score_final" in df.columns:
        df["score_bin"] = qbin(num(df, "score_final"), "score", [0.5, 1, 2, 3, 5, 10])
    if "ratio_bin" not in df.columns and "ratio_valor_media_pagador_90d" in df.columns:
        df["ratio_bin"] = qbin(num(df, "ratio_valor_media_pagador_90d"), "ratio", [0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    if "qtd_rec_bin" not in df.columns and "qtd_pix_recebidos_180d" in df.columns:
        df["qtd_rec_bin"] = qbin(num(df, "qtd_pix_recebidos_180d"), "qtdrec", [0, 1, 2, 5, 10, 20, 50, 100])
    if "vl_bin" not in df.columns and "vl_pix" in df.columns:
        df["vl_bin"] = qbin(num(df, "vl_pix"), "vl", [20, 50, 100, 250, 500, 1000, 5000, 10000])

    if "module_quiet" not in df.columns:
        se = num(df, ["se_score_x", "se_score_y", "se_score"], 0.0)
        se_count = num(df, ["se_patterns_count", "se_pattern_count"], 0.0)
        beh = num(df, ["beh_score", "behavioral_score"], 0.0)
        beh_count = num(df, ["beh_factors_count", "behavioral_risk_factor_count"], 0.0)
        runtime = num(df, "runtime_flagged", 0.0)
        strong = (se >= 40) | (se_count >= 2) | (beh >= 25) | (beh_count >= 2) | (runtime >= 1)
        df["module_quiet"] = np.where(strong, "module_strong", "module_quiet")

    return df


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in FEATURE_COLS:
        if c in df.columns:
            out[c] = df[c].astype("string").fillna("<MISSING>").astype(str)
    return out


def stability_counts(df: pd.DataFrame, mask: np.ndarray) -> tuple[int, int, bool]:
    # mask ja representa remocao marginal.
    fp_mask = mask & (df["is_fraud"].to_numpy(dtype=int) == 0)
    if not fp_mask.any():
        return 0, 0, False

    tmp = df.loc[fp_mask, ["temporal_split", "event_month"]].copy()
    n_splits = int(tmp["temporal_split"].astype(str).nunique())
    n_months = int(tmp["event_month"].astype(str).nunique())
    has_nontrain = bool((tmp["temporal_split"].astype(str).str.upper() != "TRAIN").any())
    return n_splits, n_months, has_nontrain


def candidate_ok(
    *,
    tp_loss: int,
    fp_removed: int,
    n_splits: int,
    n_months: int,
    has_nontrain: bool,
    combo_size: int,
    min_fp_removed: int,
    min_temporal_splits: int,
    min_months: int,
    require_nontrain_support: bool,
) -> bool:
    if tp_loss != 0 or fp_removed < min_fp_removed:
        return False
    if n_splits < min_temporal_splits:
        return False
    if n_months < min_months:
        return False
    if require_nontrain_support and not has_nontrain:
        return False
    # Regra extra: combos de 4, se habilitados, precisam de suporte ainda melhor.
    if combo_size >= 4 and (n_splits < 3 or n_months < 3):
        return False
    return True


def add_candidate(
    out: list[Candidate],
    *,
    df: pd.DataFrame,
    family: str,
    description: str,
    mask: np.ndarray,
    params: dict[str, Any],
    combo_size: int,
    min_fp_removed: int,
    min_temporal_splits: int,
    min_months: int,
    require_nontrain_support: bool,
) -> None:
    y = df["is_fraud"].to_numpy(dtype=int)
    if not mask.any():
        return
    tp_loss = int(((y == 1) & mask).sum())
    fp_removed = int(((y == 0) & mask).sum())
    n_splits, n_months, has_nontrain = stability_counts(df, mask)

    if not candidate_ok(
        tp_loss=tp_loss,
        fp_removed=fp_removed,
        n_splits=n_splits,
        n_months=n_months,
        has_nontrain=has_nontrain,
        combo_size=combo_size,
        min_fp_removed=min_fp_removed,
        min_temporal_splits=min_temporal_splits,
        min_months=min_months,
        require_nontrain_support=require_nontrain_support,
    ):
        return

    out.append(Candidate(
        rule_id=f"r3p_fp_{len(out):05d}",
        family=family,
        description=description,
        mask=mask,
        tp_loss=tp_loss,
        fp_removed=fp_removed,
        n_removed=int(mask.sum()),
        n_temporal_splits_with_fp_removed=n_splits,
        n_months_with_fp_removed=n_months,
        has_nontrain_support=has_nontrain,
        combo_size=combo_size,
        params=params,
    ))


def mine_candidates(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    *,
    min_fp_removed: int,
    max_combo_size: int,
    top_groups_per_combo: int,
    min_temporal_splits: int,
    min_months: int,
    require_nontrain_support: bool,
) -> list[Candidate]:
    y = df["is_fraud"].to_numpy(dtype=int)
    alerted = base_pred.astype(int) == 1
    out: list[Candidate] = []

    # Numeric thresholds: poucos cortes quantilicos, evitando grid caro.
    for c in NUMERIC_COLS:
        if c not in df.columns:
            continue
        vals = num(df, c).to_numpy(dtype=float)
        active = vals[alerted]
        if len(active) == 0:
            continue
        try:
            cuts = sorted(set(float(x) for x in np.quantile(active, [0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 0.95, 0.97]) if np.isfinite(x)))
        except Exception:
            cuts = []
        for cut in cuts:
            for direction in ["le", "ge"]:
                mask = alerted & ((vals <= cut) if direction == "le" else (vals >= cut))
                desc = f"alert AND {c}<={cut:g}" if direction == "le" else f"alert AND {c}>={cut:g}"
                add_candidate(
                    out,
                    df=df,
                    family="r3p_numeric_tp0_stable",
                    description=desc,
                    mask=mask,
                    params={"type": "numeric_headroom", "col": c, "direction": direction, "cut": cut},
                    combo_size=1,
                    min_fp_removed=min_fp_removed,
                    min_temporal_splits=min_temporal_splits,
                    min_months=min_months,
                    require_nontrain_support=require_nontrain_support,
                )

    feat = feature_frame(df)
    cols = list(feat.columns)
    bins = [c for c in cols if c.endswith("_bin") or c == "module_quiet"]
    important = ["ds_tipo_chave_norm", "value_band", "mbk_available_flag", "first_receiver_flag_real", "periodo_dia"]
    idx = np.where(alerted)[0]

    for r in range(1, max_combo_size + 1):
        for combo in itertools.combinations(cols, r):
            combo = list(combo)
            if r == 1 and combo[0] not in bins + ["ds_tipo_chave_norm", "value_band"]:
                continue
            if r >= 2 and not any(c in combo for c in important + bins):
                continue

            sub = feat.iloc[idx][combo]
            if sub.empty:
                continue

            group_rows = []
            for key, rel in sub.groupby(combo, dropna=False).indices.items():
                rows = sub.iloc[list(rel)].index.to_numpy(dtype=int)
                if len(rows) < min_fp_removed:
                    continue
                mask = np.zeros(len(df), dtype=bool)
                mask[rows] = True
                mask &= alerted

                tp_loss = int(((y == 1) & mask).sum())
                fp_removed = int(((y == 0) & mask).sum())
                n_splits, n_months, has_nontrain = stability_counts(df, mask)
                if candidate_ok(
                    tp_loss=tp_loss,
                    fp_removed=fp_removed,
                    n_splits=n_splits,
                    n_months=n_months,
                    has_nontrain=has_nontrain,
                    combo_size=r,
                    min_fp_removed=min_fp_removed,
                    min_temporal_splits=min_temporal_splits,
                    min_months=min_months,
                    require_nontrain_support=require_nontrain_support,
                ):
                    group_rows.append((-fp_removed, key, mask, fp_removed))

            group_rows.sort()
            for _, key, mask, _ in group_rows[:top_groups_per_combo]:
                vals = key if isinstance(key, tuple) else (key,)
                vals = [str(v) for v in vals]
                desc = "alert AND " + " AND ".join([f"{c}={v}" for c, v in zip(combo, vals)])
                add_candidate(
                    out,
                    df=df,
                    family="r3p_combo_tp0_stable",
                    description=desc,
                    mask=mask,
                    params={"type": "combo_headroom", "combo_cols": combo, "combo_values": vals},
                    combo_size=r,
                    min_fp_removed=min_fp_removed,
                    min_temporal_splits=min_temporal_splits,
                    min_months=min_months,
                    require_nontrain_support=require_nontrain_support,
                )

    # Dedup por mascara, mantendo regra mais simples e maior FP.
    best: dict[bytes, Candidate] = {}
    for c in out:
        key = np.packbits(c.mask).tobytes()
        old = best.get(key)
        if old is None:
            best[key] = c
        else:
            rank_new = (c.fp_removed, -c.combo_size, -len(c.description))
            rank_old = (old.fp_removed, -old.combo_size, -len(old.description))
            if rank_new > rank_old:
                best[key] = c

    out = list(best.values())
    out.sort(key=lambda c: (-c.fp_removed, c.combo_size, -c.n_temporal_splits_with_fp_removed, -c.n_months_with_fp_removed, len(c.description)))
    for i, c in enumerate(out):
        c.rule_id = f"r3p_fp_{i:05d}"
    return out


def candidate_dicts(cands: list[Candidate]) -> list[dict[str, Any]]:
    return [{
        "rule_id": c.rule_id,
        "family": c.family,
        "description": c.description,
        "tp_loss": int(c.tp_loss),
        "fp_removed": int(c.fp_removed),
        "n_removed": int(c.n_removed),
        "combo_size": int(c.combo_size),
        "n_temporal_splits_with_fp_removed": int(c.n_temporal_splits_with_fp_removed),
        "n_months_with_fp_removed": int(c.n_months_with_fp_removed),
        "has_nontrain_support": bool(c.has_nontrain_support),
        "params_json": json.dumps(c.params, ensure_ascii=False),
    } for c in cands]


def recompute_marginal_candidate(df: pd.DataFrame, cand: Candidate, current_pred: np.ndarray) -> Candidate | None:
    alerted = current_pred.astype(int) == 1
    mask = cand.mask & alerted
    y = df["is_fraud"].to_numpy(dtype=int)
    tp_loss = int(((y == 1) & mask).sum())
    fp_removed = int(((y == 0) & mask).sum())
    n_splits, n_months, has_nontrain = stability_counts(df, mask)
    if not mask.any() or fp_removed <= 0:
        return None
    return Candidate(
        rule_id=cand.rule_id,
        family=cand.family,
        description=cand.description,
        mask=mask,
        tp_loss=tp_loss,
        fp_removed=fp_removed,
        n_removed=int(mask.sum()),
        n_temporal_splits_with_fp_removed=n_splits,
        n_months_with_fp_removed=n_months,
        has_nontrain_support=has_nontrain,
        combo_size=cand.combo_size,
        params=cand.params,
    )


def select_rules(
    df: pd.DataFrame,
    cands: list[Candidate],
    base_pred: np.ndarray,
    *,
    max_rules: int,
    max_seconds: int,
    min_fp_removed: int,
    min_temporal_splits: int,
    min_months: int,
    require_nontrain_support: bool,
) -> tuple[np.ndarray, list[Candidate], pd.DataFrame, str]:
    t0 = time.perf_counter()
    y = df["is_fraud"].to_numpy(dtype=int)
    current = base_pred.copy().astype(int)
    selected: list[Candidate] = []
    used: set[str] = set()
    rows: list[dict[str, Any]] = []
    stop = "completed"

    # Limitar candidatos para manter execução curta e auditável.
    pool = cands[:2000]

    for depth in range(1, max_rules + 1):
        if time.perf_counter() - t0 >= max_seconds:
            stop = f"max_seconds_before_rule_{depth}"
            break

        best: Candidate | None = None
        for cand in pool:
            if cand.rule_id in used:
                continue
            marginal = recompute_marginal_candidate(df, cand, current)
            if marginal is None:
                continue
            if not candidate_ok(
                tp_loss=marginal.tp_loss,
                fp_removed=marginal.fp_removed,
                n_splits=marginal.n_temporal_splits_with_fp_removed,
                n_months=marginal.n_months_with_fp_removed,
                has_nontrain=marginal.has_nontrain_support,
                combo_size=marginal.combo_size,
                min_fp_removed=min_fp_removed,
                min_temporal_splits=min_temporal_splits,
                min_months=min_months,
                require_nontrain_support=require_nontrain_support,
            ):
                continue
            if best is None:
                best = marginal
            else:
                rank = (marginal.fp_removed, -marginal.combo_size, marginal.n_temporal_splits_with_fp_removed, marginal.n_months_with_fp_removed, -len(marginal.description))
                best_rank = (best.fp_removed, -best.combo_size, best.n_temporal_splits_with_fp_removed, best.n_months_with_fp_removed, -len(best.description))
                if rank > best_rank:
                    best = marginal

        if best is None:
            stop = f"no_more_stable_tp0_fp_rules_at_depth_{depth}"
            break

        current[best.mask] = 0
        used.add(best.rule_id)
        selected.append(best)
        m = metrics(y, current)
        rows.append({
            "depth": depth,
            "rule_id": best.rule_id,
            "family": best.family,
            "description": best.description,
            "marginal_fp_removed": int(best.fp_removed),
            "cumulative_fp_removed": int(sum(c.fp_removed for c in selected)),
            "tp_loss": int(best.tp_loss),
            "combo_size": int(best.combo_size),
            "n_temporal_splits_with_fp_removed": int(best.n_temporal_splits_with_fp_removed),
            "n_months_with_fp_removed": int(best.n_months_with_fp_removed),
            "has_nontrain_support": bool(best.has_nontrain_support),
            **m,
        })

    if not rows:
        rows.append({
            "depth": 0,
            "rule_id": "",
            "family": "",
            "description": "",
            "marginal_fp_removed": 0,
            "cumulative_fp_removed": 0,
            "tp_loss": 0,
            "combo_size": 0,
            "n_temporal_splits_with_fp_removed": 0,
            "n_months_with_fp_removed": 0,
            "has_nontrain_support": False,
            **metrics(y, base_pred),
        })

    return current, selected, pd.DataFrame(rows), stop


def segment_audit(df: pd.DataFrame, base_pred: np.ndarray, final_pred: np.ndarray) -> pd.DataFrame:
    y = df["is_fraud"].to_numpy(dtype=int)
    rows = []
    segment_cols = [
        "temporal_split",
        "event_month",
        "ds_tipo_chave_norm",
        "value_band",
        "mbk_available_flag",
        "periodo_dia",
    ]
    for col in segment_cols:
        if col not in df.columns:
            continue
        for val, idx in df.groupby(col, dropna=False).indices.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            b = metrics(y[idx_arr], base_pred[idx_arr])
            f = metrics(y[idx_arr], final_pred[idx_arr])
            rows.append({
                "segment_col": col,
                "segment_value": str(val),
                "n_rows": int(len(idx_arr)),
                "n_frauds": int(y[idx_arr].sum()),
                "fp_removed": int(b["fp"] - f["fp"]),
                "tp_loss": int(b["tp"] - f["tp"]),
                "fn_delta": int(f["fn"] - b["fn"]),
                "base_tp": b["tp"],
                "base_fp": b["fp"],
                "base_fn": b["fn"],
                "final_tp": f["tp"],
                "final_fp": f["fp"],
                "final_fn": f["fn"],
                "final_recall": f["recall"],
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["tp_loss", "fn_delta", "fp_removed"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def make_report(summary: dict[str, Any], base_val: dict[str, Any], frontier: pd.DataFrame, selected: pd.DataFrame, stability: pd.DataFrame, segments: pd.DataFrame) -> str:
    lines = [
        "# EXP-014B-R3P — Residual FP-only Conservative Reducer",
        "",
        "## Resultado executivo",
        f"- Status: `{summary['objective_status']}`",
        f"- All pass: `{summary['all_pass']}`",
        f"- Base R3O-FROZEN: `{summary['base_r3o_frozen_metrics']}`",
        f"- Métricas recomendadas R3P: `{summary['recommended_metrics']}`",
        f"- FP removidos vs R3O: `{summary['fp_removed_vs_r3o']}`",
        f"- TP loss vs R3O: `{summary['tp_loss_vs_r3o']}`",
        f"- FN delta vs R3O: `{summary['fn_delta_vs_r3o']}`",
        "",
        "## Validação da base",
        "```json",
        json.dumps(base_val, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Fronteira selecionada",
    ]

    if frontier.empty:
        lines.append("Nenhuma regra selecionada.")
    else:
        show = ["depth", "marginal_fp_removed", "cumulative_fp_removed", "tp", "fp", "fn", "precision", "recall", "fpr", "description"]
        lines.append(frontier[[c for c in show if c in frontier.columns]].to_markdown(index=False))

    lines += ["", "## Regras selecionadas"]
    if selected.empty:
        lines.append("Nenhuma regra selecionada.")
    else:
        show = ["rule_id", "family", "description", "fp_removed", "tp_loss", "combo_size", "n_temporal_splits_with_fp_removed", "n_months_with_fp_removed"]
        lines.append(selected[[c for c in show if c in selected.columns]].to_markdown(index=False))

    lines += ["", "## Auditoria de estabilidade das regras selecionadas"]
    if stability.empty:
        lines.append("Nenhuma regra selecionada.")
    else:
        show = ["rule_id", "fp_removed", "tp_loss", "combo_size", "n_temporal_splits_with_fp_removed", "n_months_with_fp_removed", "has_nontrain_support", "description"]
        lines.append(stability[[c for c in show if c in stability.columns]].to_markdown(index=False))

    lines += ["", "## Robustez por segmento"]
    if segments.empty:
        lines.append("Sem auditoria de segmento.")
    else:
        show = ["segment_col", "segment_value", "n_rows", "n_frauds", "fp_removed", "tp_loss", "fn_delta", "final_tp", "final_fp", "final_fn", "final_recall"]
        lines.append(segments[[c for c in show if c in segments.columns]].head(40).to_markdown(index=False))

    lines += ["", "## Decisão sugerida"]
    if summary["all_pass"] and summary["fp_removed_vs_r3o"] > 0:
        lines.append("R3P gerou candidato FP-only conservador superior ao R3O-FROZEN. Próximo passo: EXP-014B-R3P-FROZEN, sem nova mineração.")
    else:
        lines.append("R3P não encontrou ganho conservador suficiente. Manter R3O-FROZEN como benchmark principal e iniciar hardening/produção shadow.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--min-fp-removed", type=int, default=15)
    parser.add_argument("--max-rules", type=int, default=10)
    parser.add_argument("--max-combo-size", type=int, default=2)
    parser.add_argument("--top-groups-per-combo", type=int, default=20)
    parser.add_argument("--min-temporal-splits", type=int, default=2)
    parser.add_argument("--min-months", type=int, default=2)
    parser.add_argument("--require-nontrain-support", action="store_true", default=True)
    parser.add_argument("--no-require-nontrain-support", dest="require_nontrain_support", action="store_false")
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--no-write-predictions", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014B-R3P — Residual FP-only Conservative Reducer")
    log("=" * 80)

    input_path = Path(args.input)
    artifact_path = Path(args.artifact)
    if not input_path.exists():
        raise FileNotFoundError(f"input nao encontrado: {input_path}")
    if not artifact_path.exists():
        raise FileNotFoundError(f"artifact nao encontrado: {artifact_path}")

    df = add_bins(normalize(pd.read_csv(input_path, low_memory=False)))
    artifact = load_json(artifact_path)
    y = df["is_fraud"].to_numpy(dtype=int)

    missing = []
    for c in ["is_fraud", BASE_COL]:
        if c not in df.columns:
            missing.append(c)

    contract = {
        "n_rows": int(len(df)),
        "n_frauds": int(df["is_fraud"].sum()) if "is_fraud" in df.columns else None,
        "input_path": str(input_path),
        "artifact_path": str(artifact_path),
        "base_col": BASE_COL,
        "missing": missing,
        "contract_ok": not missing,
        "constraints": {
            "min_fp_removed": args.min_fp_removed,
            "max_rules": args.max_rules,
            "max_combo_size": args.max_combo_size,
            "min_temporal_splits": args.min_temporal_splits,
            "min_months": args.min_months,
            "require_nontrain_support": args.require_nontrain_support,
            "max_seconds": args.max_seconds,
        },
    }
    dump_json(contract, out / "01_input_contract.json")
    if missing:
        raise RuntimeError(f"Contrato falhou: {missing}")

    base_pred = df[BASE_COL].to_numpy(dtype=int)
    base_metrics = metrics(y, base_pred)
    expected_metrics = artifact.get("frozen_metrics") or EXPECTED_BASE
    wl_base, wh_base = wilson(base_metrics["tp"], int(y.sum()))
    base_validation = {
        "expected_base_metrics": expected_metrics,
        "actual_base_metrics": base_metrics,
        "base_metrics_match_artifact": all(base_metrics.get(k) == expected_metrics.get(k) for k in ["tp", "fp", "fn"]),
        "fn_zero_preserved": base_metrics["fn"] == 0,
        "tp_expected": base_metrics["tp"] == EXPECTED_BASE["tp"],
        "fp_expected": base_metrics["fp"] == EXPECTED_BASE["fp"],
        "wilson_low": wl_base,
        "wilson_high": wh_base,
        "wilson_pass": wl_base >= EXPECTED_BASE["wilson_low_min"],
        "all_pass": bool(base_metrics["tp"] == EXPECTED_BASE["tp"] and base_metrics["fp"] == EXPECTED_BASE["fp"] and base_metrics["fn"] == EXPECTED_BASE["fn"] and wl_base >= EXPECTED_BASE["wilson_low_min"]),
        "status": "PASS_R3O_FROZEN_BASE_VALIDATED" if base_metrics["tp"] == EXPECTED_BASE["tp"] and base_metrics["fp"] == EXPECTED_BASE["fp"] and base_metrics["fn"] == EXPECTED_BASE["fn"] and wl_base >= EXPECTED_BASE["wilson_low_min"] else "FAIL_R3O_FROZEN_BASE_DIVERGENCE",
    }
    dump_json(base_validation, out / "02_base_validation.json")

    log(f"Base R3O-FROZEN metrics: {base_metrics}")
    log("[B] Minerando candidatos FP-only conservadores...")

    cands = mine_candidates(
        df,
        base_pred,
        min_fp_removed=args.min_fp_removed,
        max_combo_size=args.max_combo_size,
        top_groups_per_combo=args.top_groups_per_combo,
        min_temporal_splits=args.min_temporal_splits,
        min_months=args.min_months,
        require_nontrain_support=args.require_nontrain_support,
    )
    cand_df = pd.DataFrame(candidate_dicts(cands))
    cand_df.to_csv(out / "03_fp_candidates.csv", index=False)
    log(f"Candidatos conservadores TP0: {len(cands)}")

    final_pred, selected, frontier, stop = select_rules(
        df,
        cands,
        base_pred,
        max_rules=args.max_rules,
        max_seconds=args.max_seconds,
        min_fp_removed=args.min_fp_removed,
        min_temporal_splits=args.min_temporal_splits,
        min_months=args.min_months,
        require_nontrain_support=args.require_nontrain_support,
    )
    frontier.to_csv(out / "04_selection_frontier.csv", index=False)

    selected_df = pd.DataFrame(candidate_dicts(selected))
    selected_df.to_csv(out / "05_selected_fp_rules.csv", index=False)
    selected_df.to_csv(out / "06_rule_stability_audit.csv", index=False)

    final_metrics = metrics(y, final_pred)
    fp_removed = base_metrics["fp"] - final_metrics["fp"]
    tp_loss = base_metrics["tp"] - final_metrics["tp"]
    fn_delta = final_metrics["fn"] - base_metrics["fn"]
    wl, wh = wilson(final_metrics["tp"], int(y.sum()))

    df[FINAL_COL] = final_pred.astype(int)
    segments = segment_audit(df, base_pred, final_pred)
    segments.to_csv(out / "10_robustness_by_segment.csv", index=False)

    objective_status = "DONE_R3O_FROZEN_BASE_VALIDATED" if base_validation["all_pass"] else "DONE_R3O_FROZEN_BASE_NOT_VALIDATED"
    objective_status += "_FP_ONLY_REDUCED" if fp_removed > 0 else "_FP_ONLY_NO_GAIN"
    objective_status += "_FN_ZERO_PRESERVED" if final_metrics["fn"] == 0 and tp_loss == 0 else "_FN_ZERO_BROKEN"

    all_pass = bool(base_validation["all_pass"] and final_metrics["fn"] == 0 and tp_loss == 0 and fp_removed > 0)

    artifact_out = {
        "experiment": "EXP-014B-R3P",
        "policy_name": "r3p_residual_fp_only_conservative_reducer",
        "objective_status": objective_status,
        "source_artifact": str(artifact_path),
        "input_path": str(input_path),
        "base_col": BASE_COL,
        "final_pred_col": FINAL_COL,
        "base_r3o_frozen_metrics": base_metrics,
        "recommended_metrics": final_metrics,
        "fp_removed_vs_r3o": int(fp_removed),
        "tp_loss_vs_r3o": int(tp_loss),
        "fn_delta_vs_r3o": int(fn_delta),
        "wilson_low": wl,
        "wilson_high": wh,
        "base_validation": base_validation,
        "selected_fp_rules": selected_df.to_dict(orient="records") if not selected_df.empty else [],
        "constraints": contract["constraints"],
        "stop_reason": stop,
        "notes": [
            "No rescues, no threshold changes, no runtime call.",
            "Candidates require global TP_loss=0 and minimum temporal/month support.",
            "If this candidate improves FP, run EXP-014B-R3P-FROZEN before any promotion.",
        ],
    }
    dump_json(artifact_out, out / "07_policy_artifact_recommended.json")

    if not args.no_write_predictions:
        df.to_csv(out / "08_predictions_recommended.csv", index=False)

    summary = {
        "experiment": "EXP-014B-R3P",
        "status": "DONE",
        "objective_status": objective_status,
        "n_rows": int(len(df)),
        "n_frauds": int(df["is_fraud"].sum()),
        "input_path": str(input_path),
        "artifact_path": str(artifact_path),
        "base_validation_status": base_validation["status"],
        "base_r3o_frozen_metrics": base_metrics,
        "recommended_metrics": final_metrics,
        "fp_removed_vs_r3o": int(fp_removed),
        "tp_loss_vs_r3o": int(tp_loss),
        "fn_delta_vs_r3o": int(fn_delta),
        "n_fp_candidates": int(len(cands)),
        "n_selected_fp_rules": int(len(selected)),
        "stop_reason": stop,
        "wilson_low": wl,
        "wilson_high": wh,
        "all_pass": all_pass,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(out),
    }
    dump_json(summary, out / "00_run_summary.json")

    report = make_report(summary, base_validation, frontier, selected_df, selected_df, segments)
    (out / "09_exp014b_r3p_report.md").write_text(report, encoding="utf-8")

    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        out / "00_run_summary.json",
        out / "01_input_contract.json",
        out / "02_base_validation.json",
        out / "03_fp_candidates.csv",
        out / "04_selection_frontier.csv",
        out / "05_selected_fp_rules.csv",
        out / "06_rule_stability_audit.csv",
        out / "07_policy_artifact_recommended.json",
        out / "09_exp014b_r3p_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
