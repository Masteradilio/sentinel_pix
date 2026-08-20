#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3A — Champion Replay + Expanded Residual FP Mining

Objetivo:
  Retomar corretamente a trilha EXP-013K/013L, sem substituir o champion por
  uma política nova de threshold amplo.

  O experimento faz três coisas, nessa ordem:

  1. Reproduz o champion EXP-013K/013L no dataset pequeno:
       TP=118, FP=199, FN=6, recall=95.16%, precision=37.22%

  2. Recupera a base original pred_STRICT_RECALL95_SAFE_ONLY e tenta aplicá-la
     no dataset expandido de 1.465 fraudes:
       dados/exp014a_expanded_scored_input.csv

  3. Só se o replay expandido mantiver recall >= 95%, aplica a lógica EXP-013K
     de mineração residual FP-only sobre o dataset expandido, com:
       TP_loss global = 0
       TP_loss por bloco temporal = 0
       preservação de module_strong quando require_module_quiet=true

Por que este script existe:
  EXP-014B-R1/R2 encontraram políticas high-recall por threshold global, mas
  não eram comparáveis ao champion EXP-013K/013L. Este script volta à pergunta
  correta:

    "Quando aplicamos a política campeã original no dataset expandido,
     ela mantém recall >= 95% e qual FP/FPR ela gera?"

Entradas default:
  Pequeno/champion:
    resultados/experimentos/EXP-013K/07_selected_predictions.csv
    resultados/experimentos/EXP-013K/12_policy_artifact.json

  Expandido:
    dados/exp014a_expanded_scored_input.csv

Uso:
  python scripts/exp_014b_r3a_champion_replay_expanded.py

Se a base não for recuperada automaticamente:
  python scripts/exp_014b_r3a_champion_replay_expanded.py --base-threshold-col lgbm_r4_score --base-threshold 0.0024302950309567253

Modo diagnóstico, caso a inferência de threshold não seja perfeita:
  python scripts/exp_014b_r3a_champion_replay_expanded.py --allow-imperfect-base-recipe

Saídas:
  resultados/experimentos/EXP-014B-R3A/
    00_run_summary.json
    01_small_champion_reproduction.csv
    02_base_recovery_report.csv
    03_expanded_replay_metrics.csv
    04_frozen_rule_impact_expanded.csv
    05_residual_fp_candidates_expanded.csv
    06_frontier.csv
    07_selected_rules_expanded.csv
    08_final_metrics.csv
    09_time_block_metrics.csv
    10_wilson_recall_ci.csv
    11_bootstrap_summary.csv
    12_false_negatives.csv
    13_false_positives_sample.csv
    14_policy_artifact.json
    15_predictions.csv
    16_exp014b_r3a_report.md
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
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "dados").exists() else Path.cwd()

DEFAULT_SMALL_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013K" / "07_selected_predictions.csv"
DEFAULT_PRIOR_BASE_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013J" / "06_predictions_by_scenario.csv"
DEFAULT_POLICY = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013K" / "12_policy_artifact.json"
DEFAULT_EXPANDED_INPUT = PROJECT_ROOT / "dados" / "exp014a_expanded_scored_input.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3A"

BASE_COL = "pred_STRICT_RECALL95_SAFE_ONLY"
EXP013K_FINAL_COL = "exp013k_residual_fp_pred"

BASE_PRED_CANDIDATES = [
    "pred_STRICT_RECALL95_SAFE_ONLY",
    "exp013k_base_pred",
    "exp013h_frozen_pred",
    "exp013g_micro_pred",
    "pred_HIGH_RECALL_95",
]

SCORE_COL_CANDIDATES = [
    "lgbm_r4_score",
    "r4_score",
    "lgbm_mapped",
    "lgbm_raw",
    "score_final",
    "if_percentile",
    "if_percentile_x",
    "if_percentile_y",
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
    params: dict[str, Any]


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

    for c in BASE_PRED_CANDIDATES + [EXP013K_FINAL_COL, "exp014a_frozen_pred"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "runtime_flagged" not in df.columns:
        if "decisao" in df.columns:
            df["runtime_flagged"] = df["decisao"].astype(str).str.upper().isin({"CONFIRMAR", "BLOQUEAR"}).astype(int)
        elif "exp014a_frozen_pred" in df.columns:
            df["runtime_flagged"] = df["exp014a_frozen_pred"].astype(int)
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


def add_bins_and_guards(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    lgbm = num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0)
    ifp = num(df, ["if_percentile_x", "if_percentile_y", "if_percentile"], 0.0)
    vl = num(df, "vl_pix", 0.0)
    score_final = num(df, "score_final", 0.0)
    qtd_rec = num(df, "qtd_pix_recebidos_180d", 0.0)
    valor_rec = num(df, "valor_total_recebido_180d", 0.0)
    ratio = num(df, "ratio_valor_media_pagador_90d", 0.0)

    if "lgbm_bin" not in df.columns:
        df["lgbm_bin"] = qbin_series(lgbm, "lgbm", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1])
    if "if_bin" not in df.columns:
        df["if_bin"] = qbin_series(ifp, "if", [0.32, 0.5, 0.7, 0.85, 0.95])
    if "vl_bin" not in df.columns:
        df["vl_bin"] = qbin_series(vl, "vl", [20, 50, 100, 250, 500, 1000, 5000, 10000])
    if "score_bin" not in df.columns:
        df["score_bin"] = qbin_series(score_final, "score", [0.5, 1, 2, 3, 5, 10])
    if "qtd_rec_bin" not in df.columns:
        df["qtd_rec_bin"] = qbin_series(qtd_rec, "qtdrec", [0, 1, 2, 5, 10, 20, 50, 100])
    if "valor_rec_bin" not in df.columns:
        df["valor_rec_bin"] = qbin_series(valor_rec, "valrec", [0, 100, 500, 1000, 5000, 10000, 25000])
    if "ratio_bin" not in df.columns:
        df["ratio_bin"] = qbin_series(ratio, "ratio", [0.05, 0.1, 0.2, 0.5, 1, 2, 5])

    preserve = strong_module_preserve(df)
    df["module_quiet"] = np.where(preserve, "module_strong", "module_quiet")
    return df


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


def load_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Policy artifact não encontrado: {path}")
    policy = json.loads(path.read_text(encoding="utf-8"))
    if "selected_rules" not in policy:
        raise RuntimeError("Policy artifact não contém selected_rules.")
    return policy


def parse_params_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    try:
        return json.loads(str(raw))
    except Exception:
        return {}


def infer_base_recipe(small: pd.DataFrame, target_col: str, min_match: float) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    if target_col not in small.columns:
        return None, pd.DataFrame([{"status": "target_col_missing", "target_col": target_col}])

    y = pd.to_numeric(small[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    rows = []

    # Direct alias columns.
    for c in BASE_PRED_CANDIDATES:
        if c in small.columns and c != target_col:
            pred = pd.to_numeric(small[c], errors="coerce").fillna(0).astype(int).to_numpy()
            match = float((pred == y).mean())
            rows.append({
                "recipe_type": "alias_column",
                "score_col": c,
                "direction": None,
                "threshold": None,
                "match_rate": match,
                "exact_match": bool(match == 1.0),
                "positive_rate": float(pred.mean()),
            })

    # Score thresholds.
    for c in [c for c in SCORE_COL_CANDIDATES if c in small.columns]:
        scores = pd.to_numeric(small[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if scores.notna().sum() == 0:
            continue

        vals = sorted(set(float(v) for v in scores.dropna().unique()))
        if len(vals) > 2000:
            vals = sorted(set(float(v) for v in scores.quantile(np.linspace(0, 1, 2000)).to_numpy()))

        # Add midpoints between adjacent observed values to avoid ties.
        mids = []
        for a, b in zip(vals[:-1], vals[1:]):
            mids.append((a + b) / 2.0)
        thresholds = sorted(set(vals + mids))

        arr = scores.fillna(0.0).astype(float).to_numpy()
        for direction in ["ge", "le"]:
            best_for_col = None
            for th in thresholds:
                pred = (arr >= th).astype(int) if direction == "ge" else (arr <= th).astype(int)
                match = float((pred == y).mean())
                if best_for_col is None or match > best_for_col["match_rate"]:
                    best_for_col = {
                        "recipe_type": "threshold",
                        "score_col": c,
                        "direction": direction,
                        "threshold": float(th),
                        "match_rate": match,
                        "exact_match": bool(match == 1.0),
                        "positive_rate": float(pred.mean()),
                    }
                    if match == 1.0:
                        break
            if best_for_col is not None:
                rows.append(best_for_col)

    report = pd.DataFrame(rows).sort_values(["exact_match", "match_rate"], ascending=[False, False]).reset_index(drop=True)
    if report.empty:
        return None, report

    best = report.iloc[0].to_dict()
    if bool(best["exact_match"]) or float(best["match_rate"]) >= min_match:
        if best["recipe_type"] == "alias_column":
            return {"type": "alias_column", "source_col": best["score_col"], "match_rate": float(best["match_rate"]), "exact_match": bool(best["exact_match"])}, report
        return {
            "type": "threshold",
            "score_col": best["score_col"],
            "direction": best["direction"],
            "threshold": float(best["threshold"]),
            "match_rate": float(best["match_rate"]),
            "exact_match": bool(best["exact_match"]),
        }, report

    return None, report


def choose_portable_base_recipe(
    recipe: dict[str, Any] | None,
    report: pd.DataFrame,
    expanded: pd.DataFrame,
    min_match_rate: float,
    allow_imperfect: bool,
) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    """
    Garante que a receita da base seja aplicável ao dataset expandido.

    O erro original ocorreu porque exp013k_base_pred existe no dataset pequeno,
    mas não no expandido. Esta função evita escolher aliases não portáveis e
    prefere threshold em score_col presente no expandido.
    """
    rows = pd.DataFrame() if report is None or report.empty else report.copy()

    # 1) existing_column/alias só é válido se a coluna existir no expandido.
    if recipe:
        rtype = recipe.get("type")
        if rtype == "existing_column" and recipe.get("col") in expanded.columns:
            return recipe, rows
        if rtype == "alias_column" and recipe.get("source_col") in expanded.columns:
            return recipe, rows
        if rtype == "threshold" and recipe.get("score_col") in expanded.columns:
            return recipe, rows

    if rows.empty:
        return None, pd.DataFrame([{
            "status": "no_recipe_report_available",
            "reason": "infer_base_recipe não retornou candidatos.",
        }])

    # 2) Marcar portabilidade.
    rows = rows.copy()
    rows["portable_to_expanded"] = False
    rows["portable_reason"] = ""

    for idx, row in rows.iterrows():
        recipe_type = str(row.get("recipe_type", ""))
        score_col = row.get("score_col")
        if recipe_type == "alias_column":
            ok = isinstance(score_col, str) and score_col in expanded.columns
            rows.loc[idx, "portable_to_expanded"] = ok
            rows.loc[idx, "portable_reason"] = "alias_exists_in_expanded" if ok else "alias_missing_in_expanded"
        elif recipe_type == "threshold":
            ok = isinstance(score_col, str) and score_col in expanded.columns
            rows.loc[idx, "portable_to_expanded"] = ok
            rows.loc[idx, "portable_reason"] = "threshold_score_exists_in_expanded" if ok else "threshold_score_missing_in_expanded"
        else:
            rows.loc[idx, "portable_reason"] = f"unsupported_recipe_type:{recipe_type}"

    # 3) Preferir threshold portável, porque é reprodutível fora do dataset pequeno.
    threshold_rows = rows[
        (rows["recipe_type"].astype(str) == "threshold")
        & (rows["portable_to_expanded"] == True)
    ].copy()

    if not threshold_rows.empty:
        threshold_rows["match_rate"] = pd.to_numeric(threshold_rows["match_rate"], errors="coerce").fillna(0.0)
        threshold_rows["exact_match_bool"] = threshold_rows["exact_match"].astype(str).str.lower().isin(["true", "1", "1.0"])
        threshold_rows = threshold_rows.sort_values(
            ["exact_match_bool", "match_rate"],
            ascending=[False, False],
        ).reset_index(drop=True)

        best = threshold_rows.iloc[0].to_dict()
        match_rate = float(best.get("match_rate", 0.0))
        exact = bool(best.get("exact_match_bool", False))

        if exact or match_rate >= min_match_rate or allow_imperfect:
            chosen = {
                "type": "threshold",
                "score_col": str(best["score_col"]),
                "direction": str(best["direction"]),
                "threshold": float(best["threshold"]),
                "match_rate": match_rate,
                "exact_match": exact,
                "chosen_by": "portable_threshold_fallback",
            }
            rows["chosen_portable_recipe"] = False
            mask = (
                (rows["recipe_type"].astype(str) == "threshold")
                & (rows["score_col"].astype(str) == str(best["score_col"]))
                & (rows["direction"].astype(str) == str(best["direction"]))
                & (pd.to_numeric(rows["threshold"], errors="coerce") == float(best["threshold"]))
            )
            rows.loc[mask, "chosen_portable_recipe"] = True
            return chosen, rows

    # 4) Alias portável só como fallback.
    alias_rows = rows[
        (rows["recipe_type"].astype(str) == "alias_column")
        & (rows["portable_to_expanded"] == True)
    ].copy()

    if not alias_rows.empty:
        alias_rows["match_rate"] = pd.to_numeric(alias_rows["match_rate"], errors="coerce").fillna(0.0)
        alias_rows["exact_match_bool"] = alias_rows["exact_match"].astype(str).str.lower().isin(["true", "1", "1.0"])
        alias_rows = alias_rows.sort_values(
            ["exact_match_bool", "match_rate"],
            ascending=[False, False],
        ).reset_index(drop=True)

        best = alias_rows.iloc[0].to_dict()
        match_rate = float(best.get("match_rate", 0.0))
        exact = bool(best.get("exact_match_bool", False))

        if exact or match_rate >= min_match_rate or allow_imperfect:
            chosen = {
                "type": "alias_column",
                "source_col": str(best["score_col"]),
                "match_rate": match_rate,
                "exact_match": exact,
                "chosen_by": "portable_alias_fallback",
            }
            rows["chosen_portable_recipe"] = False
            rows.loc[
                (rows["recipe_type"].astype(str) == "alias_column")
                & (rows["score_col"].astype(str) == str(best["score_col"])),
                "chosen_portable_recipe"
            ] = True
            return chosen, rows

    rows["chosen_portable_recipe"] = False
    return None, rows


def apply_base_recipe(df: pd.DataFrame, recipe: dict[str, Any]) -> np.ndarray:
    if recipe["type"] == "existing_column":
        c = recipe["col"]
        if c not in df.columns:
            raise RuntimeError(f"Coluna base existente ausente no expandido: {c}")
        return pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).to_numpy()

    if recipe["type"] == "alias_column":
        c = recipe["source_col"]
        if c not in df.columns:
            raise RuntimeError(f"Coluna alias da base ausente no expandido: {c}")
        return pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).to_numpy()

    if recipe["type"] == "threshold":
        c = recipe["score_col"]
        if c not in df.columns:
            raise RuntimeError(f"Coluna de score do recipe ausente no expandido: {c}")
        scores = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        if recipe["direction"] == "ge":
            return (scores >= float(recipe["threshold"])).astype(int).to_numpy()
        return (scores <= float(recipe["threshold"])).astype(int).to_numpy()

    raise RuntimeError(f"Recipe base desconhecido: {recipe}")


def get_series_for_rule_col(df: pd.DataFrame, c: str) -> pd.Series:
    if c in df.columns:
        return text(df, c)
    # add_bins_and_guards should have created the known bins; fail fast if not.
    raise RuntimeError(f"Coluna necessária para regra ausente: {c}")


def rule_mask(df: pd.DataFrame, rule: dict[str, Any], current_pred: np.ndarray, enforce_module_quiet: bool) -> np.ndarray:
    params = parse_params_json(rule.get("params_json", {}))
    if not params and isinstance(rule.get("params"), dict):
        params = rule.get("params", {})

    cols = params.get("combo_cols", [])
    vals = params.get("combo_values", [])
    require_module_quiet = bool(params.get("require_module_quiet", False))

    if not cols:
        desc = str(rule.get("description", ""))
        cols, vals = [], []
        for part in desc.split(" AND "):
            if "=" in part:
                c, v = part.split("=", 1)
                cols.append(c.strip())
                vals.append(v.strip())

    if not cols:
        raise RuntimeError(f"Não consegui parsear regra congelada: {rule}")

    mask = np.ones(len(df), dtype=bool)
    for c, v in zip(cols, vals):
        series = get_series_for_rule_col(df, c)
        mask = mask & (series.astype(str).to_numpy() == str(v))

    if enforce_module_quiet and require_module_quiet:
        if "module_quiet" not in df.columns:
            raise RuntimeError("Regra exige module_quiet, mas coluna não existe.")
        mask = mask & (df["module_quiet"].astype(str).to_numpy() == "module_quiet")

    return mask & (current_pred.astype(int) == 1)


def apply_frozen_exp013k(df: pd.DataFrame, policy: dict[str, Any], base_pred: np.ndarray, enforce_module_quiet: bool) -> tuple[np.ndarray, pd.DataFrame]:
    y = df["is_fraud"].to_numpy(dtype=int)
    pred = base_pred.astype(int).copy()
    rows = []

    for idx, rule in enumerate(policy.get("selected_rules", [])):
        mask = rule_mask(df, rule, pred, enforce_module_quiet=enforce_module_quiet)
        tp_loss = int(((y == 1) & mask).sum())
        fp_removed = int(((y == 0) & mask).sum())
        pred[mask] = 0
        rows.append({
            "rule_index": idx,
            "description": rule.get("description"),
            "tp_loss": tp_loss,
            "fp_removed": fp_removed,
            "n_removed": int(mask.sum()),
            "params_json": rule.get("params_json"),
        })

    return pred, pd.DataFrame(rows)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    for c in VETO_FEATURES_BASE:
        if c in df.columns:
            feat[c] = df[c].astype("string").fillna("<MISSING>").astype(str)
    return feat


def sanitize_id(s: str, max_len: int = 140) -> str:
    t = re.sub(r"[^A-Za-z0-9_]+", "_", str(s))
    t = re.sub(r"_+", "_", t).strip("_")
    return t[:max_len] or "rule"


def candidate_df(cands: list[VetoCandidate]) -> pd.DataFrame:
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
    } for i, c in enumerate(cands)])


def dedupe_candidates(cands: list[VetoCandidate]) -> list[VetoCandidate]:
    best: dict[bytes, VetoCandidate] = {}
    for c in cands:
        key = np.packbits(c.mask).tobytes()
        old = best.get(key)
        if old is None or (c.fp_removed, -c.tp_loss, -len(c.description)) > (old.fp_removed, -old.tp_loss, -len(old.description)):
            best[key] = c
    out = list(best.values())
    out.sort(key=lambda c: (c.tp_loss, c.block_tp_loss_max, -c.fp_removed, len(c.description)))
    return out


def mine_residual_fp_candidates(
    df: pd.DataFrame,
    current_pred: np.ndarray,
    blocks: pd.Series,
    min_fp_removed: int,
    max_combo_size: int,
    top_groups_per_combo: int,
    require_module_quiet: bool,
) -> list[VetoCandidate]:
    y = df["is_fraud"].to_numpy(dtype=int)
    pred_pos = current_pred.astype(bool)
    preserve = strong_module_preserve(df)
    feat = build_feature_frame(df)

    candidate_cols = list(feat.columns)
    base_cols = [c for c in ["value_band", "ds_tipo_chave_norm", "periodo_dia", "first_receiver_flag_real", "mbk_available_flag"] if c in feat.columns]
    bin_cols = [c for c in candidate_cols if c.endswith("_bin") or c == "module_quiet"]

    combos = []
    for r in range(2, max_combo_size + 1):
        for combo in itertools.combinations(candidate_cols, r):
            combo = list(combo)
            if not any(c in base_cols for c in combo):
                continue
            if not any(c in bin_cols for c in combo):
                continue
            if len([c for c in combo if c in bin_cols]) > 3:
                continue
            combos.append(combo)

    log(f"  combos residuais gerados={len(combos)}")
    cands: list[VetoCandidate] = []
    bvals = blocks.to_numpy()

    for combo in combos:
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

            if tp_loss != 0 or fp_removed < min_fp_removed:
                continue

            block_tp_loss_max = 0
            for b in sorted(blocks.dropna().unique()):
                bm = mask & (bvals == b)
                block_tp_loss_max = max(block_tp_loss_max, int(((y == 1) & bm).sum()))
            if block_tp_loss_max != 0:
                continue

            group_rows.append((fp_removed, key, mask, tp_loss, block_tp_loss_max))

        if not group_rows:
            continue

        group_rows.sort(key=lambda x: x[0], reverse=True)
        for fp_removed, key, mask, tp_loss, block_tp_loss_max in group_rows[:top_groups_per_combo]:
            key_tuple = key if isinstance(key, tuple) else (key,)
            vals = [str(v) for v in key_tuple]
            desc = " AND ".join([f"{c}={v}" for c, v in zip(combo, vals)])
            rid = sanitize_id(f"r3a_{len(cands):05d}_{desc}")
            cands.append(VetoCandidate(
                rule_id=rid,
                family="expanded_residual_combo_veto",
                description=desc,
                cols=combo,
                vals=vals,
                mask=mask,
                tp_loss=tp_loss,
                fp_removed=fp_removed,
                n_removed=int(mask.sum()),
                block_tp_loss_max=block_tp_loss_max,
                params={"combo_cols": combo, "combo_values": vals, "require_module_quiet": require_module_quiet},
            ))

    out = dedupe_candidates(cands)
    log(f"  candidatos residuais TP0/blocoTP0 após dedupe={len(out)}")
    return out


def search_best_vetos(
    cands: list[VetoCandidate],
    base_pred: np.ndarray,
    y: np.ndarray,
    max_candidates: int,
    beam_width: int,
    max_rules: int,
    max_seconds: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, BeamState, list[VetoCandidate], str]:
    t0 = time.perf_counter()
    usable = [c for c in cands if c.tp_loss == 0 and c.block_tp_loss_max == 0]
    usable.sort(key=lambda c: (c.fp_removed, -len(c.description)), reverse=True)
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

            next_states: dict[bytes, BeamState] = {}
            for state in states:
                last = state.rule_indices[-1] if state.rule_indices else -1
                for i in range(last + 1, len(usable)):
                    c = usable[i]
                    new_mask = state.mask | c.mask
                    if np.array_equal(new_mask, state.mask):
                        continue

                    # All candidates are TP-loss zero; union remains TP-loss zero.
                    fp_removed = int(new_mask.sum())
                    if fp_removed <= state.fp_removed:
                        continue

                    key = np.packbits(new_mask).tobytes()
                    ns = BeamState(new_mask, state.rule_indices + (i,), 0, fp_removed)
                    old = next_states.get(key)
                    if old is None or (ns.fp_removed, -len(ns.rule_indices)) > (old.fp_removed, -len(old.rule_indices)):
                        next_states[key] = ns

            if not next_states:
                stop_reason = f"no_next_states_at_depth_{depth}"
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
                    "tp_loss": 0,
                    "fp_removed": s.fp_removed,
                    "n_rules": len(s.rule_indices),
                    **m,
                    "rule_ids": "|".join(usable[i].rule_id for i in s.rule_indices),
                    "rule_descriptions": " || ".join(usable[i].description for i in s.rule_indices),
                })

            pd.DataFrame(rows).to_csv(output_dir / f"checkpoint_frontier_depth_{depth:02d}.csv", index=False)
            selected_df = candidate_df([usable[i] for i in best.rule_indices])
            selected_df.to_csv(output_dir / f"checkpoint_selected_depth_{depth:02d}.csv", index=False)

            log(f"  depth={depth}: states={len(states)}, best_fp_removed={best.fp_removed}, elapsed_s={time.perf_counter()-t0:.1f}")

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

    frontier = pd.DataFrame(rows).sort_values(["fp", "n_rules"], ascending=[True, True]).reset_index(drop=True)
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
            "dt_min": str(part["data_pix"].min().date()) if "data_pix" in part.columns and part["data_pix"].notna().any() else None,
            "dt_max": str(part["data_pix"].max().date()) if "data_pix" in part.columns and part["data_pix"].notna().any() else None,
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


def make_report(summary: dict[str, Any], reproduction: pd.DataFrame, base_recovery: pd.DataFrame, replay_metrics: pd.DataFrame, final_metrics_df: pd.DataFrame, selected_rules: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014B-R3A — Champion Replay + Expanded Residual FP Mining")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Base recipe mode: `{summary['base_recipe'].get('type') if summary.get('base_recipe') else None}`")
    lines.append(f"- Mining executed: `{summary['mining_executed']}`")
    lines.append("")
    lines.append("## Reprodução do champion pequeno")
    lines.append(reproduction.to_markdown(index=False))
    lines.append("")
    lines.append("## Recuperação da base")
    if base_recovery.empty:
        lines.append("Sem relatório de inferência.")
    else:
        lines.append(base_recovery.head(20).to_markdown(index=False))
    lines.append("")
    lines.append("## Replay expandido")
    lines.append(replay_metrics.to_markdown(index=False))
    lines.append("")
    lines.append("## Política final R3A")
    lines.append(final_metrics_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Regras residuais novas")
    if selected_rules.empty:
        lines.append("Nenhuma regra residual adicional selecionada.")
    else:
        lines.append(selected_rules[["family", "description", "tp_loss", "fp_removed", "block_tp_loss_max"]].to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretação")
    if not summary.get("base_recipe"):
        lines.append("A base original não foi recuperada com segurança. Não é correto minerar novos vetos enquanto `pred_STRICT_RECALL95_SAFE_ONLY` não for reconstruída de forma reprodutível.")
    elif not summary["expanded_replay_target_met"]:
        lines.append("A política campeã não manteve recall >=95% no expandido. O próximo passo deve ser análise dos FNs/TPs do replay, não redução de FP.")
    elif summary["fp_removed_vs_replay"] > 0:
        lines.append("A política campeã foi aplicada no expandido e a mineração residual encontrou vetos adicionais TP0/blocoTP0. O próximo passo é validação congelada EXP-014B-R3B sem nova mineração.")
    else:
        lines.append("A política campeã foi aplicada no expandido, mas nenhum veto residual adicional seguro foi encontrado. Esta pode ser a fronteira atual para essa base.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--small-input", default=str(DEFAULT_SMALL_INPUT))
    parser.add_argument("--prior-base-input", default=str(DEFAULT_PRIOR_BASE_INPUT))
    parser.add_argument("--expanded-input", default=str(DEFAULT_EXPANDED_INPUT))
    parser.add_argument("--policy-artifact", default=str(DEFAULT_POLICY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--time-blocks", type=int, default=10)
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-threshold-col", default=None)
    parser.add_argument("--base-threshold", type=float, default=None)
    parser.add_argument("--base-threshold-direction", default="ge", choices=["ge", "le"])
    parser.add_argument("--allow-imperfect-base-recipe", action="store_true")
    parser.add_argument("--base-min-match-rate", type=float, default=0.995)
    parser.add_argument("--min-fp-removed", type=int, default=25)
    parser.add_argument("--max-combo-size", type=int, default=4)
    parser.add_argument("--top-groups-per-combo", type=int, default=40)
    parser.add_argument("--max-candidates", type=int, default=400)
    parser.add_argument("--beam-width", type=int, default=180)
    parser.add_argument("--max-rules", type=int, default=8)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--no-mine", action="store_true")
    parser.add_argument("--no-write-predictions", action="store_true")
    parser.add_argument("--enforce-module-quiet", action="store_true", default=True)
    args = parser.parse_args()

    t0 = time.perf_counter()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    small_path = Path(args.small_input)
    prior_base_path = Path(args.prior_base_input)
    expanded_path = Path(args.expanded_input)
    policy_path = Path(args.policy_artifact)

    log("=" * 80)
    log("EXP-014B-R3A — Champion Replay + Expanded Residual FP Mining")
    log("=" * 80)
    log(f"Small input: {small_path}")
    log(f"Expanded input: {expanded_path}")
    log(f"Policy: {policy_path}")

    if not small_path.exists():
        raise FileNotFoundError(f"small-input não encontrado: {small_path}")
    if not expanded_path.exists():
        raise FileNotFoundError(f"expanded-input não encontrado: {expanded_path}")

    small = add_bins_and_guards(normalize_columns(pd.read_csv(small_path, low_memory=False)))
    expanded = add_bins_and_guards(normalize_columns(pd.read_csv(expanded_path, low_memory=False)))
    policy = load_policy(policy_path)

    y_small = small["is_fraud"].to_numpy(dtype=int)
    y_exp = expanded["is_fraud"].to_numpy(dtype=int)

    # 1) Champion reproduction.
    small_rows = []
    if BASE_COL in small.columns:
        small_rows.append({"policy_name": "SMALL_BASE_STRICT_RECALL95", **compute_metrics(y_small, small[BASE_COL].to_numpy(dtype=int))})
    if EXP013K_FINAL_COL in small.columns:
        small_final = small[EXP013K_FINAL_COL].to_numpy(dtype=int)
        small_rows.append({"policy_name": "SMALL_READY_EXP013K_FINAL", **compute_metrics(y_small, small_final)})

    if BASE_COL in small.columns:
        small_reapplied, small_imp = apply_frozen_exp013k(small, policy, small[BASE_COL].to_numpy(dtype=int), enforce_module_quiet=args.enforce_module_quiet)
        small_rows.append({"policy_name": "SMALL_REAPPLIED_EXP013K_POLICY", **compute_metrics(y_small, small_reapplied)})
    else:
        small_imp = pd.DataFrame()

    reproduction_df = pd.DataFrame(small_rows)
    reproduction_df.to_csv(outdir / "01_small_champion_reproduction.csv", index=False)
    if not small_imp.empty:
        small_imp.to_csv(outdir / "small_rule_reapplication_impact.csv", index=False)

    # 2) Base recovery.
    base_recipe = None
    base_recovery_df = pd.DataFrame()

    if BASE_COL in expanded.columns:
        base_recipe = {"type": "existing_column", "col": BASE_COL, "exact_match": True, "match_rate": 1.0}
        base_recovery_df = pd.DataFrame([{"recipe_type": "existing_column", "score_col": BASE_COL, "match_rate": 1.0, "exact_match": True}])
    elif args.base_threshold_col and args.base_threshold is not None:
        base_recipe = {
            "type": "threshold",
            "score_col": args.base_threshold_col,
            "direction": args.base_threshold_direction,
            "threshold": float(args.base_threshold),
            "exact_match": None,
            "match_rate": None,
            "provided_by_user": True,
        }
        base_recovery_df = pd.DataFrame([{"recipe_type": "user_threshold", "score_col": args.base_threshold_col, "direction": args.base_threshold_direction, "threshold": args.base_threshold}])
    else:
        base_source = small
        if BASE_COL not in base_source.columns and prior_base_path.exists():
            base_source = add_bins_and_guards(normalize_columns(pd.read_csv(prior_base_path, low_memory=False)))
        recipe, report = infer_base_recipe(base_source, BASE_COL, args.base_min_match_rate)
        base_recipe, base_recovery_df = choose_portable_base_recipe(
            recipe=recipe,
            report=report,
            expanded=expanded,
            min_match_rate=args.base_min_match_rate,
            allow_imperfect=args.allow_imperfect_base_recipe,
        )

    base_recovery_df.to_csv(outdir / "02_base_recovery_report.csv", index=False)

    if not base_recipe:
        summary = {
            "experiment": "EXP-014B-R3A",
            "status": "DONE",
            "objective_status": "DONE_BASE_RECIPE_NOT_RECOVERED",
            "base_recipe": None,
            "mining_executed": False,
            "message": "Não foi possível recuperar pred_STRICT_RECALL95_SAFE_ONLY no expandido com segurança. Informe --base-threshold-col/--base-threshold ou gere a coluna base no input expandido.",
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
            "output_dir": str(outdir),
        }
        dump_json(summary, outdir / "00_run_summary.json")
        pd.DataFrame().to_csv(outdir / "03_expanded_replay_metrics.csv", index=False)
        report = make_report(summary, reproduction_df, base_recovery_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        (outdir / "16_exp014b_r3a_report.md").write_text(report, encoding="utf-8")
        log(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    # 3) Expanded replay.
    expanded_base_pred = apply_base_recipe(expanded, base_recipe)
    expanded["exp014b_r3a_base_pred"] = expanded_base_pred

    expanded_frozen_pred, rule_impact = apply_frozen_exp013k(expanded, policy, expanded_base_pred, enforce_module_quiet=args.enforce_module_quiet)
    expanded["exp014b_r3a_frozen_exp013k_pred"] = expanded_frozen_pred

    replay_rows = [
        {"policy_name": "EXPANDED_BASE_REPLAY_STRICT_RECALL95", **compute_metrics(y_exp, expanded_base_pred)},
        {"policy_name": "EXPANDED_REPLAY_EXP013K_FROZEN", **compute_metrics(y_exp, expanded_frozen_pred)},
    ]
    for runtime_col in ["exp014a_frozen_pred", "exp013k_residual_fp_pred"]:
        if runtime_col in expanded.columns:
            replay_rows.insert(0, {"policy_name": f"RUNTIME_OR_EXISTING_{runtime_col}", **compute_metrics(y_exp, expanded[runtime_col].to_numpy(dtype=int))})
            break

    replay_df = pd.DataFrame(replay_rows)
    replay_df.to_csv(outdir / "03_expanded_replay_metrics.csv", index=False)
    rule_impact.to_csv(outdir / "04_frozen_rule_impact_expanded.csv", index=False)

    replay_metrics = compute_metrics(y_exp, expanded_frozen_pred)
    total_frauds = int(y_exp.sum())
    min_tp_required = int(math.ceil(args.target_recall * total_frauds))
    expanded_replay_target_met = replay_metrics["tp"] >= min_tp_required and replay_metrics["recall"] >= args.target_recall

    blocks = make_time_blocks(expanded, args.time_blocks)
    mining_executed = False
    stop_reason = None
    candidates = []
    frontier = pd.DataFrame()
    selected = []
    best = BeamState(mask=np.zeros(len(y_exp), dtype=bool), rule_indices=tuple(), tp_loss=0, fp_removed=0)
    final_pred = expanded_frozen_pred.copy()

    if expanded_replay_target_met and not args.no_mine:
        log("[1/2] Minerando residuais FP-only no expandido a partir do champion replay...")
        candidates = mine_residual_fp_candidates(
            df=expanded,
            current_pred=expanded_frozen_pred,
            blocks=blocks,
            min_fp_removed=args.min_fp_removed,
            max_combo_size=args.max_combo_size,
            top_groups_per_combo=args.top_groups_per_combo,
            require_module_quiet=True,
        )
        candidate_df(candidates).to_csv(outdir / "05_residual_fp_candidates_expanded.csv", index=False)

        log("[2/2] Beam search residual...")
        frontier, best, selected, stop_reason = search_best_vetos(
            cands=candidates,
            base_pred=expanded_frozen_pred,
            y=y_exp,
            max_candidates=args.max_candidates,
            beam_width=args.beam_width,
            max_rules=args.max_rules,
            max_seconds=args.max_seconds,
            output_dir=outdir,
        )
        mining_executed = True
        final_pred = expanded_frozen_pred.copy()
        final_pred[best.mask] = 0
    else:
        pd.DataFrame().to_csv(outdir / "05_residual_fp_candidates_expanded.csv", index=False)
        if not expanded_replay_target_met:
            stop_reason = "expanded_replay_target_recall_not_met_skip_mining"
        elif args.no_mine:
            stop_reason = "no_mine_requested"
        else:
            stop_reason = "unknown_skip_mining"

    frontier.to_csv(outdir / "06_frontier.csv", index=False)
    selected_df = candidate_df(selected)
    selected_df.to_csv(outdir / "07_selected_rules_expanded.csv", index=False)

    expanded["exp014b_r3a_final_pred"] = final_pred
    final_metrics = compute_metrics(y_exp, final_pred)
    final_df = pd.DataFrame([
        {"policy_name": "EXPANDED_REPLAY_EXP013K_FROZEN", **replay_metrics},
        {"policy_name": "EXP014B_R3A_FINAL_AFTER_ADDITIONAL_RESIDUAL_MINING", **final_metrics},
    ])
    final_df.to_csv(outdir / "08_final_metrics.csv", index=False)

    block_df = pd.concat([
        block_metrics(expanded, expanded_base_pred, blocks, "EXPANDED_BASE_REPLAY_STRICT_RECALL95"),
        block_metrics(expanded, expanded_frozen_pred, blocks, "EXPANDED_REPLAY_EXP013K_FROZEN"),
        block_metrics(expanded, final_pred, blocks, "EXP014B_R3A_FINAL"),
    ], ignore_index=True)
    block_df.to_csv(outdir / "09_time_block_metrics.csv", index=False)

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
    wilson_df.to_csv(outdir / "10_wilson_recall_ci.csv", index=False)

    boot_df = bootstrap_summary(expanded, "exp014b_r3a_final_pred", args.bootstrap_iters, args.seed, args.target_recall)
    boot_df.to_csv(outdir / "11_bootstrap_summary.csv", index=False)

    expanded[(expanded["is_fraud"] == 1) & (expanded["exp014b_r3a_final_pred"] == 0)].to_csv(outdir / "12_false_negatives.csv", index=False)
    fp = expanded[(expanded["is_fraud"] == 0) & (expanded["exp014b_r3a_final_pred"] == 1)].copy()
    if len(fp) > 5000:
        fp = fp.sample(5000, random_state=args.seed)
    fp.to_csv(outdir / "13_false_positives_sample.csv", index=False)

    if not args.no_write_predictions:
        expanded.to_csv(outdir / "15_predictions.csv", index=False)

    fp_removed_vs_replay = int(replay_metrics["fp"] - final_metrics["fp"])
    tp_loss_vs_replay = int(replay_metrics["tp"] - final_metrics["tp"])

    objective_status = "DONE"
    objective_status += "_EXPANDED_REPLAY_TARGET_MET" if expanded_replay_target_met else "_EXPANDED_REPLAY_TARGET_NOT_MET"
    objective_status += "_FINAL_TARGET_MET" if final_metrics["recall"] >= args.target_recall else "_FINAL_TARGET_NOT_MET"
    objective_status += "_FP_REDUCED" if fp_removed_vs_replay > 0 else "_FP_NOT_REDUCED"
    objective_status += "_TPLOSS0" if tp_loss_vs_replay == 0 else "_TPLOSS_GT0"
    objective_status += "_WILSON_PASS" if wilson_low >= args.target_recall else "_WILSON_NOT_PASS"

    artifact = {
        "experiment": "EXP-014B-R3A",
        "policy_name": "champion_replay_expanded_residual_fp_mining",
        "objective_status": objective_status,
        "base_recipe": base_recipe,
        "target_recall": args.target_recall,
        "min_tp_required": min_tp_required,
        "small_reproduction": reproduction_df.to_dict(orient="records"),
        "expanded_replay_metrics": replay_metrics,
        "final_metrics": final_metrics,
        "fp_removed_vs_replay": fp_removed_vs_replay,
        "tp_loss_vs_replay": tp_loss_vs_replay,
        "frozen_rule_impact_expanded": rule_impact.to_dict(orient="records"),
        "selected_new_rules": selected_df.to_dict(orient="records") if not selected_df.empty else [],
        "notes": [
            "Replays EXP-013K/013L champion instead of replacing it with broad threshold frontier.",
            "Enforces require_module_quiet from EXP-013K params_json.",
            "Additional residual mining only runs if expanded champion replay keeps recall >= 95%.",
            "If accepted, next step is frozen validation EXP-014B-R3B without mining."
        ],
    }
    dump_json(artifact, outdir / "14_policy_artifact.json")

    summary = {
        "experiment": "EXP-014B-R3A",
        "status": "DONE",
        "objective_status": objective_status,
        "small_input": str(small_path),
        "expanded_input": str(expanded_path),
        "policy_artifact": str(policy_path),
        "n_rows_expanded": int(len(expanded)),
        "n_frauds_expanded": total_frauds,
        "base_recipe": base_recipe,
        "expanded_replay_target_met": expanded_replay_target_met,
        "mining_executed": mining_executed,
        "stop_reason": stop_reason,
        "expanded_replay_metrics": replay_metrics,
        "final_metrics": final_metrics,
        "fp_removed_vs_replay": fp_removed_vs_replay,
        "tp_loss_vs_replay": tp_loss_vs_replay,
        "n_candidates": int(len(candidates)),
        "n_selected_new_rules": int(len(selected)),
        "wilson_recall_low": wilson_low,
        "wilson_recall_high": wilson_high,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(outdir),
    }
    dump_json(summary, outdir / "00_run_summary.json")

    report = make_report(summary, reproduction_df, base_recovery_df, replay_df, final_df, selected_df)
    (outdir / "16_exp014b_r3a_report.md").write_text(report, encoding="utf-8")

    log("")
    log("=" * 80)
    log("EXP-014B-R3A CONCLUÍDO")
    log("=" * 80)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        outdir / "00_run_summary.json",
        outdir / "01_small_champion_reproduction.csv",
        outdir / "02_base_recovery_report.csv",
        outdir / "03_expanded_replay_metrics.csv",
        outdir / "04_frozen_rule_impact_expanded.csv",
        outdir / "05_residual_fp_candidates_expanded.csv",
        outdir / "07_selected_rules_expanded.csv",
        outdir / "08_final_metrics.csv",
        outdir / "09_time_block_metrics.csv",
        outdir / "10_wilson_recall_ci.csv",
        outdir / "11_bootstrap_summary.csv",
        outdir / "14_policy_artifact.json",
        outdir / "16_exp014b_r3a_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
