#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3H — Residual FP Second curto sobre BALANCED_R3G

Objetivo:
  Rodar uma iteração curta e reprodutível de FP Second em cima do novo
  benchmark expandido:

      BALANCED_R3G_QUICK_FP_SECOND
      TP=1409
      FP=5520
      FN=56
      recall=96,177%
      precision=20,335%
      FPR=4,912%

  Meta:
      manter FN=56 / TP=1409 por padrão;
      tentar reduzir FP de 5520 para abaixo de 5000;
      execução curta, tipicamente 2 a 3 minutos.

Entrada preferencial:
  resultados/experimentos/EXP-014B-R3G/09_predictions.csv

Coluna base preferencial:
  exp014b_r3g_balanced_final_pred

Fallback:
  Se o CSV de predições não existir, o script tenta reconstruir a política
  usando:
    resultados/experimentos/EXP-014B-R3G/07_policy_artifact_balanced.json
  sobre:
    dados/exp014a_expanded_scored_input.csv

Uso padrão:
  python scripts/exp_014b_r3h_residual_fp_second_balanced_r3g.py

Execução rápida:
  python scripts/exp_014b_r3h_residual_fp_second_balanced_r3g.py --max-seconds 120 --max-rules 5 --max-candidates 500 --beam-width 140

Execução um pouco mais profunda, ainda curta:
  python scripts/exp_014b_r3h_residual_fp_second_balanced_r3g.py --max-seconds 240 --max-rules 8 --max-candidates 900 --beam-width 220 --max-combo-size 4

Opcional, se aceitar perder no máximo 1 TP apenas com troca excelente:
  python scripts/exp_014b_r3h_residual_fp_second_balanced_r3g.py --max-tp-loss 1 --min-fp-per-tp 500

Saídas:
  resultados/experimentos/EXP-014B-R3H/
    00_run_summary.json
    01_input_contract.json
    02_base_metrics.csv
    03_residual_fp_profile.csv
    04_candidates.csv
    05_frontier.csv
    06_selected_rules.csv
    07_policy_metrics.csv
    08_time_block_metrics.csv
    09_wilson_recall_ci.csv
    10_false_negatives.csv
    11_false_positives_sample.csv
    12_policy_artifact.json
    13_predictions.csv
    14_exp014b_r3h_report.md
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
DEFAULT_R3G_PREDICTIONS = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3G" / "09_predictions.csv"
DEFAULT_R3G_POLICY = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3G" / "07_policy_artifact_balanced.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3H"

BASE_COL_DEFAULT = "exp014b_r3g_balanced_final_pred"
FINAL_COL = "exp014b_r3h_final_pred"

R3G_BENCHMARK = {
    "source": "BALANCED_R3G_QUICK_FP_SECOND",
    "tp": 1409,
    "fp": 5520,
    "fn": 56,
    "recall": 0.96177474,
    "precision": 0.20334825,
    "fpr": 0.0491195,
}

SMALL_BENCHMARK = {
    "source": "SMALL_REAPPLIED_EXP013K_POLICY",
    "tp": 118,
    "fp": 199,
    "fn": 6,
    "recall": 0.9516,
    "precision": 0.3722,
    "fpr": 0.0202,
}

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

# Numeric candidates are precision vetoes over the currently-alerted set.
# "le" means low values are suspicious as false-positive region.
# "ge" means very high receiver-history/ratio-like values may be false-positive region.
NUMERIC_VETO_COLS = {
    "lgbm_r4_score": "le",
    "lgbm_mapped": "le",
    "lgbm_raw": "le",
    "score_final": "le",
    "if_percentile": "le",
    "if_percentile_x": "le",
    "if_percentile_y": "le",
    "vl_pix": "le",
    "ratio_valor_media_pagador_90d": "ge",
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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

    if "lgbm_bin" not in df.columns and pick_col(df, ["lgbm_r4_score", "lgbm_mapped", "lgbm_raw"]):
        df["lgbm_bin"] = qbin_series(num(df, ["lgbm_r4_score", "lgbm_mapped", "lgbm_raw"], 0.0), "lgbm", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1])
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


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) / n) + (z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


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
    bvals = blocks.to_numpy()
    rows = []
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


def apply_dp_recipe(df: pd.DataFrame, recipe: dict[str, Any]) -> np.ndarray:
    if recipe.get("type") != "global_recall_budget_dp":
        raise RuntimeError(f"Recipe não suportada: {recipe.get('type')}")
    pred = np.zeros(len(df), dtype=int)
    segment_cols = recipe["segment_cols"]
    scores_cache: dict[str, np.ndarray] = {}

    for seg in recipe["segments"]:
        mask = np.ones(len(df), dtype=bool)
        for c in segment_cols:
            val = str(seg["segment_values"][c])
            if c not in df.columns:
                return np.zeros(len(df), dtype=int)
            mask = mask & (df[c].astype("string").fillna("<MISSING>").astype(str).to_numpy() == val)

        direction = seg.get("direction")
        if direction == "none":
            continue
        if direction == "all":
            pred[mask] = 1
            continue

        col = seg["score_col"]
        if col not in scores_cache:
            scores_cache[col] = num(df, col, 0.0).to_numpy(dtype=float)
        pred[mask] = (scores_cache[col][mask] >= float(seg["threshold"])).astype(int)

    return pred


def rule_mask(df: pd.DataFrame, rule: dict[str, Any], current_pred: np.ndarray) -> np.ndarray:
    params_raw = rule.get("params_json") or rule.get("params") or "{}"
    if isinstance(params_raw, dict):
        params = params_raw
    else:
        params = json.loads(str(params_raw).replace("Infinity", "1e999"))

    mask = np.ones(len(df), dtype=bool)
    rtype = params.get("type")

    if rtype == "combo":
        for c, v in zip(params.get("combo_cols", []), params.get("combo_values", [])):
            if c not in df.columns:
                return np.zeros(len(df), dtype=bool)
            mask = mask & (df[c].astype("string").fillna("<MISSING>").astype(str).to_numpy() == str(v))
    elif rtype == "numeric_threshold":
        c = params.get("col")
        if c not in df.columns:
            return np.zeros(len(df), dtype=bool)
        vals = num(df, c, 0.0).to_numpy(dtype=float)
        cut = float(params.get("cut"))
        direction = params.get("direction")
        mask = mask & ((vals <= cut) if direction == "le" else (vals >= cut))
    else:
        return np.zeros(len(df), dtype=bool)

    if params.get("require_module_quiet", False) and "module_quiet" in df.columns:
        mask = mask & (df["module_quiet"].astype(str).to_numpy() == "module_quiet")

    return mask & (current_pred.astype(int) == 1)


def apply_rules(df: pd.DataFrame, base_pred: np.ndarray, rules: list[dict[str, Any]]) -> tuple[np.ndarray, pd.DataFrame]:
    y = df["is_fraud"].to_numpy(dtype=int)
    pred = base_pred.copy().astype(int)
    rows = []
    for i, rule in enumerate(rules):
        m = rule_mask(df, rule, pred)
        tp_loss = int(((y == 1) & m).sum())
        fp_removed = int(((y == 0) & m).sum())
        pred[m] = 0
        rows.append({
            "rule_index": i,
            "rule_id": rule.get("rule_id"),
            "family": rule.get("family"),
            "description": rule.get("description"),
            "tp_loss": tp_loss,
            "fp_removed": fp_removed,
            "n_removed": int(m.sum()),
            "params_json": rule.get("params_json"),
        })
    return pred, pd.DataFrame(rows)


def load_or_reconstruct_input(predictions_path: Path, input_path: Path, policy_path: Path, base_col: str) -> tuple[pd.DataFrame, str]:
    if predictions_path.exists():
        df = add_bins_and_guards(normalize_columns(pd.read_csv(predictions_path, low_memory=False)))
        if base_col in df.columns:
            df[base_col] = pd.to_numeric(df[base_col], errors="coerce").fillna(0).astype(int)
            return df, "loaded_r3g_predictions"

    if not input_path.exists():
        raise FileNotFoundError(f"Input base não encontrado: {input_path}")
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy artifact R3G não encontrado: {policy_path}")

    df = add_bins_and_guards(normalize_columns(pd.read_csv(input_path, low_memory=False)))
    policy = load_json(policy_path)
    base_pred = apply_dp_recipe(df, policy["recipe"])
    final_pred, _ = apply_rules(df, base_pred, policy.get("selected_rules", []))
    df[base_col] = final_pred.astype(int)
    return df, "reconstructed_from_r3g_policy_artifact"


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
    quantiles: list[float],
) -> list[VetoCandidate]:
    y = df["is_fraud"].to_numpy(dtype=int)
    pred_pos = base_pred.astype(bool)
    if require_module_quiet and "module_quiet" in df.columns:
        pred_pos = pred_pos & (df["module_quiet"].astype(str).to_numpy() == "module_quiet")

    out: list[VetoCandidate] = []
    fixed_cuts = {
        "lgbm_r4_score": [0.02, 0.03, 0.04, 0.05, 0.075, 0.1],
        "lgbm_mapped": [0.02, 0.03, 0.04, 0.05, 0.075, 0.1],
        "lgbm_raw": [0.02, 0.03, 0.04, 0.05, 0.075, 0.1],
        "score_final": [0.5, 1, 2, 3, 5, 10],
        "if_percentile": [0.32, 0.5, 0.7, 0.85],
        "if_percentile_x": [0.32, 0.5, 0.7, 0.85],
        "if_percentile_y": [0.32, 0.5, 0.7, 0.85],
        "vl_pix": [20, 50, 100, 250, 500, 1000],
        "ratio_valor_media_pagador_90d": [0.5, 1, 2, 5, 10],
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
            cuts.extend([float(x) for x in np.quantile(pp, quantiles)])
        except Exception:
            pass
        cuts = sorted(set(float(x) for x in cuts if np.isfinite(x)))

        for cut in cuts:
            if direction == "le":
                mask = pred_pos & (vals <= cut)
                desc = f"{col}<={cut:g}"
            else:
                mask = pred_pos & (vals >= cut)
                desc = f"{col}>={cut:g}"

            add_candidate(
                out, "r3h_num", "residual_numeric_veto", desc, mask, y, blocks,
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

    combos: list[list[str]] = []
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
                rule_id=f"r3h_combo_{len(out):05d}",
                family="residual_microsegment_veto",
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
        k = np.packbits(c.mask).tobytes()
        old = best.get(k)
        if old is None or (c.fp_removed, -c.tp_loss, -len(c.description)) > (old.fp_removed, -old.tp_loss, -len(old.description)):
            best[k] = c
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
    max_tp_loss: int,
    max_candidates: int,
    beam_width: int,
    max_rules: int,
    max_seconds: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, State, list[VetoCandidate], str]:
    t0 = time.perf_counter()
    usable = [c for c in cands if c.tp_loss <= max_tp_loss]
    usable.sort(key=lambda c: (c.tp_loss > 0, -c.fp_removed if c.tp_loss == 0 else -c.fp_per_tp, -c.fp_removed))
    usable = usable[:max_candidates]

    fraud_idx = np.where(y == 1)[0]
    zero_loss_mode = (max_tp_loss == 0 and all(c.tp_loss == 0 for c in usable))

    pending_limit = max(beam_width * 8, 1000)
    pending_keep = max(beam_width * 4, 500)

    def rank_state(s: State):
        return (s.fp_removed, -s.tp_loss, -len(s.rule_indices))

    def prune_pending(d: dict[bytes, State], keep: int) -> dict[bytes, State]:
        if len(d) <= keep:
            return d
        return dict(sorted(d.items(), key=lambda kv: rank_state(kv[1]), reverse=True)[:keep])

    initial = State(np.zeros(len(y), dtype=bool), tuple(), 0, 0)
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
                        if tp_loss > max_tp_loss:
                            continue
                        fp_removed = total - tp_loss

                    if fp_removed <= state.fp_removed:
                        continue

                    ns = State(new_mask, state.rule_indices + (i,), tp_loss, fp_removed)
                    key = np.packbits(new_mask).tobytes()
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
                if not stop_reason.startswith("max_seconds"):
                    stop_reason = f"no_next_states_at_depth_{depth}"
                break

            states = sorted(next_states.values(), key=rank_state, reverse=True)[:beam_width]
            if rank_state(states[0]) > rank_state(best):
                best = states[0]

            for s in states[:50]:
                pred = base_pred.copy()
                pred[s.mask] = 0
                rows.append({
                    "depth": depth,
                    "tp_loss": s.tp_loss,
                    "fp_removed": s.fp_removed,
                    "n_rules": len(s.rule_indices),
                    **compute_metrics(y, pred),
                    "rule_ids": "|".join(usable[i].rule_id for i in s.rule_indices),
                    "rule_descriptions": " || ".join(usable[i].description for i in s.rule_indices),
                })

            pd.DataFrame(rows).to_csv(output_dir / f"checkpoint_frontier_depth_{depth:02d}.csv", index=False)
            log(
                f"  depth={depth}: best_fp_removed={best.fp_removed}, tp_loss={best.tp_loss}/{max_tp_loss}, "
                f"states={len(states)}, expansions={expansions}, prunes={prunes}, depth_s={time.perf_counter()-depth_t0:.1f}"
            )

            if time.perf_counter() - t0 >= max_seconds:
                stop_reason = f"max_seconds_after_depth_{depth}"
                break

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt_saved_best"
        log("KeyboardInterrupt capturado; salvando melhor estado.")

    if not rows:
        rows = [{
            "depth": 0,
            "tp_loss": 0,
            "fp_removed": 0,
            "n_rules": 0,
            **compute_metrics(y, base_pred),
            "rule_ids": "",
            "rule_descriptions": "",
        }]

    frontier = pd.DataFrame(rows).sort_values(["fp", "fn"], ascending=[True, True]).reset_index(drop=True)
    selected = [usable[i] for i in best.rule_indices]
    return frontier, best, selected, stop_reason


def build_residual_profile(df: pd.DataFrame, base_pred: np.ndarray, output_dir: Path) -> pd.DataFrame:
    fp_df = df[(df["is_fraud"] == 0) & (base_pred == 1)].copy()
    rows = []
    for c in FEATURE_COLS:
        if c not in fp_df.columns:
            continue
        vc = fp_df[c].astype("string").fillna("<MISSING>").value_counts().head(20)
        for val, n in vc.items():
            rows.append({"feature": c, "value": str(val), "fp_count": int(n)})
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "03_residual_fp_profile.csv", index=False)
    return out


def make_report(summary: dict[str, Any], rules_df: pd.DataFrame, metrics_df: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014B-R3H — Residual FP Second curto sobre BALANCED_R3G")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Input mode: `{summary['input_mode']}`")
    lines.append(f"- Base: `{summary['base_metrics']}`")
    lines.append(f"- Final: `{summary['final_metrics']}`")
    lines.append(f"- FP removidos: `{summary['fp_removed_vs_base']}`")
    lines.append(f"- TP loss: `{summary['tp_loss_vs_base']}`")
    lines.append(f"- Stop reason: `{summary['stop_reason']}`")
    lines.append("")
    lines.append("## Métricas comparativas")
    lines.append(metrics_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Regras selecionadas")
    if rules_df.empty:
        lines.append("Nenhuma regra selecionada.")
    else:
        cols = ["rule_id", "family", "description", "tp_loss", "fp_removed", "block_tp_loss_max", "fp_per_tp"]
        lines.append(rules_df[[c for c in cols if c in rules_df.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Próximo passo")
    if summary["final_metrics"]["fp"] < 5000 and summary["tp_loss_vs_base"] == 0:
        lines.append("Marco FP<5000 atingido sem aumentar FN. Próximo passo recomendado: validação congelada curta e auditoria dos 56 FNs.")
    elif summary["fp_removed_vs_base"] > 0:
        lines.append("Houve ganho incremental. Próximo passo: repetir apenas se houver hipótese nova; caso contrário, iniciar auditoria dos 56 FNs.")
    else:
        lines.append("Sem ganho relevante. Próximo passo: auditoria dos 56 FNs ou hard-negative mining.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--predictions", default=str(DEFAULT_R3G_PREDICTIONS))
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--policy", default=str(DEFAULT_R3G_POLICY))
    parser.add_argument("--base-col", default=BASE_COL_DEFAULT)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-tp-loss", type=int, default=0)
    parser.add_argument("--min-fp-removed", type=int, default=25)
    parser.add_argument("--max-combo-size", type=int, default=4)
    parser.add_argument("--top-groups-per-combo", type=int, default=50)
    parser.add_argument("--max-block-tp-loss", type=int, default=0)
    parser.add_argument("--min-fp-per-tp", type=float, default=500.0)
    parser.add_argument("--max-candidates", type=int, default=800)
    parser.add_argument("--beam-width", type=int, default=180)
    parser.add_argument("--max-rules", type=int, default=8)
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--time-blocks", type=int, default=10)
    parser.add_argument("--require-module-quiet", action="store_true", default=True)
    parser.add_argument("--quantiles", default="0.03,0.05,0.10,0.20,0.30,0.50,0.70")
    parser.add_argument("--no-write-predictions", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014B-R3H — Residual FP Second curto sobre BALANCED_R3G")
    log("=" * 80)

    df, input_mode = load_or_reconstruct_input(
        predictions_path=Path(args.predictions),
        input_path=Path(args.input),
        policy_path=Path(args.policy),
        base_col=args.base_col,
    )

    if args.base_col not in df.columns:
        raise RuntimeError(f"Coluna base ausente: {args.base_col}")

    df[args.base_col] = pd.to_numeric(df[args.base_col], errors="coerce").fillna(0).astype(int)
    base_pred = df[args.base_col].to_numpy(dtype=int)
    y = df["is_fraud"].to_numpy(dtype=int)
    blocks = make_time_blocks(df, args.time_blocks)

    base_metrics = compute_metrics(y, base_pred)
    wl_base, wh_base = wilson_ci(base_metrics["tp"], int(y.sum()))

    contract = {
        "input_mode": input_mode,
        "n_rows": int(len(df)),
        "n_frauds": int(y.sum()),
        "base_col": args.base_col,
        "base_metrics": base_metrics,
        "base_wilson_low": wl_base,
        "base_wilson_high": wh_base,
        "required_cols_present": {
            "is_fraud": "is_fraud" in df.columns,
            args.base_col: args.base_col in df.columns,
        },
    }
    dump_json(contract, output_dir / "01_input_contract.json")
    pd.DataFrame([{"policy_name": "R3G_BALANCED_BASE_FOR_R3H", **base_metrics}]).to_csv(output_dir / "02_base_metrics.csv", index=False)

    log(f"Input mode: {input_mode}")
    log(f"Base metrics: {base_metrics}")

    build_residual_profile(df, base_pred, output_dir)

    q = [float(x.strip()) for x in str(args.quantiles).split(",") if x.strip()]

    allowed_tp_loss = int(args.max_tp_loss)
    log("[1/4] Minerando candidatos residuais...")
    num_cands = mine_numeric_candidates(
        df=df,
        base_pred=base_pred,
        blocks=blocks,
        allowed_tp_loss=allowed_tp_loss,
        min_fp_removed=args.min_fp_removed,
        max_block_tp_loss=args.max_block_tp_loss,
        min_fp_per_tp=args.min_fp_per_tp,
        require_module_quiet=args.require_module_quiet,
        quantiles=q,
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
    cdf.to_csv(output_dir / "04_candidates.csv", index=False)
    log(f"Candidatos: numeric={len(num_cands)}, combo={len(combo_cands)}, dedupe={len(cands)}")

    log("[2/4] Busca curta FP Second...")
    frontier, best, selected_rules, stop_reason = search_best_vetos(
        cands=cands,
        base_pred=base_pred,
        y=y,
        max_tp_loss=allowed_tp_loss,
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
    df[FINAL_COL] = final_pred.astype(int)
    final_metrics = compute_metrics(y, final_pred)
    wl_final, wh_final = wilson_ci(final_metrics["tp"], int(y.sum()))

    fp_removed = base_metrics["fp"] - final_metrics["fp"]
    tp_loss = base_metrics["tp"] - final_metrics["tp"]

    log(f"Final metrics: {final_metrics}")

    log("[3/4] Métricas e artefatos...")
    policy_rows = [
        {"policy_name": "R3G_BALANCED_INPUT", **base_metrics},
        {"policy_name": "EXP014B_R3H_FINAL", **final_metrics},
    ]
    for c in ["exp014b_r3g_extreme_final_pred", "exp014b_r3e_final_pred", "exp014b_r3d_final_pred"]:
        if c in df.columns:
            policy_rows.insert(0, {"policy_name": c, **compute_metrics(y, df[c].to_numpy(dtype=int))})

    metrics_df = pd.DataFrame(policy_rows)
    metrics_df["fp_gap_vs_r3g_balanced"] = metrics_df["fp"] - R3G_BENCHMARK["fp"]
    metrics_df["fn_gap_vs_r3g_balanced"] = metrics_df["fn"] - R3G_BENCHMARK["fn"]
    metrics_df["fpr_gap_vs_small"] = metrics_df["fpr"] - SMALL_BENCHMARK["fpr"]
    metrics_df["precision_gap_vs_small"] = metrics_df["precision"] - SMALL_BENCHMARK["precision"]
    metrics_df.to_csv(output_dir / "07_policy_metrics.csv", index=False)

    block_df = pd.concat([
        block_metrics(df, base_pred, blocks, "R3G_BALANCED_INPUT"),
        block_metrics(df, final_pred, blocks, "EXP014B_R3H_FINAL"),
    ], ignore_index=True)
    block_df.to_csv(output_dir / "08_time_block_metrics.csv", index=False)

    wilson_df = pd.DataFrame([
        {
            "policy_name": "R3G_BALANCED_INPUT",
            "tp": base_metrics["tp"],
            "n_frauds": int(y.sum()),
            "recall": base_metrics["recall"],
            "wilson_low": wl_base,
            "wilson_high": wh_base,
        },
        {
            "policy_name": "EXP014B_R3H_FINAL",
            "tp": final_metrics["tp"],
            "n_frauds": int(y.sum()),
            "recall": final_metrics["recall"],
            "wilson_low": wl_final,
            "wilson_high": wh_final,
        },
    ])
    wilson_df.to_csv(output_dir / "09_wilson_recall_ci.csv", index=False)

    df[(df["is_fraud"] == 1) & (df[FINAL_COL] == 0)].to_csv(output_dir / "10_false_negatives.csv", index=False)
    fp_df = df[(df["is_fraud"] == 0) & (df[FINAL_COL] == 1)].copy()
    if len(fp_df) > 5000:
        fp_df = fp_df.sample(5000, random_state=42)
    fp_df.to_csv(output_dir / "11_false_positives_sample.csv", index=False)

    artifact = {
        "experiment": "EXP-014B-R3H",
        "policy_name": "residual_fp_second_balanced_r3g",
        "source_base": "BALANCED_R3G_QUICK_FP_SECOND",
        "input_mode": input_mode,
        "base_col": args.base_col,
        "final_col": FINAL_COL,
        "base_metrics": base_metrics,
        "final_metrics": final_metrics,
        "fp_removed_vs_base": int(fp_removed),
        "tp_loss_vs_base": int(tp_loss),
        "wilson_low": wl_final,
        "wilson_high": wh_final,
        "stop_reason": stop_reason,
        "selected_rules": rules_df.to_dict(orient="records") if not rules_df.empty else [],
        "constraints": {
            "max_tp_loss": args.max_tp_loss,
            "max_block_tp_loss": args.max_block_tp_loss,
            "min_fp_per_tp": args.min_fp_per_tp,
            "max_seconds": args.max_seconds,
        },
        "notes": [
            "Short residual FP Second run over BALANCED_R3G.",
            "Default preserves TP/FN exactly via max_tp_loss=0.",
            "Not a promotion until frozen validation confirms stability."
        ],
    }
    dump_json(artifact, output_dir / "12_policy_artifact.json")

    if not args.no_write_predictions:
        df.to_csv(output_dir / "13_predictions.csv", index=False)

    objective_status = "DONE"
    objective_status += "_FP_REDUCED" if fp_removed > 0 else "_FP_NOT_REDUCED"
    objective_status += "_TPLOSS0" if tp_loss == 0 else "_TP_LOSS"
    objective_status += "_BELOW_5000FP" if final_metrics["fp"] < 5000 else "_FP_GE_5000"
    objective_status += "_WILSON_PASS_95" if wl_final >= 0.95 else "_WILSON_NOT_PASS_95"

    summary = {
        "experiment": "EXP-014B-R3H",
        "status": "DONE",
        "objective_status": objective_status,
        "input_mode": input_mode,
        "n_rows": int(len(df)),
        "n_frauds": int(y.sum()),
        "base_col": args.base_col,
        "base_metrics": base_metrics,
        "final_metrics": final_metrics,
        "fp_removed_vs_base": int(fp_removed),
        "tp_loss_vs_base": int(tp_loss),
        "wilson_recall_low": wl_final,
        "wilson_recall_high": wh_final,
        "n_numeric_candidates": int(len(num_cands)),
        "n_combo_candidates": int(len(combo_cands)),
        "n_candidates": int(len(cands)),
        "n_selected_rules": int(len(selected_rules)),
        "stop_reason": stop_reason,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, rules_df, metrics_df)
    (output_dir / "14_exp014b_r3h_report.md").write_text(report, encoding="utf-8")

    log("[4/4] Concluído.")
    log(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
