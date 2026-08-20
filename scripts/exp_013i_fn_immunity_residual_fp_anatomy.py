#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-013I — FN Immunity & Residual FP Anatomy

Objetivo:
  Fazer diagnóstico cirúrgico da política congelada vencedora do EXP-013H:
      high_recall95_micro_refined_policy

Perguntas:
  1. Quem são os 6 FNs atuais e quais sinais precisamos preservar?
  2. Como são os 414 FPs remanescentes?
  3. Ainda existem microvetos com TP_loss=0 para remover FP?
  4. Quais módulos/sinais extra-LGBM ajudam a proteger FNs ou vetar FPs?
  5. Quais microações candidatas podem alimentar o EXP-013J sem busca ampla?

Princípio:
  Diagnóstico primeiro, ação depois.
  Este script NÃO promove política e NÃO faz busca combinatória pesada.

Entrada default:
  resultados/experimentos/EXP-013H/05_frozen_predictions.csv

Coluna de predição usada:
  --pred-col exp013h_frozen_pred

Uso:
  python scripts/exp_013i_fn_immunity_residual_fp_anatomy.py

Saídas:
  resultados/experimentos/EXP-013I/
    00_run_summary.json
    01_current_policy_metrics.csv
    02_false_negatives_profile.csv
    03_false_positives_profile_sample.csv
    04_numeric_error_anatomy.csv
    05_segment_positive_anatomy.csv
    06_safe_veto_candidates_tp0.csv
    07_near_safe_veto_candidates_tp1.csv
    08_fn_immunity_candidates.csv
    09_module_signal_matrix.csv
    10_recommended_next_actions.csv
    11_fn_immunity_residual_fp_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
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

DEFAULT_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013H" / "05_frozen_predictions.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013I"

ID_COL_CANDIDATES = ["transaction_id", "cd_pix", "id_transacao", "end_to_end_id"]

PREFERRED_NUMERIC = [
    "lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw",
    "score_final", "if_percentile_x", "if_percentile_y", "if_percentile", "if_raw",
    "se_score_x", "se_score_y", "se_score", "se_patterns_count", "se_pattern_count",
    "beh_score", "behavioral_score", "beh_factors_count", "behavioral_risk_factor_count",
    "vl_pix", "ratio_valor_media_pagador_90d", "ratio_valor_maximo_pagador_180d",
    "qtd_pix_recebidos_180d", "qtd_pix_recebidos_90d", "qtd_pix_recebidos_30d",
    "valor_total_recebido_180d", "valor_total_recebido_90d", "valor_total_recebido_30d",
    "soma_pagadores_distintos_dia_recebedor_180d",
    "qtd_pix_pagador_7d", "qtd_pix_pagador_30d", "qtd_pix_pagador_90d", "qtd_pix_pagador_180d",
    "valor_total_pagador_7d", "valor_total_pagador_30d", "valor_total_pagador_90d", "valor_total_pagador_180d",
    "mbk_completeness_score", "mbk_available_flag", "first_receiver_flag_real",
]

PREFERRED_CATEGORICAL = [
    "value_band", "ds_tipo_chave_norm", "ds_tipo_chave", "periodo_dia",
    "first_receiver_flag_real", "mbk_available_flag", "decisao", "motivo",
    "rule_name", "device_name", "metodo_autenticacao", "topaz_transacao_rejeitada",
]

SEGMENT_SETS = [
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
    ["first_receiver_flag_real", "value_band", "ds_tipo_chave_norm"],
]

SEGMENT_LGBM_THRESHOLDS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]


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


def infer_pred_col(df: pd.DataFrame, requested: str | None) -> str:
    if requested and requested in df.columns:
        return requested

    candidates = [
        "exp013h_frozen_pred",
        "exp013g_micro_pred",
        "pred_HIGH_RECALL_95",
        "exp013e_refined_pred",
        "shadow_exp012d_flagged",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise RuntimeError("Não encontrei coluna de predição. Use --pred-col.")


def add_error_labels(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    out = df.copy()
    out[pred_col] = pd.to_numeric(out[pred_col], errors="coerce").fillna(0).astype(int)

    y = out["is_fraud"].astype(int)
    p = out[pred_col].astype(int)

    out["exp013i_error_type"] = np.select(
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


def safe_cols_for_profile(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in ID_COL_CANDIDATES + ["is_fraud", "exp013i_error_type", "data_pix", "event_datetime"]:
        if c in df.columns and c not in cols:
            cols.append(c)

    for c in PREFERRED_NUMERIC + PREFERRED_CATEGORICAL:
        if c in df.columns and c not in cols:
            cols.append(c)

    return cols


def numeric_error_anatomy(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    et = df["exp013i_error_type"]

    num_cols = []
    for c in PREFERRED_NUMERIC:
        if c in df.columns and c not in num_cols:
            num_cols.append(c)

    for c in df.columns:
        if c in num_cols or c in {"is_fraud"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if c.lower().endswith("_id") or "cpf" in c.lower() or "cnpj" in c.lower():
                continue
            if pd.to_numeric(df[c], errors="coerce").nunique(dropna=True) > 1:
                num_cols.append(c)

    groups = ["TP", "FP", "FN", "TN"]

    for c in num_cols:
        vals = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if vals.nunique(dropna=True) <= 1:
            continue

        row = {"feature": c}
        for g in groups:
            x = vals[et == g].dropna().astype(float)
            row[f"n_{g}"] = int(len(x))
            row[f"mean_{g}"] = float(x.mean()) if len(x) else np.nan
            row[f"median_{g}"] = float(x.median()) if len(x) else np.nan
            row[f"p10_{g}"] = float(x.quantile(0.10)) if len(x) else np.nan
            row[f"p90_{g}"] = float(x.quantile(0.90)) if len(x) else np.nan

        # Heuristic useful scores:
        row["fn_vs_fp_median_gap"] = (
            row.get("median_FN", np.nan) - row.get("median_FP", np.nan)
            if pd.notna(row.get("median_FN", np.nan)) and pd.notna(row.get("median_FP", np.nan)) else np.nan
        )
        row["tp_vs_fp_median_gap"] = (
            row.get("median_TP", np.nan) - row.get("median_FP", np.nan)
            if pd.notna(row.get("median_TP", np.nan)) and pd.notna(row.get("median_FP", np.nan)) else np.nan
        )

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["abs_tp_vs_fp_gap"] = out["tp_vs_fp_median_gap"].abs()
    out["abs_fn_vs_fp_gap"] = out["fn_vs_fp_median_gap"].abs()
    return out.sort_values(["abs_tp_vs_fp_gap", "abs_fn_vs_fp_gap"], ascending=[False, False]).reset_index(drop=True)


def segment_positive_anatomy(df: pd.DataFrame, pred_col: str, min_n: int) -> pd.DataFrame:
    rows = []
    pred_pos = df[pred_col].astype(int) == 1
    y = df["is_fraud"].astype(int).to_numpy()

    for cols in SEGMENT_SETS:
        if any(c not in df.columns for c in cols):
            continue

        tmp = pd.DataFrame(index=df.index)
        for c in cols:
            tmp[c] = text(df, c)

        grouped = tmp[pred_pos].groupby(cols, dropna=False).indices

        for key, idxs_rel in grouped.items():
            idxs = tmp[pred_pos].iloc[list(idxs_rel)].index.to_numpy(dtype=int)
            if len(idxs) < min_n:
                continue

            tp = int(((y == 1) & pred_pos.to_numpy() & np.isin(np.arange(len(df)), idxs)).sum())
            fp = int(((y == 0) & pred_pos.to_numpy() & np.isin(np.arange(len(df)), idxs)).sum())

            key_tuple = key if isinstance(key, tuple) else (key,)
            rows.append({
                "segment_cols": "|".join(cols),
                "segment_key": " AND ".join([f"{c}={v}" for c, v in zip(cols, key_tuple)]),
                "n_pred_pos": int(len(idxs)),
                "tp": tp,
                "fp": fp,
                "precision": float(tp / max(tp + fp, 1)),
                "fp_share_of_all_fp": float(fp / max(((df["exp013i_error_type"] == "FP").sum()), 1)),
                "tp_loss_if_veto": tp,
                "fp_removed_if_veto": fp,
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values(["tp_loss_if_veto", "fp_removed_if_veto"], ascending=[True, False]).reset_index(drop=True)


def candidate_vetos(df: pd.DataFrame, pred_col: str, min_fp_removed: int, max_tp_loss_near: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_pos = df[pred_col].astype(int).to_numpy() == 1
    y = df["is_fraud"].astype(int).to_numpy()
    lgbm = num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0).to_numpy(dtype=float)

    rows = []

    def add_candidate(family: str, description: str, mask: np.ndarray, params: dict[str, Any]) -> None:
        effective = np.asarray(mask, dtype=bool) & pred_pos
        if not effective.any():
            return
        tp_loss = int(((y == 1) & effective).sum())
        fp_removed = int(((y == 0) & effective).sum())
        if fp_removed < min_fp_removed:
            return
        recall_after = (int(((y == 1) & pred_pos).sum()) - tp_loss) / max(int((y == 1).sum()), 1)
        rows.append({
            "family": family,
            "description": description,
            "tp_loss": tp_loss,
            "fp_removed": fp_removed,
            "n_removed": int(effective.sum()),
            "recall_after": float(recall_after),
            "params_json": json.dumps(params, ensure_ascii=False),
        })

    # Segment-only and segment+LGBM candidates.
    for cols in SEGMENT_SETS:
        if any(c not in df.columns for c in cols):
            continue

        tmp = pd.DataFrame(index=df.index)
        for c in cols:
            tmp[c] = text(df, c)

        grouped = tmp[pred_pos].groupby(cols, dropna=False).indices
        for key, idxs_rel in grouped.items():
            idxs = tmp[pred_pos].iloc[list(idxs_rel)].index.to_numpy(dtype=int)
            if len(idxs) < min_fp_removed:
                continue

            base_mask = np.zeros(len(df), dtype=bool)
            base_mask[idxs] = True

            key_tuple = key if isinstance(key, tuple) else (key,)
            desc = " AND ".join([f"{c}={v}" for c, v in zip(cols, key_tuple)])

            add_candidate(
                "segment_veto",
                desc,
                base_mask,
                {"segment_cols": cols, "segment_values": [str(v) for v in key_tuple]},
            )

            for th in SEGMENT_LGBM_THRESHOLDS:
                add_candidate(
                    "segment_lgbm_veto",
                    f"{desc} AND lgbm<{th}",
                    base_mask & (lgbm < th),
                    {"segment_cols": cols, "segment_values": [str(v) for v in key_tuple], "lgbm_lt": th},
                )

    # Numeric residual candidates with TP=0 likely. Keep simple and diagnostic.
    specs = [
        ("lgbm_r4_score", num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0), "lt", [0.003, 0.005, 0.01, 0.02, 0.03]),
        ("if_percentile", num(df, ["if_percentile_x", "if_percentile_y", "if_percentile"], 0.0), "lt", [0.32, 0.5, 0.7]),
        ("vl_pix", num(df, "vl_pix", 0.0), "lt", [20, 50, 100, 250]),
        ("qtd_pix_recebidos_180d", num(df, "qtd_pix_recebidos_180d", 0.0), "gt", [20, 50, 100]),
        ("valor_total_recebido_180d", num(df, "valor_total_recebido_180d", 0.0), "gt", [5000, 10000, 25000]),
    ]

    for feat, vals, op, thresholds in specs:
        v = vals.to_numpy(dtype=float)
        for th in thresholds:
            mask = (v < th) if op == "lt" else (v > th)
            add_candidate(
                "numeric_veto",
                f"{feat}{'<' if op == 'lt' else '>'}{th}",
                mask,
                {"feature": feat, "op": op, "threshold": th},
            )

    if not rows:
        empty = pd.DataFrame()
        return empty, empty

    allc = pd.DataFrame(rows).drop_duplicates(subset=["family", "description"]).sort_values(["tp_loss", "fp_removed"], ascending=[True, False]).reset_index(drop=True)
    safe = allc[allc["tp_loss"] == 0].copy().reset_index(drop=True)
    near = allc[(allc["tp_loss"] > 0) & (allc["tp_loss"] <= max_tp_loss_near)].copy().reset_index(drop=True)
    return safe, near


def fn_immunity_candidates(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    fn = df[df["exp013i_error_type"] == "FN"].copy()
    fp = df[df["exp013i_error_type"] == "FP"].copy()
    tp = df[df["exp013i_error_type"] == "TP"].copy()

    rows = []
    if fn.empty:
        return pd.DataFrame()

    # Numeric ranges covering all FNs; estimate how many FPs/TPs would be preserved if used as immunity.
    numeric_specs = []
    for c in PREFERRED_NUMERIC:
        if c in df.columns:
            numeric_specs.append(c)

    for c in numeric_specs:
        vals_fn = pd.to_numeric(fn[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if vals_fn.empty:
            continue

        lo = float(vals_fn.min())
        hi = float(vals_fn.max())

        vals_all = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        mask = vals_all.between(lo, hi, inclusive="both").fillna(False).to_numpy(dtype=bool)

        rows.append({
            "family": "numeric_fn_range",
            "description": f"preserve if {c} between {lo:.8g} and {hi:.8g}",
            "feature": c,
            "fn_covered": int(((df["exp013i_error_type"] == "FN") & mask).sum()),
            "tp_covered": int(((df["exp013i_error_type"] == "TP") & mask).sum()),
            "fp_preserved_cost": int(((df["exp013i_error_type"] == "FP") & mask).sum()),
            "tn_covered": int(((df["exp013i_error_type"] == "TN") & mask).sum()),
            "specificity_score": float(int(((df["exp013i_error_type"] == "FN") & mask).sum()) / max(int(mask.sum()), 1)),
        })

    # Categorical exact values that cover FNs.
    for cols in SEGMENT_SETS:
        if any(c not in df.columns for c in cols):
            continue

        key_df = pd.DataFrame(index=df.index)
        for c in cols:
            key_df[c] = text(df, c)

        fn_keys = key_df.loc[df["exp013i_error_type"] == "FN"].drop_duplicates()
        for _, row in fn_keys.iterrows():
            mask = np.ones(len(df), dtype=bool)
            parts = []
            for c in cols:
                val = str(row[c])
                parts.append(f"{c}={val}")
                mask = mask & (key_df[c] == val).to_numpy(dtype=bool)

            rows.append({
                "family": "segment_fn_immunity",
                "description": "preserve if " + " AND ".join(parts),
                "feature": "|".join(cols),
                "fn_covered": int(((df["exp013i_error_type"] == "FN") & mask).sum()),
                "tp_covered": int(((df["exp013i_error_type"] == "TP") & mask).sum()),
                "fp_preserved_cost": int(((df["exp013i_error_type"] == "FP") & mask).sum()),
                "tn_covered": int(((df["exp013i_error_type"] == "TN") & mask).sum()),
                "specificity_score": float(int(((df["exp013i_error_type"] == "FN") & mask).sum()) / max(int(mask.sum()), 1)),
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values(["fn_covered", "fp_preserved_cost", "specificity_score"], ascending=[False, True, False]).reset_index(drop=True)


def module_signal_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ygroups = ["TP", "FP", "FN", "TN"]

    signals = {
        "lgbm_ge_0_02": num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0) >= 0.02,
        "lgbm_ge_0_05": num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0) >= 0.05,
        "if_ge_0_70": num(df, ["if_percentile_x", "if_percentile_y", "if_percentile"], 0.0) >= 0.70,
        "if_ge_0_90": num(df, ["if_percentile_x", "if_percentile_y", "if_percentile"], 0.0) >= 0.90,
        "se_score_ge_40": num(df, ["se_score_x", "se_score_y", "se_score"], 0.0) >= 40,
        "se_score_ge_65": num(df, ["se_score_x", "se_score_y", "se_score"], 0.0) >= 65,
        "se_patterns_ge_2": num(df, ["se_patterns_count", "se_pattern_count"], 0.0) >= 2,
        "beh_score_ge_25": num(df, ["beh_score", "behavioral_score"], 0.0) >= 25,
        "beh_score_ge_45": num(df, ["beh_score", "behavioral_score"], 0.0) >= 45,
        "runtime_flagged": num(df, "runtime_flagged", 0.0) >= 1,
    }

    for signal_name, mask_series in signals.items():
        mask = mask_series.to_numpy(dtype=bool) if hasattr(mask_series, "to_numpy") else np.asarray(mask_series, dtype=bool)
        row = {"signal": signal_name, "n_total": int(mask.sum())}
        for g in ygroups:
            denom = int((df["exp013i_error_type"] == g).sum())
            cnt = int(((df["exp013i_error_type"] == g) & mask).sum())
            row[f"n_{g}"] = cnt
            row[f"rate_{g}"] = float(cnt / max(denom, 1))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["rate_FN", "rate_FP"], ascending=[False, True]).reset_index(drop=True)


def recommended_actions(safe: pd.DataFrame, near: pd.DataFrame, immunity: pd.DataFrame, modules: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows = []

    if not safe.empty:
        for _, r in safe.head(top_n).iterrows():
            rows.append({
                "priority": "P1_SAFE_VETO",
                "action": "test_in_EXP013J",
                "description": r["description"],
                "expected_tp_loss": int(r["tp_loss"]),
                "expected_fp_removed": int(r["fp_removed"]),
                "rationale": "Candidato de veto com TP_loss=0 na amostra atual.",
            })

    if not immunity.empty:
        for _, r in immunity.head(top_n).iterrows():
            rows.append({
                "priority": "P2_FN_IMMUNITY",
                "action": "consider_preserve_guard",
                "description": r["description"],
                "expected_tp_loss": None,
                "expected_fp_removed": None,
                "rationale": f"Cobre {r['fn_covered']} FN(s) com custo FP_preserved={r['fp_preserved_cost']}.",
            })

    if not near.empty:
        for _, r in near.head(top_n).iterrows():
            rows.append({
                "priority": "P3_NEAR_SAFE_DIAGNOSTIC",
                "action": "diagnostic_only",
                "description": r["description"],
                "expected_tp_loss": int(r["tp_loss"]),
                "expected_fp_removed": int(r["fp_removed"]),
                "rationale": "Remove FP mas perde TP; só considerar se uma imunidade recuperar TP antes.",
            })

    if not modules.empty:
        top = modules.head(5)
        for _, r in top.iterrows():
            rows.append({
                "priority": "P4_MODULE_SIGNAL_REVIEW",
                "action": "review_module_signal",
                "description": r["signal"],
                "expected_tp_loss": None,
                "expected_fp_removed": None,
                "rationale": f"Signal rate FN={r['rate_FN']:.3f}, FP={r['rate_FP']:.3f}; útil para preservar ou vetar.",
            })

    return pd.DataFrame(rows)


def make_report(summary: dict[str, Any], safe: pd.DataFrame, immunity: pd.DataFrame, modules: pd.DataFrame, recs: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-013I — FN Immunity & Residual FP Anatomy")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Métrica atual: TP={summary['current_metrics']['tp']}, FP={summary['current_metrics']['fp']}, FN={summary['current_metrics']['fn']}, recall={summary['current_metrics']['recall']}, precision={summary['current_metrics']['precision']}")
    lines.append(f"- FNs analisados: {summary['n_fn']}")
    lines.append(f"- FPs remanescentes analisados: {summary['n_fp']}")
    lines.append(f"- Candidatos safe veto TP=0: {summary['n_safe_veto_candidates']}")
    lines.append(f"- Candidatos imunidade FN: {summary['n_fn_immunity_candidates']}")
    lines.append("")
    lines.append("## Top safe veto candidates")
    if safe.empty:
        lines.append("Nenhum candidato TP_loss=0 encontrado com os filtros atuais.")
    else:
        show = ["family", "description", "tp_loss", "fp_removed", "recall_after"]
        lines.append(safe[show].head(20).to_markdown(index=False))
    lines.append("")
    lines.append("## Top FN immunity candidates")
    if immunity.empty:
        lines.append("Nenhuma regra de imunidade candidata encontrada.")
    else:
        show = ["family", "description", "fn_covered", "tp_covered", "fp_preserved_cost", "specificity_score"]
        lines.append(immunity[show].head(20).to_markdown(index=False))
    lines.append("")
    lines.append("## Module signal matrix")
    if modules.empty:
        lines.append("Sem matriz de sinais.")
    else:
        show = ["signal", "n_TP", "rate_TP", "n_FP", "rate_FP", "n_FN", "rate_FN"]
        lines.append(modules[show].to_markdown(index=False))
    lines.append("")
    lines.append("## Recomendações")
    if recs.empty:
        lines.append("Sem recomendações automáticas.")
    else:
        lines.append(recs.to_markdown(index=False))
    lines.append("")
    lines.append("## Próximo passo")
    lines.append("Usar apenas candidatos P1/P2 em uma rodada EXP-013J pequena e congelável, mantendo TP>=118 e recall>=95%.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pred-col", default=None)
    parser.add_argument("--min-segment-n", type=int, default=10)
    parser.add_argument("--min-fp-removed", type=int, default=5)
    parser.add_argument("--max-tp-loss-near", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-013I — FN Immunity & Residual FP Anatomy")
    log("=" * 80)
    log(f"Input: {input_path}")
    log(f"Output: {output_dir}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    pred_col = infer_pred_col(df, args.pred_col)
    df = add_error_labels(df, pred_col)

    y = df["is_fraud"].to_numpy(dtype=int)
    pred = df[pred_col].to_numpy(dtype=int)
    metrics = compute_metrics(y, pred)

    log(f"Policy: {pred_col}")
    log(f"Metrics: TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} recall={metrics['recall']} precision={metrics['precision']}")

    pd.DataFrame([{"pred_col": pred_col, **metrics}]).to_csv(output_dir / "01_current_policy_metrics.csv", index=False)

    profile_cols = safe_cols_for_profile(df)
    fns = df[df["exp013i_error_type"] == "FN"][profile_cols].copy()
    fps = df[df["exp013i_error_type"] == "FP"][profile_cols].copy()
    fns.to_csv(output_dir / "02_false_negatives_profile.csv", index=False)
    fps.head(1000).to_csv(output_dir / "03_false_positives_profile_sample.csv", index=False)

    log("[1/5] Anatomia numérica...")
    numeric = numeric_error_anatomy(df)
    numeric.to_csv(output_dir / "04_numeric_error_anatomy.csv", index=False)

    log("[2/5] Anatomia de segmentos positivos...")
    seg = segment_positive_anatomy(df, pred_col, args.min_segment_n)
    seg.to_csv(output_dir / "05_segment_positive_anatomy.csv", index=False)

    log("[3/5] Candidatos de veto seguro...")
    safe, near = candidate_vetos(df, pred_col, args.min_fp_removed, args.max_tp_loss_near)
    safe.to_csv(output_dir / "06_safe_veto_candidates_tp0.csv", index=False)
    near.to_csv(output_dir / "07_near_safe_veto_candidates_tp1.csv", index=False)

    log("[4/5] Candidatos de imunidade FN e matriz de módulos...")
    immunity = fn_immunity_candidates(df, pred_col)
    immunity.to_csv(output_dir / "08_fn_immunity_candidates.csv", index=False)

    modules = module_signal_matrix(df)
    modules.to_csv(output_dir / "09_module_signal_matrix.csv", index=False)

    log("[5/5] Recomendações...")
    recs = recommended_actions(safe, near, immunity, modules, args.top_n)
    recs.to_csv(output_dir / "10_recommended_next_actions.csv", index=False)

    objective_status = "DIAGNOSTIC_DONE"
    if not safe.empty:
        objective_status += "_SAFE_VETO_CANDIDATES_FOUND"
    if not immunity.empty:
        objective_status += "_FN_IMMUNITY_CANDIDATES_FOUND"

    summary = {
        "experiment": "EXP-013I",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "pred_col": pred_col,
        "n_rows": int(len(df)),
        "total_frauds": int(df["is_fraud"].sum()),
        "current_metrics": metrics,
        "n_fn": int((df["exp013i_error_type"] == "FN").sum()),
        "n_fp": int((df["exp013i_error_type"] == "FP").sum()),
        "n_tp": int((df["exp013i_error_type"] == "TP").sum()),
        "n_tn": int((df["exp013i_error_type"] == "TN").sum()),
        "n_safe_veto_candidates": int(len(safe)),
        "n_near_safe_candidates": int(len(near)),
        "n_fn_immunity_candidates": int(len(immunity)),
        "top_safe_veto_candidates": safe.head(10).to_dict(orient="records") if not safe.empty else [],
        "top_fn_immunity_candidates": immunity.head(10).to_dict(orient="records") if not immunity.empty else [],
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, safe, immunity, modules, recs)
    (output_dir / "11_fn_immunity_residual_fp_report.md").write_text(report, encoding="utf-8")

    log("")
    log("=" * 80)
    log("EXP-013I CONCLUÍDO")
    log("=" * 80)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "02_false_negatives_profile.csv",
        output_dir / "04_numeric_error_anatomy.csv",
        output_dir / "06_safe_veto_candidates_tp0.csv",
        output_dir / "08_fn_immunity_candidates.csv",
        output_dir / "09_module_signal_matrix.csv",
        output_dir / "10_recommended_next_actions.csv",
        output_dir / "11_fn_immunity_residual_fp_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
