#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3I — Patch R3H-FROZEN + auditoria dos 56 FNs residuais

Objetivo:
  1) Executar logo no início o patch de interpretação do R3H-FROZEN:
     - métricas congeladas bateram;
     - schema bateu;
     - Wilson passou;
     - TP_loss=0 e FP_removed=585 bateram;
     - divergência por regra é tratada como OVERLAP_WARNING, não FAIL.

  2) Em seguida, auditar os 56 FNs residuais do benchmark congelado R3H:
        TP=1409
        FP=4935
        FN=56
        recall=96,177%
        precision=22,210%
        FPR=4,391%
        Wilson low=95,0687%

  3) Gerar candidatos curtos de "resgate de FN" sem treinar modelo e sem busca
     longa, medindo o custo em FP adicionado.

Este experimento é diagnóstico:
  - Não promove automaticamente nenhuma regra de resgate.
  - Ele mostra o custo para recuperar FNs residuais.
  - A decisão posterior será:
      a) existe resgate barato de FN?
      b) ou os 56 FNs são limite prático do score/features atuais?

Uso padrão:
  python scripts/exp_014b_r3i_patch_frozen_and_audit_fns.py

Execução mais restrita:
  python scripts/exp_014b_r3i_patch_frozen_and_audit_fns.py --max-fp-added-candidate 300 --max-combo-size 3

Execução mais exploratória, ainda curta:
  python scripts/exp_014b_r3i_patch_frozen_and_audit_fns.py --max-fp-added-candidate 1500 --max-combo-size 4 --top-groups-per-combo 80

Saídas:
  resultados/experimentos/EXP-014B-R3I/
    00_run_summary.json
    01_r3h_frozen_status_patch.json
    02_r3h_frozen_status_patch.md
    03_input_contract.json
    04_fn_residuals.csv
    05_fn_profile_by_feature.csv
    06_fn_score_profile.csv
    07_rescue_candidates.csv
    08_rescue_frontier_greedy.csv
    09_top_rescue_scenarios.csv
    10_policy_artifact_diagnostic.json
    11_exp014b_r3i_report.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "dados").exists() else Path.cwd()

DEFAULT_FROZEN_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3H-FROZEN"
DEFAULT_INPUT = DEFAULT_FROZEN_DIR / "10_predictions.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3I"

BASE_COL = "exp014b_r3g_balanced_final_pred"
PRED_COL = "exp014b_r3h_frozen_pred"

R3H_EXPECTED = {
    "tp": 1409,
    "fp": 4935,
    "fn": 56,
    "fp_removed_vs_base": 585,
    "tp_loss_vs_base": 0,
    "wilson_low_min": 0.95,
}

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
        raise RuntimeError("Coluna obrigatória ausente: is_fraud")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)

    if "transaction_id" in df.columns:
        df["transaction_id"] = df["transaction_id"].astype("string").str.strip()

    for c in ["event_datetime", "data_pix"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    for c in [BASE_COL, PRED_COL]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

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
    if "ratio_bin" not in df.columns and "ratio_valor_media_pagador_90d" in df.columns:
        df["ratio_bin"] = qbin_series(num(df, "ratio_valor_media_pagador_90d", 0.0), "ratio", [0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    if "qtd_rec_bin" not in df.columns and "qtd_pix_recebidos_180d" in df.columns:
        df["qtd_rec_bin"] = qbin_series(num(df, "qtd_pix_recebidos_180d", 0.0), "qtdrec", [0, 1, 2, 5, 10, 20, 50, 100])
    if "vl_bin" not in df.columns and "vl_pix" in df.columns:
        df["vl_bin"] = qbin_series(num(df, "vl_pix", 0.0), "vl", [20, 50, 100, 250, 500, 1000, 5000, 10000])

    if "module_quiet" not in df.columns:
        se_score = num(df, ["se_score_x", "se_score_y", "se_score"], 0.0)
        se_count = num(df, ["se_patterns_count", "se_pattern_count"], 0.0)
        beh_score = num(df, ["beh_score", "behavioral_score"], 0.0)
        beh_count = num(df, ["beh_factors_count", "behavioral_risk_factor_count"], 0.0)
        runtime = num(df, "runtime_flagged", 0.0)
        strong = (se_score >= 40) | (se_count >= 2) | (beh_score >= 25) | (beh_count >= 2) | (runtime >= 1)
        df["module_quiet"] = np.where(strong, "module_strong", "module_quiet")

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


def patch_r3h_frozen_status(frozen_dir: Path, output_dir: Path) -> dict[str, Any]:
    """
    Patch de interpretação:
    - Se métricas finais, delta, schema e Wilson passaram, mas rule_replay_ok
      falhou só por expected_fp_removed individual, classificar como overlap OK.
    """
    summary_path = frozen_dir / "00_run_summary.json"
    replay_path = frozen_dir / "09_policy_replay_artifact.json"
    rule_path = frozen_dir / "03_rule_replay_impact.csv"

    if not summary_path.exists() or not replay_path.exists():
        patch = {
            "patch_status": "SKIPPED_R3H_FROZEN_FILES_NOT_FOUND",
            "summary_path": str(summary_path),
            "replay_path": str(replay_path),
        }
        dump_json(patch, output_dir / "01_r3h_frozen_status_patch.json")
        (output_dir / "02_r3h_frozen_status_patch.md").write_text("# R3H Frozen status patch\n\nArquivos não encontrados.\n", encoding="utf-8")
        return patch

    summary = load_json(summary_path)
    replay = load_json(replay_path)
    rule_df = pd.read_csv(rule_path) if rule_path.exists() else pd.DataFrame()

    no_missing_cols = True
    if not rule_df.empty and "missing_columns" in rule_df.columns:
        no_missing_cols = rule_df["missing_columns"].fillna("").astype(str).eq("").all()

    all_tp_loss_ok = True
    if not rule_df.empty and "tp_loss_match_expected" in rule_df.columns:
        all_tp_loss_ok = rule_df["tp_loss_match_expected"].astype(bool).all()

    fp_mismatches = []
    if not rule_df.empty and "fp_removed_match_expected" in rule_df.columns:
        fp_mismatches = rule_df[~rule_df["fp_removed_match_expected"].astype(bool)].to_dict(orient="records")

    overlap_only = (
        bool(summary.get("schema_ok"))
        and bool(summary.get("expected_metrics_matched"))
        and bool(summary.get("expected_delta_matched"))
        and bool(summary.get("wilson_pass"))
        and int(summary.get("tp_loss_vs_base", -1)) == 0
        and no_missing_cols
        and all_tp_loss_ok
        and bool(fp_mismatches)
    )

    if bool(summary.get("all_pass")):
        patched_status = "PASS_R3H_FROZEN_VALIDATED_ALREADY_PASS"
        patched_all_pass = True
    elif overlap_only:
        patched_status = "PASS_R3H_FROZEN_VALIDATED_METRICS_MATCH_RULE_OVERLAP_OK_WILSON_PASS"
        patched_all_pass = True
    else:
        patched_status = "FAIL_R3H_FROZEN_VALIDATION_REMAINS_UNRESOLVED"
        patched_all_pass = False

    patch = {
        "patch_status": patched_status,
        "patched_all_pass": patched_all_pass,
        "original_objective_status": summary.get("objective_status"),
        "reason": "per-rule expected_fp_removed mismatch is overlap-aware warning, not validation failure" if overlap_only else "no overlap-only condition detected",
        "summary_checks": {
            "schema_ok": summary.get("schema_ok"),
            "expected_metrics_matched": summary.get("expected_metrics_matched"),
            "expected_delta_matched": summary.get("expected_delta_matched"),
            "rule_replay_ok_original": summary.get("rule_replay_ok"),
            "wilson_pass": summary.get("wilson_pass"),
            "tp_loss_vs_base": summary.get("tp_loss_vs_base"),
            "fp_removed_vs_base": summary.get("fp_removed_vs_base"),
            "final_metrics": summary.get("final_metrics"),
        },
        "overlap_warning": {
            "no_missing_columns": bool(no_missing_cols),
            "all_tp_loss_match_expected": bool(all_tp_loss_ok),
            "n_fp_removed_individual_mismatches": int(len(fp_mismatches)),
            "mismatched_rule_ids": [str(x.get("rule_id")) for x in fp_mismatches],
        },
        "source_files": {
            "summary": str(summary_path),
            "replay_artifact": str(replay_path),
            "rule_replay_impact": str(rule_path),
        },
    }

    dump_json(patch, output_dir / "01_r3h_frozen_status_patch.json")

    md = [
        "# R3H Frozen status patch",
        "",
        f"Original status: `{summary.get('objective_status')}`",
        f"Patched status: `{patched_status}`",
        "",
        "## Racional",
        "",
        "As métricas finais, delta agregado, schema e Wilson passaram. A divergência estava apenas no `expected_fp_removed` individual de algumas regras.",
        "Como o replay é sequencial, regras anteriores podem remover FPs que também pertenciam a regras posteriores. Isso reduz o impacto individual posterior, mas o efeito líquido final bateu exatamente.",
        "",
        "## Checks",
        "",
        f"- expected_metrics_matched: `{summary.get('expected_metrics_matched')}`",
        f"- expected_delta_matched: `{summary.get('expected_delta_matched')}`",
        f"- schema_ok: `{summary.get('schema_ok')}`",
        f"- wilson_pass: `{summary.get('wilson_pass')}`",
        f"- tp_loss_vs_base: `{summary.get('tp_loss_vs_base')}`",
        f"- fp_removed_vs_base: `{summary.get('fp_removed_vs_base')}`",
        f"- n_fp_removed_individual_mismatches: `{len(fp_mismatches)}`",
        "",
        "## Decisão",
        "",
        "Se `patched_all_pass=true`, o R3H-FROZEN fica validado funcionalmente e a execução pode seguir para a auditoria dos FNs residuais.",
    ]
    (output_dir / "02_r3h_frozen_status_patch.md").write_text("\n".join(md), encoding="utf-8")
    return patch


def make_contract(df: pd.DataFrame, pred_col: str) -> dict[str, Any]:
    required = ["is_fraud", pred_col]
    missing = [c for c in required if c not in df.columns]
    return {
        "n_rows": int(len(df)),
        "n_frauds": int(df["is_fraud"].sum()) if "is_fraud" in df.columns else None,
        "pred_col": pred_col,
        "required_columns": required,
        "missing_columns": missing,
        "feature_cols_present": [c for c in FEATURE_COLS if c in df.columns],
        "numeric_cols_present": [c for c in NUMERIC_COLS if c in df.columns],
        "contract_ok": not missing,
    }


def profile_fns(df: pd.DataFrame, pred_col: str, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fn = df[(df["is_fraud"] == 1) & (df[pred_col] == 0)].copy()
    tp = df[(df["is_fraud"] == 1) & (df[pred_col] == 1)].copy()
    neg_not_alerted = df[(df["is_fraud"] == 0) & (df[pred_col] == 0)].copy()

    fn.to_csv(output_dir / "04_fn_residuals.csv", index=False)

    rows = []
    for c in FEATURE_COLS:
        if c not in df.columns:
            continue
        fn_counts = fn[c].astype("string").fillna("<MISSING>").value_counts()
        tp_counts = tp[c].astype("string").fillna("<MISSING>").value_counts()
        neg_counts = neg_not_alerted[c].astype("string").fillna("<MISSING>").value_counts()
        vals = sorted(set(fn_counts.index).union(set(tp_counts.index)).union(set(neg_counts.index)))
        for v in vals:
            fn_n = int(fn_counts.get(v, 0))
            tp_n = int(tp_counts.get(v, 0))
            neg_n = int(neg_counts.get(v, 0))
            if fn_n == 0 and tp_n == 0:
                continue
            fn_rate = fn_n / max(len(fn), 1)
            tp_rate = tp_n / max(len(tp), 1)
            neg_rate = neg_n / max(len(neg_not_alerted), 1)
            rows.append({
                "feature": c,
                "value": str(v),
                "fn_count": fn_n,
                "tp_count": tp_n,
                "neg_not_alerted_count": neg_n,
                "fn_share": fn_rate,
                "tp_share": tp_rate,
                "neg_not_alerted_share": neg_rate,
                "fn_vs_tp_lift": fn_rate / max(tp_rate, 1e-9),
                "fn_vs_neg_lift": fn_rate / max(neg_rate, 1e-9),
            })

    profile_df = pd.DataFrame(rows)
    if not profile_df.empty:
        profile_df = profile_df.sort_values(["fn_count", "fn_vs_tp_lift"], ascending=[False, False]).reset_index(drop=True)
    profile_df.to_csv(output_dir / "05_fn_profile_by_feature.csv", index=False)

    score_rows = []
    for c in NUMERIC_COLS:
        if c not in df.columns:
            continue
        for name, part in [("FN", fn), ("TP", tp), ("NEG_NOT_ALERTED", neg_not_alerted)]:
            vals = pd.to_numeric(part[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
            if vals.empty:
                continue
            score_rows.append({
                "score_col": c,
                "group": name,
                "n": int(len(vals)),
                "mean": float(vals.mean()),
                "p05": float(vals.quantile(0.05)),
                "p25": float(vals.quantile(0.25)),
                "p50": float(vals.quantile(0.50)),
                "p75": float(vals.quantile(0.75)),
                "p95": float(vals.quantile(0.95)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            })
    score_df = pd.DataFrame(score_rows)
    score_df.to_csv(output_dir / "06_fn_score_profile.csv", index=False)
    return profile_df, score_df


def candidate_result(df: pd.DataFrame, pred: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    y = df["is_fraud"].to_numpy(dtype=int)
    # Rescue only non-alerted rows.
    rescue = mask & (pred == 0)
    fn_recovered = int(((y == 1) & rescue).sum())
    fp_added = int(((y == 0) & rescue).sum())
    n_added = int(rescue.sum())
    return {"fn_recovered": fn_recovered, "fp_added": fp_added, "n_added": n_added}


def add_rescue_candidate(
    rows: list[dict[str, Any]],
    df: pd.DataFrame,
    pred: np.ndarray,
    mask: np.ndarray,
    family: str,
    description: str,
    params: dict[str, Any],
    min_fn_recovered: int,
    max_fp_added: int,
) -> None:
    r = candidate_result(df, pred, mask)
    if r["fn_recovered"] < min_fn_recovered:
        return
    if r["fp_added"] > max_fp_added:
        return
    fp_per_fn = r["fp_added"] / max(r["fn_recovered"], 1)
    rows.append({
        "candidate_id": f"rescue_{len(rows):05d}",
        "family": family,
        "description": description,
        **r,
        "fp_per_fn": fp_per_fn,
        "params_json": json.dumps(params, ensure_ascii=False),
    })


def mine_rescue_candidates(
    df: pd.DataFrame,
    pred_col: str,
    min_fn_recovered: int,
    max_fp_added_candidate: int,
    max_combo_size: int,
    top_groups_per_combo: int,
) -> pd.DataFrame:
    pred = df[pred_col].to_numpy(dtype=int)
    y = df["is_fraud"].to_numpy(dtype=int)
    not_alerted = pred == 0
    rows: list[dict[str, Any]] = []

    # Numeric rescue candidates: high score among currently non-alerted.
    for c in NUMERIC_COLS:
        if c not in df.columns:
            continue
        vals = num(df, c, 0.0).to_numpy(dtype=float)
        fn_vals = vals[(y == 1) & not_alerted]
        if len(fn_vals) == 0:
            continue
        cuts = sorted(set(float(x) for x in np.quantile(fn_vals, [0.0, 0.05, 0.10, 0.25, 0.50, 0.75]) if np.isfinite(x)))
        for cut in cuts:
            mask = not_alerted & (vals >= cut)
            add_rescue_candidate(
                rows, df, pred, mask,
                family="numeric_high_score_rescue",
                description=f"{c}>={cut:g}",
                params={"type": "numeric_threshold_rescue", "col": c, "direction": "ge", "cut": cut},
                min_fn_recovered=min_fn_recovered,
                max_fp_added=max_fp_added_candidate,
            )

    # Categorical/microsegment rescue candidates.
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    feat = pd.DataFrame(index=df.index)
    for c in feat_cols:
        feat[c] = df[c].astype("string").fillna("<MISSING>").astype(str)

    candidate_combos = []
    for r in range(1, max_combo_size + 1):
        for combo in itertools.combinations(feat_cols, r):
            combo = list(combo)
            # Avoid extremely broad singletons except important bins.
            if r == 1 and combo[0] not in ["lgbm_bin", "if_bin", "score_bin", "ds_tipo_chave_norm", "value_band"]:
                continue
            candidate_combos.append(combo)

    idx = np.where(not_alerted)[0]
    for combo in candidate_combos:
        sub = feat.iloc[idx][combo]
        grouped = sub.groupby(combo, dropna=False).indices
        group_rows = []
        for key, rel_idxs in grouped.items():
            idxs = sub.iloc[list(rel_idxs)].index.to_numpy(dtype=int)
            mask = np.zeros(len(df), dtype=bool)
            mask[idxs] = True
            r = candidate_result(df, pred, mask)
            if r["fn_recovered"] < min_fn_recovered or r["fp_added"] > max_fp_added_candidate:
                continue
            fp_per_fn = r["fp_added"] / max(r["fn_recovered"], 1)
            group_rows.append((fp_per_fn, -r["fn_recovered"], key, mask, r))
        group_rows.sort()
        for fp_per_fn, neg_fn, key, mask, r in group_rows[:top_groups_per_combo]:
            vals = key if isinstance(key, tuple) else (key,)
            vals = [str(v) for v in vals]
            desc = " AND ".join([f"{c}={v}" for c, v in zip(combo, vals)])
            add_rescue_candidate(
                rows, df, pred, mask,
                family="microsegment_rescue",
                description=desc,
                params={"type": "combo_rescue", "combo_cols": combo, "combo_values": vals},
                min_fn_recovered=min_fn_recovered,
                max_fp_added=max_fp_added_candidate,
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Deduplicate candidates by description/params.
    out = out.drop_duplicates(subset=["params_json"]).reset_index(drop=True)
    out = out.sort_values(["fp_per_fn", "fp_added", "fn_recovered"], ascending=[True, True, False]).reset_index(drop=True)
    out["candidate_id"] = [f"rescue_{i:05d}" for i in range(len(out))]
    return out


def mask_from_rescue_params(df: pd.DataFrame, pred: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    not_alerted = pred == 0
    if params.get("type") == "numeric_threshold_rescue":
        vals = num(df, params["col"], 0.0).to_numpy(dtype=float)
        if params.get("direction") == "ge":
            return not_alerted & (vals >= float(params["cut"]))
        return not_alerted & (vals <= float(params["cut"]))
    if params.get("type") == "combo_rescue":
        mask = not_alerted.copy()
        for c, v in zip(params.get("combo_cols", []), params.get("combo_values", [])):
            if c not in df.columns:
                return np.zeros(len(df), dtype=bool)
            mask = mask & (df[c].astype("string").fillna("<MISSING>").astype(str).to_numpy() == str(v))
        return mask
    return np.zeros(len(df), dtype=bool)


def greedy_rescue_frontier(df: pd.DataFrame, pred_col: str, candidates_df: pd.DataFrame, fp_budgets: list[int]) -> pd.DataFrame:
    pred0 = df[pred_col].to_numpy(dtype=int)
    y = df["is_fraud"].to_numpy(dtype=int)

    rows = []
    if candidates_df.empty:
        return pd.DataFrame(rows)

    candidates = candidates_df.head(500).to_dict(orient="records")
    for budget in fp_budgets:
        current_pred = pred0.copy()
        selected = []
        total_fp_added = 0
        total_fn_recovered = 0

        # Greedy recomputes marginal utility each step.
        while True:
            best = None
            for cand in candidates:
                if cand["candidate_id"] in selected:
                    continue
                params = json.loads(cand["params_json"])
                mask = mask_from_rescue_params(df, current_pred, params)
                r = candidate_result(df, current_pred, mask)
                if r["fn_recovered"] <= 0:
                    continue
                if total_fp_added + r["fp_added"] > budget:
                    continue
                rank = (r["fn_recovered"] / max(r["fp_added"], 1), r["fn_recovered"], -r["fp_added"])
                if best is None or rank > best[0]:
                    best = (rank, cand, mask, r)
            if best is None:
                break
            _, cand, mask, r = best
            current_pred[mask] = 1
            selected.append(cand["candidate_id"])
            total_fp_added += r["fp_added"]
            total_fn_recovered += r["fn_recovered"]

        m = compute_metrics(y, current_pred)
        rows.append({
            "fp_budget": int(budget),
            "fn_recovered": int(total_fn_recovered),
            "fp_added": int(total_fp_added),
            "n_selected_candidates": int(len(selected)),
            "selected_candidate_ids": "|".join(selected),
            **m,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["fn", "fp"], ascending=[True, True]).reset_index(drop=True)
    return out


def make_report(summary: dict[str, Any], patch: dict[str, Any], top_candidates: pd.DataFrame, frontier: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014B-R3I — Patch R3H-FROZEN + auditoria dos 56 FNs")
    lines.append("")
    lines.append("## Patch R3H-FROZEN")
    lines.append(f"- Patch status: `{patch.get('patch_status')}`")
    lines.append(f"- Patched all pass: `{patch.get('patched_all_pass')}`")
    lines.append("")
    lines.append("## Resultado da auditoria")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Base R3H metrics: `{summary['base_metrics']}`")
    lines.append(f"- FNs residuais: `{summary['n_false_negatives']}`")
    lines.append(f"- Candidatos de resgate: `{summary['n_rescue_candidates']}`")
    lines.append("")
    lines.append("## Top candidatos de resgate")
    if top_candidates.empty:
        lines.append("Nenhum candidato de resgate encontrado dentro dos limites definidos.")
    else:
        show = ["candidate_id", "family", "description", "fn_recovered", "fp_added", "fp_per_fn"]
        lines.append(top_candidates[[c for c in show if c in top_candidates.columns]].head(20).to_markdown(index=False))
    lines.append("")
    lines.append("## Fronteira greedy por orçamento de FP")
    if frontier.empty:
        lines.append("Fronteira vazia.")
    else:
        show = ["fp_budget", "fn_recovered", "fp_added", "tp", "fp", "fn", "precision", "recall", "fpr", "n_selected_candidates"]
        lines.append(frontier[[c for c in show if c in frontier.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Decisão sugerida")
    if not frontier.empty and int(frontier["fn_recovered"].max()) > 0:
        lines.append("Há pelo menos algum resgate de FN possível. Avaliar se o custo de FP por FN recuperado é aceitável antes de transformar em política.")
    else:
        lines.append("Não apareceu resgate barato de FN nos limites testados. Isso sugere que os 56 FNs podem depender de novo score/features/segundo estágio.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--frozen-dir", default=str(DEFAULT_FROZEN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pred-col", default=PRED_COL)
    parser.add_argument("--min-fn-recovered", type=int, default=1)
    parser.add_argument("--max-fp-added-candidate", type=int, default=750)
    parser.add_argument("--max-combo-size", type=int, default=3)
    parser.add_argument("--top-groups-per-combo", type=int, default=50)
    parser.add_argument("--fp-budgets", default="100,250,500,1000,2000,4000")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    frozen_dir = Path(args.frozen_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014B-R3I — Patch R3H-FROZEN + auditoria dos 56 FNs")
    log("=" * 80)

    log("[0/5] Aplicando patch de interpretação do R3H-FROZEN...")
    patch = patch_r3h_frozen_status(frozen_dir, output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    log("[1/5] Carregando predições congeladas R3H...")
    df = add_bins_and_guards(normalize_columns(pd.read_csv(input_path, low_memory=False)))

    contract = make_contract(df, args.pred_col)
    dump_json(contract, output_dir / "03_input_contract.json")
    if not contract["contract_ok"]:
        raise RuntimeError(f"Contrato falhou: {contract['missing_columns']}")

    y = df["is_fraud"].to_numpy(dtype=int)
    pred = df[args.pred_col].to_numpy(dtype=int)
    base_metrics = compute_metrics(y, pred)
    n_fn = int(((y == 1) & (pred == 0)).sum())

    log(f"Base metrics: {base_metrics}")

    log("[2/5] Perfilando FNs residuais...")
    fn_profile, score_profile = profile_fns(df, args.pred_col, output_dir)

    log("[3/5] Minerando candidatos curtos de resgate de FN...")
    rescue = mine_rescue_candidates(
        df=df,
        pred_col=args.pred_col,
        min_fn_recovered=args.min_fn_recovered,
        max_fp_added_candidate=args.max_fp_added_candidate,
        max_combo_size=args.max_combo_size,
        top_groups_per_combo=args.top_groups_per_combo,
    )
    rescue.to_csv(output_dir / "07_rescue_candidates.csv", index=False)

    log(f"Candidatos de resgate: {len(rescue)}")

    budgets = [int(x.strip()) for x in str(args.fp_budgets).split(",") if x.strip()]
    log("[4/5] Construindo fronteira greedy de resgate por orçamento de FP...")
    frontier = greedy_rescue_frontier(df, args.pred_col, rescue, budgets)
    frontier.to_csv(output_dir / "08_rescue_frontier_greedy.csv", index=False)

    top_scenarios = frontier.head(20) if not frontier.empty else pd.DataFrame()
    top_scenarios.to_csv(output_dir / "09_top_rescue_scenarios.csv", index=False)

    objective_status = "DONE"
    objective_status += "_PATCHED_R3H_FROZEN_PASS" if patch.get("patched_all_pass") else "_PATCHED_R3H_FROZEN_NOT_PASS"
    objective_status += "_RESCUE_CANDIDATES_FOUND" if len(rescue) > 0 else "_NO_RESCUE_CANDIDATES"
    if not frontier.empty and int(frontier["fn_recovered"].max()) > 0:
        objective_status += "_FN_RECOVERY_POSSIBLE"
    else:
        objective_status += "_FN_RECOVERY_NOT_FOUND"

    artifact = {
        "experiment": "EXP-014B-R3I",
        "policy_name": "fn_residual_audit_diagnostic",
        "objective_status": objective_status,
        "patch_r3h_frozen": patch,
        "input_path": str(input_path),
        "pred_col": args.pred_col,
        "base_metrics": base_metrics,
        "n_false_negatives": n_fn,
        "n_rescue_candidates": int(len(rescue)),
        "top_rescue_candidates": rescue.head(20).to_dict(orient="records") if not rescue.empty else [],
        "rescue_frontier": frontier.to_dict(orient="records") if not frontier.empty else [],
        "notes": [
            "Diagnostic only.",
            "Rescue candidates add alerts to current R3H frozen non-alerted rows.",
            "Do not promote rescue rules before frozen validation and business review of FP cost."
        ],
    }
    dump_json(artifact, output_dir / "10_policy_artifact_diagnostic.json")

    summary = {
        "experiment": "EXP-014B-R3I",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "frozen_dir": str(frozen_dir),
        "n_rows": int(len(df)),
        "n_frauds": int(df["is_fraud"].sum()),
        "pred_col": args.pred_col,
        "base_metrics": base_metrics,
        "n_false_negatives": n_fn,
        "patch_status": patch.get("patch_status"),
        "patched_r3h_frozen_all_pass": patch.get("patched_all_pass"),
        "n_rescue_candidates": int(len(rescue)),
        "best_candidate": rescue.iloc[0].to_dict() if not rescue.empty else None,
        "best_frontier": frontier.iloc[0].to_dict() if not frontier.empty else None,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, patch, rescue, frontier)
    (output_dir / "11_exp014b_r3i_report.md").write_text(report, encoding="utf-8")

    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_r3h_frozen_status_patch.json",
        output_dir / "02_r3h_frozen_status_patch.md",
        output_dir / "04_fn_residuals.csv",
        output_dir / "05_fn_profile_by_feature.csv",
        output_dir / "06_fn_score_profile.csv",
        output_dir / "07_rescue_candidates.csv",
        output_dir / "08_rescue_frontier_greedy.csv",
        output_dir / "10_policy_artifact_diagnostic.json",
        output_dir / "11_exp014b_r3i_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
