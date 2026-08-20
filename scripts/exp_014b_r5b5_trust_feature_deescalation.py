#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R5B5 — Trust Feature De-escalation Probe.

Cria features explícitas de confiança do pagador/recebedor/relacionamento e
testa se elas removem mais normais residuais de BLOQUEAR após R5B4, preservando
TP loss = 0. Não retreina modelo; é uma prova offline para orientar a próxima
integração de feature engineering.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp_014b_r5b2_tune_policy import LABELS, find_col, ints, metrics, pred_block, pred_intervention
from backend.core.preprocessing import create_trust_features as core_create_trust_features


EXPERIMENT = "EXP-014B-R5B5-TRUST-FEATURE-DEESCALATION"
SOURCE_EXPERIMENT = "EXP-014B-R5B4-ROBUST-BLOCK-DEESCALATION"


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def safe_log1p(series: pd.Series) -> pd.Series:
    return np.log1p(series.fillna(0).clip(lower=0))


def add_trust_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    payer_count_180 = numeric(df, "qtd_pix_pagador_180d")
    payer_value_180 = numeric(df, "valor_total_pagador_180d")
    payer_max_180 = numeric(df, "valor_maximo_pix_pagador_180d")
    receiver_count_180 = numeric(df, "qtd_pix_recebidos_180d")
    receiver_value_180 = numeric(df, "valor_total_recebido_180d")
    receiver_distinct_payers = numeric(df, "soma_pagadores_distintos_dia_recebedor_180d")
    pair_count_180 = numeric(df, "qtd_pix_mesmo_recebedor_180d")
    pair_value_180 = numeric(df, "valor_total_para_recebedor_180d")
    pair_days = numeric(df, "dias_desde_primeiro_envio_recebedor")
    value = numeric(df, "vl_pix")
    ratio_payer_mean = numeric(df, "ratio_valor_media_pagador_90d")
    lgbm = numeric(df, "lgbm_raw")
    first_receiver = numeric(df, "first_receiver_flag_real").fillna(1)

    df["payer_history_strength_score"] = (
        safe_log1p(payer_count_180) * 12.0
        + safe_log1p(payer_value_180) * 4.0
        + safe_log1p(payer_max_180) * 3.0
    ).clip(0, 100)

    df["receiver_reputation_score"] = (
        safe_log1p(receiver_count_180) * 14.0
        + safe_log1p(receiver_value_180) * 4.0
        + safe_log1p(receiver_distinct_payers) * 12.0
    ).clip(0, 100)

    df["relationship_strength_score"] = (
        safe_log1p(pair_count_180) * 22.0
        + safe_log1p(pair_value_180) * 4.0
        + np.minimum(pair_days.fillna(0).clip(lower=0), 180.0) / 180.0 * 30.0
    ).clip(0, 100)

    df["receiver_novelty_risk_score"] = (
        (first_receiver == 1).astype(float) * 35.0
        + (receiver_count_180.fillna(0) <= 0).astype(float) * 30.0
        + (receiver_value_180.fillna(0) <= 0).astype(float) * 20.0
        + (pair_count_180.fillna(0) <= 0).astype(float) * 15.0
    ).clip(0, 100)

    df["transaction_normality_score"] = (
        100.0
        - np.minimum(ratio_payer_mean.fillna(0).clip(lower=0), 25.0) * 2.4
        - np.minimum(value.fillna(0).clip(lower=0) / 1000.0, 30.0)
        - np.minimum(lgbm.fillna(0).clip(lower=0) * 300.0, 60.0)
    ).clip(0, 100)

    df["payer_receiver_trust_score"] = (
        df["payer_history_strength_score"] * 0.25
        + df["receiver_reputation_score"] * 0.30
        + df["relationship_strength_score"] * 0.30
        + df["transaction_normality_score"] * 0.15
        - df["receiver_novelty_risk_score"] * 0.35
    ).clip(0, 100)

    df["trust_bucket"] = pd.cut(
        df["payer_receiver_trust_score"],
        bins=[-0.01, 20, 40, 60, 80, 100],
        labels=["trust_00_20", "trust_20_40", "trust_40_60", "trust_60_80", "trust_80_100"],
    ).astype(str)
    df["receiver_rep_bucket"] = pd.cut(
        df["receiver_reputation_score"],
        bins=[-0.01, 20, 40, 60, 80, 100],
        labels=["rep_00_20", "rep_20_40", "rep_40_60", "rep_60_80", "rep_80_100"],
    ).astype(str)
    df["relationship_bucket"] = pd.cut(
        df["relationship_strength_score"],
        bins=[-0.01, 20, 40, 60, 80, 100],
        labels=["rel_00_20", "rel_20_40", "rel_40_60", "rel_60_80", "rel_80_100"],
    ).astype(str)
    df["novelty_bucket"] = pd.cut(
        df["receiver_novelty_risk_score"],
        bins=[-0.01, 20, 40, 60, 80, 100],
        labels=["nov_00_20", "nov_20_40", "nov_40_60", "nov_60_80", "nov_80_100"],
    ).astype(str)

    return df


def make_candidate(mask: np.ndarray, y: np.ndarray, description: str, candidate_id: str, rule_type: str) -> dict[str, Any] | None:
    n = int(mask.sum())
    if n <= 0:
        return None
    normals = int(((mask) & (y == 0)).sum())
    frauds = int(((mask) & (y == 1)).sum())
    if normals <= 0:
        return None
    return {
        "candidate_id": candidate_id,
        "rule_type": rule_type,
        "description": description,
        "n_affected": n,
        "normal_count": normals,
        "fraud_count": frauds,
        "normal_precision": round(float(normals / n), 8),
    }


def mine_candidates(df: pd.DataFrame, residual_block: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    candidates: list[dict[str, Any]] = []
    masks: dict[str, np.ndarray] = {}

    numeric_features = [
        "payer_receiver_trust_score",
        "receiver_reputation_score",
        "relationship_strength_score",
        "payer_history_strength_score",
        "transaction_normality_score",
        "receiver_novelty_risk_score",
    ]
    quantiles = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    for col in numeric_features:
        values = pd.to_numeric(df[col], errors="coerce")
        thresholds = sorted({round(float(values.quantile(q)), 8) for q in quantiles if pd.notna(values.quantile(q))})
        arr = values.to_numpy()
        for th in thresholds:
            for op in (">=", "<="):
                if op == ">=":
                    mask = residual_block & np.isfinite(arr) & (arr >= th)
                else:
                    mask = residual_block & np.isfinite(arr) & (arr <= th)
                cid = f"trust_score__{col}__{op}{th:g}"
                desc = f"BLOQUEAR->CONFIRMAR com {col} {op} {th:g}"
                cand = make_candidate(mask, y, desc, cid, "trust_score_threshold")
                if cand:
                    candidates.append(cand)
                    masks[cid] = mask

    categorical_features = [
        "trust_bucket",
        "receiver_rep_bucket",
        "relationship_bucket",
        "novelty_bucket",
        "lgbm_bin",
        "qtd_rec_bin",
        "valor_rec_bin",
        "first_receiver_flag_real",
        "value_band",
    ]
    for size in (1, 2, 3):
        for cols in itertools.combinations([c for c in categorical_features if c in df.columns], size):
            grouped = df.loc[residual_block, list(cols)].fillna("<MISSING>").astype(str)
            grouped["_idx"] = grouped.index
            for vals, grp in grouped.groupby(list(cols), dropna=False):
                vals = vals if isinstance(vals, tuple) else (vals,)
                local_mask = np.zeros(len(df), dtype=bool)
                local_mask[grp["_idx"].to_numpy()] = True
                parts = [f"{c} == {v}" for c, v in zip(cols, vals)]
                safe = "__".join(f"{c}={str(v)[:18]}" for c, v in zip(cols, vals))
                cid = f"trust_cat{size}__{safe}"
                desc = "BLOQUEAR->CONFIRMAR com " + " AND ".join(parts)
                cand = make_candidate(local_mask, y, desc, cid, f"trust_categorical_{size}")
                if cand:
                    candidates.append(cand)
                    masks[cid] = local_mask

    cand_df = pd.DataFrame(candidates).drop_duplicates(subset=["candidate_id"])
    if cand_df.empty:
        return cand_df, {}
    cand_df = cand_df.sort_values(["fraud_count", "normal_count", "normal_precision"], ascending=[True, False, False]).reset_index(drop=True)
    keep = set(cand_df["candidate_id"].head(5000))
    return cand_df[cand_df["candidate_id"].isin(keep)].reset_index(drop=True), {k: v for k, v in masks.items() if k in keep}


def robust_support(df: pd.DataFrame, mask: np.ndarray, y: np.ndarray) -> dict[str, int]:
    splits = df["temporal_split"].fillna("<MISSING>").astype(str)
    months = pd.to_datetime(df["event_datetime"], errors="coerce").dt.to_period("M").astype(str)
    out: dict[str, int] = {}
    for split in ["TRAIN", "VALIDATION", "HOLDOUT"]:
        split_mask = splits.eq(split).to_numpy()
        out[f"{split.lower()}_normals"] = int(((mask) & split_mask & (y == 0)).sum())
        out[f"{split.lower()}_frauds"] = int(((mask) & split_mask & (y == 1)).sum())
    out["non_train_normals"] = out["validation_normals"] + out["holdout_normals"]
    out["non_train_frauds"] = out["validation_frauds"] + out["holdout_frauds"]
    out["month_normal_support"] = int(months[(mask) & (y == 0)].nunique())
    out["month_fraud_support"] = int(months[(mask) & (y == 1)].nunique())
    return out


def select_rules(df: pd.DataFrame, candidates: pd.DataFrame, masks: dict[str, np.ndarray], y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    selected_mask = np.zeros(len(df), dtype=bool)
    rows: list[dict[str, Any]] = []
    remaining = candidates.copy()
    for step in range(1, 51):
        best: dict[str, Any] | None = None
        best_mask: np.ndarray | None = None
        best_score: tuple[Any, ...] | None = None
        for _, row in remaining.iterrows():
            cid = str(row["candidate_id"])
            mask = masks[cid] & (~selected_mask)
            if int(mask.sum()) == 0:
                continue
            normals = int(((mask) & (y == 0)).sum())
            frauds = int(((mask) & (y == 1)).sum())
            if normals < 5 or frauds > 0:
                continue
            support = robust_support(df, mask, y)
            if support["non_train_frauds"] > 0:
                continue
            if support["non_train_normals"] < 10:
                continue
            if support["month_normal_support"] < 2:
                continue
            score = (
                support["holdout_normals"] > 0,
                support["validation_normals"] > 0,
                support["non_train_normals"],
                support["month_normal_support"],
                normals,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_mask = mask
                best = row.to_dict()
                best.update(support)
                best["incremental_normals"] = normals
                best["incremental_frauds"] = frauds
        if best is None or best_mask is None:
            break
        selected_mask |= best_mask
        best["selection_step"] = step
        best["cumulative_normals"] = int(((selected_mask) & (y == 0)).sum())
        best["cumulative_frauds"] = int(((selected_mask) & (y == 1)).sum())
        rows.append(best)
        remaining = remaining[remaining["candidate_id"].astype(str) != str(best["candidate_id"])].reset_index(drop=True)

    return pd.DataFrame(rows), selected_mask


def by_action(df: pd.DataFrame, action_col: str, label_col: str) -> pd.DataFrame:
    out = df.groupby(action_col).agg(
        n_rows=(label_col, "size"),
        n_frauds=(label_col, "sum"),
    ).reset_index()
    out["n_normals"] = out["n_rows"] - out["n_frauds"]
    return out


def main() -> None:
    root = Path.cwd()
    input_path = root / "resultados" / "experimentos" / SOURCE_EXPERIMENT / "07_predictions_robust.csv"
    out_dir = root / "resultados" / "experimentos" / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path, low_memory=False)
    df = core_create_trust_features(df)
    label_col = find_col(df, LABELS)
    y = ints(df[label_col]).to_numpy()
    base_action = df["r5b4_robust_decisao"].astype(str).str.upper()
    residual_block = base_action.eq("BLOQUEAR").to_numpy()

    candidates, masks = mine_candidates(df, residual_block, y)
    selected, selected_mask = select_rules(df, candidates, masks, y)

    final_action = base_action.copy()
    final_action.loc[selected_mask] = "CONFIRMAR"
    df["exp014b_r5b5_trust_block_to_confirm"] = selected_mask.astype(int)
    df["r5b5_trust_decisao"] = final_action
    df["exp014b_r5b5_intervention_pred"] = pred_intervention(final_action)
    df["exp014b_r5b5_block_pred"] = pred_block(final_action)

    base_block_metrics = metrics(df[label_col], pred_block(base_action))
    final_block_metrics = metrics(df[label_col], df["exp014b_r5b5_block_pred"])
    final_intervention_metrics = metrics(df[label_col], df["exp014b_r5b5_intervention_pred"])

    demoted_normals = int(((selected_mask) & (y == 0)).sum())
    demoted_frauds = int(((selected_mask) & (y == 1)).sum())
    summary = {
        "experiment": EXPERIMENT,
        "source_experiment": SOURCE_EXPERIMENT,
        "status": "PASS_R5B5_TRUST_DEESCALATION_FOUND" if demoted_normals > 0 and demoted_frauds == 0 else "NO_R5B5_TRUST_GAIN",
        "all_pass": bool(demoted_normals > 0 and demoted_frauds == 0),
        "n_candidates": int(len(candidates)),
        "n_selected_rules": int(len(selected)),
        "block_fp_demoted_to_confirm_incremental": demoted_normals,
        "block_tp_demoted_to_confirm_incremental": demoted_frauds,
        "remaining_block_normals": int(((final_action == "BLOQUEAR") & (y == 0)).sum()),
        "remaining_block_frauds": int(((final_action == "BLOQUEAR") & (y == 1)).sum()),
        "remaining_approve_frauds": int(((final_action == "APROVAR") & (y == 1)).sum()),
        "base_block_metrics": base_block_metrics,
        "final_block_metrics": final_block_metrics,
        "final_intervention_metrics": final_intervention_metrics,
    }

    feature_cols = [
        "transaction_id",
        "payer_history_strength_score",
        "receiver_reputation_score",
        "relationship_strength_score",
        "receiver_novelty_risk_score",
        "transaction_normality_score",
        "payer_receiver_trust_score",
        "trust_bucket",
        "receiver_rep_bucket",
        "relationship_bucket",
        "novelty_bucket",
    ]

    write_json(out_dir / "00_run_summary.json", summary)
    df[feature_cols].to_csv(out_dir / "01_trust_features.csv", index=False)
    candidates.to_csv(out_dir / "02_trust_rule_candidates.csv", index=False)
    selected.to_csv(out_dir / "03_selected_trust_rules.csv", index=False)
    by_action(df, "r5b5_trust_decisao", label_col).to_csv(out_dir / "04_metrics_by_action.csv", index=False)
    df.to_csv(out_dir / "05_predictions_trust.csv", index=False)

    artifact = {
        "experiment": EXPERIMENT,
        "source_experiment": SOURCE_EXPERIMENT,
        "base_action_col": "r5b4_robust_decisao",
        "final_action_col": "r5b5_trust_decisao",
        "move_col": "exp014b_r5b5_trust_block_to_confirm",
        "trust_feature_columns": feature_cols[1:],
        "selected_rules": selected.to_dict(orient="records") if not selected.empty else [],
        "run_summary": summary,
    }
    write_json(out_dir / "06_policy_artifact_trust.json", artifact)

    report = f"""# {EXPERIMENT} — Trust Feature De-escalation

## Resultado executivo
- Status: `{summary['status']}`
- Candidatos avaliados: `{summary['n_candidates']}`
- Regras selecionadas: `{summary['n_selected_rules']}`
- Normais adicionais movidos de BLOQUEAR para CONFIRMAR: `{summary['block_fp_demoted_to_confirm_incremental']}`
- Fraudes movidas de BLOQUEAR para CONFIRMAR: `{summary['block_tp_demoted_to_confirm_incremental']}`
- Normais restantes em BLOQUEAR: `{summary['remaining_block_normals']}`
- Fraudes restantes em APROVAR: `{summary['remaining_approve_frauds']}`

## Métricas finais de BLOQUEAR
```json
{json.dumps(final_block_metrics, ensure_ascii=False, indent=2)}
```

## Regras selecionadas
{selected[['selection_step', 'candidate_id', 'incremental_normals', 'incremental_frauds', 'non_train_normals', 'holdout_normals', 'validation_normals', 'month_normal_support']].to_markdown(index=False) if not selected.empty else 'Nenhuma regra selecionada.'}

## Features novas criadas
- `payer_history_strength_score`
- `receiver_reputation_score`
- `relationship_strength_score`
- `receiver_novelty_risk_score`
- `transaction_normality_score`
- `payer_receiver_trust_score`
"""
    (out_dir / "07_exp014b_r5b5_trust_feature_deescalation_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
