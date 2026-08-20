#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R5B8 — Broad Residual Rule Mining.

Minera regras robustas sobre o residual BLOQUEAR pós R5B5, usando features de
trust e variáveis apontadas pelo shadow R5B7. O objetivo é encontrar novas
ilhas de baixo risco para BLOQUEAR -> CONFIRMAR com zero fraude demovida.
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp_014b_r5b2_tune_policy import LABELS, find_col, ints, metrics, pred_block, pred_intervention


EXPERIMENT = "EXP-014B-R5B8-BROAD-RESIDUAL-RULE-MINING"
SOURCE_EXPERIMENT = "EXP-014B-R5B5-TRUST-FEATURE-DEESCALATION"
INPUT_FILE = PROJECT_ROOT / "resultados" / "experimentos" / SOURCE_EXPERIMENT / "05_predictions_trust.csv"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / EXPERIMENT
BASE_ACTION_COL = "r5b5_trust_decisao"
FINAL_ACTION_COL = "r5b8_broad_rules_decisao"
MOVE_COL = "exp014b_r5b8_broad_rules_block_to_confirm"


NUMERIC_FEATURES = [
    "lgbm_raw",
    "lgbm_mapped",
    "topaz_risk_score",
    "transaction_normality_score",
    "receiver_reputation_score",
    "payer_history_strength_score",
    "qtd_pix_pagador_180d",
    "valor_total_pagador_180d",
    "valor_total_pagador_90d",
    "valor_maximo_pix_pagador_180d",
    "ratio_valor_media_pagador_90d",
    "valor_total_recebido_30d",
    "dias_desde_primeiro_envio_recebedor",
    "soma_recebedores_distintos_dia_180d",
    "vl_pix",
]

BIN_FEATURES = [
    "lgbm_raw",
    "topaz_risk_score",
    "transaction_normality_score",
    "receiver_reputation_score",
    "payer_history_strength_score",
    "qtd_pix_pagador_180d",
    "valor_total_pagador_180d",
    "valor_total_recebido_30d",
]

CATEGORICAL_FEATURES = [
    "trust_bucket",
    "receiver_rep_bucket",
    "relationship_bucket",
    "novelty_bucket",
    "lgbm_bin",
    "qtd_rec_bin",
    "valor_rec_bin",
    "first_receiver_flag_real",
    "value_band",
    "periodo_dia",
    "ds_tipo_chave_norm",
]


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def norm_action(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().str.strip()


def safe_candidate_part(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value))


def add_bins(df: pd.DataFrame, residual_mask: np.ndarray) -> tuple[pd.DataFrame, list[str], dict[str, list[float | str]]]:
    df = df.copy()
    bin_cols: list[str] = []
    bin_edges: dict[str, list[float | str]] = {}
    for col in BIN_FEATURES:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        residual_values = values.loc[residual_mask].replace([np.inf, -np.inf], np.nan).dropna()
        if residual_values.nunique() < 4:
            continue
        bin_col = f"{col}__qbin"
        try:
            _, edges = pd.qcut(residual_values, q=5, retbins=True, duplicates="drop")
        except ValueError:
            continue
        if len(edges) < 3:
            continue
        edges[0] = -np.inf
        edges[-1] = np.inf
        labels = [f"{col}_q{i}" for i in range(1, len(edges))]
        df[bin_col] = pd.cut(values, bins=edges, labels=labels, include_lowest=True).astype(str)
        bin_cols.append(bin_col)
        serializable_edges: list[float | str] = []
        for edge in edges:
            if np.isneginf(edge):
                serializable_edges.append("-inf")
            elif np.isposinf(edge):
                serializable_edges.append("inf")
            else:
                serializable_edges.append(round(float(edge), 8))
        bin_edges[bin_col] = serializable_edges
    return df, bin_cols, bin_edges


def make_candidate(mask: np.ndarray, y: np.ndarray, candidate_id: str, description: str, rule_type: str) -> dict[str, Any] | None:
    n = int(mask.sum())
    if n <= 0:
        return None
    normals = int((mask & (y == 0)).sum())
    frauds = int((mask & (y == 1)).sum())
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


def mine_candidates(df: pd.DataFrame, residual_mask: np.ndarray, y: np.ndarray, bin_cols: list[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    candidates: list[dict[str, Any]] = []
    masks: dict[str, np.ndarray] = {}

    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        residual_values = values.loc[residual_mask].replace([np.inf, -np.inf], np.nan).dropna()
        thresholds = sorted({round(float(residual_values.quantile(q)), 8) for q in np.linspace(0.10, 0.90, 9) if residual_values.size})
        arr = values.to_numpy()
        for th in thresholds:
            for op in (">=", "<="):
                mask = residual_mask & np.isfinite(arr) & ((arr >= th) if op == ">=" else (arr <= th))
                cid = f"num__{col}__{op}{th:g}"
                desc = f"{col} {op} {th:g}"
                cand = make_candidate(mask, y, cid, desc, "numeric_threshold")
                if cand:
                    candidates.append(cand)
                    masks[cid] = mask

    cat_cols = [c for c in CATEGORICAL_FEATURES + bin_cols if c in df.columns]
    for size in (1, 2):
        for cols in itertools.combinations(cat_cols, size):
            if any(df.loc[residual_mask, c].nunique(dropna=False) > 25 for c in cols):
                continue
            grouped = df.loc[residual_mask, list(cols)].fillna("<MISSING>").astype(str)
            grouped["_idx"] = grouped.index
            for vals, grp in grouped.groupby(list(cols), dropna=False):
                vals = vals if isinstance(vals, tuple) else (vals,)
                if len(grp) < 5:
                    continue
                mask = np.zeros(len(df), dtype=bool)
                mask[grp["_idx"].to_numpy()] = True
                parts = [f"{c} == {v}" for c, v in zip(cols, vals)]
                safe = "__".join(f"{c}={safe_candidate_part(v)}" for c, v in zip(cols, vals))
                cid = f"cat{size}__{safe}"
                cand = make_candidate(mask, y, cid, " AND ".join(parts), f"categorical_{size}")
                if cand:
                    candidates.append(cand)
                    masks[cid] = mask

    cand_df = pd.DataFrame(candidates).drop_duplicates(subset=["candidate_id"])
    if cand_df.empty:
        return cand_df, {}
    cand_df = cand_df.sort_values(["fraud_count", "normal_count", "normal_precision"], ascending=[True, False, False])
    keep = set(cand_df.head(10000)["candidate_id"].astype(str))
    return cand_df[cand_df["candidate_id"].isin(keep)].reset_index(drop=True), {k: v for k, v in masks.items() if k in keep}


def support(df: pd.DataFrame, mask: np.ndarray, y: np.ndarray) -> dict[str, int]:
    splits = df["temporal_split"].fillna("<MISSING>").astype(str).str.upper()
    months = pd.to_datetime(df["event_datetime"], errors="coerce").dt.to_period("M").astype(str)
    out: dict[str, int] = {}
    for split in ["TRAIN", "VALIDATION", "HOLDOUT"]:
        split_mask = splits.eq(split).to_numpy()
        out[f"{split.lower()}_normals"] = int((mask & split_mask & (y == 0)).sum())
        out[f"{split.lower()}_frauds"] = int((mask & split_mask & (y == 1)).sum())
    out["non_train_normals"] = out["validation_normals"] + out["holdout_normals"]
    out["non_train_frauds"] = out["validation_frauds"] + out["holdout_frauds"]
    out["month_normal_support"] = int(months[mask & (y == 0)].nunique())
    out["month_fraud_support"] = int(months[mask & (y == 1)].nunique())
    return out


def select_rules(df: pd.DataFrame, candidates: pd.DataFrame, masks: dict[str, np.ndarray], y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    selected_mask = np.zeros(len(df), dtype=bool)
    rows: list[dict[str, Any]] = []
    remaining = candidates.copy()
    for step in range(1, 101):
        best = None
        best_mask = None
        best_score = None
        for _, row in remaining.iterrows():
            cid = str(row["candidate_id"])
            mask = masks[cid] & (~selected_mask)
            normals = int((mask & (y == 0)).sum())
            frauds = int((mask & (y == 1)).sum())
            if normals < 5 or frauds > 0:
                continue
            sup = support(df, mask, y)
            if sup["non_train_frauds"] > 0 or sup["non_train_normals"] < 10 or sup["month_normal_support"] < 2:
                continue
            score = (
                sup["holdout_normals"] > 0,
                sup["validation_normals"] > 0,
                sup["non_train_normals"],
                sup["month_normal_support"],
                normals,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_mask = mask
                best = row.to_dict()
                best.update(sup)
                best["incremental_normals"] = normals
                best["incremental_frauds"] = frauds
        if best is None or best_mask is None:
            break
        selected_mask |= best_mask
        best["selection_step"] = step
        best["cumulative_normals"] = int((selected_mask & (y == 0)).sum())
        best["cumulative_frauds"] = int((selected_mask & (y == 1)).sum())
        rows.append(best)
        remaining = remaining[remaining["candidate_id"].astype(str) != str(best["candidate_id"])].reset_index(drop=True)
    return pd.DataFrame(rows), selected_mask


def by_action(df: pd.DataFrame, action_col: str, label_col: str) -> pd.DataFrame:
    out = df.groupby(action_col).agg(n_rows=(label_col, "size"), n_frauds=(label_col, "sum")).reset_index()
    out["n_normals"] = out["n_rows"] - out["n_frauds"]
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    label_col = find_col(df, LABELS)
    y = ints(df[label_col]).to_numpy()
    base_action = norm_action(df[BASE_ACTION_COL])
    residual_mask = base_action.eq("BLOQUEAR").to_numpy()
    df, bin_cols, bin_edges = add_bins(df, residual_mask)

    candidates, masks = mine_candidates(df, residual_mask, y, bin_cols)
    selected, selected_mask = select_rules(df, candidates, masks, y)

    final_action = base_action.copy()
    final_action.loc[selected_mask] = "CONFIRMAR"
    df[MOVE_COL] = selected_mask.astype(int)
    df[FINAL_ACTION_COL] = final_action
    df["exp014b_r5b8_intervention_pred"] = pred_intervention(final_action)
    df["exp014b_r5b8_block_pred"] = pred_block(final_action)

    demoted_normals = int((selected_mask & (y == 0)).sum())
    demoted_frauds = int((selected_mask & (y == 1)).sum())
    final_block_metrics = metrics(df[label_col], df["exp014b_r5b8_block_pred"])
    summary = {
        "experiment": EXPERIMENT,
        "source_experiment": SOURCE_EXPERIMENT,
        "status": "PASS_R5B8_BROAD_RULES_FOUND" if demoted_normals > 0 and demoted_frauds == 0 else "NO_R5B8_BROAD_RULE_GAIN",
        "all_pass": bool(demoted_normals > 0 and demoted_frauds == 0),
        "n_candidates": int(len(candidates)),
        "n_selected_rules": int(len(selected)),
        "derived_bin_columns": bin_cols,
        "derived_bin_edges": bin_edges,
        "block_fp_demoted_to_confirm_incremental": demoted_normals,
        "block_tp_demoted_to_confirm_incremental": demoted_frauds,
        "remaining_block_normals": int(((final_action == "BLOQUEAR") & (y == 0)).sum()),
        "remaining_block_frauds": int(((final_action == "BLOQUEAR") & (y == 1)).sum()),
        "remaining_approve_frauds": int(((final_action == "APROVAR") & (y == 1)).sum()),
        "base_block_metrics": metrics(df[label_col], pred_block(base_action)),
        "final_block_metrics": final_block_metrics,
        "final_intervention_metrics": metrics(df[label_col], df["exp014b_r5b8_intervention_pred"]),
    }

    write_json(OUT_DIR / "00_run_summary.json", summary)
    candidates.to_csv(OUT_DIR / "01_broad_rule_candidates.csv", index=False)
    selected.to_csv(OUT_DIR / "02_selected_broad_rules.csv", index=False)
    by_action(df, FINAL_ACTION_COL, label_col).to_csv(OUT_DIR / "03_metrics_by_action.csv", index=False)
    df.to_csv(OUT_DIR / "04_predictions_broad_rules.csv", index=False)
    write_json(OUT_DIR / "05_policy_artifact_broad_rules.json", {
        "experiment": EXPERIMENT,
        "source_experiment": SOURCE_EXPERIMENT,
        "base_action_col": BASE_ACTION_COL,
        "final_action_col": FINAL_ACTION_COL,
        "move_col": MOVE_COL,
        "derived_bin_edges": bin_edges,
        "selected_rules": selected.to_dict(orient="records") if not selected.empty else [],
        "run_summary": summary,
    })

    report = f"""# {EXPERIMENT} — Mineração ampla de regras residuais

## Resultado executivo
- Status: `{summary['status']}`
- Candidatos avaliados: `{summary['n_candidates']}`
- Regras selecionadas: `{summary['n_selected_rules']}`
- Normais adicionais movidos de BLOQUEAR para CONFIRMAR: `{summary['block_fp_demoted_to_confirm_incremental']}`
- Fraudes movidas de BLOQUEAR para CONFIRMAR: `{summary['block_tp_demoted_to_confirm_incremental']}`
- Normais restantes em BLOQUEAR: `{summary['remaining_block_normals']}`
- Fraudes restantes em BLOQUEAR: `{summary['remaining_block_frauds']}`
- Fraudes restantes em APROVAR: `{summary['remaining_approve_frauds']}`

## Métricas finais de BLOQUEAR
```json
{json.dumps(final_block_metrics, ensure_ascii=False, indent=2)}
```

## Regras selecionadas
{selected[['selection_step', 'candidate_id', 'incremental_normals', 'incremental_frauds', 'non_train_normals', 'holdout_normals', 'validation_normals', 'month_normal_support']].to_markdown(index=False) if not selected.empty else 'Nenhuma regra selecionada.'}

## Decisão técnica
Este experimento amplia a mineração do R5B5, mas preserva os mesmos critérios
conservadores: zero fraude demovida, suporte fora de treino e suporte em pelo
menos dois meses. Regras aprovadas aqui são candidatas para revisão manual antes
de qualquer integração de política.
"""
    (OUT_DIR / "06_exp014b_r5b8_broad_residual_rule_mining_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
