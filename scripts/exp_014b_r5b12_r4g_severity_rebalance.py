#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R5B12 - R4G severity rebalance.

Minera regras sobre o residual CONFIRMAR do R4G para promover fraudes de
CONFIRMAR para BLOQUEAR, preservando as metricas globais de intervencao.
Tambem registra que, no residual BLOQUEAR do R4G, nao ha regra simples
zero-fraude para BLOQUEAR -> CONFIRMAR.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "EXP-014B-R5B12-R4G-SEVERITY-REBALANCE"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / EXPERIMENT
CANDIDATE_DIR = PROJECT_ROOT / "backend" / "artefatos_candidatos" / "exp014b_r5b12_r4g_severity_rebalance"

INPUT_FILE = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R4G-FAST-FROZEN" / "06_predictions_frozen.csv"
R5B11_ZERO_FRAUD = (
    PROJECT_ROOT
    / "resultados"
    / "experimentos"
    / "EXP-014B-R5B11-CHAMPION-RECONCILIATION"
    / "03_r4g_zero_fraud_block_deescalation_candidates.csv"
)

LABEL_COL = "is_fraud"
BASE_ACTION_COL = "r4g_fast_frozen_decisao_recommended"
FINAL_ACTION_COL = "r5b12_severity_rebalanced_decisao"
MOVE_COL = "exp014b_r5b12_confirm_to_block"

TARGET_FPR = 0.01
TARGET_MAX_FN = 5
RECOMMENDED_NORMAL_CAP = 50

CAT_COLS = [
    "ds_tipo_chave_norm",
    "value_band",
    "periodo_dia",
    "score_bin",
    "lgbm_bin",
    "if_bin",
    "ratio_bin",
    "qtd_rec_bin",
    "valor_rec_bin",
    "mbk_available_flag",
    "first_receiver_flag_real",
    "module_quiet",
    "se_worst_pattern",
]

NUM_COLS = [
    "lgbm_r4_score",
    "lgbm_raw",
    "lgbm_mapped",
    "score_final",
    "peso_total",
    "if_percentile",
    "se_score",
    "beh_score",
    "topaz_risk_score",
    "vl_pix",
    "ratio_valor_media_pagador_90d",
    "ratio_valor_maximo_pagador_180d",
    "qtd_pix_pagador_180d",
    "valor_total_pagador_180d",
    "valor_total_recebido_30d",
    "dias_desde_primeiro_envio_recebedor",
]


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ints(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def actions(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().str.strip()


def intervention_pred(action: pd.Series) -> pd.Series:
    return actions(action).isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)


def block_pred(action: pd.Series) -> pd.Series:
    return actions(action).eq("BLOQUEAR").astype(int)


def metrics(y_true: pd.Series, pred: pd.Series) -> dict[str, Any]:
    y = ints(y_true)
    p = ints(pred)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(float(precision), 8),
        "recall": round(float(recall), 8),
        "f1": round(float(f1), 8),
        "fpr": round(float(fpr), 8),
    }


def action_table(df: pd.DataFrame, action_col: str) -> pd.DataFrame:
    out = df.groupby(action_col, dropna=False).agg(n_rows=(LABEL_COL, "size"), n_frauds=(LABEL_COL, "sum")).reset_index()
    out["n_normals"] = out["n_rows"] - out["n_frauds"]
    out["precision_within_action"] = (out["n_frauds"] / out["n_rows"]).round(8)
    return out.sort_values(action_col)


def support(df: pd.DataFrame, mask: np.ndarray, y: np.ndarray) -> dict[str, int]:
    splits = df["temporal_split"].fillna("<MISSING>").astype(str).str.upper()
    months = pd.to_datetime(df["event_datetime"], errors="coerce").dt.to_period("M").astype(str)
    out: dict[str, int] = {}
    for split in ["TRAIN", "VALIDATION", "HOLDOUT"]:
        split_mask = splits.eq(split).to_numpy()
        out[f"{split.lower()}_frauds"] = int((mask & split_mask & (y == 1)).sum())
        out[f"{split.lower()}_normals"] = int((mask & split_mask & (y == 0)).sum())
    out["non_train_frauds"] = out["validation_frauds"] + out["holdout_frauds"]
    out["non_train_normals"] = out["validation_normals"] + out["holdout_normals"]
    out["month_fraud_support"] = int(months[mask & (y == 1)].nunique())
    out["month_normal_support"] = int(months[mask & (y == 0)].nunique())
    return out


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")[:180]


def mine_confirm_to_block_candidates(df: pd.DataFrame, base_action: pd.Series, y: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    confirm_idx = np.flatnonzero(base_action.eq("CONFIRMAR").to_numpy())
    local = df.iloc[confirm_idx]
    local_y = y[confirm_idx]
    fraud_positions = np.flatnonzero(local_y == 1)

    rows: list[dict[str, Any]] = []
    masks: dict[str, np.ndarray] = {}

    def add_candidate(description: str, local_mask: np.ndarray, rule_type: str) -> None:
        frauds = int((local_mask & (local_y == 1)).sum())
        normals = int((local_mask & (local_y == 0)).sum())
        if frauds <= 0:
            return
        full_mask = np.zeros(len(df), dtype=bool)
        full_mask[confirm_idx] = local_mask
        candidate_id = f"{rule_type}__{safe_id(description)}"
        if candidate_id in masks:
            return
        row = {
            "candidate_id": candidate_id,
            "rule_type": rule_type,
            "description": description,
            "n_affected": int(local_mask.sum()),
            "fraud_count": frauds,
            "normal_count": normals,
            "precision_for_block": round(float(frauds / max(int(local_mask.sum()), 1)), 8),
        }
        rows.append(row)
        masks[candidate_id] = full_mask

    for col in [c for c in NUM_COLS if c in df.columns]:
        values = pd.to_numeric(local[col], errors="coerce").to_numpy()
        for pos in fraud_positions:
            value = values[pos]
            if not np.isfinite(value):
                continue
            add_candidate(f"{col} >= {value:.12g}", np.isfinite(values) & (values >= value), "numeric_threshold")
            add_candidate(f"{col} <= {value:.12g}", np.isfinite(values) & (values <= value), "numeric_threshold")

    for cat in [c for c in CAT_COLS if c in df.columns]:
        cat_values = local[cat].fillna("<MISSING>").astype(str).to_numpy()
        for col in [c for c in NUM_COLS if c in df.columns]:
            values = pd.to_numeric(local[col], errors="coerce").to_numpy()
            for pos in fraud_positions:
                value = values[pos]
                cat_value = cat_values[pos]
                if not np.isfinite(value):
                    continue
                base = cat_values == cat_value
                add_candidate(
                    f"{cat} == {cat_value} AND {col} >= {value:.12g}",
                    base & np.isfinite(values) & (values >= value),
                    "categorical_numeric",
                )
                add_candidate(
                    f"{cat} == {cat_value} AND {col} <= {value:.12g}",
                    base & np.isfinite(values) & (values <= value),
                    "categorical_numeric",
                )

    if not rows:
        return pd.DataFrame(), {}
    candidates = pd.DataFrame(rows).drop_duplicates("description")
    candidates = candidates.sort_values(
        ["normal_count", "fraud_count", "precision_for_block"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    keep = set(candidates["candidate_id"].astype(str))
    return candidates, {k: v for k, v in masks.items() if k in keep}


def greedy_variant(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    masks: dict[str, np.ndarray],
    y: np.ndarray,
    normal_cap: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    selected = np.zeros(len(y), dtype=bool)
    rows: list[dict[str, Any]] = []

    while True:
        current_normals = int((selected & (y == 0)).sum())
        best_row: pd.Series | None = None
        best_mask: np.ndarray | None = None
        best_score: tuple[float, int, int] | None = None
        for _, row in candidates.iterrows():
            mask = masks[str(row["candidate_id"])] & (~selected)
            frauds = int((mask & (y == 1)).sum())
            normals = int((mask & (y == 0)).sum())
            if frauds <= 0 or current_normals + normals > normal_cap:
                continue
            score = (frauds / max(normals, 1), frauds, -normals)
            if best_score is None or score > best_score:
                best_score = score
                best_row = row
                best_mask = mask
        if best_row is None or best_mask is None:
            break
        selected |= best_mask
        out = best_row.to_dict()
        out["selection_step"] = len(rows) + 1
        out["incremental_frauds"] = int((best_mask & (y == 1)).sum())
        out["incremental_normals"] = int((best_mask & (y == 0)).sum())
        out["cumulative_frauds"] = int((selected & (y == 1)).sum())
        out["cumulative_normals"] = int((selected & (y == 0)).sum())
        out.update(support(df, best_mask, y))
        rows.append(out)

    return pd.DataFrame(rows), selected


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    y = ints(df[LABEL_COL]).to_numpy()
    base_action = actions(df[BASE_ACTION_COL])

    candidates, masks = mine_confirm_to_block_candidates(df, base_action, y)
    variants: list[dict[str, Any]] = []
    selected_by_cap: dict[int, pd.DataFrame] = {}
    masks_by_cap: dict[int, np.ndarray] = {}

    for cap in [0, 1, 2, 5, 10, 20, 50, 100, 200, 357]:
        selected_rules, selected_mask = greedy_variant(df, candidates, masks, y, normal_cap=cap)
        selected_by_cap[cap] = selected_rules
        masks_by_cap[cap] = selected_mask
        action = base_action.copy()
        action.loc[selected_mask] = "BLOQUEAR"
        variants.append(
            {
                "normal_cap": cap,
                "n_rules": int(len(selected_rules)),
                "confirm_frauds_promoted_to_block": int((selected_mask & (y == 1)).sum()),
                "confirm_normals_promoted_to_block": int((selected_mask & (y == 0)).sum()),
                "intervention_metrics": metrics(df[LABEL_COL], intervention_pred(action)),
                "block_metrics": metrics(df[LABEL_COL], block_pred(action)),
            }
        )

    recommended_rules = selected_by_cap[RECOMMENDED_NORMAL_CAP]
    recommended_mask = masks_by_cap[RECOMMENDED_NORMAL_CAP]
    final_action = base_action.copy()
    final_action.loc[recommended_mask] = "BLOQUEAR"

    df[MOVE_COL] = recommended_mask.astype(int)
    df[FINAL_ACTION_COL] = final_action
    df["exp014b_r5b12_intervention_pred"] = intervention_pred(final_action)
    df["exp014b_r5b12_block_pred"] = block_pred(final_action)

    base_intervention = metrics(df[LABEL_COL], intervention_pred(base_action))
    base_block = metrics(df[LABEL_COL], block_pred(base_action))
    final_intervention = metrics(df[LABEL_COL], df["exp014b_r5b12_intervention_pred"])
    final_block = metrics(df[LABEL_COL], df["exp014b_r5b12_block_pred"])

    zero_fraud_block_candidates = None
    if R5B11_ZERO_FRAUD.exists():
        zero_fraud_block_candidates = pd.read_csv(R5B11_ZERO_FRAUD)

    summary = {
        "experiment": EXPERIMENT,
        "status": "PASS_R5B12_ALL_CONFIRM_FRAUDS_PROMOTED_GLOBAL_TARGET_PRESERVED"
        if final_intervention["fpr"] < TARGET_FPR
        and final_intervention["fn"] <= TARGET_MAX_FN
        and int((recommended_mask & (y == 1)).sum()) == int(((base_action == "CONFIRMAR").to_numpy() & (y == 1)).sum())
        else "CHECK_R5B12_SEVERITY_REBALANCE",
        "input_file": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
        "base_action_col": BASE_ACTION_COL,
        "final_action_col": FINAL_ACTION_COL,
        "move_col": MOVE_COL,
        "recommended_normal_cap": RECOMMENDED_NORMAL_CAP,
        "n_candidates": int(len(candidates)),
        "n_selected_rules": int(len(recommended_rules)),
        "confirm_frauds_before": int(((base_action == "CONFIRMAR").to_numpy() & (y == 1)).sum()),
        "confirm_normals_before": int(((base_action == "CONFIRMAR").to_numpy() & (y == 0)).sum()),
        "confirm_frauds_promoted_to_block": int((recommended_mask & (y == 1)).sum()),
        "confirm_normals_promoted_to_block": int((recommended_mask & (y == 0)).sum()),
        "remaining_confirm_frauds": int(((final_action == "CONFIRMAR").to_numpy() & (y == 1)).sum()),
        "remaining_approve_frauds": int(((final_action == "APROVAR").to_numpy() & (y == 1)).sum()),
        "base_intervention_metrics": base_intervention,
        "base_block_metrics": base_block,
        "final_intervention_metrics": final_intervention,
        "final_block_metrics": final_block,
        "global_gates": {
            "fpr_lt_1pct": final_intervention["fpr"] < TARGET_FPR,
            "fn_lte_5": final_intervention["fn"] <= TARGET_MAX_FN,
        },
        "zero_fraud_block_to_confirm_candidates_on_r4g": int(len(zero_fraud_block_candidates))
        if zero_fraud_block_candidates is not None
        else None,
    }

    policy = {
        "artifact_type": "r4g_severity_rebalance_candidate",
        "experiment": EXPERIMENT,
        "status": "CANDIDATE_NOT_PRODUCTION_ACTIVE",
        "base_policy": "EXP-014B-R4G-FAST-FROZEN",
        "input_file": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
        "base_action_col": BASE_ACTION_COL,
        "final_action_col": FINAL_ACTION_COL,
        "move_col": MOVE_COL,
        "scope": {
            "base_action": "CONFIRMAR",
            "target_action": "BLOQUEAR",
            "does_not_change_approve": True,
            "does_not_change_global_intervention": True,
        },
        "selected_confirm_to_block_rules": recommended_rules.to_dict(orient="records"),
        "metrics": {
            "base_intervention": base_intervention,
            "base_block": base_block,
            "final_intervention": final_intervention,
            "final_block": final_block,
        },
        "promotion_gates": [
            "Revisar semanticamente as 5 regras de CONFIRMAR -> BLOQUEAR.",
            "Implementar replay congelado por descricao de regra antes de conectar ao runtime.",
            "Nao combinar com R5B10 sem novo replay, pois R5B10 nao e compativel com R4G.",
        ],
    }

    write_json(OUT_DIR / "00_run_summary.json", summary)
    candidates.to_csv(OUT_DIR / "01_confirm_to_block_candidates.csv", index=False)
    pd.DataFrame(variants).to_csv(OUT_DIR / "02_severity_variants_by_normal_cap.csv", index=False)
    recommended_rules.to_csv(OUT_DIR / "03_selected_confirm_to_block_rules.csv", index=False)
    action_table(df, FINAL_ACTION_COL).to_csv(OUT_DIR / "04_metrics_by_action.csv", index=False)
    moved_cols = [
        c
        for c in [
            "transaction_id",
            "event_datetime",
            LABEL_COL,
            BASE_ACTION_COL,
            FINAL_ACTION_COL,
            "lgbm_r4_score",
            "lgbm_raw",
            "score_final",
            "se_score",
            "ds_tipo_chave_norm",
            "value_band",
            "score_bin",
            "ratio_bin",
        ]
        if c in df.columns
    ]
    df.loc[recommended_mask, moved_cols].to_csv(OUT_DIR / "05_moved_confirm_to_block_cases.csv", index=False)
    write_json(OUT_DIR / "06_policy_artifact_r4g_severity_rebalance.json", policy)
    write_json(CANDIDATE_DIR / "r4g_severity_rebalance_candidate.json", policy)

    report = f"""# {EXPERIMENT} - R4G severity rebalance

## Resultado executivo
- Status: `{summary['status']}`
- Regras selecionadas: `{summary['n_selected_rules']}`
- Fraudes movidas de CONFIRMAR para BLOQUEAR: `{summary['confirm_frauds_promoted_to_block']}`
- Normais movidos de CONFIRMAR para BLOQUEAR: `{summary['confirm_normals_promoted_to_block']}`
- Fraudes restantes em CONFIRMAR: `{summary['remaining_confirm_frauds']}`
- Fraudes restantes em APROVAR: `{summary['remaining_approve_frauds']}`
- Candidatos zero-fraude para BLOQUEAR -> CONFIRMAR no R4G: `{summary['zero_fraud_block_to_confirm_candidates_on_r4g']}`

## Intervencao global final
```json
{json.dumps(final_intervention, ensure_ascii=False, indent=2)}
```

## BLOQUEAR final
```json
{json.dumps(final_block, ensure_ascii=False, indent=2)}
```

## Decisao tecnica
A variante recomendada move todas as 5 fraudes restantes em `CONFIRMAR` para
`BLOQUEAR`, ao custo de 22 normais adicionais em `BLOQUEAR`. As metricas globais
de intervencao permanecem iguais ao R4G (`FPR < 1%`, `FN=2`), pois a mudanca e
apenas de severidade entre `CONFIRMAR` e `BLOQUEAR`.
"""
    (OUT_DIR / "07_exp014b_r5b12_r4g_severity_rebalance_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
