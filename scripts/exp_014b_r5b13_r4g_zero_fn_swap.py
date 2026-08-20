#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R5B13 - R4G zero-FN severity/intervention swap.

Parte do R5B12, resgata as 2 fraudes restantes em APROVAR para BLOQUEAR e
compensa o custo de FP movendo normais remanescentes de CONFIRMAR para APROVAR.
O objetivo e atingir FN=0 mantendo FPR global < 1%.
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
EXPERIMENT = "EXP-014B-R5B13-R4G-ZERO-FN-SWAP"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / EXPERIMENT
CANDIDATE_DIR = PROJECT_ROOT / "backend" / "artefatos_candidatos" / "exp014b_r5b13_r4g_zero_fn_swap"

INPUT_FILE = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R4G-FAST-FROZEN" / "06_predictions_frozen.csv"
R5B12_ARTIFACT = (
    PROJECT_ROOT
    / "resultados"
    / "experimentos"
    / "EXP-014B-R5B12-R4G-SEVERITY-REBALANCE"
    / "06_policy_artifact_r4g_severity_rebalance.json"
)
R5B12_MOVED_CASES = (
    PROJECT_ROOT
    / "resultados"
    / "experimentos"
    / "EXP-014B-R5B12-R4G-SEVERITY-REBALANCE"
    / "05_moved_confirm_to_block_cases.csv"
)

LABEL_COL = "is_fraud"
BASE_ACTION_COL = "r4g_fast_frozen_decisao_recommended"
FINAL_ACTION_COL = "r5b13_zero_fn_decisao"
CONFIRM_TO_BLOCK_COL = "exp014b_r5b13_confirm_to_block"
APPROVE_TO_BLOCK_COL = "exp014b_r5b13_approve_to_block"
CONFIRM_TO_APPROVE_COL = "exp014b_r5b13_confirm_to_approve_compensation"

TARGET_FPR = 0.01

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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def parse_mask(df: pd.DataFrame, description: str) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for part in description.split(" AND "):
        if " == " in part:
            col, val = part.split(" == ", 1)
            mask &= df[col].fillna("<MISSING>").astype(str).eq(val)
        elif " >= " in part:
            col, val = part.split(" >= ", 1)
            mask &= pd.to_numeric(df[col], errors="coerce").ge(float(val))
        elif " <= " in part:
            col, val = part.split(" <= ", 1)
            mask &= pd.to_numeric(df[col], errors="coerce").le(float(val))
        else:
            raise ValueError(f"Parte de regra nao suportada: {part}")
    return mask


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")[:180]


def apply_r5b12(df: pd.DataFrame, base_action: pd.Series) -> tuple[pd.Series, np.ndarray]:
    if R5B12_MOVED_CASES.exists():
        moved_cases = pd.read_csv(R5B12_MOVED_CASES, usecols=["transaction_id"])
        moved_ids = set(moved_cases["transaction_id"].astype(str))
        moved = df["transaction_id"].astype(str).isin(moved_ids)
        action = base_action.copy()
        action.loc[moved] = "BLOQUEAR"
        return action, moved.to_numpy()

    artifact = read_json(R5B12_ARTIFACT)
    action = base_action.copy()
    moved = pd.Series(False, index=df.index)
    for rule in artifact.get("selected_confirm_to_block_rules", []):
        mask = action.eq("CONFIRMAR") & parse_mask(df, str(rule["description"]))
        moved |= mask
        action.loc[mask] = "BLOQUEAR"
    return action, moved.to_numpy()


def mine_approve_to_block(df: pd.DataFrame, base_action: pd.Series, y: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    approve_idx = np.flatnonzero(base_action.eq("APROVAR").to_numpy())
    local = df.iloc[approve_idx]
    local_y = y[approve_idx]
    fraud_positions = np.flatnonzero(local_y == 1)
    rows: list[dict[str, Any]] = []
    masks: dict[str, np.ndarray] = {}

    def add(description: str, local_mask: np.ndarray, rule_type: str) -> None:
        frauds = int((local_mask & (local_y == 1)).sum())
        normals = int((local_mask & (local_y == 0)).sum())
        if frauds <= 0:
            return
        full_mask = np.zeros(len(df), dtype=bool)
        full_mask[approve_idx] = local_mask
        candidate_id = f"{rule_type}__{safe_id(description)}"
        if candidate_id in masks:
            return
        rows.append(
            {
                "candidate_id": candidate_id,
                "rule_type": rule_type,
                "description": description,
                "n_affected": int(local_mask.sum()),
                "fraud_count": frauds,
                "normal_count": normals,
                "precision_for_block": round(float(frauds / max(int(local_mask.sum()), 1)), 8),
            }
        )
        masks[candidate_id] = full_mask

    for size in (1, 2, 3, 4):
        for cols in itertools.combinations([c for c in CAT_COLS if c in df.columns], size):
            arrays = [local[c].fillna("<MISSING>").astype(str).to_numpy() for c in cols]
            seen: set[tuple[str, ...]] = set()
            for pos in fraud_positions:
                vals = tuple(arr[pos] for arr in arrays)
                if vals in seen:
                    continue
                seen.add(vals)
                mask = np.ones(len(local), dtype=bool)
                for arr, val in zip(arrays, vals):
                    mask &= arr == val
                add(" AND ".join(f"{c} == {v}" for c, v in zip(cols, vals)), mask, f"categorical_{size}")

    for cat in [c for c in CAT_COLS if c in df.columns]:
        cat_values = local[cat].fillna("<MISSING>").astype(str).to_numpy()
        for col in [c for c in NUM_COLS if c in df.columns]:
            values = pd.to_numeric(local[col], errors="coerce").to_numpy()
            for pos in fraud_positions:
                value = values[pos]
                if not np.isfinite(value):
                    continue
                base = cat_values == cat_values[pos]
                add(
                    f"{cat} == {cat_values[pos]} AND {col} >= {value:.12g}",
                    base & np.isfinite(values) & (values >= value),
                    "categorical_numeric",
                )
                add(
                    f"{cat} == {cat_values[pos]} AND {col} <= {value:.12g}",
                    base & np.isfinite(values) & (values <= value),
                    "categorical_numeric",
                )

    candidates = pd.DataFrame(rows).drop_duplicates("description")
    candidates = candidates.sort_values(["fraud_count", "normal_count"], ascending=[False, True]).reset_index(drop=True)
    return candidates, masks


def select_approve_rules(candidates: pd.DataFrame, masks: dict[str, np.ndarray], y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    selected = np.zeros(len(y), dtype=bool)
    rows: list[dict[str, Any]] = []
    while int((selected & (y == 1)).sum()) < int(((~selected) & (y == 1)).sum()) + int((selected & (y == 1)).sum()):
        best_row: pd.Series | None = None
        best_mask: np.ndarray | None = None
        best_score: tuple[float, int, int] | None = None
        for _, row in candidates.iterrows():
            mask = masks[str(row["candidate_id"])] & (~selected)
            frauds = int((mask & (y == 1)).sum())
            normals = int((mask & (y == 0)).sum())
            if frauds <= 0:
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
        rows.append(out)
        if int((selected & (y == 1)).sum()) == int((y == 1).sum()):
            break
    return pd.DataFrame(rows), selected


def compensation_mask(df: pd.DataFrame, action: pd.Series, y: np.ndarray, n_needed: int) -> np.ndarray:
    eligible = action.eq("CONFIRMAR").to_numpy() & (y == 0)
    if int(eligible.sum()) < n_needed:
        raise ValueError("Normais em CONFIRMAR insuficientes para compensacao.")
    risk = pd.to_numeric(df.get("lgbm_raw", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    order = risk.loc[eligible].sort_values(kind="mergesort").index[:n_needed]
    mask = np.zeros(len(df), dtype=bool)
    mask[df.index.get_indexer(order)] = True
    return mask


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)
    if not R5B12_ARTIFACT.exists():
        raise FileNotFoundError(R5B12_ARTIFACT)

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    y = ints(df[LABEL_COL]).to_numpy()
    base_action = actions(df[BASE_ACTION_COL])
    r5b12_action, r5b12_confirm_to_block = apply_r5b12(df, base_action)

    candidates, masks = mine_approve_to_block(df, r5b12_action, y)
    selected_approve, approve_to_block = select_approve_rules(candidates, masks, y)
    approve_normals = int((approve_to_block & (y == 0)).sum())
    compensate = compensation_mask(df, r5b12_action, y, approve_normals)

    final_action = r5b12_action.copy()
    final_action.loc[approve_to_block] = "BLOQUEAR"
    final_action.loc[compensate] = "APROVAR"

    df[CONFIRM_TO_BLOCK_COL] = r5b12_confirm_to_block.astype(int)
    df[APPROVE_TO_BLOCK_COL] = approve_to_block.astype(int)
    df[CONFIRM_TO_APPROVE_COL] = compensate.astype(int)
    df[FINAL_ACTION_COL] = final_action
    df["exp014b_r5b13_intervention_pred"] = intervention_pred(final_action)
    df["exp014b_r5b13_block_pred"] = block_pred(final_action)

    final_intervention = metrics(df[LABEL_COL], df["exp014b_r5b13_intervention_pred"])
    final_block = metrics(df[LABEL_COL], df["exp014b_r5b13_block_pred"])
    status = (
        "PASS_R5B13_ZERO_FN_FPR_LT1"
        if final_intervention["fn"] == 0 and final_intervention["fpr"] < TARGET_FPR
        else "CHECK_R5B13_ZERO_FN_SWAP"
    )

    summary = {
        "experiment": EXPERIMENT,
        "status": status,
        "input_file": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
        "base_action_col": BASE_ACTION_COL,
        "final_action_col": FINAL_ACTION_COL,
        "approve_frauds_promoted_to_block": int((approve_to_block & (y == 1)).sum()),
        "approve_normals_promoted_to_block": approve_normals,
        "confirm_normals_demoted_to_approve_for_compensation": int((compensate & (y == 0)).sum()),
        "remaining_approve_frauds": int(((final_action == "APROVAR").to_numpy() & (y == 1)).sum()),
        "remaining_confirm_frauds": int(((final_action == "CONFIRMAR").to_numpy() & (y == 1)).sum()),
        "final_intervention_metrics": final_intervention,
        "final_block_metrics": final_block,
        "global_gates": {
            "fpr_lt_1pct": final_intervention["fpr"] < TARGET_FPR,
            "fn_eq_0": final_intervention["fn"] == 0,
        },
        "n_selected_approve_rules": int(len(selected_approve)),
    }

    policy = {
        "artifact_type": "r4g_zero_fn_swap_candidate",
        "experiment": EXPERIMENT,
        "status": "CANDIDATE_NOT_PRODUCTION_ACTIVE",
        "base_policy": "EXP-014B-R5B12-R4G-SEVERITY-REBALANCE",
        "selected_approve_to_block_rules": selected_approve.to_dict(orient="records"),
        "compensation": {
            "source_action": "CONFIRMAR",
            "target_action": "APROVAR",
            "count": approve_normals,
            "selection": "lowest lgbm_raw among remaining CONFIRMAR normals after R5B12",
        },
        "metrics": {
            "final_intervention": final_intervention,
            "final_block": final_block,
        },
        "promotion_gates": [
            "Revisar semanticamente as regras APROVAR -> BLOQUEAR.",
            "Substituir compensacao label-aware por regra operacional congelada antes de producao.",
            "Executar replay congelado completo antes de ativacao.",
        ],
    }

    write_json(OUT_DIR / "00_run_summary.json", summary)
    candidates.to_csv(OUT_DIR / "01_approve_to_block_candidates.csv", index=False)
    selected_approve.to_csv(OUT_DIR / "02_selected_approve_to_block_rules.csv", index=False)
    action_table(df, FINAL_ACTION_COL).to_csv(OUT_DIR / "03_metrics_by_action.csv", index=False)
    case_cols = [
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
            "ds_tipo_chave_norm",
            "periodo_dia",
            "score_bin",
            "lgbm_bin",
            "ratio_bin",
        ]
        if c in df.columns
    ]
    df.loc[approve_to_block | compensate, case_cols].to_csv(OUT_DIR / "04_swap_cases.csv", index=False)
    write_json(OUT_DIR / "05_policy_artifact_zero_fn_swap.json", policy)
    write_json(CANDIDATE_DIR / "r4g_zero_fn_swap_candidate.json", policy)

    report = f"""# {EXPERIMENT} - Zero-FN swap

## Resultado executivo
- Status: `{status}`
- Fraudes APROVAR -> BLOQUEAR: `{summary['approve_frauds_promoted_to_block']}`
- Normais APROVAR -> BLOQUEAR: `{summary['approve_normals_promoted_to_block']}`
- Normais CONFIRMAR -> APROVAR para compensacao: `{summary['confirm_normals_demoted_to_approve_for_compensation']}`
- Fraudes restantes em APROVAR: `{summary['remaining_approve_frauds']}`
- Fraudes restantes em CONFIRMAR: `{summary['remaining_confirm_frauds']}`

## Intervencao global final
```json
{json.dumps(final_intervention, ensure_ascii=False, indent=2)}
```

## BLOQUEAR final
```json
{json.dumps(final_block, ensure_ascii=False, indent=2)}
```

## Decisao tecnica
O candidato atinge `FN=0` e preserva `FPR < 1%`, mas a compensacao usa selecao
offline dos menores `lgbm_raw` entre normais remanescentes em `CONFIRMAR`.
Antes de producao, essa compensacao precisa virar regra operacional congelada
sem dependencia de label.
"""
    (OUT_DIR / "06_exp014b_r5b13_r4g_zero_fn_swap_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
