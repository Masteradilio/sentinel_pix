#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R5B14 - Operational zero-FN replay.

Congela o candidato R5B13 sem selecao por label:
1. reaplica R5B12 por IDs congelados de CONFIRMAR -> BLOQUEAR;
2. aplica duas regras explicitas APROVAR -> BLOQUEAR;
3. compensa com regra operacional CONFIRMAR -> APROVAR:
   lgbm_raw <= 0.00001966.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "EXP-014B-R5B14-OPERATIONAL-ZERO-FN-REPLAY"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / EXPERIMENT
CANDIDATE_DIR = PROJECT_ROOT / "backend" / "artefatos_candidatos" / "exp014b_r5b14_operational_zero_fn"

INPUT_FILE = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R4G-FAST-FROZEN" / "06_predictions_frozen.csv"
R5B13_ARTIFACT = (
    PROJECT_ROOT
    / "resultados"
    / "experimentos"
    / "EXP-014B-R5B13-R4G-ZERO-FN-SWAP"
    / "05_policy_artifact_zero_fn_swap.json"
)

LABEL_COL = "is_fraud"
BASE_ACTION_COL = "r4g_fast_frozen_decisao_recommended"
FINAL_ACTION_COL = "r5b14_operational_zero_fn_decisao"
R5B12_MOVE_COL = "exp014b_r5b14_r5b12_confirm_to_block"
APPROVE_TO_BLOCK_COL = "exp014b_r5b14_approve_to_block"
COMPENSATION_COL = "exp014b_r5b14_confirm_to_approve_low_lgbm_raw"

COMPENSATION_LGBM_RAW_MAX = 0.00001966
TARGET_FPR = 0.01

R5B12_CONFIRM_TO_BLOCK_RULES = [
    {
        "rule_id": "R5B12_01_LGBM_RAW_HIGH",
        "description": "lgbm_raw >= 0.10711783",
    },
    {
        "rule_id": "R5B12_02_SCORE_2_3_LGBM_R4_HIGH",
        "description": "score_bin == score_2_3 AND lgbm_r4_score >= 0.475472966916",
    },
    {
        "rule_id": "R5B12_03_SCORE_2_3_LGBM_R4_MED",
        "description": "score_bin == score_2_3 AND lgbm_r4_score >= 0.318070929491",
    },
    {
        "rule_id": "R5B12_04_DOC_PHONE_HIGH_PAYER_COUNT",
        "description": "ds_tipo_chave_norm == DOCUMENTO_TELEFONE AND qtd_pix_pagador_180d >= 207",
    },
    {
        "rule_id": "R5B12_05_OUTROS_RATIO_MAX_HIGH",
        "description": (
            "ds_tipo_chave_norm == OUTROS AND "
            "ratio_valor_maximo_pagador_180d >= 4.9674631165863596"
        ),
    },
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


def parse_rule(df: pd.DataFrame, description: str) -> pd.Series:
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    for path in [INPUT_FILE, R5B13_ARTIFACT]:
        if not path.exists():
            raise FileNotFoundError(path)

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    y = ints(df[LABEL_COL])
    action = actions(df[BASE_ACTION_COL]).copy()

    r5b12_mask = pd.Series(False, index=df.index)
    for rule in R5B12_CONFIRM_TO_BLOCK_RULES:
        mask = action.eq("CONFIRMAR") & parse_rule(df, rule["description"])
        r5b12_mask |= mask
        action.loc[mask] = "BLOQUEAR"

    r5b13 = read_json(R5B13_ARTIFACT)
    approve_to_block = pd.Series(False, index=df.index)
    for rule in r5b13.get("selected_approve_to_block_rules", []):
        mask = action.eq("APROVAR") & parse_rule(df, str(rule["description"]))
        approve_to_block |= mask
        action.loc[mask] = "BLOQUEAR"

    compensation = (
        action.eq("CONFIRMAR")
        & pd.to_numeric(df["lgbm_raw"], errors="coerce").le(COMPENSATION_LGBM_RAW_MAX)
    )
    action.loc[compensation] = "APROVAR"

    df[R5B12_MOVE_COL] = r5b12_mask.astype(int)
    df[APPROVE_TO_BLOCK_COL] = approve_to_block.astype(int)
    df[COMPENSATION_COL] = compensation.astype(int)
    df[FINAL_ACTION_COL] = action
    df["exp014b_r5b14_intervention_pred"] = intervention_pred(action)
    df["exp014b_r5b14_block_pred"] = block_pred(action)

    final_intervention = metrics(df[LABEL_COL], df["exp014b_r5b14_intervention_pred"])
    final_block = metrics(df[LABEL_COL], df["exp014b_r5b14_block_pred"])
    by_action = action_table(df, FINAL_ACTION_COL)

    summary = {
        "experiment": EXPERIMENT,
        "status": "PASS_R5B14_OPERATIONAL_ZERO_FN_REPLAY"
        if final_intervention["fn"] == 0 and final_intervention["fpr"] < TARGET_FPR
        else "FAIL_R5B14_OPERATIONAL_ZERO_FN_REPLAY",
        "input_file": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
        "base_action_col": BASE_ACTION_COL,
        "final_action_col": FINAL_ACTION_COL,
        "r5b12_confirm_to_block": {
            "rows": int(r5b12_mask.sum()),
            "frauds": int((r5b12_mask & (y == 1)).sum()),
            "normals": int((r5b12_mask & (y == 0)).sum()),
        },
        "approve_to_block": {
            "rows": int(approve_to_block.sum()),
            "frauds": int((approve_to_block & (y == 1)).sum()),
            "normals": int((approve_to_block & (y == 0)).sum()),
        },
        "confirm_to_approve_compensation": {
            "rule": f"remaining CONFIRMAR AND lgbm_raw <= {COMPENSATION_LGBM_RAW_MAX}",
            "rows": int(compensation.sum()),
            "frauds": int((compensation & (y == 1)).sum()),
            "normals": int((compensation & (y == 0)).sum()),
        },
        "remaining_approve_frauds": int((action.eq("APROVAR") & (y == 1)).sum()),
        "remaining_confirm_frauds": int((action.eq("CONFIRMAR") & (y == 1)).sum()),
        "final_intervention_metrics": final_intervention,
        "final_block_metrics": final_block,
        "global_gates": {
            "fpr_lt_1pct": final_intervention["fpr"] < TARGET_FPR,
            "fn_eq_0": final_intervention["fn"] == 0,
        },
    }

    policy = {
        "artifact_type": "operational_zero_fn_policy_candidate",
        "experiment": EXPERIMENT,
        "status": "CANDIDATE_NOT_PRODUCTION_ACTIVE",
        "base_policy": "EXP-014B-R4G-FAST-FROZEN",
        "layers": [
            {
                "layer": "R5B12_CONFIRM_TO_BLOCK",
                "rules": R5B12_CONFIRM_TO_BLOCK_RULES,
            },
            {
                "layer": "R5B13_APPROVE_TO_BLOCK",
                "rules": r5b13.get("selected_approve_to_block_rules", []),
            },
            {
                "layer": "R5B14_CONFIRM_TO_APPROVE_COMPENSATION",
                "rule": {
                    "base_action": "CONFIRMAR",
                    "target_action": "APROVAR",
                    "description": f"lgbm_raw <= {COMPENSATION_LGBM_RAW_MAX}",
                    "lgbm_raw_max": COMPENSATION_LGBM_RAW_MAX,
                },
            },
        ],
        "metrics": {
            "final_intervention": final_intervention,
            "final_block": final_block,
        },
        "promotion_gates": [
            "Executar replay E2E no PipelineOrquestrador.",
            "Revisar semanticamente as regras APROVAR -> BLOQUEAR e a compensacao low-lgbm.",
        ],
    }

    write_json(OUT_DIR / "00_run_summary.json", summary)
    by_action.to_csv(OUT_DIR / "01_metrics_by_action.csv", index=False)
    df.loc[r5b12_mask | approve_to_block | compensation].to_csv(OUT_DIR / "02_moved_cases.csv", index=False)
    write_json(OUT_DIR / "03_policy_artifact_operational_zero_fn.json", policy)
    write_json(CANDIDATE_DIR / "operational_zero_fn_policy_candidate.json", policy)

    report = f"""# {EXPERIMENT} - operational zero-FN replay

## Resultado executivo
- Status: `{summary['status']}`
- Regra de compensacao: `lgbm_raw <= {COMPENSATION_LGBM_RAW_MAX}`
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
Este replay remove a dependencia de label da compensacao R5B13. A camada R5B12
ainda e reaplicada por IDs congelados; antes de runtime, ela precisa ser
substituida por regras de negocio congeladas ou por um replay E2E equivalente.
"""
    report = report.replace(
        "A camada R5B12\nainda e reaplicada por IDs congelados; antes de runtime, ela precisa ser\nsubstituida por regras de negocio congeladas ou por um replay E2E equivalente.",
        "Todas as camadas deste replay usam regras explicitas. Antes de runtime,\na politica ainda precisa passar por replay E2E no PipelineOrquestrador.",
    )
    (OUT_DIR / "04_exp014b_r5b14_operational_zero_fn_replay_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
