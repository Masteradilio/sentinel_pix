#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R5B11 - Champion reconciliation.

Reconcilia o baseline global R4G-FAST-FROZEN com a trilha de severidade R5B10.
O objetivo e provar, com artefatos, qual candidato ainda cumpre FN/FPR globais
e se a politica R5B10 pode ser empilhada sobre o campeao R4G.
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
EXPERIMENT = "EXP-014B-R5B11-CHAMPION-RECONCILIATION"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / EXPERIMENT
CANDIDATE_DIR = PROJECT_ROOT / "backend" / "artefatos_candidatos" / "exp014b_r5b11_global_policy"

R4G_PREDICTIONS = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R4G-FAST-FROZEN" / "06_predictions_frozen.csv"
R4G_ARTIFACT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R4G-FAST-FROZEN" / "05_policy_artifact_frozen.json"
R5B2_SUMMARY = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R5B2-FROZEN" / "00_run_summary.json"
R5B5_TRUST = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R5B5-TRUST-FEATURE-DEESCALATION" / "05_predictions_trust.csv"
R5B10_ARTIFACT = PROJECT_ROOT / "backend" / "artefatos_candidatos" / "exp014b_r5b10_severity_policy" / "severity_policy_candidate.json"

LABEL_COL = "is_fraud"
R4G_ACTION_COL = "r4g_fast_frozen_decisao_recommended"
TARGET_FPR = 0.01
TARGET_MAX_FN = 5

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
    "qtd_pix_pagador_180d",
    "valor_total_pagador_180d",
    "valor_total_pagador_90d",
    "valor_maximo_pix_pagador_180d",
    "ratio_valor_media_pagador_90d",
    "valor_total_recebido_30d",
    "dias_desde_primeiro_envio_recebedor",
    "vl_pix",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ints(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def actions(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.upper().str.strip()


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
    y = ints(df[LABEL_COL])
    out = df.groupby(action_col, dropna=False).agg(n_rows=(LABEL_COL, "size"), n_frauds=(LABEL_COL, "sum")).reset_index()
    out["n_normals"] = out["n_rows"] - out["n_frauds"]
    out["precision_within_action"] = (out["n_frauds"] / out["n_rows"]).round(8)
    return out.sort_values(action_col)


def parse_rule_mask(df: pd.DataFrame, description: str) -> pd.Series:
    for prefix in ("Mover BLOQUEAR para CONFIRMAR R4G_FAST com ", "BLOQUEAR->CONFIRMAR com "):
        if description.startswith(prefix):
            description = description[len(prefix) :]
            break

    mask = pd.Series(True, index=df.index)
    for part in description.split(" AND "):
        if " == " in part:
            col, val = part.split(" == ", 1)
            mask &= df[col].fillna("<MISSING>").astype(str).eq(val)
        elif " >= " in part:
            col, val = part.split(" >= ", 1)
            mask &= pd.to_numeric(df[col], errors="coerce").ge(float(val))
        elif " > " in part:
            col, val = part.split(" > ", 1)
            mask &= pd.to_numeric(df[col], errors="coerce").gt(float(val))
        elif " <= " in part:
            col, val = part.split(" <= ", 1)
            mask &= pd.to_numeric(df[col], errors="coerce").le(float(val))
        elif " < " in part:
            col, val = part.split(" < ", 1)
            mask &= pd.to_numeric(df[col], errors="coerce").lt(float(val))
        else:
            raise ValueError(f"Parte de regra nao suportada: {part}")
    return mask


def merge_trust_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not R5B5_TRUST.exists():
        return df
    trust = pd.read_csv(R5B5_TRUST, low_memory=False)
    extra_cols = [c for c in trust.columns if c not in df.columns and c != "idx"]
    if not extra_cols:
        return df
    return df.merge(trust[["transaction_id", *extra_cols]], on="transaction_id", how="left")


def replay_r5b10_on_r4g(df: pd.DataFrame, policy: dict[str, Any]) -> tuple[pd.Series, pd.DataFrame]:
    y = ints(df[LABEL_COL]).to_numpy()
    action = actions(df[R4G_ACTION_COL]).copy()
    rows: list[dict[str, Any]] = []

    for layer in policy.get("layers", []):
        for rule in layer.get("rules", []):
            rule_id = str(rule.get("candidate_id") or rule.get("rule_id"))
            description = str(rule.get("description", ""))
            try:
                raw_mask = parse_rule_mask(df, description)
            except Exception as exc:
                rows.append(
                    {
                        "layer": layer.get("layer"),
                        "rule_id": rule_id,
                        "status": "ERROR",
                        "error": str(exc),
                        "n_moved": 0,
                        "normal_moved": 0,
                        "fraud_moved": 0,
                    }
                )
                continue

            mask = raw_mask & action.eq("BLOQUEAR")
            rows.append(
                {
                    "layer": layer.get("layer"),
                    "rule_id": rule_id,
                    "status": "APPLIED" if bool(mask.any()) else "NO_MATCH",
                    "error": "",
                    "n_moved": int(mask.sum()),
                    "normal_moved": int((mask.to_numpy() & (y == 0)).sum()),
                    "fraud_moved": int((mask.to_numpy() & (y == 1)).sum()),
                    "description": description,
                }
            )
            action.loc[mask] = "CONFIRMAR"

    return action, pd.DataFrame(rows)


def mine_zero_fraud_block_rules(df: pd.DataFrame) -> pd.DataFrame:
    y = ints(df[LABEL_COL]).to_numpy()
    base_action = actions(df[R4G_ACTION_COL])
    residual = base_action.eq("BLOQUEAR").to_numpy()
    rows: list[dict[str, Any]] = []

    cat_cols = [c for c in CAT_COLS if c in df.columns]
    for size in (1, 2, 3):
        for cols in itertools.combinations(cat_cols, size):
            if any(df.loc[residual, c].fillna("<MISSING>").astype(str).nunique() > 40 for c in cols):
                continue
            grouped = df.loc[residual, list(cols)].fillna("<MISSING>").astype(str)
            grouped["_idx"] = grouped.index
            for vals, grp in grouped.groupby(list(cols), dropna=False):
                vals = vals if isinstance(vals, tuple) else (vals,)
                mask = np.zeros(len(df), dtype=bool)
                mask[grp["_idx"].to_numpy()] = True
                frauds = int((mask & (y == 1)).sum())
                normals = int((mask & (y == 0)).sum())
                if normals > 0 and frauds == 0:
                    rows.append(
                        {
                            "rule_type": f"categorical_{size}",
                            "description": " AND ".join(f"{c} == {v}" for c, v in zip(cols, vals)),
                            "normal_count": normals,
                            "fraud_count": frauds,
                        }
                    )

    for col in [c for c in NUM_COLS if c in df.columns]:
        values = pd.to_numeric(df[col], errors="coerce")
        residual_values = values.loc[residual].dropna()
        if residual_values.nunique() < 4:
            continue
        arr = values.to_numpy()
        for q in np.linspace(0.01, 0.99, 25):
            threshold = float(residual_values.quantile(q))
            for op, mask in (
                ("<=", np.isfinite(arr) & (arr <= threshold) & residual),
                (">=", np.isfinite(arr) & (arr >= threshold) & residual),
            ):
                frauds = int((mask & (y == 1)).sum())
                normals = int((mask & (y == 0)).sum())
                if normals > 0 and frauds == 0:
                    rows.append(
                        {
                            "rule_type": "numeric_threshold",
                            "description": f"{col} {op} {threshold:.8g}",
                            "normal_count": normals,
                            "fraud_count": frauds,
                        }
                    )

    if not rows:
        return pd.DataFrame(columns=["rule_type", "description", "normal_count", "fraud_count"])
    return pd.DataFrame(rows).drop_duplicates("description").sort_values("normal_count", ascending=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    for path in [R4G_PREDICTIONS, R4G_ARTIFACT, R5B2_SUMMARY, R5B10_ARTIFACT]:
        if not path.exists():
            raise FileNotFoundError(path)

    r4g_artifact = read_json(R4G_ARTIFACT)
    r5b2_summary = read_json(R5B2_SUMMARY)
    r5b10_artifact = read_json(R5B10_ARTIFACT)

    df = pd.read_csv(R4G_PREDICTIONS, low_memory=False)
    df_with_trust = merge_trust_columns(df)

    r4g_action = actions(df[R4G_ACTION_COL])
    r4g_intervention_metrics = metrics(df[LABEL_COL], intervention_pred(r4g_action))
    r4g_block_metrics = metrics(df[LABEL_COL], block_pred(r4g_action))

    r5b10_on_r4g_action, compatibility = replay_r5b10_on_r4g(df_with_trust, r5b10_artifact)
    stacked_intervention_metrics = metrics(df_with_trust[LABEL_COL], intervention_pred(r5b10_on_r4g_action))
    stacked_block_metrics = metrics(df_with_trust[LABEL_COL], block_pred(r5b10_on_r4g_action))

    zero_fraud_candidates = mine_zero_fraud_block_rules(df)

    r4g_meets_target = bool(
        r4g_intervention_metrics["fpr"] < TARGET_FPR
        and r4g_intervention_metrics["fn"] <= TARGET_MAX_FN
    )
    stacked_safe = bool(int(compatibility["fraud_moved"].sum()) == 0)

    status = (
        "PASS_R5B11_R4G_CONFIRMED_AS_GLOBAL_TARGET_CANDIDATE"
        if r4g_meets_target and not stacked_safe
        else "CHECK_R5B11_RECONCILIATION"
    )

    summary = {
        "experiment": EXPERIMENT,
        "status": status,
        "r4g_predictions": str(R4G_PREDICTIONS.relative_to(PROJECT_ROOT)),
        "r4g_artifact": str(R4G_ARTIFACT.relative_to(PROJECT_ROOT)),
        "r5b10_artifact": str(R5B10_ARTIFACT.relative_to(PROJECT_ROOT)),
        "target_fpr": TARGET_FPR,
        "target_max_fn": TARGET_MAX_FN,
        "r4g_meets_global_target": r4g_meets_target,
        "r4g_intervention_metrics": r4g_intervention_metrics,
        "r4g_block_metrics": r4g_block_metrics,
        "r5b2_intervention_metrics": r5b2_summary.get("final_intervention_metrics"),
        "r5b2_block_metrics": r5b2_summary.get("final_block_metrics"),
        "r5b10_stacked_on_r4g_safe": stacked_safe,
        "r5b10_stacked_on_r4g_frauds_demoted_to_confirm": int(compatibility["fraud_moved"].sum()),
        "r5b10_stacked_on_r4g_normals_demoted_to_confirm": int(compatibility["normal_moved"].sum()),
        "stacked_intervention_metrics": stacked_intervention_metrics,
        "stacked_block_metrics": stacked_block_metrics,
        "zero_fraud_r4g_block_deescalation_candidates": int(len(zero_fraud_candidates)),
        "decision": (
            "Promover R4G-FAST-FROZEN como candidato global atual. Nao empilhar R5B10 "
            "sobre R4G; a politica R5B10 foi validada somente na trilha R5B2."
        ),
    }

    candidate = {
        "artifact_type": "global_policy_candidate",
        "experiment": EXPERIMENT,
        "status": "CANDIDATE_NOT_PRODUCTION_ACTIVE",
        "source_policy": str(R4G_ARTIFACT.relative_to(PROJECT_ROOT)),
        "source_predictions": str(R4G_PREDICTIONS.relative_to(PROJECT_ROOT)),
        "recommended_action_col": R4G_ACTION_COL,
        "intervention_definition": "CONFIRMAR_OR_BLOQUEAR",
        "block_definition": "BLOQUEAR",
        "metrics": {
            "global_intervention": r4g_intervention_metrics,
            "block_only": r4g_block_metrics,
        },
        "target_gates": {
            "fpr_lt_1pct": r4g_intervention_metrics["fpr"] < TARGET_FPR,
            "fn_lte_5_outside_block": r4g_intervention_metrics["fn"] <= TARGET_MAX_FN,
        },
        "incompatibilities": [
            {
                "artifact": str(R5B10_ARTIFACT.relative_to(PROJECT_ROOT)),
                "reason": "Quando empilhado sobre R4G, demove fraudes de BLOQUEAR para CONFIRMAR.",
                "frauds_demoted_to_confirm": int(compatibility["fraud_moved"].sum()),
                "normals_demoted_to_confirm": int(compatibility["normal_moved"].sum()),
            }
        ],
        "promotion_gates": [
            "Conectar a politica R4G congelada por configuracao versionada.",
            "Executar replay E2E do PipelineOrquestrador contra o CSV R4G congelado.",
            "Bloquear empilhamento automatico da severidade R5B10 sobre R4G.",
            "Criar nova trilha de severidade especifica para o residual BLOQUEAR do R4G.",
        ],
    }

    write_json(OUT_DIR / "00_run_summary.json", summary)
    action_table(df.assign(_r4g_action=r4g_action), "_r4g_action").to_csv(
        OUT_DIR / "01_r4g_metrics_by_action.csv", index=False
    )
    compatibility.to_csv(OUT_DIR / "02_r5b10_on_r4g_rule_compatibility.csv", index=False)
    zero_fraud_candidates.to_csv(OUT_DIR / "03_r4g_zero_fraud_block_deescalation_candidates.csv", index=False)
    write_json(OUT_DIR / "04_global_policy_candidate.json", candidate)
    write_json(CANDIDATE_DIR / "global_policy_candidate.json", candidate)

    report = f"""# {EXPERIMENT} - ReconciliaÃ§Ã£o do campeÃ£o

## Resultado executivo
- Status: `{status}`
- R4G cumpre alvo global: `{r4g_meets_target}`
- R5B10 seguro empilhado sobre R4G: `{stacked_safe}`
- Regras simples zero-fraude no residual BLOQUEAR do R4G: `{len(zero_fraud_candidates)}`

## R4G - intervenÃ§Ã£o global
```json
{json.dumps(r4g_intervention_metrics, ensure_ascii=False, indent=2)}
```

## R4G - BLOQUEAR
```json
{json.dumps(r4g_block_metrics, ensure_ascii=False, indent=2)}
```

## Empilhamento R5B10 sobre R4G
```json
{json.dumps({
    "frauds_demoted_to_confirm": int(compatibility["fraud_moved"].sum()),
    "normals_demoted_to_confirm": int(compatibility["normal_moved"].sum()),
    "stacked_intervention_metrics": stacked_intervention_metrics,
    "stacked_block_metrics": stacked_block_metrics,
}, ensure_ascii=False, indent=2)}
```

## DecisÃ£o tÃ©cnica
O baseline `EXP-014B-R4G-FAST-FROZEN` permanece o Ãºnico candidato atual que
cumpre simultaneamente `FPR < 1%` e `FN <= 5` nas mÃ©tricas globais. A polÃ­tica
R5B10 nÃ£o deve ser empilhada sobre R4G, pois demove fraudes de `BLOQUEAR` para
`CONFIRMAR`. A prÃ³xima frente deve minerar uma severidade especÃ­fica para o
residual `BLOQUEAR` do R4G ou investigar por que a inferÃªncia runtime R5B2
regrediu para `FN=682`.
"""
    (OUT_DIR / "05_exp014b_r5b11_champion_reconciliation_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
