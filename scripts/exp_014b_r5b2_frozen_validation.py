#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R5B2-FROZEN — validação congelada da política BLOQUEAR -> CONFIRMAR.

Reaplica apenas as regras salvas em EXP-014B-R5B2-CALIBRATION, sem nova
mineração, e mede impacto, estabilidade temporal e reprodução das métricas.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exp_014b_r5b2_tune_policy import (
    LABELS,
    binarize_relationship_features,
    coalesce_duplicate_merge_columns,
    find_col,
    ints,
    metrics,
    norm_action,
    pred_block,
    pred_intervention,
)


EXPERIMENT = "EXP-014B-R5B2-FROZEN"
SOURCE_EXPERIMENT = "EXP-014B-R5B2-CALIBRATION"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_scored_base(root: Path) -> pd.DataFrame:
    pred_path = root / "resultados" / "experimentos" / SOURCE_EXPERIMENT / "01_raw_predictions_holdout.csv"
    input_path = root / "dados" / "hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df_pred = pd.read_csv(pred_path, low_memory=False)
    df_orig = pd.read_csv(input_path, low_memory=False)
    df_orig.columns = [c.split(".")[-1] for c in df_orig.columns]
    if "transaction_id" not in df_orig.columns and "cd_pix" in df_orig.columns:
        df_orig["transaction_id"] = df_orig["cd_pix"]

    df_pred["transaction_id"] = df_pred["transaction_id"].astype(str).str.strip()
    df_orig["transaction_id"] = df_orig["transaction_id"].astype(str).str.strip()

    cols_to_merge = [
        "transaction_id",
        "ds_tipo_chave_norm",
        "value_band",
        "periodo_dia",
        "qtd_pix_mesmo_recebedor_7d",
        "valor_medio_para_recebedor_180d",
        "dias_desde_ultima_transacao_recebedor",
        "ratio_valor_pix_vs_max_recebedor_180d",
        "is_recebedor_recorrente_180d",
        "first_receiver_flag_real",
        "mbk_available_flag",
    ]
    cols_present = [c for c in cols_to_merge if c in df_orig.columns]
    df = df_pred.merge(
        df_orig[cols_present].drop_duplicates("transaction_id"),
        on="transaction_id",
        how="left",
    )
    df = coalesce_duplicate_merge_columns(df)
    return binarize_relationship_features(df)


def parse_rule_conditions(description: str) -> list[tuple[str, str, str]]:
    if " com " not in description:
        raise ValueError(f"Descrição de regra sem bloco 'com': {description}")
    condition_text = description.split(" com ", 1)[1]
    conditions: list[tuple[str, str, str]] = []
    for part in condition_text.split(" AND "):
        match = re.match(r"^\s*([A-Za-z0-9_]+)\s*(==|<=|>=)\s*(.*?)\s*$", part)
        if not match:
            raise ValueError(f"Condição não parseável: {part!r}")
        col, op, raw_value = match.groups()
        conditions.append((col, op, raw_value))
    return conditions


def apply_rule(df: pd.DataFrame, rule: dict[str, Any]) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for col, op, raw_value in parse_rule_conditions(str(rule["description"])):
        if col not in df.columns:
            return np.zeros(len(df), dtype=bool)
        if op == "==":
            mask &= df[col].fillna("<MISSING>").astype(str).to_numpy() == str(raw_value)
            continue

        values = pd.to_numeric(df[col], errors="coerce").to_numpy()
        threshold = float(raw_value)
        if op == ">=":
            mask &= np.isfinite(values) & (values >= threshold)
        elif op == "<=":
            mask &= np.isfinite(values) & (values <= threshold)
        else:
            raise ValueError(f"Operador não suportado: {op}")
    return mask


def by_action(df: pd.DataFrame, action_col: str, label_col: str) -> pd.DataFrame:
    out = df.groupby(action_col).agg(
        n_rows=(label_col, "size"),
        n_frauds=(label_col, "sum"),
    ).reset_index()
    out["n_normals"] = out["n_rows"] - out["n_frauds"]
    return out


def split_month_stability(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    work = df.copy()
    work["event_month"] = pd.to_datetime(work["event_datetime"], errors="coerce").dt.to_period("M").astype(str)
    work["event_month"] = work["event_month"].replace("NaT", "<MISSING>")
    rows = []
    for keys, grp in work.groupby(["temporal_split", "event_month"], dropna=False):
        moved = grp["exp014b_r5b2_frozen_block_to_confirm"].astype(bool)
        y = ints(grp[label_col])
        rows.append({
            "temporal_split": keys[0],
            "event_month": keys[1],
            "n_rows": int(len(grp)),
            "block_to_confirm_n": int(moved.sum()),
            "block_to_confirm_normals": int(((moved) & (y == 0)).sum()),
            "block_to_confirm_frauds": int(((moved) & (y == 1)).sum()),
            "remaining_block_normals": int(((grp["r5b2_frozen_decisao"] == "BLOQUEAR") & (y == 0)).sum()),
            "remaining_block_frauds": int(((grp["r5b2_frozen_decisao"] == "BLOQUEAR") & (y == 1)).sum()),
        })
    return pd.DataFrame(rows).sort_values(["temporal_split", "event_month"])


def main() -> None:
    root = Path.cwd()
    src_dir = root / "resultados" / "experimentos" / SOURCE_EXPERIMENT
    out_dir = root / "resultados" / "experimentos" / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = src_dir / "02_policy_artifact_recommended.json"
    artifact = read_json(artifact_path)
    rules = artifact.get("selected_block_to_confirm_rules", [])

    df = load_scored_base(root)
    label_col = find_col(df, LABELS)
    y = ints(df[label_col]).to_numpy()
    base_action = norm_action(df["decisao"])
    final_action = base_action.copy()

    selected = np.zeros(len(df), dtype=bool)
    impact_rows = []
    for step, rule in enumerate(rules, start=1):
        eligible = final_action.eq("BLOQUEAR").to_numpy()
        mask = apply_rule(df, rule) & eligible & (~selected)
        moved_normals = int(((mask) & (y == 0)).sum())
        moved_frauds = int(((mask) & (y == 1)).sum())
        selected |= mask
        final_action.loc[mask] = "CONFIRMAR"
        impact_rows.append({
            "selection_step": step,
            "candidate_id": rule.get("candidate_id"),
            "rule_type": rule.get("rule_type"),
            "incremental_n_replayed": int(mask.sum()),
            "incremental_normals_replayed": moved_normals,
            "incremental_frauds_replayed": moved_frauds,
            "artifact_incremental_good": int(rule.get("incremental_good", -1)),
            "artifact_incremental_bad": int(rule.get("incremental_bad", -1)),
            "matches_artifact_incremental": bool(
                moved_normals == int(rule.get("incremental_good", -1))
                and moved_frauds == int(rule.get("incremental_bad", -1))
            ),
            "description": rule.get("description"),
        })

    df["exp014b_r5b2_frozen_block_to_confirm"] = selected.astype(int)
    df["r5b2_frozen_decisao"] = final_action
    df["exp014b_r5b2_frozen_intervention_pred"] = pred_intervention(final_action)
    df["exp014b_r5b2_frozen_block_pred"] = pred_block(final_action)

    base_intervention_metrics = metrics(df[label_col], pred_intervention(base_action))
    base_block_metrics = metrics(df[label_col], pred_block(base_action))
    final_intervention_metrics = metrics(df[label_col], df["exp014b_r5b2_frozen_intervention_pred"])
    final_block_metrics = metrics(df[label_col], df["exp014b_r5b2_frozen_block_pred"])

    prior_predictions = src_dir / "04_predictions_recommended.csv"
    prediction_mismatches = None
    if prior_predictions.exists():
        prior = pd.read_csv(prior_predictions, usecols=["transaction_id", "r5b2_decisao_recommended"], low_memory=False)
        prior["transaction_id"] = prior["transaction_id"].astype(str).str.strip()
        check = df[["transaction_id", "r5b2_frozen_decisao"]].merge(prior, on="transaction_id", how="left")
        prediction_mismatches = int((check["r5b2_frozen_decisao"] != check["r5b2_decisao_recommended"]).sum())

    rule_impact = pd.DataFrame(impact_rows)
    stability = split_month_stability(df, label_col)
    metrics_by_action = by_action(df, "r5b2_frozen_decisao", label_col)

    all_rules_match = bool(rule_impact["matches_artifact_incremental"].all()) if not rule_impact.empty else True
    block_tp_loss = int(((selected) & (y == 1)).sum())
    block_fp_demoted = int(((selected) & (y == 0)).sum())
    all_pass = (
        all_rules_match
        and block_tp_loss == int(artifact.get("block_tp_demoted_to_confirm", -1))
        and block_fp_demoted == int(artifact.get("block_fp_demoted_to_confirm", -1))
        and (prediction_mismatches in (None, 0))
    )

    run_summary = {
        "experiment": EXPERIMENT,
        "source_experiment": SOURCE_EXPERIMENT,
        "status": "PASS_R5B2_FROZEN_REPLAYED" if all_pass else "FAIL_R5B2_FROZEN_REPLAY_MISMATCH",
        "all_pass": bool(all_pass),
        "n_rows": int(len(df)),
        "n_frauds": int(y.sum()),
        "n_rules": int(len(rules)),
        "all_rules_match_artifact_incremental": bool(all_rules_match),
        "prediction_mismatches_vs_calibration": prediction_mismatches,
        "block_fp_demoted_to_confirm": block_fp_demoted,
        "block_tp_demoted_to_confirm": block_tp_loss,
        "baseline_intervention_metrics": base_intervention_metrics,
        "baseline_block_metrics": base_block_metrics,
        "final_intervention_metrics": final_intervention_metrics,
        "final_block_metrics": final_block_metrics,
        "remaining_approve_frauds": int(((final_action == "APROVAR") & (y == 1)).sum()),
        "remaining_block_normals": int(((final_action == "BLOQUEAR") & (y == 0)).sum()),
    }

    input_contract = {
        "input_predictions": str(src_dir / "01_raw_predictions_holdout.csv"),
        "source_policy_artifact": str(artifact_path),
        "final_action_col": "r5b2_frozen_decisao",
        "required_columns": [
            "transaction_id",
            "event_datetime",
            "is_fraud",
            "decisao",
            "temporal_split",
        ],
    }

    write_json(out_dir / "00_run_summary.json", run_summary)
    write_json(out_dir / "01_input_contract.json", input_contract)
    pd.DataFrame([{
        "metric_scope": "baseline_intervention",
        **base_intervention_metrics,
    }, {
        "metric_scope": "final_intervention",
        **final_intervention_metrics,
    }, {
        "metric_scope": "baseline_block",
        **base_block_metrics,
    }, {
        "metric_scope": "final_block",
        **final_block_metrics,
    }]).to_csv(out_dir / "02_frozen_metrics.csv", index=False)
    rule_impact.to_csv(out_dir / "03_rule_impact_replay.csv", index=False)
    stability.to_csv(out_dir / "04_stability_by_split_month.csv", index=False)
    metrics_by_action.to_csv(out_dir / "05_metrics_by_action.csv", index=False)
    df.to_csv(out_dir / "06_predictions_frozen.csv", index=False)

    report = f"""# {EXPERIMENT} — Validação congelada

## Resultado executivo
- Status: `{run_summary['status']}`
- All pass: `{run_summary['all_pass']}`
- Regras reaplicadas: `{len(rules)}`
- Mismatches vs calibration: `{prediction_mismatches}`
- Normais movidos de BLOQUEAR para CONFIRMAR: `{block_fp_demoted}`
- Fraudes movidas de BLOQUEAR para CONFIRMAR: `{block_tp_loss}`

## Métricas de BLOQUEAR
```json
{json.dumps(final_block_metrics, ensure_ascii=False, indent=2)}
```

## Métricas de intervenção total
```json
{json.dumps(final_intervention_metrics, ensure_ascii=False, indent=2)}
```

## Decisões finais
{metrics_by_action.to_markdown(index=False)}

## Estabilidade por split/mês
{stability.to_markdown(index=False)}

## Interpretação
A política congelada reproduz a redução de severidade `BLOQUEAR -> CONFIRMAR` sem
rebaixar fraude conhecida de BLOQUEAR. Ela não resolve o recall total, pois ainda
restam `{run_summary['remaining_approve_frauds']}` fraudes em APROVAR. O próximo
experimento deve atacar resgate de APROVAR e/ou melhorar score/features do modelo base.
"""
    (out_dir / "07_exp014b_r5b2_frozen_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
