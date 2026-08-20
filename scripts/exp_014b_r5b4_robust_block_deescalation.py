#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R5B4 — seleção robusta das regras BLOQUEAR -> CONFIRMAR.

Parte do artifact R5B2-FROZEN, mede suporte por split/mês e seleciona apenas
regras com evidência fora de TRAIN. O objetivo é manter a redução de bloqueio
indevido com menor risco de overfit antes de qualquer integração produtiva.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exp_014b_r5b2_frozen_validation import apply_rule
from exp_014b_r5b2_tune_policy import LABELS, find_col, ints, metrics, norm_action, pred_block, pred_intervention


EXPERIMENT = "EXP-014B-R5B4-ROBUST-BLOCK-DEESCALATION"
SOURCE_FROZEN = "EXP-014B-R5B2-FROZEN"
SOURCE_CALIBRATION = "EXP-014B-R5B2-CALIBRATION"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def by_action(df: pd.DataFrame, action_col: str, label_col: str) -> pd.DataFrame:
    out = df.groupby(action_col).agg(
        n_rows=(label_col, "size"),
        n_frauds=(label_col, "sum"),
    ).reset_index()
    out["n_normals"] = out["n_rows"] - out["n_frauds"]
    return out


def compute_rule_support(df: pd.DataFrame, rules: list[dict[str, Any]], y: np.ndarray) -> pd.DataFrame:
    base_block = norm_action(df["decisao"]).eq("BLOQUEAR").to_numpy()
    event_month = pd.to_datetime(df["event_datetime"], errors="coerce").dt.to_period("M").astype(str)
    temporal_split = df["temporal_split"].fillna("<MISSING>").astype(str)
    rows = []

    for rule in rules:
        full_mask = apply_rule(df, rule) & base_block
        normal_mask = full_mask & (y == 0)
        fraud_mask = full_mask & (y == 1)
        split_normals = {}
        split_frauds = {}
        for split in sorted(temporal_split.unique()):
            split_mask = temporal_split.eq(split).to_numpy()
            split_normals[split] = int((normal_mask & split_mask).sum())
            split_frauds[split] = int((fraud_mask & split_mask).sum())

        month_normal_support = int(event_month[normal_mask].nunique())
        month_fraud_support = int(event_month[fraud_mask].nunique())
        non_train_normals = sum(v for k, v in split_normals.items() if k != "TRAIN")
        non_train_frauds = sum(v for k, v in split_frauds.items() if k != "TRAIN")
        rows.append({
            "candidate_id": rule.get("candidate_id"),
            "rule_type": rule.get("rule_type"),
            "description": rule.get("description"),
            "total_normals": int(normal_mask.sum()),
            "total_frauds": int(fraud_mask.sum()),
            "train_normals": int(split_normals.get("TRAIN", 0)),
            "validation_normals": int(split_normals.get("VALIDATION", 0)),
            "holdout_normals": int(split_normals.get("HOLDOUT", 0)),
            "non_train_normals": int(non_train_normals),
            "train_frauds": int(split_frauds.get("TRAIN", 0)),
            "validation_frauds": int(split_frauds.get("VALIDATION", 0)),
            "holdout_frauds": int(split_frauds.get("HOLDOUT", 0)),
            "non_train_frauds": int(non_train_frauds),
            "month_normal_support": month_normal_support,
            "month_fraud_support": month_fraud_support,
        })

    return pd.DataFrame(rows)


def select_robust_rules(rule_support: pd.DataFrame) -> pd.DataFrame:
    robust = rule_support[
        (rule_support["total_frauds"] == 0)
        & (rule_support["non_train_frauds"] == 0)
        & (rule_support["non_train_normals"] >= 20)
        & (rule_support["month_normal_support"] >= 2)
    ].copy()
    if robust.empty:
        return robust

    robust["has_holdout_support"] = robust["holdout_normals"] > 0
    robust["has_validation_support"] = robust["validation_normals"] > 0
    robust = robust.sort_values(
        [
            "has_holdout_support",
            "has_validation_support",
            "non_train_normals",
            "month_normal_support",
            "total_normals",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    robust["selection_step"] = np.arange(1, len(robust) + 1)
    return robust


def replay_selected(df: pd.DataFrame, rules_by_id: dict[str, dict[str, Any]], selected: pd.DataFrame, y: np.ndarray) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
    final_action = norm_action(df["decisao"])
    selected_mask = np.zeros(len(df), dtype=bool)
    rows = []
    for _, row in selected.iterrows():
        rule = rules_by_id[str(row["candidate_id"])]
        eligible = final_action.eq("BLOQUEAR").to_numpy()
        mask = apply_rule(df, rule) & eligible & (~selected_mask)
        normal_count = int(((mask) & (y == 0)).sum())
        fraud_count = int(((mask) & (y == 1)).sum())
        if normal_count <= 0 and fraud_count <= 0:
            continue
        selected_mask |= mask
        final_action.loc[mask] = "CONFIRMAR"
        out = row.to_dict()
        out.update({
            "incremental_n": int(mask.sum()),
            "incremental_normals": normal_count,
            "incremental_frauds": fraud_count,
            "cumulative_normals": int(((selected_mask) & (y == 0)).sum()),
            "cumulative_frauds": int(((selected_mask) & (y == 1)).sum()),
        })
        rows.append(out)
    return final_action, pd.DataFrame(rows), selected_mask


def split_month_stability(df: pd.DataFrame, label_col: str, action_col: str, move_col: str) -> pd.DataFrame:
    work = df.copy()
    work["event_month"] = pd.to_datetime(work["event_datetime"], errors="coerce").dt.to_period("M").astype(str)
    work["event_month"] = work["event_month"].replace("NaT", "<MISSING>")
    y = ints(work[label_col])
    moved = work[move_col].astype(bool)
    rows = []
    for (split, month), grp in work.groupby(["temporal_split", "event_month"], dropna=False):
        idx = grp.index
        rows.append({
            "temporal_split": split,
            "event_month": month,
            "n_rows": int(len(grp)),
            "block_to_confirm_normals": int((moved.loc[idx] & (y.loc[idx] == 0)).sum()),
            "block_to_confirm_frauds": int((moved.loc[idx] & (y.loc[idx] == 1)).sum()),
            "remaining_block_normals": int(((grp[action_col] == "BLOQUEAR") & (y.loc[idx] == 0)).sum()),
            "remaining_block_frauds": int(((grp[action_col] == "BLOQUEAR") & (y.loc[idx] == 1)).sum()),
        })
    return pd.DataFrame(rows).sort_values(["temporal_split", "event_month"])


def main() -> None:
    root = Path.cwd()
    frozen_dir = root / "resultados" / "experimentos" / SOURCE_FROZEN
    calibration_dir = root / "resultados" / "experimentos" / SOURCE_CALIBRATION
    out_dir = root / "resultados" / "experimentos" / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = frozen_dir / "06_predictions_frozen.csv"
    artifact_path = calibration_dir / "02_policy_artifact_recommended.json"
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    if not artifact_path.exists():
        raise FileNotFoundError(artifact_path)

    df = pd.read_csv(predictions_path, low_memory=False)
    artifact = read_json(artifact_path)
    rules = artifact.get("selected_block_to_confirm_rules", [])
    rules_by_id = {str(r["candidate_id"]): r for r in rules}
    label_col = find_col(df, LABELS)
    y = ints(df[label_col]).to_numpy()

    rule_support = compute_rule_support(df, rules, y)
    selected = select_robust_rules(rule_support)
    final_action, selected_replay, selected_mask = replay_selected(df, rules_by_id, selected, y)

    df["exp014b_r5b4_robust_block_to_confirm"] = selected_mask.astype(int)
    df["r5b4_robust_decisao"] = final_action
    df["exp014b_r5b4_intervention_pred"] = pred_intervention(final_action)
    df["exp014b_r5b4_block_pred"] = pred_block(final_action)

    base_action = norm_action(df["decisao"])
    base_intervention_metrics = metrics(df[label_col], pred_intervention(base_action))
    base_block_metrics = metrics(df[label_col], pred_block(base_action))
    final_intervention_metrics = metrics(df[label_col], df["exp014b_r5b4_intervention_pred"])
    final_block_metrics = metrics(df[label_col], df["exp014b_r5b4_block_pred"])
    metrics_by_action = by_action(df, "r5b4_robust_decisao", label_col)
    stability = split_month_stability(
        df,
        label_col=label_col,
        action_col="r5b4_robust_decisao",
        move_col="exp014b_r5b4_robust_block_to_confirm",
    )

    demoted_normals = int(((selected_mask) & (y == 0)).sum())
    demoted_frauds = int(((selected_mask) & (y == 1)).sum())
    all_pass = bool(demoted_frauds == 0 and len(selected_replay) > 0)
    summary = {
        "experiment": EXPERIMENT,
        "source_frozen": SOURCE_FROZEN,
        "source_calibration": SOURCE_CALIBRATION,
        "status": "PASS_R5B4_ROBUST_POLICY_SELECTED" if all_pass else "FAIL_R5B4_NO_ROBUST_POLICY",
        "all_pass": all_pass,
        "n_input_rules": int(len(rules)),
        "n_selected_rules": int(len(selected_replay)),
        "block_fp_demoted_to_confirm": demoted_normals,
        "block_tp_demoted_to_confirm": demoted_frauds,
        "remaining_block_normals": int(((final_action == "BLOQUEAR") & (y == 0)).sum()),
        "remaining_block_frauds": int(((final_action == "BLOQUEAR") & (y == 1)).sum()),
        "remaining_approve_frauds": int(((final_action == "APROVAR") & (y == 1)).sum()),
        "baseline_intervention_metrics": base_intervention_metrics,
        "baseline_block_metrics": base_block_metrics,
        "final_intervention_metrics": final_intervention_metrics,
        "final_block_metrics": final_block_metrics,
        "selection_criteria": {
            "total_frauds": 0,
            "non_train_frauds": 0,
            "non_train_normals_min": 20,
            "month_normal_support_min": 2,
        },
    }

    robust_artifact = {
        "experiment": EXPERIMENT,
        "source_policy_artifact": str(artifact_path),
        "base_action_col": "decisao",
        "final_action_col": "r5b4_robust_decisao",
        "move_col": "exp014b_r5b4_robust_block_to_confirm",
        "selected_block_to_confirm_rules": [
            rules_by_id[str(cid)] for cid in selected_replay["candidate_id"].astype(str).tolist()
        ] if not selected_replay.empty else [],
        "run_summary": summary,
    }

    write_json(out_dir / "00_run_summary.json", summary)
    rule_support.to_csv(out_dir / "01_rule_support_by_split.csv", index=False)
    selected_replay.to_csv(out_dir / "02_selected_robust_rules.csv", index=False)
    write_json(out_dir / "03_policy_artifact_robust.json", robust_artifact)
    pd.DataFrame([
        {"metric_scope": "baseline_intervention", **base_intervention_metrics},
        {"metric_scope": "final_intervention", **final_intervention_metrics},
        {"metric_scope": "baseline_block", **base_block_metrics},
        {"metric_scope": "final_block", **final_block_metrics},
    ]).to_csv(out_dir / "04_robust_metrics.csv", index=False)
    metrics_by_action.to_csv(out_dir / "05_metrics_by_action.csv", index=False)
    stability.to_csv(out_dir / "06_stability_by_split_month.csv", index=False)
    df.to_csv(out_dir / "07_predictions_robust.csv", index=False)

    report = f"""# {EXPERIMENT} — Política robusta BLOQUEAR -> CONFIRMAR

## Resultado executivo
- Status: `{summary['status']}`
- Regras candidatas R5B2: `{summary['n_input_rules']}`
- Regras robustas selecionadas: `{summary['n_selected_rules']}`
- Normais movidos de BLOQUEAR para CONFIRMAR: `{summary['block_fp_demoted_to_confirm']}`
- Fraudes movidas de BLOQUEAR para CONFIRMAR: `{summary['block_tp_demoted_to_confirm']}`
- Normais restantes em BLOQUEAR: `{summary['remaining_block_normals']}`
- Fraudes restantes em APROVAR: `{summary['remaining_approve_frauds']}`

## Critério de seleção
```json
{json.dumps(summary['selection_criteria'], ensure_ascii=False, indent=2)}
```

## Métricas finais de BLOQUEAR
```json
{json.dumps(final_block_metrics, ensure_ascii=False, indent=2)}
```

## Decisões finais
{metrics_by_action.to_markdown(index=False)}

## Estabilidade por split/mês
{stability.to_markdown(index=False)}

## Regras selecionadas
{selected_replay[['selection_step', 'candidate_id', 'incremental_normals', 'incremental_frauds', 'non_train_normals', 'holdout_normals', 'validation_normals', 'month_normal_support']].to_markdown(index=False) if not selected_replay.empty else 'Nenhuma regra selecionada.'}
"""
    (out_dir / "08_exp014b_r5b4_robust_block_deescalation_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
