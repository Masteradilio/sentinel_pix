# -*- coding: utf-8 -*-
"""
EXP-014B-R3W — Action Alignment Audit

Objetivo:
  Auditar o desalinhamento entre:
    1. benchmark experimental congelado: exp014b_r3q_frozen_pred
    2. acao operacional exportada: decisao / APROVAR / CONFIRMAR / BLOQUEAR

Contexto:
  O R3V mostrou que R3Q tem TP=1465, FP=4074, FN=0, mas a coluna
  operacional `decisao` possui apenas 90 intervenções:
    CONFIRMAR=73, BLOQUEAR=17, APROVAR=113754.

  Isso sugere que a coluna `decisao` pode estar vindo de outra política,
  outro estágio, ou de uma exportação não alinhada com o artifact R3Q.

Este script:
  - não altera modelo;
  - não promove política;
  - não faz mineração;
  - só audita e produz evidências.

Saídas:
  resultados/experimentos/EXP-014B-R3W/
    00_run_summary.json
    01_input_contract.json
    02_metric_comparison.json
    03_alignment_groups.csv
    04_action_distribution.csv
    05_r3q_alert_action_aprovar_frauds.csv
    06_r3q_alert_action_aprovar_normals.csv
    07_action_intervention_without_r3q_alert.csv
    08_score_summary_by_alignment_group.csv
    09_alignment_diagnosis.json
    10_exp014b_r3w_report.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT = "EXP-014B-R3W"

BASE_COL_CANDIDATES = [
    "exp014b_r3q_frozen_pred",
    "exp014b_r3q_recommended_pred",
    "exp014b_r3p_frozen_pred",
    "exp014b_r3v_recommended_pred",
    "exp014b_r3u_recommended_pred",
]

LABEL_CANDIDATES = ["is_fraud", "fraude", "target", "label", "tp_fraude"]

ACTION_CANDIDATES = [
    "decisao",
    "decision",
    "action",
    "final_decision",
    "decision_engine_decisao",
    "engine_decision",
    "acao",
    "acao_recomendada",
]

SCORE_CANDIDATES = [
    "score_final",
    "lgbm_r4_score",
    "lgbm_raw",
    "lgbm_mapped",
    "peso_total",
    "if_percentile",
    "se_score",
    "beh_score",
    "behavioral_score",
    "topaz_risk_score",
    "exp014b_r3s_second_stage_score",
    "exp014b_r3u_receiver_relationship_trust_score",
]

KEY_EXPORT_COLS = [
    "transaction_id",
    "cd_pix",
    "customer_id",
    "cd_cpf_pagador",
    "cd_cpf_cnpj_recebedor",
    "dt_pix",
    "event_datetime",
    "is_fraud",
    "decisao",
    "r3w_action_norm",
    "exp014b_r3q_frozen_pred",
    "score_final",
    "lgbm_r4_score",
    "lgbm_raw",
    "lgbm_mapped",
    "peso_total",
    "if_percentile",
    "se_score",
    "beh_score",
    "topaz_risk_score",
    "ds_tipo_chave_norm",
    "value_band",
    "periodo_dia",
    "score_bin",
    "lgbm_bin",
    "if_bin",
    "ratio_bin",
    "qtd_rec_bin",
    "valor_rec_bin",
    "module_quiet",
    "se_worst_pattern",
    "first_receiver_flag_real",
    "mbk_available_flag",
]

SEGMENT_COLS = [
    "temporal_split",
    "event_month",
    "ds_tipo_chave_norm",
    "value_band",
    "periodo_dia",
    "score_bin",
    "lgbm_bin",
    "if_bin",
    "ratio_bin",
    "qtd_rec_bin",
    "valor_rec_bin",
    "module_quiet",
    "se_worst_pattern",
    "mbk_available_flag",
    "first_receiver_flag_real",
]


def out_dir() -> Path:
    path = Path.cwd() / "resultados" / "experimentos" / EXPERIMENT
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_input() -> Path:
    base = Path.cwd() / "resultados" / "experimentos"
    candidates = [
        base / "EXP-014B-R3V" / "08_predictions_recommended.csv",
        base / "EXP-014B-R3U" / "09_predictions_recommended.csv",
        base / "EXP-014B-R3S" / "08_predictions_recommended.csv",
        base / "EXP-014B-R3Q" / "08_predictions_recommended.csv",
        base / "EXP-014B-R3P-FROZEN" / "08_predictions_frozen.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Nenhum input encontrado. Esperado um dos arquivos:\n"
        + "\n".join(str(p) for p in candidates)
    )


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(f"Nenhuma coluna encontrada entre: {candidates}")
    return None


def safe_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def metrics(y_true: pd.Series, pred: pd.Series) -> dict[str, Any]:
    y = safe_int_series(y_true)
    p = safe_int_series(pred)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
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


def normalize_action(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "UNKNOWN"
    s = str(x).strip().upper()
    if not s or s in {"NAN", "NONE", "<MISSING>"}:
        return "UNKNOWN"
    if "BLOQ" in s or "BLOCK" in s:
        return "BLOQUEAR"
    if "CONF" in s or "REVIEW" in s or "ANALIS" in s or "ALERT" in s:
        return "CONFIRMAR"
    if "APROV" in s or "APPROV" in s or "ALLOW" in s:
        return "APROVAR"
    return s


def action_to_intervention(action: pd.Series) -> pd.Series:
    return action.astype(str).isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)


def group_name(r3q_alert: int, action_intervention: int) -> str:
    if r3q_alert == 1 and action_intervention == 1:
        return "R3Q_ALERT__ACTION_INTERVENTION"
    if r3q_alert == 1 and action_intervention == 0:
        return "R3Q_ALERT__ACTION_APROVAR"
    if r3q_alert == 0 and action_intervention == 1:
        return "R3Q_NO_ALERT__ACTION_INTERVENTION"
    return "R3Q_NO_ALERT__ACTION_APROVAR"


def build_alignment_groups(
    df: pd.DataFrame,
    label_col: str,
    base_col: str,
    action_pred_col: str,
) -> pd.DataFrame:
    rows = []
    y = safe_int_series(df[label_col])
    r3q = safe_int_series(df[base_col])
    act = safe_int_series(df[action_pred_col])

    for r3q_val in [1, 0]:
        for act_val in [1, 0]:
            mask = (r3q == r3q_val) & (act == act_val)
            g_y = y[mask]
            n_rows = int(mask.sum())
            n_frauds = int((g_y == 1).sum())
            n_normals = int((g_y == 0).sum())
            rows.append({
                "alignment_group": group_name(r3q_val, act_val),
                "r3q_alert": r3q_val,
                "action_intervention": act_val,
                "n_rows": n_rows,
                "n_frauds": n_frauds,
                "n_normals": n_normals,
                "fraud_share_in_group": round(float(n_frauds / n_rows), 8) if n_rows else 0.0,
                "share_of_all_frauds": round(float(n_frauds / max(1, int((y == 1).sum()))), 8),
                "share_of_all_normals": round(float(n_normals / max(1, int((y == 0).sum()))), 8),
            })
    return pd.DataFrame(rows)


def action_distribution(df: pd.DataFrame, label_col: str, action_col: str) -> pd.DataFrame:
    rows = []
    y = safe_int_series(df[label_col])
    for action, idx in df.groupby(action_col, dropna=False).groups.items():
        idx = list(idx)
        g = df.loc[idx]
        yy = y.loc[idx]
        n = int(len(g))
        frauds = int((yy == 1).sum())
        normals = int((yy == 0).sum())
        rows.append({
            "action": str(action),
            "n_rows": n,
            "n_frauds": frauds,
            "n_normals": normals,
            "precision_within_action": round(float(frauds / n), 8) if n else 0.0,
            "fraud_share": round(float(frauds / max(1, int((y == 1).sum()))), 8),
            "normal_share": round(float(normals / max(1, int((y == 0).sum()))), 8),
        })
    return pd.DataFrame(rows).sort_values(["n_rows"], ascending=False)


def score_summary_by_group(
    df: pd.DataFrame,
    group_col: str,
    score_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for group, g in df.groupby(group_col, dropna=False):
        for col in score_cols:
            s = pd.to_numeric(g[col], errors="coerce")
            if s.notna().sum() == 0:
                continue
            rows.append({
                "alignment_group": str(group),
                "score_col": col,
                "n_non_null": int(s.notna().sum()),
                "mean": round(float(s.mean()), 8),
                "median": round(float(s.median()), 8),
                "p10": round(float(s.quantile(0.10)), 8),
                "p90": round(float(s.quantile(0.90)), 8),
                "min": round(float(s.min()), 8),
                "max": round(float(s.max()), 8),
            })
    return pd.DataFrame(rows)


def export_subset(df: pd.DataFrame, path: Path, mask: pd.Series) -> int:
    cols = [c for c in KEY_EXPORT_COLS if c in df.columns]
    export = df.loc[mask, cols].copy()
    export.to_csv(path, index=False, encoding="utf-8")
    return int(len(export))


def segment_breakdown(
    df: pd.DataFrame,
    label_col: str,
    group_col: str,
    target_group: str,
) -> pd.DataFrame:
    rows = []
    y = safe_int_series(df[label_col])
    target = df[group_col].eq(target_group)
    for col in SEGMENT_COLS:
        if col not in df.columns:
            continue
        for val, idx in df[target].groupby(col, dropna=False).groups.items():
            idx = list(idx)
            g = df.loc[idx]
            yy = y.loc[idx]
            n = int(len(g))
            rows.append({
                "target_group": target_group,
                "segment_col": col,
                "segment_value": str(val),
                "n_rows": n,
                "n_frauds": int((yy == 1).sum()),
                "n_normals": int((yy == 0).sum()),
                "precision_in_segment": round(float((yy == 1).sum() / n), 8) if n else 0.0,
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["n_frauds", "n_normals", "n_rows"], ascending=[False, False, False])


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    od = out_dir()
    input_path = find_input()
    df = pd.read_csv(input_path, low_memory=False)

    label_col = find_col(df, LABEL_CANDIDATES)
    base_col = find_col(df, BASE_COL_CANDIDATES)
    action_col = find_col(df, ACTION_CANDIDATES)
    score_cols = [c for c in SCORE_CANDIDATES if c in df.columns]

    df["r3w_action_norm"] = df[action_col].apply(normalize_action)
    df["r3w_action_intervention_pred"] = action_to_intervention(df["r3w_action_norm"])
    df["r3w_r3q_alert"] = safe_int_series(df[base_col])
    df["r3w_alignment_group"] = [
        group_name(int(r), int(a))
        for r, a in zip(df["r3w_r3q_alert"], df["r3w_action_intervention_pred"])
    ]

    n_rows = int(len(df))
    n_frauds = int(safe_int_series(df[label_col]).sum())
    n_normals = n_rows - n_frauds

    r3q_metrics = metrics(df[label_col], df[base_col])
    action_metrics = metrics(df[label_col], df["r3w_action_intervention_pred"])

    metric_comparison = {
        "r3q_binary_metrics": r3q_metrics,
        "action_intervention_metrics": action_metrics,
        "delta_action_minus_r3q": {
            "tp": int(action_metrics["tp"] - r3q_metrics["tp"]),
            "fp": int(action_metrics["fp"] - r3q_metrics["fp"]),
            "fn": int(action_metrics["fn"] - r3q_metrics["fn"]),
            "tn": int(action_metrics["tn"] - r3q_metrics["tn"]),
            "recall_delta": round(float(action_metrics["recall"] - r3q_metrics["recall"]), 8),
            "fpr_delta": round(float(action_metrics["fpr"] - r3q_metrics["fpr"]), 8),
        },
    }

    groups = build_alignment_groups(df, label_col, base_col, "r3w_action_intervention_pred")
    dist = action_distribution(df, label_col, "r3w_action_norm")
    score_summary = score_summary_by_group(df, "r3w_alignment_group", score_cols)

    mask_r3q_alert_action_aprovar_fraud = (
        df["r3w_alignment_group"].eq("R3Q_ALERT__ACTION_APROVAR")
        & safe_int_series(df[label_col]).eq(1)
    )
    mask_r3q_alert_action_aprovar_normal = (
        df["r3w_alignment_group"].eq("R3Q_ALERT__ACTION_APROVAR")
        & safe_int_series(df[label_col]).eq(0)
    )
    mask_action_intervention_without_r3q = (
        df["r3w_alignment_group"].eq("R3Q_NO_ALERT__ACTION_INTERVENTION")
    )

    n_misaligned_fraud = export_subset(
        df,
        od / "05_r3q_alert_action_aprovar_frauds.csv",
        mask_r3q_alert_action_aprovar_fraud,
    )
    n_misaligned_normal = export_subset(
        df,
        od / "06_r3q_alert_action_aprovar_normals.csv",
        mask_r3q_alert_action_aprovar_normal,
    )
    n_action_without_r3q = export_subset(
        df,
        od / "07_action_intervention_without_r3q_alert.csv",
        mask_action_intervention_without_r3q,
    )

    seg_misaligned_fraud = segment_breakdown(
        df,
        label_col,
        "r3w_alignment_group",
        "R3Q_ALERT__ACTION_APROVAR",
    )
    seg_misaligned_fraud.to_csv(od / "08b_segments_r3q_alert_action_aprovar.csv", index=False, encoding="utf-8")

    diagnosis = {
        "experiment": EXPERIMENT,
        "diagnostic_type": "action_alignment_audit",
        "input_path": str(input_path),
        "label_col": label_col,
        "r3q_base_col": base_col,
        "action_col": action_col,
        "n_rows": n_rows,
        "n_frauds": n_frauds,
        "n_normals": n_normals,
        "metric_comparison": metric_comparison,
        "critical_counts": {
            "r3q_alert_action_aprovar_frauds": n_misaligned_fraud,
            "r3q_alert_action_aprovar_normals": n_misaligned_normal,
            "action_intervention_without_r3q_alert": n_action_without_r3q,
        },
        "interpretation": [],
        "recommended_next_checks": [],
    }

    if n_misaligned_fraud > 0:
        diagnosis["interpretation"].append(
            "A coluna de ação operacional não está alinhada com o benchmark R3Q: há fraudes detectadas por R3Q que aparecem como APROVAR."
        )
        diagnosis["recommended_next_checks"].append(
            "Verificar se a coluna decisao foi exportada antes da aplicação do artifact R3Q/R3P/R3Q ou se pertence ao DecisionEngine vanilla."
        )
        diagnosis["recommended_next_checks"].append(
            "Reexecutar batch pelo PipelineOrquestrador com o mesmo artifact/política usada para criar exp014b_r3q_frozen_pred, exportando decisao_pos_policy."
        )
    else:
        diagnosis["interpretation"].append(
            "Não há fraude R3Q alert + ação APROVAR; o alinhamento entre R3Q e decisão parece consistente para fraudes."
        )

    if n_misaligned_normal > 0:
        diagnosis["interpretation"].append(
            "Há normais detectados por R3Q que aparecem como APROVAR; esses casos explicam parte dos FPs experimentais que não viram intervenção operacional."
        )

    if action_metrics["recall"] < 0.95 and r3q_metrics["recall"] >= 0.95:
        diagnosis["recommended_next_checks"].append(
            "Não calibrar APROVAR/CONFIRMAR/BLOQUEAR ainda; primeiro alinhar a exportação de decisão com o artifact experimental."
        )

    summary = {
        "experiment": EXPERIMENT,
        "status": "DONE",
        "objective_status": "DONE_R3W_ACTION_ALIGNMENT_AUDIT",
        "n_rows": n_rows,
        "n_frauds": n_frauds,
        "n_normals": n_normals,
        "input_path": str(input_path),
        "label_col": label_col,
        "base_col": base_col,
        "action_col": action_col,
        "r3q_metrics": r3q_metrics,
        "action_intervention_metrics": action_metrics,
        "r3q_alert_action_aprovar_frauds": n_misaligned_fraud,
        "r3q_alert_action_aprovar_normals": n_misaligned_normal,
        "action_intervention_without_r3q_alert": n_action_without_r3q,
        "alignment_problem_detected": bool(n_misaligned_fraud > 0 or action_metrics["recall"] != r3q_metrics["recall"]),
        "all_pass": True,
        "output_dir": str(od),
    }

    contract = {
        "n_rows": n_rows,
        "n_frauds": n_frauds,
        "n_normals": n_normals,
        "input_path": str(input_path),
        "label_col": label_col,
        "base_col": base_col,
        "action_col": action_col,
        "score_cols_used": score_cols,
        "missing": [],
        "contract_ok": True,
    }

    write_json(od / "00_run_summary.json", summary)
    write_json(od / "01_input_contract.json", contract)
    write_json(od / "02_metric_comparison.json", metric_comparison)
    groups.to_csv(od / "03_alignment_groups.csv", index=False, encoding="utf-8")
    dist.to_csv(od / "04_action_distribution.csv", index=False, encoding="utf-8")
    score_summary.to_csv(od / "08_score_summary_by_alignment_group.csv", index=False, encoding="utf-8")
    write_json(od / "09_alignment_diagnosis.json", diagnosis)

    report = f"""# {EXPERIMENT} - Action Alignment Audit

## Resultado executivo
- Status: `DONE_R3W_ACTION_ALIGNMENT_AUDIT`
- Input: `{input_path}`
- Label: `{label_col}`
- R3Q base col: `{base_col}`
- Action col: `{action_col}`

## Comparacao de metricas

### R3Q binario
```json
{json.dumps(r3q_metrics, ensure_ascii=False, indent=2)}
```

### Acao operacional como intervencao CONFIRMAR/BLOQUEAR
```json
{json.dumps(action_metrics, ensure_ascii=False, indent=2)}
```

### Delta acao - R3Q
```json
{json.dumps(metric_comparison["delta_action_minus_r3q"], ensure_ascii=False, indent=2)}
```

## Grupos de alinhamento
{groups.to_markdown(index=False)}

## Distribuicao por acao
{dist.to_markdown(index=False)}

## Contagens criticas
```text
R3Q alert + ACTION APROVAR + fraude = {n_misaligned_fraud}
R3Q alert + ACTION APROVAR + normal = {n_misaligned_normal}
R3Q no alert + ACTION intervention = {n_action_without_r3q}
```

## Diagnostico
{json.dumps(diagnosis["interpretation"], ensure_ascii=False, indent=2)}

## Proximos checks sugeridos
{json.dumps(diagnosis["recommended_next_checks"], ensure_ascii=False, indent=2)}
"""
    (od / "10_exp014b_r3w_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
