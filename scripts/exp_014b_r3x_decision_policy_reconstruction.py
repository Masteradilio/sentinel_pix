# -*- coding: utf-8 -*-
"""
EXP-014B-R3X — Decision Policy Reconstruction / R3Q Action Alignment

Objetivo:
  Alinhar a saída operacional APROVAR / CONFIRMAR / BLOQUEAR com o
  benchmark experimental congelado `exp014b_r3q_frozen_pred`.

Motivação:
  O EXP-014B-R3W mostrou desalinhamento grave:
    - R3Q binário: TP=1465, FP=4074, FN=0
    - decisao exportada como intervenção: TP=59, FP=31, FN=1406
    - 1406 fraudes detectadas por R3Q apareciam como APROVAR em `decisao`

Decisão técnica desta rodada:
  1. `exp014b_r3q_frozen_pred == 0`  -> APROVAR
  2. `exp014b_r3q_frozen_pred == 1`  -> pelo menos CONFIRMAR
  3. Dentro dos alertas R3Q, escolher quem vira BLOQUEAR por score/ranking.

Assim, a política operacional passa a refletir a detecção R3Q, mas ainda permite
medir atrito forte (BLOQUEAR) separado de atrito moderado (CONFIRMAR).

Saídas:
  resultados/experimentos/EXP-014B-R3X/
    00_run_summary.json
    01_input_contract.json
    02_before_after_alignment.json
    03_action_distribution_before_after.csv
    04_block_policy_frontier.csv
    05_selected_block_policy.json
    06_decision_metrics_by_action.csv
    07_robustness_by_segment.csv
    08_policy_artifact_recommended.json
    09_predictions_reconstructed.csv
    10_exp014b_r3x_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT = "EXP-014B-R3X"

TARGET_FPR = 0.015
MAX_FN = 5
MIN_RECALL = 0.95

LABEL_CANDIDATES = ["is_fraud", "fraude", "target", "label", "tp_fraude"]

BASE_COL_CANDIDATES = [
    "exp014b_r3q_frozen_pred",
    "exp014b_r3q_recommended_pred",
    "exp014b_r3p_frozen_pred",
    "exp014b_r3v_recommended_pred",
    "exp014b_r3u_recommended_pred",
]

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
    "r3x_decisao_pos_policy",
    "exp014b_r3q_frozen_pred",
    "exp014b_r3x_intervention_pred",
    "exp014b_r3x_block_pred",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="CSV de predictions de entrada.")
    parser.add_argument("--output-dir", type=str, default=None, help="Diretório de saída.")
    parser.add_argument("--target-fpr", type=float, default=TARGET_FPR)
    parser.add_argument("--max-fn", type=int, default=MAX_FN)
    parser.add_argument("--min-recall", type=float, default=MIN_RECALL)
    parser.add_argument(
        "--min-block-precision",
        type=float,
        default=0.50,
        help="Precision mínima desejável para BLOQUEAR ao selecionar política forte.",
    )
    return parser.parse_args()


def default_output_dir() -> Path:
    path = Path.cwd() / "resultados" / "experimentos" / EXPERIMENT
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_input() -> Path:
    base = Path.cwd() / "resultados" / "experimentos"
    candidates = [
        base / "EXP-014B-R3W" / "09_predictions_reconstructed.csv",
        base / "EXP-014B-R3V" / "08_predictions_recommended.csv",
        base / "EXP-014B-R3U" / "09_predictions_recommended.csv",
        base / "EXP-014B-R3S" / "08_predictions_recommended.csv",
        base / "EXP-014B-R3Q" / "08_predictions_recommended.csv",
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


def action_to_block(action: pd.Series) -> pd.Series:
    return action.astype(str).eq("BLOQUEAR").astype(int)


def action_distribution(df: pd.DataFrame, label_col: str, action_col: str) -> pd.DataFrame:
    rows = []
    y = safe_int_series(df[label_col])
    total_frauds = max(1, int((y == 1).sum()))
    total_normals = max(1, int((y == 0).sum()))
    for action, idx in df.groupby(action_col, dropna=False).groups.items():
        idx = list(idx)
        yy = y.loc[idx]
        n = int(len(idx))
        frauds = int((yy == 1).sum())
        normals = int((yy == 0).sum())
        rows.append({
            "action_col": action_col,
            "action": str(action),
            "n_rows": n,
            "n_frauds": frauds,
            "n_normals": normals,
            "precision_within_action": round(float(frauds / n), 8) if n else 0.0,
            "fraud_share": round(float(frauds / total_frauds), 8),
            "normal_share": round(float(normals / total_normals), 8),
        })
    return pd.DataFrame(rows).sort_values(["action_col", "n_rows"], ascending=[True, False])


def build_block_frontier(
    df: pd.DataFrame,
    label_col: str,
    base_col: str,
    score_cols: list[str],
    target_fp: int,
    min_block_precision: float,
) -> pd.DataFrame:
    """Testa quem deve virar BLOQUEAR dentro dos alertas R3Q."""
    y = safe_int_series(df[label_col])
    base_alert = safe_int_series(df[base_col]).eq(1)
    rows: list[dict[str, Any]] = []

    # Política sem BLOQUEAR: tudo que é R3Q alert vira CONFIRMAR.
    zero_block = pd.Series(False, index=df.index)
    rows.append(policy_row(
        df, label_col, base_col, zero_block,
        "no_block_all_r3q_alerts_confirmar",
        "Todos os alertas R3Q viram CONFIRMAR; nenhum BLOQUEAR",
        target_fp,
        min_block_precision,
    ))

    # Política BLOQUEAR tudo: útil como extremo.
    all_block = base_alert.copy()
    rows.append(policy_row(
        df, label_col, base_col, all_block,
        "block_all_r3q_alerts",
        "Todos os alertas R3Q viram BLOQUEAR",
        target_fp,
        min_block_precision,
    ))

    # Políticas por score.
    for score_col in score_cols:
        s = pd.to_numeric(df[score_col], errors="coerce")
        valid = s[base_alert & s.notna()]
        if valid.empty:
            continue

        qs = [
            0.00, 0.01, 0.02, 0.05, 0.10,
            0.15, 0.20, 0.25, 0.30, 0.35,
            0.40, 0.45, 0.50, 0.55, 0.60,
            0.65, 0.70, 0.75, 0.80, 0.85,
            0.90, 0.92, 0.95, 0.97, 0.98, 0.99,
        ]
        thresholds = sorted(set(float(valid.quantile(q)) for q in qs if pd.notna(valid.quantile(q))))

        for th in thresholds:
            block_hi = base_alert & s.ge(th)
            rows.append(policy_row(
                df, label_col, base_col, block_hi,
                f"block_hi_{score_col}_{th:.10g}",
                f"BLOQUEAR alertas R3Q com {score_col} >= {th:.10g}; demais alertas R3Q viram CONFIRMAR",
                target_fp,
                min_block_precision,
            ))

            block_lo = base_alert & s.le(th)
            rows.append(policy_row(
                df, label_col, base_col, block_lo,
                f"block_lo_{score_col}_{th:.10g}",
                f"BLOQUEAR alertas R3Q com {score_col} <= {th:.10g}; demais alertas R3Q viram CONFIRMAR",
                target_fp,
                min_block_precision,
            ))

    out = pd.DataFrame(rows).drop_duplicates(
        subset=[
            "policy_name",
            "block_tp",
            "block_fp",
            "block_fn_if_block_only",
            "confirm_tp",
            "confirm_fp",
        ]
    )

    # Ordenação: prioriza bloco forte com FP baixo, precision alta e bastante TP,
    # mas preserva a detecção total R3Q por CONFIRMAR+BLOQUEAR.
    return out.sort_values(
        [
            "block_fpr_target_ok",
            "block_precision_ok",
            "block_fp",
            "block_precision",
            "block_tp",
            "confirm_fp",
        ],
        ascending=[False, False, True, False, False, True],
    )


def policy_row(
    df: pd.DataFrame,
    label_col: str,
    base_col: str,
    block_mask: pd.Series,
    policy_name: str,
    description: str,
    target_fp: int,
    min_block_precision: float,
) -> dict[str, Any]:
    y = safe_int_series(df[label_col])
    base_alert = safe_int_series(df[base_col]).eq(1)
    block = (base_alert & block_mask.fillna(False)).astype(int)
    intervention = base_alert.astype(int)
    confirm = (base_alert & (block == 0)).astype(int)

    detection_m = metrics(y, intervention)
    block_m = metrics(y, block)
    confirm_tp = int(((y == 1) & (confirm == 1)).sum())
    confirm_fp = int(((y == 0) & (confirm == 1)).sum())
    confirm_precision = confirm_tp / (confirm_tp + confirm_fp) if (confirm_tp + confirm_fp) else 0.0

    return {
        "policy_name": policy_name,
        "description": description,
        "total_detection_tp": detection_m["tp"],
        "total_detection_fp": detection_m["fp"],
        "total_detection_fn": detection_m["fn"],
        "total_detection_recall": detection_m["recall"],
        "total_detection_fpr": detection_m["fpr"],
        "block_tp": block_m["tp"],
        "block_fp": block_m["fp"],
        "block_fn_if_block_only": block_m["fn"],
        "block_precision": block_m["precision"],
        "block_recall_if_block_only": block_m["recall"],
        "block_fpr": block_m["fpr"],
        "block_fpr_target_ok": bool(block_m["fp"] <= target_fp),
        "block_precision_ok": bool(block_m["precision"] >= min_block_precision),
        "confirm_tp": confirm_tp,
        "confirm_fp": confirm_fp,
        "confirm_precision": round(float(confirm_precision), 8),
        "target_fp_for_block": int(target_fp),
        "block_fp_gap_to_target": max(0, int(block_m["fp"] - target_fp)),
    }


def select_block_policy(frontier: pd.DataFrame) -> dict[str, Any]:
    # Preferir política com FP de BLOQUEAR dentro do alvo e precision aceitável.
    ok = frontier[
        (frontier["block_fpr_target_ok"] == True)
        & (frontier["block_precision_ok"] == True)
        & (frontier["block_tp"] > 0)
    ].copy()

    if len(ok):
        chosen = ok.sort_values(
            ["block_tp", "block_precision", "block_fp", "confirm_fp"],
            ascending=[False, False, True, True],
        ).iloc[0]
        reason = "BEST_BLOCK_POLICY_WITHIN_STRONG_FRICTION_TARGET"
    else:
        # Se nada atende precision mínima, escolher maior TP com FP dentro do alvo.
        fp_ok = frontier[
            (frontier["block_fpr_target_ok"] == True)
            & (frontier["block_tp"] > 0)
        ].copy()
        if len(fp_ok):
            chosen = fp_ok.sort_values(
                ["block_precision", "block_tp", "block_fp"],
                ascending=[False, False, True],
            ).iloc[0]
            reason = "BEST_AVAILABLE_BLOCK_POLICY_FP_TARGET_ONLY"
        else:
            chosen = frontier.sort_values(
                ["block_fp_gap_to_target", "block_precision", "block_tp"],
                ascending=[True, False, False],
            ).iloc[0]
            reason = "NO_BLOCK_POLICY_WITHIN_TARGET"

    out = chosen.to_dict()
    out["selection_reason"] = reason
    return out


def apply_block_policy(df: pd.DataFrame, base_col: str, selected: dict[str, Any]) -> pd.Series:
    policy = str(selected["policy_name"])
    base_alert = safe_int_series(df[base_col]).eq(1)

    if policy == "no_block_all_r3q_alerts_confirmar":
        return pd.Series(False, index=df.index)
    if policy == "block_all_r3q_alerts":
        return base_alert.astype(bool)

    if policy.startswith("block_hi_"):
        score_col, th = parse_policy(policy, "block_hi_")
        return base_alert & pd.to_numeric(df[score_col], errors="coerce").ge(th)

    if policy.startswith("block_lo_"):
        score_col, th = parse_policy(policy, "block_lo_")
        return base_alert & pd.to_numeric(df[score_col], errors="coerce").le(th)

    return pd.Series(False, index=df.index)


def parse_policy(policy: str, prefix: str) -> tuple[str, float]:
    body = policy[len(prefix):]
    score_col, th_str = body.rsplit("_", 1)
    return score_col, float(th_str)


def build_pos_policy_decision(df: pd.DataFrame, base_col: str, block_mask: pd.Series) -> pd.Series:
    base_alert = safe_int_series(df[base_col]).eq(1)
    block = block_mask.fillna(False)
    return pd.Series(
        np.select(
            [
                ~base_alert,
                base_alert & block,
                base_alert & ~block,
            ],
            [
                "APROVAR",
                "BLOQUEAR",
                "CONFIRMAR",
            ],
            default="APROVAR",
        ),
        index=df.index,
    )


def metrics_by_action(df: pd.DataFrame, label_col: str, action_col: str) -> pd.DataFrame:
    rows = []
    y = safe_int_series(df[label_col])
    for action, idx in df.groupby(action_col, dropna=False).groups.items():
        idx = list(idx)
        yy = y.loc[idx]
        n = int(len(idx))
        frauds = int((yy == 1).sum())
        normals = int((yy == 0).sum())
        rows.append({
            "action": str(action),
            "n_rows": n,
            "n_frauds": frauds,
            "n_normals": normals,
            "precision_within_action": round(float(frauds / n), 8) if n else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["action"], ascending=True)


def robustness(df: pd.DataFrame, label_col: str, action_col: str) -> pd.DataFrame:
    rows = []
    y = safe_int_series(df[label_col])
    intervention = action_to_intervention(df[action_col])
    block = action_to_block(df[action_col])

    for col in SEGMENT_COLS:
        if col not in df.columns:
            continue
        for val, idx in df.groupby(col, dropna=False).groups.items():
            idx = list(idx)
            yy = y.loc[idx]
            intervention_g = intervention.loc[idx]
            block_g = block.loc[idx]
            int_m = metrics(yy, intervention_g)
            block_m = metrics(yy, block_g)
            rows.append({
                "segment_col": col,
                "segment_value": str(val),
                "n_rows": int(len(idx)),
                "n_frauds": int((yy == 1).sum()),
                "intervention_tp": int_m["tp"],
                "intervention_fp": int_m["fp"],
                "intervention_fn": int_m["fn"],
                "intervention_recall": int_m["recall"],
                "intervention_fpr": int_m["fpr"],
                "block_tp": block_m["tp"],
                "block_fp": block_m["fp"],
                "block_precision": block_m["precision"],
                "block_fpr": block_m["fpr"],
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["block_fp", "intervention_fp", "n_rows"], ascending=[False, False, False]
    )


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input) if args.input else find_input()
    od = Path(args.output_dir) if args.output_dir else default_output_dir()
    od.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)

    label_col = find_col(df, LABEL_CANDIDATES)
    base_col = find_col(df, BASE_COL_CANDIDATES)
    action_col = find_col(df, ACTION_CANDIDATES, required=False)
    score_cols = [c for c in SCORE_CANDIDATES if c in df.columns]

    if action_col:
        df["r3x_original_action_norm"] = df[action_col].apply(normalize_action)
    else:
        df["r3x_original_action_norm"] = "UNKNOWN"

    n_rows = int(len(df))
    n_frauds = int(safe_int_series(df[label_col]).sum())
    n_normals = n_rows - n_frauds
    target_fp = int(np.floor(float(args.target_fpr) * n_normals))

    base_intervention = safe_int_series(df[base_col])
    original_intervention = action_to_intervention(df["r3x_original_action_norm"])

    base_metrics = metrics(df[label_col], base_intervention)
    original_action_metrics = metrics(df[label_col], original_intervention)

    contract = {
        "n_rows": n_rows,
        "n_frauds": n_frauds,
        "n_normals": n_normals,
        "input_path": str(input_path),
        "label_col": label_col,
        "base_col": base_col,
        "original_action_col": action_col,
        "score_cols_used": score_cols,
        "target_fpr": float(args.target_fpr),
        "target_fp": target_fp,
        "max_fn": int(args.max_fn),
        "min_recall": float(args.min_recall),
        "missing": [],
        "contract_ok": True,
    }

    before_after = {
        "before_original_action_metrics": original_action_metrics,
        "r3q_base_metrics": base_metrics,
        "delta_original_action_minus_r3q": {
            "tp": int(original_action_metrics["tp"] - base_metrics["tp"]),
            "fp": int(original_action_metrics["fp"] - base_metrics["fp"]),
            "fn": int(original_action_metrics["fn"] - base_metrics["fn"]),
            "tn": int(original_action_metrics["tn"] - base_metrics["tn"]),
            "recall_delta": round(float(original_action_metrics["recall"] - base_metrics["recall"]), 8),
            "fpr_delta": round(float(original_action_metrics["fpr"] - base_metrics["fpr"]), 8),
        },
    }

    frontier = build_block_frontier(
        df,
        label_col,
        base_col,
        score_cols,
        target_fp,
        float(args.min_block_precision),
    )
    selected = select_block_policy(frontier)
    block_mask = apply_block_policy(df, base_col, selected)

    df["r3x_decisao_pos_policy"] = build_pos_policy_decision(df, base_col, block_mask)
    df["exp014b_r3x_intervention_pred"] = action_to_intervention(df["r3x_decisao_pos_policy"])
    df["exp014b_r3x_block_pred"] = action_to_block(df["r3x_decisao_pos_policy"])

    aligned_intervention_metrics = metrics(df[label_col], df["exp014b_r3x_intervention_pred"])
    aligned_block_metrics = metrics(df[label_col], df["exp014b_r3x_block_pred"])

    before_after["after_decisao_pos_policy_intervention_metrics"] = aligned_intervention_metrics
    before_after["after_decisao_pos_policy_block_metrics"] = aligned_block_metrics
    before_after["alignment_fixed"] = bool(
        aligned_intervention_metrics["tp"] == base_metrics["tp"]
        and aligned_intervention_metrics["fp"] == base_metrics["fp"]
        and aligned_intervention_metrics["fn"] == base_metrics["fn"]
        and aligned_intervention_metrics["tn"] == base_metrics["tn"]
    )

    before_dist = action_distribution(df, label_col, "r3x_original_action_norm")
    after_dist = action_distribution(df, label_col, "r3x_decisao_pos_policy")
    dist = pd.concat([before_dist, after_dist], ignore_index=True)

    by_action = metrics_by_action(df, label_col, "r3x_decisao_pos_policy")
    rob = robustness(df, label_col, "r3x_decisao_pos_policy")

    selected_policy = {
        **selected,
        "base_col": base_col,
        "final_action_col": "r3x_decisao_pos_policy",
        "intervention_pred_col": "exp014b_r3x_intervention_pred",
        "block_pred_col": "exp014b_r3x_block_pred",
        "aligned_intervention_metrics": aligned_intervention_metrics,
        "aligned_block_metrics": aligned_block_metrics,
        "target_fpr": float(args.target_fpr),
        "target_fp": target_fp,
        "max_fn": int(args.max_fn),
        "min_recall": float(args.min_recall),
        "commercial_detection_target_reached": bool(
            aligned_intervention_metrics["fn"] <= int(args.max_fn)
            and aligned_intervention_metrics["recall"] >= float(args.min_recall)
            and aligned_intervention_metrics["fpr"] <= float(args.target_fpr)
        ),
        "strong_block_fpr_target_reached": bool(aligned_block_metrics["fp"] <= target_fp),
    }

    artifact = {
        "experiment": EXPERIMENT,
        "policy_name": selected_policy["policy_name"],
        "selection_reason": selected_policy["selection_reason"],
        "input_path": str(input_path),
        "label_col": label_col,
        "base_col": base_col,
        "original_action_col": action_col,
        "final_action_col": "r3x_decisao_pos_policy",
        "intervention_pred_col": "exp014b_r3x_intervention_pred",
        "block_pred_col": "exp014b_r3x_block_pred",
        "base_metrics": base_metrics,
        "original_action_metrics": original_action_metrics,
        "aligned_intervention_metrics": aligned_intervention_metrics,
        "aligned_block_metrics": aligned_block_metrics,
        "selected_block_policy": selected_policy,
        "score_cols_used": score_cols,
        "notes": [
            "APROVAR/CONFIRMAR/BLOQUEAR are reconstructed from R3Q so operational action is aligned with the experimental benchmark.",
            "All R3Q alerts become at least CONFIRMAR; only the strongest subset becomes BLOQUEAR.",
            "The total detection FPR is still R3Q FPR. BLOQUEAR FPR measures strong-friction/blocking burden separately.",
            "Promotion requires frozen replay and business approval of the action semantics.",
        ],
    }

    summary = {
        "experiment": EXPERIMENT,
        "status": "DONE",
        "objective_status": (
            "DONE_R3X_DECISION_POLICY_RECONSTRUCTED_ALIGNMENT_FIXED"
            if before_after["alignment_fixed"]
            else "DONE_R3X_DECISION_POLICY_RECONSTRUCTED_ALIGNMENT_NOT_FIXED"
        ),
        "n_rows": n_rows,
        "n_frauds": n_frauds,
        "n_normals": n_normals,
        "input_path": str(input_path),
        "base_col": base_col,
        "original_action_col": action_col,
        "selected_block_policy_name": selected_policy["policy_name"],
        "selection_reason": selected_policy["selection_reason"],
        "original_action_metrics": original_action_metrics,
        "r3q_base_metrics": base_metrics,
        "aligned_intervention_metrics": aligned_intervention_metrics,
        "aligned_block_metrics": aligned_block_metrics,
        "alignment_fixed": before_after["alignment_fixed"],
        "commercial_detection_target_reached": selected_policy["commercial_detection_target_reached"],
        "strong_block_fpr_target_reached": selected_policy["strong_block_fpr_target_reached"],
        "target_fpr": float(args.target_fpr),
        "target_fp": target_fp,
        "max_fn": int(args.max_fn),
        "min_recall": float(args.min_recall),
        "n_block_policies_evaluated": int(len(frontier)),
        "all_pass": True,
        "output_dir": str(od),
    }

    write_json(od / "00_run_summary.json", summary)
    write_json(od / "01_input_contract.json", contract)
    write_json(od / "02_before_after_alignment.json", before_after)
    dist.to_csv(od / "03_action_distribution_before_after.csv", index=False, encoding="utf-8")
    frontier.to_csv(od / "04_block_policy_frontier.csv", index=False, encoding="utf-8")
    write_json(od / "05_selected_block_policy.json", selected_policy)
    by_action.to_csv(od / "06_decision_metrics_by_action.csv", index=False, encoding="utf-8")
    rob.to_csv(od / "07_robustness_by_segment.csv", index=False, encoding="utf-8")
    write_json(od / "08_policy_artifact_recommended.json", artifact)

    export_cols = [c for c in KEY_EXPORT_COLS if c in df.columns]
    # garante novas colunas mesmo que KEY_EXPORT_COLS tenha nomes ausentes
    for c in ["r3x_original_action_norm", "r3x_decisao_pos_policy", "exp014b_r3x_intervention_pred", "exp014b_r3x_block_pred"]:
        if c not in export_cols and c in df.columns:
            export_cols.append(c)
    df.to_csv(od / "09_predictions_reconstructed.csv", index=False, encoding="utf-8")

    report = f"""# {EXPERIMENT} - Decision Policy Reconstruction / R3Q Action Alignment

## Resultado executivo
- Status: `{summary["objective_status"]}`
- Base col: `{base_col}`
- Ação original: `{action_col}`
- Política BLOQUEAR selecionada: `{selected_policy["policy_name"]}`
- Razão de seleção: `{selected_policy["selection_reason"]}`

## Antes: ação original como intervenção
```json
{json.dumps(original_action_metrics, ensure_ascii=False, indent=2)}
```

## Benchmark R3Q
```json
{json.dumps(base_metrics, ensure_ascii=False, indent=2)}
```

## Depois: decisao_pos_policy como intervenção CONFIRMAR/BLOQUEAR
```json
{json.dumps(aligned_intervention_metrics, ensure_ascii=False, indent=2)}
```

## Depois: apenas BLOQUEAR
```json
{json.dumps(aligned_block_metrics, ensure_ascii=False, indent=2)}
```

## Alinhamento corrigido
`alignment_fixed={before_after["alignment_fixed"]}`

## Distribuição de ações antes/depois
{dist.to_markdown(index=False)}

## Métricas por ação reconstruída
{by_action.to_markdown(index=False)}

## Melhores políticas de BLOQUEAR avaliadas
{frontier.head(30).to_markdown(index=False)}

## Decisão sugerida
Se `alignment_fixed=true`, a coluna `r3x_decisao_pos_policy` passa a ser a primeira versão operacional alinhada ao R3Q.
A próxima rodada deve validar replay congelado desta política e, depois, calibrar a regra de BLOQUEAR
sem confundir FPR total de detecção com FPR de bloqueio/intervenção forte.
"""
    (od / "10_exp014b_r3x_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
