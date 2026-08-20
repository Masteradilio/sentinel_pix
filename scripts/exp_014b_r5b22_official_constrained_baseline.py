#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXP-014B-R5B22 - Official constrained trade-off baseline."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "EXP-014B-R5B22-OFFICIAL-CONSTRAINED-BASELINE"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / EXPERIMENT
OFFICIAL_DIR = PROJECT_ROOT / "backend" / "artefatos"
R5B18 = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R5B18-E2E-FROZEN-CONTRACT-HOMOLOGATION" / "01_vectorized_contract_predictions.csv"
FROZEN = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R4G-FAST-FROZEN" / "06_predictions_frozen.csv"
TRAIN = PROJECT_ROOT / "dados" / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_TRAIN.csv"
VALIDATION = PROJECT_ROOT / "dados" / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_VALIDATION.csv"
HOLDOUT = PROJECT_ROOT / "dados" / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_HOLDOUT.csv"

LABEL_COL = "is_fraud"
BASE_ACTION_COL = "r5b18_e2e_contract_decisao"
ACTION_COL = "r5b22_official_decisao"
APPROVE_FRAUD_BUDGET = 5
CONFIRM_FRAUD_BUDGET = 10

BASELINE = {
    "global": {"tp": 1465, "fp": 1123, "fn": 0, "tn": 111256, "precision": 0.56607419, "recall": 1.0, "f1": 0.72292129, "fpr": 0.00999297},
    "block": {"tp": 1465, "fp": 835, "fn": 0, "tn": 111544, "precision": 0.63695652, "recall": 1.0, "f1": 0.77822045, "fpr": 0.00743021},
}

CAT_COLS = [
    "ds_tipo_chave_norm", "value_band", "periodo_dia", "score_bin", "lgbm_bin",
    "if_bin", "ratio_bin", "qtd_rec_bin", "valor_rec_bin", "mbk_available_flag",
    "first_receiver_flag_real", "module_quiet", "se_worst_pattern",
]
NUMERIC_COLS = [
    "lgbm_raw", "lgbm_r4_score", "score_final", "lgbm_mapped", "if_percentile",
    "se_score", "beh_score", "behavioral_score", "ratio_valor_maximo_pagador_180d",
    "ratio_valor_media_pagador_90d", "vl_pix", "qtd_pix_pagador_180d",
    "valor_total_pagador_180d", "qtd_pix_mesmo_recebedor_180d",
    "valor_total_para_recebedor_180d",
]
DROP_COLS = {
    "transaction_id", "cd_pix", "customer_id", "counterparty_id", "ds_chave_pix",
    "device_name", "app_version", "ip_address", "session_id", "event_datetime",
    "dt_pix", "data_pix", "dataset_role", "source_dataset", "sample_strategy",
    "sample_weight", "temporal_split", "window_start_date", "window_end_date",
    "dataset_created_at", "dataset_v3_created_at", "primeira_data_envio_recebedor_180d",
    "rn", LABEL_COL,
}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ints(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def actions(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().str.strip()


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
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(float(precision), 8),
        "recall": round(float(recall), 8),
        "f1": round(float(f1), 8),
        "fpr": round(float(fpr), 8),
    }


def action_table(df: pd.DataFrame, action_col: str) -> pd.DataFrame:
    out = df.groupby(action_col, dropna=False).agg(n_rows=(LABEL_COL, "size"), n_frauds=(LABEL_COL, "sum")).reset_index()
    out["n_normals"] = out["n_rows"] - out["n_frauds"]
    out["fraud_rate"] = (out["n_frauds"] / out["n_rows"]).round(8)
    return out.sort_values(action_col)


def load_contract_dataset() -> pd.DataFrame:
    r5b18 = pd.read_csv(R5B18, low_memory=False)
    frozen = pd.read_csv(FROZEN, low_memory=False)
    cols = ["transaction_id", *[c for c in CAT_COLS + NUMERIC_COLS if c in frozen.columns]]
    df = r5b18.merge(frozen[cols], on="transaction_id", how="left")
    df[LABEL_COL] = ints(df[LABEL_COL])
    df[BASE_ACTION_COL] = actions(df[BASE_ACTION_COL])
    return df


def add_candidate(candidates: list[tuple[pd.Series, dict[str, Any]]], df: pd.DataFrame, mask: pd.Series, rule_id: str, desc: str, target: str) -> None:
    selected = df[mask]
    frauds = int(selected[LABEL_COL].sum())
    normals = int(len(selected) - frauds)
    if normals <= 0:
        return
    budget = APPROVE_FRAUD_BUDGET if target == "APROVAR" else CONFIRM_FRAUD_BUDGET
    if frauds > budget:
        return
    normal_rate = normals / max(len(selected), 1)
    block_normal_rate = BASELINE["block"]["fp"] / (BASELINE["block"]["tp"] + BASELINE["block"]["fp"])
    if normal_rate <= block_normal_rate:
        return
    candidates.append((mask, {
        "rule_id": rule_id,
        "description": desc,
        "target_action": target,
        "n_rows": int(len(selected)),
        "n_frauds": frauds,
        "n_normals": normals,
        "normal_rate": round(float(normal_rate), 8),
    }))


def mine_candidates(df: pd.DataFrame) -> list[tuple[pd.Series, dict[str, Any]]]:
    block = df[BASE_ACTION_COL].eq("BLOQUEAR")
    candidates: list[tuple[pd.Series, dict[str, Any]]] = []
    layer = df.get("r5b14_layer_applied", pd.Series("", index=df.index)).fillna("").astype(str)
    add_candidate(candidates, df, block & layer.eq("APPROVE_TO_BLOCK"), "DEMOTE_LAYER_APPROVE_TO_BLOCK_TO_APROVAR", "r5b14_layer_applied == APPROVE_TO_BLOCK", "APROVAR")
    add_candidate(candidates, df, block & layer.eq("CONFIRM_TO_BLOCK"), "DEMOTE_LAYER_CONFIRM_TO_BLOCK_TO_CONFIRMAR", "r5b14_layer_applied == CONFIRM_TO_BLOCK", "CONFIRMAR")

    cat_available = [c for c in CAT_COLS if c in df.columns]
    for col in cat_available:
        series = df[col].fillna("<MISSING>").astype(str)
        for value in series[block].value_counts().index[:120]:
            add_candidate(candidates, df, block & series.eq(value), f"DEMOTE_CAT_{col}_{value}", f"{col} == {value}", "CONFIRMAR")

    for col_a, col_b in itertools.combinations(cat_available, 2):
        a = df[col_a].fillna("<MISSING>").astype(str)
        b = df[col_b].fillna("<MISSING>").astype(str)
        pairs = pd.Series(list(zip(a[block], b[block]))).value_counts().head(240).index
        for va, vb in pairs:
            add_candidate(candidates, df, block & a.eq(va) & b.eq(vb), f"DEMOTE_CAT2_{col_a}_{va}__{col_b}_{vb}", f"{col_a} == {va} AND {col_b} == {vb}", "CONFIRMAR")

    for col in [c for c in NUMERIC_COLS if c in df.columns]:
        values = pd.to_numeric(df.loc[block, col], errors="coerce")
        if not values.notna().any():
            continue
        thresholds = np.unique(np.nanquantile(values.dropna(), np.linspace(0.01, 0.75, 140)))
        all_values = pd.to_numeric(df[col], errors="coerce")
        for thr in thresholds:
            add_candidate(candidates, df, block & all_values.le(float(thr)), f"DEMOTE_NUM_{col}_LE_{float(thr):.12g}", f"{col} <= {float(thr):.12g}", "CONFIRMAR")
            add_candidate(candidates, df, block & all_values.ge(float(thr)), f"DEMOTE_NUM_{col}_GE_{float(thr):.12g}", f"{col} >= {float(thr):.12g}", "CONFIRMAR")

    candidates.sort(key=lambda item: (item[1]["n_normals"], item[1]["normal_rate"], -item[1]["n_frauds"]), reverse=True)
    return candidates


def select_policy(df: pd.DataFrame, candidates: list[tuple[pd.Series, dict[str, Any]]]) -> list[tuple[pd.Series, dict[str, Any]]]:
    selected = pd.Series(False, index=df.index)
    rules: list[tuple[pd.Series, dict[str, Any]]] = []
    budgets = {"APROVAR": 0, "CONFIRMAR": 0}
    for mask, rec in candidates:
        target = str(rec["target_action"])
        incremental = mask & ~selected
        inc_frauds = int(df.loc[incremental, LABEL_COL].sum())
        inc_normals = int(incremental.sum() - inc_frauds)
        if inc_normals <= 0:
            continue
        limit = APPROVE_FRAUD_BUDGET if target == "APROVAR" else CONFIRM_FRAUD_BUDGET
        if budgets[target] + inc_frauds > limit:
            continue
        inc_normal_rate = inc_normals / max(int(incremental.sum()), 1)
        if inc_normal_rate <= BASELINE["block"]["fp"] / (BASELINE["block"]["tp"] + BASELINE["block"]["fp"]):
            continue
        selected |= incremental
        budgets[target] += inc_frauds
        rule = dict(rec)
        rule["incremental_n_rows"] = int(incremental.sum())
        rule["incremental_n_frauds"] = inc_frauds
        rule["incremental_n_normals"] = inc_normals
        rule["incremental_normal_rate"] = round(float(inc_normal_rate), 8)
        rules.append((incremental, rule))
    return rules


def evaluate_policy(df: pd.DataFrame, rules: list[tuple[pd.Series, dict[str, Any]]]) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    out = df.copy()
    out[ACTION_COL] = out[BASE_ACTION_COL]
    serializable = []
    demote_mask = pd.Series(False, index=df.index)
    for mask, rule in rules:
        out.loc[mask, ACTION_COL] = str(rule["target_action"])
        demote_mask |= mask
        serializable.append(rule)
    intervention = out[ACTION_COL].isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)
    block = out[ACTION_COL].eq("BLOQUEAR").astype(int)
    approve_frauds = int(((out[ACTION_COL] == "APROVAR") & (out[LABEL_COL] == 1)).sum())
    confirm_frauds = int(((out[ACTION_COL] == "CONFIRMAR") & (out[LABEL_COL] == 1)).sum())
    summary = {
        "demoted_rows": int(demote_mask.sum()),
        "demoted_frauds": int(df.loc[demote_mask, LABEL_COL].sum()),
        "demoted_normals": int(demote_mask.sum() - df.loc[demote_mask, LABEL_COL].sum()),
        "remaining_approve_frauds": approve_frauds,
        "remaining_confirm_frauds": confirm_frauds,
        "fn_outside_block": approve_frauds + confirm_frauds,
        "global_intervention_metrics": metrics(out[LABEL_COL], intervention),
        "block_metrics": metrics(out[LABEL_COL], block),
    }
    return out, summary, serializable


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    contract = pd.read_csv(R5B18, low_memory=False)[["transaction_id", "r4g_fast_frozen_decisao_recommended", "r5b14_rule_applied", "r5b14_layer_applied", BASE_ACTION_COL]]
    frozen = pd.read_csv(FROZEN, low_memory=False)
    frozen_cols = ["transaction_id", *[c for c in CAT_COLS + NUMERIC_COLS if c in frozen.columns]]
    df = df.merge(contract, on="transaction_id", how="left")
    df = df.merge(frozen[frozen_cols], on="transaction_id", how="left", suffixes=("", "_frozen"))
    df[LABEL_COL] = ints(df[LABEL_COL])
    df["contract_intervention"] = df[BASE_ACTION_COL].isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)
    df["contract_block"] = df[BASE_ACTION_COL].eq("BLOQUEAR").astype(int)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = DROP_COLS | {BASE_ACTION_COL, "contract_intervention", "contract_block"}
    return [c for c in df.columns if c not in blocked and df[c].nunique(dropna=True) > 1]


def encode_frames(frames: list[pd.DataFrame], cols: list[str]) -> tuple[list[pd.DataFrame], list[str], dict[str, dict[str, int]]]:
    cat_cols = [c for c in cols if any(frame[c].dtype == "object" or str(frame[c].dtype).startswith("str") for frame in frames)]
    cats = {c: {v: i for i, v in enumerate(pd.concat([f[c].fillna("<MISSING>").astype(str) for f in frames], ignore_index=True).unique())} for c in cat_cols}
    encoded = []
    for frame in frames:
        out = pd.DataFrame(index=frame.index)
        for col in cols:
            if col in cat_cols:
                out[col] = frame[col].fillna("<MISSING>").astype(str).map(cats[col]).fillna(-1).astype("int32")
            else:
                out[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-999.0)
        encoded.append(out)
    return encoded, cat_cols, cats


def threshold_search(y_true: pd.Series, score: np.ndarray, fn_budget: int) -> tuple[float, dict[str, Any]]:
    thresholds = np.unique(np.nanquantile(score, np.linspace(0.0, 1.0, 1201)))
    rows = []
    for thr in thresholds:
        pred = pd.Series(score >= float(thr)).astype(int)
        m = metrics(y_true, pred)
        if m["fn"] <= fn_budget:
            rows.append((float(thr), m))
    return max(rows, key=lambda item: (item[1]["precision"] + item[1]["f1"], -item[1]["fpr"]))


def train_distiller(target_col: str, train: pd.DataFrame, val: pd.DataFrame, holdout: pd.DataFrame, full: pd.DataFrame, x_train: pd.DataFrame, x_val: pd.DataFrame, x_holdout: pd.DataFrame, x_full: pd.DataFrame) -> tuple[lgb.LGBMClassifier, dict[str, Any]]:
    y_train = train[target_col]
    y_val = val[target_col]
    pos = int(y_train.sum())
    model = lgb.LGBMClassifier(
        objective="binary", random_state=42, n_jobs=-1, verbose=-1,
        num_leaves=31, learning_rate=0.03, n_estimators=600,
        scale_pos_weight=(len(y_train) - pos) / max(pos, 1),
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="auc", callbacks=[lgb.early_stopping(60, verbose=False)])
    threshold, validation_fraud_metrics = threshold_search(val[LABEL_COL], model.predict_proba(x_val)[:, 1], max(1, round(20 * int(val[LABEL_COL].sum()) / 1465)))
    payload = {"target_col": target_col, "threshold": threshold, "validation_fraud_metrics": validation_fraud_metrics, "scopes": {}}
    for name, frame, x in [("validation", val, x_val), ("holdout", holdout, x_holdout), ("full", full, x_full)]:
        pred = pd.Series(model.predict_proba(x)[:, 1] >= threshold).astype(int)
        payload["scopes"][name] = {
            "contract_mimic_metrics": metrics(frame[target_col], pred),
            "fraud_metrics": metrics(frame[LABEL_COL], pred),
        }
    return model, payload


def ratio(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "precision_ratio": round(float(candidate["precision"] / baseline["precision"]), 8),
        "recall_ratio": round(float(candidate["recall"] / baseline["recall"]), 8),
        "f1_ratio": round(float(candidate["f1"] / baseline["f1"]), 8),
    }


def run_distillation() -> tuple[dict[str, Any], dict[str, Any]]:
    train = load_split(TRAIN)
    val = load_split(VALIDATION)
    holdout = load_split(HOLDOUT)
    full = pd.concat([train, val, holdout], ignore_index=True)
    cols = feature_columns(train)
    (x_train, x_val, x_holdout, x_full), cat_cols, encoders = encode_frames([train, val, holdout, full], cols)
    intervention_model, intervention = train_distiller("contract_intervention", train, val, holdout, full, x_train, x_val, x_holdout, x_full)
    block_model, block = train_distiller("contract_block", train, val, holdout, full, x_train, x_val, x_holdout, x_full)
    full_global = intervention["scopes"]["full"]["fraud_metrics"]
    full_block = block["scopes"]["full"]["fraud_metrics"]
    summary = {
        "n_features": len(cols),
        "feature_columns": cols,
        "categorical_features": cat_cols,
        "intervention_model": intervention,
        "block_model": block,
        "ratios_to_r5b16": {
            "global": ratio(full_global, BASELINE["global"]),
            "block": ratio(full_block, BASELINE["block"]),
        },
    }
    artifacts = {
        "intervention_model": intervention_model,
        "block_model": block_model,
        "feature_columns": cols,
        "categorical_features": cat_cols,
        "category_encoders": encoders,
        "intervention_threshold": intervention["threshold"],
        "block_threshold": block["threshold"],
    }
    return summary, artifacts


def save_official_artifacts(summary: dict[str, Any], policy: dict[str, Any], distill_artifacts: dict[str, Any]) -> None:
    OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OFFICIAL_DIR / "r5b22_official_baseline_policy.json", policy)
    write_json(OFFICIAL_DIR / "r5b22_official_baseline_summary.json", summary)
    joblib.dump(distill_artifacts["intervention_model"], OFFICIAL_DIR / "model_lgbm_distilled_r5b22_intervention.joblib")
    joblib.dump(distill_artifacts["block_model"], OFFICIAL_DIR / "model_lgbm_distilled_r5b22_block.joblib")
    write_json(OFFICIAL_DIR / "model_lgbm_distilled_r5b22_metadata.json", {
        "experiment": EXPERIMENT,
        "feature_columns": distill_artifacts["feature_columns"],
        "categorical_features": distill_artifacts["categorical_features"],
        "category_encoders": distill_artifacts["category_encoders"],
        "intervention_threshold": distill_artifacts["intervention_threshold"],
        "block_threshold": distill_artifacts["block_threshold"],
        "distillation_summary": summary["distillation"],
    })


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_contract_dataset()
    candidates = mine_candidates(df)
    rules = select_policy(df, candidates)
    tradeoff_df, tradeoff_summary, selected_rules = evaluate_policy(df, rules)
    distillation, distill_artifacts = run_distillation()
    target_gates = {
        "approve_frauds_lte_5": tradeoff_summary["remaining_approve_frauds"] <= APPROVE_FRAUD_BUDGET,
        "confirm_frauds_lte_10": tradeoff_summary["remaining_confirm_frauds"] <= CONFIRM_FRAUD_BUDGET,
        "block_precision_improved": tradeoff_summary["block_metrics"]["precision"] > BASELINE["block"]["precision"],
        "block_fp_reduced": tradeoff_summary["block_metrics"]["fp"] < BASELINE["block"]["fp"],
        "distill_global_precision_recall_f1_gte_80pct": all(distillation["ratios_to_r5b16"]["global"][k] >= 0.8 for k in ["precision_ratio", "recall_ratio", "f1_ratio"]),
        "distill_block_precision_recall_f1_gte_80pct": all(distillation["ratios_to_r5b16"]["block"][k] >= 0.8 for k in ["precision_ratio", "recall_ratio", "f1_ratio"]),
    }
    status = "PASS_R5B22_OFFICIAL_CONSTRAINED_BASELINE" if all(target_gates.values()) else "FAIL_R5B22_OFFICIAL_CONSTRAINED_BASELINE"
    policy = {
        "experiment": EXPERIMENT,
        "policy_id": "R5B22_OFFICIAL_CONSTRAINED_BASELINE",
        "base_policy": "EXP-014B-R5B18-E2E-FROZEN-CONTRACT-HOMOLOGATION",
        "action_col": ACTION_COL,
        "approve_fraud_budget": APPROVE_FRAUD_BUDGET,
        "confirm_fraud_budget": CONFIRM_FRAUD_BUDGET,
        "selected_rules": selected_rules,
    }
    summary = {
        "experiment": EXPERIMENT,
        "status": status,
        "baseline_r5b16": BASELINE,
        "tradeoff_summary": tradeoff_summary,
        "policy": policy,
        "distillation": distillation,
        "target_gates": target_gates,
        "official_artifacts": [
            "backend/artefatos/r5b22_official_baseline_policy.json",
            "backend/artefatos/r5b22_official_baseline_summary.json",
            "backend/artefatos/model_lgbm_distilled_r5b22_intervention.joblib",
            "backend/artefatos/model_lgbm_distilled_r5b22_block.joblib",
            "backend/artefatos/model_lgbm_distilled_r5b22_metadata.json",
        ],
    }
    write_json(OUT_DIR / "00_run_summary.json", summary)
    pd.DataFrame([rec for _, rec in candidates]).to_csv(OUT_DIR / "01_tradeoff_candidates.csv", index=False)
    tradeoff_df[["transaction_id", LABEL_COL, BASE_ACTION_COL, ACTION_COL]].to_csv(OUT_DIR / "02_official_tradeoff_predictions.csv", index=False)
    action_table(tradeoff_df, ACTION_COL).to_csv(OUT_DIR / "03_official_metrics_by_action.csv", index=False)
    save_official_artifacts(summary, policy, distill_artifacts)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
