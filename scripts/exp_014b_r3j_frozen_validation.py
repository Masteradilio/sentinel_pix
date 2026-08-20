#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3J-FROZEN — Frozen Validation do candidato Rescue100

Objetivo:
  Validar de forma congelada o candidato recomendado pelo EXP-014B-R3J:

      EXP014B_R3J_RESCUE100
      TP=1422
      FP=4965
      FN=43
      recall=97,065%
      precision=22,264%
      FPR=4,418%
      Wilson low≈96,070%

  O script NÃO minera novos resgates, NÃO recalibra threshold e NÃO faz beam search.

Ele apenas:
  1. Carrega o R3H congelado como base.
  2. Lê o policy artifact recomendado do R3J.
  3. Lê os rescue candidates congelados do R3I.
  4. Reaplica exatamente os candidate_ids do cenário recomendado.
  5. Reaplica as regras de re-tightening do R3J.
  6. Confirma se reproduz exatamente TP=1422, FP=4965, FN=43.
  7. Salva os impactos por rescue e por re-tightening.

Entradas padrão:
  Base:
    resultados/experimentos/EXP-014B-R3H-FROZEN/10_predictions.csv

  R3J policy:
    resultados/experimentos/EXP-014B-R3J/08_policy_artifact_recommended.json

  R3I rescue candidates:
    resultados/experimentos/EXP-014B-R3I/07_rescue_candidates.csv

Uso:
  python scripts/exp_014b_r3j_frozen_validation.py

Critério de PASS:
  - schema OK;
  - scenario recomendado encontrado;
  - final_metrics = TP=1422, FP=4965, FN=43;
  - rescue recupera 13 FNs;
  - re-tightening remove 60 FPs;
  - re-tightening TP_loss=0;
  - Wilson low >= 0.95.

Saídas:
  resultados/experimentos/EXP-014B-R3J-FROZEN/
    00_run_summary.json
    01_input_contract.json
    02_base_metrics.csv
    03_rescue_replay_impact.csv
    04_retightening_replay_impact.csv
    05_frozen_metrics.csv
    06_wilson_recall_ci.csv
    07_false_negatives.csv
    08_false_positives_sample.csv
    09_policy_replay_artifact.json
    10_predictions.csv
    11_exp014b_r3j_frozen_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "dados").exists() else Path.cwd()

DEFAULT_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3H-FROZEN" / "10_predictions.csv"
DEFAULT_POLICY = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3J" / "08_policy_artifact_recommended.json"
DEFAULT_RESCUE_CANDIDATES = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3I" / "07_rescue_candidates.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3J-FROZEN"

BASE_COL = "exp014b_r3h_frozen_pred"
RESCUE_COL = "exp014b_r3j_frozen_rescue_pred"
FINAL_COL = "exp014b_r3j_frozen_pred"

EXPECTED = {
    "scenario": "rescue_budget_100",
    "tp": 1422,
    "fp": 4965,
    "fn": 43,
    "recall": 0.97064846,
    "precision": 0.22263974,
    "fpr": 0.04418085,
    "rescue_fn_recovered": 13,
    "rescue_fp_added": 90,
    "retightening_fp_removed": 60,
    "retightening_tp_loss": 0,
    "wilson_low_min": 0.95,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().split(".")[-1] for c in df.columns]

    if "transaction_id" not in df.columns and "cd_pix" in df.columns:
        df["transaction_id"] = df["cd_pix"]
    if "event_datetime" not in df.columns and "dt_pix" in df.columns:
        df["event_datetime"] = df["dt_pix"]

    if "is_fraud" not in df.columns:
        raise RuntimeError("Coluna obrigatória ausente: is_fraud")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)

    if "transaction_id" in df.columns:
        df["transaction_id"] = df["transaction_id"].astype("string").str.strip()

    for c in ["event_datetime", "data_pix"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    for c in [BASE_COL, "exp014b_r3h_frozen_pred", "exp014b_r3j_recommended_pred"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df.reset_index(drop=True)


def pick_col(df: pd.DataFrame, names: str | list[str]) -> str | None:
    if isinstance(names, str):
        names = [names]
    for n in names:
        if n in df.columns:
            return n
    return None


def num(df: pd.DataFrame, names: str | list[str], default: float = 0.0) -> pd.Series:
    col = pick_col(df, names)
    if col is None:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def qbin_series(s: pd.Series, name: str, bins: list[float]) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    labels = []
    edges = [-np.inf] + bins + [np.inf]
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        if np.isneginf(left):
            labels.append(f"{name}_LT_{right:g}")
        elif np.isposinf(right):
            labels.append(f"{name}_GE_{left:g}")
        else:
            labels.append(f"{name}_{left:g}_{right:g}")
    return pd.cut(vals, bins=edges, labels=labels, include_lowest=True).astype("string").fillna(f"{name}_MISSING").astype(str)


def add_bins_and_guards(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recria bins somente se ausentes.
    A entrada normal vem do R3H-FROZEN e já tem as colunas usadas pelo R3I/R3J.
    """
    df = df.copy()

    if "lgbm_bin" not in df.columns and pick_col(df, ["lgbm_r4_score", "lgbm_mapped", "lgbm_raw"]):
        df["lgbm_bin"] = qbin_series(num(df, ["lgbm_r4_score", "lgbm_mapped", "lgbm_raw"], 0.0), "lgbm", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1])
    if "if_bin" not in df.columns and pick_col(df, ["if_percentile", "if_percentile_x", "if_percentile_y"]):
        df["if_bin"] = qbin_series(num(df, ["if_percentile", "if_percentile_x", "if_percentile_y"], 0.0), "if", [0.32, 0.5, 0.7, 0.85, 0.95])
    if "score_bin" not in df.columns and "score_final" in df.columns:
        df["score_bin"] = qbin_series(num(df, "score_final", 0.0), "score", [0.5, 1, 2, 3, 5, 10])
    if "ratio_bin" not in df.columns and "ratio_valor_media_pagador_90d" in df.columns:
        df["ratio_bin"] = qbin_series(num(df, "ratio_valor_media_pagador_90d", 0.0), "ratio", [0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    if "qtd_rec_bin" not in df.columns and "qtd_pix_recebidos_180d" in df.columns:
        df["qtd_rec_bin"] = qbin_series(num(df, "qtd_pix_recebidos_180d", 0.0), "qtdrec", [0, 1, 2, 5, 10, 20, 50, 100])
    if "vl_bin" not in df.columns and "vl_pix" in df.columns:
        df["vl_bin"] = qbin_series(num(df, "vl_pix", 0.0), "vl", [20, 50, 100, 250, 500, 1000, 5000, 10000])

    if "module_quiet" not in df.columns:
        se_score = num(df, ["se_score_x", "se_score_y", "se_score"], 0.0)
        se_count = num(df, ["se_patterns_count", "se_pattern_count"], 0.0)
        beh_score = num(df, ["beh_score", "behavioral_score"], 0.0)
        beh_count = num(df, ["beh_factors_count", "behavioral_risk_factor_count"], 0.0)
        runtime = num(df, "runtime_flagged", 0.0)
        strong = (se_score >= 40) | (se_count >= 2) | (beh_score >= 25) | (beh_count >= 2) | (runtime >= 1)
        df["module_quiet"] = np.where(strong, "module_strong", "module_quiet")

    return df


def compute_metrics(y_true, y_pred) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 8),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 8),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 8),
        "fpr": round(float(fp / max(fp + tn, 1)), 8),
    }


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) / n) + (z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def parse_params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    return json.loads(str(raw).replace("Infinity", "1e999"))


def mask_from_params(df: pd.DataFrame, current_pred: np.ndarray, params: dict[str, Any], mode: str) -> tuple[np.ndarray, list[str]]:
    """
    mode='rescue': apply to not-alerted rows and set to 1.
    mode='retighten': apply to alerted rows and set to 0.
    """
    missing = []

    if params.get("type") in ["numeric_threshold_rescue", "numeric_retighten"]:
        c = params.get("col")
        if c not in df.columns:
            return np.zeros(len(df), dtype=bool), [str(c)]
        vals = num(df, c, 0.0).to_numpy(dtype=float)
        direction = params.get("direction")
        cut = float(params.get("cut"))
        mask = vals >= cut if direction == "ge" else vals <= cut

    elif params.get("type") in ["combo_rescue", "combo_retighten"]:
        mask = np.ones(len(df), dtype=bool)
        for c, v in zip(params.get("combo_cols", []), params.get("combo_values", [])):
            if c not in df.columns:
                missing.append(str(c))
                return np.zeros(len(df), dtype=bool), missing
            mask = mask & (df[c].astype("string").fillna("<MISSING>").astype(str).to_numpy() == str(v))
    else:
        return np.zeros(len(df), dtype=bool), [f"unsupported_type:{params.get('type')}"]

    if params.get("require_module_quiet", False):
        if "module_quiet" not in df.columns:
            return np.zeros(len(df), dtype=bool), ["module_quiet"]
        mask = mask & (df["module_quiet"].astype(str).to_numpy() == "module_quiet")

    if mode == "rescue":
        mask = mask & (current_pred.astype(int) == 0)
    elif mode == "retighten":
        mask = mask & (current_pred.astype(int) == 1)
        # R3J retighten rules were trained with scope rescue_added_only.
        # The artifact descriptions/params include scope, but the actual scope is enforced
        # by the current_pred after rescue. To avoid touching original R3H alerts, the
        # caller passes an additional added_only_mask when needed via params not available here.
    else:
        raise RuntimeError(f"mode inválido: {mode}")

    return mask, missing


def apply_rescues(df: pd.DataFrame, base_pred: np.ndarray, rescue_df: pd.DataFrame, selected_ids: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    y = df["is_fraud"].to_numpy(dtype=int)
    pred = base_pred.copy().astype(int)
    rows = []
    missing_all = []

    selected = rescue_df[rescue_df["candidate_id"].astype(str).isin(set(selected_ids))].copy()
    order = {cid: i for i, cid in enumerate(selected_ids)}
    selected["_order"] = selected["candidate_id"].astype(str).map(order)
    selected = selected.sort_values("_order")

    for _, row in selected.iterrows():
        params = parse_params(row["params_json"])
        mask, missing = mask_from_params(df, pred, params, mode="rescue")
        fn_recovered = int(((y == 1) & mask).sum())
        fp_added = int(((y == 0) & mask).sum())
        pred[mask] = 1

        rows.append({
            "candidate_id": row["candidate_id"],
            "description": row.get("description"),
            "family": row.get("family"),
            "fn_recovered_replay": fn_recovered,
            "fp_added_replay": fp_added,
            "n_added_replay": int(mask.sum()),
            "expected_fn_recovered": row.get("fn_recovered"),
            "expected_fp_added": row.get("fp_added"),
            "missing_columns": "|".join(missing),
            "params_json": row["params_json"],
        })
        missing_all.extend(missing)

    return pred, pd.DataFrame(rows)


def apply_retighten(df: pd.DataFrame, rescue_pred: np.ndarray, base_pred: np.ndarray, selected_rules: list[dict[str, Any]]) -> tuple[np.ndarray, pd.DataFrame]:
    y = df["is_fraud"].to_numpy(dtype=int)
    pred = rescue_pred.copy().astype(int)
    added_only = (rescue_pred.astype(int) == 1) & (base_pred.astype(int) == 0)
    rows = []

    for i, rule in enumerate(selected_rules):
        params = parse_params(rule.get("params_json") or rule.get("params") or "{}")
        # First compute generic retighten mask on current alerted rows.
        mask, missing = mask_from_params(df, pred, params, mode="retighten")
        # Enforce R3J frozen scope: only alerts added by rescue layer.
        if params.get("scope") == "rescue_added_only":
            mask = mask & added_only

        tp_loss = int(((y == 1) & mask).sum())
        fp_removed = int(((y == 0) & mask).sum())
        pred[mask] = 0

        rows.append({
            "rule_index": i,
            "rule_id": rule.get("rule_id"),
            "family": rule.get("family"),
            "description": rule.get("description"),
            "tp_loss_replay": tp_loss,
            "fp_removed_replay": fp_removed,
            "n_removed_replay": int(mask.sum()),
            "expected_tp_loss": rule.get("tp_loss"),
            "expected_fp_removed": rule.get("fp_removed"),
            "tp_loss_match_expected": (rule.get("tp_loss") is None) or int(rule.get("tp_loss")) == tp_loss,
            "fp_removed_match_expected": (rule.get("fp_removed") is None) or int(rule.get("fp_removed")) == fp_removed,
            "missing_columns": "|".join(missing),
            "params_json": rule.get("params_json"),
        })

    return pred, pd.DataFrame(rows)


def required_cols_from_params(params: dict[str, Any]) -> list[str]:
    if params.get("type") in ["numeric_threshold_rescue", "numeric_retighten"]:
        return [str(params.get("col"))]
    if params.get("type") in ["combo_rescue", "combo_retighten"]:
        return [str(c) for c in params.get("combo_cols", [])]
    return []


def make_contract(df: pd.DataFrame, policy: dict[str, Any], rescue_df: pd.DataFrame, scenario_name: str, base_col: str) -> dict[str, Any]:
    missing = []
    if "is_fraud" not in df.columns:
        missing.append("is_fraud")
    if base_col not in df.columns:
        missing.append(base_col)
    if rescue_df.empty:
        missing.append("rescue_candidates_nonempty")
    if scenario_name not in policy.get("scenario_artifacts", {}):
        missing.append(f"scenario_artifact:{scenario_name}")

    required_cols = {"is_fraud", base_col}
    scenario = policy.get("scenario_artifacts", {}).get(scenario_name, {})
    selected_ids = scenario.get("selected_rescue_candidate_ids", [])
    rescue_sel = rescue_df[rescue_df["candidate_id"].astype(str).isin(set(selected_ids))]
    for _, row in rescue_sel.iterrows():
        required_cols.update(required_cols_from_params(parse_params(row["params_json"])))

    for rule in scenario.get("selected_retighten_rules", []):
        params = parse_params(rule.get("params_json") or rule.get("params") or "{}")
        required_cols.update(required_cols_from_params(params))
        if params.get("require_module_quiet", False):
            required_cols.add("module_quiet")

    for c in sorted(required_cols):
        if c and c not in df.columns:
            missing.append(c)

    return {
        "n_rows": int(len(df)),
        "n_frauds": int(df["is_fraud"].sum()) if "is_fraud" in df.columns else None,
        "base_col": base_col,
        "scenario_name": scenario_name,
        "recommended_scenario_in_policy": policy.get("recommended_scenario"),
        "n_selected_rescue_ids": int(len(selected_ids)),
        "n_selected_retighten_rules": int(len(scenario.get("selected_retighten_rules", []))),
        "required_columns": sorted(required_cols),
        "missing_columns": sorted(set(missing)),
        "contract_ok": not missing,
    }


def make_report(summary: dict[str, Any], rescue_impact: pd.DataFrame, retighten_impact: pd.DataFrame, metrics_df: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014B-R3J-FROZEN — Frozen Validation")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Cenário: `{summary['scenario_name']}`")
    lines.append(f"- Final metrics: `{summary['final_metrics']}`")
    lines.append(f"- Wilson low: `{summary['wilson_recall_low']}`")
    lines.append("")
    lines.append("## Métricas")
    lines.append(metrics_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Checks")
    lines.append(f"- Expected metrics matched: `{summary['expected_metrics_matched']}`")
    lines.append(f"- Rescue FN recovered: `{summary['rescue_fn_recovered_vs_base']}`")
    lines.append(f"- Rescue FP added: `{summary['rescue_fp_added_vs_base']}`")
    lines.append(f"- Retightening FP removed: `{summary['retightening_fp_removed']}`")
    lines.append(f"- Retightening TP loss: `{summary['retightening_tp_loss']}`")
    lines.append(f"- Schema OK: `{summary['schema_ok']}`")
    lines.append(f"- Replay OK: `{summary['replay_ok']}`")
    lines.append("")
    lines.append("## Impacto dos rescues")
    if rescue_impact.empty:
        lines.append("Nenhum rescue aplicado.")
    else:
        show = ["candidate_id", "description", "fn_recovered_replay", "fp_added_replay", "missing_columns"]
        lines.append(rescue_impact[[c for c in show if c in rescue_impact.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Impacto do re-tightening")
    if retighten_impact.empty:
        lines.append("Nenhuma regra de re-tightening aplicada.")
    else:
        show = ["rule_id", "description", "tp_loss_replay", "fp_removed_replay", "missing_columns"]
        lines.append(retighten_impact[[c for c in show if c in retighten_impact.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Decisão")
    if str(summary["objective_status"]).startswith("PASS"):
        lines.append("O candidato R3J_RESCUE100 foi validado congelado. Próxima rodada: buscar microevolução curta FN First/FP Second ou preparar consolidação no Manifest/Journal.")
    else:
        lines.append("Validação congelada falhou. Corrigir divergências antes de promover.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--rescue-candidates", default=str(DEFAULT_RESCUE_CANDIDATES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--base-col", default=BASE_COL)
    parser.add_argument("--expected-tp", type=int, default=EXPECTED["tp"])
    parser.add_argument("--expected-fp", type=int, default=EXPECTED["fp"])
    parser.add_argument("--expected-fn", type=int, default=EXPECTED["fn"])
    parser.add_argument("--expected-rescue-fn-recovered", type=int, default=EXPECTED["rescue_fn_recovered"])
    parser.add_argument("--expected-rescue-fp-added", type=int, default=EXPECTED["rescue_fp_added"])
    parser.add_argument("--expected-retightening-fp-removed", type=int, default=EXPECTED["retightening_fp_removed"])
    parser.add_argument("--expected-retightening-tp-loss", type=int, default=EXPECTED["retightening_tp_loss"])
    parser.add_argument("--wilson-low-min", type=float, default=EXPECTED["wilson_low_min"])
    parser.add_argument("--no-write-predictions", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    policy_path = Path(args.policy)
    rescue_path = Path(args.rescue_candidates)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014B-R3J-FROZEN — Frozen Validation")
    log("=" * 80)
    log(f"Input: {input_path}")
    log(f"Policy: {policy_path}")
    log(f"Rescue candidates: {rescue_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy artifact não encontrado: {policy_path}")
    if not rescue_path.exists():
        raise FileNotFoundError(f"Rescue candidates não encontrado: {rescue_path}")

    df = add_bins_and_guards(normalize_columns(pd.read_csv(input_path, low_memory=False)))
    policy = load_json(policy_path)
    rescue_df = pd.read_csv(rescue_path)

    scenario_name = args.scenario or policy.get("recommended_scenario") or EXPECTED["scenario"]
    contract = make_contract(df, policy, rescue_df, scenario_name, args.base_col)
    dump_json(contract, output_dir / "01_input_contract.json")

    if not contract["contract_ok"]:
        summary = {
            "experiment": "EXP-014B-R3J-FROZEN",
            "status": "DONE",
            "objective_status": "FAIL_CONTRACT_NOT_OK",
            "contract": contract,
            "input_path": str(input_path),
            "policy_path": str(policy_path),
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
            "output_dir": str(output_dir),
        }
        dump_json(summary, output_dir / "00_run_summary.json")
        log(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    y = df["is_fraud"].to_numpy(dtype=int)
    base_pred = pd.to_numeric(df[args.base_col], errors="coerce").fillna(0).astype(int).to_numpy()
    base_metrics = compute_metrics(y, base_pred)
    pd.DataFrame([{"policy_name": "R3H_FROZEN_BASE", **base_metrics}]).to_csv(output_dir / "02_base_metrics.csv", index=False)

    scenario = policy["scenario_artifacts"][scenario_name]
    selected_ids = [str(x) for x in scenario.get("selected_rescue_candidate_ids", [])]
    selected_rules = scenario.get("selected_retighten_rules", [])

    rescue_pred, rescue_impact = apply_rescues(df, base_pred, rescue_df, selected_ids)
    df[RESCUE_COL] = rescue_pred.astype(int)
    rescue_metrics = compute_metrics(y, rescue_pred)
    rescue_impact.to_csv(output_dir / "03_rescue_replay_impact.csv", index=False)

    final_pred, retighten_impact = apply_retighten(df, rescue_pred, base_pred, selected_rules)
    df[FINAL_COL] = final_pred.astype(int)
    final_metrics = compute_metrics(y, final_pred)
    retighten_impact.to_csv(output_dir / "04_retightening_replay_impact.csv", index=False)

    rescue_fn_recovered = base_metrics["fn"] - rescue_metrics["fn"]
    rescue_fp_added = rescue_metrics["fp"] - base_metrics["fp"]
    retightening_fp_removed = rescue_metrics["fp"] - final_metrics["fp"]
    retightening_tp_loss = rescue_metrics["tp"] - final_metrics["tp"]

    metrics_df = pd.DataFrame([
        {"policy_name": "R3H_FROZEN_BASE", **base_metrics},
        {"policy_name": "R3J_FROZEN_RESCUE_BEFORE_RETIGHTEN", **rescue_metrics},
        {"policy_name": "EXP014B_R3J_FROZEN_FINAL", **final_metrics},
    ])
    metrics_df.to_csv(output_dir / "05_frozen_metrics.csv", index=False)

    wl, wh = wilson_ci(final_metrics["tp"], int(y.sum()))
    wilson_df = pd.DataFrame([{
        "policy_name": "EXP014B_R3J_FROZEN_FINAL",
        "tp": final_metrics["tp"],
        "n_frauds": int(y.sum()),
        "recall": final_metrics["recall"],
        "wilson_low": wl,
        "wilson_high": wh,
        "wilson_low_min": args.wilson_low_min,
        "wilson_pass": bool(wl >= args.wilson_low_min),
    }])
    wilson_df.to_csv(output_dir / "06_wilson_recall_ci.csv", index=False)

    df[(df["is_fraud"] == 1) & (df[FINAL_COL] == 0)].to_csv(output_dir / "07_false_negatives.csv", index=False)
    fp_df = df[(df["is_fraud"] == 0) & (df[FINAL_COL] == 1)].copy()
    if len(fp_df) > 5000:
        fp_df = fp_df.sample(5000, random_state=42)
    fp_df.to_csv(output_dir / "08_false_positives_sample.csv", index=False)

    expected_metrics_matched = (
        final_metrics["tp"] == int(args.expected_tp)
        and final_metrics["fp"] == int(args.expected_fp)
        and final_metrics["fn"] == int(args.expected_fn)
    )
    rescue_expected_matched = (
        int(rescue_fn_recovered) == int(args.expected_rescue_fn_recovered)
        and int(rescue_fp_added) == int(args.expected_rescue_fp_added)
    )
    retighten_expected_matched = (
        int(retightening_fp_removed) == int(args.expected_retightening_fp_removed)
        and int(retightening_tp_loss) == int(args.expected_retightening_tp_loss)
    )
    schema_ok = bool(contract["contract_ok"])
    rescue_replay_ok = rescue_impact["missing_columns"].fillna("").astype(str).eq("").all() if not rescue_impact.empty else False
    retighten_replay_ok = retighten_impact["missing_columns"].fillna("").astype(str).eq("").all() if not retighten_impact.empty else False
    wilson_pass = bool(wl >= args.wilson_low_min)

    # Per-rule FP may overlap, but this frozen replay is sequential from artifact.
    # We require aggregate metrics and missing-column checks, not standalone equality for every rule.
    replay_ok = bool(rescue_replay_ok and retighten_replay_ok and rescue_expected_matched and retighten_expected_matched)
    all_pass = bool(schema_ok and expected_metrics_matched and replay_ok and wilson_pass)

    objective_status = "PASS_R3J_FROZEN_VALIDATED" if all_pass else "FAIL_R3J_FROZEN_DIVERGENCE"
    objective_status += "_METRICS_MATCH" if expected_metrics_matched else "_METRICS_MISMATCH"
    objective_status += "_RESCUE_MATCH" if rescue_expected_matched else "_RESCUE_MISMATCH"
    objective_status += "_RETIGHTEN_MATCH" if retighten_expected_matched else "_RETIGHTEN_MISMATCH"
    objective_status += "_WILSON_PASS" if wilson_pass else "_WILSON_FAIL"

    replay_artifact = {
        "experiment": "EXP-014B-R3J-FROZEN",
        "policy_name": "r3j_rescue100_frozen_replay_validation",
        "scenario_name": scenario_name,
        "source_policy_artifact": str(policy_path),
        "source_rescue_candidates": str(rescue_path),
        "input_path": str(input_path),
        "base_col": args.base_col,
        "rescue_col": RESCUE_COL,
        "final_col": FINAL_COL,
        "base_metrics": base_metrics,
        "rescue_metrics_before_retighten": rescue_metrics,
        "final_metrics": final_metrics,
        "rescue_fn_recovered_vs_base": int(rescue_fn_recovered),
        "rescue_fp_added_vs_base": int(rescue_fp_added),
        "retightening_fp_removed": int(retightening_fp_removed),
        "retightening_tp_loss": int(retightening_tp_loss),
        "wilson_low": wl,
        "wilson_high": wh,
        "checks": {
            "schema_ok": schema_ok,
            "expected_metrics_matched": expected_metrics_matched,
            "rescue_expected_matched": rescue_expected_matched,
            "retighten_expected_matched": retighten_expected_matched,
            "rescue_replay_ok": bool(rescue_replay_ok),
            "retighten_replay_ok": bool(retighten_replay_ok),
            "wilson_pass": wilson_pass,
            "all_pass": all_pass,
        },
        "selected_rescue_candidate_ids": selected_ids,
        "selected_retighten_rules": selected_rules,
        "notes": [
            "Frozen validation only: no rescue mining, no re-tightening search.",
            "R3J_RESCUE100 is a promotion candidate only if all checks pass.",
            "Next step after PASS: decide between consolidating R3J as benchmark or running one more short FN-first/FP-second microevolution."
        ],
    }
    dump_json(replay_artifact, output_dir / "09_policy_replay_artifact.json")

    if not args.no_write_predictions:
        df.to_csv(output_dir / "10_predictions.csv", index=False)

    summary = {
        "experiment": "EXP-014B-R3J-FROZEN",
        "status": "DONE",
        "objective_status": objective_status,
        "scenario_name": scenario_name,
        "input_path": str(input_path),
        "policy_path": str(policy_path),
        "rescue_candidates_path": str(rescue_path),
        "n_rows": int(len(df)),
        "n_frauds": int(df["is_fraud"].sum()),
        "base_metrics": base_metrics,
        "rescue_metrics_before_retighten": rescue_metrics,
        "final_metrics": final_metrics,
        "rescue_fn_recovered_vs_base": int(rescue_fn_recovered),
        "rescue_fp_added_vs_base": int(rescue_fp_added),
        "retightening_fp_removed": int(retightening_fp_removed),
        "retightening_tp_loss": int(retightening_tp_loss),
        "wilson_recall_low": wl,
        "wilson_recall_high": wh,
        "schema_ok": schema_ok,
        "expected_metrics_matched": expected_metrics_matched,
        "rescue_expected_matched": rescue_expected_matched,
        "retighten_expected_matched": retighten_expected_matched,
        "replay_ok": replay_ok,
        "wilson_pass": wilson_pass,
        "all_pass": all_pass,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, rescue_impact, retighten_impact, metrics_df)
    (output_dir / "11_exp014b_r3j_frozen_report.md").write_text(report, encoding="utf-8")

    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_input_contract.json",
        output_dir / "03_rescue_replay_impact.csv",
        output_dir / "04_retightening_replay_impact.csv",
        output_dir / "05_frozen_metrics.csv",
        output_dir / "06_wilson_recall_ci.csv",
        output_dir / "09_policy_replay_artifact.json",
        output_dir / "11_exp014b_r3j_frozen_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
