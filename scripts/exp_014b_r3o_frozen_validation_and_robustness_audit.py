#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
EXP-014B-R3O-FROZEN - Frozen Validation + Robustness Audit

Objetivo:
  1) Reaplicar o artifact recomendado do EXP-014B-R3O sem nova mineracao.
  2) Confirmar que o replay congelado reproduz TP=1465, FP=4252, FN=0.
  3) Comparar o replay com a coluna exp014b_r3o_recommended_pred, quando existir.
  4) Auditar robustez por tempo, splits e segmentos.
  5) Classificar regras por estabilidade e gerar variante conservadora stable-only.

Uso recomendado:
  python scripts/exp_014b_r3o_frozen_validation_and_robustness_audit.py

Uso com parametros explicitos:
  python scripts/exp_014b_r3o_frozen_validation_and_robustness_audit.py ^
    --input resultados\experimentos\EXP-014B-R3O\09_predictions_recommended.csv ^
    --artifact resultados\experimentos\EXP-014B-R3O\08_policy_artifact_recommended.json ^
    --output-dir resultados\experimentos\EXP-014B-R3O-FROZEN

Saidas:
  resultados/experimentos/EXP-014B-R3O-FROZEN/
    00_run_summary.json
    01_input_contract.json
    02_frozen_validation.json
    03_frozen_metrics.csv
    04_rule_replay_impact.csv
    05_robustness_by_segment.csv
    06_rule_stability_audit.csv
    07_conservative_stable_rules.csv
    08_policy_artifact_frozen.json
    09_predictions_frozen.csv
    10_exp014b_r3o_frozen_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "dados").exists() else Path.cwd()

DEFAULT_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3O" / "09_predictions_recommended.csv"
DEFAULT_ARTIFACT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3O" / "08_policy_artifact_recommended.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3O-FROZEN"

R3N_BASE_COL = "exp014b_r3n_frozen_pred"
R3O_EXISTING_COL = "exp014b_r3o_recommended_pred"
R3O_FROZEN_COL = "exp014b_r3o_frozen_pred"
R3O_CONSERVATIVE_COL = "exp014b_r3o_conservative_stable_pred"

SEGMENT_COLS = [
    "temporal_split",
    "event_month",
    "ds_tipo_chave_norm",
    "value_band",
    "mbk_available_flag",
    "sample_strategy",
    "source_dataset",
    "periodo_dia",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().split(".")[-1] for c in df.columns]

    if "transaction_id" not in df.columns and "cd_pix" in df.columns:
        df["transaction_id"] = df["cd_pix"]
    if "event_datetime" not in df.columns and "dt_pix" in df.columns:
        df["event_datetime"] = df["dt_pix"]

    if "is_fraud" not in df.columns:
        raise RuntimeError("Coluna obrigatoria ausente: is_fraud")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)

    for c in [R3N_BASE_COL, R3O_EXISTING_COL, R3O_FROZEN_COL, R3O_CONSERVATIVE_COL]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "event_datetime" in df.columns:
        dt = pd.to_datetime(df["event_datetime"], errors="coerce")
        df["event_datetime"] = dt
        df["event_month"] = dt.dt.strftime("%Y-%m").fillna("MISSING")
    elif "dt_pix" in df.columns:
        dt = pd.to_datetime(df["dt_pix"], errors="coerce")
        df["event_datetime"] = dt
        df["event_month"] = dt.dt.strftime("%Y-%m").fillna("MISSING")
    else:
        df["event_month"] = "MISSING"

    return df.reset_index(drop=True)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
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


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n) + (z * z / (4 * n * n))) ** 0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def parse_params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    return json.loads(str(raw).replace("Infinity", "1e999"))


def num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def rule_mask(df: pd.DataFrame, current_pred: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    rtype = params.get("type")

    if rtype in ["numeric_headroom", "numeric_retighten"]:
        col = params.get("col")
        if col not in df.columns:
            return np.zeros(len(df), dtype=bool)
        vals = num(df, col).to_numpy(dtype=float)
        cut = float(params.get("cut"))
        direction = params.get("direction")
        mask = vals >= cut if direction == "ge" else vals <= cut

    elif rtype in ["combo_headroom", "combo_retighten"]:
        mask = np.ones(len(df), dtype=bool)
        for col, val in zip(params.get("combo_cols", []), params.get("combo_values", [])):
            if col not in df.columns:
                return np.zeros(len(df), dtype=bool)
            mask &= (df[col].astype("string").fillna("<MISSING>").astype(str).to_numpy() == str(val))

    else:
        return np.zeros(len(df), dtype=bool)

    if params.get("require_module_quiet", False):
        if "module_quiet" not in df.columns:
            return np.zeros(len(df), dtype=bool)
        mask &= df["module_quiet"].astype(str).to_numpy() == "module_quiet"

    mask &= current_pred.astype(int) == 1
    return mask


def apply_fp_rules(df: pd.DataFrame, base_pred: np.ndarray, rules: list[dict[str, Any]], y: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    current = base_pred.copy().astype(int)
    rows: list[dict[str, Any]] = []

    for i, rule in enumerate(rules):
        params = parse_params(rule.get("params_json") or rule.get("params") or "{}")
        mask = rule_mask(df, current, params)
        tp_loss = int(((y == 1) & mask).sum())
        fp_removed = int(((y == 0) & mask).sum())
        n_effect = int(mask.sum())
        current[mask] = 0
        m = metrics(y, current)
        rows.append({
            "rule_index": i,
            "rule_id": rule.get("rule_id") or f"rule_{i:03d}",
            "family": rule.get("family"),
            "description": rule.get("description"),
            "tp_loss": tp_loss,
            "fp_removed": fp_removed,
            "n_effect": n_effect,
            "cumulative_fp_removed": int(sum(r["fp_removed"] for r in rows) + fp_removed),
            "params_json": json.dumps(params, ensure_ascii=False),
            **m,
        })

    return current, pd.DataFrame(rows)


def segment_metrics(df: pd.DataFrame, y: np.ndarray, base_pred: np.ndarray, final_pred: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for col in SEGMENT_COLS:
        if col not in df.columns:
            continue
        s = df[col].astype("string").fillna("MISSING").astype(str)
        for val, idx in s.groupby(s, dropna=False).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            if len(idx) == 0:
                continue
            y_sub = y[idx]
            b = base_pred[idx]
            f = final_pred[idx]
            base_m = metrics(y_sub, b)
            final_m = metrics(y_sub, f)
            rows.append({
                "segment_col": col,
                "segment_value": str(val),
                "n_rows": int(len(idx)),
                "n_frauds": int(y_sub.sum()),
                "base_tp": base_m["tp"],
                "base_fp": base_m["fp"],
                "base_fn": base_m["fn"],
                "final_tp": final_m["tp"],
                "final_fp": final_m["fp"],
                "final_fn": final_m["fn"],
                "fp_removed": int(base_m["fp"] - final_m["fp"]),
                "tp_loss": int(base_m["tp"] - final_m["tp"]),
                "fn_delta": int(final_m["fn"] - base_m["fn"]),
                "final_precision": final_m["precision"],
                "final_recall": final_m["recall"],
                "final_fpr": final_m["fpr"],
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["segment_col", "fp_removed"], ascending=[True, False]).reset_index(drop=True)
    return out


def classify_rule_stability(df: pd.DataFrame, y: np.ndarray, base_pred: np.ndarray, rules: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    current = base_pred.copy().astype(int)
    rows: list[dict[str, Any]] = []
    stable_rules: list[dict[str, Any]] = []

    for i, rule in enumerate(rules):
        params = parse_params(rule.get("params_json") or rule.get("params") or "{}")
        mask = rule_mask(df, current, params)
        tp_loss = int(((y == 1) & mask).sum())
        fp_removed = int(((y == 0) & mask).sum())

        removed_fp_mask = mask & (y == 0)

        def nonzero_segments(col: str) -> int:
            if col not in df.columns or not removed_fp_mask.any():
                return 0
            vals = df.loc[removed_fp_mask, col].astype("string").fillna("MISSING").astype(str)
            return int(vals.nunique(dropna=False))

        n_temporal_splits = nonzero_segments("temporal_split")
        n_months = nonzero_segments("event_month")
        n_key_types = nonzero_segments("ds_tipo_chave_norm")
        n_value_bands = nonzero_segments("value_band")

        if tp_loss > 0:
            cls = "REJECT_TP_LOSS"
        elif fp_removed <= 0:
            cls = "NO_EFFECT_IN_REPLAY"
        elif n_months >= 2 and n_temporal_splits >= 2:
            cls = "STABLE"
        elif n_months >= 2 and n_temporal_splits >= 1 and fp_removed >= 25:
            cls = "KEEP_WITH_CAUTION"
        elif fp_removed >= 30 and n_months >= 1:
            cls = "LOW_SUPPORT_BUT_MATERIAL"
        else:
            cls = "LOW_SUPPORT"

        row = {
            "rule_index": i,
            "rule_id": rule.get("rule_id") or f"rule_{i:03d}",
            "family": rule.get("family"),
            "description": rule.get("description"),
            "tp_loss": tp_loss,
            "fp_removed": fp_removed,
            "n_effect": int(mask.sum()),
            "n_temporal_splits_with_fp_removed": n_temporal_splits,
            "n_months_with_fp_removed": n_months,
            "n_key_types_with_fp_removed": n_key_types,
            "n_value_bands_with_fp_removed": n_value_bands,
            "stability_class": cls,
            "params_json": json.dumps(params, ensure_ascii=False),
        }
        rows.append(row)

        if cls == "STABLE":
            stable_rules.append(rule)

        current[mask] = 0

    return pd.DataFrame(rows), stable_rules


def approx_equal_metrics(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for k in ["tp", "fp", "fn", "tn"]:
        if int(actual.get(k, -999999)) != int(expected.get(k, -999998)):
            return False
    return True


def make_report(summary: dict[str, Any], validation: dict[str, Any], rule_impact: pd.DataFrame, seg: pd.DataFrame, stability: pd.DataFrame, stable_rules: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# EXP-014B-R3O-FROZEN - Frozen Validation + Robustness Audit")
    lines.append("")
    lines.append("## Resultado executivo")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- All pass: `{summary['all_pass']}`")
    lines.append(f"- Metricas R3N base: `{summary['base_r3n_metrics']}`")
    lines.append(f"- Metricas R3O frozen: `{summary['frozen_metrics']}`")
    lines.append(f"- FP removidos vs R3N: `{summary['fp_removed_vs_r3n']}`")
    lines.append(f"- TP loss vs R3N: `{summary['tp_loss_vs_r3n']}`")
    lines.append(f"- FN delta vs R3N: `{summary['fn_delta_vs_r3n']}`")
    lines.append(f"- Mismatches vs coluna R3O existente: `{summary['prediction_mismatches_vs_existing']}`")
    lines.append("")
    lines.append("## Validacao congelada")
    lines.append(f"```json\n{json.dumps(validation, ensure_ascii=False, indent=2)}\n```")
    lines.append("")
    lines.append("## Regras reaplicadas")
    if rule_impact.empty:
        lines.append("Nenhuma regra reaplicada.")
    else:
        show = ["rule_index", "rule_id", "fp_removed", "tp_loss", "tp", "fp", "fn", "precision", "recall", "fpr", "description"]
        lines.append(rule_impact[[c for c in show if c in rule_impact.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Auditoria de estabilidade das regras")
    if stability.empty:
        lines.append("Auditoria vazia.")
    else:
        show = ["rule_id", "fp_removed", "tp_loss", "n_temporal_splits_with_fp_removed", "n_months_with_fp_removed", "stability_class", "description"]
        lines.append(stability[[c for c in show if c in stability.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Variante conservadora stable-only")
    if stable_rules.empty:
        lines.append("Nenhuma regra STABLE selecionada para variante conservadora.")
    else:
        show = ["rule_id", "fp_removed", "stability_class", "description"]
        lines.append(stable_rules[[c for c in show if c in stable_rules.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Robustez por segmento - piores/mais relevantes")
    if seg.empty:
        lines.append("Sem segmentos auditados.")
    else:
        show = ["segment_col", "segment_value", "n_rows", "n_frauds", "fp_removed", "tp_loss", "fn_delta", "final_tp", "final_fp", "final_fn", "final_recall"]
        focus = seg.sort_values(["tp_loss", "fn_delta", "fp_removed"], ascending=[False, False, False]).head(80)
        lines.append(focus[[c for c in show if c in focus.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Decisao sugerida")
    if summary.get("all_pass"):
        lines.append("R3O-FROZEN validado. O R3O pode substituir o R3N como benchmark congelado principal. A proxima rodada deve ser R3P FP-only conservadora sobre a base R3O-FROZEN, com regras mais restritas e auditoria temporal por regra.")
    else:
        lines.append("Nao promover. Corrigir divergencia de replay/artifact antes de qualquer nova mineracao.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--no-write-predictions", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    artifact_path = Path(args.artifact)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014B-R3O-FROZEN - Frozen Validation + Robustness Audit")
    log("=" * 80)

    if not input_path.exists():
        raise FileNotFoundError(f"input nao encontrado: {input_path}")
    if not artifact_path.exists():
        raise FileNotFoundError(f"artifact nao encontrado: {artifact_path}")

    df = normalize(pd.read_csv(input_path, low_memory=False))
    artifact = load_json(artifact_path)
    y = df["is_fraud"].to_numpy(dtype=int)

    rules = artifact.get("selected_fp_rules", [])
    expected_base = artifact.get("base_r3n_frozen_metrics", {})
    expected_final = artifact.get("recommended_metrics", {})

    missing = []
    for c in ["is_fraud", R3N_BASE_COL]:
        if c not in df.columns:
            missing.append(c)
    if not rules:
        missing.append("artifact.selected_fp_rules")

    contract = {
        "n_rows": int(len(df)),
        "n_frauds": int(df["is_fraud"].sum()) if "is_fraud" in df.columns else None,
        "input_path": str(input_path),
        "artifact_path": str(artifact_path),
        "base_col": R3N_BASE_COL,
        "existing_r3o_col_present": bool(R3O_EXISTING_COL in df.columns),
        "n_selected_rules_in_artifact": int(len(rules)),
        "missing": missing,
        "contract_ok": not missing,
    }
    dump_json(contract, out / "01_input_contract.json")
    if missing:
        raise RuntimeError(f"Contrato falhou: {missing}")

    base_pred = df[R3N_BASE_COL].to_numpy(dtype=int)
    base_m = metrics(y, base_pred)

    frozen_pred, rule_impact = apply_fp_rules(df, base_pred, rules, y)
    frozen_m = metrics(y, frozen_pred)
    df[R3O_FROZEN_COL] = frozen_pred.astype(int)

    existing_m = None
    mismatches = None
    if R3O_EXISTING_COL in df.columns:
        existing_pred = df[R3O_EXISTING_COL].to_numpy(dtype=int)
        existing_m = metrics(y, existing_pred)
        mismatches = int((existing_pred != frozen_pred).sum())
    else:
        mismatches = -1

    fp_removed_vs_r3n = int(base_m["fp"] - frozen_m["fp"])
    tp_loss_vs_r3n = int(base_m["tp"] - frozen_m["tp"])
    fn_delta_vs_r3n = int(frozen_m["fn"] - base_m["fn"])
    wl, wh = wilson(frozen_m["tp"], int(y.sum()))

    stability, stable_rules_list = classify_rule_stability(df, y, base_pred, rules)
    stable_rule_ids = set(stability.loc[stability["stability_class"] == "STABLE", "rule_id"].astype(str)) if not stability.empty else set()
    stable_rules = [r for r in rules if str(r.get("rule_id")) in stable_rule_ids]

    conservative_pred, conservative_impact = apply_fp_rules(df, base_pred, stable_rules, y) if stable_rules else (base_pred.copy(), pd.DataFrame())
    conservative_m = metrics(y, conservative_pred)
    df[R3O_CONSERVATIVE_COL] = conservative_pred.astype(int)

    seg = segment_metrics(df, y, base_pred, frozen_pred)

    base_match = approx_equal_metrics(base_m, expected_base)
    final_match = approx_equal_metrics(frozen_m, expected_final)
    existing_match = bool(mismatches == 0) if R3O_EXISTING_COL in df.columns else True
    fn_zero_preserved = bool(frozen_m["fn"] == 0 and tp_loss_vs_r3n == 0)
    fp_reduced = bool(fp_removed_vs_r3n > 0)
    rule_tp_loss_zero = bool(rule_impact.empty or int(rule_impact["tp_loss"].sum()) == 0)
    segment_tp_loss_zero = bool(seg.empty or int(seg["tp_loss"].max()) == 0)
    segment_fn_delta_zero = bool(seg.empty or int(seg["fn_delta"].max()) == 0)

    validation = {
        "expected_base_metrics": expected_base,
        "actual_base_metrics": base_m,
        "expected_final_metrics": expected_final,
        "actual_frozen_metrics": frozen_m,
        "existing_r3o_metrics": existing_m,
        "prediction_mismatches_vs_existing": mismatches,
        "fp_removed_vs_r3n": fp_removed_vs_r3n,
        "tp_loss_vs_r3n": tp_loss_vs_r3n,
        "fn_delta_vs_r3n": fn_delta_vs_r3n,
        "wilson_low": wl,
        "wilson_high": wh,
        "base_metrics_match_artifact": base_match,
        "final_metrics_match_artifact": final_match,
        "existing_prediction_match": existing_match,
        "fn_zero_preserved": fn_zero_preserved,
        "fp_reduced": fp_reduced,
        "rule_tp_loss_zero": rule_tp_loss_zero,
        "segment_tp_loss_zero": segment_tp_loss_zero,
        "segment_fn_delta_zero": segment_fn_delta_zero,
    }

    all_pass = bool(
        base_match
        and final_match
        and existing_match
        and fn_zero_preserved
        and fp_reduced
        and rule_tp_loss_zero
        and segment_tp_loss_zero
        and segment_fn_delta_zero
    )

    status = "PASS_R3O_FROZEN_VALIDATED" if all_pass else "FAIL_R3O_FROZEN_DIVERGENCE"
    validation["all_pass"] = all_pass
    validation["status"] = status
    dump_json(validation, out / "02_frozen_validation.json")

    metrics_rows = [
        {"policy_name": "R3N_FROZEN_BASE", **base_m},
        {"policy_name": "R3O_FROZEN_REPLAY", **frozen_m},
        {"policy_name": "R3O_CONSERVATIVE_STABLE_ONLY", **conservative_m},
    ]
    if existing_m is not None:
        metrics_rows.append({"policy_name": "R3O_EXISTING_COLUMN", **existing_m})
    pd.DataFrame(metrics_rows).to_csv(out / "03_frozen_metrics.csv", index=False)
    rule_impact.to_csv(out / "04_rule_replay_impact.csv", index=False)
    seg.to_csv(out / "05_robustness_by_segment.csv", index=False)
    stability.to_csv(out / "06_rule_stability_audit.csv", index=False)
    stable_rules_df = stability[stability["stability_class"] == "STABLE"].copy() if not stability.empty else pd.DataFrame()
    stable_rules_df.to_csv(out / "07_conservative_stable_rules.csv", index=False)

    objective_status = status
    if all_pass:
        objective_status += "_FN_ZERO_PRESERVED_FP_REDUCED"
    else:
        objective_status += "_DO_NOT_PROMOTE"

    frozen_artifact = {
        "experiment": "EXP-014B-R3O-FROZEN",
        "policy_name": "r3o_frozen_zero_fn_fp_only_reducer",
        "objective_status": objective_status,
        "source_artifact": str(artifact_path),
        "input_path": str(input_path),
        "base_col": R3N_BASE_COL,
        "frozen_pred_col": R3O_FROZEN_COL,
        "base_r3n_metrics": base_m,
        "frozen_metrics": frozen_m,
        "conservative_stable_only_metrics": conservative_m,
        "fp_removed_vs_r3n": fp_removed_vs_r3n,
        "tp_loss_vs_r3n": tp_loss_vs_r3n,
        "fn_delta_vs_r3n": fn_delta_vs_r3n,
        "wilson_low": wl,
        "wilson_high": wh,
        "validation": validation,
        "selected_fp_rules": rules,
        "stable_rule_ids": sorted(stable_rule_ids),
        "n_rules": int(len(rules)),
        "n_stable_rules": int(len(stable_rules)),
        "notes": [
            "Frozen validation reapplies only selected_fp_rules from EXP-014B-R3O artifact.",
            "No new mining, no rescues, no threshold changes.",
            "Promotion requires exact replay, FN=0, TP loss=0, and zero mismatch vs existing R3O prediction when available.",
        ],
    }
    dump_json(frozen_artifact, out / "08_policy_artifact_frozen.json")

    if not args.no_write_predictions:
        df.to_csv(out / "09_predictions_frozen.csv", index=False)

    summary = {
        "experiment": "EXP-014B-R3O-FROZEN",
        "status": "DONE",
        "objective_status": objective_status,
        "n_rows": int(len(df)),
        "n_frauds": int(y.sum()),
        "input_path": str(input_path),
        "artifact_path": str(artifact_path),
        "base_r3n_metrics": base_m,
        "frozen_metrics": frozen_m,
        "conservative_stable_only_metrics": conservative_m,
        "fp_removed_vs_r3n": fp_removed_vs_r3n,
        "tp_loss_vs_r3n": tp_loss_vs_r3n,
        "fn_delta_vs_r3n": fn_delta_vs_r3n,
        "prediction_mismatches_vs_existing": mismatches,
        "n_rules": int(len(rules)),
        "n_stable_rules": int(len(stable_rules)),
        "wilson_low": wl,
        "wilson_high": wh,
        "validation_status": status,
        "all_pass": all_pass,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(out),
    }
    dump_json(summary, out / "00_run_summary.json")

    report = make_report(summary, validation, rule_impact, seg, stability, stable_rules_df)
    (out / "10_exp014b_r3o_frozen_report.md").write_text(report, encoding="utf-8")

    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        out / "00_run_summary.json",
        out / "01_input_contract.json",
        out / "02_frozen_validation.json",
        out / "03_frozen_metrics.csv",
        out / "04_rule_replay_impact.csv",
        out / "05_robustness_by_segment.csv",
        out / "06_rule_stability_audit.csv",
        out / "08_policy_artifact_frozen.json",
        out / "10_exp014b_r3o_frozen_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
