#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-013J — Safe + One-TP Micro-Veto Consolidation

Objetivo:
  Partir da política congelada EXP-013H/EXP-013G:
      TP=118, FP=414, FN=6, recall=95.16%

  E testar duas trilhas de microevolução:

  A) STRICT_RECALL95_SAFE_ONLY
     - Não perde nenhum TP adicional.
     - Mantém TP>=118 e recall>=95%.
     - Usa candidatos TP_loss=0 do EXP-013I.

  B) ONE_TP_EXCEPTION_HIGH_RETURN
     - Permite perder no máximo 1 TP adicional.
     - Isso derruba recall para ~94.35% se TP=117/124.
     - Só deve ser considerado se a redução de FP for muito alta.
     - Usa candidatos TP_loss=0 + TP_loss=1 do EXP-013I.
     - Esta trilha é EXCEÇÃO executiva, não substitui a meta técnica de recall>=95%.

Motivo:
  O EXP-013I encontrou:
    - 92 candidatos de veto seguro TP_loss=0;
    - 165 candidatos near-safe TP_loss=1;
    - 78 candidatos de imunidade FN.

Observação importante:
  "165 candidatos near-safe" é quantidade de regras candidatas, não 165 FPs garantidos.
  Este script calcula o ganho real considerando sobreposição entre regras.

Entradas default:
  resultados/experimentos/EXP-013H/05_frozen_predictions.csv
  resultados/experimentos/EXP-013I/06_safe_veto_candidates_tp0.csv
  resultados/experimentos/EXP-013I/07_near_safe_veto_candidates_tp1.csv

Uso:
  python scripts/exp_013j_safe_and_onetp_microveto.py

Execução mais profunda:
  python scripts/exp_013j_safe_and_onetp_microveto.py --max-candidates 250 --beam-width 200 --max-depth 8

Saídas:
  resultados/experimentos/EXP-013J/
    00_run_summary.json
    01_base_metrics.csv
    02_candidates_recomputed.csv
    03_frontier_by_scenario.csv
    04_scenario_metrics.csv
    05_selected_rules_by_scenario.csv
    06_predictions_by_scenario.csv
    07_false_negatives_by_scenario.csv
    08_false_positives_by_scenario.csv
    09_time_block_metrics.csv
    10_bootstrap_confidence_intervals.csv
    11_policy_artifacts.json
    12_exp013j_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
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
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "backend").exists() else Path.cwd()

DEFAULT_PREDICTIONS = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013H" / "05_frozen_predictions.csv"
DEFAULT_SAFE = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013I" / "06_safe_veto_candidates_tp0.csv"
DEFAULT_NEAR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013I" / "07_near_safe_veto_candidates_tp1.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013J"

PRED_COL_CANDIDATES = ["exp013h_frozen_pred", "exp013g_micro_pred", "pred_HIGH_RECALL_95"]


@dataclass
class Rule:
    rule_id: str
    source: str
    family: str
    description: str
    mask: np.ndarray
    tp_loss: int
    fp_removed: int
    params: dict[str, Any]


@dataclass
class State:
    mask: np.ndarray
    rule_indices: tuple[int, ...]
    tp_loss: int
    fp_removed: int


def log(msg: str) -> None:
    print(msg, flush=True)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().split(".")[-1] for c in df.columns]

    if "transaction_id" not in df.columns and "cd_pix" in df.columns:
        df["transaction_id"] = df["cd_pix"]
    if "event_datetime" not in df.columns and "dt_pix" in df.columns:
        df["event_datetime"] = df["dt_pix"]

    if "is_fraud" not in df.columns:
        raise RuntimeError("Coluna is_fraud ausente.")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)

    for c in PRED_COL_CANDIDATES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "transaction_id" in df.columns:
        df["transaction_id"] = df["transaction_id"].astype("string").str.strip()

    for c in ["event_datetime", "data_pix"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    return df.reset_index(drop=True)


def pick_pred_col(df: pd.DataFrame, requested: str | None) -> str:
    if requested and requested in df.columns:
        return requested

    for c in PRED_COL_CANDIDATES:
        if c in df.columns:
            return c

    raise RuntimeError("Não encontrei coluna de predição. Use --pred-col.")


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


def text(df: pd.DataFrame, names: str | list[str], default: str = "<MISSING>") -> pd.Series:
    col = pick_col(df, names)
    if col is None:
        return pd.Series([default] * len(df), index=df.index, dtype="string")
    return df[col].astype("string").fillna(default).astype(str)


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


def parse_params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    try:
        return json.loads(str(raw))
    except Exception:
        return {}


def sanitize_id(text_value: str, max_len: int = 110) -> str:
    t = re.sub(r"[^A-Za-z0-9_]+", "_", str(text_value))
    t = re.sub(r"_+", "_", t).strip("_")
    return t[:max_len] or "rule"


def lgbm_series(df: pd.DataFrame) -> pd.Series:
    return num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0)


def build_mask_from_candidate(df: pd.DataFrame, row: pd.Series) -> np.ndarray:
    family = str(row.get("family", ""))
    desc = str(row.get("description", ""))
    params = parse_params(row.get("params_json"))

    mask = np.ones(len(df), dtype=bool)

    if family in {"segment_veto", "segment_lgbm_veto"}:
        cols = params.get("segment_cols", [])
        vals = params.get("segment_values", [])

        if not cols:
            # Fallback parse from description: "a=b AND c=d AND lgbm<0.02"
            parts = [p.strip() for p in desc.split(" AND ")]
            cols, vals = [], []
            for p in parts:
                if "=" in p and not p.startswith("lgbm"):
                    c, v = p.split("=", 1)
                    cols.append(c.strip())
                    vals.append(v.strip())

        for c, v in zip(cols, vals):
            if c not in df.columns:
                return np.zeros(len(df), dtype=bool)
            mask = mask & (text(df, c) == str(v)).to_numpy(dtype=bool)

        lgbm_lt = params.get("lgbm_lt")
        if lgbm_lt is None and "lgbm<" in desc:
            try:
                lgbm_lt = float(desc.split("lgbm<", 1)[1].split()[0])
            except Exception:
                lgbm_lt = None
        if family == "segment_lgbm_veto" and lgbm_lt is not None:
            mask = mask & (lgbm_series(df) < float(lgbm_lt)).to_numpy(dtype=bool)

        return mask

    if family in {"numeric_veto", "lgbm_threshold", "lgbm_threshold_preserve", "score_final_threshold"}:
        feature = params.get("feature")
        op = params.get("op")
        threshold = params.get("threshold")

        if not feature:
            # Fallback parse "feature<th" or "feature>th"
            m = re.match(r"([A-Za-z0-9_]+)\s*([<>])\s*([0-9.eE+-]+)", desc)
            if not m:
                return np.zeros(len(df), dtype=bool)
            feature, op_symbol, threshold = m.group(1), m.group(2), float(m.group(3))
            op = "lt" if op_symbol == "<" else "gt"

        vals = num(df, feature, np.nan)
        th = float(threshold)
        if op in {"lt", "<"}:
            return (vals < th).to_numpy(dtype=bool)
        if op in {"gt", ">"}:
            return (vals > th).to_numpy(dtype=bool)
        return np.zeros(len(df), dtype=bool)

    # Unknown candidate family: ignore safely.
    return np.zeros(len(df), dtype=bool)


def load_candidates(path: Path, source_name: str, df: pd.DataFrame, base_pred: np.ndarray, y: np.ndarray, min_fp_removed: int) -> list[Rule]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de candidatos não encontrado: {path}")

    cdf = pd.read_csv(path, low_memory=False)
    if cdf.empty:
        return []

    rules = []
    for i, row in cdf.iterrows():
        mask = build_mask_from_candidate(df, row)
        mask = mask & (base_pred == 1)

        if not mask.any():
            continue

        tp_loss = int(((y == 1) & mask).sum())
        fp_removed = int(((y == 0) & mask).sum())

        if fp_removed < min_fp_removed:
            continue

        desc = str(row.get("description", f"{source_name}_{i}"))
        family = str(row.get("family", "unknown"))
        rid = sanitize_id(f"{source_name}_{i}_{family}_{desc}")
        params = parse_params(row.get("params_json"))

        rules.append(Rule(
            rule_id=rid,
            source=source_name,
            family=family,
            description=desc,
            mask=mask,
            tp_loss=tp_loss,
            fp_removed=fp_removed,
            params=params,
        ))

    return rules


def dedupe_rules(rules: list[Rule]) -> list[Rule]:
    best: dict[bytes, Rule] = {}
    for r in rules:
        key = np.packbits(r.mask).tobytes()
        old = best.get(key)
        if old is None:
            best[key] = r
        else:
            new_key = (r.fp_removed, -r.tp_loss, -len(r.description))
            old_key = (old.fp_removed, -old.tp_loss, -len(old.description))
            if new_key > old_key:
                best[key] = r

    out = list(best.values())
    out.sort(key=lambda r: (r.tp_loss, -r.fp_removed, -r.fp_removed / max(r.tp_loss, 1)))
    return out


def rules_dataframe(rules: list[Rule]) -> pd.DataFrame:
    return pd.DataFrame([{
        "rule_index": i,
        "rule_id": r.rule_id,
        "source": r.source,
        "family": r.family,
        "description": r.description,
        "tp_loss": r.tp_loss,
        "fp_removed": r.fp_removed,
        "fp_per_tp_loss": r.fp_removed / max(r.tp_loss, 1),
        "params_json": json.dumps(r.params, ensure_ascii=False),
    } for i, r in enumerate(rules)])


def search_scenario(
    scenario: str,
    rules: list[Rule],
    base_pred: np.ndarray,
    y: np.ndarray,
    max_tp_loss: int,
    max_candidates: int,
    beam_width: int,
    max_depth: int,
) -> tuple[pd.DataFrame, State, list[Rule]]:
    candidates = [r for r in rules if r.tp_loss <= max_tp_loss]
    candidates.sort(key=lambda r: (r.tp_loss == 0, r.fp_removed / max(r.tp_loss, 1), r.fp_removed, -r.tp_loss), reverse=True)
    candidates = candidates[:max_candidates]

    zero = np.zeros(len(y), dtype=bool)
    base_state = State(mask=zero, rule_indices=tuple(), tp_loss=0, fp_removed=0)
    states = [base_state]
    best = base_state
    rows = []

    log(f"  {scenario}: candidates={len(candidates)}, max_tp_loss={max_tp_loss}")

    for depth in range(1, max_depth + 1):
        next_states: dict[bytes, State] = {}

        for state in states:
            last = state.rule_indices[-1] if state.rule_indices else -1
            for ridx in range(last + 1, len(candidates)):
                r = candidates[ridx]
                new_mask = state.mask | r.mask
                if np.array_equal(new_mask, state.mask):
                    continue

                tp_loss = int(((y == 1) & new_mask).sum())
                if tp_loss > max_tp_loss:
                    continue

                fp_removed = int(((y == 0) & new_mask).sum())
                if fp_removed <= state.fp_removed:
                    continue

                key = np.packbits(new_mask).tobytes()
                ns = State(new_mask, state.rule_indices + (ridx,), tp_loss, fp_removed)
                old = next_states.get(key)
                if old is None or (ns.fp_removed, -ns.tp_loss, -len(ns.rule_indices)) > (old.fp_removed, -old.tp_loss, -len(old.rule_indices)):
                    next_states[key] = ns

        if not next_states:
            break

        states = sorted(next_states.values(), key=lambda s: (s.fp_removed, -s.tp_loss, -len(s.rule_indices)), reverse=True)[:beam_width]

        if (states[0].fp_removed, -states[0].tp_loss) > (best.fp_removed, -best.tp_loss):
            best = states[0]

        for s in states[:50]:
            pred = base_pred.copy()
            pred[s.mask] = 0
            m = compute_metrics(y, pred)
            rows.append({
                "scenario": scenario,
                "depth": depth,
                "tp_loss": s.tp_loss,
                "fp_removed": s.fp_removed,
                "n_rules": len(s.rule_indices),
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "rule_ids": "|".join(candidates[i].rule_id for i in s.rule_indices),
                "rule_descriptions": " || ".join(candidates[i].description for i in s.rule_indices),
            })

    if not rows:
        pred = base_pred.copy()
        m = compute_metrics(y, pred)
        rows = [{
            "scenario": scenario,
            "depth": 0,
            "tp_loss": 0,
            "fp_removed": 0,
            "n_rules": 0,
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "rule_ids": "",
            "rule_descriptions": "",
        }]

    frontier = pd.DataFrame(rows).sort_values(["fp", "tp"], ascending=[True, False]).reset_index(drop=True)
    return frontier, best, candidates


def make_time_blocks(df: pd.DataFrame, n_blocks: int) -> pd.Series:
    if "data_pix" in df.columns and df["data_pix"].notna().any():
        dates = pd.to_datetime(df["data_pix"], errors="coerce")
    elif "event_datetime" in df.columns and df["event_datetime"].notna().any():
        dates = pd.to_datetime(df["event_datetime"], errors="coerce")
    else:
        return pd.qcut(np.arange(len(df)), q=min(n_blocks, len(df)), labels=False, duplicates="drop").astype(int)

    tmp = pd.DataFrame({"date": dates, "_idx": np.arange(len(df))}).sort_values(["date", "_idx"])
    tmp["block"] = pd.qcut(np.arange(len(tmp)), q=min(n_blocks, len(tmp)), labels=False, duplicates="drop")
    out = pd.Series(index=tmp["_idx"].values, data=tmp["block"].values).sort_index()
    return out.astype(int)


def block_metrics(df: pd.DataFrame, pred: np.ndarray, n_blocks: int, scenario: str) -> pd.DataFrame:
    blocks = make_time_blocks(df, n_blocks)
    rows = []
    for b in sorted(blocks.dropna().unique()):
        part = df.loc[blocks == b].copy()
        pred_b = pred[blocks == b]
        m = compute_metrics(part["is_fraud"].to_numpy(dtype=int), pred_b)
        m.update({
            "scenario": scenario,
            "block": int(b),
            "n_rows": int(len(part)),
            "n_frauds": int(part["is_fraud"].sum()),
            "dt_min": str(part["data_pix"].min().date()) if "data_pix" in part.columns and part["data_pix"].notna().any() else None,
            "dt_max": str(part["data_pix"].max().date()) if "data_pix" in part.columns and part["data_pix"].notna().any() else None,
        })
        rows.append(m)
    return pd.DataFrame(rows)


def bootstrap_eval(df: pd.DataFrame, pred_col: str, iters: int, seed: int, target_recall: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n = len(df)
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        y = df.iloc[idx]["is_fraud"].to_numpy(dtype=int)
        pred = df.iloc[idx][pred_col].to_numpy(dtype=int)
        rows.append(compute_metrics(y, pred))

    boot = pd.DataFrame(rows)
    out = []
    for metric in ["tp", "fp", "fn", "precision", "recall", "f1", "fpr"]:
        vals = boot[metric].astype(float)
        out.append({
            "metric": metric,
            "mean": float(vals.mean()),
            "p025": float(vals.quantile(0.025)),
            "p050": float(vals.quantile(0.50)),
            "p975": float(vals.quantile(0.975)),
            "target_recall": target_recall if metric == "recall" else None,
            "p_below_target_recall": float((boot["recall"] < target_recall).mean()) if metric == "recall" else None,
        })
    return pd.DataFrame(out)


def make_report(summary: dict[str, Any], scenario_metrics: pd.DataFrame, selected_rules: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-013J — Safe + One-TP Micro-Veto Consolidation")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Base EXP-013H: TP={summary['base_metrics']['tp']}, FP={summary['base_metrics']['fp']}, FN={summary['base_metrics']['fn']}, recall={summary['base_metrics']['recall']}")
    lines.append("")
    lines.append("## Cenários")
    show = ["scenario", "tp", "fp", "fn", "precision", "recall", "fp_removed_vs_base", "tp_loss_vs_base", "n_rules"]
    lines.append(scenario_metrics[show].to_markdown(index=False))
    lines.append("")
    lines.append("## Regras selecionadas")
    if selected_rules.empty:
        lines.append("Nenhuma regra selecionada.")
    else:
        show_rules = ["scenario", "source", "family", "description", "tp_loss", "fp_removed"]
        lines.append(selected_rules[show_rules].to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretação")
    lines.append("A trilha STRICT_RECALL95_SAFE_ONLY é a única compatível com a meta técnica fixa de recall >=95%.")
    lines.append("A trilha ONE_TP_EXCEPTION_HIGH_RETURN é deliberadamente excepcional: se perder 1 TP, ficará abaixo de 95% neste dataset de 124 fraudes, mas pode ser analisada como trade-off executivo se a redução de FP for extraordinária.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--safe-candidates", default=str(DEFAULT_SAFE))
    parser.add_argument("--near-candidates", default=str(DEFAULT_NEAR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pred-col", default=None)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--beam-width", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--min-fp-removed", type=int, default=5)
    parser.add_argument("--bootstrap-iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-blocks", type=int, default=5)
    parser.add_argument("--one-tp-min-extra-fp", type=int, default=100, help="Threshold informativo para considerar que perder 1 TP valeu a pena.")
    args = parser.parse_args()

    t0 = time.perf_counter()
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-013J — Safe + One-TP Micro-Veto Consolidation")
    log("=" * 80)
    log(f"Predictions: {predictions_path}")
    log(f"Output: {output_dir}")

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions não encontrado: {predictions_path}")

    df = normalize_columns(pd.read_csv(predictions_path, low_memory=False))
    pred_col = pick_pred_col(df, args.pred_col)
    y = df["is_fraud"].to_numpy(dtype=int)
    base_pred = df[pred_col].to_numpy(dtype=int)

    total_frauds = int(y.sum())
    min_tp_recall95 = int(math.ceil(0.95 * total_frauds))
    base_metrics = compute_metrics(y, base_pred)

    if base_metrics["tp"] < min_tp_recall95:
        raise RuntimeError(f"Base não cumpre recall>=95%. TP={base_metrics['tp']} min={min_tp_recall95}")

    log(f"Base {pred_col}: TP={base_metrics['tp']} FP={base_metrics['fp']} FN={base_metrics['fn']} recall={base_metrics['recall']}")
    pd.DataFrame([{"pred_col": pred_col, **base_metrics}]).to_csv(output_dir / "01_base_metrics.csv", index=False)

    safe_rules = load_candidates(Path(args.safe_candidates), "safe_tp0", df, base_pred, y, args.min_fp_removed)
    near_rules = load_candidates(Path(args.near_candidates), "near_tp1", df, base_pred, y, args.min_fp_removed)
    all_rules = dedupe_rules(safe_rules + near_rules)

    rdf = rules_dataframe(all_rules)
    rdf.to_csv(output_dir / "02_candidates_recomputed.csv", index=False)

    log(f"Candidates recomputed: safe={len(safe_rules)}, near={len(near_rules)}, unique={len(all_rules)}")

    scenario_defs = [
        {
            "scenario": "STRICT_RECALL95_SAFE_ONLY",
            "max_tp_loss": 0,
            "target_recall": 0.95,
            "technical_status": "production_candidate_if_validated",
        },
        {
            "scenario": "ONE_TP_EXCEPTION_HIGH_RETURN",
            "max_tp_loss": 1,
            "target_recall": (base_metrics["tp"] - 1) / max(total_frauds, 1),
            "technical_status": "executive_tradeoff_only_below_95_if_tp_lost",
        },
    ]

    frontiers = []
    scenario_rows = []
    selected_rule_rows = []
    predictions_out = df.copy()
    block_rows = []
    boot_rows = []
    policy_artifacts = {}

    for sdef in scenario_defs:
        scenario = sdef["scenario"]
        frontier, best, candidates = search_scenario(
            scenario=scenario,
            rules=all_rules,
            base_pred=base_pred,
            y=y,
            max_tp_loss=sdef["max_tp_loss"],
            max_candidates=args.max_candidates,
            beam_width=args.beam_width,
            max_depth=args.max_depth,
        )
        frontiers.append(frontier)

        pred = base_pred.copy()
        pred[best.mask] = 0
        pred_col_out = f"pred_{scenario}"
        predictions_out[pred_col_out] = pred

        m = compute_metrics(y, pred)
        fp_removed_vs_base = base_metrics["fp"] - m["fp"]
        tp_loss_vs_base = base_metrics["tp"] - m["tp"]

        selected = [candidates[i] for i in best.rule_indices]
        for r in selected:
            selected_rule_rows.append({
                "scenario": scenario,
                "rule_id": r.rule_id,
                "source": r.source,
                "family": r.family,
                "description": r.description,
                "tp_loss": r.tp_loss,
                "fp_removed": r.fp_removed,
                "params_json": json.dumps(r.params, ensure_ascii=False),
            })

        row = {
            "scenario": scenario,
            "target_recall": sdef["target_recall"],
            "max_tp_loss_allowed": sdef["max_tp_loss"],
            "technical_status": sdef["technical_status"],
            **m,
            "fp_removed_vs_base": int(fp_removed_vs_base),
            "tp_loss_vs_base": int(tp_loss_vs_base),
            "n_rules": len(selected),
            "rule_ids": "|".join(r.rule_id for r in selected),
            "rule_descriptions": " || ".join(r.description for r in selected),
            "one_tp_exception_met_min_extra_fp": bool(scenario != "ONE_TP_EXCEPTION_HIGH_RETURN" or fp_removed_vs_base >= args.one_tp_min_extra_fp),
        }
        scenario_rows.append(row)

        bm = block_metrics(df, pred, args.time_blocks, scenario)
        block_rows.append(bm)

        boot = bootstrap_eval(pd.concat([df, pd.Series(pred, name=pred_col_out)], axis=1), pred_col_out, args.bootstrap_iters, args.seed, sdef["target_recall"])
        boot["scenario"] = scenario
        boot_rows.append(boot)

        policy_artifacts[scenario] = {
            "scenario": scenario,
            "base_pred_col": pred_col,
            "target_recall": sdef["target_recall"],
            "max_tp_loss_allowed": sdef["max_tp_loss"],
            "metrics": m,
            "fp_removed_vs_base": int(fp_removed_vs_base),
            "tp_loss_vs_base": int(tp_loss_vs_base),
            "rules": [{
                "rule_id": r.rule_id,
                "source": r.source,
                "family": r.family,
                "description": r.description,
                "tp_loss": r.tp_loss,
                "fp_removed": r.fp_removed,
                "params": r.params,
            } for r in selected],
        }

        log(f"  {scenario}: TP={m['tp']} FP={m['fp']} FN={m['fn']} recall={m['recall']} precision={m['precision']} fp_removed={fp_removed_vs_base} tp_loss={tp_loss_vs_base}")

    frontier_df = pd.concat(frontiers, ignore_index=True) if frontiers else pd.DataFrame()
    frontier_df.to_csv(output_dir / "03_frontier_by_scenario.csv", index=False)

    scenario_metrics = pd.DataFrame(scenario_rows)
    scenario_metrics.to_csv(output_dir / "04_scenario_metrics.csv", index=False)

    selected_rules = pd.DataFrame(selected_rule_rows)
    selected_rules.to_csv(output_dir / "05_selected_rules_by_scenario.csv", index=False)

    predictions_out.to_csv(output_dir / "06_predictions_by_scenario.csv", index=False)

    fn_parts, fp_parts = [], []
    for sdef in scenario_defs:
        scenario = sdef["scenario"]
        col = f"pred_{scenario}"
        fn = predictions_out[(predictions_out["is_fraud"] == 1) & (predictions_out[col] == 0)].copy()
        fp = predictions_out[(predictions_out["is_fraud"] == 0) & (predictions_out[col] == 1)].copy()
        fn["scenario"] = scenario
        fp["scenario"] = scenario
        fn_parts.append(fn)
        fp_parts.append(fp)

    pd.concat(fn_parts, ignore_index=True).to_csv(output_dir / "07_false_negatives_by_scenario.csv", index=False)
    pd.concat(fp_parts, ignore_index=True).to_csv(output_dir / "08_false_positives_by_scenario.csv", index=False)

    block_df = pd.concat(block_rows, ignore_index=True) if block_rows else pd.DataFrame()
    block_df.to_csv(output_dir / "09_time_block_metrics.csv", index=False)

    boot_df = pd.concat(boot_rows, ignore_index=True) if boot_rows else pd.DataFrame()
    boot_df.to_csv(output_dir / "10_bootstrap_confidence_intervals.csv", index=False)

    dump_json(policy_artifacts, output_dir / "11_policy_artifacts.json")

    strict = scenario_metrics[scenario_metrics["scenario"] == "STRICT_RECALL95_SAFE_ONLY"].iloc[0].to_dict()
    exc = scenario_metrics[scenario_metrics["scenario"] == "ONE_TP_EXCEPTION_HIGH_RETURN"].iloc[0].to_dict()

    objective_status = "DONE"
    if strict["fp"] < base_metrics["fp"] and strict["tp"] >= min_tp_recall95:
        objective_status += "_STRICT_RECALL95_IMPROVED"
    else:
        objective_status += "_STRICT_RECALL95_NOT_IMPROVED"

    if exc["tp_loss_vs_base"] <= 1 and exc["fp_removed_vs_base"] >= args.one_tp_min_extra_fp:
        objective_status += "_ONETP_EXCEPTION_HAS_HIGH_RETURN"
    else:
        objective_status += "_ONETP_EXCEPTION_NOT_HIGH_RETURN"

    summary = {
        "experiment": "EXP-013J",
        "status": "DONE",
        "objective_status": objective_status,
        "predictions_path": str(predictions_path),
        "pred_col": pred_col,
        "n_rows": int(len(df)),
        "total_frauds": total_frauds,
        "min_tp_recall95": min_tp_recall95,
        "base_metrics": base_metrics,
        "n_safe_rules_loaded": int(len(safe_rules)),
        "n_near_rules_loaded": int(len(near_rules)),
        "n_unique_rules": int(len(all_rules)),
        "scenario_metrics": scenario_metrics.to_dict(orient="records"),
        "one_tp_min_extra_fp": args.one_tp_min_extra_fp,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, scenario_metrics, selected_rules)
    (output_dir / "12_exp013j_report.md").write_text(report, encoding="utf-8")

    log("")
    log("=" * 80)
    log("EXP-013J CONCLUÍDO")
    log("=" * 80)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "03_frontier_by_scenario.csv",
        output_dir / "04_scenario_metrics.csv",
        output_dir / "05_selected_rules_by_scenario.csv",
        output_dir / "09_time_block_metrics.csv",
        output_dir / "10_bootstrap_confidence_intervals.csv",
        output_dir / "11_policy_artifacts.json",
        output_dir / "12_exp013j_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
