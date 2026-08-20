#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R1 — Resume Safe Beam from Partial EXP-014B

Objetivo:
  Refazer a etapa final do EXP-014B usando os resultados parciais já gerados:
    - 02_threshold_sweep.csv
    - 03_base_policy_metrics.csv
    - 04_veto_candidates.csv

  Sem repetir o sweep de threshold nem a mineração de candidatos, e evitando
  o travamento observado no depth=10.

Correções em relação ao EXP-014B original:
  1. Default max-rules=9, pois o run anterior já mostrou ganho forte até depth 9.
  2. Não recalcula TP_loss a cada expansão do beam:
       os candidatos importados já são TP_loss=0 e block_tp_loss_max=0;
       portanto a união deles também tem TP_loss=0.
  3. Reconstrói máscaras apenas dos top candidatos escolhidos, não dos 9.142 todos.
  4. Checkpoint por profundidade.
  5. Se houver KeyboardInterrupt, salva o melhor estado encontrado em vez de perder tudo.
  6. max_seconds opcional para encerrar com sucesso antes de ficar caro demais.
  7. Saída separada em EXP-014B-R1, sem sobrescrever EXP-014B.

Uso recomendado:
  python scripts/exp_014b_r1_resume_safe_beam.py

Equivalente ao run interrompido, mas seguro:
  python scripts/exp_014b_r1_resume_safe_beam.py --max-candidates 500 --beam-width 250 --max-rules 9 --bootstrap-iters 100

Se quiser tentar depth 10 com trava de tempo:
  python scripts/exp_014b_r1_resume_safe_beam.py --max-candidates 500 --beam-width 250 --max-rules 10 --max-seconds 900

Entradas default:
  dados/exp014a_expanded_scored_input.csv
  resultados/experimentos/EXP-014B/02_threshold_sweep.csv
  resultados/experimentos/EXP-014B/04_veto_candidates.csv

Saídas:
  resultados/experimentos/EXP-014B-R1/
    00_run_summary.json
    01_base_policy_metrics.csv
    02_candidates_used.csv
    03_frontier.csv
    04_selected_rules.csv
    05_policy_metrics.csv
    06_time_block_metrics.csv
    07_wilson_recall_ci.csv
    08_bootstrap_summary.csv
    09_false_negatives.csv
    10_false_positives_sample.csv
    11_policy_artifact.json
    12_exp014b_r1_report.md
    13_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import math
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
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "dados").exists() else Path.cwd()

DEFAULT_INPUT = PROJECT_ROOT / "dados" / "exp014a_expanded_scored_input.csv"
DEFAULT_EXP014B_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R1"

SCORE_COL_CANDIDATES = [
    "lgbm_r4_score",
    "r4_score",
    "lgbm_mapped",
    "lgbm_raw",
    "score_final",
]


@dataclass
class Candidate:
    rule_id: str
    description: str
    cols: list[str]
    vals: list[str]
    mask: np.ndarray
    fp_removed: int
    n_removed: int


@dataclass
class State:
    mask: np.ndarray
    rule_indices: tuple[int, ...]
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

    for c in ["event_datetime", "data_pix"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if "transaction_id" in df.columns:
        df["transaction_id"] = df["transaction_id"].astype("string").str.strip()

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


def ensure_bins_and_guards(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "lgbm_bin" not in df.columns and pick_col(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"]):
        df["lgbm_bin"] = qbin_series(num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0), "lgbm", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1])

    if "if_bin" not in df.columns and pick_col(df, ["if_percentile", "if_percentile_x", "if_percentile_y"]):
        df["if_bin"] = qbin_series(num(df, ["if_percentile", "if_percentile_x", "if_percentile_y"], 0.0), "if", [0.32, 0.5, 0.7, 0.85, 0.95])

    if "score_bin" not in df.columns and "score_final" in df.columns:
        df["score_bin"] = qbin_series(num(df, "score_final", 0.0), "score", [0.5, 1, 2, 3, 5, 10])

    if "ratio_bin" not in df.columns and "ratio_valor_media_pagador_90d" in df.columns:
        df["ratio_bin"] = qbin_series(num(df, "ratio_valor_media_pagador_90d", 0.0), "ratio", [0.05, 0.1, 0.2, 0.5, 1, 2, 5])

    if "vl_bin" not in df.columns and "vl_pix" in df.columns:
        df["vl_bin"] = qbin_series(num(df, "vl_pix", 0.0), "vl", [20, 50, 100, 250, 500, 1000, 5000, 10000])

    se_score = num(df, ["se_score_x", "se_score_y", "se_score"], 0.0)
    se_count = num(df, ["se_patterns_count", "se_pattern_count"], 0.0)
    beh_score = num(df, ["beh_score", "behavioral_score"], 0.0)
    beh_count = num(df, ["beh_factors_count", "behavioral_risk_factor_count"], 0.0)
    runtime = num(df, "runtime_flagged", 0.0)

    module_strong = (
        (se_score >= 40)
        | (se_count >= 2)
        | (beh_score >= 25)
        | (beh_count >= 2)
        | (runtime >= 1)
    )
    df["module_quiet"] = np.where(module_strong, "module_strong", "module_quiet")

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


def contract_report(df: pd.DataFrame) -> dict[str, Any]:
    missing = []
    if "is_fraud" not in df.columns:
        missing.append("is_fraud")
    if not any(c in df.columns for c in ["event_datetime", "data_pix", "dt_pix"]):
        missing.append("event_datetime_or_data_pix")
    if not any(c in df.columns for c in SCORE_COL_CANDIDATES):
        missing.append("risk_score_column")

    for logical, alternatives in {
        "lgbm_bin": [["lgbm_bin"], ["lgbm_r4_score"], ["r4_score"], ["lgbm_mapped"], ["lgbm_raw"]],
        "if_bin": [["if_bin"], ["if_percentile"], ["if_percentile_x"], ["if_percentile_y"]],
        "score_bin": [["score_bin"], ["score_final"]],
        "ratio_bin": [["ratio_bin"], ["ratio_valor_media_pagador_90d"]],
        "vl_bin": [["vl_bin"], ["vl_pix"]],
        "value_band": [["value_band"]],
        "ds_tipo_chave_norm": [["ds_tipo_chave_norm"]],
        "first_receiver_flag_real": [["first_receiver_flag_real"]],
        "mbk_available_flag": [["mbk_available_flag"]],
    }.items():
        if not any(all(c in df.columns for c in alt) for alt in alternatives):
            missing.append(f"feature_or_bin:{logical}")

    return {
        "contract_ok": len(missing) == 0,
        "missing": missing,
        "n_rows": int(len(df)),
        "n_frauds": int(df["is_fraud"].sum()) if "is_fraud" in df.columns else None,
        "score_cols_present": [c for c in SCORE_COL_CANDIDATES if c in df.columns],
    }


def select_base_policy_from_sweep(sweep: pd.DataFrame, target_recall: float) -> dict[str, Any]:
    strict = sweep[(sweep["full_recall"] >= target_recall) & (sweep["val_recall"] >= target_recall)].copy()
    if not strict.empty:
        row = strict.sort_values(["full_fp", "val_fp", "full_precision"], ascending=[True, True, False]).iloc[0]
        status = "STRICT_FULL_AND_VALIDATION_RECALL_MET"
    else:
        full = sweep[sweep["full_recall"] >= target_recall].copy()
        if not full.empty:
            row = full.sort_values(["full_fp", "val_recall", "full_precision"], ascending=[True, False, False]).iloc[0]
            status = "FULL_RECALL_MET_VALIDATION_WARNING"
        else:
            row = sweep.sort_values(["full_recall", "full_fp"], ascending=[False, True]).iloc[0]
            status = "TARGET_NOT_MET_BEST_AVAILABLE"

    return {
        "selection_status": status,
        "score_col": str(row["score_col"]),
        "direction": str(row["direction"]),
        "threshold": float(row["threshold"]),
        "full_metrics": {
            "tp": int(row["full_tp"]),
            "fp": int(row["full_fp"]),
            "fn": int(row["full_fn"]),
            "precision": float(row["full_precision"]),
            "recall": float(row["full_recall"]),
            "fpr": float(row["full_fpr"]),
        },
        "validation_metrics": {
            "tp": int(row["val_tp"]),
            "fp": int(row["val_fp"]),
            "fn": int(row["val_fn"]),
            "precision": float(row["val_precision"]),
            "recall": float(row["val_recall"]),
            "fpr": float(row["val_fpr"]),
        },
        "discovery_metrics": {
            "tp": int(row["disc_tp"]),
            "fp": int(row["disc_fp"]),
            "fn": int(row["disc_fn"]),
            "precision": float(row["disc_precision"]),
            "recall": float(row["disc_recall"]),
            "fpr": float(row["disc_fpr"]),
        },
    }


def apply_threshold(df: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    scores = pd.to_numeric(df[spec["score_col"]], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    if spec["direction"] == "ge":
        return (scores >= float(spec["threshold"])).astype(int).to_numpy()
    return (scores <= float(spec["threshold"])).astype(int).to_numpy()


def parse_list_field(raw: Any) -> list[str]:
    if pd.isna(raw):
        return []
    return [str(x) for x in str(raw).split("|")]


def reconstruct_candidates(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    candidates_csv: Path,
    max_candidates: int,
    min_fp_removed: int,
) -> tuple[list[Candidate], pd.DataFrame]:
    raw = pd.read_csv(candidates_csv, low_memory=False)
    raw = raw.sort_values(["fp_removed", "n_removed"], ascending=[False, False]).reset_index(drop=True)
    raw = raw[raw["fp_removed"] >= min_fp_removed].copy()

    # Load more rows than max_candidates because some can fail after reconstruction.
    raw = raw.head(max_candidates * 3).copy()

    y = df["is_fraud"].to_numpy(dtype=int)
    pred_pos = base_pred.astype(bool)
    out: list[Candidate] = []
    rows = []

    for _, row in raw.iterrows():
        cols = parse_list_field(row.get("cols"))
        vals = parse_list_field(row.get("vals"))
        if not cols or len(cols) != len(vals):
            continue

        mask = np.ones(len(df), dtype=bool)
        ok = True
        for c, v in zip(cols, vals):
            if c not in df.columns:
                ok = False
                break
            mask = mask & (df[c].astype(str) == str(v))

        if not ok:
            continue

        mask = mask & pred_pos
        tp_loss = int(((y == 1) & mask).sum())
        n_removed = int(mask.sum())
        fp_removed = int(((y == 0) & mask).sum())

        # Keep only truly safe candidates after reconstruction.
        if tp_loss != 0 or fp_removed < min_fp_removed:
            continue

        cid = str(row.get("rule_id", f"candidate_{len(out):05d}"))
        desc = str(row.get("description", " AND ".join([f"{c}={v}" for c, v in zip(cols, vals)])))

        out.append(Candidate(
            rule_id=cid,
            description=desc,
            cols=cols,
            vals=vals,
            mask=mask,
            fp_removed=fp_removed,
            n_removed=n_removed,
        ))
        rows.append({
            "candidate_index": len(out) - 1,
            "rule_id": cid,
            "description": desc,
            "cols": "|".join(cols),
            "vals": "|".join(vals),
            "tp_loss": tp_loss,
            "fp_removed": fp_removed,
            "n_removed": n_removed,
            "source_fp_removed": int(row.get("fp_removed", fp_removed)),
        })

        if len(out) >= max_candidates:
            break

    used = pd.DataFrame(rows)
    return out, used


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


def search_best_vetos_safe(
    candidates: list[Candidate],
    base_pred: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    beam_width: int,
    max_rules: int,
    max_seconds: int | None,
    checkpoint_each_depth: bool,
) -> tuple[pd.DataFrame, State, list[Candidate], str]:
    t0 = time.perf_counter()
    zero = np.zeros(len(y), dtype=bool)
    initial = State(mask=zero, rule_indices=tuple(), fp_removed=0)
    states = [initial]
    best = initial
    rows = []
    stop_reason = "completed"

    # Every candidate is reconstructed with tp_loss=0, so every union also has TP_loss=0.
    # Therefore fp_removed == union_mask.sum(), no need to recompute ((y==1)&mask).sum() in the inner loop.
    try:
        for depth in range(1, max_rules + 1):
            if max_seconds is not None and (time.perf_counter() - t0) >= max_seconds:
                stop_reason = f"max_seconds_before_depth_{depth}"
                break

            next_states: dict[bytes, State] = {}
            depth_t0 = time.perf_counter()

            for state in states:
                last = state.rule_indices[-1] if state.rule_indices else -1
                for i in range(last + 1, len(candidates)):
                    c = candidates[i]
                    new_mask = state.mask | c.mask
                    if np.array_equal(new_mask, state.mask):
                        continue

                    fp_removed = int(new_mask.sum())
                    if fp_removed <= state.fp_removed:
                        continue

                    key = np.packbits(new_mask).tobytes()
                    ns = State(new_mask, state.rule_indices + (i,), fp_removed)
                    old = next_states.get(key)
                    if old is None or (ns.fp_removed, -len(ns.rule_indices)) > (old.fp_removed, -len(old.rule_indices)):
                        next_states[key] = ns

            if not next_states:
                stop_reason = f"no_next_states_at_depth_{depth}"
                break

            states = sorted(next_states.values(), key=lambda s: (s.fp_removed, -len(s.rule_indices)), reverse=True)[:beam_width]

            if states[0].fp_removed > best.fp_removed:
                best = states[0]

            for s in states[:50]:
                pred = base_pred.copy()
                pred[s.mask] = 0
                m = compute_metrics(y, pred)
                rows.append({
                    "depth": depth,
                    "tp_loss": 0,
                    "fp_removed": int(s.fp_removed),
                    "n_rules": len(s.rule_indices),
                    **m,
                    "rule_ids": "|".join(candidates[i].rule_id for i in s.rule_indices),
                    "rule_descriptions": " || ".join(candidates[i].description for i in s.rule_indices),
                })

            if checkpoint_each_depth:
                ck = pd.DataFrame(rows)
                ck.to_csv(output_dir / f"checkpoint_frontier_depth_{depth:02d}.csv", index=False)
                selected_ck = selected_rules_df([candidates[i] for i in best.rule_indices])
                selected_ck.to_csv(output_dir / f"checkpoint_selected_depth_{depth:02d}.csv", index=False)

            elapsed_depth = time.perf_counter() - depth_t0
            elapsed_total = time.perf_counter() - t0
            log(f"  safe beam depth={depth}: best_fp_removed={best.fp_removed}, states={len(states)}, depth_s={elapsed_depth:.1f}, total_s={elapsed_total:.1f}")

            if max_seconds is not None and elapsed_total >= max_seconds:
                stop_reason = f"max_seconds_after_depth_{depth}"
                break

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt_saved_best"
        log("KeyboardInterrupt capturado: salvando melhor estado encontrado.")

    if not rows:
        m = compute_metrics(y, base_pred)
        rows = [{
            "depth": 0,
            "tp_loss": 0,
            "fp_removed": 0,
            "n_rules": 0,
            **m,
            "rule_ids": "",
            "rule_descriptions": "",
        }]

    frontier = pd.DataFrame(rows).sort_values(["fp", "n_rules"], ascending=[True, True]).reset_index(drop=True)
    selected = [candidates[i] for i in best.rule_indices]
    return frontier, best, selected, stop_reason


def selected_rules_df(selected: list[Candidate]) -> pd.DataFrame:
    return pd.DataFrame([{
        "selected_order": i,
        "rule_id": c.rule_id,
        "description": c.description,
        "cols": "|".join(c.cols),
        "vals": "|".join(c.vals),
        "tp_loss": 0,
        "fp_removed": int(c.fp_removed),
        "n_removed": int(c.n_removed),
    } for i, c in enumerate(selected)])


def block_metrics(df: pd.DataFrame, pred: np.ndarray, blocks: pd.Series, policy_name: str) -> pd.DataFrame:
    rows = []
    y = df["is_fraud"].to_numpy(dtype=int)
    bvals = blocks.to_numpy()
    for b in sorted(blocks.dropna().unique()):
        idx = bvals == b
        part = df.loc[idx]
        m = compute_metrics(y[idx], pred[idx])
        rows.append({
            "policy_name": policy_name,
            "block": int(b),
            "n_rows": int(len(part)),
            "n_frauds": int(part["is_fraud"].sum()),
            **m,
        })
    return pd.DataFrame(rows)


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) / n) + (z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_summary(df: pd.DataFrame, pred_col: str, iters: int, seed: int, target_recall: float) -> pd.DataFrame:
    if iters <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    y_all = df["is_fraud"].to_numpy(dtype=int)
    pred_all = df[pred_col].to_numpy(dtype=int)
    pos_idx = np.where(y_all == 1)[0]
    neg_idx = np.where(y_all == 0)[0]

    rows = []
    for _ in range(iters):
        s_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        s_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([s_pos, s_neg])
        rows.append(compute_metrics(y_all[idx], pred_all[idx]))

    boot = pd.DataFrame(rows)
    out = []
    for metric in ["tp", "fp", "fn", "precision", "recall", "f1", "fpr"]:
        vals = boot[metric].astype(float)
        out.append({
            "method": "stratified_class",
            "metric": metric,
            "mean": float(vals.mean()),
            "p025": float(vals.quantile(0.025)),
            "p050": float(vals.quantile(0.50)),
            "p975": float(vals.quantile(0.975)),
            "target_recall": target_recall if metric == "recall" else None,
            "p_below_target_recall": float((boot["recall"] < target_recall).mean()) if metric == "recall" else None,
        })
    return pd.DataFrame(out)


def make_report(summary: dict[str, Any], policy_metrics: pd.DataFrame, selected: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014B-R1 — Resume Safe Beam")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Stop reason: `{summary['stop_reason']}`")
    lines.append(f"- Base score: `{summary['base_policy']['score_col']}` `{summary['base_policy']['direction']}` `{summary['base_policy']['threshold']}`")
    lines.append("")
    lines.append("## Métricas")
    lines.append(policy_metrics.to_markdown(index=False))
    lines.append("")
    lines.append("## Regras selecionadas")
    if selected.empty:
        lines.append("Nenhuma regra selecionada.")
    else:
        lines.append(selected.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretação")
    lines.append("Esta rodada reaproveita os parciais do EXP-014B e evita o erro do depth=10 usando beam seguro, checkpoint e encerramento controlado.")
    if summary["selected_metrics"]["recall"] >= summary["target_recall"] and summary["tp_loss_vs_base"] == 0:
        lines.append("A política mantém a meta de recall e não perde TP em relação à base. O próximo passo recomendado é comparar FP residual e decidir se congelamos em EXP-014C ou fazemos R2 com uma base menos ampla.")
    else:
        lines.append("A política não cumpriu integralmente os gates; usar como diagnóstico.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--partial-dir", default=str(DEFAULT_EXP014B_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--min-fp-removed", type=int, default=25)
    parser.add_argument("--beam-width", type=int, default=250)
    parser.add_argument("--max-rules", type=int, default=9)
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--time-blocks", type=int, default=10)
    parser.add_argument("--bootstrap-iters", type=int, default=100)
    parser.add_argument("--false-positive-sample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-write-predictions", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    partial_dir = Path(args.partial_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_path = partial_dir / "02_threshold_sweep.csv"
    candidates_path = partial_dir / "04_veto_candidates.csv"

    log("=" * 80)
    log("EXP-014B-R1 — Resume Safe Beam")
    log("=" * 80)
    log(f"Input: {input_path}")
    log(f"Partial dir: {partial_dir}")
    log(f"Output: {output_dir}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")
    if not sweep_path.exists():
        raise FileNotFoundError(f"Threshold sweep parcial não encontrado: {sweep_path}")
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidatos parciais não encontrados: {candidates_path}")

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    df = ensure_bins_and_guards(df)

    contract = contract_report(df)
    dump_json(contract, output_dir / "input_contract.json")
    if not contract["contract_ok"]:
        raise RuntimeError(f"Contrato de input falhou: {contract['missing']}")

    y = df["is_fraud"].to_numpy(dtype=int)

    log("[1/4] Carregando threshold sweep e reconstruindo base...")
    sweep = pd.read_csv(sweep_path)
    base_policy = select_base_policy_from_sweep(sweep, args.target_recall)
    base_pred = apply_threshold(df, base_policy)
    df["exp014b_r1_base_high_recall_pred"] = base_pred
    base_m = compute_metrics(y, base_pred)
    pd.DataFrame([{"policy_name": "EXP014B_R1_BASE_HIGH_RECALL", **base_m}]).to_csv(output_dir / "01_base_policy_metrics.csv", index=False)
    log(f"Base policy: {base_policy}")

    log("[2/4] Reconstruindo top candidatos seguros a partir do CSV parcial...")
    candidates, used_df = reconstruct_candidates(
        df=df,
        base_pred=base_pred,
        candidates_csv=candidates_path,
        max_candidates=args.max_candidates,
        min_fp_removed=args.min_fp_removed,
    )
    used_df.to_csv(output_dir / "02_candidates_used.csv", index=False)
    log(f"Candidatos usados: {len(candidates)}")

    if not candidates:
        raise RuntimeError("Nenhum candidato seguro foi reconstruído.")

    log("[3/4] Safe beam search com checkpoint e limite de tempo...")
    frontier, best, selected, stop_reason = search_best_vetos_safe(
        candidates=candidates,
        base_pred=base_pred,
        y=y,
        output_dir=output_dir,
        beam_width=args.beam_width,
        max_rules=args.max_rules,
        max_seconds=args.max_seconds,
        checkpoint_each_depth=not args.no_checkpoints,
    )
    frontier.to_csv(output_dir / "03_frontier.csv", index=False)

    selected_df = selected_rules_df(selected)
    selected_df.to_csv(output_dir / "04_selected_rules.csv", index=False)

    final_pred = base_pred.copy()
    final_pred[best.mask] = 0
    df["exp014b_r1_safe_beam_pred"] = final_pred

    base_metrics = compute_metrics(y, base_pred)
    selected_metrics = compute_metrics(y, final_pred)
    fp_removed_vs_base = int(base_metrics["fp"] - selected_metrics["fp"])
    tp_loss_vs_base = int(base_metrics["tp"] - selected_metrics["tp"])

    policy_rows = [
        {"policy_name": "EXP014B_R1_BASE_HIGH_RECALL", **base_metrics},
        {"policy_name": "EXP014B_R1_SAFE_BEAM", **selected_metrics},
    ]

    for runtime_col in ["exp014a_frozen_pred", "exp013k_residual_fp_pred"]:
        if runtime_col in df.columns:
            policy_rows.insert(0, {"policy_name": f"RUNTIME_FINAL_{runtime_col}", **compute_metrics(y, df[runtime_col].to_numpy(dtype=int))})
            break

    policy_metrics = pd.DataFrame(policy_rows)
    policy_metrics.to_csv(output_dir / "05_policy_metrics.csv", index=False)

    log("[4/4] Blocos, Wilson, bootstrap e artefatos...")
    blocks = make_time_blocks(df, args.time_blocks)
    block_df = pd.concat([
        block_metrics(df, base_pred, blocks, "EXP014B_R1_BASE_HIGH_RECALL"),
        block_metrics(df, final_pred, blocks, "EXP014B_R1_SAFE_BEAM"),
    ], ignore_index=True)
    block_df.to_csv(output_dir / "06_time_block_metrics.csv", index=False)

    total_frauds = int(df["is_fraud"].sum())
    min_tp_required = int(math.ceil(args.target_recall * total_frauds))
    wilson_low, wilson_high = wilson_ci(selected_metrics["tp"], total_frauds)
    wilson_df = pd.DataFrame([{
        "metric": "recall",
        "successes_tp": selected_metrics["tp"],
        "n_frauds": total_frauds,
        "point_estimate": selected_metrics["recall"],
        "wilson_low": wilson_low,
        "wilson_high": wilson_high,
        "target_recall": args.target_recall,
        "min_tp_required": min_tp_required,
        "tp_buffer_vs_target": selected_metrics["tp"] - min_tp_required,
        "wilson_low_ge_target": bool(wilson_low >= args.target_recall),
    }])
    wilson_df.to_csv(output_dir / "07_wilson_recall_ci.csv", index=False)

    boot_df = bootstrap_summary(df, "exp014b_r1_safe_beam_pred", args.bootstrap_iters, args.seed, args.target_recall)
    boot_df.to_csv(output_dir / "08_bootstrap_summary.csv", index=False)

    fn = df[(df["is_fraud"] == 1) & (df["exp014b_r1_safe_beam_pred"] == 0)].copy()
    fp = df[(df["is_fraud"] == 0) & (df["exp014b_r1_safe_beam_pred"] == 1)].copy()
    fn.to_csv(output_dir / "09_false_negatives.csv", index=False)
    if len(fp) > args.false_positive_sample:
        fp = fp.sample(args.false_positive_sample, random_state=args.seed)
    fp.to_csv(output_dir / "10_false_positives_sample.csv", index=False)

    if not args.no_write_predictions:
        df.to_csv(output_dir / "13_predictions.csv", index=False)

    objective_status = "DONE"
    objective_status += "_TARGET_RECALL_MET" if selected_metrics["recall"] >= args.target_recall else "_TARGET_RECALL_NOT_MET"
    objective_status += "_TPLOSS0" if tp_loss_vs_base == 0 else "_TPLOSS_GT0"
    objective_status += "_FP_REDUCED" if fp_removed_vs_base > 0 else "_FP_NOT_REDUCED"
    objective_status += "_WILSON_PASS" if wilson_low >= args.target_recall else "_WILSON_NOT_PASS"

    artifact = {
        "experiment": "EXP-014B-R1",
        "policy_name": "expanded_high_recall_safe_beam_from_exp014b_partials",
        "objective_status": objective_status,
        "stop_reason": stop_reason,
        "base_policy": base_policy,
        "base_metrics": base_metrics,
        "selected_metrics": selected_metrics,
        "fp_removed_vs_base": fp_removed_vs_base,
        "tp_loss_vs_base": tp_loss_vs_base,
        "wilson": wilson_df.to_dict(orient="records")[0],
        "selected_rules": selected_df.to_dict(orient="records") if not selected_df.empty else [],
        "notes": [
            "Uses EXP-014B partial threshold sweep and veto candidates.",
            "Does not rerun threshold sweep or candidate mining.",
            "Safe beam assumes reconstructed candidates have TP_loss=0, making unions TP_loss=0.",
            "Default max_rules=9 avoids the previous depth=10 stall.",
            "Checkpoints are written per depth."
        ],
    }
    dump_json(artifact, output_dir / "11_policy_artifact.json")

    summary = {
        "experiment": "EXP-014B-R1",
        "status": "DONE",
        "objective_status": objective_status,
        "stop_reason": stop_reason,
        "input_path": str(input_path),
        "partial_dir": str(partial_dir),
        "n_rows": int(len(df)),
        "n_frauds": total_frauds,
        "target_recall": args.target_recall,
        "base_policy": base_policy,
        "base_metrics": base_metrics,
        "selected_metrics": selected_metrics,
        "fp_removed_vs_base": fp_removed_vs_base,
        "tp_loss_vs_base": tp_loss_vs_base,
        "n_candidates_used": int(len(candidates)),
        "n_selected_rules": int(len(selected)),
        "max_rules": args.max_rules,
        "beam_width": args.beam_width,
        "max_seconds": args.max_seconds,
        "wilson_recall_low": wilson_low,
        "wilson_recall_high": wilson_high,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, policy_metrics, selected_df)
    (output_dir / "12_exp014b_r1_report.md").write_text(report, encoding="utf-8")

    log("")
    log("=" * 80)
    log("EXP-014B-R1 CONCLUÍDO")
    log("=" * 80)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_base_policy_metrics.csv",
        output_dir / "02_candidates_used.csv",
        output_dir / "03_frontier.csv",
        output_dir / "04_selected_rules.csv",
        output_dir / "05_policy_metrics.csv",
        output_dir / "06_time_block_metrics.csv",
        output_dir / "07_wilson_recall_ci.csv",
        output_dir / "08_bootstrap_summary.csv",
        output_dir / "11_policy_artifact.json",
        output_dir / "12_exp014b_r1_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
