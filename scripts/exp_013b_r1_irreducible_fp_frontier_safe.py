#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-013B-R1 - Irreducible FP Frontier SAFE

Versao segura do EXP-013B. O objetivo e encontrar combinacoes pequenas de
vetos estatisticos que reduzam falsos positivos mantendo recall >= 95%, sem
busca combinatoria explosiva.

Entrada default:
  resultados/experimentos/EXP-012E/04_comparison_by_transaction.csv

Execucao:
  python scripts/exp_013b_r1_irreducible_fp_frontier_safe.py

Mais profundo, ainda seguro:
  python scripts/exp_013b_r1_irreducible_fp_frontier_safe.py --max-candidates 500 --beam-width 100 --max-depth 8

Ultra conservador:
  python scripts/exp_013b_r1_irreducible_fp_frontier_safe.py --max-candidates 100 --beam-width 20 --max-depth 4 --timeout-seconds 120
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
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
DEFAULT_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-012E" / "04_comparison_by_transaction.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013B-R1"
FLAGGED_DECISIONS = {"CONFIRMAR", "BLOQUEAR"}


@dataclass
class Rule:
    rule_id: str
    family: str
    description: str
    mask: np.ndarray
    tp_loss: int
    fp_removed: int
    n_removed: int
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def fp_per_tp(self) -> float:
        return self.fp_removed / max(self.tp_loss, 1)


@dataclass
class State:
    removed_mask: np.ndarray
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
    if "is_fraud" not in df.columns:
        raise RuntimeError("Coluna is_fraud ausente.")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)
    if "shadow_exp012d_flagged" not in df.columns:
        for c in ["exp012d_pred", "r4_pred", "lgbm_r4_pred"]:
            if c in df.columns:
                df["shadow_exp012d_flagged"] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
                break
    if "shadow_exp012d_flagged" not in df.columns:
        raise RuntimeError("Nao encontrei shadow_exp012d_flagged/exp012d_pred/r4_pred/lgbm_r4_pred.")
    df["shadow_exp012d_flagged"] = pd.to_numeric(df["shadow_exp012d_flagged"], errors="coerce").fillna(0).astype(int)
    if "runtime_flagged" not in df.columns:
        if "decisao" in df.columns:
            df["runtime_flagged"] = df["decisao"].astype(str).str.upper().isin(FLAGGED_DECISIONS).astype(int)
        else:
            df["runtime_flagged"] = 0
    df["runtime_flagged"] = pd.to_numeric(df["runtime_flagged"], errors="coerce").fillna(0).astype(int)
    if "transaction_id" in df.columns:
        df["transaction_id"] = df["transaction_id"].astype("string").str.strip()
    return df.reset_index(drop=True)


def pick_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def num(df: pd.DataFrame, names: str | list[str], default: float = 0.0) -> pd.Series:
    if isinstance(names, str):
        names = [names]
    col = pick_col(df, names)
    if col is None:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def txt(df: pd.DataFrame, names: str | list[str], default: str = "<MISSING>") -> pd.Series:
    if isinstance(names, str):
        names = [names]
    col = pick_col(df, names)
    if col is None:
        return pd.Series([default] * len(df), index=df.index, dtype="string")
    return df[col].astype("string").fillna(default).astype(str)


def boolish(df: pd.DataFrame, names: str | list[str], default: bool = False) -> pd.Series:
    if isinstance(names, str):
        names = [names]
    col = pick_col(df, names)
    if col is None:
        return pd.Series([default] * len(df), index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(default)
    return s.astype(str).str.upper().isin({"1", "1.0", "TRUE", "T", "SIM", "YES", "Y"})


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 8),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 8),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 8),
        "fpr": round(float(fp / max(fp + tn, 1)), 8),
    }


def strong_preserve_mask(df_pos: pd.DataFrame) -> np.ndarray:
    se_score = num(df_pos, ["se_score_x", "se_score_y", "se_score"], 0.0)
    se_count = num(df_pos, ["se_patterns_count", "se_pattern_count"], 0.0)
    beh_score = num(df_pos, ["beh_score", "behavioral_score"], 0.0)
    beh_count = num(df_pos, ["beh_factors_count", "behavioral_risk_factor_count"], 0.0)
    runtime = num(df_pos, "runtime_flagged", 0.0)
    cascade = boolish(df_pos, "cascade_triggered", False)
    decisao = txt(df_pos, "decisao", "").str.upper()
    return ((se_score >= 65) | (se_count >= 2) | (beh_score >= 45) | (beh_count >= 2) | (runtime >= 1) | decisao.isin(FLAGGED_DECISIONS) | cascade).to_numpy(dtype=bool)


def sanitize_id(value: str, max_len: int = 90) -> str:
    out = re.sub(r"[^A-Za-z0-9_]+", "_", str(value))
    out = re.sub(r"_+", "_", out).strip("_")
    return out[:max_len] or "rule"


def add_rule(rules: list[Rule], y_pos: np.ndarray, family: str, desc: str, mask: np.ndarray, params: dict[str, Any], max_tp_loss: int, min_fp_removed: int) -> None:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return
    tp_loss = int(y_pos[mask].sum())
    fp_removed = int(((1 - y_pos)[mask]).sum())
    n_removed = int(mask.sum())
    if fp_removed < min_fp_removed or tp_loss > max_tp_loss:
        return
    rid = sanitize_id(f"{family}_{len(rules):05d}_{desc}")
    rules.append(Rule(rid, family, desc, mask, tp_loss, fp_removed, n_removed, params))


def qvals(series: pd.Series, quantiles: list[float]) -> list[float]:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if vals.empty or vals.nunique() <= 1:
        return []
    return sorted(set(float(x) for x in vals.quantile(quantiles).dropna().tolist()))


def generate_rules(df_pos: pd.DataFrame, y_pos: np.ndarray, max_tp_loss: int, min_fp_removed: int, min_segment_n: int) -> list[Rule]:
    rules: list[Rule] = []
    preserve = strong_preserve_mask(df_pos)
    lgbm = num(df_pos, "lgbm_r4_score", 0.0)
    score_final = num(df_pos, "score_final", np.nan)
    ifp = num(df_pos, ["if_percentile_x", "if_percentile_y", "if_percentile"], 0.0)
    se_score = num(df_pos, ["se_score_x", "se_score_y", "se_score"], 0.0)
    se_count = num(df_pos, ["se_patterns_count", "se_pattern_count"], 0.0)
    beh_score = num(df_pos, ["beh_score", "behavioral_score"], 0.0)
    beh_count = num(df_pos, ["beh_factors_count", "behavioral_risk_factor_count"], 0.0)
    vl = num(df_pos, "vl_pix", 0.0)
    ratio = num(df_pos, "ratio_valor_media_pagador_90d", 0.0)
    ratio_max = num(df_pos, "ratio_valor_maximo_pagador_180d", 0.0)
    qtd_rec_180 = num(df_pos, "qtd_pix_recebidos_180d", 0.0)
    qtd_rec_90 = num(df_pos, "qtd_pix_recebidos_90d", 0.0)
    valor_rec_180 = num(df_pos, "valor_total_recebido_180d", 0.0)
    pagadores_dist = num(df_pos, "soma_pagadores_distintos_dia_recebedor_180d", 0.0)

    log("[1/5] Gerando regras numericas...")
    numeric_specs = [
        ("lgbm_r4_score", lgbm, "lt", [0.00051351172, 0.00076308066, 0.001, 0.0019429789, 0.003, 0.005, 0.01, 0.02, 0.05]),
        ("score_final", score_final, "lt", [0.50, 0.74, 0.76, 1.0, 2.0, 3.0, 4.0]),
        ("if_percentile", ifp, "lt", [0.320032, 0.50, 0.70, 0.85, 0.95]),
        ("vl_pix", vl, "lt", [10, 15, 20, 25, 50, 100, 250, 500]),
        ("ratio_valor_media_pagador_90d", ratio, "lt", [0.068208507, 0.10726481, 0.19765786, 0.5, 1.0]),
        ("ratio_valor_maximo_pagador_180d", ratio_max, "lt", [0.0025752, 0.0048909025, 0.05, 0.1, 0.2]),
        ("qtd_pix_recebidos_180d", qtd_rec_180, "gt", [1, 2, 5, 10, 20, 32, 50]),
        ("qtd_pix_recebidos_90d", qtd_rec_90, "gt", [1, 2, 5, 10, 20, 32]),
        ("valor_total_recebido_180d", valor_rec_180, "gt", [1, 10, 50, 81, 100, 500, 1000, 2000, 10000]),
        ("soma_pagadores_distintos_dia_recebedor_180d", pagadores_dist, "gt", [1, 2, 5, 10, 20, 28.4, 50]),
    ]
    for feat, values, op, fixed in numeric_specs:
        thresholds = list(fixed) + qvals(values, [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90])
        thresholds = sorted(set(round(float(t), 12) for t in thresholds if pd.notna(t)))
        for th in thresholds:
            base = (values < th).to_numpy(dtype=bool) if op == "lt" else (values > th).to_numpy(dtype=bool)
            sym = "<" if op == "lt" else ">"
            add_rule(rules, y_pos, "numeric", f"{feat} {sym} {th}", base, {"feature": feat, "op": op, "threshold": th, "preserve": False}, max_tp_loss, min_fp_removed)
            add_rule(rules, y_pos, "numeric_preserve", f"{feat} {sym} {th} AND NOT strong_preserve", base & (~preserve), {"feature": feat, "op": op, "threshold": th, "preserve": True}, max_tp_loss, min_fp_removed)
    log(f"    regras ate aqui: {len(rules)}")

    log("[2/5] Gerando regras compostas...")
    for lth in [0.00076308066, 0.0019429789, 0.003, 0.005, 0.01, 0.02]:
        for ifth in [0.320032, 0.50, 0.70, 0.85, 0.95]:
            m = ((lgbm < lth) & (ifp < ifth) & (se_score <= 20) & (se_count < 2) & (beh_score <= 25) & (beh_count < 2) & (~preserve)).to_numpy(dtype=bool)
            add_rule(rules, y_pos, "quiet_veto", f"lgbm<{lth} AND if<{ifth} AND SE/BEH quiet", m, {"lgbm_lt": lth, "if_lt": ifth}, max_tp_loss, min_fp_removed)

    for lth in [0.0019429789, 0.005, 0.01, 0.02, 0.05]:
        for qth in [1, 2, 5, 10, 20, 50]:
            m = ((lgbm < lth) & ((qtd_rec_180 > qth) | (qtd_rec_90 > qth)) & (~preserve)).to_numpy(dtype=bool)
            add_rule(rules, y_pos, "receiver_established", f"lgbm<{lth} AND receiver_qtd>{qth}", m, {"lgbm_lt": lth, "receiver_qtd_gt": qth}, max_tp_loss, min_fp_removed)
        for vth in [1, 10, 50, 81, 100, 500, 1000, 2000]:
            m = ((lgbm < lth) & (valor_rec_180 > vth) & (~preserve)).to_numpy(dtype=bool)
            add_rule(rules, y_pos, "receiver_value_established", f"lgbm<{lth} AND receiver_value_180d>{vth}", m, {"lgbm_lt": lth, "receiver_value_gt": vth}, max_tp_loss, min_fp_removed)
        for vlth in [20, 50, 100, 250, 500]:
            m = ((lgbm < lth) & (vl < vlth) & (ifp < 0.95) & (~preserve)).to_numpy(dtype=bool)
            add_rule(rules, y_pos, "low_value_weak_signal", f"lgbm<{lth} AND vl_pix<{vlth} AND if<0.95", m, {"lgbm_lt": lth, "vl_lt": vlth}, max_tp_loss, min_fp_removed)
    log(f"    regras ate aqui: {len(rules)}")

    log("[3/5] Gerando regras segmentadas...")
    segment_sets = [
        ["ds_tipo_chave_norm"], ["value_band"], ["periodo_dia"], ["mbk_available_flag"], ["first_receiver_flag_real"],
        ["value_band", "ds_tipo_chave_norm"], ["first_receiver_flag_real", "ds_tipo_chave_norm"], ["first_receiver_flag_real", "value_band"],
        ["periodo_dia", "value_band"], ["mbk_available_flag", "ds_tipo_chave_norm"], ["mbk_available_flag", "ds_tipo_chave_norm", "value_band"],
    ]
    for cols in segment_sets:
        if any(c not in df_pos.columns for c in cols):
            continue
        key_frame = pd.DataFrame(index=df_pos.index)
        for c in cols:
            key_frame[c] = txt(df_pos, c)
        grouped = key_frame.groupby(cols, dropna=False).indices
        for key, idxs in grouped.items():
            idxs = np.array(list(idxs), dtype=int)
            if len(idxs) < min_segment_n:
                continue
            mask = np.zeros(len(df_pos), dtype=bool)
            mask[idxs] = True
            tp = int(y_pos[mask].sum())
            fp = int((1 - y_pos[mask]).sum())
            if fp < min_fp_removed or tp > max_tp_loss:
                continue
            key_tuple = key if isinstance(key, tuple) else (key,)
            desc = " AND ".join([f"{c}={v}" for c, v in zip(cols, key_tuple)])
            add_rule(rules, y_pos, "segment", desc, mask, {"segment_cols": cols, "segment_values": [str(v) for v in key_tuple], "preserve": False}, max_tp_loss, min_fp_removed)
            add_rule(rules, y_pos, "segment_preserve", desc + " AND NOT strong_preserve", mask & (~preserve), {"segment_cols": cols, "segment_values": [str(v) for v in key_tuple], "preserve": True}, max_tp_loss, min_fp_removed)
            for lth in [0.0019429789, 0.005, 0.01, 0.02]:
                sm = mask & (lgbm.to_numpy(dtype=float) < lth) & (~preserve)
                add_rule(rules, y_pos, "segment_lgbm", desc + f" AND lgbm<{lth}", sm, {"segment_cols": cols, "segment_values": [str(v) for v in key_tuple], "lgbm_lt": lth, "preserve": True}, max_tp_loss, min_fp_removed)
    log(f"    regras antes dedupe: {len(rules)}")
    return dedupe(rules)


def dedupe(rules: list[Rule]) -> list[Rule]:
    best: dict[bytes, Rule] = {}
    for r in rules:
        key = np.packbits(r.mask).tobytes()
        old = best.get(key)
        if old is None or (r.fp_removed, -r.tp_loss, -len(r.description)) > (old.fp_removed, -old.tp_loss, -len(old.description)):
            best[key] = r
    uniq = list(best.values())
    uniq.sort(key=lambda r: (r.tp_loss, -r.fp_removed, -r.fp_per_tp, len(r.description)))
    dominant: list[Rule] = []
    best_fp_by_tp: dict[int, int] = {}
    for r in uniq:
        is_dominated = any(tp <= r.tp_loss and fp >= r.fp_removed for tp, fp in best_fp_by_tp.items())
        if not is_dominated:
            dominant.append(r)
            best_fp_by_tp[r.tp_loss] = max(best_fp_by_tp.get(r.tp_loss, -1), r.fp_removed)
    ratio_best = sorted(uniq, key=lambda r: (r.fp_per_tp, r.fp_removed, -r.tp_loss), reverse=True)[:500]
    merged = {np.packbits(r.mask).tobytes(): r for r in dominant}
    for r in ratio_best:
        merged[np.packbits(r.mask).tobytes()] = r
    out = list(merged.values())
    out.sort(key=lambda r: (r.tp_loss, -r.fp_removed, -r.fp_per_tp))
    log(f"    regras apos dedupe: {len(out)}")
    return out


def rules_df(rules: list[Rule]) -> pd.DataFrame:
    return pd.DataFrame([{
        "rule_index": i, "rule_id": r.rule_id, "family": r.family, "description": r.description,
        "tp_loss": r.tp_loss, "fp_removed": r.fp_removed, "n_removed": r.n_removed,
        "fp_per_tp_loss": r.fp_per_tp, "params_json": json.dumps(r.params, ensure_ascii=False),
    } for i, r in enumerate(rules)])


def state_key(mask: np.ndarray) -> bytes:
    return np.packbits(mask).tobytes()


def run_beam(rules: list[Rule], y_pos: np.ndarray, max_tp_loss: int, max_candidates: int, beam_width: int, max_depth: int, timeout_seconds: int, outdir: Path) -> tuple[pd.DataFrame, State]:
    log("[4/5] Selecionando pool de candidatos...")
    ordered = sorted(enumerate(rules), key=lambda x: (x[1].tp_loss == 0, x[1].fp_removed, x[1].fp_per_tp), reverse=True)
    pairs = ordered[:max_candidates]
    original_indices = [i for i, _ in pairs]
    pool = [r for _, r in pairs]
    pd.DataFrame({
        "candidate_order": range(len(pool)), "original_rule_index": original_indices,
        "rule_id": [r.rule_id for r in pool], "family": [r.family for r in pool],
        "description": [r.description for r in pool], "tp_loss": [r.tp_loss for r in pool], "fp_removed": [r.fp_removed for r in pool],
    }).to_csv(outdir / "candidate_pool_used.csv", index=False)
    log(f"    pool={len(pool)} beam_width={beam_width} max_depth={max_depth} max_tp_loss={max_tp_loss}")
    start = time.perf_counter()
    zero = np.zeros(len(y_pos), dtype=bool)
    initial = State(zero, tuple(), 0, 0)
    states = [initial]
    best = initial
    frontier_rows = []
    for depth in range(1, max_depth + 1):
        elapsed = time.perf_counter() - start
        if timeout_seconds and elapsed > timeout_seconds:
            log(f"    [STOP] timeout antes da profundidade {depth}: {elapsed:.1f}s")
            break
        log(f"[5/5] depth {depth}/{max_depth} states={len(states)} best_fp_removed={best.fp_removed} elapsed={elapsed:.1f}s")
        next_by_mask: dict[bytes, State] = {}
        for state in states:
            last = state.rule_indices[-1] if state.rule_indices else -1
            for cidx in range(last + 1, len(pool)):
                r = pool[cidx]
                new_mask = state.removed_mask | r.mask
                if np.array_equal(new_mask, state.removed_mask):
                    continue
                tp_loss = int(y_pos[new_mask].sum())
                if tp_loss > max_tp_loss:
                    continue
                fp_removed = int(((1 - y_pos)[new_mask]).sum())
                if fp_removed <= state.fp_removed:
                    continue
                ns = State(new_mask, state.rule_indices + (cidx,), tp_loss, fp_removed)
                key = state_key(new_mask)
                old = next_by_mask.get(key)
                if old is None or (ns.fp_removed, -ns.tp_loss, -len(ns.rule_indices)) > (old.fp_removed, -old.tp_loss, -len(old.rule_indices)):
                    next_by_mask[key] = ns
        if not next_by_mask:
            log("    sem expansoes validas; encerrando.")
            break
        nxt = list(next_by_mask.values())
        nxt.sort(key=lambda s: (s.fp_removed, -s.tp_loss, -len(s.rule_indices)), reverse=True)
        states = nxt[:beam_width]
        if (states[0].fp_removed, -states[0].tp_loss) > (best.fp_removed, -best.tp_loss):
            best = states[0]
        rows = []
        for st in states[:500]:
            rows.append({
                "depth": depth, "tp_loss": st.tp_loss, "fp_removed": st.fp_removed, "n_rules": len(st.rule_indices),
                "candidate_rule_indices": "|".join(str(i) for i in st.rule_indices),
                "original_rule_indices": "|".join(str(original_indices[i]) for i in st.rule_indices),
                "rule_ids": "|".join(pool[i].rule_id for i in st.rule_indices),
                "rule_descriptions": " || ".join(pool[i].description for i in st.rule_indices),
            })
        pd.DataFrame(rows).to_csv(outdir / f"partial_frontier_depth_{depth}.csv", index=False)
        frontier_rows.extend(rows)
    if not frontier_rows:
        frontier_rows = [{"depth": 0, "tp_loss": 0, "fp_removed": 0, "n_rules": 0, "candidate_rule_indices": "", "original_rule_indices": "", "rule_ids": "", "rule_descriptions": ""}]
    # translate best pool indexes to original rule indexes
    best = State(best.removed_mask, tuple(original_indices[i] for i in best.rule_indices), best.tp_loss, best.fp_removed)
    return pd.DataFrame(frontier_rows), best


def apply_best(df: pd.DataFrame, df_pos: pd.DataFrame, best: State) -> tuple[pd.DataFrame, dict[str, Any]]:
    pred = np.zeros(len(df), dtype=int)
    pos_idx = df_pos["_orig_index"].to_numpy(dtype=int)
    keep = ~best.removed_mask
    pred[pos_idx[keep]] = 1
    out = df.copy()
    out["exp013b_r1_pred"] = pred
    out["exp013b_r1_removed_by_veto"] = 0
    out.loc[pos_idx[best.removed_mask], "exp013b_r1_removed_by_veto"] = 1
    return out, metrics(out["is_fraud"].to_numpy(dtype=int), pred)


def make_report(summary: dict[str, Any], champ_rules: pd.DataFrame) -> str:
    lines = [
        "# EXP-013B-R1 - Irreducible FP Frontier SAFE", "",
        "## Resultado", f"- Status: `{summary['objective_status']}`", f"- Target recall: {summary['target_recall']}",
        f"- Base shadow: TP={summary['base_metrics']['tp']}, FP={summary['base_metrics']['fp']}, FN={summary['base_metrics']['fn']}, recall={summary['base_metrics']['recall']}",
        f"- Champion: TP={summary['champion_metrics']['tp']}, FP={summary['champion_metrics']['fp']}, FN={summary['champion_metrics']['fn']}, recall={summary['champion_metrics']['recall']}, precision={summary['champion_metrics']['precision']}",
        f"- FP removidos: {summary['fp_removed_vs_base']}", f"- TP perdidos: {summary['tp_lost_vs_base']}", "",
        "## Regras do campeao",
    ]
    if champ_rules.empty:
        lines.append("Nenhuma regra selecionada.")
    else:
        lines.append(champ_rules[["family", "description", "tp_loss", "fp_removed"]].to_markdown(index=False))
    lines.extend(["", "## Interpretacao", "Este e o minimo de FP encontrado pela busca segura R1 dentro do espaco controlado de vetos estatisticos. Antes de promover, validar em EXP-013C fora da mesma amostra."])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--beam-width", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-fp-removed", type=int, default=10)
    parser.add_argument("--min-segment-n", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    t0 = time.perf_counter()
    input_path = Path(args.input)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    log("=" * 80)
    log("EXP-013B-R1 - Irreducible FP Frontier SAFE")
    log("=" * 80)
    log(f"Input: {input_path}")
    log(f"Output: {outdir}")
    log(f"Params: target_recall={args.target_recall} max_candidates={args.max_candidates} beam_width={args.beam_width} max_depth={args.max_depth} timeout={args.timeout_seconds}s")
    if not input_path.exists():
        raise FileNotFoundError(f"Input nao encontrado: {input_path}")
    df = normalize_columns(pd.read_csv(input_path, low_memory=False)).reset_index(drop=True)
    df["_orig_index"] = np.arange(len(df))
    y_all = df["is_fraud"].to_numpy(dtype=int)
    base_pred = df["shadow_exp012d_flagged"].to_numpy(dtype=int)
    base_m = metrics(y_all, base_pred)
    total_frauds = int(y_all.sum())
    min_tp_required = int(math.ceil(args.target_recall * total_frauds))
    max_fn_allowed = total_frauds - min_tp_required
    df_pos = df[df["shadow_exp012d_flagged"] == 1].copy().reset_index(drop=True)
    y_pos = df_pos["is_fraud"].to_numpy(dtype=int)
    base_tp = int(y_pos.sum())
    base_fp = int((1 - y_pos).sum())
    max_tp_loss = max(0, base_tp - min_tp_required)
    log(f"Rows={len(df)} Frauds={total_frauds} Normals={len(df) - total_frauds}")
    log(f"Base shadow: TP={base_m['tp']} FP={base_m['fp']} FN={base_m['fn']} recall={base_m['recall']}")
    log(f"Target recall={args.target_recall} => min_tp_required={min_tp_required} max_fn_allowed={max_fn_allowed} max_tp_loss_from_shadow={max_tp_loss}")
    rules = generate_rules(df_pos, y_pos, max_tp_loss, args.min_fp_removed, args.min_segment_n)
    if not rules:
        raise RuntimeError("Nenhuma regra candidata foi gerada com os limites atuais.")
    rdf = rules_df(rules)
    rdf.to_csv(outdir / "01_candidate_veto_rules.csv", index=False)
    single = rdf.copy()
    single["remaining_tp_if_single"] = base_tp - single["tp_loss"]
    single["remaining_fp_if_single"] = base_fp - single["fp_removed"]
    single["recall_total_if_single"] = single["remaining_tp_if_single"] / max(total_frauds, 1)
    single["precision_total_if_single"] = single["remaining_tp_if_single"] / (single["remaining_tp_if_single"] + single["remaining_fp_if_single"])
    single.to_csv(outdir / "02_single_rule_metrics.csv", index=False)
    frontier, best = run_beam(rules, y_pos, max_tp_loss, args.max_candidates, args.beam_width, args.max_depth, args.timeout_seconds, outdir)
    frontier["remaining_tp"] = base_tp - frontier["tp_loss"]
    frontier["remaining_fp"] = base_fp - frontier["fp_removed"]
    frontier["fn_total"] = total_frauds - frontier["remaining_tp"]
    frontier["recall_total"] = frontier["remaining_tp"] / max(total_frauds, 1)
    frontier["precision_total"] = frontier["remaining_tp"] / (frontier["remaining_tp"] + frontier["remaining_fp"])
    frontier = frontier.sort_values(["remaining_fp", "tp_loss", "n_rules"], ascending=[True, True, True]).reset_index(drop=True)
    frontier.to_csv(outdir / "03_irreducible_frontier.csv", index=False)
    champ_rules = rdf[rdf["rule_index"].isin(list(best.rule_indices))].copy()
    champ_rules.to_csv(outdir / "04_champion_policy_rules.csv", index=False)
    preds, champ_m = apply_best(df, df_pos, best)
    preds.to_csv(outdir / "05_champion_predictions.csv", index=False)
    preds[(preds["is_fraud"] == 1) & (preds["exp013b_r1_pred"] == 0)].to_csv(outdir / "06_champion_false_negatives.csv", index=False)
    preds[(preds["is_fraud"] == 0) & (preds["exp013b_r1_pred"] == 1)].to_csv(outdir / "07_champion_false_positives.csv", index=False)
    preds[(preds["is_fraud"] == 0) & (preds["exp013b_r1_removed_by_veto"] == 1)].to_csv(outdir / "08_removed_false_positives.csv", index=False)
    fp_removed = int(base_m["fp"] - champ_m["fp"])
    tp_lost = int(base_m["tp"] - champ_m["tp"])
    status = "TARGET_RECALL_MET" if champ_m["recall"] >= args.target_recall else "TARGET_RECALL_NOT_MET"
    status += "_FP_REDUCED" if fp_removed > 0 else "_FP_NOT_REDUCED"
    summary = {
        "experiment": "EXP-013B-R1", "status": "DONE", "safe_version": True, "objective_status": status,
        "input_path": str(input_path), "n_rows": int(len(df)), "total_frauds": total_frauds,
        "target_recall": args.target_recall, "min_tp_required": min_tp_required, "max_fn_allowed": max_fn_allowed,
        "max_tp_loss_from_shadow_allowed": max_tp_loss, "base_metrics": base_m,
        "base_shadow_positive_tp": base_tp, "base_shadow_positive_fp": base_fp,
        "n_candidate_rules": int(len(rdf)),
        "search_params": {"max_candidates": args.max_candidates, "beam_width": args.beam_width, "max_depth": args.max_depth, "min_fp_removed": args.min_fp_removed, "min_segment_n": args.min_segment_n, "timeout_seconds": args.timeout_seconds},
        "champion_metrics": champ_m, "champion_rule_count": int(len(champ_rules)),
        "champion_rule_ids": champ_rules["rule_id"].tolist() if not champ_rules.empty else [],
        "fp_removed_vs_base": fp_removed, "tp_lost_vs_base": tp_lost,
        "remaining_fp_minimum_found": int(champ_m["fp"]), "remaining_fn": int(champ_m["fn"]),
        "elapsed_seconds": round(time.perf_counter() - t0, 2), "output_dir": str(outdir),
    }
    dump_json(summary, outdir / "00_run_summary.json")
    dump_json({
        "experiment": "EXP-013B-R1", "policy_name": "safe_statistical_veto_frontier",
        "target_recall": args.target_recall, "objective_status": status,
        "base_metrics": base_m, "champion_metrics": champ_m,
        "rules": champ_rules[["rule_id", "family", "description", "params_json", "tp_loss", "fp_removed"]].to_dict(orient="records") if not champ_rules.empty else [],
        "notes": ["Aplicar como camada de veto apos predição positiva do EXP-012D.", "Validar em EXP-013C antes de qualquer promocao."],
    }, outdir / "10_policy_artifact.json")
    (outdir / "09_irreducible_analysis.md").write_text(make_report(summary, champ_rules), encoding="utf-8")
    log("")
    log("=" * 80)
    log("EXP-013B-R1 CONCLUIDO")
    log("=" * 80)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in ["00_run_summary.json", "01_candidate_veto_rules.csv", "02_single_rule_metrics.csv", "03_irreducible_frontier.csv", "04_champion_policy_rules.csv", "05_champion_predictions.csv", "09_irreducible_analysis.md", "10_policy_artifact.json"]:
        log(f"  {outdir / p}")


if __name__ == "__main__":
    main()
