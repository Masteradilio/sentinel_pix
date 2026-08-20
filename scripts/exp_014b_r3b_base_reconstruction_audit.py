#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3B — Base Reconstruction Audit

Objetivo:
  Recuperar de forma reprodutível a base do champion EXP-013K/013L:

      pred_STRICT_RECALL95_SAFE_ONLY / exp013k_base_pred

  e gerar uma versão expandida com essa base aplicada em:

      dados/exp014b_r3b_expanded_with_base.csv

Por que existe:
  O EXP-014B-R3A reproduziu o champion pequeno, mas bloqueou corretamente o
  replay expandido porque a base original foi recuperada apenas como alias:

      exp013k_base_pred

  Essa coluna existe no artefato pequeno, mas não no dataset expandido.

  Agora precisamos descobrir uma receita portável para reconstruir essa base,
  ou registrar que ela não é recuperável com os artefatos atuais.

O que este script faz:
  1. Inventaria artefatos EXP-013* que contenham pred_STRICT_RECALL95_SAFE_ONLY
     ou exp013k_base_pred.
  2. Busca no código fonte menções a pred_STRICT_RECALL95_SAFE_ONLY,
     exp013k_base_pred e residual_fp_mined_tp0_policy.
  3. Testa receitas portáveis:
       - threshold simples em scores existentes no expandido;
       - expressões simples com threshold + condição categórica;
       - surrogate decision-tree treinado para reproduzir a base.
  4. Seleciona a melhor receita se passar fidelidade mínima.
  5. Aplica a receita no dataset expandido e grava a coluna:
       pred_STRICT_RECALL95_SAFE_ONLY
       exp013k_base_pred
  6. Reaplica a política congelada EXP-013K sobre o expandido, se possível.
  7. Gera métricas, Wilson, blocos temporais e relatório.

Importante:
  - Se a receita selecionada for surrogate, trate como reconstrução diagnóstica
    até congelar/validar no EXP-014B-R3C.
  - Este script NÃO minera novas regras. Ele apenas reconstrói a base.
  - Se nenhuma receita passar o mínimo, ele encerra sem crash e explica por quê.

Uso padrão:
  python scripts/exp_014b_r3b_base_reconstruction_audit.py

Mais tolerante para diagnóstico:
  python scripts/exp_014b_r3b_base_reconstruction_audit.py --min-fidelity 0.98 --allow-surrogate

Mais rígido:
  python scripts/exp_014b_r3b_base_reconstruction_audit.py --min-fidelity 0.999 --require-exact-or-threshold

Depois, se gerar dados/exp014b_r3b_expanded_with_base.csv:
  python scripts/exp_014b_r3a_champion_replay_expanded.py --expanded-input dados\\exp014b_r3b_expanded_with_base.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "dados").exists() else Path.cwd()

DEFAULT_EXPANDED_INPUT = PROJECT_ROOT / "dados" / "exp014a_expanded_scored_input.csv"
DEFAULT_EXPANDED_OUTPUT = PROJECT_ROOT / "dados" / "exp014b_r3b_expanded_with_base.csv"
DEFAULT_POLICY = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013K" / "12_policy_artifact.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3B"

TARGET_COLS = [
    "pred_STRICT_RECALL95_SAFE_ONLY",
    "exp013k_base_pred",
]

FINAL_POLICY_COL = "exp013k_residual_fp_pred"

SOURCE_SEARCH_ROOTS = [
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "backend",
    PROJECT_ROOT / "resultados" / "experimentos",
    PROJECT_ROOT / "docs",
]

PRIOR_GLOBS = [
    "resultados/experimentos/EXP-013*/**/*.csv",
    "resultados/experimentos/EXP-013*/**/*.json",
]

SCORE_COLS = [
    "lgbm_r4_score",
    "r4_score",
    "lgbm_mapped",
    "lgbm_raw",
    "score_final",
    "if_percentile",
    "if_percentile_x",
    "if_percentile_y",
]

CATEGORICAL_COLS = [
    "value_band",
    "ds_tipo_chave_norm",
    "periodo_dia",
    "first_receiver_flag_real",
    "mbk_available_flag",
    "module_quiet",
    "lgbm_bin",
    "if_bin",
    "score_bin",
    "vl_bin",
    "ratio_bin",
    "qtd_rec_bin",
    "valor_rec_bin",
]

NUMERIC_SURROGATE_COLS = [
    "lgbm_r4_score",
    "r4_score",
    "lgbm_mapped",
    "lgbm_raw",
    "score_final",
    "if_percentile",
    "if_percentile_x",
    "if_percentile_y",
    "vl_pix",
    "ratio_valor_media_pagador_90d",
    "qtd_pix_recebidos_180d",
    "valor_total_recebido_180d",
    "first_receiver_flag_real",
    "mbk_available_flag",
    "runtime_flagged",
]

TERMS_TO_SCAN = [
    "pred_STRICT_RECALL95_SAFE_ONLY",
    "exp013k_base_pred",
    "residual_fp_mined_tp0_policy",
    "STRICT_RECALL95",
    "HIGH_RECALL_95",
]


@dataclass
class RecipeResult:
    recipe_type: str
    name: str
    recipe: dict[str, Any]
    match_rate_all: float
    match_rate_test: float | None
    target_positive_rate: float
    pred_positive_rate_all: float
    accepted: bool
    reason: str


def log(msg: str) -> None:
    print(msg, flush=True)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def normalize_columns(df: pd.DataFrame, require_label: bool = False) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().split(".")[-1] for c in df.columns]

    if "transaction_id" not in df.columns and "cd_pix" in df.columns:
        df["transaction_id"] = df["cd_pix"]
    if "event_datetime" not in df.columns and "dt_pix" in df.columns:
        df["event_datetime"] = df["dt_pix"]

    if "is_fraud" in df.columns:
        df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)
    elif require_label:
        raise RuntimeError("Coluna is_fraud ausente.")

    if "decisao" in df.columns and "runtime_flagged" not in df.columns:
        df["runtime_flagged"] = df["decisao"].astype(str).str.upper().isin({"CONFIRMAR", "BLOQUEAR"}).astype(int)
    if "runtime_flagged" not in df.columns:
        df["runtime_flagged"] = 0

    for c in TARGET_COLS + [FINAL_POLICY_COL, "runtime_flagged", "exp014a_frozen_pred"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "transaction_id" in df.columns:
        df["transaction_id"] = df["transaction_id"].astype("string").str.strip()

    for c in ["event_datetime", "data_pix"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

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
    df = df.copy()

    if "lgbm_bin" not in df.columns and pick_col(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"]):
        df["lgbm_bin"] = qbin_series(num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0), "lgbm", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1])
    if "if_bin" not in df.columns and pick_col(df, ["if_percentile", "if_percentile_x", "if_percentile_y"]):
        df["if_bin"] = qbin_series(num(df, ["if_percentile", "if_percentile_x", "if_percentile_y"], 0.0), "if", [0.32, 0.5, 0.7, 0.85, 0.95])
    if "score_bin" not in df.columns and "score_final" in df.columns:
        df["score_bin"] = qbin_series(num(df, "score_final", 0.0), "score", [0.5, 1, 2, 3, 5, 10])
    if "vl_bin" not in df.columns and "vl_pix" in df.columns:
        df["vl_bin"] = qbin_series(num(df, "vl_pix", 0.0), "vl", [20, 50, 100, 250, 500, 1000, 5000, 10000])
    if "ratio_bin" not in df.columns and "ratio_valor_media_pagador_90d" in df.columns:
        df["ratio_bin"] = qbin_series(num(df, "ratio_valor_media_pagador_90d", 0.0), "ratio", [0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    if "qtd_rec_bin" not in df.columns and "qtd_pix_recebidos_180d" in df.columns:
        df["qtd_rec_bin"] = qbin_series(num(df, "qtd_pix_recebidos_180d", 0.0), "qtdrec", [0, 1, 2, 5, 10, 20, 50, 100])
    if "valor_rec_bin" not in df.columns and "valor_total_recebido_180d" in df.columns:
        df["valor_rec_bin"] = qbin_series(num(df, "valor_total_recebido_180d", 0.0), "valrec", [0, 100, 500, 1000, 5000, 10000, 25000])

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


def find_target_col(df: pd.DataFrame) -> str | None:
    for c in TARGET_COLS:
        if c in df.columns:
            return c
    return None


def scan_prior_artifacts(project_root: Path, output_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    rows = []
    target_files = []

    seen = set()
    for pattern in PRIOR_GLOBS:
        for p in project_root.glob(pattern):
            if not p.is_file():
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)

            rec = {"path": str(p), "suffix": p.suffix.lower(), "has_target": False, "target_cols": "", "n_rows": None, "error": None}
            try:
                if p.suffix.lower() == ".csv":
                    cols = pd.read_csv(p, nrows=0).columns
                    cols = [str(c).strip().split(".")[-1] for c in cols]
                    target_cols = [c for c in TARGET_COLS if c in cols]
                    rec["has_target"] = bool(target_cols)
                    rec["target_cols"] = "|".join(target_cols)
                    if target_cols:
                        rec["n_rows"] = int(sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore")) - 1)
                        target_files.append(p)
                elif p.suffix.lower() == ".json":
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                    target_cols = [c for c in TARGET_COLS if c in txt]
                    rec["has_target"] = bool(target_cols)
                    rec["target_cols"] = "|".join(target_cols)
            except Exception as exc:
                rec["error"] = str(exc)[:300]
            rows.append(rec)

    inv = pd.DataFrame(rows).sort_values(["has_target", "path"], ascending=[False, True]).reset_index(drop=True)
    inv.to_csv(output_dir / "01_source_inventory.csv", index=False)
    return inv, target_files


def scan_code_origins(project_root: Path, output_dir: Path, max_hits_per_file: int = 12) -> pd.DataFrame:
    rows = []
    exts = {".py", ".md", ".json", ".txt", ".yaml", ".yml"}
    roots = [p for p in SOURCE_SEARCH_ROOTS if p.exists()]

    for root in roots:
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue

            hits = 0
            for i, line in enumerate(txt, start=1):
                if any(term in line for term in TERMS_TO_SCAN):
                    rows.append({
                        "path": str(p),
                        "line": i,
                        "term_hits": "|".join([t for t in TERMS_TO_SCAN if t in line]),
                        "text": line.strip()[:500],
                    })
                    hits += 1
                    if hits >= max_hits_per_file:
                        break

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "02_code_origin_hits.csv", index=False)
    return out


def choose_small_source(target_files: list[Path], explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"small-source não encontrado: {p}")
        return p

    # Prefer EXP-013K selected predictions, then EXP-013J, then any.
    preferences = ["EXP-013K", "07_selected_predictions", "EXP-013J", "06_predictions"]
    def score(p: Path) -> int:
        s = str(p)
        total = 0
        for i, pref in enumerate(preferences):
            if pref in s:
                total += 100 - i
        return total

    if not target_files:
        raise FileNotFoundError("Nenhum CSV EXP-013* com coluna alvo foi encontrado.")
    return sorted(target_files, key=score, reverse=True)[0]


def get_threshold_candidates(s: pd.Series, max_values: int) -> list[float]:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if vals.empty:
        return []
    unique = np.sort(vals.unique())
    if len(unique) > max_values:
        unique = np.sort(vals.quantile(np.linspace(0, 1, max_values)).unique())

    mids = []
    for a, b in zip(unique[:-1], unique[1:]):
        mids.append(float((a + b) / 2.0))
    thresholds = sorted(set([float(x) for x in unique] + mids + [float(vals.min()), float(vals.max())]))
    return thresholds


def evaluate_match(target: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred).astype(int)
    return float((target == pred).mean()), float(pred.mean())


def recipe_to_row(rr: RecipeResult) -> dict[str, Any]:
    return {
        "recipe_type": rr.recipe_type,
        "name": rr.name,
        "match_rate_all": rr.match_rate_all,
        "match_rate_test": rr.match_rate_test,
        "target_positive_rate": rr.target_positive_rate,
        "pred_positive_rate_all": rr.pred_positive_rate_all,
        "accepted": rr.accepted,
        "reason": rr.reason,
        "recipe_json": json.dumps(rr.recipe, ensure_ascii=False),
    }


def candidate_aliases(small: pd.DataFrame, expanded: pd.DataFrame, target: np.ndarray, min_fidelity: float) -> list[RecipeResult]:
    out = []
    for c in small.columns:
        if c not in expanded.columns:
            continue
        if c in TARGET_COLS:
            continue
        vals = pd.to_numeric(small[c], errors="coerce")
        uniq = set(vals.dropna().astype(int).unique()) if vals.notna().any() else set()
        if not uniq.issubset({0, 1}):
            continue
        pred = vals.fillna(0).astype(int).to_numpy()
        match, pos = evaluate_match(target, pred)
        recipe = {"type": "alias_column", "source_col": c}
        out.append(RecipeResult(
            "alias_column", c, recipe, match, None, float(target.mean()), pos,
            match >= min_fidelity, "alias_exists_in_expanded" if match >= min_fidelity else "below_min_fidelity",
        ))
    return out


def candidate_thresholds(small: pd.DataFrame, expanded: pd.DataFrame, target: np.ndarray, min_fidelity: float, max_values: int) -> list[RecipeResult]:
    out = []
    for c in [x for x in SCORE_COLS if x in small.columns and x in expanded.columns]:
        arr = pd.to_numeric(small[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float).to_numpy()
        thresholds = get_threshold_candidates(small[c], max_values)
        for direction in ["ge", "le"]:
            best = None
            for th in thresholds:
                pred = (arr >= th).astype(int) if direction == "ge" else (arr <= th).astype(int)
                match, pos = evaluate_match(target, pred)
                if best is None or match > best[0]:
                    best = (match, pos, th)
            if best:
                match, pos, th = best
                recipe = {"type": "threshold", "score_col": c, "direction": direction, "threshold": float(th)}
                out.append(RecipeResult(
                    "threshold", f"{c}_{direction}_{th:.12g}", recipe, match, None, float(target.mean()), pos,
                    match >= min_fidelity, "portable_threshold" if match >= min_fidelity else "below_min_fidelity",
                ))
    return out


def build_condition_candidates(small: pd.DataFrame, expanded: pd.DataFrame, max_values_per_col: int = 20) -> list[tuple[str, str]]:
    conds = []
    for c in [x for x in CATEGORICAL_COLS if x in small.columns and x in expanded.columns]:
        vals = small[c].astype("string").fillna("<MISSING>").value_counts().head(max_values_per_col).index.tolist()
        for v in vals:
            conds.append((c, str(v)))
    return conds


def eval_condition(df: pd.DataFrame, col: str, val: str) -> np.ndarray:
    return (df[col].astype("string").fillna("<MISSING>").astype(str) == str(val)).to_numpy(dtype=bool)


def candidate_simple_expressions(
    small: pd.DataFrame,
    expanded: pd.DataFrame,
    target: np.ndarray,
    min_fidelity: float,
    max_values: int,
    max_conditions: int,
) -> list[RecipeResult]:
    out = []

    # Use top threshold candidates as base predicates.
    threshold_results = candidate_thresholds(small, expanded, target, min_fidelity=0.0, max_values=max_values)
    threshold_results = sorted(threshold_results, key=lambda r: r.match_rate_all, reverse=True)[:30]
    conds = build_condition_candidates(small, expanded)[:max_conditions]

    for tr in threshold_results:
        rec = tr.recipe
        s = pd.to_numeric(small[rec["score_col"]], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        base = (s >= rec["threshold"]).to_numpy(dtype=bool) if rec["direction"] == "ge" else (s <= rec["threshold"]).to_numpy(dtype=bool)

        for col, val in conds:
            cond = eval_condition(small, col, val)

            for op in ["or", "and", "and_not", "or_not"]:
                if op == "or":
                    pred = base | cond
                elif op == "and":
                    pred = base & cond
                elif op == "and_not":
                    pred = base & (~cond)
                else:
                    pred = base | (~cond)

                match, pos = evaluate_match(target, pred.astype(int))
                recipe = {
                    "type": "simple_expression",
                    "op": op,
                    "threshold_predicate": rec,
                    "condition": {"col": col, "value": val},
                }
                out.append(RecipeResult(
                    "simple_expression", f"{rec['score_col']}_{rec['direction']}_{op}_{col}_{val}",
                    recipe, match, None, float(target.mean()), pos,
                    match >= min_fidelity, "portable_expression" if match >= min_fidelity else "below_min_fidelity",
                ))

    # Return only useful top expressions to avoid huge output.
    out.sort(key=lambda r: r.match_rate_all, reverse=True)
    return out[:500]


def prep_surrogate_X(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in num_cols:
        X[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    for c in cat_cols:
        X[c] = df[c].astype("string").fillna("<MISSING>").astype(str)
    return X



def train_surrogates(
    small: pd.DataFrame,
    expanded: pd.DataFrame,
    target: np.ndarray,
    min_fidelity: float,
    max_depth: int,
    min_leaf_values: list[int],
    seed: int,
    output_dir: Path,
) -> tuple[list[RecipeResult], Any | None, dict[str, Any] | None, pd.DataFrame]:
    """
    Versão robusta/densa do surrogate.

    Evita o erro:
      scipy.sparse does not support dtype object

    Estratégia:
      - numéricos -> float64 denso;
      - categóricos -> OneHotEncoder denso float64;
      - ColumnTransformer -> saída densa;
      - cada árvore falha isoladamente, sem derrubar o experimento.
    """
    from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

    def _to_float_array_for_surrogate(X):
        return np.asarray(X, dtype=np.float64)

    def _make_dense_ohe():
        try:
            return OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=2,
                sparse_output=False,
                dtype=np.float64,
            )
        except TypeError:
            return OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=2,
                sparse=False,
                dtype=np.float64,
            )

    num_cols = [c for c in NUMERIC_SURROGATE_COLS if c in small.columns and c in expanded.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in small.columns and c in expanded.columns]

    X = prep_surrogate_X(small, num_cols, cat_cols)
    y = target.astype(int)

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            FunctionTransformer(_to_float_array_for_surrogate, validate=False),
            num_cols,
        ))
    if cat_cols:
        transformers.append(("cat", _make_dense_ohe(), cat_cols))

    if not transformers:
        empty = pd.DataFrame([{
            "status": "NO_COMMON_FEATURES_FOR_SURROGATE",
            "error": "Sem colunas comuns entre small e expanded para surrogate.",
        }])
        empty.to_csv(output_dir / "04_surrogate_fidelity.csv", index=False)
        return [], None, None, empty

    results = []
    best_model = None
    best_meta = None
    rows = []

    try:
        stratify = y if len(np.unique(y)) == 2 and min(np.bincount(y)) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=seed,
            stratify=stratify,
        )
    except Exception as exc:
        rows.append({
            "status": "TRAIN_TEST_SPLIT_FAILED",
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            "num_cols": "|".join(num_cols),
            "cat_cols": "|".join(cat_cols),
        })
        fidelity_df = pd.DataFrame(rows)
        fidelity_df.to_csv(output_dir / "04_surrogate_fidelity.csv", index=False)
        return [], None, None, fidelity_df

    for depth in range(2, max_depth + 1):
        for min_leaf in min_leaf_values:
            clf = Pipeline([
                ("prep", ColumnTransformer(
                    transformers=transformers,
                    remainder="drop",
                    sparse_threshold=0.0,
                )),
                ("tree", DecisionTreeClassifier(
                    max_depth=depth,
                    min_samples_leaf=min_leaf,
                    random_state=seed,
                )),
            ])

            try:
                clf.fit(X_train, y_train)
                pred_train = clf.predict(X_train)
                pred_test = clf.predict(X_test)
                pred_all = clf.predict(X)
            except Exception as exc:
                rows.append({
                    "max_depth": depth,
                    "min_samples_leaf": min_leaf,
                    "match_rate_all": None,
                    "match_rate_test": None,
                    "match_rate_train": None,
                    "target_positive_rate": float(np.mean(y)),
                    "pred_positive_rate_all": None,
                    "num_cols": "|".join(num_cols),
                    "cat_cols": "|".join(cat_cols),
                    "status": "FIT_FAILED",
                    "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                })
                continue

            match_train = float((pred_train == y_train).mean())
            match_test = float((pred_test == y_test).mean())
            match_all = float((pred_all == y).mean())
            pos_all = float(np.mean(pred_all))
            target_pos = float(np.mean(y))

            meta = {
                "type": "decision_tree_surrogate",
                "num_cols": num_cols,
                "cat_cols": cat_cols,
                "max_depth": depth,
                "min_samples_leaf": min_leaf,
                "match_rate_all": match_all,
                "match_rate_test": match_test,
                "target_positive_rate": target_pos,
                "pred_positive_rate_all": pos_all,
                "surrogate_preprocessor": "dense_float64_column_transformer",
            }

            rows.append({
                **meta,
                "match_rate_train": match_train,
                "status": "OK",
                "error": None,
            })

            rr = RecipeResult(
                "decision_tree_surrogate",
                f"tree_depth{depth}_leaf{min_leaf}",
                meta,
                match_all,
                match_test,
                target_pos,
                pos_all,
                (match_all >= min_fidelity and match_test >= min_fidelity),
                "surrogate_tree" if (match_all >= min_fidelity and match_test >= min_fidelity) else "below_min_fidelity",
            )
            results.append(rr)

            current_rank = (
                match_test,
                match_all,
                -abs(pos_all - target_pos),
                -depth,
                -min_leaf,
            )
            if best_meta is None or current_rank > best_meta["rank"]:
                best_model = clf
                best_meta = {"rank": current_rank, "recipe": meta, "name": rr.name}

    fidelity_df = pd.DataFrame(rows)
    if not fidelity_df.empty:
        sort_cols = [c for c in ["match_rate_test", "match_rate_all"] if c in fidelity_df.columns]
        if sort_cols:
            fidelity_df = fidelity_df.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last").reset_index(drop=True)
    fidelity_df.to_csv(output_dir / "04_surrogate_fidelity.csv", index=False)

    if best_model is not None:
        try:
            import joblib
            joblib.dump(best_model, output_dir / "base_reconstruction_surrogate.joblib")
            tree = best_model.named_steps["tree"]
            prep = best_model.named_steps["prep"]
            try:
                feature_names = prep.get_feature_names_out()
                tree_txt = export_text(tree, feature_names=[str(x) for x in feature_names])
            except Exception:
                tree_txt = export_text(tree)
            (output_dir / "base_reconstruction_surrogate_tree.txt").write_text(tree_txt, encoding="utf-8")
        except Exception as exc:
            (output_dir / "surrogate_save_warning.txt").write_text(
                f"{type(exc).__name__}: {str(exc)}",
                encoding="utf-8",
            )

    best_recipe = best_meta["recipe"] if best_meta else None
    return results, best_model, best_recipe, fidelity_df


def apply_recipe(df: pd.DataFrame, recipe: dict[str, Any], model: Any | None = None) -> np.ndarray:
    rtype = recipe.get("type")

    if rtype == "alias_column":
        c = recipe["source_col"]
        if c not in df.columns:
            raise RuntimeError(f"Alias ausente: {c}")
        return pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).to_numpy()

    if rtype == "threshold":
        c = recipe["score_col"]
        if c not in df.columns:
            raise RuntimeError(f"Score ausente: {c}")
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        if recipe["direction"] == "ge":
            return (s >= float(recipe["threshold"])).astype(int).to_numpy()
        return (s <= float(recipe["threshold"])).astype(int).to_numpy()

    if rtype == "simple_expression":
        tp = recipe["threshold_predicate"]
        base = apply_recipe(df, tp).astype(bool)
        cond_spec = recipe["condition"]
        cond = eval_condition(df, cond_spec["col"], cond_spec["value"])
        op = recipe["op"]
        if op == "or":
            return (base | cond).astype(int)
        if op == "and":
            return (base & cond).astype(int)
        if op == "and_not":
            return (base & (~cond)).astype(int)
        if op == "or_not":
            return (base | (~cond)).astype(int)
        raise RuntimeError(f"Operador desconhecido: {op}")

    if rtype == "decision_tree_surrogate":
        if model is None:
            raise RuntimeError("Recipe surrogate exige model carregado no mesmo run.")
        X = prep_surrogate_X(df, recipe["num_cols"], recipe["cat_cols"])
        return model.predict(X).astype(int)

    raise RuntimeError(f"Recipe desconhecida: {recipe}")


def load_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Policy artifact não encontrado: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "selected_rules" not in obj:
        raise RuntimeError("Policy artifact sem selected_rules.")
    return obj


def parse_params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(str(raw))
    except Exception:
        return {}


def rule_mask(df: pd.DataFrame, rule: dict[str, Any], current_pred: np.ndarray) -> np.ndarray:
    params = parse_params(rule.get("params_json", {}))
    if not params and isinstance(rule.get("params"), dict):
        params = rule["params"]

    cols = params.get("combo_cols", [])
    vals = params.get("combo_values", [])
    require_module_quiet = bool(params.get("require_module_quiet", False))

    if not cols:
        desc = str(rule.get("description", ""))
        cols, vals = [], []
        for part in desc.split(" AND "):
            if "=" in part:
                c, v = part.split("=", 1)
                cols.append(c.strip())
                vals.append(v.strip())

    if not cols:
        raise RuntimeError(f"Não consegui parsear regra: {rule}")

    mask = np.ones(len(df), dtype=bool)
    for c, v in zip(cols, vals):
        if c not in df.columns:
            raise RuntimeError(f"Coluna da regra ausente: {c}")
        mask = mask & (df[c].astype("string").fillna("<MISSING>").astype(str).to_numpy() == str(v))

    if require_module_quiet:
        if "module_quiet" not in df.columns:
            raise RuntimeError("Regra exige module_quiet.")
        mask = mask & (df["module_quiet"].astype(str).to_numpy() == "module_quiet")

    return mask & (current_pred.astype(int) == 1)


def apply_policy(df: pd.DataFrame, policy: dict[str, Any], base_pred: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    y = df["is_fraud"].to_numpy(dtype=int)
    pred = base_pred.astype(int).copy()
    rows = []

    for idx, rule in enumerate(policy.get("selected_rules", [])):
        mask = rule_mask(df, rule, pred)
        tp_loss = int(((y == 1) & mask).sum())
        fp_removed = int(((y == 0) & mask).sum())
        pred[mask] = 0
        rows.append({
            "rule_index": idx,
            "description": rule.get("description"),
            "tp_loss": tp_loss,
            "fp_removed": fp_removed,
            "n_removed": int(mask.sum()),
            "params_json": rule.get("params_json"),
        })
    return pred, pd.DataFrame(rows)


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


def block_metrics(df: pd.DataFrame, pred: np.ndarray, blocks: pd.Series, policy_name: str) -> pd.DataFrame:
    y = df["is_fraud"].to_numpy(dtype=int)
    rows = []
    bvals = blocks.to_numpy()
    for b in sorted(blocks.dropna().unique()):
        idx = bvals == b
        part = df.loc[idx]
        rows.append({
            "policy_name": policy_name,
            "block": int(b),
            "n_rows": int(len(part)),
            "n_frauds": int(part["is_fraud"].sum()),
            **compute_metrics(y[idx], pred[idx]),
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


def make_report(summary: dict[str, Any], recipes: pd.DataFrame, replay: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014B-R3B — Base Reconstruction Audit")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Selected recipe type: `{summary.get('selected_recipe_type')}`")
    lines.append(f"- Expanded output built: `{summary.get('expanded_output_built')}`")
    lines.append("")
    lines.append("## Melhor receita")
    if summary.get("selected_recipe"):
        lines.append("```json")
        lines.append(json.dumps(summary["selected_recipe"], ensure_ascii=False, indent=2))
        lines.append("```")
    else:
        lines.append("Nenhuma receita aceita.")
    lines.append("")
    lines.append("## Top receitas")
    if recipes.empty:
        lines.append("Sem receitas.")
    else:
        cols = ["recipe_type", "name", "match_rate_all", "match_rate_test", "pred_positive_rate_all", "accepted", "reason"]
        show = [c for c in cols if c in recipes.columns]
        lines.append(recipes[show].head(20).to_markdown(index=False))
    lines.append("")
    lines.append("## Replay expandido")
    if replay.empty:
        lines.append("Não executado.")
    else:
        lines.append(replay.to_markdown(index=False))
    lines.append("")
    lines.append("## Próximo passo")
    if summary.get("expanded_output_built"):
        lines.append("Rodar:")
        lines.append("```powershell")
        lines.append("python scripts\\exp_014b_r3a_champion_replay_expanded.py --expanded-input dados\\exp014b_r3b_expanded_with_base.csv")
        lines.append("```")
    else:
        lines.append("Localizar a origem exata de `exp013k_base_pred` nos scripts/artefatos ou reduzir o critério de fidelidade apenas para diagnóstico.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--expanded-input", default=str(DEFAULT_EXPANDED_INPUT))
    parser.add_argument("--expanded-output", default=str(DEFAULT_EXPANDED_OUTPUT))
    parser.add_argument("--policy-artifact", default=str(DEFAULT_POLICY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--small-source", default=None)
    parser.add_argument("--min-fidelity", type=float, default=0.995)
    parser.add_argument("--require-exact-or-threshold", action="store_true")
    parser.add_argument("--allow-surrogate", action="store_true", default=True)
    parser.add_argument("--max-threshold-values", type=int, default=2000)
    parser.add_argument("--max-expression-conditions", type=int, default=120)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument("--min-leaf-values", default="1,2,5,10")
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--time-blocks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-write-expanded", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expanded_path = Path(args.expanded_input)
    if not expanded_path.exists():
        raise FileNotFoundError(f"expanded-input não encontrado: {expanded_path}")

    log("=" * 80)
    log("EXP-014B-R3B — Base Reconstruction Audit")
    log("=" * 80)
    log(f"Expanded input: {expanded_path}")

    expanded = add_bins_and_guards(normalize_columns(pd.read_csv(expanded_path, low_memory=False), require_label=True))
    policy = load_policy(Path(args.policy_artifact))

    # 1. Inventory + code scan.
    inv, target_files = scan_prior_artifacts(PROJECT_ROOT, output_dir)
    code_hits = scan_code_origins(PROJECT_ROOT, output_dir)

    small_path = choose_small_source(target_files, args.small_source)
    small = add_bins_and_guards(normalize_columns(pd.read_csv(small_path, low_memory=False), require_label=True))
    target_col = find_target_col(small)
    if target_col is None:
        raise RuntimeError(f"Fonte pequena não contém target base: {small_path}")

    target = pd.to_numeric(small[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    y_small = small["is_fraud"].to_numpy(dtype=int)
    y_exp = expanded["is_fraud"].to_numpy(dtype=int)

    log(f"Small source: {small_path}")
    log(f"Target base col: {target_col}")

    # 2. Candidate recipes.
    recipe_results: list[RecipeResult] = []
    recipe_results.extend(candidate_aliases(small, expanded, target, args.min_fidelity))
    recipe_results.extend(candidate_thresholds(small, expanded, target, args.min_fidelity, args.max_threshold_values))
    recipe_results.extend(candidate_simple_expressions(small, expanded, target, args.min_fidelity, args.max_threshold_values, args.max_expression_conditions))

    surrogate_results = []
    best_model = None
    best_surrogate_recipe = None
    fidelity_df = pd.DataFrame()

    if args.allow_surrogate and not args.require_exact_or_threshold:
        min_leaf_values = [int(x.strip()) for x in str(args.min_leaf_values).split(",") if x.strip()]
        surrogate_results, best_model, best_surrogate_recipe, fidelity_df = train_surrogates(
            small=small,
            expanded=expanded,
            target=target,
            min_fidelity=args.min_fidelity,
            max_depth=args.max_tree_depth,
            min_leaf_values=min_leaf_values,
            seed=args.seed,
            output_dir=output_dir,
        )
        recipe_results.extend(surrogate_results)
    else:
        pd.DataFrame().to_csv(output_dir / "04_surrogate_fidelity.csv", index=False)

    recipes_df = pd.DataFrame([recipe_to_row(r) for r in recipe_results])
    if not recipes_df.empty:
        recipes_df = recipes_df.sort_values(
            ["accepted", "match_rate_test", "match_rate_all", "recipe_type"],
            ascending=[False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)
    recipes_df.to_csv(output_dir / "03_base_recipe_candidates.csv", index=False)

    accepted = [r for r in recipe_results if r.accepted]
    # Prefer exact/threshold/expression over surrogate unless surrogate is the only accepted.
    def recipe_rank(r: RecipeResult):
        type_rank = {"alias_column": 0, "threshold": 1, "simple_expression": 2, "decision_tree_surrogate": 3}.get(r.recipe_type, 9)
        mt = r.match_rate_test if r.match_rate_test is not None else r.match_rate_all
        return (type_rank, -mt, -r.match_rate_all, abs(r.pred_positive_rate_all - r.target_positive_rate))

    selected = sorted(accepted, key=recipe_rank)[0] if accepted else None

    selected_model = None
    if selected and selected.recipe_type == "decision_tree_surrogate":
        selected_model = best_model

    expanded_output_built = False
    replay_df = pd.DataFrame()
    rule_impact = pd.DataFrame()
    block_df = pd.DataFrame()
    wilson_df = pd.DataFrame()
    selected_recipe = selected.recipe if selected else None

    small_recon_metrics = pd.DataFrame()
    expanded_base_metrics = pd.DataFrame()

    if selected:
        log(f"Selected recipe: {selected.recipe_type} {selected.name} match={selected.match_rate_all:.6f}")

        small_pred = apply_recipe(small, selected.recipe, selected_model)
        small_recon_metrics = pd.DataFrame([
            {"policy_name": "SMALL_TARGET_BASE", **compute_metrics(y_small, target)},
            {"policy_name": "SMALL_RECONSTRUCTED_BASE", **compute_metrics(y_small, small_pred)},
        ])
        small_recon_metrics["base_match_rate"] = [1.0, float((small_pred == target).mean())]
        small_recon_metrics.to_csv(output_dir / "05_small_reconstruction_metrics.csv", index=False)

        expanded_base_pred = apply_recipe(expanded, selected.recipe, selected_model)
        expanded["pred_STRICT_RECALL95_SAFE_ONLY"] = expanded_base_pred.astype(int)
        expanded["exp013k_base_pred"] = expanded_base_pred.astype(int)

        expanded_base_metrics = pd.DataFrame([
            {"policy_name": "EXPANDED_RECONSTRUCTED_BASE", **compute_metrics(y_exp, expanded_base_pred)}
        ])
        expanded_base_metrics.to_csv(output_dir / "06_expanded_base_metrics.csv", index=False)

        frozen_pred, rule_impact = apply_policy(expanded, policy, expanded_base_pred)
        expanded["exp013k_residual_fp_pred"] = frozen_pred.astype(int)

        replay_rows = [
            {"policy_name": "EXPANDED_RECONSTRUCTED_BASE", **compute_metrics(y_exp, expanded_base_pred)},
            {"policy_name": "EXPANDED_RECONSTRUCTED_BASE_PLUS_EXP013K", **compute_metrics(y_exp, frozen_pred)},
        ]
        for runtime_col in ["exp014a_frozen_pred"]:
            if runtime_col in expanded.columns:
                replay_rows.insert(0, {"policy_name": f"RUNTIME_FINAL_{runtime_col}", **compute_metrics(y_exp, expanded[runtime_col].to_numpy(dtype=int))})
                break
        replay_df = pd.DataFrame(replay_rows)
        replay_df.to_csv(output_dir / "07_expanded_replay_metrics.csv", index=False)
        rule_impact.to_csv(output_dir / "08_rule_impact_expanded.csv", index=False)

        blocks = make_time_blocks(expanded, args.time_blocks)
        block_df = pd.concat([
            block_metrics(expanded, expanded_base_pred, blocks, "EXPANDED_RECONSTRUCTED_BASE"),
            block_metrics(expanded, frozen_pred, blocks, "EXPANDED_RECONSTRUCTED_BASE_PLUS_EXP013K"),
        ], ignore_index=True)
        block_df.to_csv(output_dir / "09_time_block_metrics.csv", index=False)

        final_m = compute_metrics(y_exp, frozen_pred)
        total_frauds = int(y_exp.sum())
        min_tp_required = int(math.ceil(args.target_recall * total_frauds))
        wl, wh = wilson_ci(final_m["tp"], total_frauds)
        wilson_df = pd.DataFrame([{
            "metric": "recall",
            "successes_tp": final_m["tp"],
            "n_frauds": total_frauds,
            "point_estimate": final_m["recall"],
            "wilson_low": wl,
            "wilson_high": wh,
            "target_recall": args.target_recall,
            "min_tp_required": min_tp_required,
            "tp_buffer_vs_target": final_m["tp"] - min_tp_required,
            "wilson_low_ge_target": bool(wl >= args.target_recall),
        }])
        wilson_df.to_csv(output_dir / "10_wilson_recall_ci.csv", index=False)

        expanded[(expanded["is_fraud"] == 1) & (expanded["exp013k_residual_fp_pred"] == 0)].to_csv(output_dir / "11_false_negatives.csv", index=False)
        fp = expanded[(expanded["is_fraud"] == 0) & (expanded["exp013k_residual_fp_pred"] == 1)].copy()
        if len(fp) > 5000:
            fp = fp.sample(5000, random_state=args.seed)
        fp.to_csv(output_dir / "12_false_positives_sample.csv", index=False)

        if not args.no_write_expanded:
            out_path = Path(args.expanded_output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            expanded.to_csv(out_path, index=False)
            expanded_output_built = True
    else:
        pd.DataFrame().to_csv(output_dir / "05_small_reconstruction_metrics.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "06_expanded_base_metrics.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "07_expanded_replay_metrics.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "08_rule_impact_expanded.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "09_time_block_metrics.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "10_wilson_recall_ci.csv", index=False)

    if selected:
        final_metrics = replay_df.iloc[-1].to_dict() if not replay_df.empty else {}
        final_recall = float(final_metrics.get("recall", 0.0))
        objective_status = "DONE_BASE_RECIPE_ACCEPTED"
        objective_status += "_SURROGATE" if selected.recipe_type == "decision_tree_surrogate" else "_PORTABLE"
        objective_status += "_EXPANDED_BUILT" if expanded_output_built else "_EXPANDED_NOT_BUILT"
        objective_status += "_REPLAY_TARGET_MET" if final_recall >= args.target_recall else "_REPLAY_TARGET_NOT_MET"
    else:
        objective_status = "DONE_BASE_RECIPE_NOT_RECOVERED"

    artifact = {
        "experiment": "EXP-014B-R3B",
        "policy_name": "base_reconstruction_audit",
        "objective_status": objective_status,
        "small_source": str(small_path),
        "target_col": target_col,
        "selected_recipe": selected_recipe,
        "selected_recipe_type": selected.recipe_type if selected else None,
        "selected_recipe_match_all": selected.match_rate_all if selected else None,
        "selected_recipe_match_test": selected.match_rate_test if selected else None,
        "expanded_output": str(args.expanded_output),
        "expanded_output_built": expanded_output_built,
        "notes": [
            "Reconstructs pred_STRICT_RECALL95_SAFE_ONLY/exp013k_base_pred for expanded dataset.",
            "If selected recipe is surrogate, treat as diagnostic until frozen validation.",
            "No residual mining is performed here."
        ],
    }
    dump_json(artifact, output_dir / "13_base_recipe_artifact.json")

    summary = {
        "experiment": "EXP-014B-R3B",
        "status": "DONE",
        "objective_status": objective_status,
        "small_source": str(small_path),
        "expanded_input": str(expanded_path),
        "expanded_output": str(args.expanded_output),
        "expanded_output_built": expanded_output_built,
        "target_col": target_col,
        "n_rows_small": int(len(small)),
        "n_rows_expanded": int(len(expanded)),
        "n_frauds_expanded": int(expanded["is_fraud"].sum()),
        "min_fidelity": args.min_fidelity,
        "selected_recipe": selected_recipe,
        "selected_recipe_type": selected.recipe_type if selected else None,
        "selected_recipe_name": selected.name if selected else None,
        "selected_recipe_match_all": selected.match_rate_all if selected else None,
        "selected_recipe_match_test": selected.match_rate_test if selected else None,
        "selected_recipe_target_positive_rate": selected.target_positive_rate if selected else None,
        "selected_recipe_pred_positive_rate_all": selected.pred_positive_rate_all if selected else None,
        "expanded_replay_metrics": replay_df.to_dict(orient="records") if not replay_df.empty else [],
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, recipes_df, replay_df)
    (output_dir / "14_exp014b_r3b_report.md").write_text(report, encoding="utf-8")

    log("")
    log("=" * 80)
    log("EXP-014B-R3B CONCLUÍDO")
    log("=" * 80)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_source_inventory.csv",
        output_dir / "02_code_origin_hits.csv",
        output_dir / "03_base_recipe_candidates.csv",
        output_dir / "04_surrogate_fidelity.csv",
        output_dir / "05_small_reconstruction_metrics.csv",
        output_dir / "06_expanded_base_metrics.csv",
        output_dir / "07_expanded_replay_metrics.csv",
        output_dir / "08_rule_impact_expanded.csv",
        output_dir / "10_wilson_recall_ci.csv",
        output_dir / "13_base_recipe_artifact.json",
        output_dir / "14_exp014b_r3b_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
