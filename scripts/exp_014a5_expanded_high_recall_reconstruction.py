#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014A-5 — Expanded High-Recall Policy Reconstruction

Contexto:
  O EXP-014A-4 criou com sucesso:
      dados/exp014a_expanded_scored_input.csv
  com 113.844 linhas, 1.465 fraudes, score_final, if_percentile, decisao
  e exp014a_frozen_pred derivado do runtime oficial.

  Porém, o runtime oficial atual é ultraconservador:
      TP=59, FP=31, FN=1406, recall≈4,03%

  Isso NÃO é a política high-recall vencedora. A política EXP-013K partia de:
      pred_STRICT_RECALL95_SAFE_ONLY
  e depois aplicava 10 microvetos residuais.

Objetivo:
  Validar no dataset expandido uma reconstrução da política high-recall:
    1. Construir/obter pred_STRICT_RECALL95_SAFE_ONLY no dataset expandido.
    2. Aplicar as 10 regras congeladas do EXP-013K.
    3. Medir TP, FP, FN, recall, precision, FPR, Wilson e bootstrap.
    4. Comparar contra o runtime oficial ultraconservador do EXP-014A-4.
    5. Rodar rápido, sem chamar runtime novamente.

Modos de reconstrução da base:
  A) Se o input já tiver pred_STRICT_RECALL95_SAFE_ONLY, usa direto.
  B) Se --base-threshold-col e --base-threshold forem informados, usa threshold congelado.
  C) Caso contrário, cria um SURROGATE congelado a partir de artefato anterior
     que contenha pred_STRICT_RECALL95_SAFE_ONLY.

Importante:
  - O modo SURROGATE é uma reconstrução da política high-recall, não prova que
    encontramos a regra original perfeita.
  - O script mede a fidelidade do surrogate contra os artefatos anteriores.
  - Se a fidelidade for baixa, o resultado deve ser classificado como diagnóstico.

Uso:
  python scripts/exp_014a5_expanded_high_recall_reconstruction.py

Com threshold congelado explícito:
  python scripts/exp_014a5_expanded_high_recall_reconstruction.py --base-threshold-col lgbm_r4_score --base-threshold 0.001

Saídas:
  resultados/experimentos/EXP-014A-5/
    00_run_summary.json
    01_input_contract.json
    02_surrogate_fidelity.csv
    03_global_metrics.csv
    04_rule_impact.csv
    05_time_block_metrics.csv
    06_wilson_recall_ci.csv
    07_bootstrap_summary.csv
    08_false_negatives.csv
    09_false_positives_sample.csv
    10_exp014a5_report.md
    11_selected_policy_artifact.json
    12_predictions.csv
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
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "dados").exists() else Path.cwd()

DEFAULT_INPUT = PROJECT_ROOT / "dados" / "exp014a_expanded_scored_input.csv"
DEFAULT_POLICY = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013K" / "12_policy_artifact.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014A-5"

PRIOR_CANDIDATES = [
    PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013J" / "06_predictions_by_scenario.csv",
    PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013K" / "07_selected_predictions.csv",
    PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013H" / "05_frozen_predictions.csv",
]

TARGET_BASE_COL = "pred_STRICT_RECALL95_SAFE_ONLY"
RUNTIME_FINAL_COLS = ["exp014a_frozen_pred", "exp013k_residual_fp_pred"]

NUMERIC_FEATURES = [
    "lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw",
    "score_final", "if_percentile", "if_percentile_x", "if_percentile_y",
    "vl_pix", "ratio_valor_media_pagador_90d",
    "qtd_pix_recebidos_180d", "valor_total_recebido_180d",
    "first_receiver_flag_real", "mbk_available_flag",
]

CATEGORICAL_FEATURES = [
    "value_band", "ds_tipo_chave_norm", "periodo_dia",
    "lgbm_bin", "if_bin", "score_bin", "vl_bin", "ratio_bin",
    "module_quiet",
]


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

    if "is_fraud" in df.columns:
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


def load_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Policy artifact não encontrado: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "selected_rules" not in obj:
        raise RuntimeError("Policy artifact sem selected_rules.")
    return obj


def parse_rule_params(rule: dict[str, Any]) -> dict[str, Any]:
    raw = rule.get("params_json")
    if isinstance(raw, dict):
        return raw
    if raw:
        try:
            return json.loads(str(raw))
        except Exception:
            pass
    if isinstance(rule.get("params"), dict):
        return rule["params"]
    return {}


def apply_rule_mask(df: pd.DataFrame, rule: dict[str, Any], current_pred: np.ndarray) -> np.ndarray:
    params = parse_rule_params(rule)
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
        mask = mask & (df[c].astype(str) == str(v))

    if require_module_quiet and "module_quiet" in df.columns:
        mask = mask & (df["module_quiet"].astype(str) == "module_quiet")

    return mask & (current_pred == 1)


def apply_exp013k_policy(df: pd.DataFrame, base_pred: np.ndarray, policy: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame]:
    pred = base_pred.astype(int).copy()
    y = df["is_fraud"].to_numpy(dtype=int)

    rows = []
    for idx, rule in enumerate(policy.get("selected_rules", [])):
        mask = apply_rule_mask(df, rule, pred)
        tp_loss = int(((y == 1) & mask).sum())
        fp_removed = int(((y == 0) & mask).sum())
        pred[mask] = 0
        rows.append({
            "rule_index": idx,
            "description": rule.get("description"),
            "tp_loss": tp_loss,
            "fp_removed": fp_removed,
            "n_removed": int(mask.sum()),
        })

    return pred, pd.DataFrame(rows)


def contract_report(df: pd.DataFrame) -> dict[str, Any]:
    missing = []
    if "is_fraud" not in df.columns:
        missing.append("is_fraud")
    if not any(c in df.columns for c in ["event_datetime", "data_pix", "dt_pix"]):
        missing.append("event_datetime_or_data_pix")
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
        "has_target_base_col": TARGET_BASE_COL in df.columns,
        "runtime_final_cols_present": [c for c in RUNTIME_FINAL_COLS if c in df.columns],
        "n_rows": int(len(df)),
        "n_frauds": int(df["is_fraud"].sum()) if "is_fraud" in df.columns else None,
    }


def prep_X(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in num_cols:
        X[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for c in cat_cols:
        X[c] = df[c].astype("string").fillna("<MISSING>").astype(str)
    return X


def find_prior_file(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"prior-predictions não encontrado: {p}")
        return p
    for p in PRIOR_CANDIDATES:
        if p.exists():
            try:
                cols = pd.read_csv(p, nrows=0).columns
                cols = [str(c).strip().split(".")[-1] for c in cols]
                if TARGET_BASE_COL in cols:
                    return p
            except Exception:
                pass
    raise FileNotFoundError(
        f"Não encontrei artefato anterior com {TARGET_BASE_COL}. Informe --prior-predictions."
    )


def train_surrogate(prior_path: Path, expanded: pd.DataFrame, max_depth: int, min_leaf: int, seed: int):
    prior = normalize_columns(pd.read_csv(prior_path, low_memory=False))
    prior = ensure_bins_and_guards(prior)

    if TARGET_BASE_COL not in prior.columns:
        raise RuntimeError(f"Prior file não contém {TARGET_BASE_COL}: {prior_path}")

    num_cols = [c for c in NUMERIC_FEATURES if c in prior.columns and c in expanded.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in prior.columns and c in expanded.columns]

    if not num_cols and not cat_cols:
        raise RuntimeError("Sem features comuns para treinar surrogate.")

    X = prep_X(prior, num_cols, cat_cols)
    y = pd.to_numeric(prior[TARGET_BASE_COL], errors="coerce").fillna(0).astype(int)

    transformers = []
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), cat_cols))

    clf = Pipeline([
        ("prep", ColumnTransformer(transformers=transformers, remainder="drop")),
        ("tree", DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            random_state=seed,
        )),
    ])

    fidelity_rows = []
    if y.nunique() > 1 and len(y) >= 100:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=seed, stratify=y
            )
            clf.fit(X_train, y_train)
            pred_test = clf.predict(X_test)
            pred_train = clf.predict(X_train)
            fidelity_rows.append({
                "split": "train",
                "n": int(len(y_train)),
                "match_rate": float((pred_train == y_train).mean()),
                "target_positive_rate": float(y_train.mean()),
                "pred_positive_rate": float(np.mean(pred_train)),
            })
            fidelity_rows.append({
                "split": "test",
                "n": int(len(y_test)),
                "match_rate": float((pred_test == y_test).mean()),
                "target_positive_rate": float(y_test.mean()),
                "pred_positive_rate": float(np.mean(pred_test)),
            })
        except Exception as exc:
            fidelity_rows.append({
                "split": "split_failed",
                "n": int(len(y)),
                "match_rate": None,
                "target_positive_rate": float(y.mean()),
                "pred_positive_rate": None,
                "error": str(exc)[:500],
            })

    clf.fit(X, y)
    pred_all = clf.predict(X)
    fidelity_rows.append({
        "split": "all_refit",
        "n": int(len(y)),
        "match_rate": float((pred_all == y).mean()),
        "target_positive_rate": float(y.mean()),
        "pred_positive_rate": float(np.mean(pred_all)),
    })

    meta = {
        "prior_path": str(prior_path),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "max_depth": max_depth,
        "min_leaf": min_leaf,
        "mode": "SURROGATE_FROM_PRIOR_ARTIFACT",
    }

    return clf, pd.DataFrame(fidelity_rows), meta


def construct_base_prediction(df: pd.DataFrame, args):
    if TARGET_BASE_COL in df.columns:
        pred = pd.to_numeric(df[TARGET_BASE_COL], errors="coerce").fillna(0).astype(int).to_numpy()
        return pred, "EXISTING_TARGET_BASE_COL", pd.DataFrame(), {"base_col": TARGET_BASE_COL}

    if args.base_threshold_col and args.base_threshold is not None:
        if args.base_threshold_col not in df.columns:
            raise RuntimeError(f"--base-threshold-col não existe no input: {args.base_threshold_col}")
        pred = (pd.to_numeric(df[args.base_threshold_col], errors="coerce").fillna(0.0) >= float(args.base_threshold)).astype(int).to_numpy()
        return pred, "FROZEN_EXPLICIT_THRESHOLD", pd.DataFrame([{
            "split": "explicit_threshold",
            "feature": args.base_threshold_col,
            "threshold": args.base_threshold,
        }]), {"feature": args.base_threshold_col, "threshold": args.base_threshold}

    prior_path = find_prior_file(args.prior_predictions)
    model, fidelity, meta = train_surrogate(
        prior_path=prior_path,
        expanded=df,
        max_depth=args.surrogate_max_depth,
        min_leaf=args.surrogate_min_leaf,
        seed=args.seed,
    )
    X_exp = prep_X(df, meta["num_cols"], meta["cat_cols"])
    pred = model.predict(X_exp).astype(int)

    return pred, "SURROGATE_FROM_PRIOR_ARTIFACT", fidelity, meta


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
    rows = []
    for b in sorted(blocks.dropna().unique()):
        idx = blocks.to_numpy() == b
        part = df.loc[idx].copy()
        m = compute_metrics(part["is_fraud"].to_numpy(dtype=int), pred[idx])
        m.update({
            "policy_name": policy_name,
            "block": int(b),
            "n_rows": int(len(part)),
            "n_frauds": int(part["is_fraud"].sum()),
        })
        rows.append(m)
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


def make_report(summary: dict[str, Any], global_metrics: pd.DataFrame, fidelity: pd.DataFrame, rule_impact: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014A-5 — Expanded High-Recall Policy Reconstruction")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Base construction mode: `{summary['base_construction_mode']}`")
    lines.append(f"- Policy candidate: `{summary['policy_name']}`")
    lines.append("")
    lines.append("## Métricas globais")
    lines.append(global_metrics.to_markdown(index=False))
    lines.append("")
    lines.append("## Fidelidade da reconstrução da base")
    if fidelity.empty:
        lines.append("Não aplicável.")
    else:
        lines.append(fidelity.to_markdown(index=False))
    lines.append("")
    lines.append("## Impacto das regras EXP-013K")
    if rule_impact.empty:
        lines.append("Sem regras aplicadas.")
    else:
        lines.append(rule_impact.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretação")
    if summary["base_construction_mode"] == "SURROGATE_FROM_PRIOR_ARTIFACT":
        lines.append("A política high-recall foi reconstruída por surrogate treinado apenas em artefatos anteriores. Se a fidelidade for alta, o resultado é um bom diagnóstico de generalização; ainda assim, para promoção, idealmente devemos localizar a regra original `pred_STRICT_RECALL95_SAFE_ONLY`.")
    elif summary["base_construction_mode"] == "EXISTING_TARGET_BASE_COL":
        lines.append("A política usou a coluna base original existente no input expandido. Este é o modo mais forte.")
    elif summary["base_construction_mode"] == "FROZEN_EXPLICIT_THRESHOLD":
        lines.append("A política usou threshold explícito congelado informado por parâmetro.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--policy-artifact", default=str(DEFAULT_POLICY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prior-predictions", default=None)
    parser.add_argument("--base-threshold-col", default=None)
    parser.add_argument("--base-threshold", type=float, default=None)
    parser.add_argument("--surrogate-max-depth", type=int, default=6)
    parser.add_argument("--surrogate-min-leaf", type=int, default=10)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--time-blocks", type=int, default=10)
    parser.add_argument("--bootstrap-iters", type=int, default=300)
    parser.add_argument("--false-positive-sample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-write-predictions", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014A-5 — Expanded High-Recall Policy Reconstruction")
    log("=" * 80)
    log(f"Input: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    df = ensure_bins_and_guards(df)
    policy = load_policy(Path(args.policy_artifact))

    contract = contract_report(df)
    dump_json(contract, output_dir / "01_input_contract.json")
    if not contract["contract_ok"]:
        raise RuntimeError(f"Contrato de input falhou: {contract['missing']}")

    runtime_col = None
    for c in RUNTIME_FINAL_COLS:
        if c in df.columns:
            runtime_col = c
            break

    base_pred, base_mode, fidelity, base_meta = construct_base_prediction(df, args)
    df["pred_STRICT_RECALL95_SAFE_ONLY_RECONSTRUCTED"] = base_pred

    policy_pred, rule_impact = apply_exp013k_policy(df, base_pred, policy)
    df["exp014a5_high_recall_policy_pred"] = policy_pred

    y = df["is_fraud"].to_numpy(dtype=int)

    rows = []
    if runtime_col:
        rows.append({"policy_name": f"RUNTIME_FINAL_{runtime_col}", **compute_metrics(y, df[runtime_col].to_numpy(dtype=int))})
    rows.append({"policy_name": "RECONSTRUCTED_BASE_STRICT_RECALL95", **compute_metrics(y, base_pred)})
    rows.append({"policy_name": "EXP014A5_HIGH_RECALL_PLUS_EXP013K_VETOES", **compute_metrics(y, policy_pred)})
    global_metrics = pd.DataFrame(rows)
    global_metrics.to_csv(output_dir / "03_global_metrics.csv", index=False)

    fidelity.to_csv(output_dir / "02_surrogate_fidelity.csv", index=False)
    rule_impact.to_csv(output_dir / "04_rule_impact.csv", index=False)

    blocks = make_time_blocks(df, args.time_blocks)
    block_parts = []
    if runtime_col:
        block_parts.append(block_metrics(df, df[runtime_col].to_numpy(dtype=int), blocks, f"RUNTIME_FINAL_{runtime_col}"))
    block_parts.append(block_metrics(df, base_pred, blocks, "RECONSTRUCTED_BASE_STRICT_RECALL95"))
    block_parts.append(block_metrics(df, policy_pred, blocks, "EXP014A5_HIGH_RECALL_PLUS_EXP013K_VETOES"))
    block_df = pd.concat(block_parts, ignore_index=True)
    block_df.to_csv(output_dir / "05_time_block_metrics.csv", index=False)

    selected_metrics = compute_metrics(y, policy_pred)
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
    wilson_df.to_csv(output_dir / "06_wilson_recall_ci.csv", index=False)

    boot_df = bootstrap_summary(df, "exp014a5_high_recall_policy_pred", args.bootstrap_iters, args.seed, args.target_recall)
    boot_df.to_csv(output_dir / "07_bootstrap_summary.csv", index=False)

    fn = df[(df["is_fraud"] == 1) & (df["exp014a5_high_recall_policy_pred"] == 0)].copy()
    fp = df[(df["is_fraud"] == 0) & (df["exp014a5_high_recall_policy_pred"] == 1)].copy()
    fn.to_csv(output_dir / "08_false_negatives.csv", index=False)
    if len(fp) > args.false_positive_sample:
        fp = fp.sample(args.false_positive_sample, random_state=args.seed)
    fp.to_csv(output_dir / "09_false_positives_sample.csv", index=False)

    if not args.no_write_predictions:
        df.to_csv(output_dir / "12_predictions.csv", index=False)

    base_fidelity_warning = False
    if not fidelity.empty and "match_rate" in fidelity.columns:
        test_rows = fidelity[fidelity["split"] == "test"]
        if not test_rows.empty:
            mr = test_rows["match_rate"].iloc[0]
            if pd.notna(mr) and float(mr) < 0.98:
                base_fidelity_warning = True

    objective_status = "DONE"
    objective_status += "_TARGET_RECALL_MET" if selected_metrics["recall"] >= args.target_recall else "_TARGET_RECALL_NOT_MET"
    objective_status += "_WILSON_PASS" if wilson_low >= args.target_recall else "_WILSON_NOT_PASS"
    if base_mode == "SURROGATE_FROM_PRIOR_ARTIFACT":
        objective_status += "_SURROGATE_BASE"
        if base_fidelity_warning:
            objective_status += "_LOW_FIDELITY_WARNING"

    artifact = {
        "experiment": "EXP-014A-5",
        "policy_name": "expanded_high_recall_reconstructed_policy",
        "objective_status": objective_status,
        "base_construction_mode": base_mode,
        "base_meta": base_meta,
        "selected_metrics": selected_metrics,
        "wilson": wilson_df.to_dict(orient="records")[0],
        "rule_impact": rule_impact.to_dict(orient="records"),
        "notes": [
            "No runtime call. Uses EXP-014A-4 scored input.",
            "Applies frozen EXP-013K residual veto rules.",
            "If base mode is SURROGATE, treat as diagnostic until original pred_STRICT_RECALL95_SAFE_ONLY rule is located."
        ],
    }
    dump_json(artifact, output_dir / "11_selected_policy_artifact.json")

    summary = {
        "experiment": "EXP-014A-5",
        "status": "DONE",
        "objective_status": objective_status,
        "policy_name": "EXP014A5_HIGH_RECALL_PLUS_EXP013K_VETOES",
        "input_path": str(input_path),
        "n_rows": int(len(df)),
        "n_frauds": total_frauds,
        "target_recall": args.target_recall,
        "base_construction_mode": base_mode,
        "base_meta": base_meta,
        "base_fidelity_warning": base_fidelity_warning,
        "runtime_col": runtime_col,
        "selected_metrics": selected_metrics,
        "wilson_recall_low": wilson_low,
        "wilson_recall_high": wilson_high,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, global_metrics, fidelity, rule_impact)
    (output_dir / "10_exp014a5_report.md").write_text(report, encoding="utf-8")

    log("")
    log("=" * 80)
    log("EXP-014A-5 CONCLUÍDO")
    log("=" * 80)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "02_surrogate_fidelity.csv",
        output_dir / "03_global_metrics.csv",
        output_dir / "04_rule_impact.csv",
        output_dir / "05_time_block_metrics.csv",
        output_dir / "06_wilson_recall_ci.csv",
        output_dir / "07_bootstrap_summary.csv",
        output_dir / "10_exp014a5_report.md",
        output_dir / "11_selected_policy_artifact.json",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
