# -*- coding: utf-8 -*-
"""
EXP-014B-R4B-FAST — FPR < 1% pathfinder, versão computacionalmente eficiente.

Motivo da versão FAST:
  A versão ampla anterior fazia busca combinatória pesada com quads/quints e podia
  ficar muitas horas presa antes mesmo da seleção. Esta versão evita explosão
  combinatória e evita fragmentar o DataFrame.

Estratégia:
  - Trabalha apenas no subconjunto elegível: linhas atualmente em CONFIRMAR ou BLOQUEAR.
  - Minera candidatos por groupby vetorizado em combinações categóricas podadas.
  - Usa limiares numéricos por quantis, sem inserir colunas repetidamente no DataFrame.
  - Armazena candidatos como arrays de índices e usa greedy incremental em numpy.
  - Para assim que atingir FPR < 1% com FN total <= 5, salvo --continue-after-target.

Entrada padrão:
  1) resultados/experimentos/EXP-014B-R4A-FROZEN/06_predictions_frozen.csv
  2) fallback: resultados/experimentos/EXP-014B-R4A/09_predictions_recommended.csv

Saídas:
  resultados/experimentos/EXP-014B-R4B-FAST/
    00_run_summary.json
    01_input_contract.json
    02_base_metrics.json
    03_candidates.csv
    04_selection_frontier.csv
    05_selected_demotions.csv
    06_decision_metrics_by_action.csv
    07_robustness_by_segment.csv
    08_policy_artifact_recommended.json
    09_predictions_recommended.csv
    10_exp014b_r4b_fast_report.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT = "EXP-014B-R4B-FAST"

LABEL_CANDIDATES = ["is_fraud", "fraude", "target", "label", "tp_fraude"]
ACTION_CANDIDATES = [
    "r4a_frozen_decisao_recommended",
    "r4a_decisao_recommended",
    "r3z_frozen_decisao_recommended",
    "r3z_decisao_recommended",
]
BLOCK_CANDIDATES = [
    "exp014b_r4a_frozen_block_pred",
    "exp014b_r4a_block_pred",
    "exp014b_r3z_frozen_block_pred",
    "exp014b_r3z_block_pred",
]
INTERVENTION_CANDIDATES = [
    "exp014b_r4a_frozen_intervention_pred",
    "exp014b_r4a_intervention_pred",
    "exp014b_r3z_frozen_intervention_pred",
    "exp014b_r3z_intervention_pred",
]

# Ordem prioriza as colunas que mais apareceram nos vencedores R3Z/R4A.
CATEGORICAL_PRIORITY = [
    "action",  # virtual, criada como _eligible_action
    "ds_tipo_chave_norm",
    "value_band",
    "periodo_dia",
    "score_bin",
    "lgbm_bin",
    "if_bin",
    "ratio_bin",
    "qtd_rec_bin",
    "valor_rec_bin",
    "mbk_available_flag",
    "module_quiet",
    "se_worst_pattern",
    "first_receiver_flag_real",
    "r3u_receiver_trust_bucket",
    "r3u_relationship_bucket",
    "r3u_receiver_strong_flag",
    "r3u_relationship_strong_flag",
    "r3u_receiver_known_flag",
    "r3u_relationship_recurrent_flag",
    "r3u_mbk_quality_flag",
    "r3u_se_missing_flag",
    "r3u_module_quiet_flag",
]

SCORE_COLS = [
    "lgbm_r4_score",
    "score_final",
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
    "_eligible_action",
    "ds_tipo_chave_norm",
    "value_band",
    "periodo_dia",
    "score_bin",
    "lgbm_bin",
    "if_bin",
    "ratio_bin",
    "qtd_rec_bin",
    "valor_rec_bin",
    "mbk_available_flag",
    "module_quiet",
    "se_worst_pattern",
    "first_receiver_flag_real",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=str, default=None)
    p.add_argument("--artifact", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--target-fpr", type=float, default=0.01, help="Meta estrita: resultado deve ficar menor que este FPR.")
    p.add_argument("--max-total-fn", type=int, default=5)
    p.add_argument("--eligible-actions", type=str, default="CONFIRMAR,BLOQUEAR")
    p.add_argument("--min-incremental-fp", type=int, default=1)
    p.add_argument("--min-support", type=int, default=3)
    p.add_argument("--max-rules", type=int, default=120)
    p.add_argument("--max-candidates", type=int, default=5000)
    p.add_argument("--max-cat-cols", type=int, default=16)
    p.add_argument("--max-pair-combos", type=int, default=180)
    p.add_argument("--max-triplet-combos", type=int, default=240)
    p.add_argument("--max-quad-combos", type=int, default=240)
    p.add_argument("--enable-quads", action="store_true")
    p.add_argument("--no-score-cats", action="store_true")
    p.add_argument("--continue-after-target", action="store_true")
    return p.parse_args()


def default_paths() -> tuple[Path, Path | None, Path]:
    root = Path.cwd()
    frozen = root / "resultados" / "experimentos" / "EXP-014B-R4A-FROZEN" / "06_predictions_frozen.csv"
    rec = root / "resultados" / "experimentos" / "EXP-014B-R4A" / "09_predictions_recommended.csv"
    pred = frozen if frozen.exists() else rec
    artifact = root / "resultados" / "experimentos" / "EXP-014B-R4A-FROZEN" / "05_policy_artifact_frozen.json"
    if not artifact.exists():
        artifact = root / "resultados" / "experimentos" / "EXP-014B-R4A" / "08_policy_artifact_recommended.json"
    out = root / "resultados" / "experimentos" / EXPERIMENT
    return pred, artifact if artifact.exists() else None, out


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


def norm_action(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def action_to_intervention(action: pd.Series) -> pd.Series:
    return norm_action(action).isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)


def action_to_block(action: pd.Series) -> pd.Series:
    return norm_action(action).eq("BLOQUEAR").astype(int)


def metrics(y_true: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(pred, dtype=int)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
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


def metrics_by_action(df: pd.DataFrame, label_col: str, action_col: str) -> pd.DataFrame:
    y = int_series(df[label_col])
    rows = []
    for action, idx in df.groupby(action_col, dropna=False).groups.items():
        yy = y.loc[list(idx)]
        n = int(len(yy))
        frauds = int((yy == 1).sum())
        normals = int((yy == 0).sum())
        rows.append({
            "action": str(action),
            "n_rows": n,
            "n_frauds": frauds,
            "n_normals": normals,
            "precision_within_action": round(float(frauds / n), 8) if n else 0.0,
        })
    return pd.DataFrame(rows).sort_values("action")


def strict_target_fp(n_normals: int, target_fpr: float) -> int:
    """Maior FP inteiro tal que FP/n_normals < target_fpr."""
    return int(math.ceil(target_fpr * n_normals) - 1)


def sanitize_value(v: Any) -> str:
    return str(v).replace("|", "_").replace("\n", " ")[:80]


def candidate_record(
    idx: np.ndarray,
    y_np: np.ndarray,
    rule_type: str,
    description: str,
    candidate_id: str,
) -> dict[str, Any] | None:
    if idx.size == 0:
        return None
    yy = y_np[idx]
    tp_loss = int((yy == 1).sum())
    fp_removed = int((yy == 0).sum())
    if fp_removed <= 0:
        return None
    return {
        "candidate_id": candidate_id,
        "rule_type": rule_type,
        "description": description,
        "idx": idx.astype(np.int32, copy=False),
        "n_demoted": int(idx.size),
        "fp_removed": fp_removed,
        "tp_loss": tp_loss,
        "precision_demoted": round(float(tp_loss / idx.size), 8) if idx.size else 0.0,
        "fp_per_tp_loss": round(float(fp_removed / max(tp_loss, 1)), 8),
    }


def candidate_public(c: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in c.items() if k != "idx"}


def choose_cat_cols(df: pd.DataFrame, eligible_idx: np.ndarray, label_col: str, max_cat_cols: int) -> list[str]:
    y = int_series(df[label_col]).to_numpy(dtype=int)
    cols = [c for c in CATEGORICAL_PRIORITY if c in df.columns or c == "action"]
    cols = ["_eligible_action" if c == "action" else c for c in cols]
    scored = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df.iloc[eligible_idx][c].fillna("<MISSING>").astype(str)
        nunique = int(s.nunique(dropna=False))
        if nunique < 2 or nunique > 80:
            continue
        # Score: melhor grupo zero-TP ou baixo-TP nesta coluna.
        tmp = pd.DataFrame({"v": s.to_numpy(), "y": y[eligible_idx]}, index=eligible_idx)
        g = tmp.groupby("v", sort=False)["y"].agg(["count", "sum"])
        g["fp"] = g["count"] - g["sum"]
        safe = g[g["sum"] <= 5]
        best_fp = int(safe["fp"].max()) if not safe.empty else 0
        scored.append((best_fp, -nunique, c))
    scored.sort(reverse=True)
    return [c for _, _, c in scored[:max_cat_cols]]


def make_group_candidates(
    df: pd.DataFrame,
    y_np: np.ndarray,
    eligible_idx: np.ndarray,
    cols: tuple[str, ...],
    min_support: int,
    min_fp: int,
    max_group_tp: int,
    limit_groups: int,
) -> list[dict[str, Any]]:
    sub = df.iloc[eligible_idx][list(cols)].copy()
    for c in cols:
        sub[c] = sub[c].fillna("<MISSING>").astype(str)
    sub["__rowid"] = eligible_idx
    sub["__y"] = y_np[eligible_idx]

    # Primeiro agrega sem materializar grupos para podar; depois pega índices só dos grupos filtrados.
    agg = sub.groupby(list(cols), dropna=False, sort=False)["__y"].agg(["count", "sum"])
    agg["fp"] = agg["count"] - agg["sum"]
    agg = agg[(agg["count"] >= min_support) & (agg["fp"] >= min_fp) & (agg["sum"] <= max_group_tp)]
    if agg.empty:
        return []
    agg = agg.sort_values(["sum", "fp", "count"], ascending=[True, False, False]).head(limit_groups)

    keys = set(agg.index.tolist())
    if len(cols) == 1:
        keys = set((k,) for k in keys)

    out: list[dict[str, Any]] = []
    grouped = sub.groupby(list(cols), dropna=False, sort=False)
    for key, grp in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        if key_tuple not in keys:
            continue
        idx = grp["__rowid"].to_numpy(dtype=np.int32)
        parts = [f"{c} == {v}" for c, v in zip(cols, key_tuple)]
        desc = "Demover intervenção R4B FAST com " + " AND ".join(parts)
        cid = f"cat{len(cols)}__" + "__".join(f"{c}={sanitize_value(v)}" for c, v in zip(cols, key_tuple))
        rec = candidate_record(idx, y_np, f"categorical_{len(cols)}", desc, cid)
        if rec:
            out.append(rec)
    return out


def make_score_candidates(
    df: pd.DataFrame,
    y_np: np.ndarray,
    eligible_idx: np.ndarray,
    min_fp: int,
    max_group_tp: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    qs = [0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 0.995]
    for col in SCORE_COLS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        valid = eligible_idx[~np.isnan(s[eligible_idx])]
        if valid.size == 0:
            continue
        thresholds = sorted(set(float(np.nanquantile(s[valid], q)) for q in qs))
        for th in thresholds:
            for op, mask in [
                ("<=", s[eligible_idx] <= th),
                (">=", s[eligible_idx] >= th),
            ]:
                idx = eligible_idx[mask & ~np.isnan(s[eligible_idx])]
                if idx.size == 0:
                    continue
                yy = y_np[idx]
                fp = int((yy == 0).sum())
                tp = int((yy == 1).sum())
                if fp < min_fp or tp > max_group_tp:
                    continue
                desc = f"Demover intervenção R4B FAST com {col} {op} {th:.12g}"
                cid = f"score_{'lo' if op == '<=' else 'hi'}__{col}__{th:.12g}"
                rec = candidate_record(idx, y_np, "score_threshold", desc, cid)
                if rec:
                    out.append(rec)
    return out


def make_score_cat_candidates(
    df: pd.DataFrame,
    y_np: np.ndarray,
    eligible_idx: np.ndarray,
    cat_cols: list[str],
    min_support: int,
    min_fp: int,
    max_group_tp: int,
    limit_groups_per_col: int = 40,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    score_cols = [c for c in SCORE_COLS if c in df.columns][:8]
    qs = [0.10, 0.20, 0.30, 0.70, 0.80, 0.90, 0.95]

    for cat_col in cat_cols[:12]:
        s_cat = df.iloc[eligible_idx][cat_col].fillna("<MISSING>").astype(str)
        vc = s_cat.value_counts(dropna=False).head(limit_groups_per_col)
        cat_values = list(vc[vc >= min_support].index)
        if not cat_values:
            continue

        for score_col in score_cols:
            s_score = pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=float)
            valid = eligible_idx[~np.isnan(s_score[eligible_idx])]
            if valid.size == 0:
                continue
            thresholds = sorted(set(float(np.nanquantile(s_score[valid], q)) for q in qs))
            full_cat = df[cat_col].fillna("<MISSING>").astype(str).to_numpy()
            for val in cat_values:
                base_idx = eligible_idx[full_cat[eligible_idx] == str(val)]
                if base_idx.size < min_support:
                    continue
                for th in thresholds:
                    for op, mask in [
                        ("<=", s_score[base_idx] <= th),
                        (">=", s_score[base_idx] >= th),
                    ]:
                        idx = base_idx[mask & ~np.isnan(s_score[base_idx])]
                        if idx.size == 0:
                            continue
                        yy = y_np[idx]
                        fp = int((yy == 0).sum())
                        tp = int((yy == 1).sum())
                        if fp < min_fp or tp > max_group_tp:
                            continue
                        desc = f"Demover intervenção R4B FAST com {cat_col} == {val} AND {score_col} {op} {th:.12g}"
                        cid = f"scorecat__{cat_col}={sanitize_value(val)}__{score_col}__{op}{th:.12g}"
                        rec = candidate_record(idx, y_np, "score_cat_1", desc, cid)
                        if rec:
                            out.append(rec)
    return out


def build_candidates(
    df: pd.DataFrame,
    label_col: str,
    action_col: str,
    eligible_actions: set[str],
    max_total_fn: int,
    base_fn: int,
    min_support: int,
    min_fp: int,
    max_candidates: int,
    max_cat_cols: int,
    max_pair_combos: int,
    max_triplet_combos: int,
    max_quad_combos: int,
    enable_quads: bool,
    no_score_cats: bool,
) -> tuple[list[dict[str, Any]], list[str], np.ndarray]:
    y_np = int_series(df[label_col]).to_numpy(dtype=int)
    action = norm_action(df[action_col])
    eligible_mask = action.isin(sorted(eligible_actions)).to_numpy()
    eligible_idx = np.where(eligible_mask)[0].astype(np.int32)

    if eligible_idx.size == 0:
        return [], [], eligible_idx

    # Cria uma única coluna auxiliar por cópia compacta, sem fragmentar o DataFrame.
    df["_eligible_action"] = action

    cat_cols = choose_cat_cols(df, eligible_idx, label_col, max_cat_cols=max_cat_cols)
    max_group_tp = max(0, int(max_total_fn) - int(base_fn))
    out: list[dict[str, Any]] = []

    out.extend(make_score_candidates(df, y_np, eligible_idx, min_fp=min_fp, max_group_tp=max_group_tp))

    # Singles/pairs/triplets/quads com poda de combinações.
    combo_specs = [(1, max_cat_cols, len(cat_cols)), (2, max_pair_combos, len(cat_cols)), (3, max_triplet_combos, min(len(cat_cols), 14))]
    if enable_quads:
        combo_specs.append((4, max_quad_combos, min(len(cat_cols), 12)))

    for size, limit_combos, limit_cols in combo_specs:
        combos = list(itertools.combinations(cat_cols[:limit_cols], size))
        if size >= 2:
            # Ordena combinações que envolvem features mais prioritárias no começo e limita.
            combos = combos[:limit_combos]
        for cols in combos:
            out.extend(
                make_group_candidates(
                    df, y_np, eligible_idx, cols,
                    min_support=min_support,
                    min_fp=min_fp,
                    max_group_tp=max_group_tp,
                    limit_groups=max(20, max_candidates // 20),
                )
            )

    if not no_score_cats:
        out.extend(
            make_score_cat_candidates(
                df, y_np, eligible_idx, cat_cols,
                min_support=min_support,
                min_fp=min_fp,
                max_group_tp=max_group_tp,
            )
        )

    # Dedup por descrição, filtra e limita.
    seen: set[str] = set()
    dedup: list[dict[str, Any]] = []
    for c in out:
        if c["description"] in seen:
            continue
        seen.add(c["description"])
        if int(c["fp_removed"]) < min_fp:
            continue
        if int(c["tp_loss"]) > max_group_tp:
            continue
        dedup.append(c)

    dedup.sort(key=lambda c: (int(c["tp_loss"]), -int(c["fp_removed"]), -float(c["fp_per_tp_loss"]), int(c["n_demoted"])))
    return dedup[:max_candidates], cat_cols, eligible_idx


def apply_demotions(action: pd.Series, demoted: np.ndarray) -> pd.Series:
    out = norm_action(action).copy()
    out.iloc[np.where(demoted)[0]] = "APROVAR"
    return out


def greedy_select(
    candidates: list[dict[str, Any]],
    y_np: np.ndarray,
    base_pred: np.ndarray,
    base_fn: int,
    base_fp: int,
    target_fp: int,
    max_total_fn: int,
    max_rules: int,
    min_incremental_fp: int,
    continue_after_target: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    n = len(y_np)
    demoted = np.zeros(n, dtype=bool)
    selected: list[dict[str, Any]] = []
    frontier: list[dict[str, Any]] = []
    cumulative_fp = 0
    cumulative_fn = 0
    active = candidates.copy()

    for step in range(1, max_rules + 1):
        current_fp = base_fp - cumulative_fp
        current_fn = base_fn + cumulative_fn
        if current_fp <= target_fp and not continue_after_target:
            break

        best = None
        best_new_idx = None
        best_score = None

        for c in active:
            idx = c["idx"]
            new_idx = idx[~demoted[idx]]
            if new_idx.size == 0:
                continue
            yy = y_np[new_idx]
            fp_gain = int((yy == 0).sum())
            fn_gain = int((yy == 1).sum())
            if fp_gain < min_incremental_fp:
                continue
            if base_fn + cumulative_fn + fn_gain > max_total_fn:
                continue

            # prioriza zero FN, depois razão FP/FN, depois FP absoluto.
            would_reach = int(base_fp - cumulative_fp - fp_gain) <= target_fp
            score = (
                1 if would_reach else 0,
                1 if fn_gain == 0 else 0,
                fp_gain / max(fn_gain, 1),
                fp_gain,
                -fn_gain,
                -int(new_idx.size),
            )
            if best is None or score > best_score:
                best = c
                best_new_idx = new_idx
                best_score = score
                best_fp_gain = fp_gain
                best_fn_gain = fn_gain

        if best is None or best_new_idx is None:
            break

        demoted[best_new_idx] = True
        cumulative_fp += int(best_fp_gain)
        cumulative_fn += int(best_fn_gain)
        result_fp = int(base_fp - cumulative_fp)
        result_fn = int(base_fn + cumulative_fn)

        rec = candidate_public(best)
        rec.update({
            "selection_step": step,
            "incremental_n_demoted": int(best_new_idx.size),
            "incremental_fp_removed": int(best_fp_gain),
            "incremental_tp_loss": int(best_fn_gain),
            "cumulative_n_demoted": int(demoted.sum()),
            "cumulative_fp_removed": int(cumulative_fp),
            "cumulative_fn_added": int(cumulative_fn),
            "result_fp": result_fp,
            "result_fn": result_fn,
            "target_reached": bool(result_fp <= target_fp),
        })
        selected.append(rec)
        frontier.append({
            "selection_step": step,
            "selected_candidate_id": rec["candidate_id"],
            "selected_description": rec["description"],
            "incremental_n_demoted": rec["incremental_n_demoted"],
            "incremental_fp_removed": rec["incremental_fp_removed"],
            "incremental_tp_loss": rec["incremental_tp_loss"],
            "cumulative_n_demoted": rec["cumulative_n_demoted"],
            "cumulative_fp_removed": rec["cumulative_fp_removed"],
            "cumulative_fn_added": rec["cumulative_fn_added"],
            "result_fp": result_fp,
            "result_fn": result_fn,
            "target_reached": bool(result_fp <= target_fp),
        })
        active = [c for c in active if c["description"] != best["description"]]

    return selected, frontier, demoted


def robustness(df: pd.DataFrame, label_col: str, before_col: str, after_col: str) -> pd.DataFrame:
    rows = []
    y = int_series(df[label_col])
    before = action_to_intervention(df[before_col])
    after = action_to_intervention(df[after_col])
    for col in SEGMENT_COLS:
        if col not in df.columns:
            continue
        for val, idx in df.groupby(col, dropna=False).groups.items():
            idx = list(idx)
            before_m = metrics(y.loc[idx], before.loc[idx])
            after_m = metrics(y.loc[idx], after.loc[idx])
            rows.append({
                "segment_col": col,
                "segment_value": str(val),
                "n_rows": int(len(idx)),
                "n_frauds": int((y.loc[idx] == 1).sum()),
                "fp_removed": int(before_m["fp"] - after_m["fp"]),
                "tp_loss": int(before_m["tp"] - after_m["tp"]),
                "before_fp": before_m["fp"],
                "after_fp": after_m["fp"],
                "before_tp": before_m["tp"],
                "after_tp": after_m["tp"],
                "after_fn": after_m["fn"],
            })
    return pd.DataFrame(rows).sort_values(["fp_removed", "n_rows"], ascending=[False, False]) if rows else pd.DataFrame()


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    default_pred, default_artifact, default_out = default_paths()
    pred_path = Path(args.predictions) if args.predictions else default_pred
    artifact_path = Path(args.artifact) if args.artifact else default_artifact
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions não encontrado: {pred_path}")

    df = pd.read_csv(pred_path, low_memory=False).copy()
    artifact = None
    if artifact_path and Path(artifact_path).exists():
        artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))

    label_col = find_col(df, LABEL_CANDIDATES)
    action_col = find_col(df, ACTION_CANDIDATES)
    block_col = find_col(df, BLOCK_CANDIDATES, required=False)
    inter_col = find_col(df, INTERVENTION_CANDIDATES, required=False)

    action = norm_action(df[action_col])
    df["_eligible_action"] = action
    y = int_series(df[label_col]).to_numpy(dtype=int)
    base_pred = int_series(df[inter_col]).to_numpy(dtype=int) if inter_col else action_to_intervention(action).to_numpy(dtype=int)
    base_block = int_series(df[block_col]).to_numpy(dtype=int) if block_col else action_to_block(action).to_numpy(dtype=int)

    base_m = metrics(y, base_pred)
    block_m = metrics(y, base_block)
    n_normals = int((y == 0).sum())
    target_fp = strict_target_fp(n_normals, float(args.target_fpr))
    eligible_actions = {a.strip().upper() for a in str(args.eligible_actions).split(",") if a.strip()}

    candidates, cat_cols_used, eligible_idx = build_candidates(
        df=df,
        label_col=label_col,
        action_col=action_col,
        eligible_actions=eligible_actions,
        max_total_fn=int(args.max_total_fn),
        base_fn=int(base_m["fn"]),
        min_support=int(args.min_support),
        min_fp=int(args.min_incremental_fp),
        max_candidates=int(args.max_candidates),
        max_cat_cols=int(args.max_cat_cols),
        max_pair_combos=int(args.max_pair_combos),
        max_triplet_combos=int(args.max_triplet_combos),
        max_quad_combos=int(args.max_quad_combos),
        enable_quads=bool(args.enable_quads),
        no_score_cats=bool(args.no_score_cats),
    )

    selected, frontier, demoted = greedy_select(
        candidates=candidates,
        y_np=y,
        base_pred=base_pred,
        base_fn=int(base_m["fn"]),
        base_fp=int(base_m["fp"]),
        target_fp=int(target_fp),
        max_total_fn=int(args.max_total_fn),
        max_rules=int(args.max_rules),
        min_incremental_fp=int(args.min_incremental_fp),
        continue_after_target=bool(args.continue_after_target),
    )

    final_action = action.copy()
    final_action.iloc[np.where(demoted)[0]] = "APROVAR"
    df["exp014b_r4b_fast_demote_to_approve"] = demoted.astype(int)
    df["r4b_fast_decisao_recommended"] = final_action
    df["exp014b_r4b_fast_intervention_pred"] = action_to_intervention(final_action).astype(int)
    df["exp014b_r4b_fast_block_pred"] = action_to_block(final_action).astype(int)

    final_m = metrics(y, df["exp014b_r4b_fast_intervention_pred"].to_numpy(dtype=int))
    final_block_m = metrics(y, df["exp014b_r4b_fast_block_pred"].to_numpy(dtype=int))
    target_reached = bool(final_m["fp"] <= target_fp and final_m["fpr"] < float(args.target_fpr))
    block_fp_removed = int(block_m["fp"] - final_block_m["fp"])
    block_tp_loss = int(block_m["tp"] - final_block_m["tp"])

    by_act = metrics_by_action(df, label_col, "r4b_fast_decisao_recommended")
    rob = robustness(df, label_col, action_col, "r4b_fast_decisao_recommended")

    candidate_public_df = pd.DataFrame([candidate_public(c) for c in candidates])
    selected_df = pd.DataFrame(selected)
    frontier_df = pd.DataFrame(frontier)

    summary = {
        "experiment": EXPERIMENT,
        "status": "DONE",
        "objective_status": (
            "DONE_R4B_FAST_FPR_LT1_TARGET_REACHED_WITHIN_FN_BUDGET"
            if target_reached and final_m["fn"] <= int(args.max_total_fn)
            else "DONE_R4B_FAST_FPR_LT1_TARGET_NOT_REACHED_BUT_IMPROVED"
            if final_m["fp"] < base_m["fp"] and final_m["fn"] <= int(args.max_total_fn)
            else "DONE_R4B_FAST_NO_SAFE_IMPROVEMENT"
        ),
        "n_rows": int(len(df)),
        "n_frauds": int((y == 1).sum()),
        "n_normals": n_normals,
        "predictions_path": str(pred_path),
        "artifact_path": str(artifact_path) if artifact_path else None,
        "label_col": label_col,
        "action_col": action_col,
        "block_col": block_col,
        "intervention_col": inter_col,
        "eligible_actions": sorted(eligible_actions),
        "baseline_intervention_metrics": base_m,
        "baseline_block_metrics": block_m,
        "final_intervention_metrics": final_m,
        "final_block_metrics": final_block_m,
        "target_fpr_strict": float(args.target_fpr),
        "target_fp_strict": int(target_fp),
        "target_reached": target_reached,
        "gap_to_target_fp": max(0, int(final_m["fp"] - target_fp)),
        "fp_removed_total": int(base_m["fp"] - final_m["fp"]),
        "fn_added_total": int(final_m["fn"] - base_m["fn"]),
        "block_fp_removed": block_fp_removed,
        "block_tp_loss": block_tp_loss,
        "n_candidates_evaluated": int(len(candidates)),
        "n_selected_rules": int(len(selected)),
        "n_eligible_rows": int(len(eligible_idx)),
        "cat_cols_used": cat_cols_used,
        "min_support": int(args.min_support),
        "min_incremental_fp": int(args.min_incremental_fp),
        "max_rules": int(args.max_rules),
        "max_total_fn": int(args.max_total_fn),
        "enable_quads": bool(args.enable_quads),
        "all_pass": bool(final_m["fn"] <= int(args.max_total_fn)),
        "output_dir": str(out_dir),
    }
    contract = {
        "predictions_path": str(pred_path),
        "artifact_path": str(artifact_path) if artifact_path else None,
        "label_col": label_col,
        "action_col": action_col,
        "block_col": block_col,
        "intervention_col": inter_col,
        "eligible_actions": sorted(eligible_actions),
        "target_fpr_strict": float(args.target_fpr),
        "target_fp_strict": int(target_fp),
        "max_total_fn": int(args.max_total_fn),
        "contract_ok": True,
        "missing": [],
    }
    rec_artifact = {
        "experiment": EXPERIMENT,
        "input_predictions_path": str(pred_path),
        "base_action_col": action_col,
        "final_action_col": "r4b_fast_decisao_recommended",
        "demote_col": "exp014b_r4b_fast_demote_to_approve",
        "intervention_pred_col": "exp014b_r4b_fast_intervention_pred",
        "block_pred_col": "exp014b_r4b_fast_block_pred",
        "baseline_intervention_metrics": base_m,
        "baseline_block_metrics": block_m,
        "final_intervention_metrics": final_m,
        "final_block_metrics": final_block_m,
        "target_fpr_strict": float(args.target_fpr),
        "target_fp_strict": int(target_fp),
        "target_reached": target_reached,
        "gap_to_target_fp": max(0, int(final_m["fp"] - target_fp)),
        "fp_removed_total": int(base_m["fp"] - final_m["fp"]),
        "fn_added_total": int(final_m["fn"] - base_m["fn"]),
        "block_fp_removed": block_fp_removed,
        "block_tp_loss": block_tp_loss,
        "selected_demotions": selected,
        "notes": [
            "FAST search: candidate mining is grouped and pruned before greedy selection.",
            "Eligible actions can include CONFIRMAR and BLOQUEAR.",
            "Promotion requires frozen replay if accepted.",
        ],
    }

    write_json(out_dir / "00_run_summary.json", summary)
    write_json(out_dir / "01_input_contract.json", contract)
    write_json(out_dir / "02_base_metrics.json", {"baseline_intervention_metrics": base_m, "baseline_block_metrics": block_m, "artifact_status": artifact.get("frozen_validation_status") if isinstance(artifact, dict) else None})
    candidate_public_df.to_csv(out_dir / "03_candidates.csv", index=False, encoding="utf-8")
    frontier_df.to_csv(out_dir / "04_selection_frontier.csv", index=False, encoding="utf-8")
    selected_df.to_csv(out_dir / "05_selected_demotions.csv", index=False, encoding="utf-8")
    by_act.to_csv(out_dir / "06_decision_metrics_by_action.csv", index=False, encoding="utf-8")
    rob.to_csv(out_dir / "07_robustness_by_segment.csv", index=False, encoding="utf-8")
    write_json(out_dir / "08_policy_artifact_recommended.json", rec_artifact)
    df.to_csv(out_dir / "09_predictions_recommended.csv", index=False, encoding="utf-8")

    report = f"""# {EXPERIMENT} - FPR < 1% FAST Pathfinder

## Resultado executivo
- Status: `{summary['objective_status']}`
- All pass: `{summary['all_pass']}`
- Target FPR strict: `{args.target_fpr}`
- Target FP strict: `{target_fp}`
- Target reached: `{target_reached}`
- Gap FP to target: `{summary['gap_to_target_fp']}`
- FP removidos total: `{summary['fp_removed_total']}`
- FN adicionados: `{summary['fn_added_total']}`
- Block FP removidos: `{block_fp_removed}`
- Block TP loss: `{block_tp_loss}`
- Candidatos avaliados: `{len(candidates)}`
- Regras selecionadas: `{len(selected)}`

## Baseline intervenção
```json
{json.dumps(base_m, ensure_ascii=False, indent=2)}
```

## Final intervenção
```json
{json.dumps(final_m, ensure_ascii=False, indent=2)}
```

## Baseline BLOQUEAR
```json
{json.dumps(block_m, ensure_ascii=False, indent=2)}
```

## Final BLOQUEAR
```json
{json.dumps(final_block_m, ensure_ascii=False, indent=2)}
```

## Métricas por ação final
{by_act.to_markdown(index=False)}

## Regras selecionadas
{selected_df.to_markdown(index=False) if not selected_df.empty else 'Nenhuma regra selecionada.'}

## Frontier
{frontier_df.to_markdown(index=False) if not frontier_df.empty else 'Nenhuma seleção possível.'}

## Decisão sugerida
Se `target_reached=true`, executar frozen validation. Se não, avaliar o gap residual.
"""
    (out_dir / "10_exp014b_r4b_fast_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
