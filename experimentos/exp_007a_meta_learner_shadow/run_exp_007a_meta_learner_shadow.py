"""
EXP-007A — Meta-Learner Shadow

Objetivo:
  Verificar, em modo shadow/artifact-only, se os sinais atuais ainda conseguem
  separar os FNs residuais pós-C1 dos legítimos sem aumentar FP.

Este experimento:
  - Não chama PipelineOrquestrador.
  - Não altera DecisionEngine.
  - Não altera scoring_config.json.
  - Não troca artefatos.
  - Não roda grid E2E.
  - Usa baseline_predictions_seed_42/123 do EXP-006C-R2.
  - Reaplica C1 pós-runtime em artifact-only com min_score=58.
  - Treina meta-learners shadow com OOF prediction.
  - Avalia overlay APROVAR -> CONFIRMAR baseado no score shadow.
  - Só recomenda quick-E2E se houver ganho líquido com FP controlado.

Entradas esperadas:
  resultados/experimentos/EXP-006C-R2/
    baseline_predictions_seed_42.csv
    baseline_predictions_seed_123.csv

Saídas:
  resultados/experimentos/EXP-007A/
    00_input_summary.json
    01_post_c1_baseline_metrics.csv
    02_oof_model_metrics.csv
    03_threshold_sweep.csv
    04_residual_fn_ranking.csv
    05_candidate_overlay_eval.json
    06_feature_importance.csv
    07_recommendation.md
    08_next_experiment_spec.md
"""

from __future__ import annotations

import json
import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# =========================================================
# PATHS
# =========================================================

EXP_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists() and (p / "resultados").exists():
            return p
    return start.parent.parent


PROJECT_ROOT = find_project_root(EXP_DIR)
INPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-006C-R2"
OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-007A"


DECISOES_POSITIVAS = {"CONFIRMAR", "BLOQUEAR"}
RANDOM_STATE = 42


# =========================================================
# HELPERS
# =========================================================

def safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(x) for x in obj]
    if isinstance(obj, tuple):
        return [safe_json(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return [safe_json(x) for x in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe_json(obj), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def flagged(df: pd.DataFrame) -> pd.Series:
    return df["decisao"].astype(str).isin(DECISOES_POSITIVAS)


def compute_metrics(df: pd.DataFrame, label: str) -> dict[str, Any]:
    y = df["is_fraud"].astype(int)
    p = flagged(df).astype(int)

    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)

    return {
        "label": label,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": round(precision, 6),
        "Recall": round(recall, 6),
        "F1": round(f1, 6),
        "FPR": round(fpr, 8),
    }


def normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    text_cols = {
        "transaction_id",
        "customer_id",
        "decisao",
        "veto_reason",
        "veto_suppressed_reason",
        "source_file",
        "exp006f_c1_reason",
        "decisao_original_exp006f_c1",
        "decisao_original_shadow",
    }

    for col in out.columns:
        if col in text_cols:
            continue

        # pandas mais novo não aceita errors="ignore".
        # Então tentamos converter e, se a coluna for textual demais,
        # mantemos como estava.
        try:
            converted = pd.to_numeric(out[col], errors="coerce")

            # Só substitui se houver pelo menos algum valor numérico útil
            # ou se a coluna já parecia numérica.
            non_null_original = out[col].notna().sum()
            non_null_converted = converted.notna().sum()

            if non_null_converted > 0 or non_null_original == 0:
                out[col] = converted

        except Exception:
            pass

    if "is_fraud" in out.columns:
        out["is_fraud"] = pd.to_numeric(out["is_fraud"], errors="coerce").fillna(0).astype(int)

    if "decisao" in out.columns:
        out["decisao"] = out["decisao"].astype(str)

    return out


# =========================================================
# LOAD + POST-C1 BASELINE
# =========================================================

def load_seed(seed: int) -> pd.DataFrame:
    path = INPUT_DIR / f"baseline_predictions_seed_{seed}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path)
    df["seed"] = seed
    df["source_file"] = str(path)

    required = [
        "is_fraud",
        "decisao",
        "vl_pix",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "lgbm_raw",
        "score_final",
        "se_score",
        "beh_score",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} sem colunas obrigatórias: {missing}")

    return normalize_numeric(df)


def apply_c1_post_runtime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reaplica C1 conforme consolidada:
      min_score=58
      max_score=62
    """
    out = df.copy()

    for c in [
        "vl_pix",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "lgbm_raw",
        "score_final",
        "se_score",
        "beh_score",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    mask = (
        out["decisao"].astype(str).eq("APROVAR")
        & out["first_receiver_flag"].astype(int).eq(1)
        & out["pix_key_random_flag"].astype(int).eq(0)
        & out["qt_tempo_relacionamento_mes"].le(12)
        & out["vl_pix"].ge(100.0)
        & out["vl_pix"].lt(500.0)
        & out["lgbm_raw"].ge(0.06)
        & out["lgbm_raw"].lt(0.10)
        & out["score_final"].ge(58.0)
        & out["score_final"].lt(62.0)
        & out["se_score"].le(0.0)
        & out["beh_score"].le(0.0)
    )

    out["exp006f_c1_applied_shadow"] = mask

    idx = out.index[mask]
    out.loc[idx, "decisao_original_exp006f_c1"] = out.loc[idx, "decisao"]
    out.loc[idx, "score_final_original_exp006f_c1"] = out.loc[idx, "score_final"]
    out.loc[idx, "decisao"] = "CONFIRMAR"
    out.loc[idx, "score_final"] = out.loc[idx, "score_final"].apply(lambda x: max(float(x), 62.0))

    return out


def load_post_c1_predictions() -> pd.DataFrame:
    frames = []

    for seed in [42, 123]:
        df = load_seed(seed)
        df = apply_c1_post_runtime(df)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    # Para treino shadow, deduplicar por transaction_id quando existir.
    if "transaction_id" in all_df.columns:
        dedup = all_df.sort_values(["is_fraud", "seed"], ascending=[False, True])
        dedup = dedup.drop_duplicates(subset=["transaction_id"], keep="first").copy()
    else:
        dedup = all_df.drop_duplicates().copy()

    return all_df, dedup


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()

    # Features derivadas simples, baratas e interpretáveis.
    def num(col: str, default: float = 0.0) -> pd.Series:
        if col not in out.columns:
            return pd.Series(default, index=out.index, dtype="float64")
        return pd.to_numeric(out[col], errors="coerce").fillna(default)

    vl = num("vl_pix", 0.0).clip(lower=0)
    rel = num("qt_tempo_relacionamento_mes", 999.0).clip(lower=0)
    idade = num("nr_idade", 0.0).clip(lower=0)
    first = num("first_receiver_flag", 0.0).clip(0, 1)
    pix_random = num("pix_key_random_flag", 0.0).clip(0, 1)
    lgbm = num("lgbm_raw", 0.0).clip(lower=0)
    ifp = num("if_percentile", 0.0).clip(lower=0)
    se = num("se_score", 0.0)
    beh = num("beh_score", 0.0)
    score = num("score_final", 0.0)

    out["meta_log_vl_pix"] = np.log1p(vl)
    out["meta_rel_inv"] = 1.0 / (rel + 1.0)
    out["meta_score_gap_confirmar"] = 62.0 - score
    out["meta_score_near_confirmar"] = ((score >= 55.0) & (score < 62.0)).astype(int)
    out["meta_lgbm_gray"] = ((lgbm >= 0.05) & (lgbm < 0.20)).astype(int)
    out["meta_lgbm_very_low"] = (lgbm < 0.01).astype(int)
    out["meta_if_high"] = (ifp >= 0.95).astype(int)
    out["meta_if_extreme"] = (ifp >= 0.985).astype(int)
    out["meta_rel_curto"] = (rel <= 12).astype(int)
    out["meta_valor_100_500"] = ((vl >= 100) & (vl < 500)).astype(int)
    out["meta_valor_5k_plus"] = (vl >= 5000).astype(int)
    out["meta_valor_15k_plus"] = (vl >= 15000).astype(int)
    out["meta_lgbm_x_first"] = lgbm * first
    out["meta_if_x_first"] = ifp * first
    out["meta_score_x_first"] = score * first
    out["meta_valor_x_first"] = np.log1p(vl) * first
    out["meta_relcurto_x_first"] = (rel <= 12).astype(int) * first
    out["meta_se_or_beh"] = ((se > 0) | (beh > 0)).astype(int)
    out["meta_se_plus_beh"] = se + beh
    out["meta_age_60_plus"] = (idade >= 60).astype(int)

    candidate_features = [
        # módulos
        "lgbm_raw",
        "lgbm_mapped",
        "if_percentile",
        "se_score",
        "beh_score",
        "score_final",
        # transação/cliente
        "vl_pix",
        "nr_idade",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        # derivadas
        "meta_log_vl_pix",
        "meta_rel_inv",
        "meta_score_gap_confirmar",
        "meta_score_near_confirmar",
        "meta_lgbm_gray",
        "meta_lgbm_very_low",
        "meta_if_high",
        "meta_if_extreme",
        "meta_rel_curto",
        "meta_valor_100_500",
        "meta_valor_5k_plus",
        "meta_valor_15k_plus",
        "meta_lgbm_x_first",
        "meta_if_x_first",
        "meta_score_x_first",
        "meta_valor_x_first",
        "meta_relcurto_x_first",
        "meta_se_or_beh",
        "meta_se_plus_beh",
        "meta_age_60_plus",
    ]

    features = [c for c in candidate_features if c in out.columns]

    X = out[features].copy()

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    return X, features


def get_models() -> dict[str, Any]:
    return {
        "LOGREG_BALANCED": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            )),
        ]),
        "RF_SHALLOW_BALANCED": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=160,
                max_depth=5,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "EXTRATREES_SHALLOW_BALANCED": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=180,
                max_depth=5,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }


# =========================================================
# OOF SHADOW
# =========================================================

def oof_predict_models(df_unique: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, features = build_feature_frame(df_unique)
    y = df_unique["is_fraud"].astype(int).values

    models = get_models()

    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    oof = df_unique.copy().reset_index(drop=True)
    model_rows = []
    importances = []

    for model_name, model in models.items():
        probs = np.zeros(len(oof), dtype=float)

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
            model.fit(X.iloc[train_idx], y[train_idx])
            probs[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]

        oof[f"shadow_prob_{model_name}"] = probs

        try:
            auc = roc_auc_score(y, probs)
        except Exception:
            auc = np.nan

        try:
            ap = average_precision_score(y, probs)
        except Exception:
            ap = np.nan

        model_rows.append({
            "model": model_name,
            "oof_auc": round(float(auc), 6) if not np.isnan(auc) else None,
            "oof_average_precision": round(float(ap), 6) if not np.isnan(ap) else None,
            "n_rows": int(len(oof)),
            "n_frauds": int(y.sum()),
            "n_features": len(features),
        })

        # Feature importance: fit final no dataset único para interpretação.
        try:
            model.fit(X, y)
            final_model = model.named_steps.get("model")

            if hasattr(final_model, "feature_importances_"):
                vals = final_model.feature_importances_
                for f, v in zip(features, vals):
                    importances.append({
                        "model": model_name,
                        "feature": f,
                        "importance": float(v),
                    })
            elif hasattr(final_model, "coef_"):
                vals = np.abs(final_model.coef_[0])
                for f, v in zip(features, vals):
                    importances.append({
                        "model": model_name,
                        "feature": f,
                        "importance": float(v),
                    })
        except Exception:
            pass

    # Ensemble simples por média.
    prob_cols = [c for c in oof.columns if c.startswith("shadow_prob_")]
    oof["shadow_prob_ENSEMBLE_MEAN"] = oof[prob_cols].mean(axis=1)

    try:
        auc = roc_auc_score(y, oof["shadow_prob_ENSEMBLE_MEAN"].values)
    except Exception:
        auc = np.nan

    try:
        ap = average_precision_score(y, oof["shadow_prob_ENSEMBLE_MEAN"].values)
    except Exception:
        ap = np.nan

    model_rows.append({
        "model": "ENSEMBLE_MEAN",
        "oof_auc": round(float(auc), 6) if not np.isnan(auc) else None,
        "oof_average_precision": round(float(ap), 6) if not np.isnan(ap) else None,
        "n_rows": int(len(oof)),
        "n_frauds": int(y.sum()),
        "n_features": len(features),
    })

    model_df = pd.DataFrame(model_rows)
    imp_df = pd.DataFrame(importances)

    if not imp_df.empty:
        imp_df = (
            imp_df
            .sort_values(["model", "importance"], ascending=[True, False])
            .groupby("model", as_index=False)
            .head(30)
        )

    return oof, model_df, imp_df


# =========================================================
# THRESHOLD / OVERLAY EVAL
# =========================================================

def overlay_candidate(df: pd.DataFrame, score_col: str, threshold: float) -> pd.DataFrame:
    out = df.copy()

    out["shadow_candidate_hit"] = (
        out["decisao"].astype(str).eq("APROVAR")
        & pd.to_numeric(out[score_col], errors="coerce").fillna(0).ge(threshold)
    )

    idx = out.index[out["shadow_candidate_hit"]]
    out.loc[idx, "decisao_original_shadow"] = out.loc[idx, "decisao"]
    out.loc[idx, "score_final_original_shadow"] = out.loc[idx, "score_final"] if "score_final" in out.columns else np.nan
    out.loc[idx, "decisao"] = "CONFIRMAR"

    if "score_final" in out.columns:
        out.loc[idx, "score_final"] = pd.to_numeric(out.loc[idx, "score_final"], errors="coerce").fillna(0).apply(lambda x: max(float(x), 62.0))

    return out


def compare_delta(base: pd.DataFrame, cand: pd.DataFrame) -> dict[str, Any]:
    y = base["is_fraud"].astype(int)
    b = flagged(base)
    c = flagged(cand)

    recovered_fn = y.eq(1) & (~b) & c
    added_fp = y.eq(0) & (~b) & c
    lost_tp = y.eq(1) & b & (~c)
    removed_fp = y.eq(0) & b & (~c)

    return {
        "fns_recuperados": int(recovered_fn.sum()),
        "fps_adicionados": int(added_fp.sum()),
        "tps_perdidos": int(lost_tp.sum()),
        "fps_removidos": int(removed_fp.sum()),
        "rule_hits": int(cand.get("shadow_candidate_hit", pd.Series(False, index=cand.index)).sum()),
    }


def threshold_sweep(oof: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    prob_cols = [c for c in oof.columns if c.startswith("shadow_prob_")]

    thresholds = sorted(set(
        [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
        + [round(x, 4) for x in np.quantile(oof["shadow_prob_ENSEMBLE_MEAN"], [0.90, 0.95, 0.97, 0.98, 0.99]).tolist()]
    ))

    rows = []

    base_metrics = compute_metrics(oof, "POST_C1_BASELINE")

    for score_col in prob_cols:
        for th in thresholds:
            cand = overlay_candidate(oof, score_col=score_col, threshold=th)
            cm = compute_metrics(cand, f"{score_col}_{th}")
            delta = compare_delta(oof, cand)

            rows.append({
                "score_col": score_col,
                "threshold": float(th),
                **cm,
                **delta,
                "delta_FN_vs_post_c1": cm["FN"] - base_metrics["FN"],
                "delta_FP_vs_post_c1": cm["FP"] - base_metrics["FP"],
                "delta_F1_vs_post_c1": round(cm["F1"] - base_metrics["F1"], 6),
                "passes_strict_shadow": bool(
                    delta["fns_recuperados"] >= 1
                    and delta["fps_adicionados"] == 0
                    and delta["tps_perdidos"] == 0
                    and cm["F1"] >= base_metrics["F1"]
                ),
                "passes_relaxed_shadow": bool(
                    delta["fns_recuperados"] >= 1
                    and delta["fps_adicionados"] <= 1
                    and delta["tps_perdidos"] == 0
                    and cm["F1"] >= base_metrics["F1"]
                ),
            })

    return pd.DataFrame(rows).sort_values(
        ["passes_strict_shadow", "passes_relaxed_shadow", "fns_recuperados", "fps_adicionados", "F1"],
        ascending=[False, False, False, True, False],
    )


def build_residual_fn_ranking(oof: pd.DataFrame) -> pd.DataFrame:
    base_flag = flagged(oof)
    residual = oof[(oof["is_fraud"].astype(int).eq(1)) & (~base_flag)].copy()

    prob_cols = [c for c in oof.columns if c.startswith("shadow_prob_")]

    keep_cols = [
        "transaction_id",
        "customer_id",
        "seed",
        "vl_pix",
        "nr_idade",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "lgbm_raw",
        "lgbm_mapped",
        "if_percentile",
        "se_score",
        "beh_score",
        "score_final",
        "decisao",
        "exp006f_c1_applied_shadow",
    ]
    keep_cols = [c for c in keep_cols if c in residual.columns]

    out = residual[keep_cols + prob_cols].copy()

    if "shadow_prob_ENSEMBLE_MEAN" in out.columns:
        out = out.sort_values("shadow_prob_ENSEMBLE_MEAN", ascending=False)

    return out


def choose_candidate(sweep: pd.DataFrame) -> dict[str, Any]:
    strict = sweep[sweep["passes_strict_shadow"]].copy()

    if not strict.empty:
        best = strict.sort_values(
            ["fns_recuperados", "fps_adicionados", "F1", "threshold"],
            ascending=[False, True, False, False],
        ).iloc[0].to_dict()

        return {
            "status": "CANDIDATO_STRICT_PARA_EXP007B_QUICK_E2E",
            "candidate": best,
            "next_action": "Criar EXP-007B quick-E2E com 1 único overlay meta-learner shadow.",
        }

    relaxed = sweep[sweep["passes_relaxed_shadow"]].copy()

    if not relaxed.empty:
        best = relaxed.sort_values(
            ["fns_recuperados", "fps_adicionados", "F1", "threshold"],
            ascending=[False, True, False, False],
        ).iloc[0].to_dict()

        return {
            "status": "CANDIDATO_RELAXED_REQUER_AUDITORIA_MANUAL",
            "candidate": best,
            "next_action": "Não rodar E2E ainda. Auditar os FPs adicionados antes.",
        }

    return {
        "status": "SEM_CANDIDATO_SEGURO",
        "candidate": None,
        "next_action": (
            "Não rodar EXP-007B. Os sinais atuais não geraram overlay seguro. "
            "Considerar novas fontes de dados ou encerrar FASE 2 como próxima do limite atual."
        ),
    }


# =========================================================
# REPORTS
# =========================================================

def write_recommendation(
    path: Path,
    baseline_metrics: pd.DataFrame,
    model_df: pd.DataFrame,
    sweep: pd.DataFrame,
    candidate_decision: dict[str, Any],
    residual_fn_ranking: pd.DataFrame,
) -> None:
    lines = [
        "# EXP-007A — Meta-Learner Shadow",
        "",
        f"Gerado em: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        f"- Status: `{candidate_decision['status']}`",
        "",
        "## Baseline pós-C1",
        "",
        "| Seed/Conjunto | TP | FP | FN | Precision | Recall | F1 | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, r in baseline_metrics.iterrows():
        lines.append(
            f"| `{r['label']}` | {int(r['TP'])} | {int(r['FP'])} | {int(r['FN'])} | "
            f"{float(r['Precision']):.4%} | {float(r['Recall']):.4%} | "
            f"{float(r['F1']):.4f} | {float(r['FPR']):.4%} |"
        )

    lines.extend([
        "",
        "## Qualidade shadow dos meta-learners",
        "",
        "| Modelo | OOF AUC | OOF AP | Linhas | Fraudes | Features |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    for _, r in model_df.iterrows():
        lines.append(
            f"| `{r['model']}` | {r['oof_auc']} | {r['oof_average_precision']} | "
            f"{int(r['n_rows'])} | {int(r['n_frauds'])} | {int(r['n_features'])} |"
        )

    lines.extend([
        "",
        "## Melhor candidato do sweep",
        "",
    ])

    candidate = candidate_decision.get("candidate")
    if candidate:
        lines.extend([
            f"- Score: `{candidate['score_col']}`",
            f"- Threshold: `{candidate['threshold']}`",
            f"- TP={candidate['TP']}, FP={candidate['FP']}, FN={candidate['FN']}, F1={candidate['F1']}",
            f"- FNs recuperados: `{candidate['fns_recuperados']}`",
            f"- FPs adicionados: `{candidate['fps_adicionados']}`",
            f"- TPs perdidos: `{candidate['tps_perdidos']}`",
            f"- Rule hits: `{candidate['rule_hits']}`",
        ])
    else:
        lines.append("Nenhum candidato seguro encontrado.")

    lines.extend([
        "",
        "## Residual FNs pós-C1",
        "",
        f"- Quantidade residual no conjunto único: `{len(residual_fn_ranking)}`",
        "",
    ])

    if not residual_fn_ranking.empty:
        lines.append("| Tx | Valor | Rel. meses | LGBM | IF | SE | BEH | Score | Shadow ensemble |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

        for _, r in residual_fn_ranking.head(12).iterrows():
            lines.append(
                f"| `{r.get('transaction_id', '')}` | {float(r.get('vl_pix', 0)):.2f} | "
                f"{float(r.get('qt_tempo_relacionamento_mes', 0)):.0f} | "
                f"{float(r.get('lgbm_raw', 0)):.5f} | {float(r.get('if_percentile', 0)):.5f} | "
                f"{float(r.get('se_score', 0)):.2f} | {float(r.get('beh_score', 0)):.2f} | "
                f"{float(r.get('score_final', 0)):.2f} | "
                f"{float(r.get('shadow_prob_ENSEMBLE_MEAN', 0)):.5f} |"
            )

    lines.extend([
        "",
        "## Decisão",
        "",
        candidate_decision["next_action"],
        "",
    ])

    if candidate_decision["status"] == "SEM_CANDIDATO_SEGURO":
        lines.extend([
            "O meta-learner shadow não encontrou um overlay seguro para recuperar FN sem custo em FP.",
            "Isso sugere que os FNs remanescentes podem estar próximos do limite dos sinais atuais.",
            "Próximo caminho: novas fontes de dados ou análise manual dos FNs residuais.",
        ])
    elif candidate_decision["status"].startswith("CANDIDATO_STRICT"):
        lines.extend([
            "Há candidato shadow estrito. Ainda não promover.",
            "Próximo passo permitido: EXP-007B quick-E2E com exatamente 1 candidato.",
        ])
    else:
        lines.extend([
            "Há candidato relaxado, mas com possível FP. Não rodar E2E antes de auditoria manual.",
        ])

    path.write_text("\n".join(lines), encoding="utf-8")


def write_next_experiment(path: Path, decision: dict[str, Any]) -> None:
    if decision["status"].startswith("CANDIDATO_STRICT"):
        title = "EXP-007B — Quick-E2E Meta-Learner Overlay"
        objective = (
            "Validar em runtime real um único overlay shadow, sem grid, usando baseline pós-C1."
        )
        constraints = [
            "Rodar baseline + 1 candidato.",
            "Sem múltiplos thresholds.",
            "Parar se FP subir, FN não cair ou F1 piorar.",
        ]
    elif decision["status"].startswith("CANDIDATO_RELAXED"):
        title = "EXP-007B-AUDIT — Manual FP Audit"
        objective = (
            "Auditar os FPs adicionados pelo candidato relaxado antes de qualquer E2E."
        )
        constraints = [
            "Não rodar E2E.",
            "Inspecionar casos adicionados.",
            "Só avançar se FP adicionado for justificável ou removível.",
        ]
    else:
        title = "FASE 3 — Novas fontes de dados / reputação do recebedor"
        objective = (
            "Os sinais atuais não parecem suficientes para reduzir FNs residuais sem custo. "
            "Priorizar dados de recebedor, grafo, device/session, MED/contestação."
        )
        constraints = [
            "Não criar novas regras manuais sobre first_receiver.",
            "Não treinar outro LGBM apenas com os mesmos sinais.",
            "Documentar FNs remanescentes como provavelmente data-limited.",
        ]

    lines = [
        "# Próximo experimento recomendado",
        "",
        f"## {title}",
        "",
        "## Objetivo",
        "",
        objective,
        "",
        "## Restrições",
        "",
    ]

    for c in constraints:
        lines.append(f"- {c}")

    path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EXP-007A — Meta-Learner Shadow")
    print("=" * 72)

    print("[1/7] Carregando baseline e aplicando C1 pós-runtime...")
    all_df, unique_df = load_post_c1_predictions()

    print(f"[OK] Linhas totais: {len(all_df)}")
    print(f"[OK] Linhas únicas para shadow: {len(unique_df)}")
    print(f"[OK] Fraudes únicas: {int(unique_df['is_fraud'].sum())}")

    baseline_rows = []

    for seed, g in all_df.groupby("seed"):
        baseline_rows.append(compute_metrics(g, f"POST_C1_seed_{seed}"))

    baseline_rows.append(compute_metrics(unique_df, "POST_C1_UNIQUE_UNION"))
    baseline_metrics = pd.DataFrame(baseline_rows)
    baseline_metrics.to_csv(OUTPUT_DIR / "01_post_c1_baseline_metrics.csv", index=False, encoding="utf-8-sig")

    write_json(
        OUTPUT_DIR / "00_input_summary.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "input_dir": str(INPUT_DIR),
            "output_dir": str(OUTPUT_DIR),
            "rows_total": int(len(all_df)),
            "rows_unique": int(len(unique_df)),
            "frauds_unique": int(unique_df["is_fraud"].sum()),
            "post_c1_baseline_metrics": baseline_rows,
            "note": (
                "C1 é reaplicada em artifact-only com min_score=58 para representar baseline pós-C1."
            ),
        },
    )

    print("[2/7] Treinando meta-learners shadow com OOF...")
    oof, model_df, importance_df = oof_predict_models(unique_df)

    model_df.to_csv(OUTPUT_DIR / "02_oof_model_metrics.csv", index=False, encoding="utf-8-sig")
    importance_df.to_csv(OUTPUT_DIR / "06_feature_importance.csv", index=False, encoding="utf-8-sig")

    print("[OK] Modelos shadow:")
    print(model_df.to_string(index=False))

    print("[3/7] Rodando threshold sweep artifact-only...")
    sweep = threshold_sweep(oof, model_df)
    sweep.to_csv(OUTPUT_DIR / "03_threshold_sweep.csv", index=False, encoding="utf-8-sig")

    print("[4/7] Gerando ranking dos FNs residuais...")
    residual_fn_ranking = build_residual_fn_ranking(oof)
    residual_fn_ranking.to_csv(OUTPUT_DIR / "04_residual_fn_ranking.csv", index=False, encoding="utf-8-sig")

    print("[5/7] Selecionando candidato, se existir...")
    decision = choose_candidate(sweep)
    write_json(OUTPUT_DIR / "05_candidate_overlay_eval.json", decision)

    print(f"[OK] Decisão: {decision['status']}")

    print("[6/7] Escrevendo relatório...")
    write_recommendation(
        OUTPUT_DIR / "07_recommendation.md",
        baseline_metrics=baseline_metrics,
        model_df=model_df,
        sweep=sweep,
        candidate_decision=decision,
        residual_fn_ranking=residual_fn_ranking,
    )

    print("[7/7] Escrevendo próximo experimento...")
    write_next_experiment(
        OUTPUT_DIR / "08_next_experiment_spec.md",
        decision=decision,
    )

    print()
    print("[OK] EXP-007A concluído sem E2E.")
    print(f"[OK] Artefatos em: {OUTPUT_DIR}")
    print()
    print("Arquivos principais:")
    print(f"  {OUTPUT_DIR / '01_post_c1_baseline_metrics.csv'}")
    print(f"  {OUTPUT_DIR / '02_oof_model_metrics.csv'}")
    print(f"  {OUTPUT_DIR / '03_threshold_sweep.csv'}")
    print(f"  {OUTPUT_DIR / '04_residual_fn_ranking.csv'}")
    print(f"  {OUTPUT_DIR / '05_candidate_overlay_eval.json'}")
    print(f"  {OUTPUT_DIR / '07_recommendation.md'}")
    print(f"  {OUTPUT_DIR / '08_next_experiment_spec.md'}")


if __name__ == "__main__":
    main()