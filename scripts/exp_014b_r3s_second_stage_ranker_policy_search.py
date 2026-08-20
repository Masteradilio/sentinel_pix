#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3S — Second-stage FP Ranker / Commercial Policy Search

Objetivo:
  Sair da otica FN=0 a qualquer custo e buscar uma politica comercial com:
    - FN <= 5 (abaixo de 6)
    - recall >= 95%
    - FPR <= 1.5%

Contexto:
  O primeiro estagio R3Q-FROZEN/R3Q recomendado preserva FN=0, mas ainda tem muitos FPs.
  O R3S treina/avalia um segundo estagio apenas dentro dos alertas atuais para separar:
    alerta final / alta prioridade vs democao para baixo risco.

Importante:
  - Nao altera modelo em producao.
  - Nao usa rescues.
  - Nao usa runtime externo.
  - Nao usa temporal_split/event_month/source_dataset/sample_strategy como feature.
  - Gera cenarios de politica e evidencia de robustez antes de qualquer frozen/promoção.

Uso:
  python scripts/exp_014b_r3s_second_stage_ranker_policy_search.py

Saidas:
  resultados/experimentos/EXP-014B-R3S/
    00_run_summary.json
    01_input_contract.json
    02_base_validation.json
    03_ranker_model_scores.csv
    04_policy_frontier.csv
    05_selected_policy.json
    06_robustness_by_segment.csv
    07_policy_artifact_recommended.json
    08_predictions_recommended.csv
    09_exp014b_r3s_report.md
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
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None
try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "dados").exists() else Path.cwd()

DEFAULT_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3R" / "09_predictions_recommended.csv"
DEFAULT_R3Q_ARTIFACT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3Q" / "07_policy_artifact_recommended.json"
DEFAULT_OUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3S"

BASE_COL_CANDIDATES = [
    "exp014b_r3q_frozen_pred",
    "exp014b_r3q_recommended_pred",
    "exp014b_r3p_frozen_pred",
]

FINAL_COL = "exp014b_r3s_recommended_pred"
SCORE_COL = "exp014b_r3s_second_stage_score"
POLICY_COL = "exp014b_r3s_policy_name"

NUMERIC_FEATURES = [
    # Scores/modelos
    "lgbm_r4_score", "lgbm_raw", "lgbm_mapped", "score_final",
    "if_percentile", "if_raw", "se_score", "se_patterns_count",
    "beh_score", "beh_factors_count", "topaz_risk_score",
    "runtime_flagged", "peso_total", "cascade_triggered", "veto_aplicado",
    # Valor/perfil
    "vl_pix", "nr_idade", "qt_tempo_relacionamento_mes",
    "is_first_tx_trimestre", "first_receiver_flag", "first_receiver_flag_real",
    "burst_30m_flag", "pix_key_random_flag", "perfil_vulneravel_se_flag",
    "mbk_available_flag", "mbk_completeness_score",
    # Features Big Data / janela
    "qtd_pix_pagador_7d", "qtd_pix_pagador_30d", "qtd_pix_pagador_90d", "qtd_pix_pagador_180d",
    "valor_total_pagador_7d", "valor_total_pagador_30d", "valor_total_pagador_90d", "valor_total_pagador_180d",
    "max_qtd_pix_dia_pagador_7d", "max_qtd_pix_dia_pagador_30d", "valor_maximo_pix_pagador_180d",
    "soma_recebedores_distintos_dia_180d",
    "qtd_pix_mesmo_recebedor_30d", "qtd_pix_mesmo_recebedor_90d", "qtd_pix_mesmo_recebedor_180d",
    "valor_total_para_recebedor_30d", "valor_total_para_recebedor_90d", "valor_total_para_recebedor_180d",
    "dias_desde_primeiro_envio_recebedor",
    "qtd_pix_recebidos_30d", "qtd_pix_recebidos_90d", "qtd_pix_recebidos_180d",
    "valor_total_recebido_30d", "valor_total_recebido_90d", "valor_total_recebido_180d",
    "soma_pagadores_distintos_dia_recebedor_180d", "max_qtd_pix_recebidos_dia_180d",
    "burst_daily_7d_flag", "ratio_valor_media_pagador_90d", "ratio_valor_maximo_pagador_180d",
    # Tempo/sessao
    "hour", "latencia_rede_ms", "tempo_interacao_ms", "tempo_processamento_host_ms",
]

CATEGORICAL_FEATURES = [
    "ds_tipo_chave_norm", "periodo_dia", "value_band",
    "lgbm_bin", "if_bin", "score_bin", "ratio_bin", "qtd_rec_bin", "vl_bin", "valor_rec_bin",
    "module_quiet", "se_worst_pattern", "metodo_autenticacao",
]

SEGMENT_COLS = [
    "temporal_split", "event_month", "ds_tipo_chave_norm", "value_band",
    "periodo_dia", "mbk_available_flag",
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
    for c in BASE_COL_CANDIDATES + [FINAL_COL]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    if "event_month" not in df.columns and "event_datetime" in df.columns:
        dt = pd.to_datetime(df["event_datetime"], errors="coerce")
        df["event_month"] = dt.dt.to_period("M").astype(str)
    return df.reset_index(drop=True)


def pick_base_col(df: pd.DataFrame) -> str:
    for c in BASE_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise RuntimeError(f"Nenhuma coluna base encontrada: {BASE_COL_CANDIDATES}")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 8),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 8),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 8),
        "fpr": round(float(fp / max(fp + tn, 1)), 8),
    }


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n) + (z * z / (4 * n * n))) ** 0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def prepare_features(df_alerts: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    num = [c for c in NUMERIC_FEATURES if c in df_alerts.columns]
    cat = [c for c in CATEGORICAL_FEATURES if c in df_alerts.columns]
    out = df_alerts[num + cat].copy()
    for c in num:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for c in cat:
        out[c] = out[c].astype("string").fillna("<MISSING>").astype(str)
    return out, num, cat


def encode_ranker_features(df_alerts: pd.DataFrame, num: list[str], cat: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Prepara matriz leve para LightGBM: numericos + categoricos codificados.

    A codificacao e salva em mappings apenas para auditoria; este script e experimental,
    nao e o artifact final de producao.
    """
    X = pd.DataFrame(index=df_alerts.index)
    mappings: dict[str, list[str]] = {}
    for c in num:
        X[c] = pd.to_numeric(df_alerts[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for c in cat:
        ser = df_alerts[c].astype("string").fillna("<MISSING>").astype(str)
        cats = sorted(ser.unique().tolist())
        mappings[c] = cats
        code_map = {v: i for i, v in enumerate(cats)}
        X[c] = ser.map(code_map).fillna(-1).astype(int)
    return X, mappings


def build_ranker_scores(
    X: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    out_dir: Path,
) -> dict[str, np.ndarray]:
    """Treina modelos leves de segundo estagio e retorna scores para todos os alertas."""
    scores: dict[str, np.ndarray] = {}
    if LGBMClassifier is None:
        return scores

    configs = {
        "lgbm_ranker_balanced": dict(
            n_estimators=90, learning_rate=0.05, num_leaves=15, max_depth=4,
            min_child_samples=25, subsample=0.9, colsample_bytree=0.85,
            reg_alpha=1.0, reg_lambda=8.0, class_weight="balanced", random_state=42,
            n_jobs=1, verbose=-1,
        ),
        "lgbm_ranker_conservative": dict(
            n_estimators=70, learning_rate=0.04, num_leaves=9, max_depth=3,
            min_child_samples=40, subsample=0.9, colsample_bytree=0.75,
            reg_alpha=2.0, reg_lambda=12.0, class_weight="balanced", random_state=123,
            n_jobs=1, verbose=-1,
        ),
    }

    for name, params in configs.items():
        try:
            clf = LGBMClassifier(**params)
            clf.fit(X.iloc[train_mask], y[train_mask])
            scores[name] = clf.predict_proba(X)[:, 1].astype(float)
            if joblib is not None:
                model_dir = out_dir / "models"
                model_dir.mkdir(exist_ok=True)
                joblib.dump(clf, model_dir / f"{name}.joblib")
        except Exception as e:
            log(f"WARN: falha ao treinar {name}: {e}")
    return scores


def add_heuristic_scores(df_alerts: pd.DataFrame) -> dict[str, np.ndarray]:
    def rank01(s: pd.Series, reverse: bool = False) -> np.ndarray:
        vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
        r = vals.rank(method="average", pct=True).fillna(0.0).to_numpy(dtype=float)
        return 1.0 - r if reverse else r

    scores: dict[str, np.ndarray] = {}
    pieces = []
    weights = []
    if "lgbm_r4_score" in df_alerts.columns:
        pieces.append(rank01(df_alerts["lgbm_r4_score"])); weights.append(3.0)
    if "lgbm_mapped" in df_alerts.columns:
        pieces.append(rank01(df_alerts["lgbm_mapped"])); weights.append(2.0)
    if "score_final" in df_alerts.columns:
        pieces.append(rank01(df_alerts["score_final"])); weights.append(1.5)
    if "if_percentile" in df_alerts.columns:
        pieces.append(rank01(df_alerts["if_percentile"])); weights.append(1.0)
    if "se_score" in df_alerts.columns:
        pieces.append(rank01(df_alerts["se_score"])); weights.append(1.0)
    if "beh_score" in df_alerts.columns:
        pieces.append(rank01(df_alerts["beh_score"])); weights.append(1.0)
    if pieces:
        w = np.asarray(weights, dtype=float)
        mat = np.vstack(pieces).T
        scores["heuristic_weighted_score"] = (mat @ w) / w.sum()
    for c in ["lgbm_r4_score", "lgbm_mapped", "score_final", "if_percentile"]:
        if c in df_alerts.columns:
            scores[f"raw_{c}"] = rank01(df_alerts[c])
    return scores


def score_model_quality(y: np.ndarray, score: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    yy = y[mask]
    ss = score[mask]
    out: dict[str, Any] = {}
    if len(np.unique(yy)) < 2:
        out["roc_auc"] = None
        out["average_precision"] = None
    else:
        try:
            out["roc_auc"] = round(float(roc_auc_score(yy, ss)), 8)
        except Exception:
            out["roc_auc"] = None
        try:
            out["average_precision"] = round(float(average_precision_score(yy, ss)), 8)
        except Exception:
            out["average_precision"] = None
    return out


def final_pred_from_alert_scores(n_rows: int, base_mask: np.ndarray, alert_scores: np.ndarray, threshold: float) -> np.ndarray:
    pred = np.zeros(n_rows, dtype=int)
    pred[base_mask] = (alert_scores >= threshold).astype(int)
    return pred


def policy_rows_for_score(
    df: pd.DataFrame,
    base_mask: np.ndarray,
    y: np.ndarray,
    alert_scores: np.ndarray,
    model_name: str,
    target_fpr: float,
    max_fn: int,
    min_recall: float,
) -> list[dict[str, Any]]:
    """Gera fronteira de thresholds de forma vetorizada sobre os alertas."""
    n_total = int(len(y))
    total_frauds = int((y == 1).sum())
    n_normals = int((y == 0).sum())
    target_fp = int(np.floor(n_normals * target_fpr))

    y_alert = y[base_mask].astype(int)
    score = np.asarray(alert_scores, dtype=float)
    order = np.argsort(-score)
    score_sorted = score[order]
    y_sorted = y_alert[order]

    cum_tp = np.cumsum(y_sorted == 1)
    cum_fp = np.cumsum(y_sorted == 0)

    # Avalia apenas pontos onde o threshold muda, isto e, fim de cada bloco de score.
    change = np.r_[score_sorted[1:] != score_sorted[:-1], True]
    idxs = np.where(change)[0]

    rows = []
    for i in idxs:
        th = float(score_sorted[i])
        tp = int(cum_tp[i])
        fp = int(cum_fp[i])
        fn = int(total_frauds - tp)
        tn = int(n_normals - fp)
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float((2 * precision * recall) / max(precision + recall, 1e-12))
        fpr = float(fp / max(fp + tn, 1))
        rows.append({
            "model_name": model_name,
            "threshold": th,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 8), "recall": round(recall, 8),
            "f1": round(f1, 8), "fpr": round(fpr, 8),
            "fp_removed_vs_base": int(((y == 0) & base_mask).sum() - fp),
            "fn_delta_vs_base": int(fn - int(((y == 1) & (~base_mask)).sum())),
            "target_fpr": target_fpr,
            "target_fp": target_fp,
            "target_gap_fp": int(max(0, fp - target_fp)),
            "target_fpr_reached": bool(fpr <= target_fpr),
            "fn_budget_ok": bool(fn <= max_fn),
            "recall_ok": bool(recall >= min_recall),
            "commercial_target_ok": bool(fn <= max_fn and recall >= min_recall and fpr <= target_fpr),
        })
    return rows


def choose_policy(frontier: pd.DataFrame, max_fn: int, min_recall: float, target_fpr: float) -> dict[str, Any]:
    if frontier.empty:
        raise RuntimeError("Fronteira vazia")
    ok = frontier[
        (frontier["fn"] <= max_fn)
        & (frontier["recall"] >= min_recall)
        & (frontier["fpr"] <= target_fpr)
    ].copy()
    if not ok.empty:
        ok = ok.sort_values(["fp", "fn", "fpr", "threshold"], ascending=[True, True, True, False])
        row = ok.iloc[0].to_dict()
        row["selection_reason"] = "MEETS_COMMERCIAL_TARGET"
        return row
    admissible = frontier[(frontier["fn"] <= max_fn) & (frontier["recall"] >= min_recall)].copy()
    if not admissible.empty:
        admissible = admissible.sort_values(["target_gap_fp", "fp", "fn", "fpr"], ascending=[True, True, True, True])
        row = admissible.iloc[0].to_dict()
        row["selection_reason"] = "BEST_GAP_WITHIN_FN_RECALL_BUDGET"
        return row
    # Fallback: menor gap mesmo sem bater budget.
    tmp = frontier.copy().sort_values(["target_gap_fp", "fn", "fp"], ascending=[True, True, True])
    row = tmp.iloc[0].to_dict()
    row["selection_reason"] = "BEST_GAP_OVERALL_BUT_BUDGET_NOT_MET"
    return row


def robustness_by_segment(df: pd.DataFrame, y: np.ndarray, pred: np.ndarray, base_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in SEGMENT_COLS:
        if c not in df.columns:
            continue
        vals = df[c].astype("string").fillna("<MISSING>").astype(str)
        for v, idx in vals.groupby(vals).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            yy = y[idx]
            pp = pred[idx]
            bb = base_pred[idx]
            m = metrics(yy, pp)
            bm = metrics(yy, bb)
            rows.append({
                "segment_col": c,
                "segment_value": str(v),
                "n_rows": int(len(idx)),
                "n_frauds": int(yy.sum()),
                "base_fp": bm["fp"],
                "final_fp": m["fp"],
                "fp_removed": int(bm["fp"] - m["fp"]),
                "fn_delta": int(m["fn"] - bm["fn"]),
                "final_tp": m["tp"],
                "final_fp": m["fp"],
                "final_fn": m["fn"],
                "final_recall": m["recall"],
                "final_fpr": m["fpr"],
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["fn_delta", "fp_removed", "n_frauds"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def make_report(summary: dict[str, Any], selected: dict[str, Any], model_scores: pd.DataFrame, frontier: pd.DataFrame, robust: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014B-R3S - Second-stage FP Ranker / Commercial Policy Search")
    lines.append("")
    lines.append("## Resultado executivo")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- All pass: `{summary['all_pass']}`")
    lines.append(f"- Base: `{summary['base_metrics']}`")
    lines.append(f"- Política recomendada: `{summary['recommended_policy_name']}`")
    lines.append(f"- Métricas recomendadas: `{summary['recommended_metrics']}`")
    lines.append(f"- FP removidos vs base: `{summary['fp_removed_vs_base']}`")
    lines.append(f"- FN delta vs base: `{summary['fn_delta_vs_base']}`")
    lines.append(f"- Alvo FPR: `{summary['target_fpr']}` | FP max alvo: `{summary['target_fp']}`")
    lines.append(f"- Gap até alvo FPR: `{summary['target_gap_fp']}` FP")
    lines.append(f"- Target comercial atingido: `{summary['commercial_target_reached']}`")
    lines.append("")
    lines.append("## Qualidade dos scores/rankers")
    if model_scores.empty:
        lines.append("Sem scores avaliados.")
    else:
        show = [c for c in ["model_name", "split", "roc_auc", "average_precision", "n_alerts", "n_frauds"] if c in model_scores.columns]
        lines.append(model_scores[show].to_markdown(index=False))
    lines.append("")
    lines.append("## Política selecionada")
    lines.append("```json")
    lines.append(json.dumps(selected, ensure_ascii=False, indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## Fronteira - melhores por modelo")
    if frontier.empty:
        lines.append("Fronteira vazia.")
    else:
        show = ["model_name", "threshold", "tp", "fp", "fn", "precision", "recall", "fpr", "fp_removed_vs_base", "target_gap_fp", "commercial_target_ok"]
        tmp = frontier.sort_values(["commercial_target_ok", "target_gap_fp", "fp"], ascending=[False, True, True]).groupby("model_name").head(5)
        lines.append(tmp[[c for c in show if c in tmp.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Robustez por segmento")
    if robust.empty:
        lines.append("Robustez vazia.")
    else:
        show = ["segment_col", "segment_value", "n_rows", "n_frauds", "fp_removed", "fn_delta", "final_tp", "final_fp", "final_fn", "final_recall", "final_fpr"]
        lines.append(robust[[c for c in show if c in robust.columns]].head(40).to_markdown(index=False))
    lines.append("")
    lines.append("## Decisao sugerida")
    if summary.get("commercial_target_reached"):
        lines.append("O segundo estágio encontrou uma política que atende FN/recall/FPR. Próximo passo: validação congelada e auditoria temporal estrita do R3S.")
    else:
        lines.append("O segundo estágio ainda não atingiu o alvo comercial. Usar o resultado para diagnosticar gap e decidir entre novas features/ranker mais forte ou revisão do objetivo operacional.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--artifact", default=str(DEFAULT_R3Q_ARTIFACT))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--target-fpr", type=float, default=0.015)
    ap.add_argument("--max-fn", type=int, default=5)
    ap.add_argument("--min-recall", type=float, default=0.95)
    ap.add_argument("--train-split", default="TRAIN")
    ap.add_argument("--validation-split", default="VALIDATION")
    ap.add_argument("--holdout-split", default="HOLDOUT")
    ap.add_argument("--no-write-predictions", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    artifact_path = Path(args.artifact)
    if not input_path.exists():
        raise FileNotFoundError(f"input nao encontrado: {input_path}")

    log("=" * 88)
    log("EXP-014B-R3S — Second-stage FP Ranker / Commercial Policy Search")
    log("=" * 88)

    df = normalize(pd.read_csv(input_path, low_memory=False))
    y = df["is_fraud"].to_numpy(dtype=int)
    base_col = pick_base_col(df)
    base_pred = df[base_col].to_numpy(dtype=int)
    base_mask = base_pred == 1
    base_m = metrics(y, base_pred)
    n_normals = int((y == 0).sum())
    target_fp = int(np.floor(n_normals * args.target_fpr))

    missing = []
    if "temporal_split" not in df.columns:
        missing.append("temporal_split")
    if not base_mask.any():
        missing.append("base alerts empty")
    contract = {
        "n_rows": int(len(df)),
        "n_frauds": int(y.sum()),
        "n_normals": n_normals,
        "input_path": str(input_path),
        "artifact_path": str(artifact_path),
        "base_col": base_col,
        "base_alerts": int(base_mask.sum()),
        "base_frauds_in_alerts": int(((y == 1) & base_mask).sum()),
        "target_fpr": args.target_fpr,
        "target_fp": target_fp,
        "max_fn": args.max_fn,
        "min_recall": args.min_recall,
        "missing": missing,
        "contract_ok": not missing,
    }
    dump_json(contract, out / "01_input_contract.json")
    if missing:
        raise RuntimeError(f"Contrato falhou: {missing}")

    wl, wh = wilson_ci(base_m["tp"], int(y.sum()))
    base_validation = {
        "base_metrics": base_m,
        "wilson_low": wl,
        "wilson_high": wh,
        "fn_within_new_budget": bool(base_m["fn"] <= args.max_fn),
        "recall_ok": bool(base_m["recall"] >= args.min_recall),
        "fpr_target_ok": bool(base_m["fpr"] <= args.target_fpr),
        "all_pass": bool(base_m["fn"] <= args.max_fn and base_m["recall"] >= args.min_recall),
        "status": "PASS_BASE_VALIDATED" if (base_m["fn"] <= args.max_fn and base_m["recall"] >= args.min_recall) else "FAIL_BASE_OUTSIDE_BUDGET",
    }
    dump_json(base_validation, out / "02_base_validation.json")

    df_alerts = df.loc[base_mask].copy().reset_index(drop=False).rename(columns={"index": "_orig_index"})
    y_alerts = df_alerts["is_fraud"].to_numpy(dtype=int)
    X_alerts, num, cat = prepare_features(df_alerts)

    splits = df_alerts["temporal_split"].astype("string").fillna("<MISSING>").astype(str) if "temporal_split" in df_alerts.columns else pd.Series([args.train_split] * len(df_alerts))
    train_mask = splits.eq(args.train_split).to_numpy()
    val_mask = splits.eq(args.validation_split).to_numpy()
    hold_mask = splits.eq(args.holdout_split).to_numpy()

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise RuntimeError("Split temporal insuficiente para treinar/validar ranker")

    log(f"Base metrics: {base_m}")
    log(f"Alertas base: {len(df_alerts)} | fraudes nos alertas: {int(y_alerts.sum())}")
    log(f"Target FPR <= {args.target_fpr:.4f} => FP <= {target_fp}")

    alert_scores_by_model: dict[str, np.ndarray] = {}
    quality_rows = []

    # Heuristicas sem treino.
    for name, score in add_heuristic_scores(df_alerts).items():
        alert_scores_by_model[name] = score.astype(float)

    # Modelos supervisionados treinados apenas no TRAIN.
    X_encoded, category_mappings = encode_ranker_features(df_alerts, num, cat)
    for name, score in build_ranker_scores(X_encoded, y_alerts, train_mask, out).items():
        alert_scores_by_model[name] = score

    for name, score in alert_scores_by_model.items():
        for split_name, mask in [("TRAIN", train_mask), ("VALIDATION", val_mask), ("HOLDOUT", hold_mask), ("ALL_ALERTS", np.ones(len(score), dtype=bool))]:
            q = score_model_quality(y_alerts, score, mask)
            quality_rows.append({
                "model_name": name,
                "split": split_name,
                "n_alerts": int(mask.sum()),
                "n_frauds": int(y_alerts[mask].sum()),
                **q,
            })
    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(out / "03_ranker_model_scores.csv", index=False)

    frontier_rows = []
    for name, score in alert_scores_by_model.items():
        frontier_rows.extend(policy_rows_for_score(df, base_mask, y, score, name, args.target_fpr, args.max_fn, args.min_recall))
    frontier = pd.DataFrame(frontier_rows)
    if frontier.empty:
        raise RuntimeError("Nenhum score de segundo estagio foi gerado")
    frontier = frontier.sort_values(["commercial_target_ok", "target_gap_fp", "fp", "fn"], ascending=[False, True, True, True]).reset_index(drop=True)
    frontier.to_csv(out / "04_policy_frontier.csv", index=False)

    selected = choose_policy(frontier, args.max_fn, args.min_recall, args.target_fpr)
    selected_model = str(selected["model_name"])
    selected_threshold = float(selected["threshold"])
    selected_score = alert_scores_by_model[selected_model]
    final_pred = final_pred_from_alert_scores(len(df), base_mask, selected_score, selected_threshold)
    final_m = metrics(y, final_pred)
    fp_removed = int(base_m["fp"] - final_m["fp"])
    fn_delta = int(final_m["fn"] - base_m["fn"])
    wl2, wh2 = wilson_ci(final_m["tp"], int(y.sum()))

    selected_policy = {
        "policy_name": f"r3s_second_stage_{selected_model}",
        "selected_model": selected_model,
        "selected_threshold": selected_threshold,
        "selection_reason": selected.get("selection_reason"),
        "base_col": base_col,
        "final_pred_col": FINAL_COL,
        "score_col": SCORE_COL,
        "feature_columns_numeric": num,
        "feature_columns_categorical": cat,
        "category_mappings": category_mappings if "category_mappings" in locals() else {},
        "target_fpr": args.target_fpr,
        "target_fp": target_fp,
        "max_fn": args.max_fn,
        "min_recall": args.min_recall,
        "base_metrics": base_m,
        "recommended_metrics": final_m,
        "fp_removed_vs_base": fp_removed,
        "fn_delta_vs_base": fn_delta,
        "target_gap_fp": int(max(0, final_m["fp"] - target_fp)),
        "commercial_target_reached": bool(final_m["fn"] <= args.max_fn and final_m["recall"] >= args.min_recall and final_m["fpr"] <= args.target_fpr),
        "wilson_low": wl2,
        "wilson_high": wh2,
    }
    dump_json(selected_policy, out / "05_selected_policy.json")

    df[SCORE_COL] = np.nan
    df.loc[base_mask, SCORE_COL] = selected_score
    df[FINAL_COL] = final_pred.astype(int)
    df[POLICY_COL] = selected_policy["policy_name"]

    robust = robustness_by_segment(df, y, final_pred, base_pred)
    robust.to_csv(out / "06_robustness_by_segment.csv", index=False)

    objective_status = "DONE_R3S_SECOND_STAGE_POLICY_SEARCH"
    objective_status += "_COMMERCIAL_TARGET_REACHED" if selected_policy["commercial_target_reached"] else "_COMMERCIAL_TARGET_NOT_REACHED"
    objective_status += "_FN_RECALL_BUDGET_OK" if (final_m["fn"] <= args.max_fn and final_m["recall"] >= args.min_recall) else "_FN_RECALL_BUDGET_FAIL"
    objective_status += "_FPR_TARGET_OK" if final_m["fpr"] <= args.target_fpr else "_FPR_TARGET_FAIL"

    artifact = {
        "experiment": "EXP-014B-R3S",
        "policy_name": selected_policy["policy_name"],
        "objective_status": objective_status,
        "input_path": str(input_path),
        "source_artifact": str(artifact_path),
        "base_col": base_col,
        "final_pred_col": FINAL_COL,
        "score_col": SCORE_COL,
        "selected_policy": selected_policy,
        "base_validation": base_validation,
        "model_quality": quality_df.to_dict(orient="records"),
        "constraints": {
            "target_fpr": args.target_fpr,
            "target_fp": target_fp,
            "max_fn": args.max_fn,
            "min_recall": args.min_recall,
            "train_split": args.train_split,
            "validation_split": args.validation_split,
            "holdout_split": args.holdout_split,
        },
        "notes": [
            "Second-stage ranker trained/evaluated only inside first-stage alerts.",
            "temporal_split, event_month, source_dataset and sample_strategy are not used as model features.",
            "Promotion requires frozen validation and business review of FN cases if FN > 0.",
        ],
    }
    dump_json(artifact, out / "07_policy_artifact_recommended.json")

    if not args.no_write_predictions:
        df.to_csv(out / "08_predictions_recommended.csv", index=False)

    summary = {
        "experiment": "EXP-014B-R3S",
        "status": "DONE",
        "objective_status": objective_status,
        "n_rows": int(len(df)),
        "n_frauds": int(y.sum()),
        "n_normals": n_normals,
        "input_path": str(input_path),
        "artifact_path": str(artifact_path),
        "base_col": base_col,
        "base_metrics": base_m,
        "target_fpr": args.target_fpr,
        "target_fp": target_fp,
        "max_fn": args.max_fn,
        "min_recall": args.min_recall,
        "recommended_policy_name": selected_policy["policy_name"],
        "selected_model": selected_model,
        "selected_threshold": selected_threshold,
        "selection_reason": selected.get("selection_reason"),
        "recommended_metrics": final_m,
        "fp_removed_vs_base": fp_removed,
        "fn_delta_vs_base": fn_delta,
        "target_gap_fp": selected_policy["target_gap_fp"],
        "commercial_target_reached": selected_policy["commercial_target_reached"],
        "wilson_low": wl2,
        "wilson_high": wh2,
        "n_rankers_evaluated": int(len(alert_scores_by_model)),
        "all_pass": bool(final_m["fn"] <= args.max_fn and final_m["recall"] >= args.min_recall),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(out),
    }
    dump_json(summary, out / "00_run_summary.json")

    report = make_report(summary, selected_policy, quality_df, frontier, robust)
    (out / "09_exp014b_r3s_report.md").write_text(report, encoding="utf-8")

    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        out / "00_run_summary.json",
        out / "01_input_contract.json",
        out / "02_base_validation.json",
        out / "03_ranker_model_scores.csv",
        out / "04_policy_frontier.csv",
        out / "05_selected_policy.json",
        out / "06_robustness_by_segment.csv",
        out / "07_policy_artifact_recommended.json",
        out / "09_exp014b_r3s_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
