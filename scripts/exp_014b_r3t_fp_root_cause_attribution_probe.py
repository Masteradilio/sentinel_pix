#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3T - FP Root Cause Attribution Probe

Objetivo:
  Diagnosticar por que o benchmark atual ainda produz muitos falsos positivos.
  Esta rodada NAO promove nova politica. Ela gera evidencias para orientar
  a proxima iteracao com novas estrategias.

Contexto:
  Melhor benchmark operacional atual: R3Q-FROZEN/R3Q replay
    TP=1465, FP=4074, FN=0, FPR~3.625%
  Meta comercial relaxada:
    FN <= 5, recall >= 95%, FPR <= 1.5%  (FP <= ~1685)

O script mede:
  - pareto de falsos positivos por feature e segmento;
  - hotspots de baixa precisao dentro dos alertas atuais;
  - separabilidade univariada FP vs TP nos scores/features numericas;
  - interacoes de 2 features que concentram FP;
  - hipoteses auditaveis para a proxima rodada.

Uso:
  python scripts/exp_014b_r3t_fp_root_cause_attribution_probe.py

Saidas:
  resultados/experimentos/EXP-014B-R3T/
    00_run_summary.json
    01_input_contract.json
    02_base_metrics.json
    03_fp_pareto_by_feature.csv
    04_low_precision_segments.csv
    05_numeric_separation.csv
    06_pairwise_fp_hotspots.csv
    07_root_cause_hypotheses.json
    08_exp014b_r3t_report.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "dados").exists() else Path.cwd()

DEFAULT_OUTPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3T"
DEFAULT_INPUT_CANDIDATES = [
    PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3S" / "08_predictions_recommended.csv",
    PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3R" / "09_predictions_recommended.csv",
    PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3Q" / "08_predictions_recommended.csv",
]

PREFERRED_BASE_COLS = [
    "exp014b_r3q_frozen_pred",
    "exp014b_r3q_recommended_pred",
    "exp014b_r3p_frozen_pred",
]

TARGET_FPR = 0.015
MAX_FN = 5
MIN_RECALL = 0.95

CATEGORICAL_CANDIDATES = [
    "ds_tipo_chave_norm", "periodo_dia", "value_band",
    "lgbm_bin", "if_bin", "score_bin", "ratio_bin", "qtd_rec_bin", "vl_bin", "valor_rec_bin",
    "module_quiet", "se_worst_pattern", "metodo_autenticacao",
    "mbk_available_flag", "first_receiver_flag", "first_receiver_flag_real", "pix_key_random_flag",
    "burst_30m_flag", "is_first_tx_trimestre", "perfil_vulneravel_se_flag",
    "runtime_flagged", "cascade_triggered", "veto_aplicado",
    "topaz_transacao_rejeitada", "device_missing_flag", "host_time_missing_flag", "topaz_missing_flag",
]

NUMERIC_CANDIDATES = [
    "lgbm_r4_score", "lgbm_raw", "lgbm_mapped", "score_final",
    "if_percentile", "if_raw", "se_score", "se_patterns_count", "beh_score", "beh_factors_count",
    "topaz_risk_score", "peso_total", "vl_pix", "nr_idade", "qt_tempo_relacionamento_mes",
    "qtd_pix_pagador_7d", "qtd_pix_pagador_30d", "qtd_pix_pagador_90d", "qtd_pix_pagador_180d",
    "valor_total_pagador_7d", "valor_total_pagador_30d", "valor_total_pagador_90d", "valor_total_pagador_180d",
    "valor_maximo_pix_pagador_180d", "soma_recebedores_distintos_dia_180d",
    "qtd_pix_mesmo_recebedor_30d", "qtd_pix_mesmo_recebedor_90d", "qtd_pix_mesmo_recebedor_180d",
    "valor_total_para_recebedor_30d", "valor_total_para_recebedor_90d", "valor_total_para_recebedor_180d",
    "dias_desde_primeiro_envio_recebedor",
    "qtd_pix_recebidos_30d", "qtd_pix_recebidos_90d", "qtd_pix_recebidos_180d",
    "valor_total_recebido_30d", "valor_total_recebido_90d", "valor_total_recebido_180d",
    "soma_pagadores_distintos_dia_recebedor_180d", "max_qtd_pix_recebidos_dia_180d",
    "ratio_valor_media_pagador_90d", "ratio_valor_maximo_pagador_180d",
    "hour", "latencia_rede_ms", "tempo_interacao_ms", "tempo_processamento_host_ms",
]

PAIRWISE_ANCHORS = [
    "ds_tipo_chave_norm", "value_band", "periodo_dia", "lgbm_bin", "if_bin", "score_bin", "ratio_bin", "qtd_rec_bin", "vl_bin", "module_quiet", "mbk_available_flag",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def pick_existing_path(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().split(".")[-1] for c in df.columns]
    if "is_fraud" not in df.columns:
        raise RuntimeError("Coluna obrigatoria ausente: is_fraud")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)
    for c in df.columns:
        if c.startswith("exp014b_") and c.endswith("_pred"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    if "event_datetime" in df.columns:
        df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
        if "event_month" not in df.columns:
            df["event_month"] = df["event_datetime"].dt.to_period("M").astype(str)
    return df.reset_index(drop=True)


def pick_base_col(df: pd.DataFrame, requested: str | None = None) -> str:
    if requested and requested in df.columns:
        return requested
    for c in PREFERRED_BASE_COLS:
        if c in df.columns:
            return c
    pred_cols = [c for c in df.columns if c.startswith("exp014b_") and c.endswith("_pred")]
    if not pred_cols:
        raise RuntimeError("Nenhuma coluna de predicao exp014b_*_pred encontrada")
    return pred_cols[-1]


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


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def num(c: str, default: float = 0.0) -> pd.Series:
        return pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    def qbin(col: str, out: str, bins: list[float], prefix: str):
        if out in df.columns or col not in df.columns:
            return
        labels = []
        edges = [-np.inf] + bins + [np.inf]
        for a, b in zip(edges[:-1], edges[1:]):
            if np.isneginf(a): labels.append(f"{prefix}_LT_{b:g}")
            elif np.isposinf(b): labels.append(f"{prefix}_GE_{a:g}")
            else: labels.append(f"{prefix}_{a:g}_{b:g}")
        df[out] = pd.cut(num(col), bins=edges, labels=labels, include_lowest=True).astype("string").fillna(f"{prefix}_MISSING").astype(str)
    # Recreate common bins when absent.
    for col in ["lgbm_r4_score", "lgbm_raw", "lgbm_mapped"]:
        if col in df.columns:
            qbin(col, "lgbm_bin", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1], "lgbm")
            break
    qbin("if_percentile", "if_bin", [0.32, 0.5, 0.7, 0.85, 0.95], "if")
    qbin("score_final", "score_bin", [0.5, 1, 2, 3, 5, 10], "score")
    qbin("ratio_valor_media_pagador_90d", "ratio_bin", [0.05, 0.1, 0.2, 0.5, 1, 2, 5], "ratio")
    qbin("qtd_pix_recebidos_180d", "qtd_rec_bin", [0, 1, 2, 5, 10, 20, 50, 100], "qtdrec")
    qbin("vl_pix", "vl_bin", [20, 50, 100, 250, 500, 1000, 5000, 10000], "vl")
    qbin("valor_total_recebido_180d", "valor_rec_bin", [0, 100, 500, 1000, 5000, 10000, 25000], "valrec")
    if "module_quiet" not in df.columns:
        se = pd.to_numeric(df.get("se_score", 0), errors="coerce").fillna(0)
        sec = pd.to_numeric(df.get("se_patterns_count", df.get("se_pattern_count", 0)), errors="coerce").fillna(0)
        beh = pd.to_numeric(df.get("beh_score", df.get("behavioral_score", 0)), errors="coerce").fillna(0)
        behc = pd.to_numeric(df.get("beh_factors_count", df.get("behavioral_risk_factor_count", 0)), errors="coerce").fillna(0)
        runtime = pd.to_numeric(df.get("runtime_flagged", 0), errors="coerce").fillna(0)
        strong = (se >= 40) | (sec >= 2) | (beh >= 25) | (behc >= 2) | (runtime >= 1)
        df["module_quiet"] = np.where(strong, "module_strong", "module_quiet")
    return df


def segment_table(df: pd.DataFrame, base_col: str, cols: list[str], min_alerts: int, top_n_per_feature: int) -> pd.DataFrame:
    y = df["is_fraud"].to_numpy(dtype=int)
    pred = df[base_col].to_numpy(dtype=int)
    alerts = pred == 1
    total_fp = int(((alerts) & (y == 0)).sum())
    total_tp = int(((alerts) & (y == 1)).sum())
    base_precision = total_tp / max(total_tp + total_fp, 1)
    rows: list[dict[str, Any]] = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df.loc[alerts, c].astype("string").fillna("<MISSING>").astype(str)
        yy = y[alerts]
        tmp = pd.DataFrame({"value": s.to_numpy(), "y": yy})
        g = tmp.groupby("value", dropna=False)["y"].agg(["count", "sum"]).reset_index()
        g = g.rename(columns={"count": "alert_count", "sum": "tp"})
        g["fp"] = g["alert_count"] - g["tp"]
        g = g[g["alert_count"] >= min_alerts].copy()
        if g.empty:
            continue
        g["precision"] = g["tp"] / g["alert_count"].clip(lower=1)
        g["fp_share"] = g["fp"] / max(total_fp, 1)
        g["tp_share"] = g["tp"] / max(total_tp, 1)
        g["fp_to_tp_ratio"] = g["fp"] / g["tp"].replace(0, np.nan)
        g["precision_gap_vs_base"] = g["precision"] - base_precision
        g["feature"] = c
        g["base_precision"] = base_precision
        g["diagnostic_score"] = g["fp"] * (base_precision - g["precision"]).clip(lower=0)
        rows.extend(g.sort_values(["diagnostic_score", "fp"], ascending=[False, False]).head(top_n_per_feature).to_dict(orient="records"))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    cols_out = ["feature", "value", "alert_count", "tp", "fp", "precision", "base_precision", "precision_gap_vs_base", "fp_share", "tp_share", "fp_to_tp_ratio", "diagnostic_score"]
    return out[cols_out].sort_values(["diagnostic_score", "fp"], ascending=[False, False]).reset_index(drop=True)


def pairwise_hotspots(df: pd.DataFrame, base_col: str, cols: list[str], min_fp: int, min_alerts: int, top_n: int) -> pd.DataFrame:
    y = df["is_fraud"].to_numpy(dtype=int)
    pred = df[base_col].to_numpy(dtype=int)
    alerts = pred == 1
    total_fp = int(((alerts) & (y == 0)).sum())
    total_tp = int(((alerts) & (y == 1)).sum())
    base_precision = total_tp / max(total_tp + total_fp, 1)
    rows: list[dict[str, Any]] = []
    existing = [c for c in cols if c in df.columns]
    for a, b in itertools.combinations(existing, 2):
        sub = pd.DataFrame({
            a: df.loc[alerts, a].astype("string").fillna("<MISSING>").astype(str).to_numpy(),
            b: df.loc[alerts, b].astype("string").fillna("<MISSING>").astype(str).to_numpy(),
            "y": y[alerts],
        })
        g = sub.groupby([a, b], dropna=False)["y"].agg(["count", "sum"]).reset_index().rename(columns={"count": "alert_count", "sum": "tp"})
        g["fp"] = g["alert_count"] - g["tp"]
        g = g[(g["alert_count"] >= min_alerts) & (g["fp"] >= min_fp)].copy()
        if g.empty:
            continue
        g["precision"] = g["tp"] / g["alert_count"].clip(lower=1)
        g["fp_share"] = g["fp"] / max(total_fp, 1)
        g["diagnostic_score"] = g["fp"] * (base_precision - g["precision"]).clip(lower=0)
        for _, r in g.sort_values(["diagnostic_score", "fp"], ascending=[False, False]).head(20).iterrows():
            rows.append({
                "feature_a": a, "value_a": str(r[a]), "feature_b": b, "value_b": str(r[b]),
                "alert_count": int(r["alert_count"]), "tp": int(r["tp"]), "fp": int(r["fp"]),
                "precision": float(r["precision"]), "base_precision": base_precision,
                "precision_gap_vs_base": float(r["precision"] - base_precision),
                "fp_share": float(r["fp_share"]), "diagnostic_score": float(r["diagnostic_score"]),
                "description": f"alert AND {a}={r[a]} AND {b}={r[b]}",
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["diagnostic_score", "fp"], ascending=[False, False]).head(top_n).reset_index(drop=True)


def numeric_separation(df: pd.DataFrame, base_col: str, cols: list[str]) -> pd.DataFrame:
    y = df["is_fraud"].to_numpy(dtype=int)
    alerts = df[base_col].to_numpy(dtype=int) == 1
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df.loc[alerts, c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        mask = s.notna().to_numpy()
        if mask.sum() < 50:
            continue
        yy = y[alerts][mask]
        if len(np.unique(yy)) < 2:
            continue
        vals = s.to_numpy()[mask].astype(float)
        try:
            auc_fraud_high = float(roc_auc_score(yy, vals))
            ap_fraud_high = float(average_precision_score(yy, vals))
        except Exception:
            auc_fraud_high = np.nan
            ap_fraud_high = np.nan
        vals_tp = vals[yy == 1]
        vals_fp = vals[yy == 0]
        if len(vals_tp) == 0 or len(vals_fp) == 0:
            continue
        rows.append({
            "feature": c,
            "n_alerts_non_null": int(mask.sum()),
            "n_tp": int((yy == 1).sum()),
            "n_fp": int((yy == 0).sum()),
            "auc_fraud_high": auc_fraud_high,
            "auc_fp_high": 1.0 - auc_fraud_high if np.isfinite(auc_fraud_high) else np.nan,
            "average_precision_fraud_high": ap_fraud_high,
            "tp_median": float(np.median(vals_tp)),
            "fp_median": float(np.median(vals_fp)),
            "tp_p10": float(np.quantile(vals_tp, 0.10)),
            "tp_p90": float(np.quantile(vals_tp, 0.90)),
            "fp_p10": float(np.quantile(vals_fp, 0.10)),
            "fp_p90": float(np.quantile(vals_fp, 0.90)),
            "abs_auc_distance_from_random": abs(auc_fraud_high - 0.5) if np.isfinite(auc_fraud_high) else np.nan,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("abs_auc_distance_from_random", ascending=False).reset_index(drop=True)


def build_hypotheses(summary: dict[str, Any], seg: pd.DataFrame, pair: pd.DataFrame, num: pd.DataFrame) -> dict[str, Any]:
    top_single = seg.head(15).to_dict(orient="records") if not seg.empty else []
    top_pair = pair.head(15).to_dict(orient="records") if not pair.empty else []
    numeric_best = num.head(15).to_dict(orient="records") if not num.empty else []
    dominant_features = []
    if not seg.empty:
        dominant_features = (
            seg.groupby("feature")["diagnostic_score"].sum().sort_values(ascending=False).head(10).reset_index().to_dict(orient="records")
        )
    next_steps = [
        {
            "strategy": "commercial_trust_features",
            "why": "Se os principais hotspots forem combinacoes de recebedor, valor, chave e historico, precisamos medir confianca transacional e reputacao do recebedor; os scores atuais nao separam o suficiente.",
            "examples": [
                "idade real do relacionamento pagador-recebedor",
                "recorrencia benigno-confirmada por par pagador-recebedor",
                "reputacao do recebedor: diversidade de pagadores, estornos/contestacoes, idade cadastral",
                "device/session trust e mudancas recentes de dispositivo/canal",
            ],
        },
        {
            "strategy": "segment_specific_modeling",
            "why": "Se poucos segmentos concentram FP com precisao baixa, treinar/calibrar thresholds por familia de segmento pode reduzir FP sem afetar segmentos de alto risco.",
            "examples": ["modelo especifico para value_band", "calibracao por tipo de chave", "politicas por MBK disponivel vs ausente"],
        },
        {
            "strategy": "false_positive_label_enrichment",
            "why": "O alvo FPR<=1.5% provavelmente exige labels comerciais adicionais alem de is_fraud, distinguindo normal confirmado, cliente incomodado, alerta analisado, contestacao e perda evitada.",
            "examples": ["resultado da analise manual", "contestacao posterior", "bloqueio/reversao", "atrito percebido"],
        },
    ]
    return {
        "experiment": "EXP-014B-R3T",
        "diagnostic_type": "fp_root_cause_attribution",
        "base_summary": summary,
        "dominant_single_feature_families": dominant_features,
        "top_single_feature_hotspots": top_single,
        "top_pairwise_hotspots": top_pair,
        "most_separating_numeric_features_inside_alerts": numeric_best,
        "interpretation": [
            "Esta rodada nao promove politica; ela localiza onde os FPs se concentram dentro dos alertas atuais.",
            "Hotspots com alto FP e baixa precision indicam onde investigar novas features ou calibracoes segmentadas.",
            "Features numericas com AUC perto de 0.5 dentro dos alertas indicam baixa capacidade de separar TP de FP no segundo estagio.",
        ],
        "recommended_next_strategies": next_steps,
    }


def make_report(summary, contract, metrics_obj, seg, low, num, pair, hyp) -> str:
    lines = []
    lines.append("# EXP-014B-R3T - FP Root Cause Attribution Probe")
    lines.append("")
    lines.append("## Resultado executivo")
    lines.append(f"- Base analisada: `{summary['base_col']}`")
    lines.append(f"- Metricas base: `{metrics_obj}`")
    lines.append(f"- Alvo comercial: `FN<={summary['max_fn']}`, `recall>={summary['min_recall']}`, `FPR<={summary['target_fpr']}`")
    lines.append(f"- FP max alvo: `{summary['target_fp']}`")
    lines.append(f"- Gap atual ate alvo: `{summary['target_gap_fp']}` FP")
    lines.append(f"- Total de alertas: `{summary['base_alerts']}`")
    lines.append(f"- FPs nos alertas: `{summary['base_fp']}`")
    lines.append(f"- TPs nos alertas: `{summary['base_tp']}`")
    lines.append("")
    lines.append("## Top segmentos de maior concentracao FP/baixa precisao")
    if low.empty:
        lines.append("Nenhum segmento relevante encontrado.")
    else:
        cols = ["feature", "value", "alert_count", "tp", "fp", "precision", "fp_share", "diagnostic_score"]
        lines.append(low[[c for c in cols if c in low.columns]].head(30).to_markdown(index=False))
    lines.append("")
    lines.append("## Hotspots por pares de features")
    if pair.empty:
        lines.append("Nenhum par relevante encontrado.")
    else:
        cols = ["description", "alert_count", "tp", "fp", "precision", "fp_share", "diagnostic_score"]
        lines.append(pair[[c for c in cols if c in pair.columns]].head(30).to_markdown(index=False))
    lines.append("")
    lines.append("## Separabilidade numerica dentro dos alertas")
    if num.empty:
        lines.append("Nenhuma feature numerica avaliavel encontrada.")
    else:
        cols = ["feature", "auc_fraud_high", "average_precision_fraud_high", "tp_median", "fp_median", "tp_p10", "tp_p90", "fp_p10", "fp_p90"]
        lines.append(num[[c for c in cols if c in num.columns]].head(30).to_markdown(index=False))
    lines.append("")
    lines.append("## Hipoteses principais")
    for i, item in enumerate(hyp.get("dominant_single_feature_families", [])[:8], start=1):
        lines.append(f"{i}. `{item['feature']}` aparece como familia dominante de diagnostico (score agregado={item['diagnostic_score']:.3f}).")
    lines.append("")
    lines.append("## Decisao sugerida")
    lines.append("Nao promover politica nesta rodada. Usar os hotspots para escolher a proxima estrategia: novas features de confianca comercial, calibracao por segmento ou enriquecimento de labels de FP.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", default=None)
    ap.add_argument("--base-col", default=None)
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--target-fpr", type=float, default=TARGET_FPR)
    ap.add_argument("--max-fn", type=int, default=MAX_FN)
    ap.add_argument("--min-recall", type=float, default=MIN_RECALL)
    ap.add_argument("--min-alerts-segment", type=int, default=20)
    ap.add_argument("--top-n-per-feature", type=int, default=25)
    ap.add_argument("--pairwise-min-fp", type=int, default=20)
    ap.add_argument("--pairwise-min-alerts", type=int, default=25)
    ap.add_argument("--pairwise-top-n", type=int, default=200)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input) if args.input else pick_existing_path(DEFAULT_INPUT_CANDIDATES)
    if input_path is None or not input_path.exists():
        raise FileNotFoundError("Nenhum input encontrado. Informe --input apontando para predictions CSV.")

    log("=" * 80)
    log("EXP-014B-R3T - FP Root Cause Attribution Probe")
    log("=" * 80)
    log(f"Input: {input_path}")

    df = normalize(pd.read_csv(input_path, low_memory=False))
    df = add_bins(df)
    base_col = pick_base_col(df, args.base_col)
    y = df["is_fraud"].to_numpy(dtype=int)
    pred = df[base_col].to_numpy(dtype=int)
    m = metrics(y, pred)
    n_normals = int((y == 0).sum())
    target_fp = int(np.floor(args.target_fpr * n_normals))
    target_gap = max(0, m["fp"] - target_fp)
    alerts = pred == 1

    missing = []
    if "is_fraud" not in df.columns:
        missing.append("is_fraud")
    if base_col not in df.columns:
        missing.append(base_col)
    contract = {
        "n_rows": int(len(df)),
        "n_frauds": int(y.sum()),
        "n_normals": n_normals,
        "input_path": str(input_path),
        "base_col": base_col,
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

    base_obj = {
        "base_metrics": m,
        "target_fpr_ok": bool(m["fpr"] <= args.target_fpr),
        "fn_budget_ok": bool(m["fn"] <= args.max_fn),
        "recall_ok": bool(m["recall"] >= args.min_recall),
        "target_fp": target_fp,
        "target_gap_fp": target_gap,
    }
    dump_json(base_obj, out / "02_base_metrics.json")

    cat_cols = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
    seg = segment_table(df, base_col, cat_cols, args.min_alerts_segment, args.top_n_per_feature)
    seg.to_csv(out / "03_fp_pareto_by_feature.csv", index=False)

    # Low precision: below global precision, then sorted by diagnostic score.
    low = seg.copy()
    if not low.empty:
        low = low[low["precision"] < low["base_precision"]].sort_values(["diagnostic_score", "fp"], ascending=[False, False]).reset_index(drop=True)
    low.to_csv(out / "04_low_precision_segments.csv", index=False)

    num = numeric_separation(df, base_col, [c for c in NUMERIC_CANDIDATES if c in df.columns])
    num.to_csv(out / "05_numeric_separation.csv", index=False)

    pair_cols = [c for c in PAIRWISE_ANCHORS if c in df.columns]
    pair = pairwise_hotspots(df, base_col, pair_cols, args.pairwise_min_fp, args.pairwise_min_alerts, args.pairwise_top_n)
    pair.to_csv(out / "06_pairwise_fp_hotspots.csv", index=False)

    summary = {
        "experiment": "EXP-014B-R3T",
        "status": "DONE",
        "objective_status": "DONE_FP_ROOT_CAUSE_ATTRIBUTION_ONLY_NO_POLICY_PROMOTION",
        "n_rows": int(len(df)),
        "n_frauds": int(y.sum()),
        "n_normals": n_normals,
        "input_path": str(input_path),
        "base_col": base_col,
        "base_metrics": m,
        "base_alerts": int(alerts.sum()),
        "base_tp": int(((alerts) & (y == 1)).sum()),
        "base_fp": int(((alerts) & (y == 0)).sum()),
        "target_fpr": args.target_fpr,
        "target_fp": target_fp,
        "target_gap_fp": target_gap,
        "max_fn": args.max_fn,
        "min_recall": args.min_recall,
        "n_categorical_features_analyzed": int(len(cat_cols)),
        "n_numeric_features_analyzed": int(len(num)),
        "n_segment_rows": int(len(seg)),
        "n_low_precision_segment_rows": int(len(low)),
        "n_pairwise_hotspots": int(len(pair)),
        "all_pass": True,
        "output_dir": str(out),
    }

    hyp = build_hypotheses(summary, seg, pair, num)
    dump_json(hyp, out / "07_root_cause_hypotheses.json")
    dump_json(summary, out / "00_run_summary.json")
    (out / "08_exp014b_r3t_report.md").write_text(make_report(summary, contract, m, seg, low, num, pair, hyp), encoding="utf-8")

    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        out / "00_run_summary.json",
        out / "01_input_contract.json",
        out / "02_base_metrics.json",
        out / "03_fp_pareto_by_feature.csv",
        out / "04_low_precision_segments.csv",
        out / "05_numeric_separation.csv",
        out / "06_pairwise_fp_hotspots.csv",
        out / "07_root_cause_hypotheses.json",
        out / "08_exp014b_r3t_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
