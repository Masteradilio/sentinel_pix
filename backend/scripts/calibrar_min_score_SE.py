"""
calibrar_min_score_SE.py — Frente 2: Calibração dos min_score por Padrão

Objetivo:
  Para cada padrão do SE v3.0, varrer diferentes valores de min_score
  e encontrar o ponto ótimo no tradeoff Precision × Recall.

Metodologia:
  1. Carrega o CSV detalhado da validação retroativa (se_validacao_detalhado.csv)
  2. Para cada padrão, reconstrói o score interno de cada transação
  3. Varre min_score de 3 a 15 e calcula TP/FP/Precision/Recall/F1
  4. Encontra o min_score ótimo por critério configurável
  5. Simula impacto global com os novos min_scores

Critérios de otimização disponíveis:
  - "max_f1":      Maximiza F1 (equilíbrio precision-recall)
  - "max_f2":      Maximiza F2 (favorece recall sobre precision)
  - "min_fpr_10":  Minimiza FPR mantendo recall ≥ 10%
  - "prec_floor":  Maximiza recall com precision ≥ threshold

Outputs:
  - relatorio/se_calibracao_curvas.csv
  - relatorio/se_calibracao_recomendacao.json
  - relatorio/se_calibracao_simulacao.json
  - relatorio/se_calibracao_relatorio.html

Uso:
  python calibrar_min_score.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =========================================================
# PATHS
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

for candidate in [SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent]:
    if (candidate / "backend").exists() and (candidate / "dados").exists():
        PROJECT_ROOT = candidate
        break

BACKEND_DIR = PROJECT_ROOT / "backend"
DADOS_DIR = PROJECT_ROOT / "dados"
RELATORIO_DIR = PROJECT_ROOT / "relatorio"
CORE_DIR = BACKEND_DIR / "core"

RELATORIO_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from core.social_engineering import SocialEngineeringDetector

# =========================================================
# CONFIG
# =========================================================
INPUT_DATA = DADOS_DIR / "base_mvp_model_ready_optimized.csv"

# Outputs
OUT_CURVAS = RELATORIO_DIR / "se_calibracao_curvas.csv"
OUT_RECOMENDACAO = RELATORIO_DIR / "se_calibracao_recomendacao.json"
OUT_SIMULACAO = RELATORIO_DIR / "se_calibracao_simulacao.json"
OUT_HTML = RELATORIO_DIR / "se_calibracao_relatorio.html"

# Critério de otimização
# "max_f1", "max_f2", "min_fpr_10", "prec_floor"
OPTIMIZATION_CRITERIA = "max_f1"
PRECISION_FLOOR = 0.15  # Para critério "prec_floor"

# Range de min_score a testar
MIN_SCORE_RANGE = range(3, 16)  # 3 a 15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("se_calibracao")


# =========================================================
# HELPERS
# =========================================================
def safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        v = float(val)
        return default if np.isnan(v) or np.isinf(v) else v
    except (ValueError, TypeError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    return int(safe_float(val, float(default)))


def row_to_features(row: pd.Series) -> Dict[str, Any]:
    result = {}
    for col in row.index:
        val = row[col]
        if isinstance(val, (np.integer,)):
            result[col] = int(val)
        elif isinstance(val, (np.floating,)):
            result[col] = float(val) if not np.isnan(val) else None
        elif isinstance(val, (np.bool_,)):
            result[col] = bool(val)
        elif isinstance(val, pd.Timestamp):
            result[col] = val.to_pydatetime()
        elif pd.isna(val):
            result[col] = None
        else:
            result[col] = val
    return result


def fbeta_score(precision: float, recall: float, beta: float = 1.0) -> float:
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def find_optimal_min_score(
    curve_data: List[Dict],
    criteria: str = "max_f1",
    precision_floor: float = 0.15,
) -> Dict:
    """Encontra o min_score ótimo dado um critério."""

    if not curve_data:
        return {"min_score": 5, "reason": "sem dados"}

    if criteria == "max_f1":
        best = max(curve_data, key=lambda x: x["f1"])
        return {**best, "reason": f"max F1 = {best['f1']:.4f}"}

    elif criteria == "max_f2":
        best = max(curve_data, key=lambda x: x["f2"])
        return {**best, "reason": f"max F2 = {best['f2']:.4f}"}

    elif criteria == "min_fpr_10":
        candidates = [d for d in curve_data if d["recall"] >= 0.10]
        if not candidates:
            candidates = curve_data
        best = min(candidates, key=lambda x: x["fpr"])
        return {**best, "reason": f"min FPR={best['fpr']:.6f} com recall≥10%"}

    elif criteria == "prec_floor":
        candidates = [d for d in curve_data if d["precision"] >= precision_floor]
        if not candidates:
            # Relaxar: pegar o que tem maior precision
            best = max(curve_data, key=lambda x: x["precision"])
            return {
                **best,
                "reason": (
                    f"precision floor {precision_floor} não atingido, "
                    f"melhor precision = {best['precision']:.4f}"
                ),
            }
        best = max(candidates, key=lambda x: x["recall"])
        return {
            **best,
            "reason": (
                f"max recall = {best['recall']:.4f} com "
                f"precision ≥ {precision_floor}"
            ),
        }

    return {**curve_data[0], "reason": "fallback"}


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 70)
    print("FRENTE 2 — Calibração dos min_score por Padrão")
    print("=" * 70)
    print(f"  Critério de otimização: {OPTIMIZATION_CRITERIA}")
    print(f"  Range de min_score: {list(MIN_SCORE_RANGE)}")

    # ─── 1. Carregar dados ──────────────────────────────────
    print(f"\n[1/5] Carregando dados...")
    df = pd.read_csv(INPUT_DATA, low_memory=False)
    df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
    n_total = len(df)
    n_fraud = int(df["is_fraud"].sum())
    n_normal = n_total - n_fraud
    print(f"  {n_total:,} transações ({n_fraud} fraudes, {n_normal:,} normais)")

    # ─── 2. Rodar SE com score DESAGREGADO por padrão ───────
    print(f"\n[2/5] Calculando scores internos por padrão para cada transação...")

    detector = SocialEngineeringDetector()
    patterns_config = detector.PATTERNS

    # Coletar dados brutos: para cada transação, o score interno de cada padrão
    # (antes de aplicar min_score)
    raw_scores: Dict[str, List[Dict]] = {
        pattern_name: [] for pattern_name in patterns_config
    }

    t0 = time.perf_counter()
    log_interval = max(1, n_total // 10)

    for idx in range(n_total):
        row = df.iloc[idx]
        features = row_to_features(row)
        f = detector._adapt_features(features)

        # Avaliar indicadores
        active_indicators: Dict[str, bool] = {}
        for name, check in detector.INDICATORS.items():
            try:
                active_indicators[name] = check(f)
            except Exception:
                active_indicators[name] = False

        is_fraud = int(row.get("is_fraud", 0))

        # Para cada padrão, calcular score interno SEM min_score gate
        for pattern_name, config in patterns_config.items():
            score = 0
            matched: List[str] = []

            required_ok = True
            for ind in config["required"]:
                if active_indicators.get(ind, False):
                    score += 2
                    matched.append(ind)
                else:
                    required_ok = False
                    break

            if not required_ok:
                # Required não atendido — score 0 (padrão nunca ativa)
                raw_scores[pattern_name].append({
                    "idx": idx,
                    "is_fraud": is_fraud,
                    "internal_score": 0,
                    "required_met": False,
                    "n_matched": 0,
                })
                continue

            for ind in config["optional"]:
                if active_indicators.get(ind, False):
                    score += 1
                    matched.append(ind)

            raw_scores[pattern_name].append({
                "idx": idx,
                "is_fraud": is_fraud,
                "internal_score": score,
                "required_met": True,
                "n_matched": len(matched),
            })

        if (idx + 1) % log_interval == 0:
            elapsed = time.perf_counter() - t0
            rate = (idx + 1) / elapsed
            print(f"  {idx+1:>7,}/{n_total:,} ({(idx+1)/n_total*100:.0f}%) | {rate:.0f} tx/s")

    elapsed = time.perf_counter() - t0
    print(f"  ✓ Concluído em {elapsed:.1f}s")

    # ─── 3. Calcular curvas por padrão ──────────────────────
    print(f"\n[3/5] Calculando curvas Precision-Recall por padrão...")

    all_curves: List[Dict] = []
    recommendations: Dict[str, Dict] = {}

    for pattern_name, config in patterns_config.items():
        current_min_score = config["min_score"]
        severity = config["severity"]

        # Filtrar transações onde required foi atendido
        df_pattern = pd.DataFrame(raw_scores[pattern_name])
        df_req_met = df_pattern[df_pattern["required_met"]]

        # Total de fraudes e normais (universo completo)
        total_fraud = n_fraud
        total_normal = n_normal

        print(f"\n  ── {pattern_name} (atual min_score={current_min_score}) ──")
        print(f"  Required atendido em: {len(df_req_met):,} transações")

        if len(df_req_met) == 0:
            print(f"  ⚠ Nenhuma transação atende required — pulando")
            recommendations[pattern_name] = {
                "current_min_score": current_min_score,
                "recommended_min_score": current_min_score,
                "reason": "nenhuma transação atende required",
            }
            continue

        # Score distribution
        scores_fraud = df_req_met[df_req_met["is_fraud"] == 1]["internal_score"]
        scores_normal = df_req_met[df_req_met["is_fraud"] == 0]["internal_score"]

        print(f"  Fraudes com required: {len(scores_fraud)} | score range: {scores_fraud.min()}-{scores_fraud.max()}")
        print(f"  Normais com required: {len(scores_normal)} | score range: {scores_normal.min()}-{scores_normal.max()}")

        # Varrer min_score thresholds
        curve_data: List[Dict] = []

        print(f"\n  {'min_score':>10} {'TP':>5} {'FP':>6} {'Prec':>8} {'Recall':>8} {'F1':>8} {'F2':>8} {'FPR':>10}")
        print(f"  {'-'*70}")

        for ms in MIN_SCORE_RANGE:
            # Transações que ativariam este padrão com este min_score
            activated = df_req_met[df_req_met["internal_score"] >= ms]
            tp = int((activated["is_fraud"] == 1).sum())
            fp = int((activated["is_fraud"] == 0).sum())
            fn = total_fraud - tp
            tn = total_normal - fp

            precision = tp / max(tp + fp, 1)
            recall = tp / max(total_fraud, 1)
            f1 = fbeta_score(precision, recall, beta=1.0)
            f2 = fbeta_score(precision, recall, beta=2.0)
            fpr = fp / max(total_normal, 1)

            entry = {
                "pattern": pattern_name,
                "severity": severity,
                "min_score": ms,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "f2": round(f2, 6),
                "fpr": round(fpr, 8),
                "total_ativacoes": tp + fp,
                "is_current": ms == current_min_score,
            }
            curve_data.append(entry)
            all_curves.append(entry)

            marker = " ◄── atual" if ms == current_min_score else ""
            print(
                f"  {ms:>10} {tp:>5} {fp:>6} "
                f"{precision:>8.4f} {recall:>8.4f} "
                f"{f1:>8.4f} {f2:>8.4f} {fpr:>10.6f}{marker}"
            )

        # Encontrar ótimo
        optimal = find_optimal_min_score(
            curve_data, OPTIMIZATION_CRITERIA, PRECISION_FLOOR
        )

        recommendations[pattern_name] = {
            "current_min_score": current_min_score,
            "recommended_min_score": optimal["min_score"],
            "reason": optimal["reason"],
            "optimal_tp": optimal["tp"],
            "optimal_fp": optimal["fp"],
            "optimal_precision": optimal["precision"],
            "optimal_recall": optimal["recall"],
            "optimal_f1": optimal["f1"],
            "optimal_fpr": optimal["fpr"],
            "current_tp": next(
                (d["tp"] for d in curve_data if d["min_score"] == current_min_score),
                0,
            ),
            "current_fp": next(
                (d["fp"] for d in curve_data if d["min_score"] == current_min_score),
                0,
            ),
            "current_precision": next(
                (d["precision"] for d in curve_data if d["min_score"] == current_min_score),
                0,
            ),
            "current_f1": next(
                (d["f1"] for d in curve_data if d["min_score"] == current_min_score),
                0,
            ),
        }

        rec = recommendations[pattern_name]
        delta = "=" if rec["recommended_min_score"] == current_min_score else (
            "↑" if rec["recommended_min_score"] > current_min_score else "↓"
        )
        print(
            f"\n  ✅ Recomendação: min_score {current_min_score} → "
            f"{rec['recommended_min_score']} ({delta}) | {rec['reason']}"
        )

    # Salvar curvas
    df_curves = pd.DataFrame(all_curves)
    df_curves.to_csv(OUT_CURVAS, index=False)
    print(f"\n  Curvas salvas: {OUT_CURVAS}")

    # ─── 4. Simular impacto global ──────────────────────────
    print(f"\n[4/5] Simulando impacto global com min_scores recomendados...")

    # Recalcular SE completo com novos min_scores
    new_min_scores = {
        name: rec["recommended_min_score"]
        for name, rec in recommendations.items()
        if "recommended_min_score" in rec
    }

    # Patch detector com novos min_scores
    detector_new = SocialEngineeringDetector()
    for name, ms in new_min_scores.items():
        if name in detector_new.PATTERNS:
            detector_new.PATTERNS[name]["min_score"] = ms

    # Rodar no dataset completo
    results_new: List[Dict] = []
    t0 = time.perf_counter()

    for idx in range(n_total):
        row = df.iloc[idx]
        features = row_to_features(row)
        se_result = detector_new.detect_from_pipeline(features)
        results_new.append({
            "is_fraud": int(row.get("is_fraud", 0)),
            "se_score": se_result.se_score,
            "n_patterns": len(se_result.patterns),
            "patterns": "|".join(p.pattern_name for p in se_result.patterns),
        })

    elapsed = time.perf_counter() - t0
    print(f"  ✓ Simulação completa em {elapsed:.1f}s")

    df_new = pd.DataFrame(results_new)
    fraud_mask = df_new["is_fraud"] == 1
    normal_mask = df_new["is_fraud"] == 0

    se_active = df_new["se_score"] > 0
    new_tp = int((se_active & fraud_mask).sum())
    new_fp = int((se_active & normal_mask).sum())
    new_recall = new_tp / n_fraud
    new_precision = new_tp / max(new_tp + new_fp, 1)
    new_f1 = fbeta_score(new_precision, new_recall)
    new_fpr = new_fp / n_normal

    # Comparar com v3.0 atual (dos dados da Frente 1)
    simulacao = {
        "data_simulacao": datetime.now().isoformat(),
        "criterio_otimizacao": OPTIMIZATION_CRITERIA,
        "new_min_scores": new_min_scores,
        "global_metrics": {
            "v3_current": {
                "tp": 219,
                "fp": 957,
                "recall": 0.6169,
                "precision": 0.1862,
                "f1": 0.2861,
                "fpr": 0.00957,
            },
            "v3_calibrated": {
                "tp": new_tp,
                "fp": new_fp,
                "recall": round(new_recall, 4),
                "precision": round(new_precision, 4),
                "f1": round(new_f1, 4),
                "fpr": round(new_fpr, 6),
            },
        },
        "per_pattern": {},
    }

    # Métricas por padrão na simulação
    for pattern_name in patterns_config:
        pattern_mask = df_new["patterns"].str.contains(pattern_name, na=False)
        tp = int((pattern_mask & fraud_mask).sum())
        fp = int((pattern_mask & normal_mask).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(n_fraud, 1)
        simulacao["per_pattern"][pattern_name] = {
            "tp": tp,
            "fp": fp,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(fbeta_score(prec, rec), 4),
        }

    with open(OUT_SIMULACAO, "w", encoding="utf-8") as f:
        json.dump(simulacao, f, ensure_ascii=False, indent=2, default=str)

    # Salvar recomendações
    output_rec = {
        "data_geracao": datetime.now().isoformat(),
        "criterio": OPTIMIZATION_CRITERIA,
        "precision_floor": PRECISION_FLOOR if OPTIMIZATION_CRITERIA == "prec_floor" else None,
        "recommendations": recommendations,
    }
    with open(OUT_RECOMENDACAO, "w", encoding="utf-8") as f:
        json.dump(output_rec, f, ensure_ascii=False, indent=2, default=str)

    # ─── 5. Gerar relatório HTML ────────────────────────────
    print(f"\n[5/5] Gerando relatório HTML...")
    _generate_html(recommendations, simulacao, all_curves, patterns_config, n_total)

    # ─── Resumo Final ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("RESUMO — Frente 2: Calibração de min_score")
    print(f"{'='*70}")

    print(f"\n  Critério: {OPTIMIZATION_CRITERIA}")
    print(f"\n  {'Padrão':<30} {'Atual':>6} {'Novo':>6} {'Δ':>4} {'TP':>5} {'FP':>6} {'Prec':>8} {'F1':>8}")
    print(f"  {'-'*78}")

    for name, rec in recommendations.items():
        if "recommended_min_score" not in rec:
            continue
        curr = rec["current_min_score"]
        new = rec["recommended_min_score"]
        delta = "=" if new == curr else ("↑" if new > curr else "↓")
        print(
            f"  {name:<30} {curr:>6} {new:>6} {delta:>4} "
            f"{rec.get('optimal_tp', '?'):>5} {rec.get('optimal_fp', '?'):>6} "
            f"{rec.get('optimal_precision', 0):>8.4f} {rec.get('optimal_f1', 0):>8.4f}"
        )

    print(f"\n  Impacto global (v3.0 atual → v3.0 calibrado):")
    curr_g = simulacao["global_metrics"]["v3_current"]
    new_g = simulacao["global_metrics"]["v3_calibrated"]
    print(f"    TP:        {curr_g['tp']} → {new_g['tp']}")
    print(f"    FP:        {curr_g['fp']} → {new_g['fp']}")
    print(f"    Precision: {curr_g['precision']:.4f} → {new_g['precision']:.4f}")
    print(f"    Recall:    {curr_g['recall']:.4f} → {new_g['recall']:.4f}")
    print(f"    F1:        {curr_g['f1']:.4f} → {new_g['f1']:.4f}")
    print(f"    FPR:       {curr_g['fpr']:.6f} → {new_g['fpr']:.6f}")

    print(f"\n  Artefatos:")
    for p in [OUT_CURVAS, OUT_RECOMENDACAO, OUT_SIMULACAO, OUT_HTML]:
        if p.exists():
            print(f"    {p.name} ({p.stat().st_size/1024:.0f} KB)")

    print(f"\n{'='*70}")


# =========================================================
# HTML REPORT
# =========================================================
def _generate_html(
    recommendations: Dict,
    simulacao: Dict,
    all_curves: List[Dict],
    patterns_config: Dict,
    n_total: int,
):
    """Gera relatório HTML da calibração."""

    curr_g = simulacao["global_metrics"]["v3_current"]
    new_g = simulacao["global_metrics"]["v3_calibrated"]

    # Tabela de recomendações
    rec_rows = ""
    for name, rec in recommendations.items():
        if "recommended_min_score" not in rec:
            continue
        curr = rec["current_min_score"]
        new = rec["recommended_min_score"]
        changed = curr != new
        color = "#00d4aa" if changed else "#888"
        arrow = "→" if changed else "="

        rec_rows += f"""
        <tr style="{'background:rgba(0,212,170,0.05)' if changed else ''}">
            <td><strong>{name}</strong></td>
            <td>{curr}</td>
            <td style="color:{color};font-weight:bold">{new}</td>
            <td style="color:{color}">{arrow}</td>
            <td>{rec.get('optimal_tp', '-')}</td>
            <td>{rec.get('optimal_fp', '-')}</td>
            <td>{rec.get('optimal_precision', 0):.4f}</td>
            <td>{rec.get('optimal_recall', 0):.4f}</td>
            <td>{rec.get('optimal_f1', 0):.4f}</td>
            <td style="font-size:11px;color:#888">{rec.get('reason', '')}</td>
        </tr>"""

    # Tabela de curvas por padrão (compacta)
    curve_sections = ""
    for pattern_name in patterns_config:
        pattern_curves = [c for c in all_curves if c["pattern"] == pattern_name]
        if not pattern_curves:
            continue

        rows = ""
        rec = recommendations.get(pattern_name, {})
        rec_ms = rec.get("recommended_min_score", -1)

        for c in pattern_curves:
            is_current = c["is_current"]
            is_recommended = c["min_score"] == rec_ms
            style = ""
            marker = ""
            if is_recommended and is_current:
                style = "background:rgba(0,212,170,0.08)"
                marker = "◄ atual + recomendado"
            elif is_recommended:
                style = "background:rgba(0,212,170,0.1)"
                marker = "◄ RECOMENDADO"
            elif is_current:
                style = "background:rgba(255,159,67,0.08)"
                marker = "◄ atual"

            rows += f"""
            <tr style="{style}">
                <td>{c['min_score']}</td>
                <td class="highlight">{c['tp']}</td>
                <td class="warning">{c['fp']}</td>
                <td>{c['precision']:.4f}</td>
                <td>{c['recall']:.4f}</td>
                <td><strong>{c['f1']:.4f}</strong></td>
                <td>{c['f2']:.4f}</td>
                <td>{c['fpr']:.6f}</td>
                <td style="font-size:11px;color:#00d4aa">{marker}</td>
            </tr>"""

        curve_sections += f"""
        <div class="section">
            <h2>📈 {pattern_name}</h2>
            <table>
                <tr><th>min_score</th><th>TP</th><th>FP</th>
                    <th>Precision</th><th>Recall</th><th>F1</th>
                    <th>F2</th><th>FPR</th><th></th></tr>
                {rows}
            </table>
        </div>"""

    # Per-pattern na simulação
    sim_rows = ""
    for name, metrics in simulacao["per_pattern"].items():
        sim_rows += f"""
        <tr>
            <td><strong>{name}</strong></td>
            <td class="highlight">{metrics['tp']}</td>
            <td class="warning">{metrics['fp']}</td>
            <td>{metrics['precision']:.4f}</td>
            <td>{metrics['recall']:.4f}</td>
            <td>{metrics['f1']:.4f}</td>
        </tr>"""

    # Delta indicators
    tp_delta = new_g['tp'] - curr_g['tp']
    fp_delta = new_g['fp'] - curr_g['fp']
    tp_color = "#00d4aa" if tp_delta >= 0 else "#ff6b6b"
    fp_color = "#00d4aa" if fp_delta <= 0 else "#ff6b6b"

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Calibração min_score — Módulo SE v3.0</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #0e1117; color: #e0e0e0;
               padding: 30px 50px; line-height: 1.6; }}
        .header {{ text-align: center; margin-bottom: 40px; border-bottom: 3px solid #6c5ce7;
                   padding-bottom: 20px; }}
        .header h1 {{ color: #6c5ce7; font-size: 28px; }}
        .header .subtitle {{ color: #888; font-size: 14px; margin-top: 8px; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
                     margin-bottom: 30px; }}
        .kpi {{ background: #1a1d23; border-radius: 12px; padding: 20px; text-align: center;
                border: 2px solid #333; }}
        .kpi .value {{ font-size: 32px; font-weight: 800; }}
        .kpi .label {{ font-size: 11px; color: #888; margin-top: 6px; text-transform: uppercase; }}
        .kpi .detail {{ font-size: 11px; color: #666; margin-top: 4px; }}
        .section {{ background: #1a1d23; border-radius: 12px; padding: 24px; margin-bottom: 20px;
                    border: 1px solid #2a2d33; }}
        .section h2 {{ color: #6c5ce7; margin-bottom: 16px; font-size: 18px;
                       border-bottom: 1px solid #333; padding-bottom: 8px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
        th {{ color: #6c5ce7; font-size: 11px; text-transform: uppercase; padding: 10px 12px;
              text-align: left; border-bottom: 2px solid #333; }}
        td {{ padding: 8px 12px; border-bottom: 1px solid #2a2d33; }}
        tr:hover {{ background: rgba(108,92,231,0.05); }}
        .highlight {{ color: #00d4aa; font-weight: bold; }}
        .warning {{ color: #ffd93d; }}
        .callout {{ background: rgba(108,92,231,0.08); border-left: 4px solid #6c5ce7;
                    padding: 14px 18px; margin: 14px 0; border-radius: 0 8px 8px 0; }}
        .callout.success {{ border-left-color: #00d4aa; background: rgba(0,212,170,0.08); }}
        .footer {{ text-align: center; color: #555; font-size: 11px; margin-top: 40px;
                   padding-top: 16px; border-top: 1px solid #333; }}
    </style>
</head>
<body>

<div class="header">
    <h1>🎯 Calibração de min_score — SE v3.0</h1>
    <p class="subtitle">
        Frente 2 | Critério: {OPTIMIZATION_CRITERIA} |
        Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </p>
</div>

<!-- KPIs: Impacto Global -->
<div class="kpi-grid">
    <div class="kpi" style="border-color:{tp_color}">
        <div class="value" style="color:{tp_color}">{new_g['tp']}</div>
        <div class="label">TP (novo)</div>
        <div class="detail">era {curr_g['tp']} ({tp_delta:+d})</div>
    </div>
    <div class="kpi" style="border-color:{fp_color}">
        <div class="value" style="color:{fp_color}">{new_g['fp']}</div>
        <div class="label">FP (novo)</div>
        <div class="detail">era {curr_g['fp']} ({fp_delta:+d})</div>
    </div>
    <div class="kpi" style="border-color:#6c5ce7">
        <div class="value" style="color:#6c5ce7">{new_g['precision']:.2%}</div>
        <div class="label">Precision</div>
        <div class="detail">era {curr_g['precision']:.2%}</div>
    </div>
    <div class="kpi" style="border-color:#ff9f43">
        <div class="value" style="color:#ff9f43">{new_g['recall']:.2%}</div>
        <div class="label">Recall</div>
        <div class="detail">era {curr_g['recall']:.2%}</div>
    </div>
</div>

<!-- Recomendações -->
<div class="section">
    <h2>📋 Recomendações de min_score</h2>
    <table>
        <tr><th>Padrão</th><th>Atual</th><th>Novo</th><th>Δ</th>
            <th>TP</th><th>FP</th><th>Prec</th><th>Recall</th>
            <th>F1</th><th>Razão</th></tr>
        {rec_rows}
    </table>
</div>

<!-- Simulação por padrão -->
<div class="section">
    <h2>🔄 Simulação com novos min_scores</h2>
    <table>
        <tr><th>Padrão</th><th>TP</th><th>FP</th>
            <th>Precision</th><th>Recall</th><th>F1</th></tr>
        {sim_rows}
    </table>
</div>

<!-- Curvas por padrão -->
{curve_sections}

<div class="section">
    <h2>📋 Próximos Passos</h2>
    <div class="callout success">
        <strong>Frente 2 concluída.</strong><br>
        Aplicar os min_scores recomendados no social_engineering.py v3.0 e
        re-rodar a validação retroativa para confirmar.<br><br>
        Próximas frentes:<br>
        • <strong>Frente 3:</strong> Expandir padrões com novos indicadores de alta discriminação<br>
        • <strong>Frente 4:</strong> Documentar cada regra com evidência + referência regulatória
    </div>
</div>

<div class="footer">
    Frente 2 — Calibração min_score | SE v3.0 |
    {n_total:,} transações | Critério: {OPTIMIZATION_CRITERIA}
</div>

</body>
</html>"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Relatório HTML: {OUT_HTML}")


# =========================================================
# ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    main()
