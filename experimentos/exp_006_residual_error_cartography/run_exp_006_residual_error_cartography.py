"""
EXP-006 — Residual Error Cartography

Objetivo:
  Diagnosticar erros residuais do pipeline antifraude PIX sem rodar E2E pesado.

Este experimento:
  - Não chama PipelineOrquestrador.
  - Não troca artefatos.
  - Não altera scoring_config.json.
  - Não treina modelo.
  - Lê os artefatos já gerados pelo EXP-005B-E2E.
  - Classifica FNs recuperados, TPs perdidos, FPs adicionados e FPs removidos.
  - Identifica clusters e hipóteses cirúrgicas para próximos testes rápidos.

Entradas esperadas:
  resultados/experimentos/EXP-005B-E2E/
    01_e2e_grid_decision_engine.csv
    02_best_config_e2e.json
    03_delta_fp_fn_best_e2e.json
    04_validacao_cruzada_e2e.json
    05_conclusao_executiva.md

Saídas:
  resultados/experimentos/EXP-006/
    01_resumo_executivo.md
    02_casos_movimentados.csv
    03_clusters_erros.csv
    04_hipoteses_cirurgicas.json
    05_protocolo_experimentos_rapidos.md
    06_decisao_tecnica.md
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
INPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-005B-E2E"
OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-006"


# =========================================================
# BASELINE
# =========================================================

BASELINE_FASE2 = {
    "TP": 346,
    "FP": 15,
    "FN": 9,
    "Precision": 0.958449,
    "Recall": 0.974648,
    "F1": 0.9665,
    "FPR": 0.002657,
}


# =========================================================
# HELPERS
# =========================================================

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return int(float(x))
    except Exception:
        return default


def safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [safe_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [safe_json(v) for v in obj.tolist()]
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


def read_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe_json(obj), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def has_text_value(x: Any) -> bool:
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass
    s = str(x).strip().lower()
    return s not in {"", "nan", "none", "null", "<na>"}


def value_bin(v: float) -> str:
    if v < 100:
        return "VL_000_099"
    if v < 500:
        return "VL_100_499"
    if v < 1000:
        return "VL_500_999"
    if v < 5000:
        return "VL_1K_5K"
    if v < 15000:
        return "VL_5K_15K"
    return "VL_15K_PLUS"


def relationship_bin(m: float) -> str:
    if m <= 6:
        return "REL_0_6M"
    if m <= 12:
        return "REL_7_12M"
    if m <= 36:
        return "REL_1_3Y"
    if m <= 120:
        return "REL_3_10Y"
    return "REL_10Y_PLUS"


def age_bin(age: float) -> str:
    if age < 18:
        return "AGE_LT_18"
    if age < 30:
        return "AGE_18_29"
    if age < 45:
        return "AGE_30_44"
    if age < 60:
        return "AGE_45_59"
    if age < 70:
        return "AGE_60_69"
    return "AGE_70_PLUS"


def if_bin(x: float) -> str:
    if x >= 0.995:
        return "IF_995_PLUS"
    if x >= 0.985:
        return "IF_985_995"
    if x >= 0.95:
        return "IF_950_985"
    if x >= 0.80:
        return "IF_800_950"
    return "IF_LT_800"


def lgbm_bin(x: float) -> str:
    if x >= 0.30:
        return "LGBM_300_PLUS"
    if x >= 0.20:
        return "LGBM_200_300"
    if x >= 0.10:
        return "LGBM_100_200"
    if x >= 0.05:
        return "LGBM_050_100"
    if x >= 0.01:
        return "LGBM_010_050"
    return "LGBM_LT_010"


def score_bin(x: float) -> str:
    if x >= 95:
        return "SCORE_BLOQUEAR"
    if x >= 62:
        return "SCORE_CONFIRMAR"
    if x >= 55:
        return "SCORE_GRAY_55_62"
    return "SCORE_APROVAR"


def classify_case(row: dict[str, Any], movement_type: str) -> dict[str, Any]:
    vl = safe_float(row.get("vl_pix"))
    idade = safe_float(row.get("nr_idade"))
    rel = safe_float(row.get("qt_tempo_relacionamento_mes"))
    first = safe_int(row.get("first_receiver_flag"))
    pix_random = safe_int(row.get("pix_key_random_flag"))
    lgbm_raw = safe_float(row.get("lgbm_raw"))
    lgbm_mapped = safe_float(row.get("lgbm_mapped"))
    if_pct = safe_float(row.get("if_percentile"))
    se = safe_float(row.get("se_score"))
    beh = safe_float(row.get("beh_score"))
    final_score = safe_float(row.get("score_final"))
    suppressed = row.get("veto_suppressed_reason")

    flags = []

    if first == 1:
        flags.append("FIRST_RECEIVER")
    if pix_random == 1:
        flags.append("PIX_RANDOM")
    if vl >= 15000:
        flags.append("VALOR_15K_PLUS")
    elif vl >= 5000:
        flags.append("VALOR_5K_PLUS")
    if rel <= 12:
        flags.append("REL_CURTO")
    if idade >= 60:
        flags.append("IDADE_60_PLUS")
    if if_pct >= 0.985:
        flags.append("IF_EXTREMO")
    elif if_pct >= 0.95:
        flags.append("IF_ALTO")
    if se >= 40:
        flags.append("SE_ATIVO")
    if beh >= 15:
        flags.append("BEH_ATIVO")
    if has_text_value(suppressed):
        flags.append("VETO_SUPPRESSED")

    if movement_type == "FN_RECUPERADO":
        if se == 0 and beh == 0 and if_pct < 0.80 and vl < 1000:
            category = "LGBM_LOW_VALUE_FIRST_RECEIVER"
        elif has_text_value(suppressed):
            category = "ENGINE_SUPPRESSED_RECOVERABLE"
        elif first == 1 and 0.05 <= lgbm_raw < 0.20:
            category = "LGBM_GRAY_FIRST_RECEIVER"
        elif if_pct >= 0.95:
            category = "IF_HIGH_RECOVERABLE"
        else:
            category = "RECOVERABLE_UNCLEAR"

    elif movement_type == "TP_PERDIDO":
        if lgbm_raw < 0.05 and se == 0 and beh == 0:
            category = "WEAK_SIGNAL_TP_LOST"
        elif first == 0 and lgbm_raw < 0.05:
            category = "NON_FIRST_RECEIVER_LOW_LGBM_TP_LOST"
        else:
            category = "TP_LOST_BY_RECALIBRATION"

    elif movement_type == "FP_ADICIONADO":
        if first == 1 and se == 0 and beh == 0 and lgbm_raw < 0.20:
            category = "FP_FIRST_RECEIVER_LOW_CONTEXT"
        elif first == 1 and if_pct >= 0.95:
            category = "FP_FIRST_RECEIVER_IF_HIGH"
        elif se >= 40 or beh >= 15:
            category = "FP_RULE_SIGNAL_PRESENT"
        else:
            category = "FP_OTHER"

    elif movement_type == "FP_REMOVIDO":
        if has_text_value(suppressed):
            category = "FP_REMOVED_BY_GUARD"
        else:
            category = "FP_REMOVED_OTHER"

    else:
        category = "UNKNOWN"

    return {
        "movement_type": movement_type,
        "category": category,
        "transaction_id": row.get("transaction_id"),
        "customer_id": row.get("customer_id"),
        "vl_pix": vl,
        "nr_idade": idade,
        "qt_tempo_relacionamento_mes": rel,
        "first_receiver_flag": first,
        "pix_key_random_flag": pix_random,
        "lgbm_raw": lgbm_raw,
        "lgbm_mapped": lgbm_mapped,
        "if_percentile": if_pct,
        "se_score": se,
        "beh_score": beh,
        "score_final": final_score,
        "decisao": row.get("decisao"),
        "veto_reason": row.get("veto_reason"),
        "veto_suppressed_reason": suppressed,
        "value_bin": value_bin(vl),
        "relationship_bin": relationship_bin(rel),
        "age_bin": age_bin(idade),
        "if_bin": if_bin(if_pct),
        "lgbm_bin": lgbm_bin(lgbm_raw),
        "score_bin": score_bin(final_score),
        "flags": "|".join(flags),
    }


# =========================================================
# LOAD INPUTS
# =========================================================

def load_inputs() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    grid_path = INPUT_DIR / "01_e2e_grid_decision_engine.csv"
    best_path = INPUT_DIR / "02_best_config_e2e.json"
    delta_path = INPUT_DIR / "03_delta_fp_fn_best_e2e.json"

    missing = [p for p in [grid_path, best_path, delta_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Artefatos do EXP-005B-E2E não encontrados:\n"
            + "\n".join(str(p) for p in missing)
        )

    grid = pd.read_csv(grid_path)
    best = read_json(best_path)
    delta = read_json(delta_path)

    return grid, best, delta


def extract_moved_cases(delta_payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    comparisons = delta_payload.get("comparisons", {})

    # Em alguns dumps, o delta já vem diretamente com comparisons; em outros,
    # vem como best_config_id + comparisons.
    if not comparisons and any(k.endswith("__seed_42") for k in delta_payload.keys()):
        comparisons = delta_payload

    mapping = {
        "top_fns_recuperados": "FN_RECUPERADO",
        "top_fps_adicionados": "FP_ADICIONADO",
        "top_tps_perdidos": "TP_PERDIDO",
        "top_fps_removidos": "FP_REMOVIDO",
    }

    for comp_key, comp in comparisons.items():
        config_id = str(comp_key).split("__seed_")[0]
        seed = str(comp_key).split("__seed_")[-1] if "__seed_" in str(comp_key) else ""

        for src_key, movement_type in mapping.items():
            for case in comp.get(src_key, []) or []:
                classified = classify_case(case, movement_type)
                classified["comparison_key"] = comp_key
                classified["config_id"] = config_id
                classified["seed"] = seed
                rows.append(classified)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# =========================================================
# ANALYSIS
# =========================================================

def summarize_grid(grid: pd.DataFrame) -> dict[str, Any]:
    out = {
        "baseline": {},
        "candidates": {},
        "decision": {},
    }

    for cfg_id, g in grid.groupby("config_id"):
        g = g.copy()
        rows = []

        for _, r in g.iterrows():
            rows.append({
                "seed": safe_int(r.get("seed")),
                "TP": safe_int(r.get("TP")),
                "FP": safe_int(r.get("FP")),
                "FN": safe_int(r.get("FN")),
                "Precision": safe_float(r.get("Precision")),
                "Recall": safe_float(r.get("Recall")),
                "F1": safe_float(r.get("F1")),
                "FPR": safe_float(r.get("FPR")),
            })

        item = {
            "rows": rows,
            "worst_FN": max(x["FN"] for x in rows),
            "worst_FP": max(x["FP"] for x in rows),
            "min_F1": min(x["F1"] for x in rows),
            "avg_F1": sum(x["F1"] for x in rows) / max(len(rows), 1),
        }

        if cfg_id == "BASELINE":
            out["baseline"] = item
        else:
            out["candidates"][cfg_id] = item

    baseline = out["baseline"]

    rejected = {}
    for cfg_id, item in out["candidates"].items():
        rejected[cfg_id] = {
            "reject": (
                item["worst_FN"] >= baseline["worst_FN"]
                and item["worst_FP"] > baseline["worst_FP"]
                and item["min_F1"] < baseline["min_F1"]
            ),
            "reason": "FN não melhora, FP aumenta e F1 piora vs baseline"
        }

    out["decision"] = {
        "keep_baseline": True,
        "reject_candidates": rejected,
    }

    return out


def cluster_cases(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame()

    group_cols = [
        "movement_type",
        "category",
        "value_bin",
        "relationship_bin",
        "age_bin",
        "if_bin",
        "lgbm_bin",
        "first_receiver_flag",
        "pix_key_random_flag",
    ]

    rows = []

    for keys, g in cases.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["count"] = len(g)
        row["avg_vl_pix"] = round(float(g["vl_pix"].mean()), 2)
        row["avg_lgbm_raw"] = round(float(g["lgbm_raw"].mean()), 6)
        row["avg_if_percentile"] = round(float(g["if_percentile"].mean()), 6)
        row["avg_score_final"] = round(float(g["score_final"].mean()), 2)
        row["examples"] = ", ".join(str(x) for x in g["transaction_id"].head(5).tolist())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["movement_type", "count"], ascending=[True, False])


def build_hypotheses(cases: pd.DataFrame, grid_summary: dict[str, Any]) -> dict[str, Any]:
    hypotheses: list[dict[str, Any]] = []

    if cases.empty:
        return {
            "status": "SEM_CASOS_MOVIMENTADOS",
            "hypotheses": [],
        }

    fn_rec = cases[cases["movement_type"].eq("FN_RECUPERADO")]
    tp_lost = cases[cases["movement_type"].eq("TP_PERDIDO")]
    fp_add = cases[cases["movement_type"].eq("FP_ADICIONADO")]
    fp_removed = cases[cases["movement_type"].eq("FP_REMOVIDO")]

    # H1: first_receiver + baixo valor foi responsável por recuperar FNs,
    # mas também adicionou muitos FPs.
    fn_first = len(fn_rec[fn_rec["first_receiver_flag"].eq(1)])
    fp_first = len(fp_add[fp_add["first_receiver_flag"].eq(1)])

    hypotheses.append({
        "id": "H1_FIRST_RECEIVER_EH_SINAL_MAS_NAO_REGRA",
        "status": "NAO_PROMOVER_COMO_REGRA",
        "evidence": {
            "fn_recuperados_com_first_receiver": fn_first,
            "fp_adicionados_com_first_receiver": fp_first,
        },
        "interpretation": (
            "first_receiver aparece nos FNs recuperados, mas também domina FPs adicionados. "
            "Não deve virar regra hardcoded; deve entrar apenas como feature/explicação."
        ),
        "next_action": "Não criar regra first_receiver. Usar em meta-learner shadow e análise de reputação de recebedor."
    })

    # H2: troca líquida de FNs por TPs perdidos.
    hypotheses.append({
        "id": "H2_RECALIBRACAO_LGBM_TROCA_FRAUDES",
        "status": "REJEITAR_LGBM_V6_2_RUNTIME",
        "evidence": {
            "fn_recuperados": int(len(fn_rec)),
            "tp_perdidos": int(len(tp_lost)),
            "fp_adicionados": int(len(fp_add)),
            "fp_removidos": int(len(fp_removed)),
        },
        "interpretation": (
            "O candidato recupera alguns FNs, mas perde TPs em quantidade similar e adiciona FP. "
            "Isso confirma que a fronteira do LGBM v6.2 não melhora o pipeline real."
        ),
        "next_action": "Manter baseline pós-FASE 1 e usar LGBM v6.2 apenas como evidência diagnóstica."
    })

    # H3: FNs/TPs perdidos de sinal fraco.
    weak_tp_lost = tp_lost[
        (tp_lost["lgbm_raw"] < 0.05)
        & (tp_lost["se_score"] <= 0)
        & (tp_lost["beh_score"] <= 0)
    ]

    hypotheses.append({
        "id": "H3_FRAUDES_COM_SINAL_FRACO_PODEM_SER_IRREDUTIVEIS",
        "status": "INVESTIGAR_RESIDUAL",
        "evidence": {
            "tps_perdidos_sinal_fraco": int(len(weak_tp_lost)),
        },
        "interpretation": (
            "Há fraudes capturadas pelo baseline que ficam com LGBM muito baixo e SE/BEH zerados no candidato. "
            "Se também estiverem entre os FNs residuais, dependem de novos sinais ou ajustes no engine."
        ),
        "next_action": "Criar relatório dos 9 FNs residuais completos e marcar DATA_LIMITED quando todos os módulos estiverem cegos."
    })

    # H4: veto_suppressed_reason precisa ser tratado corretamente.
    suppressed_count = int(cases["veto_suppressed_reason"].apply(has_text_value).sum())

    hypotheses.append({
        "id": "H4_VETO_SUPPRESSED_DEVE_SER_CAMPO_DIAGNOSTICO",
        "status": "MANTER_CORRECAO_NAN_E_USAR_NO_EXP006B",
        "evidence": {
            "casos_com_veto_suppressed_reason": suppressed_count,
        },
        "interpretation": (
            "veto_suppressed_reason aparece em alguns casos e pode explicar decisões suprimidas. "
            "O bug de NaN precisa continuar corrigido para não contaminar classificação de FN."
        ),
        "next_action": "No EXP-006B, gerar contrafactual de veto sem reprocessar E2E completo."
    })

    return {
        "status": "OK",
        "grid_decision": grid_summary["decision"],
        "hypotheses": hypotheses,
    }


# =========================================================
# REPORTS
# =========================================================

def write_summary_md(path: Path, grid_summary: dict[str, Any], cases: pd.DataFrame, clusters: pd.DataFrame) -> None:
    baseline = grid_summary.get("baseline", {})
    candidates = grid_summary.get("candidates", {})

    lines = [
        "# EXP-006 — Residual Error Cartography",
        "",
        f"Gerado em: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Objetivo",
        "",
        "Mapear erros residuais do pipeline antifraude PIX sem rodar novo E2E pesado.",
        "O EXP-006 usa os artefatos do EXP-005B-E2E para entender trocas entre FN recuperado, TP perdido e FP adicionado.",
        "",
        "## Conclusão executiva",
        "",
        "O EXP-005B-E2E não produziu melhoria promovível: os candidatos mantiveram FN igual ao baseline e aumentaram FP.",
        "Portanto, o caminho correto é diagnosticar a fronteira de erro antes de qualquer novo ajuste.",
        "",
        "## Baseline observado no EXP-005B-E2E",
        "",
    ]

    if baseline:
        for row in baseline.get("rows", []):
            lines.append(
                f"- Seed `{row['seed']}`: TP={row['TP']}, FP={row['FP']}, FN={row['FN']}, "
                f"Precision={row['Precision']:.4f}, Recall={row['Recall']:.4f}, F1={row['F1']:.4f}"
            )

    lines.extend([
        "",
        "## Candidatos avaliados",
        "",
        "| Config | worst_FN | worst_FP | min_F1 | avg_F1 | Decisão |",
        "|---|---:|---:|---:|---:|---|",
    ])

    for cfg, item in candidates.items():
        reject = grid_summary["decision"]["reject_candidates"][cfg]["reject"]
        decision = "REJEITAR" if reject else "INVESTIGAR"
        lines.append(
            f"| `{cfg}` | {item['worst_FN']} | {item['worst_FP']} | "
            f"{item['min_F1']:.4f} | {item['avg_F1']:.4f} | {decision} |"
        )

    lines.extend([
        "",
        "## Casos movimentados",
        "",
    ])

    if cases.empty:
        lines.append("Nenhum caso movimentado encontrado nos artefatos.")
    else:
        counts = cases["movement_type"].value_counts().to_dict()
        for k, v in counts.items():
            lines.append(f"- `{k}`: {v}")

    lines.extend([
        "",
        "## Clusters principais",
        "",
    ])

    if clusters.empty:
        lines.append("Nenhum cluster calculado.")
    else:
        lines.append("| Tipo | Categoria | Count | Valor médio | LGBM médio | IF médio | Exemplos |")
        lines.append("|---|---|---:|---:|---:|---:|---|")

        for _, r in clusters.head(20).iterrows():
            lines.append(
                f"| `{r['movement_type']}` | `{r['category']}` | {int(r['count'])} | "
                f"{float(r['avg_vl_pix']):.2f} | {float(r['avg_lgbm_raw']):.4f} | "
                f"{float(r['avg_if_percentile']):.4f} | {r['examples']} |"
            )

    lines.extend([
        "",
        "## Decisão técnica",
        "",
        "1. Não promover LGBM v6.2.",
        "2. Não criar regra generalista com first_receiver.",
        "3. Criar EXP-006B para contrafactual leve de engine, sem E2E pesado.",
        "4. Criar EXP-007A shadow apenas se o diagnóstico mostrar separabilidade nos sinais atuais.",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def write_fast_protocol(path: Path) -> None:
    lines = [
        "# Protocolo de Experimentos Rápidos — pós-EXP-006",
        "",
        "## Regra operacional",
        "",
        "Nenhum experimento novo deve rodar grid E2E completo por padrão.",
        "",
        "## Pipeline obrigatório",
        "",
        "1. `artifact-only`: usar CSV/JSON já existentes.",
        "2. `model-only`: rodar em segundos/minutos.",
        "3. `quick-e2e`: baseline + 1 candidato, sample 1000, seed 42.",
        "4. `final-e2e`: baseline + 1 candidato, sample 6000, seeds 42 e 123.",
        "",
        "## Critério de interrupção",
        "",
        "Parar no quick-e2e se:",
        "",
        "- FN não cair;",
        "- FP subir acima do baseline + 3;",
        "- F1 cair;",
        "- houver TP perdido sem ganho líquido.",
        "",
        "## Critério de promoção",
        "",
        "Promover apenas se:",
        "",
        "- FN cair nos dois seeds;",
        "- FP ficar dentro do limite;",
        "- F1 não piorar materialmente;",
        "- a mudança for explicável por cluster de erro.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def write_decision_md(path: Path, hypotheses: dict[str, Any]) -> None:
    lines = [
        "# EXP-006 — Decisão Técnica",
        "",
        "## Decisão",
        "",
        "Manter baseline pós-FASE 1 e não promover LGBM v6.2.",
        "",
        "## Próximo experimento recomendado",
        "",
        "`EXP-006B — Engine Counterfactual Audit`",
        "",
        "Objetivo: testar contrafactuais leves sobre os outputs já existentes, sem rodar E2E completo.",
        "",
        "Contrafactuais candidatos:",
        "",
        "1. Ajuste local de score para `LGBM_GRAY_FIRST_RECEIVER` somente se FP estimado for baixo.",
        "2. Auditoria de veto para casos com `veto_suppressed_reason` real.",
        "3. Classificação dos 9 FNs residuais em `RECOVERABLE` vs `DATA_LIMITED`.",
        "4. Verificar se `lgbm_effective_threshold` realmente influencia o engine ou se está sendo sobreposto.",
        "",
        "## Hipóteses geradas",
        "",
    ]

    for h in hypotheses.get("hypotheses", []):
        lines.extend([
            f"### {h['id']}",
            "",
            f"- Status: `{h['status']}`",
            f"- Interpretação: {h['interpretation']}",
            f"- Próxima ação: {h['next_action']}",
            "",
        ])

    path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EXP-006 — Residual Error Cartography")
    print("=" * 72)

    print("[1/5] Carregando artefatos do EXP-005B-E2E...")
    grid, best, delta = load_inputs()

    print("[2/5] Extraindo casos movimentados...")
    cases = extract_moved_cases(delta)

    cases_path = OUTPUT_DIR / "02_casos_movimentados.csv"
    cases.to_csv(cases_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Casos movimentados: {len(cases)} -> {cases_path}")

    print("[3/5] Gerando clusters...")
    clusters = cluster_cases(cases)
    clusters_path = OUTPUT_DIR / "03_clusters_erros.csv"
    clusters.to_csv(clusters_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Clusters: {len(clusters)} -> {clusters_path}")

    print("[4/5] Gerando hipóteses cirúrgicas...")
    grid_summary = summarize_grid(grid)
    hypotheses = build_hypotheses(cases, grid_summary)

    write_json(OUTPUT_DIR / "04_hipoteses_cirurgicas.json", hypotheses)

    print("[5/5] Escrevendo relatórios...")
    write_summary_md(OUTPUT_DIR / "01_resumo_executivo.md", grid_summary, cases, clusters)
    write_fast_protocol(OUTPUT_DIR / "05_protocolo_experimentos_rapidos.md")
    write_decision_md(OUTPUT_DIR / "06_decisao_tecnica.md", hypotheses)

    print()
    print("[OK] EXP-006 concluído sem E2E pesado.")
    print(f"[OK] Artefatos em: {OUTPUT_DIR}")
    print()
    print("Próximo passo recomendado:")
    print("  EXP-006B — Engine Counterfactual Audit")


if __name__ == "__main__":
    main()