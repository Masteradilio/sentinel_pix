"""
EXP-006D — Baseline Residual FN Census

Objetivo:
  Fazer o censo completo dos FNs e FPs residuais do baseline pós-FASE 1,
  sem rodar novo E2E.

Este experimento:
  - Não chama PipelineOrquestrador.
  - Não treina modelo.
  - Não troca artefatos.
  - Não altera scoring_config.json.
  - Usa as predições baseline já salvas pelo EXP-006C-R2.
  - Classifica cada FN residual por sinais disponíveis.
  - Classifica FPs atuais para buscar redução de FP sem aumentar FN.
  - Gera recomendação para o próximo experimento rápido.

Entradas esperadas:
  resultados/experimentos/EXP-006C-R2/
    baseline_predictions_seed_42.csv
    baseline_predictions_seed_123.csv

Saídas:
  resultados/experimentos/EXP-006D-FN-CENSUS/
    00_input_summary.json
    01_fn_residual_census.csv
    02_fp_census.csv
    03_fn_clusters.csv
    04_fp_clusters.csv
    05_recoverability_assessment.json
    06_recomendacoes.md
    07_next_experiment_spec.md
"""

from __future__ import annotations

import json
import math
from collections import Counter
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

INPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-006C-R2"
OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-006D-FN-CENSUS"


# =========================================================
# CONSTANTES
# =========================================================

BASELINE_REF = {
    "seed_42": {
        "TP": 346,
        "FP": 14,
        "FN": 9,
        "Precision": 0.961111,
        "Recall": 0.974648,
        "F1": 0.967832,
    },
    "seed_123": {
        "TP": 346,
        "FP": 12,
        "FN": 9,
        "Precision": 0.96648,
        "Recall": 0.974648,
        "F1": 0.970547,
    },
}

DECISOES_POSITIVAS = {"CONFIRMAR", "BLOQUEAR"}


# =========================================================
# HELPERS
# =========================================================

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        if pd.isna(x):
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
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe_json(obj), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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


def score_bin(x: float) -> str:
    if x >= 95:
        return "SCORE_BLOQUEAR"
    if x >= 62:
        return "SCORE_CONFIRMAR"
    if x >= 55:
        return "SCORE_GRAY_55_62"
    if x >= 45:
        return "SCORE_GRAY_45_55"
    return "SCORE_APROVAR_LOW"


def boolish_flag(row: pd.Series, col: str) -> int:
    if col not in row.index:
        return 0
    return safe_int(row.get(col), 0)


def get_col(row: pd.Series, col: str, default: Any = None) -> Any:
    if col not in row.index:
        return default
    return row.get(col, default)


# =========================================================
# LOAD
# =========================================================

def load_baseline_predictions() -> pd.DataFrame:
    paths = {
        42: INPUT_DIR / "baseline_predictions_seed_42.csv",
        123: INPUT_DIR / "baseline_predictions_seed_123.csv",
    }

    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Arquivos baseline do EXP-006C-R2 não encontrados:\n"
            + "\n".join(str(p) for p in missing)
            + "\nRode primeiro o EXP-006C-R2 --final ou copie os CSVs para o diretório esperado."
        )

    frames = []

    for seed, path in paths.items():
        df = pd.read_csv(path)
        df["seed"] = seed
        df["source_file"] = str(path)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    if "is_fraud" not in out.columns:
        raise ValueError("Predições baseline precisam conter coluna is_fraud.")

    if "decisao" not in out.columns:
        raise ValueError("Predições baseline precisam conter coluna decisao.")

    out["is_fraud"] = pd.to_numeric(out["is_fraud"], errors="coerce").fillna(0).astype(int)
    out["decisao"] = out["decisao"].astype(str)

    return out


# =========================================================
# CLASSIFICAÇÃO FN / FP
# =========================================================

def classify_fn(row: pd.Series) -> dict[str, Any]:
    vl = safe_float(get_col(row, "vl_pix", 0))
    idade = safe_float(get_col(row, "nr_idade", 0))
    rel = safe_float(get_col(row, "qt_tempo_relacionamento_mes", 999))
    first = boolish_flag(row, "first_receiver_flag")
    pix_random = boolish_flag(row, "pix_key_random_flag")

    lgbm_raw = safe_float(get_col(row, "lgbm_raw", 0))
    lgbm_mapped = safe_float(get_col(row, "lgbm_mapped", 0))
    if_pct = safe_float(get_col(row, "if_percentile", 0))
    se = safe_float(get_col(row, "se_score", 0))
    beh = safe_float(get_col(row, "beh_score", 0))
    score_final = safe_float(get_col(row, "score_final", 0))

    veto_reason = get_col(row, "veto_reason", "")
    veto_suppressed = get_col(row, "veto_suppressed_reason", "")

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
    if lgbm_raw >= 0.05:
        flags.append("LGBM_HAS_SIGNAL")
    if if_pct >= 0.985:
        flags.append("IF_EXTREMO")
    elif if_pct >= 0.95:
        flags.append("IF_ALTO")
    if se >= 40:
        flags.append("SE_ATIVO")
    if beh >= 15:
        flags.append("BEH_ATIVO")
    if score_final >= 55:
        flags.append("NEAR_THRESHOLD")
    if has_text_value(veto_suppressed):
        flags.append("VETO_SUPPRESSED")

    # Classificação principal.
    if has_text_value(veto_suppressed):
        recoverability = "ENGINE_SUPPRESSED"
        reason = "Existe veto_suppressed_reason real; investigar guard rail/veto antes de nova regra."

    elif score_final >= 55 and score_final < 62:
        recoverability = "NEAR_THRESHOLD"
        reason = "Score final está perto do threshold de confirmação; candidato a auditoria de composição de score."

    elif vl >= 15000 and rel <= 12 and first == 1 and if_pct >= 0.985 and (se >= 40 or beh >= 15):
        recoverability = "CONTEXTUAL_HIGH_VALUE"
        reason = "Padrão de alto valor/conta nova/primeiro recebedor com IF e regra; verificar se V1 deveria capturar."

    elif if_pct >= 0.985 and (se >= 40 or beh >= 15):
        recoverability = "MODULE_SIGNAL_STRONG"
        reason = "IF extremo com SE/BEH ativo; possivelmente recuperável por ajuste cirúrgico do engine."

    elif lgbm_raw >= 0.05 and first == 1:
        recoverability = "LGBM_FIRST_RECEIVER_GRAY"
        reason = "LGBM em zona cinza e first_receiver; sinal existe, mas risco de FP alto."

    elif if_pct >= 0.95 or se >= 40 or beh >= 15:
        recoverability = "MODULE_SIGNAL_WEAK"
        reason = "Algum módulo tem sinal, mas não há convergência suficiente."

    elif lgbm_raw < 0.05 and if_pct < 0.80 and se <= 0 and beh <= 0 and score_final < 45:
        recoverability = "DATA_LIMITED_WEAK_ALL_MODULES"
        reason = "Todos os módulos estão fracos; provável dependência de novos dados."

    else:
        recoverability = "MIXED_OR_UNCLEAR"
        reason = "Sinais mistos; precisa de análise manual ou meta-learner shadow."

    # Ação recomendada.
    if recoverability in {"ENGINE_SUPPRESSED", "CONTEXTUAL_HIGH_VALUE"}:
        next_action = "AUDITAR_ENGINE_CIRURGICO"
    elif recoverability == "NEAR_THRESHOLD":
        next_action = "AUDITAR_COMPOSICAO_SCORE"
    elif recoverability == "MODULE_SIGNAL_STRONG":
        next_action = "TESTAR_CONTRAFACTUAL_CIRURGICO"
    elif recoverability == "DATA_LIMITED_WEAK_ALL_MODULES":
        next_action = "MARCAR_COMO_PROVAVEL_DATA_LIMITED"
    else:
        next_action = "MANTER_PARA_META_LEARNER_SHADOW"

    return {
        "recoverability_class": recoverability,
        "recoverability_reason": reason,
        "next_action": next_action,
        "flags": "|".join(flags),
        "value_bin": value_bin(vl),
        "relationship_bin": relationship_bin(rel),
        "age_bin": age_bin(idade),
        "lgbm_bin": lgbm_bin(lgbm_raw),
        "if_bin": if_bin(if_pct),
        "score_bin": score_bin(score_final),
        "vl_pix_num": vl,
        "idade_num": idade,
        "rel_meses_num": rel,
        "first_receiver_num": first,
        "pix_random_num": pix_random,
        "lgbm_raw_num": lgbm_raw,
        "lgbm_mapped_num": lgbm_mapped,
        "if_percentile_num": if_pct,
        "se_score_num": se,
        "beh_score_num": beh,
        "score_final_num": score_final,
        "veto_reason_norm": "" if not has_text_value(veto_reason) else str(veto_reason),
        "veto_suppressed_reason_norm": "" if not has_text_value(veto_suppressed) else str(veto_suppressed),
    }


def classify_fp(row: pd.Series) -> dict[str, Any]:
    vl = safe_float(get_col(row, "vl_pix", 0))
    idade = safe_float(get_col(row, "nr_idade", 0))
    rel = safe_float(get_col(row, "qt_tempo_relacionamento_mes", 999))
    first = boolish_flag(row, "first_receiver_flag")
    pix_random = boolish_flag(row, "pix_key_random_flag")

    lgbm_raw = safe_float(get_col(row, "lgbm_raw", 0))
    if_pct = safe_float(get_col(row, "if_percentile", 0))
    se = safe_float(get_col(row, "se_score", 0))
    beh = safe_float(get_col(row, "beh_score", 0))
    score_final = safe_float(get_col(row, "score_final", 0))

    veto_reason = get_col(row, "veto_reason", "")
    veto_suppressed = get_col(row, "veto_suppressed_reason", "")

    flags = []

    if first == 1:
        flags.append("FIRST_RECEIVER")
    if pix_random == 1:
        flags.append("PIX_RANDOM")
    if vl >= 5000:
        flags.append("VALOR_5K_PLUS")
    if rel <= 12:
        flags.append("REL_CURTO")
    if if_pct >= 0.985:
        flags.append("IF_EXTREMO")
    elif if_pct >= 0.95:
        flags.append("IF_ALTO")
    if se >= 40:
        flags.append("SE_ATIVO")
    if beh >= 15:
        flags.append("BEH_ATIVO")
    if lgbm_raw < 0.01:
        flags.append("LGBM_VERY_LOW")
    elif lgbm_raw < 0.05:
        flags.append("LGBM_LOW")
    if has_text_value(veto_suppressed):
        flags.append("VETO_SUPPRESSED")

    # Classificação para redução de FP.
    if lgbm_raw < 0.01 and if_pct < 0.80 and se <= 0 and beh <= 0:
        reducibility = "FP_REDUCIBLE_WEAK_ALL_MODULES"
        reason = "Todos os módulos parecem fracos; candidato a redução de FP se não afetar TP."

    elif lgbm_raw < 0.05 and se <= 0 and beh <= 0 and if_pct < 0.95:
        reducibility = "FP_REDUCIBLE_LOW_SIGNAL"
        reason = "Sinal baixo; pode ser alvo de guard rail mais fino."

    elif first == 1 and se <= 0 and beh <= 0 and if_pct < 0.95:
        reducibility = "FP_FIRST_RECEIVER_LOW_CONTEXT"
        reason = "Provável falso positivo induzido por first_receiver sem suporte de SE/BEH/IF extremo."

    elif score_final >= 62 and score_final < 65:
        reducibility = "FP_NEAR_THRESHOLD"
        reason = "Confirmado por pequena margem; candidato a auditoria de score."

    elif has_text_value(veto_suppressed):
        reducibility = "FP_WITH_SUPPRESSED_VETO"
        reason = "Existe veto suprimido; investigar por que ainda confirmou."

    else:
        reducibility = "FP_NOT_OBVIOUS"
        reason = "FP com sinais fortes ou mistos; cautela para não perder TP."

    if reducibility in {"FP_REDUCIBLE_WEAK_ALL_MODULES", "FP_REDUCIBLE_LOW_SIGNAL", "FP_NEAR_THRESHOLD"}:
        next_action = "CANDIDATO_REDUCAO_FP_SHADOW"
    else:
        next_action = "NAO_AJUSTAR_SEM_AUDITORIA_MANUAL"

    return {
        "fp_reducibility_class": reducibility,
        "fp_reducibility_reason": reason,
        "next_action": next_action,
        "flags": "|".join(flags),
        "value_bin": value_bin(vl),
        "relationship_bin": relationship_bin(rel),
        "age_bin": age_bin(idade),
        "lgbm_bin": lgbm_bin(lgbm_raw),
        "if_bin": if_bin(if_pct),
        "score_bin": score_bin(score_final),
        "vl_pix_num": vl,
        "idade_num": idade,
        "rel_meses_num": rel,
        "first_receiver_num": first,
        "pix_random_num": pix_random,
        "lgbm_raw_num": lgbm_raw,
        "if_percentile_num": if_pct,
        "se_score_num": se,
        "beh_score_num": beh,
        "score_final_num": score_final,
        "veto_reason_norm": "" if not has_text_value(veto_reason) else str(veto_reason),
        "veto_suppressed_reason_norm": "" if not has_text_value(veto_suppressed) else str(veto_suppressed),
    }


def enrich_rows(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    classifier = classify_fn if kind == "FN" else classify_fp
    rows = []

    for _, row in df.iterrows():
        base = row.to_dict()
        extra = classifier(row)
        base.update(extra)
        rows.append(base)

    return pd.DataFrame(rows)


# =========================================================
# MÉTRICAS
# =========================================================

def flagged(df: pd.DataFrame) -> pd.Series:
    return df["decisao"].astype(str).isin(DECISOES_POSITIVAS)


def compute_seed_metrics(df: pd.DataFrame) -> dict[str, Any]:
    y = df["is_fraud"].astype(int)
    pred = flagged(df).astype(int)

    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": round(precision, 6),
        "Recall": round(recall, 6),
        "F1": round(f1, 6),
        "FPR": round(fpr, 8),
    }


def build_clusters(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if kind == "FN":
        group_cols = [
            "recoverability_class",
            "next_action",
            "value_bin",
            "relationship_bin",
            "age_bin",
            "lgbm_bin",
            "if_bin",
            "score_bin",
            "first_receiver_num",
            "pix_random_num",
        ]
    else:
        group_cols = [
            "fp_reducibility_class",
            "next_action",
            "value_bin",
            "relationship_bin",
            "age_bin",
            "lgbm_bin",
            "if_bin",
            "score_bin",
            "first_receiver_num",
            "pix_random_num",
        ]

    rows = []

    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["count"] = int(len(g))
        row["seeds"] = ",".join(str(x) for x in sorted(g["seed"].dropna().unique().tolist()))
        row["avg_vl_pix"] = round(float(g["vl_pix_num"].mean()), 2)
        row["avg_lgbm_raw"] = round(float(g["lgbm_raw_num"].mean()), 6)
        row["avg_if_percentile"] = round(float(g["if_percentile_num"].mean()), 6)
        row["avg_se_score"] = round(float(g["se_score_num"].mean()), 2)
        row["avg_beh_score"] = round(float(g["beh_score_num"].mean()), 2)
        row["avg_score_final"] = round(float(g["score_final_num"].mean()), 2)
        id_col = "transaction_id" if "transaction_id" in g.columns else None
        row["examples"] = ", ".join(str(x) for x in g[id_col].dropna().astype(str).head(5).tolist()) if id_col else ""
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["count", "avg_vl_pix"], ascending=[False, False])


def build_recoverability_assessment(fn_df: pd.DataFrame, fp_df: pd.DataFrame, metrics: dict[str, Any]) -> dict[str, Any]:
    fn_counts = fn_df["recoverability_class"].value_counts().to_dict() if not fn_df.empty else {}
    fp_counts = fp_df["fp_reducibility_class"].value_counts().to_dict() if not fp_df.empty else {}

    unique_fn = fn_df.drop_duplicates(subset=["transaction_id"]) if "transaction_id" in fn_df.columns else fn_df
    unique_fp = fp_df.drop_duplicates(subset=["transaction_id"]) if "transaction_id" in fp_df.columns else fp_df

    fn_action_counts = unique_fn["next_action"].value_counts().to_dict() if not unique_fn.empty else {}
    fp_action_counts = unique_fp["next_action"].value_counts().to_dict() if not unique_fp.empty else {}

    likely_data_limited = int((unique_fn["recoverability_class"] == "DATA_LIMITED_WEAK_ALL_MODULES").sum()) if not unique_fn.empty else 0
    likely_recoverable = int(unique_fn["recoverability_class"].isin([
        "ENGINE_SUPPRESSED",
        "NEAR_THRESHOLD",
        "CONTEXTUAL_HIGH_VALUE",
        "MODULE_SIGNAL_STRONG",
    ]).sum()) if not unique_fn.empty else 0

    fp_reducible = int(unique_fp["fp_reducibility_class"].isin([
        "FP_REDUCIBLE_WEAK_ALL_MODULES",
        "FP_REDUCIBLE_LOW_SIGNAL",
        "FP_NEAR_THRESHOLD",
    ]).sum()) if not unique_fp.empty else 0

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed_metrics": metrics,
        "fn": {
            "rows": int(len(fn_df)),
            "unique_transactions": int(len(unique_fn)),
            "class_counts": fn_counts,
            "action_counts": fn_action_counts,
            "likely_recoverable_count": likely_recoverable,
            "likely_data_limited_count": likely_data_limited,
        },
        "fp": {
            "rows": int(len(fp_df)),
            "unique_transactions": int(len(unique_fp)),
            "class_counts": fp_counts,
            "action_counts": fp_action_counts,
            "likely_reducible_count": fp_reducible,
        },
        "interpretation": {
            "if_many_data_limited": (
                "Se a maioria dos FNs estiver em DATA_LIMITED_WEAK_ALL_MODULES, "
                "a redução adicional de FN depende de novas fontes como reputação de recebedor, grafo, device/session."
            ),
            "if_many_near_threshold": (
                "Se houver FNs em NEAR_THRESHOLD, priorizar auditoria de composição de score antes de novo modelo."
            ),
            "if_many_fp_reducible": (
                "Se houver FPs em classes reducíveis, testar guard rail anti-FP em shadow, mas só promover se não perder TP."
            ),
        },
    }


# =========================================================
# RELATÓRIOS
# =========================================================

def write_recommendations(path: Path, fn_df: pd.DataFrame, fp_df: pd.DataFrame, assessment: dict[str, Any]) -> None:
    fn_info = assessment["fn"]
    fp_info = assessment["fp"]

    lines = [
        "# EXP-006D — Baseline Residual FN Census",
        "",
        f"Gerado em: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Objetivo",
        "",
        "Classificar os FNs e FPs residuais do baseline pós-FASE 1 usando apenas artefatos existentes.",
        "",
        "## Métricas baseline observadas",
        "",
        "| Seed | TP | FP | FN | Precision | Recall | F1 | FPR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for seed, m in assessment["seed_metrics"].items():
        lines.append(
            f"| {seed} | {m['TP']} | {m['FP']} | {m['FN']} | "
            f"{m['Precision']:.4%} | {m['Recall']:.4%} | {m['F1']:.4f} | {m['FPR']:.4%} |"
        )

    lines.extend([
        "",
        "## Censo dos FNs",
        "",
        f"- Linhas FN: `{fn_info['rows']}`",
        f"- Transações FN únicas: `{fn_info['unique_transactions']}`",
        f"- Provavelmente recuperáveis: `{fn_info['likely_recoverable_count']}`",
        f"- Provavelmente limitadas por dados: `{fn_info['likely_data_limited_count']}`",
        "",
        "### Classes FN",
        "",
    ])

    for k, v in fn_info["class_counts"].items():
        lines.append(f"- `{k}`: {v}")

    lines.extend([
        "",
        "## Censo dos FPs",
        "",
        f"- Linhas FP: `{fp_info['rows']}`",
        f"- Transações FP únicas: `{fp_info['unique_transactions']}`",
        f"- FPs potencialmente reducíveis: `{fp_info['likely_reducible_count']}`",
        "",
        "### Classes FP",
        "",
    ])

    for k, v in fp_info["class_counts"].items():
        lines.append(f"- `{k}`: {v}")

    lines.extend([
        "",
        "## Recomendação técnica",
        "",
    ])

    if fn_info["likely_recoverable_count"] > 0:
        lines.extend([
            "Há FNs com algum sinal recuperável. O próximo experimento deve ser artifact-only ou shadow, focado somente nesses casos.",
            "Não criar regra ampla; gerar contrafactual por classe FN e estimar impacto sobre FPs/TPs antes de qualquer E2E.",
        ])
    elif fn_info["likely_data_limited_count"] >= max(1, fn_info["unique_transactions"] // 2):
        lines.extend([
            "A maioria dos FNs parece limitada pelos dados atuais.",
            "A próxima melhoria relevante provavelmente depende de novas fontes: reputação/grafo do recebedor, device/session, MED/contestação ou histórico externo.",
        ])
    else:
        lines.extend([
            "Os FNs têm sinais mistos. Priorizar meta-learner shadow ou auditoria manual antes de novo patch no engine.",
        ])

    if fp_info["likely_reducible_count"] > 0:
        lines.extend([
            "",
            "Também há FPs potencialmente reducíveis. O próximo experimento pode testar um guard rail anti-FP em shadow, desde que valide que nenhum TP seria perdido.",
        ])

    lines.extend([
        "",
        "## Próximo passo recomendado",
        "",
        "`EXP-006E — Residual FN/FP Counterfactual Designer`",
        "",
        "Objetivo: a partir do censo, gerar 1 única hipótese candidata, artifact-only, e só então decidir se vale quick-E2E.",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def write_next_experiment_spec(path: Path, assessment: dict[str, Any]) -> None:
    fn_info = assessment["fn"]
    fp_info = assessment["fp"]

    if fn_info["likely_recoverable_count"] > 0:
        recommendation = "EXP-006E — Residual FN Counterfactual Designer"
        objective = (
            "Usar o censo dos FNs para desenhar uma hipótese cirúrgica que recupere pelo menos 1 FN "
            "sem adicionar FP e sem perder TP."
        )
    elif fp_info["likely_reducible_count"] > 0:
        recommendation = "EXP-006E — FP Reduction Shadow Guard"
        objective = (
            "Testar em shadow se é possível reduzir FPs atuais sem perder nenhum TP, usando apenas casos com sinais fracos."
        )
    else:
        recommendation = "EXP-007A — Meta-Learner Shadow"
        objective = (
            "Treinar meta-learner shadow para verificar se há separabilidade não-linear nos sinais atuais. "
            "Se não houver, encerrar FASE 2 como limitada pelos dados."
        )

    lines = [
        "# EXP-006D — Próximo Experimento",
        "",
        f"## Recomendado: {recommendation}",
        "",
        "## Objetivo",
        "",
        objective,
        "",
        "## Regras de produtividade",
        "",
        "1. Rodar primeiro artifact-only.",
        "2. Não rodar grid E2E.",
        "3. Não testar mais de 1 candidato por quick-E2E.",
        "4. Só rodar quick-E2E se a simulação artifact-only mostrar ganho líquido.",
        "5. Parar se FP subir, FN não cair ou houver TP perdido.",
        "",
        "## Critério mínimo para quick-E2E",
        "",
        "- Delta FN estimado <= -1; ou",
        "- Delta FP estimado <= -2 sem TP perdido.",
        "",
        "## Critério de promoção",
        "",
        "- FN cai nos dois seeds; ou FP cai nos dois seeds;",
        "- F1 não piora;",
        "- nenhum TP perdido;",
        "- explicação causal clara por classe de erro.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EXP-006D — Baseline Residual FN Census")
    print("=" * 72)

    print("[1/7] Carregando predições baseline...")
    all_preds = load_baseline_predictions()

    print(f"[OK] Linhas carregadas: {len(all_preds)}")
    print(f"[OK] Seeds: {sorted(all_preds['seed'].unique().tolist())}")

    print("[2/7] Calculando métricas por seed...")
    seed_metrics = {}
    for seed, g in all_preds.groupby("seed"):
        seed_metrics[str(seed)] = compute_seed_metrics(g)

    print("[OK] Métricas:")
    for seed, m in seed_metrics.items():
        print(f"  seed={seed}: TP={m['TP']} FP={m['FP']} FN={m['FN']} F1={m['F1']}")

    print("[3/7] Extraindo FNs e FPs baseline...")
    is_flagged = flagged(all_preds)

    fn_raw = all_preds[(all_preds["is_fraud"] == 1) & (~is_flagged)].copy()
    fp_raw = all_preds[(all_preds["is_fraud"] == 0) & (is_flagged)].copy()

    print(f"[OK] FN rows: {len(fn_raw)}")
    print(f"[OK] FP rows: {len(fp_raw)}")

    print("[4/7] Classificando FNs...")
    fn_df = enrich_rows(fn_raw, kind="FN")
    fn_path = OUTPUT_DIR / "01_fn_residual_census.csv"
    fn_df.to_csv(fn_path, index=False, encoding="utf-8-sig")
    print(f"[OK] FN census salvo: {fn_path}")

    print("[5/7] Classificando FPs...")
    fp_df = enrich_rows(fp_raw, kind="FP")
    fp_path = OUTPUT_DIR / "02_fp_census.csv"
    fp_df.to_csv(fp_path, index=False, encoding="utf-8-sig")
    print(f"[OK] FP census salvo: {fp_path}")

    print("[6/7] Gerando clusters e assessment...")
    fn_clusters = build_clusters(fn_df, kind="FN")
    fp_clusters = build_clusters(fp_df, kind="FP")

    fn_clusters_path = OUTPUT_DIR / "03_fn_clusters.csv"
    fp_clusters_path = OUTPUT_DIR / "04_fp_clusters.csv"

    fn_clusters.to_csv(fn_clusters_path, index=False, encoding="utf-8-sig")
    fp_clusters.to_csv(fp_clusters_path, index=False, encoding="utf-8-sig")

    assessment = build_recoverability_assessment(fn_df, fp_df, seed_metrics)
    write_json(OUTPUT_DIR / "05_recoverability_assessment.json", assessment)

    print(f"[OK] FN clusters salvo: {fn_clusters_path}")
    print(f"[OK] FP clusters salvo: {fp_clusters_path}")

    print("[7/7] Escrevendo recomendações...")
    write_json(
        OUTPUT_DIR / "00_input_summary.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "input_dir": str(INPUT_DIR),
            "output_dir": str(OUTPUT_DIR),
            "baseline_reference": BASELINE_REF,
            "loaded_rows": int(len(all_preds)),
            "fn_rows": int(len(fn_df)),
            "fp_rows": int(len(fp_df)),
            "seed_metrics": seed_metrics,
        },
    )

    write_recommendations(OUTPUT_DIR / "06_recomendacoes.md", fn_df, fp_df, assessment)
    write_next_experiment_spec(OUTPUT_DIR / "07_next_experiment_spec.md", assessment)

    print()
    print("[OK] EXP-006D concluído sem E2E.")
    print(f"[OK] Artefatos em: {OUTPUT_DIR}")
    print()
    print("Arquivos principais:")
    print(f"  {OUTPUT_DIR / '01_fn_residual_census.csv'}")
    print(f"  {OUTPUT_DIR / '02_fp_census.csv'}")
    print(f"  {OUTPUT_DIR / '05_recoverability_assessment.json'}")
    print(f"  {OUTPUT_DIR / '06_recomendacoes.md'}")
    print(f"  {OUTPUT_DIR / '07_next_experiment_spec.md'}")


if __name__ == "__main__":
    main()