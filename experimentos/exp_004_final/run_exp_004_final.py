"""
experimentos/exp_004_final/run_exp_004_final.py

EXP-004-FINAL: Capstone de Politica Contextual para fechamento da FASE 1.

Objetivo:
  Fechar a FASE 1 com um unico experimento que testa, de forma controlada:
    V0 BASELINE atual: EXP-001 + EXP-002
    V1 Guard rail contextual para alto valor SE+BEH
    V2 RATE_LIMIT_ANOMALO comportamental
    V3 PRIMEIRO_RECEIVER_VALOR_ANOMALO
    V4 Combo final: V1 + V2 + V3

Estrategia:
  1. Carrega dados completos.
  2. Pre-computa features temporais por cliente no dataset completo.
  3. Gera sample estratificado, preservando 100% das fraudes.
  4. Processa o sample UMA vez via PipelineOrquestrador real.
  5. Aplica variantes post-hoc sem alterar engine/social/behavioral.
  6. Gera artefatos comparaveis aos EXP-001/002/003.
  7. Valida a variante vencedora em seed independente.

Artefatos gerados:
  resultados/experimentos/EXP-004-FINAL/
    01_tabela_comparativa.csv
    02_delta_fp_fn_por_variante.json
    03_fn_residuais_classificados.csv
    04_validacao_cruzada.json
    05_conclusao_fase_1.md
    06_meta_shadow_report.md

Uso:
  python experimentos/exp_004_final/run_exp_004_final.py --workers 4
  python experimentos/exp_004_final/run_exp_004_final.py --workers 4 --sample 6000
  python experimentos/exp_004_final/run_exp_004_final.py --workers 4 --skip-validation
  python experimentos/exp_004_final/run_exp_004_final.py --workers 4 --save-predictions

Observacao importante:
  Este runner e um experimento de politica post-hoc. Se V1/V2/V3/V4 vencerem,
  a etapa seguinte e promover a regra vencedora para o Decision Engine, SE ou BEH.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =========================================================
# PATHS ROBUSTOS
# =========================================================

EXP_DIR = Path(__file__).resolve().parent


def _find_project_root(start: Path) -> Path:
    """Encontra a raiz do projeto procurando backend/, dados/ e experimentos/."""
    candidates = [start, *start.parents]
    for p in candidates:
        if (p / "backend").exists() and (p / "dados").exists() and (p / "experimentos").exists():
            return p

    # Fallback para a estrutura padrao: experimentos/exp_004_final/run_exp_004_final.py
    return start.parent.parent


PROJECT_ROOT = _find_project_root(EXP_DIR)
sys.path.insert(0, str(PROJECT_ROOT))


from experimentos.utils_experimentos import (  # noqa: E402
    compute_metrics,
    get_experiment_output_dir,
    get_logger,
    load_dataset,
    print_section,
    process_dataframe_via_orquestrador,
    safe_json_dump,
    stratified_sample,
)


# =========================================================
# CONFIGURACAO GERAL
# =========================================================

EXP_ID = "EXP-004-FINAL"
EXP_TITLE = "Capstone de Politica Contextual para fechamento da FASE 1"
logger = get_logger(EXP_ID)

BASELINE_ENGINE_OVERRIDES: dict[str, Any] = {
    # Estado atual esperado pos EXP-001 + EXP-002
    "threshold_confirmar": 62.0,
    "threshold_bloquear": 95.0,
    "lgbm_guard_enabled": True,
    "lgbm_guard_threshold": 0.30,

    # Garantir que EXP-003 residual nao interfira neste capstone,
    # se o EngineConfig atual ja expuser estes campos.
    "se_pattern_residual_enabled": False,
    "exp003_residual_confirm_enabled": False,
}

VARIANTS: list[dict[str, Any]] = [
    {
        "id": "V1_GUARD_CONTEXTUAL",
        "label": "Excecao cirurgica ao guard rail: alto valor SE+BEH",
        "use_guard_exception": True,
        "use_rate_limit": False,
        "use_primeiro_receiver": False,
    },
    {
        "id": "V2_RATE_LIMIT",
        "label": "RATE_LIMIT_ANOMALO comportamental",
        "use_guard_exception": False,
        "use_rate_limit": True,
        "use_primeiro_receiver": False,
    },
    {
        "id": "V3_PRIMEIRO_RECEIVER",
        "label": "PRIMEIRO_RECEIVER_VALOR_ANOMALO",
        "use_guard_exception": False,
        "use_rate_limit": False,
        "use_primeiro_receiver": True,
    },
    {
        "id": "V4_COMBO_FINAL",
        "label": "Guard contextual + Rate limit + Primeiro receiver",
        "use_guard_exception": True,
        "use_rate_limit": True,
        "use_primeiro_receiver": True,
    },
]

RULE_COLUMNS = {
    "GUARD_EXCEPTION_ALTO_VALOR_SE_BEH": "exp004_guard_exception_alto_valor_se_beh",
    "RATE_LIMIT_ANOMALO": "exp004_rate_limit_anomalo",
    "PRIMEIRO_RECEIVER_VALOR_ANOMALO": "exp004_primeiro_receiver_valor_anomalo",
}

CONTEXT_COLUMNS = [
    "exp004_tx_count_30m",
    "exp004_tx_count_60m",
    "exp004_sum_vl_30m",
    "exp004_sum_vl_60m",
    "exp004_distinct_receivers_30m",
    "exp004_distinct_receivers_60m",
    "exp004_distinct_keys_30m",
    "exp004_distinct_keys_60m",
    "exp004_hist_count_90d",
    "exp004_hist_avg_90d",
    "exp004_hist_sum_90d",
]


# =========================================================
# HELPERS DE COLUNAS E TIPOS
# =========================================================

def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_key_series(s: pd.Series) -> pd.Series:
    """Normaliza chave textual para merge robusto."""
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.replace({"nan": "", "None": "", "NaT": "", "<NA>": ""})
    return out


def _text_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="object")
    return df[col].fillna(default).astype(str)


def _num_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _bool_numeric(df: pd.DataFrame, col: str, default: int = 0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="int64")
    s = df[col]
    if s.dtype == bool:
        return s.astype(int)
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(int)


def _digits_only_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.replace(r"\D", "", regex=True)


def _flagged(df: pd.DataFrame) -> pd.Series:
    return df["decisao"].astype(str).isin(["CONFIRMAR", "BLOQUEAR"])


def _select_existing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def _infer_pf_like(df: pd.DataFrame) -> pd.Series:
    """
    Infere pessoa fisica de forma conservadora.

    Prioridade:
      1. Se existir tipo_pessoa/tp_pessoa, usa F/J.
      2. Se nao existir, considera PF quando idade esta entre 1 e 110
         e o documento nao parece CNPJ completo de 14 digitos.
      3. Exclui naturalmente casos PJ com idade=0.
    """
    idx = df.index
    pf_by_tipo = pd.Series(False, index=idx)
    has_tipo = pd.Series(False, index=idx)

    tipo_col = _first_existing_col(df, ["tipo_pessoa", "tp_pessoa", "ds_tipo_pessoa"])
    if tipo_col is not None:
        tipo = _text_series(df, tipo_col).str.upper().str.strip().str[0]
        has_tipo = tipo.isin(["F", "J"])
        pf_by_tipo = tipo.eq("F")

    idade = _num_series(df, "nr_idade", default=0)
    customer_digits = _digits_only_series(_text_series(df, "customer_id"))
    doc_len = customer_digits.str.len()

    pf_inferida = idade.between(1, 110) & (doc_len < 14)
    return (has_tipo & pf_by_tipo) | (~has_tipo & pf_inferida)


# =========================================================
# NORMALIZACAO DE OUTPUT DO PIPELINE
# =========================================================

def _ensure_prediction_aliases(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Garante aliases canonicos usados pelo experimento.

    O PipelineOrquestrador ja costuma devolver estas colunas, mas este helper
    reduz fragilidade se algum nome variar entre versoes.
    """
    df = preds.copy()

    alias_map: dict[str, list[str]] = {
        "transaction_id": ["transaction_id", "cd_pix", "id_transacao"],
        "customer_id": ["customer_id", "cd_cpf_pagador", "cpf_pagador", "documento_pagador"],
        "receiver_id": ["receiver_id", "cd_cpf_cnpj_recebedor", "cpf_cnpj_recebedor"],
        "event_datetime": ["event_datetime", "dt_pix", "data_hora", "timestamp"],
        "lgbm_raw": ["lgbm_raw", "score_lgbm_raw", "p_lgbm"],
        "lgbm_mapped": ["lgbm_mapped", "score_lgbm_mapped"],
        "if_percentile": ["if_percentile", "score_if", "if_score", "p_if"],
        "if_raw": ["if_raw", "score_if_raw"],
        "se_score": ["se_score", "social_engineering_score"],
        "beh_score": ["beh_score", "behavioral_score", "behavioral_risk_score"],
        "se_worst_pattern": ["se_worst_pattern", "worst_pattern"],
        "pix_key_random_flag": ["pix_key_random_flag", "pix_random_flag", "is_pix_key_random"],
        "first_receiver_flag": ["first_receiver_flag", "tp_primeiro_envio_recebedor_trimestre"],
        "is_first_tx_trimestre": ["is_first_tx_trimestre"],
        "burst_30m_flag": ["burst_30m_flag"],
        "tx_count_prev_30m": ["tx_count_prev_30m"],
        "qt_tempo_relacionamento_mes": ["qt_tempo_relacionamento_mes"],
        "nr_idade": ["nr_idade"],
        "vl_pix": ["vl_pix", "valor_pix"],
        "veto_reason": ["veto_reason"],
        "veto_suppressed_reason": ["veto_suppressed_reason"],
    }

    for canonical, candidates in alias_map.items():
        if canonical not in df.columns:
            found = _first_existing_col(df, candidates)
            if found is not None:
                df[canonical] = df[found]

    # Defaults para colunas criticas.
    defaults: dict[str, Any] = {
        "transaction_id": "",
        "customer_id": "",
        "receiver_id": "",
        "event_datetime": pd.NaT,
        "lgbm_raw": 0.0,
        "lgbm_mapped": 0.0,
        "if_percentile": 0.0,
        "if_raw": 0.0,
        "se_score": 0.0,
        "beh_score": 0.0,
        "se_worst_pattern": "",
        "pix_key_random_flag": 0,
        "first_receiver_flag": 0,
        "is_first_tx_trimestre": 0,
        "burst_30m_flag": 0,
        "tx_count_prev_30m": 0,
        "qt_tempo_relacionamento_mes": 999,
        "nr_idade": 0,
        "vl_pix": 0.0,
        "veto_reason": "",
        "veto_suppressed_reason": "",
    }

    for c, default in defaults.items():
        if c not in df.columns:
            df[c] = default

    if "decisao" not in df.columns:
        raise ValueError("Coluna obrigatoria ausente no output do pipeline: decisao")

    if "is_fraud" not in df.columns:
        raise ValueError("Coluna obrigatoria ausente no output do pipeline: is_fraud")

    return df


# =========================================================
# CONTEXTO TEMPORAL POR CLIENTE
# =========================================================

def _compute_temporal_context(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-computa features temporais por cliente usando o dataset completo.

    As janelas sao causais no nivel do processamento ordenado:
      - historico 90d e calculado ANTES de inserir a transacao corrente
      - janelas 30m/60m incluem a transacao corrente, pois representam o
        estado no momento da decisao desta transacao

    Retorna DataFrame com:
      _tx_key + CONTEXT_COLUMNS
    """
    logger.info("Pre-computando contexto temporal EXP-004-FINAL no dataset completo...")

    df = df_full.copy().reset_index(drop=True)

    tx_col = _first_existing_col(df, ["transaction_id", "cd_pix", "id_transacao"])
    customer_col = _first_existing_col(df, ["customer_id", "cd_cpf_pagador", "cpf_pagador", "documento_pagador"])
    receiver_col = _first_existing_col(df, ["receiver_id", "cd_cpf_cnpj_recebedor", "cpf_cnpj_recebedor"])
    dt_col = _first_existing_col(df, ["event_datetime", "dt_pix", "data_hora", "timestamp"])
    key_col = _first_existing_col(df, ["ds_chave_pix", "pix_key", "chave_pix"])
    value_col = _first_existing_col(df, ["vl_pix", "valor_pix"])

    missing = []
    if tx_col is None:
        missing.append("transaction_id/cd_pix")
    if customer_col is None:
        missing.append("customer_id/cd_cpf_pagador")
    if dt_col is None:
        missing.append("event_datetime/dt_pix")
    if value_col is None:
        missing.append("vl_pix")

    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes para contexto temporal: {missing}")

    df["_tx_key"] = _normalize_key_series(df[tx_col])
    df["_customer_id"] = _normalize_key_series(df[customer_col])
    df["_receiver_id"] = _normalize_key_series(df[receiver_col]) if receiver_col else ""
    df["_pix_key"] = _normalize_key_series(df[key_col]) if key_col else ""
    df["_event_dt"] = pd.to_datetime(df[dt_col], errors="coerce")
    df["_vl_pix"] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0).astype(float)

    n = len(df)
    arrays: dict[str, np.ndarray] = {
        "exp004_tx_count_30m": np.zeros(n, dtype=float),
        "exp004_tx_count_60m": np.zeros(n, dtype=float),
        "exp004_sum_vl_30m": np.zeros(n, dtype=float),
        "exp004_sum_vl_60m": np.zeros(n, dtype=float),
        "exp004_distinct_receivers_30m": np.zeros(n, dtype=float),
        "exp004_distinct_receivers_60m": np.zeros(n, dtype=float),
        "exp004_distinct_keys_30m": np.zeros(n, dtype=float),
        "exp004_distinct_keys_60m": np.zeros(n, dtype=float),
        "exp004_hist_count_90d": np.zeros(n, dtype=float),
        "exp004_hist_avg_90d": np.zeros(n, dtype=float),
        "exp004_hist_sum_90d": np.zeros(n, dtype=float),
    }

    valid = df[df["_event_dt"].notna() & df["_customer_id"].ne("")].copy()
    valid = valid.sort_values(["_customer_id", "_event_dt", "_tx_key"], kind="mergesort")

    window_30m = pd.Timedelta(minutes=30)
    window_60m = pd.Timedelta(minutes=60)
    window_90d = pd.Timedelta(days=90)

    for _, g in valid.groupby("_customer_id", sort=False):
        q30: deque[tuple[pd.Timestamp, float, str, str]] = deque()
        q60: deque[tuple[pd.Timestamp, float, str, str]] = deque()
        q90: deque[tuple[pd.Timestamp, float, str, str]] = deque()

        sum30 = 0.0
        sum60 = 0.0
        sum90 = 0.0

        for idx, row in g.iterrows():
            ts = row["_event_dt"]
            amount = float(row["_vl_pix"])
            receiver = str(row["_receiver_id"])
            pix_key = str(row["_pix_key"])

            cutoff30 = ts - window_30m
            cutoff60 = ts - window_60m
            cutoff90 = ts - window_90d

            while q30 and q30[0][0] < cutoff30:
                _, old_amount, _, _ = q30.popleft()
                sum30 -= old_amount

            while q60 and q60[0][0] < cutoff60:
                _, old_amount, _, _ = q60.popleft()
                sum60 -= old_amount

            while q90 and q90[0][0] < cutoff90:
                _, old_amount, _, _ = q90.popleft()
                sum90 -= old_amount

            # Historico 90d ANTES da transacao corrente.
            hist_count = len(q90)
            arrays["exp004_hist_count_90d"][idx] = hist_count
            arrays["exp004_hist_sum_90d"][idx] = max(sum90, 0.0)
            arrays["exp004_hist_avg_90d"][idx] = (sum90 / hist_count) if hist_count > 0 else 0.0

            # Janelas recentes INCLUINDO a transacao corrente.
            item = (ts, amount, receiver, pix_key)

            q30.append(item)
            sum30 += amount

            q60.append(item)
            sum60 += amount

            q90.append(item)
            sum90 += amount

            receivers_30 = {x[2] for x in q30 if x[2] not in ("", "nan", "None")}
            receivers_60 = {x[2] for x in q60 if x[2] not in ("", "nan", "None")}
            keys_30 = {x[3] for x in q30 if x[3] not in ("", "nan", "None")}
            keys_60 = {x[3] for x in q60 if x[3] not in ("", "nan", "None")}

            arrays["exp004_tx_count_30m"][idx] = len(q30)
            arrays["exp004_tx_count_60m"][idx] = len(q60)
            arrays["exp004_sum_vl_30m"][idx] = max(sum30, 0.0)
            arrays["exp004_sum_vl_60m"][idx] = max(sum60, 0.0)
            arrays["exp004_distinct_receivers_30m"][idx] = len(receivers_30)
            arrays["exp004_distinct_receivers_60m"][idx] = len(receivers_60)
            arrays["exp004_distinct_keys_30m"][idx] = len(keys_30)
            arrays["exp004_distinct_keys_60m"][idx] = len(keys_60)

    context = pd.DataFrame({"_tx_key": df["_tx_key"]})
    for c, arr in arrays.items():
        context[c] = arr

    context = context.drop_duplicates("_tx_key", keep="last")
    logger.info("Contexto temporal pronto: %d transacoes com chave unica", len(context))
    return context


def _merge_context(preds: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    preds = _ensure_prediction_aliases(preds)
    preds = preds.copy()
    preds["_tx_key"] = _normalize_key_series(preds["transaction_id"])

    merged = preds.merge(context, on="_tx_key", how="left")

    for c in CONTEXT_COLUMNS:
        if c not in merged.columns:
            merged[c] = 0.0
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

    return merged


# =========================================================
# REGRAS DO EXP-004-FINAL
# =========================================================

def _add_rule_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos os sinais candidatos do EXP-004-FINAL.

    Nenhuma decisao e alterada aqui.
    """
    df = _ensure_prediction_aliases(df_in).copy()

    pf_like = _infer_pf_like(df)
    idade = _num_series(df, "nr_idade", 0)
    rel = _num_series(df, "qt_tempo_relacionamento_mes", 999)
    vl = _num_series(df, "vl_pix", 0)
    lgbm = _num_series(df, "lgbm_raw", 0)
    ifp = _num_series(df, "if_percentile", 0)
    se = _num_series(df, "se_score", 0)
    beh = _num_series(df, "beh_score", 0)
    first_receiver = _bool_numeric(df, "first_receiver_flag", 0).eq(1)
    pix_random = _bool_numeric(df, "pix_key_random_flag", 0).eq(1)

    suppressed_reason = _text_series(df, "veto_suppressed_reason")
    suppressed_se_beh_valor_novo = suppressed_reason.str.contains(
        r"SE\+BEH_VALOR_NOVO",
        case=False,
        regex=True,
        na=False,
    )

    # -----------------------------------------------------
    # V1 - Excecao cirurgica ao guard rail
    # -----------------------------------------------------
    # Alvo principal: fraude real R$20k com:
    #   PF, idade valida, relacionamento curto, first_receiver,
    #   IF muito alto, SE=40, BEH=15, LGBM baixo porem nao zero.
    #
    # Exclui o FP PJ/CNPJ de R$20k do EXP-002 por:
    #   pf_like == False e lgbm_raw < 0.01.
    guard_exception = (
        pf_like
        & idade.between(18, 90)
        & (vl >= 15000.0)
        & (rel <= 12)
        & first_receiver
        & (ifp >= 0.985)
        & (se >= 40.0)
        & (beh >= 15.0)
        & (lgbm >= 0.01)
        & (lgbm < 0.30)
        & (suppressed_se_beh_valor_novo | suppressed_reason.ne(""))
    )

    df["exp004_pf_like"] = pf_like.astype(int)
    df[RULE_COLUMNS["GUARD_EXCEPTION_ALTO_VALOR_SE_BEH"]] = guard_exception.astype(bool)

    # -----------------------------------------------------
    # V2 - RATE_LIMIT_ANOMALO
    # -----------------------------------------------------
    tx_count_30m = _num_series(df, "exp004_tx_count_30m", 0)
    tx_count_60m = _num_series(df, "exp004_tx_count_60m", 0)
    sum_vl_30m = _num_series(df, "exp004_sum_vl_30m", 0)
    sum_vl_60m = _num_series(df, "exp004_sum_vl_60m", 0)
    distinct_receivers_30m = _num_series(df, "exp004_distinct_receivers_30m", 0)
    hist_count_90d = _num_series(df, "exp004_hist_count_90d", 0)
    hist_avg_90d = _num_series(df, "exp004_hist_avg_90d", 0)

    t1_burst_formiguinha = (
        (tx_count_30m >= 3)
        & (sum_vl_30m >= 700.0)
    )

    hist_threshold = pd.Series(
        np.maximum(1000.0, 2.5 * hist_avg_90d.astype(float)),
        index=df.index,
    )

    t2_esvaziamento_fracionado = (
        (tx_count_60m >= 2)
        & (hist_count_90d >= 3)
        & (sum_vl_60m >= hist_threshold)
    )

    t3_multiplos_recebedores_curto = (
        (tx_count_60m >= 3)
        & (distinct_receivers_30m >= 2)
        & (sum_vl_60m >= 800.0)
    )

    rate_score = pd.Series(0.0, index=df.index)
    rate_score = rate_score + np.where(t1_burst_formiguinha, 20.0, 0.0)
    rate_score = rate_score + np.where(t2_esvaziamento_fracionado, 25.0, 0.0)
    rate_score = rate_score + np.where(t3_multiplos_recebedores_curto, 15.0, 0.0)
    rate_score = pd.Series(np.minimum(rate_score, 50.0), index=df.index)

    rate_limit_anomalo = (
        (rate_score >= 35.0)
        & ((ifp >= 0.85) | (lgbm >= 0.08))
    )

    df["exp004_rate_t1_burst_formiguinha"] = t1_burst_formiguinha.astype(bool)
    df["exp004_rate_t2_esvaziamento_fracionado"] = t2_esvaziamento_fracionado.astype(bool)
    df["exp004_rate_t3_multiplos_recebedores_curto"] = t3_multiplos_recebedores_curto.astype(bool)
    df["exp004_rate_score"] = rate_score.round(2)
    df[RULE_COLUMNS["RATE_LIMIT_ANOMALO"]] = rate_limit_anomalo.astype(bool)

    # -----------------------------------------------------
    # V3 - PRIMEIRO_RECEIVER_VALOR_ANOMALO
    # -----------------------------------------------------
    pr_threshold_hist = pd.Series(
        np.maximum(800.0, 3.0 * hist_avg_90d.astype(float)),
        index=df.index,
    )

    valor_anomalo_cliente_com_historico = (
        (hist_count_90d >= 5)
        & (hist_avg_90d > 0)
        & (vl >= pr_threshold_hist)
    )

    valor_anomalo_cliente_sem_historico = (
        (hist_count_90d < 5)
        & (vl >= 1000.0)
    )

    pr_convergencia = (
        pix_random
        | (ifp >= 0.90)
        | (lgbm >= 0.20)
        | (idade <= 25)
        | (idade >= 60)
        | (rel <= 12)
    )

    pr_base = (
        pf_like
        & first_receiver
        & (valor_anomalo_cliente_com_historico | valor_anomalo_cliente_sem_historico)
        & pr_convergencia
    )

    pr_score = pd.Series(0.0, index=df.index)
    pr_score = pr_score + np.where(pr_base, 35.0, 0.0)
    pr_score = pr_score + np.where(pr_base & pix_random, 15.0, 0.0)
    pr_score = pr_score + np.where(pr_base & (ifp >= 0.90), 10.0, 0.0)
    pr_score = pr_score + np.where(pr_base & ((idade <= 25) | (idade >= 60)), 10.0, 0.0)
    pr_score = pr_score + np.where(pr_base & (rel <= 12), 10.0, 0.0)
    pr_score = pd.Series(np.minimum(pr_score, 75.0), index=df.index)

    primeiro_receiver_valor_anomalo = pr_base & (pr_score >= 55.0)

    df["exp004_pr_valor_anomalo_com_historico"] = valor_anomalo_cliente_com_historico.astype(bool)
    df["exp004_pr_valor_anomalo_sem_historico"] = valor_anomalo_cliente_sem_historico.astype(bool)
    df["exp004_pr_convergencia"] = pr_convergencia.astype(bool)
    df["exp004_primeiro_receiver_score"] = pr_score.round(2)
    df[RULE_COLUMNS["PRIMEIRO_RECEIVER_VALOR_ANOMALO"]] = primeiro_receiver_valor_anomalo.astype(bool)

    return df


def _apply_policy_variant(baseline_preds: pd.DataFrame, variant_cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Aplica uma variante post-hoc.

    Regra operacional:
      - Nunca rebaixa BLOQUEAR/CONFIRMAR para APROVAR.
      - Nunca transforma CONFIRMAR em BLOQUEAR.
      - As regras novas so podem elevar APROVAR -> CONFIRMAR.
    """
    df = _add_rule_columns(baseline_preds)
    df = df.copy()

    original_decisao = df["decisao"].astype(str).copy()
    original_flagged = original_decisao.isin(["CONFIRMAR", "BLOQUEAR"])

    selected_masks: dict[str, pd.Series] = {}

    if variant_cfg.get("use_guard_exception", False):
        selected_masks["GUARD_EXCEPTION_ALTO_VALOR_SE_BEH"] = df[RULE_COLUMNS["GUARD_EXCEPTION_ALTO_VALOR_SE_BEH"]].astype(bool)

    if variant_cfg.get("use_rate_limit", False):
        selected_masks["RATE_LIMIT_ANOMALO"] = df[RULE_COLUMNS["RATE_LIMIT_ANOMALO"]].astype(bool)

    if variant_cfg.get("use_primeiro_receiver", False):
        selected_masks["PRIMEIRO_RECEIVER_VALOR_ANOMALO"] = df[RULE_COLUMNS["PRIMEIRO_RECEIVER_VALOR_ANOMALO"]].astype(bool)

    trigger = pd.Series(False, index=df.index)
    reason = pd.Series("", index=df.index, dtype="object")

    for rule_name, mask in selected_masks.items():
        mask = mask.astype(bool)
        trigger = trigger | mask
        reason = reason.where(~mask, np.where(reason.eq(""), rule_name, reason + "|" + rule_name))

    upgrade = trigger & (~original_flagged)

    df["decisao_original"] = original_decisao
    df["exp004_variant_id"] = variant_cfg["id"]
    df["exp004_policy_triggered"] = trigger.astype(bool)
    df["exp004_policy_upgrade"] = upgrade.astype(bool)
    df["exp004_policy_reason"] = reason

    df.loc[upgrade, "decisao"] = "CONFIRMAR"

    if "veto_reason" not in df.columns:
        df["veto_reason"] = ""

    df.loc[upgrade, "veto_reason"] = (
        "EXP004-FINAL CONFIRMAR: " + df.loc[upgrade, "exp004_policy_reason"].astype(str)
    )

    return df


# =========================================================
# METRICAS, COMPARACOES E ANALISES
# =========================================================

def _evaluate_predictions(preds: pd.DataFrame, label: str, variant_id: str) -> dict[str, Any]:
    y_true = preds["is_fraud"].astype(int).values
    y_pred = _flagged(preds).astype(int).values
    metrics = compute_metrics(y_true, y_pred, label)

    policy_triggered = preds.get(
        "exp004_policy_triggered",
        pd.Series(False, index=preds.index),
    ).astype(bool)

    policy_upgrade = preds.get(
        "exp004_policy_upgrade",
        pd.Series(False, index=preds.index),
    ).astype(bool)

    y_true_series = preds["is_fraud"].astype(int)

    hits = int(policy_triggered.sum())
    hits_fraud = int((policy_triggered & y_true_series.eq(1)).sum())
    upgrades = int(policy_upgrade.sum())
    upgrades_fraud = int((policy_upgrade & y_true_series.eq(1)).sum())
    upgrades_legit = int((policy_upgrade & y_true_series.eq(0)).sum())

    valor_fn_recuperado = float(_num_series(preds, "vl_pix", 0).loc[policy_upgrade & y_true_series.eq(1)].sum())
    valor_fp_adicionado = float(_num_series(preds, "vl_pix", 0).loc[policy_upgrade & y_true_series.eq(0)].sum())

    row = {
        "variante_id": variant_id,
        "label": label,
        **metrics.to_dict(),
        "policy_hits": hits,
        "policy_hits_fraud": hits_fraud,
        "policy_precision": round(hits_fraud / max(hits, 1), 6),
        "upgrades_aprovar_para_confirmar": upgrades,
        "upgrades_fraude": upgrades_fraud,
        "upgrades_legitima": upgrades_legit,
        "valor_fn_recuperado": round(valor_fn_recuperado, 2),
        "valor_fp_adicionado": round(valor_fp_adicionado, 2),
    }
    return row


def _top_records(df: pd.DataFrame, mask: pd.Series, n: int = 15) -> list[dict[str, Any]]:
    if mask.sum() == 0:
        return []

    cols = [
        "transaction_id",
        "customer_id",
        "receiver_id",
        "event_datetime",
        "is_fraud",
        "vl_pix",
        "nr_idade",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "burst_30m_flag",
        "lgbm_raw",
        "lgbm_mapped",
        "if_percentile",
        "if_raw",
        "se_score",
        "beh_score",
        "se_worst_pattern",
        "score_final",
        "decisao_original",
        "decisao",
        "veto_reason",
        "veto_suppressed_reason",
        "exp004_policy_reason",
        "exp004_rate_score",
        "exp004_primeiro_receiver_score",
        "exp004_tx_count_30m",
        "exp004_sum_vl_30m",
        "exp004_tx_count_60m",
        "exp004_sum_vl_60m",
        "exp004_distinct_receivers_30m",
        "exp004_hist_count_90d",
        "exp004_hist_avg_90d",
    ]
    cols = _select_existing_cols(df, cols)

    out = df.loc[mask, cols].copy()
    if "vl_pix" in out.columns:
        out = out.sort_values("vl_pix", ascending=False)

    return out.head(n).to_dict(orient="records")


def _compare_to_baseline(baseline: pd.DataFrame, variant: pd.DataFrame) -> dict[str, Any]:
    baseline_flagged = _flagged(baseline)
    variant_flagged = _flagged(variant)
    y_true = baseline["is_fraud"].astype(int)

    removed_fp_mask = y_true.eq(0) & baseline_flagged & (~variant_flagged)
    lost_tp_mask = y_true.eq(1) & baseline_flagged & (~variant_flagged)
    recovered_fn_mask = y_true.eq(1) & (~baseline_flagged) & variant_flagged
    added_fp_mask = y_true.eq(0) & (~baseline_flagged) & variant_flagged

    return {
        "fps_removidos": {
            "total": int(removed_fp_mask.sum()),
            "top_por_valor": _top_records(baseline, removed_fp_mask),
        },
        "tps_perdidos": {
            "total": int(lost_tp_mask.sum()),
            "top_por_valor": _top_records(baseline, lost_tp_mask),
        },
        "fns_recuperados": {
            "total": int(recovered_fn_mask.sum()),
            "valor_total": round(float(_num_series(variant, "vl_pix", 0).loc[recovered_fn_mask].sum()), 2),
            "top_por_valor": _top_records(variant, recovered_fn_mask),
        },
        "fps_adicionados": {
            "total": int(added_fp_mask.sum()),
            "valor_total": round(float(_num_series(variant, "vl_pix", 0).loc[added_fp_mask].sum()), 2),
            "top_por_valor": _top_records(variant, added_fp_mask),
        },
    }


def _rule_stats(preds: pd.DataFrame) -> dict[str, Any]:
    y_true = preds["is_fraud"].astype(int)
    original_flagged = preds["decisao_original"].astype(str).isin(["CONFIRMAR", "BLOQUEAR"]) if "decisao_original" in preds.columns else _flagged(preds)

    stats: dict[str, Any] = {}

    for rule_name, col in RULE_COLUMNS.items():
        if col not in preds.columns:
            continue

        mask = preds[col].astype(bool)
        fraud = mask & y_true.eq(1)
        legit = mask & y_true.eq(0)
        recovered = mask & y_true.eq(1) & (~original_flagged)
        added_fp = mask & y_true.eq(0) & (~original_flagged)

        stats[rule_name] = {
            "total_hits": int(mask.sum()),
            "fraud_hits": int(fraud.sum()),
            "legit_hits": int(legit.sum()),
            "precision_hits": round(int(fraud.sum()) / max(int(mask.sum()), 1), 6),
            "hits_ja_flagged_baseline": int((mask & original_flagged).sum()),
            "fns_recuperaveis": int(recovered.sum()),
            "fps_potenciais_adicionados": int(added_fp.sum()),
            "top_hits_por_valor": _top_records(preds, mask, n=10),
        }

    return stats


def _add_deltas_and_utility(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.copy()
    baseline = df.loc[df["variante_id"] == "BASELINE"].iloc[0]

    df["delta_TP"] = df["TP"] - baseline["TP"]
    df["delta_FP"] = df["FP"] - baseline["FP"]
    df["delta_FN"] = df["FN"] - baseline["FN"]
    df["delta_F1"] = (df["F1"] - baseline["F1"]).round(6)
    df["delta_Recall"] = (df["Recall"] - baseline["Recall"]).round(6)
    df["delta_Precision"] = (df["Precision"] - baseline["Precision"]).round(6)

    # Utilidade antifraude simples:
    #   +10 por FN recuperado / TP adicional
    #   -1 por FP novo
    #   +2 por FP removido, se algum dia houver regra que reduza FP
    #   -15 por TP perdido
    df["utility"] = (
        10 * df["delta_TP"].clip(lower=0)
        - 1 * df["delta_FP"].clip(lower=0)
        + 2 * (-df["delta_FP"]).clip(lower=0)
        - 15 * (-df["delta_TP"]).clip(lower=0)
    ).astype(float)

    return df


def _pick_winner(
    results_df: pd.DataFrame,
    fp_max: int,
    precision_min: float,
    recall_min: float,
    fpr_max: float,
    require_f1_non_decreasing: bool = True,
) -> str:
    baseline = results_df.loc[results_df["variante_id"] == "BASELINE"].iloc[0]
    candidates = results_df[results_df["variante_id"] != "BASELINE"].copy()

    if candidates.empty:
        return "BASELINE"

    hard_mask = (
        (candidates["TP"] >= baseline["TP"])
        & (candidates["FP"] <= fp_max)
        & (candidates["Precision"] >= precision_min)
        & (candidates["Recall"] >= recall_min)
        & (candidates["FPR"] <= fpr_max)
        & (candidates["delta_TP"] >= 1)
    )

    if require_f1_non_decreasing:
        hard_mask = hard_mask & (candidates["F1"] >= baseline["F1"])

    eligible = candidates[hard_mask].copy()

    if eligible.empty:
        # Fallback: melhoria local que nao passe recall_min agressivo.
        fallback = candidates[
            (candidates["TP"] >= baseline["TP"])
            & (candidates["FP"] <= fp_max)
            & (candidates["Precision"] >= precision_min)
            & (candidates["FPR"] <= fpr_max)
            & (candidates["delta_TP"] >= 1)
        ].copy()

        if fallback.empty:
            return "BASELINE"

        eligible = fallback

    eligible = eligible.sort_values(
        ["utility", "TP", "F1", "FP"],
        ascending=[False, False, False, True],
    )

    return str(eligible.iloc[0]["variante_id"])


# =========================================================
# CLASSIFICACAO DOS FNs RESIDUAIS
# =========================================================

def _classify_one_fn(row: pd.Series) -> tuple[str, str]:
    lgbm = _safe_float(row.get("lgbm_raw"))
    ifp = _safe_float(row.get("if_percentile"))
    se = _safe_float(row.get("se_score"))
    beh = _safe_float(row.get("beh_score"))
    vl = _safe_float(row.get("vl_pix"))
    idade = _safe_float(row.get("nr_idade"))
    first_receiver = _safe_int(row.get("first_receiver_flag")) == 1
    pix_random = _safe_int(row.get("pix_key_random_flag")) == 1
    rate_score = _safe_float(row.get("exp004_rate_score"))
    pr_score = _safe_float(row.get("exp004_primeiro_receiver_score"))
    suppressed = str(row.get("veto_suppressed_reason", "") or "").strip()

    if idade <= 0 or pd.isna(row.get("customer_id", np.nan)):
        return "DATA_QUALITY_SUSPECT", "idade/customer_id ausente ou suspeito"

    if suppressed:
        return "GUARD_SUPPRESSED_CANDIDATE", "ha veto suprimido que merece revisao manual"

    if rate_score >= 20:
        return "RATE_LIMIT_CANDIDATE", "sinal temporal parcial, mas abaixo do limiar de confirmacao"

    if pr_score > 0 or (first_receiver and (pix_random or ifp >= 0.80 or lgbm >= 0.10)):
        return "PRIMEIRO_RECEIVER_CANDIDATE", "primeiro recebedor com algum sinal auxiliar, mas insuficiente na regra atual"

    if lgbm >= 0.10 or ifp >= 0.80:
        return "LGBM_BORDERLINE", "LGBM ou IF ainda mostram sinal fraco/moderado"

    if vl <= 500 and lgbm < 0.10 and ifp < 0.80 and se == 0 and beh == 0:
        return "LOW_VALUE_ECONOMICALLY_TOLERABLE", "baixo valor e nenhum sinal forte nas features atuais"

    if lgbm < 0.10 and ifp < 0.80 and se == 0 and beh == 0 and rate_score == 0:
        return "IRREDUTIVEL_FEATURES_ATUAIS", "sem sinal discriminante nas features atuais"

    return "RESIDUAL_REVIEW", "caso residual requer inspecao"


def _classify_residual_fns(preds: pd.DataFrame) -> pd.DataFrame:
    y_true = preds["is_fraud"].astype(int)
    fn_mask = y_true.eq(1) & (~_flagged(preds))

    if fn_mask.sum() == 0:
        return pd.DataFrame(columns=[
            "categoria_residual",
            "motivo_classificacao",
            "transaction_id",
            "customer_id",
            "vl_pix",
        ])

    fn_df = preds.loc[fn_mask].copy()
    classifications = fn_df.apply(_classify_one_fn, axis=1)
    fn_df["categoria_residual"] = [x[0] for x in classifications]
    fn_df["motivo_classificacao"] = [x[1] for x in classifications]

    cols = [
        "categoria_residual",
        "motivo_classificacao",
        "transaction_id",
        "customer_id",
        "receiver_id",
        "event_datetime",
        "vl_pix",
        "nr_idade",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "burst_30m_flag",
        "lgbm_raw",
        "lgbm_mapped",
        "if_percentile",
        "if_raw",
        "se_score",
        "beh_score",
        "se_worst_pattern",
        "score_final",
        "decisao",
        "veto_reason",
        "veto_suppressed_reason",
        "exp004_policy_reason",
        "exp004_rate_score",
        "exp004_primeiro_receiver_score",
        "exp004_tx_count_30m",
        "exp004_sum_vl_30m",
        "exp004_tx_count_60m",
        "exp004_sum_vl_60m",
        "exp004_distinct_receivers_30m",
        "exp004_hist_count_90d",
        "exp004_hist_avg_90d",
    ]
    cols = _select_existing_cols(fn_df, cols)

    return fn_df[cols].sort_values(["vl_pix"], ascending=False)


# =========================================================
# EXECUCAO POR SEED
# =========================================================

def _run_seed(
    sample_df: pd.DataFrame,
    context: pd.DataFrame,
    workers: int,
    seed_label: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    """
    Processa o sample uma vez no baseline e aplica variantes post-hoc.
    """
    logger.info("[%s] Processando baseline via PipelineOrquestrador real...", seed_label)

    baseline_raw = process_dataframe_via_orquestrador(
        sample_df,
        workers=workers,
        logger=logger,
        engine_config_overrides=BASELINE_ENGINE_OVERRIDES,
    )

    baseline = _merge_context(baseline_raw, context)
    baseline = _add_rule_columns(baseline)
    baseline["decisao_original"] = baseline["decisao"].astype(str)
    baseline["exp004_variant_id"] = "BASELINE"
    baseline["exp004_policy_triggered"] = False
    baseline["exp004_policy_upgrade"] = False
    baseline["exp004_policy_reason"] = ""

    predictions: dict[str, pd.DataFrame] = {"BASELINE": baseline}

    rows: list[dict[str, Any]] = [
        _evaluate_predictions(
            baseline,
            label="Baseline atual EXP-001 + EXP-002",
            variant_id="BASELINE",
        )
    ]

    comparisons: dict[str, Any] = {}
    rule_summaries: dict[str, Any] = {}

    for variant_cfg in VARIANTS:
        variant_id = variant_cfg["id"]
        logger.info("[%s] Aplicando politica post-hoc: %s", seed_label, variant_id)

        variant_preds = _apply_policy_variant(baseline, variant_cfg)
        predictions[variant_id] = variant_preds

        rows.append(_evaluate_predictions(
            variant_preds,
            label=variant_cfg["label"],
            variant_id=variant_id,
        ))

        comparisons[variant_id] = _compare_to_baseline(baseline, variant_preds)
        rule_summaries[variant_id] = _rule_stats(variant_preds)

    results_df = _add_deltas_and_utility(pd.DataFrame(rows))
    return results_df, predictions, comparisons, rule_summaries


# =========================================================
# RELATORIOS
# =========================================================

def _write_conclusion(
    path: Path,
    results_df: pd.DataFrame,
    winner_id: str,
    validation_payload: dict[str, Any] | None,
    residual_df: pd.DataFrame,
    fp_max: int,
    precision_min: float,
    recall_min: float,
    fpr_max: float,
) -> None:
    baseline = results_df.loc[results_df["variante_id"] == "BASELINE"].iloc[0]
    winner = results_df.loc[results_df["variante_id"] == winner_id].iloc[0]

    val_lines: list[str] = []
    if validation_payload is not None:
        val_metrics = validation_payload.get("winner_metrics", {})
        val_lines = [
            "",
            "## Validacao cruzada",
            "",
            f"- Seed: `{validation_payload.get('seed')}`",
            f"- Vencedor validado: `{validation_payload.get('winner_id')}`",
            f"- TP={val_metrics.get('TP')}, FP={val_metrics.get('FP')}, FN={val_metrics.get('FN')}",
            f"- Precision={_safe_float(val_metrics.get('Precision')):.4%}",
            f"- Recall={_safe_float(val_metrics.get('Recall')):.4%}",
            f"- F1={_safe_float(val_metrics.get('F1')):.4f}",
            f"- FPR={_safe_float(val_metrics.get('FPR')):.4%}",
        ]
    else:
        val_lines = [
            "",
            "## Validacao cruzada",
            "",
            "- Pulada por `--skip-validation`.",
        ]

    fn_cats = {}
    if not residual_df.empty and "categoria_residual" in residual_df.columns:
        fn_cats = residual_df["categoria_residual"].value_counts().to_dict()

    acceptance = {
        "fp_max": int(fp_max),
        "precision_min": float(precision_min),
        "recall_min": float(recall_min),
        "fpr_max": float(fpr_max),
        "winner_fp_ok": bool(winner["FP"] <= fp_max),
        "winner_precision_ok": bool(winner["Precision"] >= precision_min),
        "winner_recall_ok": bool(winner["Recall"] >= recall_min),
        "winner_fpr_ok": bool(winner["FPR"] <= fpr_max),
        "winner_f1_non_decreasing": bool(winner["F1"] >= baseline["F1"]),
        "winner_tp_non_decreasing": bool(winner["TP"] >= baseline["TP"]),
    }
    approved = all(acceptance.values()) and winner_id != "BASELINE"

    if approved:
        recommendation = (
            f"Promover `{winner_id}` para patch de runtime, implementando a regra vencedora "
            "no modulo apropriado e mantendo modo shadow por uma janela operacional."
        )
    elif winner_id != "BASELINE":
        recommendation = (
            f"`{winner_id}` teve melhoria local, mas nao passou todos os criterios fortes. "
            "Considerar deploy parcial ou ajustar thresholds antes de promover."
        )
    else:
        recommendation = (
            "Manter baseline EXP-001+EXP-002 e encerrar FASE 1 com classificacao dos FNs "
            "residuais. Proximos ganhos provavelmente dependem de novas features ou FASE 2."
        )

    lines = [
        f"# {EXP_ID} - Conclusao da FASE 1",
        "",
        f"- Vencedor: `{winner_id}`",
        f"- Status: `{'APROVADO' if approved else 'NAO_APROVADO_AUTOMATICAMENTE'}`",
        "",
        "## Resultado principal",
        "",
        f"- Baseline: TP={int(baseline['TP'])}, FP={int(baseline['FP'])}, FN={int(baseline['FN'])}, "
        f"Precision={baseline['Precision']:.4%}, Recall={baseline['Recall']:.4%}, F1={baseline['F1']:.4f}, FPR={baseline['FPR']:.4%}",
        f"- Vencedor: TP={int(winner['TP'])}, FP={int(winner['FP'])}, FN={int(winner['FN'])}, "
        f"Precision={winner['Precision']:.4%}, Recall={winner['Recall']:.4%}, F1={winner['F1']:.4f}, FPR={winner['FPR']:.4%}",
        f"- Delta: TP={int(winner['delta_TP']):+d}, FP={int(winner['delta_FP']):+d}, "
        f"FN={int(winner['delta_FN']):+d}, F1={winner['delta_F1']:+.4f}",
        f"- Utility antifraude: {winner['utility']:.2f}",
        f"- Valor de FN recuperado: R$ {winner['valor_fn_recuperado']:.2f}",
        f"- Valor de FP adicionado: R$ {winner['valor_fp_adicionado']:.2f}",
        "",
        "## Criterios de aceite",
        "",
        f"- FP <= {fp_max}: `{acceptance['winner_fp_ok']}`",
        f"- Precision >= {precision_min:.2%}: `{acceptance['winner_precision_ok']}`",
        f"- Recall >= {recall_min:.2%}: `{acceptance['winner_recall_ok']}`",
        f"- FPR <= {fpr_max:.2%}: `{acceptance['winner_fpr_ok']}`",
        f"- F1 nao decrescente: `{acceptance['winner_f1_non_decreasing']}`",
        f"- TP nao decrescente: `{acceptance['winner_tp_non_decreasing']}`",
        *val_lines,
        "",
        "## FNs residuais",
        "",
        f"- Total de FNs residuais no vencedor: `{int(winner['FN'])}`",
        f"- Categorias: `{json.dumps(fn_cats, ensure_ascii=False)}`",
        "",
        "## Recomendacao",
        "",
        recommendation,
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_meta_shadow_report(
    path: Path,
    results_df: pd.DataFrame,
    comparisons: dict[str, Any],
    rule_summaries: dict[str, Any],
    residual_df: pd.DataFrame,
) -> None:
    baseline = results_df.loc[results_df["variante_id"] == "BASELINE"].iloc[0]

    lines = [
        f"# {EXP_ID} - Meta Shadow Report",
        "",
        "Este arquivo nao treina um meta-learner. Ele resume os sinais shadow que podem alimentar a FASE 2.",
        "",
        "## Baseline",
        "",
        f"- TP={int(baseline['TP'])}, FP={int(baseline['FP'])}, FN={int(baseline['FN'])}, F1={baseline['F1']:.4f}",
        "",
        "## Variantes e cobertura",
        "",
    ]

    for _, row in results_df.iterrows():
        vid = row["variante_id"]
        lines.extend([
            f"### {vid}",
            "",
            f"- TP={int(row['TP'])}, FP={int(row['FP'])}, FN={int(row['FN'])}, F1={row['F1']:.4f}",
            f"- Delta TP={int(row['delta_TP']):+d}, Delta FP={int(row['delta_FP']):+d}, Delta FN={int(row['delta_FN']):+d}",
            f"- Policy hits={int(row['policy_hits'])}, hits fraude={int(row['policy_hits_fraud'])}, precision hits={row['policy_precision']:.4f}",
            f"- Upgrades={int(row['upgrades_aprovar_para_confirmar'])}, upgrades fraude={int(row['upgrades_fraude'])}, upgrades legitima={int(row['upgrades_legitima'])}",
            "",
        ])

    lines.extend([
        "## Rule stats por variante",
        "",
        "Resumo completo esta em `02_delta_fp_fn_por_variante.json`.",
        "",
    ])

    for vid, summary in rule_summaries.items():
        lines.append(f"### {vid}")
        lines.append("")
        for rule_name, stats in summary.items():
            lines.append(
                f"- `{rule_name}`: hits={stats.get('total_hits')}, "
                f"fraud_hits={stats.get('fraud_hits')}, "
                f"precision={_safe_float(stats.get('precision_hits')):.4f}, "
                f"fns_recuperaveis={stats.get('fns_recuperaveis')}, "
                f"fps_potenciais={stats.get('fps_potenciais_adicionados')}"
            )
        lines.append("")

    if residual_df.empty:
        lines.extend([
            "## FNs residuais",
            "",
            "Nenhum FN residual no vencedor.",
            "",
        ])
    else:
        lines.extend([
            "## FNs residuais classificados",
            "",
        ])
        cat_counts = residual_df["categoria_residual"].value_counts().to_dict() if "categoria_residual" in residual_df else {}
        for cat, count in cat_counts.items():
            lines.append(f"- `{cat}`: {count}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=f"{EXP_ID} - {EXP_TITLE}")
    parser.add_argument("--sample", type=int, default=6000, help="Tamanho do sample estratificado.")
    parser.add_argument("--workers", type=int, default=1, help="Workers para PipelineOrquestrador.")
    parser.add_argument("--seed", type=int, default=42, help="Seed do sample principal.")
    parser.add_argument("--validation-seed", type=int, default=123, help="Seed da validacao cruzada.")
    parser.add_argument("--skip-validation", action="store_true", help="Pula validacao cruzada.")
    parser.add_argument("--save-predictions", action="store_true", help="Salva CSVs de baseline e vencedor.")
    parser.add_argument("--fp-max", type=int, default=20, help="Maximo absoluto de FP permitido no sample principal.")
    parser.add_argument("--precision-min", type=float, default=0.94, help="Precision minima para aprovacao.")
    parser.add_argument("--recall-min", type=float, default=0.9831, help="Recall minimo forte para aprovacao.")
    parser.add_argument("--fpr-max", type=float, default=0.005, help="FPR maximo para aprovacao.")
    args = parser.parse_args()

    t0 = time.perf_counter()

    print_section(f"{EXP_ID} - {EXP_TITLE}")

    output_dir = get_experiment_output_dir(EXP_ID)
    logger.info("Output dir: %s", output_dir)
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Sample size: %s | Seed principal: %s | Workers: %s", args.sample, args.seed, args.workers)

    # -----------------------------------------------------
    # 1. Carregar dataset completo e contexto temporal
    # -----------------------------------------------------
    print_section("1. Carregar dataset e pre-computar contexto temporal")
    df_full = load_dataset()
    logger.info("Dataset completo: %d tx | fraudes=%d", len(df_full), int(df_full["is_fraud"].sum()))

    context = _compute_temporal_context(df_full)

    # -----------------------------------------------------
    # 2. Sample principal
    # -----------------------------------------------------
    print_section("2. Gerar sample principal e processar baseline real")
    sample_df = stratified_sample(df_full, n=args.sample, seed=args.seed, logger=logger)

    results_df, predictions, comparisons, rule_summaries = _run_seed(
        sample_df=sample_df,
        context=context,
        workers=args.workers,
        seed_label=f"seed={args.seed}",
    )

    winner_id = _pick_winner(
        results_df=results_df,
        fp_max=args.fp_max,
        precision_min=args.precision_min,
        recall_min=args.recall_min,
        fpr_max=args.fpr_max,
        require_f1_non_decreasing=True,
    )

    winner_preds = predictions[winner_id]
    residual_df = _classify_residual_fns(winner_preds)

    # -----------------------------------------------------
    # 3. Salvar artefatos principais
    # -----------------------------------------------------
    print_section("3. Salvar artefatos principais")
    results_path = output_dir / "01_tabela_comparativa.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    delta_payload = {
        "experiment_id": EXP_ID,
        "baseline_engine_overrides": BASELINE_ENGINE_OVERRIDES,
        "winner_id": winner_id,
        "comparisons": comparisons,
        "rule_summaries": rule_summaries,
        "results": results_df.to_dict(orient="records"),
    }
    safe_json_dump(delta_payload, output_dir / "02_delta_fp_fn_por_variante.json")

    residual_df.to_csv(
        output_dir / "03_fn_residuais_classificados.csv",
        index=False,
        encoding="utf-8-sig",
    )

    logger.info("Tabela comparativa salva: %s", results_path)
    logger.info("Vencedor preliminar: %s", winner_id)

    # -----------------------------------------------------
    # 4. Validacao cruzada
    # -----------------------------------------------------
    validation_payload: dict[str, Any] | None = None

    if not args.skip_validation:
        print_section("4. Validacao cruzada em seed independente")
        val_sample = stratified_sample(df_full, n=args.sample, seed=args.validation_seed, logger=logger)

        val_results_df, val_predictions, val_comparisons, val_rule_summaries = _run_seed(
            sample_df=val_sample,
            context=context,
            workers=args.workers,
            seed_label=f"validation_seed={args.validation_seed}",
        )

        if winner_id in val_predictions:
            winner_val_preds = val_predictions[winner_id]
            winner_val_row = val_results_df.loc[val_results_df["variante_id"] == winner_id].iloc[0].to_dict()
        else:
            winner_val_preds = val_predictions["BASELINE"]
            winner_val_row = val_results_df.loc[val_results_df["variante_id"] == "BASELINE"].iloc[0].to_dict()

        validation_payload = {
            "winner_id": winner_id,
            "seed": args.validation_seed,
            "sample": args.sample,
            "winner_metrics": winner_val_row,
            "all_metrics": val_results_df.to_dict(orient="records"),
            "winner_comparison_vs_validation_baseline": val_comparisons.get(winner_id, {}),
            "winner_rule_summary": val_rule_summaries.get(winner_id, {}),
            "fn_residuais_vencedor": _classify_residual_fns(winner_val_preds).to_dict(orient="records"),
        }

        safe_json_dump(validation_payload, output_dir / "04_validacao_cruzada.json")
    else:
        validation_payload = None
        safe_json_dump(
            {"skipped": True, "reason": "--skip-validation"},
            output_dir / "04_validacao_cruzada.json",
        )

    # -----------------------------------------------------
    # 5. Relatorios executivos
    # -----------------------------------------------------
    print_section("5. Gerar conclusao e meta shadow report")

    _write_conclusion(
        path=output_dir / "05_conclusao_fase_1.md",
        results_df=results_df,
        winner_id=winner_id,
        validation_payload=validation_payload,
        residual_df=residual_df,
        fp_max=args.fp_max,
        precision_min=args.precision_min,
        recall_min=args.recall_min,
        fpr_max=args.fpr_max,
    )

    _write_meta_shadow_report(
        path=output_dir / "06_meta_shadow_report.md",
        results_df=results_df,
        comparisons=comparisons,
        rule_summaries=rule_summaries,
        residual_df=residual_df,
    )

    # -----------------------------------------------------
    # 6. Opcional: salvar predicoes
    # -----------------------------------------------------
    if args.save_predictions:
        print_section("6. Salvar predicoes detalhadas")
        predictions["BASELINE"].to_csv(
            output_dir / "00_predictions_baseline.csv",
            index=False,
            encoding="utf-8-sig",
        )
        winner_preds.to_csv(
            output_dir / f"00_predictions_{winner_id}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # -----------------------------------------------------
    # 7. Log final
    # -----------------------------------------------------
    baseline = results_df.loc[results_df["variante_id"] == "BASELINE"].iloc[0]
    winner = results_df.loc[results_df["variante_id"] == winner_id].iloc[0]

    logger.info("============================================================")
    logger.info("EXP-004-FINAL concluido")
    logger.info("Vencedor: %s", winner_id)
    logger.info(
        "Baseline: TP=%d FP=%d FN=%d F1=%.4f Recall=%.4f Precision=%.4f",
        int(baseline["TP"]),
        int(baseline["FP"]),
        int(baseline["FN"]),
        float(baseline["F1"]),
        float(baseline["Recall"]),
        float(baseline["Precision"]),
    )
    logger.info(
        "Vencedor: TP=%d FP=%d FN=%d F1=%.4f Recall=%.4f Precision=%.4f",
        int(winner["TP"]),
        int(winner["FP"]),
        int(winner["FN"]),
        float(winner["F1"]),
        float(winner["Recall"]),
        float(winner["Precision"]),
    )
    logger.info(
        "Delta: TP=%+d FP=%+d FN=%+d F1=%+.4f",
        int(winner["delta_TP"]),
        int(winner["delta_FP"]),
        int(winner["delta_FN"]),
        float(winner["delta_F1"]),
    )
    logger.info("Artefatos salvos em: %s", output_dir)
    logger.info("Tempo total: %.1fs", time.perf_counter() - t0)
    logger.info("============================================================")


if __name__ == "__main__":
    main()