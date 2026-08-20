"""
Políticas de severidade pós-decisão para o motor PIX.

Este módulo mantém regras explícitas e ordenadas para rebaixar casos de
`BLOQUEAR` para `CONFIRMAR` quando há evidência offline congelada de baixo risco.
As regras não alteram o classificador principal nem promovem `APROVAR`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


R5B8_POLICY_ID = "EXP-014B-R5B8-BROAD-RESIDUAL-RULE-MINING"
R5B8_RULE_SET_VERSION = "2026-06-11"
R5B14_POLICY_ID = "EXP-014B-R5B14-OPERATIONAL-ZERO-FN-REPLAY"
R5B14_RULE_SET_VERSION = "2026-06-12"
R5B16_POLICY_ID = "EXP-014B-R5B16-CONSOLIDATED-OPERATIONAL-BASELINE"
R5B16_FROZEN_ACTION_COL = "r4g_fast_frozen_decisao_recommended"
R5B14_COMPENSATION_LGBM_RAW_MAX = 0.00001966


@dataclass(frozen=True)
class SeverityRule:
    """Regra sequencial de de-escalonamento de severidade."""

    rule_id: str
    description: str
    target_action: str = "CONFIRMAR"


R5B8_BLOCK_TO_CONFIRM_RULES: tuple[SeverityRule, ...] = (
    SeverityRule(
        rule_id="R5B8_01_RELATIONSHIP_AGE_GTE_35D",
        description="dias_desde_primeiro_envio_recebedor >= 35",
    ),
    SeverityRule(
        rule_id="R5B8_02_LOW_RECEIVER_REP_LOW_PAYER_COUNT",
        description=(
            "receiver_reputation_score > 70.57506297 AND "
            "qtd_pix_pagador_180d > 175"
        ),
    ),
    SeverityRule(
        rule_id="R5B8_03_LOW_RECEIVER_VALUE_LOW_PAYER_VALUE",
        description=(
            "valor_rec_bin == val_rec_lt_5k AND "
            "valor_total_pagador_180d > 212416.178"
        ),
    ),
)


R5B14_CONFIRM_TO_BLOCK_RULES: tuple[SeverityRule, ...] = (
    SeverityRule(
        rule_id="R5B14_CTB_01_LGBM_RAW_HIGH",
        description="lgbm_raw >= 0.10711783",
        target_action="BLOQUEAR",
    ),
    SeverityRule(
        rule_id="R5B14_CTB_02_SCORE_2_3_LGBM_R4_HIGH",
        description="score_bin == score_2_3 AND lgbm_r4_score >= 0.475472966916",
        target_action="BLOQUEAR",
    ),
    SeverityRule(
        rule_id="R5B14_CTB_03_SCORE_2_3_LGBM_R4_MED",
        description="score_bin == score_2_3 AND lgbm_r4_score >= 0.318070929491",
        target_action="BLOQUEAR",
    ),
    SeverityRule(
        rule_id="R5B14_CTB_04_DOC_PHONE_HIGH_PAYER_COUNT",
        description="ds_tipo_chave_norm == DOCUMENTO_TELEFONE AND qtd_pix_pagador_180d >= 207",
        target_action="BLOQUEAR",
    ),
    SeverityRule(
        rule_id="R5B14_CTB_05_OUTROS_RATIO_MAX_HIGH",
        description=(
            "ds_tipo_chave_norm == OUTROS AND "
            "ratio_valor_maximo_pagador_180d >= 4.9674631165863596"
        ),
        target_action="BLOQUEAR",
    ),
)


R5B14_APPROVE_TO_BLOCK_RULES: tuple[SeverityRule, ...] = (
    SeverityRule(
        rule_id="R5B14_ATB_01_DOC_PHONE_MORNING_SCORE_HIGH",
        description=(
            "ds_tipo_chave_norm == DOCUMENTO_TELEFONE AND "
            "periodo_dia == manha AND "
            "score_bin == score_GE_10 AND "
            "lgbm_bin == lgbm_GE_0.1"
        ),
        target_action="BLOQUEAR",
    ),
    SeverityRule(
        rule_id="R5B14_ATB_02_NIGHT_SCORE_1_2_RATIO_HIGH",
        description=(
            "periodo_dia == noite AND "
            "score_bin == score_1_2 AND "
            "lgbm_bin == lgbm_GE_0.1 AND "
            "ratio_bin == ratio_GE_5"
        ),
        target_action="BLOQUEAR",
    ),
)


R5B14_CONFIRM_TO_APPROVE_RULES: tuple[SeverityRule, ...] = (
    SeverityRule(
        rule_id="R5B14_CTA_01_LOW_LGBM_RAW_COMPENSATION",
        description=f"lgbm_raw <= {R5B14_COMPENSATION_LGBM_RAW_MAX}",
        target_action="APROVAR",
    ),
)


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _str(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype=str)
    return df[col].fillna("").astype(str)


def _score_bin(df: pd.DataFrame) -> pd.Series:
    existing = _str(df, "score_bin").str.strip()
    if "score_final" not in df.columns:
        return existing

    score = _num(df, "score_final")
    derived = pd.Series("", index=df.index, dtype=object)
    derived.loc[score < 1.0] = "score_0_1"
    derived.loc[(score >= 1.0) & (score < 2.0)] = "score_1_2"
    derived.loc[(score >= 2.0) & (score < 3.0)] = "score_2_3"
    derived.loc[(score >= 3.0) & (score < 5.0)] = "score_3_5"
    derived.loc[(score >= 5.0) & (score < 10.0)] = "score_5_10"
    derived.loc[score >= 10.0] = "score_GE_10"
    return existing.where(existing.ne(""), derived)


def _lgbm_bin(df: pd.DataFrame) -> pd.Series:
    existing = _str(df, "lgbm_bin").str.strip()
    if "lgbm_raw" not in df.columns:
        return existing
    derived = pd.Series("", index=df.index, dtype=object)
    derived.loc[_num(df, "lgbm_raw") >= 0.1] = "lgbm_GE_0.1"
    return existing.where(existing.ne(""), derived)


def _ratio_bin(df: pd.DataFrame) -> pd.Series:
    existing = _str(df, "ratio_bin").str.strip()
    if "ratio_valor_maximo_pagador_180d" not in df.columns:
        return existing
    derived = pd.Series("", index=df.index, dtype=object)
    derived.loc[_num(df, "ratio_valor_maximo_pagador_180d") >= 5.0] = "ratio_GE_5"
    return existing.where(existing.ne(""), derived)


def normalize_action(actions: pd.Series) -> pd.Series:
    return actions.fillna("").astype(str).str.upper().str.strip()


def r5b8_rule_mask(df: pd.DataFrame, rule_id: str) -> np.ndarray:
    """Retorna a máscara bruta de uma regra R5B8, sem considerar ordem."""

    if rule_id == "R5B8_01_RELATIONSHIP_AGE_GTE_35D":
        return (_num(df, "dias_desde_primeiro_envio_recebedor") >= 35.0).fillna(False).to_numpy()

    if rule_id == "R5B8_02_LOW_RECEIVER_REP_LOW_PAYER_COUNT":
        receiver_rep = _num(df, "receiver_reputation_score")
        payer_count = _num(df, "qtd_pix_pagador_180d")
        return ((receiver_rep > 70.57506297) & (payer_count > 175.0)).fillna(False).to_numpy()

    if rule_id == "R5B8_03_LOW_RECEIVER_VALUE_LOW_PAYER_VALUE":
        receiver_value_bin = _str(df, "valor_rec_bin").str.lower().str.strip()
        payer_value = _num(df, "valor_total_pagador_180d")
        return ((receiver_value_bin == "val_rec_lt_5k") & (payer_value > 212416.178)).fillna(False).to_numpy()

    raise KeyError(f"Regra de severidade desconhecida: {rule_id}")


def apply_r5b8_block_deescalation(
    df: pd.DataFrame,
    base_actions: pd.Series | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Aplica a política R5B8 de forma sequencial sobre casos ainda em BLOQUEAR.

    Retorna a série de decisões finais e um dataframe de trace com uma coluna
    booleana por regra, além de `r5b8_any_rule_applied` e `r5b8_rule_applied`.
    """

    if base_actions is None:
        if "decisao" not in df.columns:
            raise KeyError("base_actions ausente e coluna 'decisao' não encontrada.")
        base_actions = df["decisao"]

    final_actions = normalize_action(base_actions).copy()
    trace = pd.DataFrame(index=df.index)
    trace["r5b8_any_rule_applied"] = False
    trace["r5b8_rule_applied"] = ""

    for rule in R5B8_BLOCK_TO_CONFIRM_RULES:
        eligible = final_actions.eq("BLOQUEAR").to_numpy()
        mask = r5b8_rule_mask(df, rule.rule_id) & eligible
        trace[rule.rule_id] = mask
        if mask.any():
            final_actions.loc[mask] = rule.target_action
            trace.loc[mask, "r5b8_any_rule_applied"] = True
            trace.loc[mask, "r5b8_rule_applied"] = rule.rule_id

    return final_actions, trace


def r5b8_policy_metadata() -> dict[str, Any]:
    return {
        "policy_id": R5B8_POLICY_ID,
        "rule_set_version": R5B8_RULE_SET_VERSION,
        "base_action": "BLOQUEAR",
        "target_action": "CONFIRMAR",
        "rules": [
            {
                "rule_id": rule.rule_id,
                "description": rule.description,
                "target_action": rule.target_action,
            }
            for rule in R5B8_BLOCK_TO_CONFIRM_RULES
        ],
    }


def r5b14_rule_mask(df: pd.DataFrame, rule_id: str) -> np.ndarray:
    """Retorna a mascara bruta de uma regra R5B14, sem considerar ordem."""

    if rule_id == "R5B14_CTB_01_LGBM_RAW_HIGH":
        return (_num(df, "lgbm_raw") >= 0.10711783).fillna(False).to_numpy()

    if rule_id == "R5B14_CTB_02_SCORE_2_3_LGBM_R4_HIGH":
        score_bin = _score_bin(df)
        lgbm_r4 = _num(df, "lgbm_r4_score", default=np.nan)
        if "lgbm_r4_score" not in df.columns:
            lgbm_r4 = _num(df, "lgbm_raw")
        return ((score_bin == "score_2_3") & (lgbm_r4 >= 0.475472966916)).fillna(False).to_numpy()

    if rule_id == "R5B14_CTB_03_SCORE_2_3_LGBM_R4_MED":
        score_bin = _score_bin(df)
        lgbm_r4 = _num(df, "lgbm_r4_score", default=np.nan)
        if "lgbm_r4_score" not in df.columns:
            lgbm_r4 = _num(df, "lgbm_raw")
        return ((score_bin == "score_2_3") & (lgbm_r4 >= 0.318070929491)).fillna(False).to_numpy()

    if rule_id == "R5B14_CTB_04_DOC_PHONE_HIGH_PAYER_COUNT":
        key_type = _str(df, "ds_tipo_chave_norm").str.strip()
        payer_count = _num(df, "qtd_pix_pagador_180d")
        return ((key_type == "DOCUMENTO_TELEFONE") & (payer_count >= 207.0)).fillna(False).to_numpy()

    if rule_id == "R5B14_CTB_05_OUTROS_RATIO_MAX_HIGH":
        key_type = _str(df, "ds_tipo_chave_norm").str.strip()
        ratio_max = _num(df, "ratio_valor_maximo_pagador_180d")
        return ((key_type == "OUTROS") & (ratio_max >= 4.9674631165863596)).fillna(False).to_numpy()

    if rule_id == "R5B14_ATB_01_DOC_PHONE_MORNING_SCORE_HIGH":
        key_type = _str(df, "ds_tipo_chave_norm").str.strip()
        period = _str(df, "periodo_dia").str.strip()
        score_bin = _score_bin(df)
        lgbm_bin = _lgbm_bin(df)
        return (
            (key_type == "DOCUMENTO_TELEFONE")
            & (period == "manha")
            & (score_bin == "score_GE_10")
            & (lgbm_bin == "lgbm_GE_0.1")
        ).fillna(False).to_numpy()

    if rule_id == "R5B14_ATB_02_NIGHT_SCORE_1_2_RATIO_HIGH":
        period = _str(df, "periodo_dia").str.strip()
        score_bin = _score_bin(df)
        lgbm_bin = _lgbm_bin(df)
        ratio_bin = _ratio_bin(df)
        return (
            (period == "noite")
            & (score_bin == "score_1_2")
            & (lgbm_bin == "lgbm_GE_0.1")
            & (ratio_bin == "ratio_GE_5")
        ).fillna(False).to_numpy()

    if rule_id == "R5B14_CTA_01_LOW_LGBM_RAW_COMPENSATION":
        return (_num(df, "lgbm_raw") <= R5B14_COMPENSATION_LGBM_RAW_MAX).fillna(False).to_numpy()

    raise KeyError(f"Regra R5B14 desconhecida: {rule_id}")


def apply_r5b14_operational_zero_fn_policy(
    df: pd.DataFrame,
    base_actions: pd.Series | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Aplica a politica R5B14 em tres camadas sequenciais:

    1. CONFIRMAR -> BLOQUEAR para concentrar fraudes em BLOQUEAR.
    2. APROVAR -> BLOQUEAR para remover fraudes aprovadas.
    3. CONFIRMAR -> APROVAR low-LGBM para compensar FP global.
    """

    if base_actions is None:
        if "decisao" not in df.columns:
            raise KeyError("base_actions ausente e coluna 'decisao' nao encontrada.")
        base_actions = df["decisao"]

    final_actions = normalize_action(base_actions).copy()
    trace = pd.DataFrame(index=df.index)
    trace["r5b14_any_rule_applied"] = False
    trace["r5b14_rule_applied"] = ""
    trace["r5b14_layer_applied"] = ""

    layers: tuple[tuple[str, str, tuple[SeverityRule, ...]], ...] = (
        ("CONFIRMAR", "CONFIRM_TO_BLOCK", R5B14_CONFIRM_TO_BLOCK_RULES),
        ("APROVAR", "APPROVE_TO_BLOCK", R5B14_APPROVE_TO_BLOCK_RULES),
        ("CONFIRMAR", "CONFIRM_TO_APPROVE", R5B14_CONFIRM_TO_APPROVE_RULES),
    )

    for source_action, layer, rules in layers:
        for rule in rules:
            eligible = final_actions.eq(source_action).to_numpy()
            mask = r5b14_rule_mask(df, rule.rule_id) & eligible
            trace[rule.rule_id] = mask
            if mask.any():
                final_actions.loc[mask] = rule.target_action
                trace.loc[mask, "r5b14_any_rule_applied"] = True
                trace.loc[mask, "r5b14_rule_applied"] = rule.rule_id
                trace.loc[mask, "r5b14_layer_applied"] = layer

    return final_actions, trace


def apply_r5b16_frozen_contract_policy(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Aplica o contrato R5B16 usando R4G frozen como decisao-base."""

    if R5B16_FROZEN_ACTION_COL not in df.columns:
        raise KeyError(f"Coluna frozen ausente: {R5B16_FROZEN_ACTION_COL}")

    final_actions, trace = apply_r5b14_operational_zero_fn_policy(
        df,
        df[R5B16_FROZEN_ACTION_COL],
    )
    trace["r5b16_frozen_base_action"] = normalize_action(df[R5B16_FROZEN_ACTION_COL])
    return final_actions, trace


def r5b14_policy_metadata() -> dict[str, Any]:
    return {
        "policy_id": R5B14_POLICY_ID,
        "rule_set_version": R5B14_RULE_SET_VERSION,
        "base_policy": "EXP-014B-R4G-FAST-FROZEN",
        "layers": [
            {
                "layer": "CONFIRM_TO_BLOCK",
                "base_action": "CONFIRMAR",
                "target_action": "BLOQUEAR",
                "rules": [
                    {"rule_id": rule.rule_id, "description": rule.description}
                    for rule in R5B14_CONFIRM_TO_BLOCK_RULES
                ],
            },
            {
                "layer": "APPROVE_TO_BLOCK",
                "base_action": "APROVAR",
                "target_action": "BLOQUEAR",
                "rules": [
                    {"rule_id": rule.rule_id, "description": rule.description}
                    for rule in R5B14_APPROVE_TO_BLOCK_RULES
                ],
            },
            {
                "layer": "CONFIRM_TO_APPROVE",
                "base_action": "CONFIRMAR",
                "target_action": "APROVAR",
                "rules": [
                    {"rule_id": rule.rule_id, "description": rule.description}
                    for rule in R5B14_CONFIRM_TO_APPROVE_RULES
                ],
            },
        ],
    }


def r5b16_policy_metadata() -> dict[str, Any]:
    metadata = r5b14_policy_metadata()
    return {
        "policy_id": R5B16_POLICY_ID,
        "rule_set_version": R5B14_RULE_SET_VERSION,
        "base_action_col": R5B16_FROZEN_ACTION_COL,
        "base_policy": "EXP-014B-R4G-FAST-FROZEN",
        "overlay_policy": metadata["policy_id"],
        "layers": metadata["layers"],
    }
