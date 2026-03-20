"""
rule_engine.py — Motor de Regras de Negócio para Fraude PIX
============================================================

Regras da equipe de analistas de fraude, implementadas como score
normalizado [0, 1] para integração no ensemble de inferência.

Cada regra gera um sub-score com peso definido. O score final é a
soma ponderada normalizada pelo peso máximo total.

Regras ativas (8):
  1. PIX em < 30min (velocity burst curto)
  2. Pressão de Limite (valor vs limite noturno BC + mediana + faixas)
  3. Idade do cliente (vulnerabilidade a engenharia social)
  4. Tempo de relacionamento com o banco
  5. Chave PIX aleatória (EVP)
  6. Horário noturno (20h-6h)
  7. Velocity checks (burst diário)
  8. Topaz score (risco do dispositivo)

Regras excluídas nesta versão:
  - Conta laranja (será tratada no módulo de Graph Analytics)
  - Autorização prévia (sem dados disponíveis)

Regulamentação de referência:
  - BC: Limite noturno PF→PF = R$ 1.000 (20h-6h)
  - BC: 99% das transferências PF < R$ 15.000
  - BC: Aumento de limite leva 24-48h para efetivar

Peso máximo total: 26 pontos
Score normalizado: [0.0, 1.0]

Integração no ensemble:
  score_final = LGBM * 0.65 + rule_score_normalized * 0.15 + IF * 0.20

Autor: Equipe Anomalia PIX
Versão: 1.0
Data: Março 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional


# =========================================================
# CONFIGURAÇÃO DE PESOS
# =========================================================

RULE_WEIGHTS = {
    "pix_30min": 2,
    "pressao_limite": 4,
    "idade": 3,
    "tempo_relacionamento": 3,
    "chave_aleatoria": 2,
    "horario_noturno": 3,
    "velocity_burst": 4,
    "topaz": 5,
}

MAX_TOTAL_WEIGHT = sum(RULE_WEIGHTS.values())  # 26

# Constantes regulatórias (Banco Central)
BC_LIMITE_NOTURNO_PF = 1000.0      # R$ 1.000 (20h-6h)
BC_FAIXA_ALTA = 5000.0             # Faixa de alto valor
BC_FAIXA_MUITO_ALTA = 15000.0      # 99% das tx PF ficam abaixo
BC_HORA_INICIO_NOTURNO = 20        # 20h
BC_HORA_FIM_NOTURNO = 6            # 6h


# =========================================================
# FUNÇÕES DE CADA REGRA
# =========================================================

def rule_pix_30min(
    minutes_since_prev_tx: Optional[float],
    qt_pix_30min: Optional[float] = None,
) -> Tuple[int, str]:
    """
    Regra 1: PIX em menos de 30 minutos desde a transação anterior.

    Lógica:
      - Intervalo > 30min → 0 (sem risco)
      - 1 tx anterior em < 30min → peso 1
      - 2+ tx anteriores em < 30min → peso 2

    Args:
        minutes_since_prev_tx: Minutos desde a última transação
        qt_pix_30min: Quantidade de PIX nos últimos 30min (se disponível)

    Returns:
        (score, detalhamento)
    """
    if minutes_since_prev_tx is None or pd.isna(minutes_since_prev_tx):
        return 0, "sem_historico"

    if minutes_since_prev_tx > 30:
        return 0, f"intervalo_normal_{minutes_since_prev_tx:.0f}min"

    if qt_pix_30min is not None and not pd.isna(qt_pix_30min) and qt_pix_30min >= 2:
        return 2, f"burst_{int(qt_pix_30min)}_tx_em_{minutes_since_prev_tx:.0f}min"

    return 1, f"1_tx_em_{minutes_since_prev_tx:.0f}min"


def rule_pressao_limite(
    vl_pix: Optional[float],
    hour: Optional[float],
    ratio_valor_mediana: Optional[float] = None,
    is_first_tx_trimestre: Optional[int] = None,
) -> Tuple[int, str]:
    """
    Regra 2: Pressão de Limite PIX.

    Combina 3 dimensões:
      a) Violação do limite noturno regulatório do BC (R$ 1.000 entre 20h-6h)
      b) Valor absoluto em faixas de risco (baseado em dados do BC)
      c) Ratio vs mediana do cliente (proxy de uso próximo ao limite)

    Regulamentação BC:
      - Noturno PF→PF: R$ 1.000 padrão (aumento leva 24-48h)
      - 99% das tx PF ficam abaixo de R$ 15.000

    Args:
        vl_pix: Valor da transação em R$
        hour: Hora da transação (0-23)
        ratio_valor_mediana: Razão valor/mediana do cliente
        is_first_tx_trimestre: Flag de primeira tx no trimestre

    Returns:
        (score 0-4, detalhamento)
    """
    if vl_pix is None or pd.isna(vl_pix):
        return 0, "sem_valor"

    score = 0
    reasons = []

    # --- (A) Limite noturno regulatório do BC ---
    is_noturno = False
    if hour is not None and not pd.isna(hour):
        is_noturno = (hour >= BC_HORA_INICIO_NOTURNO or hour < BC_HORA_FIM_NOTURNO)

    if is_noturno and vl_pix > BC_LIMITE_NOTURNO_PF:
        score += 2
        reasons.append(f"noturno_acima_limite_BC_R${vl_pix:.0f}")
    elif is_noturno and vl_pix > BC_LIMITE_NOTURNO_PF * 0.5:
        score += 1
        reasons.append(f"noturno_proximo_limite_BC_R${vl_pix:.0f}")

    # --- (B) Valor absoluto em faixas de risco ---
    if vl_pix >= BC_FAIXA_MUITO_ALTA:
        score += 2
        reasons.append(f"valor_extremo_R${vl_pix:.0f}_acima_99pct")
    elif vl_pix >= BC_FAIXA_ALTA:
        score += 1
        reasons.append(f"valor_alto_R${vl_pix:.0f}")

    # --- (C) Ratio vs mediana do cliente ---
    first_tx = (is_first_tx_trimestre == 1) if is_first_tx_trimestre is not None else False

    if ratio_valor_mediana is not None and not pd.isna(ratio_valor_mediana) and not first_tx:
        if ratio_valor_mediana >= 10:
            score += 2
            reasons.append(f"ratio_mediana_{ratio_valor_mediana:.1f}x_extremo")
        elif ratio_valor_mediana >= 5:
            score += 1
            reasons.append(f"ratio_mediana_{ratio_valor_mediana:.1f}x_alto")

    # Cap em peso máximo da regra
    score = min(RULE_WEIGHTS["pressao_limite"], score)

    if not reasons:
        detail = f"valor_normal_R${vl_pix:.0f}"
    else:
        detail = " | ".join(reasons)

    return score, detail


def rule_idade(nr_idade: Optional[float]) -> Tuple[int, str]:
    """
    Regra 3: Idade do cliente — vulnerabilidade a engenharia social.

    Baseado em estatísticas Febraban 2023:
      - 64% das vítimas de golpes PIX têm 60+ anos
      - Mulheres idosas são 2.3x mais vítimas

    Lógica:
      - 60-65 anos → peso 1
      - 66-75 anos → peso 2
      - 76+ anos → peso 3

    Args:
        nr_idade: Idade do cliente em anos

    Returns:
        (score 0-3, detalhamento)
    """
    if nr_idade is None or pd.isna(nr_idade):
        return 0, "sem_idade"

    idade = int(nr_idade)

    if idade >= 76:
        return 3, f"idoso_vulneravel_76+_{idade}a"
    elif idade >= 66:
        return 2, f"idoso_66-75_{idade}a"
    elif idade >= 60:
        return 1, f"idoso_60-65_{idade}a"

    return 0, f"idade_{idade}a"


def rule_tempo_relacionamento(
    qt_tempo_relacionamento_mes: Optional[float],
) -> Tuple[int, str]:
    """
    Regra 4: Tempo de relacionamento com o banco.

    Contas novas são alvos ou instrumentos de fraude:
      - Conta laranja: aberta para receber/redistribuir
      - Conta comprometida: cliente novo sem histórico

    Lógica (em dias aproximados):
      - 61-90 dias (2-3 meses) → peso 1
      - 31-60 dias (1-2 meses) → peso 2
      - 0-30 dias (< 1 mês)   → peso 3

    Args:
        qt_tempo_relacionamento_mes: Tempo de relacionamento em meses

    Returns:
        (score 0-3, detalhamento)
    """
    if qt_tempo_relacionamento_mes is None or pd.isna(qt_tempo_relacionamento_mes):
        return 0, "sem_info_relacionamento"

    dias = qt_tempo_relacionamento_mes * 30.44

    if dias <= 30:
        return 3, f"conta_nova_{qt_tempo_relacionamento_mes:.1f}m_({dias:.0f}d)"
    elif dias <= 60:
        return 2, f"conta_recente_{qt_tempo_relacionamento_mes:.1f}m"
    elif dias <= 90:
        return 1, f"conta_jovem_{qt_tempo_relacionamento_mes:.1f}m"

    return 0, f"conta_estabelecida_{qt_tempo_relacionamento_mes:.0f}m"


def rule_chave_aleatoria(
    pix_key_random_flag: Optional[float],
) -> Tuple[int, str]:
    """
    Regra 5: Chave PIX aleatória (EVP/UUID).

    Chave aleatória indica que o pagador NÃO conhece o CPF, celular
    ou email do recebedor — apenas a chave UUID fornecida pelo golpista.

    Estatística: 76% das fraudes PIX usam chave aleatória.

    Lógica:
      - Chave aleatória → peso 2
      - Outra chave → peso 0

    Args:
        pix_key_random_flag: 1 se chave aleatória, 0 caso contrário

    Returns:
        (score 0-2, detalhamento)
    """
    if pix_key_random_flag is None or pd.isna(pix_key_random_flag):
        return 0, "sem_info_chave"

    if int(pix_key_random_flag) == 1:
        return 2, "chave_evp_uuid"

    return 0, "chave_identificavel"


def rule_horario_noturno(hour: Optional[float]) -> Tuple[int, str]:
    """
    Regra 6: Horário noturno (20h-6h).

    Regulamentação BC: Período noturno = 20h às 6h (ou 22h às 6h).
    Estatísticas:
      - 85% dos sequestros relâmpago ocorrem entre 22h-5h
      - Limite noturno PF→PF = R$ 1.000 (aumento demora 24-48h)

    Nota: O golpe do falso funcionário ocorre em HORÁRIO COMERCIAL,
    mas esta regra cobre os casos de coação/sequestro.

    Lógica:
      - 20h-6h → peso 3

    Args:
        hour: Hora da transação (0-23)

    Returns:
        (score 0-3, detalhamento)
    """
    if hour is None or pd.isna(hour):
        return 0, "sem_hora"

    h = int(hour)

    if h >= BC_HORA_INICIO_NOTURNO or h < BC_HORA_FIM_NOTURNO:
        return 3, f"noturno_{h}h"

    return 0, f"diurno_{h}h"


def rule_velocity_burst(
    qt_pix_dia_maximo_trimestre: Optional[float],
) -> Tuple[int, str]:
    """
    Regra 7: Velocity checks — burst de transações diárias.

    Detecta padrões de esvaziamento de conta ou automação.
    O máximo diário do trimestre indica o pior caso observado.

    Lógica:
      - Máximo diário > 50 → peso 4 (automação/bot)
      - Máximo diário > 20 → peso 3 (burst alto)
      - Máximo diário > 10 → peso 2 (burst moderado)

    Args:
        qt_pix_dia_maximo_trimestre: Máximo de PIX em um único dia no trimestre

    Returns:
        (score 0-4, detalhamento)
    """
    if qt_pix_dia_maximo_trimestre is None or pd.isna(qt_pix_dia_maximo_trimestre):
        return 0, "sem_historico_velocity"

    maximo = int(qt_pix_dia_maximo_trimestre)

    if maximo > 50:
        return 4, f"burst_extremo_{maximo}/dia"
    elif maximo > 20:
        return 3, f"burst_alto_{maximo}/dia"
    elif maximo > 10:
        return 2, f"burst_moderado_{maximo}/dia"

    return 0, f"normal_{maximo}/dia"


def rule_topaz(topaz_score_filled: Optional[float]) -> Tuple[int, str]:
    """
    Regra 8: Score Topaz (FICO/Grupo Stefanini).

    Score de risco do dispositivo/sessão, gerado pelo sistema antifraude
    Topaz. Escala 0-5.

    Interpretação:
      - 0-1: Baixo risco → peso 0
      - 2: Risco moderado → peso 2
      - 3: Risco alto → peso 3
      - 4-5: Risco crítico → peso 4-5

    Args:
        topaz_score_filled: Score Topaz (0-5, 0=sem dado)

    Returns:
        (score 0-5, detalhamento)
    """
    if topaz_score_filled is None or pd.isna(topaz_score_filled):
        return 0, "sem_topaz"

    score = int(min(5, max(0, topaz_score_filled)))

    if score <= 1:
        return 0, f"topaz_{score}_dispositivo_ok"

    labels = {
        2: "moderado",
        3: "alto",
        4: "muito_alto",
        5: "critico",
    }
    return score, f"topaz_{score}_{labels.get(score, 'desconhecido')}"


# =========================================================
# MOTOR PRINCIPAL
# =========================================================

def compute_rule_score(row: pd.Series) -> Dict[str, Any]:
    """
    Computa todas as 8 regras para uma transação.

    Args:
        row: pd.Series com as features da transação.
             Features esperadas:
               - minutes_since_prev_tx (float)
               - vl_pix (float)
               - hour (int/float)
               - ratio_valor_mediana (float)
               - is_first_tx_trimestre (int)
               - nr_idade (float)
               - qt_tempo_relacionamento_mes (float)
               - pix_key_random_flag (int)
               - qt_pix_dia_maximo_trimestre (float)
               - topaz_score_filled (float)

    Returns:
        Dict com:
          - rule_score_raw: soma dos pesos (0 a MAX_TOTAL_WEIGHT)
          - rule_score_normalized: score normalizado [0, 1]
          - rules_triggered_count: quantas regras foram ativadas
          - rules_triggered: lista legível das regras ativadas
          - rule_details: dict com detalhes de cada regra
    """
    results = {}

    # Regra 1: PIX em < 30min
    s1, d1 = rule_pix_30min(
        row.get("minutes_since_prev_tx"),
        row.get("qt_pix_30min"),
    )
    results["pix_30min"] = {"score": s1, "max": RULE_WEIGHTS["pix_30min"], "detail": d1}

    # Regra 2: Pressão de Limite
    s2, d2 = rule_pressao_limite(
        row.get("vl_pix"),
        row.get("hour"),
        row.get("ratio_valor_mediana"),
        row.get("is_first_tx_trimestre"),
    )
    results["pressao_limite"] = {"score": s2, "max": RULE_WEIGHTS["pressao_limite"], "detail": d2}

    # Regra 3: Idade
    s3, d3 = rule_idade(row.get("nr_idade"))
    results["idade"] = {"score": s3, "max": RULE_WEIGHTS["idade"], "detail": d3}

    # Regra 4: Tempo de relacionamento
    s4, d4 = rule_tempo_relacionamento(row.get("qt_tempo_relacionamento_mes"))
    results["tempo_relacionamento"] = {"score": s4, "max": RULE_WEIGHTS["tempo_relacionamento"], "detail": d4}

    # Regra 5: Chave aleatória
    s5, d5 = rule_chave_aleatoria(row.get("pix_key_random_flag"))
    results["chave_aleatoria"] = {"score": s5, "max": RULE_WEIGHTS["chave_aleatoria"], "detail": d5}

    # Regra 6: Horário noturno
    s6, d6 = rule_horario_noturno(row.get("hour"))
    results["horario_noturno"] = {"score": s6, "max": RULE_WEIGHTS["horario_noturno"], "detail": d6}

    # Regra 7: Velocity burst
    s7, d7 = rule_velocity_burst(row.get("qt_pix_dia_maximo_trimestre"))
    results["velocity_burst"] = {"score": s7, "max": RULE_WEIGHTS["velocity_burst"], "detail": d7}

    # Regra 8: Topaz
    s8, d8 = rule_topaz(row.get("topaz_score_filled"))
    results["topaz"] = {"score": s8, "max": RULE_WEIGHTS["topaz"], "detail": d8}

    # --- Agregar ---
    total_raw = sum(r["score"] for r in results.values())
    total_normalized = min(1.0, total_raw / MAX_TOTAL_WEIGHT)

    triggered = [
        f"{name}({r['score']}/{r['max']}): {r['detail']}"
        for name, r in results.items()
        if r["score"] > 0
    ]

    return {
        "rule_score_raw": int(total_raw),
        "rule_score_normalized": float(round(total_normalized, 6)),
        "rules_triggered_count": len(triggered),
        "rules_triggered": triggered,
        "rule_details": results,
    }


def compute_rule_scores_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computa regras para um DataFrame inteiro.

    Args:
        df: DataFrame com features das transações

    Returns:
        DataFrame com colunas:
          - rule_score_raw (int)
          - rule_score_normalized (float)
          - rules_triggered_count (int)
          - rules_triggered (str)
          - rule_pix_30min ... rule_topaz (int, sub-scores individuais)
    """
    results = df.apply(compute_rule_score, axis=1)

    out = pd.DataFrame({
        "rule_score_raw": results.apply(lambda x: x["rule_score_raw"]),
        "rule_score_normalized": results.apply(lambda x: x["rule_score_normalized"]),
        "rules_triggered_count": results.apply(lambda x: x["rules_triggered_count"]),
        "rules_triggered": results.apply(
            lambda x: " | ".join(x["rules_triggered"]) if x["rules_triggered"] else ""
        ),
    })

    # Sub-scores individuais para auditoria
    for rule_name in RULE_WEIGHTS.keys():
        out[f"rule_{rule_name}"] = results.apply(
            lambda x: x["rule_details"][rule_name]["score"]
        )

    return out


# =========================================================
# ENSEMBLE SCORE (para uso no pipeline de inferência)
# =========================================================

def compute_ensemble_score(
    lgbm_score: float,
    rule_score_normalized: float,
    if_score: float,
    is_first_tx: bool,
    lgbm_low: float = 0.0001,
    lgbm_high: float = 0.7,
    w_lgbm: float = 0.65,
    w_rule: float = 0.15,
    w_if: float = 0.20,
) -> Dict[str, Any]:
    """
    Calcula o score final do ensemble com 3 componentes.

    Pesos: LGBM=0.65, Regras=0.15, IF=0.20

    O IF só contribui para primeiras tx com LGBM na zona cinzenta.
    Quando IF inativo, redistribui peso proporcionalmente entre LGBM e Regras.

    Args:
        lgbm_score: Score do LightGBM [0, 1]
        rule_score_normalized: Score das regras [0, 1]
        if_score: Score do Isolation Forest [0, 1]
        is_first_tx: Flag de primeira transação no trimestre
        lgbm_low: Limite inferior da zona cinzenta do LGBM
        lgbm_high: Limite superior da zona cinzenta do LGBM
        w_lgbm: Peso do LGBM
        w_rule: Peso das regras
        w_if: Peso do IF

    Returns:
        Dict com score final e componentes
    """
    # Condição de ativação do IF
    if_active = (
        is_first_tx
        and lgbm_score >= lgbm_low
        and lgbm_score <= lgbm_high
    )

    if if_active:
        final_w_lgbm = w_lgbm
        final_w_rule = w_rule
        final_w_if = w_if
    else:
        # Redistribuir peso do IF proporcionalmente
        ratio_lgbm = w_lgbm / (w_lgbm + w_rule)
        final_w_lgbm = w_lgbm + w_if * ratio_lgbm
        final_w_rule = w_rule + w_if * (1 - ratio_lgbm)
        final_w_if = 0.0

    final_score = (
        final_w_lgbm * lgbm_score
        + final_w_rule * rule_score_normalized
        + final_w_if * if_score
    )

    return {
        "final_score": float(np.clip(final_score, 0.0, 1.0)),
        "lgbm_score": float(lgbm_score),
        "rule_score_normalized": float(rule_score_normalized),
        "if_score": float(if_score),
        "w_lgbm": float(round(final_w_lgbm, 4)),
        "w_rule": float(round(final_w_rule, 4)),
        "w_if": float(round(final_w_if, 4)),
        "if_active": bool(if_active),
    }


# =========================================================
# UTILIDADES
# =========================================================

def get_rule_summary() -> Dict[str, Any]:
    """Retorna resumo das regras configuradas."""
    return {
        "version": "1.0",
        "n_rules": len(RULE_WEIGHTS),
        "rules": {
            name: {"max_weight": weight}
            for name, weight in RULE_WEIGHTS.items()
        },
        "max_total_weight": MAX_TOTAL_WEIGHT,
        "regulatory_params": {
            "bc_limite_noturno_pf": BC_LIMITE_NOTURNO_PF,
            "bc_faixa_alta": BC_FAIXA_ALTA,
            "bc_faixa_muito_alta": BC_FAIXA_MUITO_ALTA,
            "bc_hora_inicio_noturno": BC_HORA_INICIO_NOTURNO,
            "bc_hora_fim_noturno": BC_HORA_FIM_NOTURNO,
        },
        "ensemble_weights": {
            "lgbm": 0.65,
            "rules": 0.15,
            "isolation_forest": 0.20,
        },
    }


# =========================================================
# MAIN — Teste rápido
# =========================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RULE ENGINE — Teste de Cenários")
    print("=" * 70)

    scenarios = [
        {
            "name": "Idosa vítima de falso funcionário",
            "data": {
                "vl_pix": 5000.0,
                "hour": 14,
                "nr_idade": 72,
                "qt_tempo_relacionamento_mes": 240,
                "pix_key_random_flag": 1,
                "topaz_score_filled": 0,
                "minutes_since_prev_tx": 45,
                "qt_pix_dia_maximo_trimestre": 3,
                "ratio_valor_mediana": 8.5,
                "is_first_tx_trimestre": 0,
            },
        },
        {
            "name": "Falso sequestro madrugada",
            "data": {
                "vl_pix": 4999.0,
                "hour": 3,
                "nr_idade": 45,
                "qt_tempo_relacionamento_mes": 120,
                "pix_key_random_flag": 1,
                "topaz_score_filled": 0,
                "minutes_since_prev_tx": 3,
                "qt_pix_dia_maximo_trimestre": 5,
                "ratio_valor_mediana": 6.2,
                "is_first_tx_trimestre": 0,
            },
        },
        {
            "name": "Conta laranja jovem",
            "data": {
                "vl_pix": 8000.0,
                "hour": 10,
                "nr_idade": 22,
                "qt_tempo_relacionamento_mes": 0.5,
                "pix_key_random_flag": 1,
                "topaz_score_filled": 3,
                "minutes_since_prev_tx": 2,
                "qt_pix_dia_maximo_trimestre": 25,
                "ratio_valor_mediana": 15.0,
                "is_first_tx_trimestre": 1,
            },
        },
        {
            "name": "Transação normal",
            "data": {
                "vl_pix": 50.0,
                "hour": 14,
                "nr_idade": 35,
                "qt_tempo_relacionamento_mes": 60,
                "pix_key_random_flag": 0,
                "topaz_score_filled": 0,
                "minutes_since_prev_tx": 4320,
                "qt_pix_dia_maximo_trimestre": 3,
                "ratio_valor_mediana": 0.8,
                "is_first_tx_trimestre": 0,
            },
        },
        {
            "name": "PIX noturno acima do limite BC",
            "data": {
                "vl_pix": 3000.0,
                "hour": 23,
                "nr_idade": 55,
                "qt_tempo_relacionamento_mes": 180,
                "pix_key_random_flag": 0,
                "topaz_score_filled": 2,
                "minutes_since_prev_tx": 120,
                "qt_pix_dia_maximo_trimestre": 4,
                "ratio_valor_mediana": 2.5,
                "is_first_tx_trimestre": 0,
            },
        },
    ]

    for scenario in scenarios:
        row = pd.Series(scenario["data"])
        result = compute_rule_score(row)

        print(f"\n{'─' * 60}")
        print(f"Cenário: {scenario['name']}")
        print(f"  Score raw: {result['rule_score_raw']}/{MAX_TOTAL_WEIGHT}")
        print(f"  Score normalizado: {result['rule_score_normalized']:.4f}")
        print(f"  Regras ativadas: {result['rules_triggered_count']}")
        for rule in result["rules_triggered"]:
            print(f"    → {rule}")

    # Teste de ensemble
    print(f"\n{'=' * 60}")
    print("Teste de Ensemble Score:")
    ensemble = compute_ensemble_score(
        lgbm_score=0.85,
        rule_score_normalized=0.35,
        if_score=0.70,
        is_first_tx=True,
    )
    print(f"  LGBM=0.85, Rules=0.35, IF=0.70 (1ª tx)")
    print(f"  IF ativo: {ensemble['if_active']}")
    print(f"  Pesos: LGBM={ensemble['w_lgbm']}, Rules={ensemble['w_rule']}, IF={ensemble['w_if']}")
    print(f"  Score final: {ensemble['final_score']:.4f}")

    ensemble2 = compute_ensemble_score(
        lgbm_score=0.85,
        rule_score_normalized=0.35,
        if_score=0.70,
        is_first_tx=False,
    )
    print(f"\n  LGBM=0.85, Rules=0.35, IF=0.70 (tx regular)")
    print(f"  IF ativo: {ensemble2['if_active']}")
    print(f"  Pesos: LGBM={ensemble2['w_lgbm']}, Rules={ensemble2['w_rule']}, IF={ensemble2['w_if']}")
    print(f"  Score final: {ensemble2['final_score']:.4f}")

    print(f"\n{'=' * 60}")
    print("Resumo das regras:")
    summary = get_rule_summary()
    for name, info in summary["rules"].items():
        print(f"  {name}: max_weight={info['max_weight']}")
    print(f"  TOTAL: {summary['max_total_weight']}")
    print(f"\n  Ensemble: {summary['ensemble_weights']}")
