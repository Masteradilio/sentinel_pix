# Catálogo Oficial de Regras - Baseline R5B22

Este documento lista as regras sistêmicas ativas no baseline oficial de produção.
A decisão do motor R5B22 é baseada no contrato estático `r4g_fast_frozen_decisao_recommended` em conjunto com a aplicação sequencial das políticas abaixo, guiando o treinamento do modelo Aluno.

## 1. Contrato Professor (Base Decisória)

Representa as flags e restrições históricas utilizadas como rótulo de aprendizado para o modelo destilado.

- **Decisão-base:** `r4g_fast_frozen_decisao_recommended` atua como o ponto de partida do contrato congelado, não consistindo em uma regra condicional simples, mas no veredito fixo de pipeline consolidado.
- **Flags rastreadas pelo Aluno:** `r5b14_rule_applied`, `r5b14_layer_applied`, entre outras `_frozen` features.

## 2. Política R5B14 - Prevenção de Falsos Negativos (Severidade)

Regras ativadas para elevar o risco de transações potencialmente enganosas.

| Rule ID | Camada | Condição | Ação | Objetivo |
|---------|--------|----------|------|----------|
| `R5B14_CTB_01_LGBM_RAW_HIGH` | CONFIRM_TO_BLOCK | `lgbm_raw >= 0.10711783` | Escalonar CONFIRMAR para BLOQUEAR | Prevenção de Falso Negativo (Segurança) |
| `R5B14_CTB_02_SCORE_2_3_LGBM_R4_HIGH` | CONFIRM_TO_BLOCK | `score_bin == score_2_3 AND lgbm_r4_score >= 0.475472966916` | Escalonar CONFIRMAR para BLOQUEAR | Prevenção de Falso Negativo (Segurança) |
| `R5B14_CTB_03_SCORE_2_3_LGBM_R4_MED` | CONFIRM_TO_BLOCK | `score_bin == score_2_3 AND lgbm_r4_score >= 0.318070929491` | Escalonar CONFIRMAR para BLOQUEAR | Prevenção de Falso Negativo (Segurança) |
| `R5B14_CTB_04_DOC_PHONE_HIGH_PAYER_COUNT` | CONFIRM_TO_BLOCK | `ds_tipo_chave_norm == DOCUMENTO_TELEFONE AND qtd_pix_pagador_180d >= 207` | Escalonar CONFIRMAR para BLOQUEAR | Prevenção de Falso Negativo (Segurança) |
| `R5B14_CTB_05_OUTROS_RATIO_MAX_HIGH` | CONFIRM_TO_BLOCK | `ds_tipo_chave_norm == OUTROS AND ratio_valor_maximo_pagador_180d >= 4.9674631165863596` | Escalonar CONFIRMAR para BLOQUEAR | Prevenção de Falso Negativo (Segurança) |
| `R5B14_ATB_01_DOC_PHONE_MORNING_SCORE_HIGH` | APPROVE_TO_BLOCK | `ds_tipo_chave_norm == DOCUMENTO_TELEFONE AND periodo_dia == manha AND score_bin == score_GE_10 AND lgbm_bin == lgbm_GE_0.1` | Escalonar APROVAR para BLOQUEAR | Prevenção de Falso Negativo (Segurança) |
| `R5B14_ATB_02_NIGHT_SCORE_1_2_RATIO_HIGH` | APPROVE_TO_BLOCK | `periodo_dia == noite AND score_bin == score_1_2 AND lgbm_bin == lgbm_GE_0.1 AND ratio_bin == ratio_GE_5` | Escalonar APROVAR para BLOQUEAR | Prevenção de Falso Negativo (Segurança) |
| `R5B14_CTA_01_LOW_LGBM_RAW_COMPENSATION` | CONFIRM_TO_APPROVE | `lgbm_raw <= 1.966e-05` | Desescalonar CONFIRMAR para APROVAR | Redução de Falso Positivo (Compensação) |

## 3. Política R5B22 - Controle de FPR (Suavização)

Regras aplicadas para reduzir bloqueios e confirmações indevidos que passaram na malha R5B14.

| Rule ID | Alvo | Condição | Fraudes (Inc.) | Normais (Inc.) |
|---------|------|----------|----------------|----------------|
| `DEMOTE_LAYER_APPROVE_TO_BLOCK_TO_APROVAR` | Reverter intervenção para APROVAR | `r5b14_layer_applied == APPROVE_TO_BLOCK` | 2 | 47 |
| `DEMOTE_LAYER_CONFIRM_TO_BLOCK_TO_CONFIRMAR` | Reverter intervenção para CONFIRMAR | `r5b14_layer_applied == CONFIRM_TO_BLOCK` | 5 | 22 |
| `DEMOTE_CAT2_ds_tipo_chave_norm_OUTROS__lgbm_bin_lgbm_0.05_0.1` | Reverter intervenção para CONFIRMAR | `ds_tipo_chave_norm == OUTROS AND lgbm_bin == lgbm_0.05_0.1` | 4 | 4 |
| `DEMOTE_CAT2_value_band_E_5000_10000__lgbm_bin_lgbm_0.05_0.1` | Reverter intervenção para CONFIRMAR | `value_band == E_5000_10000 AND lgbm_bin == lgbm_0.05_0.1` | 1 | 2 |

## 4. Política Candidata: Severity Policy

O módulo `backend/core/severity_policy.py` contém as regras estruturais listadas na Política R5B14, sendo o mecanismo oficialmente encarregado no baseline. Modificações ou adições a este arquivo só devem ser incorporadas ao baseline após passar pelas provas empíricas e pela geração da máscara R5B22.
