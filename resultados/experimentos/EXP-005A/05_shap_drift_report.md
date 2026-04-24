# EXP-005A — Feature Importance / Drift Report

- Vencedor: `LGBM_C_SPW_2_0X`
- Feature set: `baseline_features`
- Usa novas features EXP-005A: `False`
- Numero de features: `52`

## Observacao

Este relatorio usa `feature_importances_` do LightGBM como proxy rapido.
SHAP completo deve ser feito posteriormente no model card, se o candidato avancar no EXP-005B.

## Top features por importancia

| Feature | Importance |
|---|---:|
| `vl_pix` | 2159 |
| `nr_idade` | 1486 |
| `vl_latencia_rede_media_trimestre` | 1118 |
| `diff_latencia_cliente` | 1008 |
| `qt_tempo_relacionamento_mes` | 985 |
| `hour` | 952 |
| `ratio_latencia_cliente` | 930 |
| `qt_intervalo_transacao_minuto` | 601 |
| `rule_score_raw` | 548 |
| `minutes_since_prev_tx` | 488 |
| `diff_intervalo_vs_mediana` | 471 |
| `ratio_pix_renda` | 404 |
| `vl_mediana_pix_trimestre` | 356 |
| `qt_aparelhos_distintos_trimestre` | 285 |
| `vl_renda_cliente` | 209 |
| `qt_envio_recebedor_trimestre` | 198 |
| `pix_key_random_flag` | 122 |
| `qt_intervalo_mediana_trimestre` | 111 |
| `ratio_intervalo_vs_mediana` | 104 |
| `diff_valor_mediana` | 95 |
| `rule_topaz_score` | 91 |
| `topaz_risk_score` | 89 |
| `first_receiver_flag` | 80 |
| `rule_random_key_score` | 51 |
| `key_tx_count_prev` | 47 |
| `vl_desvio_padrao_pix_trimestre` | 44 |
| `ratio_valor_mediana` | 43 |
| `receiver_tx_count_prev` | 43 |
| `burst_30m_flag` | 35 |
| `is_segmento_premium_flag` | 35 |
| `vl_pix_over_1000_flag` | 35 |
| `qt_pix_dia_maximo_trimestre` | 30 |
| `distinct_keys_so_far` | 25 |
| `ratio_valor_desvio_padrao` | 24 |
| `rule_age_score` | 23 |
| `device_missing_flag` | 19 |
| `first_key_flag` | 15 |
| `qt_total_pix_trimestre` | 14 |
| `qt_intervalo_desvio_padrao_trimestre` | 13 |
| `renda_missing_flag` | 12 |

## Risco de deploy

- Se o vencedor usa features `exp005_*`, elas precisam ser promovidas para `preprocessing.py` e `PipelineOrquestrador` antes de qualquer runtime.
- EXP-005A nao e deployavel sozinho; ele gera candidato para EXP-005B.
