# EXP-014B-R5B3-RESIDUAL-FEATURE-GAP — Diagnóstico de resíduos pós R5B2

## Grupos críticos
- Normais ainda em BLOQUEAR: `3153`
- Fraudes em BLOQUEAR: `279`
- Fraudes ainda em APROVAR: `682`
- Normais em APROVAR: `96795`

## Principais diferenças — normal BLOQUEAR vs fraude BLOQUEAR
| feature                             |   block_normal_n |   block_fraud_n |   block_normal_missing_rate |   block_fraud_missing_rate |   block_normal_median |   block_fraud_median |   median_delta_a_minus_b |   block_normal_p90 |   block_fraud_p90 |   abs_median_delta |
|:------------------------------------|-----------------:|----------------:|----------------------------:|---------------------------:|----------------------:|---------------------:|-------------------------:|-------------------:|------------------:|-------------------:|
| valor_total_pagador_180d            |             3153 |             279 |                    0        |                   0        |           34732.2     |           16026.8    |              18705.4     |        938466      |        89536.3    |        18705.4     |
| valor_total_pagador_90d             |             3153 |             279 |                    0        |                   0        |           30888.2     |           15052      |              15836.2     |        803803      |        88065.2    |        15836.2     |
| valor_maximo_pix_pagador_180d       |             3153 |             279 |                    0        |                   0        |            6600       |            2600      |               4000       |         97488      |        21000      |         4000       |
| vl_pix                              |             3153 |             279 |                    0        |                   0        |            6077.47    |            2998.91   |               3078.56    |         25000      |        12115      |         3078.56    |
| dias_desde_primeiro_envio_recebedor |              952 |              25 |                    0.698065 |                   0.910394 |              46       |               5      |                 41       |           121      |            8      |           41       |
| qtd_pix_pagador_30d                 |             3153 |             279 |                    0        |                   0        |              35       |              27      |                  8       |           122      |           88.6    |            8       |
| ratio_valor_media_pagador_90d       |             3153 |             279 |                    0        |                   0        |               8.74858 |              14.7289 |                 -5.98031 |            50.7882 |           60.8412 |            5.98031 |
| qtd_pix_pagador_180d                |             3153 |             279 |                    0        |                   0        |              67       |              72      |                 -5       |           338.8    |          210.6    |            5       |
| soma_recebedores_distintos_dia_180d |             3153 |             279 |                    0        |                   0        |              62       |              64      |                 -2       |           305      |          193.4    |            2       |
| max_qtd_pix_dia_pagador_30d         |             3153 |             279 |                    0        |                   0        |               6       |               5      |                  1       |            19      |           11      |            1       |

| feature                  | value               |   block_normal_n |   block_fraud_n |   block_normal_rate |   block_fraud_rate |   rate_delta_a_minus_b |   abs_rate_delta |
|:-------------------------|:--------------------|-----------------:|----------------:|--------------------:|-------------------:|-----------------------:|-----------------:|
| sample_strategy          | N2_MATCHED_CONTROLS |             3153 |               0 |            1        |           0        |               1        |         1        |
| sample_strategy          | P1_MAF_RECENTE_180D |                0 |             279 |            0        |           1        |              -1        |         1        |
| value_band               | D_1000_5000         |             1004 |             180 |            0.318427 |           0.645161 |              -0.326734 |         0.326734 |
| qtd_rec_bin              | rec_0               |             1684 |             222 |            0.534095 |           0.795699 |              -0.261604 |         0.261604 |
| valor_rec_bin            | val_rec_0           |             1684 |             222 |            0.534095 |           0.795699 |              -0.261604 |         0.261604 |
| qtd_rec_bin              | rec_1_10            |             1333 |              51 |            0.422772 |           0.182796 |               0.239976 |         0.239976 |
| lgbm_bin                 | lgbm_LT_0.05        |             1051 |              27 |            0.333333 |           0.096774 |               0.236559 |         0.236559 |
| first_receiver_flag_real | 1                   |             2201 |             254 |            0.698065 |           0.910394 |              -0.212329 |         0.212329 |
| first_receiver_flag_real | 0                   |              952 |              25 |            0.301935 |           0.089606 |               0.212329 |         0.212329 |
| lgbm_bin                 | lgbm_0.15_0.35      |              111 |              69 |            0.035205 |           0.247312 |              -0.212107 |         0.212107 |

## Principais diferenças — fraude APROVAR vs normal APROVAR
| feature                             |   approve_fraud_n |   approve_normal_n |   approve_fraud_missing_rate |   approve_normal_missing_rate |   approve_fraud_median |   approve_normal_median |   median_delta_a_minus_b |   approve_fraud_p90 |   approve_normal_p90 |   abs_median_delta |
|:------------------------------------|------------------:|-------------------:|-----------------------------:|------------------------------:|-----------------------:|------------------------:|-------------------------:|--------------------:|---------------------:|-------------------:|
| valor_total_pagador_180d            |               682 |              96795 |                     0        |                      0        |                5050    |                13162.4  |                 -8112.39 |            57295.9  |     125204           |            8112.39 |
| valor_total_pagador_90d             |               682 |              96795 |                     0        |                      0        |                4221.32 |                10622.9  |                 -6401.55 |            46965.5  |      99461.6         |            6401.55 |
| valor_total_recebido_180d           |               682 |              96795 |                     0        |                      0        |                   0    |                 3650    |                 -3650    |             7730.06 |          3.11094e+06 |            3650    |
| valor_total_recebido_90d            |               682 |              96795 |                     0        |                      0        |                   0    |                 2948.33 |                 -2948.33 |             7730.06 |          2.67736e+06 |            2948.33 |
| valor_total_recebido_30d            |               682 |              96795 |                     0        |                      0        |                   0    |                 1059.79 |                 -1059.79 |             6339.1  |          1.4736e+06  |            1059.79 |
| vl_pix                              |               682 |              96795 |                     0        |                      0        |                 996.5  |                  500    |                   496.5  |             6590    |       5053.07        |             496.5  |
| valor_maximo_pix_pagador_180d       |               572 |              92839 |                     0.16129  |                      0.04087  |                1700    |                 2000    |                  -300    |            13900    |      20000           |             300    |
| valor_total_para_recebedor_180d     |               682 |              96795 |                     0        |                      0        |                   0    |                  190.6  |                  -190.6  |             1950    |      11800           |             190.6  |
| valor_total_para_recebedor_90d      |               682 |              96795 |                     0        |                      0        |                   0    |                  145    |                  -145    |             1950    |       9639.9         |             145    |
| dias_desde_primeiro_envio_recebedor |               128 |              59520 |                     0.812317 |                      0.385092 |                   7.5  |                   60    |                   -52.5  |               64.3  |        140           |              52.5  |

| feature                  | value               |   approve_fraud_n |   approve_normal_n |   approve_fraud_rate |   approve_normal_rate |   rate_delta_a_minus_b |   abs_rate_delta |
|:-------------------------|:--------------------|------------------:|-------------------:|---------------------:|----------------------:|-----------------------:|-----------------:|
| sample_strategy          | P1_MAF_RECENTE_180D |               682 |                  0 |             1        |              0        |               1        |         1        |
| sample_strategy          | N2_MATCHED_CONTROLS |                 0 |              92241 |             0        |              0.952952 |              -0.952952 |         0.952952 |
| lgbm_bin                 | lgbm_0.05_0.15      |               559 |              15279 |             0.819648 |              0.157849 |               0.661799 |         0.661799 |
| lgbm_bin                 | lgbm_LT_0.05        |               123 |              81516 |             0.180352 |              0.842151 |              -0.661799 |         0.661799 |
| qtd_rec_bin              | rec_0               |               404 |              13588 |             0.592375 |              0.140379 |               0.451996 |         0.451996 |
| valor_rec_bin            | val_rec_0           |               404 |              13588 |             0.592375 |              0.140379 |               0.451996 |         0.451996 |
| first_receiver_flag_real | 1                   |               554 |              37275 |             0.812317 |              0.385092 |               0.427225 |         0.427225 |
| first_receiver_flag_real | 0                   |               128 |              59520 |             0.187683 |              0.614908 |              -0.427225 |         0.427225 |
| qtd_rec_bin              | rec_11_plus         |                62 |              45983 |             0.090909 |              0.475056 |              -0.384146 |         0.384146 |
| valor_rec_bin            | val_rec_gt_5k       |                83 |              44416 |             0.121701 |              0.458867 |              -0.337166 |         0.337166 |

## Próximas ações recomendadas
1. Congelar um subconjunto produtivo/robusto das regras `BLOQUEAR -> CONFIRMAR`.
2. Criar features de reputação do recebedor e força do relacionamento pagador-recebedor.
3. Rodar experimento separado para resgatar fraudes em `APROVAR`, pois ainda restam `682` casos.
