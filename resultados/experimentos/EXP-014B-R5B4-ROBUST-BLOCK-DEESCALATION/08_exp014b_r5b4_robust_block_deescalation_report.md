# EXP-014B-R5B4-ROBUST-BLOCK-DEESCALATION — Política robusta BLOQUEAR -> CONFIRMAR

## Resultado executivo
- Status: `PASS_R5B4_ROBUST_POLICY_SELECTED`
- Regras candidatas R5B2: `60`
- Regras robustas selecionadas: `57`
- Normais movidos de BLOQUEAR para CONFIRMAR: `11038`
- Fraudes movidas de BLOQUEAR para CONFIRMAR: `0`
- Normais restantes em BLOQUEAR: `3182`
- Fraudes restantes em APROVAR: `682`

## Critério de seleção
```json
{
  "total_frauds": 0,
  "non_train_frauds": 0,
  "non_train_normals_min": 20,
  "month_normal_support_min": 2
}
```

## Métricas finais de BLOQUEAR
```json
{
  "tp": 279,
  "fp": 3182,
  "fn": 1186,
  "tn": 109197,
  "precision": 0.08061254,
  "recall": 0.19044369,
  "f1": 0.11327649,
  "fpr": 0.0283149
}
```

## Decisões finais
| r5b4_robust_decisao   |   n_rows |   n_frauds |   n_normals |
|:----------------------|---------:|-----------:|------------:|
| APROVAR               |    97477 |        682 |       96795 |
| BLOQUEAR              |     3461 |        279 |        3182 |
| CONFIRMAR             |    12906 |        504 |       12402 |

## Estabilidade por split/mês
| temporal_split   | event_month   |   n_rows |   block_to_confirm_normals |   block_to_confirm_frauds |   remaining_block_normals |   remaining_block_frauds |
|:-----------------|:--------------|---------:|---------------------------:|--------------------------:|--------------------------:|-------------------------:|
| HOLDOUT          | 2026-04       |     2357 |                        259 |                         0 |                        55 |                       13 |
| HOLDOUT          | 2026-05       |    15015 |                       1430 |                         0 |                       173 |                        7 |
| TRAIN            | 2025-11       |     2090 |                         31 |                         0 |                        20 |                        3 |
| TRAIN            | 2025-12       |    23173 |                       2430 |                         0 |                      1149 |                       68 |
| TRAIN            | 2026-01       |    18829 |                       1792 |                         0 |                       554 |                       53 |
| TRAIN            | 2026-02       |    17291 |                       1721 |                         0 |                       462 |                       48 |
| TRAIN            | 2026-03       |    17297 |                       1797 |                         0 |                       409 |                       44 |
| VALIDATION       | 2026-03       |     1971 |                        152 |                         0 |                        44 |                        3 |
| VALIDATION       | 2026-04       |    15821 |                       1426 |                         0 |                       316 |                       40 |

## Regras selecionadas
|   selection_step | candidate_id                                                                                                 |   incremental_normals |   incremental_frauds |   non_train_normals |   holdout_normals |   validation_normals |   month_normal_support |
|-----------------:|:-------------------------------------------------------------------------------------------------------------|----------------------:|---------------------:|--------------------:|------------------:|---------------------:|-----------------------:|
|                1 | block_to_confirm_cat2__value_band=F_10000_PLUS__qtd_rec_bin=rec_11_plus                                      |                  1944 |                    0 |                 736 |               372 |                  364 |                      7 |
|                2 | block_to_confirm_cat2__periodo_dia=manha__qtd_rec_bin=rec_11_plus                                            |                  1237 |                    0 |                 730 |               370 |                  360 |                      7 |
|                3 | block_to_confirm_cat3__lgbm_bin=lgbm_LT_0.05__valor_rec_bin=val_rec_gt_5k__first_receiver_flag_real=1        |                  1444 |                    0 |                 623 |               312 |                  311 |                      7 |
|                4 | block_to_confirm_cat3__ds_tipo_chave_norm=DOCUMENTO_TELEFONE__lgbm_bin=lgbm_LT_0.05__qtd_rec_bin=rec_11_plus |                   339 |                    0 |                 567 |               304 |                  263 |                      7 |
|                5 | block_to_confirm_cat3__lgbm_bin=lgbm_LT_0.05__qtd_rec_bin=rec_11_plus__first_receiver_flag_real=1            |                    40 |                    0 |                 509 |               257 |                  252 |                      7 |
|                6 | block_to_confirm_cat3__periodo_dia=tarde__ratio_bin=ratio_1_5__first_receiver_flag_real=0                    |                   977 |                    0 |                 442 |               218 |                  224 |                      7 |
|                7 | block_to_confirm_cat3__lgbm_bin=lgbm_LT_0.05__ratio_bin=ratio_GE_5__mbk_available_flag=0                     |                   311 |                    0 |                 423 |               376 |                   47 |                      6 |
|                8 | block_to_confirm_cat3__ds_tipo_chave_norm=OUTROS__lgbm_bin=lgbm_LT_0.05__ratio_bin=ratio_GE_5                |                   692 |                    0 |                 419 |               198 |                  221 |                      7 |
|                9 | block_to_confirm_cat3__value_band=F_10000_PLUS__lgbm_bin=lgbm_LT_0.05__mbk_available_flag=0                  |                   179 |                    0 |                 404 |               336 |                   68 |                      6 |
|               10 | block_to_confirm_cat3__ds_tipo_chave_norm=OUTROS__periodo_dia=tarde__first_receiver_flag_real=0              |                    81 |                    0 |                 329 |               151 |                  178 |                      6 |
|               11 | block_to_confirm_cat3__value_band=E_5000_10000__periodo_dia=tarde__ratio_bin=ratio_1_5                       |                   216 |                    0 |                 306 |               146 |                  160 |                      7 |
|               12 | block_to_confirm_cat3__lgbm_bin=lgbm_LT_0.05__mbk_available_flag=0__first_receiver_flag_real=1               |                   148 |                    0 |                 279 |               244 |                   35 |                      6 |
|               13 | block_to_confirm_cat3__ds_tipo_chave_norm=CHAVE_ALEATORIA__value_band=F_10000_PLUS__lgbm_bin=lgbm_LT_0.05    |                   164 |                    0 |                 265 |               128 |                  137 |                      7 |
|               14 | block_to_confirm_cat3__value_band=E_5000_10000__periodo_dia=noite__lgbm_bin=lgbm_LT_0.05                     |                   511 |                    0 |                 256 |               134 |                  122 |                      7 |
|               15 | block_to_confirm_cat3__periodo_dia=noite__lgbm_bin=lgbm_LT_0.05__ratio_bin=ratio_GE_5                        |                   104 |                    0 |                 236 |               110 |                  126 |                      7 |
|               16 | block_to_confirm_cat3__value_band=E_5000_10000__ratio_bin=ratio_1_5__mbk_available_flag=0                    |                    66 |                    0 |                 228 |               201 |                   27 |                      7 |
|               17 | block_to_confirm_cat3__ds_tipo_chave_norm=EMAIL__ratio_bin=ratio_GE_5__valor_rec_bin=val_rec_gt_5k           |                   207 |                    0 |                 219 |               107 |                  112 |                      6 |
|               18 | block_to_confirm_cat3__value_band=E_5000_10000__valor_rec_bin=val_rec_gt_5k__first_receiver_flag_real=1      |                    41 |                    0 |                 194 |                93 |                  101 |                      7 |
|               19 | block_to_confirm_cat2__ratio_bin=ratio_LT_1__qtd_rec_bin=rec_11_plus                                         |                    84 |                    0 |                 183 |               102 |                   81 |                      6 |
|               20 | block_to_confirm_cat3__ds_tipo_chave_norm=DOCUMENTO_TELEFONE__lgbm_bin=lgbm_LT_0.05__ratio_bin=ratio_LT_1    |                   272 |                    0 |                 161 |                83 |                   78 |                      7 |
|               21 | block_to_confirm_cat3__ratio_bin=ratio_1_5__qtd_rec_bin=rec_1_10__mbk_available_flag=0                       |                    24 |                    0 |                 156 |               131 |                   25 |                      6 |
|               22 | block_to_confirm_cat3__value_band=E_5000_10000__lgbm_bin=lgbm_LT_0.05__valor_rec_bin=val_rec_lt_5k           |                   241 |                    0 |                 135 |                65 |                   70 |                      7 |
|               23 | block_to_confirm_cat3__value_band=D_1000_5000__periodo_dia=manha__lgbm_bin=lgbm_LT_0.05                      |                   135 |                    0 |                 132 |                72 |                   60 |                      7 |
|               24 | block_to_confirm_cat3__value_band=F_10000_PLUS__periodo_dia=manha__mbk_available_flag=0                      |                    28 |                    0 |                 130 |               120 |                   10 |                      6 |
|               25 | block_to_confirm_cat3__ds_tipo_chave_norm=CHAVE_ALEATORIA__value_band=E_5000_10000__ratio_bin=ratio_1_5      |                    27 |                    0 |                 126 |                64 |                   62 |                      7 |
|               26 | block_to_confirm_cat3__ds_tipo_chave_norm=OUTROS__periodo_dia=noite__mbk_available_flag=1                    |                   254 |                    0 |                 111 |                45 |                   66 |                      7 |
|               27 | block_to_confirm_cat3__periodo_dia=manha__qtd_rec_bin=rec_1_10__mbk_available_flag=0                         |                    23 |                    0 |                 104 |                94 |                   10 |                      6 |
|               28 | block_to_confirm_cat3__value_band=D_1000_5000__periodo_dia=manha__valor_rec_bin=val_rec_gt_5k                |                    11 |                    0 |                  98 |                51 |                   47 |                      6 |
|               29 | block_to_confirm_cat3__ds_tipo_chave_norm=CHAVE_ALEATORIA__value_band=F_10000_PLUS__ratio_bin=ratio_1_5      |                    11 |                    0 |                  97 |                52 |                   45 |                      6 |
|               30 | block_to_confirm_cat3__ds_tipo_chave_norm=EMAIL__value_band=E_5000_10000__ratio_bin=ratio_1_5                |                    50 |                    0 |                  96 |                46 |                   50 |                      7 |
|               31 | block_to_confirm_cat1__ds_tipo_chave_norm=INFORMACAO_AUSENTE                                                 |                   227 |                    0 |                  94 |                56 |                   38 |                      6 |
|               32 | block_to_confirm_cat3__value_band=D_1000_5000__lgbm_bin=lgbm_LT_0.05__ratio_bin=ratio_LT_1                   |                    69 |                    0 |                  93 |                55 |                   38 |                      7 |
|               33 | block_to_confirm_cat3__ds_tipo_chave_norm=OUTROS__value_band=F_10000_PLUS__mbk_available_flag=0              |                    15 |                    0 |                  93 |                86 |                    7 |                      6 |
|               34 | block_to_confirm_cat3__value_band=E_5000_10000__periodo_dia=noite__mbk_available_flag=0                      |                    17 |                    0 |                  83 |                73 |                   10 |                      6 |
|               35 | block_to_confirm_cat3__ds_tipo_chave_norm=EMAIL__ratio_bin=ratio_1_5__mbk_available_flag=0                   |                    20 |                    0 |                  80 |                62 |                   18 |                      6 |
|               36 | block_to_confirm_cat3__periodo_dia=noite__valor_rec_bin=val_rec_lt_5k__mbk_available_flag=1                  |                   116 |                    0 |                  66 |                32 |                   34 |                      7 |
|               37 | block_to_confirm_cat3__lgbm_bin=lgbm_LT_0.05__ratio_bin=ratio_1_5__qtd_rec_bin=rec_0                         |                   203 |                    0 |                  61 |                30 |                   31 |                      7 |
|               38 | block_to_confirm_cat3__ds_tipo_chave_norm=EMAIL__value_band=F_10000_PLUS__mbk_available_flag=0               |                     6 |                    0 |                  60 |                50 |                   10 |                      6 |
|               39 | block_to_confirm_cat3__value_band=E_5000_10000__ratio_bin=ratio_1_5__valor_rec_bin=val_rec_lt_5k             |                     9 |                    0 |                  53 |                25 |                   28 |                      7 |
|               40 | block_to_confirm_cat3__ds_tipo_chave_norm=EMAIL__periodo_dia=manha__mbk_available_flag=0                     |                    13 |                    0 |                  51 |                45 |                    6 |                      6 |
|               41 | block_to_confirm_cat3__value_band=F_10000_PLUS__ratio_bin=ratio_LT_1__mbk_available_flag=1                   |                    36 |                    0 |                  50 |                21 |                   29 |                      7 |
|               42 | block_to_confirm_cat3__value_band=F_10000_PLUS__lgbm_bin=lgbm_LT_0.05__qtd_rec_bin=rec_0                     |                    49 |                    0 |                  47 |                20 |                   27 |                      7 |
|               43 | block_to_confirm_cat3__ds_tipo_chave_norm=OUTROS__ratio_bin=ratio_1_5__qtd_rec_bin=rec_0                     |                   134 |                    0 |                  44 |                25 |                   19 |                      7 |
|               44 | block_to_confirm_cat3__ds_tipo_chave_norm=CHAVE_ALEATORIA__ratio_bin=ratio_1_5__qtd_rec_bin=rec_1_10         |                    19 |                    0 |                  41 |                20 |                   21 |                      7 |
|               45 | block_to_confirm_score__vl_pix__>=152999.902                                                                 |                    28 |                    0 |                  41 |                21 |                   20 |                      6 |
|               46 | block_to_confirm_cat3__value_band=D_1000_5000__ratio_bin=ratio_LT_1__mbk_available_flag=0                    |                    12 |                    0 |                  40 |                34 |                    6 |                      6 |
|               47 | block_to_confirm_cat3__ds_tipo_chave_norm=CHAVE_ALEATORIA__periodo_dia=noite__mbk_available_flag=0           |                    12 |                    0 |                  40 |                33 |                    7 |                      6 |
|               48 | block_to_confirm_cat3__ds_tipo_chave_norm=EMAIL__periodo_dia=tarde__ratio_bin=ratio_LT_1                     |                    46 |                    0 |                  34 |                19 |                   15 |                      6 |
|               49 | block_to_confirm_cat3__value_band=E_5000_10000__periodo_dia=noite__ratio_bin=ratio_LT_1                      |                     7 |                    0 |                  34 |                16 |                   18 |                      6 |
|               50 | block_to_confirm_cat3__ds_tipo_chave_norm=EMAIL__ratio_bin=ratio_LT_1__first_receiver_flag_real=1            |                    18 |                    0 |                  33 |                17 |                   16 |                      7 |
|               51 | block_to_confirm_cat3__ds_tipo_chave_norm=OUTROS__periodo_dia=noite__qtd_rec_bin=rec_0                       |                    24 |                    0 |                  31 |                16 |                   15 |                      7 |
|               52 | block_to_confirm_cat3__ratio_bin=ratio_1_5__valor_rec_bin=val_rec_lt_5k__first_receiver_flag_real=0          |                    10 |                    0 |                  29 |                14 |                   15 |                      7 |
|               53 | block_to_confirm_cat3__ds_tipo_chave_norm=EMAIL__ratio_bin=ratio_1_5__valor_rec_bin=val_rec_lt_5k            |                    28 |                    0 |                  27 |                13 |                   14 |                      7 |
|               54 | block_to_confirm_cat3__ds_tipo_chave_norm=EMAIL__periodo_dia=noite__mbk_available_flag=0                     |                    16 |                    0 |                  27 |                21 |                    6 |                      6 |
|               55 | block_to_confirm_cat1__periodo_dia=madrugada                                                                 |                    16 |                    0 |                  24 |                14 |                   10 |                      7 |
|               56 | block_to_confirm_cat3__periodo_dia=manha__lgbm_bin=lgbm_LT_0.05__qtd_rec_bin=rec_0                           |                    14 |                    0 |                  23 |                11 |                   12 |                      7 |
|               57 | block_to_confirm_cat3__ds_tipo_chave_norm=EMAIL__value_band=F_10000_PLUS__periodo_dia=noite                  |                    13 |                    0 |                  21 |                 7 |                   14 |                      6 |
