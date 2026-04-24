# EXP-004-FINAL - Meta Shadow Report

Este arquivo nao treina um meta-learner. Ele resume os sinais shadow que podem alimentar a FASE 2.

## Baseline

- TP=345, FP=15, FN=10, F1=0.9650

## Variantes e cobertura

### BASELINE

- TP=345, FP=15, FN=10, F1=0.9650
- Delta TP=+0, Delta FP=+0, Delta FN=+0
- Policy hits=0, hits fraude=0, precision hits=0.0000
- Upgrades=0, upgrades fraude=0, upgrades legitima=0

### V1_GUARD_CONTEXTUAL

- TP=346, FP=15, FN=9, F1=0.9665
- Delta TP=+1, Delta FP=+0, Delta FN=-1
- Policy hits=1, hits fraude=1, precision hits=1.0000
- Upgrades=1, upgrades fraude=1, upgrades legitima=0

### V2_RATE_LIMIT

- TP=345, FP=15, FN=10, F1=0.9650
- Delta TP=+0, Delta FP=+0, Delta FN=+0
- Policy hits=57, hits fraude=57, precision hits=1.0000
- Upgrades=0, upgrades fraude=0, upgrades legitima=0

### V3_PRIMEIRO_RECEIVER

- TP=346, FP=95, FN=9, F1=0.8693
- Delta TP=+1, Delta FP=+80, Delta FN=-1
- Policy hits=208, hits fraude=122, precision hits=0.5865
- Upgrades=81, upgrades fraude=1, upgrades legitima=80

### V4_COMBO_FINAL

- TP=346, FP=95, FN=9, F1=0.8693
- Delta TP=+1, Delta FP=+80, Delta FN=-1
- Policy hits=264, hits fraude=178, precision hits=0.6742
- Upgrades=81, upgrades fraude=1, upgrades legitima=80

## Rule stats por variante

Resumo completo esta em `02_delta_fp_fn_por_variante.json`.

### V1_GUARD_CONTEXTUAL

- `GUARD_EXCEPTION_ALTO_VALOR_SE_BEH`: hits=1, fraud_hits=1, precision=1.0000, fns_recuperaveis=1, fps_potenciais=0
- `RATE_LIMIT_ANOMALO`: hits=57, fraud_hits=57, precision=1.0000, fns_recuperaveis=0, fps_potenciais=0
- `PRIMEIRO_RECEIVER_VALOR_ANOMALO`: hits=208, fraud_hits=122, precision=0.5865, fns_recuperaveis=1, fps_potenciais=80

### V2_RATE_LIMIT

- `GUARD_EXCEPTION_ALTO_VALOR_SE_BEH`: hits=1, fraud_hits=1, precision=1.0000, fns_recuperaveis=1, fps_potenciais=0
- `RATE_LIMIT_ANOMALO`: hits=57, fraud_hits=57, precision=1.0000, fns_recuperaveis=0, fps_potenciais=0
- `PRIMEIRO_RECEIVER_VALOR_ANOMALO`: hits=208, fraud_hits=122, precision=0.5865, fns_recuperaveis=1, fps_potenciais=80

### V3_PRIMEIRO_RECEIVER

- `GUARD_EXCEPTION_ALTO_VALOR_SE_BEH`: hits=1, fraud_hits=1, precision=1.0000, fns_recuperaveis=1, fps_potenciais=0
- `RATE_LIMIT_ANOMALO`: hits=57, fraud_hits=57, precision=1.0000, fns_recuperaveis=0, fps_potenciais=0
- `PRIMEIRO_RECEIVER_VALOR_ANOMALO`: hits=208, fraud_hits=122, precision=0.5865, fns_recuperaveis=1, fps_potenciais=80

### V4_COMBO_FINAL

- `GUARD_EXCEPTION_ALTO_VALOR_SE_BEH`: hits=1, fraud_hits=1, precision=1.0000, fns_recuperaveis=1, fps_potenciais=0
- `RATE_LIMIT_ANOMALO`: hits=57, fraud_hits=57, precision=1.0000, fns_recuperaveis=0, fps_potenciais=0
- `PRIMEIRO_RECEIVER_VALOR_ANOMALO`: hits=208, fraud_hits=122, precision=0.5865, fns_recuperaveis=1, fps_potenciais=80

## FNs residuais classificados

- `GUARD_SUPPRESSED_CANDIDATE`: 9
