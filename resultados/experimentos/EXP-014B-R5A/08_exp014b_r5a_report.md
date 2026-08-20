# EXP-014B-R5A - Residual Error Feature Diagnosis

## Executive summary
- Baseline: `EXP-014B-R4G-FAST-FROZEN`
- Input: `resultados\experimentos\EXP-014B-R4G-FAST-FROZEN\06_predictions_frozen.csv`
- Used fallback: `False`
- Rows: `113844`

## Residual groups
- approve_fraud_count: `2`
- approve_normal_count: `111256`
- block_fraud_count: `1458`
- block_normal_count: `766`
- confirm_fraud_count: `5`
- confirm_normal_count: `357`

## Key comparisons
- `05_feature_contrast_approve_fraud_vs_approve_normal.csv` compares frauds still approved against normal approvals.
- `06_feature_contrast_block_normal_vs_block_fraud.csv` compares normal transactions blocked against frauds blocked.

## Candidate feature gaps
- `missing_pix_key_age`: No explicit PIX key age feature was found.
- `missing_receiver_account_age`: No explicit receiver account age feature was found.
- `missing_receiver_fraud_rate`: No historical receiver fraud rate feature was found.
- `missing_receiver_temporal_stability`: No explicit receiver temporal stability score was found.

## Notes
- This is diagnostic only; no model was trained.
- Gap inference is based on column names, not data lineage proof.
- Promote no new baseline from this artifact alone.
