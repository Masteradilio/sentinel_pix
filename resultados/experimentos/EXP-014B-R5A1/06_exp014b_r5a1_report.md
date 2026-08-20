# EXP-014B-R5A1 - Minimal Feature Set and Redundancy Audit

## Executive summary
- Baseline: `EXP-014B-R4G-FAST-FROZEN`
- Rows: `113844`
- Columns audited: `205`
- Minimal frozen replay columns: `5`
- Redundancy candidates: `32`
- Train/runtime LGBM feature set equal: `False`

## Column taxonomy
- `bin_feature`: 6
- `diagnostic_only`: 3
- `experiment_legacy`: 60
- `id`: 5
- `label`: 2
- `metadata`: 10
- `model_score`: 11
- `policy_column`: 31
- `raw_or_engineered_feature`: 77

## Initial roles
- `candidate_feature_review`: 68
- `diagnostic_or_policy_only`: 16
- `drop_from_feature_engineering`: 91
- `keep_for_model_candidate`: 10
- `keep_for_replay_contract`: 5
- `keep_out_of_model_contract_only`: 15

## Main warning
The predictions file has many policy and legacy experiment columns. They must not be treated as primary model features in R5B/R5C.

## Required next use
Use `05_feature_keep_drop_recommendations.csv` as the gate before adding relationship or receiver reputation features.
