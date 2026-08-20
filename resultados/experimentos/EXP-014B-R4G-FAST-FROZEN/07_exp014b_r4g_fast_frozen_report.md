# EXP-014B-R4G-FAST-FROZEN - Frozen validation

## Resultado executivo
- Status: `PASS_R4G_FAST_FROZEN_VALIDATED_REPLAY_OK`
- All pass: `True`
- Prediction mismatches: `0`
- APROVAR -> CONFIRMAR: `13`
- BLOQUEAR -> CONFIRMAR: `0`

## Métricas de intervenção congelada
```json
{
  "tp": 1463,
  "fp": 1123,
  "fn": 2,
  "tn": 111256,
  "precision": 0.56573859,
  "recall": 0.99863481,
  "f1": 0.72229079,
  "fpr": 0.00999297
}
```

## Métricas de BLOQUEAR congelado
```json
{
  "tp": 1458,
  "fp": 766,
  "fn": 7,
  "tn": 111613,
  "precision": 0.65557554,
  "recall": 0.99522184,
  "f1": 0.79045812,
  "fpr": 0.00681622
}
```

## Métricas por ação
| action    |   n_rows |   n_frauds |   n_normals |   precision_within_action |
|:----------|---------:|-----------:|------------:|--------------------------:|
| APROVAR   |   111258 |          2 |      111256 |                 1.798e-05 |
| BLOQUEAR  |     2224 |       1458 |         766 |                 0.655576  |
| CONFIRMAR |      362 |          5 |         357 |                 0.0138122 |

## Validação
```json
{
  "status": "PASS_R4G_FAST_FROZEN_VALIDATED_REPLAY_OK",
  "base_predictions_path": "C:\\Users\\u857755\\OneDrive - BRB - Banco de Brasilia SA\\Documentos\\Projetos\\squad_IA\\PIX\\rebuild_pix\\resultados\\experimentos\\EXP-014B-R4F-FROZEN\\06_predictions_frozen.csv",
  "artifact_path": "C:\\Users\\u857755\\OneDrive - BRB - Banco de Brasilia SA\\Documentos\\Projetos\\squad_IA\\PIX\\rebuild_pix\\resultados\\experimentos\\EXP-014B-R4G-FAST\\10_policy_artifact_recommended.json",
  "reference_predictions_path": "C:\\Users\\u857755\\OneDrive - BRB - Banco de Brasilia SA\\Documentos\\Projetos\\squad_IA\\PIX\\rebuild_pix\\resultados\\experimentos\\EXP-014B-R4G-FAST\\11_predictions_recommended.csv",
  "base_action_col": "r4f_frozen_decisao_recommended",
  "frozen_action_col": "r4g_fast_frozen_decisao_recommended",
  "frozen_approve_to_confirm_col": "exp014b_r4g_fast_frozen_approve_to_confirm",
  "frozen_block_to_confirm_col": "exp014b_r4g_fast_frozen_block_to_confirm",
  "frozen_intervention_col": "exp014b_r4g_fast_frozen_intervention_pred",
  "frozen_block_col": "exp014b_r4g_fast_frozen_block_pred",
  "expected_intervention_metrics": {
    "tp": 1463,
    "fp": 1123,
    "fn": 2,
    "tn": 111256,
    "precision": 0.56573859,
    "recall": 0.99863481,
    "f1": 0.72229079,
    "fpr": 0.00999297
  },
  "frozen_intervention_metrics": {
    "tp": 1463,
    "fp": 1123,
    "fn": 2,
    "tn": 111256,
    "precision": 0.56573859,
    "recall": 0.99863481,
    "f1": 0.72229079,
    "fpr": 0.00999297
  },
  "expected_block_metrics": {
    "tp": 1458,
    "fp": 766,
    "fn": 7,
    "tn": 111613,
    "precision": 0.65557554,
    "recall": 0.99522184,
    "f1": 0.79045812,
    "fpr": 0.00681622
  },
  "frozen_block_metrics": {
    "tp": 1458,
    "fp": 766,
    "fn": 7,
    "tn": 111613,
    "precision": 0.65557554,
    "recall": 0.99522184,
    "f1": 0.79045812,
    "fpr": 0.00681622
  },
  "intervention_match_expected": true,
  "block_match_expected": true,
  "n_any_mismatches": 0,
  "all_pass": true
}
```

## Decisão sugerida
Se PASS, consolidar R4G-FAST-FROZEN como melhor baseline geral até agora.
