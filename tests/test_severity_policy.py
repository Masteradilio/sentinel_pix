import unittest

import pandas as pd

from backend.core.severity_policy import (
    apply_r5b8_block_deescalation,
    apply_r5b14_operational_zero_fn_policy,
    apply_r5b16_frozen_contract_policy,
    r5b8_policy_metadata,
    r5b14_policy_metadata,
    r5b16_policy_metadata,
)


class TestSeverityPolicy(unittest.TestCase):
    def test_r5b8_policy_applies_only_to_blocked_cases_in_order(self):
        df = pd.DataFrame(
            {
                "decisao": ["BLOQUEAR", "BLOQUEAR", "BLOQUEAR", "CONFIRMAR", "APROVAR"],
                "dias_desde_primeiro_envio_recebedor": [35, 1, 1, 80, 80],
                "receiver_reputation_score": [100, 80.0, 60.0, 10.0, 10.0],
                "qtd_pix_pagador_180d": [100, 176.0, 100.0, 1.0, 1.0],
                "valor_rec_bin": ["val_rec_ge_5k", "val_rec_ge_5k", "val_rec_lt_5k", "val_rec_lt_5k", "val_rec_lt_5k"],
                "valor_total_pagador_180d": [10000.0, 10000.0, 212416.179, 100.0, 100.0],
            }
        )

        final_actions, trace = apply_r5b8_block_deescalation(df)

        self.assertEqual(final_actions.tolist(), ["CONFIRMAR", "CONFIRMAR", "CONFIRMAR", "CONFIRMAR", "APROVAR"])
        self.assertEqual(trace["r5b8_any_rule_applied"].tolist(), [True, True, True, False, False])
        self.assertEqual(
            trace["r5b8_rule_applied"].tolist(),
            [
                "R5B8_01_RELATIONSHIP_AGE_GTE_35D",
                "R5B8_02_LOW_RECEIVER_REP_LOW_PAYER_COUNT",
                "R5B8_03_LOW_RECEIVER_VALUE_LOW_PAYER_VALUE",
                "",
                "",
            ],
        )

    def test_r5b8_policy_metadata_is_explicit(self):
        metadata = r5b8_policy_metadata()

        self.assertEqual(metadata["policy_id"], "EXP-014B-R5B8-BROAD-RESIDUAL-RULE-MINING")
        self.assertEqual(len(metadata["rules"]), 3)
        self.assertIn("dias_desde_primeiro_envio_recebedor >= 35", metadata["rules"][0]["description"])

    def test_r5b14_policy_applies_three_layers_in_order(self):
        df = pd.DataFrame(
            {
                "decisao": ["CONFIRMAR", "APROVAR", "CONFIRMAR", "BLOQUEAR"],
                "lgbm_raw": [0.2, 0.0, 0.00001, 0.00001],
                "lgbm_r4_score": [0.0, 0.0, 0.0, 0.0],
                "score_bin": ["score_0_1", "score_GE_10", "score_0_1", "score_0_1"],
                "ds_tipo_chave_norm": ["EMAIL", "DOCUMENTO_TELEFONE", "EMAIL", "EMAIL"],
                "qtd_pix_pagador_180d": [0, 0, 0, 0],
                "ratio_valor_maximo_pagador_180d": [0.0, 0.0, 0.0, 0.0],
                "periodo_dia": ["tarde", "manha", "tarde", "tarde"],
                "lgbm_bin": ["lgbm_LT_0.1", "lgbm_GE_0.1", "lgbm_LT_0.1", "lgbm_LT_0.1"],
                "ratio_bin": ["ratio_LT_5", "ratio_LT_5", "ratio_LT_5", "ratio_LT_5"],
            }
        )

        final_actions, trace = apply_r5b14_operational_zero_fn_policy(df)

        self.assertEqual(final_actions.tolist(), ["BLOQUEAR", "BLOQUEAR", "APROVAR", "BLOQUEAR"])
        self.assertEqual(
            trace["r5b14_layer_applied"].tolist(),
            ["CONFIRM_TO_BLOCK", "APPROVE_TO_BLOCK", "CONFIRM_TO_APPROVE", ""],
        )

    def test_r5b14_policy_derives_runtime_bins_when_missing(self):
        df = pd.DataFrame(
            {
                "decisao": ["APROVAR", "APROVAR"],
                "score_final": [12.0, 1.5],
                "lgbm_raw": [0.2, 0.2],
                "ratio_valor_maximo_pagador_180d": [1.0, 6.0],
                "ds_tipo_chave_norm": ["DOCUMENTO_TELEFONE", "EMAIL"],
                "periodo_dia": ["manha", "noite"],
            }
        )

        final_actions, trace = apply_r5b14_operational_zero_fn_policy(df)

        self.assertEqual(final_actions.tolist(), ["BLOQUEAR", "BLOQUEAR"])
        self.assertEqual(
            trace["r5b14_rule_applied"].tolist(),
            [
                "R5B14_ATB_01_DOC_PHONE_MORNING_SCORE_HIGH",
                "R5B14_ATB_02_NIGHT_SCORE_1_2_RATIO_HIGH",
            ],
        )

    def test_r5b14_policy_metadata_is_explicit(self):
        metadata = r5b14_policy_metadata()

        self.assertEqual(metadata["policy_id"], "EXP-014B-R5B14-OPERATIONAL-ZERO-FN-REPLAY")
        self.assertEqual([layer["layer"] for layer in metadata["layers"]], [
            "CONFIRM_TO_BLOCK",
            "APPROVE_TO_BLOCK",
            "CONFIRM_TO_APPROVE",
        ])

    def test_r5b16_frozen_contract_uses_r4g_action_as_base(self):
        df = pd.DataFrame(
            {
                "r4g_fast_frozen_decisao_recommended": ["APROVAR", "CONFIRMAR", "BLOQUEAR"],
                "score_final": [12.0, 84.0, 1.0],
                "lgbm_raw": [0.2, 0.2, 0.0],
                "ds_tipo_chave_norm": ["DOCUMENTO_TELEFONE", "EMAIL", "EMAIL"],
                "periodo_dia": ["manha", "tarde", "tarde"],
            }
        )

        final_actions, trace = apply_r5b16_frozen_contract_policy(df)

        self.assertEqual(final_actions.tolist(), ["BLOQUEAR", "BLOQUEAR", "BLOQUEAR"])
        self.assertEqual(
            trace["r5b16_frozen_base_action"].tolist(),
            ["APROVAR", "CONFIRMAR", "BLOQUEAR"],
        )

    def test_r5b16_policy_metadata_is_explicit(self):
        metadata = r5b16_policy_metadata()

        self.assertEqual(metadata["policy_id"], "EXP-014B-R5B16-CONSOLIDATED-OPERATIONAL-BASELINE")
        self.assertEqual(metadata["base_action_col"], "r4g_fast_frozen_decisao_recommended")


if __name__ == "__main__":
    unittest.main()
