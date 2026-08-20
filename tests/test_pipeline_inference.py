#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_pipeline_inference.py — Testes de Integração de Inferência Ponta a Ponta
"""

import sys
from pathlib import Path
import unittest

# Configurar path para importar o backend
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
CORE_DIR = BACKEND_DIR / "core"
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from core.pipeline_orquestrador import PipelineOrquestrador


class TestPipelineInference(unittest.TestCase):

    def setUp(self) -> None:
        # Inicializar o orquestrador oficial de inferência
        self.pipeline = PipelineOrquestrador(shap_enabled=False)

    def test_end_to_end_inference_with_v3_features(self) -> None:
        """Garante que o pipeline de inferência processa transações com features v3."""
        
        # Mock de dados transacionais vindos da API (inclui metadados básicos e features v3)
        mock_transaction = {
            # Metadados de identificação
            "cd_pix": "E0000000020260610120000000000001",
            "dt_pix": "2026-06-10 12:00:00",
            "cd_cpf_pagador": "12345678901",
            "cd_cpf_cnpj_recebedor": "98765432109",
            
            # Dados básicos de transação
            "ds_chave_pix": "+5561999999999",
            "ds_tipo_chave": "TELEFONE",
            "vl_pix": 1500.00,
            "device_name": "iPhone14,2",
            "app_version": "7.30.1",
            "ip_address": "177.100.20.10",
            "latencia_rede_ms": 120.0,
            "tempo_interacao_ms": 3200.0,
            "tempo_processamento_host_ms": 140.0,
            "metodo_autenticacao": "biometria",
            "session_id": "a1b2c3d4-e5f6-7a8b-9c0d-e1f2a3b4c5d6",
            "topaz_risk_score": 0.5,
            "topaz_transacao_rejeitada": 0.0,
            "is_agendamento_recorrente": "false",
            "vl_renda_cliente": 8500.00,
            "ds_sexo": "M",
            "ds_estado_civil": "SOLTEIRO",
            "ds_segmento": "PREMIUM",
            "qt_dependentes": 0.0,
            
            # features históricas v3 hidratadas pelo Big Data/Preprocessor
            "qtd_pix_pagador_7d": 5.0,
            "qtd_pix_pagador_30d": 18.0,
            "qtd_pix_pagador_90d": 50.0,
            "qtd_pix_pagador_180d": 85.0,
            "valor_total_pagador_7d": 3500.00,
            "valor_total_pagador_30d": 12500.00,
            "valor_total_pagador_90d": 32000.00,
            "valor_total_pagador_180d": 54000.00,
            "max_qtd_pix_dia_pagador_7d": 2.0,
            "max_qtd_pix_dia_pagador_30d": 4.0,
            "valor_maximo_pix_pagador_180d": 5000.00,
            "soma_recebedores_distintos_dia_180d": 12.0,
            "qtd_pix_mesmo_recebedor_30d": 2.0,
            "qtd_pix_mesmo_recebedor_90d": 4.0,
            "qtd_pix_mesmo_recebedor_180d": 8.0,
            "valor_total_para_recebedor_30d": 2000.00,
            "valor_total_para_recebedor_90d": 4000.00,
            "valor_total_para_recebedor_180d": 8000.00,
            "primeira_data_envio_recebedor_180d": "2026-01-15 10:00:00",
            "primeiro_envio_para_recebedor_180d": 0.0,
            "dias_desde_primeiro_envio_recebedor": 146.0,
            "qtd_pix_recebidos_30d": 15.0,
            "qtd_pix_recebidos_90d": 40.0,
            "qtd_pix_recebidos_180d": 90.0,
            "valor_total_recebido_30d": 15000.00,
            "valor_total_recebido_90d": 42000.00,
            "valor_total_recebido_180d": 95000.00,
            "soma_pagadores_distintos_dia_recebedor_180d": 25.0,
            "max_qtd_pix_recebidos_dia_180d": 5.0,
            "burst_daily_7d_flag": 0.0,
            "first_receiver_flag_real": 0.0,
            "ratio_valor_media_pagador_90d": 2.34,
            "ratio_valor_maximo_pagador_180d": 0.30
        }

        # Rodar inferência ponta a ponta
        result = self.pipeline.analisar(mock_transaction)

        # Asserts estruturais da resposta de decisão
        self.assertIsNotNone(result, "O pipeline retornou None")
        self.assertIn("decisao", result)
        self.assertIn("score_final", result)
        self.assertIn("metadata", result)
        self.assertIn("componentes", result)

        # Validar tipo das decisões
        self.assertIn(result["decisao"], {"APROVAR", "CONFIRMAR", "BLOQUEAR"})
        self.assertTrue(isinstance(result["score_final"], float))
        self.assertTrue(0.0 <= result["score_final"] <= 100.0)

        # Validar componentes do score
        componentes = result["componentes"]
        self.assertIn("lgbm_raw", componentes)
        self.assertIn("lgbm_mapped", componentes)
        if "if_score" in componentes:
            self.assertTrue(isinstance(componentes["if_score"], float))

        raw_df = self.pipeline._prepare_raw(mock_transaction)
        features_df = self.pipeline._create_features(raw_df)
        for feature in [
            "payer_history_strength_score",
            "receiver_reputation_score",
            "relationship_strength_score",
            "receiver_novelty_risk_score",
            "transaction_normality_score",
            "payer_receiver_trust_score",
        ]:
            self.assertIn(feature, features_df.columns)
            value = float(features_df[feature].iloc[0])
            self.assertTrue(0.0 <= value <= 100.0, f"{feature} fora de [0,100]: {value}")

        for feature in [
            "trust_bucket",
            "receiver_rep_bucket",
            "relationship_bucket",
            "novelty_bucket",
        ]:
            self.assertIn(feature, features_df.columns)
            self.assertTrue(str(features_df[feature].iloc[0]))


if __name__ == "__main__":
    unittest.main()
