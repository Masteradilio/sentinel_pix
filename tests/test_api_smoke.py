#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_api_smoke.py — Testes de Fumaça HTTP para a API REST Antifraude PIX
"""

import sys
from pathlib import Path
import unittest
from fastapi.testclient import TestClient

# Configurar caminhos para importação do backend
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
CORE_DIR = BACKEND_DIR / "core"
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Importar o app da API
from api import app


class TestApiSmoke(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # Criar cliente de testes do FastAPI usando context manager para ativar o lifespan
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_ctx.__exit__(None, None, None)

    def test_root_endpoint(self) -> None:
        """Garante que o endpoint raiz retorna informações válidas da API."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "API Antifraude PIX")
        self.assertIn("version", data)
        self.assertIn("endpoints", data)

    def test_health_endpoint(self) -> None:
        """Garante que o endpoint de health check responde com sucesso."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("pipeline_version", data)
        self.assertIn("components", data)
        self.assertIn("metrics", data)

    def test_status_endpoint(self) -> None:
        """Garante que o endpoint de status detalhado retorna configurações e estado da engine."""
        response = self.client.get("/api/v1/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("pipeline", data)
        self.assertIn("engine", data)
        self.assertIn("config", data)
        self.assertIn("metrics", data)

    def test_metrics_endpoint(self) -> None:
        """Garante que o endpoint de métricas retorna o dicionário com contadores da API."""
        response = self.client.get("/api/v1/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_requests", data)
        self.assertIn("total_transactions", data)
        self.assertIn("latency_avg_ms", data)

    def test_analyze_valid_transaction(self) -> None:
        """Garante que a análise de uma transação válida retorna a estrutura correta."""
        # Payload com features v3 completas para simular o comportamento integrado
        payload = {
            "cd_pix": "E0000000020260610120000000000999",
            "dt_pix": "2026-06-10 12:00:00",
            "cd_cpf_pagador": "98765432101",
            "cd_cpf_cnpj_recebedor": "12345678909",
            "ds_chave_pix": "+5561988888888",
            "ds_tipo_chave": "TELEFONE",
            "vl_pix": 250.00,
            "device_name": "iPhone14,2",
            "app_version": "7.30.1",
            "ip_address": "177.100.20.10",
            "latencia_rede_ms": 50.0,
            "tempo_interacao_ms": 2500.0,
            "tempo_processamento_host_ms": 40.0,
            "metodo_autenticacao": "senha",
            "session_id": "b2c3d4e5-f6a7-8b9c-0d1e-2f3a4b5c6d7e",
            "topaz_risk_score": 0.1,
            "topaz_transacao_rejeitada": 0.0,
            "is_agendamento_recorrente": "false",
            "vl_renda_cliente": 5000.00,
            "ds_sexo": "F",
            "ds_estado_civil": "CASADO",
            "ds_segmento": "VAREJO",
            "qt_dependentes": 1.0,
            # Features v3
            "qtd_pix_pagador_7d": 1.0,
            "qtd_pix_pagador_30d": 5.0,
            "qtd_pix_pagador_90d": 12.0,
            "qtd_pix_pagador_180d": 25.0,
            "valor_total_pagador_7d": 150.00,
            "valor_total_pagador_30d": 750.00,
            "valor_total_pagador_90d": 1800.00,
            "valor_total_pagador_180d": 3750.00,
            "max_qtd_pix_dia_pagador_7d": 1.0,
            "max_qtd_pix_dia_pagador_30d": 2.0,
            "valor_maximo_pix_pagador_180d": 300.00,
            "soma_recebedores_distintos_dia_180d": 5.0,
            "qtd_pix_mesmo_recebedor_30d": 1.0,
            "qtd_pix_mesmo_recebedor_90d": 2.0,
            "qtd_pix_mesmo_recebedor_180d": 3.0,
            "valor_total_para_recebedor_30d": 100.00,
            "valor_total_para_recebedor_90d": 200.00,
            "valor_total_para_recebedor_180d": 300.00,
            "primeira_data_envio_recebedor_180d": "2026-03-10 10:00:00",
            "primeiro_envio_para_recebedor_180d": 0.0,
            "dias_desde_primeiro_envio_recebedor": 92.0,
            "qtd_pix_recebidos_30d": 10.0,
            "qtd_pix_recebidos_90d": 30.0,
            "qtd_pix_recebidos_180d": 60.0,
            "valor_total_recebido_30d": 3000.00,
            "valor_total_recebido_90d": 9000.00,
            "valor_total_recebido_180d": 18000.00,
            "soma_pagadores_distintos_dia_recebedor_180d": 15.0,
            "max_qtd_pix_recebidos_dia_180d": 3.0,
            "burst_daily_7d_flag": 0.0,
            "first_receiver_flag_real": 0.0,
            "ratio_valor_media_pagador_90d": 1.25,
            "ratio_valor_maximo_pagador_180d": 0.83
        }
        
        response = self.client.post("/api/v1/analyze", json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("decisao", data)
        self.assertIn("score_final", data)
        self.assertIn("componentes", data)
        self.assertIn(data["decisao"], {"APROVAR", "CONFIRMAR", "BLOQUEAR"})

    def test_analyze_invalid_transaction(self) -> None:
        """Garante que a API rejeita payloads malformados ou com valores inválidos com status 422."""
        # vl_pix <= 0 é inválido
        payload = {
            "cd_pix": "E0000000020260610120000000000999",
            "dt_pix": "2026-06-10 12:00:00",
            "cd_cpf_pagador": "98765432101",
            "vl_pix": -100.00
        }
        response = self.client.post("/api/v1/analyze", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_batch_inference(self) -> None:
        """Garante que o processamento em lote (batch) de transações responde corretamente."""
        payload = {
            "transactions": [
                {
                    "cd_pix": "E0000000020260610120000000000005",
                    "dt_pix": "2026-06-10 12:00:00",
                    "cd_cpf_pagador": "98765432101",
                    "vl_pix": 50.00,
                    # Adicionar features mínimas para evitar NaNs/falhas se o preprocessor for rígido
                    "qtd_pix_pagador_7d": 1.0,
                    "qtd_pix_pagador_30d": 2.0,
                    "qtd_pix_pagador_90d": 5.0,
                    "qtd_pix_pagador_180d": 10.0,
                    "valor_total_pagador_7d": 50.0,
                    "valor_total_pagador_30d": 100.0,
                    "valor_total_pagador_90d": 250.0,
                    "valor_total_pagador_180d": 500.0,
                    "max_qtd_pix_dia_pagador_7d": 1.0,
                    "max_qtd_pix_dia_pagador_30d": 1.0,
                    "valor_maximo_pix_pagador_180d": 50.0,
                    "soma_recebedores_distintos_dia_180d": 1.0,
                    "qtd_pix_mesmo_recebedor_30d": 1.0,
                    "qtd_pix_mesmo_recebedor_90d": 1.0,
                    "qtd_pix_mesmo_recebedor_180d": 1.0,
                    "valor_total_para_recebedor_30d": 50.0,
                    "valor_total_para_recebedor_90d": 50.0,
                    "valor_total_para_recebedor_180d": 50.0,
                    "primeira_data_envio_recebedor_180d": "2026-06-10 12:00:00",
                    "primeiro_envio_para_recebedor_180d": 0.0,
                    "dias_desde_primeiro_envio_recebedor": 0.0,
                    "qtd_pix_recebidos_30d": 1.0,
                    "qtd_pix_recebidos_90d": 1.0,
                    "qtd_pix_recebidos_180d": 1.0,
                    "valor_total_recebido_30d": 50.0,
                    "valor_total_recebido_90d": 50.0,
                    "valor_total_recebido_180d": 50.0,
                    "soma_pagadores_distintos_dia_recebedor_180d": 1.0,
                    "max_qtd_pix_recebidos_dia_180d": 1.0,
                    "burst_daily_7d_flag": 0.0,
                    "first_receiver_flag_real": 0.0,
                    "ratio_valor_media_pagador_90d": 1.0,
                    "ratio_valor_maximo_pagador_180d": 1.0
                }
            ]
        }
        response = self.client.post("/api/v1/batch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total"], 1)
        self.assertIn("resultados", data)
        self.assertIn("resumo", data)

    def test_cache_reset(self) -> None:
        """Garante que a limpeza de cache responde com sucesso."""
        response = self.client.post("/api/v1/cache/reset")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("customers_removed", data)


if __name__ == "__main__":
    unittest.main()
