#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_api_smoke.py — Testes de Fumaça HTTP para a API REST Sentinel-PIX v2.0
"""

import sys
from pathlib import Path
import unittest
from fastapi.testclient import TestClient

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
CORE_DIR = BACKEND_DIR / "core"
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from backend.api import app


class TestApiSmoke(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
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
        self.assertIn("service", data)
        self.assertIn("version", data)
        self.assertEqual(data["status"], "online")

    def test_health_endpoint(self) -> None:
        """Garante que o endpoint de health check responde com sucesso."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("engine", data)
        self.assertIn("online_store", data)

    def test_metrics_endpoint(self) -> None:
        """Garante que o endpoint de métricas retorna o dicionário com contadores da API."""
        response = self.client.get("/api/v1/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_requests", data)
        self.assertIn("decisions", data)
        self.assertIn("latency_ms", data)

    def test_drift_endpoint(self) -> None:
        """Garante que o endpoint de drift retorna métricas estatísticas."""
        response = self.client.get("/api/v1/drift")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("max_psi", data)

    def test_analyze_valid_transaction(self) -> None:
        """Garante que a análise de uma transação válida com payload leve retorna a estrutura correta."""
        payload = {
            "account_id": "acc_100001",
            "receiver_pix_key": "recebedor_comum@pix.me",
            "amount": 250.00,
            "channel": "MOBILE_APP"
        }
        response = self.client.post("/api/v1/analyze", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("decisao", data)
        self.assertIn("score_final", data)
        self.assertIn("explicabilidade", data)
        self.assertIn(data["decisao"], {"APROVAR", "CONFIRMAR", "BLOQUEAR"})

    def test_analyze_invalid_transaction(self) -> None:
        """Garante que a API rejeita payloads malformados ou com valores inválidos com status 422."""
        payload = {
            "account_id": "acc_100001",
            "receiver_pix_key": "recebedor@pix.me",
            "amount": -100.00
        }
        response = self.client.post("/api/v1/analyze", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_batch_inference(self) -> None:
        """Garante que o processamento em lote (batch) de transações responde corretamente."""
        payload = [
            {
                "account_id": "acc_100001",
                "receiver_pix_key": "rec1@pix.me",
                "amount": 50.00
            },
            {
                "account_id": "acc_100002",
                "receiver_pix_key": "rec2@pix.me",
                "amount": 120.00
            }
        ]
        response = self.client.post("/api/v1/batch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total"], 2)
        self.assertIn("results", data)


if __name__ == "__main__":
    unittest.main()
