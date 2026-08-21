"""
test_sentinel_e2e.py — Testes End-to-End da nova arquitetura Sentinel-PIX
Valida:
  1. Leitura e fallback da Dual Feature Store (Offline SQL + Online Redis/Memory)
  2. Enriquecimento de payload leve (6-8 features)
  3. Inferência completa no motor com SHAP e decisões
  4. Gravação de auditoria para CONFIRMAR/BLOQUEAR
  5. Cálculo de Data Drift e PSI
"""

import pytest
from fastapi.testclient import TestClient
from backend.api import app, _enrich_transaction, PixTransactionRequest
from backend.feature_store.offline_store import offline_store
from backend.feature_store.online_store import online_store
from backend.mlops.audit_logger import audit_logger
from backend.mlops.drift_detector import drift_detector


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_offline_store_profile_generation():
    profile = offline_store.get_customer_profile("acc_test_999")
    assert profile is not None
    assert "account_id" in profile
    assert profile["account_id"] == "acc_test_999"
    assert profile["credit_score"] >= 300
    assert profile["monthly_income"] > 0


def test_online_store_features():
    features = online_store.get_online_features("acc_test_999", "chave_test_123")
    assert "pix_count_1h" in features
    assert "pix_sum_24h" in features
    assert "receiver_suspected_mule_score" in features


def test_transaction_enrichment():
    req = PixTransactionRequest(
        account_id="acc_test_101",
        receiver_pix_key="chave_destino_xyz@pix.com",
        amount=150.0
    )
    enriched = _enrich_transaction(req)
    assert enriched["id_cliente"] == "acc_test_101"
    assert enriched["vl_transacao"] == 150.0
    assert "idade_conta_dias" in enriched
    assert "score_credito" in enriched
    assert "qt_pix_1h" in enriched
    assert "_feature_enrichment_time_ms" in enriched


def test_api_analyze_light_payload(client):
    payload = {
        "account_id": "acc_100001",
        "receiver_pix_key": "recebedor_comum@pix.me",
        "amount": 75.50
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "decisao" in data
    assert data["decisao"] in ["APROVAR", "CONFIRMAR", "BLOQUEAR"]
    assert "score_final" in data
    assert "explicabilidade" in data
    assert "metadata" in data


def test_api_analyze_fraud_scenario_and_audit(client):
    # Simular cenário suspeito de alto valor para nova chave com esvaziamento
    payload = {
        "account_id": "acc_100005",
        "receiver_pix_key": "mule_chave_pix_001@pix.me",
        "amount": 48000.0,
        "extra_features": {
            "r4g_fast_frozen_decisao_recommended": "BLOQUEAR",
            "is_horario_noturno": 1
        }
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decisao"] in ["CONFIRMAR", "BLOQUEAR"]
    
    # Verificar se foi registrado na fila de auditoria
    cases_resp = client.get("/api/v1/cases")
    assert cases_resp.status_code == 200
    cases = cases_resp.json()["cases"]
    assert len(cases) > 0


def test_drift_metrics_endpoint(client):
    response = client.get("/api/v1/drift")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "max_psi" in data
    assert "metrics" in data


def test_health_check_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
