import os
import csv
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from backend.core.graph_engineering import GraphInvestigationEngine

@pytest.fixture
def temp_csv_path():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = Path(f.name)
    yield path
    if path.exists():
        path.unlink()

@pytest.fixture
def enabled_engine(temp_csv_path, monkeypatch):
    monkeypatch.setenv("GRAPH_INVESTIGATION_ENABLED", "true")
    monkeypatch.setenv("GRAPH_INVESTIGATION_REPORT_PATH", str(temp_csv_path))
    engine = GraphInvestigationEngine()
    return engine

def test_engine_disabled_by_default(monkeypatch):
    monkeypatch.setenv("GRAPH_INVESTIGATION_ENABLED", "false")
    engine = GraphInvestigationEngine()
    assert engine.enabled is False
    assert not engine.report_path.exists()

def test_engine_creates_header(enabled_engine, temp_csv_path):
    assert temp_csv_path.exists()
    with open(temp_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert "transaction_id" in header
        assert "graph_fanout_score" in header

def test_engine_skips_approve_transactions(enabled_engine, temp_csv_path):
    tx_data = {"transaction_id": "1", "cd_cpf_pagador": "p1", "cd_cpf_cnpj_recebedor": "r1", "vl_pix": 100}
    res_data = {"decisao": "APROVAR", "score_final": 10}
    
    enabled_engine.process_transaction(tx_data, res_data)
    
    with open(temp_csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1 # Apenas o header

def test_engine_logs_blocked_transactions_with_metrics(enabled_engine, temp_csv_path):
    tx_data = {
        "transaction_id": "tx1",
        "customer_id": "payer1",
        "counterparty_id": "receiver1",
        "vl_pix": 5500,
        "event_datetime": datetime.utcnow().isoformat()
    }
    res_data = {"decisao": "BLOQUEAR", "score_final": 98, "r5b22_rule_applied": "RULE_X"}
    
    enabled_engine.process_transaction(tx_data, res_data)
    
    with open(temp_csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 2 # Header + 1 linha
        row = reader[1]
        assert row[0] == "tx1"
        assert row[2] == "payer1"
        assert row[3] == "receiver1"
        assert row[4] == "5500.0"
        assert row[5] == "BLOQUEAR"
        assert row[8] == "RULE_X"
        # Mule score deve ser > 0 porque o valor é > 5000 e decisão é BLOQUEAR
        mule_score = int(row[18])
        assert mule_score >= 30

def test_engine_tolerates_missing_fields(enabled_engine, temp_csv_path):
    tx_data = {"cd_pix": "tx2"} # Faltam fields
    res_data = {"decisao": "CONFIRMAR"}
    
    enabled_engine.process_transaction(tx_data, res_data)
    
    with open(temp_csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 2
        row = reader[1]
        assert row[0] == "tx2"
        assert row[2] == "unknown_payer"
        assert row[3] == "unknown_receiver"
