"""
attack_scenarios.py — Arquétipos e Cenários de Ataque para o Simulador de Fraude PIX
Modela padrões criminosos reais (Engenharia Social, Mule Rings, Account Takeover, Bursts)
e transações normais legítimas para teste do motor.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime
from typing import Any, Dict


def generate_normal_transaction(account_ids: list[str]) -> Dict[str, Any]:
    """Gera uma transação legítima comum (baixo risco)."""
    acc_id = random.choice(account_ids) if account_ids else f"acc_{random.randint(100000, 100500)}"
    amount = round(random.choice([
        random.uniform(5.0, 80.0),    # Café, lanche, mercado
        random.uniform(80.0, 350.0),  # Contas, compras médias
        random.uniform(350.0, 1200.0) # Aluguel, serviços
    ]), 2)

    return {
        "transaction_id": f"tx_{uuid.uuid4().hex[:12]}",
        "account_id": acc_id,
        "receiver_pix_key": f"chave_legitima_{random.randint(100, 999)}@banco.com.br",
        "receiver_key_type": random.choice(["CPF", "EMAIL", "PHONE", "EVP"]),
        "amount": amount,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "channel": "MOBILE_APP",
        "device_id": f"dev_{acc_id[-6:]}",
        "scenario": "NORMAL_LEGITIMATE"
    }


def generate_fake_central_scam(account_ids: list[str]) -> Dict[str, Any]:
    """
    Golpe da Falsa Central Telefônica (Engenharia Social).
    Vítima é induzida por criminoso simulando central de segurança a transferir
    alto valor para uma 'conta segura' / chave desconhecida.
    """
    acc_id = random.choice(account_ids) if account_ids else f"acc_{random.randint(100000, 100500)}"
    amount = round(random.uniform(12500.0, 24000.0), 2)
    mule_idx = random.randint(1, 20)

    return {
        "transaction_id": f"tx_scam_{uuid.uuid4().hex[:10]}",
        "account_id": acc_id,
        "receiver_pix_key": f"mule_chave_pix_{mule_idx:03d}@pix.me",
        "receiver_key_type": "EVP",
        "amount": amount,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "channel": "MOBILE_APP",
        "device_id": f"dev_attacker_{random.randint(10, 99)}",
        "extra_features": {
            "topaz_risk_score": 0.96,
            "topaz_transacao_rejeitada": 1.0,
            "r4g_fast_frozen_decisao_recommended": "BLOQUEAR",
            "first_receiver_flag_real": 1.0,
            "se_worst_pattern": "FALSA_CENTRAL",
            "duracao_sessao_app_seg": 650,
            "velocidade_digitacao_wpm": 18.5,
            "recebedor_mule_score": 0.92
        },
        "scenario": "GOLPE_FALSA_CENTRAL"
    }


def generate_mule_ring_burst(account_ids: list[str]) -> Dict[str, Any]:
    """
    Ataque de Anel de Mulas (Mule Ring Burst).
    Transferências rápidas e repetidas fracionadas para esvaziar a conta da vítima.
    """
    acc_id = random.choice(account_ids) if account_ids else f"acc_{random.randint(100000, 100500)}"
    amount = round(random.uniform(3500.0, 5000.0), 2)
    mule_idx = random.randint(1, 10)

    return {
        "transaction_id": f"tx_mule_{uuid.uuid4().hex[:10]}",
        "account_id": acc_id,
        "receiver_pix_key": f"mule_chave_pix_{mule_idx:03d}@pix.me",
        "receiver_key_type": "CPF",
        "amount": amount,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "channel": "MOBILE_APP",
        "device_id": f"dev_mule_{random.randint(100, 999)}",
        "extra_features": {
            "topaz_risk_score": 0.91,
            "r4g_fast_frozen_decisao_recommended": "CONFIRMAR",
            "first_receiver_flag_real": 1.0,
            "qt_pix_1h": 5,
            "vl_pix_1h": 14500.0,
            "tempo_desde_ultima_tx_seg": 15,
            "recebedor_mule_score": 0.95
        },
        "scenario": "MULE_RING_BURST"
    }


def generate_night_drain(account_ids: list[str]) -> Dict[str, Any]:
    """
    Esvaziamento Noturno / Account Takeover.
    Acesso no meio da madrugada com dispositivo desconhecido tentando drenar limite.
    """
    acc_id = random.choice(account_ids) if account_ids else f"acc_{random.randint(100000, 100500)}"
    amount = round(random.uniform(980.0, 1200.0), 2)
    mule_idx = random.randint(11, 20)

    # Simular horário noturno (03:30 AM)
    now = datetime.utcnow()
    night_ts = now.replace(hour=3, minute=random.randint(10, 55)).isoformat() + "Z"

    return {
        "transaction_id": f"tx_night_{uuid.uuid4().hex[:10]}",
        "account_id": acc_id,
        "receiver_pix_key": f"mule_chave_pix_{mule_idx:03d}@pix.me",
        "receiver_key_type": "PHONE",
        "amount": amount,
        "timestamp": night_ts,
        "channel": "MOBILE_APP",
        "device_id": f"dev_foreign_{uuid.uuid4().hex[:6]}",
        "extra_features": {
            "is_horario_noturno": 1,
            "hora_transacao": 3,
            "is_dispositivo_conhecido": 0,
            "falhas_login_24h": 3,
            "topaz_risk_score": 0.89,
            "first_receiver_flag_real": 1.0,
            "r4g_fast_frozen_decisao_recommended": "BLOQUEAR"
        },
        "scenario": "NIGHT_DRAIN_ATO"
    }
