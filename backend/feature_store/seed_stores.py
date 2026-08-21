"""
seed_stores.py — Popula o Offline e Online Feature Store com Dados Sintéticos
Gera perfis de clientes, limites e agregados comportamentais 100% sintéticos e realistas,
garantindo conformidade LGPD/GDPR e fornecendo um ambiente model-ready para testes e demonstrações.
"""

from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import List

from backend.feature_store.offline_store import offline_store
from backend.feature_store.online_store import online_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("seed_stores")

NAMES_FIRST = ["Lucas", "Mariana", "Gabriel", "Beatriz", "Matheus", "Larissa", "Felipe", "Camila", "Rodrigo", "Juliana", "Carlos", "Fernanda", "Bruno", "Patricia", "Eduardo"]
NAMES_LAST = ["Silva", "Santos", "Oliveira", "Souza", "Rodrigues", "Ferreira", "Alves", "Pereira", "Lima", "Gomes", "Costa", "Ribeiro", "Martins", "Carvalho", "Almeida"]


def generate_synthetic_customers(n: int = 500) -> List[str]:
    """Gera e insere n perfis sintéticos no Offline e Online Feature Store."""
    account_ids = []
    logger.info(f"Populando Feature Stores com {n} clientes sintéticos...")

    for i in range(1, n + 1):
        acc_id = f"acc_{100000 + i}"
        name = f"{random.choice(NAMES_FIRST)} {random.choice(NAMES_LAST)}"
        cpf = f"{random.randint(100,999)}.{random.randint(100,999)}.{random.randint(100,999)}-{random.randint(10,99)}"
        age_days = random.randint(30, 2500)
        income = round(random.lognormvariate(8.2, 0.6), 2)
        score = int(min(990, max(300, random.gauss(680, 120))))
        
        kyc = "VERIFIED"
        if random.random() < 0.04:
            kyc = "PENDING"
        elif random.random() < 0.01:
            kyc = "REJECTED"

        # Limites
        day_limit = round(income * random.uniform(0.8, 2.5), -2)
        night_limit = round(min(1000.0, day_limit * 0.2), -1)

        # Offline profile
        profile = {
            "account_id": acc_id,
            "customer_name": name,
            "cpf_cnpj": cpf,
            "account_creation_days": age_days,
            "kyc_status": kyc,
            "credit_score": score,
            "monthly_income": income,
            "pix_day_limit": day_limit,
            "pix_night_limit": night_limit,
            "historical_disputes_count": 1 if random.random() < 0.03 else 0,
            "is_pep": 1 if random.random() < 0.01 else 0,
            "trusted_devices_count": random.choice([1, 1, 2, 2, 3]),
            "primary_device_id": f"dev_{acc_id[-6:]}",
            "risk_segment": "PRIME" if income > 12000 else ("HIGH_RISK" if score < 450 else "STANDARD")
        }
        offline_store.upsert_customer_profile(profile)

        # Online profile (Redis / Memory)
        online_data = {
            "pix_count_1h": random.choices([0, 1, 2, 3, 5], weights=[0.85, 0.10, 0.03, 0.015, 0.005])[0],
            "pix_sum_1h": round(random.uniform(0, 800), 2),
            "pix_count_24h": random.randint(0, 6),
            "pix_sum_24h": round(random.uniform(50, 3500), 2),
            "distinct_receivers_24h": random.randint(1, 4),
            "last_tx_time_diff_sec": random.randint(300, 86400),
            "recent_avg_amount_30d": round(random.uniform(80, 1200), 2),
            "mobile_session_duration_sec": random.randint(20, 300),
            "mobile_typing_speed_wpm": round(random.uniform(25.0, 65.0), 1),
            "mobile_battery_level": random.randint(15, 100),
            "is_device_known": 1 if random.random() > 0.05 else 0,
            "failed_login_attempts_24h": 1 if random.random() < 0.02 else 0
        }
        online_store.set_key_data(f"features:acc:{acc_id}", online_data)
        account_ids.append(acc_id)

    # Criar e popular chaves de contas mulas (para testes de grafos / regras SE)
    logger.info("Populando 20 nós de contas mulas conhecidas para teste do Graph Engine...")
    for j in range(1, 21):
        mule_key = f"mule_chave_pix_{j:03d}@pix.me"
        mule_online = {
            "receiver_is_new": 1 if j <= 10 else 0,
            "receiver_inflow_count_24h": random.randint(25, 120),
            "receiver_inflow_sum_24h": round(random.uniform(15000, 180000), 2),
            "receiver_unique_senders_24h": random.randint(18, 90),
            "receiver_suspected_mule_score": round(random.uniform(0.75, 0.98), 2),
            "receiver_account_age_days": random.randint(5, 45)
        }
        online_store.set_key_data(f"features:rec:{mule_key}", mule_online)

    logger.info(f"Seed finalizado com sucesso! {len(account_ids)} contas e 20 chaves mulas carregadas.")
    return account_ids


if __name__ == "__main__":
    generate_synthetic_customers(500)
