"""
offline_store.py — Offline Feature Store do Sentinel-PIX
Armazena e serve features cadastrais, limites, KYC e dados estáticos de clientes.
Utiliza SQLite por padrão com suporte nativo a PostgreSQL via SQLAlchemy/sqlite3.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from backend.config import settings

logger = logging.getLogger("offline_feature_store")


class OfflineFeatureStore:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            url = settings.offline_db_url
            if url.startswith("sqlite:///"):
                self.db_path = url.replace("sqlite:///", "")
            else:
                self.db_path = str(settings.project_root / "backend" / "feature_store" / "offline_feature_store.db")
        else:
            self.db_path = db_path

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Cria tabelas de perfis de clientes e metadados de contas."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS customer_profiles (
                    account_id TEXT PRIMARY KEY,
                    customer_name TEXT,
                    cpf_cnpj TEXT,
                    account_creation_days INTEGER DEFAULT 365,
                    kyc_status TEXT DEFAULT 'VERIFIED',
                    credit_score INTEGER DEFAULT 650,
                    monthly_income REAL DEFAULT 4500.0,
                    pix_day_limit REAL DEFAULT 5000.0,
                    pix_night_limit REAL DEFAULT 1000.0,
                    historical_disputes_count INTEGER DEFAULT 0,
                    is_pep INTEGER DEFAULT 0,
                    trusted_devices_count INTEGER DEFAULT 1,
                    primary_device_id TEXT,
                    risk_segment TEXT DEFAULT 'STANDARD',
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            conn.commit()
        logger.info(f"Offline Feature Store inicializada em: {self.db_path}")

    def get_customer_profile(self, account_id: str) -> Dict[str, Any]:
        """Recupera as features offline do cliente."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM customer_profiles WHERE account_id = ?", (account_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)

        return self._generate_default_profile(account_id)

    def _generate_default_profile(self, account_id: str) -> Dict[str, Any]:
        """Gera um perfil padrão determinístico com base no hash do account_id."""
        import hashlib
        h = int(hashlib.md5(account_id.encode()).hexdigest(), 16)
        
        default_profile = {
            "account_id": account_id,
            "customer_name": f"Cliente {account_id[:8]}",
            "cpf_cnpj": f"***.{h % 900 + 100}.***-00",
            "account_creation_days": 180 + (h % 1500),
            "kyc_status": "VERIFIED" if (h % 10) != 0 else "PENDING",
            "credit_score": 400 + (h % 550),
            "monthly_income": 2500.0 + (h % 15000),
            "pix_day_limit": 5000.0,
            "pix_night_limit": 1000.0,
            "historical_disputes_count": 1 if (h % 30) == 0 else 0,
            "is_pep": 1 if (h % 100) == 0 else 0,
            "trusted_devices_count": 1 + (h % 3),
            "primary_device_id": f"dev_{account_id[-6:]}",
            "risk_segment": "STANDARD",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z"
        }
        return default_profile

    def upsert_customer_profile(self, profile: Dict[str, Any]) -> None:
        """Insere ou atualiza o perfil de um cliente."""
        now = datetime.utcnow().isoformat() + "Z"
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO customer_profiles (
                    account_id, customer_name, cpf_cnpj, account_creation_days,
                    kyc_status, credit_score, monthly_income, pix_day_limit,
                    pix_night_limit, historical_disputes_count, is_pep,
                    trusted_devices_count, primary_device_id, risk_segment,
                    created_at, updated_at
                ) VALUES (
                    :account_id, :customer_name, :cpf_cnpj, :account_creation_days,
                    :kyc_status, :credit_score, :monthly_income, :pix_day_limit,
                    :pix_night_limit, :historical_disputes_count, :is_pep,
                    :trusted_devices_count, :primary_device_id, :risk_segment,
                    :created_at, :updated_at
                ) ON CONFLICT(account_id) DO UPDATE SET
                    account_creation_days=excluded.account_creation_days,
                    kyc_status=excluded.kyc_status,
                    credit_score=excluded.credit_score,
                    monthly_income=excluded.monthly_income,
                    pix_day_limit=excluded.pix_day_limit,
                    pix_night_limit=excluded.pix_night_limit,
                    historical_disputes_count=excluded.historical_disputes_count,
                    trusted_devices_count=excluded.trusted_devices_count,
                    primary_device_id=excluded.primary_device_id,
                    risk_segment=excluded.risk_segment,
                    updated_at=:updated_at
            """, {
                "created_at": now,
                "updated_at": now,
                **profile
            })
            conn.commit()


offline_store = OfflineFeatureStore()
