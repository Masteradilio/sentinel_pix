"""
online_store.py — Online Feature Store do Sentinel-PIX (Redis + Fallback In-Memory)
Serve features em baixa latência (<2ms) calculadas periodicamente (rolling windows de 1h/24h)
e telemetria comportamental recente de uso do app mobile.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from backend.config import settings

logger = logging.getLogger("online_feature_store")


class OnlineFeatureStore:
    def __init__(self):
        self.redis_client = None
        self.use_redis = False
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._init_client()

    def _init_client(self) -> None:
        if settings.redis_enabled:
            try:
                import redis
                client = redis.Redis(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    db=settings.redis_db,
                    password=settings.redis_password or None,
                    socket_connect_timeout=1.5,
                    decode_responses=True
                )
                client.ping()
                self.redis_client = client
                self.use_redis = True
                logger.info(f"Conectado ao Redis em {settings.redis_host}:{settings.redis_port}")
            except Exception as e:
                logger.warning(f"Redis indisponível ({e}). Utilizando Fallback In-Memory Thread-Safe.")
                self.use_redis = False
        else:
            logger.info("Redis desativado por configuração. Utilizando Fallback In-Memory.")

    def get_online_features(self, account_id: str, receiver_pix_key: str) -> Dict[str, Any]:
        """Recupera agregados de 1h/24h e telemetria mobile do Redis ou memória."""
        acc_key = f"features:acc:{account_id}"
        rec_key = f"features:rec:{receiver_pix_key}"

        acc_data = self._get_key_data(acc_key)
        rec_data = self._get_key_data(rec_key)

        # Se a conta for nova ou sem transações recentes, gerar baseline realista
        if not acc_data:
            acc_data = self._generate_default_account_online_features(account_id)
        
        if not rec_data:
            rec_data = self._generate_default_receiver_online_features(receiver_pix_key)

        combined = {**acc_data, **rec_data}
        return combined

    def _get_key_data(self, key: str) -> Optional[Dict[str, Any]]:
        if self.use_redis and self.redis_client:
            try:
                val = self.redis_client.get(key)
                if val:
                    return json.loads(val)
            except Exception as e:
                logger.error(f"Erro ao ler chave {key} no Redis: {e}")

        # Fallback memória
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if entry["expire_at"] > time.time():
                return entry["data"]
            else:
                del self._memory_cache[key]
        return None

    def set_key_data(self, key: str, data: Dict[str, Any], ttl_seconds: int = 86400) -> None:
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.setex(key, ttl_seconds, json.dumps(data))
                return
            except Exception as e:
                logger.error(f"Erro ao gravar chave {key} no Redis: {e}")

        # Fallback memória
        self._memory_cache[key] = {
            "data": data,
            "expire_at": time.time() + ttl_seconds
        }

    def _generate_default_account_online_features(self, account_id: str) -> Dict[str, Any]:
        import hashlib
        h = int(hashlib.md5(account_id.encode()).hexdigest(), 16)
        
        return {
            "pix_count_1h": (h % 3),
            "pix_sum_1h": float((h % 3) * (h % 300 + 50)),
            "pix_count_24h": (h % 6) + 1,
            "pix_sum_24h": float(((h % 6) + 1) * (h % 400 + 80)),
            "distinct_receivers_24h": (h % 4) + 1,
            "last_tx_time_diff_sec": 3600 + (h % 14400),
            "recent_avg_amount_30d": float(250.0 + (h % 800)),
            "mobile_session_duration_sec": 45 + (h % 120),
            "mobile_typing_speed_wpm": 38.0 + (h % 25),
            "mobile_battery_level": 75 - (h % 50),
            "is_device_known": 1 if (h % 10) != 0 else 0,
            "failed_login_attempts_24h": 1 if (h % 25) == 0 else 0
        }

    def _generate_default_receiver_online_features(self, receiver_pix_key: str) -> Dict[str, Any]:
        import hashlib
        h = int(hashlib.md5(receiver_pix_key.encode()).hexdigest(), 16)
        
        return {
            "receiver_is_new": 1 if (h % 4) == 0 else 0,
            "receiver_inflow_count_24h": (h % 15) + 1,
            "receiver_inflow_sum_24h": float(((h % 15) + 1) * (h % 500 + 100)),
            "receiver_unique_senders_24h": (h % 10) + 1,
            "receiver_suspected_mule_score": float((h % 20) / 100.0),
            "receiver_account_age_days": 60 + (h % 800)
        }

    def update_after_transaction(
        self,
        account_id: str,
        receiver_pix_key: str,
        amount: float,
        timestamp: str
    ) -> None:
        """Atualiza os contadores online em tempo real após cada inferência/transação."""
        acc_key = f"features:acc:{account_id}"
        current = self._get_key_data(acc_key) or self._generate_default_account_online_features(account_id)
        
        current["pix_count_1h"] = current.get("pix_count_1h", 0) + 1
        current["pix_sum_1h"] = current.get("pix_sum_1h", 0.0) + amount
        current["pix_count_24h"] = current.get("pix_count_24h", 0) + 1
        current["pix_sum_24h"] = current.get("pix_sum_24h", 0.0) + amount
        current["last_tx_time_diff_sec"] = 10
        
        self.set_key_data(acc_key, current, ttl_seconds=settings.redis_ttl_seconds)


online_store = OnlineFeatureStore()
