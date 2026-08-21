"""
generator.py — Gerador Contínuo de Tráfego de Transações PIX em Tempo Real
Gera fluxo contínuo de transações sintéticas com calibração estatística realista de produção:
- ~95.0% Transações Legítimas (APROVAR)
- ~3.5% Transações com Fricção Inteligente (CONFIRMAR - 2FA / Biometria)
- ~1.5% Transações com Bloqueio Preventivo Imediato (BLOQUEAR)
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, Dict, Optional
import requests

from backend.simulator.attack_scenarios import (
    generate_normal_transaction,
    generate_fake_central_scam,
    generate_mule_ring_burst,
    generate_night_drain
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("traffic_generator")


class PixTrafficGenerator:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip("/")
        self.running = False
        self.tps = 1.0  # Transações por segundo
        self.attack_mode: Optional[str] = None  # None = mix natural de produção
        self.account_pool = [f"acc_{100000 + i}" for i in range(1, 501)]

    def generate_single(self, force_scenario: Optional[str] = None) -> Dict[str, Any]:
        """Gera uma única transação baseada no cenário configurado."""
        scenario = force_scenario or self.attack_mode

        if scenario == "GOLPE_FALSA_CENTRAL":
            return generate_fake_central_scam(self.account_pool)
        elif scenario == "MULE_RING_BURST":
            return generate_mule_ring_burst(self.account_pool)
        elif scenario == "NIGHT_DRAIN_ATO":
            return generate_night_drain(self.account_pool)
        elif scenario == "NORMAL_LEGITIMATE":
            return generate_normal_transaction(self.account_pool)
        else:
            # Mix Realista de Produção Bancária:
            # 95.0% Legítimo (APROVAR)
            # 3.5% Step-up / 2FA (CONFIRMAR)
            # 1.5% Bloqueio Preventivo (BLOQUEAR)
            r = random.random()
            if r < 0.950:
                return generate_normal_transaction(self.account_pool)
            elif r < 0.985:
                return generate_mule_ring_burst(self.account_pool)
            elif r < 0.995:
                return generate_fake_central_scam(self.account_pool)
            else:
                return generate_night_drain(self.account_pool)

    def send_transaction(self, tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Envia a transação via HTTP para o endpoint /api/v1/analyze."""
        try:
            resp = requests.post(f"{self.api_url}/api/v1/analyze", json=tx, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"[{tx['scenario']}] R$ {tx['amount']} -> {data['decisao']} (Score: {data['score_final']})")
                return data
            else:
                logger.error(f"Erro da API: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.error(f"Falha de conexão com a API ({self.api_url}): {e}")
        return None

    def run_loop(self, max_transactions: Optional[int] = None, callback: Optional[Callable] = None) -> None:
        """Loop contínuo de envio de transações."""
        self.running = True
        logger.info(f"Iniciando gerador de tráfego PIX em {self.api_url} (TPS: {self.tps})...")
        count = 0

        while self.running:
            tx = self.generate_single()
            result = self.send_transaction(tx)
            
            if callback and result:
                callback(tx, result)

            count += 1
            if max_transactions and count >= max_transactions:
                break

            sleep_time = max(0.05, 1.0 / max(self.tps, 0.1))
            time.sleep(sleep_time)

        self.running = False
        logger.info("Gerador de tráfego finalizado.")

    def stop(self) -> None:
        self.running = False


generator = PixTrafficGenerator()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulador de Tráfego PIX Sentinel")
    parser.add_argument("--url", default="http://localhost:8000", help="URL da API")
    parser.add_argument("--tps", type=float, default=2.0, help="Transações por segundo")
    parser.add_argument("--scenario", default=None, choices=["NORMAL_LEGITIMATE", "GOLPE_FALSA_CENTRAL", "MULE_RING_BURST", "NIGHT_DRAIN_ATO"], help="Forçar cenário")
    parser.add_argument("--count", type=int, default=None, help="Número de transações a enviar")
    args = parser.parse_args()

    g = PixTrafficGenerator(api_url=args.url)
    g.tps = args.tps
    g.attack_mode = args.scenario
    g.run_loop(max_transactions=args.count)
