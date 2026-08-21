"""
generator.py — Gerador Contínuo de Tráfego de Transações PIX em Tempo Real
Gera fluxo de transações sintéticas com calibração estatística realista de produção:
- 95.0% Transações Legítimas (APROVAR - 950 em 1.000)
- 3.5% Transações com Fricção Inteligente (CONFIRMAR - 35 em 1.000)
- 1.5% Transações com Bloqueio Preventivo (BLOQUEAR - 15 em 1.000)
Totalizando exatamente 50 transações com interferência operacional e 950 legítimas em um lote de 1.000.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional
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
        self.tps = 2.0  # Transações por segundo
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
            # Mix Realista de Produção Bancária (95.0% / 3.5% / 1.5%)
            r = random.random()
            if r < 0.950:
                return generate_normal_transaction(self.account_pool)
            elif r < 0.985:
                return generate_mule_ring_burst(self.account_pool)
            elif r < 0.995:
                return generate_fake_central_scam(self.account_pool)
            else:
                return generate_night_drain(self.account_pool)

    def generate_batch_1000(self) -> List[Dict[str, Any]]:
        """
        Gera um lote calibrado de EXATAMENTE 1.000 transações:
        - 950 Legítimas (APROVAR - 95.0%)
        - 35 Mule Ring Burst (CONFIRMAR - 3.5%)
        - 10 Golpe da Falsa Central (BLOQUEAR - 1.0%)
        - 5 Esvaziamento Noturno (BLOQUEAR - 0.5%)
        Total: 950 APROVAR, 50 Casos de Interferência (35 CONFIRMAR + 15 BLOQUEAR).
        """
        batch = []
        for _ in range(950):
            batch.append(generate_normal_transaction(self.account_pool))
        for _ in range(35):
            batch.append(generate_mule_ring_burst(self.account_pool))
        for _ in range(10):
            batch.append(generate_fake_central_scam(self.account_pool))
        for _ in range(5):
            batch.append(generate_night_drain(self.account_pool))

        random.shuffle(batch)
        return batch

    def send_transaction(self, tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Envia a transação via HTTP para o endpoint /api/v1/analyze."""
        try:
            resp = requests.post(f"{self.api_url}/api/v1/analyze", json=tx, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"[{tx.get('scenario', 'TX')}] R$ {tx.get('amount', 0)} -> {data.get('decisao')} (Score: {data.get('score_final')})")
                return data
            else:
                logger.error(f"Erro da API: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.error(f"Falha de conexão com a API ({self.api_url}): {e}")
        return None

    def run_loop(self, max_transactions: Optional[int] = None, callback: Optional[Callable] = None) -> None:
        """Loop contínuo ou em lote de envio de transações."""
        self.running = True
        logger.info(f"Iniciando gerador de tráfego PIX em {self.api_url} (TPS: {self.tps})...")
        
        if max_transactions == 1000 and self.attack_mode is None:
            batch = self.generate_batch_1000()
            for tx in batch:
                if not self.running:
                    break
                result = self.send_transaction(tx)
                if callback and result:
                    callback(tx, result)
                time.sleep(max(0.01, 1.0 / max(self.tps, 0.1)))
        else:
            count = 0
            while self.running:
                tx = self.generate_single()
                result = self.send_transaction(tx)
                
                if callback and result:
                    callback(tx, result)

                count += 1
                if max_transactions and count >= max_transactions:
                    break

                time.sleep(max(0.01, 1.0 / max(self.tps, 0.1)))

        self.running = False
        logger.info("Gerador de tráfego finalizado.")

    def stop(self) -> None:
        self.running = False


generator = PixTrafficGenerator()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulador de Tráfego PIX Sentinel")
    parser.add_argument("--url", default="http://localhost:8000", help="URL da API")
    parser.add_argument("--tps", type=float, default=5.0, help="Transações por segundo")
    parser.add_argument("--scenario", default=None, choices=["NORMAL_LEGITIMATE", "GOLPE_FALSA_CENTRAL", "MULE_RING_BURST", "NIGHT_DRAIN_ATO"], help="Forçar cenário")
    parser.add_argument("--count", type=int, default=1000, help="Número de transações a enviar (default: 1000)")
    args = parser.parse_args()

    g = PixTrafficGenerator(api_url=args.url)
    g.tps = args.tps
    g.attack_mode = args.scenario
    g.run_loop(max_transactions=args.count)
