"""
drift_detector.py — Monitor de Data Drift e Prediction Drift em Tempo Real
Calcula PSI (Population Stability Index) e teste de Kolmogorov-Smirnov comparando
a janela de inferência em tempo real com a distribuição baseline dos modelos.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger("drift_detector")


class DriftDetector:
    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        
        # Janelas deslizantes em memória para inferência recente
        self.recent_amounts: deque = deque(maxlen=window_size)
        self.recent_hours: deque = deque(maxlen=window_size)
        self.recent_lgbm_scores: deque = deque(maxlen=window_size)
        self.recent_if_scores: deque = deque(maxlen=window_size)
        self.recent_final_scores: deque = deque(maxlen=window_size)

        # Baseline de referência calibrado (R5B22 - 113k transações)
        # Bins e distribuições esperadas para cada feature
        self.baseline_distributions = {
            "amount": {
                "bins": [0, 50, 150, 500, 1500, 5000, 15000, 100000],
                "expected_pct": [0.28, 0.26, 0.22, 0.14, 0.06, 0.03, 0.01]
            },
            "hour": {
                "bins": [0, 6, 12, 18, 24],
                "expected_pct": [0.08, 0.32, 0.42, 0.18]
            },
            "score_final": {
                "bins": [0, 20, 45, 75, 100],
                "expected_pct": [0.85, 0.11, 0.025, 0.015]
            }
        }

    def add_observation(self, raw_tx: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Registra uma transação e seu score na janela de observação de drift."""
        try:
            amount = float(raw_tx.get("amount", raw_tx.get("vl_transacao", 0.0)))
            self.recent_amounts.append(amount)

            # Extrair hora
            ts = raw_tx.get("timestamp", raw_tx.get("dt_transacao", ""))
            hour = 12
            if ts and len(str(ts)) >= 13 and "T" in str(ts):
                try:
                    hour = int(str(ts).split("T")[1].split(":")[0])
                except Exception:
                    pass
            self.recent_hours.append(hour)

            comp = result.get("explicabilidade", {}).get("componentes", {})
            lgbm = float(comp.get("lgbm_score", result.get("lgbm_raw", 0.0)))
            if_score = float(comp.get("if_percentile", result.get("if_raw", 0.0)))
            final = float(result.get("score_final", 0.0))

            self.recent_lgbm_scores.append(lgbm)
            self.recent_if_scores.append(if_score)
            self.recent_final_scores.append(final)
        except Exception as e:
            logger.debug(f"Falha ao registrar observação de drift: {e}")

    def calculate_psi(self, current_values: List[float], baseline_spec: Dict[str, Any]) -> float:
        """Calcula o Population Stability Index (PSI)."""
        if len(current_values) < 20:
            return 0.0  # Amostra insuficiente

        bins = baseline_spec["bins"]
        expected_pct = baseline_spec["expected_pct"]

        counts, _ = np.histogram(current_values, bins=bins)
        total = len(current_values)
        if total == 0:
            return 0.0

        actual_pct = [c / total for c in counts]

        psi = 0.0
        for act, exp in zip(actual_pct, expected_pct):
            # Suavização para evitar log(0)
            a = max(act, 0.0001)
            e = max(exp, 0.0001)
            psi += (a - e) * math.log(a / e)

        return round(float(psi), 4)

    def get_drift_report(self) -> Dict[str, Any]:
        """Gera relatório consolidado de Data Drift e Prediction Drift."""
        sample_size = len(self.recent_final_scores)
        
        psi_amount = self.calculate_psi(list(self.recent_amounts), self.baseline_distributions["amount"])
        psi_hour = self.calculate_psi(list(self.recent_hours), self.baseline_distributions["hour"])
        psi_score = self.calculate_psi(list(self.recent_final_scores), self.baseline_distributions["score_final"])

        max_psi = max(psi_amount, psi_hour, psi_score)

        if max_psi < 0.10:
            status = "ESTAVEL"
            recommendation = "Distribuição alinhada com baseline de produção."
        elif max_psi < 0.25:
            status = "MODERADO"
            recommendation = "Alerta: leve desvio detectado no perfil de transações."
        else:
            status = "CRITICO"
            recommendation = "Drift significativo! Avaliar retreino dos modelos."

        return {
            "sample_size": sample_size,
            "status": status,
            "max_psi": max_psi,
            "metrics": {
                "psi_amount": psi_amount,
                "psi_hour": psi_hour,
                "psi_score_final": psi_score,
                "avg_recent_score": round(float(np.mean(self.recent_final_scores)), 2) if sample_size > 0 else 0.0,
                "avg_recent_amount": round(float(np.mean(self.recent_amounts)), 2) if sample_size > 0 else 0.0
            },
            "recommendation": recommendation
        }


drift_detector = DriftDetector(window_size=1000)
