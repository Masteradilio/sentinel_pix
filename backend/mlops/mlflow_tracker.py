"""
mlflow_tracker.py — Integração MLOps com MLflow Tracking & Registry
Registra experimentos, hiperparâmetros, métricas oficiais R5B22,
artefatos serializados e rodadas de avaliação contínua.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from backend.config import settings

logger = logging.getLogger("mlflow_tracker")


class MLflowTracker:
    def __init__(self):
        self.enabled = False
        self.mlflow = None
        self._setup()

    def _setup(self) -> None:
        try:
            import mlflow
            self.mlflow = mlflow
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
            self.enabled = True
            logger.info(f"MLflow conectado: {settings.mlflow_tracking_uri} (Experimento: {settings.mlflow_experiment_name})")
        except Exception as e:
            logger.warning(f"MLflow indisponível ou falha de inicialização ({e}). Operando em modo stub.")
            self.enabled = False

    def log_baseline_r5b22(self) -> Optional[str]:
        """Registra a rodada oficial do baseline R5B22 no MLflow."""
        if not self.enabled or not self.mlflow:
            return None

        try:
            with self.mlflow.start_run(run_name="baseline-official-r5b22") as run:
                # 1. Tags & Parâmetros
                self.mlflow.set_tags({
                    "model_version": "1.5.0-r5b22",
                    "architecture": "Hybrid Ensemble (Distilled LightGBM + Isolation Forest + Graph + SE + BEH)",
                    "target_framework": "SPI / PIX Instant Payments",
                    "environment": settings.environment
                })

                self.mlflow.log_params({
                    "threshold_confirmar": settings.threshold_confirmar,
                    "threshold_bloquear": settings.threshold_bloquear,
                    "veto_threshold": settings.veto_threshold,
                    "num_features_catalog": 78,
                    "validation_transactions": 113844,
                    "confirmed_frauds": 1465
                })

                # 2. Métricas Oficiais
                self.mlflow.log_metrics({
                    "global_recall": 0.99863481,
                    "global_precision": 0.57621111,
                    "global_f1": 0.73076923,
                    "global_fpr": 0.00957474,
                    "block_precision": 0.65657479,
                    "block_recall": 0.99180887,
                    "block_f1": 0.79010332,
                    "block_fpr": 0.00676283,
                    "frauds_in_approve": 2,
                    "frauds_in_confirm": 10,
                    "frauds_in_block": 1453
                })

                # 3. Artefatos de Configuração
                policy_file = settings.artefatos_dir / "r5b22_official_baseline_policy.json"
                if policy_file.exists():
                    self.mlflow.log_artifact(str(policy_file))

                summary_file = settings.artefatos_dir / "r5b22_official_baseline_summary.json"
                if summary_file.exists():
                    self.mlflow.log_artifact(str(summary_file))

                logger.info(f"Baseline R5B22 registrado com sucesso no MLflow (Run ID: {run.info.run_id})")
                return run.info.run_id
        except Exception as e:
            logger.error(f"Erro ao registrar no MLflow: {e}")
            return None


mlflow_tracker = MLflowTracker()
