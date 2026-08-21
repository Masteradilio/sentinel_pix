"""
config.py — Configurações Globais do Sentinel-PIX
Centraliza parâmetros de conexão com Offline Feature Store (SQL),
Online Feature Store (Redis), MLflow, Motor de Decisão e Auditoria.
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
ARTEFATOS_DIR = BACKEND_DIR / "artefatos"
FEATURE_STORE_DIR = BACKEND_DIR / "feature_store"
MLOPS_DIR = BACKEND_DIR / "mlops"

FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)
MLOPS_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    # --- Aplicação & API ---
    app_name: str = "Sentinel-PIX Anti-Fraud Hybrid Engine"
    app_version: str = "1.5.0-r5b22"
    environment: str = Field(default_factory=lambda: os.getenv("ENV", "development"))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    
    # --- Paths ---
    project_root: Path = PROJECT_ROOT
    backend_dir: Path = BACKEND_DIR
    artefatos_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("ARTEFATOS_DIR", str(ARTEFATOS_DIR)))
    )

    # --- Offline Feature Store (PostgreSQL / SQLite) ---
    offline_db_url: str = Field(
        default_factory=lambda: os.getenv(
            "OFFLINE_DB_URL",
            f"sqlite:///{FEATURE_STORE_DIR / 'offline_feature_store.db'}"
        )
    )

    # --- Online Feature Store (Redis) ---
    redis_host: str = Field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = Field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    redis_db: int = Field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    redis_password: str = Field(default_factory=lambda: os.getenv("REDIS_PASSWORD", ""))
    redis_enabled: bool = Field(default_factory=lambda: os.getenv("REDIS_ENABLED", "true").lower() == "true")
    redis_ttl_seconds: int = Field(default_factory=lambda: int(os.getenv("REDIS_TTL_SECONDS", "86400")))

    # --- Audit & Triage Database ---
    audit_db_url: str = Field(
        default_factory=lambda: os.getenv(
            "AUDIT_DB_URL",
            f"sqlite:///{FEATURE_STORE_DIR / 'fraud_investigation_cases.db'}"
        )
    )

    # --- MLflow & Observability ---
    mlflow_tracking_uri: str = Field(
        default_factory=lambda: os.getenv(
            "MLFLOW_TRACKING_URI",
            f"sqlite:///{MLOPS_DIR / 'mlflow.db'}"
        )
    )
    mlflow_experiment_name: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "sentinel-pix-production")
    )

    # --- Decision Engine Thresholds ---
    threshold_confirmar: float = Field(default_factory=lambda: float(os.getenv("THRESHOLD_CONFIRMAR", "45.0")))
    threshold_bloquear: float = Field(default_factory=lambda: float(os.getenv("THRESHOLD_BLOQUEAR", "75.0")))
    veto_threshold: float = Field(default_factory=lambda: float(os.getenv("VETO_THRESHOLD", "85.0")))


settings = Settings()
