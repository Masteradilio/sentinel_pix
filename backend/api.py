"""
api.py v2.0 — API REST Sentinel-PIX (Real-Time Ingestion, Dual Feature Store & MLOps)
Suporta payload leve (6-8 features) com enriquecimento automatico via
Offline Feature Store (SQL) e Online Feature Store (Redis),
calculo de explicabilidade SHAP, logging de auditoria e monitoramento de Drift.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from backend.config import settings
from backend.feature_store.offline_store import offline_store
from backend.feature_store.online_store import online_store
from backend.mlops.audit_logger import audit_logger
from backend.mlops.drift_detector import drift_detector
from backend.mlops.mlflow_tracker import mlflow_tracker

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")


class PixTransactionRequest(BaseModel):
    transaction_id: str = Field(default_factory=lambda: f"tx_{uuid.uuid4().hex[:12]}", description="ID unico da transacao")
    account_id: str = Field(..., description="Identificador da conta de origem (pagador)")
    receiver_pix_key: str = Field(..., description="Chave PIX de destino (recebedor)")
    receiver_key_type: str = Field(default="CPF", description="Tipo da chave PIX (CPF, CNPJ, EMAIL, PHONE, EVP)")
    amount: float = Field(..., gt=0.0, description="Valor da transferencia em R$")
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z", description="Timestamp ISO 8601")
    device_id: Optional[str] = Field(default=None, description="ID do dispositivo mobile")
    channel: str = Field(default="MOBILE_APP", description="Canal de origem (MOBILE_APP, INTERNET_BANKING, API)")
    explain: bool = Field(default=True, description="Se True, calcula SHAP")
    extra_features: Optional[Dict[str, Any]] = Field(default=None, description="Features adicionais para override")


class UpdateCaseStatusRequest(BaseModel):
    status: str = Field(..., description="Novo status (APPROVED_BY_ANALYST, CONFIRMED_FRAUD, ARCHIVED)")
    notes: Optional[str] = Field(default="", description="Parecer tecnico do analista de fraude")


class PipelineStatus:
    def __init__(self):
        self.started_at: str = datetime.utcnow().isoformat() + "Z"
        self.total_requests: int = 0
        self.total_aprovados: int = 0
        self.total_confirmados: int = 0
        self.total_bloqueados: int = 0
        self.total_errors: int = 0
        self.latency_samples: List[float] = []

    def record(self, decision: str, latency_ms: float):
        self.total_requests += 1
        d = decision.upper()
        if d == "APROVAR":
            self.total_aprovados += 1
        elif d == "CONFIRMAR":
            self.total_confirmados += 1
        elif d == "BLOQUEAR":
            self.total_bloqueados += 1
        
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > 2000:
            self.latency_samples = self.latency_samples[-1000:]


api_metrics = PipelineStatus()
pipeline = None


def _load_pipeline():
    global pipeline
    from backend.core.pipeline_orquestrador import PipelineOrquestrador
    from backend.core.decision_engine import EngineConfig

    engine_config = EngineConfig(
        artefatos_dir=str(settings.artefatos_dir),
        threshold_confirmar=settings.threshold_confirmar,
        threshold_bloquear=settings.threshold_bloquear,
        veto_threshold=settings.veto_threshold,
    )
    pipeline = PipelineOrquestrador(
        artefatos_dir=str(settings.artefatos_dir),
        engine_config=engine_config,
    )
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Iniciando {settings.app_name} v{settings.app_version}...")
    _load_pipeline()
    mlflow_tracker.log_baseline_r5b22()
    logger.info("Motor e MLOps inicializados com sucesso!")
    yield
    logger.info("Encerrando Sentinel-PIX API...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Motor Hibrido de Deteccao de Fraude em Pagamentos PIX em Tempo Real",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _enrich_transaction(req: PixTransactionRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()

    offline_feats = offline_store.get_customer_profile(req.account_id)
    online_feats = online_store.get_online_features(req.account_id, req.receiver_pix_key)

    ts = req.timestamp or datetime.utcnow().isoformat() + "Z"
    hour = 12
    minute = 0
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        hour = dt.hour
        minute = dt.minute
    except Exception:
        pass

    is_night = 1 if (hour >= 20 or hour < 6) else 0
    amount = float(req.amount)
    
    day_limit = float(offline_feats.get("pix_day_limit", 5000.0))
    night_limit = float(offline_feats.get("pix_night_limit", 1000.0))
    current_limit = night_limit if is_night else day_limit
    limit_utilization = round(amount / max(current_limit, 1.0), 4)

    enriched = {
        "id_transacao": req.transaction_id,
        "id_cliente": req.account_id,
        "chave_pix": req.receiver_pix_key,
        "tipo_chave": req.receiver_key_type,
        "vl_transacao": amount,
        "dt_transacao": ts,
        "hora_transacao": hour,
        "minuto_transacao": minute,
        "is_horario_noturno": is_night,
        "canal_transacao": req.channel,
        "dispositivo_id": req.device_id or offline_feats.get("primary_device_id", "dev_unknown"),
        
        "idade_conta_dias": offline_feats.get("account_creation_days", 365),
        "score_credito": offline_feats.get("credit_score", 650),
        "renda_mensal": offline_feats.get("monthly_income", 4500.0),
        "limite_pix_diurno": day_limit,
        "limite_pix_noturno": night_limit,
        "tx_utilizacao_limite": limit_utilization,
        "historico_contestações": offline_feats.get("historical_disputes_count", 0),
        "is_pep": offline_feats.get("is_pep", 0),
        "qtd_dispositivos_confiaveis": offline_feats.get("trusted_devices_count", 1),
        
        "qt_pix_1h": online_feats.get("pix_count_1h", 0),
        "vl_pix_1h": online_feats.get("pix_sum_1h", 0.0),
        "qt_pix_24h": online_feats.get("pix_count_24h", 1),
        "vl_pix_24h": online_feats.get("pix_sum_24h", amount),
        "qt_recebedores_distintos_24h": online_feats.get("distinct_receivers_24h", 1),
        "tempo_desde_ultima_tx_seg": online_feats.get("last_tx_time_diff_sec", 3600),
        "vl_medio_30d": online_feats.get("recent_avg_amount_30d", 300.0),
        "duracao_sessao_app_seg": online_feats.get("mobile_session_duration_sec", 60),
        "velocidade_digitacao_wpm": online_feats.get("mobile_typing_speed_wpm", 40.0),
        "nivel_bateria_aparelho": online_feats.get("mobile_battery_level", 80),
        "is_dispositivo_conhecido": online_feats.get("is_device_known", 1),
        "falhas_login_24h": online_feats.get("failed_login_attempts_24h", 0),

        "recebedor_is_novo": online_feats.get("receiver_is_new", 0),
        "recebedor_qtd_entradas_24h": online_feats.get("receiver_inflow_count_24h", 1),
        "recebedor_vl_entradas_24h": online_feats.get("receiver_inflow_sum_24h", amount),
        "recebedor_mule_score": online_feats.get("receiver_suspected_mule_score", 0.0),
        "recebedor_idade_conta_dias": online_feats.get("receiver_account_age_days", 180)
    }

    if req.extra_features:
        enriched.update(req.extra_features)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    enriched["_feature_enrichment_time_ms"] = round(elapsed_ms, 2)
    return enriched


@app.get("/", tags=["Info"])
def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "online",
        "docs": "/docs",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.post("/api/v1/analyze", tags=["Inferencia"])
def analyze_transaction(req: PixTransactionRequest, background_tasks: BackgroundTasks):
    t0 = time.perf_counter()

    if pipeline is None:
        _load_pipeline()

    try:
        enriched_data = _enrich_transaction(req)
        res = pipeline.analisar(enriched_data)

        decisao = res.get("decisao", "APROVAR")
        score_final = float(res.get("score_final", 0.0))
        confianca = res.get("confianca", "MEDIA")

        explicabilidade = res.get("explicabilidade", {})
        if "shap" in res and res["shap"]:
            explicabilidade["shap_top_features"] = res["shap"].get("top_features", {})

        response_payload = {
            "transaction_id": req.transaction_id,
            "account_id": req.account_id,
            "receiver_pix_key": req.receiver_pix_key,
            "amount": req.amount,
            "decisao": decisao,
            "score_final": score_final,
            "confianca": confianca,
            "explicabilidade": explicabilidade,
            "metadata": {
                "engine_version": settings.app_version,
                "enrichment_latency_ms": enriched_data.get("_feature_enrichment_time_ms", 0.0),
                "r5b22_policy_applied": res.get("r5b22_policy_applied"),
                "r5b22_rule_applied": res.get("r5b22_rule_applied"),
                "veto_aplicado": res.get("veto_aplicado")
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        background_tasks.add_task(audit_logger.log_decision, response_payload, req.model_dump())
        background_tasks.add_task(drift_detector.add_observation, req.model_dump(), response_payload)
        background_tasks.add_task(
            online_store.update_after_transaction,
            req.account_id,
            req.receiver_pix_key,
            req.amount,
            req.timestamp or ""
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        response_payload["metadata"]["total_latency_ms"] = round(elapsed_ms, 2)
        api_metrics.record(decisao, elapsed_ms)

        return response_payload

    except Exception as e:
        logger.exception(f"Erro ao processar transacao {req.transaction_id}: {e}")
        api_metrics.total_errors += 1
        raise HTTPException(status_code=500, detail=f"Erro interno no motor: {str(e)}")


@app.post("/api/v1/batch", tags=["Inferencia"])
def batch_analyze(transactions: List[PixTransactionRequest]):
    results = []
    for tx in transactions:
        enriched = _enrich_transaction(tx)
        res = pipeline.analisar(enriched)
        results.append({
            "transaction_id": tx.transaction_id,
            "decisao": res.get("decisao"),
            "score_final": res.get("score_final"),
            "confianca": res.get("confianca"),
            "motivo_principal": res.get("explicabilidade", {}).get("motivo_principal", "")
        })
    return {"total": len(results), "results": results}


@app.get("/api/v1/cases", tags=["Mesa de Fraude & Auditoria"])
def list_investigation_cases(limit: int = 50, status: Optional[str] = None):
    cases = audit_logger.list_cases(limit=limit, status=status)
    return {"total": len(cases), "cases": cases}


@app.post("/api/v1/cases/{case_id}/action", tags=["Mesa de Fraude & Auditoria"])
def update_case_action(case_id: str, payload: UpdateCaseStatusRequest):
    success = audit_logger.update_case_status(case_id, payload.status, payload.notes or "")
    if not success:
        raise HTTPException(status_code=404, detail="Caso nao encontrado")
    return {"status": "success", "case_id": case_id, "new_status": payload.status}


@app.get("/api/v1/drift", tags=["MLOps & Observabilidade"])
def get_drift_metrics():
    return drift_detector.get_drift_report()


@app.get("/api/v1/metrics", tags=["MLOps & Observabilidade"])
def get_api_metrics():
    latencies = api_metrics.latency_samples
    p50 = round(float(sorted(latencies)[len(latencies)//2]), 2) if latencies else 0.0
    p95 = round(float(sorted(latencies)[int(len(latencies)*0.95)]), 2) if latencies else 0.0
    p99 = round(float(sorted(latencies)[int(len(latencies)*0.99)]), 2) if latencies else 0.0

    return {
        "started_at": api_metrics.started_at,
        "total_requests": api_metrics.total_requests,
        "decisions": {
            "aprovados": api_metrics.total_aprovados,
            "confirmados": api_metrics.total_confirmados,
            "bloqueados": api_metrics.total_bloqueados
        },
        "rates": {
            "approval_rate": round(api_metrics.total_aprovados / max(api_metrics.total_requests, 1) * 100, 2),
            "confirm_rate": round(api_metrics.total_confirmados / max(api_metrics.total_requests, 1) * 100, 2),
            "block_rate": round(api_metrics.total_bloqueados / max(api_metrics.total_requests, 1) * 100, 2),
        },
        "errors": api_metrics.total_errors,
        "latency_ms": {
            "p50": p50,
            "p95": p95,
            "p99": p99
        }
    }


@app.get("/api/v1/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "engine": "ready" if pipeline is not None else "initializing",
        "offline_store": "sqlite_connected",
        "online_store": "redis_connected" if online_store.use_redis else "memory_fallback_active",
        "mlflow": "tracking_active" if mlflow_tracker.enabled else "stub_mode",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
