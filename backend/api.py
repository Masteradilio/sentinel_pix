"""
api.py v1.0 — API REST para Detecção de Fraude PIX

Camada HTTP fina sobre o PipelineOrquestrador.
Responsabilidades:
  - Endpoints REST (analyze, batch, health)
  - Validação de input (Pydantic)
  - Serialização de output
  - CORS, logging, error handling
  - Métricas básicas (contadores, latência)

O que NÃO faz:
  - Feature engineering (→ orquestrador)
  - Scoring (→ decision_engine)
  - Detecção de padrões (→ social_engineering / behavioral)

Uso:
  # Desenvolvimento
  uvicorn api:app --reload --host 0.0.0.0 --port 8000

  # Produção
  gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

  # Docker
  CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

Endpoints:
  POST /api/v1/analyze       → Analisa 1 transação (tempo real)
  POST /api/v1/batch         → Analisa N transações (lote)
  GET  /api/v1/health        → Health check completo
  GET  /api/v1/status        → Status detalhado dos componentes
  GET  /                     → Info básica da API
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# =========================================================
# LOGGING
# =========================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")


# =========================================================
# PIPELINE IMPORT (lazy — carrega no startup)
# =========================================================
pipeline = None


def _load_pipeline():
    """Carrega o pipeline orquestrador."""
    global pipeline
    from pipeline_orquestrador import PipelineOrquestrador
    from core.decision_engine import EngineConfig

    artefatos_dir = os.getenv("ARTEFATOS_DIR", "backend/artefatos")

    config_overrides = {}
    if os.getenv("THRESHOLD_CONFIRMAR"):
        config_overrides["threshold_confirmar"] = float(os.getenv("THRESHOLD_CONFIRMAR"))
    if os.getenv("THRESHOLD_BLOQUEAR"):
        config_overrides["threshold_bloquear"] = float(os.getenv("THRESHOLD_BLOQUEAR"))
    if os.getenv("VETO_THRESHOLD"):
        config_overrides["veto_threshold"] = float(os.getenv("VETO_THRESHOLD"))

    engine_config = EngineConfig(artefatos_dir=artefatos_dir, **config_overrides)
    pipeline = PipelineOrquestrador(
        artefatos_dir=artefatos_dir,
        engine_config=engine_config,
    )
    return pipeline


# =========================================================
# MÉTRICAS SIMPLES (in-memory — substituir por Prometheus em prod)
# =========================================================
class Metrics:
    """Contadores simples de métricas da API."""

    def __init__(self):
        self.total_requests: int = 0
        self.total_errors: int = 0
        self.total_transactions: int = 0
        self.decisions: Dict[str, int] = {
            "APROVAR": 0,
            "CONFIRMAR": 0,
            "BLOQUEAR": 0,
            "ERRO": 0,
        }
        self.latency_sum_ms: float = 0.0
        self.latency_max_ms: float = 0.0
        self.started_at: str = datetime.utcnow().isoformat() + "Z"

    def record_request(self, decisao: str, latency_ms: float):
        self.total_requests += 1
        self.total_transactions += 1
        self.decisions[decisao] = self.decisions.get(decisao, 0) + 1
        self.latency_sum_ms += latency_ms
        self.latency_max_ms = max(self.latency_max_ms, latency_ms)

    def record_error(self):
        self.total_requests += 1
        self.total_errors += 1
        self.decisions["ERRO"] += 1

    def record_batch(self, results: List[Dict], latency_ms: float):
        self.total_requests += 1
        self.total_transactions += len(results)
        self.latency_sum_ms += latency_ms
        self.latency_max_ms = max(self.latency_max_ms, latency_ms)
        for r in results:
            d = r.get("decisao", "ERRO")
            self.decisions[d] = self.decisions.get(d, 0) + 1

    @property
    def avg_latency_ms(self) -> float:
        if self.total_transactions == 0:
            return 0.0
        return self.latency_sum_ms / self.total_transactions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_transactions": self.total_transactions,
            "total_errors": self.total_errors,
            "decisions": self.decisions,
            "latency_avg_ms": round(self.avg_latency_ms, 1),
            "latency_max_ms": round(self.latency_max_ms, 1),
            "started_at": self.started_at,
            "uptime_seconds": round(
                (datetime.utcnow() - datetime.fromisoformat(self.started_at.replace("Z", ""))).total_seconds()
            ),
        }


metrics = Metrics()


# =========================================================
# PYDANTIC MODELS — Input Validation
# =========================================================
class TransactionInput(BaseModel):
    """Dados brutos de uma transação PIX para análise."""

    # Obrigatórios
    cd_pix: str = Field(..., description="Identificador único da transação PIX", min_length=1)
    dt_pix: str = Field(..., description="Data/hora da transação (ISO 8601)", min_length=10)
    cd_cpf_pagador: str = Field(..., description="CPF do pagador", min_length=11)
    vl_pix: float = Field(..., description="Valor do PIX em reais", gt=0)

    # Recebedor
    cd_cpf_cnpj_recebedor: Optional[str] = Field(None, description="CPF/CNPJ do recebedor")
    ds_chave_pix: Optional[str] = Field(None, description="Chave PIX utilizada")
    ds_tipo_chave: Optional[str] = Field(None, description="Tipo da chave PIX")

    # Histórico trimestral
    qt_total_pix_trimestre: Optional[float] = Field(None, ge=0)
    vl_mediana_pix_trimestre: Optional[float] = Field(None, ge=0)
    vl_desvio_padrao_pix_trimestre: Optional[float] = Field(None, ge=0)
    qt_intervalo_transacao_minuto: Optional[float] = Field(None)
    qt_intervalo_mediana_trimestre: Optional[float] = Field(None, ge=0)
    qt_intervalo_desvio_padrao_trimestre: Optional[float] = Field(None, ge=0)
    qt_pix_dia_maximo_trimestre: Optional[float] = Field(None, ge=0)

    # Device / App
    device_name: Optional[str] = None
    app_version: Optional[str] = None
    ip_address: Optional[str] = None

    # Latência / Interação
    latencia_rede_ms: Optional[float] = None
    vl_latencia_rede_media_trimestre: Optional[float] = None
    tempo_interacao_ms: Optional[float] = None
    vl_tempo_interacao_medio_trimestre: Optional[float] = None
    tempo_processamento_host_ms: Optional[float] = None

    # Autenticação / Sessão
    metodo_autenticacao: Optional[str] = None
    session_id: Optional[str] = None

    # Topaz
    cd_retorno: Optional[str] = None
    topaz_risk_score: Optional[float] = None
    topaz_transacao_rejeitada: Optional[float] = None

    # Agendamento
    is_agendamento_recorrente: Optional[str] = None

    # Perfil do cliente
    qt_aparelhos_distintos_trimestre: Optional[float] = None
    nr_idade: Optional[float] = Field(None, ge=0, le=150)
    qt_tempo_relacionamento_mes: Optional[float] = Field(None, ge=0)

    # v2.1b — Big Data
    vl_renda_cliente: Optional[float] = Field(None, ge=0)
    ds_sexo: Optional[str] = None
    ds_estado_civil: Optional[str] = None
    ds_segmento: Optional[str] = None
    qt_dependentes: Optional[float] = Field(None, ge=0)

    @field_validator("vl_pix")
    @classmethod
    def validate_vl_pix(cls, v):
        if v <= 0:
            raise ValueError("vl_pix deve ser maior que zero")
        if v > 1_000_000:
            logger.warning(f"PIX com valor muito alto: R${v:,.2f}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dict, excluindo campos None."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class BatchInput(BaseModel):
    """Input para análise em lote."""
    transactions: List[TransactionInput] = Field(
        ...,
        description="Lista de transações para análise",
        min_length=1,
        max_length=1000,
    )


class AnalyzeResponse(BaseModel):
    """Response padronizado da análise."""
    decisao: str
    score_final: float
    score_raw: float
    transaction_id: Optional[str] = None
    customer_id: Optional[str] = None
    timestamp: Optional[str] = None
    vl_pix: Optional[float] = None
    componentes: Dict[str, Any] = {}
    agravantes: List[Dict[str, Any]] = []
    peso_total: int = 0
    peso_maximo: int = 65
    social_engineering: Dict[str, Any] = {}
    behavioral: Dict[str, Any] = {}
    veto_aplicado: Optional[str] = None
    atenuantes: List[str] = []
    faixas: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class BatchResponse(BaseModel):
    """Response do batch."""
    total: int
    resultados: List[Dict[str, Any]]
    resumo: Dict[str, Any]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """Response do health check."""
    status: str
    pipeline_version: str
    components: Dict[str, Any]
    cache: Dict[str, Any]
    thresholds: Dict[str, Any]
    metrics: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Response de erro padronizado."""
    error: str
    detail: Optional[str] = None
    status_code: int


# =========================================================
# LIFESPAN (startup / shutdown)
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia ciclo de vida da aplicação."""
    # --- Startup ---
    logger.info("=" * 60)
    logger.info("  API Antifraude PIX v1.0 — Iniciando...")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    try:
        _load_pipeline()
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"  Pipeline carregado em {elapsed:.0f}ms")
        logger.info(f"  Status: {'✅ HEALTHY' if pipeline.available else '⚠️ DEGRADED'}")
    except Exception as e:
        logger.error(f"  ❌ Falha ao carregar pipeline: {e}")
        raise

    logger.info("=" * 60)
    logger.info("  API pronta para receber requisições")
    logger.info("=" * 60)

    yield

    # --- Shutdown ---
    logger.info("API encerrando...")


# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(
    title="API Antifraude PIX",
    description=(
        "API REST para detecção de fraude em transações PIX em tempo real.\n\n"
        "**Faixas de decisão:**\n"
        "- 🟢 **APROVAR** `[0, 60)` — Liberar automaticamente\n"
        "- 🟡 **CONFIRMAR** `[60, 85)` — Autenticação adicional\n"
        "- 🔴 **BLOQUEAR** `[85, 100]` — Análise humana\n\n"
        "**Componentes:** LightGBM + Isolation Forest + 21 Agravantes + "
        "Social Engineering (11 padrões) + Behavioral Analytics (12 fatores)"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# MIDDLEWARE — Logging de requisições
# =========================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Loga todas as requisições com latência."""
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000

    # Não logar health checks repetidos
    if request.url.path not in ("/api/v1/health", "/favicon.ico"):
        logger.info(
            f"{request.method} {request.url.path} → "
            f"{response.status_code} ({elapsed:.0f}ms)"
        )

    response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
    return response


# =========================================================
# EXCEPTION HANDLERS
# =========================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    metrics.record_error()
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    metrics.record_error()
    logger.error(f"Erro não tratado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erro interno do servidor",
            "detail": str(exc) if os.getenv("DEBUG", "").lower() in ("1", "true") else None,
            "status_code": 500,
        },
    )


# =========================================================
# ENDPOINTS
# =========================================================

# ─── Root ───────────────────────────────────────────────
@app.get("/", tags=["Info"])
async def root():
    """Informações básicas da API."""
    return {
        "name": "API Antifraude PIX",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "analyze": "POST /api/v1/analyze",
            "batch": "POST /api/v1/batch",
            "health": "GET /api/v1/health",
            "status": "GET /api/v1/status",
        },
    }


# ─── Analyze (1 transação — tempo real) ─────────────────
@app.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
    tags=["Análise"],
    summary="Analisa uma transação PIX",
    description=(
        "Recebe dados brutos de uma transação PIX e retorna decisão, "
        "score (0-100), agravantes, padrões de engenharia social e "
        "análise comportamental."
    ),
)
async def analyze_transaction(transaction: TransactionInput):
    """
    Analisa uma transação PIX em tempo real.

    **Latência esperada:** < 50ms (p95)

    **Campos obrigatórios:** cd_pix, dt_pix, cd_cpf_pagador, vl_pix

    **Retorna:**
    - `decisao`: APROVAR, CONFIRMAR ou BLOQUEAR
    - `score_final`: 0-100
    - `agravantes`: Lista de fatores de risco detectados
    - `social_engineering`: Padrões de golpe detectados
    - `behavioral`: Análise comportamental
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline não inicializado")

    t0 = time.perf_counter()

    try:
        result = pipeline.analisar(transaction.to_dict())
        elapsed = (time.perf_counter() - t0) * 1000

        metrics.record_request(result["decisao"], elapsed)
        return result

    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        metrics.record_error()
        logger.error(f"Erro ao analisar transação {transaction.cd_pix}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar transação: {str(e)}",
        )


# ─── Batch (N transações) ──────────────────────────────
@app.post(
    "/api/v1/batch",
    response_model=BatchResponse,
    tags=["Análise"],
    summary="Analisa múltiplas transações PIX",
    description="Processa até 1000 transações em um único request.",
)
async def analyze_batch(batch: BatchInput):
    """
    Analisa múltiplas transações PIX em lote.

    **Limite:** 1000 transações por request.
    **Processamento:** Sequencial (cada tx ~30-50ms).
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline não inicializado")

    t0 = time.perf_counter()

    try:
        tx_dicts = [tx.to_dict() for tx in batch.transactions]
        results = pipeline.analisar_batch(tx_dicts)
        elapsed = (time.perf_counter() - t0) * 1000

        metrics.record_batch(results, elapsed)

        # Resumo
        decisoes = {}
        scores = []
        for r in results:
            d = r.get("decisao", "ERRO")
            decisoes[d] = decisoes.get(d, 0) + 1
            if r.get("score_final") is not None and r["score_final"] >= 0:
                scores.append(r["score_final"])

        import numpy as np
        scores_arr = np.array(scores) if scores else np.array([0])

        return {
            "total": len(results),
            "resultados": results,
            "resumo": {
                "decisoes": decisoes,
                "score_medio": round(float(scores_arr.mean()), 2),
                "score_mediano": round(float(np.median(scores_arr)), 2),
                "score_max": round(float(scores_arr.max()), 2),
                "score_min": round(float(scores_arr.min()), 2),
            },
            "metadata": {
                "total_transactions": len(results),
                "latency_total_ms": round(elapsed, 1),
                "latency_avg_ms": round(elapsed / max(len(results), 1), 1),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        }

    except Exception as e:
        metrics.record_error()
        logger.error(f"Erro no batch ({len(batch.transactions)} tx): {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar batch: {str(e)}",
        )


# ─── Health Check ───────────────────────────────────────
@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["Operacional"],
    summary="Health check do sistema",
    description="Retorna status de todos os componentes e métricas.",
)
async def health_check():
    """
    Health check completo.

    Retorna `status: healthy` se o pipeline está operacional,
    ou `status: degraded` se algum componente falhou.

    **Uso:** Kubernetes liveness/readiness probes, load balancers.
    """
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "pipeline_version": "N/A",
                "components": {},
                "cache": {},
                "thresholds": {},
                "metrics": metrics.to_dict(),
            },
        )

    status = pipeline.get_status()
    status["metrics"] = metrics.to_dict()
    return status


# ─── Status Detalhado ───────────────────────────────────
@app.get(
    "/api/v1/status",
    tags=["Operacional"],
    summary="Status detalhado dos componentes",
)
async def detailed_status():
    """Status detalhado incluindo versões de modelos e configuração."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline não inicializado")

    pipeline_status = pipeline.get_status()
    engine_status = pipeline.engine.get_status()

    return {
        "pipeline": pipeline_status,
        "engine": engine_status,
        "config": {
            "threshold_confirmar": pipeline.engine.config.threshold_confirmar,
            "threshold_bloquear": pipeline.engine.config.threshold_bloquear,
            "veto_threshold": pipeline.engine.config.veto_threshold,
            "peso_maximo": pipeline.engine.config.peso_maximo,
        },
        "metrics": metrics.to_dict(),
        "environment": {
            "log_level": LOG_LEVEL,
            "cors_origins": ALLOWED_ORIGINS,
            "debug": os.getenv("DEBUG", "false"),
        },
    }


# ─── Métricas ──────────────────────────────────────────
@app.get(
    "/api/v1/metrics",
    tags=["Operacional"],
    summary="Métricas da API",
)
async def get_metrics():
    """Métricas de uso da API (contadores, latência, distribuição de decisões)."""
    return metrics.to_dict()


# ─── Reset Cache (operacional) ─────────────────────────
@app.post(
    "/api/v1/cache/reset",
    tags=["Operacional"],
    summary="Reseta cache de histórico de clientes",
)
async def reset_cache():
    """
    Reseta o cache de histórico de clientes.

    **Cuidado:** Isso afeta features sequenciais (first_receiver_flag,
    burst_30m_flag, etc.) até que o cache seja reconstruído.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline não inicializado")

    customers_before = len(pipeline._customer_history)
    pipeline.reset_cache()

    return {
        "message": "Cache resetado com sucesso",
        "customers_removed": customers_before,
    }


# =========================================================
# ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() in ("1", "true")
    workers = int(os.getenv("API_WORKERS", "1"))

    print(f"\n🚀 Iniciando API Antifraude PIX v1.0")
    print(f"   Host: {host}:{port}")
    print(f"   Workers: {workers}")
    print(f"   Reload: {reload}")
    print(f"   Docs: http://{host}:{port}/docs")
    print()

    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=LOG_LEVEL.lower(),
    )
