"""
experimentos/utils_experimentos.py — Helpers compartilhados entre experimentos.

Este módulo centraliza funcionalidades usadas por múltiplos experimentos da FASE 1:
  - Setup de paths (adiciona backend/core ao sys.path)
  - Cálculo de métricas binárias
  - Carregamento/sampling do dataset
  - Processamento em batch via PipelineOrquestrador real (reusa script v2)
  - Serialização JSON segura
  - Formatação de output (tabelas, seções)

Filosofia: zero duplicação. Reusa o máximo possível de
backend/scripts/simular_pipeline_e2e_v2.py.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# UTF-8 no Windows — aplicado de forma segura para multiprocessing
def _setup_utf8_stdout() -> None:
    """Configura stdout/stderr para UTF-8 no Windows de forma segura.

    Usa reconfigure() quando disponível (Python 3.7+) em vez de substituir
    sys.stdout por um novo TextIOWrapper. Isso evita o erro
    'I/O operation on closed file' que ocorre em multiprocessing no Windows
    quando processos filhos re-importam este módulo.

    Também pula o setup em processos filhos (detectados via multiprocessing).
    """
    if sys.platform != "win32":
        return

    # Detectar se estamos num processo filho (spawn do multiprocessing)
    try:
        import multiprocessing
        if multiprocessing.current_process().name != "MainProcess":
            return  # Processo filho — não mexer em stdout
    except Exception:
        pass

    # Tentar reconfigure() primeiro (Python 3.7+, modo seguro)
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # Fallback: se reconfigure falhar, ignora silenciosamente
        # Melhor ter output com encoding errado do que crash
        pass


_setup_utf8_stdout()



# =========================================================
# PATHS — resolve layout do projeto
# =========================================================
UTILS_PATH = Path(__file__).resolve()
EXPERIMENTOS_DIR = UTILS_PATH.parent
PROJECT_ROOT = EXPERIMENTOS_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
CORE_DIR = BACKEND_DIR / "core"
SCRIPTS_DIR = BACKEND_DIR / "scripts"
ARTEFATOS_DIR = BACKEND_DIR / "artefatos"
DADOS_DIR = PROJECT_ROOT / "dados"
DATASET_PATH = DADOS_DIR / "base_treino_final.csv"
RESULTADOS_EXP_DIR = PROJECT_ROOT / "resultados" / "experimentos"


def setup_sys_path() -> None:
    """Adiciona diretórios necessários ao sys.path na ordem correta.

    Ordem importa: CORE_DIR primeiro para resolver `from preprocessing import ...`
    usado dentro do pipeline_orquestrador.
    """
    for p in (CORE_DIR, BACKEND_DIR, SCRIPTS_DIR, PROJECT_ROOT):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


# =========================================================
# LOGGING
# =========================================================
def get_logger(name: str = "experimento") -> logging.Logger:
    """Configura logger padrão para experimentos."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(name)

    # Silenciar engines durante batch
    for mod in (
        "core.decision_engine",
        "core.behavioral_analytics",
        "core.social_engineering",
        "core.pipeline_orquestrador",
        "preprocessing",
    ):
        logging.getLogger(mod).setLevel(logging.WARNING)

    return logger


# =========================================================
# FORMATAÇÃO
# =========================================================
def print_section(title: str, width: int = 72) -> None:
    """Imprime cabeçalho de seção formatado."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def safe_json_dump(obj: Any, path: Path) -> None:
    """Serializa dict/list para JSON com fallback para tipos numpy/pandas."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_default)


# =========================================================
# MÉTRICAS
# =========================================================
@dataclass
class BinaryMetrics:
    """Métricas de classificação binária."""
    label: str
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    fpr: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "TP": self.tp,
            "FP": self.fp,
            "FN": self.fn,
            "TN": self.tn,
            "Precision": round(self.precision, 6),
            "Recall": round(self.recall, 6),
            "F1": round(self.f1, 6),
            "FPR": round(self.fpr, 8),
        }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> BinaryMetrics:
    """Calcula métricas binárias padronizadas.

    Args:
        y_true: Array binário {0, 1} com labels reais.
        y_pred: Array binário {0, 1} com predições.
        label: Rótulo descritivo.

    Returns:
        BinaryMetrics com TP/FP/FN/TN/Precision/Recall/F1/FPR.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    fpr = fp / max(fp + tn, 1)

    return BinaryMetrics(
        label=label,
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
    )


# =========================================================
# DATASET
# =========================================================
def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Carrega o dataset base com dtypes corretos e ordenação temporal."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    df = pd.read_csv(path)
    if "event_datetime" in df.columns:
        df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("event_datetime").reset_index(drop=True)

    return df


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    seed: int = 42,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Sample estratificado por is_fraud, preservando TODAS as fraudes.

    Args:
        df: DataFrame completo.
        n: Tamanho do sample alvo.
        seed: Semente para reprodutibilidade.
        logger: Logger opcional para reportar.

    Returns:
        DataFrame ordenado por event_datetime.
    """
    fraud = df[df["is_fraud"] == 1].copy()
    normal = df[df["is_fraud"] == 0].copy()

    n_fraud = len(fraud)
    n_normal_needed = max(0, n - n_fraud)
    n_normal_needed = min(n_normal_needed, len(normal))

    rng = np.random.RandomState(seed)
    normal_idx = rng.choice(normal.index, size=n_normal_needed, replace=False)
    normal_sample = normal.loc[sorted(normal_idx)]

    sample = pd.concat([fraud, normal_sample], ignore_index=False)
    sample = sample.sort_values("event_datetime").reset_index(drop=True)

    if logger:
        logger.info(
            f"Sample estratificado (seed={seed}): {len(sample):,} tx "
            f"({n_fraud} fraudes + {len(normal_sample):,} normais)"
        )

    return sample


# =========================================================
# PROCESSAMENTO VIA ORQUESTRADOR REAL
# =========================================================
def process_dataframe_via_orquestrador(
    df: pd.DataFrame,
    workers: int = 1,
    logger: logging.Logger | None = None,
    engine_config_overrides: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Processa DataFrame inteiro via PipelineOrquestrador real.

    Reusa a lógica de simular_pipeline_e2e_v2.py para garantir paridade
    com a validação da FASE 0.

    Args:
        df: DataFrame com as transações a processar.
        workers: 1 = sequencial, >1 = multiprocessing.
        logger: Logger opcional.

    Returns:
        DataFrame com colunas de predição (score_final, decisao, lgbm_raw, etc).
    """
    setup_sys_path()

    # Import tardio — requer sys.path configurado
    from simular_pipeline_e2e_v2 import (
        process_batch_parallel,
        process_batch_sequential,
    )

    if logger:
        logger.info(f"Processando {len(df):,} tx via PipelineOrquestrador (workers={workers})")

    t0 = time.perf_counter()

    if workers > 1:
        predictions_df = process_batch_parallel(
            df,
            n_workers=workers,
            engine_config_overrides=engine_config_overrides,
        )
    else:
        predictions_df = process_batch_sequential(
            df,
            engine_config_overrides=engine_config_overrides,
        )

    elapsed = time.perf_counter() - t0
    if logger:
        logger.info(
            f"Processamento concluído em {elapsed:.1f}s "
            f"({len(df) / elapsed:.1f} tx/s)"
        )

    return predictions_df


# =========================================================
# APLICAÇÃO DE THRESHOLD PÓS-PROCESSAMENTO
# =========================================================
def apply_threshold_to_predictions(
    predictions_df: pd.DataFrame,
    threshold_confirmar: int,
    threshold_bloquear: int = 90,
) -> pd.Series:
    """Reaplica thresholds sobre score_final para simular variantes do EXP-001.

    IMPORTANTE: isso simula a decisão de thresholds diferentes SEM reprocessar
    as tx. Usa apenas score_final (que já foi calculado uma vez pelo orquestrador).

    ATENÇÃO: vetos cirúrgicos aplicados pelo engine (BLOQUEAR/CONFIRMAR por veto)
    são preservados — esta função NÃO desfaz decisões baseadas em vetos, apenas
    reavalia a zona score-based (entre threshold_confirmar e threshold_bloquear).

    Args:
        predictions_df: DataFrame com colunas `score_final` e `veto_aplicado`.
        threshold_confirmar: Score mínimo para CONFIRMAR (variante do experimento).
        threshold_bloquear: Score mínimo para BLOQUEAR (mantido em 90 por default).

    Returns:
        Series com decisões {APROVAR, CONFIRMAR, BLOQUEAR}.
    """
    score = predictions_df["score_final"].astype(float).values
    veto = predictions_df.get("veto_aplicado")

    # Decisão base pelo score
    decisao = np.where(
        score >= threshold_bloquear, "BLOQUEAR",
        np.where(score >= threshold_confirmar, "CONFIRMAR", "APROVAR"),
    )

    # Preservar vetos que FORÇARAM BLOQUEAR/CONFIRMAR
    # (um veto presente + decisão original agressiva deve ser mantida)
    if veto is not None:
        veto_values = veto.fillna("").astype(str).values
        original_decisao = predictions_df["decisao"].astype(str).values

        # Se havia veto e decisão original era BLOQUEAR/CONFIRMAR → manter original
        veto_mask = (veto_values != "") & (
            np.isin(original_decisao, ["BLOQUEAR", "CONFIRMAR"])
        )
        decisao = np.where(veto_mask, original_decisao, decisao)

    return pd.Series(decisao, index=predictions_df.index, name="decisao_variante")


# =========================================================
# OUTPUT DIR
# =========================================================
def get_experiment_output_dir(exp_id: str) -> Path:
    """Retorna e cria diretório de output para um experimento.

    Args:
        exp_id: Ex: 'EXP-001', 'EXP-002'.

    Returns:
        Path do diretório criado (resultados/experimentos/EXP-XXX/).
    """
    output_dir = RESULTADOS_EXP_DIR / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
