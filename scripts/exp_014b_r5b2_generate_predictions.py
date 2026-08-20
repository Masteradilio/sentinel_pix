#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R5B2-CALIBRATION — Gerador de Predições do Orquestrador pós-Fase 2.

Este script executa a inferência em lote do PipelineOrquestrador oficial
usando os novos modelos canônicos sobre o dataset expandido v3 inteiro.
A saída servirá como dados de base para a sintonia fina de políticas.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import pandas as pd

# Paths
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
CORE_DIR = BACKEND_DIR / "core"
DADOS_DIR = PROJECT_ROOT / "dados"
OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R5B2-CALIBRATION"

sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Input / Output
INPUT_PATH = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv"
TARGET_PATH = OUTPUT_DIR / "01_raw_predictions_holdout.csv"


def main():
    import os
    os.environ["USE_PRECOMPUTED_FEATURES"] = "1"
    print("=" * 80)
    print("EXP-014B-R5B2-CALIBRATION — Gerador de Predições de Base")
    print("=" * 80)
    print(f"Input:  {INPUT_PATH}")
    print(f"Target: {TARGET_PATH}")

    if not INPUT_PATH.exists():
        print(f"❌ Input não encontrado: {INPUT_PATH}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando base de dados...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    
    # Ordenar cronologicamente para consistência de cache
    if "event_datetime" in df.columns:
        df["event_datetime"] = pd.to_datetime(df["event_datetime"])
        df = df.sort_values("event_datetime").reset_index(drop=True)
    elif "dt_pix" in df.columns:
        df["dt_pix"] = pd.to_datetime(df["dt_pix"])
        df = df.sort_values("dt_pix").reset_index(drop=True)

    print(f"Total de linhas carregadas: {len(df):,} | Fraudes: {df['is_fraud'].sum():,}")

    # Importar modulo de simulacao oficial
    try:
        from backend.scripts import simular_pipeline_e2e_v2 as sim
    except ImportError as e:
        print(f"❌ Falha ao importar simular_pipeline_e2e_v2: {e}")
        sys.exit(1)

    print("Iniciando processamento paralelo (14 workers)...")
    t0 = time.perf_counter()
    
    # Processa via orquestrador usando 14 workers paralelos
    predictions_df = sim.process_batch_parallel(
        df, 
        n_workers=14,
        engine_config_overrides={
            "threshold_confirmar": 77.0,
            "threshold_bloquear": 95.0,
            "shap_enabled": False
        }
    )
    
    elapsed = time.perf_counter() - t0
    print(f"Processamento concluído em {elapsed / 60:.2f}min ({len(df) / elapsed:.1f} tx/s)")

    print(f"Gravando predições brutas em: {TARGET_PATH}...")
    predictions_df.to_csv(TARGET_PATH, index=False)
    print("Predições gravadas com sucesso.")
    print("=" * 80)


if __name__ == "__main__":
    main()
