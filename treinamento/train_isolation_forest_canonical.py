#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_isolation_forest_canonical.py — Script Canônico do Isolation Forest (EXP-014B-R5A3 / R5B22)

Este script substitui os scripts de treino antigos do Isolation Forest e treina o modelo 
não-supervisionado sobre os splits expandidos v3 do Big Data (TRAIN, VALIDATION, HOLDOUT), 
usando apenas as transações legítimas para o ajuste, aplicando RobustScaler e salvando
os artefatos de percentis de score em produção.

⚠️ NOTA R5B22: O Isolation Forest é mantido como componente consultivo no runtime
R5B22. Suas saídas (if_percentile) alimentam a engenharia de features do orquestrador,
mas ele NÃO define sozinho a decisão do baseline oficial.
Os artefatos if_features.json, o scaler e a referência de scores são atualizados
aqui mantendo compatibilidade com as políticas vigentes.
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DADOS_DIR = PROJECT_ROOT / "dados"
ARTEFATOS_DIR = PROJECT_ROOT / "backend" / "artefatos"
RESULT_DIR = SCRIPT_DIR / "resultado_treino_if"

ARTEFATOS_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
TRAIN_DATA = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_TRAIN.csv"
VAL_DATA = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_VALIDATION.csv"
HOLD_DATA = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_HOLDOUT.csv"

# Parâmetros de treino
RANDOM_STATE = 42
FIXED_PARAMS = {
    "n_estimators": 800,
    "max_samples": 0.8,
    "max_features": 0.7,
}
CONTAMINATION_GRID = [0.005, 0.01, 0.02, 0.03]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa prefixos de colunas SQL no dataframe."""
    df = df.copy()
    df.columns = [c.split(".")[-1] for c in df.columns]
    return df


def create_features_if(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas e interações estruturadas para o Isolation Forest."""
    df = df.copy()
    
    # Preenchimentos
    vl_pix = df["vl_pix"].fillna(0)
    burst = df["burst_daily_7d_flag"].fillna(0)
    first_recv = df["first_receiver_flag_real"].fillna(0)
    distinct_recv = df["soma_recebedores_distintos_dia_180d"].fillna(1)
    
    # 1. log do valor (comprime outliers de valor extremo)
    df["log_vl_pix"] = np.log1p(vl_pix.clip(lower=0))
    
    # 2. interações de negócio com sinal de fraude
    df["valor_x_burst"] = vl_pix * (burst + 1)
    df["burst_x_distinct_recv"] = burst * distinct_recv
    
    return df


def anomaly_score_percentile(raw_scores: np.ndarray, ref_scores: np.ndarray) -> np.ndarray:
    """Converte raw decision_function scores para percentis [0, 1] baseados no treino."""
    inverted = -raw_scores
    ref = -ref_scores
    return np.array([np.mean(ref <= v) for v in inverted])


def evaluate_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    """Métricas básicas."""
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None,
        "average_precision": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / max(fp + tn, 1))
    }


def main():
    print("=" * 80)
    print("Treinamento Canônico do Isolation Forest — Dataset Expandido v3")
    print("=" * 80)
    
    # 1. Carregar dados
    if not (TRAIN_DATA.exists() and VAL_DATA.exists() and HOLD_DATA.exists()):
        print("❌ Splits de treino, validação ou holdout não encontrados na pasta dados!")
        sys.exit(1)
        
    print("Carregando datasets...")
    df_train = clean_columns(pd.read_csv(TRAIN_DATA))
    df_val = clean_columns(pd.read_csv(VAL_DATA))
    df_hold = clean_columns(pd.read_csv(HOLD_DATA))
    
    # 2. Criar as features estruturadas do IF
    print("\nExecutando engenharia de features de interação...")
    df_train = create_features_if(df_train)
    df_val = create_features_if(df_val)
    df_hold = create_features_if(df_hold)
    
    # Lista canônica de features para o Isolation Forest
    if_features = [
        "vl_pix",
        "log_vl_pix",
        "topaz_risk_score",
        "qtd_pix_pagador_180d",
        "max_qtd_pix_dia_pagador_7d",
        "soma_recebedores_distintos_dia_180d",
        "first_receiver_flag_real",
        "burst_daily_7d_flag",
        "dias_desde_primeiro_envio_recebedor",
        "valor_total_para_recebedor_180d",
        "valor_x_burst",
        "burst_x_distinct_recv",
        "ratio_valor_media_pagador_90d",
        "ratio_valor_maximo_pagador_180d"
    ]
    
    # Filtrar features existentes
    available_features = [f for f in if_features if f in df_train.columns]
    print(f"Features disponíveis para o IF: {len(available_features)}/{len(if_features)}")
    
    # 3. Preparação das matrizes e Imputação
    # MUDANÇA: O Isolation Forest é treinado apenas nos dados legítimos (normais) do treino!
    train_normal = df_train[df_train["is_fraud"] == 0].copy()
    
    X_train_raw = train_normal[available_features].copy()
    X_val_raw = df_val[available_features].copy()
    X_hold_raw = df_hold[available_features].copy()
    
    y_val = df_val["is_fraud"].values
    y_hold = df_hold["is_fraud"].values
    
    # Imputar nulos com a mediana obtida do treino
    medians = X_train_raw.median().fillna(0)
    
    X_train_imputed = X_train_raw.fillna(medians).replace([np.inf, -np.inf], 0)
    X_val_imputed = X_val_raw.fillna(medians).replace([np.inf, -np.inf], 0)
    X_hold_imputed = X_hold_raw.fillna(medians).replace([np.inf, -np.inf], 0)
    
    # 4. Escalonamento robusto
    print("Aplicando RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_hold_scaled = scaler.transform(X_hold_imputed)
    
    # 5. Grid Search de Contaminação
    print("\nExecutando busca de hiperparâmetros (contamination)...")
    best_ap = -1.0
    best_contamination = None
    best_model = None
    best_ref_scores = None
    
    for contam in CONTAMINATION_GRID:
        print(f"  Ajustando com contamination={contam}...")
        model_if = IsolationForest(
            n_estimators=FIXED_PARAMS["n_estimators"],
            max_samples=FIXED_PARAMS["max_samples"],
            max_features=FIXED_PARAMS["max_features"],
            contamination=contam,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model_if.fit(X_train_scaled)
        
        # Obter scores
        ref_scores = model_if.decision_function(X_train_scaled)
        raw_val_scores = model_if.decision_function(X_val_scaled)
        
        # Converter para percentis
        percentiles_val = anomaly_score_percentile(raw_val_scores, ref_scores)
        
        # Calcular métricas
        val_ap = average_precision_score(y_val, percentiles_val)
        val_auc = roc_auc_score(y_val, percentiles_val)
        
        print(f"    -> AP em Validação: {val_ap:.4f} | AUC: {val_auc:.4f}")
        
        if val_ap > best_ap:
            best_ap = val_ap
            best_contamination = contam
            best_model = model_if
            best_ref_scores = ref_scores
            
    print(f"\nMelhor contaminação: {best_contamination} (AP: {best_ap:.4f})")
    
    # 6. Avaliação final com o melhor modelo no holdout
    raw_hold_scores = best_model.decision_function(X_hold_scaled)
    percentiles_hold = anomaly_score_percentile(raw_hold_scores, best_ref_scores)
    
    raw_train_all = best_model.decision_function(scaler.transform(df_train[available_features].fillna(medians).replace([np.inf, -np.inf], 0)))
    percentiles_train = anomaly_score_percentile(raw_train_all, best_ref_scores)
    y_train_all = df_train["is_fraud"].values
    
    metrics_train = evaluate_at_threshold(y_train_all, percentiles_train, 0.5)
    metrics_val = evaluate_at_threshold(y_val, anomaly_score_percentile(best_model.decision_function(X_val_scaled), best_ref_scores), 0.5)
    metrics_hold = evaluate_at_threshold(y_hold, percentiles_hold, 0.5)
    
    print("\n--- Desempenho do Isolation Forest (Threshold=0.5) ---")
    print(f"  Treino:    ROC-AUC: {metrics_train['roc_auc']:.4f} | AP: {metrics_train['average_precision']:.4f} | F1: {metrics_train['f1']:.4f} | Recall: {metrics_train['recall']:.4f} | FPR: {metrics_train['fpr']:.4%}")
    print(f"  Validação: ROC-AUC: {metrics_val['roc_auc']:.4f} | AP: {metrics_val['average_precision']:.4f} | F1: {metrics_val['f1']:.4f} | Recall: {metrics_val['recall']:.4f} | FPR: {metrics_val['fpr']:.4%}")
    print(f"  Holdout:   ROC-AUC: {metrics_hold['roc_auc']:.4f} | AP: {metrics_hold['average_precision']:.4f} | F1: {metrics_hold['f1']:.4f} | Recall: {metrics_hold['recall']:.4f} | FPR: {metrics_hold['fpr']:.4%}")
    
    # 7. Gravação dos artefatos oficiais de produção
    print("\nSalvando artefatos oficiais...")
    
    # Salvar modelo IF e RobustScaler
    joblib.dump(best_model, ARTEFATOS_DIR / "model_isolation_forest.joblib")
    joblib.dump(scaler, ARTEFATOS_DIR / "scaler_isolation_forest.joblib")
    print("  Modelo e RobustScaler serializados com sucesso em: backend/artefatos/")
    
    # Salvar arquivo de referência de scores de treino
    np.save(ARTEFATOS_DIR / "if_ref_raw_train.npy", best_ref_scores)
    print("  Array de scores de referência do treino salvo.")
    
    # Salvar if_features.json
    with open(ARTEFATOS_DIR / "if_features.json", "w", encoding="utf-8") as f:
        json.dump(available_features, f, ensure_ascii=False, indent=2)
    print("  Arquivo if_features.json atualizado com a taxonomia real do treino.")
    
    # Salvar relatório técnico local
    report_dict = {
        "model": "Isolation Forest Canônico v3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "contamination": best_contamination,
        "n_estimators": FIXED_PARAMS["n_estimators"],
        "features": available_features,
        "train_metrics": metrics_train,
        "validation_metrics": metrics_val,
        "holdout_metrics": metrics_hold
    }
    with open(RESULT_DIR / "metricas_if_canonical.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)
        
    # Salvar predições de holdout para controle local
    holdout_preds = pd.DataFrame({
        "transaction_id": df_hold["transaction_id"].values if "transaction_id" in df_hold.columns else df_hold.index,
        "is_fraud": y_hold,
        "if_raw_score": raw_hold_scores,
        "if_percentile": percentiles_hold
    })
    holdout_preds.to_csv(RESULT_DIR / "holdout_predictions.csv", index=False)
    print("  Todas as predições e métricas do holdout gravadas com sucesso.")
    print("=" * 80)


if __name__ == "__main__":
    main()
