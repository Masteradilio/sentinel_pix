#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_lgbm_canonical.py — Script Canônico de Treino do LightGBM (EXP-014B-R5A3)

Este script substitui os scripts de treino antigos e treina o LightGBM diretamente 
sobre os splits expandidos v3 do Big Data (TRAIN, VALIDATION, HOLDOUT), limpando
os prefixos SQL e consolidando o contrato definitivo de features.
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
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
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
RESULT_DIR = SCRIPT_DIR / "resultado_treino_lgbm"

ARTEFATOS_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
TRAIN_DATA = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_TRAIN.csv"
VAL_DATA = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_VALIDATION.csv"
HOLD_DATA = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_HOLDOUT.csv"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa prefixos de colunas SQL no dataframe."""
    df = df.copy()
    df.columns = [c.split(".")[-1] for c in df.columns]
    return df


def evaluate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Calcula métricas de performance."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "threshold": round(float(threshold), 6),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 6) if len(np.unique(y_true)) > 1 else None,
        "average_precision": round(float(average_precision_score(y_true, y_prob)), 6) if len(np.unique(y_true)) > 1 else None,
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "fpr": round(float(fp / max(fp + tn, 1)), 6)
    }


def find_best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Busca o melhor threshold maximizando F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.005, 0.96, 0.005):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def main():
    print("=" * 80)
    print("⚠️ AVISO DE DEPRECIAÇÃO: Treinamento Canônico do LightGBM ⚠️")
    print("Este script (EXP-014B-R5A3) está OBSOLETO para o baseline R5B22.")
    print("Para o treinamento semanal oficial (Baseline Constrained R5B22), utilize:")
    print("   python backend/modelos/train_lgbm_distilled_r5b22.py")
    print("=" * 80)
    print("Continuando com o treinamento legado v3...\n")
    
    # 1. Carregar os dados
    if not (TRAIN_DATA.exists() and VAL_DATA.exists() and HOLD_DATA.exists()):
        print("❌ Splits de treino, validação ou holdout não encontrados na pasta dados!")
        sys.exit(1)
        
    print("Carregando datasets...")
    df_train = clean_columns(pd.read_csv(TRAIN_DATA))
    df_val = clean_columns(pd.read_csv(VAL_DATA))
    df_hold = clean_columns(pd.read_csv(HOLD_DATA))
    
    print(f"  Treino:    {df_train.shape[0]:,} linhas, {df_train['is_fraud'].sum():,} fraudes")
    print(f"  Validação: {df_val.shape[0]:,} linhas, {df_val['is_fraud'].sum():,} fraudes")
    print(f"  Holdout:   {df_hold.shape[0]:,} linhas, {df_hold['is_fraud'].sum():,} fraudes")
    
    # 2. Definição de colunas de metadados, target e IDs
    exclude_cols = {
        "transaction_id", "cd_pix", "customer_id", "counterparty_id", 
        "event_datetime", "dt_pix", "data_pix", "is_fraud", 
        "dataset_role", "source_dataset", "sample_strategy", "sample_weight", 
        "temporal_split", "window_start_date", "window_end_date", 
        "dataset_created_at", "dataset_v3_created_at", "rn",
        "ds_chave_pix", "session_id", "primeira_data_envio_recebedor_180d"
    }
    
    # Identificar colunas candidatas a features
    all_cols = set(df_train.columns)
    feature_cols = sorted(list(all_cols - exclude_cols))
    
    print(f"\nTotal de features iniciais: {len(feature_cols)}")
    
    # 3. Tratamento de variáveis categóricas
    categorical_cols = [
        "ds_tipo_chave_norm", "periodo_dia", "value_band", 
        "device_name", "app_version", "ip_address", "metodo_autenticacao"
    ]
    
    # Encoders
    encoders = {}
    for col in categorical_cols:
        if col in df_train.columns:
            print(f"  Codificando coluna categórica: {col}")
            le = LabelEncoder()
            
            # Garantir dados do treino/validação/holdout consolidados para evitar categorias não vistas
            combined_series = pd.concat([df_train[col], df_val[col], df_hold[col]]).astype(str).fillna("missing")
            le.fit(combined_series)
            
            df_train[col] = le.transform(df_train[col].astype(str).fillna("missing"))
            df_val[col] = le.transform(df_val[col].astype(str).fillna("missing"))
            df_hold[col] = le.transform(df_hold[col].astype(str).fillna("missing"))
            encoders[col] = le
            
    # Salvar encoders de categorias
    joblib.dump(encoders, ARTEFATOS_DIR / "lgbm_label_encoders.joblib")
    
    # Preparar X e y
    X_train = df_train[feature_cols]
    y_train = df_train["is_fraud"].astype(int)
    
    X_val = df_val[feature_cols]
    y_val = df_val["is_fraud"].astype(int)
    
    X_hold = df_hold[feature_cols]
    y_hold = df_hold["is_fraud"].astype(int)
    
    # 4. Hiperparâmetros e pesos
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw = n_neg / max(n_pos, 1)
    
    print(f"\nProporção das classes: 1 fraud para {spw:.2f} normais. scale_pos_weight = {spw:.2f}")
    
    model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=2500,
        learning_rate=0.01,
        num_leaves=63,
        max_depth=7,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=spw,
        verbose=-1
    )
    
    # Treino com early stopping
    print("Iniciando ajuste do LightGBM...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[
            early_stopping(stopping_rounds=150, verbose=False),
            log_evaluation(period=200)
        ]
    )
    
    best_iteration = model.best_iteration_
    print(f"  Ajuste concluído! Melhor iteração: {best_iteration}")
    
    # 5. Avaliação do modelo nos splits
    p_train = model.predict_proba(X_train)[:, 1]
    p_val = model.predict_proba(X_val)[:, 1]
    p_hold = model.predict_proba(X_hold)[:, 1]
    
    best_th = find_best_threshold_by_f1(y_val, p_val)
    print(f"\nMelhor threshold encontrado na validação: {best_th:.4f}")
    
    metrics_train = evaluate_metrics(y_train, p_train, best_th)
    metrics_val = evaluate_metrics(y_val, p_val, best_th)
    metrics_hold = evaluate_metrics(y_hold, p_hold, best_th)
    
    print("\n--- Desempenho do LightGBM ---")
    print(f"  Treino:    ROC-AUC: {metrics_train['roc_auc']:.4f} | AP: {metrics_train['average_precision']:.4f} | F1: {metrics_train['f1']:.4f} | Recall: {metrics_train['recall']:.4f} | FPR: {metrics_train['fpr']:.4%}")
    print(f"  Validação: ROC-AUC: {metrics_val['roc_auc']:.4f} | AP: {metrics_val['average_precision']:.4f} | F1: {metrics_val['f1']:.4f} | Recall: {metrics_val['recall']:.4f} | FPR: {metrics_val['fpr']:.4%}")
    print(f"  Holdout:   ROC-AUC: {metrics_hold['roc_auc']:.4f} | AP: {metrics_hold['average_precision']:.4f} | F1: {metrics_hold['f1']:.4f} | Recall: {metrics_hold['recall']:.4f} | FPR: {metrics_hold['fpr']:.4%}")
    
    # 6. Salvar Artefatos
    print("\nGravando artefatos...")
    
    # Salvar modelo serializado oficial
    joblib.dump(model, ARTEFATOS_DIR / "model_lightgbm.joblib")
    print(f"  Modelo LightGBM salvo em: backend/artefatos/model_lightgbm.joblib")
    
    # Salvar lgbm_features.json consolidado
    lgbm_features_json = {
        "version": "v3.0_canonical",
        "n_features": len(feature_cols),
        "features": feature_cols,
        "core_features": [f for f in feature_cols if not f.startswith("graph_")],
        "extra_features": [f for f in feature_cols if "renda" in f or "perfil" in f],
        "leakage_fixed_features": [f for f in feature_cols if "trimestre" in f or "pagador" in f or "recebedor" in f]
    }
    with open(ARTEFATOS_DIR / "lgbm_features.json", "w", encoding="utf-8") as f:
        json.dump(lgbm_features_json, f, ensure_ascii=False, indent=2)
    print(f"  Contrato de features salvo em: backend/artefatos/lgbm_features.json")
    
    # Salvar métricas locais de controle
    metrics_summary = {
        "model": "LightGBM Canônico v3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_iteration": int(best_iteration),
        "scale_pos_weight": float(spw),
        "best_threshold": float(best_th),
        "train_metrics": metrics_train,
        "validation_metrics": metrics_val,
        "holdout_metrics": metrics_hold
    }
    with open(RESULT_DIR / "metricas_lgbm_canonical.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
        
    # Salvar importância de features
    importances = model.booster_.feature_importance(importance_type="gain")
    feat_imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": importances
    }).sort_values("importance_gain", ascending=False)
    feat_imp_df.to_csv(RESULT_DIR / "feature_importance.csv", index=False)
    
    # Salvar predições de holdout e validação
    holdout_preds = pd.DataFrame({
        "transaction_id": df_hold["transaction_id"].values if "transaction_id" in df_hold.columns else df_hold.index,
        "is_fraud": y_hold.values,
        "lgbm_score": p_hold,
        "lgbm_pred": (p_hold >= best_th).astype(int)
    })
    holdout_preds.to_csv(RESULT_DIR / "holdout_predictions.csv", index=False)
    print("  Todas as predições e métricas do holdout gravadas com sucesso.")
    print("=" * 80)


if __name__ == "__main__":
    main()
