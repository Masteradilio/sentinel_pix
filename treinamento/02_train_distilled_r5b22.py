#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_lgbm_distilled_r5b22.py — Treino Semanal do LightGBM Destilado (Baseline R5B22)

Este script treina os modelos alunos (intervenção e bloqueio) a partir do contrato 
professor R5B16/R5B18, preservando o baseline R5B22.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Any, Tuple, List, Dict

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# =========================================================
# PATHS
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DADOS_DIR = PROJECT_ROOT / "dados"
ARTEFATOS_DIR = PROJECT_ROOT / "backend" / "artefatos"
RESULT_DIR = SCRIPT_DIR / "resultado_treino_r5b22"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Datasets canônicos
TRAIN_DATA = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_TRAIN.csv"
VAL_DATA = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_VALIDATION.csv"
HOLD_DATA = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_HOLDOUT.csv"

# Metadados oficiais
METADATA_PATH = ARTEFATOS_DIR / "model_lgbm_distilled_r5b22_metadata.json"

# Datasets do professor (Temporário/Mock para extração dos targets se não fornecido via Pipeline)
# Idealmente em produção, as predições do professor (r5b18_e2e_contract_decisao e features frozen) 
# já devem vir anexadas à base de treino.
R5B18_PATH = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R5B18-E2E-FROZEN-CONTRACT-HOMOLOGATION" / "01_vectorized_contract_predictions.csv"
FROZEN_PATH = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R4G-FAST-FROZEN" / "06_predictions_frozen.csv"

def load_metadata() -> Dict[str, Any]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadados oficiais ausentes em {METADATA_PATH}")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.split(".")[-1] for c in df.columns]
    return df

def get_teacher_targets(df: pd.DataFrame, r5b18_df: pd.DataFrame, frozen_df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Mescla o dataframe com as decisões e features congeladas do professor."""
    base_action_col = "r5b18_e2e_contract_decisao"
    
    contract = r5b18_df[["transaction_id", "r4g_fast_frozen_decisao_recommended", "r5b14_rule_applied", "r5b14_layer_applied", base_action_col]]
    
    # Precisamos garantir as features congeladas se não existirem no df original
    cat_cols = metadata["categorical_features"]
    frozen_cols = ["transaction_id"] + [c for c in metadata["feature_columns"] if c in frozen_df.columns]
    
    df = df.merge(contract, on="transaction_id", how="left")
    df = df.merge(frozen_df[frozen_cols], on="transaction_id", how="left", suffixes=("", "_frozen"))
    
    # Criar targets do professor
    df["contract_intervention"] = df[base_action_col].isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)
    df["contract_block"] = df[base_action_col].eq("BLOQUEAR").astype(int)
    
    return df

def validate_and_prepare_features(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    features = metadata["feature_columns"]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Features obrigatórias ausentes no dataset: {missing}")
    
    # Forçar ordem e filtrar
    X = df[features].copy()
    
    # Aplicar category encoders
    encoders = metadata["category_encoders"]
    for col in metadata["categorical_features"]:
        if col not in encoders:
            continue
        mapping = encoders[col]
        # Se for nova categoria, mapear para <MISSING> ou -1
        # No Baseline, a categoria <MISSING> geralmente existe no dicionário.
        missing_val = mapping.get("<MISSING>", -1)
        
        # O mapping no metadata é string -> int
        X[col] = X[col].fillna("<MISSING>").astype(str).map(mapping).fillna(missing_val).astype("int32")
        
    # Converter numéricos
    for col in features:
        if col not in metadata["categorical_features"]:
            X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-999.0)
            
    return X

def metrics(y_true: pd.Series, pred: pd.Series) -> dict[str, Any]:
    y = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)
    p = pd.to_numeric(pred, errors="coerce").fillna(0).astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    fpr = fp / max(fp + tn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(float(precision), 8),
        "recall": round(float(recall), 8),
        "f1": round(float(f1), 8),
        "fpr": round(float(fpr), 8),
    }

def train_distilled_model(
    target_col: str, 
    X_train: pd.DataFrame, y_train: pd.Series, 
    X_val: pd.DataFrame, y_val: pd.Series
) -> lgb.LGBMClassifier:
    pos = int(y_train.sum())
    model = lgb.LGBMClassifier(
        objective="binary", 
        random_state=42, 
        n_jobs=-1, 
        verbose=-1,
        num_leaves=31, 
        learning_rate=0.03, 
        n_estimators=600,
        scale_pos_weight=(len(y_train) - pos) / max(pos, 1),
    )
    
    model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        eval_metric="auc", 
        callbacks=[lgb.early_stopping(60, verbose=False)]
    )
    return model

def main():
    print("=" * 80)
    print("Treino do Aluno Destilado R5B22 (Intervenção e Bloqueio)")
    print("=" * 80)
    
    metadata = load_metadata()
    
    # 1. Carregar Dados
    print("Carregando datasets e definindo targets pelo professor...")
    if not (TRAIN_DATA.exists() and VAL_DATA.exists() and HOLD_DATA.exists()):
        print("❌ Splits canônicos não encontrados.")
        sys.exit(1)
        
    df_train = clean_columns(pd.read_csv(TRAIN_DATA, low_memory=False))
    df_val = clean_columns(pd.read_csv(VAL_DATA, low_memory=False))
    df_hold = clean_columns(pd.read_csv(HOLD_DATA, low_memory=False))
    
    if R5B18_PATH.exists() and FROZEN_PATH.exists():
        r5b18 = pd.read_csv(R5B18_PATH, low_memory=False)
        frozen = pd.read_csv(FROZEN_PATH, low_memory=False)
        df_train = get_teacher_targets(df_train, r5b18, frozen, metadata)
        df_val = get_teacher_targets(df_val, r5b18, frozen, metadata)
        df_hold = get_teacher_targets(df_hold, r5b18, frozen, metadata)
    else:
        print("❌ Bases do professor (R5B18 e FROZEN) não encontradas. Não é possível destilar.")
        sys.exit(1)
        
    # 2. Features e Categoricos
    print("Processando feature contract R5B22 (78 features)...")
    X_train = validate_and_prepare_features(df_train, metadata)
    X_val = validate_and_prepare_features(df_val, metadata)
    X_hold = validate_and_prepare_features(df_hold, metadata)
    
    # 3. Treinar Intervention
    print("\nTreinando modelo Intervention (APROVAR vs CONFIRMAR/BLOQUEAR)...")
    model_intervention = train_distilled_model("contract_intervention", X_train, df_train["contract_intervention"], X_val, df_val["contract_intervention"])
    p_intervention_val = model_intervention.predict_proba(X_val)[:, 1]
    p_intervention_hold = model_intervention.predict_proba(X_hold)[:, 1]
    
    # 4. Treinar Block
    print("Treinando modelo Block (BLOQUEAR vs APROVAR/CONFIRMAR)...")
    model_block = train_distilled_model("contract_block", X_train, df_train["contract_block"], X_val, df_val["contract_block"])
    p_block_val = model_block.predict_proba(X_val)[:, 1]
    p_block_hold = model_block.predict_proba(X_hold)[:, 1]
    
    # 5. Avaliação e Gates (usando os thresholds originais para verificação de reprodutibilidade)
    intervention_th = metadata["intervention_threshold"]
    block_th = metadata["block_threshold"]
    
    # O modelo busca aproximar o professor e segurar os indicadores de fraude.
    # Avaliar no validation/holdout contra IS_FRAUD
    pred_int_val = (p_intervention_val >= intervention_th).astype(int)
    pred_int_hold = (p_intervention_hold >= intervention_th).astype(int)
    pred_blk_val = (p_block_val >= block_th).astype(int)
    pred_blk_hold = (p_block_hold >= block_th).astype(int)
    
    val_int_metrics = metrics(df_val["is_fraud"], pred_int_val)
    hold_int_metrics = metrics(df_hold["is_fraud"], pred_int_hold)
    val_blk_metrics = metrics(df_val["is_fraud"], pred_blk_val)
    hold_blk_metrics = metrics(df_hold["is_fraud"], pred_blk_hold)
    
    print("\n--- Resultados Holdout (vs Fraud Real) ---")
    print(f"Intervention -> F1: {hold_int_metrics['f1']:.4f} | Prec: {hold_int_metrics['precision']:.4f} | Rec: {hold_int_metrics['recall']:.4f} | FPR: {hold_int_metrics['fpr']:.4%}")
    print(f"Block        -> F1: {hold_blk_metrics['f1']:.4f} | Prec: {hold_blk_metrics['precision']:.4f} | Rec: {hold_blk_metrics['recall']:.4f} | FPR: {hold_blk_metrics['fpr']:.4%}")
    
    # 6. Salvar Artefatos localmente
    print("\nGravando artefatos...")
    joblib.dump(model_intervention, RESULT_DIR / "model_lgbm_distilled_r5b22_intervention.joblib")
    joblib.dump(model_block, RESULT_DIR / "model_lgbm_distilled_r5b22_block.joblib")
    
    # Feature importances
    pd.DataFrame({
        "feature": X_train.columns,
        "importance": model_intervention.feature_importances_
    }).sort_values("importance", ascending=False).to_csv(RESULT_DIR / "feature_importance_intervention.csv", index=False)
    
    pd.DataFrame({
        "feature": X_train.columns,
        "importance": model_block.feature_importances_
    }).sort_values("importance", ascending=False).to_csv(RESULT_DIR / "feature_importance_block.csv", index=False)
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "intervention_threshold": intervention_th,
        "block_threshold": block_th,
        "metrics": {
            "validation_intervention": val_int_metrics,
            "holdout_intervention": hold_int_metrics,
            "validation_block": val_blk_metrics,
            "holdout_block": hold_blk_metrics,
        }
    }
    
    with open(RESULT_DIR / "metricas_r5b22_distilled.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"Artefatos gerados em: {RESULT_DIR}")
    print("Para promover a produção, copie os joblibs para backend/artefatos/ mediante validação explícita.")

if __name__ == "__main__":
    main()
