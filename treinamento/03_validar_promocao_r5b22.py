#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_validar_promocao_r5b22.py

O Gatekeeper de Qualidade (Caixa 3 do Oozie).
Avalia as métricas geradas pelo script de treinamento e valida se o modelo
pode ser promovido e publicado no HDFS, ou se a esteira deve ser interrompida.

Critérios Mínimos Estipulados (Flexibilizados para novos padrões):
- Recall (Intervenção) >= 97% (0.9700)
- FPR (Intervenção) <= 2% (0.0200)
"""

import sys
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
METRICS_FILE = SCRIPT_DIR / "resultado_treino_r5b22" / "metricas_r5b22_distilled.json"

# Limiares flexíveis 
MIN_RECALL = 0.9700
MAX_FPR = 0.0200

def main():
    print("=" * 80)
    print(" [Oozie Action] Gatekeeper Quality - Validação de Métricas ")
    print("=" * 80)

    if not METRICS_FILE.exists():
        print(f"❌ ERRO CRÍTICO: Arquivo de métricas não encontrado em {METRICS_FILE}")
        sys.exit(1)

    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Coleta as métricas de holdout do classificador de Intervenção
    holdout_metrics = data.get("metrics", {}).get("holdout_intervention", {})
    if not holdout_metrics:
        print("❌ ERRO CRÍTICO: Métricas de holdout ausentes no arquivo JSON.")
        sys.exit(1)

    recall = holdout_metrics.get("recall", 0.0)
    fpr = holdout_metrics.get("fpr", 1.0)
    
    tp = holdout_metrics.get("tp", 0)
    fn = holdout_metrics.get("fn", 0)
    fp = holdout_metrics.get("fp", 0)

    print(f"Avaliando resultados do Holdout (Intervenção):")
    print(f"  Recall obtido: {recall:.4%} (Mínimo exigido: {MIN_RECALL:.4%})")
    print(f"  FPR obtido:    {fpr:.4%} (Máximo tolerado: {MAX_FPR:.4%})")
    print(f"  TP: {tp} | FN: {fn} | FP: {fp}")
    print("-" * 50)

    approved = True
    if recall < MIN_RECALL:
        print("❌ FALHA: Recall está abaixo do limite estipulado!")
        approved = False
    
    if fpr > MAX_FPR:
        print("❌ FALHA: Taxa de Falsos Positivos (FPR) está acima do limite tolerável!")
        approved = False

    if not approved:
        print("🚨 O MODELO APRESENTOU DEGRADAÇÃO SIGNIFICATIVA E FOI REJEITADO PELO GATEKEEPER.")
        print("🚨 O pipeline do Oozie será abortado via Exit Code 1. (KILL NODE)")
        sys.exit(1)
        
    print("✅ SUCESSO: O modelo passou por todos os gates de segurança!")
    print("✅ Prosseguindo para a Publicação no HDFS.")
    sys.exit(0)

if __name__ == "__main__":
    main()