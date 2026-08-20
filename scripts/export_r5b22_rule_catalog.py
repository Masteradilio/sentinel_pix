#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_r5b22_rule_catalog.py

Extrai automaticamente as regras oficiais R5B14 e R5B22 para a construção
de um catálogo auditável das regras que atuam no motor de decisão.

Gera saídas em Markdown, CSV e JSON.
"""

import json
import csv
import sys
import os
from pathlib import Path

# Add project root to sys.path to allow importing backend
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.severity_policy import (
    R5B14_CONFIRM_TO_BLOCK_RULES,
    R5B14_APPROVE_TO_BLOCK_RULES,
    R5B14_CONFIRM_TO_APPROVE_RULES
)

R5B22_POLICY_PATH = PROJECT_ROOT / "backend" / "artefatos" / "r5b22_official_baseline_policy.json"
MD_OUTPUT_PATH = PROJECT_ROOT / "docs" / "catalogo_regras_r5b22.md"
CSV_OUTPUT_PATH = PROJECT_ROOT / "resultados" / "r5b22_rule_catalog.csv"
JSON_OUTPUT_PATH = PROJECT_ROOT / "resultados" / "r5b22_rule_catalog.json"


def carregar_regras_r5b22() -> list:
    if not R5B22_POLICY_PATH.exists():
        print(f"Aviso: {R5B22_POLICY_PATH} não encontrado.")
        return []
    with open(R5B22_POLICY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("selected_rules", [])


def exportar():
    catalog = []

    # Processar R5B14
    for r in R5B14_CONFIRM_TO_BLOCK_RULES:
        catalog.append({
            "policy": "R5B14",
            "rule_id": r.rule_id,
            "layer": "CONFIRM_TO_BLOCK",
            "condition": r.description,
            "action": "Escalonar CONFIRMAR para BLOQUEAR",
            "objective": "Prevenção de Falso Negativo (Segurança)",
            "status": "Baseline Oficial",
            "incremental_n_rows": None,
            "incremental_n_frauds": None,
            "incremental_n_normals": None
        })

    for r in R5B14_APPROVE_TO_BLOCK_RULES:
        catalog.append({
            "policy": "R5B14",
            "rule_id": r.rule_id,
            "layer": "APPROVE_TO_BLOCK",
            "condition": r.description,
            "action": "Escalonar APROVAR para BLOQUEAR",
            "objective": "Prevenção de Falso Negativo (Segurança)",
            "status": "Baseline Oficial",
            "incremental_n_rows": None,
            "incremental_n_frauds": None,
            "incremental_n_normals": None
        })

    for r in R5B14_CONFIRM_TO_APPROVE_RULES:
        catalog.append({
            "policy": "R5B14",
            "rule_id": r.rule_id,
            "layer": "CONFIRM_TO_APPROVE",
            "condition": r.description,
            "action": "Desescalonar CONFIRMAR para APROVAR",
            "objective": "Redução de Falso Positivo (Compensação)",
            "status": "Baseline Oficial",
            "incremental_n_rows": None,
            "incremental_n_frauds": None,
            "incremental_n_normals": None
        })

    # Processar R5B22
    r5b22_rules = carregar_regras_r5b22()
    for r in r5b22_rules:
        target_action = r.get("target_action", "N/A")
        inc_rows = r.get("incremental_n_rows", 0)
        inc_frauds = r.get("incremental_n_frauds", 0)
        inc_normals = r.get("incremental_n_normals", 0)
        catalog.append({
            "policy": "R5B22",
            "rule_id": r.get("rule_id", ""),
            "layer": "DEMOTION",
            "condition": r.get("description", ""),
            "action": f"Reverter intervenção para {target_action}",
            "objective": "Controle de FPR global (Trade-off)",
            "status": "Baseline Oficial",
            "incremental_n_rows": inc_rows,
            "incremental_n_frauds": inc_frauds,
            "incremental_n_normals": inc_normals
        })

    # Adicionar o contrato professor
    catalog.append({
        "policy": "R5B16/R5B18",
        "rule_id": "r4g_fast_frozen_decisao_recommended",
        "layer": "BASE_CONTRACT",
        "condition": "Output estático do pipeline original R4G Fast Frozen",
        "action": "Fornecer decisão âncora/default",
        "objective": "Sinal do Professor (Estabilidade)",
        "status": "Baseline Oficial",
        "incremental_n_rows": None,
        "incremental_n_frauds": None,
        "incremental_n_normals": None
    })

    # 1. Gerar Markdown
    with open(MD_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("# Catálogo Oficial de Regras - Baseline R5B22\n\n")
        f.write("Este documento lista as regras sistêmicas ativas no baseline oficial de produção.\n")
        f.write("A decisão do motor R5B22 é baseada no contrato estático `r4g_fast_frozen_decisao_recommended` em conjunto com a aplicação sequencial das políticas abaixo, guiando o treinamento do modelo Aluno.\n\n")

        f.write("## 1. Contrato Professor (Base Decisória)\n\n")
        f.write("Representa as flags e restrições históricas utilizadas como rótulo de aprendizado para o modelo destilado.\n\n")
        f.write("- **Decisão-base:** `r4g_fast_frozen_decisao_recommended` atua como o ponto de partida do contrato congelado, não consistindo em uma regra condicional simples, mas no veredito fixo de pipeline consolidado.\n")
        f.write("- **Flags rastreadas pelo Aluno:** `r5b14_rule_applied`, `r5b14_layer_applied`, entre outras `_frozen` features.\n\n")

        f.write("## 2. Política R5B14 - Prevenção de Falsos Negativos (Severidade)\n\n")
        f.write("Regras ativadas para elevar o risco de transações potencialmente enganosas.\n\n")
        f.write("| Rule ID | Camada | Condição | Ação | Objetivo |\n")
        f.write("|---------|--------|----------|------|----------|\n")
        for r in catalog:
            if r["policy"] == "R5B14":
                f.write(f"| `{r['rule_id']}` | {r['layer']} | `{r['condition']}` | {r['action']} | {r['objective']} |\n")

        f.write("\n## 3. Política R5B22 - Controle de FPR (Suavização)\n\n")
        f.write("Regras aplicadas para reduzir bloqueios e confirmações indevidos que passaram na malha R5B14.\n\n")
        f.write("| Rule ID | Alvo | Condição | Fraudes (Inc.) | Normais (Inc.) |\n")
        f.write("|---------|------|----------|----------------|----------------|\n")
        for r in catalog:
            if r["policy"] == "R5B22":
                f.write(f"| `{r['rule_id']}` | {r['action']} | `{r['condition']}` | {r.get('incremental_n_frauds')} | {r.get('incremental_n_normals')} |\n")

        f.write("\n## 4. Política Candidata: Severity Policy\n\n")
        f.write("O módulo `backend/core/severity_policy.py` contém as regras estruturais listadas na Política R5B14, sendo o mecanismo oficialmente encarregado no baseline. Modificações ou adições a este arquivo só devem ser incorporadas ao baseline após passar pelas provas empíricas e pela geração da máscara R5B22.\n")

    # 2. Gerar JSON
    JSON_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    # 3. Gerar CSV
    CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if catalog:
        with open(CSV_OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=catalog[0].keys())
            writer.writeheader()
            for row in catalog:
                writer.writerow(row)

    print("Catálogo de Regras R5B22 gerado com sucesso!")
    print(f"- {MD_OUTPUT_PATH}")
    print(f"- {JSON_OUTPUT_PATH}")
    print(f"- {CSV_OUTPUT_PATH}")

if __name__ == "__main__":
    exportar()
