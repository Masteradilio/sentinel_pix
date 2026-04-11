"""
otimizar_features.py — Remove features de baixa relevância das bases de dados
===============================================================================

Remove 28 features identificadas pela análise de relevância, reduzindo
de 80 para 52 features no modelo LightGBM.

Bases processadas:
  - dados/base_mvp_model_ready.csv  (dados de treino do modelo)
  - dados/dados_pix_normais.csv     (transações normais brutas)
  - dados/dados_pix_fraudes.csv     (transações fraudulentas brutas)

Uso:
  python otimizar_features.py

Saída:
  - dados/base_mvp_model_ready_optimized.csv
  - dados/dados_pix_normais_optimized.csv
  - dados/dados_pix_fraudes_optimized.csv
  - dados/feature_optimization_log.json
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# =========================================================
# PATHS
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# Detectar raiz do projeto
if (SCRIPT_DIR / "dados").exists():
    PROJECT_ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "dados").exists():
    PROJECT_ROOT = SCRIPT_DIR.parent
elif SCRIPT_DIR.name == "backend" and (SCRIPT_DIR.parent / "dados").exists():
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

DADOS_DIR = PROJECT_ROOT / "dados"

# Bases de entrada
BASE_MODEL_READY = DADOS_DIR / "base_mvp_model_ready.csv"
BASE_NORMAIS = DADOS_DIR / "dados_pix_normais.csv"
BASE_FRAUDES = DADOS_DIR / "dados_pix_fraudes.csv"

# Bases de saída (otimizadas)
OUT_MODEL_READY = DADOS_DIR / "base_mvp_model_ready_optimized.csv"
OUT_NORMAIS = DADOS_DIR / "dados_pix_normais_optimized.csv"
OUT_FRAUDES = DADOS_DIR / "dados_pix_fraudes_optimized.csv"
OUT_LOG = DADOS_DIR / "feature_optimization_log.json"


# =========================================================
# FEATURES A REMOVER DO MODELO LGBM (28 features)
# =========================================================
# Baseado na análise de relevância: score composto < 0.05,
# permutation importance ≤ 0, near-zero variance, ou
# duplicatas exatas (correlação = 1.0)

FEATURES_REMOVER_MODEL_READY = [
    # --- Duplicatas exatas (manter apenas uma de cada par) ---
    "log_vl_pix",                         # duplicata de vl_pix (corr=1.0)
    "topaz_score_filled",                 # duplicata de topaz_risk_score (corr=1.0)
    "rule_score_normalized",              # duplicata de rule_score_raw (corr=1.0)
    "latencia_missing_flag",              # duplicata de host_time_missing_flag (corr=1.0)
    "rule_velocity_score",                # duplicata de burst_30m_flag (corr=1.0)
    "rule_mule_account_score",            # duplicata de first_receiver_flag (corr=1.0)

    # --- Score = 0.000 (zero contribuição em todos os testes) ---
    "qt_dependentes",                     # score=0.000, Near-Zero Variance
    "is_login_biometria_flag",            # score=0.000, Near-Zero Variance
    "topaz_rejeitada_flag",               # score=0.000, Near-Zero Variance

    # --- Near-Zero Variance + Permutation ≤ 0 + Score < 0.05 ---
    "is_login_senha_flag",                # score=0.004
    "is_agendamento_recorrente_flag",     # score=0.003
    "pix_key_missing_flag_derived",       # score=0.009, NZV
    "metodo_auth_encoded",                # score=0.031, NZV
    "tempo_interacao_missing_flag",       # score=0.035, NZV
    "app_version_missing_flag",           # score=0.032
    "app_version_minor",                  # score=0.028
    "auth_method_missing_flag",           # score=0.027
    "pix_key_email_flag",                 # score=0.014
    "pix_key_document_flag",              # score=0.032
    "pix_key_other_flag",                 # score=0.025
    "day_of_week",                        # score=0.022
    "is_business_hours",                  # score=0.049
    "pix_over_100pct_renda_flag",         # score=0.037
    "latencia_rede_ms_final",             # score=0.040, NZV
    "tempo_processamento_host_ms",        # score=0.037, NZV
    "latencia_host_ratio",                # score=0.048
    "receiver_document_same_as_customer_flag",  # score=0.049, NZV
    "is_sexo_feminino_flag",              # score=0.047
]

# Features a remover das bases BRUTAS (dados_pix_normais / dados_pix_fraudes)
# Estas são os CAMPOS BRUTOS que só alimentam features removidas
# CUIDADO: só removemos campos brutos que NÃO alimentam nenhuma feature mantida
CAMPOS_BRUTOS_REMOVER = [
    # qt_dependentes → só gera qt_dependentes (removida)
    "qt_dependentes",
    # is_agendamento_recorrente → só gera is_agendamento_recorrente_flag (removida)
    "is_agendamento_recorrente",
]

# Prefixo das colunas brutas (têm prefixo de tabela nos CSVs brutos)
PREFIXO_NORMAIS = "tb_pix_anomalia_normais_trim_poc_v2."
PREFIXO_FRAUDES = "tb_pix_anomalia_fraudes_trim_poc_v2."


# =========================================================
# PROCESSAMENTO
# =========================================================
def process_model_ready(input_path, output_path):
    """Remove features derivadas da base model_ready."""
    print(f"\n{'='*60}")
    print(f"  Processando: {input_path.name}")
    print(f"{'='*60}")

    if not input_path.exists():
        print(f"  ❌ Arquivo não encontrado: {input_path}")
        return None

    df = pd.read_csv(input_path)
    n_cols_antes = len(df.columns)
    n_rows = len(df)
    print(f"  Linhas: {n_rows}")
    print(f"  Colunas antes: {n_cols_antes}")

    # Identificar quais features existem na base
    features_encontradas = [f for f in FEATURES_REMOVER_MODEL_READY if f in df.columns]
    features_nao_encontradas = [f for f in FEATURES_REMOVER_MODEL_READY if f not in df.columns]

    if features_nao_encontradas:
        print(f"  ⚠ Features não encontradas (já removidas?): {len(features_nao_encontradas)}")
        for f in features_nao_encontradas:
            print(f"    - {f}")

    # Remover
    df_optimized = df.drop(columns=features_encontradas, errors="ignore")
    n_cols_depois = len(df_optimized.columns)

    print(f"  Features removidas: {len(features_encontradas)}")
    print(f"  Colunas depois: {n_cols_depois}")
    print(f"  Redução: {n_cols_antes} → {n_cols_depois} ({n_cols_antes - n_cols_depois} removidas)")

    # Listar features removidas
    print(f"\n  Features removidas:")
    for f in sorted(features_encontradas):
        print(f"    ❌ {f}")

    # Salvar
    df_optimized.to_csv(output_path, index=False)
    print(f"\n  ✅ Salvo: {output_path}")
    print(f"  Tamanho: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return {
        "arquivo": input_path.name,
        "linhas": n_rows,
        "colunas_antes": n_cols_antes,
        "colunas_depois": n_cols_depois,
        "features_removidas": features_encontradas,
        "features_nao_encontradas": features_nao_encontradas,
    }


def process_base_bruta(input_path, output_path, prefixo):
    """Remove campos brutos das bases de normais/fraudes."""
    print(f"\n{'='*60}")
    print(f"  Processando: {input_path.name}")
    print(f"{'='*60}")

    if not input_path.exists():
        print(f"  ❌ Arquivo não encontrado: {input_path}")
        return None

    df = pd.read_csv(input_path)
    n_cols_antes = len(df.columns)
    n_rows = len(df)
    print(f"  Linhas: {n_rows}")
    print(f"  Colunas antes: {n_cols_antes}")

    # Montar nomes com prefixo
    colunas_remover = []
    for campo in CAMPOS_BRUTOS_REMOVER:
        col_com_prefixo = f"{prefixo}{campo}"
        col_sem_prefixo = campo

        if col_com_prefixo in df.columns:
            colunas_remover.append(col_com_prefixo)
        elif col_sem_prefixo in df.columns:
            colunas_remover.append(col_sem_prefixo)

    if not colunas_remover:
        print(f"  ℹ Nenhuma coluna bruta para remover nesta base")
        # Copiar sem alteração
        df.to_csv(output_path, index=False)
        print(f"  ✅ Copiado sem alterações: {output_path}")
        return {
            "arquivo": input_path.name,
            "linhas": n_rows,
            "colunas_antes": n_cols_antes,
            "colunas_depois": n_cols_antes,
            "campos_removidos": [],
        }

    # Remover
    df_optimized = df.drop(columns=colunas_remover, errors="ignore")
    n_cols_depois = len(df_optimized.columns)

    print(f"  Campos brutos removidos: {len(colunas_remover)}")
    for c in colunas_remover:
        print(f"    ❌ {c}")
    print(f"  Colunas: {n_cols_antes} → {n_cols_depois}")

    # Salvar
    df_optimized.to_csv(output_path, index=False)
    print(f"\n  ✅ Salvo: {output_path}")
    print(f"  Tamanho: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return {
        "arquivo": input_path.name,
        "linhas": n_rows,
        "colunas_antes": n_cols_antes,
        "colunas_depois": n_cols_depois,
        "campos_removidos": colunas_remover,
    }


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n" + "=" * 60)
    print("  OTIMIZAÇÃO DE FEATURES — Motor Antifraude PIX v2.1")
    print(f"  {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 60)
    print(f"\n  Features a remover do modelo: {len(FEATURES_REMOVER_MODEL_READY)}")
    print(f"  Campos brutos a remover: {len(CAMPOS_BRUTOS_REMOVER)}")
    print(f"  Diretório de dados: {DADOS_DIR}")

    results = {}

    # 1. Base model_ready (a mais importante)
    r1 = process_model_ready(BASE_MODEL_READY, OUT_MODEL_READY)
    if r1:
        results["base_model_ready"] = r1

    # 2. Base normais (bruta)
    r2 = process_base_bruta(BASE_NORMAIS, OUT_NORMAIS, PREFIXO_NORMAIS)
    if r2:
        results["dados_pix_normais"] = r2

    # 3. Base fraudes (bruta)
    r3 = process_base_bruta(BASE_FRAUDES, OUT_FRAUDES, PREFIXO_FRAUDES)
    if r3:
        results["dados_pix_fraudes"] = r3

    # Salvar log
    log = {
        "data_execucao": datetime.now().isoformat(),
        "features_removidas_modelo": FEATURES_REMOVER_MODEL_READY,
        "campos_brutos_removidos": CAMPOS_BRUTOS_REMOVER,
        "features_antes": 80,
        "features_depois": 80 - len(FEATURES_REMOVER_MODEL_READY),
        "resultados": results,
    }

    with open(OUT_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Log salvo: {OUT_LOG}")

    # Resumo
    print(f"\n{'='*60}")
    print(f"  RESUMO DA OTIMIZAÇÃO")
    print(f"{'='*60}")
    print(f"  Modelo: 80 → {80 - len(FEATURES_REMOVER_MODEL_READY)} features")
    print(f"  Features removidas: {len(FEATURES_REMOVER_MODEL_READY)}")
    print(f"  Campos brutos removidos: {len(CAMPOS_BRUTOS_REMOVER)}")
    print(f"\n  Arquivos gerados:")
    print(f"    {OUT_MODEL_READY.name}")
    print(f"    {OUT_NORMAIS.name}")
    print(f"    {OUT_FRAUDES.name}")
    print(f"\n  ⚠️  PRÓXIMO PASSO:")
    print(f"     Atualize o INPUT_DATA nos scripts de treino para apontar")
    print(f"     para 'base_mvp_model_ready_optimized.csv'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
