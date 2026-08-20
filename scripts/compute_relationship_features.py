#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_relationship_features.py — Engenharia de Features de Relacionamento (EXP-014B-R5B)

Calcula de forma incremental e sem leakage temporal 5 novas features de relacionamento:
1. qtd_pix_mesmo_recebedor_7d
2. valor_medio_para_recebedor_180d
3. dias_desde_ultima_transacao_recebedor
4. ratio_valor_pix_vs_max_recebedor_180d
5. is_recebedor_recorrente_180d
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Determinar paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DADOS_DIR = PROJECT_ROOT / "dados"

TRAIN_PATH = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_TRAIN.csv"
VAL_PATH = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_VALIDATION.csv"
HOLD_PATH = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_HOLDOUT.csv"

# Janelas temporais em segundos
SEC_7D = 7 * 86400
SEC_180D = 180 * 86400


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove prefixo de colunas SQL (ex: 'tb_pix_dataset_v3_features_180d_v1.')."""
    df = df.copy()
    df.columns = [c.split(".")[-1] for c in df.columns]
    return df


def main():
    print("=" * 80)
    print("Engenharia de Features de Relacionamento — Cálculo Offline (Fase 2)")
    print("=" * 80)

    # 1. Carregar splits
    if not (TRAIN_PATH.exists() and VAL_PATH.exists() and HOLD_PATH.exists()):
        print("[ERRO] Splits de treino, validação ou holdout não encontrados na pasta dados!")
        sys.exit(1)

    print("Carregando datasets canônicos...")
    df_train = clean_columns(pd.read_csv(TRAIN_PATH))
    df_val = clean_columns(pd.read_csv(VAL_PATH))
    df_hold = clean_columns(pd.read_csv(HOLD_PATH))

    # Marcar origem dos splits para separação posterior
    df_train["_split_origin"] = "TRAIN"
    df_val["_split_origin"] = "VALIDATION"
    df_hold["_split_origin"] = "HOLDOUT"

    # Concatenar para ordenação global e cálculo incremental correto
    df_all = pd.concat([df_train, df_val, df_hold], ignore_index=True)
    print(f"Total de registros a processar: {len(df_all):,}")

    # Converter datetime para pandas timestamp
    df_all["event_datetime"] = pd.to_datetime(df_all["event_datetime"])

    # Ordenar globalmente por tempo
    df_all = df_all.sort_values(["event_datetime", "transaction_id"]).reset_index(drop=True)

    n = len(df_all)
    ts_epoch = (df_all["event_datetime"].astype("int64") // 10**9).values
    senders = df_all["customer_id"].values
    receivers = df_all["counterparty_id"].values
    values = df_all["vl_pix"].values.astype(np.float64)

    # Pré-alocar arrays de resultados (usando np.nan para missings/nulos)
    qtd_7d = np.zeros(n, dtype=np.int32)
    val_mean_180d = np.full(n, np.nan, dtype=np.float64)
    dias_desde_ultimo = np.full(n, np.nan, dtype=np.float64)
    ratio_max_180d = np.full(n, np.nan, dtype=np.float64)
    recorrente_180d = np.zeros(n, dtype=np.int32)

    # Dicionário de histórico do par (customer_id, counterparty_id) -> lista de (timestamp_s, valor)
    # A lista será mantida ordenada por tempo
    pair_history = {}

    print("\nIniciando cálculo incremental de relacionamento (janela deslizante)...")
    t0 = time.perf_counter()
    log_interval = max(1, n // 10)

    for i in range(n):
        s = senders[i]
        r = receivers[i]
        v = values[i]
        ts = ts_epoch[i]

        if pd.isna(s) or pd.isna(r):
            continue

        s_str = str(s)
        r_str = str(r)
        pair_key = (s_str, r_str)

        # 1. Obter histórico anterior se existir
        hist = pair_history.get(pair_key, [])

        if hist:
            # Filtrar janela de 180 dias
            cutoff_180d = ts - SEC_180D
            # Limpar registros mais antigos que 180d para manter a lista compacta
            # Como hist é cronológico, podemos usar list comprehension
            hist = [item for item in hist if item[0] >= cutoff_180d]
            pair_history[pair_key] = hist  # Atualizar histórico limpo

            # Se ainda houver transações na janela
            if hist:
                # Filtrar janela de 7 dias
                cutoff_7d = ts - SEC_7D
                txs_7d = [item for item in hist if item[0] >= cutoff_7d]

                # Features calculadas:
                # 1. qtd_pix_mesmo_recebedor_7d
                qtd_7d[i] = len(txs_7d)

                # 2. valor_medio_para_recebedor_180d
                vals_180d = [item[1] for item in hist]
                val_mean_180d[i] = np.mean(vals_180d)

                # 3. dias_desde_ultima_transacao_recebedor
                ts_anterior = hist[-1][0]
                dias_desde_ultimo[i] = max((ts - ts_anterior) / 86400.0, 0.0)

                # 4. ratio_valor_pix_vs_max_recebedor_180d
                max_val_180d = np.max(vals_180d)
                if max_val_180d > 0:
                    ratio_max_180d[i] = v / max_val_180d

                # 5. is_recebedor_recorrente_180d
                # Se o número de transações na janela de 180d for >= 2 (com a atual será a 3ª ou mais)
                recorrente_180d[i] = 1 if len(hist) >= 2 else 0

        # 2. Registrar a transação atual no histórico do par
        if pair_key not in pair_history:
            pair_history[pair_key] = []
        pair_history[pair_key].append((ts, v))

        # Progresso
        if (i + 1) % log_interval == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / max(elapsed, 0.01)
            eta = (n - i - 1) / max(rate, 0.01)
            print(f"  Processados: {i+1:,}/{n:,} tx ({int((i+1)/n*100)}%) — {elapsed:.1f}s | ETA: {eta:.1f}s")

    # 3. Injetar colunas calculadas no DataFrame global
    df_all["qtd_pix_mesmo_recebedor_7d"] = qtd_7d
    df_all["valor_medio_para_recebedor_180d"] = val_mean_180d
    df_all["dias_desde_ultima_transacao_recebedor"] = dias_desde_ultimo
    df_all["ratio_valor_pix_vs_max_recebedor_180d"] = ratio_max_180d
    df_all["is_recebedor_recorrente_180d"] = recorrente_180d

    elapsed_total = time.perf_counter() - t0
    print(f"\n[OK] Features de relacionamento calculadas em {elapsed_total:.2f}s!")

    # 4. Separar novamente nos splits originais baseados no _split_origin
    print("\nSalvando novos datasets atualizados...")

    splits = ["TRAIN", "VALIDATION", "HOLDOUT"]
    paths = {
        "TRAIN": TRAIN_PATH,
        "VALIDATION": VAL_PATH,
        "HOLDOUT": HOLD_PATH
    }

    for split_name in splits:
        df_split = df_all[df_all["_split_origin"] == split_name].copy()
        # Remover coluna temporária de controle
        df_split = df_split.drop(columns=["_split_origin"])
        
        path = paths[split_name]
        # Salvar por cima do original
        df_split.to_csv(path, index=False)
        print(f"  Salvo: {path.name} | Shape: {df_split.shape} | Fraudes: {df_split['is_fraud'].sum()}")

    print("\n" + "=" * 80)
    print("Finalizado com Sucesso!")
    print("=" * 80)


if __name__ == "__main__":
    main()
