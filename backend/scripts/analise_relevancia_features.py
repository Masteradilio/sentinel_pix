"""
analise_features_relevancia.py v1.0 — Análise Estatística de Relevância de Features
====================================================================================

Objetivo:
  Avaliar a contribuição de cada feature ao modelo de detecção de fraude PIX,
  identificando candidatas à remoção para otimizar a ingestão do Big Data
  (~10h atualmente) sem comprometer a variância explicada e a performance.

Testes realizados:
  1. Importância do LightGBM (split + gain)
  2. Permutation Importance (impacto real no AUC)
  3. Correlação entre features (redundância)
  4. Mutual Information (dependência não-linear com o target)
  5. Teste de Levene / Brown-Forsythe (heterocedasticidade entre classes)
  6. Teste de Mann-Whitney U (separação univariada fraude vs normal)
  7. Variance Inflation Factor — VIF (multicolinearidade)
  8. PCA — Variância Explicada Acumulada
  9. Near-Zero Variance (features quase constantes)
  10. Análise de Grupos de Features por Origem (mapeamento para campos brutos)
  11. Simulação de remoção incremental com impacto no AUC

Uso:
  python analise_features_relevancia.py

Saída:
  - relatorio/analise_features/  (gráficos + CSVs + relatório HTML)
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from scipy import stats
from scipy.stats import mannwhitneyu, levene, spearmanr, kendalltau
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# =========================================================
# PATHS
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "backend").exists() and (SCRIPT_DIR / "dados").exists():
    PROJECT_ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "backend").exists():
    PROJECT_ROOT = SCRIPT_DIR.parent
elif SCRIPT_DIR.name == "backend":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

ARTEFATOS_DIR = PROJECT_ROOT / "backend" / "artefatos"
if not ARTEFATOS_DIR.exists():
    ARTEFATOS_DIR = SCRIPT_DIR / "artefatos"

OUTPUT_DIR = PROJECT_ROOT / "relatorio" / "analise_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LGBM_PATH = ARTEFATOS_DIR / "model_lightgbm.joblib"
LGBM_FEATURES_PATH = ARTEFATOS_DIR / "lgbm_features.json"
X_TEST_PATH = ARTEFATOS_DIR / "X_test.csv"
Y_TEST_PATH = ARTEFATOS_DIR / "y_test.csv"

# =========================================================
# ESTILO DOS GRÁFICOS
# =========================================================
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#1a1d23",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#ffffff",
    "text.color": "#ffffff",
    "xtick.color": "#cccccc",
    "ytick.color": "#cccccc",
    "grid.color": "#333333",
    "grid.alpha": 0.3,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

COLORS = {
    "primary": "#00d4aa",
    "secondary": "#ff6b6b",
    "accent": "#4ecdc4",
    "warning": "#ffd93d",
    "info": "#6c5ce7",
    "bg_card": "#1a1d23",
    "text": "#ffffff",
    "text_muted": "#888888",
}


# =========================================================
# MAPEAMENTO: Feature derivada → Campo bruto de origem
# =========================================================
FEATURE_TO_SOURCE = {
    # --- Valor e Desvio (origem: vl_pix, vl_mediana, vl_desvio) ---
    "vl_pix": "vl_pix",
    "log_vl_pix": "vl_pix",
    "vl_pix_over_1000_flag": "vl_pix",
    "ratio_valor_mediana": ["vl_pix", "vl_mediana_pix_trimestre"],
    "diff_valor_mediana": ["vl_pix", "vl_mediana_pix_trimestre"],
    "ratio_valor_desvio_padrao": ["vl_pix", "vl_desvio_padrao_pix_trimestre"],
    "zscore_valor_aprox": ["vl_pix", "vl_mediana_pix_trimestre", "vl_desvio_padrao_pix_trimestre"],
    "vl_mediana_pix_trimestre": "vl_mediana_pix_trimestre",
    "vl_desvio_padrao_pix_trimestre": "vl_desvio_padrao_pix_trimestre",

    # --- Frequência e Velocity ---
    "qt_total_pix_trimestre": "qt_total_pix_trimestre",
    "is_first_tx_trimestre": "qt_total_pix_trimestre",
    "qt_intervalo_transacao_minuto": "qt_intervalo_transacao_minuto",
    "qt_intervalo_mediana_trimestre": "qt_intervalo_mediana_trimestre",
    "qt_intervalo_desvio_padrao_trimestre": "qt_intervalo_desvio_padrao_trimestre",
    "ratio_intervalo_vs_mediana": ["qt_intervalo_transacao_minuto", "qt_intervalo_mediana_trimestre"],
    "diff_intervalo_vs_mediana": ["qt_intervalo_transacao_minuto", "qt_intervalo_mediana_trimestre"],
    "zscore_intervalo_aprox": ["qt_intervalo_transacao_minuto", "qt_intervalo_mediana_trimestre",
                                "qt_intervalo_desvio_padrao_trimestre"],
    "qt_pix_dia_maximo_trimestre": "qt_pix_dia_maximo_trimestre",
    "minutes_since_prev_tx": "dt_pix",
    "tx_count_prev_30m": "dt_pix",
    "burst_30m_flag": "dt_pix",

    # --- Recebedor ---
    "receiver_tx_count_prev": ["cd_cpf_cnpj_recebedor", "dt_pix"],
    "first_receiver_flag": ["cd_cpf_cnpj_recebedor", "dt_pix"],
    "distinct_receivers_so_far": ["cd_cpf_cnpj_recebedor", "dt_pix"],
    "tp_primeiro_envio_recebedor_trimestre": "tp_primeiro_envio_recebedor_trimestre",
    "qt_envio_recebedor_trimestre": "qt_envio_recebedor_trimestre",
    "receiver_document_same_as_customer_flag": ["cd_cpf_pagador", "cd_cpf_cnpj_recebedor"],

    # --- Chave PIX ---
    "pix_key_random_flag": "ds_tipo_chave",
    "pix_key_email_flag": "ds_tipo_chave",
    "pix_key_document_flag": "ds_tipo_chave",
    "pix_key_other_flag": "ds_tipo_chave",
    "pix_key_missing_flag_derived": "ds_tipo_chave",
    "key_tx_count_prev": ["ds_chave_pix", "dt_pix"],
    "first_key_flag": ["ds_chave_pix", "dt_pix"],
    "distinct_keys_so_far": ["ds_chave_pix", "dt_pix"],

    # --- Temporal ---
    "hour": "dt_pix",
    "day_of_week": "dt_pix",
    "is_business_hours": "dt_pix",

    # --- Perfil do Cliente ---
    "nr_idade": "nr_idade",
    "qt_tempo_relacionamento_mes": "qt_tempo_relacionamento_mes",
    "is_sexo_feminino_flag": "ds_sexo",
    "is_viuvo_flag": "ds_estado_civil",
    "is_segmento_premium_flag": "ds_segmento",
    "vl_renda_cliente": "vl_renda_cliente",
    "qt_dependentes": "qt_dependentes",
    "perfil_vulneravel_se_flag": ["nr_idade", "ds_estado_civil", "qt_dependentes"],

    # --- Renda ---
    "ratio_pix_renda": ["vl_pix", "vl_renda_cliente"],
    "pix_over_50pct_renda_flag": ["vl_pix", "vl_renda_cliente"],
    "pix_over_100pct_renda_flag": ["vl_pix", "vl_renda_cliente"],
    "renda_missing_flag": "vl_renda_cliente",

    # --- Dispositivo e Sessão ---
    "latencia_rede_ms_final": "latencia_rede_ms",
    "vl_latencia_rede_media_trimestre": "vl_latencia_rede_media_trimestre",
    "ratio_latencia_cliente": ["latencia_rede_ms", "vl_latencia_rede_media_trimestre"],
    "diff_latencia_cliente": ["latencia_rede_ms", "vl_latencia_rede_media_trimestre"],
    "tempo_processamento_host_ms": "tempo_processamento_host_ms",
    "latencia_host_ratio": ["latencia_rede_ms", "tempo_processamento_host_ms"],
    "qt_aparelhos_distintos_trimestre": "qt_aparelhos_distintos_trimestre",
    "app_version_minor": "app_version",
    "device_missing_flag": "device_name",
    "app_version_missing_flag": "app_version",
    "latencia_missing_flag": "latencia_rede_ms",
    "host_time_missing_flag": "tempo_processamento_host_ms",
    "tempo_interacao_missing_flag": "tempo_interacao_ms",

    # --- Autenticação ---
    "metodo_auth_encoded": "metodo_autenticacao",
    "is_login_senha_flag": "metodo_autenticacao",
    "is_login_biometria_flag": "metodo_autenticacao",
    "auth_method_missing_flag": "metodo_autenticacao",
    "is_agendamento_recorrente_flag": "is_agendamento_recorrente",

    # --- Topaz ---
    "topaz_risk_score": "topaz_risk_score",
    "topaz_score_filled": "topaz_risk_score",
    "topaz_rejeitada_flag": "topaz_transacao_rejeitada",
    "topaz_missing_flag": "topaz_risk_score",

    # --- Regras de Negócio (derivadas de múltiplas fontes) ---
    "rule_age_score": "nr_idade",
    "rule_relationship_score": "qt_tempo_relacionamento_mes",
    "rule_mule_account_score": ["nr_idade", "qt_tempo_relacionamento_mes"],
    "rule_random_key_score": "ds_tipo_chave",
    "rule_velocity_score": "dt_pix",
    "rule_topaz_score": "topaz_risk_score",
    "rule_score_raw": "DERIVADA_MULTIPLA",
    "rule_score_normalized": "DERIVADA_MULTIPLA",
}

# Campos brutos de origem no Big Data
SOURCE_FIELDS = {
    "BLK (Extrato PIX)": [
        "cd_pix", "dt_pix", "cd_cpf_pagador", "cd_cpf_cnpj_recebedor",
        "ds_chave_pix", "ds_tipo_chave", "vl_pix",
        "qt_total_pix_trimestre", "vl_mediana_pix_trimestre",
        "vl_desvio_padrao_pix_trimestre", "qt_intervalo_transacao_minuto",
        "qt_intervalo_mediana_trimestre", "qt_intervalo_desvio_padrao_trimestre",
        "qt_pix_dia_maximo_trimestre", "tp_primeiro_envio_recebedor_trimestre",
        "qt_envio_recebedor_trimestre",
    ],
    "MBK (Mobile Banking)": [
        "device_name", "app_version", "ip_address",
        "latencia_rede_ms", "vl_latencia_rede_media_trimestre",
        "tempo_interacao_ms", "vl_tempo_interacao_medio_trimestre",
        "tempo_processamento_host_ms", "metodo_autenticacao",
        "session_id", "cd_retorno", "topaz_risk_score",
        "topaz_transacao_rejeitada", "is_agendamento_recorrente",
        "qt_aparelhos_distintos_trimestre",
    ],
    "AOX (Cadastro)": [
        "nr_idade", "qt_tempo_relacionamento_mes",
        "ds_sexo", "ds_estado_civil", "vl_renda_cliente", "qt_dependentes",
    ],
    "DNA (Segmentação)": [
        "ds_segmento",
    ],
}


# =========================================================
# LOAD DATA
# =========================================================
def load_data():
    print("\n" + "=" * 70)
    print("  CARREGAMENTO DE DADOS E ARTEFATOS")
    print("=" * 70)

    if not LGBM_PATH.exists():
        print(f"  ❌ Modelo não encontrado: {LGBM_PATH}")
        sys.exit(1)

    lgbm = joblib.load(LGBM_PATH)
    print(f"  ✅ LightGBM carregado: {type(lgbm).__name__}")

    # Features
    if hasattr(lgbm, "feature_name_"):
        features = list(lgbm.feature_name_)
    elif LGBM_FEATURES_PATH.exists():
        with open(LGBM_FEATURES_PATH) as f:
            features = json.load(f)
    else:
        print("  ❌ Features não encontradas!")
        sys.exit(1)
    print(f"  ✅ Features: {len(features)}")

    # Dados de teste
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]
    print(f"  ✅ X_test: {X_test.shape} | Fraudes: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

    # Garantir que todas as features existam
    for f in features:
        if f not in X_test.columns:
            X_test[f] = 0
    X = X_test[features].fillna(0)

    return lgbm, features, X, y_test


# =========================================================
# TESTE 1: Importância LightGBM (split + gain)
# =========================================================
def test_lgbm_importance(lgbm, features) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  TESTE 1: Importância LightGBM (Split + Gain)")
    print("=" * 70)

    imp_split = lgbm.feature_importances_  # default = split
    try:
        imp_gain = lgbm.booster_.feature_importance(importance_type="gain")
    except Exception:
        imp_gain = np.zeros(len(features))

    df = pd.DataFrame({
        "feature": features,
        "importance_split": imp_split,
        "importance_gain": imp_gain,
    })
    df["rank_split"] = df["importance_split"].rank(ascending=False).astype(int)
    df["rank_gain"] = df["importance_gain"].rank(ascending=False).astype(int)
    df["rank_avg"] = ((df["rank_split"] + df["rank_gain"]) / 2).round(1)

    # Normalizar para 0-100
    df["split_pct"] = (df["importance_split"] / df["importance_split"].sum() * 100).round(3)
    df["gain_pct"] = (df["importance_gain"] / df["importance_gain"].sum() * 100).round(3) if df["importance_gain"].sum() > 0 else 0

    zero_split = (df["importance_split"] == 0).sum()
    zero_gain = (df["importance_gain"] == 0).sum()
    print(f"  Features com split=0: {zero_split}")
    print(f"  Features com gain=0:  {zero_gain}")
    print(f"  Top 5 (split): {df.nsmallest(5, 'rank_split')[['feature', 'split_pct']].to_string(index=False)}")

    return df.sort_values("rank_avg")


# =========================================================
# TESTE 2: Permutation Importance
# =========================================================
def test_permutation_importance(lgbm, X, y, n_repeats=10) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(f"  TESTE 2: Permutation Importance ({n_repeats} repetições)")
    print("=" * 70)

    t0 = time.time()
    result = permutation_importance(
        lgbm, X, y, n_repeats=n_repeats, scoring="roc_auc",
        random_state=42, n_jobs=-1
    )
    elapsed = time.time() - t0
    print(f"  ✅ Concluído em {elapsed:.1f}s")

    df = pd.DataFrame({
        "feature": X.columns,
        "perm_importance_mean": result.importances_mean,
        "perm_importance_std": result.importances_std,
    })
    df["perm_rank"] = df["perm_importance_mean"].rank(ascending=False).astype(int)

    # Features com importância negativa ou zero = candidatas fortes à remoção
    negative = (df["perm_importance_mean"] <= 0).sum()
    print(f"  Features com importância ≤ 0: {negative} (candidatas à remoção)")
    print(f"  Top 5: {df.nsmallest(5, 'perm_rank')[['feature', 'perm_importance_mean']].to_string(index=False)}")

    return df


# =========================================================
# TESTE 3: Correlação entre Features (Spearman)
# =========================================================
def test_correlation(X, threshold=0.90) -> Tuple[pd.DataFrame, List[Tuple]]:
    print("\n" + "=" * 70)
    print(f"  TESTE 3: Correlação Spearman (threshold={threshold})")
    print("=" * 70)

    corr_matrix = X.corr(method="spearman")

    # Encontrar pares altamente correlacionados
    high_corr_pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                high_corr_pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "correlation": round(corr_matrix.iloc[i, j], 4),
                    "abs_corr": round(abs(corr_matrix.iloc[i, j]), 4),
                })

    df_pairs = pd.DataFrame(high_corr_pairs).sort_values("abs_corr", ascending=False)
    print(f"  Pares com |corr| ≥ {threshold}: {len(df_pairs)}")
    if len(df_pairs) > 0:
        print(f"  Top 5 pares:")
        for _, row in df_pairs.head(5).iterrows():
            print(f"    {row['feature_1']} ↔ {row['feature_2']}: {row['correlation']:.4f}")

    # Média de correlação absoluta por feature
    mean_abs_corr = corr_matrix.abs().mean()
    df_corr = pd.DataFrame({
        "feature": mean_abs_corr.index,
        "mean_abs_corr": mean_abs_corr.values,
    })

    return df_corr, df_pairs


# =========================================================
# TESTE 4: Mutual Information
# =========================================================
def test_mutual_information(X, y) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  TESTE 4: Mutual Information (dependência não-linear com target)")
    print("=" * 70)

    t0 = time.time()
    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    elapsed = time.time() - t0
    print(f"  ✅ Concluído em {elapsed:.1f}s")

    df = pd.DataFrame({
        "feature": X.columns,
        "mutual_info": mi,
    })
    df["mi_rank"] = df["mutual_info"].rank(ascending=False).astype(int)

    zero_mi = (df["mutual_info"] < 0.001).sum()
    print(f"  Features com MI ≈ 0: {zero_mi}")
    print(f"  Top 5: {df.nsmallest(5, 'mi_rank')[['feature', 'mutual_info']].to_string(index=False)}")

    return df


# =========================================================
# TESTE 5: Levene (Heterocedasticidade entre classes)
# =========================================================
def test_levene(X, y) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  TESTE 5: Teste de Levene (heterocedasticidade fraude vs normal)")
    print("=" * 70)

    results = []
    fraud_mask = y == 1
    normal_mask = y == 0

    for col in X.columns:
        vals_normal = X.loc[normal_mask, col].dropna()
        vals_fraud = X.loc[fraud_mask, col].dropna()

        if len(vals_fraud) < 2 or len(vals_normal) < 2:
            results.append({"feature": col, "levene_stat": np.nan, "levene_p": np.nan})
            continue

        try:
            stat, p = levene(vals_normal, vals_fraud, center="median")
            results.append({"feature": col, "levene_stat": round(stat, 4), "levene_p": round(p, 6)})
        except Exception:
            results.append({"feature": col, "levene_stat": np.nan, "levene_p": np.nan})

    df = pd.DataFrame(results)
    sig = (df["levene_p"] < 0.05).sum()
    print(f"  Features com variância significativamente diferente (p<0.05): {sig}/{len(df)}")

    return df


# =========================================================
# TESTE 6: Mann-Whitney U (separação univariada)
# =========================================================
def test_mann_whitney(X, y) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  TESTE 6: Mann-Whitney U (separação univariada fraude vs normal)")
    print("=" * 70)

    results = []
    fraud_mask = y == 1
    normal_mask = y == 0

    for col in X.columns:
        vals_normal = X.loc[normal_mask, col].dropna()
        vals_fraud = X.loc[fraud_mask, col].dropna()

        if len(vals_fraud) < 2 or len(vals_normal) < 2:
            results.append({
                "feature": col, "mw_stat": np.nan, "mw_p": np.nan,
                "effect_size_r": np.nan,
            })
            continue

        try:
            stat, p = mannwhitneyu(vals_fraud, vals_normal, alternative="two-sided")
            n1, n2 = len(vals_fraud), len(vals_normal)
            z = (stat - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            r = abs(z) / np.sqrt(n1 + n2)  # effect size
            results.append({
                "feature": col, "mw_stat": round(stat, 2), "mw_p": round(p, 8),
                "effect_size_r": round(r, 4),
            })
        except Exception:
            results.append({
                "feature": col, "mw_stat": np.nan, "mw_p": np.nan,
                "effect_size_r": np.nan,
            })

    df = pd.DataFrame(results)
    df["mw_rank"] = df["effect_size_r"].rank(ascending=False).astype(int)
    sig = (df["mw_p"] < 0.05).sum()
    print(f"  Features com separação significativa (p<0.05): {sig}/{len(df)}")
    print(f"  Top 5 (effect size): {df.nsmallest(5, 'mw_rank')[['feature', 'effect_size_r']].to_string(index=False)}")

    return df


# =========================================================
# TESTE 7: VIF (Variance Inflation Factor)
# =========================================================
def test_vif(X, max_features=50) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(f"  TESTE 7: VIF — Variance Inflation Factor")
    print("=" * 70)

    # VIF é computacionalmente caro com 80 features
    # Selecionar as top features por variância para manter o cálculo viável
    var_order = X.var().sort_values(ascending=False)
    # Remover features com variância zero
    non_zero_var = var_order[var_order > 1e-10].index[:max_features]
    X_vif = X[non_zero_var].copy()

    # Remover colunas com valores infinitos
    X_vif = X_vif.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"  Calculando VIF para {len(X_vif.columns)} features...")
    t0 = time.time()

    vif_results = []
    X_arr = X_vif.values
    for i in range(X_arr.shape[1]):
        try:
            vif = variance_inflation_factor(X_arr, i)
            vif_results.append({
                "feature": X_vif.columns[i],
                "vif": round(vif, 2) if not np.isinf(vif) else 999.0,
            })
        except Exception:
            vif_results.append({"feature": X_vif.columns[i], "vif": np.nan})

    elapsed = time.time() - t0
    print(f"  ✅ Concluído em {elapsed:.1f}s")

    df = pd.DataFrame(vif_results)
    high_vif = (df["vif"] > 10).sum()
    print(f"  Features com VIF > 10 (multicolinearidade alta): {high_vif}")
    print(f"  Features com VIF > 50 (severa): {(df['vif'] > 50).sum()}")

    return df


# =========================================================
# TESTE 8: PCA — Variância Explicada Acumulada
# =========================================================
def test_pca(X) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("  TESTE 8: PCA — Variância Explicada Acumulada")
    print("=" * 70)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.replace([np.inf, -np.inf], np.nan).fillna(0))

    pca = PCA(random_state=42)
    pca.fit(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    # Marcos importantes
    thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
    n_components = {}
    for th in thresholds:
        n = int(np.searchsorted(cumvar, th) + 1)
        n_components[f"{int(th*100)}%"] = min(n, len(cumvar))
        print(f"  {int(th*100)}% variância: {min(n, len(cumvar))} componentes (de {len(cumvar)})")

    pca_data = {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": cumvar,
        "n_components": n_components,
        "total_features": len(X.columns),
    }

    # Ratio de compressão para 90%
    n90 = n_components.get("90%", len(X.columns))
    ratio = (1 - n90 / len(X.columns)) * 100
    print(f"\n  📊 Compressão possível para 90% de variância: {ratio:.1f}% ({len(X.columns)} → {n90} componentes)")

    return pca_data


# =========================================================
# TESTE 9: Near-Zero Variance
# =========================================================
def test_near_zero_variance(X, threshold_ratio=19.0, threshold_pct=10.0) -> pd.DataFrame:
    """
    Identifica features com variância quase zero.
    
    - threshold_ratio: razão entre frequência do valor mais comum e segundo mais comum (> 19 = suspeita)
    - threshold_pct: % de valores únicos sobre total (< 10% = suspeita)
    """
    print("\n" + "=" * 70)
    print("  TESTE 9: Near-Zero Variance")
    print("=" * 70)

    results = []
    n = len(X)
    for col in X.columns:
        vals = X[col].dropna()
        n_unique = vals.nunique()
        pct_unique = n_unique / n * 100

        vc = vals.value_counts()
        if len(vc) >= 2:
            freq_ratio = vc.iloc[0] / vc.iloc[1]
        else:
            freq_ratio = n  # apenas 1 valor único

        is_nzv = freq_ratio > threshold_ratio and pct_unique < threshold_pct

        results.append({
            "feature": col,
            "n_unique": n_unique,
            "pct_unique": round(pct_unique, 2),
            "freq_ratio": round(freq_ratio, 2),
            "is_near_zero_var": is_nzv,
        })

    df = pd.DataFrame(results)
    nzv_count = df["is_near_zero_var"].sum()
    print(f"  Features near-zero-variance: {nzv_count}/{len(df)}")
    if nzv_count > 0:
        nzv_features = df[df["is_near_zero_var"]]["feature"].tolist()
        print(f"    → {nzv_features[:10]}{'...' if len(nzv_features) > 10 else ''}")

    return df


# =========================================================
# TESTE 10: Análise por Fonte de Dados
# =========================================================
def test_source_analysis(consolidated: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  TESTE 10: Análise de Contribuição por Fonte de Dados")
    print("=" * 70)

    # Mapear cada feature para seu campo bruto de origem
    feature_sources = []
    for feat in consolidated["feature"]:
        source = FEATURE_TO_SOURCE.get(feat, "DESCONHECIDA")
        if isinstance(source, list):
            source = source[0]  # usar a fonte primária

        # Descobrir qual sistema fonte
        sistema = "DERIVADA"
        for sistema_nome, campos in SOURCE_FIELDS.items():
            if source in campos:
                sistema = sistema_nome
                break

        feature_sources.append({"feature": feat, "campo_bruto": source, "sistema_fonte": sistema})

    df_sources = pd.DataFrame(feature_sources)
    df_merged = consolidated.merge(df_sources, on="feature", how="left")

    # Sumarizar por sistema
    summary = df_merged.groupby("sistema_fonte").agg(
        n_features=("feature", "count"),
        avg_composite_score=("composite_score", "mean"),
        max_composite_score=("composite_score", "max"),
        n_removable=("recomendacao", lambda x: (x == "REMOVER").sum()),
    ).round(3)

    print(f"\n  Contribuição por Sistema Fonte:")
    for idx, row in summary.iterrows():
        print(f"    {idx}: {row['n_features']} features | "
              f"Score médio: {row['avg_composite_score']:.3f} | "
              f"Removíveis: {row['n_removable']}")

    return df_merged


# =========================================================
# TESTE 11: Simulação de Remoção Incremental
# =========================================================
def test_removal_simulation(lgbm, X, y, consolidated: pd.DataFrame, steps=10) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  TESTE 11: Simulação de Remoção Incremental de Features")
    print("=" * 70)

    # Baseline
    y_proba_full = lgbm.predict_proba(X)[:, 1]
    auc_full = roc_auc_score(y, y_proba_full)
    print(f"  Baseline AUC (todas {len(X.columns)} features): {auc_full:.6f}")

    # Ordenar features pela pior composite score (candidatas à remoção)
    removal_order = consolidated.sort_values("composite_score", ascending=True)["feature"].tolist()

    results = [{"n_removed": 0, "n_remaining": len(X.columns), "auc": auc_full, "delta_auc": 0.0}]

    # Remover em blocos
    remove_sizes = np.linspace(5, min(40, len(X.columns) - 10), steps).astype(int)
    remove_sizes = sorted(set(remove_sizes))

    for n_remove in remove_sizes:
        features_to_remove = removal_order[:n_remove]
        features_remaining = [f for f in X.columns if f not in features_to_remove]

        if len(features_remaining) < 10:
            break

        X_reduced = X[features_remaining]
        try:
            y_proba = lgbm.predict_proba(X_reduced)[:, 1]
            auc = roc_auc_score(y, y_proba)
        except Exception:
            # LightGBM pode falhar se features faltam — recriar com zeros
            X_temp = X.copy()
            for f in features_to_remove:
                X_temp[f] = 0
            y_proba = lgbm.predict_proba(X_temp)[:, 1]
            auc = roc_auc_score(y, y_proba)

        delta = auc - auc_full
        results.append({
            "n_removed": n_remove,
            "n_remaining": len(features_remaining),
            "auc": round(auc, 6),
            "delta_auc": round(delta, 6),
        })
        print(f"    -{n_remove:2d} features → AUC={auc:.6f} (Δ={delta:+.6f})")

    return pd.DataFrame(results)


# =========================================================
# CONSOLIDAÇÃO: Score Composto de Relevância
# =========================================================
def consolidate_results(
    df_lgbm, df_perm, df_corr, df_mi, df_levene, df_mw, df_vif, df_nzv,
    features
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  CONSOLIDAÇÃO: Score Composto de Relevância")
    print("=" * 70)

    df = pd.DataFrame({"feature": features})

    # Merge de todos os testes
    df = df.merge(df_lgbm[["feature", "split_pct", "gain_pct", "rank_avg"]], on="feature", how="left")
    df = df.merge(df_perm[["feature", "perm_importance_mean", "perm_rank"]], on="feature", how="left")
    df = df.merge(df_corr[["feature", "mean_abs_corr"]], on="feature", how="left")
    df = df.merge(df_mi[["feature", "mutual_info", "mi_rank"]], on="feature", how="left")
    df = df.merge(df_levene[["feature", "levene_p"]], on="feature", how="left")
    df = df.merge(df_mw[["feature", "effect_size_r", "mw_rank"]], on="feature", how="left")
    df = df.merge(df_nzv[["feature", "is_near_zero_var", "freq_ratio"]], on="feature", how="left")

    if df_vif is not None and len(df_vif) > 0:
        df = df.merge(df_vif[["feature", "vif"]], on="feature", how="left")
    else:
        df["vif"] = np.nan

    # Normalizar cada métrica para 0-1 (1 = mais importante)
    def _norm(s):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)

    df["norm_split"] = _norm(df["split_pct"].fillna(0))
    df["norm_gain"] = _norm(df["gain_pct"].fillna(0))
    df["norm_perm"] = _norm(df["perm_importance_mean"].fillna(0))
    df["norm_mi"] = _norm(df["mutual_info"].fillna(0))
    df["norm_mw"] = _norm(df["effect_size_r"].fillna(0))

    # Score composto ponderado
    # Pesos: Perm (30%) + MI (20%) + LGBM_gain (20%) + LGBM_split (15%) + MW (15%)
    df["composite_score"] = (
        0.30 * df["norm_perm"] +
        0.20 * df["norm_mi"] +
        0.20 * df["norm_gain"] +
        0.15 * df["norm_split"] +
        0.15 * df["norm_mw"]
    ).round(4)

    df["composite_rank"] = df["composite_score"].rank(ascending=False).astype(int)

    # Recomendação
    def _recommend(row):
        score = row["composite_score"]
        nzv = row.get("is_near_zero_var", False)
        perm = row.get("perm_importance_mean", 0)

        if nzv and perm <= 0:
            return "REMOVER"
        elif score < 0.05 and perm <= 0:
            return "REMOVER"
        elif score < 0.10:
            return "AVALIAR"
        elif score < 0.20:
            return "MANTER_BAIXA"
        else:
            return "MANTER"

    df["recomendacao"] = df.apply(_recommend, axis=1)

    counts = df["recomendacao"].value_counts()
    print(f"\n  Recomendações:")
    for rec, count in counts.items():
        icon = {"MANTER": "✅", "MANTER_BAIXA": "🟡", "AVALIAR": "🟠", "REMOVER": "❌"}.get(rec, "?")
        print(f"    {icon} {rec}: {count} features")

    return df.sort_values("composite_rank")


# =========================================================
# GRÁFICOS
# =========================================================
def plot_dashboard(consolidated, pca_data, df_pairs, removal_sim):
    print("\n" + "=" * 70)
    print("  GERANDO GRÁFICOS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(
        "Análise de Relevância de Features — Motor Antifraude PIX v2.1\n"
        f"{len(consolidated)} features | {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        fontsize=15, fontweight="bold", color=COLORS["primary"], y=1.02,
    )

    # 1. Top 20 Features (Composite Score)
    ax = axes[0, 0]
    top20 = consolidated.head(20).sort_values("composite_score")
    colors_bar = [COLORS["primary"] if r == "MANTER" else
                  COLORS["warning"] if r == "MANTER_BAIXA" else
                  COLORS["secondary"] for r in top20["recomendacao"]]
    ax.barh(top20["feature"], top20["composite_score"], color=colors_bar)
    ax.set_title("Top 20 Features (Score Composto)", fontweight="bold")
    ax.set_xlabel("Score")

    # 2. Bottom 20 Features (candidatas à remoção)
    ax = axes[0, 1]
    bottom20 = consolidated.tail(20).sort_values("composite_score", ascending=False)
    colors_bar = [COLORS["secondary"] if r == "REMOVER" else
                  COLORS["warning"] if r == "AVALIAR" else
                  COLORS["accent"] for r in bottom20["recomendacao"]]
    ax.barh(bottom20["feature"], bottom20["composite_score"], color=colors_bar)
    ax.set_title("Bottom 20 Features (Candidatas à Remoção)", fontweight="bold")
    ax.set_xlabel("Score")

    # 3. PCA Variância Acumulada
    ax = axes[0, 2]
    cumvar = pca_data["cumulative_variance"]
    n_components = pca_data["n_components"]
    ax.plot(range(1, len(cumvar) + 1), cumvar * 100, color=COLORS["primary"], lw=2.5)
    ax.axhline(y=90, color=COLORS["warning"], ls="--", lw=1.5, label=f"90% → {n_components.get('90%', '?')} comp.")
    ax.axhline(y=95, color=COLORS["secondary"], ls="--", lw=1.5, label=f"95% → {n_components.get('95%', '?')} comp.")
    ax.set_xlabel("Nº de Componentes")
    ax.set_ylabel("Variância Acumulada (%)")
    ax.set_title("PCA — Variância Explicada", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True)

    # 4. Distribuição dos Scores Compostos
    ax = axes[1, 0]
    scores = consolidated["composite_score"]
    ax.hist(scores, bins=30, color=COLORS["primary"], alpha=0.7, edgecolor="#333")
    ax.axvline(x=0.10, color=COLORS["warning"], ls="--", lw=2, label="Limiar AVALIAR (0.10)")
    ax.axvline(x=0.05, color=COLORS["secondary"], ls="--", lw=2, label="Limiar REMOVER (0.05)")
    ax.set_xlabel("Score Composto")
    ax.set_ylabel("Contagem")
    ax.set_title("Distribuição dos Scores", fontweight="bold")
    ax.legend(fontsize=9)

    # 5. Simulação de Remoção
    ax = axes[1, 1]
    if removal_sim is not None and len(removal_sim) > 1:
        ax.plot(removal_sim["n_removed"], removal_sim["auc"], color=COLORS["primary"], lw=2.5, marker="o", ms=6)
        ax.axhline(y=removal_sim["auc"].iloc[0], color=COLORS["text_muted"], ls=":", lw=1, label="Baseline")
        ax.axhline(y=removal_sim["auc"].iloc[0] - 0.001, color=COLORS["warning"], ls="--", lw=1.5,
                    label="Baseline - 0.001")
        ax.set_xlabel("Features Removidas")
        ax.set_ylabel("AUC-ROC")
        ax.set_title("Impacto da Remoção no AUC", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True)

    # 6. Recomendações (pie)
    ax = axes[1, 2]
    rec_counts = consolidated["recomendacao"].value_counts()
    colors_pie = {
        "MANTER": COLORS["primary"], "MANTER_BAIXA": COLORS["accent"],
        "AVALIAR": COLORS["warning"], "REMOVER": COLORS["secondary"],
    }
    ax.pie(
        rec_counts.values,
        labels=[f"{k}\n({v})" for k, v in rec_counts.items()],
        colors=[colors_pie.get(k, "#888") for k in rec_counts.index],
        autopct="%1.0f%%", startangle=90,
        textprops={"color": "white", "fontsize": 11},
    )
    ax.set_title("Recomendações por Feature", fontweight="bold")

    plt.tight_layout()
    output = OUTPUT_DIR / "dashboard_features.png"
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ Dashboard salvo: {output}")

    # Mapa de correlação (top 40 features)
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 14))
    top40 = consolidated.head(40)["feature"].tolist()
    corr_sub = consolidated.set_index("feature").loc[
        [f for f in top40 if f in consolidated["feature"].values]
    ]
    # Vamos plotar a correlação entre as top 40
    from matplotlib.colors import LinearSegmentedColormap
    X_global = pd.read_csv(X_TEST_PATH)
    for f in top40:
        if f not in X_global.columns:
            X_global[f] = 0
    corr_top = X_global[top40].corr(method="spearman")
    cmap = LinearSegmentedColormap.from_list("custom", ["#ff6b6b", "#1a1d23", "#00d4aa"])
    im = ax2.imshow(corr_top.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(top40)))
    ax2.set_yticks(range(len(top40)))
    ax2.set_xticklabels(top40, rotation=90, fontsize=7)
    ax2.set_yticklabels(top40, fontsize=7)
    ax2.set_title("Correlação Spearman — Top 40 Features", fontweight="bold", pad=20)
    fig2.colorbar(im, ax=ax2, shrink=0.8)
    plt.tight_layout()
    output2 = OUTPUT_DIR / "correlation_matrix_top40.png"
    fig2.savefig(output2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"  ✅ Correlação salva: {output2}")


# =========================================================
# EXPORTAR RESULTADOS
# =========================================================
def export_results(consolidated, df_pairs, pca_data, removal_sim):
    print("\n" + "=" * 70)
    print("  EXPORTANDO RESULTADOS")
    print("=" * 70)

    # CSV principal
    csv_path = OUTPUT_DIR / "feature_relevance_scores.csv"
    consolidated.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  ✅ Scores: {csv_path}")

    # CSV pares correlacionados
    if len(df_pairs) > 0:
        pairs_path = OUTPUT_DIR / "high_correlation_pairs.csv"
        df_pairs.to_csv(pairs_path, index=False)
        print(f"  ✅ Pares correlacionados: {pairs_path}")

    # CSV simulação de remoção
    if removal_sim is not None:
        sim_path = OUTPUT_DIR / "removal_simulation.csv"
        removal_sim.to_csv(sim_path, index=False)
        print(f"  ✅ Simulação: {sim_path}")

    # JSON resumo
    summary = {
        "data_geracao": datetime.now().isoformat(),
        "total_features": len(consolidated),
        "recomendacoes": consolidated["recomendacao"].value_counts().to_dict(),
        "features_remover": consolidated[consolidated["recomendacao"] == "REMOVER"]["feature"].tolist(),
        "features_avaliar": consolidated[consolidated["recomendacao"] == "AVALIAR"]["feature"].tolist(),
        "pca": {
            "n_90pct": pca_data["n_components"].get("90%"),
            "n_95pct": pca_data["n_components"].get("95%"),
            "total": pca_data["total_features"],
        },
        "top10_features": consolidated.head(10)[["feature", "composite_score"]].to_dict("records"),
        "bottom10_features": consolidated.tail(10)[["feature", "composite_score"]].to_dict("records"),
    }
    json_path = OUTPUT_DIR / "feature_analysis_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Resumo JSON: {json_path}")

    # Relatório de texto
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  RELATÓRIO DE RELEVÂNCIA DE FEATURES")
    report_lines.append(f"  {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append(f"Total de features analisadas: {len(consolidated)}")
    report_lines.append("")
    report_lines.append("RECOMENDAÇÕES:")
    for rec in ["MANTER", "MANTER_BAIXA", "AVALIAR", "REMOVER"]:
        feats = consolidated[consolidated["recomendacao"] == rec]["feature"].tolist()
        report_lines.append(f"\n  {rec} ({len(feats)} features):")
        for f in feats:
            score = consolidated.loc[consolidated["feature"] == f, "composite_score"].values[0]
            report_lines.append(f"    • {f} (score={score:.4f})")

    report_lines.append(f"\nPCA — 90% variância: {pca_data['n_components'].get('90%')} componentes")
    report_lines.append(f"PCA — 95% variância: {pca_data['n_components'].get('95%')} componentes")

    report_lines.append(f"\nPares altamente correlacionados (|r| ≥ 0.90): {len(df_pairs)}")
    if len(df_pairs) > 0:
        for _, row in df_pairs.head(10).iterrows():
            report_lines.append(f"  {row['feature_1']} ↔ {row['feature_2']}: {row['correlation']:.4f}")

    txt_path = OUTPUT_DIR / "feature_analysis_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  ✅ Relatório TXT: {txt_path}")


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n" + "=" * 70)
    print("  ANÁLISE ESTATÍSTICA DE RELEVÂNCIA DE FEATURES")
    print(f"  Motor Antifraude PIX v2.1 | {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 70)

    t_global = time.time()

    # Carregar dados
    lgbm, features, X, y = load_data()

    # Executar testes
    df_lgbm = test_lgbm_importance(lgbm, features)
    df_perm = test_permutation_importance(lgbm, X, y, n_repeats=10)
    df_corr, df_pairs = test_correlation(X, threshold=0.90)
    df_mi = test_mutual_information(X, y)
    df_levene_result = test_levene(X, y)
    df_mw = test_mann_whitney(X, y)
    df_vif = test_vif(X, max_features=50)
    pca_data = test_pca(X)
    df_nzv = test_near_zero_variance(X)

    # Consolidar
    consolidated = consolidate_results(
        df_lgbm, df_perm, df_corr, df_mi, df_levene_result, df_mw, df_vif, df_nzv,
        features
    )

    # Análise por fonte
    consolidated_with_source = test_source_analysis(consolidated)

    # Simulação de remoção
    removal_sim = test_removal_simulation(lgbm, X, y, consolidated, steps=8)

    # Gráficos
    plot_dashboard(consolidated, pca_data, df_pairs, removal_sim)

    # Exportar
    export_results(consolidated_with_source, df_pairs, pca_data, removal_sim)

    elapsed_total = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"  ✅ ANÁLISE COMPLETA em {elapsed_total:.1f}s")
    print(f"  Saída: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    # Resumo executivo final
    n_remover = (consolidated["recomendacao"] == "REMOVER").sum()
    n_avaliar = (consolidated["recomendacao"] == "AVALIAR").sum()
    n90 = pca_data["n_components"].get("90%", len(features))

    print(f"  📊 RESUMO EXECUTIVO:")
    print(f"     Features totais:           {len(features)}")
    print(f"     Recomendadas para REMOVER:  {n_remover}")
    print(f"     Recomendadas para AVALIAR:  {n_avaliar}")
    print(f"     PCA 90% variância:          {n90} componentes")
    print(f"     Redução potencial:          {n_remover + n_avaliar} features ({(n_remover + n_avaliar)/len(features)*100:.0f}%)")

    if removal_sim is not None and len(removal_sim) > 1:
        safe_removal = removal_sim[removal_sim["delta_auc"] > -0.001]
        if len(safe_removal) > 1:
            max_safe = safe_removal["n_removed"].max()
            print(f"     Remoção segura (ΔAUC < 0.001): até {max_safe} features")


if __name__ == "__main__":
    main()
