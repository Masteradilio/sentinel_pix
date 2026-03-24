"""
gerar_artefatos_faltantes.py v3 — Gera X_test/y_test com TODAS as 80 features
===============================================================================

Mudanças v2 → v3:
  1. Aplica PixPreprocessor.transform() nos dados ANTES do split
     para gerar as 18 features extras (renda, perfil, login, etc.)
  2. Se preprocessor falhar, gera as 18 features INLINE com a mesma
     lógica do preprocessing.py v3.1
  3. Valida que X_test tem EXATAMENTE as mesmas features do modelo LGBM
  4. Diagnóstico detalhado: mostra cobertura de cada feature nova
  5. Gera X_train.csv opcional para recalibrar scoring_config

Uso:
  python gerar_artefatos_faltantes.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURAÇÃO DE PATHS
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
DADOS_DIR = PROJECT_ROOT / "dados"
MODELOS_DIR = PROJECT_ROOT / "backend" / "modelos"
BACKEND_DIR = PROJECT_ROOT / "backend"

# Garantir que diretórios existem
ARTEFATOS_DIR.mkdir(parents=True, exist_ok=True)

print(f"  Project root: {PROJECT_ROOT}")
print(f"  Artefatos:    {ARTEFATOS_DIR}")
print(f"  Dados:        {DADOS_DIR}")
print(f"  Modelos:      {MODELOS_DIR}")

# =========================================================
# IMPORTAR PixPreprocessor
# =========================================================
sys.path.insert(0, str(MODELOS_DIR))
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

PixPreprocessor = None
for import_path, import_stmt in [
    ("modelos", "from preprocessing import PixPreprocessor"),
    ("backend.modelos", "from backend.modelos.preprocessing import PixPreprocessor"),
    ("backend.core", "from backend.core.preprocessing import PixPreprocessor"),
]:
    try:
        exec(import_stmt)
        print(f"\n  ✅ PixPreprocessor importado via {import_path}")
        break
    except ImportError:
        continue

if PixPreprocessor is None:
    print(f"\n  ⚠️  PixPreprocessor não importado — usará geração inline das features")

# =========================================================
# PATHS DOS ARTEFATOS
# =========================================================
LGBM_MODEL_PATH = ARTEFATOS_DIR / "model_lightgbm.joblib"
IF_MODEL_PATH = ARTEFATOS_DIR / "model_isolation_forest.joblib"
PREPROCESSOR_PATH = ARTEFATOS_DIR / "preprocessing.joblib"
IF_SCALER_PATH = ARTEFATOS_DIR / "scaler_isolation_forest.joblib"
IF_CONFIG_PATH = ARTEFATOS_DIR / "isolation_forest_config.json"
IF_RAW_TRAIN_PATH = ARTEFATOS_DIR / "if_ref_raw_train.npy"
METRICAS_LGBM_PATH = ARTEFATOS_DIR / "metricas_lightgbm.json"
METRICAS_IF_PATH = ARTEFATOS_DIR / "metricas_isolation_forest.json"
FEATURE_IMP_LGBM_PATH = ARTEFATOS_DIR / "feature_importance_lightgbm.csv"
PREDICOES_LGBM_PATH = ARTEFATOS_DIR / "predicoes_teste_lightgbm.csv"

# Dados
BASE_MVP_PATH = DADOS_DIR / "base_mvp_model_ready.csv"

# Tentar caminhos alternativos para a base
BASE_ALTERNATIVES = [
    DADOS_DIR / "base_mvp_model_ready.csv",
    PROJECT_ROOT / "dados" / "base_mvp_model_ready.csv",
    BACKEND_DIR / "dados" / "base_mvp_model_ready.csv",
    ARTEFATOS_DIR / "base_mvp_model_ready.csv",
    DADOS_DIR / "base_mvp.csv",
    DADOS_DIR / "base_completa.csv",
]

# Saída
LGBM_FEATURES_PATH = ARTEFATOS_DIR / "lgbm_features.json"
IF_FEATURES_PATH = ARTEFATOS_DIR / "if_features.json"
X_TEST_PATH = ARTEFATOS_DIR / "X_test.csv"
Y_TEST_PATH = ARTEFATOS_DIR / "y_test.csv"
X_TRAIN_PATH = ARTEFATOS_DIR / "X_train.csv"
Y_TRAIN_PATH = ARTEFATOS_DIR / "y_train.csv"
ARTEFATOS_MANIFEST_PATH = ARTEFATOS_DIR / "artefatos_manifest.json"


def print_header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def check_file(path: Path, required: bool = True) -> bool:
    exists = path.exists()
    status = "✅" if exists else ("❌" if required else "⚠️")
    size = f"({path.stat().st_size / 1024:.0f} KB)" if exists else ""
    print(f"  {status} {path.name} {size}")
    return exists


def _safe_float(val, default=None):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        v = float(val)
        return default if v != v else v
    except (ValueError, TypeError):
        return default


# =========================================================
# 1. VALIDAR ARTEFATOS EXISTENTES
# =========================================================
def validate_existing_artifacts():
    print_header("1. VALIDAÇÃO DOS ARTEFATOS EXISTENTES")

    print("\n  Modelos:")
    check_file(LGBM_MODEL_PATH, required=True)
    check_file(IF_MODEL_PATH, required=False)
    check_file(PREPROCESSOR_PATH, required=False)
    check_file(IF_SCALER_PATH, required=False)

    print("\n  Configs:")
    check_file(IF_CONFIG_PATH, required=False)
    check_file(IF_RAW_TRAIN_PATH, required=False)
    check_file(METRICAS_LGBM_PATH, required=False)

    print("\n  Dados:")
    found_base = False
    for alt in BASE_ALTERNATIVES:
        if alt.exists():
            print(f"  ✅ Base encontrada: {alt}")
            found_base = True
            break
    if not found_base:
        print(f"  ❌ Base CSV não encontrada em nenhum caminho!")

    if not LGBM_MODEL_PATH.exists():
        print(f"\n  ❌ ERRO CRÍTICO: model_lightgbm.joblib não encontrado!")
        sys.exit(1)


# =========================================================
# 2. EXTRAIR FEATURES DO LGBM (fonte de verdade)
# =========================================================
def extract_lgbm_features() -> list:
    print_header("2. EXTRAINDO FEATURES DO LGBM (fonte de verdade)")

    lgbm_model = joblib.load(LGBM_MODEL_PATH)
    features = None

    for method_name, getter in [
        ("feature_name_", lambda m: list(m.feature_name_)),
        ("booster_.feature_name()", lambda m: m.booster_.feature_name()),
        ("feature_names_in_", lambda m: list(m.feature_names_in_)),
    ]:
        try:
            features = getter(lgbm_model)
            print(f"  Extraído via {method_name}: {len(features)} features")
            break
        except (AttributeError, TypeError):
            continue

    if features is None:
        print("  ❌ Não foi possível extrair features do modelo!")
        sys.exit(1)

    n_model = lgbm_model.n_features_in_ if hasattr(lgbm_model, "n_features_in_") else len(features)
    print(f"  Modelo espera: {n_model} features")

    # Salvar
    with open(LGBM_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Salvo: {LGBM_FEATURES_PATH}")

    # Classificar
    rule_feats = [f for f in features if f.startswith("rule_")]
    flag_feats = [f for f in features if f.endswith("_flag")]
    renda_feats = [f for f in features if "renda" in f.lower()]
    perfil_feats = [f for f in features if any(k in f for k in ["viuvo", "vulneravel", "sexo", "premium", "dependentes"])]
    auth_feats = [f for f in features if any(k in f for k in ["login", "auth", "agendamento", "biometria", "senha"])]

    print(f"\n  Classificação:")
    print(f"    Rules:     {len(rule_feats)} — {rule_feats}")
    print(f"    Flags:     {len(flag_feats)}")
    print(f"    Renda:     {len(renda_feats)} — {renda_feats}")
    print(f"    Perfil:    {len(perfil_feats)} — {perfil_feats}")
    print(f"    Auth:      {len(auth_feats)} — {auth_feats}")

    return features


# =========================================================
# 3. EXTRAIR FEATURES DO IF
# =========================================================
def extract_if_features() -> list:
    print_header("3. EXTRAINDO FEATURES DO IF")

    if IF_CONFIG_PATH.exists():
        with open(IF_CONFIG_PATH, "r") as f:
            config = json.load(f)

        if "features" in config:
            features = config["features"]
            print(f"  IF features: {len(features)}")
            for feat in features:
                print(f"    → {feat}")

            with open(IF_FEATURES_PATH, "w", encoding="utf-8") as f:
                json.dump(features, f, ensure_ascii=False, indent=2)
            print(f"  ✅ Salvo: {IF_FEATURES_PATH}")
            return features

    print("  ⚠️  IF config não encontrada")
    return []


# =========================================================
# 4. GERAR AS 18 FEATURES EXTRAS (inline)
# =========================================================
def generate_missing_features(df: pd.DataFrame, lgbm_features: list) -> pd.DataFrame:
    """
    Gera as features que existem no modelo mas não no CSV.
    Replica EXATAMENTE a lógica do preprocessing.py v3.1.
    """
    existing = set(df.columns)
    needed = set(lgbm_features)
    missing = needed - existing

    if not missing:
        print(f"  ✅ Todas as {len(lgbm_features)} features já existem no DataFrame")
        return df

    print(f"\n  Gerando {len(missing)} features faltantes inline:")

    # ─── RENDA ───
    if "vl_renda_cliente" in missing:
        # Tentar extrair de colunas brutas
        renda_candidates = ["vl_renda", "renda", "vl_renda_mensal", "income", "vl_renda_cliente"]
        for col in renda_candidates:
            if col in df.columns and col != "vl_renda_cliente":
                df["vl_renda_cliente"] = df[col].copy()
                print(f"    + vl_renda_cliente ← {col}")
                break
        else:
            df["vl_renda_cliente"] = 0.0
            print(f"    + vl_renda_cliente = 0.0 (sem coluna fonte)")

    if "renda_missing_flag" in missing:
        if "vl_renda_cliente" in df.columns:
            df["renda_missing_flag"] = (df["vl_renda_cliente"].isna() | (df["vl_renda_cliente"] <= 0)).astype(int)
        else:
            df["renda_missing_flag"] = 1
        print(f"    + renda_missing_flag (cobertura: {(df['renda_missing_flag'] == 0).mean():.1%})")

    if "ratio_pix_renda" in missing:
        if "vl_renda_cliente" in df.columns and "vl_pix" in df.columns:
            renda = df["vl_renda_cliente"].replace(0, np.nan)
            df["ratio_pix_renda"] = (df["vl_pix"] / renda).fillna(0).clip(upper=100)
        else:
            df["ratio_pix_renda"] = 0.0
        print(f"    + ratio_pix_renda (mediana: {df['ratio_pix_renda'].median():.4f})")

    if "pix_over_50pct_renda_flag" in missing:
        if "ratio_pix_renda" in df.columns:
            df["pix_over_50pct_renda_flag"] = (df["ratio_pix_renda"] >= 0.5).astype(int)
        else:
            df["pix_over_50pct_renda_flag"] = 0
        print(f"    + pix_over_50pct_renda_flag (positivos: {df['pix_over_50pct_renda_flag'].sum()})")

    if "pix_over_100pct_renda_flag" in missing:
        if "ratio_pix_renda" in df.columns:
            df["pix_over_100pct_renda_flag"] = (df["ratio_pix_renda"] >= 1.0).astype(int)
        else:
            df["pix_over_100pct_renda_flag"] = 0
        print(f"    + pix_over_100pct_renda_flag (positivos: {df['pix_over_100pct_renda_flag'].sum()})")

    # ─── PERFIL ───
    if "is_sexo_feminino_flag" in missing:
        sexo_candidates = ["ds_sexo", "sexo", "gender"]
        for col in sexo_candidates:
            if col in df.columns:
                df["is_sexo_feminino_flag"] = df[col].astype(str).str.strip().str.upper().isin(["F", "FEMININO", "FEMALE"]).astype(int)
                print(f"    + is_sexo_feminino_flag ← {col} (positivos: {df['is_sexo_feminino_flag'].sum()})")
                break
        else:
            df["is_sexo_feminino_flag"] = 0
            print(f"    + is_sexo_feminino_flag = 0 (sem coluna fonte)")

    if "is_viuvo_flag" in missing:
        estado_civil_candidates = ["ds_estado_civil", "estado_civil", "marital_status"]
        for col in estado_civil_candidates:
            if col in df.columns:
                df["is_viuvo_flag"] = df[col].astype(str).str.strip().str.upper().isin(["VIUVO", "VIÚVO", "VIUVA", "VIÚVA", "V"]).astype(int)
                print(f"    + is_viuvo_flag ← {col} (positivos: {df['is_viuvo_flag'].sum()})")
                break
        else:
            df["is_viuvo_flag"] = 0
            print(f"    + is_viuvo_flag = 0 (sem coluna fonte)")

    if "is_segmento_premium_flag" in missing:
        seg_candidates = ["ds_segmento", "segmento", "segment"]
        for col in seg_candidates:
            if col in df.columns:
                df["is_segmento_premium_flag"] = df[col].astype(str).str.strip().str.upper().isin([
                    "PREMIUM", "PRIVATE", "PERSONNALITE", "ALTA_RENDA", "SELECT", "UNICLASS"
                ]).astype(int)
                print(f"    + is_segmento_premium_flag ← {col} (positivos: {df['is_segmento_premium_flag'].sum()})")
                break
        else:
            df["is_segmento_premium_flag"] = 0
            print(f"    + is_segmento_premium_flag = 0 (sem coluna fonte)")

    if "qt_dependentes" in missing:
        dep_candidates = ["qt_dependentes", "dependentes", "nr_dependentes"]
        for col in dep_candidates:
            if col in df.columns and col != "qt_dependentes":
                df["qt_dependentes"] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                print(f"    + qt_dependentes ← {col}")
                break
        else:
            df["qt_dependentes"] = 0
            print(f"    + qt_dependentes = 0 (sem coluna fonte)")

    if "perfil_vulneravel_se_flag" in missing:
        nr_idade = df.get("nr_idade", pd.Series(0, index=df.index))
        is_viuvo = df.get("is_viuvo_flag", pd.Series(0, index=df.index))
        qt_dep = df.get("qt_dependentes", pd.Series(0, index=df.index))
        df["perfil_vulneravel_se_flag"] = (
            (nr_idade >= 60) &
            (is_viuvo == 1) &
            (qt_dep == 0)
        ).astype(int)
        print(f"    + perfil_vulneravel_se_flag (positivos: {df['perfil_vulneravel_se_flag'].sum()})")

    # ─── RECEBEDOR ───
    if "tp_primeiro_envio_recebedor_trimestre" in missing:
        fr_candidates = ["tp_primeiro_envio_recebedor_trimestre"]
        found = False
        for col in fr_candidates:
            if col in df.columns:
                found = True
                break
        if not found:
            # Derivar de first_receiver_flag se disponível
            if "first_receiver_flag" in df.columns:
                df["tp_primeiro_envio_recebedor_trimestre"] = df["first_receiver_flag"].copy()
                print(f"    + tp_primeiro_envio_recebedor_trimestre ← first_receiver_flag")
            else:
                df["tp_primeiro_envio_recebedor_trimestre"] = 0
                print(f"    + tp_primeiro_envio_recebedor_trimestre = 0")

    if "qt_envio_recebedor_trimestre" in missing:
        candidates = ["qt_envio_recebedor_trimestre"]
        found = False
        for col in candidates:
            if col in df.columns:
                found = True
                break
        if not found:
            # Se first_receiver_flag=1, nunca enviou → qt=0; senão assume 1+
            if "first_receiver_flag" in df.columns:
                df["qt_envio_recebedor_trimestre"] = np.where(
                    df["first_receiver_flag"] == 1, 0, 1
                )
                print(f"    + qt_envio_recebedor_trimestre ← derivado de first_receiver_flag")
            else:
                df["qt_envio_recebedor_trimestre"] = 1
                print(f"    + qt_envio_recebedor_trimestre = 1 (default)")

    # ─── AUTENTICAÇÃO ───
    if "metodo_auth_encoded" in missing:
        auth_candidates = ["metodo_autenticacao", "ds_metodo_autenticacao", "auth_method"]
        for col in auth_candidates:
            if col in df.columns:
                mapping = {}
                raw = df[col].astype(str).str.strip().str.lower()
                df["metodo_auth_encoded"] = raw.map(lambda v: (
                    1 if v in ("1", "bio", "biometria", "biometric") or "bio" in str(v) else
                    2 if v in ("2", "senha", "password") or "senha" in str(v) else
                    3 if v in ("3", "pin") or "pin" in str(v) else
                    0
                ))
                print(f"    + metodo_auth_encoded ← {col}")
                break
        else:
            df["metodo_auth_encoded"] = 0
            print(f"    + metodo_auth_encoded = 0 (sem coluna fonte)")

    if "is_login_senha_flag" in missing:
        if "metodo_auth_encoded" in df.columns:
            df["is_login_senha_flag"] = (df["metodo_auth_encoded"] == 2).astype(int)
        else:
            df["is_login_senha_flag"] = 0
        print(f"    + is_login_senha_flag (positivos: {df['is_login_senha_flag'].sum()})")

    if "is_login_biometria_flag" in missing:
        if "metodo_auth_encoded" in df.columns:
            df["is_login_biometria_flag"] = (df["metodo_auth_encoded"] == 1).astype(int)
        else:
            df["is_login_biometria_flag"] = 0
        print(f"    + is_login_biometria_flag (positivos: {df['is_login_biometria_flag'].sum()})")

    if "is_agendamento_recorrente_flag" in missing:
        ag_candidates = ["is_agendamento_recorrente", "agendamento_recorrente"]
        for col in ag_candidates:
            if col in df.columns:
                df["is_agendamento_recorrente_flag"] = df[col].astype(str).str.strip().str.lower().isin(["true", "1", "sim", "s"]).astype(int)
                print(f"    + is_agendamento_recorrente_flag ← {col}")
                break
        else:
            df["is_agendamento_recorrente_flag"] = 0
            print(f"    + is_agendamento_recorrente_flag = 0")

    # ─── INTERAÇÃO ───
    if "tempo_interacao_missing_flag" in missing:
        tempo_candidates = ["tempo_interacao_ms", "tempo_interacao_ms_final"]
        for col in tempo_candidates:
            if col in df.columns:
                df["tempo_interacao_missing_flag"] = (df[col].isna() | (df[col] <= 0)).astype(int)
                print(f"    + tempo_interacao_missing_flag ← {col} (missing: {df['tempo_interacao_missing_flag'].mean():.1%})")
                break
        else:
            df["tempo_interacao_missing_flag"] = 1
            print(f"    + tempo_interacao_missing_flag = 1 (sem coluna fonte)")

    # ─── TOPAZ ───
    if "topaz_rejeitada_flag" in missing:
        topaz_candidates = ["ds_status_topaz", "status_topaz", "topaz_status"]
        for col in topaz_candidates:
            if col in df.columns:
                df["topaz_rejeitada_flag"] = df[col].astype(str).str.strip().str.upper().isin(["REJEITADA", "REJECTED", "R"]).astype(int)
                print(f"    + topaz_rejeitada_flag ← {col} (positivos: {df['topaz_rejeitada_flag'].sum()})")
                break
        else:
            # Tentar derivar do topaz_score
            if "topaz_score_filled" in df.columns:
                df["topaz_rejeitada_flag"] = (df["topaz_score_filled"] >= 900).astype(int)
                print(f"    + topaz_rejeitada_flag ← topaz_score_filled >= 900 (positivos: {df['topaz_rejeitada_flag'].sum()})")
            else:
                df["topaz_rejeitada_flag"] = 0
                print(f"    + topaz_rejeitada_flag = 0")

    # ─── VERIFICAÇÃO FINAL ───
    still_missing = [f for f in lgbm_features if f not in df.columns]
    if still_missing:
        print(f"\n  ⚠️  Ainda faltam {len(still_missing)} features — preenchendo com 0:")
        for feat in still_missing:
            df[feat] = 0
            print(f"    + {feat} = 0 (fallback)")

    return df


# =========================================================
# 5. GERAR X_TEST / Y_TEST COM TODAS AS FEATURES
# =========================================================
def generate_test_data(lgbm_features: list):
    print_header("4. GERANDO X_test / y_test COM 80 FEATURES")

    # Encontrar base
    base_path = None
    for alt in BASE_ALTERNATIVES:
        if alt.exists():
            base_path = alt
            break

    if base_path is None:
        print(f"  ❌ Base CSV não encontrada!")
        return None, None

    print(f"  ✅ Base: {base_path}")
    df = pd.read_csv(base_path)
    print(f"  Shape original: {df.shape}")
    print(f"  Colunas ({len(df.columns)}): {list(df.columns[:10])}...")

    # Identificar target
    target_col = None
    for col in ["is_fraud", "is_fraude", "fraud", "target", "label", "tp_fraude"]:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        print(f"  ❌ Coluna target não encontrada!")
        print(f"  Colunas disponíveis: {list(df.columns)}")
        return None, None

    print(f"  Target: '{target_col}'")
    print(f"  Distribuição:")
    print(f"    Normal: {(df[target_col] == 0).sum()}")
    print(f"    Fraude: {(df[target_col] == 1).sum()} ({df[target_col].mean()*100:.2f}%)")

    # Separar target ANTES de gerar features
    y = df[target_col].copy()
    df_features = df.drop(columns=[target_col])

    # ─── Tentar preprocessor primeiro ───
    preprocessor_used = False
    if PREPROCESSOR_PATH.exists() and PixPreprocessor is not None:
        print(f"\n  Tentando aplicar PixPreprocessor...")
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            result = preprocessor.transform(df_features)
            if isinstance(result, pd.DataFrame):
                df_features = result
                preprocessor_used = True
                print(f"  ✅ PixPreprocessor aplicado! Shape: {df_features.shape}")
            elif isinstance(result, tuple):
                df_features = result[0] if isinstance(result[0], pd.DataFrame) else pd.DataFrame(result[0])
                preprocessor_used = True
                print(f"  ✅ PixPreprocessor aplicado (tuple)! Shape: {df_features.shape}")
        except Exception as e:
            print(f"  ⚠️  PixPreprocessor falhou: {e}")
            print(f"  Usando geração inline das features...")

    # ─── Gerar features faltantes inline ───
    missing_count_before = sum(1 for f in lgbm_features if f not in df_features.columns)
    if missing_count_before > 0:
        print(f"\n  {missing_count_before} features faltantes — gerando inline...")
        df_features = generate_missing_features(df_features, lgbm_features)

    # ─── Verificação de match ───
    final_missing = [f for f in lgbm_features if f not in df_features.columns]
    final_extra = [f for f in df_features.columns if f not in lgbm_features]

    print(f"\n  ═══ VERIFICAÇÃO DE FEATURES ═══")
    print(f"  Features do modelo:    {len(lgbm_features)}")
    print(f"  Features no DataFrame: {len(df_features.columns)}")
    print(f"  Match:                 {len(lgbm_features) - len(final_missing)}/{len(lgbm_features)}")
    if final_missing:
        print(f"  ❌ Ainda faltam:       {final_missing}")
    else:
        print(f"  ✅ TODAS as features do modelo estão presentes!")

    # ─── Diagnóstico de cobertura das features novas ───
    new_features = [
        "ratio_pix_renda", "vl_renda_cliente", "pix_over_50pct_renda_flag",
        "pix_over_100pct_renda_flag", "renda_missing_flag",
        "perfil_vulneravel_se_flag", "is_sexo_feminino_flag", "is_viuvo_flag",
        "is_segmento_premium_flag", "qt_dependentes",
        "tp_primeiro_envio_recebedor_trimestre", "qt_envio_recebedor_trimestre",
        "is_agendamento_recorrente_flag", "metodo_auth_encoded",
        "is_login_senha_flag", "is_login_biometria_flag",
        "tempo_interacao_missing_flag", "topaz_rejeitada_flag",
    ]
    print(f"\n  ═══ COBERTURA DAS 18 FEATURES NOVAS ═══")
    print(f"  {'Feature':<42s} {'Non-zero':>10s} {'% > 0':>8s} {'Mean':>10s}")
    print(f"  {'─'*42} {'─'*10} {'─'*8} {'─'*10}")
    for feat in new_features:
        if feat in df_features.columns:
            col = df_features[feat]
            n_nonzero = (col != 0).sum()
            pct = n_nonzero / len(col) * 100
            mean_val = col.mean()
            quality = "✅" if pct > 1 else "⚠️" if pct > 0 else "❌"
            print(f"  {quality} {feat:<40s} {n_nonzero:>10,} {pct:>7.1f}% {mean_val:>10.4f}")
        else:
            print(f"  ❌ {feat:<40s} {'MISSING':>10s}")

    # ─── Split ───
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        df_features, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n  Split (random_state=42, test_size=0.2, stratified):")
    print(f"    X_train: {X_train.shape}  fraudes: {y_train.sum()}")
    print(f"    X_test:  {X_test.shape}  fraudes: {y_test.sum()}")

    # ─── Selecionar apenas features do modelo (ordem correta) ───
    X_test_final = X_test[lgbm_features].copy()
    X_train_final = X_train[lgbm_features].copy()

    # Fill NaN restantes
    nan_cols_test = X_test_final.columns[X_test_final.isna().any()].tolist()
    nan_cols_train = X_train_final.columns[X_train_final.isna().any()].tolist()
    if nan_cols_test:
        print(f"  ⚠️  {len(nan_cols_test)} colunas com NaN no X_test — preenchendo com 0")
        X_test_final = X_test_final.fillna(0)
    if nan_cols_train:
        X_train_final = X_train_final.fillna(0)

    # ─── Validação rápida com o modelo ───
    print(f"\n  ═══ VALIDAÇÃO: Teste rápido com LGBM ═══")
    try:
        lgbm = joblib.load(LGBM_MODEL_PATH)
        proba = lgbm.predict_proba(X_test_final.head(100))[:, 1]
        y_sample = y_test.head(100).values
        fraud_scores = proba[y_sample == 1]
        normal_scores = proba[y_sample == 0]
        print(f"  ✅ predict_proba funcionou! (100 amostras)")
        print(f"     Normais: min={normal_scores.min():.6f}, max={normal_scores.max():.6f}")
        if len(fraud_scores) > 0:
            print(f"     Fraudes:  min={fraud_scores.min():.6f}, max={fraud_scores.max():.6f}")
        else:
            print(f"     (Nenhuma fraude nas 100 primeiras amostras)")
    except Exception as e:
        print(f"  ❌ Validação falhou: {e}")
        return None, None

    # ─── Salvar ───
    X_test_final.to_csv(X_TEST_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False, header=True)
    print(f"\n  ✅ X_test salvo: {X_TEST_PATH} ({X_test_final.shape})")
    print(f"  ✅ y_test salvo: {Y_TEST_PATH} ({y_test.shape})")

    # Salvar train também (útil para recalibrar scoring_config)
    X_train_final.to_csv(X_TRAIN_PATH, index=False)
    y_train.to_csv(Y_TRAIN_PATH, index=False, header=True)
    print(f"  ✅ X_train salvo: {X_TRAIN_PATH} ({X_train_final.shape})")
    print(f"  ✅ y_train salvo: {Y_TRAIN_PATH} ({y_train.shape})")

    return X_test_final, y_test


# =========================================================
# 6. INSPECIONAR MODELOS
# =========================================================
def inspect_models():
    print_header("5. INSPEÇÃO DOS MODELOS")

    info = {}

    # LGBM
    print("\n  --- LightGBM ---")
    lgbm = joblib.load(LGBM_MODEL_PATH)
    print(f"  Tipo: {type(lgbm).__name__}")
    for attr in ["n_estimators", "max_depth", "learning_rate", "n_features_in_"]:
        if hasattr(lgbm, attr):
            print(f"  {attr}: {getattr(lgbm, attr)}")
    info["lgbm"] = {
        "type": type(lgbm).__name__,
        "n_features": getattr(lgbm, "n_features_in_", None),
        "n_estimators": getattr(lgbm, "n_estimators", None),
    }

    # IF
    if IF_MODEL_PATH.exists():
        print("\n  --- Isolation Forest ---")
        if_model = joblib.load(IF_MODEL_PATH)
        print(f"  Tipo: {type(if_model).__name__}")
        for attr in ["n_estimators", "contamination", "n_features_in_"]:
            if hasattr(if_model, attr):
                print(f"  {attr}: {getattr(if_model, attr)}")
        info["isolation_forest"] = {
            "type": type(if_model).__name__,
            "n_features": getattr(if_model, "n_features_in_", None),
            "n_estimators": getattr(if_model, "n_estimators", None),
        }

    # IF Ref Scores
    if IF_RAW_TRAIN_PATH.exists():
        print("\n  --- IF Reference Scores ---")
        ref = np.load(IF_RAW_TRAIN_PATH)
        print(f"  Shape: {ref.shape}, min={ref.min():.4f}, max={ref.max():.4f}")

    # Métricas treino
    if METRICAS_LGBM_PATH.exists():
        print("\n  --- Métricas LGBM (treino) ---")
        with open(METRICAS_LGBM_PATH) as f:
            met = json.load(f)
        for k, v in met.items():
            if isinstance(v, (int, float, str)):
                print(f"  {k}: {v}")
        info["metricas_lgbm"] = met

    return info


# =========================================================
# 7. GERAR MANIFESTO
# =========================================================
def generate_manifest(lgbm_features, if_features, model_info):
    print_header("6. GERANDO MANIFESTO")

    manifest = {
        "gerado_em": pd.Timestamp.now().isoformat(),
        "versao": "3.0",
        "lgbm_features_count": len(lgbm_features),
        "lgbm_features": lgbm_features,
        "if_features": if_features,
        "modelos": model_info,
        "artefatos": {},
    }

    for f in sorted(ARTEFATOS_DIR.iterdir()):
        if f.is_file():
            manifest["artefatos"][f.name] = {
                "tamanho_kb": round(f.stat().st_size / 1024, 1),
            }

    with open(ARTEFATOS_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    print(f"  ✅ Manifesto salvo: {ARTEFATOS_MANIFEST_PATH}")


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n")
    print("█" * 70)
    print("█  GERAÇÃO DE ARTEFATOS v3 — Com features completas              █")
    print("█" * 70)

    # 1. Validar
    validate_existing_artifacts()

    # 2. Features LGBM
    lgbm_features = extract_lgbm_features()

    # 3. Features IF
    if_features = extract_if_features()

    # 4. Gerar X_test/y_test com 80 features
    X_test, y_test = generate_test_data(lgbm_features)

    # 5. Inspecionar modelos
    model_info = inspect_models()

    # 6. Manifesto
    generate_manifest(lgbm_features, if_features or [], model_info)

    # ─── Resumo Final ───
    print(f"\n\n{'█' * 70}")
    print("█  RESUMO FINAL")
    print(f"{'█' * 70}")

    for name, path in [
        ("lgbm_features.json", LGBM_FEATURES_PATH),
        ("if_features.json", IF_FEATURES_PATH),
        ("X_test.csv", X_TEST_PATH),
        ("y_test.csv", Y_TEST_PATH),
        ("X_train.csv", X_TRAIN_PATH),
        ("y_train.csv", Y_TRAIN_PATH),
        ("manifesto", ARTEFATOS_MANIFEST_PATH),
    ]:
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  ✅ {name:<25s} ({size_kb:,.0f} KB)")
        else:
            print(f"  ❌ {name}")

    if X_test is not None:
        print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │  ✅ X_test gerado com {X_test.shape[1]} features (match com modelo LGBM)    │
  │  ✅ {X_test.shape[0]:,} amostras, {int(y_test.sum())} fraudes                          │
  │                                                                  │
  │  PRÓXIMO PASSO:                                                  │
  │  Execute teste_pipeline_relatorio.py para ver os resultados      │
  │  com as features corretas.                                       │
  └──────────────────────────────────────────────────────────────────┘
""")
    else:
        print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │  ❌ Não foi possível gerar X_test com features completas        │
  │                                                                  │
  │  Verifique:                                                      │
  │  1. base_mvp_model_ready.csv existe em dados/                    │
  │  2. Tem coluna target (is_fraud, is_fraude, etc.)                │
  │  3. Colunas brutas para derivar features (ds_sexo, etc.)         │
  └──────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
