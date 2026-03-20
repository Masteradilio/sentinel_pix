"""
gerar_artefatos_faltantes.py — Gera artefatos necessários para o pipeline
===========================================================================

Versão 2: Corrige caminhos e importa PixPreprocessor.

Gera:
  1. lgbm_features.json — Lista ordenada de features do LGBM
  2. if_features.json — Lista de features do Isolation Forest
  3. X_test.csv / y_test.csv — Dados de teste (mesmo split do treino)
  4. Validação completa + inspeção dos artefatos

Autor: Equipe Anomalia PIX
Data: Março 2026
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

# Raiz do projeto = onde este script está OU um nível acima de backend/
SCRIPT_DIR = Path(__file__).resolve().parent

# Detectar raiz automaticamente
if (SCRIPT_DIR / "backend").exists() and (SCRIPT_DIR / "dados").exists():
    PROJECT_ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "backend").exists():
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

ARTEFATOS_DIR = PROJECT_ROOT / "backend" / "artefatos"
DADOS_DIR = PROJECT_ROOT / "dados"               # ← CORRIGIDO: raiz/dados
MODELOS_DIR = PROJECT_ROOT / "backend" / "modelos"

print(f"  Project root: {PROJECT_ROOT}")
print(f"  Artefatos: {ARTEFATOS_DIR}")
print(f"  Dados: {DADOS_DIR}")
print(f"  Modelos: {MODELOS_DIR}")

# =========================================================
# IMPORTAR PixPreprocessor (necessário para carregar o joblib)
# =========================================================

# Adicionar modelos/ ao path para importar preprocessing.py
sys.path.insert(0, str(MODELOS_DIR))

try:
    from preprocessing import PixPreprocessor
    print(f"\n  ✅ PixPreprocessor importado de {MODELOS_DIR / 'preprocessing.py'}")
except ImportError as e:
    print(f"\n  ⚠️  Não conseguiu importar PixPreprocessor: {e}")
    print(f"     Tentando importar de backend/modelos/...")
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "backend" / "modelos"))
        from preprocessing import PixPreprocessor
        print(f"  ✅ PixPreprocessor importado (tentativa 2)")
    except ImportError:
        print(f"  ❌ PixPreprocessor não encontrado. O preprocessor não poderá ser carregado.")
        PixPreprocessor = None

# Artefatos existentes
LGBM_MODEL_PATH = ARTEFATOS_DIR / "model_lightgbm.joblib"
LGBM_CALIBRATED_PATH = ARTEFATOS_DIR / "model_lightgbm_calibrated.joblib"
IF_MODEL_PATH = ARTEFATOS_DIR / "model_isolation_forest.joblib"
PREPROCESSOR_PATH = ARTEFATOS_DIR / "preprocessing.joblib"
IF_SCALER_PATH = ARTEFATOS_DIR / "scaler_isolation_forest.joblib"
IF_CONFIG_PATH = ARTEFATOS_DIR / "isolation_forest_config.json"
IF_RAW_TRAIN_PATH = ARTEFATOS_DIR / "if_ref_raw_train.npy"
METRICAS_LGBM_PATH = ARTEFATOS_DIR / "metricas_lightgbm.json"
METRICAS_IF_PATH = ARTEFATOS_DIR / "metricas_isolation_forest.json"
FEATURE_IMP_LGBM_PATH = ARTEFATOS_DIR / "feature_importance_lightgbm.csv"
PREDICOES_LGBM_PATH = ARTEFATOS_DIR / "predicoes_teste_lightgbm.csv"
PREDICOES_IF_PATH = ARTEFATOS_DIR / "predicoes_teste_isolation_forest.csv"

# Dados
BASE_MVP_PATH = DADOS_DIR / "base_mvp_model_ready.csv"

# Artefatos a gerar
LGBM_FEATURES_PATH = ARTEFATOS_DIR / "lgbm_features.json"
IF_FEATURES_PATH = ARTEFATOS_DIR / "if_features.json"
X_TEST_PATH = ARTEFATOS_DIR / "X_test.csv"
Y_TEST_PATH = ARTEFATOS_DIR / "y_test.csv"
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


# =========================================================
# 1. VALIDAR ARTEFATOS EXISTENTES
# =========================================================

def validate_existing_artifacts():
    print_header("1. VALIDAÇÃO DOS ARTEFATOS EXISTENTES")

    print("\n  Modelos:")
    check_file(LGBM_MODEL_PATH, required=True)
    check_file(LGBM_CALIBRATED_PATH, required=False)
    check_file(IF_MODEL_PATH, required=False)
    check_file(PREPROCESSOR_PATH, required=True)
    check_file(IF_SCALER_PATH, required=False)

    print("\n  Configs:")
    check_file(IF_CONFIG_PATH, required=False)
    check_file(IF_RAW_TRAIN_PATH, required=False)
    check_file(METRICAS_LGBM_PATH, required=False)
    check_file(METRICAS_IF_PATH, required=False)

    print("\n  Análises:")
    check_file(FEATURE_IMP_LGBM_PATH, required=False)
    check_file(PREDICOES_LGBM_PATH, required=False)
    check_file(PREDICOES_IF_PATH, required=False)

    print("\n  Dados:")
    check_file(BASE_MVP_PATH, required=True)

    if not LGBM_MODEL_PATH.exists():
        print("\n  ❌ ERRO CRÍTICO: model_lightgbm.joblib não encontrado!")
        sys.exit(1)


# =========================================================
# 2. EXTRAIR FEATURES DO LGBM
# =========================================================

def generate_lgbm_features():
    print_header("2. EXTRAINDO FEATURES DO LGBM")

    lgbm_model = joblib.load(LGBM_MODEL_PATH)
    features = None

    # Tentar múltiplos métodos
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
        print("  ❌ Não foi possível extrair features!")
        sys.exit(1)

    # Classificar features por tipo
    rule_features = [f for f in features if f.startswith("rule_")]
    flag_features = [f for f in features if f.endswith("_flag")]
    value_features = [f for f in features if f.startswith("vl_") or f.startswith("qt_") or f.startswith("nr_")]
    other_features = [f for f in features if f not in rule_features + flag_features + value_features]

    print(f"\n  Classificação das {len(features)} features:")
    print(f"    Rule features (regras como input): {len(rule_features)}")
    for f in rule_features:
        print(f"      → {f}")
    print(f"    Flag features: {len(flag_features)}")
    print(f"    Value/Quant features: {len(value_features)}")
    print(f"    Outras: {len(other_features)}")

    with open(LGBM_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ Salvo: {LGBM_FEATURES_PATH}")
    return features


# =========================================================
# 3. EXTRAIR FEATURES DO IF
# =========================================================

def generate_if_features():
    print_header("3. EXTRAINDO FEATURES DO ISOLATION FOREST")

    if IF_CONFIG_PATH.exists():
        with open(IF_CONFIG_PATH, "r") as f:
            config = json.load(f)

        print(f"  Config carregada. Chaves: {list(config.keys())}")

        if "features" in config:
            features = config["features"]
            print(f"  Features do IF: {len(features)}")
            for f in features:
                print(f"    → {f}")

            with open(IF_FEATURES_PATH, "w", encoding="utf-8") as f_out:
                json.dump(features, f_out, ensure_ascii=False, indent=2)
            print(f"\n  ✅ Salvo: {IF_FEATURES_PATH}")

            # Mostrar demais configs
            for key in config:
                if key != "features":
                    val = config[key]
                    if isinstance(val, (str, int, float, bool)):
                        print(f"  {key}: {val}")
                    elif isinstance(val, dict):
                        print(f"  {key}: dict com {len(val)} chaves")
                    elif isinstance(val, list) and len(val) <= 20:
                        print(f"  {key}: list com {len(val)} elementos")
                    else:
                        print(f"  {key}: {type(val).__name__}")

            return features

    print("  ⚠️  Config não encontrada, tentando do modelo...")
    return None


# =========================================================
# 4. GERAR X_TEST / Y_TEST
# =========================================================

def generate_test_split(lgbm_features: list):
    print_header("4. GERANDO DADOS DE TESTE (X_test / y_test)")

    if not BASE_MVP_PATH.exists():
        print(f"  ❌ Base não encontrada: {BASE_MVP_PATH}")
        print(f"  Tentando caminhos alternativos...")

        # Tentar encontrar em outros lugares
        alternatives = [
            PROJECT_ROOT / "dados" / "base_mvp_model_ready.csv",
            PROJECT_ROOT / "backend" / "dados" / "base_mvp_model_ready.csv",
            ARTEFATOS_DIR / "base_mvp_model_ready.csv",
        ]
        found = None
        for alt in alternatives:
            if alt.exists():
                found = alt
                break

        if found is None:
            print(f"  ❌ Nenhum caminho alternativo encontrado!")
            print(f"  Tentando gerar X_test/y_test das predições salvas...")
            return generate_test_from_predictions(lgbm_features)

        base_path = found
    else:
        base_path = BASE_MVP_PATH

    print(f"  ✅ Base encontrada: {base_path}")
    df = pd.read_csv(base_path)
    print(f"  Shape: {df.shape}")
    print(f"  Colunas ({len(df.columns)}): {list(df.columns)}")

    # Identificar target
    target_col = None
    for col in ["is_fraud", "is_fraude", "fraud", "target", "label", "tp_fraude"]:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        print(f"  ❌ Coluna target não encontrada!")
        return generate_test_from_predictions(lgbm_features)

    print(f"  Target: '{target_col}'")
    print(f"  Distribuição:\n{df[target_col].value_counts()}")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n  X_test: {X_test.shape}")
    print(f"  y_test: fraudes={y_test.sum()} ({y_test.mean()*100:.2f}%)")

    # Verificar match com LGBM features
    if lgbm_features:
        missing = [f for f in lgbm_features if f not in X_test.columns]
        if missing:
            print(f"\n  ⚠️  Features do LGBM faltando no X_test ({len(missing)}):")
            for f in missing:
                print(f"    FALTA: {f}")

    X_test.to_csv(X_TEST_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False, header=True)

    print(f"\n  ✅ X_test salvo: {X_TEST_PATH}")
    print(f"  ✅ y_test salvo: {Y_TEST_PATH}")
    return X_test, y_test


def generate_test_from_predictions(lgbm_features):
    """Fallback: gerar X_test/y_test a partir das predições já salvas."""
    print(f"\n  Tentando gerar de predicoes_teste_lightgbm.csv...")

    if not PREDICOES_LGBM_PATH.exists():
        print(f"  ❌ predicoes_teste_lightgbm.csv não encontrado!")
        return None

    df_pred = pd.read_csv(PREDICOES_LGBM_PATH)
    print(f"  Predições LGBM: {df_pred.shape}")
    print(f"  Colunas: {list(df_pred.columns)}")

    # Identificar coluna de target real
    target_candidates = ["y_true", "y_real", "is_fraud", "is_fraude", "label", "target", "tp_fraude"]
    y_col = None
    for col in target_candidates:
        if col in df_pred.columns:
            y_col = col
            break

    if y_col is None:
        print(f"  ❌ Coluna target não encontrada nas predições!")
        return None

    print(f"  Target: '{y_col}'")
    y_test = df_pred[y_col]

    # Se o arquivo de predições contiver as features também
    feature_cols = [c for c in df_pred.columns if c in (lgbm_features or [])]
    if feature_cols:
        X_test = df_pred[feature_cols]
        print(f"  Features encontradas: {len(feature_cols)}")
    else:
        print(f"  ⚠️  Features não encontradas nas predições")
        print(f"  Salvando apenas y_test")
        X_test = None

    # Salvar
    y_test.to_csv(Y_TEST_PATH, index=False, header=True)
    print(f"  ✅ y_test salvo: {Y_TEST_PATH} ({y_test.shape})")

    if X_test is not None:
        X_test.to_csv(X_TEST_PATH, index=False)
        print(f"  ✅ X_test salvo: {X_TEST_PATH} ({X_test.shape})")

    return X_test, y_test


# =========================================================
# 5. INSPECIONAR PREPROCESSOR
# =========================================================

def inspect_preprocessor():
    print_header("5. INSPEÇÃO DO PREPROCESSOR (PixPreprocessor)")

    if PixPreprocessor is None:
        print("  ❌ PixPreprocessor não importado. Não é possível carregar o joblib.")
        print("  Inspecionando o arquivo preprocessing.py diretamente...")
        inspect_preprocessing_source()
        return None

    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"  ✅ Preprocessor carregado!")
        print(f"  Tipo: {type(preprocessor).__name__}")
    except Exception as e:
        print(f"  ❌ Erro ao carregar: {e}")
        inspect_preprocessing_source()
        return None

    # Inspecionar atributos
    print(f"\n  Atributos do PixPreprocessor:")
    for attr in sorted(dir(preprocessor)):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(preprocessor, attr)
            if callable(val):
                print(f"    📎 {attr}() — método")
            elif isinstance(val, (str, int, float, bool)):
                print(f"    📋 {attr} = {val}")
            elif isinstance(val, (list, tuple)):
                print(f"    📋 {attr} = list({len(val)} items) → {val[:3]}...")
            elif isinstance(val, dict):
                print(f"    📋 {attr} = dict({len(val)} keys) → {list(val.keys())[:5]}...")
            elif isinstance(val, np.ndarray):
                print(f"    📋 {attr} = ndarray{val.shape}")
            elif isinstance(val, pd.DataFrame):
                print(f"    📋 {attr} = DataFrame{val.shape}")
            else:
                print(f"    📋 {attr} = {type(val).__name__}")
        except Exception as e:
            print(f"    ⚠️  {attr}: erro ao acessar ({e})")

    # Verificar métodos importantes
    print(f"\n  Métodos disponíveis:")
    for method in ["fit", "transform", "fit_transform", "get_feature_names",
                    "get_feature_names_out", "preprocess", "process"]:
        has = hasattr(preprocessor, method) and callable(getattr(preprocessor, method))
        print(f"    {'✅' if has else '❌'} {method}()")

    # Testar com dados reais
    if BASE_MVP_PATH.exists():
        print(f"\n  Teste de transformação com dados reais:")
        df = pd.read_csv(BASE_MVP_PATH, nrows=5)
        print(f"    Input: {df.shape}, colunas: {list(df.columns[:8])}...")

        # Tentar diferentes métodos
        for method_name in ["transform", "preprocess", "fit_transform"]:
            if hasattr(preprocessor, method_name):
                try:
                    func = getattr(preprocessor, method_name)
                    result = func(df)
                    if isinstance(result, pd.DataFrame):
                        print(f"    ✅ {method_name}() → DataFrame{result.shape}")
                        print(f"       Colunas saída: {list(result.columns)}")
                    elif isinstance(result, np.ndarray):
                        print(f"    ✅ {method_name}() → ndarray{result.shape}")
                    elif isinstance(result, tuple):
                        print(f"    ✅ {method_name}() → tuple de {len(result)} elementos")
                        for i, r in enumerate(result):
                            if isinstance(r, pd.DataFrame):
                                print(f"       [{i}]: DataFrame{r.shape} → {list(r.columns[:5])}...")
                            elif isinstance(r, pd.Series):
                                print(f"       [{i}]: Series{r.shape} → {r.name}")
                            elif isinstance(r, np.ndarray):
                                print(f"       [{i}]: ndarray{r.shape}")
                            else:
                                print(f"       [{i}]: {type(r).__name__}")
                    else:
                        print(f"    ✅ {method_name}() → {type(result).__name__}")
                    break
                except Exception as e:
                    print(f"    ⚠️  {method_name}() falhou: {e}")

    return preprocessor


def inspect_preprocessing_source():
    """Lê o código-fonte do preprocessing.py para entender a classe."""
    preprocessing_path = MODELOS_DIR / "preprocessing.py"
    if not preprocessing_path.exists():
        print(f"  ❌ {preprocessing_path} não encontrado")
        return

    print(f"\n  Lendo {preprocessing_path}...")
    with open(preprocessing_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Extrair informações-chave
    lines = source.split("\n")

    print(f"  Total de linhas: {len(lines)}")

    # Encontrar definição da classe
    for i, line in enumerate(lines):
        if "class PixPreprocessor" in line:
            print(f"\n  Classe encontrada na linha {i+1}:")
            # Mostrar as próximas 50 linhas (ou até a próxima classe/def de topo)
            for j in range(i, min(i + 80, len(lines))):
                print(f"    {j+1:4d} | {lines[j]}")
            break

    # Encontrar __init__
    for i, line in enumerate(lines):
        if "def __init__" in line and i > 0:
            print(f"\n  __init__ na linha {i+1}:")
            for j in range(i, min(i + 30, len(lines))):
                if lines[j].strip() and not lines[j].strip().startswith("#"):
                    print(f"    {j+1:4d} | {lines[j]}")
                if j > i and lines[j].strip().startswith("def ") and "init" not in lines[j]:
                    break

    # Encontrar transform/preprocess
    for i, line in enumerate(lines):
        if ("def transform" in line or "def preprocess" in line) and "self" in line:
            print(f"\n  Método na linha {i+1}:")
            for j in range(i, min(i + 50, len(lines))):
                print(f"    {j+1:4d} | {lines[j]}")
                if j > i + 2 and lines[j].strip().startswith("def ") and "transform" not in lines[j] and "preprocess" not in lines[j]:
                    break


# =========================================================
# 6. INSPECIONAR MODELOS
# =========================================================

def inspect_models(lgbm_features):
    print_header("6. INSPEÇÃO DOS MODELOS")

    info = {}

    # LGBM
    print("\n  --- LightGBM ---")
    lgbm = joblib.load(LGBM_MODEL_PATH)
    print(f"  Tipo: {type(lgbm).__name__}")
    for attr in ["n_estimators", "max_depth", "learning_rate", "n_features_in_",
                  "classes_", "objective", "boosting_type"]:
        if hasattr(lgbm, attr):
            print(f"  {attr}: {getattr(lgbm, attr)}")

    info["lgbm"] = {
        "type": type(lgbm).__name__,
        "n_features": getattr(lgbm, "n_features_in_", None),
        "n_estimators": getattr(lgbm, "n_estimators", None),
    }

    # LGBM Calibrado
    if LGBM_CALIBRATED_PATH.exists():
        print("\n  --- LightGBM Calibrado ---")
        lgbm_cal = joblib.load(LGBM_CALIBRATED_PATH)
        print(f"  Tipo: {type(lgbm_cal).__name__}")
        if hasattr(lgbm_cal, "calibrated_classifiers_"):
            print(f"  N calibrators: {len(lgbm_cal.calibrated_classifiers_)}")
        if hasattr(lgbm_cal, "method"):
            print(f"  Método calibração: {lgbm_cal.method}")
        info["lgbm_calibrated"] = {"type": type(lgbm_cal).__name__}

    # IF
    if IF_MODEL_PATH.exists():
        print("\n  --- Isolation Forest ---")
        if_model = joblib.load(IF_MODEL_PATH)
        print(f"  Tipo: {type(if_model).__name__}")
        for attr in ["n_estimators", "contamination", "n_features_in_",
                      "max_samples", "max_features"]:
            if hasattr(if_model, attr):
                print(f"  {attr}: {getattr(if_model, attr)}")

        info["isolation_forest"] = {
            "type": type(if_model).__name__,
            "n_features": getattr(if_model, "n_features_in_", None),
            "n_estimators": getattr(if_model, "n_estimators", None),
            "contamination": str(getattr(if_model, "contamination", None)),
        }

    # IF Scaler
    if IF_SCALER_PATH.exists():
        print("\n  --- IF Scaler ---")
        scaler = joblib.load(IF_SCALER_PATH)
        print(f"  Tipo: {type(scaler).__name__}")
        if hasattr(scaler, "n_features_in_"):
            print(f"  n_features_in_: {scaler.n_features_in_}")
        if hasattr(scaler, "feature_names_in_"):
            print(f"  features: {list(scaler.feature_names_in_)}")

    # IF Raw Train Scores
    if IF_RAW_TRAIN_PATH.exists():
        print("\n  --- IF Reference Scores ---")
        ref_scores = np.load(IF_RAW_TRAIN_PATH)
        print(f"  Shape: {ref_scores.shape}")
        print(f"  Min: {ref_scores.min():.6f}")
        print(f"  Max: {ref_scores.max():.6f}")
        print(f"  Mean: {ref_scores.mean():.6f}")
        print(f"  Std: {ref_scores.std():.6f}")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            print(f"  P{p}: {np.percentile(ref_scores, p):.6f}")

    # Métricas
    if METRICAS_LGBM_PATH.exists():
        print("\n  --- Métricas do LGBM (salvas no treino) ---")
        with open(METRICAS_LGBM_PATH, "r") as f:
            metricas = json.load(f)
        for k, v in metricas.items():
            if isinstance(v, (int, float, str)):
                print(f"  {k}: {v}")
            elif isinstance(v, dict):
                print(f"  {k}:")
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float, str)):
                        print(f"    {k2}: {v2}")
        info["metricas_lgbm"] = metricas

    if METRICAS_IF_PATH.exists():
        print("\n  --- Métricas do IF (salvas no treino) ---")
        with open(METRICAS_IF_PATH, "r") as f:
            metricas_if = json.load(f)
        for k, v in metricas_if.items():
            if isinstance(v, (int, float, str)):
                print(f"  {k}: {v}")
        info["metricas_if"] = metricas_if

    return info


# =========================================================
# 7. INSPECIONAR PREDIÇÕES EXISTENTES
# =========================================================

def inspect_existing_predictions():
    print_header("7. INSPEÇÃO DAS PREDIÇÕES EXISTENTES")

    for name, path in [
        ("LGBM", PREDICOES_LGBM_PATH),
        ("IF", PREDICOES_IF_PATH),
    ]:
        if path.exists():
            print(f"\n  --- Predições {name} ---")
            df = pd.read_csv(path, nrows=5)
            print(f"  Shape total: {pd.read_csv(path).shape}")
            print(f"  Colunas: {list(df.columns)}")
            print(f"  Primeiras linhas:")
            print(df.to_string(index=False, max_cols=10))
        else:
            print(f"\n  ⚠️  {name}: {path.name} não encontrado")


# =========================================================
# 8. GERAR MANIFESTO
# =========================================================

def generate_manifest(lgbm_features, if_features, model_info):
    print_header("8. GERANDO MANIFESTO DOS ARTEFATOS")

    manifest = {
        "gerado_em": pd.Timestamp.now().isoformat(),
        "projeto": "Anomalia PIX",
        "versao": "1.0",
        "artefatos": {},
        "modelos": model_info,
        "lgbm_features": lgbm_features,
        "if_features": if_features,
    }

    for f in sorted(ARTEFATOS_DIR.iterdir()):
        if f.is_file():
            manifest["artefatos"][f.name] = {
                "tamanho_kb": round(f.stat().st_size / 1024, 1),
                "modificado": pd.Timestamp(f.stat().st_mtime, unit="s").isoformat(),
            }

    with open(ARTEFATOS_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    print(f"  ✅ Manifesto salvo: {ARTEFATOS_MANIFEST_PATH}")
    print(f"  Total de artefatos: {len(manifest['artefatos'])}")


# =========================================================
# 9. INSPEÇÃO DO PREPROCESSING.PY (código-fonte)
# =========================================================

def inspect_preprocessing_code():
    """Mostra o código completo do preprocessing.py."""
    print_header("9. CÓDIGO-FONTE DO PREPROCESSING.PY")

    preprocessing_path = MODELOS_DIR / "preprocessing.py"
    if not preprocessing_path.exists():
        print(f"  ❌ {preprocessing_path} não encontrado")
        return

    with open(preprocessing_path, "r", encoding="utf-8") as f:
        source = f.read()

    print(f"  Arquivo: {preprocessing_path}")
    print(f"  Tamanho: {len(source)} chars, {len(source.splitlines())} linhas")
    print(f"\n  {'─' * 60}")
    print(source)
    print(f"  {'─' * 60}")


# =========================================================
# MAIN
# =========================================================

def main():
    print("\n")
    print("█" * 70)
    print("  GERAÇÃO DE ARTEFATOS FALTANTES — v2")
    print("  Sistema Anomalia PIX")
    print("█" * 70)

    # 1. Validar existentes
    validate_existing_artifacts()

    # 2. Extrair features do LGBM
    lgbm_features = generate_lgbm_features()

    # 3. Extrair features do IF
    if_features = generate_if_features()

    # 4. Gerar X_test / y_test
    generate_test_split(lgbm_features)

    # 5. Inspecionar preprocessor
    inspect_preprocessor()

    # 6. Inspecionar modelos
    model_info = inspect_models(lgbm_features)

    # 7. Inspecionar predições existentes
    inspect_existing_predictions()

    # 8. Gerar manifesto
    generate_manifest(lgbm_features, if_features, model_info)

    # 9. Código-fonte do preprocessing
    inspect_preprocessing_code()

    # Resumo final
    print(f"\n\n{'█' * 70}")
    print("  RESUMO FINAL")
    print(f"{'█' * 70}")

    for path in [LGBM_FEATURES_PATH, IF_FEATURES_PATH, X_TEST_PATH,
                 Y_TEST_PATH, ARTEFATOS_MANIFEST_PATH]:
        if path.exists():
            print(f"  ✅ {path.name}")
        else:
            print(f"  ❌ {path.name}")

    print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │  AÇÃO NECESSÁRIA:                                                │
  │                                                                  │
  │  Cole o output completo deste script na conversa.                │
  │  Com base nas informações extraídas, vou:                        │
  │                                                                  │
  │  1. Reescrever pipeline_inferencia.py com:                       │
  │     - PixPreprocessor integrado                                  │
  │     - Caminhos corretos dos artefatos                            │
  │     - Rule scores como INPUT do LGBM (não ensemble separado)     │
  │     - IF scaler + reference scores                               │
  │                                                                  │
  │  2. Reescrever teste_pipeline_relatorio.py com:                  │
  │     - Caminhos corretos                                          │
  │     - Preprocessor integrado                                     │
  │     - Relatório executivo preciso                                │
  └──────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    main()
