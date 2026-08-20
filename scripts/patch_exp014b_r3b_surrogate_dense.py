#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Patch EXP-014B-R3B — Dense Surrogate Preprocessor

Corrige o erro:

  ValueError: scipy.sparse does not support dtype object

Causa provável:
  No treinamento do surrogate, o ColumnTransformer estava combinando
  colunas numéricas em passthrough + OneHotEncoder sparse. Em algumas versões
  de sklearn/scipy, isso pode gerar matriz sparse com dtype object quando há
  mistura de DataFrame/categorias string/números.

Correção:
  1. Força as colunas numéricas para numpy.float64 via FunctionTransformer.
  2. Força OneHotEncoder a emitir float64 denso.
  3. Força ColumnTransformer a retornar matriz densa com sparse_threshold=0.0.
  4. Adiciona try/except no fit do surrogate para registrar falhas e continuar,
     sem derrubar o experimento.

Uso:
  python scripts\\patch_exp014b_r3b_surrogate_dense.py

Depois:
  python scripts\\exp_014b_r3b_base_reconstruction_audit.py
"""

from pathlib import Path

SCRIPT = Path("scripts/exp_014b_r3b_base_reconstruction_audit.py")

if not SCRIPT.exists():
    raise FileNotFoundError(f"Não encontrei {SCRIPT}. Execute na raiz do projeto rebuild_pix.")

text = SCRIPT.read_text(encoding="utf-8")

# 1) Import FunctionTransformer.
old_import = "from sklearn.preprocessing import OneHotEncoder\\n"
new_import = "from sklearn.preprocessing import OneHotEncoder, FunctionTransformer\\n"
if old_import in text:
    text = text.replace(old_import, new_import)
elif new_import not in text:
    raise RuntimeError("Não encontrei import de OneHotEncoder para substituir.")

# 2) Inserir helpers antes do dataclass.
if "def _to_float_array_for_surrogate(" not in text:
    marker = "\\n\\n@dataclass\\nclass RecipeResult:"
    if marker not in text:
        raise RuntimeError("Não encontrei ponto de inserção antes de RecipeResult.")
    helper = """

def _to_float_array_for_surrogate(X):
    \\\"\\\"\\\"Converte bloco numérico do ColumnTransformer para float64 denso.\\\"\\\"\\\"
    return np.asarray(X, dtype=np.float64)


def _make_dense_ohe():
    \\\"\\\"\\\"OneHotEncoder compatível com versões novas e antigas do sklearn.\\\"\\\"\\\"
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=2,
            sparse_output=False,
            dtype=np.float64,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=2,
            sparse=False,
            dtype=np.float64,
        )
"""
    text = text.replace(marker, helper + marker)

# 3) Trocar transformers do surrogate.
old_block = """    transformers = []
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), cat_cols))
    if not transformers:
        return [], None, None, pd.DataFrame()
"""

new_block = """    transformers = []
    if num_cols:
        transformers.append((
            "num",
            FunctionTransformer(_to_float_array_for_surrogate, validate=False),
            num_cols,
        ))
    if cat_cols:
        transformers.append(("cat", _make_dense_ohe(), cat_cols))
    if not transformers:
        return [], None, None, pd.DataFrame()
"""

if old_block in text:
    text = text.replace(old_block, new_block)
else:
    if "_make_dense_ohe()" not in text:
        raise RuntimeError("Não encontrei bloco de transformers do surrogate para substituir.")

# 4) Trocar ColumnTransformer para dense.
old_ct = """                ("prep", ColumnTransformer(transformers=transformers, remainder="drop")),
"""
new_ct = """                ("prep", ColumnTransformer(
                    transformers=transformers,
                    remainder="drop",
                    sparse_threshold=0.0,
                )),
"""
if old_ct in text:
    text = text.replace(old_ct, new_ct)
else:
    if "sparse_threshold=0.0" not in text:
        raise RuntimeError("Não encontrei ColumnTransformer do surrogate para substituir.")

# 5) Adicionar try/except no clf.fit para não derrubar tudo.
old_fit = """            clf.fit(X_train, y_train)
            pred_train = clf.predict(X_train)
            pred_test = clf.predict(X_test)
            pred_all = clf.predict(X)
"""

new_fit = """            try:
                clf.fit(X_train, y_train)
                pred_train = clf.predict(X_train)
                pred_test = clf.predict(X_test)
                pred_all = clf.predict(X)
            except Exception as exc:
                rows.append({
                    "max_depth": depth,
                    "min_samples_leaf": min_leaf,
                    "match_rate_all": None,
                    "match_rate_test": None,
                    "target_positive_rate": float(np.mean(y)),
                    "pred_positive_rate_all": None,
                    "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                })
                continue
"""

if old_fit in text:
    text = text.replace(old_fit, new_fit)
else:
    start = text.find("for depth in range(2, max_depth + 1):")
    end = text.find("fidelity_df = pd.DataFrame(rows)", start)
    if start == -1 or end == -1 or "except Exception as exc:" not in text[start:end]:
        raise RuntimeError("Não encontrei bloco clf.fit do surrogate para proteger com try/except.")

SCRIPT.write_text(text, encoding="utf-8", newline="\\n")
print(f"Patch aplicado com sucesso em {SCRIPT}")
