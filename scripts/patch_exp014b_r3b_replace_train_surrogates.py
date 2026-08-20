#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Patch EXP-014B-R3B — Replace train_surrogates with dense robust version

Corrige dois problemas:

1) O patch anterior falhou procurando uma linha exata de import:
     RuntimeError: Não encontrei import de OneHotEncoder para substituir.

2) O EXP-014B-R3B continuou quebrando no surrogate:
     ValueError: scipy.sparse does not support dtype object

Este patch é mais robusto porque NÃO depende do import existente. Ele substitui
a função inteira train_surrogates(...) por uma versão que:
  - importa OneHotEncoder/FunctionTransformer localmente;
  - converte numéricos para float64;
  - força OneHotEncoder denso float64;
  - força ColumnTransformer denso com sparse_threshold=0.0;
  - registra erros por configuração de árvore e continua;
  - se todos os surrogates falharem, o experimento termina controladamente
    em vez de quebrar o terminal.

Uso:
  python scripts\\patch_exp014b_r3b_replace_train_surrogates.py

Depois:
  python scripts\\exp_014b_r3b_base_reconstruction_audit.py

Atalho sem surrogate, caso você queira apenas inventário/threshold/expressões:
  python scripts\\exp_014b_r3b_base_reconstruction_audit.py --require-exact-or-threshold
"""

from pathlib import Path

SCRIPT = Path("scripts/exp_014b_r3b_base_reconstruction_audit.py")

if not SCRIPT.exists():
    raise FileNotFoundError(f"Não encontrei {SCRIPT}. Execute na raiz do projeto rebuild_pix.")

text = SCRIPT.read_text(encoding="utf-8")

start = text.find("\ndef train_surrogates(")
if start == -1:
    start = text.find("def train_surrogates(")

end = text.find("\ndef apply_recipe(", start)
if start == -1 or end == -1:
    raise RuntimeError("Não consegui localizar o bloco def train_surrogates(...) até def apply_recipe(...).")

new_func = """
def train_surrogates(
    small: pd.DataFrame,
    expanded: pd.DataFrame,
    target: np.ndarray,
    min_fidelity: float,
    max_depth: int,
    min_leaf_values: list[int],
    seed: int,
    output_dir: Path,
) -> tuple[list[RecipeResult], Any | None, dict[str, Any] | None, pd.DataFrame]:
    \"""
    Versão robusta/densa do surrogate.

    Evita o erro:
      scipy.sparse does not support dtype object

    Estratégia:
      - numéricos -> float64 denso;
      - categóricos -> OneHotEncoder denso float64;
      - ColumnTransformer -> saída densa;
      - cada árvore falha isoladamente, sem derrubar o experimento.
    \"""
    from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

    def _to_float_array_for_surrogate(X):
        return np.asarray(X, dtype=np.float64)

    def _make_dense_ohe():
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

    num_cols = [c for c in NUMERIC_SURROGATE_COLS if c in small.columns and c in expanded.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in small.columns and c in expanded.columns]

    X = prep_surrogate_X(small, num_cols, cat_cols)
    y = target.astype(int)

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            FunctionTransformer(_to_float_array_for_surrogate, validate=False),
            num_cols,
        ))
    if cat_cols:
        transformers.append(("cat", _make_dense_ohe(), cat_cols))

    if not transformers:
        empty = pd.DataFrame([{
            "status": "NO_COMMON_FEATURES_FOR_SURROGATE",
            "error": "Sem colunas comuns entre small e expanded para surrogate.",
        }])
        empty.to_csv(output_dir / "04_surrogate_fidelity.csv", index=False)
        return [], None, None, empty

    results = []
    best_model = None
    best_meta = None
    rows = []

    try:
        stratify = y if len(np.unique(y)) == 2 and min(np.bincount(y)) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=seed,
            stratify=stratify,
        )
    except Exception as exc:
        rows.append({
            "status": "TRAIN_TEST_SPLIT_FAILED",
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            "num_cols": "|".join(num_cols),
            "cat_cols": "|".join(cat_cols),
        })
        fidelity_df = pd.DataFrame(rows)
        fidelity_df.to_csv(output_dir / "04_surrogate_fidelity.csv", index=False)
        return [], None, None, fidelity_df

    for depth in range(2, max_depth + 1):
        for min_leaf in min_leaf_values:
            clf = Pipeline([
                ("prep", ColumnTransformer(
                    transformers=transformers,
                    remainder="drop",
                    sparse_threshold=0.0,
                )),
                ("tree", DecisionTreeClassifier(
                    max_depth=depth,
                    min_samples_leaf=min_leaf,
                    random_state=seed,
                )),
            ])

            try:
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
                    "match_rate_train": None,
                    "target_positive_rate": float(np.mean(y)),
                    "pred_positive_rate_all": None,
                    "num_cols": "|".join(num_cols),
                    "cat_cols": "|".join(cat_cols),
                    "status": "FIT_FAILED",
                    "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                })
                continue

            match_train = float((pred_train == y_train).mean())
            match_test = float((pred_test == y_test).mean())
            match_all = float((pred_all == y).mean())
            pos_all = float(np.mean(pred_all))
            target_pos = float(np.mean(y))

            meta = {
                "type": "decision_tree_surrogate",
                "num_cols": num_cols,
                "cat_cols": cat_cols,
                "max_depth": depth,
                "min_samples_leaf": min_leaf,
                "match_rate_all": match_all,
                "match_rate_test": match_test,
                "target_positive_rate": target_pos,
                "pred_positive_rate_all": pos_all,
                "surrogate_preprocessor": "dense_float64_column_transformer",
            }

            rows.append({
                **meta,
                "match_rate_train": match_train,
                "status": "OK",
                "error": None,
            })

            rr = RecipeResult(
                "decision_tree_surrogate",
                f"tree_depth{depth}_leaf{min_leaf}",
                meta,
                match_all,
                match_test,
                target_pos,
                pos_all,
                (match_all >= min_fidelity and match_test >= min_fidelity),
                "surrogate_tree" if (match_all >= min_fidelity and match_test >= min_fidelity) else "below_min_fidelity",
            )
            results.append(rr)

            current_rank = (
                match_test,
                match_all,
                -abs(pos_all - target_pos),
                -depth,
                -min_leaf,
            )
            if best_meta is None or current_rank > best_meta["rank"]:
                best_model = clf
                best_meta = {"rank": current_rank, "recipe": meta, "name": rr.name}

    fidelity_df = pd.DataFrame(rows)
    if not fidelity_df.empty:
        sort_cols = [c for c in ["match_rate_test", "match_rate_all"] if c in fidelity_df.columns]
        if sort_cols:
            fidelity_df = fidelity_df.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last").reset_index(drop=True)
    fidelity_df.to_csv(output_dir / "04_surrogate_fidelity.csv", index=False)

    if best_model is not None:
        try:
            import joblib
            joblib.dump(best_model, output_dir / "base_reconstruction_surrogate.joblib")
            tree = best_model.named_steps["tree"]
            prep = best_model.named_steps["prep"]
            try:
                feature_names = prep.get_feature_names_out()
                tree_txt = export_text(tree, feature_names=[str(x) for x in feature_names])
            except Exception:
                tree_txt = export_text(tree)
            (output_dir / "base_reconstruction_surrogate_tree.txt").write_text(tree_txt, encoding="utf-8")
        except Exception as exc:
            (output_dir / "surrogate_save_warning.txt").write_text(
                f"{type(exc).__name__}: {str(exc)}",
                encoding="utf-8",
            )

    best_recipe = best_meta["recipe"] if best_meta else None
    return results, best_model, best_recipe, fidelity_df

"""

text = text[:start] + "\n" + new_func + text[end:]

SCRIPT.write_text(text, encoding="utf-8", newline="\n")
print(f"Patch aplicado com sucesso em {SCRIPT}")
