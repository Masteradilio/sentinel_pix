#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Patch EXP-014B-R3A — Portable Base Recipe

Corrige o erro:
  RuntimeError: Coluna alias da base ausente no expandido: exp013k_base_pred

Causa:
  O script inferiu que exp013k_base_pred era a melhor "receita" para reconstruir
  pred_STRICT_RECALL95_SAFE_ONLY, mas essa coluna só existe no artefato pequeno
  do EXP-013K. Ela não existe no dataset expandido.

Correção:
  1. Evita usar alias_column se a coluna não existir no expandido.
  2. Dá preferência a receitas portáveis por threshold em colunas de score que
     existam no expandido.
  3. Se não houver receita portável segura, encerra de forma controlada com
     DONE_BASE_RECIPE_NOT_RECOVERED, em vez de quebrar no meio.

Uso:
  python scripts\patch_exp014b_r3a_portable_base_recipe.py

Depois:
  python scripts\exp_014b_r3a_champion_replay_expanded.py

Opcional diagnóstico:
  python scripts\exp_014b_r3a_champion_replay_expanded.py --allow-imperfect-base-recipe
"""

from pathlib import Path

SCRIPT = Path("scripts/exp_014b_r3a_champion_replay_expanded.py")

if not SCRIPT.exists():
    raise FileNotFoundError(f"Não encontrei {SCRIPT}. Execute na raiz do projeto rebuild_pix.")

text = SCRIPT.read_text(encoding="utf-8")

if "def choose_portable_base_recipe(" not in text:
    marker = "\n\ndef apply_base_recipe(df: pd.DataFrame, recipe: dict[str, Any]) -> np.ndarray:\n"
    if marker not in text:
        raise RuntimeError("Não encontrei o ponto para inserir choose_portable_base_recipe antes de apply_base_recipe.")

    helper = """

def choose_portable_base_recipe(
    recipe: dict[str, Any] | None,
    report: pd.DataFrame,
    expanded: pd.DataFrame,
    min_match_rate: float,
    allow_imperfect: bool,
) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    \"\"\"
    Garante que a receita da base seja aplicável ao dataset expandido.

    O erro original ocorreu porque exp013k_base_pred existe no dataset pequeno,
    mas não no expandido. Esta função evita escolher aliases não portáveis e
    prefere threshold em score_col presente no expandido.
    \"\"\"
    rows = pd.DataFrame() if report is None or report.empty else report.copy()

    # 1) existing_column/alias só é válido se a coluna existir no expandido.
    if recipe:
        rtype = recipe.get("type")
        if rtype == "existing_column" and recipe.get("col") in expanded.columns:
            return recipe, rows
        if rtype == "alias_column" and recipe.get("source_col") in expanded.columns:
            return recipe, rows
        if rtype == "threshold" and recipe.get("score_col") in expanded.columns:
            return recipe, rows

    if rows.empty:
        return None, pd.DataFrame([{
            "status": "no_recipe_report_available",
            "reason": "infer_base_recipe não retornou candidatos.",
        }])

    # 2) Marcar portabilidade.
    rows = rows.copy()
    rows["portable_to_expanded"] = False
    rows["portable_reason"] = ""

    for idx, row in rows.iterrows():
        recipe_type = str(row.get("recipe_type", ""))
        score_col = row.get("score_col")
        if recipe_type == "alias_column":
            ok = isinstance(score_col, str) and score_col in expanded.columns
            rows.loc[idx, "portable_to_expanded"] = ok
            rows.loc[idx, "portable_reason"] = "alias_exists_in_expanded" if ok else "alias_missing_in_expanded"
        elif recipe_type == "threshold":
            ok = isinstance(score_col, str) and score_col in expanded.columns
            rows.loc[idx, "portable_to_expanded"] = ok
            rows.loc[idx, "portable_reason"] = "threshold_score_exists_in_expanded" if ok else "threshold_score_missing_in_expanded"
        else:
            rows.loc[idx, "portable_reason"] = f"unsupported_recipe_type:{recipe_type}"

    # 3) Preferir threshold portável, porque é reprodutível fora do dataset pequeno.
    threshold_rows = rows[
        (rows["recipe_type"].astype(str) == "threshold")
        & (rows["portable_to_expanded"] == True)
    ].copy()

    if not threshold_rows.empty:
        threshold_rows["match_rate"] = pd.to_numeric(threshold_rows["match_rate"], errors="coerce").fillna(0.0)
        threshold_rows["exact_match_bool"] = threshold_rows["exact_match"].astype(str).str.lower().isin(["true", "1", "1.0"])
        threshold_rows = threshold_rows.sort_values(
            ["exact_match_bool", "match_rate"],
            ascending=[False, False],
        ).reset_index(drop=True)

        best = threshold_rows.iloc[0].to_dict()
        match_rate = float(best.get("match_rate", 0.0))
        exact = bool(best.get("exact_match_bool", False))

        if exact or match_rate >= min_match_rate or allow_imperfect:
            chosen = {
                "type": "threshold",
                "score_col": str(best["score_col"]),
                "direction": str(best["direction"]),
                "threshold": float(best["threshold"]),
                "match_rate": match_rate,
                "exact_match": exact,
                "chosen_by": "portable_threshold_fallback",
            }
            rows["chosen_portable_recipe"] = False
            mask = (
                (rows["recipe_type"].astype(str) == "threshold")
                & (rows["score_col"].astype(str) == str(best["score_col"]))
                & (rows["direction"].astype(str) == str(best["direction"]))
                & (pd.to_numeric(rows["threshold"], errors="coerce") == float(best["threshold"]))
            )
            rows.loc[mask, "chosen_portable_recipe"] = True
            return chosen, rows

    # 4) Alias portável só como fallback.
    alias_rows = rows[
        (rows["recipe_type"].astype(str) == "alias_column")
        & (rows["portable_to_expanded"] == True)
    ].copy()

    if not alias_rows.empty:
        alias_rows["match_rate"] = pd.to_numeric(alias_rows["match_rate"], errors="coerce").fillna(0.0)
        alias_rows["exact_match_bool"] = alias_rows["exact_match"].astype(str).str.lower().isin(["true", "1", "1.0"])
        alias_rows = alias_rows.sort_values(
            ["exact_match_bool", "match_rate"],
            ascending=[False, False],
        ).reset_index(drop=True)

        best = alias_rows.iloc[0].to_dict()
        match_rate = float(best.get("match_rate", 0.0))
        exact = bool(best.get("exact_match_bool", False))

        if exact or match_rate >= min_match_rate or allow_imperfect:
            chosen = {
                "type": "alias_column",
                "source_col": str(best["score_col"]),
                "match_rate": match_rate,
                "exact_match": exact,
                "chosen_by": "portable_alias_fallback",
            }
            rows["chosen_portable_recipe"] = False
            rows.loc[
                (rows["recipe_type"].astype(str) == "alias_column")
                & (rows["score_col"].astype(str) == str(best["score_col"])),
                "chosen_portable_recipe"
            ] = True
            return chosen, rows

    rows["chosen_portable_recipe"] = False
    return None, rows
"""
    text = text.replace(marker, helper + marker)

old = """        recipe, report = infer_base_recipe(base_source, BASE_COL, args.base_min_match_rate)
        base_recovery_df = report
        if recipe and (recipe.get("exact_match") or args.allow_imperfect_base_recipe):
            base_recipe = recipe
"""

new = """        recipe, report = infer_base_recipe(base_source, BASE_COL, args.base_min_match_rate)
        base_recipe, base_recovery_df = choose_portable_base_recipe(
            recipe=recipe,
            report=report,
            expanded=expanded,
            min_match_rate=args.base_min_match_rate,
            allow_imperfect=args.allow_imperfect_base_recipe,
        )
"""

if old in text:
    text = text.replace(old, new)
else:
    block_start = text.find("recipe, report = infer_base_recipe(base_source, BASE_COL, args.base_min_match_rate)")
    block_end = text.find("base_recovery_df.to_csv", block_start)
    if block_start == -1 or block_end == -1 or "choose_portable_base_recipe" not in text[block_start:block_end]:
        raise RuntimeError("Não encontrei o bloco de inferência da base para substituir. Talvez o script esteja diferente.")
    # If already patched, do nothing.

SCRIPT.write_text(text, encoding="utf-8", newline="\n")
print(f"Patch aplicado com sucesso em {SCRIPT}")
