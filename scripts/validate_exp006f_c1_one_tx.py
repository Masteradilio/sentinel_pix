"""
Validação mínima da C1 no runtime real.

Processa apenas a transação recuperada pelo EXP-006F:
  E0000020820260205003505340630525

Espera:
  decisao == CONFIRMAR
  exp006f_c1_applied == True
"""

from __future__ import annotations

import sys
from pathlib import Path


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists() and (p / "experimentos").exists():
            return p
    return start.parent


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from experimentos.utils_experimentos import (  # noqa: E402
    get_logger,
    load_dataset,
    process_dataframe_via_orquestrador,
)


TARGET_TX = "E0000020820260205003505340630525"


def main() -> None:
    logger = get_logger("VALIDATE-C1")

    df = load_dataset()

    if "transaction_id" not in df.columns:
        raise RuntimeError("Dataset não possui coluna transaction_id.")

    sample = df[df["transaction_id"].astype(str).eq(TARGET_TX)].copy()

    if sample.empty:
        raise RuntimeError(f"Transação alvo não encontrada no dataset: {TARGET_TX}")

    print(f"[OK] Transação alvo encontrada: {len(sample)} linha(s)")

    preds = process_dataframe_via_orquestrador(
        sample,
        workers=1,
        logger=logger,
        engine_config_overrides=None,
    )

    cols = [
        "transaction_id",
        "is_fraud",
        "vl_pix",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "lgbm_raw",
        "score_final",
        "decisao",
        "exp006f_c1_applied",
        "exp006f_c1_reason",
        "decisao_original_exp006f_c1",
        "score_final_original_exp006f_c1",
    ]
    cols = [c for c in cols if c in preds.columns]

    print(preds[cols].to_string(index=False))

    decisao = str(preds.iloc[0].get("decisao", ""))
    applied = bool(preds.iloc[0].get("exp006f_c1_applied", False))

    if decisao == "CONFIRMAR" and applied:
        print("[OK] C1 ativa no runtime real.")
    else:
        raise RuntimeError(
            f"C1 não foi aplicada como esperado. decisao={decisao}, exp006f_c1_applied={applied}"
        )


if __name__ == "__main__":
    main()