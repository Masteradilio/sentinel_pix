"""
scripts/patch_exp006f_c1_simular_e2e_wrapper.py

Aplica EXP-006F-C1 no DataFrame final produzido por:
  backend/scripts/simular_pipeline_e2e_v2.py

Este é o ponto determinístico usado por process_dataframe_via_orquestrador()
nos experimentos.

C1:
  decisao == APROVAR
  first_receiver_flag == 1
  pix_key_random_flag == 0
  qt_tempo_relacionamento_mes <= 12
  100 <= vl_pix < 500
  0.06 <= lgbm_raw < 0.10
  58 <= score_final < 62
  se_score <= 0
  beh_score <= 0

Ação:
  APROVAR -> CONFIRMAR

Uso:
  python scripts\\patch_exp006f_c1_simular_e2e_wrapper.py
  python -m py_compile backend\\scripts\\simular_pipeline_e2e_v2.py
  python scripts\\validate_exp006f_c1_one_tx.py
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists():
            return p
    return start.parent


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)
SIMULAR_PATH = PROJECT_ROOT / "backend" / "scripts" / "simular_pipeline_e2e_v2.py"


WRAPPER_BLOCK = r'''

# ============================================================
# EXP-006F-C1 PATCH — Final predictions_df wrapper
# ============================================================
# Aplica C1 no DataFrame final produzido por process_batch_sequential/parallel.
# Este é o ponto usado por experimentos.utils_experimentos.process_dataframe_via_orquestrador.

def _exp006f_c1_read_cfg(key, default):
    try:
        import json as _json
        from pathlib import Path as _Path

        here = _Path(__file__).resolve()
        root = here.parents[2]
        scoring_path = root / "backend" / "artefatos" / "scoring_config.json"

        cfg = _json.loads(scoring_path.read_text(encoding="utf-8"))
        return cfg.get(key, default)
    except Exception:
        return default


def _exp006f_c1_apply_to_predictions_df(predictions_df):
    try:
        import pandas as _pd
        import numpy as _np

        if predictions_df is None or not hasattr(predictions_df, "copy"):
            return predictions_df

        df = predictions_df.copy()

        required = [
            "decisao",
            "vl_pix",
            "qt_tempo_relacionamento_mes",
            "first_receiver_flag",
            "pix_key_random_flag",
            "lgbm_raw",
            "se_score",
            "beh_score",
            "score_final",
        ]

        missing = [c for c in required if c not in df.columns]
        if missing:
            return predictions_df

        enabled = bool(_exp006f_c1_read_cfg("exp006f_c1_enabled", True))
        if not enabled:
            return predictions_df

        def _num(col, default=0.0):
            return _pd.to_numeric(df[col], errors="coerce").fillna(default)

        min_score = float(_exp006f_c1_read_cfg("exp006f_c1_min_score", 58.0))
        max_score = float(_exp006f_c1_read_cfg("exp006f_c1_max_score", 62.0))
        min_valor = float(_exp006f_c1_read_cfg("exp006f_c1_min_valor", 100.0))
        max_valor = float(_exp006f_c1_read_cfg("exp006f_c1_max_valor", 500.0))
        max_rel = float(_exp006f_c1_read_cfg("exp006f_c1_max_rel_meses", 12.0))
        min_lgbm = float(_exp006f_c1_read_cfg("exp006f_c1_min_lgbm_raw", 0.06))
        max_lgbm = float(_exp006f_c1_read_cfg("exp006f_c1_max_lgbm_raw", 0.10))
        max_se = float(_exp006f_c1_read_cfg("exp006f_c1_max_se_score", 0.0))
        max_beh = float(_exp006f_c1_read_cfg("exp006f_c1_max_beh_score", 0.0))

        decisao = df["decisao"].astype(str).str.upper()
        vl = _num("vl_pix")
        rel = _num("qt_tempo_relacionamento_mes", 999.0)
        first = _num("first_receiver_flag").astype(int)
        pix_random = _num("pix_key_random_flag").astype(int)
        lgbm = _num("lgbm_raw")
        se = _num("se_score")
        beh = _num("beh_score")
        score = _num("score_final")

        mask = (
            decisao.eq("APROVAR")
            & first.eq(1)
            & pix_random.eq(0)
            & rel.le(max_rel)
            & vl.ge(min_valor)
            & vl.lt(max_valor)
            & lgbm.ge(min_lgbm)
            & lgbm.lt(max_lgbm)
            & score.ge(min_score)
            & score.lt(max_score)
            & se.le(max_se)
            & beh.le(max_beh)
        )

        if not bool(mask.any()):
            if "exp006f_c1_applied" not in df.columns:
                df["exp006f_c1_applied"] = False
            return df

        if "exp006f_c1_applied" not in df.columns:
            df["exp006f_c1_applied"] = False

        if "decisao_original_exp006f_c1" not in df.columns:
            df["decisao_original_exp006f_c1"] = ""

        if "score_final_original_exp006f_c1" not in df.columns:
            df["score_final_original_exp006f_c1"] = _np.nan

        if "exp006f_c1_reason" not in df.columns:
            df["exp006f_c1_reason"] = ""

        idx = df.index[mask]

        df.loc[idx, "decisao_original_exp006f_c1"] = df.loc[idx, "decisao"]
        df.loc[idx, "score_final_original_exp006f_c1"] = df.loc[idx, "score_final"]
        df.loc[idx, "decisao"] = "CONFIRMAR"
        df.loc[idx, "score_final"] = score.loc[idx].apply(lambda x: max(float(x), 62.0))
        df.loc[idx, "exp006f_c1_applied"] = True
        df.loc[idx, "exp006f_c1_reason"] = (
            "C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER: APROVAR->CONFIRMAR | "
            "rel<=12, first_receiver=1, pix_random=0, 100<=vl<500, "
            "0.06<=lgbm<0.10, score_min_runtime, SE=0, BEH=0"
        )

        return df

    except Exception as exc:
        try:
            logger.warning("Falha ao aplicar EXP-006F-C1 no predictions_df final: %s", exc)
        except Exception:
            pass
        return predictions_df


def _exp006f_c1_wrap_batch_function(fn):
    try:
        if getattr(fn, "_exp006f_c1_final_df_wrapped", False):
            return fn
    except Exception:
        pass

    def _wrapped(*args, **kwargs):
        predictions_df = fn(*args, **kwargs)
        return _exp006f_c1_apply_to_predictions_df(predictions_df)

    try:
        _wrapped.__name__ = getattr(fn, "__name__", "_wrapped")
        _wrapped.__doc__ = getattr(fn, "__doc__", None)
        _wrapped._exp006f_c1_final_df_wrapped = True
    except Exception:
        pass

    return _wrapped


try:
    if "process_batch_sequential" in globals():
        process_batch_sequential = _exp006f_c1_wrap_batch_function(process_batch_sequential)

    if "process_batch_parallel" in globals():
        process_batch_parallel = _exp006f_c1_wrap_batch_function(process_batch_parallel)

    EXP006F_C1_FINAL_DF_WRAPPER_ACTIVE = True

    try:
        logger.info("EXP-006F-C1 final predictions_df wrapper ativo")
    except Exception:
        pass

except Exception as exc:
    EXP006F_C1_FINAL_DF_WRAPPER_ACTIVE = False
    try:
        logger.warning("Falha ao ativar EXP-006F-C1 final predictions_df wrapper: %s", exc)
    except Exception:
        pass

'''


def main() -> None:
    if not SIMULAR_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {SIMULAR_PATH}")

    text = SIMULAR_PATH.read_text(encoding="utf-8")

    if "EXP006F_C1_FINAL_DF_WRAPPER_ACTIVE" in text:
        print("[OK] Wrapper C1 final predictions_df já existe. Nada a aplicar.")
        return

    backup_path = SIMULAR_PATH.with_suffix(
        f".py.bak_exp006f_c1_finaldf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(SIMULAR_PATH, backup_path)
    print(f"[OK] Backup criado: {backup_path}")

    SIMULAR_PATH.write_text(text.rstrip() + "\n" + WRAPPER_BLOCK + "\n", encoding="utf-8")
    print(f"[OK] Wrapper C1 aplicado em: {SIMULAR_PATH}")


if __name__ == "__main__":
    main()