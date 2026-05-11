"""
scripts/patch_exp006f_c1_pipeline_wrapper.py

Ativa C1 no PipelineOrquestrador, no ponto em que o resultado final já contém
decisao/score_final/lgbm_raw/features.

Este patch é mais determinístico que o wrapper no PixDecisionEngine porque
atua sobre o dicionário final retornado pelo orquestrador.

Uso:
  python scripts\\patch_exp006f_c1_pipeline_wrapper.py
  python -m py_compile backend\\core\\pipeline_orquestrador.py
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
PIPELINE_PATH = PROJECT_ROOT / "backend" / "core" / "pipeline_orquestrador.py"


WRAPPER_BLOCK = r'''

# ============================================================
# EXP-006F-C1 PATCH — PipelineOrquestrador final-result wrapper
# ============================================================
# Aplica C1 no resultado final já montado pelo PipelineOrquestrador.
#
# Regra:
#   decisao == APROVAR
#   first_receiver_flag == 1
#   pix_key_random_flag == 0
#   qt_tempo_relacionamento_mes <= 12
#   100 <= vl_pix < 500
#   0.06 <= lgbm_raw < 0.10
#   58 <= score_final < 62
#   se_score <= 0
#   beh_score <= 0
#
# Ação:
#   APROVAR -> CONFIRMAR
#
# Evidência EXP-006F:
#   seed 42: TP 346->347, FP 14->14, FN 9->8
#   seed 123: TP 346->347, FP 12->12, FN 9->8

def _exp006f_c1_get_config_value(key, default):
    try:
        import json as _json
        from pathlib import Path as _Path

        here = _Path(__file__).resolve()
        root = here.parents[2] if len(here.parents) >= 3 else here.parent.parent
        scoring_path = root / "backend" / "artefatos" / "scoring_config.json"

        if not scoring_path.exists():
            scoring_path = here.parent.parent / "artefatos" / "scoring_config.json"

        cfg = _json.loads(scoring_path.read_text(encoding="utf-8"))
        return cfg.get(key, default)
    except Exception:
        return default


def _exp006f_c1_as_float(obj, key, default=0.0):
    try:
        value = obj.get(key, default)
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _exp006f_c1_as_int(obj, key, default=0):
    try:
        value = obj.get(key, default)
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _exp006f_c1_merge_tx_result(tx, result):
    merged = {}

    try:
        if isinstance(tx, dict):
            merged.update(tx)
    except Exception:
        pass

    try:
        if isinstance(result, dict):
            merged.update(result)
    except Exception:
        pass

    return merged


def _exp006f_c1_apply_to_final_dict(tx, result):
    try:
        if not isinstance(result, dict):
            return result

        if str(result.get("decisao", "")).upper() != "APROVAR":
            return result

        enabled = bool(_exp006f_c1_get_config_value("exp006f_c1_enabled", True))
        if not enabled:
            return result

        merged = _exp006f_c1_merge_tx_result(tx, result)

        vl_pix = _exp006f_c1_as_float(merged, "vl_pix", 0.0)
        rel = _exp006f_c1_as_float(merged, "qt_tempo_relacionamento_mes", 999.0)
        first_receiver = _exp006f_c1_as_int(merged, "first_receiver_flag", 0)
        pix_random = _exp006f_c1_as_int(merged, "pix_key_random_flag", 0)
        lgbm_raw = _exp006f_c1_as_float(merged, "lgbm_raw", 0.0)
        score_final = _exp006f_c1_as_float(merged, "score_final", 0.0)
        se_score = _exp006f_c1_as_float(merged, "se_score", 0.0)
        beh_score = _exp006f_c1_as_float(merged, "beh_score", 0.0)

        min_score = float(_exp006f_c1_get_config_value("exp006f_c1_min_score", 58.0))
        max_score = float(_exp006f_c1_get_config_value("exp006f_c1_max_score", 62.0))
        min_valor = float(_exp006f_c1_get_config_value("exp006f_c1_min_valor", 100.0))
        max_valor = float(_exp006f_c1_get_config_value("exp006f_c1_max_valor", 500.0))
        max_rel = float(_exp006f_c1_get_config_value("exp006f_c1_max_rel_meses", 12.0))
        min_lgbm = float(_exp006f_c1_get_config_value("exp006f_c1_min_lgbm_raw", 0.06))
        max_lgbm = float(_exp006f_c1_get_config_value("exp006f_c1_max_lgbm_raw", 0.10))
        max_se = float(_exp006f_c1_get_config_value("exp006f_c1_max_se_score", 0.0))
        max_beh = float(_exp006f_c1_get_config_value("exp006f_c1_max_beh_score", 0.0))

        require_first = bool(_exp006f_c1_get_config_value("exp006f_c1_require_first_receiver", True))
        require_not_random = bool(_exp006f_c1_get_config_value("exp006f_c1_require_not_pix_random", True))

        if require_first and first_receiver != 1:
            return result

        if require_not_random and pix_random != 0:
            return result

        if not (min_valor <= vl_pix < max_valor):
            return result

        if rel > max_rel:
            return result

        if not (min_lgbm <= lgbm_raw < max_lgbm):
            return result

        if not (min_score <= score_final < max_score):
            return result

        if se_score > max_se:
            return result

        if beh_score > max_beh:
            return result

        out = dict(result)
        out["decisao_original_exp006f_c1"] = result.get("decisao")
        out["score_final_original_exp006f_c1"] = score_final
        out["decisao"] = "CONFIRMAR"
        out["score_final"] = max(score_final, 62.0)
        out["exp006f_c1_applied"] = True
        out["exp006f_c1_reason"] = (
            "C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER: APROVAR->CONFIRMAR | "
            "rel<=12, first_receiver=1, pix_random=0, 100<=vl<500, "
            "0.06<=lgbm<0.10, score_min_runtime, SE=0, BEH=0"
        )

        return out

    except Exception as exc:
        try:
            logger.warning("Falha ao aplicar C1 no PipelineOrquestrador: %s", exc)
        except Exception:
            pass
        return result


def _exp006f_c1_apply_to_any_result(tx, result):
    try:
        if isinstance(result, dict):
            return _exp006f_c1_apply_to_final_dict(tx, result)

        if isinstance(result, list):
            changed = False
            out = []
            for item in result:
                if isinstance(item, dict):
                    new_item = _exp006f_c1_apply_to_final_dict(tx, item)
                    changed = changed or (new_item is not item)
                    out.append(new_item)
                else:
                    out.append(item)
            return out if changed else result

        if isinstance(result, tuple):
            changed = False
            out = []
            for item in result:
                if isinstance(item, dict):
                    new_item = _exp006f_c1_apply_to_final_dict(tx, item)
                    changed = changed or (new_item is not item)
                    out.append(new_item)
                else:
                    out.append(item)
            return tuple(out) if changed else result

        return result
    except Exception:
        return result


def _exp006f_c1_extract_tx(args, kwargs):
    try:
        for key in ("tx", "transaction", "transacao", "row", "registro", "dados"):
            if key in kwargs:
                return kwargs[key]

        for arg in args:
            if isinstance(arg, dict):
                return arg

        return {}
    except Exception:
        return {}


def _exp006f_c1_wrap_pipeline_method(method):
    try:
        if getattr(method, "_exp006f_c1_pipeline_wrapped", False):
            return method
    except Exception:
        pass

    def _wrapped(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        tx = _exp006f_c1_extract_tx(args, kwargs)
        return _exp006f_c1_apply_to_any_result(tx, result)

    try:
        _wrapped.__name__ = getattr(method, "__name__", "_wrapped")
        _wrapped.__doc__ = getattr(method, "__doc__", None)
        _wrapped._exp006f_c1_pipeline_wrapped = True
    except Exception:
        pass

    return _wrapped


def _exp006f_c1_bind_pipeline_wrappers():
    try:
        import inspect as _inspect

        cls = globals().get("PipelineOrquestrador")
        if cls is None:
            return

        bound = 0

        for name, attr in list(cls.__dict__.items()):
            if name.startswith("__"):
                continue

            if getattr(attr, "_exp006f_c1_pipeline_wrapped", False):
                continue

            if not _inspect.isfunction(attr):
                continue

            should_wrap = False

            try:
                src = _inspect.getsource(attr)
                if (
                    "decisao" in src
                    or "score_final" in src
                    or "process" in name.lower()
                    or "execut" in name.lower()
                    or "orquestr" in name.lower()
                ):
                    should_wrap = True
            except Exception:
                if not name.startswith("_"):
                    should_wrap = True

            if not should_wrap:
                continue

            setattr(cls, name, _exp006f_c1_wrap_pipeline_method(attr))
            bound += 1

        cls._exp006f_c1_pipeline_wrappers_bound = bound

        try:
            logger.info("EXP-006F-C1 pipeline wrappers bound: %s", bound)
        except Exception:
            pass

    except Exception as exc:
        try:
            logger.warning("Falha ao ativar C1 no PipelineOrquestrador: %s", exc)
        except Exception:
            pass


_exp006f_c1_bind_pipeline_wrappers()

'''


def main() -> None:
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {PIPELINE_PATH}")

    text = PIPELINE_PATH.read_text(encoding="utf-8")

    if "_exp006f_c1_bind_pipeline_wrappers" in text:
        print("[OK] Wrapper C1 no PipelineOrquestrador já existe. Nada a aplicar.")
        return

    backup_path = PIPELINE_PATH.with_suffix(
        f".py.bak_exp006f_c1_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(PIPELINE_PATH, backup_path)
    print(f"[OK] Backup criado: {backup_path}")

    PIPELINE_PATH.write_text(text.rstrip() + "\n" + WRAPPER_BLOCK + "\n", encoding="utf-8")
    print(f"[OK] Wrapper C1 aplicado no PipelineOrquestrador: {PIPELINE_PATH}")


if __name__ == "__main__":
    main()