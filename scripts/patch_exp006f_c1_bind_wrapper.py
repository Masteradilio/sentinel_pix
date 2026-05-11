"""
scripts/patch_exp006f_c1_bind_wrapper.py

Ativa a regra EXP-006F-C1 no fluxo real do PixDecisionEngine sem depender
do ponto exato de `return result`.

O patch adiciona um wrapper defensivo no final de decision_engine.py:
  - intercepta métodos do PixDecisionEngine que retornam dict com "decisao";
  - aplica _apply_exp006f_c1_near_threshold_exception(tx, result);
  - só altera resultado se a decisão ainda for APROVAR e todas as condições C1 baterem;
  - é idempotente e não altera scoring_config.json.

Uso:
  python scripts\\patch_exp006f_c1_bind_wrapper.py
  python -m py_compile backend\\core\\decision_engine.py
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
ENGINE_PATH = PROJECT_ROOT / "backend" / "core" / "decision_engine.py"


WRAPPER_BLOCK = r'''

# ============================================================
# EXP-006F-C1 PATCH — Runtime wrapper defensivo
# ============================================================
# Este bloco ativa a exceção C1 mesmo quando não foi possível injetar a chamada
# no ponto exato de return do PixDecisionEngine.
#
# Ele embrulha métodos do PixDecisionEngine que retornam dict contendo "decisao".
# A regra C1 só altera o resultado quando:
#   - exp006f_c1_enabled=True;
#   - resultado ainda é APROVAR;
#   - todas as condições estreitas da C1 batem.
#
# Idempotente: se rodar mais de uma vez, não embrulha novamente.

def _exp006f_c1_extract_tx_from_call(args, kwargs, result):
    try:
        for key in ("tx", "transaction", "transacao", "row", "registro", "dados"):
            if key in kwargs:
                return kwargs[key]

        if args:
            return args[0]

        if isinstance(result, dict):
            return result

        return {}
    except Exception:
        return {}


def _exp006f_c1_apply_to_result(self, tx, result):
    try:
        if isinstance(result, dict):
            if "decisao" in result:
                return self._apply_exp006f_c1_near_threshold_exception(tx, result)
            return result

        if isinstance(result, tuple):
            changed = False
            items = list(result)
            for i, item in enumerate(items):
                if isinstance(item, dict) and "decisao" in item:
                    items[i] = self._apply_exp006f_c1_near_threshold_exception(tx, item)
                    changed = True
            return tuple(items) if changed else result

        if isinstance(result, list):
            changed = False
            items = list(result)
            for i, item in enumerate(items):
                if isinstance(item, dict) and "decisao" in item:
                    items[i] = self._apply_exp006f_c1_near_threshold_exception(tx, item)
                    changed = True
            return items if changed else result

        return result

    except Exception as exc:
        try:
            logger.warning("Falha no wrapper EXP-006F-C1: %s", exc)
        except Exception:
            pass
        return result


def _exp006f_c1_wrap_method(method):
    try:
        if getattr(method, "_exp006f_c1_wrapped", False):
            return method
    except Exception:
        pass

    def _wrapped(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        tx = _exp006f_c1_extract_tx_from_call(args, kwargs, result)
        return _exp006f_c1_apply_to_result(self, tx, result)

    try:
        _wrapped.__name__ = getattr(method, "__name__", "_wrapped")
        _wrapped.__doc__ = getattr(method, "__doc__", None)
        _wrapped._exp006f_c1_wrapped = True
    except Exception:
        pass

    return _wrapped


def _exp006f_c1_bind_runtime_wrappers():
    try:
        import inspect as _inspect

        excluded = {
            "__init__",
            "_apply_exp006f_c1_near_threshold_exception",
            "_coerce_config_value",
            "_hydrate_config_from_scoring_config",
        }

        bound = 0

        for name, attr in list(PixDecisionEngine.__dict__.items()):
            if name in excluded:
                continue

            if name.startswith("__"):
                continue

            if not _inspect.isfunction(attr):
                continue

            if getattr(attr, "_exp006f_c1_wrapped", False):
                continue

            should_wrap = False

            try:
                src = _inspect.getsource(attr)
                if "decisao" in src or "score_final" in src or "return " in src:
                    should_wrap = True
            except Exception:
                # Fallback: embrulhar métodos públicos comuns.
                if not name.startswith("_"):
                    should_wrap = True

            if not should_wrap:
                continue

            setattr(PixDecisionEngine, name, _exp006f_c1_wrap_method(attr))
            bound += 1

        PixDecisionEngine._exp006f_c1_wrappers_bound = bound

        try:
            logger.info("EXP-006F-C1 runtime wrappers bound: %s", bound)
        except Exception:
            pass

    except Exception as exc:
        try:
            logger.warning("Falha ao ativar wrappers EXP-006F-C1: %s", exc)
        except Exception:
            pass


_exp006f_c1_bind_runtime_wrappers()

'''


def main() -> None:
    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ENGINE_PATH}")

    text = ENGINE_PATH.read_text(encoding="utf-8")

    if "_exp006f_c1_bind_runtime_wrappers" in text:
        print("[OK] Wrapper EXP-006F-C1 já existe. Nada a aplicar.")
        return

    if "def _apply_exp006f_c1_near_threshold_exception" not in text:
        raise RuntimeError(
            "O método _apply_exp006f_c1_near_threshold_exception ainda não existe. "
            "Rode primeiro scripts/patch_permanente_exp006f_c1.py"
        )

    backup_path = ENGINE_PATH.with_suffix(
        f".py.bak_exp006f_c1_wrapper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(ENGINE_PATH, backup_path)
    print(f"[OK] Backup criado: {backup_path}")

    ENGINE_PATH.write_text(text.rstrip() + "\n" + WRAPPER_BLOCK + "\n", encoding="utf-8")

    print(f"[OK] Wrapper EXP-006F-C1 aplicado em: {ENGINE_PATH}")


if __name__ == "__main__":
    main()