"""
scripts/fix_exp008d_hydrate_config_method.py

Reintroduz de forma defensiva o método:
  PixDecisionEngine._hydrate_config_from_scoring_config

Motivo:
  Após cleanup EXP-008D, o runtime passou a falhar em:
    AttributeError: 'PixDecisionEngine' object has no attribute '_hydrate_config_from_scoring_config'

Este patch:
  - não altera scoring_config.json;
  - não altera a regra C1;
  - não altera simular_pipeline_e2e_v2.py;
  - apenas adiciona um binding defensivo no final de decision_engine.py;
  - valida sintaxe antes de gravar.

Uso:
  python scripts\\fix_exp008d_hydrate_config_method.py
  python -m py_compile backend\\core\\decision_engine.py
  python -m pytest tests\\test_regression_post_fase2.py -q
  python -m pytest tests\\test_regression_post_fase2.py -q -m slow
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


ROOT = find_project_root(Path(__file__).resolve().parent)
ENGINE_PATH = ROOT / "backend" / "core" / "decision_engine.py"


PATCH_MARKER = "EXP-008D PATCH — Defensive hydrate_config binding"


PATCH_BLOCK = r'''

# ============================================================
# EXP-008D PATCH — Defensive hydrate_config binding
# ============================================================
# Reintroduz PixDecisionEngine._hydrate_config_from_scoring_config quando
# o método não estiver definido na classe.
#
# Necessário porque _load_all() chama:
#   self._hydrate_config_from_scoring_config(default_config)
#
# Este binding é defensivo:
#   - só aplica se o método ainda não existir;
#   - copia para self.config apenas chaves já existentes;
#   - tenta preservar tipos bool/int/float/str dos campos do EngineConfig;
#   - não altera scoring_config.json.

def _exp008d_coerce_config_value(current_value, raw_value):
    try:
        if raw_value is None:
            return current_value

        if isinstance(current_value, bool):
            if isinstance(raw_value, str):
                return raw_value.strip().lower() in {"1", "true", "yes", "y", "sim", "s"}
            return bool(raw_value)

        if isinstance(current_value, int) and not isinstance(current_value, bool):
            return int(float(raw_value))

        if isinstance(current_value, float):
            return float(raw_value)

        return raw_value

    except Exception:
        return current_value


def _exp008d_hydrate_config_from_scoring_config(self, scoring_config=None):
    try:
        if scoring_config is None:
            try:
                import json as _json
                from pathlib import Path as _Path

                here = _Path(__file__).resolve()
                root = here.parents[2] if len(here.parents) >= 3 else here.parent.parent
                scoring_path = root / "backend" / "artefatos" / "scoring_config.json"

                if not scoring_path.exists():
                    scoring_path = here.parent.parent / "artefatos" / "scoring_config.json"

                scoring_config = _json.loads(scoring_path.read_text(encoding="utf-8"))
            except Exception:
                scoring_config = {}

        if not isinstance(scoring_config, dict):
            return getattr(self, "config", None)

        cfg_obj = getattr(self, "config", None)
        if cfg_obj is None:
            return None

        # Atualiza apenas atributos já conhecidos do EngineConfig.
        for key, raw_value in scoring_config.items():
            if not hasattr(cfg_obj, key):
                continue

            current_value = getattr(cfg_obj, key)
            setattr(cfg_obj, key, _exp008d_coerce_config_value(current_value, raw_value))

        return cfg_obj

    except Exception as exc:
        try:
            logger.warning("Falha em _hydrate_config_from_scoring_config defensivo: %s", exc)
        except Exception:
            pass
        return getattr(self, "config", None)


def _exp008d_bind_hydrate_config_method():
    try:
        if "PixDecisionEngine" not in globals():
            return

        if not hasattr(PixDecisionEngine, "_hydrate_config_from_scoring_config"):
            setattr(
                PixDecisionEngine,
                "_hydrate_config_from_scoring_config",
                _exp008d_hydrate_config_from_scoring_config,
            )
            try:
                logger.info("EXP-008D hydrate_config binding aplicado")
            except Exception:
                pass

    except Exception as exc:
        try:
            logger.warning("Falha ao aplicar EXP-008D hydrate_config binding: %s", exc)
        except Exception:
            pass


_exp008d_bind_hydrate_config_method()

'''


def backup(path: Path) -> Path:
    backup_path = path.with_suffix(
        path.suffix + f".bak_fix_hydrate_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(path, backup_path)
    print(f"[OK] Backup criado: {backup_path}")
    return backup_path


def main() -> None:
    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ENGINE_PATH}")

    text = ENGINE_PATH.read_text(encoding="utf-8")

    if PATCH_MARKER in text:
        print("[OK] Patch hydrate_config já existe. Nada a aplicar.")
        return

    if "class PixDecisionEngine" not in text:
        raise RuntimeError("Não encontrei class PixDecisionEngine no decision_engine.py")

    if "class EngineConfig" not in text:
        raise RuntimeError("Não encontrei class EngineConfig no decision_engine.py")

    new_text = text.rstrip() + "\n" + PATCH_BLOCK + "\n"

    # Valida sintaxe antes de gravar.
    compile(new_text, str(ENGINE_PATH), "exec")

    backup(ENGINE_PATH)
    ENGINE_PATH.write_text(new_text, encoding="utf-8")

    print("[OK] Binding defensivo _hydrate_config_from_scoring_config aplicado.")
    print(f"[OK] Arquivo atualizado: {ENGINE_PATH}")


if __name__ == "__main__":
    main()