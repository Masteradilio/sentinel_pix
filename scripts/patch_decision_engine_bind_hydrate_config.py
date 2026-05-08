"""
scripts/patch_decision_engine_bind_hydrate_config.py

Corrige decision_engine.py quando o método _hydrate_config_from_scoring_config
existe no texto, mas não está vinculado à classe PixDecisionEngine.

A solução adiciona um shim no final do arquivo:

    PixDecisionEngine._coerce_config_value = ...
    PixDecisionEngine._hydrate_config_from_scoring_config = ...

Isso evita mexer em indentação interna da classe.
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


SHIM = r'''

# ============================================================
# FASE 2 PATCH — Bind seguro de hidratação do scoring_config
# ============================================================
# Este bloco é intencionalmente definido fora da classe e vinculado
# dinamicamente para corrigir casos em que o método foi inserido no
# arquivo, mas não ficou como método real de PixDecisionEngine.

def _fase2_coerce_config_value(self, default_value, raw_value):
    if raw_value is None:
        return default_value

    if isinstance(default_value, bool):
        if isinstance(raw_value, str):
            return raw_value.strip().lower() in {"1", "true", "yes", "sim", "s"}
        return bool(raw_value)

    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(float(raw_value))

    if isinstance(default_value, float):
        return float(raw_value)

    return raw_value


def _fase2_hydrate_config_from_scoring_config(self, default_config):
    scoring = getattr(self, "scoring_config", None)

    if not isinstance(scoring, dict):
        return

    fields = getattr(EngineConfig, "__dataclass_fields__", {})

    if not fields:
        return

    ignored = sorted([k for k in scoring.keys() if k not in fields])

    if ignored:
        try:
            logger.debug(
                "Chaves do scoring_config ignoradas por não existirem no EngineConfig: %s",
                ignored,
            )
        except Exception:
            pass

    for name in fields.keys():
        if name not in scoring:
            continue

        # Nunca alterar diretório de artefatos por scoring_config.
        if name == "artefatos_dir":
            continue

        try:
            current_value = getattr(self.config, name)
            default_value = getattr(default_config, name)
        except Exception:
            continue

        # Se current_value difere do default, provavelmente veio de override
        # explícito no experimento. Não sobrescrever.
        if current_value != default_value:
            continue

        try:
            setattr(
                self.config,
                name,
                self._coerce_config_value(default_value, scoring[name]),
            )
        except Exception as exc:
            try:
                logger.warning(
                    "Falha ao hidratar EngineConfig.%s a partir do scoring_config: %s",
                    name,
                    exc,
                )
            except Exception:
                pass


# Bind defensivo: mesmo que já exista texto com o nome do método em outro lugar,
# garantimos que a classe PixDecisionEngine realmente tenha os métodos.
if not hasattr(PixDecisionEngine, "_coerce_config_value"):
    PixDecisionEngine._coerce_config_value = _fase2_coerce_config_value

if not hasattr(PixDecisionEngine, "_hydrate_config_from_scoring_config"):
    PixDecisionEngine._hydrate_config_from_scoring_config = _fase2_hydrate_config_from_scoring_config

'''


def main() -> None:
    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ENGINE_PATH}")

    text = ENGINE_PATH.read_text(encoding="utf-8")

    if "_fase2_hydrate_config_from_scoring_config" in text:
        print("[OK] Shim FASE 2 já existe no decision_engine.py. Nada a aplicar.")
        return

    backup_path = ENGINE_PATH.with_suffix(
        f".py.bak_bind_hydrate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(ENGINE_PATH, backup_path)
    print(f"[OK] Backup criado: {backup_path}")

    ENGINE_PATH.write_text(text.rstrip() + "\n" + SHIM + "\n", encoding="utf-8")
    print(f"[OK] Shim aplicado em: {ENGINE_PATH}")


if __name__ == "__main__":
    main()