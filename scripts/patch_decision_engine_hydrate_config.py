"""
scripts/patch_decision_engine_hydrate_config.py

Corrige o decision_engine.py quando existe a chamada:

    self._hydrate_config_from_scoring_config(default_config)

mas os métodos auxiliares ainda não foram adicionados à classe PixDecisionEngine.

O patch:
  - faz backup timestampado;
  - adiciona _coerce_config_value();
  - adiciona _hydrate_config_from_scoring_config();
  - não altera regras de decisão;
  - não promove LGBM v6.2;
  - apenas torna o carregamento do scoring_config seguro.
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


METHOD_BLOCK = r'''
    def _coerce_config_value(self, default_value, raw_value):
        """
        Converte valor vindo do scoring_config.json para o tipo esperado
        pelo EngineConfig.

        Usado apenas para hidratar campos que já existem no EngineConfig.
        Campos extras no JSON são ignorados com segurança.
        """
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

    def _hydrate_config_from_scoring_config(self, default_config):
        """
        Hidrata self.config com chaves do scoring_config.json.

        Regras:
          1. Só aplica campos que já existem no EngineConfig atual.
          2. Ignora chaves extras do JSON, como flags futuras.
          3. Não sobrescreve valores passados explicitamente no EngineConfig,
             pois compara o valor atual com o default.
          4. Evita quebrar o runtime quando scoring_config.json contém campos
             ainda não suportados pelo código.
        """
        scoring = getattr(self, "scoring_config", None)

        if not isinstance(scoring, dict):
            return

        fields = getattr(EngineConfig, "__dataclass_fields__", {})

        if not fields:
            return

        ignored = []

        for name in scoring.keys():
            if name not in fields:
                ignored.append(name)

        if ignored:
            try:
                logger.debug(
                    "Chaves do scoring_config ignoradas por não existirem no EngineConfig: %s",
                    sorted(ignored),
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

            # Se o valor atual difere do default, provavelmente veio de override
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

'''


def main() -> None:
    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ENGINE_PATH}")

    text = ENGINE_PATH.read_text(encoding="utf-8")

    if "self._hydrate_config_from_scoring_config(default_config)" not in text:
        print("[INFO] A chamada _hydrate_config_from_scoring_config(default_config) não existe. Nada a corrigir.")
        return

    if "def _hydrate_config_from_scoring_config" in text:
        print("[OK] Método _hydrate_config_from_scoring_config já existe. Nada a aplicar.")
        return

    marker = "\n    def _load_all(self) -> None:"

    if marker not in text:
        marker = "\n    def _load_all(self):"

    if marker not in text:
        raise RuntimeError("Não encontrei o método _load_all para inserir o patch antes dele.")

    backup_path = ENGINE_PATH.with_suffix(
        f".py.bak_hydrate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(ENGINE_PATH, backup_path)
    print(f"[OK] Backup criado: {backup_path}")

    patched = text.replace(marker, METHOD_BLOCK + marker, 1)

    ENGINE_PATH.write_text(patched, encoding="utf-8")
    print(f"[OK] Patch aplicado em: {ENGINE_PATH}")
    print("[INFO] Agora valide com py_compile.")


if __name__ == "__main__":
    main()