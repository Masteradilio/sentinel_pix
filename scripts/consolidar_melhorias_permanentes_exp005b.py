"""
scripts/consolidar_melhorias_permanentes_exp005b.py

Consolida o resultado do EXP-005B-E2E.

Decisão:
  - NÃO promove LGBM v6.2.
  - NÃO promove thresholds candidatos 0.07/0.15/0.20/0.30.
  - Mantém o baseline pós-FASE 1.
  - Mantém scoring_config.json válido.
  - Mantém guard rail LGBM do EXP-002.
  - Mantém exceção contextual V1_GUARD_CONTEXTUAL do EXP-004-FINAL.
  - Gera manifesto explícito rejeitando o LGBM v6.2 para runtime.

Uso:
  python scripts\\consolidar_melhorias_permanentes_exp005b.py
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists():
            return p
    return start.parent


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)

SCORING_PATH = PROJECT_ROOT / "backend" / "artefatos" / "scoring_config.json"
ENGINE_PATH = PROJECT_ROOT / "backend" / "core" / "decision_engine.py"
CANDIDATE_DIR = PROJECT_ROOT / "backend" / "artefatos_candidatos" / "exp_005a_lgbm_v6_2_recall"
RESULT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-005B-E2E"


GUARD_FIELDS_BLOCK = """\
    # --- EXP-004-FINAL: excecao cirurgica ao guard rail LGBM ---
    guard_exception_alto_valor_se_beh_enabled: bool = False
    guard_exception_alto_valor_min: float = 15000.0
    guard_exception_alto_valor_rel_max: float = 12.0
    guard_exception_alto_valor_if_min: float = 0.985
    guard_exception_alto_valor_lgbm_min: float = 0.01
    guard_exception_alto_valor_age_min: int = 18
    guard_exception_alto_valor_age_max: int = 90
    guard_exception_alto_valor_require_first_receiver: bool = True
    guard_exception_alto_valor_require_pf: bool = True

"""


HYDRATE_SHIM = r'''

# ============================================================
# FASE 2 PATCH — Bind seguro de hidratação do scoring_config
# ============================================================
# Este bloco garante que PixDecisionEngine consiga hidratar EngineConfig
# a partir de scoring_config.json sem quebrar com chaves extras.

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

    for name in fields.keys():
        if name not in scoring:
            continue

        if name == "artefatos_dir":
            continue

        try:
            current_value = getattr(self.config, name)
            default_value = getattr(default_config, name)
        except Exception:
            continue

        # Não sobrescrever override explícito passado pelo experimento/runtime.
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


if not hasattr(PixDecisionEngine, "_coerce_config_value"):
    PixDecisionEngine._coerce_config_value = _fase2_coerce_config_value

if not hasattr(PixDecisionEngine, "_hydrate_config_from_scoring_config"):
    PixDecisionEngine._hydrate_config_from_scoring_config = _fase2_hydrate_config_from_scoring_config

'''


def backup(path: Path, suffix: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    backup_path = path.with_suffix(
        f"{path.suffix}.bak_{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(path, backup_path)
    print(f"[OK] Backup criado: {backup_path}")
    return backup_path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"JSON não é objeto dict: {path}")

    return obj


def write_json(path: Path, obj: dict[str, Any]) -> None:
    rendered = json.dumps(obj, indent=2, ensure_ascii=False)
    json.loads(rendered)
    path.write_text(rendered + "\n", encoding="utf-8")
    json.loads(path.read_text(encoding="utf-8"))


def set_thresholds_in_faixas(config: dict[str, Any]) -> None:
    faixas = config.get("faixas_decisao")

    if isinstance(faixas, dict):
        for key, value in faixas.items():
            if not isinstance(value, dict):
                continue

            k = str(key).lower()

            if "aprovar" in k:
                for field in ["max", "max_score", "score_max", "threshold_max"]:
                    if field in value:
                        value[field] = 62.0

            if "confirmar" in k:
                for field in ["min", "min_score", "score_min", "threshold", "threshold_min"]:
                    if field in value:
                        value[field] = 62.0
                for field in ["max", "max_score", "score_max", "threshold_max"]:
                    if field in value:
                        value[field] = 95.0

            if "bloquear" in k:
                for field in ["min", "min_score", "score_min", "threshold", "threshold_min"]:
                    if field in value:
                        value[field] = 95.0

    elif isinstance(faixas, list):
        for item in faixas:
            if not isinstance(item, dict):
                continue

            label = " ".join(
                str(item.get(k, ""))
                for k in ["decisao", "nome", "label", "categoria", "tipo"]
            ).lower()

            if "aprovar" in label:
                for field in ["max", "max_score", "score_max", "threshold_max"]:
                    if field in item:
                        item[field] = 62.0

            if "confirmar" in label:
                for field in ["min", "min_score", "score_min", "threshold", "threshold_min"]:
                    if field in item:
                        item[field] = 62.0
                for field in ["max", "max_score", "score_max", "threshold_max"]:
                    if field in item:
                        item[field] = 95.0

            if "bloquear" in label:
                for field in ["min", "min_score", "score_min", "threshold", "threshold_min"]:
                    if field in item:
                        item[field] = 95.0


def consolidate_scoring_config() -> None:
    backup(SCORING_PATH, "consolida_exp005b")

    config = load_json(SCORING_PATH)

    # Baseline permanente pós-FASE 1.
    config["threshold_confirmar"] = 62.0
    config["threshold_bloquear"] = 95.0

    # Guard rail aprovado no EXP-002.
    config["lgbm_guard_enabled"] = True
    config["lgbm_guard_threshold"] = 0.30

    # EXP-003 não aprovado.
    config["se_pattern_residual_enabled"] = False
    config["exp003_residual_confirm_enabled"] = False

    # EXP-004-FINAL V1: manter como exceção contextual permanente.
    config["guard_exception_alto_valor_se_beh_enabled"] = True
    config["guard_exception_alto_valor_min"] = 15000.0
    config["guard_exception_alto_valor_rel_max"] = 12.0
    config["guard_exception_alto_valor_if_min"] = 0.985
    config["guard_exception_alto_valor_lgbm_min"] = 0.01
    config["guard_exception_alto_valor_age_min"] = 18
    config["guard_exception_alto_valor_age_max"] = 90
    config["guard_exception_alto_valor_require_first_receiver"] = True
    config["guard_exception_alto_valor_require_pf"] = True

    # Não promover thresholds do EXP-005B.
    # Se existir uma chave temporária do experimento, remover.
    config.pop("_metadata_exp005b_e2e_temp", None)

    # Se lgbm_effective_threshold existir, não promovemos 0.07/0.15/0.20/0.30
    # como achado novo; mantemos valor conservador compatível com guard rail.
    if "lgbm_effective_threshold" in config:
        config["lgbm_effective_threshold"] = 0.30

    set_thresholds_in_faixas(config)

    config["_metadata_exp005b_decision"] = {
        "decided_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "EXP-005B-E2E",
        "decision": "REJEITAR_LGBM_V6_2_PARA_RUNTIME",
        "promote_lgbm_v6_2": False,
        "promote_candidate_thresholds": False,
        "keep_baseline_fase1": True,
        "reason": [
            "E2E real não reduziu FN: baseline e candidatos ficaram com FN=9.",
            "Candidatos aumentaram FP e reduziram F1.",
            "CAND_007_RECALL foi escolhido pelo seletor do experimento, mas não superou baseline.",
            "LGBM v6.2 deve permanecer apenas em artefatos_candidatos."
        ]
    }

    write_json(SCORING_PATH, config)
    print(f"[OK] scoring_config consolidado: {SCORING_PATH}")


def find_engine_config_block(lines: list[str]) -> tuple[int, int]:
    start = None

    for i, line in enumerate(lines):
        if line.startswith("class EngineConfig"):
            start = i
            break

    if start is None:
        raise RuntimeError("Não encontrei class EngineConfig.")

    end = len(lines)

    for i in range(start + 1, len(lines)):
        line = lines[i]

        if not line.strip():
            continue

        if not line.startswith((" ", "\t")) and (
            line.startswith("class ") or line.startswith("def ") or line.startswith("@")
        ):
            end = i
            break

    return start, end


def ensure_engine_config_guard_fields(text: str) -> str:
    lines = text.splitlines(keepends=True)
    start, end = find_engine_config_block(lines)
    block = "".join(lines[start:end])

    if "guard_exception_alto_valor_se_beh_enabled:" in block:
        print("[OK] EngineConfig já contém campos guard_exception.")
        return text

    insert_idx = None

    for i in range(start, end):
        if "lgbm_guard_threshold" in lines[i]:
            insert_idx = i + 1
            break

    if insert_idx is None:
        for i in range(start, end):
            if "lgbm_guard_enabled" in lines[i]:
                insert_idx = i + 1
                break

    if insert_idx is None:
        insert_idx = end

    new_lines = lines[:insert_idx] + [GUARD_FIELDS_BLOCK] + lines[insert_idx:]
    print("[OK] Campos guard_exception adicionados ao EngineConfig.")
    return "".join(new_lines)


def ensure_hydrate_shim(text: str) -> str:
    if "_fase2_hydrate_config_from_scoring_config" in text:
        print("[OK] Shim de hidratação FASE 2 já existe.")
        return text

    print("[OK] Shim de hidratação FASE 2 adicionado.")
    return text.rstrip() + "\n" + HYDRATE_SHIM + "\n"


def consolidate_decision_engine() -> None:
    backup(ENGINE_PATH, "consolida_exp005b")

    text = ENGINE_PATH.read_text(encoding="utf-8")
    text = ensure_engine_config_guard_fields(text)
    text = ensure_hydrate_shim(text)

    ENGINE_PATH.write_text(text, encoding="utf-8")
    print(f"[OK] decision_engine consolidado: {ENGINE_PATH}")


def write_promotion_manifest() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "EXP-005B-E2E",
        "decision_at": datetime.now().isoformat(timespec="seconds"),
        "promotion_decision": "KEEP_BASELINE_REJECT_LGBM_V6_2",
        "promoted": {
            "scoring_config_valid_json": True,
            "engine_config_guard_exception_fields": True,
            "scoring_config_hydration_support": True,
            "fase1_guard_exception_contextual": True
        },
        "not_promoted": {
            "lgbm_v6_2_recall_candidate": True,
            "candidate_threshold_0_07": True,
            "candidate_threshold_0_15": True,
            "candidate_threshold_0_20": True,
            "candidate_threshold_0_30": True
        },
        "final_runtime_policy": {
            "threshold_confirmar": 62.0,
            "threshold_bloquear": 95.0,
            "lgbm_guard_enabled": True,
            "lgbm_guard_threshold": 0.30,
            "se_pattern_residual_enabled": False,
            "exp003_residual_confirm_enabled": False,
            "guard_exception_alto_valor_se_beh_enabled": True
        },
        "notes": [
            "O E2E real do EXP-005B não reduziu FN.",
            "Todos os candidatos mantiveram FN=9 e aumentaram FP.",
            "O seletor do experimento indicou CAND_007_RECALL, mas a análise executiva rejeita essa promoção.",
            "Próximos experimentos devem usar E2E rápido: baseline + 1 candidato, sample pequeno, salvamento incremental."
        ]
    }

    path = RESULT_DIR / "06_decisao_promocao_permanente.json"
    write_json(path, manifest)
    print(f"[OK] Manifesto de decisão salvo: {path}")


def quarantine_accidental_candidate_file() -> None:
    """
    Segurança extra: se o arquivo candidato tiver sido copiado com o próprio nome
    para backend/artefatos por acidente, mover para quarentena.

    Não tenta restaurar modelos sobrescritos com outro nome, porque isso exige
    saber qual backup operacional deve ser usado.
    """
    accidental = PROJECT_ROOT / "backend" / "artefatos" / "lgbm_v6_2_recall_candidate.joblib"

    if not accidental.exists():
        print("[OK] Nenhum lgbm_v6_2_recall_candidate.joblib ativo em backend/artefatos.")
        return

    quarantine = CANDIDATE_DIR / "quarantine_not_promoted"
    quarantine.mkdir(parents=True, exist_ok=True)

    dst = quarantine / f"lgbm_v6_2_recall_candidate_not_promoted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    shutil.move(str(accidental), str(dst))
    print(f"[OK] Modelo candidato acidental movido para quarentena: {dst}")


def main() -> None:
    print("=== Consolidação permanente EXP-005B ===")

    consolidate_scoring_config()
    consolidate_decision_engine()
    quarantine_accidental_candidate_file()
    write_promotion_manifest()

    print()
    print("[OK] Consolidação concluída.")
    print("[INFO] LGBM v6.2 NÃO foi promovido.")
    print("[INFO] Baseline pós-FASE 1 preservado.")
    print("[INFO] Execute as validações recomendadas em seguida.")


if __name__ == "__main__":
    main()