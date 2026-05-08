"""
scripts/patch_scoring_config_fase2.py

Reparo cirúrgico do backend/artefatos/scoring_config.json.

Corrige o caso atual:
  - Um bloco órfão { "guard_exception_alto_valor_se_beh_enabled": ... }
    foi colado dentro do JSON principal.
  - O JSON quebra na linha 128.
  - Este script remove esse bloco por linhas e reinsere as chaves no objeto principal.

Não promove LGBM v6.2.
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


def strip_json_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    lines = []
    for line in text.splitlines():
        in_string = False
        escaped = False
        cut_at = None

        for i in range(len(line) - 1):
            ch = line[i]

            if escaped:
                escaped = False
                continue

            if ch == "\\":
                escaped = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if not in_string and line[i:i + 2] == "//":
                cut_at = i
                break

        lines.append(line if cut_at is None else line[:cut_at])

    return "\n".join(lines)


def remove_trailing_commas(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def sanitize_json_text(text: str) -> str:
    text = text.lstrip("\ufeff")
    text = strip_json_comments(text)
    text = remove_trailing_commas(text)
    return text


def remove_orphan_guard_block_by_lines(text: str) -> tuple[str, dict[str, Any]]:
    """
    Remove bloco órfão guard_exception por linhas.

    Espera algo parecido com:

      {
        "guard_exception_alto_valor_se_beh_enabled": true,
        ...
      }

    Retorna:
      texto sem o bloco, objeto com as chaves removidas.
    """
    lines = text.splitlines()

    guard_line_idx = None
    for i, line in enumerate(lines):
        if '"guard_exception_alto_valor_se_beh_enabled"' in line:
            guard_line_idx = i
            break

    if guard_line_idx is None:
        return text, {}

    # Volta até a linha que abre o objeto órfão.
    start_idx = None
    for i in range(guard_line_idx, -1, -1):
        if lines[i].strip() == "{":
            start_idx = i
            break

    if start_idx is None:
        raise RuntimeError("Encontrei guard_exception, mas não encontrei a linha '{' que abre o bloco órfão.")

    # Avança até o primeiro fechamento simples do bloco órfão.
    # Este bloco não deveria ter objetos aninhados.
    end_idx = None
    for i in range(guard_line_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped in {"}", "},"}:
            end_idx = i
            break

    if end_idx is None:
        raise RuntimeError("Encontrei abertura do bloco órfão, mas não encontrei fechamento '}'.")

    orphan_text = "\n".join(lines[start_idx:end_idx + 1])
    orphan_text_clean = sanitize_json_text(orphan_text).strip()

    # O bloco órfão pode ter sido capturado como:
    #   {
    #     ...
    #   },
    # Essa vírgula final é válida dentro de um objeto maior,
    # mas inválida ao parsear o bloco isoladamente.
    if orphan_text_clean.endswith(","):
        orphan_text_clean = orphan_text_clean[:-1].rstrip()

    try:
        orphan_obj = json.loads(orphan_text_clean)
    except Exception as exc:
        print("[ERRO] Falha ao ler o bloco órfão capturado.")
        print("Bloco capturado:")
        print(orphan_text)
        raise RuntimeError(f"Falha ao parsear bloco órfão: {exc}") from exc

    if not isinstance(orphan_obj, dict):
        raise RuntimeError("Bloco órfão não resultou em objeto JSON dict.")

    # Ao remover o bloco órfão, também precisamos fechar o objeto pai
    # onde ele foi colado indevidamente. No caso atual, ele entrou logo
    # depois do item "BLOQUEAR" dentro de "faixas_decisao".
    #
    # Tentamos gerar duas versões:
    #   1. sem inserir fechamento extra;
    #   2. inserindo "  }," no ponto onde o bloco órfão estava.
    #
    # A função retorna a versão que virar JSON válido.
    tail_lines = lines[end_idx + 1:]

    candidate_versions = []

    # Versão A: remoção pura.
    candidate_versions.append(lines[:start_idx] + tail_lines)

    # Versão B: remoção + fechamento do objeto pai.
    candidate_versions.append(lines[:start_idx] + ["  },"] + tail_lines)

    last_error = None

    for candidate_lines in candidate_versions:
        candidate_text = "\n".join(candidate_lines)
        candidate_text = sanitize_json_text(candidate_text)

        try:
            json.loads(candidate_text)
            return candidate_text, orphan_obj
        except json.JSONDecodeError as exc:
            last_error = exc

    raise RuntimeError(
        "Removi o bloco órfão, mas nenhuma versão do JSON ficou válida. "
        f"Último erro: {last_error}"
    )

    


def load_scoring_config_repaired(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")

    diagnostics = {
        "loaded_mode": None,
        "repairs": [],
        "orphan_keys_merged": [],
    }

    # 1. Tenta direto.
    try:
        obj = json.loads(raw)
        diagnostics["loaded_mode"] = "raw"
        return obj, diagnostics
    except json.JSONDecodeError as exc:
        diagnostics["raw_error"] = str(exc)

    # 2. Tenta sanitização simples.
    sanitized = sanitize_json_text(raw)
    try:
        obj = json.loads(sanitized)
        diagnostics["loaded_mode"] = "sanitized"
        diagnostics["repairs"].append("comments_or_trailing_commas")
        return obj, diagnostics
    except json.JSONDecodeError as exc:
        diagnostics["sanitized_error"] = str(exc)

    # 3. Remove bloco órfão por linhas.
    repaired_text, orphan_obj = remove_orphan_guard_block_by_lines(raw)

    try:
        base_obj = json.loads(repaired_text)
    except json.JSONDecodeError as exc:
        lines = repaired_text.splitlines()
        line_no = exc.lineno
        start = max(line_no - 5, 0)
        end = min(line_no + 5, len(lines))

        print("[ERRO] Mesmo após remover bloco órfão, JSON ainda inválido.")
        print(f"Erro: {exc}")
        print("\nContexto do arquivo reparado:")
        for i in range(start, end):
            marker = ">>" if (i + 1) == line_no else "  "
            print(f"{marker} {i + 1:04d}: {lines[i]}")

        raise

    if not isinstance(base_obj, dict):
        raise RuntimeError("JSON base reparado não é objeto dict.")

    base_obj.update(orphan_obj)

    diagnostics["loaded_mode"] = "line_based_orphan_guard_block_removed"
    diagnostics["repairs"].append("orphan_guard_exception_block_removed_by_lines")
    diagnostics["orphan_keys_merged"] = sorted(orphan_obj.keys())

    return base_obj, diagnostics


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


def apply_permanent_fase2_patch(config: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    config["threshold_confirmar"] = 62.0
    config["threshold_bloquear"] = 95.0

    config["lgbm_guard_enabled"] = True
    config["lgbm_guard_threshold"] = 0.30

    config["se_pattern_residual_enabled"] = False
    config["exp003_residual_confirm_enabled"] = False

    config["guard_exception_alto_valor_se_beh_enabled"] = True
    config["guard_exception_alto_valor_min"] = 15000.0
    config["guard_exception_alto_valor_rel_max"] = 12.0
    config["guard_exception_alto_valor_if_min"] = 0.985
    config["guard_exception_alto_valor_lgbm_min"] = 0.01
    config["guard_exception_alto_valor_age_min"] = 18
    config["guard_exception_alto_valor_age_max"] = 90
    config["guard_exception_alto_valor_require_first_receiver"] = True
    config["guard_exception_alto_valor_require_pf"] = True

    set_thresholds_in_faixas(config)

    config["_metadata_fase2_patch"] = {
        "patched_at": datetime.now().isoformat(timespec="seconds"),
        "patch": "FASE2_BASELINE_PERMANENTE",
        "diagnostics": diagnostics,
        "notes": [
            "JSON validado e reformatado.",
            "Bloco órfão guard_exception, se existia, foi removido por linhas e mesclado ao objeto principal.",
            "Promovido baseline pós-FASE 1.",
            "LGBM v6.2 ainda não promovido; será testado temporariamente no EXP-005B-E2E."
        ]
    }

    return config


def main() -> None:
    if not SCORING_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {SCORING_PATH}")

    backup_path = SCORING_PATH.with_suffix(
        f".json.bak_fase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(SCORING_PATH, backup_path)
    print(f"[OK] Backup criado: {backup_path}")

    config, diagnostics = load_scoring_config_repaired(SCORING_PATH)
    config = apply_permanent_fase2_patch(config, diagnostics)

    rendered = json.dumps(config, indent=2, ensure_ascii=False)
    json.loads(rendered)

    SCORING_PATH.write_text(rendered + "\n", encoding="utf-8")

    json.loads(SCORING_PATH.read_text(encoding="utf-8"))

    print(f"[OK] scoring_config.json corrigido e atualizado: {SCORING_PATH}")
    print("[OK] JSON válido.")
    print(f"[INFO] Modo de carregamento: {diagnostics.get('loaded_mode')}")
    print(f"[INFO] Reparos aplicados: {diagnostics.get('repairs')}")
    print(f"[INFO] Chaves órfãs mescladas: {diagnostics.get('orphan_keys_merged')}")
    print("[INFO] LGBM v6.2 NÃO foi promovido permanentemente neste patch.")


if __name__ == "__main__":
    main()