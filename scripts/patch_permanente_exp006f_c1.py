"""
scripts/patch_permanente_exp006f_c1.py

Promove a regra C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER aprovada no EXP-006F.

O que faz:
  1. Faz backup de decision_engine.py e scoring_config.json.
  2. Adiciona campos configuráveis ao EngineConfig.
  3. Adiciona método _apply_exp006f_c1_near_threshold_exception ao PixDecisionEngine.
  4. Injeta chamada segura antes do retorno final de decisão, se encontrar ponto compatível.
  5. Atualiza scoring_config.json com C1 habilitada.

Regra C1:
  decisao == APROVAR
  first_receiver_flag == 1
  pix_key_random_flag == 0
  qt_tempo_relacionamento_mes <= 12
  100 <= vl_pix < 500
  0.06 <= lgbm_raw < 0.10
  60 <= score_final < 62
  se_score <= 0
  beh_score <= 0

Decisão:
  APROVAR -> CONFIRMAR

Uso:
  python scripts\\patch_permanente_exp006f_c1.py
  python -m py_compile backend\\core\\decision_engine.py
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

ENGINE_PATH = PROJECT_ROOT / "backend" / "core" / "decision_engine.py"
SCORING_PATH = PROJECT_ROOT / "backend" / "artefatos" / "scoring_config.json"


ENGINE_CONFIG_FIELDS = """\
    # --- EXP-006F-C1: near-threshold residual FN recovery ---
    exp006f_c1_enabled: bool = True
    exp006f_c1_min_score: float = 60.0
    exp006f_c1_max_score: float = 62.0
    exp006f_c1_min_valor: float = 100.0
    exp006f_c1_max_valor: float = 500.0
    exp006f_c1_max_rel_meses: float = 12.0
    exp006f_c1_min_lgbm_raw: float = 0.06
    exp006f_c1_max_lgbm_raw: float = 0.10
    exp006f_c1_require_first_receiver: bool = True
    exp006f_c1_require_not_pix_random: bool = True
    exp006f_c1_max_se_score: float = 0.0
    exp006f_c1_max_beh_score: float = 0.0

"""


METHOD_BLOCK = r'''
    def _apply_exp006f_c1_near_threshold_exception(self, tx: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """
        EXP-006F-C1 — exceção cirúrgica para recuperar FN residual near-threshold.

        Promove APROVAR -> CONFIRMAR quando:
          - C1 está habilitada;
          - decisão atual é APROVAR;
          - score_final em [60, 62);
          - first_receiver=1;
          - pix_key_random=0;
          - relacionamento <= 12 meses;
          - 100 <= vl_pix < 500;
          - 0.06 <= lgbm_raw < 0.10;
          - se_score <= 0;
          - beh_score <= 0.

        Evidência EXP-006F:
          - seed 42: FN 9 -> 8, FP 14 -> 14;
          - seed 123: FN 9 -> 8, FP 12 -> 12;
          - 0 FP adicionado, 0 TP perdido.
        """
        try:
            if not getattr(self.config, "exp006f_c1_enabled", False):
                return result

            decisao = str(result.get("decisao", "")).upper()
            if decisao != "APROVAR":
                return result

            def _num(obj: Dict[str, Any], key: str, default: float = 0.0) -> float:
                try:
                    value = obj.get(key, default)
                    if value is None:
                        return default
                    return float(value)
                except Exception:
                    return default

            def _num_any(keys, default: float = 0.0) -> float:
                for key in keys:
                    if key in result:
                        return _num(result, key, default)
                    if key in tx:
                        return _num(tx, key, default)
                return default

            vl_pix = _num_any(["vl_pix", "valor", "amount"], 0.0)
            rel_meses = _num_any(["qt_tempo_relacionamento_mes", "relacionamento_meses"], 999.0)
            first_receiver = int(_num_any(["first_receiver_flag"], 0.0))
            pix_random = int(_num_any(["pix_key_random_flag"], 0.0))
            lgbm_raw = _num_any(["lgbm_raw", "lgbm_score", "score_lgbm_raw"], 0.0)
            se_score = _num_any(["se_score", "social_engineering_score"], 0.0)
            beh_score = _num_any(["beh_score", "behavioral_score"], 0.0)
            score_final = _num_any(["score_final", "score"], 0.0)

            if getattr(self.config, "exp006f_c1_require_first_receiver", True) and first_receiver != 1:
                return result

            if getattr(self.config, "exp006f_c1_require_not_pix_random", True) and pix_random != 0:
                return result

            if not (
                getattr(self.config, "exp006f_c1_min_valor", 100.0)
                <= vl_pix
                < getattr(self.config, "exp006f_c1_max_valor", 500.0)
            ):
                return result

            if rel_meses > getattr(self.config, "exp006f_c1_max_rel_meses", 12.0):
                return result

            if not (
                getattr(self.config, "exp006f_c1_min_lgbm_raw", 0.06)
                <= lgbm_raw
                < getattr(self.config, "exp006f_c1_max_lgbm_raw", 0.10)
            ):
                return result

            if not (
                getattr(self.config, "exp006f_c1_min_score", 60.0)
                <= score_final
                < getattr(self.config, "exp006f_c1_max_score", 62.0)
            ):
                return result

            if se_score > getattr(self.config, "exp006f_c1_max_se_score", 0.0):
                return result

            if beh_score > getattr(self.config, "exp006f_c1_max_beh_score", 0.0):
                return result

            result = dict(result)
            result["decisao_original_exp006f_c1"] = result.get("decisao")
            result["score_final_original_exp006f_c1"] = score_final
            result["decisao"] = "CONFIRMAR"
            result["score_final"] = max(score_final, 62.0)
            result["exp006f_c1_applied"] = True
            result["exp006f_c1_reason"] = (
                "C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER: APROVAR->CONFIRMAR | "
                "rel<=12, first_receiver=1, pix_random=0, 100<=vl<500, "
                "0.06<=lgbm<0.10, 60<=score<62, SE=0, BEH=0"
            )

            return result

        except Exception as exc:
            try:
                logger.warning("Falha ao aplicar EXP-006F-C1: %s", exc)
            except Exception:
                pass
            return result

'''


SHIM_BLOCK = r'''

# ============================================================
# EXP-006F-C1 PATCH — Bind defensivo da exceção near-threshold
# ============================================================

def _exp006f_c1_bound_method(self, tx, result):
    return PixDecisionEngine._apply_exp006f_c1_near_threshold_exception(self, tx, result)

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


def find_engine_config_block(lines: list[str]) -> tuple[int, int]:
    start = None

    for i, line in enumerate(lines):
        if line.startswith("class EngineConfig"):
            start = i
            break

    if start is None:
        raise RuntimeError("Não encontrei class EngineConfig em decision_engine.py")

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


def insert_engine_config_fields(text: str) -> str:
    if "exp006f_c1_enabled:" in text:
        print("[OK] Campos EXP-006F-C1 já existem no EngineConfig.")
        return text

    lines = text.splitlines(keepends=True)
    start, end = find_engine_config_block(lines)

    insert_idx = None

    # Inserir depois de campos EXP-004, se existirem.
    for i in range(start, end):
        if "guard_exception_alto_valor_require_pf" in lines[i]:
            insert_idx = i + 1
            break

    # Fallback: depois de lgbm_guard_threshold.
    if insert_idx is None:
        for i in range(start, end):
            if "lgbm_guard_threshold" in lines[i]:
                insert_idx = i + 1
                break

    if insert_idx is None:
        insert_idx = end

    new_lines = lines[:insert_idx] + [ENGINE_CONFIG_FIELDS] + lines[insert_idx:]
    print("[OK] Campos EXP-006F-C1 adicionados ao EngineConfig.")
    return "".join(new_lines)


def insert_method(text: str) -> str:
    if "def _apply_exp006f_c1_near_threshold_exception" in text:
        print("[OK] Método EXP-006F-C1 já existe.")
        return text

    marker_candidates = [
        "\n    def _load_all(self)",
        "\n    def _hydrate_config_from_scoring_config",
        "\n    def _coerce_config_value",
    ]

    marker = None
    for m in marker_candidates:
        if m in text:
            marker = m
            break

    if marker is None:
        raise RuntimeError("Não encontrei ponto seguro para inserir método EXP-006F-C1.")

    patched = text.replace(marker, METHOD_BLOCK + marker, 1)
    print("[OK] Método EXP-006F-C1 inserido.")
    return patched


def inject_call(text: str) -> str:
    """
    Tenta injetar a chamada ao C1 antes dos returns finais mais comuns.

    Como não sabemos exatamente a estrutura local do seu decision_engine.py,
    o patch procura padrões conservadores. Se não encontrar, imprime instrução manual.
    """
    if "_apply_exp006f_c1_near_threshold_exception(tx, result)" in text:
        print("[OK] Chamada EXP-006F-C1 já existe.")
        return text

    patterns = [
        (
            "return result\n",
            "result = self._apply_exp006f_c1_near_threshold_exception(tx, result)\n        return result\n",
        ),
        (
            "return resultado\n",
            "resultado = self._apply_exp006f_c1_near_threshold_exception(tx, resultado)\n        return resultado\n",
        ),
    ]

    for old, new in patterns:
        count = text.count(old)
        if count == 1:
            print("[OK] Chamada EXP-006F-C1 injetada antes do return final.")
            return text.replace(old, new, 1)

    print("[AVISO] Não consegui injetar automaticamente a chamada C1 com segurança.")
    print("[AVISO] O método e os campos foram adicionados, mas talvez você precise inserir manualmente:")
    print("        result = self._apply_exp006f_c1_near_threshold_exception(tx, result)")
    print("        return result")
    return text


def patch_decision_engine() -> None:
    backup(ENGINE_PATH, "exp006f_c1")

    text = ENGINE_PATH.read_text(encoding="utf-8")
    text = insert_engine_config_fields(text)
    text = insert_method(text)
    text = inject_call(text)

    ENGINE_PATH.write_text(text, encoding="utf-8")
    print(f"[OK] decision_engine atualizado: {ENGINE_PATH}")


def patch_scoring_config() -> None:
    backup(SCORING_PATH, "exp006f_c1")

    with SCORING_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)

    config["exp006f_c1_enabled"] = True
    config["exp006f_c1_min_score"] = 60.0
    config["exp006f_c1_max_score"] = 62.0
    config["exp006f_c1_min_valor"] = 100.0
    config["exp006f_c1_max_valor"] = 500.0
    config["exp006f_c1_max_rel_meses"] = 12.0
    config["exp006f_c1_min_lgbm_raw"] = 0.06
    config["exp006f_c1_max_lgbm_raw"] = 0.10
    config["exp006f_c1_require_first_receiver"] = True
    config["exp006f_c1_require_not_pix_random"] = True
    config["exp006f_c1_max_se_score"] = 0.0
    config["exp006f_c1_max_beh_score"] = 0.0

    config["_metadata_exp006f_c1"] = {
        "patched_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "EXP-006F-C1",
        "decision": "PROMOTED_PERMANENT_CONFIGURABLE",
        "evidence": {
            "seed_42": {
                "baseline": {"TP": 346, "FP": 14, "FN": 9, "F1": 0.9678},
                "c1": {"TP": 347, "FP": 14, "FN": 8, "F1": 0.9693},
            },
            "seed_123": {
                "baseline": {"TP": 346, "FP": 12, "FN": 9, "F1": 0.9705},
                "c1": {"TP": 347, "FP": 12, "FN": 8, "F1": 0.9720},
            },
        },
        "notes": [
            "C1 recuperou 1 FN nos dois seeds.",
            "C1 adicionou 0 FP e perdeu 0 TP.",
            "Regra mantida configurável/desligável por exp006f_c1_enabled."
        ],
    }

    rendered = json.dumps(config, indent=2, ensure_ascii=False)
    json.loads(rendered)

    SCORING_PATH.write_text(rendered + "\n", encoding="utf-8")
    print(f"[OK] scoring_config atualizado: {SCORING_PATH}")


def main() -> None:
    print("=== Patch permanente EXP-006F-C1 ===")
    patch_decision_engine()
    patch_scoring_config()
    print()
    print("[OK] Patch permanente EXP-006F-C1 concluído.")
    print("[INFO] Próximas validações:")
    print("  python -m py_compile backend\\core\\decision_engine.py")
    print("  python -c \"import json; json.load(open(r'backend\\artefatos\\scoring_config.json', encoding='utf-8')); print('scoring_config OK')\"")


if __name__ == "__main__":
    main()