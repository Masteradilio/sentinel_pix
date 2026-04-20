from __future__ import annotations

import json
from pathlib import Path
import unittest

from backend.core.decision_engine import EngineConfig, PixDecisionEngine
from backend.core.social_engineering import SocialEngineeringDetector


def _write_json(path: Path, payload: dict | list[str]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class TestExp003ResidualPattern(unittest.TestCase):
    def test_se_pattern_disabled_does_not_trigger(self) -> None:
        detector = SocialEngineeringDetector(
            pattern_config={"se_pattern_residual_enabled": False}
        )

        result = detector.detect_from_pipeline(_matching_features())

        self.assertNotIn(
            "IDOSO_JOVEM_VALOR_MODERADO_RESIDUAL",
            [p.pattern_name for p in result.patterns],
        )

    def test_se_pattern_enabled_triggers_for_residual_cluster(self) -> None:
        detector = SocialEngineeringDetector(
            pattern_config={"se_pattern_residual_enabled": True}
        )

        result = detector.detect_from_pipeline(_matching_features())

        pattern_names = [p.pattern_name for p in result.patterns]
        self.assertIn("IDOSO_JOVEM_VALOR_MODERADO_RESIDUAL", pattern_names)
        self.assertGreater(result.se_score, 0.0)

    def test_engine_confirms_residual_pattern(self) -> None:
        with TemporaryArtifactsDir(Path(self._testMethodName)) as artefatos_dir:
            engine = _build_engine(artefatos_dir, lgbm_guard_enabled=False)

            score_final, veto_desc, veto_suppressed_reason = engine._aplicar_veto(
                score_mapped=55.89,
                lgbm_raw=0.0532,
                if_score=0.9179,
                if_active=True,
                cascade_results=[],
                se_score=25.0,
                se_patterns=["IDOSO_JOVEM_VALOR_MODERADO_RESIDUAL"],
                behavioral_score=0.0,
                features={
                    "first_receiver_flag": 1,
                    "vl_pix": 1650.0,
                    "qt_tempo_relacionamento_mes": 9,
                },
            )

            self.assertEqual(score_final, 62.0)
            self.assertIn("RESIDUAL", veto_desc or "")
            self.assertIsNone(veto_suppressed_reason)

    def test_engine_respects_lgbm_guard_for_residual_pattern(self) -> None:
        with TemporaryArtifactsDir(Path(self._testMethodName)) as artefatos_dir:
            engine = _build_engine(artefatos_dir, lgbm_guard_enabled=True)

            score_final, veto_desc, veto_suppressed_reason = engine._aplicar_veto(
                score_mapped=55.89,
                lgbm_raw=0.0532,
                if_score=0.9179,
                if_active=True,
                cascade_results=[],
                se_score=25.0,
                se_patterns=["IDOSO_JOVEM_VALOR_MODERADO_RESIDUAL"],
                behavioral_score=0.0,
                features={
                    "first_receiver_flag": 1,
                    "vl_pix": 1650.0,
                    "qt_tempo_relacionamento_mes": 9,
                },
            )

            self.assertEqual(score_final, 55.89)
            self.assertIsNone(veto_desc)
            self.assertIn("LGBM_GUARD_RAIL", veto_suppressed_reason or "")


def _matching_features() -> dict:
    return {
        "nr_idade": 64,
        "vl_pix": 1650.0,
        "qt_tempo_relacionamento_mes": 9,
        "first_receiver_flag": 1,
        "if_percentile": 0.9179,
        "tx_count_prev_30m": 0,
        "burst_30m_flag": 0,
        "qt_pix_dia_maximo_trimestre": 1,
        "is_first_tx_trimestre": 1,
        "ratio_valor_mediana": 3.2,
        "pix_key_random_flag": 0,
        "qt_envio_recebedor_trimestre": 0,
        "distinct_receivers_so_far": 1,
        "pix_over_100pct_renda_flag": 0,
        "pix_over_50pct_renda_flag": 0,
        "renda_missing_flag": 0,
        "is_segmento_premium_flag": 0,
        "perfil_vulneravel_se_flag": 0,
        "is_login_senha_flag": 0,
        "is_agendamento_recorrente_flag": 0,
        "hour": 16,
        "day_of_week": 2,
        "qt_intervalo_transacao_minuto": 120.0,
    }


def _build_engine(
    artefatos_dir: Path,
    *,
    lgbm_guard_enabled: bool,
) -> PixDecisionEngine:
    _write_json(artefatos_dir / "lgbm_features.json", ["feature_a"])
    _write_json(
        artefatos_dir / "scoring_config.json",
        {
            "mapeamento": {
                "anchors_raw": [0.0, 1.0],
                "anchors_out": [0.0, 100.0],
            },
                "faixas_decisao": {
                    "aprovar": {"threshold": 62.0},
                    "confirmar": {"threshold": 62.0},
                    "bloquear": {"threshold": 95.0},
                },
                "lgbm_guard_enabled": lgbm_guard_enabled,
                "lgbm_guard_threshold": 0.30,
                "exp003_residual_confirm_enabled": True,
            },
        )
    return PixDecisionEngine(EngineConfig(artefatos_dir=str(artefatos_dir)))


class TemporaryArtifactsDir:
    def __init__(self, base_name: Path) -> None:
        self.root = Path("tests") / ".tmp" / base_name

    def __enter__(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def __exit__(self, exc_type, exc, tb) -> None:
        for child in sorted(self.root.glob("**/*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        if self.root.exists():
            self.root.rmdir()
        parent = self.root.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
