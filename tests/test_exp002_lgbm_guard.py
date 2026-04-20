from __future__ import annotations

import json
from pathlib import Path
import unittest

from backend.core.decision_engine import CascadeResult, EngineConfig, PixDecisionEngine


def _write_json(path: Path, payload: dict | list[str]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class TestExp002LgbmGuard(unittest.TestCase):
    def test_guard_suppresses_se_behavioral_block_veto(self) -> None:
        with TemporaryArtifactsDir(Path(self._testMethodName)) as artefatos_dir:
            engine = self._build_engine(artefatos_dir, lgbm_guard_enabled=True)

            score_final, veto_desc, veto_suppressed_reason = engine._aplicar_veto(
                score_mapped=50.0,
                lgbm_raw=0.10,
                if_score=0.0,
                if_active=False,
                cascade_results=[],
                se_score=60.0,
                behavioral_score=25.0,
            )

            self.assertEqual(score_final, 50.0)
            self.assertIsNone(veto_desc)
            self.assertIsNotNone(veto_suppressed_reason)
            self.assertIn("LGBM_GUARD_RAIL", veto_suppressed_reason)

    def test_guard_disabled_preserves_same_veto(self) -> None:
        with TemporaryArtifactsDir(Path(self._testMethodName)) as artefatos_dir:
            engine = self._build_engine(artefatos_dir, lgbm_guard_enabled=False)

            score_final, veto_desc, veto_suppressed_reason = engine._aplicar_veto(
                score_mapped=50.0,
                lgbm_raw=0.10,
                if_score=0.0,
                if_active=False,
                cascade_results=[],
                se_score=60.0,
                behavioral_score=25.0,
            )

            self.assertEqual(score_final, 95.0)
            self.assertIn("VETO BLOQUEAR", veto_desc or "")
            self.assertIsNone(veto_suppressed_reason)

    def test_guard_suppresses_cascade_confirmar(self) -> None:
        with TemporaryArtifactsDir(Path(self._testMethodName)) as artefatos_dir:
            engine = self._build_engine(artefatos_dir, lgbm_guard_enabled=True)

            score_final, veto_desc, veto_suppressed_reason = engine._aplicar_veto(
                score_mapped=40.0,
                lgbm_raw=0.10,
                if_score=0.0,
                if_active=False,
                cascade_results=[
                    CascadeResult(
                        triggered=True,
                        rule_id="C3",
                        rule_name="C3",
                        action="CONFIRMAR",
                    )
                ],
                se_score=0.0,
                behavioral_score=0.0,
            )

            self.assertEqual(score_final, 40.0)
            self.assertIsNone(veto_desc)
            self.assertIn("LGBM_GUARD_RAIL", veto_suppressed_reason or "")

    def test_guard_does_not_suppress_cascade_bloquear(self) -> None:
        with TemporaryArtifactsDir(Path(self._testMethodName)) as artefatos_dir:
            engine = self._build_engine(artefatos_dir, lgbm_guard_enabled=True)

            score_final, veto_desc, veto_suppressed_reason = engine._aplicar_veto(
                score_mapped=40.0,
                lgbm_raw=0.10,
                if_score=0.0,
                if_active=False,
                cascade_results=[
                    CascadeResult(
                        triggered=True,
                        rule_id="C1",
                        rule_name="C1",
                        action="BLOQUEAR",
                    )
                ],
                se_score=0.0,
                behavioral_score=0.0,
            )

            self.assertEqual(score_final, 95.0)
            self.assertIn("VETO BLOQUEAR", veto_desc or "")
            self.assertIsNone(veto_suppressed_reason)

    @staticmethod
    def _build_engine(
        artefatos_dir: Path,
        *,
        lgbm_guard_enabled: bool,
        lgbm_guard_threshold: float = 0.30,
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
                "lgbm_guard_threshold": lgbm_guard_threshold,
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
