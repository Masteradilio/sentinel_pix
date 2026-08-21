"""
audit_logger.py — Auditoria e Fila de Investigação de Fraudes (Mesa de Fraude)
Registra automaticamente todas as transações classificadas como CONFIRMAR e BLOQUEAR
para posterior análise humana, feedback loop e governança regulatória.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import settings

logger = logging.getLogger("audit_logger")


class AuditLogger:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            url = settings.audit_db_url
            if url.startswith("sqlite:///"):
                self.db_path = url.replace("sqlite:///", "")
            else:
                self.db_path = str(settings.project_root / "backend" / "feature_store" / "fraud_investigation_cases.db")
        else:
            self.db_path = db_path

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fraud_investigation_cases (
                    case_id TEXT PRIMARY KEY,
                    transaction_id TEXT,
                    account_id TEXT,
                    receiver_pix_key TEXT,
                    amount REAL,
                    decisao TEXT,
                    score_final REAL,
                    confianca TEXT,
                    motivo_principal TEXT,
                    regras_acionadas TEXT,
                    se_patterns TEXT,
                    beh_factors TEXT,
                    shap_top_features TEXT,
                    status_investigacao TEXT DEFAULT 'PENDING',
                    analyst_notes TEXT DEFAULT '',
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            conn.commit()
        logger.info(f"Fila de Investigação inicializada em: {self.db_path}")

    def log_decision(self, result: Dict[str, Any], raw_tx: Dict[str, Any]) -> Optional[str]:
        """Registra a transação se a decisão exigir intervenção (CONFIRMAR ou BLOQUEAR)."""
        decisao = result.get("decisao", "APROVAR").upper()
        if decisao not in ("CONFIRMAR", "BLOQUEAR"):
            return None

        case_id = f"case_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow().isoformat() + "Z"

        # Extrair explicabilidade e componentes
        explicabilidade = result.get("explicabilidade", {})
        motivo = explicabilidade.get("motivo_principal", "Risco detectado pelo motor híbrido")
        comp = explicabilidade.get("componentes", {})
        
        se_patterns = json.dumps(comp.get("se_patterns", []))
        beh_factors = json.dumps(comp.get("beh_factors", []))
        
        # Regras / vetos / cascade
        regras = []
        if result.get("veto_aplicado"):
            regras.append(f"Veto: {result.get('veto_aplicado')}")
        if result.get("cascade", {}).get("triggered"):
            regras.extend(result.get("cascade", {}).get("rules", []))
        if result.get("r5b22_rule_applied"):
            regras.append(f"R5B22: {result.get('r5b22_rule_applied')}")
        
        regras_json = json.dumps(regras)
        shap_json = json.dumps(explicabilidade.get("shap_top_features", {}))

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO fraud_investigation_cases (
                    case_id, transaction_id, account_id, receiver_pix_key,
                    amount, decisao, score_final, confianca, motivo_principal,
                    regras_acionadas, se_patterns, beh_factors, shap_top_features,
                    status_investigacao, analyst_notes, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                case_id,
                str(raw_tx.get("transaction_id", raw_tx.get("id_transacao", ""))),
                str(raw_tx.get("account_id", raw_tx.get("id_cliente", ""))),
                str(raw_tx.get("receiver_pix_key", raw_tx.get("chave_pix", ""))),
                float(raw_tx.get("amount", raw_tx.get("vl_transacao", 0.0))),
                decisao,
                float(result.get("score_final", 0.0)),
                str(result.get("confianca", "MEDIA")),
                motivo,
                regras_json,
                se_patterns,
                beh_factors,
                shap_json,
                "PENDING",
                "",
                now,
                now
            ))
            conn.commit()

        logger.info(f"Caso de fraude registrado: {case_id} ({decisao} - R$ {raw_tx.get('amount')})")
        return case_id

    def list_cases(self, limit: int = 100, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lista os casos para exibição na Mesa de Fraude / Dashboard."""
        query = "SELECT * FROM fraud_investigation_cases"
        params: List[Any] = []
        if status:
            query += " WHERE status_investigacao = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def update_case_status(self, case_id: str, new_status: str, notes: str = "") -> bool:
        """Permite que o analista aprove, confirme fraude ou arquive o caso."""
        now = datetime.utcnow().isoformat() + "Z"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE fraud_investigation_cases
                SET status_investigacao = ?, analyst_notes = ?, updated_at = ?
                WHERE case_id = ?
            """, (new_status, notes, now, case_id))
            conn.commit()
            return cursor.rowcount > 0


audit_logger = AuditLogger()
