import os
import csv
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from pathlib import Path

logger = logging.getLogger(__name__)

class GraphInvestigationEngine:
    """
    Engine de Investigação de Grafos Pós-Decisão.
    
    Analisa padrões topológicos localmente (ex: contas mulas, contas ponte e fanout)
    sem comprometer a latência da API. O processamento é ideal para transações 
    intervencionadas (CONFIRMAR ou BLOQUEAR) e produz um CSV investigativo incremental.
    """

    def __init__(self):
        # Acesso opcional configurado via variável de ambiente
        self.enabled = os.environ.get("GRAPH_INVESTIGATION_ENABLED", "false").lower() in ("true", "1", "yes")
        
        # Path do relatorio CSV
        report_path_env = os.environ.get("GRAPH_INVESTIGATION_REPORT_PATH")
        if report_path_env:
            self.report_path = Path(report_path_env)
        else:
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.report_path = base_dir / "resultados" / "investigacao" / "graph_investigation_report.csv"
            
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_lock = threading.Lock()
        
        # Memória local por nó (CPF/ID)
        # defaultdict(lambda: deque(maxlen=1000))
        # Para limitar o uso de memória no processo local mantemos os eventos recentes
        self.payer_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.receiver_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        if not self.enabled:
            return
            
        header = [
            "transaction_id", "timestamp", "payer_id", "receiver_id", "vl_pix",
            "decisao", "score_final", "r5b14_rule", "r5b22_rule",
            "graph_in_degree_receiver_1h", "graph_in_degree_receiver_24h",
            "graph_out_degree_payer_24h", "graph_unique_payers_to_receiver_24h",
            "graph_unique_receivers_from_payer_24h", "graph_receiver_total_amount_24h",
            "graph_payer_total_amount_24h", "graph_receiver_first_seen_age_hours",
            "graph_is_new_receiver", "graph_suspected_mule_score", 
            "graph_bridge_account_score", "graph_fanout_score"
        ]
        
        with self._csv_lock:
            # Check if file exists and is empty
            needs_header = not self.report_path.exists() or self.report_path.stat().st_size == 0
            if needs_header:
                try:
                    with open(self.report_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                except Exception as e:
                    logger.warning(f"Erro ao inicializar CSV header Graph Investigation: {e}")

    def process_transaction(self, tx_data: Dict[str, Any], result_data: Dict[str, Any]) -> None:
        """
        Processa e grava assincronamente a transação se aplicável.
        """
        if not self.enabled:
            return

        decisao = result_data.get("decisao", "")
        # Opcional: Analisar somente o que passou de APROVAR para poupar recurso local
        if decisao == "APROVAR":
            return

        # Para não travar a API, podemos delegar para lógica isolada dentro deste método,
        # ou se o framework for async, chamar com create_task. 
        # Neste caso o módulo em si opera de forma síncrona, mas com traps de exceção fortes.
        try:
            self._evaluate_and_log(tx_data, result_data)
        except Exception as e:
            logger.warning(f"Erro silenciado no Graph Investigation Engine: {e}", exc_info=True)

    def _evaluate_and_log(self, tx: Dict[str, Any], res: Dict[str, Any]) -> None:
        # 1. Parsing robusto
        tx_id = tx.get("transaction_id") or tx.get("cd_pix", "unknown_tx")
        payer_id = tx.get("customer_id") or tx.get("cd_cpf_pagador", "unknown_payer")
        receiver_id = tx.get("counterparty_id") or tx.get("cd_cpf_cnpj_recebedor")
        
        if not receiver_id:
            logger.warning("Graph Engine ignorando evento: 'counterparty_id' não encontrado.")
            receiver_id = "unknown_receiver"

        try:
            # Tentar pegar o dt_pix ou event_datetime e converter
            dt_str = tx.get("event_datetime") or tx.get("dt_pix")
            if dt_str:
                dt_str = dt_str.replace("Z", "")
                dt_event = datetime.fromisoformat(dt_str)
            else:
                dt_event = datetime.utcnow()
        except ValueError:
            dt_event = datetime.utcnow()

        try:
            vl_pix = float(tx.get("vl_pix", 0.0))
        except (ValueError, TypeError):
            vl_pix = 0.0

        # Atualizar janelas
        event = {"dt": dt_event, "vl": vl_pix, "payer": payer_id, "receiver": receiver_id}
        self.payer_history[payer_id].append(event)
        self.receiver_history[receiver_id].append(event)

        # 2. Computar Features Básicas
        now_1h = dt_event - timedelta(hours=1)
        now_24h = dt_event - timedelta(hours=24)

        # Histórico do Recebedor
        recv_hist = self.receiver_history.get(receiver_id, [])
        in_deg_1h = sum(1 for e in recv_hist if e["dt"] >= now_1h)
        in_deg_24h = sum(1 for e in recv_hist if e["dt"] >= now_24h)
        
        unique_payers_24h = set(e["payer"] for e in recv_hist if e["dt"] >= now_24h)
        recv_total_amt_24h = sum(e["vl"] for e in recv_hist if e["dt"] >= now_24h)
        
        first_seen_age_hours = 0.0
        if recv_hist:
            first_event_dt = min(e["dt"] for e in recv_hist)
            first_seen_age_hours = (dt_event - first_event_dt).total_seconds() / 3600.0
            
        is_new_receiver = 1 if first_seen_age_hours < 24.0 else 0

        # Histórico do Pagador
        payer_hist = self.payer_history.get(payer_id, [])
        out_deg_24h = sum(1 for e in payer_hist if e["dt"] >= now_24h)
        unique_recvs_24h = set(e["receiver"] for e in payer_hist if e["dt"] >= now_24h)
        payer_total_amt_24h = sum(e["vl"] for e in payer_hist if e["dt"] >= now_24h)

        # 3. Computar Scores Investigativos (Heurísticas Isoladas)
        # Mula Score
        mule_score = 0
        if len(unique_payers_24h) >= 3 and is_new_receiver:
            mule_score += 30
        if recv_total_amt_24h > 5000:
            mule_score += 20
        if res.get("decisao") == "BLOQUEAR":
            mule_score += 10
            
        # Bridge Account Score (Conta Ponte)
        bridge_score = 0
        if len(unique_payers_24h) >= 1 and len(unique_recvs_24h) >= 1:
            bridge_score += 50
        if bridge_score > 0 and (recv_total_amt_24h > 0 and payer_total_amt_24h > 0):
            ratio = min(recv_total_amt_24h, payer_total_amt_24h) / max(recv_total_amt_24h, payer_total_amt_24h)
            if ratio > 0.8: # Valor que entra é quase igual ao valor que sai
                bridge_score += 40

        # Fanout Score
        fanout_score = 0
        if len(unique_recvs_24h) >= 4:
            fanout_score += 40
            if is_new_receiver:
                fanout_score += 20

        # 4. Gravar Arquivo
        row = [
            tx_id,
            dt_event.isoformat(),
            payer_id,
            receiver_id,
            vl_pix,
            res.get("decisao"),
            res.get("score_final"),
            res.get("r5b14_rule_applied", ""),
            res.get("r5b22_rule_applied", ""),
            in_deg_1h,
            in_deg_24h,
            out_deg_24h,
            len(unique_payers_24h),
            len(unique_recvs_24h),
            round(recv_total_amt_24h, 2),
            round(payer_total_amt_24h, 2),
            round(first_seen_age_hours, 2),
            is_new_receiver,
            mule_score,
            bridge_score,
            fanout_score
        ]

        with self._csv_lock:
            try:
                with open(self.report_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
            except Exception as e:
                logger.error(f"Erro ao gravar linha de investigação de grafos: {e}")
