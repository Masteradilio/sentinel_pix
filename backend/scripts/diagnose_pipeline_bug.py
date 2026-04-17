"""
diagnose_pipeline_bug_v2.py — Diagnóstico fino por SUB-ETAPA do _create_features
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
CORE_DIR = BACKEND_DIR / "core"
DADOS_DIR = PROJECT_ROOT / "dados"
DATASET_PATH = DADOS_DIR / "base_treino_final.csv"

sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


def snapshot(df: pd.DataFrame, tag: str) -> None:
    """Snapshot mínimo com foco nas colunas críticas."""
    cid = df["customer_id"].iloc[0] if "customer_id" in df.columns else "<AUSENTE>"
    edt = df["event_datetime"].iloc[0] if "event_datetime" in df.columns else "<AUSENTE>"
    vpix = df["vl_pix"].iloc[0] if "vl_pix" in df.columns else "<AUSENTE>"
    dup = df.columns[df.columns.duplicated()].tolist()
    flag = "🚨" if (pd.isna(cid) or pd.isna(edt) or dup) else "✅"
    print(f"  {flag} [{tag}] shape={df.shape} | cid={cid!r} | edt={edt!r} | vpix={vpix!r} | dup={dup}")


def main() -> None:
    print("=" * 72)
    print("  DIAGNÓSTICO FINO — _create_features passo a passo")
    print("=" * 72)

    df_full = pd.read_csv(DATASET_PATH, nrows=10)
    df_full["event_datetime"] = pd.to_datetime(df_full["event_datetime"], errors="coerce")
    row_dict = df_full.iloc[0].to_dict()

    from core.pipeline_orquestrador import PipelineOrquestrador
    orq = PipelineOrquestrador(shap_enabled=False)

    print("\n[A] df_raw após _prepare_raw:")
    df = orq._prepare_raw(row_dict)
    snapshot(df, "INÍCIO _create_features")

    # Agora vamos replicar internamente as etapas de _create_features
    # e logar entre cada uma. Preciso que você ABRA o arquivo e
    # monte a lista de chamadas na ordem em que aparecem no método.

    # Baseado no stack trace anterior, o método chama ao menos:
    # 1. _create_temporal_features(df)
    # 2. _create_value_features(df)
    # 3. _create_sequential_features(df)
    # 4. outras?

    # Tentamos descobrir via introspecção
    import inspect
    src = inspect.getsource(orq._create_features)
    print("\n[B] Código de _create_features:")
    print("─" * 72)
    print(src)
    print("─" * 72)

    # Executar passo a passo se conseguirmos
    print("\n[C] Executando sub-etapas manualmente (ordem comum):")

    df = orq._prepare_raw(row_dict)
    snapshot(df, "após _prepare_raw")

    method_names = [
        "_create_temporal_features",
        "_create_value_features",
        "_create_sequential_features",
        "_create_key_features",
        "_create_receiver_features",
        "_create_aggregated_features",
    ]

    for name in method_names:
        method = getattr(orq, name, None)
        if method is None:
            print(f"  ⏭️  {name}: não existe (pulando)")
            continue
        try:
            df = method(df)
            snapshot(df, f"após {name}")
        except Exception as e:
            print(f"  ❌ {name} FALHOU: {type(e).__name__}: {e}")
            traceback.print_exc()
            return


if __name__ == "__main__":
    main()
