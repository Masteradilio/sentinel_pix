#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3C — Port EXP-013J Scenario Factory to Expanded Dataset

Objetivo:
  Fazer a continuidade correta do EXP-013 no dataset expandido com 1.465 fraudes.

  Em vez de reconstruir pred_STRICT_RECALL95_SAFE_ONLY por surrogate, este
  experimento tenta portar/rodar a própria fábrica de cenários do EXP-013J:

      scripts/exp_013j_safe_and_onetp_microveto.py

  A meta é gerar no dataset expandido:

      pred_STRICT_RECALL95_SAFE_ONLY

  Depois, reaplicar a política congelada do EXP-013K sobre essa base e medir
  métricas comparáveis ao champion pequeno.

Por que este script existe:
  O EXP-014B-R3B mostrou que:
    - nenhuma receita portável foi aceita com fidelidade mínima 0.995;
    - o melhor surrogate ficou perto, mas abaixo do alvo;
    - portanto, o caminho certo é portar a origem real da base:
      EXP-013J STRICT_RECALL95_SAFE_ONLY.

Estratégia:
  1. Localizar scripts/exp_013j_safe_and_onetp_microveto.py.
  2. Tentar executar o EXP-013J original com argumentos comuns:
       --input / --output-dir
       --input-csv / --output-dir
       --source / --output-dir
  3. Se falhar, criar uma cópia patchada do EXP-013J em scripts/
     trocando defaults de input/output por:
       dados/exp014a_expanded_scored_input.csv
       resultados/experimentos/EXP-014B-R3C/exp013j_port_run
  4. Procurar qualquer CSV gerado com pred_STRICT_RECALL95_SAFE_ONLY.
  5. Se encontrar, montar:
       dados/exp014b_r3c_expanded_with_exp013j_base.csv
  6. Reaplicar EXP-013K congelado e medir:
       TP, FP, FN, recall, precision, FPR, Wilson, blocos temporais.
  7. Se não encontrar, encerrar sem crash e produzir relatório com:
       tentativas executadas,
       stderr/stdout,
       janelas de código da origem,
       próximos comandos manuais.

Uso:
  python scripts/exp_014b_r3c_port_exp013j_scenario_factory.py

Se você já tiver gerado manualmente um CSV com pred_STRICT_RECALL95_SAFE_ONLY:
  python scripts/exp_014b_r3c_port_exp013j_scenario_factory.py --source-predictions caminho\\06_predictions_by_scenario.csv

Se quiser apenas diagnosticar a origem sem executar:
  python scripts/exp_014b_r3c_port_exp013j_scenario_factory.py --no-execute
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "dados").exists() else Path.cwd()

DEFAULT_EXPANDED_INPUT = PROJECT_ROOT / "dados" / "exp014a_expanded_scored_input.csv"
DEFAULT_EXPANDED_OUTPUT = PROJECT_ROOT / "dados" / "exp014b_r3c_expanded_with_exp013j_base.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R3C"
DEFAULT_EXP013J_SCRIPT = PROJECT_ROOT / "scripts" / "exp_013j_safe_and_onetp_microveto.py"
DEFAULT_POLICY = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013K" / "12_policy_artifact.json"

BASE_COL = "pred_STRICT_RECALL95_SAFE_ONLY"
FINAL_COL = "exp014b_r3c_exp013k_replay_pred"

TERMS_TO_WINDOW = [
    "STRICT_RECALL95_SAFE_ONLY",
    "pred_STRICT_RECALL95_SAFE_ONLY",
    "scenario_metrics",
    "predictions",
    "argparse",
    "DEFAULT_INPUT",
    "DEFAULT_OUTPUT",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def normalize_columns(df: pd.DataFrame, require_label: bool = True) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().split(".")[-1] for c in df.columns]

    if "transaction_id" not in df.columns and "cd_pix" in df.columns:
        df["transaction_id"] = df["cd_pix"]
    if "event_datetime" not in df.columns and "dt_pix" in df.columns:
        df["event_datetime"] = df["dt_pix"]

    if "is_fraud" in df.columns:
        df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)
    elif require_label:
        raise RuntimeError("Coluna is_fraud ausente.")

    if "transaction_id" in df.columns:
        df["transaction_id"] = df["transaction_id"].astype("string").str.strip()

    if "decisao" in df.columns and "runtime_flagged" not in df.columns:
        df["runtime_flagged"] = df["decisao"].astype(str).str.upper().isin({"CONFIRMAR", "BLOQUEAR"}).astype(int)
    if "runtime_flagged" not in df.columns:
        df["runtime_flagged"] = 0

    for c in ["event_datetime", "data_pix"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    return df.reset_index(drop=True)


def pick_col(df: pd.DataFrame, names: str | list[str]) -> str | None:
    if isinstance(names, str):
        names = [names]
    for n in names:
        if n in df.columns:
            return n
    return None


def num(df: pd.DataFrame, names: str | list[str], default: float = 0.0) -> pd.Series:
    col = pick_col(df, names)
    if col is None:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def qbin_series(s: pd.Series, name: str, bins: list[float]) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    labels = []
    edges = [-np.inf] + bins + [np.inf]
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        if np.isneginf(left):
            labels.append(f"{name}_LT_{right:g}")
        elif np.isposinf(right):
            labels.append(f"{name}_GE_{left:g}")
        else:
            labels.append(f"{name}_{left:g}_{right:g}")
    return pd.cut(vals, bins=edges, labels=labels, include_lowest=True).astype("string").fillna(f"{name}_MISSING").astype(str)


def add_bins_and_guards(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "lgbm_bin" not in df.columns and pick_col(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"]):
        df["lgbm_bin"] = qbin_series(num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0), "lgbm", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1])
    if "if_bin" not in df.columns and pick_col(df, ["if_percentile", "if_percentile_x", "if_percentile_y"]):
        df["if_bin"] = qbin_series(num(df, ["if_percentile", "if_percentile_x", "if_percentile_y"], 0.0), "if", [0.32, 0.5, 0.7, 0.85, 0.95])
    if "score_bin" not in df.columns and "score_final" in df.columns:
        df["score_bin"] = qbin_series(num(df, "score_final", 0.0), "score", [0.5, 1, 2, 3, 5, 10])
    if "vl_bin" not in df.columns and "vl_pix" in df.columns:
        df["vl_bin"] = qbin_series(num(df, "vl_pix", 0.0), "vl", [20, 50, 100, 250, 500, 1000, 5000, 10000])
    if "ratio_bin" not in df.columns and "ratio_valor_media_pagador_90d" in df.columns:
        df["ratio_bin"] = qbin_series(num(df, "ratio_valor_media_pagador_90d", 0.0), "ratio", [0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    if "qtd_rec_bin" not in df.columns and "qtd_pix_recebidos_180d" in df.columns:
        df["qtd_rec_bin"] = qbin_series(num(df, "qtd_pix_recebidos_180d", 0.0), "qtdrec", [0, 1, 2, 5, 10, 20, 50, 100])
    if "valor_rec_bin" not in df.columns and "valor_total_recebido_180d" in df.columns:
        df["valor_rec_bin"] = qbin_series(num(df, "valor_total_recebido_180d", 0.0), "valrec", [0, 100, 500, 1000, 5000, 10000, 25000])

    se_score = num(df, ["se_score_x", "se_score_y", "se_score"], 0.0)
    se_count = num(df, ["se_patterns_count", "se_pattern_count"], 0.0)
    beh_score = num(df, ["beh_score", "behavioral_score"], 0.0)
    beh_count = num(df, ["beh_factors_count", "behavioral_risk_factor_count"], 0.0)
    runtime = num(df, "runtime_flagged", 0.0)

    module_strong = (
        (se_score >= 40)
        | (se_count >= 2)
        | (beh_score >= 25)
        | (beh_count >= 2)
        | (runtime >= 1)
    )
    df["module_quiet"] = np.where(module_strong, "module_strong", "module_quiet")
    return df


def compute_metrics(y_true, y_pred) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 8),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 8),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 8),
        "fpr": round(float(fp / max(fp + tn, 1)), 8),
    }


def load_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Policy artifact não encontrado: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "selected_rules" not in obj:
        raise RuntimeError("Policy artifact sem selected_rules.")
    return obj


def parse_params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(str(raw))
    except Exception:
        return {}


def rule_mask(df: pd.DataFrame, rule: dict[str, Any], current_pred: np.ndarray) -> np.ndarray:
    params = parse_params(rule.get("params_json", {}))
    if not params and isinstance(rule.get("params"), dict):
        params = rule["params"]

    cols = params.get("combo_cols", [])
    vals = params.get("combo_values", [])
    require_module_quiet = bool(params.get("require_module_quiet", False))

    if not cols:
        desc = str(rule.get("description", ""))
        cols, vals = [], []
        for part in desc.split(" AND "):
            if "=" in part:
                c, v = part.split("=", 1)
                cols.append(c.strip())
                vals.append(v.strip())

    if not cols:
        raise RuntimeError(f"Não consegui parsear regra: {rule}")

    mask = np.ones(len(df), dtype=bool)
    for c, v in zip(cols, vals):
        if c not in df.columns:
            raise RuntimeError(f"Coluna da regra ausente: {c}")
        mask = mask & (df[c].astype("string").fillna("<MISSING>").astype(str).to_numpy() == str(v))

    if require_module_quiet:
        if "module_quiet" not in df.columns:
            raise RuntimeError("Regra exige module_quiet.")
        mask = mask & (df["module_quiet"].astype(str).to_numpy() == "module_quiet")

    return mask & (current_pred.astype(int) == 1)


def apply_policy(df: pd.DataFrame, policy: dict[str, Any], base_pred: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    y = df["is_fraud"].to_numpy(dtype=int)
    pred = base_pred.astype(int).copy()
    rows = []

    for idx, rule in enumerate(policy.get("selected_rules", [])):
        mask = rule_mask(df, rule, pred)
        tp_loss = int(((y == 1) & mask).sum())
        fp_removed = int(((y == 0) & mask).sum())
        pred[mask] = 0
        rows.append({
            "rule_index": idx,
            "description": rule.get("description"),
            "tp_loss": tp_loss,
            "fp_removed": fp_removed,
            "n_removed": int(mask.sum()),
            "params_json": rule.get("params_json"),
        })
    return pred, pd.DataFrame(rows)


def make_time_blocks(df: pd.DataFrame, n_blocks: int) -> pd.Series:
    if "data_pix" in df.columns and df["data_pix"].notna().any():
        dates = pd.to_datetime(df["data_pix"], errors="coerce")
    elif "event_datetime" in df.columns and df["event_datetime"].notna().any():
        dates = pd.to_datetime(df["event_datetime"], errors="coerce")
    else:
        return pd.qcut(np.arange(len(df)), q=min(n_blocks, len(df)), labels=False, duplicates="drop").astype(int)

    tmp = pd.DataFrame({"date": dates, "_idx": np.arange(len(df))}).sort_values(["date", "_idx"])
    tmp["block"] = pd.qcut(np.arange(len(tmp)), q=min(n_blocks, len(tmp)), labels=False, duplicates="drop")
    out = pd.Series(index=tmp["_idx"].values, data=tmp["block"].values).sort_index()
    return out.astype(int)


def block_metrics(df: pd.DataFrame, pred: np.ndarray, blocks: pd.Series, policy_name: str) -> pd.DataFrame:
    y = df["is_fraud"].to_numpy(dtype=int)
    rows = []
    bvals = blocks.to_numpy()
    for b in sorted(blocks.dropna().unique()):
        idx = bvals == b
        part = df.loc[idx]
        rows.append({
            "policy_name": policy_name,
            "block": int(b),
            "n_rows": int(len(part)),
            "n_frauds": int(part["is_fraud"].sum()),
            **compute_metrics(y[idx], pred[idx]),
        })
    return pd.DataFrame(rows)


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) / n) + (z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def read_source_windows(script_path: Path, output_dir: Path, radius: int = 18) -> pd.DataFrame:
    rows = []
    if not script_path.exists():
        return pd.DataFrame([{"error": f"script_not_found:{script_path}"}])

    lines = script_path.read_text(encoding="utf-8", errors="replace").splitlines()
    used_ranges = []

    for i, line in enumerate(lines, start=1):
        if any(t in line for t in TERMS_TO_WINDOW):
            start = max(1, i - radius)
            end = min(len(lines), i + radius)
            if any(not (end < a or start > b) for a, b in used_ranges):
                continue
            used_ranges.append((start, end))
            rows.append({
                "start_line": start,
                "end_line": end,
                "term_line": i,
                "terms": "|".join([t for t in TERMS_TO_WINDOW if t in line]),
                "text": "\n".join(f"{j:04d}: {lines[j-1]}" for j in range(start, end + 1)),
            })

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "03_source_windows.csv", index=False)

    md_lines = ["# EXP-013J source windows", ""]
    for _, row in out.iterrows():
        md_lines.append(f"## Lines {row['start_line']}-{row['end_line']} — {row['terms']}")
        md_lines.append("")
        md_lines.append("```python")
        md_lines.append(str(row["text"]))
        md_lines.append("```")
        md_lines.append("")
    (output_dir / "03_source_windows.md").write_text("\n".join(md_lines), encoding="utf-8")
    return out


def run_command(cmd: list[str], cwd: Path, timeout: int, output_dir: Path, attempt_name: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    stdout_path = output_dir / f"{attempt_name}_stdout.txt"
    stderr_path = output_dir / f"{attempt_name}_stderr.txt"

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            timeout=timeout,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            env=env,
        )
        stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="replace")
        return {
            "attempt": attempt_name,
            "cmd": " ".join([str(x) for x in cmd]),
            "returncode": proc.returncode,
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "stdout_tail": (proc.stdout or "")[-1200:],
            "stderr_tail": (proc.stderr or "")[-1200:],
        }
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8", errors="replace")
        stderr_path.write_text((exc.stderr or "") + "\nTIMEOUT", encoding="utf-8", errors="replace")
        return {
            "attempt": attempt_name,
            "cmd": " ".join([str(x) for x in cmd]),
            "returncode": "TIMEOUT",
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "stdout_tail": str(exc.stdout or "")[-1200:],
            "stderr_tail": str(exc.stderr or "")[-1200:] + "\nTIMEOUT",
        }
    except Exception as exc:
        stderr_path.write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8", errors="replace")
        return {
            "attempt": attempt_name,
            "cmd": " ".join([str(x) for x in cmd]),
            "returncode": "EXCEPTION",
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "stdout_tail": "",
            "stderr_tail": f"{type(exc).__name__}: {exc}",
        }


def find_prediction_csvs(paths: list[Path]) -> list[Path]:
    out = []
    seen = set()
    for root in paths:
        if not root.exists():
            continue
        for p in root.rglob("*.csv"):
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                cols = pd.read_csv(p, nrows=0).columns
                cols = [str(c).strip().split(".")[-1] for c in cols]
                if BASE_COL in cols:
                    out.append(p)
            except Exception:
                continue
    return out


def extract_default_paths_from_source(text: str) -> list[str]:
    # Lightweight extractor of quoted paths that look like previous experiment files.
    patterns = [
        r'["\']([^"\']*EXP-013[^"\']*)["\']',
        r'["\']([^"\']*dados[^"\']*\.csv)["\']',
        r'["\']([^"\']*resultados[^"\']*)["\']',
    ]
    hits = []
    for pat in patterns:
        hits.extend(re.findall(pat, text, flags=re.IGNORECASE))
    # Keep unique, long-ish strings.
    seen = set()
    out = []
    for h in hits:
        if len(h) < 8:
            continue
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def create_patched_exp013j(script_path: Path, expanded_input: Path, run_dir: Path, output_dir: Path) -> Path:
    text = script_path.read_text(encoding="utf-8", errors="replace")
    patched = text

    # Replace obvious DEFAULT_INPUT / DEFAULT_OUTPUT_DIR assignments when present.
    patched = re.sub(
        r'(DEFAULT_INPUT\s*=\s*)([^\n]+)',
        lambda m: f'{m.group(1)}Path(r"{expanded_input}")',
        patched,
    )
    patched = re.sub(
        r'(DEFAULT_OUTPUT_DIR\s*=\s*)([^\n]+)',
        lambda m: f'{m.group(1)}Path(r"{run_dir}")',
        patched,
    )
    patched = re.sub(
        r'(OUTPUT_DIR\s*=\s*)([^\n]+)',
        lambda m: f'{m.group(1)}Path(r"{run_dir}")',
        patched,
    )

    # Replace strings referencing common prior input CSVs with expanded input.
    for old in extract_default_paths_from_source(text):
        low = old.lower()
        if old.endswith(".csv") and ("exp-013" in low or "dados" in low or "resultados" in low):
            patched = patched.replace(old, str(expanded_input))
        elif "exp-013j" in low or "resultados" in low:
            # Avoid replacing policy artifacts; only output-like dirs.
            if not old.endswith(".json"):
                patched = patched.replace(old, str(run_dir))

    # As a safety, append a banner comment.
    patched = "# AUTO-GENERATED by EXP-014B-R3C. Do not edit manually.\n" + patched

    patched_path = output_dir / "exp_014b_r3c_exp013j_ported_copy.py"
    patched_path.write_text(patched, encoding="utf-8", newline="\n")
    return patched_path


def try_execute_exp013j(
    script_path: Path,
    expanded_input: Path,
    output_dir: Path,
    timeout: int,
    no_execute: bool,
) -> tuple[list[dict[str, Any]], list[Path]]:
    attempts = []
    run_root = output_dir / "exp013j_port_run"
    run_root.mkdir(parents=True, exist_ok=True)

    if no_execute:
        return attempts, []

    python_exe = sys.executable or "python"

    candidate_cmds = [
        [python_exe, str(script_path), "--input", str(expanded_input), "--output-dir", str(run_root)],
        [python_exe, str(script_path), "--input-csv", str(expanded_input), "--output-dir", str(run_root)],
        [python_exe, str(script_path), "--source", str(expanded_input), "--output-dir", str(run_root)],
        [python_exe, str(script_path), "--data", str(expanded_input), "--output-dir", str(run_root)],
    ]

    for idx, cmd in enumerate(candidate_cmds, start=1):
        name = f"attempt_{idx:02d}_direct"
        log(f"  tentando {name}: {' '.join(cmd)}")
        attempts.append(run_command(cmd, PROJECT_ROOT, timeout, output_dir, name))
        found = find_prediction_csvs([run_root, output_dir])
        if found:
            return attempts, found

    # Ported copy attempt.
    patched = create_patched_exp013j(script_path, expanded_input, run_root, output_dir)
    cmd = [python_exe, str(patched)]
    name = "attempt_99_patched_copy"
    log(f"  tentando {name}: {' '.join(cmd)}")
    attempts.append(run_command(cmd, PROJECT_ROOT, timeout, output_dir, name))

    found = find_prediction_csvs([run_root, output_dir])
    return attempts, found


def select_predictions_file(files: list[Path]) -> Path | None:
    if not files:
        return None

    def score(p: Path) -> int:
        s = str(p).lower()
        v = 0
        if "06_predictions_by_scenario" in s:
            v += 1000
        if "prediction" in s:
            v += 100
        if "false" in s:
            v -= 200
        if "sample" in s:
            v -= 100
        return v

    return sorted(files, key=score, reverse=True)[0]


def merge_base_into_expanded(expanded: pd.DataFrame, pred_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    pred_df = normalize_columns(pred_df, require_label=False)

    if BASE_COL not in pred_df.columns:
        raise RuntimeError(f"Arquivo de predições não contém {BASE_COL}.")

    out = expanded.copy()

    # Prefer transaction_id merge.
    if "transaction_id" in out.columns and "transaction_id" in pred_df.columns:
        small = pred_df[["transaction_id", BASE_COL]].drop_duplicates("transaction_id").copy()
        merged = out.merge(small, on="transaction_id", how="left", suffixes=("", "_from_exp013j"))
        source_col = BASE_COL
        if BASE_COL + "_from_exp013j" in merged.columns:
            source_col = BASE_COL + "_from_exp013j"
        missing = int(merged[source_col].isna().sum())
        if missing == 0:
            merged[BASE_COL] = pd.to_numeric(merged[source_col], errors="coerce").fillna(0).astype(int)
            merged["exp013k_base_pred"] = merged[BASE_COL]
            return merged, "merge_on_transaction_id"
        # If partial merge, still allow only when same len can fix by index.
        out = expanded.copy()

    if len(pred_df) == len(out):
        out[BASE_COL] = pd.to_numeric(pred_df[BASE_COL], errors="coerce").fillna(0).astype(int).to_numpy()
        out["exp013k_base_pred"] = out[BASE_COL]
        return out, "index_aligned_same_length"

    raise RuntimeError(
        f"Não foi possível alinhar predições: len(expanded)={len(out)}, len(predictions)={len(pred_df)}."
    )


def process_source_predictions(pred_path: Path, expanded_input: Path, output_path: Path, policy: dict[str, Any], output_dir: Path, time_blocks: int, target_recall: float) -> dict[str, Any]:
    expanded = add_bins_and_guards(normalize_columns(pd.read_csv(expanded_input, low_memory=False), require_label=True))
    pred_df = pd.read_csv(pred_path, low_memory=False)

    expanded_with_base, merge_mode = merge_base_into_expanded(expanded, pred_df)
    y = expanded_with_base["is_fraud"].to_numpy(dtype=int)
    base_pred = expanded_with_base[BASE_COL].to_numpy(dtype=int)

    final_pred, rule_impact = apply_policy(expanded_with_base, policy, base_pred)
    expanded_with_base[FINAL_COL] = final_pred.astype(int)
    expanded_with_base["exp013k_residual_fp_pred"] = final_pred.astype(int)

    base_metrics = compute_metrics(y, base_pred)
    final_metrics = compute_metrics(y, final_pred)

    metrics_df = pd.DataFrame([
        {"policy_name": "EXP014B_R3C_EXPANDED_EXP013J_BASE", **base_metrics},
        {"policy_name": "EXP014B_R3C_EXPANDED_EXP013J_BASE_PLUS_EXP013K", **final_metrics},
    ])
    metrics_df.to_csv(output_dir / "06_expanded_replay_metrics.csv", index=False)
    rule_impact.to_csv(output_dir / "07_rule_impact_expanded.csv", index=False)

    blocks = make_time_blocks(expanded_with_base, time_blocks)
    block_df = pd.concat([
        block_metrics(expanded_with_base, base_pred, blocks, "EXP014B_R3C_EXPANDED_EXP013J_BASE"),
        block_metrics(expanded_with_base, final_pred, blocks, "EXP014B_R3C_EXPANDED_EXP013J_BASE_PLUS_EXP013K"),
    ], ignore_index=True)
    block_df.to_csv(output_dir / "08_time_block_metrics.csv", index=False)

    total_frauds = int(y.sum())
    min_tp_required = int(math.ceil(target_recall * total_frauds))
    wl, wh = wilson_ci(final_metrics["tp"], total_frauds)
    wilson_df = pd.DataFrame([{
        "metric": "recall",
        "successes_tp": final_metrics["tp"],
        "n_frauds": total_frauds,
        "point_estimate": final_metrics["recall"],
        "wilson_low": wl,
        "wilson_high": wh,
        "target_recall": target_recall,
        "min_tp_required": min_tp_required,
        "tp_buffer_vs_target": final_metrics["tp"] - min_tp_required,
        "wilson_low_ge_target": bool(wl >= target_recall),
    }])
    wilson_df.to_csv(output_dir / "09_wilson_recall_ci.csv", index=False)

    expanded_with_base[(expanded_with_base["is_fraud"] == 1) & (expanded_with_base[FINAL_COL] == 0)].to_csv(output_dir / "10_false_negatives.csv", index=False)
    fp = expanded_with_base[(expanded_with_base["is_fraud"] == 0) & (expanded_with_base[FINAL_COL] == 1)].copy()
    if len(fp) > 5000:
        fp = fp.sample(5000, random_state=42)
    fp.to_csv(output_dir / "11_false_positives_sample.csv", index=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    expanded_with_base.to_csv(output_path, index=False)

    return {
        "source_predictions": str(pred_path),
        "merge_mode": merge_mode,
        "expanded_output": str(output_path),
        "expanded_output_built": True,
        "base_metrics": base_metrics,
        "final_metrics": final_metrics,
        "wilson_low": wl,
        "wilson_high": wh,
        "min_tp_required": min_tp_required,
        "target_met": final_metrics["recall"] >= target_recall,
        "wilson_pass": wl >= target_recall,
    }


def make_report(summary: dict[str, Any], attempts_df: pd.DataFrame, found_df: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014B-R3C — Port EXP-013J Scenario Factory")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Expanded output built: `{summary.get('expanded_output_built')}`")
    lines.append("")
    if summary.get("source_predictions"):
        lines.append(f"- Source predictions: `{summary['source_predictions']}`")
    if summary.get("expanded_output"):
        lines.append(f"- Expanded output: `{summary['expanded_output']}`")
    lines.append("")
    lines.append("## Métricas")
    if summary.get("base_metrics"):
        lines.append("### Base EXP-013J no expandido")
        lines.append("```json")
        lines.append(json.dumps(summary["base_metrics"], ensure_ascii=False, indent=2))
        lines.append("```")
    if summary.get("final_metrics"):
        lines.append("### Base + EXP-013K no expandido")
        lines.append("```json")
        lines.append(json.dumps(summary["final_metrics"], ensure_ascii=False, indent=2))
        lines.append("```")
    lines.append("")
    lines.append("## Tentativas de execução")
    if attempts_df.empty:
        lines.append("Nenhuma tentativa executada.")
    else:
        show = ["attempt", "returncode", "elapsed_seconds", "cmd", "stderr_tail"]
        lines.append(attempts_df[[c for c in show if c in attempts_df.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## CSVs encontrados com pred_STRICT_RECALL95_SAFE_ONLY")
    if found_df.empty:
        lines.append("Nenhum CSV encontrado.")
    else:
        lines.append(found_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Próximo passo")
    if summary.get("expanded_output_built"):
        lines.append("Rodar o replay/mineração residual congelada:")
        lines.append("```powershell")
        lines.append("python scripts\\exp_014b_r3a_champion_replay_expanded.py --expanded-input dados\\exp014b_r3c_expanded_with_exp013j_base.csv")
        lines.append("```")
    else:
        lines.append("Abrir `03_source_windows.md` e `02_execution_attempts.csv` para ajustar manualmente o EXP-013J ou informar `--source-predictions` com o CSV gerado.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--expanded-input", default=str(DEFAULT_EXPANDED_INPUT))
    parser.add_argument("--expanded-output", default=str(DEFAULT_EXPANDED_OUTPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--exp013j-script", default=str(DEFAULT_EXP013J_SCRIPT))
    parser.add_argument("--policy-artifact", default=str(DEFAULT_POLICY))
    parser.add_argument("--source-predictions", default=None)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--time-blocks", type=int, default=10)
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expanded_input = Path(args.expanded_input)
    expanded_output = Path(args.expanded_output)
    exp013j_script = Path(args.exp013j_script)
    policy_path = Path(args.policy_artifact)

    log("=" * 80)
    log("EXP-014B-R3C — Port EXP-013J Scenario Factory")
    log("=" * 80)
    log(f"Expanded input: {expanded_input}")
    log(f"EXP-013J script: {exp013j_script}")

    if not expanded_input.exists():
        raise FileNotFoundError(f"expanded-input não encontrado: {expanded_input}")
    if not policy_path.exists():
        raise FileNotFoundError(f"policy-artifact não encontrado: {policy_path}")

    policy = load_policy(policy_path)

    source_windows = read_source_windows(exp013j_script, output_dir)
    attempts = []
    found_files = []

    if args.source_predictions:
        pred_path = Path(args.source_predictions)
        if not pred_path.exists():
            raise FileNotFoundError(f"source-predictions não encontrado: {pred_path}")
        found_files = [pred_path]
    else:
        if not exp013j_script.exists():
            attempts.append({
                "attempt": "locate_exp013j",
                "returncode": "SCRIPT_NOT_FOUND",
                "cmd": str(exp013j_script),
                "stderr_tail": "scripts/exp_013j_safe_and_onetp_microveto.py não encontrado.",
            })
        else:
            attempts, found_files = try_execute_exp013j(
                script_path=exp013j_script,
                expanded_input=expanded_input,
                output_dir=output_dir,
                timeout=args.timeout,
                no_execute=args.no_execute,
            )

    attempts_df = pd.DataFrame(attempts)
    attempts_df.to_csv(output_dir / "02_execution_attempts.csv", index=False)

    # Add any manually existing outputs from prior runs too.
    extra_found = find_prediction_csvs([output_dir, output_dir / "exp013j_port_run"])
    for p in extra_found:
        if p not in found_files:
            found_files.append(p)

    found_df = pd.DataFrame([{"path": str(p)} for p in found_files])
    found_df.to_csv(output_dir / "04_found_prediction_files.csv", index=False)

    selected_pred_file = select_predictions_file(found_files)

    result = {}
    objective_status = "DONE_EXP013J_PORT_NO_PREDICTIONS"
    if selected_pred_file is not None:
        log(f"Processando predições encontradas: {selected_pred_file}")
        try:
            result = process_source_predictions(
                pred_path=selected_pred_file,
                expanded_input=expanded_input,
                output_path=expanded_output,
                policy=policy,
                output_dir=output_dir,
                time_blocks=args.time_blocks,
                target_recall=args.target_recall,
            )
            objective_status = "DONE_EXP013J_BASE_PORTED"
            objective_status += "_TARGET_MET" if result["target_met"] else "_TARGET_NOT_MET"
            objective_status += "_WILSON_PASS" if result["wilson_pass"] else "_WILSON_NOT_PASS"
        except Exception as exc:
            result = {
                "source_predictions": str(selected_pred_file),
                "expanded_output_built": False,
                "processing_error": f"{type(exc).__name__}: {str(exc)[:1000]}",
            }
            objective_status = "DONE_PREDICTIONS_FOUND_PROCESSING_FAILED"

    summary = {
        "experiment": "EXP-014B-R3C",
        "status": "DONE",
        "objective_status": objective_status,
        "expanded_input": str(expanded_input),
        "exp013j_script": str(exp013j_script),
        "policy_artifact": str(policy_path),
        "n_execution_attempts": int(len(attempts_df)),
        "n_found_prediction_files": int(len(found_files)),
        "selected_prediction_file": str(selected_pred_file) if selected_pred_file else None,
        **result,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    artifact = {
        "experiment": "EXP-014B-R3C",
        "policy_name": "port_exp013j_scenario_factory_to_expanded",
        "objective_status": objective_status,
        "summary": summary,
        "notes": [
            "Attempts to generate pred_STRICT_RECALL95_SAFE_ONLY by running/porting EXP-013J.",
            "If successful, writes dados/exp014b_r3c_expanded_with_exp013j_base.csv.",
            "If unsuccessful, inspect source windows and execution attempts."
        ],
    }
    dump_json(artifact, output_dir / "12_policy_artifact.json")

    report = make_report(summary, attempts_df, found_df)
    (output_dir / "13_exp014b_r3c_report.md").write_text(report, encoding="utf-8")

    log("")
    log("=" * 80)
    log("EXP-014B-R3C CONCLUÍDO")
    log("=" * 80)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "02_execution_attempts.csv",
        output_dir / "03_source_windows.md",
        output_dir / "04_found_prediction_files.csv",
        output_dir / "06_expanded_replay_metrics.csv",
        output_dir / "07_rule_impact_expanded.csv",
        output_dir / "08_time_block_metrics.csv",
        output_dir / "09_wilson_recall_ci.csv",
        output_dir / "12_policy_artifact.json",
        output_dir / "13_exp014b_r3c_report.md",
    ]:
        if p.exists():
            log(f"  {p}")


if __name__ == "__main__":
    main()
