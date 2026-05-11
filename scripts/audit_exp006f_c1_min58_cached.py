from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "resultados" / "experimentos" / "EXP-006C-R2"
OUT_DIR = ROOT / "resultados" / "experimentos" / "EXP-006F-C1-MIN58-AUDIT"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_seed(seed: int) -> pd.DataFrame:
    path = INPUT_DIR / f"baseline_predictions_seed_{seed}.csv"
    df = pd.read_csv(path)
    df["seed"] = seed
    return df


def flagged(df: pd.DataFrame) -> pd.Series:
    return df["decisao"].astype(str).isin(["CONFIRMAR", "BLOQUEAR"])


def metrics(df: pd.DataFrame) -> dict:
    y = df["is_fraud"].astype(int)
    p = flagged(df).astype(int)

    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": round(precision, 6),
        "Recall": round(recall, 6),
        "F1": round(f1, 6),
    }


def apply_c1_min58(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in [
        "vl_pix",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "lgbm_raw",
        "score_final",
        "se_score",
        "beh_score",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    mask = (
        out["decisao"].astype(str).eq("APROVAR")
        & out["first_receiver_flag"].astype(int).eq(1)
        & out["pix_key_random_flag"].astype(int).eq(0)
        & out["qt_tempo_relacionamento_mes"].le(12)
        & out["vl_pix"].ge(100)
        & out["vl_pix"].lt(500)
        & out["lgbm_raw"].ge(0.06)
        & out["lgbm_raw"].lt(0.10)
        & out["score_final"].ge(58.0)
        & out["score_final"].lt(62.0)
        & out["se_score"].le(0)
        & out["beh_score"].le(0)
    )

    out["exp006f_c1_min58_hit"] = mask
    out.loc[mask, "decisao_original_exp006f_c1"] = out.loc[mask, "decisao"]
    out.loc[mask, "score_final_original_exp006f_c1"] = out.loc[mask, "score_final"]
    out.loc[mask, "decisao"] = "CONFIRMAR"
    out.loc[mask, "score_final"] = out.loc[mask, "score_final"].apply(lambda x: max(float(x), 62.0))

    return out


def compare(base: pd.DataFrame, cand: pd.DataFrame) -> dict:
    y = base["is_fraud"].astype(int)
    b = flagged(base)
    c = flagged(cand)

    rec_fn = y.eq(1) & (~b) & c
    add_fp = y.eq(0) & (~b) & c
    lost_tp = y.eq(1) & b & (~c)

    cols = [
        "seed",
        "transaction_id",
        "customer_id",
        "is_fraud",
        "vl_pix",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "lgbm_raw",
        "score_final",
        "decisao",
        "exp006f_c1_min58_hit",
    ]
    cols = [c for c in cols if c in cand.columns]

    return {
        "fns_recuperados": int(rec_fn.sum()),
        "fps_adicionados": int(add_fp.sum()),
        "tps_perdidos": int(lost_tp.sum()),
        "rule_hits": int(cand["exp006f_c1_min58_hit"].sum()),
        "hits": cand.loc[cand["exp006f_c1_min58_hit"], cols].to_dict(orient="records"),
        "fps_adicionados_rows": cand.loc[add_fp, cols].to_dict(orient="records"),
    }


def main() -> None:
    rows = []
    deltas = {}

    for seed in [42, 123]:
        base = load_seed(seed)
        cand = apply_c1_min58(base)

        bm = metrics(base)
        cm = metrics(cand)

        bm["seed"] = seed
        bm["config"] = "BASELINE"

        cm["seed"] = seed
        cm["config"] = "C1_MIN58"

        rows.extend([bm, cm])
        deltas[str(seed)] = compare(base, cand)

        cand[cand["exp006f_c1_min58_hit"]].to_csv(
            OUT_DIR / f"hits_seed_{seed}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame(rows).to_csv(
        OUT_DIR / "metrics_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (OUT_DIR / "delta_by_seed.json").write_text(
        json.dumps(deltas, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[OK] Auditoria concluída.")
    print(f"[OK] Artefatos em: {OUT_DIR}")
    print(json.dumps(deltas, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()