from __future__ import annotations

import pandas as pd
from pathlib import Path


ROOT = Path(".").resolve()
EXP_DIR = ROOT / "resultados" / "experimentos" / "EXP-009D"

fact_path = EXP_DIR / "07_powerbi_decision_fact.csv"
daily_path = EXP_DIR / "05_daily_metrics.csv"
rq_path = EXP_DIR / "08_powerbi_review_queue_fact.csv"

fact = pd.read_csv(fact_path)
daily = pd.read_csv(daily_path)
rq = pd.read_csv(rq_path)


def years_from(series: pd.Series) -> list[int]:
    years = pd.to_datetime(series, errors="coerce").dt.year.dropna().astype(int).unique()
    return sorted(years.tolist())


print("=== EXP-009D Date Validation ===")
print()

print("decision_fact rows:", len(fact))
print("transaction_date vazias:", int(fact["transaction_date"].isna().sum()))
print("transaction_date strings vazias:", int(fact["transaction_date"].fillna("").astype(str).str.strip().eq("").sum()))
print("anos decision_fact:", years_from(fact["transaction_date"]))
print()

print("daily rows:", len(daily))
print("anos daily:", years_from(daily["transaction_day"]))
print()

print("review_queue rows:", len(rq))
print("review_queue transaction_date vazias:", int(rq["transaction_date"].isna().sum()))
print("review_queue transaction_date strings vazias:", int(rq["transaction_date"].fillna("").astype(str).str.strip().eq("").sum()))
print("anos review_queue:", years_from(rq["transaction_date"]))
print()

ok = True

fact_years = years_from(fact["transaction_date"])
daily_years = years_from(daily["transaction_day"])
rq_years = years_from(rq["transaction_date"])

if int(fact["transaction_date"].isna().sum()) != 0:
    ok = False

if int(fact["transaction_date"].fillna("").astype(str).str.strip().eq("").sum()) != 0:
    ok = False

if int(rq["transaction_date"].isna().sum()) != 0:
    ok = False

if int(rq["transaction_date"].fillna("").astype(str).str.strip().eq("").sum()) != 0:
    ok = False

allowed_years = {2025, 2026}

if not set(fact_years).issubset(allowed_years):
    ok = False

if not set(daily_years).issubset(allowed_years):
    ok = False

if not set(rq_years).issubset(allowed_years):
    ok = False

print("STATUS:", "OK" if ok else "FALHOU")

if not ok:
    raise SystemExit(1)