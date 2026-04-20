import sys
from pathlib import Path
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experimentos.utils_experimentos import load_dataset

df = load_dataset()
residual = df[df["customer_id"].astype(str).str.contains("32339437172")]
print(f"Found {len(residual)} rows")
if not residual.empty:
    row = residual.iloc[0].to_dict()
    print(json.dumps({k: str(v) for k, v in row.items()}, indent=2))
