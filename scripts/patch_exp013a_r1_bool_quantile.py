#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Patch EXP-013A-R1 - corrige quantile sobre booleanos.

from pathlib import Path

SCRIPT = Path("scripts/exp_013a_statistical_fp_tp_diagnostics.py")

if not SCRIPT.exists():
    raise FileNotFoundError(f"Script não encontrado: {SCRIPT}")

original = SCRIPT.read_text(encoding="utf-8")
txt = original

patterns = [
    (
        'x1 = pd.to_numeric(group_df.loc[y == 1, col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()\n        x0 = pd.to_numeric(group_df.loc[y == 0, col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()',
        'x1 = (\n            pd.to_numeric(group_df.loc[y == 1, col], errors="coerce")\n            .replace([np.inf, -np.inf], np.nan)\n            .dropna()\n            .astype(float)\n        )\n        x0 = (\n            pd.to_numeric(group_df.loc[y == 0, col], errors="coerce")\n            .replace([np.inf, -np.inf], np.nan)\n            .dropna()\n            .astype(float)\n        )',
    ),
    (
        "x1=pd.to_numeric(group_df.loc[y==1,col],errors=\'coerce\').replace([np.inf,-np.inf],np.nan).dropna(); x0=pd.to_numeric(group_df.loc[y==0,col],errors=\'coerce\').replace([np.inf,-np.inf],np.nan).dropna()",
        "x1=pd.to_numeric(group_df.loc[y==1,col],errors=\'coerce\').replace([np.inf,-np.inf],np.nan).dropna().astype(float); x0=pd.to_numeric(group_df.loc[y==0,col],errors=\'coerce\').replace([np.inf,-np.inf],np.nan).dropna().astype(float)",
    ),
    (
        'x1 = pd.to_numeric(group_df.loc[y==1, col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()\n        x0 = pd.to_numeric(group_df.loc[y==0, col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()',
        'x1 = pd.to_numeric(group_df.loc[y==1, col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)\n        x0 = pd.to_numeric(group_df.loc[y==0, col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)',
    ),
]

replaced = False
for old, new in patterns:
    if old in txt:
        txt = txt.replace(old, new, 1)
        replaced = True
        break

if not replaced:
    raise RuntimeError(
        "Não encontrei o trecho esperado para aplicar o patch. "
        "Procure no script por x1 = pd.to_numeric dentro de run_numeric_tests "
        "e adicione .astype(float) depois de .dropna() em x1 e x0."
    )

backup = SCRIPT.with_suffix(".py.bak_exp013a_r1")
backup.write_text(original, encoding="utf-8")
SCRIPT.write_text(txt, encoding="utf-8", newline="\n")

print(f"[OK] Patch aplicado em: {SCRIPT}")
print(f"[OK] Backup salvo em: {backup}")
print("[NEXT] Execute novamente:")
print("python scripts\\exp_013a_statistical_fp_tp_diagnostics.py")
