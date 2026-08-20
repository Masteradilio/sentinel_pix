#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Patch EXP-013A-R2 — robusto

Corrige o erro:
  TypeError: numpy boolean subtract

Causa:
  x1/x0 entram como boolean/int misto em run_numeric_tests, e quantile()
  pode falhar em algumas versões do numpy/pandas.

O que faz:
  1. Cria backup do script atual.
  2. Injeta um helper seguro _numeric_float_clean().
  3. Troca as linhas x1 = ... e x0 = ... dentro de run_numeric_tests
     por chamadas ao helper, sem depender do formato exato do código anterior.
"""

from pathlib import Path
import re

SCRIPT = Path("scripts/exp_013a_statistical_fp_tp_diagnostics.py")

if not SCRIPT.exists():
    raise FileNotFoundError(f"Script não encontrado: {SCRIPT}")

txt = SCRIPT.read_text(encoding="utf-8")
original = txt

backup = SCRIPT.with_suffix(".py.bak_exp013a_r2")
backup.write_text(original, encoding="utf-8")

# 1) Injeta helper depois das imports, se ainda não existir.
helper = 