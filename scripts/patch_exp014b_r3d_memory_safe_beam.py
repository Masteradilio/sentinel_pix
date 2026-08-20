#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Patch EXP-014B-R3D — Memory-Safe Beam Search

Corrige o erro:

  numpy._core._exceptions._ArrayMemoryError:
  Unable to allocate 111 KiB for an array with shape (113844,) and data type bool

Causa:
  O beam search estava acumulando muitas máscaras booleanas grandes em
  next_states antes de podar para beam_width. Com max_candidates=800 e
  beam_width=250, o depth 3 pode gerar centenas de milhares de estados,
  consumindo muita RAM.

Correção:
  1. Substitui a função search_best_vetos por uma versão memory-safe.
  2. Poda next_states durante a construção, não só no final do depth.
  3. Quando tp_budget=0, usa modo TP0:
       - não recalcula ((y == 1) & mask).sum()
       - fp_removed = mask.sum()
  4. Quando tp_budget>0, calcula tp_loss só nos índices fraudulentos:
       - tp_loss = mask[fraud_idx].sum()
       - fp_removed = mask.sum() - tp_loss
     evitando alocar arrays auxiliares de 113.844 posições.
  5. Mantém checkpoints e KeyboardInterrupt seguro.

Uso:
  python scripts\\patch_exp014b_r3d_memory_safe_beam.py

Depois, rode novamente o R3D. Sugestão segura:
  python scripts\\exp_014b_r3d_expanded_champion_anchored_fp_reducer.py --max-combo-size 4 --max-candidates 800 --beam-width 250 --max-rules 10 --max-seconds 900 --bootstrap-iters 100

Se quiser primeiro uma confirmação rápida:
  python scripts\\exp_014b_r3d_expanded_champion_anchored_fp_reducer.py --max-combo-size 4 --max-candidates 500 --beam-width 120 --max-rules 4 --max-seconds 300 --bootstrap-iters 50
"""

from pathlib import Path

SCRIPT = Path("scripts/exp_014b_r3d_expanded_champion_anchored_fp_reducer.py")

if not SCRIPT.exists():
    raise FileNotFoundError(f"Não encontrei {SCRIPT}. Execute na raiz do projeto rebuild_pix.")

text = SCRIPT.read_text(encoding="utf-8")

start = text.find("\ndef search_best_vetos(")
if start == -1:
    start = text.find("def search_best_vetos(")

end = text.find("\ndef wilson_ci(", start)
if start == -1 or end == -1:
    raise RuntimeError("Não consegui localizar def search_best_vetos(...) até def wilson_ci(...).")

new_func = """
def search_best_vetos(
    cands: list[VetoCandidate],
    base_pred: np.ndarray,
    y: np.ndarray,
    tp_budget: int,
    max_candidates: int,
    beam_width: int,
    max_rules: int,
    max_seconds: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, State, list[VetoCandidate], str]:
    \"""
    Beam search memory-safe.

    Diferenças contra a versão original:
      - poda next_states durante o depth;
      - evita alocações do tipo ((y == 1) & new_mask);
      - quando tp_budget=0 e candidatos são TP0, union também é TP0.
    \"""
    t0 = time.perf_counter()

    usable = [c for c in cands if c.tp_loss <= tp_budget]
    usable.sort(key=lambda c: (c.tp_loss > 0, -c.fp_removed if c.tp_loss == 0 else -c.fp_per_tp, -c.fp_removed))
    usable = usable[:max_candidates]

    fraud_idx = np.where(y == 1)[0]
    zero_loss_mode = (tp_budget == 0 and all(c.tp_loss == 0 for c in usable))

    # Limites internos para impedir explosão de memória.
    pending_limit = max(beam_width * 8, 1000)
    pending_keep = max(beam_width * 4, 500)

    def rank_state(s: State):
        return (s.fp_removed, -s.tp_loss, -len(s.rule_indices))

    def prune_pending(d: dict[bytes, State], keep: int) -> dict[bytes, State]:
        if len(d) <= keep:
            return d
        items = sorted(d.items(), key=lambda kv: rank_state(kv[1]), reverse=True)[:keep]
        return dict(items)

    zero = np.zeros(len(y), dtype=bool)
    initial = State(zero, tuple(), 0, 0)
    states = [initial]
    best = initial
    rows = []
    stop_reason = "completed"

    try:
        for depth in range(1, max_rules + 1):
            elapsed = time.perf_counter() - t0
            if elapsed >= max_seconds:
                stop_reason = f"max_seconds_before_depth_{depth}"
                break

            next_states: dict[bytes, State] = {}
            depth_t0 = time.perf_counter()
            expansions = 0
            prunes = 0

            for state in states:
                last = state.rule_indices[-1] if state.rule_indices else -1
                old_total_removed = state.tp_loss + state.fp_removed

                for i in range(last + 1, len(usable)):
                    c = usable[i]
                    new_mask = state.mask | c.mask
                    new_total_removed = int(new_mask.sum())

                    # Se a união não adicionou nada, descarte.
                    if new_total_removed <= old_total_removed:
                        continue

                    if zero_loss_mode:
                        tp_loss = 0
                        fp_removed = new_total_removed
                    else:
                        # Computa TP loss só nos índices positivos para evitar array temporário grande.
                        tp_loss = int(new_mask[fraud_idx].sum()) if len(fraud_idx) else 0
                        if tp_loss > tp_budget:
                            continue
                        fp_removed = new_total_removed - tp_loss

                    if fp_removed <= state.fp_removed:
                        continue

                    key = np.packbits(new_mask).tobytes()
                    ns = State(new_mask, state.rule_indices + (i,), tp_loss, fp_removed)
                    old = next_states.get(key)
                    if old is None or rank_state(ns) > rank_state(old):
                        next_states[key] = ns

                    expansions += 1
                    if len(next_states) > pending_limit:
                        next_states = prune_pending(next_states, pending_keep)
                        prunes += 1

                # Checagem de tempo também dentro do loop externo.
                if time.perf_counter() - t0 >= max_seconds:
                    stop_reason = f"max_seconds_during_depth_{depth}"
                    break

            if not next_states:
                if stop_reason.startswith("max_seconds"):
                    break
                stop_reason = f"no_next_states_at_depth_{depth}"
                break

            states = sorted(next_states.values(), key=rank_state, reverse=True)[:beam_width]

            if rank_state(states[0]) > rank_state(best):
                best = states[0]

            for s in states[:50]:
                pred = base_pred.copy()
                pred[s.mask] = 0
                m = compute_metrics(y, pred)
                rows.append({
                    "depth": depth,
                    "tp_loss": s.tp_loss,
                    "fp_removed": s.fp_removed,
                    "n_rules": len(s.rule_indices),
                    **m,
                    "rule_ids": "|".join(usable[i].rule_id for i in s.rule_indices),
                    "rule_descriptions": " || ".join(usable[i].description for i in s.rule_indices),
                })

            pd.DataFrame(rows).to_csv(output_dir / f"checkpoint_frontier_depth_{depth:02d}.csv", index=False)

            log(
                f"  depth={depth}: best_fp_removed={best.fp_removed}, "
                f"tp_loss={best.tp_loss}/{tp_budget}, states={len(states)}, "
                f"expansions={expansions}, prunes={prunes}, "
                f"depth_s={time.perf_counter()-depth_t0:.1f}"
            )

            if time.perf_counter() - t0 >= max_seconds:
                stop_reason = f"max_seconds_after_depth_{depth}"
                break

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt_saved_best"
        log("KeyboardInterrupt capturado; salvando melhor estado.")

    if not rows:
        m = compute_metrics(y, base_pred)
        rows = [{
            "depth": 0,
            "tp_loss": 0,
            "fp_removed": 0,
            "n_rules": 0,
            **m,
            "rule_ids": "",
            "rule_descriptions": "",
        }]

    frontier = pd.DataFrame(rows).sort_values(["fp", "tp"], ascending=[True, False]).reset_index(drop=True)
    selected = [usable[i] for i in best.rule_indices]
    return frontier, best, selected, stop_reason

"""

text = text[:start] + "\n" + new_func + text[end:]

SCRIPT.write_text(text, encoding="utf-8", newline="\n")
print(f"Patch aplicado com sucesso em {SCRIPT}")
