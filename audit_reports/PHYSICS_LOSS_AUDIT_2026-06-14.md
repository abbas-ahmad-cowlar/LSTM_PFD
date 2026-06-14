# Physics-loss audit & remediation plan — 2026-06-14

> Triggered by the owner after `scripts/verify_physics_consistency.py` showed the
> physics loss expects frequencies that don't match the data. Full trace of the
> defect, its blast radius (by execution, not docs), what is sound, and the
> remediation. **Standard: nothing factually wrong, inconsistent, or misleading
> may remain.**

## 1. The defect (3 layers, in `physics_constrained_cnn.compute_physics_loss`)

The loss relies on `FaultSignatureDatabase` (`packages/core/models/physics/fault_signatures.py`):

1. **[FIXED already]** It was non-differentiable (argmax) → `requires_grad=False`,
   zero gradient. Corrected to a softmax-weighted penalty (PROTOCOL §7 amendment).
2. **Wrong bearing type.** The DB defines **rolling-element** signatures —
   `outer_race`, `inner_race`, `ball` with `BPFO/BPFI/BSF/FTF` frequencies — which
   are **physically meaningless for journal/hydrodynamic bearings**. Via
   `FAULT_LABELS_PINN` (`utils/constants.py:94`), journal class 6 `usure` (wear) →
   `'wear'` signature = `BPFO/BPFI/FTF` → the loss checked rolling-element pass
   frequencies for a journal fault. `outer_race/inner_race/ball` signatures are
   never reached (no journal class maps to them — dead entries).
3. **Mixed faults unmapped.** Classes 8–10 map to `'combined_*'` names that **do
   not exist** in the signature dict → `KeyError` → the loss's `except: continue`
   silently contributed **zero** physics constraint for all 3 mixed classes.

Plus the earlier-noted **broadband mismatch**: cavitation/wear/lubrication are
broadband/impulsive by design, but the DB encodes narrow expected peaks.

**Root cause:** the signature DB was a generic rolling-element artifact, **never
validated against this project's journal-bearing generator.** No CI linked the
physics the *models* assume to the physics the *data* contains — so it drifted
unnoticed (this is the 3rd physics-loss defect found, all in the same module).

## 2. Blast radius (verified by execution)

The ONLY live consumer of the broken DB is `physics_constrained_cnn.compute_physics_loss`
(grep: only this model imports `FaultSignatureDatabase`). hybrid_pinn / multitask_pinn
do NOT use it. `pinn_trainer.py`, `physics_loss_functions.py`, `pinn_evaluator.py`,
`physics_interpretability.py` reference it but are imported by **no** experiment
script (dead w.r.t. results).

| Artifact | Uses broken physics loss/DB? | Status |
|---|---|---|
| **Generator + Dataset v2** | NO — `fault_modeler.py`/`generator.py` have own physics, no DB import; 34-test CI | **SOUND** |
| **Benchmark C2** (`results/benchmark/`) | NO — pure `CrossEntropyLoss` (`run_benchmark.py:127`); pc_cnn was physics-OFF (§8.0) | **SOUND** |
| **§8.1 noise** | NO — eval of frozen checkpoints | **SOUND** |
| **§8.3 severity-OOD** | pc_cnn ran physics-OFF (inert) → was architecture-only | sound *as reported* (never a physics test) |
| **§8.5 hybrid_pinn metadata** | NO — forward-path metadata, no DB | **SOUND** |
| **§8.6b calibration** | NO DB; uses pc_cnn ckpt trained w/ bad loss | mild taint (checkpoint only) |
| **Deployment C5** | NO | **SOUND** |
| **§8.2 data-efficiency (pc_cnn FIXED arm)** | YES | **CONTAMINATED — re-run** |
| **§8.4 ablation (FIXED)** | YES | **CONTAMINATED — re-run** |
| **§8.6a XAI alignment** | YES — alignment bands come from the broken DB | **CONTAMINATED — recompute** |

**The paper's backbone — dataset (C1), frozen benchmark (C2), noise, deployment,
§8.5 — is provably independent of the bug and sound.** The contamination is
confined to the three physics-loss-based Phase-5 results.

## 3. Remediation plan

**Tier A — re-confirm the foundation (cheap, high-value):**
- Extend `verify_physics_consistency.py` to ALL 11 classes (incl. broadband +
  mixed) and assert the **generator's** output matches `docs/PHYSICS.md`. If the
  generator is sound (expected — it's CI-tested and independent), the dataset,
  benchmark, noise, deployment, and §8.5 **stand as-is, no rerun.**

**Tier B — fix the physics-loss layer (the real work):**
1. **Rebuild `FaultSignatureDatabase` from the generator's journal-bearing
   physics** (`fault_modeler.py` + `docs/PHYSICS.md` are ground truth), for all 11
   classes including mixed. Remove the rolling-element entries.
2. **Add a CI test** asserting DB ↔ generator consistency for every class — so the
   physics the models assume can never again diverge from the physics the data has
   (the systemic fix; same discipline as the 34 generator tests).
3. **Decide the physics-loss formulation** for broadband/impulsive faults
   (band-energy / envelope / kurtosis term, not just narrow peaks) — pre-register
   as a PROTOCOL §7 amendment.

**Tier C — re-run only the contaminated experiments** with the corrected DB:
§8.4 (ablation, 9 runs), §8.2 pc_cnn arm (9 runs), §8.6a (recompute, laptop).
Optional: §8.3 pc_cnn now WITH correct physics. Then **update FINDINGS** (the C3
verdict may hold or change — reported honestly either way; C4 alignment recomputed
against correct bands).

**Tier D — quarantine dead/wrong physics modules:** `physics_loss_functions.py`,
`pinn_trainer.py` physics path, `pinn_evaluator.py`, `physics_interpretability.py`
reference the same broken DB and are unused — fix or clearly mark deprecated so
they cannot mislead a future reader.

## 4. Recommendation

**Scoped remediation (Tiers A–D), NOT a full protocol rebuild.** Deleting the
benchmark/dataset would destroy work that is *provably* independent of this bug
(§2) — that is waste, not rigor. Tiers A–D meet the standard "nothing wrong or
misleading remains": the foundation is re-confirmed, the broken layer is rebuilt
+ CI-locked, the contaminated results are re-run, and the dead wrong code is
quarantined. Estimated: ~1 day laptop+Colab (DB rebuild + CI + ~18 GPU runs +
recompute + FINDINGS). A from-scratch protocol rebuild would cost weeks and
re-derive identical benchmark/dataset numbers.
