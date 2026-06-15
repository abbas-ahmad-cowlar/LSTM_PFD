# PROJECT STATE — session handoff & cold-start prompt

> **READ THIS FIRST, FULLY.** This is the working memory of the LSTM-PFD effort,
> written as a cold-start prompt for a fresh AI session. It tells you who the
> owner is and how to work with him, the compute setup and its quirks, the full
> honest history with numbers, the **current state and remediation plan**, what
> you must do next, and **what must never happen**. After this file, read the
> docs and code listed in §0 before acting.
>
> **Maintenance duty:** update this file at every phase gate and at the end of
> every session. Keep it truthful and current; it is the single source of truth.
>
> **Last updated: 2026-06-16, session 7.** One-line status: Phase 5 merged to
> `main` (Gate 5), **but** internal + independent external audits found the
> *physics-informed-model* evidence invalid (wrong-bearing-type physics, an inert
> loss, window-level statistics, mislabeled rows). We are deep into a 5-step
> **remediation** on branch `p6/docs`. **Steps 1–4 DONE; Step 5 reruns DONE
> (window-level); a record-level recompute of those reruns is RUNNING right now**
> (background python, PID-orphaned, writing `results/phase5_bandenergy/summary_record_level.json`).
> **The big new thing:** after the physics loss was finally implemented correctly
> (band-energy vs a *frozen healthy-class reference*, per-sample rpm), the reruns
> show — at window level, 3 seeds, **not yet a claim** — **no clean-accuracy gain
> BUT strong stable noise robustness at high physics weight (§8.4 w=1.0: 0.51 pt
> drop at 5 dB vs 4.99 CE-only / 1.70 vanilla) and a +5.6 severity-OOD-to-incipient
> gain (§8.3 dir B)**. This is a *promising reversal* of the earlier "no physics
> advantage" negative — it must survive the record-level recompute + McNemar
> before it becomes a finding. Suite: **261 passed, 6 deselected.**

---

## 0. If you are a fresh session, do this first

**Read, in this order (do not skip):**
1. This whole file.
2. `results/FINDINGS.md` — **read §0 first; it overrides §1–§5** (which overclaim
   and are kept only for provenance). §0 reflects the corrected, record-level
   benchmark verdict; it does **not** yet include the band-energy rerun result.
3. `results/phase5_bandenergy/findings_bandenergy.md` — the band-energy rerun
   result (window-level; the promising noise/OOD signal). **This is the newest
   science.**
4. `audit_reports/INDEPENDENT_SCIENCE_AUDIT_2026-06-14.md` — external auditor;
   authority on the *corrected* blast radius.
5. `audit_reports/PHYSICS_LOSS_AUDIT_2026-06-14.md` — internal audit + remediation
   plan (§6 = expanded scope + endorsed sequence).
6. `README.md` (reconciled overview), `experiments/PROTOCOL.md` (§7 amendments
   §8.0-bis…§8.0-quinquies + §8 pre-registrations), `docs/PHYSICS.md` (normative
   physics — the ground truth the model-side physics MUST match),
   `results/README.md` (results index), `CONVERGENCE_PLAN.md` (phase tracker).

**Then read these CODE files to understand the machinery (do NOT trust docstrings
— verify by execution):**
- Generator (ground-truth physics): `data/signal_generation/{fault_modeler,generator}.py`,
  `config/data_config.py`.
- Model-side physics (the remediation core):
  - `packages/core/models/physics/fault_signatures.py` — rebuilt journal-bearing
    signature DB + `load_healthy_reference()`.
  - `packages/core/models/physics/healthy_reference.json` — the **frozen
    healthy-class band-energy reference** (committed artifact).
  - `packages/core/models/pinn/physics_constrained_cnn.py` — `compute_physics_loss`
    is now the **band-energy-vs-healthy-reference** loss (per-sample rpm,
    full-window FFT). THE live physics loss.
  - `packages/core/models/physics/bearing_dynamics.py` (⚠ rolling-element SKF-6205
    defaults — quarantined), `packages/core/models/pinn/hybrid_pinn.py`
    (⚠ rolling-element physics branch — quarantined for physics claims),
    `packages/core/training/{physics_loss_functions,pinn_trainer}.py`
    (⚠ generic inert loss — quarantined; only the model-method loss was fixed).
- Experiments/stats: `scripts/run_phase5_gpu.py` (the §8.2/§8.3/§8.4 runner;
  `ops=True` wires per-sample rpm), `scripts/run_benchmark.py` (pure CE),
  `scripts/aggregate_benchmark_record_level.py` (the record-level method, Step 2),
  `scripts/phase5_bandenergy_record_level.py` (record-level for the reruns —
  **currently running**), `scripts/compute_healthy_reference.py`,
  `scripts/audit_physics_penalties.py` (flat-vs-healthy before/after), `data/dataset.py`
  (`WindowedView.record_index`).
- Tests: `tests/test_physics_signatures.py` (34 generator-physics),
  `tests/test_signature_db_consistency.py` (DB↔data vs the healthy reference),
  `tests/test_physics_band_energy_loss.py` (band-energy loss contract),
  `tests/test_physics_quarantine.py` (pins the inert/stale paths).

**Then check live state & trust:**
- `git branch --show-current` → **`p6/docs`** (remediation branch; `main` holds
  Gate-5 Phase 5 @`bb67026`). Branch is ~10 commits ahead of session-5; all pushed
  to `origin/p6/docs`. **NOT merged to main.**
- `$env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe -m pytest -q`
  → expect **261 passed, 6 deselected**.
- **Check the running record-level job:** `Test-Path results/phase5_bandenergy/summary_record_level.json`.
  If absent, it's still running (log `logs/phase5_be_recordlevel.log`; cache
  `results/phase5_bandenergy/_record_cache/*.npy` climbing toward ~60). If it died,
  re-run `python scripts/phase5_bandenergy_record_level.py` (resume-safe via cache).
- Never state a number without `(evidence: command → artifact)`. Verify by
  execution, not by reading.

**Then do the work (current step):** finish **Step 5** — (a) collect the
record-level recompute when the job lands, confirm/retract the window-level
noise/OOD signal with McNemar; (b) commit `scripts/phase5_bandenergy_record_level.py`
+ the summary; (c) recompute **§8.6a XAI** against the corrected bands; (d) then
**rewrite + re-ratify FINDINGS** with the owner. Do not loosen wording (§6); do
not claim any physics benefit until the record-level numbers support it.

---

## 1. The owner — who you're working with

**Syed Abbas Ahmad** (git `abbas-ahmad-cowlar`; physics background — comfortable
with physics reasoning; ratified the fault-model equations, PROTOCOL, and the
band-energy/healthy-reference loss personally). Working agreement:

- **Decision-maker at gates.** Owner-gated steps need his explicit sign-off
  (he ratified: tier system, PHYSICS.md, DATASET_V2.md, PROTOCOL.md, the
  band-energy loss formulation, and the **healthy-reference correction** —
  "use the actual healthy-class spectral reference, not the flat/uniform
  spectrum"). Quote his ratification in the doc when given.
- **Deeply distrusts unverified claims** — this repo burned him with fabricated
  results. NEVER present a number without an artifact path. Verify by execution.
  He commissioned an **independent external audit** of our own work and caught us
  being too soft once. He also catches real bugs (he spotted that the loss's flat
  baseline disagreed with the CI test) — engage his physics reasoning honestly.
- **Has attachment to impressive features** ("my heart doesn't want to let go").
  Use the tier system, not arguments, if keep/cut tension reappears.
- **Runs the hands-on compute** (Colab) but is not a shell expert (hit the
  `ln -sfn` trap; copied a Drive folder cross-account and triple-nested it). Give
  exact copy-paste, platform-correct, cell-by-cell commands with expected output
  and a STOP condition.
- Works in stretches across days; long jobs must survive his absence.
- Asks good skeptical questions — answer with data and honest scenario analysis,
  **never reassurance**. When he says "don't soften it," don't.

## 2. Compute setup — exact, with quirks

### 2.1 Laptop (primary; where Claude Code runs)
Windows 11 Pro, repo `C:\Users\COWLAR\projects\lstm-pfd`, PowerShell + Git Bash.
venv: **Python 3.14.0, torch 2.9.1+cpu — NO GPU** (`requirements.lock.txt`).
Fine for eval/XAI/aggregation/record-level recompute; training is GPU-only.
**Quirks:** (a) cp1252 console — emoji/Unicode prints crash; set
`PYTHONIOENCODING=utf-8` and pass `encoding='utf-8'` to `write_text`;
(b) `torch.onnx.export` needs `dynamo=False`; (c) `logs/` gitignored;
(d) PowerShell here-strings break on `--%`/quotes — for commit messages write a
file then `git commit -F logs/commitmsg.txt` (the Bash tool's bash heredoc also
works). **Long jobs:** run detached and resume-safe; do NOT background a python
with `&` *inside* a backgrounded shell (orphans it — happened this session;
it survived by luck). Prefer one foreground-style background job that writes a
log + a completion artifact.

### 2.2 Google Colab (de-facto GPU; free Tesla T4, ~75× the laptop)
Sessions ephemeral (~12 h; idle disconnects kill the VM). Owner's Drive:
`MyDrive/lstm-pfd/` with the dataset at **`data/dataset_v2.h5`**.
**ACTIVE runbook: `experiments/COLAB_PHASE5_RERUN_RUNBOOK.md`** (clone `p6/docs`,
verify the loss+reference in-runtime, copy dataset, symlink-before-first-run,
GPU smoke, then the 42-run resume-safe queue). Workflow: clone → checkout branch →
copy dataset → **symlink `results/phase5` to a fresh Drive folder BEFORE the first
run**. Two hard-won gotchas baked into the runbook:
  - **The repo ships `results/phase5/` populated with the OLD inert runs**, so
    `ln -sfn <drive> results/phase5` lands the link *inside* it (I2 trap) and the
    queue skips everything as "complete". Fix: **`rm -rf results/phase5` first**,
    then symlink (verify the `->` arrow + 0 metrics.json).
  - **Cross-account Drive copy double/triple-nests** (`results_phase5_bandenergy/
    results_phase5_bandenergy/...`). Resume on a different account = locate the
    real root (`find /content/drive/MyDrive -path '*pinn_ablation/w0.1/seed0/metrics.json'`)
    and re-`ln -sfn` to its parent.
Results come home by Drive download → extract → **verify counts + provenance**.
Other runbooks (`COLAB_DATAEFF_RUNBOOK.md`, `COLAB_PHASE5_*`, `OFFICE_PC_RUNBOOK.md`)
are historical.

### 2.3 Office PC — never used; Colab proved sufficient. Assume no state there.

### 2.4 Archives
`D:\Libraries\` holds full result downloads WITH checkpoints (`.pth`, out of git):
`results_phase5-…` (45 inert "before"), `results_phase5_fixed_full-…` (9 fixed
§8.4), `results_phase5_dataeff_fixed` (21 fixed §8.2). The **band-energy reruns**
live in-repo at `results/phase5_bandenergy/` (42 metrics.json committed; the 42
checkpoints ~1.9 GB are on the laptop, gitignored via `results/**/*.pth` — keep
them for the record-level recompute, then archive to `D:\Libraries`). `results/`
keeps json/md only.

## 3. The big picture (framing — corrected, then re-opening)

A physics-based **synthetic-data** platform for **journal/hydrodynamic bearing**
fault diagnosis (11 classes: sain, desalignement, desequilibre, jeu, lubrification,
cavitation, usure, oilwhirl + 3 mixed). The durable asset is the physics-grounded
signal generator (`data/signal_generation/`; normative `docs/PHYSICS.md`; 34-test
spectral CI). Journal-bearing data is scarce publicly (literature is rolling-element:
CWRU/Paderborn) — the niche. Generator/taxonomy = established physics; **cite prior
art, don't claim novelty there.** Contributions: **C1** dataset/generator · **C2**
frozen benchmark (the backbone) · **C3** the physics-informed-learning assessment
(see below) · **C4** physics-consistent XAI (to be recomputed) · **C5** deployment.
Venue tier (honest): IEEE Access / Sensors / Measurement / workshop / arXiv —
**not** a top mechanical-systems venue. **Synthetic-only; no real-world validation
— state everywhere.**

**The C3 story is mid-pivot.** Through every *broken* version of the physics loss
(never-called → inert/argmax → tonal-only → flat-baseline) the verdict was "no
physics advantage, even harmful" and we were heading for a *decisive negative*
paper. Now that the loss is **correct** (journal-bearing band-energy judged vs the
real healthy-class reference, per-sample rpm), the reruns show **no clean-accuracy
gain but emerging benefits in noise and severity-OOD** — suggesting the earlier
negative was substantially an *artifact of the broken physics*. **If record-level
confirms,** C3 moves from "physics doesn't help" to "physics doesn't help clean
accuracy but earns a real noise-robustness + OOD benefit." Until then it is a
*promising signal*, not a claim.

**Origin story:** the 2026-06-11 audit (`audit_reports/PROJECT_AUDIT_2026-06-11.md`)
found fabricated results (a paper claiming 98.1% with zero experiments run),
~130K LOC mostly unvalidated, a broken PINN, 45 failing tests. Prime rule: **only
execution evidence counts.** Old code recoverable at tag `pre-convergence-2026-06`.

## 4. History — Gates 0–4 PASSED, merged to `main`

| Gate | What happened | Numbers |
|---|---|---|
| Audit 06-11 | fabrications inventoried | 45 failed/220 passed; results/ empty |
| 0 Ratify | tag `pre-convergence-2026-06`; fake-results purge; env lock | — |
| 1 Stabilize | HybridPINN forward fixed; data-gen tests; ONNX dynamo=False | suite 328 green; CNN1D v1 86.48% |
| 2 Prune | registry 81→11 honest keys; −34.5K LOC; dashboard frozen | suite 206 green |
| 3 Physics & data | PHYSICS.md ratified; 34 spectral CI; **dataset_v2.h5** (3,520 rec, 80/class/severity, record-split, SNR-20/10/5, leakage-checked, DVC) | CNN1D v2 90.53% |
| 4 Benchmark | PROTOCOL frozen; classical + deep 8×3; ensemble; deployment | table below |

**Phase-4 benchmark** (test acc %, 2,640 windows, mean±std/3 seeds;
`results/benchmark/summary.md`, now banner-marked) — **READ WITH §5 CAVEATS:**
`physics_constrained_cnn` was trained **CE-only** (architecture row); `multitask_pinn`
**not** multitask; significance was window-level (**now recomputed at record level,
Step 2**).

| Model | Window acc | Note |
|---|---|---|
| voting_ensemble | 96.48 | single-seed |
| resnet18 | 96.14±0.28 | strongest vanilla |
| cnn_lstm | 96.12±0.16 | |
| physics_constrained_cnn | 95.98±0.36 | **CE-only — not a physics result** |
| RandomForest | 94.61±0.05 | classical bar |
| cnn1d | 91.94±2.84 | |
| multitask_pinn | 90.28±0.64 | **not trained multitask** |
| hybrid_pinn | 90.04±0.51 | rolling-element branch (quarantined) |
| patchtst | 89.85±0.19 | |
| attention_cnn | 89.37±4.82 | 1 seed collapsed |

**Record-level (Step 2, `results/benchmark/summary_record_level.md`):** soft-voting
5 windows/record → near-ceiling (RF 98.74, top deep ~99, CE-only pc_cnn 98.99);
**no row shows a physics advantage** (best vanilla cnn_lstm ties CE-only pc_cnn to
the record, McNemar p=1). Deployment (C5): ResNet18→ONNX FP32 **13 ms/window CPU**;
INT8 4× smaller but 10–15× slower (honest negative).

## 5. CURRENT STATE — physics remediation (branch `p6/docs`)

### 5.1 The physics-loss saga (five discoveries, all in `compute_physics_loss`)
- **§8.0** — `physics_constrained_cnn.forward()` is a plain CNN; physics enters
  ONLY via `compute_physics_loss`, which Phase-4 never called → benchmark pc_cnn is
  physics-OFF.
- **§8.0-bis** — that loss was **non-differentiable** (argmax) → inert (byte-identical
  w-sweep). Made the model-method version differentiable (softmax-weighted); §8.4/§8.2
  reran — still no help.
- **§8.0-ter** — the signature DB encoded **ROLLING-ELEMENT** physics (BPFO/BPFI/BSF/FTF),
  wrong for journal bearings, with the 3 mixed classes unmapped. **Rebuilt** from
  PHYSICS.md §4 (all 11 classes) + `tests/test_signature_db_consistency.py`. Owner
  ratified a **band-energy** loss formulation.
- **§8.0-quater** — implemented band-energy, but first as **concentration vs a flat/
  uniform spectrum**. Per-sample rpm wired into §8.2/§8.3/§8.4 (`ops=True`).
- **§8.0-quinquies (owner correction)** — flat baseline let **healthy-shared energy
  masquerade as faults** (healthy carries 8.3% of energy in the 1-6 Hz lube band,
  1.3% in the 1X band → a healthy signal scored a *perfect* lubrication / strong
  imbalance match). Owner: **use the FROZEN HEALTHY-CLASS REFERENCE.** Now
  `pen_b = relu(1 − frac_b / H_ref[c][b])`, per band, averaged; `H_ref` =
  `healthy_reference.json` (1,120 healthy train windows; `scripts/compute_healthy_reference.py`).
  The CI test + the loss share this one reference. Before/after proof:
  `scripts/audit_physics_penalties.py` (masquerade penalties 0.000/0.198 → 0.326/0.573).

### 5.2 Corrected blast radius
**Sound / done:**
- Dataset v2 as a synthetic, internally-consistent benchmark (cavitation only
  weakly — qualify it). Generator independent of all broken physics.
- Benchmark accuracies as classification results, **relabeled** (Step 3) and
  **recomputed at record level** (Step 2). No physics advantage on clean accuracy.
- §8.1 noise (frozen-checkpoint eval), deployment C5.
- The **band-energy loss** is now correct & tested (Step 4); the **§8.2/§8.3/§8.4
  reruns** with it are done at window level (Step 5) — see §5.3/§5.4.

**Still invalid / pending:**
- **§8.5 HybridPINN** — physics branch still **rolling-element** (`BearingDynamics`
  SKF-6205). Quarantined; excluded from the reruns. Rebuild on journal-bearing
  features (1X/2X/3X, oil-whirl sub-sync, low-freq lube, cavitation HF, wear
  broadband) before any §8.5 claim — separate task, fresh pre-reg, I4 check.
- **§8.6 / C4 XAI + calibration** — still on old broken-DB bands / pre-correction
  checkpoints. Recompute against the corrected bands (Step 5c).
- **generic `PhysicalConstraintLoss`/`PINNTrainer`** — still inert; quarantined,
  not fixed (only the model-method loss was). Pinned by `test_physics_quarantine.py`.
- **Record-level significance of the band-energy reruns** — RUNNING now; until it
  lands, the noise/OOD signal is window-level only.

### 5.3 Remediation sequence (external-auditor-endorsed) — status
1. **Reconcile docs** — **DONE** (FINDINGS §0, this file, PROTOCOL §7, audit §6,
   README, results/README).
2. **Record-level statistics (benchmark)** — **DONE** (`scripts/aggregate_benchmark_record_level.py`
   → `results/benchmark/summary_record_level.{json,md}`; near-ceiling, no physics
   advantage on clean accuracy; window-level `summary.*` superseded).
3. **Quarantine/relabel** — **DONE** (benchmark `summary.*` relabeled + banner;
   quarantine docstrings on the inert loss path, HybridPINN rolling-element branch,
   `BearingDynamics`; `scripts/research/pinn_ablation.py` blocked;
   `test_physics_quarantine.py`).
4. **Band-energy loss + healthy reference** — **DONE** (§5.1 §8.0-quater/quinquies;
   `compute_physics_loss`, `healthy_reference.json`, aligned CI test,
   `test_physics_band_energy_loss.py`, per-sample rpm wiring).
5. **Rerun physics-forward experiments** — **reruns DONE 2026-06-15 (window-level);
   record-level recompute RUNNING; XAI + FINDINGS NEXT.** 42 runs (Colab T4, code
   @`ce344d1`) in `results/phase5_bandenergy/{pinn_ablation,data_efficiency,severity_ood}`
   (`findings_bandenergy.md`). Window-level result in §5.4. NOW:
   (a) `scripts/phase5_bandenergy_record_level.py` is running → confirm/retract via
   record-level acc + cluster-bootstrap + McNemar (w=0 vs w=1.0 @5dB; pc_cnn w=1.0
   vs resnet18 @5dB; dir B); **commit the script + summary when done.**
   (b) recompute **§8.6a XAI** against the corrected bands (laptop).
   (c) **rewrite + re-ratify FINDINGS** with the owner. §8.5 excluded.

### 5.4 Band-energy rerun numbers (window-level, 3 seeds — pending record-level)
`results/phase5_bandenergy/findings_bandenergy.md` has the tables. Highlights:
- **§8.4 ablation/noise:** clean ~96% flat across w (no clean cost). 5 dB:
  w=0 (CE-only) 91.00 (degr 4.99) · w=0.1 90.77 · w=0.3 89.46 · **w=1.0 95.64±0.22
  (degr 0.51, all 3 seeds ~95.5)** · vanilla resnet18 94.43 (degr 1.70). → strong,
  stable **noise robustness at high physics weight**; reverses the old contaminated
  "harmful at w=1.0 (83.1)".
- **§8.3 severity-OOD:** dir A (→ severe) tied ~97.3; **dir B (→ incipient) pc_cnn
  79.04±0.61 vs resnet18 73.43±3.29 (+5.6, low-var)**.
- **§8.2 data-eff:** ~neutral (50% non-overlapping win, 25% marginal, 10% slightly
  behind 92.60 vs 93.60, 100% tied) — much improved over the contaminated "harmful
  at 10% (91.11)".

**Old contaminated numbers (reference only — do NOT cite):** §8.2 fixed pc_cnn hurt
at 10% (91.11 vs 93.60); §8.4 fixed harmful at w=1.0 5 dB (83.1); §8.5 89.76 vs
90.04 (null, rolling-element); §8.6a ratio 0.849 vs 0.716; §8.6b ECE 0.022 vs 0.028.
(`results/phase5_fixed/`, `phase5_dataeff_fixed/`, `xai_alignment/`, `uncertainty/`.)

## 6. What must NOT happen (guardrails)
- **Do NOT loosen wording.** Never "the benchmark is sound" without "as a
  classification benchmark." Never "decisive negative." **Never claim a physics
  benefit (accuracy/noise/data-eff/OOD/XAI/calibration) until the RECORD-LEVEL
  recompute supports it** — the window-level noise/OOD signal is promising but NOT
  a claim. The record-level recompute + McNemar is the gate.
- **Do NOT treat the band-energy reruns as final** until record-level lands and is
  committed; do NOT rehabilitate §8.5 (HybridPINN rolling-element) or C4 XAI
  without first rebuilding/recomputing them.
- **Do NOT skip record-level statistics.** Window-level significance is invalid.
- **Do NOT present a number without an artifact path + provenance.** Verify by
  execution; honor the sanity-gate discipline (it caught a real cache bug this
  session).
- **Do NOT modify the generator/dataset/`healthy_reference.json` mid-analysis** —
  they are frozen. A change = new version + documented rerun + re-pre-registration.
- **Do NOT commit to `main`** — work on `p6/docs`; merge only at a gate with suite
  green and owner sign-off.
- **Do NOT skip the owner gate** on physics/protocol changes — dated §7 amendments;
  quote his ratification.
- **Keep the repo self-contained** — no references to any other/external project in
  committed files (owner instruction; provenance nuance lives only in private
  session memory).
- Respect laptop quirks (§2.1): UTF-8; commit via `-F`; don't `&`-orphan jobs.

## 7. Incidents & lessons (do not repeat)
- **I1** session-bound/`&`-orphaned process death → detached + resume-safe + a
  completion artifact.
- **I2** Colab `ln -sfn` into an existing (committed) `results/phase5` → `rm -rf`
  it first, then symlink BEFORE the first run.
- **I3** fake Drive mount (`mkdir` before `drive.mount`) → mount first.
- **I4** factory/class default desync broke checkpoint loading → after any
  model-code change, verify `create_model(key)` still loads recorded checkpoints
  (the sanity gate: argmax reproduces recorded window accuracy).
- **I5** cp1252 crashes on Unicode → UTF-8 everywhere (`encoding='utf-8'`).
- **I6** full-5 s records saturate (~100%) → 1 s windowing.
- **I7 (the big physics lesson)** a model named "physics_*" proves nothing: verify
  the physics is (a) wired into training, (b) differentiable, (c) the correct
  bearing type, (d) consistent with the generator, AND (e) judged against the
  right baseline (the **real healthy class**, not a flat spectrum). Five separate
  defects all passed a green suite. Tests must assert gradient flow + DB↔data
  consistency, not just "runs."
- **I8 (statistics)** windowed data → record-level statistics (2,640 windows from
  528 records are NOT independent).
- **I9 (Colab Drive)** cross-account folder copies double/triple-nest; always
  locate the real run-root by a known metrics.json path before resuming.
- **I10 (caching)** a result-cache key must include the eval SPLIT, not just
  checkpoint+record-count — clean and 5 dB collided this session; the sanity gate
  caught it. Cache keys must encode every axis that changes the output.

## 8. Conventions (the honesty machinery)
1. Only execution evidence counts; numbers trace to `results/` with SHA + host + seed.
2. `CONVERGENCE_PLAN.md` checkboxes need `(evidence: …)`; tracker updated each session.
3. PROTOCOL frozen — dated §7 amendments; §8 pre-registrations before running.
4. Phase branches `pN/...` → `main` at gates. Current: **`p6/docs`** (pushed, not merged).
5. Suite green before merge: `pytest -q` → **261 passed, 6 deselected** (dashboard
   frozen until Phase D).
6. Anti-regrowth: fixed-size tiers; cut at the tag + `BACKLOG.md`.
7. `results/`: small json/md/png/csv committed; **`*.pth` checkpoints + h5 out of
   git** (`.gitignore: results/**/*.pth`; DVC for the dataset; archives in `D:\Libraries`).
8. External-audit discipline: neutral prompts (`audit_reports/INDEPENDENT_AUDIT_PROMPT.md`);
   their reports are authoritative on scope.

## 9. Key files map
| What | Where |
|---|---|
| **This handoff** | `PROJECT_STATE.md` |
| Findings (read §0 first) | `results/FINDINGS.md` |
| **Band-energy rerun findings (newest)** | `results/phase5_bandenergy/findings_bandenergy.md` |
| External audit (authoritative on scope) | `audit_reports/INDEPENDENT_SCIENCE_AUDIT_2026-06-14.md` |
| Internal physics audit + plan | `audit_reports/PHYSICS_LOSS_AUDIT_2026-06-14.md` |
| Origin audit | `audit_reports/PROJECT_AUDIT_2026-06-11.md` |
| Physics (normative) + CI | `docs/PHYSICS.md`, `tests/test_physics_signatures.py` |
| Protocol (§8.0-bis…quinquies) | `experiments/PROTOCOL.md` |
| Dataset v2 | `experiments/DATASET_V2.md`, `data/generated/dataset_v2.h5` (+DVC) |
| **Frozen healthy reference** | `packages/core/models/physics/healthy_reference.json` |
| Model-side physics | `packages/core/models/physics/{bearing_dynamics,fault_signatures}.py`, `.../pinn/{physics_constrained_cnn,hybrid_pinn,multitask_pinn}.py`, `.../training/{physics_loss_functions,pinn_trainer}.py` |
| Benchmark results (window + record level) | `results/benchmark/summary.md`, `summary_record_level.md` |
| **Band-energy reruns** | `results/phase5_bandenergy/{pinn_ablation,data_efficiency,severity_ood}/`, `summary_record_level.json` (being written) |
| Phase-5 (old, contaminated) | `results/phase5/`, `phase5_fixed/`, `phase5_dataeff_fixed/`, `noise_robustness/`, `xai_alignment/`, `uncertainty/` |
| Scripts (new this remediation) | `scripts/{compute_healthy_reference,audit_physics_penalties,aggregate_benchmark_record_level,phase5_bandenergy_record_level}.py` |
| Runners | `scripts/{run_benchmark,aggregate_benchmark,run_phase5_gpu,run_xai_calibration}.py` |
| **Active Colab runbook** | `experiments/COLAB_PHASE5_RERUN_RUNBOOK.md` |
| Tests (physics) | `tests/test_{physics_signatures,signature_db_consistency,physics_band_energy_loss,physics_quarantine}.py` |
| Deferred / frozen | `BACKLOG.md` · dashboard `packages/dashboard/` |
