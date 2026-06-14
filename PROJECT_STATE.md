# PROJECT STATE — session handoff & cold-start prompt

> **READ THIS FIRST, FULLY.** This is the working memory of the LSTM-PFD effort,
> written as a cold-start prompt for a fresh AI session. It tells you who the
> owner is and how to work with him, the compute setup and its quirks, the full
> honest history with numbers, the **current crisis and remediation plan**, what
> you must do next, and **what must never happen**. After this file, read the
> docs and code listed in §0 before acting.
>
> **Maintenance duty:** update this file at every phase gate and at the end of
> every session. Keep it truthful and current; it is the single source of truth.
>
> **Last updated: 2026-06-14, session 5.** Status in one line: Phase 5 finished
> and merged to `main` (Gate 5), **but** an internal audit then an independent
> external audit found the project's *physics-informed-model* evidence is
> invalid (wrong-bearing-type physics, an inert loss, window-level statistics,
> mislabeled rows). We are mid-**remediation** on branch `p6/docs`. The
> dataset/benchmark survive *as a synthetic classification benchmark*; the
> physics-model claims do not. **Current step: Step 2 of the 5-step remediation
> (recompute all statistics at record level).**

---

## 0. If you are a fresh session, do this first

**Read, in this order (do not skip):**
1. This whole file.
2. `results/FINDINGS.md` — **read §0 first; it is authoritative and supersedes
   §1–§5 of that file** (which overclaim and are kept only for provenance).
3. `audit_reports/INDEPENDENT_SCIENCE_AUDIT_2026-06-14.md` — the external
   auditor's report; it is the authority on the *corrected* blast radius.
4. `audit_reports/PHYSICS_LOSS_AUDIT_2026-06-14.md` — our internal audit + the
   remediation plan (§6 has the expanded scope and the endorsed sequence).
5. `README.md` (honest overview), `experiments/PROTOCOL.md` (§7 amendments +
   §8 pre-registrations), `docs/PHYSICS.md` (normative physics — the ground
   truth the model-side physics MUST match), `CONVERGENCE_PLAN.md` (Phase-6
   banner + tracker), `results/README.md` (results index).

**Then read these CODE files to actually understand the machinery (do not trust
docstrings — verify):**
- Generator (ground-truth physics): `data/signal_generation/fault_modeler.py`,
  `data/signal_generation/generator.py`, `config/data_config.py`.
- Physics used by models: `packages/core/models/physics/bearing_dynamics.py`
  (⚠ rolling-element SKF-6205 defaults — the root rot), `fault_signatures.py`
  (rebuilt journal-bearing DB), `packages/core/models/pinn/physics_constrained_cnn.py`
  (the `compute_physics_loss` — currently tonal-only, needs band-energy),
  `packages/core/models/pinn/hybrid_pinn.py` (⚠ rolling-element physics branch),
  `packages/core/training/physics_loss_functions.py` + `pinn_trainer.py`
  (⚠ still-inert generic loss).
- Experiments/stats: `scripts/run_phase5_gpu.py`, `scripts/run_benchmark.py`
  (pure CE — mislabeled "physics" rows), `scripts/aggregate_benchmark.py`
  (⚠ window-level stats — must become record-level), `data/dataset.py`
  (`WindowedView`), `scripts/run_xai_calibration.py`,
  `scripts/verify_physics_consistency.py`.
- Tests: `tests/test_physics_signatures.py` (34 generator-physics tests),
  `tests/test_signature_db_consistency.py` (DB↔data guard, new).

**Then check live state & trust:**
- `git branch --show-current` → expect **`p6/docs`** (the remediation branch;
  `main` holds Gate-5-merged Phase 5 @`bb67026`).
- `$env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe -m pytest -q`
  → expect **251 passed, 6 deselected**.
- Never mark a plan checkbox or state a number without `(evidence: command →
  artifact)`. Verify by execution, not by reading.

**Then do the work:** continue the remediation at **Step 2** (§5). Do not jump
ahead; do not loosen wording (§6).

---

## 1. The owner — who you're working with

**Syed Abbas Ahmad** (git `abbas-ahmad-cowlar`; physics background — comfortable
with physics reasoning; ratified the fault-model equations and PROTOCOL
personally). Working agreement, learned over this project:

- **Decision-maker at gates.** Owner-gated steps need his explicit sign-off
  (he ratified: tier system, PHYSICS.md, DATASET_V2.md, PROTOCOL.md, the
  band-energy loss formulation). Quote his ratification in the doc when given.
- **Deeply distrusts unverified claims** — this repo burned him with fabricated
  results before. NEVER present a number without an artifact path. He mandated:
  verify by execution, not by reading code. He commissioned an **independent
  external audit** of our own work — he wants the unvarnished truth, including
  our mistakes, and he caught us being too soft once already.
- **Has attachment to impressive features** ("my heart doesn't want to let go").
  If keep/cut tension reappears, use the tier system, not arguments.
- **Runs the hands-on compute** (Colab, office PC) but is not a shell expert
  (pasted Windows paths into bash, ran cells without `!`, hit the `ln -sfn`
  trap). Give him exact copy-paste, platform-correct commands with expected
  output. Runbooks must be cell-by-cell.
- Works in stretches across days; long jobs must survive his absence.
- Asks good skeptical questions — answer with data and honest scenario analysis,
  **never reassurance**. When he says "don't soften it," don't.

## 2. Compute setup — exact, with quirks

### 2.1 Laptop (primary; where Claude Code runs)
Windows 11 Pro, repo `C:\Users\COWLAR\projects\lstm-pfd`, PowerShell + Git Bash.
venv: **Python 3.14.0, torch 2.9.1+cpu — NO GPU** (`requirements.lock.txt`).
Fine for eval/XAI/classical/aggregation; training is overnight-only.
**Quirks:** (a) cp1252 console — emoji/Unicode prints crash; set
`PYTHONIOENCODING=utf-8` and pass `encoding='utf-8'` to `write_text`;
(b) `torch.onnx.export` needs `dynamo=False`; (c) `logs/` is gitignored (scripts
mkdir it); (d) PowerShell here-strings break on `--%`/inner quotes — for git
commit messages use a file: `git commit -F logs/commitmsg.txt`.
**Detached-launch rule (hard lesson, ~8 h lost):** session-bound background jobs
die with the Claude session. For long jobs use
`Start-Process <venv python> -ArgumentList <script> -WindowStyle Hidden -RedirectStandardOutput logs\x.log`.
All training/eval scripts are resume-safe (metrics.json marks completion).

### 2.2 Google Colab (de-facto GPU; free Tesla T4, ~75× the laptop)
Sessions ephemeral (~12 h, idle disconnects kill the VM). Owner's Drive:
`MyDrive/lstm-pfd/` with the dataset at **`data/dataset_v2.h5`** and result
folders (`results_benchmark`, `results_phase5`, `results_phase5_fixed`,
`results_phase5_dataeff_fixed`). Workflow: clone → checkout branch → copy dataset
to local disk → **symlink `results/<dir>` to Drive BEFORE the first run** (the
`ln -sfn`-into-existing-dir trap, I2) → run; rerun the same cell to resume.
Results come home by Drive download → extract into `results/` → **verify counts +
provenance before using**. Downloads arrive nested (`<name>-<timestamp>/<name>/…`)
— relocate to the exact symlink target or the queue restarts. **Active runbook:
`experiments/COLAB_DATAEFF_RUNBOOK.md`** (self-contained, needs only the
dataset). Others (`COLAB_PHASE5_RUNBOOK.md`, `COLAB_PHASE5_FIXED_RUNBOOK.md`,
`OFFICE_PC_RUNBOOK.md`) are historical.

### 2.3 Office PC — never used; Colab proved sufficient. Assume no state there.

### 2.4 Archives
`D:\Libraries\` holds the full result downloads WITH checkpoints (`.pth`, kept
out of git): `results_phase5-…T100807Z…` (45 inert "before"),
`results_phase5_fixed_full-…T221621Z…` (9 fixed §8.4),
`results_phase5_dataeff_fixed` (21 fixed §8.2). `results/` keeps json/md only.

## 3. The big picture (framing CORRECTED post-audit)

A physics-based **synthetic-data** research platform for **journal/hydrodynamic
bearing** fault diagnosis (11 classes: sain, desalignement, desequilibre, jeu,
lubrification, cavitation, usure, oilwhirl + 3 mixed). The asset is the
physics-grounded signal generator (`data/signal_generation/`; normative
`docs/PHYSICS.md`; 34-test spectral CI in `tests/test_physics_signatures.py`).
Journal-bearing data is scarce publicly (literature is rolling-element:
CWRU/Paderborn) — that's the niche. The generator/taxonomy is built on
established bearing-fault physics — **do not claim it as a novel contribution;
cite prior art.**

**Original aspiration (NOW LARGELY REFUTED):** "show where physics-informed
learning beats data-driven (less data, OOD, noise) with physics-consistent XAI."
**Honest current framing:** a reproducible synthetic journal-bearing
**classification benchmark** + a *rigorous, honest assessment* of whether
physics-informed learning helps — and so far, on this clean synthetic data, the
stored results show **no physics advantage**. Contributions, reweighted:
**C1** dataset/generator · **C2** frozen benchmark (the backbone) · **C3** an
honest *negative* on physics-informed accuracy (pending valid reruns) · **C4**
physics-consistent XAI (currently contaminated — see §5) · **C5** deployment.
Venue tier (honest): IEEE Access / Sensors / Measurement special issue /
workshop / arXiv — **not** a top mechanical-systems venue. **Synthetic-only; no
real-world validation — state this everywhere.**

**Origin story:** the 2026-06-11 audit (`audit_reports/PROJECT_AUDIT_2026-06-11.md`)
found fabricated results (a paper claiming 98.1% with zero experiments run),
~130K LOC mostly unvalidated, a broken PINN, one half-trained model, 45 failing
tests. The rebuild runs under one prime rule: **only execution evidence counts.**
Old code recoverable at tag `pre-convergence-2026-06`.

## 4. History — Gates 0–4 PASSED, merged to `main`

| Gate | What happened | Numbers |
|---|---|---|
| Audit 06-11 | fabrications inventoried | 45 failed/220 passed; results/ empty |
| 0 Ratify | tag `pre-convergence-2026-06`; fake-results purge; env lock | — |
| 1 Stabilize | HybridPINN forward fixed; collection crasher; data-gen tests; ONNX dynamo=False | suite 328 green; CNN1D v1 86.48% |
| 2 Prune | registry 81→11 honest keys; −34.5K LOC; dashboard frozen | suite 206 green |
| 3 Physics & data | PHYSICS.md (owner-ratified); 34 spectral CI tests; **dataset_v2.h5** (3,520 rec, 320/class, exact 80/class/severity, record-level splits, SNR-20/10/5, leakage-checked, DVC) | CNN1D v2 90.53% |
| 4 Benchmark | PROTOCOL frozen; classical (laptop) + deep 8×3 (Colab); ensemble; deployment | table below |

**Phase-4 benchmark** (test acc %, 2,640 windows, mean±std/3 seeds;
`results/benchmark/summary.md`) — **READ WITH POST-AUDIT CAVEATS (§5):** the
`physics_constrained_cnn` row was trained **CE-only** (physics OFF — relabel as
architecture/CE); `multitask_pinn` was **not** trained multitask; significance
was computed at **window level (must be record-level)**.

| Model | Acc | Note |
|---|---|---|
| voting_ensemble | 96.48 | single-seed |
| resnet18 | 96.14±0.28 | strongest vanilla |
| cnn_lstm | 96.12±0.16 | |
| physics_constrained_cnn | 95.98±0.36 | **CE-only — not a physics result** |
| RandomForest | 94.61±0.05 | classical bar |
| cnn1d | 91.94±2.84 | |
| multitask_pinn | 90.28±0.64 | **not trained multitask** |
| hybrid_pinn | 90.04±0.51 | rolling-element physics branch (see §5) |
| patchtst | 89.85±0.19 | |
| attention_cnn | 89.37±4.82 | 1 seed collapsed |

Deployment (C5): ResNet18→ONNX FP32 **13 ms/window CPU**; INT8 4× smaller but
10–15× slower (honest negative). `results/deployment/appendix.md`.

## 5. CURRENT STATE — physics remediation (branch `p6/docs`)

### 5.1 How we got here (the physics saga, four discoveries)
- **§8.0** — `physics_constrained_cnn.forward()` is a plain CNN; physics only
  enters via `compute_physics_loss`, which Phase-4 training never called → the
  benchmark pc_cnn is physics-OFF.
- **§8.0-bis** — that loss was **non-differentiable** (argmax) → zero gradient,
  inert (proven: physics-weight sweep gave byte-identical runs). We made the
  model-method version differentiable (softmax-weighted) and re-ran §8.4/§8.2;
  physics still didn't help (neutral/harmful).
- **§8.0-ter** — the signature DB (`fault_signatures.py`) encoded
  **ROLLING-ELEMENT** physics (BPFO/BPFI/BSF/FTF, outer/inner/ball) — wrong for
  journal bearings — with the 3 mixed classes unmapped (zero constraint). We
  **rebuilt it from PHYSICS.md §4** (correct journal signatures, all 11 classes)
  and added `tests/test_signature_db_consistency.py` (locks DB↔generated data;
  11/11 pass). Owner **ratified** a *band-energy consistency* loss formulation
  (PROTOCOL §7) — **NOT yet implemented.**
- **External audit (`INDEPENDENT_SCIENCE_AUDIT_2026-06-14.md`)** — corroborated
  the above + the pause, but found the blast radius **broader** than we scoped.

### 5.2 Corrected blast radius (use this EXACT framing)
**Sound — keep:**
- Dataset v2 **as a synthetic, internally-consistent benchmark** (balanced,
  group-split by record, leakage-checked, signatures present; cavitation only
  weakly — qualify it). The generator is independent of all the broken physics.
- Benchmark accuracies **as classification results** — but **labels + ALL
  significance are not-yet-sound** (relabel CE-only/non-multitask rows; recompute
  stats at record level).
- §8.1 noise (eval of frozen checkpoints), deployment C5.

**Invalid / contaminated — must fix or quarantine before any claim:**
- **§8.4 / §8.2 "fixed" pc_cnn** — used the **incomplete tonal-only** loss
  (band-energy not implemented; broadband/mixed faults unconstrained).
- **§8.5 HybridPINN** — its physics branch uses **rolling-element** `BearingDynamics`
  (SKF 6205) features → physically wrong for the data. (Earlier "§8.5 independent
  /sound" was WRONG — withdrawn.)
- **§8.6 / C4** — XAI alignment used the old broken-DB bands; calibration used
  pc_cnn checkpoints trained with the incomplete loss.
- **generic `PhysicalConstraintLoss` / `PINNTrainer`** — still non-differentiable
  / inert (only the model-method loss was fixed).
- **All statistics** — computed on 2,640 correlated windows; must be at the
  **528-record** level (5 windows/record share fault, noise, operating point).

**The headline negative is NARROW:** the *stored artifacts* show no physics
advantage, but the *correct* experiment (band-energy loss + journal-bearing
HybridPINN, record-level stats) **has not been run.** Do not call it "decisive."

### 5.3 Remediation sequence (external-auditor-endorsed) — DO IN ORDER
1. **Reconcile docs** to the corrected blast radius + guardrails — **DONE**
   (FINDINGS §0, this file, PROTOCOL §7, PHYSICS_LOSS_AUDIT §6).
2. **Recompute ALL statistics at record level** — **NEXT.** Re-eval the
   benchmark checkpoints (in `results/benchmark/deep/*/seed*/best_model.pth`, no
   retraining) to get per-window predictions → aggregate per 5-s record
   (majority vote / mean logits) → McNemar/Wilcoxon + cluster-bootstrap CIs at
   528 records → correct `results/benchmark/summary.*` + FINDINGS.
3. **Quarantine/relabel invalid rows:** pc_cnn (CE-only) → architecture label;
   multitask_pinn → single-task label; HybridPINN rolling-element physics →
   rebuild on journal-bearing features or remove from physics claims; mark
   `PhysicalConstraintLoss`/`PINNTrainer` physics path non-authoritative; archive
   stale scripts (`scripts/research/pinn_ablation.py` references args that don't
   exist).
4. **Implement + gradient-test the band-energy loss** (ratified): use
   `get_expected_bands`, per-sample rpm, handle empty-tonal classes; tests assert
   `requires_grad`, nonzero param grads, and per-class (tonal/broadband/mixed)
   behavior.
5. **Only then rerun** physics-forward experiments (§8.4, §8.2 pc_cnn, §8.5 with
   corrected HybridPINN if retained, §8.6a recompute) from a frozen manifest.
   Then **rewrite + re-ratify FINDINGS**.

### 5.4 Stored numbers so far (provisional, contaminated per §5.2 — for reference only)
§8.2 fixed pc_cnn: hurts at 10% (91.11±3.29 vs vanilla 93.60), neutral 25/50/100.
§8.4 fixed: clean ~96 flat; 5 dB neutral low-w, harmful at w=1.0 (83.1).
§8.5: 89.76 vs blind 90.04 (null). §8.6a: ratio 0.849 vs 0.716. §8.6b: ECE 0.022
vs 0.028. (`results/phase5_fixed/`, `phase5_dataeff_fixed/`, `xai_alignment/`,
`uncertainty/`.) **None are publishable until §5.3 completes.**

## 6. What must NOT happen (guardrails)

- **Do NOT loosen the wording.** Never "the benchmark is sound" — always "sound
  *as a classification benchmark, pending relabel + record-level stats*." Never
  "decisive negative" — the correct physics experiment hasn't run. Never claim
  any physics benefit (accuracy/robustness/data-efficiency/OOD/interpretability/
  calibration) until §5.3 reruns support it.
- **Do NOT resume from the old narrow blast radius.** §8.5 and the statistics are
  contaminated; treat all physics-forward claims as quarantined.
- **Do NOT skip record-level statistics.** Window-level significance is invalid.
- **Do NOT present a number without an artifact path + provenance** (git SHA,
  seed, host). Verify by execution.
- **Do NOT modify the generator / dataset mid-analysis** (would invalidate the
  frozen benchmark). Generator changes = new dataset version + documented rerun.
- **Do NOT commit to `main` directly** — work on a `pN/...` branch, merge at
  gates with the suite green.
- **Do NOT skip the owner gate** on physics/protocol changes — record §7
  amendments; quote his ratification.
- **Keep the repo self-contained and independent** — do not add references to
  any other/external project in committed files (owner instruction; any
  provenance nuance lives only in private session memory, never in the repo).
- Respect the laptop quirks (§2.1): UTF-8, commit via `-F` file, detached launch
  for long jobs.

## 7. Incidents & lessons (do not repeat)
- **I1** session-bound process death → detached `Start-Process` + resume.
- **I2** Colab `ln -sfn` into existing dir → symlink BEFORE first run.
- **I3** fake Drive mount (`mkdir` before `drive.mount`) → mount first.
- **I4** factory/class default desync broke checkpoint loading → after any
  model-code change, verify `create_model(key)` still loads recorded checkpoints.
- **I5** cp1252 crashes on Unicode prints/writes → UTF-8 everywhere.
- **I6** full-5 s records saturate (~100%) → 1 s windowing chosen.
- **I7 (physics machinery, the big one)** a model named "physics_*" proves
  nothing: verify the physics is (a) actually *wired into training*, (b)
  *differentiable*, (c) the *correct bearing type*, and (d) *consistent with the
  generator* (CI-lock it). Three separate physics defects (inert loss, wrong
  bearing type, never-called loss) all passed a green test suite. Tests must
  assert gradient flow + DB↔data consistency, not just "runs."
- **I8 (statistics)** windowed data → record-level statistics. 2,640 windows from
  528 records are NOT independent.

## 8. Conventions (the honesty machinery)
1. Only execution evidence counts; numbers trace to `results/` artifacts with
   SHA + host + seed.
2. `CONVERGENCE_PLAN.md` checkboxes need `(evidence: …)`; tracker updated each
   session.
3. PROTOCOL frozen — changes are dated §7 amendments; §8 pre-registrations
   before running.
4. Phase branches `pN/...` → `main` at gates. Current: **`p6/docs`**.
5. Suite green before merge: `pytest -q` (251 passed, 6 deselected; dashboard
   frozen until Phase D).
6. Anti-regrowth: tiers fixed-size; cut things at the tag + `BACKLOG.md`.
7. `results/`: small json/md/png/csv committed; checkpoints + h5 out of git (DVC
   for the dataset; full archives in `D:\Libraries`).
8. External-audit discipline: prompts to outside auditors stay neutral
   (`audit_reports/INDEPENDENT_AUDIT_PROMPT.md`); their reports are authoritative
   on scope.

## 9. Key files map
| What | Where |
|---|---|
| **This handoff** | `PROJECT_STATE.md` |
| Findings (read §0 first — supersedes rest) | `results/FINDINGS.md` |
| **External audit (authoritative on scope)** | `audit_reports/INDEPENDENT_SCIENCE_AUDIT_2026-06-14.md` |
| Internal physics audit + remediation plan | `audit_reports/PHYSICS_LOSS_AUDIT_2026-06-14.md` (§6 expanded scope) |
| Neutral auditor prompt | `audit_reports/INDEPENDENT_AUDIT_PROMPT.md` |
| Origin audit | `audit_reports/PROJECT_AUDIT_2026-06-11.md` |
| Step plan + tracker | `CONVERGENCE_PLAN.md` |
| Physics (normative) + CI | `docs/PHYSICS.md`, `tests/test_physics_signatures.py` |
| Protocol + §7 amendments + §8 preregs | `experiments/PROTOCOL.md` |
| Dataset v2 design / file | `experiments/DATASET_V2.md`, `data/generated/dataset_v2.h5` (+DVC) |
| Benchmark / deployment results | `results/benchmark/`, `results/deployment/` |
| Phase-5 results | `results/phase5/`, `phase5_fixed/`, `phase5_dataeff_fixed/`, `noise_robustness/`, `xai_alignment/`, `uncertainty/` (+ `results/README.md` index) |
| Generator (ground truth) | `data/signal_generation/{fault_modeler,generator}.py`, `config/data_config.py` |
| Model-side physics | `packages/core/models/physics/{bearing_dynamics,fault_signatures}.py`, `packages/core/models/pinn/{physics_constrained_cnn,hybrid_pinn,multitask_pinn}.py`, `packages/core/training/{physics_loss_functions,pinn_trainer}.py` |
| Runners | `scripts/{run_benchmark,aggregate_benchmark,run_noise_robustness,run_phase5_gpu,run_xai_calibration,verify_physics_consistency}.py` |
| Active Colab runbook | `experiments/COLAB_DATAEFF_RUNBOOK.md` |
| Deferred ideas | `BACKLOG.md` · frozen dashboard `packages/dashboard/` |
