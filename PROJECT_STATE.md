# PROJECT STATE — session handoff & cold-start prompt

> **READ THIS FIRST, FULLY.** Working memory of the LSTM-PFD effort, written as a
> cold-start prompt for a fresh AI session: who the owner is and how to work with
> him, the compute setup and its quirks, the honest history, the **current state**,
> what's next, and **what must never happen**. After this file, read the docs/code
> listed in §0 before acting.
>
> **Maintenance duty:** update this file at every gate and at the end of every
> session. Keep it truthful and current; it is the single source of truth.
>
> **Last updated: 2026-06-24, session 11.**

## Current status — THE PROJECT IS A COMPLETE NEGATIVE (Phase 7 → submission)

A physics-based **synthetic** dataset + frozen benchmark for **journal/hydrodynamic
bearing** fault diagnosis, built to test one question: **does physics-informed
learning beat purely data-driven baselines?** After a long remediation (the physics
loss turned out to be broken five different ways) and **four** independent-audit
rounds, the verdict is settled and **ratified**:

> **Physics-informed learning gives NO advantage on any axis tested** — clean
> accuracy, noise robustness, data-efficiency, severity-OOD, interpretability, or
> calibration. The last candidate positive (a 5 dB noise-robustness benefit that
> looked significant at **n=3**) was stress-tested with a pre-registered **n=12** grid
> plus a matched-strength **non-physics** control (§8.8, `results/p7_strengthen/`) and
> **did NOT replicate**: correct physics degrades **3.47** pt vs cross-entropy-only's
> **3.54** pt (seed-level **Wilcoxon p=0.79**); no arm is robust on ≥10/12 seeds; the
> n=3 "win" was a **seed artifact** (the same grid's seeds {0,1,2} reproduce it, the
> other nine erase it).

The honest, **publishable** contribution: **synthetic dataset/generator (C1) + frozen
record-level benchmark (C2) + a rigorous COMPLETE NEGATIVE on physics-informed
learning (C3) + a methodological caution + deployment (C5).** Synthetic-only, no
real-rig validation. Ratified verdict: **`results/FINDINGS.md` §0 (RATIFIED
2026-06-24).** All four audit rounds reproduced every decisive number by execution and
agree on this framing (the fourth, 2026-06-24, did a cache-free from-checkpoint
recompute).

**Branch: `main`** (Phase 7 merged here at this gate; `p7/strengthen` is the working
branch). Suite **268 passed, 6 deselected**. **The science has CONVERGED — STOP adding
experiments.** Remaining work is writing + packaging, not science.

### What's next (Phase 7 → submission), in order
1. ~~Owner re-ratifies FINDINGS §0~~ **DONE 2026-06-24**; ~~merge `p7/strengthen`→`main`~~ (this gate).
2. **Repo cleanup:** ~~prune the `config/docs/` pre-convergence relic tree~~ **DONE
   2026-06-24** — removed the whole **dead/broken MkDocs site** (91 files + the
   orphaned `mkdocs.yml`: not CI-deployed, broken nav, "Production-Ready Platform"
   overclaim; recoverable from git). **Still pending:** rename the misleading
   `ops_aware` metrics field (it is the **eval** flag; training **did** use per-sample
   rpm — confirmed by the 2026-06-24 Opus audit). On branch `p7/submission`.
3. **Reproducibility package (audit M1):** pin a **content hash** of
   `data/generated/dataset_v2.h5`; archive the 48 §8.8 checkpoints + the
   Phase-5/benchmark checkpoints to **Zenodo**; a **provenance manifest** (command /
   commit / dataset-hash / checkpoint / random-reference-hash per paper table).
4. **Write the manuscript from scratch** off FINDINGS §0 — dataset + benchmark +
   complete negative + methodological caution; **record-level tables only**;
   synthetic-only. Title shape: *"A Synthetic Journal-Bearing Benchmark for
   Stress-Testing Physics-Informed Fault Diagnosis."*
5. **Choose venue:** a datasets-&-benchmarks track (NeurIPS/ICML) / IEEE Access /
   *Sensors* / *Measurement* / a PHM workshop. **Not** a top mechanical-systems or
   top-ML venue (synthetic-only, negative headline).

---

## 0. If you are a fresh session, do this first

**Read, in order:**
1. This whole file.
2. **`results/FINDINGS.md` — §0 is the RATIFIED verdict (complete negative);** it is
   authoritative. (The pre-audit §1–§5 were removed as superseded.)
3. The n=12 result that settled it: `results/p7_strengthen/p7_strengthen_record_level.json`
   + analysis `scripts/p7_strengthen_record_level.py`; pre-registration
   `experiments/PROTOCOL.md` §8.8.
4. The **fourth audit round** (most relevant; both reproduced everything, no critical
   findings): `audit_reports/INDEPENDENT_AUDIT_2026-06-24_{GPT5,CLAUDE}.md`. (Earlier
   rounds were removed before each new round; recoverable from git history +
   `C:\Users\COWLAR\projects\_lstm_audit_backup_2026-06-*\`.)
5. `README.md`, `results/README.md`, `experiments/PROTOCOL.md`, `docs/PHYSICS.md`
   (normative generator physics). `CONVERGENCE_PLAN.md` lags — **trust FINDINGS §0 +
   this file** over it.

**Verify live state (verify by execution — do NOT trust docstrings):**
- `git branch --show-current` → `main`.
- `$env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe -m pytest -q` → **268
  passed, 6 deselected**.
- Reproduce the headline negative: `.\venv\Scripts\python.exe scripts\p7_strengthen_record_level.py`
  (cache-backed; the auditors also reproduced it cache-free via
  `scripts/audit_independent_recompute.py`). Expect correct ≈ CE-only, Wilcoxon p=0.79.
- Never state a number without `(evidence: command → artifact)`.

**The work now is WRITING + PACKAGING** (see "What's next" above), **not experiments.**

---

## 1. The owner — who you're working with

**Syed Abbas Ahmad** (git `abbas-ahmad-cowlar`; physics background — comfortable with
physics reasoning; ratified the fault-model equations, PROTOCOL, the band-energy/
healthy-reference loss, the §8.8 pre-registration, and the final complete-negative
FINDINGS personally). Working agreement:

- **Decision-maker at gates.** Owner-gated steps need his explicit sign-off; quote his
  ratification in the doc when given.
- **Deeply distrusts unverified claims** — this repo burned him with fabricated
  results. NEVER present a number without an artifact path. Verify by execution. He
  commissioned the independent external audits of our own work and caught us being too
  soft once. Engage his physics reasoning honestly.
- **Has attachment to impressive features** ("my heart doesn't want to let go"). Use
  the tier system, not arguments, if keep/cut tension reappears.
- **Runs the hands-on compute** (Colab) but is not a shell expert (hit the `ln -sfn`
  trap; cross-account Drive nesting). Give exact copy-paste, platform-correct,
  cell-by-cell commands with expected output and a STOP condition.
- **Works in stretches across days; long jobs must survive his absence.** His laptop
  **sleeps when he steps away**, which *pauses* CPU jobs (a ~1 h analysis took a day in
  ~16-min bursts — see I12). For anything that must finish: run it **synchronously
  while he is present**, or make it resume-safe and tell him to keep the lid open.
- Asks good skeptical questions — answer with data and honest scenario analysis,
  **never reassurance**. When he says "don't soften it," don't.

## 2. Compute setup — exact, with quirks

### 2.1 Laptop (primary; where Claude Code runs)
Windows 11 Pro, repo `C:\Users\COWLAR\projects\lstm-pfd`, PowerShell + Git Bash.
venv: **Python 3.14.0, torch 2.9.1+cpu — NO GPU** (`requirements.lock.txt`). Fine for
eval / aggregation / record-level recompute; training is GPU-only.
**Quirks:** (a) cp1252 console — set `PYTHONIOENCODING=utf-8`, pass `encoding='utf-8'`
to `write_text`; (b) `torch.onnx.export` needs `dynamo=False`; (c) `logs/` gitignored;
(d) PowerShell here-strings break on `--%`/quotes — commit via `git commit -F <file>`
or a bash heredoc; (e) the machine **sleeps on idle and pauses CPU jobs** (I12) — run
must-finish jobs synchronously while the owner is present, or keep-awake + lid open.
A from-checkpoint record-level eval is ~38 s/checkpoint/split on CPU (cache it).

### 2.2 Google Colab (de-facto GPU; free Tesla T4, ~75× the laptop)
Sessions ephemeral (~12 h; idle disconnects kill the VM). Owner's Drive:
`MyDrive/lstm-pfd/` with the dataset at `data/dataset_v2.h5` and result folders.
**No active runbook** — the science is done; all runbooks
(`experiments/COLAB_*RUNBOOK.md`, incl. the last one used, `COLAB_P7_STRENGTHEN_RUNBOOK.md`)
are **historical**. Hard-won Colab gotchas if you ever run more (I2/I3/I9): mount Drive
*before* writing under it; symlink the output dir to a **fresh** Drive folder *before*
the first run and verify the `->` arrow + 0 completion-markers; cross-account copies
double/triple-nest. Results come home by Drive download → verify counts + provenance.

### 2.3 Office PC — never used; Colab proved sufficient. Assume no state there.

### 2.4 Archives
`D:\Libraries\` holds full result downloads WITH checkpoints (`.pth`, out of git) from
Phase 5. The **§8.8 n=12 checkpoints** (48 × `best_model.pth`) live in-repo at
`results/p7_strengthen/**` (gitignored via `results/**/*.pth`; ~2.1 GB on the laptop —
**these must go to Zenodo for the repro package**). `results/` keeps json/md only.

## 3. The big picture (framing — complete negative)

A physics-based **synthetic-data** platform for **journal/hydrodynamic bearing** fault
diagnosis (11 classes: sain, desalignement, desequilibre, jeu, lubrification,
cavitation, usure, oilwhirl + 3 mixed). The durable asset is the physics-grounded
signal generator (`data/signal_generation/`; normative `docs/PHYSICS.md`; 34-test
spectral CI). Journal-bearing data is scarce publicly (literature is rolling-element:
CWRU/Paderborn) — the niche. Generator/taxonomy = established physics; **cite prior
art, don't claim novelty there.**

**Contributions (final):** **C1** dataset/generator · **C2** frozen record-level
benchmark (no physics accuracy advantage; best vanilla ties CE-only pc_cnn, McNemar
p=1) · **C3** a **rigorous, complete NEGATIVE** on physics-informed learning (no
advantage on accuracy / noise / data-efficiency / severity-OOD / interpretability /
calibration) · a **methodological caution** (below) · **C5** deployment. *(C4
"physics-consistent XAI" was a hoped-for positive that did NOT survive — it is part of
the negative now, not a contribution.)*

**The methodological caution (a contribution in its own right), two prongs:** (a) a
same-architecture noise result that was "significant" at **n=3** (within-seed McNemar
p=1.2e-4) **dissolved at n=12** once the seed (not the window) was the inferential unit
— a concrete warning about seed counts and estimands in PINN ablations; (b) a physics
loss that **passed a green test suite while silently broken** (non-differentiable
argmax → wrong-bearing-type → tonal-only → flat-baseline, five defects in series). The
rigor that caught both (record-level statistics, pre-registered scrambled *and*
random-band controls, an inert-path hard-block) is the transferable lesson.

Venue tier (honest): datasets-&-benchmarks track / IEEE Access / Sensors / Measurement
/ PHM workshop / arXiv — **not** a top venue. **Synthetic-only; state everywhere.**

**Origin story:** the 2026-06-11 audit found fabricated results (a paper claiming 98.1%
on CWRU with zero experiments run), ~130K LOC mostly unvalidated, a broken PINN, 45
failing tests. Prime rule: **only execution evidence counts.** Old code recoverable at
tag `pre-convergence-2026-06`.

## 4. History — gates passed

| Gate | What happened | Numbers |
|---|---|---|
| Audit 06-11 | fabrications inventoried | 45 failed/220 passed; results/ empty |
| 0 Ratify | tag `pre-convergence-2026-06`; fake-results purge; env lock | — |
| 1 Stabilize | HybridPINN forward fixed; data-gen tests; ONNX dynamo=False | suite 328 green; CNN1D v1 86.48% |
| 2 Prune | registry 81→11 honest keys; −34.5K LOC; dashboard frozen | suite 206 green |
| 3 Physics & data | PHYSICS.md ratified; 34 spectral CI; **dataset_v2.h5** (3,520 rec, 80/class/severity, record-split, SNR-20/10/5, leakage-checked) | CNN1D v2 90.53% |
| 4 Benchmark | PROTOCOL frozen; classical + deep 8×3; ensemble; deployment | record-level near-ceiling; no physics advantage |
| 5 Remediation (Phase 6) | physics loss fixed (band-energy vs frozen healthy ref); record-level stats; 3 audit rounds; **merged `p6/docs`→`main`** `d16af5a` | suite 263; one n=3 noise positive left standing |
| Phase 7 strengthen | pre-registered **§8.8 n=12 grid** + non-physics control → **the positive did NOT replicate**; 4th audit confirms; complete negative ratified; merged `p7/strengthen`→`main` | suite 268; **complete negative** |

**Phase-4 benchmark, record level** (`results/benchmark/summary_record_level.md`):
soft-vote 5 windows/record → near-ceiling (RF 98.74, best deep cnn_lstm 99.43, CE-only
pc_cnn 98.99); **no row shows a physics advantage** (best vanilla cnn_lstm ties CE-only
pc_cnn, gap +0.00, McNemar p=1). Rows honestly relabeled (pc_cnn = CE-only/architecture
physics-OFF; multitask = single-task; hybrid = rolling-element + constant metadata).
Deployment (C5): ResNet18→ONNX FP32 13 ms/window CPU; INT8 4× smaller but 10–15× slower
(honest negative).

## 5. How we reached the complete negative (history — past tense)

This is the trail, kept so a fresh agent understands *why* the verdict is what it is.
None of it is a current claim; the only current verdict is FINDINGS §0.

- **The physics-loss saga (5 defects, all in `compute_physics_loss`).** §8.0 it was
  never called in the benchmark (pc_cnn there is CE-only); §8.0-bis non-differentiable
  (argmax → inert, byte-identical w-sweep); §8.0-ter the signature DB was
  **rolling-element** (BPFO/BPFI), wrong bearing type, mixed classes unmapped —
  **rebuilt** from PHYSICS.md §4; §8.0-quater band-energy but vs a flat spectrum;
  §8.0-quinquies (owner correction) judged against the **frozen healthy-class
  reference** (`packages/core/models/physics/healthy_reference.json`,
  `pen_b = relu(1 − frac_b / H_ref[c][b])`). The corrected loss is real and tested
  (`tests/test_physics_band_energy_loss.py`); the inert generic path is hard-blocked
  (`tests/test_physics_quarantine.py`); HybridPINN's rolling-element branch stays
  quarantined/excluded.
- **Phase 6 reruns (n=3).** With the corrected loss, the §8.2/§8.3/§8.4 reruns
  (`results/phase5_bandenergy/`, now **SUPERSEDED**) showed everything negative
  **except** a 5 dB noise-robustness benefit at w=1.0 that survived record-level n=3
  (McNemar 14–0, p=1.2e-4). A pre-registered **§8.7 scrambled-reference control** then
  showed even *wrong real* bands reproduced it → "not physics-specific, at best a
  spectral regularizer." Three audit rounds reproduced the n=3 numbers and ratified a
  narrowly-worded Gate-5 verdict (merged `d16af5a`).
- **Phase 7 (n=12) killed it.** "Strengthen-then-write": a pre-registered **§8.8 n=12
  grid** (`results/p7_strengthen/`, PROTOCOL §8.8) added a same-code-path CE-only arm
  and a matched-structure **random non-fault-band** control, at 12 seeds. Result: no
  arm beats CE-only (Wilcoxon p ≥ 0.21 for all three w=1.0 arms; none robust ≥10/12);
  the random *non-fault* arm is the **most** robust and correct physics the **weakest**
  — so not even "a spectral regularizer helped." The n=3 result was pure seed luck
  (this grid's seeds {0,1,2} reproduce 4.29 vs 0.06; seeds 3–11 erase it). The fourth
  audit round (GPT-5 + Opus, 2026-06-24) reproduced this cache-free and confirmed the
  complete negative. FINDINGS §0 rewritten + ratified; the stale fabricated
  `config/docs/paper/main.tex` + `Final_Report_UNVALIDATED.pdf` removed/stubbed.

## 6. What must NOT happen (guardrails)
- **The study is a COMPLETE NEGATIVE. Claim NO physics benefit** (accuracy / noise /
  data-efficiency / severity-OOD / interpretability / calibration), and **do NOT**
  reintroduce any "a spectral regularizer helped" wording — n=12 killed even that.
- **Never quote the within-seed / representative-seed McNemar (14–0, p=1.2e-4) as
  evidence.** It is the confounded estimand; the **seed-level Wilcoxon (n.s.)**
  governs. Accuracy stats use the **record (528)**; cross-run claims use the **seed**.
- **Do NOT re-open the science** — it has converged. No new experiments, no subgroup/
  metric hunting for a surviving positive (that would be p-hacking against the
  pre-registration). The remaining work is writing + packaging.
- **Do NOT present a number without an artifact path + provenance.** Verify by
  execution; honor the sanity-gate discipline.
- **Do NOT modify the generator / dataset / `healthy_reference.json` /
  `random_reference.json`** — frozen. A change = new version + documented rerun +
  re-pre-registration.
- **Do NOT commit to `main`** except at a gate with suite green and owner sign-off;
  work on a `pN/...` branch.
- **Keep the repo self-contained** — no references to any other/external project in
  committed files (owner instruction; provenance nuance lives only in private memory).
- Respect laptop quirks (§2.1): UTF-8; commit via `-F`; don't `&`-orphan jobs; run
  must-finish CPU jobs synchronously (the laptop sleeps — I12).

## 7. Incidents & lessons (do not repeat)
- **I1** session-bound/`&`-orphaned process death → detached + resume-safe + completion artifact.
- **I2** Colab `ln -sfn` into an existing (committed) results dir → `rm -rf` first, symlink BEFORE the first run, verify the arrow + 0 markers.
- **I3** fake Drive mount (`mkdir` before `drive.mount`) → mount first.
- **I4** factory/class default desync broke checkpoint loading → after any model-code change, verify `create_model(key)` still loads recorded checkpoints (sanity gate: argmax reproduces recorded window accuracy).
- **I5** cp1252 crashes on Unicode → UTF-8 everywhere.
- **I6** full-5 s records saturate (~100%) → 1 s windowing.
- **I7 (the big physics lesson)** a model named "physics_*" proves nothing: verify the physics is (a) wired into training, (b) differentiable, (c) the correct bearing type, (d) consistent with the generator, (e) judged against the right baseline (the real healthy class). Five defects all passed a green suite. Tests must assert gradient flow + DB↔data consistency, not just "runs."
- **I8 (statistics)** windowed data → record-level statistics (2,640 windows from 528 records are NOT independent).
- **I9 (Colab Drive)** cross-account folder copies double/triple-nest; locate the real run-root by a known metrics.json path before resuming.
- **I10 (caching)** a result-cache key must encode every axis that changes the output (the split, not just checkpoint+record-count) — clean vs 5 dB collided once; the sanity gate caught it.
- **I11 (estimand reporting)** a point estimate and its CI must be the SAME estimand (don't pair a seed-mean gap with a representative-seed CI).
- **I12 (the n=3→n=12 lesson + the laptop-sleep ops trap)** a within-seed McNemar that looks decisive (14–0, p=1.2e-4) can be a **seed artifact** — at n=3 it was; the **seed-level** test at n=12 is non-significant. Use the seed as the unit and pre-register the seed-level estimand BEFORE running. Ops corollary: the n=12 record-level recompute (~1 h CPU) kept getting **paused by the laptop sleeping** while the owner was away (it crawled across a day in ~16-min bursts); a keep-awake (`SetThreadExecutionState`) blocks idle-sleep but **not lid-close**. Run must-finish CPU jobs synchronously with the owner present; better, have the GPU runner dump per-window probs at eval time so no checkpoint re-inference is needed.

## 8. Conventions (the honesty machinery)
1. Only execution evidence counts; numbers trace to `results/` with SHA + host + seed.
2. `CONVERGENCE_PLAN.md` checkboxes need `(evidence: …)` — note the tracker currently lags; FINDINGS §0 + this file are authoritative.
3. PROTOCOL frozen — dated §7 amendments; §8 pre-registrations before running.
4. Phase branches `pN/...` → `main` at gates. Current: merged through Phase 7 on `main`; working branch `p7/strengthen`.
5. Suite green before merge: `pytest -q` → **268 passed, 6 deselected** (dashboard frozen until Phase D).
6. Anti-regrowth: fixed-size tiers; cut at the tag + `BACKLOG.md`.
7. `results/`: small json/md/png/csv committed; **`*.pth` checkpoints + h5 out of git** (`.gitignore: results/**/*.pth`; archives in `D:\Libraries` / Zenodo for the repro package).
8. External-audit discipline: neutral, model-tailored prompts (outside the repo); reports are authoritative on scope; prior reports deleted before a new round so each audit is uncontaminated (recoverable from git history + backup).

## 9. Key files map
| What | Where |
|---|---|
| **This handoff** | `PROJECT_STATE.md` |
| **RATIFIED verdict (read first)** | `results/FINDINGS.md` §0 (complete negative, 2026-06-24) |
| **The decisive n=12 result** | `results/p7_strengthen/p7_strengthen_record_level.json`; analysis `scripts/p7_strengthen_record_level.py`; checkpoints `results/p7_strengthen/**` (gitignored) |
| **Pre-registration** | `experiments/PROTOCOL.md` §8.8 (n=12 grid + decision rule); §8.7 (F9 scramble, historical) |
| **Fourth-round audits + auditor scripts** | `audit_reports/INDEPENDENT_AUDIT_2026-06-24_{GPT5,CLAUDE}.md`; `scripts/audit_{independent_recompute,verify_random_control}.py` |
| Prior audits — removed before each round | `audit_reports/NOTE_prior_reports_removed.md`; backups `C:\Users\COWLAR\projects\_lstm_audit_backup_2026-06-{22,24}\` |
| Dataset v2 | `data/generated/dataset_v2.h5`, `experiments/DATASET_V2.md`, `dataset_card.yaml` |
| Benchmark (record level) | `results/benchmark/summary_record_level.md` |
| Generator physics (normative) + CI | `docs/PHYSICS.md`, `tests/test_physics_signatures.py` |
| Model-side physics (tested → negative) | `packages/core/models/pinn/physics_constrained_cnn.py`; `packages/core/models/physics/{fault_signatures.py,healthy_reference.json,random_reference.json}` |
| Runner | `scripts/run_phase5_gpu.py` (`--control {f9_scramble,random_bands}`, `--weights`, `--out-root`) |
| **Historical / SUPERSEDED (provenance only — do NOT cite as current)** | `results/phase5_bandenergy/` (n=3 band-energy reruns + §8.7 F9 control, banner-marked SUPERSEDED); `results/phase5*/`, `results/{xai_alignment,uncertainty,noise_robustness}/` (pre-remediation / negative side-results) |
| Tests (physics) | `tests/test_{physics_signatures,signature_db_consistency,physics_band_energy_loss,physics_random_band_control,physics_quarantine}.py` |
| Deferred / frozen | `BACKLOG.md` · dashboard `packages/dashboard/` |
