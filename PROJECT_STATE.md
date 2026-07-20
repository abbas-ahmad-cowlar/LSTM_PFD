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
> **Last updated: 2026-06-26, session 13.** (Since the previous 2026-06-25 update:
> `p7/submission`→`main` merged, `p7/repo-tidy`→`main` merged, docs-staleness
> cleanup, **push to public GitHub**, and a **git-history rewrite** — see §Current
> status and the guardrails in §6.)

## Current status — COMPLETE NEGATIVE, manuscript drafted, repo public (Phase 7 → submission/packaging)

A physics-based **synthetic** dataset + frozen benchmark for **journal/hydrodynamic
bearing** fault diagnosis, built to test one question: **does physics-informed
learning beat purely data-driven baselines?** After a long remediation (the physics
loss turned out to be broken five different ways) and **four** independent-audit
rounds plus a fifth "swift" audit, the verdict is settled and **ratified**:

> **Physics-informed learning gives NO advantage on any axis tested** — clean
> accuracy, noise robustness, data-efficiency, severity-OOD, interpretability, or
> calibration. The last candidate positive (a 5 dB noise-robustness benefit that
> looked significant at **n=3**) was stress-tested with a pre-registered **n=12** grid
> plus a matched-strength **non-physics** control (§8.8,
> `results/noise_seed_robustness/`) and **did NOT replicate**: correct physics
> degrades **3.47** pt vs cross-entropy-only's **3.54** pt (seed-level **Wilcoxon
> p=0.79**); no arm is robust on ≥10/12 seeds; the random non-fault-band control is
> the *most* robust and correct physics the *weakest* of the three w=1.0 arms. The
> n=3 "win" was a **seed artifact** (that grid's seeds {0,1,2} reproduce 4.29 vs 0.06;
> seeds 3–11 erase it).

The honest, **publishable** contribution: **synthetic dataset/generator (C1) + frozen
record-level benchmark (C2) + a rigorous COMPLETE NEGATIVE on physics-informed
learning (C3) + a methodological caution (C4) + deployment (C5).** Synthetic-only, no
real-rig validation. Ratified verdict: **`results/FINDINGS.md` §0 (RATIFIED
2026-06-24).** Science has **CONVERGED — STOP adding experiments.**

**Where the work is now.**
- **Branch: `main` @ `cf92673`** (verified 2026-06-26). Everything is merged to `main`:
  the Phase-7 science, the JBFD-11 **manuscript** (was `p7/submission`), and the
  **repo tidy** (was `p7/repo-tidy`). `main` is **pushed to public GitHub**
  (`https://github.com/abbas-ahmad-cowlar/LSTM_PFD`). Suite **268 passed, 6 deselected**.
- **Manuscript = FULL FIRST DRAFT COMPLETE** in `paper/` — a self-contained
  NeurIPS-style LaTeX build that compiles to **11 pp, box-clean** (0 overfull/underfull,
  no undefined refs), with **12 web-verified DOI citations**. See §4 / §9.
- **Remaining work is packaging + submission** (owner-side): the Zenodo checkpoint
  deposit → a real DOI → finalize the paper's release wording → arXiv. See §What's next.

> ⚠️ **The git history was rewritten on 2026-06-26 (see §6).** ALL commit SHAs from
> 2026-06-11 onward were reassigned. **Any commit SHA written in older docs no longer
> resolves.** Trust `git log` and the SHAs verified in *this* file, not historical hashes.

### What's next (Phase 7 → submission)
- ✅ **FINDINGS ratified** (`results/FINDINGS.md` §0, 2026-06-24, complete negative).
- ✅ **Reproducibility package (audit M1):** dataset/reference SHA-256 pinned;
  `results/PROVENANCE_MANIFEST.md` written. **The ONE remaining repro item is the
  OWNER's: upload the ~2.1 GB (48 §8.8 checkpoints; more for the full package) to
  Zenodo and drop the DOI into the manifest + paper §7** (still `<TBD>`).
- ✅ **Manuscript full first draft** — all 9 sections + Appendix A; Tables T1–T5;
  Figures F1–F3; novelty pass (closest prior art Gecgel 2021 cited & distinguished;
  narrowed novelty claim); merged to `main`; pushed to public GitHub.
- ✅ **Repo tidy** — result dirs renamed content-keyed; superseded dirs removed (§9).
- ✅ **Swift audit (2026-06-26)** — no science/manuscript issues; doc-staleness fixes applied.
- ✅ **Merged to `main` + pushed to public GitHub**; contributor list cleaned + fake
  paper purged from history (§6).
- **NEXT (owner-side, the critical path):**
  1. **Zenodo deposit.** Build the archive (dataset `data/generated/dataset_v2.h5` +
     the 117 checkpoints under `results/noise_seed_robustness/` + `band_energy_reruns/`
     + `benchmark/deep/`), upload, **Reserve DOI**.
     Archive command: `tar -cvf jbfd11_checkpoints.tar results/noise_seed_robustness results/band_energy_reruns results/benchmark/deep`.
  2. **Finalize release wording.** With the real DOI: edit paper §7 + `dataset_card.yaml`
     `doi:` + `results/PROVENANCE_MANIFEST.md`, flipping "**will** archive" → "**archived**
     (DOI …)". Until the DOI exists, the release wording MUST stay conditional.
  3. **Finalize byline.** Currently "Syed Abbas Ahmad, Pakistan Institute of
     Engineering and Applied Sciences (PIEAS)" (full form, NO Cowlar thanks-note);
     contact `syedabbasahmad6@gmail.com`. Finalize before the arXiv post.
  4. **AI-assistance disclosure** in the paper — owner's call, per target venue.
  5. **arXiv** (cs.LG primary + eess.SP secondary; optional cross-list stat.ML) →
     then a **PHM / trustworthy-ML-evaluation workshop**. **NeurIPS ED / KDD D&B 2027**
     is the upgrade target (2026 D&B deadline passed). IEEE Access only if real-rig
     data is ever added. NOT a top mechanical-systems / top-ML venue (synthetic-only,
     negative headline).
- **Optional cleanups:** delete the merged phase branches from GitHub; ask GitHub
  Support to expedite GC of the purged fake-paper blobs; verify `references.bib`
  DOIs/pages before camera-ready; optionally add Oh 2016 / Jebur & Soud 2025 to §2.

---

## 0. If you are a fresh session, do this first

**Read, in order:**
1. This whole file.
2. **`results/FINDINGS.md` — §0 is the RATIFIED verdict (complete negative);** it is
   authoritative and freezes the claims the paper may/may not make. (Pre-audit §1–§5 removed.)
3. The n=12 result that settled it: `results/noise_seed_robustness/noise_seed_robustness_record_level.json`
   + analysis `scripts/p7_strengthen_record_level.py`; pre-registration `experiments/PROTOCOL.md` §8.8.
4. **The manuscript:** `paper/main.tex` (the full first draft; builds with the bundled
   `paper/neurips_2023.sty`), `paper/OUTLINE.md` (superseded framing skeleton — the
   paper supersedes it), `results/PROVENANCE_MANIFEST.md` (repro chain + hashes; Zenodo DOI = TBD).
5. The fifth/**swift** + fourth audits: `audit_reports/INDEPENDENT_AUDIT_2026-06-24_{GPT5,CLAUDE}.md`
   (both reproduced everything cache-free; no critical findings). Prior rounds removed
   (`audit_reports/NOTE_prior_reports_removed.md`; recoverable from git).
6. `README.md`, `results/README.md`, `experiments/PROTOCOL.md`, `docs/PHYSICS.md`
   (normative generator physics). `CONVERGENCE_PLAN.md` is **SUPERSEDED/HISTORICAL**
   (banner-marked; its checkboxes and its going-in "physics helps" framing are
   hypotheses, NOT results) — trust FINDINGS §0 + this file over it.

**Verify live state (verify by execution — do NOT trust docstrings). Verified 2026-06-26:**
- `git branch --show-current` → **`main`** (HEAD **`cf92673`**; in sync with `origin/main`).
- `$env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe -m pytest -q` → **268 passed, 6 deselected**.
- `cd paper; latexmk -pdf main.tex` → **11 pp, 0 overfull/underfull, no undefined refs**.
- Checkpoints present: `results/noise_seed_robustness/` **48** `*.pth`,
  `results/band_energy_reruns/` **45**, `results/benchmark/deep/` **24**.
- Dataset hash: `sha256sum data/generated/dataset_v2.h5` → **`f72ad35b…`** (matches manifest).
- Reproduce the headline negative: `.\venv\Scripts\python.exe scripts\p7_strengthen_record_level.py`
  (cache-backed) → correct ≈ CE-only, Wilcoxon **p=0.79**. (Cache-free from-checkpoint
  recompute exists: `scripts/audit_independent_recompute.py`; ~1 h CPU — see I12.)
- Never state a number without `(evidence: command → artifact)`.

**Current result-directory names (content-keyed after the 2026-06 tidy):**
| Dir | Contribution | Was (old jargon name — do NOT use) |
|---|---|---|
| `results/noise_seed_robustness/` | C3 §8.8 headline n=12 grid (48 ckpts) | `p7_strengthen` |
| `results/band_energy_reruns/` | C3 §8.2 data-eff + §8.3 OOD (45 ckpts; §8.4/§8.7 superseded-in-file) | `phase5_bandenergy` |
| `results/interpretability/` | C3 §8.6a (IG attribution) | `xai_alignment` |
| `results/calibration/` | C3 §8.6b (MC-dropout ECE) | `uncertainty` |
| `results/benchmark/` | C2 (24 ckpts) | (unchanged) |
| `results/dataset_v2_validation/` | C1 | (unchanged) |
| `results/deployment/` | C5 | (unchanged) |
**DELETED (removed from git history, gone):** `results/phase5*`, `results/noise_robustness`,
`results/cnn1d_v{1,2}_baseline`, and `config/docs/` (the dead MkDocs site that held the
fabricated `Final_Report_UNVALIDATED.pdf` — see §6).

**The work now is PACKAGING + SUBMISSION** (Zenodo → DOI → arXiv), **not experiments.**

---

## 1. The owner — who you're working with

**Syed Abbas Ahmad** (git author `Syed Abbas Ahmad <abbas.ahmad@cowlar.com>`; GitHub
`abbas-ahmad-cowlar`). From **PIEAS** (Pakistan Institute of Engineering and Applied
Sciences), currently working with **Cowlar Design Studio**. Physics background —
comfortable with physics reasoning; personally ratified the fault-model equations,
PROTOCOL, the band-energy/healthy-reference loss, the §8.8 pre-registration, and the
final complete-negative FINDINGS. Working agreement:

- **Decision-maker at gates.** Owner-gated steps need his explicit sign-off; quote his
  ratification when given.
- **Deeply distrusts unverified claims** — this repo burned him with fabricated results.
  NEVER present a number without an artifact path. Verify by execution. He commissioned
  independent external audits of our own work and caught us being too soft once. Engage
  his physics reasoning honestly; when he says "don't soften it," don't.
- **Cares about his professional image / integrity of the public record.** He had (a)
  all `Co-Authored-By: Claude` trailers stripped from history (GitHub was listing Claude
  as a contributor) and (b) the fabricated "98.1% CWRU" paper purged from history. See §6.
- **Has attachment to impressive features** — use the tier system, not arguments, if a
  keep/cut tension reappears.
- **Runs the hands-on compute** (Colab) but is not a shell expert. Give exact copy-paste,
  platform-correct, cell-by-cell commands with expected output and a STOP condition.
- **Works in stretches across days; long jobs must survive his absence.** His laptop
  **sleeps when he steps away**, which *pauses* CPU jobs (I12). Run must-finish work
  synchronously while he is present, or make it resume-safe and tell him to keep the lid open.

## 2. Compute setup — exact, with quirks

### 2.1 Laptop (primary; where Claude Code runs)
Windows 11 Pro, repo `C:\Users\COWLAR\projects\lstm-pfd`, PowerShell + Git Bash.
venv: **Python 3.14.0, torch 2.9.1+cpu — NO GPU** (`requirements.lock.txt`). Fine for
eval / aggregation / record-level recompute / LaTeX; training is GPU-only.
**Quirks:** (a) cp1252 console — set `PYTHONIOENCODING=utf-8`, pass `encoding='utf-8'`
to `write_text`; (b) `torch.onnx.export` needs `dynamo=False`; (c) `logs/` gitignored;
(d) PowerShell here-strings break on `--%`/quotes — commit via `git commit -F <file>`
or a bash heredoc; (e) the machine **sleeps on idle and pauses CPU jobs** (I12); (f)
LaTeX = **MiKTeX** (`pdflatex`/`latexmk` on PATH; AutoInstall was enabled so builds
are non-interactive); (g) the `.git` is ~750 MB (large historical blobs) — a full
`git bundle --all` exceeds the 5-min foreground limit, so run heavy git ops
(`filter-repo`, `gc`) in the background.

### 2.2 Google Colab (de-facto GPU; free Tesla T4)
Sessions ephemeral (~12 h). Owner's Drive: `MyDrive/lstm-pfd/`. **No active runbook** —
the science is done; all `experiments/COLAB_*RUNBOOK.md` are **historical**. Hard-won
Colab gotchas if you ever run more (I2/I3/I9): mount Drive *before* writing under it;
symlink the output dir to a **fresh** Drive folder before the first run; cross-account
copies double/triple-nest; results come home by Drive download → verify counts + provenance.

### 2.3 Office PC — never used; Colab proved sufficient. Assume no state there.

### 2.4 Archives
`D:\Libraries\` holds full result downloads WITH checkpoints (`.pth`, out of git) from
Phase 5. The **§8.8 n=12 checkpoints** (48 × `best_model.pth`) live in-repo at
`results/noise_seed_robustness/**` (gitignored via `results/**/*.pth`; ~2.1 GB — **these
must go to Zenodo for the repro package**). `results/` keeps json/md/png only.

## 3. The big picture (framing — complete negative)

A physics-based **synthetic-data** platform for **journal/hydrodynamic bearing** fault
diagnosis (11 classes: sain, desalignement, desequilibre, jeu, lubrification,
cavitation, usure, oilwhirl + 3 mixed). The durable asset is the physics-grounded
signal generator (`data/signal_generation/`; normative `docs/PHYSICS.md`; 34-test
spectral CI). Journal-bearing data is scarce publicly (literature is rolling-element:
CWRU/Paderborn) — the niche. Generator/taxonomy = established physics; **cite prior
art, don't claim novelty there** (closest prior art = Gecgel et al. 2021, ASME
J. Tribol. — simulated journal-bearing DL for wear; cited & distinguished in §2).

**Contributions (final):** **C1** dataset/generator · **C2** frozen record-level
benchmark (no physics accuracy advantage; best vanilla ties CE-only pc_cnn, McNemar
p=1) · **C3** a **rigorous, complete NEGATIVE** on physics-informed learning (no
advantage on accuracy / noise / data-efficiency / severity-OOD / interpretability /
calibration) · **C4** a **methodological caution** · **C5** deployment (appendix).

**The methodological caution (C4), two prongs:** (a) a same-architecture noise result
"significant" at **n=3** (within-seed McNemar p=1.2e-4) that **dissolved at n=12** once
the seed (not the window) was the inferential unit — a concrete warning about seed
counts and estimands in PINN ablations; (b) a physics loss that **passed a green test
suite while silently broken** (non-differentiable argmax → wrong-bearing-type → tonal-only
→ flat-baseline → fixed; five defects in series). The rigor that caught both (record-level
statistics, pre-registered scrambled *and* random-band controls, an inert-path hard-block)
is the transferable lesson.

**Novelty boundary (2026-06-25 scan, web-verified):** do NOT claim "first synthetic
journal-bearing dataset." The defensible claim (in §2) = "to our knowledge, the first
*released synthetic journal-bearing benchmark designed specifically to stress-test
physics-informed fault diagnosis with record-level and seed-level inference and matched
non-physics controls*." Adjacent prior art cited & distinguished: Gecgel 2021, Jeon 2020,
Zeynivand 2026, Lu 2023, Kim & Kim 2024, Vieira 2025.

**Venue tier (honest):** datasets-&-benchmarks track / PHM workshop / arXiv — **not** a
top venue. **Synthetic-only; state everywhere.**

**Origin story:** the 2026-06-11 audit found fabricated results (a paper claiming 98.1%
on CWRU with zero experiments run — the `Final_Report_UNVALIDATED.pdf`, since purged),
~130K LOC mostly unvalidated, a broken PINN, 45 failing tests. Prime rule: **only
execution evidence counts.** Old code recoverable at tag `pre-convergence-2026-06`.

## 4. History — gates passed

| Gate | What happened | Numbers |
|---|---|---|
| Audit 06-11 | fabrications inventoried | 45 failed/220 passed; results/ empty |
| 0 Ratify | tag `pre-convergence-2026-06`; fake-results purge; env lock | — |
| 1 Stabilize | HybridPINN forward fixed; data-gen tests; ONNX dynamo=False | suite 328 green; CNN1D v1 86.48% |
| 2 Prune | registry 81→11 honest keys; −34.5K LOC; dashboard frozen | suite 206 green |
| 3 Physics & data | PHYSICS.md ratified; 34 spectral CI; **dataset_v2.h5** (3,520 rec) | CNN1D v2 90.53% |
| 4 Benchmark | PROTOCOL frozen; classical + deep 8×3; ensemble; deployment | record-level near-ceiling; no physics advantage |
| 5 Remediation (Phase 6) | physics loss fixed (band-energy vs frozen healthy ref); record-level stats; 3 audit rounds | suite 263; one n=3 noise positive left standing |
| 7 Strengthen | pre-registered **§8.8 n=12 grid** + non-physics control → **positive did NOT replicate**; 4th audit confirms; complete negative ratified | suite 268; **complete negative** |
| 7 Submission | manuscript full first draft (11 pp); novelty pass; merged to `main` | suite 268; box-clean |
| 7 Tidy/publish | content-key rename; swift audit; push to public GitHub; history rewrite (§6) | suite 268; `main`@`cf92673` |

> **NOTE:** commit SHAs for these gates were all reassigned by the 2026-06-26 history
> rewrite. Read the current trail with `git log --oneline` (current HEAD `cf92673`;
> the manuscript merge is `a2a4f78`, the tidy merge `067bae1`).

**Phase-4 benchmark, record level** (`results/benchmark/summary_record_level.md`):
soft-vote 5 windows/record → near-ceiling (RF 98.74, best deep cnn_lstm 99.43, CE-only
pc_cnn 98.99); **no row shows a physics advantage** (best vanilla cnn_lstm ties CE-only
pc_cnn, gap +0.00, McNemar p=1). Rows honestly relabeled (pc_cnn = CE-only/physics-OFF;
multitask = single-task; hybrid = rolling-element + constant metadata).
Deployment (C5): ResNet18→ONNX FP32 ~13 ms/window CPU; INT8 4× smaller but ~15× slower
(honest negative). (`results/deployment/appendix.md`.)

## 5. How we reached the complete negative (history — past tense)

The trail, kept so a fresh agent understands *why* the verdict is what it is. None is a
current claim; the only current verdict is FINDINGS §0.

- **The physics-loss saga (5 defects, all in `compute_physics_loss`).** §8.0 never
  called in the benchmark (pc_cnn there is CE-only); §8.0-bis non-differentiable (argmax
  → inert, byte-identical w-sweep); §8.0-ter the signature DB was **rolling-element**
  (BPFO/BPFI), wrong bearing type — **rebuilt** from PHYSICS.md §4; §8.0-quater band-energy
  but vs a flat spectrum; §8.0-quinquies (owner correction) judged against the **frozen
  healthy-class reference** (`healthy_reference.json`, `pen_b = relu(1 − frac_b/H_ref[c][b])`).
  The corrected loss is real and tested; the inert generic path is hard-blocked
  (`tests/test_physics_quarantine.py`).
- **Phase 6 (n=3).** With the corrected loss, everything negative **except** a 5 dB
  noise benefit at w=1.0 that survived record-level n=3 (McNemar 14–0, p=1.2e-4). A
  pre-registered **§8.7 scrambled-reference control** showed even *wrong real* bands
  reproduced it → "not physics-specific." Three audit rounds ratified a narrow Gate-5 verdict.
- **Phase 7 (n=12) killed it.** A pre-registered **§8.8 n=12 grid**
  (`results/noise_seed_robustness/`, PROTOCOL §8.8) added a same-code-path CE-only arm
  and a matched-structure **random non-fault-band** control at 12 seeds. Result: no arm
  beats CE-only (Wilcoxon p ≥ 0.21; none robust ≥10/12); the random *non-fault* arm is
  the **most** robust and correct physics the **weakest** — so not even "a spectral
  regularizer helped." The n=3 result was pure seed luck. Fourth audit (GPT-5 + Opus,
  2026-06-24) reproduced cache-free; FINDINGS §0 rewritten + ratified.

## 6. What must NOT happen (guardrails)
- **The study is a COMPLETE NEGATIVE. Claim NO physics benefit** (accuracy / noise /
  data-efficiency / severity-OOD / interpretability / calibration), and **do NOT**
  reintroduce any "a spectral regularizer helped" wording — n=12 killed even that.
- **Never quote the within-seed / representative-seed McNemar (14–0, p=1.2e-4) as
  evidence.** It is the confounded estimand; the **seed-level Wilcoxon (n.s.)** governs.
  Accuracy stats use the **record (528)**; cross-run claims use the **seed**.
- **Do NOT re-open the science** — it has converged. No new experiments, no subgroup/
  metric hunting for a surviving positive (p-hacking against the pre-registration).
- **Do NOT present a number without an artifact path + provenance.** Verify by execution.
- **Do NOT modify the generator / dataset / `healthy_reference.json` /
  `random_reference.json`** — frozen. A change = new version + documented rerun + re-pre-registration.
- **Release wording stays conditional ("will archive") until the Zenodo DOI is real.**
- **Keep the repo self-contained** — no references to any other/external project in
  committed files (owner instruction).
- **GIT HISTORY was rewritten 2026-06-26 — respect it:**
  - **NEVER add `Co-Authored-By: Claude …` trailers to commits in this repo.** The owner
    had all such trailers stripped from history (they listed Claude as a GitHub
    contributor, damaging his image). Commit as the owner only. This **overrides** the
    generic harness default that appends that trailer. (See memory `no-claude-coauthor-trailers`.)
  - The fabricated `config/docs/.../Final_Report_UNVALIDATED.pdf` + `config/docs/paper/`
    were **purged from all history** and force-pushed. Do NOT resurrect or reference them.
  - The rewrite (git-filter-repo + force-push) **changed every commit SHA** from
    2026-06-11 onward. Old hashes in any doc are dangling — trust `git log`.
  - Committing to `main` is fine at a gate with suite green + owner sign-off (the phase
    branches are all merged). Push to `origin` only when the owner asks.
- Respect laptop quirks (§2.1): UTF-8; commit via `-F`; run must-finish CPU jobs
  synchronously (the laptop sleeps — I12); background heavy git ops.

## 7. Incidents & lessons (do not repeat)
- **I1** session-bound/`&`-orphaned process death → detached + resume-safe + completion artifact.
- **I2** Colab `ln -sfn` into an existing results dir → `rm -rf` first, symlink BEFORE the first run, verify the arrow + 0 markers.
- **I3** fake Drive mount (`mkdir` before `drive.mount`) → mount first.
- **I4** factory/class default desync broke checkpoint loading → after any model-code change, verify `create_model(key)` still loads recorded checkpoints (sanity gate).
- **I5** cp1252 crashes on Unicode → UTF-8 everywhere.
- **I6** full-5 s records saturate (~100%) → 1 s windowing.
- **I7 (the big physics lesson)** a model named "physics_*" proves nothing: verify the physics is (a) wired into training, (b) differentiable, (c) the correct bearing type, (d) consistent with the generator, (e) judged against the right baseline. Five defects all passed a green suite. Tests must assert gradient flow + DB↔data consistency, not just "runs."
- **I8 (statistics)** windowed data → record-level statistics (2,640 windows from 528 records are NOT independent).
- **I9 (Colab Drive)** cross-account folder copies double/triple-nest; locate the real run-root by a known metrics.json path before resuming.
- **I10 (caching)** a result-cache key must encode every axis that changes the output (the split, not just checkpoint+record-count).
- **I11 (estimand reporting)** a point estimate and its CI must be the SAME estimand.
- **I12 (n=3→n=12 + laptop-sleep)** a within-seed McNemar that looks decisive (14–0, p=1.2e-4) can be a **seed artifact**; use the seed as the unit and pre-register the seed-level estimand. Ops corollary: the n=12 record-level recompute (~1 h CPU) kept getting **paused by the laptop sleeping** while away; run must-finish CPU jobs synchronously with the owner present.
- **I13 (history rewrite / stale refs)** `git filter-repo` twice (trailer strip) + once (path purge) + force-push cleaned `main`, but verification via `git rev-list --all` still showed the fake PDF — it was hanging off stale local-only `refs/original/*` (old filter-branch backups) + another agent's `refs/codex/*` turn-diff checkpoints. Delete those refs + `git gc --prune=now` before trusting a "fully purged" check. Never trust the first "done" on a history rewrite — verify against real refs, and remember GitHub keeps old blobs reachable-by-SHA until it GCs.

## 8. Conventions (the honesty machinery)
1. Only execution evidence counts; numbers trace to `results/` with SHA + host + seed.
2. `CONVERGENCE_PLAN.md` is **SUPERSEDED/HISTORICAL** (banner-marked); FINDINGS §0 + this file are authoritative. Its checkboxes and "physics helps" framing are pre-outcome hypotheses.
3. PROTOCOL frozen — dated §7 amendments; §8 pre-registrations before running. Do NOT silently rewrite a frozen amendment.
4. Phase branches `pN/...` → `main` at gates. **Current: everything merged to `main` (HEAD `cf92673`); pushed to public GitHub.**
5. Suite green before merge: `pytest -q` → **268 passed, 6 deselected** (dashboard frozen; 6 deselected).
6. Anti-regrowth: fixed-size tiers; cut at the tag + `BACKLOG.md`.
7. `results/`: small json/md/png committed; **`*.pth` checkpoints + h5 out of git** (`.gitignore: results/**/*.pth`; archives in `D:\Libraries` / Zenodo for the repro package).
8. External-audit discipline: neutral, model-tailored prompts (outside the repo); reports authoritative on scope; prior reports deleted before a new round (recoverable from git + backup).
9. **No `Co-Authored-By: Claude` trailers** (§6). Commit as the owner.

## 9. Key files map
| What | Where |
|---|---|
| **This handoff** | `PROJECT_STATE.md` |
| **RATIFIED verdict (read first)** | `results/FINDINGS.md` §0 (complete negative, 2026-06-24) |
| **The manuscript (full first draft)** | `paper/main.tex` (+ bundled `paper/neurips_2023.sty`, `paper/tables/T1–T4.tex`, `paper/references.bib` w/ DOIs). Builds `latexmk -pdf` → 11 pp box-clean. Figures via `scripts/make_paper_figures.py` → `paper/figures/F1–F3.pdf`. Framing skeleton (superseded): `paper/OUTLINE.md`. |
| **The decisive n=12 result** | `results/noise_seed_robustness/noise_seed_robustness_record_level.json`; analysis `scripts/p7_strengthen_record_level.py`; checkpoints `results/noise_seed_robustness/**` (48, gitignored) |
| **Pre-registration** | `experiments/PROTOCOL.md` §8.8 (n=12 grid + decision rule); §8.7 (F9 scramble, historical) |
| **Reproducibility** | `results/PROVENANCE_MANIFEST.md` (chain + content hashes; Zenodo DOI = TBD). SHA-256: dataset `f72ad35b…`; `healthy_reference.json` `3274296e…`; `random_reference.json` `946590a4…` |
| **Audits (4th round + auditor scripts)** | `audit_reports/INDEPENDENT_AUDIT_2026-06-24_{GPT5,CLAUDE}.md`; `scripts/audit_{independent_recompute,verify_random_control}.py`; removed priors: `audit_reports/NOTE_prior_reports_removed.md` |
| Dataset v2 | `data/generated/dataset_v2.h5`, `experiments/DATASET_V2.md`, `dataset_card.yaml` |
| Benchmark (record level) | `results/benchmark/summary_record_level.md` |
| Other regimes (each negative) | `results/band_energy_reruns/summary_record_level.json` (§8.2/§8.3); `results/interpretability/alignment.json` (§8.6a); `results/calibration/calibration.json` (§8.6b) |
| Generator physics (normative) + CI | `docs/PHYSICS.md`, `tests/test_physics_signatures.py` |
| Model-side physics (tested → negative) | `packages/core/models/pinn/physics_constrained_cnn.py`; `packages/core/models/physics/{fault_signatures.py,healthy_reference.json,random_reference.json}` |
| Runner | `scripts/run_phase5_gpu.py` (`--control {f9_scramble,random_bands}`, `--weights`, `--out-root`) |
| Figure generator | `scripts/make_paper_figures.py` (F1 per-seed spread, F2 the n=3→n=12 dissolution [headline], F3 signature map) |
| Tests (physics) | `tests/test_{physics_signatures,signature_db_consistency,physics_band_energy_loss,physics_random_band_control,physics_quarantine}.py` |
| Superseded / historical (do NOT cite as current) | `CONVERGENCE_PLAN.md` (banner), `paper/OUTLINE.md` (banner), `experiments/COLAB_*RUNBOOK.md`, the in-file SUPERSEDED §8.4/§8.7 parts of `results/band_energy_reruns/` |
| Deferred / frozen | `BACKLOG.md` · dashboard `packages/dashboard/` |
| **Persistent memory** (loaded each session) | `MEMORY.md` index → `lstm-pfd-project-state`, `no-claude-coauthor-trailers`, `lstm-pfd-workflow-preferences`, `lstm-pfd-independent-provenance`, `user-affiliation`, `portfolio-revamp-assets` |
