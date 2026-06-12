# PROJECT STATE — the session handoff document

> **READ THIS FIRST.** This file is the complete working memory of the LSTM-PFD
> convergence effort, written as a cold-start prompt for a fresh AI session
> (or a new collaborator). It covers: who the owner is and how to work with
> him, the compute setup and its quirks, the full history with numbers, the
> live state of running work, and exactly what to do next.
> Step-level detail lives in `CONVERGENCE_PLAN.md`; this is the narrative + context.
>
> **Maintenance duty**: update this file at every phase gate and at the end of
> every working session. It is linked from the README.
>
> Last updated: **2026-06-13, end of session 1** (Phase 5 in progress;
> noise-robustness queue at 23/24 on the laptop; Phase-5 GPU queue built and
> awaiting the owner's Colab session)

---

## 0. If you are a fresh session, do this first

1. Read this file fully, then skim `CONVERGENCE_PLAN.md`'s Progress Tracker
   and the Phase-5 section.
2. Check live state:
   - `git branch --show-current` (expect `p5/physics-exp`; `main` holds Gates 0–4)
   - `find results/noise_robustness -name metrics.json | wc -l` → 24 = §8.1
     evaluation done; if `results/noise_robustness/summary.md` is missing, run
     `./venv/Scripts/python.exe scripts/run_noise_robustness.py --summarize-only`
   - Ask the owner whether he ran the Phase-5 Colab session
     (`results/phase5/...` arriving via Google Drive download, 45 runs expected).
3. Run the trust command before changing anything: `pytest -q`
   (expect ~240 passed, 0 failed, 6 deselected).
4. Never mark plan checkboxes without `(evidence: command → artifact)`.

## 1. The owner — who you're working with

**Syed Abbas Ahmad** (git author; GitHub `abbas-ahmad-cowlar`; physics
background — comfortable with physics reasoning, ratified the fault-model
equations personally). Communication style and working agreement, learned over
this project:

- **He is the decision-maker at gates.** Phases need his explicit sign-off at
  owner-gated steps (he ratified: the tier system, PHYSICS.md, DATASET_V2.md,
  PROTOCOL.md). Quote his ratification in the doc when he gives it.
- **He deeply distrusts unverified claims** — this repo burned him with
  fabricated results before. NEVER present a number without an artifact path.
  He explicitly mandated: verify by execution, not by reading code.
- **He has attachment to impressive features** ("my heart doesn't want to let
  go") — the Tier system (Part I §3 of the plan) was built to resolve exactly
  this. If keep/cut tension reappears, use tiers, not arguments.
- **He runs the hands-on compute** (Colab sessions, office PC) but is not a
  shell expert: he pasted Windows backslash paths into bash (`scripts\run...`
  fails on Linux), ran notebook cells without `!`, created a fake Drive mount
  with `mkdir` before mounting, and hit the `ln -sfn`-into-existing-dir trap.
  **Always give him exact copy-paste commands, platform-correct, with the
  expected output described.** Runbooks must be cell-by-cell.
- He works in stretches (office during day, laptop evenings; "came back from
  office, it's the next day"). Long-running jobs must survive his absence and
  session closures.
- He asks good skeptical questions ("is it going okay?", "what if results are
  boring?") — answer with data and honest scenario analysis, not reassurance.

## 2. Compute setup — exact, with quirks

### 2.1 Laptop (primary; where Claude Code runs)
- Windows 11 Pro, repo at `C:\Users\COWLAR\projects\lstm-pfd`, PowerShell +
  Git Bash available. venv: **Python 3.14.0, torch 2.9.1+cpu** — NO GPU.
  Locked in `requirements.lock.txt`.
- Speed: ~10 min/epoch for windowed CNN1D training (12,320 windows); ~60 s to
  evaluate one checkpoint over 3 SNR test sets. Fine for: evaluation, XAI,
  classical ML, aggregation; overnight-only for training.
- **Quirks**: (a) cp1252 console — any emoji/Unicode print crashes; set
  `PYTHONIOENCODING=utf-8` or avoid fancy chars in print/log/write_text
  (always pass `encoding='utf-8'` to write_text); (b) `torch.onnx.export`
  needs `dynamo=False` (onnxscript broken on py3.14); (c) `logs/` is
  gitignored → scripts must mkdir it; (d) NamedTemporaryFile reopen fails on
  Windows — use TemporaryDirectory.
- **Detached-launch rule (hard lesson, 8h lost)**: session-bound
  `run_in_background` dies with the Claude session. Long jobs:
  `Start-Process -FilePath <venv python> -ArgumentList <script>,... -WorkingDirectory <repo> -WindowStyle Hidden -RedirectStandardOutput logs\x.log -RedirectStandardError logs\x.err`
  All training/eval scripts are resume-safe (metrics.json marks completion;
  checkpoints carry optimizer state).

### 2.2 Google Colab (the de-facto GPU; free tier, Tesla T4 15.6 GB)
- **75× faster than the laptop** (cnn1d: 8 s/epoch vs 600 s). The entire
  24-run Phase-4 matrix took ~75 minutes. Phase-5's ~45 runs ≈ 4–6 h.
- Sessions are ephemeral (~12 h cap, idle disconnects kill the VM). The
  owner's Colab workflow (he runs it; you prepare cells): clone repo →
  checkout phase branch → copy `dataset_v2.h5` from Drive → **symlink
  `results/<dir>` to Drive BEFORE the first run** (if the dir already exists
  locally, `ln -sfn` silently creates the link INSIDE it — this burned us; a
  10-min `shutil.copytree` backup loop in a notebook cell is the fallback) →
  smoke run → main queue cell (rerun same cell after disconnect; it resumes).
- Owner's Google Drive: `MyDrive/lstm-pfd/` holds the dataset at
  `data/dataset_v2.h5` (moved into the `data/` subfolder 2026-06-13) and
  result folders (`results_benchmark`, `results_phase5`). Results come home by
  Drive web-UI download → extract into `results/...` on the laptop → Claude
  verifies counts + provenance before using.
- Runbooks: `experiments/OFFICE_PC_RUNBOOK.md` — benchmark appendix (done) and
  **Phase-5 appendix (ready, not yet executed)**.

### 2.3 Office PC (basic GPU, can run for days)
- **Never actually used** — Colab proved faster and sufficient. The Windows
  runbook (main body of OFFICE_PC_RUNBOOK.md) exists if ever needed. Don't
  assume any state on that machine.

## 3. The big picture (unchanged since ratification)

A physics-based synthetic-data research platform for **journal/hydrodynamic
bearing fault diagnosis** (11 classes; French fault names: sain,
desalignement, desequilibre, jeu, lubrification, cavitation, usure, oilwhirl
+ 3 mixed). The crown jewel is the physics-grounded signal generator
(`data/signal_generation/`; normative doc `docs/PHYSICS.md`; enforced by the
34-test CI battery `tests/test_physics_signatures.py` — the generator cannot
silently drift). Journal-bearing data is scarce publicly (literature is
rolling-element: CWRU/Paderborn) — that's the niche.

**End goal — a submitted honest paper**:
> *A physics-based simulation framework for journal-bearing fault diagnosis,
> with a frozen-protocol benchmark showing where physics-informed learning
> beats purely data-driven models: less data, unseen severities, noisy
> signals — with explanations consistent with known fault physics.*

Contributions: C1 dataset+generator · C2 frozen benchmark · C3 physics
advantage (data-efficiency / severity-OOD / noise) · C4 physics-consistent
XAI · C5 deployment appendix. Venues: MSSP / Measurement / IEEE Access /
Sensors. **Synthetic-only — stated everywhere; no real-world validation.**

**Origin story (why everything is the way it is)**: the 2026-06-11 audit
(`audit_reports/PROJECT_AUDIT_2026-06-11.md`) found fabricated results (a paper
claiming 98.1% with zero experiments run), ~130K LOC mostly unvalidated, a
broken flagship PINN, one half-trained model, 45 failing tests. The rebuild
runs under one prime rule: **only execution evidence counts**. Old code is
recoverable at tag `pre-convergence-2026-06`.

## 4. Complete history (Gates 0–4 PASSED, merged to main)

| Phase / Gate | What happened | Hard numbers |
|---|---|---|
| Audit (06-11) | 5 parallel agent audits + direct verification; fabrications inventoried | 45 failed/220 passed tests; results/ empty |
| 0 Ratify | Tag `pre-convergence-2026-06`; fake-results purge (reproducibility README table, paper warning header); BACKLOG.md; env lock | — |
| 1 Stabilize | **HybridPINN forward fixed** (silent `hasattr(backbone,'fc')` head-strip no-op → explicit `extract_features()` + `include_head=False` contract on CNN1D/ResNet1D; same fix MultitaskPINN); pytest-collection crasher rewritten; data-gen tests rewritten by agent (55); ONNX dynamo=False; API datetime-serialization bug; dashboard boot fix (timeboxed) | suite → 328 passed/0 failed; first artifact: CNN1D v1 eval 86.48% |
| 2 Prune | Registry 81→**11 honest keys** (8 T1 + 3 T2); deleted 23 architectures, contrastive/TFR/2D stacks, fake-output scripts (np.random "results"), experiments/, integration/, helm/k8s, 6 broken workflows → 2 honest ones; dashboard decoupled (frozen) | core LOC −32% (~34.5K lines deleted); suite 206 green |
| 3 Physics & data | `docs/PHYSICS.md` (owner-ratified; its draft kurtosis claim was REFUTED by the test battery and corrected — the honesty loop works); 34 spectral CI tests; **dataset_v2.h5**: 3,520 records, 320/class, EXACT 80/class/severity stratification, record-level splits with per-split metadata (fixes v1's split-shuffled-metadata defect), SNR-20/10/5 test variants, leakage-checked, DVC; `WindowedView` (1 s windows, group-aware) | CNN1D v2 baseline **90.53%** / F1 0.9013 (2,640 test windows) |
| 4 Benchmark | `experiments/PROTOCOL.md` ratified+FROZEN (Adam 1e-3, batch 64, ≤60 ep, patience 10, no schedulers/aug, 3 seeds, test-touched-once); classical tier on laptop; deep 8×3 matrix on Colab T4 (~75 min); ensemble + aggregation + McNemar/Wilcoxon; deployment appendix; README/CHANGELOG carry real numbers (zero PENDINGs) | table below |

**The Phase-4 benchmark table** (test acc %, 2,640 windows, mean±std/3 seeds;
`results/benchmark/summary.md`):

| Model | Acc | Note |
|---|---|---|
| voting_ensemble | **96.48** | cnn_lstm+pc_cnn+resnet18 members |
| resnet18 | 96.14±0.28 | statistical tie with next two (McNemar p>0.2) |
| cnn_lstm | 96.12±0.16 | namesake; 'simple' backbone (see incident I4) |
| physics_constrained_cnn | 95.98±0.36 | **physics-OFF in Phase 4!** (see §8.0 discovery) |
| RandomForest (36 feats) | 94.61±0.05 | the classical bar; SVM/GB 94.05 |
| cnn1d | 91.94±2.84 | high seed variance |
| multitask_pinn | 90.28±0.64 | |
| hybrid_pinn | 90.04±0.51 | constant-metadata caveat (§8.5 experiment) |
| patchtst | 89.85±0.19 | |
| attention_cnn | 89.37±4.82 | 1 seed collapsed to 9.09% mid-training |

Deployment (C5): ResNet18→ONNX FP32 **13 ms/window CPU** (parity 1.5e-4);
INT8 4× smaller but **10–15× slower** (honest negative — dynamic quant doesn't
help conv nets); FastAPI smoke served the ONNX and classified a real record
correctly. `results/deployment/appendix.md`.

## 5. Phase 5 — CURRENT WORK (branch `p5/physics-exp`)

All experiments **pre-registered in `experiments/PROTOCOL.md` §8** (hypothesis,
metric, decision rule — written BEFORE running; never reverse this order).

**§8.0 DISCOVERY (changes interpretation of Phase 4)**: code review found
`physics_constrained_cnn.forward()` is a plain CNN — physics enters only via
`compute_physics_loss()`, which Phase-4 training never called. So Phase-4's
pc_cnn rows are the **w=0 (physics-off) arm**. §8.2/8.3 run it with physics ON
at fixed w=0.3; §8.4 sweeps w (its w=0 arm = Phase-4 runs, reused).

| Prereg | Experiment | Compute | Status |
|---|---|---|---|
| §8.1 | Noise robustness: all 24 frozen checkpoints × SNR-20/10/5 | laptop | **DONE 24/24, summary written** (`results/noise_robustness/summary.{md,json,png}`). Headline: family mean degradation clean→5dB: **physics 8.51 vs vanilla 15.54** (prereg rule favors physics) — BUT outlier-sensitive: attention_cnn's collapse (Δ50.6) drags the vanilla mean; excluding it vanilla≈6.8 beats physics. Most robust single model is VANILLA resnet18 (Δ1.70, still 94.4% at 5 dB); clean-data co-champion cnn_lstm is noise-fragile (Δ12.7). pc_cnn Δ4.99 (best physics). Full honest read → FINDINGS.md at Gate 5; remember pc_cnn here is physics-OFF (§8.0) — §8.4's w>0 arms at 5 dB are the real physics-noise test. One incident en route (I4) |
| §8.2 | Data efficiency: pc_cnn(w=0.3) & resnet18 × {10,25,50,100}% × 3 seeds | Colab | script ready, NOT run |
| §8.3 | Severity-OOD: both directions (train low→test severe; train high→test incipient) | Colab | script ready, NOT run |
| §8.4 | Physics-weight ablation w∈{0.1,0.3,1.0} (+Phase-4 as w=0), eval clean+5 dB | Colab | script ready, NOT run |
| §8.5 | hybrid_pinn with TRUE per-record metadata (rpm/load/viscosity from v2; mapping documented in script header) vs Phase-4 constant-defaults | Colab | script ready, NOT run |
| §8.6 | XAI alignment (SHAP/IG energy in PHYSICS.md frequency bands) + MC-dropout calibration | laptop | **scripts NOT built yet** — build after Colab results land |

- Colab queue: `scripts/run_phase5_gpu.py` (~45 runs, resume-safe,
  `--only <experiment>` subsets, `--smoke`). Runbook: Phase-5 appendix in
  `experiments/OFFICE_PC_RUNBOOK.md`. **The owner is about to run this.**
- Optional in same session: Tier-2 benchmark rows
  (`python scripts/run_benchmark.py --models multi_scale_cnn se_resnet18 signal_transformer`).
  Old pre-convergence data suggests se_resnet18 is strong, multi_scale_cnn may
  collapse — both expectations are recorded.

**After Colab results come home (Drive download → `results/phase5/`)**:
verify 45 metrics.json + provenance → build aggregation/analysis per prereg
decision rules → build & run §8.6 on laptop → FINDINGS.md (owner reviews) →
Gate 5 → merge → Phase 6 (docs consolidation/archive) → Phase 7 (paper from
scratch against results/ only).

## 6. Incidents & lessons (do not repeat these)

- **I1 — Session-bound process death**: P3.5 training died when the Claude
  session restarted; laptop was blameless. → detached `Start-Process` +
  `--resume` everywhere. (Cost: ~8 h wall time.)
- **I2 — Colab `ln -sfn` trap**: symlinking onto an EXISTING dir puts the link
  inside it; results went to ephemeral disk. → symlink BEFORE first run;
  fallback = notebook-cell copytree loop every 10 min (that loop saved the
  whole Phase-4 matrix).
- **I3 — Fake Drive mount**: `mkdir -p /content/drive/...` before `drive.mount`
  created a local impostor dir and the mount then refused. → mount first;
  if blocked: `mv /content/drive /content/drive_fake`, mount, rescue, delete.
- **I4 — Factory/class default desync**: fixing CNNLSTM's stale backbone
  import made `create_cnn_lstm`'s `backbone='resnet18'` default suddenly real,
  so fresh models stopped loading the benchmark ('simple'-backbone)
  checkpoints — killed the noise queue at 7/24. → BOTH class and factory
  defaults pinned to the benchmarked arch; noise runner got per-run
  try/except. General rule: after any model-code change, verify
  `create_model(key)` still loads the recorded checkpoints.
- **I5 — cp1252**: emoji/Greek in prints or write_text without encoding crash
  on this laptop (killed a leakage-check print and the appendix writer).
- **I6 — Old-era methodology**: the owner's pre-convergence Colab run (March)
  showed full-5s-records saturate (~100% for ResNets, se_resnet18=100%) —
  cited as evidence FOR the 1 s windowing decision; attention_cnn NaN'd there
  too (instability is architectural, reportable); multi_scale_cnn collapsed
  there too.

## 7. Conventions (the honesty machinery)

1. Only execution evidence counts; numbers must trace to `results/` artifacts
   with git SHA + host + seed provenance.
2. `CONVERGENCE_PLAN.md` checkboxes need `(evidence: …)`; gates get evidence
   blocks; tracker table updated every session.
3. PROTOCOL is frozen — changes are dated §7 amendments; Phase-5 experiments
   are pre-registered in §8 before running.
4. Phase branches `pN/...` → merge to `main` at gates. Current: `p5/physics-exp`.
5. Suite green before merge: `pytest -q` (dashboard tests deselected — frozen
   until Phase D; the dashboard BOOTS but is otherwise untouched).
6. Anti-regrowth: tiers are fixed-size (T1=12 rows, T2=3, cap forever);
   promoting in requires demoting out. Cut things live at the tag + BACKLOG.md.
7. results/: small json/md/png/csv committed; checkpoints + h5 stay out of
   git (DVC for the dataset).

## 8. Key files map

| What | Where |
|---|---|
| Step-level plan + tracker + gate evidence | `CONVERGENCE_PLAN.md` |
| Audit (origin) | `audit_reports/PROJECT_AUDIT_2026-06-11.md` |
| Physics (normative) + CI battery | `docs/PHYSICS.md`, `tests/test_physics_signatures.py` |
| Dataset v2 design / card / file | `experiments/DATASET_V2.md`, `dataset_card.yaml`, `data/generated/dataset_v2.h5` (+DVC) |
| Protocol + amendments + §8 preregs | `experiments/PROTOCOL.md` |
| Benchmark results / deployment | `results/benchmark/summary.{md,json,png}`, `results/deployment/appendix.md` |
| Noise robustness (§8.1) | `results/noise_robustness/` (+ summary after completion) |
| Phase-5 GPU queue + runbooks | `scripts/run_phase5_gpu.py`, `experiments/OFFICE_PC_RUNBOOK.md` |
| Other runners | `scripts/run_benchmark.py`, `run_classical_baselines.py`, `aggregate_benchmark.py`, `run_noise_robustness.py`, `generate_dataset_v2.py`, `evaluate_checkpoint.py`, `deployment_appendix.py`, `train_baseline_v2.py`, `pinn_sanity_train.py` |
| Deferred ideas | `BACKLOG.md` · Frozen dashboard: `packages/dashboard/` (Phase D) |
