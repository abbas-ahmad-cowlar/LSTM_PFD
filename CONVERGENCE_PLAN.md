# LSTM-PFD Convergence Plan — Physics-First Rebuild

> **Goal**: Converge this project from a sprawling, partially-fabricated "everything platform"
> into a **lean, honest, physics-informed fault-diagnosis research project** that produces
> real, publishable results — using only the compute we actually have (CPU laptop overnight,
> office PC with basic GPU for multi-day runs, Colab occasionally).
>
> **Supersedes**: `implementation_plan.md` (the old master plan — to be archived in Phase 6).
> **Evidence base**: `audit_reports/PROJECT_AUDIT_2026-06-11.md`. Every claim there was
> verified by execution; this plan inherits its rule: **only execution evidence counts.**
> A feature "works" when a command ran and produced an artifact we can open.
>
> **Dashboard policy (decided)**: The dashboard is **frozen, not deleted**. It stays in
> `packages/dashboard/`, gets decoupled from core (Phase 2), and is rehabilitated later as a
> separate effort (Phase D, after the science is done). No UI work before then.

---

## Part I — Strategy: What This Project Is Now

### 1. The reframe: physics project, not data-science zoo

The project's genuinely rare asset is **not** the 34 models — every grad student has a model
zoo. It is the **physics-based synthetic vibration generator for journal/hydrodynamic
bearings** (Sommerfeld-scaled lubrication behavior, oil whirl, cavitation, 11-class taxonomy
including mixed faults). Nearly all public bearing-fault research uses rolling-element data
(CWRU, Paderborn); journal-bearing fault data is scarce. That scarcity is our opening.

**The publishable story** (one paper, coherent, achievable on our compute):

> *A physics-based simulation framework for journal-bearing fault diagnosis, with a
> benchmark showing that physics-informed learning beats purely data-driven models
> where it matters: less data, unseen severities, noisy signals — and produces
> explanations consistent with known fault physics.*

Four concrete contributions, each mapped to work we can actually do:

| # | Contribution | Why it's credible for us | Phase |
|---|---|---|---|
| C1 | **Open synthetic dataset + generator** for journal-bearing faults (validated spectral signatures, severity/operating-condition sweeps, dataset card, DVC) | Generator already works; needs validation tests + a v2 dataset designed for experiments | 3 |
| C2 | **Honest benchmark** of classical ML vs CNN/ResNet/Transformer vs PINN (multi-seed, statistical tests) | All infrastructure exists; just was never run | 4 |
| C3 | **Physics-informed advantage analysis**: data-efficiency curves, severity-shift OOD, noise robustness — PINN vs matched vanilla baseline | This is where PINNs actually shine in the literature; experiments are cheap (subsets, shifted test sets) | 5 |
| C4 | **Physics-consistent XAI**: do saliency/SHAP attributions concentrate at fault-characteristic frequencies? Quantified alignment metric | XAI code exists; envelope/spectral ground truth is known *because the data is synthetic* — a unique advantage of simulation | 5 |

**Honesty constraint to carry into the paper**: results are synthetic-only. We say so
prominently, position the work as "simulation framework + benchmark," and list sim-to-real
transfer as future work. Realistic venues: *Mechanical Systems and Signal Processing*,
*Measurement*, *IEEE Sensors / IEEE Access*, *Sensors (MDPI)* — not IEEE TII with fabricated
"domain expert validation." The old paper draft is quarantined (Phase 6).

### 2. The triage framework (how every keep/cut decision is made)

Classify every feature/module by **evidence level**, then apply the rules. This is the tool
to use whenever something new is found that this plan doesn't list.

**Evidence levels:**
- **WIRED** — ran end-to-end, artifact exists (e.g., signal generator, CNN training).
- **PARTIAL (≥50%)** — substantive backend exists, fails for identifiable, fixable reasons
  (e.g., HybridPINN forward bug, stale data-gen tests).
- **FACADE** — surface exists (UI/API/script), backend missing or returns placeholders
  (e.g., feature-engineering tab, `industrial_validation.py` hardcoded accuracies).
- **ASPIRATIONAL** — exists only in docs/plans (SaaS multi-tenancy, K8s production).

**Rules:**
1. WIRED → keep, protect with a test.
2. PARTIAL → fix **only if** it serves contributions C1–C4 or the core pipeline; otherwise freeze.
3. FACADE → remove the deceptive surface (the lying part), regardless of effort already spent.
4. ASPIRATIONAL → strike from docs; a one-line entry in `BACKLOG.md` at most.
5. **Out-of-scope overrides quality**: well-written code that serves the wrong goal
   (2FA, webhooks, billing hooks) is still cut. Good code in the wrong project is a liability.
6. **Nothing is destroyed**: tag `pre-convergence-2026-06` before any deletion; git history
   keeps everything recoverable. Deleting from the working tree is reversible; carrying dead
   weight is what's expensive.

### 3. The keep/cut verdict (the decisions you asked me to make)

#### Models — keep 10, cut 24

**KEEP (the curated zoo — every kept model earns a row in the paper's results table):**

| Tier | Models | Role in paper |
|---|---|---|
| Classical baselines | RandomForest, SVM, GradientBoosting (on 36 hand-crafted features) | "Feature engineering + shallow ML" reference tier; trains in minutes on CPU |
| Deep baselines | **CNN1D** (proven), **ResNet18-1D**, **one transformer: PatchTST** | "Pure data-driven" tier across 3 architecture families |
| Physics-informed | **HybridPINN** (fix first), **PhysicsConstrainedCNN**, **MultitaskPINN** | The paper's core tier; 3 variants enable a physics-ablation table |
| Aggregate | **VotingEnsemble** (soft, top-3) | One ensemble row, nearly free once others are trained |

**CUT (delete after tagging; rationale per group):**
- **EfficientNet-1D B0–B7** (8 variants) — ImageNet-scaling ported to 1D adds nothing at our
  dataset size; 8 rows of noise in any results table.
- **WideResNet-1D (4 variants), SE-ResNet** — redundant with ResNet18 for our story.
- **ViT-1D (3 sizes), TSMixer, SignalTransformer** — one transformer (PatchTST) is enough;
  transformer-architecture comparison is not our contribution.
- **CNN-LSTM, CNN-TCN, CNNTransformerHybrid (3 variants), MultiScaleCNN/Dilated, AttentionCNN
  variants, DualStreamCNN** — hybrid-architecture exploration is a different paper.
- **Spectrogram 2D family (ResNet2D/EfficientNet2D + 6 presets) + the entire TFR pipeline**
  (tfr_dataset.py ×3 classes, spectrogram trainer/evaluator/transforms) — heaviest orphaned
  subsystem; time-frequency input is a fine *future* extension, not core.
- **Contrastive stack** (SignalEncoder, ContrastiveClassifier, `training/contrastive/`,
  `contrast_learning_tfr.py`) — SimCLR pretraining is a separate research program.
- **KnowledgeGraphPINN** — speculative; three PINN variants is already a full ablation.
- **Stacking/Boosting/MoE ensembles, Early/Late Fusion** — one ensemble suffices.
- **Classical NeuralNetwork + StackedEnsemble wrappers** — redundant with the DL tier.
- **Trainer complexity serving cut models**: `knowledge_distillation.py`,
  `progressive_resizing.py`, spectrogram trainer, legacy shims (`cnn_1d.py`, `resnet_1d.py`,
  `hybrid_pinn.py` root shims, `legacy_ensemble.py`, re-export shim files) and the ~47
  registry aliases (registry shrinks to ~12 honest entries).

#### Data layer — keep 1 dataset class

**KEEP**: `BearingFaultDataset` (+ `transforms.py`, `dataloader.py`, `signal_augmentation.py`,
`signal_validation.py`, the whole `signal_generation/` package).
**CUT**: `cnn_dataset.py` (RawSignalDataset, CachedRawSignalDataset),
`streaming_hdf5_dataset.py` (datasets fit in RAM), all TFR datasets,
`data_validator.py` (orphaned MATLAB comparison), `cnn_dataloader.py` shim.

#### Evaluation / XAI — keep what C2–C4 need

**KEEP**: evaluator, cnn_evaluator, pinn_evaluator, statistical_analysis, cross_validation,
check_data_leakage, robustness_tester, confusion_analyzer, roc_analyzer, error_analysis,
benchmark. **XAI**: shap_explainer, integrated_gradients, attention/saliency visualization,
uncertainty_quantification (optional, P2 — nice paper figure).
**CUT**: spectrogram_evaluator, ensemble_evaluator (voting needs no special evaluator),
temporal_cv (no temporal split in our design), anchors, concept_activation_vectors,
counterfactual_explanations, partial_dependence (tabular-oriented), lime (SHAP+IG suffice).

#### Research / benchmark scripts

**KEEP & REPAIR**: `ood_testing.py` (real training loop — core of C3), `pinn_ablation.py` +
`pinn_comparison.py` (repair against fixed HybridPINN — core of C3), `xai_metrics.py` (C4),
`scripts/train_overnight.py` (proven), `scripts/colab/` (trim to curated zoo),
`scripts/compare_results.py`.
**CUT**: `failure_analysis.py` (np.random fake results), `industrial_validation.py`
(hardcoded placeholder accuracies), `ablation_study.py` (self-contained toy model — superseded
by pinn_ablation on real models), `experiments/cnn_experiment.py` (broken imports),
`integration/unified_pipeline.py` + friends (phase-orchestration stubs), `benchmarks/`
resource/scalability scripts unless trivially working.

#### Infrastructure

**KEEP**: one minimal `docker-compose.yml` (FastAPI inference + model volume — fix paths),
the FastAPI server (it's real), ONNX dynamic-quantization path, 2 lean CI workflows
(lint + tests on push; no docs build, no deploy).
**CUT**: Helm charts, K8s manifests, nginx/ssl service blocks, deploy.yml workflow,
`deliverables/HANDOVER_PACKAGE/` (regenerate honestly at the end), ONNX *static* quantization
stub, `site/` (stale built docs), mkdocs (until docs stabilize — a good README + docs/ folder
beats a broken docs site).

#### Dashboard (frozen — Phase D, later)

Stays in-tree, decoupled, excluded from core CI. Its own triage (login missing, XAI/NAS/
testing/feature-eng tabs are facade, services partially real) is **deferred wholesale** —
including the trivial `REFRESH_INTERVAL_MS` boot fix, which lands in Phase D.1 as its first
step. We do not spend UI hours before the science exists. (Rationale: you confirmed it ran
before March; the boot bug is recent and one-line — it will keep.)

---

## Part II — The Phases

Branch discipline: one branch per phase (`p1/stabilize`, `p2/prune`, ...), merged to `main`
at each exit gate. Tag `pre-convergence-2026-06` on `main` before P2 deletions.
`CONVERGENCE_PLAN.md` is the single source of progress: mark items `[x]` as they complete,
with a one-line note of the *evidence* (command + artifact).

---

### Phase 0 — Ratify & safety net (half a day, laptop)

- [ ] **0.1** Review Part I keep/cut lists; strike or add items (owner decision). Anything not
      explicitly kept is cut by default.
- [ ] **0.2** Tag current `main` as `pre-convergence-2026-06`; push tag.
- [ ] **0.3** Create `BACKLOG.md` (one-liners for everything aspirational we're cutting:
      SaaS, K8s, NAS, contrastive, TFR/2D models, distillation, real-data validation).
- [ ] **0.4** Immediate honesty hotfixes (tiny, do now, on `main`):
      delete fake results table in `reproducibility/README.md:62-67`;
      add "⚠️ contains unvalidated placeholder results — do not cite" header to
      `config/docs/paper/main.tex` and rename `config/docs/reports/Final_Report.pdf` →
      `Final_Report_UNVALIDATED.pdf`.
- [ ] **0.5** Pin environment: commit `requirements.lock.txt` from the working venv
      (Python 3.14.0 / torch 2.9.1). Office-GPU and Colab get their own lock files in P4.

**Exit gate**: tag exists; no doc in the repo presents invented numbers as results.

---

### Phase 1 — Stabilize the spine (3–5 days, laptop, CPU)

Fix everything PARTIAL that the kept set depends on. No new features.

- [ ] **1.1** **Fix HybridPINN forward pass** (P0). Diagnose shape mismatch
      (`mat1 4x75 vs 576x256`): the physics-feature branch produces 75-dim input where the
      fusion layer expects 576. Likely a feature-extractor/config drift during the March
      refactor. Fix → all 9 `tests/test_pinn.py` + 2 `tests/test_models.py` PINN tests green.
      *This is the single most important code fix in the project — C3 depends on it.*
- [ ] **1.2** Verify PhysicsConstrainedCNN + MultitaskPINN forward/backward with a smoke test
      (they share physics components with HybridPINN; assume broken until proven).
- [ ] **1.3** Fix pytest collection: rewrite `tests/utilities/test_training_imports.py` as a
      proper test (no module-level `sys.exit`).
- [ ] **1.4** Rewrite `tests/test_data_generation.py` against the current
      `signal_generation/` API (~20 stale tests). While there: convert them into *physics
      validation tests* where cheap (see 3.2) instead of API-echo tests.
- [ ] **1.5** Fix remaining real failures in kept scope: integration test imports
      (`test_comprehensive.py` ×3), Windows `PermissionError` in `test_models.py`
      serialization, ONNX export TypeError (`test_deployment.py`) — or mark ONNX tests
      `skipif(no onnxruntime)` and install onnxruntime in the venv.
- [ ] **1.6** Mark dashboard tests with `@pytest.mark.dashboard`, excluded by default
      (`pytest.ini` addopts `-m "not dashboard"`).
- [ ] **1.7** Establish the *trust command*:
      `pytest -q` → **0 failures** (skips allowed only with reasons).
- [ ] **1.8** Train-resume sanity: load `checkpoints/cnn/best_model.pth`, evaluate on the test
      split of `data/generated/dataset.h5`, write `results/cnn1d_baseline_eval.json` +
      confusion matrix PNG. *First-ever artifact in `results/` — proves the evaluation
      pipeline end-to-end and gives the first real number for the project.*

**Exit gate**: `pytest -q` green; HybridPINN trains 2 epochs on CPU without error
(`scripts/train_overnight.py --model pinn --epochs 2` or equivalent); `results/` non-empty.

---

### Phase 2 — The great pruning (2–3 days, laptop)

Execute the Part I cut lists. Order matters: prune → fix imports → tests stay green.

- [ ] **2.1** Delete cut models + their factory entries/aliases; registry shrinks to ~12
      entries. Update `tests/test_all_models.py` / `test_factory_wiring.py` accordingly.
- [ ] **2.2** Delete cut trainers (distillation, progressive resizing, spectrogram),
      contrastive package, fusion/ensemble extras, TFR/data-layer extras, shims.
- [ ] **2.3** Delete cut evaluation/XAI modules; delete fake-output scripts
      (`failure_analysis.py`, `industrial_validation.py`, `ablation_study.py`,
      `experiments/`, `integration/` orchestration stubs).
- [ ] **2.4** Infrastructure cut: remove helm/, kubernetes/, deploy.yml, nginx/ssl blocks
      from docker-compose; fix compose to the minimal real stack (api + model volume,
      correct `checkpoints/cnn/best_model.pth` path, correct Dockerfile context).
      `docker compose config` validates; `docker compose up api` serves `/health`
      (verify locally once; CPU image).
- [ ] **2.5** Decouple dashboard: core must import nothing from `packages/dashboard`;
      dashboard gets `packages/dashboard/requirements.txt`; CI ignores it; README states
      "dashboard: experimental, frozen — see Phase D".
- [ ] **2.6** Slim CI: two workflows — `lint.yml` (black/isort/flake8 on changed code),
      `test.yml` (pytest, current action versions, no docs build, no benchmark step).
      Both must pass on the PR for this phase.
- [ ] **2.7** Repo hygiene: delete `tmp_gitlog.txt`, `.coverage`, stale `site/`; prune the
      15+ dead remote branches (`git push origin --delete ...` after confirming merged).
- [ ] **2.8** Re-run `pytest -q` + retrain CNN1D for 2 epochs → confirm nothing kept broke.

**Exit gate**: LOC drops by roughly a third or more; `pytest -q` green; minimal compose
stack runs; zero imports from core → dashboard.

---

### Phase 3 — Physics & data hardening (4–6 days, laptop + overnight CPU runs)

Turn the generator from "plausible" to "defensible" — this is contribution C1.

- [ ] **3.1** **Document the physics**: one `docs/PHYSICS.md` consolidating the fault model
      equations (per fault: signal model, characteristic frequencies, severity scaling,
      operating-condition coupling, which coefficients are empirical and why). Source
      material exists in `data/PHYSICS_MODEL_GUIDE.md` + `utils/physics_constants.py`.
- [ ] **3.2** **Spectral-signature validation tests** (the scientific heart of C1): for each
      fault class, an automated test asserting the expected spectral content — misalignment
      → 2×/3× harmonics; imbalance → 1× with speed dependence; oil whirl → 0.42–0.48×
      sub-synchronous peak; cavitation → high-frequency burst energy; lubrication →
      Sommerfeld-dependent stick-slip signature; mixed faults → superposition of components.
      These become permanent CI tests — *the generator can never silently drift again.*
- [ ] **3.3** **Design dataset v2 for the experiments we'll run** (this design decision
      shapes Phases 4–5):
      - **Window decision**: segment 5 s signals into 1 s windows (20,480 samples) with
        group-aware splits (all windows of one signal stay in one split — leakage-checked
        via `check_data_leakage.py`). Effect: ~5× more samples, ~5× cheaper per-sample
        training — decisive for CPU/weak-GPU feasibility.
      - Stratified severity levels per fault (e.g., 3 levels × labeled) → enables
        severity-shift OOD (train on mild/moderate, test on severe).
      - Operating-condition sweep (speed/load grid, stored in metadata) → enables
        condition-shift OOD.
      - Multiple SNR variants of the test set → noise-robustness curves without retraining.
      - Fixed seeds; full metadata; dataset card updated.
- [ ] **3.4** Generate v2 overnight on the laptop (v1 took ~100 s for 2,860 signals — v2 at
      ~3–5× size is still < 1 h; HDF5 writing dominates). Validate with 3.2 suite + class
      balance + leakage check. Track with DVC.
- [ ] **3.5** Re-baseline CNN1D on v2 (overnight CPU run, windowed input) → confirms the
      windowing pipeline and gives the v2 reference number.

**Exit gate**: physics tests green in CI; `data/generated/dataset_v2.h5` + dataset card +
DVC; CNN1D v2 baseline result in `results/`.

---

### Phase 4 — The benchmark matrix (1–2 weeks wall-clock; office GPU + Colab; mostly machine time)

Contribution C2. Protocol first, then let machines work.

- [ ] **4.1** Freeze the protocol in `experiments/PROTOCOL.md`: fixed splits, 3 seeds
      (extend to 5 only if GPU time allows), same early-stopping budget (e.g., max 60 epochs,
      patience 10), same optimizer policy per family, all configs committed. No
      per-model hand-tuning beyond one documented LR sweep — this is a fairness benchmark,
      not a leaderboard chase.
- [ ] **4.2** Compute assignment:
      - **Laptop (CPU, overnight)**: classical baselines (minutes), evaluation jobs, XAI.
      - **Office PC (GPU, days)**: CNN1D, ResNet18, PatchTST, 3 PINN variants × 3 seeds
        ≈ 21 runs; with 1 s windows expect roughly 0.5–2 h/run on a basic GPU → fits in a
        few days of unattended running. `scripts/train_overnight.py` extended into
        `scripts/run_benchmark.py` (sequential queue, resume-safe, writes per-run JSON).
      - **Colab**: spillover/parallel lane using the existing (trimmed) `scripts/colab/`
        batches; also the fallback if the office GPU disappoints.
- [ ] **4.3** Run the matrix. Every run produces: checkpoint, history JSON, test metrics
      JSON, confusion matrix. `results/` becomes the project's proof.
- [ ] **4.4** Aggregate: `scripts/compare_results.py` → mean±std table, per-class F1,
      paired statistical tests (Wilcoxon over seeds), one significance-annotated figure.
- [ ] **4.5** Latency benchmark (CPU inference, batch=1) for the honest "deployability" row.
- [ ] **4.6** **Replace every `[PENDING]` in README/CHANGELOG with measured numbers.**

**Exit gate**: ≥ 10 model rows × 3 seeds of real results in `results/`; README table filled;
best model identified.

---### Phase 5 — Physics-informed experiments (1–2 weeks; the paper's heart — C3 + C4)

- [ ] **5.1** **Data efficiency** (GPU): train PINN-best and CNN/ResNet-best on 10/25/50/100%
      of training data × 3 seeds → accuracy-vs-data curves. *The classic PINN win; cheap
      because subsets train faster.*
- [ ] **5.2** **Severity-shift OOD** (GPU, uses v2 design): train on mild+moderate, test on
      severe (and reverse). Repair & use `scripts/research/ood_testing.py`.
- [ ] **5.3** **Noise robustness** (laptop): evaluate frozen Phase-4 checkpoints across the
      SNR-variant test sets → robustness curves. No retraining needed.
- [ ] **5.4** **Physics ablation** (GPU): HybridPINN with each physics-loss component
      on/off + physics-weight sweep (repair `pinn_ablation.py`). McNemar's/Wilcoxon tests.
- [ ] **5.5** **Physics-consistent XAI** (laptop): SHAP + Integrated Gradients on the
      best PINN and best vanilla model; compute attribution energy at fault-characteristic
      frequency bands vs elsewhere (we *know* ground truth — synthetic); report an
      alignment score per class + 2–3 qualitative figures. (`xai_metrics.py`, repaired.)
- [ ] **5.6** (Optional, P2) Uncertainty: MC-dropout calibration curve of best PINN —
      one figure if time permits.

**Exit gate**: four results sub-directories (data-efficiency, OOD, ablation, XAI) with JSONs
+ publication-quality figures; the central claim (where physics helps, quantified) is
supported or honestly refuted — *either outcome is publishable as long as it's real.*

---

### Phase 6 — Documentation convergence & archive (2–3 days, laptop)

- [ ] **6.1** Create `archive/` at repo root: move IDB reports (`config/docs/idb_reports/`,
      ~30 files), old audits (`CONFIG_DATA_AUDIT.md`, `REPO_STRUCTURE_AUDIT.md`),
      `implementation_plan.md`, fabricated `paper/` + `Final_Report_UNVALIDATED.pdf`, with a
      one-page `archive/README.md` explaining why each group was archived.
- [ ] **6.2** Create real `docs/` root (the location README already links to): move keepers —
      ARCHITECTURE.md, GETTING_STARTED.md, PHYSICS.md (from 3.1), PROTOCOL.md, training/
      evaluation/XAI guides from `packages/`, DEPLOYMENT (trimmed to compose-only),
      DOCUMENTATION_STANDARDS.md. Target: **≤ 20 living documents** total.
- [ ] **6.3** Rewrite README around reality: physics-first pitch, real results table,
      honest quick-start (verified by running it on a clean clone), dashboard marked frozen.
- [ ] **6.4** Delete or fix every broken doc link; drop mkdocs (or fix `docs_dir` if we
      decide to keep a site — default: drop until after the paper).
- [ ] **6.5** Doc-honesty CI guard: a simple grep test failing on "accuracy ≥ N%" claims in
      docs unless the number exists in `results/` (crude but effective tripwire).

**Exit gate**: ≤ 20 living docs; zero fabricated claims outside `archive/`; clean-clone
quick-start verified by execution.

---

### Phase 7 — Paper & reproducibility package (1–2 weeks, writing-dominant)

- [ ] **7.1** Write the paper from scratch against `results/` only (the old tex is archived;
      its references.bib may be salvaged). Structure mirrors C1–C4. Limitations section
      states synthetic-only scope plainly.
- [ ] **7.2** Figures/tables generated by committed scripts from `results/` (no hand-edited
      numbers anywhere — `generate_tables.py` exists in `reproducibility/`, repair it).
- [ ] **7.3** Reproducibility package: pinned env, seeds, DVC data, one
      `reproducibility/run_all.py` that regenerates every table/figure; verified on a clean
      machine (office PC counts).
- [ ] **7.4** Venue decision (MSSP / Measurement / IEEE Access / Sensors) once we see effect
      sizes; Zenodo deposit for dataset+code at submission time.

**Exit gate**: submitted manuscript whose every number traces to a committed artifact.

---

### Phase D — Dashboard rehabilitation (deferred; after Phase 5, parallel to 6–7 if desired)

Scoped now so it doesn't creep earlier:

- [ ] **D.1** Fix boot (`REFRESH_INTERVAL_MS` order bug + utils name-shadowing).
- [ ] **D.2** Triage pages with the Part-I framework. Likely keeps (≥50% + wired to real
      backend): experiment browser, training monitor, data-generation page, experiment
      comparison. Likely cuts: NAS, testing dashboard, feature-engineering tab, XAI tab
      (until wired to Phase-5 outputs), webhooks/2FA/API-keys surface (no login exists).
- [ ] **D.3** Wire kept pages to the *real* Phase-4/5 artifacts (results/ JSONs, checkpoints).
- [ ] **D.4** Either add the missing login or remove auth-dependent surface; decide then.
- [ ] **D.5** Re-include dashboard tests in CI once green.

---

## Part III — Operating Rules (how we avoid relapse)

1. **Execution evidence or it didn't happen.** No checkbox without a command + artifact noted.
2. **No new docs describing future work** outside `BACKLOG.md` and this plan.
3. **No new model/feature additions until Phase 5 is done.** Additions are how this repo got sick.
4. **Tests green before every merge to `main`** (`pytest -q`, dashboard excluded until Phase D).
5. **Weekly cadence suggestion**: laptop runs queued before sleep; office GPU loaded Friday →
   Monday; review artifacts, not logs.
6. **When unsure whether to cut something**: apply Part I §2; if still unsure, cut — the tag
   keeps it recoverable, and the default must favor lean.

## Effort & compute summary

| Phase | Calendar | Human effort | Machine time | Where |
|---|---|---|---|---|
| 0 Ratify | 0.5 d | 0.5 d | — | laptop |
| 1 Stabilize | 3–5 d | 3–4 d | overnight smoke runs | laptop |
| 2 Prune | 2–3 d | 2–3 d | — | laptop |
| 3 Physics/data | 4–6 d | 3–4 d | 1–2 overnights | laptop |
| 4 Benchmark | 1–2 wk | 2–3 d | ~30–60 GPU-h | office GPU + Colab |
| 5 Physics exps | 1–2 wk | 3–4 d | ~30–50 GPU-h + CPU eval | office GPU + laptop |
| 6 Docs | 2–3 d | 2–3 d | — | laptop |
| 7 Paper | 1–2 wk | writing | — | — |
| **Total** | **~7–9 weeks** | | | |

---

*Plan authored 2026-06-11 on branch `audit/project-state-2026-06`, based on
`audit_reports/PROJECT_AUDIT_2026-06-11.md`. Mark progress in this file only.*
