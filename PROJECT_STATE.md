# PROJECT STATE — the handoff document

> **Purpose**: cold-start context for anyone (human or AI assistant) picking up
> this project in a fresh session. Read this first, then `CONVERGENCE_PLAN.md`
> for step-level detail. **Update this file at every phase gate and at the end
> of every working session.**
>
> Last updated: **2026-06-13** (after Gate 4; Phase 5 starting)

---

## 1. The big picture

**What this project is**: a physics-based synthetic-data research platform for
**journal/hydrodynamic bearing fault diagnosis** (11 fault classes). Its rare
asset is the physics-grounded vibration signal generator
(`data/signal_generation/` — Sommerfeld-scaled lubrication, oil whirl,
cavitation; documented in `docs/PHYSICS.md`, enforced by a 34-test CI battery).
Public bearing research is dominated by rolling-element data; journal-bearing
data is scarce — that's our opening.

**End goal**: a submitted, honest paper:
> *A physics-based simulation framework for journal-bearing fault diagnosis,
> with a frozen-protocol benchmark showing where physics-informed learning
> beats purely data-driven models: less data, unseen severities, noisy
> signals — with explanations consistent with known fault physics.*

Contributions: **C1** validated open dataset+generator · **C2** honest
multi-seed benchmark · **C3** physics-advantage analysis (data-efficiency,
severity-OOD, noise) · **C4** physics-consistent XAI · **C5** deployment
appendix. Target venues: MSSP / Measurement / IEEE Access / Sensors.
**Scope honesty**: synthetic-only; no real-world validation (stated in README).

**History you must know**: in June 2026 an audit
(`audit_reports/PROJECT_AUDIT_2026-06-11.md`) found this repo full of
fabricated results (a paper claiming 98.1% accuracy with zero experiments),
~130K LOC of largely unvalidated code, a broken flagship PINN, and one
half-trained model. Everything since is a phased rebuild under one rule:
**only execution evidence counts** — every number traces to an artifact in
`results/` with git-SHA provenance. The pre-rebuild code is preserved at tag
`pre-convergence-2026-06`.

## 2. Current status (Gates 0–4 PASSED, merged to `main`)

| Phase | Result |
|---|---|
| 0–1 Stabilize | Fake numbers purged; HybridPINN forward fixed (extract_features contract); suite 45 failed → 0 failed |
| 2 Prune | ~34.5K LOC deleted; model registry 81 → 11 honest entries (tier system, Part I §3–4 of plan) |
| 3 Physics & data | `docs/PHYSICS.md` (owner-approved); 34 spectral-signature CI tests; **dataset_v2.h5** (3,520 records, exact class×severity stratification, SNR-20/10/5 test variants, per-split metadata, DVC); CNN1D windowed baseline 90.53% |
| 4 Benchmark | **Full frozen-protocol matrix** (`experiments/PROTOCOL.md`, ratified): see table below; deployment appendix; README carries real numbers |

**The benchmark table** (test acc, 2,640 one-second windows, 3 seeds —
full: `results/benchmark/summary.md`):

- voting_ensemble **96.48** > resnet18 **96.14±0.28** ≈ cnn_lstm **96.12±0.16**
  ≈ physics_constrained_cnn **95.98±0.36** (McNemar p>0.2 — statistical tie)
- classical bar: RandomForest **94.61±0.05** (36 expert features)
- cnn1d 91.94±2.84 · multitask_pinn 90.28 · hybrid_pinn 90.04 · patchtst 89.85
  · attention_cnn 89.37±4.82 (one seed collapsed mid-training — recorded as-is)
- Deployment: ResNet18 → ONNX FP32 **13 ms/window CPU**; INT8 4× smaller but
  10–15× slower (honest negative); FastAPI smoke passed serving the ONNX.

**Key open scientific threads**:
1. Physics-as-constraint ties best vanilla on clean data; the physics *win*
   must come (if anywhere) from Phase 5's regimes.
2. **hybrid_pinn caveat**: protocol fed it constant default metadata → its
   physics branch was starved. v2 stores TRUE per-record operating conditions
   → pre-registered Phase-5 experiment.
3. Healthy class is the weakest everywhere (incipient faults ≈ noise floor in
   1 s windows) — severity-graded analysis material.

## 3. What's next: Phase 5 (in progress) → 6 → 7

**Phase 5 — physics experiments** (branch `p5/physics-exp`; pre-registrations
in `PROTOCOL.md` §8 BEFORE each run — never run first):
- 5.3 Noise robustness (laptop, no training): all 24 frozen checkpoints ×
  test_snr20/10/5 groups → degradation curves. *(LAUNCHED detached 2026-06-13,
  ~2–3 h; watch `logs/noise_robustness.log`; then
  `python scripts/run_noise_robustness.py --summarize-only` regenerates summary)*
- 5.1 Data efficiency (GPU): best-physics + best-vanilla × {10,25,50,100}%
  train × 3 seeds.
- 5.2 Severity-shift OOD (GPU): train incipient+mild+moderate → test severe
  (and reverse) — pure metadata filters on v2.
- 5.4 PINN ablation (GPU): physics-loss terms on/off + weight sweep.
- 5.5 XAI alignment (laptop): SHAP/IG attribution energy at known fault
  frequencies; attention maps.
- 5.6 MC-dropout calibration (laptop).
- NEW (pre-registered): hybrid_pinn with TRUE metadata vs constant-default.

**Phase 6**: docs consolidation (archive `config/docs/idb_reports`, fabricated
paper to `archive/`, real `docs/` root, ≤20 living docs).
**Phase 7**: paper written from scratch against `results/` only; repro package;
Zenodo. **Phase D** (frozen): dashboard rehab — boots, but untouched until the
science is done.

## 4. How to operate (conventions that keep this honest)

- **Plan of record**: `CONVERGENCE_PLAN.md` — checkboxes only with
  `(evidence: command → artifact)` notes; gates close with a quoted evidence
  block; phases live on `pN/...` branches, merged to `main` at gates.
- **Protocol discipline**: `experiments/PROTOCOL.md` is FROZEN; changes are
  dated amendments in §7. Pre-register Phase-5 experiments before running.
- **Artifacts**: every run writes `metrics.json` (with git SHA, host, seed,
  config) under `results/`; small JSONs/MD/PNGs are committed, checkpoints/h5
  are not (DVC for data). A run is COMPLETE iff its metrics.json exists —
  all runners skip complete runs and resume interrupted ones.
- **Long jobs**: NEVER session-bound. Windows:
  `Start-Process -FilePath <venv python> -ArgumentList <script> -WindowStyle Hidden`.
  Colab: results dir symlinked/backed up to Google Drive (runbooks:
  `experiments/OFFICE_PC_RUNBOOK.md` incl. Colab appendix).
- **Environment quirks**: Windows py3.14 venv (`requirements.lock.txt`);
  emoji prints crash cp1252 console → `PYTHONIOENCODING=utf-8`; ONNX export
  needs `dynamo=False`; `logs/` is gitignored (scripts mkdir it).
- **Trust command**: `pytest -q` (240 passed, 0 failed; dashboard tests
  deselected while frozen).

## 5. Key files map

| What | Where |
|---|---|
| Master plan + progress tracker | `CONVERGENCE_PLAN.md` |
| This handoff | `PROJECT_STATE.md` |
| Audit (why the rebuild) | `audit_reports/PROJECT_AUDIT_2026-06-11.md` |
| Physics reference (normative) | `docs/PHYSICS.md` + `tests/test_physics_signatures.py` |
| Dataset v2 design / card | `experiments/DATASET_V2.md`, `dataset_card.yaml` |
| Frozen protocol (+amendments, pre-regs) | `experiments/PROTOCOL.md` |
| Benchmark results | `results/benchmark/summary.{md,json,png}` + per-run dirs |
| Deployment appendix | `results/deployment/appendix.md` |
| Runners | `scripts/run_benchmark.py`, `run_classical_baselines.py`, `aggregate_benchmark.py`, `generate_dataset_v2.py`, `evaluate_checkpoint.py` |
| GPU/Colab runbook | `experiments/OFFICE_PC_RUNBOOK.md` |
| Deferred ideas | `BACKLOG.md` |
