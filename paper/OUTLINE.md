# Manuscript outline + abstract (DRAFT for owner review)

> Target framing: **Datasets & Benchmarks / Evaluations style** (arXiv first → a
> trustworthy-ML / PHM workshop; NeurIPS ED / KDD D&B 2027 as the upgrade target;
> IEEE Access only later if real-rig data is added). Everything here is drawn from
> the **ratified `results/FINDINGS.md` §0** (complete negative) — no claim beyond it.
> **Synthetic-only; state everywhere.** This is the framing skeleton to lock before
> writing full sections; numbers are record-level + seed-level per the manifest.

## Title (candidates — pick one)
1. **"A Synthetic Journal-Bearing Benchmark for Stress-Testing Physics-Informed Fault Diagnosis"**
2. "When Physics-Informed Fault Diagnosis Does Not Help: A Controlled Synthetic Benchmark and a Seed-Robust Negative Result"
3. "Stress-Testing Physics-Informed Learning on a CI-Locked Synthetic Journal-Bearing Benchmark"

Dataset/artifact name (pick one, used throughout): **JB-Synth** · JBFD-11 · SynJB.

## Abstract (full draft)
Physics-informed neural networks are widely proposed for bearing fault diagnosis,
yet rigorous, leakage-controlled evidence that physics priors actually *help* is
scarce — and the journal/hydrodynamic-bearing regime, unlike the rolling-element
datasets that dominate the literature (CWRU, Paderborn), has almost no public
benchmarks. We release **[NAME]**, a physics-grounded **synthetic** journal-bearing
vibration dataset (11 fault classes, severity-stratified, record-level
leakage-checked, SNR-20/10/5 variants) produced by a CI-locked signal generator,
together with a **frozen, pre-registered, record-level benchmark** of 11 classical,
deep, and physics-informed models. Using this controlled testbed we stress-test
physics-informed learning across the regimes where it is theorized to help — noise
robustness, data-efficiency, severity out-of-distribution, interpretability, and
calibration — and find **no advantage over strong data-driven baselines on any of
them**. We further show how readily such a benefit can be *imagined*: a
same-architecture noise-robustness result that is statistically significant at n=3
seeds (within-seed McNemar p=1.2e-4) **dissolves at n=12** (seed-level Wilcoxon
p=0.79), and a pre-registered control that replaces the physics targets with
**random non-fault frequency bands** is the *most* robust arm of all. Our
contribution is three-fold: a reusable physics-grounded benchmark, a **rigorous,
complete negative** on physics-informed learning in this setting, and a concrete
**methodological caution** about seed counts and estimands in physics-informed
evaluation. All artifacts — generator, content-hashed dataset, checkpoints, and
analysis — are released, and every headline number was independently reproduced from
raw checkpoints by external auditors. *(Synthetic-only; no real-rig validation.)*

## Contributions (the frozen claims — FINDINGS §0)
1. **C1** A CI-locked physics-grounded synthetic journal-bearing dataset + generator
   (11 classes, severity-stratified, record-level leakage-checked, SNR variants).
2. **C2** A frozen, pre-registered, **record-level** 11-model benchmark — no model,
   physics-informed or not, shows an accuracy advantage (near-ceiling; best vanilla
   ties CE-only PC-CNN, McNemar p=1).
3. **C3** A **rigorous, complete negative**: across noise, accuracy, data-efficiency,
   severity-OOD, interpretability, and calibration, the implemented physics-informed
   mechanisms give no advantage.
4. A **methodological caution**: (a) a result significant at n=3 dissolves at n=12
   (seed unit, pre-registered seed-level estimand, matched-strength non-physics
   control); (b) a physics loss that passed a green suite while silently broken.
5. A **released, externally-reproduced** artifact set (manifest, hashes, Zenodo,
   independent from-checkpoint audit).

## Section outline (D&B / Evaluations style)
1. **Introduction** — journal-bearing data scarcity (literature = rolling-element);
   the appeal and the unproven-ness of physics-informed diagnosis; what a *controlled
   synthetic* testbed buys; contributions (above). End with the headline negative + the
   n=3→n=12 hook.
2. **Related work** — vibration diagnosis (signal processing, deep learning); PINNs
   for diagnosis; synthetic vibration data; negative results & evaluation rigor / the
   reproducibility-in-ML literature.
3. **Dataset & generator (C1)** — `docs/PHYSICS.md` physics (1X/2X/3X, sub-sync
   oil-whirl, low-freq lube, cavitation HF, wear broadband); 11 classes; severity
   stratification; record-level group splits + leakage check; SNR variants; the
   34-test spectral CI. Datasheet. **Limitations:** synthetic; cavitation weakly
   expressed.
4. **Benchmark & protocol (C2)** — frozen protocol (Adam 1e-3 / batch 64 / ≤60 ep /
   patience 10 / 3 seeds / test-touched-once); 11 models; **why the record (528) not
   the window (2640) is the unit**; near-ceiling table; no physics accuracy advantage.
5. **Stress-testing physics-informed learning (C3, the negative)** — the PC-CNN +
   band-energy-vs-healthy-reference loss (per-sample rpm); the regimes; **the §8.8 n=12
   noise grid**: CE-only / correct / scramble / random-band × 12 seeds; the
   pre-registered **seed-level** estimand + decision rule; result table (degradation,
   robust-seed counts, Wilcoxon) → no arm beats CE-only; data-eff / severity-OOD /
   §8.6a XAI / §8.6b calibration each negative.
6. **A methodological caution** — the **n=3→n=12 dissolution** (within-seed McNemar
   14–0 p=1.2e-4 → seed-level Wilcoxon p=0.79); random non-fault bands are the most
   robust arm; the broken-loss saga (5 defects, green suite); recommendations for
   physics-informed evaluation (seed as unit, pre-register the estimand, scrambled +
   matched-strength non-physics controls).
7. **Reproducibility** — `results/PROVENANCE_MANIFEST.md`; content hashes; Zenodo
   checkpoint archive; the independent from-checkpoint audit (`audit_reports/…2026-06-24…`).
8. **Limitations** — synthetic-only / no real rig; near-ceiling clean task; single
   dataset; scope = the *implemented* mechanisms, not "no physics prior could ever
   help"; deployment (C5) as an applied note.
9. **Conclusion.**

## Tables / figures (record-level + seed-level only)
- **T1** benchmark record-level (11 models; near-ceiling; no physics edge). [`results/benchmark/summary_record_level.md`]
- **T2** §8.8 n=12: per-arm clean→5 dB degradation (mean/median), robust-seed count, Wilcoxon vs CE-only. [`results/p7_strengthen/p7_strengthen_record_level.json`]
- **T3** the other negatives at a glance (data-eff §8.2, severity-OOD §8.3, XAI §8.6a, calibration §8.6b).
- **F1** per-seed 5 dB degradation spread for the 4 arms (shows the seed variance that fooled n=3).
- **F2** the n=3 vs n=12 dissolution (the headline figure).
- **F3** generator/dataset overview (classes × signatures × severity).

## Open writing decisions for the owner
- Dataset name; title choice; author/affiliation; arXiv category (cs.LG + eess.SP).
- Whether to include the deployment (C5) appendix (minor, applied).
- Format: LaTeX (NeurIPS/D&B style file works for arXiv + workshop).
