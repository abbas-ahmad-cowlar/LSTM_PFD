# Manuscript framing (DRAFT for owner review) — Evaluations & Datasets style

> **Scope of this doc:** the *framing skeleton only* — title options, abstract,
> contribution bullets, section outline, **claim boundaries**, and the table/figure
> list. **No full sections** until the owner signs off on the framing.
>
> **Style:** strict **NeurIPS Datasets & Benchmarks / Evaluations** (not a casual
> workshop note). **Immediate path:** arXiv + a suitable workshop (PHM /
> trustworthy-ML evaluation). **Upgrade target:** NeurIPS ED / KDD D&B **2027**.
>
> **Core claim (binding):** a synthetic journal-bearing dataset + frozen benchmark +
> a **rigorous, complete negative on the *tested* physics-informed methods** + an
> n=3→n=12 **seed-fragility** caution. **No "physics improves …" wording anywhere.**
> Every number is record-level (528) + seed-level; nothing exceeds `results/FINDINGS.md`
> §0 (ratified). **Synthetic-only — stated in title-adjacent text, abstract, intro,
> and limitations.**

---

## 1. Title — SELECTED
> **"A Synthetic Journal-Bearing Benchmark for Stress-Testing Physics-Informed Fault Diagnosis"**

(Alternatives considered, not used: "When Physics-Informed Fault Diagnosis Does Not
Help (Yet): …"; "… A Complete Negative".)

Artifact name — SELECTED: **JBFD-11** (clearer than JB-Synth / SynJB).

## 2. Abstract (full draft)
Physics-informed neural networks are increasingly proposed for bearing fault
diagnosis, but controlled, leakage-free evidence that physics priors actually *help*
is scarce — and the journal/hydrodynamic-bearing regime, unlike the rolling-element
datasets that dominate the literature (CWRU, Paderborn), has almost no public
benchmarks. We release **JBFD-11**, a physics-grounded **synthetic** journal-bearing
vibration dataset (11 fault classes, severity-stratified, record-level
leakage-checked, with SNR-20/10/5 variants) from a generator whose spectral
signatures are locked by a 34-test CI battery, together with a **frozen benchmark of
11 classical, deep, and physics-informed models, analyzed at the record level**. We
use this controlled testbed to stress-test the physics-informed methods we implement
— a band-energy spectral-consistency loss judged against a frozen healthy-class
reference, with per-sample operating conditions — across the regimes where such priors
are theorized to help: noise robustness, data-efficiency, severity out-of-distribution,
interpretability, and calibration. On every axis, **the tested physics-informed
methods show no statistically supported, seed-robust advantage** over strong
data-driven baselines. We further document how readily an apparent benefit arises and
then disappears under scrutiny: a promising n=3 analysis — a highly significant
*representative-seed* McNemar result (14–0, p=1.2e-4) — **dissolves under a
pre-registered n=12 *seed-level* test** (Wilcoxon p=0.79); a pre-registered control
that swaps the physics targets for **random non-fault frequency bands** has the lowest
mean degradation numerically yet still fails the pre-registered robustness criterion.
Our contribution is a reusable physics-grounded benchmark, a rigorous complete
negative on the tested methods, and a concrete methodological caution about seed
counts and estimands in physics-informed evaluation. We release the dataset,
generator, and analysis code, and **will** archive the trained checkpoints and full
reproducibility package on a public repository; every headline number was
independently recomputed from the raw checkpoints by separate audit scripts.
*(Synthetic-only; no real-rig validation.)*

## 3. Contribution bullets
- **A reusable, CI-locked synthetic journal-bearing dataset + generator** (11 classes,
  severity-stratified, record-level leakage-checked, SNR variants) — a controlled
  testbed for a regime with almost no public benchmarks. **[C1]**
- **A frozen, record-level 11-model benchmark** (protocol fixed in advance; analyzed
  at the 528-record, not the 2,640-window, level) on which no model — physics-informed
  or not — shows an accuracy advantage. **[C2]**  *(Reserve the word "pre-registered"
  for the §8.8 n=12 grid, whose decision rule was committed before the runs.)*
- **A rigorous, complete negative**: across noise, data-efficiency, severity-OOD,
  interpretability, and calibration, the **physics-informed methods we implement and
  test** give no advantage over data-driven baselines. **[C3]**
- **A methodological caution for physics-informed evaluation**: (i) an effect
  significant at n=3 dissolves at n=12 under a pre-registered seed-level estimand and a
  matched-strength non-physics control; (ii) a physics loss can pass a green test suite
  while silently broken. **[C4-methodology]**
- **A reproducible, content-hashed artifact set** (generator, dataset, analysis,
  provenance manifest; trained checkpoints to be archived publicly), with the headline
  numbers **independently recomputed from the raw checkpoints by separate audit
  scripts**. **[C5-repro]**

## 4. Section outline (E&D conventions)
1. **Introduction** — the gap (journal-bearing data scarcity; physics-informed
   diagnosis proposed but under-evaluated); what a controlled synthetic testbed buys;
   contributions; the headline negative + the n=3→n=12 hook. *(synthetic-only stated here.)*
2. **Related work** — vibration diagnosis (signal processing → deep learning); PINNs
   / physics-informed diagnosis; synthetic vibration data; evaluation rigor, negative
   results, and reproducibility in ML.
3. **Dataset & generator [C1]** — physics (`docs/PHYSICS.md`: 1X/2X/3X, sub-sync
   oil-whirl, low-freq lube, cavitation HF, wear broadband); 11 classes; severity
   stratification; **record-level group splits + leakage check**; SNR variants; the
   34-test spectral CI. **Datasheet** (motivation, composition, collection,
   preprocessing, uses, distribution, maintenance) — table/appendix. Limitations:
   synthetic; cavitation weakly expressed.
4. **Benchmark & protocol [C2]** — frozen protocol (Adam 1e-3 / batch 64 / ≤60 ep /
   patience 10 / 3 seeds / test-touched-once); 11 models; **why the record (528), not
   the window (2,640), is the unit**; near-ceiling results; no physics accuracy edge.
5. **Stress-testing the tested physics-informed methods [C3]** — the implemented
   method: PC-CNN + band-energy-vs-healthy-reference consistency loss (per-sample rpm);
   the five regimes; the **pre-registered §8.8 n=12 grid** (CE-only / correct / scramble
   / random-band × 12 seeds), the **seed-level estimand + fixed decision rule**, the
   result; the other regimes (§8.2 data-eff, §8.3 severity-OOD, §8.6a XAI, §8.6b
   calibration) each negative.
6. **Methodological caution [C4]** — the **n=3→n=12 dissolution** (within-seed McNemar
   14–0 p=1.2e-4 → seed-level Wilcoxon p=0.79; random non-fault bands the most robust
   arm); the broken-loss saga (five defects, green suite); recommendations (seed as
   unit; pre-register the estimand; scrambled + matched-strength non-physics controls).
7. **Reproducibility [C5]** — `results/PROVENANCE_MANIFEST.md`; content hashes; Zenodo
   checkpoint archive; the independent from-checkpoint reproduction
   (`audit_reports/…2026-06-24…`); hosting / license / maintenance plan; D&B
   reproducibility checklist.
8. **Limitations & scope** — *the* section reviewers will read: synthetic-only / no real
   rig; near-ceiling clean task; single dataset; **the negative is about the *methods we
   implemented and tested*, not a claim that no physics prior could ever help**.
9. **Conclusion.**

## 5. Claim boundaries (what we DO and do NOT claim — binding)
**We claim:**
- A released, CI-locked synthetic journal-bearing dataset + generator [C1].
- A frozen, pre-registered, record-level benchmark on which no model shows an accuracy
  advantage [C2].
- On this benchmark, **the physics-informed methods we implemented and tested** show no
  advantage over data-driven baselines on any evaluated axis [C3].
- A specific, reproduced seed-fragility result (n=3 significant → n=12 null) and the
  methodological lessons around it [C4].

**We do NOT claim (do not let any wording drift here):**
- That physics improves accuracy, noise robustness, data-efficiency, severity-OOD,
  interpretability, or calibration — **no "physics improves …" of any kind**, including
  the noise result. Not even "a spectral regularizer helped" (n=12 retired it).
- That **no** physics prior could **ever** help — the negative is bounded to the
  *implemented/tested* mechanisms (one family: a spectral-consistency loss), not a
  universal impossibility.
- Any real-world / real-rig / field performance — **synthetic-only**.
- Any window-level, within-seed, or representative-seed significance as evidence — the
  **record-level + seed-level** estimands govern; the within-seed McNemar is reported
  only as the *cautionary* confounded statistic.
- That data-driven models are "good" in absolute terms — the synthetic task is
  near-ceiling; the point is *relative* (physics adds nothing), not an accuracy claim.

## 6. Tables / figures (record-level + seed-level only)
- **T1 — benchmark (record level).** 11 models, near-ceiling, no physics accuracy edge
  (best vanilla ties CE-only PC-CNN, McNemar p=1). [`results/benchmark/summary_record_level.md`]
- **T2 — §8.8 n=12 grid.** Per-arm clean→5 dB degradation (mean/median), robust-seed
  count, seed-level Wilcoxon vs CE-only. [`results/p7_strengthen/p7_strengthen_record_level.json`]
- **T3 — the other regimes at a glance.** Data-eff §8.2 / severity-OOD §8.3 / XAI §8.6a
  / calibration §8.6b — each negative, record-level.
- **T4 — datasheet summary** (classes × severities × splits × SNR variants; hashes).
- **F1 — per-seed 5 dB degradation spread** for the four arms (visualizes the seed
  variance that fooled n=3).
- **F2 — the n=3 vs n=12 dissolution** (the headline figure: same data, the apparent
  gap vanishes as seeds grow).
- **F3 — dataset/generator overview** (classes ↔ physical signatures ↔ severity).

## 7. Decisions
- ~~Title + artifact name~~ — **DECIDED:** Title 1 + **JBFD-11**.
- ~~Deployment (C5)~~ — **DECIDED (review):** keep out of the main paper; **appendix
  at most**.
- **Still open:** author/affiliation; arXiv categories (cs.LG + eess.SP); LaTeX
  template (a NeurIPS-style class serves arXiv + a workshop + a 2027 D&B resubmission).
- **Encoding:** this file is clean UTF-8 (`—`, `§`, `→`, `–`); any `â†`/`Â§` you see is
  a copy-paste artifact, not in the source. The manuscript is LaTeX, where these become
  `\textemdash`, `\S`, `\rightarrow`, etc. — no raw Unicode in the `.tex`.
