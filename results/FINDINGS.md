# FINDINGS — Phase 5/6/7 synthesis (DRAFT for owner re-ratification)

> Honest synthesis of every Phase-5 experiment, the Phase-6 physics remediation,
> and the Phase-7 §8.8 n=12 strengthen grid. Every number traces to a committed
> artifact under `results/`. This memo freezes the list of claims the paper may and
> may not make. **Synthetic-only study — no real-world validation, stated everywhere.**
>
> Status: **DRAFT — 2026-06-24, AWAITING OWNER RE-RATIFICATION.** The Gate-5 verdict
> (ratified 2026-06-23) let one positive stand — a narrow 5 dB noise-robustness
> benefit, then based on **n=3** seeds. A pre-registered **n=12 grid (§8.8,
> `results/p7_strengthen/`)** plus a matched-strength **non-physics** control was then
> run to stress-test it, and a **fourth independent-audit round** (2026-06-24, GPT-5 +
> a fresh-memory Opus 4.8;
> `audit_reports/INDEPENDENT_AUDIT_2026-06-24_{GPT5,CLAUDE}.md`) reproduced the result
> by execution (Opus cache-free from all 48 checkpoints). **Outcome: the last positive
> does NOT replicate at n=12.** §0 below is rewritten to the resulting **complete
> negative**; it supersedes the prior §0 and the PRE-AUDIT §1–§5 (removed). Do not
> cite as ratified until the owner re-ratifies.

## 0. DRAFT verdict (2026-06-24) — authoritative once ratified

### One-paragraph verdict
On a single synthetic journal-bearing dataset, **physics-informed learning provides
no advantage over strong data-driven baselines on any axis tested** — clean accuracy,
**noise robustness**, data-efficiency, severity-OOD, interpretability, and
calibration were each tested at the record level and each **failed**. The last
candidate positive (a 5 dB noise-robustness benefit that looked significant at n=3:
the band-energy w=1.0 model degraded 0.06 pt vs cross-entropy-only's 4.29 pt) was
stress-tested with a **pre-registered n=12 grid (§8.8)** that added a matched-strength
**random non-fault-band** control and tripled the seeds. It **did not replicate**: at
n=12 the correct-physics model degrades **3.47 pt** vs CE-only's **3.54 pt** —
statistically indistinguishable (seed-level **Wilcoxon p=0.79**), no arm clears the
pre-registered robustness bar, and the n=3 "win" is reproduced exactly by *this
grid's own* first three seeds, then erased by the other nine. The durable
contributions are the physics-grounded **dataset/generator (C1)**, the frozen
reproducible record-level **benchmark (C2)**, a **rigorous, complete NEGATIVE on
physics-informed learning (C3)** with a **methodological caution**, and **deployment
(C5)**. The paper must be framed as a **dataset + benchmark + negative result**; it
may **not** claim any physics benefit — not even "a spectral regularizer helped."

### Supportable now — claims the paper MAY make (each artifact-linked)
- **C1 — dataset/generator.** `data/generated/dataset_v2.h5`: 3,520 records, 11
  journal-bearing classes, exact class×severity balance, record-level
  leakage-checked splits (independently re-verified leakage-free by content hash, both
  fourth-round audits), SNR-20/10/5 variants; generator CI-locked (34 spectral tests).
  Cavitation only weakly expressed spectrally — qualify any cavitation claim.
  (`results/dataset_v2_validation/`, `tests/test_physics_signatures.py`.)
- **C2 — benchmark, as a CLASSIFICATION benchmark.** Pre-registered, 11 models,
  3 seeds; significance at the **528-record level**
  (`results/benchmark/summary_record_level.md`). Near-ceiling (RF 98.74, top deep
  cnn_lstm 99.43, CE-only pc_cnn 98.99); **no row shows a physics accuracy advantage**
  (best vanilla cnn_lstm ties CE-only pc_cnn, gap +0.00, McNemar p=1). Rows honestly
  relabeled: pc_cnn = CE-only/architecture (physics loss OFF), multitask = single-task,
  hybrid = rolling-element + constant metadata.
- **C3 — a rigorous, COMPLETE NEGATIVE on physics-informed learning.** Across every
  regime physics is theorized to help, the implemented physics-informed mechanisms do
  not beat data-driven baselines:
  - **Noise robustness — does NOT survive (§8.8, n=12, `results/p7_strengthen/`).**
    Record-level clean→5 dB degradation (528 records, soft-vote, 12 seeds): CE-only
    **3.54±3.85** (robust 4/12), correct-physics w=1.0 **3.47±3.42** (5/12), scramble
    **4.89±6.08** (7/12), random-band **2.15±3.34** (9/12) — all dominated by seed
    variance. Pre-registered seed-level **Wilcoxon vs CE-only**: correct **p=0.79**,
    scramble **p=0.75**, random **p=0.21** — all non-significant; **no arm** is robust
    on ≥10/12 seeds. The earlier n=3 positive (0.06 vs 4.29, within-seed McNemar
    14–0 p=1.2e-4) was a **seed artifact**: this grid's seeds {0,1,2} reproduce it
    (CE 4.29 / correct 0.06), and seeds 3–11 erase it. The random *non-fault* control
    is the **most** robust arm and correct physics the **weakest** of the three w=1.0
    arms — so the data do not support "physics" or even "a spectral regularizer."
  - **Clean accuracy** — no advantage (record-level near-ceiling; C2).
  - **Data-efficiency (§8.2)** — neutral; ahead non-overlapping at only 1 of 3 reduced
    fractions → fails the prereg rule. (`summary_record_level.json`.)
  - **Severity-OOD (§8.3)** — dir A tied at ceiling; dir B favors physics on the point
    estimate but McNemar p=0.39, gap CI [−2.27, +8.33] spans zero → direction-only.
  - **Interpretability (§8.6a)** — reverses (vanilla in-band 0.146 > physics 0.099,
    band-aware); IG gives no support for a physics-attention mechanism.
  - **Calibration (§8.6b)** — a wash (5 dB ECE direction flips between MC-dropout runs).
- **Methodological caution (a contribution in its own right) — two prongs.**
  (a) **A "significant" n=3 result that dissolved at n=12** — a same-architecture
  ablation with a within-seed McNemar of p=1.2e-4 became non-significant once the
  seed, not the window, was used as the inferential unit and the count rose to 12. A
  concrete warning about seed counts and estimands in PINN ablations. (b) **A physics
  loss that passed a green test suite while being silently broken** — non-differentiable
  (argmax) → wrong-bearing-type → tonal-only → flat-baseline, five defects in series.
  The rigor that caught both (record-level statistics, pre-registered scrambled *and*
  random-band controls, the inert-path hard-block `tests/test_physics_quarantine.py`)
  is the transferable lesson.
- **C5 — deployment.** ResNet18→ONNX FP32 ~13 ms/window CPU; INT8 4× smaller but
  10–15× slower (honest negative). (`results/deployment/appendix.md`.)

### What did NOT survive — may NOT be claimed
- **Noise robustness** — does not replicate at n=12 (§8.8 above). This was the last
  candidate positive; it is now retired.
- **Clean accuracy, data-efficiency, severity-OOD, interpretability, calibration** —
  each tested, none survives (detail under C3).
- **Any "a spectral regularizer helped" wording** — the n=12 grid kills even the
  narrow Phase-6 framing (random non-fault bands are the most robust arm; correct
  physics is the weakest w=1.0 arm; all n.s. vs CE-only).
- **§8.5 HybridPINN** — excluded (physics branch still rolling-element; not a
  journal-bearing physics test).
- Any **"physics-informed learning helps"** family claim; any **window-level** or
  **within-/representative-seed** significance as a headline; any real-bearing claim.

### Required wording limits (do not loosen)
- The study is a **complete negative**: "on this synthetic benchmark, the implemented
  physics-informed mechanisms do not outperform data-driven baselines on any tested
  axis; a promising n=3 noise-robustness result failed to replicate at n=12." Do
  **not** claim a physics benefit of any kind, and do **not** claim "a spectral
  regularizer helped."
- The **inferential unit is the record (528)** for accuracy and the **seed** for
  cross-run claims. **Never quote the within-seed McNemar (14–0, p=1.2e-4) as
  evidence** — it is the confounded estimand; the seed-level Wilcoxon (n.s.) governs.
- Report the **n=3→n=12 dissolution as the finding**, prominently, not as a footnote.
- Synthetic-only; no real-rig validation; near-ceiling clean task.

### Open before paper-ready (Phase 7 — submission tasks)
1. **Remove/stub the stale fabricated paper** — `config/docs/paper/main.tex` +
   `config/docs/reports/Final_Report_UNVALIDATED.pdf` (CWRU/98.1%/expert-validated/
   rolling-element — a different, non-existent study). Both fourth-round audits: it
   must not reach a referee. **Being done in Phase 7 (this docs pass).**
2. **Reconcile the verdict docs** — README, `results/README.md`, `PROJECT_STATE.md`,
   `results/phase5_bandenergy/findings_bandenergy.md` to the n=12 reality. (In progress.)
3. **Externalize reproducibility (audit M1)** — pin a **content hash** of
   `dataset_v2.h5`, archive the checkpoints (Zenodo); the chain command→commit→
   dataset-hash→checkpoint→result must be closeable by a cold referee.
4. **Provenance manifest** — one command/commit/dataset-hash/checkpoint/
   random-reference-hash per paper table.
5. **`ops_aware` field rename (audit M2)** — it records the *eval* flag; the *training*
   loss did use per-sample rpm (independently confirmed, 2026-06-24 Opus). Rename /
   add a `train_metadata_rpm_used` field so it cannot be misread.
6. **Write the manuscript from scratch** from this §0 — dataset + benchmark + complete
   negative + methodological caution; record-level tables only; synthetic-only.

### Status (Phase 7)
The §8.8 n=12 grid (`results/p7_strengthen/`, analysis
`scripts/p7_strengthen_record_level.py`, pre-registered PROTOCOL §8.8) ran on Colab
(48 runs, single commit `4a2063d`); record-level seed-level analysis committed
(`0797258`). The **fourth independent-audit round** (GPT-5 + fresh Opus, 2026-06-24)
reproduced every decisive number by execution — no critical findings — and both
auditors' verdict is: **trustworthy; publishable only as a synthetic dataset + frozen
benchmark + rigorous complete negative + methodological caution, not as any physics
win.** This memo is the DRAFT reflecting that; **owner re-ratification is the gate.**

---

> §1–§5 (the PRE-AUDIT Phase-5 draft) were **removed 2026-06-24**: they overclaimed an
> interpretability/calibration gain and framed C3 as a clean-data negative-with-a-
> surviving-noise-positive, all superseded by §0. Recoverable from git history; must
> not be cited.
