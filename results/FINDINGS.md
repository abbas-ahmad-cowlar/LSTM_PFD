# FINDINGS — Phase 5/6 synthesis (DRAFT for owner re-ratification, Gate 5)

> Honest synthesis of every Phase-5 experiment plus the Phase-6 physics
> remediation. Every number traces to a committed artifact under `results/`. This
> memo freezes the list of claims the paper may and may not make. **Synthetic-only
> study — no real-world validation, stated everywhere.**
>
> Status: **DRAFT — 2026-06-17, AWAITING OWNER RE-RATIFICATION (not ratified).**
> Reflects the corrected, **record-level**, **two-independent-audit-reviewed**
> state after the band-energy remediation
> (`audit_reports/INDEPENDENT_SCIENCE_AUDIT_2026-06-14.md` opened the blast radius;
> `...2026-06-16.md` reviewed the fix and **independently reproduced every
> record-level number**). **§0 below is the current authoritative verdict; it
> supersedes the 2026-06-14 interim correction it replaces AND the PRE-AUDIT §1–§5
> (retained for provenance only).** The owner's sign-off is the gate — do not cite
> as ratified.

## 0. DRAFT verdict (2026-06-17) — authoritative, supersedes §1–§5

### One-paragraph verdict
On a single synthetic journal-bearing dataset, a **correctly-wired PC-CNN
band-energy consistency loss at high weight (w=1.0)** delivers a **real,
record-level 5 dB noise-robustness improvement** over the **same architecture**
trained cross-entropy-only — the one physics benefit in the study that survives
record-level statistics and an independent reproduction. There is **no
clean-accuracy gain** (a small near-ceiling clean trade-off), and **no
statistically-supported data-efficiency, severity-OOD, interpretability, or
calibration advantage** — each was tested and did **not** survive. The durable
contributions are the physics-grounded **dataset/generator (C1)**, the frozen
reproducible **benchmark (C2)**, the **narrow noise-robustness result (C3)** plus
a methodological caution, and **deployment (C5)**. The result must be framed as
"the *implemented band-energy term* helped," not "physics-informed learning
helps" — see the wording limits below.

### Supportable now — claims the paper MAY make (each artifact-linked)
- **C1 — dataset/generator.** `data/generated/dataset_v2.h5`: 3,520 records, 11
  journal-bearing classes, exact class×severity balance, record-level
  leakage-checked splits, SNR-20/10/5 variants; generator CI-locked (34 spectral
  tests). Cavitation only weakly expressed spectrally — qualify any cavitation
  claim. (`results/dataset_v2_validation/`, `tests/test_physics_signatures.py`.)
- **C2 — benchmark, as a CLASSIFICATION benchmark.** Pre-registered, 11 models,
  3 seeds; significance at the **528-record level**
  (`results/benchmark/summary_record_level.md`). Near-ceiling (RF 98.74, top deep
  ~99, CE-only pc_cnn 98.99); **no row shows a physics accuracy advantage** (best
  vanilla cnn_lstm ties CE-only pc_cnn, McNemar p=1). Rows honestly relabeled:
  pc_cnn = CE-only/architecture, multitask = single-task, hybrid = rolling-element
  + constant metadata.
- **C3 — the one surviving physics positive: NOISE ROBUSTNESS (same-architecture
  ablation).** Band-energy loss vs the frozen healthy reference, per-sample rpm
  (`packages/core/models/pinn/physics_constrained_cnn.py`); record level
  (`results/phase5_bandenergy/summary_record_level.json`): pc_cnn **w=1.0**
  degrades **0.06 pt** clean→5 dB vs the identical-architecture CE-only model's
  **4.29 pt**; representative best-val-seed **McNemar 14–0, p=1.2e-4, gap +2.65
  pts [1.33, 4.17]**. Mechanism (independently verified, 2026-06-16 audit): the 14
  rescued records are noisy `lubrification` that CE-only mislabels
  `mixed_wear_lube`. Reverses the earlier contaminated "harmful at w=1.0 (83.1)".
  Secondary, weaker: vs best vanilla resnet18 it is significant but
  **small/fragile** (6–0, p=0.031, +1.14 [0.38, 2.08]; flips to p=0.0625 vs
  resnet's best-noise seed) — report as secondary/near-ceiling.
- **C5 — deployment.** ResNet18→ONNX FP32 ~13 ms/window CPU; INT8 4× smaller but
  10–15× slower (honest negative). (`results/deployment/appendix.md`.)
- **Methodological caution (a contribution in its own right).** A naive
  frequency-consistency physics loss was silently non-differentiable (argmax);
  even once differentiable it was wrong-bearing-type → tonal-only → flat-baseline
  — five defects that all passed a green test suite. The corrected
  band-energy-vs-healthy-reference loss is the fix; the inert generic path is now
  hard-blocked (`tests/test_physics_quarantine.py`).

### What did NOT survive — may NOT be claimed
- **Clean accuracy**: no physics advantage (record-level near-ceiling).
- **Data-efficiency (§8.2)**: neutral — ahead non-overlapping at only 1 of 3
  reduced fractions → fails the prereg rule; the old "harmful at 10%" is gone (now
  tied). (`summary_record_level.json`.)
- **Severity-OOD (§8.3)**: dir A tied at ceiling; dir B favors physics on the
  point estimate but is **not significant** (McNemar p=0.39; representative gap CI
  [−2.27, +8.33] spans zero). Direction-only. (`summary_record_level.json`.)
- **Interpretability §8.6a**: **reverses** under the corrected DB + w=1.0
  checkpoint — vanilla 1.042 > physics 0.856. The metric is **tonal-only and
  excludes lubrification/cavitation** (the broadband classes, incl. the noise
  mechanism), so this is "not more *tonally* aligned," not proof of less physics
  attention — but as it stands it is **not** a positive.
  (`results/xai_alignment/alignment.json`, `findings_8_6.md`.)
- **Calibration §8.6b**: a wash — pc_cnn better clean ECE (0.018 vs 0.024), worse
  at 5 dB (0.026 vs 0.022); both <0.03; single-seed.
  (`results/uncertainty/calibration.json`.)
- **§8.5 HybridPINN**: excluded — physics branch still rolling-element (SKF-6205);
  not a journal-bearing physics test.
- Any **"physics-informed learning helps"** family claim; any window-level
  significance; any real-bearing claim.

### Required wording limits (do not loosen)
- Say "**the implemented PC-CNN band-energy consistency loss at w=1.0** improved
  5 dB robustness in a **same-architecture ablation** on synthetic data" — NOT
  "physics-informed learning improves robustness."
- The loss is a **band-consistency regularizer**, not a mechanistic diagnostic
  solver (`sain` has no bands → it cannot penalize a healthy prediction).
- State: n=3 seeds; near-ceiling (small discordant counts); the vs-resnet edge is
  secondary/seed-sensitive; the small clean trade-off (w=1.0 clean 98.61 vs
  98.99–99.68 for lower weights); synthetic-only.
- The benefit is attributable to **this implemented term**; it is **not isolated**
  from generic high-weight regularization (no controls run yet — see below).

### Open before paper-ready (owner decisions)
1. **F9 controls (GPU)** — entropy/logit reg at matched strength, random-band,
   permuted/wrong-class healthy-reference — to isolate physics from
   "regularization at high weight." Without them the claim stays "the band-energy
   term helped," not "physics helped."
2. **F13 — more than n=3 seeds (GPU)** for stronger seed-level inference.
3. **Band-aware §8.6a (laptop)** — recompute alignment with `get_expected_bands`
   (incl. the 1–6 Hz lube / 1.4–2.6 kHz cavitation absolute bands) + a documented
   control, to fairly test broadband attention (the tonal-only metric cannot).
4. **§8.5 HybridPINN** — rebuild on journal-bearing features or leave excluded.
5. **Provenance manifest** — one command/commit/dataset-hash/artifact per paper
   table (audit Rec 8).

### Remediation status (Phase 6)
Steps 1–4 done (docs reconciled; record-level benchmark stats; quarantine/relabel;
band-energy loss + frozen healthy reference + CI). Step 5: reruns done; **5a
record-level confirmation done** (`summary_record_level.json`, `0696790`); **5b
audit fixes done** (F6 estimator consistency, F10 inert-path hard-block,
`a2e09d9`); **5c XAI/calibration recompute done** (`2b534bf`, this memo). Two
independent external audits reviewed the work; the 2026-06-16 auditor reproduced
every record-level number. **This memo is the DRAFT; owner re-ratification is the
remaining gate.**

---

> ⚠️ The sections below (§1–§5) are the PRE-AUDIT draft, retained for provenance.
> They overclaim per §0 and must not be cited until remediation completes.

## 1. One-paragraph verdict (SUPERSEDED — see §0)

On clean synthetic journal-bearing data, **physics-informed learning does not
beat purely data-driven models on accuracy** — tested rigorously across noise,
data-efficiency, severity-OOD, a physics-weight ablation, and operating-
condition metadata, including the low-data regime where physics priors are
*supposed* to help most (there it actively hurts). The data-driven CNN already
captures the discriminative spectral structure the generator embeds. Physics
training does, however, yield a **modest interpretability and calibration gain**
(more physics-aligned attributions, lower ECE) at no accuracy cost. The durable
contributions are the **physics-grounded dataset/generator (C1)**, the **frozen
reproducible benchmark (C2)**, this **rigorous negative (C3)** plus a
methodological caution, and the **interpretability result (C4)**.

## 2. Results by contribution (all artifact-linked)

### C1 — Dataset & generator (`results/dataset_v2_validation/`)
`dataset_v2.h5`: 3,520 records, 320/class, 11 journal-bearing fault classes,
exact 80/class/severity stratification, record-level leakage-checked splits,
SNR-20/10/5 test variants. Generator is physics-normative (`docs/PHYSICS.md`),
enforced by a 34-test spectral CI battery (`tests/test_physics_signatures.py`).

### C2 — Frozen benchmark (`results/benchmark/summary.md`)
Pre-registered protocol (Adam 1e-3, batch 64, ≤60 ep, patience 10, 3 seeds,
test-touched-once). Headline (test acc %, 2,640 windows):
voting_ensemble **96.48** · resnet18 96.14±0.28 · cnn_lstm 96.12±0.16 ·
physics_constrained_cnn 95.98±0.36 (physics-OFF, see §8.0) · RandomForest
94.61±0.05 · cnn1d 91.94±2.84 · multitask_pinn 90.28 · hybrid_pinn 90.04 ·
patchtst 89.85 · attention_cnn 89.37 (1 seed collapsed).

### C3 — Physics advantage: a complete, decisive NEGATIVE
- **§8.1 Noise** (`results/noise_robustness/summary.md`): physics-family mean
  degradation clean→5 dB 8.51 vs vanilla 15.54 — *but* outlier-driven
  (attention_cnn Δ50.6 inflates the vanilla mean; excluding it, vanilla ≈6.8
  beats physics). Most noise-robust single model is **vanilla resnet18**
  (Δ1.70, still 94.4% at 5 dB). Verdict: no robust physics edge.
- **§8.2 Data-efficiency** (`results/phase5_dataeff_fixed/before_after_8_2.md`):
  fixed (differentiable) physics loss at 10% data → **91.11±3.29, worse** than
  vanilla 93.60 and than its own inert version 93.55, one seed crashing to 87.5;
  25/50/100% within noise. Prereg rule: physics wins **0/3** → REJECTED. Low
  data — physics's best-case regime — is where it does the most harm.
- **§8.3 Severity-OOD** (`results/phase5/severity_ood/`): train→test severe,
  pc_cnn 96.87 vs resnet18 97.37 (vanilla ahead); train→test incipient, pc_cnn
  79.80 vs 73.43 (pc_cnn backbone +6.4). Split, and an *architecture* effect
  (physics was inert in these runs) — not a physics-as-such result.
- **§8.4 Physics-weight ablation** (`results/phase5_fixed/before_after_8_4.md`):
  before the fix the loss was inert (all w byte-identical); after the fix, clean
  flat ~96, 5 dB neutral at low w (92.5 vs 91.0, within ±5–6 noise) and
  **harmful at high w** (w=1.0 → 83.1, one seed 71.7). More physics weight →
  worse, more unstable. REJECTED.
- **§8.5 True metadata** (`results/phase5/true_metadata/`): hybrid_pinn with true
  per-record rpm/load/viscosity → 89.76 vs Phase-4 blind 90.04. No gain. NULL.

### Methodological finding (a paper contribution in its own right)
`PhysicsConstrainedCNN.compute_physics_loss` was **silently non-differentiable**
(routed through `argmax`; `requires_grad=False`), so the physics weight had zero
training effect — undetectable without execution checks (proven by byte-identical
w-sweep runs). Corrected to a softmax-weighted differentiable penalty
(`experiments/PHYSICS_LOSS_DIAGNOSIS.md`, PROTOCOL §7 amendment); even corrected,
no accuracy benefit. A common, easily-missed PINN pitfall.

### C4 — Physics-consistent XAI & calibration (modest POSITIVE)
(`results/xai_alignment/findings_8_6.md`; resnet18 and pc_cnn share the identical
3.85M-param backbone, isolating the physics-loss effect.)
- **8.6a**: physics model's IG attributions are more concentrated in PHYSICS.md
  characteristic-frequency bands than the same-backbone vanilla model
  (specificity ratio 0.849 vs 0.716). Physics training shifts *what the model
  attends to* toward physical frequencies — even without an accuracy gain.
- **8.6b**: physics model better calibrated (ECE clean 0.022 vs 0.028; 5 dB
  0.018 vs 0.027); both reach 100% accuracy at 50% coverage (reject option).

### C5 — Deployment (`results/deployment/appendix.md`)
ResNet18→ONNX FP32 **13 ms/window CPU** (parity 1.5e-4); INT8 4× smaller but
10–15× slower — honest negative (dynamic quant doesn't help conv nets).

## 3. Claims the paper MAY make (frozen)
1. A physics-grounded synthetic journal-bearing dataset + CI-enforced generator. [C1]
2. A frozen, pre-registered, reproducible 11-model benchmark with honest stats. [C2]
3. A rigorous, decisive negative: physics-informed learning does not improve
   accuracy over data-driven baselines on clean synthetic journal-bearing data,
   across all five tested regimes — including the low-data steelman. [C3]
4. A methodological caution: a naive frequency-consistency physics loss can be
   silently non-differentiable; corrected, it still does not help accuracy.
5. A modest interpretability/calibration benefit from physics training, same
   backbone (more physics-aligned attributions; lower ECE). [C4]
6. Deployment characterization (ONNX latency; INT8 negative). [C5]

## 4. Claims the paper may NOT make
- That physics improves classification accuracy (false — rejected in every regime).
- Any real-world / real-bearing performance claim (study is synthetic-only).
- A strong/statistically-robust C4 claim *as-is* — see open items below.

## 4b. Validity of the negative — physics self-consistency check (2026-06-14)

Added after ratification (does not change the §3–4 claims — it *bounds* them).
Run `scripts/verify_physics_consistency.py`: it compares the frequencies the
physics loss expects (`FaultSignatureDatabase`, same source the loss uses)
against the actual mean spectra of `dataset_v2` signals per class.

- **Tonal faults** (imbalance 1X, misalignment 2X/3X, clearance, oil whirl
  sub-sync ~0.45X): the expected characteristic frequencies ARE present as
  dominant peaks in the data. The generator physics is **self-consistent** here
  — the signatures a physics model could exploit genuinely exist.
- **Broadband/impulsive faults** (cavitation, wear, lubrication): energy is
  broadband/impulsive *by design* (e.g. cavitation kurtosis ≈ 6.8), so the
  signature DB's **narrow expected-frequency list is a poor encoding** of these
  signatures — the physics loss is looking for narrow peaks that physically
  aren't narrow.
- **Mixed faults** (3 classes): `get_expected_frequencies` has **no entry** and
  the lookup fails → the physics loss contributed **zero** constraint for them.

**Implication (honest bound on C3):** the generator is sound, but our physics-
*loss* encoding is partial — good for tonal faults, crude for broadband, absent
for mixed. So the loss-based negative is partly attributable to an incomplete
encoding, NOT proof that *no* physics prior could ever help. The paper must
state C3 as *"the physics-informed mechanisms we implemented, as formulated, do
not improve accuracy"* — not a universal claim. Why the verdict is still robust:
(a) the data-driven models already reach ~96% (little headroom); (b) physics
actively HURT at 10% data (§8.2); (c) a second, independent mechanism
(hybrid_pinn metadata, §8.5) was also null. A better physics encoding
(broadband + mixed) is honest future work.

## 5. Open items before paper-ready (carry into Phase 6/7)
- Improve/disclose the physics-loss frequency encoding (broadband + mixed-fault
  signatures) — see §4b; either fix-and-retest or report as a stated limitation.
- Seed-average the §8.6 calibration ECE (currently single checkpoint each).
- §8.6a: control-band sensitivity check (absolute fractions depend on it).
- Generate figures from the JSONs: data-efficiency curve, reliability diagram,
  reject curve, per-class alignment, noise-degradation curves.
- §8.4 McNemar (w=0 vs best-w at 5 dB) via a quick checkpoint re-eval to capture
  paired per-window predictions (not stored in metrics.json).

## 6. Honest framing recommendation
Strongest, most defensible framing is a **dataset + benchmark paper** that uses
the benchmark to deliver a rigorous negative on physics-informed learning plus
an interpretability finding — turning "synthetic-only" from a weakness into the
contribution (a released tool). Venue tier: IEEE Access / Sensors / Measurement
special issue / workshop / arXiv preprint — not a top mechanical-systems venue
(synthetic-only ceiling).
