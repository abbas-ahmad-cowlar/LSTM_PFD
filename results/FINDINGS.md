# FINDINGS — Phase 5 synthesis (for owner ratification, Gate 5)

> Honest synthesis of every Phase-5 experiment. Every number traces to a
> committed artifact under `results/`. This memo freezes the list of claims the
> paper may and may not make. **Synthetic-only study — no real-world validation,
> stated everywhere.**
>
> Status: **SUPERSEDED / UNDER REVISION — 2026-06-14, after independent external
> audit** (`audit_reports/INDEPENDENT_SCIENCE_AUDIT_2026-06-14.md`). The prior
> owner ratification is **on hold**. The audit found the physics-model evidence
> more broadly contaminated than this memo assumed. **§1–§4 below overclaimed and
> are SUSPENDED pending remediation — read §0 first; it overrides them.**

## 0. Post-audit correction (authoritative — overrides §1–§4)

**Supportable now (tightened wording — do not loosen):**
- Dataset v2 is sound **as a synthetic, internally consistent benchmark**:
  balanced, group-split by record, leakage-checked, intended signatures present
  (cavitation only weakly so — qualify any cavitation claim).
- The benchmark accuracies are plausible **as classification results**; the rows
  are now **relabeled** (Step 3, done): `physics_constrained_cnn` = **CE-only /
  architecture** (physics loss OFF), `multitask_pinn` = **single-task**,
  `hybrid_pinn` = **rolling-element branch + constant metadata** — none is a
  physics result (`results/benchmark/summary.md` banner). Significance has been
  **recomputed at the
  528-record level** (cluster-bootstrap, Step 2 —
  `results/benchmark/summary_record_level.md`), superseding the window-level
  p-values (which used 2,640 correlated windows). Direction is unchanged:
  soft-voting the 5 windows/record pushes the benchmark **near-ceiling**
  (RandomForest 98.74%, top deep ~99%, CE-only pc_cnn 98.99%) and **no row shows
  a physics advantage** — the best vanilla (cnn_lstm) and the CE-only pc_cnn tie
  to the record (gap +0.00 pts, McNemar p=1). High-variance rows (cnn1d,
  attention_cnn) carry one collapsed seed (±4.3 std), now shown with consistent CIs.
- The **stored artifacts show no physics accuracy advantage.** This is **NOT** a
  definitive negative about correctly-implemented journal-bearing
  physics-informed learning — **that experiment has not been run.** The §8.4/§8.2
  "fixed" runs used an **incomplete tonal-only** loss (ratified band-energy not
  yet implemented); §8.5 HybridPINN uses **rolling-element** (SKF 6205
  BPFO/BPFI/BSF/FTF) features — wrong for journal-bearing data; §8.6/C4 (XAI
  alignment + calibration) rests on the old broken-DB bands and incomplete-loss
  checkpoints. All are **contaminated / not yet valid**.

**May NOT claim (yet):** any physics benefit — accuracy, noise robustness,
data-efficiency, severity-OOD, interpretability, or calibration; a "rigorous"
or "decisive" negative; family-level noise robustness; any window-level
significance.

**Remediation sequence (endorsed by the external auditor):** (1) reconcile docs
to this corrected blast radius [done — incl. README]; (2) recompute all
statistics at **record level** [**DONE 2026-06-14** —
`results/benchmark/summary_record_level.md`,
`scripts/aggregate_benchmark_record_level.py`]; (3) quarantine/relabel invalid
rows [**DONE 2026-06-14** — summary relabeled + banner; quarantine docstrings on
inert `PhysicalConstraintLoss`/`PINNTrainer`/`PINNEvaluator`, rolling-element
HybridPINN branch + `BearingDynamics`; stale `pinn_ablation.py` blocked;
`tests/test_physics_quarantine.py` pins it]; (4) implement + gradient-test the
ratified **band-energy** loss [**DONE 2026-06-14** — `compute_physics_loss`
rewritten to band-energy consistency judged vs the **frozen healthy-class
reference** (owner-corrected from a flat baseline; per-sample rpm;
tonal+broadband+mixed; differentiable); frozen `healthy_reference.json` + aligned
CI test; before/after audit shows the flat baseline let healthy masquerade as
lubrification/imbalance; PROTOCOL §8.0-quinquies]; (5) rerun the physics-forward experiments with the
band-energy loss + record-level stats [**NEXT, GPU/Colab** —
`experiments/COLAB_PHASE5_RERUN_RUNBOOK.md`]. Then this memo is rewritten and
**re-ratified**.

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
