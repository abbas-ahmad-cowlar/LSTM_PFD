# Independent Science Audit — LSTM-PFD (journal-bearing fault diagnosis)

**Auditor:** Independent technical reviewer (Claude Opus 4.8), read + execute access.
**Date:** 2026-06-22
**Repo / branch:** `C:\Users\COWLAR\projects\lstm-pfd` @ `p6/docs` (clean), tip `b9d748c`.
**Scope:** Independent re-derivation of the project's science from code, data, and
stored artifacts. I did **not** read any prior audit report (they were removed on
purpose); this is my own view, formed by execution.

> Owner's literal question: *"Is this real and defensible, or are we polishing a
> null result into a publication so we can publish any scrap?"*
>
> **Short answer:** It is **real, reproducible, and unusually honest** — but the
> "positive" is thin. This is, fundamentally, a **rigorous, well-controlled
> negative** on physics-informed learning, plus **one narrow, real, but fragile
> noise-robustness effect that the maintainers' own pre-registered control correctly
> demotes from "physics" to "a generic high-weight spectral regularizer."** You are
> **not** polishing a null into a false positive — if anything you have repeatedly
> demoted your own positives. The science is sound and publishable as a
> dataset+benchmark+negative paper at a mid-tier venue, with the wording limits you
> already wrote. The one thing the remediation **missed**: a stale pre-remediation
> paper draft (`config/docs/paper/main.tex`) still in the repo that fabricates a
> different study (CWRU data, 98.1%, expert-validated XAI, rolling-element physics).
> Delete or rewrite it before anyone mistakes it for the paper.

---

## 1. Executive summary

I reproduced the entire surviving-claim pipeline end-to-end and it holds:

- **Test suite:** `263 passed, 6 deselected` — exactly as claimed. The 6 deselected
  are `-m "not dashboard"` (pytest.ini), legitimately out of scope.
- **Dataset is clean.** 3,520 records (2464/528/528), exact class×severity balance,
  **zero cross-split leakage** (record hashes *and* raw-signal bytes disjoint across
  train/val/test). Per-record rpm genuinely varies (3240–3959 Hz), so per-sample rpm
  in the loss actually matters — and it flowed.
- **The physics loss is genuinely fixed.** It is differentiable, the gradient
  reaches model parameters (I measured grad-norm 1.63 to params), it was **active**
  during the §8.4/F9 reruns with per-sample rpm, and the inert legacy path is
  hard-blocked by tests. This is a real correction of the old `argmax` non-grad bug.
- **The frozen healthy reference is sound.** Computed from **train-only** healthy
  windows (n=1120 = 224×5); I regenerated it and it matches the committed JSON to
  **0.0e+00**.
- **The headline numbers are real, not asserted.** I (a) recomputed every §8.4 / F9
  number directly from the cached predictions and got an exact match to the committed
  JSONs, and (b) re-evaluated four key checkpoints **from scratch** (loading
  `best_model.pth`, fresh forward pass) — the cached arrays are **byte-identical**
  (max prob diff 0.0e+00). The cache is not fabricated.
- **The same-architecture claim is literally true.** `physics_constrained_cnn` and
  `resnet18` share the identical ResNet1D backbone, **3,853,195 params each**,
  identical state-dict keys. The w=0 (CE-only) and w=1.0 arms used the **same frozen
  budget** (Adam 1e-3, batch 64, ≤60 ep, patience 10).
- **The representative-seed choice is honest.** seed2 is the genuine **best-val**
  seed for *every* arm, and for CE-only it is also its *most noise-robust* seed — so
  the headline McNemar is run against CE-only's **strongest** seed (conservative).
- **The §8.7 control was genuinely pre-registered** (protocol + decision rule
  committed 2026-06-17; scramble actually trained 2026-06-21, four days later) and
  the derangement is valid and does change the loss.

**The one real positive, stated precisely:** at 5 dB SNR, record level (528
records), the band-energy loss at **w=1.0** loses **0.06 pt** clean→5 dB vs the
identical-architecture CE-only model's **4.29 pt**; at the best-val seed the paired
McNemar is **14–0, p=1.2e-4**, the gap **+2.65 pt [1.33, 4.17]**. I reproduced all of
this. The mechanism is exactly as claimed: **14 noisy `lubrification` records that
CE-only mislabels `mixed_wear_lube` are rescued** (I confirmed: 14/14, all that
class pair). **But** the pre-registered scrambled-reference control reproduces the
robustness with wrong per-class targets (scramble vs correct, best-val seed: 0–1,
p=1.0), so the effect is **not specific to correct journal-bearing physics**.

**What this means:** As a *physics-informed-learning* result, the project is a
**negative** — correctly and honestly so. The surviving "positive" is a
spectral-regularization noise effect that is **(i)** narrow (one class pair),
**(ii)** near-ceiling, **(iii)** n=3-fragile, and **(iv)** shown by the control not
to be physics. With the maintainers' existing wording limits it is defensible; with
any looser wording it is not.

**Severity ledger (details below):**

| # | Finding | Severity |
|---|---|---|
| F1 | Stale paper draft `main.tex` + `Final_Report_UNVALIDATED.pdf` fabricate a different study (CWRU, 98.1%, expert-validated XAI, rolling-element BPFO/BPFI physics) — directly contradict the verified verdict; remediation reconciled README/PROJECT_STATE but missed these | **Major** |
| F2 | Headline rests on n=3 seeds with very large noise-regime variance (CE std 3.08, scramble std 5.31) and a single confusable class pair; cross-seed inference is under-powered | **Major (inherent limitation)** |
| F3 | FINDINGS §0 says the scramble "reproduces the robustness," but the seed-mean degradation (2.84) is **intermediate** and actually closer to CE-only (4.29) than to correct physics (0.06); the "reproduces" reading holds only at the representative seed | **Minor** |
| F4 | The "correct physics buys cross-seed stability" sub-claim is a 3/3-vs-2/3 seed difference at n=3 — not resolvable; slightly over-stated | **Minor** |
| F5 | §8.7 uses one permutation of *real* bands; it cleanly refutes "correct mapping is necessary" but does not, by itself, prove "any generic regularizer of this strength would do it" (no entropy/random-band/label-smoothing control at matched strength). Conservative direction, so it does not inflate claims | **Minor** |
| F6 | w=0 arm is borrowed from the Phase-4 benchmark (different commit/run pipeline) rather than retrained inside the w-sweep; same arch + same budget, so fair, but worth a provenance line | **Minor** |
| F7 | `dataset_card.yaml` mislabels `training_windows: 17600` (that is the *total* across all splits; train is 12320). IG steps: paper says 50, actual run 32 | **Minor / cosmetic** |

No **critical** findings. Nothing I could find rises to fabrication, leakage, or a
broken statistical estimand in the *current* surviving analysis.

---

## 2. What I actually ran (reproducibility log)

All commands under `$env:PYTHONIOENCODING='utf-8'`, venv at `.\venv`.

1. **Suite:** `pytest -q` → `263 passed, 6 deselected, 49 warnings in 77.66s`.
2. **Leakage:** loaded `dataset_v2.h5`; compared `record_hashes` and raw signal
   bytes across splits → all unique, all pairwise overlaps **0**.
3. **Healthy reference:** recomputed from train `sain` windows → matches
   `healthy_reference.json` to 0.0e+00, n=1120.
4. **Physics loss:** instantiated `PhysicsConstrainedCNN`; confirmed grad reaches
   params (norm 1.63), and that setting `reference_permutation` changes the loss
   (0.2077 → 0.1484).
5. **§8.4 / F9 recompute from cache:** soft-voted the cached window probs per record,
   recomputed accuracy / degradation / exact McNemar / cluster-bootstrap CI — exact
   match to `summary_record_level.json` and `f9_scramble_record_level.json`.
6. **Cache faithfulness:** re-evaluated `pinn_ablation/w1.0/seed2`,
   `benchmark/.../physics_constrained_cnn/seed2`, `benchmark/.../resnet18/seed2`, and
   `pinn_ablation_scramble/w1.0/seed0` from their checkpoints → cached arrays
   byte-identical (max |Δprob| = 0.0e+00); window-acc matches `metrics.json`.
7. **Benchmark record-level:** recomputed from `record_level/_cache` → cnn_lstm
   99.43, resnet18 99.18, RF 98.74, pc_cnn 98.99 — matches `summary_record_level.md`.
8. **Provenance:** checked `git_sha`, budgets, best-val per seed, pre-registration
   commit dates vs scramble run timestamps.

---

## 3. Per-area findings

### Area 1 — Dataset & frozen healthy reference — **SOUND**

- `data/generated/dataset_v2.h5`: groups `train/val/test/test_snr5/test_snr10/
  test_snr20`. 11 classes, 320 records/class, 80/class/severity, splits
  2464/528/528. Exact balance verified per split.
- **Leakage:** `record_hashes` are unique within and across splits; raw signal bytes
  are also disjoint across splits. Windowing is record-grouped by construction
  (`data/dataset.py:609` `WindowedView`; window→record map is `idx // wpr`), so the 5
  windows of a record cannot cross a split. **No leakage.**
- **Frozen reference** (`scripts/compute_healthy_reference.py`,
  `packages/core/models/physics/healthy_reference.json`): computed from `train`
  `sain` only, per-record rpm-matched, full-window rfft (1 Hz res to resolve the 1–6 Hz
  lube band). Provenance block records git sha, split, n=1120, fs/window. I
  regenerated it: **identical to 0.0e+00**. No val/test leakage into the loss
  baseline. The reference is what makes "signature present = energy *above healthy*"
  meaningful (e.g. the 1–6 Hz lube band carries 8.3% of healthy energy, the 1X band
  1.25%) — a real improvement over a flat baseline.
- **Caveats the maintainers already disclose and I confirm:** the synthetic task is
  **near-ceiling** (record-level 98–99.4% for the strong models), cavitation is
  weakly expressed, and the loss's band encoding is partial (good for tonal, crude
  for broadband, absent where a class's bands don't fire). These are honest bounds.

### Area 2 — Physics loss as implemented — **CORRECT AND ACTIVE**

`packages/core/models/pinn/physics_constrained_cnn.py:138` `compute_physics_loss`:

- **Differentiable & reaches params.** The per-class penalty `pen[B,C]` is built
  under `torch.no_grad()` (a function of the input spectrum, per-sample rpm, and the
  frozen reference — correctly constant w.r.t. params); the loss
  `(probs*pen).sum(1).mean()` carries gradient through `probs = softmax(logits)`.
  Measured grad-norm to params = 1.63 (nonzero). This is the genuine fix of the old
  inert `argmax` formulation.
- **Active during the reruns with per-sample rpm.** `run_phase5_gpu.py` trains the
  §8.4 and F9 arms with `std_loaders(..., ops=True)`, so each batch carries `rpm`
  → `unpack_batch` → `compute_physics_loss(..., metadata)`; tonal bands are placed
  at the record's own shaft frequency. Given rpm varies ±6% around 3600, this is not
  cosmetic. (The `ops_aware: False` tag in metrics.json refers to *eval*, not
  training — training metadata still flowed.)
- **Journal-bearing bands are correct.** `fault_signatures.py` encodes tonal
  harmonics of Ω=rpm/60 (1X/2X/3X, sub-sync 0.45X) plus absolute Hz bands (1–6 Hz
  stick-slip, 1.4–2.6 kHz cavitation), per `docs/PHYSICS.md §4`, with all 3 mixed
  classes present. The old rolling-element (BPFO/BPFI) DB is gone from the loss path.
- **Quarantine is real.** `tests/test_physics_quarantine.py` asserts the inert
  generic losses and `PINNTrainer(lambda_physics>0)` **raise** "quarantined and
  hard-blocked," and the stale `scripts/research/pinn_ablation.py` refuses to run.
  All four tests pass.

### Area 3 — The surviving noise-robustness claim — **REAL, REPRODUCIBLE, NARROW, FRAGILE**

My recompute from cache (identical to `summary_record_level.json`):

| w | clean (rec) | 5 dB (rec) | degr | per-seed 5 dB |
|---|---|---|---|---|
| 0 (CE-only) | 98.99 ± 0.45 | 94.70 ± 3.08 | 4.29 | 96.78 / 90.34 / 96.97 |
| 0.1 | 99.12 ± 0.09 | 94.13 ± 2.96 | 4.99 | 91.67 / 98.30 / 92.42 |
| 0.3 | 99.68 ± 0.09 | 93.75 ± 3.95 | 5.93 | 89.02 / 93.56 / 98.67 |
| **1.0** | 98.61 ± 0.62 | **98.55 ± 0.76** | **0.06** | 97.92 / 98.11 / 99.62 |
| resnet18 | (99.18) | 97.66 ± 1.30 | 1.52 | 95.83 / 98.67 / 98.48 |

- **w1.0 vs CE-only @5 dB (best-val seed2):** McNemar **14–0, p=1.221e-4**, gap
  **+2.65 [1.33, 4.17]**. Reproduced exactly.
- **w1.0 vs resnet18 @5 dB (seed2):** **6–0, p=0.0312**, gap +1.14 [0.38, 2.08]; vs
  resnet's *best-noise* seed it is **5–0, p=0.0625** (not significant). Reproduced.
- **Mechanism:** at seed2, w1.0 rescues exactly **14 records over CE-only, every one
  a `lubrification` record CE-only calls `mixed_wear_lube`.** Confirmed 14/14.

**Why it is real:** all three correct-physics seeds are robust (97.9/98.1/99.6, std
0.76); the comparison is same-arch, same budget, against CE-only's strongest seed; on
identical frozen noisy test signals. It is not a degradation-framing trick — the
*absolute* 5 dB accuracy is higher and the McNemar is on absolute predictions.

**Why it is narrow/fragile (F2, Major-but-acknowledged):**
- It is essentially one confusable class pair near the noise floor.
- It lives entirely in the top ~1–4% accuracy headroom of a near-ceiling synthetic
  task. Soft-voting over 5 windows inflates accuracy (e.g. w1.0 seed2 window-acc
  95.53 → record-acc 99.62), compressing all margins toward the ceiling.
- Cross-seed inference is n=3 with huge noise-regime variance. The within-seed
  McNemar is strong **conditional on a good seed**; the cross-seed story is
  "3/3 robust vs ~2/3 robust." The maintainers flag every one of these.

**Verdict:** the result is correctly computed and the same-architecture comparison is
fair. It is a genuine, if minor and fragile, effect.

### Area 4 — The §8.7 scrambled-reference control — **SOUND, PRE-REGISTERED, CORRECTLY CONSERVATIVE (with two nuances)**

- **Design is fair.** `reference_permutation = [0,10,5,9,6,2,8,4,7,1,3]` is a valid
  derangement (class 0 fixed; no fault keeps its own bands; I verified). It keeps the
  loss strength/structure and only swaps the per-class targets; I confirmed it changes
  the loss value. Same budget, same arch, w=1.0.
- **Pre-registered for real.** PROTOCOL §8.7 + the decision rule landed `2026-06-17`;
  the scramble checkpoints trained `2026-06-21 20:40–21:07`. Not post-hoc.
- **Result (reproduced):** scramble clean 98.42, 5 dB 95.58 ± **5.31** (per-seed
  88.07/99.24/99.43), degr 2.84. Best-val seed2: scramble vs CE-only **14–1,
  p=9.8e-4**; scramble vs correct **0–1, p=1.0**.
- **Interpretation is the right call.** Scrambling the per-class physics did not
  destroy the robustness → the **correct** journal-bearing mapping is **not
  necessary** → the word "physics" cannot be claimed. This is the conservative
  reading the pre-registered rule mandates for an intermediate result. Good.

**Nuance F3 (Minor):** the *seed-mean degradation* 2.84 is genuinely **intermediate**
and is in fact closer to CE-only (4.29) than to correct physics (0.06). The "scramble
reproduces the robustness" statement (FINDINGS §0) is only true at the
**representative seed**; on the seed-mean it does not fully reproduce. The
`findings_bandenergy.md` is more careful ("2 of 3 seeds robust," "LARGELY GENERIC").

**Nuance F4 (Minor):** the compensating claim — "correct physics only buys cross-seed
stability" — is a **3/3-robust vs 2/3-robust** difference at n=3; you cannot
distinguish "correct physics makes it reliable" from "scramble got unlucky on one
seed." Both directions are under-powered.

**Nuance F5 (Minor):** the control swaps *real* fault bands. It decisively refutes
"the correct class→band assignment matters," but it does not by itself establish that
*any* high-weight regularizer of similar magnitude (entropy penalty, random
non-physical bands, label smoothing) would reproduce the effect. The
`findings_8_6.md` explicitly lists those as the remaining controls. Importantly, all
of this points the **conservative** way (refusing to claim physics), so it does not
inflate the contribution. **Net: no matter how §8.7 is read — "generic" or "correct
physics adds a bit of stability" — the `physics-informed learning helps` claim is
unsupported.** The maintainers reached the correct conclusion.

### Area 5 — The non-surviving results — **CORRECTLY SET ASIDE**

- **Data-efficiency (§8.2):** record-level, pc_cnn ahead non-overlapping at only 1 of
  3 reduced fractions (25%: +1.26; 10%/50% tied) → fails the prereg "≥2 fractions"
  rule. Correctly called neutral. The old "harmful at 10%" is gone (now tied). Fine.
- **Severity-OOD (§8.3):** dir A tied at 100% ceiling; dir B favors physics on the
  point estimate (82.58 vs 76.01, repgap +3.03) but **McNemar p=0.39** and the gap CI
  **[−2.27, +8.33]** spans zero (only ~132 incipient records). Correctly reported as
  direction-only, **not buried** — the suggestive point estimate and low variance are
  stated. Honest.
- **Interpretability (§8.6a):** I checked `alignment.json` against `findings_8_6.md`.
  Tonal specificity **reverses** (vanilla 1.042 > physics 0.856); band-aware also
  reverses (vanilla 0.146 > physics 0.099); on `lubrification` — the noise-rescue
  class — both models put ≈0 attribution in the lube band (0.0072 vs 0.0024). The C4
  interpretability positive is **correctly retracted**; IG gives no support for a
  "physics-attention" mechanism. This is the opposite of polishing.
- **Calibration (§8.6b):** documented as a run-to-run MC-dropout wash (5 dB direction
  flips between runs). Correctly downgraded to a non-result.

This whole section is the strongest evidence *against* the "polishing a null" worry:
the maintainers retracted two of their own former positives (C4 interpretability +
calibration) on re-examination.

### Area 6 — Benchmark & big picture — **VALID AS A CLASSIFICATION BENCHMARK**

- Record-level benchmark recomputed from cache: RF 98.74, svm 97.73, gb 97.92,
  cnn1d 93.43, attention_cnn 93.94, **cnn_lstm 99.43**, resnet18 99.18, patchtst
  90.40, hybrid_pinn 90.34, **pc_cnn (CE-only) 98.99**, multitask_pinn 90.47. Matches
  `summary_record_level.md`.
- **No physics-labeled row beats vanilla on accuracy** (best vanilla cnn_lstm 99.43 ≥
  CE-only pc_cnn 98.99; the genuinely physics-trained hybrid/multitask are ~90).
  Honest relabeling (pc_cnn = CE-only; multitask = single-task; hybrid =
  rolling-element + constant metadata) is accurate.
- The backbone is corrected to the **528-record** unit (windows are not independent),
  which is the right call and was done before the surviving-claim analysis.

### Area 7 — The remediation itself — **DIAGNOSIS RIGHT, FIXES SOUND, ONE GAP**

- The diagnosis (inert argmax loss; wrong-bearing-type DB; flat baseline; window-level
  over-counting) is corroborated by the code and the byte-identical-w-sweep symptom
  history. The fixes (differentiable band-energy loss vs a frozen train-only healthy
  reference; per-sample rpm; record-level stats; quarantine of inert paths;
  estimand-consistent paired tests) are real and tested.
- The keep/redo/exclude scope is justified: §8.5 HybridPINN correctly **excluded**
  (still rolling-element). The estimand fix (don't pair a seed-mean gap with a
  representative-seed CI) is implemented and documented in the JSON.
- **The gap (F1):** the remediation reconciled `README.md`, `PROJECT_STATE.md`,
  `results/FINDINGS.md`, `results/README.md`, and `dataset_card.yaml` to the surviving
  verdict — but **missed the actual paper artifacts** in the legacy `config/docs`
  tree (see Area 8). New problems introduced: none that I found.

### Area 8 — Other findings

- **F1 (Major) — stale paper fabricates a different study.**
  `config/docs/paper/main.tex` (last touched in the early `P0` commit `19ab2ee`,
  never updated through Phase 6) and `config/docs/reports/Final_Report_UNVALIDATED.pdf`
  describe a study that **does not exist**:
  - abstract: *"improving both accuracy and physical consistency … evaluate on the
    **CWRU bearing dataset**, achieving **98.1% accuracy** … faithful explanations
    **validated by domain experts**."* The project uses a **synthetic journal-bearing**
    dataset (CWRU appears **nowhere** in code/data — I grepped), there is no 98.1%
    result, there are no domain experts, and the XAI positive was retracted.
  - methods: physics constraints written as **rolling-element** `f_BPFO`/`f_BPFI` —
    the exact wrong-bearing-type physics the remediation quarantined.
  - claims "statistically significant improvements" in accuracy — contradicts the
    verified verdict.
  This is not active fraud (it is a clearly pre-remediation relic, and the PDF is at
  least labeled `UNVALIDATED`), but it is a **landmine**: it is the only paper-shaped
  artifact in the repo and it is wholly wrong. Per "code/data beat prose," it must be
  deleted or rewritten from `results/FINDINGS.md` before any submission.
- **F6 (Minor) — w=0 provenance.** The CE-only (w=0) arm is the Phase-4 benchmark
  checkpoint (git `e498deb`), reused; the w=0.1/0.3/1.0 arms are from
  `run_phase5_gpu.py` (git `ce344d1`). Same architecture, same frozen budget, same
  data, so the comparison is fair, but the w-sweep is not a single self-contained run;
  state this in the paper.
- **F7 (Minor) — cosmetic doc mismatches.** `dataset_card.yaml:32`
  `training_windows: 17600` is actually the *total* windows across splits (train is
  12320 = 2464×5). The stale paper says IG = 50 steps; the actual run used 32
  (`alignment.json`). Neither affects results.

---

## 4. Big picture — answering the owner directly

**What is this project actually doing?** It builds a physics-grounded **synthetic**
journal-bearing vibration dataset (11 classes, severity-stratified, leakage-checked),
a frozen pre-registered **benchmark** of 11 classical/deep/physics-informed models,
and then asks a single sharp question: *does physics-informed training actually
help?* It answers that question carefully across noise, data-efficiency,
severity-OOD, interpretability, and calibration.

**Is it heading in a scientifically sound direction?** Yes. The methodology is
better than most published PINN-for-diagnosis work I have seen: record-level
statistics on the correct independent unit, exact McNemar with reported discordant
counts, cluster bootstrap, a consistent estimand, a **pre-registered** scrambled
control to interrogate its own surviving result, and a documented quarantine of
broken code. The team repeatedly chose the conservative reading and retracted its own
positives.

**Has it found anything real and defensible — or is it salvaging scraps?** It has
found something **real but small**, and it is **not** salvaging a scrap into a false
claim. The honest summary of the science is:

1. A usable synthetic dataset + generator (CI-locked). *Defensible.*
2. A frozen, reproducible, honestly-reported classification benchmark on which
   **physics buys no accuracy**. *Defensible — and it is a negative.*
3. A methodological caution (a naive frequency-consistency physics loss can be
   silently non-differentiable; five defects passed a green suite). *Defensible and
   genuinely useful.*
4. **One** surviving positive — a 5 dB noise-robustness gain at w=1.0 in a
   same-architecture ablation — which the team's **own control shows is a generic
   spectral regularizer, not physics.** *Defensible only with that exact framing.*

So the truthful headline is closer to: **"On a near-ceiling synthetic journal-bearing
benchmark, physics-informed training does not improve accuracy, interpretability, or
calibration; a high-weight band-energy consistency term does improve 5 dB noise
robustness in a same-architecture ablation, but a scrambled-reference control shows
this is generic spectral regularization rather than correct physics."** That is a
legitimate, if modest, scientific contribution — a careful negative plus a
well-characterized small effect — not a polished null masquerading as a discovery.

**Would I stake my own name on the surviving claim as written?** With the
maintainers' existing wording limits — "the implemented band-energy term at w=1.0
improved 5 dB robustness in a same-architecture ablation on synthetic data, n=3,
near-ceiling, shown by control to be a spectral regularizer not physics" — **yes,
narrowly.** Strip any of those qualifiers (especially "physics-informed learning
improves robustness," or any real-bearing/accuracy claim) and **no.** The danger is
not the current text; it is the stale `main.tex`, which says exactly the things you
may not say.

---

## 5. Verdict — trustworthy and publishable?

- **Trustworthy:** **Yes.** Within the surviving analysis I found no fabrication, no
  leakage, no broken estimand, and every headline number reproduced from raw
  checkpoints. The reporting is candid to a fault.
- **Publishable:** **Yes, as a dataset + benchmark + rigorous-negative paper** (with
  the methodological caution and the controlled, demoted noise effect), at the
  **honesty bar the maintainers already set.** Venue tier: **IEEE Access / Sensors /
  Measurement special issue / a workshop / arXiv** — *not* a top mechanical-systems or
  top ML venue. Reviewers will (correctly) push on n=3, the near-ceiling synthetic
  task, and the single-class mechanism; the paper must lead with those, not bury them.
- **Not publishable** in any form that frames the noise result as evidence that
  *physics* (or physics-informed learning generally) helps, or that implies
  real-world performance. The repo's own verdict already forbids this; the stale paper
  violates it.

**Honesty bar to clear before submission:** (1) more seeds (n≥10) to turn the
seed-level story from anecdote into inference; (2) at least one non-physics
regularizer control at matched strength to fully earn the word "generic"; (3) removal
or full rewrite of the stale paper artifacts.

---

## 6. Prioritized recommendations

1. **(F1, do first) Delete or rewrite the stale paper.** Remove/replace
   `config/docs/paper/main.tex` and `config/docs/reports/Final_Report_UNVALIDATED.pdf`,
   or regenerate the paper strictly from `results/FINDINGS.md §0`. As it stands the
   only paper-shaped file in the repo claims CWRU data, 98.1%, expert-validated XAI,
   and rolling-element physics — all false. This is the single highest-risk item.
2. **(F2) Add seeds.** Run the §8.4 w∈{0,1.0} and the F9 scramble arms at n≥10 seeds.
   This is the maintainers' own open item F13 and is now the difference between "a
   significant within-seed McNemar" and "a robust cross-seed effect." Cheap on GPU,
   high value, and it directly de-risks the only positive.
3. **(F5) Run a matched-strength non-physics regularizer control** (entropy penalty
   and/or random non-physical bands at w=1.0). Only this lets the paper say "generic
   regularizer" rather than "the correct per-class mapping is unnecessary."
4. **(F3/F4) Tighten the §8.7 wording in FINDINGS §0** to match
   `findings_bandenergy.md`: say the scramble reproduces the robustness *at the
   representative seed* and *for 2 of 3 seeds*, that the seed-mean degradation is
   *intermediate*, and that n=3 cannot resolve the "stability" sub-claim.
5. **(F6/F7) Provenance hygiene.** Add a one-line note that the w=0 arm is the reused
   benchmark checkpoint (same arch/budget); fix `dataset_card.yaml` `training_windows`;
   align the IG-steps number. Ship the per-table provenance manifest already on the
   open-items list.
6. **(Optional) State the ceiling explicitly as a limitation, not a footnote.** The
   synthetic task is near-ceiling; the entire positive lives in the last few percent
   on one class pair. Saying so up front strengthens, not weakens, the paper.

---

### Bottom line

This is **not** a case of polishing a null result into a publication. It is a
careful, reproducible, self-skeptical study whose honest result is a **rigorous
negative on physics-informed learning plus one small, controlled, non-physics noise
effect.** Everything I could recompute, reproduced. The science is defensible at a
mid-tier venue **if and only if** it is reported with the wording limits the
maintainers have already written — and the stale paper draft that violates those
limits is removed.
