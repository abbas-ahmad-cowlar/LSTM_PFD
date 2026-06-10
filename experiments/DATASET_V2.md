# Dataset v2 Design Note

> **Status**: Proposal for owner ratification (Convergence Plan P3.3).
> Every Phase 4–5 experiment maps to a design feature here. Once ratified and
> generated, the generator is **frozen** (Plan Part IV rule 8): changes bump
> the version (v2.1) and trigger reruns of affected experiments only.

## A. Windowing — 5 s records, 1 s training windows

**Decision**: Generate full **5 s records** (102,400 samples — unchanged
physics, keeps low-frequency content like 2–5 Hz stick-slip resolvable at
generation), but train/evaluate on **1 s windows (20,480 samples)** cut at
**load time** by a `WindowedView` wrapper over `BearingFaultDataset`
(5 non-overlapping windows per record).

- **Group-aware splits**: all windows of one record share that record's split.
  Splits are computed at the *record* level at generation; the wrapper never
  crosses records. Verified by `check_data_leakage.py` (record-hash based).
- **Why**: ~5× more training samples at ~5× less compute each — the decisive
  feasibility lever for CPU/basic-GPU training. Window-at-load (vs at
  generation) keeps the HDF5 5× smaller and the window length changeable
  without regeneration.
- **Caveat accepted**: a 1 s window sees only 1–6 periods of the 1–6 Hz
  stick-slip signature (lubrification). The impacts + temp/Sommerfeld coupling
  still mark the class; if per-class accuracy on `lubrification` collapses,
  the fallback is 2 s windows (documented decision point, not a silent change).

## B. Composition & severity stratification

| Parameter | v1 | **v2** |
|---|---|---|
| Records per class | 200 + 30% aug | **320 base + 0 aug** (augmentation moves to train-time transforms if needed — cleaner provenance) |
| Classes | 11 | 11 (unchanged taxonomy) |
| Total records | 2,860 | **3,520** |
| Training windows | — | ~12,320 train / 2,640 val / 2,640 test (70/15/15 record-level) |
| Severity | random choice | **stratified**: 80 records per class per severity level (incipient/mild/moderate/severe), deterministic cycling |
| Size | 1.0 GB | ~1.4 GB |

Stratification is implemented via a `severity_override` parameter on
`generate_single_signal` (backward-compatible; default None = v1 random
behavior); `scripts/generate_dataset_v2.py` cycles the four levels
deterministically.

**`sain` severity convention**: healthy records have no physical severity
(metadata records `nominal`), but each carries a severity *slot label*
(cycled like the faults) in the per-split `severities` array. Consequence:
severity-filtered OOD subsets (e.g. "test on severe only") automatically
retain a balanced share of healthy records — the diagnosis task stays
11-class in every condition. The slot label is a sampling device, not a
physical claim; `metadata.severity` is authoritative.

## C. Designed-in experiment support

1. **Severity-shift OOD (P5.2)**: with 80 records/class/severity, the splits
   "train on incipient+mild+moderate, test on severe" (and reverse) are pure
   metadata filters — no regeneration.
2. **Condition-shift checks**: speed (±10%), load (30–100%), temperature
   (40–80 °C) sampled per record and stored in metadata → enables post-hoc
   "train low-load / test high-load" analyses without new data.
3. **Noise-robustness curves (P5.3)**: the **test split only** is additionally
   exported at 4 SNR variants — clean (as generated), **20 dB, 10 dB, 5 dB** —
   by adding calibrated AWGN relative to each record's clean RMS. Stored as
   `test_snr20/`, `test_snr10/`, `test_snr5/` groups in the same HDF5.
   No retraining needed: frozen Phase-4 checkpoints are evaluated across
   variants.
4. **Physics validity (P3.2)**: every record passes the spectral-signature
   test battery (sampled per class) before the file is accepted.

## D. Provenance & validation gates

- Seed: 42 (generation), recorded per record with full operating-condition
  metadata (severity, rpm, load%, temp, Sommerfeld, Reynolds).
- File: `data/generated/dataset_v2.h5` + `dataset_card.yaml` update + DVC.
- Acceptance gates (all must pass before v2 is used):
  1. `pytest tests/test_physics_signatures.py` green on sampled v2 records;
  2. class balance exact (320/class), severity balance exact (80/class/level);
  3. `check_data_leakage.py`: zero record-hash overlap across splits;
  4. `dataset_comparison.py` v1-vs-v2 statistics report archived to
     `results/dataset_v2_validation/`.

## E. What v2 deliberately does NOT include

- Advanced-physics toggles (PHYSICS.md §5) — off, frozen.
- Real/imported data — out of scope (BACKLOG: sim-to-real).
- TFR/spectrogram exports — cut in Phase 2.
- Variable signal duration / sampling rate — fixed at 5 s / 20,480 Hz.

---

*Owner ratification (P3.3 DoD): pending. Ratifying this note authorizes the
generator change (severity stratification flag + SNR test variants), v2
generation (P3.4, ~1 h laptop), and the CNN1D v2 re-baseline (P3.5, overnight).*
