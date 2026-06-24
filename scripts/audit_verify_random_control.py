"""Independent verification that the §8.8 random-band arm is a genuine control.

Checks, from the committed artifact + the real DB:
  1. matched STRUCTURE: per class, random tonal count == real tonal count and
     random band count == real band count; each random tonal half-width equals the
     real one it replaced; each random band width equals the real band width.
  2. NON-PHYSICAL: every random tonal multiplier interval is disjoint from every
     real tonal multiplier interval (holds at ALL rpm, multiplier space); every
     random Hz band is disjoint from every real Hz band AND every real tonal Hz
     footprint over the dataset rpm range.
  3. IDENTICAL loss form + GRADIENTS FLOW: build the model with random_signature /
     random_reference, compute the physics loss on a batch, confirm it is finite,
     >0, and backprops a non-zero gradient to model parameters (same as correct).

Auditor: Claude/Opus  2026-06-24
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packages.core.models.physics.fault_signatures import (  # noqa: E402
    FaultSignatureDatabase, load_random_reference)
from packages.core.models.model_factory import create_model  # noqa: E402
from utils.constants import FAULT_TYPES  # noqa: E402

RPM_MIN, RPM_MAX = 3240.0, 3960.0
OMEGA_MIN, OMEGA_MAX = RPM_MIN / 60.0, RPM_MAX / 60.0


def overlaps(a, b, intervals, margin=0.0):
    return any(not (b + margin <= lo or a - margin >= hi) for lo, hi in intervals)


def main():
    db = FaultSignatureDatabase()
    rand_sig, rand_ref = load_random_reference()
    assert rand_sig is not None, "random reference missing"

    # ---- 1. matched structure ----
    print('=== 1. matched structure (real vs random, per class) ===')
    struct_ok = True
    for nm in FAULT_TYPES:
        real, rnd = db.signatures[nm], rand_sig[nm]
        n_t_ok = len(real.tonal) == len(rnd.tonal)
        n_b_ok = len(real.bands_hz) == len(rnd.bands_hz)
        # half-widths preserved (tonal), widths preserved (bands)
        hw_ok = all(abs(rt[1] - tt[1]) < 1e-9 for tt, rt in zip(real.tonal, rnd.tonal))
        w_ok = all(abs((rb[1] - rb[0]) - (tb[1] - tb[0])) < 1e-6
                   for tb, rb in zip(real.bands_hz, rnd.bands_hz))
        ok = n_t_ok and n_b_ok and hw_ok and w_ok
        struct_ok &= ok
        if real.tonal or real.bands_hz:
            print(f'  {nm:26s} tonal {len(real.tonal)}->{len(rnd.tonal)} '
                  f'bands {len(real.bands_hz)}->{len(rnd.bands_hz)} '
                  f'hw_match={hw_ok} width_match={w_ok}  {"OK" if ok else "MISMATCH"}')
    print('  STRUCTURE MATCHED:', struct_ok)

    # ---- 2. non-physical (disjoint from all real bands at any rpm) ----
    print('\n=== 2. non-overlap with ALL real characteristic bands ===')
    real_mult = [(m * (1 - hw), m * (1 + hw))
                 for nm in FAULT_TYPES for m, hw in db.signatures[nm].tonal]
    real_hz = [(lo, hi) for nm in FAULT_TYPES for lo, hi in db.signatures[nm].bands_hz]
    real_tonal_hz = [(m * OMEGA_MIN * (1 - hw), m * OMEGA_MAX * (1 + hw))
                     for nm in FAULT_TYPES for m, hw in db.signatures[nm].tonal]
    tonal_viol = band_viol = 0
    for nm in FAULT_TYPES:
        for (m, hw) in rand_sig[nm].tonal:
            if overlaps(m * (1 - hw), m * (1 + hw), real_mult):
                tonal_viol += 1; print(f'  TONAL OVERLAP {nm} {m}')
        for (lo, hi) in rand_sig[nm].bands_hz:
            if overlaps(lo, hi, real_hz + real_tonal_hz):
                band_viol += 1; print(f'  BAND OVERLAP {nm} {lo}-{hi}')
    print(f'  tonal overlaps with real: {tonal_viol}')
    print(f'  band  overlaps with real: {band_viol}')
    print('  ALL RANDOM BANDS NON-PHYSICAL:', tonal_viol == 0 and band_viol == 0)
    # show the actual random band Hz ranges (sanity: are they plausibly off-fault?)
    print('  random absolute Hz bands:',
          sorted({(lo, hi) for nm in FAULT_TYPES for lo, hi in rand_sig[nm].bands_hz}))
    print('  real absolute Hz bands:  ', sorted(set(real_hz)))

    # ---- 3. identical loss form + gradient flow ----
    print('\n=== 3. loss is finite/positive and backprops (random vs correct) ===')
    torch.manual_seed(0)
    sig = torch.randn(8, 1, 20480, requires_grad=False)
    meta = {'rpm': torch.full((8,), 3600.0)}

    def grad_norm(use_random):
        m = create_model('physics_constrained_cnn', num_classes=11)
        if use_random:
            m.random_signature, m.random_reference = rand_sig, rand_ref
        m.train()
        logits = m(sig)
        loss, d = m.compute_physics_loss(sig, logits, meta)
        m.zero_grad()
        loss.backward()
        g = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
        return float(loss), d, g

    lc, dc, gc = grad_norm(False)
    lr_, dr, gr = grad_norm(True)
    print(f'  correct: loss={lc:.4f} mean_pen={dc["mean_penalty"]:.4f} grad_sum={gc:.4e}')
    print(f'  random : loss={lr_:.4f} mean_pen={dr["mean_penalty"]:.4f} grad_sum={gr:.4e}')
    print('  GRADIENTS FLOW (random):', gr > 0)


if __name__ == '__main__':
    main()
