# Physics-loss diagnosis & differentiable fix (§8.0-bis)

> Root-cause analysis and fix design for the discovery that
> `PhysicsConstrainedCNN.compute_physics_loss` is **non-differentiable** and
> therefore contributes **zero gradient** to training. Written for owner
> review. The fix is implemented on branch `p5/physics-loss-fix` and is NOT
> merged until the 6 pending Phase-5 runs complete on the inert ("before")
> code.

## 1. The symptom (execution evidence)

In the 39/45 Phase-5 partial download
(`results/results_phase5-20260613T073028Z-3-001`), the §8.4 ablation runs at
`w=0.1` and `w=0.3` are **byte-identical per seed**:

| seed | clean acc | 5 dB acc | best epoch | best val |
|---|---|---|---|---|
| 0 | 95.6061 | 93.1439 | 54 | 96.7424 |
| 1 | 95.8712 | 86.8939 | 28 | 96.3258 |
| 2 | 96.4773 | 92.9545 | 36 | 97.0076 |

Identical to 4 decimal places on every metric. Two different physics weights
produced the *same trained model*. The data_efficiency `frac100` pc_cnn runs
(also `w=0.3`) match the same three rows — consistent, since it is the same
configuration.

## 2. The root cause (code)

`compute_physics_loss` ([physics_constrained_cnn.py:123](../packages/core/models/pinn/physics_constrained_cnn.py)):

```python
predicted_classes = torch.argmax(predictions, dim=1)   # line 154 — NON-DIFFERENTIABLE
...
expected_freqs = self.signature_db.get_expected_frequencies(pred_class, rpm, top_k)
peak_freqs     = freq_bins[topk(magnitude_of_input_signal)]
loss_i         = F.relu(min_distance - tolerance)       # built from input FFT + DB lookup
```

`torch.argmax` returns integer indices — it has no gradient. Everything
downstream is then a function of (a) the **input signal's** FFT magnitude and
(b) a **frequency-table lookup** keyed by the argmax'd class and rpm. Neither
depends on a model parameter through a differentiable path. So:

```
phys.requires_grad == False
phys.grad_fn       == None
phys.backward()    -> RuntimeError: element 0 ... does not require grad
```

(All three verified directly.) Adding `physics_w * phys` to the CE loss adds a
**constant**: its gradient w.r.t. the weights is exactly zero, so `physics_w`
cannot change training. This is **§8.0 round 2** — Phase 4 never *called* the
loss; Phase 5 calls it but it carries no gradient. Net effect is identical:
pc_cnn's "physics" never touches its weights.

## 3. The fix (design)

The loss should reward predictions whose class is **spectrally consistent**
with the signal, in a way that flows gradient to the logits. Replace the hard
`argmax` selection with a **soft, softmax-probability-weighted** penalty over
all classes:

```
probs[B,C]  = softmax(logits)                      # differentiable in params
pen[B,C]    = per-class frequency-mismatch penalty # built from input FFT + DB
                                                    #   (constant w.r.t. params)
physics_loss = (probs * pen).sum(dim=1).mean()     # gradient: d/dprobs = pen
```

Mechanism: `pen[i,c]` measures how poorly sample *i*'s observed spectral peaks
match class *c*'s expected characteristic frequencies (same frequency-DB and
tolerance logic as before, now computed for **every** class rather than only
the argmax'd one). Weighting by `probs` means the model is penalized for
placing probability mass on a fault class whose characteristic frequencies are
**absent** from the signal — and the gradient pushes that probability down.
Healthy/`sain` and any class with no characteristic fault frequencies get
`pen = 0` (no penalty), which is physically correct.

Properties:
- `physics_loss.requires_grad == True`, `.backward()` produces non-zero
  parameter gradients (verified).
- Different `physics_w` now changes the gradient → §8.4 becomes a real test.
- No new dependencies; ~C× the physics-loss FLOPs (C=11 classes vs 1) — cheap
  relative to the backbone.
- The penalty term `pen` is detached-by-construction (no params in it), so the
  only learning signal is "shift probability away from spectrally-inconsistent
  classes", which is exactly the intended physics constraint.

## 4. What this fix does and does NOT require re-running

| Artifact | Uses `compute_physics_loss`? | Re-run after fix? |
|---|---|---|
| Phase-4 benchmark (`results/benchmark/`) | No — pure CE, frozen protocol | **NO** |
| §8.1 noise robustness | No — eval of frozen checkpoints | **NO** |
| §8.2 data-efficiency (pc_cnn arm) | Yes (w=0.3) | Optional — to test if real physics changes the curve |
| §8.3 severity-OOD (pc_cnn arm) | Yes (w=0.3) | Optional — same |
| §8.4 PINN ablation | Yes (the whole point) | **YES** — this is the experiment the fix unlocks |
| §8.5 hybrid_pinn true-metadata | No — physics via forward path | **NO** (separate, live mechanism) |

So the fix's rerun scope is **§8.4 (required) ± §8.2/8.3 (optional)** on Colab,
under the identical frozen budget and seeds {0,1,2}. The "before" numbers are
the inert runs already on `p5/physics-exp`; the "after" numbers come from this
branch. That before/after contrast is itself a reportable methodological
finding for the paper.

## 5. Process / gating

- This is a **physics-mechanism change** → owner-ratified, recorded as a dated
  `experiments/PROTOCOL.md §7` amendment before any rerun (the §8.4
  pre-registration is unchanged; only the loss implementation is corrected,
  which the amendment documents).
- Branch `p5/physics-loss-fix` holds the code; it is merged only after the 6
  pending inert-code runs land and the owner approves this design.
