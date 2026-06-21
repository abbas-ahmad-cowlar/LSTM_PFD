"""F9 scrambled-reference control — record-level verdict (PROTOCOL §8.7).

Compares the scrambled-physics w=1.0 arm against CE-only (w=0) and correct-physics
w=1.0 at 5 dB, at the RECORD level (528 records, soft-vote), reusing the validated
machinery in scripts/phase5_bandenergy_record_level.py (same cache, same sanity
gate, same representative best-val-seed McNemar).

Pre-registered decision rule (§8.7, fixed before the control ran):
  - scramble degradation ~ CE-only (large) AND >> correct-w1.0 (~0) -> PHYSICS-SPECIFIC.
  - scramble degradation ~ correct-w1.0 (small/robust)             -> GENERIC REG.
  - intermediate -> report quantitatively, lean conservative (narrow wording).

Output: results/phase5_bandenergy/f9_scramble_record_level.json
Usage:  python scripts/f9_scramble_record_level.py
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase5_bandenergy_record_level import (  # noqa: E402
    load_split, record_probs, acc, mcnemar_full, repgap_pts, best_val_seed,
    metrics_for, ev_acc, BE, BENCH, SEEDS,
)


def main():
    te_sig, te_lab, _ = load_split('test')
    s5_sig, s5_lab, _ = load_split('test_snr5')
    print(f'test records: {len(te_lab)}')

    arms = {
        'w0_CEonly':     [BENCH / 'physics_constrained_cnn' / f'seed{s}' for s in SEEDS],
        'w1.0_correct':  [BE / 'pinn_ablation' / 'w1.0' / f'seed{s}' for s in SEEDS],
        'w1.0_scramble': [BE / 'pinn_ablation_scramble' / 'w1.0' / f'seed{s}' for s in SEEDS],
    }

    data = {}
    for name, dirs in arms.items():
        clean, snr5, snr5_corr = [], [], {}
        for rd in dirs:
            m = metrics_for(rd)
            exp_clean = ev_acc(m, 'test') if 'evals' in m else m['accuracy']
            rp, tg = record_probs('physics_constrained_cnn', rd / 'best_model.pth',
                                  te_sig, te_lab, exp_clean, 'test')
            clean.append(acc(rp, tg))
            exp_s5 = ev_acc(m, 'test_snr5') if ('evals' in m and 'test_snr5' in m['evals']) else None
            rp5, tg5 = record_probs('physics_constrained_cnn', rd / 'best_model.pth',
                                    s5_sig, s5_lab, exp_s5, 'snr5')
            snr5.append(acc(rp5, tg5))
            snr5_corr[rd] = (rp5.argmax(1) == tg5)
        rep = best_val_seed(dirs)
        data[name] = {'clean': clean, 'snr5': snr5, 'snr5_rep': snr5_corr[rep],
                      'rep_seed': rep.name}
        degr = float(np.mean(clean) - np.mean(snr5))
        print(f'  {name:14s}: clean {np.mean(clean):.2f}±{np.std(clean):.2f} | '
              f'5dB {np.mean(snr5):.2f}±{np.std(snr5):.2f} | degr {degr:.2f} | rep {rep.name}')

    def arm_row(name):
        v = data[name]
        return {'clean': [round(float(np.mean(v['clean'])), 2), round(float(np.std(v['clean'])), 2)],
                'snr5': [round(float(np.mean(v['snr5'])), 2), round(float(np.std(v['snr5'])), 2)],
                'snr5_per_seed': [round(x, 2) for x in v['snr5']],
                'degradation_seedmean': round(float(np.mean(v['clean']) - np.mean(v['snr5'])), 2),
                'representative_seed': v['rep_seed']}

    out = {
        'scope': 'F9 scrambled-reference control vs CE-only and correct-physics w=1.0, '
                 'record-level (528), 5 dB. PROTOCOL §8.7.',
        'arms': {k: arm_row(k) for k in arms},
    }

    # representative best-val-seed McNemar at 5 dB (same estimand as repseed gap)
    print('\n  representative best-val-seed McNemar @5 dB (b-c, p, gap):')
    for a, b in [('w1.0_scramble', 'w0_CEonly'), ('w1.0_scramble', 'w1.0_correct')]:
        bb, cc, p = mcnemar_full(data[a]['snr5_rep'], data[b]['snr5_rep'])
        gap = repgap_pts(data[a]['snr5_rep'], data[b]['snr5_rep'])
        out[f'mcnemar_{a}_vs_{b}_5dB'] = {'discordant_a_better_b_better': [bb, cc],
                                          'p': p, 'repseed_gap_pts': gap}
        print(f'    {a} vs {b}: {bb}-{cc}  p={p:.6g}  repseed_gap={gap:+.2f}')

    # pre-registered decision-rule inputs
    d_ce = out['arms']['w0_CEonly']['degradation_seedmean']
    d_correct = out['arms']['w1.0_correct']['degradation_seedmean']
    d_scr = out['arms']['w1.0_scramble']['degradation_seedmean']
    out['decision_inputs'] = {
        'degradation_seedmean': {'CEonly': d_ce, 'correct_w1.0': d_correct, 'scramble_w1.0': d_scr},
        'scramble_snr5_per_seed': out['arms']['w1.0_scramble']['snr5_per_seed'],
        'note': 'Apply §8.7 rule to these + the McNemar; mind the scramble seed variance.',
    }
    (BE / 'f9_scramble_record_level.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"\n  degradation (seed-mean): CE-only {d_ce} | correct-w1.0 {d_correct} | scramble-w1.0 {d_scr}")
    print('  wrote results/phase5_bandenergy/f9_scramble_record_level.json')


if __name__ == '__main__':
    main()
