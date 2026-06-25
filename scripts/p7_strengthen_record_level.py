"""§8.8 strengthen grid — record-level, SEED-LEVEL verdict (PROTOCOL §8.8).

The n=12 grid (4 arms x 12 seeds, results/noise_seed_robustness/) recomputed at the RECORD
level (528 records, soft-vote), then judged at the SEED level — the pre-registered
estimand that turns the old within-seed McNemar (n=3) into a real cross-seed result.

Arms: CE-only (w=0, same code path), correct band-energy (w=1.0), F9 scramble
(w=1.0, wrong REAL bands, §8.7), random-band (w=1.0, random NON-FAULT bands, §8.8).

Reuses the validated soft-vote + sanity-gate machinery in
scripts/phase5_bandenergy_record_level.py (window acc from each re-eval must match
the recorded metrics.json, or the script aborts).

Pre-registered decision rule (§8.8, fixed BEFORE the run; tau = 1.0 pt):
  - random ~ correct (both robust on >=10/12 seeds, both Wilcoxon-sig vs CE-only)
        -> GENERIC SPECTRAL REGULARIZER, earned (band locations don't matter).
  - random ~ CE-only (not robust) while correct/scramble robust
        -> narrow to "a spectral-FAULT-band regularizer".
  - intermediate -> report quantitatively, lean conservative.
Primary estimand: Wilcoxon signed-rank on paired per-seed clean->5dB degradation,
each arm vs CE-only (n=12); median degradation; robust-seed count (degr < tau).

Output: results/noise_seed_robustness/noise_seed_robustness_record_level.json
Usage:  python scripts/p7_strengthen_record_level.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase5_bandenergy_record_level import (  # noqa: E402
    load_split, record_probs, acc, mcnemar_full, repgap_pts, best_val_seed,
    metrics_for, ev_acc,
)

P7 = PROJECT_ROOT / 'results/noise_seed_robustness'
SEEDS = list(range(12))
TAU = 1.0  # pre-registered "robust" threshold: clean->5dB record degradation < 1.0 pt

ARMS = {
    'w0_CEonly':     P7 / 'pinn_ablation' / 'w0.0',
    'w1.0_correct':  P7 / 'pinn_ablation' / 'w1.0',
    'w1.0_scramble': P7 / 'pinn_ablation_scramble' / 'w1.0',
    'w1.0_random':   P7 / 'pinn_ablation_random' / 'w1.0',
}


def main():
    t0 = time.time()
    te_sig, te_lab, _ = load_split('test')
    s5_sig, s5_lab, _ = load_split('test_snr5')
    print(f'test records: {len(te_lab)}  | seeds: {len(SEEDS)}\n')

    data = {}
    for name, root in ARMS.items():
        dirs = [root / f'seed{s}' for s in SEEDS]
        clean, snr5, snr5_rep = [], [], {}
        for rd in dirs:
            m = metrics_for(rd)
            rp, tg = record_probs('physics_constrained_cnn', rd / 'best_model.pth',
                                  te_sig, te_lab, ev_acc(m, 'test'), 'test')
            rp5, tg5 = record_probs('physics_constrained_cnn', rd / 'best_model.pth',
                                    s5_sig, s5_lab, ev_acc(m, 'test_snr5'), 'snr5')
            clean.append(acc(rp, tg)); snr5.append(acc(rp5, tg5))
            snr5_rep[rd] = (rp5.argmax(1) == tg5)
        clean = np.array(clean); snr5 = np.array(snr5); degr = clean - snr5
        rep = best_val_seed(dirs)
        data[name] = {'clean': clean, 'snr5': snr5, 'degr': degr,
                      'snr5_rep': snr5_rep[rep], 'rep_seed': rep.name}
        n_robust = int(np.sum(degr < TAU))
        print(f'  {name:14s}: clean {clean.mean():.2f}±{clean.std():.2f} | '
              f'5dB {snr5.mean():.2f}±{snr5.std():.2f} | degr {degr.mean():.2f}±{degr.std():.2f} '
              f'(median {np.median(degr):.2f}) | robust {n_robust}/12')

    def arm_row(name):
        v = data[name]
        return {
            'clean_seedmean': [round(float(v['clean'].mean()), 2), round(float(v['clean'].std()), 2)],
            'snr5_seedmean': [round(float(v['snr5'].mean()), 2), round(float(v['snr5'].std()), 2)],
            'degradation_seedmean': round(float(v['degr'].mean()), 2),
            'degradation_median': round(float(np.median(v['degr'])), 2),
            'degradation_per_seed': [round(float(x), 2) for x in v['degr']],
            'snr5_per_seed': [round(float(x), 2) for x in v['snr5']],
            'robust_seed_count_tau1.0': int(np.sum(v['degr'] < TAU)),
            'representative_seed': v['rep_seed'],
        }

    out = {
        'scope': '§8.8 strengthen grid: record-level (528) + SEED-LEVEL (n=12). '
                 'Clean->5dB degradation per seed; Wilcoxon signed-rank vs CE-only; '
                 'robust-seed count (degr < tau=1.0). PROTOCOL §8.8.',
        'tau_robust_pt': TAU,
        'arms': {k: arm_row(k) for k in ARMS},
    }

    # ---- primary estimand: seed-level Wilcoxon, each arm vs CE-only -------------
    print('\n  seed-level Wilcoxon signed-rank vs CE-only (paired per-seed degradation, n=12):')
    ce = data['w0_CEonly']['degr']
    out['wilcoxon_vs_CEonly'] = {}
    for name in ('w1.0_correct', 'w1.0_scramble', 'w1.0_random'):
        d = data[name]['degr']
        diff = ce - d  # >0 => this arm degrades LESS than CE-only (more robust)
        # two-sided (headline, conservative) + one-sided 'more robust than CE-only'
        try:
            p_two = float(wilcoxon(ce, d, zero_method='wilcox').pvalue)
            p_less = float(wilcoxon(ce, d, alternative='greater', zero_method='wilcox').pvalue)
        except ValueError:  # all-zero differences
            p_two = p_less = 1.0
        out['wilcoxon_vs_CEonly'][name] = {
            'median_paired_degr_reduction_pts': round(float(np.median(diff)), 2),
            'mean_paired_degr_reduction_pts': round(float(np.mean(diff)), 2),
            'p_two_sided': p_two,
            'p_one_sided_more_robust': p_less,
            'n_seeds_more_robust_than_CEonly': int(np.sum(diff > 0)),
        }
        print(f'    {name:14s}: median Δdegr {np.median(diff):+.2f} | p(2-sided) {p_two:.4g} | '
              f'p(more-robust) {p_less:.4g} | {int(np.sum(diff > 0))}/12 seeds beat CE-only')

    # ---- continuity: representative best-val-seed McNemar @5dB vs CE-only -------
    out['repseed_mcnemar_vs_CEonly_5dB'] = {}
    for name in ('w1.0_correct', 'w1.0_scramble', 'w1.0_random'):
        bb, cc, p = mcnemar_full(data[name]['snr5_rep'], data['w0_CEonly']['snr5_rep'])
        out['repseed_mcnemar_vs_CEonly_5dB'][name] = {
            'discordant_arm_better_ce_better': [bb, cc], 'p': p,
            'repseed_gap_pts': repgap_pts(data[name]['snr5_rep'], data['w0_CEonly']['snr5_rep'])}

    # ---- pre-registered decision-rule evaluation -------------------------------
    rob = {k: out['arms'][k]['robust_seed_count_tau1.0'] for k in ARMS}
    wil = out['wilcoxon_vs_CEonly']
    def sig(name): return wil[name]['p_two_sided'] < 0.05
    correct_robust = rob['w1.0_correct'] >= 10 and sig('w1.0_correct')
    random_robust = rob['w1.0_random'] >= 10 and sig('w1.0_random')
    if random_robust and correct_robust:
        verdict = ('GENERIC SPECTRAL REGULARIZER (earned): random non-fault bands '
                   'reproduce the robustness like correct physics — band locations '
                   'do not matter. FINDINGS §0 wording stands.')
    elif correct_robust and rob['w1.0_random'] < rob['w1.0_correct'] and not random_robust:
        verdict = ('SPECTRAL-FAULT-BAND regularizer: random bands do NOT reproduce '
                   'the robustness while correct/scramble do — the effect needs REAL '
                   'fault-frequency bands. Narrow FINDINGS §0 accordingly.')
    else:
        verdict = ('NEITHER arm is robust at n=12 (robust-seed counts below 10/12 '
                   'and/or Wilcoxon n.s.): the n=3 noise-robustness benefit does NOT '
                   'replicate at n=12 — it was seed-fragile. Report conservative.')
    out['decision'] = {
        'rule': 'PROTOCOL §8.8; tau=1.0pt; robust = degr<tau on >=10/12 seeds AND '
                'Wilcoxon p<0.05 vs CE-only.',
        'robust_seed_counts': rob,
        'correct_meets_robust_bar': bool(correct_robust),
        'random_meets_robust_bar': bool(random_robust),
        'verdict': verdict,
    }
    print(f'\n  robust-seed counts (degr<{TAU}): ' +
          ' | '.join(f'{k} {v}/12' for k, v in rob.items()))
    print(f'\n  VERDICT: {verdict}')

    (P7 / 'noise_seed_robustness_record_level.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'\n  wrote results/noise_seed_robustness/noise_seed_robustness_record_level.json in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
