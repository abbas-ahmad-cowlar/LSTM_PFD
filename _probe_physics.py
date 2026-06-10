"""Scratch probe for test margin calibration. DELETE after use."""
import numpy as np
from scipy import signal as sps
from scipy.stats import kurtosis

from config.data_config import DataConfig
from data.signal_generation import FaultModeler, SignalGenerator

OMEGA = 60.0


def make_cfg(T):
    c = DataConfig()
    c.signal.T = T
    return c


def gen(m, fault, seed, sev=0.8, S=0.15, sv=1.0):
    np.random.seed(seed)
    N = m.N
    return m.generate_fault_signal(
        fault, np.full(N, sev), np.ones(N),
        omega=2 * np.pi * OMEGA, Omega=OMEGA,
        load_factor=0.65, temp_factor=1.0, operating_factor=0.65,
        physics_factor=1.0, sommerfeld=S, speed_variation=sv,
    )


def psd(x, fs):
    return sps.welch(x, fs=fs, nperseg=len(x))


def be(f, P, lo, hi):
    msk = (f >= lo) & (f <= hi)
    return float(np.sum(P[msk]) * (f[1] - f[0]))


def flatness(P):
    P = np.clip(P, 1e-30, None)
    return float(np.exp(np.mean(np.log(P))) / np.mean(P))


m2 = FaultModeler(make_cfg(2.0))
m5 = FaultModeler(make_cfg(5.0))
fs = 20480

print("== lubrification kurtosis (T=5, sev .8) ==")
for seed in (0, 1, 2, 3, 4):
    x = gen(m5, "lubrification", seed)
    print(seed, "pearson:", round(kurtosis(x, fisher=False), 3))

print("== lubrification band + monotonicity ==")
x = gen(m5, "lubrification", 3)
f, P = psd(x, fs)
print("E(1-6):", be(f, P, 1, 6), "E(50-70):", be(f, P, 50, 70))
a = gen(m5, "lubrification", 3, S=0.08)
b = gen(m5, "lubrification", 3, S=0.40)
print("rms S=.08:", np.sqrt(np.mean(a**2)), "S=.40:", np.sqrt(np.mean(b**2)))

print("== cavitation (T=2) ==")
for seed in (0, 1, 2):
    x = gen(m2, "cavitation", seed)
    f, P = psd(x, fs)
    tot = be(f, P, 0, fs / 2)
    print(seed, "frac 1400-2600:", round(be(f, P, 1400, 2600) / tot, 3),
          "kurt:", round(kurtosis(x, fisher=False), 1))

print("== oilwhirl dominant (T=2) ==")
for seed in (0, 1, 2):
    x = gen(m2, "oilwhirl", seed)
    f, P = psd(x, fs)
    msk = f >= 1.0
    print(seed, "peak:", f[msk][np.argmax(P[msk])])
a = gen(m2, "oilwhirl", 1, S=0.08)
b = gen(m2, "oilwhirl", 1, S=0.40)
print("rms S=.08:", np.sqrt(np.mean(a**2)), "S=.40:", np.sqrt(np.mean(b**2)))

print("== jeu (T=2) ==")
x = gen(m2, "jeu", 0)
f, P = psd(x, fs)
msk = f >= 1.0
print("global peak:", f[msk][np.argmax(P[msk])])
print("E(sub 24-30):", be(f, P, 24, 30), "E(58-62):", be(f, P, 58, 62),
      "E(118-122):", be(f, P, 118, 122), "E(300-400):", be(f, P, 300, 400))

print("== desalignement (T=2) ==")
x = gen(m2, "desalignement", 0)
f, P = psd(x, fs)
for lo, hi in [(110, 130), (170, 190)]:
    msk = (f >= lo) & (f <= hi)
    print("peak in", lo, hi, ":", f[msk][np.argmax(P[msk])])
print("E1X:", be(f, P, 58, 62), "E2X:", be(f, P, 118, 122), "E3X:", be(f, P, 178, 182))
fl_desal = flatness(P)
print("flatness:", fl_desal)
a = gen(m2, "desalignement", 5, sev=0.9)
b = gen(m2, "desalignement", 5, sev=0.3)
print("rms sev .9:", np.sqrt(np.mean(a**2)), ".3:", np.sqrt(np.mean(b**2)))

print("== desequilibre (T=2) ==")
x = gen(m2, "desequilibre", 0)
f, P = psd(x, fs)
msk = f >= 1.0
print("global peak:", f[msk][np.argmax(P[msk])])
a = gen(m2, "desequilibre", 2, sv=1.1)
b = gen(m2, "desequilibre", 2, sv=0.9)
print("rms sv 1.1:", np.sqrt(np.mean(a**2)), "0.9:", np.sqrt(np.mean(b**2)))

print("== usure (T=2) ==")
x = gen(m2, "usure", 0)
f, P = psd(x, fs)
print("flatness:", flatness(P), "vs desal:", fl_desal)
print("E1X:", be(f, P, 58, 62), "Efar:", be(f, P, 300, 400))

print("== mixed_misalign_imbalance (T=2) ==")
x = gen(m2, "mixed_misalign_imbalance", 0)
f, P = psd(x, fs)
print("E1X:", be(f, P, 58, 62), "E2X:", be(f, P, 118, 122),
      "E3X:", be(f, P, 178, 182), "Efar:", be(f, P, 300, 400))

print("== mixed_wear_lube (T=5) ==")
x = gen(m5, "mixed_wear_lube", 0)
f, P = psd(x, fs)
print("E(1-6):", be(f, P, 1, 6), "Efar:", be(f, P, 300, 400),
      "flatness:", flatness(P))

print("== mixed_cavit_jeu (T=2) ==")
for seed in (0, 1):
    x = gen(m2, "mixed_cavit_jeu", seed)
    f, P = psd(x, fs)
    msk = (f >= 24) & (f <= 30)
    print(seed, "E(1400-2600):", be(f, P, 1400, 2600),
          "E(4000-5200):", be(f, P, 4000, 5200),
          "Esub:", be(f, P, 24, 30), "E(33-39):", be(f, P, 33, 39),
          "sub peak:", f[msk][np.argmax(P[msk])])

print("== generator-level sain RMS (T=0.5) ==")
c = DataConfig(num_signals_per_fault=8, rng_seed=42)
c.signal.T = 0.5
c.augmentation.enabled = False
c.fault.include_healthy = True
for k in c.fault.single_faults:
    c.fault.single_faults[k] = k in ("desequilibre", "oilwhirl")
for k in c.fault.mixed_faults:
    c.fault.mixed_faults[k] = False
g = SignalGenerator(c)
ds = g.generate_dataset()
import collections
acc = collections.defaultdict(list)
for sig, lab in zip(ds["signals"], ds["labels"]):
    acc[lab].append(np.sqrt(np.mean(sig**2)))
for lab, v in acc.items():
    print(lab, "mean rms:", np.mean(v))
