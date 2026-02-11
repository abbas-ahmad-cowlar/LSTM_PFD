# Feature Catalog — Definitive Feature Reference

> All 52 features implemented in the Features module, verified against source code.

## Summary

| Category                      | Count  | File                   |
| ----------------------------- | ------ | ---------------------- |
| Time-Domain                   | 7      | `time_domain.py`       |
| Frequency-Domain              | 12     | `frequency_domain.py`  |
| Envelope Analysis             | 4      | `envelope_analysis.py` |
| Wavelet / Cepstral            | 7      | `wavelet_features.py`  |
| Bispectrum (Higher-Order)     | 6      | `bispectrum.py`        |
| **Base Total**                | **36** | —                      |
| CWT (Advanced)                | 4      | `advanced_features.py` |
| WPT (Advanced)                | 4      | `advanced_features.py` |
| Nonlinear Dynamics (Advanced) | 4      | `advanced_features.py` |
| Spectrogram / STFT (Advanced) | 4      | `advanced_features.py` |
| **Advanced Total**            | **16** | —                      |
| **Grand Total**               | **52** | —                      |

---

## Base Features (36)

These are extracted by `FeatureExtractor.extract_features()` in the canonical order defined by `_get_feature_names()`.

### Time-Domain Features (7)

Extracted by `extract_time_domain_features()` in `time_domain.py`.

| #   | Feature Name      | Formula                | Description                                        | Line     |
| --- | ----------------- | ---------------------- | -------------------------------------------------- | -------- |
| 1   | `RMS`             | √(mean(x²))            | Root mean square — signal energy                   | L19–31   |
| 2   | `Kurtosis`        | E[(x−μ)⁴]/σ⁴ − 3       | 4th moment, excess (Fisher). Detects impulsiveness | L34–47   |
| 3   | `Skewness`        | E[(x−μ)³]/σ³           | 3rd moment. Measures distribution asymmetry        | L50–62   |
| 4   | `CrestFactor`     | peak / RMS             | Peak-to-RMS ratio. Indicates impulsive behavior    | L65–80   |
| 5   | `ShapeFactor`     | RMS / mean(\|x\|)      | Shape of the amplitude distribution                | L83–97   |
| 6   | `ImpulseFactor`   | peak / mean(\|x\|)     | Peak-to-mean absolute ratio                        | L100–114 |
| 7   | `ClearanceFactor` | peak / (mean(√\|x\|))² | Sensitive to early bearing defects                 | L117–131 |

### Frequency-Domain Features (12)

Extracted by `extract_frequency_domain_features()` in `frequency_domain.py`. Uses FFT-based PSD computation.

| #   | Feature Name         | Formula                 | Description                                | Line     |
| --- | -------------------- | ----------------------- | ------------------------------------------ | -------- |
| 8   | `DominantFreq`       | argmax(PSD)             | Peak frequency in the spectrum (Hz)        | L43–55   |
| 9   | `SpectralCentroid`   | Σ(f·P(f)) / Σ(P(f))     | Center of mass of spectrum (Hz)            | L58–75   |
| 10  | `SpectralEntropy`    | −Σ(P_norm·ln(P_norm))   | Shannon entropy; high = broadband noise    | L78–97   |
| 11  | `LowBandEnergy`      | Σ PSD for 0–500 Hz      | Low-frequency band energy                  | L100–116 |
| 12  | `MidBandEnergy`      | Σ PSD for 500–2000 Hz   | Mid-frequency band energy                  | L100–116 |
| 13  | `HighBandEnergy`     | Σ PSD for 2000–5000 Hz  | High-frequency band energy                 | L100–116 |
| 14  | `VeryHighBandEnergy` | Σ PSD for 5000–10000 Hz | Very high-frequency band energy            | L100–116 |
| 15  | `TotalSpectralPower` | Σ PSD                   | Total spectral power                       | L156–224 |
| 16  | `SpectralStd`        | std(PSD)                | Standard deviation of PSD                  | L156–224 |
| 17  | `Harmonic2X1X`       | amp(2f₀) / amp(f₀)      | 2× harmonic ratio — indicates misalignment | L119–153 |
| 18  | `Harmonic3X1X`       | amp(3f₀) / amp(f₀)      | 3× harmonic ratio — indicates looseness    | L119–153 |
| 19  | `SpectralPeakiness`  | max(PSD) / mean(PSD)    | Peak-to-mean ratio in spectrum             | L156–224 |

### Envelope Analysis Features (4)

Extracted by `extract_envelope_features()` in `envelope_analysis.py`. Uses Hilbert transform.

| #   | Feature Name       | Formula          | Description                                                                                | Line    |
| --- | ------------------ | ---------------- | ------------------------------------------------------------------------------------------ | ------- |
| 20  | `EnvelopeRMS`      | √(mean(env²))    | RMS of Hilbert envelope                                                                    | L40–50  |
| 21  | `EnvelopeKurtosis` | kurtosis(env)    | Envelope impulsiveness                                                                     | L53–66  |
| 22  | `EnvelopePeak`     | max(env)         | Peak value of envelope                                                                     | L69–79  |
| 23  | `ModulationFreq`   | argmax(FFT(env)) | Dominant modulation frequency (Hz) — often corresponds to fault characteristic frequencies | L82–110 |

### Wavelet / Cepstral Features (7)

Extracted by `extract_wavelet_features()` in `wavelet_features.py`. Uses DWT (Daubechies-4, level 5) and cepstral analysis.

| #   | Feature Name         | Formula                            | Description                                           | Line     |
| --- | -------------------- | ---------------------------------- | ----------------------------------------------------- | -------- |
| 24  | `WaveletEnergyRatio` | Σ(detail energy) / Σ(total energy) | High-freq to total energy ratio                       | L47–64   |
| 25  | `WaveletKurtosis`    | mean(kurtosis(cD_i))               | Mean kurtosis across detail coefficients              | L67–94   |
| 26  | `WaveletEnergy_D1`   | Σ(cD1²)                            | Energy at detail level 1 (highest freq)               | L169–221 |
| 27  | `WaveletEnergy_D3`   | Σ(cD3²)                            | Energy at detail level 3 (mid freq)                   | L169–221 |
| 28  | `WaveletEnergy_D5`   | Σ(cD5²)                            | Energy at detail level 5 (low freq)                   | L169–221 |
| 29  | `CepstralPeakRatio`  | max(cepstrum) / mean(cepstrum)     | Peak-to-mean in cepstrum; detects periodic components | L97–128  |
| 30  | `QuefrencyCentroid`  | Σ(q·C(q)) / Σ(C(q))                | Centroid of cepstrum (seconds)                        | L131–166 |

### Bispectrum / Higher-Order Features (6)

Extracted by `extract_bispectrum_features()` in `bispectrum.py`. Captures phase coupling and nonlinear interactions.

| #   | Feature Name          | Formula                          | Description                                       | Line     |
| --- | --------------------- | -------------------------------- | ------------------------------------------------- | -------- |
| 31  | `BispectrumPeak`      | max(\|B(f)\|)                    | Peak bispectrum magnitude — strong phase coupling | L70–82   |
| 32  | `BispectrumMean`      | mean(\|B(f)\|)                   | Mean bispectrum value                             | L85–95   |
| 33  | `BispectrumEntropy`   | −Σ(B_norm·ln(B_norm))            | Entropy of bispectrum distribution                | L98–116  |
| 34  | `PhaseCoupling`       | tanh(ΣB / ΣP)                    | Quadratic phase coupling strength (0–1)           | L119–150 |
| 35  | `NonlinearityIndex`   | (bispec_norm + \|skewness\|) / 2 | Deviation from Gaussianity                        | L153–179 |
| 36  | `BispectrumPeakRatio` | peak / mean                      | Peak-to-mean ratio in bispectrum                  | L182–228 |

---

## Advanced Features (16)

Extracted by `extract_advanced_features()` in `advanced_features.py`. **~10× slower** than base features.

### Continuous Wavelet Transform — CWT (4)

Extracted by `extract_cwt_features()`. Uses Morlet wavelet over 127 scales.

| #   | Feature Name       | Formula               | Description                 | Line   |
| --- | ------------------ | --------------------- | --------------------------- | ------ |
| 37  | `CWT_TotalEnergy`  | Σ(energy per scale)   | Total CWT energy            | L27–69 |
| 38  | `CWT_PeakEnergy`   | max(energy per scale) | Peak CWT scale energy       | L27–69 |
| 39  | `CWT_EnergyRatio`  | peak / total          | Concentration of energy     | L27–69 |
| 40  | `CWT_DominantFreq` | freq at peak scale    | Dominant CWT frequency (Hz) | L27–69 |

### Wavelet Packet Transform — WPT (4)

Extracted by `extract_wpt_features()`. Uses db4 wavelet, level 4 (16 terminal nodes).

| #   | Feature Name        | Formula                | Description                         | Line    |
| --- | ------------------- | ---------------------- | ----------------------------------- | ------- |
| 41  | `WPT_MaxEnergy`     | max(node energies)     | Maximum node energy                 | L72–118 |
| 42  | `WPT_EnergyEntropy` | entropy(norm energies) | Distribution of energy across nodes | L72–118 |
| 43  | `WPT_EnergyStd`     | std(node energies)     | Variability of node energies        | L72–118 |
| 44  | `WPT_EnergyRatio`   | max / total            | Energy concentration                | L72–118 |

### Nonlinear Dynamics (4)

Extracted by `extract_nonlinear_features()`.

| #   | Feature Name         | Formula                    | Description                              | Line     |
| --- | -------------------- | -------------------------- | ---------------------------------------- | -------- |
| 45  | `SampleEntropy`      | −ln(φ(m+1)/φ(m))           | Complexity measure; lower = more regular | L121–160 |
| 46  | `ApproximateEntropy` | φ(m) − φ(m+1)              | Regularity measure                       | L163–187 |
| 47  | `DFA_Alpha`          | slope in log F(n) vs log n | Detrended fluctuation analysis exponent  | L190–249 |
| 48  | `HurstExponent`      | = DFA_Alpha                | Long-range correlation (= α for fBm)     | L252–290 |

### Spectrogram / STFT (4)

Computed inline in `extract_advanced_features()` using `scipy.signal.spectrogram` (nperseg=256).

| #   | Feature Name          | Formula                  | Description                            | Line     |
| --- | --------------------- | ------------------------ | -------------------------------------- | -------- |
| 49  | `Spectrogram_Entropy` | mean(entropy(Sxx[:, t])) | Mean spectral entropy over time frames | L323–340 |
| 50  | `TF_Peak`             | max(Sxx)                 | Peak value in spectrogram              | L323–340 |
| 51  | `TF_Mean`             | mean(Sxx)                | Mean spectrogram value                 | L323–340 |
| 52  | `TF_Std`              | std(Sxx)                 | Standard deviation of spectrogram      | L323–340 |

---

## Canonical Feature Order (Base 36)

The `FeatureExtractor._get_feature_names()` method (line 239–283 of `feature_extractor.py`) defines this canonical order:

```
 1. RMS                  8. DominantFreq         20. EnvelopeRMS       24. WaveletEnergyRatio  31. BispectrumPeak
 2. Kurtosis             9. SpectralCentroid     21. EnvelopeKurtosis  25. WaveletKurtosis     32. BispectrumMean
 3. Skewness            10. SpectralEntropy      22. EnvelopePeak      26. WaveletEnergy_D1    33. BispectrumEntropy
 4. CrestFactor         11. LowBandEnergy        23. ModulationFreq    27. WaveletEnergy_D3    34. PhaseCoupling
 5. ShapeFactor         12. MidBandEnergy                              28. WaveletEnergy_D5    35. NonlinearityIndex
 6. ImpulseFactor       13. HighBandEnergy                             29. CepstralPeakRatio   36. BispectrumPeakRatio
 7. ClearanceFactor     14. VeryHighBandEnergy                         30. QuefrencyCentroid
                        15. TotalSpectralPower
                        16. SpectralStd
                        17. Harmonic2X1X
                        18. Harmonic3X1X
                        19. SpectralPeakiness
```
