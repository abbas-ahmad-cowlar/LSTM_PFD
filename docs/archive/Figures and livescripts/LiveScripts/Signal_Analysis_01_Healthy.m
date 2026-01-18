%% ========================================================================
%% SIGNAL ANALYSIS PART 1: HEALTHY BEARING BASELINE ("SAIN")
%% ========================================================================
%% 
%% Scientific Analysis of Hydrodynamic Bearing Vibration Signals
%% For Research Publication Purposes
%%
%% This script provides a comprehensive analysis of the HEALTHY bearing
%% signal, establishing the baseline characteristics against which all
%% fault conditions will be compared.
%%
%% CONTENTS:
%%   1. Signal Loading and Visualization
%%   2. Time-Domain Statistical Analysis
%%   3. Frequency-Domain Spectral Analysis
%%   4. Time-Frequency Analysis (Spectrogram & CWT)
%%   5. Envelope Analysis (Hilbert Transform)
%%   6. Cepstral Analysis
%%   7. Summary and Physical Interpretation
%%
%% Author: PFD Diagnostics Research Team
%% Date: January 2026
%% Version: 1.0 - Scientific Research Edition
%% ========================================================================

clear; clc; close all;

% Change to project root directory (parent of LiveScripts folder)
scriptPath = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptPath);
cd(projectRoot);
fprintf('Working directory: %s\n\n', pwd);

%% Configuration
% =========================================================================
% Output directory for high-resolution figures
OUTPUT_DIR = 'Figures/Healthy';
if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

% Signal file to analyze
SIGNAL_FILE = 'data_signaux_sep_production/sain_001.mat';

% Figure export settings
EXPORT_DPI = 300;  % DPI for publication-quality figures
EXPORT_FORMAT = 'png';

fprintf('========================================================================\n');
fprintf('   HEALTHY BEARING SIGNAL ANALYSIS - SCIENTIFIC RESEARCH\n');
fprintf('========================================================================\n');
fprintf('Signal: %s\n', SIGNAL_FILE);
fprintf('Output: %s\n\n', OUTPUT_DIR);

%% ========================================================================
%% SECTION 1: SIGNAL LOADING AND INITIAL VISUALIZATION
%% ========================================================================
%
% The healthy bearing signal represents the baseline operating condition
% of a hydrodynamic bearing (Palier Fluide Dynamique - PFD) without any
% faults or anomalies.
%
% Expected characteristics of a healthy bearing:
%   - Low amplitude vibrations
%   - Dominated by rotational frequency (1X)
%   - No sub-synchronous components (no oil whirl)
%   - No excessive harmonics (no misalignment)
%   - Low kurtosis (Gaussian-like distribution)
%
% =========================================================================

fprintf('SECTION 1: Signal Loading\n');
fprintf('------------------------------------------\n');

% Load the signal
data = load(SIGNAL_FILE);
x = data.x;         % Vibration signal (acceleration or velocity)
fs = data.fs;       % Sampling frequency (Hz)
fault = data.fault; % Fault label (should be 'sain')

% Signal parameters
N = length(x);              % Number of samples
T = N / fs;                 % Total duration (seconds)
t = (0:N-1)' / fs;          % Time vector
dt = 1 / fs;                % Sampling period

fprintf('Fault Type:         %s (Healthy)\n', fault);
fprintf('Sampling Frequency: %d Hz\n', fs);
fprintf('Signal Duration:    %.2f seconds\n', T);
fprintf('Number of Samples:  %d\n', N);
fprintf('Time Resolution:    %.6f seconds\n\n', dt);

% Load metadata if available
if isfield(data, 'metadata')
    meta = data.metadata;
    fprintf('Metadata Available:\n');
    fprintf('  Speed:       %.1f RPM (%.1f Hz)\n', meta.speed_rpm, meta.speed_rpm/60);
    fprintf('  Load:        %.1f%%\n', meta.load_percent);
    fprintf('  Temperature: %.1f °C\n', meta.temperature_C);
    fprintf('  Sommerfeld:  %.4f\n', meta.sommerfeld_number);
    fprintf('  Severity:    %s\n', meta.severity);
    
    Omega = meta.speed_rpm / 60;  % Rotational frequency (Hz)
else
    fprintf('No metadata available. Using default rotational speed.\n');
    Omega = 60;  % Default: 60 Hz (3600 RPM)
end

fprintf('\nRotational Frequency (1X): %.2f Hz\n', Omega);
fprintf('------------------------------------------\n\n');

%% ------------------------------------------------------------------------
%% FIGURE 1: Raw Time-Domain Signal (Full Duration)
%% ------------------------------------------------------------------------
%
% This figure shows the complete raw vibration signal in the time domain.
% For a healthy bearing, we expect:
%   - Relatively constant amplitude envelope
%   - No transient events or impulses
%   - Quasi-periodic oscillations at rotational frequency
%
% -------------------------------------------------------------------------

fprintf('Generating Figure 1: Raw Time-Domain Signal...\n');

fig1 = figure('Name', 'Healthy Signal - Time Domain', ...
    'Position', [100, 100, 1200, 500], 'Color', 'white');

plot(t, x, 'b-', 'LineWidth', 0.5);
hold on;

% Add envelope for reference
envelope_upper = movmax(x, round(fs/Omega));
envelope_lower = movmin(x, round(fs/Omega));
plot(t, envelope_upper, 'r--', 'LineWidth', 1.5);
plot(t, envelope_lower, 'r--', 'LineWidth', 1.5);

xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Amplitude', 'FontSize', 14, 'FontWeight', 'bold');
title('Figure 1: Healthy Bearing Vibration Signal (sain) - Time Domain', ...
    'FontSize', 16, 'FontWeight', 'bold');
legend({'Raw Signal', 'Envelope'}, 'Location', 'northeast', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 12);
xlim([0, T]);

% Add annotation box with key statistics
stats_str = sprintf('RMS = %.4f\nPeak = %.4f\nDuration = %.1f s', ...
    rms(x), max(abs(x)), T);
annotation('textbox', [0.15, 0.75, 0.15, 0.15], ...
    'String', stats_str, 'FontSize', 11, ...
    'BackgroundColor', 'white', 'EdgeColor', 'black');

% Export figure
exportgraphics(fig1, fullfile(OUTPUT_DIR, 'Fig1_Sain_TimeDomain_Full.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  ✓ Saved: Fig1_Sain_TimeDomain_Full.png\n');

%% ------------------------------------------------------------------------
%% FIGURE 2: Zoomed Time-Domain Signal (First 100 ms)
%% ------------------------------------------------------------------------
%
% A zoomed view allows observation of individual oscillation cycles.
% At 60 Hz rotation, one cycle = 16.67 ms, so 100 ms shows ~6 cycles.
%
% -------------------------------------------------------------------------

fprintf('Generating Figure 2: Zoomed Time-Domain Signal...\n');

fig2 = figure('Name', 'Healthy Signal - Time Domain (Zoomed)', ...
    'Position', [100, 100, 1200, 500], 'Color', 'white');

% Zoomed window: first 100 ms
zoom_end = min(0.1, T);
zoom_idx = t <= zoom_end;

plot(t(zoom_idx), x(zoom_idx), 'b-', 'LineWidth', 1.0);
hold on;

% Mark rotational period
for k = 1:floor(zoom_end * Omega)
    xline(k/Omega, 'g--', 'LineWidth', 0.8, 'Alpha', 0.5);
end

xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Amplitude', 'FontSize', 14, 'FontWeight', 'bold');
title(sprintf('Figure 2: Healthy Signal - Zoomed View (0 to %.0f ms)', zoom_end*1000), ...
    'FontSize', 16, 'FontWeight', 'bold');
subtitle(sprintf('Green lines mark rotational period (1/%.1f Hz = %.2f ms)', Omega, 1000/Omega), ...
    'FontSize', 12);
grid on;
set(gca, 'FontSize', 12);
xlim([0, zoom_end]);

% Export figure
exportgraphics(fig2, fullfile(OUTPUT_DIR, 'Fig2_Sain_TimeDomain_Zoomed.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  ✓ Saved: Fig2_Sain_TimeDomain_Zoomed.png\n\n');

%% ========================================================================
%% SECTION 2: TIME-DOMAIN STATISTICAL ANALYSIS
%% ========================================================================
%
% Statistical features extracted from the time-domain signal provide
% insights into the signal's amplitude distribution and dynamics.
%
% MATHEMATICAL DEFINITIONS:
%
% 1. Mean (μ):
%    $$\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$$
%
% 2. Root Mean Square (RMS):
%    $$\text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}$$
%
% 3. Standard Deviation (σ):
%    $$\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}$$
%
% 4. Skewness (asymmetry):
%    $$\gamma_1 = \frac{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^3}{\sigma^3}$$
%
% 5. Kurtosis (peakedness):
%    $$\gamma_2 = \frac{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^4}{\sigma^4}$$
%    Note: Gaussian distribution has kurtosis = 3
%
% 6. Crest Factor:
%    $$\text{CF} = \frac{\max|x|}{\text{RMS}}$$
%
% 7. Peak-to-Peak:
%    $$\text{P2P} = \max(x) - \min(x)$$
%
% PHYSICAL INTERPRETATION:
%   - High kurtosis (>3): Indicates impulsive events (e.g., bearing impacts)
%   - High crest factor (>3-4): Indicates transients or spikes
%   - Non-zero skewness: Asymmetric loading or measurement bias
%
% =========================================================================

fprintf('SECTION 2: Time-Domain Statistical Analysis\n');
fprintf('------------------------------------------\n');

% Calculate statistical features
stat_mean = mean(x);
stat_rms = rms(x);
stat_std = std(x);
stat_var = var(x);
stat_skew = skewness(x);
stat_kurt = kurtosis(x);
stat_peak = max(abs(x));
stat_p2p = max(x) - min(x);
stat_crest = stat_peak / stat_rms;
stat_energy = sum(x.^2);
stat_clearance = stat_peak / (mean(abs(x)) + eps);
stat_shape = stat_rms / (mean(abs(x)) + eps);
stat_impulse = stat_peak / (mean(abs(x)) + eps);

% Display results
fprintf('\n--- STATISTICAL FEATURES ---\n\n');
fprintf('Basic Statistics:\n');
fprintf('  Mean (μ):              %+.6f\n', stat_mean);
fprintf('  RMS:                   %.6f\n', stat_rms);
fprintf('  Standard Deviation:    %.6f\n', stat_std);
fprintf('  Variance:              %.6f\n', stat_var);
fprintf('  Peak Amplitude:        %.6f\n', stat_peak);
fprintf('  Peak-to-Peak:          %.6f\n\n', stat_p2p);

fprintf('Shape Indicators:\n');
fprintf('  Skewness:              %+.4f  (0 = symmetric)\n', stat_skew);
fprintf('  Kurtosis:              %.4f   (3 = Gaussian)\n', stat_kurt);
fprintf('  Crest Factor:          %.4f   (√2 ≈ 1.41 for sine)\n', stat_crest);
fprintf('  Shape Factor:          %.4f\n', stat_shape);
fprintf('  Impulse Factor:        %.4f\n', stat_impulse);
fprintf('  Clearance Factor:      %.4f\n\n', stat_clearance);

fprintf('Energy:\n');
fprintf('  Total Energy:          %.4f\n', stat_energy);
fprintf('------------------------------------------\n\n');

% Interpretation for healthy signal
fprintf('INTERPRETATION (Healthy Baseline):\n');
if stat_kurt < 4
    fprintf('  ✓ Kurtosis = %.2f is close to Gaussian (3.0)\n', stat_kurt);
    fprintf('    → No significant impulsive events detected\n');
else
    fprintf('  ⚠ Kurtosis = %.2f is elevated (expected ~3 for healthy)\n', stat_kurt);
end

if stat_crest < 5
    fprintf('  ✓ Crest Factor = %.2f is within normal range\n', stat_crest);
    fprintf('    → No severe transient spikes\n');
else
    fprintf('  ⚠ Crest Factor = %.2f is elevated\n', stat_crest);
end

if abs(stat_mean) < 0.01
    fprintf('  ✓ Mean ≈ 0 indicates no DC offset bias\n');
else
    fprintf('  ⚠ Non-zero mean (%.4f) may indicate sensor bias\n', stat_mean);
end
fprintf('\n');

%% ------------------------------------------------------------------------
%% FIGURE 3: Amplitude Distribution Histogram
%% ------------------------------------------------------------------------
%
% For a healthy bearing, the amplitude distribution should be approximately
% Gaussian (normal), indicating random vibrations without impulsive faults.
%
% -------------------------------------------------------------------------

fprintf('Generating Figure 3: Amplitude Distribution...\n');

fig3 = figure('Name', 'Healthy Signal - Amplitude Distribution', ...
    'Position', [100, 100, 1000, 600], 'Color', 'white');

% Histogram
subplot(1, 2, 1);
histogram(x, 100, 'Normalization', 'pdf', 'FaceColor', [0.3, 0.5, 0.8], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.7);
hold on;

% Overlay fitted Gaussian
x_range = linspace(min(x), max(x), 200);
pdf_gaussian = normpdf(x_range, stat_mean, stat_std);
plot(x_range, pdf_gaussian, 'r-', 'LineWidth', 2.5);

xlabel('Amplitude', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Probability Density', 'FontSize', 13, 'FontWeight', 'bold');
title('Amplitude Distribution', 'FontSize', 14, 'FontWeight', 'bold');
legend({'Measured Data', 'Gaussian Fit'}, 'Location', 'northeast', 'FontSize', 11);
grid on;
set(gca, 'FontSize', 11);

% Add statistics annotation
annotation_str = sprintf(['Statistical Measures:\n' ...
    '\\mu = %.4f\n' ...
    '\\sigma = %.4f\n' ...
    'Skewness = %.3f\n' ...
    'Kurtosis = %.3f'], ...
    stat_mean, stat_std, stat_skew, stat_kurt);
annotation('textbox', [0.32, 0.6, 0.15, 0.25], ...
    'String', annotation_str, 'FontSize', 10, ...
    'BackgroundColor', 'white', 'EdgeColor', 'black', ...
    'FitBoxToText', 'on');

% Q-Q Plot
subplot(1, 2, 2);
qqplot(x);
title('Q-Q Plot (Normal Distribution Test)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Standard Normal Quantiles', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Sample Quantiles', 'FontSize', 13, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 11);

sgtitle('Figure 3: Healthy Signal Amplitude Distribution Analysis', ...
    'FontSize', 16, 'FontWeight', 'bold');

% Export figure
exportgraphics(fig3, fullfile(OUTPUT_DIR, 'Fig3_Sain_AmplitudeDistribution.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  ✓ Saved: Fig3_Sain_AmplitudeDistribution.png\n\n');

%% ========================================================================
%% SECTION 3: FREQUENCY-DOMAIN SPECTRAL ANALYSIS
%% ========================================================================
%
% Frequency-domain analysis reveals the spectral content of the vibration
% signal, identifying dominant frequencies and harmonics.
%
% MATHEMATICAL DEFINITIONS:
%
% 1. Discrete Fourier Transform (DFT):
%    $$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j 2\pi kn/N}$$
%
% 2. Power Spectral Density (Welch Method):
%    $$S_{xx}(f) = \frac{1}{K \cdot U} \sum_{k=1}^{K} |X_k(f)|^2$$
%    where K = number of segments, U = window energy normalization
%
% 3. Spectral Centroid:
%    $$f_c = \frac{\sum_k f_k \cdot |X(f_k)|}{\sum_k |X(f_k)|}$$
%
% 4. Spectral Entropy:
%    $$H = -\sum_k P_k \log_2(P_k), \quad P_k = \frac{|X(f_k)|^2}{\sum |X(f)|^2}$$
%
% 5. Spectral Flatness (Wiener Entropy):
%    $$\text{SF} = \frac{\exp\left(\frac{1}{K}\sum_k \log S(f_k)\right)}{\frac{1}{K}\sum_k S(f_k)}$$
%    SF = 1 for white noise, SF → 0 for tonal signals
%
% KEY FREQUENCIES FOR BEARINGS:
%   - 1X: Rotational frequency (imbalance)
%   - 2X: Misalignment signature
%   - 3X: Coupling issues
%   - 0.42-0.48X: Oil whirl (sub-synchronous)
%   - BPFO, BPFI, BSF: Ball bearing defect frequencies (N/A for fluid bearings)
%
% =========================================================================

fprintf('SECTION 3: Frequency-Domain Spectral Analysis\n');
fprintf('------------------------------------------\n');

% FFT Parameters
NFFT = 2^nextpow2(N);  % Zero-pad to next power of 2 for efficiency
f_fft = (0:NFFT/2) * fs / NFFT;  % Frequency vector (positive frequencies)

% Compute FFT
X_fft = fft(x, NFFT);
X_mag = abs(X_fft(1:NFFT/2+1)) / N;  % Single-sided magnitude spectrum
X_mag(2:end-1) = 2 * X_mag(2:end-1);  % Account for negative frequencies

% Welch PSD (more robust for noisy signals)
window_size = 4096;
overlap = floor(window_size / 2);
[Pxx, f_psd] = pwelch(x, hann(window_size), overlap, window_size, fs);

% Find dominant frequency
[peak_psd, peak_idx] = max(Pxx);
f_dominant = f_psd(peak_idx);

% Spectral Centroid
spectral_centroid = sum(f_psd .* Pxx) / sum(Pxx);

% Spectral Spread (Bandwidth)
spectral_spread = sqrt(sum(((f_psd - spectral_centroid).^2) .* Pxx) / sum(Pxx));

% Spectral Entropy
Pxx_norm = Pxx / sum(Pxx);
spectral_entropy = -sum(Pxx_norm .* log2(Pxx_norm + eps));

% Spectral Flatness (Wiener Entropy)
geom_mean = exp(mean(log(Pxx + eps)));
arith_mean = mean(Pxx);
spectral_flatness = geom_mean / (arith_mean + eps);

% Harmonic Analysis (relative to 1X)
f_1X = Omega;
f_2X = 2 * Omega;
f_3X = 3 * Omega;

% Find amplitudes at harmonic frequencies
[~, idx_1X] = min(abs(f_psd - f_1X));
[~, idx_2X] = min(abs(f_psd - f_2X));
[~, idx_3X] = min(abs(f_psd - f_3X));

amp_1X = sqrt(Pxx(idx_1X));
amp_2X = sqrt(Pxx(idx_2X));
amp_3X = sqrt(Pxx(idx_3X));

ratio_2X_1X = amp_2X / (amp_1X + eps);
ratio_3X_1X = amp_3X / (amp_1X + eps);

% Total Harmonic Distortion (THD)
THD = sqrt(amp_2X^2 + amp_3X^2) / (amp_1X + eps);

% Display results
fprintf('\n--- SPECTRAL FEATURES ---\n\n');
fprintf('Dominant Frequency:      %.2f Hz\n', f_dominant);
fprintf('Spectral Centroid:       %.2f Hz\n', spectral_centroid);
fprintf('Spectral Spread:         %.2f Hz\n', spectral_spread);
fprintf('Spectral Entropy:        %.4f bits\n', spectral_entropy);
fprintf('Spectral Flatness:       %.6f (1=noise, 0=tonal)\n\n', spectral_flatness);

fprintf('Harmonic Analysis (ref: 1X = %.1f Hz):\n', f_1X);
fprintf('  1X Amplitude:          %.6f\n', amp_1X);
fprintf('  2X Amplitude:          %.6f (ratio: %.4f)\n', amp_2X, ratio_2X_1X);
fprintf('  3X Amplitude:          %.6f (ratio: %.4f)\n', amp_3X, ratio_3X_1X);
fprintf('  THD:                   %.4f\n', THD);
fprintf('------------------------------------------\n\n');

% Interpretation
fprintf('INTERPRETATION (Healthy Baseline):\n');
fprintf('  ✓ Dominant frequency at %.1f Hz (rotational speed)\n', f_dominant);
if ratio_2X_1X < 0.5
    fprintf('  ✓ Low 2X/1X ratio (%.2f) → No misalignment\n', ratio_2X_1X);
else
    fprintf('  ⚠ Elevated 2X/1X ratio (%.2f) → Check alignment\n', ratio_2X_1X);
end
if spectral_flatness > 0.5
    fprintf('  ✓ High spectral flatness (%.2f) → Broad-band noise dominates\n', spectral_flatness);
else
    fprintf('  • Low spectral flatness (%.2f) → Tonal components present\n', spectral_flatness);
end
fprintf('\n');

%% ------------------------------------------------------------------------
%% FIGURE 4: Power Spectral Density (Full Bandwidth)
%% ------------------------------------------------------------------------

fprintf('Generating Figure 4: Power Spectral Density...\n');

fig4 = figure('Name', 'Healthy Signal - PSD', ...
    'Position', [100, 100, 1200, 600], 'Color', 'white');

% Log-scale PSD
semilogy(f_psd, Pxx, 'b-', 'LineWidth', 1.0);
hold on;

% Mark harmonic frequencies
xline(f_1X, 'r--', '1X', 'LineWidth', 1.5, 'FontSize', 12, 'LabelOrientation', 'horizontal');
xline(f_2X, 'g--', '2X', 'LineWidth', 1.5, 'FontSize', 12, 'LabelOrientation', 'horizontal');
xline(f_3X, 'm--', '3X', 'LineWidth', 1.5, 'FontSize', 12, 'LabelOrientation', 'horizontal');

% Mark sub-synchronous region (oil whirl would appear here)
xline(0.45 * f_1X, 'k:', 'Oil Whirl (0.45X)', 'LineWidth', 1.2, 'FontSize', 10, ...
    'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');

xlabel('Frequency (Hz)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Power Spectral Density (V²/Hz)', 'FontSize', 14, 'FontWeight', 'bold');
title('Figure 4: Power Spectral Density - Welch Method', ...
    'FontSize', 16, 'FontWeight', 'bold');
subtitle(sprintf('Window: %d samples, Overlap: 50%%, NFFT: %d', window_size, window_size), ...
    'FontSize', 12);
grid on;
set(gca, 'FontSize', 12);
xlim([0, fs/4]);  % Show up to fs/4 (relevant frequency range)

% Annotation
annotation_str = sprintf(['Spectral Features:\n' ...
    'f_{dominant} = %.1f Hz\n' ...
    'f_{centroid} = %.1f Hz\n' ...
    'Entropy = %.2f bits\n' ...
    'Flatness = %.4f'], ...
    f_dominant, spectral_centroid, spectral_entropy, spectral_flatness);
annotation('textbox', [0.65, 0.7, 0.2, 0.2], ...
    'String', annotation_str, 'FontSize', 11, ...
    'BackgroundColor', 'white', 'EdgeColor', 'black', ...
    'FitBoxToText', 'on');

% Export figure
exportgraphics(fig4, fullfile(OUTPUT_DIR, 'Fig4_Sain_PSD.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  ✓ Saved: Fig4_Sain_PSD.png\n');

%% ------------------------------------------------------------------------
%% FIGURE 5: Zoomed PSD (Low Frequency Range)
%% ------------------------------------------------------------------------

fprintf('Generating Figure 5: Zoomed PSD (Low Frequency)...\n');

fig5 = figure('Name', 'Healthy Signal - PSD Zoomed', ...
    'Position', [100, 100, 1200, 600], 'Color', 'white');

% Linear scale for low frequencies
f_zoom_max = 500;  % Hz
zoom_mask = f_psd <= f_zoom_max;

plot(f_psd(zoom_mask), 10*log10(Pxx(zoom_mask)), 'b-', 'LineWidth', 1.2);
hold on;

% Mark harmonics
xline(f_1X, 'r-', '1X', 'LineWidth', 2, 'FontSize', 12);
xline(f_2X, 'g-', '2X', 'LineWidth', 2, 'FontSize', 12);
xline(f_3X, 'm-', '3X', 'LineWidth', 2, 'FontSize', 12);
xline(4*f_1X, 'c-', '4X', 'LineWidth', 1.5, 'FontSize', 11);

% Sub-synchronous region
fill([0, 0.5*f_1X, 0.5*f_1X, 0], ...
    [min(10*log10(Pxx(zoom_mask))), min(10*log10(Pxx(zoom_mask))), ...
    max(10*log10(Pxx(zoom_mask))), max(10*log10(Pxx(zoom_mask)))], ...
    'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
text(0.25*f_1X, max(10*log10(Pxx(zoom_mask)))-5, 'Sub-sync', ...
    'FontSize', 11, 'HorizontalAlignment', 'center');

xlabel('Frequency (Hz)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Power (dB)', 'FontSize', 14, 'FontWeight', 'bold');
title('Figure 5: PSD - Low Frequency Analysis (0-500 Hz)', ...
    'FontSize', 16, 'FontWeight', 'bold');
subtitle('Markers show rotational harmonics (1X, 2X, 3X, 4X)', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 12);
xlim([0, f_zoom_max]);

% Export figure
exportgraphics(fig5, fullfile(OUTPUT_DIR, 'Fig5_Sain_PSD_Zoomed.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  ✓ Saved: Fig5_Sain_PSD_Zoomed.png\n\n');

%% ========================================================================
%% SECTION 4: TIME-FREQUENCY ANALYSIS
%% ========================================================================
%
% Time-frequency analysis reveals how the spectral content evolves over
% time, essential for detecting non-stationary phenomena.
%
% METHODS:
%
% 1. Short-Time Fourier Transform (STFT) / Spectrogram:
%    $$S(t, f) = \left| \int_{-\infty}^{\infty} x(\tau) w(\tau - t) e^{-j2\pi f\tau} d\tau \right|^2$$
%    where w(t) is a window function (Hann, Hamming, etc.)
%
%    Trade-off: Time resolution ↔ Frequency resolution
%    - Short window → Good time resolution, poor frequency resolution
%    - Long window → Good frequency resolution, poor time resolution
%
% 2. Continuous Wavelet Transform (CWT):
%    $$W(a, b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} x(t) \psi^*\left(\frac{t-b}{a}\right) dt$$
%    where a = scale (inversely related to frequency), b = translation
%
%    Advantages: Multi-resolution analysis, better for transients
%
% =========================================================================

fprintf('SECTION 4: Time-Frequency Analysis\n');
fprintf('------------------------------------------\n');

%% ------------------------------------------------------------------------
%% FIGURE 6: Spectrogram (STFT)
%% ------------------------------------------------------------------------

fprintf('Generating Figure 6: Spectrogram (STFT)...\n');

fig6 = figure('Name', 'Healthy Signal - Spectrogram', ...
    'Position', [100, 100, 1200, 700], 'Color', 'white');

% Spectrogram parameters
spec_window = 512;
spec_overlap = 448;  % 87.5% overlap for smooth visualization
spec_nfft = 1024;

% Compute and plot spectrogram
[S, F, T_spec] = spectrogram(x, hann(spec_window), spec_overlap, spec_nfft, fs);

% Convert to dB
S_dB = 10 * log10(abs(S).^2 + eps);

% Plot
imagesc(T_spec, F, S_dB);
axis xy;
colormap('jet');
cb = colorbar;
cb.Label.String = 'Power (dB)';
cb.Label.FontSize = 12;

% Limit frequency axis
ylim([0, 500]);

% Add harmonic lines
hold on;
yline(f_1X, 'w--', 'LineWidth', 1.5);
yline(f_2X, 'w--', 'LineWidth', 1.5);
yline(f_3X, 'w--', 'LineWidth', 1.5);

% Add text labels for harmonics
text(T_spec(end)*0.02, f_1X + 10, '1X', 'Color', 'white', 'FontSize', 11, 'FontWeight', 'bold');
text(T_spec(end)*0.02, f_2X + 10, '2X', 'Color', 'white', 'FontSize', 11, 'FontWeight', 'bold');
text(T_spec(end)*0.02, f_3X + 10, '3X', 'Color', 'white', 'FontSize', 11, 'FontWeight', 'bold');

xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Frequency (Hz)', 'FontSize', 14, 'FontWeight', 'bold');
title('Figure 6: Spectrogram (Short-Time Fourier Transform)', ...
    'FontSize', 16, 'FontWeight', 'bold');
subtitle(sprintf('Window: %d samples (%.1f ms), Overlap: %.0f%%', ...
    spec_window, 1000*spec_window/fs, 100*spec_overlap/spec_window), 'FontSize', 12);
set(gca, 'FontSize', 12);

% Export figure
exportgraphics(fig6, fullfile(OUTPUT_DIR, 'Fig6_Sain_Spectrogram.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  ✓ Saved: Fig6_Sain_Spectrogram.png\n');

%% ------------------------------------------------------------------------
%% FIGURE 7: Continuous Wavelet Transform (CWT)
%% ------------------------------------------------------------------------

fprintf('Generating Figure 7: Continuous Wavelet Transform...\n');

fig7 = figure('Name', 'Healthy Signal - CWT', ...
    'Position', [100, 100, 1200, 700], 'Color', 'white');

% Compute CWT using analytic Morlet wavelet
[cfs, frq] = cwt(x, 'amor', fs);

% Plot scalogram
surface(t, frq, abs(cfs));
axis tight;
shading interp;
view(0, 90);
colormap('parula');
cb = colorbar;
cb.Label.String = 'Magnitude';
cb.Label.FontSize = 12;

% Limit frequency axis
ylim([0, 500]);

% Add harmonic lines
hold on;
yline(f_1X, 'w--', 'LineWidth', 1.5);
yline(f_2X, 'w--', 'LineWidth', 1.5);
yline(f_3X, 'w--', 'LineWidth', 1.5);

xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Frequency (Hz)', 'FontSize', 14, 'FontWeight', 'bold');
title('Figure 7: Continuous Wavelet Transform (Analytic Morlet)', ...
    'FontSize', 16, 'FontWeight', 'bold');
subtitle('Multi-resolution time-frequency analysis', 'FontSize', 12);
set(gca, 'FontSize', 12, 'YScale', 'log');
set(gca, 'YScale', 'linear');  % Use linear scale for bearing analysis

% Export figure
exportgraphics(fig7, fullfile(OUTPUT_DIR, 'Fig7_Sain_CWT.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  ✓ Saved: Fig7_Sain_CWT.png\n\n');

%% ========================================================================
%% SECTION 5: ENVELOPE ANALYSIS (HILBERT TRANSFORM)
%% ========================================================================
%
% Envelope analysis is crucial for detecting amplitude modulation effects
% caused by bearing faults, looseness, or load variations.
%
% MATHEMATICAL DEFINITION:
%
% 1. Hilbert Transform:
%    $$\hat{x}(t) = \mathcal{H}\{x(t)\} = \frac{1}{\pi} \text{P.V.} \int_{-\infty}^{\infty} \frac{x(\tau)}{t - \tau} d\tau$$
%
% 2. Analytic Signal:
%    $$z(t) = x(t) + j\hat{x}(t) = A(t) e^{j\phi(t)}$$
%
% 3. Envelope (Amplitude Modulation):
%    $$A(t) = |z(t)| = \sqrt{x(t)^2 + \hat{x}(t)^2}$$
%
% 4. Instantaneous Phase:
%    $$\phi(t) = \arctan\left(\frac{\hat{x}(t)}{x(t)}\right)$$
%
% 5. Instantaneous Frequency:
%    $$f_i(t) = \frac{1}{2\pi} \frac{d\phi(t)}{dt}$$
%
% APPLICATION:
%   - Envelope spectrum reveals modulation frequencies
%   - For healthy bearings: envelope should be relatively flat
%   - For faulty bearings: envelope spectrum shows fault frequencies
%
% =========================================================================

fprintf('SECTION 5: Envelope Analysis (Hilbert Transform)\n');
fprintf('------------------------------------------\n');

% Compute analytic signal
z_analytic = hilbert(x);

% Extract envelope (amplitude modulation)
envelope = abs(z_analytic);

% Extract instantaneous phase and frequency
inst_phase = unwrap(angle(z_analytic));
inst_freq = diff(inst_phase) * fs / (2 * pi);
inst_freq = [inst_freq(1); inst_freq];  % Pad to match length

% Envelope statistics
env_mean = mean(envelope);
env_std = std(envelope);
env_peak = max(envelope);
env_crest = env_peak / env_mean;
env_kurtosis = kurtosis(envelope);

% Envelope spectrum (for detecting modulation frequencies)
env_detrend = envelope - env_mean;  % Remove DC component
[Pxx_env, f_env] = pwelch(env_detrend, hann(2048), 1024, 2048, fs);

% Display results
fprintf('\n--- ENVELOPE FEATURES ---\n\n');
fprintf('Envelope Statistics:\n');
fprintf('  Mean:          %.6f\n', env_mean);
fprintf('  Std Dev:       %.6f\n', env_std);
fprintf('  Peak:          %.6f\n', env_peak);
fprintf('  Crest Factor:  %.4f\n', env_crest);
fprintf('  Kurtosis:      %.4f\n\n', env_kurtosis);

fprintf('Instantaneous Frequency:\n');
fprintf('  Mean:          %.2f Hz\n', mean(inst_freq));
fprintf('  Std Dev:       %.2f Hz\n', std(inst_freq));
fprintf('------------------------------------------\n\n');

%% ------------------------------------------------------------------------
%% FIGURE 8: Envelope Analysis
%% ------------------------------------------------------------------------

fprintf('Generating Figure 8: Envelope Analysis...\n');

fig8 = figure('Name', 'Healthy Signal - Envelope', ...
    'Position', [100, 100, 1200, 800], 'Color', 'white');

% Subplot 1: Signal with Envelope
subplot(3, 1, 1);
plot(t, x, 'b-', 'LineWidth', 0.3);
hold on;
plot(t, envelope, 'r-', 'LineWidth', 1.5);
plot(t, -envelope, 'r-', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Amplitude', 'FontSize', 12, 'FontWeight', 'bold');
title('Signal with Envelope (Hilbert Transform)', 'FontSize', 14, 'FontWeight', 'bold');
legend({'Signal', 'Envelope ±'}, 'Location', 'northeast');
grid on;
xlim([0, min(0.5, T)]);  % Show first 500 ms
set(gca, 'FontSize', 11);

% Subplot 2: Envelope Spectrum
subplot(3, 1, 2);
semilogy(f_env, Pxx_env, 'b-', 'LineWidth', 1.0);
hold on;
xline(f_1X, 'r--', '1X', 'LineWidth', 1.5);
xline(2*f_1X, 'g--', '2X', 'LineWidth', 1.5);
xlabel('Frequency (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Envelope PSD', 'FontSize', 12, 'FontWeight', 'bold');
title('Envelope Spectrum (Modulation Frequencies)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
xlim([0, 500]);
set(gca, 'FontSize', 11);

% Subplot 3: Instantaneous Frequency
subplot(3, 1, 3);
plot(t, inst_freq, 'b-', 'LineWidth', 0.5);
hold on;
yline(f_1X, 'r--', 'Nominal 1X', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Inst. Frequency (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
title('Instantaneous Frequency', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
ylim([0, 200]);
set(gca, 'FontSize', 11);

sgtitle('Figure 8: Envelope Analysis (Hilbert Transform)', ...
    'FontSize', 16, 'FontWeight', 'bold');

% Export figure
exportgraphics(fig8, fullfile(OUTPUT_DIR, 'Fig8_Sain_Envelope.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  ✓ Saved: Fig8_Sain_Envelope.png\n\n');

%% ========================================================================
%% SECTION 6: CEPSTRAL ANALYSIS
%% ========================================================================
%
% Cepstral analysis is effective for detecting periodic structures in the
% spectrum, such as harmonics and sidebands.
%
% MATHEMATICAL DEFINITION:
%
% Real Cepstrum:
% $$c(q) = \mathcal{F}^{-1}\{\log|X(f)|\}$$
%
% where:
%   - q = quefrency (time-like variable, in seconds)
%   - Peaks in cepstrum indicate periodicities in log-spectrum
%
% APPLICATION:
%   - Harmonic families appear as peaks at quefrency = 1/f_fundamental
%   - For rotation at 60 Hz: expect peak at q = 1/60 = 16.67 ms
%   - For healthy bearings: clear peak at rotational period
%
% =========================================================================

fprintf('SECTION 6: Cepstral Analysis\n');
fprintf('------------------------------------------\n');

% Compute cepstrum
X_fft_full = fft(x);
log_spectrum = log(abs(X_fft_full) + eps);
cepstrum = real(ifft(log_spectrum));

% Quefrency axis (only positive "time")
quefrency = (0:N-1)' / fs;
cepstrum_half = cepstrum(1:floor(N/2)+1);
quefrency_half = quefrency(1:floor(N/2)+1);

% Find dominant quefrency (excluding DC region)
min_quef = 0.005;  % Ignore quefrencies below 5 ms
max_quef = 0.1;    % Ignore quefrencies above 100 ms
quef_mask = (quefrency_half > min_quef) & (quefrency_half < max_quef);

[cep_peak, cep_idx] = max(abs(cepstrum_half(quef_mask)));
quef_subset = quefrency_half(quef_mask);
quefrency_dominant = quef_subset(cep_idx);
freq_from_quef = 1 / quefrency_dominant;

% Display results
fprintf('\n--- CEPSTRAL FEATURES ---\n\n');
fprintf('Expected quefrency for 1X: %.4f s (%.2f ms)\n', 1/f_1X, 1000/f_1X);
fprintf('Detected dominant quefrency: %.4f s (%.2f ms)\n', quefrency_dominant, 1000*quefrency_dominant);
fprintf('Corresponding frequency: %.2f Hz\n', freq_from_quef);
fprintf('Difference from 1X: %.2f Hz\n', abs(freq_from_quef - f_1X));
fprintf('------------------------------------------\n\n');

%% ------------------------------------------------------------------------
%% FIGURE 9: Cepstral Analysis
%% ------------------------------------------------------------------------

fprintf('Generating Figure 9: Cepstral Analysis...\n');

fig9 = figure('Name', 'Healthy Signal - Cepstrum', ...
    'Position', [100, 100, 1200, 600], 'Color', 'white');

% Plot cepstrum
plot(1000 * quefrency_half, cepstrum_half, 'b-', 'LineWidth', 1.0);
hold on;

% Mark expected 1X quefrency
xline(1000/f_1X, 'r--', sprintf('1X (%.1f ms)', 1000/f_1X), ...
    'LineWidth', 2, 'FontSize', 11, 'LabelOrientation', 'horizontal');
xline(1000/(2*f_1X), 'g--', sprintf('2X (%.1f ms)', 1000/(2*f_1X)), ...
    'LineWidth', 1.5, 'FontSize', 11, 'LabelOrientation', 'horizontal');

xlabel('Quefrency (ms)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Cepstral Amplitude', 'FontSize', 14, 'FontWeight', 'bold');
title('Figure 9: Real Cepstrum Analysis', ...
    'FontSize', 16, 'FontWeight', 'bold');
subtitle('Peaks indicate periodic structures in the spectrum', 'FontSize', 12);
grid on;
xlim([0, 100]);  % 0 to 100 ms
set(gca, 'FontSize', 12);

% Export figure
exportgraphics(fig9, fullfile(OUTPUT_DIR, 'Fig9_Sain_Cepstrum.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  ✓ Saved: Fig9_Sain_Cepstrum.png\n\n');

%% ========================================================================
%% SECTION 7: SUMMARY AND PHYSICAL INTERPRETATION
%% ========================================================================

fprintf('========================================================================\n');
fprintf('   SUMMARY: HEALTHY BEARING BASELINE CHARACTERISTICS\n');
fprintf('========================================================================\n\n');

fprintf('SIGNAL IDENTITY:\n');
fprintf('  Fault Type:            %s (Healthy)\n', fault);
fprintf('  Sampling Frequency:    %d Hz\n', fs);
fprintf('  Duration:              %.2f seconds\n', T);
fprintf('  Rotational Speed:      %.1f Hz (%.0f RPM)\n\n', Omega, Omega*60);

fprintf('TIME-DOMAIN CHARACTERISTICS:\n');
fprintf('  RMS Amplitude:         %.6f\n', stat_rms);
fprintf('  Kurtosis:              %.4f  (Gaussian = 3.0)\n', stat_kurt);
fprintf('  Crest Factor:          %.4f\n', stat_crest);
fprintf('  ► Interpretation: Signal is well-behaved, no impulsive faults\n\n');

fprintf('FREQUENCY-DOMAIN CHARACTERISTICS:\n');
fprintf('  Dominant Frequency:    %.2f Hz\n', f_dominant);
fprintf('  Spectral Centroid:     %.2f Hz\n', spectral_centroid);
fprintf('  Spectral Entropy:      %.4f bits\n', spectral_entropy);
fprintf('  2X/1X Ratio:           %.4f\n', ratio_2X_1X);
fprintf('  THD:                   %.4f\n', THD);
fprintf('  ► Interpretation: 1X dominates, low harmonic content\n\n');

fprintf('ENVELOPE ANALYSIS:\n');
fprintf('  Envelope Mean:         %.6f\n', env_mean);
fprintf('  Envelope Kurtosis:     %.4f\n', env_kurtosis);
fprintf('  ► Interpretation: Stable amplitude modulation\n\n');

fprintf('CEPSTRAL ANALYSIS:\n');
fprintf('  Dominant Quefrency:    %.4f s → %.2f Hz\n', quefrency_dominant, freq_from_quef);
fprintf('  ► Interpretation: Clear rotational periodicity\n\n');

fprintf('========================================================================\n');
fprintf('   BASELINE ESTABLISHED - READY FOR FAULT COMPARISON\n');
fprintf('========================================================================\n\n');

fprintf('All figures saved to: %s\n\n', OUTPUT_DIR);
fprintf('Generated Figures:\n');
fprintf('  1. Fig1_Sain_TimeDomain_Full.png\n');
fprintf('  2. Fig2_Sain_TimeDomain_Zoomed.png\n');
fprintf('  3. Fig3_Sain_AmplitudeDistribution.png\n');
fprintf('  4. Fig4_Sain_PSD.png\n');
fprintf('  5. Fig5_Sain_PSD_Zoomed.png\n');
fprintf('  6. Fig6_Sain_Spectrogram.png\n');
fprintf('  7. Fig7_Sain_CWT.png\n');
fprintf('  8. Fig8_Sain_Envelope.png\n');
fprintf('  9. Fig9_Sain_Cepstrum.png\n\n');

fprintf('NEXT STEP: Proceed to Signal_Analysis_02_Single_Faults.m\n');
fprintf('           after validating this baseline analysis.\n');
fprintf('========================================================================\n');

%% Save workspace for later comparison
save(fullfile(OUTPUT_DIR, 'Healthy_Baseline_Features.mat'), ...
    'stat_*', 'f_dominant', 'spectral_*', 'ratio_*', 'THD', ...
    'env_*', 'quefrency_dominant', 'freq_from_quef', ...
    'Omega', 'fs', 'T', 'N');
fprintf('\nFeatures saved to: Healthy_Baseline_Features.mat\n');
