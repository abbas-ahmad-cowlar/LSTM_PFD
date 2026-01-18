%% ========================================================================
%% SIGNAL ANALYSIS PART 2: SINGLE FAULT SIGNALS
%% ========================================================================
%%
%% Scientific Analysis of Hydrodynamic Bearing Vibration Signals
%% For Research Publication Purposes
%%
%% This script analyzes the 7 SINGLE FAULT types and compares them
%% against the healthy baseline established in Part 1.
%%
%% FAULT TYPES ANALYZED:
%%   1. Desalignement (Misalignment) - Characterized by 2X, 3X harmonics
%%   2. Desequilibre (Imbalance) - Dominant 1X at rotational frequency
%%   3. Jeu (Bearing Clearance) - Broadband noise, multiple harmonics
%%   4. Lubrification (Lubrication Issues) - High-frequency content
%%   5. Cavitation - Random broadband noise, turbulence
%%   6. Usure (Wear) - Gradual surface degradation signature
%%   7. Oilwhirl (Oil Whirl) - Sub-synchronous 0.42-0.48X component
%%
%% Author: PFD Diagnostics Research Team
%% Date: January 2026
%% Version: 1.0 - Scientific Research Edition
%% ========================================================================

clear; clc; close all;

% Change to project root directory
scriptPath = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptPath);
cd(projectRoot);
fprintf('Working directory: %s\n\n', pwd);

%% ========================================================================
%% CONFIGURATION
%% ========================================================================

% Output directories
OUTPUT_DIR_BASE = 'Figures/Faults';
DATA_DIR = 'data_signaux_sep_production';

% Figure settings
EXPORT_DPI = 300;
EXPORT_FORMAT = 'png';

% Fault types to analyze (single faults only)
FAULT_TYPES = {
    'desalignement', 'Misalignment', '2X and 3X harmonics dominate due to angular/parallel offset';
    'desequilibre', 'Imbalance', '1X amplitude proportional to ω² (speed squared)';
    'jeu', 'Bearing Clearance', 'Multiple harmonics and sub-harmonics, possibly chaotic';
    'lubrification', 'Lubrication Issues', 'High-frequency friction, stick-slip phenomena';
    'cavitation', 'Cavitation', 'Broadband noise from bubble collapse, random impacts';
    'usure', 'Wear', 'Gradual surface degradation, increased friction harmonics';
    'oilwhirl', 'Oil Whirl', 'Sub-synchronous component at 0.42-0.48× shaft speed'
};

NUM_FAULTS = size(FAULT_TYPES, 1);

% Create output directories
for i = 1:NUM_FAULTS
    fault_dir = fullfile(OUTPUT_DIR_BASE, FAULT_TYPES{i, 1});
    if ~exist(fault_dir, 'dir')
        mkdir(fault_dir);
    end
end

fprintf('========================================================================\n');
fprintf('   SINGLE FAULT SIGNAL ANALYSIS - SCIENTIFIC RESEARCH\n');
fprintf('========================================================================\n');
fprintf('Analyzing %d fault types\n', NUM_FAULTS);
fprintf('Output: %s/<fault_name>/\n\n', OUTPUT_DIR_BASE);

%% ========================================================================
%% LOAD HEALTHY BASELINE FOR COMPARISON
%% ========================================================================

fprintf('Loading healthy baseline features...\n');
baseline_file = 'Figures/Healthy/Healthy_Baseline_Features.mat';
if exist(baseline_file, 'file')
    baseline = load(baseline_file);
    fprintf('  ✓ Baseline loaded: fs=%d Hz, Omega=%.2f Hz\n\n', baseline.fs, baseline.Omega);
else
    warning('Healthy baseline not found. Run Signal_Analysis_01_Healthy.m first.');
    baseline = struct();
    baseline.Omega = 60;  % Default
    baseline.fs = 20480;  % Default
end

%% ========================================================================
%% ANALYSIS FUNCTIONS (Reusable for each fault type)
%% ========================================================================

% Define inline helper function for feature extraction
extractFeatures = @(x, fs) struct(...
    'mean', mean(x), ...
    'rms', rms(x), ...
    'std', std(x), ...
    'skewness', skewness(x), ...
    'kurtosis', kurtosis(x), ...
    'peak', max(abs(x)), ...
    'crest', max(abs(x)) / rms(x), ...
    'p2p', max(x) - min(x));

%% ========================================================================
%% MAIN ANALYSIS LOOP - ITERATE THROUGH ALL FAULT TYPES
%% ========================================================================

% Store all results for comparative analysis
all_results = struct();

for fault_idx = 1:NUM_FAULTS
    fault_code = FAULT_TYPES{fault_idx, 1};
    fault_name = FAULT_TYPES{fault_idx, 2};
    fault_physics = FAULT_TYPES{fault_idx, 3};
    
    fprintf('========================================================================\n');
    fprintf('  FAULT %d/%d: %s (%s)\n', fault_idx, NUM_FAULTS, fault_name, fault_code);
    fprintf('========================================================================\n');
    fprintf('Physics: %s\n\n', fault_physics);
    
    output_dir = fullfile(OUTPUT_DIR_BASE, fault_code);
    
    %% --------------------------------------------------------------------
    %% SECTION 1: Load Signal
    %% --------------------------------------------------------------------
    
    signal_file = fullfile(DATA_DIR, [fault_code '_001.mat']);
    fprintf('Loading: %s\n', signal_file);
    
    if ~exist(signal_file, 'file')
        fprintf('  ⚠ Signal file not found. Skipping.\n\n');
        continue;
    end
    
    data = load(signal_file);
    x = data.x;
    fs = data.fs;
    fault = data.fault;
    N = length(x);
    T = N / fs;
    t = (0:N-1)' / fs;
    
    % Get rotational frequency from metadata or baseline
    if isfield(data, 'metadata')
        Omega = data.metadata.speed_rpm / 60;
        meta = data.metadata;
        fprintf('  Speed: %.1f RPM (%.1f Hz)\n', meta.speed_rpm, Omega);
        fprintf('  Load: %.1f%%, Temp: %.1f°C\n', meta.load_percent, meta.temperature_C);
        fprintf('  Severity: %s\n', meta.severity);
    else
        Omega = baseline.Omega;
        fprintf('  Using baseline Omega = %.1f Hz\n', Omega);
    end
    fprintf('\n');
    
    %% --------------------------------------------------------------------
    %% SECTION 2: Time-Domain Analysis
    %% --------------------------------------------------------------------
    
    fprintf('Time-Domain Analysis...\n');
    
    % Extract features
    feat = extractFeatures(x, fs);
    
    fprintf('  RMS:      %.6f\n', feat.rms);
    fprintf('  Kurtosis: %.4f\n', feat.kurtosis);
    fprintf('  Crest:    %.4f\n', feat.crest);
    
    % Store for comparison
    all_results.(fault_code).features = feat;
    all_results.(fault_code).Omega = Omega;
    
    %% --------------------------------------------------------------------
    %% FIGURE A: Time-Domain Signal (Full + Zoomed)
    %% --------------------------------------------------------------------
    
    fig_time = figure('Name', [fault_name ' - Time Domain'], ...
        'Position', [100, 100, 1200, 700], 'Color', 'white', 'Visible', 'off');
    
    % Full signal
    subplot(2, 1, 1);
    plot(t, x, 'b-', 'LineWidth', 0.5);
    hold on;
    env_upper = movmax(x, round(fs/Omega));
    env_lower = movmin(x, round(fs/Omega));
    plot(t, env_upper, 'r--', 'LineWidth', 1.2);
    plot(t, env_lower, 'r--', 'LineWidth', 1.2);
    xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Amplitude', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('%s - Full Signal (%.2f s)', fault_name, T), ...
        'FontSize', 14, 'FontWeight', 'bold');
    legend({'Signal', 'Envelope'}, 'Location', 'northeast');
    grid on;
    xlim([0, T]);
    set(gca, 'FontSize', 11);
    
    % Zoomed view (first 100 ms)
    subplot(2, 1, 2);
    zoom_end = min(0.1, T);
    zoom_idx = t <= zoom_end;
    plot(t(zoom_idx), x(zoom_idx), 'b-', 'LineWidth', 0.8);
    hold on;
    for k = 1:floor(zoom_end * Omega)
        xline(k/Omega, 'g--', 'LineWidth', 0.7, 'Alpha', 0.5);
    end
    xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Amplitude', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('Zoomed (0-100 ms) - Green: 1X Period (%.2f ms)', 1000/Omega), ...
        'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    xlim([0, zoom_end]);
    set(gca, 'FontSize', 11);
    
    % Add annotation with key stats
    annotation('textbox', [0.75, 0.15, 0.2, 0.12], ...
        'String', sprintf('RMS = %.4f\nKurt = %.2f\nCrest = %.2f', ...
            feat.rms, feat.kurtosis, feat.crest), ...
        'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');
    
    sgtitle(sprintf('Figure A: %s (%s) - Time Domain', fault_name, fault_code), ...
        'FontSize', 16, 'FontWeight', 'bold');
    
    exportgraphics(fig_time, fullfile(output_dir, sprintf('FigA_%s_TimeDomain.png', fault_code)), ...
        'Resolution', EXPORT_DPI);
    close(fig_time);
    fprintf('  ✓ FigA_%s_TimeDomain.png\n', fault_code);
    
    %% --------------------------------------------------------------------
    %% SECTION 3: Frequency-Domain Analysis
    %% --------------------------------------------------------------------
    
    fprintf('Frequency-Domain Analysis...\n');
    
    % Welch PSD
    window_size = 4096;
    overlap = floor(window_size / 2);
    [Pxx, f_psd] = pwelch(x, hann(window_size), overlap, window_size, fs);
    
    % Find dominant frequency
    [~, peak_idx] = max(Pxx);
    f_dominant = f_psd(peak_idx);
    
    % Harmonic analysis
    f_1X = Omega;
    f_2X = 2 * Omega;
    f_3X = 3 * Omega;
    f_subsync = 0.45 * Omega;  % Oil whirl frequency
    
    [~, idx_1X] = min(abs(f_psd - f_1X));
    [~, idx_2X] = min(abs(f_psd - f_2X));
    [~, idx_3X] = min(abs(f_psd - f_3X));
    [~, idx_sub] = min(abs(f_psd - f_subsync));
    
    amp_1X = sqrt(Pxx(idx_1X));
    amp_2X = sqrt(Pxx(idx_2X));
    amp_3X = sqrt(Pxx(idx_3X));
    amp_sub = sqrt(Pxx(idx_sub));
    
    ratio_2X_1X = amp_2X / (amp_1X + eps);
    ratio_3X_1X = amp_3X / (amp_1X + eps);
    ratio_sub_1X = amp_sub / (amp_1X + eps);
    
    fprintf('  Dominant: %.1f Hz\n', f_dominant);
    fprintf('  1X: %.6f, 2X/1X: %.3f, 3X/1X: %.3f\n', amp_1X, ratio_2X_1X, ratio_3X_1X);
    fprintf('  Sub-sync (0.45X): %.6f, Ratio: %.3f\n', amp_sub, ratio_sub_1X);
    
    % Store spectral features
    all_results.(fault_code).f_dominant = f_dominant;
    all_results.(fault_code).ratio_2X_1X = ratio_2X_1X;
    all_results.(fault_code).ratio_3X_1X = ratio_3X_1X;
    all_results.(fault_code).ratio_sub_1X = ratio_sub_1X;
    
    %% --------------------------------------------------------------------
    %% FIGURE B: Power Spectral Density
    %% --------------------------------------------------------------------
    
    fig_psd = figure('Name', [fault_name ' - PSD'], ...
        'Position', [100, 100, 1200, 600], 'Color', 'white', 'Visible', 'off');
    
    % Full PSD
    subplot(2, 1, 1);
    semilogy(f_psd, Pxx, 'b-', 'LineWidth', 0.8);
    hold on;
    xline(f_1X, 'r--', '1X', 'LineWidth', 1.5, 'FontSize', 10);
    xline(f_2X, 'g--', '2X', 'LineWidth', 1.5, 'FontSize', 10);
    xline(f_3X, 'm--', '3X', 'LineWidth', 1.5, 'FontSize', 10);
    xline(f_subsync, 'k:', '0.45X', 'LineWidth', 1.2, 'FontSize', 9);
    xlabel('Frequency (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('PSD (V²/Hz)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('%s - Power Spectral Density', fault_name), ...
        'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    xlim([0, fs/4]);
    set(gca, 'FontSize', 11);
    
    % Zoomed PSD (0-500 Hz)
    subplot(2, 1, 2);
    f_zoom_max = 500;
    zoom_mask = f_psd <= f_zoom_max;
    plot(f_psd(zoom_mask), 10*log10(Pxx(zoom_mask)), 'b-', 'LineWidth', 1.0);
    hold on;
    xline(f_1X, 'r-', '1X', 'LineWidth', 2, 'FontSize', 11);
    xline(f_2X, 'g-', '2X', 'LineWidth', 2, 'FontSize', 11);
    xline(f_3X, 'm-', '3X', 'LineWidth', 2, 'FontSize', 11);
    if strcmp(fault_code, 'oilwhirl')
        xline(f_subsync, 'c-', 'Oil Whirl', 'LineWidth', 2, 'FontSize', 11);
    end
    xlabel('Frequency (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Power (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Low Frequency Detail (0-500 Hz)', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    xlim([0, f_zoom_max]);
    set(gca, 'FontSize', 11);
    
    % Annotation with harmonic ratios
    annotation('textbox', [0.72, 0.55, 0.18, 0.12], ...
        'String', sprintf('2X/1X = %.3f\n3X/1X = %.3f\nSub/1X = %.3f', ...
            ratio_2X_1X, ratio_3X_1X, ratio_sub_1X), ...
        'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');
    
    sgtitle(sprintf('Figure B: %s - Frequency Analysis', fault_name), ...
        'FontSize', 16, 'FontWeight', 'bold');
    
    exportgraphics(fig_psd, fullfile(output_dir, sprintf('FigB_%s_PSD.png', fault_code)), ...
        'Resolution', EXPORT_DPI);
    close(fig_psd);
    fprintf('  ✓ FigB_%s_PSD.png\n', fault_code);
    
    %% --------------------------------------------------------------------
    %% FIGURE C: Spectrogram (Time-Frequency)
    %% --------------------------------------------------------------------
    
    fprintf('Time-Frequency Analysis...\n');
    
    fig_spec = figure('Name', [fault_name ' - Spectrogram'], ...
        'Position', [100, 100, 1200, 600], 'Color', 'white', 'Visible', 'off');
    
    spec_window = 512;
    spec_overlap = 448;
    spec_nfft = 1024;
    
    [S, F, T_spec] = spectrogram(x, hann(spec_window), spec_overlap, spec_nfft, fs);
    S_dB = 10 * log10(abs(S).^2 + eps);
    
    imagesc(T_spec, F, S_dB);
    axis xy;
    colormap('jet');
    cb = colorbar;
    cb.Label.String = 'Power (dB)';
    cb.Label.FontSize = 11;
    ylim([0, 500]);
    
    hold on;
    yline(f_1X, 'w--', '1X', 'LineWidth', 1.5, 'FontSize', 10);
    yline(f_2X, 'w--', '2X', 'LineWidth', 1.5, 'FontSize', 10);
    yline(f_3X, 'w--', '3X', 'LineWidth', 1.5, 'FontSize', 10);
    if strcmp(fault_code, 'oilwhirl')
        yline(f_subsync, 'w:', 'Oil Whirl', 'LineWidth', 1.5, 'FontSize', 10);
    end
    
    xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Frequency (Hz)', 'FontSize', 14, 'FontWeight', 'bold');
    title(sprintf('Figure C: %s - Spectrogram (STFT)', fault_name), ...
        'FontSize', 16, 'FontWeight', 'bold');
    set(gca, 'FontSize', 12);
    
    exportgraphics(fig_spec, fullfile(output_dir, sprintf('FigC_%s_Spectrogram.png', fault_code)), ...
        'Resolution', EXPORT_DPI);
    close(fig_spec);
    fprintf('  ✓ FigC_%s_Spectrogram.png\n', fault_code);
    
    %% --------------------------------------------------------------------
    %% FIGURE D: Envelope Analysis
    %% --------------------------------------------------------------------
    
    fprintf('Envelope Analysis...\n');
    
    fig_env = figure('Name', [fault_name ' - Envelope'], ...
        'Position', [100, 100, 1200, 600], 'Color', 'white', 'Visible', 'off');
    
    % Compute envelope
    z_analytic = hilbert(x);
    envelope = abs(z_analytic);
    
    % Envelope spectrum
    env_detrend = envelope - mean(envelope);
    [Pxx_env, f_env] = pwelch(env_detrend, hann(2048), 1024, 2048, fs);
    
    % Signal with envelope (zoomed)
    subplot(2, 1, 1);
    zoom_samples = min(round(0.3*fs), N);  % 300 ms
    t_zoom = t(1:zoom_samples);
    plot(t_zoom, x(1:zoom_samples), 'b-', 'LineWidth', 0.3);
    hold on;
    plot(t_zoom, envelope(1:zoom_samples), 'r-', 'LineWidth', 1.5);
    plot(t_zoom, -envelope(1:zoom_samples), 'r-', 'LineWidth', 1.5);
    xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Amplitude', 'FontSize', 12, 'FontWeight', 'bold');
    title('Signal with Envelope (First 300 ms)', 'FontSize', 14, 'FontWeight', 'bold');
    legend({'Signal', 'Envelope'}, 'Location', 'northeast');
    grid on;
    set(gca, 'FontSize', 11);
    
    % Envelope spectrum
    subplot(2, 1, 2);
    semilogy(f_env, Pxx_env, 'b-', 'LineWidth', 1.0);
    hold on;
    xline(f_1X, 'r--', '1X', 'LineWidth', 1.5);
    xline(f_2X, 'g--', '2X', 'LineWidth', 1.5);
    xlabel('Frequency (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Envelope PSD', 'FontSize', 12, 'FontWeight', 'bold');
    title('Envelope Spectrum (Modulation Frequencies)', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    xlim([0, 500]);
    set(gca, 'FontSize', 11);
    
    sgtitle(sprintf('Figure D: %s - Envelope Analysis', fault_name), ...
        'FontSize', 16, 'FontWeight', 'bold');
    
    exportgraphics(fig_env, fullfile(output_dir, sprintf('FigD_%s_Envelope.png', fault_code)), ...
        'Resolution', EXPORT_DPI);
    close(fig_env);
    fprintf('  ✓ FigD_%s_Envelope.png\n', fault_code);
    
    %% --------------------------------------------------------------------
    %% PHYSICAL INTERPRETATION
    %% --------------------------------------------------------------------
    
    fprintf('\nPhysical Interpretation for %s:\n', fault_name);
    
    % Interpret based on fault type
    switch fault_code
        case 'desalignement'
            fprintf('  → Expected: High 2X and 3X harmonics due to angular/parallel misalignment\n');
            fprintf('  → Measured: 2X/1X = %.3f, 3X/1X = %.3f\n', ratio_2X_1X, ratio_3X_1X);
            if ratio_2X_1X > 0.3
                fprintf('  ✓ CONFIRMED: Elevated 2X component detected\n');
            end
            
        case 'desequilibre'
            fprintf('  → Expected: Dominant 1X component (proportional to ω²)\n');
            fprintf('  → Measured: 1X amplitude = %.6f\n', amp_1X);
            if ratio_2X_1X < 0.3 && ratio_3X_1X < 0.3
                fprintf('  ✓ CONFIRMED: 1X dominates with low harmonic content\n');
            end
            
        case 'jeu'
            fprintf('  → Expected: Multiple harmonics, sub-harmonics, broadband noise\n');
            fprintf('  → Kurtosis = %.2f (high indicates impulsive content)\n', feat.kurtosis);
            
        case 'lubrification'
            fprintf('  → Expected: High-frequency content, stick-slip friction\n');
            fprintf('  → Check high-frequency PSD for elevated energy\n');
            
        case 'cavitation'
            fprintf('  → Expected: Broadband random noise from bubble collapse\n');
            fprintf('  → Spectral flatness indicates noise-like behavior\n');
            
        case 'usure'
            fprintf('  → Expected: Gradual degradation, increased friction\n');
            fprintf('  → Compare with healthy baseline for RMS increase\n');
            
        case 'oilwhirl'
            fprintf('  → Expected: Sub-synchronous component at 0.42-0.48× shaft speed\n');
            fprintf('  → Measured: Sub/1X ratio = %.3f at %.1f Hz\n', ratio_sub_1X, f_subsync);
            if ratio_sub_1X > 0.1
                fprintf('  ✓ CONFIRMED: Sub-synchronous component detected\n');
            end
    end
    
    fprintf('\n');
end

%% ========================================================================
%% SECTION 4: COMPARATIVE SUMMARY
%% ========================================================================

fprintf('========================================================================\n');
fprintf('   COMPARATIVE ANALYSIS: ALL SINGLE FAULTS\n');
fprintf('========================================================================\n\n');

% Create comparison table
fault_names = fieldnames(all_results);
num_analyzed = length(fault_names);

fprintf('%-15s %10s %10s %10s %10s %10s\n', ...
    'Fault', 'RMS', 'Kurtosis', 'Crest', '2X/1X', 'Sub/1X');
fprintf('%s\n', repmat('-', 1, 70));

for i = 1:num_analyzed
    fname = fault_names{i};
    r = all_results.(fname);
    fprintf('%-15s %10.4f %10.2f %10.2f %10.3f %10.3f\n', ...
        fname, r.features.rms, r.features.kurtosis, r.features.crest, ...
        r.ratio_2X_1X, r.ratio_sub_1X);
end

%% ========================================================================
%% FIGURE E: Comparative Bar Charts
%% ========================================================================

fprintf('\nGenerating comparative figures...\n');

fig_comp = figure('Name', 'Comparative Analysis', ...
    'Position', [100, 100, 1400, 800], 'Color', 'white', 'Visible', 'off');

% Extract data for plotting
rms_vals = zeros(1, num_analyzed);
kurt_vals = zeros(1, num_analyzed);
crest_vals = zeros(1, num_analyzed);
ratio_2X_vals = zeros(1, num_analyzed);
ratio_sub_vals = zeros(1, num_analyzed);

for i = 1:num_analyzed
    fname = fault_names{i};
    rms_vals(i) = all_results.(fname).features.rms;
    kurt_vals(i) = all_results.(fname).features.kurtosis;
    crest_vals(i) = all_results.(fname).features.crest;
    ratio_2X_vals(i) = all_results.(fname).ratio_2X_1X;
    ratio_sub_vals(i) = all_results.(fname).ratio_sub_1X;
end

% Subplot 1: RMS
subplot(2, 3, 1);
bar(rms_vals, 'FaceColor', [0.3, 0.5, 0.8]);
set(gca, 'XTickLabel', fault_names, 'XTickLabelRotation', 45);
ylabel('RMS Amplitude', 'FontWeight', 'bold');
title('RMS Comparison', 'FontWeight', 'bold');
grid on;

% Subplot 2: Kurtosis
subplot(2, 3, 2);
bar(kurt_vals, 'FaceColor', [0.8, 0.4, 0.3]);
hold on;
yline(3, 'k--', 'Gaussian', 'LineWidth', 1.5);
set(gca, 'XTickLabel', fault_names, 'XTickLabelRotation', 45);
ylabel('Kurtosis', 'FontWeight', 'bold');
title('Kurtosis Comparison', 'FontWeight', 'bold');
grid on;

% Subplot 3: Crest Factor
subplot(2, 3, 3);
bar(crest_vals, 'FaceColor', [0.4, 0.7, 0.4]);
set(gca, 'XTickLabel', fault_names, 'XTickLabelRotation', 45);
ylabel('Crest Factor', 'FontWeight', 'bold');
title('Crest Factor Comparison', 'FontWeight', 'bold');
grid on;

% Subplot 4: 2X/1X Ratio (Misalignment indicator)
subplot(2, 3, 4);
bar(ratio_2X_vals, 'FaceColor', [0.7, 0.5, 0.8]);
set(gca, 'XTickLabel', fault_names, 'XTickLabelRotation', 45);
ylabel('2X/1X Ratio', 'FontWeight', 'bold');
title('2X/1X Ratio (Misalignment)', 'FontWeight', 'bold');
grid on;

% Subplot 5: Sub-synchronous Ratio (Oil whirl indicator)
subplot(2, 3, 5);
bar(ratio_sub_vals, 'FaceColor', [0.9, 0.6, 0.3]);
set(gca, 'XTickLabel', fault_names, 'XTickLabelRotation', 45);
ylabel('Sub/1X Ratio', 'FontWeight', 'bold');
title('Sub-sync/1X Ratio (Oil Whirl)', 'FontWeight', 'bold');
grid on;

% Subplot 6: Radar/Spider chart placeholder (using grouped bar)
subplot(2, 3, 6);
% Normalize all metrics for comparison
norm_data = [rms_vals/max(rms_vals); ...
             kurt_vals/max(kurt_vals); ...
             crest_vals/max(crest_vals); ...
             ratio_2X_vals/max(ratio_2X_vals+eps); ...
             ratio_sub_vals/max(ratio_sub_vals+eps)]';
bar(norm_data);
set(gca, 'XTickLabel', fault_names, 'XTickLabelRotation', 45);
legend({'RMS', 'Kurt', 'Crest', '2X/1X', 'Sub/1X'}, 'Location', 'bestoutside');
ylabel('Normalized Value', 'FontWeight', 'bold');
title('Multi-Feature Comparison', 'FontWeight', 'bold');
grid on;

sgtitle('Figure E: Comparative Analysis - All Single Faults', ...
    'FontSize', 16, 'FontWeight', 'bold');

exportgraphics(fig_comp, fullfile(OUTPUT_DIR_BASE, 'FigE_Comparative_SingleFaults.png'), ...
    'Resolution', EXPORT_DPI);
close(fig_comp);
fprintf('  ✓ FigE_Comparative_SingleFaults.png\n');

%% ========================================================================
%% SAVE RESULTS
%% ========================================================================

save(fullfile(OUTPUT_DIR_BASE, 'SingleFaults_Analysis_Results.mat'), 'all_results');
fprintf('\n✓ Results saved to: SingleFaults_Analysis_Results.mat\n');

%% ========================================================================
%% SUMMARY
%% ========================================================================

fprintf('\n========================================================================\n');
fprintf('   SINGLE FAULT ANALYSIS COMPLETE\n');
fprintf('========================================================================\n\n');

fprintf('Generated Figures:\n');
for i = 1:NUM_FAULTS
    fprintf('  %s/:\n', FAULT_TYPES{i, 1});
    fprintf('    - FigA: Time Domain\n');
    fprintf('    - FigB: PSD\n');
    fprintf('    - FigC: Spectrogram\n');
    fprintf('    - FigD: Envelope\n');
end

fprintf('\n  Comparative:\n');
fprintf('    - FigE_Comparative_SingleFaults.png\n');

fprintf('\nNEXT STEP: Run Signal_Analysis_03_Mixed_Faults.m\n');
fprintf('========================================================================\n');
