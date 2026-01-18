%% ========================================================================
%% SIGNAL ANALYSIS PART 3: MIXED FAULT SIGNALS
%% ========================================================================
%%
%% Scientific Analysis of Hydrodynamic Bearing Vibration Signals
%% For Research Publication Purposes
%%
%% This script analyzes the 3 MIXED FAULT combinations where multiple
%% fault signatures overlap and interact. Mixed faults are more challenging
%% to diagnose and represent real-world scenarios.
%%
%% MIXED FAULT TYPES ANALYZED:
%%   1. Mixed Misalign + Imbalance - Combined 1X and 2X/3X signatures
%%   2. Mixed Wear + Lube - Friction and lubrication degradation
%%   3. Mixed Cavit + Jeu - Cavitation with mechanical looseness
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

OUTPUT_DIR_BASE = 'Figures/Mixed';
DATA_DIR = 'data_signaux_sep_production';
EXPORT_DPI = 300;

% Mixed fault types
MIXED_FAULTS = {
    'mixed_misalign_imbalance', 'Misalignment + Imbalance', ...
        'Combined angular offset (2X, 3X) with mass unbalance (1X)';
    'mixed_wear_lube', 'Wear + Lubrication', ...
        'Surface degradation combined with lubricant breakdown';
    'mixed_cavit_jeu', 'Cavitation + Clearance', ...
        'Bubble collapse noise overlaid on mechanical looseness'
};

NUM_MIXED = size(MIXED_FAULTS, 1);

% Create output directories
if ~exist(OUTPUT_DIR_BASE, 'dir')
    mkdir(OUTPUT_DIR_BASE);
end
for i = 1:NUM_MIXED
    fault_dir = fullfile(OUTPUT_DIR_BASE, MIXED_FAULTS{i, 1});
    if ~exist(fault_dir, 'dir')
        mkdir(fault_dir);
    end
end

fprintf('========================================================================\n');
fprintf('   MIXED FAULT SIGNAL ANALYSIS - SCIENTIFIC RESEARCH\n');
fprintf('========================================================================\n');
fprintf('Analyzing %d mixed fault types\n\n', NUM_MIXED);

%% ========================================================================
%% LOAD BASELINES FOR COMPARISON
%% ========================================================================

fprintf('Loading baseline features...\n');

% Load healthy baseline
healthy_file = 'Figures/Healthy/Healthy_Baseline_Features.mat';
if exist(healthy_file, 'file')
    healthy = load(healthy_file);
    fprintf('  ✓ Healthy baseline loaded\n');
else
    healthy = struct('stat_rms', 0.05, 'Omega', 60);
    fprintf('  ⚠ Healthy baseline not found\n');
end

% Load single fault results
single_file = 'Figures/Faults/SingleFaults_Analysis_Results.mat';
if exist(single_file, 'file')
    single_faults = load(single_file);
    fprintf('  ✓ Single fault results loaded\n\n');
else
    single_faults = struct();
    fprintf('  ⚠ Single fault results not found\n\n');
end

%% ========================================================================
%% MAIN ANALYSIS LOOP
%% ========================================================================

all_mixed_results = struct();

for mix_idx = 1:NUM_MIXED
    fault_code = MIXED_FAULTS{mix_idx, 1};
    fault_name = MIXED_FAULTS{mix_idx, 2};
    fault_physics = MIXED_FAULTS{mix_idx, 3};
    
    fprintf('========================================================================\n');
    fprintf('  MIXED FAULT %d/%d: %s\n', mix_idx, NUM_MIXED, fault_name);
    fprintf('========================================================================\n');
    fprintf('Code: %s\n', fault_code);
    fprintf('Physics: %s\n\n', fault_physics);
    
    output_dir = fullfile(OUTPUT_DIR_BASE, fault_code);
    
    %% Load Signal
    signal_file = fullfile(DATA_DIR, [fault_code '_001.mat']);
    if ~exist(signal_file, 'file')
        fprintf('  ⚠ Signal not found. Skipping.\n\n');
        continue;
    end
    
    data = load(signal_file);
    x = data.x;
    fs = data.fs;
    N = length(x);
    T = N / fs;
    t = (0:N-1)' / fs;
    
    if isfield(data, 'metadata')
        Omega = data.metadata.speed_rpm / 60;
        meta = data.metadata;
        fprintf('  Speed: %.1f Hz, Load: %.1f%%, Temp: %.1f°C\n', ...
            Omega, meta.load_percent, meta.temperature_C);
    else
        Omega = healthy.Omega;
    end
    
    %% Time-Domain Features
    fprintf('\nTime-Domain Analysis...\n');
    feat = struct();
    feat.mean = mean(x);
    feat.rms = rms(x);
    feat.std = std(x);
    feat.skewness = skewness(x);
    feat.kurtosis = kurtosis(x);
    feat.peak = max(abs(x));
    feat.crest = feat.peak / feat.rms;
    
    fprintf('  RMS: %.6f, Kurtosis: %.2f, Crest: %.2f\n', ...
        feat.rms, feat.kurtosis, feat.crest);
    
    all_mixed_results.(fault_code).features = feat;
    all_mixed_results.(fault_code).Omega = Omega;
    
    %% FIGURE A: Time-Domain Analysis
    fig_time = figure('Name', [fault_name ' - Time'], ...
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
    title(sprintf('%s - Full Signal', fault_name), 'FontSize', 14, 'FontWeight', 'bold');
    legend({'Signal', 'Envelope'}, 'Location', 'northeast');
    grid on;
    xlim([0, T]);
    
    % Zoomed
    subplot(2, 1, 2);
    zoom_end = min(0.15, T);
    zoom_idx = t <= zoom_end;
    plot(t(zoom_idx), x(zoom_idx), 'b-', 'LineWidth', 0.8);
    xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Amplitude', 'FontSize', 12, 'FontWeight', 'bold');
    title('Zoomed View (0-150 ms)', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    xlim([0, zoom_end]);
    
    annotation('textbox', [0.72, 0.15, 0.18, 0.10], ...
        'String', sprintf('RMS = %.4f\nKurt = %.2f', feat.rms, feat.kurtosis), ...
        'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');
    
    sgtitle(sprintf('Figure A: %s - Time Domain', fault_name), ...
        'FontSize', 16, 'FontWeight', 'bold');
    
    exportgraphics(fig_time, fullfile(output_dir, sprintf('FigA_%s_Time.png', fault_code)), ...
        'Resolution', EXPORT_DPI);
    close(fig_time);
    fprintf('  ✓ FigA saved\n');
    
    %% Frequency-Domain Analysis
    fprintf('Frequency-Domain Analysis...\n');
    
    [Pxx, f_psd] = pwelch(x, hann(4096), 2048, 4096, fs);
    
    f_1X = Omega;
    f_2X = 2 * Omega;
    f_3X = 3 * Omega;
    f_sub = 0.45 * Omega;
    
    [~, idx_1X] = min(abs(f_psd - f_1X));
    [~, idx_2X] = min(abs(f_psd - f_2X));
    [~, idx_3X] = min(abs(f_psd - f_3X));
    [~, idx_sub] = min(abs(f_psd - f_sub));
    
    amp_1X = sqrt(Pxx(idx_1X));
    amp_2X = sqrt(Pxx(idx_2X));
    amp_3X = sqrt(Pxx(idx_3X));
    amp_sub = sqrt(Pxx(idx_sub));
    
    ratio_2X = amp_2X / (amp_1X + eps);
    ratio_3X = amp_3X / (amp_1X + eps);
    ratio_sub = amp_sub / (amp_1X + eps);
    
    all_mixed_results.(fault_code).ratio_2X = ratio_2X;
    all_mixed_results.(fault_code).ratio_3X = ratio_3X;
    all_mixed_results.(fault_code).ratio_sub = ratio_sub;
    
    fprintf('  1X: %.6f, 2X/1X: %.3f, 3X/1X: %.3f, Sub/1X: %.3f\n', ...
        amp_1X, ratio_2X, ratio_3X, ratio_sub);
    
    %% FIGURE B: PSD
    fig_psd = figure('Name', [fault_name ' - PSD'], ...
        'Position', [100, 100, 1200, 600], 'Color', 'white', 'Visible', 'off');
    
    subplot(2, 1, 1);
    semilogy(f_psd, Pxx, 'b-', 'LineWidth', 0.8);
    hold on;
    xline(f_1X, 'r--', '1X', 'LineWidth', 1.5);
    xline(f_2X, 'g--', '2X', 'LineWidth', 1.5);
    xline(f_3X, 'm--', '3X', 'LineWidth', 1.5);
    xlabel('Frequency (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('PSD', 'FontSize', 12, 'FontWeight', 'bold');
    title('Full Spectrum', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    xlim([0, fs/4]);
    
    subplot(2, 1, 2);
    zoom_mask = f_psd <= 500;
    plot(f_psd(zoom_mask), 10*log10(Pxx(zoom_mask)), 'b-', 'LineWidth', 1.0);
    hold on;
    xline(f_1X, 'r-', '1X', 'LineWidth', 2);
    xline(f_2X, 'g-', '2X', 'LineWidth', 2);
    xline(f_3X, 'm-', '3X', 'LineWidth', 2);
    xlabel('Frequency (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Power (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Low Frequency Detail', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    xlim([0, 500]);
    
    sgtitle(sprintf('Figure B: %s - PSD', fault_name), ...
        'FontSize', 16, 'FontWeight', 'bold');
    
    exportgraphics(fig_psd, fullfile(output_dir, sprintf('FigB_%s_PSD.png', fault_code)), ...
        'Resolution', EXPORT_DPI);
    close(fig_psd);
    fprintf('  ✓ FigB saved\n');
    
    %% FIGURE C: Spectrogram
    fprintf('Time-Frequency Analysis...\n');
    
    fig_spec = figure('Name', [fault_name ' - Spectrogram'], ...
        'Position', [100, 100, 1200, 600], 'Color', 'white', 'Visible', 'off');
    
    [S, F, T_spec] = spectrogram(x, hann(512), 448, 1024, fs);
    S_dB = 10 * log10(abs(S).^2 + eps);
    
    imagesc(T_spec, F, S_dB);
    axis xy;
    colormap('jet');
    colorbar;
    ylim([0, 500]);
    hold on;
    yline(f_1X, 'w--', 'LineWidth', 1.5);
    yline(f_2X, 'w--', 'LineWidth', 1.5);
    
    xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Frequency (Hz)', 'FontSize', 14, 'FontWeight', 'bold');
    title(sprintf('Figure C: %s - Spectrogram', fault_name), ...
        'FontSize', 16, 'FontWeight', 'bold');
    
    exportgraphics(fig_spec, fullfile(output_dir, sprintf('FigC_%s_Spectrogram.png', fault_code)), ...
        'Resolution', EXPORT_DPI);
    close(fig_spec);
    fprintf('  ✓ FigC saved\n');
    
    %% FIGURE D: Fault Decomposition Concept
    % Show how the mixed signal might be decomposed into component faults
    fprintf('Fault Decomposition...\n');
    
    fig_decomp = figure('Name', [fault_name ' - Decomposition'], ...
        'Position', [100, 100, 1200, 800], 'Color', 'white', 'Visible', 'off');
    
    % Original signal with envelope
    subplot(3, 1, 1);
    z_analytic = hilbert(x);
    envelope = abs(z_analytic);
    zoom_samples = min(round(0.3*fs), N);
    plot(t(1:zoom_samples), x(1:zoom_samples), 'b-', 'LineWidth', 0.3);
    hold on;
    plot(t(1:zoom_samples), envelope(1:zoom_samples), 'r-', 'LineWidth', 1.5);
    xlabel('Time (s)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Amplitude', 'FontSize', 11, 'FontWeight', 'bold');
    title('Mixed Signal with Envelope', 'FontSize', 13, 'FontWeight', 'bold');
    legend({'Signal', 'Envelope'}, 'Location', 'northeast');
    grid on;
    
    % Envelope spectrum (modulation)
    subplot(3, 1, 2);
    env_detrend = envelope - mean(envelope);
    [Pxx_env, f_env] = pwelch(env_detrend, hann(2048), 1024, 2048, fs);
    semilogy(f_env, Pxx_env, 'b-', 'LineWidth', 1.0);
    hold on;
    xline(f_1X, 'r--', '1X', 'LineWidth', 1.5);
    xline(f_2X, 'g--', '2X', 'LineWidth', 1.5);
    xlabel('Frequency (Hz)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Envelope PSD', 'FontSize', 11, 'FontWeight', 'bold');
    title('Envelope Spectrum (Amplitude Modulation)', 'FontSize', 13, 'FontWeight', 'bold');
    grid on;
    xlim([0, 300]);
    
    % Harmonic contribution bar chart
    subplot(3, 1, 3);
    harmonics = [amp_1X, amp_2X, amp_3X, amp_sub];
    bar_colors = [0.8, 0.3, 0.3; 0.3, 0.7, 0.3; 0.3, 0.3, 0.8; 0.8, 0.6, 0.2];
    b = bar(harmonics);
    b.FaceColor = 'flat';
    b.CData = bar_colors;
    set(gca, 'XTickLabel', {'1X', '2X', '3X', '0.45X'}, 'FontSize', 11);
    ylabel('Amplitude', 'FontSize', 11, 'FontWeight', 'bold');
    title('Harmonic Component Amplitudes', 'FontSize', 13, 'FontWeight', 'bold');
    grid on;
    
    % Add interpretation text
    switch fault_code
        case 'mixed_misalign_imbalance'
            interp_text = {'Interpretation:', ...
                '• 1X: Imbalance contribution', ...
                '• 2X/3X: Misalignment contribution', ...
                '• Combined signature overlaps'};
        case 'mixed_wear_lube'
            interp_text = {'Interpretation:', ...
                '• Broadband friction noise', ...
                '• Modulated by wear patterns', ...
                '• Non-linear stick-slip effects'};
        case 'mixed_cavit_jeu'
            interp_text = {'Interpretation:', ...
                '• Random bubble collapse impacts', ...
                '• Sub-harmonics from looseness', ...
                '• High variance envelope'};
        otherwise
            interp_text = {'Mixed fault signature'};
    end
    annotation('textbox', [0.72, 0.05, 0.22, 0.15], ...
        'String', interp_text, 'FontSize', 9, ...
        'BackgroundColor', 'white', 'EdgeColor', 'black', ...
        'FitBoxToText', 'on');
    
    sgtitle(sprintf('Figure D: %s - Fault Decomposition', fault_name), ...
        'FontSize', 16, 'FontWeight', 'bold');
    
    exportgraphics(fig_decomp, fullfile(output_dir, sprintf('FigD_%s_Decomposition.png', fault_code)), ...
        'Resolution', EXPORT_DPI);
    close(fig_decomp);
    fprintf('  ✓ FigD saved\n\n');
end

%% ========================================================================
%% COMPARATIVE ANALYSIS
%% ========================================================================

fprintf('========================================================================\n');
fprintf('   COMPARATIVE ANALYSIS: MIXED vs SINGLE FAULTS\n');
fprintf('========================================================================\n\n');

% Create comparison figure
fig_comp = figure('Name', 'Mixed vs Single Faults', ...
    'Position', [100, 100, 1400, 600], 'Color', 'white', 'Visible', 'off');

mixed_names = fieldnames(all_mixed_results);
num_mixed_analyzed = length(mixed_names);

% Extract mixed fault features
mixed_rms = zeros(1, num_mixed_analyzed);
mixed_kurt = zeros(1, num_mixed_analyzed);
mixed_2X = zeros(1, num_mixed_analyzed);

for i = 1:num_mixed_analyzed
    mixed_rms(i) = all_mixed_results.(mixed_names{i}).features.rms;
    mixed_kurt(i) = all_mixed_results.(mixed_names{i}).features.kurtosis;
    mixed_2X(i) = all_mixed_results.(mixed_names{i}).ratio_2X;
end

% Comparison bars
subplot(1, 3, 1);
bar(mixed_rms, 'FaceColor', [0.6, 0.3, 0.6]);
set(gca, 'XTickLabel', {'Mis+Imb', 'Wear+Lube', 'Cav+Jeu'}, 'XTickLabelRotation', 30);
ylabel('RMS Amplitude', 'FontWeight', 'bold');
title('RMS Comparison', 'FontWeight', 'bold');
grid on;

subplot(1, 3, 2);
bar(mixed_kurt, 'FaceColor', [0.3, 0.6, 0.6]);
hold on;
yline(3, 'k--', 'Gaussian', 'LineWidth', 1.5);
set(gca, 'XTickLabel', {'Mis+Imb', 'Wear+Lube', 'Cav+Jeu'}, 'XTickLabelRotation', 30);
ylabel('Kurtosis', 'FontWeight', 'bold');
title('Kurtosis Comparison', 'FontWeight', 'bold');
grid on;

subplot(1, 3, 3);
bar(mixed_2X, 'FaceColor', [0.6, 0.6, 0.3]);
set(gca, 'XTickLabel', {'Mis+Imb', 'Wear+Lube', 'Cav+Jeu'}, 'XTickLabelRotation', 30);
ylabel('2X/1X Ratio', 'FontWeight', 'bold');
title('2X/1X Ratio Comparison', 'FontWeight', 'bold');
grid on;

sgtitle('Figure E: Mixed Fault Comparative Analysis', ...
    'FontSize', 16, 'FontWeight', 'bold');

exportgraphics(fig_comp, fullfile(OUTPUT_DIR_BASE, 'FigE_Mixed_Comparative.png'), ...
    'Resolution', EXPORT_DPI);
close(fig_comp);
fprintf('  ✓ FigE_Mixed_Comparative.png saved\n');

%% Save results
save(fullfile(OUTPUT_DIR_BASE, 'MixedFaults_Analysis_Results.mat'), 'all_mixed_results');
fprintf('  ✓ Results saved\n');

%% ========================================================================
%% SUMMARY
%% ========================================================================

fprintf('\n========================================================================\n');
fprintf('   MIXED FAULT ANALYSIS COMPLETE\n');
fprintf('========================================================================\n\n');

fprintf('Generated per fault:\n');
for i = 1:NUM_MIXED
    fprintf('  %s/: FigA-D (4 figures)\n', MIXED_FAULTS{i, 1});
end
fprintf('\n  Comparative: FigE_Mixed_Comparative.png\n');

fprintf('\nNEXT STEP: Run Signal_Analysis_04_Comparative.m\n');
fprintf('========================================================================\n');
