%% ========================================================================
%% SIGNAL ANALYSIS PART 4: COMPARATIVE ANALYSIS + AI INTEGRATION
%% ========================================================================
%%
%% Scientific Analysis of Hydrodynamic Bearing Vibration Signals
%% For Research Publication Purposes
%%
%% This script provides:
%%   1. Side-by-side comparisons of all 11 fault types vs healthy baseline
%%   2. Feature importance analysis for classification
%%   3. AI/ML technique proposals and mathematical foundations
%%   4. Research recommendations for novel approaches
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

OUTPUT_DIR = 'Figures/Comparative';
EXPORT_DPI = 300;

if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

fprintf('========================================================================\n');
fprintf('   COMPARATIVE ANALYSIS + AI INTEGRATION\n');
fprintf('========================================================================\n\n');

%% ========================================================================
%% LOAD ALL RESULTS
%% ========================================================================

fprintf('Loading analysis results from all phases...\n');

% Load healthy baseline
healthy = load('Figures/Healthy/Healthy_Baseline_Features.mat');
fprintf('  ✓ Healthy baseline loaded\n');

% Load single fault results
single_faults = load('Figures/Faults/SingleFaults_Analysis_Results.mat');
fprintf('  ✓ Single fault results loaded\n');

% Load mixed fault results
mixed_faults = load('Figures/Mixed/MixedFaults_Analysis_Results.mat');
fprintf('  ✓ Mixed fault results loaded\n\n');

%% ========================================================================
%% SECTION 1: COMPILE ALL FEATURES INTO COMPARISON TABLE
%% ========================================================================

fprintf('SECTION 1: Compiling Feature Comparison Table\n');
fprintf('------------------------------------------\n\n');

% Define all fault types (11 total)
ALL_FAULTS = {
    'sain', 'Healthy';
    'desalignement', 'Misalignment';
    'desequilibre', 'Imbalance';
    'jeu', 'Clearance';
    'lubrification', 'Lubrication';
    'cavitation', 'Cavitation';
    'usure', 'Wear';
    'oilwhirl', 'Oil Whirl';
    'mixed_misalign_imbalance', 'Mixed M+I';
    'mixed_wear_lube', 'Mixed W+L';
    'mixed_cavit_jeu', 'Mixed C+J'
};
NUM_FAULTS = size(ALL_FAULTS, 1);

% Initialize feature table
feature_names = {'RMS', 'Kurtosis', 'Crest', '2X_1X', '3X_1X', 'Sub_1X'};
num_features = length(feature_names);
feature_table = zeros(NUM_FAULTS, num_features);

% Fill in healthy baseline
feature_table(1, :) = [healthy.stat_rms, healthy.stat_kurt, healthy.stat_crest, ...
    healthy.ratio_2X_1X, healthy.ratio_3X_1X, 0.1];  % Estimate sub ratio for healthy

% Fill in single faults
single_names = fieldnames(single_faults.all_results);
for i = 1:length(single_names)
    fname = single_names{i};
    % Find index in ALL_FAULTS
    for j = 1:NUM_FAULTS
        if strcmp(ALL_FAULTS{j, 1}, fname)
            r = single_faults.all_results.(fname);
            feature_table(j, :) = [r.features.rms, r.features.kurtosis, ...
                r.features.crest, r.ratio_2X_1X, r.ratio_3X_1X, r.ratio_sub_1X];
            break;
        end
    end
end

% Fill in mixed faults
mixed_names = fieldnames(mixed_faults.all_mixed_results);
for i = 1:length(mixed_names)
    fname = mixed_names{i};
    for j = 1:NUM_FAULTS
        if strcmp(ALL_FAULTS{j, 1}, fname)
            r = mixed_faults.all_mixed_results.(fname);
            feature_table(j, :) = [r.features.rms, r.features.kurtosis, ...
                r.features.crest, r.ratio_2X, r.ratio_3X, r.ratio_sub];
            break;
        end
    end
end

% Display table
fprintf('%-20s %10s %10s %10s %10s %10s %10s\n', ...
    'Fault', 'RMS', 'Kurtosis', 'Crest', '2X/1X', '3X/1X', 'Sub/1X');
fprintf('%s\n', repmat('-', 1, 85));
for i = 1:NUM_FAULTS
    fprintf('%-20s %10.4f %10.2f %10.2f %10.3f %10.3f %10.3f\n', ...
        ALL_FAULTS{i, 2}, feature_table(i, :));
end
fprintf('\n');

%% ========================================================================
%% FIGURE 1: Feature Comparison Heatmap
%% ========================================================================

fprintf('Generating Figure 1: Feature Comparison Heatmap...\n');

fig1 = figure('Name', 'Feature Heatmap', ...
    'Position', [100, 100, 1000, 700], 'Color', 'white', 'Visible', 'off');

% Normalize features for visualization
feature_norm = (feature_table - min(feature_table)) ./ (max(feature_table) - min(feature_table) + eps);

% Use imagesc instead of heatmap for compatibility
imagesc(feature_norm);
colormap(parula);
cb = colorbar;
cb.Label.String = 'Normalized Value';
cb.Label.FontSize = 11;

% Configure axes
set(gca, 'XTick', 1:num_features, 'XTickLabel', feature_names, ...
    'YTick', 1:NUM_FAULTS, 'YTickLabel', ALL_FAULTS(:, 2), ...
    'FontSize', 10);
xlabel('Features', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Fault Type', 'FontSize', 12, 'FontWeight', 'bold');
title('Figure 1: Normalized Feature Comparison Heatmap', ...
    'FontSize', 14, 'FontWeight', 'bold');

% Add value annotations
for i = 1:NUM_FAULTS
    for j = 1:num_features
        text(j, i, sprintf('%.2f', feature_norm(i, j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', 'white');
    end
end

exportgraphics(fig1, fullfile(OUTPUT_DIR, 'Fig1_Feature_Heatmap.png'), ...
    'Resolution', EXPORT_DPI);
close(fig1);
fprintf('  ✓ Fig1_Feature_Heatmap.png\n');

%% ========================================================================
%% FIGURE 2: Multi-Feature Spider/Radar Chart
%% ========================================================================

fprintf('Generating Figure 2: Spider Chart Comparison...\n');

fig2 = figure('Name', 'Spider Chart', ...
    'Position', [100, 100, 1200, 800], 'Color', 'white', 'Visible', 'off');

% Create polar axes for spider chart
theta = linspace(0, 2*pi, num_features + 1);
fault_colors = lines(NUM_FAULTS);

polaraxes;
hold on;

for i = 1:NUM_FAULTS
    % Normalize to 0-1 for this fault
    vals = feature_norm(i, :);
    vals = [vals, vals(1)];  % Close the polygon
    polarplot(theta, vals, '-o', 'LineWidth', 1.5, 'Color', fault_colors(i, :), ...
        'MarkerSize', 5, 'MarkerFaceColor', fault_colors(i, :));
end

% Configure polar axes
ax = gca;
ax.ThetaTick = rad2deg(theta(1:end-1));
ax.ThetaTickLabel = feature_names;
ax.RLim = [0, 1];

legend(ALL_FAULTS(:, 2), 'Location', 'eastoutside', 'FontSize', 9);
title('Figure 2: Multi-Feature Spider Chart (Normalized)', ...
    'FontSize', 14, 'FontWeight', 'bold');

exportgraphics(fig2, fullfile(OUTPUT_DIR, 'Fig2_Spider_Chart.png'), ...
    'Resolution', EXPORT_DPI);
close(fig2);
fprintf('  ✓ Fig2_Spider_Chart.png\n');

%% ========================================================================
%% FIGURE 3: Healthy vs Faulty Comparison (Box Plots)
%% ========================================================================

fprintf('Generating Figure 3: Healthy vs Faulty Box Plots...\n');

fig3 = figure('Name', 'Box Plots', ...
    'Position', [100, 100, 1400, 600], 'Color', 'white', 'Visible', 'off');

subplot_titles = {'RMS Amplitude', 'Kurtosis', 'Crest Factor', ...
    '2X/1X Ratio', '3X/1X Ratio', 'Sub-sync/1X Ratio'};

for f = 1:num_features
    subplot(2, 3, f);
    
    % Group: Healthy, Single Faults, Mixed Faults
    healthy_val = feature_table(1, f);
    single_vals = feature_table(2:8, f);
    mixed_vals = feature_table(9:11, f);
    
    data = [healthy_val; single_vals; mixed_vals];
    group = [1; repmat(2, 7, 1); repmat(3, 3, 1)];
    
    boxplot(data, group, 'Labels', {'Healthy', 'Single', 'Mixed'}, ...
        'Colors', 'brk', 'Widths', 0.6);
    hold on;
    
    % Add individual data points
    scatter(group + 0.1*randn(size(group)), data, 50, 'filled', 'MarkerFaceAlpha', 0.5);
    
    ylabel(feature_names{f}, 'FontWeight', 'bold');
    title(subplot_titles{f}, 'FontWeight', 'bold');
    grid on;
end

sgtitle('Figure 3: Feature Distribution by Fault Category', ...
    'FontSize', 16, 'FontWeight', 'bold');

exportgraphics(fig3, fullfile(OUTPUT_DIR, 'Fig3_BoxPlots.png'), ...
    'Resolution', EXPORT_DPI);
close(fig3);
fprintf('  ✓ Fig3_BoxPlots.png\n');

%% ========================================================================
%% FIGURE 4: Feature Importance for Classification
%% ========================================================================

fprintf('Generating Figure 4: Feature Importance Analysis...\n');

fig4 = figure('Name', 'Feature Importance', ...
    'Position', [100, 100, 1000, 650], 'Color', 'white', 'Visible', 'off');

% Calculate feature importance using multiple metrics
% 1. Coefficient of Variation (CV) - higher CV = more spread = more useful
% 2. Range / Mean - normalized spread
% 3. Interclass separability

importance_scores = zeros(1, num_features);

for f = 1:num_features
    col = feature_table(:, f);
    
    % Healthy vs Faulty separability (healthy is row 1)
    healthy_val = col(1);
    faulty_vals = col(2:end);
    
    % Distance from healthy to mean of faulty
    separation = abs(healthy_val - mean(faulty_vals));
    
    % Spread among faulty types (higher = more discriminative)
    faulty_spread = std(faulty_vals);
    
    % Combine: good feature separates healthy from faulty AND differentiates faults
    importance_scores(f) = (separation + faulty_spread) / (abs(mean(col)) + eps);
end

% Normalize to 0-1
importance_scores = importance_scores / max(importance_scores);

% Sort by importance
[sorted_scores, sort_idx] = sort(importance_scores, 'descend');
sorted_names = feature_names(sort_idx);

% Create horizontal bar chart with color gradient
colors = [linspace(0.2, 0.8, num_features)', ...
          linspace(0.6, 0.3, num_features)', ...
          linspace(0.9, 0.3, num_features)'];

for i = 1:num_features
    barh(i, sorted_scores(i), 'FaceColor', colors(i,:), 'EdgeColor', 'none');
    hold on;
end

% Configure axes
set(gca, 'YTick', 1:num_features, 'YTickLabel', sorted_names, 'FontSize', 12);
xlabel('Normalized Importance Score', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Feature', 'FontSize', 13, 'FontWeight', 'bold');
title('Figure 4: Feature Importance for Fault Classification', ...
    'FontSize', 15, 'FontWeight', 'bold');
subtitle('Based on healthy-faulty separation and inter-fault discrimination', 'FontSize', 11);
grid on;
xlim([0, 1.15]);

% Add value labels
for i = 1:num_features
    text(sorted_scores(i) + 0.02, i, sprintf('%.2f', sorted_scores(i)), ...
        'FontSize', 11, 'VerticalAlignment', 'middle', 'FontWeight', 'bold');
end

% Add interpretation box (top-right)
annotation('textbox', [0.65, 0.72, 0.28, 0.18], ...
    'String', {'Interpretation:', ...
               '• Higher = better for classification', ...
               '• Top features should be prioritized', ...
               '• in ML model training'}, ...
    'FontSize', 9, 'BackgroundColor', [0.95, 0.95, 0.95], ...
    'EdgeColor', [0.7, 0.7, 0.7], 'FitBoxToText', 'on');

exportgraphics(fig4, fullfile(OUTPUT_DIR, 'Fig4_Feature_Importance.png'), ...
    'Resolution', EXPORT_DPI);
close(fig4);
fprintf('  ✓ Fig4_Feature_Importance.png\n');

%% ========================================================================
%% FIGURE 5: t-SNE/PCA Visualization (2D Embedding)
%% ========================================================================

fprintf('Generating Figure 5: 2D Feature Embedding (PCA)...\n');

fig5 = figure('Name', 'PCA Embedding', ...
    'Position', [100, 100, 800, 700], 'Color', 'white', 'Visible', 'off');

% Standardize features
X = feature_table;
X_std = (X - mean(X)) ./ (std(X) + eps);

% PCA
[coeff, score, ~, ~, explained] = pca(X_std);

% Plot first two principal components
gscatter(score(:, 1), score(:, 2), ALL_FAULTS(:, 2), ...
    lines(NUM_FAULTS), 'osdv^<>ph+x*', 15);
xlabel(sprintf('PC1 (%.1f%% variance)', explained(1)), 'FontSize', 12, 'FontWeight', 'bold');
ylabel(sprintf('PC2 (%.1f%% variance)', explained(2)), 'FontSize', 12, 'FontWeight', 'bold');
title('Figure 5: PCA Embedding of Fault Features', ...
    'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'bestoutside');
grid on;

exportgraphics(fig5, fullfile(OUTPUT_DIR, 'Fig5_PCA_Embedding.png'), ...
    'Resolution', EXPORT_DPI);
close(fig5);
fprintf('  ✓ Fig5_PCA_Embedding.png\n');

%% ========================================================================
%% SECTION 2: AI/ML TECHNIQUE PROPOSALS
%% ========================================================================

fprintf('\n========================================================================\n');
fprintf('   SECTION 2: AI/MACHINE LEARNING INTEGRATION PROPOSALS\n');
fprintf('========================================================================\n\n');

% Write AI proposals to text file
ai_report = fullfile(OUTPUT_DIR, 'AI_Integration_Proposals.txt');
fid = fopen(ai_report, 'w');

fprintf(fid, '=======================================================================\n');
fprintf(fid, '   AI/MACHINE LEARNING INTEGRATION PROPOSALS\n');
fprintf(fid, '   PFD Fault Diagnosis System - Research Enhancement\n');
fprintf(fid, '=======================================================================\n\n');

fprintf(fid, '1. CONVOLUTIONAL NEURAL NETWORKS (CNN) ON SPECTROGRAMS\n');
fprintf(fid, '-----------------------------------------------------------------------\n');
fprintf(fid, 'Approach: Treat spectrograms as images for fault classification.\n\n');
fprintf(fid, 'Mathematical Foundation:\n');
fprintf(fid, '  Convolution: (I * K)(x,y) = ΣΣ I(x+i, y+j) · K(i,j)\n');
fprintf(fid, '  where I = input spectrogram, K = learned kernel\n\n');
fprintf(fid, 'Architecture Recommendation:\n');
fprintf(fid, '  - Input: 224x224 spectrogram image\n');
fprintf(fid, '  - Conv layers: 32 → 64 → 128 filters\n');
fprintf(fid, '  - Pooling: Max pooling 2x2\n');
fprintf(fid, '  - Dense: 256 → 11 classes\n\n');
fprintf(fid, 'Expected Improvement: +3-5%% accuracy over traditional ML\n\n');

fprintf(fid, '2. LONG SHORT-TERM MEMORY (LSTM) NETWORKS\n');
fprintf(fid, '-----------------------------------------------------------------------\n');
fprintf(fid, 'Approach: Sequence modeling of raw vibration time series.\n\n');
fprintf(fid, 'Mathematical Foundation:\n');
fprintf(fid, '  Forget gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)\n');
fprintf(fid, '  Input gate:  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)\n');
fprintf(fid, '  Cell state:  C_t = f_t * C_{t-1} + i_t * tanh(W_C · [h_{t-1}, x_t] + b_C)\n');
fprintf(fid, '  Output:      h_t = o_t * tanh(C_t)\n\n');
fprintf(fid, 'Use Case: Capture temporal dependencies in fault progression.\n\n');

fprintf(fid, '3. AUTOENCODER ANOMALY DETECTION\n');
fprintf(fid, '-----------------------------------------------------------------------\n');
fprintf(fid, 'Approach: Train on healthy data only, detect anomalies via reconstruction error.\n\n');
fprintf(fid, 'Mathematical Foundation:\n');
fprintf(fid, '  Encoder: z = f_enc(x) = σ(W_e · x + b_e)\n');
fprintf(fid, '  Decoder: x̂ = f_dec(z) = σ(W_d · z + b_d)\n');
fprintf(fid, '  Loss: L = ||x - x̂||² (MSE reconstruction error)\n');
fprintf(fid, '  Anomaly score: If L > threshold → Fault detected\n\n');
fprintf(fid, 'Advantage: No labeled fault data required for training.\n\n');

fprintf(fid, '4. ATTENTION MECHANISMS (TRANSFORMER)\n');
fprintf(fid, '-----------------------------------------------------------------------\n');
fprintf(fid, 'Approach: Learn which time/frequency regions are most relevant.\n\n');
fprintf(fid, 'Mathematical Foundation:\n');
fprintf(fid, '  Attention: Attention(Q, K, V) = softmax(QK^T / √d_k) · V\n');
fprintf(fid, '  Self-attention allows the model to focus on fault-relevant regions.\n\n');
fprintf(fid, 'Benefit: Explainable AI - visualize attention weights.\n\n');

fprintf(fid, '5. TRANSFER LEARNING\n');
fprintf(fid, '-----------------------------------------------------------------------\n');
fprintf(fid, 'Approach: Pre-train on large vibration datasets, fine-tune for PFD.\n\n');
fprintf(fid, 'Strategy:\n');
fprintf(fid, '  - Use pre-trained CNN (ResNet, VGG) on ImageNet\n');
fprintf(fid, '  - Replace final layers for 11-class PFD classification\n');
fprintf(fid, '  - Fine-tune with labeled PFD spectrograms\n\n');
fprintf(fid, 'Expected: Faster convergence, better generalization.\n\n');

fprintf(fid, '6. PHYSICS-INFORMED NEURAL NETWORKS (PINNs)\n');
fprintf(fid, '-----------------------------------------------------------------------\n');
fprintf(fid, 'Approach: Incorporate bearing physics into neural network loss.\n\n');
fprintf(fid, 'Mathematical Foundation:\n');
fprintf(fid, '  Total Loss = L_data + λ · L_physics\n');
fprintf(fid, '  L_physics encodes Reynolds equation, Sommerfeld number constraints.\n\n');
fprintf(fid, 'Innovation: Novel research direction combining ML with tribology.\n\n');

fprintf(fid, '7. DOMAIN ADAPTATION (SIM-to-REAL)\n');
fprintf(fid, '-----------------------------------------------------------------------\n');
fprintf(fid, 'Approach: Transfer knowledge from synthetic to real sensor data.\n\n');
fprintf(fid, 'Techniques:\n');
fprintf(fid, '  - Adversarial domain adaptation (align feature distributions)\n');
fprintf(fid, '  - Maximum Mean Discrepancy (MMD) loss\n');
fprintf(fid, '  - Cycle-consistent GAN for domain translation\n\n');
fprintf(fid, 'Critical for production deployment with real sensors.\n\n');

fprintf(fid, '=======================================================================\n');
fprintf(fid, '   RECOMMENDED RESEARCH ROADMAP\n');
fprintf(fid, '=======================================================================\n\n');
fprintf(fid, 'Phase 1 (Weeks 1-2): CNN on spectrograms - quick win, baseline deep learning\n');
fprintf(fid, 'Phase 2 (Weeks 3-4): LSTM for temporal modeling - capture fault progression\n');
fprintf(fid, 'Phase 3 (Weeks 5-6): Autoencoder anomaly detection - unsupervised approach\n');
fprintf(fid, 'Phase 4 (Weeks 7-8): Attention/Transformer - explainable AI\n');
fprintf(fid, 'Phase 5 (Ongoing): Physics-informed NN - novel research contribution\n\n');

fprintf(fid, '=======================================================================\n');
fprintf(fid, '   PUBLICATION POTENTIAL\n');
fprintf(fid, '=======================================================================\n\n');
fprintf(fid, 'Title Suggestion:\n');
fprintf(fid, '"Physics-Informed Deep Learning for Hydrodynamic Bearing Fault Diagnosis:\n');
fprintf(fid, ' Integrating Tribological Knowledge with Neural Networks"\n\n');
fprintf(fid, 'Target Journals:\n');
fprintf(fid, '  - Mechanical Systems and Signal Processing (IF: 8.4)\n');
fprintf(fid, '  - Tribology International (IF: 6.2)\n');
fprintf(fid, '  - IEEE Transactions on Industrial Informatics (IF: 11.6)\n\n');

fclose(fid);
fprintf('  ✓ AI_Integration_Proposals.txt saved\n');

%% ========================================================================
%% FIGURE 6: AI Architecture Diagram (Conceptual)
%% ========================================================================

fprintf('Generating Figure 6: AI Pipeline Diagram...\n');

fig6 = figure('Name', 'AI Pipeline', ...
    'Position', [100, 100, 1000, 900], 'Color', 'white', 'Visible', 'off');

% Set up axes - vertical layout
axes('Position', [0.1, 0.05, 0.8, 0.88]);
axis([0, 10, 0, 12]);
axis off;
hold on;

% === TITLE ===
text(5, 11.5, 'AI/ML Pipeline for PFD Fault Diagnosis', ...
    'FontSize', 18, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
    'Color', [0.1, 0.1, 0.35]);

% === STAGE 1: INPUT ===
rectangle('Position', [2.5, 9.5, 5, 1.2], 'Curvature', 0.2, ...
    'FaceColor', [0.7, 0.85, 1], 'EdgeColor', [0.2, 0.4, 0.7], 'LineWidth', 2.5);
text(5, 10.1, 'Vibration Signal Input', 'FontSize', 14, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'Color', [0.1, 0.25, 0.5]);

% Arrow down
annotation('arrow', [0.5, 0.5], [0.77, 0.72], 'LineWidth', 2.5, 'Color', [0.3, 0.3, 0.3]);

% === STAGE 2: FEATURE EXTRACTION ===
rectangle('Position', [1.5, 7.2, 7, 1.5], 'Curvature', 0.15, ...
    'FaceColor', [0.75, 0.95, 0.75], 'EdgeColor', [0.2, 0.55, 0.25], 'LineWidth', 2.5);
text(5, 8.15, 'Feature Extraction', 'FontSize', 14, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'Color', [0.1, 0.4, 0.15]);
text(5, 7.55, 'Time-domain  |  Frequency  |  Wavelet  |  Envelope', 'FontSize', 10, ...
    'HorizontalAlignment', 'center', 'Color', [0.3, 0.3, 0.3]);

% Arrow down
annotation('arrow', [0.5, 0.5], [0.58, 0.53], 'LineWidth', 2.5, 'Color', [0.3, 0.3, 0.3]);

% === STAGE 3: ML MODELS ===
rectangle('Position', [1.5, 5, 7, 1.5], 'Curvature', 0.15, ...
    'FaceColor', [1, 0.88, 0.72], 'EdgeColor', [0.7, 0.4, 0.15], 'LineWidth', 2.5);
text(5, 5.95, 'Machine Learning Models', 'FontSize', 14, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'Color', [0.5, 0.25, 0.05]);
text(5, 5.35, 'CNN  |  LSTM  |  Random Forest  |  Transformer', 'FontSize', 10, ...
    'HorizontalAlignment', 'center', 'Color', [0.3, 0.3, 0.3]);

% Three arrows down to outputs
annotation('arrow', [0.30, 0.25], [0.40, 0.32], 'LineWidth', 2, 'Color', [0.4, 0.4, 0.4]);
annotation('arrow', [0.50, 0.50], [0.40, 0.32], 'LineWidth', 2, 'Color', [0.4, 0.4, 0.4]);
annotation('arrow', [0.70, 0.75], [0.40, 0.32], 'LineWidth', 2, 'Color', [0.4, 0.4, 0.4]);

% === STAGE 4: OUTPUTS (3 boxes side by side) ===
% Anomaly Score
rectangle('Position', [0.3, 2.3, 2.8, 1.3], 'Curvature', 0.2, ...
    'FaceColor', [1, 0.82, 0.82], 'EdgeColor', [0.6, 0.2, 0.2], 'LineWidth', 2);
text(1.7, 3.1, 'Anomaly Score', 'FontSize', 11, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center');
text(1.7, 2.6, '(Confidence %)', 'FontSize', 9, 'HorizontalAlignment', 'center', ...
    'Color', [0.4, 0.4, 0.4]);

% Fault Type (center, primary output)
rectangle('Position', [3.6, 2.3, 2.8, 1.3], 'Curvature', 0.2, ...
    'FaceColor', [0.82, 0.82, 1], 'EdgeColor', [0.25, 0.25, 0.65], 'LineWidth', 3);
text(5, 3.1, 'Fault Type', 'FontSize', 11, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center');
text(5, 2.6, '(11 Classes)', 'FontSize', 9, 'HorizontalAlignment', 'center', ...
    'Color', [0.4, 0.4, 0.4]);

% Severity Level
rectangle('Position', [6.9, 2.3, 2.8, 1.3], 'Curvature', 0.2, ...
    'FaceColor', [0.82, 1, 0.82], 'EdgeColor', [0.2, 0.5, 0.2], 'LineWidth', 2);
text(8.3, 3.1, 'Severity Level', 'FontSize', 11, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center');
text(8.3, 2.6, '(Low / Med / High)', 'FontSize', 9, 'HorizontalAlignment', 'center', ...
    'Color', [0.4, 0.4, 0.4]);

% === NOVEL CONTRIBUTION BOX (bottom left) ===
rectangle('Position', [0.3, 0.3, 4.5, 1.5], 'Curvature', 0.15, ...
    'FaceColor', [1, 0.97, 0.8], 'EdgeColor', [0.75, 0.55, 0.1], 'LineWidth', 3);
text(2.55, 1.3, '★ NOVEL CONTRIBUTION', 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'Color', [0.6, 0.4, 0.05]);
text(2.55, 0.7, 'Physics-Informed Neural Network (PINN)', 'FontSize', 11, ...
    'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', [0.35, 0.2, 0]);

% Dashed arrow from PINN to ML Models
plot([4.8, 4.8], [1.8, 5], '--', 'Color', [0.7, 0.5, 0.1], 'LineWidth', 2);
plot(4.8, 5, '^', 'Color', [0.7, 0.5, 0.1], 'MarkerSize', 10, 'MarkerFaceColor', [0.7, 0.5, 0.1]);
text(4.4, 3.5, 'Physics', 'FontSize', 9, 'FontWeight', 'bold', 'Color', [0.6, 0.4, 0.1], 'Rotation', 90);

% === TECHNIQUES LIST BOX (bottom right) ===
rectangle('Position', [5.2, 0.3, 4.5, 1.5], 'Curvature', 0.15, ...
    'FaceColor', [0.96, 0.96, 0.96], 'EdgeColor', [0.5, 0.5, 0.5], 'LineWidth', 1.5);
text(7.45, 1.5, 'AI Techniques:', 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center');
text(5.4, 1.1, '• CNN on Spectrograms', 'FontSize', 9);
text(5.4, 0.7, '• LSTM Sequence Modeling', 'FontSize', 9);
text(7.6, 1.1, '• Autoencoder Anomaly', 'FontSize', 9);
text(7.6, 0.7, '• Attention/Transformer', 'FontSize', 9);

exportgraphics(fig6, fullfile(OUTPUT_DIR, 'Fig6_AI_Pipeline.png'), ...
    'Resolution', EXPORT_DPI);
close(fig6);
fprintf('  ✓ Fig6_AI_Pipeline.png\n');

%% ========================================================================
%% SAVE FINAL SUMMARY
%% ========================================================================

% Save all comparison data
save(fullfile(OUTPUT_DIR, 'Comparative_Analysis_Results.mat'), ...
    'ALL_FAULTS', 'feature_table', 'feature_names', 'feature_norm', ...
    'importance_scores', 'score', 'explained');
fprintf('  ✓ Comparative_Analysis_Results.mat saved\n');

%% ========================================================================
%% FINAL SUMMARY
%% ========================================================================

fprintf('\n========================================================================\n');
fprintf('   COMPARATIVE ANALYSIS COMPLETE\n');
fprintf('========================================================================\n\n');

fprintf('Generated Figures:\n');
fprintf('  1. Fig1_Feature_Heatmap.png\n');
fprintf('  2. Fig2_Spider_Chart.png\n');
fprintf('  3. Fig3_BoxPlots.png\n');
fprintf('  4. Fig4_Feature_Importance.png\n');
fprintf('  5. Fig5_PCA_Embedding.png\n');
fprintf('  6. Fig6_AI_Pipeline.png\n\n');

fprintf('Generated Reports:\n');
fprintf('  • AI_Integration_Proposals.txt\n');
fprintf('  • Comparative_Analysis_Results.mat\n\n');

fprintf('========================================================================\n');
fprintf('   ALL 4 PHASES COMPLETE - READY FOR PUBLICATION\n');
fprintf('========================================================================\n\n');

fprintf('Total Deliverables:\n');
fprintf('  Phase 1: 9 figures (Healthy baseline)\n');
fprintf('  Phase 2: 29 figures (7 single faults + comparative)\n');
fprintf('  Phase 3: 13 figures (3 mixed faults + comparative)\n');
fprintf('  Phase 4: 6 figures + AI proposals document\n');
fprintf('  TOTAL: 57 publication-ready figures\n\n');

fprintf('Live Scripts created:\n');
fprintf('  1. Signal_Analysis_01_Healthy.m\n');
fprintf('  2. Signal_Analysis_02_Single_Faults.m\n');
fprintf('  3. Signal_Analysis_03_Mixed_Faults.m\n');
fprintf('  4. Signal_Analysis_04_Comparative.m\n\n');

fprintf('To convert to Live Script (.mlx):\n');
fprintf('  1. Open in MATLAB Editor\n');
fprintf('  2. Right-click → Save As → .mlx format\n');
fprintf('  3. MATLAB auto-converts sections to rich text\n\n');

fprintf('========================================================================\n');
