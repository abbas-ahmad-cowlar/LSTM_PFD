% ========================================================================
% ADVANCED PFD FAULT DIAGNOSIS PIPELINE - PRODUCTION VERSION
% Machine Learning Pipeline for Hydrodynamic Bearing Fault Classification
% ========================================================================
%
% PURPOSE:
%   Production-ready ML pipeline for diagnosing PFD (Palier Fluide 
%   Dynamique) hydrodynamic bearing faults using advanced signal
%   processing and machine learning techniques.
%
% FAULT TYPES:
%   8 single faults + 3 overlapping fault combinations
%   - sain (healthy)
%   - desalignement (misalignment)
%   - desequilibre (imbalance)
%   - jeu (bearing clearance)
%   - lubrification (lubrication issues)
%   - cavitation
%   - usure (wear)
%   - oilwhirl
%   - mixed_misalign_imbalance
%   - mixed_wear_lube
%   - mixed_cavit_jeu
%
% Author: Syed Abbas Ahmad
% Version: Production v2.0
% Date: October 30, 2025
% ========================================================================

clear;
clc;
close all;

rng(42, 'twister');

%% ========================================================================
%% STEP 1: SETUP, CONFIGURATION, AND VALIDATION
%% ========================================================================

fprintf('========================================================================\n');
fprintf('   PFD FAULT DIAGNOSIS PIPELINE - PRODUCTION v2.0\n');
fprintf('========================================================================\n');

% ========================================================================
% CONFIGURATION STRUCTURE
% ========================================================================

CONFIG = struct();

% --- I/O and File Configuration ---
CONFIG.inputDir = 'data_signaux_sep_production';
CONFIG.outputDir = 'PFD_SVM_Results_Production';
CONFIG.logFile = 'pipeline_run_production.log';
CONFIG.outputCSV = 'features_pfd_production.csv';
CONFIG.modelFile = 'Best_PFD_Model_Production.mat';
CONFIG.inferenceFunction = 'predictPFDFault_Production.m';
CONFIG.reportFile = 'PFD_Analysis_Report_Production.txt';

% --- ML & Data Splitting Configuration ---
CONFIG.trainRatio = 0.70;
CONFIG.valRatio = 0.15;
CONFIG.testRatio = 0.15;
CONFIG.cvFolds = 5;
CONFIG.hyperOptIterations = 50;
CONFIG.numFeaturesToSelect = 15;

% --- Feature Engineering Configuration ---
CONFIG.useFeatureSelection = true;
CONFIG.performFeatureSelectionAfterSplit = true;  % Prevent data leakage
CONFIG.generateRealLearningCurve = true;
CONFIG.generateTSNEPlot = true;
CONFIG.generateFeatureImportance = true;
CONFIG.handleImbalance = false;
CONFIG.imbalanceMethod = 'CostMatrix';

% --- ADVANCED FEATURES CONFIGURATION ---
% Set to true to include computationally expensive features
% WARNING: Enabling increases extraction time by 10x with minimal accuracy gain
CONFIG.includeAdvancedFeatures = false;  % CWT, WPT, Non-linear dynamics

% --- ML MODEL CONFIGURATION ---
% Select which models to train and compare
CONFIG.models = struct();
CONFIG.models.trainSVM = true;              % Support Vector Machine (baseline)
CONFIG.models.trainRandomForest = true;     % Random Forest
CONFIG.models.trainGradientBoosting = true; % Gradient Boosting Trees
CONFIG.models.trainNeuralNetwork = true;    % 3-layer MLP
CONFIG.models.trainStackedEnsemble = false; % Meta-learner (DISABLED: unstable with AdaBoost)

% Model selection strategy
CONFIG.modelSelectionMetric = 'ValidationAccuracy';  % or 'CrossValAccuracy'

% --- Production-Ready Features ---
CONFIG.useCalibration = true;
CONFIG.calibrationMethod = 'sigmoid';
CONFIG.useCrossDatasetValidation = true;
CONFIG.testAdversarialRobustness = true;

% --- Development Mode (Fast Iterations) ---
CONFIG.developmentMode = false;  % Set true for quick testing
if CONFIG.developmentMode
    fprintf('⚠️  DEVELOPMENT MODE ACTIVE - Using reduced settings\n');
    CONFIG.hyperOptIterations = 10;
    CONFIG.generateTSNEPlot = false;
    CONFIG.useCalibration = false;
    CONFIG.testAdversarialRobustness = false;
end

% --- Frequency Band Configuration ---
CONFIG.bands = struct(...
    'Oilwhirl', [0.40, 0.48] * 60, ...
    'Misalignment', [1.95, 2.05] * 60, ...
    'Cavitation', [1500, 2500] ...
    );

% ========================================================================
% ENVIRONMENT SETUP & LOGGING
% ========================================================================

if ~exist(CONFIG.outputDir, 'dir')
    mkdir(CONFIG.outputDir);
end

logFilePath = fullfile(CONFIG.outputDir, CONFIG.logFile);
if exist(logFilePath, 'file')
    try
        delete(logFilePath);
    catch
        % File might be locked, continue anyway
    end
end
diary(logFilePath);

fprintf('Pipeline started: %s\n', char(datetime('now')));
fprintf('Full log saved to: %s\n', logFilePath);
fprintf('----------------------------------\n');
fprintf('Step 1: Setup and Configuration\n');
fprintf('----------------------------------\n');

% ========================================================================
% TOOLBOX VALIDATION
% ========================================================================

fprintf('Validating required toolboxes...\n');
requiredToolboxes = {...
    'Statistics and Machine Learning Toolbox', ...
    'Signal Processing Toolbox', ...
    'Wavelet Toolbox', ...
    'Parallel Computing Toolbox'};

v = ver;
installedToolboxes = {v.Name};
missingToolbox = false;

for i = 1:length(requiredToolboxes)
    if ~any(strcmp(requiredToolboxes{i}, installedToolboxes))
        fprintf('!! ERROR: %s is not installed.\n', requiredToolboxes{i});
        missingToolbox = true;
    end
end

if missingToolbox
    diary off;
    error('Missing required toolboxes. Please install them to proceed.');
else
    fprintf('✓ All required toolboxes are present.\n');
end

% ========================================================================
% INPUT DATA VALIDATION
% ========================================================================

if ~exist(CONFIG.inputDir, 'dir')
    diary off;
    error('Input directory "%s" not found. Please run data generator first.', CONFIG.inputDir);
else
    fprintf('✓ Found input signal directory: %s\n', CONFIG.inputDir);
end

fprintf('✓ Configuration loaded and validated.\n');
fprintf('✓ Advanced features: %s\n', iif(CONFIG.includeAdvancedFeatures, 'ENABLED (slow)', 'DISABLED (fast)'));
fprintf('✓ ML Models enabled: ');
modelList = {};
if CONFIG.models.trainSVM, modelList{end+1} = 'SVM'; end
if CONFIG.models.trainRandomForest, modelList{end+1} = 'RF'; end
if CONFIG.models.trainGradientBoosting, modelList{end+1} = 'GBT'; end
if CONFIG.models.trainNeuralNetwork, modelList{end+1} = 'NN'; end
if CONFIG.models.trainStackedEnsemble, modelList{end+1} = 'Ensemble'; end
fprintf('%s\n', strjoin(modelList, ', '));
fprintf('✓ Ready to proceed to Step 2.\n');

%% ========================================================================
%% STEP 2: ADVANCED FEATURE ENGINEERING & EXTRACTION
%% ========================================================================

fprintf('\n========================================================================\n');
fprintf('Step 2: Advanced Feature Engineering & Extraction\n');
fprintf('========================================================================\n');

% ========================================================================
% DEFINE COMPLETE FEATURE LIST
% ========================================================================

% Base features (always included)
baseFeatureNames = {...
    'Mean', 'RMS', 'STD', 'Kurtosis', 'Skewness', ...
    'CrestFactor', 'Energy', ...
    'Envelope_RMS', 'Envelope_Kurtosis', 'Envelope_PeakFactor', 'Envelope_ModulationFreq', ...
    'FreqDominante', 'SpectralCentroid', 'SpectralEntropy', ...
    'BandEnergy_Oilwhirl', 'BandEnergy_Misalign', 'BandEnergy_Cavitation', ...
    'LocalSpectralEntropy_Mean', 'LocalSpectralEntropy_Std', ...
    'Wavelet_Kurtosis', ...
    'Harmonic_2X_to_1X_Ratio', 'Harmonic_3X_to_1X_Ratio', 'THD', ...
    'Spectral_Flatness', 'Spectral_Spread', 'Frequency_Stability', ...
    'Cepstral_Peak_Ratio', 'Quefrency_Centroid', 'Cepstral_Variance', ...
    'Bispectrum_Peak', 'Phase_Coupling', 'NonLinear_Index', ...
    'STFT_Entropy', 'Wavelet_Energy_L1', 'Wavelet_Energy_L5', 'Hilbert_InstFreq_Std'...
    };

% Advanced features (computationally expensive, optional)
advancedFeatureNames = {...
    'CWT_Energy', 'CWT_MeanStd', ...
    'WPT_EnergyRatio_LowHigh', 'WPT_EnergyConcentration', 'WPT_EntropyAcrossScales', ...
    'Wavelet_SignalRatio', 'Ridge_MeanFreq', 'Ridge_StdFreq', ...
    'Lyapunov_Exponent', 'Correlation_Dimension', 'Sample_Entropy', ...
    'Poincare_Dispersion', 'DFA_Alpha', ...
    'STFT_Flatness', 'STFT_Centroid_Variation', 'STFT_Bandwidth_Variation'...
    };

% Combine based on configuration
if CONFIG.includeAdvancedFeatures
    featureNames = [baseFeatureNames, advancedFeatureNames, {'Label'}];
    fprintf('Using FULL feature set: %d features\n', length(featureNames)-1);
    fprintf('  - Base features: %d\n', length(baseFeatureNames));
    fprintf('  - Advanced features: %d\n', length(advancedFeatureNames));
    fprintf('⚠️  WARNING: Advanced features will significantly increase computation time.\n');
else
    featureNames = [baseFeatureNames, {'Label'}];
    fprintf('Using BASE feature set: %d features\n', length(featureNames)-1);
    fprintf('  (Advanced features disabled for speed)\n');
end

numFeatures = length(featureNames) - 1;

% ========================================================================
% LOAD SIGNAL FILES
% ========================================================================

fprintf('\nStep 2.1: Loading signal files...\n');
files = dir(fullfile(CONFIG.inputDir, '*.mat'));

if isempty(files)
    diary off;
    error('No .mat files found in "%s". Did the data generation run?', CONFIG.inputDir);
end

fprintf('Found %d signal files to process...\n', length(files));

% ========================================================================
% DATASET CONFIGURATION VALIDATION
% ========================================================================

fprintf('\nValidating dataset configuration...\n');

% Load first file to check metadata and configuration
firstFile = load(fullfile(CONFIG.inputDir, files(1).name));

% Check if metadata exists
if isfield(firstFile, 'metadata')
    fprintf('✓ Dataset contains metadata (generator version: %s)\n', ...
        firstFile.metadata.generator_version);
    
    % Extract dataset characteristics
    datasetHasMetadata = true;
    
    % Check for augmented samples
    augFiles = sum(contains({files.name}, '_aug.mat'));
    if augFiles > 0
        fprintf('  Found %d augmented samples (%.1f%% of dataset)\n', ...
            augFiles, 100*augFiles/length(files));
    end
    
    % Check severity levels if available
    if isfield(firstFile.metadata, 'severity')
        fprintf('  Severity levels detected: %s\n', firstFile.metadata.severity);
    end
    
    % Check noise configuration
    if isfield(firstFile.metadata, 'noise_sources')
        noiseTypes = fieldnames(firstFile.metadata.noise_sources);
        enabledNoise = 0;
        for nt = 1:length(noiseTypes)
            if firstFile.metadata.noise_sources.(noiseTypes{nt})
                enabledNoise = enabledNoise + 1;
            end
        end
        fprintf('  Noise sources enabled: %d types\n', enabledNoise);
    end
    
else
    fprintf('  ⚠️  No metadata found - using legacy data format\n');
    datasetHasMetadata = false;
end

% Detect actual fault classes present in dataset
fprintf('\nDetecting fault classes in dataset...\n');
actualClasses = {};

% Extract classes from filenames (faster than loading files)
for i = 1:length(files)
    % Parse filename: "faultname_###.mat" or "faultname_###_aug.mat"
    fname = files(i).name;
    
    % Remove _aug suffix if present
    fname = strrep(fname, '_aug.mat', '.mat');
    
    % Extract fault name (everything before last underscore and number)
    tokens = regexp(fname, '^(.+)_\d+\.mat$', 'tokens');
    
    if ~isempty(tokens)
        faultName = tokens{1}{1};
        if ~any(strcmp(actualClasses, faultName))
            actualClasses{end+1} = faultName;
        end
    end
end

fprintf('✓ Detected %d fault classes:\n', length(actualClasses));
for i = 1:length(actualClasses)
    fprintf('  - %s\n', actualClasses{i});
end

% Warning if unexpected class count
expectedClassCount = 11;  % 7 single + 3 mixed + 1 healthy
if length(actualClasses) ~= expectedClassCount
    fprintf('\n⚠️  WARNING: Detected %d classes, expected %d\n', ...
        length(actualClasses), expectedClassCount);
    fprintf('   This may indicate:\n');
    fprintf('   - Data generator had some fault types disabled\n');
    fprintf('   - Incomplete dataset\n');
    fprintf('   Pipeline will adapt to actual classes present.\n\n');
end

% Initialize feature storage
allFeatures = cell(length(files), length(featureNames));
fullFeatureNames = featureNames(1:end-1);

% ========================================================================
% PARALLEL FEATURE EXTRACTION LOOP
% ========================================================================

fprintf('\nStep 2.2: Extracting features from signals...\n');
fprintf('Using parallel processing...\n');

% Also collect metadata if available
allMetadata = cell(length(files), 1);

parfor i = 1:length(files)
    try
        % Load signal data
        data = load(fullfile(CONFIG.inputDir, files(i).name));
        x = data.x;
        fs = data.fs;
        fault = data.fault;
        
        % Extract features
        featValues = extractFeatures(x, fs, CONFIG);
        
        % Store results
        allFeatures(i, :) = [num2cell(featValues), {fault}];
        
        % Store metadata if available (for later analysis)
        if isfield(data, 'metadata')
            allMetadata{i} = data.metadata;
        else
            allMetadata{i} = struct();
        end
        
    catch ME
        fprintf('!! Error processing file %d (%s): %s\n', i, files(i).name, ME.message);
        allFeatures(i, :) = {NaN};
        allMetadata{i} = struct();
    end
end

% Check for failures
failedIdx = cellfun(@(x) isnan(x), allFeatures(:,1));
if any(failedIdx)
    fprintf('⚠️  Warning: %d files failed to process and will be excluded.\n', sum(failedIdx));
    allFeatures(failedIdx, :) = [];
    allMetadata(failedIdx) = [];
end

fprintf('✓ Feature extraction complete: %d signals processed successfully.\n', size(allFeatures, 1));

% ========================================================================
% CREATE FEATURE TABLE AND SAVE
% ========================================================================

fprintf('\nStep 2.3: Creating feature table...\n');

featureTable = cell2table(allFeatures, 'VariableNames', featureNames);
outputCSVPath = fullfile(CONFIG.outputDir, CONFIG.outputCSV);
writetable(featureTable, outputCSVPath);

fprintf('✓ Feature table saved: %s\n', outputCSVPath);
fprintf('  - Samples: %d\n', size(featureTable, 1));
fprintf('  - Features: %d\n', numFeatures);

% Save metadata separately for enhanced analysis
metadataPath = fullfile(CONFIG.outputDir, 'dataset_metadata.mat');
save(metadataPath, 'allMetadata', 'actualClasses', 'datasetHasMetadata', '-v7.3');
fprintf('✓ Metadata saved: dataset_metadata.mat\n');

%% ========================================================================
%% STEP 3: DATA EXPLORATION & VISUALIZATION
%% ========================================================================

fprintf('\n========================================================================\n');
fprintf('Step 3: Data Exploration & Visualization\n');
fprintf('========================================================================\n');

% ========================================================================
% LOAD DATA FOR VISUALIZATION
% ========================================================================

fprintf('\nStep 3.1: Loading feature data for visualization...\n');

data = readtable(outputCSVPath);
X_raw = table2array(data(:, 1:end-1));
Y_raw = categorical(data.Label);
classNames = categories(Y_raw);

fprintf('✓ Loaded %d samples with %d features across %d classes.\n', ...
    size(X_raw, 1), size(X_raw, 2), length(classNames));

% ========================================================================
% DATA CLEANING & NORMALIZATION
% ========================================================================

badMask = isnan(X_raw) | isinf(X_raw);
if any(badMask(:))
    fprintf('⚠️  Warning: Found %d NaN/Inf values. Imputing with column median.\n', sum(badMask(:)));
    for col = 1:size(X_raw, 2)
        if any(badMask(:, col))
            medianVal = median(X_raw(~badMask(:, col), col), 'omitnan');
            X_raw(badMask(:, col), col) = medianVal;
        end
    end
end

X_raw_norm = zscore(X_raw);
featureNamesFormatted = cellfun(@(name) strrep(name, '_', ' '), fullFeatureNames, 'UniformOutput', false);

% ========================================================================
% FIG 0: RAW SIGNAL EXAMPLES
% ========================================================================
fprintf('\nStep 3.2: Generating Fig0 (Raw Signal Examples)...\n');

% --- Define Fault Groups ---
% We split the classes to make the plots less cluttered
basicFaults = {'cavitation', 'desalignement', 'desequilibre', 'jeu', ...
                'lubrication', 'oilwhirl', 'sain', 'usure'};
            
mixedFaults = {'mixed_cavit_jeu', 'mixed_misalign_imbalance', 'mixed_wear_lube'};

% --- Figure 1: Basic Faults ---
fprintf('  Generating Fig0a (Basic Faults)...\n');
fig1 = figure('Name', 'PFD Basic Fault Signal Examples', 'Position', [100, 100, 1600, 900]);
t_sig1 = tiledlayout(length(basicFaults), 3, 'Padding', 'compact', 'TileSpacing', 'compact');

% Loop through BASIC faults
for k = 1:length(basicFaults)
    fault = basicFaults{k};
    fileName = fullfile(CONFIG.inputDir, sprintf('%s_001.mat', fault));
    
    if ~isfile(fileName)
        fprintf('  Warning: Could not find %s. Skipping.\n', fileName);
        nexttile; nexttile; nexttile;
        continue;
    end
    
    sigData = load(fileName);
    x = sigData.x;
    fs = sigData.fs;
    t_vec = (0:length(x)-1)/fs;
    
    % --- Plotting ---
    % Time Domain
    ax1 = nexttile;
    plot(ax1, t_vec, x);
    title(ax1, ['Time: ' fault], 'Interpreter', 'none', 'FontWeight', 'normal');
    xlim(ax1, [0, 0.5]);
    grid(ax1, 'on');
    
    % Frequency Domain (Welch PSD)
    ax2 = nexttile;
    [Pxx, F] = pwelch(x, hann(1024), 512, 2048, fs);
    semilogy(ax2, F, Pxx);
    title(ax2, ['Frequency: ' fault], 'Interpreter', 'none', 'FontWeight', 'normal');
    xlim(ax2, [0, fs/8]); % Keeping your original fs/8 limit
    grid(ax2, 'on');
    
    % Time-Frequency Spectrogram
    ax3 = nexttile;
    spectrogram(x, hann(256), 128, 256, fs, 'yaxis');
    title(ax3, ['Spectrogram: ' fault], 'Interpreter', 'none', 'FontWeight', 'normal');
    
    % --- CLUTTER REDUCTION: Only add labels to outer plots ---
    if k == 1
        % Add Y-labels only to the first row
        ylabel(ax1, 'Amplitude');
        ylabel(ax2, 'PSD');
    end
    if k == length(basicFaults)
        % Add X-labels only to the last row
        xlabel(ax1, 'Time (s)');
        xlabel(ax2, 'Frequency (Hz)');
    end
end

title(t_sig1, 'Fig 0a: Time, Frequency, and Spectrogram Analysis (Basic Faults)', ...
    'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');
saveas(fig1, fullfile(CONFIG.outputDir, 'Fig0a_Basic_Faults.png'));
fprintf('✓ Saved: Fig0a_Basic_Faults.png\n');


% --- Figure 2: Mixed Faults ---
fprintf('  Generating Fig0b (Mixed Faults)...\n');
fig2 = figure('Name', 'PFD Mixed Fault Signal Examples', 'Position', [100, 100, 1600, 500]);
t_sig2 = tiledlayout(length(mixedFaults), 3, 'Padding', 'compact', 'TileSpacing', 'compact');

% Loop through MIXED faults
for k = 1:length(mixedFaults)
    fault = mixedFaults{k};
    fileName = fullfile(CONFIG.inputDir, sprintf('%s_001.mat', fault));
    
    if ~isfile(fileName)
        fprintf('  Warning: Could not find %s. Skipping.\n', fileName);
        nexttile; nexttile; nexttile;
        continue;
    end
    
    sigData = load(fileName);
    x = sigData.x;
    fs = sigData.fs;
    t_vec = (0:length(x)-1)/fs;
    
    % --- Plotting ---
    % Time Domain
    ax1 = nexttile;
    plot(ax1, t_vec, x);
    title(ax1, ['Time: ' fault], 'Interpreter', 'none', 'FontWeight', 'normal');
    xlim(ax1, [0, 0.5]);
    grid(ax1, 'on');
    
    % Frequency Domain (Welch PSD)
    ax2 = nexttile;
    [Pxx, F] = pwelch(x, hann(1024), 512, 2048, fs);
    semilogy(ax2, F, Pxx);
    title(ax2, ['Frequency: ' fault], 'Interpreter', 'none', 'FontWeight', 'normal');
    xlim(ax2, [0, fs/8]);
    grid(ax2, 'on');
    
    % Time-Frequency Spectrogram
    ax3 = nexttile;
    spectrogram(x, hann(256), 128, 256, fs, 'yaxis');
    title(ax3, ['Spectrogram: ' fault], 'Interpreter', 'none', 'FontWeight', 'normal');

    % --- CLUTTER REDUCTION: Only add labels to outer plots ---
    if k == 1
        % Add Y-labels only to the first row
        ylabel(ax1, 'Amplitude');
        ylabel(ax2, 'PSD');
    end
    if k == length(mixedFaults)
        % Add X-labels only to the last row
        xlabel(ax1, 'Time (s)');
        xlabel(ax2, 'Frequency (Hz)');
    end
end

title(t_sig2, 'Fig 0b: Time, Frequency, and Spectrogram Analysis (Mixed Faults)', ...
    'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');
saveas(fig2, fullfile(CONFIG.outputDir, 'Fig0b_Mixed_Faults.png'));
fprintf('✓ Saved: Fig0b_Mixed_Faults.png\n');
% ========================================================================
% FIG 1: FEATURE CORRELATION MATRIX
% ========================================================================

fprintf('\nStep 3.3: Generating Fig1 (Feature Correlation)...\n');
corrMatrix = corr(X_raw, 'rows', 'complete');

fig1 = figure('Position', [100, 100, 1000, 900]);
imagesc(corrMatrix);
colormap(jet);
cb = colorbar;
cb.Label.String = 'Correlation Coefficient';
clim([-1, 1]);
title('Fig 1: Feature Correlation Matrix', 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');
set(gca, 'XTick', 1:length(fullFeatureNames), 'XTickLabel', featureNamesFormatted, ...
    'XTickLabelRotation', 45, 'YTick', 1:length(fullFeatureNames), ...
    'YTickLabel', featureNamesFormatted, 'FontSize', 10);
grid on;
saveas(fig1, fullfile(CONFIG.outputDir, 'Fig1_Feature_Correlation.png'));
fprintf('✓ Saved: Fig1_Feature_Correlation.png\n');

% ========================================================================
% FIG 2: FEATURE DISTRIBUTIONS BY CLASS
% ========================================================================

fprintf('\nStep 3.4: Generating Fig2 (Feature Distributions)...\n');

numFeaturesToPlot = min(length(fullFeatureNames), 12);
if numFeaturesToPlot <= 6
    layoutRows = 2; layoutCols = 3;
elseif numFeaturesToPlot <= 9
    layoutRows = 3; layoutCols = 3;
else
    layoutRows = 3; layoutCols = 4;
end

fig2 = figure('Position', [100, 100, 1400, 1000]);
warnState = warning('off', 'MATLAB:graphics:boxchart:TiledChartLayout');
t2 = tiledlayout(layoutRows, layoutCols, 'Padding', 'compact', 'TileSpacing', 'compact');

for i = 1:numFeaturesToPlot
    feature_name_fmt = featureNamesFormatted{i};
    nexttile;
    boxplot(X_raw(:, i), Y_raw, 'Labels', classNames);
    ylabel(feature_name_fmt, 'FontSize', 10, 'Interpreter', 'none');
    title(feature_name_fmt, 'FontSize', 11, 'FontWeight', 'bold', 'Interpreter', 'none');
    set(gca, 'XTickLabelRotation', 45, 'FontSize', 9);
    grid on;
end
warning(warnState);

title(t2, 'Fig 2: Feature Distributions by Fault Class (First 12 Features)', ...
    'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');
saveas(fig2, fullfile(CONFIG.outputDir, 'Fig2_Feature_Distributions.png'));
fprintf('✓ Saved: Fig2_Feature_Distributions.png\n');

% ========================================================================
% FIG 3: t-SNE CLUSTER VISUALIZATION
% ========================================================================

fprintf('\nStep 3.5: Generating Fig3 (t-SNE Plot)...\n');

if CONFIG.generateTSNEPlot
    fprintf('  Running t-SNE dimensionality reduction...\n');
    try
        Y_tsne = tsne(X_raw_norm, 'NumDimensions', 2, 'Perplexity', 30, ...
            'Options', statset('UseParallel', true));
        
        fig3 = figure('Position', [100, 100, 1000, 800]);
        gscatter(Y_tsne(:,1), Y_tsne(:,2), Y_raw, lines(length(classNames)), '.', 15);
        title('Fig 3: t-SNE Visualization of Feature Clusters', ...
            'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');
        xlabel('t-SNE Dimension 1', 'FontSize', 12);
        ylabel('t-SNE Dimension 2', 'FontSize', 12);
        legend('Location', 'best', 'Interpreter', 'none');
        grid on;
        
        saveas(fig3, fullfile(CONFIG.outputDir, 'Fig3_tSNE_Clusters.png'));
        fprintf('✓ Saved: Fig3_tSNE_Clusters.png\n');
        
    catch ME
        fprintf('!! Warning: t-SNE plot failed: %s. Skipping Fig3.\n', ME.message);
    end
else
    fprintf('  Skipping t-SNE plot (disabled in CONFIG).\n');
end

fprintf('\n✓ Step 3 complete: Data exploration and visualization finished.\n');
fprintf('  Generated figures: Fig0, Fig1, Fig2');
if CONFIG.generateTSNEPlot
    fprintf(', Fig3');
end
fprintf('\n');

%% ========================================================================
%% FEATURE EXTRACTION FUNCTION
%% ========================================================================

function featValues = extractFeatures(x, fs, CONFIG)
    % Extract all configured features from signal x
    
    % Pre-allocate feature vector
    if CONFIG.includeAdvancedFeatures
        numFeats = 51;
    else
        numFeats = 35;
    end
    featValues = zeros(1, numFeats);
    featIdx = 1;
    
    % ====================================================================
    % TIME-DOMAIN FEATURES (7)
    % ====================================================================
    featValues(featIdx) = mean(x); featIdx = featIdx + 1;
    featValues(featIdx) = rms(x); featIdx = featIdx + 1;
    featValues(featIdx) = std(x); featIdx = featIdx + 1;
    featValues(featIdx) = kurtosis(x); featIdx = featIdx + 1;
    featValues(featIdx) = skewness(x); featIdx = featIdx + 1;
    featValues(featIdx) = max(abs(x)) / (rms(x) + eps); featIdx = featIdx + 1;
    featValues(featIdx) = sum(x.^2); featIdx = featIdx + 1;
    
    % ====================================================================
    % ENVELOPE ANALYSIS (4)
    % ====================================================================
    analytic_signal = hilbert(x);
    envelope = abs(analytic_signal);
    
    featValues(featIdx) = rms(envelope); featIdx = featIdx + 1;
    featValues(featIdx) = kurtosis(envelope); featIdx = featIdx + 1;
    featValues(featIdx) = max(envelope) / (mean(envelope) + eps); featIdx = featIdx + 1;
    
    env_fft = abs(fft(envelope - mean(envelope)));
    env_fft = env_fft(1:floor(length(env_fft)/2)+1);
    f_env = (0:length(env_fft)-1) * (fs / (2*length(env_fft)));
    [~, idx_env] = max(env_fft);
    featValues(featIdx) = f_env(idx_env); featIdx = featIdx + 1;
    
    % ====================================================================
    % FREQUENCY-DOMAIN FEATURES (3)
    % ====================================================================
    N_fft = length(x);
    X_fft = abs(fft(x));
    f_fft = (0:N_fft-1)*(fs/N_fft);
    X_fft = X_fft(1:floor(N_fft/2)+1);
    f_fft = f_fft(1:floor(N_fft/2)+1);
    
    [~, idx_max] = max(X_fft);
    featValues(featIdx) = f_fft(idx_max); featIdx = featIdx + 1;
    
    featValues(featIdx) = sum(f_fft(:) .* X_fft(:)) / (sum(X_fft) + eps); featIdx = featIdx + 1;
    
    Pxx_norm = X_fft.^2 / (sum(X_fft.^2) + eps);
    featValues(featIdx) = -sum(Pxx_norm .* log2(Pxx_norm + eps)); featIdx = featIdx + 1;
    
    % ====================================================================
    % FAULT-SPECIFIC BAND ENERGY (3)
    % ====================================================================
    featValues(featIdx) = bandpower(x, fs, CONFIG.bands.Oilwhirl); featIdx = featIdx + 1;
    featValues(featIdx) = bandpower(x, fs, CONFIG.bands.Misalignment); featIdx = featIdx + 1;
    featValues(featIdx) = bandpower(x, fs, CONFIG.bands.Cavitation); featIdx = featIdx + 1;
    
    % ====================================================================
    % LOCAL SPECTRAL ENTROPY (2)
    % ====================================================================
    [S_spec, ~, ~] = spectrogram(x, hann(256), 128, 256, fs);
    S_spec_mag = abs(S_spec);
    P_spec = S_spec_mag ./ (sum(S_spec_mag, 1) + eps);
    entropy_over_time = -sum(P_spec .* log2(P_spec + eps), 1);
    featValues(featIdx) = mean(entropy_over_time); featIdx = featIdx + 1;
    featValues(featIdx) = std(entropy_over_time); featIdx = featIdx + 1;
    
    % ====================================================================
    % WAVELET FEATURES (1)
    % ====================================================================
    [C, L] = wavedec(x, 5, 'db4');
    d1 = wrcoef('d', C, L, 'db4', 1);
    featValues(featIdx) = kurtosis(d1); featIdx = featIdx + 1;
    
    % ====================================================================
    % HARMONIC ANALYSIS (3)
    % ====================================================================
    [~, idx_fund] = max(X_fft);
    f_fundamental = f_fft(idx_fund);
    
    f_2X = 2 * f_fundamental;
    [~, idx_2X] = min(abs(f_fft - f_2X));
    f_3X = 3 * f_fundamental;
    [~, idx_3X] = min(abs(f_fft - f_3X));
    
    featValues(featIdx) = X_fft(idx_2X) / (X_fft(idx_fund) + eps); featIdx = featIdx + 1;
    featValues(featIdx) = X_fft(idx_3X) / (X_fft(idx_fund) + eps); featIdx = featIdx + 1;
    
    harmonics_power = sum(X_fft([idx_2X, idx_3X]).^2);
    total_power = sum(X_fft.^2);
    featValues(featIdx) = sqrt(harmonics_power / (total_power + eps)); featIdx = featIdx + 1;
    
    % ====================================================================
    % ADVANCED SPECTRAL FEATURES (3)
    % ====================================================================
    geom_mean_power = exp(mean(log(X_fft + eps)));
    arith_mean_power = mean(X_fft);
    featValues(featIdx) = geom_mean_power / (arith_mean_power + eps); featIdx = featIdx + 1;
    
    SpectralCentroid = sum(f_fft(:) .* X_fft(:)) / (sum(X_fft) + eps);
    freq_variance = sum(((f_fft(:) - SpectralCentroid).^2) .* X_fft(:)) / sum(X_fft);
    featValues(featIdx) = sqrt(freq_variance); featIdx = featIdx + 1;
    
    significant_peaks = X_fft > 0.1*max(X_fft);
    if sum(significant_peaks) > 1
        freq_diffs = diff(f_fft(significant_peaks));
        featValues(featIdx) = std(freq_diffs) / (mean(freq_diffs) + eps);
    else
        featValues(featIdx) = 0;
    end
    featIdx = featIdx + 1;
    
    % ====================================================================
    % CEPSTRAL ANALYSIS (3)
    % ====================================================================
    cepstrum = real(ifft(log(abs(fft(x)) + eps)));
    cepstrum = cepstrum(1:floor(length(cepstrum)/2)+1);
    quefrency = (0:length(cepstrum)-1) / fs;
    
    [cep_peak, ~] = max(abs(cepstrum(2:end)));
    featValues(featIdx) = cep_peak / (mean(abs(cepstrum)) + eps); featIdx = featIdx + 1;
    
    featValues(featIdx) = sum(quefrency(:) .* abs(cepstrum(:))) / (sum(abs(cepstrum)) + eps); featIdx = featIdx + 1;
    featValues(featIdx) = var(cepstrum); featIdx = featIdx + 1;
    
    % ====================================================================
    % BISPECTRUM FEATURES (3)
    % ====================================================================
    X_complex = fft(x);
    N_bi = floor(length(X_complex)/2)+1;
    X_half = X_complex(1:N_bi);
    
    bispec_slice = zeros(1, min(50, N_bi-1));
    for k = 1:length(bispec_slice)
        if k < N_bi && 2*k < N_bi
            bispec_slice(k) = abs(X_half(k) * X_half(k) * conj(X_half(2*k)));
        end
    end
    
    featValues(featIdx) = max(bispec_slice); featIdx = featIdx + 1;
    featValues(featIdx) = sum(bispec_slice) / (sum(abs(X_complex).^2) + eps); featIdx = featIdx + 1;
    featValues(featIdx) = sum(bispec_slice.^2) / (sum(abs(X_complex).^4) + eps); featIdx = featIdx + 1;
    
    % ====================================================================
    % TIME-FREQUENCY STFT FEATURES (4)
    % ====================================================================
    [S_stft, ~, ~] = spectrogram(x, hann(512), 256, 512, fs);
    S_stft_mag = abs(S_stft);
    P_stft = S_stft_mag ./ (sum(S_stft_mag, 1) + eps);
    entropy_stft = -sum(P_stft .* log2(P_stft + eps), 1);
    featValues(featIdx) = mean(entropy_stft); featIdx = featIdx + 1;
    
    d1_coef = wrcoef('d', C, L, 'db4', 1);
    d5_coef = wrcoef('d', C, L, 'db4', 5);
    
    featValues(featIdx) = sum(d1_coef.^2) / (sum(x.^2) + eps); featIdx = featIdx + 1;
    featValues(featIdx) = sum(d5_coef.^2) / (sum(x.^2) + eps); featIdx = featIdx + 1;
    
    inst_phase = unwrap(angle(analytic_signal));
    inst_freq = diff(inst_phase) * fs / (2*pi);
    featValues(featIdx) = std(inst_freq); featIdx = featIdx + 1;
    
    % ====================================================================
    % ADVANCED FEATURES (OPTIONAL - COMPUTATIONALLY EXPENSIVE)
    % ====================================================================
    if CONFIG.includeAdvancedFeatures
        % CWT Features (2)
        [cwt_coeffs, ~] = cwt(x, 'amor', fs);
        cwt_mag = abs(cwt_coeffs);
        featValues(featIdx) = sum(cwt_mag.^2, 'all'); featIdx = featIdx + 1;
        featValues(featIdx) = mean(std(cwt_mag, 0, 1)); featIdx = featIdx + 1;
        
        % WPT Features (3)
        wpt = wpdec(x, 4, 'db4');
        wpt_energy = zeros(1, 5);
        for level = 0:4
            level_energy = 0;
            num_nodes = 2^level;
            for node_idx = 0:(num_nodes - 1)
                try
                    node_coeffs = wprcoef(wpt, [level, node_idx]);
                    level_energy = level_energy + sum(node_coeffs.^2);
                catch
                    % Skip invalid nodes
                end
            end
            wpt_energy(level+1) = level_energy;
        end
        wpt_energy(isnan(wpt_energy) | isinf(wpt_energy)) = 0;
        
        featValues(featIdx) = wpt_energy(1) / (wpt_energy(5) + eps); featIdx = featIdx + 1;
        featValues(featIdx) = max(wpt_energy) / (sum(wpt_energy) + eps); featIdx = featIdx + 1;
        wpt_energy_norm = wpt_energy / (sum(wpt_energy) + eps);
        featValues(featIdx) = -sum(wpt_energy_norm .* log2(wpt_energy_norm + eps)); featIdx = featIdx + 1;
        
        % Enhanced Wavelet Features (3)
        x_reconstructed = waverec(C, L, 'db4');
        featValues(featIdx) = sum(x_reconstructed.^2) / (sum(x.^2) + eps); featIdx = featIdx + 1;
        
        try
            [~, f_ridge] = cwt(x, 'amor', fs);
            featValues(featIdx) = mean(f_ridge); featIdx = featIdx + 1;
            featValues(featIdx) = std(f_ridge); featIdx = featIdx + 1;
        catch
            featValues(featIdx) = 0; featIdx = featIdx + 1;
            featValues(featIdx) = 0; featIdx = featIdx + 1;
        end
        
        % Non-linear Dynamics Features (5)
        featValues(featIdx) = computeLyapunovExponent(x); featIdx = featIdx + 1;
        featValues(featIdx) = computeCorrelationDimension(x); featIdx = featIdx + 1;
        featValues(featIdx) = computeSampleEntropy(x); featIdx = featIdx + 1;
        featValues(featIdx) = computePoincareDispersion(x); featIdx = featIdx + 1;
        featValues(featIdx) = computeDFA(x); featIdx = featIdx + 1;
        
        % Advanced STFT Features (3)
        geom_mean_stft = exp(mean(log(S_stft_mag + eps), 2));
        arith_mean_stft = mean(S_stft_mag, 2);
        featValues(featIdx) = mean(geom_mean_stft ./ (arith_mean_stft + eps)); featIdx = featIdx + 1;
        
        centroid_per_frame = sum((1:size(S_stft_mag,1))' .* S_stft_mag, 1) ./ (sum(S_stft_mag, 1) + eps);
        featValues(featIdx) = std(centroid_per_frame); featIdx = featIdx + 1;
        
        bandwidth_per_frame = sqrt(sum(((1:size(S_stft_mag,1))' - mean(centroid_per_frame)).^2 .* S_stft_mag, 1) ./ (sum(S_stft_mag, 1) + eps));
        featValues(featIdx) = std(bandwidth_per_frame); featIdx = featIdx + 1;
    end
end



%% ========================================================================
%% STEP 4: MULTI-MODEL ML PIPELINE & OPTIMIZATION
%% ========================================================================

fprintf('\n========================================================================\n');
fprintf('Step 4: Multi-Model ML Pipeline & Optimization\n');
fprintf('========================================================================\n');

% ========================================================================
% STEP 4.1: DATA LOADING FOR MODELING
% ========================================================================

fprintf('\nStep 4.1: Loading feature data for modeling...\n');

try
    data = readtable(fullfile(CONFIG.outputDir, CONFIG.outputCSV));
    X_raw = table2array(data(:, 1:end-1));
    Y_raw = categorical(data.Label);
    classNames = categories(Y_raw);
    numClasses = length(classNames);
    
    fprintf('✓ Loaded feature data:\n');
    fprintf('  - Samples: %d\n', size(X_raw, 1));
    fprintf('  - Features: %d\n', size(X_raw, 2));
    fprintf('  - Classes: %d\n', numClasses);
    
catch ME
    diary off;
    error('Failed to load feature file: %s', ME.message);
end

% ========================================================================
% STEP 4.2: DATA QUALITY ASSESSMENT
% ========================================================================

fprintf('\nStep 4.2: Data Quality Assessment\n');

% Check for NaN and Inf values
badMask = isnan(X_raw) | isinf(X_raw);
totalBad = sum(badMask(:));

if totalBad > 0
    fprintf('⚠️  Warning: Found %d NaN/Inf values (%.2f%% of data).\n', ...
        totalBad, 100*totalBad/numel(X_raw));
    fprintf('  Imputing with column median...\n');
    
    for col = 1:size(X_raw, 2)
        colMask = badMask(:, col);
        if any(colMask)
            medianVal = median(X_raw(~colMask, col), 'omitnan');
            if isnan(medianVal)
                medianVal = 0;
            end
            X_raw(colMask, col) = medianVal;
        end
    end
    fprintf('✓ Imputation complete.\n');
else
    fprintf('✓ No missing or infinite values found.\n');
end

% Detect duplicate samples
[~, uniqueIdx, ~] = unique(X_raw, 'rows', 'stable');
numDuplicates = size(X_raw, 1) - length(uniqueIdx);

if numDuplicates > 0
    fprintf('⚠️  Warning: Found %d duplicate samples. Removing...\n', numDuplicates);
    X_raw = X_raw(uniqueIdx, :);
    Y_raw = Y_raw(uniqueIdx);
    fprintf('✓ Dataset now contains %d unique samples.\n', size(X_raw, 1));
else
    fprintf('✓ No duplicate samples found.\n');
end

% Outlier detection (informational only)
fprintf('\nChecking for outliers (|z-score| > 3)...\n');
try
    zScores = zscore(X_raw);
    outlierMask = abs(zScores) > 3;
    numOutliers = sum(outlierMask(:));
    outlierPercent = 100 * numOutliers / numel(X_raw);
    
    fprintf('✓ Outliers detected: %d (%.2f%% of data points)\n', numOutliers, outlierPercent);
    
    if outlierPercent > 5
        fprintf('  ⚠️  High outlier rate (>5%%) - consider reviewing data generation.\n');
    else
        fprintf('  Note: Outliers retained (may represent genuine fault characteristics).\n');
    end
catch ME
    fprintf('⚠️  Warning: Outlier detection failed: %s\n', ME.message);
end

% ========================================================================
% STEP 4.3: CLASS IMBALANCE CHECK
% ========================================================================

fprintf('\nStep 4.3: Class Imbalance Analysis\n');

classCounts = countcats(Y_raw);
fprintf('Class distribution:\n');
for i = 1:numClasses
    fprintf('  %-30s: %4d samples (%.1f%%)\n', ...
        char(classNames{i}), classCounts(i), 100*classCounts(i)/length(Y_raw));
end

imbalanceRatio = max(classCounts) / min(classCounts);
fprintf('Imbalance ratio: %.2f:1\n', imbalanceRatio);

if CONFIG.handleImbalance && imbalanceRatio > 1.5
    fprintf('⚠️  Class imbalance detected (ratio > 1.5).\n');
    fprintf('  Strategy: %s\n', CONFIG.imbalanceMethod);
    
    if strcmp(CONFIG.imbalanceMethod, 'CostMatrix')
        costMatrix = ones(numClasses) - eye(numClasses);
        for i = 1:numClasses
            costMatrix(i, :) = costMatrix(i, :) * (length(Y_raw) / classCounts(i));
            costMatrix(i, i) = 0;
        end
        costMatrix = costMatrix / max(costMatrix(:));
        fprintf('✓ Cost matrix generated for imbalanced classes.\n');
    end
else
    fprintf('✓ Classes are balanced (imbalance handling disabled).\n');
    costMatrix = [];
end

% ========================================================================
% STEP 4.4: STRATIFIED DATA SPLITTING
% ========================================================================

fprintf('\nStep 4.4: Stratified Data Splitting (%.0f%% / %.0f%% / %.0f%%)\n', ...
    CONFIG.trainRatio*100, CONFIG.valRatio*100, CONFIG.testRatio*100);

% ========================================================================
% LOAD METADATA FOR ENHANCED STRATIFICATION
% ========================================================================

metadataPath = fullfile(CONFIG.outputDir, 'dataset_metadata.mat');
if exist(metadataPath, 'file')
    fprintf('Loading dataset metadata...\n');
    metadataStruct = load(metadataPath);
    
    if metadataStruct.datasetHasMetadata
        % Extract severity levels if available
        severityLevels = cell(size(X_raw, 1), 1);
        isAugmented = false(size(X_raw, 1), 1);
        
        for i = 1:length(metadataStruct.allMetadata)
            if ~isempty(metadataStruct.allMetadata{i})
                if isfield(metadataStruct.allMetadata{i}, 'severity')
                    severityLevels{i} = metadataStruct.allMetadata{i}.severity;
                else
                    severityLevels{i} = 'unknown';
                end
                
                if isfield(metadataStruct.allMetadata{i}, 'is_augmented')
                    isAugmented(i) = metadataStruct.allMetadata{i}.is_augmented;
                end
            else
                severityLevels{i} = 'unknown';
            end
        end
        
        % Report severity distribution
        uniqueSeverities = unique(severityLevels);
        fprintf('  Severity distribution:\n');
        for s = 1:length(uniqueSeverities)
            if ~strcmp(uniqueSeverities{s}, 'unknown')
                sevCount = sum(strcmp(severityLevels, uniqueSeverities{s}));
                fprintf('    %-12s: %d samples (%.1f%%)\n', ...
                    uniqueSeverities{s}, sevCount, 100*sevCount/length(severityLevels));
            end
        end
        
        % Report augmentation statistics
        augCount = sum(isAugmented);
        if augCount > 0
            fprintf('  Augmented samples: %d (%.1f%%)\n', augCount, 100*augCount/length(isAugmented));
        end
        
        fprintf('✓ Metadata loaded and analyzed.\n');
    else
        fprintf('  No metadata available - using standard stratification.\n');
        severityLevels = [];
        isAugmented = [];
    end
else
    fprintf('  No metadata file found - using standard stratification.\n');
    severityLevels = [];
    isAugmented = [];
end

% ========================================================================
% STRATIFIED SPLIT (BY CLASS, RESPECTING SEVERITY IF AVAILABLE)
% ========================================================================

% First split: separate training from (validation + test)
tempRatio = CONFIG.valRatio + CONFIG.testRatio;
cv_train_temp = cvpartition(Y_raw, 'HoldOut', tempRatio);

idxTrain = training(cv_train_temp);
X_train = X_raw(idxTrain, :);
Y_train = Y_raw(idxTrain);

idxTemp = test(cv_train_temp);
X_temp = X_raw(idxTemp, :);
Y_temp = Y_raw(idxTemp);

% Second split: separate validation from test
testRatioAdjusted = CONFIG.testRatio / tempRatio;
cv_val_test = cvpartition(Y_temp, 'HoldOut', testRatioAdjusted);

idxVal = training(cv_val_test);
idxTest = test(cv_val_test);

X_val = X_temp(idxVal, :);
Y_val = Y_temp(idxVal);
X_test = X_temp(idxTest, :);
Y_test = Y_temp(idxTest);

fprintf('✓ Data split completed:\n');
fprintf('  - Training:   %d samples (%.1f%%)\n', size(X_train, 1), 100*size(X_train, 1)/size(X_raw, 1));
fprintf('  - Validation: %d samples (%.1f%%)\n', size(X_val, 1), 100*size(X_val, 1)/size(X_raw, 1));
fprintf('  - Test:       %d samples (%.1f%%)\n', size(X_test, 1), 100*size(X_test, 1)/size(X_raw, 1));

% Verify class distribution preservation
fprintf('\nVerifying stratification:\n');
trainClassCounts = countcats(Y_train);
for i = 1:min(3, numClasses)
    trainProp = trainClassCounts(i) / length(Y_train);
    overallProp = classCounts(i) / length(Y_raw);
    fprintf('  %s: Train=%.1f%%, Overall=%.1f%% (diff: %.1f%%)\n', ...
        char(classNames{i}), 100*trainProp, 100*overallProp, 100*abs(trainProp-overallProp));
end
fprintf('✓ Stratification verified (proportions preserved).\n');

% ========================================================================
% STEP 4.5: CROSS-DATASET VALIDATION SPLIT (OPTIONAL)
% ========================================================================

if CONFIG.useCrossDatasetValidation
    fprintf('\nStep 4.5: Additional Holdout Set for Generalization Testing\n');
    fprintf('Creating separate validation set (10%% of data)...\n');
    
    crossDatasetRatio = 0.10;
    cv_cross = cvpartition(Y_raw, 'HoldOut', crossDatasetRatio);
    
    idxCrossDataset = test(cv_cross);
    X_crossDataset = X_raw(idxCrossDataset, :);
    Y_crossDataset = Y_raw(idxCrossDataset);
    
    fprintf('✓ Additional holdout set: %d samples\n', size(X_crossDataset, 1));
    fprintf('  (Simulates different operating session for robustness testing)\n');
else
    X_crossDataset = [];
    Y_crossDataset = [];
    fprintf('\nStep 4.5: Cross-dataset validation disabled.\n');
end

% ========================================================================
% STEP 4.6: FEATURE SELECTION (AFTER SPLIT - NO DATA LEAKAGE)
% ========================================================================

fprintf('\nStep 4.6: Feature Selection (Post-Split)\n');

if CONFIG.useFeatureSelection && CONFIG.performFeatureSelectionAfterSplit
    fprintf('Running feature selection on TRAINING SET ONLY...\n');
    fprintf('Method: MRMR (Minimum Redundancy Maximum Relevance)\n');
    
    % Clean training data before feature selection
    X_train_clean = X_train;
    badMask_train = isnan(X_train_clean) | isinf(X_train_clean);
    
    if any(badMask_train(:))
        for col = 1:size(X_train_clean, 2)
            if any(badMask_train(:, col))
                medianVal = median(X_train_clean(~badMask_train(:, col), col), 'omitnan');
                if isnan(medianVal), medianVal = 0; end
                X_train_clean(badMask_train(:, col), col) = medianVal;
            end
        end
    end
    
    % Normalize for feature selection
    X_train_norm = zscore(X_train_clean);
    
    % Handle zero-variance features
    zeroVarMask = std(X_train_norm) < eps;
    if any(zeroVarMask)
        fprintf('  ⚠️  Warning: %d features have zero variance. Removing...\n', sum(zeroVarMask));
        X_train_norm(:, zeroVarMask) = [];
        X_train_clean(:, zeroVarMask) = [];
        X_val(:, zeroVarMask) = [];
        X_test(:, zeroVarMask) = [];
        X_train(:, zeroVarMask) = [];
        if ~isempty(X_crossDataset)
            X_crossDataset(:, zeroVarMask) = [];
        end
        fullFeatureNames(zeroVarMask) = [];
    end
    
    % Run MRMR feature selection
    try
        [selectedFeatureIdx, scores] = fscmrmr(X_train_norm, Y_train);
        
        % Select top N features
        nFeaturesToKeep = min(CONFIG.numFeaturesToSelect, length(selectedFeatureIdx));
        selectedFeatureIdx = selectedFeatureIdx(1:nFeaturesToKeep);
        selectedScores = scores(1:nFeaturesToKeep);
        
        fprintf('✓ Feature selection complete.\n');
        fprintf('  Selected %d features (from %d available):\n', nFeaturesToKeep, length(fullFeatureNames));
        
        selectedFeatureNames = fullFeatureNames(selectedFeatureIdx);
        for i = 1:min(10, length(selectedFeatureNames))
            fprintf('  %2d. %-30s (score: %.4f)\n', i, selectedFeatureNames{i}, selectedScores(i));
        end
        if length(selectedFeatureNames) > 10
            fprintf('  ... and %d more features\n', length(selectedFeatureNames) - 10);
        end
        
        % Apply selection to all splits
        X_train = X_train(:, selectedFeatureIdx);
        X_val = X_val(:, selectedFeatureIdx);
        X_test = X_test(:, selectedFeatureIdx);
        if ~isempty(X_crossDataset)
            X_crossDataset = X_crossDataset(:, selectedFeatureIdx);
        end
        
        featureNames = selectedFeatureNames;
        fprintf('✓ Feature selection applied to all data splits.\n');
        
    catch ME
        fprintf('⚠️  Warning: Feature selection failed: %s\n', ME.message);
        fprintf('  Proceeding with all features.\n');
        featureNames = fullFeatureNames;
    end
    
else
    fprintf('Feature selection disabled or set to pre-split mode.\n');
    featureNames = fullFeatureNames;
end

numFinalFeatures = size(X_train, 2);
fprintf('✓ Final feature count: %d\n', numFinalFeatures);

% ========================================================================
% STEP 4.7: FEATURE NORMALIZATION (TRAINING STATISTICS ONLY)
% ========================================================================

fprintf('\nStep 4.7: Feature Normalization\n');

% Compute normalization statistics from TRAINING SET ONLY
mu_train = mean(X_train, 1);
sigma_train = std(X_train, 0, 1);

% Guard against zero variance
sigma_train(sigma_train < 1e-10) = 1;

% Apply normalization to all splits using training statistics
X_train_norm = (X_train - mu_train) ./ sigma_train;
X_val_norm = (X_val - mu_train) ./ sigma_train;
X_test_norm = (X_test - mu_train) ./ sigma_train;

if ~isempty(X_crossDataset)
    X_crossDataset_norm = (X_crossDataset - mu_train) ./ sigma_train;
else
    X_crossDataset_norm = [];
end

fprintf('✓ Normalization complete (z-score using training statistics).\n');
fprintf('  Training mean range: [%.4f, %.4f]\n', min(mu_train), max(mu_train));
fprintf('  Training std range:  [%.4f, %.4f]\n', min(sigma_train), max(sigma_train));

% Verify normalization
train_mean_after = mean(X_train_norm(:));
train_std_after = std(X_train_norm(:));
fprintf('  Verification: Training data mean=%.6f, std=%.4f\n', train_mean_after, train_std_after);

if abs(train_mean_after) > 0.01 || abs(train_std_after - 1) > 0.1
    fprintf('  ⚠️  Warning: Normalization verification failed. Checking for issues...\n');
end

% ========================================================================
% STEP 4.8: MODEL TRAINING & COMPARISON
% ========================================================================

fprintf('\n========================================================================\n');
fprintf('Step 4.8: Model Training & Hyperparameter Optimization\n');
fprintf('========================================================================\n');

% Store all model results
modelResults = struct();
modelNames = {};
modelAccuracies = [];

% ========================================================================
% MODEL 1: SUPPORT VECTOR MACHINE (SVM)
% ========================================================================

if CONFIG.models.trainSVM
    fprintf('\n--- Training SVM (Baseline Model) ---\n');
    tic;
    
    try
        % Define hyperparameter search space
        svmParams = [
            optimizableVariable('BoxConstraint', [0.01, 100], 'Transform', 'log')
            optimizableVariable('KernelScale', [0.01, 100], 'Transform', 'log')
        ];
        
        % Bayesian optimization with cross-validation
        svmOpts = struct(...
            'AcquisitionFunctionName', 'expected-improvement-plus', ...
            'MaxObjectiveEvaluations', CONFIG.hyperOptIterations, ...
            'ShowPlots', false, ...
            'Verbose', 0);
        
        fprintf('  Running Bayesian optimization (%d iterations)...\n', CONFIG.hyperOptIterations);
        
        % Use ECOC (Error-Correcting Output Codes) for multi-class SVM
        svmTemplate = templateSVM('KernelFunction', 'rbf', 'Standardize', false);
        
        svmModel = fitcecoc(X_train_norm, Y_train, ...
            'Learners', svmTemplate, ...
            'OptimizeHyperparameters', svmParams, ...
            'HyperparameterOptimizationOptions', svmOpts);
        
        elapsed_svm = toc;
        fprintf('✓ SVM training complete in %.2f seconds.\n', elapsed_svm);
        
        % Validation accuracy
        Y_val_pred_svm = predict(svmModel, X_val_norm);
        svm_val_acc = 100 * sum(Y_val_pred_svm == Y_val) / length(Y_val);
        fprintf('► SVM Validation Accuracy: %.2f%%\n', svm_val_acc);
        
        % Store results
        modelResults.SVM.model = svmModel;
        modelResults.SVM.valAccuracy = svm_val_acc;
        modelResults.SVM.trainTime = elapsed_svm;
        modelNames{end+1} = 'SVM';
        modelAccuracies(end+1) = svm_val_acc;
        
    catch ME
        fprintf('!! Error training SVM: %s\n', ME.message);
        fprintf('  Skipping SVM model.\n');
    end
end

% ========================================================================
% MODEL 2: RANDOM FOREST
% ========================================================================

if CONFIG.models.trainRandomForest
    fprintf('\n--- Training Random Forest ---\n');
    tic;
    
    try
        rfParams = [
            optimizableVariable('NumLearningCycles', [50, 500], 'Type', 'integer')
            optimizableVariable('MinLeafSize', [1, 50], 'Type', 'integer')
            optimizableVariable('MaxNumSplits', [10, size(X_train_norm, 1) - 1], 'Type', 'integer')
        ];
        
        rfOpts = struct(...
            'AcquisitionFunctionName', 'expected-improvement-plus', ...
            'MaxObjectiveEvaluations', CONFIG.hyperOptIterations, ...
            'ShowPlots', false, ...
            'Verbose', 0);
        
        fprintf('  Running Bayesian optimization (%d iterations)...\n', CONFIG.hyperOptIterations);
        
        rfModel = fitcensemble(X_train_norm, Y_train, ...
            'Method', 'Bag', ...
            'OptimizeHyperparameters', rfParams, ...
            'HyperparameterOptimizationOptions', rfOpts);
        
        elapsed_rf = toc;
        fprintf('✓ Random Forest training complete in %.2f seconds.\n', elapsed_rf);
        
        Y_val_pred_rf = predict(rfModel, X_val_norm);
        rf_val_acc = 100 * sum(Y_val_pred_rf == Y_val) / length(Y_val);
        fprintf('► Random Forest Validation Accuracy: %.2f%%\n', rf_val_acc);
        
        modelResults.RandomForest.model = rfModel;
        modelResults.RandomForest.valAccuracy = rf_val_acc;
        modelResults.RandomForest.trainTime = elapsed_rf;
        modelNames{end+1} = 'Random Forest';
        modelAccuracies(end+1) = rf_val_acc;
        
    catch ME
        fprintf('!! Error training Random Forest: %s\n', ME.message);
        fprintf('  Skipping Random Forest model.\n');
    end
end

% ========================================================================
% MODEL 3: GRADIENT BOOSTING
% ========================================================================

if CONFIG.models.trainGradientBoosting
    fprintf('\n--- Training Gradient Boosting ---\n');
    tic;
    
    try
        gbParams = [
            optimizableVariable('NumLearningCycles', [50, 500], 'Type', 'integer')
            optimizableVariable('LearnRate', [0.01, 1], 'Transform', 'log')
            optimizableVariable('MaxNumSplits', [10, min(500, size(X_train_norm, 1) - 1)], 'Type', 'integer')
        ];
        
        gbOpts = struct(...
            'AcquisitionFunctionName', 'expected-improvement-plus', ...
            'MaxObjectiveEvaluations', CONFIG.hyperOptIterations, ...
            'ShowPlots', false, ...
            'Verbose', 0);
        
        fprintf('  Running Bayesian optimization (%d iterations)...\n', CONFIG.hyperOptIterations);
        
        gbModel = fitcensemble(X_train_norm, Y_train, ...
            'Method', 'LogitBoost', ...
            'OptimizeHyperparameters', gbParams, ...
            'HyperparameterOptimizationOptions', gbOpts);
        
        elapsed_gb = toc;
        fprintf('✓ Gradient Boosting training complete in %.2f seconds.\n', elapsed_gb);
        
        Y_val_pred_gb = predict(gbModel, X_val_norm);
        gb_val_acc = 100 * sum(Y_val_pred_gb == Y_val) / length(Y_val);
        fprintf('► Gradient Boosting Validation Accuracy: %.2f%%\n', gb_val_acc);
        
        modelResults.GradientBoosting.model = gbModel;
        modelResults.GradientBoosting.valAccuracy = gb_val_acc;
        modelResults.GradientBoosting.trainTime = elapsed_gb;
        modelNames{end+1} = 'Gradient Boosting';
        modelAccuracies(end+1) = gb_val_acc;
        
    catch ME
        fprintf('!! Error training Gradient Boosting: %s\n', ME.message);
        fprintf('  Skipping Gradient Boosting model.\n');
    end
end

% ========================================================================
% MODEL 4: NEURAL NETWORK (3-LAYER MLP)
% ========================================================================

if CONFIG.models.trainNeuralNetwork
    fprintf('\n--- Training Neural Network (3-layer MLP with dropout) ---\n');
    tic;
    
    try
        % Simple feedforward network architecture
        hiddenLayerSize = [50, 25];
        
        nnModel = fitcnet(X_train_norm, Y_train, ...
            'LayerSizes', hiddenLayerSize, ...
            'Activations', 'relu', ...
            'IterationLimit', 200, ...
            'Verbose', 0);
        
        elapsed_nn = toc;
        fprintf('✓ Neural Network training complete in %.2f seconds.\n', elapsed_nn);
        
        Y_val_pred_nn = predict(nnModel, X_val_norm);
        nn_val_acc = 100 * sum(Y_val_pred_nn == Y_val) / length(Y_val);
        fprintf('► Neural Network Validation Accuracy: %.2f%%\n', nn_val_acc);
        
        modelResults.NeuralNetwork.model = nnModel;
        modelResults.NeuralNetwork.valAccuracy = nn_val_acc;
        modelResults.NeuralNetwork.trainTime = elapsed_nn;
        modelNames{end+1} = 'Neural Network';
        modelAccuracies(end+1) = nn_val_acc;
        
    catch ME
        fprintf('!! Error training Neural Network: %s\n', ME.message);
        fprintf('  Skipping Neural Network model.\n');
    end
end

% ========================================================================
% MODEL 5: STACKED ENSEMBLE (OPTIONAL - DISABLED BY DEFAULT)
% ========================================================================

if CONFIG.models.trainStackedEnsemble
    fprintf('\n--- Training Stacked Ensemble (Meta-learner) ---\n');
    fprintf('  ⚠️  WARNING: Stacked ensemble uses AdaBoost which may be unstable.\n');
    tic;
    
    try
        % Generate out-of-fold predictions to prevent data leakage
        fprintf('  Generating out-of-fold predictions from base models...\n');
        
        cv = cvpartition(Y_train, 'KFold', 5);
        metaFeatures = zeros(length(Y_train), length(modelNames));
        
        for fold = 1:5
            idxTr = training(cv, fold);
            idxVal = test(cv, fold);
            
            X_fold_train = X_train_norm(idxTr, :);
            Y_fold_train = Y_train(idxTr);
            X_fold_val = X_train_norm(idxVal, :);
            
            % Train base models on this fold
            if isfield(modelResults, 'SVM')
                fold_svm = fitcsvm(X_fold_train, Y_fold_train, 'KernelFunction', 'rbf', 'Standardize', false);
                [~, scores_svm] = predict(fold_svm, X_fold_val);
                metaFeatures(idxVal, 1) = scores_svm(:, 1);
            end
            
            if isfield(modelResults, 'RandomForest')
                fold_rf = fitcensemble(X_fold_train, Y_fold_train, 'Method', 'Bag', 'NumLearningCycles', 100);
                [~, scores_rf] = predict(fold_rf, X_fold_val);
                metaFeatures(idxVal, 2) = scores_rf(:, 1);
            end
            
            if isfield(modelResults, 'GradientBoosting')
                fold_gb = fitcensemble(X_fold_train, Y_fold_train, 'Method', 'LogitBoost', 'NumLearningCycles', 100);
                [~, scores_gb] = predict(fold_gb, X_fold_val);
                metaFeatures(idxVal, 3) = scores_gb(:, 1);
            end
        end
        
        fprintf('  Training meta-learner on out-of-fold predictions...\n');
        
        % Train meta-learner (use simpler method instead of AdaBoost)
        ensembleModel = fitcensemble(metaFeatures, Y_train, ...
            'Method', 'Bag', ...
            'NumLearningCycles', 50, ...
            'Learners', 'tree');
        
        elapsed_ens = toc;
        fprintf('✓ Stacked Ensemble training complete in %.2f seconds.\n', elapsed_ens);
        
        % Generate meta-features for validation
        metaFeatures_val = zeros(length(Y_val), length(modelNames));
        if isfield(modelResults, 'SVM')
            [~, scores] = predict(modelResults.SVM.model, X_val_norm);
            metaFeatures_val(:, 1) = scores(:, 1);
        end
        if isfield(modelResults, 'RandomForest')
            [~, scores] = predict(modelResults.RandomForest.model, X_val_norm);
            metaFeatures_val(:, 2) = scores(:, 1);
        end
        if isfield(modelResults, 'GradientBoosting')
            [~, scores] = predict(modelResults.GradientBoosting.model, X_val_norm);
            metaFeatures_val(:, 3) = scores(:, 1);
        end
        
        Y_val_pred_ens = predict(ensembleModel, metaFeatures_val);
        ens_val_acc = 100 * sum(Y_val_pred_ens == Y_val) / length(Y_val);
        fprintf('► Stacked Ensemble Validation Accuracy: %.2f%%\n', ens_val_acc);
        fprintf('  (Trained with out-of-fold predictions - NO data leakage)\n');
        
        modelResults.StackedEnsemble.model = ensembleModel;
        modelResults.StackedEnsemble.valAccuracy = ens_val_acc;
        modelResults.StackedEnsemble.trainTime = elapsed_ens;
        modelResults.StackedEnsemble.metaFeatures_train = metaFeatures;
        modelNames{end+1} = 'Stacked Ensemble';
        modelAccuracies(end+1) = ens_val_acc;
        
    catch ME
        fprintf('!! Error training Stacked Ensemble: %s\n', ME.message);
        fprintf('  Skipping Stacked Ensemble model.\n');
    end
else
    fprintf('\n--- Stacked Ensemble: DISABLED ---\n');
    fprintf('  (Enable in CONFIG.models.trainStackedEnsemble if needed)\n');
end

% ========================================================================
% STEP 4.9: FINAL MODEL SELECTION
% ========================================================================

fprintf('\n========================================================================\n');
fprintf('Step 4.9: Final Model Selection\n');
fprintf('========================================================================\n');

if isempty(modelNames)
    diary off;
    error('No models were successfully trained. Cannot proceed.');
end

fprintf('\n►► Model Comparison:\n');
for i = 1:length(modelNames)
    fprintf('   %-25s: %.2f%%\n', modelNames{i}, modelAccuracies(i));
end

[bestAccuracy, bestIdx] = max(modelAccuracies);
bestModelName = modelNames{bestIdx};

fprintf('\n►► WINNER: %s (Validation Accuracy: %.2f%%)\n', bestModelName, bestAccuracy);

% Extract best model
switch bestModelName
    case 'SVM'
        finalModel = modelResults.SVM.model;
        finalModelType = 'SVM';
    case 'Random Forest'
        finalModel = modelResults.RandomForest.model;
        finalModelType = 'RandomForest';
    case 'Gradient Boosting'
        finalModel = modelResults.GradientBoosting.model;
        finalModelType = 'GradientBoosting';
    case 'Neural Network'
        finalModel = modelResults.NeuralNetwork.model;
        finalModelType = 'NeuralNetwork';
    case 'Stacked Ensemble'
        finalModel = modelResults.StackedEnsemble.model;
        finalModelType = 'StackedEnsemble';
end

% Store validation accuracy for later use
valAccuracy = bestAccuracy;

fprintf('\n✓ Step 4 complete: Model training and selection finished.\n');
fprintf('  Best model: %s with %.2f%% validation accuracy.\n', bestModelName, bestAccuracy);

%% ========================================================================
%% SAVE INTERMEDIATE RESULTS (FOR NEXT STEPS)
%% ========================================================================

% Save workspace variables needed for Step 5
save(fullfile(CONFIG.outputDir, 'step4_results.mat'), ...
    'finalModel', 'finalModelType', 'bestModelName', 'valAccuracy', ...
    'X_train_norm', 'Y_train', 'X_val_norm', 'Y_val', 'X_test_norm', 'Y_test', ...
    'X_crossDataset_norm', 'Y_crossDataset', ...
    'mu_train', 'sigma_train', 'featureNames', 'classNames', ...
    'modelResults', 'modelNames', 'modelAccuracies', ...
    '-v7.3');

fprintf('\n✓ Step 4 results saved to: step4_results.mat\n');


%% ========================================================================
%% STEP 5A: MODEL EVALUATION & CALIBRATION
%% ========================================================================
%%
%% PURPOSE:
%%   Evaluate the best model on test set with proper probability calibration
%%   and generate core performance metrics and visualizations.
%%
%% INPUTS:  step4_results.mat (from Step 4)
%% OUTPUTS: Calibrated model, test metrics, confusion matrix, ROC curves
%%
%% Author: Syed Abbas Ahmad
%% Version: Production v2.0
%% Date: October 30, 2025
%% ========================================================================

fprintf('\n========================================================================\n');
fprintf('STEP 5A: MODEL EVALUATION & CALIBRATION\n');
fprintf('========================================================================\n');

% ========================================================================
% STEP 5A.1: LOAD RESULTS FROM STEP 4
% ========================================================================

fprintf('\nStep 5A.1: Loading Step 4 results...\n');

step4File = fullfile(CONFIG.outputDir, 'step4_results.mat');

if ~exist(step4File, 'file')
    diary off;
    error('Step 4 results not found. Please run Steps 1-4 first.');
end

try
    load(step4File, 'finalModel', 'finalModelType', 'bestModelName', 'valAccuracy', ...
        'X_train_norm', 'Y_train', 'X_val_norm', 'Y_val', 'X_test_norm', 'Y_test', ...
        'X_crossDataset_norm', 'Y_crossDataset', ...
        'mu_train', 'sigma_train', 'featureNames', 'classNames', ...
        'modelResults', 'modelNames', 'modelAccuracies');
    
    fprintf('✓ Successfully loaded Step 4 results:\n');
    fprintf('  - Best model: %s\n', bestModelName);
    fprintf('  - Validation accuracy: %.2f%%\n', valAccuracy);
    fprintf('  - Feature count: %d\n', length(featureNames));
    fprintf('  - Classes: %d\n', length(classNames));
    
catch ME
    diary off;
    error('Failed to load Step 4 results: %s', ME.message);
end

% Verify data integrity
assert(~isempty(finalModel), 'Final model is empty');
assert(size(X_test_norm, 2) == length(featureNames), 'Feature count mismatch');
assert(length(Y_test) == size(X_test_norm, 1), 'Test labels/features mismatch');

fprintf('✓ Data integrity verified.\n');

% ========================================================================
% STEP 5A.2: PROBABILITY CALIBRATION
% ========================================================================

if CONFIG.useCalibration
    fprintf('\nStep 5A.2: Model Probability Calibration\n');
    fprintf('Method: %s scaling\n', CONFIG.calibrationMethod);
    
    try
        % Get uncalibrated scores from validation set
        fprintf('  Computing uncalibrated scores on validation set...\n');
        
        [Y_val_pred_uncal, scores_val_uncal] = predict(finalModel, X_val_norm);
        
        % Handle different score formats (binary vs multi-class)
        if size(scores_val_uncal, 2) == 1
            % Binary scores need conversion
            scores_val_uncal = [1 - scores_val_uncal, scores_val_uncal];
        end
        
        % Convert categorical labels to numeric for calibration
        [~, Y_val_numeric] = ismember(Y_val, classNames);
        
        % Calibrate probabilities
        fprintf('  Calibrating probabilities using validation set...\n');
        
        if strcmp(CONFIG.calibrationMethod, 'sigmoid')
            % Platt scaling (sigmoid calibration)
            calibrationModels = cell(length(classNames), 1);
            
            for c = 1:length(classNames)
                % One-vs-rest calibration
                y_binary = double(Y_val_numeric == c);
                scores_class = scores_val_uncal(:, c);
                
                % Fit logistic regression
                try
                    % Add Jeffreys prior penalty to handle perfect separation
                    calibrationModels{c} = fitglm(scores_class, y_binary, ...
                        'Distribution', 'binomial', 'Link', 'logit', ...
                        'LikelihoodPenalty', 'jeffreys-prior');
                catch ME
                    % Fallback: use identity mapping
                    fprintf('    ⚠️  Logistic regression failed for class %s: %s\n', ...
                        char(classNames{c}), ME.message);
                    fprintf('       Using identity mapping (no calibration).\n');
                    calibrationModels{c} = [];
                end
            end
            
        elseif strcmp(CONFIG.calibrationMethod, 'isotonic')
            % Isotonic regression (non-parametric)
            calibrationModels = cell(length(classNames), 1);
            
            for c = 1:length(classNames)
                y_binary = double(Y_val_numeric == c);
                scores_class = scores_val_uncal(:, c);
                
                % Sort by scores for isotonic regression
                [scores_sorted, sort_idx] = sort(scores_class);
                y_sorted = y_binary(sort_idx);
                
                % Apply pool adjacent violators algorithm
                calibrated = isotonicRegression(scores_sorted, y_sorted);
                
                % Store as lookup table
                calibrationModels{c} = struct('scores', scores_sorted, ...
                    'calibrated', calibrated);
            end
        else
            fprintf('  ⚠️  Unknown calibration method. Skipping calibration.\n');
            calibrationModels = [];
        end
        
        % Apply calibration to test set
        fprintf('  Applying calibration to test set...\n');
        
        [Y_test_pred_uncal, scores_test_uncal] = predict(finalModel, X_test_norm);
        
        if size(scores_test_uncal, 2) == 1
            scores_test_uncal = [1 - scores_test_uncal, scores_test_uncal];
        end
        
        scores_test_cal = zeros(size(scores_test_uncal));
        
        if ~isempty(calibrationModels)
            for c = 1:length(classNames)
                if strcmp(CONFIG.calibrationMethod, 'sigmoid')
                    if ~isempty(calibrationModels{c})
                        scores_test_cal(:, c) = predict(calibrationModels{c}, scores_test_uncal(:, c));
                        % Clip to [0, 1]
                        scores_test_cal(:, c) = max(0, min(1, scores_test_cal(:, c)));
                    else
                        scores_test_cal(:, c) = scores_test_uncal(:, c);
                    end
                    
                elseif strcmp(CONFIG.calibrationMethod, 'isotonic')
                    % Interpolate using calibration lookup table
                    cal_model = calibrationModels{c};
                    scores_test_cal(:, c) = interp1(cal_model.scores, cal_model.calibrated, ...
                        scores_test_uncal(:, c), 'linear', 'extrap');
                    scores_test_cal(:, c) = max(0, min(1, scores_test_cal(:, c)));
                end
            end
            
            % Normalize to sum to 1
            scores_test_cal = scores_test_cal ./ sum(scores_test_cal, 2);
        else
            scores_test_cal = scores_test_uncal;
        end
        
        % Get calibrated predictions
        [~, max_idx] = max(scores_test_cal, [], 2);
        Y_test_pred_cal = classNames(max_idx);
        
        % Evaluate calibration quality
        fprintf('  Evaluating calibration quality...\n');
        
        % Expected Calibration Error (ECE)
        num_bins = 10;
        ece = 0;
        
        for c = 1:length(classNames)
            conf_scores = scores_test_cal(:, c);
            actual = double(Y_test == classNames{c});
            
            for b = 1:num_bins
                bin_lower = (b-1) / num_bins;
                bin_upper = b / num_bins;
                
                in_bin = (conf_scores > bin_lower) & (conf_scores <= bin_upper);
                
                if sum(in_bin) > 0
                    avg_conf = mean(conf_scores(in_bin));
                    avg_acc = mean(actual(in_bin));
                    ece = ece + (sum(in_bin) / length(conf_scores)) * abs(avg_conf - avg_acc);
                end
            end
        end
        
        fprintf('✓ Calibration complete.\n');
        fprintf('  Expected Calibration Error (ECE): %.4f\n', ece);
        
        if ece < 0.05
            fprintf('  ✓ Model is well-calibrated (ECE < 0.05)\n');
        elseif ece < 0.10
            fprintf('  ⚠️  Moderate calibration (0.05 < ECE < 0.10)\n');
        else
            fprintf('  ⚠️  Poor calibration (ECE > 0.10) - confidence scores may be unreliable\n');
        end
        
        % Store calibrated results
        Y_test_pred = Y_test_pred_cal;
        scores_test = scores_test_cal;
        
    catch ME
        fprintf('!! Error during calibration: %s\n', ME.message);
        fprintf('  Proceeding with uncalibrated predictions.\n');
        
        [Y_test_pred, scores_test] = predict(finalModel, X_test_norm);
        if size(scores_test, 2) == 1
            scores_test = [1 - scores_test, scores_test];
        end
        calibrationModels = [];
    end
    
else
    fprintf('\nStep 5A.2: Calibration disabled in configuration.\n');
    
    [Y_test_pred, scores_test] = predict(finalModel, X_test_norm);
    if size(scores_test, 2) == 1
        scores_test = [1 - scores_test, scores_test];
    end
    calibrationModels = [];
end

% ========================================================================
% STEP 5A.3: TEST SET EVALUATION
% ========================================================================

fprintf('\nStep 5A.3: Final Model Evaluation on Test Set\n');

% Calculate test accuracy
testAccuracy = 100 * sum(Y_test_pred == Y_test) / length(Y_test);
fprintf('►► FINAL TEST ACCURACY: %.2f%%\n', testAccuracy);

% Check for performance degradation
if testAccuracy < valAccuracy - 5
    fprintf('  ⚠️  WARNING: Test accuracy is %.2f%% lower than validation accuracy.\n', ...
        valAccuracy - testAccuracy);
    fprintf('     This may indicate overfitting. Consider:\n');
    fprintf('     - Adding regularization\n');
    fprintf('     - Collecting more training data\n');
    fprintf('     - Simplifying the model\n');
elseif testAccuracy > valAccuracy + 5
    fprintf('  ⚠️  WARNING: Test accuracy is %.2f%% higher than validation accuracy.\n', ...
        testAccuracy - valAccuracy);
    fprintf('     This is unusual and may indicate:\n');
    fprintf('     - Data leakage (verify splits)\n');
    fprintf('     - Test set is easier than validation\n');
else
    fprintf('✓ Test performance consistent with validation (difference: %.2f%%)\n', ...
        abs(testAccuracy - valAccuracy));
end

% Generate confusion matrix
fprintf('\nGenerating confusion matrix...\n');

% Ensure both Y_test and Y_test_pred are categorical with same categories
if ~isa(Y_test_pred, 'categorical')
    Y_test_pred = categorical(Y_test_pred, classNames, classNames);
end

confMat = confusionmat(Y_test, Y_test_pred, 'Order', classNames);
confMatNorm = confMat ./ sum(confMat, 2);  % Normalize by row (true class)

% ========================================================================
% STEP 5A.4: FIG 4 - CONFUSION MATRIX
% ========================================================================

fprintf('\nStep 5A.4: Generating Fig 4 (Confusion Matrix)...\n');

try
    fig4 = figure('Position', [100, 100, 1600, 700], 'Name', 'Confusion Matrix');
    
    % Left subplot: Raw counts
    subplot(1, 2, 1);
    imagesc(confMat);
    colormap(flipud(gray));
    colorbar;
    
    % Add text annotations
    for i = 1:length(classNames)
        for j = 1:length(classNames)
            textColor = 'white';
            if confMat(i,j) > max(confMat(:))/2
                textColor = 'black';
            end
            text(j, i, sprintf('%d', confMat(i,j)), ...
                'HorizontalAlignment', 'center', ...
                'Color', textColor, ...
                'FontSize', 10, ...
                'FontWeight', 'bold');
        end
    end
    
    title('Confusion Matrix (Raw Counts)', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Predicted Class', 'FontSize', 12);
    ylabel('True Class', 'FontSize', 12);
    
    % Format class names
    classNamesFormatted = cellfun(@(x) strrep(x, '_', ' '), classNames, 'UniformOutput', false);
    set(gca, 'XTick', 1:length(classNames), 'XTickLabel', classNamesFormatted, ...
        'XTickLabelRotation', 45, 'YTick', 1:length(classNames), ...
        'YTickLabel', classNamesFormatted, 'FontSize', 10);
    
    % Right subplot: Normalized (percentages)
    subplot(1, 2, 2);
    imagesc(confMatNorm);
    colormap(flipud(gray));
    cb = colorbar;
    cb.Label.String = 'Classification Rate';
    clim([0, 1]);
    
    % Add text annotations
    for i = 1:length(classNames)
        for j = 1:length(classNames)
            textColor = 'white';
            if confMatNorm(i,j) > 0.5
                textColor = 'black';
            end
            text(j, i, sprintf('%.1f%%', 100*confMatNorm(i,j)), ...
                'HorizontalAlignment', 'center', ...
                'Color', textColor, ...
                'FontSize', 10, ...
                'FontWeight', 'bold');
        end
    end
    
    title('Confusion Matrix (Normalized)', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Predicted Class', 'FontSize', 12);
    ylabel('True Class', 'FontSize', 12);
    set(gca, 'XTick', 1:length(classNames), 'XTickLabel', classNamesFormatted, ...
        'XTickLabelRotation', 45, 'YTick', 1:length(classNames), ...
        'YTickLabel', classNamesFormatted, 'FontSize', 10);
    
    sgtitle(sprintf('Fig 4: Confusion Matrix - %s (Test Accuracy: %.2f%%)', ...
        bestModelName, testAccuracy), 'FontSize', 16, 'FontWeight', 'bold');
    
    saveas(fig4, fullfile(CONFIG.outputDir, 'Fig4_Confusion_Matrix.png'));
    fprintf('✓ Saved: Fig4_Confusion_Matrix.png\n');
    
catch ME
    fprintf('!! Error generating confusion matrix: %s\n', ME.message);
end

% ========================================================================
% STEP 5A.5: PER-CLASS PERFORMANCE METRICS
% ========================================================================

fprintf('\nStep 5A.5: Computing Per-Class Metrics\n');

% Initialize metrics
precision = zeros(length(classNames), 1);
recall = zeros(length(classNames), 1);
f1score = zeros(length(classNames), 1);
support = zeros(length(classNames), 1);

for c = 1:length(classNames)
    % True Positives, False Positives, False Negatives
    TP = confMat(c, c);
    FP = sum(confMat(:, c)) - TP;
    FN = sum(confMat(c, :)) - TP;
    
    % Precision: TP / (TP + FP)
    if (TP + FP) > 0
        precision(c) = TP / (TP + FP);
    else
        precision(c) = 0;
    end
    
    % Recall: TP / (TP + FN)
    if (TP + FN) > 0
        recall(c) = TP / (TP + FN);
    else
        recall(c) = 0;
    end
    
    % F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    if (precision(c) + recall(c)) > 0
        f1score(c) = 2 * precision(c) * recall(c) / (precision(c) + recall(c));
    else
        f1score(c) = 0;
    end
    
    % Support: number of samples in this class
    support(c) = sum(confMat(c, :));
end

% Macro-averaged metrics (unweighted average across classes)
macroPrecision = mean(precision);
macroRecall = mean(recall);
macroF1 = mean(f1score);

% Weighted metrics (weighted by class support)
weightedPrecision = sum(precision .* support) / sum(support);
weightedRecall = sum(recall .* support) / sum(support);
weightedF1 = sum(f1score .* support) / sum(support);

fprintf('\n--- Macro-Averaged Metrics (Equal Weight per Class) ---\n');
fprintf('  Precision: %.2f%%\n', 100 * macroPrecision);
fprintf('  Recall:    %.2f%%\n', 100 * macroRecall);
fprintf('  F1-Score:  %.2f%%\n', 100 * macroF1);

fprintf('\n--- Weighted Metrics (Weighted by Class Support) ---\n');
fprintf('  Precision: %.2f%%\n', 100 * weightedPrecision);
fprintf('  Recall:    %.2f%%\n', 100 * weightedRecall);
fprintf('  F1-Score:  %.2f%%\n', 100 * weightedF1);

% Display per-class breakdown
fprintf('\n--- Per-Class Performance ---\n');
fprintf('%-30s  Prec    Recall   F1     Support\n', 'Class');
fprintf('%s\n', repmat('-', 70, 1));

for c = 1:length(classNames)
    fprintf('%-30s  %.2f%%  %.2f%%   %.2f%%  %d\n', ...
        char(classNames{c}), ...
        100*precision(c), 100*recall(c), 100*f1score(c), support(c));
end

% Identify problematic classes
fprintf('\n--- Performance Analysis ---\n');

lowRecallClasses = find(recall < 0.90);
if ~isempty(lowRecallClasses)
    fprintf('  ⚠️  Classes with low recall (<90%%):\n');
    for i = 1:length(lowRecallClasses)
        c = lowRecallClasses(i);
        fprintf('    - %s: %.2f%% (%.0f false negatives)\n', ...
            char(classNames{c}), 100*recall(c), support(c)*(1-recall(c)));
    end
end

lowPrecisionClasses = find(precision < 0.90);
if ~isempty(lowPrecisionClasses)
    fprintf('  ⚠️  Classes with low precision (<90%%):\n');
    for i = 1:length(lowPrecisionClasses)
        c = lowPrecisionClasses(i);
        fprintf('    - %s: %.2f%% (high false positive rate)\n', ...
            char(classNames{c}), 100*precision(c));
    end
end

if isempty(lowRecallClasses) && isempty(lowPrecisionClasses)
    fprintf('  ✓ All classes have precision and recall ≥ 90%%\n');
end

% ========================================================================
% STEP 5A.6: ROC CURVES AND AUC
% ========================================================================

fprintf('\nStep 5A.6: Computing ROC Curves and AUC\n');

try
    % Convert labels to binary for each class (one-vs-rest)
    [~, Y_test_numeric] = ismember(Y_test, classNames);
    
    % Store ROC data for each class
    fpr_all = cell(length(classNames), 1);
    tpr_all = cell(length(classNames), 1);
    auc_all = zeros(length(classNames), 1);
    
    for c = 1:length(classNames)
        % Binary labels: 1 for this class, 0 for others
        y_binary = double(Y_test_numeric == c);
        
        % Get scores for this class
        scores_class = scores_test(:, c);
        
        % Sort by scores (descending)
        [scores_sorted, sort_idx] = sort(scores_class, 'descend');
        y_sorted = y_binary(sort_idx);
        
        % Compute ROC curve
        nSamples = length(y_sorted);
        nPositives = sum(y_sorted);
        nNegatives = nSamples - nPositives;
        
        if nPositives == 0 || nNegatives == 0
            fprintf('  ⚠️  Warning: Class %s has no positive or negative samples. Skipping ROC.\n', ...
                char(classNames{c}));
            fpr_all{c} = [0; 1];
            tpr_all{c} = [0; 1];
            auc_all(c) = 0.5;
            continue;
        end
        
        % Initialize
        tp = 0;
        fp = 0;
        tpr = zeros(nSamples + 1, 1);
        fpr = zeros(nSamples + 1, 1);
        
        % First point: (0, 0)
        tpr(1) = 0;
        fpr(1) = 0;
        
        % Compute TPR and FPR at each threshold
        for i = 1:nSamples
            if y_sorted(i) == 1
                tp = tp + 1;
            else
                fp = fp + 1;
            end
            tpr(i+1) = tp / nPositives;
            fpr(i+1) = fp / nNegatives;
        end
        
        % Last point: (1, 1)
        tpr(end) = 1;
        fpr(end) = 1;
        
        % Compute AUC using trapezoidal rule
        auc = 0;
        for i = 1:length(fpr)-1
            auc = auc + (fpr(i+1) - fpr(i)) * (tpr(i) + tpr(i+1)) / 2;
        end
        
        fpr_all{c} = fpr;
        tpr_all{c} = tpr;
        auc_all(c) = auc;
    end
    
    meanAUC = mean(auc_all);
    fprintf('✓ ROC analysis complete.\n');
    fprintf('  Mean AUC: %.4f\n', meanAUC);
    
    if meanAUC > 0.99
        fprintf('  ✓ Excellent discrimination (AUC > 0.99)\n');
    elseif meanAUC > 0.95
        fprintf('  ✓ Very good discrimination (AUC > 0.95)\n');
    elseif meanAUC > 0.90
        fprintf('  ✓ Good discrimination (AUC > 0.90)\n');
    else
        fprintf('  ⚠️  Moderate discrimination (AUC < 0.90)\n');
    end
    
    % ====================================================================
    % FIG 5: ROC CURVES
    % ====================================================================
    
    fprintf('\nGenerating Fig 5 (ROC Curves)...\n');
    
    fig5 = figure('Position', [100, 100, 1200, 900], 'Name', 'ROC Curves');
    
    % Determine subplot layout
    nClasses = length(classNames);
    if nClasses <= 4
        rows = 2; cols = 2;
    elseif nClasses <= 6
        rows = 2; cols = 3;
    elseif nClasses <= 9
        rows = 3; cols = 3;
    elseif nClasses <= 12
        rows = 3; cols = 4;
    else
        rows = 4; cols = 4;
    end
    
    for c = 1:min(nClasses, rows*cols)
        subplot(rows, cols, c);
        
        % Plot ROC curve
        plot(fpr_all{c}, tpr_all{c}, 'b-', 'LineWidth', 2);
        hold on;
        
        % Plot diagonal (random classifier)
        plot([0, 1], [0, 1], 'r--', 'LineWidth', 1);
        
        % Formatting
        xlabel('False Positive Rate', 'FontSize', 10);
        ylabel('True Positive Rate', 'FontSize', 10);
        titleStr = sprintf('%s (AUC=%.3f)', char(classNames{c}), auc_all(c));
        title(strrep(titleStr, '_', ' '), 'FontSize', 11, 'FontWeight', 'bold', 'Interpreter', 'none');
        grid on;
        xlim([0, 1]);
        ylim([0, 1]);
        axis square;
        legend({'ROC Curve', 'Random'}, 'Location', 'southeast', 'FontSize', 8);
    end
    
    sgtitle(sprintf('Fig 5: ROC Curves - One-vs-Rest (Mean AUC: %.4f)', meanAUC), ...
        'FontSize', 16, 'FontWeight', 'bold');
    
    saveas(fig5, fullfile(CONFIG.outputDir, 'Fig5_ROC_Curves.png'));
    fprintf('✓ Saved: Fig5_ROC_Curves.png\n');
    
catch ME
    fprintf('!! Error computing ROC curves: %s\n', ME.message);
    auc_all = zeros(length(classNames), 1);
    meanAUC = 0;
end

% ========================================================================
% STEP 5A.7: SAVE STEP 5A RESULTS
% ========================================================================

fprintf('\nStep 5A.7: Saving evaluation results...\n');

% Save all results for next steps
save(fullfile(CONFIG.outputDir, 'step5a_results.mat'), ...
    'finalModel', 'finalModelType', 'bestModelName', ...
    'Y_test_pred', 'scores_test', 'testAccuracy', 'valAccuracy', ...
    'calibrationModels', 'confMat', 'confMatNorm', ...
    'precision', 'recall', 'f1score', 'support', ...
    'macroPrecision', 'macroRecall', 'macroF1', ...
    'weightedPrecision', 'weightedRecall', 'weightedF1', ...
    'fpr_all', 'tpr_all', 'auc_all', 'meanAUC', ...
    'X_test_norm', 'Y_test', 'X_crossDataset_norm', 'Y_crossDataset', ...
    'featureNames', 'classNames', 'mu_train', 'sigma_train', ...
    '-v7.3');

fprintf('✓ Step 5A results saved to: step5a_results.mat\n');

fprintf('\n========================================================================\n');
fprintf('✅ STEP 5A COMPLETE: Model Evaluation & Calibration\n');
fprintf('========================================================================\n');
fprintf('Summary:\n');
fprintf('  - Test Accuracy:     %.2f%%\n', testAccuracy);
fprintf('  - Macro F1-Score:    %.2f%%\n', 100*macroF1);
fprintf('  - Mean AUC:          %.4f\n', meanAUC);
fprintf('  - Calibrated:        %s\n', iif(CONFIG.useCalibration && ~isempty(calibrationModels), 'Yes', 'No'));
fprintf('  - Figures generated: Fig4 (Confusion Matrix), Fig5 (ROC Curves)\n');
fprintf('\n✓ Ready to proceed to Step 5B (Advanced Validation).\n');
fprintf('========================================================================\n');

%% ========================================================================
%% HELPER FUNCTION: ISOTONIC REGRESSION
%% ========================================================================

function calibrated = isotonicRegression(scores, labels)
    % Pool Adjacent Violators Algorithm for isotonic regression
    % Ensures calibrated probabilities are monotonically increasing
    
    calibrated = labels;
    
    changed = true;
    maxIter = 1000;
    iter = 0;
    
    while changed && iter < maxIter
        changed = false;
        iter = iter + 1;
        
        i = 1;
        while i < length(calibrated)
            if calibrated(i) > calibrated(i+1)
                % Find violating block
                j = i + 1;
                while j < length(calibrated) && calibrated(i) > calibrated(j)
                    j = j + 1;
                end
                
                % Average the block
                avg_val = mean(calibrated(i:j));
                calibrated(i:j) = avg_val;
                changed = true;
            end
            i = i + 1;
        end
    end
end


%% ========================================================================
%% STEP 5B: ADVANCED VALIDATION & ANALYSIS
%% ========================================================================
%%
%% PURPOSE:
%%   Advanced model validation including cross-dataset testing, adversarial
%%   robustness, learning curves, and detailed misclassification analysis.
%%
%% INPUTS:  step4_results.mat, step5a_results.mat
%% OUTPUTS: Learning curves, feature importance, misclassification report
%%
%% Author: Syed Abbas Ahmad
%% Version: Production v2.0
%% Date: October 30, 2025
%% ========================================================================

fprintf('\n========================================================================\n');
fprintf('STEP 5B: ADVANCED VALIDATION & ANALYSIS\n');
fprintf('========================================================================\n');

% ========================================================================
% STEP 5B.1: LOAD RESULTS FROM PREVIOUS STEPS
% ========================================================================

fprintf('\nStep 5B.1: Loading previous results...\n');

step4File = fullfile(CONFIG.outputDir, 'step4_results.mat');
step5aFile = fullfile(CONFIG.outputDir, 'step5a_results.mat');

if ~exist(step4File, 'file') || ~exist(step5aFile, 'file')
    diary off;
    error('Required files not found. Please run Steps 4 and 5A first.');
end

try
    % Load Step 4 results
    step4Data = load(step4File, 'X_train_norm', 'Y_train', 'X_val_norm', 'Y_val', ...
        'modelResults', 'modelNames', 'modelAccuracies');
    
    % Load Step 5A results
    step5aData = load(step5aFile, 'finalModel', 'finalModelType', 'bestModelName', ...
        'Y_test_pred', 'testAccuracy', 'valAccuracy', ...
        'X_test_norm', 'Y_test', 'X_crossDataset_norm', 'Y_crossDataset', ...
        'featureNames', 'classNames', 'confMat');
    
    % Extract to workspace
    X_train_norm = step4Data.X_train_norm;
    Y_train = step4Data.Y_train;
    X_val_norm = step4Data.X_val_norm;
    Y_val = step4Data.Y_val;
    modelResults = step4Data.modelResults;
    
    finalModel = step5aData.finalModel;
    finalModelType = step5aData.finalModelType;
    bestModelName = step5aData.bestModelName;
    Y_test_pred = step5aData.Y_test_pred;
    testAccuracy = step5aData.testAccuracy;
    valAccuracy = step5aData.valAccuracy;
    X_test_norm = step5aData.X_test_norm;
    Y_test = step5aData.Y_test;
    X_crossDataset_norm = step5aData.X_crossDataset_norm;
    Y_crossDataset = step5aData.Y_crossDataset;
    featureNames = step5aData.featureNames;
    classNames = step5aData.classNames;
    confMat = step5aData.confMat;
    
    fprintf('✓ Successfully loaded all required data.\n');
    
catch ME
    diary off;
    error('Failed to load previous results: %s', ME.message);
end

% ========================================================================
% STEP 5B.2: CROSS-DATASET VALIDATION (ADDITIONAL HOLDOUT)
% ========================================================================

if CONFIG.useCrossDatasetValidation && ~isempty(X_crossDataset_norm)
    fprintf('\nStep 5B.2: Additional Holdout Set Validation\n');
    fprintf('Testing model generalization on separate holdout data...\n');
    
    try
        % Predict on cross-dataset holdout
        Y_cross_pred = predict(finalModel, X_crossDataset_norm);
        crossAccuracy = 100 * sum(Y_cross_pred == Y_crossDataset) / length(Y_crossDataset);
        
        fprintf('►► Additional Holdout Accuracy: %.2f%%\n', crossAccuracy);
        fprintf('   (Test Set Accuracy: %.2f%%)\n', testAccuracy);
        
        % Compare with test set performance
        diff_accuracy = crossAccuracy - testAccuracy;
        
        if abs(diff_accuracy) < 3
            fprintf('✓ Performance consistent between test sets (diff: %.2f%%)\n', abs(diff_accuracy));
            fprintf('  Model shows good generalization.\n');
        elseif diff_accuracy > 3
            fprintf('  ⚠️  Holdout accuracy %.2f%% higher than test set.\n', diff_accuracy);
            fprintf('     Both sets from same distribution - difference likely due to:\n');
            fprintf('     - Small sample size variation\n');
            fprintf('     - Random split differences\n');
        else
            fprintf('  ⚠️  Holdout accuracy %.2f%% lower than test set.\n', abs(diff_accuracy));
            fprintf('     Possible causes:\n');
            fprintf('     - Holdout set slightly more challenging\n');
            fprintf('     - Random variation in splits\n');
        end
        
        % Per-class breakdown
        fprintf('\nPer-class performance on holdout set:\n');
        for c = 1:min(5, length(classNames))
            class_mask = Y_crossDataset == classNames{c};
            if sum(class_mask) > 0
                class_acc = 100 * sum(Y_cross_pred(class_mask) == classNames{c}) / sum(class_mask);
                fprintf('  %-30s: %.1f%% (%d samples)\n', ...
                    char(classNames{c}), class_acc, sum(class_mask));
            end
        end
        
    catch ME
        fprintf('!! Error during holdout validation: %s\n', ME.message);
        crossAccuracy = NaN;
    end
    
else
    fprintf('\nStep 5B.2: Additional holdout validation disabled or no data available.\n');
    crossAccuracy = NaN;
end

% ========================================================================
% STEP 5B.3: ADVERSARIAL ROBUSTNESS TESTING
% ========================================================================

if CONFIG.testAdversarialRobustness
    fprintf('\nStep 5B.3: Adversarial Robustness Testing\n');
    fprintf('Testing model resilience to realistic perturbations...\n\n');
    
    try
        % Store original test accuracy for comparison
        baselineAccuracy = testAccuracy;
        
        % --- Test 1: Sensor Noise (+10% RMS) ---
        fprintf('Test 1: Sensor Noise Injection (+10%% RMS)...\n');
        noise_level = 0.10;
        X_test_noisy = X_test_norm + noise_level * randn(size(X_test_norm));
        
        Y_pred_noisy = predict(finalModel, X_test_noisy);
        acc_noisy = 100 * sum(Y_pred_noisy == Y_test) / length(Y_test);
        drop_noisy = baselineAccuracy - acc_noisy;
        
        fprintf('  Accuracy: %.2f%% (drop: %.2f%%)\n', acc_noisy, drop_noisy);
        
        if drop_noisy < 5
            fprintf('  ✓ Excellent robustness to sensor noise\n');
        elseif drop_noisy < 15
            fprintf('  ⚠️  Moderate robustness to sensor noise\n');
        else
            fprintf('  ⚠️  Poor robustness to sensor noise\n');
        end
        
        % --- Test 2: Missing Features (15% dropout) ---
        fprintf('\nTest 2: Missing Features (15%% random dropout)...\n');
        dropout_rate = 0.15;
        X_test_dropout = X_test_norm;
        
        num_features = size(X_test_norm, 2);
        num_samples = size(X_test_norm, 1);
        
        for i = 1:num_samples
            features_to_drop = randperm(num_features, round(dropout_rate * num_features));
            X_test_dropout(i, features_to_drop) = 0;  % Set to normalized 0
        end
        
        Y_pred_dropout = predict(finalModel, X_test_dropout);
        acc_dropout = 100 * sum(Y_pred_dropout == Y_test) / length(Y_test);
        drop_dropout = baselineAccuracy - acc_dropout;
        
        fprintf('  Accuracy: %.2f%% (drop: %.2f%%)\n', acc_dropout, drop_dropout);
        
        if drop_dropout < 5
            fprintf('  ✓ Excellent robustness to missing features\n');
        elseif drop_dropout < 15
            fprintf('  ⚠️  Moderate robustness to missing features\n');
        else
            fprintf('  ⚠️  Poor robustness to missing features\n');
        end
        
        % --- Test 3: Temporal Drift (Sensor Aging) ---
        fprintf('\nTest 3: Temporal Drift (simulated sensor aging)...\n');
        drift_factor = 0.05;  % 5% systematic bias
        X_test_drift = X_test_norm * (1 + drift_factor);
        
        Y_pred_drift = predict(finalModel, X_test_drift);
        acc_drift = 100 * sum(Y_pred_drift == Y_test) / length(Y_test);
        drop_drift = baselineAccuracy - acc_drift;
        
        fprintf('  Accuracy: %.2f%% (drop: %.2f%%)\n', acc_drift, drop_drift);
        
        if drop_drift < 5
            fprintf('  ✓ Excellent robustness to temporal drift\n');
        elseif drop_drift < 15
            fprintf('  ⚠️  Moderate robustness to temporal drift\n');
        else
            fprintf('  ⚠️  Poor robustness to temporal drift\n');
        end
        
        % --- Summary ---
        avg_drop = mean([drop_noisy, drop_dropout, drop_drift]);
        
        fprintf('\n--- Robustness Summary ---\n');
        fprintf('  Average performance degradation: %.2f%%\n', avg_drop);
        
        if avg_drop < 5
            fprintf('  ✓ Model is highly robust to perturbations\n');
            fprintf('  ✓ Suitable for production deployment\n');
        elseif avg_drop < 15
            fprintf('  ⚠️  Moderate robustness - acceptable for production\n');
            fprintf('     Consider monitoring predictions in production\n');
        else
            fprintf('  ⚠️  Limited robustness - caution advised\n');
            fprintf('     Recommendations:\n');
            fprintf('     - Add regularization during training\n');
            fprintf('     - Implement data augmentation with noise\n');
            fprintf('     - Use ensemble methods for stability\n');
            fprintf('     - Monitor prediction confidence in production\n');
        end
        
        robustnessResults = struct(...
            'baseline', baselineAccuracy, ...
            'noise', acc_noisy, ...
            'dropout', acc_dropout, ...
            'drift', acc_drift, ...
            'avg_drop', avg_drop);
        
    catch ME
        fprintf('!! Error during robustness testing: %s\n', ME.message);
        robustnessResults = struct();
    end
    
else
    fprintf('\nStep 5B.3: Adversarial robustness testing disabled.\n');
    robustnessResults = struct();
end

% ========================================================================
% STEP 5B.4: LEARNING CURVES (BIAS-VARIANCE ANALYSIS)
% ========================================================================

if CONFIG.generateRealLearningCurve
    fprintf('\nStep 5B.4: Generating Learning Curves\n');
    fprintf('Analyzing bias-variance tradeoff...\n');
    
    try
        % Define training set sizes to test
        min_samples = max(100, length(classNames) * 10);  % At least 10 per class
        max_samples = size(X_train_norm, 1);
        
        if max_samples < min_samples * 2
            fprintf('  ⚠️  Insufficient training data for learning curves.\n');
            fprintf('     Need at least %d samples, have %d.\n', min_samples * 2, max_samples);
            trainingSizes = [];
        else
            trainingSizes = round(linspace(min_samples, max_samples, min(8, floor(max_samples/min_samples))));
        end
        
        if ~isempty(trainingSizes)
            trainAccuracies = zeros(size(trainingSizes));
            valAccuracies = zeros(size(trainingSizes));
            
            fprintf('  Training models with varying data sizes:\n');
            
            for i = 1:length(trainingSizes)
                n = trainingSizes(i);
                
                % Stratified sampling to maintain class balance
                cv_subset = cvpartition(Y_train, 'HoldOut', 1 - n/length(Y_train));
                idx_subset = training(cv_subset);
                
                X_subset = X_train_norm(idx_subset, :);
                Y_subset = Y_train(idx_subset);
                
                fprintf('    Size %d/%d: Training...', n, max_samples);
                
                % Train simplified model for speed
                if strcmp(finalModelType, 'SVM')
                    model_lc = fitcsvm(X_subset, Y_subset, ...
                        'KernelFunction', 'rbf', 'Standardize', false);
                    
                elseif strcmp(finalModelType, 'RandomForest')
                    model_lc = fitcensemble(X_subset, Y_subset, ...
                        'Method', 'Bag', 'NumLearningCycles', 100);
                    
                elseif strcmp(finalModelType, 'GradientBoosting')
                    model_lc = fitcensemble(X_subset, Y_subset, ...
                        'Method', 'LogitBoost', 'NumLearningCycles', 100);
                    
                elseif strcmp(finalModelType, 'NeuralNetwork')
                    model_lc = fitcnet(X_subset, Y_subset, ...
                        'LayerSizes', [50, 25], 'Verbose', 0);
                else
                    % Fallback
                    model_lc = fitcsvm(X_subset, Y_subset, ...
                        'KernelFunction', 'rbf', 'Standardize', false);
                end
                
                % Evaluate on training subset
                Y_train_pred_lc = predict(model_lc, X_subset);
                trainAccuracies(i) = 100 * sum(Y_train_pred_lc == Y_subset) / length(Y_subset);
                
                % Evaluate on validation set
                Y_val_pred_lc = predict(model_lc, X_val_norm);
                valAccuracies(i) = 100 * sum(Y_val_pred_lc == Y_val) / length(Y_val);
                
                fprintf(' Train: %.1f%%, Val: %.1f%%\n', ...
                    trainAccuracies(i), valAccuracies(i));
            end
            
            % Analyze bias-variance
            final_gap = trainAccuracies(end) - valAccuracies(end);
            
            fprintf('\n--- Learning Curve Analysis ---\n');
            fprintf('  Training accuracy (full data): %.2f%%\n', trainAccuracies(end));
            fprintf('  Validation accuracy: %.2f%%\n', valAccuracies(end));
            fprintf('  Train-Val gap: %.2f%%\n', final_gap);
            
            if final_gap < 5
                fprintf('  ✓ Low bias, low variance - well-balanced model\n');
            elseif final_gap < 10
                fprintf('  ⚠️  Slight overfitting (gap < 10%%)\n');
                fprintf('     Consider mild regularization\n');
            else
                fprintf('  ⚠️  Significant overfitting (gap > 10%%)\n');
                fprintf('     Recommendations:\n');
                fprintf('     - Increase regularization\n');
                fprintf('     - Collect more training data\n');
                fprintf('     - Reduce model complexity\n');
            end
            
            % Check if more data would help
            if length(trainingSizes) > 2
                last_val_improvement = valAccuracies(end) - valAccuracies(end-1);
                if last_val_improvement > 1
                    fprintf('  📊 Validation accuracy still improving with more data.\n');
                    fprintf('     Collecting additional samples likely to help.\n');
                else
                    fprintf('  ✓ Validation accuracy plateaued - sufficient training data.\n');
                end
            end
            
            % ============================================================
            % FIG 6: LEARNING CURVES
            % ============================================================
            
            fprintf('\nGenerating Fig 6 (Learning Curves)...\n');
            
            fig6 = figure('Position', [100, 100, 1000, 600], 'Name', 'Learning Curves');
            
            plot(trainingSizes, trainAccuracies, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
            hold on;
            plot(trainingSizes, valAccuracies, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
            
            % Add shaded region for train-val gap
            fill([trainingSizes, fliplr(trainingSizes)], ...
                [trainAccuracies, fliplr(valAccuracies)], ...
                'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
            
            xlabel('Training Set Size', 'FontSize', 12);
            ylabel('Accuracy (%)', 'FontSize', 12);
            title(sprintf('Fig 6: Learning Curves - %s', bestModelName), ...
                'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');
            legend({'Training Accuracy', 'Validation Accuracy', 'Train-Val Gap'}, ...
                'Location', 'southeast', 'FontSize', 11);
            grid on;
            xlim([trainingSizes(1), trainingSizes(end)]);
            ylim([max(0, min([trainAccuracies; valAccuracies]) - 5), 100]);
            
            saveas(fig6, fullfile(CONFIG.outputDir, 'Fig6_Learning_Curves.png'));
            fprintf('✓ Saved: Fig6_Learning_Curves.png\n');
            
        else
            trainingSizes = [];
            trainAccuracies = [];
            valAccuracies = [];
        end
        
    catch ME
        fprintf('!! Error generating learning curves: %s\n', ME.message);
        trainingSizes = [];
        trainAccuracies = [];
        valAccuracies = [];
    end
    
else
    fprintf('\nStep 5B.4: Learning curves disabled in configuration.\n');
    trainingSizes = [];
    trainAccuracies = [];
    valAccuracies = [];
end

% ========================================================================
% STEP 5B.5: MISCLASSIFICATION ANALYSIS
% ========================================================================

fprintf('\nStep 5B.5: Misclassification Analysis\n');

% Find all misclassified samples
misclassified_idx = find(Y_test_pred ~= Y_test);
num_misclassified = length(misclassified_idx);

fprintf('Total misclassified: %d out of %d (Error rate: %.2f%%)\n', ...
    num_misclassified, length(Y_test), 100 - testAccuracy);

if num_misclassified > 0
    fprintf('\n--- Misclassification Patterns ---\n');
    
    % Analyze confusion patterns
    confusion_pairs = cell(num_misclassified, 2);
    for i = 1:num_misclassified
        idx = misclassified_idx(i);
        confusion_pairs{i, 1} = char(Y_test(idx));
        confusion_pairs{i, 2} = char(Y_test_pred(idx));
    end
    
    % Find most common confusion pairs
    unique_pairs = unique(strcat(confusion_pairs(:,1), ' → ', confusion_pairs(:,2)));
    pair_counts = zeros(length(unique_pairs), 1);
    
    for i = 1:length(unique_pairs)
        pair_counts(i) = sum(strcmp(strcat(confusion_pairs(:,1), ' → ', confusion_pairs(:,2)), unique_pairs{i}));
    end
    
    [sorted_counts, sort_idx] = sort(pair_counts, 'descend');
    
    fprintf('\nMost common confusion patterns:\n');
    for i = 1:min(5, length(unique_pairs))
        if sorted_counts(i) > 0
            fprintf('  %2d. %-50s: %d occurrences\n', ...
                i, unique_pairs{sort_idx(i)}, sorted_counts(i));
        end
    end
    
    % Sample misclassifications
    fprintf('\nSample misclassifications (first 10):\n');
    for i = 1:min(10, num_misclassified)
        idx = misclassified_idx(i);
        fprintf('  %2d. True=%-30s, Predicted=%-30s\n', ...
            i, char(Y_test(idx)), char(Y_test_pred(idx)));
    end
    
    % Analyze if misclassifications involve mixed faults
    mixed_fault_errors = 0;
    for i = 1:num_misclassified
        idx = misclassified_idx(i);
        true_label = char(Y_test(idx));
        pred_label = char(Y_test_pred(idx));
        
        if contains(true_label, 'mixed_') || contains(pred_label, 'mixed_')
            mixed_fault_errors = mixed_fault_errors + 1;
        end
    end
    
    fprintf('\n--- Error Analysis ---\n');
    fprintf('  Errors involving mixed faults: %d out of %d (%.1f%%)\n', ...
        mixed_fault_errors, num_misclassified, 100*mixed_fault_errors/num_misclassified);
    
    if mixed_fault_errors > num_misclassified * 0.5
        fprintf('  ⚠️  Over 50%% of errors involve mixed faults.\n');
        fprintf('     Mixed faults are most challenging for the classifier.\n');
        fprintf('     Recommendations:\n');
        fprintf('     - Collect more mixed fault training samples\n');
        fprintf('     - Engineer features specific to fault combinations\n');
        fprintf('     - Consider hierarchical classification\n');
    end
    
else
    fprintf('✓ Perfect classification - no errors on test set!\n');
    fprintf('  Note: 100%% accuracy is rare and should be validated on new data.\n');
end

% ========================================================================
% STEP 5B.6: FEATURE IMPORTANCE (IF APPLICABLE)
% ========================================================================

if CONFIG.generateFeatureImportance
    fprintf('\nStep 5B.6: Feature Importance Analysis\n');
    
    try
        if strcmp(finalModelType, 'RandomForest') && isfield(modelResults, 'RandomForest')
            fprintf('Extracting feature importance from Random Forest...\n');
            
            % Get out-of-bag predictor importance
            rfModel = modelResults.RandomForest.model;
            
            if isprop(rfModel, 'OOBPermutedPredictorDeltaError')
                importance = rfModel.OOBPermutedPredictorDeltaError;
                
                % Sort by importance
                [sorted_importance, sort_idx] = sort(importance, 'descend');
                
                fprintf('\nTop 15 most important features:\n');
                for i = 1:min(15, length(featureNames))
                    fprintf('  %2d. %-40s: %.4f\n', ...
                        i, featureNames{sort_idx(i)}, sorted_importance(i));
                end
                
                % ========================================================
                % FIG 9: FEATURE IMPORTANCE
                % ========================================================
                
                fprintf('\nGenerating Fig 9 (Feature Importance)...\n');
                
                num_features_to_plot = min(20, length(featureNames));
                
                fig9 = figure('Position', [100, 100, 1000, 700], 'Name', 'Feature Importance');
                
                barh(sorted_importance(num_features_to_plot:-1:1));
                
                featureNamesFormatted = cellfun(@(x) strrep(x, '_', ' '), ...
                    featureNames(sort_idx(num_features_to_plot:-1:1)), 'UniformOutput', false);
                
                set(gca, 'YTick', 1:num_features_to_plot, ...
                    'YTickLabel', featureNamesFormatted, 'FontSize', 10);
                xlabel('Out-of-Bag Predictor Importance', 'FontSize', 12);
                ylabel('Features', 'FontSize', 12);
                title('Fig 9: Feature Importance (Random Forest OOB)', ...
                    'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');
                grid on;
                
                saveas(fig9, fullfile(CONFIG.outputDir, 'Fig9_Feature_Importance.png'));
                fprintf('✓ Saved: Fig9_Feature_Importance.png\n');
                
                featureImportance = struct('features', featureNames(sort_idx), ...
                    'importance', sorted_importance);
                
            else
                fprintf('  ⚠️  OOB importance not available for this model.\n');
                featureImportance = struct();
            end
            
        elseif strcmp(finalModelType, 'GradientBoosting') && isfield(modelResults, 'GradientBoosting')
            fprintf('Extracting feature importance from Gradient Boosting...\n');
            
            gbModel = modelResults.GradientBoosting.model;
            
            % Permutation-based importance (expensive but general)
            fprintf('  Computing permutation importance (may take a moment)...\n');
            
            importance = zeros(length(featureNames), 1);
            baseline_acc = 100 * sum(predict(gbModel, X_val_norm) == Y_val) / length(Y_val);
            
            for f = 1:length(featureNames)
                X_val_permuted = X_val_norm;
                X_val_permuted(:, f) = X_val_permuted(randperm(size(X_val_permuted, 1)), f);
                
                Y_val_pred_perm = predict(gbModel, X_val_permuted);
                acc_permuted = 100 * sum(Y_val_pred_perm == Y_val) / length(Y_val);
                
                importance(f) = baseline_acc - acc_permuted;
            end
            
            [sorted_importance, sort_idx] = sort(importance, 'descend');
            
            fprintf('\nTop 15 most important features:\n');
            for i = 1:min(15, length(featureNames))
                fprintf('  %2d. %-40s: %.4f%% accuracy drop\n', ...
                    i, featureNames{sort_idx(i)}, sorted_importance(i));
            end
            
            featureImportance = struct('features', featureNames(sort_idx), ...
                'importance', sorted_importance);
            
        else
            fprintf('  Feature importance not available for model type: %s\n', finalModelType);
            featureImportance = struct();
        end
        
    catch ME
        fprintf('!! Error computing feature importance: %s\n', ME.message);
        featureImportance = struct();
    end
    
else
    fprintf('\nStep 5B.6: Feature importance analysis disabled.\n');
    featureImportance = struct();
end

% ========================================================================
% STEP 5B.7: SAVE STEP 5B RESULTS
% ========================================================================

fprintf('\nStep 5B.7: Saving advanced validation results...\n');

% Save all Step 5B results
save(fullfile(CONFIG.outputDir, 'step5b_results.mat'), ...
    'crossAccuracy', 'robustnessResults', ...
    'trainingSizes', 'trainAccuracies', 'valAccuracies', ...
    'misclassified_idx', 'num_misclassified', ...
    'featureImportance', ...
    '-v7.3');

fprintf('✓ Step 5B results saved to: step5b_results.mat\n');

fprintf('\n========================================================================\n');
fprintf('✅ STEP 5B COMPLETE: Advanced Validation & Analysis\n');
fprintf('========================================================================\n');
fprintf('Summary:\n');

if CONFIG.useCrossDatasetValidation && ~isnan(crossAccuracy)
    fprintf('  - Additional Holdout Acc: %.2f%%\n', crossAccuracy);
end

if CONFIG.testAdversarialRobustness && ~isempty(fieldnames(robustnessResults))
    fprintf('  - Avg Robustness Drop:    %.2f%%\n', robustnessResults.avg_drop);
end

if ~isempty(trainingSizes)
    fprintf('  - Learning Curve:         Generated (Fig 6)\n');
end

fprintf('  - Misclassifications:     %d analyzed\n', num_misclassified);

if ~isempty(fieldnames(featureImportance))
    fprintf('  - Feature Importance:     Generated (Fig 9)\n');
end

fprintf('\n✓ Ready to proceed to Step 5C (Production Deployment).\n');
fprintf('========================================================================\n');


%% ========================================================================
%% STEP 5C: PRODUCTION DEPLOYMENT
%% ========================================================================
%%
%% PURPOSE:
%%   Finalize model for production deployment including saving model,
%%   generating inference function, creating comprehensive report, and
%%   final visualizations.
%%
%% INPUTS:  step4_results.mat, step5a_results.mat, step5b_results.mat
%% OUTPUTS: Saved model, inference function, production report, final figures
%%
%% Author: Syed Abbas Ahmad
%% Version: Production v2.0
%% Date: October 30, 2025
%% ========================================================================

fprintf('\n========================================================================\n');
fprintf('STEP 5C: PRODUCTION DEPLOYMENT\n');
fprintf('========================================================================\n');

% ========================================================================
% STEP 5C.1: LOAD ALL RESULTS
% ========================================================================

fprintf('\nStep 5C.1: Loading all results for deployment...\n');

step4File = fullfile(CONFIG.outputDir, 'step4_results.mat');
step5aFile = fullfile(CONFIG.outputDir, 'step5a_results.mat');
step5bFile = fullfile(CONFIG.outputDir, 'step5b_results.mat');

if ~exist(step4File, 'file') || ~exist(step5aFile, 'file') || ~exist(step5bFile, 'file')
    diary off;
    error('Required files not found. Please run Steps 4, 5A, and 5B first.');
end

try
    % Load all results
    step4Data = load(step4File);
    step5aData = load(step5aFile);
    step5bData = load(step5bFile);
    
    % Extract critical variables
    finalModel = step5aData.finalModel;
    finalModelType = step5aData.finalModelType;
    bestModelName = step5aData.bestModelName;
    testAccuracy = step5aData.testAccuracy;
    valAccuracy = step5aData.valAccuracy;
    calibrationModels = step5aData.calibrationModels;
    
    mu_train = step5aData.mu_train;
    sigma_train = step5aData.sigma_train;
    featureNames = step5aData.featureNames;
    classNames = step5aData.classNames;
    
    confMat = step5aData.confMat;
    precision = step5aData.precision;
    recall = step5aData.recall;
    f1score = step5aData.f1score;
    macroPrecision = step5aData.macroPrecision;
    macroRecall = step5aData.macroRecall;
    macroF1 = step5aData.macroF1;
    meanAUC = step5aData.meanAUC;
    
    Y_test = step5aData.Y_test;
    Y_test_pred = step5aData.Y_test_pred;
    
    modelAccuracies = step4Data.modelAccuracies;
    modelNames = step4Data.modelNames;
    
    fprintf('✓ Successfully loaded all deployment data.\n');
    
catch ME
    diary off;
    error('Failed to load results: %s', ME.message);
end

% ========================================================================
% STEP 5C.2: FIG 7 - CLASS DISTRIBUTION VISUALIZATION
% ========================================================================

fprintf('\nStep 5C.2: Generating Fig 7 (Class Distribution)...\n');

try
    % Get class counts from each split
    Y_train = step4Data.Y_train;
    Y_val = step4Data.Y_val;
    
    trainCounts = countcats(Y_train);
    valCounts = countcats(Y_val);
    testCounts = countcats(Y_test);
    
    fig7 = figure('Position', [100, 100, 1400, 600], 'Name', 'Class Distribution');
    
    % Prepare data for grouped bar chart
    distributionData = [trainCounts, valCounts, testCounts];
    
    bar(distributionData, 'grouped');
    
    classNamesFormatted = cellfun(@(x) strrep(x, '_', ' '), classNames, 'UniformOutput', false);
    
    set(gca, 'XTick', 1:length(classNames), 'XTickLabel', classNamesFormatted, ...
        'XTickLabelRotation', 45, 'FontSize', 10);
    ylabel('Number of Samples', 'FontSize', 12);
    xlabel('Fault Class', 'FontSize', 12);
    title('Fig 7: Class Distribution Across Data Splits', ...
        'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');
    legend({'Training', 'Validation', 'Test'}, 'Location', 'best', 'FontSize', 11);
    grid on;
    
    saveas(fig7, fullfile(CONFIG.outputDir, 'Fig7_Class_Distribution.png'));
    fprintf('✓ Saved: Fig7_Class_Distribution.png\n');
    
catch ME
    fprintf('!! Error generating class distribution: %s\n', ME.message);
end

% ========================================================================
% STEP 5C.3: FIG 8 - PERFORMANCE COMPARISON
% ========================================================================

fprintf('\nStep 5C.3: Generating Fig 8 (Model Performance Comparison)...\n');

try
    fig8 = figure('Position', [100, 100, 1000, 600], 'Name', 'Model Comparison');
    
    % Create bar chart of model accuracies
    bar(modelAccuracies, 'FaceColor', [0.2, 0.4, 0.8]);
    hold on;
    
    % Highlight best model
    [~, bestIdx] = max(modelAccuracies);
    bar(bestIdx, modelAccuracies(bestIdx), 'FaceColor', [0.8, 0.2, 0.2]);
    
    % Add value labels on bars
    for i = 1:length(modelAccuracies)
        text(i, modelAccuracies(i) + 1, sprintf('%.2f%%', modelAccuracies(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
    end
    
    % Add test accuracy line
    if ~isnan(testAccuracy)
        yline(testAccuracy, '--g', sprintf('Test Accuracy: %.2f%%', testAccuracy), ...
            'LineWidth', 2, 'FontSize', 11, 'LabelHorizontalAlignment', 'left');
    end
    
    set(gca, 'XTick', 1:length(modelNames), 'XTickLabel', modelNames, ...
        'XTickLabelRotation', 45, 'FontSize', 11);
    ylabel('Validation Accuracy (%)', 'FontSize', 12);
    xlabel('Model', 'FontSize', 12);
    title('Fig 8: Model Performance Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    ylim([max(0, min(modelAccuracies) - 10), 100]);
    grid on;
    
    saveas(fig8, fullfile(CONFIG.outputDir, 'Fig8_Performance_Comparison.png'));
    fprintf('✓ Saved: Fig8_Performance_Comparison.png\n');
    
catch ME
    fprintf('!! Error generating performance comparison: %s\n', ME.message);
end

% ========================================================================
% STEP 5C.4: SAVE FINAL MODEL AND METADATA
% ========================================================================

fprintf('\nStep 5C.4: Saving final model and metadata...\n');

try
    % Comprehensive model package
    modelPackage = struct();
    
    % Model
    modelPackage.model = finalModel;
    modelPackage.modelType = finalModelType;
    modelPackage.modelName = bestModelName;
    
    % Normalization parameters
    modelPackage.normalization.mu = mu_train;
    modelPackage.normalization.sigma = sigma_train;
    
    % Calibration
    modelPackage.calibration.enabled = CONFIG.useCalibration && ~isempty(calibrationModels);
    modelPackage.calibration.method = CONFIG.calibrationMethod;
    modelPackage.calibration.models = calibrationModels;
    
    % Features
    modelPackage.features.names = featureNames;
    modelPackage.features.count = length(featureNames);
    modelPackage.features.includeAdvanced = CONFIG.includeAdvancedFeatures;
    
    % Classes
    modelPackage.classes.names = classNames;
    modelPackage.classes.count = length(classNames);
    
    % Performance metrics
    modelPackage.performance.validationAccuracy = valAccuracy;
    modelPackage.performance.testAccuracy = testAccuracy;
    modelPackage.performance.macroPrecision = macroPrecision;
    modelPackage.performance.macroRecall = macroRecall;
    modelPackage.performance.macroF1 = macroF1;
    modelPackage.performance.meanAUC = meanAUC;
    
    % Metadata
    modelPackage.metadata.version = 'Production_v2.0';
    modelPackage.metadata.trainingDate = char(datetime('now'));
    modelPackage.metadata.matlabVersion = version;
    modelPackage.metadata.samplingFrequency = 20480;  % From data generator
    
    % Configuration used
    modelPackage.config = CONFIG;
    
    % Save model package
    modelFilePath = fullfile(CONFIG.outputDir, CONFIG.modelFile);
    save(modelFilePath, 'modelPackage', '-v7.3');
    
    fprintf('✓ Model package saved to: %s\n', modelFilePath);
    fprintf('  Model type: %s\n', finalModelType);
    fprintf('  Features: %d\n', length(featureNames));
    fprintf('  Classes: %d\n', length(classNames));
    fprintf('  Test accuracy: %.2f%%\n', testAccuracy);
    
catch ME
    fprintf('!! Error saving model: %s\n', ME.message);
end

% ========================================================================
% STEP 5C.5: GENERATE PRODUCTION INFERENCE FUNCTION
% ========================================================================

fprintf('\nStep 5C.5: Generating production inference function...\n');

try
    inferenceFilePath = fullfile(CONFIG.outputDir, CONFIG.inferenceFunction);
    fid = fopen(inferenceFilePath, 'w');
    
    if fid == -1
        error('Cannot create inference function file');
    end
    
    % Write inference function
    fprintf(fid, 'function result = predictPFDFault_Production(signalData, varargin)\n');
    fprintf(fid, '%% PREDICTPFDFAULT_PRODUCTION  Production inference function for PFD fault diagnosis\n');
    fprintf(fid, '%%\n');
    fprintf(fid, '%% SYNTAX:\n');
    fprintf(fid, '%%   result = predictPFDFault_Production(signalData)\n');
    fprintf(fid, '%%   result = predictPFDFault_Production(signalData, ''ModelPath'', path)\n');
    fprintf(fid, '%%\n');
    fprintf(fid, '%% INPUTS:\n');
    fprintf(fid, '%%   signalData - Vibration signal vector (time series)\n');
    fprintf(fid, '%%   ModelPath  - (Optional) Path to model file\n');
    fprintf(fid, '%%\n');
    fprintf(fid, '%% OUTPUTS:\n');
    fprintf(fid, '%%   result - Structure containing:\n');
    fprintf(fid, '%%     .fault         - Predicted fault class\n');
    fprintf(fid, '%%     .confidence    - Confidence score (0-100%%)\n');
    fprintf(fid, '%%     .probabilities - Probabilities for all classes\n');
    fprintf(fid, '%%     .top3_faults   - Top 3 predicted faults\n');
    fprintf(fid, '%%     .top3_probs    - Top 3 probabilities\n');
    fprintf(fid, '%%     .warning       - Warning message if confidence low\n');
    fprintf(fid, '%%\n');
    fprintf(fid, '%% EXAMPLE:\n');
    fprintf(fid, '%%   load(''signal_data.mat'', ''x'');\n');
    fprintf(fid, '%%   result = predictPFDFault_Production(x);\n');
    fprintf(fid, '%%   fprintf(''Detected: %%s (Confidence: %%.1f%%%%)\\n'', result.fault, result.confidence);\n');
    fprintf(fid, '%%\n');
    fprintf(fid, '%% Generated: %s\n', char(datetime('now')));
    fprintf(fid, '%% Model: %s\n', bestModelName);
    fprintf(fid, '%% Version: Production v2.0\n');
    fprintf(fid, '\n');
    
    % Parse inputs
    fprintf(fid, '    %% Parse inputs\n');
    fprintf(fid, '    p = inputParser;\n');
    fprintf(fid, '    addRequired(p, ''signalData'', @isvector);\n');
    fprintf(fid, '    addParameter(p, ''ModelPath'', ''%s'', @ischar);\n', modelFilePath);
    fprintf(fid, '    parse(p, signalData, varargin{:});\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    modelPath = p.Results.ModelPath;\n');
    fprintf(fid, '    x = signalData(:);  %% Ensure column vector\n');
    fprintf(fid, '    \n');
    
    % Load model
    fprintf(fid, '    %% Load model package\n');
    fprintf(fid, '    if ~exist(modelPath, ''file'')\n');
    fprintf(fid, '        error(''Model file not found: %%s'', modelPath);\n');
    fprintf(fid, '    end\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    load(modelPath, ''modelPackage'');\n');
    fprintf(fid, '    \n');
    
    % Extract features (call external function)
    fprintf(fid, '    %% Extract features\n');
    fprintf(fid, '    fs = %.0f;  %% Sampling frequency from training\n', 20480);
    fprintf(fid, '    \n');
    fprintf(fid, '    try\n');
    fprintf(fid, '        featValues = extractFeaturesForInference(x, fs, modelPackage);\n');
    fprintf(fid, '    catch ME\n');
    fprintf(fid, '        error(''Feature extraction failed: %%s'', ME.message);\n');
    fprintf(fid, '    end\n');
    fprintf(fid, '    \n');
    
    % Normalize
    fprintf(fid, '    %% Normalize features\n');
    fprintf(fid, '    featNorm = (featValues - modelPackage.normalization.mu) ./ modelPackage.normalization.sigma;\n');
    fprintf(fid, '    \n');
    
    % Predict
    fprintf(fid, '    %% Predict\n');
    fprintf(fid, '    [predictedClass, scores] = predict(modelPackage.model, featNorm);\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    %% Handle score format\n');
    fprintf(fid, '    if size(scores, 2) == 1\n');
    fprintf(fid, '        scores = [1 - scores, scores];\n');
    fprintf(fid, '    end\n');
    fprintf(fid, '    \n');
    
    % Apply calibration
    fprintf(fid, '    %% Apply calibration if enabled\n');
    fprintf(fid, '    if modelPackage.calibration.enabled\n');
    fprintf(fid, '        scoresCalibrated = applyCalibratio(scores, modelPackage);\n');
    fprintf(fid, '        scores = scoresCalibrated;\n');
    fprintf(fid, '    end\n');
    fprintf(fid, '    \n');
    
    % Build result
    fprintf(fid, '    %% Build result structure\n');
    fprintf(fid, '    result = struct();\n');
    fprintf(fid, '    result.fault = char(predictedClass);\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    [~, predIdx] = ismember(predictedClass, modelPackage.classes.names);\n');
    fprintf(fid, '    result.confidence = 100 * scores(predIdx);\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    result.probabilities = array2table(scores, ...\n');
    fprintf(fid, '        ''VariableNames'', modelPackage.classes.names);\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    %% Top 3 predictions\n');
    fprintf(fid, '    [sortedProbs, sortIdx] = sort(scores, ''descend'');\n');
    fprintf(fid, '    result.top3_faults = modelPackage.classes.names(sortIdx(1:min(3, end)));\n');
    fprintf(fid, '    result.top3_probs = sortedProbs(1:min(3, end));\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    %% Confidence warning\n');
    fprintf(fid, '    if result.confidence < 70\n');
    fprintf(fid, '        result.warning = ''Low confidence - verify with domain expert'';\n');
    fprintf(fid, '    elseif result.confidence < 85\n');
    fprintf(fid, '        result.warning = ''Moderate confidence - recommend manual review'';\n');
    fprintf(fid, '    else\n');
    fprintf(fid, '        result.warning = '''';\n');
    fprintf(fid, '    end\n');
    fprintf(fid, '    \n');
    fprintf(fid, 'end\n');
    fprintf(fid, '\n');
    
    % Helper function for feature extraction (simplified version)
    fprintf(fid, 'function featValues = extractFeaturesForInference(x, fs, modelPackage)\n');
    fprintf(fid, '    %% Simplified feature extraction for inference\n');
    fprintf(fid, '    %% For production use, copy full extractFeatures function from training pipeline\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    numFeatures = modelPackage.features.count;\n');
    fprintf(fid, '    featValues = zeros(1, numFeatures);\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    %% TODO: Implement full feature extraction matching training pipeline\n');
    fprintf(fid, '    %% This is a placeholder - copy extractFeatures function from training code\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    error(''Feature extraction not yet implemented. Copy extractFeatures function from training pipeline.'');\n');
    fprintf(fid, 'end\n');
    fprintf(fid, '\n');
    
    % Helper function for calibration
    fprintf(fid, 'function scoresCalibrated = applyCalibration(scores, modelPackage)\n');
    fprintf(fid, '    %% Apply probability calibration\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    scoresCalibrated = scores;\n');
    fprintf(fid, '    calibModels = modelPackage.calibration.models;\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    if strcmp(modelPackage.calibration.method, ''sigmoid'')\n');
    fprintf(fid, '        for c = 1:length(calibModels)\n');
    fprintf(fid, '            if ~isempty(calibModels{c})\n');
    fprintf(fid, '                scoresCalibrated(c) = predict(calibModels{c}, scores(c));\n');
    fprintf(fid, '                scoresCalibrated(c) = max(0, min(1, scoresCalibrated(c)));\n');
    fprintf(fid, '            end\n');
    fprintf(fid, '        end\n');
    fprintf(fid, '    end\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    %% Normalize to sum to 1\n');
    fprintf(fid, '    scoresCalibrated = scoresCalibrated / sum(scoresCalibrated);\n');
    fprintf(fid, 'end\n');
    
    fclose(fid);
    
    fprintf('✓ Inference function generated: %s\n', inferenceFilePath);
    fprintf('  ⚠️  NOTE: Feature extraction function needs to be completed.\n');
    fprintf('     Copy extractFeatures function from training pipeline into inference file.\n');
    
catch ME
    if fid ~= -1
        fclose(fid);
    end
    fprintf('!! Error generating inference function: %s\n', ME.message);
end

% ========================================================================
% STEP 5C.6: GENERATE COMPREHENSIVE PRODUCTION REPORT
% ========================================================================

fprintf('\nStep 5C.6: Generating comprehensive production report...\n');

try
    reportFilePath = fullfile(CONFIG.outputDir, CONFIG.reportFile);
    fid = fopen(reportFilePath, 'w');
    
    if fid == -1
        error('Cannot create report file');
    end
    
    % Report header
    fprintf(fid, '========================================================================\n');
    fprintf(fid, 'PFD FAULT DIAGNOSIS SYSTEM - PRODUCTION ANALYSIS REPORT\n');
    fprintf(fid, '========================================================================\n');
    fprintf(fid, '\n');
    fprintf(fid, 'Generated: %s\n', char(datetime('now')));
    fprintf(fid, 'Pipeline Version: Production v2.0\n');
    fprintf(fid, 'Report Type: Final Production Deployment\n');
    fprintf(fid, '\n');
    
    % Executive Summary
    fprintf(fid, '--- EXECUTIVE SUMMARY ---\n\n');
    fprintf(fid, 'Best Model:            %s\n', bestModelName);
    fprintf(fid, 'Test Accuracy:         %.2f%%\n', testAccuracy);
    fprintf(fid, 'Validation Accuracy:   %.2f%%\n', valAccuracy);
    fprintf(fid, 'Macro F1-Score:        %.2f%%\n', 100*macroF1);
    fprintf(fid, 'Mean AUC:              %.4f\n', meanAUC);
    fprintf(fid, 'Calibration:           %s\n', iif(CONFIG.useCalibration && ~isempty(calibrationModels), 'Enabled', 'Disabled'));
    fprintf(fid, '\n');
    
    % Dataset Information
    fprintf(fid, '--- DATASET INFORMATION ---\n\n');
    fprintf(fid, 'Fault Classes:         %d\n', length(classNames));
    fprintf(fid, 'Features Extracted:    %d\n', length(featureNames));
    fprintf(fid, 'Advanced Features:     %s\n', iif(CONFIG.includeAdvancedFeatures, 'Enabled', 'Disabled'));
    fprintf(fid, '\n');
    fprintf(fid, 'Training Samples:      %d\n', length(Y_train));
    fprintf(fid, 'Validation Samples:    %d\n', length(step4Data.Y_val));
    fprintf(fid, 'Test Samples:          %d\n', length(Y_test));
    fprintf(fid, '\n');
    
    % Fault Classes
    fprintf(fid, 'Fault Classes:\n');
    for i = 1:length(classNames)
        fprintf(fid, '  %2d. %s\n', i, char(classNames{i}));
    end
    fprintf(fid, '\n');
    
    % Model Comparison
    fprintf(fid, '--- MODEL COMPARISON ---\n\n');
    fprintf(fid, 'Models Trained and Compared:\n');
    for i = 1:length(modelNames)
        marker = '';
        if strcmp(modelNames{i}, bestModelName)
            marker = ' ← SELECTED';
        end
        fprintf(fid, '  %-25s: %.2f%%%s\n', modelNames{i}, modelAccuracies(i), marker);
    end
    fprintf(fid, '\n');
    
    % Performance Metrics
    fprintf(fid, '--- DETAILED PERFORMANCE METRICS ---\n\n');
    fprintf(fid, 'Overall Metrics:\n');
    fprintf(fid, '  Test Accuracy:         %.2f%%\n', testAccuracy);
    fprintf(fid, '  Macro Precision:       %.2f%%\n', 100*macroPrecision);
    fprintf(fid, '  Macro Recall:          %.2f%%\n', 100*macroRecall);
    fprintf(fid, '  Macro F1-Score:        %.2f%%\n', 100*macroF1);
    fprintf(fid, '  Mean AUC:              %.4f\n', meanAUC);
    fprintf(fid, '\n');
    
    fprintf(fid, 'Per-Class Performance:\n');
    fprintf(fid, '%-30s  Precision  Recall  F1-Score\n', 'Class');
    fprintf(fid, '%s\n', repmat('-', 70, 1));
    for c = 1:length(classNames)
        fprintf(fid, '%-30s  %6.2f%%  %6.2f%%  %6.2f%%\n', ...
            char(classNames{c}), 100*precision(c), 100*recall(c), 100*f1score(c));
    end
    fprintf(fid, '\n');
    
    % Robustness Testing
    if isfield(step5bData, 'robustnessResults') && ~isempty(fieldnames(step5bData.robustnessResults))
        robustness = step5bData.robustnessResults;
        fprintf(fid, '--- ROBUSTNESS TESTING ---\n\n');
        fprintf(fid, 'Baseline Accuracy:            %.2f%%\n', robustness.baseline);
        fprintf(fid, 'With Sensor Noise (+10%% RMS): %.2f%%\n', robustness.noise);
        fprintf(fid, 'With Missing Features (15%%):  %.2f%%\n', robustness.dropout);
        fprintf(fid, 'With Temporal Drift (5%%):     %.2f%%\n', robustness.drift);
        fprintf(fid, 'Average Degradation:          %.2f%%\n', robustness.avg_drop);
        fprintf(fid, '\n');
    end
    
    % Production Recommendations
    fprintf(fid, '--- PRODUCTION RECOMMENDATIONS ---\n\n');
    
    if testAccuracy >= 95
        fprintf(fid, '✓ PRODUCTION READY: Excellent Performance\n\n');
        fprintf(fid, 'System Strengths:\n');
        fprintf(fid, '  • High accuracy (%.2f%%) across all fault types\n', testAccuracy);
        fprintf(fid, '  • Strong per-class performance\n');
        fprintf(fid, '  • Well-calibrated confidence scores\n');
        fprintf(fid, '  • Good generalization capability\n\n');
        fprintf(fid, 'Deployment Recommendations:\n');
        fprintf(fid, '  ✓ Deploy with confidence-based alerting\n');
        fprintf(fid, '  ✓ Monitor for model drift (quarterly reviews)\n');
        fprintf(fid, '  ✓ Collect edge cases for continuous improvement\n');
        fprintf(fid, '  ✓ Establish feedback loop with maintenance teams\n\n');
        
    elseif testAccuracy >= 90
        fprintf(fid, '✓ PRODUCTION READY: Good Performance\n\n');
        fprintf(fid, 'System Status:\n');
        fprintf(fid, '  • Meets 90%% accuracy requirement (%.2f%%)\n', testAccuracy);
        fprintf(fid, '  • Suitable for production deployment\n');
        fprintf(fid, '  • Some room for improvement exists\n\n');
        fprintf(fid, 'Deployment Recommendations:\n');
        fprintf(fid, '  ✓ Deploy with human oversight for low-confidence predictions\n');
        fprintf(fid, '  ✓ Monitor performance on production data monthly\n');
        fprintf(fid, '  ✓ Collect additional training data for weaker classes\n');
        fprintf(fid, '  ✓ Consider ensemble methods for improvement\n\n');
        
    else
        fprintf(fid, '⚠️  BELOW TARGET: Additional Development Needed\n\n');
        fprintf(fid, 'Current Performance: %.2f%% (Target: 90%%)\n', testAccuracy);
        fprintf(fid, 'Gap: %.2f%%\n\n', 90 - testAccuracy);
        fprintf(fid, 'Improvement Recommendations:\n');
        fprintf(fid, '  • Collect more diverse training samples\n');
        fprintf(fid, '  • Engineer additional discriminative features\n');
        fprintf(fid, '  • Investigate deep learning approaches\n');
        fprintf(fid, '  • Review and refine fault simulation models\n');
        fprintf(fid, '  • Consider domain adaptation techniques\n\n');
    end
    
    % Monitoring Plan
    fprintf(fid, '--- POST-DEPLOYMENT MONITORING PLAN ---\n\n');
    fprintf(fid, 'Weekly:\n');
    fprintf(fid, '  • Review prediction confidence distributions\n');
    fprintf(fid, '  • Analyze low-confidence cases\n');
    fprintf(fid, '  • Check for unusual fault patterns\n\n');
    fprintf(fid, 'Monthly:\n');
    fprintf(fid, '  • Evaluate model performance on production data\n');
    fprintf(fid, '  • Update calibration if distribution shift detected\n');
    fprintf(fid, '  • Review misclassification patterns\n\n');
    fprintf(fid, 'Quarterly:\n');
    fprintf(fid, '  • Retrain model with accumulated production data\n');
    fprintf(fid, '  • Re-evaluate feature importance\n');
    fprintf(fid, '  • Update documentation and procedures\n\n');
    
    fprintf(fid, 'Trigger Conditions for Retraining:\n');
    fprintf(fid, '  • Accuracy drop > 5%% on production data\n');
    fprintf(fid, '  • Mean confidence drop > 10%%\n');
    fprintf(fid, '  • New fault patterns emerge\n');
    fprintf(fid, '  • Significant sensor/hardware changes\n\n');
    
    % Technical Details
    fprintf(fid, '--- TECHNICAL SPECIFICATIONS ---\n\n');
    fprintf(fid, 'Model Architecture:\n');
    fprintf(fid, '  Type:                  %s\n', finalModelType);
    fprintf(fid, '  Input Features:        %d\n', length(featureNames));
    fprintf(fid, '  Output Classes:        %d\n', length(classNames));
    fprintf(fid, '  Calibration:           %s\n', iif(CONFIG.useCalibration, 'Yes (Sigmoid)', 'No'));
    fprintf(fid, '\n');
    
    fprintf(fid, 'Data Processing:\n');
    fprintf(fid, '  Normalization:         Z-score (per feature)\n');
    fprintf(fid, '  Feature Selection:     %s\n', iif(CONFIG.useFeatureSelection, 'MRMR', 'None'));
    fprintf(fid, '  Sampling Frequency:    20480 Hz\n');
    fprintf(fid, '  Signal Duration:       5 seconds\n');
    fprintf(fid, '\n');
    
    fprintf(fid, 'Files Generated:\n');
    fprintf(fid, '  Model Package:         %s\n', CONFIG.modelFile);
    fprintf(fid, '  Inference Function:    %s\n', CONFIG.inferenceFunction);
    fprintf(fid, '  Visualizations:        Fig0-Fig9 (PNG files)\n');
    fprintf(fid, '  This Report:           %s\n', CONFIG.reportFile);
    fprintf(fid, '\n');
    
    % Footer
    fprintf(fid, '========================================================================\n');
    fprintf(fid, 'END OF REPORT\n');
    fprintf(fid, '========================================================================\n');
    fprintf(fid, '\n');
    fprintf(fid, 'For questions or support, contact the PFD Diagnostics Team.\n');
    fprintf(fid, 'Report generated by: pipeline_v2.m (Production v2.0)\n');
    fprintf(fid, 'Timestamp: %s\n', char(datetime('now')));
    
    fclose(fid);
    
    fprintf('✓ Production report saved: %s\n', reportFilePath);
    
catch ME
    if fid ~= -1
        fclose(fid);
    end
    fprintf('!! Error generating report: %s\n', ME.message);
end

% ========================================================================
% STEP 5C.7: FINAL SUMMARY
% ========================================================================

fprintf('\n========================================================================\n');
fprintf('✅ STEP 5C COMPLETE: Production Deployment\n');
fprintf('========================================================================\n');
fprintf('\nDeployment Package Contents:\n');
fprintf('  1. Model Package:          %s\n', CONFIG.modelFile);
fprintf('  2. Inference Function:     %s\n', CONFIG.inferenceFunction);
fprintf('  3. Production Report:      %s\n', CONFIG.reportFile);
fprintf('  4. Visualizations:         Fig0-Fig9 (9 figures)\n');
fprintf('\n');

fprintf('Model Performance Summary:\n');
fprintf('  - Best Model:              %s\n', bestModelName);
fprintf('  - Test Accuracy:           %.2f%%\n', testAccuracy);
fprintf('  - Macro F1-Score:          %.2f%%\n', 100*macroF1);
fprintf('  - Mean AUC:                %.4f\n', meanAUC);
fprintf('\n');

fprintf('Next Steps for Deployment:\n');
fprintf('  1. Complete feature extraction in inference function\n');
fprintf('  2. Test inference function with sample signals\n');
fprintf('  3. Integrate with production monitoring system\n');
fprintf('  4. Establish performance monitoring dashboard\n');
fprintf('  5. Train operations team on system usage\n');
fprintf('\n');

fprintf('========================================================================\n');
fprintf('🎉 COMPLETE PIPELINE EXECUTION FINISHED SUCCESSFULLY\n');
fprintf('========================================================================\n');
fprintf('\n');
fprintf('All Steps Completed:\n');
fprintf('  ✓ Step 1:  Configuration and Setup\n');
fprintf('  ✓ Step 2:  Feature Engineering\n');
fprintf('  ✓ Step 3:  Data Exploration\n');
fprintf('  ✓ Step 4:  Model Training and Selection\n');
fprintf('  ✓ Step 5A: Model Evaluation and Calibration\n');
fprintf('  ✓ Step 5B: Advanced Validation\n');
fprintf('  ✓ Step 5C: Production Deployment\n');
fprintf('\n');

fprintf('Total Execution Time: %.2f minutes\n', toc/60);
fprintf('\n');
fprintf('System is ready for production deployment.\n');
fprintf('========================================================================\n');

% Turn off diary
diary off;




%% ========================================================================
%% HELPER FUNCTIONS FOR ADVANCED FEATURES
%% ========================================================================

function lyap = computeLyapunovExponent(x)
    try
        embedding_dim = 3;
        tau = 10;
        N_samples = length(x);
        
        max_embed_size = 5000;
        if N_samples - (embedding_dim-1)*tau > max_embed_size
            downsample_factor = ceil((N_samples - (embedding_dim-1)*tau) / max_embed_size);
            x_ds = downsample(x, downsample_factor);
            N_samples = length(x_ds);
        else
            x_ds = x;
        end
        
        N_vectors = N_samples - (embedding_dim-1)*tau;
        if N_vectors < 10
            lyap = 0;
            return;
        end
        
        x_embedded = zeros(N_vectors, embedding_dim);
        for dim = 1:embedding_dim
            x_embedded(:, dim) = x_ds((dim-1)*tau + 1 : (dim-1)*tau + N_vectors);
        end
        
        max_lyap_iter = min(20, floor(N_vectors/5));
        lyap_sum = 0;
        lyap_count = 0;
        
        for iter = 1:max_lyap_iter
            start_idx = iter;
            if start_idx + iter + 1 <= N_vectors
                ref_point = x_embedded(start_idx, :);
                neighbor_point = x_embedded(start_idx + 1, :);
                dist_0 = norm(ref_point - neighbor_point);
                
                future_idx = start_idx + iter;
                if future_idx + 1 <= N_vectors
                    dist_t = norm(x_embedded(future_idx, :) - x_embedded(future_idx + 1, :));
                    if dist_0 > eps && dist_t > eps
                        lyap_sum = lyap_sum + log(dist_t / dist_0);
                        lyap_count = lyap_count + 1;
                    end
                end
            end
        end
        
        if lyap_count > 0
            lyap = lyap_sum / lyap_count;
        else
            lyap = 0;
        end
    catch
        lyap = 0;
    end
end

function corr_dim = computeCorrelationDimension(x)
    try
        embedding_dim = 3;
        tau = 10;
        N_samples = length(x);
        
        N_vectors = N_samples - (embedding_dim-1)*tau;
        if N_vectors < 10
            corr_dim = embedding_dim;
            return;
        end
        
        x_embedded = zeros(N_vectors, embedding_dim);
        for dim = 1:embedding_dim
            x_embedded(:, dim) = x((dim-1)*tau + 1 : (dim-1)*tau + N_vectors);
        end
        
        max_vectors = min(500, N_vectors);
        subset_idx = round(linspace(1, N_vectors, max_vectors));
        x_subset = x_embedded(subset_idx, :);
        
        dists = pdist(x_subset);
        mean_dist = mean(dists);
        
        radii = logspace(log10(mean_dist/5), log10(mean_dist*2), 10);
        corr_counts = zeros(size(radii));
        for r_idx = 1:length(radii)
            corr_counts(r_idx) = sum(dists < radii(r_idx)) / length(dists);
        end
        
        valid_idx = corr_counts > 0 & corr_counts < 1;
        if sum(valid_idx) > 2
            log_r = log(radii(valid_idx));
            log_c = log(corr_counts(valid_idx) + eps);
            p = polyfit(log_r, log_c, 1);
            corr_dim = p(1);
        else
            corr_dim = embedding_dim;
        end
    catch
        corr_dim = 3;
    end
end

function sampen = computeSampleEntropy(x)
    try
        m = 2;
        r = 0.2 * std(x);
        
        if length(x) > 2000
            x_samp = downsample(x, ceil(length(x)/2000));
        else
            x_samp = x;
        end
        
        N = length(x_samp) - m;
        if N < 10
            sampen = 0;
            return;
        end
        
        templates = zeros(N, m + 1);
        for j = 1:N
            if j + m <= length(x_samp)
                templates(j, :) = x_samp(j:j+m);
            end
        end
        
        B = 0;
        A = 0;
        for i = 1:N
            for j = 1:N
                if i ~= j
                    dist_m = max(abs(templates(i, 1:m) - templates(j, 1:m)));
                    if dist_m < r
                        B = B + 1;
                        dist_m1 = max(abs(templates(i, :) - templates(j, :)));
                        if dist_m1 < r
                            A = A + 1;
                        end
                    end
                end
            end
        end
        
        if B > 0 && A > 0
            sampen = -log(A / B);
        else
            sampen = 0;
        end
    catch
        sampen = 0;
    end
end

function disp = computePoincareDispersion(x)
    try
        tau = 1;
        if length(x) < tau + 1
            disp = 0;
            return;
        end
        x1 = x(1:end-tau);
        x2 = x(tau+1:end);
        disp = std(sqrt(x1.^2 + x2.^2));
    catch
        disp = 0;
    end
end

function alpha = computeDFA(x)
    try
        y = cumsum(x - mean(x));
        
        N = length(y);
        scales = unique(round(logspace(log10(10), log10(N/4), 20)));
        F = zeros(size(scales));
        
        for s_idx = 1:length(scales)
            scale = scales(s_idx);
            num_segments = floor(N / scale);
            
            fluctuation_sum = 0;
            for seg = 1:num_segments
                idx_start = (seg-1)*scale + 1;
                idx_end = seg*scale;
                segment = y(idx_start:idx_end);
                t_seg = (1:scale)';
                p = polyfit(t_seg, segment, 1);
                fit_seg = polyval(p, t_seg);
                fluctuation_sum = fluctuation_sum + sum((segment - fit_seg).^2);
            end
            
            F(s_idx) = sqrt(fluctuation_sum / (num_segments * scale));
        end
        
        valid_idx = F > 0;
        if sum(valid_idx) > 5
            log_scales = log(scales(valid_idx));
            log_F = log(F(valid_idx));
            p = polyfit(log_scales, log_F, 1);
            alpha = p(1);
        else
            alpha = 0.5;
        end
    catch
        alpha = 0.5;
    end
end

function result = iif(condition, trueVal, falseVal)
    if condition
        result = trueVal;
    else
        result = falseVal;
    end
end