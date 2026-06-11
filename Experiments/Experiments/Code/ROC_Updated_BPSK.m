clc;
clear;
close all;

%% ================= USER INPUT =================
% You must provide for each model:
%   yTrue_modelX   -> N x 1 vector of true class labels
%   scores_modelX  -> N x K matrix of predicted scores/probabilities
%
% Example:
% load('ShallowCNN_A_scores.mat');   % contains yTrue_modelA, scores_modelA
% load('DeepCNN_B_scores.mat');      % contains yTrue_modelB, scores_modelB
% ...
%
% Labels can be numeric like 1..20
% scores(:,k) should be the score/probability for class k

% --------- Example loading section ----------
% load('ShallowCNN_A_scores.mat');    % yTrue_modelA, scores_modelA
% load('DeepCNN_B_scores.mat');       % yTrue_modelB, scores_modelB
% load('ResNet_C_scores.mat');        % yTrue_modelC, scores_modelC
% load('Inception_D_scores.mat');     % yTrue_modelD, scores_modelD
% load('DenseNet_E_scores.mat');      % yTrue_modelE, scores_modelE

% Put all models into cell arrays
modelNames = { ...
    'ShallowCNN-A', ...
    'DeepCNN-B', ...
    'ResNet-C', ...
    'Inception-D', ...
    'DenseNet-E'};

% Replace these with your actual variables after loading
yTrueAll = { ...
    yTrue_modelA, ...
    yTrue_modelB, ...
    yTrue_modelC, ...
    yTrue_modelD, ...
    yTrue_modelE};

scoresAll = { ...
    scores_modelA, ...
    scores_modelB, ...
    scores_modelC, ...
    scores_modelD, ...
    scores_modelE};

nBoot = 500;          % bootstrap iterations for confidence bands
confLevel = 0.95;     % 95% confidence band
fprGrid = linspace(0,1,501);   % common FPR grid

%% ================= MAIN LOOP =================
for m = 1:length(modelNames)
    yTrue  = yTrueAll{m};
    scores = scoresAll{m};

    % Ensure column vector
    yTrue = yTrue(:);

    % Get unique classes
    classLabels = unique(yTrue);
    K = numel(classLabels);

    % Convert labels to one-hot
    Ybin = zeros(length(yTrue), K);
    for k = 1:K
        Ybin(:,k) = (yTrue == classLabels(k));
    end

    % ======== ORIGINAL MICRO / MACRO ROC ========
    [fprMicro, tprMicro, aucMicro] = computeMicroROC(Ybin, scores);
    [fprMacro, tprMacro, aucMacro] = computeMacroROC(Ybin, scores, fprGrid);

    % ======== BOOTSTRAP CONFIDENCE BANDS ========
    tprMicroBoot = zeros(nBoot, numel(fprGrid));
    tprMacroBoot = zeros(nBoot, numel(fprGrid));

    N = length(yTrue);

    for b = 1:nBoot
        idx = randsample(N, N, true);

        yBoot = yTrue(idx);
        sBoot = scores(idx,:);

        YbinBoot = zeros(N, K);
        for k = 1:K
            YbinBoot(:,k) = (yBoot == classLabels(k));
        end

        try
            [~, tprMicro_b] = computeMicroROC_interp(YbinBoot, sBoot, fprGrid);
            [~, tprMacro_b] = computeMacroROC(YbinBoot, sBoot, fprGrid);

            tprMicroBoot(b,:) = tprMicro_b;
            tprMacroBoot(b,:) = tprMacro_b;
        catch
            tprMicroBoot(b,:) = nan(1, numel(fprGrid));
            tprMacroBoot(b,:) = nan(1, numel(fprGrid));
        end
    end

    alpha = (1 - confLevel)/2;

    microLower = quantile(tprMicroBoot, alpha, 1, 'Method','approximate');
    microUpper = quantile(tprMicroBoot, 1-alpha, 1, 'Method','approximate');

    macroLower = quantile(tprMacroBoot, alpha, 1, 'Method','approximate');
    macroUpper = quantile(tprMacroBoot, 1-alpha, 1, 'Method','approximate');

    % ======== PLOT ========
    figure('Name', modelNames{m}, 'Color', 'w');
    hold on;
    grid on;
    box on;

    % Micro confidence band
    fill([fprGrid fliplr(fprGrid)], ...
         [microLower fliplr(microUpper)], ...
         [0.85 0.90 1.00], ...
         'EdgeColor', 'none', ...
         'FaceAlpha', 0.35);

    % Macro confidence band
    fill([fprGrid fliplr(fprGrid)], ...
         [macroLower fliplr(macroUpper)], ...
         [1.00 0.85 0.85], ...
         'EdgeColor', 'none', ...
         'FaceAlpha', 0.35);

    % Plot ROC curves
    plot(fprMicro, tprMicro, 'b-', 'LineWidth', 2);
    plot(fprMacro, tprMacro, 'r-', 'LineWidth', 2);

    % Diagonal line
    plot([0 1], [0 1], 'k--', 'LineWidth', 1.2);

    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(sprintf('%s : Micro/Macro-average ROC with Confidence Bands', modelNames{m}));

    legend({ ...
        sprintf('Micro 95%% CI'), ...
        sprintf('Macro 95%% CI'), ...
        sprintf('Micro-average ROC (AUC = %.4f)', aucMicro), ...
        sprintf('Macro-average ROC (AUC = %.4f)', aucMacro), ...
        'Random Guess'}, ...
        'Location', 'southeast');

    hold off;

    % Optional save
    saveas(gcf, [modelNames{m} '_ROC_with_CI.png']);
end

%% ================= FUNCTIONS =================

function [fprMicro, tprMicro, aucMicro] = computeMicroROC(Ybin, scores)
    yTrueFlat  = Ybin(:);
    scoreFlat  = scores(:);

    [fprMicro, tprMicro, ~, aucMicro] = perfcurve(yTrueFlat, scoreFlat, 1);
end

function [fprGrid, tprInterp] = computeMicroROC_interp(Ybin, scores, fprGrid)
    yTrueFlat = Ybin(:);
    scoreFlat = scores(:);

    [fpr, tpr] = perfcurve(yTrueFlat, scoreFlat, 1);
    [fprUnique, ia] = unique(fpr);
    tprUnique = tpr(ia);

    tprInterp = interp1(fprUnique, tprUnique, fprGrid, 'linear', 'extrap');
    tprInterp = max(0, min(1, tprInterp));
end

function [fprGrid, tprMacro, aucMacro] = computeMacroROC(Ybin, scores, fprGrid)
    K = size(Ybin,2);

    tprAll = zeros(K, numel(fprGrid));
    aucEach = zeros(K,1);

    for k = 1:K
        [fpr, tpr, ~, aucEach(k)] = perfcurve(Ybin(:,k), scores(:,k), 1);

        [fprUnique, ia] = unique(fpr);
        tprUnique = tpr(ia);

        tprInterp = interp1(fprUnique, tprUnique, fprGrid, 'linear', 'extrap');
        tprInterp = max(0, min(1, tprInterp));

        tprAll(k,:) = tprInterp;
    end

    tprMacro = mean(tprAll, 1, 'omitnan');
    aucMacro = mean(aucEach, 'omitnan');
end