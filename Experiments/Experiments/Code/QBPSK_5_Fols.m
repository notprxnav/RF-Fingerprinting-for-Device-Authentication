clc;
clear all;
close all;
%%   PART 1 — DEVICE SIGNAL GENERATION (20 realistic devices)

%% PARAMETERS
Nsym = 200000;      
M = 4;              
fs = 1e6;
t = (0:Nsym-1)' / fs;
numDevices = 20;

%% Base QPSK signal
data = randi([0 M-1], Nsym, 1);
baseSig = pskmod(data, M, pi/4);

%% DEVICE PARAMETERS
devices = struct;

for k = 1:numDevices
    devices(k).freqOffset     = randi([-700 700]);     
    devices(k).iqGain         = 0.02 * randn;
    devices(k).iqPhase        = deg2rad(2 * randn);
    devices(k).phaseErr       = deg2rad(6 * randn);
    devices(k).paAlpha        = 0.6 + 0.9 * rand;
    devices(k).dcOffsetI      = 0.02 * randn;
    devices(k).dcOffsetQ      = 0.02 * randn;
    devices(k).sco            = 1 + (randn * 50e-6);
    devices(k).phaseNoiseStd  = 0.002 + 0.005*rand;
end

%% GENERATE SIGNAL FOR EACH DEVICE
deviceSig = cell(1, numDevices);

for k = 1:numDevices
    sig = baseSig;

    % (1) Sampling Clock Offset
    P = 1000;  
    Q = round(P / devices(k).sco);
    sig = resample(sig, P, Q);
    sig = sig(1:Nsym);

    % (2) Frequency Offset
    sig = sig .* exp(1j * 2 * pi * devices(k).freqOffset * t);

    % (3) IQ imbalance
    I = real(sig);
    Qs = imag(sig);
    I = (1 + devices(k).iqGain).*I;
    Qs = (1 - devices(k).iqGain).*Qs;

    Q_rot = Qs*cos(devices(k).iqPhase) + I*sin(devices(k).iqPhase);
    sig = I + 1j * Q_rot;

    % (4) Static phase
    sig = sig .* exp(1j * devices(k).phaseErr);

    % (5) PA nonlinearity
    A = abs(sig);
    sig = sig ./ (1 + A.^2).^(devices(k).paAlpha/2);

    % (6) DC offsets
    sig = sig + devices(k).dcOffsetI + 1j*devices(k).dcOffsetQ;

    % (7) Phase noise
    pn = cumsum(devices(k).phaseNoiseStd * randn(Nsym,1));
    sig = sig .* exp(1j * pn);

    deviceSig{k} = sig;
end

save("RF_DeviceSignals.mat","deviceSig","numDevices","Nsym","fs");
fprintf("\nSaved: RF_DeviceSignals.mat\n");
%% =============================================================
%   PART 2 — PREPROCESSED DATASET (TIME + FFT Features)
%   + PLOTS FOR 5 RANDOM DEVICES
% =============================================================

clear; clc;
load("RF_DeviceSignals.mat");   % loads: deviceSig, numDevices, Nsym, fs

fprintf("\n=== PART 2: DATASET CREATION + DEVICE PLOTS ===\n");
fprintf("Loaded %d devices, Nsym = %d, fs = %.2e\n", numDevices, Nsym, fs);

%% -------------------------------------------------------------
%   VISUALIZE ANY 5 DEVICES (from full-length signals)
% -------------------------------------------------------------
numPlot = 5;                          % number of devices to show
plotIdx = randperm(numDevices, numPlot);

for p = 1:numPlot
    devID = plotIdx(p);
    sig   = deviceSig{devID};        % complex baseband: length Nsym
    I     = real(sig);
    Q     = imag(sig);

    % For time plots, just show first 2000 samples for clarity
    Nshow = min(2000, length(sig));

    figure('Name',sprintf('Device %d',devID),'NumberTitle','off');

    %% (1) I(t)
    subplot(3,2,1);
    plot(I(1:Nshow),'LineWidth',1);
    grid on;
    title(sprintf('Device %d – I(t)', devID));
    xlabel('Samples'); ylabel('I');

    %% (2) Q(t)
    subplot(3,2,2);
    plot(Q(1:Nshow),'LineWidth',1);
    grid on;
    title(sprintf('Device %d – Q(t)', devID));
    xlabel('Samples'); ylabel('Q');

    %% (3) Constellation (scatter of full signal, or subset if large)
    subplot(3,2,3);
    plot(I(1:10:end), Q(1:10:end),'.','MarkerSize',2);
    grid on; axis equal;
    title('Constellation');
    xlabel('I'); ylabel('Q');

    %% (4) Spectrum
    subplot(3,2,4);
    Nfft = 4096;
    S    = fftshift(abs(fft(sig, Nfft)));
    f    = linspace(-fs/2, fs/2, Nfft);
    plot(f, 20*log10(S + 1e-12),'LineWidth',1);
    grid on;
    title('Spectrum (Magnitude, dB)');
    xlabel('Frequency (Hz)'); ylabel('Mag (dB)');

    %% (5) I/Q Trajectory (overlay)
    subplot(3,2,[5 6]);
    plot(I(1:Nshow),'LineWidth',1.1); hold on;
    plot(Q(1:Nshow),'LineWidth',1.1);
    grid on;
    legend('I','Q');
    title('I/Q Trajectory (First Samples)');
    xlabel('Samples'); ylabel('Amplitude');
end

%% -------------------------------------------------------------
%   CREATE FRAME-BASED DATASET X, Y (for DL models)
% -------------------------------------------------------------

frameLen = 1024;                         % samples per frame
framesPerDevice = floor(Nsym / frameLen);

fprintf("\nEach device → %d frames (frameLen = %d)\n", ...
        framesPerDevice, frameLen);

X = [];          % 2 × frameLen × Nframes
Y = [];          % labels (device index)

for k = 1:numDevices
    sig = deviceSig{k};

    for n = 1:framesPerDevice
        idx   = (n-1)*frameLen + (1:frameLen);
        frame = sig(idx);

        % 2 × frameLen (row: I, Q)
        frameIQ = [real(frame).'; imag(frame).'];

        % stack in 3rd dimension
        X = cat(3, X, frameIQ);
        Y = [Y; k];
    end
end

Y = categorical(Y);

fprintf("\nDataset created successfully.\n");
fprintf("Total frames: %d\n", size(X,3));

%% OPTIONAL: Visualize one random frame
sampleID = randi(size(X,3));

figure('Name','Sample Frame from Dataset','NumberTitle','off');
subplot(2,1,1);
plot(X(1,:,sampleID)); grid on;
title(sprintf('Sample Frame %d – I component', sampleID));
xlabel('Samples'); ylabel('I');

subplot(2,1,2);
plot(X(2,:,sampleID)); grid on;
title(sprintf('Sample Frame %d – Q component', sampleID));
xlabel('Samples'); ylabel('Q');

%% -------------------------------------------------------------
%   SAVE DATASET FOR PART 3
% -------------------------------------------------------------
save("RF_Dataset.mat", "X","Y","frameLen","numDevices","framesPerDevice");
fprintf("\nSaved: RF_Dataset.mat (for Part 3 DL training)\n");

%% =============================================================
%   PART 3 — QPSK, 5-FOLD CV USING FIXED BEST HYPERPARAMETERS
% =============================================================
clc; clear; close all;
load("RF_Dataset_QPSK.mat");   % X: 2×frameLen×N, Y: categorical

fprintf("\n=== PART 3: QPSK — 5-FOLD CV WITH FIXED BEST HYPERPARAMETERS ===\n");

rng(1);

K        = 5;
N        = size(X,3);
frameLen = size(X,2);
numDevs  = numel(categories(Y));
classes  = categories(Y);

X4D = reshape(X, 2, frameLen, 1, N);
cv  = cvpartition(Y,'KFold',K);

modelNames = ["ShallowCNN","DeepCNN","ResNet","Inception","DenseNet"];

Final = table('Size',[5 15], ...
    'VariableTypes',["string", repmat("double",1,14)], ...
    'VariableNames',["Model", ...
    "MeanFoldAcc","StdFoldAcc", ...
    "WA","UA", ...
    "MacroPrecision","MacroRecall","MacroF1", ...
    "WeightedPrecision","WeightedRecall","WeightedF1", ...
    "MicroPrecision","MicroRecall","MicroF1","Kappa"]);

for modelID = 1:5

    fprintf("\n=====================================================\n");
    fprintf("MODEL: %s\n", modelNames(modelID));
    fprintf("=====================================================\n");

    foldAcc = zeros(K,1);
    Csum    = zeros(numDevs,numDevs);

    for fold = 1:K
        fprintf("\n---------------- Fold %d / %d ----------------\n", fold, K);

        tr = training(cv, fold);
        te = test(cv, fold);

        XTrain = X4D(:,:,:,tr);
        YTrain = Y(tr);
        XTest  = X4D(:,:,:,te);
        YTest  = Y(te);

        switch modelID

            %% A — Shallow CNN
            case 1
                % Best used: F=64, FC=64, Drop=0.20, LR=1e-4, Epochs=25
                layers = [
                    imageInputLayer([2 frameLen 1],"Normalization","none")

                    convolution2dLayer([1 7],64,"Padding","same")
                    batchNormalizationLayer
                    reluLayer
                    maxPooling2dLayer([1 2],"Stride",[1 2])

                    convolution2dLayer([1 5],128,"Padding","same")
                    batchNormalizationLayer
                    reluLayer

                    globalAveragePooling2dLayer
                    fullyConnectedLayer(64)
                    reluLayer
                    dropoutLayer(0.2)
                    fullyConnectedLayer(numDevs)
                    softmaxLayer
                    classificationLayer
                ];

                opts = trainingOptions("adam", ...
                    "MaxEpochs",25, ...
                    "MiniBatchSize",32, ...
                    "InitialLearnRate",1e-4, ...
                    "Shuffle","every-epoch", ...
                    "Verbose",false, ...
                    "Plots","none");

            %% B — Deep CNN
            case 2
                % Best used: F=80, FC=128, Drop=0.40, LR=1e-4, Epochs=30
                layers = [
                    imageInputLayer([2 frameLen 1],"Normalization","none")

                    convolution2dLayer([1 7],80,"Padding","same")
                    batchNormalizationLayer
                    reluLayer
                    maxPooling2dLayer([1 2],"Stride",[1 2])

                    convolution2dLayer([1 5],160,"Padding","same")
                    batchNormalizationLayer
                    reluLayer

                    convolution2dLayer([1 3],160,"Padding","same")
                    batchNormalizationLayer
                    reluLayer

                    globalAveragePooling2dLayer
                    fullyConnectedLayer(128)
                    reluLayer
                    dropoutLayer(0.4)
                    fullyConnectedLayer(numDevs)
                    softmaxLayer
                    classificationLayer
                ];

                opts = trainingOptions("adam", ...
                    "MaxEpochs",30, ...
                    "MiniBatchSize",32, ...
                    "InitialLearnRate",1e-4, ...
                    "Shuffle","every-epoch", ...
                    "Verbose",false, ...
                    "Plots","none");

            %% C — ResNet
            case 3
                % Best used: F=64, FC=64, Drop=0.50, LR=1e-3, Epochs=25
                baseLayers = [
                    imageInputLayer([2 frameLen 1],"Normalization","none","Name","in")

                    convolution2dLayer([1 7],64,"Padding","same","Name","c1")
                    batchNormalizationLayer("Name","bn1")
                    reluLayer("Name","relu1")

                    convolution2dLayer([1 3],64,"Padding","same","Name","c2")
                    batchNormalizationLayer("Name","bn2")
                    reluLayer("Name","relu2")

                    additionLayer(2,"Name","add")
                    reluLayer("Name","out")

                    globalAveragePooling2dLayer("Name","gap")
                    fullyConnectedLayer(64,"Name","fc1")
                    reluLayer("Name","relu_fc")
                    dropoutLayer(0.5,"Name","drop")
                    fullyConnectedLayer(numDevs,"Name","fc_out")
                    softmaxLayer("Name","soft")
                    classificationLayer("Name","class")
                ];

                LG = layerGraph(baseLayers);
                LG = connectLayers(LG,"relu1","add/in2");
                layers = LG;

                opts = trainingOptions("adam", ...
                    "MaxEpochs",25, ...
                    "MiniBatchSize",32, ...
                    "InitialLearnRate",1e-3, ...
                    "Shuffle","every-epoch", ...
                    "Verbose",false, ...
                    "Plots","none");

            %% D — Inception
            case 4
                % Best used: F=32, FC=128, Drop=0.50, LR=5e-4, Epochs=25
                F = 32;
                LG = layerGraph();

                inp = imageInputLayer([2 frameLen 1],"Normalization","none","Name","in");
                LG = addLayers(LG,inp);

                b1 = [
                    convolution2dLayer([1 1],F,"Padding","same","Name","b1_conv")
                    batchNormalizationLayer("Name","b1_bn")
                    reluLayer("Name","b1_relu")
                ];
                LG = addLayers(LG,b1);

                b2 = [
                    convolution2dLayer([1 3],F,"Padding","same","Name","b2_conv")
                    batchNormalizationLayer("Name","b2_bn")
                    reluLayer("Name","b2_relu")
                ];
                LG = addLayers(LG,b2);

                b3 = [
                    convolution2dLayer([1 5],F,"Padding","same","Name","b3_conv")
                    batchNormalizationLayer("Name","b3_bn")
                    reluLayer("Name","b3_relu")
                ];
                LG = addLayers(LG,b3);

                concat = depthConcatenationLayer(3,"Name","concat");
                LG = addLayers(LG,concat);

                tail = [
                    convolution2dLayer([1 3],2*F,"Padding","same","Name","t_conv")
                    batchNormalizationLayer("Name","t_bn")
                    reluLayer("Name","t_relu")

                    globalAveragePooling2dLayer("Name","gap")
                    fullyConnectedLayer(128,"Name","fc1")
                    reluLayer("Name","relu_fc")
                    dropoutLayer(0.5,"Name","drop")
                    fullyConnectedLayer(numDevs,"Name","fc_out")
                    softmaxLayer("Name","soft")
                    classificationLayer("Name","class")
                ];
                LG = addLayers(LG,tail);

                LG = connectLayers(LG,"in","b1_conv");
                LG = connectLayers(LG,"in","b2_conv");
                LG = connectLayers(LG,"in","b3_conv");

                LG = connectLayers(LG,"b1_relu","concat/in1");
                LG = connectLayers(LG,"b2_relu","concat/in2");
                LG = connectLayers(LG,"b3_relu","concat/in3");

                LG = connectLayers(LG,"concat","t_conv");
                layers = LG;

                opts = trainingOptions("adam", ...
                    "MaxEpochs",25, ...
                    "MiniBatchSize",32, ...
                    "InitialLearnRate",5e-4, ...
                    "Shuffle","every-epoch", ...
                    "Verbose",false, ...
                    "Plots","none");

            %% E — DenseNet
            case 5
                % Best used: G=32, FC=128, Drop=0.50, LR=5e-4, Epochs=25
                G = 32;
                LG = layerGraph();

                inp = imageInputLayer([2 frameLen 1],"Normalization","none","Name","in");
                LG = addLayers(LG,inp);

                c1 = [
                    convolution2dLayer([1 7],G,"Padding","same","Name","c1")
                    batchNormalizationLayer("Name","bn1")
                    reluLayer("Name","relu1")
                ];
                LG = addLayers(LG,c1);

                c2 = [
                    convolution2dLayer([1 5],G,"Padding","same","Name","c2")
                    batchNormalizationLayer("Name","bn2")
                    reluLayer("Name","relu2")
                ];
                LG = addLayers(LG,c2);

                concat1 = depthConcatenationLayer(2,"Name","concat1");
                LG = addLayers(LG,concat1);

                c3 = [
                    convolution2dLayer([1 3],G,"Padding","same","Name","c3")
                    batchNormalizationLayer("Name","bn3")
                    reluLayer("Name","relu3")
                ];
                LG = addLayers(LG,c3);

                concat2 = depthConcatenationLayer(2,"Name","concat2");
                LG = addLayers(LG,concat2);

                tail = [
                    globalAveragePooling2dLayer("Name","gap")
                    fullyConnectedLayer(128,"Name","fc1")
                    reluLayer("Name","relu_fc")
                    dropoutLayer(0.5,"Name","drop")
                    fullyConnectedLayer(numDevs,"Name","fc_out")
                    softmaxLayer("Name","soft")
                    classificationLayer("Name","class")
                ];
                LG = addLayers(LG,tail);

                LG = connectLayers(LG,"in","c1");
                LG = connectLayers(LG,"in","c2");
                LG = connectLayers(LG,"relu1","concat1/in1");
                LG = connectLayers(LG,"relu2","concat1/in2");
                LG = connectLayers(LG,"concat1","c3");
                LG = connectLayers(LG,"concat1","concat2/in1");
                LG = connectLayers(LG,"relu3","concat2/in2");
                LG = connectLayers(LG,"concat2","gap");
                layers = LG;

                opts = trainingOptions("adam", ...
                    "MaxEpochs",25, ...
                    "MiniBatchSize",32, ...
                    "InitialLearnRate",5e-4, ...
                    "Shuffle","every-epoch", ...
                    "Verbose",false, ...
                    "Plots","none");
        end

        net   = trainNetwork(XTrain,YTrain,layers,opts);
        YPred = classify(net,XTest);

        C = confusionmat(YTest, YPred, 'Order', categorical(classes));
        Csum = Csum + C;

        acc = mean(YPred == YTest);
        foldAcc(fold) = acc;

        fprintf("Fold %d Accuracy = %.4f (%.2f%%)\n", fold, acc, 100*acc);
    end

    % ===== final combined report =====
    TP = diag(Csum);
    FP = sum(Csum,1)' - TP;
    FN = sum(Csum,2)  - TP;
    support = sum(Csum,2);
    totalSupport = sum(support);

    precision = TP ./ max(TP+FP, eps);
    recall    = TP ./ max(TP+FN, eps);
    f1        = 2*(precision.*recall) ./ max(precision+recall, eps);

    macroPrecision = mean(precision);
    macroRecall    = mean(recall);
    macroF1        = mean(f1);

    weightedPrecision = sum(precision .* support) / max(totalSupport, eps);
    weightedRecall    = sum(recall    .* support) / max(totalSupport, eps);
    weightedF1        = sum(f1        .* support) / max(totalSupport, eps);

    microTP = sum(TP);
    microFP = sum(FP);
    microFN = sum(FN);

    microPrecision = microTP / max(microTP + microFP, eps);
    microRecall    = microTP / max(microTP + microFN, eps);
    microF1        = 2*microPrecision*microRecall / max(microPrecision + microRecall, eps);

    WA = sum(TP) / max(sum(Csum(:)), eps);
    UA = macroRecall;

    Ntot = sum(Csum(:));
    po = sum(diag(Csum)) / max(Ntot, eps);
    pe = sum(sum(Csum,1)'.*sum(Csum,2)) / max(Ntot^2, eps);
    kappa = (po - pe) / max(1 - pe, eps);

    fprintf("\n=====================================================\n");
    fprintf("FINAL REPORT — %s\n", modelNames(modelID));
    fprintf("=====================================================\n");
    fprintf("Mean Fold Accuracy = %.4f ± %.4f  (%.2f%% ± %.2f%%)\n", mean(foldAcc), std(foldAcc), 100*mean(foldAcc), 100*std(foldAcc));
    fprintf("Weighted Accuracy (WA)   = %.4f (%.2f%%)\n", WA, 100*WA);
    fprintf("Unweighted Accuracy (UA) = %.4f (%.2f%%)\n", UA, 100*UA);
    fprintf("Macro Precision          = %.4f\n", macroPrecision);
    fprintf("Macro Recall             = %.4f\n", macroRecall);
    fprintf("Macro F1                 = %.4f\n", macroF1);
    fprintf("Weighted Precision       = %.4f\n", weightedPrecision);
    fprintf("Weighted Recall          = %.4f\n", weightedRecall);
    fprintf("Weighted F1              = %.4f\n", weightedF1);
    fprintf("Micro Precision          = %.4f\n", microPrecision);
    fprintf("Micro Recall             = %.4f\n", microRecall);
    fprintf("Micro F1                 = %.4f\n", microF1);
    fprintf("Cohen Kappa              = %.4f\n", kappa);

    disp('Confusion Matrix (rows=true, cols=predicted):');
    disp(Csum);

    T = table((1:numDevs).', precision, recall, f1, support, ...
        'VariableNames',{'Class','Precision','Recall','F1','Support'});
    disp(T);

    figure;
    confusionchart(Csum, classes);
    title("Final Confusion Matrix — " + modelNames(modelID) + " (QPSK)");

    Final.Model(modelID)             = modelNames(modelID);
    Final.MeanFoldAcc(modelID)       = mean(foldAcc);
    Final.StdFoldAcc(modelID)        = std(foldAcc);
    Final.WA(modelID)                = WA;
    Final.UA(modelID)                = UA;
    Final.MacroPrecision(modelID)    = macroPrecision;
    Final.MacroRecall(modelID)       = macroRecall;
    Final.MacroF1(modelID)           = macroF1;
    Final.WeightedPrecision(modelID) = weightedPrecision;
    Final.WeightedRecall(modelID)    = weightedRecall;
    Final.WeightedF1(modelID)        = weightedF1;
    Final.MicroPrecision(modelID)    = microPrecision;
    Final.MicroRecall(modelID)       = microRecall;
    Final.MicroF1(modelID)           = microF1;
    Final.Kappa(modelID)             = kappa;
end

fprintf("\n================== FINAL SUMMARY TABLE — QPSK ==================\n");
disp(Final);

writetable(Final,"QPSK_5Fold_FixedBestParams_Results.csv");
fprintf("Saved: QPSK_5Fold_FixedBestParams_Results.csv\n");
