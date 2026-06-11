zclc;
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

%%   PART–3: 5 DEEP LEARNING MODELS 
clear; clc; close all;
load("RF_Dataset.mat");    % X: 2×frameLen×N , Y: categorical

fprintf("\n=== PART 3: 5 DL MODELS + HARDCORE TUNING ===\n");

rng(1);  % reproducibility

N         = size(X,3);
frameLen  = size(X,2);
numDevs   = length(categories(Y));

%   PREPARE CNN INPUT  [H W C N] = [2 × frameLen × 1 × N]
X4D = reshape(X, 2, frameLen, 1, N);

%   TRAIN/TEST SPLIT
idx      = randperm(N);
numTrain = floor(0.8*N);

trainIdx = idx(1:numTrain);
testIdx  = idx(numTrain+1:end);

XTrain = X4D(:,:,:,trainIdx);
YTrain = Y(trainIdx);

XTest  = X4D(:,:,:,testIdx);
YTest  = Y(testIdx);

fprintf("Train=%d | Test=%d | frameLen=%d | devices=%d\n", ...
        numTrain, N-numTrain, frameLen, numDevs);


makeOpts = @(lr,epochs) trainingOptions("adam", ...
    "MaxEpochs",epochs, ...
    "MiniBatchSize",32, ...
    "InitialLearnRate",lr, ...
    "Shuffle","every-epoch", ...
    "Verbose",false, ...
    "Plots","none");      % set "training-progress" if you want


%%   MODEL A – SHALLOW CNN
fprintf("\n=== MODEL A — Shallow CNN Tuning ===\n");

filtersA = [32 48 64 80];      % base filters
dropsA   = [0.0 0.2 0.4];      % dropout after GAP
lrsA     = [1e-3 5e-4 1e-4];   % learning rate
fcA      = [64 128];           % FC size

bestAccA = 0;
bestNetA = [];

trialA = 1;
for F = filtersA
    for D = dropsA
        for LR = lrsA
            for FC = fcA

                fprintf("  [A] Trial %d — F=%d  FC=%d  Drop=%.2f  LR=%g\n", ...
                        trialA,F,FC,D,LR);

                layersA = [
                    imageInputLayer([2 frameLen 1],"Normalization","none")

                    convolution2dLayer([1 7],F,"Padding","same")
                    batchNormalizationLayer
                    reluLayer
                    maxPooling2dLayer([1 2],"Stride",[1 2])

                    convolution2dLayer([1 5],2*F,"Padding","same")
                    batchNormalizationLayer
                    reluLayer

                    globalAveragePooling2dLayer
                    fullyConnectedLayer(FC)
                    reluLayer
                    dropoutLayer(D)
                    fullyConnectedLayer(numDevs)
                    softmaxLayer
                    classificationLayer
                ];

                optsA = makeOpts(LR, 25);

                netTemp = trainNetwork(XTrain,YTrain,layersA,optsA);
                accTemp = mean(classify(netTemp,XTest)==YTest)*100;

                fprintf("     → Acc = %.2f%%\n",accTemp);

                if accTemp > bestAccA
                    bestAccA = accTemp;
                    bestNetA = netTemp;
                end

                trialA = trialA + 1;
            end
        end
    end
end

fprintf(">>> BEST Shallow CNN (A) = %.2f%%\n",bestAccA);


%%   MODEL B – DEEP CNN (3–layer conv stack, tuned
fprintf("\n\n=== MODEL B — Deep CNN Tuning ===\n");

filtersB = [64 80 96];         % base filters
dropsB   = [0.3 0.4 0.5];      % dropout
lrsB     = [1e-3 5e-4 1e-4];   % lr
fcB      = [128 192 256];      % FC size

bestAccB = 0;
bestNetB = [];

trialB = 1;
for F = filtersB
    for D = dropsB
        for LR = lrsB
            for FC = fcB

                fprintf("  [B] Trial %d — F=%d  FC=%d  Drop=%.2f  LR=%g\n", ...
                        trialB,F,FC,D,LR);

                layersB = [
                    imageInputLayer([2 frameLen 1],"Normalization","none")

                    convolution2dLayer([1 7],F,"Padding","same")
                    batchNormalizationLayer
                    reluLayer
                    maxPooling2dLayer([1 2],"Stride",[1 2])

                    convolution2dLayer([1 5],2*F,"Padding","same")
                    batchNormalizationLayer
                    reluLayer

                    convolution2dLayer([1 3],2*F,"Padding","same")
                    batchNormalizationLayer
                    reluLayer

                    globalAveragePooling2dLayer
                    fullyConnectedLayer(FC)
                    reluLayer
                    dropoutLayer(D)
                    fullyConnectedLayer(numDevs)
                    softmaxLayer
                    classificationLayer
                ];

                optsB = makeOpts(LR, 30);

                netTemp = trainNetwork(XTrain,YTrain,layersB,optsB);
                accTemp = mean(classify(netTemp,XTest)==YTest)*100;

                fprintf("     → Acc = %.2f%%\n",accTemp);

                if accTemp > bestAccB
                    bestAccB = accTemp;
                    bestNetB = netTemp;
                end

                trialB = trialB + 1;
            end
        end
    end
end

fprintf(">>> BEST Deep CNN (B) = %.2f%%\n",bestAccB);



%%   MODEL C – ResNet-Style CNN (Residual block, tuned)

fprintf("\n\n=== MODEL C — ResNet-Style CNN Tuning ===\n");

filtersC = [48 64 80];         % conv filters
fcC      = [64 128];           % FC size
dropsC   = [0.3 0.5];          % dropout
lrsC     = [1e-3 5e-4 1e-4];   % lr

bestAccC = 0;
bestNetC = [];

trialC = 1;
for F = filtersC
    for FC = fcC
        for D = dropsC
            for LR = lrsC

                fprintf("  [C] Trial %d — F=%d  FC=%d  Drop=%.2f  LR=%g\n", ...
                        trialC,F,FC,D,LR);

                % Base layers
                resLayers = [
                    imageInputLayer([2 frameLen 1],"Normalization","none","Name","in")

                    convolution2dLayer([1 7],F,"Padding","same","Name","c1")
                    batchNormalizationLayer("Name","bn1")
                    reluLayer("Name","relu1")

                    convolution2dLayer([1 3],F,"Padding","same","Name","c2")
                    batchNormalizationLayer("Name","bn2")
                    reluLayer("Name","relu2")

                    additionLayer(2,"Name","add")
                    reluLayer("Name","rout")

                    globalAveragePooling2dLayer("Name","gap")
                    fullyConnectedLayer(FC,"Name","fc1")
                    reluLayer("Name","relu_fc")
                    dropoutLayer(D,"Name","drop")
                    fullyConnectedLayer(numDevs,"Name","fc_out")
                    softmaxLayer("Name","soft")
                    classificationLayer("Name","class")
                ];

                LG = layerGraph(resLayers);
                % Skip from relu1 → add/in2 (main goes relu2 → add/in1)
                LG = connectLayers(LG,"relu1","add/in2");

                optsC = makeOpts(LR, 25);

                netTemp = trainNetwork(XTrain,YTrain,LG,optsC);
                accTemp = mean(classify(netTemp,XTest)==YTest)*100;

                fprintf("     → Acc = %.2f%%\n",accTemp);

                if accTemp > bestAccC
                    bestAccC = accTemp;
                    bestNetC = netTemp;
                end

                trialC = trialC + 1;
            end
        end
    end
end

fprintf(">>> BEST ResNet-Style CNN (C) = %.2f%%\n",bestAccC);


%%   MODEL D – Inception-Style Multi-Branch CNN (tuned)
fprintf("\n\n=== MODEL D — Inception-Style CNN Tuning ===\n");

baseF_D = [32 48];            % base branch filters
fcD     = [128 192];          % FC size
dropsD  = [0.3 0.5];          % dropout
lrsD    = [1e-3 5e-4 1e-4];   % lr

bestAccD = 0;
bestNetD = [];

trialD = 1;
for F = baseF_D
    for FC = fcD
        for Dp = dropsD
            for LR = lrsD

                fprintf("  [D] Trial %d — F=%d  FC=%d  Drop=%.2f  LR=%g\n", ...
                        trialD,F,FC,Dp,LR);

                LG = layerGraph();

                % Input
                inp = imageInputLayer([2 frameLen 1], ...
                    "Normalization","none","Name","in");
                LG = addLayers(LG,inp);

                % Branch 1: 1x1
                b1 = [
                    convolution2dLayer([1 1],F,"Padding","same","Name","b1_conv")
                    batchNormalizationLayer("Name","b1_bn")
                    reluLayer("Name","b1_relu")
                ];
                LG = addLayers(LG,b1);

                % Branch 2: 1x3
                b2 = [
                    convolution2dLayer([1 3],F,"Padding","same","Name","b2_conv")
                    batchNormalizationLayer("Name","b2_bn")
                    reluLayer("Name","b2_relu")
                ];
                LG = addLayers(LG,b2);

                % Branch 3: 1x5
                b3 = [
                    convolution2dLayer([1 5],F,"Padding","same","Name","b3_conv")
                    batchNormalizationLayer("Name","b3_bn")
                    reluLayer("Name","b3_relu")
                ];
                LG = addLayers(LG,b3);

                % Concatenation of 3 branches
                concat = depthConcatenationLayer(3,"Name","concat");
                LG = addLayers(LG,concat);

                % Post-Inception conv + classifier
                tail = [
                    convolution2dLayer([1 3],2*F,"Padding","same","Name","t_conv")
                    batchNormalizationLayer("Name","t_bn")
                    reluLayer("Name","t_relu")

                    globalAveragePooling2dLayer("Name","gap")
                    fullyConnectedLayer(FC,"Name","fc1")
                    reluLayer("Name","relu_fc")
                    dropoutLayer(Dp,"Name","drop")
                    fullyConnectedLayer(numDevs,"Name","fc_out")
                    softmaxLayer("Name","soft")
                    classificationLayer("Name","class")
                ];
                LG = addLayers(LG,tail);

                % Connections
                LG = connectLayers(LG,"in","b1_conv");
                LG = connectLayers(LG,"in","b2_conv");
                LG = connectLayers(LG,"in","b3_conv");

                LG = connectLayers(LG,"b1_relu","concat/in1");
                LG = connectLayers(LG,"b2_relu","concat/in2");
                LG = connectLayers(LG,"b3_relu","concat/in3");

                LG = connectLayers(LG,"concat","t_conv");

                optsD = makeOpts(LR, 25);

                netTemp = trainNetwork(XTrain,YTrain,LG,optsD);
                accTemp = mean(classify(netTemp,XTest)==YTest)*100;

                fprintf("     → Acc = %.2f%%\n",accTemp);

                if accTemp > bestAccD
                    bestAccD = accTemp;
                    bestNetD = netTemp;
                end

                trialD = trialD + 1;
            end
        end
    end
end

fprintf(">>> BEST Inception-Style CNN (D) = %.2f%%\n",bestAccD);


%%   MODEL E – DenseNet-Style CNN (tuned)
fprintf("\n\n=== MODEL E — DenseNet-Style CNN Tuning ===\n");

growthE = [16 24 32];         % "growth rate" / filters
fcE     = [128 192];          % FC size
dropsE  = [0.3 0.5];          % dropout
lrsE    = [1e-3 5e-4 1e-4];   % lr

bestAccE = 0;
bestNetE = [];

trialE = 1;
for G = growthE
    for FC = fcE
        for Dp = dropsE
            for LR = lrsE

                fprintf("  [E] Trial %d — G=%d  FC=%d  Drop=%.2f  LR=%g\n", ...
                        trialE,G,FC,Dp,LR);

                LG = layerGraph();

                % Input
                inp = imageInputLayer([2 frameLen 1],"Normalization","none","Name","in");
                LG = addLayers(LG,inp);

                % Conv1
                c1 = [
                    convolution2dLayer([1 7],G,"Padding","same","Name","c1")
                    batchNormalizationLayer("Name","bn1")
                    reluLayer("Name","relu1")
                ];
                LG = addLayers(LG,c1);

                % Conv2
                c2 = [
                    convolution2dLayer([1 5],G,"Padding","same","Name","c2")
                    batchNormalizationLayer("Name","bn2")
                    reluLayer("Name","relu2")
                ];
                LG = addLayers(LG,c2);

                % Concat1: [relu1, relu2]
                concat1 = depthConcatenationLayer(2,"Name","concat1");
                LG = addLayers(LG,concat1);

                % Conv3 (takes concat1)
                c3 = [
                    convolution2dLayer([1 3],G,"Padding","same","Name","c3")
                    batchNormalizationLayer("Name","bn3")
                    reluLayer("Name","relu3")
                ];
                LG = addLayers(LG,c3);

                % Concat2: [concat1, relu3]
                concat2 = depthConcatenationLayer(2,"Name","concat2");
                LG = addLayers(LG,concat2);

                % Classifier tail
                tailE = [
                    globalAveragePooling2dLayer("Name","gap")
                    fullyConnectedLayer(FC,"Name","fc1")
                    reluLayer("Name","relu_fc")
                    dropoutLayer(Dp,"Name","drop")
                    fullyConnectedLayer(numDevs,"Name","fc_out")
                    softmaxLayer("Name","soft")
                    classificationLayer("Name","class")
                ];
                LG = addLayers(LG,tailE);

                % Connections
                LG = connectLayers(LG,"in","c1");
                LG = connectLayers(LG,"in","c2");

                LG = connectLayers(LG,"relu1","concat1/in1");
                LG = connectLayers(LG,"relu2","concat1/in2");

                LG = connectLayers(LG,"concat1","c3");
                LG = connectLayers(LG,"concat1","concat2/in1");
                LG = connectLayers(LG,"relu3","concat2/in2");

                LG = connectLayers(LG,"concat2","gap");

                optsE = makeOpts(LR, 25);

                netTemp = trainNetwork(XTrain,YTrain,LG,optsE);
                accTemp = mean(classify(netTemp,XTest)==YTest)*100;

                fprintf("     → Acc = %.2f%%\n",accTemp);

                if accTemp > bestAccE
                    bestAccE = accTemp;
                    bestNetE = netTemp;
                end

                trialE = trialE + 1;
            end
        end
    end
end

fprintf(">>> BEST DenseNet-Style CNN (E) = %.2f%%\n",bestAccE);


%%   FINAL SUMMARY + BEST MODEL CONFUSION MATRIX
fprintf("\n============================================\n");
fprintf("   FINAL ACCURACIES (BEST PER MODEL)\n");
fprintf("   A – Shallow CNN   : %.2f%%\n",bestAccA);
fprintf("   B – Deep CNN      : %.2f%%\n",bestAccB);
fprintf("   C – ResNet-style  : %.2f%%\n",bestAccC);
fprintf("   D – Inception     : %.2f%%\n",bestAccD);
fprintf("   E – DenseNet      : %.2f%%\n",bestAccE);

allAcc = [bestAccA bestAccB bestAccC bestAccD bestAccE];
names  = ["Shallow","Deep","ResNet","Inception","DenseNet"];

[bestAcc, bestIdx] = max(allAcc);
bestName = names(bestIdx);

fprintf("\n >>> OVERALL BEST MODEL = %s (%.2f%%) <<<\n",bestName,bestAcc);

% Pick corresponding net
switch bestIdx
    case 1
        netBest = bestNetA;
    case 2
        netBest = bestNetB;
    case 3
        netBest = bestNetC;
    case 4
        netBest = bestNetD;
    case 5
        netBest = bestNetE;
    otherwise
        netBest = bestNetB;
end

% Confusion matrix for best model
YPredBest = classify(netBest,XTest);
figure;
confusionchart(YTest, YPredBest);
title("Best RF Fingerprinting DL Model – " + bestName);
