%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  RF Device Fingerprinting with QPSK + Rayleigh Channel
%  PART 1  – QPSK signal generation with impairments
%  PART 2  – Rayleigh + AWGN dataset creation
%  PART 3  – 5-fold CV using fixed best hyperparameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;

%% =============================================================
%   PART 1 — DEVICE SIGNAL GENERATION (QPSK)
% =============================================================
Nsym       = 200000;
M          = 4;          % QPSK
fs         = 1e6;
t          = (0:Nsym-1)' / fs;
numDevices = 20;

data    = randi([0 M-1], Nsym, 1);
baseSig = pskmod(data, M, pi/4);

devices = struct;
for k = 1:numDevices
    devices(k).freqOffset    = randi([-700 700]);
    devices(k).iqGain        = 0.02 * randn;
    devices(k).iqPhase       = deg2rad(2 * randn);
    devices(k).phaseErr      = deg2rad(6 * randn);
    devices(k).paAlpha       = 0.6 + 0.9 * rand;
    devices(k).dcOffsetI     = 0.02 * randn;
    devices(k).dcOffsetQ     = 0.02 * randn;
    devices(k).sco           = 1 + (randn * 50e-6);
    devices(k).phaseNoiseStd = 0.002 + 0.005*rand;
end

deviceSig = cell(1, numDevices);

for k = 1:numDevices
    sig = baseSig;

    P = 1000;
    Q = round(P / devices(k).sco);
    sig = resample(sig, P, Q);
    sig = sig(1:Nsym);

    sig = sig .* exp(1j * 2 * pi * devices(k).freqOffset * t);

    I  = real(sig);
    Qs = imag(sig);
    I  = (1 + devices(k).iqGain) .* I;
    Qs = (1 - devices(k).iqGain) .* Qs;
    Q_rot = Qs*cos(devices(k).iqPhase) + I*sin(devices(k).iqPhase);
    sig   = I + 1j * Q_rot;

    sig = sig .* exp(1j * devices(k).phaseErr);

    A   = abs(sig);
    sig = sig ./ (1 + A.^2).^(devices(k).paAlpha/2);

    sig = sig + devices(k).dcOffsetI + 1j*devices(k).dcOffsetQ;

    pn  = cumsum(devices(k).phaseNoiseStd * randn(Nsym,1));
    sig = sig .* exp(1j * pn);

    deviceSig{k} = sig;
end

save("RF_DeviceSignals_QPSK.mat","deviceSig","numDevices","Nsym","fs");
fprintf("\nSaved: RF_DeviceSignals_QPSK.mat\n");

%% =============================================================
%   PART 2 — RAYLEIGH + AWGN DATASET CREATION
% =============================================================
clear; clc; close all;
load("RF_DeviceSignals_QPSK.mat");

fprintf("\n=== PART 2: QPSK + RAYLEIGH CHANNEL DATASET ===\n");

SNRdB        = 20;
frameLen     = 1024;
blockLen_ch  = frameLen;

deviceSig_R = cell(1,numDevices);

for k = 1:numDevices
    sig = deviceSig{k};
    Nsig = length(sig);
    sig_ch = zeros(Nsig,1);

    numBlocks = ceil(Nsig/blockLen_ch);

    for b = 1:numBlocks
        i1 = (b-1)*blockLen_ch + 1;
        i2 = min(b*blockLen_ch, Nsig);

        xblk = sig(i1:i2);

        h = (randn + 1j*randn)/sqrt(2);
        yblk = h .* xblk;

        Ps = mean(abs(yblk).^2);
        SNRlin = 10^(SNRdB/10);
        Pn = Ps / SNRlin;
        n = sqrt(Pn/2) * (randn(size(yblk)) + 1j*randn(size(yblk)));

        sig_ch(i1:i2) = yblk + n;
    end

    deviceSig_R{k} = sig_ch;
end

fprintf("Rayleigh+AWGN applied at SNR = %g dB\n", SNRdB);

plot2 = randperm(numDevices,2);
for ii = 1:2
    devID = plot2(ii);
    sig0 = deviceSig{devID};
    sigR = deviceSig_R{devID};
    Nshow = 2000;

    figure('Name',sprintf('QPSK Dev %d Clean vs Rayleigh',devID),'NumberTitle','off');

    subplot(2,2,1);
    plot(real(sig0(1:Nshow))); grid on;
    title(sprintf('Clean I(t) - Dev %d',devID));

    subplot(2,2,2);
    plot(real(sigR(1:Nshow))); grid on;
    title(sprintf('Rayleigh I(t) - Dev %d',devID));

    subplot(2,2,3);
    plot(real(sig0(1:10:end)), imag(sig0(1:10:end)), '.'); grid on; axis equal;
    title('Clean Constellation');

    subplot(2,2,4);
    plot(real(sigR(1:10:end)), imag(sigR(1:10:end)), '.'); grid on; axis equal;
    title(sprintf('Rayleigh Constellation (SNR=%gdB)',SNRdB));
end

framesPerDevice = floor(Nsym / frameLen);
X = [];
Y = [];

for k = 1:numDevices
    sig = deviceSig_R{k};

    for n = 1:framesPerDevice
        idx   = (n-1)*frameLen + (1:frameLen);
        frame = sig(idx);

        frameIQ = [real(frame).'; imag(frame).'];
        X = cat(3, X, frameIQ);
        Y = [Y; k];
    end
end

Y = categorical(Y);

save("RF_Dataset_QPSK_Rayleigh.mat","X","Y","frameLen","numDevices","framesPerDevice","SNRdB");
fprintf("Saved: RF_Dataset_QPSK_Rayleigh.mat\n");

%% =============================================================
%   PART 3 — QPSK RAYLEIGH, 5-FOLD CV WITH FIXED BEST PARAMS
% =============================================================
clear; clc; close all;
load("RF_Dataset_QPSK_Rayleigh.mat");

fprintf("\n=== PART 3: QPSK + RAYLEIGH — 5-FOLD CV ===\n");

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
            case 1
                % Shallow: F=64, FC=64, Drop=0.20, LR=1e-4
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
                opts = trainingOptions("adam","MaxEpochs",25,"MiniBatchSize",32, ...
                    "InitialLearnRate",1e-4,"Shuffle","every-epoch","Verbose",false,"Plots","none");

            case 2
                % Deep: F=80, FC=128, Drop=0.40, LR=1e-4
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
                opts = trainingOptions("adam","MaxEpochs",30,"MiniBatchSize",32, ...
                    "InitialLearnRate",1e-4,"Shuffle","every-epoch","Verbose",false,"Plots","none");

            case 3
                % ResNet: F=64, FC=64, Drop=0.50, LR=1e-3
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
                opts = trainingOptions("adam","MaxEpochs",25,"MiniBatchSize",32, ...
                    "InitialLearnRate",1e-3,"Shuffle","every-epoch","Verbose",false,"Plots","none");

            case 4
                % Inception: F=32, FC=128, Drop=0.50, LR=5e-4
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
                opts = trainingOptions("adam","MaxEpochs",25,"MiniBatchSize",32, ...
                    "InitialLearnRate",5e-4,"Shuffle","every-epoch","Verbose",false,"Plots","none");

            case 5
                % DenseNet: G=32, FC=128, Drop=0.50, LR=5e-4
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
                opts = trainingOptions("adam","MaxEpochs",25,"MiniBatchSize",32, ...
                    "InitialLearnRate",5e-4,"Shuffle","every-epoch","Verbose",false,"Plots","none");
        end

        net   = trainNetwork(XTrain,YTrain,layers,opts);
        YPred = classify(net,XTest);

        C = confusionmat(YTest, YPred, 'Order', categorical(classes));
        Csum = Csum + C;

        acc = mean(YPred == YTest);
        foldAcc(fold) = acc;

        fprintf("Fold %d Accuracy = %.4f (%.2f%%)\n", fold, acc, 100*acc);
    end

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
    weightedRecall    = sum(recall .* support) / max(totalSupport, eps);
    weightedF1        = sum(f1 .* support) / max(totalSupport, eps);

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
    title("Final Confusion Matrix — " + modelNames(modelID) + " (QPSK Rayleigh)");

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

fprintf("\n================== FINAL SUMMARY TABLE — QPSK RAYLEIGH ==================\n");
disp(Final);

writetable(Final,"QPSK_Rayleigh_5Fold_FixedBestParams_Results.csv");
fprintf("Saved: QPSK_Rayleigh_5Fold_FixedBestParams_Results.csv\n");