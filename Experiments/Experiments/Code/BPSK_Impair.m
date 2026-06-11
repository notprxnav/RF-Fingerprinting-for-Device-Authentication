%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  BPSK RF Fingerprinting
%  ONE IMPAIRMENT AT A TIME ANALYSIS
%  - 20 devices
%  - 5-fold CV
%  - same fixed hyperparameters as before
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;

%% ===================== GLOBAL SETTINGS =====================
Nsym       = 200000;
M          = 2;              % BPSK
fs         = 1e6;
t          = (0:Nsym-1)' / fs;
numDevices = 20;
frameLen   = 1024;
rng(1);

data    = randi([0 M-1], Nsym, 1);
baseSig = pskmod(data, M);

impairmentNames = { ...
    'SamplingClockOffset', ...
    'FrequencyOffset', ...
    'IQGainImbalance', ...
    'IQPhaseImbalance', ...
    'StaticPhaseError', ...
    'PANonlinearity', ...
    'DCOffsetI', ...
    'DCOffsetQ', ...
    'PhaseNoise'};

modelNames = ["Shallow","Deep","ResNet","Inception","DenseNet"];

%% ===================== MAIN LOOP OVER IMPAIRMENTS =====================
for impID = 1:numel(impairmentNames)

    fprintf('\n\n============================================================\n');
    fprintf(' RUNNING IMPAIRMENT CASE: %s\n', impairmentNames{impID});
    fprintf('============================================================\n');

    %% ---------------------------------------------------------
    %  PART 1 — SIGNAL GENERATION (ONLY ONE IMPAIRMENT ACTIVE)
    %% ---------------------------------------------------------
    devices = struct;

    for k = 1:numDevices
        % neutral/default values
        devices(k).freqOffset    = 0;
        devices(k).iqGain        = 0;
        devices(k).iqPhase       = 0;
        devices(k).phaseErr      = 0;
        devices(k).paAlpha       = 0;     % means no PA compression in our implementation
        devices(k).dcOffsetI     = 0;
        devices(k).dcOffsetQ     = 0;
        devices(k).sco           = 1;
        devices(k).phaseNoiseStd = 0;

        % activate only one impairment
        switch impairmentNames{impID}
            case 'SamplingClockOffset'
                devices(k).sco = 1 + (randn * 50e-6);

            case 'FrequencyOffset'
                devices(k).freqOffset = randi([-700 700]);

            case 'IQGainImbalance'
                devices(k).iqGain = 0.02 * randn;

            case 'IQPhaseImbalance'
                devices(k).iqPhase = deg2rad(2 * randn);

            case 'StaticPhaseError'
                devices(k).phaseErr = deg2rad(6 * randn);

            case 'PANonlinearity'
                devices(k).paAlpha = 0.6 + 0.9 * rand;

            case 'DCOffsetI'
                devices(k).dcOffsetI = 0.02 * randn;

            case 'DCOffsetQ'
                devices(k).dcOffsetQ = 0.02 * randn;

            case 'PhaseNoise'
                devices(k).phaseNoiseStd = 0.002 + 0.005*rand;
        end
    end

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
        I  = real(sig);
        Qs = imag(sig);

        I  = (1 + devices(k).iqGain) .* I;
        Qs = (1 - devices(k).iqGain) .* Qs;

        Q_rot = Qs*cos(devices(k).iqPhase) + I*sin(devices(k).iqPhase);
        sig   = I + 1j*Q_rot;

        % (4) Static phase error
        sig = sig .* exp(1j * devices(k).phaseErr);

        % (5) PA nonlinearity
        if devices(k).paAlpha ~= 0
            A   = abs(sig);
            sig = sig ./ (1 + A.^2).^(devices(k).paAlpha/2);
        end

        % (6) DC offsets
        sig = sig + devices(k).dcOffsetI + 1j*devices(k).dcOffsetQ;

        % (7) Phase noise
        if devices(k).phaseNoiseStd ~= 0
            pn  = cumsum(devices(k).phaseNoiseStd * randn(Nsym,1));
            sig = sig .* exp(1j * pn);
        end

        deviceSig{k} = sig;
    end

    %% ---------------------------------------------------------
    %  PART 2 — DATASET CREATION
    %% ---------------------------------------------------------
    framesPerDevice = floor(Nsym / frameLen);
    X = [];
    Y = [];

    for k = 1:numDevices
        sig = deviceSig{k};

        for n = 1:framesPerDevice
            idx   = (n-1)*frameLen + (1:frameLen);
            frame = sig(idx);

            frameIQ = [real(frame).'; imag(frame).'];
            X = cat(3, X, frameIQ);
            Y = [Y; k];
        end
    end

    Y = categorical(Y);

    %% ---------------------------------------------------------
    %  PART 3 — 5-FOLD CV
    %% ---------------------------------------------------------
    N        = size(X,3);
    numDevs  = numel(categories(Y));
    classes  = categories(Y);
    X4D      = reshape(X, 2, frameLen, 1, N);
    K        = 5;
    cv       = cvpartition(Y,'KFold',K);

    Final = table('Size',[numel(modelNames) 15], ...
        'VariableTypes',["string", repmat("double",1,14)], ...
        'VariableNames',["Model", ...
        "MeanFoldAcc","StdFoldAcc", ...
        "WA","UA", ...
        "MacroPrecision","MacroRecall","MacroF1", ...
        "WeightedPrecision","WeightedRecall","WeightedF1", ...
        "MicroPrecision","MicroRecall","MicroF1","Kappa"]);

    for modelID = 1:numel(modelNames)

        fprintf('\n=====================================================\n');
        fprintf(' IMPAIRMENT: %s | MODEL: %s\n', impairmentNames{impID}, modelNames(modelID));
        fprintf('=====================================================\n');

        foldAcc = zeros(K,1);
        Csum    = zeros(numDevs,numDevs);

        for fold = 1:K
            fprintf('\n---------------- Fold %d / %d ----------------\n', fold, K);

            tr = training(cv, fold);
            te = test(cv, fold);

            XTrain = X4D(:,:,:,tr);
            YTrain = Y(tr);
            XTest  = X4D(:,:,:,te);
            YTest  = Y(te);

            switch modelID
                case 1
                    % Shallow: BPSK best
                    layers = [
                        imageInputLayer([2 frameLen 1],"Normalization","none")
                        convolution2dLayer([1 7],80,"Padding","same")
                        batchNormalizationLayer
                        reluLayer
                        maxPooling2dLayer([1 2],"Stride",[1 2])
                        convolution2dLayer([1 5],160,"Padding","same")
                        batchNormalizationLayer
                        reluLayer
                        globalAveragePooling2dLayer
                        fullyConnectedLayer(128)
                        reluLayer
                        dropoutLayer(0.2)
                        fullyConnectedLayer(numDevs)
                        softmaxLayer
                        classificationLayer
                    ];
                    opts = trainingOptions("adam","MaxEpochs",25,"MiniBatchSize",32, ...
                        "InitialLearnRate",1e-3,"Shuffle","every-epoch","Verbose",false,"Plots","none");

                case 2
                    % Deep: BPSK best
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
                        fullyConnectedLayer(256)
                        reluLayer
                        dropoutLayer(0.3)
                        fullyConnectedLayer(numDevs)
                        softmaxLayer
                        classificationLayer
                    ];
                    opts = trainingOptions("adam","MaxEpochs",30,"MiniBatchSize",32, ...
                        "InitialLearnRate",5e-4,"Shuffle","every-epoch","Verbose",false,"Plots","none");

                case 3
                    % ResNet: BPSK best
                    baseLayers = [
                        imageInputLayer([2 frameLen 1],"Normalization","none","Name","in")
                        convolution2dLayer([1 7],48,"Padding","same","Name","c1")
                        batchNormalizationLayer("Name","bn1")
                        reluLayer("Name","relu1")
                        convolution2dLayer([1 3],48,"Padding","same","Name","c2")
                        batchNormalizationLayer("Name","bn2")
                        reluLayer("Name","relu2")
                        additionLayer(2,"Name","add")
                        reluLayer("Name","out")
                        globalAveragePooling2dLayer("Name","gap")
                        fullyConnectedLayer(128,"Name","fc1")
                        reluLayer("Name","relu_fc")
                        dropoutLayer(0.3,"Name","drop")
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
                    % Inception: BPSK best
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
                        fullyConnectedLayer(192,"Name","fc1")
                        reluLayer("Name","relu_fc")
                        dropoutLayer(0.3,"Name","drop")
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
                        "InitialLearnRate",1e-3,"Shuffle","every-epoch","Verbose",false,"Plots","none");

                case 5
                    % DenseNet: BPSK best
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
                        fullyConnectedLayer(192,"Name","fc1")
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
                        "InitialLearnRate",1e-3,"Shuffle","every-epoch","Verbose",false,"Plots","none");
            end

            net   = trainNetwork(XTrain,YTrain,layers,opts);
            YPred = classify(net,XTest);

            C = confusionmat(YTest, YPred, 'Order', categorical(classes));
            Csum = Csum + C;

            acc = mean(YPred == YTest);
            foldAcc(fold) = acc;

            fprintf('Fold %d Accuracy = %.4f (%.2f%%)\n', fold, acc, 100*acc);
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

        fprintf('\n=====================================================\n');
        fprintf('FINAL REPORT — %s\n', modelNames(modelID));
        fprintf('=====================================================\n');
        fprintf('Mean Fold Accuracy = %.4f ± %.4f  (%.2f%% ± %.2f%%)\n', mean(foldAcc), std(foldAcc), 100*mean(foldAcc), 100*std(foldAcc));
        fprintf('Weighted Accuracy (WA)   = %.4f (%.2f%%)\n', WA, 100*WA);
        fprintf('Unweighted Accuracy (UA) = %.4f (%.2f%%)\n', UA, 100*UA);
        fprintf('Macro Precision          = %.4f\n', macroPrecision);
        fprintf('Macro Recall             = %.4f\n', macroRecall);
        fprintf('Macro F1                 = %.4f\n', macroF1);
        fprintf('Weighted Precision       = %.4f\n', weightedPrecision);
        fprintf('Weighted Recall          = %.4f\n', weightedRecall);
        fprintf('Weighted F1              = %.4f\n', weightedF1);
        fprintf('Micro Precision          = %.4f\n', microPrecision);
        fprintf('Micro Recall             = %.4f\n', microRecall);
        fprintf('Micro F1                 = %.4f\n', microF1);
        fprintf('Cohen Kappa              = %.4f\n', kappa);

        disp('Confusion Matrix (rows=true, cols=predicted):');
        disp(Csum);

        T = table((1:numDevs).', precision, recall, f1, support, ...
            'VariableNames',{'Class','Precision','Recall','F1','Support'});
        disp(T);

        figure;
        confusionchart(Csum, classes);
        title("Final Confusion Matrix — " + modelNames(modelID) + " — " + impairmentNames{impID} + " (BPSK)");

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

    fprintf('\n================== FINAL SUMMARY TABLE — %s (BPSK) ==================\n', impairmentNames{impID});
    disp(Final);

    outName = "BPSK_OneImpairment_" + impairmentNames{impID} + "_Results.csv";
    writetable(Final, outName);
    fprintf('Saved: %s\n', outName);
end