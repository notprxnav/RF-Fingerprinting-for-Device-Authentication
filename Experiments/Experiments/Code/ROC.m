%% ================================================================
%   MULTI-MODEL ROC PLOTS (A,B,C,D,E)
%   Requires: netA, netB, netC, netD, netE, XTest, YTest
% ================================================================

models = {netA, netB, netC, netD, netE};
modelNames = ["Shallow CNN", "Deep CNN", "ResNet", "Inception", "DenseNet"];

numModels = length(models);
classes = categories(YTest);
numClasses = length(classes);

figure;
tiledlayout(3,2);   % 5 models in grid

for m = 1:numModels
    nexttile;
    net = models{m};

    % ----- Get class scores -----
    scores = predict(net, XTest);   % N × numClasses

    % Convert YTest → one-hot
    Ytrue = onehotencode(YTest, 2); % N × numClasses

    hold on;
    for c = 1:numClasses
        [Xroc, Yroc, ~, AUC] = perfcurve(Ytrue(:,c), scores(:,c), 1);
        plot(Xroc, Yroc, 'LineWidth', 1.5);
    end
    hold off;

    grid on;
    title(modelNames(m) + " — ROC");
    xlabel("False Positive Rate");
    ylabel("True Positive Rate");
    legend(classes, "Location","SouthEast");
end

sgtitle("ROC Curves for All 5 Models");
