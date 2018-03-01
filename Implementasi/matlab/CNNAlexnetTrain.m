% CNN using AlextNet
% https://nl.mathworks.com/help/nnet/ref/alexnet.html?searchHighlight=alexnet&s_tid=doc_srchtitle

% Load pre-trained AlexNet
net = alexnet();

% Prepare Training and Test Image Sets
[trainingSet, testSet] = getdataset();

% Transfer Layers to New Network
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(trainingSet.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Train Network
miniBatchSize = 10;
numIterationsPerEpoch = floor(numel(trainingSet.Labels)/miniBatchSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',4,...
    'InitialLearnRate',1e-4,...
    'Verbose',false,...
    'Plots','training-progress',...
    'ValidationData',testSet,...
    'ValidationFrequency',numIterationsPerEpoch);

fprintf('Train Network');
tic;
netTransfer_b = trainNetwork(trainingSet,layers,options);
toc

% % Classify Test Images
% predictedLabels = classify(netTransfer,testSet);
% testLabels = testSet.Labels;
% accuracy = mean(predictedLabels == testLabels)
% 
% % Extract Training Features from the last layer of CNN
% featureLayer = 'fc7';
% fprintf('Extracting Traning Features');
% tic;
% trainingFeatures = activations(netTransfer, trainingSet, featureLayer, 'MiniBatchSize', miniBatchSize, 'OutputAs', 'columns');
% toc
% 
% % Extract Test Features from the last layer of CNN
% fprintf('Extracting Test Features');
% tic;
% testFeatures = activations(netTransfer, testSet, featureLayer, 'MiniBatchSize',miniBatchSize);
% toc