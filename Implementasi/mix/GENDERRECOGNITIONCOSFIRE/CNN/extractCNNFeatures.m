function [accuracy,datacnn] = extractCNNFeatures(dataFolder,dirlist,cnnType)
% Matlab Pretrained Convolutional Neural Networks https://nl.mathworks.com/help/nnet/ug/pretrained-convolutional-neural-networks.html

% Create folder results/cnn/cnnmodel if it does not exist
if ~exist(strcat(dataFolder,'results/cnn/',cnnType))
    mkdir(strcat(dataFolder,'results/cnn/',cnnType));
end

% Initialize global variables
featureLayer = '';
miniBatchSize = 10;

% Load pre-trained CNN
if (strcmp(cnnType,'alexnet'))
   net = alexnet(); 
   featureLayer = 'fc7';
   sz = net.Layers(1).InputSize;
elseif (strcmp(cnnType,'vggnet16'))
   net = vgg16;
   featureLayer = 'fc7';
   sz = net.Layers(1).InputSize;
elseif (strcmp(cnnType,'vggnet19'))
   net = vgg19;
   featureLayer = 'fc7';
   sz = net.Layers(1).InputSize;
elseif (strcmp(cnnType,'googlenet'))
   net = googlenet;
   featureLayer = 'fc';
   sz = net.Layers(1).InputSize;
elseif (strcmp(cnnType,'custom'))
   featureLayer = 'fc7';
   % Specify files to import 
   protofile = 'deploy_gender.prototxt'; 
   % Import network 
   layers = importCaffeLayers(protofile);
   sz = layers(1).InputSize;
end  

% Load Pretrained Model
[modelExists,pretrainedNetTransfer] = modelCNNExist(dataFolder,cnnType);
if (modelExists == 1)
    netTransfer = pretrainedNetTransfer.netTransfer; 
end

% Load Datasets
[trainingSet, testSet] = getCNNDataset(dirlist,sz);

% If model does not exist, then train CNN
if (modelExists == 0)
    numClasses = numel(categories(trainingSet.Labels));
    % AlexNet
    if (strcmp(cnnType,'alexnet') || strcmp(cnnType,'vggnet16')|| strcmp(cnnType,'vggnet19'))
        % Transfer Layers to New Network
        layersTransfer = net.Layers(1:end-3);
        layers = [
            layersTransfer
            fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
            softmaxLayer
            classificationLayer];
       validationFrequency = 3; %floor((numel(trainingSet.Labels)/miniBatchSize));
       options = trainingOptions('sgdm',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',15,...
        'InitialLearnRate',1e-4,...
        'Verbose',false,...
        'Plots','training-progress'); %,...
        %'ValidationData',testSet,...
        %'ValidationFrequency',validationFrequency);
    
    elseif (strcmp(cnnType,'googlenet'))
        lgraph = layerGraph(net);
        lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});
        newLayers = [
            fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor', 20)
            softmaxLayer('Name','softmax')
            classificationLayer('Name','classoutput')];
        lgraph = addLayers(lgraph,newLayers);
        layers = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');
        validationFrequency = floor((numel(trainingSet.Labels)/miniBatchSize));
        options = trainingOptions('sgdm',...
        'MiniBatchSize',10,...
        'MaxEpochs',8,...
        'InitialLearnRate',1e-4,...
        'VerboseFrequency',1,...
        'Plots','training-progress',...
        'ValidationData',testSet,...
        'ValidationFrequency',validationFrequency);
    
    elseif (strcmp(cnnType,'custom'))
%         layers = [...
%                 imageInputLayer(sz)
%                 convolution2dLayer(5,32)
%                 reluLayer
%                 maxPooling2dLayer(2,'Stride',2)
%                 convolution2dLayer(5,64)
%                 reluLayer
%                 maxPooling2dLayer(2,'Stride',2)
%                 fullyConnectedLayer(4096,'Name','fc1')
%                 fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20, 'Name','fc2')
%                 softmaxLayer
%                 classificationLayer];
        miniBatchSize = 10;
        validationFrequency = floor((numel(trainingSet.Labels)/miniBatchSize));
        options = trainingOptions('sgdm',...
                'MaxEpochs',10,...
                'InitialLearnRate',1e-4,...
                'Verbose',false,...
                'Plots','training-progress',...
                'ValidationData',testSet,...
                'ValidationFrequency',validationFrequency);
    end
    
    % Train Network
    fprintf('Training Network');
    tic;
    netTransfer = trainNetwork(trainingSet,layers,options);
    toc
    
    % Save model
    save(strcat(dataFolder,'results/cnn/',cnnType,'/netTransfer.mat'),'netTransfer');

    % Classify Test Images
    predictedLabels = classify(netTransfer,testSet);
    accuracy = mean(predictedLabels == testSet.Labels);
    fprintf('CNN Recognition Rate: %2.6f\n',accuracy);

    % Extract Training Features from the last layer of CNN
    fprintf('Extracting Traning Features');
    tic;
    % trainingFeaturesCNN = activations(netTransfer, trainingSet, featureLayer, 'MiniBatchSize', miniBatchSize, 'OutputAs', 'columns');
    trainingFeaturesCNN = activations(netTransfer, trainingSet, featureLayer);
    toc

    % Extract Test Features from the last layer of CNN
    fprintf('Extracting Test Features');
    tic;
    % testFeaturesCNN = activations(netTransfer, testSet, featureLayer, 'MiniBatchSize',miniBatchSize);
    testFeaturesCNN = activations(netTransfer, testSet, featureLayer);
    toc
    
    % Set all variables to a struct
    if (strcmp(cnnType,'googlenet'))
       [straining1,igntraining1,straining3,igntraining2] = size(trainingFeaturesCNN);
       datacnn.training.features = reshape(permute(trainingFeaturesCNN,[1 3 2 4]),straining1*straining3,[]);
       datacnn.training.features = datacnn.training.features;
       
       [stesting1,igntesting1,stesting3,igntesting2] = size(testFeaturesCNN);
       datacnn.testing.features = reshape(permute(testFeaturesCNN,[1 3 2 4]),stesting1*stesting3,[]);
       datacnn.testing.features = datacnn.testing.features;
    else
        datacnn.training.features = trainingFeaturesCNN;
        datacnn.testing.features = testFeaturesCNN;
    end
    
    datacnn.training.labels = trainingSet.Labels;
    datacnn.testing.labels = testSet.Labels;

    % Normalize features between 0 and 1
    datacnn.training.normalizedfeatures = datacnn.training.features - min(datacnn.training.features(:));
    datacnn.training.normalizedfeatures = datacnn.training.normalizedfeatures ./ max(datacnn.training.normalizedfeatures(:));
    datacnn.testing.normalizedfeatures = datacnn.testing.features - min(datacnn.testing.features(:));
    datacnn.testing.normalizedfeatures = datacnn.testing.normalizedfeatures ./ max(datacnn.testing.normalizedfeatures(:));

    % Save cnn features and its accuracy
    save(strcat(dataFolder,'results/cnn/',cnnType,'/datacnn.mat'),'datacnn');
    save(strcat(dataFolder,'results/cnn/',cnnType,'/accuracy.mat'),'accuracy');
else
    % Load CNN Features
    [dataExists,datacnn] = getCNNFeatures(dataFolder,cnnType);
    % Load CNN Accuracy
    [accuracyExists,accuracy] = getCNNAccuracy(dataFolder,cnnType);
    
    % If accuracy does not exist, then recompute
    if (accuracyExists == 0)
        % Classify Test Images
        predictedLabels = classify(netTransfer,testSet);
        accuracy = mean(predictedLabels == testSet.Labels);
        fprintf('CNN Recognition Rate: %2.6f\n',accuracy);
    end
    
    % If features do not exist, then recompute
    if (dataExists == 0)
        % Extract Training Features from the last layer of CNN
        fprintf('Extracting Traning Features');
        tic;
        % trainingFeaturesCNN = activations(netTransfer, trainingSet, featureLayer, 'MiniBatchSize', miniBatchSize, 'OutputAs', 'columns');
        trainingFeaturesCNN = activations(netTransfer, trainingSet, featureLayer);
        toc

        % Extract Test Features from the last layer of CNN
        fprintf('Extracting Test Features');
        tic;
        %testFeaturesCNN = activations(netTransfer, testSet, featureLayer, 'MiniBatchSize',miniBatchSize);
        testFeaturesCNN = activations(netTransfer, testSet, featureLayer);
        toc

        % Set all variables to a struct
        if (strcmp(cnnType,'googlenet'))
           [straining1,igntraining1,straining3,igntraining2] = size(trainingFeaturesCNN);
           datacnn.training.features = reshape(permute(trainingFeaturesCNN,[1 3 2 4]),straining1*straining3,[]);
           datacnn.training.features = datacnn.training.features;

           [stesting1,igntesting1,stesting3,igntesting2] = size(testFeaturesCNN);
           datacnn.testing.features = reshape(permute(testFeaturesCNN,[1 3 2 4]),stesting1*stesting3,[]);
           datacnn.testing.features = datacnn.testing.features;
        else
            datacnn.training.features = trainingFeaturesCNN;
            datacnn.testing.features = testFeaturesCNN;
        end
        
        datacnn.training.labels = trainingSet.Labels;
        datacnn.testing.labels = testSet.Labels;
        
        % Normalize features between 0 and 1
        datacnn.training.normalizedfeatures = datacnn.training.features - min(datacnn.training.features(:));
        datacnn.training.normalizedfeatures = datacnn.training.normalizedfeatures ./ max(datacnn.training.normalizedfeatures(:));
        datacnn.testing.normalizedfeatures = datacnn.testing.features - min(datacnn.testing.features(:));
        datacnn.testing.normalizedfeatures = datacnn.testing.normalizedfeatures ./ max(datacnn.testing.normalizedfeatures(:));

        % Save model
        save(strcat(dataFolder,'results/cnn/',cnnType,'/datacnn.mat'),'datacnn');
    end
end