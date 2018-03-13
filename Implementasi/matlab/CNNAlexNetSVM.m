% CNN using AlexNet Architecture
% Library : https://nl.mathworks.com/help/vision/examples/image-category-classification-using-deep-learning.html
% helpview('nnet','alexnet')

%============================== CNN =================================
% Load pre-trained AlexNet
% net = alexnet();
% sz = net.Layers(1).InputSize;
% I = imread('training/1/3.jpg');
% I = I(1:sz(1),1:sz(2),1:sz(3));
% label = classify(net,I)

% % Load Images
numberoftrainingsample = 822;
rootFolder = fullfile('./training'); % define output folder
categories = {'male', 'female'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

% %Verify total numbers of each category
 tbl = countEachLabel(imds)
% 
% % Find the first instance of an image for each category
% % cat = find(imds.Labels == 'cat', 1);
% % dog = find(imds.Labels == 'dog', 1);
% % 
% % %Plot the first instance of an image
% % figure
% % subplot(1,3,1);
% % imshow(readimage(imds,cat))
% % subplot(1,3,2);
% % imshow(readimage(imds,dog))
% 
% % Load pre-trained AlexNet
net = alexnet();
% 
% % View the CNN architecture
% net.Layers;
% 
% % Inspect the first layer
% %net.Layers(1)
% 
% % Inspect the last layer
% %net.Layers(end)
% 
% % Number of class names for ImageNet classification task
% numel(net.Layers(end).ClassNames);
% 
% % Use splitEachLabel method to trim the set.
% imds = splitEachLabel(imds, numberoftrainingsample, 'randomize');
% 
% % Set the ImageDatastore ReadFcn
% imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
% 
% % Prepare Training and Test Image Sets
% [trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');
% 
% % Extract Training Features from the last layer of CNN
featureLayer = 'fc7';
trainingFeatures = activations(net, trainingSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% 
% %=========================== SVM ==============================
% % Train A Multiclass SVM Classifier Using CNN Features
% classifier = fitcecoc(trainingFeatures, trainingLabels, ...
% 'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
% 
% % Evaluate Classifier
% % Extract test features using the CNN
% testFeatures = activations(net, testSet, featureLayer, 'MiniBatchSize',32);
% % Pass CNN image features to trained classifier
% predictedLabels = predict(classifier, testFeatures);
% % Get the known labels
% testLabels = testSet.Labels;
% % Tabulate the results using a confusion matrix.
% confMat = confusionmat(testLabels, predictedLabels);
% % Convert confusion matrix into percentage form
% confMatPercentage = bsxfun(@rdivide,confMat,sum(confMat,2))


