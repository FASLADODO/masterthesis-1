%=========================== SVM ==============================
% Train A Multiclass SVM Classifier Using CNN Features
% classifier_svm = fitcecoc(trainingSet, trainingLabels, ...
% 'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

% % Evaluate Classifier
% % Extract test features using the CNN
% testFeatures = activations(net, testSet, featureLayer, 'MiniBatchSize',32);
% % Pass CNN image features to trained classifier
% predictedLabels = predict(classifier, testFeatures);
% % Get the known labels
% testLabels = testSet.Labels;
% % Tabulate the results using a confusion matrix.
% confMat = confusionmat(testLabels, predictedLabels)
% % Convert confusion matrix into percentage form
% confMatPercentage = bsxfun(@rdivide,confMat,sum(confMat,2))


dataLayer = 'data';
trainingInputs = activations(net, trainingSet, dataLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');