% =========================== SVM Classifier ==============================
% Train A Multiclass SVM Classifier Using CNN Features
classifier = fitcecoc(trainingFeatures, trainingSet.Labels,'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

% Evaluate Classifier
% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures);
% Get the known labels
testLabels = testSet.Labels;
% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);
% Convert confusion matrix into percentage form
confMatPercentage = bsxfun(@rdivide,confMat,sum(confMat,2))
% Accuracy
accuracy = (confMat(1,1)+confMat(2,2))/sum(sum(confMat))

