function [confMat,confMatPercentage,accuracy] = testCNNModel(model,data)
% Evaluate Classifier
predictedLabels = predict(model, data.features);
% Tabulate the results using a confusion matrix.
confMat = confusionmat(data.labels, predictedLabels);
% Convert confusion matrix into percentage form
confMatPercentage = bsxfun(@rdivide,confMat,sum(confMat,2));
% Accuracy
accuracy = (confMat(1,1)+confMat(2,2))/sum(sum(confMat));
end