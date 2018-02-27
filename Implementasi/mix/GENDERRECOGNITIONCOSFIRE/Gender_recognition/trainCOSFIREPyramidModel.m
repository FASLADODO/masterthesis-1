% Configure an SVM classification model with a chi-squared kernel
function [modelPyramid, trainingKernel] = trainCOSFIREPyramidModel(outdir,data,noperators)
if ~exist([outdir,filesep,'trainingKernel.mat'])
    kernel = @(X,Y) customkernel(X,Y,noperators,1,'chi2');
    numTrain = size(data.training.desc,1);   
    trainingKernel = [(1:numTrain)', kernel(data.training.desc,data.training.desc)];
    modelPyramid = svmtrain(data.training.labels',trainingKernel,'-t 4 -c 1 -q');
    save([outdir,filesep,'trainingKernel.mat'],'trainingKernel');
    save([outdir,filesep,'modelPyramid.mat'],'modelPyramid');
else
    load([outdir,filesep,'trainingKernel.mat'],'trainingKernel');
    load([outdir,filesep,'modelPyramid.mat'],'modelPyramid');
end