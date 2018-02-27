% Apply the given SVM model to the test descriptors and quantify the results
function [result, testingKernel, svmscore] = testCOSFIREPyramidModel(outdir,data,noperators,model)
if ~exist([outdir,filesep,'result.mat'])
    kernel = @(X,Y) customkernel(X,Y,noperators,1,'chi2');
    numTest = size(data.testing.desc,1);
    testingKernel = [(1:numTest)' , kernel(data.testing.desc,data.training.desc)];    
    [pl,~,svmscore] = svmpredict(data.testing.labels',testingKernel,model);    
    result = classperf(data.testing.labels,pl);           
    save([outdir,filesep,'result.mat'],'result');
    save([outdir,filesep,'testingKernel.mat'],'testingKernel');
    save([outdir,filesep,'svmscore.mat'],'svmscore');
else
    load([outdir,filesep,'result.mat']);
    load([outdir,filesep,'testingKernel.mat']);
    load([outdir,filesep,'svmscore.mat']);
end