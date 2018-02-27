% If the descriptors have already been computed then we load otherwise
% otherwise we compute them
function [trainingdata,testdata] = getCOSFIREdescriptors(outdir,dataset,operatorlist)

if ~exist([outdir,filesep,'COSFIREdescriptor.mat'])    
    % Compute the descriptors of male and female images for both training and test sets
    COSFIREdescriptor.training.males = getCOSFIREdescriptor(dataset.training.males,operatorlist);
    COSFIREdescriptor.training.females = getCOSFIREdescriptor(dataset.training.females,operatorlist);    
    COSFIREdescriptor.testing.males = getCOSFIREdescriptor(dataset.testing.males,operatorlist);
    COSFIREdescriptor.testing.females = getCOSFIREdescriptor(dataset.testing.females,operatorlist);        
    save([outdir,filesep,'COSFIREdescriptor.mat'],'COSFIREdescriptor');       
else
    load([outdir,filesep,'COSFIREdescriptor.mat']);    
end

% Concatenate the male and female descriptors into training and test matrices
trainingdata = vertcat(COSFIREdescriptor.training.males,COSFIREdescriptor.training.females);
testdata = vertcat(COSFIREdescriptor.testing.males,COSFIREdescriptor.testing.females);