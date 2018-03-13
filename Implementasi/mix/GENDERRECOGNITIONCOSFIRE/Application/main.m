noperatorspergender = 90;
dataFolder = './GENDER-FERET/';
doFaceDetection = 1;
resizeWidth = 128;

% Update the Matlab path with the following
addpath('../COSFIRE/');
addpath('../Gabor/');
addpath('../Gender_recognition/');
addpath('../libsvm_3_21/matlab/');
addpath('../CNN/');

%===================== LOAD DATASET ========================

% Configure the image paths. If you want to organize your folder in a
% different way, please modify these lines.
dirlist.maledir             = strcat(dataFolder,filesep,'male');
dirlist.femaledir           = strcat(dataFolder,filesep,'female');
dirlist.trainingmaledir     = strcat(dirlist.maledir,filesep,'training_set');
dirlist.trainingfemaledir   = strcat(dirlist.femaledir,filesep,'training_set');
dirlist.testingmaledir      = strcat(dirlist.maledir,filesep,'test_set');
dirlist.testingfemaledir    = strcat(dirlist.femaledir,filesep,'test_set');

% Create the folder for the results. The results of all the experiments will 
% be saved in this folder.
outdir = [dataFolder,sprintf('/results/noperatorspergender=%d',noperatorspergender)];
if ~exist(outdir)
    mkdir(outdir);
end

%======================= COSFIRE ==========================
% Load the dataset and, if needed, perform face detection and resize. If
% the faces are already available, they are loaded from the file 'dataset.mat'. 
datasetCOSFIRE = getDataset(dataFolder,dirlist,doFaceDetection,resizeWidth);

% Configure the COSFIRE operators. If the operators are already available, 
% they are loaded from the file 'operatorlist.mat'. 
operatorlist = getCOSFIREoperators(outdir,datasetCOSFIRE,noperatorspergender);

% Get training and test descriptors. If the descriptors are already available, 
% they are loaded from the file 'COSFIREdescriptor.mat'.
[data.training.desc,data.testing.desc] = getCOSFIREdescriptors(outdir,datasetCOSFIRE,operatorlist);

% Set training and test labels
data.training.labels = [ones(1,size(datasetCOSFIRE.training.males,3)),ones(1,size(datasetCOSFIRE.training.females,3))*2];
data.testing.labels = [ones(1,size(datasetCOSFIRE.testing.males,3)),ones(1,size(datasetCOSFIRE.testing.females,3))*2];

% Normalize training and test data
data = normalizeData(data,numel(operatorlist));
data.training.features = data.training.desc';
data.testing.features = data.testing.desc';

% Training classification SVM models with Chi-Squared Kernel
[model.pyramid, kernel.training] = trainCOSFIREPyramidModel(outdir,data,numel(operatorlist));

% Evaluate test data with SVM models
[result.info, kernel.testing, result.svmscore] = testCOSFIREPyramidModel(outdir,data,numel(operatorlist),model.pyramid);
accuracycosfire = result.info{16};
fprintf('Recognition Rate: %2.6f\n',accuracycosfire);

%===================== CNN ====================================
cnnType = 'custom';
[accuracycnn, datacnn] = extractCNNFeatures(dataFolder,dirlist,cnnType);

% Merge features CNN and COSFIRE
datacnncosfire.training.features = [datacnn.training.features';data.training.desc'];
datacnncosfire.training.features = datacnncosfire.training.features';
datacnncosfire.training.normalizedfeatures = [datacnn.training.normalizedfeatures';data.training.desc'];
datacnncosfire.training.normalizedfeatures = datacnncosfire.training.normalizedfeatures';

datacnncosfire.testing.features = [datacnn.testing.features';data.testing.desc'];
datacnncosfire.testing.features = datacnncosfire.testing.features';
datacnncosfire.testing.normalizedfeatures = [datacnn.testing.normalizedfeatures';data.testing.desc'];
datacnncosfire.testing.normalizedfeatures = datacnncosfire.testing.normalizedfeatures';

datacnncosfire.training.labels = datacnn.training.labels;
datacnncosfire.testing.labels = datacnn.testing.labels;

%============================ Train SVM ==============================
% Fit Image Classifier CNN
classifierCNN = fitcecoc(datacnn.training.features,datacnn.training.labels);
predictedCNNLabels = predict(classifierCNN,datacnn.testing.features);
accuracyCNN = mean(predictedCNNLabels == datacnn.testing.labels);
fprintf('\nOriginal CNN accuracy %d',accuracyCNN);

% Fit Image Classifier CNN
classifierNormalizedCNN = fitcecoc(datacnn.training.normalizedfeatures,datacnn.training.labels);
predictedNormalizedCNNLabels = predict(classifierNormalizedCNN,datacnn.testing.normalizedfeatures);
accuracyNormalizedCNN = mean(predictedNormalizedCNNLabels == datacnn.testing.labels);
fprintf('\nNormalized CNN accuracy %d',accuracyNormalizedCNN);

% Fit Image Classifier CNNCOSFIRE
classifierCNNCOSFIRE = fitcecoc(datacnncosfire.training.features,datacnncosfire.training.labels);
predictedCNNCSOFIRELabels = predict(classifierCNNCOSFIRE,datacnncosfire.testing.features);
accuracyCNNCOSFIRE = mean(predictedCNNCSOFIRELabels == datacnncosfire.testing.labels);
fprintf('\nCNNCSOFIRE accuracy %d\n',accuracyCNNCOSFIRE);

classifierNormalizedCNNCOSFIRE = fitcecoc(datacnncosfire.training.normalizedfeatures,datacnncosfire.training.labels);
predictedNormalizedCNNCSOFIRELabels = predict(classifierNormalizedCNNCOSFIRE,datacnncosfire.testing.normalizedfeatures);
accuracyNormalizedCNNCOSFIRE = mean(predictedNormalizedCNNCSOFIRELabels == datacnncosfire.testing.labels);
fprintf('\nNormalized CNNCSOFIRE accuracy %d\n',accuracyNormalizedCNNCOSFIRE);

% Plot Accuracy of all models
x = categorical({'Original CNN', 'Normalized CNN','COSFIRE','Original CNN&COSFIRE', 'Normalized CNN&COSFIRE'});
y = [accuracyCNN accuracyNormalizedCNN accuracycosfire accuracyCNNCOSFIRE accuracyNormalizedCNNCOSFIRE];
bar(x,y)
