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

% Load the dataset and, if needed, perform face detection and resize. If
% the faces are already available, they are loaded from the file 'dataset.mat'. 
datasetCOSFIRE = getDataset(dataFolder,dirlist,doFaceDetection,resizeWidth);

% Load dataset needed for CNN
sizecnn = 227;
[trainingSet, testSet] = getdataset(dirlist,sizecnn);

%======================= COSFIRE ==========================
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
accuracy_cosfire = result.info.CorrectRate;
fprintf('Recognition Rate: %2.6f\n',result.info.CorrectRate);

%===================== CNN ====================================
% %Load pre-trained AlexNet
% net = alexnet();
% % View the CNN architecture
% net.Layers;
% % Extract Training Features from the last layer of CNN
% featureLayer = 'fc7';
% datacnn.training.features = activations(net, trainingSet, featureLayer,'MiniBatchSize', 32, 'OutputAs', 'columns');
% datacnn.training.features = double(datacnn.training.features)';
% % Get training labels from the trainingSet
% datacnn.training.labels = double(trainingSet.Labels);
% % Extract test features using the CNN
% datacnn.testing.features = activations(net, testSet, featureLayer, 'MiniBatchSize',32);
% datacnn.testing.features = double(datacnn.testing.features);
% % Get the known labels
% datacnn.testing.labels = double(testSet.Labels);

%==================== MERGE COSFIRE AND CNN FEATURES ===================
% datacnncosfire.training.features = [data.training.desc';datacnn.training.features'];
% datacnncosfire.testing.features = [data.testing.desc';datacnn.testing.features'];
% datacnncosfire.training.labels = data.training.labels;
% datacnncosfire.testing.labels = data.testing.labels;

% Training classification SVM models
% SVMModelCNN = fitcsvm(datacnn.training.features,datacnn.training.labels,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
% SVMModelCOSFIRE = fitcsvm(data.training.desc,data.training.labels,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
% SVMModelCOSFIREANDCNN = fitcsvm(datacnncosfire.training.features',datacnncosfire.training.labels,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');

% Evaluate classifier
%accuracy_cnn = evaluateclassifier(SVMModelCNN,datacnn');
%accuracy_cosfire = evaluateclassifier(SVMModelCOSFIRE,data);
%accuracy_cnncosfire = evaluateclassifier(SVMModelCOSFIREANDCNN,datacnncosfire);

% Plot Accuracy
accuracy_cnn = 0.8941;
accuracy_cnncosfire = 0.9237;
x = categorical({'CNN','COSFIRE','CNN&COSFIRE'});
y = [accuracy_cnn accuracy_cosfire accuracy_cnncosfire];
bar(x,y)
