% Variables
noperators = 180;
extractFeatures = 0;
doFaceDetection = 0;
cnnType = 'vggface2'
% Datasource
dataFolder  = 'GENDER-FERET';

% Configure the image paths. If you want to organize your folder in a
% different way, please modify these lines.
dirlist.maledir             = strcat(dataFolder,filesep,'male');
dirlist.femaledir           = strcat(dataFolder,filesep,'female');
dirlist.trainingmaledir     = strcat(dirlist.maledir,filesep,'training_set');
dirlist.trainingfemaledir   = strcat(dirlist.femaledir,filesep,'training_set');
dirlist.testingmaledir      = strcat(dirlist.maledir,filesep,'test_set');
dirlist.testingfemaledir    = strcat(dirlist.femaledir,filesep,'test_set');

% Create folder results/cnn/cnnmodel if it does not exist
if ~exist(strcat('./results/',cnnType))
    mkdir(strcat('./results/',cnnType));
end

% Load pre-trained CNN
net = vgg16;
   
% Prepare Training and Test Image Sets
sz = net.Layers(1).InputSize;
[trainingSet, testSet] = getCNNDataset(dirlist,doFaceDetection,sz);

% Load vggface training features
% vggfacefeatures = load(strcat('./results/vggface/vggfacefeatures.mat'));
% datacnn.training.features = vggfacefeatures.trainingfeatures;
% datacnn.training.labels = trainingSet.Labels;
% datacnn.testing.features = vggfacefeatures.testfeatures;
% datacnn.testing.labels = testSet.Labels;

% Normalization
fun = @(x) normr(x);
datacnn.training.normalizedfeatures = blkproc(datacnn.training.features,[size(datacnn.training.features,1),noperators],fun);
datacnn.testing.normalizedfeatures = blkproc(datacnn.testing.features,[size(datacnn.testing.features,1),noperators],fun);

% Save extracted features
if ~exist(strcat('./results/',cnnType,'/datacnn.mat'))
    save(strcat('./results/',cnnType,'/datacnn.mat'),'datacnn');
end

% Fit Image Classifier CNN
classifierCNN = fitcecoc(datacnn.training.normalizedfeatures,datacnn.training.labels);
predictedCNNLabels = predict(classifierCNN,datacnn.testing.normalizedfeatures);
accuracyCNN = mean(predictedCNNLabels == datacnn.testing.labels);
fprintf('\nOriginal CNN accuracy %d',accuracyCNN);

% Load COSFIRE data
data = load(strcat('./results/cosfire/data.mat'),'data');
data = data.data;

% Fit Image Classifier COSFIRE
classifierCOSFIRE = fitcecoc(data.training.desc,datacnn.training.labels);
predictedCOSFIRELabels = predict(classifierCOSFIRE,data.testing.desc);
accuracyCOSFIRE = mean(predictedCOSFIRELabels == datacnn.testing.labels);
fprintf('\nCOSFIRE accuracy %d',accuracyCOSFIRE);

% Merge CNN and COSFIRE features
datacnncosfire.training.features = [datacnn.training.features';data.training.desc'];
datacnncosfire.training.features = datacnncosfire.training.features';
datacnncosfire.training.normalizedfeatures = [datacnn.training.normalizedfeatures';data.training.desc'];
datacnncosfire.training.normalizedfeatures = datacnncosfire.training.normalizedfeatures';
datacnncosfire.training.labels = datacnn.training.labels;

datacnncosfire.testing.features = [datacnn.testing.features';data.testing.desc'];
datacnncosfire.testing.features = datacnncosfire.testing.features';
datacnncosfire.testing.normalizedfeatures = [datacnn.testing.normalizedfeatures';data.testing.desc'];
datacnncosfire.testing.normalizedfeatures = datacnncosfire.testing.normalizedfeatures';
datacnncosfire.testing.labels = datacnn.testing.labels;

% Fit Image Classifier CNNCOSFIRE
% classifierCNNCOSFIRE = fitcecoc(datacnncosfire.training.features,datacnncosfire.training.labels);
% predictedCNNCSOFIRELabels = predict(classifierCNNCOSFIRE,datacnncosfire.testing.features);
% accuracyCNNCOSFIRE = mean(predictedCNNCSOFIRELabels == datacnncosfire.testing.labels);
% fprintf('\nCNNCSOFIRE accuracy %d\n',accuracyCNNCOSFIRE);

classifierNormalizedCNNCOSFIRE = fitcecoc(datacnncosfire.training.normalizedfeatures,datacnncosfire.training.labels);
predictedNormalizedCNNCSOFIRELabels = predict(classifierNormalizedCNNCOSFIRE,datacnncosfire.testing.normalizedfeatures);
accuracyNormalizedCNNCOSFIRE = mean(predictedNormalizedCNNCSOFIRELabels == datacnncosfire.testing.labels);
fprintf('\nNormalized CNNCSOFIRE accuracy %d\n',accuracyNormalizedCNNCOSFIRE);


