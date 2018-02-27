function [imdstraining, imdstest] = getdataset(dirlist)

% datasetdir = fullfile(pwd,'GENDER-FERET');
% maletrainingfolder = fullfile(fullfile(datasetdir, 'male'),'training_set');
% femaletrainingfolder = fullfile(fullfile(datasetdir, 'female'),'training_set');
% maletestfolder = fullfile(fullfile(datasetdir, 'male'),'test_set');
% femaletestfolder = fullfile(fullfile(datasetdir, 'female'),'test_set');

imdsmaletraining = imageDatastore(dirlist.trainingmaledir);
imdsmaletraining.Labels = ones(length(imdsmaletraining.Files),1);

imdsfemaletraining = imageDatastore(dirlist.trainingfemaledir);
imdsfemaletraining.Labels = ones(length(imdsmaletraining.Files),1)*2;

imdsmaletest = imageDatastore(dirlist.testingmaledir);
imdsmaletest.Labels =  ones(length(imdsmaletest.Files),1);

imdsfemaletest = imageDatastore(dirlist.testingfemaledir);
imdsfemaletest.Labels = ones(length(imdsfemaletest.Files),1)*2;

% Merge imdsmaletraining and imdsfemaletraining
imdstraining = imageDatastore(cat(1,imdsmaletraining.Files,imdsfemaletraining.Files));
imdstraining.Labels = cat(1,imdsmaletraining.Labels,imdsfemaletraining.Labels);

% Merge imdsmaletest and imdsfemaletest
imdstest = imageDatastore(cat(1,imdsmaletest.Files,imdsfemaletest.Files));
imdstest.Labels = cat(1,imdsmaletest.Labels,imdsfemaletest.Labels);

% Convert labels to categorical
imdstraining.Labels = categorical(imdstraining.Labels);
imdstest.Labels = categorical(imdstest.Labels);

% Set each ImageDatastore ReadFcn
imdstraining.ReadFcn = @(filename)readAndPreprocessImage(filename);
imdstest.ReadFcn = @(filename)readAndPreprocessImage(filename);
end

