function [imdstraining, imdstest] = getdataset(dirlist)
% Set path for each training and test set
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

