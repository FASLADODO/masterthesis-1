% Read all face images of given data set and organize them in a struct
% The struct is stored in the dataFolder, so that next we run the
% experiment, we would not need to read the face images again.
function dataset = getDataset(dataFolder,dirlist,dofacedetection,resizeWidth)

if ~exist(strcat(dataFolder,filesep,'dataset.mat'))
    dataset.training.males   = getFaces(dirlist.trainingmaledir,resizeWidth,dofacedetection);
    dataset.training.females = getFaces(dirlist.trainingfemaledir,resizeWidth,dofacedetection);
    dataset.testing.males    = getFaces(dirlist.testingmaledir,resizeWidth,dofacedetection);
    dataset.testing.females  = getFaces(dirlist.testingfemaledir,resizeWidth,dofacedetection);
    save(strcat(dataFolder,filesep,'dataset.mat'),'dataset');
else
    load(strcat(dataFolder,filesep,'dataset.mat'));    
end

