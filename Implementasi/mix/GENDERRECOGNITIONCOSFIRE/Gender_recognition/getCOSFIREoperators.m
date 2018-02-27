% Configure COSFIRE filters using male and female training face images
% The filters are saved to the outdir folder so that next time we run the
% experiment we do not need to configure a new set of filters.
function operatorlist = getCOSFIREoperators(outdir,dataset,noperators)
if ~exist([outdir,filesep,'operatorlist.mat'])
    fprintf('Configuring COSFIRE filters from Male face images\n');
    operatormalelist = configureCOSFIREfilters(dataset.training.males,noperators,1);
    operatorfemalelist = configureCOSFIREfilters(dataset.training.females,noperators,1);
    operatorlist = horzcat(operatormalelist,operatorfemalelist);
    save([outdir,filesep,'operatorlist.mat'],'operatorlist');
else
    load([outdir,filesep,'operatorlist.mat']);
end