function [status,net]= ModelExist(dataFolder,cnntype)
    if ~exist(strcat(dataFolder,'results/cnn/',cnntype,'/netTransfer.mat'))
        status = 0;
        net = [];
    else
        status = 1;
        net = load(strcat(dataFolder,'results/cnn/',cnntype,'/netTransfer.mat')); 
    end

