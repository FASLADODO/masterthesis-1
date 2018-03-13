function [status,accuracy]=GetCNNAccuracy(dataFolder,cnntype)
   if ~exist(strcat(dataFolder,'results/cnn/',cnntype,'/accuracy.mat'))        
        status = 0;
        accuracy = 0;
   else
        status = 1;
        accuracy = load(strcat(dataFolder,'results/cnn/',cnntype,'/accuracy.mat'));
        accuracy = accuracy.accuracy;    
   end