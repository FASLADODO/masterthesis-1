function [status,datacnn]=GetCNNFeatures(dataFolder,cnntype)
   if ~exist(strcat(dataFolder,'results/cnn/',cnntype,'/datacnn.mat'))        
        status = 0;
        datacnn = [];
   else
        status = 1;
        datacnn = load(strcat(dataFolder,'results/cnn/',cnntype,'/datacnn.mat')); 
	datacnn = datacnn.datacnn;   
   end