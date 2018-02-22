function TEST_COSFIRE_Application(param,img_prototype)
% First, compile the c-function called maxblurring_written_in_c.c
d = dir('../COSFIRE/maxblurring.*');
if isempty(d)
    mex ../COSFIRE/written_in_c_maxblurring.c -outdir ../COSFIRE -output maxblurring;
end

% Update the Matlab path with the following
path('../COSFIRE/',path);
path('../Gabor/',path);


params = Parameters(param);
X = imread(img_prototype);
Y = imresize(X, 0.1);
prototype = preprocessImage(Y);
x = 70; y = 65;

% Configure a COSFIRE operator
operator = configureCOSFIRE(prototype,round([y,x]),params);  

% Show the structure of the COSFIRE operator
viewCOSFIREstructure(operator);       

% testingImage = preprocessImage(prototype);
% output = applyCOSFIRE(testingImage,operator);
% 
% figure;maximaPoints(testingImage,{output},8,1);     

figure;
for i = 1:4
    filename = strcat('test/','stop',int2str(i),'.jpg');
    testingImage = imread(filename);
    Y = imresize(testingImage, 0.2);
    testingImage = preprocessImage(Y);

    % Apply the COSFIRE to the input image
    output = applyCOSFIRE(testingImage,operator);

    % Show the detected points based on the maxima responses of output.
    figure; maximaPoints(testingImage,{output},8,1);    
% end      

end