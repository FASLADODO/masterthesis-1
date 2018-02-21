function COSFIRE_Application(example)
% Keypoint detection by a COSFIRE filter
%
% VERSION 01/07/2012
% CREATED BY: George Azzopardi and Nicolai Petkov, University of Groningen,
%             Johann Bernoulli Institute for Mathematics and Computer Science, Intelligent Systems
%
% If you use this script please cite the following paper:
%   George Azzopardi and Nicolai Petkov, "Trainable COSFIRE filters for
%   keypoint detection and pattern recognition", IEEE Transactions on Pattern 
%   Analysis and Machine Intelligence, vol. 35(2), pp. 490-503, 2013.
% 
% A COSFIRE filter is automatically configured by a prototype pattern and
% can then be used to detect the same and similar patterns in given images. The configuration 
% comprises selecting given channels of a bank of orientation-selective filters 
% and determining certain blur and shift parameters. While here we use Gabor
% filters, they are not intrinsic to the method and any
% other orientation-selective filters can be used. The area of support of the resulting 
% COSFIRE filter is adaptive as it is composed of the support of a number of 
% orientation-selective filters whose geometrical arrangement around a point of 
% interest is learned from a single prototype pattern. A COSFIRE filter response 
% is computed as the weighted geometric mean of the blurred and shifted responses 
% of the selected Gabor filters. 
% 
% Example 1: COSFIRE_Application(1); % Runs an experiment to detect
% synthetic bifurcation using rotation-, scale- and reflection-invariance
%
% Example 2: COSFIRE_Application(2); % Runs an experiment to detect a
% traffic sign of pedestrian crossing in 16 images of complex scenes.
%
% You may modify the code to process your own images in different
% applications

% First, compile the c-function called maxblurring_written_in_c.c
d = dir('../COSFIRE/maxblurring.*');
if isempty(d)
    mex ../COSFIRE/written_in_c_maxblurring.c -outdir ../COSFIRE -output maxblurring;
end

% Update the Matlab path with the following
path('../COSFIRE/',path);
path('../Gabor/',path);

switch (example)
    case 1
        % An example to detect synthetic bifurcations
        params = Parameters(example);
        
        prototype = imread('pattern.bmp');
        prototype = preprocessImage(prototype);       
        x = 132; y = 116;

        % Configure a COSFIRE operator
        operator = configureCOSFIRE(prototype,round([y,x]),params);    

        % Show the structure of the COSFIRE operator
        viewCOSFIREstructure(operator);       
        
        testingImage = imread('pattern.bmp');
        output = applyCOSFIRE(testingImage,operator);
        
        figure;maximaPoints(testingImage,{output},8,1);            
    case 2
        % An example to detect a traffic sign of interest in complex
        % scenes
        params = Parameters(example);
        
        prototype = imread('Pedestrian.png');
        prototype = preprocessImage(prototype);
        x = 23; y = 23;

        % Configure a COSFIRE operator
        operator = configureCOSFIRE(prototype,round([y,x]),params);    

        % Show the structure of the COSFIRE operator
        viewCOSFIREstructure(operator);       

        figure;
        d = dir('pedestrian_*');
        for i = 1:numel(d)
            testingImage = imread(d(i).name);
            testingImage = preprocessImage(testingImage);

            % Apply the COSFIRE to the input image
            output = applyCOSFIRE(testingImage,operator);

            % Show the detected points based on the maxima responses of output.
            figure; maximaPoints(testingImage,{output},8,1);    
        end            
    otherwise
        error('Invalid example number');
end

function im = preprocessImage(im)

% If image is coloured then convert it to grayscale
if size(im,3) > 1
    im = double(rgb2gray(im));
end

if max(im(:)) > 1
    im = im / 255;
end
