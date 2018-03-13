%Function to resize image as required for the CNN.
function Iout = readAndPreprocessImage(filename,size)
    % Read Image    
    I = imread(filename);
    
    % Adjust size of the image as required for the CNN.
    Iout = I(1:size(1),1:size(2),1:size(3))  ;
    % Some images may be grayscale. Replicate the image 3 times to
    % create an RGB image.
%     if ismatrix(I)
%         I = cat(3,I,I,I);
%     end

    % Resize the image as required for the CNN.
%     Iout = imresize(I, [227 227]);

    % Note that the aspect ratio is not preserved. In Caltech 101, the
    % object of interest is centered in the image and occupies a
    % majority of the image scene. Therefore, preserving the aspect
    % ratio is not critical. However, for other data sets, it may prove
    % beneficial to preserve the aspect ratio of the original image
    % when resizing.
end