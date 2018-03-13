%Function to resize image as required for the CNN.
function Iout = readAndPreprocessImage(filename,doFaceDetection,size)
    % Read Image    
    img = imread(filename);
    % Apply ViolaJones Face detector
    
    if (doFaceDetection == 1)
        faceDetector = vision.CascadeObjectDetector;
        BB = step(faceDetector,img); % Detect faces

        if isempty(BB)
           face = I;
        else
    %         IFaces = insertObjectAnnotation(img, 'rectangle', BB, 'Face');   
    %         figure, imshow(IFaces), title('Detected faces');
    %    
           if length(BB) > 1
                [~,idx] = max(BB(:,3));
                face = img(BB(idx,2):BB(idx,2)+BB(idx,4)-1,BB(idx,1):BB(idx,1)+BB(idx,3)-1);
           else
                face = img(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
           end
        end        
    end
    
            
    % Some images may be grayscale. Replicate the image 3 times to
    % create an RGB image.
    if ismatrix(face)
        img = cat(size(3),face,face,face);
    end

    % Resize the image as required for the CNN.
     Iout = imresize(img,[size(1) size(2)]);
     
    % Note that the aspect ratio is not preserved. In Caltech 101, the
    % object of interest is centered in the image and occupies a
    % majority of the image scene. Therefore, preserving the aspect
    % ratio is not critical. However, for other data sets, it may prove
    % beneficial to preserve the aspect ratio of the original image
    % when resizing.
end