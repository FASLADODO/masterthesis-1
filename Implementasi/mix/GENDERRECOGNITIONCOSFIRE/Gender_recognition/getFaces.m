% Read face images from indir directory
% If dofacedetection = 1 then apply ViolaJones algorithm to localize the
% face, otherwise assume that the image contains only the face
% If resizewidth is NaN then we keep the original size of the face,
% otherwise we resize the face to [resizewidth, resizewidth]
function facelist = getFaces(indir,resizewidth,dofacedetection)

% ViolaJones Face detector
faceDetector = vision.CascadeObjectDetector;
facelist = [];
ind = 1;
d = dir(strcat(indir,filesep,'*.jpg'));
for i = 1:numel(d)
    fprintf('reading image %d of %d\n',i,numel(d));
    img = imread(strcat(indir,filesep,d(i).name)); %Read input image
    
    if ndims(img) == 3
        img = rgb2gray(img); % convert to gray
    end
    
    if dofacedetection == 1
        BB = step(faceDetector,img); % Detect faces

        if isempty(BB)
            continue;
        elseif size(BB,1) > 1
            [~,idx] = max(BB(:,3));
            face = img(BB(idx,2):BB(idx,2)+BB(idx,4)-1,BB(idx,1):BB(idx,1)+BB(idx,3)-1);
        else
            face = img(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
        end
    else
        face = img;
    end
    
    if ~isnan(resizewidth)
        facelist(:,:,ind) = double(imresize(face,[resizewidth,resizewidth]));
    else
        facelist(:,:,ind) = face;
    end
    ind = ind + 1;
end