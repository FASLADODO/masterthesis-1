% Load image
I = imread('stop1.jpg');
% Convert image to grayscale.
I = rgb2gray(I);
% Apply Gabors Filters
wavelength = 4;
orientation = 90;
[mag,phase] = imgaborfilt(I,wavelength,orientation);

%Show Images
figure
subplot(1,3,1);
imshow(I);
title('Original Image');
subplot(1,3,2);
imshow(mag,[])
title('Gabor magnitude');
subplot(1,3,3);
imshow(phase,[]);
title('Gabor phase');