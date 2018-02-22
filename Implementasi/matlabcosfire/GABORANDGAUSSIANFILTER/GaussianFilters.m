% Load image
img = imread('stop1.jpg');

% Apply Gaussian Filters
filteredImg = imgaussfilt(img,2);
imshow(filteredImg)