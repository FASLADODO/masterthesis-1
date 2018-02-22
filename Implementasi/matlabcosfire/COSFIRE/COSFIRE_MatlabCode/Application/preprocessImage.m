
function im = preprocessImage(im)

    % If image is coloured then convert it to grayscale
    if size(im,3) > 1
        im = double(rgb2gray(im));
    end

    if max(im(:)) > 1
        im = im / 255;
    end
end