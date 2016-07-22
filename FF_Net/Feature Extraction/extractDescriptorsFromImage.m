function [ descriptors ] = extractDescriptorsFromImage( image )
%   Detailed explanation goes here
    if ismatrix(image) % convert color images to grayscale
          grayImage = single(image);
       else
          grayImage = single(rgb2gray(image));
    end
    % peak_thresh and edge_thresha better to be chosen from cross
    % validation
    [points,descriptors] = vl_sift(grayImage) ;
    
end

