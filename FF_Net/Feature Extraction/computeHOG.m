function [ HOGvec, Label ] = computeHOG( ImageSets, featureSize, cellSize, imgSize )
% HOG Feature extraction:
%
% Inputs->
%
% ImageSets : imageSet class
% featureSize: HOG feature size depending on image size and cell size
% cellSize   : HOG cell size in form of [x x]
% imgSize   : image size for HOG feature computation
%
% ***************************  Needs INRIA's yael library ******************
% Output->
%
% HOGvec : output HOG feature vector (row Vector)
% Label  : output label vector

numImages = sum([ImageSets.Count]);
Label = zeros(1, numImages);
HOGvec  = zeros(numImages,featureSize,'single');

count = 0;
for i = 1:numel(ImageSets)   
    for j = 1:ImageSets(i).Count
        img = read(ImageSets(i),j);   
        img = imresize(img,imgSize);
        HOGvec(count+j,:)=extractHOGFeatures(img,'CellSize',cellSize);
    end
    
    if(strcmp(ImageSets(i).Description, 'rain'))
        Label(count+1:count+j) = 1;
    elseif(strcmp(ImageSets(i).Description, 'noRain'))
        Label(count+1:count+j) = -1;
%    elseif(strcmp(trainingSet(i).Description, 'laptop'))
%        label(count+1:count+j) = 3;
    end

    count = count + ImageSets(i).Count;
end

end

