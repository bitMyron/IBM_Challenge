function [ BoWvec, Label ] = encodingImage( ImageSets, Vocabulary, Norm )
% SIFT Feature extraction:          via hard voting (i.e, assignment to the nearest(K=1) codeword)
%
% Inputs->
%
% ImageSets : imageSet class
% Vocubulary: visual words or centroids or cluster centers (each column is a word)
% Norm      : Normalizatin index with the number of features(words) in the bag
%
%             0-no normalization
%             1-with noramlization (making it independent of number of features in the image)
%
% ***************************  Needs INRIA's yael library ******************
% Output->
%
% BoWvec : output BoW vector (column Vector)
% Label  : output corresponding label vector 

numImages = sum([ImageSets.Count]);
Label = zeros(1, numImages);
BoWvec = zeros(size(Vocabulary,2), numImages);

count = 0;
for i = 1:numel(ImageSets) 
% notice the images of each class are sequentially computed by BoW algorithm   
    for j = 1:ImageSets(i).Count
        img = read(ImageSets(i),j);                     
        [feats] = extractDescriptorsFromImage(img);
        BoWvec(:,count+j)=computeBoV(Vocabulary, feats, Norm);
    end
    
    if(strcmp(ImageSets(i).Description, 'airplanes'))
        Label(count+1:count+j) = 1;
    elseif(strcmp(ImageSets(i).Description, 'ferry'))
        Label(count+1:count+j) = -1;
%    elseif(strcmp(trainingSet(i).Description, 'laptop'))
%        label(count+1:count+j) = 3;
    end

    count = count + ImageSets(i).Count;
end

end

