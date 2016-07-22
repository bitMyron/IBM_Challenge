rootFolder = fullfile('C:\Users\Administrator\Documents\MATLAB\NeuralIBM', 'dataset');
imgSets = [ imageSet(fullfile(rootFolder, 'rain')), ...
            imageSet(fullfile(rootFolder, 'noRain'))  ];
%        , ...
%            imageSet(fullfile(rootFolder, 'laptop'))       
%{imgSets.Description }; % display all labels on one line
%[imgSets.Count]  ;       % show the corresponding count of images
minSetCount = min([imgSets.Count]);
imgSets = partition(imgSets, minSetCount, 'randomize');
[trainingSet, validationSet] = partition(imgSets, 0.3, 'randomize');

% feature extraction
descriptors = [];
numSets = numel(imgSets);
for categoryIndex=1:numSets
    imgSet = imgSets(categoryIndex);
    for i = 1:imgSet.Count
              img = read(imgSet,i);                     
              [tempDescriptors] = extractDescriptorsFromImage(img);
              descriptors = [descriptors tempDescriptors]; 
    end     
end

% dictionary of words
iter = 100;
codewords = 200;
vocabulary = learnCodebook(descriptors, codewords, iter);
clear descriptors % clean memory
disp('dictionary is ready')

% encode images
if_norm = 1;
[ BoWvec, label ] = encodingImage( trainingSet, vocabulary, if_norm ); % hard assignment
% numImages = sum([trainingSet.Count]);
% label = zeros(1, numImages);
% BoWvec = zeros(codewords, numImages);
% 
% count = 0;
% for i = 1:numel(trainingSet) 
% % notice the images of each class are sequentially computed by BoW algorithm   
%     for j = 1:trainingSet(i).Count
%         img = read(trainingSet(i),j);                     
%         [feats] = extractDescriptorsFromImage(img);
%         BoWvec(:,count+j)=computeBoV(vocabulary, feats, if_norm);
%     end
%     
%     if(strcmp(trainingSet(i).Description, 'airplanes'))
%         label(count+1:count+j) = 1;
%     elseif(strcmp(trainingSet(i).Description, 'ferry'))
%         label(count+1:count+j) = -1;
% %    elseif(strcmp(trainingSet(i).Description, 'laptop'))
% %        label(count+1:count+j) = 3;
%     end
% 
%     count = count + trainingSet(i).Count;
% end
disp('word encoding is done')

% training a SVM classifier
lambda = 0.01 ; % Regularization parameter
maxIter = 1000 ; % Maximum number of iterations
rowrank = randperm(size(BoWvec, 2)); % generate a random sequence of training
BoWvec = BoWvec(:, rowrank);
label = label(rowrank);
[w, b, info] = vl_svmtrain(BoWvec, label, lambda, 'MaxNumIterations', maxIter);
disp('SVM training is done')

% test on SVM classifier
[ BoWvec, label ] = encodingImage( validationSet, vocabulary, if_norm );
% numImages = sum([validationSet.Count]);
% label = zeros(1, numImages);
% BoWvec = zeros(codewords, numImages);
% 
% count = 0;
% for i = 1:numel(validationSet) 
% % notice the images of each class are sequentially computed by BoW algorithm   
%     for j = 1:validationSet(i).Count
%         img = read(validationSet(i),j);                     
%         [feats] = extractDescriptorsFromImage(img);
%         BoWvec(:,count+j)=computeBoV(vocabulary, feats, if_norm);
%     end
%     
%     if(strcmp(validationSet(i).Description, 'airplanes'))
%         label(count+1:count+j) = 1;
%     elseif(strcmp(validationSet(i).Description, 'ferry'))
%         label(count+1:count+j) = -1;
% %    elseif(strcmp(trainingSet(i).Description, 'laptop'))
% %        label(count+1:count+j) = 3;
%     end
% 
%     count = count + validationSet(i).Count;
% end
[~,~,~, scores] = vl_svmtrain(BoWvec, label, 0, 'model', w, 'bias', b, 'solver', 'none') ;
vl_roc(label, scores) ;
%scores = zeros(1, numImages);
%for i = 1: numImages
%    [~,~,~, score] = vl_svmtrain(BoWvec(:,i), label(i), 0, 'model', w, 'bias', b, 'solver', 'none') ;
%    scores(i) = score;
%end
disp('test is done')

 
