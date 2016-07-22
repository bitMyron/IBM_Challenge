rootFolder = fullfile('C:\Users\Administrator\Documents\MATLAB\NeuralIBM', 'dataset');
imgSets = [ imageSet(fullfile(rootFolder, 'rain')), ...
            imageSet(fullfile(rootFolder, 'noRain'))  ];
%        , ...
%            imageSet(fullfile(rootFolder, 'laptop'))       
%{imgSets.Description }; % display all labels on one line
%[imgSets.Count]  ;       % show the corresponding count of images
% minSetCount = min([imgSets.Count]);
% imgSets = partition(imgSets, minSetCount, 'randomize');
trainingSet = imgSets;


% determine HOG parameter and hog feature size
imgSize = [220 300]; % image resize
cellSize = [4 4]; % the HOG cellsize should be varied with repeated classifier training by visualization
[hog_cellsize, vis] = extractHOGFeatures(imresize(read(trainingSet(1),1),imgSize),'CellSize',cellSize);
hogFeatureSize = length(hog_cellsize);

% hog feature extraction
[ trainingFeatures, label ] = computeHOG( trainingSet, hogFeatureSize, cellSize, imgSize );
% numImages = sum([trainingSet.Count]);
% Label = zeros(1, numImages);
% trainingFeatures  = zeros(numImages,hogFeatureSize,'single');
% 
% count = 0;
% for i = 1:numel(trainingSet)   
%     for j = 1:trainingSet(i).Count
%         img = read(trainingSet(i),j);   
%         img = imresize(img,imgSize);
%         trainingFeatures(count+j,:)=extractHOGFeatures(img,'CellSize',cellSize);
%     end
%     
%     if(strcmp(trainingSet(i).Description, 'ferry'))
%         Label(count+1:count+j) = 1;
%     elseif(strcmp(trainingSet(i).Description, 'laptop'))
%         Label(count+1:count+j) = -1;
% %    elseif(strcmp(trainingSet(i).Description, 'laptop'))
% %        label(count+1:count+j) = 3;
%     end
% 
%     count = count + trainingSet(i).Count;
% end

% SVM training process
% svm = svmtrain(trainingFeatures, label);
% 
% % SVM testing process
% [ validationFeatures, Label ] = computeHOG( validationSet, hogFeatureSize, cellSize, imgSize );
% % numImages = sum([validationSet.Count]);
% % label = zeros(1, numImages);
% % validationFeatures  = zeros(numImages,hogFeatureSize,'single');
% % 
% % count = 0;
% % for i = 1:numel(validationSet)   
% %     for j = 1:validationSet(i).Count
% %         img = read(validationSet(i),j);   
% %         img = imresize(img,imgSize);
% %         validationFeatures(count+j,:)=extractHOGFeatures(img,'CellSize',cellSize);
% %     end
% %     
% %     if(strcmp(validationSet(i).Description, 'ferry'))
% %         label(count+1:count+j) = 1;
% %     elseif(strcmp(validationSet(i).Description, 'laptop'))
% %         label(count+1:count+j) = -1;
% % %    elseif(strcmp(trainingSet(i).Description, 'laptop'))
% % %        label(count+1:count+j) = 3;
% %     end
% % 
% %     count = count + validationSet(i).Count;
% % end
% 
% scores = svmclassify(svm, validationFeatures);

