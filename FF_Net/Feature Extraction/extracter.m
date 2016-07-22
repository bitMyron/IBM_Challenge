clc
clear

rootFolder = fullfile('C:\Users\Administrator\Documents\MATLAB', 'LabelRain');
% %info = imfinfo('Bergweide_2016-04-27 15_06_22.jpg');
imgSets = [];
dirinfo = dir();
dirinfo(~[dirinfo.isdir]) = []; 
tf = ismember( {dirinfo.name}, {'.', '..'});
dirinfo(tf) = [];  %remove current and parent directory.
for K = 1 : length(dirinfo)
  thisdir = dirinfo(K).name;
  % subdirinfo{K} = dir(fullfile(thisdir, '*.jpg'));
  imgSets = [imgSets, ...
             imageSet(fullfile(rootFolder, thisdir))];
end
count = 0;
for i = 1:numel(imgSets)   
    for j = 1:imgSets(i).Count
        %img = read(trainingSet(i),j);   
        imgInfo(j+count,:)= imfinfo(imgSets(1,i).ImageLocation{j});
    end
    count = count + imgSets(i).Count;
end
% camName = getfield(dirinfo(), 'name');
% subdirinfo = cell(length(dirinfo));
% for K = 1 : length(dirinfo)
%   thisdir = dirinfo(K).name;
%   subdirinfo{K} = dir(fullfile(thisdir, '*.jpg'));
% end
for K = 1 : length(imgInfo)
  thisfile = imgInfo(K).Filename;
  thistag = imgInfo(K).Make;
  des = ['C:\Users\Administrator\Documents\MATLAB\LabelRain\0000000000000000000\' thistag '\'];
  mkdir(des);
  % subdirinfo{K} = dir(fullfile(thisdir, '*.jpg'));
  copyfile(thisfile,des)
end
