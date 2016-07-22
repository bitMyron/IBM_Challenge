clear;
clc;
% A deblocking example by iterative steering kernel regression
%
% written by hiro, June 17 2007

% load image
img = double(rgb2gray(imread('rain.jpg')));
[N,M] = size(img);


% load the degraded image
y = img;

% pilot estimation by second order classic kernel regression
h = 0.5;    % the global smoothing parameter
r = 1;      % the upscaling factor
ksize =3;  % the kernel size
[zc, zx1c, zx2c] = ckr2_regular(y, h, r, ksize);

% iteartive steering kernel regression (second order)
IT = 3;     % the total number of iterations
wsize = 9;  % the size of the local orientation analysis window
lambda = 2;  % the regularization for the elongation parameter
alpha = 0.6; % the structure sensitive parameter
h = 2.4;     % the global smoothing parameter
ksize = 11;  % the kernel size
z = zeros(N, M, IT+1);
zx1 = zeros(N, M, IT+1);
zx2 = zeros(N, M, IT+1);
rmse = zeros(IT+1, 1);
z(:,:,1) = y;
zx1(:,:,1) = zx1c;
zx2(:,:,1) = zx2c;
error = img - y;
rmse(1) = sqrt(mymse(error(:)));

for i = 2 : IT+1
    % compute steering matrix
    C = steering(zx1(:,:,i-1), zx2(:,:,i-1), ones(size(img)), wsize, lambda, alpha);
    % steering kernel regression
    [zs, zx1s, zx2s] = skr2_regular(z(:,:,i-1), h, C, r, ksize);
    z(:,:,i) = zs;
    zx1(:,:,i) = zx1s;
    zx2(:,:,i) = zx2s;
    % root mean square error
    error = img - zs;
    rmse(i) = sqrt(mymse(error(:)));
    rmse(i);
    %figure(99); imagesc(zs); colormap(gray); axis image; pause(1);
end

% display images
figure; imagesc(y); colormap(gray); axis image;
title(['The JPEG image, STD=25, RMSE=', num2str(rmse(1))]);
figure; imagesc(zx2s); colormap(gray); axis image;
title(['ISKR, 3 iterations, RMSE=', num2str(rmse(4))]);