close all;
clear all;

img = imread("cameraman.tif");

if size(img, 3) == 3  % Check if the image has 3 channels (RGB)
    grayImg = rgb2gray(img);
else
    grayImg = img;  % If the image is already grayscale
end

smoothedImg = imgaussfilt(grayImg, 1);  % Apply slight blur to reduce noise

% Apply edge detection (Canny with adjusted thresholds)
edges = edge(smoothedImg, 'Canny', [0.01, 0.3]);

figure;
imshow(edges, []);
title('Edge Detection Output');

%%

% Fill in the edges selectively
mask = imfill(edges, 'holes'); 

se = strel('disk', 3);  % Structuring element for closing
closedEdges = imclose(mask, se);  % Perform closing to fill gaps in edges

% Filter out small regions
labeledMask = bwlabel(closedEdges);
regionProps = regionprops(labeledMask, 'Area');

% Define a minimum
minSize = 300;

% Filter out
filteredMask = ismember(labeledMask, find([regionProps.Area] > minSize));

% Smoothen transition
filteredMask = imdilate(filteredMask, strel('disk', 1)); 

% Convert the mask to double for operations
mask = double(filteredMask);

% Visualize the improved mask
figure;
imshow(mask, []);
title('Refined Mask with Region Filtering');

%%

% ´Gaussian blur ´
blurredImg = imgaussfilt(grayImg, 2);


grayImg = double(grayImg); 
blurredImg = double(blurredImg);

% Combine the foreground and blurred
portraitImg = mask .* grayImg + (1 - mask) .* blurredImg;

portraitImg = uint8(portraitImg);

figure;
subplot(1,2,1);
imshow(portraitImg);
title('Portrait Mode Effect');
% -----
subplot(1,2,2);
imshow(grayImg, []);
title('Original Image');
