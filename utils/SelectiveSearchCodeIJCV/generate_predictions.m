% This demo shows how to use the software described in our IJCV paper: 
%   Selective Search for Object Recognition,
%   J.R.R. Uijlings, K.E.A. van de Sande, T. Gevers, A.W.M. Smeulders, IJCV 2013
%%
addpath('Dependencies');

fprintf('Demo of how to run the code for:\n');
fprintf('   J. Uijlings, K. van de Sande, T. Gevers, A. Smeulders\n');
fprintf('   Segmentation as Selective Search for Object Recognition\n');
fprintf('   IJCV 2013\n\n');

% Compile anisotropic gaussian filter
if(~exist('anigauss'))
    fprintf('Compiling the anisotropic gauss filtering of:\n');
    fprintf('   J. Geusebroek, A. Smeulders, and J. van de Weijer\n');
    fprintf('   Fast anisotropic gauss filtering\n');
    fprintf('   IEEE Transactions on Image Processing, 2003\n');
    fprintf('Source code/Project page:\n');
    fprintf('   http://staff.science.uva.nl/~mark/downloads.html#anigauss\n\n');
    mex Dependencies/anigaussm/anigauss_mex.c Dependencies/anigaussm/anigauss.c -output anigauss
end

if(~exist('mexCountWordsIndex'))
    mex Dependencies/mexCountWordsIndex.cpp
end

% Compile the code of Felzenszwalb and Huttenlocher, IJCV 2004.
if(~exist('mexFelzenSegmentIndex'))
    fprintf('Compiling the segmentation algorithm of:\n');
    fprintf('   P. Felzenszwalb and D. Huttenlocher\n');
    fprintf('   Efficient Graph-Based Image Segmentation\n');
    fprintf('   International Journal of Computer Vision, 2004\n');
    fprintf('Source code/Project page:\n');
    fprintf('   http://www.cs.brown.edu/~pff/segment/\n');
    fprintf('Note: A small Matlab wrapper was made. See demo.m for usage\n\n');
%     fprintf('   
    mex Dependencies/FelzenSegment/mexFelzenSegmentIndex.cpp -output mexFelzenSegmentIndex;
end

%%
% Parameters. Note that this controls the number of hierarchical
% segmentations which are combined.
colorTypes = {'Rgb', 'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};

% Here you specify which similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
ks = [50 100 150 300]; % controls size of segments of initial segmentation. 
sigma = 0.8;

% After segmentation, filter out boxes which have a width/height smaller
% than minBoxWidth (default = 20 pixels).
minBoxWidth = 20;

% Comment the following three lines for the 'quality' version
colorTypes = colorTypes(1:2); % 'Fast' uses HSV and Lab
simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies
ks = ks(1:2);

% Test the boxes
fprintf('After box extraction, boxes smaller than %d pixels will be removed\n', minBoxWidth);
totalTime = 0;

imgFolderPath = '/home/zsy/data/flickr30k/flickr30k-images';

load('../dataSplits.mat', 'testfns');
testIms = cell(1,1000);
for n = 1:1000
    testIms{n} = strcat(testfns{n},'.jpg');
end
for i=1:length(testIms)
    fprintf('%d \n', i);
    
    % VOCopts.img
    im = imread(fullfile(imgFolderPath,testIms{i}));
    idx = 1;
    for j=1:length(ks)
        k = ks(j); % Segmentation threshold k
        minSize = k; % We set minSize = k
        for n = 1:length(colorTypes)
            colorType = colorTypes{n};
            tic;
            [boxesT{idx} blobIndIm blobBoxes hierarchy priorityT{idx}] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
            totalTime = totalTime + toc;
            idx = idx + 1;
        end
    end
    boxes{i} = cat(1, boxesT{:}); % Concatenate boxes from all hierarchies
    priority = cat(1, priorityT{:}); % Concatenate priorities
    
    % Do pseudo random sorting as in paper
    priority = priority .* rand(size(priority));
    [priority sortIds] = sort(priority, 'ascend');
    boxes{i} = boxes{i}(sortIds,:);


end
fprintf('\n');

%%
tic
for i=1:length(boxes)
    boxes{i} = FilterBoxesWidth(boxes{i}, minBoxWidth);
    boxes{i} = BoxRemoveDuplicates(boxes{i});
    boxes{i} = boxes{i}(1:100,:);
end
totalTime = totalTime + toc;

fprintf('Time per image: %.2f\nNow evaluating the boxes on Pascal 2007...\n', totalTime ./ length(testIms));
save('SelectiveSearch_Flickr30k_test_k300.mat','boxes','testIms');
%% Show a couple of good boxes in the image
% fprintf('Showing examples of good boxes\n');
% goodBoxes = boxes{5}([1 2 3 4 5 6], :);
% im = imread(fullfile(imgFolderPath,testIms{5}));
% figure; 
% for j=1:6
%     subplot(2, 3, j);
%     boxIm = im(goodBoxes(j,1):goodBoxes(j,3), goodBoxes(j,2):goodBoxes(j,4), :);
%     imshow(boxIm);
% end
