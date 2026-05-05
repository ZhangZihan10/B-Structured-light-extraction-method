clear;
clc;
close all;

%% =========================================================
%  1. 读取图像
% =========================================================
imgName = 'image01.jpg';
black_image = imread(imgName);

if size(black_image, 3) == 1
    black_image = cat(3, black_image, black_image, black_image);
end

img = im2double(black_image);
[imgH, imgW, ~] = size(img);

R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);

hsvImage = rgb2hsv(img);
H = hsvImage(:,:,1);
S = hsvImage(:,:,2);
V = hsvImage(:,:,3);

%% =========================================================
%  2. 第一块木块 ROI
%  如果框偏了，只调这四个参数
% =========================================================
x1 = round(0.375 * imgW);
x2 = round(0.455 * imgW);

y1 = round(0.200 * imgH);
y2 = round(0.315 * imgH);

roiMask = false(imgH, imgW);
roiMask(y1:y2, x1:x2) = true;

figure;
imshow(black_image);
hold on;
rectangle('Position', [x1, y1, x2-x1, y2-y1], ...
          'EdgeColor', 'g', 'LineWidth', 2);
title('第一块木块 ROI');
hold off;

%% =========================================================
%  3. 在第一块木块中，只取激光所在的高度区域
%  这一版已经往上移了
% =========================================================
laserY1 = round(y1 + 0.25 * (y2 - y1));
laserY2 = round(y1 + 0.60 * (y2 - y1));

laserMask = false(imgH, imgW);
laserMask(laserY1:laserY2, x1:x2) = true;

figure;
imshow(black_image);
hold on;
rectangle('Position', [x1, laserY1, x2-x1, laserY2-laserY1], ...
          'EdgeColor', 'y', 'LineWidth', 2);
title('第一块木块激光搜索区域');
hold off;

%% =========================================================
%  4. 红色激光增强
% =========================================================

% HSV 红色范围
redHSV = ((H >= 0.000 & H <= 0.090) | ...
          (H >= 0.900 & H <= 1.000)) & ...
          (S > 0.10) & ...
          (V > 0.20);

% RGB 红色优势
redRGB = (R > 0.18) & ...
         (R > 1.01 * G) & ...
         (R > 1.01 * B);

% 红色得分
redScore = R - max(G, B);
redScore(redScore < 0) = 0;

% 只保留搜索区域
redScore(~laserMask) = 0;

% 平滑一下，让线更连续
redScoreSmooth = imgaussfilt(redScore, 0.8);

%% =========================================================
%  5. 生成候选区域
% =========================================================
candidateMask = laserMask & redHSV & redRGB & (redScoreSmooth > 0.012);

candidateMask = bwareaopen(candidateMask, 1);
candidateMask = imclose(candidateMask, strel('line', 5, 0));

figure;
imshow(imdilate(candidateMask, strel('disk', 2)));
title('第一块红色激光候选区域');

%% =========================================================
%  6. 用行投影找激光中心行
% =========================================================
scoreOnlyROI = redScoreSmooth;
scoreOnlyROI(~candidateMask) = 0;

rowProfile = sum(scoreOnlyROI(:, x1:x2), 2);
rowProfileSmooth = movmean(rowProfile, 5);

roiRows = laserY1:laserY2;
[~, maxIdx] = max(rowProfileSmooth(roiRows));
bestRow = roiRows(maxIdx);

% 手动微调：
% 如果线偏下，改成负数，例如 -1、-2
% 如果线偏上，改成正数，例如 1、2
manualYOffset = 3;
bestRow = bestRow + manualYOffset;

disp(['第一块激光中心行 y = ', num2str(bestRow)]);

%% =========================================================
%  7. 在 bestRow 附近提取横向连续激光区域
% =========================================================
searchHalfHeight = 3;

yy1 = max(bestRow - searchHalfHeight, laserY1);
yy2 = min(bestRow + searchHalfHeight, laserY2);

bandScore = max(redScoreSmooth(yy1:yy2, x1:x2), [], 1);

validVals = bandScore(bandScore > 0);
if isempty(validVals)
    error('第一块没有检测到有效激光，请检查 ROI 或降低阈值。');
end

colThreshold = mean(validVals) + 0.15 * std(validVals);
colMask = bandScore > colThreshold;

% 去掉太短的小噪声
colMask = bwareaopen(colMask, 3);

cc = bwconncomp(colMask);

if isempty(cc.PixelIdxList)
    error('没有找到连续激光线段，请降低阈值。');
end

% 选择得分最高的一段
bestScore = -inf;
bestId = 1;

for i = 1:length(cc.PixelIdxList)
    idx = cc.PixelIdxList{i};
    thisScore = sum(bandScore(idx)) + 2 * length(idx);

    if thisScore > bestScore
        bestScore = thisScore;
        bestId = i;
    end
end

colsLocal = cc.PixelIdxList{bestId};
colsGlobal = x1 + colsLocal - 1;

xStart = min(colsGlobal);
xEnd   = max(colsGlobal);

% 左右稍微扩展一点，让覆盖更完整
lineMargin = 3;
xStart = max(xStart - lineMargin, x1);
xEnd   = min(xEnd + lineMargin, x2);

%% =========================================================
%  8. 最终激光线
%  第一块强制用水平线，不做斜拟合
% =========================================================
xDraw = xStart:xEnd;
yDraw = bestRow * ones(size(xDraw));

disp('第一块检测完成');
disp(['激光线范围：x = ', num2str(xStart), ' 到 ', num2str(xEnd)]);
disp(['激光线 y = ', num2str(bestRow)]);

%% =========================================================
%  9. 黑底图显示
% =========================================================
lineMask = false(imgH, imgW);
ind = sub2ind([imgH, imgW], yDraw, xDraw);
lineMask(ind) = true;

lineMaskShow = imdilate(lineMask, strel('disk', 5));

figure;
imshow(lineMaskShow);
title('第一块激光线黑底显示');

%% =========================================================
%  10. 原图显示
% =========================================================
figure;
imshow(black_image);
hold on;
title('第一块原色木块上的激光直线检测结果');

rectangle('Position', [x1, laserY1, x2-x1, laserY2-laserY1], ...
          'EdgeColor', 'y', 'LineWidth', 1.5);

% 候选点（红点）
for xx = xStart:xEnd
    plot(xx, bestRow, 'r.', 'MarkerSize', 12);
end

% 最终检测线（绿色，加粗）
plot(xDraw, yDraw, 'g-', 'LineWidth', 7);

hold off;

%% =========================================================
%  11. 局部放大显示
% =========================================================
cropPad = 25;

cropX1 = max(x1 - cropPad, 1);
cropY1 = max(y1 - cropPad, 1);
cropX2 = min(x2 + cropPad, imgW);
cropY2 = min(y2 + cropPad, imgH);

cropImg = black_image(cropY1:cropY2, cropX1:cropX2, :);

figure;
imshow(cropImg);
hold on;
title('局部放大 - 第一块木块激光线');

plot(xDraw - cropX1 + 1, yDraw - cropY1 + 1, ...
     'g-', 'LineWidth', 7);

for xx = xStart:xEnd
    plot(xx - cropX1 + 1, bestRow - cropY1 + 1, ...
         'r.', 'MarkerSize', 12);
end

rectangle('Position', [x1-cropX1+1, laserY1-cropY1+1, ...
                       x2-x1, laserY2-laserY1], ...
          'EdgeColor', 'y', 'LineWidth', 1.5);

hold off;