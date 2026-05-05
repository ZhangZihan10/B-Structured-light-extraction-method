clear;
clc;
close all;

%% =========================================================
%  1. 读取图像
% ==========================================================
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
%  2. 第三块灰紫木块大致 ROI
%  如果绿色框没有框住第三块，就只调这里
% ==========================================================

x1 = round(0.515 * imgW);
x2 = round(0.565 * imgW);

y1 = round(0.190 * imgH);
y2 = round(0.320 * imgH);

roughMask = false(imgH, imgW);
roughMask(y1:y2, x1:x2) = true;

figure;
imshow(black_image);
hold on;
rectangle('Position', [x1, y1, x2-x1, y2-y1], ...
          'EdgeColor', 'g', 'LineWidth', 2);
title('第三块灰紫木块粗略 ROI');
hold off;

%% =========================================================
%  3. 在 ROI 内自动提取第三块木块本体
%  目的：排除周围白色网格纸
% ==========================================================

Hroi = H(y1:y2, x1:x2);
Sroi = S(y1:y2, x1:x2);
Vroi = V(y1:y2, x1:x2);

% 木块相对白纸更暗，且有一定颜色饱和度
blockLocalMask = ((Vroi < 0.72) | (Sroi > 0.16));

blockLocalMask = bwareaopen(blockLocalMask, 20);
blockLocalMask = imclose(blockLocalMask, strel('disk', 3));
blockLocalMask = imfill(blockLocalMask, 'holes');

ccBlock = bwconncomp(blockLocalMask);
statsBlock = regionprops(ccBlock, 'Area', 'BoundingBox', 'Centroid');

if isempty(statsBlock)
    error('没有找到第三块木块本体，请放宽 blockLocalMask 条件。');
end

% 选择面积最大的区域作为第三块木块
areas = [statsBlock.Area];
[~, idBlock] = max(areas);
bboxBlock = statsBlock(idBlock).BoundingBox;

bx = bboxBlock(1);
by = bboxBlock(2);
bw = bboxBlock(3);
bh = bboxBlock(4);

% 转换到全图坐标
block_x1 = x1 + floor(bx) - 1;
block_x2 = x1 + ceil(bx + bw) - 1;
block_y1 = y1 + floor(by) - 1;
block_y2 = y1 + ceil(by + bh) - 1;

block_x1 = max(block_x1, 1);
block_x2 = min(block_x2, imgW);
block_y1 = max(block_y1, 1);
block_y2 = min(block_y2, imgH);

figure;
imshow(black_image);
hold on;
rectangle('Position', [block_x1, block_y1, block_x2-block_x1, block_y2-block_y1], ...
          'EdgeColor', 'c', 'LineWidth', 2);
title('第三块木块本体区域');
hold off;

%% =========================================================
%  4. 只在第三块木块本体中下部寻找激光
%  这里非常关键：不能再搜到旁边网格纸
% ==========================================================

laser_y1 = round(block_y1 + 0.35 * (block_y2 - block_y1));
laser_y2 = round(block_y1 + 0.95 * (block_y2 - block_y1));

laser_x1 = block_x1;
laser_x2 = block_x2;

laserMask = false(imgH, imgW);
laserMask(laser_y1:laser_y2, laser_x1:laser_x2) = true;

figure;
imshow(black_image);
hold on;
rectangle('Position', [laser_x1, laser_y1, laser_x2-laser_x1, laser_y2-laser_y1], ...
          'EdgeColor', 'y', 'LineWidth', 2);
title('第三块木块内部激光搜索区域');
hold off;

%% =========================================================
%  5. 针对灰紫背景增强红色/粉红激光
% ==========================================================

try
    R_bg = medfilt2(R, [15 15], 'symmetric');
catch
    R_bg = medfilt2(R, [15 15]);
end

localRed = R - R_bg;
localRed(localRed < 0) = 0;

sumRGB = R + G + B + eps;
rNorm = R ./ sumRGB;
gNorm = G ./ sumRGB;
bNorm = B ./ sumRGB;

% 灰紫背景上的红/粉色激光得分
laserScore = 0.78 * rNorm + 0.22 * bNorm - 0.88 * gNorm + 1.35 * localRed;

laserScore(~laserMask) = 0;

try
    laserScoreSmooth = imgaussfilt(laserScore, 0.7);
catch
    hGauss = fspecial('gaussian', [5 5], 0.7);
    laserScoreSmooth = imfilter(laserScore, hGauss, 'replicate');
end

%% =========================================================
%  6. 提取红色/粉红色激光候选区域
% ==========================================================

redPinkHSV = ((H >= 0.000 & H <= 0.135) | ...
              (H >= 0.830 & H <= 1.000)) & ...
              (S > 0.030) & ...
              (V > 0.080);

redPinkRGB = (R > 0.08) & ...
             (R > 0.82 * G) & ...
             (R > 0.25 * B);

vals = laserScoreSmooth(laserMask);
vals = vals(vals > 0);

if isempty(vals)
    error('第三块木块内部没有检测到红色激光响应。');
end

% 用较高分位数做阈值，减少网格纸误检
scoreThreshold = prctile(vals, 86);

candidateMask = laserMask & redPinkHSV & redPinkRGB & ...
                (laserScoreSmooth > scoreThreshold);

candidateMask = bwareaopen(candidateMask, 2);
candidateMask = imclose(candidateMask, strel('line', 5, 0));

figure;
imshow(imdilate(candidateMask, strel('disk', 2)));
title('第三块红色激光候选区域');

%% =========================================================
%  7. 选择真正的激光连通区域
%  关键：只保留最像“横向激光线”的连通块
% ==========================================================

cc = bwconncomp(candidateMask);
stats = regionprops(cc, 'Area', 'BoundingBox', 'PixelIdxList');

if isempty(stats)
    error('没有找到有效的激光连通区域，请降低 scoreThreshold 或检查 ROI。');
end

bestScore = -inf;
bestId = 1;

for i = 1:length(stats)
    box = stats(i).BoundingBox;
    area = stats(i).Area;

    wBox = box(3);
    hBox = box(4);

    % 横向线段应该宽大于高
    aspectScore = wBox / max(hBox, 1);

    % 激光不能太大，否则可能是误检区域
    if area < 2
        continue;
    end

    % 综合评分：面积 + 横向程度
    thisScore = area + 5 * aspectScore;

    if thisScore > bestScore
        bestScore = thisScore;
        bestId = i;
    end
end

laserComponent = false(imgH, imgW);
laserComponent(stats(bestId).PixelIdxList) = true;

[yPoints, xPoints] = find(laserComponent);

if length(xPoints) < 2
    error('有效激光点太少，无法拟合直线。');
end

%% =========================================================
%  8. 拟合激光线
%  只在激光连通区域范围内画线，不再画满整个 ROI
% ==========================================================

if length(xPoints) >= 3
    p = polyfit(xPoints, yPoints, 1);
    k_raw = p(1);

    % 第三块表面上的激光基本接近水平，限制斜率，防止被噪声拉歪
    k = max(min(k_raw, 0.015), -0.015);

    bLine = median(yPoints - k * xPoints);
else
    k = 0;
    bLine = median(yPoints);
end

% 只在真实激光点范围内画线
lineMargin = 2;

xStart = max(min(xPoints) - lineMargin, laser_x1);
xEnd   = min(max(xPoints) + lineMargin, laser_x2);

xDraw = xStart:xEnd;
yDraw = round(k * xDraw + bLine);

valid = xDraw >= 1 & xDraw <= imgW & ...
        yDraw >= 1 & yDraw <= imgH;

xDraw = xDraw(valid);
yDraw = yDraw(valid);

disp('第三块检测完成：');
disp(['第三块激光直线方程： y = ', num2str(k), ' * x + ', num2str(bLine)]);
disp(['激光线绘制范围 x = ', num2str(xStart), ' 到 ', num2str(xEnd)]);

%% =========================================================
%  9. 原图显示结果
% ==========================================================

figure;
imshow(black_image);
hold on;
title('第三块灰紫小木块红色激光检测结果');

rectangle('Position', [laser_x1, laser_y1, laser_x2-laser_x1, laser_y2-laser_y1], ...
          'EdgeColor', 'y', 'LineWidth', 1.5);

plot(xPoints, yPoints, 'r.', 'MarkerSize', 12);
plot(xDraw, yDraw, 'g-', 'LineWidth', 4);

hold off;

%% =========================================================
%  10. 黑底图显示结果
% ==========================================================

lineMask = false(imgH, imgW);

if ~isempty(xDraw)
    ind = sub2ind([imgH, imgW], yDraw, xDraw);
    lineMask(ind) = true;
end

lineMaskShow = imdilate(lineMask, strel('disk', 4));

figure;
imshow(lineMaskShow);
title('第三块激光线黑底图');

%% =========================================================
%  11. 局部放大显示
% ==========================================================

cropPad = 20;

cropX1 = max(laser_x1 - cropPad, 1);
cropY1 = max(laser_y1 - cropPad, 1);
cropX2 = min(laser_x2 + cropPad, imgW);
cropY2 = min(laser_y2 + cropPad, imgH);

cropImg = black_image(cropY1:cropY2, cropX1:cropX2, :);

figure;
imshow(cropImg);
hold on;
title('局部放大 - 第三块灰紫小木块激光线');

plot(xPoints - cropX1 + 1, yPoints - cropY1 + 1, ...
     'r.', 'MarkerSize', 12);

plot(xDraw - cropX1 + 1, yDraw - cropY1 + 1, ...
     'r-', 'LineWidth', 4);

rectangle('Position', [laser_x1-cropX1+1, laser_y1-cropY1+1, ...
                       laser_x2-laser_x1, laser_y2-laser_y1], ...
          'EdgeColor', 'y', 'LineWidth', 1.5);

hold off;