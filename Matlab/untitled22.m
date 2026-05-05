clear;
clc;
close all;

%% =========================================================
%  0. 参数区
% ==========================================================
imgName = 'image01.jpg';

% 第二块蓝色木块大致搜索区域
search_y1_ratio = 0.14;
search_y2_ratio = 0.36;
search_x1_ratio = 0.36;
search_x2_ratio = 0.52;

% 第二块蓝色木块大致中心位置
target_cx_ratio = 0.44;
target_cy_ratio = 0.25;

% 只在蓝色面内部寻找紫红色激光
% 这里不要太靠上，也不要太靠下
laser_y1_factor = 0.35;
laser_y2_factor = 0.82;

% 手动微调最终线条位置
% 如果线偏上，改成 1、2、3
% 如果线偏下，改成 -1、-2、-3
manualYOffset = 0;

% 显示粗细
line_thickness_mask = 4;
line_width_overlay = 4;

%% =========================================================
%  1. 读取图像
% ==========================================================
black_image = imread(imgName);

if size(black_image, 3) == 1
    black_image = cat(3, black_image, black_image, black_image);
end

img = im2double(black_image);
[imgH, imgW, ~] = size(img);

%% =========================================================
%  2. RGB 与 HSV
% ==========================================================
hsvImage = rgb2hsv(img);

H = hsvImage(:,:,1);
S = hsvImage(:,:,2);
V = hsvImage(:,:,3);

R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);

%% =========================================================
%  3. 自动寻找第二块蓝色木块
% ==========================================================
searchMask = false(imgH, imgW);

sy1 = max(round(search_y1_ratio * imgH), 1);
sy2 = min(round(search_y2_ratio * imgH), imgH);
sx1 = max(round(search_x1_ratio * imgW), 1);
sx2 = min(round(search_x2_ratio * imgW), imgW);

searchMask(sy1:sy2, sx1:sx2) = true;

% 蓝色区域条件
blueMask = (H > 0.52 & H < 0.72) & ...
           (S > 0.15) & ...
           (V > 0.10) & ...
           (B > 0.15) & ...
           (B > 1.02 * R) & ...
           (B > 1.02 * G) & ...
           searchMask;

blueMask = bwareaopen(blueMask, 8);
blueMask = imclose(blueMask, strel('disk', 2));

figure;
imshow(blueMask);
title('Blue Mask - 第二块蓝色木块');

cc = bwconncomp(blueMask);
stats = regionprops(cc, 'Area', 'BoundingBox', 'Centroid');

if isempty(stats)
    error('没有找到第二块蓝色木块，请放宽蓝色阈值。');
end

% 选择最接近中间第二块木块的蓝色区域
scores = zeros(length(stats), 1);

for i = 1:length(stats)
    cx = stats(i).Centroid(1);
    cy = stats(i).Centroid(2);
    area = stats(i).Area;

    posPenalty = 800 * abs(cx / imgW - target_cx_ratio) + ...
                 400 * abs(cy / imgH - target_cy_ratio);

    scores(i) = area - posPenalty;
end

[~, idx] = max(scores);
bbox = stats(idx).BoundingBox;

bx = bbox(1);
by = bbox(2);
bw = bbox(3);
bh = bbox(4);

%% =========================================================
%  4. 构造蓝色面内部 ROI
%  重点：只看蓝色面，不看上方红色块，也不看最底部反光
% ==========================================================
laser_x1 = max(floor(bx - 0.10 * bw), 1);
laser_x2 = min(ceil(bx + 1.10 * bw), imgW);

laser_y1 = max(floor(by + laser_y1_factor * bh), 1);
laser_y2 = min(ceil(by + laser_y2_factor * bh), imgH);

laserROIMask = false(imgH, imgW);
laserROIMask(laser_y1:laser_y2, laser_x1:laser_x2) = true;

figure;
imshow(black_image);
hold on;
rectangle('Position', [laser_x1, laser_y1, laser_x2-laser_x1, laser_y2-laser_y1], ...
          'EdgeColor', 'g', 'LineWidth', 2);
title('ROI - 第二块蓝色面内部紫红色激光搜索区域');
hold off;

%% =========================================================
%  5. 针对蓝色背景增强紫红色激光
% ==========================================================
% 蓝色背景会导致 B 通道很强，所以不能要求 R > B
% 紫红激光的特点是：
% 1. R 相对周围蓝色面增加
% 2. G 较低
% 3. R 和 B 同时存在，看起来偏紫红

% 局部红色增强：把红色通道减去周围背景
R_bg = medfilt2(R, [15 15], 'symmetric');
localRed = R - R_bg;
localRed(localRed < 0) = 0;

% 归一化颜色比例，减少亮度影响
sumRGB = R + G + B + eps;
rNorm = R ./ sumRGB;
gNorm = G ./ sumRGB;
bNorm = B ./ sumRGB;

% 紫红色得分：增强 R，保留 B，压制 G
purpleScore = 0.55 * rNorm + 0.35 * bNorm - 0.90 * gNorm + 1.20 * localRed;

% 只在蓝色面 ROI 内计算
purpleScore(~laserROIMask) = 0;

% 平滑一下，增强连续线条
purpleScoreSmooth = imgaussfilt(purpleScore, 0.8);

% 候选区域阈值：使用自适应阈值
vals = purpleScoreSmooth(laserROIMask);
vals = vals(vals > 0);

if isempty(vals)
    error('ROI 内没有有效紫红色响应。');
end

scoreThreshold = mean(vals) + 0.45 * std(vals);

candidateMask = laserROIMask & (purpleScoreSmooth > scoreThreshold);

% 去除小噪声，轻微连接横向线
candidateMask = bwareaopen(candidateMask, 1);
candidateMask = imclose(candidateMask, strel('line', 5, 0));

candidateMaskShow = imdilate(candidateMask, strel('disk', 2));

figure;
imshow(candidateMaskShow);
title('Purple Candidate Mask - 蓝色面上的紫红色激光候选区域');

%% =========================================================
%  6. 估计蓝色面方向
%  用蓝色块上边缘估计斜率，使激光线和蓝色面平行
% ==========================================================
blueTopX = [];
blueTopY = [];

colStart = max(round(bx), 1);
colEnd   = min(round(bx + bw), imgW);

for col = colStart:colEnd
    rows = find(blueMask(:, col));
    if ~isempty(rows)
        blueTopX(end+1) = col;
        blueTopY(end+1) = min(rows);
    end
end

if length(blueTopX) >= 5
    pBlue = polyfit(blueTopX, blueTopY, 1);
    kBlue = pBlue(1);
else
    kBlue = 0;
end

disp(['蓝色面参考斜率 kBlue = ', num2str(kBlue)]);

%% =========================================================
%  7. 用行投影找出最明显的紫红色横线
%  这一步比逐点识别更稳定，适合弱激光
% ==========================================================
scoreOnlyROI = purpleScoreSmooth;
scoreOnlyROI(~laserROIMask) = 0;

rowProfile = sum(scoreOnlyROI(:, laser_x1:laser_x2), 2);
rowProfileSmooth = movmean(rowProfile, 5);

% 只在 ROI 的 y 范围内找最大行
roiRows = laser_y1:laser_y2;
[~, maxIdx] = max(rowProfileSmooth(roiRows));

bestRow = roiRows(maxIdx);

% 手动微调最终线位置
bestRow = bestRow + manualYOffset;

disp(['检测到的紫红激光中心行 y = ', num2str(bestRow)]);

%% =========================================================
%  8. 在 bestRow 附近提取有效点
% ==========================================================
xPoints = [];
yPoints = [];

searchHalfHeight = 4;

for col = laser_x1:laser_x2

    yMin = max(bestRow - searchHalfHeight, laser_y1);
    yMax = min(bestRow + searchHalfHeight, laser_y2);

    columnData = purpleScoreSmooth(yMin:yMax, col);

    if isempty(columnData)
        continue;
    end

    [maxVal, rowIndex] = max(columnData);
    realY = yMin + rowIndex - 1;

    % 只要比局部均值高一点就保留，避免弱激光被删掉
    if maxVal > 0.6 * scoreThreshold
        xPoints(end+1) = col;
        yPoints(end+1) = realY;
    end
end

disp(['用于拟合的激光点数量：', num2str(length(xPoints))]);

%% =========================================================
%  9. 生成最终激光直线
%  如果点足够，用点修正截距；如果点少，直接使用 bestRow
% ==========================================================
centerX = mean([laser_x1, laser_x2]);

if length(xPoints) >= 3
    % 固定斜率为蓝色面斜率，只求截距
    bLaser = median(yPoints - kBlue * xPoints);
else
    % 点太少时，仍然输出一条位于蓝色面上的线
    bLaser = bestRow - kBlue * centerX;
end

xDraw = laser_x1:laser_x2;
yDraw = round(kBlue * xDraw + bLaser);

valid = (xDraw >= 1) & (xDraw <= imgW) & ...
        (yDraw >= 1) & (yDraw <= imgH);

xDraw = xDraw(valid);
yDraw = yDraw(valid);

disp('检测完成：');
disp(['激光直线方程： y = ', num2str(kBlue), ' * x + ', num2str(bLaser)]);

%% =========================================================
%  10. 黑底图显示完整激光线
% ==========================================================
lineMask = false(imgH, imgW);

ind = sub2ind([imgH, imgW], yDraw, xDraw);
lineMask(ind) = true;

lineMaskShow = imdilate(lineMask, strel('disk', line_thickness_mask));

figure;
imshow(lineMaskShow);
title('Laser Line - 第二块蓝色面上的紫红色激光线');

%% =========================================================
%  11. 原图显示结果
% ==========================================================
figure;
imshow(black_image);
hold on;
title('第二块蓝色木块紫红色激光检测结果');

% ROI 框
rectangle('Position', [laser_x1, laser_y1, laser_x2-laser_x1, laser_y2-laser_y1], ...
          'EdgeColor', 'y', 'LineWidth', 1.5);

% 检测点
if length(xPoints) >= 1
    plot(xPoints, yPoints, 'r.', 'MarkerSize', 16);
end

% 最终完整激光线
plot(xDraw, yDraw, 'g-', 'LineWidth', line_width_overlay);

hold off;

%% =========================================================
%  12. 局部放大显示
% ==========================================================
cropPad = 25;

cropX1 = max(laser_x1 - cropPad, 1);
cropY1 = max(laser_y1 - cropPad, 1);
cropX2 = min(laser_x2 + cropPad, imgW);
cropY2 = min(laser_y2 + cropPad, imgH);

cropImg = black_image(cropY1:cropY2, cropX1:cropX2, :);

figure;
imshow(cropImg);
hold on;
title('局部放大 - 第二块蓝色面激光线');

plot(xDraw - cropX1 + 1, yDraw - cropY1 + 1, 'r-', 'LineWidth', line_width_overlay);

if length(xPoints) >= 1
    plot(xPoints - cropX1 + 1, yPoints - cropY1 + 1, 'r.', 'MarkerSize', 16);
end

rectangle('Position', [laser_x1-cropX1+1, laser_y1-cropY1+1, ...
                       laser_x2-laser_x1, laser_y2-laser_y1], ...
          'EdgeColor', 'y', 'LineWidth', 1.5);

hold off;