clear;
clc;
close all;

ticProgramTotal = tic;   % 记录整个程序总运行时间
%% =========================================================
% 0. 参数设置
%% =========================================================
netFile = '3DtrainedNetresnet50-50.mat';

% 单张有激光的小方块图像
imgName = 'image04.jpg';

% 网络训练尺寸
inputSize = [720, 1280];

% 类别名称，必须和训练时完全一致
% BG 是背景，其余是小方块类别
classNames = ["BG","black","red","purple"];
backgroundClass = "BG";
targetClasses = classNames(classNames ~= backgroundClass);

% 激光模式：
% "red"    = 红色 / 粉红色 / 紫红色激光
% "bright" = 白色 / 灰白色亮线
% "auto"   = 两种都考虑
laserMode = "auto";

% 小方块区域向外扩展范围，用于提取候选结构光
% 最终画线仍会限制在小方块本体上
objectSearchDilateRadius = 6;

% 是否限制实验大区域
useWorkspace = false;

% 如果 useWorkspace = true，则使用这个范围
workspaceRatio = [0.25, 0.05, 0.55, 0.55];

% 每个类别是否只保留最大连通区域
keepLargestObjectPerClass = true;

%% =========================================================
% 1. 加载语义分割网络
%% =========================================================
load(netFile);

if ~exist('net','var')
    error('模型文件中没有变量 net，请检查 mat 文件。');
end

%% =========================================================
% 2. 读取单张有激光图像
%% =========================================================
I0 = imread(imgName);

if size(I0, 3) == 1
    I0 = cat(3, I0, I0, I0);
end

I = imresize(I0, inputSize);
[imgH, imgW, ~] = size(I);

figure;
imshow(I);
title('原始有激光图像');

%% =========================================================
% 3. 语义分割：识别小方块类别和区域
%% =========================================================
tic;
C = semanticseg(I, net, 'MiniBatchSize', 1);
elapsedTime = toc;

fprintf('语义分割时间：%.4f 秒\n', elapsedTime);

cmap = lines(numel(classNames));

B_overlay = labeloverlay(I, C, ...
    'Colormap', cmap, ...
    'Transparency', 0.45);

figure;
imshow(B_overlay);
title('Semantic Segmentation Result');
pixelLabelColorbar(cmap, classNames);

%% =========================================================
% 4. 构造实验有效区域
%% =========================================================
if useWorkspace
    sceneMask = createWorkspaceMask(imgH, imgW, workspaceRatio);
else
    sceneMask = true(imgH, imgW);
end

figure;
imshow(sceneMask);
title('实验有效区域 sceneMask');

%% =========================================================
% 5. 从语义分割结果中提取每个小方块
%% =========================================================
objects = buildObjectsFromSegmentation( ...
    C, targetClasses, sceneMask, keepLargestObjectPerClass);

figure;
imshow(I);
hold on;
title('语义分割得到的小方块区域');

for i = 1:length(objects)

    visboundaries(objects(i).mask, ...
        'Color', 'y', ...
        'LineWidth', 0.8);

    rectangle('Position', objects(i).bbox, ...
        'EdgeColor', 'c', ...
        'LineWidth', 1.2);

    text(objects(i).bbox(1), max(objects(i).bbox(2)-8, 1), ...
        char(objects(i).className), ...
        'Color', 'y', ...
        'FontSize', 10, ...
        'FontWeight', 'bold');

end

hold off;

%% =========================================================
% 6. 全图计算结构光响应和候选区域
% 注意：这里只生成候选，不直接选择最终结构光
%% =========================================================
ticLaserTotal = tic;      % 结构光提取总计时开始

I_double = im2double(I);

R = I_double(:,:,1);
G = I_double(:,:,2);
B = I_double(:,:,3);

HSV = rgb2hsv(I_double);
H = HSV(:,:,1);
S = HSV(:,:,2);
V = HSV(:,:,3);

ticCandidate = tic;

[laserScore, laserCandidateAll] = buildGlobalLaserCandidate( ...
    R, G, B, H, S, V, sceneMask, laserMode);

candidateTime = toc(ticCandidate);
fprintf('全图结构光候选提取时间：%.4f 秒\n', candidateTime);
figure;
imshow(laserCandidateAll);
title('全图结构光候选区域 laserCandidateAll');

%% =========================================================
% 7. 用小方块语义 mask 过滤结构光候选，并逐个提取中心线
%% =========================================================
ticCenterline = tic;

results = extractLaserForEachObject( ...
    objects, laserCandidateAll, laserScore, objectSearchDilateRadius);

centerlineTime = toc(ticCenterline);
laserTotalTime = toc(ticLaserTotal);

fprintf('小方块结构光中心线提取时间：%.4f 秒\n', centerlineTime);
fprintf('结构光提取总时间：%.4f 秒\n', laserTotalTime);
%% =========================================================
% 8. 显示最终结果：只保留小方块上的结构光
%% =========================================================
figure;
imshow(I);
hold on;
title('小方块上的结构光提取与类别归属结果');

for i = 1:length(results)

    rectangle('Position', results(i).bbox, ...
        'EdgeColor', 'c', ...
        'LineWidth', 1.2);

    if ~isempty(results(i).xCandidate)
        plot(results(i).xCandidate, results(i).yCandidate, ...
            'r.', 'MarkerSize', 8);
    end

    if ~isempty(results(i).xLine)
        plot(results(i).xLine, results(i).yLine, ...
            'g-', 'LineWidth', 3);
    end

    if isempty(results(i).xLine)
        labelText = [char(results(i).className), '：未检测到'];
    else
        labelText = [char(results(i).className), '：结构光'];
    end

    text(results(i).bbox(1), max(results(i).bbox(2)-8, 1), ...
        labelText, ...
        'Color', 'y', ...
        'FontSize', 10, ...
        'FontWeight', 'bold');

end

hold off;

%% =========================================================
% 9. 黑底显示小方块上的结构光中心线
%% =========================================================
lineMaskAll = false(imgH, imgW);

for i = 1:length(results)

    if isempty(results(i).xLine)
        continue;
    end

    x = round(results(i).xLine);
    y = round(results(i).yLine);

    valid = x >= 1 & x <= imgW & ...
            y >= 1 & y <= imgH;

    x = x(valid);
    y = y(valid);

    if isempty(x)
        continue;
    end

    ind = sub2ind([imgH, imgW], y, x);
    lineMaskAll(ind) = true;

end

lineMaskShow = imdilate(lineMaskAll, strel('disk', 3));

figure;
imshow(lineMaskShow);
title('黑底显示小方块上的结构光中心线');

%% =========================================================
% 10. 输出结果
%% =========================================================
disp('================ 结构光提取与归属结果 ================');

for i = 1:length(results)

    if isempty(results(i).xLine)
        disp([char(results(i).className), '：未检测到有效结构光条']);
    else
        disp([char(results(i).className), ...
            '：中心线端点 = (', ...
            num2str(round(results(i).xLine(1))), ', ', ...
            num2str(round(results(i).yLine(1))), ') 到 (', ...
            num2str(round(results(i).xLine(end))), ', ', ...
            num2str(round(results(i).yLine(end))), ')']);
    end

end
programTotalTime = toc(ticProgramTotal);

disp('================ 程序运行时间统计 ================');
fprintf('语义分割时间：%.4f 秒\n', elapsedTime);
fprintf('全图结构光候选提取时间：%.4f 秒\n', candidateTime);
fprintf('小方块结构光中心线提取时间：%.4f 秒\n', centerlineTime);
fprintf('结构光提取总时间：%.4f 秒\n', laserTotalTime);
fprintf('程序总运行时间：%.4f 秒\n', programTotalTime);

%% ========================================================================
%                               函数区
%% ========================================================================

function workspaceMask = createWorkspaceMask(imgH, imgW, workspaceRatio)

    x = round(workspaceRatio(1) * imgW);
    y = round(workspaceRatio(2) * imgH);
    w = round(workspaceRatio(3) * imgW);
    h = round(workspaceRatio(4) * imgH);

    x1 = max(x, 1);
    y1 = max(y, 1);
    x2 = min(x + w, imgW);
    y2 = min(y + h, imgH);

    workspaceMask = false(imgH, imgW);
    workspaceMask(y1:y2, x1:x2) = true;

end

function objects = buildObjectsFromSegmentation( ...
    C, targetClasses, sceneMask, keepLargestObjectPerClass)

    [imgH, imgW] = size(C);
    totalPixels = imgH * imgW;

    objects = struct('className', {}, ...
        'mask', {}, ...
        'bbox', {}, ...
        'area', {});

    count = 0;

    for c = 1:numel(targetClasses)

        className = targetClasses(c);

        classMask = (C == className);
        classMask = classMask & sceneMask;

        classMask = bwareaopen(classMask, 20);
        classMask = imclose(classMask, strel('disk', 5));
        classMask = imfill(classMask, 'holes');

        cc = bwconncomp(classMask);
        stats = regionprops(cc, 'Area', 'BoundingBox', 'PixelIdxList');

        if isempty(stats)
            continue;
        end

        if keepLargestObjectPerClass
            areas = [stats.Area];
            [~, maxId] = max(areas);
            stats = stats(maxId);
        end

        for k = 1:length(stats)

            area = stats(k).Area;
            bbox = stats(k).BoundingBox;

            x = bbox(1);
            y = bbox(2);
            w = bbox(3);
            h = bbox(4);

            areaOK = area > 25 && area < 0.03 * totalPixels;

            sizeOK = w > 4 && h > 4 && ...
                     w < 0.25 * imgW && ...
                     h < 0.25 * imgH;

            borderOK = x > 2 && y > 2 && ...
                       x + w < imgW - 2 && ...
                       y + h < imgH - 2;

            aspectRatio = w / max(h, 1);
            aspectOK = aspectRatio > 0.20 && aspectRatio < 5.00;

            if ~(areaOK && sizeOK && borderOK && aspectOK)
                continue;
            end

            oneMask = false(imgH, imgW);
            oneMask(stats(k).PixelIdxList) = true;

            oneMask = imclose(oneMask, strel('disk', 4));
            oneMask = imfill(oneMask, 'holes');

            count = count + 1;

            objects(count).className = className;
            objects(count).mask = oneMask;
            objects(count).bbox = bbox;
            objects(count).area = area;

        end

    end

end

function [laserScoreSmooth, candidateMask] = buildGlobalLaserCandidate( ...
    R, G, B, H, S, V, sceneMask, laserMode)

    try
        R_bg = medfilt2(R, [25 25], 'symmetric');
        V_bg = medfilt2(V, [25 25], 'symmetric');
    catch
        R_bg = medfilt2(R, [25 25]);
        V_bg = medfilt2(V, [25 25]);
    end

    localRed = R - R_bg;
    localRed(localRed < 0) = 0;

    localBright = V - V_bg;
    localBright(localBright < 0) = 0;

    sumRGB = R + G + B + eps;

    rNorm = R ./ sumRGB;
    gNorm = G ./ sumRGB;
    bNorm = B ./ sumRGB;

    redExcess = R - max(G, B);
    redExcess(redExcess < 0) = 0;

    pinkExcess = 0.75 * R + 0.30 * B - 0.90 * G;
    pinkExcess(pinkExcess < 0) = 0;

    try
        redTophat = imtophat(R, strel('disk', 5));
        brightTophat = imtophat(V, strel('disk', 5));
    catch
        redTophat = R;
        brightTophat = V;
    end

    scoreRed = 1.60 * localRed + ...
               1.20 * redTophat + ...
               0.80 * brightTophat + ...
               0.25 * redExcess + ...
               0.40 * pinkExcess + ...
               0.20 * rNorm + ...
               0.10 * bNorm - ...
               0.40 * gNorm;

    scoreBright = 1.30 * localBright + ...
                  1.10 * brightTophat + ...
                  0.20 * V - ...
                  0.10 * S;

    if laserMode == "red"
        laserScore = scoreRed;
    elseif laserMode == "bright"
        laserScore = scoreBright;
    else
        laserScore = max(scoreRed, scoreBright);
    end

    laserScore(~sceneMask) = 0;

    try
        laserScoreSmooth = imgaussfilt(laserScore, 0.7);
    catch
        h = fspecial('gaussian', [5 5], 0.7);
        laserScoreSmooth = imfilter(laserScore, h, 'replicate');
    end

    redHSV = ((H >= 0.000 & H <= 0.170) | ...
              (H >= 0.800 & H <= 1.000)) & ...
              (S > 0.015) & ...
              (V > 0.045);

    redRGB = (R > 0.045) & ...
             (R > 0.65 * G) & ...
             (R > 0.18 * B);

    magentaCandidate = (R > 0.040) & ...
                       (B > 0.040) & ...
                       (R > 0.55 * G) & ...
                       (B > 0.50 * G) & ...
                       (localRed > 0.003);

    purpleOnBlueCandidate = (R > 0.035) & ...
                            (B > 0.055) & ...
                            (R > 0.45 * G) & ...
                            (B > 0.75 * G) & ...
                            ((localRed > 0.002) | ...
                             (redTophat > 0.006) | ...
                             (brightTophat > 0.006));

    colorCandidate = redHSV | redRGB | magentaCandidate | purpleOnBlueCandidate;

    localLineCandidate = (localRed > 0.004) | ...
                         (redTophat > 0.010) | ...
                         (brightTophat > 0.012);

    brightVals = brightTophat(sceneMask);
    brightVals = brightVals(brightVals > 0);

    if isempty(brightVals)
        brightThreshold = 0.03;
    else
        brightThreshold = prctile(brightVals, 97);
    end

    brightCandidate = (brightTophat > brightThreshold) & ...
                      (V > 0.12);

    redRaw = colorCandidate & localLineCandidate;
    brightRaw = brightCandidate;

    if laserMode == "red"
        rawCandidate = redRaw;
    elseif laserMode == "bright"
        rawCandidate = brightRaw;
    else
        rawCandidate = redRaw | brightRaw;
    end

    rawCandidate = rawCandidate & sceneMask;

    vals = laserScoreSmooth(rawCandidate);
    vals = vals(vals > 0);

    if isempty(vals)
        vals = laserScoreSmooth(sceneMask);
        vals = vals(vals > 0);
    end

    if isempty(vals)
        threshold = 0.02;
    else
        threshold = prctile(vals, 80);
    end

    candidateMask = rawCandidate & ...
                    (laserScoreSmooth > threshold);

    candidateMask = bwareaopen(candidateMask, 1);

    maskFinal = candidateMask;
    angles = [0, 20, 40, 60, 90, 120, 140, 160];

    for i = 1:length(angles)
        maskFinal = maskFinal | imclose(candidateMask, strel('line', 3, angles(i)));
    end

    candidateMask = bwareaopen(maskFinal, 2);

end

function results = extractLaserForEachObject( ...
    objects, laserCandidateAll, laserScore, searchDilateRadius)

    results = struct('className', {}, ...
        'bbox', {}, ...
        'xCandidate', {}, ...
        'yCandidate', {}, ...
        'xLine', {}, ...
        'yLine', {}, ...
        'score', {});

    for i = 1:length(objects)

        objMask = objects(i).mask;
        objClass = string(objects(i).className);

        [imgH, imgW] = size(objMask);

        objSearch = imdilate(objMask, strel('disk', searchDilateRadius));

        %% =================================================
        % red 类小方块特殊处理：
        % 真实结构光在中下部蓝色面上，不在红色上表面
        %% =================================================
        if objClass == "red"

            bbox = objects(i).bbox;

            x = bbox(1);
            y = bbox(2);
            w = bbox(3);
            h = bbox(4);

            x1 = max(round(x + 0.05 * w), 1);
            x2 = min(round(x + 0.95 * w), imgW);

            y1 = max(round(y + 0.38 * h), 1);
            y2 = min(round(y + 0.92 * h), imgH);

            faceMask = false(imgH, imgW);
            faceMask(y1:y2, x1:x2) = true;

            localAllowed = objSearch & faceMask;

        else
            localAllowed = objSearch;
        end

        %% =================================================
        % 用小方块区域过滤全图结构光候选
        %% =================================================
        localCandidate = laserCandidateAll & localAllowed;
        localCandidate = bwareaopen(localCandidate, 1);

        results(i).className = objects(i).className;
        results(i).bbox = objects(i).bbox;
        results(i).xCandidate = [];
        results(i).yCandidate = [];
        results(i).xLine = [];
        results(i).yLine = [];
        results(i).score = 0;

        if nnz(localCandidate) < 2
            continue;
        end

        %% =================================================
        % 当前小方块内部局部阈值
        %% =================================================
        vals = laserScore(localCandidate);
        vals = vals(vals > 0);

        if isempty(vals)
            continue;
        end

        if objClass == "red"
            localThreshold = prctile(vals, 55);
        else
            localThreshold = prctile(vals, 70);
        end

        localLaser = localCandidate & ...
                     (laserScore >= localThreshold);

        localLaser = bwareaopen(localLaser, 1);

        %% =================================================
        % 连通域分析
        %% =================================================
        cc = bwconncomp(localLaser);

        if cc.NumObjects == 0
            continue;
        end

        stats = regionprops(cc, ...
            'Area', ...
            'BoundingBox', ...
            'PixelIdxList', ...
            'MajorAxisLength', ...
            'MinorAxisLength', ...
            'Orientation');

        bestScore = -inf;
        bestId = [];

        objArea = objects(i).area;
        objBbox = objects(i).bbox;
        objW = objBbox(3);
        objH = objBbox(4);

        for k = 1:length(stats)

            area = stats(k).Area;
            bbox = stats(k).BoundingBox;

            w = bbox(3);
            h = bbox(4);

            majorLen = stats(k).MajorAxisLength;
            minorLen = max(stats(k).MinorAxisLength, 1);
            lineRatio = majorLen / minorLen;

            angle = abs(stats(k).Orientation);

            areaOK = area >= 1 && area <= 0.35 * objArea;
            widthOK = w >= max(3, 0.15 * objW);
            heightOK = h <= max(6, 0.45 * objH);
            lineOK = lineRatio >= 1.10;
            angleOK = angle <= 45;

            if ~(areaOK && widthOK && heightOK && lineOK && angleOK)
                continue;
            end

            idx = stats(k).PixelIdxList;

            thisScore = sum(laserScore(idx)) + ...
                        6.0 * lineRatio + ...
                        2.0 * w - ...
                        3.0 * h - ...
                        0.5 * area;

            if thisScore > bestScore
                bestScore = thisScore;
                bestId = k;
            end

        end

        %% =================================================
        % 如果筛选过严，则放宽一次
        %% =================================================
        if isempty(bestId)

            for k = 1:length(stats)

                area = stats(k).Area;
                bbox = stats(k).BoundingBox;

                w = bbox(3);
                h = bbox(4);

                majorLen = stats(k).MajorAxisLength;
                minorLen = max(stats(k).MinorAxisLength, 1);
                lineRatio = majorLen / minorLen;

                areaOK = area >= 1 && area <= 0.50 * objArea;
                heightOK = h <= max(8, 0.60 * objH);

                if ~(areaOK && heightOK)
                    continue;
                end

                idx = stats(k).PixelIdxList;

                thisScore = sum(laserScore(idx)) + ...
                            3.0 * lineRatio + ...
                            1.0 * w - ...
                            2.0 * h;

                if thisScore > bestScore
                    bestScore = thisScore;
                    bestId = k;
                end

            end

        end

        if isempty(bestId)
            continue;
        end

        %% =================================================
        % 取当前小方块内最佳结构光区域
        %% =================================================
        bestMask = false(size(localLaser));
        bestMask(stats(bestId).PixelIdxList) = true;

        [yBest, xBest] = find(bestMask);

        results(i).xCandidate = xBest;
        results(i).yCandidate = yBest;
        results(i).score = bestScore;

        if length(xBest) < 2
            results(i).xLine = xBest;
            results(i).yLine = yBest;
            continue;
        end

        %% =================================================
        % red 类小方块：强制在蓝色面内横向延长
        % 这样解决中间方块结构光太短的问题
        %% =================================================
        if objClass == "red"

            % 只允许最终线落在小方块本体 + 蓝色面区域内
            redFaceMask = objMask & localAllowed;

            [yFace, xFace] = find(redFaceMask);

            if ~isempty(xFace)

                % 横向范围：只取方块蓝色面内部，不超过方块
                % 想更长：改成 5 和 95
                % 想更短：改成 15 和 85
                xLeft  = prctile(xFace, 8);
                xRight = prctile(xFace, 92);

                % y 坐标使用真实检测到的淡紫色结构光高度
                yCenter = round(median(yBest));

                % 防止 yCenter 超出蓝色面区域
                yTop = prctile(yFace, 8);
                yBottom = prctile(yFace, 92);
                yCenter = min(max(yCenter, yTop), yBottom);

                xLine = linspace(xLeft, xRight, 120);
                yLine = yCenter * ones(size(xLine));

                [xLineFinal, yLineFinal] = filterLineInsideMask( ...
                    xLine, yLine, redFaceMask);

                if numel(xLineFinal) >= 2
                    results(i).xLine = xLineFinal;
                    results(i).yLine = yLineFinal;
                    continue;
                end
            end
        end

        %% =================================================
        % black / purple 正常 PCA 拟合
        %% =================================================
        points = [xBest, yBest];

        centerPoint = mean(points, 1);
        pointsZero = points - centerPoint;

        [~, ~, Vpca] = svd(pointsZero, 'econ');

        direction = Vpca(:, 1)';

        tCand = pointsZero * direction';

        t1 = min(tCand);
        t2 = max(tCand);

        lenCand = t2 - t1;

        if lenCand < 1
            lenCand = 1;
        end

        if objClass == "black"
            expandRatio = 0.15;
            maxLen = 0.60 * objW;
        else
            expandRatio = 0.15;
            maxLen = 0.70 * objW;
        end

        tLine1 = t1 - expandRatio * lenCand;
        tLine2 = t2 + expandRatio * lenCand;

        currentLen = tLine2 - tLine1;

        if currentLen > maxLen
            tMid = (tLine1 + tLine2) / 2;
            tLine1 = tMid - maxLen / 2;
            tLine2 = tMid + maxLen / 2;
        end

        xLine = centerPoint(1) + linspace(tLine1, tLine2, 100) * direction(1);
        yLine = centerPoint(2) + linspace(tLine1, tLine2, 100) * direction(2);

        finalMask = objMask;

        [xLineFinal, yLineFinal] = filterLineInsideMask(xLine, yLine, finalMask);

        if numel(xLineFinal) < 2
            xLineFinal = xBest;
            yLineFinal = yBest;
        end

        results(i).xLine = xLineFinal;
        results(i).yLine = yLineFinal;

    end

end

function [xOut, yOut] = filterLineInsideMask(xLine, yLine, mask)

    [imgH, imgW] = size(mask);

    xRound = round(xLine);
    yRound = round(yLine);

    valid = xRound >= 1 & xRound <= imgW & ...
            yRound >= 1 & yRound <= imgH;

    xLine = xLine(valid);
    yLine = yLine(valid);
    xRound = xRound(valid);
    yRound = yRound(valid);

    if isempty(xLine)
        xOut = [];
        yOut = [];
        return;
    end

    ind = sub2ind([imgH, imgW], yRound, xRound);
    inside = mask(ind);

    xOut = xLine(inside);
    yOut = yLine(inside);

end

function pixelLabelColorbar(cmap, classNames)

    colormap(gca, cmap);

    c = colorbar;

    n = numel(classNames);

    c.Ticks = linspace(1/(2*n), 1 - 1/(2*n), n);
    c.TickLabels = classNames;
    c.TickLength = 0;

end