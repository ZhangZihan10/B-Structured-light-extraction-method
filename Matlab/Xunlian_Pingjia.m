clear

%% 1) 载入网络与 groundTruth
load trainedNetResnet50-100.mat   % 已训练的网络，例如 trainedNetResnet50
load gTruth.mat                   % groundTruth 数据

% groundTruth 对象别名
if exist('groundTruth','var') && ~exist('gTruth','var')
    gTruth = groundTruth;
end

%% 2) 图像数据存储：来自 DataSource.Source
imds = imageDatastore(gTruth.DataSource.Source);

%% 3) 提取像素标签类别及其像素ID（仅选 PixelLabel 类型）
defs = gTruth.LabelDefinitions;
isPixel = defs.Type == "PixelLabel";
classNames    = defs.Name(isPixel);
pixelLabelIDs = defs.PixelLabelID(isPixel);

%% 4) 取出像素标注图的路径列（而不是整张 LabelData 表）
% 常见列名为 'PixelLabelData'；若不同，请替换为实际列名
assert(ismember('PixelLabelData', gTruth.LabelData.Properties.VariableNames), ...
    '在 gTruth.LabelData 中未找到 PixelLabelData 列，请检查列名。');

labelPaths = gTruth.LabelData.PixelLabelData;   % cellstr/string 各行是一个标注图路径
labelPaths = string(labelPaths)';               % 转为 1xN 行向量

%% 5) 像素标签数据存储
pxdsTruth = pixelLabelDatastore(labelPaths, classNames, pixelLabelIDs);

%% 6) 预测
if exist('trainedNetResnet50','var')
    net = trainedNetResnet50;
end

pxdsPred = semanticseg(imds, net, ...
    'MiniBatchSize', 8, ...
    'Verbose', false);

%% 评估
metrics = evaluateSemanticSegmentation(pxdsPred, pxdsTruth, ...
    'Metrics', ["global-accuracy","accuracy","iou","weighted-iou","bfscore"]);

%% 8) 输出结果
disp('=== 数据集整体指标 ===');
disp(metrics.DataSetMetrics);   % MeanAccuracy, MeanIoU, WeightedIoU 等

disp('=== 各类别指标 ===');
disp(metrics.ClassMetrics);     % 每类的 Accuracy / Precision / Recall / F1 / IoU

% 如果只要 Precision / Recall / F1 表格
%T = metrics.ClassMetrics(:, {'Name','Precision','Recall','F1Score'});
%disp('=== 各类别 Precision / Recall / F1 ===');
%disp(T);


%% —— 计算每类 Precision/Recall/F1（逐像素）——
numClasses = numel(classNames);
TP = zeros(numClasses,1);
FP = zeros(numClasses,1);
FN = zeros(numClasses,1);
PIX = zeros(numClasses,1);  % 每类像素总数（=TP+FN）

numImgs = numel(imds.Files);
for i = 1:numImgs
    gt  = readimage(pxdsTruth, i);   % 真实分割：categorical 矩阵
    pr  = readimage(pxdsPred,  i);   % 预测分割：categorical 矩阵

    % 确保两边类别集一致（必要时统一类别顺序）
    if ~isequal(categories(gt), categories(pr))
        gt = categorical(gt, classNames);
        pr = categorical(pr, classNames);
    end

    for c = 1:numClasses
        cls = classNames(c);
        g = (gt == cls);
        p = (pr == cls);

        tp = nnz(g & p);
        fp = nnz(~g & p);
        fn = nnz(g & ~p);

        TP(c)  = TP(c)  + tp;
        FP(c)  = FP(c)  + fp;
        FN(c)  = FN(c)  + fn;
        PIX(c) = PIX(c) + tp + fn;   % 该类的真实像素总数
    end
end

% 避免除零
epsv = 1e-12;

prec = TP ./ max(TP + FP, epsv);
recl = TP ./ max(TP + FN, epsv);
f1   = 2 * prec .* recl ./ max(prec + recl, epsv);
iou2 = TP ./ max(TP + FP + FN, epsv);  % 自算 IoU（可与 metrics 里的 IoU 对照）

% 宏平均（各类等权）
macroP  = mean(prec);
macroR  = mean(recl);
macroF1 = mean(f1);

% 加权平均（按每类像素数加权，更贴近数据分布）
w = PIX / max(sum(PIX), epsv);
wP = sum(w .* prec);
wR = sum(w .* recl);
wF1= sum(w .* f1);

% 输出成表格
T = table(classNames, TP, FP, FN, prec, recl, f1, iou2, ...
    'VariableNames',{'Class','TP','FP','FN','Precision','Recall','F1','IoU_from_counts'});
disp('=== Per-class metrics (computed from pixels) ===');
disp(T);

fprintf('\n=== Macro averages ===\nP=%.4f  R=%.4f  F1=%.4f\n', macroP, macroR, macroF1);
fprintf('=== Weighted averages ===\nP=%.4f  R=%.4f  F1=%.4f\n', wP, wR, wF1);
