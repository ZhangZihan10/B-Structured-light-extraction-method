%clc
%clear;
%load gTruth.mat;
%imageDir='测试图片132';
%imds=imageDatastore(imageDir);

%classNames=["bluecube","greencube","orangecube","yellwosqhere","redsqhere","background"];
%pxds=pixelLabelDatastore(gTruth);

%imageSize=[640 640 3];
%numClasses=numel(classNames);
%lgraph=deeplabv3plusLayers(imageSize,numClasses,"resnet50");

%pximds=pixelLabelImageDatastore(imds,pxds,'OutputSize',[640 640 3],...
    %'ColorPreprocessing','gray2rgb');

%opts=trainingOptions("sgdm",'ExecutionEnvironment','gpu',...
   % 'InitialLearnRate',0.001,'MiniBatchSize',4,'Plots',...
   % 'training-progress','MaxEpochs',35);

%[net,info]=trainNetwork(pximds,lgraph,opts);

%save('trainedNet50-10.mat','net');
%save('trainedInfo50-10.mat','info');
% 加载验证数据
clc
clear;
load gTruth.mat;
% 使用gTruth中的图像路径创建imds，确保与pxds顺序一致
imds = imageDatastore(gTruth.DataSource.Source); % 替换原来的imageDir方式

classNames = ["bluecube","greencube","orangecube","yellwosqhere","redsqhere","background"];
pxds = pixelLabelDatastore(gTruth);

imageSize = [640 640 3];
numClasses = numel(classNames);
lgraph = deeplabv3plusLayers(imageSize, numClasses, "mobilenetv2");

pximds = pixelLabelImageDatastore(imds, pxds, 'OutputSize', [640 640 3],...
    'ColorPreprocessing', 'gray2rgb');

opts = trainingOptions("sgdm", 'ExecutionEnvironment', 'gpu',...
    'InitialLearnRate', 0.001, 'MiniBatchSize', 4, 'Plots',...
    'training-progress', 'MaxEpochs', 100);

[net, info] = trainNetwork(pximds, lgraph, opts);

save('trainedNetMobilenetv2-100.mat', 'net');
save('trainedInfoMobilenetv2-100.mat', 'info');
