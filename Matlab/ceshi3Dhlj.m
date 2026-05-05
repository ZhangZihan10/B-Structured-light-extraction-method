clear
load 3DtrainedNetresnet50-10.mat;%trainedNet1为resnet18模型，trainedNet3为resnet50模型
%[file,path]=uigetfile('D:\桌面文件夹\robot course\arduino\视觉识别\语义分割虚拟\测试图片\');
%filepath=fullfile(path,file);
I=imread('a15.jpg');
%cam=webcam(2);
%preview(cam);
%cam.Resolution='1920x1080';
%cam.Brightness=-10;%调整相机亮度
%I =snapshot(cam);

figure(1);
imshow(I);

I=imresize(I,[1920, 1920]);%imresize(I,[1080, 1080]);

C=semanticseg(I,net,'MiniBatchSize', 32);
%pxds =pixelLabelDatastore(I,classes,labelIDs);
%classes=["green","red", "blue","background"];%["red", "blue","green","background"];
classes=["obstacleboard","cube","platform"];%["red", "blue","green","background"];
%classes=["Bei","Red", "Green","Black","Grey"];
cmap=camvidColorMap;%需要更改内参数
B=labeloverlay(I,C,'ColorMap',cmap,'Transparency',0.4);
figure(2);
imshow(B),title("Semantic segmentation Result");
pixelLabelColorbar(cmap,classes);

% 创建 mask：只保留 cube 区域
cubeMask = C == 'obstacleboard';

% 将 mask 扩展为 3 通道（与 RGB 对应）
cubeMask3Channel = repmat(cubeMask, [1, 1, 3]);

% 将原始图像中非 cube 区域置为黑色
cubeOnlyImage = I; % 复制原图
cubeOnlyImage(~cubeMask3Channel) = 0; % 非 cube 区域设为 0

% 显示结果
figure;
imshow(cubeOnlyImage);
title('Only Cube Region');