clear
clc
load '3DtrainedNetresnet50-50.mat';%trainedNet1为resnet18模型，trainedNet3为resnet50模型
%[file,path]=uigetfile('D:\桌面文件夹\robot course\arduino\视觉识别\语义分割虚拟\测试图片\');
%filepath=fullfile(path,file);
I=imread('image01.jpg');
%cam=webcam(2);
%preview(cam);
%cam.Resolution='1920x1080';
%cam.Brightness=-10;%调整相机亮度
%I =snapshot(cam);

figure(1);
imshow(I);



I=imresize(I,[720, 1280]);%imresize(I,[720, 1280]);
tic;  % 开始计时
C=semanticseg(I,net,'MiniBatchSize', 32);
%pxds =pixelLabelDatastore(I,classes,labelIDs);
%classes=["green","red", "blue","background"];%["red", "blue","green","background"];
classes=["BG","black","red","purple"];%["red", "blue","green","background"];
%classes=["Bei","Red", "Green","Black","Grey"];
cmap=camvidColorMap;%需要更改内参数
B=labeloverlay(I,C,'ColorMap',cmap,'Transparency',0.4);

elapsedTime = toc;  % 结束计时
fprintf('程序运行时间: %.4f 秒\n', elapsedTime);

figure(2);
imshow(B),title("Semantic segmentation Result");
pixelLabelColorbar(cmap,classes);
