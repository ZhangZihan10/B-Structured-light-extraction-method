%cam=webcam(2);
%preview(cam);

%cam.Resolution='1920x1080';
%cam.Brightness=0;%调整相机亮度

%image =snapshot(cam);

%load('Omni_Calib_Results1.mat'); % Calib parameters
%ocam_model = calib_data.ocam_model; % Calib parameters

%第一种算法，改进型
%black_image=ceshiBnew(image);%识别黑色方块，提取黑色方块所在区域.使用CNN
%imshow(black_image);

%图像处理，增强激光线段
% 将图像从RGB转换到HSV
black_image=imread("image01.jpg");
hsvImage = rgb2hsv(black_image);

% 转换红色激光的RGB值到HSV值（手动估算）
red_rgb = [240/255, 0, 0]; % RGB值转换到0-1范围
red_hsv = rgb2hsv(reshape(red_rgb, [1 1 3])); % 转换到HSV

% 红色激光的HSV范围，使用一些偏移量
hue = red_hsv(1);
saturation = red_hsv(2);
value = red_hsv(3);

% 设置HSV范围（根据需要调整偏移量）
hue_offset = 0.05;
sat_offset = 0.5;
val_offset = 0.5;

lower_red1 = [hue - hue_offset, max(saturation - sat_offset, 0), max(value - val_offset, 0)];
upper_red1 = [hue + hue_offset, 1, 1];

% 创建掩膜，提取红色部分
redMask = (hsvImage(:,:,1) >= lower_red1(1)) & (hsvImage(:,:,1) <= upper_red1(1)) & ...
          (hsvImage(:,:,2) >= lower_red1(2)) & (hsvImage(:,:,2) <= upper_red1(2)) & ...
          (hsvImage(:,:,3) >= lower_red1(3)) & (hsvImage(:,:,3) <= upper_red1(3));

% 显示红色部分掩膜
figure;
imshow(redMask);
title('Red Mask');

% 对掩膜进行形态学操作（膨胀和腐蚀）
se = strel('disk', 1); % 使用较小的结构元素
redMask = imdilate(redMask, se);
redMask = imerode(redMask, se);

% 显示形态学处理后的掩膜
figure;
imshow(redMask);
title('Morphologically Processed Red Mask');

%第二种算法
img = las_segm(black_image);

%第三种方法
% 提取红色通道img=black_image;
redChannel = black_image(:,:,1);

% 将红色通道转换为灰度图像
grayRed = mat2gray(redChannel);

% 二值化处理
binaryRed = imbinarize(grayRed, 'adaptive', 'Sensitivity', 0.4);

% 形态学操作，去除噪点和增强线条
se = strel('line', 10, 0);
dilatedRed = imdilate(binaryRed, se);
erodedRed = imerode(dilatedRed, se);

% 使用Canny边缘检测
edges = edge(erodedRed, 'canny');

% 霍夫变换检测直线
[H, theta, rho] = hough(edges);
peaks = houghpeaks(H, 5, 'threshold', ceil(0.3*max(H(:))));
lines = houghlines(edges, theta, rho, peaks, 'FillGap', 20, 'MinLength', 40);

% 显示结果
figure, imshow(black_image), hold on
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'green');
end
hold off