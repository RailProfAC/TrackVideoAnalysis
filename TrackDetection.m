clear all
close all
clc

filename = 'DriversDeskView7.png';
In = imread(filename);

tic
I = rgb2gray(In);
I = (I(1:end,1:1200));
Io = I;
I = imgaussfilt(I,5);
I(find(I > 150)) = 0;
I = 255-I;

E = edge(I, 'canny');

[H,T,R] = hough(E, 'Theta', -10:0.5:10);

P  = houghpeaks(H,2,'threshold',ceil(0.2*max(H(:))), 'NHoodSize', 41*[1, 1]);
x = T(P(:,2)); y = R(P(:,1));
lines = houghlines(I,T,R,P,'FillGap',5,'MinLength',7);
toc

figure
subplot(221)
imshow(I)
subplot(222)
imagesc(E)
subplot(223)
imagesc(T, R, H)
hold on
plot(x,y,'s','color','red');
subplot(224)
imshow(In), hold on
max_len = 0;
XY = [];
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',5,'Color','blue');
    XY = [XY; xy];
    
   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end
XYsave = XY;
XY = sortrows(XY);
XY(2,1) = XY(1,1) + (XY(2,1) - XY(1,1))*2/3;
XY(2,2) = XY(1,2)*1/3;
XY(3,1) = XY(4,1) + (XY(3,1) - XY(4,1))*2/3;
XY(3,2) = XY(4,2)*1/3;
% Plot beginnings and ends of lines
%plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
f = fill(XY(:,1), XY(:,2), 'r');
alpha(f, 0.75)
%Find people
pd = vision.PeopleDetector('ClassificationThreshold', 1, ...
    'ClassificationModel','UprightPeople_96x48');
[bboxes, scores] = pd(Io);
index = find(scores == max(scores));
Ia = insertObjectAnnotation(In,'rectangle',bboxes(index,:),scores(index));
figure 
imshow(Ia)
hold
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',5,'Color','blue');
   XYh = [[xy(:,1)-[50;50]; xy(:,1)-[50;50]],[xy(:,2); xy(:,2)]];
   XYh = sortrows(XYh);
   %h = fill(XYh(:,1), XYh(:,2), 'b');
end
%f = fill(XY(:,1), XY(:,2), 'r');
alpha(f, 0.75)
%% Birds eye view
focalLength    = [800,800];[609.4362, 644.2161]; % [fx, fy] in pixel units
principalPoint = [360, 640]; % [cx, cy] optical center in pixel coordinates
imageSize      = size(I);           % [nrows, mcols]
camIntrinsics = cameraIntrinsics(focalLength, principalPoint, imageSize);

height = 1;    % mounting height in meters from the ground
pitch  = 30;        % pitch of the camera in degrees

sensor = monoCamera(camIntrinsics, height, 'Pitch', pitch, 'Roll', 10, 'Yaw', 25);

% Using vehicle coordinates, define area to transform
distAheadOfSensor = 10; % in meters, as previously specified in monoCamera height input
spaceToOneSide    = 2;  % all other distance quantities are also in meters
bottomOffset      = 2;

outView   = [bottomOffset, distAheadOfSensor, -spaceToOneSide, spaceToOneSide]; % [xmin, xmax, ymin, ymax]
imageSize = [NaN, 250]; % output image width in pixels; height is chosen automatically to preserve units per pixel ratio

birdsEyeConfig = birdsEyeView(sensor, outView, imageSize);

birdsEyeImage = transformImage(birdsEyeConfig, I);
figure
imshow(birdsEyeImage)

%% Rail detection

%birdsEyeImage = rgb2gray(birdsEyeImage);
outView   = [bottomOffset, distAheadOfSensor, -spaceToOneSide, spaceToOneSide]; % [xmin, xmax, ymin, ymax]
birdsEyeConfig = birdsEyeView(sensor, outView, imageSize);

% Lane marker segmentation ROI in world units
vehicleROI = outView - [-1, 0, -1, 1]; % look 3 meters to left and right, and 4 meters ahead of the sensor
approxLaneMarkerWidthVehicle = 0.25; % 25 centimeters

% Detect lane features
laneSensitivity = 0.25;
birdsEyeViewBW = segmentLaneMarkerRidge(birdsEyeImage, birdsEyeConfig, approxLaneMarkerWidthVehicle,...
'ROI', vehicleROI, 'Sensitivity', laneSensitivity);
%%
% Obtain lane candidate points in vehicle coordinates
[imageX, imageY] = find(birdsEyeViewBW);
xyBoundaryPoints = imageToVehicle(birdsEyeConfig, [imageY, imageX]);
maxLanes      = 2; % look for maximum of two lane markers
boundaryWidth = 3*approxLaneMarkerWidthVehicle; % expand boundary width

[boundaries, boundaryPoints] = findParabolicLaneBoundaries(xyBoundaryPoints,boundaryWidth, ...
    'MaxNumBoundaries', maxLanes, 'validateBoundaryFcn', @validateBoundaryFcn);

figure
imshow(birdsEyeViewBW)