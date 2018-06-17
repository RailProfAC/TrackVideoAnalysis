clear all
close all
clc

filename = 'DriversDeskView3.png';
In = imread(filename);
tic
I = rgb2gray(In);
I = I(1:end, 400:800);
Io = I;
I = imgaussfilt(I,5);
%I(find(I > 150)) = 0;
I = 255-I;

E = edge(I, 'canny');

[H,T,R] = hough(E);%, 'Theta', -10:0.5:10);

P  = houghpeaks(H,2,'threshold',ceil(0.2*max(H(:))));
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
imshow(Io), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end

