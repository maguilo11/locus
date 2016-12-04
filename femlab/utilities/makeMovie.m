close all
clear all;
load control;
load mesh;
% Open movie object to export movie
writerObj = VideoWriter('topo.avi');
open(writerObj);
% Set golden ratios to generate high-quality images
height=1.0/1.618;
width=2;
scale=300;
xpos=600;
ypos=600;
% Open figure object
%figure('Renderer','zbuffer');
figure;
trisurf(mesh.t,mesh.p(:,1),mesh.p(:,2),control(:,1),'facecolor','interp');
% Set figure format
axis tight;
set(gca,'NextPlot','replaceChildren');
view(0,90);
shading interp;
set(gcf,'color','white');
set(gcf,'Position',[xpos ypos scale*width scale*height])
for j = 1:size(control,2)
    trisurf(mesh.t,mesh.p(:,1),mesh.p(:,2),control(:,j),'facecolor','interp');
    axis tight;
    view(0,90);
    shading interp;
    set(gcf,'color','white');
    set(gcf,'Position',[xpos ypos scale*width scale*height]) % Set figure format
    frame = getframe;
    writeVideo(writerObj,frame);
    data(j) = frame;
end
% Close movie object 
close(writerObj);
% Play the movie
movie(data,size(control,2))