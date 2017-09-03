function multiMatPlot(input)

global GLB_INVP;

cutoff = 0.3;
scale_factor = 1;

% Define colors
red = [1 0 0];
green = [0 1 0];
colors = [red; green];
% Plot data
figure;
axis off; hold on;
set(gca,'dataAspectRatio', [1 1 1]);
zlim([scale_factor*cutoff scale_factor]);
az = 0;
el = 90;
view(az,el);
light;
for index = 1:GLB_INVP.num_materials
    begin_index = ((index-1)*GLB_INVP.nVertGrid) + 1;
    end_index = index*GLB_INVP.nVertGrid;
    data = scale_factor.*input(begin_index:end_index)';
    trisurf(GLB_INVP.mesh.t,GLB_INVP.mesh.p(:,1),GLB_INVP.mesh.p(:,2),...
            data,'FaceColor',colors(index,:),'EdgeColor','none'); 
end

end