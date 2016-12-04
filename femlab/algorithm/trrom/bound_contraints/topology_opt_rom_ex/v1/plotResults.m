function plotResults(control)

global GLB_INVP;

set(0, 'defaultaxesfontsize',16,'defaultaxesfontweight','normal',...
    'defaultaxeslinewidth',1.0,...
    'defaultlinelinewidth',1.0,'defaultpatchlinewidth',1.0,...
    'defaulttextfontsize',16,'defaulttextfontweight','normal');

show(GLB_INVP.mesh.t, GLB_INVP.mesh.p, control);
shading interp;
control_color_range = caxis;
control_color_range(1) = control_color_range(1) - ...
    0.5*abs(control_color_range(1));
control_color_range(2) = control_color_range(2) + ...
    0.0*abs(control_color_range(2));
caxis(control_color_range); % set colormap limits
colorbar('vert');
colormap('jet');
view(0,90);

end