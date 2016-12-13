function plotResults(Output,StateData)

global GLB_INVP;

set(0, 'defaultaxesfontsize',16,'defaultaxesfontweight','normal',...
    'defaultaxeslinewidth',1.0,...
    'defaultlinelinewidth',1.0,'defaultpatchlinewidth',1.0,...
    'defaulttextfontsize',16,'defaulttextfontweight','normal');

show(GLB_INVP.mesh.t, GLB_INVP.mesh.p, Output.Control);
control_color_range = caxis;
control_color_range(1) = control_color_range(1) - ...
    0.5*abs(control_color_range(1));
control_color_range(2) = control_color_range(2) + ...
    0.0*abs(control_color_range(2));
caxis(control_color_range); % set colormap limits
xlabel('x');
ylabel('y');
colorbar('vert')
title('Optimal Topology')

end