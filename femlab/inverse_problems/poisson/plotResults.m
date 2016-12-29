function plotResults(Output,StateData)

global GLB_INVP;

set(0, 'defaultaxesfontsize',16,'defaultaxesfontweight','normal',...
    'defaultaxeslinewidth',1.0,...
    'defaultlinelinewidth',1.0,'defaultpatchlinewidth',1.0,...
    'defaulttextfontsize',16,'defaulttextfontweight','normal');

%%%% Control Goal 
show(GLB_INVP.mesh_fine.t, GLB_INVP.mesh_fine.p, GLB_INVP.kappa_fine);
control_color_range = caxis;
control_color_range(1) = control_color_range(1) - ...
    0.5*abs(control_color_range(1));
control_color_range(2) = control_color_range(2) + ...
    0.0*abs(control_color_range(2));
caxis(control_color_range); % set colormap limits
xlabel('x');
ylabel('y');
color('white');
colorbar('vert')
title('Target Control')

%%%% Control Results 
[~,~,IntrControl] = griddata(GLB_INVP.mesh.p(:,1),GLB_INVP.mesh.p(:,2),...
    Output.Control,GLB_INVP.mesh_fine.p(:,1),GLB_INVP.mesh_fine.p(:,2),'cubic');
show(GLB_INVP.mesh_fine.t, GLB_INVP.mesh_fine.p, IntrControl);
caxis(control_color_range); % set colormap limits
xlabel('x');
ylabel('y');
color('white');
colorbar('vert')
title('Optimal Control')
error = norm(IntrControl - GLB_INVP.kappa_fine) / norm(GLB_INVP.kappa_fine);
fprintf('Unfiltered Control l2 Error=%f\n',error);
error = ((IntrControl - GLB_INVP.kappa_fine)' * GLB_INVP.Mf * ...
    (IntrControl - GLB_INVP.kappa_fine))^0.5 / ...
    (GLB_INVP.kappa_fine' * GLB_INVP.Mf * GLB_INVP.kappa_fine)^(0.5);
fprintf('Unfiltered Control L2 Error=%f\n',error);

%%%% Control Results 
factor = 1;
%factor = 0.85;
[~,~,IntrControl] = griddata(GLB_INVP.mesh.p(:,1),GLB_INVP.mesh.p(:,2),...
    Output.Control,GLB_INVP.mesh_fine.p(:,1),GLB_INVP.mesh_fine.p(:,2),'cubic');
show(GLB_INVP.mesh_fine.t, GLB_INVP.mesh_fine.p, factor.*IntrControl);
caxis(control_color_range); % set colormap limits
xlabel('x');
ylabel('y');
color('white');
colorbar('vert')
title('Optimal Control')

%%%% Filtered Control 
alpha = 1/1e4;
%alpha = 1/2e3;
index = find(Output.Control > 0.13);
Output.Control(index) = factor * Output.Control(index);
force = GLB_INVP.M * Output.Control';
Kf = (alpha*GLB_INVP.S + GLB_INVP.M);
FilteredControl = Kf \ (force);
[~,~,IntrFilteredControl] = griddata(GLB_INVP.mesh.p(:,1),GLB_INVP.mesh.p(:,2),...
    FilteredControl,GLB_INVP.mesh_fine.p(:,1),GLB_INVP.mesh_fine.p(:,2),'cubic');
show(GLB_INVP.mesh_fine.t, GLB_INVP.mesh_fine.p, IntrFilteredControl);
caxis(control_color_range); % set colormap limits
xlabel('x');
ylabel('y');
color('white');
colorbar('vert')
title('Filtered Control')
% Error
error = norm(IntrFilteredControl - GLB_INVP.kappa_fine) / ...
    norm(GLB_INVP.kappa_fine);
fprintf('Filtered Control l2 Error=%f\n',error);
error = ((IntrFilteredControl - GLB_INVP.kappa_fine)' * GLB_INVP.Mf * ...
    (IntrFilteredControl - GLB_INVP.kappa_fine))^0.5 / ...
    (GLB_INVP.kappa_fine' * GLB_INVP.Mf * GLB_INVP.kappa_fine)^(0.5);
fprintf('Filtered Control L2 Error=%f\n',error);

%%%% State Results 
show(GLB_INVP.mesh.t, GLB_INVP.mesh.p, StateData.FinalState);
control_color_range = caxis;
control_color_range(1) = control_color_range(1) - ...
    0.5*abs(control_color_range(1));
control_color_range(2) = control_color_range(2) + ...
    0.0*abs(control_color_range(2));
caxis(control_color_range); % set colormap limits
xlabel('x');
ylabel('y');
color('white');
colorbar('vert')
title('Optimal State')

end