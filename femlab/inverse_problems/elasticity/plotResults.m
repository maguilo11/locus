function plotResults(Output,StateData)

global GLB_INVP;

set(0, 'defaultaxesfontsize',16,'defaultaxesfontweight','normal',...
    'defaultaxeslinewidth',1.0,...
    'defaultlinelinewidth',1.0,'defaultpatchlinewidth',1.0,...
    'defaulttextfontsize',16,'defaulttextfontweight','normal');

%%%% Target Shear Goal 
show(GLB_INVP.mesh_fine.t, GLB_INVP.mesh_fine.p, GLB_INVP.mu_fine);
shear_color_range = caxis;
shear_color_range(1) = shear_color_range(1) - ...
    0.3*abs(shear_color_range(1));
shear_color_range(2) = shear_color_range(2) + ...
    0*abs(shear_color_range(2));
caxis(shear_color_range); % set colormap limits
xlabel('x');
ylabel('y');
color('white');
colorbar('vert')
title('Target Shear')

%%%% Target Shear Goal 
show(GLB_INVP.mesh_fine.t, GLB_INVP.mesh_fine.p, GLB_INVP.kappa_fine);
bulk_color_range = caxis;
bulk_color_range(1) = bulk_color_range(1) - ...
    0.3*abs(bulk_color_range(1));
bulk_color_range(2) = bulk_color_range(2) + ...
    0*abs(bulk_color_range(2));
caxis(bulk_color_range); % set colormap limits
xlabel('x');
ylabel('y');
color('white');
colorbar('vert')
title('Target Bulk')

%%%% Shear Modulus Results 
[~,~,IntrpShear] = griddata(GLB_INVP.mesh.p(:,1),GLB_INVP.mesh.p(:,2),...
    Output.Control(1:GLB_INVP.nVertGrid),GLB_INVP.mesh_fine.p(:,1),...
    GLB_INVP.mesh_fine.p(:,2),'cubic');
show(GLB_INVP.mesh_fine.t, GLB_INVP.mesh_fine.p, IntrpShear);
caxis(shear_color_range); % set colormap limits
xlabel('x');
ylabel('y');
colorbar('vert')
title('Optimal Shear Modulus')
% Error
error = 0.5*(norm(IntrpShear - GLB_INVP.mu_fine) / ...
    norm(GLB_INVP.mu_fine));
fprintf('Unfiltered Shear l2 Error=%f\n',error);
error = 0.5*(((IntrpShear - GLB_INVP.mu_fine)' * GLB_INVP.Mf * ...
    (IntrpShear - GLB_INVP.mu_fine))^0.5 / ...
    (GLB_INVP.mu_fine' * GLB_INVP.Mf * GLB_INVP.mu_fine)^(0.5));
fprintf('Unfiltered Shear L2 Error=%f\n',error);

%%%% Bulk Modulus Results 
[~,~,IntrpBulk] = griddata(GLB_INVP.mesh.p(:,1),GLB_INVP.mesh.p(:,2),...
    Output.Control(1+GLB_INVP.nVertGrid:end),GLB_INVP.mesh_fine.p(:,1),...
    GLB_INVP.mesh_fine.p(:,2),'cubic');
show(GLB_INVP.mesh_fine.t, GLB_INVP.mesh_fine.p, IntrpBulk);
caxis(bulk_color_range); % set colormap limits
xlabel('x');
ylabel('y');
colorbar('vert')
title('Optimal Bulk Modulus')
% Error
error = 0.5*(norm(IntrpBulk - GLB_INVP.kappa_fine) / ...
    norm(GLB_INVP.kappa_fine));
fprintf('Unfiltered Bulk l2 Error=%f\n',error);
error = 0.5*(((IntrpBulk - GLB_INVP.kappa_fine)' * GLB_INVP.Mf * ...
    (IntrpBulk - GLB_INVP.kappa_fine))^0.5 / ...
    (GLB_INVP.kappa_fine' * GLB_INVP.Mf * GLB_INVP.kappa_fine)^(0.5));
fprintf('Unfiltered Bulk L2 Error=%f\n',error);

%%%%%%%%%%%%%%%%% Filtered Shear and Bulk Modulus Results %%%%%%%%%%%%%%%%%
alpha = 1/1e4;
%index = find(Output.Control > 0.15);
%Output.Control(index) = factor * Output.Control(index);
force = [GLB_INVP.Ms * Output.Control(1:GLB_INVP.nVertGrid);
         GLB_INVP.Ms * Output.Control(1+GLB_INVP.nVertGrid:end)];
Kf = [(alpha*GLB_INVP.Ss + GLB_INVP.Ms) zeros(size(GLB_INVP.Ms));
      zeros(size(GLB_INVP.Ms)) (alpha*GLB_INVP.Ss + GLB_INVP.Ms)];
controlFiltered = Kf \ force;
% Shear
[~,~,IntrpFilteredShear] = griddata(GLB_INVP.mesh.p(:,1),GLB_INVP.mesh.p(:,2),...
    controlFiltered(1:GLB_INVP.nVertGrid),GLB_INVP.mesh_fine.p(:,1),...
    GLB_INVP.mesh_fine.p(:,2),'cubic');
show(GLB_INVP.mesh_fine.t, GLB_INVP.mesh_fine.p, IntrpFilteredShear);
caxis(shear_color_range); % set colormap limits
xlabel('x');
ylabel('y');
colorbar('vert')
title('Filtered Shear Modulus')
% Error
error = 0.5*(norm(IntrpFilteredShear - GLB_INVP.mu_fine) / ...
    norm(GLB_INVP.mu_fine));
fprintf('Filtered Shear l2 Error=%f\n',error);
error = 0.5*(((IntrpFilteredShear - GLB_INVP.mu_fine)' * GLB_INVP.Mf * ...
    (IntrpFilteredShear - GLB_INVP.mu_fine))^0.5 / ...
    (GLB_INVP.mu_fine' * GLB_INVP.Mf * GLB_INVP.mu_fine)^(0.5));
fprintf('Filtered Shear L2 Error=%f\n',error);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bulk Modulus %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
[~,~,IntrpFilteredBulk] = griddata(GLB_INVP.mesh.p(:,1),GLB_INVP.mesh.p(:,2),...
    controlFiltered(1+GLB_INVP.nVertGrid:end),GLB_INVP.mesh_fine.p(:,1),...
    GLB_INVP.mesh_fine.p(:,2),'cubic');
show(GLB_INVP.mesh_fine.t, GLB_INVP.mesh_fine.p, IntrpFilteredBulk);
caxis(bulk_color_range); % set colormap limits
xlabel('x');
ylabel('y');
colorbar('vert')
title('Filtered Bulk Modulus')
% Error
error = 0.5*(norm(IntrpFilteredBulk - GLB_INVP.kappa_fine) / ...
    norm(GLB_INVP.kappa_fine));
fprintf('Filtered Bulk l2 Error=%f\n',error);
error = 0.5*(((IntrpFilteredBulk - GLB_INVP.kappa_fine)' * GLB_INVP.Mf * ...
    (IntrpFilteredBulk - GLB_INVP.kappa_fine))^0.5 / ...
    (GLB_INVP.kappa_fine' * GLB_INVP.Mf * GLB_INVP.kappa_fine)^(0.5));
fprintf('Filtered Bulk L2 Error=%f\n',error);

%%%% State Results - X-Dir
show(GLB_INVP.mesh.t, GLB_INVP.mesh.p, StateData.FinalState(1:2:end));
control_color_range = caxis;
control_color_range(1) = control_color_range(1) - ...
    0.5*abs(control_color_range(1));
control_color_range(2) = control_color_range(2) + ...
    0.5*abs(control_color_range(2));
caxis(control_color_range); % set colormap limits
xlabel('x');
ylabel('y');
colorbar('vert')
title('Optimal State - X-Dir')

%%%% State Results - Y-Dir
show(GLB_INVP.mesh.t, GLB_INVP.mesh.p, StateData.FinalState(2:2:end));
control_color_range = caxis;
control_color_range(1) = control_color_range(1) - ...
    0.5*abs(control_color_range(1));
control_color_range(2) = control_color_range(2) + ...
    0.5*abs(control_color_range(2));
caxis(control_color_range); % set colormap limits
xlabel('x');
ylabel('y');
colorbar('vert')
title('Optimal State - Y-Dir')

end