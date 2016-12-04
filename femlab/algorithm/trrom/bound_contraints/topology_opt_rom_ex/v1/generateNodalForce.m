function [usr_par] = generateNodalForce(usr_par)
%
%  generateTraction(usr_par)
%
%  PURPOSE: compute Neumann boundary data (i.e. surface tractions)
%
%  Input:
%           usr_par    contains all input parameters, as well as additional
%                      computed quantities
%
%  Output:
%           usr_par    contains all input parameters, as well as additional
%                      computed quantities
%
%  AUTHOR:  Miguel Aguilo
%           Sandia National Laboratories
%           August 17, 2011

%%%%%%%%%%% build right hand side
%angles = [90; 80; 70; 100; 110; 120];
angles = [90; 80; 50];
force = zeros(usr_par.spaceDim*usr_par.nVertGrid, size(angles,1));
for index=1:size(angles,1);
    % **** Load Down ****
    % node = (usr_par.DomainSpecs.nx+1)*(usr_par.DomainSpecs.ny+1) - ...
    %     (usr_par.DomainSpecs.ny);
    % dof = usr_par.spaceDim*node;
    % force(dof) = -1e0*sin(mydeg2rad(angle));
    % force(dof-1) = -1e0*cos(mydeg2rad(angle));
    % **** Load Up ****
    % node = (usr_par.DomainSpecs.nx+1)*(usr_par.DomainSpecs.ny+1);
    % dof = usr_par.spaceDim*node;
    % force(dof) = 1e0;
    % **** Load Middle ****
    node = (usr_par.DomainSpecs.nx+1)*(usr_par.DomainSpecs.ny+1) - ...
        ((usr_par.DomainSpecs.ny) / 2);
    dof = usr_par.spaceDim*node;
    force(dof,index) = -1e0*sin(mydeg2rad(angles(index)));
    force(dof-1,index) = -1e0*cos(mydeg2rad(angles(index)));
    node = (usr_par.DomainSpecs.nx+1)*(usr_par.DomainSpecs.ny+1) - ...
        ((usr_par.DomainSpecs.ny) / 2) + 1;
    dof = usr_par.spaceDim*node;
    force(dof,index) = -1e0*sin(mydeg2rad(angles(index)));
    force(dof-1,index) = -1e0*cos(mydeg2rad(angles(index)));
    node = (usr_par.DomainSpecs.nx+1)*(usr_par.DomainSpecs.ny+1) - ...
        ((usr_par.DomainSpecs.ny) / 2) - 1;
    dof = usr_par.spaceDim*node;
    force(dof,index) = -1e0*sin(mydeg2rad(angles(index)));
    force(dof-1,index) = -1e0*cos(mydeg2rad(angles(index)));
    % **** Load Mitchell Bridge ****
    % node = usr_par.DomainSpecs.ny+1;
    % dof = usr_par.spaceDim*node;
    % force(dof) = -1e0;
end
% **** SET LOAD ****
usr_par.force = force;
%%%%%%%%%%%% Generate dirichlet boundary data
usr_par.u_dirichlet = zeros(usr_par.spaceDim*size(usr_par.mesh.p,1),1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [value] = mydeg2rad(angle)
value = angle*pi()/180;
end