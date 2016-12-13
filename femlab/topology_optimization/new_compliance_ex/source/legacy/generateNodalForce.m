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
force = zeros(usr_par.spaceDim*usr_par.nVertGrid, 1);
% **** Load Down ****
node = (usr_par.DomainSpecs.nx+1)*(usr_par.DomainSpecs.ny+1) - ...
    (usr_par.DomainSpecs.ny);
dof = usr_par.spaceDim*node;
force(dof) = -1e0;
% **** Load Up ****
% node = (usr_par.DomainSpecs.nx+1)*(usr_par.DomainSpecs.ny+1);
% dof = usr_par.spaceDim*node;
% force(dof) = 1e0;
% **** Load Middle ****
% node = (usr_par.DomainSpecs.nx+1)*(usr_par.DomainSpecs.ny+1) - ...
%     ((usr_par.DomainSpecs.ny) / 2);
% dof = usr_par.spaceDim*node;
% force(dof) = -1e0;
% node = (usr_par.DomainSpecs.nx+1)*(usr_par.DomainSpecs.ny+1) - ...
%     ((usr_par.DomainSpecs.ny) / 2) + 1;
% dof = usr_par.spaceDim*node;
% force(dof) = -1e0;
% node = (usr_par.DomainSpecs.nx+1)*(usr_par.DomainSpecs.ny+1) - ...
%     ((usr_par.DomainSpecs.ny) / 2) + 2;
% dof = usr_par.spaceDim*node;
% force(dof) = -1e0;
% **** Load Mitchell Bridge ****
% node = usr_par.DomainSpecs.ny+1;
% dof = usr_par.spaceDim*node;
% force(dof) = -1e0;
% **** SET LOAD ****
usr_par.force = force;

%%%%%%%%%%%% Generate dirichlet boundary data
usr_par.u_dirichlet = zeros(usr_par.spaceDim*size(usr_par.mesh.p,1),1);