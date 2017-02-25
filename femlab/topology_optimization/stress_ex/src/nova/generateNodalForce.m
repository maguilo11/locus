function [struc] = generateNodalForce(struc)
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
force = zeros(struc.spaceDim*struc.nVertGrid, 1);
nodeset = struc.mesh.nodeset{1};
dof = nodeset*struc.spaceDim;
force(dof) = -1e1;
struc.force = force;

end