function [dirichlet_bdry_nodes,dirichlet_bdry_dof, u_dirichlet] = getDirichletBdry(mesh)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cubit side id for quadratical elements:
%   1. top = 1
%   2. right = 2
%   3. bottom = 3
%   4. left = 4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if size(mesh.p,2) == 2      % in 2D, the bottom side is labeled 3
  bottom = 3;
elseif size(mesh.p,2) == 3  % in 3D, the bottom side is labeled 2
  bottom = 2;
end

% Get Dirichlet nodes
spaceDim  = size(mesh.p, 2);
nVertGrid = max( max(mesh.t) ); 
dirichlet_bdry_nodes = mesh.sidenodes_ss{1};
dirichlet_bdry_nodes = unique(dirichlet_bdry_nodes);

% Compute Dirichlet degrees of freedom
dirichlet_dof = [];
for i=1:spaceDim
  dirichlet_dof = [dirichlet_dof; (spaceDim*dirichlet_bdry_nodes) - (spaceDim-i)];
end

% Set prescribed Dirichlet boundary conditions
dirichlet_bdry_dof = sort(dirichlet_dof);
u_dirichlet = zeros(spaceDim*nVertGrid,1);
u_dirichlet(dirichlet_bdry_dof) = 0;

end
