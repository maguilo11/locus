function [state,K] = solvePDE(struc,control,force)
%
%   solvePDE(usr_par,k,f)
%
%   PURPOSE: Solve the following PDE using FEM:
%       
%       - div(k grad(u))  = f              in Omega
%                       u = u_D            on Gamma_D
%          (k grad(u))'*n = g              on Gamma_N
%
%   The problem domain Omega is the square (xmin,xmax)x(ymin,ymax).
%
%   u     - state
%   k     - material parameter
%   f     - source term
%
%   Input:
%           struc      struct with all input parameters and additional
%                      information for analysis
%  
%   Output:
%           state      FEM solution
%
%   AUTHOR:  Miguel Aguilo
%            Dookie's Corp.
%

spaceDim      = struc.spaceDim;
nVertGrid     = struc.nVertGrid;
numFields     = struc.numFields;
numCubPoints  = struc.numCubPoints;
numCells      = struc.numCells;

%%%%%%%%%%% Initialization of lhs (i.e. solution) vector.
state = zeros(nVertGrid,1);

% evaluate material parameter k (diffusion coefficients) at the cubature points
control_at_dof = control( struc.mesh.t');
control_at_cub_points = zeros(numCubPoints, numCells);
intrepid_evaluate(control_at_cub_points, control_at_dof, ...
    struc.transformed_val_at_cub_points);

%%%%%%%%%%% combine transformed gradients with diffusion parameters
control_times_transformed_grad_at_cub_points = zeros(spaceDim, ...
    numCubPoints, numFields, numCells);
intrepid_scalarMultiplyDataField(control_times_transformed_grad_at_cub_points, ...
    control_at_cub_points, struc.transformed_grad_at_cub_points);

%%%%%%%%%%% integrate stiffnes matrix
cell_stiffness_matrices = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_stiffness_matrices, ...
    control_times_transformed_grad_at_cub_points, ...
    struc.weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global stiffness matrix
cell_stiffness_matrices = reshape(cell_stiffness_matrices, 1, ...
    numel(cell_stiffness_matrices));
K = sparse(struc.iIdx, struc.jIdx, cell_stiffness_matrices);

%%%%%%%%%%% Apply Dirichlet conditions to rhs vector. 
if( ~isempty(state) )
    state(unique(struc.dirichlet)) = struc.u_dirichlet( ...
        unique(struc.dirichlet) );
    rhs = force - K * state;
end

state(struc.FreeNodes) = ...
    K(struc.FreeNodes,struc.FreeNodes) \ rhs(struc.FreeNodes);

end
