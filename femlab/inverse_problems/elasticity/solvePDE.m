function [state,K] = solvePDE(struc,shear_modulus,bulk_modulus)
%
%   solvePDE(struc,shear_modulus,bulk_modulus)
%
%   PURPOSE: Solve the following PDE using FEM:
%
%     - div( dev_sigma + vol_sigma ) = f      in Omega
%                                  u = u_D    on Gamma_D
%         (dev_sigma + vol_sigma)'*n = g      on Gamma_N
%                          dev_sigma = 2mu*( grad(u) -1/3 tr(grad(u)I )
%                          vol_sigma = kappa*tr(grad(u))I
%
%   The problem domain Omega is the square (xmin,xmax)x(ymin,ymax).
%
%   Input:
%           struc    contains all input parameters, as well as additional
%                    computed quantities
%
%   Output:
%
%           state    FEM solution
%
%   AUTHOR:  Miguel Aguilo
%            Dookie's Coorp.
%            August 15, 2011

spaceDim      = struc.spaceDim;
nVertGrid     = struc.nVertGrid;
numDof        = struc.numDof;
numCubPoints  = struc.numCubPoints;
numCells      = struc.numCells;

%%%%%%%%%%% Initialization of lhs (i.e. solution) vector.
state = zeros(spaceDim*nVertGrid,1);

%%%%%%%%%%% evaluate shear modulus at the cubature points
shear_at_dof = shear_modulus( struc.mesh.t');
shear_at_cub_points = zeros(numCubPoints, numCells);
intrepid_evaluate(shear_at_cub_points, shear_at_dof, ...
    struc.transformed_val_at_cub_points);

%%%%%%%%%%% evaluate bulk modulus at the cubature points
bulk_at_dof = bulk_modulus( struc.mesh.t');
bulk_at_cub_points = zeros(numCubPoints, numCells);
intrepid_evaluate(bulk_at_cub_points, bulk_at_dof, ...
    struc.transformed_val_at_cub_points);

%%%%%%%%%%% combine Bmat with shear modulus
shear_times_Bmat = zeros(3, numCubPoints, numDof, numCells);
intrepid_scalarMultiplyDataField( shear_times_Bmat, ...
    shear_at_cub_points, struc.Bmat);

%%%%%%%%%%% matrix-vector product, deviatoric coefficient mat. and fields
Ddev_times_shear_times_Bmat = zeros(3, numCubPoints, numDof, numCells);
intrepid_tensorMultiplyDataField( Ddev_times_shear_times_Bmat, ...
    struc.Ddev, shear_times_Bmat);

%%%%%%%%%%% integrate deviatoric stiffnes matrix
cell_Kdev_matrices = zeros(numDof, numDof, numCells);
intrepid_integrate(cell_Kdev_matrices, ...
    Ddev_times_shear_times_Bmat, struc.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% build global deviatoric stiffness matrix
Kdev_matrices = reshape(cell_Kdev_matrices, 1, numel(cell_Kdev_matrices));
Kdev = sparse(struc.iIdxDof, struc.jIdxDof, Kdev_matrices);

%%%%%%%%%%% combine Bmat with bulk modulus
bulk_times_Bmat = zeros(3, numCubPoints, numDof, numCells);
intrepid_scalarMultiplyDataField( bulk_times_Bmat, ...
    bulk_at_cub_points, struc.Bmat);

%%%%%%%%%%% matrix-vector product, volumetric coefficient mat. and fields
Dvol_times_bulk_times_Bmat = zeros(3, numCubPoints, numDof, numCells);
intrepid_tensorMultiplyDataField( Dvol_times_bulk_times_Bmat, ...
    struc.Dvol, bulk_times_Bmat);

%%%%%%%%%%% integrate volumetric stiffnes matrix
cell_Kvol_matrices = zeros(numDof, numDof, numCells);
intrepid_integrate(cell_Kvol_matrices, ...
    Dvol_times_bulk_times_Bmat, struc.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% build global volumetric stiffness matrix
Kvol_matrices = reshape(cell_Kvol_matrices, 1, numel(cell_Kvol_matrices));
Kvol = sparse(struc.iIdxDof, struc.jIdxDof, Kvol_matrices);

%%%%%%%%%%% compute stiffness matrix
K = Kdev + Kvol;

%%%%%%%%%%% Apply Dirichlet conditions to rhs vector.
if( ~isempty(state) )
    state(unique(struc.dirichlet)) = struc.u_dirichlet( ...
        unique(struc.dirichlet) );
    force = struc.force - K * state;
end

%%%%%%%%%%% solve system of equations
state(struc.FreeNodes) = K(struc.FreeNodes,struc.FreeNodes) ...
    \ force(struc.FreeNodes);

end
