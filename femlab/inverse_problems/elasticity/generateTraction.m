function [struc] = generateTraction(struc)
%
%  generateTraction(usr_par)
%
%  PURPOSE: compute Neumann boundary data (i.e. surface tractions)
%
%  Input:
%           struc    contains all input parameters, as well as additional
%                    computed quantities
%
%  Output:
%           struc    contains all input parameters, as well as additional
%                    computed quantities
%
%  AUTHOR:  Miguel Aguilo
%           Dookie's Coorp.
%           August 17, 2011

spaceDim = size(struc.cubPointsSide1PhysCoord,1);
numCubPointsSide1 = size(struc.cubPointsSide1PhysCoord,2);
numCells = size(struc.cubPointsSide1PhysCoord,3);
numDof = struc.numDof;

%%%%%%%%%%% build right hand side
forces = zeros(spaceDim, numCubPointsSide1, numCells);
forces(1,:,struc.NeumannCellsUS) = 0;
forces(2,:,struc.NeumannCellsUS) = -1e0;

%%%%%%%%%%% integrate right hand side
cell_forces = zeros(numDof, numCells);
intrepid_integrate(cell_forces, forces, struc.US_wNmat, 'COMP_BLAS');

%%%%%%%%%%% store force vector
reshape_force = reshape(cell_forces, 1, numel(cell_forces));
sparse_force = sparse(struc.iVecIdxDof, struc.iVecIdxDof, reshape_force);
struc.force = spdiags(sparse_force,0);

%%%%%%%%%%%% Generate dirichlet boundary data
struc.u_dirichlet = zeros(spaceDim*size(struc.mesh.p,1),1);

end
