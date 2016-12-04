function [force] = generateFlux(struc)
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

numCells = struc.numCells;
numFields = struc.numFields;

%%%%%%%%%%% build right hand side
flux = zeros(numFields,numCells);
flux(:,struc.NeumannCellsBS) = 1e0;

%%%%%%%%%%% integrate right hand side
cell_forces = zeros(numFields, numCells);
intrepid_integrate(cell_forces, flux, ...
    struc.weighted_transformed_val_at_cub_points_side0_refcell,'COMP_BLAS');

%%%% build global rhs vector
force_vector = reshape(cell_forces, 1, numel(cell_forces));
sparse_force = sparse(struc.iVecIdx, struc.iVecIdx, force_vector);
force = spdiags(sparse_force,0);

end
