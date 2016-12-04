function [struc] = generateProbSpecificData(struc)
%
%  generateProbSpecificData(usr_par)
%
%  PURPOSE: Generates additional problem-specific constant quantities,
%           which require some computational work, but can be computed once
%           and reused, thus speeding up the overall computation.
%
%  Input:
%           usr_par    contains all input parameters, as well as additional
%                      computed quantities
%
%  Output:
%
%           usr_par    contains all previous input parameters, as well as
%                      additional computed quantities; however, additional
%                      problem-specific constant quantities were added
%
%  AUTHOR:  Miguel Aguilo
%           Sandia National Laboratories
%           August 11, 2011

spaceDim = struc.spaceDim;
cellType = struc.mesh.cellType;
numCellNodes = struc.numFields;
numCellDof = struc.numDof;
nVertGrid = struc.nVertGrid;
numCubPoints  = struc.numCubPoints;
numCubPointsSide = struc.numCubPointsSide;
numCells = struc.numCells;
numStress = struc.numStress;

%%%%%%%%%%%%%%%%% Process Dirichlet boundary conditions
[dirichlet_nodes, dirichlet_dof, u_dirichlet] = getDirichletBdry(struc.mesh);

%%%%%%%%%%%%%%%%% Initialization of free dof and node arrays
FreeDof = setdiff(1:spaceDim*nVertGrid , dirichlet_dof);
FreeNodes = setdiff(1:nVertGrid, unique( dirichlet_nodes ));

%%%%%%%%%%% evaluate cell Jacobians
cellJacobians  = zeros(spaceDim, spaceDim, numCubPoints, numCells);
intrepid_setJacobian(cellJacobians, struc.cubPoints, ...
    struc.cellNodes, cellType);
struc.cellJacobians = cellJacobians;

%%%%%%%%%%% evaluate inverses of cell Jacobians
cellJacobianInvs = zeros(spaceDim, spaceDim, numCubPoints, numCells);
intrepid_setJacobianInv(cellJacobianInvs, cellJacobians);
struc.cellJacobianInvs = cellJacobianInvs;

%%%%%%%%%%% evaluate determinants of cell Jacobians
cellJacobianDets  = zeros(numCubPoints, numCells);
intrepid_setJacobianDet(cellJacobianDets, cellJacobians);
struc.cellJacobianDets = cellJacobianDets;

%%%%%%%%%%% evaluate basis (value, gradient)
val_at_cub_points = zeros(numCubPoints, numCellNodes);
grad_at_cub_points = zeros(spaceDim, numCubPoints, numCellNodes);
intrepid_getBasisValues(val_at_cub_points, struc.cubPoints, ...
    'OPERATOR_VALUE', cellType, 1);
intrepid_getBasisValues(grad_at_cub_points, struc.cubPoints, ...
    'OPERATOR_GRAD', cellType, 1);
struc.val_at_cub_points = val_at_cub_points;
struc.grad_at_cub_points = grad_at_cub_points;

%%%%%%%%%%% compute cell measures
weighted_measure = zeros(numCubPoints, numCells);
intrepid_computeCellMeasure(weighted_measure, ...
    cellJacobianDets, struc.cubWeights);
struc.weighted_measure = weighted_measure;

%%%%%%%%%%% transform gradients
transformed_grad_at_cub_points = zeros(spaceDim, numCubPoints, ...
    numCellNodes, numCells);
intrepid_HGRADtransformGRAD(transformed_grad_at_cub_points, ...
    cellJacobianInvs, grad_at_cub_points);
struc.transformed_grad_at_cub_points = transformed_grad_at_cub_points;

%%%%%%%%%%% transform values
transformed_val_at_cub_points = zeros(numCubPoints, numCellNodes, numCells);
intrepid_HGRADtransformVALUE(transformed_val_at_cub_points, ...
    val_at_cub_points);
struc.transformed_val_at_cub_points = transformed_val_at_cub_points;

%%%%%%%%%%% combine transformed gradients with measures
weighted_transformed_grad_at_cub_points = zeros(spaceDim, ...
    numCubPoints, numCellNodes, numCells);
intrepid_multiplyMeasure(weighted_transformed_grad_at_cub_points, ...
    weighted_measure, transformed_grad_at_cub_points);
struc.weighted_transformed_grad_at_cub_points = ...
    weighted_transformed_grad_at_cub_points;

%%%%%%%%%%% combine transformed values with measures
weighted_transformed_val_at_cub_points = zeros(numCubPoints, ...
    numCellNodes, numCells);
intrepid_multiplyMeasure(weighted_transformed_val_at_cub_points, ...
    weighted_measure, transformed_val_at_cub_points);
struc.weighted_transformed_val_at_cub_points = ...
    weighted_transformed_val_at_cub_points;

%%%%%%%%%%% integrate scalar field stiffness matrix
cell_stiffness_matrices = zeros(numCellNodes, numCellNodes, numCells);
intrepid_integrate(cell_stiffness_matrices, ...
    transformed_grad_at_cub_points, ...
    weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global scalar field stiffness matrix
cell_stiffness_matrices = reshape(cell_stiffness_matrices, 1, ...
    numel(cell_stiffness_matrices));
stiff_mat = sparse(struc.iIdxVertices, ...
    struc.jIdxVertices, cell_stiffness_matrices);

%%%%%%%%%%% integrate scalar field mass matrix
cell_mass_matrices = zeros(numCellNodes, numCellNodes, numCells);
intrepid_integrate(cell_mass_matrices, ...
    transformed_val_at_cub_points, ...
    weighted_transformed_val_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global scalar field mass matrix
one = ones(numCellNodes,1);
elem_mass_volume = zeros(numCells,1);
for i=1:numCells
    elem_mass_volume(i) = sum(cell_mass_matrices(:,:,i) * one);
end
cell_mass_matrices_reshape = reshape(cell_mass_matrices, 1, ...
    numel(cell_mass_matrices));
mass_mat = sparse(struc.iIdxVertices, ...
    struc.jIdxVertices, cell_mass_matrices_reshape);

%%%%%%%%%%% get cell deviatoric matrices
cell_deviatoric_matrices = zeros(numStress,numStress, numCubPoints, numCells);
intrepid_getCellDeviatoricMat(cell_deviatoric_matrices, spaceDim);

%%%%%%%%%%% get cell volumetric matrices
cell_volumetric_matrices = zeros(numStress,numStress, numCubPoints, numCells);
intrepid_getCellVolumetricMat(cell_volumetric_matrices, spaceDim);

%%%%%%%%%%% get cell N matrices
cell_N_matrices = zeros(spaceDim,numCubPoints, numCellDof, numCells);
cell_N_matrices(1,:,1:spaceDim:end,:) = ...
    transformed_val_at_cub_points(:,:,:);
cell_N_matrices(2,:,2:spaceDim:end,:) = ...
    transformed_val_at_cub_points(:,:,:);
struc.Nmat = cell_N_matrices;

%%%%%%%%%%% get cell weighted N matrices
cell_weighted_N_matrices = zeros(spaceDim,numCubPoints, numCellDof, numCells);
cell_weighted_N_matrices(1,:,1:spaceDim:end,:) = ...
    weighted_transformed_val_at_cub_points(:,:,:);
cell_weighted_N_matrices(2,:,2:spaceDim:end,:) = ...
    weighted_transformed_val_at_cub_points(:,:,:);
struc.weighted_Nmat = cell_weighted_N_matrices;

%%%%%%%%%%% get cell B matrices
cell_B_matrices = zeros(numStress, numCubPoints, numCellDof, numCells);
cell_B_matrices(1,:,1:spaceDim:end, :) = ...
    transformed_grad_at_cub_points(1,:,:,:);
cell_B_matrices(2,:,2:spaceDim:end, :) = ...
    transformed_grad_at_cub_points(2,:,:,:);
cell_B_matrices(3,:,1:spaceDim:end, :) = ...
    transformed_grad_at_cub_points(2,:,:,:);
cell_B_matrices(3,:,2:spaceDim:end, :) = ...
    transformed_grad_at_cub_points(1,:,:,:);
struc.Bmat = cell_B_matrices;

%%%%%%%%%%% get cell weighted B matrices
cell_weighted_B_matrices = zeros(numStress, numCubPoints, numCellDof, numCells);
cell_weighted_B_matrices(1,:,1:spaceDim:end, :) = ...
    weighted_transformed_grad_at_cub_points(1,:,:,:);
cell_weighted_B_matrices(2,:,2:spaceDim:end, :) = ...
    weighted_transformed_grad_at_cub_points(2,:,:,:);
cell_weighted_B_matrices(3,:,1:spaceDim:end, :) = ...
    weighted_transformed_grad_at_cub_points(2,:,:,:);
cell_weighted_B_matrices(3,:,2:spaceDim:end, :) = ...
    weighted_transformed_grad_at_cub_points(1,:,:,:);
struc.weighted_Bmat = cell_weighted_B_matrices;

%%%%%%%%%%% integrate naked state mass matrix
cell_state_mass_matrices = zeros(numCellDof, numCellDof, numCells);
intrepid_integrate(cell_state_mass_matrices, cell_N_matrices, ...
    cell_weighted_N_matrices, 'COMP_BLAS');

%%%%%%%%%%% build global naked state mass matrix
cell_state_mass_matrices = reshape(cell_state_mass_matrices, 1, ...
    numel(cell_state_mass_matrices));
state_mass_mat = sparse(struc.iIdxDof, ...
    struc.jIdxDof, cell_state_mass_matrices);

%%%%%%%%%%% get computational mesh cubature points physical frame
cubPointsPhysCoord = zeros(spaceDim, numCubPoints, numCells);
intrepid_mapToPhysicalFrame(cubPointsPhysCoord, struc.cubPoints, ...
    struc.cellNodes, cellType);

%%%%%%%%%%% get computational mesh side0 cubature points physical frame
cubPointsSide0PhysCoord = zeros(spaceDim, numCubPointsSide, numCells);
intrepid_mapToPhysicalFrame(cubPointsSide0PhysCoord, ...
    struc.cubPointsSide0Ref, struc.cellNodes, cellType);

%%%%%%%%%%% get computational mesh side1 cubature points physical frame
cubPointsSide1PhysCoord = zeros(spaceDim, numCubPointsSide, numCells);
intrepid_mapToPhysicalFrame(cubPointsSide1PhysCoord, ...
    struc.cubPointsSide1Ref, struc.cellNodes, cellType);

%%%%%%%%%%% get computational mesh side2 cubature points physical frame
cubPointsSide2PhysCoord = zeros(spaceDim, numCubPointsSide, numCells);
intrepid_mapToPhysicalFrame(cubPointsSide2PhysCoord, ...
    struc.cubPointsSide2Ref, struc.cellNodes, cellType);

struc.Ss = stiff_mat;
struc.Ms = mass_mat;
struc.M = state_mass_mat;
struc.Ddev = cell_deviatoric_matrices;
struc.Dvol  = cell_volumetric_matrices;
struc.u_dirichlet = u_dirichlet;
struc.FreeDof = FreeDof;
struc.FreeNodes = FreeNodes;
struc.ElemVolume  = elem_mass_volume;
struc.CellMassMatrices = cell_mass_matrices;
struc.dirichlet_dof = dirichlet_dof;

%%%%%%%%%%%%%%%%% process Neumann conditions
if(struc.neumann)
    [struc] = getNeumannBdry(struc);
end

end
