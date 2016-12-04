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
%           Denis Ridzal
%           Sandia National Laboratories
%           August 11, 2011

spaceDim         = struc.spaceDim;
cellType         = struc.cellType;
numFields        = struc.numFields;
numDof           = struc.numDof;
nVertGrid        = struc.nVertGrid;
numCubPoints     = struc.numCubPoints;
numCubPointsSide = struc.numCubPointsSide;
numCells         = struc.numCells;
nx               = struc.nxint;
ny               = struc.nyint;

%%%%%%%%%%%% set boundary markers
%struc.mesh.e(1:ny,5) = 1;                 % edges on left boundary
struc.mesh.e(ny+1:nx+ny,5) = 2;           % edges on top boundary
struc.mesh.e(nx+2*ny+1:2*(nx+ny),5) = 1;  % edges on lower boundary

%%%%%%%%%%% Initialization of free nodes array.
dirichletNodes = unique ( [ struc.mesh.e(struc.mesh.e(:,5) == 1,2)' ...
                   struc.mesh.e(struc.mesh.e(:,5) == 1,3)' ] );
dirichlet = [spaceDim*dirichletNodes spaceDim*dirichletNodes-1];      
FreeNodes = setdiff(1:spaceDim*nVertGrid, unique( dirichlet ));

%%%%%%%%%%% Initialization of free nodes array.
NeumannNodes = unique ( [ struc.mesh.e(struc.mesh.e(:,5) == 2,2)' ...
                   struc.mesh.e(struc.mesh.e(:,5) == 2,3)' ] );
NeumannDof = [spaceDim*NeumannNodes-1; spaceDim*NeumannNodes];

%%%%%%%%%%% initialization of Neumann cells array
NeumannCellsLS = 2:2:2*ny;
NeumannCellsUS = 2*ny:2*ny:2*nx*ny;
NeumannCellsRS = ( 2*ny*(nx-1) ) + 1:2:2*nx*ny;
NeumannCellsBS = 1:2*ny:( 2*ny*(nx-1) ) + 1; 

%%%%%%%%%%% evaluate Jacobians at reference subcell (side0)
jacobiansSide0Ref  = zeros(spaceDim, spaceDim, numCubPointsSide, numCells);
intrepid_setJacobian(jacobiansSide0Ref, struc.cubPointsSide0Ref, ...
    struc.cellNodes, cellType);

%%%%%%%%%%% evaluate Jacobians at reference subcell (side1)
jacobiansSide1Ref  = zeros(spaceDim, spaceDim, numCubPointsSide, numCells);
intrepid_setJacobian(jacobiansSide1Ref, struc.cubPointsSide1Ref, ...
    struc.cellNodes, cellType);

%%%%%%%%%%% evaluate Jacobians at reference subcell (side2)
jacobiansSide2Ref  = zeros(spaceDim, spaceDim, numCubPointsSide, numCells);
intrepid_setJacobian(jacobiansSide2Ref, struc.cubPointsSide2Ref, ...
    struc.cellNodes, cellType);

%%%%%%%%%%% evaluate determinants of reference subcell (side0) Jacobians
jacobiansDetSide0Ref = zeros(numCubPointsSide, numCells);
intrepid_setJacobianDet(jacobiansDetSide0Ref, jacobiansSide0Ref);

%%%%%%%%%%% evaluate determinants of reference subcell (side1) Jacobians
jacobiansDetSide1Ref = zeros(numCubPointsSide, numCells);
intrepid_setJacobianDet(jacobiansDetSide1Ref, jacobiansSide1Ref);

%%%%%%%%%%% evaluate determinants of reference subcell (side2) Jacobians
jacobiansDetSide2Ref = zeros(numCubPointsSide, numCells);
intrepid_setJacobianDet(jacobiansDetSide2Ref, jacobiansSide2Ref);

%%%%%%%%%%% compute reference subcell (side0) measures
weighted_measure_side0_refcell = zeros(numCubPointsSide, numCells);
intrepid_computeEdgeMeasure(weighted_measure_side0_refcell, ...
    jacobiansSide0Ref, struc.cubWeightsSide, 0, cellType);

%%%%%%%%%%% compute reference subcell (side1) measures
weighted_measure_side1_refcell = zeros(numCubPointsSide, numCells);
intrepid_computeEdgeMeasure(weighted_measure_side1_refcell, ...
    jacobiansSide1Ref, struc.cubWeightsSide, 1, cellType);

%%%%%%%%%%% compute reference subcell (side2) measures
weighted_measure_side2_refcell = zeros(numCubPointsSide, numCells);
intrepid_computeEdgeMeasure(weighted_measure_side2_refcell, ...
    jacobiansSide2Ref, struc.cubWeightsSide, 2, cellType);

%%%%%%%%%%% evaluate reference subcell (side0) basis values  
val_at_cub_points_side0_refcell = zeros(numCubPointsSide, numFields);
intrepid_getBasisValues(val_at_cub_points_side0_refcell, ...
    struc.cubPointsSide0Ref, 'OPERATOR_VALUE', cellType, 1);

%%%%%%%%%%% evaluate reference subcell (side1) basis values  
val_at_cub_points_side1_refcell = zeros(numCubPointsSide, numFields);
intrepid_getBasisValues(val_at_cub_points_side1_refcell, ...
    struc.cubPointsSide1Ref, 'OPERATOR_VALUE', cellType, 1);

%%%%%%%%%%% evaluate reference subcell (side2) basis values  
val_at_cub_points_side2_refcell = zeros(numCubPointsSide, numFields);
intrepid_getBasisValues(val_at_cub_points_side2_refcell, ...
    struc.cubPointsSide2Ref, 'OPERATOR_VALUE', cellType, 1);

%%%%%%%%%%% transform basis values of reference subcell (side0) 
transformed_val_at_cub_points_side0_refcell = ...
    zeros(numCubPointsSide, numFields, numCells);
intrepid_HGRADtransformVALUE( ...
    transformed_val_at_cub_points_side0_refcell, ...
    val_at_cub_points_side0_refcell);

%%%%%%%%%%% transform basis values of reference subcell (side1) 
transformed_val_at_cub_points_side1_refcell = ...
    zeros(numCubPointsSide, numFields, numCells);
intrepid_HGRADtransformVALUE( ...
    transformed_val_at_cub_points_side1_refcell, ...
    val_at_cub_points_side1_refcell);

%%%%%%%%%%% transform basis values of reference subcell (side2) 
transformed_val_at_cub_points_side2_refcell = ...
    zeros(numCubPointsSide, numFields, numCells);
intrepid_HGRADtransformVALUE( ...
    transformed_val_at_cub_points_side2_refcell, ...
    val_at_cub_points_side2_refcell);

%%%%%%%%%%% combine reference side0 transformed basis values with measures 
weighted_transformed_val_at_cub_points_side0_refcell = ...
    zeros(numCubPointsSide, numFields, numCells);
intrepid_multiplyMeasure( ...
    weighted_transformed_val_at_cub_points_side0_refcell, ...
    weighted_measure_side0_refcell, ...
    transformed_val_at_cub_points_side0_refcell);

%%%%%%%%%%% combine reference side1 transformed basis values with measures 
weighted_transformed_val_at_cub_points_side1_refcell = ...
    zeros(numCubPointsSide, numFields, numCells);
intrepid_multiplyMeasure( ...
    weighted_transformed_val_at_cub_points_side1_refcell, ...
    weighted_measure_side1_refcell, ...
    transformed_val_at_cub_points_side1_refcell);

%%%%%%%%%%% combine reference side2 transformed basis values with measures 
weighted_transformed_val_at_cub_points_side2_refcell = ...
    zeros(numCubPointsSide, numFields, numCells);
intrepid_multiplyMeasure( ...
    weighted_transformed_val_at_cub_points_side2_refcell, ...
    weighted_measure_side2_refcell, ...
    transformed_val_at_cub_points_side2_refcell);
struc.weighted_transformed_val_at_cub_points_side2_refcell = ...
    weighted_transformed_val_at_cub_points_side2_refcell;

%%%%%%%%%%% get bottom side (BS) N matrices
cell_BS_N_matrices = zeros(spaceDim, numCubPointsSide, numDof, numCells);

cell_BS_N_matrices(1,:,1:spaceDim:end,NeumannCellsBS) = ...
    transformed_val_at_cub_points_side0_refcell(:,:,NeumannCellsBS);
cell_BS_N_matrices(2,:,2:spaceDim:end,NeumannCellsBS) = ...
    transformed_val_at_cub_points_side0_refcell(:,:,NeumannCellsBS);

struc.BS_Nmat = cell_BS_N_matrices;

%%%%%%%%%%% get right side (RS) N matrices
cell_RS_N_matrices = zeros(spaceDim, numCubPointsSide, numDof, numCells);

cell_RS_N_matrices(1,:,1:spaceDim:end,NeumannCellsRS) = ...
    transformed_val_at_cub_points_side1_refcell(:,:,NeumannCellsRS);
cell_RS_N_matrices(2,:,2:spaceDim:end,NeumannCellsRS) = ...
    transformed_val_at_cub_points_side1_refcell(:,:,NeumannCellsRS);

struc.RS_Nmat = cell_RS_N_matrices;

%%%%%%%%%%% get upper side (US) N matrices
cell_US_N_matrices = zeros(spaceDim, numCubPointsSide, numDof, numCells);

cell_US_N_matrices(1,:,1:spaceDim:end,NeumannCellsUS) = ...
    transformed_val_at_cub_points_side1_refcell(:,:,NeumannCellsUS);
cell_US_N_matrices(2,:,2:spaceDim:end,NeumannCellsUS) = ...
    transformed_val_at_cub_points_side1_refcell(:,:,NeumannCellsUS);

struc.US_Nmat = cell_US_N_matrices;

%%%%%%%%%%% get left side (LS) N matrices
cell_LS_N_matrices = zeros(spaceDim, numCubPointsSide, numDof, numCells);

cell_LS_N_matrices(1,:,1:spaceDim:end,NeumannCellsLS) = ...
    transformed_val_at_cub_points_side2_refcell(:,:,NeumannCellsLS);
cell_LS_N_matrices(2,:,2:spaceDim:end,NeumannCellsLS) = ...
    transformed_val_at_cub_points_side2_refcell(:,:,NeumannCellsLS);

struc.LS_Nmat = cell_LS_N_matrices;

%%%%%%%%%%% get weighted bottom side (BS) N matrices
cell_weighted_BS_N_matrices = ...
    zeros(spaceDim, numCubPointsSide, numDof, numCells);

cell_weighted_BS_N_matrices(1,:,1:spaceDim:end,NeumannCellsBS) = ...
    weighted_transformed_val_at_cub_points_side0_refcell(:,:,NeumannCellsBS);
cell_weighted_BS_N_matrices(2,:,2:spaceDim:end,NeumannCellsBS) = ...
    weighted_transformed_val_at_cub_points_side0_refcell(:,:,NeumannCellsBS);

struc.BS_wNmat = cell_weighted_BS_N_matrices;

%%%%%%%%%%% get weighted right side (RS) N matrices
cell_weighted_RS_N_matrices = ...
    zeros(spaceDim, numCubPointsSide, numDof, numCells);

cell_weighted_RS_N_matrices(1,:,1:spaceDim:end,NeumannCellsRS) = ...
    weighted_transformed_val_at_cub_points_side1_refcell(:,:,NeumannCellsRS);
cell_weighted_RS_N_matrices(2,:,2:spaceDim:end,NeumannCellsRS) = ...
    weighted_transformed_val_at_cub_points_side1_refcell(:,:,NeumannCellsRS);

struc.RS_wNmat = cell_weighted_RS_N_matrices;

%%%%%%%%%%% get weighted upper side (US) N matrices
cell_weighted_US_N_matrices = ...
    zeros(spaceDim, numCubPointsSide, numDof, numCells);

cell_weighted_US_N_matrices(1,:,1:spaceDim:end,NeumannCellsUS) = ...
    weighted_transformed_val_at_cub_points_side1_refcell(:,:,NeumannCellsUS);
cell_weighted_US_N_matrices(2,:,2:spaceDim:end,NeumannCellsUS) = ...
    weighted_transformed_val_at_cub_points_side1_refcell(:,:,NeumannCellsUS);

struc.US_wNmat = cell_weighted_US_N_matrices;

%%%%%%%%%%% get left side (LS) weighted N matrices
cell_weighted_LS_N_matrices = ...
    zeros(spaceDim, numCubPointsSide, numDof, numCells);

cell_weighted_LS_N_matrices(1,:,1:spaceDim:end,NeumannCellsLS) = ...
    weighted_transformed_val_at_cub_points_side2_refcell(:,:,NeumannCellsLS);
cell_weighted_LS_N_matrices(2,:,2:spaceDim:end,NeumannCellsLS) = ...
    weighted_transformed_val_at_cub_points_side2_refcell(:,:,NeumannCellsLS);

struc.LS_wNmat = cell_weighted_LS_N_matrices;

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
val_at_cub_points = zeros(numCubPoints, numFields);
grad_at_cub_points = zeros(spaceDim, numCubPoints, numFields);
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
    numFields, numCells);
intrepid_HGRADtransformGRAD(transformed_grad_at_cub_points, ...
    cellJacobianInvs, grad_at_cub_points);
struc.transformed_grad_at_cub_points = transformed_grad_at_cub_points;

%%%%%%%%%%% transform values
transformed_val_at_cub_points = zeros(numCubPoints, numFields, numCells);
intrepid_HGRADtransformVALUE(transformed_val_at_cub_points, ...
    val_at_cub_points);
struc.transformed_val_at_cub_points = transformed_val_at_cub_points;

%%%%%%%%%%% combine transformed gradients with measures
weighted_transformed_grad_at_cub_points = zeros(spaceDim, ...
    numCubPoints, numFields, numCells);
intrepid_multiplyMeasure(weighted_transformed_grad_at_cub_points, ...
    weighted_measure, transformed_grad_at_cub_points);
struc.weighted_transformed_grad_at_cub_points = ...
    weighted_transformed_grad_at_cub_points;

%%%%%%%%%%% combine transformed values with measures
weighted_transformed_val_at_cub_points = zeros(numCubPoints, ...
    numFields, numCells);
intrepid_multiplyMeasure(weighted_transformed_val_at_cub_points, ...
    weighted_measure, transformed_val_at_cub_points);
struc.weighted_transformed_val_at_cub_points = ...
    weighted_transformed_val_at_cub_points;

%%%%%%%%%%% integrate scalar field stiffness matrix
cell_stiffness_matrices = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_stiffness_matrices, ...
    transformed_grad_at_cub_points, ...
    weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global scalar field stiffness matrix
cell_stiffness_matrices = reshape(cell_stiffness_matrices, 1, ...
    numel(cell_stiffness_matrices));
stiff_mat = sparse(struc.iIdxVertices, ...
    struc.jIdxVertices, cell_stiffness_matrices);

%%%%%%%%%%% integrate scalar field mass matrix
cell_mass_matrices = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_mass_matrices, ...
    transformed_val_at_cub_points, ...
    weighted_transformed_val_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global scalar field mass matrix
cell_mass_matrices = reshape(cell_mass_matrices, 1, ...
    numel(cell_mass_matrices));
mass_mat = sparse(struc.iIdxVertices, ...
    struc.jIdxVertices, cell_mass_matrices);

%%%%%%%%%%% get cell deviatoric matrices
cell_deviatoric_matrices = zeros(3, 3, numCubPoints, numCells);
intrepid_getCellDeviatoricMat(cell_deviatoric_matrices, spaceDim);

%%%%%%%%%%% get cell volumetric matrices
cell_volumetric_matrices = zeros(3, 3, numCubPoints, numCells);
intrepid_getCellVolumetricMat(cell_volumetric_matrices, spaceDim);

%%%%%%%%%%% get cell N matrices
cell_N_matrices = zeros(spaceDim,numCubPoints, numDof, numCells);
cell_N_matrices(1,:,1:spaceDim:end,:) = ...
    transformed_val_at_cub_points(:,:,:);
cell_N_matrices(2,:,2:spaceDim:end,:) = ...
    transformed_val_at_cub_points(:,:,:);
struc.Nmat = cell_N_matrices;

%%%%%%%%%%% get cell weighted N matrices
cell_weighted_N_matrices = zeros(spaceDim,numCubPoints, numDof, numCells);
cell_weighted_N_matrices(1,:,1:spaceDim:end,:) = ...
    weighted_transformed_val_at_cub_points(:,:,:);
cell_weighted_N_matrices(2,:,2:spaceDim:end,:) = ...
    weighted_transformed_val_at_cub_points(:,:,:);
struc.weighted_Nmat = cell_weighted_N_matrices;

%%%%%%%%%%% get cell B matrices
cell_B_matrices = zeros(3, numCubPoints, numDof, numCells);
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
cell_weighted_B_matrices = zeros(3, numCubPoints, numDof, numCells);
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
cell_state_mass_matrices = zeros(numDof, numDof, numCells);
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

struc.Ss          = stiff_mat;
struc.Ms          = mass_mat;
struc.M           = state_mass_mat;
struc.Ddev        = cell_deviatoric_matrices;
struc.Dvol        = cell_volumetric_matrices;
struc.dirichlet   = dirichlet;
struc.FreeNodes   = FreeNodes;

struc.NeumannDof = NeumannDof;

struc.NeumannCellsLS = NeumannCellsLS;
struc.NeumannCellsUS = NeumannCellsUS;
struc.NeumannCellsRS = NeumannCellsRS;
struc.NeumannCellsBS = NeumannCellsBS;

struc.cubPointsPhysCoord      = cubPointsPhysCoord;
struc.cubPointsSide0PhysCoord = cubPointsSide0PhysCoord;
struc.cubPointsSide1PhysCoord = cubPointsSide1PhysCoord;
struc.cubPointsSide2PhysCoord = cubPointsSide2PhysCoord;

end
