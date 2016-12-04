function [usr_par] = generateProbSpecificData(usr_par)
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

spaceDim         = usr_par.spaceDim;
cellType         = usr_par.cellType;
numFields        = usr_par.numFields;
numDof           = usr_par.numDof;
nVertGrid        = usr_par.nVertGrid;
numCubPoints     = usr_par.numCubPoints;
numCubPointsSide = usr_par.numCubPointsSide;
numCells         = usr_par.numCells;
nx               = usr_par.DomainSpecs.nx;
ny               = usr_par.DomainSpecs.ny;

%%%%%%%%%%%% set boundary markers
usr_par.mesh.e(1:ny,5) = 1;                  % edges on left boundary
%usr_par.mesh.e(ny+1:nx+ny,5) = 2;           % edges on top boundary
%usr_par.mesh.e(nx+2*ny+1:2*(nx+ny),5) = 1;  % edges on lower boundary

%%%%%%%%%%% Initialization of free nodes array.
%**** Fixed Boundary Condition / Left Wall ****
dirichletNodes = unique ( [ usr_par.mesh.e(usr_par.mesh.e(:,5) == 1,2)' ...
                   usr_par.mesh.e(usr_par.mesh.e(:,5) == 1,3)' ] );
dirichlet = [2*dirichletNodes 2*dirichletNodes-1];      
FreeDof = setdiff(1:2*nVertGrid, unique( dirichlet ));
FreeVertex = setdiff(1:nVertGrid, unique( dirichletNodes ));
% **** Roller Boundary Condition/ Mitchell Bridge
% dirichletNodes = unique ( [ usr_par.mesh.e(usr_par.mesh.e(:,5) == 1,2)' ...
%                    usr_par.mesh.e(usr_par.mesh.e(:,5) == 1,3)' ] );
% node = (usr_par.DomainSpecs.nx+1)*(usr_par.DomainSpecs.ny+1) - ...
%     (usr_par.DomainSpecs.ny);
% dirichlet = [2*dirichletNodes-1 2*node];      
% FreeDof = setdiff(1:2*nVertGrid, unique( dirichlet ));
% FreeVertex = setdiff(1:nVertGrid, unique( dirichletNodes ));

%%%%%%%%%%% initialization of neumann nodes array.
NeumannNodesLS = unique( usr_par.mesh.e(1:ny,1:2) );
NeumannNodesUS = unique( usr_par.mesh.e(1+ny:ny+nx,1:2) );
NeumannNodesRS = unique( usr_par.mesh.e(1+ny+nx:ny+nx+ny,1:2) );
NeumannNodesBS = unique( usr_par.mesh.e(1+ny+nx+ny:end,1:2) );

%%%%%%%%%%% initialization of Neumann cells array
%NeumannCellsLS = 2:2:2*nx;
%NeumannCellsUS = 2*nx:2*nx:2*nx*nx;
%NeumannCellsRS = ( 2*nx*(nx-1) ) + 1:2:2*nx*nx;
%NeumannCellsBS = 1:2*nx:( 2*nx*(nx-1) ) + 1; 
NeumannCellsLS = 2:2:2*ny;
NeumannCellsUS = 2*ny:2*ny:2*nx*ny;
NeumannCellsRS = ( 2*ny*(nx-1) ) + 1:2:2*nx*ny;
NeumannCellsBS = 1:2*ny:( 2*ny*(nx-1) ) + 1; 

%%%%%%%%%%% evaluate Jacobians at reference subcell (side0)
jacobiansSide0Ref  = zeros(spaceDim, spaceDim, numCubPointsSide, numCells);
intrepid_setJacobian(jacobiansSide0Ref, usr_par.cubPointsSide0Ref, ...
    usr_par.cellNodes, cellType);

%%%%%%%%%%% evaluate Jacobians at reference subcell (side1)
jacobiansSide1Ref  = zeros(spaceDim, spaceDim, numCubPointsSide, numCells);
intrepid_setJacobian(jacobiansSide1Ref, usr_par.cubPointsSide1Ref, ...
    usr_par.cellNodes, cellType);

%%%%%%%%%%% evaluate Jacobians at reference subcell (side2)
jacobiansSide2Ref  = zeros(spaceDim, spaceDim, numCubPointsSide, numCells);
intrepid_setJacobian(jacobiansSide2Ref, usr_par.cubPointsSide2Ref, ...
    usr_par.cellNodes, cellType);

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
    jacobiansSide0Ref, usr_par.cubWeightsSide, 0, cellType);

%%%%%%%%%%% compute reference subcell (side1) measures
weighted_measure_side1_refcell = zeros(numCubPointsSide, numCells);
intrepid_computeEdgeMeasure(weighted_measure_side1_refcell, ...
    jacobiansSide1Ref, usr_par.cubWeightsSide, 1, cellType);

%%%%%%%%%%% compute reference subcell (side2) measures
weighted_measure_side2_refcell = zeros(numCubPointsSide, numCells);
intrepid_computeEdgeMeasure(weighted_measure_side2_refcell, ...
    jacobiansSide2Ref, usr_par.cubWeightsSide, 2, cellType);

%%%%%%%%%%% evaluate reference subcell (side0) basis values  
val_at_cub_points_side0_refcell = zeros(numCubPointsSide, numFields);
intrepid_getBasisValues(val_at_cub_points_side0_refcell, ...
    usr_par.cubPointsSide0Ref, 'OPERATOR_VALUE', cellType, 1);

%%%%%%%%%%% evaluate reference subcell (side1) basis values  
val_at_cub_points_side1_refcell = zeros(numCubPointsSide, numFields);
intrepid_getBasisValues(val_at_cub_points_side1_refcell, ...
    usr_par.cubPointsSide1Ref, 'OPERATOR_VALUE', cellType, 1);

%%%%%%%%%%% evaluate reference subcell (side2) basis values  
val_at_cub_points_side2_refcell = zeros(numCubPointsSide, numFields);
intrepid_getBasisValues(val_at_cub_points_side2_refcell, ...
    usr_par.cubPointsSide2Ref, 'OPERATOR_VALUE', cellType, 1);

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
usr_par.weighted_transformed_val_at_cub_points_side2_refcell = ...
    weighted_transformed_val_at_cub_points_side2_refcell;

%%%%%%%%%%% get bottom side (BS) N matrices
cell_BS_N_matrices = zeros(spaceDim, numCubPointsSide, numDof, numCells);

cell_BS_N_matrices(1,:,1:spaceDim:end,NeumannCellsBS) = ...
    transformed_val_at_cub_points_side0_refcell(:,:,NeumannCellsBS);
cell_BS_N_matrices(2,:,2:spaceDim:end,NeumannCellsBS) = ...
    transformed_val_at_cub_points_side0_refcell(:,:,NeumannCellsBS);

usr_par.BS_Nmat = cell_BS_N_matrices;

%%%%%%%%%%% get right side (RS) N matrices
cell_RS_N_matrices = zeros(spaceDim, numCubPointsSide, numDof, numCells);

cell_RS_N_matrices(1,:,1:spaceDim:end,NeumannCellsRS) = ...
    transformed_val_at_cub_points_side1_refcell(:,:,NeumannCellsRS);
cell_RS_N_matrices(2,:,2:spaceDim:end,NeumannCellsRS) = ...
    transformed_val_at_cub_points_side1_refcell(:,:,NeumannCellsRS);

usr_par.RS_Nmat = cell_RS_N_matrices;

%%%%%%%%%%% get upper side (US) N matrices
cell_US_N_matrices = zeros(spaceDim, numCubPointsSide, numDof, numCells);

cell_US_N_matrices(1,:,1:spaceDim:end,NeumannCellsUS) = ...
    transformed_val_at_cub_points_side1_refcell(:,:,NeumannCellsUS);
cell_US_N_matrices(2,:,2:spaceDim:end,NeumannCellsUS) = ...
    transformed_val_at_cub_points_side1_refcell(:,:,NeumannCellsUS);

usr_par.US_Nmat = cell_US_N_matrices;

%%%%%%%%%%% get left side (LS) N matrices
cell_LS_N_matrices = zeros(spaceDim, numCubPointsSide, numDof, numCells);

cell_LS_N_matrices(1,:,1:spaceDim:end,NeumannCellsLS) = ...
    transformed_val_at_cub_points_side2_refcell(:,:,NeumannCellsLS);
cell_LS_N_matrices(2,:,2:spaceDim:end,NeumannCellsLS) = ...
    transformed_val_at_cub_points_side2_refcell(:,:,NeumannCellsLS);

usr_par.LS_Nmat = cell_LS_N_matrices;

%%%%%%%%%%% get weighted bottom side (BS) N matrices
cell_weighted_BS_N_matrices = ...
    zeros(spaceDim, numCubPointsSide, numDof, numCells);

cell_weighted_BS_N_matrices(1,:,1:spaceDim:end,NeumannCellsBS) = ...
    weighted_transformed_val_at_cub_points_side0_refcell(:,:,NeumannCellsBS);
cell_weighted_BS_N_matrices(2,:,2:spaceDim:end,NeumannCellsBS) = ...
    weighted_transformed_val_at_cub_points_side0_refcell(:,:,NeumannCellsBS);

usr_par.BS_wNmat = cell_weighted_BS_N_matrices;

%%%%%%%%%%% get weighted right side (RS) N matrices
cell_weighted_RS_N_matrices = ...
    zeros(spaceDim, numCubPointsSide, numDof, numCells);

cell_weighted_RS_N_matrices(1,:,1:spaceDim:end,NeumannCellsRS) = ...
    weighted_transformed_val_at_cub_points_side1_refcell(:,:,NeumannCellsRS);
cell_weighted_RS_N_matrices(2,:,2:spaceDim:end,NeumannCellsRS) = ...
    weighted_transformed_val_at_cub_points_side1_refcell(:,:,NeumannCellsRS);

usr_par.RS_wNmat = cell_weighted_RS_N_matrices;

%%%%%%%%%%% get weighted upper side (US) N matrices
cell_weighted_US_N_matrices = ...
    zeros(spaceDim, numCubPointsSide, numDof, numCells);

cell_weighted_US_N_matrices(1,:,1:spaceDim:end,NeumannCellsUS) = ...
    weighted_transformed_val_at_cub_points_side1_refcell(:,:,NeumannCellsUS);
cell_weighted_US_N_matrices(2,:,2:spaceDim:end,NeumannCellsUS) = ...
    weighted_transformed_val_at_cub_points_side1_refcell(:,:,NeumannCellsUS);

usr_par.US_wNmat = cell_weighted_US_N_matrices;

%%%%%%%%%%% get left side (LS) weighted N matrices
cell_weighted_LS_N_matrices = ...
    zeros(spaceDim, numCubPointsSide, numDof, numCells);

cell_weighted_LS_N_matrices(1,:,1:spaceDim:end,NeumannCellsLS) = ...
    weighted_transformed_val_at_cub_points_side2_refcell(:,:,NeumannCellsLS);
cell_weighted_LS_N_matrices(2,:,2:spaceDim:end,NeumannCellsLS) = ...
    weighted_transformed_val_at_cub_points_side2_refcell(:,:,NeumannCellsLS);

usr_par.LS_wNmat = cell_weighted_LS_N_matrices;

%%%%%%%%%%% evaluate cell Jacobians
cellJacobians  = zeros(spaceDim, spaceDim, numCubPoints, numCells);
intrepid_setJacobian(cellJacobians, usr_par.cubPoints, ...
    usr_par.cellNodes, cellType);
usr_par.cellJacobians = cellJacobians;

%%%%%%%%%%% evaluate inverses of cell Jacobians
cellJacobianInvs = zeros(spaceDim, spaceDim, numCubPoints, numCells);
intrepid_setJacobianInv(cellJacobianInvs, cellJacobians);
usr_par.cellJacobianInvs = cellJacobianInvs;

%%%%%%%%%%% evaluate determinants of cell Jacobians
cellJacobianDets  = zeros(numCubPoints, numCells);
intrepid_setJacobianDet(cellJacobianDets, cellJacobians);
usr_par.cellJacobianDets = cellJacobianDets;

%%%%%%%%%%% evaluate basis (value, gradient)
val_at_cub_points = zeros(numCubPoints, numFields);
grad_at_cub_points = zeros(spaceDim, numCubPoints, numFields);
intrepid_getBasisValues(val_at_cub_points, usr_par.cubPoints, ...
    'OPERATOR_VALUE', cellType, 1);
intrepid_getBasisValues(grad_at_cub_points, usr_par.cubPoints, ...
    'OPERATOR_GRAD', cellType, 1);
usr_par.val_at_cub_points = val_at_cub_points;
usr_par.grad_at_cub_points = grad_at_cub_points;

%%%%%%%%%%% compute cell measures
weighted_measure = zeros(numCubPoints, numCells);
intrepid_computeCellMeasure(weighted_measure, ...
    cellJacobianDets, usr_par.cubWeights);
usr_par.weighted_measure = weighted_measure;

%%%%%%%%%%% transform gradients
transformed_grad_at_cub_points = zeros(spaceDim, numCubPoints, ...
    numFields, numCells);
intrepid_HGRADtransformGRAD(transformed_grad_at_cub_points, ...
    cellJacobianInvs, grad_at_cub_points);
usr_par.transformed_grad_at_cub_points = transformed_grad_at_cub_points;

%%%%%%%%%%% transform values
transformed_val_at_cub_points = zeros(numCubPoints, numFields, numCells);
intrepid_HGRADtransformVALUE(transformed_val_at_cub_points, ...
    val_at_cub_points);
usr_par.transformed_val_at_cub_points = transformed_val_at_cub_points;

%%%%%%%%%%% combine transformed gradients with measures
weighted_transformed_grad_at_cub_points = zeros(spaceDim, ...
    numCubPoints, numFields, numCells);
intrepid_multiplyMeasure(weighted_transformed_grad_at_cub_points, ...
    weighted_measure, transformed_grad_at_cub_points);
usr_par.weighted_transformed_grad_at_cub_points = ...
    weighted_transformed_grad_at_cub_points;

%%%%%%%%%%% combine transformed values with measures
weighted_transformed_val_at_cub_points = zeros(numCubPoints, ...
    numFields, numCells);
intrepid_multiplyMeasure(weighted_transformed_val_at_cub_points, ...
    weighted_measure, transformed_val_at_cub_points);
usr_par.weighted_transformed_val_at_cub_points = ...
    weighted_transformed_val_at_cub_points;

%%%%%%%%%%% integrate scalar field stiffness matrix
cell_stiffness_matrices = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_stiffness_matrices, ...
    transformed_grad_at_cub_points, ...
    weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global scalar field stiffness matrix
cell_stiffness_matrices = reshape(cell_stiffness_matrices, 1, ...
    numel(cell_stiffness_matrices));
stiff_mat = sparse(usr_par.iIdxVertices, ...
    usr_par.jIdxVertices, cell_stiffness_matrices);

%%%%%%%%%%% integrate scalar field mass matrix
cell_mass_matrices = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_mass_matrices, ...
    transformed_val_at_cub_points, ...
    weighted_transformed_val_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global scalar field mass matrix
one = ones(numFields,1);
elem_volume = zeros(numCells,1);
for i=1:numCells
   elem_volume(i) = sum(cell_mass_matrices(:,:,i) * one); 
end
cell_mass_matrices_reshape = reshape(cell_mass_matrices, 1, ...
    numel(cell_mass_matrices));
mass_mat = sparse(usr_par.iIdxVertices, ...
    usr_par.jIdxVertices, cell_mass_matrices_reshape);

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
usr_par.Nmat = cell_N_matrices;

%%%%%%%%%%% get cell weighted N matrices
cell_weighted_N_matrices = zeros(spaceDim,numCubPoints, numDof, numCells);
cell_weighted_N_matrices(1,:,1:spaceDim:end,:) = ...
    weighted_transformed_val_at_cub_points(:,:,:);
cell_weighted_N_matrices(2,:,2:spaceDim:end,:) = ...
    weighted_transformed_val_at_cub_points(:,:,:);
usr_par.weighted_Nmat = cell_weighted_N_matrices;

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
usr_par.Bmat = cell_B_matrices;

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
usr_par.weighted_Bmat = cell_weighted_B_matrices;

%%%%%%%%%%% integrate naked state mass matrix
cell_state_mass_matrices = zeros(numDof, numDof, numCells);
intrepid_integrate(cell_state_mass_matrices, cell_N_matrices, ...
    cell_weighted_N_matrices, 'COMP_BLAS');

%%%%%%%%%%% build global naked state mass matrix
cell_state_mass_matrices = reshape(cell_state_mass_matrices, 1, ...
    numel(cell_state_mass_matrices));
state_mass_mat = sparse(usr_par.iIdxDof, ...
    usr_par.jIdxDof, cell_state_mass_matrices);

%%%%%%%%%%% get computational mesh cubature points physical frame
cubPointsPhysCoord = zeros(spaceDim, numCubPoints, numCells);
intrepid_mapToPhysicalFrame(cubPointsPhysCoord, usr_par.cubPoints, ...
    usr_par.cellNodes, cellType);

%%%%%%%%%%% get computational mesh side0 cubature points physical frame
cubPointsSide0PhysCoord = zeros(spaceDim, numCubPointsSide, numCells);
intrepid_mapToPhysicalFrame(cubPointsSide0PhysCoord, ...
    usr_par.cubPointsSide0Ref, usr_par.cellNodes, cellType);

%%%%%%%%%%% get computational mesh side1 cubature points physical frame
cubPointsSide1PhysCoord = zeros(spaceDim, numCubPointsSide, numCells);
intrepid_mapToPhysicalFrame(cubPointsSide1PhysCoord, ...
    usr_par.cubPointsSide1Ref, usr_par.cellNodes, cellType);

%%%%%%%%%%% get computational mesh side2 cubature points physical frame
cubPointsSide2PhysCoord = zeros(spaceDim, numCubPointsSide, numCells);
intrepid_mapToPhysicalFrame(cubPointsSide2PhysCoord, ...
    usr_par.cubPointsSide2Ref, usr_par.cellNodes, cellType);

usr_par.Ss          = stiff_mat;
usr_par.Ms          = mass_mat;
usr_par.M           = state_mass_mat;
usr_par.Ddev        = cell_deviatoric_matrices;
usr_par.Dvol        = cell_volumetric_matrices;
usr_par.nu          = size(FreeDof, 2);
usr_par.nm          = nVertGrid;
usr_par.nk          = nVertGrid;
usr_par.dirichlet   = dirichlet;
usr_par.FreeDof     = FreeDof;
usr_par.FreeVertex  = FreeVertex;
usr_par.ElemVolume  = elem_volume;
usr_par.CellMassMatrices = cell_mass_matrices;

usr_par.NeumannNodesLS = NeumannNodesLS;
usr_par.NeumannNodesUS = NeumannNodesUS;
usr_par.NeumannNodesRS = NeumannNodesRS;
usr_par.NeumannNodesBS = NeumannNodesBS;

usr_par.NeumannCellsLS = NeumannCellsLS;
usr_par.NeumannCellsUS = NeumannCellsUS;
usr_par.NeumannCellsRS = NeumannCellsRS;
usr_par.NeumannCellsBS = NeumannCellsBS;

usr_par.cubPointsPhysCoord      = cubPointsPhysCoord;
usr_par.cubPointsSide0PhysCoord = cubPointsSide0PhysCoord;
usr_par.cubPointsSide1PhysCoord = cubPointsSide1PhysCoord;
usr_par.cubPointsSide2PhysCoord = cubPointsSide2PhysCoord;

end
