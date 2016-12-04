function [struc] = getSideSetShapeFunctions(struc)
%
%  getSideSetShapeFunctions(usr_par)
%
%  PURPOSE: Generates subcells shape functions. This data can be computed 
%           once and reused, thus speeding up the overall computation.
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
%           August 16, 2011

spaceDim         = struc.spaceDim;
cellType         = struc.cellType;
numFields        = struc.numFields;
numCubPointsSide = struc.numCubPointsSide;
numCells         = struc.numCells;
nx               = struc.nxint;
ny               = struc.nyint;

%%%%%%%%%%% initialization of Neumann cells array
struc.NeumannCellsLS = 2:2:2*ny;
struc.NeumannCellsUS = 2*ny:2*ny:2*nx*ny;
struc.NeumannCellsRS = ( 2*ny*(nx-1) ) + 1:2:2*nx*ny;
struc.NeumannCellsBS = 1:2*ny:( 2*ny*(nx-1) ) + 1; 

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
% side 0 = bottom side
weighted_transformed_val_at_cub_points_side0_refcell = ...
    zeros(numCubPointsSide, numFields, numCells);
intrepid_multiplyMeasure( ...
    weighted_transformed_val_at_cub_points_side0_refcell, ...
    weighted_measure_side0_refcell, ...
    transformed_val_at_cub_points_side0_refcell);
struc.weighted_transformed_val_at_cub_points_side0_refcell = ...
    weighted_transformed_val_at_cub_points_side0_refcell;

%%%%%%%%%%% combine reference side1 transformed basis values with measures 
% side 1 = upper side & right side
weighted_transformed_val_at_cub_points_side1_refcell = ...
    zeros(numCubPointsSide, numFields, numCells);
intrepid_multiplyMeasure( ...
    weighted_transformed_val_at_cub_points_side1_refcell, ...
    weighted_measure_side1_refcell, ...
    transformed_val_at_cub_points_side1_refcell);
struc.weighted_transformed_val_at_cub_points_side1_refcell = ...
    weighted_transformed_val_at_cub_points_side1_refcell;

%%%%%%%%%%% combine reference side2 transformed basis values with measures 
% side 2 = left side
weighted_transformed_val_at_cub_points_side2_refcell = ...
    zeros(numCubPointsSide, numFields, numCells);
intrepid_multiplyMeasure( ...
    weighted_transformed_val_at_cub_points_side2_refcell, ...
    weighted_measure_side2_refcell, ...
    transformed_val_at_cub_points_side2_refcell);
struc.weighted_transformed_val_at_cub_points_side2_refcell = ...
    weighted_transformed_val_at_cub_points_side2_refcell;

end