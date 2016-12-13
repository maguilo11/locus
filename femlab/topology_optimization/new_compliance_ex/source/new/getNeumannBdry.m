function [struc] = getNeumannBdry(struc)

top = 1; % given by Cubit, both in 2D and 3D

bdry_cells  = struc.mesh.elem_ss{top};
local_side_ids = struc.mesh.side_ss{top};

clear wNmat
clear traction;
for i=1:struc.mesh.sidesPerCell
    neumann_cells = bdry_cells(local_side_ids == i-1);
    struc.neumann_cells{i} = neumann_cells;
    numCells = length(neumann_cells);
    cellNodesNeumann = struc.cellNodes(:,:,neumann_cells);
    
    if isempty(cellNodesNeumann)
        wNmat{i} = [];
        traction{i} = [];
    else
        %%%%%%%%%%% evaluate Jacobians at reference subcell
        jacobiansSideRef = ...
            zeros(spaceDim, spaceDim, numCubPointsSide, numCells);
        intrepid_setJacobian(jacobiansSideRef, struc.cubPointsSideRef{i}, ...
            cellNodesNeumann, cellType);
        
        %%%%%%%%%%% evaluate determinants of reference subcell Jacobians
        jacobiansDetSideRef = zeros(numCubPointsSide, numCells);
        intrepid_setJacobianDet(jacobiansDetSideRef, jacobiansSideRef);
        
        %%%%%%%%%%% compute reference subcell measures
        weighted_measure_side_refcell = zeros(numCubPointsSide, numCells);
        switch spaceDim
            case 2
                intrepid_computeEdgeMeasure(weighted_measure_side_refcell, ...
                    jacobiansSideRef, ...
                    struc.cubWeightsSide, ...
                    i-1, cellType);
            case 3
                intrepid_computeFaceMeasure(weighted_measure_side_refcell, ...
                    jacobiansSideRef, ...
                    struc.cubWeightsSide, ...
                    i-1, cellType);
            otherwise
                error('Only 2D and 3D problems can be solved');
        end
        
        %%%%%%%%%%% evaluate reference subcell (side0) basis values
        val_at_cub_points_side_refcell = zeros(numCubPointsSide, numFields);
        intrepid_getBasisValues(val_at_cub_points_side_refcell, ...
            struc.cubPointsSideRef{i}, ...
            'OPERATOR_VALUE', cellType, 1);
        
        %%%%%%%%%%% transform basis values of reference subcell
        transformed_val_at_cub_points_side_refcell = ...
            zeros(numCubPointsSide, numFields, numCells);
        intrepid_HGRADtransformVALUE( ...
            transformed_val_at_cub_points_side_refcell, ...
            val_at_cub_points_side_refcell);
        
        %%%%%%%%%%% combine reference side transformed basis values with measures
        weighted_transformed_val_at_cub_points_side_refcell = ...
            zeros(numCubPointsSide, numFields, numCells);
        intrepid_multiplyMeasure( ...
            weighted_transformed_val_at_cub_points_side_refcell, ...
            weighted_measure_side_refcell, ...
            transformed_val_at_cub_points_side_refcell);
        
        %%%%%%%%%%% get weighted Neumann matrices
        cell_weighted_Neumann_matrices = ...
            zeros(spaceDim, numCubPointsSide, numDof, numCells);
        
        for d=1:spaceDim
            cell_weighted_Neumann_matrices(d,:,d:spaceDim:end,:) = ...
                weighted_transformed_val_at_cub_points_side_refcell;
        end
        
        wNmat{i} = cell_weighted_Neumann_matrices;
        
        %%%%%%%%%%% get computational mesh side2 cubature points physical frame
        cubPointsSidePhysCoord = zeros(spaceDim, numCubPointsSide, numCells);
        intrepid_mapToPhysicalFrame(cubPointsSidePhysCoord, ...
            struc.cubPointsSideRef{i}, ...
            cellNodesNeumann, cellType);
        
        %%%%%%%%%%% get traction forces
        traction{i} = generateTraction(cubPointsSidePhysCoord);
    end
    
end
