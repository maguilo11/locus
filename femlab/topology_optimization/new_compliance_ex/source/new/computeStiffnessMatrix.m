function [stiffness_matrix] = ...
    computeStiffnessMatrix(shear_modulus, bulk_modulus)

global GLB_INVP;
number_of_materials = size(shear_modulus,2);
stiffness_matrix = ...
    zeros(GLB_INVP.numDof, GLB_INVP.numDof, GLB_INVP.numCells,number_of_materials);
for material_index=1:number_of_materials
    %%%%%%%%%%% evaluate shear modulus at the cubature points
    this_shear_modulus = shear_modulus(:,material_index);
    shear_modulus_at_dof = this_shear_modulus(GLB_INVP.mesh.t');
    shear_modulus_at_cub_points = zeros(GLB_INVP.numCubPoints, GLB_INVP.numCells);
    intrepid_evaluate(shear_modulus_at_cub_points, shear_modulus_at_dof, ...
        GLB_INVP.transformed_val_at_cub_points);
    
    %%%%%%%%%%% evaluate bulk modulus at the cubature points
    this_bulk_modulus = bulk_modulus(:,material_index);
    bulk_modulus_at_dof = this_bulk_modulus(GLB_INVP.mesh.t');
    bulk_modulus_at_cub_points = zeros(GLB_INVP.numCubPoints, GLB_INVP.numCells);
    intrepid_evaluate(bulk_modulus_at_cub_points, bulk_modulus_at_dof, ...
        GLB_INVP.transformed_val_at_cub_points);
    
    %%%%%%%%%%% combine Bmat with shear modulus
    shear_modulus_times_Bmat = ...
        zeros(3, GLB_INVP.numCubPoints, GLB_INVP.numDof, GLB_INVP.numCells);
    intrepid_scalarMultiplyDataField( shear_modulus_times_Bmat, ...
        shear_modulus_at_cub_points, GLB_INVP.Bmat);
    
    %%%%%%%%%%% matrix-vector product, deviatoric coefficient mat. and fields
    Ddev_times_shear_modulus_times_Bmat = ...
        zeros(3, GLB_INVP.numCubPoints, GLB_INVP.numDof, GLB_INVP.numCells);
    intrepid_tensorMultiplyDataField( Ddev_times_shear_modulus_times_Bmat, ...
        GLB_INVP.Ddev, shear_modulus_times_Bmat);
    
    %%%%%%%%%%% integrate deviatoric stiffnes matrix
    cell_deviatoric_stiffness_matrices = ...
        zeros(GLB_INVP.numDof, GLB_INVP.numDof, GLB_INVP.numCells);
    intrepid_integrate(cell_deviatoric_stiffness_matrices, ...
        Ddev_times_shear_modulus_times_Bmat, GLB_INVP.weighted_Bmat, 'COMP_BLAS');
    
    %%%%%%%%%%% combine Bmat with bulk modulus
    bulk_modulus_times_Bmat = ...
        zeros(3, GLB_INVP.numCubPoints, GLB_INVP.numDof, GLB_INVP.numCells);
    intrepid_scalarMultiplyDataField( bulk_modulus_times_Bmat, ...
        bulk_modulus_at_cub_points, GLB_INVP.Bmat);
    
    %%%%%%%%%%% matrix-vector product, volumetric coefficient mat. and fields
    Dvol_times_bulk_modulus_times_Bmat = ...
        zeros(3, GLB_INVP.numCubPoints, GLB_INVP.numDof, GLB_INVP.numCells);
    intrepid_tensorMultiplyDataField( Dvol_times_bulk_modulus_times_Bmat, ...
        GLB_INVP.Dvol, bulk_modulus_times_Bmat);
    
    %%%%%%%%%%% integrate volumetric stiffnes matrix
    cell_volumetric_stiffness_matrices = ...
        zeros(GLB_INVP.numDof, GLB_INVP.numDof, GLB_INVP.numCells);
    intrepid_integrate(cell_volumetric_stiffness_matrices, ...
        Dvol_times_bulk_modulus_times_Bmat, GLB_INVP.weighted_Bmat, 'COMP_BLAS');
    
    %%%%%%%%%%% compute stiffness matrix
    stiffness_matrix(:,:,:,material_index) = ...
        cell_deviatoric_stiffness_matrices + cell_volumetric_stiffness_matrices;
end
end