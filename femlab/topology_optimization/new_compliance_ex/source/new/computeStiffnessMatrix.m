function [K] = computeStiffnessMatrix(mu, kappa)

global GLB_INVP;

%%%%%%%%%%% evaluate shear modulus at the cubature points
mu_at_dof = mu( GLB_INVP.mesh.t');
mu_at_cub_points = zeros(GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_evaluate(mu_at_cub_points, mu_at_dof, ...
    GLB_INVP.transformed_val_at_cub_points);
    
%%%%%%%%%%% evaluate bulk modulus at the cubature points
kappa_at_dof = kappa( GLB_INVP.mesh.t');
kappa_at_cub_points = zeros(GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_evaluate(kappa_at_cub_points, kappa_at_dof, ...
    GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% combine Bmat with shear modulus
mu_times_Bmat = ...
    zeros(3, GLB_INVP.numCubPoints, GLB_INVP.numDof, GLB_INVP.numCells);
intrepid_scalarMultiplyDataField( mu_times_Bmat, ...
    mu_at_cub_points, GLB_INVP.Bmat);

%%%%%%%%%%% matrix-vector product, deviatoric coefficient mat. and fields
Ddev_times_mu_times_Bmat = ...
    zeros(3, GLB_INVP.numCubPoints, GLB_INVP.numDof, GLB_INVP.numCells);
intrepid_tensorMultiplyDataField( Ddev_times_mu_times_Bmat, ...
    GLB_INVP.Ddev, mu_times_Bmat);

%%%%%%%%%%% integrate deviatoric stiffnes matrix
cell_deviatoric_stiffness_matrices = ...
    zeros(GLB_INVP.numDof, GLB_INVP.numDof, GLB_INVP.numCells);
intrepid_integrate(cell_deviatoric_stiffness_matrices, ...
    Ddev_times_mu_times_Bmat, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% combine Bmat with bulk modulus
kappa_times_Bmat = ...
    zeros(3, GLB_INVP.numCubPoints, GLB_INVP.numDof, GLB_INVP.numCells);
intrepid_scalarMultiplyDataField( kappa_times_Bmat, ...
    kappa_at_cub_points, GLB_INVP.Bmat);

%%%%%%%%%%% matrix-vector product, volumetric coefficient mat. and fields
Dvol_times_kappa_times_Bmat = ...
    zeros(3, GLB_INVP.numCubPoints, GLB_INVP.numDof, GLB_INVP.numCells);
intrepid_tensorMultiplyDataField( Dvol_times_kappa_times_Bmat, ...
    GLB_INVP.Dvol, kappa_times_Bmat);

%%%%%%%%%%% integrate volumetric stiffnes matrix
cell_volumetric_stiffness_matrices = ...
    zeros(GLB_INVP.numDof, GLB_INVP.numDof, GLB_INVP.numCells);
intrepid_integrate(cell_volumetric_stiffness_matrices, ...
    Dvol_times_kappa_times_Bmat, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% compute stiffness matrix
K = cell_deviatoric_stiffness_matrices + cell_volumetric_stiffness_matrices;

end