function [K_perturbed] = computePerturbedStiffness(u,mu,kappa)

global GLB_INVP;

%%%%%%%%%%% evaluate shear modulus at the cubature points
mu_at_dof = mu( GLB_INVP.mesh.t');
mu_at_cub_points = zeros(GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_evaluate(mu_at_cub_points, mu_at_dof, ...
    GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% evaluate bulk modulus at the cubature points
kappa_at_dof = kappa(GLB_INVP.mesh.t');
kappa_at_cub_points = zeros(GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_evaluate(kappa_at_cub_points, kappa_at_dof, ...
    GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% get state solution at degrees of freedom of traingle i
u_new = zeros(GLB_INVP.spaceDim*GLB_INVP.nVertGrid,1);
u_new(GLB_INVP.FreeNodes) = u;
u_at_dof = u_new(GLB_INVP.mesh.d');

%%%%%%%%%%% evaluate strain field at the cubature points
strain_at_cub_points = zeros(3, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_evaluate(strain_at_cub_points, u_at_dof, GLB_INVP.Bmat);


%%%%%%%%%%% combine material deviatoric tensor with shear modulus
mu_times_Ddev = zeros(3, 3, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_scalarMultiplyDataData(mu_times_Ddev, ...
    mu_at_cub_points, GLB_INVP.Ddev);
                            
%%%%%%%%%%% compute deviatoric strain field (D_mu * epsilon)
deviatoric_strain = zeros(3, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_tensorMultiplyDataData(deviatoric_strain, ...
    mu_times_Ddev, strain_at_cub_points);

%%%%%%%%%%% combine deviatoric strain with basis functions (D_mu *
%%%%%%%%%%% epsilon)*phi
deviatoric_strain_times_phi = ...
    zeros(3, GLB_INVP.numCubPoints, GLB_INVP.numFields, GLB_INVP.numCells);
intrepid_multiplyVectorDataField( deviatoric_strain_times_phi, ...
    deviatoric_strain, GLB_INVP.transformed_val_at_cub_points );

%%%%%%%%%%% integrate matrix Shear*Bmat*(phi*deviatoric_strain)
int_Bmat_times_deviatoric_strain_times_phi = ...
    zeros(GLB_INVP.numDof, GLB_INVP.numFields, GLB_INVP.numCells);
intrepid_integrate( int_Bmat_times_deviatoric_strain_times_phi, ...
    deviatoric_strain_times_phi, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% assemble matrix matrix Bmat*phi*volumetric_strain
int_Bmat_times_deviatoric_strain_times_phi_reshape = ...
    reshape(int_Bmat_times_deviatoric_strain_times_phi, 1, ...
    numel(int_Bmat_times_deviatoric_strain_times_phi));
K_mu = sparse(GLB_INVP.iIdxMix, GLB_INVP.jIdxMix, ...
    int_Bmat_times_deviatoric_strain_times_phi_reshape);

%%%%%%%%%%% combine material volumetric tensor with bulk modulus
kappa_times_Dvol = zeros(3, 3, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_scalarMultiplyDataData(kappa_times_Dvol, ...
    kappa_at_cub_points, GLB_INVP.Dvol);
                            
%%%%%%%%%%% compute volumetric strain field
volumetric_strain = zeros(3, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_tensorMultiplyDataData(volumetric_strain, ...
    kappa_times_Dvol, strain_at_cub_points);

%%%%%%%%%%% combine volumetric strain with weighted basis functions
volumetric_strain_times_phi = ...
    zeros(3, GLB_INVP.numCubPoints, GLB_INVP.numFields, GLB_INVP.numCells);
intrepid_multiplyVectorDataField( volumetric_strain_times_phi, ...
    volumetric_strain, GLB_INVP.transformed_val_at_cub_points );

%%%%%%%%%%% integrate matrix Bmat*phi*volumetric_strain
int_Bmat_times_volumetric_strain_times_phi = ...
    zeros(GLB_INVP.numDof, GLB_INVP.numFields, GLB_INVP.numCells);
intrepid_integrate( int_Bmat_times_volumetric_strain_times_phi, ...
    volumetric_strain_times_phi, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% assemble matrix matrix Bmat*phi*volumetric_strain
int_Bmat_times_volumetric_strain_times_phi_reshape = ...
    reshape(int_Bmat_times_volumetric_strain_times_phi, 1, ...
    numel(int_Bmat_times_volumetric_strain_times_phi));
K_kappa = sparse(GLB_INVP.iIdxMix, GLB_INVP.jIdxMix, ...
    int_Bmat_times_volumetric_strain_times_phi_reshape);

K_perturbed = K_mu + K_kappa;

end