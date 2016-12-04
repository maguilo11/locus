function [Operators] = objectiveFunction()
Operators.evaluate=@(control)evaluate(control);
Operators.firstDerivative=@(control)firstDerivative(control);
Operators.secondDerivative=@(control,dcontrol)secondDerivative(control,dcontrol);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = evaluate(control)

global GLB_INVP;

%%%% Solve PDE
EqualityConstraint = equalityConstraint();
[state] = EqualityConstraint.solve(control);

%%%% compute potential energy misfit
residual = GLB_INVP.NewK*GLB_INVP.exp_state - GLB_INVP.force;
potential = 0.5 * GLB_INVP.theta * ...
    (state'*residual - GLB_INVP.exp_state'*residual) * ...
    (state'*residual - GLB_INVP.exp_state'*residual);

%%%% Get shear and bulk modulus ffom control array
shear_modulus = control(1:GLB_INVP.nVertGrid);
bulk_modulus = control(GLB_INVP.nVertGrid+1:end);

%%%% compute regularization term
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = 0.5 * GLB_INVP.beta * (shear_modulus'*(GLB_INVP.Ms*shear_modulus)) + ...
            0.5 * GLB_INVP.beta * (bulk_modulus'*(GLB_INVP.Ms*bulk_modulus));
    case{'TV'}
        reg = 0.5*GLB_INVP.beta * ...
              sqrt( shear_modulus' * (GLB_INVP.Ss * shear_modulus) ...
              + GLB_INVP.gamma ) + ...  
            0.5*GLB_INVP.beta * sqrt( bulk_modulus' * (GLB_INVP.Ss * bulk_modulus) ...
              + GLB_INVP.gamma );
end

GLB_INVP.state = state;
output = potential + reg;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivative(control)

global GLB_INVP;

numDof       = GLB_INVP.numDof;
numCubPoints = GLB_INVP.numCubPoints;
numCells     = GLB_INVP.numCells;
nVertGrid    = GLB_INVP.nVertGrid;
numFields    = GLB_INVP.numFields;

%%%% Get shear and bulk modulus ffom control array
shear_modulus = control(1:nVertGrid);
bulk_modulus = control(nVertGrid+1:end);
%%
%%%%%%%%%%% evaluate shear modulus at the cubature points
shear_at_dof = shear_modulus( GLB_INVP.mesh.t');
shear_at_cub_points = zeros(numCubPoints, numCells);
intrepid_evaluate(shear_at_cub_points, shear_at_dof, ...
    GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% evaluate bulk modulus at the cubature points
bulk_at_dof = bulk_modulus( GLB_INVP.mesh.t');
bulk_at_cub_points = zeros(numCubPoints, numCells);
intrepid_evaluate(bulk_at_cub_points, bulk_at_dof, ...
    GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% combine Bmat with shear modulus
shear_times_Bmat = zeros(3, numCubPoints, numDof, numCells);
intrepid_scalarMultiplyDataField( shear_times_Bmat, ...
    shear_at_cub_points, GLB_INVP.Bmat);

%%%%%%%%%%% matrix-vector product, deviatoric coefficient mat. and fields
Ddev_times_shear_times_Bmat = zeros(3, numCubPoints, numDof, numCells);
intrepid_tensorMultiplyDataField( Ddev_times_shear_times_Bmat, ...
    GLB_INVP.Ddev, shear_times_Bmat);

%%%%%%%%%%% integrate deviatoric stiffnes matrix
cell_Kdev_matrices = zeros(numDof, numDof, numCells);
intrepid_integrate(cell_Kdev_matrices, ...
    Ddev_times_shear_times_Bmat, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% build global deviatoric stiffness matrix
Kdev_matrices = reshape(cell_Kdev_matrices, 1, numel(cell_Kdev_matrices));
Kdev = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, Kdev_matrices);

%%%%%%%%%%% combine Bmat with bulk modulus
bulk_times_Bmat = zeros(3, numCubPoints, numDof, numCells);
intrepid_scalarMultiplyDataField( bulk_times_Bmat, ...
    bulk_at_cub_points, GLB_INVP.Bmat);

%%%%%%%%%%% matrix-vector product, volumetric coefficient mat. and fields
Dvol_times_bulk_times_Bmat = zeros(3, numCubPoints, numDof, numCells);
intrepid_tensorMultiplyDataField( Dvol_times_bulk_times_Bmat, ...
    GLB_INVP.Dvol, bulk_times_Bmat);

%%%%%%%%%%% integrate volumetric stiffnes matrix
cell_Kvol_matrices = zeros(numDof, numDof, numCells);
intrepid_integrate(cell_Kvol_matrices, ...
    Dvol_times_bulk_times_Bmat, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% build global volumetric stiffness matrix
Kvol_matrices = reshape(cell_Kvol_matrices, 1, numel(cell_Kvol_matrices));
Kvol = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, Kvol_matrices);

%%%%%%%%%%% compute stiffness matrix
K = Kdev + Kvol;
%%
%%%%%%%%%%% evaluate experimental strain field at the cubature points
data_at_dof = GLB_INVP.exp_state(GLB_INVP.mesh.d');
exp_strain_at_cub_points = zeros(3, numCubPoints, numCells);
intrepid_evaluate(exp_strain_at_cub_points, data_at_dof, GLB_INVP.Bmat);

%%%%%%%%%%% compute deviatoric strain field
exp_deviatoric_strain = zeros(3, numCubPoints, numCells);
intrepid_tensorMultiplyDataData(exp_deviatoric_strain, ...
    GLB_INVP.Ddev, exp_strain_at_cub_points);

%%%%%%%%%%% combine deviatoric strain with basis functions
exp_deviatoric_strain_times_phi = zeros(3, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( exp_deviatoric_strain_times_phi, ...
    exp_deviatoric_strain, GLB_INVP.transformed_val_at_cub_points );

%%%%%%%%%%% integrate matrix Bmat*phi*deviatoric_strain
exp_cell_dK_dev_matrices = zeros(numDof, numFields, numCells);
intrepid_integrate( exp_cell_dK_dev_matrices, ...
    exp_deviatoric_strain_times_phi, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% assemble matrix matrix Bmat*phi*volumetric_strain
exp_dK_dev_matrices = reshape(exp_cell_dK_dev_matrices, 1, numel(exp_cell_dK_dev_matrices));
exp_dK_dev = sparse(GLB_INVP.iIdxMix, GLB_INVP.jIdxMix, exp_dK_dev_matrices);

%%%%%%%%%%% compute volumetric strain field
exp_volumetric_strain = zeros(3, numCubPoints, numCells);
intrepid_tensorMultiplyDataData(exp_volumetric_strain, ...
    GLB_INVP.Dvol, exp_strain_at_cub_points);

%%%%%%%%%%% combine volumetric strain with weighted basis functions
exp_volumetric_strain_times_phi = zeros(3, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( exp_volumetric_strain_times_phi, ...
    exp_volumetric_strain, GLB_INVP.transformed_val_at_cub_points );

%%%%%%%%%%% integrate matrix Bmat*phi*volumetric_strain
exp_cell_dK_vol_matrices = zeros(numDof, numFields, numCells);
intrepid_integrate( exp_cell_dK_vol_matrices, ...
    exp_volumetric_strain_times_phi, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% assemble matrix matrix Bmat*phi*volumetric_strain
exp_dK_vol_matrices = reshape(exp_cell_dK_vol_matrices, 1, numel(exp_cell_dK_vol_matrices));
exp_dK_vol = sparse(GLB_INVP.iIdxMix, GLB_INVP.jIdxMix, exp_dK_vol_matrices);

%%

%%%%%%%%%%% get state solution at traingle's dof
state_at_dof = GLB_INVP.state(GLB_INVP.mesh.d');

%%%%%%%%%%% evaluate strain field at the cubature points
strain_at_cub_points = zeros(3, numCubPoints, numCells);
intrepid_evaluate(strain_at_cub_points, state_at_dof, GLB_INVP.Bmat);

%%%%%%%%%%% compute deviatoric strain field
deviatoric_strain = zeros(3, numCubPoints, numCells);
intrepid_tensorMultiplyDataData(deviatoric_strain, ...
    GLB_INVP.Ddev, strain_at_cub_points);

%%%%%%%%%%% combine deviatoric strain with basis functions
deviatoric_strain_times_phi = zeros(3, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( deviatoric_strain_times_phi, ...
    deviatoric_strain, GLB_INVP.transformed_val_at_cub_points );

%%%%%%%%%%% integrate matrix Bmat*phi*deviatoric_strain
cell_dK_dev_matrices = zeros(numDof, numFields, numCells);
intrepid_integrate( cell_dK_dev_matrices, ...
    deviatoric_strain_times_phi, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% assemble matrix matrix Bmat*phi*volumetric_strain
dK_dev_matrices = reshape(cell_dK_dev_matrices, 1, numel(cell_dK_dev_matrices));
dK_dev = sparse(GLB_INVP.iIdxMix, GLB_INVP.jIdxMix, dK_dev_matrices);

%%%%%%%%%%% compute volumetric strain field
volumetric_strain = zeros(3, numCubPoints, numCells);
intrepid_tensorMultiplyDataData(volumetric_strain, ...
    GLB_INVP.Dvol, strain_at_cub_points);

%%%%%%%%%%% combine volumetric strain with weighted basis functions
volumetric_strain_times_phi = zeros(3, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( volumetric_strain_times_phi, ...
    volumetric_strain, GLB_INVP.transformed_val_at_cub_points );

%%%%%%%%%%% integrate matrix Bmat*phi*volumetric_strain
cell_dK_vol_matrices = zeros(numDof, numFields, numCells);
intrepid_integrate( cell_dK_vol_matrices, ...
    volumetric_strain_times_phi, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% assemble matrix matrix Bmat*phi*volumetric_strain
dK_vol_matrices = reshape(cell_dK_vol_matrices, 1, numel(cell_dK_vol_matrices));
dK_vol = sparse(GLB_INVP.iIdxMix, GLB_INVP.jIdxMix, dK_vol_matrices);
%%
%%%%%%%%%%% potential contribution 
residual = K*GLB_INVP.exp_state - GLB_INVP.force;
alpha = (GLB_INVP.state'*residual - GLB_INVP.exp_state'*residual);
residual_misfit = [ (alpha*GLB_INVP.theta*((GLB_INVP.state'*dK_dev) - ...
                      (GLB_INVP.exp_state'*exp_dK_dev)))'; ...
                    (alpha*GLB_INVP.theta*((GLB_INVP.state'*dK_vol) - ...
                      (GLB_INVP.exp_state'*exp_dK_vol)))'];

%%%%%%%%%%% regularization contribution
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = [ GLB_INVP.beta * (GLB_INVP.Ms*shear_modulus); ...
                GLB_INVP.beta * (GLB_INVP.Ms*bulk_modulus) ];
    case{'TV'}
        reg = [ GLB_INVP.beta * 0.5 * ( 1.0 / sqrt( ...
                 shear_modulus' * (GLB_INVP.Ss * shear_modulus) + ...
                 GLB_INVP.gamma ) ) * (GLB_INVP.Ss * shear_modulus); ...
                GLB_INVP.beta * 0.5 * ( 1.0 / sqrt( ...
                 bulk_modulus' * (GLB_INVP.Ss * bulk_modulus) + ...
                 GLB_INVP.gamma ) ) * (GLB_INVP.Ss * bulk_modulus) ];
end

output = residual_misfit + reg;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivative(control,dcontrol)

global GLB_INVP;

numDof       = GLB_INVP.numDof;
numCubPoints = GLB_INVP.numCubPoints;
numCells     = GLB_INVP.numCells;
nVertGrid    = GLB_INVP.nVertGrid;
numFields    = GLB_INVP.numFields;

%%%%%%%%%%% Get shear and bulk modulus from control array
shear_modulus = control(1:nVertGrid);
bulk_modulus = control(nVertGrid+1:end);

%%%%%%%%%%% evaluate strain field at the cubature points
data_at_dof = GLB_INVP.exp_state(GLB_INVP.mesh.d');
exp_strain_at_cub_points = zeros(3, numCubPoints, numCells);
intrepid_evaluate(exp_strain_at_cub_points, data_at_dof, GLB_INVP.Bmat);

%%%%%%%%%%% compute deviatoric strain field
exp_deviatoric_strain = zeros(3, numCubPoints, numCells);
intrepid_tensorMultiplyDataData(exp_deviatoric_strain, ...
    GLB_INVP.Ddev, exp_strain_at_cub_points);

%%%%%%%%%%% combine deviatoric strain with basis functions
exp_deviatoric_strain_times_phi = zeros(3, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( exp_deviatoric_strain_times_phi, ...
    exp_deviatoric_strain, GLB_INVP.transformed_val_at_cub_points );

%%%%%%%%%%% integrate matrix Bmat*phi*deviatoric_strain
exp_cell_dK_dev_matrices = zeros(numDof, numFields, numCells);
intrepid_integrate( exp_cell_dK_dev_matrices, ...
    exp_deviatoric_strain_times_phi, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% assemble matrix matrix Bmat*phi*volumetric_strain
exp_dK_dev_matrices = ...
    reshape(exp_cell_dK_dev_matrices, 1, numel(exp_cell_dK_dev_matrices));
exp_dK_dev = sparse(GLB_INVP.iIdxMix, GLB_INVP.jIdxMix, exp_dK_dev_matrices);

%%%%%%%%%%% compute volumetric strain field
exp_volumetric_strain = zeros(3, numCubPoints, numCells);
intrepid_tensorMultiplyDataData(exp_volumetric_strain, ...
    GLB_INVP.Dvol, exp_strain_at_cub_points);

%%%%%%%%%%% combine volumetric strain with weighted basis functions
exp_volumetric_strain_times_phi = zeros(3, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( exp_volumetric_strain_times_phi, ...
    exp_volumetric_strain, GLB_INVP.transformed_val_at_cub_points );

%%%%%%%%%%% integrate matrix Bmat*phi*volumetric_strain
exp_cell_dK_vol_matrices = zeros(numDof, numFields, numCells);
intrepid_integrate( exp_cell_dK_vol_matrices, ...
    exp_volumetric_strain_times_phi, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% assemble matrix matrix Bmat*phi*volumetric_strain
exp_dK_vol_matrices = ...
    reshape(exp_cell_dK_vol_matrices, 1, numel(exp_cell_dK_vol_matrices));
exp_dK_vol = sparse(GLB_INVP.iIdxMix, GLB_INVP.jIdxMix, exp_dK_vol_matrices);

%%

%%%% get shear and bulk modulus perturbations from dcontrol array
delta_shear_modulus = dcontrol(1:nVertGrid);
delta_bulk_modulus = dcontrol(nVertGrid+1:end);

%%%%%%%%%%% evaluate delta shear modulus at the cubature points
delta_shear_at_dof = delta_shear_modulus( GLB_INVP.mesh.t');
delta_shear_at_cub_points = zeros(numCubPoints, numCells);
intrepid_evaluate(delta_shear_at_cub_points, delta_shear_at_dof, ...
    GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% evaluate delta bulk modulus at the cubature points
delta_bulk_at_dof = delta_bulk_modulus( GLB_INVP.mesh.t');
delta_bulk_at_cub_points = zeros(numCubPoints, numCells);
intrepid_evaluate(delta_bulk_at_cub_points, delta_bulk_at_dof, ...
    GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% combine Bmat with delta shear modulus
delta_shear_times_Bmat = zeros(3, numCubPoints, numDof, numCells);
intrepid_scalarMultiplyDataField( delta_shear_times_Bmat, ...
    delta_shear_at_cub_points, GLB_INVP.Bmat);

%%%%%%%%%%% matrix-vector product, deviatoric coefficient mat. and fields
Ddev_times_shear_times_Bmat = zeros(3, numCubPoints, numDof, numCells);
intrepid_tensorMultiplyDataField( Ddev_times_shear_times_Bmat, ...
    GLB_INVP.Ddev, delta_shear_times_Bmat);

%%%%%%%%%%% integrate deviatoric stiffnes matrix
cell_dKdev_matrices = zeros(numDof, numDof, numCells);
intrepid_integrate(cell_dKdev_matrices, ...
    Ddev_times_shear_times_Bmat, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% build global deviatoric stiffness matrix
dKdev_matrices = reshape(cell_dKdev_matrices, 1, numel(cell_dKdev_matrices));
dKdev = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, dKdev_matrices);

%%%%%%%%%%% combine Bmat with bulk modulus
delta_bulk_times_Bmat = zeros(3, numCubPoints, numDof, numCells);
intrepid_scalarMultiplyDataField( delta_bulk_times_Bmat, ...
    delta_bulk_at_cub_points, GLB_INVP.Bmat);

%%%%%%%%%%% matrix-vector product, volumetric coefficient mat. and fields
Dvol_times_delta_bulk_times_Bmat = zeros(3, numCubPoints, numDof, numCells);
intrepid_tensorMultiplyDataField( Dvol_times_delta_bulk_times_Bmat, ...
    GLB_INVP.Dvol, delta_bulk_times_Bmat);

%%%%%%%%%%% integrate volumetric stiffnes matrix
cell_dKvol_matrices = zeros(numDof, numDof, numCells);
intrepid_integrate(cell_dKvol_matrices, ...
    Dvol_times_delta_bulk_times_Bmat, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% build global volumetric stiffness matrix
dKvol_matrices = reshape(cell_dKvol_matrices, 1, numel(cell_dKvol_matrices));
dKvol = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, dKvol_matrices);
%%
%%%%%%%%%%% evaluate shear modulus at the cubature points
shear_at_dof = shear_modulus( GLB_INVP.mesh.t');
shear_at_cub_points = zeros(numCubPoints, numCells);
intrepid_evaluate(shear_at_cub_points, shear_at_dof, ...
    GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% evaluate bulk modulus at the cubature points
bulk_at_dof = bulk_modulus( GLB_INVP.mesh.t');
bulk_at_cub_points = zeros(numCubPoints, numCells);
intrepid_evaluate(bulk_at_cub_points, bulk_at_dof, ...
    GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% combine Bmat with shear modulus
shear_times_Bmat = zeros(3, numCubPoints, numDof, numCells);
intrepid_scalarMultiplyDataField( shear_times_Bmat, ...
    shear_at_cub_points, GLB_INVP.Bmat);

%%%%%%%%%%% matrix-vector product, deviatoric coefficient mat. and fields
Ddev_times_shear_times_Bmat = zeros(3, numCubPoints, numDof, numCells);
intrepid_tensorMultiplyDataField( Ddev_times_shear_times_Bmat, ...
    GLB_INVP.Ddev, shear_times_Bmat);

%%%%%%%%%%% integrate deviatoric stiffnes matrix
cell_Kdev_matrices = zeros(numDof, numDof, numCells);
intrepid_integrate(cell_Kdev_matrices, ...
    Ddev_times_shear_times_Bmat, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% build global deviatoric stiffness matrix
Kdev_matrices = reshape(cell_Kdev_matrices, 1, numel(cell_Kdev_matrices));
Kdev = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, Kdev_matrices);

%%%%%%%%%%% combine Bmat with bulk modulus
bulk_times_Bmat = zeros(3, numCubPoints, numDof, numCells);
intrepid_scalarMultiplyDataField( bulk_times_Bmat, ...
    bulk_at_cub_points, GLB_INVP.Bmat);

%%%%%%%%%%% matrix-vector product, volumetric coefficient mat. and fields
Dvol_times_bulk_times_Bmat = zeros(3, numCubPoints, numDof, numCells);
intrepid_tensorMultiplyDataField( Dvol_times_bulk_times_Bmat, ...
    GLB_INVP.Dvol, bulk_times_Bmat);

%%%%%%%%%%% integrate volumetric stiffnes matrix
cell_Kvol_matrices = zeros(numDof, numDof, numCells);
intrepid_integrate(cell_Kvol_matrices, ...
    Dvol_times_bulk_times_Bmat, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% build global volumetric stiffness matrix
Kvol_matrices = reshape(cell_Kvol_matrices, 1, numel(cell_Kvol_matrices));
Kvol = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, Kvol_matrices);

%%%%%%%%%%% compute stiffness matrix
K = Kdev + Kvol;
%%

%%%%%%%%%%% compute potential contribution
EqualityConstraint = equalityConstraint();
[rhs] = EqualityConstraint.firstDerivativeWrtControl(GLB_INVP.state,control,dcontrol);
[du] = EqualityConstraint.applyInverseJacobianWrtState(GLB_INVP.state,control,rhs);
residual = K*GLB_INVP.exp_state - GLB_INVP.force;

upsilon = du'*residual;
alpha = (GLB_INVP.state'*residual - GLB_INVP.exp_state'*residual);
gamma = (GLB_INVP.state'*(dKdev*GLB_INVP.exp_state) - GLB_INVP.exp_state'*(dKdev*GLB_INVP.exp_state)) + ...
    (GLB_INVP.state'*(dKvol*GLB_INVP.exp_state) - GLB_INVP.exp_state'*(dKvol*GLB_INVP.exp_state));
data_misfit = GLB_INVP.exp_state - GLB_INVP.state;
dv = -upsilon*GLB_INVP.theta*data_misfit + ...
    gamma*GLB_INVP.theta*data_misfit + ...
    alpha*GLB_INVP.theta*du;

%%%%%%%%%%% compute regularization term
switch GLB_INVP.reg
    case{'tikhonov'}
        reg = [ GLB_INVP.beta * (GLB_INVP.Ms * delta_shear_modulus); ...
                GLB_INVP.beta * (GLB_INVP.Ms * delta_bulk_modulus) ];
    case{'TV'}
        S_bulk  = GLB_INVP.Ss * bulk_modulus;
        St_bulk = GLB_INVP.Ss' * bulk_modulus;
        S_shear  = GLB_INVP.Ss * shear_modulus;
        St_shear = GLB_INVP.Ss' * shear_modulus;
        reg = [ -0.5 * GLB_INVP.beta * ( ...
                 ( (shear_modulus' * S_shear + GLB_INVP.gamma)^(-3/2) ) * ...
                 ((St_shear'*delta_shear_modulus)*S_shear) ...
                 - ((1.0 / sqrt(shear_modulus' * S_shear + GLB_INVP.gamma)) * ...
                 (GLB_INVP.Ss * delta_shear_modulus)) );
                -0.5 * GLB_INVP.beta * ( ...
                 ( (bulk_modulus' * S_bulk + GLB_INVP.gamma)^(-3/2) ) * ...
                 ((St_bulk'*delta_bulk_modulus)*S_bulk) ...
                 - ((1.0 / sqrt(bulk_modulus' * S_bulk + GLB_INVP.gamma)) * ...
                 (GLB_INVP.Ss * delta_bulk_modulus)) ) ...
              ];
end

Lzu_times_du = [GLB_INVP.theta * upsilon * ((GLB_INVP.state'*exp_dK_dev) - (GLB_INVP.exp_state'*exp_dK_dev))'; ...
                GLB_INVP.theta * upsilon * ((GLB_INVP.state'*exp_dK_vol) - (GLB_INVP.exp_state'*exp_dK_vol))' ];
Lzz_times_dz = [gamma * GLB_INVP.theta * ((GLB_INVP.state'*exp_dK_dev) - (GLB_INVP.exp_state'*exp_dK_dev))'; ...
                gamma * GLB_INVP.theta * ((GLB_INVP.state'*exp_dK_vol) - (GLB_INVP.exp_state'*exp_dK_vol))' ];
Lzz_times_dz = Lzz_times_dz + reg;
Lzv_times_dv = EqualityConstraint.adjointFirstDerivativeWrtControl(GLB_INVP.state,control,dv);

output = Lzu_times_du + Lzz_times_dz + Lzv_times_dv;

end

end