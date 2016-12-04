function [Operators] = objectiveFunction()
Operators.evaluate=@(state,control)evaluate(state,control);
Operators.firstDerivativeWrtState=...
    @(state,control)firstDerivativeWrtState(state,control);
Operators.firstDerivativeWrtControl=...
    @(state,control)firstDerivativeWrtControl(state,control);
Operators.secondDerivativeWrtStateState=...
    @(state,control,dstate)secondDerivativeWrtStateState(state,control,dstate);
Operators.secondDerivativeWrtStateControl=...
    @(state,control,dcontrol)secondDerivativeWrtStateControl(state,control,dcontrol);
Operators.secondDerivativeWrtControlState=...
    @(state,control,dstate)secondDerivativeWrtControlState(state,control,dstate);
Operators.secondDerivativeWrtControlControl=...
    @(state,control,dcontrol)secondDerivativeWrtControlControl(state,control,dcontrol);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = evaluate(state,control)

global GLB_INVP;

numDof       = GLB_INVP.numDof;
numCubPoints = GLB_INVP.numCubPoints;
numCells     = GLB_INVP.numCells;
nVertGrid    = GLB_INVP.nVertGrid;

%%%% Get shear and bulk modulus ffom control array
shear_modulus = control(1:nVertGrid);
bulk_modulus = control(nVertGrid+1:end);

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

%%%% compute potential energy misfit
potential = 0.5 * GLB_INVP.theta * ...
    (state'*(K*state) - GLB_INVP.exp_state'*(K*GLB_INVP.exp_state)) * ...
    (state'*(K*state) - GLB_INVP.exp_state'*(K*GLB_INVP.exp_state));

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

output = potential + reg;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtControl(state,control)

global GLB_INVP;

numDof       = GLB_INVP.numDof;
numCubPoints = GLB_INVP.numCubPoints;
numCells     = GLB_INVP.numCells;
nVertGrid    = GLB_INVP.nVertGrid;
numFields    = GLB_INVP.numFields;

%%%% Get shear and bulk modulus ffom control array
shear_modulus = control(1:nVertGrid);
bulk_modulus = control(nVertGrid+1:end);

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
%%%%%%%%%%% evaluate strain field at the cubature points
state_at_dof = state(GLB_INVP.mesh.d');
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
state_dK_dev = sparse(GLB_INVP.iIdxMix, GLB_INVP.jIdxMix, dK_dev_matrices);

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
state_dK_vol = sparse(GLB_INVP.iIdxMix, GLB_INVP.jIdxMix, dK_vol_matrices);

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
%%%%%%%%%%% potential contribution 
alpha = (state'*(K*state) - GLB_INVP.exp_state'*(K*GLB_INVP.exp_state));
potential = [ GLB_INVP.theta * ...
               (alpha*((state'*state_dK_dev) - ...
               (GLB_INVP.exp_state'*exp_dK_dev)))'; ...
              GLB_INVP.theta * ...
               (alpha*((state'*state_dK_vol) - ...
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

output = potential + reg;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtState(state,control)

global GLB_INVP;

numDof       = GLB_INVP.numDof;
numCubPoints = GLB_INVP.numCubPoints;
numCells     = GLB_INVP.numCells;
nVertGrid    = GLB_INVP.nVertGrid;

%%%% Get shear and bulk modulus ffom control array
shear_modulus = control(1:nVertGrid);
bulk_modulus = control(nVertGrid+1:end);

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

%%%% compute alpha constant
alpha = (state'*(K*state) - GLB_INVP.exp_state'*(K*GLB_INVP.exp_state));

%%%% compute derivative constribution
output = 2*alpha*GLB_INVP.theta*K*state;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateState(state,control,dstate)

global GLB_INVP;

numDof       = GLB_INVP.numDof;
numCubPoints = GLB_INVP.numCubPoints;
numCells     = GLB_INVP.numCells;
nVertGrid    = GLB_INVP.nVertGrid;

%%%% Get shear and bulk modulus ffom control array
shear_modulus = control(1:nVertGrid);
bulk_modulus = control(nVertGrid+1:end);

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

%%%% compute alpha constant
alpha = (state'*(K*state) - GLB_INVP.exp_state'*(K*GLB_INVP.exp_state));

%%%% compute output
beta = state'*(K*dstate);
output = 2*GLB_INVP.theta*(alpha*(K*dstate) + (2*beta*(K*state)));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateControl(state,control,dcontrol)

global GLB_INVP;

numDof       = GLB_INVP.numDof;
numCubPoints = GLB_INVP.numCubPoints;
numCells     = GLB_INVP.numCells;
nVertGrid    = GLB_INVP.nVertGrid;

%%%% Get shear and bulk modulus ffom control array
shear_modulus = control(1:nVertGrid);
bulk_modulus = control(nVertGrid+1:end);

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

%%%%%%%%%%% compute stiffness matrix
dK = dKdev + dKvol;

%%%% compute alpha and beta constants
beta = (state'*(dK*state) - GLB_INVP.exp_state'*(dK*GLB_INVP.exp_state));
alpha = (state'*(K*state) - GLB_INVP.exp_state'*(K*GLB_INVP.exp_state));

%%%% compute output 
output = 2*GLB_INVP.theta*((alpha*(dK*state)) + (beta*(K*state)));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlState(state,control,dstate)

global GLB_INVP;

numDof       = GLB_INVP.numDof;
numCubPoints = GLB_INVP.numCubPoints;
numCells     = GLB_INVP.numCells;
nVertGrid    = GLB_INVP.nVertGrid;
numFields    = GLB_INVP.numFields;

%%%% Get shear and bulk modulus ffom control array
shear_modulus = control(1:nVertGrid);
bulk_modulus = control(nVertGrid+1:end);

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

%%%%%%%%%%% evaluate strain field at the cubature points
state_at_dof = state(GLB_INVP.mesh.d');
strain_at_cub_points = zeros(3, numCubPoints, numCells);
intrepid_evaluate(strain_at_cub_points, state_at_dof, GLB_INVP.Bmat);

%%%%%%%%%%% compute deviatoric strain field
state_deviatoric_strain = zeros(3, numCubPoints, numCells);
intrepid_tensorMultiplyDataData(state_deviatoric_strain, ...
    GLB_INVP.Ddev, strain_at_cub_points);

%%%%%%%%%%% combine deviatoric strain with basis functions
state_deviatoric_strain_times_phi = zeros(3, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( state_deviatoric_strain_times_phi, ...
    state_deviatoric_strain, GLB_INVP.transformed_val_at_cub_points );

%%%%%%%%%%% integrate matrix Bmat*phi*deviatoric_strain
state_cell_dK_dev_matrices = zeros(numDof, numFields, numCells);
intrepid_integrate( state_cell_dK_dev_matrices, ...
    state_deviatoric_strain_times_phi, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% assemble matrix matrix Bmat*phi*volumetric_strain
state_dK_dev_matrices = ...
    reshape(state_cell_dK_dev_matrices, 1, numel(state_cell_dK_dev_matrices));
state_dK_dev = sparse(GLB_INVP.iIdxMix, GLB_INVP.jIdxMix, state_dK_dev_matrices);

%%%%%%%%%%% compute volumetric strain field
state_volumetric_strain = zeros(3, numCubPoints, numCells);
intrepid_tensorMultiplyDataData(state_volumetric_strain, ...
    GLB_INVP.Dvol, strain_at_cub_points);

%%%%%%%%%%% combine volumetric strain with weighted basis functions
state_volumetric_strain_times_phi = zeros(3, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( state_volumetric_strain_times_phi, ...
    state_volumetric_strain, GLB_INVP.transformed_val_at_cub_points );

%%%%%%%%%%% integrate matrix Bmat*phi*volumetric_strain
state_cell_dK_vol_matrices = zeros(numDof, numFields, numCells);
intrepid_integrate( state_cell_dK_vol_matrices, ...
    state_volumetric_strain_times_phi, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% assemble matrix matrix Bmat*phi*volumetric_strain
state_dK_vol_matrices = ...
    reshape(state_cell_dK_vol_matrices, 1, numel(state_cell_dK_vol_matrices));
state_dK_vol = sparse(GLB_INVP.iIdxMix, GLB_INVP.jIdxMix, state_dK_vol_matrices);

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

%%%% compute alpha and beta constants
beta = state'*(K*dstate);
alpha = (state'*(K*state) - GLB_INVP.exp_state'*(K*GLB_INVP.exp_state));

%%%% compute output
output = [ 2*GLB_INVP.theta*(alpha*(dstate'*state_dK_dev) + ...
            (beta*(state'*state_dK_dev - GLB_INVP.exp_state'*exp_dK_dev)))'; ...
           2*GLB_INVP.theta*(alpha*(dstate'*state_dK_vol) + ...
            (beta*(state'*state_dK_vol - GLB_INVP.exp_state'*exp_dK_vol)))' ];

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlControl(state,control,dcontrol)

global GLB_INVP;

numDof       = GLB_INVP.numDof;
numCubPoints = GLB_INVP.numCubPoints;
numCells     = GLB_INVP.numCells;
nVertGrid    = GLB_INVP.nVertGrid;
numFields    = GLB_INVP.numFields;

%%%%%%%%%%% Get shear and bulk modulus from control array
shear_modulus = control(1:nVertGrid);
bulk_modulus = control(nVertGrid+1:end);

%%%%%%%%%%% get shear and bulk modulus perturbations from dcontrol array
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
dK_dev_matrices = reshape(cell_dKdev_matrices, 1, numel(cell_dKdev_matrices));
dK_dev = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, dK_dev_matrices);

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
dK_vol_matrices = reshape(cell_dKvol_matrices, 1, numel(cell_dKvol_matrices));
dK_vol = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, dK_vol_matrices);

%%%%%%%%%%% evaluate strain field at the cubature points
state_at_dof = state(GLB_INVP.mesh.d');
strain_at_cub_points = zeros(3, numCubPoints, numCells);
intrepid_evaluate(strain_at_cub_points, state_at_dof, GLB_INVP.Bmat);

%%%%%%%%%%% compute deviatoric strain field
state_deviatoric_strain = zeros(3, numCubPoints, numCells);
intrepid_tensorMultiplyDataData(state_deviatoric_strain, ...
    GLB_INVP.Ddev, strain_at_cub_points);

%%%%%%%%%%% combine deviatoric strain with basis functions
state_deviatoric_strain_times_phi = zeros(3, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( state_deviatoric_strain_times_phi, ...
    state_deviatoric_strain, GLB_INVP.transformed_val_at_cub_points );

%%%%%%%%%%% integrate matrix Bmat*phi*deviatoric_strain
state_cell_dK_dev_matrices = zeros(numDof, numFields, numCells);
intrepid_integrate( state_cell_dK_dev_matrices, ...
    state_deviatoric_strain_times_phi, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% assemble matrix matrix Bmat*phi*volumetric_strain
state_dK_dev_matrices = ...
    reshape(state_cell_dK_dev_matrices, 1, numel(state_cell_dK_dev_matrices));
state_dK_dev = sparse(GLB_INVP.iIdxMix, GLB_INVP.jIdxMix, state_dK_dev_matrices);

%%%%%%%%%%% compute volumetric strain field
state_volumetric_strain = zeros(3, numCubPoints, numCells);
intrepid_tensorMultiplyDataData(state_volumetric_strain, ...
    GLB_INVP.Dvol, strain_at_cub_points);

%%%%%%%%%%% combine volumetric strain with weighted basis functions
state_volumetric_strain_times_phi = zeros(3, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( state_volumetric_strain_times_phi, ...
    state_volumetric_strain, GLB_INVP.transformed_val_at_cub_points );

%%%%%%%%%%% integrate matrix Bmat*phi*volumetric_strain
state_cell_dK_vol_matrices = zeros(numDof, numFields, numCells);
intrepid_integrate( state_cell_dK_vol_matrices, ...
    state_volumetric_strain_times_phi, GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% assemble matrix matrix Bmat*phi*volumetric_strain
state_dK_vol_matrices = ...
    reshape(state_cell_dK_vol_matrices, 1, numel(state_cell_dK_vol_matrices));
state_dK_vol = sparse(GLB_INVP.iIdxMix, GLB_INVP.jIdxMix, state_dK_vol_matrices);

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

%%%%%%%%%%% compute beta constant
beta = (state'*(dK_dev*state) - GLB_INVP.exp_state'*(dK_dev*GLB_INVP.exp_state)) + ...
    (state'*(dK_vol*state) - GLB_INVP.exp_state'*(dK_vol*GLB_INVP.exp_state));

%%%%%%%%%%% compute potential contribution
potential = [beta * GLB_INVP.theta * ...
              ((state'*state_dK_dev) - (GLB_INVP.exp_state'*exp_dK_dev))'; ...
             beta * GLB_INVP.theta * ...
              ((state'*state_dK_vol) - (GLB_INVP.exp_state'*exp_dK_vol))' ];

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

output = potential + reg;

end
