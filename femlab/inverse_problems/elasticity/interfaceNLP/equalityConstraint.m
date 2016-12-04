function [Operators] = equalityConstraint()
Operators.solve=@(control)solve(control);
Operators.applyInverseJacobianWrtState=...
    @(state,control,rhs)applyInverseJacobianWrtState(state,control,rhs);
Operators.applyInverseAdjointJacobianWrtState=...
    @(state,control,rhs)applyInverseAdjointJacobianWrtState(state,control,rhs);
Operators.residual=@(state,control)residual(state,control);
Operators.firstDerivativeWrtState=...
    @(state,control,dcontrol)firstDerivativeWrtState(state,control,dcontrol);
Operators.firstDerivativeWrtControl=...
    @(state,control,dcontrol)firstDerivativeWrtControl(state,control,dcontrol);
Operators.adjointFirstDerivativeWrtState=...
    @(state,control,dual)adjointFirstDerivativeWrtState(state,control,dual);
Operators.adjointFirstDerivativeWrtControl=...
    @(state,control,dual)adjointFirstDerivativeWrtControl(state,control,dual);
Operators.secondDerivativeWrtStateState=...
    @(state,control,dual,dstate)secondDerivativeWrtStateState(state,control,dual,dstate);
Operators.secondDerivativeWrtStateControl=...
    @(state,control,dual,dcontrol)secondDerivativeWrtStateControl(state,control,dual,dcontrol);
Operators.secondDerivativeWrtControlState=...
    @(state,control,dual,dstate)secondDerivativeWrtControlState(state,control,dual,dstate);
Operators.secondDerivativeWrtControlControl=...
    @(state,control,dual,dcontrol)secondDerivativeWrtControlControl(state,control,dual,dcontrol);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [state] = solve(control)

global GLB_INVP;

spaceDim     = GLB_INVP.spaceDim;
numDof       = GLB_INVP.numDof;
numCubPoints = GLB_INVP.numCubPoints;
numCells     = GLB_INVP.numCells;
nVertGrid    = GLB_INVP.nVertGrid;

%%%%%%%%%%% get shear and bulk modulus from control array
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

%%%%%%%%%%% Apply Dirichlet conditions to rhs vector.
state = zeros(spaceDim*nVertGrid,1);
if( ~isempty(state) )
    state(unique(GLB_INVP.dirichlet)) = ...
        GLB_INVP.u_dirichlet( unique(GLB_INVP.dirichlet) );
    rhs = GLB_INVP.force - K * state;
end

%%%%%%%%%%% Solve system of equations
state(GLB_INVP.FreeNodes) = ...
    K(GLB_INVP.FreeNodes,GLB_INVP.FreeNodes) \ rhs(GLB_INVP.FreeNodes);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = applyInverseJacobianWrtState(state,control,rhs)

global GLB_INVP;

numDof       = GLB_INVP.numDof;
numCubPoints = GLB_INVP.numCubPoints;
numCells     = GLB_INVP.numCells;
nVertGrid    = GLB_INVP.nVertGrid;

%%%% Get shear and bulk modulus from control array
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

%%%%%%%%%%% Solve system of equations
output = zeros(size(state));
output(GLB_INVP.FreeNodes) = ...
    K(GLB_INVP.FreeNodes,GLB_INVP.FreeNodes) \ rhs(GLB_INVP.FreeNodes);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = applyInverseAdjointJacobianWrtState(state,control,rhs)

global GLB_INVP;

numDof       = GLB_INVP.numDof;
numCubPoints = GLB_INVP.numCubPoints;
numCells     = GLB_INVP.numCells;
nVertGrid    = GLB_INVP.nVertGrid;

%%%% Get shear and bulk modulus from control array
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

%%%%%%%%%%% Solve system of equations
output(GLB_INVP.FreeNodes) = ...
    K(GLB_INVP.FreeNodes,GLB_INVP.FreeNodes) \ rhs(GLB_INVP.FreeNodes);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = residual(state,control)

global GLB_INVP;

numDof       = GLB_INVP.numDof;
numCubPoints = GLB_INVP.numCubPoints;
numCells     = GLB_INVP.numCells;
nVertGrid    = GLB_INVP.nVertGrid;

%%%% Get shear and bulk modulus from control array
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

%%%%%%%%%%% Compute residual
output = K*state - GLB_INVP.force;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtState(state,control,dstate)

global GLB_INVP;

numDof       = GLB_INVP.numDof;
numCubPoints = GLB_INVP.numCubPoints;
numCells     = GLB_INVP.numCells;
nVertGrid    = GLB_INVP.nVertGrid;

%%%% Get shear and bulk modulus from control array
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

%%%%%%%%%%% Apply perturbation to matrix operator
output = K*dstate;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtControl(state,control,dcontrol)

global GLB_INVP;

numDof       = GLB_INVP.numDof;
numFields    = GLB_INVP.numFields;
numCubPoints = GLB_INVP.numCubPoints;
numCells     = GLB_INVP.numCells;
nVertGrid    = GLB_INVP.nVertGrid;

%%%% Get shear and bulk modulus perturbations from dcontrol array
delta_shear_modulus = dcontrol(1:nVertGrid);
delta_bulk_modulus = dcontrol(nVertGrid+1:end);

%%%%%%%%%%% get state solution at traingle's dof
state_at_dof = state(GLB_INVP.mesh.d');

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

%%%%%%%%%%% Apply perturbation to matrix operator
output = (dK_dev*delta_shear_modulus) + (dK_vol*delta_bulk_modulus);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = adjointFirstDerivativeWrtState(state,control,dual)

global GLB_INVP;

numDof       = GLB_INVP.numDof;
numCubPoints = GLB_INVP.numCubPoints;
numCells     = GLB_INVP.numCells;
nVertGrid    = GLB_INVP.nVertGrid;

%%%%%%%%%%% Get shear and bulk modulus from control array
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

%%%%%%%%%%% Apply perturbation to matrix operator
output = K*dual;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = adjointFirstDerivativeWrtControl(state,control,dual)

global GLB_INVP;

numDof       = GLB_INVP.numDof;
numFields    = GLB_INVP.numFields;
numCubPoints = GLB_INVP.numCubPoints;
numCells     = GLB_INVP.numCells;

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

%%%%%%%%%%% Apply dual to perturbed matrix operator
output = [(dK_dev')*dual; ...
          (dK_vol')*dual];

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateState(state,control,dual,dstate)
output=zeros(size(state));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateControl(state,control,dual,dcontrol)

global GLB_INVP;

numDof       = GLB_INVP.numDof;
numCubPoints = GLB_INVP.numCubPoints;
numCells     = GLB_INVP.numCells;
nVertGrid    = GLB_INVP.nVertGrid;

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

%%%%%%%%%%% Apply dual to matrix operator
output = dK*dual;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlState(state,control,dual,dstate)

global GLB_INVP;

numDof       = GLB_INVP.numDof;
numCubPoints = GLB_INVP.numCubPoints;
numCells     = GLB_INVP.numCells;
numFields    = GLB_INVP.numFields;

%%%%%%%%%%% evaluate dual strain field at cubature points
dual_at_dof = dual( GLB_INVP.mesh.d');
strain_dual_at_cub_points = zeros(3, numCubPoints, numCells);
intrepid_evaluate(strain_dual_at_cub_points,dual_at_dof, GLB_INVP.Bmat);

%%%%%%%%%%% compute dual deviatoric strain field
deviatoric_strain_dual = zeros(3, numCubPoints, numCells);
intrepid_tensorMultiplyDataData(deviatoric_strain_dual, ...
    GLB_INVP.Ddev, strain_dual_at_cub_points);

%%%%%%%%%%% combine transformed values with dual deviatoric strain
deviatoric_strain_dual_times_phi = ...
    zeros(3, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( deviatoric_strain_dual_times_phi, ...
    deviatoric_strain_dual, GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% integrate matrix Bmat*deviatoric_strain(dual)*phi
cell_dK_dev_matrices = zeros(numDof, numFields, numCells);
intrepid_integrate( cell_dK_dev_matrices, ...
    deviatoric_strain_dual_times_phi, ...
    GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% assemble matrix Bmat*deviatoric_strain(dual)*phi
dK_dev_matrices = reshape(cell_dK_dev_matrices, 1, numel(cell_dK_dev_matrices));
dK_dev = sparse(GLB_INVP.iIdxMix, GLB_INVP.jIdxMix, dK_dev_matrices);

%%%%%%%%%%% compute dual volumetric strain field
volumetric_strain_dual = zeros(3, numCubPoints, numCells);
intrepid_tensorMultiplyDataData(volumetric_strain_dual, ...
    GLB_INVP.Dvol, strain_dual_at_cub_points);

%%%%%%%%%%% combine transformed values with dual volumetric strain
volumetric_strain_dual_times_phi = ...
    zeros(3, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( volumetric_strain_dual_times_phi, ...
    volumetric_strain_dual, GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% integrate matrix Bmat*volumetric_strain(dual)*phi
cell_dK_vol_matrices = zeros(numDof, numFields, numCells);
intrepid_integrate( ...
    cell_dK_vol_matrices, volumetric_strain_dual_times_phi, ...
    GLB_INVP.weighted_Bmat, 'COMP_BLAS');

%%%%%%%%%%% assemble matrix Bmat*volumetric_strain(dual)*phi
dK_vol_matrices = ...
    reshape(cell_dK_vol_matrices, 1, numel(cell_dK_vol_matrices));
dK_vol = sparse(GLB_INVP.iIdxMix, GLB_INVP.jIdxMix, dK_vol_matrices);

%%%%%%%%%%% Apply dual to perturbed matrix operator
output = [(dK_dev')*dstate; ...
          (dK_vol')*dstate];

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlControl(state,control,dual,dcontrol)
output = zeros(size(control));
end
