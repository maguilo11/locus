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

output = zeros(size(state));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = applyInverseAdjointJacobianWrtState(state,control,rhs)

output = zeros(size(state));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = residual(state,control)

output = zeros(size(state));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtState(state,control,dstate)

output = zeros(size(state));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtControl(state,control,dcontrol)

output = zeros(size(state));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = adjointFirstDerivativeWrtState(state,control,dual)

output = zeros(size(state));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = adjointFirstDerivativeWrtControl(state,control,dual)

output = zeros(size(control));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateState(state,control,dual,dstate)

output=zeros(size(state));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateControl(state,control,dual,dcontrol)

output = zeros(size(state));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlState(state,control,dual,dstate)

output = zeros(size(control));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlControl(state,control,dual,dcontrol)

output = zeros(size(control));

end
