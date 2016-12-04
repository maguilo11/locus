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

spaceDim      = GLB_INVP.spaceDim;
nVertGrid     = GLB_INVP.nVertGrid;
numFields     = GLB_INVP.numFields;
numCubPoints  = GLB_INVP.numCubPoints;
numCells      = GLB_INVP.numCells;

%%%%%%%%%%% Initialization of lhs (i.e. solution) vector.
state = zeros(nVertGrid,1);

% evaluate material parameter k (diffusion coefficients) at the cubature points
control_at_dof = control( GLB_INVP.mesh.t');
control_at_cub_points = zeros(numCubPoints, numCells);
intrepid_evaluate(control_at_cub_points, control_at_dof, ...
    GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% combine transformed gradients with diffusion parameters
control_times_transformed_grad_at_cub_points = ...
    zeros(spaceDim, numCubPoints, numFields, numCells);
intrepid_scalarMultiplyDataField(control_times_transformed_grad_at_cub_points, ...
    control_at_cub_points, GLB_INVP.transformed_grad_at_cub_points);

%%%%%%%%%%% integrate stiffnes matrix
cell_stiffness_matrices = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_stiffness_matrices, ...
    control_times_transformed_grad_at_cub_points, ...
    GLB_INVP.weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global stiffness matrix
matrix = reshape(cell_stiffness_matrices, 1, numel(cell_stiffness_matrices));
K = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, matrix);

%%%%%%%%%%% Apply Dirichlet conditions to rhs vector.
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

output = zeros(size(state));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlControl(state,control,dual,dcontrol)

output = zeros(size(control));

end
