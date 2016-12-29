function [Operators] = equalityConstraint()
% Solves
Operators.solve=@(control)solve(control);
Operators.applyInverseJacobianWrtState=...
    @(state,control,rhs)applyInverseJacobianWrtState(state,control,rhs);
Operators.applyInverseAdjointJacobianWrtState=...
    @(state,control,rhs)applyInverseAdjointJacobianWrtState(state,control,rhs);
Operators.residual=@(state,control)residual(state,control);
% First order derivatives
Operators.partialDerivativeState=...
    @(state,control,dcontrol)partialDerivativeState(state,control,dcontrol);
Operators.partialDerivativeControl=...
    @(state,control,dcontrol)partialDerivativeControl(state,control,dcontrol);
Operators.adjointPartialDerivativeState=...
    @(state,control,dual)adjointPartialDerivativeState(state,control,dual);
Operators.adjointPartialDerivativeControl=...
    @(state,control,dual)adjointPartialDerivativeControl(state,control,dual);
% Second order derivatives
Operators.partialDerivativeStateState=...
    @(state,control,dual,dstate)partialDerivativeStateState(state,control,dual,dstate);
Operators.partialDerivativeStateControl=...
    @(state,control,dual,dcontrol)partialDerivativeStateControl(state,control,dual,dcontrol);
Operators.partialDerivativeControlState=...
    @(state,control,dual,dstate)partialDerivativeControlState(state,control,dual,dstate);
Operators.partialDerivativeControlControl=...
    @(state,control,dual,dcontrol)partialDerivativeControlControl(state,control,dual,dcontrol);
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
GLB_INVP.NewK = K;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = applyInverseJacobianWrtState(state,control,rhs)

global GLB_INVP;

spaceDim      = GLB_INVP.spaceDim;
numFields     = GLB_INVP.numFields;
numCubPoints  = GLB_INVP.numCubPoints;
numCells      = GLB_INVP.numCells;

output = zeros(size(state));

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
cell_stiffness_matrices = reshape(cell_stiffness_matrices, 1, ...
    numel(cell_stiffness_matrices));
K = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, cell_stiffness_matrices);

%%%%%%%%%%% Solve system of equations
output(GLB_INVP.FreeNodes) = ...
    K(GLB_INVP.FreeNodes,GLB_INVP.FreeNodes) \ rhs(GLB_INVP.FreeNodes);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = applyInverseAdjointJacobianWrtState(state,control,rhs)

global GLB_INVP;

spaceDim      = GLB_INVP.spaceDim;
numFields     = GLB_INVP.numFields;
numCubPoints  = GLB_INVP.numCubPoints;
numCells      = GLB_INVP.numCells;

output = zeros(size(state));

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
cell_stiffness_matrices = reshape(cell_stiffness_matrices, 1, ...
    numel(cell_stiffness_matrices));
K = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, cell_stiffness_matrices);

%%%%%%%%%%% Solve system of equations
output(GLB_INVP.FreeNodes) = ...
    K(GLB_INVP.FreeNodes,GLB_INVP.FreeNodes) \ rhs(GLB_INVP.FreeNodes);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = residual(state,control)

global GLB_INVP;

spaceDim      = GLB_INVP.spaceDim;
numFields     = GLB_INVP.numFields;
numCubPoints  = GLB_INVP.numCubPoints;
numCells      = GLB_INVP.numCells;

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
cell_stiffness_matrices = reshape(cell_stiffness_matrices, 1, ...
    numel(cell_stiffness_matrices));
K = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, cell_stiffness_matrices);

%%%%%%%%%%% Compute residual
output = K*state - GLB_INVP.force;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeState(state,control,dstate)

global GLB_INVP;

spaceDim      = GLB_INVP.spaceDim;
numFields     = GLB_INVP.numFields;
numCubPoints  = GLB_INVP.numCubPoints;
numCells      = GLB_INVP.numCells;

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
cell_stiffness_matrices = reshape(cell_stiffness_matrices, 1, ...
    numel(cell_stiffness_matrices));
K = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, cell_stiffness_matrices);

%%%%%%%%%%% Apply perturbation to matrix operator
output = K*dstate;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeControl(state,control,dcontrol)

global GLB_INVP;

spaceDim      = GLB_INVP.spaceDim;
numFields     = GLB_INVP.numFields;
numCubPoints  = GLB_INVP.numCubPoints;
numCells      = GLB_INVP.numCells;

% evaluate material parameter k (diffusion coefficients) at the cubature points
dcontrol_at_dof = dcontrol( GLB_INVP.mesh.t');
dcontrol_at_cub_points = zeros(numCubPoints, numCells);
intrepid_evaluate(dcontrol_at_cub_points, dcontrol_at_dof, ...
    GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% combine transformed gradients with diffusion parameters
dcontrol_times_transformed_grad_at_cub_points = ...
    zeros(spaceDim, numCubPoints, numFields, numCells);
intrepid_scalarMultiplyDataField(dcontrol_times_transformed_grad_at_cub_points, ...
    dcontrol_at_cub_points, GLB_INVP.transformed_grad_at_cub_points);

%%%%%%%%%%% integrate stiffnes matrix
cell_stiffness_matrices = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_stiffness_matrices, ...
    dcontrol_times_transformed_grad_at_cub_points, ...
    GLB_INVP.weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global stiffness matrix
cell_stiffness_matrices = reshape(cell_stiffness_matrices, 1, ...
    numel(cell_stiffness_matrices));
dK = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, cell_stiffness_matrices);

%%%%%%%%%%% Apply perturbation to matrix operator
output = dK*state;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = adjointPartialDerivativeState(state,control,dual)

global GLB_INVP;

spaceDim      = GLB_INVP.spaceDim;
numFields     = GLB_INVP.numFields;
numCubPoints  = GLB_INVP.numCubPoints;
numCells      = GLB_INVP.numCells;

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
cell_stiffness_matrices = reshape(cell_stiffness_matrices, 1, ...
    numel(cell_stiffness_matrices));
K = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, cell_stiffness_matrices);

%%%%%%%%%%% Apply perturbation to matrix operator
output = K*dual;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = adjointPartialDerivativeControl(state,control,dual)

global GLB_INVP;

spaceDim      = GLB_INVP.spaceDim;
numFields     = GLB_INVP.numFields;
numCubPoints  = GLB_INVP.numCubPoints;
numCells      = GLB_INVP.numCells;

%%%%%%%%%%% evaluate gradient of u (state solution) at the cubature points
state_at_dof = state( GLB_INVP.mesh.t');
grad_u_at_cub_points = zeros(spaceDim, numCubPoints, numCells);
intrepid_evaluate(grad_u_at_cub_points, state_at_dof, ...
    GLB_INVP.transformed_grad_at_cub_points);

%%%%%%%%%%% combine transformed values with gradient of u 
grad_u_at_cub_points_times_transformed_val_at_cub_points = ...
    zeros(spaceDim, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( ...
    grad_u_at_cub_points_times_transformed_val_at_cub_points, ...
    grad_u_at_cub_points, GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% integrate stiffnes matrix phi_i*grad(u)*grad(phi_j)
matrices = zeros(numFields, numFields, numCells);
intrepid_integrate(matrices, ...
    grad_u_at_cub_points_times_transformed_val_at_cub_points, ...
    GLB_INVP.weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global stiffness matrix phi_i*grad(u)*grad(phi_j)
matrices = reshape(matrices, 1, numel(matrices));
dK = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, matrices);

%%%%%%%%%%% Apply dual to perturbed matrix operator
output = dual'*dK;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeStateState(state,control,dual,dstate)
output=zeros(size(state));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeStateControl(state,control,dual,dcontrol)

global GLB_INVP;

spaceDim      = GLB_INVP.spaceDim;
numFields     = GLB_INVP.numFields;
numCubPoints  = GLB_INVP.numCubPoints;
numCells      = GLB_INVP.numCells;

% evaluate material parameter k (diffusion coefficients) at the cubature points
dcontrol_at_dof = dcontrol( GLB_INVP.mesh.t');
dcontrol_at_cub_points = zeros(numCubPoints, numCells);
intrepid_evaluate(dcontrol_at_cub_points, dcontrol_at_dof, ...
    GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% combine transformed gradients with diffusion parameters
dcontrol_times_transformed_grad_at_cub_points = ...
    zeros(spaceDim, numCubPoints, numFields, numCells);
intrepid_scalarMultiplyDataField(dcontrol_times_transformed_grad_at_cub_points, ...
    dcontrol_at_cub_points, GLB_INVP.transformed_grad_at_cub_points);

%%%%%%%%%%% integrate stiffnes matrix
cell_stiffness_matrices = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_stiffness_matrices, ...
    dcontrol_times_transformed_grad_at_cub_points, ...
    GLB_INVP.weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global stiffness matrix
matrix=reshape(cell_stiffness_matrices, 1, numel(cell_stiffness_matrices));
dK = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, matrix);

%%%%%%%%%%% Apply perturbation to matrix operator
output = dK*dual;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeControlState(state,control,dual,dstate)

global GLB_INVP;

spaceDim      = GLB_INVP.spaceDim;
numFields     = GLB_INVP.numFields;
numCubPoints  = GLB_INVP.numCubPoints;
numCells      = GLB_INVP.numCells;

%%%%%%%%%%% evaluate gradient of u (state solution) at the cubature points
dual_at_dof = dual( GLB_INVP.mesh.t');
grad_dual_at_cub_points = zeros(spaceDim, numCubPoints, numCells);
intrepid_evaluate(grad_dual_at_cub_points, dual_at_dof, ...
    GLB_INVP.transformed_grad_at_cub_points);

%%%%%%%%%%% combine transformed values with gradient of u 
grad_dual_at_cub_points_times_transformed_val_at_cub_points = ...
    zeros(spaceDim, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( ...
    grad_dual_at_cub_points_times_transformed_val_at_cub_points, ...
    grad_dual_at_cub_points, GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% integrate stiffnes matrix phi_i*grad(u)*grad(phi_j)
matrices = zeros(numFields, numFields, numCells);
intrepid_integrate(matrices, ...
    grad_dual_at_cub_points_times_transformed_val_at_cub_points, ...
    GLB_INVP.weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global stiffness matrix phi_i*grad(u)*grad(phi_j)
matrices = reshape(matrices, 1, numel(matrices));
dK = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, matrices);

%%%%%%%%%%% Apply dual to perturbed matrix operator
output = dstate'*dK;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeControlControl(state,control,dual,dcontrol)
output = zeros(size(control));
end
