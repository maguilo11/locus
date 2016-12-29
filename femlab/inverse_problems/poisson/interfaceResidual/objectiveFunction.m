function [Operators] = objectiveFunction()
Operators.value=@(state,control)value(state,control);
% First order derivatives
Operators.partialDerivativeState=...
    @(state,control)partialDerivativeState(state,control);
Operators.partialDerivativeControl=...
    @(state,control)partialDerivativeControl(state,control);
% Second order derivatives
Operators.partialDerivativeStateState=...
    @(state,control,dstate)partialDerivativeStateState(state,control,dstate);
Operators.partialDerivativeStateControl=...
    @(state,control,dcontrol)partialDerivativeStateControl(state,control,dcontrol);
Operators.partialDerivativeControlState=...
    @(state,control,dstate)partialDerivativeControlState(state,control,dstate);
Operators.partialDerivativeControlControl=...
    @(state,control,dcontrol)partialDerivativeControlControl(state,control,dcontrol);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = value(state,control)

global GLB_INVP;

spaceDim     = GLB_INVP.spaceDim;
numCells     = GLB_INVP.numCells;
numFields    = GLB_INVP.numFields;
numCubPoints = GLB_INVP.numCubPoints;

%%

%%%%%%%%%%% evaluate control at the cubature points
control_at_dof = control( GLB_INVP.mesh.t');
control_at_cub_points = zeros(numCubPoints, numCells);
intrepid_evaluate(control_at_cub_points, control_at_dof, ...
    GLB_INVP.transformed_val_at_cub_points);

%%%% combine transformed gradients with diffusion parameters
control_times_transformed_grad_at_cub_points = ...
    zeros(spaceDim, numCubPoints, numFields, numCells);
intrepid_scalarMultiplyDataField(control_times_transformed_grad_at_cub_points, ...
    control_at_cub_points, GLB_INVP.transformed_grad_at_cub_points);

%%%% integrate stiffnes matrix
cell_stiffness_matrices = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_stiffness_matrices, ...
    control_times_transformed_grad_at_cub_points, ...
    GLB_INVP.weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%% build global stiffness matrix
matrix = reshape(cell_stiffness_matrices, 1, numel(cell_stiffness_matrices));
K = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, matrix);

%%

%%%% compute potential energy misfit
residual = K*GLB_INVP.exp_state - GLB_INVP.force;
potential = 0.5 * GLB_INVP.theta * (state'*residual - GLB_INVP.exp_state'*residual) * ...
    (state'*residual - GLB_INVP.exp_state'*residual);

%%%% compute regularization term
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = 0.5 * GLB_INVP.beta * (control'*(GLB_INVP.Ms*control));
    case{'TV'}
        reg = 0.5*GLB_INVP.beta * ...
            sqrt( (control' * (GLB_INVP.S * control)) + GLB_INVP.gamma );
end

output = potential + reg;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeControl(state,control)

global GLB_INVP;

spaceDim     = GLB_INVP.spaceDim;
numCells     = GLB_INVP.numCells;
numFields    = GLB_INVP.numFields;
numCubPoints = GLB_INVP.numCubPoints;

%%

%%%%%%%%%%% evaluate control at the cubature points
control_at_dof = control( GLB_INVP.mesh.t');
control_at_cub_points = zeros(numCubPoints, numCells);
intrepid_evaluate(control_at_cub_points, control_at_dof, ...
    GLB_INVP.transformed_val_at_cub_points);

%%%% combine transformed gradients with diffusion parameters
control_times_transformed_grad_at_cub_points = ...
    zeros(spaceDim, numCubPoints, numFields, numCells);
intrepid_scalarMultiplyDataField(control_times_transformed_grad_at_cub_points, ...
    control_at_cub_points, GLB_INVP.transformed_grad_at_cub_points);

%%%% integrate stiffnes matrix
cell_stiffness_matrices = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_stiffness_matrices, ...
    control_times_transformed_grad_at_cub_points, ...
    GLB_INVP.weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%% build global stiffness matrix
matrix = reshape(cell_stiffness_matrices, 1, numel(cell_stiffness_matrices));
K = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, matrix);

%%
%%%%%%%%%%% evaluate experimental strain field at the cubature points
data_at_dof = GLB_INVP.exp_state( GLB_INVP.mesh.t');
grad_data_at_cub_points = zeros(spaceDim, numCubPoints, numCells);
intrepid_evaluate(grad_data_at_cub_points, data_at_dof, ...
    GLB_INVP.transformed_grad_at_cub_points);

%%%%%%%%%%% combine transformed values with gradient of u 
grad_data_at_cub_points_times_transformed_val_at_cub_points = ...
    zeros(spaceDim, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( ...
    grad_data_at_cub_points_times_transformed_val_at_cub_points, ...
    grad_data_at_cub_points, GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% integrate stiffnes matrix phi_i*grad(u)*grad(phi_j)
cell_stiff_matrices_data = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_stiff_matrices_data, ...
    grad_data_at_cub_points_times_transformed_val_at_cub_points, ...
    GLB_INVP.weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global stiffness matrix phi_i*grad(u)*grad(phi_j)
matrix = ...
    reshape(cell_stiff_matrices_data, 1, numel(cell_stiff_matrices_data));
dK_data = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, matrix);

%%

%%%%%%%%%%% potential contribution 
residual = K*GLB_INVP.exp_state - GLB_INVP.force;
alpha = (state'*residual - GLB_INVP.exp_state'*residual);
residual_misfit = (GLB_INVP.theta * alpha * ((state'*dK_data) - ...
    (GLB_INVP.exp_state'*dK_data)))';

%%%%%%%%%%% compute regularization term
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = GLB_INVP.beta * (GLB_INVP.Ms*control);
    case{'TV'}
        reg = GLB_INVP.beta * 0.5 * ...
            ( 1.0 / sqrt( control' * (GLB_INVP.S * control) + ...
            GLB_INVP.gamma ) ) * (GLB_INVP.S * control);
end

output = residual_misfit + reg;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeState(state,control)

global GLB_INVP;

spaceDim     = GLB_INVP.spaceDim;
numCells     = GLB_INVP.numCells;
numFields    = GLB_INVP.numFields;
numCubPoints = GLB_INVP.numCubPoints;

%%%% evaluate diffusion coefficients at the cubature points
control_at_dof = control( GLB_INVP.mesh.t');
control_at_cub_points = zeros(numCubPoints, numCells);
intrepid_evaluate(control_at_cub_points, control_at_dof, ...
    GLB_INVP.transformed_val_at_cub_points);

%%%% combine transformed gradients with diffusion parameters
control_times_transformed_grad_at_cub_points = ...
    zeros(spaceDim, numCubPoints, numFields, numCells);
intrepid_scalarMultiplyDataField(control_times_transformed_grad_at_cub_points, ...
    control_at_cub_points, GLB_INVP.transformed_grad_at_cub_points);

%%%% integrate stiffnes matrix
cell_stiffness_matrices = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_stiffness_matrices, ...
    control_times_transformed_grad_at_cub_points, ...
    GLB_INVP.weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%% build global stiffness matrix
matrix = reshape(cell_stiffness_matrices, 1, numel(cell_stiffness_matrices));
K = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, matrix);

%%

%%%% compute alpha constant
residual = K*GLB_INVP.exp_state - GLB_INVP.force;
alpha = (state'*residual - GLB_INVP.exp_state'*residual);

%%%% compute derivative constribution
output = GLB_INVP.theta * alpha*residual;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeStateState(state,control,dstate)

global GLB_INVP;

spaceDim     = GLB_INVP.spaceDim;
numCells     = GLB_INVP.numCells;
numFields    = GLB_INVP.numFields;
numCubPoints = GLB_INVP.numCubPoints;

%%%% evaluate diffusion coefficients at the cubature points
control_at_dof = control( GLB_INVP.mesh.t');
control_at_cub_points = zeros(numCubPoints, numCells);
intrepid_evaluate(control_at_cub_points, control_at_dof, ...
    GLB_INVP.transformed_val_at_cub_points);

%%%% combine transformed gradients with diffusion parameters
control_times_transformed_grad_at_cub_points = ...
    zeros(spaceDim, numCubPoints, numFields, numCells);
intrepid_scalarMultiplyDataField(control_times_transformed_grad_at_cub_points, ...
    control_at_cub_points, GLB_INVP.transformed_grad_at_cub_points);

%%%% integrate stiffnes matrix
cell_stiffness_matrices = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_stiffness_matrices, ...
    control_times_transformed_grad_at_cub_points, ...
    GLB_INVP.weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%% build global stiffness matrix
matrix = reshape(cell_stiffness_matrices, 1, numel(cell_stiffness_matrices));
K = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, matrix);

%%

%%%% compute alpha constant
residual = K*GLB_INVP.exp_state - GLB_INVP.force;

%%%% compute output
beta = dstate'*residual;
output = GLB_INVP.theta * beta * residual;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeStateControl(state,control,dcontrol)

global GLB_INVP;

spaceDim     = GLB_INVP.spaceDim;
numCells     = GLB_INVP.numCells;
numFields    = GLB_INVP.numFields;
numCubPoints = GLB_INVP.numCubPoints;

%%

%%%% evaluate diffusion coefficients at the cubature points
control_at_dof = control( GLB_INVP.mesh.t');
control_at_cub_points = zeros(numCubPoints, numCells);
intrepid_evaluate(control_at_cub_points, control_at_dof, ...
    GLB_INVP.transformed_val_at_cub_points);

%%%% combine transformed gradients with diffusion parameters
control_times_transformed_grad_at_cub_points = ...
    zeros(spaceDim, numCubPoints, numFields, numCells);
intrepid_scalarMultiplyDataField(control_times_transformed_grad_at_cub_points, ...
    control_at_cub_points, GLB_INVP.transformed_grad_at_cub_points);

%%%% integrate stiffnes matrix
cell_stiffness_matrices = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_stiffness_matrices, ...
    control_times_transformed_grad_at_cub_points, ...
    GLB_INVP.weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%% build global stiffness matrix
matrix = reshape(cell_stiffness_matrices, 1, numel(cell_stiffness_matrices));
K = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, matrix);

%%

% evaluate diffusion coefficients at the cubature points
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

%%

%%%% compute alpha and beta constants
residual = K*GLB_INVP.exp_state - GLB_INVP.force;
alpha = (state'*residual - GLB_INVP.exp_state'*residual);
beta = (state'*dK*GLB_INVP.exp_state) - (GLB_INVP.exp_state'*dK*GLB_INVP.exp_state);

%%%% compute output 
output = GLB_INVP.theta * (alpha*(dK*GLB_INVP.exp_state) + (beta*residual));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeControlState(state,control,dstate)

global GLB_INVP;

spaceDim     = GLB_INVP.spaceDim;
numCells     = GLB_INVP.numCells;
numFields    = GLB_INVP.numFields;
numCubPoints = GLB_INVP.numCubPoints;

%%

%%%% evaluate diffusion coefficients at the cubature points
control_at_dof = control( GLB_INVP.mesh.t');
control_at_cub_points = zeros(numCubPoints, numCells);
intrepid_evaluate(control_at_cub_points, control_at_dof, ...
    GLB_INVP.transformed_val_at_cub_points);

%%%% combine transformed gradients with diffusion parameters
control_times_transformed_grad_at_cub_points = ...
    zeros(spaceDim, numCubPoints, numFields, numCells);
intrepid_scalarMultiplyDataField(control_times_transformed_grad_at_cub_points, ...
    control_at_cub_points, GLB_INVP.transformed_grad_at_cub_points);

%%%% integrate stiffnes matrix
cell_stiffness_matrices = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_stiffness_matrices, ...
    control_times_transformed_grad_at_cub_points, ...
    GLB_INVP.weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%% build global stiffness matrix
matrix = reshape(cell_stiffness_matrices, 1, numel(cell_stiffness_matrices));
K = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, matrix);

%%

%%%%%%%%%%% evaluate gradient of state at the cubature points
data_at_dof = GLB_INVP.exp_state( GLB_INVP.mesh.t');
grad_data_at_cub_points = zeros(spaceDim, numCubPoints, numCells);
intrepid_evaluate(grad_data_at_cub_points, data_at_dof, ...
    GLB_INVP.transformed_grad_at_cub_points);

%%%%%%%%%%% combine transformed values with gradient of u 
grad_data_at_cub_points_times_transformed_val_at_cub_points = ...
    zeros(spaceDim, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( ...
    grad_data_at_cub_points_times_transformed_val_at_cub_points, ...
    grad_data_at_cub_points, GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% integrate stiffnes matrix phi_i*grad(u)*grad(phi_j)
cell_stiff_matrices_data = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_stiff_matrices_data, ...
    grad_data_at_cub_points_times_transformed_val_at_cub_points, ...
    GLB_INVP.weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global stiffness matrix phi_i*grad(u)*grad(phi_j)
matrix = ...
    reshape(cell_stiff_matrices_data, 1, numel(cell_stiff_matrices_data));
dK_data = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, matrix);

%%

%%%% compute alpha and beta constants
residual = K*GLB_INVP.exp_state - GLB_INVP.force;
beta = dstate'*residual;
alpha = state'*residual - GLB_INVP.exp_state'*residual;

%%%% compute output
output = GLB_INVP.theta * (alpha*(dstate'*dK_data) + ...
    (beta*(state'*dK_data - GLB_INVP.exp_state'*dK_data)))';

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeControlControl(state,control,dcontrol)

global GLB_INVP;

spaceDim     = GLB_INVP.spaceDim;
numCells     = GLB_INVP.numCells;
numFields    = GLB_INVP.numFields;
numCubPoints = GLB_INVP.numCubPoints;

%%

%%%%%%%%%%% evaluate perturbed diffusion coefficients at the cubature points
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

%%

%%%%%%%%%%% evaluate gradient of state at the cubature points
data_at_dof = GLB_INVP.exp_state( GLB_INVP.mesh.t');
grad_data_at_cub_points = zeros(spaceDim, numCubPoints, numCells);
intrepid_evaluate(grad_data_at_cub_points, data_at_dof, ...
    GLB_INVP.transformed_grad_at_cub_points);

%%%%%%%%%%% combine transformed values with gradient of u 
grad_data_at_cub_points_times_transformed_val_at_cub_points = ...
    zeros(spaceDim, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( ...
    grad_data_at_cub_points_times_transformed_val_at_cub_points, ...
    grad_data_at_cub_points, GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% integrate stiffnes matrix phi_i*grad(u)*grad(phi_j)
cell_stiff_matrices_data = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_stiff_matrices_data, ...
    grad_data_at_cub_points_times_transformed_val_at_cub_points, ...
    GLB_INVP.weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global stiffness matrix phi_i*grad(u)*grad(phi_j)
matrix = ...
    reshape(cell_stiff_matrices_data, 1, numel(cell_stiff_matrices_data));
dK_data = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, matrix);

%%

%%%%%%%%%%% compute potential contribution
beta = (state'*(dK*GLB_INVP.exp_state)) - ...
    (GLB_INVP.exp_state'*(dK*GLB_INVP.exp_state));
potential = GLB_INVP.theta * beta * ((state'*dK_data) - (GLB_INVP.exp_state'*dK_data))';

%%%%%%%%%%% compute regularization term
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = GLB_INVP.beta .* (GLB_INVP.Ms*dcontrol);
    case{'TV'}
        S_k  = GLB_INVP.S * control;
        St_k = GLB_INVP.S' * control;
        reg = (-0.5 * GLB_INVP.beta * ( ...
            ( (control' * S_k + GLB_INVP.gamma)^(-3/2) ) * ((St_k'*dcontrol)*S_k) ...
            - ((1.0 / sqrt(control' * S_k + GLB_INVP.gamma)) * (GLB_INVP.S*dcontrol)) ));
end

output = potential + reg;

end
