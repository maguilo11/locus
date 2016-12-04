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

spaceDim      = GLB_INVP.spaceDim;
numFields     = GLB_INVP.numFields;
numCubPoints  = GLB_INVP.numCubPoints;
numCells      = GLB_INVP.numCells;

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

%%%% compute potential energy misfit
potential = 0.5 * GLB_INVP.theta * ...
    0.5*(state'*(K*state) - GLB_INVP.exp_state'*(K*GLB_INVP.exp_state)) * ...
    0.5*(state'*(K*state) - GLB_INVP.exp_state'*(K*GLB_INVP.exp_state));

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

function [output] = firstDerivativeWrtControl(state,control)

global GLB_INVP;

spaceDim      = GLB_INVP.spaceDim;
numFields     = GLB_INVP.numFields;
numCubPoints  = GLB_INVP.numCubPoints;
numCells      = GLB_INVP.numCells;

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

%%%%%%%%%%% evaluate gradient of state at the cubature points
state_at_dof = state( GLB_INVP.mesh.t');
grad_state_at_cub_points = zeros(spaceDim, numCubPoints, numCells);
intrepid_evaluate(grad_state_at_cub_points, state_at_dof, ...
    GLB_INVP.transformed_grad_at_cub_points);

%%%%%%%%%%% combine transformed values with gradient of u 
grad_state_at_cub_points_times_transformed_val_at_cub_points = ...
    zeros(spaceDim, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( ...
    grad_state_at_cub_points_times_transformed_val_at_cub_points, ...
    grad_state_at_cub_points, GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% integrate stiffnes matrix phi_i*grad(u)*grad(phi_j)
cell_stiff_matrices = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_stiff_matrices, ...
    grad_state_at_cub_points_times_transformed_val_at_cub_points, ...
    GLB_INVP.weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global stiffness matrix phi_i*grad(u)*grad(phi_j)
matrices = reshape(cell_stiff_matrices, 1, numel(cell_stiff_matrices));
dK_state = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, matrices);

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

%%%% potential contribution 
alpha = 0.5*(state'*(K*state) - GLB_INVP.exp_state'*(K*GLB_INVP.exp_state));
potential = GLB_INVP.theta * ...
    (alpha*0.5*((state'*dK_state) - (GLB_INVP.exp_state'*dK_data)))';

%%%%%%%%%%% compute regularization term
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = GLB_INVP.beta * (GLB_INVP.Ms*control);
    case{'TV'}
        reg = GLB_INVP.beta * 0.5 * ...
            ( 1.0 / sqrt( control' * (GLB_INVP.S * control) + ...
            GLB_INVP.gamma ) ) * (GLB_INVP.S * control);
end

output = potential + reg;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtState(state,control)

global GLB_INVP;

spaceDim      = GLB_INVP.spaceDim;
numFields     = GLB_INVP.numFields;
numCubPoints  = GLB_INVP.numCubPoints;
numCells      = GLB_INVP.numCells;

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

%%%% compute alpha constant
alpha = 0.5*(state'*(K*state) - GLB_INVP.exp_state'*(K*GLB_INVP.exp_state));

%%%% compute derivative constribution
output = alpha*GLB_INVP.theta*K*state;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateState(state,control,dstate)

global GLB_INVP;

spaceDim      = GLB_INVP.spaceDim;
numFields     = GLB_INVP.numFields;
numCubPoints  = GLB_INVP.numCubPoints;
numCells      = GLB_INVP.numCells;

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

%%%% compute alpha constant
alpha = 0.5*(state'*(K*state) - GLB_INVP.exp_state'*(K*GLB_INVP.exp_state));

%%%% compute output
beta = state'*(K*dstate);
output = GLB_INVP.theta*(alpha*(K*dstate) + (beta*(K*state)));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateControl(state,control,dcontrol)

global GLB_INVP;

spaceDim      = GLB_INVP.spaceDim;
numFields     = GLB_INVP.numFields;
numCubPoints  = GLB_INVP.numCubPoints;
numCells      = GLB_INVP.numCells;

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

%%%% compute alpha and beta constants
beta = 0.5*(state'*(dK*state) - GLB_INVP.exp_state'*(dK*GLB_INVP.exp_state));
alpha = 0.5*(state'*(K*state) - GLB_INVP.exp_state'*(K*GLB_INVP.exp_state));

%%%% compute output 
output = GLB_INVP.theta*((alpha*(dK*state)) + (beta*(K*state)));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlState(state,control,dstate)

global GLB_INVP;

spaceDim      = GLB_INVP.spaceDim;
numFields     = GLB_INVP.numFields;
numCubPoints  = GLB_INVP.numCubPoints;
numCells      = GLB_INVP.numCells;

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

%%%%%%%%%%% evaluate gradient of state at the cubature points
state_at_dof = state( GLB_INVP.mesh.t');
grad_state_at_cub_points = zeros(spaceDim, numCubPoints, numCells);
intrepid_evaluate(grad_state_at_cub_points, state_at_dof, ...
    GLB_INVP.transformed_grad_at_cub_points);

%%%%%%%%%%% combine transformed values with gradient of u 
grad_state_at_cub_points_times_transformed_val_at_cub_points = ...
    zeros(spaceDim, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( ...
    grad_state_at_cub_points_times_transformed_val_at_cub_points, ...
    grad_state_at_cub_points, GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% integrate stiffnes matrix phi_i*grad(u)*grad(phi_j)
cell_stiff_matrices = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_stiff_matrices, ...
    grad_state_at_cub_points_times_transformed_val_at_cub_points, ...
    GLB_INVP.weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global stiffness matrix phi_i*grad(u)*grad(phi_j)
matrices = reshape(cell_stiff_matrices, 1, numel(cell_stiff_matrices));
dK_state = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, matrices);

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

%%%% compute alpha and beta constants
beta = state'*(K*dstate);
alpha = 0.5*(state'*(K*state) - GLB_INVP.exp_state'*(K*GLB_INVP.exp_state));

%%%% compute output
output = GLB_INVP.theta*(alpha*(dstate'*dK_state) + ...
    (beta*0.5*(state'*dK_state - GLB_INVP.exp_state'*dK_data)))';

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlControl(state,control,dcontrol)

global GLB_INVP;

spaceDim      = GLB_INVP.spaceDim;
numFields     = GLB_INVP.numFields;
numCubPoints  = GLB_INVP.numCubPoints;
numCells      = GLB_INVP.numCells;

%%%%%%%%%%% evaluate gradient of state at the cubature points
state_at_dof = state( GLB_INVP.mesh.t');
grad_state_at_cub_points = zeros(spaceDim, numCubPoints, numCells);
intrepid_evaluate(grad_state_at_cub_points, state_at_dof, ...
    GLB_INVP.transformed_grad_at_cub_points);

%%%%%%%%%%% combine transformed values with gradient of u 
grad_state_at_cub_points_times_transformed_val_at_cub_points = ...
    zeros(spaceDim, numCubPoints, numFields, numCells);
intrepid_multiplyVectorDataField( ...
    grad_state_at_cub_points_times_transformed_val_at_cub_points, ...
    grad_state_at_cub_points, GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% integrate stiffnes matrix phi_i*grad(u)*grad(phi_j)
cell_stiff_matrices = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_stiff_matrices, ...
    grad_state_at_cub_points_times_transformed_val_at_cub_points, ...
    GLB_INVP.weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global stiffness matrix phi_i*grad(u)*grad(phi_j)
matrices = reshape(cell_stiff_matrices, 1, numel(cell_stiff_matrices));
dK_state = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, matrices);

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

%%%% compute beta constants
beta = 0.5*(state'*(dK*state) - GLB_INVP.exp_state'*(dK*GLB_INVP.exp_state));

%%%% compute potential contribution
potential = beta*0.5*GLB_INVP.theta*...
    ((state'*dK_state) - (GLB_INVP.exp_state'*dK_data))';

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
