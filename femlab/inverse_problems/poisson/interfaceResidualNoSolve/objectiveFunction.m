function [Operators] = objectiveFunction()
Operators.value=@(control)value(control);
Operators.gradient=@(control)gradient(control);
Operators.hessian=@(control,dcontrol)hessian(control,dcontrol);
% Gauss Newton term
Operators.partialDerivativeControlControl=...
    @(state,control,dcontrol)partialDerivativeControlControl(state,control,dcontrol);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = value(control)

global GLB_INVP;

%%%% Solve PDE
EqualityConstraint = equalityConstraint();
[state] = EqualityConstraint.solve(control);

%%%% compute potential energy misfit
IndexedExpState = GLB_INVP.IndexMat*GLB_INVP.exp_state;
residual =  GLB_INVP.NewK*IndexedExpState - GLB_INVP.force;
alpha = state'*residual - IndexedExpState'*residual;
potential = 0.5 * GLB_INVP.theta * alpha * alpha;

%%%% compute regularization term
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = 0.5 * GLB_INVP.beta * (control*(GLB_INVP.Ms*control'));
    case{'TV'}
        reg = 0.5*GLB_INVP.beta * ...
            sqrt( (control * (GLB_INVP.S * control')) + GLB_INVP.gamma );
end

GLB_INVP.state = state;
output = potential + reg;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = gradient(control)

global GLB_INVP;

spaceDim     = GLB_INVP.spaceDim;
numCells     = GLB_INVP.numCells;
numFields    = GLB_INVP.numFields;
numCubPoints = GLB_INVP.numCubPoints;

%%

%%%%%%%%%%% evaluate experimental strain field at the cubature points
IndexedExpState = GLB_INVP.IndexMat*GLB_INVP.exp_state;
data_at_dof = IndexedExpState(GLB_INVP.mesh.t');
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

%%%%%%%%%%% evaluate gradient of u 
state_at_dof = GLB_INVP.state(GLB_INVP.mesh.t');
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
cell_stiff_matrices_state = zeros(numFields, numFields, numCells);
intrepid_integrate(cell_stiff_matrices_state, ...
    grad_state_at_cub_points_times_transformed_val_at_cub_points, ...
    GLB_INVP.weighted_transformed_grad_at_cub_points, 'COMP_BLAS');

%%%%%%%%%%% build global stiffness matrix phi_i*grad(u)*grad(phi_j)
matrix = ...
    reshape(cell_stiff_matrices_state, 1, numel(cell_stiff_matrices_state));
dK_state = sparse(GLB_INVP.iIdx, GLB_INVP.jIdx, matrix);

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
%%%%%%%%%%% potential contribution 
residual = K*IndexedExpState - GLB_INVP.force;
alpha = (GLB_INVP.state'*residual - IndexedExpState'*residual);
residual_misfit = (GLB_INVP.theta*alpha*((GLB_INVP.state'*dK_state) - ...
    (IndexedExpState'*dK_data)))';

%%%%%%%%%%% regularization contribution
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = GLB_INVP.beta * (GLB_INVP.Ms*control');
    case{'TV'}
        reg = GLB_INVP.beta * 0.5 * ...
            ( 1.0 / sqrt( control * (GLB_INVP.S * control') + ...
            GLB_INVP.gamma ) ) * (GLB_INVP.S * control');
end

output = residual_misfit + reg;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = hessian(control,dcontrol)

global GLB_INVP;

spaceDim     = GLB_INVP.spaceDim;
numCells     = GLB_INVP.numCells;
numFields    = GLB_INVP.numFields;
numCubPoints = GLB_INVP.numCubPoints;

%%

%%%%%%%%%%% evaluate strain field at the cubature points
IndexedExpState = GLB_INVP.IndexMat*GLB_INVP.exp_state;
data_at_dof = IndexedExpState(GLB_INVP.mesh.t');
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

%%%%%%%%%%% compute potential contribution
EqualityConstraint = equalityConstraint();
[rhs] = EqualityConstraint.firstDerivativeWrtControl(GLB_INVP.state,control,dcontrol);
[du] = EqualityConstraint.applyInverseJacobianWrtState(GLB_INVP.state,control,rhs);

residual = K*IndexedExpState - GLB_INVP.force;
upsilon = du'*residual;
alpha = (GLB_INVP.state'*residual - IndexedExpState'*residual);
gamma = (GLB_INVP.state'*(dK*IndexedExpState) ...
    - IndexedExpState'*(dK*IndexedExpState));
data_misfit = IndexedExpState - GLB_INVP.state;
dv = -upsilon*data_misfit + gamma*data_misfit + alpha*du;

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

Lzu_times_du = upsilon * ((GLB_INVP.state'*dK_data) - (IndexedExpState'*dK_data))';
Lzz_times_dz = gamma * ((GLB_INVP.state'*dK_data) - (IndexedExpState'*dK_data))';
Lzz_times_dz = Lzz_times_dz + reg;
Lzv_times_dv = EqualityConstraint.adjointFirstDerivativeWrtControl(GLB_INVP.state,control,dv);

output = Lzu_times_du + Lzz_times_dz + Lzv_times_dv';

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

end