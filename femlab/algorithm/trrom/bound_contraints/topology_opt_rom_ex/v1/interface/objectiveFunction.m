function [Operators] = objectiveFunction()
% HIGH FIDELITY OPERATORS 
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

%%%%%%%% HIGH FIDELITY OPERATORS %%%%%%%%

function [output] = evaluate(State,Control)

global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
penalty = zeros(GLB_INVP.numCells,1);
control_at_dof = Control.current( GLB_INVP.mesh.t');
for cell=1:GLB_INVP.numCells
    penalty(cell) = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell)*control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty(cell) = penalty(cell)^pow;
end

%%%%%%%%%%% Penalized Cell Stiffness Matrices
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat;
for cell=1:GLB_INVP.numCells
    PenalizedStiffMatPerCell(:,:,cell) = ...
        GLB_INVP.MinCellStifsnessMat(:,:,cell) + ...
        (penalty(cell) * (GLB_INVP.CellStifsnessMat(:,:,cell) ...
        - GLB_INVP.MinCellStifsnessMat(:,:,cell)));
end

%%%%%%%%%%% build stiffness matrix
StiffMat = ...
    reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
    
%%%%%%%%%%% Add contribution from each load case
compliance = 0;
numLoadCases = size(GLB_INVP.force,2);
for index=1:numLoadCases
    new = (GLB_INVP.theta(index)/2) * ...
        (State.current(:,index)'*K*State.current(:,index));
    compliance = compliance + new;
end

%%%%%%%%%%% compute volume term
current_volume = sum(GLB_INVP.Ms*Control.current);
misfit = current_volume - GLB_INVP.VolumeFraction * GLB_INVP.OriginalVolume;
volume = 0.5 * GLB_INVP.alpha * misfit * misfit;

%%%%%%%%%%% compute regularization term
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = 0.5 * GLB_INVP.beta * (Control.current'*(GLB_INVP.Ms*Control.current));
    case{'TV'}
        reg = 0.5*GLB_INVP.beta * ...
            sqrt( (Control.current' * (GLB_INVP.Ss * Control.current)) + GLB_INVP.gamma );
end

output = compliance + volume + reg;

end

%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtState(State,Control)
output = 0;
end

%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtControl(State,Control)

global GLB_INVP;

%%%%%%%%%%% Set necessary data for calculation
one = ones(GLB_INVP.nVertGrid,1);
simpPenalty = GLB_INVP.SimpPenalty;
control_at_dof = Control.current( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = zeros(size(GLB_INVP.CellMassMatrices));
%%%%%%%%%%% Add contribution from each load case
numLoadCases = size(GLB_INVP.force,2);
compliance = zeros(size(Control.current,1),1);
for index=1:numLoadCases
    %%%%%%%%%%% Compute Penalty Matrix
    state = State.current(:,index);
    state_at_dof = state(GLB_INVP.mesh.d');
    for cell=1:GLB_INVP.numCells
        penalty = ...
            sum(GLB_INVP.CellMassMatrices(:,:,cell) * ...
            control_at_dof(:,cell)) / GLB_INVP.ElemVolume(cell);
        penalty = simpPenalty * penalty^(simpPenalty-1);
        factor = state_at_dof(:,cell)' * ...
            (penalty .* (GLB_INVP.CellStifsnessMat(:,:,cell) ...
            - GLB_INVP.MinCellStifsnessMat(:,:,cell))) ...
            * state_at_dof(:,cell);
        PenalizedMassMatPerCell(:,:,cell) = ...
            -(GLB_INVP.theta(index)/2) * factor * ...
            (1/GLB_INVP.ElemVolume(cell)) * ...
            GLB_INVP.CellMassMatrices(:,:,cell);
    end
    %%%%%%%%%%% Assemble compliance term gradient
    matrix = ...
        reshape(PenalizedMassMatPerCell, 1, numel(PenalizedMassMatPerCell));
    Matrix = sparse(GLB_INVP.iIdxVertices, GLB_INVP.jIdxVertices, matrix);
    compliance = compliance + (Matrix * one);
end

%%%%%%%%%%% compute volume term
current_volume = sum(GLB_INVP.Ms*Control.current);
misfit = current_volume - GLB_INVP.VolumeFraction * GLB_INVP.OriginalVolume;
mass_matrix_times_one = GLB_INVP.Ms * one;
volume = GLB_INVP.alpha * misfit * mass_matrix_times_one;

%%%%%%%%%%% compute regularization term
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = GLB_INVP.beta * (GLB_INVP.Ms*Control.current);
    case{'TV'}
        reg = GLB_INVP.beta * 0.5 * ...
            ( 1.0 / sqrt( Control.current' * (GLB_INVP.Ss * Control.current) + ...
            GLB_INVP.gamma ) ) * (GLB_INVP.Ss * Control.current);
end

output = compliance + volume + reg;

end

%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateState(state,Control,dstate)
output = 0;
end

%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateControl(State,Control,dcontrol)
output = 0;
end

%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlState(State,Control,dstate)
output = 0;
end

%%%%%%%%%%%%%%%%%%%

function [output] =secondDerivativeWrtControlControl(State,Control,dcontrol)
global GLB_INVP;

%%%%%%%%%%% Set necessary data for computation
one = ones(GLB_INVP.nVertGrid,1);
simpPenalty = GLB_INVP.SimpPenalty;
control_at_dof = Control.current( GLB_INVP.mesh.t');
dcontrol_at_dof = dcontrol( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = zeros(size(GLB_INVP.CellMassMatrices));
%%%%%%%%%%% Add contribution from each load case
numLoadCases = size(GLB_INVP.force,2);
compliance = zeros(size(Control.current,1),1);
for index=1:numLoadCases
    %%%%%%%%%%% Compute Penalty Matrix
    state = State.current(:,index);
    state_at_dof = state(GLB_INVP.mesh.d');
    for cell=1:GLB_INVP.numCells
        penalty = ...
            sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
            / GLB_INVP.ElemVolume(cell);
        penalty = simpPenalty * (simpPenalty-1) * penalty^(simpPenalty-2);
        factor_one = state_at_dof(:,cell)' * ...
            ((penalty * (GLB_INVP.CellStifsnessMat(:,:,cell) ...
            - GLB_INVP.MinCellStifsnessMat(:,:,cell))) ...
            * state_at_dof(:,cell));
        factor_two = (1/GLB_INVP.ElemVolume(cell)) * ...
            sum(GLB_INVP.CellMassMatrices(:,:,cell) * dcontrol_at_dof(:,cell));
        PenalizedMassMatPerCell(:,:,cell) = (GLB_INVP.theta(index)/2) * ...
            factor_one * factor_two * (1/GLB_INVP.ElemVolume(cell)) * ...
            GLB_INVP.CellMassMatrices(:,:,cell);
    end
    %%%%%%%%%%% Assemble compliance term gradient
    MassMat = ...
        reshape(PenalizedMassMatPerCell, 1, numel(PenalizedMassMatPerCell));
    M = sparse(GLB_INVP.iIdxVertices, GLB_INVP.jIdxVertices, MassMat);
    compliance = compliance + M * one;
end

%%%%%%%%%%% compute volume term
mass_matrix_times_one = GLB_INVP.Ms * one;
factor = dcontrol' * mass_matrix_times_one;
volume = GLB_INVP.alpha * factor * mass_matrix_times_one;

switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = GLB_INVP.beta .* (GLB_INVP.Ms*dcontrol);
    case{'TV'}
        S_k  = GLB_INVP.Ss * Control.current;
        St_k = GLB_INVP.Ss' * Control.current;
        reg = (-0.5 * GLB_INVP.beta * ( ...
            ( (Control.current' * S_k + GLB_INVP.gamma)^(-3/2) ) * ((St_k'*dcontrol)*S_k) ...
            - ((1.0 / sqrt(Control.current' * S_k + GLB_INVP.gamma)) * (GLB_INVP.Ss*dcontrol)) ));
end

output = compliance + volume + reg;

end
