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

%%%%%%%%%%% Penalized Cell Stiffness Matrices
power = GLB_INVP.SimpPenalty;
filtered_control = GLB_INVP.Filter * control;
control_at_dof = filtered_control( GLB_INVP.mesh.t');
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat;
for cell=1:GLB_INVP.numCells
    penalty = sum(GLB_INVP.CellMassMatrices(:,:,cell) * ...
        control_at_dof(:,cell)) / GLB_INVP.ElemVolume(cell);
    penalty = penalty^power;
    PenalizedStiffMatPerCell(:,:,cell) = ...
        penalty * GLB_INVP.CellStifsnessMat(:,:,cell);
end

%%%%%%%%%%% build global deviatoric stiffness matrix
StiffMat = reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
compliance = (GLB_INVP.theta/2) * (state'*K*state);

%%%%%%%%%%% compute volume term
one = ones(GLB_INVP.nVertGrid,1);
current_volume = one' * GLB_INVP.Ms * control;
misfit = current_volume - GLB_INVP.VolumeFraction * GLB_INVP.OriginalVolume;
volume = 0.5 * GLB_INVP.alpha * misfit * misfit;

switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = 0.5 * GLB_INVP.beta * (control'*(GLB_INVP.Ms*control));
    case{'TV'}
        reg = 0.5*GLB_INVP.beta * ...
            sqrt( (control' * (GLB_INVP.Ss * control)) + GLB_INVP.gamma );
end

output = compliance + volume + reg;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtControl(state,control)

global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
power = GLB_INVP.SimpPenalty;
state_at_dof = state(GLB_INVP.mesh.d');
filtered_control = GLB_INVP.Filter*control;
control_at_dof = filtered_control( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;
for cell=1:GLB_INVP.numCells
    penalty = sum(GLB_INVP.CellMassMatrices(:,:,cell) * ...
        control_at_dof(:,cell)) / GLB_INVP.ElemVolume(cell);
    penalty = power * penalty^(power-1);
    factor = state_at_dof(:,cell)' * (penalty .* ...
        GLB_INVP.CellStifsnessMat(:,:,cell)) * state_at_dof(:,cell);
    PenalizedMassMatPerCell(:,:,cell) = (GLB_INVP.theta/2) * factor * ...
        (1/GLB_INVP.ElemVolume(cell)) * PenalizedMassMatPerCell(:,:,cell);
end

%%%%%%%%%%% Assemble compliance term gradient
Mass = reshape(PenalizedMassMatPerCell, 1, numel(PenalizedMassMatPerCell));
M = sparse(GLB_INVP.iIdxVertices, GLB_INVP.jIdxVertices, Mass);
one = ones(GLB_INVP.nVertGrid,1);
compliance = GLB_INVP.Filter * (M * one);

%%%%%%%%%%% compute volume term
current_vol = one' * (GLB_INVP.Ms * control);
misfit = current_vol - GLB_INVP.VolumeFraction * GLB_INVP.OriginalVolume;
mass_matrix_times_one = GLB_INVP.Ms * one;
volume = GLB_INVP.alpha * misfit * mass_matrix_times_one;

switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = GLB_INVP.beta * (GLB_INVP.Ms*control);
    case{'TV'}
        reg = GLB_INVP.beta * 0.5 * ...
            ( 1.0 / sqrt( control' * (GLB_INVP.Ss * control) + ...
            GLB_INVP.gamma ) ) * (GLB_INVP.Ss * control);
end

output = compliance + volume + reg;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtState(state,control)

global GLB_INVP;

%%%%%%%%%%% Penalized Cell Stiffness Matrices
power = GLB_INVP.SimpPenalty;
filtered_control = GLB_INVP.Filter*control;
control_at_dof = filtered_control( GLB_INVP.mesh.t');
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat;
for cell=1:GLB_INVP.numCells
    penalty = sum(GLB_INVP.CellMassMatrices(:,:,cell) * ...
        control_at_dof(:,cell)) / GLB_INVP.ElemVolume(cell);
    penalty = penalty^power;
    PenalizedStiffMatPerCell(:,:,cell) = ...
        penalty * GLB_INVP.CellStifsnessMat(:,:,cell);
end

%%%%%%%%%%% build global deviatoric stiffness matrix
StiffMat = reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
output = GLB_INVP.theta .* (K*state);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateState(state,control,dstate)

global GLB_INVP;

%%%%%%%%%%% Penalized Cell Stiffness Matrices
power = GLB_INVP.SimpPenalty;
filtered_control = GLB_INVP.Filter*control;
control_at_dof = filtered_control( GLB_INVP.mesh.t');
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell)*control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = penalty^power;
    PenalizedStiffMatPerCell(:,:,cell) = ...
        penalty * GLB_INVP.CellStifsnessMat(:,:,cell);
end

%%%%%%%%%%% build global deviatoric stiffness matrix
StiffMat = reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
output = GLB_INVP.theta .* (K*dstate);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateControl(state,control,dcontrol)
global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
power = GLB_INVP.SimpPenalty;
filtered_control = GLB_INVP.Filter*control;
control_at_dof = filtered_control( GLB_INVP.mesh.t');
filtered_dcontrol = GLB_INVP.Filter*dcontrol;
dcontrol_at_dof = filtered_dcontrol( GLB_INVP.mesh.t');
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = power * penalty^(power-1);
    factor = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * dcontrol_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    PenalizedStiffMatPerCell(:,:,cell) = GLB_INVP.theta * factor * ...
        penalty * PenalizedStiffMatPerCell(:,:,cell);
end

%%%%%%%%%%% Assemble compliance term gradient
StiffMat = ...
    reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
output = K * state;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlState(state,control,dstate)

global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
power = GLB_INVP.SimpPenalty;
state_at_dof = state(GLB_INVP.mesh.d');
dstate_at_dof = dstate(GLB_INVP.mesh.d');
filtered_control = GLB_INVP.Filter*control;
control_at_dof = filtered_control( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = power * penalty^(power-1);
    factor = state_at_dof(:,cell)' * (penalty .* ...
        GLB_INVP.CellStifsnessMat(:,:,cell)) * dstate_at_dof(:,cell);
    PenalizedMassMatPerCell(:,:,cell) = GLB_INVP.theta * factor * ...
        (1/GLB_INVP.ElemVolume(cell)) * PenalizedMassMatPerCell(:,:,cell);
end

%%%%%%%%%%% Assemble compliance term gradient
MassMat = ...
    reshape(PenalizedMassMatPerCell, 1, numel(PenalizedMassMatPerCell));
M = sparse(GLB_INVP.iIdxVertices, GLB_INVP.jIdxVertices, MassMat);
one = ones(GLB_INVP.nVertGrid,1);
output = GLB_INVP.Filter * (M * one);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlControl(state,control,dcontrol)

global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
power = GLB_INVP.SimpPenalty;
state_at_dof = state(GLB_INVP.mesh.d');
filtered_control = GLB_INVP.Filter*control;
control_at_dof = filtered_control( GLB_INVP.mesh.t');
filtered_dcontrol = GLB_INVP.Filter*dcontrol;
dcontrol_at_dof = filtered_dcontrol( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = power * (power-1) * penalty^(power-2);
    factor_one = state_at_dof(:,cell)' * ((penalty * ...
        GLB_INVP.CellStifsnessMat(:,:,cell)) * state_at_dof(:,cell));
    factor_two = (1/GLB_INVP.ElemVolume(cell)) * ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * dcontrol_at_dof(:,cell));
    PenalizedMassMatPerCell(:,:,cell) = (GLB_INVP.theta/2) * ...
        factor_one * factor_two * (1/GLB_INVP.ElemVolume(cell)) * ... 
        PenalizedMassMatPerCell(:,:,cell);
end

%%%%%%%%%%% Assemble compliance term gradient
MassMat = ...
    reshape(PenalizedMassMatPerCell, 1, numel(PenalizedMassMatPerCell));
M = sparse(GLB_INVP.iIdxVertices, GLB_INVP.jIdxVertices, MassMat);
one = ones(GLB_INVP.nVertGrid,1);
compliance = GLB_INVP.Filter * (M * one);

%%%%%%%%%%% compute volume term
mass_matrix_times_one = GLB_INVP.Filter * (GLB_INVP.Ms * one);
factor = filtered_dcontrol' * mass_matrix_times_one;
volume = GLB_INVP.alpha * factor * mass_matrix_times_one;

switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = GLB_INVP.beta .* (GLB_INVP.Ms*dcontrol);
    case{'TV'}
        S_k  = GLB_INVP.Ss * control;
        St_k = GLB_INVP.Ss' * control;
        reg = (-0.5 * GLB_INVP.beta * ( ...
            ( (control' * S_k + GLB_INVP.gamma)^(-3/2) ) * ((St_k'*dcontrol)*S_k) ...
            - ((1.0 / sqrt(control' * S_k + GLB_INVP.gamma)) * (GLB_INVP.Ss*dcontrol)) ));
end

output = compliance + volume + reg;

end