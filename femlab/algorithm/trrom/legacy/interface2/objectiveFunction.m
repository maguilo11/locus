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
% LOW FIDELITY OPERATORS 
Operators.lowFidelityEvaluate=...
    @(stateReducedBasisCoeff,stateBasis,control)lowFidelityEvaluate(stateReducedBasisCoeff,stateBasis,control);
Operators.lowFidelityFirstDerivativeWrtControl=...
    @(stateReducedBasisCoeff,stateBasis,control)lowFidelityFirstDerivativeWrtState(stateReducedBasisCoeff,stateBasis,control);
Operators.lowFidelityFirstDerivativeWrtControl=...
    @(stateReducedBasisCoeff,stateBasis,control)lowFidelityFirstDerivativeWrtControl(stateReducedBasisCoeff,stateBasis,control);
Operators.lowFidelitySecondDerivativeWrtStateState=...
    @(stateReducedBasisCoeff,stateBasis,control,dstate)lowFidelitySecondDerivativeWrtStateState(stateReducedBasisCoeff,stateBasis,control,dstate);
Operators.lowFidelitySecondDerivativeWrtStateControl=...
    @(stateReducedBasisCoeff,stateBasis,control,dcontrol)lowFidelitySecondDerivativeWrtStateControl(stateReducedBasisCoeff,stateBasis,control,dcontrol);
Operators.lowFidelitySecondDerivativeWrtControlState=...
    @(stateReducedBasisCoeff,stateBasis,control,dstate)lowFidelitySecondDerivativeWrtControlState(stateReducedBasisCoeff,stateBasis,control,dstate);
Operators.lowFidelitySecondDerivativeWrtControlControl=...
    @(stateReducedBasisCoeff,stateBasis,control,dcontrol)lowFidelitySecondDerivativeWrtControlControl(stateReducedBasisCoeff,stateBasis,control,dcontrol);
end

%%%%%%%% HIGH FIDELITY OPERATORS %%%%%%%%

function [output] = evaluate(state,control)
global GLB_INVP;
%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
penalty = zeros(GLB_INVP.numCells,1);
control_at_dof = control( GLB_INVP.mesh.t');
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
StiffMat = reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
compliance = (GLB_INVP.theta/2) * (state'*K*state);

%%%%%%%%%%% compute volume term
current_volume = sum(GLB_INVP.Ms*control);
misfit = current_volume - GLB_INVP.VolumeFraction * GLB_INVP.OriginalVolume;
volume = 0.5 * GLB_INVP.alpha * misfit * misfit;

%%%%%%%%%%% compute regularization term
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = 0.5 * GLB_INVP.beta * (control'*(GLB_INVP.Ms*control));
    case{'TV'}
        reg = 0.5*GLB_INVP.beta * ...
            sqrt( (control' * (GLB_INVP.Ss * control)) + GLB_INVP.gamma );
end

output = compliance + volume + reg;

end

%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtState(state,control)
global GLB_INVP;

%%%%%%%%%%% Penalized Cell Stiffness Matrices
[penalty] = getMaterialPenalty(control, GLB_INVP.model_t, 'zero');
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat;
for cell=1:GLB_INVP.numCells
    PenalizedStiffMatPerCell(:,:,cell) = ...
        penalty(cell) * GLB_INVP.CellStifsnessMat(:,:,cell);
end

%%%%%%%%%%% build global deviatoric stiffness matrix
StiffMat = reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
output = GLB_INVP.theta .* (K*state);
end

%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtControl(state,control)

global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
simpPenalty = GLB_INVP.SimpPenalty;
state_at_dof = state(GLB_INVP.mesh.d');
control_at_dof = control( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = simpPenalty * penalty^(simpPenalty-1);
    factor = state_at_dof(:,cell)' * ...
        (penalty .* (GLB_INVP.CellStifsnessMat(:,:,cell) ...
        - GLB_INVP.MinCellStifsnessMat(:,:,cell))) ...
        * state_at_dof(:,cell);
    PenalizedMassMatPerCell(:,:,cell) = -(GLB_INVP.theta/2) * factor * ...
        (1/GLB_INVP.ElemVolume(cell)) * PenalizedMassMatPerCell(:,:,cell);
end

%%%%%%%%%%% Assemble compliance term gradient
matrix = ...
    reshape(PenalizedMassMatPerCell, 1, numel(PenalizedMassMatPerCell));
Matrix = sparse(GLB_INVP.iIdxVertices, GLB_INVP.jIdxVertices, matrix);
one = ones(GLB_INVP.nVertGrid,1);
compliance = Matrix * one;

%%%%%%%%%%% compute volume term
current_volume = sum(GLB_INVP.Ms*control);
misfit = current_volume - GLB_INVP.VolumeFraction * GLB_INVP.OriginalVolume;
mass_matrix_times_one = GLB_INVP.Ms * one;
volume = GLB_INVP.alpha * misfit * mass_matrix_times_one;

%%%%%%%%%%% compute regularization term
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

%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateState(state,control,dstate)

global GLB_INVP;

%%%%%%%%%%% Penalized Cell Stiffness Matrices
[penalty] = getMaterialPenalty(control, GLB_INVP.model_t, 'zero');
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat;
for cell=1:GLB_INVP.numCells
    PenalizedStiffMatPerCell(:,:,cell) = ...
        penalty(cell) * GLB_INVP.CellStifsnessMat(:,:,cell);
end

%%%%%%%%%%% build global deviatoric stiffness matrix
StiffMat = reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
output = GLB_INVP.theta .* (K*dstate);

end

%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateControl(state,control,dcontrol)
global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
control_at_dof = control( GLB_INVP.mesh.t');
dcontrol_at_dof = dcontrol( GLB_INVP.mesh.t');
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = pow * penalty^(pow-1);
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

%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlState(state,control,dstate)

global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
state_at_dof = state(GLB_INVP.mesh.d');
dstate_at_dof = dstate(GLB_INVP.mesh.d');
control_at_dof = control( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = pow * penalty^(pow-1);
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
output = M * one;

end

%%%%%%%%%%%%%%%%%%%

function [output] =secondDerivativeWrtControlControl(state,control,dcontrol)
global GLB_INVP;
%%%%%%%%%%% Compute Penalty Parameters
simpPenalty = GLB_INVP.SimpPenalty;
state_at_dof = state(GLB_INVP.mesh.d');
control_at_dof = control( GLB_INVP.mesh.t');
dcontrol_at_dof = dcontrol( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;

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
    PenalizedMassMatPerCell(:,:,cell) = (GLB_INVP.theta/2) * ...
        factor_one * factor_two * (1/GLB_INVP.ElemVolume(cell)) * ... 
        PenalizedMassMatPerCell(:,:,cell);
end

%%%%%%%%%%% Assemble compliance term gradient
MassMat = ...
    reshape(PenalizedMassMatPerCell, 1, numel(PenalizedMassMatPerCell));
M = sparse(GLB_INVP.iIdxVertices, GLB_INVP.jIdxVertices, MassMat);
one = ones(GLB_INVP.nVertGrid,1);
compliance = M * one;

%%%%%%%%%%% compute volume term
mass_matrix_times_one = GLB_INVP.Ms * one;
factor = dcontrol' * mass_matrix_times_one;
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

%%%%%%%%%%%%%%%%%%%% LOW FIDELITY OPERATORS %%%%%%%%%%%%%%%%%%%%


function [output] = lowFidelityEvaluate(stateReducedBasisCoeff,stateBasis,control)
global GLB_INVP;
%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
penalty = zeros(GLB_INVP.numCells,1);
control_at_dof = control( GLB_INVP.mesh.t');
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
StiffMat = reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
Kr = stateBasis' * K * stateBasis;
compliance = (GLB_INVP.theta/2) * (stateReducedBasisCoeff'*Kr*stateReducedBasisCoeff);

%%%%%%%%%%% compute volume term
current_volume = sum(GLB_INVP.Ms*control);
misfit = current_volume - GLB_INVP.VolumeFraction * GLB_INVP.OriginalVolume;
volume = 0.5 * GLB_INVP.alpha * misfit * misfit;

%%%%%%%%%%% compute regularization term
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = 0.5 * GLB_INVP.beta * (control'*(GLB_INVP.Ms*control));
    case{'TV'}
        reg = 0.5*GLB_INVP.beta * ...
            sqrt( (control' * (GLB_INVP.Ss * control)) + GLB_INVP.gamma );
end

output = compliance + volume + reg;

end

%%%%%%%%%%%%%%%%%%%

function [output] = lowFidelityFirstDerivativeWrtState(stateReducedBasisCoeff,stateBasis,control)
global GLB_INVP;

state = stateBasis*stateReducedBasisCoeff;

%%%%%%%%%%% Penalized Cell Stiffness Matrices
[penalty] = getMaterialPenalty(control, GLB_INVP.model_t, 'zero');
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat;
for cell=1:GLB_INVP.numCells
    PenalizedStiffMatPerCell(:,:,cell) = ...
        penalty(cell) * GLB_INVP.CellStifsnessMat(:,:,cell);
end

%%%%%%%%%%% build global deviatoric stiffness matrix
StiffMat = reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
output = GLB_INVP.theta .* (K*state);
end

%%%%%%%%%%%%%%%%%%%

function [output] = lowFidelityFirstDerivativeWrtControl(stateReducedBasisCoeff,stateBasis,control)

global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
simpPenalty = GLB_INVP.SimpPenalty;
state = stateBasis*stateReducedBasisCoeff;
state_at_dof = state(GLB_INVP.mesh.d');
control_at_dof = control( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = simpPenalty * penalty^(simpPenalty-1);
    factor = state_at_dof(:,cell)' * ...
        (penalty .* (GLB_INVP.CellStifsnessMat(:,:,cell) ...
        - GLB_INVP.MinCellStifsnessMat(:,:,cell))) ...
        * state_at_dof(:,cell);
    PenalizedMassMatPerCell(:,:,cell) = -(GLB_INVP.theta/2) * factor * ...
        (1/GLB_INVP.ElemVolume(cell)) * PenalizedMassMatPerCell(:,:,cell);
end

%%%%%%%%%%% Assemble compliance term gradient
matrix = ...
    reshape(PenalizedMassMatPerCell, 1, numel(PenalizedMassMatPerCell));
Matrix = sparse(GLB_INVP.iIdxVertices, GLB_INVP.jIdxVertices, matrix);
one = ones(GLB_INVP.nVertGrid,1);
compliance = Matrix * one;

%%%%%%%%%%% compute volume term
current_volume = sum(GLB_INVP.Ms*control);
misfit = current_volume - GLB_INVP.VolumeFraction * GLB_INVP.OriginalVolume;
mass_matrix_times_one = GLB_INVP.Ms * one;
volume = GLB_INVP.alpha * misfit * mass_matrix_times_one;

%%%%%%%%%%% compute regularization term
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

function [output] = lowFidelitySecondDerivativeWrtStateState(stateReducedBasisCoeff,stateBasis,control,dstateCoeff)

global GLB_INVP;

dstate = stateBasis*dstateCoeff;
%%%%%%%%%%% Penalized Cell Stiffness Matrices
[penalty] = getMaterialPenalty(control, GLB_INVP.model_t, 'zero');
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat;
for cell=1:GLB_INVP.numCells
    PenalizedStiffMatPerCell(:,:,cell) = ...
        penalty(cell) * GLB_INVP.CellStifsnessMat(:,:,cell);
end

%%%%%%%%%%% build global deviatoric stiffness matrix
StiffMat = reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
output = GLB_INVP.theta .* (K*dstate);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = lowFidelitySecondDerivativeWrtStateControl(stateReducedBasisCoeff,stateBasis,control,dcontrol)
global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
state = stateBasis*stateReducedBasisCoeff;
control_at_dof = control( GLB_INVP.mesh.t');
dcontrol_at_dof = dcontrol( GLB_INVP.mesh.t');
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = pow * penalty^(pow-1);
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

function [output] = lowFidelitySecondDerivativeWrtControlState(stateReducedBasisCoeff,stateBasis,control,dstateCoeff)
global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
state = stateBasis*stateReducedBasisCoeff;
state_at_dof = state(GLB_INVP.mesh.d');
dstate = stateBasis*dstateCoeff;
dstate_at_dof = dstate(GLB_INVP.mesh.d');
control_at_dof = control( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = pow * penalty^(pow-1);
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
output = M * one;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = lowFidelitySecondDerivativeWrtControlControl(stateReducedBasisCoeff,stateBasis,control,dcontrol)
global GLB_INVP;
%%%%%%%%%%% Compute Penalty Parameters
simpPenalty = GLB_INVP.SimpPenalty;
state_at_dof = state(GLB_INVP.mesh.d');
control_at_dof = control( GLB_INVP.mesh.t');
dcontrol_at_dof = dcontrol( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;

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
    PenalizedMassMatPerCell(:,:,cell) = (GLB_INVP.theta/2) * ...
        factor_one * factor_two * (1/GLB_INVP.ElemVolume(cell)) * ... 
        PenalizedMassMatPerCell(:,:,cell);
end

%%%%%%%%%%% Assemble compliance term gradient
MassMat = ...
    reshape(PenalizedMassMatPerCell, 1, numel(PenalizedMassMatPerCell));
M = sparse(GLB_INVP.iIdxVertices, GLB_INVP.jIdxVertices, MassMat);
one = ones(GLB_INVP.nVertGrid,1);
compliance = M * one;

%%%%%%%%%%% compute volume term
mass_matrix_times_one = GLB_INVP.Ms * one;
factor = dcontrol' * mass_matrix_times_one;
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
