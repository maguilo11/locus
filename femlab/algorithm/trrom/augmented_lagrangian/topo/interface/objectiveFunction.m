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
StiffMat = reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
compliance = (GLB_INVP.theta/2) * (State.current'*K*State.current);

%%%%%%%%%%% compute regularization term
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = 0.5 * GLB_INVP.beta * (Control.current'*(GLB_INVP.Ms*Control.current));
    case{'TV'}
        reg = 0.5*GLB_INVP.beta * ...
            sqrt( (Control.current' * (GLB_INVP.Ss * Control.current)) + GLB_INVP.gamma );
end

output = compliance;

end

%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtState(State,Control)

global GLB_INVP;

%%%%%%%%%%% Penalized Cell Stiffness Matrices
[penalty] = getMaterialPenalty(Control.current, GLB_INVP.model_t, 'zero');
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat;
for cell=1:GLB_INVP.numCells
    PenalizedStiffMatPerCell(:,:,cell) = ...
        GLB_INVP.MinCellStifsnessMat(:,:,cell) + ...
        (penalty(cell) * (GLB_INVP.CellStifsnessMat(:,:,cell) ...
        - GLB_INVP.MinCellStifsnessMat(:,:,cell)));
end

%%%%%%%%%%% build global deviatoric stiffness matrix
StiffMat = reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
output = GLB_INVP.theta .* (K*State.current);
end

%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtControl(State,Control)

global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
simpPenalty = GLB_INVP.SimpPenalty;
state_at_dof = State.current(GLB_INVP.mesh.d');
control_at_dof = Control.current( GLB_INVP.mesh.t');
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

%%%%%%%%%%% compute regularization term
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = GLB_INVP.beta * (GLB_INVP.Ms*Control.current);
    case{'TV'}
        reg = GLB_INVP.beta * 0.5 * ...
            ( 1.0 / sqrt( Control.current' * (GLB_INVP.Ss * Control.current) + ...
            GLB_INVP.gamma ) ) * (GLB_INVP.Ss * Control.current);
end

output = compliance;

end

%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateState(state,Control,dstate)

global GLB_INVP;

%%%%%%%%%%% Penalized Cell Stiffness Matrices
[penalty] = getMaterialPenalty(Control.current, GLB_INVP.model_t, 'zero');
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

function [output] = secondDerivativeWrtStateControl(State,Control,dcontrol)
global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
control_at_dof = Control.current( GLB_INVP.mesh.t');
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
output = K * State.current;

end

%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlState(State,Control,dstate)

global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
state_at_dof = State.current(GLB_INVP.mesh.d');
dstate_at_dof = dstate(GLB_INVP.mesh.d');
control_at_dof = Control.current( GLB_INVP.mesh.t');
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

function [output] =secondDerivativeWrtControlControl(State,Control,dcontrol)
global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
simpPenalty = GLB_INVP.SimpPenalty;
state_at_dof = State.current(GLB_INVP.mesh.d');
control_at_dof = Control.current( GLB_INVP.mesh.t');
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

output = compliance + reg;

end
