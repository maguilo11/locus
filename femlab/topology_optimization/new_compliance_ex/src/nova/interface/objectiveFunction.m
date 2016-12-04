function [Operators] = objectiveFunction()
Operators.evaluate=@(state,control)evaluate(state,control);
Operators.firstDerivativeWrtState=...
    @(state,control)firstDerivativeWrtState(state,control);
Operators.gradient=@(state,control)gradient(state,control);
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

%%%%%%%%%%% compute regularization term
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = 0.5 * GLB_INVP.beta * (control'*(GLB_INVP.Ms*control));
    case{'TV'}
        reg = 0.5*GLB_INVP.beta * ...
            sqrt( (control' * (GLB_INVP.Ss * control)) + GLB_INVP.gamma );
end

output = compliance + reg;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = gradient(state,control)

global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
state_at_dof = state(GLB_INVP.mesh.dof);
control_at_dof = control( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = pow * penalty^(pow-1);
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
        reg = GLB_INVP.beta * (GLB_INVP.Ms*control);
    case{'TV'}
        reg = GLB_INVP.beta * 0.5 * ...
            ( 1.0 / sqrt( control' * (GLB_INVP.Ss * control) + ...
            GLB_INVP.gamma ) ) * (GLB_INVP.Ss * control);
end

output = compliance + reg;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtState(state,control)
output = zeros(size(state));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateState(state,control,dstate)
output = zeros(size(state));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateControl(state,control,dcontrol)
output = zeros(size(state));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlState(state,control,dstate)
output = zeros(size(control));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlControl(state,control,dcontrol)
global GLB_INVP;
%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
state_at_dof = state(GLB_INVP.mesh.dof);
control_at_dof = control( GLB_INVP.mesh.t');
dcontrol_at_dof = dcontrol( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;

for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = pow * (pow-1) * penalty^(pow-2);
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
        S_k  = GLB_INVP.Ss * control;
        St_k = GLB_INVP.Ss' * control;
        reg = (-0.5 * GLB_INVP.beta * ( ...
            ( (control' * S_k + GLB_INVP.gamma)^(-3/2) ) * ((St_k'*dcontrol)*S_k) ...
            - ((1.0 / sqrt(control' * S_k + GLB_INVP.gamma)) * (GLB_INVP.Ss*dcontrol)) ));
end

output = compliance + reg;

end