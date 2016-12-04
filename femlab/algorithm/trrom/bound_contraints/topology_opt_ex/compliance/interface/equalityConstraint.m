function [Operators] = equalityConstraint()
%% HIGH FIDELITY OPERATORS
Operators.solve=@(state,control)solve(state,control);
Operators.applyInverseJacobianWrtState=...
    @(state,control,rhs)applyInverseJacobianWrtState(state,control,rhs);
Operators.applyInverseAdjointJacobianWrtState=...
    @(state,control,rhs)applyInverseAdjointJacobianWrtState(state,control,rhs);
Operators.residual=@(state,control)residual(state,control);
Operators.firstDerivativeWrtState=...
    @(state,control,dstate)firstDerivativeWrtState(state,control,dstate);
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

function [State] = solve(State,Control)

global GLB_INVP;

spaceDim  = GLB_INVP.spaceDim;
nVertGrid = GLB_INVP.nVertGrid;

%%%%%%%%%%% Initialization of lhs (i.e. solution) vector.
state = zeros(spaceDim*nVertGrid,1);

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
StiffMat = ...
    reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);

%%%%%%%%%%% Apply Dirichlet conditions to rhs vector.
if( ~isempty(state) )
    state(unique(GLB_INVP.dirichlet)) = ...
        GLB_INVP.u_dirichlet( unique(GLB_INVP.dirichlet) );
    force = GLB_INVP.force - K * state;
end

%%%%%%%%%%% Compute new state solution
state(GLB_INVP.FreeDof) = ...
    K(GLB_INVP.FreeDof,GLB_INVP.FreeDof) \ force(GLB_INVP.FreeDof);
State.current = state;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = applyInverseJacobianWrtState(state,Control,rhs)
global GLB_INVP;

spaceDim  = GLB_INVP.spaceDim;
nVertGrid = GLB_INVP.nVertGrid;

%%%%%%%%%%% Initialization of lhs (i.e. solution) vector.
output = zeros(spaceDim*nVertGrid,1);

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
StiffMat = ...
    reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);

%%%%%%%%%%% Solve
output(GLB_INVP.FreeDof) = K(GLB_INVP.FreeDof,GLB_INVP.FreeDof) ...
    \ rhs(GLB_INVP.FreeDof);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = applyInverseAdjointJacobianWrtState(State,Control,rhs)

output = -State.current;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = residual(State,Control)

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
StiffMat = ...
    reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);

output = zeros(GLB_INVP.spaceDim*GLB_INVP.nVertGrid,1);
output(GLB_INVP.FreeDof) = ...
    K(GLB_INVP.FreeDof,GLB_INVP.FreeDof)*State.current(GLB_INVP.FreeDof) ...
    - GLB_INVP.force(GLB_INVP.FreeDof);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtState(state,Control,dstate)
global GLB_INVP;

%%%%%%%%%%% Penalized Cell Stiffness Matrices
[penalty] = getMaterialPenalty(Control.current, GLB_INVP.model_t, 'zero');
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat;
for cell=1:GLB_INVP.numCells
    PenalizedStiffMatPerCell(:,:,cell) = ...
        penalty(cell) * GLB_INVP.CellStifsnessMat(:,:,cell);
end

%%%%%%%%%%% build global deviatoric stiffness matrix
StiffMat = ...
    reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);

output = K*dstate;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtControl(State,Control,dcontrol)
global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
control_at_dof = Control.current( GLB_INVP.mesh.t');
dcontrol_at_dof = dcontrol(GLB_INVP.mesh.t');
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = pow * penalty^(pow-1);
    factor = (1/GLB_INVP.ElemVolume(cell)) * ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * dcontrol_at_dof(:,cell));
    PenalizedStiffMatPerCell(:,:,cell) = penalty * factor * ...
        PenalizedStiffMatPerCell(:,:,cell);
end

%%%%%%%%%%% Assemble compliance term gradient
StiffMat = ...
    reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);

output = K * State.current;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = adjointFirstDerivativeWrtState(state,Control,dual)
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
        penalty(cell) * GLB_INVP.CellStifsnessMat(:,:,cell);
end

%%%%%%%%%%% build global deviatoric stiffness matrix
StiffMat = ...
    reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);

output = K*dual;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = adjointFirstDerivativeWrtControl(State,Control,dual)

global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
dual_at_dof = dual(GLB_INVP.mesh.d');
state_at_dof = State.current(GLB_INVP.mesh.d');
control_at_dof = Control.current( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = pow * penalty^(pow-1);
    factor = dual_at_dof(:,cell)' * (penalty .* ...
        GLB_INVP.CellStifsnessMat(:,:,cell)) * state_at_dof(:,cell);
    PenalizedMassMatPerCell(:,:,cell) = factor * ...
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

function [output] = secondDerivativeWrtStateState(State,Control,dual,dstate)
output=zeros(size(State.current));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateControl(state,Control,dual,dcontrol)
global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
control_at_dof = Control.current( GLB_INVP.mesh.t');
dcontrol_at_dof = dcontrol(GLB_INVP.mesh.t');
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = pow * penalty^(pow-1);
    factor = (1/GLB_INVP.ElemVolume(cell)) * ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * dcontrol_at_dof(:,cell));
    PenalizedStiffMatPerCell(:,:,cell) = penalty * factor * ...
        PenalizedStiffMatPerCell(:,:,cell);
end

%%%%%%%%%%% Assemble compliance term gradient
StiffMat = ...
    reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
output = K * dual;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlState(state,Control,dual,dstate)
global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
dual_at_dof = dual(GLB_INVP.mesh.d');
dstate_at_dof = dstate(GLB_INVP.mesh.d');
control_at_dof = Control.current( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = pow * penalty^(pow-1);
    factor = dual_at_dof(:,cell)' * (penalty .* ...
        GLB_INVP.CellStifsnessMat(:,:,cell)) * dstate_at_dof(:,cell);
    PenalizedMassMatPerCell(:,:,cell) = factor * ...
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

function [output] = secondDerivativeWrtControlControl(State,Control,dual,dcontrol)

global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
simpPenalty = GLB_INVP.SimpPenalty;
state_at_dof = State.current(GLB_INVP.mesh.d');
dual_at_dof = -GLB_INVP.theta*state_at_dof;
control_at_dof = Control.current( GLB_INVP.mesh.t');
dcontrol_at_dof = dcontrol( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = simpPenalty * (simpPenalty-1) * penalty^(simpPenalty-2);
    factor_one = dual_at_dof(:,cell)' * (penalty * ...
        (GLB_INVP.CellStifsnessMat(:,:,cell) * state_at_dof(:,cell)));
    factor_two = (1/GLB_INVP.ElemVolume(cell)) * ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * dcontrol_at_dof(:,cell));
    PenalizedMassMatPerCell(:,:,cell) = factor_one * factor_two * ...
        (1/GLB_INVP.ElemVolume(cell)) * PenalizedMassMatPerCell(:,:,cell);
end

%%%%%%%%%%% Assemble compliance term gradient
MassMat = ...
    reshape(PenalizedMassMatPerCell, 1, numel(PenalizedMassMatPerCell));
M = sparse(GLB_INVP.iIdxVertices, GLB_INVP.jIdxVertices, MassMat);
one = ones(GLB_INVP.nVertGrid,1);
output = M * one;

end