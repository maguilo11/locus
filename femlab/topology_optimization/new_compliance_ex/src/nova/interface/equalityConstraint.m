function [Operators] = equalityConstraint()
Operators.solve=@(control)solve(control);
Operators.applyInverseJacobianWrtState=...
    @(state,control,rhs)applyInverseJacobianWrtState(state,control,rhs);
Operators.applyInverseAdjointJacobianWrtState=...
    @(state,control,rhs)applyInverseAdjointJacobianWrtState(state,control,rhs);
Operators.residual=@(state,control)residual(state,control);
Operators.firstDerivativeWrtState=...
    @(state,control,dcontrol)firstDerivativeWrtState(state,control,dcontrol);
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

function [state] = solve(control)

global GLB_INVP;

spaceDim  = GLB_INVP.spaceDim;
nVertGrid = GLB_INVP.nVertGrid;

%%%%%%%%%%% Initialization of lhs (i.e. solution) vector.
state = zeros(spaceDim*nVertGrid,1);

%%%%%%%%%%% Penalized Cell Stiffness Matrices
pow = GLB_INVP.SimpPenalty;
control_at_dof = control( GLB_INVP.mesh.t');
PenalizedStiffMatPerCell = zeros(size(GLB_INVP.CellStifsnessMat));
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = penalty^pow;
    PenalizedStiffMatPerCell(:,:,cell) = GLB_INVP.MinCellStifsnessMat(:,:,cell) + ...
        (penalty * (GLB_INVP.CellStifsnessMat(:,:,cell) - GLB_INVP.MinCellStifsnessMat(:,:,cell)));
end

%%%%%%%%%%% build global deviatoric stiffness matrix
StiffMat = ...
    reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);

%%%%%%%%%%% Apply Dirichlet conditions to rhs vector.
if( ~isempty(state) )
    state(unique(GLB_INVP.dirichlet_dof)) = ...
        GLB_INVP.u_dirichlet( unique(GLB_INVP.dirichlet_dof) );
    force = GLB_INVP.force - K * state;
end

%%%%%%%%%%% Computation of the solution
state(GLB_INVP.FreeDof) = ...
    K(GLB_INVP.FreeDof,GLB_INVP.FreeDof) \ force(GLB_INVP.FreeDof);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = applyInverseJacobianWrtState(state,control,rhs)

global GLB_INVP;
spaceDim  = GLB_INVP.spaceDim;
nVertGrid = GLB_INVP.nVertGrid;

%%%%%%%%%%% Initialization of lhs (i.e. solution) vector.
output = zeros(spaceDim*nVertGrid,1);

%%%%%%%%%%% Penalized Cell Stiffness Matrices
pow = GLB_INVP.SimpPenalty;
control_at_dof = control( GLB_INVP.mesh.t');
PenalizedStiffMatPerCell = zeros(size(GLB_INVP.CellStifsnessMat));
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = penalty^pow;
    PenalizedStiffMatPerCell(:,:,cell) = GLB_INVP.MinCellStifsnessMat(:,:,cell) + ...
        (penalty * (GLB_INVP.CellStifsnessMat(:,:,cell) - GLB_INVP.MinCellStifsnessMat(:,:,cell)));
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

function [dual] = applyInverseAdjointJacobianWrtState(state,control,rhs)

global GLB_INVP;
spaceDim  = GLB_INVP.spaceDim;
nVertGrid = GLB_INVP.nVertGrid;

%%%%%%%%%%% Initialization of lhs (i.e. solution) vector.
dual = zeros(spaceDim*nVertGrid,1);

%%%%%%%%%%% Penalized Cell Stiffness Matrices
pow = GLB_INVP.SimpPenalty;
control_at_dof = control( GLB_INVP.mesh.t');
PenalizedStiffMatPerCell = zeros(size(GLB_INVP.CellStifsnessMat));
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = penalty^pow;
    PenalizedStiffMatPerCell(:,:,cell) = GLB_INVP.MinCellStifsnessMat(:,:,cell) + ...
        (penalty * (GLB_INVP.CellStifsnessMat(:,:,cell) - GLB_INVP.MinCellStifsnessMat(:,:,cell)));
end

%%%%%%%%%%% build global deviatoric stiffness matrix
StiffMat = ...
    reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);

%%%%%%%%%%% Solve
dual(GLB_INVP.FreeDof) = K(GLB_INVP.FreeDof,GLB_INVP.FreeDof) ...
    \ rhs(GLB_INVP.FreeDof);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = residual(state,control)

global GLB_INVP;

%%%%%%%%%%% Penalized Cell Stiffness Matrices
pow = GLB_INVP.SimpPenalty;
control_at_dof = control( GLB_INVP.mesh.t');
PenalizedStiffMatPerCell = zeros(size(GLB_INVP.CellStifsnessMat));
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = penalty^pow;
    PenalizedStiffMatPerCell(:,:,cell) = GLB_INVP.MinCellStifsnessMat(:,:,cell) + ...
        (penalty * (GLB_INVP.CellStifsnessMat(:,:,cell) - GLB_INVP.MinCellStifsnessMat(:,:,cell)));
end

%%%%%%%%%%% build global deviatoric stiffness matrix
StiffMat = ...
    reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);

output = (K*state) - GLB_INVP.force;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtState(state,control,dstate)

global GLB_INVP;

%%%%%%%%%%% Penalized Cell Stiffness Matrices
pow = GLB_INVP.SimpPenalty;
control_at_dof = control( GLB_INVP.mesh.t');
PenalizedStiffMatPerCell = zeros(size(GLB_INVP.CellStifsnessMat));
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = penalty^pow;
    PenalizedStiffMatPerCell(:,:,cell) = GLB_INVP.MinCellStifsnessMat(:,:,cell) + ...
        (penalty * (GLB_INVP.CellStifsnessMat(:,:,cell) - GLB_INVP.MinCellStifsnessMat(:,:,cell)));
end

%%%%%%%%%%% build global deviatoric stiffness matrix
StiffMat = ...
    reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);

output = K*dstate;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtControl(state,control,dcontrol)

global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
control_at_dof = control( GLB_INVP.mesh.t');
dcontrol_at_dof = dcontrol(GLB_INVP.mesh.t');
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat - GLB_INVP.MinCellStifsnessMat;
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
output = K * state;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = adjointFirstDerivativeWrtState(state,control,dual)

global GLB_INVP;

%%%%%%%%%%% Penalized Cell Stiffness Matrices
pow = GLB_INVP.SimpPenalty;
control_at_dof = control( GLB_INVP.mesh.t');
PenalizedStiffMatPerCell = zeros(size(GLB_INVP.CellStifsnessMat));
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell)*control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = penalty^pow;
    PenalizedStiffMatPerCell(:,:,cell) = GLB_INVP.MinCellStifsnessMat(:,:,cell) + ...
        (penalty * (GLB_INVP.CellStifsnessMat(:,:,cell) - GLB_INVP.MinCellStifsnessMat(:,:,cell)));
end

%%%%%%%%%%% build global deviatoric stiffness matrix
StiffMat = ...
    reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);

output = K*dual;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = adjointFirstDerivativeWrtControl(state,control,dual)

global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
dual_at_dof = dual(GLB_INVP.mesh.dof);
state_at_dof = state(GLB_INVP.mesh.dof);
control_at_dof = control( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = pow * penalty^(pow-1);
    factor = dual_at_dof(:,cell)' * ( penalty .* (GLB_INVP.CellStifsnessMat(:,:,cell) ...
        - GLB_INVP.MinCellStifsnessMat(:,:,cell)) ) * state_at_dof(:,cell);
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

function [output] = secondDerivativeWrtStateState(state,control,dual,dstate)
output=zeros(size(state));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateControl(state,control,dual,dcontrol)

global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
control_at_dof = control( GLB_INVP.mesh.t');
dcontrol_at_dof = dcontrol(GLB_INVP.mesh.t');
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat - GLB_INVP.MinCellStifsnessMat;
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

function [output] = secondDerivativeWrtControlState(state,control,dual,dstate)

global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
dual_at_dof = dual(GLB_INVP.mesh.dof);
dstate_at_dof = dstate(GLB_INVP.mesh.dof);
control_at_dof = control( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = pow * penalty^(pow-1);
    factor = dual_at_dof(:,cell)' * ( penalty .* (GLB_INVP.CellStifsnessMat(:,:,cell) ...
        - GLB_INVP.MinCellStifsnessMat(:,:,cell)) ) * dstate_at_dof(:,cell);
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

function [output] = secondDerivativeWrtControlControl(state,control,dual,dcontrol)

global GLB_INVP;

%%%%%%%%%%% Compute Penalty Parameters
pow = GLB_INVP.SimpPenalty;
dual_at_dof = dual(GLB_INVP.mesh.dof);
state_at_dof = state(GLB_INVP.mesh.dof);
control_at_dof = control( GLB_INVP.mesh.t');
dcontrol_at_dof = dcontrol( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = pow * (pow-1) * penalty^(pow-2);
    factor_one = dual_at_dof(:,cell)' * ( penalty * (( GLB_INVP.CellStifsnessMat(:,:,cell) ...
        - GLB_INVP.MinCellStifsnessMat(:,:,cell) ) * state_at_dof(:,cell)) );
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