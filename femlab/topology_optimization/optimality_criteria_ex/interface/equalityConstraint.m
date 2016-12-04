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
[penalty] = getMaterialPenalty(control, GLB_INVP.model_t, 'zero');
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat;
for cell=1:GLB_INVP.numCells
    PenalizedStiffMatPerCell(:,:,cell) = ...
        penalty(cell) * GLB_INVP.CellStifsnessMat(:,:,cell);
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

%%%%%%%%%%% Computation of the solution
state(GLB_INVP.FreeDof) = ...
    K(GLB_INVP.FreeDof,GLB_INVP.FreeDof) \ force(GLB_INVP.FreeDof);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = applyInverseJacobianWrtState(state,control,rhs)
output = zeros(size(state));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = residual(state,control)

global GLB_INVP;

%%%%%%%%%%% Penalized Cell Stiffness Matrices
[penalty] = getMaterialPenalty(control, GLB_INVP.model_t, 'zero');
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat;
for cell=1:GLB_INVP.numCells
    PenalizedStiffMatPerCell(:,:,cell) = ...
        penalty(cell) * GLB_INVP.CellStifsnessMat(:,:,cell);
end

%%%%%%%%%%% build global deviatoric stiffness matrix
StiffMat = ...
    reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);

output = (K*state) - GLB_INVP.force;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtState(state,control,dstate)
output = zeros(size(state));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtControl(state,control,dcontrol)
output = zeros(size(state));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = adjointFirstDerivativeWrtState(state,control,dual)
output = zeros(size(state));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = adjointFirstDerivativeWrtControl(state,control,dual)
output = zeros(size(control));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateState(state,control,dual,dstate)
output=zeros(size(state));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateControl(state,control,dual,dcontrol)
output=zeros(size(state));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlState(state,control,dual,dstate)
output=zeros(size(control));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlControl(state,control,dual,dcontrol)
output=zeros(size(control));
end