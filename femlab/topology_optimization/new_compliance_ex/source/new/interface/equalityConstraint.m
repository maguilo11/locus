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

%%%%%%%%%%% Apply filter to controls
filtered_control = zeros(size(control));
for material_index=1:GLB_INVP.num_materials
    end_index = nVertGrid*material_index;
    begin_index = nVertGrid*(material_index-1) + 1;
    filtered_control(begin_index:end_index) = ...
        GLB_INVP.Filter*control(begin_index:end_index);
end

%%%%%%%%%%% Penalize Cell Stiffness Matrices
nodal_controls = GLB_INVP.InterpolationRule.transform(filtered_control);
PenalizedStiffMatPerCell = ...
    zeros(GLB_INVP.numDof, GLB_INVP.numDof, GLB_INVP.numCells);
for material_index=1:GLB_INVP.num_materials
    for cell=1:GLB_INVP.numCells
        material_penalty = ...
            GLB_INVP.PenaltyModel.evaluate(material_index,cell,nodal_controls);
        interpolatio_rule_panalty = ...
            GLB_INVP.InterpolationRule.evaluate(material_index,cell,nodal_controls);
        scale_factor = interpolatio_rule_panalty * material_penalty;
        PenalizedStiffMatPerCell(:,:,cell) = PenalizedStiffMatPerCell(:,:,cell) + ...
            (scale_factor .* GLB_INVP.CellStifsnessMat(:,:,cell,material_index));
    end
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
nodal_controls = GLB_INVP.InterpolationRule.transform(control);
PenalizedStiffMatPerCell = zeros(size(GLB_INVP.CellStifsnessMat));
for material_index=1:GLB_INVP.num_materials
    for cell=1:GLB_INVP.numCells
        material_penalty = ...
            GLB_INVP.PenaltyModel.evaluate(material_index,cell,nodal_controls);
        interpolatio_rule_panalty = ...
            GLB_INVP.InterpolationRule.evaluate(material_index,cell,nodal_controls);
        scale_factor = interpolatio_rule_panalty * material_penalty;
        PenalizedStiffMatPerCell(:,:,cell) = PenalizedStiffMatPerCell(:,:,cell) + ...
            (scale_factor .* GLB_INVP.CellStifsnessMat(:,:,cell,material_index));
    end
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

%%%%%%%%%%% Penalized Cell Stiffness Matrices
nodal_controls = GLB_INVP.InterpolationRule.transform(control);
PenalizedStiffMatPerCell = zeros(size(GLB_INVP.CellStifsnessMat));
for material_index=1:GLB_INVP.num_materials
    for cell=1:GLB_INVP.numCells
        material_penalty = ...
            GLB_INVP.PenaltyModel.evaluate(material_index,cell,nodal_controls);
        interpolatio_rule_panalty = ...
            GLB_INVP.InterpolationRule.evaluate(material_index,cell,nodal_controls);
        scale_factor = interpolatio_rule_panalty * material_penalty;
        PenalizedStiffMatPerCell(:,:,cell) = PenalizedStiffMatPerCell(:,:,cell) + ...
            (scale_factor .* GLB_INVP.CellStifsnessMat(:,:,cell,material_index));
    end
end

%%%%%%%%%%% build global deviatoric stiffness matrix
StiffMat = ...
    reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);

%%%%%%%%%%% Solve
dual = zeros(spaceDim*nVertGrid,1);
dual(GLB_INVP.FreeDof) = K(GLB_INVP.FreeDof,GLB_INVP.FreeDof) ...
    \ rhs(GLB_INVP.FreeDof);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = residual(state,control)

global GLB_INVP;

%%%%%%%%%%% Penalized Cell Stiffness Matrices
nodal_controls = GLB_INVP.InterpolationRule.transform(control);
PenalizedStiffMatPerCell = zeros(size(GLB_INVP.CellStifsnessMat));
for material_index=1:GLB_INVP.num_materials
    for cell=1:GLB_INVP.numCells
        material_penalty = ...
            GLB_INVP.PenaltyModel.evaluate(material_index,cell,nodal_controls);
        interpolatio_rule_panalty = ...
            GLB_INVP.InterpolationRule.evaluate(material_index,cell,nodal_controls);
        scale_factor = interpolatio_rule_panalty * material_penalty;
        PenalizedStiffMatPerCell(:,:,cell) = PenalizedStiffMatPerCell(:,:,cell) + ...
            (scale_factor .* GLB_INVP.CellStifsnessMat(:,:,cell,material_index));
    end
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
nodal_controls = GLB_INVP.InterpolationRule.transform(control);
PenalizedStiffMatPerCell = zeros(size(GLB_INVP.CellStifsnessMat));
for material_index=1:GLB_INVP.num_materials
    for cell=1:GLB_INVP.numCells
        material_penalty = ...
            GLB_INVP.PenaltyModel.evaluate(material_index,cell,nodal_controls);
        interpolatio_rule_panalty = ...
            GLB_INVP.InterpolationRule.evaluate(material_index,cell,nodal_controls);
        scale_factor = interpolatio_rule_panalty * material_penalty;
        PenalizedStiffMatPerCell(:,:,cell) = PenalizedStiffMatPerCell(:,:,cell) + ...
            (scale_factor .* GLB_INVP.CellStifsnessMat(:,:,cell,material_index));
    end
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

%%%%%%%%%%% Compute penalized sensitivities
nodal_controls = GLB_INVP.InterpolationRule.transform(control);
nodal_dcontrols = GLB_INVP.InterpolationRule.transform(dcontrol);
PenalizedStiffMatPerCell = zeros(size(GLB_INVP.CellStifsnessMat));
for material_index=1:GLB_INVP.num_materials
    for cell=1:GLB_INVP.numCells
        material_penalty = ...
            GLB_INVP.PenaltyModel.sensitivity(material_index,cell,nodal_controls);
        scale_factor = sum(GLB_INVP.CellMassMatrices(:,:,cell) * ...
            nodal_dcontrols(:,cell,material_index));
        cell_stiffness_matrix = ...
            GLB_INVP.InterpolationRule.sensitivity(material_index,cell,nodal_controls);
        cell_stiffness_matrix = (material_penalty * scale_factor) .* ...
            cell_stiffness_matrix;
        PenalizedStiffMatPerCell(:,:,cell) = cell_stiffness_matrix + ...
            PenalizedStiffMatPerCell(:,:,cell);
    end
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
nodal_controls = GLB_INVP.InterpolationRule.transform(control);
PenalizedStiffMatPerCell = zeros(size(GLB_INVP.CellStifsnessMat));
for material_index=1:GLB_INVP.num_materials
    for cell=1:GLB_INVP.numCells
        material_penalty = ...
            GLB_INVP.PenaltyModel.evaluate(material_index,cell,nodal_controls);
        interpolatio_rule_panalty = ...
            GLB_INVP.InterpolationRule.evaluate(material_index,cell,nodal_controls);
        scale_factor = interpolatio_rule_panalty * material_penalty;
        PenalizedStiffMatPerCell(:,:,cell) = PenalizedStiffMatPerCell(:,:,cell) + ...
            (scale_factor .* GLB_INVP.CellStifsnessMat(:,:,cell,material_index));
    end
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
nodal_duals = dual(GLB_INVP.mesh.dof);
nodal_states = state(GLB_INVP.mesh.dof);
nodal_controls = control( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;
for cell=1:GLB_INVP.numCells
    material_penalty = ...
        GLB_INVP.PenaltyModel.sensitivity(material_index,cell,nodal_controls);
    scale_factor = nodal_duals(:,cell)' * ...
        ( material_penalty .* GLB_INVP.CellStifsnessMat(:,:,cell) ) ...
        * nodal_states(:,cell);
    PenalizedMassMatPerCell(:,:,cell) = ...
        scale_factor * PenalizedMassMatPerCell(:,:,cell);
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
nodal_controls = control( GLB_INVP.mesh.t');
nodal_dcontrols = dcontrol(GLB_INVP.mesh.t');
PenalizedStiffMatPerCell = GLB_INVP.CellStifsnessMat ;
for cell=1:GLB_INVP.numCells
    penalty = GLB_INVP.PenaltyModel.sensitivity(material_index,cell,nodal_controls);
    scale_factor = sum(GLB_INVP.CellMassMatrices(:,:,cell) * nodal_dcontrols(:,cell));
    PenalizedStiffMatPerCell(:,:,cell) = penalty * scale_factor * ...
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
nodal_duals = dual(GLB_INVP.mesh.dof);
nodal_dstates = dstate(GLB_INVP.mesh.dof);
nodal_controls = control( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;
for cell=1:GLB_INVP.numCells
    material_penalty = ...
        GLB_INVP.PenaltyModel.sensitivity(material_index,cell,nodal_controls);
    scale_factor = nodal_duals(:,cell)' * ...
        ( material_penalty .* GLB_INVP.CellStifsnessMat(:,:,cell) ) ...
        * nodal_dstates(:,cell);
    PenalizedMassMatPerCell(:,:,cell) = ...
        scale_factor * PenalizedMassMatPerCell(:,:,cell);
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
min_stiffness = GLB_INVP.min_stiffness;
nodal_duals = dual(GLB_INVP.mesh.dof);
nodal_states = state(GLB_INVP.mesh.dof);
nodal_controls = control( GLB_INVP.mesh.t');
nodal_dcontrols = dcontrol( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;
for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * nodal_controls(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = pow * (pow-1) * (1 - min_stiffness) * penalty^(pow-2);
    factor_one = nodal_duals(:,cell)' * ...
        ( penalty .* GLB_INVP.CellStifsnessMat(:,:,cell) ) ...
        * nodal_states(:,cell) ;
    factor_two = (1/GLB_INVP.ElemVolume(cell)) * ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * nodal_dcontrols(:,cell));
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