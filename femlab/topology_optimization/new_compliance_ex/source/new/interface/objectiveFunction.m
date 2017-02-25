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

%%%%%%%%%%% Apply filter to controls
filtered_control = zeros(size(control));
for material_index=1:GLB_INVP.num_materials
    end_index = GLB_INVP.nVertGrid*material_index;
    begin_index = GLB_INVP.nVertGrid*(material_index-1) + 1;
    filtered_control(begin_index:end_index) = ...
        GLB_INVP.Filter*control(begin_index:end_index);
end

compliance = 0;
nodal_controls = GLB_INVP.InterpolationRule.transform(filtered_control);
for material_index=1:GLB_INVP.num_materials
    %%%%%%%%%%% Compute contribution for this material %%%%%%%%%%%
    PenalizedStiffMatPerCell = ...
        GLB_INVP.CellStifsnessMat(:,:,:,material_index);
    for cell=1:GLB_INVP.numCells
        material_penalty = ...
            GLB_INVP.PenaltyModel.evaluate(material_index,cell,nodal_controls);
        interpolatio_rule_panalty = ...
            GLB_INVP.InterpolationRule.evaluate(material_index,cell,nodal_controls);
        scale_factor = interpolatio_rule_panalty * material_penalty;
        PenalizedStiffMatPerCell(:,:,cell) = ...
            (scale_factor .* PenalizedStiffMatPerCell(:,:,cell));
    end
    %%%%%%%%%%% build stiffness matrix %%%%%%%%%%%
    StiffMat = reshape(PenalizedStiffMatPerCell, 1, numel(PenalizedStiffMatPerCell));
    K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
    %%%%%%%%%%% compute this material contribution %%%%%%%%%%%
    compliance = compliance + (GLB_INVP.theta/2) * (state'*K*state);
end

%%%%%%%%%%% compute regularization term
reg = 0;
switch GLB_INVP.reg
    case{'Tikhonov'}
        for material_index=1:GLB_INVP.num_materials
            last = material_index * GLB_INVP.nVertGrid;
            first = 1 + ((material_index-1)*GLB_INVP.nVertGrid);
            reg = reg + 0.5 * GLB_INVP.beta * (control(first:last)' * ...
                (GLB_INVP.Ms*control(first:last)));
        end
    case{'TV'}
        for material_index=1:GLB_INVP.num_materials
            last = material_index * GLB_INVP.nVertGrid;
            first = 1 + ((material_index-1)*GLB_INVP.nVertGrid);
            reg = reg + 0.5*GLB_INVP.beta * sqrt( (control(first:last)' * ...
                (GLB_INVP.Ss * control(first:last))) + GLB_INVP.gamma );
        end
end

output = compliance + reg;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = gradient(state,control)

global GLB_INVP;

%%%%%%%%%%% Apply filter to controls
filtered_control = zeros(size(control));
for material_index=1:GLB_INVP.num_materials
    end_index = GLB_INVP.nVertGrid*material_index;
    begin_index = GLB_INVP.nVertGrid*(material_index-1) + 1;
    filtered_control(begin_index:end_index) = ...
        GLB_INVP.Filter*control(begin_index:end_index);
end

%%%%%%%%%%% Compute penalized sensitivities
one = ones(GLB_INVP.nVertGrid,1);
compliance = zeros(size(control));
nodal_states = state(GLB_INVP.mesh.dof);
nodal_controls = GLB_INVP.InterpolationRule.transform(control);
for material_index=1:GLB_INVP.num_materials
    %%%%%%%%%%% Compute contribution for this material %%%%%%%%%%%
    PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;
    for cell=1:GLB_INVP.numCells
        material_penalty = ...
            GLB_INVP.PenaltyModel.sensitivity(material_index,cell,nodal_controls);
        cell_stiffness_matrix = ...
            GLB_INVP.InterpolationRule.sensitivity(material_index,cell,nodal_controls);
        interpolation_rule_penalty = nodal_states(:,cell)' * ...
            (cell_stiffness_matrix * nodal_states(:,cell));
        scale_factor = -(GLB_INVP.theta/2) * ...
            material_penalty * interpolation_rule_penalty;
        PenalizedMassMatPerCell(:,:,cell) = ...
            scale_factor .* PenalizedMassMatPerCell(:,:,cell);
    end
    %%%%%%%%%%% Compute gradient contribution for this material %%%%%%%%%%%
    this_material_matrix = ...
        reshape(PenalizedMassMatPerCell, 1, numel(PenalizedMassMatPerCell));
    ThisMaterialMatrix = ...
        sparse(GLB_INVP.iIdxVertices, GLB_INVP.jIdxVertices, this_material_matrix);
    last = material_index * GLB_INVP.nVertGrid;
    first = 1 + ((material_index-1)*GLB_INVP.nVertGrid);
    compliance(first:last) = GLB_INVP.Filter' * (ThisMaterialMatrix * one);
end

%%%%%%%%%%% compute regularization term
reg = zeros(size(control));
switch GLB_INVP.reg
    case{'Tikhonov'}
        for material_index=1:GLB_INVP.num_materials
            last = material_index * GLB_INVP.nVertGrid;
            first = 1 + ((material_index-1)*GLB_INVP.nVertGrid);
            reg(first:last) = ...
                GLB_INVP.beta * (GLB_INVP.Ms*control(first:last));
        end
    case{'TV'}
        for material_index=1:GLB_INVP.num_materials
            last = material_index * GLB_INVP.nVertGrid;
            first = 1 + ((material_index-1)*GLB_INVP.nVertGrid);
            reg(first:last) = GLB_INVP.beta * 0.5 * ...
                ( 1.0 / sqrt( control(first:last)' * ...
                (GLB_INVP.Ss * control(first:last)) + ...
                GLB_INVP.gamma ) ) * (GLB_INVP.Ss * control(first:last));
        end
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
min_stiffness = GLB_INVP.min_stiffness;
state_at_dof = state(GLB_INVP.mesh.dof);
control_at_dof = control( GLB_INVP.mesh.t');
dcontrol_at_dof = dcontrol( GLB_INVP.mesh.t');
PenalizedMassMatPerCell = GLB_INVP.CellMassMatrices;

for cell=1:GLB_INVP.numCells
    penalty = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell) * control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
    penalty = pow * (pow-1) * (1 - min_stiffness) * penalty^(pow-2);
    factor_one = state_at_dof(:,cell)' * ...
        (penalty .* GLB_INVP.CellStifsnessMat(:,:,cell)) ...
        * state_at_dof(:,cell);
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