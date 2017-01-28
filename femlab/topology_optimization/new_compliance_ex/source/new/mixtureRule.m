function [Rule] = mixtureRule
Rule.transform=@(controls)transform(controls);
Rule.evaluate=...
    @(this_material_index,cell_index,controls)evaluate(this_material_index,cell_index,controls);
Rule.sensitivity=...
    @(this_material_index,cell_index,controls)sensitivity(this_material_index,cell_index,controls);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [nodal_controls] = transform(controls)

global GLB_INVP;

num_materials = GLB_INVP.num_materials;
control_matrix = reshape(controls,GLB_INVP.nVertGrid,num_materials);
nodal_controls = zeros(GLB_INVP.numFields,GLB_INVP.numCells,num_materials);
for material_index=1:num_materials
    this_material_control = control_matrix(:,material_index);
    nodal_controls(:,:,material_index) = ...
        this_material_control(GLB_INVP.mesh.t');
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [value] = evaluate(this_material_index,cell_index,nodal_controls)

global GLB_INVP;

if(this_material_index == 1)
    % material one interpolation (mixture) factor
    material_index_two = 2;
    this_cell_material_penalty = ...
        GLB_INVP.PenaltyModel.evaluate(material_index_two,cell_index,nodal_controls);
    value = (1 - this_cell_material_penalty);
else
    % material two interpolation (mixture) factor
    material_index_one = 1;
    value = GLB_INVP.PenaltyModel.evaluate(material_index_one,cell_index,nodal_controls);
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [cell_stiffness_matrix] = ...
    sensitivity(this_material_index,cell_index,nodal_controls)

global GLB_INVP;

material_index_one = 1;
material_index_two = 2;
if(this_material_index == 1)
    % material one interpolation (mixture) factor
    this_cell_material_penalty = ...
        GLB_INVP.PenaltyModel.evaluate(material_index_two,cell_index,nodal_controls);
    cell_stiffness_matrix = ...
        (1 - this_cell_material_penalty) * GLB_INVP.CellStifsnessMat(:,:,cell_index,material_index_one) + ...
        this_cell_material_penalty * GLB_INVP.CellStifsnessMat(:,:,cell_index,material_index_two);
else
    % material two interpolation (mixture) factor
    this_cell_material_penalty = ...
        GLB_INVP.PenaltyModel.evaluate(material_index_one,cell_index,nodal_controls);
    cell_stiffness_matrix = GLB_INVP.CellStifsnessMat(:,:,cell_index,material_index_two) - ...
        GLB_INVP.CellStifsnessMat(:,:,cell_index,material_index_one);
    cell_stiffness_matrix = this_cell_material_penalty .* cell_stiffness_matrix;
end

end