function [Rule] = summationRule
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

value = 1;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [cell_stiffness_matrix] = ...
    sensitivity(this_material_index,cell_index,nodal_controls)

global GLB_INVP;

cell_stiffness_matrix = ...
    GLB_INVP.CellStifsnessMat(:,:,cell_index,this_material_index);

end