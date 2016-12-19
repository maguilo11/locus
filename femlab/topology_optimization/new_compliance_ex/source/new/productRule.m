function [Rule] = productRule
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

value = 1;
for material_index=1:GLB_INVP.num_materials
   if(material_index ~= this_material_index)
       this_cell_material_penalty = ...
           GLB_INVP.PenaltyModel.evaluate(material_index,cell_index,nodal_controls);
       value = value * (1 - this_cell_material_penalty);
   end
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [cell_stiffness_matrix] = ...
    sensitivity(this_material_index,cell_index,nodal_controls)

global GLB_INVP;

for material_index_k=1:GLB_INVP.num_materials
    product_penalty_two = 1;
    cell_stiffness_matrix = ...
        GLB_INVP.CellStifsnessMat(:,:,cell_index,material_index_k);
    for material_index_l=1:GLB_INVP.num_materials
        if( (material_index_l ~= this_material_index) && (material_index_l ~= material_index_k) )
            cell_material_penalty = ...
                GLB_INVP.PenaltyModel.evaluate(material_index_l,cell_index,nodal_controls);
            product_penalty_two = product_penalty_two * (1 - cell_material_penalty);
        end
    end
    
    if(material_index_k ~= this_material_index)
        material_penalty = ...
            GLB_INVP.PenaltyModel.evaluate(material_index_k,cell_index,nodal_controls);
        scale_factor = -1. * material_penalty * product_penalty_two;
        cell_stiffness_matrix = cell_stiffness_matrix + ...
            (scale_factor .* cell_stiffness_matrix);
    end
end

end