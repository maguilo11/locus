function [PenaltyModel] = modifiedSIMP
PenaltyModel.evaluate=...
    @(material_index,cell_index,controls)evaluate(material_index,cell_index,controls);
PenaltyModel.sensitivity=...
    @(material_index,cell_index,controls)sensitivity(material_index,cell_index,controls);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [value] = evaluate(material_index,cell_index,controls)

global GLB_INVP;

simp_penalty = GLB_INVP.SimpPenalty(material_index);
min_stiffness = GLB_INVP.min_stiffness(material_index);
cell_controls_at_cub_points = GLB_INVP.CellMassMatrices(:,:,cell_index) * ...
    controls(:,cell_index,material_index);
average_material_penalty = ...
    sum(cell_controls_at_cub_points) / GLB_INVP.ElemVolume(cell_index);
value = min_stiffness + ...
    ((1-min_stiffness) * average_material_penalty^simp_penalty);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [value] = sensitivity(material_index,cell_index,controls)

global GLB_INVP;

simp_penalty = GLB_INVP.SimpPenalty(material_index);
power_integer = simp_penalty - 1;
min_stiffness = GLB_INVP.min_stiffness(material_index);
cell_controls_at_cub_points = GLB_INVP.CellMassMatrices(:,:,cell_index) * ...
    controls(:,cell_index,material_index);
average_material_penalty = ...
    sum(cell_controls_at_cub_points) / GLB_INVP.ElemVolume(cell_index);
scale_factor = ...
    simp_penalty * (1-min_stiffness) * (1/GLB_INVP.ElemVolume(cell_index));
value = scale_factor * average_material_penalty^power_integer;

end