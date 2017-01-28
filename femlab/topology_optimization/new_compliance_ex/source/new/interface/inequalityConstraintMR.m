function [Operators] = inequalityConstraintMR
Operators.value=@(index)value(index);
Operators.evaluate=@(state,control,index)evaluate(state,control,index);
Operators.gradient=@(state,control,index)gradient(state,control,index);
end
%%%%%%%%%%%%%% Evaluate inequality constraint residual %%%%%%%%%%%%%% 
function [output] = value(index)

global GLB_INVP;
output = GLB_INVP.VolumeFraction;

end % end value

%%%%%%%%%%%%%% Evaluate inequality constraint %%%%%%%%%%%%%%

function [output] = evaluate(state,control,index)

global GLB_INVP;

material_one = sum(GLB_INVP.Ms*control(1:GLB_INVP.nVertGrid));
material_two = sum(GLB_INVP.Ms*control(1+GLB_INVP.nVertGrid:2*GLB_INVP.nVertGrid));

component_one = material_one*(1-material_two) / GLB_INVP.OriginalVolume;
component_two = material_one*material_two / GLB_INVP.OriginalVolume;

output = 1 - component_one - component_two;

end % end evaluate 

%%%%%%%%%%%%%% Compute inequality constraint gradient %%%%%%%%%%%%%%

function [output] = gradient(state,control,index)

global GLB_INVP;

output = zeros(size(control));
one = ones(GLB_INVP.nVertGrid,1);
mass_matrix_times_one = GLB_INVP.Ms * one;
material_two = sum(GLB_INVP.Ms*control(1+GLB_INVP.nVertGrid:2*GLB_INVP.nVertGrid));
component_one = (mass_matrix_times_one*(1-material_two)) / GLB_INVP.OriginalVolume;
component_two = (mass_matrix_times_one*material_two) / GLB_INVP.OriginalVolume;
output(1:GLB_INVP.nVertGrid) = -component_one - component_two;

end % end inequality constraint 