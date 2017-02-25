function [Operators] = inequalityConstraint()
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

%%%%%%%%%%% Apply filter to controls
filtered_control = zeros(size(control));
for material_index=1:GLB_INVP.num_materials
    end_index = GLB_INVP.nVertGrid*material_index;
    begin_index = GLB_INVP.nVertGrid*(material_index-1) + 1;
    filtered_control(begin_index:end_index) = ...
        GLB_INVP.Filter*control(begin_index:end_index);
end

output = 0;
for material_index=1:GLB_INVP.num_materials
    last = material_index * GLB_INVP.nVertGrid;
    first = 1 + ((material_index-1)*GLB_INVP.nVertGrid);
    output = output + ...
        (sum(GLB_INVP.Ms*filtered_control(first:last)) / GLB_INVP.OriginalVolume);
end

end % end evaluate 

%%%%%%%%%%%%%% Compute inequality constraint gradient %%%%%%%%%%%%%%

function [output] = gradient(state,control,index)

global GLB_INVP;

output = zeros(size(control));
one = ones(GLB_INVP.nVertGrid,1);
for material_index=1:GLB_INVP.num_materials
    last = material_index * GLB_INVP.nVertGrid;
    first = 1 + ((material_index-1)*GLB_INVP.nVertGrid);
    mass_matrix_times_one = GLB_INVP.Filter' * (GLB_INVP.Ms * one);
    output(first:last) = mass_matrix_times_one ./ GLB_INVP.OriginalVolume;
end % end gradient

end % end inequality constraint 