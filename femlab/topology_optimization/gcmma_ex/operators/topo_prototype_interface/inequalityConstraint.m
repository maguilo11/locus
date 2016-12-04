function [Operators] = inequalityConstraint()
Operators.value=@(index)value(index);
Operators.evaluate=@(state,control,index)evaluate(state,control,index);
Operators.gradient=@(state,control,index)gradient(state,control,index);
end

% Evaluate inequality constraint residual

function [output] = value(index)

global GLB_INVP;
output = GLB_INVP.VolumeFraction;

end

% Evaluate inequality constraint

function [output] = evaluate(state,control,index)

global GLB_INVP;

output = sum(GLB_INVP.Ms*control) / GLB_INVP.OriginalVolume;

end

% Compute inequality constraint gradient

function [output] = gradient(state,control,index)

global GLB_INVP;
one = ones(GLB_INVP.nVertGrid,1);
mass_matrix_times_one = GLB_INVP.Ms * one;
output = mass_matrix_times_one ./ GLB_INVP.OriginalVolume;;

end