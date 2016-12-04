function [Operators] = inequalityConstraint()
Operators.value=@value;
Operators.evaluate=@(control)evaluate(control);
Operators.firstDerivative=@(control)firstDerivative(control);
Operators.secondDerivative=...
    @(control,dcontrol)secondDerivative(control,dcontrol);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = value()

global GLB_INVP;

output = GLB_INVP.VolumeFraction;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = evaluate(control)

global GLB_INVP;

filter_control = GLB_INVP.Filter*control;
current_volume = sum(GLB_INVP.Ms*filter_control);
output = current_volume / GLB_INVP.OriginalVolume;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivative(control)

global GLB_INVP;

one = ones(GLB_INVP.nVertGrid,1);
output = (GLB_INVP.Filter * (GLB_INVP.Ms*one)) ./ GLB_INVP.OriginalVolume;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivative(control,dcontrol)

output = zeros(size(dcontrol));

end