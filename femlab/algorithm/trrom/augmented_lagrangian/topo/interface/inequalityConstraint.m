function [Operators] = inequalityConstraint()
% HIGH FIDELITY OPERATORS 
Operators.value=@()value();
Operators.residual=@(state,control)residual(state,control);
Operators.firstDerivativeWrtState=...
    @(state,control)firstDerivativeWrtState(state,control);
Operators.firstDerivativeWrtControl=...
    @(state,control)firstDerivativeWrtControl(state,control);
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

function [volume] = value()

global GLB_INVP;

volume = GLB_INVP.VolumeFraction;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [value] = residual(State,Control)

global GLB_INVP;

value = (sum(GLB_INVP.Ms*Control.current) / GLB_INVP.OriginalVolume) - ...
    GLB_INVP.VolumeFraction;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [derivative] = firstDerivativeWrtState(State,Control)

derivative = zeros(size(State.current));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [value] = firstDerivativeWrtControl(State,Control)

global GLB_INVP;

one = ones(GLB_INVP.nVertGrid,1);
mass_matrix_times_one = (GLB_INVP.Ms * one)';
value = mass_matrix_times_one ./ GLB_INVP.OriginalVolume;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [value] = secondDerivativeWrtControlControl(State,Control,Vector)

value = zeros(size(Control.current));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [value] = secondDerivativeWrtControlState(State,Control,Vector)

value = zeros(size(Control.current));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [value] = secondDerivativeWrtStateControl(State,Control,Vector)

value = zeros(size(State.current));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [value] = secondDerivativeWrtStateState(State,Control,Vector)

value = zeros(size(State.current));

end