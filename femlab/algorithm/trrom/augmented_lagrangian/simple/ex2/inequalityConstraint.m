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

function [value] = value()

value = 0;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [value] = residual(State,Control)

% value = [1 - Control.current(1)^2 - Control.current(2)^2; ...
%     Control.current(1) + Control.current(2)];
value = (Control.current(1)^2 + Control.current(2)^2) - 1;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [derivative] = firstDerivativeWrtState(State,Control)

derivative = zeros(size(State.current));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [jacobian] = firstDerivativeWrtControl(State,Control)

% jacobian = [-2*Control.current(1),-2*Control.current(2); 
%             1,1];
jacobian = [2*Control.current(1), 2*Control.current(2)];


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlControl(State,Control,Vector)

nz = size(Control.current,1);
hessian = [2, 0; ...
           0, 2];
% output = [hessian * Vector(1:nz), zeros(nz,1)];
output = hessian * Vector(1:nz);

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

function [output] = secondDerivativeWrtStateState(State,Control,Vector)

output = zeros(size(State.current));

end