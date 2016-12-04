function [Operators] = objectiveFunction()
% HIGH FIDELITY OPERATORS 
Operators.evaluate=@(state,control)evaluate(state,control);
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

%%%%%%%% HIGH FIDELITY OPERATORS %%%%%%%%

function [output] = evaluate(State,Control)


output = ( Control.current(1) - 1 )^2 + (2 * ((Control.current(2) - 2 )^2));
% output = 1 - Control.current(1)^2 - Control.current(2)^2;
% output = Control.current(1) + Control.current(2);

end

%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtState(State,Control)

output = zeros(size(State.current));

end

%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtControl(State,Control)

output = [2 * ( Control.current(1) - 1 ); 
          4 * ( Control.current(2) - 2 )];
% output = [-2*Control.current(1);
%           -2*Control.current(2)];

end

%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateState(State,Control,dstate)

output = zeros(size(State.current));

end

%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateControl(State,Control,dcontrol)

output = zeros(size(State.current));

end

%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlState(State,Control,dstate)

output = zeros(size(Control.current));

end

%%%%%%%%%%%%%%%%%%%

function [output] =secondDerivativeWrtControlControl(State,Control,dcontrol)

nz = size(Control.current,1);
dz = dcontrol(1:nz);
hessian = [2, 0; ...
           0, 4];
% hessian = [-2, 0; ...
%            0, -2];
output = hessian * dz;
%output = [0;0];

end
