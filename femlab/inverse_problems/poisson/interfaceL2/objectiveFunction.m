function [Operators] = objectiveFunction()
Operators.value=@(state,control)value(state,control);
% First order derivatives
Operators.partialDerivativeState=...
    @(state,control)partialDerivativeState(state,control);
Operators.partialDerivativeControl=...
    @(state,control)partialDerivativeControl(state,control);
% Second order derivatives
Operators.partialDerivativeStateState=...
    @(state,control,dstate)partialDerivativeStateState(state,control,dstate);
Operators.partialDerivativeStateControl=...
    @(state,control,dcontrol)partialDerivativeStateControl(state,control,dcontrol);
Operators.partialDerivativeControlState=...
    @(state,control,dstate)partialDerivativeControlState(state,control,dstate);
Operators.partialDerivativeControlControl=...
    @(state,control,dcontrol)partialDerivativeControlControl(state,control,dcontrol);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = value(state,control)

global GLB_INVP;

%%%% compute data misfit
state_diff = state - GLB_INVP.exp_state;
data_misfit = 0.5 * GLB_INVP.alpha * ( state_diff' * GLB_INVP.M * state_diff );

%%%% compute regularization term
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = 0.5 * GLB_INVP.beta * (control'*(GLB_INVP.Ms*control));
    case{'TV'}
        reg = 0.5*GLB_INVP.beta * ...
            sqrt( (control' * (GLB_INVP.S * control)) + GLB_INVP.gamma );
end

output = data_misfit + reg;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeControl(state,control)

global GLB_INVP;

%%%%%%%%%%% compute regularization term
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = GLB_INVP.beta * (GLB_INVP.Ms*control);
    case{'TV'}
        reg = GLB_INVP.beta * 0.5 * ...
            ( 1.0 / sqrt( control' * (GLB_INVP.S * control) + ...
            GLB_INVP.gamma ) ) * (GLB_INVP.S * control);
end

output = reg;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeState(state,control)

global GLB_INVP;

%%%% compute derivative contribution 
state_diff = state - GLB_INVP.exp_state;
output = GLB_INVP.alpha * GLB_INVP.M*state_diff;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeStateState(state,control,dstate)

global GLB_INVP;

%%%% compute output
output = GLB_INVP.alpha * GLB_INVP.M*dstate;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeStateControl(state,control,dcontrol)

output = zeros(size(state));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeControlState(state,control,dstate)

output = zeros(size(control));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeControlControl(state,control,dcontrol)

global GLB_INVP;

%%%%%%%%%%% compute regularization term
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = GLB_INVP.beta .* (GLB_INVP.Ms*dcontrol);
    case{'TV'}
        S_k  = GLB_INVP.S * control;
        St_k = GLB_INVP.S' * control;
        reg = (-0.5 * GLB_INVP.beta * ( ...
            ( (control' * S_k + GLB_INVP.gamma)^(-3/2) ) * ((St_k'*dcontrol)*S_k) ...
            - ((1.0 / sqrt(control' * S_k + GLB_INVP.gamma)) * (GLB_INVP.S*dcontrol)) ));
end

output = reg;

end
