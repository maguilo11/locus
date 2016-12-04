function [Operators] = objectiveFunction()
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = evaluate(state,control)

global GLB_INVP;

%%%% Get shear and bulk modulus from control array
shear_modulus = control(1:GLB_INVP.nVertGrid);
bulk_modulus = control(GLB_INVP.nVertGrid+1:end);

%%%% compute data misfit term
data_misfit = state - GLB_INVP.exp_state;
misfit = 0.5 * GLB_INVP.alpha * (data_misfit' * (GLB_INVP.M * data_misfit ));

%%%% compute regularization term
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = 0.5 * GLB_INVP.beta * (shear_modulus'*(GLB_INVP.Ms*shear_modulus)) + ...
            0.5 * GLB_INVP.beta * (bulk_modulus'*(GLB_INVP.Ms*bulk_modulus));
    case{'TV'}
        reg = 0.5*GLB_INVP.beta * ...
              sqrt( shear_modulus' * (GLB_INVP.Ss * shear_modulus) ...
              + GLB_INVP.gamma ) + ...  
            0.5*GLB_INVP.beta * sqrt( bulk_modulus' * (GLB_INVP.Ss * bulk_modulus) ...
              + GLB_INVP.gamma );
end

output = misfit + reg;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtControl(state,control)

global GLB_INVP;
              
%%%%%%%%%%% get shear and bulk modulus from control array
shear_modulus = control(1:GLB_INVP.nVertGrid);
bulk_modulus = control(GLB_INVP.nVertGrid+1:end);

%%%%%%%%%%% regularization contribution
switch GLB_INVP.reg
    case{'Tikhonov'}
        reg = [ GLB_INVP.beta * (GLB_INVP.Ms*shear_modulus); ...
                GLB_INVP.beta * (GLB_INVP.Ms*bulk_modulus) ];
    case{'TV'}
        reg = [ GLB_INVP.beta * 0.5 * ( 1.0 / sqrt( ...
                 shear_modulus' * (GLB_INVP.Ss * shear_modulus) + ...
                 GLB_INVP.gamma ) ) * (GLB_INVP.Ss * shear_modulus); ...
                GLB_INVP.beta * 0.5 * ( 1.0 / sqrt( ...
                 bulk_modulus' * (GLB_INVP.Ss * bulk_modulus) + ...
                 GLB_INVP.gamma ) ) * (GLB_INVP.Ss * bulk_modulus) ];
end

output = reg;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtState(state,control)

global GLB_INVP;

%%%%%%%%%%% compute data misfit term
data_misfit = state - GLB_INVP.exp_state;
output = GLB_INVP.alpha .* (GLB_INVP.M * data_misfit);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateState(state,control,dstate)

global GLB_INVP;

output = GLB_INVP.alpha .* (GLB_INVP.M * dstate);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtStateControl(state,control,dcontrol)

output = zeros(size(state));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlState(state,control,dstate)

output = zeros(size(control));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivativeWrtControlControl(state,control,dcontrol)

global GLB_INVP;

%%%%%%%%%%% get shear and bulk modulus from control array
shear_modulus = control(1:GLB_INVP.nVertGrid);
bulk_modulus = control(GLB_INVP.nVertGrid+1:end);

%%%%%%%%%%% get shear and bulk modulus perturbations from dcontrol array
delta_shear_modulus = dcontrol(1:GLB_INVP.nVertGrid);
delta_bulk_modulus = dcontrol(GLB_INVP.nVertGrid+1:end);

%%%%%%%%%%% compute regularization term
switch GLB_INVP.reg
    case{'tikhonov'}
        reg = [ GLB_INVP.beta * (GLB_INVP.Ms * delta_shear_modulus); ...
                GLB_INVP.beta * (GLB_INVP.Ms * delta_bulk_modulus) ];
    case{'TV'}
        S_bulk  = GLB_INVP.Ss * bulk_modulus;
        St_bulk = GLB_INVP.Ss' * bulk_modulus;
        S_shear  = GLB_INVP.Ss * shear_modulus;
        St_shear = GLB_INVP.Ss' * shear_modulus;
        reg = [ -0.5 * GLB_INVP.beta * ( ...
                 ( (shear_modulus' * S_shear + GLB_INVP.gamma)^(-3/2) ) * ...
                 ((St_shear'*delta_shear_modulus)*S_shear) ...
                 - ((1.0 / sqrt(shear_modulus' * S_shear + GLB_INVP.gamma)) * ...
                 (GLB_INVP.Ss * delta_shear_modulus)) );
                -0.5 * GLB_INVP.beta * ( ...
                 ( (bulk_modulus' * S_bulk + GLB_INVP.gamma)^(-3/2) ) * ...
                 ((St_bulk'*delta_bulk_modulus)*S_bulk) ...
                 - ((1.0 / sqrt(bulk_modulus' * S_bulk + GLB_INVP.gamma)) * ...
                 (GLB_INVP.Ss * delta_bulk_modulus)) ) ...
              ];
end

output = reg;

end
