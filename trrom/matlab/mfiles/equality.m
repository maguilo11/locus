function [Operators] = equality()
Operators.solve=@(control,active_indices,fidelity) solve(control,active_indices,fidelity);
Operators.applyInverseJacobianState=@(state,control,rhs) applyInverseJacobianState(state,control,rhs);
Operators.applyAdjointInverseJacobianState=@(state,control,rhs) applyAdjointInverseJacobianState(state,control,rhs);
Operators.partialDerivativeState=@(state,control,vector) partialDerivativeState(state,control,vector);
Operators.partialDerivativeControl=@(state,control,vector) partialDerivativeControl(state,control,vector);
Operators.adjointPartialDerivativeState=@(state,control,dual) adjointPartialDerivativeState(state,control,dual);
Operators.adjointPartialDerivativeControl=@(state,control,dual) adjointPartialDerivativeControl(state,control,dual);
Operators.adjointPartialDerivativeStateState=@(state,control,dual,vector) adjointPartialDerivativeStateState(state,control,dual,vector);
Operators.adjointPartialDerivativeControlState=@(state,control,dual,vector) adjointPartialDerivativeControlState(state,control,dual,vector);
Operators.adjointPartialDerivativeStateControl=@(state,control,dual,vector) adjointPartialDerivativeStateControl(state,control,dual,vector);
Operators.adjointPartialDerivativeControlControl=@(state,control,dual,vector) adjointPartialDerivativeControlControl(state,control,dual,vector);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output,left_hand_side,right_hand_side] = solve(control,active_indices,fidelity)
if(strcmp(fidelity,'HIGH_FIDELITY'))
    left_hand_side = 1:1:81;
    right_hand_side = 1:1:9;
    output = 23.*ones(size(control));
else
    output = zeros(size(control));
    left_hand_side = 1:1:81;
    left_hand_side = left_hand_side .* active_indices;
    right_hand_side = 2:2:18;
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = applyInverseJacobianState(state,control,rhs)
output = state.*control + rhs;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = applyAdjointInverseJacobianState(state,control,rhs)
output = state.*control + 2.*rhs;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeState(state,control,vector)
output = 4.*(state.*control) + vector;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = partialDerivativeControl(state,control,vector)
output = 2.*(state.*control) + vector;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = adjointPartialDerivativeState(state,control,dual)
output = state + control + dual;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = adjointPartialDerivativeControl(state,control,dual)
output = 2 .* (state + control + dual);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = adjointPartialDerivativeControlState(state,control,dual,vector)
output = (state.*control) + (dual.*vector);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = adjointPartialDerivativeControlControl(state,control,dual,vector)
output = 2.*(state.*control) + 2.*(dual.*vector);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = adjointPartialDerivativeStateState(state,control,dual,vector)
output = (state.*control) + 3.*(dual.*vector);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = adjointPartialDerivativeStateControl(state,control,dual,vector)
output = (state.*control) + 4.*(dual.*vector);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
