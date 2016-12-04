function [Operators] = equalityConstraint()
Operators.solve=@(control)solve(control);
Operators.gradient=@(state,control)gradient(state,control);
end
% Evaluate equality constraint
function [output] = solve(control)
output = zeros(size(control));
end
% Compute equality constraint first derivative w.r.t. control
function [output] = gradient(state,control)
output = zeros(size(control));
end