function [Operators] = objectiveFunction()
Operators.evaluate=@(state,control)evaluate(state,control);
Operators.gradient=@(state,control)gradient(state,control);
end
% Objective function evaluation
function [output] = evaluate(state,control)
output = 0.0624*sum(control);
end
% objective function gradient
function [output] = gradient(state,control)
output = 0.0624.*ones(size(control));
end