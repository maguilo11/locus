function [Operators] = objectiveFunction()
Operators.evaluate=@(primal) evaluate(primal);
Operators.firstDerivative=@(primal) firstDerivative(primal);
Operators.secondDerivative=@(primal,dprimal) secondDerivative(primal, dprimal);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = evaluate(primal)
output = 0.0624 * sum(primal);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivative(primal)
output = 0.0624 * ones(size(primal));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivative(primal, dprimal)
output = zeros(size(primal));
end