function [Operators] = objectiveFunction()
Operators.evaluate=@(primal) evaluate(primal);
Operators.firstDerivative=@(primal) firstDerivative(primal);
Operators.secondDerivative=@(primal,dprimal) secondDerivative(primal, dprimal);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = evaluate(primal)
output = 100 * power((primal(2) - primal(1) * primal(1)), 2.) ...
    + power(1 - primal(1), 2.);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivative(primal)
output = zeros(size(primal));
output(1) = -400. * (primal(2) - power(primal(1), 2.)) ...
    * primal(1) + 2. * primal(1) - 2.;
output(2) = 200. * (primal(2) - power(primal(1), 2.));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivative(primal, dprimal)
output = zeros(size(primal));
a = 400 * (primal(2) - power(primal(1), 2.));
b = 800 * power(primal(1), 2.);
c = (2 - a + b) * dprimal(1);
d = 400 * primal(1) * dprimal(2);
output(1) = c - d;
output(2) = (-400. * primal(1) * dprimal(1)) + (200. * dprimal(2));
end