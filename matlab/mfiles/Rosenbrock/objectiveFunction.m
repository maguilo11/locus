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
    output(1) = -400. * (primal(2) - power(primal(1), 2.)) ...
        * primal(1) + 2. * primal(1) - 2.;
    output(2) = 200. * (primal(2) - power(primal(1), 2.));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivative(primal, dprimal)
    output(1) = ((2 - 400 * (primal(2) - power(primal(1), 2.)) ...
        + 800 * power(primal(1), 2.)) * dprimal(1)) ...
        - (400 * primal(1) * dprimal(2));
    output(2) = (-400. * primal(1) * dprimal(1)) + (200. * dprimal(2));
end